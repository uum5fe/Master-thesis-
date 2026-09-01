"""Finding the tones at the top of the band.

Two things stopped this detector somewhere around a kilohertz, and neither
looked like what it was.  Both are cheap to pin, because a synthetic sweep
has a known answer.

A sweep generator dwells for a fixed number of CYCLES, so the dwell is n/f
seconds and shrinks as the frequency rises.  That is what makes the top of
the band a different regime from the bottom, and it is the property both of
these bugs keyed on.
"""

from __future__ import annotations

import numpy as np
import pytest

import eis_local as E


FS = 100_000.0
N_CYC = 20


def sweep(freqs, fs=FS, n_cyc=N_CYC, noise=0.05, gap_s=0.005, seed=0):
    """A stepped sine with a fixed number of cycles per step."""
    rng = np.random.default_rng(seed)
    chunks = [np.zeros(int(0.05 * fs))]
    for f in freqs:
        n = int(round(n_cyc / f * fs))
        t = np.arange(n) / fs
        chunks.append(np.sin(2 * np.pi * f * t))
        chunks.append(np.zeros(int(gap_s * fs)))
    x = np.concatenate(chunks)
    return x + noise * rng.standard_normal(len(x))


def found(steps, f, tol=0.03) -> bool:
    if not steps:
        return False
    got = np.array([s.freq for s in steps])
    return bool(np.any(np.abs(got / f - 1.0) < tol))


# ---------------------------------------------------------------------------
# the whole point
# ---------------------------------------------------------------------------

def test_every_tone_up_to_4_khz_is_found_at_100_khz() -> None:
    """The requirement: a 100 kHz recording must yield tones up to 4 kHz.

    Before the window floor and the spectral seed were fixed this scored
    11/16, and the five it missed were 915, 1170, 1913, 2446 and 4000 Hz --
    i.e. everything above a kilohertz, which is exactly the symptom that
    gets reported as "the high-frequency excitation is too weak to detect".
    """
    freqs = np.geomspace(100.0, 4000.0, 16)
    steps = E.detect_schedule(sweep(freqs), FS, ppd=12, f_lo=50.0,
                              f_hi=45_000.0, min_snr_db=-30.0, verbose=False)
    missed = [f for f in freqs if not found(steps, f)]
    assert not missed, (
        f"missed {len(missed)} of {len(freqs)} tones: "
        + ", ".join(f"{f:.0f} Hz" for f in missed))


def test_the_top_of_the_band_survives_real_noise() -> None:
    """Not a fair-weather result: the tones must survive a noisy reference."""
    freqs = np.geomspace(100.0, 4000.0, 16)
    steps = E.detect_schedule(sweep(freqs, noise=0.5), FS, ppd=12, f_lo=50.0,
                              f_hi=45_000.0, min_snr_db=-30.0, verbose=False)
    hits = sum(found(steps, f) for f in freqs)
    assert hits >= 15, f"only {hits}/16 tones survived at noise 0.5"


# ---------------------------------------------------------------------------
# bug 1: an analysis window wider than the dwell it measures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("f", [1000.0, 2000.0, 4000.0])
def test_the_demodulation_window_never_outlasts_a_short_dwell(f: float) -> None:
    """The window is n_cyc periods, so it scales with the dwell.

    It used to be floored at 40 ms, which is longer than a twenty-cycle
    dwell above 500 Hz.  Past that the window averages the tone together
    with the silence either side of it and the amplitude falls off as
    dwell/window -- about -18 dB at 4 kHz.  Turning the SNR gate down does
    not recover it, because the dilution scales peak and background alike.
    """
    x = sweep(np.array([f]))
    _env, win = E.demod_envelope(x, FS, f)
    dwell_samples = N_CYC / f * FS
    assert win <= dwell_samples, (
        f"at {f:.0f} Hz the window is {win / FS * 1e3:.1f} ms against a "
        f"{dwell_samples / FS * 1e3:.1f} ms dwell")


def test_the_window_still_has_enough_samples_to_average() -> None:
    """Removing the time floor must not leave a window of four samples.

    The floor was protecting something real; it just said it in the wrong
    units.  The guarantee is a sample count, which is what it was for.
    """
    x = sweep(np.array([4000.0]), fs=10_000.0)
    _env, win = E.demod_envelope(x, 10_000.0, 4000.0)
    assert win >= 8


# ---------------------------------------------------------------------------
# bug 2: a Gauss-Newton started outside its own basin
# ---------------------------------------------------------------------------

def test_the_sine_fit_is_seeded_inside_its_basin() -> None:
    """fit4 converges to the NEAREST optimum, not the best one.

    Its basin is roughly the DFT main lobe, +/- 1/(2N) for N cycles, so
    +/- 2.6 % on a twenty-cycle dwell -- while the 12-points-per-decade grid
    is 21 % apart.  On an ISOLATED tone that is survivable, because there is
    nothing else for the iteration to settle on.  In a real sweep there is:
    the dwell finder leaves a slice of the neighbouring steps in the window,
    and started 6 % away fit4 settles on that instead.  It does not fail
    loudly -- it reports a confident frequency, wrong by several percent,
    with an SNR around -14 dB, which reads downstream as a weak tone rather
    than as a fitting failure.  So the sweep, not a lone sine, is the
    setting this has to be tested in.
    """
    freqs = np.geomspace(100.0, 4000.0, 16)
    x = sweep(freqs, noise=0.02)
    true_f, grid_point = 1169.6070952851458, 1241.7

    env, win = E.demod_envelope(x, FS, grid_point)
    k = int(np.argmax(env)) + win // 2
    a, b = E.dwell_window(x, FS, grid_point, k)

    from_grid, _A, _r, snr_grid = E.fit4(x[a:b], FS, grid_point)
    seed = E.dominant_frequency(x[a:b], FS, grid_point / 1.3, grid_point * 1.3)
    from_seed, _A2, _r2, snr_seed = E.fit4(x[a:b], FS, seed)

    assert abs(from_grid / true_f - 1) > 0.03, (
        "this test is only meaningful if the raw grid start really does "
        f"miss; it returned {from_grid:.1f} for {true_f:.1f}")
    assert snr_grid < 0, (
        "the failure mode is a confident-looking fit with a terrible "
        f"residual; got {snr_grid:.1f} dB")
    assert abs(from_seed / true_f - 1) < 0.005, (
        f"spectral seed gave {from_seed:.1f}, want {true_f:.1f}")
    assert snr_seed > 20, f"seeded fit should be clean; got {snr_seed:.1f} dB"


def test_dominant_frequency_ignores_a_neighbouring_step() -> None:
    """The bracket holds a leftover slice of the next dwell; ignore it."""
    fs = 100_000.0
    n = int(0.02 * fs)
    t = np.arange(n) / fs
    wanted, neighbour = 1200.0, 1500.0
    y = np.sin(2 * np.pi * wanted * t)
    y[int(0.85 * n):] += 0.8 * np.sin(2 * np.pi * neighbour * t[int(0.85 * n):])
    assert E.dominant_frequency(y, fs, 900.0, 1600.0) == pytest.approx(
        wanted, rel=0.01)


# ---------------------------------------------------------------------------
# reading only what you need
# ---------------------------------------------------------------------------

def test_a_bounded_channel_read_equals_slicing_the_full_one(tmp_path) -> None:
    """Same numbers, a fraction of the file touched.

    `fam.channel(c)[:n]` materialises the whole column first and then throws
    most of it away. Locally that costs nothing noticeable, which is why it
    survived; over SMB it is the whole recording of every card pulled across
    the wire to answer a question about the first twenty seconds.
    """
    fs, n_ch, n = 10_000.0, 8, 40_000
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n, n_ch)).astype("<f4")
    names = ["UC1"] + [str(i) for i in range(1, n_ch)]
    cp = ",".join(f"7,32,{nm}" for nm in names)
    header = (f"|CF,2,1,1;|CK,1,3,1,1;|CD,2,{1.0 / fs},1;"
              f"|CR,1,{n_ch},1,0,1;|CP,{cp};|CS,1,{data.nbytes},").encode("latin-1")
    path = tmp_path / "card.DAT"
    path.write_bytes(header + data.tobytes())

    fam = E.FamosFile(path)
    full = fam.channel("UC1")
    assert len(full) == n
    for stop in (1, 1000, n // 3, n):
        assert np.array_equal(fam.channel("UC1", 0, stop), full[:stop])
    assert np.array_equal(fam.channel("UC1", 500, 1500), full[500:1500])


def test_a_bounded_read_clamps_instead_of_failing(tmp_path) -> None:
    """A caller asking for twenty seconds of a five-second card gets five."""
    fs, n_ch, n = 10_000.0, 4, 5_000
    data = np.zeros((n, n_ch), dtype="<f4")
    names = ["UC1", "1", "2", "3"]
    cp = ",".join(f"7,32,{nm}" for nm in names)
    header = (f"|CF,2,1,1;|CK,1,3,1,1;|CD,2,{1.0 / fs},1;"
              f"|CR,1,{n_ch},1,0,1;|CP,{cp};|CS,1,{data.nbytes},").encode("latin-1")
    path = tmp_path / "short.DAT"
    path.write_bytes(header + data.tobytes())

    fam = E.FamosFile(path)
    assert len(fam.channel("UC1", 0, 10 * n)) == n
    assert len(fam.channel("UC1", n, n + 100)) == 0
    assert len(fam.channel("UC1", 100, 50)) == 0


# ---------------------------------------------------------------------------
# reading the header at all
# ---------------------------------------------------------------------------

def _write_card(path, n_ch=8, n=2000, fs=10_000.0, name_len=1):
    """A FAMOS card whose header length is controlled by the channel names."""
    names = ["UC1"] + [f"{'c' * name_len}{i}" for i in range(1, n_ch)]
    cp = ",".join(f"7,32,{nm}" for nm in names)
    data = np.zeros((n, n_ch), dtype="<f4")
    header = (f"|CF,2,1,1;|CK,1,3,1,1;|CD,2,{1.0 / fs},1;"
              f"|CR,1,{n_ch},1,0,1;|CP,{cp};|CS,1,{data.nbytes},"
              ).encode("latin-1")
    path.write_bytes(header + data.tobytes())
    return len(header)


def test_a_header_longer_than_eight_kilobytes_is_still_read(tmp_path) -> None:
    """The read grows until it finds |CS.

    It was a flat read(8192). The header carries one entry per channel, so a
    card with many channels pushes |CS past that window -- and the only
    thing said about it was "incomplete FAMOS header", which points at the
    file rather than at the reader and is indistinguishable from the file
    genuinely being some other format.
    """
    path = tmp_path / "many.DAT"
    size = _write_card(path, n_ch=200, name_len=40)
    assert size > 8192, f"this test needs a header past 8 KB; got {size}"

    fam = E.FamosFile(path)
    assert fam.n_ch == 200
    assert fam.fs == pytest.approx(10_000.0)
    assert len(fam.names) == 200


def test_a_short_header_still_reads(tmp_path) -> None:
    """The ordinary case must not regress."""
    path = tmp_path / "few.DAT"
    _write_card(path, n_ch=8)
    fam = E.FamosFile(path)
    assert fam.n_ch == 8 and fam.n_samples == 2000


def test_a_file_that_is_not_famos_says_so_and_says_what_to_run(tmp_path) -> None:
    """Naming the symptom is not a diagnosis.

    The two causes need opposite responses -- a header longer than the reader
    looked at, versus a file that is not FAMOS at all -- and which one it is
    follows directly from which keys were present.
    """
    path = tmp_path / "other.DAT"
    path.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 5000)
    with pytest.raises(ValueError) as excinfo:
        E.FamosFile(path)
    message = str(excinfo.value)
    assert "none of the keys" in message
    assert "identify_file.py" in message, "it must say what to run next"
    assert "89 48 44 46" in message, "and show the bytes it judged on"


def test_a_famos_file_with_an_odd_layout_is_told_apart(tmp_path) -> None:
    """Some keys present means the reader is wrong, not the file."""
    path = tmp_path / "odd.DAT"
    path.write_bytes(b"|CF,2,1,1;|CD,2,0.0001,1;|CR,1,8,1,0,1;"
                     + b"\x00" * 4000)
    with pytest.raises(ValueError) as excinfo:
        E.FamosFile(path)
    message = str(excinfo.value)
    assert "|CD (sample interval)" in message
    assert "|CS (payload start)" in message
    assert "laid out differently" in message
    assert "identify_file.py" not in message, (
        "a file that clearly IS FAMOS should not be sent to the "
        "what-format-is-this tool")


# ---------------------------------------------------------------------------
# evaluating a subset of a folder
# ---------------------------------------------------------------------------

def _famos_v1_card(path, n_ch=6, n=200, fs=10_000.0):
    names = ["UC1"] + [str(i) for i in range(1, n_ch)]
    cp = ",".join(f"7,32,{nm}" for nm in names)
    data = np.zeros((n, n_ch), dtype="<f4")
    path.write_bytes(
        (f"|CF,2,1,1;|CK,1,3,1,1;|CD,2,{1.0 / fs},1;|CR,1,{n_ch},1,0,1;"
         f"|CP,{cp};|CS,1,{data.nbytes},").encode("latin-1") + data.tobytes())


def test_only_the_named_cards_are_discovered(tmp_path) -> None:
    """A folder can hold more than one measurement, or more than one rate.

    Copying gigabytes to build a folder that holds only the subset is the
    alternative, so the subset is named instead.
    """
    import bronze as B
    from config import DEFAULT
    for card in range(1, 6):
        _famos_v1_card(
            tmp_path / f"Leepa_2612025_Current_45A_Test_01_Karte_{card}.DAT")

    cfg = DEFAULT.replace(dat_dir=tmp_path, leepa="2612025", condition="45A")
    assert len(B.discover_files(cfg)) == 5

    picked = B.discover_files(
        cfg.replace(only_cards=frozenset({"Karte_2", "Karte_3"})))
    assert [p.name[-5] for p in picked] == ["2", "3"]


def test_a_card_filter_that_matches_nothing_says_so(tmp_path) -> None:
    """Silently evaluating all five when two were asked for is worse."""
    import bronze as B
    from config import DEFAULT
    for card in (1, 2):
        _famos_v1_card(
            tmp_path / f"Leepa_2612025_Current_45A_Test_01_Karte_{card}.DAT")
    cfg = DEFAULT.replace(dat_dir=tmp_path, leepa="2612025", condition="45A",
                          only_cards=frozenset({"Karte_9"}))
    with pytest.raises(SystemExit) as excinfo:
        B.discover_files(cfg)
    assert "matched none" in str(excinfo.value)


def test_the_result_name_is_separate_from_the_condition() -> None:
    """--condition FINDS the files; --label NAMES what was produced.

    A subset of a five-card 45 A folder is still discovered as "45A" and is
    not the 45 A plate, so the two cannot be the same string.
    """
    from config import DEFAULT
    cfg = DEFAULT.replace(condition="45A")
    assert cfg.result_name == "45A"
    assert cfg.replace(label="45A_g1_50kHz").result_name == "45A_g1_50kHz"
    assert cfg.replace(label="45A_g1_50kHz").condition == "45A", (
        "the label must not change what the file discovery looks for")
