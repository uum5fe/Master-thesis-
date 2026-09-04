"""The Signals tab: does it show the recording, or a story about it?

The point of this tab is that it is drawn from the pipeline's own readers and
estimators, so what appears on screen is what the evaluation did.  These tests
guard that: a synthetic sweep with a known burst, a known tone and a known
channel scan goes in, and the numbers the tab would print come back out.

The fixture is written at the real logger's shape -- 72 segments, 4 cell-voltage
taps and 4 temperatures, scanned 1.1 us apart -- because two of the things under
test only exist at that shape.  The Nyquist-zone discriminant reads the phase
ramp across the scan, so it needs the full 86.9 us span to have any leverage;
and the burst detector works on the composite of all 72 segments, so a
three-channel stand-in would hand it a far cleaner signal than it ever sees.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.views import signals


# ---------------------------------------------------------------------------
# a synthetic R2-D2 sweep, written the way the logger writes one
# ---------------------------------------------------------------------------

FS = 11001.1026            # the logger's row rate
SCAN_STEP_US = 1.1         # one channel to the next, printed in the file
N_SEG = 72
DUR_S = 0.4
BURST = (0.25, 0.75)       # as a fraction of the record


def _write_point(path, freq, dur=DUR_S, fs=FS, seed=0):
    """One point file: a scanned plate, a burst, and silence around it.

    Every channel is sampled at its OWN instant, offset by its slot in the
    scan, so the alias and the skew arise from the data rather than being
    announced in it.
    """
    rng = np.random.default_rng(seed)
    cols = [f"s{s}" for s in range(1, N_SEG + 1)] \
        + ["uc1", "uc2", "uc3", "uc4"] \
        + [f"temp{i}" for i in range(1, 5)]
    shifts = np.arange(len(cols)) * SCAN_STEP_US
    n = int(fs * dur)
    row_t = np.arange(n) / fs
    on = (row_t >= BURST[0] * dur) & (row_t <= BURST[1] * dur)

    # A little real spread: the local impedance is not the same everywhere, so
    # the segment phasors must not be assumed identical for the scan test to
    # mean anything.
    seg_gain = rng.normal(1.0, 0.06, N_SEG)
    seg_phase = np.deg2rad(rng.normal(0.0, 3.0, N_SEG))

    data = np.empty((n, len(cols)))
    for c, name in enumerate(cols):
        tt = row_t + shifts[c] * 1e-6          # this channel's own instants
        if name.startswith("temp"):
            data[:, c] = 60.0 + 3.0 * int(name[-1]) + rng.normal(0, .02, n)
        elif name.startswith("uc"):
            base = 0.0 if name in ("uc1", "uc3") else 0.61
            sign = -0.5 if name in ("uc1", "uc3") else 0.5
            wave = np.cos(2 * np.pi * freq * tt) * on
            data[:, c] = base + sign * 4e-4 * wave + rng.normal(0, 2e-6, n)
        else:
            k = int(name[1:]) - 1
            wave = np.cos(2 * np.pi * freq * tt + seg_phase[k]) * on
            data[:, c] = (2.0 + 8e-3 * seg_gain[k] * wave
                          + rng.normal(0, 2e-4, n))

    lines = ["\t".join(["timestamp"] + cols),
             "\t".join(["timeshifts"] + [f"{v:.6f}" for v in shifts])]
    for i in range(n):
        us = int(round(i * 1e6 / fs))
        lines.append(f"2026.04.20 10:21:{11 + us // 1_000_000:02d},"
                     f"{us % 1_000_000:06d}\t"
                     + "\t".join(f"{v:.6f}" for v in data[i]))
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    folder = tmp_path_factory.mktemp("sweeps") / "sweep"
    folder.mkdir()
    (folder / "metadata.csv").write_text(
        "R2-D2 Local EIS Measurements\nDate: 20.04.2026 10:21\n"
        "Software Version: V2.5\nCoefficients: TestPlate\n"
        "Leepa: FC0000000-00\n")
    # One tone comfortably below Nyquist (5500.55 Hz), one above it: the second
    # is the case the tab exists to make visible.
    _write_point(folder / "p1.csv", 2000.0, seed=1)
    _write_point(folder / "p2.csv", 8015.625, seed=2)
    return folder


@pytest.fixture(scope="module")
def points(sweep):
    from app.data.sources import CsvLoggerSource
    runs = CsvLoggerSource([sweep]).scan()
    assert runs, "the sweep folder was not recognised"
    return runs[0], signals._csv_points(runs[0])


# ---------------------------------------------------------------------------


def test_a_sweep_folder_is_one_selectable_run(points):
    ref, _pts = points
    assert ref.kind == "csvlog"
    # Identified by the cell it measured, not by the folder it landed in.
    assert ref.measurement_id == "FC0000000-00"
    assert len(ref.files) == 2
    assert ref.detail["coefficients"] == "TestPlate"


def test_the_burst_is_found_and_the_silence_is_left_out(points):
    """The window must sit inside the excitation, not merely overlap it.

    A window that leaks into the silence puts full-weight noise into the fit
    and biases sigma; the detector is deliberately conservative and gives back
    a clean interior, so what is asserted here is containment and coverage,
    not recovery of the exact edges.
    """
    _ref, pts = points
    assert all(p["ok"] for p in pts), [p.get("reason") for p in pts]
    for p in pts:
        burst = p["burst"]
        assert burst["found"]
        t0, t1 = burst["t0_s"] / DUR_S, burst["t1_s"] / DUR_S
        assert BURST[0] <= t0 < 0.45, f"{p['name']}: window starts in silence"
        assert 0.55 < t1 <= BURST[1], f"{p['name']}: window ends in silence"
        assert burst["fraction"] > 0.25          # and it kept most of the burst
        assert burst["peak_over_floor"] > 10


def test_a_point_above_nyquist_is_unfolded_and_flagged(points):
    _ref, pts = points
    pts = {p["name"]: p for p in pts}

    below = pts["p1"]
    assert not below["undersampled"]
    assert below["f_true"] == pytest.approx(2000.0, rel=2e-3)

    above = pts["p2"]
    assert above["undersampled"]
    assert above["f_true"] == pytest.approx(8015.625, rel=2e-3)
    # and the recorded tone really is the fold, not the truth
    assert above["f_alias"] == pytest.approx(FS - 8015.625, rel=5e-3)
    # the segment phasors only line up under the unfolded hypothesis
    assert above["R"] > above["R_base"] + 0.15


def test_the_figures_draw_and_say_what_they_show(points):
    _ref, pts = points
    data = {"source": "csvlog", "points": pts}

    overview = signals._csv_overview(data, 1)
    assert overview.data, "the overview drew nothing"
    assert "above Nyquist" in overview.layout.title.text

    dwell, sync, table = signals._csv_dwell(data, 1, "9", 6)
    assert len(dwell.data) == 2                     # samples + fitted sine
    assert "burst" in dwell.layout.title.text
    # three traces: reference, segment as recorded, segment de-skewed
    assert len(sync.data) == 3
    assert "µs" in sync.layout.title.text and "°" in sync.layout.title.text
    assert table is not None


def test_the_scan_offset_is_reported_at_the_analogue_frequency(points):
    """The delay is printed in the file; the phase it causes is not.

    A scan offset of tau costs 2*pi*f*tau of phase at the frequency the signal
    ACTUALLY had.  Quoting it at the folded frequency understates it by the
    ratio of the two, which for the delivered data is more than a factor of
    ten -- so the title has to carry the analogue number.
    """
    _ref, pts = points
    data = {"source": "csvlog", "points": pts}
    _d, sync, _t = signals._csv_dwell(data, 1, "1", 6)
    title = sync.layout.title.text
    above = [p for p in pts if p["name"] == "p2"][0]
    assert f"{above['f_true']:.4g}" in title
    assert f"{above['f_alias']:.4g}" in title


# ---------------------------------------------------------------------------
# the FAMOS side: five free-running cards, and what may be compared with what
# ---------------------------------------------------------------------------

DWELLS = (500.0, 1000.0, 2000.0)


@pytest.fixture(scope="module")
def cards(tmp_path_factory):
    """Two cards of a stepped sweep, with a bulk offset between them.

    Card 2 is written a full 3 ms late, which is what a free-running card
    actually looks like.  That offset is common to everything on card 2, so it
    cancels in a ratio taken WITHIN card 2 and does not cancel in a ratio taken
    across cards -- which is the whole reason the tab has to pick a reference.
    """
    from datetime import datetime
    from tests.synthetic import write_famos

    fs, seg_s, gap_s = 10_000.0, 0.5, 0.1
    rng = np.random.default_rng(3)
    n = int(fs * len(DWELLS) * (seg_s + gap_s))
    t = np.arange(n) / fs

    def sweep(delay_s=0.0, amp=1.0):
        y = np.zeros(n)
        for k, f in enumerate(DWELLS):
            a = int(fs * k * (seg_s + gap_s))
            b = a + int(fs * seg_s)
            y[a:b] = amp * np.sin(2 * np.pi * f * (t[a:b] - delay_s))
        return y

    folder = tmp_path_factory.mktemp("famos")
    files = []
    for card, (names, lag) in {1: (["1", "2"], 0.0),
                               2: (["17", "18"], 3e-3)}.items():
        chans = {}
        for s in names:
            chans[s] = 2.0 + 8e-3 * sweep(lag) + rng.normal(0, 2e-6, n)
        chans[f"UC{card}"] = 0.65 + 4e-3 * sweep(lag) + rng.normal(0, 2e-7, n)
        chans[f"Temp_{card}"] = 0.62 + rng.normal(0, 1e-5, n)
        p = folder / f"Leepa_SYNTH_Current_150A_Test_01_Karte_{card}.DAT"
        write_famos(p, chans, 1.0 / fs, datetime(2026, 4, 20, 10, 21))
        files.append(str(p))
    return files


def test_each_segment_is_traced_back_to_the_card_that_carries_it(cards):
    index = signals._segment_index(cards)
    assert set(index) == {"1", "2", "17", "18"}
    assert index["1"] == index["2"] == cards[0]
    assert index["17"] == index["18"] == cards[1]


def test_a_segment_is_referenced_to_the_cell_voltage_on_its_own_card(cards):
    """Pairing across cards would put the inter-card offset into the ratio.

    Card 2 here is 3 ms late.  At the top dwell that is six full periods of
    phase -- so a segment on card 2 compared against UC1 would produce a
    confident, entirely fictitious impedance angle.
    """
    fam, ref = signals._card_for_segment(cards, "17")
    assert fam is not None
    assert ref == "UC2", "segment 17 was referenced to another card's copy"
    assert "17" in fam.names

    fam1, ref1 = signals._card_for_segment(cards, "1")
    assert ref1 == "UC1"
    assert "17" not in fam1.names


def test_the_dwells_are_found_where_they_were_written(cards):
    """Only the dwells the evaluation kept, and each inside its own step.

    The hard end of a dwell splatters, and the detector raises a handful of
    weak candidates out of that splatter.  Bronze discards them, so the tab
    must discard them too -- a dropdown entry no impedance came from is worse
    than no entry at all.
    """
    fam, ref = signals._card_for_segment(cards, "1")
    steps = signals._schedule(fam, ref)

    assert len(steps) == len(DWELLS), \
        f"expected {len(DWELLS)} dwells, got {[round(s['freq'], 1) for s in steps]}"
    # offered in the order they were played, so "step 3" means the third one
    assert [s["start"] for s in steps] == sorted(s["start"] for s in steps)

    seg_s, gap_s, fs = 0.5, 0.1, fam.fs
    for k, (s, want) in enumerate(zip(steps, DWELLS)):
        assert abs(s["freq"] - want) / want < 0.02
        assert s["snr_db"] > 20
        # and the window sits inside the dwell that was written, not across it
        a, b = k * (seg_s + gap_s), k * (seg_s + gap_s) + seg_s
        assert a <= s["start"] / fs < s["stop"] / fs <= b


# ---------------------------------------------------------------------------
# finding the sweeps in the first place
# ---------------------------------------------------------------------------


def test_the_check_reports_each_sweep_it_found(sweep, monkeypatch):
    """`--check` has to answer "why is the Signals tab empty?".

    An unset EIS_CSV_ROOT and a wrongly-shaped folder both present as no data,
    and the app cannot tell them apart at the point the user notices.  The
    report names the cell and the point count per sweep so that "nothing is
    showing" becomes a fact about the configuration.
    """
    from app.settings import Settings
    from app.diagnose import report

    monkeypatch.setenv("EIS_SKIP_DOTENV", "1")
    settings = Settings(csv_roots=[str(sweep.parent)])
    text = report(settings)

    body = text[text.index("RAW R2-D2 CSV SWEEPS"):]
    assert "1 sweep folder(s) found" in body
    assert "FC0000000-00" in body        # the cell, read from metadata.csv
    assert "2 frequency point(s)" in body


def test_the_check_says_so_when_the_root_is_unset(monkeypatch):
    from app.settings import Settings
    from app.diagnose import report

    monkeypatch.setenv("EIS_SKIP_DOTENV", "1")
    text = report(Settings(csv_roots=[], famos_roots=[], results_roots=[]))
    body = text[text.index("RAW R2-D2 CSV SWEEPS"):]
    assert "EIS_CSV_ROOT" in body
    assert "not set" in body
