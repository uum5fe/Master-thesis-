"""Cross-card alignment: the thing that cost 43 of 68 segments on the 45 A set.

The Dewetron cards are armed separately, so the excitation schedule -- stored
as sample indices -- means a different instant on each card.  Getting the
offset wrong does not look like an alignment problem downstream: the dwell
window lands on the wrong tone, the sine fit finds nothing there, and silver
reports it as an SNR failure on every segment of the affected card.
"""

from __future__ import annotations

import numpy as np
import pytest

import bronze as B
from config import DEFAULT


FS = 10_000.0


def sweep(n: int, seed: int = 0, noise: float = 0.0) -> np.ndarray:
    """A stepped-sine record like the reference channel carries."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    t = np.arange(n) / FS
    dwell = int(0.25 * FS)
    start = int(0.5 * FS)
    for freq in np.geomspace(1.0, 250.0, 24):
        stop = min(n, start + dwell)
        if start >= n:
            break
        seg = slice(start, stop)
        x[seg] = np.sin(2 * np.pi * freq * t[seg])
        start = stop + int(0.1 * FS)
    if noise:
        x = x + noise * rng.standard_normal(n)
    return x - x.mean()


def _hf_sweep(n: int, fs: float) -> np.ndarray:
    """A stepped sweep reaching into the kilohertz, at any sample rate."""
    x = np.zeros(n)
    at = int(0.05 * fs)
    for f in np.geomspace(20.0, 3000.0, 18):
        dwell = int(min(0.08 * fs, max(20 / f * fs, 0.01 * fs)))
        stop = min(n, at + dwell)
        if at >= n:
            break
        t = np.arange(stop - at) / fs
        x[at:stop] = np.sin(2 * np.pi * f * t)
        at = stop + int(0.005 * fs)
    return x - x.mean()


# ---------------------------------------------------------------------------
# the peak that the old height gate threw away
# ---------------------------------------------------------------------------

def test_a_correct_lag_is_found_even_when_the_peak_is_low() -> None:
    """The 45 A case: |r| ~ 0.27 for a lag that is exactly right.

    Card 3's true offset is 5.712 s, recorded in bronze.py's own docstring
    from an earlier measured run. The field data found it and the old gate
    threw it away for scoring 0.276 against a threshold of 0.30.
    """
    from config import DEFAULT

    n = int(60 * FS)
    clean = sweep(n)
    shift = int(5.7121 * FS)
    a = clean + 0.35 * np.random.default_rng(1).standard_normal(n)
    b = np.roll(clean, shift) + 0.35 * np.random.default_rng(2).standard_normal(n)

    lag, corr, prom = B._best_lag(b, a, max_lag=int(12 * FS))

    assert lag == pytest.approx(shift, abs=2), "the lag itself must be right"
    assert 0.2 < abs(corr) < 0.30, (
        "this test is only meaningful at the field data's own correlation, "
        f"below the old 0.30 threshold; got {corr:.3f}")
    assert prom > DEFAULT.align_min_prominence


def test_noise_alone_produces_no_prominent_peak() -> None:
    """The gate has to still refuse when there is nothing to find.

    Thirty seeds rather than one: a threshold justified by a single draw of
    the null is not justified at all, and the point of this number is that
    it sits clear of the WHOLE null distribution, not of one sample from it.
    """
    from config import DEFAULT

    n = int(60 * FS)
    worst = 0.0
    for seed in range(30):
        rng = np.random.default_rng(1000 + seed)
        _lag, _corr, prom = B._best_lag(
            rng.standard_normal(n), rng.standard_normal(n),
            max_lag=int(12 * FS))
        worst = max(worst, prom)

    assert worst < DEFAULT.align_min_prominence, (
        f"unrelated records reached prominence {worst:.1f}, at or above the "
        f"gate of {DEFAULT.align_min_prominence}")
    assert worst * 2 < DEFAULT.align_min_prominence, (
        f"the gate should clear the null distribution with room to spare; "
        f"null max {worst:.1f} vs gate {DEFAULT.align_min_prominence}")


def test_prominence_is_scale_free() -> None:
    """Scaling a channel changes |r| not at all and prominence not at all.

    Worth pinning: the whole argument for prominence is that it measures the
    peak against its own background rather than against an absolute number.
    """
    n = int(30 * FS)
    clean = sweep(n)
    a = clean + 1.5 * np.random.default_rng(3).standard_normal(n)
    b = np.roll(clean, 1234) + 1.5 * np.random.default_rng(4).standard_normal(n)

    lag1, corr1, prom1 = B._best_lag(b, a, max_lag=int(5 * FS))
    lag2, corr2, prom2 = B._best_lag(1000.0 * b, a, max_lag=int(5 * FS))

    assert lag1 == lag2
    assert corr1 == pytest.approx(corr2, rel=1e-9)
    assert prom1 == pytest.approx(prom2, rel=1e-9)


# ---------------------------------------------------------------------------
# choosing the anchor
# ---------------------------------------------------------------------------

def test_the_anchor_is_not_just_the_first_card() -> None:
    """Card 1's reference was the degraded one, so it made a bad anchor.

    Anchoring on it scored every other card weakly and the height gate then
    refused all of them. The anchor has to be the card the others agree with
    best, not the one that sorts first.
    """
    n = int(40 * FS)
    clean = sweep(n)
    rng = np.random.default_rng(11)

    # card 1: mostly dead reference -- a little signal buried in a lot of noise
    traces = {"card1": 0.05 * clean + 3.0 * rng.standard_normal(n)}
    # cards 2-4: healthy, mutually consistent, each at its own offset
    for i, shift in enumerate((0, 25_359, 25_492), start=2):
        traces[f"card{i}"] = (np.roll(clean, shift)
                              + 0.3 * rng.standard_normal(n))

    stems = list(traces)
    anchor = B._pick_anchor(stems, traces, max_lag=int(12 * FS), log=None)
    assert anchor != "card1", "the degraded card must not become the anchor"


def test_a_two_card_run_keeps_the_first_card() -> None:
    """With two cards there is no majority to appeal to, so nothing to pick."""
    n = int(10 * FS)
    traces = {"a": sweep(n), "b": sweep(n)}
    assert B._pick_anchor(["a", "b"], traces, int(5 * FS), None) == "a"


# ---------------------------------------------------------------------------
# the one-way shift
# ---------------------------------------------------------------------------

def test_a_refused_lag_is_ignored_by_both_halves_or_neither() -> None:
    """consensus_schedule and the phasor pass must read `lag` the same way.

    consensus_schedule moves each card's detected windows ONTO the common
    time base by subtracting the lag; process_card moves them back by adding
    it. If only one of the two honours `applied`, a refused card is shifted
    one way and never back, and every window on it is off by exactly the lag
    that was judged untrustworthy -- which is how a refused alignment turned
    into 43 segments failing the SNR gate rather than into a warning.
    """
    import inspect

    src = inspect.getsource(B)
    consensus = src[src.index("def consensus_schedule"):]
    consensus = consensus[:consensus.index("\ndef ", 1)]

    assert 'info.get("applied")' in consensus, (
        "consensus_schedule must gate its shift on `applied`, exactly as the "
        "phasor pass does")


# ---------------------------------------------------------------------------
# the band the correlation runs over
# ---------------------------------------------------------------------------

def test_the_alignment_band_follows_the_excitation() -> None:
    """It used to be the literal 0.5 .. 300 Hz, whatever was in the record.

    That is the right band for a sweep that ends at 250 Hz and it deletes
    the entire excitation of one that starts above 300.  What then goes into
    the cross-correlation is the noise that survived the filter, and the
    correlation does not fail -- it returns the best agreement between two
    noise records and scores its prominence against that same noise. A run
    can come back "aligned" on lags that are pure chance.
    """
    fs = 100_000.0
    n = int(2.0 * fs)
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    # excitation entirely above the old 300 Hz ceiling
    hf = sum(np.sin(2 * np.pi * f * t) for f in (1200.0, 2500.0, 4000.0))
    traces = {"a": hf + 0.1 * rng.standard_normal(n),
              "b": hf + 0.1 * rng.standard_normal(n)}

    lo, hi = B._energy_band(traces, fs, DEFAULT, log=None)
    assert hi > 300.0, (
        f"the band stopped at {hi:.0f} Hz, so the excitation at 1.2-4 kHz "
        f"would have been filtered out before the correlation saw it")
    assert lo < 1500.0 and hi > 3000.0


def test_a_high_rate_card_still_gets_a_prominent_peak() -> None:
    """The prominence guard was 5000 SAMPLES, i.e. 0.5 s at 10 kHz and
    0.05 s at 100 kHz.

    On a fast card most of the peak's own skirt therefore stayed in the
    "background", inflating the MAD and pushing a perfectly good lag below
    the gate -- and a refused lag is not a warning downstream, it is every
    segment on that card failing silver's SNR gate.
    """
    fs = 100_000.0
    rng = np.random.default_rng(1)
    # A SWEEP, not a sum of tones. A sum of harmonically related sines is
    # periodic, and the cross-correlation of a periodic signal is ambiguous
    # by construction -- every lag differing by a period scores identically.
    # That is a real property of the measurement, handled separately; it is
    # not what this test is about.
    clean = _hf_sweep(int(3.0 * fs), fs)
    n = len(clean)
    shift = int(1.2345 * fs)
    a = clean + 0.5 * rng.standard_normal(n)
    b = np.roll(clean, shift) + 0.5 * rng.standard_normal(n)

    lag, _corr, prom = B._best_lag(b, a, max_lag=int(5 * fs))
    assert lag == pytest.approx(shift, abs=2)
    assert prom > DEFAULT.align_min_prominence, (
        f"a correct lag on a 100 kHz card scored prominence {prom:.1f}, "
        f"below the gate of {DEFAULT.align_min_prominence}")


def test_the_decimated_search_returns_a_full_rate_lag() -> None:
    """Searching at 1/q rate must still answer in full-rate samples.

    Without decimation the anchor vote alone is n*(n-1) FFTs over twice a
    100 kHz record; with it, the coarse winner is refined at the full rate
    over the one decimated bin it fell in.
    """
    fs = 100_000.0
    rng = np.random.default_rng(2)
    clean = _hf_sweep(int(2.0 * fs), fs)
    n = len(clean)
    shift = 54_321
    a = clean + 0.3 * rng.standard_normal(n)
    b = np.roll(clean, shift) + 0.3 * rng.standard_normal(n)

    full = B._best_lag(b, a, max_lag=int(1.0 * fs))[0]
    dec = B._best_lag(b, a, max_lag=int(1.0 * fs), decim=8)[0]
    assert full == pytest.approx(shift, abs=2)
    assert dec == pytest.approx(shift, abs=2), (
        f"decimated search gave {dec}, full rate {full}, truth {shift}")


def test_the_reported_lag_seconds_use_the_cards_own_rate() -> None:
    """This divided by a hard-coded 10000.0.

    Every lag reported for a plate recorded at any other rate was therefore
    wrong by exactly that ratio -- in the one number a reader would use to
    sanity-check the alignment against the header stamps.
    """
    from pathlib import Path
    fs = 100_000.0
    cards = {"c1": B.CardInfo(path=Path("c1.DAT"), stem="c1", fs=fs, n_ch=16,
                              n_samples=1000, duration_s=0.01, ref_name="UC1",
                              ref_slot=0, n_segments=15)}
    run = B.BronzeRun(
        schedule=[], channels={}, spectra={}, cards=cards, grid={},
        config_digest="x", input_digest="y", n_files=1,
        lags={"c1": {"lag": int(2.5 * fs), "corr": 1.0, "applied": True}})
    assert run.summary()["card_lag_s"]["c1"] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# a periodic excitation makes the lag ambiguous by whole periods
# ---------------------------------------------------------------------------

def test_a_periodic_multisine_gives_several_equal_peaks() -> None:
    """Pinning the hazard itself, before pinning the fix.

    A designed multisine puts every tone on a multiple of a base frequency,
    so the excitation repeats every 1/f0 seconds and so does its
    autocorrelation.  Every lag differing by a period scores identically --
    not approximately, identically -- so the largest peak is chosen among
    equals by noise.
    """
    fs = 10_000.0
    n = int(4.0 * fs)
    t = np.arange(n) / fs
    f0 = 5.0                                    # period 0.2 s = 2000 samples
    rng = np.random.default_rng(0)
    clean = sum(np.sin(2 * np.pi * (k * f0) * t) for k in (2, 3, 5, 8, 13))
    shift = 7000                                # 3.5 periods
    a = clean + 0.2 * rng.standard_normal(n)
    b = np.roll(clean, shift) + 0.2 * rng.standard_normal(n)

    cands = B._lag_candidates(b, a, max_lag=int(1.5 * fs))
    assert len(cands) >= 2, "a periodic excitation must produce ties"
    gaps = np.diff(sorted(c[0] for c in cands))
    assert np.allclose(gaps, 2000, atol=40), (
        f"the ties should sit one base period (2000 samples) apart, got {gaps}")


def test_the_header_stamps_break_the_tie() -> None:
    """The stamps are coarse, but they are not periodic.

    That is the whole reason they can settle this -- provided the period is
    longer than their resolution, which is the case tested here.
    """
    from datetime import datetime, timedelta
    from pathlib import Path

    fs = 10_000.0
    n = int(30.0 * fs)
    t = np.arange(n) / fs
    f0 = 0.5                                    # period 2 s = 20000 samples
    shift = 60_000                              # 3 periods
    rng = np.random.default_rng(1)
    clean = sum(np.sin(2 * np.pi * (k * f0) * t) for k in (2, 3, 5, 8, 13))
    a = clean + 0.2 * rng.standard_normal(n)
    b = np.roll(clean, shift) + 0.2 * rng.standard_normal(n)

    t0 = datetime(2025, 7, 16, 7, 45, 46)

    def _card(stem, off):
        return B.CardInfo(path=Path(f"{stem}.DAT"), stem=stem, fs=fs, n_ch=16,
                          n_samples=n, duration_s=n / fs, ref_name="UC1",
                          ref_slot=0, n_segments=15,
                          start_time=t0 + timedelta(seconds=off))

    # b's copy of the event sits 60000 samples (6 s) later in its own array,
    # which is what being armed 6 s EARLIER looks like.
    cards = {"anchor": _card("anchor", 6), "other": _card("other", 0)}
    tb = B.timebase_report(cards, DEFAULT)

    raw = B._best_lag(b, a, max_lag=int(9 * fs))[0]
    chosen, _corr = B._disambiguate(
        "other", "anchor", raw, 0.5, b, a, int(9 * fs), 1, cards, fs, tb,
        DEFAULT, B.utils.get_logger(False))
    assert chosen == pytest.approx(shift, abs=60), (
        f"raw peak {raw}, header-guided choice {chosen}, truth {shift}")


def test_an_ambiguity_finer_than_the_stamps_is_reported_not_guessed() -> None:
    """A 5 Hz multisine repeats every 0.2 s; the stamps tick at ~1 s.

    Ten candidates fit inside one tick, so snapping to the nearest would be
    picking one of them by rounding error and reporting it as measured. The
    honest output is the residual ambiguity.
    """
    from datetime import datetime, timedelta
    from pathlib import Path
    import logging

    fs = 10_000.0
    n = int(4.0 * fs)
    t = np.arange(n) / fs
    f0, shift = 5.0, 7000
    rng = np.random.default_rng(1)
    clean = sum(np.sin(2 * np.pi * (k * f0) * t) for k in (2, 3, 5, 8, 13))
    a = clean + 0.2 * rng.standard_normal(n)
    b = np.roll(clean, shift) + 0.2 * rng.standard_normal(n)

    t0 = datetime(2025, 7, 16, 7, 45, 46)

    def _card(stem, off):
        return B.CardInfo(path=Path(f"{stem}.DAT"), stem=stem, fs=fs, n_ch=16,
                          n_samples=n, duration_s=n / fs, ref_name="UC1",
                          ref_slot=0, n_segments=15,
                          start_time=t0 + timedelta(seconds=off))

    cards = {"anchor": _card("anchor", 1), "other": _card("other", 0)}
    tb = B.timebase_report(cards, DEFAULT)

    seen: list[str] = []

    class _Log:
        def info(self, m): seen.append(str(m))
        def warning(self, m): seen.append(str(m))
        def error(self, m): seen.append(str(m))

    raw = B._best_lag(b, a, max_lag=int(1.5 * fs))[0]
    chosen, _c = B._disambiguate("other", "anchor", raw, 0.5, b, a,
                                 int(1.5 * fs), 1, cards, fs, tb, DEFAULT,
                                 _Log())
    assert chosen == raw, "nothing should be snapped when the stamps cannot tell"
    assert any("ambiguous" in m and "cannot choose" in m for m in seen), seen
