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
