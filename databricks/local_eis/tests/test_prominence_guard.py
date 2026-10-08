"""The prominence guard is a duration, and widening it costs nothing.

`_prominence` scores the correlation peak against the rest of the curve, and
the guard is the region around the peak that is NOT counted as background --
the peak's own shoulders, which are part of the peak.

The guard used to be a fixed 5000 samples, which is 0.5 s on a 10 kHz card
and 0.05 s on a 100 kHz one: the same recording scored differently for no
reason but its sample rate. The shoulders are about 1/align_f_lo_hz wide,
a duration, so the guard is one too.

These tests pin the measurement that chose the 2.0 s default: widening the
guard lifts a real peak and leaves the null where it is.
"""

from __future__ import annotations

import numpy as np
import pytest

import bronze as B
from config import DEFAULT


FS = 10_000.0
N = int(30 * FS)


def _sweep(n: int) -> np.ndarray:
    x = np.zeros(n)
    t = np.arange(n) / FS
    span = n // 24
    for k, freq in enumerate(np.geomspace(1.0, 250.0, 24)):
        a, b = k * span, min(n, k * span + int(0.7 * span))
        x[a:b] = np.sin(2 * np.pi * freq * t[a:b])
    return x - x.mean()


def _real_peak_prominence(guard: int) -> float:
    clean = _sweep(N)
    shift = int(5.7121 * FS)
    a = clean + 0.35 * np.random.default_rng(1).standard_normal(N)
    b = np.roll(clean, shift) + 0.35 * np.random.default_rng(2).standard_normal(N)
    lag, _corr, prom = B._best_lag(b, a, max_lag=int(12 * FS), guard=guard)
    assert lag == pytest.approx(shift, abs=2), "fixture broken: wrong lag"
    return prom


def _null_max_prominence(guard: int, seeds: int = 8) -> float:
    worst = 0.0
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        _l, _c, p = B._best_lag(rng.standard_normal(N), rng.standard_normal(N),
                                max_lag=int(12 * FS), guard=guard)
        worst = max(worst, p)
    return worst


def test_a_wider_guard_lifts_a_real_peak() -> None:
    """Counting the peak's own shoulders as background understates it."""
    narrow = _real_peak_prominence(int(0.05 * FS))
    wide = _real_peak_prominence(int(2.00 * FS))
    assert wide > 1.2 * narrow


def test_a_wider_guard_does_not_lift_the_null() -> None:
    """The reason the change is free.

    Noise has no shoulders, so excluding a wider region around its highest
    point changes nothing about the background that point is scored against.
    The null ceiling therefore stays put while real peaks climb -- which is
    why align_min_prominence did not have to be re-derived when the guard
    grew from 5000 samples to 2 s, and why nothing that passed the gate
    before can fail it now.
    """
    narrow = _null_max_prominence(int(0.05 * FS))
    wide = _null_max_prominence(int(2.00 * FS))
    assert wide == pytest.approx(narrow, rel=0.10)


def test_separation_improves_with_the_configured_guard() -> None:
    """End to end: the shipped default separates better than the old one."""
    old = _real_peak_prominence(5000) / _null_max_prominence(5000)
    new_guard = B._guard_samples(DEFAULT, FS)
    new = _real_peak_prominence(new_guard) / _null_max_prominence(new_guard)
    assert new > old


def test_the_guard_is_taken_from_the_config_in_samples() -> None:
    """A duration means a different sample count on every card."""
    cfg = DEFAULT.replace(align_guard_s=2.0)
    assert B._guard_samples(cfg, 10_000.0) == 20_000
    assert B._guard_samples(cfg, 100_000.0) == 200_000
    # zero or missing falls back to the historical fixed count
    assert B._guard_samples(DEFAULT.replace(align_guard_s=0.0),
                            10_000.0) == B.GUARD_SAMPLES_LEGACY


def test_a_starved_background_is_nan_not_a_number() -> None:
    """A guard wider than the curve leaves nothing to score against.

    Returning NaN matters: NaN fails `prom >= align_min_prominence`, so a
    starved diagnostic refuses the lag instead of accepting it on a score
    computed from a handful of samples.
    """
    mag = np.abs(np.random.default_rng(0).standard_normal(200))
    assert np.isnan(B._prominence(mag, 100, guard=95))
