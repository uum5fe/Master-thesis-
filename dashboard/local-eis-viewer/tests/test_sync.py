"""Synchronisation accuracy against imposed, known delays."""

from __future__ import annotations

import numpy as np
import pytest

from eis.sync.drift import estimate_drift
from eis.sync.resample import (
    advance_affine, fractional_delay_fft, resample_at, valid_span,
)
from eis.sync.skew import closure_residual, estimate_delay, gcc_phat

FS = 10_000.0
BAND = (20.0, 4000.0)
TONES = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 3571.0]


def multisine(n: int, seed: int = 0, noise: float = 5e-7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    x = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f in TONES)
    x *= 4e-3 / x.std()
    return x + noise * rng.standard_normal(n)


def impose(x: np.ndarray, tau_s: float, drift_ppm: float = 0.0) -> np.ndarray:
    """Return a copy that lags ``x`` by ``tau(t) = tau_s + drift*t``."""
    n = len(x)
    if drift_ppm == 0.0:
        # Exact for a constant delay: whole samples by rolling, the remainder
        # by an FFT phase ramp.  Keeps the fixture from limiting the accuracy
        # the estimator is being measured against.
        whole = int(np.round(tau_s * FS))
        frac = tau_s - whole / FS
        return np.roll(fractional_delay_fft(x, frac, FS), whole)
    return resample_at(
        x, np.arange(n) * (1.0 - drift_ppm * 1e-6) - tau_s * FS
    )


# ---------------------------------------------------------------------------

@pytest.mark.parametrize("samples", [0.0, 0.3, 1.7, 12.4, -5.2, 31.0])
def test_constant_delay_recovered_within_budget(samples: float) -> None:
    """The 73 ns budget is 0.1 deg at 3.8 kHz; assert well inside it."""
    n = int(FS * 30)
    ref = multisine(n, seed=1)
    tau = samples / FS
    test = impose(ref, tau) + 5e-7 * np.random.default_rng(2).standard_normal(n)

    est = estimate_delay(ref, test, FS, band_hz=BAND, nperseg=1 << 15)
    assert est.ok
    assert abs(est.tau_s - tau) < 73e-9, f"error {(est.tau_s - tau) * 1e9:.1f} ns"
    assert abs(est.intercept_rad) < 0.05


def test_delay_much_larger_than_a_sample_does_not_wrap() -> None:
    """Phase unwrapping is only safe after the coarse stage removes the bulk."""
    n = int(FS * 30)
    ref = multisine(n, seed=3)
    tau = 137.0 / FS                       # 13.7 ms: many wraps at 3.8 kHz
    test = impose(ref, tau)
    # Trim the ends: imposing a 137-sample shift necessarily invents samples
    # there, and that is a property of the test fixture, not the estimator.
    edge = 500
    est = estimate_delay(
        ref[edge:-edge], test[edge:-edge], FS, band_hz=BAND, nperseg=1 << 15
    )
    assert abs(est.tau_s - tau) < 200e-9


def test_clock_drift_is_separated_from_offset() -> None:
    n = int(FS * 60)
    ref = multisine(n, seed=4)
    tau0, slope = 3.1e-3, 6.0
    test = impose(ref, tau0, slope)

    drift = estimate_drift(ref, test, FS, n_blocks=10, band_hz=BAND)
    assert drift.ok
    assert abs(drift.delay_slope_ppm - slope) < 0.2
    assert drift.clock_rate_ppm == pytest.approx(-drift.delay_slope_ppm)
    assert drift.significant(duration_s=60.0, f_max_hz=3800.0)


def test_drift_correction_round_trip() -> None:
    n = int(FS * 60)
    ref = multisine(n, seed=5)
    test = impose(ref, 1.5e-3, 4.0)

    drift = estimate_drift(ref, test, FS, n_blocks=10, band_hz=BAND)
    fixed = advance_affine(test, FS, drift.tau0_s, drift.delay_slope_ppm)
    lo, hi = valid_span(n, FS, drift.tau0_s, drift.delay_slope_ppm)

    after = estimate_drift(ref[lo:hi], fixed[lo:hi], FS, n_blocks=10, band_hz=BAND)
    assert abs(after.delay_slope_ppm) < 0.05
    residual = estimate_delay(ref[lo:hi], fixed[lo:hi], FS, band_hz=BAND, nperseg=1 << 15)
    assert abs(residual.tau_s) < 2e-6


def test_no_common_signal_is_rejected_not_guessed() -> None:
    """An uncorrelated channel must fail loudly rather than return a number."""
    n = int(FS * 20)
    ref = multisine(n, seed=6)
    noise = 1e-5 * np.random.default_rng(7).standard_normal(n)
    est = estimate_delay(ref, noise, FS, band_hz=BAND, nperseg=1 << 15)
    assert not est.ok
    assert est.note


def test_fractional_delay_is_band_limited_and_exact() -> None:
    rng = np.random.default_rng(8)
    n = 1 << 16
    spectrum = np.fft.rfft(rng.standard_normal(n))
    spectrum[np.fft.rfftfreq(n, 1 / FS) > 4000] = 0
    x = np.fft.irfft(spectrum, n)
    x /= x.std()

    truth = fractional_delay_fft(x, -0.37 / FS, FS)
    got = resample_at(x, np.arange(n) + 0.37)
    interior = slice(300, n - 300)
    assert np.sqrt(np.mean((got[interior] - truth[interior]) ** 2)) < 1e-3


def test_triplet_closure_detects_an_inconsistent_estimate() -> None:
    consistent = {(1, 2): 100e-9, (2, 3): 50e-9, (1, 3): 150e-9}
    assert abs(closure_residual(consistent)[(1, 2, 3)]) < 1e-12

    broken = {(1, 2): 100e-9, (2, 3): 50e-9, (1, 3): 900e-9}
    assert abs(closure_residual(broken)[(1, 2, 3)]) > 500e-9


def test_coarse_search_is_bounded_and_reports_ambiguity() -> None:
    n = int(FS * 10)
    ref = multisine(n, seed=9)
    _, sharp = gcc_phat(ref, impose(ref, 2.0 / FS), FS, band_hz=BAND)
    assert sharp > 1.5
    _, sharp_noise = gcc_phat(
        ref, np.random.default_rng(10).standard_normal(n), FS, band_hz=BAND
    )
    assert sharp_noise < sharp
