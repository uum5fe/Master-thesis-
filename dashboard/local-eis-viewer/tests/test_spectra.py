"""Impedance estimation: bias, leakage, gating and uncertainty."""

from __future__ import annotations

import numpy as np
import pytest

from eis.spectra import (
    apply_notch, detect_tones, impedance_synchronous, impedance_welch,
    relative_uncertainty, transfer_estimators,
)

FS = 10_000.0
TONES = [2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 3571.0]


def reference_Z(f: np.ndarray) -> np.ndarray:
    """Rs + jwL + one depressed arc."""
    w = 2 * np.pi * np.asarray(f, float)
    Rs, L, Rct, tau, n = 0.014, 8e-9, 0.018, 2.5e-3, 0.88
    return Rs + 1j * w * L + Rct / (1.0 + (1j * w) ** n * (tau**n / Rct) * Rct)


def make_pair(
    duration_s: float = 40.0, noise_i: float = 0.0, noise_u: float = 0.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Current and voltage consistent with ``reference_Z``."""
    rng = np.random.default_rng(seed)
    n = int(FS * duration_s)
    t = np.arange(n) / FS
    u = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f in TONES)
    u *= 4e-3 / u.std()

    freqs = np.fft.rfftfreq(n, 1 / FS)
    U = np.fft.rfft(u)
    I = np.where(freqs > 0, U / reference_Z(np.maximum(freqs, 1e-9)), 0.0)
    i = np.fft.irfft(I, n)

    return (
        i + noise_i * rng.standard_normal(n),
        u + noise_u * rng.standard_normal(n),
    )


# ---------------------------------------------------------------------------

def test_synchronous_dft_is_leakage_free() -> None:
    i, u = make_pair()
    result = impedance_synchronous(i, u, FS, 1.0, TONES, periods=4)
    error = np.abs(result.Z - reference_Z(result.f)) / np.abs(reference_Z(result.f))
    assert len(result.f) == len(TONES)
    assert np.median(error) < 1e-3
    assert result.method == "synchronous"


def test_synchronous_beats_welch_on_off_bin_tones() -> None:
    """Welch's nearest-bin pick biases off-bin tones; synchronous DFT does not."""
    i, u = make_pair()
    sync = impedance_synchronous(i, u, FS, 1.0, TONES, periods=4)
    welch = impedance_welch(i, u, FS, nperseg=8192, f_min=1.0, f_max=4000.0)
    welch = welch.gate(0.95)

    def err(res):
        Z_true = reference_Z(res.f)
        return float(np.max(np.abs(res.Z - Z_true) / np.abs(Z_true)))

    assert err(sync) < err(welch)


def test_synchronous_rejects_a_non_integer_window() -> None:
    i, u = make_pair(duration_s=10.0)
    with pytest.raises(ValueError, match="not an integer"):
        impedance_synchronous(i, u, FS, base_frequency_hz=3.3, tones_hz=TONES)


def test_h1_is_biased_low_and_hv_is_not() -> None:
    """The core reason the default estimator is Hv, not H1."""
    i, u = make_pair(noise_i=3.0, seed=3)      # heavy noise on the *input*
    h1 = impedance_synchronous(i, u, FS, 1.0, TONES, periods=4, estimator="h1")
    h2 = impedance_synchronous(i, u, FS, 1.0, TONES, periods=4, estimator="h2")
    hv = impedance_synchronous(i, u, FS, 1.0, TONES, periods=4, estimator="hv")
    Z_true = reference_Z(hv.f)

    assert np.median(hv.coherence) < 0.9, "the test needs degraded coherence"

    bias_h1 = np.median(np.abs(h1.Z) / np.abs(Z_true))
    bias_h2 = np.median(np.abs(h2.Z) / np.abs(Z_true))
    bias_hv = np.median(np.abs(hv.Z) / np.abs(Z_true))

    assert bias_h1 < 0.95, "H1 must under-read when the input is noisy"
    assert bias_h2 > 1.05, "H2 must over-read"
    assert bias_h1 < bias_hv < bias_h2, "Hv must sit inside the H1/H2 bracket"
    assert abs(bias_hv - 1.0) < abs(bias_h1 - 1.0), "Hv must beat H1 here"


def test_h2_over_h1_equals_inverse_coherence() -> None:
    Sxx = np.array([4.0 + 0j, 9.0 + 0j])
    Syy = np.array([25.0 + 0j, 49.0 + 0j])
    Sxy = np.array([8.0 + 2j, 17.0 - 3j])
    H1, H2, _, gamma2 = transfer_estimators(Sxx, Syy, Sxy)
    assert np.allclose(np.abs(H2 / H1), 1.0 / gamma2)


def test_uncertainty_predicts_the_actual_scatter() -> None:
    """The reported sigma must be a real error bar, not decoration."""
    errors, predicted = [], []
    for seed in range(12):
        i, u = make_pair(duration_s=40.0, noise_i=6e-3, noise_u=4e-5, seed=100 + seed)
        res = impedance_synchronous(i, u, FS, 1.0, TONES, periods=4)
        Z_true = reference_Z(res.f)
        errors.append(np.abs(res.Z - Z_true) / np.abs(Z_true))
        predicted.append(res.sigma_rel)

    observed = np.std(np.concatenate(errors))
    claimed = np.median(np.concatenate(predicted))
    assert 0.3 < observed / claimed < 3.0, (observed, claimed)


def test_window_gate_removes_a_transient() -> None:
    i, u = make_pair(duration_s=40.0, seed=5)
    corrupted = u.copy()
    corrupted[150_000:158_000] += 0.5          # a load transient

    gated = impedance_synchronous(i, corrupted, FS, 1.0, TONES, periods=4, gate=True)
    ungated = impedance_synchronous(i, corrupted, FS, 1.0, TONES, periods=4, gate=False)
    Z_true = reference_Z(gated.f)

    assert gated.n_windows_used < gated.n_windows_total
    assert np.median(np.abs(gated.Z - Z_true) / np.abs(Z_true)) < np.median(
        np.abs(ungated.Z - Z_true) / np.abs(Z_true)
    )


def test_notch_is_zero_phase() -> None:
    """A causal notch would inject exactly the delay the pipeline removes."""
    rng = np.random.default_rng(6)
    n = 1 << 15
    t = np.arange(n) / FS
    x = np.sin(2 * np.pi * 200 * t) + 0.01 * rng.standard_normal(n)
    y = apply_notch(x, FS, [50.0, 100.0, 150.0], 30.0)
    interior = slice(2000, n - 2000)
    lag = np.argmax(
        np.correlate(y[interior], x[interior], mode="same")
    ) - len(x[interior]) // 2
    assert lag == 0


def test_relative_uncertainty_shrinks_with_averages() -> None:
    gamma2 = np.array([0.9])
    assert relative_uncertainty(gamma2, 100.0) < relative_uncertainty(gamma2, 10.0)
    assert relative_uncertainty(np.array([1.0 - 1e-12]), 10.0) < 1e-5


def test_detect_tones_finds_the_excitation() -> None:
    f = np.linspace(0, 1000, 5000)
    coherence = np.full_like(f, 0.2)
    for centre in (100.0, 250.0, 700.0):
        coherence[np.abs(f - centre) < 0.5] = 0.99
    found = detect_tones(f, coherence, threshold=0.95, gap_hz=3.0)
    assert np.allclose(np.round(found), [100, 250, 700], atol=1)
