"""The high-frequency accuracy chain, against impedances the test knows.

Each test states the fault it imposes and the number the pipeline has to get
back.  The faults are the ones that actually happen: a via shunt with loop
inductance, a mismatched anti-alias filter, a residual delay shared by a card,
and in-plane current shared between neighbouring segments.
"""

from __future__ import annotations

import numpy as np
import pytest

from eis import hf

FS = 10_000.0


def reference_Z(f: np.ndarray, Rs: float = 0.014, L: float = 8e-9) -> np.ndarray:
    """Rs + jwL + one depressed arc - the shape a segment actually has."""
    w = 2 * np.pi * np.asarray(f, float)
    Rct, tau, n = 0.018, 2.5e-3, 0.88
    return Rs + 1j * w * L + Rct / (1.0 + (1j * w) ** n * (tau**n / Rct) * Rct)


# ---------------------------------------------------------------------------
# 1. The complex instrument response
# ---------------------------------------------------------------------------

def test_shunt_time_constant_uses_the_area_normalised_resistance() -> None:
    """H is a voltage per current *density*, so the area is part of R."""
    # 1.05e-4 V*cm^2/A over 4.235 cm^2 is a 24.8 uOhm shunt.
    tau = hf.shunt_time_constant(1.05e-4, 4.235, inductance_nh=0.6)
    assert tau == pytest.approx(0.6e-9 / (1.05e-4 / 4.235), rel=1e-12)
    assert 20e-6 < tau < 30e-6, "tens of microseconds, not nanoseconds"
    # Which is tens of degrees at the top of the band - the whole point.
    response = hf.InstrumentResponse(shunt_tau_s=tau)
    assert response.phase_deg_at(3800.0) > 25.0


def test_shunt_correction_inverts_an_imposed_shunt_inductance() -> None:
    from tests.synthetic import shunt_tau_s

    f = np.geomspace(1.0, 3800.0, 60)
    w = 2 * np.pi * f
    H, area, inductance = 1.05e-4, 4.235, 0.6
    tau = shunt_tau_s(H, area, inductance)

    # The current is read through a shunt that is also a small inductor, so the
    # apparent impedance is divided by (1 + jw*tau).
    Z_true = reference_Z(f)
    Z_measured = Z_true / (1.0 + 1j * w * tau)

    assert np.max(np.abs(Z_measured - Z_true) / np.abs(Z_true)) > 0.4, (
        "the fault has to be large or the test proves nothing"
    )
    response = hf.InstrumentResponse(
        shunt_tau_s=hf.shunt_time_constant(H, area, inductance)
    )
    Z_corrected = Z_measured * response.factor(f)
    assert np.max(np.abs(Z_corrected - Z_true) / np.abs(Z_true)) < 1e-12


def test_zero_inductance_is_exactly_a_no_op() -> None:
    f = np.geomspace(1.0, 3800.0, 20)
    assert hf.shunt_time_constant(1.05e-4, 4.235, 0.0) == 0.0
    assert not hf.InstrumentResponse().active
    assert np.allclose(hf.InstrumentResponse().factor(f), 1.0)


def test_butterworth_response_is_unity_at_dc_and_minus_3db_at_the_corner() -> None:
    f = np.array([0.0, 1000.0])
    response = hf.butterworth_response(f, order=4, corner_hz=1000.0)
    assert response[0] == pytest.approx(1.0, abs=1e-12)
    assert abs(response[1]) == pytest.approx(1 / np.sqrt(2), rel=1e-6)


def test_matched_anti_alias_filters_cancel_and_a_mismatch_does_not() -> None:
    f = np.geomspace(100.0, 3800.0, 40)
    matched = hf.InstrumentResponse(aa_order=4, aa_corner_hz=4000.0,
                                    aa_corner_mismatch=0.0)
    assert np.allclose(matched.factor(f), 1.0)

    skewed = hf.InstrumentResponse(aa_order=4, aa_corner_hz=4000.0,
                                   aa_corner_mismatch=0.02)
    assert 0.05 < abs(skewed.phase_deg_at(3800.0)) < 5.0, (
        "a 2 % corner tolerance is worth a fraction of a degree - small, but "
        "not nothing, and it is systematic"
    )


# ---------------------------------------------------------------------------
# 2. The high-frequency resistance fit
# ---------------------------------------------------------------------------

def test_hf_fit_recovers_rs_and_l_and_beats_taking_the_median() -> None:
    """The median of the top real parts is biased by the inductive tail."""
    rng = np.random.default_rng(0)
    f = np.geomspace(1.0, 3800.0, 80)
    Rs, L = 0.0142, 9e-9
    Z = reference_Z(f, Rs, L)
    noise = 2e-4 * np.abs(Z) * (
        rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))
    )
    sigma = 2e-4 * np.abs(Z)

    fit = hf.fit_hf_resistance(f, Z + noise, sigma)
    assert fit.ok
    assert abs(fit.rs_ohm - Rs) / Rs < 0.01
    assert abs(fit.l_h - L) / L < 0.5
    assert np.isfinite(fit.rs_sigma_ohm) and fit.rs_sigma_ohm > 0

    order = np.argsort(f)
    n_hf = max(2, len(f) // 5)
    median_estimate = float(np.median(np.sort((Z + noise).real[order])[-n_hf:]))
    assert abs(fit.rs_ohm - Rs) < abs(median_estimate - Rs), (
        "the weighted Rs + jwL + arc fit must beat the median of the top "
        "real parts, which is what it replaced"
    )


def test_hf_fit_reports_rather_than_guesses_when_there_are_too_few_points() -> None:
    f = np.array([1000.0, 2000.0])
    fit = hf.fit_hf_resistance(f, reference_Z(f), min_points=4)
    assert not fit.ok
    assert "2 points" in fit.note


def test_crossover_frequency_is_where_the_arc_turns_inductive() -> None:
    f = np.geomspace(1.0, 3800.0, 80)
    fit = hf.fit_hf_resistance(f, reference_Z(f, L=9e-9))
    crossover = fit.crossover_hz
    assert np.isfinite(crossover)
    # -Im(Z) of the truth must change sign near the reported crossover.
    below = -reference_Z(np.array([crossover * 0.5]), L=9e-9).imag[0]
    above = -reference_Z(np.array([crossover * 2.0]), L=9e-9).imag[0]
    assert below > 0 > above


# ---------------------------------------------------------------------------
# 3. Pooled identification of the shared delay
# ---------------------------------------------------------------------------

def _card_of_segments(
    delay_s: float, n: int = 16, seed: int = 3, rs_spread: float = 0.30,
    noise: float = 1e-4,
) -> dict[int, hf.HFRFit]:
    """A card whose segments share ``delay_s`` and differ in Rs and L."""
    rng = np.random.default_rng(seed)
    f = np.geomspace(300.0, 3800.0, 60)
    w = 2 * np.pi * f
    fits: dict[int, hf.HFRFit] = {}
    for k in range(n):
        Rs = 0.014 * (1.0 + rs_spread * k / max(n - 1, 1))
        L = 8e-9 * (1.0 + 0.25 * rng.standard_normal())   # independent of Rs
        Z = reference_Z(f, Rs, L)
        scatter = noise * np.abs(Z) * (
            rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))
        )
        # A shared delay shows up as exp(+jw*eps) on the measured spectrum.
        measured = (Z + scatter) * np.exp(1j * w * delay_s)
        fits[k + 1] = hf.fit_hf_resistance(f, measured, noise * np.abs(Z))
    return fits


def test_decorrelation_recovers_a_delay_shared_by_a_card() -> None:
    """L_k = L_true + eps*Rs_k, so the L-vs-Rs slope *is* the delay."""
    imposed = 400e-9
    result = hf.identify_common_delay(_card_of_segments(imposed), bound_s=2e-6)

    assert result.ok
    assert abs(result.delay_s - imposed) < 3 * result.delay_sigma_s
    assert abs(result.correlation_after) < abs(result.correlation_before), (
        "the correction has to remove the correlation that identified it"
    )


def test_decorrelation_returns_zero_when_there_is_no_delay() -> None:
    result = hf.identify_common_delay(_card_of_segments(0.0), bound_s=2e-6)
    assert result.ok
    assert abs(result.delay_s) < 3 * result.delay_sigma_s


def test_decorrelation_declines_when_rs_does_not_vary() -> None:
    """Without spread in Rs the regression is the ratio of two noises."""
    result = hf.identify_common_delay(
        _card_of_segments(400e-9, rs_spread=0.0), bound_s=2e-6
    )
    assert not result.ok
    assert "not identifiable" in result.note


def test_decorrelation_declines_on_too_few_segments() -> None:
    result = hf.identify_common_delay(
        _card_of_segments(400e-9, n=3), bound_s=2e-6, min_segments=4
    )
    assert not result.ok
    assert result.delay_s == 0.0


def test_decorrelation_is_bounded_by_the_measured_skew() -> None:
    """An unbounded search would absorb the cell's real inductance."""
    result = hf.identify_common_delay(_card_of_segments(4e-6), bound_s=100e-9)
    assert result.clipped
    assert abs(result.delay_s) == pytest.approx(100e-9)
    assert "clipped" in result.note


def test_pooled_kk_scan_recovers_a_coarse_delay() -> None:
    """The fallback for a card whose timing is only good to microseconds."""
    rng = np.random.default_rng(5)
    f = np.geomspace(2.0, 3800.0, 40)
    w = 2 * np.pi * f
    imposed = 6e-6
    spectra = {}
    for k in range(5):
        Z = reference_Z(f, 0.014 * (1 + 0.05 * k))
        sigma = 3e-4 * np.abs(Z)
        scatter = sigma * (
            rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))
        )
        # Measured spectrum carries exp(+jw*imposed); the correction that
        # removes it is exp(-jw*imposed), i.e. delay_s = +imposed.
        spectra[k] = ((Z + scatter) * np.exp(1j * w * imposed), sigma)
    packed = {k: (f, Z, s) for k, (Z, s) in spectra.items()}

    result = hf.pooled_kk_delay(
        packed, bound_s=30e-6, n_grid=25, max_elements=20,
        max_inductance_h=50e-9, rs_reference_ohm=0.015,
    )
    assert result.ok
    # Accurate to the inductance the bound still permits - which is what the
    # reported sigma says, and it is an order better than the ten microseconds
    # a segment-proxy delay is otherwise known to.
    assert abs(result.delay_s - imposed) < result.delay_sigma_s
    assert result.delay_sigma_s < 5e-6
    assert "inductance the bound still allows" in result.note


def test_reference_anchor_measures_the_delay_with_no_assumption() -> None:
    f = np.geomspace(10.0, 3800.0, 50)
    imposed = 250e-9
    Z_ref = reference_Z(f)
    Z_plate = Z_ref * np.exp(1j * 2 * np.pi * f * imposed)
    result = hf.delay_from_reference(f, Z_plate, Z_ref, band_hz=(100.0, 3800.0))
    assert result.ok
    assert result.delay_s == pytest.approx(imposed, rel=0.05)


# ---------------------------------------------------------------------------
# 4. In-plane crosstalk
# ---------------------------------------------------------------------------

def _grid_coords(n_cols: int = 8, n_rows: int = 4):
    coords = {}
    for i in range(n_cols * n_rows):
        r, c = divmod(i, n_cols)
        coords[i + 1] = (c * 2.0 + 1.0, r * 2.0 + 1.0, 0.9, 0.9)
    return coords


def test_crosstalk_deconvolution_undoes_the_spatial_smoothing() -> None:
    coords = _grid_coords()
    segments = sorted(coords)
    model = hf.build_crosstalk_model(segments, coords, alpha=0.15,
                                     regularisation=1e-6)
    assert model.active
    assert model.mean_neighbours > 2.0
    assert np.allclose(model.matrix.sum(axis=1), 1.0), (
        "current is redistributed, not created"
    )

    f = np.geomspace(1.0, 3800.0, 20)
    # A sharp spatial feature: one segment much more resistive than its
    # neighbours - exactly what smoothing hides.
    Y_true = np.array([
        1.0 / reference_Z(f, 0.030 if s == 12 else 0.014) for s in segments
    ])
    Y_measured = model.matrix @ Y_true

    smeared = np.abs(Y_measured[segments.index(12)] - Y_true[segments.index(12)])
    assert np.median(smeared / np.abs(Y_true[segments.index(12)])) > 0.05

    recovered, change = hf.deconvolve_crosstalk(model, Y_measured)
    error = np.abs(recovered - Y_true) / np.abs(Y_true)
    assert np.median(error) < 0.01
    assert change > 0


def test_crosstalk_is_disabled_by_default_and_says_so() -> None:
    coords = _grid_coords()
    model = hf.build_crosstalk_model(sorted(coords), coords, alpha=0.0)
    assert not model.active
    assert "disabled" in model.note
    Y = np.ones((len(coords), 5), complex)
    recovered, change = hf.deconvolve_crosstalk(model, Y)
    assert change == 0.0
    assert np.allclose(recovered, Y)


# ---------------------------------------------------------------------------
# 5. Uncertainty propagation
# ---------------------------------------------------------------------------

def test_timing_uncertainty_enters_the_phase_uncertainty() -> None:
    """A 10 us proxy is not as good as a 10 ns measurement, and must say so."""
    f = np.array([1.0, 100.0, 3800.0])
    sigma = np.full(3, 0.001)

    tight = hf.inflate_sigma_for_timing(sigma, f, 10e-9)
    loose = hf.inflate_sigma_for_timing(sigma, f, 10e-6)

    assert np.allclose(tight, sigma, rtol=0.05), "nanoseconds barely register"
    assert loose[0] == pytest.approx(sigma[0], rel=0.05), "and never at 1 Hz"
    assert loose[-1] > 0.2, "but microseconds dominate at the top of the band"
    assert np.all(loose >= sigma)


def test_zero_timing_uncertainty_changes_nothing() -> None:
    f = np.array([1.0, 3800.0])
    sigma = np.array([0.01, 0.01])
    assert np.allclose(hf.inflate_sigma_for_timing(sigma, f, 0.0), sigma)
    assert np.allclose(hf.inflate_sigma_for_timing(sigma, f, float("nan")), sigma)
