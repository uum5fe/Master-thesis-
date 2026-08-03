"""Data-validity checks: Kramers-Kronig, stationarity, plate consistency.

Kramers-Kronig testing here follows the linear (Lin-KK) approach: fit the
measured spectrum with a sum of RC (Voigt) elements, which is KK-compliant *by
construction*, and inspect the residuals.  Two design points matter.

* **The model order is chosen automatically** with the mu-criterion.  A fixed
  element count either under-fits (everything looks non-compliant) or over-fits
  (everything passes, because the model can chase noise).
* **The residual is a spectrum, not a scalar.**  Its shape identifies the
  fault:

  =========================================  ==================================
  residual signature                         diagnosis
  =========================================  ==================================
  random, within a percent, both parts       causal, linear, stationary - good
  systematic in Im, growing with frequency   uncorrected time delay
  divergent at the low-frequency end         drift / non-stationarity
  isolated points off in both parts          leakage or a contaminated tone
  =========================================  ==================================

The delay signature is usable, not just diagnostic: a Voigt series is
minimum-phase and cannot represent ``exp(-j*omega*dt)``, so scanning the delay
that minimises the KK residual recovers a residual timing error.  It is offered
here only as a *bounded refinement* of the measured skew, because over a finite
band a delay is degenerate with an inductance (``L = -Rs*dt`` to first order) -
an unbounded KK-optimal delay would happily eat the cell's real inductance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Lin-KK
# ---------------------------------------------------------------------------

@dataclass
class KKResult:
    n_elements: int
    mu: float
    residual_real: np.ndarray
    residual_imag: np.ndarray
    max_residual_pct: float
    rms_residual_pct: float
    Z_fit: np.ndarray
    passed: bool
    shape_class: str = "random"
    note: str = ""


def _voigt_design(f: np.ndarray, taus: np.ndarray, with_inductance: bool) -> np.ndarray:
    """Stacked [real; imag] design matrix for Rs + jwL + sum R_k/(1+jw tau_k)."""
    w = 2.0 * np.pi * f
    n = len(f)
    cols = [np.concatenate([np.ones(n), np.zeros(n)])]            # Rs
    if with_inductance:
        cols.append(np.concatenate([np.zeros(n), w]))             # jwL
    for tau in taus:
        denom = 1.0 + (w * tau) ** 2
        cols.append(np.concatenate([1.0 / denom, -w * tau / denom]))
    return np.column_stack(cols)


def _fit_voigt(
    f: np.ndarray, Z: np.ndarray, n_elements: int, weights: np.ndarray,
    with_inductance: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted linear least squares.  Returns ``(coeffs, taus, Z_fit)``."""
    taus = np.logspace(
        np.log10(1.0 / (2 * np.pi * f.max())),
        np.log10(1.0 / (2 * np.pi * max(f.min(), 1e-6))),
        n_elements,
    )
    A = _voigt_design(f, taus, with_inductance)
    b = np.concatenate([Z.real, Z.imag])
    w = np.concatenate([weights, weights])
    coeffs, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
    fitted = A @ coeffs
    return coeffs, taus, fitted[: len(f)] + 1j * fitted[len(f):]


def _mu(coeffs: np.ndarray, with_inductance: bool) -> float:
    """Schoenleber mu-criterion: 1 - sum|R_neg| / sum|R_pos|."""
    start = 2 if with_inductance else 1
    R = coeffs[start:]
    pos = float(np.abs(R[R > 0]).sum())
    neg = float(np.abs(R[R < 0]).sum())
    if pos <= 0:
        return 0.0
    return 1.0 - neg / pos


def _classify_residual(f: np.ndarray, dre: np.ndarray, dim: np.ndarray) -> str:
    """Name the dominant residual pattern."""
    if len(f) < 6:
        return "insufficient_points"
    lf = f <= np.quantile(f, 0.3)
    hf = f >= np.quantile(f, 0.7)
    scale = max(float(np.std(np.concatenate([dre, dim]))), 1e-12)

    # A pure delay tilts the imaginary residual monotonically with frequency.
    slope = np.polyfit(f, dim, 1)[0]
    tilt = abs(slope) * (f.max() - f.min()) / scale
    corr = float(np.corrcoef(f, dim)[0, 1]) if np.std(dim) > 0 else 0.0
    if tilt > 2.0 and abs(corr) > 0.7:
        return "delay_like"
    if np.mean(np.abs(dre[lf])) + np.mean(np.abs(dim[lf])) > 3.0 * (
        np.mean(np.abs(dre[hf])) + np.mean(np.abs(dim[hf])) + 1e-15
    ):
        return "lf_divergent"
    spikes = int(np.sum(np.abs(dre) + np.abs(dim) > 5 * scale))
    if spikes and spikes <= max(2, len(f) // 20):
        return "isolated_outliers"
    return "random"


def lin_kk(
    f: np.ndarray, Z: np.ndarray, sigma: np.ndarray | None = None,
    max_elements: int = 30, mu_target: float = 0.85,
    max_residual_pct: float = 1.0, with_inductance: bool = True,
) -> KKResult:
    """Linear Kramers-Kronig test with automatic model order."""
    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    order = np.argsort(f)
    f, Z = f[order], Z[order]
    if sigma is not None:
        sigma = np.asarray(sigma, float)[order]

    if len(f) < 6:
        return KKResult(
            0, 0.0, np.array([]), np.array([]), 100.0, 100.0, Z.copy(), False,
            "insufficient_points", f"only {len(f)} points; KK test needs >= 6",
        )

    # Inverse-variance weights where an uncertainty is available, otherwise
    # proportional weighting (the standard choice for impedance data).
    weights = 1.0 / np.maximum(sigma, 1e-15) if sigma is not None else 1.0 / np.abs(Z)

    best = None
    max_m = int(min(max_elements, max(3, len(f) // 2 - 1)))
    for m in range(3, max_m + 1):
        coeffs, _, Z_fit = _fit_voigt(f, Z, m, weights, with_inductance)
        mu = _mu(coeffs, with_inductance)
        best = (m, mu, Z_fit)
        if mu < mu_target:
            break

    m, mu, Z_fit = best                                   # type: ignore[misc]
    scale = np.abs(Z)
    dre = (Z.real - Z_fit.real) / scale
    dim = (Z.imag - Z_fit.imag) / scale
    max_pct = float(np.max(np.abs(np.concatenate([dre, dim])))) * 100.0
    rms_pct = float(np.sqrt(np.mean(np.concatenate([dre, dim]) ** 2))) * 100.0

    return KKResult(
        n_elements=m, mu=float(mu), residual_real=dre, residual_imag=dim,
        max_residual_pct=max_pct, rms_residual_pct=rms_pct, Z_fit=Z_fit,
        passed=bool(max_pct <= max_residual_pct),
        shape_class=_classify_residual(f, dre, dim),
    )


def kk_optimal_delay(
    f: np.ndarray, Z: np.ndarray, sigma: np.ndarray | None = None,
    bound_s: float = 200e-9, n_grid: int = 81, **kk_kwargs,
) -> tuple[float, float]:
    """Residual delay that minimises the KK residual, searched within a bound.

    Returns ``(delay_s, rms_residual_pct)``.  The bound must come from the
    measured skew uncertainty - see the module docstring for why an
    unconstrained search is not meaningful.
    """
    grid = np.linspace(-abs(bound_s), abs(bound_s), n_grid)
    best_tau, best_rms = 0.0, np.inf
    for tau in grid:
        rotated = Z * np.exp(1j * 2 * np.pi * f * tau)
        result = lin_kk(f, rotated, sigma, **kk_kwargs)
        if result.rms_residual_pct < best_rms:
            best_rms, best_tau = result.rms_residual_pct, float(tau)
    return best_tau, float(best_rms)


# ---------------------------------------------------------------------------
# Stationarity
# ---------------------------------------------------------------------------

@dataclass
class StationarityResult:
    max_relative_difference: float
    median_relative_difference: float
    passed: bool
    note: str = ""


def stationarity_split_half(
    Z_first: np.ndarray, Z_second: np.ndarray, tolerance: float = 0.05
) -> StationarityResult:
    """Compare impedances from the first and second half of the record.

    Kramers-Kronig assumes a time-invariant system.  A cell that dries out or
    floods during a 120 s record violates that premise, and the KK residual
    alone does not always reveal it.
    """
    n = min(len(Z_first), len(Z_second))
    if n == 0:
        return StationarityResult(np.inf, np.inf, False, "no overlapping points")
    scale = 0.5 * (np.abs(Z_first[:n]) + np.abs(Z_second[:n]))
    rel = np.abs(Z_first[:n] - Z_second[:n]) / np.maximum(scale, 1e-15)
    return StationarityResult(
        max_relative_difference=float(np.max(rel)),
        median_relative_difference=float(np.median(rel)),
        passed=bool(np.median(rel) <= tolerance),
    )


# ---------------------------------------------------------------------------
# Whole-plate consistency
# ---------------------------------------------------------------------------

@dataclass
class AdmittanceSumResult:
    f: np.ndarray = field(default_factory=lambda: np.array([]))
    Z_plate: np.ndarray = field(default_factory=lambda: np.array([], complex))
    Z_reference: np.ndarray | None = None
    median_relative_difference: float | None = None
    n_segments: int = 0
    note: str = ""


def admittance_sum(
    spectra: dict[int, tuple[np.ndarray, np.ndarray]],
    segment_area_cm2: float,
    cell_area_cm2: float,
    reference: tuple[np.ndarray, np.ndarray] | None = None,
) -> AdmittanceSumResult:
    """Combine segment impedances into a whole-plate impedance.

    The segments share one cell voltage and are therefore electrically in
    parallel::

        A_cell / Z_cell(f)  =  sum_k  A_k / Z_k(f)

    Comparing the reconstructed plate impedance against an independent
    cell-level instrument tests the calibration chain, the polarity map and the
    delay corrections simultaneously, at every frequency - a much stronger
    statement than agreement on the ohmic resistance alone.

    ``spectra`` maps segment number to ``(f, Z)``; all segments must share a
    frequency grid.
    """
    if not spectra:
        return AdmittanceSumResult(note="no segment spectra supplied")

    # Segments can end up with different frequency sets because the quality
    # gate keeps different points, so the sum is formed on the frequencies
    # that survive on *every* segment rather than demanding identical grids.
    def key(values: np.ndarray) -> np.ndarray:
        return np.round(np.asarray(values, float), 6)

    common = None
    for f_k, _ in spectra.values():
        ks = set(key(f_k).tolist())
        common = ks if common is None else (common & ks)
    if not common or len(common) < 2:
        return AdmittanceSumResult(
            note="segments share fewer than two common frequencies; "
                 "cannot form a plate admittance sum"
        )
    f0 = np.array(sorted(common), dtype=float)

    Y = np.zeros(len(f0), dtype=complex)
    used = 0
    for _, (f_k, Z_k) in spectra.items():
        lookup = dict(zip(key(f_k).tolist(), np.asarray(Z_k, complex)))
        picked = np.array([lookup[v] for v in np.round(f0, 6).tolist()])
        Y += segment_area_cm2 / picked
        used += 1
    Z_plate = cell_area_cm2 / Y

    result = AdmittanceSumResult(
        f=f0, Z_plate=Z_plate, n_segments=used,
        note=f"summed {used} segments over {len(f0)} common frequencies",
    )
    if reference is not None:
        f_ref, Z_ref = np.asarray(reference[0], float), np.asarray(reference[1], complex)
        order = np.argsort(f_ref)
        interp = np.interp(f0, f_ref[order], Z_ref[order].real) + 1j * np.interp(
            f0, f_ref[order], Z_ref[order].imag
        )
        inside = (f0 >= f_ref.min()) & (f0 <= f_ref.max())
        if inside.any():
            rel = np.abs(Z_plate[inside] - interp[inside]) / np.abs(interp[inside])
            result.Z_reference = interp
            result.median_relative_difference = float(np.median(rel))
    return result
