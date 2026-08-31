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
that minimises the KK residual recovers a residual timing error.  Two
conditions have to hold for that to mean anything, and both live in
:func:`eis.hf.pooled_kk_delay`, which is where the scan is implemented:

* the Voigt basis must be fitted **without** its ``j*omega*L`` term, which
  otherwise absorbs a small delay exactly and leaves the residual blind to it;
* the search must be **bounded by the measured skew**, because over a finite
  band a delay is degenerate with an inductance (``L = -Rs*dt`` to first
  order) and an unbounded search eats the cell's real inductance.
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
    #: Largest residual expressed in units of that point's own uncertainty.
    #: This, not the percentage, is what makes the verdict meaningful on a
    #: segment whose high-frequency points are honestly imprecise.
    max_normalised_residual: float = float("nan")
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
    with_inductance: bool = True, max_inductance: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted linear least squares.  Returns ``(coeffs, taus, Z_fit)``.

    ``max_inductance`` box-constrains the ``jwL`` coefficient.  It exists for
    the delay scan in :func:`eis.hf.pooled_kk_delay` and it is the thing that
    makes that scan mean anything: an unconstrained ``L`` absorbs a delay
    exactly, so the residual cannot see one, while dropping the term entirely
    makes the fit absorb the cell's *real* inductance into the delay instead.
    Bounding ``L`` to what the wiring can physically be is what separates the
    two.  With one box constraint the projection is exact - fit free, and if
    the bound is violated, pin the coefficient there and refit the rest.
    """
    taus = np.logspace(
        np.log10(1.0 / (2 * np.pi * f.max())),
        np.log10(1.0 / (2 * np.pi * max(f.min(), 1e-6))),
        n_elements,
    )
    A = _voigt_design(f, taus, with_inductance)
    b = np.concatenate([Z.real, Z.imag])
    w = np.concatenate([weights, weights])
    coeffs, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)

    if with_inductance and max_inductance is not None and abs(coeffs[1]) > max_inductance:
        pinned = float(np.sign(coeffs[1]) * max_inductance)
        keep = [i for i in range(A.shape[1]) if i != 1]
        reduced = A[:, keep]
        residual_b = b - A[:, 1] * pinned
        sub, *_ = np.linalg.lstsq(
            reduced * w[:, None], residual_b * w, rcond=None
        )
        coeffs = np.insert(sub, 1, pinned)

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
    max_sigma: float = 3.0, sigma_total: np.ndarray | None = None,
    max_inductance: float | None = None,
) -> KKResult:
    """Linear Kramers-Kronig test with automatic model order.

    The verdict uses **both** an absolute and a statistical criterion, and a
    point has to fail both to count as a violation:

    * ``|residual| > max_residual_pct`` - the deviation is large enough to
      matter physically;
    * ``|residual| > max_sigma * sigma`` - and large enough that the noise on
      that point cannot explain it.

    The second half is what makes the test usable on real data.  A segment
    whose card was timed from a proxy carries an honest 20 % phase uncertainty
    at the top of the band; judging its 2 % residual against a flat 1 %
    threshold declares the data non-causal when all it is, is imprecise.
    Conversely a 1.2 % residual on a point known to 0.05 % is a genuine
    systematic error and still fails.  ``max_sigma = 0`` restores the plain
    absolute test.

    ``sigma`` weights the fit and should be the **random** uncertainty only.
    ``sigma_total`` judges the residual and should include the systematic terms
    as well; it defaults to ``sigma``.
    """
    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    order = np.argsort(f)
    f, Z = f[order], Z[order]
    if sigma is not None:
        sigma = np.asarray(sigma, float)[order]
    if sigma_total is not None:
        sigma_total = np.asarray(sigma_total, float)[order]
    verdict_sigma = sigma if sigma_total is None else sigma_total

    if len(f) < 6:
        return KKResult(
            0, 0.0, np.array([]), np.array([]), 100.0, 100.0, Z.copy(), False,
            "insufficient_points", f"only {len(f)} points; KK test needs >= 6",
        )

    # Inverse-variance weights where an uncertainty is available, otherwise
    # proportional weighting (the standard choice for impedance data).
    weights = 1.0 / np.maximum(sigma, 1e-15) if sigma is not None else 1.0 / np.abs(Z)

    # --- model order ------------------------------------------------------
    # The mu-criterion is a test for over-fitting: mu falls when the fit starts
    # buying accuracy with oscillating positive and negative resistances.  It is
    # *not* monotonic in the number of elements - on real data it alternates,
    # because an even element count can place a pair symmetrically where an odd
    # one cannot - so stopping at the first value below the target selects a
    # badly under-fitted model.  On a clean single-arc spectrum that rule
    # returns four elements and a 13 % model residual, which then gets reported
    # as the data's Kramers-Kronig residual.
    #
    # So every order is evaluated, the ones that show over-fitting are
    # discarded, and among the rest the smallest order that gets within a
    # factor of the best achievable residual wins.  Parsimony among the models
    # that fit, rather than the first model that stops improving.
    max_m = int(min(max_elements, max(3, len(f) // 2 - 1)))
    trials = []
    for m in range(3, max_m + 1):
        coeffs, _, Z_fit = _fit_voigt(
            f, Z, m, weights, with_inductance, max_inductance
        )
        misfit = float(np.sqrt(np.mean(
            (np.concatenate([Z.real - Z_fit.real, Z.imag - Z_fit.imag])
             / np.concatenate([np.abs(Z), np.abs(Z)])) ** 2
        )))
        trials.append((m, _mu(coeffs, with_inductance), misfit, Z_fit))

    pool = [t for t in trials if t[1] >= mu_target] or trials
    best_misfit = min(t[2] for t in pool)
    m, mu, _, Z_fit = next(
        t for t in pool if t[2] <= max(2.0 * best_misfit, 1e-15)
    )
    scale = np.abs(Z)
    dre = (Z.real - Z_fit.real) / scale
    dim = (Z.imag - Z_fit.imag) / scale
    residual = np.abs(np.concatenate([dre, dim]))
    max_pct = float(np.max(residual)) * 100.0
    rms_pct = float(np.sqrt(np.mean(residual ** 2))) * 100.0

    # Relative uncertainty per point; the same value bounds both parts, since
    # sigma_|Z|/|Z| and sigma_phi[rad] are equal for a transfer-function
    # estimate (Bendat-Piersol).
    if verdict_sigma is not None:
        sigma_rel = np.concatenate([verdict_sigma, verdict_sigma]) / np.concatenate(
            [scale, scale]
        )
        normalised = residual / np.maximum(sigma_rel, 1e-12)
        max_normalised = float(np.max(normalised))
        violation = (residual > max_residual_pct / 100.0) & (
            normalised > max_sigma if max_sigma > 0 else True
        )
    else:
        max_normalised = float("nan")
        violation = residual > max_residual_pct / 100.0

    return KKResult(
        n_elements=m, mu=float(mu), residual_real=dre, residual_imag=dim,
        max_residual_pct=max_pct, rms_residual_pct=rms_pct, Z_fit=Z_fit,
        passed=bool(not violation.any()),
        shape_class=_classify_residual(f, dre, dim),
        max_normalised_residual=max_normalised,
    )


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
