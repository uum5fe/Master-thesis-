"""Equivalent-circuit fitting with uncertainties and model selection.

Three things separate this from a bare ``least_squares`` call:

* **The objective is weighted** by the per-point uncertainty produced in
  :mod:`eis.spectra`.  Unweighted fitting lets the noisiest points - usually
  the low-frequency ones - dominate, which is exactly backwards.
* **Strictly positive parameters are fitted in log space.**  ``Rs``, ``Rct``
  and ``Y0`` span decades; box constraints on linear parameters leave the
  problem badly conditioned and the optimiser stalls on the boundary.
* **The circuit is selected, not assumed.**  A ladder of models is fitted and
  compared by corrected AIC, so complexity has to be earned.  A two-arc model
  forced onto a spectrum with gas-transport limitation returns confident,
  meaningless numbers; the ladder makes the inadequacy visible.

Every fitted parameter is returned with a standard error from the Jacobian at
the optimum, and a parameter whose relative error is too large is flagged
rather than plotted as if it were known.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

#: model name -> parameter names, in order
MODELS: dict[str, list[str]] = {
    "R_L":       ["Rs", "L"],
    "R_L_1RQ":   ["Rs", "L", "Rct1", "Y01", "n1"],
    "R_L_2RQ":   ["Rs", "L", "Rct1", "Y01", "n1", "Rct2", "Y02", "n2"],
    "R_L_1RQ_W": ["Rs", "L", "Rct1", "Y01", "n1", "Aw", "tau_w"],
}

#: parameters fitted in log space (strictly positive, wide dynamic range)
_LOG_PARAMS = {"Rs", "L", "Rct1", "Rct2", "Y01", "Y02", "Aw", "tau_w"}


def _zarc(w: np.ndarray, R: float, Y0: float, n: float) -> np.ndarray:
    """R in parallel with a constant-phase element."""
    return R / (1.0 + (1j * w) ** n * Y0 * R)


def _warburg_finite(w: np.ndarray, Aw: float, tau: float) -> np.ndarray:
    """Finite-length (transmissive) Warburg element for gas transport."""
    s = np.sqrt(1j * w * tau)
    return Aw * np.tanh(s) / np.where(np.abs(s) < 1e-12, 1e-12, s)


def z_model(model: str, f: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """Evaluate an equivalent-circuit model."""
    w = 2.0 * np.pi * np.asarray(f, float)
    Z = params["Rs"] + 1j * w * params["L"]
    if model in ("R_L_1RQ", "R_L_2RQ", "R_L_1RQ_W"):
        Z = Z + _zarc(w, params["Rct1"], params["Y01"], params["n1"])
    if model == "R_L_2RQ":
        Z = Z + _zarc(w, params["Rct2"], params["Y02"], params["n2"])
    if model == "R_L_1RQ_W":
        Z = Z + _warburg_finite(w, params["Aw"], params["tau_w"])
    return Z


# ---------------------------------------------------------------------------
# Parameter vector <-> dict
# ---------------------------------------------------------------------------

def _to_vector(model: str, params: dict[str, float]) -> np.ndarray:
    out = []
    for name in MODELS[model]:
        value = params[name]
        out.append(np.log(max(value, 1e-30)) if name in _LOG_PARAMS else value)
    return np.asarray(out, float)


def _to_params(model: str, vector: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, value in zip(MODELS[model], vector):
        out[name] = float(np.exp(value)) if name in _LOG_PARAMS else float(value)
    return out


def _bounds(model: str) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = [], []
    for name in MODELS[model]:
        if name in _LOG_PARAMS:
            lo.append(np.log(1e-14)); hi.append(np.log(1e6))
        else:                                   # CPE exponents
            lo.append(0.3); hi.append(1.0)
    return np.asarray(lo), np.asarray(hi)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class ECMFit:
    model: str
    params: dict[str, float]
    sigma: dict[str, float]
    chi2: float
    chi2_reduced: float
    aicc: float
    n_points: int
    success: bool
    Z_fit: np.ndarray = field(default_factory=lambda: np.array([], complex))
    poorly_determined: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def r_polarisation(self) -> float:
        """Total polarisation resistance (everything except Rs and L)."""
        return float(
            sum(v for k, v in self.params.items() if k.startswith(("Rct", "Aw")))
        )

    def relative_sigma(self, name: str) -> float:
        value = abs(self.params.get(name, 0.0))
        if value <= 0:
            return float("inf")
        return self.sigma.get(name, float("inf")) / value


# ---------------------------------------------------------------------------
# Initial guesses
# ---------------------------------------------------------------------------

def _initial_guess(model: str, f: np.ndarray, Z: np.ndarray) -> dict[str, float]:
    """Physics-informed start values read off the spectrum."""
    order = np.argsort(f)
    f_s, Z_s = f[order], Z[order]

    n_hf = max(2, len(f_s) // 6)
    Rs0 = max(float(np.median(Z_s.real[-n_hf:])), 1e-9)
    R_total = max(float(Z_s.real.max() - Rs0), Rs0 * 0.05)

    # Peak of -Im(Z) marks the dominant time constant: w_peak = 1/(R*Y0)^(1/n).
    neg_im = -Z_s.imag
    peak = int(np.argmax(neg_im)) if np.any(neg_im > 0) else len(f_s) // 2
    f_peak = max(float(f_s[peak]), f_s.min())

    guess: dict[str, float] = {"Rs": Rs0, "L": 1e-8}
    if model in ("R_L_1RQ", "R_L_1RQ_W"):
        guess |= {"Rct1": R_total, "n1": 0.85,
                  "Y01": 1.0 / (R_total * (2 * np.pi * f_peak) ** 0.85)}
    if model == "R_L_2RQ":
        # Split the arc: a fast one near the top of the band, a slow one at the
        # observed peak.
        f_fast = max(f_peak * 8.0, f_s.max() / 4.0)
        guess |= {
            "Rct1": 0.3 * R_total, "n1": 0.85,
            "Y01": 1.0 / (0.3 * R_total * (2 * np.pi * f_fast) ** 0.85),
            "Rct2": 0.7 * R_total, "n2": 0.80,
            "Y02": 1.0 / (0.7 * R_total * (2 * np.pi * f_peak) ** 0.80),
        }
    if model == "R_L_1RQ_W":
        guess |= {"Aw": 0.3 * R_total, "tau_w": 1.0 / (2 * np.pi * max(f_s.min(), 1e-3))}
    return guess


def _perturb(guess: dict[str, float], rng: np.random.Generator) -> dict[str, float]:
    out = {}
    for name, value in guess.items():
        if name in _LOG_PARAMS:
            out[name] = float(value * np.exp(rng.normal(0.0, 0.7)))
        else:
            out[name] = float(np.clip(value + rng.normal(0.0, 0.06), 0.35, 0.98))
    return out


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_ecm(
    f: np.ndarray, Z: np.ndarray, model: str = "R_L_2RQ",
    sigma: np.ndarray | None = None, n_starts: int = 6, max_nfev: int = 5000,
    max_relative_sigma: float = 0.30, seed: int = 0,
) -> ECMFit | None:
    """Weighted, multi-start fit of one equivalent-circuit model."""
    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    if len(f) < len(MODELS[model]):
        return None

    # Inverse-variance weights, or proportional weighting as the fallback.
    w = 1.0 / np.maximum(sigma, 1e-15) if sigma is not None else 1.0 / np.abs(Z)
    weights = np.concatenate([w, w])
    lo, hi = _bounds(model)

    def residuals(vector: np.ndarray) -> np.ndarray:
        p = _to_params(model, vector)
        Zf = z_model(model, f, p)
        return np.concatenate([Zf.real - Z.real, Zf.imag - Z.imag]) * weights

    base = _initial_guess(model, f, Z)
    rng = np.random.default_rng(seed)
    best = None
    for attempt in range(max(1, n_starts)):
        start = base if attempt == 0 else _perturb(base, rng)
        try:
            x0 = np.clip(_to_vector(model, start), lo, hi)
            result = least_squares(
                residuals, x0, bounds=(lo, hi), method="trf", max_nfev=max_nfev
            )
        except Exception:
            continue
        if best is None or result.cost < best.cost:
            best = result
    if best is None:
        return None

    params = _to_params(model, best.x)
    Z_fit = z_model(model, f, params)
    chi2 = float(2.0 * best.cost)
    n_res, n_par = 2 * len(f), len(MODELS[model])
    dof = max(n_res - n_par, 1)
    chi2_red = chi2 / dof

    # Covariance of the *log* parameters from the Jacobian, scaled by the
    # reduced chi-square so that an imperfect noise model still gives a
    # sensible error bar.
    sigmas: dict[str, float] = {}
    try:
        J = best.jac
        cov = np.linalg.inv(J.T @ J) * chi2_red
        diag = np.clip(np.diag(cov), 0.0, None)
        for name, value, var in zip(MODELS[model], best.x, diag):
            s = float(np.sqrt(var))
            # d(exp(u)) = exp(u) du : convert a log-space error to linear.
            sigmas[name] = float(np.exp(value) * s) if name in _LOG_PARAMS else s
    except np.linalg.LinAlgError:
        sigmas = {name: float("inf") for name in MODELS[model]}

    aic = chi2 + 2 * n_par
    aicc = aic + (2 * n_par * (n_par + 1)) / max(n_res - n_par - 1, 1)

    fit = ECMFit(
        model=model, params=params, sigma=sigmas, chi2=chi2,
        chi2_reduced=chi2_red, aicc=float(aicc), n_points=len(f),
        success=bool(best.success), Z_fit=Z_fit,
    )
    fit.poorly_determined = [
        name for name in MODELS[model]
        if fit.relative_sigma(name) > max_relative_sigma and name not in ("n1", "n2")
    ]
    return fit


def select_model(
    f: np.ndarray, Z: np.ndarray, models: list[str] | None = None,
    sigma: np.ndarray | None = None, **kwargs,
) -> tuple[ECMFit | None, dict[str, ECMFit]]:
    """Fit a ladder of circuits and return the one preferred by corrected AIC."""
    candidates = models or ["R_L", "R_L_1RQ", "R_L_2RQ"]
    fits: dict[str, ECMFit] = {}
    for name in candidates:
        if name not in MODELS:
            continue
        fit = fit_ecm(f, Z, model=name, sigma=sigma, **kwargs)
        if fit is not None and np.isfinite(fit.aicc):
            fits[name] = fit
    if not fits:
        return None, {}
    best_name = min(fits, key=lambda k: fits[k].aicc)
    best = fits[best_name]
    if len(fits) > 1:
        runner_up = sorted(fits.values(), key=lambda x: x.aicc)[1]
        best.note = (
            f"selected by AICc over {len(fits)} models; "
            f"delta_AICc to {runner_up.model} = {runner_up.aicc - best.aicc:.1f}"
        )
    return best, fits
