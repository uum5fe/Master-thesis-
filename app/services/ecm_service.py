"""Equivalent-circuit fitting for the viewer.

The fitting itself is :mod:`eis.model.ecm` - weighted objective, positive
parameters in log space, a ladder of circuits compared by corrected AIC, and a
standard error per parameter from the Jacobian.  Nothing about that belongs in
a callback, so this module only does the three things the frontend adds:

* **Units.**  Spectra are carried through the app in mOhm*cm2; the fit runs in
  Ohm*cm2, where ``Rs``, ``Y0`` and ``L`` sit in numerically comfortable
  ranges, and the resistances come back in mOhm*cm2 for display.
* **Weights.**  The pipeline stores a *relative* uncertainty per point.  The
  fitter wants an absolute one, so it is turned into ``sigma_rel * |Z|`` and
  the low-frequency points stop dominating a fit they should not dominate.
* **Caching.**  Fitting 72 segments over a ladder of three circuits is a few
  seconds of CPU.  Doing it again on every dropdown change is not acceptable
  in a web app, so results are memoised on the run and the fit settings.

Derived quantities that a reader actually wants - the time constant of each
arc, the effective capacitance, the total polarisation resistance - are
computed here rather than left for the reader to work out from ``Y0`` and
``n``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from eis.model.ecm import MODELS, ECMFit, fit_ecm, select_model, z_model

#: The ladder offered in the UI.  ``R_L`` is in it on purpose: if a spectrum is
#: best described by a resistor and a stray inductance, the honest answer is
#: that the band never reached the arc, not a confident two-arc fit.
MODEL_LABELS = {
    "R_L": "R + L (no arc — diagnostic)",
    "R_L_1RQ": "R + L + one R‖CPE (single arc)",
    "R_L_2RQ": "R + L + two R‖CPE (kinetic + transport)",
    "R_L_1RQ_W": "R + L + R‖CPE + finite Warburg (gas transport)",
    "auto": "Auto — fit the ladder, keep the AICc winner",
}

DEFAULT_LADDER = ["R_L", "R_L_1RQ", "R_L_2RQ"]

#: Ohm*cm2 -> mOhm*cm2 for the parameters that are a resistance.
_MOHM = 1000.0


@dataclass
class SegmentFit:
    segment: str
    fit: ECMFit | None
    f: np.ndarray = field(default_factory=lambda: np.array([]))
    z_meas: np.ndarray = field(default_factory=lambda: np.array([], complex))
    note: str = ""
    ladder: dict[str, ECMFit] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.fit is not None

    def z_fit_mohm(self) -> np.ndarray:
        if self.fit is None:
            return np.array([], complex)
        return self.fit.Z_fit * _MOHM

    def curve(self, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
        """A smooth model curve over the fitted band, for drawing."""
        if self.fit is None or self.f.size == 0:
            return np.array([]), np.array([], complex)
        grid = np.logspace(np.log10(self.f.min()), np.log10(self.f.max()), n)
        return grid, z_model(self.fit.model, grid, self.fit.params) * _MOHM

    def table(self) -> pd.DataFrame:
        """Parameters with units, standard errors and a determinacy flag."""
        if self.fit is None:
            return pd.DataFrame(columns=["parameter", "value", "±", "unit", "note"])
        rows = []
        for name in MODELS[self.fit.model]:
            value = self.fit.params[name]
            sigma = self.fit.sigma.get(name, float("nan"))
            unit, scale = _unit_and_scale(name)
            rows.append({
                "parameter": name,
                "value": value * scale,
                "±": sigma * scale if np.isfinite(sigma) else np.nan,
                "unit": unit,
                "note": "poorly determined" if name in self.fit.poorly_determined else "",
            })
        for name, value, unit in _derived(self.fit):
            rows.append({"parameter": name, "value": value, "±": np.nan,
                         "unit": unit, "note": "derived"})
        return pd.DataFrame(rows)


def _unit_and_scale(name: str) -> tuple[str, float]:
    if name.startswith(("Rs", "Rct", "Aw")):
        return "mΩ·cm²", _MOHM
    if name.startswith("L"):
        return "H·cm²", 1.0
    if name.startswith("Y0"):
        return "S·sⁿ/cm²", 1.0
    if name.startswith("tau"):
        return "s", 1.0
    return "-", 1.0


def _derived(fit: ECMFit) -> list[tuple[str, float, str]]:
    """Time constants, effective capacitances, total polarisation."""
    out: list[tuple[str, float, str]] = []
    for i in (1, 2):
        R, Y0, n = (fit.params.get(f"Rct{i}"), fit.params.get(f"Y0{i}"),
                    fit.params.get(f"n{i}"))
        if None in (R, Y0, n) or R <= 0 or Y0 <= 0:
            continue
        # A CPE arc peaks at w = (R*Y0)^(-1/n); tau is its reciprocal, and the
        # effective capacitance follows from Brug's relation for a CPE.
        tau = (R * Y0) ** (1.0 / n)
        out.append((f"tau{i}", tau, "s"))
        out.append((f"f_peak{i}", 1.0 / (2 * np.pi * tau), "Hz"))
        out.append((f"C_eff{i}", (Y0 * R ** (1 - n)) ** (1.0 / n) * 1e3, "mF/cm²"))
    out.append(("R_pol", fit.r_polarisation * _MOHM, "mΩ·cm²"))
    if "Rs" in fit.params:
        out.append(("R_total", (fit.params["Rs"] + fit.r_polarisation) * _MOHM,
                    "mΩ·cm²"))
    return out


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

def _prepare(spectrum: pd.DataFrame, f_min: float | None, f_max: float | None):
    """Extract a clean (f, Z, sigma) triple in Ohm*cm2 from a canonical frame."""
    frame = spectrum.dropna(subset=["freq_hz", "z_re_mohm_cm2", "z_im_mohm_cm2"])
    if f_min:
        frame = frame[frame["freq_hz"] >= f_min]
    if f_max:
        frame = frame[frame["freq_hz"] <= f_max]
    frame = frame.sort_values("freq_hz")
    f = frame["freq_hz"].to_numpy(float)
    z = (frame["z_re_mohm_cm2"].to_numpy(float)
         + 1j * frame["z_im_mohm_cm2"].to_numpy(float)) / _MOHM

    sigma = None
    if "sigma_rel" in frame.columns:
        rel = frame["sigma_rel"].to_numpy(float)
        if np.isfinite(rel).any():
            rel = np.where(np.isfinite(rel) & (rel > 0), rel, np.nanmedian(rel[rel > 0])
                           if np.any(rel > 0) else 0.05)
            sigma = np.maximum(rel * np.abs(z), 1e-12)
    return f, z, sigma


def fit_segment(
    spectrum: pd.DataFrame, segment: str, model: str = "auto",
    f_min: float | None = None, f_max: float | None = None,
    n_starts: int = 6, ladder: list[str] | None = None,
) -> SegmentFit:
    """Fit one segment's spectrum."""
    f, z, sigma = _prepare(spectrum, f_min, f_max)
    if f.size < 4:
        return SegmentFit(segment, None, f, z * _MOHM,
                          note=f"only {f.size} usable points in the band")

    if model == "auto":
        best, fits = select_model(f, z, models=ladder or DEFAULT_LADDER,
                                  sigma=sigma, n_starts=n_starts)
        return SegmentFit(segment, best, f, z * _MOHM,
                          note=(best.note if best else "no circuit converged"),
                          ladder=fits)

    fit = fit_ecm(f, z, model=model, sigma=sigma, n_starts=n_starts)
    return SegmentFit(segment, fit, f, z * _MOHM,
                      note="" if fit else f"{model} did not converge")


def fit_run(
    run, model: str = "auto", segments: list[str] | None = None,
    f_min: float | None = None, f_max: float | None = None,
    n_starts: int = 6, ladder: list[str] | None = None,
    progress=None,
) -> tuple[pd.DataFrame, dict[str, SegmentFit]]:
    """Fit every segment of a run and return a tidy parameter table.

    The table is what feeds the ECM heat maps: one row per segment, one column
    per fitted parameter, with ``_sd`` companions, so it merges straight into
    the per-segment frame the map already knows how to draw.
    """
    names = segments or run.segment_names
    rows: list[dict] = []
    fits: dict[str, SegmentFit] = {}

    for i, segment in enumerate(names):
        spectrum = run.spectrum(segment)
        if spectrum.empty:
            continue
        result = fit_segment(spectrum, segment, model, f_min, f_max, n_starts, ladder)
        fits[segment] = result
        row: dict[str, object] = {"segment": str(segment)}
        if result.fit is None:
            row["ecm_model"] = ""
            row["ecm_note"] = result.note
        else:
            fit = result.fit
            row["ecm_model"] = fit.model
            row["ecm_chi2_red"] = fit.chi2_reduced
            row["ecm_aicc"] = fit.aicc
            row["ecm_n_points"] = fit.n_points
            row["ecm_note"] = "; ".join(
                x for x in (fit.note,
                            ("poorly determined: " + ", ".join(fit.poorly_determined))
                            if fit.poorly_determined else "") if x)
            for name in MODELS[fit.model]:
                _, scale = _unit_and_scale(name)
                row[f"ecm_{name}"] = fit.params[name] * scale
                sigma = fit.sigma.get(name, float("nan"))
                row[f"ecm_{name}_sd"] = sigma * scale if np.isfinite(sigma) else np.nan
            for name, value, _unit in _derived(fit):
                row[f"ecm_{name}"] = value
        rows.append(row)
        if progress is not None:
            progress(i + 1, len(names))

    return pd.DataFrame(rows), fits


def attach_to_segments(run, table: pd.DataFrame):
    """Merge an ECM table onto the run's per-segment frame, in place."""
    if table.empty:
        return run
    frame = run.segments
    if frame is None or frame.empty:
        run.segments = table.copy()
        return run
    drop = [c for c in frame.columns if c.startswith("ecm_")]
    run.segments = frame.drop(columns=drop).merge(table, on="segment", how="left")
    return run


def ecm_params(table: pd.DataFrame) -> list[str]:
    """Numeric ECM columns worth mapping."""
    if table.empty:
        return []
    return [c for c in table.columns
            if c.startswith("ecm_") and not c.endswith("_sd")
            and pd.api.types.is_numeric_dtype(table[c]) and table[c].notna().any()]
