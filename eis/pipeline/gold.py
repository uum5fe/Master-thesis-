"""Gold tier: validated spectra -> physical parameters, maps and a report.

Gold is where the spectra stop being spectra and become the thing the thesis
is actually about: a spatial picture of what the cell is doing, with every
number carrying an uncertainty and a provenance stamp.

Three jobs:

* **Equivalent-circuit fitting** over a ladder of models selected by corrected
  AIC, weighted by the uncertainty Silver produced.  Every segment that is
  active is fitted, including the ones flagged as poor - the fit is labelled,
  not withheld.
* **Scalar extraction.**  One code path turns any per-segment quantity, static
  or frequency-resolved, into a map: ``rs_hf``, ``rp``, ``quality``,
  ``coherence``, ``z_mod@f``, ``phase@f`` and the rest.  Adding a new map means
  adding a key, not a plotting function.
* **Publication.**  Tables, static figures, and a single self-contained HTML
  page where clicking a segment on the plate shows its spectrum.

Nothing is hidden here either.  A segment whose fit is poorly determined is
drawn hatched, not dropped; the map's colour scale is computed from the
segments that are trustworthy so one broken channel cannot flatten it, but the
broken channel is still on the plate where a reader can see it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from eis.calibrate import TemperatureField
from eis.model.ecm import ECMFit, select_model
from eis.pipeline.bronze import SyncReport
from eis.pipeline.config import PipelineConfig
from eis.pipeline.silver import (
    SegmentSpectrum, SilverCondition, common_mode_frame, impedance_frame,
    segment_frame,
)
from eis.pipeline.utils import nearest_index, write_table
from eis.validate import AdmittanceSumResult

#: Frequency-resolved map keys are written ``name@f`` and read at the frequency
#: the caller asks for.
FREQUENCY_KEYS = {"z_mod@f", "z_real@f", "neg_z_imag@f", "phase@f", "coherence@f"}


# ---------------------------------------------------------------------------
# The finished condition
# ---------------------------------------------------------------------------

@dataclass
class ConditionResult:
    """One operating point, all three tiers folded together."""

    measurement_id: str
    condition: str
    fs: float
    duration_s: float
    sync: SyncReport
    temperature: TemperatureField
    segments: dict[int, SegmentSpectrum] = field(default_factory=dict)
    frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    common_mode: dict = field(default_factory=dict)
    crosstalk_change: float = 0.0
    plate: AdmittanceSumResult | None = None
    tone_check: str = ""
    provenance: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    # -- views -----------------------------------------------------------
    @property
    def active_segments(self) -> list[int]:
        return [s for s, r in sorted(self.segments.items()) if r.active]

    @property
    def rejected(self) -> dict[int, str]:
        """Segments that produced nothing usable at all.

        Kept as a mapping for continuity with the older interface, but it is
        now a *view* over the classified segments rather than a bin that things
        were thrown into: a segment appears here only when it has fewer usable
        points than the activity threshold, and it is still present in
        :attr:`segments` with its status, flags and note intact.
        """
        return {
            s: (r.note or r.status)
            for s, r in sorted(self.segments.items()) if not r.active
        }

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.segments.values():
            counts[r.status] = counts.get(r.status, 0) + 1
        return counts

    # -- tables ----------------------------------------------------------
    def impedance_frame(self, segment_area_cm2: float) -> pd.DataFrame:
        return impedance_frame(
            _as_silver(self), segment_area_cm2, self.provenance
        )

    def segment_frame(self, segment_area_cm2: float) -> pd.DataFrame:
        frame = segment_frame(_as_silver(self), segment_area_cm2, self.provenance)
        return _append_ecm_columns(frame, self, segment_area_cm2)

    def common_mode_frame(self) -> pd.DataFrame:
        return common_mode_frame(_as_silver(self))


def _as_silver(result: ConditionResult) -> SilverCondition:
    """Adapt a finished condition back to the shape the Silver writers expect."""
    return SilverCondition(
        measurement_id=result.measurement_id, condition=result.condition,
        fs=result.fs, duration_s=result.duration_s, sync=result.sync,
        segments=result.segments, frequencies=result.frequencies,
        common_mode=result.common_mode, plate=result.plate,
        tone_check=result.tone_check, notes=result.notes,
    )


# ---------------------------------------------------------------------------
# Equivalent-circuit fitting
# ---------------------------------------------------------------------------

def fit_models(result: ConditionResult, cfg: PipelineConfig, log=print) -> None:
    """Fit the circuit ladder to every segment worth fitting."""
    fitted = 0
    attempted = 0
    for record in result.segments.values():
        if not record.active:
            continue
        if not cfg.model.fit_all_active_segments and record.status != "ok":
            continue
        spectrum = record.spectrum
        if len(spectrum.f) < 5:
            continue
        attempted += 1
        best, allfits = select_model(
            spectrum.f, spectrum.Z, cfg.model.models, spectrum.sigma_random,
            n_starts=cfg.model.n_starts, max_nfev=cfg.model.max_nfev,
            max_relative_sigma=cfg.model.max_relative_sigma,
        )
        record.ecm, record.ecm_all = best, allfits
        fitted += best is not None
    chosen: dict[str, int] = {}
    for record in result.segments.values():
        if record.ecm:
            chosen[record.ecm.model] = chosen.get(record.ecm.model, 0) + 1
    log(f"  ECM: {fitted}/{attempted} fitted; selected {chosen}")


def _append_ecm_columns(
    frame: pd.DataFrame, result: ConditionResult, segment_area_cm2: float
) -> pd.DataFrame:
    """Add the circuit parameters, area-normalised where that makes sense."""
    if frame.empty:
        return frame
    asr = segment_area_cm2 * 1e3
    extra: dict[int, dict] = {}
    for segment, record in result.segments.items():
        fit: ECMFit | None = record.ecm
        if fit is None:
            continue
        row: dict = {
            "ecm_model": fit.model,
            "ecm_chi2_reduced": fit.chi2_reduced,
            "ecm_aicc": fit.aicc,
            "ecm_poorly_determined": ",".join(fit.poorly_determined),
            "ecm_rp_mohm_cm2": fit.r_polarisation * asr,
        }
        for name, value in fit.params.items():
            scale = asr if name.startswith(("Rs", "Rct", "Aw")) else 1.0
            suffix = "_mohm_cm2" if scale != 1.0 else ""
            row[f"ecm_{name}{suffix}"] = value * scale
            row[f"ecm_{name}{suffix}_sigma"] = fit.sigma.get(name, np.nan) * scale
        extra[segment] = row
    if not extra:
        return frame
    addition = pd.DataFrame.from_dict(extra, orient="index")
    addition.index.name = "segment"
    return frame.merge(addition.reset_index(), on="segment", how="left")


# ---------------------------------------------------------------------------
# Scalar extraction: one code path, N maps
# ---------------------------------------------------------------------------

def scalar_map(
    result: ConditionResult, key: str, segment_area_cm2: float,
    frequency_hz: float | None = None, active_only: bool = False,
) -> dict[int, float]:
    """Per-segment value of ``key``, for a plate map or a table.

    Frequency-resolved keys end in ``@f`` and are read at the point of the
    shared grid nearest ``frequency_hz``.  Because every segment kept every
    frequency it measured - the coherence gate marks rather than deletes -
    that lookup is well defined for all of them, which is what makes an
    animated map over frequency possible.
    """
    asr = segment_area_cm2 * 1e3
    values: dict[int, float] = {}
    for segment, r in sorted(result.segments.items()):
        if active_only and not r.active:
            continue
        value: float
        if key in FREQUENCY_KEYS:
            spectrum = r.spectrum_all
            if len(spectrum.f) == 0 or frequency_hz is None:
                continue
            i = nearest_index(spectrum.f, frequency_hz)
            z = spectrum.Z[i]
            value = {
                "z_mod@f": abs(z) * asr,
                "z_real@f": z.real * asr,
                "neg_z_imag@f": -z.imag * asr,
                "phase@f": float(np.degrees(np.angle(z))),
                "coherence@f": float(spectrum.coherence[i]),
            }[key]
        elif key == "rs_hf":
            value = r.hfr.rs_ohm * asr
        elif key == "rs_hf_sigma":
            value = r.hfr.rs_sigma_ohm * asr
        elif key == "hf_inductance_nh":
            value = r.hfr.l_h * 1e9
        elif key == "rp":
            value = r.ecm.r_polarisation * asr if r.ecm else float("nan")
        elif key == "ecm_rs":
            value = r.ecm.params["Rs"] * asr if r.ecm else float("nan")
        elif key == "chi2_reduced":
            value = r.ecm.chi2_reduced if r.ecm else float("nan")
        elif key == "quality":
            value = r.quality
        elif key == "coherence":
            value = r.median_coherence
        elif key == "sigma_rel":
            value = r.median_sigma_rel
        elif key == "kk_max_residual_pct":
            value = r.kk.max_residual_pct if r.kk else float("nan")
        elif key == "temperature_c":
            value = r.temperature_c
        elif key == "timing_sigma_ns":
            value = r.timing_sigma_s * 1e9
        else:
            raise KeyError(f"unknown map key {key!r}")
        if np.isfinite(value):
            values[segment] = float(value)
    return values


def map_label(key: str) -> tuple[str, str]:
    """``(title, unit)`` for a map key."""
    return {
        "rs_hf": ("high-frequency resistance", r"$R_s$ [m$\Omega\cdot$cm$^2$]"),
        "rs_hf_sigma": ("uncertainty on Rs", r"$\sigma(R_s)$ [m$\Omega\cdot$cm$^2$]"),
        "hf_inductance_nh": ("fitted series inductance", "L [nH]"),
        "rp": ("polarisation resistance", r"$R_p$ [m$\Omega\cdot$cm$^2$]"),
        "ecm_rs": ("ECM ohmic resistance", r"$R_s$ [m$\Omega\cdot$cm$^2$]"),
        "chi2_reduced": ("fit quality", r"$\chi^2_{red}$"),
        "quality": ("segment quality score", "0 - 1"),
        "coherence": ("median coherence", r"$\gamma^2$"),
        "sigma_rel": ("median relative uncertainty", r"$\sigma/|Z|$"),
        "kk_max_residual_pct": ("Kramers-Kronig residual", "max |residual| [%]"),
        "temperature_c": ("segment temperature", "T [degC]"),
        "timing_sigma_ns": ("timing uncertainty", r"$\sigma_\tau$ [ns]"),
        "z_mod@f": ("|Z| at f", r"$|Z|$ [m$\Omega\cdot$cm$^2$]"),
        "z_real@f": ("Z' at f", r"$Z'$ [m$\Omega\cdot$cm$^2$]"),
        "neg_z_imag@f": ("-Z'' at f", r"$-Z''$ [m$\Omega\cdot$cm$^2$]"),
        "phase@f": ("phase at f", "phase [deg]"),
        "coherence@f": ("coherence at f", r"$\gamma^2$"),
    }.get(key, (key, ""))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_gold(
    silver: SilverCondition,
    cfg: PipelineConfig,
    temperature: TemperatureField,
    provenance: dict[str, str],
    log=print,
) -> ConditionResult:
    """Fit circuits and assemble the finished condition."""
    result = ConditionResult(
        measurement_id=silver.measurement_id, condition=silver.condition,
        fs=silver.fs, duration_s=silver.duration_s, sync=silver.sync,
        temperature=temperature, segments=silver.segments,
        frequencies=silver.frequencies, common_mode=silver.common_mode,
        crosstalk_change=silver.crosstalk_change, plate=silver.plate,
        tone_check=silver.tone_check, provenance=provenance,
        notes=list(silver.notes),
    )
    if cfg.model.run_ecm:
        fit_models(result, cfg, log)
    return result


# ---------------------------------------------------------------------------
# Publication
# ---------------------------------------------------------------------------

def write_outputs(
    cfg: PipelineConfig, results: dict[str, ConditionResult], log=print
) -> dict[str, Path]:
    """Persist impedance, per-segment, synchronisation and correction tables."""
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    area = cfg.geometry.segment_area_cm2

    tables = {
        "impedance": pd.concat(
            [r.impedance_frame(area) for r in results.values()], ignore_index=True
        ),
        "segments": pd.concat(
            [r.segment_frame(area) for r in results.values()], ignore_index=True
        ),
        "sync": pd.concat(
            [r.sync.to_frame().assign(condition=c) for c, r in results.items()],
            ignore_index=True,
        ),
    }
    common = [r.common_mode_frame() for r in results.values()]
    common = [f for f in common if not f.empty]
    if common:
        tables["hf_common_mode"] = pd.concat(common, ignore_index=True)

    paths: dict[str, Path] = {}
    for name, frame in tables.items():
        path = write_table(frame, out / f"{name}.parquet")
        paths[name] = path
        log(f"  wrote {path}  ({len(frame):,} rows)")

    config_path = out / "config.yaml"
    try:
        cfg.save(config_path)
        paths["config"] = config_path
    except Exception:
        pass
    return paths


def write_report(
    cfg: PipelineConfig, results: dict[str, ConditionResult], log=print
) -> list[Path]:
    """Static figures plus the interactive plate view."""
    written: list[Path] = []
    out = Path(cfg.output_dir)

    if cfg.report.write_figures:
        from eis.viz import write_report_figures

        for condition, result in results.items():
            written += write_report_figures(
                result, cfg, out / "figures" / condition
            )

    if cfg.report.write_dashboard and results:
        from eis.dashboard import write_dashboard

        path = write_dashboard(results, cfg, out / "plate_dashboard.html")
        if path is not None:
            written.append(path)
    for path in written:
        log(f"  figure: {path}")
    return written


def summarise(result: ConditionResult, cfg: PipelineConfig) -> str:
    """One paragraph a reader can check the run against."""
    counts = result.status_counts()
    rs = scalar_map(result, "rs_hf", cfg.geometry.segment_area_cm2, active_only=True)
    lines = [
        f"{result.measurement_id} / {result.condition}: "
        f"{len(result.active_segments)}/{len(result.segments)} segments active "
        f"({', '.join(f'{v} {k}' for k, v in sorted(counts.items()))})",
    ]
    if rs:
        values = np.array(list(rs.values()))
        lines.append(
            f"  Rs across the plate: {values.min():.2f} - {values.max():.2f} "
            f"mOhm*cm2 (median {np.median(values):.2f}, "
            f"spread {values.std() / max(np.mean(values), 1e-12):.1%})"
        )
    applied = [c for c, r in result.common_mode.items() if r.applied]
    if applied:
        delays = [result.common_mode[c].delay_s * 1e9 for c in applied]
        lines.append(
            f"  common-mode delay removed on cards {sorted(applied)}: "
            f"{', '.join(f'{d:+.0f} ns' for d in delays)}"
        )
    if result.crosstalk_change:
        lines.append(
            f"  in-plane crosstalk deconvolution changed the admittances by a "
            f"median {result.crosstalk_change:.1%}"
        )
    if result.plate is not None and result.plate.median_relative_difference is not None:
        lines.append(
            f"  plate admittance sum agrees with the reference instrument to "
            f"{result.plate.median_relative_difference:.1%}"
        )
    return "\n".join(lines)
