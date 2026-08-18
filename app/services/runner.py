"""Turning raw FAMOS recordings into results, from inside the app.

Selecting a ``.DAT`` recording gives you files, not spectra: bronze/silver/gold
still has to run.  That work is minutes of CPU over gigabytes of raw data, so
it is a background job with progress, never a blocking callback, and it is
disabled by default (``EIS_ALLOW_INLINE_PIPELINE``) because a Databricks App
container is sized for serving a UI, not for signal processing.

Where the app does not have the compute, the same selection should be handed
to a Databricks Job on a proper cluster; the results land in the results root
and appear in the picker after a Refresh.  That path is described in
``docs/FRONTEND.md`` - this module deliberately does not smuggle heavy compute
into the web tier without the operator turning it on.
"""

from __future__ import annotations

from pathlib import Path

from app.data.sources import RunRef
from app.plates.registry import PlateGeometry
from app.settings import SETTINGS, Settings


class PipelineUnavailable(RuntimeError):
    """Raised when an inline run is asked for but not permitted or possible."""


def output_dir(ref: RunRef, settings: Settings = SETTINGS) -> Path:
    return Path(settings.scratch_results_root) / ref.measurement_id / ref.condition


def preflight(ref: RunRef, settings: Settings = SETTINGS) -> list[str]:
    """Everything that would stop a run, reported before starting one."""
    problems: list[str] = []
    if not settings.allow_inline_pipeline:
        problems.append(
            "inline pipeline runs are disabled; set EIS_ALLOW_INLINE_PIPELINE=1 "
            "on a deployment that has the compute, or run the pipeline as a "
            "Databricks Job writing into EIS_RESULTS_ROOT")
    if not ref.files:
        problems.append("no .DAT files in this selection")
    if not settings.curr_cal:
        problems.append(
            "EIS_CURR_CAL is not set; the per-segment shunt calibration is the "
            "only absolute scale in the chain, so impedances would be in shunt "
            "volts per amp rather than in ohms")
    elif not Path(settings.curr_cal).exists():
        problems.append(f"shunt calibration not found: {settings.curr_cal}")
    return problems


def run_famos(progress, ref: RunRef, geom: PlateGeometry,
              settings: Settings = SETTINGS,
              conditions: list[str] | None = None) -> str:
    """Run the packaged pipeline over one FAMOS selection.

    Signature matches what :class:`app.services.store.JobTable` expects:
    ``progress(done, total, message)`` first, everything else after.
    """
    problems = preflight(ref, settings)
    if problems:
        raise PipelineUnavailable("; ".join(problems))

    from eis.config import PipelineConfig
    from eis.pipeline import run_measurement

    out = output_dir(ref, settings)
    out.mkdir(parents=True, exist_ok=True)

    progress(0, 3, "building configuration")
    cfg = PipelineConfig()
    cfg.measurement_id = ref.measurement_id
    cfg.conditions = conditions or [ref.condition]
    cfg.raw_dir = ref.path
    cfg.output_dir = str(out)
    cfg.calibration.shunt_csv = settings.curr_cal
    cfg.calibration.temperature_csv = settings.temp_cal

    # The plate generation decides the areas the pipeline scales by. Feeding
    # the geometry in here is what keeps a Gen-2 plate from being silently
    # processed with Gen-1 areas.
    areas = geom.areas()
    cfg.geometry.n_segments = geom.n_segments
    cfg.geometry.cell_area_cm2 = geom.cell_area_cm2
    cfg.geometry.plate_w_mm = geom.plate_w_mm
    cfg.geometry.plate_h_mm = geom.plate_h_mm
    cfg.geometry.segment_area_cm2 = sum(areas.values()) / max(len(areas), 1)
    cfg.geometry.segment_coords = {
        int(name): (
            geom.centroid_mm(name)[0] / 10.0, geom.centroid_mm(name)[1] / 10.0,
            (geom.bounds_mm(name)[2] - geom.bounds_mm(name)[0]) / 20.0,
            (geom.bounds_mm(name)[3] - geom.bounds_mm(name)[1]) / 20.0,
        )
        for name in geom.order() if str(name).isdigit()
    }

    progress(1, 3, f"processing {len(ref.files)} card file(s)")
    run_measurement(cfg, write=True, verbose=False)

    progress(2, 3, "writing results")
    cfg.save(out / "config.yaml")
    progress(3, 3, f"results written to {out}")
    return str(out)
