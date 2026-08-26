r"""Running the bronze/silver/gold pipeline over raw FAMOS recordings.

The pipeline lives in ``local_eis/`` — the same modules that ran on Databricks,
byte for byte, with only the notebook runner left behind. It is invoked as a
**subprocess** rather than imported, for three reasons:

* Its modules import each other flatly (``import utils``, ``from config import
  Config``). Importing them into this process would put a directory holding
  ``config.py``, ``utils.py`` and ``main.py`` on ``sys.path``, where those
  generic names can shadow anything. A subprocess with its own working
  directory keeps that contained, and leaves the science code unedited so a
  newer copy of the pipeline is a straight file replacement.
* Bronze reads every sample of every card. A run that exhausts memory or
  crashes takes its own process down, not the dashboard.
* Progress can be read from its own log output instead of being guessed at.

Nothing here knows a path: the pipeline directory, the recordings, the
calibration and the output root all come from :mod:`app.settings`, hence from
``.env``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from app.data.sources import RunRef
from app.plates.registry import PlateGeometry
from app.settings import SETTINGS, Settings

#: Bundled pipeline; overridden by EIS_PIPELINE_DIR.
BUNDLED_PIPELINE = Path(__file__).resolve().parents[2] / "local_eis"

#: Log lines that mark a stage, in order. Used to turn the pipeline's own
#: banners into a progress bar rather than inventing a fake one.
STAGES = ("BRONZE", "SILVER", "GOLD", "DONE")


class PipelineUnavailable(RuntimeError):
    """Raised when a run is asked for but cannot be started."""


def pipeline_dir(settings: Settings = SETTINGS) -> Path:
    return Path(settings.pipeline_dir) if settings.pipeline_dir else BUNDLED_PIPELINE


def output_dir(ref: RunRef, settings: Settings = SETTINGS) -> Path:
    """Where one condition's results go: straight into the results layout.

    Writing to ``<root>/<order id>/<condition>`` rather than to a scratch area
    means a finished run is immediately visible to the viewer, with no copying
    step for anybody to forget.
    """
    root = (Path(settings.results_roots[0]) if settings.results_roots
            else Path(settings.scratch_results_root))
    return root / ref.measurement_id / ref.condition


def preflight(ref: RunRef, settings: Settings = SETTINGS) -> list[str]:
    """Everything that would stop a run, reported before one is started."""
    problems: list[str] = []

    directory = pipeline_dir(settings)
    if not (directory / "main.py").is_file():
        problems.append(
            f"pipeline not found at {directory} — set EIS_PIPELINE_DIR to the "
            "folder holding main.py, bronze.py, silver.py and gold.py")

    if not ref.files:
        problems.append("no .DAT files in this selection")

    if not settings.curr_cal:
        problems.append(
            "EIS_CURR_CAL is not set. The per-segment shunt calibration is the "
            "only absolute scale in the chain once the potentiostat is gone; "
            "without it the impedances come out in shunt volts per amp rather "
            "than in ohms, and every map is wrong by an unknown factor.")
    elif not Path(settings.curr_cal).exists():
        problems.append(f"shunt calibration not found: {settings.curr_cal}")

    if not settings.allow_inline_pipeline:
        problems.append(
            "running the pipeline from the app is disabled; set "
            "EIS_ALLOW_INLINE_PIPELINE=1 in .env, or run it from the command "
            "line with run_evaluation.py")

    return problems


def warnings_for(settings: Settings = SETTINGS) -> list[str]:
    """Non-fatal problems worth saying out loud before a long run."""
    notes: list[str] = []
    try:
        import matplotlib                                   # noqa: F401
    except ImportError:
        notes.append(
            "matplotlib is not installed. The run will finish bronze, silver "
            "and gold and then fail while writing figures, leaving "
            "gold_manifest.json unwritten — the maps still work, but the "
            "Overview statistics will be missing. `pip install matplotlib`.")
    if not settings.temp_cal:
        notes.append(
            "EIS_TEMP_CAL is not set; the plate temperature falls back to a "
            "fixed value and is flagged as such in the output.")
    return notes


def build_command(ref: RunRef, out: Path, settings: Settings = SETTINGS,
                  equal_areas: bool = False, no_png: bool = False,
                  stop_after: str = "gold",
                  geom: PlateGeometry | None = None) -> list[str]:
    """The exact command line, so it can be shown, logged and reproduced."""
    argv = [
        sys.executable, "main.py",
        "--dat", str(Path(ref.path)),
        "--curr-cal", str(settings.curr_cal),
        "--leepa", ref.measurement_id,
        "--condition", ref.condition,
        "--out", str(out),
        "--stop-after", stop_after,
    ]
    # THE PLATE. This used to be dropped: run_famos took a geometry and never
    # passed it on, so every run evaluated whatever the pipeline defaulted to
    # regardless of the generation picked in the sidebar. A Gen 2 plate
    # evaluated with Gen 1 areas is wrong in every area-weighted quantity.
    if geom is not None:
        argv += ["--plate", "gen2" if "gen2" in geom.key else "gen1"]
    # The whole-cell reference, if there is one to find. Without this the
    # comparison never runs and its tab has nothing to show.
    for root in settings.resolved_gamry_roots():
        if any(root.rglob("*.dta")) or any(root.rglob("*.DTA")):
            argv += ["--gamry", str(root)]
            break
    if settings.temp_cal:
        argv += ["--temp-cal", str(settings.temp_cal)]
    if settings.areas_file:
        argv += ["--areas", str(settings.areas_file)]
    if equal_areas:
        argv.append("--equal-areas")
    if no_png:
        argv.append("--no-png")
    return argv


def run_famos(progress, ref: RunRef, geom: PlateGeometry | None = None,
              settings: Settings = SETTINGS, equal_areas: bool = False,
              no_png: bool = False, stop_after: str = "gold",
              log_lines: list[str] | None = None) -> str:
    """Process one condition. Signature matches :class:`~app.services.store.JobTable`."""
    problems = preflight(ref, settings)
    if problems:
        raise PipelineUnavailable("; ".join(problems))

    out = output_dir(ref, settings)
    out.mkdir(parents=True, exist_ok=True)
    argv = build_command(ref, out, settings, equal_areas, no_png, stop_after,
                         geom=geom)
    directory = pipeline_dir(settings)

    progress(0, len(STAGES), f"starting: {' '.join(argv[1:])}")
    collected = log_lines if log_lines is not None else []

    # The parent reads the child's output as UTF-8, so the child must write it
    # as UTF-8. On Windows it defaults to the console code page (cp1252 here),
    # which cannot carry the box drawing in the banners or the units in the
    # results -- and every such line would come back mangled.
    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    process = subprocess.Popen(
        argv, cwd=str(directory), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
        encoding="utf-8", errors="replace", env=child_env,
    )
    stage = 0
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip()
        if not line:
            continue
        collected.append(line)
        for index, marker in enumerate(STAGES):
            if marker in line:
                stage = max(stage, index + 1)
        progress(stage, len(STAGES), line[:110])

    code = process.wait()
    if code != 0:
        tail = "\n".join(collected[-12:])
        raise PipelineUnavailable(
            f"pipeline exited with code {code}. Last output:\n{tail}")

    progress(len(STAGES), len(STAGES), f"results written to {out}")
    return str(out)


def self_test(settings: Settings = SETTINGS) -> tuple[bool, str]:
    """Run the pipeline's own synthetic checks; needs no measurement data."""
    directory = pipeline_dir(settings)
    if not (directory / "main.py").is_file():
        return False, f"pipeline not found at {directory}"
    result = subprocess.run(
        [sys.executable, "main.py", "--self-test"], cwd=str(directory),
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    return result.returncode == 0, (result.stdout or "") + (result.stderr or "")
