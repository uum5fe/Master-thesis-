"""Deployment settings, all from the environment.

The pipeline this frontend sits on top of grew up inside one notebook with one
person's paths baked into it (``/Workspace/Users/<user>/...``,
``/Volumes/<catalog>/.../Famos``).  That is exactly what stops a colleague from
using it.  Nothing here has a personal path in it: every location is an
environment variable with a neutral default, so the same image runs on a
laptop, on a Databricks App and in CI.

In Databricks Apps these go in ``app.yaml`` under ``env:``.  Locally, put them
in a ``.env`` file next to ``run_dashboard.py`` - copy ``.env.example`` and edit
it.  That file is the one place a path is ever written down; no module in this
project contains a personal path, and none needs editing to point the viewer
somewhere new.

Precedence, strongest first:

    1. command-line flags   (``run_dashboard.py --famos ...``)
    2. the real environment (``set EIS_FAMOS_ROOT=...``)
    3. ``.env``

so a flag can always override the file for a one-off, without editing it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: Set to the .env actually read, for the startup banner to report.
DOTENV_LOADED: str = ""


def load_dotenv(path: str | Path | None = None) -> str:
    r"""Fill in unset variables from a ``.env`` file.

    Deliberately does not overwrite anything already in the environment: a
    command-line flag or a shell export has to beat the file, or a one-off
    override would mean editing the file and remembering to change it back.

    Windows paths are the reason quoting is handled: a value like
    ``C:\Users\me\OneDrive - Bosch Group\Local_Eis`` contains spaces and
    backslashes, and neither needs escaping here - the whole rest of the line
    is the value, with surrounding quotes stripped if present.
    """
    global DOTENV_LOADED
    candidates = ([Path(path)] if path else
                  [Path(__file__).resolve().parent.parent / ".env",
                   Path.cwd() / ".env"])
    for candidate in candidates:
        if not candidate.is_file():
            continue
        for line in candidate.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip().removeprefix("export ").strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value
        DOTENV_LOADED = str(candidate)
        return DOTENV_LOADED
    return ""


load_dotenv()


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_list(name: str) -> list[str]:
    r"""Split on the platform's path separator, but not on a Windows drive colon.

    ``os.pathsep`` is ``;`` on Windows and ``:`` on POSIX, so ``C:\data`` stays
    one path on Windows. Several roots are given by separating them with that
    same character.
    """
    return [p.strip() for p in _env(name).split(os.pathsep) if p.strip()]


@dataclass(frozen=True)
class Settings:
    # -- where computed results live ----------------------------------------
    #: One or more roots holding finished pipeline output, laid out as
    #: ``<root>/<measurement_id>/<condition>/{silver,gold}/*.csv``.
    #: On Databricks this is normally a Volumes path.
    results_roots: list[str] = field(default_factory=lambda: _env_list("EIS_RESULTS_ROOT"))

    # -- where raw FAMOS recordings live ------------------------------------
    famos_roots: list[str] = field(default_factory=lambda: _env_list("EIS_FAMOS_ROOT"))
    famos_glob: str = field(default_factory=lambda: _env(
        "EIS_FAMOS_GLOB", "Leepa_*_Current_*_Test_*_Karte_*.DAT"))

    # -- datago (Unity Catalog) ---------------------------------------------
    datago_metadata_table: str = field(default_factory=lambda: _env("EIS_DATAGO_METADATA_TABLE"))
    datago_properties_table: str = field(default_factory=lambda: _env("EIS_DATAGO_PROPERTIES_TABLE"))
    datago_signal_table: str = field(default_factory=lambda: _env("EIS_DATAGO_SIGNAL_TABLE"))
    datago_measurement_type: str = field(default_factory=lambda: _env(
        "EIS_DATAGO_MEASUREMENT_TYPE", "GALVEIS"))
    warehouse_id: str = field(default_factory=lambda: _env("DATABRICKS_WAREHOUSE_ID"))
    databricks_host: str = field(default_factory=lambda: _env("DATABRICKS_HOST"))

    # -- calibration --------------------------------------------------------
    curr_cal: str = field(default_factory=lambda: _env("EIS_CURR_CAL"))
    temp_cal: str = field(default_factory=lambda: _env("EIS_TEMP_CAL"))
    areas_file: str = field(default_factory=lambda: _env("EIS_AREAS_FILE"))

    # -- the processing pipeline --------------------------------------------
    #: Folder holding main.py, bronze.py, silver.py and gold.py. Empty means
    #: the copy bundled in this project, which is the same code that ran on
    #: Databricks.
    pipeline_dir: str = field(default_factory=lambda: _env("EIS_PIPELINE_DIR"))

    #: Where recordings are copied to before processing, when they live on a
    #: network share. Empty means a folder under the system temp directory.
    stage_dir: str = field(default_factory=lambda: _env("EIS_STAGE_DIR"))

    # -- plates -------------------------------------------------------------
    default_plate: str = field(default_factory=lambda: _env("EIS_DEFAULT_PLATE"))

    # -- app ----------------------------------------------------------------
    cache_dir: str = field(default_factory=lambda: _env(
        "EIS_CACHE_DIR", str(Path.home() / ".cache" / "eis-viewer")))
    #: Databricks Apps injects the port to listen on.
    port: int = field(default_factory=lambda: int(_env("DATABRICKS_APP_PORT", "") or
                                                  _env("PORT", "8050")))
    host: str = field(default_factory=lambda: _env("EIS_HOST", "0.0.0.0"))
    debug: bool = field(default_factory=lambda: _env("EIS_DEBUG", "0") not in ("", "0", "false"))
    title: str = field(default_factory=lambda: _env("EIS_TITLE", "Local EIS Viewer"))

    #: Running the pipeline on raw .DAT inside the app is heavy. Off by default;
    #: turn it on only where the app has the compute and the mount to do it.
    allow_inline_pipeline: bool = field(default_factory=lambda: _env(
        "EIS_ALLOW_INLINE_PIPELINE", "0") not in ("", "0", "false"))
    #: Where an inline run writes; it then becomes a normal results root.
    scratch_results_root: str = field(default_factory=lambda: _env(
        "EIS_SCRATCH_RESULTS", str(Path.home() / ".cache" / "eis-viewer" / "runs")))

    def resolved_results_roots(self) -> list[Path]:
        roots = [Path(p) for p in self.results_roots]
        scratch = Path(self.scratch_results_root)
        if scratch not in roots:
            roots.append(scratch)
        return roots

    def datago_configured(self) -> bool:
        return bool(self.datago_metadata_table and self.warehouse_id)

    def summary(self) -> list[tuple[str, str]]:
        """What the app is pointed at, for the Sources panel.

        Databricks rows appear only where Databricks is configured: a row
        reading "datago: (not configured)" tells a reader running on a laptop
        nothing except that there is something they might be missing.
        """
        rows = [
            ("Results roots", ", ".join(self.results_roots) or "(none set)"),
            ("FAMOS roots", ", ".join(self.famos_roots) or "(none set)"),
            ("Pipeline", self.pipeline_dir or "(bundled local_eis/)"),
            ("Calibration", self.curr_cal or "(not set)"),
            ("Process .DAT in the app",
             "enabled" if self.allow_inline_pipeline else "disabled"),
            (".env file", DOTENV_LOADED or "(none found)"),
        ]
        if self.datago_metadata_table or self.warehouse_id:
            rows += [
                ("datago", self.datago_metadata_table or "(not set)"),
                ("SQL warehouse", self.warehouse_id or "(not set)"),
            ]
        return rows


SETTINGS = Settings()
