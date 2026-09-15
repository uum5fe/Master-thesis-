"""Every path the dashboard knows, resolved from the environment.

The pipeline script carries two locations that change from person to person --
where the .DAT recordings are read from and where results are written -- and
nothing else about a run is machine-specific.  Those two, plus the optional
calibration and reference paths, are read here and nowhere else, so no module
in this project contains a personal path.

Precedence, strongest first:

    1. the real environment   (``set EIS_DAT_DIR=...``)
    2. the .env file next to run.py

A value already exported always beats the file, so a one-off override never
means editing .env and remembering to change it back.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: Absolute path of the .env actually read, for the startup banner to report.
DOTENV_LOADED: str = ""

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> str:
    r"""Fill unset variables from a ``.env`` file; return the path used.

    Deliberately does not overwrite anything already in the environment.

    Windows paths are why quoting is handled the way it is: a value like
    ``C:\Users\me\OneDrive - Bosch Group\Famos`` contains spaces and
    backslashes and neither needs escaping -- the whole rest of the line after
    the first ``=`` is the value, with surrounding quotes stripped if present.
    """
    global DOTENV_LOADED
    if os.environ.get("EIS_NO_DOTENV"):
        return ""
    p = Path(path) if path else PROJECT_ROOT / ".env"
    if not p.is_file():
        return ""
    for raw in p.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key and (override or key not in os.environ):
            os.environ[key] = val
    DOTENV_LOADED = str(p)
    return DOTENV_LOADED


def _get(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _path(name: str, default: str = "") -> Path | None:
    """A path setting, or None when unset.  Relative to the project root.

    Relative resolution matters for EIS_PIPELINE_DIR, whose sensible default
    is a sibling of this checkout; an absolute value is passed through
    untouched so a Volumes or UNC path still works.
    """
    v = _get(name, default)
    if not v:
        return None
    p = Path(v).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


@dataclass(frozen=True)
class Settings:
    """Resolved configuration.  Construct with :func:`load`."""

    dat_dir: Path | None
    out_dir: Path
    curr_cal: Path | None
    temp_cal: Path | None
    areas: Path | None
    gain: Path | None
    gamry_dir: Path | None
    bench_log: Path | None
    plate: str
    leepa: str
    conditions: list[str] = field(default_factory=list)
    pipeline_dir: Path | None = None
    python: str = ""
    port: int = 8501
    read_only: bool = False

    # -- validation ---------------------------------------------------------

    def problems(self) -> list[tuple[str, str, str]]:
        """``(severity, setting, message)`` for everything wrong or missing.

        Severity is "error" for something that stops a run, "warning" for
        something that degrades the result but still produces one.  The
        dashboard renders this list on the Setup page instead of failing at
        import time, so a half-configured .env is diagnosable in the browser
        rather than only in a traceback.
        """
        out: list[tuple[str, str, str]] = []
        if self.dat_dir is None:
            out.append(("error", "EIS_DAT_DIR",
                        "not set -- no recordings to read"))
        elif not self.dat_dir.is_dir():
            out.append(("error", "EIS_DAT_DIR",
                        f"not a directory: {self.dat_dir}"))
        elif not any(self.dat_dir.glob("*.DAT")) and \
                not any(self.dat_dir.glob("*.dat")):
            out.append(("warning", "EIS_DAT_DIR",
                        f"no .DAT files directly under {self.dat_dir}"))

        if self.out_dir.exists() and not os.access(self.out_dir, os.W_OK):
            out.append(("error", "EIS_OUT_DIR",
                        f"not writable: {self.out_dir}"))

        if self.curr_cal is None:
            out.append(("warning", "EIS_CURR_CAL",
                        "not set -- segment currents stay uncalibrated, so "
                        "R_ohmic carries the raw shunt gain"))
        elif not self.curr_cal.is_file():
            out.append(("error", "EIS_CURR_CAL",
                        f"file not found: {self.curr_cal}"))

        for label, p in (("EIS_TEMP_CAL", self.temp_cal),
                         ("EIS_AREAS", self.areas), ("EIS_GAIN", self.gain),
                         ("EIS_BENCH_LOG", self.bench_log)):
            if p is not None and not p.is_file():
                out.append(("error", label, f"file not found: {p}"))
        if self.gamry_dir is not None and not self.gamry_dir.is_dir():
            out.append(("error", "EIS_GAMRY_DIR",
                        f"not a directory: {self.gamry_dir}"))

        if not self.read_only:
            if self.pipeline_dir is None or not self.pipeline_dir.is_dir():
                out.append(("error", "EIS_PIPELINE_DIR",
                            f"not a directory: {self.pipeline_dir} -- set it "
                            f"or turn on EIS_READ_ONLY to browse results only"))
            elif not (self.pipeline_dir / "main.py").is_file():
                out.append(("error", "EIS_PIPELINE_DIR",
                            f"no main.py under {self.pipeline_dir}"))
        return out

    def ok(self) -> bool:
        return not any(sev == "error" for sev, _, _ in self.problems())

    def as_rows(self) -> list[dict[str, str]]:
        """Flat table for the Setup page."""
        def show(v) -> str:
            return "" if v is None else str(v)
        return [
            {"setting": "EIS_DAT_DIR", "value": show(self.dat_dir)},
            {"setting": "EIS_OUT_DIR", "value": show(self.out_dir)},
            {"setting": "EIS_CURR_CAL", "value": show(self.curr_cal)},
            {"setting": "EIS_TEMP_CAL", "value": show(self.temp_cal)},
            {"setting": "EIS_AREAS", "value": show(self.areas)},
            {"setting": "EIS_GAIN", "value": show(self.gain)},
            {"setting": "EIS_GAMRY_DIR", "value": show(self.gamry_dir)},
            {"setting": "EIS_BENCH_LOG", "value": show(self.bench_log)},
            {"setting": "EIS_PLATE", "value": self.plate},
            {"setting": "EIS_LEEPA", "value": self.leepa},
            {"setting": "EIS_CONDITIONS", "value": ",".join(self.conditions)},
            {"setting": "EIS_PIPELINE_DIR", "value": show(self.pipeline_dir)},
            {"setting": "EIS_PYTHON", "value": self.python or "(this one)"},
            {"setting": "EIS_READ_ONLY", "value": "1" if self.read_only else "0"},
        ]


def load(dotenv: str | Path | None = None) -> Settings:
    """Read .env (if present) and the environment into a Settings."""
    load_dotenv(dotenv)
    conds = [c.strip() for c in _get("EIS_CONDITIONS", "ALL").split(",")
             if c.strip()]
    try:
        port = int(_get("EIS_DASHBOARD_PORT", "8501"))
    except ValueError:
        port = 8501
    return Settings(
        dat_dir=_path("EIS_DAT_DIR"),
        out_dir=_path("EIS_OUT_DIR", "./results") or (PROJECT_ROOT / "results"),
        curr_cal=_path("EIS_CURR_CAL"),
        temp_cal=_path("EIS_TEMP_CAL"),
        areas=_path("EIS_AREAS"),
        gain=_path("EIS_GAIN"),
        gamry_dir=_path("EIS_GAMRY_DIR"),
        bench_log=_path("EIS_BENCH_LOG"),
        plate=_get("EIS_PLATE", "gen1"),
        leepa=_get("EIS_LEEPA"),
        conditions=conds,
        pipeline_dir=_path("EIS_PIPELINE_DIR", "../../databricks/local_eis"),
        python=_get("EIS_PYTHON"),
        port=port,
        read_only=_get("EIS_READ_ONLY", "0") not in ("", "0", "false", "False"),
    )
