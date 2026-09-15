r"""Every path the dashboard knows, resolved from the environment.

The variable names are the ones the existing Local EIS viewer already uses, so
one ``.env`` serves both applications and nothing has to be renamed:

    EIS_FAMOS_ROOT      directory of FAMOS .DAT recordings          -> --dat
    EIS_CSV_ROOT        directory or file of CSV measurements       -> --csv
    EIS_GAMRY_ROOT      folder of whole-cell Gamry .DTA sweeps      -> --gamry
    EIS_RESULTS_ROOT    where results are written                   -> --out
    EIS_CURR_CAL        per-segment current Abgleich                -> --curr-cal
    EIS_TEMP_CAL        per-sensor temperature Abgleich             -> --temp-cal
    EIS_ALLOW_INLINE_PIPELINE   1 to allow launching runs from the UI

Precedence, strongest first:

    1. the real environment   (``set EIS_FAMOS_ROOT=...``)
    2. the .env file next to run.py

A value already exported always beats the file, so a one-off override never
means editing .env and remembering to change it back.  ``EIS_SKIP_DOTENV=1``
ignores the file entirely, for a container configured purely from the
environment.
"""

from __future__ import annotations

import ntpath
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

#: Absolute path of the .env actually read, for the startup banner to report.
DOTENV_LOADED: str = ""

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: A Windows drive ("C:\...") or a UNC share ("\\server\share\...").
_WINDOWS_ABS = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|//[^/])")


def load_dotenv(path: str | Path | None = None, *, override: bool = False) -> str:
    r"""Fill unset variables from a ``.env`` file; return the path used.

    Deliberately does not overwrite anything already in the environment.

    Windows paths are why quoting is handled the way it is.  A value like::

        EIS_FAMOS_ROOT=\\bosch.com\DfsRB\...\Lokale_EIS\Daten\2612030_07_09

    contains backslashes and may contain spaces, and neither needs escaping --
    the whole rest of the line after the first ``=`` is the value, with
    surrounding quotes stripped if present.  Nothing here interprets a
    backslash as an escape, which is what would otherwise eat the ``\D`` and
    ``\2`` in that path.
    """
    global DOTENV_LOADED
    if os.environ.get("EIS_SKIP_DOTENV") or os.environ.get("EIS_NO_DOTENV"):
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


def _get(name: str, *aliases: str, default: str = "") -> str:
    for n in (name, *aliases):
        v = os.environ.get(n)
        if v and v.strip():
            return v.strip()
    return default


def is_windows_absolute(v: str) -> bool:
    r"""True for ``C:\x`` and for a UNC share ``\\server\share``.

    Needed because ``pathlib`` on Linux does not recognise either form, so a
    UNC path read from a .env would be judged *relative* and silently joined
    onto this project's directory -- turning a real share into a nonsense path
    under the checkout.  Recognising it here means a Windows path survives
    being read, printed and passed to the pipeline on any platform.
    """
    return bool(_WINDOWS_ABS.match(v))


def _path(name: str, *aliases: str, default: str = "") -> Path | None:
    """A path setting, or None when unset.  Relative to the project root."""
    v = _get(name, *aliases, default=default)
    if not v:
        return None
    if is_windows_absolute(v):
        # Keep the native spelling: normalising it through PurePosixPath on a
        # Linux box would rewrite the separators and break it on Windows.
        return Path(ntpath.normpath(v))
    p = Path(v).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _exists(p: Path | None, *, dir_: bool) -> bool | None:
    """Tri-state existence: True, False, or None when we cannot tell.

    A UNC path checked from a machine that is not on the domain, or any path
    on a share that is momentarily unreachable, raises or returns False for
    reasons that have nothing to do with the setting being wrong.  None means
    "unverifiable here", and the Setup page says so instead of crying error.
    """
    if p is None:
        return False
    # A Windows drive or UNC share cannot be resolved off Windows at all: the
    # check would return False for every one of them and brand a perfectly
    # good .env as broken.  Unverifiable, not wrong.
    if os.name != "nt" and is_windows_absolute(str(p)):
        return None
    try:
        return p.is_dir() if dir_ else p.is_file()
    except OSError:
        return None


def _flag(name: str, *aliases: str, default: bool = False) -> bool:
    v = _get(name, *aliases, default="1" if default else "0")
    return v.lower() not in ("", "0", "false", "no", "off")


def _plate(v: str) -> str:
    """``gen1_r2d2_72`` -> ``gen1``: the pipeline's --plate takes gen1/gen2."""
    m = re.match(r"(gen\d)", v.strip().lower())
    return m.group(1) if m else "gen1"


@dataclass(frozen=True)
class Settings:
    """Resolved configuration.  Construct with :func:`load`."""

    famos_root: Path | None
    csv_root: Path | None
    results_root: Path
    gamry_root: Path | None
    curr_cal: Path | None
    temp_cal: Path | None
    areas_file: Path | None
    source: str = "famos"
    plate: str = "gen1"
    leepa: str = ""
    conditions: list[str] = field(default_factory=list)
    pipeline_dir: Path | None = None
    python: str = ""
    port: int = 8501
    allow_inline_pipeline: bool = False
    title: str = "Local EIS"
    famos_glob: str = "*.DAT"

    # -- validation ---------------------------------------------------------

    @property
    def read_only(self) -> bool:
        return not self.allow_inline_pipeline

    @property
    def input_root(self) -> Path | None:
        return self.csv_root if self.source == "csv" else self.famos_root

    def problems(self) -> list[tuple[str, str, str]]:
        """``(severity, setting, message)`` for everything wrong or missing.

        "error" stops a run; "warning" degrades the result but still produces
        one; "info" is a path we could not check from here.  Rendered on the
        Setup page rather than raised at import, so a half-configured .env is
        diagnosable in the browser instead of only in a traceback.
        """
        out: list[tuple[str, str, str]] = []
        name = "EIS_CSV_ROOT" if self.source == "csv" else "EIS_FAMOS_ROOT"
        root = self.input_root
        if root is None:
            out.append(("error", name, "not set -- nothing to read"))
        else:
            ok = _exists(root, dir_=True)
            if ok is None:
                out.append(("info", name,
                            f"cannot be checked from here (a share may be "
                            f"offline): {root}"))
            elif not ok and not _exists(root, dir_=False):
                out.append(("error", name, f"not found: {root}"))

        res = _exists(self.results_root, dir_=True)
        if res is False and self.results_root.exists():
            out.append(("error", "EIS_RESULTS_ROOT",
                        f"not a directory: {self.results_root}"))

        if self.curr_cal is None:
            out.append(("warning", "EIS_CURR_CAL",
                        "not set -- segment currents stay uncalibrated, so "
                        "R_ohmic carries the raw shunt gain"))
        for label, p, want_dir in (
                ("EIS_CURR_CAL", self.curr_cal, False),
                ("EIS_TEMP_CAL", self.temp_cal, False),
                ("EIS_AREAS_FILE", self.areas_file, False),
                ("EIS_GAMRY_ROOT", self.gamry_root, True)):
            if p is None:
                continue
            ok = _exists(p, dir_=want_dir)
            if ok is None:
                out.append(("info", label, f"cannot be checked from here: {p}"))
            elif not ok:
                out.append(("error", label,
                            f"{'directory' if want_dir else 'file'} "
                            f"not found: {p}"))

        if self.allow_inline_pipeline:
            ok = _exists(self.pipeline_dir, dir_=True)
            if not ok:
                out.append(("error", "EIS_PIPELINE_DIR",
                            f"not a directory: {self.pipeline_dir} -- set it, "
                            f"or unset EIS_ALLOW_INLINE_PIPELINE to browse "
                            f"results only"))
            elif not _exists(self.pipeline_dir / "main.py", dir_=False):
                out.append(("error", "EIS_PIPELINE_DIR",
                            f"no main.py under {self.pipeline_dir}"))
        return out

    def ok(self) -> bool:
        return not any(sev == "error" for sev, _, _ in self.problems())

    def as_rows(self) -> list[dict[str, str]]:
        """Flat table for the Setup page."""
        def show(v, dir_=True) -> tuple[str, str]:
            if v is None:
                return "", "not set"
            ok = _exists(v, dir_=dir_)
            return str(v), {True: "found", False: "MISSING",
                            None: "unverifiable"}[ok]
        rows = []
        for key, val, dir_ in (
                ("EIS_FAMOS_ROOT", self.famos_root, True),
                ("EIS_CSV_ROOT", self.csv_root, True),
                ("EIS_GAMRY_ROOT", self.gamry_root, True),
                ("EIS_RESULTS_ROOT", self.results_root, True),
                ("EIS_CURR_CAL", self.curr_cal, False),
                ("EIS_TEMP_CAL", self.temp_cal, False),
                ("EIS_AREAS_FILE", self.areas_file, False),
                ("EIS_PIPELINE_DIR", self.pipeline_dir, True)):
            v, status = show(val, dir_)
            rows.append({"setting": key, "value": v, "status": status})
        for key, v in (("EIS_SOURCE", self.source),
                       ("EIS_DEFAULT_PLATE", self.plate),
                       ("EIS_LEEPA", self.leepa),
                       ("EIS_CONDITIONS", ",".join(self.conditions)),
                       ("EIS_PYTHON", self.python or "(this one)"),
                       ("EIS_ALLOW_INLINE_PIPELINE",
                        "1" if self.allow_inline_pipeline else "0")):
            rows.append({"setting": key, "value": v, "status": ""})
        return rows


def load(dotenv: str | Path | None = None) -> Settings:
    """Read .env (if present) and the environment into a Settings."""
    load_dotenv(dotenv)
    conds = [c.strip() for c in _get("EIS_CONDITIONS", default="ALL").split(",")
             if c.strip()]
    try:
        port = int(_get("EIS_DASHBOARD_PORT", "EIS_PORT", default="8501"))
    except ValueError:
        port = 8501
    source = _get("EIS_SOURCE", default="famos").lower()
    if source not in ("famos", "csv"):
        source = "famos"
    results = (_path("EIS_RESULTS_ROOT", "EIS_OUT_DIR", default="./results")
               or PROJECT_ROOT / "results")
    return Settings(
        famos_root=_path("EIS_FAMOS_ROOT", "EIS_DAT_DIR"),
        csv_root=_path("EIS_CSV_ROOT"),
        results_root=results,
        gamry_root=_path("EIS_GAMRY_ROOT", "EIS_GAMRY_DIR"),
        curr_cal=_path("EIS_CURR_CAL"),
        temp_cal=_path("EIS_TEMP_CAL"),
        areas_file=_path("EIS_AREAS_FILE", "EIS_AREAS"),
        source=source,
        plate=_plate(_get("EIS_DEFAULT_PLATE", "EIS_PLATE", default="gen1")),
        leepa=_get("EIS_LEEPA"),
        conditions=conds,
        pipeline_dir=_path("EIS_PIPELINE_DIR",
                           default="../../databricks/local_eis"),
        python=_get("EIS_PYTHON"),
        port=port,
        allow_inline_pipeline=_flag("EIS_ALLOW_INLINE_PIPELINE"),
        title=_get("EIS_TITLE", default="Local EIS"),
        famos_glob=_get("EIS_FAMOS_GLOB", default="*.DAT"),
    )
