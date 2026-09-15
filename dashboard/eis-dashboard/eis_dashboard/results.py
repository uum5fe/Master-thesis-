"""Reading what the pipeline wrote.

The pipeline writes one directory per run, laid out as::

    <EIS_OUT_DIR>/<leepa>/<condition>/
        run_manifest.json  config_used.json
        bronze/   schedule.csv  bronze_manifest.json  raw_spectra.csv
        silver/   impedance.csv  point_rejections.csv  cell_aggregate.csv
        gold/     plate_summary.csv  gold_manifest.json  map_*.png  nyquist.png

Nothing here assumes that layout is complete: a run stopped after bronze has no
gold/, and a browse-only deployment may hold results copied from elsewhere with
the PNGs stripped.  Every loader returns None or an empty frame rather than
raising, and the pages say what is missing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

#: Written by gold.save(); its presence is what makes a directory "a run".
PLATE_SUMMARY = "gold/plate_summary.csv"

#: tau below which the mass-transport bucket cannot exist. Mirrors
#: config.tau_split_kinetic_s; see band_limited_mt().
TAU_SPLIT_KINETIC_S = 1e-2


@dataclass(frozen=True)
class Run:
    """One plate/condition result directory."""

    path: Path
    leepa: str
    condition: str

    @property
    def label(self) -> str:
        return f"{self.leepa} / {self.condition}" if self.leepa else self.condition

    def has(self, rel: str) -> bool:
        return (self.path / rel).exists()


def discover(out_dir: Path, max_depth: int = 4) -> list[Run]:
    """Every run directory under ``out_dir``, newest first.

    A run is a directory holding gold/plate_summary.csv OR run_manifest.json,
    so a bronze-only run still shows up (and its pages say what it lacks)
    instead of vanishing from the picker.
    """
    if not out_dir or not Path(out_dir).is_dir():
        return []
    root = Path(out_dir)
    seen: dict[Path, Run] = {}
    for marker in (PLATE_SUMMARY, "run_manifest.json", "bronze/schedule.csv"):
        for hit in root.glob("/".join(["*"] * 0 + [marker])):
            _add(seen, root, hit, marker)
        for d in range(1, max_depth + 1):
            for hit in root.glob("/".join(["*"] * d + [marker])):
                _add(seen, root, hit, marker)
    return sorted(seen.values(),
                  key=lambda r: (r.path.stat().st_mtime, r.label),
                  reverse=True)


def _add(seen: dict, root: Path, hit: Path, marker: str) -> None:
    run_dir = hit.parent if "/" not in marker else hit.parent.parent
    if run_dir in seen:
        return
    rel = run_dir.relative_to(root).parts
    condition = rel[-1] if rel else run_dir.name
    leepa = rel[-2] if len(rel) >= 2 else ""
    seen[run_dir] = Run(path=run_dir, leepa=leepa, condition=condition)


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def read_csv(run: Run, rel: str) -> pd.DataFrame:
    """A CSV from the run, or an empty frame when it is not there."""
    p = run.path / rel
    if not p.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def read_json(run: Run, rel: str) -> dict:
    p = run.path / rel
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def plate_summary(run: Run) -> pd.DataFrame:
    """Per-segment parameters, with the segment column as a sortable int.

    gold.save() writes a blank cell for a non-finite value, which pandas reads
    as NaN -- that is the distinction the whole R_mt story below rests on, so
    it must not be filled in here.
    """
    df = read_csv(run, PLATE_SUMMARY)
    if df.empty:
        return df
    if "segment" in df:
        df["segment"] = pd.to_numeric(df["segment"], errors="coerce")
        df = df.sort_values("segment").reset_index(drop=True)
    return df


#: Columns that are geometry or bookkeeping, never a mappable parameter.
_NOT_PARAMETERS = {"segment", "cx_mm", "cy_mm", "area_cm2",
                   "class", "tier", "fault", "flags"}


def parameter_columns(df: pd.DataFrame) -> list[str]:
    """Numeric parameter columns, excluding geometry, _sd pairs and labels.

    Dtype alone is not enough to decide this, in either direction.  pandas 3
    gives a text column dtype ``str`` rather than ``object``, so an
    ``== object`` test lets "class" and "tier" through; and gold.save() writes
    an empty cell for a non-finite value, so an all-blank text column such as
    "flags" arrives as float64 and looks numeric.  Hence: an explicit name
    list, a real numeric-dtype test, and a requirement that the column hold at
    least one finite value.
    """
    out = []
    for c in df.columns:
        if c in _NOT_PARAMETERS or c.endswith("_sd"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if not pd.to_numeric(df[c], errors="coerce").notna().any():
            continue
        out.append(c)
    return out


def band_limited_mt(df: pd.DataFrame) -> pd.Series:
    """Segments whose band cannot reach the mass-transport bucket at all.

    The DRT tau grid spans 1/(2*pi*f_max) .. 1/(2*pi*f_min), so the slow
    bucket (tau >= 10 ms) is EMPTY unless the segment kept a point at or below
    1/(2*pi*0.01) = 15.9 Hz.  A segment with no such point has no
    mass-transport estimate -- which is a different statement from "its mass
    transport is zero", and the two must never be drawn the same way.

    Returns an all-False series when the run predates the tau_max column, so
    the caller can say "cannot tell" rather than silently claiming none.
    """
    if "tau_max" not in df.columns:
        return pd.Series(False, index=df.index)
    tau = pd.to_numeric(df["tau_max"], errors="coerce")
    return tau.notna() & (tau < TAU_SPLIT_KINETIC_S)


def images(run: Run, sub: str = "gold") -> list[Path]:
    d = run.path / sub
    return sorted(d.glob("*.png")) if d.is_dir() else []


def log_text(run: Run, limit: int = 400_000) -> str:
    """The captured run log, tail-truncated."""
    for name in ("run.log", "pipeline.log"):
        p = run.path / name
        if p.is_file():
            t = p.read_text(encoding="utf-8", errors="replace")
            return t if len(t) <= limit else "...\n" + t[-limit:]
    return ""
