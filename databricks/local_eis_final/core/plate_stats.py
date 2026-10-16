"""
plate_stats.py
==============
Plate-level mean and median over the 36 representative segments.

The 45 x 20 pad grid splits exactly into 36 tiles of 5 x 5 pads (9 across,
4 down; 28.0 x 30.25 mm each, 25 pads = 8.47 cm^2).  Segment k = 1..36 sits
at the centre of tile k -- its label pad is the middle pad of the tile on
both plate generations:

    tile column i = (k - 1) // 4      pad columns 5i+1 .. 5i+5
    tile row    j = (k - 1) %  4      pad rows    5j+1 .. 5j+5

Segment k STANDS FOR ITS WHOLE TILE, including the pads that belong to the
edge segments 37..72 lying inside it (tile 1 also holds parts of 37, 38, 43
and 44).  The values of 37..72 are NOT used.  Every tile has the same area,
so the plate mean and median are equal-area statistics: the denser edge
segmentation at the two ends of the plate can no longer pull them towards
the inlet and outlet.

A tile whose segment has no measured value (unmeasured, excluded, or only
rebuilt from neighbours) is left out and the count says so: "35 of 36".
"""

from __future__ import annotations

import numpy as np

import r2d2_geometry as geom

TILE = 5                                    # pads per tile side
N_TILE_COLS, N_TILE_ROWS = 9, 4
REPRESENTATIVE: tuple[str, ...] = tuple(str(k) for k in range(1, 37))
BASIS = "36 segments (1–36), one per 5×5-pad tile"


def tile_of(seg) -> tuple[int, int, int, int]:
    """(col0, col1, row0, row1), 1-based inclusive pad indices of the tile
    segment `seg` (1..36) represents."""
    k = int(seg)
    if not 1 <= k <= 36:
        raise ValueError(f"segment {seg} does not represent a tile (1..36)")
    i, j = (k - 1) // N_TILE_ROWS, (k - 1) % N_TILE_ROWS
    return TILE * i + 1, TILE * i + TILE, TILE * j + 1, TILE * j + TILE


def tile_bounds_mm(seg) -> tuple[float, float, float, float]:
    """(x0, x1, y0, y1) of the tile in mm, origin at the top-left pad."""
    c0, c1, r0, r1 = tile_of(seg)
    return ((c0 - 1) * geom.PAD_W_MM, c1 * geom.PAD_W_MM,
            (r0 - 1) * geom.PAD_H_MM, r1 * geom.PAD_H_MM)


def covered(seg, plate_name: str | None = None) -> list[str]:
    """The other segments with pads inside this tile -- the ones whose
    values the tile statistics ignore."""
    p = geom.plate(plate_name) if plate_name else geom.ACTIVE_PLATE
    c0, c1, r0, r1 = tile_of(seg)
    out = {n for n, s in p.segments.items() if n != str(int(seg))
           and any(c0 <= c <= c1 and r0 <= r <= r1 for c, r in s.pads)}
    return sorted(out, key=int)


def _finite(values: dict) -> dict[str, float]:
    out = {}
    for k, v in (values or {}).items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            out[str(int(k)) if str(k).strip().isdigit() else str(k)] = f
    return out


def tile_values(values: dict, classes: dict | None = None,
                positive: bool = False) -> dict[str, float]:
    """segment -> value for the representative segments that count: finite,
    measured (class '' or 'measured' when classes are given) and, with
    `positive`, > 0 (a non-positive resistance is a failed fit)."""
    vals = _finite(values)
    cls = {str(k): str(v) for k, v in (classes or {}).items()}
    return {k: vals[k] for k in REPRESENTATIVE
            if k in vals and cls.get(k, "measured") in ("", "measured")
            and (vals[k] > 0 or not positive)}


def tile_stats(values: dict, classes: dict | None = None,
               positive: bool = False) -> dict:
    """n, mean, median, sd, CV [%], min, max, span [%] over the 36 tiles.

    Every tile has equal weight (25 pads each), so the mean is the plate's
    area-weighted mean and the median the area median."""
    tv = tile_values(values, classes, positive)
    v = np.array(list(tv.values()), float)
    missing = [k for k in REPRESENTATIVE if k not in tv]
    base = dict(n=int(v.size), n_tiles=len(REPRESENTATIVE), missing=missing,
                basis=BASIS)
    if v.size == 0:
        return dict(base, mean=np.nan, median=np.nan, sd=np.nan,
                    cv_pct=np.nan, min=np.nan, max=np.nan, span_pct=np.nan)
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
    return dict(base, mean=mean, median=float(np.median(v)), sd=sd,
                cv_pct=100.0 * sd / abs(mean) if mean else np.nan,
                min=float(v.min()), max=float(v.max()),
                span_pct=100.0 * float(np.ptp(v)) / abs(mean) if mean else np.nan)


def summary_line(values: dict, classes: dict | None = None, dec: int = 1,
                 unit: str = "", sep: str = " · ") -> str:
    """'35 of 36 tiles · mean .. · median .. · sd .. · CV .. · min..max ..'"""
    st = tile_stats(values, classes)
    if st["n"] < 2:
        return ""
    u = f" {unit}" if unit else ""
    return sep.join([
        f"{st['n']} of {st['n_tiles']} tiles (segments 1–36, 5×5 pads each)",
        f"mean {st['mean']:.{dec}f}", f"median {st['median']:.{dec}f}",
        f"sd {st['sd']:.{dec}f}", f"CV {st['cv_pct']:.1f} %",
        f"min..max {st['min']:.{dec}f}..{st['max']:.{dec}f}{u}"])


#: plate_summary.csv columns summarised, with their display units
SUMMARY_PARAMS = (("R_ohmic", "mΩ·cm²"), ("ReZ_1kHz", "mΩ·cm²"),
                  ("R_ct", "mΩ·cm²"), ("R_mt", "mΩ·cm²"), ("R_pol", "mΩ·cm²"),
                  ("j_dc", "A/cm²"), ("Z_mag_100Hz", "mΩ·cm²"),
                  ("phase_100Hz", "°"), ("T_degC", "°C"))


def statistics_rows(summary) -> list[dict]:
    """One row per parameter of a plate_summary table (a DataFrame or the
    path of plate_summary.csv, display units): n, mean, median, sd, CV,
    min, max over the 36 tiles."""
    import pandas as pd
    df = summary if isinstance(summary, pd.DataFrame) else pd.read_csv(summary)
    if "segment" not in df.columns:
        return []
    seg = df["segment"].astype(int).astype(str)
    cls = (dict(zip(seg, df["class"].fillna("measured").astype(str)))
           if "class" in df.columns else None)
    rows = []
    for p, unit in SUMMARY_PARAMS:
        if p not in df.columns:
            continue
        vals = dict(zip(seg, pd.to_numeric(df[p], errors="coerce")))
        st = tile_stats(vals, cls)
        if not st["n"]:
            continue
        rows.append({"parameter": p, "unit": unit, "n_tiles": st["n"],
                     "mean": round(st["mean"], 5),
                     "median": round(st["median"], 5),
                     "sd": round(st["sd"], 5),
                     "cv_pct": round(st["cv_pct"], 2),
                     "min": round(st["min"], 5), "max": round(st["max"], 5),
                     "missing_tiles": " ".join(st["missing"]),
                     "basis": st["basis"]})
    return rows


def write_statistics(summary_csv, out_csv=None):
    """plate_summary.csv -> plate_statistics.csv beside it (or `out_csv`).
    Returns the path, or None when there was nothing to summarise."""
    from pathlib import Path
    import pandas as pd
    rows = statistics_rows(summary_csv)
    if not rows:
        return None
    out = Path(out_csv) if out_csv else Path(summary_csv).with_name(
        "plate_statistics.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    return out
