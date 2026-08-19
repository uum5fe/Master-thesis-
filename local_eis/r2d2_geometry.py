#!/usr/bin/env python3
"""
r2d2_geometry.py
================
Geometry of the Bosch R2-D2 72-segment current-density measurement plate:
per-segment active area, centroid, bounding box and pad membership.

WHY THIS FILE EXISTS
--------------------
The segments are NOT equal in area. They range from 0.68 cm^2 (single-pad
column at the very edge) to 8.47 cm^2 (interior 5x5 block) -- a factor of
12.5. Treating them as 72 x 4.235 cm^2 (the plate mean) is wrong for

  * current density         j_s = I_s / A_s
  * total current closure   sum(j_s * A_s) = I_cell
  * area-weighted averages  <j>, <ASR>
  * aggregation of local impedance into a cell-level spectrum
        Z_cell = A_cell / sum_s ( A_s / Z_s )       (segments in parallel)

Local ASR itself (Z_s = U_cell / j_s) is area-free -- but everything that
sums or averages across segments is not.

SOURCE OF THE NUMBERS
---------------------
Reconstructed from "R2D2_Coordinates_and_Segment_Numbering.pdf" (Bosch
CR/AES3.1, KiCad V1.0, 12.12.2025) and the copper layer plot.

Established directly from the drawing (certain):
  * Active area 252.0 x 121.0 mm = 304.92 cm^2.
  * Pad grid: 45 columns x 20 rows of 5.60 x 6.05 mm pads = 900 pads,
    pad area 0.33880 cm^2.  Every segment is a union of whole pads.
  * The label of every segment sits on one pad; those 72 pad coordinates
    are printed on the drawing and are reproduced exactly below
    (LABEL_COL, LABEL_ROW).
  * Segments 1..36 are labelled on pad columns {3,8,13,18,23,28,33,38,43}
    and pad rows {3,8,13,18}; segments 37..72 on pad columns
    {1,5,10,36,41,45} and pad rows {1,5,9,12,16,20}.

Inferred from those labels (reconstruction, see RECONSTRUCTION below):
  * The plate is cut into 15 full-height vertical strips of 1,3,2,3,1,
    5,5,5,5,5, 1,3,2,3,1 pad columns (mirror-symmetric, sums to 45).
  * The 9 wide strips carry 4 segments each (5 pad rows per segment);
    the 6 narrow strips carry 6 segments each (2,5,3,3,5,2 pad rows).
    9*4 + 6*6 = 72 segments, 900 pads.

RECONSTRUCTION -- why this is the only consistent reading
---------------------------------------------------------
  * 9 label columns x 4 label rows for segments 1..36 and 6 x 6 for
    37..72 is the only split that gives 72 with one label per segment.
  * For the wide strips the labels fall exactly on the centre pad:
    columns 3,8,...,43 are the centres of 28.0 mm strips, rows 3,8,13,18
    the centres of four 30.25 mm bands.  Both are exact, not approximate.
  * The narrow-strip labels (columns 1,5,10,36,41,45; rows 1,5,9,12,16,20)
    are mirror-symmetric about the plate centre and are the centre pads of
    the odd-sized strips/bands; the even-sized ones (2 pads) are labelled
    on the outer pad, again mirror-symmetric.
  * The resulting strip widths tile 45 columns exactly and are symmetric.

CONFIDENCE: the strip structure and the areas of the 36 interior segments
are firm.  The exact pad row split inside the six narrow edge strips
(2,5,3,3,5,2) is the best fit to the printed label rows but is the one
place where a different reading is conceivable -- e.g. (3,4,3,3,4,3).
That alternative changes only segments 37..72 and only by +-1 pad row.
VERIFY IT ONCE against the copper layer or the KiCad board file, then
this module is exact.  Everything downstream reads areas from here (or
from a CSV override), so a correction is a one-line change.

Override:  areas_from_csv("my_areas.csv") -> {"1": 5.082, ...}

Run this file directly to print the table, write segment_areas.csv and
draw the plate map.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Primitive plate constants  (all lengths in mm, areas in cm^2)
# ---------------------------------------------------------------------------

PAD_W_MM = 5.60           # pad pitch in x (45 of them = 252.0 mm)
PAD_H_MM = 6.05           # pad pitch in y (20 of them = 121.0 mm)
N_COLS = 45
N_ROWS = 20

PLATE_W_MM = PAD_W_MM * N_COLS          # 252.0
PLATE_H_MM = PAD_H_MM * N_ROWS          # 121.0
PAD_AREA_CM2 = (PAD_W_MM / 10.0) * (PAD_H_MM / 10.0)     # 0.33880
A_CELL_CM2 = PLATE_W_MM * PLATE_H_MM / 100.0             # 304.92
N_SEGMENTS = 72

# Temperature sensors: x positions on the plate (drawing page 3), full height
TEMP_SENSOR_X_MM = {"temp1": 0.0, "temp2": 84.0, "temp3": 168.0, "temp4": 252.0}

# ---------------------------------------------------------------------------
# Strip structure
# ---------------------------------------------------------------------------
# Vertical strips, given as pad-column spans (1-based, inclusive).
# "wide" strips hold 4 segments, "narrow" strips hold 6.
STRIPS: list[tuple[int, int, str]] = [
    (1, 1, "narrow"),
    (2, 4, "wide"),
    (5, 6, "narrow"),
    (7, 9, "wide"),
    (10, 10, "narrow"),
    (11, 15, "wide"),
    (16, 20, "wide"),
    (21, 25, "wide"),
    (26, 30, "wide"),
    (31, 35, "wide"),
    (36, 36, "narrow"),
    (37, 39, "wide"),
    (40, 41, "narrow"),
    (42, 44, "wide"),
    (45, 45, "narrow"),
]

# Pad-row spans (1-based, inclusive) of the segments inside a strip.
ROWS_WIDE = [(1, 5), (6, 10), (11, 15), (16, 20)]                 # 5,5,5,5
ROWS_NARROW = [(1, 2), (3, 7), (8, 10), (11, 13), (14, 18), (19, 20)]  # 2,5,3,3,5,2

# Label pad printed on the drawing for each segment (col, row), 1-based.
# Used only as a cross-check that the reconstruction reproduces the drawing.
LABEL_COL = {}
LABEL_ROW = {}


@dataclass(frozen=True)
class Segment:
    number: int
    col0: int            # 1-based inclusive pad column span
    col1: int
    row0: int            # 1-based inclusive pad row span
    row1: int

    @property
    def name(self) -> str:
        return str(self.number)

    @property
    def n_pads(self) -> int:
        return (self.col1 - self.col0 + 1) * (self.row1 - self.row0 + 1)

    @property
    def area_cm2(self) -> float:
        return self.n_pads * PAD_AREA_CM2

    @property
    def x0_mm(self) -> float:
        return (self.col0 - 1) * PAD_W_MM

    @property
    def x1_mm(self) -> float:
        return self.col1 * PAD_W_MM

    @property
    def y0_mm(self) -> float:
        return (self.row0 - 1) * PAD_H_MM

    @property
    def y1_mm(self) -> float:
        return self.row1 * PAD_H_MM

    @property
    def cx_mm(self) -> float:
        return 0.5 * (self.x0_mm + self.x1_mm)

    @property
    def cy_mm(self) -> float:
        return 0.5 * (self.y0_mm + self.y1_mm)

    @property
    def w_mm(self) -> float:
        return self.x1_mm - self.x0_mm

    @property
    def h_mm(self) -> float:
        return self.y1_mm - self.y0_mm


def _build() -> dict[str, Segment]:
    """Segment 1..36 fill the wide strips, 37..72 the narrow strips.

    Numbering runs top-to-bottom inside a strip, then strips left to right
    (segment 1 = 2nd strip / top band, segment 5 = 4th strip / top band,
    segment 37 = 1st strip / top band).  This reproduces every label pad
    printed on the drawing; see check_against_drawing().
    """
    segs: dict[str, Segment] = {}

    wide = [s for s in STRIPS if s[2] == "wide"]
    narrow = [s for s in STRIPS if s[2] == "narrow"]

    for i, (c0, c1, _) in enumerate(wide):
        for k, (r0, r1) in enumerate(ROWS_WIDE):
            n = i * len(ROWS_WIDE) + k + 1              # 1 .. 36
            segs[str(n)] = Segment(n, c0, c1, r0, r1)

    for i, (c0, c1, _) in enumerate(narrow):
        for k, (r0, r1) in enumerate(ROWS_NARROW):
            n = 36 + i * len(ROWS_NARROW) + k + 1       # 37 .. 72
            segs[str(n)] = Segment(n, c0, c1, r0, r1)

    return segs


SEGMENTS: dict[str, Segment] = _build()

# Label pads as printed on R2D2_Coordinates_and_Segment_Numbering.pdf page 2.
# (segment -> (pad column, pad row), 1-based).  Derived from the mm ticks:
# x = (col-0.5)*5.60, y = (row-0.5)*6.05.
_WIDE_LABEL_COLS = [3, 8, 13, 18, 23, 28, 33, 38, 43]
_WIDE_LABEL_ROWS = [3, 8, 13, 18]
_NARROW_LABEL_COLS = [1, 5, 10, 36, 41, 45]
_NARROW_LABEL_ROWS = [1, 5, 9, 12, 16, 20]
for _i, _c in enumerate(_WIDE_LABEL_COLS):
    for _k, _r in enumerate(_WIDE_LABEL_ROWS):
        _n = _i * 4 + _k + 1
        LABEL_COL[str(_n)], LABEL_ROW[str(_n)] = _c, _r
for _i, _c in enumerate(_NARROW_LABEL_COLS):
    for _k, _r in enumerate(_NARROW_LABEL_ROWS):
        _n = 36 + _i * 6 + _k + 1
        LABEL_COL[str(_n)], LABEL_ROW[str(_n)] = _c, _r


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------

def areas() -> dict[str, float]:
    """Segment name -> active area in cm^2."""
    return {k: s.area_cm2 for k, s in SEGMENTS.items()}


def equal_areas() -> dict[str, float]:
    """Every segment given the same area, A_CELL/72 = 4.235 cm^2.

    A DELIBERATE SIMPLIFICATION, not a correction.  The real areas span a
    factor of 12.5 and the sums that use them (DC closure, area-weighted
    aggregate, plate averages) are wrong under this assumption by exactly the
    ratio of true to nominal area.  Local ASR is unaffected, since the
    Abgleich returns a current DENSITY and the area never enters Z_s.

    Use it when you want every active segment weighted identically -- e.g.
    to see the plate field without the edge strips dominating an average --
    and read the numbers that depend on area with that in mind.
    """
    a = A_CELL_CM2 / N_SEGMENTS
    return {k: a for k in SEGMENTS}


def centroids() -> dict[str, tuple[float, float]]:
    """Segment name -> (x, y) centroid in mm, origin at the top-left pad."""
    return {k: (s.cx_mm, s.cy_mm) for k, s in SEGMENTS.items()}


def areas_from_csv(path) -> dict[str, float]:
    """Read an override table.  Accepts 'segment,area_cm2' with , or ; and
    an optional header.  Missing segments fall back to the built-in value.
    """
    out = dict(areas())
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.lower().startswith(("segment", "#")):
            continue
        parts = [p for p in line.replace(";", ",").split(",") if p != ""]
        if len(parts) < 2:
            continue
        try:
            out[str(int(float(parts[0])))] = float(parts[1].replace(",", "."))
        except ValueError:
            continue
    return out


def temperature_at(x_mm: float, sensor_T: dict[str, float]) -> float:
    """Linear interpolation of the four plate temperature sensors in x.

    The sensors sit at x = 0, 84, 168, 252 mm and span the full height, so a
    1-D interpolation along the flow direction is the honest reading of them.
    """
    pts = sorted((TEMP_SENSOR_X_MM[k], v) for k, v in sensor_T.items()
                 if k in TEMP_SENSOR_X_MM)
    if not pts:
        raise ValueError("no usable temperature sensors")
    if len(pts) == 1:
        return pts[0][1]
    xs = [p[0] for p in pts]
    ts = [p[1] for p in pts]
    if x_mm <= xs[0]:
        return ts[0]
    if x_mm >= xs[-1]:
        return ts[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x_mm <= xs[i + 1]:
            f = (x_mm - xs[i]) / (xs[i + 1] - xs[i])
            return ts[i] + f * (ts[i + 1] - ts[i])
    return ts[-1]


def segment_temperatures(sensor_T: dict[str, float]) -> dict[str, float]:
    return {k: temperature_at(s.cx_mm, sensor_T) for k, s in SEGMENTS.items()}


# ---------------------------------------------------------------------------
# Self-checks
# ---------------------------------------------------------------------------

def self_check(verbose: bool = True) -> dict:
    """Assert that the reconstruction tiles the plate exactly."""
    grid = [[0] * N_COLS for _ in range(N_ROWS)]
    for s in SEGMENTS.values():
        for r in range(s.row0 - 1, s.row1):
            for c in range(s.col0 - 1, s.col1):
                grid[r][c] += 1

    covered = sum(1 for row in grid for v in row if v == 1)
    doubled = sum(1 for row in grid for v in row if v > 1)
    empty = sum(1 for row in grid for v in row if v == 0)
    total_area = sum(s.area_cm2 for s in SEGMENTS.values())

    problems = []
    if len(SEGMENTS) != N_SEGMENTS:
        problems.append(f"{len(SEGMENTS)} segments, expected {N_SEGMENTS}")
    if doubled or empty:
        problems.append(f"{doubled} pads covered twice, {empty} pads uncovered")
    if abs(total_area - A_CELL_CM2) > 1e-6:
        problems.append(f"area sum {total_area:.4f} != {A_CELL_CM2}")

    # every printed label pad must lie inside its own segment
    bad_labels = [n for n, s in SEGMENTS.items()
                  if not (s.col0 <= LABEL_COL[n] <= s.col1
                          and s.row0 <= LABEL_ROW[n] <= s.row1)]
    if bad_labels:
        problems.append(f"label pad outside segment for {bad_labels}")

    res = {
        "n_segments": len(SEGMENTS),
        "pads_covered_once": covered,
        "pads_double": doubled,
        "pads_empty": empty,
        "area_sum_cm2": total_area,
        "area_min_cm2": min(s.area_cm2 for s in SEGMENTS.values()),
        "area_max_cm2": max(s.area_cm2 for s in SEGMENTS.values()),
        "problems": problems,
    }
    if verbose:
        print(f"segments        : {res['n_segments']}")
        print(f"pads covered 1x : {covered} / {N_COLS*N_ROWS}"
              f"  (double {doubled}, empty {empty})")
        print(f"area sum        : {total_area:.4f} cm2 "
              f"(nominal {A_CELL_CM2:.4f})")
        print(f"area range      : {res['area_min_cm2']:.4f} .. "
              f"{res['area_max_cm2']:.4f} cm2  "
              f"(x{res['area_max_cm2']/res['area_min_cm2']:.1f})")
        print(f"label pads       : {'all inside their segment' if not bad_labels else bad_labels}")
        print("PASS" if not problems else "FAIL: " + "; ".join(problems))
    return res


def write_csv(path="segment_areas.csv") -> Path:
    path = Path(path)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("# R2-D2 segmented plate -- per-segment active area\n")
        fh.write("# reconstructed from R2D2_Coordinates_and_Segment_Numbering.pdf\n")
        fh.write(f"# pad {PAD_W_MM} x {PAD_H_MM} mm = {PAD_AREA_CM2:.5f} cm2; "
                 f"{N_COLS}x{N_ROWS} pads; total {A_CELL_CM2} cm2\n")
        fh.write("segment,area_cm2,n_pads,col_from,col_to,row_from,row_to,"
                 "x0_mm,x1_mm,y0_mm,y1_mm,cx_mm,cy_mm,w_mm,h_mm\n")
        for n in sorted(SEGMENTS, key=int):
            s = SEGMENTS[n]
            fh.write(f"{n},{s.area_cm2:.5f},{s.n_pads},{s.col0},{s.col1},"
                     f"{s.row0},{s.row1},{s.x0_mm:.2f},{s.x1_mm:.2f},"
                     f"{s.y0_mm:.3f},{s.y1_mm:.3f},{s.cx_mm:.2f},"
                     f"{s.cy_mm:.3f},{s.w_mm:.2f},{s.h_mm:.3f}\n")
    return path


def plot_map(path="segment_map.png", value: dict[str, float] | None = None,
             label: str = "area  [cm$^2$]", title: str | None = None):
    """Draw the plate.  Colour = `value` per segment, default = area."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    value = value or areas()
    vals = [v for v in value.values()]
    vmin, vmax = min(vals), max(vals)
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(13, 7))
    for n in sorted(SEGMENTS, key=int):
        s = SEGMENTS[n]
        v = value.get(n)
        col = "0.85" if v is None else cmap((v - vmin) / (vmax - vmin + 1e-30))
        ax.add_patch(Rectangle((s.x0_mm, s.y0_mm), s.w_mm, s.h_mm,
                               facecolor=col, edgecolor="w", lw=1.2))
        txt = n if v is None else f"{n}\n{v:.3g}"
        ax.text(s.cx_mm, s.cy_mm, txt, ha="center", va="center", fontsize=6.5,
                color="w" if (v is not None and (v - vmin) /
                              (vmax - vmin + 1e-30) < 0.6) else "k")
    for x in TEMP_SENSOR_X_MM.values():
        ax.axvline(x, color="tab:red", ls=":", lw=1)

    ax.set(xlim=(-2, PLATE_W_MM + 2), ylim=(PLATE_H_MM + 2, -2),
           xlabel="x [mm]  (temp1 .. temp4 dotted)", ylabel="y [mm]",
           title=title or "R2-D2 plate — segment numbering and active area")
    ax.set_aspect("equal")
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    fig.colorbar(sm, ax=ax, shrink=0.8, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


if __name__ == "__main__":
    print(__doc__.split("Override:")[0])
    self_check()
    print()
    counts: dict[float, list[str]] = {}
    for n, s in sorted(SEGMENTS.items(), key=lambda kv: int(kv[0])):
        counts.setdefault(round(s.area_cm2, 5), []).append(n)
    print("distinct segment sizes:")
    for a, ns in sorted(counts.items()):
        print(f"  {a:8.4f} cm2  ({int(round(a/PAD_AREA_CM2)):2d} pads)  "
              f"x{len(ns):2d}   segments {', '.join(ns[:6])}"
              f"{' ...' if len(ns) > 6 else ''}")
    p = write_csv()
    q = plot_map()
    print(f"\nwritten: {p}  {q}")
