#!/usr/bin/env python3
"""
r2d2_geometry.py
================
Geometry of the Bosch R2-D2 72-segment current-density measurement plates:
per-segment active area, centroid, bounding box and pad membership --
for **both plate generations**.

TWO PLATES, ONE PAD GRID
------------------------
There are two hardware revisions in circulation and they are NOT
interchangeable:

    gen1   "green"  / Kashyyyk   R2D2_Coordinates_and_Segment_Numbering.pdf
    gen2   "blue"   / Naboo      Coordinates(blue).pdf

Both are 252.0 x 121.0 mm, both are built from the same 45 x 20 grid of
5.60 x 6.05 mm pads, both carry 72 segments and 4 temperature sensors at
x = 0, 84, 168, 252 mm.  What changed between them is **which pads belong to
which segment number** -- i.e. the segmentation of the 36 edge segments
(37..72) and, as a consequence, their areas.  Segments 1..36 are identical on
both plates.

That distinction is the whole reason this module is plate-aware.  Reading a
gen2 recording with the gen1 map does not fail loudly: every segment number
still exists, every channel still has an area, and the pipeline runs to
completion.  It just attributes the current of one piece of the plate to a
different piece, which silently rotates the heat map and mis-weights every
sum over segments.  Select the plate explicitly:

    import r2d2_geometry as geom
    geom.use_plate("gen2")          # or "gen1"; also accepts "blue"/"green",
                                    # "naboo"/"kashyyyk"
    geom.SEGMENTS, geom.areas(), geom.centroids(), ...

`use_plate` rebinds the module-level names, so every consumer that already
does `geom.SEGMENTS[...]` picks the change up with no edit.  The default is
gen1, which is what every historical run used.

WHY PER-SEGMENT AREAS MATTER
----------------------------
The segments are NOT equal in area.  On gen1 they range from 0.68 cm^2
(single-pad column at the very edge) to 8.47 cm^2 (interior 5x5 block) -- a
factor of 12.5; on gen2 from 1.69 to 8.47 cm^2, a factor of 5.  Treating them
as 72 x 4.235 cm^2 (the plate mean) is wrong for

  * current density         j_s = I_s / A_s
  * total current closure   sum(j_s * A_s) = I_cell
  * area-weighted averages  <j>, <ASR>
  * aggregation of local impedance into a cell-level spectrum
        Z_cell = A_cell / sum_s ( A_s / Z_s )       (segments in parallel)

Local ASR itself (Z_s = U_cell / j_s) is area-free -- but everything that
sums or averages across segments is not.

SOURCE OF THE NUMBERS
---------------------
Both plates are reconstructed from their KiCad coordinate drawings (Bosch
CR/AES3.1).  Established directly from the drawings (certain):

  * Active area 252.0 x 121.0 mm = 304.92 cm^2.
  * Pad grid: 45 columns x 20 rows of 5.60 x 6.05 mm pads = 900 pads,
    pad area 0.33880 cm^2.  Every segment is a union of whole pads.
  * The label of every segment sits on one pad, and those 72 label pad
    coordinates are printed on the drawing.  They are reproduced exactly in
    LABEL_COL / LABEL_ROW below, for both plates, read off the drawing's own
    mm ticks (x = (col-0.5)*5.60, y = (row-0.5)*6.05).

Inferred from those labels (reconstruction, see below):

  * Both plates are cut into the same 15 full-height vertical strips of
    1,3,2,3,1, 5,5,5,5,5, 1,3,2,3,1 pad columns (mirror-symmetric, sums to
    45).  The label columns 1,3,5,8,10,13,18,23,28,33,36,38,41,43,45 are the
    strip centres on both drawings, so the column structure did not change.
  * gen1: the 9 five/three-column "wide" strips carry 4 segments each (5 pad
    rows per segment); the 6 "narrow" strips carry 6 segments each with row
    spans 2,5,3,3,5,2.  9*4 + 6*6 = 72 segments, 900 pads.
  * gen2: the strips carry 4, 6, 6, 4, 2, 6, 6, 4, 6, 6, 2, 4, 6, 6, 4
    segments from left to right -- also 72 -- with the row splits given in
    _GEN2_STRIPS.  The numbering is no longer "narrow strips last": the edge
    numbers 37..72 are interleaved into the wide strips at the top and bottom
    of the plate.

RECONSTRUCTION -- why these are the only consistent readings
------------------------------------------------------------
For **gen1**:
  * 9 label columns x 4 label rows for segments 1..36 and 6 x 6 for 37..72
    is the only split that gives 72 with one label per segment.
  * For the wide strips the labels fall exactly on the centre pad: columns
    3,8,...,43 are the centres of 28.0 mm strips, rows 3,8,13,18 the centres
    of four 30.25 mm bands.  Both are exact, not approximate.
  * The narrow-strip labels (columns 1,5,10,36,41,45; rows 1,5,9,12,16,20)
    are mirror-symmetric about the plate centre.

For **gen2** the label set is again exactly mirror-symmetric about the plate
centre under (col, row) -> (46-col, 21-row): 37<->72, 38<->71, ... 54<->55.
That symmetry, plus "every label lies in its own segment", plus "the strips
tile 45 columns and 20 rows exactly", pins the layout below uniquely once the
column structure is taken to be unchanged:

  * Six labels appear on pads that belong to a *wide* strip on gen1
    (38@col2, 41@col2, 49@col11, 51@col18, 57@col28, 55@col35 and their
    mirrors).  So on gen2 those wide strips are subdivided further -- 6
    segments instead of 4 -- and the extra ones carry edge numbers.
  * Strips 11-15, 16-20, 26-30, 31-35 need label rows {1,3,8,13,18,20}
    inside six segments: only the split 2,4,4,4,4,2 does that symmetrically.
  * Strips 2-4, 5-6, 40-41, 42-44 need label rows {1,5,9,12,16,20} or
    {3,5,8,13,16,18} inside six segments: only 3,3,4,4,3,3 does both.
  * Strips 1, 7-9, 21-25, 37-39, 45 need four segments at label rows
    {1,8,13,20} or {3,8,13,18}: 5,5,5,5.
  * Strips 10 and 36 carry two labels only (rows 5 and 16): 10,10.
  * The resulting segment counts per strip, 4,6,6,4,2,6,6,4,6,6,2,4,6,6,4,
    are themselves mirror-symmetric and sum to 72, and the pad count is
    exactly 900.

CONFIDENCE.  For both plates the strip structure and the areas of the 36
interior segments are firm.  What is a reading rather than a measurement is
the exact pad-row boundary inside the strips that carry more than four
segments -- the labels constrain each boundary to within one pad row.  On
gen1 that is the 2,5,3,3,5,2 split (the alternative 3,4,3,3,4,3 is
conceivable); on gen2 the 3,3,4,4,3,3 and 2,4,4,4,4,2 splits.  Both affect
only segments 37..72 and only by +-1 pad row.  VERIFY ONCE against the copper
layer or the KiCad board file, then this module is exact.  Everything
downstream reads areas from here (or from a CSV override), so a correction is
a one-line change.

Override:  areas_from_csv("my_areas.csv") -> {"1": 5.082, ...}

Run this file directly to print the table for both plates, write
segment_areas_<plate>.csv and draw the plate maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Primitive plate constants  (all lengths in mm, areas in cm^2)
# ---------------------------------------------------------------------------
# Identical on both generations -- the pad grid and the outline did not
# change, only the grouping of pads into segments.

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


# ===========================================================================
# GEN 1  --  "green" / Kashyyyk
# ===========================================================================
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

# Label pads as printed on R2D2_Coordinates_and_Segment_Numbering.pdf page 2.
_GEN1_WIDE_LABEL_COLS = [3, 8, 13, 18, 23, 28, 33, 38, 43]
_GEN1_WIDE_LABEL_ROWS = [3, 8, 13, 18]
_GEN1_NARROW_LABEL_COLS = [1, 5, 10, 36, 41, 45]
_GEN1_NARROW_LABEL_ROWS = [1, 5, 9, 12, 16, 20]


def _build_gen1() -> tuple[dict[str, Segment], dict[str, int], dict[str, int]]:
    """Segments 1..36 fill the wide strips, 37..72 the narrow strips.

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

    lcol: dict[str, int] = {}
    lrow: dict[str, int] = {}
    for i, c in enumerate(_GEN1_WIDE_LABEL_COLS):
        for k, r in enumerate(_GEN1_WIDE_LABEL_ROWS):
            n = i * 4 + k + 1
            lcol[str(n)], lrow[str(n)] = c, r
    for i, c in enumerate(_GEN1_NARROW_LABEL_COLS):
        for k, r in enumerate(_GEN1_NARROW_LABEL_ROWS):
            n = 36 + i * 6 + k + 1
            lcol[str(n)], lrow[str(n)] = c, r
    return segs, lcol, lrow


# ===========================================================================
# GEN 2  --  "blue" / Naboo
# ===========================================================================
# Same 15 column strips as gen1.  Each entry is
#     (col_from, col_to, [(row_from, row_to, segment_number), ...])
# with the row spans tiling 1..20 exactly.  Segments 1..36 sit exactly where
# they sit on gen1 in x, but the strips that also carry an edge segment are
# split into six bands instead of four, so segments 1..36 are NOT identical
# in area to gen1 in those strips -- read areas from here, never assume.
_GEN2_STRIPS: list[tuple[int, int, list[tuple[int, int, int]]]] = [
    # left edge column: four tall segments
    (1, 1, [(1, 5, 37), (6, 10, 39), (11, 15, 40), (16, 20, 42)]),
    # 3-column strip carrying 1..4 plus edge segments 38 and 41
    (2, 4, [(1, 3, 1), (4, 6, 38), (7, 10, 2), (11, 14, 3), (15, 17, 41),
            (18, 20, 4)]),
    # 2-column edge strip: 43..48
    (5, 6, [(1, 3, 43), (4, 6, 44), (7, 10, 45), (11, 14, 46), (15, 17, 47),
            (18, 20, 48)]),
    # 3-column strip: 5..8
    (7, 9, [(1, 5, 5), (6, 10, 6), (11, 15, 7), (16, 20, 8)]),
    # 1-column edge strip split once: 50 (top half) / 53 (bottom half)
    (10, 10, [(1, 10, 50), (11, 20, 53)]),
    # 5-column strips with a thin edge segment at top and bottom
    (11, 15, [(1, 2, 49), (3, 6, 9), (7, 10, 10), (11, 14, 11), (15, 18, 12),
              (19, 20, 54)]),
    (16, 20, [(1, 2, 51), (3, 6, 13), (7, 10, 14), (11, 14, 15), (15, 18, 16),
              (19, 20, 52)]),
    # centre strip: 17..20, no edge segment
    (21, 25, [(1, 5, 17), (6, 10, 18), (11, 15, 19), (16, 20, 20)]),
    (26, 30, [(1, 2, 57), (3, 6, 21), (7, 10, 22), (11, 14, 23), (15, 18, 24),
              (19, 20, 58)]),
    (31, 35, [(1, 2, 55), (3, 6, 25), (7, 10, 26), (11, 14, 27), (15, 18, 28),
              (19, 20, 60)]),
    (36, 36, [(1, 10, 56), (11, 20, 59)]),
    (37, 39, [(1, 5, 29), (6, 10, 30), (11, 15, 31), (16, 20, 32)]),
    (40, 41, [(1, 3, 61), (4, 6, 62), (7, 10, 63), (11, 14, 64), (15, 17, 65),
              (18, 20, 66)]),
    (42, 44, [(1, 3, 33), (4, 6, 68), (7, 10, 34), (11, 14, 35), (15, 17, 71),
              (18, 20, 36)]),
    (45, 45, [(1, 5, 67), (6, 10, 69), (11, 15, 70), (16, 20, 72)]),
]

# Label pads as printed on Coordinates(blue).pdf page 2, read off the mm
# ticks exactly as for gen1.  segment -> (pad column, pad row), 1-based.
_GEN2_LABELS: dict[int, tuple[int, int]] = {
    # 1..36: the wide-strip centre columns, rows 3/8/13/18 -- same as gen1
    **{i * 4 + k + 1: (c, r)
       for i, c in enumerate([3, 8, 13, 18, 23, 28, 33, 38, 43])
       for k, r in enumerate([3, 8, 13, 18])},
    # 37..72: the edge segments, whose placement is what changed
    37: (1, 1), 38: (2, 5), 39: (1, 8), 40: (1, 13), 41: (2, 16), 42: (1, 20),
    43: (5, 1), 44: (5, 5), 45: (5, 9), 46: (5, 12), 47: (5, 16), 48: (5, 20),
    49: (11, 1), 50: (10, 5), 51: (18, 1), 52: (18, 20), 53: (10, 16),
    54: (11, 20),
    55: (35, 1), 56: (36, 5), 57: (28, 1), 58: (28, 20), 59: (36, 16),
    60: (35, 20),
    61: (41, 1), 62: (41, 5), 63: (41, 9), 64: (41, 12), 65: (41, 16),
    66: (41, 20),
    67: (45, 1), 68: (44, 5), 69: (45, 8), 70: (45, 13), 71: (44, 16),
    72: (45, 20),
}


def _build_gen2() -> tuple[dict[str, Segment], dict[str, int], dict[str, int]]:
    segs: dict[str, Segment] = {}
    for c0, c1, bands in _GEN2_STRIPS:
        for r0, r1, n in bands:
            segs[str(n)] = Segment(n, c0, c1, r0, r1)
    lcol = {str(n): cr[0] for n, cr in _GEN2_LABELS.items()}
    lrow = {str(n): cr[1] for n, cr in _GEN2_LABELS.items()}
    return segs, lcol, lrow


# ===========================================================================
# Plate registry
# ===========================================================================


@dataclass(frozen=True)
class Plate:
    key: str
    colour: str
    name: str                 # campaign name printed on the hardware
    drawing: str
    segments: dict[str, Segment]
    label_col: dict[str, int]
    label_row: dict[str, int]

    @property
    def title(self) -> str:
        return f"R2-D2 {self.key} ({self.colour} / {self.name})"


_g1_segs, _g1_lc, _g1_lr = _build_gen1()
_g2_segs, _g2_lc, _g2_lr = _build_gen2()

PLATES: dict[str, Plate] = {
    "gen1": Plate("gen1", "green", "Kashyyyk",
                  "R2D2_Coordinates_and_Segment_Numbering.pdf",
                  _g1_segs, _g1_lc, _g1_lr),
    "gen2": Plate("gen2", "blue", "Naboo",
                  "Coordinates(blue).pdf",
                  _g2_segs, _g2_lc, _g2_lr),
}

# Everything a user might reasonably type for a plate, mapped to its key.
_ALIASES = {
    "gen1": "gen1", "gen 1": "gen1", "g1": "gen1", "1": "gen1", "v1": "gen1",
    "green": "gen1", "kashyyyk": "gen1", "r2d2_green_kashyyyk": "gen1",
    "gen2": "gen2", "gen 2": "gen2", "g2": "gen2", "2": "gen2", "v2": "gen2",
    "blue": "gen2", "naboo": "gen2", "r2d2_blue_naboo": "gen2",
}

DEFAULT_PLATE = "gen1"


def plate_key(name: str | None) -> str:
    """Normalise any spelling of a plate to its registry key."""
    if not name:
        return DEFAULT_PLATE
    k = _ALIASES.get(str(name).strip().lower())
    if k is None:
        raise ValueError(
            f"unknown plate {name!r}; known: {sorted(set(_ALIASES.values()))} "
            f"(aliases: {sorted(_ALIASES)})")
    return k


def plate(name: str | None = None) -> Plate:
    return PLATES[plate_key(name)]


# --- module-level state, rebound by use_plate() ---------------------------
ACTIVE_PLATE: Plate = PLATES[DEFAULT_PLATE]
SEGMENTS: dict[str, Segment] = ACTIVE_PLATE.segments
LABEL_COL: dict[str, int] = ACTIVE_PLATE.label_col
LABEL_ROW: dict[str, int] = ACTIVE_PLATE.label_row


def use_plate(name: str | None) -> Plate:
    """Select the plate generation for the whole process.

    Rebinds SEGMENTS / LABEL_COL / LABEL_ROW so that every consumer holding
    `import r2d2_geometry as geom` sees the change without an edit.  Call it
    once, early -- main.run_pipeline() does it from cfg.plate.
    """
    global ACTIVE_PLATE, SEGMENTS, LABEL_COL, LABEL_ROW
    p = plate(name)
    ACTIVE_PLATE = p
    SEGMENTS = p.segments
    LABEL_COL = p.label_col
    LABEL_ROW = p.label_row
    return p


def active_plate() -> str:
    return ACTIVE_PLATE.key


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------

def areas(plate_name: str | None = None) -> dict[str, float]:
    """Segment name -> active area in cm^2 (active plate unless given)."""
    segs = plate(plate_name).segments if plate_name else SEGMENTS
    return {k: s.area_cm2 for k, s in segs.items()}


def equal_areas(plate_name: str | None = None) -> dict[str, float]:
    """Every segment given the same area, A_CELL/72 = 4.235 cm^2.

    A DELIBERATE SIMPLIFICATION, not a correction.  The real areas span a
    factor of 12.5 (gen1) or 5 (gen2) and the sums that use them (DC closure,
    area-weighted aggregate, plate averages) are wrong under this assumption
    by exactly the ratio of true to nominal area.  Local ASR is unaffected,
    since the Abgleich returns a current DENSITY and the area never enters
    Z_s.

    Use it when you want every active segment weighted identically -- e.g.
    to see the plate field without the edge strips dominating an average --
    and read the numbers that depend on area with that in mind.
    """
    segs = plate(plate_name).segments if plate_name else SEGMENTS
    a = A_CELL_CM2 / N_SEGMENTS
    return {k: a for k in segs}


def centroids(plate_name: str | None = None) -> dict[str, tuple[float, float]]:
    """Segment name -> (x, y) centroid in mm, origin at the top-left pad."""
    segs = plate(plate_name).segments if plate_name else SEGMENTS
    return {k: (s.cx_mm, s.cy_mm) for k, s in segs.items()}


def areas_from_csv(path) -> dict[str, float]:
    """Read an override table.  Accepts 'segment,area_cm2' with , or ; and
    an optional header.  Missing segments fall back to the built-in value
    of the ACTIVE plate.
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


def segment_temperatures(sensor_T: dict[str, float],
                         plate_name: str | None = None) -> dict[str, float]:
    segs = plate(plate_name).segments if plate_name else SEGMENTS
    return {k: temperature_at(s.cx_mm, sensor_T) for k, s in segs.items()}


def renumbering(src: str = "gen1", dst: str = "gen2") -> dict[str, str]:
    """Best-effort map src segment number -> dst segment number, by centroid.

    The two plates do not have a one-to-one segment correspondence -- the
    edge segments were re-cut, not merely renamed -- so this is a *lookup for
    orientation*, not a conversion.  Each src segment is mapped to the dst
    segment that contains its centroid.  Use it to answer "where on the gen2
    plate does gen1 segment 51 sit?", never to convert a data set.
    """
    a, b = plate(src), plate(dst)
    out: dict[str, str] = {}
    for k, s in a.segments.items():
        col = int(s.cx_mm // PAD_W_MM) + 1
        row = int(s.cy_mm // PAD_H_MM) + 1
        for k2, s2 in b.segments.items():
            if s2.col0 <= col <= s2.col1 and s2.row0 <= row <= s2.row1:
                out[k] = k2
                break
    return out


# ---------------------------------------------------------------------------
# Self-checks
# ---------------------------------------------------------------------------

def self_check(verbose: bool = True, plate_name: str | None = None) -> dict:
    """Assert that the reconstruction tiles the plate exactly."""
    p = plate(plate_name) if plate_name else ACTIVE_PLATE
    segments, lcol, lrow = p.segments, p.label_col, p.label_row

    grid = [[0] * N_COLS for _ in range(N_ROWS)]
    for s in segments.values():
        for r in range(s.row0 - 1, s.row1):
            for c in range(s.col0 - 1, s.col1):
                grid[r][c] += 1

    covered = sum(1 for row in grid for v in row if v == 1)
    doubled = sum(1 for row in grid for v in row if v > 1)
    empty = sum(1 for row in grid for v in row if v == 0)
    total_area = sum(s.area_cm2 for s in segments.values())

    problems = []
    if len(segments) != N_SEGMENTS:
        problems.append(f"{len(segments)} segments, expected {N_SEGMENTS}")
    if doubled or empty:
        problems.append(f"{doubled} pads covered twice, {empty} pads uncovered")
    if abs(total_area - A_CELL_CM2) > 1e-6:
        problems.append(f"area sum {total_area:.4f} != {A_CELL_CM2}")

    # every printed label pad must lie inside its own segment
    bad_labels = [n for n, s in segments.items()
                  if not (s.col0 <= lcol[n] <= s.col1
                          and s.row0 <= lrow[n] <= s.row1)]
    if bad_labels:
        problems.append(f"label pad outside segment for {bad_labels}")

    res = {
        "plate": p.key,
        "n_segments": len(segments),
        "pads_covered_once": covered,
        "pads_double": doubled,
        "pads_empty": empty,
        "area_sum_cm2": total_area,
        "area_min_cm2": min(s.area_cm2 for s in segments.values()),
        "area_max_cm2": max(s.area_cm2 for s in segments.values()),
        "problems": problems,
    }
    if verbose:
        print(f"plate           : {p.title}")
        print(f"segments        : {res['n_segments']}")
        print(f"pads covered 1x : {covered} / {N_COLS*N_ROWS}"
              f"  (double {doubled}, empty {empty})")
        print(f"area sum        : {total_area:.4f} cm2 "
              f"(nominal {A_CELL_CM2:.4f})")
        print(f"area range      : {res['area_min_cm2']:.4f} .. "
              f"{res['area_max_cm2']:.4f} cm2  "
              f"(x{res['area_max_cm2']/res['area_min_cm2']:.1f})")
        print(f"label pads      : {'all inside their segment' if not bad_labels else bad_labels}")
        print("PASS" if not problems else "FAIL: " + "; ".join(problems))
    return res


def write_csv(path=None, plate_name: str | None = None) -> Path:
    p = plate(plate_name) if plate_name else ACTIVE_PLATE
    path = Path(path or f"segment_areas_{p.key}.csv")
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# R2-D2 {p.key} ({p.colour}/{p.name}) -- per-segment active area\n")
        fh.write(f"# reconstructed from {p.drawing}\n")
        fh.write(f"# pad {PAD_W_MM} x {PAD_H_MM} mm = {PAD_AREA_CM2:.5f} cm2; "
                 f"{N_COLS}x{N_ROWS} pads; total {A_CELL_CM2} cm2\n")
        fh.write("segment,area_cm2,n_pads,col_from,col_to,row_from,row_to,"
                 "x0_mm,x1_mm,y0_mm,y1_mm,cx_mm,cy_mm,w_mm,h_mm\n")
        for n in sorted(p.segments, key=int):
            s = p.segments[n]
            fh.write(f"{n},{s.area_cm2:.5f},{s.n_pads},{s.col0},{s.col1},"
                     f"{s.row0},{s.row1},{s.x0_mm:.2f},{s.x1_mm:.2f},"
                     f"{s.y0_mm:.3f},{s.y1_mm:.3f},{s.cx_mm:.2f},"
                     f"{s.cy_mm:.3f},{s.w_mm:.2f},{s.h_mm:.3f}\n")
    return path


def plot_map(path="segment_map.png", value: dict[str, float] | None = None,
             label: str = "area  [cm$^2$]", title: str | None = None,
             plate_name: str | None = None):
    """Draw the plate.  Colour = `value` per segment, default = area."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    p = plate(plate_name) if plate_name else ACTIVE_PLATE
    segments = p.segments
    value = value or {k: s.area_cm2 for k, s in segments.items()}
    vals = [v for v in value.values()]
    vmin, vmax = min(vals), max(vals)
    cmap = plt.get_cmap("viridis")

    fig, ax = plt.subplots(figsize=(13, 7))
    for n in sorted(segments, key=int):
        s = segments[n]
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
           title=title or f"{p.title} — segment numbering and active area")
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
    for key in PLATES:
        use_plate(key)
        self_check()
        counts: dict[float, list[str]] = {}
        for n, s in sorted(SEGMENTS.items(), key=lambda kv: int(kv[0])):
            counts.setdefault(round(s.area_cm2, 5), []).append(n)
        print("distinct segment sizes:")
        for a, ns in sorted(counts.items()):
            print(f"  {a:8.4f} cm2  ({int(round(a/PAD_AREA_CM2)):2d} pads)  "
                  f"x{len(ns):2d}   segments {', '.join(ns[:6])}"
                  f"{' ...' if len(ns) > 6 else ''}")
        p = write_csv()
        q = plot_map(f"segment_map_{key}.png")
        print(f"written: {p}  {q}\n")
    use_plate(DEFAULT_PLATE)
