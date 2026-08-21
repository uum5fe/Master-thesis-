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
    """One measured segment: the set of pads wired to it.

    A segment is NOT necessarily a rectangle.  On the gen1 plate the segments
    along the edges are staircases -- segment 37 covers three pads of row 1,
    two of row 2 and one of row 3 -- and an earlier version of this module,
    which assumed rectangles, got 60 of the 72 areas wrong, some by more than
    a factor of two.  The pad set is therefore the primitive, and `col0`,
    `row1`, `x0_mm` and friends are the BOUNDING BOX of that set, kept because
    plotting and cropping want it.  They do not describe the shape.
    """

    number: int
    pads: tuple[tuple[int, int], ...]     # (col, row), 1-based, sorted

    # -- construction -------------------------------------------------------

    @classmethod
    def from_rect(cls, number: int, col0: int, col1: int,
                  row0: int, row1: int) -> "Segment":
        return cls(number, tuple((c, r)
                                 for c in range(col0, col1 + 1)
                                 for r in range(row0, row1 + 1)))

    @classmethod
    def from_pads(cls, number: int, pads) -> "Segment":
        return cls(number, tuple(sorted((int(c), int(r)) for c, r in pads)))

    @property
    def name(self) -> str:
        return str(self.number)

    # -- size ---------------------------------------------------------------

    @property
    def n_pads(self) -> int:
        return len(self.pads)

    @property
    def area_cm2(self) -> float:
        return self.n_pads * PAD_AREA_CM2

    # -- bounding box, in pad indices ---------------------------------------

    @property
    def col0(self) -> int:
        return min(c for c, _r in self.pads)

    @property
    def col1(self) -> int:
        return max(c for c, _r in self.pads)

    @property
    def row0(self) -> int:
        return min(r for _c, r in self.pads)

    @property
    def row1(self) -> int:
        return max(r for _c, r in self.pads)

    @property
    def is_rectangle(self) -> bool:
        return self.n_pads == ((self.col1 - self.col0 + 1)
                               * (self.row1 - self.row0 + 1))

    # -- bounding box, in mm ------------------------------------------------

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
    def w_mm(self) -> float:
        return self.x1_mm - self.x0_mm

    @property
    def h_mm(self) -> float:
        return self.y1_mm - self.y0_mm

    # -- centroid -----------------------------------------------------------
    # The mean of the pad centres, not the middle of the bounding box.  For a
    # rectangle the two agree; for a staircase they do not, and it is the
    # centroid that belongs in a spatial fit or a distance-to-inlet.

    @property
    def cx_mm(self) -> float:
        return PAD_W_MM * (sum(c for c, _r in self.pads) / self.n_pads - 0.5)

    @property
    def cy_mm(self) -> float:
        return PAD_H_MM * (sum(r for _c, r in self.pads) / self.n_pads - 0.5)

    # -- drawing ------------------------------------------------------------

    @property
    def runs(self) -> tuple[tuple[int, int, int], ...]:
        """The pad set as maximal horizontal runs (row, col_from, col_to).

        Drawing these draws the true outline with a handful of rectangles
        instead of one per pad, and they round-trip through JSON.
        """
        by_row: dict[int, list[int]] = {}
        for c, r in self.pads:
            by_row.setdefault(r, []).append(c)
        out: list[tuple[int, int, int]] = []
        for r in sorted(by_row):
            cols = sorted(by_row[r])
            start = prev = cols[0]
            for c in cols[1:]:
                if c == prev + 1:
                    prev = c
                    continue
                out.append((r, start, prev))
                start = prev = c
            out.append((r, start, prev))
        return tuple(out)


# ===========================================================================
# GEN 1  --  "green" / Kashyyyk
# ===========================================================================
# THE AUTHORITATIVE MAP: which segment every one of the 900 pads belongs to.
#
# This is the output of the plant's own `get_900_matrix`, transcribed as it is
# printed: 20 rows of 45 columns, row 1 first, column 1 leftmost -- the same
# orientation the rest of this module uses, so matrix[r-1][c-1] is the segment
# owning pad (col=c, row=r).
#
# It replaces a reconstruction that described the plate as 15 vertical strips
# cut into rectangular bands.  That model tiled the grid and reproduced the
# printed label pads, which is why it survived, but it was still wrong: the
# real segments are staircases wherever they meet the rounded corners of the
# plate.  60 of the 72 areas disagreed with this map, segment 1 by more than a
# factor of two (15 pads assumed, 7 actual).  Nothing is inferred here any
# more -- the areas, the centroids and the map are all read off these numbers.
_GEN1_MATRIX_TEXT = """
    37 37 37 43 43 43 43  5 49 49 49 49  9  9  9 13 13 13 13 13 17 17 17 17 17 21 21 21 21 21 25 25 25 55 55 55 55 29 61 61 61 61 67 67 67
    37 37  1  1 43 43  5  5  5 49 49  9  9  9  9 13 13 13 13 13 17 17 17 17 17 21 21 21 21 21 25 25 25 25 55 55 29 29 29 61 61 33 33 33 67
    37  1  1  1 44 44  5  5  5 50 50  9  9  9  9 13 13 13 13 13 17 17 17 17 17 21 21 21 21 21 25 25 25 25 56 56 29 29 29 62 62 33 33 33 67
    38  1  1 44 44 44  5  5 50 50 50  9  9  9  9 13 13 13 13 13 17 17 17 17 17 21 21 21 21 21 25 25 25 25 56 56 56 29 29 62 62 62 33 33 68
    38 38 44 44 44 44  5  5 50 50 50  9  9  9  9 13 13 13 13 13 17 17 17 17 17 21 21 21 21 21 25 25 25 25 56 56 56 29 29 62 62 62 62 68 68
    38  2  2 44 44 44  6  6 50 50 50 10 10 10 10 14 14 14 14 14 18 18 18 18 18 22 22 22 22 22 26 26 26 26 56 56 56 30 30 62 62 62 34 34 68
    39  2  2 44 44  6  6  6  6 50 50 10 10 10 10 14 14 14 14 14 18 18 18 18 18 22 22 22 22 22 26 26 26 26 56 56 30 30 30 30 62 62 34 34 69
    39  2  2 45 45 45  6  6 51 51 51 10 10 10 10 14 14 14 14 14 18 18 18 18 18 22 22 22 22 22 26 26 26 26 57 57 57 30 30 63 63 63 34 34 69
    39  2  2 45 45 45  6  6 51 51 51 10 10 10 10 14 14 14 14 14 18 18 18 18 18 22 22 22 22 22 26 26 26 26 57 57 57 30 30 63 63 63 34 34 69
    39  2  2 45 45 45  6  6 51 51 51 10 10 10 10 14 14 14 14 14 18 18 18 18 18 22 22 22 22 22 26 26 26 26 57 57 57 30 30 63 63 63 34 34 69
    40  3  3 46 46 46  7  7 52 52 52 11 11 11 11 15 15 15 15 15 19 19 19 19 19 23 23 23 23 23 27 27 27 27 58 58 58 31 31 64 64 64 35 35 70
    40  3  3 46 46 46  7  7 52 52 52 11 11 11 11 15 15 15 15 15 19 19 19 19 19 23 23 23 23 23 27 27 27 27 58 58 58 31 31 64 64 64 35 35 70
    40  3  3 46 46 46  7  7 52 52 52 11 11 11 11 15 15 15 15 15 19 19 19 19 19 23 23 23 23 23 27 27 27 27 58 58 58 31 31 64 64 64 35 35 70
    40  3  3 47 47  7  7  7  7 53 53 11 11 11 11 15 15 15 15 15 19 19 19 19 19 23 23 23 23 23 27 27 27 27 59 59 31 31 31 31 65 65 35 35 70
    41  3  3 47 47 47  7  7 53 53 53 11 11 11 11 15 15 15 15 15 19 19 19 19 19 23 23 23 23 23 27 27 27 27 59 59 59 31 31 65 65 65 35 35 71
    41 41 47 47 47 47  8  8 53 53 53 12 12 12 12 16 16 16 16 16 20 20 20 20 20 24 24 24 24 24 28 28 28 28 59 59 59 32 32 65 65 65 65 71 71
    41  4  4 47 47 47  8  8 53 53 53 12 12 12 12 16 16 16 16 16 20 20 20 20 20 24 24 24 24 24 28 28 28 28 59 59 59 32 32 65 65 65 36 36 71
    42  4  4  4 47 47  8  8  8 53 53 12 12 12 12 16 16 16 16 16 20 20 20 20 20 24 24 24 24 24 28 28 28 28 59 59 32 32 32 65 65 36 36 36 72
    42 42  4  4 48 48  8  8  8 54 54 12 12 12 12 16 16 16 16 16 20 20 20 20 20 24 24 24 24 24 28 28 28 28 60 60 32 32 32 66 66 36 36 72 72
    42 42 42 48 48 48 48  8 54 54 54 54 12 12 12 16 16 16 16 16 20 20 20 20 20 24 24 24 24 24 28 28 28 60 60 60 60 32 66 66 66 66 72 72 72
"""


def _parse_matrix(text: str) -> list[list[int]]:
    grid = [[int(v) for v in line.split()]
            for line in text.strip().splitlines() if line.strip()]
    if len(grid) != N_ROWS or any(len(r) != N_COLS for r in grid):
        raise ValueError(f"pad matrix must be {N_ROWS}x{N_COLS}")
    return grid


def _segments_from_matrix(grid: list[list[int]]) -> dict[str, Segment]:
    pads: dict[int, list[tuple[int, int]]] = {}
    for r, line in enumerate(grid, start=1):
        for c, n in enumerate(line, start=1):
            pads.setdefault(int(n), []).append((c, r))
    return {str(n): Segment.from_pads(n, ps) for n, ps in sorted(pads.items())}


def _label_pads(segs: dict[str, Segment]) -> tuple[dict[str, int], dict[str, int]]:
    """A pad inside the segment to hang its number on.

    The pad nearest the centroid AND owned by the segment -- on a staircase the
    centroid itself can fall in a neighbour, so it cannot be used directly.
    """
    lcol: dict[str, int] = {}
    lrow: dict[str, int] = {}
    for name, seg in segs.items():
        cx = sum(c for c, _r in seg.pads) / seg.n_pads
        cy = sum(r for _c, r in seg.pads) / seg.n_pads
        c, r = min(seg.pads, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
        lcol[name], lrow[name] = c, r
    return lcol, lrow


def _build_gen1() -> tuple[dict[str, Segment], dict[str, int], dict[str, int]]:
    segs = _segments_from_matrix(_parse_matrix(_GEN1_MATRIX_TEXT))
    lcol, lrow = _label_pads(segs)
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
            segs[str(n)] = Segment.from_rect(n, c0, c1, r0, r1)
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
    #: True only when the pad map came from the plant's own `get_900_matrix`.
    #: A layout that merely tiles the grid and matches the printed labels can
    #: still be wrong -- the gen1 reconstruction did both and had 60 of 72
    #: areas wrong -- so anything derived from an unverified plate is marked.
    verified: bool = False

    @property
    def title(self) -> str:
        return f"R2-D2 {self.key} ({self.colour} / {self.name})"


_g1_segs, _g1_lc, _g1_lr = _build_gen1()
_g2_segs, _g2_lc, _g2_lr = _build_gen2()

PLATES: dict[str, Plate] = {
    "gen1": Plate("gen1", "green", "Kashyyyk",
                  "get_900_matrix (plant pad map)",
                  _g1_segs, _g1_lc, _g1_lr, verified=True),
    "gen2": Plate("gen2", "blue", "Naboo",
                  "Coordinates(blue).pdf",
                  _g2_segs, _g2_lc, _g2_lr, verified=False),
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
    owner = {pad: k2 for k2, s2 in b.segments.items() for pad in s2.pads}
    out: dict[str, str] = {}
    for k, seg in a.segments.items():
        # The dst segment holding the most of this src segment's pads.  A
        # centroid test would be wrong for a staircase, whose centroid can
        # land in a neighbour.
        tally: dict[str, int] = {}
        for pad in seg.pads:
            k2 = owner.get(pad)
            if k2 is not None:
                tally[k2] = tally.get(k2, 0) + 1
        if tally:
            out[k] = max(tally.items(), key=lambda kv: kv[1])[0]
    return out


def symmetry_report(plate_name: str | None = None) -> dict:
    """Does the pad map mirror onto itself, left-right and top-bottom?

    The plate is a symmetric piece of hardware: the corner staircases at the
    four corners are the same shape, and every interior segment has a partner.
    That makes symmetry a free, independent check on a transcribed pad map --
    a single mistyped cell shows up as exactly two regions losing their
    partner, which no amount of "it tiles the grid" checking would catch.

    This does NOT compare segment NUMBERS, only the partition into shapes:
    the numbering is deliberately not symmetric.
    """
    p = plate(plate_name) if plate_name else ACTIVE_PLATE

    def shapes(flip_c: bool, flip_r: bool) -> set:
        out = set()
        for seg in p.segments.values():
            out.add(frozenset(((N_COLS + 1 - c) if flip_c else c,
                               (N_ROWS + 1 - r) if flip_r else r)
                              for c, r in seg.pads))
        return out

    base = shapes(False, False)
    lr, tb = shapes(True, False), shapes(False, True)
    unmatched_lr = sorted(
        {seg.number for seg in p.segments.values()
         if frozenset(seg.pads) not in lr})
    unmatched_tb = sorted(
        {seg.number for seg in p.segments.values()
         if frozenset(seg.pads) not in tb})
    return {
        "plate": p.key,
        "left_right": base == lr,
        "top_bottom": base == tb,
        "unmatched_left_right": unmatched_lr,
        "unmatched_top_bottom": unmatched_tb,
    }


# ---------------------------------------------------------------------------
# Self-checks
# ---------------------------------------------------------------------------

def self_check(verbose: bool = True, plate_name: str | None = None) -> dict:
    """Assert that the reconstruction tiles the plate exactly."""
    p = plate(plate_name) if plate_name else ACTIVE_PLATE
    segments, lcol, lrow = p.segments, p.label_col, p.label_row

    # Count the PADS each segment owns, not its bounding box: 40 of the 72
    # gen1 segments are staircases, and a bounding-box count would report a
    # clean tiling for a layout that overlaps badly.
    grid = [[0] * N_COLS for _ in range(N_ROWS)]
    for s in segments.values():
        for c, r in s.pads:
            grid[r - 1][c - 1] += 1

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
                  if (lcol[n], lrow[n]) not in s.pads]
    if bad_labels:
        problems.append(f"label pad outside segment for {bad_labels}")

    sym = symmetry_report(p.key)
    warnings = []
    if not (sym["left_right"] and sym["top_bottom"]):
        warnings.append(
            "pad map is not mirror-symmetric; segments without a partner: "
            f"left-right {sym['unmatched_left_right']}, "
            f"top-bottom {sym['unmatched_top_bottom']}")

    res = {
        "plate": p.key,
        "n_segments": len(segments),
        "pads_covered_once": covered,
        "pads_double": doubled,
        "pads_empty": empty,
        "area_sum_cm2": total_area,
        "area_min_cm2": min(s.area_cm2 for s in segments.values()),
        "area_max_cm2": max(s.area_cm2 for s in segments.values()),
        "n_non_rectangular": sum(1 for s in segments.values()
                                 if not s.is_rectangle),
        "verified": p.verified,
        "symmetric": bool(sym["left_right"] and sym["top_bottom"]),
        "warnings": warnings,
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
        print(f"shape           : {res['n_non_rectangular']} of "
              f"{len(segments)} segments are not rectangles")
        if not p.verified:
            print("NOTE            : this layout is a RECONSTRUCTION, not the "
                  "plant's own pad map — areas from it are provisional")
        for w in warnings:
            print(f"WARNING         : {w}")
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
        # One patch per horizontal pad run, so a staircase segment is drawn as
        # the shape it is.  Drawing the bounding box instead would overlap its
        # neighbours and quietly misreport which pads carry which value.
        for row, c0, c1 in s.runs:
            ax.add_patch(Rectangle(((c0 - 1) * PAD_W_MM, (row - 1) * PAD_H_MM),
                                   (c1 - c0 + 1) * PAD_W_MM, PAD_H_MM,
                                   facecolor=col, edgecolor=col, lw=0.0))
        ax.add_patch(Rectangle((s.x0_mm, s.y0_mm), s.w_mm, s.h_mm,
                               facecolor="none", edgecolor="none"))
        txt = n if v is None else f"{n}\n{v:.3g}"
        ax.text(s.cx_mm, s.cy_mm, txt, ha="center", va="center", fontsize=6.5,
                color="w" if (v is not None and (v - vmin) /
                              (vmax - vmin + 1e-30) < 0.6) else "k")
    # Segment boundaries: draw the edge between two pads that belong to
    # different segments.  This traces every staircase exactly.
    owner = {}
    for n, seg in segments.items():
        for c, r in seg.pads:
            owner[(c, r)] = n
    for (c, r), n in owner.items():
        x0, y0 = (c - 1) * PAD_W_MM, (r - 1) * PAD_H_MM
        if owner.get((c + 1, r)) != n:
            ax.plot([x0 + PAD_W_MM] * 2, [y0, y0 + PAD_H_MM], color="w", lw=1.2)
        if owner.get((c, r + 1)) != n:
            ax.plot([x0, x0 + PAD_W_MM], [y0 + PAD_H_MM] * 2, color="w", lw=1.2)

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
