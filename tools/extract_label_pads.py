#!/usr/bin/env python3
r"""Read segment label pads off an R2-D2 coordinates drawing, in millimetres.

    python tools/extract_label_pads.py Coordinates.pdf
    python tools/extract_label_pads.py green.pdf blue.pdf --compare

Every plate generation comes with a coordinates PDF that prints a segment
number on one pad of each segment. This reads those numbers and their positions
and reports the pad each one sits on, which is the first half of writing a
plate spec - and, run on two drawings, says exactly which segments moved.

It does NOT give segment boundaries. A label pad marks where a number is
printed, not where a segment ends; on these boards the segments are formed by
routing pads to shunts, so the boundaries live in the netlist rather than in
any drawn outline. Use this to see what changed, then get the pad-to-segment
mapping from the board file.

CALIBRATION
-----------
From the axis tick labels, whose millimetre values are printed on the drawing.
Not from the segment numbers: several of them (42, 70, 98, 126 ...) also occur
as tick text, and fitting on those drags the transform sideways by a whole pad.
Validated on the Gen-1 drawing, where it reproduces the documented layout -
label columns 1, 5, 10, 36, 41, 45 and rows 1, 5, 9, 12, 16, 20 for the narrow
strips, columns 3, 8, ... 43 and rows 3, 8, 13, 18 for the wide ones - exactly.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PAD_W_MM = 5.60
PAD_H_MM = 6.05
N_COLS, N_ROWS = 45, 20

#: Tick labels used to calibrate, and the millimetres they stand for.
X_TICKS = ("2.8", "249.2")
Y_TICKS = ("3.025", "117.975")


def read_label_pads(pdf: str | Path, page: int = 1) -> dict[int, tuple[int, int]]:
    """segment number -> (pad column, pad row), both 1-based."""
    try:
        import pymupdf
    except ImportError:                       # pragma: no cover - optional tool
        raise SystemExit("this tool needs pymupdf:  pip install pymupdf")

    words = [((x0 + x1) / 2, (y0 + y1) / 2, w)
             for x0, y0, x1, y1, w, *_ in pymupdf.open(pdf)[page].get_text("words")]

    def one(text: str):
        for x, y, w in words:
            if w == text:
                return x, y
        return None

    xa, xb = one(X_TICKS[0]), one(X_TICKS[1])
    ya, yb = one(Y_TICKS[0]), one(Y_TICKS[1])
    if not all((xa, xb, ya, yb)):
        raise SystemExit(f"{pdf}: axis tick labels {X_TICKS} / {Y_TICKS} not found; "
                         "is this the coordinates page?")

    sx = (float(X_TICKS[1]) - float(X_TICKS[0])) / (xb[0] - xa[0])
    ox = float(X_TICKS[0]) - xa[0] * sx
    sy = (float(Y_TICKS[1]) - float(Y_TICKS[0])) / (yb[1] - ya[1])
    oy = float(Y_TICKS[0]) - ya[1] * sy

    pads: dict[int, tuple[int, int]] = {}
    for x, y, w in words:
        if not re.fullmatch(r"\d{1,2}", w):
            continue
        number = int(w)
        if not 1 <= number <= 72:
            continue
        xm, ym = x * sx + ox, y * sy + oy
        if not (0 <= xm <= PAD_W_MM * N_COLS and 0 <= ym <= PAD_H_MM * N_ROWS):
            continue                          # a tick label outside the plate
        pads[number] = (min(max(int(xm // PAD_W_MM) + 1, 1), N_COLS),
                        min(max(int(ym // PAD_H_MM) + 1, 1), N_ROWS))
    return pads


def by_column(pads: dict[int, tuple[int, int]]) -> dict[int, list[tuple[int, int]]]:
    out: dict[int, list[tuple[int, int]]] = {}
    for number, (col, row) in sorted(pads.items()):
        out.setdefault(col, []).append((row, number))
    return {c: sorted(v) for c, v in sorted(out.items())}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pdf", nargs="+", help="coordinates PDF(s)")
    p.add_argument("--page", type=int, default=2,
                   help="1-based page holding the plate drawing (default 2)")
    p.add_argument("--compare", action="store_true",
                   help="with two PDFs, list only the segments that moved")
    a = p.parse_args(argv)

    tables = [read_label_pads(path, a.page - 1) for path in a.pdf]

    if a.compare and len(tables) == 2:
        first, second = tables
        moved = [n for n in sorted(set(first) | set(second))
                 if first.get(n) != second.get(n)]
        print(f"{len(moved)} of {max(len(first), len(second))} segments moved\n")
        print(f"  {'seg':>3}  {Path(a.pdf[0]).name:>28}  {Path(a.pdf[1]).name:>28}")
        for n in moved:
            print(f"  {n:>3}  {str(first.get(n, '-')):>28}  "
                  f"{str(second.get(n, '-')):>28}")
        unchanged = [n for n in first if first.get(n) == second.get(n)]
        print(f"\n  unchanged: {len(unchanged)} segments")
        return 0

    for path, pads in zip(a.pdf, tables):
        print(f"\n{'=' * 66}\n{Path(path).name}   {len(pads)} labels\n{'=' * 66}")
        for col, entries in by_column(pads).items():
            print(f"  col {col:>2}: " +
                  "  ".join(f"r{r:<2}=s{n:<2}" for r, n in entries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
