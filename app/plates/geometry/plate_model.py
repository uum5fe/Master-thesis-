"""Segmented measuring plate — geometry, numbering, ports and a realistic plot.

The pad map below is the authoritative one: `get_900_matrix`'s `s` array,
transcribed unchanged.  45 columns x 20 rows = 900 pads of 5.60 x 6.05 mm,
252.0 x 121.0 mm active area (304.92 cm2), 72 segments whose areas run from
1.355 to 8.470 cm2.  Nothing about the layout is inferred from it: a segment is
the set of pads carrying its number, so its area is exact and its boundary --
staircases included -- is walked from the pad edges.

Two numbering schemes are available and they describe the same copper:

    "wired"     the numbers printed in the matrix (the harness numbering)
    "topleft"   1 = the top segment of the leftmost full-height strip, running
                down each strip and then to the next strip rightwards

Gas ports, as installed:

    H2  in  bottom left    ->  out top right      (anode,   left -> right)
    O2  in  bottom right   ->  out top left       (cathode, right -> left)

so the two gases are in counter-flow and each inlet faces the other's outlet.

Usage
-----
    from plate_model import PLATE, draw_plate, synthetic_fields

    values = {seg: rs for seg, rs in zip(df.segment, df.rs_mohm_cm2)}
    fig, ax = draw_plate(values, label="HFR", unit="mOhm.cm2", cmap="viridis")
    fig.savefig("hfr.png", dpi=220)

Only numpy and matplotlib are required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# the pad map
# --------------------------------------------------------------------------

PAD_MATRIX: List[List[int]] = [
    [37,37,37,43,43,43,43, 5,49,49,49,49, 9, 9, 9,13,13,13,13,13,17,17,17,17,17,21,21,21,21,21,25,25,25,55,55,55,55,29,61,61,61,61,67,67,67],
    [37,37, 1, 1,43,43, 5, 5, 5,49,49, 9, 9, 9, 9,13,13,13,13,13,17,17,17,17,17,21,21,21,21,21,25,25,25,25,55,55,29,29,29,61,61,33,33,33,67],
    [37, 1, 1, 1,44,44, 5, 5, 5,50,50, 9, 9, 9, 9,13,13,13,13,13,17,17,17,17,17,21,21,21,21,21,25,25,25,25,56,56,29,29,29,62,62,33,33,33,67],
    [38, 1, 1,44,44,44, 5, 5,50,50,50, 9, 9, 9, 9,13,13,13,13,13,17,17,17,17,17,21,21,21,21,21,25,25,25,25,56,56,56,29,29,62,62,62,33,33,68],
    [38,38,44,44,44,44, 5, 5,50,50,50, 9, 9, 9, 9,13,13,13,13,13,17,17,17,17,17,21,21,21,21,21,25,25,25,25,56,56,56,29,29,62,62,62,62,68,68],
    [38, 2, 2,44,44,44, 6, 6,50,50,50,10,10,10,10,14,14,14,14,14,18,18,18,18,18,22,22,22,22,22,26,26,26,26,56,56,56,30,30,62,62,62,34,34,68],
    [39, 2, 2,44,44, 6, 6, 6, 6,50,50,10,10,10,10,14,14,14,14,14,18,18,18,18,18,22,22,22,22,22,26,26,26,26,56,56,30,30,30,30,62,62,34,34,69],
    [39, 2, 2,45,45,45, 6, 6,51,51,51,10,10,10,10,14,14,14,14,14,18,18,18,18,18,22,22,22,22,22,26,26,26,26,57,57,57,30,30,63,63,63,34,34,69],
    [39, 2, 2,45,45,45, 6, 6,51,51,51,10,10,10,10,14,14,14,14,14,18,18,18,18,18,22,22,22,22,22,26,26,26,26,57,57,57,30,30,63,63,63,34,34,69],
    [39, 2, 2,45,45,45, 6, 6,51,51,51,10,10,10,10,14,14,14,14,14,18,18,18,18,18,22,22,22,22,22,26,26,26,26,57,57,57,30,30,63,63,63,34,34,69],
    [40, 3, 3,46,46,46, 7, 7,52,52,52,11,11,11,11,15,15,15,15,15,19,19,19,19,19,23,23,23,23,23,27,27,27,27,58,58,58,31,31,64,64,64,35,35,70],
    [40, 3, 3,46,46,46, 7, 7,52,52,52,11,11,11,11,15,15,15,15,15,19,19,19,19,19,23,23,23,23,23,27,27,27,27,58,58,58,31,31,64,64,64,35,35,70],
    [40, 3, 3,46,46,46, 7, 7,52,52,52,11,11,11,11,15,15,15,15,15,19,19,19,19,19,23,23,23,23,23,27,27,27,27,58,58,58,31,31,64,64,64,35,35,70],
    [40, 3, 3,47,47, 7, 7, 7, 7,53,53,11,11,11,11,15,15,15,15,15,19,19,19,19,19,23,23,23,23,23,27,27,27,27,59,59,31,31,31,31,65,65,35,35,70],
    [41, 3, 3,47,47,47, 7, 7,53,53,53,11,11,11,11,15,15,15,15,15,19,19,19,19,19,23,23,23,23,23,27,27,27,27,59,59,59,31,31,65,65,65,35,35,71],
    [41,41,47,47,47,47, 8, 8,53,53,53,12,12,12,12,16,16,16,16,16,20,20,20,20,20,24,24,24,24,24,28,28,28,28,59,59,59,32,32,65,65,65,65,71,71],
    [41, 4, 4,47,47,47, 8, 8,53,53,53,12,12,12,12,16,16,16,16,16,20,20,20,20,20,24,24,24,24,24,28,28,28,28,59,59,59,32,32,65,65,65,36,36,71],
    [42, 4, 4, 4,47,47, 8, 8, 8,53,53,12,12,12,12,16,16,16,16,16,20,20,20,20,20,24,24,24,24,24,28,28,28,28,59,59,32,32,32,65,65,36,36,36,72],
    [42,42, 4, 4,48,48, 8, 8, 8,54,54,12,12,12,12,16,16,16,16,16,20,20,20,20,20,24,24,24,24,24,28,28,28,28,60,60,32,32,32,66,66,36,36,72,72],
    [42,42,42,48,48,48,48, 8,54,54,54,54,12,12,12,16,16,16,16,16,20,20,20,20,20,24,24,24,24,24,28,28,28,60,60,60,60,32,66,66,66,66,72,72,72],
]

PAD_W_MM = 5.60
PAD_H_MM = 6.05
N_COLS = 45
N_ROWS = 20

#: Temperature sensors, as x positions along the flow axis.
TEMP_SENSOR_X_MM = {"temp1": 0.0, "temp2": 84.0, "temp3": 168.0, "temp4": 252.0}

#: Full-height strips, left to right, each listing its wired segment numbers.
#: Six edge strips carry six segments, nine interior strips carry four.
STRIPS: List[List[int]] = [
    [37, 38, 39, 40, 41, 42], [1, 2, 3, 4], [43, 44, 45, 46, 47, 48],
    [5, 6, 7, 8], [49, 50, 51, 52, 53, 54], [9, 10, 11, 12],
    [13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28],
    [55, 56, 57, 58, 59, 60], [29, 30, 31, 32], [61, 62, 63, 64, 65, 66],
    [33, 34, 35, 36], [67, 68, 69, 70, 71, 72],
]


# --------------------------------------------------------------------------
# ports
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Port:
    """One manifold opening, with the in-plane direction of its gas."""
    key: str
    label: str
    side: str            # "anode" | "cathode"
    corner: str          # "bl" | "br" | "tl" | "tr"
    flow: str            # "in" | "out"
    #: slot rectangle in plate mm, (x, y, w, h); y grows downward from the
    #: top-left corner of the active area
    rect: Tuple[float, float, float, float]
    #: arrow (x_tail, x_tip, y) -- horizontal, along the flow axis
    arrow: Tuple[float, float, float]
    color: str


PORTS: Tuple[Port, ...] = (
    Port("h2_in",  "H₂ IN",       "anode",   "bl", "in",
         (8.0, 129.0, 62.0, 16.0), (-38.0, 2.0, 137.0), "#a5341f"),
    Port("h2_out", "H₂ OUT",      "anode",   "tr", "out",
         (182.0, -24.0, 62.0, 16.0), (250.0, 290.0, -16.0), "#a5341f"),
    Port("o2_in",  "AIR / O₂ IN", "cathode", "br", "in",
         (182.0, 129.0, 62.0, 16.0), (290.0, 250.0, 137.0), "#2c455d"),
    Port("o2_out", "O₂ OUT",      "cathode", "tl", "out",
         (8.0, -24.0, 62.0, 16.0), (2.0, -38.0, -16.0), "#2c455d"),
)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Segment:
    """A segment as the set of pads wired to it.

    `pads` are 1-based (col, row).  The shape is not assumed to be a
    rectangle: 40 of the 72 segments are staircases, and describing them as
    bounding boxes gets 60 of the 72 areas wrong.
    """
    wired: int
    topleft: int
    pads: Tuple[Tuple[int, int], ...]

    @property
    def n_pads(self) -> int:
        return len(self.pads)

    @property
    def area_cm2(self) -> float:
        return self.n_pads * (PAD_W_MM / 10.0) * (PAD_H_MM / 10.0)

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        cs = [c for c, _ in self.pads]
        rs = [r for _, r in self.pads]
        return min(cs), min(rs), max(cs), max(rs)

    @property
    def centroid_mm(self) -> Tuple[float, float]:
        cs = np.array([c for c, _ in self.pads], float)
        rs = np.array([r for _, r in self.pads], float)
        return (PAD_W_MM * (cs.mean() - 0.5), PAD_H_MM * (rs.mean() - 0.5))

    @property
    def label_point_mm(self) -> Tuple[float, float]:
        """Centre of the owned pad nearest the centroid.

        A concave segment need not contain its own centroid -- segment 67's
        falls on a pad belonging to 33 -- so a label placed at the centroid
        can land on the wrong tile.
        """
        cx = np.mean([c for c, _ in self.pads])
        cy = np.mean([r for _, r in self.pads])
        c, r = min(self.pads, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
        return ((c - 0.5) * PAD_W_MM, (r - 0.5) * PAD_H_MM)

    def rings_mm(self) -> List[List[Tuple[float, float]]]:
        """The true boundary, as one closed ring per connected loop.

        Every unit pad edge with copper on one side and nothing on the other
        is a boundary edge; chaining those counter-clockwise gives the outline
        exactly, staircases included.
        """
        pads = set(self.pads)
        edges: Dict[Tuple[float, float], Tuple[float, float]] = {}
        for c, r in pads:
            x0, y0 = (c - 1) * PAD_W_MM, (r - 1) * PAD_H_MM
            x1, y1 = c * PAD_W_MM, r * PAD_H_MM
            if (c, r - 1) not in pads:
                edges[(x0, y0)] = (x1, y0)
            if (c + 1, r) not in pads:
                edges[(x1, y0)] = (x1, y1)
            if (c, r + 1) not in pads:
                edges[(x1, y1)] = (x0, y1)
            if (c - 1, r) not in pads:
                edges[(x0, y1)] = (x0, y0)

        rings: List[List[Tuple[float, float]]] = []
        while edges:
            start = next(iter(edges))
            point, ring = start, []
            while point in edges:
                ring.append(point)
                point = edges.pop(point)
            ring.append(start)
            rings.append(_drop_collinear(ring))
        return rings


def _drop_collinear(ring: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    pts = list(ring[:-1]) if ring[0] == ring[-1] else list(ring)
    out = []
    for i, q in enumerate(pts):
        p, n = pts[i - 1], pts[(i + 1) % len(pts)]
        cross = (q[0] - p[0]) * (n[1] - q[1]) - (q[1] - p[1]) * (n[0] - q[0])
        if abs(cross) > 1e-9:
            out.append(q)
    out.append(out[0])
    return out


@dataclass
class Plate:
    """The whole plate: pad grid, segments, both numberings, ports."""

    segments: Dict[int, Segment] = field(default_factory=dict)

    # -- dimensions ---------------------------------------------------------
    w_mm: float = PAD_W_MM * N_COLS
    h_mm: float = PAD_H_MM * N_ROWS

    @property
    def area_cm2(self) -> float:
        return self.w_mm * self.h_mm / 100.0

    # -- numbering ----------------------------------------------------------
    def key(self, segment: int, scheme: str = "wired") -> int:
        return segment if scheme == "wired" else self.segments[segment].topleft

    def wired_of(self, topleft: int) -> int:
        for s in self.segments.values():
            if s.topleft == topleft:
                return s.wired
        raise KeyError(topleft)

    def renumbering(self) -> Dict[int, int]:
        """wired number -> top-left-down number."""
        return {s.wired: s.topleft for s in self.segments.values()}

    def order(self, scheme: str = "wired") -> List[int]:
        if scheme == "wired":
            return sorted(self.segments)
        return sorted(self.segments, key=lambda k: self.segments[k].topleft)

    # -- accessors ----------------------------------------------------------
    def areas_cm2(self) -> Dict[int, float]:
        return {k: s.area_cm2 for k, s in self.segments.items()}

    def centroids_mm(self) -> Dict[int, Tuple[float, float]]:
        return {k: s.centroid_mm for k, s in self.segments.items()}

    # -- integrity ----------------------------------------------------------
    def self_check(self) -> Dict[str, object]:
        """Do the segments tile the 900 pads exactly once?"""
        owner: Dict[Tuple[int, int], List[int]] = {}
        for k, s in self.segments.items():
            for pad in s.pads:
                owner.setdefault(pad, []).append(k)
        full = {(c, r) for c in range(1, N_COLS + 1) for r in range(1, N_ROWS + 1)}
        overlaps = {p: v for p, v in owner.items() if len(v) > 1}
        area_sum = sum(self.areas_cm2().values())
        return {
            "n_segments": len(self.segments),
            "pads_covered": len(owner),
            "pads_total": len(full),
            "uncovered": sorted(full - set(owner)),
            "overlapping": overlaps,
            "area_sum_cm2": area_sum,
            "cell_area_cm2": self.area_cm2,
            "area_closes": abs(area_sum - self.area_cm2) < 1e-9,
            "tiles_exactly": not overlaps and set(owner) == full,
        }

    def describe(self) -> str:
        a = list(self.areas_cm2().values())
        return (f"{len(self.segments)} segments, {self.w_mm:.1f} x {self.h_mm:.1f} mm "
                f"({self.area_cm2:.2f} cm2), areas {min(a):.3f}..{max(a):.3f} cm2")


def build_plate(matrix: Sequence[Sequence[int]] = PAD_MATRIX) -> Plate:
    pads: Dict[int, List[Tuple[int, int]]] = {}
    for r, row in enumerate(matrix, start=1):
        for c, value in enumerate(row, start=1):
            if value:
                pads.setdefault(int(value), []).append((c, r))

    # top-left-down numbering: down each full-height strip, strips left to right
    topleft: Dict[int, int] = {}
    n = 1
    for strip in STRIPS:
        by_row = sorted(strip, key=lambda k: np.mean([r for _, r in pads[k]]))
        for wired in by_row:
            topleft[wired] = n
            n += 1

    plate = Plate()
    for wired, ps in sorted(pads.items()):
        plate.segments[wired] = Segment(wired, topleft[wired],
                                        tuple(sorted(ps)))
    return plate


PLATE = build_plate()


# --------------------------------------------------------------------------
# a physically-shaped demonstration field
# --------------------------------------------------------------------------

def flow_coordinates(plate: Plate = PLATE) -> Dict[int, Tuple[float, float, float]]:
    """Per segment: (anode progress, cathode progress, edge distance), all 0..1.

    Anode progress runs 0 at the bottom-left H2 inlet to 1 at the top-right
    outlet; cathode progress runs 0 at the bottom-right air inlet to 1 at the
    top-left outlet.  Edge distance is 0 on the plate border and 1 well inside.
    """
    out = {}
    for k, s in plate.segments.items():
        cx, cy = s.centroid_mm
        u, v = cx / plate.w_mm, cy / plate.h_mm
        a = 0.72 * u + 0.28 * (1 - v)
        g = 0.72 * (1 - u) + 0.28 * (1 - v)
        e = min(min(u, 1 - u) * 3.2, min(v, 1 - v) * 3.2, 1.0)
        out[k] = (a, g, e)
    return out


def synthetic_fields(current_a: float = 150.0,
                     plate: Plate = PLATE) -> Dict[str, Dict[int, float]]:
    """Plausible counter-flow fields, for building and checking plots.

    NOT a measurement and not a simulation -- a smooth analytic stand-in with
    the right spatial structure, so the map can be exercised before real data
    is wired in.  Replace every one of these with pipeline output.
    """
    coords = flow_coordinates(plate)
    jbar = current_a / plate.area_cm2
    out = {k: {} for k in ("j", "T", "rh", "hfr", "rp", "lam", "sat")}
    for k, (a, g, e) in coords.items():
        noise = lambda i: (np.sin(k * 12.9898 + i * 78.233) * 43758.5453) % 1.0 - 0.5
        j = jbar * (1.30 - 0.34 * g - 0.11 * a) * (0.84 + 0.16 * e) * (1 + 0.035 * noise(1))
        T = 64 + 14.5 * (0.42 * a + 0.58 * g) + 9 * (j / jbar - 1) - 3.6 * (1 - e) + 0.4 * noise(2)
        rh = float(np.clip(36 + 60 * g + 4 * (1 - e) + 2.0 * noise(3), 24, 100))
        hfr = 46 + 52 * (1 - rh / 100) + 0.55 * (T - 70) + 1.4 * noise(4)
        sat = float(np.clip(0.04 + 0.62 * g * max(0.0, 1 - (T - 66) / 19)
                            + 0.05 * (1 - e) + 0.02 * noise(5), 0, 0.92))
        rp = 74 + 165 * g * g + 90 * max(0.0, sat - 0.28) + 4 * noise(6)
        lam = max(0.85, 2.05 - 1.12 * g + 0.03 * noise(7))
        for name, value in zip(out, (j, T, rh, hfr, rp, lam, sat)):
            out[name][k] = float(value)
    return out


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------

def robust_limits(values: Iterable[float], low: float = 2.0,
                  high: float = 98.0) -> Tuple[float, float]:
    """2-98 % percentiles.

    One runaway segment stretches a full-range scale until every real
    difference on the plate flattens into the same colour.
    """
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if v.size == 0:
        return 0.0, 1.0
    if v.size < 5:
        return float(v.min()), float(max(v.max(), v.min() + 1e-12))
    lo, hi = np.percentile(v, [low, high])
    if hi <= lo:
        lo, hi = float(v.min()), float(v.max())
    return float(lo), float(hi if hi > lo else lo + 1e-12)


def draw_plate(values: Mapping[int, float],
               label: str = "",
               unit: str = "",
               cmap: str = "inferno",
               scheme: str = "wired",
               side: str = "cathode",
               vmin: float | None = None,
               vmax: float | None = None,
               show_numbers: bool = True,
               show_channels: bool = True,
               field_alpha: float = 0.86,
               plate: Plate = PLATE,
               ax=None,
               figsize: Tuple[float, float] = (13.0, 8.2)):
    """Render the plate: metal body, sealing frame, ports, flow, field overlay.

    `values` is keyed by whichever numbering `scheme` names ("wired" or
    "topleft").  Missing or non-finite segments are drawn grey rather than
    left as a hole -- a hole in a heat map reads to the eye as a cold spot.

    Returns (fig, ax).
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize, to_rgb
    from matplotlib.patches import FancyArrow, FancyBboxPatch, Polygon, Rectangle

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    by_wired: Dict[int, float] = {}
    for k, s in plate.segments.items():
        key = k if scheme == "wired" else s.topleft
        v = values.get(key, np.nan)
        by_wired[k] = float(v) if v is not None else np.nan

    lo, hi = robust_limits(by_wired.values())
    lo = lo if vmin is None else vmin
    hi = hi if vmax is None else vmax
    norm = Normalize(lo, hi)
    mappable = ScalarMappable(norm=norm, cmap=cmap)

    W, H = plate.w_mm, plate.h_mm

    # -- plate body, frame, gasket -----------------------------------------
    ax.add_patch(FancyBboxPatch((-24, -28), 300, 177,
                                boxstyle="round,pad=0,rounding_size=7",
                                facecolor="#dcdfe3", edgecolor="#8b9097",
                                linewidth=1.0, zorder=0))
    ax.add_patch(Rectangle((-18.5, -22.5), 289, 166, facecolor="none",
                           edgecolor="#8b9097", linewidth=0.5, alpha=0.65, zorder=1))
    ax.add_patch(Rectangle((-6, -10), 264, 141, facecolor="#e6e8eb",
                           edgecolor="#6f747b", linewidth=0.8, hatch="////",
                           zorder=1))
    ax.add_patch(Rectangle((0, 0), W, H, facecolor="#a9afb6", edgecolor="none",
                           zorder=2))

    # milled channel texture, along the flow axis
    if show_channels:
        y = 0.0
        while y < H:
            ax.add_patch(Rectangle((0, y), W, 0.78, facecolor="#7f858c",
                                   edgecolor="none", zorder=3))
            y += 1.55

    # -- the field ----------------------------------------------------------
    for k in plate.order("wired"):
        seg = plate.segments[k]
        v = by_wired[k]
        colour = "#d4d4d7" if not np.isfinite(v) else mappable.to_rgba(v)
        for ring in seg.rings_mm():
            ax.add_patch(Polygon(ring, closed=True, facecolor=colour,
                                 edgecolor=(0.11, 0.12, 0.13, 0.55),
                                 linewidth=0.4, alpha=field_alpha, zorder=4))

    if show_numbers:
        for k, seg in plate.segments.items():
            lx, ly = seg.label_point_mm
            v = by_wired[k]
            rgb = to_rgb("#d4d4d7") if not np.isfinite(v) else mappable.to_rgba(v)[:3]
            lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
            ax.text(lx, ly, str(k if scheme == "wired" else seg.topleft),
                    ha="center", va="center", fontsize=6.4, zorder=6,
                    color="#141414" if lum > 0.42 else "#ffffff",
                    family="monospace", fontweight="bold")

    ax.add_patch(Rectangle((0, 0), W, H, facecolor="none", edgecolor="#5d5d60",
                           linewidth=1.0, zorder=7))

    # -- temperature sensors -----------------------------------------------
    for name, x in TEMP_SENSOR_X_MM.items():
        ax.plot([x, x], [-1.5, -9], color="#1d1f20", lw=0.8, zorder=8)
        ax.plot([x], [-10.6], marker="o", ms=4, mfc="#f2f2f3", mec="#1d1f20",
                mew=0.8, zorder=8)
        ax.text(x, -14.5, name.replace("temp", "T"), ha="center", va="bottom",
                fontsize=6.5, family="monospace", zorder=8)

    # -- ports and flow arrows ---------------------------------------------
    for port in PORTS:
        live = 1.0 if port.side == side else 0.3
        x, y, w, h = port.rect
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0,rounding_size=4",
                                    facecolor="#25282c", edgecolor=port.color,
                                    linewidth=1.4, alpha=live, zorder=9))
        x0, x1, ay = port.arrow
        ax.add_patch(FancyArrow(x0, ay, x1 - x0, 0, width=4.8,
                                head_width=12.4, head_length=9,
                                length_includes_head=True, alpha=live,
                                facecolor=port.color, edgecolor="none", zorder=9))
        tx = 50 if port.corner in ("tl", "bl") else 202
        ty = -34 if port.corner in ("tl", "tr") else 155
        ax.text(tx, ty, port.label, ha="left" if port.corner in ("tl", "bl") else "right",
                va="center", fontsize=11, color=port.color, alpha=live,
                fontweight="bold", zorder=9)

    # -- dimensions ---------------------------------------------------------
    ax.annotate("", xy=(0, 172), xytext=(W, 172),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color="#5d5d60"))
    ax.text(W / 2, 176, f"{W:.1f} mm · {N_COLS} pads", ha="center", va="top",
            fontsize=7.5, color="#5d5d60", family="monospace")
    ax.annotate("", xy=(288, 0), xytext=(288, H),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color="#5d5d60"))
    ax.text(292, H / 2, f"{H:.1f} mm · {N_ROWS} pads", ha="left", va="center",
            rotation=-90, fontsize=7.5, color="#5d5d60", family="monospace")

    bar = fig.colorbar(mappable, ax=ax, fraction=0.028, pad=0.045)
    bar.set_label(f"{label} [{unit}]" if unit else label, fontsize=10)
    bar.outline.set_linewidth(0.6)

    ax.set_xlim(-46, 312)
    ax.set_ylim(190, -58)          # y grows downward: row 1 at the top
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"{label} — {side} side, "
        + ("air enters bottom right, traverses right → left"
           if side == "cathode" else
           "H₂ enters bottom left, traverses left → right"),
        fontsize=11, loc="left", pad=12)
    return fig, ax


# --------------------------------------------------------------------------
# convenience: back to the 900-pad matrix, the way get_900_matrix does it
# --------------------------------------------------------------------------

def to_pad_matrix(values: Mapping[int, float], scheme: str = "wired",
                  plate: Plate = PLATE) -> np.ndarray:
    """A 20 x 45 array of the per-segment value painted onto every pad.

    The direct equivalent of `get_900_matrix(d, prop)`, for anyone who wants
    to `imshow` the plate rather than draw true segment outlines.
    """
    out = np.full((N_ROWS, N_COLS), np.nan)
    for k, seg in plate.segments.items():
        key = k if scheme == "wired" else seg.topleft
        v = values.get(key, np.nan)
        for c, r in seg.pads:
            out[r - 1, c - 1] = v
    return out


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print(PLATE.describe())
    check = PLATE.self_check()
    print(f"tiling: {check['pads_covered']}/{check['pads_total']} pads, "
          f"area {check['area_sum_cm2']:.2f}/{check['cell_area_cm2']:.2f} cm2 "
          f"[{'OK' if check['tiles_exactly'] and check['area_closes'] else 'PROBLEM'}]")
    print("wired -> top-left-down:", PLATE.renumbering())

    fields = synthetic_fields(150.0)
    draw_plate(fields["j"], label="Current density", unit="A/cm²",
               cmap="inferno", side="cathode")
    plt.tight_layout()
    plt.show()
