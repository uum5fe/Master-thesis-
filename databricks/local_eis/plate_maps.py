#!/usr/bin/env python3
"""
plate_maps.py
=============
Static plate heat maps with the VALUE written inside every segment.

WHY A SECOND PLATE DRAWING
--------------------------
The runner's maps came from `plate_model.draw_plate`, which labels each
segment with its NUMBER only -- reading a value meant matching a colour
against the colour bar by eye. And on a colour bar that runs from the plate
minimum to the plate maximum (gold.plate_heatmap does exactly that), a map as
flat as the HFR one -- most segments within a few mOhm*cm2 of each other --
has its scale stretched by a single low or high segment, and everything else
collapses into two or three indistinguishable shades. That is why the
mass-transport map "showed" its gradient and the HFR map did not: R_mt
spreads over its whole range, R_ohmic sits in a narrow band with a couple of
outliers.

This module draws from r2d2_geometry's pad map (the true staircase outlines,
not bounding boxes) and fixes both:

  * every segment carries its number AND its value, in a text colour chosen
    against the fill so it stays legible on dark and light cells;
  * the colour scale spans a ROBUST range (5th..95th percentile by default)
    of the segments on the map, with arrow ends on the colour bar when a
    value falls outside it. The outliers stay on the map, in the end colour,
    with their true value written on them -- nothing is clipped out of the
    numbers, only out of the colour stretch;
  * the uniformity statistics (mean, median, sd, CV, min..max) are printed
    under the title, so "how uniform is it" is a number and not a judgement
    about colours.

Rebuilt / inferred segments (class != "measured") are hatched, so an
estimate is never mistaken for a measurement.

The orientation is the one `r2d2_geometry.plot_map` uses (and the numbering
cell of the runner shows): pad row 1 at the top.
"""

from __future__ import annotations

import numpy as np

import r2d2_geometry as geom
from r2d2_geometry import PAD_W_MM, PAD_H_MM, PLATE_W_MM, PLATE_H_MM

try:
    from config import FLOW_ARRANGEMENT, FLOW_DESCRIPTION
except Exception:                                           # noqa: BLE001
    FLOW_ARRANGEMENT, FLOW_DESCRIPTION = "unknown", {"unknown": ""}


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


def uniformity_stats(values: dict) -> dict:
    """n, mean, median, sd, CV [%], min, max, span relative to the mean [%]."""
    v = np.array(list(_finite(values).values()), float)
    if v.size == 0:
        return dict(n=0, mean=np.nan, median=np.nan, sd=np.nan, cv_pct=np.nan,
                    min=np.nan, max=np.nan, span_pct=np.nan)
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
    return dict(n=int(v.size), mean=mean, median=float(np.median(v)), sd=sd,
                cv_pct=100.0 * sd / abs(mean) if mean else np.nan,
                min=float(v.min()), max=float(v.max()),
                span_pct=100.0 * float(np.ptp(v)) / abs(mean) if mean else np.nan)


def robust_limits(values: dict, pct=(5.0, 95.0)) -> tuple[float, float]:
    """Colour limits from percentiles, widened so a flat plate is not noise."""
    v = np.array(list(_finite(values).values()), float)
    if v.size == 0:
        return 0.0, 1.0
    if pct is None or v.size < 5:
        lo, hi = float(v.min()), float(v.max())
    else:
        lo, hi = (float(x) for x in np.percentile(v, pct))
    if hi - lo < 1e-12:
        pad = max(abs(lo) * 0.02, 1e-6)
        lo, hi = lo - pad, hi + pad
    return lo, hi


def _text_colour(rgba) -> str:
    r, g, b = rgba[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if lum > 0.55 else "white"


def draw_value_map(values: dict, label: str, unit: str = "",
                   cmap: str = "magma", decimals: int = 1,
                   title: str | None = None, classes: dict | None = None,
                   pct=(5.0, 95.0), limits: tuple[float, float] | None = None,
                   plate_name: str | None = None, figsize=(15, 7.8),
                   show_stats: bool = True):
    """Plate map, one colour and one printed value per segment.

    values   segment -> value (already in display units)
    classes  segment -> "measured" | "substituted" | "inferred" | ...
    pct      percentile range for the colour scale; None = min..max
    limits   explicit (vmin, vmax), overrides pct
    Returns (fig, ax).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import Rectangle

    p = geom.plate(plate_name) if plate_name else geom.ACTIVE_PLATE
    segments = p.segments
    vals = _finite(values)
    classes = {str(k): str(v) for k, v in (classes or {}).items()}

    vmin, vmax = limits if limits is not None else robust_limits(vals, pct)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cm = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for n in sorted(segments, key=int):
        s = segments[n]
        v = vals.get(n)
        face = (0.86, 0.86, 0.86, 1.0) if v is None else cm(norm(v))
        cls = classes.get(n, "measured")
        estimated = v is not None and cls not in ("", "measured")
        for row, c0, c1 in s.runs:
            ax.add_patch(Rectangle(((c0 - 1) * PAD_W_MM, (row - 1) * PAD_H_MM),
                                   (c1 - c0 + 1) * PAD_W_MM, PAD_H_MM,
                                   facecolor=face,
                                   # the hatch is drawn in the edge colour
                                   edgecolor=((1, 1, 1, 0.75) if estimated
                                              else face), lw=0.0,
                                   hatch=("////" if estimated else None),
                                   zorder=2))
        narrow = s.w_mm < 12.0
        tc = "0.35" if v is None else _text_colour(face)
        # the label sits on a pad the segment owns -- on a staircase the
        # centroid itself can fall in a neighbour
        lx = (geom.LABEL_COL.get(n, 0) - 0.5) * PAD_W_MM \
            if p is geom.ACTIVE_PLATE and n in geom.LABEL_COL else s.cx_mm
        ly = (geom.LABEL_ROW.get(n, 0) - 0.5) * PAD_H_MM \
            if p is geom.ACTIVE_PLATE and n in geom.LABEL_ROW else s.cy_mm
        if narrow:
            lx, ly = s.cx_mm, s.cy_mm
        ax.text(lx, ly - (1.9 if v is not None else 0), n, ha="center",
                va="center", fontsize=(5.5 if narrow else 6.5), color=tc,
                alpha=0.85, zorder=6)
        if v is not None:
            ax.text(lx, ly + 1.6, f"{v:.{decimals}f}"
                    + ("*" if estimated else ""),
                    ha="center", va="center",
                    fontsize=(6.3 if narrow else 8.6), fontweight="bold",
                    color=tc, zorder=6)

    # segment outlines: an edge between two pads owned by different segments
    owner = {pad: n for n, seg in segments.items() for pad in seg.pads}
    for (c, r), n in owner.items():
        x0, y0 = (c - 1) * PAD_W_MM, (r - 1) * PAD_H_MM
        if owner.get((c + 1, r)) != n:
            ax.plot([x0 + PAD_W_MM] * 2, [y0, y0 + PAD_H_MM], color="white",
                    lw=1.1, zorder=4, solid_capstyle="butt")
        if owner.get((c, r + 1)) != n:
            ax.plot([x0, x0 + PAD_W_MM], [y0 + PAD_H_MM] * 2, color="white",
                    lw=1.1, zorder=4, solid_capstyle="butt")
    ax.add_patch(Rectangle((0, 0), PLATE_W_MM, PLATE_H_MM, facecolor="none",
                           edgecolor="#444", lw=1.6, zorder=5))

    ax.set_xlim(-2, PLATE_W_MM + 2)
    ax.set_ylim(PLATE_H_MM + 2, -2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.set_xlabel(FLOW_DESCRIPTION.get(FLOW_ARRANGEMENT, ""), fontsize=10,
                  color="#555")

    fin = np.array(list(vals.values()), float)
    below = bool(fin.size and fin.min() < vmin - 1e-12)
    above = bool(fin.size and fin.max() > vmax + 1e-12)
    extend = ("both" if below and above else "min" if below
              else "max" if above else "neither")
    sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.015, extend=extend)
    cb.set_label(label + (f"  [{unit}]" if unit else ""), fontsize=11)

    head = title or label
    if show_stats:
        st = uniformity_stats(vals)
        u = f" {unit}" if unit else ""
        scale = ("min..max" if (limits is None and pct is None) else
                 "fixed" if limits is not None else
                 f"{pct[0]:g}th..{pct[1]:g}th percentile")
        n_est = sum(1 for k in vals if classes.get(k, "measured")
                    not in ("", "measured"))
        head += (f"\nn = {st['n']}"
                 + (f" ({n_est} rebuilt, hatched, value marked *)" if n_est
                    else "")
                 + f"   mean {st['mean']:.{decimals}f}{u}"
                 f"   median {st['median']:.{decimals}f}{u}"
                 f"   sd {st['sd']:.{decimals}f}{u}"
                 f"   CV {st['cv_pct']:.1f} %"
                 f"   min..max {st['min']:.{decimals}f}..{st['max']:.{decimals}f}{u}"
                 f"\ncolour scale: {scale}"
                 + ("; values beyond it keep the end colour and are printed "
                    "as measured" if extend != "neither" else ""))
    ax.set_title(head, fontsize=11.5, loc="left")
    fig.tight_layout()
    return fig, ax
