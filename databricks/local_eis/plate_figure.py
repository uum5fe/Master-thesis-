#!/usr/bin/env python3
"""
plate_figure.py
===============
Static plate heat map in the style of the interactive plate viewer: the
metal frame with its bolts, the gasket border, the four manifold ports with
their flow arrows (in plate coordinates: O2 OUT top / H2 IN bottom at x = 0,
H2 OUT top / AIR IN bottom at x = 252), the coolant inlet and outlet on the
two ends, the temperature-sensor pins T1..T4 on the top edge, the plate
dimensions -- and every segment filled with its value.

Colours, text sizes and the view direction come from plate_style: plotly's
Jet scale for every parameter, the value large and the segment number small,
and config.PLATE_VIEW_MIRRORED to draw the plate from the other side (the
whole drawing mirrors, so the ports stay beside the segments they feed).

Why a static copy of the viewer: the viewer (plate_viewer, outside this
folder) is interactive HTML and scales each field to its own run. A report,
a thesis figure or a side-by-side of four operating points needs a PNG with
the SAME colour scale in every condition. This module draws that PNG from
r2d2_geometry's pad map (true staircase outlines) and takes its colour
limits from config.HEATMAP_LIMITS, so one colour means one value in every
condition.

    import plate_figure
    fig = plate_figure.draw_flow_plate(values, "R_ct", title="2612030 / 450A")
    fig.savefig("plate_R_ct.png", dpi=200, bbox_inches="tight")

`values` is segment -> value in DISPLAY units (mOhm*cm2 for resistances),
exactly what plate_summary.csv holds.
"""

from __future__ import annotations

import numpy as np

import plate_style as style
import r2d2_geometry as geom
from r2d2_geometry import PAD_W_MM, PAD_H_MM

# Colour ramps of the interactive viewer. No longer the default: every map is
# drawn in plate_style's Jet scale (the reference plots' colours). Kept so a
# caller can still pass cmap="viridis" etc. explicitly.
RAMPS = {
    "viridis": ["#440154", "#414487", "#2a788e", "#22a884", "#7ad151", "#fde725"],
    "inferno": ["#040414", "#420a68", "#932667", "#dd513a", "#fca50a", "#fcffa4"],
    "magma":   ["#040414", "#51127c", "#b73779", "#fc8961", "#fcfdbf"],
    "thermal": ["#313695", "#74add1", "#e0f3f8", "#ffffbf", "#fee090",
                "#f46d43", "#a50026"],
    "cividis": ["#00224e", "#35456c", "#666970", "#948e64", "#d9c65b", "#fee838"],
    "humid":   ["#ffffd9", "#c7e9b4", "#41b6c4", "#225ea8", "#081d58"],
}

# Parameter -> (label, unit, ramp, decimals). One ramp for all of them.
_JET = style.HEATMAP_CMAP
FIELDS = {
    "R_ohmic":     ("HFR (Rs)", "mΩ·cm²", _JET, 1),
    "R_ct":        ("R_ct (charge transfer)", "mΩ·cm²", _JET, 1),
    "R_mt":        ("R_mt (mass transport)", "mΩ·cm²", _JET, 1),
    "R_pol":       ("R_pol (total polarisation)", "mΩ·cm²", _JET, 1),
    "j_dc":        ("Current density", "A/cm²", _JET, 3),
    "T_degC":      ("Temperature", "°C", _JET, 2),
    "Z_mag_100Hz": ("|Z| at 100 Hz", "mΩ·cm²", _JET, 1),
    "phase_100Hz": ("Phase at 100 Hz", "°", _JET, 1),
}

# Frame geometry in plate mm (x right, y DOWN, active area 0..W x 0..H);
# the numbers are the viewer's, so the two drawings match.
_ANODE, _CATHODE = "#a5341f", "#2c455d"
PORTS = [  # label, side, corner, rect (x, y, w, h), arrow (x0, x1, y); plate mm
    ("H₂ IN", "anode", "bl", (8.0, 129.0, 62.0, 16.0), (-38.0, 2.0, 137.0)),
    ("H₂ OUT", "anode", "tr", (182.0, -24.0, 62.0, 16.0), (250.0, 290.0, -16.0)),
    ("AIR / O₂ IN", "cathode", "br", (182.0, 129.0, 62.0, 16.0), (290.0, 250.0, 137.0)),
    ("O₂ OUT", "cathode", "tl", (8.0, -24.0, 62.0, 16.0), (2.0, -38.0, -16.0)),
]


# Coolant ports, one on each end at mid-height, between the side bolts:
# rect (x, y, w, h) on the x = 0 end; the x = W end is its mirror image.
_COOLANT_RECT = (-21.0, 47.5, 13.0, 26.0)


def _cmap(name: str):
    from matplotlib.colors import LinearSegmentedColormap
    if name in RAMPS:
        return LinearSegmentedColormap.from_list(name, RAMPS[name])
    return style.mpl_cmap(name)


_ink = style.ink


def _limits(param, vals, limits):
    if limits is not None:
        return float(limits[0]), float(limits[1])
    try:
        import config
        lim = config.heatmap_limits(param)
    except Exception:                                       # noqa: BLE001
        lim = None
    if lim is not None:
        return lim
    v = np.array(list(vals.values()), float)
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = (float(x) for x in np.percentile(v, (5, 95)))
    return (lo, hi) if hi > lo else (lo - 1.0, hi + 1.0)


def draw_flow_plate(values: dict, param: str, title: str = "",
                    subtitle: str | None = None,
                    classes: dict | None = None,
                    limits: tuple[float, float] | None = None,
                    label: str | None = None, unit: str | None = None,
                    cmap: str | None = None, decimals: int | None = None,
                    side: str = "cathode", show_values: bool = True,
                    plate_name: str | None = None, figsize=(16, 9.4)):
    """One plate heat map in the viewer's layout. Returns the figure.

    values    segment -> value (display units); missing = grey
    param     field key (R_ohmic, R_ct, R_mt, R_pol, ...): picks label,
              ramp and the fixed colour scale from config.HEATMAP_LIMITS
    classes   segment -> "measured" | "substituted" | "inferred" | ...;
              anything but "measured" is hatched and its value starred
    limits    explicit (vmin, vmax); default = config.heatmap_limits(param)
    side      "cathode" or "anode": that side's ports are drawn solid, the
              other side's faded (as the viewer does)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon

    f_label, f_unit, f_ramp, f_dec = FIELDS.get(param, (param, "", "viridis", 1))
    label = label or f_label
    unit = f_unit if unit is None else unit
    cm = _cmap(cmap or f_ramp)
    dec = f_dec if decimals is None else decimals

    p = geom.plate(plate_name) if plate_name else geom.ACTIVE_PLATE
    W, H = geom.PLATE_W_MM, geom.PLATE_H_MM
    vals = {}
    for k, v in (values or {}).items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(fv):
            vals[str(int(k)) if str(k).strip().isdigit() else str(k)] = fv
    classes = {str(k): str(v) for k, v in (classes or {}).items()}
    vmin, vmax = _limits(param, vals, limits)
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("#f2f2f3")
    ax = fig.add_axes([0.02, 0.02, 0.84, 0.86])
    cax = fig.add_axes([0.885, 0.20, 0.018, 0.52])
    ax.set_facecolor("#f2f2f3")

    # ---- frame, gasket, active area --------------------------------------
    ax.add_patch(FancyBboxPatch((-24, -28), 300, 177,
                                boxstyle="round,pad=0,rounding_size=7",
                                fc="#e1e3e6", ec="#8b9097", lw=1.0, zorder=0))
    ax.add_patch(FancyBboxPatch((-18.5, -22.5), 289, 166,
                                boxstyle="round,pad=0,rounding_size=4",
                                fc="none", ec="#8b9097", lw=0.6, alpha=0.65,
                                zorder=0.5))
    ax.add_patch(FancyBboxPatch((-6, -10), 264, 141,
                                boxstyle="round,pad=0,rounding_size=3",
                                fc="#e8e9eb", ec="#6f747b", lw=0.8,
                                hatch="////", zorder=1))
    ax.add_patch(Rectangle((0, 0), W, H, fc="#b0b6bd", ec="none", zorder=1.5))

    # ---- segments ----------------------------------------------------------
    for n in sorted(p.segments, key=int):
        s = p.segments[n]
        v = vals.get(n)
        face = (0.83, 0.83, 0.84, 1.0) if v is None else cm(norm(v))
        est = v is not None and classes.get(n, "measured") not in ("", "measured")
        for row, c0, c1 in s.runs:
            ax.add_patch(Rectangle(((c0 - 1) * PAD_W_MM, (row - 1) * PAD_H_MM),
                                   (c1 - c0 + 1) * PAD_W_MM, PAD_H_MM,
                                   fc=face, ec=((1, 1, 1, 0.8) if est else face),
                                   lw=0.0, hatch=("////" if est else None),
                                   alpha=0.95, zorder=2))
    owner = {pad: n for n, seg in p.segments.items() for pad in seg.pads}
    for (c, r), n in owner.items():
        x0, y0 = (c - 1) * PAD_W_MM, (r - 1) * PAD_H_MM
        if owner.get((c + 1, r)) != n:
            ax.plot([x0 + PAD_W_MM] * 2, [y0, y0 + PAD_H_MM],
                    color=(0.11, 0.12, 0.13, 0.55), lw=0.7, zorder=3,
                    solid_capstyle="butt")
        if owner.get((c, r + 1)) != n:
            ax.plot([x0, x0 + PAD_W_MM], [y0 + PAD_H_MM] * 2,
                    color=(0.11, 0.12, 0.13, 0.55), lw=0.7, zorder=3,
                    solid_capstyle="butt")
    ax.add_patch(Rectangle((0, 0), W, H, fc="none", ec="#5d5d60", lw=1.3,
                           zorder=4))

    # labels: number (bold) and value underneath
    for n in sorted(p.segments, key=int):
        s = p.segments[n]
        v = vals.get(n)
        face = (0.83, 0.83, 0.84, 1.0) if v is None else cm(norm(v))
        tc = _ink(face)
        narrow = s.w_mm < 12.0
        lx = ((geom.LABEL_COL.get(n, 0) - 0.5) * PAD_W_MM
              if p is geom.ACTIVE_PLATE and n in geom.LABEL_COL else s.cx_mm)
        ly = ((geom.LABEL_ROW.get(n, 0) - 0.5) * PAD_H_MM
              if p is geom.ACTIVE_PLATE and n in geom.LABEL_ROW else s.cy_mm)
        if narrow:
            lx, ly = s.cx_mm, s.cy_mm
        # the value is what the map is for: large and bold; the segment
        # number small above it
        with_val = show_values and v is not None
        ax.text(lx, ly + (style.NUMBER_DY_MM if with_val else 0), n,
                ha="center", va="center",
                fontsize=(style.NUMBER_FONT_NARROW if narrow
                          else style.NUMBER_FONT),
                family="monospace", color=tc, alpha=0.9, zorder=6)
        if with_val:
            est = classes.get(n, "measured") not in ("", "measured")
            ax.text(lx, ly + style.VALUE_DY_MM,
                    f"{v:.{dec}f}" + ("*" if est else ""),
                    ha="center", va="center",
                    fontsize=(style.VALUE_FONT_NARROW if narrow
                              else style.VALUE_FONT),
                    fontweight="bold", color=tc, zorder=6)

    # ---- temperature sensors ----------------------------------------------
    for i, (name, x) in enumerate(sorted(geom.TEMP_SENSOR_X_MM.items(),
                                         key=lambda kv: kv[1])):
        ax.plot([x, x], [-9, -1.5], color="#1d1f20", lw=0.9, zorder=5)
        ax.add_patch(Circle((x, -10.6), 1.7, fc="#f2f2f3", ec="#1d1f20",
                            lw=0.9, zorder=5))
        # T1 sits under the O2 OUT port, so its tag goes beside the pin
        first = i == 0
        ax.text(x - 3.2 if first else x, -10.6 if first else -15.2,
                f"T{i + 1}", ha=style.outward_ha("x0") if first else "center",
                va="center",
                fontsize=6.5, family="monospace", fontweight="bold",
                color="#1d1f20", zorder=7)

    # ---- ports and flow arrows ---------------------------------------------
    def arrow(x0, x1, y, w=2.4, hw=6.2, hl=9.0):
        d = 1 if x1 > x0 else -1
        n_ = x1 - d * hl
        return [(x0, y - w), (n_, y - w), (n_, y - hw), (x1, y), (n_, y + hw),
                (n_, y + w), (x0, y + w)]

    for lab, sd, corner, (x, y, w, h), (a0, a1, ay) in PORTS:
        live = 1.0 if sd == side else 0.3
        col = _ANODE if sd == "anode" else _CATHODE
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0,rounding_size=4",
                                    fc="#3b3f44", ec=col, lw=1.6, alpha=live,
                                    zorder=5))
        ax.add_patch(FancyBboxPatch((x + 2, y + 2), w - 4, h - 4,
                                    boxstyle="round,pad=0,rounding_size=3",
                                    fc="#25282c", ec="none", alpha=live,
                                    zorder=5))
        for bx in (x + 13, x + w - 13):
            ax.add_patch(Circle((bx, y + h / 2), 2.6, fc="#b7b7ba",
                                ec="#7a7a7d", lw=0.5, alpha=live, zorder=6))
            ax.add_patch(Circle((bx, y + h / 2), 1.5, fc="#4a4d52",
                                alpha=live, zorder=6))
        ax.add_patch(Polygon(arrow(a0, a1, ay), closed=True, fc=col,
                             ec="none", alpha=live, zorder=5))
        top = corner[0] == "t"
        ax.text(x + w / 2, (y - 12 if top else y + h + 12), lab,
                ha="center", va="center", fontsize=12.5,
                fontweight="bold", color=col, alpha=live, zorder=6)
        ax.text(x + w / 2, (y - 7.5 if top else y + h + 16.8),
                ("anode " if sd == "anode" else "cathode ")
                + ("inlet" if "IN" in lab else "outlet"),
                ha="center", va="center", fontsize=6.5, family="monospace",
                color="#5d5d60", alpha=live, zorder=6)

    # ---- coolant ports (config.COOLANT_INLET_END) -------------------------
    cx, cy, cw, ch = _COOLANT_RECT
    for end, (lab, role, col) in (
            (e, st) for e, sts in style.end_streams().items() for st in sts
            if st[0].startswith("COOLANT")):
        x = cx if end == "x0" else W - cx - cw
        ax.add_patch(FancyBboxPatch((x, cy), cw, ch,
                                    boxstyle="round,pad=0,rounding_size=3",
                                    fc="#3b3f44", ec=col, lw=1.6, zorder=5))
        ax.add_patch(FancyBboxPatch((x + 2, cy + 2), cw - 4, ch - 4,
                                    boxstyle="round,pad=0,rounding_size=2",
                                    fc="#25282c", ec="none", zorder=5))
        # arrow along x, outside the frame: into the port at the inlet, out
        # of it at the outlet
        out_x = -46.0 if end == "x0" else W + 46.0
        port_x = x if end == "x0" else x + cw
        a0, a1 = (out_x, port_x) if lab.endswith(" IN") else (port_x, out_x)
        ax.add_patch(Polygon(arrow(a0, a1, cy + ch / 2), closed=True, fc=col,
                             ec="none", zorder=5))
        tx = (out_x + port_x) / 2
        ax.text(tx, cy - 6.5, lab, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=col, zorder=6)
        ax.text(tx, cy + ch + 5.5, role, ha="center", va="center",
                fontsize=6.5, family="monospace", color="#5d5d60", zorder=6)

    # ---- bolts ----------------------------------------------------------------
    for i in range(10):
        for by in (-19.5, 140.5):
            bx = -14 + i * 31.1
            if any(px - 2 < bx < px + pw + 2 and py - 2 < by < py + ph + 2
                   for _, _, _, (px, py, pw, ph), _ in PORTS):
                continue
            ax.add_patch(Circle((bx, by), 2.6, fc="#b7b7ba", ec="#7a7a7d",
                                lw=0.5, zorder=5))
            ax.add_patch(Circle((bx, by), 1.5, fc="#4a4d52", zorder=5))
    for i in range(4):
        for bx in (-14, 266):
            ax.add_patch(Circle((bx, 6 + i * 36.3), 2.6, fc="#b7b7ba",
                                ec="#7a7a7d", lw=0.5, zorder=5))
            ax.add_patch(Circle((bx, 6 + i * 36.3), 1.5, fc="#4a4d52",
                                zorder=5))

    # ---- dimensions ---------------------------------------------------------
    ax.text(W / 2, 172, f"{W:.1f} × {H:.1f} mm · 45 × 20 pads   |   "
            f"{style.flow_note()}",
            ha="center", va="center", fontsize=7.5, family="monospace",
            color="#5d5d60")

    ax.set_xlim(*style.xlim(-50, W + 50))
    ax.set_ylim(177, -44)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- title + statistics ------------------------------------------------
    v = np.array(list(vals.values()), float)
    meas = np.array([x for k, x in vals.items()
                     if classes.get(k, "measured") in ("", "measured")], float)
    if subtitle is None and meas.size:
        subtitle = (f"{meas.size} measured"
                    + (f", {v.size - meas.size} rebuilt (hatched, value*)"
                       if v.size > meas.size else "")
                    + f"   ·   mean {meas.mean():.{dec}f}   median "
                      f"{np.median(meas):.{dec}f}   sd {meas.std(ddof=1):.{dec}f}"
                      f"   CV {100 * meas.std(ddof=1) / abs(meas.mean()):.1f} %"
                      f"   min..max {meas.min():.{dec}f}..{meas.max():.{dec}f} {unit}")
    fig.text(0.03, 0.965, (title + " — " if title else "") + label,
             fontsize=17, fontweight="bold", color="#1d1f20", va="top")
    if subtitle:
        fig.text(0.03, 0.925, subtitle, fontsize=10, color="#5d5d60", va="top")

    # ---- colour bar (fixed scale) -------------------------------------------
    from matplotlib.cm import ScalarMappable
    below = bool(v.size and v.min() < vmin - 1e-12)
    above = bool(v.size and v.max() > vmax + 1e-12)
    sm = ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cax, extend=("both" if below and above else
                                           "min" if below else "max" if above
                                           else "neither"))
    cb.set_ticks(np.linspace(vmin, vmax, 5))
    cb.ax.tick_params(labelsize=10)
    cb.outline.set_linewidth(0.6)
    cax.set_title(f"{label.split(' (')[0]}\n[{unit}]", fontsize=10, pad=10,
                  loc="left")
    fig.text(0.885, 0.16, "fixed scale" if limits is not None or _fixed(param)
             else "auto scale (5..95 %)", fontsize=8.5, color="#5d5d60")
    return fig


def _fixed(param) -> bool:
    try:
        import config
        return config.heatmap_limits(param) is not None
    except Exception:                                       # noqa: BLE001
        return False


def __getattr__(name):
    # FLOW_NOTE used to be a module constant from config.FLOW_DESCRIPTION;
    # it now follows the view switch, so it is computed when asked for.
    if name == "FLOW_NOTE":
        return style.flow_note()
    raise AttributeError(name)


def write_condition_maps(summary_csv, out_dir, title: str = "",
                         params=("R_ohmic", "R_ct", "R_mt", "R_pol"),
                         dpi: int = 200, side: str = "cathode") -> dict:
    """Draw plate_<param>.png for every param from a gold plate_summary.csv."""
    import csv
    from pathlib import Path
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(summary_csv, newline="", encoding="utf-8")))
    classes = {r["segment"]: r.get("class", "measured") for r in rows}
    out = {}
    for prm in params:
        vals = {r["segment"]: r[prm] for r in rows if r.get(prm) not in (None, "")}
        if not vals:
            continue
        fig = draw_flow_plate(vals, prm, title=title, classes=classes, side=side)
        path = Path(out_dir) / f"plate_{prm}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        out[prm] = str(path)
    return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python plate_figure.py <plate_summary.csv> <out_dir> [title]")
        raise SystemExit(2)
    for k, v in write_condition_maps(sys.argv[1], sys.argv[2],
                                     " ".join(sys.argv[3:])).items():
        print(f"  {k}: {v}")
