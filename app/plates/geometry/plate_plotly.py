"""The realistic plate, as a Plotly figure, for the Dash frontend.

Same drawing as `plate_model.draw_plate` but built with `plotly.graph_objects`
so it is interactive inside the app: hover a segment for its value, click one
to drive another panel.  It reads geometry from `plate_model`, so the pad map
lives in exactly one place.

    from plate_plotly import plate_figure

    fig = plate_figure(values, label="HFR", unit="mΩ·cm²", colorscale="Viridis")

Every colour, hover string and limit convention matches
`app/services/figures.py::plate_heatmap`, so the two can sit in the same app
without looking like different products.
"""

from __future__ import annotations

from typing import Mapping, Tuple

import numpy as np
import plotly.graph_objects as go

# Imported both ways: as part of the app package, and as a loose
# script (`python plate_viewer.py`) by someone who only wants the
# standalone HTML. A bare relative import breaks the second, a bare
# absolute one breaks the first.
try:
    from .plate_model import (N_COLS, N_ROWS, PLATE, PORTS, TEMP_SENSOR_X_MM, Plate,
                             robust_limits)
except ImportError:                       # pragma: no cover
    from plate_model import (N_COLS, N_ROWS, PLATE, PORTS, TEMP_SENSOR_X_MM, Plate,
                             robust_limits)

MISSING_COLOUR = "#d4d4d7"
PLATE_METAL = "#dcdfe3"
PLATE_EDGE = "#8b9097"
ACTIVE_FILL = "#a9afb6"
CHANNEL_INK = "rgba(90,96,103,0.55)"


def _ink_on(rgb: Tuple[float, float, float]) -> str:
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "rgba(20,20,20,0.88)" if lum > 0.42 else "rgba(255,255,255,0.95)"


def _sample(colorscale: str, t: float) -> str:
    from plotly.colors import sample_colorscale
    return sample_colorscale(colorscale, [float(np.clip(t, 0, 1))])[0]


def _rgb_tuple(colour: str) -> Tuple[float, float, float]:
    nums = [float(v) for v in colour[colour.index("(") + 1:colour.index(")")].split(",")[:3]]
    return tuple(v / 255.0 for v in nums)


def plate_figure(values: Mapping[int, float],
                 label: str = "",
                 unit: str = "",
                 colorscale: str = "Inferno",
                 scheme: str = "wired",
                 side: str = "cathode",
                 vmin: float | None = None,
                 vmax: float | None = None,
                 show_numbers: bool = True,
                 show_channels: bool = True,
                 field_opacity: float = 0.88,
                 plate: Plate = PLATE,
                 title: str = "") -> go.Figure:
    """One filled polygon per segment on a drawn bipolar plate."""
    by_wired = {}
    for k, s in plate.segments.items():
        key = k if scheme == "wired" else s.topleft
        v = values.get(key, np.nan)
        by_wired[k] = float(v) if v is not None else np.nan

    lo, hi = robust_limits(by_wired.values())
    lo = lo if vmin is None else vmin
    hi = hi if vmax is None else vmax
    if hi <= lo:
        hi = lo + 1e-12

    W, H = plate.w_mm, plate.h_mm
    fig = go.Figure()
    shapes, annotations = [], []

    # -- plate body, frame, gasket, active area ----------------------------
    shapes += [
        dict(type="rect", x0=-24, y0=-28, x1=276, y1=149, line=dict(color=PLATE_EDGE, width=1),
             fillcolor=PLATE_METAL, layer="below"),
        dict(type="rect", x0=-18.5, y0=-22.5, x1=270.5, y1=143.5,
             line=dict(color=PLATE_EDGE, width=0.6), fillcolor="rgba(0,0,0,0)", layer="below"),
        dict(type="rect", x0=-6, y0=-10, x1=258, y1=131, line=dict(color="#6f747b", width=0.9),
             fillcolor="#e6e8eb", layer="below"),
        dict(type="rect", x0=0, y0=0, x1=W, y1=H, line=dict(width=0),
             fillcolor=ACTIVE_FILL, layer="below"),
    ]

    if show_channels:
        y = 0.0
        while y < H:
            shapes.append(dict(type="line", x0=0, x1=W, y0=y, y1=y,
                               line=dict(color=CHANNEL_INK, width=1), layer="below"))
            y += 3.1

    # -- the field ---------------------------------------------------------
    for k in sorted(plate.segments):
        seg = plate.segments[k]
        v = by_wired[k]
        finite = np.isfinite(v)
        colour = _sample(colorscale, (v - lo) / (hi - lo)) if finite else MISSING_COLOUR

        xs, ys = [], []
        for ring in seg.rings_mm():
            if xs:
                xs.append(None); ys.append(None)
            xs += [p[0] for p in ring]
            ys += [p[1] for p in ring]

        shown = k if scheme == "wired" else seg.topleft
        other = seg.topleft if scheme == "wired" else k
        hover = (f"<b>segment {shown}</b><br>"
                 f"{label or 'value'}: "
                 + (f"{v:.4g} {unit}".strip() if finite else "not available")
                 + f"<br>{seg.area_cm2:.3f} cm² · {seg.n_pads} pads"
                 + f"<br>other numbering: #{other}<extra></extra>")

        fig.add_trace(go.Scatter(
            x=xs, y=ys, fill="toself", mode="lines",
            line=dict(color="rgba(29,31,32,0.55)", width=1),
            fillcolor=colour, opacity=field_opacity,
            customdata=[k] * len(xs), hoveron="fills",
            hovertemplate=hover, name=str(shown), showlegend=False))

        if show_numbers:
            lx, ly = seg.label_point_mm
            ink = _ink_on(_rgb_tuple(colour)) if finite else "rgba(20,20,20,0.88)"
            annotations.append(dict(x=lx, y=ly, text=str(shown), showarrow=False,
                                    font=dict(size=8, color=ink, family="ui-monospace, Menlo, monospace"),
                                    xanchor="center", yanchor="middle"))

    shapes.append(dict(type="rect", x0=0, y0=0, x1=W, y1=H,
                       line=dict(color="#5d5d60", width=1.2), fillcolor="rgba(0,0,0,0)"))

    # -- temperature sensors ------------------------------------------------
    for name, x in TEMP_SENSOR_X_MM.items():
        shapes.append(dict(type="line", x0=x, x1=x, y0=-1.5, y1=-9,
                           line=dict(color="#1d1f20", width=0.8)))
        shapes.append(dict(type="circle", x0=x - 1.7, x1=x + 1.7, y0=-12.3, y1=-8.9,
                           line=dict(color="#1d1f20", width=0.8), fillcolor="#f2f2f3"))
        annotations.append(dict(x=x, y=-16, text=name.replace("temp", "T"), showarrow=False,
                                font=dict(size=9, color="#1d1f20",
                                          family="ui-monospace, Menlo, monospace")))

    # -- ports and in-plane flow arrows -------------------------------------
    for port in PORTS:
        live = 1.0 if port.side == side else 0.3
        x, y, w, h = port.rect
        shapes.append(dict(type="rect", x0=x, y0=y, x1=x + w, y1=y + h,
                           line=dict(color=port.color, width=1.6),
                           fillcolor="#25282c", opacity=live))
        x0, x1, ay = port.arrow
        annotations.append(dict(x=x1, y=ay, ax=x0, ay=ay, xref="x", yref="y",
                                axref="x", ayref="y", text="", showarrow=True,
                                arrowhead=2, arrowsize=1.5, arrowwidth=4,
                                arrowcolor=port.color, opacity=live))
        left = port.corner in ("tl", "bl")
        annotations.append(dict(
            x=50 if left else 202, y=-38 if port.corner in ("tl", "tr") else 158,
            text=f"<b>{port.label}</b>", showarrow=False, opacity=live,
            xanchor="left" if left else "right",
            font=dict(size=13, color=port.color)))

    # -- colour bar, carried by an invisible marker trace --------------------
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
        marker=dict(colorscale=colorscale, cmin=lo, cmax=hi, size=0.1,
                    color=[lo], showscale=True,
                    colorbar=dict(title=dict(text=f"{label}<br>{unit}" if unit else label,
                                             side="right", font=dict(size=11)),
                                  thickness=14, len=0.8, outlinewidth=0.6,
                                  outlinecolor="#5d5d60", tickfont=dict(size=10)))))

    caption = ("air enters bottom right, traverses right → left, leaves top left"
               if side == "cathode" else
               "H₂ enters bottom left, traverses left → right, leaves top right")

    fig.update_layout(
        shapes=shapes, annotations=annotations,
        title=dict(text=f"{title or label} — {side} side · {caption}",
                   x=0, xanchor="left", font=dict(size=14)),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        margin=dict(l=20, r=20, t=52, b=20), height=560,
        xaxis=dict(range=[-46, 300], visible=False, constrain="domain"),
        yaxis=dict(range=[176, -52], visible=False,
                   scaleanchor="x", scaleratio=1),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#d7dbe0",
                        font=dict(size=11, color="#1f2328")))
    return fig


if __name__ == "__main__":
    from plate_model import synthetic_fields
    plate_figure(synthetic_fields(150.0)["j"], label="Current density",
                 unit="A/cm²", colorscale="Inferno").show()
