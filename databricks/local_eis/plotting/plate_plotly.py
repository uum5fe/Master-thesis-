#!/usr/bin/env python3
"""
plate_plotly.py
===============
Interactive (Plotly) plate heat map: the true segment outlines, the value
printed in every segment, hover on any pad for the segment's number, value
and class, and a dropdown to switch between parameters.

It replaces two drawings that could not be read the same way as the static
maps: gold's map_*.html (one marker per segment centroid on RdYlBu_r, so a
staircase segment showed as a dot and the colours did not match the PNGs)
and, by default, the external plate_viewer iframe in the runner, whose ramps
are its own. Colours, text sizes, the view direction and the inlet / outlet /
coolant labels all come from plate_style, exactly as in plate_figure.

The plate is drawn as a 45 x 20 pad grid (go.Heatmap): every pad takes the
value of the segment that owns it, so a staircase segment is its real shape
and hovering anywhere on it names it.

    import plate_plotly
    fig = plate_plotly.interactive_plate(
        [plate_plotly.Field("R_ohmic", "HFR (Rs)", "mΩ·cm²", values, classes)],
        title="2611976 / 45A")
    fig.write_html("plate.html", include_plotlyjs="cdn")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import plate_style as style
import r2d2_geometry as geom
from r2d2_geometry import PAD_W_MM, PAD_H_MM


@dataclass
class Field:
    """One parameter on the plate. `values`: segment -> value, display units."""
    key: str
    label: str
    unit: str
    values: dict
    classes: dict = field(default_factory=dict)
    decimals: int = 1
    limits: tuple[float, float] | None = None     # None: config / 5..95 %
    sd: dict = field(default_factory=dict)        # segment -> sd (optional)
    notes: dict = field(default_factory=dict)     # segment -> extra hover text


def _clean(values: dict) -> dict[str, float]:
    out = {}
    for k, v in (values or {}).items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            out[str(int(k)) if str(k).strip().isdigit() else str(k)] = f
    return out


def _limits(fd: Field, vals: dict[str, float]) -> tuple[float, float, str]:
    if fd.limits is not None:
        return float(fd.limits[0]), float(fd.limits[1]), "fixed scale"
    try:
        import config
        lim = config.heatmap_limits(fd.key)
    except Exception:                                       # noqa: BLE001
        lim = None
    if lim is not None:
        return float(lim[0]), float(lim[1]), "fixed scale"
    v = np.array(list(vals.values()), float)
    if v.size == 0:
        return 0.0, 1.0, "no data"
    lo, hi = (float(x) for x in np.percentile(v, (5, 95)))
    if hi - lo < 1e-12:
        pad = max(abs(lo) * 0.02, 1e-6)
        lo, hi = lo - pad, hi + pad
    return lo, hi, "auto scale (5..95 %)"


def _rgba(colorscale, t: float):
    """Colour of the scale at t in [0, 1] as an RGB(A) tuple in 0..1."""
    import plotly.colors as pc
    c = pc.sample_colorscale(colorscale, [min(max(t, 0.0), 1.0)])[0]
    r, g, b = (float(x) / 255.0 for x in c[c.index("(") + 1:-1].split(",")[:3])
    return r, g, b, 1.0


def _boundary_xy(p) -> tuple[list, list]:
    """Line segments between pads owned by different segments, + outline."""
    owner = {pad: n for n, seg in p.segments.items() for pad in seg.pads}
    xs, ys = [], []
    for (c, r), n in owner.items():
        x0, y0 = (c - 1) * PAD_W_MM, (r - 1) * PAD_H_MM
        if owner.get((c + 1, r)) != n:
            xs += [x0 + PAD_W_MM, x0 + PAD_W_MM, None]
            ys += [y0, y0 + PAD_H_MM, None]
        if owner.get((c, r + 1)) != n:
            xs += [x0, x0 + PAD_W_MM, None]
            ys += [y0 + PAD_H_MM, y0 + PAD_H_MM, None]
    W, H = geom.PLATE_W_MM, geom.PLATE_H_MM
    xs += [0, W, W, 0, 0, None]
    ys += [0, 0, H, H, 0, None]
    return xs, ys


def _label_xy(p, n: str, s) -> tuple[float, float, bool]:
    narrow = s.w_mm < 12.0
    lx = ((geom.LABEL_COL.get(n, 0) - 0.5) * PAD_W_MM
          if p is geom.ACTIVE_PLATE and n in geom.LABEL_COL else s.cx_mm)
    ly = ((geom.LABEL_ROW.get(n, 0) - 0.5) * PAD_H_MM
          if p is geom.ACTIVE_PLATE and n in geom.LABEL_ROW else s.cy_mm)
    if narrow:
        lx, ly = s.cx_mm, s.cy_mm
    return lx, ly, narrow


def _field_traces(fd: Field, p, show_colorbar: bool = True) -> list:
    """Heatmap + number labels + value labels for one field."""
    import plotly.graph_objects as go

    vals = _clean(fd.values)
    classes = {str(k): str(v) for k, v in (fd.classes or {}).items()}
    vmin, vmax, scale_note = _limits(fd, vals)
    cs = style.plotly_colorscale()
    u = f" {fd.unit}" if fd.unit else ""

    nr, nc = geom.N_ROWS, geom.N_COLS
    z = np.full((nr, nc), np.nan)
    cd = np.full((nr, nc), np.nan)
    for n, seg in p.segments.items():
        v = vals.get(n)
        for c, r in seg.pads:
            if v is not None:
                z[r - 1, c - 1] = v
            cd[r - 1, c - 1] = int(n)
    xc = (np.arange(nc) + 0.5) * PAD_W_MM
    yc = (np.arange(nr) + 0.5) * PAD_H_MM

    # per pad only the segment number (compact); the value is z
    heat = go.Heatmap(
        x=xc, y=yc, z=z, customdata=cd, colorscale=cs, zmin=vmin,
        zmax=vmax, hoverongaps=False, xgap=0, ygap=0, name=fd.label,
        hovertemplate=(f"<b>segment %{{customdata:.0f}}</b><br>{fd.label} = "
                       f"%{{z:.{fd.decimals}f}}{u}<extra></extra>"),
        showscale=show_colorbar,
        colorbar=dict(title=dict(text=f"{fd.label}<br>[{fd.unit}]"
                                 if fd.unit else fd.label, side="top"),
                      len=0.8, thickness=18,
                      tickfont=dict(size=11)))

    # labels: number small above, value large and bold below
    nx, ny, ntx, ncol = [], [], [], []
    vx, vy, vtx, vcol, vsz = [], [], [], [], []
    for n in sorted(p.segments, key=int):
        s = p.segments[n]
        v = vals.get(n)
        lx, ly, narrow = _label_xy(p, n, s)
        rgba = (style.MISSING_FILL if v is None else
                _rgba(cs, (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5))
        ink = style.ink(rgba)
        nx.append(lx)
        ny.append(ly + (style.NUMBER_DY_MM if v is not None else 0.0))
        ntx.append(n)
        ncol.append(ink)
        if v is not None:
            est = classes.get(n, "measured") not in ("", "measured")
            vx.append(lx)
            vy.append(ly + style.VALUE_DY_MM)
            vtx.append(f"<b>{v:.{fd.decimals}f}{'*' if est else ''}</b>")
            vcol.append(ink)
            vsz.append(10 if narrow else 13)
    numbers = go.Scatter(x=nx, y=ny, mode="text", text=ntx,
                         textfont=dict(size=8, color=ncol), hoverinfo="skip",
                         showlegend=False, name=f"{fd.key} numbers")
    values = go.Scatter(x=vx, y=vy, mode="text", text=vtx,
                        textfont=dict(size=vsz, color=vcol), hoverinfo="skip",
                        showlegend=False, name=f"{fd.key} values")
    heat.meta = dict(scale_note=scale_note, n=len(vals))
    return [heat, numbers, values]


def _subtitle(fd: Field) -> str:
    vals = _clean(fd.values)
    classes = {str(k): str(v) for k, v in (fd.classes or {}).items()}
    meas = np.array([v for k, v in vals.items()
                     if classes.get(k, "measured") in ("", "measured")], float)
    if meas.size < 2:
        return ""
    d = fd.decimals
    u = f" {fd.unit}" if fd.unit else ""
    n_est = len(vals) - meas.size
    return (f"{meas.size} measured" + (f", {n_est} rebuilt (value*)" if n_est
                                       else "")
            + f" · mean {meas.mean():.{d}f} · median {np.median(meas):.{d}f}"
            f" · sd {meas.std(ddof=1):.{d}f} · CV "
            f"{100 * meas.std(ddof=1) / abs(meas.mean()):.1f} %"
            f" · min..max {meas.min():.{d}f}..{meas.max():.{d}f}{u}")


def _end_annotations(W: float, H: float) -> list[dict]:
    """Inlet / outlet / coolant labels beside the two ends, with arrows
    pointing the way each stream goes ON SCREEN."""
    ann = []
    for end, x in (("x0", -4.0), ("xW", W + 4.0)):
        side = style.screen_side(end)
        xanchor = "right" if side == "left" else "left"
        for lab, role, col in style.end_streams()[end]:
            inlet = lab.endswith(" IN")
            # into the plate at an inlet, away from it at an outlet
            right = (side == "left") == inlet
            arrow = "→" if right else "←"
            # the arrow sits on the side of the label the stream moves to
            text = (f"<b>{lab} {arrow}</b>" if right
                    else f"<b>{arrow} {lab}</b>")
            y = H * {"top": 0.10, "mid": 0.5,
                     "bottom": 0.90}[style.port_slot(lab)]
            ann.append(dict(x=x, y=y, xref="x", yref="y", showarrow=False,
                            xanchor=xanchor, yanchor="middle",
                            text=f"{text}<br><span style='font-size:10px;"
                                 f"color:#5d5d60'>{role}</span>",
                            font=dict(size=13, color=col), align="center"))
    return ann


def interactive_plate(fields: list[Field], title: str = "",
                      subtitle: str = "", plate_name: str | None = None,
                      height: int = 700, width: int | None = 1400):
    """One Plotly figure; a dropdown switches between `fields`."""
    import plotly.graph_objects as go

    p = geom.plate(plate_name) if plate_name else geom.ACTIVE_PLATE
    W, H = geom.PLATE_W_MM, geom.PLATE_H_MM
    fields = [f for f in fields if _clean(f.values)]
    fig = go.Figure()
    bx, by = _boundary_xy(p)
    fig.add_trace(go.Scatter(x=bx, y=by, mode="lines", hoverinfo="skip",
                             line=dict(color="rgba(20,20,22,0.65)", width=1),
                             showlegend=False, name="segment outlines"))
    owner = ["outline"]
    notes = []
    for i, fd in enumerate(fields):
        trs = _field_traces(fd, p)
        notes.append(trs[0].meta["scale_note"])
        for tr in trs:
            tr.visible = i == 0
            fig.add_trace(tr)
            owner.append(fd.key)
    # the outline goes on top of the heat map
    fig.data = fig.data[1:] + fig.data[:1]
    owner = owner[1:] + owner[:1]

    def _title(i: int) -> str:
        fd = fields[i]
        head = (f"{title} — " if title else "") + fd.label
        sub = " | ".join(x for x in (_subtitle(fd), notes[i], subtitle,
                                     style.flow_note()) if x)
        return f"<b>{head}</b><br><sup>{sub}</sup>"

    buttons = [dict(label=fd.label, method="update",
                    args=[{"visible": [o in ("outline", fd.key)
                                       for o in owner]},
                          {"title.text": _title(i)}])
               for i, fd in enumerate(fields)]

    xr = [-48.0, W + 48.0]
    fig.update_layout(
        title=dict(text=_title(0) if fields else title, x=0.01,
                   xanchor="left"),
        xaxis=dict(range=xr[::-1] if style.mirrored() else xr,
                   visible=False, constrain="domain"),
        yaxis=dict(range=[H + 4.0, -4.0], visible=False,
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor="#f2f2f3", paper_bgcolor="white",
        height=height, width=width, margin=dict(l=10, r=10, t=110, b=10),
        annotations=_end_annotations(W, H),
        hoverlabel=dict(bgcolor="white", font_size=13),
        updatemenus=([dict(buttons=buttons, direction="down",
                           showactive=True, x=1.0, xanchor="right",
                           y=1.13, yanchor="top", bgcolor="white",
                           bordercolor="#999")] if len(fields) > 1 else []))
    return fig


def _png_uri(rgb: np.ndarray) -> str:
    import base64
    import io
    from matplotlib.image import imsave
    buf = io.BytesIO()
    imsave(buf, (np.clip(rgb, 0, 1) * 255).astype(np.uint8), format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def interpolated_plate(fd: Field, title: str = "", subtitle: str = "",
                       plate_name: str | None = None, height: int = 700,
                       width: int | None = 1400, gloss: float | None = None,
                       step_mm: float = 1.0):
    """The 2D spatial view of one field: linear interpolation between the
    segment centres, drawn with the static map's parula ramp and soft gloss
    (plate_figure.interpolate_field / shaded_rgb), a hollow square on every
    measured segment (hover: number and value), no numbers printed, and the
    same inlet / outlet / coolant labels as the per-segment view."""
    import plotly.graph_objects as go
    from matplotlib.colors import Normalize

    import plate_figure

    p = geom.plate(plate_name) if plate_name else geom.ACTIVE_PLATE
    W, H = geom.PLATE_W_MM, geom.PLATE_H_MM
    vals = _clean(fd.values)
    classes = {str(k): str(v) for k, v in (fd.classes or {}).items()}
    vmin, vmax, scale_note = _limits(fd, vals)
    cs = style.plotly_colorscale(style.INTERP_CMAP)
    u = f" {fd.unit}" if fd.unit else ""
    fig = go.Figure()
    got = plate_figure.interpolate_field(p, vals, W, H, step=step_mm)
    if got is not None:
        gx, gy, zi = got
        norm = Normalize(vmin=vmin, vmax=vmax)
        rgb = plate_figure.shaded_rgb(gx, gy, zi, style.mpl_cmap(style.INTERP_CMAP),
                                      norm, W, H, style.INTERP_GLOSS
                                      if gloss is None else gloss)
        # the shaded picture; the values are on the squares' hover
        fig.add_trace(go.Image(source=_png_uri(rgb), x0=0, y0=0,
                               dx=W / (rgb.shape[1] - 1),
                               dy=H / (rgb.shape[0] - 1), hoverinfo="skip"))
    # the colour bar, from an invisible two-point trace (a hidden heat map of
    # the whole field only for its colour bar would triple the size)
    fig.add_trace(go.Scatter(
        x=[-100, -100], y=[-100, -100], mode="markers", hoverinfo="skip",
        showlegend=False, marker=dict(
            color=[vmin, vmax], colorscale=cs, cmin=vmin, cmax=vmax, size=0.1,
            opacity=0, showscale=True,
            colorbar=dict(title=dict(text=f"{fd.label}<br>[{fd.unit}]"
                                     if fd.unit else fd.label, side="top"),
                          len=0.8, thickness=18, tickfont=dict(size=11)))))
    sx, sy, st, sc = [], [], [], []
    for n in sorted(p.segments, key=int):
        if n not in vals:
            continue
        s = p.segments[n]
        est = classes.get(n, "measured") not in ("", "measured")
        sx.append(s.cx_mm)
        sy.append(s.cy_mm)
        st.append(f"<b>segment {n}</b><br>{fd.label} = "
                  f"{vals[n]:.{fd.decimals}f}{u}<br>"
                  + (classes.get(n, "rebuilt") if est else "measured"))
        sc.append("rgba(194,24,91,0.55)" if est else style.MEASURED_MARK)
    fig.add_trace(go.Scatter(
        x=sx, y=sy, mode="markers", text=st, hoverinfo="text",
        marker=dict(symbol="square-open", size=11, color=sc,
                    line=dict(width=1.8)), showlegend=False,
        name="segments"))
    fig.add_shape(type="rect", x0=0, y0=0, x1=W, y1=H,
                  line=dict(color="#5d5d60", width=1.5))
    head = (f"{title} — " if title else "") + fd.label
    sub = " | ".join(x for x in (_subtitle(fd), scale_note, subtitle,
                                 "□ = segment measured; rest: 2D linear "
                                 "interpolation", style.flow_note()) if x)
    xr = [-48.0, W + 48.0]
    fig.update_layout(
        title=dict(text=f"<b>{head}</b><br><sup>{sub}</sup>", x=0.01,
                   xanchor="left"),
        xaxis=dict(range=xr[::-1] if style.mirrored() else xr,
                   visible=False, constrain="domain"),
        yaxis=dict(range=[H + 4.0, -4.0], visible=False,
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor="#f2f2f3", paper_bgcolor="white",
        height=height, width=width, margin=dict(l=10, r=10, t=110, b=10),
        annotations=_end_annotations(W, H),
        hoverlabel=dict(bgcolor="white", font_size=13))
    return fig
