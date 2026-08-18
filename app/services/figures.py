"""Plotly figures: plate maps, Nyquist, Bode, residuals, comparisons.

The plate map draws every segment as its own filled polygon rather than as a
cell in a raster image.  That costs a few lines but buys three things the
pipeline's static PNGs cannot give: the segments keep their true, unequal
shapes; each one carries its own hover text and click payload, so selecting a
segment on the map can drive the spectrum next to it; and a segment that was
inferred rather than measured can be drawn with a pattern fill instead of being
silently coloured like a real measurement.

A hole in a heat map reads to the eye as a cold spot, so nothing is ever left
blank: unmeasurable segments are hatched and named in the hover.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots

from app.data.model import PARAM_META, param_label
from app.plates.registry import PlateGeometry

#: Fill pattern per segment class.  ``measured`` is solid; everything else is
#: visibly not a measurement.
CLASS_PATTERN = {
    "measured": None,
    "inferred": "/",
    "bad": "x",
}
CLASS_OPACITY = {"measured": 0.95, "inferred": 0.55, "bad": 0.30}

TEMPLATE = "plotly_white"
MISSING_COLOUR = "#d9d9d9"


def _colour_for(value, vmin, vmax, colorscale) -> str:
    if value is None or not np.isfinite(value):
        return MISSING_COLOUR
    if vmax <= vmin:
        return sample_colorscale(colorscale, [0.5])[0]
    t = float(np.clip((value - vmin) / (vmax - vmin), 0.0, 1.0))
    return sample_colorscale(colorscale, [t])[0]


def _robust_limits(values, low: float = 2.0, high: float = 98.0):
    finite = np.array([v for v in values if v is not None and np.isfinite(v)], float)
    if finite.size == 0:
        return 0.0, 1.0
    if finite.size < 5:
        return float(finite.min()), float(max(finite.max(), finite.min() + 1e-12))
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        hi = lo + 1e-12
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# plate map
# ---------------------------------------------------------------------------

def plate_heatmap(
    geom: PlateGeometry,
    values: dict[str, float],
    param: str,
    sds: dict[str, float] | None = None,
    classes: dict[str, str] | None = None,
    faults: dict[str, str] | None = None,
    vmin: float | None = None, vmax: float | None = None,
    robust: bool = True,
    show_labels: bool = True,
    title: str = "",
) -> go.Figure:
    """One filled polygon per segment, in true plate geometry."""
    sds = sds or {}
    classes = classes or {}
    faults = faults or {}
    meta = PARAM_META.get(param, {})
    colorscale = meta.get("colorscale", "RdYlBu_r")

    order = geom.order()
    if vmin is None or vmax is None:
        lo, hi = (_robust_limits([values.get(s) for s in order])
                  if robust else
                  (min((v for v in values.values() if np.isfinite(v)), default=0.0),
                   max((v for v in values.values() if np.isfinite(v)), default=1.0)))
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax

    fig = go.Figure()

    for name in order:
        x0, y0, x1, y1 = geom.bounds_mm(name)
        value = values.get(name, float("nan"))
        cls = classes.get(name, "measured")
        if name in geom.known_bad and name not in classes:
            cls = "bad"
        sd = sds.get(name)
        colour = _colour_for(value, vmin, vmax, colorscale)

        text = [f"<b>segment {name}</b>"]
        if np.isfinite(value):
            unit = meta.get("unit", "")
            line = f"{meta.get('label', param)}: {value:.4g} {unit}".strip()
            if sd is not None and np.isfinite(sd):
                line += f"  (± {sd:.3g})"
            text.append(line)
        else:
            text.append(f"{meta.get('label', param)}: not available")
        text.append(f"area {geom.area_cm2(name):.3f} cm²")
        text.append(f"class: {cls}")
        if faults.get(name):
            text.append(f"fault: {faults[name]}")
        if name in geom.known_bad:
            text.append(f"known fault: {geom.known_bad[name]}")

        fig.add_trace(go.Scatter(
            x=[x0, x1, x1, x0, x0], y=[y0, y0, y1, y1, y0],
            fill="toself", mode="lines",
            line=dict(color="rgba(60,60,60,0.65)", width=1),
            fillcolor=colour,
            opacity=CLASS_OPACITY.get(cls, 0.9),
            fillpattern=dict(shape=CLASS_PATTERN.get(cls) or "",
                             fgcolor="rgba(40,40,40,0.55)", size=6),
            name=str(name), customdata=[name] * 5,
            hoveron="fills",
            hovertemplate="<br>".join(text) + "<extra></extra>",
            showlegend=False,
        ))

    if show_labels:
        cx, cy, labels = [], [], []
        for name in order:
            x, y = geom.centroid_mm(name)
            cx.append(x); cy.append(y); labels.append(name)
        fig.add_trace(go.Scatter(
            x=cx, y=cy, mode="text", text=labels,
            textfont=dict(size=8, color="rgba(20,20,20,0.75)"),
            hoverinfo="skip", showlegend=False,
        ))

    # Flow channel separators and the gas inlet marker: a map of a fuel cell
    # plate is not readable without knowing which way the gas runs.
    for y in geom.flow_channel_y_mm:
        fig.add_shape(type="line", x0=0, x1=geom.plate_w_mm, y0=y, y1=y,
                      line=dict(color="rgba(0,0,0,0.35)", width=1, dash="dot"))
    for sensor, x in (geom.temp_sensor_x_mm or {}).items():
        fig.add_trace(go.Scatter(
            x=[x], y=[-4], mode="markers", marker=dict(symbol="triangle-up", size=9,
                                                       color="#444"),
            hovertemplate=f"{sensor} at x = {x:.0f} mm<extra></extra>",
            showlegend=False))

    # Colour bar, carried by an invisible trace.
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
        marker=dict(colorscale=colorscale, cmin=vmin, cmax=vmax, size=0.1,
                    color=[vmin], showscale=True,
                    colorbar=dict(title=dict(text=meta.get("unit", ""), side="right"),
                                  thickness=14, len=0.85)),
    ))

    fig.update_layout(
        template=TEMPLATE,
        title=title or param_label(param),
        xaxis=dict(title="x [mm]  →  gas flow" if geom.flow_axis == "x" else "x [mm]",
                   range=[-6, geom.plate_w_mm + 6], constrain="domain"),
        yaxis=dict(title="y [mm]", range=[-8, geom.plate_h_mm + 6],
                   scaleanchor="x", scaleratio=1),
        margin=dict(l=50, r=10, t=50, b=40),
        height=440, hovermode="closest",
    )
    return fig


def plate_layout_figure(geom: PlateGeometry) -> go.Figure:
    """The bare arrangement, coloured by area - how a new generation is checked."""
    return plate_heatmap(
        geom, geom.areas(), "area_cm2",
        title=f"{geom.name} — segment areas [cm²]",
    ).update_layout(coloraxis_showscale=True)


# ---------------------------------------------------------------------------
# spectra
# ---------------------------------------------------------------------------

def nyquist(
    curves: list[dict], title: str = "Nyquist", show_model: bool = True,
    equal_aspect: bool = True,
) -> go.Figure:
    """Draw one or more spectra.

    Each entry: ``{name, f, z_re, z_im, z_re_model?, z_im_model?, dash?}``.
    The imaginary axis is negated, as is conventional, so a capacitive arc
    bulges upward.
    """
    fig = go.Figure()
    for curve in curves:
        z_re = np.asarray(curve["z_re"], float)
        z_im = np.asarray(curve["z_im"], float)
        f = np.asarray(curve.get("f", []), float)
        hover = ("f = %{customdata:.4g} Hz<br>Z' = %{x:.4g}<br>"
                 "-Z\" = %{y:.4g}<extra>" + str(curve["name"]) + "</extra>")
        fig.add_trace(go.Scatter(
            x=z_re, y=-z_im, mode="markers+lines", name=str(curve["name"]),
            customdata=f if f.size == z_re.size else np.zeros_like(z_re),
            marker=dict(size=6), line=dict(width=1, dash=curve.get("dash", "solid")),
            hovertemplate=hover,
        ))
        if show_model and curve.get("z_re_model") is not None:
            mre = np.asarray(curve["z_re_model"], float)
            mim = np.asarray(curve["z_im_model"], float)
            if np.isfinite(mre).any():
                fig.add_trace(go.Scatter(
                    x=mre, y=-mim, mode="lines", name=f"{curve['name']} · model",
                    line=dict(width=2, dash="dash"), hoverinfo="skip",
                ))
    fig.update_layout(
        template=TEMPLATE, title=title,
        xaxis_title="Z′ [mΩ·cm²]", yaxis_title="−Z″ [mΩ·cm²]",
        margin=dict(l=60, r=20, t=50, b=50), height=440,
        legend=dict(orientation="h", y=-0.2),
    )
    if equal_aspect:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def bode(curves: list[dict], title: str = "Bode") -> go.Figure:
    """|Z| and phase against frequency, sharing the x axis."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("|Z| [mΩ·cm²]", "phase [°]"))
    for curve in curves:
        f = np.asarray(curve["f"], float)
        z = np.asarray(curve["z_re"], float) + 1j * np.asarray(curve["z_im"], float)
        fig.add_trace(go.Scatter(x=f, y=np.abs(z), mode="markers+lines",
                                 name=str(curve["name"]), marker=dict(size=5)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=f, y=np.degrees(np.angle(z)), mode="markers+lines",
                                 name=str(curve["name"]), marker=dict(size=5),
                                 showlegend=False),
                      row=2, col=1)
    fig.update_xaxes(type="log", title_text="f [Hz]", row=2, col=1)
    fig.update_xaxes(type="log", row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_layout(template=TEMPLATE, title=title, height=520,
                      margin=dict(l=60, r=20, t=60, b=50),
                      legend=dict(orientation="h", y=-0.15))
    return fig


def fit_residuals(f, z_meas, z_fit, title: str = "Fit residuals") -> go.Figure:
    """Relative residual against frequency, real and imaginary separately.

    Structure in this plot - a residual that swings systematically rather than
    scattering about zero - is how an inadequate circuit shows itself, and it
    is invisible in the Nyquist plot once the arcs overlap.
    """
    f = np.asarray(f, float)
    z_meas = np.asarray(z_meas, complex)
    z_fit = np.asarray(z_fit, complex)
    denom = np.abs(z_meas)
    denom[denom == 0] = np.nan
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f, y=100 * (z_fit.real - z_meas.real) / denom,
                             mode="markers+lines", name="Δ Re / |Z| [%]"))
    fig.add_trace(go.Scatter(x=f, y=100 * (z_fit.imag - z_meas.imag) / denom,
                             mode="markers+lines", name="Δ Im / |Z| [%]"))
    fig.add_hline(y=0, line=dict(color="rgba(0,0,0,0.4)", width=1))
    fig.update_layout(template=TEMPLATE, title=title, height=300,
                      xaxis=dict(type="log", title="f [Hz]"),
                      yaxis_title="residual [%]",
                      margin=dict(l=60, r=20, t=50, b=45),
                      legend=dict(orientation="h", y=-0.25))
    return fig


# ---------------------------------------------------------------------------
# distributions and comparisons
# ---------------------------------------------------------------------------

def parameter_distribution(values: dict[str, float], param: str,
                           title: str = "") -> go.Figure:
    """Segment-to-segment spread of one parameter, with the plate median."""
    finite = np.array([v for v in values.values() if np.isfinite(v)], float)
    fig = go.Figure()
    if finite.size:
        fig.add_trace(go.Histogram(x=finite, nbinsx=max(8, int(np.sqrt(finite.size) * 2)),
                                   marker_color="#4c78a8", name=param))
        fig.add_vline(x=float(np.median(finite)), line=dict(color="#e45756", width=2),
                      annotation_text="median", annotation_position="top")
    fig.update_layout(template=TEMPLATE, height=280,
                      title=title or f"Distribution — {param_label(param)}",
                      xaxis_title=param_label(param), yaxis_title="segments",
                      margin=dict(l=55, r=20, t=50, b=45), showlegend=False)
    return fig


def condition_trend(series: dict[str, dict[str, float]], param: str,
                    title: str = "") -> go.Figure:
    """One line per segment across operating conditions.

    ``series`` maps condition -> {segment: value}.  This is the view that makes
    a load-dependent defect obvious: a mass-transport limitation grows with
    current density while an ohmic contact problem does not.
    """
    conditions = list(series)
    segments = sorted({s for values in series.values() for s in values},
                      key=lambda n: (0, int(n)) if str(n).isdigit() else (1, n))
    fig = go.Figure()
    # A 72-entry legend is a wall, not a key: past a couple of dozen segments
    # the lines are read as a bundle and identified by hovering instead.
    label_each = len(segments) <= 20
    for segment in segments:
        y = [series[c].get(segment, np.nan) for c in conditions]
        if not np.isfinite(np.asarray(y, float)).any():
            continue
        fig.add_trace(go.Scatter(x=conditions, y=y, mode="lines+markers",
                                 name=f"seg {segment}", line=dict(width=1),
                                 marker=dict(size=5), opacity=0.55,
                                 showlegend=label_each,
                                 hovertemplate="seg " + str(segment) +
                                               "<br>%{x}: %{y:.4g}<extra></extra>"))
    median = [float(np.nanmedian(list(series[c].values()) or [np.nan]))
              for c in conditions]
    fig.add_trace(go.Scatter(x=conditions, y=median, mode="lines+markers",
                             name="plate median", line=dict(width=4, color="#e45756")))
    fig.update_layout(template=TEMPLATE, height=420,
                      title=title or f"{param_label(param)} across conditions",
                      xaxis_title="condition", yaxis_title=param_label(param),
                      margin=dict(l=65, r=20, t=50, b=45),
                      legend=dict(font=dict(size=9)))
    return fig


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False,
                       font=dict(size=13, color="#666"),
                       xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_layout(template=TEMPLATE, height=300,
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(l=10, r=10, t=10, b=10))
    return fig
