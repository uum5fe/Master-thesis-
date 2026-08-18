"""Comparing operating conditions.

A defect that grows with load and one that does not are different defects.
Mass-transport limitation scales with current density; an ohmic contact
problem barely moves.  Neither is visible in a single map, which is why this
tab exists: the same parameter, the same plate, several currents, side by side
and as a difference map against a reference condition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dash import Input, Output, State, dcc, html

from app.data.model import PARAM_META, param_label
from app.data.sources import _condition_sort_key
from app.plates import registry
from app.services import store
from app.services.figures import condition_trend, empty_figure, plate_heatmap
from app.views import common as ui

DIFF_MODES = [
    {"label": "Difference (target − reference)", "value": "diff"},
    {"label": "Ratio (target / reference)", "value": "ratio"},
    {"label": "Relative change [%]", "value": "pct"},
]


def layout():
    return html.Div([
        ui.panel([
            html.Div([
                html.Div(ui.field("Conditions",
                                  dcc.Dropdown(id="cmp-conditions", options=[],
                                               value=[], multi=True)),
                         style={"flex": "2 1 300px"}),
                html.Div(ui.field("Parameter",
                                  dcc.Dropdown(id="cmp-param", options=[],
                                               value=None, clearable=False)),
                         style={"flex": "2 1 260px"}),
                html.Div(ui.field("Reference",
                                  dcc.Dropdown(id="cmp-reference", options=[],
                                               value=None, clearable=False),
                                  "The condition the difference map subtracts."),
                         style={"flex": "1 1 160px"}),
                html.Div(ui.field("Target",
                                  dcc.Dropdown(id="cmp-target", options=[],
                                               value=None, clearable=False)),
                         style={"flex": "1 1 160px"}),
                html.Div(ui.field("Difference as",
                                  dcc.Dropdown(id="cmp-mode", options=DIFF_MODES,
                                               value="diff", clearable=False)),
                         style={"flex": "1 1 200px"}),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ]),
        ui.panel([ui.graph("cmp-trend")]),
        html.Div([
            html.Div([ui.panel([ui.graph("cmp-diffmap")])],
                     style={"flex": "3 1 520px", "minWidth": "440px"}),
            html.Div([ui.panel([
                html.Div("Plate summary per condition",
                         style={"fontWeight": 650, "marginBottom": "6px"}),
                ui.note("Median, spread and the number of segments actually "
                        "measured. A spread that opens up with load is the "
                        "signature worth reporting."),
                html.Div(id="cmp-table"),
            ])], style={"flex": "2 1 340px", "minWidth": "320px"}),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
    ])


def _runs_for(selection, conditions):
    runs = {}
    for condition in conditions or []:
        run = store.current_run(selection["kind"], selection["measurement_id"],
                                condition, selection["plate_key"])
        if run.segments is not None and not run.segments.empty:
            runs[condition] = run
    return runs


def register(app):

    @app.callback(Output("cmp-conditions", "options"), Output("cmp-conditions", "value"),
                  Output("cmp-reference", "options"), Output("cmp-reference", "value"),
                  Output("cmp-target", "options"), Output("cmp-target", "value"),
                  Input("selection", "data"))
    def _conditions(selection):
        if not selection or not selection.get("measurement_id"):
            return [], [], [], None, [], None
        catalog = store.current_catalog()
        conds = sorted({r.condition for r in catalog.runs
                        if r.measurement_id == selection["measurement_id"]
                        and r.kind == selection["kind"]}, key=_condition_sort_key)
        options = [{"label": c, "value": c} for c in conds]
        reference = conds[0] if conds else None
        target = conds[-1] if len(conds) > 1 else reference
        return options, conds, options, reference, options, target

    @app.callback(Output("cmp-param", "options"), Output("cmp-param", "value"),
                  Input("selection", "data"), Input("cmp-conditions", "value"),
                  State("cmp-param", "value"))
    def _params(selection, conditions, current):
        if not selection or not conditions:
            return [], None
        runs = _runs_for(selection, conditions)
        if not runs:
            return [], None
        shared: set[str] | None = None
        for run in runs.values():
            params = set(run.mappable_params())
            shared = params if shared is None else (shared & params)
        ordered = [p for p in PARAM_META if p in (shared or set())]
        ordered += sorted((shared or set()) - set(ordered))
        options = [{"label": param_label(p) if p in PARAM_META else p, "value": p}
                   for p in ordered]
        value = current if current in ordered else (ordered[0] if ordered else None)
        return options, value

    @app.callback(Output("cmp-trend", "figure"), Output("cmp-diffmap", "figure"),
                  Output("cmp-table", "children"),
                  Input("selection", "data"), Input("cmp-conditions", "value"),
                  Input("cmp-param", "value"), Input("cmp-reference", "value"),
                  Input("cmp-target", "value"), Input("cmp-mode", "value"))
    def _render(selection, conditions, param, reference, target, mode):
        if not selection or not conditions or not param:
            blank = empty_figure("pick conditions and a parameter")
            return blank, empty_figure(""), ui.note("")

        runs = _runs_for(selection, conditions)
        if not runs:
            blank = empty_figure("no per-segment data for these conditions")
            return blank, empty_figure(""), ui.note("")

        series = {c: run.value(param) for c, run in runs.items()}
        trend = condition_trend(series, param,
                               title=f"{param_label(param)} — "
                                     f"{selection['measurement_id']}")

        rows = []
        for condition, values in series.items():
            finite = np.array([v for v in values.values() if np.isfinite(v)], float)
            rows.append({
                "condition": condition,
                "n segments": int(finite.size),
                "median": float(np.median(finite)) if finite.size else np.nan,
                "min": float(finite.min()) if finite.size else np.nan,
                "max": float(finite.max()) if finite.size else np.nan,
                "spread (max/min)": (float(finite.max() / finite.min())
                                     if finite.size and finite.min() > 0 else np.nan),
            })
        table = ui.table(pd.DataFrame(rows), "cmp-summary", height="300px")

        diff_fig = _difference_map(selection, series, param, reference, target, mode)
        return trend, diff_fig, table


def _difference_map(selection, series, param, reference, target, mode):
    if reference not in series or target not in series:
        return empty_figure("reference and target must both be selected conditions")
    if reference == target:
        return empty_figure("reference and target are the same condition")
    try:
        geom = registry.get(selection["plate_key"])
    except KeyError as exc:
        return empty_figure(str(exc))

    ref_values, tgt_values = series[reference], series[target]
    values: dict[str, float] = {}
    for segment, a in ref_values.items():
        b = tgt_values.get(segment)
        if b is None or not np.isfinite(a) or not np.isfinite(b):
            continue
        if mode == "ratio":
            values[segment] = b / a if a else np.nan
        elif mode == "pct":
            values[segment] = 100.0 * (b - a) / a if a else np.nan
        else:
            values[segment] = b - a

    if not values:
        return empty_figure("no segment appears in both conditions")

    # A difference is signed, so the colour scale has to be centred on the
    # neutral value or the eye reads a uniform plate as a gradient.
    neutral = 1.0 if mode == "ratio" else 0.0
    finite = np.array([v for v in values.values() if np.isfinite(v)], float)
    span = float(np.percentile(np.abs(finite - neutral), 95))
    if span <= 0 or not np.isfinite(span):
        # Identical everywhere. Say so, rather than drawing a map whose colour
        # bar is spread over floating-point dust.
        return empty_figure(
            f"{param} is identical in {reference} and {target} for every "
            f"segment — nothing to map. (Local ASR is area-free, so an "
            f"equal-area run and a true-area run agree on it exactly.)")

    unit = {"ratio": "×", "pct": "%"}.get(mode, PARAM_META.get(param, {}).get("unit", ""))
    fig = plate_heatmap(
        geom, values, param, vmin=neutral - span, vmax=neutral + span,
        title=f"{param_label(param)}: {target} vs {reference}  [{unit}]")
    fig.update_layout(height=460)
    # Diverging scale: the sign of the change is the message.
    for trace in fig.data:
        if getattr(trace, "marker", None) and getattr(trace.marker, "showscale", False):
            trace.marker.colorscale = "RdBu_r"
    return _recolour(fig, geom, values, neutral, span)


def _recolour(fig, geom, values, neutral, span):
    """Repaint the polygons on a diverging scale centred on ``neutral``."""
    from plotly.colors import sample_colorscale
    for trace in fig.data:
        name = getattr(trace, "name", None)
        if name is None or name not in values:
            continue
        value = values[name]
        t = float(np.clip((value - (neutral - span)) / (2 * span), 0.0, 1.0))
        trace.fillcolor = sample_colorscale("RdBu_r", [t])[0]
    return fig
