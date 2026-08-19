r"""Evaluating a plate calibration campaign.

The campaign answers a different question from a measurement, so this tab shows
different things. There is no impedance here and no operating point - there is
a set of current sweeps repeated at several plate temperatures, and four
questions worth asking of them:

* **Is each channel linear?** The sweep is five points from 0 to 1 A. A segment
  whose points do not sit on a line through the origin has something wrong with
  its shunt or its amplifier, and no coefficient will rescue it.
* **How does its sensitivity move with temperature?** That is what the second
  coefficient is for, and a segment far from the others has a different thermal
  path rather than a different gain.
* **Do the shipped coefficients belong to these sweeps?** Reconstructing them
  from the raw data gives a constant ratio for every segment; a segment that
  departs from that constant was calibrated from something else.
* **Did the plate come back?** A campaign that starts and ends at the same
  nominal temperature is asking exactly that, and the answer is per segment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

from app.data import calibration as cal
from app.data.model import PARAM_META, param_label
from app.plates import registry
from app.services import store
from app.services.figures import TEMPLATE, empty_figure, plate_heatmap
from app.views import common as ui

MAP_PARAMS = [
    "H_at_60C", "H_c0", "H_c1", "drift_pct_over_span",
    "linearity_worst_pct", "fit_residual_pct",
    "ratio_c0_dev_pct", "ratio_c1_dev_pct",
]


def layout():
    return html.Div([
        html.Div(id="cal-header"),
        html.Div(id="cal-warnings"),
        ui.panel([
            html.Div([
                html.Div(ui.field("Map", dcc.Dropdown(id="cal-param", options=[
                    {"label": param_label(p), "value": p} for p in MAP_PARAMS],
                    value="H_at_60C", clearable=False)),
                    style={"flex": "2 1 320px"}),
                html.Div(ui.field("Segment",
                                  dcc.Dropdown(id="cal-segment", options=[],
                                               value=None, clearable=False),
                                  "Click the map to change this."),
                         style={"flex": "1 1 160px"}),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ]),
        html.Div([
            html.Div([ui.panel([ui.graph("cal-map")])],
                     style={"flex": "3 1 520px", "minWidth": "440px"}),
            html.Div([
                ui.panel([ui.graph("cal-sweeps")]),
                ui.panel([ui.graph("cal-ht")]),
            ], style={"flex": "2 1 380px", "minWidth": "340px"}),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        html.Div([
            html.Div([ui.panel([
                html.Div("Temperature steps", style={"fontWeight": 650,
                                                     "marginBottom": "6px"}),
                ui.note("What each file claims, and what the sensors read once "
                        "temp.csv has been applied. A consistent offset is a "
                        "sensor calibration to look at; a scattered one is a "
                        "plate that had not settled."),
                html.Div(id="cal-steps"),
            ])], style={"flex": "1 1 460px", "minWidth": "400px"}),
            html.Div([ui.panel([
                html.Div("Repeat drift", style={"fontWeight": 650,
                                                "marginBottom": "6px"}),
                ui.note("Sensitivity at the end of the campaign against the "
                        "start, at the same nominal temperature."),
                html.Div(id="cal-drift"),
            ])], style={"flex": "1 1 380px", "minWidth": "340px"}),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ui.panel([
            html.Div("Per-segment results", style={"fontWeight": 650,
                                                   "marginBottom": "6px"}),
            html.Div(id="cal-table"),
        ]),
    ])


def _run(selection) -> cal.CalibrationRun | None:
    ref = store.find_ref("calibration", selection.get("measurement_id", ""),
                         selection.get("condition", ""))
    if ref is None:
        return None
    return _cached(ref.path, ref.measurement_id, store.generation())


_CACHE: dict[tuple, cal.CalibrationRun] = {}


def _cached(path: str, name: str, generation: int) -> cal.CalibrationRun:
    """Parsing 12 steps x 72 segments on every dropdown change is not free."""
    key = (path, name, generation)
    if key not in _CACHE:
        if len(_CACHE) > 8:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[key] = cal.load_campaign(path, name)
    return _CACHE[key]


def register(app):

    @app.callback(Output("cal-header", "children"), Output("cal-warnings", "children"),
                  Output("cal-segment", "options"), Output("cal-segment", "value"),
                  Output("cal-steps", "children"), Output("cal-drift", "children"),
                  Output("cal-table", "children"),
                  Input("selection", "data"), State("cal-segment", "value"))
    def _load(selection, current):
        blank = (ui.note("Select a calibration campaign in the sidebar."),
                 html.Div(), [], None, ui.note(""), ui.note(""), ui.note(""))
        if not selection or selection.get("kind") != "calibration":
            return (ui.note("Set the file format to “Calibration sweeps” and "
                            "pick a campaign."), html.Div(), [], None,
                    ui.note(""), ui.note(""), ui.note(""))
        run = _run(selection)
        if run is None or not run.steps:
            return blank

        info = cal.summary(run)
        consistent = info.get("coefficients_consistent")
        header = ui.stat_row([
            ("Campaign", run.name),
            ("Temperature steps", str(info["n_steps"])),
            ("Segments", str(info["n_segments"])),
            ("Range", f"{info['temperature_range_C']} °C"),
            ("Worst linearity", f"{info.get('worst_linearity_pct', float('nan')):.2f} %",
             "good" if info.get("worst_linearity_pct", 9) < 1 else "warn"),
            ("Repeat drift",
             f"{info.get('worst_repeat_drift_pct', float('nan')):.2f} %"
             if "worst_repeat_drift_pct" in info else "—",
             "good" if info.get("worst_repeat_drift_pct", 9) < 1 else "warn"),
            ("Coefficients",
             "consistent" if consistent else ("inconsistent" if consistent is False
                                              else "not shipped"),
             "good" if consistent else "warn"),
            ("Sensor offset",
             f"{info.get('sensor_offset_C', float('nan')):+.1f} °C",
             "good" if abs(info.get("sensor_offset_C", 9)) < 1 else "warn"),
        ])

        messages = list(run.warnings)
        if "chain_constant" in info:
            messages.append(
                f"The shipped coefficients are these sweeps' through-origin "
                f"sensitivity divided by {info['chain_constant']:.4f}, the same "
                f"for every segment to within "
                f"{info['chain_constant_spread']:.1e}. That constant is a "
                f"property of the measurement chain and is not derivable from "
                f"these files; what matters here is that it does not vary "
                f"between segments.")
        if abs(info.get("sensor_offset_C", 0)) > 1:
            messages.append(
                f"The temperature sensors read {info['sensor_offset_C']:+.1f} °C "
                f"against the step labels. Either the chamber setpoint and the "
                f"plate genuinely differ, or temp.csv is offset; the labels are "
                f"only used here when temp.csv is missing.")

        segments = cal.segment_table(run)
        names = list(segments["segment"])
        value = current if current in names else (names[0] if names else None)

        return (header,
                ui.warnings_block(messages, "What the numbers say") if messages
                else html.Div(),
                [{"label": f"segment {n}", "value": n} for n in names], value,
                ui.table(cal.step_table(run), "cal-step-table", height="300px"),
                ui.table(_drift_summary(run), "cal-drift-table", height="300px"),
                ui.table(segments, "cal-seg-table", max_rows=200))

    @app.callback(Output("cal-map", "figure"),
                  Input("selection", "data"), Input("cal-param", "value"))
    def _map(selection, param):
        if not selection or selection.get("kind") != "calibration":
            return empty_figure("select a calibration campaign")
        run = _run(selection)
        if run is None:
            return empty_figure("campaign not found; press Refresh sources")
        try:
            geom = registry.get(selection.get("plate_key", ""))
        except KeyError as exc:
            return empty_figure(str(exc))

        table = cal.segment_table(run)
        if param not in table.columns:
            return empty_figure(f"{param} not available for this campaign")
        values = {str(s): float(v) for s, v in zip(table["segment"], table[param])
                  if pd.notna(v)}
        if not values:
            return empty_figure(f"{param} could not be computed")

        fig = plate_heatmap(geom, values, param,
                            title=f"{param_label(param)} — {run.name}")
        fig.update_layout(height=520)
        return fig

    @app.callback(Output("cal-segment", "value", allow_duplicate=True),
                  Input("cal-map", "clickData"), prevent_initial_call=True)
    def _click(click):
        if not click:
            return None
        point = click["points"][0]
        segment = point.get("customdata")
        if isinstance(segment, list):
            segment = segment[0]
        return segment

    @app.callback(Output("cal-sweeps", "figure"), Output("cal-ht", "figure"),
                  Input("selection", "data"), Input("cal-segment", "value"))
    def _segment(selection, segment):
        if not selection or selection.get("kind") != "calibration" or not segment:
            return (empty_figure("pick a segment"), empty_figure(""))
        run = _run(selection)
        if run is None:
            return (empty_figure("campaign not found"), empty_figure(""))

        sweeps = go.Figure()
        temperatures, sensitivities = [], []
        for step in run.steps:
            pair = step.sweeps.get(str(segment))
            if pair is None:
                continue
            i, u = pair
            sweeps.add_trace(go.Scatter(
                x=i, y=u, mode="markers+lines",
                name=f"{step.measured_c:.0f} °C",
                hovertemplate="i = %{x:.4f} A<br>u = %{y:.4f} V<extra></extra>"))
            temperatures.append(step.measured_c)
            sensitivities.append(cal.sensitivity(i, u))
        sweeps.update_layout(
            template=TEMPLATE, height=300,
            title=f"Current sweeps — segment {segment}",
            xaxis_title="injected current i_s [A]",
            yaxis_title="shunt voltage u_s [V]",
            margin=dict(l=60, r=20, t=50, b=45),
            legend=dict(font=dict(size=9)))

        ht = go.Figure()
        if len(temperatures) >= 2:
            ht.add_trace(go.Scatter(x=temperatures, y=sensitivities, mode="markers",
                                    name="measured", marker=dict(size=9)))
            slope, intercept = np.polyfit(temperatures, sensitivities, 1)
            grid = np.linspace(min(temperatures), max(temperatures), 50)
            ht.add_trace(go.Scatter(
                x=grid, y=slope * grid + intercept, mode="lines",
                name=f"H(T) = {intercept:.4f} + {slope*1000:.4f}e-3·T",
                line=dict(dash="dash")))
        ht.update_layout(template=TEMPLATE, height=300,
                         title=f"Sensitivity against temperature — segment {segment}",
                         xaxis_title="plate temperature [°C]",
                         yaxis_title="H = u_s / i_s [V/A]",
                         margin=dict(l=60, r=20, t=50, b=45),
                         legend=dict(orientation="h", y=-0.25, font=dict(size=10)))
        return sweeps, ht


def _drift_summary(run: cal.CalibrationRun) -> pd.DataFrame:
    drift = cal.repeat_drift(run)
    if drift.empty:
        return pd.DataFrame([{"note": "no temperature was visited twice"}])
    worst = (drift.reindex(drift["drift_pct"].abs().sort_values(ascending=False).index)
             .head(15))
    return worst[["segment", "nominal_C", "first_step", "last_step",
                  "H_first", "H_last", "drift_pct"]]
