"""Equivalent-circuit fitting, per segment and across the whole plate.

The pipeline's ``R_ct`` / ``R_mt`` split comes from cutting the distribution of
relaxation times at a fixed time constant.  That is a different quantity from a
fitted circuit, and the two disagreeing is informative rather than embarrassing
- so the cross-check panel puts them on the same axes instead of quietly
replacing one with the other.

Fitting is weighted by the per-point uncertainty, positive parameters are
fitted in log space, and by default the circuit is *selected* by corrected AIC
from a ladder rather than assumed: forcing two arcs onto a spectrum whose
second arc never closed inside the band returns confident, meaningless numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, no_update

from app.data.model import param_label
from app.plates import registry
from app.services import ecm_service as E
from app.services import store
from app.services.figures import (TEMPLATE, empty_figure, fit_residuals, nyquist,
                                  plate_heatmap)
from app.views import common as ui

MODEL_OPTIONS = [{"label": E.MODEL_LABELS[k], "value": k}
                 for k in ("auto", "R_L_1RQ", "R_L_2RQ", "R_L_1RQ_W", "R_L")]


def layout():
    return html.Div([
        ui.panel([
            html.Div([
                html.Div(ui.field("Circuit",
                                  dcc.Dropdown(id="ecm-model", options=MODEL_OPTIONS,
                                               value="auto", clearable=False),
                                  "Auto fits every circuit in the ladder — R+L, "
                                  "one arc, two arcs, one arc with a Warburg — "
                                  "and keeps the one with the lowest corrected "
                                  "AIC. Adding an arc always lowers χ², so χ² "
                                  "alone would always pick the biggest circuit; "
                                  "AICc charges for each extra parameter and "
                                  "stops a second arc being invented where the "
                                  "data does not support one."),
                         style={"flex": "2 1 300px"}),
                html.Div(ui.field("Segment",
                                  dcc.Dropdown(id="ecm-segment", options=[],
                                               value=None, clearable=False)),
                         style={"flex": "1 1 160px"}),
                html.Div(ui.field("Fit from [Hz]",
                                  dcc.Input(id="ecm-fmin", type="number",
                                            debounce=True, style={"width": "100%"}),
                                  "Restricts which points the fit USES. Leave "
                                  "blank to fit everything."),
                         style={"flex": "1 1 130px"}),
                html.Div(ui.field("Fit to [Hz]",
                                  dcc.Input(id="ecm-fmax", type="number",
                                            debounce=True, style={"width": "100%"}),
                                  "Excluded points stay on the plot — use the "
                                  "Spectra tab to hide points instead."),
                         style={"flex": "1 1 130px"}),
                html.Div(ui.field("Restarts",
                                  dcc.Input(id="ecm-starts", type="number", value=6,
                                            min=1, max=24, step=1, debounce=True,
                                            style={"width": "100%"}),
                                  "A circuit fit is a non-linear least-squares "
                                  "problem with more than one minimum: the "
                                  "optimiser can settle into a bad one and "
                                  "report it confidently. Each restart begins "
                                  "from a different starting guess and the best "
                                  "result is kept. 6 is usually enough; raise it "
                                  "if a fit looks unstable when you re-run it."),
                         style={"flex": "1 1 130px"}),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
            html.Div([
                html.Button("Fit whole plate", id="ecm-fit-plate", n_clicks=0,
                            style={"padding": "8px 14px", "border": "none",
                                   "borderRadius": "6px", "cursor": "pointer",
                                   "background": ui.COLOURS["accent"],
                                   "color": "white", "marginRight": "10px"}),
                html.Span(id="ecm-job-status",
                          style={"fontSize": "12px", "color": ui.COLOURS["muted"]}),
            ]),
            dcc.Interval(id="ecm-poll", interval=1500, disabled=True),
            dcc.Store(id="ecm-job"),
            dcc.Store(id="ecm-done", data=0),
        ]),

        html.Div([
            html.Div([
                ui.panel([ui.graph("ecm-nyquist")]),
                ui.panel([ui.graph("ecm-residuals")]),
            ], style={"flex": "3 1 520px", "minWidth": "440px"}),
            html.Div([
                ui.panel([
                    html.Div(id="ecm-fit-note",
                             style={"fontSize": "12px", "color": ui.COLOURS["muted"],
                                    "marginBottom": "8px", "lineHeight": "1.5"}),
                    html.Div(id="ecm-params"),
                ]),
                ui.panel([
                    html.Div("Circuit ladder", style={"fontWeight": 650,
                                                      "marginBottom": "6px"}),
                    ui.note("Lower AICc wins. A difference under about 2 means "
                            "the data does not distinguish the two circuits."),
                    html.Div(id="ecm-ladder"),
                ]),
            ], style={"flex": "2 1 360px", "minWidth": "320px"}),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),

        ui.panel([
            html.Div("Fitted parameter across the plate",
                     style={"fontWeight": 650, "marginBottom": "6px"}),
            html.Div([
                html.Div(ui.field("Map", dcc.Dropdown(id="ecm-map-param", options=[],
                                                      value=None, clearable=False)),
                         style={"flex": "1 1 280px"}),
            ], style={"display": "flex", "gap": "12px"}),
            html.Div([
                html.Div([ui.graph("ecm-map")],
                         style={"flex": "3 1 520px", "minWidth": "440px"}),
                html.Div([ui.graph("ecm-crosscheck")],
                         style={"flex": "2 1 340px", "minWidth": "300px"}),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ]),

        ui.panel([
            html.Div("Fitted parameters, all segments",
                     style={"fontWeight": 650, "marginBottom": "6px"}),
            html.Div(id="ecm-table"),
        ]),
    ])


def _key(selection, model, fmin, fmax, starts):
    return store.ecm_key(selection["kind"], selection["measurement_id"],
                         selection["condition"], model, fmin, fmax, int(starts or 6))


def register(app):

    @app.callback(Output("ecm-segment", "options"), Output("ecm-segment", "value"),
                  Input("selection", "data"), State("ecm-segment", "value"))
    def _segments(selection, current):
        if not selection or not selection.get("measurement_id"):
            return [], None
        run = store.current_run(selection["kind"], selection["measurement_id"],
                                selection["condition"], selection["plate_key"])
        names = [s for s in run.segment_names if not run.spectrum(s).empty]
        options = [{"label": f"segment {s}", "value": s} for s in names]
        value = current if current in names else (names[0] if names else None)
        return options, value

    # -- single segment ------------------------------------------------------

    @app.callback(Output("ecm-nyquist", "figure"), Output("ecm-residuals", "figure"),
                  Output("ecm-params", "children"), Output("ecm-fit-note", "children"),
                  Output("ecm-ladder", "children"),
                  Input("selection", "data"), Input("ecm-segment", "value"),
                  Input("ecm-model", "value"), Input("ecm-fmin", "value"),
                  Input("ecm-fmax", "value"), Input("ecm-starts", "value"))
    def _fit_one(selection, segment, model, fmin, fmax, starts):
        if not selection or not segment:
            blank = empty_figure("select an order id and a segment")
            return blank, empty_figure(""), ui.note(""), "", ui.note("")
        run = store.current_run(selection["kind"], selection["measurement_id"],
                                selection["condition"], selection["plate_key"])
        spectrum = run.spectrum(str(segment))
        if spectrum.empty:
            blank = empty_figure("no spectrum for this segment")
            return blank, empty_figure(""), ui.note(""), "", ui.note("")

        result = E.fit_segment(spectrum, str(segment), model or "auto",
                               fmin, fmax, int(starts or 6))
        curve = dict(name=f"seg {segment}", f=spectrum["freq_hz"],
                     z_re=spectrum["z_re_mohm_cm2"], z_im=spectrum["z_im_mohm_cm2"])
        fig = nyquist([curve], title=f"ECM fit — segment {segment}", show_model=False)

        if result.fit is None:
            fig.update_layout(height=460)
            return (fig, empty_figure(result.note or "no fit"),
                    ui.note(result.note or "no fit"), result.note, ui.note(""))

        grid, z_curve = result.curve()
        fig.add_trace(go.Scatter(
            x=z_curve.real, y=-z_curve.imag, mode="lines",
            name=f"{result.fit.model} fit", line=dict(width=2.5, dash="dash"),
            hovertemplate="fit<extra></extra>"))
        fig.update_layout(height=460)

        residuals = fit_residuals(result.f, result.z_meas, result.z_fit_mohm(),
                                  title="Fit residuals — structure here means "
                                        "the circuit is inadequate")

        note = (f"Circuit {result.fit.model} · χ²_red = "
                f"{result.fit.chi2_reduced:.3g} · {result.fit.n_points} points"
                + (f" · {result.note}" if result.note else ""))
        if result.fit.poorly_determined:
            note += (" · poorly determined: "
                     + ", ".join(result.fit.poorly_determined))

        ladder = pd.DataFrame([
            {"circuit": name, "AICc": f.aicc, "χ²_red": f.chi2_reduced,
             "ΔAICc": f.aicc - min(x.aicc for x in result.ladder.values())}
            for name, f in sorted(result.ladder.items(), key=lambda kv: kv[1].aicc)
        ]) if result.ladder else pd.DataFrame()

        return (fig, residuals, ui.table(result.table(), "ecm-param-table",
                                         height="330px"),
                note, ui.table(ladder, "ecm-ladder-table", height="160px"))

    # -- whole plate ---------------------------------------------------------

    @app.callback(Output("ecm-job", "data"), Output("ecm-poll", "disabled"),
                  Input("ecm-fit-plate", "n_clicks"),
                  State("selection", "data"), State("ecm-model", "value"),
                  State("ecm-fmin", "value"), State("ecm-fmax", "value"),
                  State("ecm-starts", "value"),
                  prevent_initial_call=True)
    def _fit_plate(_n, selection, model, fmin, fmax, starts):
        if not selection or not selection.get("measurement_id"):
            return no_update, True
        run = store.current_run(selection["kind"], selection["measurement_id"],
                                selection["condition"], selection["plate_key"])
        key = _key(selection, model or "auto", fmin, fmax, starts)

        def work(progress):
            table, fits = E.fit_run(
                run, model or "auto", f_min=fmin, f_max=fmax,
                n_starts=int(starts or 6),
                progress=lambda done, total: progress(done, total))
            store.ecm_put(key, table, fits)
            return key

        job_id = store.JOBS.submit(f"ECM fit · {run.label()}", work)
        return {"job": job_id, "key": list(key)}, False

    @app.callback(Output("ecm-job-status", "children"),
                  Output("ecm-poll", "disabled", allow_duplicate=True),
                  Output("ecm-done", "data"),
                  Input("ecm-poll", "n_intervals"), State("ecm-job", "data"),
                  State("ecm-done", "data"), prevent_initial_call=True)
    def _poll(_n, job_data, done_count):
        if not job_data:
            return "", True, done_count
        job = store.JOBS.get(job_data["job"])
        if job is None:
            return "", True, done_count
        if job.state == "running":
            bar = "█" * (job.percent // 5) + "·" * (20 - job.percent // 5)
            return f"{bar}  {job.status_text()}", False, done_count
        return job.status_text(), True, (done_count or 0) + 1

    @app.callback(Output("ecm-map-param", "options"), Output("ecm-map-param", "value"),
                  Input("ecm-done", "data"), State("ecm-job", "data"),
                  State("ecm-map-param", "value"))
    def _map_params(_done, job_data, current):
        cached = store.ecm_get(tuple(job_data["key"])) if job_data else None
        if not cached:
            return [], None
        table, _fits = cached
        params = E.ecm_params(table)
        options = [{"label": p.replace("ecm_", ""), "value": p} for p in params]
        preferred = next((p for p in ("ecm_Rs", "ecm_R_pol", "ecm_Rct1")
                          if p in params), params[0] if params else None)
        return options, current if current in params else preferred

    @app.callback(Output("ecm-map", "figure"), Output("ecm-crosscheck", "figure"),
                  Output("ecm-table", "children"),
                  Input("ecm-map-param", "value"), Input("ecm-done", "data"),
                  State("ecm-job", "data"), State("selection", "data"))
    def _plate_results(param, _done, job_data, selection):
        cached = store.ecm_get(tuple(job_data["key"])) if job_data else None
        if not cached or not selection:
            hint = empty_figure("press “Fit whole plate” to fit every segment")
            return hint, empty_figure(""), ui.note("")
        table, _fits = cached
        run = store.current_run(selection["kind"], selection["measurement_id"],
                                selection["condition"], selection["plate_key"])
        try:
            geom = registry.get(selection["plate_key"])
        except KeyError as exc:
            return empty_figure(str(exc)), empty_figure(""), ui.note(str(exc))

        if not param or param not in table.columns:
            return (empty_figure("choose a parameter"), empty_figure(""),
                    ui.table(table, "ecm-all", max_rows=200))

        values = {str(s): float(v) for s, v in zip(table["segment"], table[param])
                  if pd.notna(v)}
        sds = {str(s): float(v) for s, v in
               zip(table["segment"], table.get(f"{param}_sd", []))
               if pd.notna(v)} if f"{param}_sd" in table.columns else {}
        fig = plate_heatmap(geom, values, param, sds=sds, classes=run.classes(),
                            title=f"{param.replace('ecm_', '')} — {run.label()}")
        fig.update_layout(height=460)
        return fig, _crosscheck(run, table), ui.table(table, "ecm-all", max_rows=200)


def _crosscheck(run, table) -> go.Figure:
    """Fitted Rs and R_pol against what the pipeline reported for the same segment.

    Agreement is reassurance; disagreement localises which assumption moved -
    the DRT time-constant split, the high-frequency intercept, or the circuit.
    """
    if run.segments is None or run.segments.empty:
        return empty_figure("no pipeline parameters to compare against")
    merged = run.segments.merge(table, on="segment", how="inner")
    pairs = [("R_ohmic", "ecm_Rs", "RΩ vs fitted Rs"),
             ("R_pol", "ecm_R_pol", "R_pol (DRT split) vs fitted R_pol")]
    fig = go.Figure()
    any_series = False
    for pipeline_col, ecm_col, label in pairs:
        if pipeline_col not in merged.columns or ecm_col not in merged.columns:
            continue
        sub = merged[["segment", pipeline_col, ecm_col]].dropna()
        if sub.empty:
            continue
        any_series = True
        fig.add_trace(go.Scatter(
            x=sub[pipeline_col], y=sub[ecm_col], mode="markers", name=label,
            text=[f"segment {s}" for s in sub["segment"]],
            hovertemplate="%{text}<br>pipeline %{x:.4g}<br>ECM %{y:.4g}<extra></extra>",
            marker=dict(size=7, opacity=0.8)))
    if not any_series:
        return empty_figure("no comparable pipeline parameters in this run")
    lo = min(min(t.x) for t in fig.data)
    hi = max(max(t.x) for t in fig.data)
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="1:1",
                             line=dict(color="rgba(0,0,0,0.45)", dash="dot"),
                             hoverinfo="skip"))
    fig.update_layout(template=TEMPLATE, height=460,
                      title="Circuit fit vs pipeline parameters",
                      xaxis_title="pipeline [mΩ·cm²]", yaxis_title="ECM [mΩ·cm²]",
                      margin=dict(l=60, r=20, t=50, b=45),
                      legend=dict(orientation="h", y=-0.18))
    return fig
