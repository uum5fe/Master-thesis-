"""Nyquist and Bode views for any set of segments, plus the cell aggregate.

The cell aggregate is the parallel combination of the segments,
``Z_cell = A_cell / Σ(A_s / Z_s)``, and putting it on the same axes as the
segments is the point of a segmented measurement: a cell spectrum that looks
healthy can hide two segments pulling in opposite directions.
"""

from __future__ import annotations

import numpy as np
from dash import Input, Output, State, dcc, html

from app.services import store
from app.services.figures import bode, empty_figure, nyquist
from app.views import common as ui


def layout():
    return html.Div([
        ui.panel([
            html.Div([
                html.Div(ui.field("Segments",
                                  dcc.Dropdown(id="sp-segments", options=[], value=[],
                                               multi=True,
                                               placeholder="pick one or more")),
                         style={"flex": "3 1 340px"}),
                html.Div(ui.field("Extras",
                                  dcc.Checklist(
                                      id="sp-extras",
                                      options=[
                                          {"label": " cell aggregate", "value": "cell"},
                                          {"label": " pipeline model", "value": "model"},
                                          {"label": " equal axes", "value": "aspect"},
                                      ],
                                      value=["cell", "model", "aspect"],
                                      labelStyle={"display": "block",
                                                  "fontSize": "12px"})),
                         style={"flex": "1 1 180px"}),
                html.Div(ui.field("Quick pick",
                                  dcc.Dropdown(
                                      id="sp-quick", clearable=True,
                                      placeholder="choose a set",
                                      options=[
                                          {"label": "Highest RΩ (5)", "value": "hi_rohmic"},
                                          {"label": "Lowest RΩ (5)", "value": "lo_rohmic"},
                                          {"label": "Highest R_mt (5)", "value": "hi_rmt"},
                                          {"label": "Flagged faults", "value": "faults"},
                                          {"label": "One per flow band", "value": "band"},
                                      ]),
                                  "Shortcuts to the segments worth looking at."),
                         style={"flex": "1 1 200px"}),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ]),
        html.Div([
            html.Div([ui.panel([ui.graph("sp-nyquist")])],
                     style={"flex": "1 1 480px", "minWidth": "420px"}),
            html.Div([ui.panel([ui.graph("sp-bode")])],
                     style={"flex": "1 1 420px", "minWidth": "380px"}),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ui.panel([
            html.Div("Selected points", style={"fontWeight": 650, "marginBottom": "6px"}),
            html.Div(id="sp-table"),
        ]),
    ])


def _pick(run, mode: str) -> list[str]:
    frame = run.segments
    if frame is None or frame.empty:
        return run.segment_names[:3]
    if mode == "faults" and "fault" in frame.columns:
        return [str(s) for s, f in zip(frame["segment"], frame["fault"]) if str(f)][:8]
    column = {"hi_rohmic": "R_ohmic", "lo_rohmic": "R_ohmic", "hi_rmt": "R_mt"}.get(mode)
    if column and column in frame.columns:
        sub = frame[["segment", column]].dropna()
        sub = sub.sort_values(column, ascending=mode.startswith("lo"))
        return [str(s) for s in sub["segment"].tail(5)] if mode.startswith("hi") \
            else [str(s) for s in sub["segment"].head(5)]
    if mode == "band" and {"cy_mm"} <= set(frame.columns):
        picks = []
        sub = frame.dropna(subset=["cy_mm"]).sort_values("cy_mm")
        for chunk in np.array_split(sub, 4):
            if len(chunk):
                picks.append(str(chunk.iloc[len(chunk) // 2]["segment"]))
        return picks
    return run.segment_names[:3]


def register(app):

    @app.callback(Output("sp-segments", "options"), Output("sp-segments", "value"),
                  Input("selection", "data"), Input("sp-quick", "value"),
                  State("sp-segments", "value"))
    def _segments(selection, quick, current):
        if not selection or not selection.get("measurement_id"):
            return [], []
        run = store.current_run(selection["kind"], selection["measurement_id"],
                                selection["condition"], selection["plate_key"])
        names = [s for s in run.segment_names if not run.spectrum(s).empty]
        options = [{"label": f"segment {s}", "value": s} for s in names]
        if quick:
            return options, [s for s in _pick(run, quick) if s in names]
        keep = [s for s in (current or []) if s in names]
        return options, keep or names[:2]

    @app.callback(Output("sp-nyquist", "figure"), Output("sp-bode", "figure"),
                  Output("sp-table", "children"),
                  Input("selection", "data"), Input("sp-segments", "value"),
                  Input("sp-extras", "value"))
    def _plots(selection, segments, extras):
        extras = extras or []
        if not selection or not selection.get("measurement_id"):
            blank = empty_figure("select an order id")
            return blank, empty_figure(""), ui.note("")
        run = store.current_run(selection["kind"], selection["measurement_id"],
                                selection["condition"], selection["plate_key"])
        if run.spectra.empty:
            msg = "; ".join(run.warnings) or "this run has no spectra"
            return empty_figure(msg), empty_figure(""), ui.note(msg)

        curves, frames = [], []
        for segment in segments or []:
            spectrum = run.spectrum(segment)
            if spectrum.empty:
                continue
            curves.append(dict(
                name=f"seg {segment}", f=spectrum["freq_hz"],
                z_re=spectrum["z_re_mohm_cm2"], z_im=spectrum["z_im_mohm_cm2"],
                z_re_model=(spectrum.get("z_re_model_mohm_cm2")
                            if "model" in extras else None),
                z_im_model=(spectrum.get("z_im_model_mohm_cm2")
                            if "model" in extras else None),
            ))
            frames.append(spectrum.assign(segment=segment))

        if "cell" in extras and run.cell is not None and not run.cell.empty:
            cell = run.cell
            curves.append(dict(name="cell aggregate", f=cell["freq_hz"],
                               z_re=cell["z_re_mohm_cm2"], z_im=cell["z_im_mohm_cm2"],
                               dash="dot"))

        if not curves:
            blank = empty_figure("no segment selected")
            return blank, empty_figure(""), ui.note("")

        fig = nyquist(curves, title=f"Nyquist — {run.label()}",
                      show_model="model" in extras,
                      equal_aspect="aspect" in extras)
        fig.update_layout(height=520)
        bode_fig = bode(curves, title=f"Bode — {run.label()}")

        import pandas as pd
        table = (pd.concat(frames, ignore_index=True)
                 if frames else pd.DataFrame())
        return fig, bode_fig, ui.table(table, "sp-points", max_rows=500)
