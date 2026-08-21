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
                html.Div(ui.field("Also draw",
                                  dcc.Checklist(
                                      id="sp-extras",
                                      options=[
                                          {"label": " whole cell (segments in parallel)",
                                           "value": "cell"},
                                          {"label": " the pipeline's fitted curve",
                                           "value": "model"},
                                      ],
                                      value=["cell", "model"],
                                      labelStyle={"display": "block",
                                                  "fontSize": "12px"}),
                                  "The whole-cell curve is Z_cell = A / Σ(A_s/Z_s): "
                                  "what a single-lead measurement of this cell would "
                                  "see. A healthy-looking cell curve can hide two "
                                  "segments pulling opposite ways, which is the point "
                                  "of drawing both."),
                         style={"flex": "1 1 220px"}),
                html.Div(ui.field("Quick pick",
                                  dcc.Dropdown(
                                      id="sp-quick", clearable=True,
                                      placeholder="choose a set",
                                      options=[
                                          {"label": "Highest RΩ (5)", "value": "hi_rohmic"},
                                          {"label": "Lowest RΩ (5)", "value": "lo_rohmic"},
                                          {"label": "Highest R_mt (5)", "value": "hi_rmt"},
                                          {"label": "Flagged faults", "value": "faults"},
                                          {"label": "One per row, inlet to outlet",
                                           "value": "band"},
                                      ]),
                                  "Shortcuts to the segments worth looking at. "
                                  "The last one takes one segment from each "
                                  "quarter of the plate across the flow, so the "
                                  "inlet-to-outlet trend shows up in four curves "
                                  "instead of seventy-two."),
                         style={"flex": "1 1 220px"}),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
            html.Div([
                html.Div(ui.field("Frequency range [Hz]",
                                  dcc.RangeSlider(
                                      id="sp-band", min=-2, max=5, step=None,
                                      value=[-2, 5], allowCross=False,
                                      marks={-2: "0.01", -1: "0.1", 0: "1",
                                             1: "10", 2: "100", 3: "1k",
                                             4: "10k", 5: "100k"},
                                      tooltip={"placement": "bottom"}),
                                  "Filters the points that are DRAWN, here and "
                                  "now. The f min / f max on the ECM tab are a "
                                  "different thing: they choose which points are "
                                  "FITTED, and leave the rest on the plot."),
                         style={"flex": "1 1 100%"}),
            ], style={"display": "flex", "gap": "12px", "marginTop": "4px"}),
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
                  Input("sp-extras", "value"), Input("sp-band", "value"))
    def _plots(selection, segments, extras, band):
        extras = extras or []
        lo, hi = (10.0 ** float(band[0]), 10.0 ** float(band[1])) \
            if band and len(band) == 2 else (0.0, float("inf"))
        if not selection or not selection.get("measurement_id"):
            blank = empty_figure("select an order id")
            return blank, empty_figure(""), ui.note("")
        run = store.current_run(selection["kind"], selection["measurement_id"],
                                selection["condition"], selection["plate_key"])
        if run.spectra.empty:
            msg = "; ".join(run.warnings) or "this run has no spectra"
            return empty_figure(msg), empty_figure(""), ui.note(msg)

        def in_band(frame):
            if frame is None or frame.empty or "freq_hz" not in frame:
                return frame
            f = frame["freq_hz"]
            return frame[(f >= lo) & (f <= hi)]

        curves, frames = [], []
        n_before = n_after = 0
        for segment in segments or []:
            spectrum = run.spectrum(segment)
            n_before += len(spectrum)
            spectrum = in_band(spectrum)
            n_after += len(spectrum)
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
            cell = in_band(run.cell)
            curves.append(dict(name="cell aggregate", f=cell["freq_hz"],
                               z_re=cell["z_re_mohm_cm2"], z_im=cell["z_im_mohm_cm2"],
                               dash="dot"))

        if not curves:
            msg = ("no points between "
                   f"{lo:.4g} and {hi:.4g} Hz — widen the frequency range"
                   if n_before else "no segment selected")
            return empty_figure(msg), empty_figure(""), ui.note(msg)

        # Equal axes are not optional on a Nyquist plot: unequal ones turn a
        # semicircle into an ellipse, and the eye reads the ellipse as the
        # shape of the arc.
        fig = nyquist(curves, title=f"Nyquist — {run.label()}",
                      show_model="model" in extras,
                      equal_aspect=True)
        fig.update_layout(height=520)
        bode_fig = bode(curves, title=f"Bode — {run.label()}")

        import pandas as pd
        table = (pd.concat(frames, ignore_index=True)
                 if frames else pd.DataFrame())
        note = ui.note(f"{n_after} of {n_before} points shown "
                       f"({lo:.4g} – {hi:.4g} Hz)") if n_after != n_before else None
        body = [note, ui.table(table, "sp-points", max_rows=500)] if note \
            else ui.table(table, "sp-points", max_rows=500)
        return fig, bode_fig, body
