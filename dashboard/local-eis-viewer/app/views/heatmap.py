"""Plate heat maps: any per-segment scalar, in true plate geometry.

Clicking a segment draws its spectrum next to the map, because "this corner of
the plate is red" is only half a finding - the other half is what the impedance
actually looks like there.
"""

from __future__ import annotations

import numpy as np
from dash import Input, Output, State, dcc, html

from app.data.model import PARAM_META, param_label
from app.plates import registry
from app.services import store
from app.services.figures import (empty_figure, nyquist, parameter_distribution,
                                  plate_heatmap)
from app.views import common as ui

#: The colour scale is fixed at robust 2–98 % percentiles and is no longer a
#: control.  It was three widgets to answer a question that has one defensible
#: answer: one runaway segment stretches a full-range scale until every real
#: difference on the plate flattens into the same colour, and hand-typed limits
#: differ between two maps that are meant to be compared.  The limits are
#: printed on the colour bar, so nothing is hidden by not offering the choice.
ROBUST_PERCENTILES = (2, 98)


def layout():
    return html.Div([
        ui.panel([
            html.Div([
                html.Div(ui.field("Parameter",
                                  dcc.Dropdown(id="hm-param", options=[], value=None,
                                               clearable=False)),
                         style={"flex": "2 1 260px"}),
                html.Div(ui.field("Labels",
                                  dcc.Checklist(id="hm-labels",
                                                options=[{"label": " segment numbers",
                                                          "value": "on"}],
                                                value=["on"])),
                         style={"flex": "0 1 150px"}),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ]),
        html.Div([
            html.Div([ui.panel([ui.graph("hm-map")])],
                     style={"flex": "3 1 560px", "minWidth": "460px"}),
            html.Div([
                ui.panel([ui.graph("hm-dist")]),
                ui.panel([
                    # Filled by a callback, so it stays a plain Div; the
                    # typography is section_title's so it does not read as a
                    # different kind of heading.
                    html.Div(id="hm-click-title",
                             style={"fontSize": "13px", "fontWeight": 650,
                                    "color": ui.COLOURS["text"],
                                    "marginBottom": "8px"}),
                    ui.graph("hm-click-spectrum"),
                ]),
            ], style={"flex": "2 1 380px", "minWidth": "340px"}),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ui.panel([
            ui.section_title("Segment table"),
            ui.note("Sortable and filterable; the Export button writes the "
                    "current view to CSV."),
            html.Div(id="hm-table"),
        ]),
    ])


def register(app):

    @app.callback(Output("hm-param", "options"), Output("hm-param", "value"),
                  Input("selection", "data"), State("hm-param", "value"))
    def _params(selection, current):
        if not selection or not selection.get("measurement_id"):
            return [], None
        run = store.current_run(selection["kind"], selection["measurement_id"],
                                selection["condition"], selection["plate_key"])
        params = run.mappable_params()
        options = [{"label": param_label(p) if p in PARAM_META else p, "value": p}
                   for p in params]
        value = current if current in params else (params[0] if params else None)
        return options, value

    @app.callback(Output("hm-map", "figure"), Output("hm-dist", "figure"),
                  Output("hm-table", "children"),
                  Input("selection", "data"), Input("hm-param", "value"),
                  Input("hm-labels", "value"))
    def _map(selection, param, labels):
        if not selection or not param:
            blank = empty_figure("select an order id and a parameter")
            return blank, empty_figure(""), ui.note("")
        run = store.current_run(selection["kind"], selection["measurement_id"],
                                selection["condition"], selection["plate_key"])
        try:
            geom = registry.get(selection["plate_key"])
        except KeyError as exc:
            blank = empty_figure(str(exc))
            return blank, empty_figure(""), ui.note(str(exc))

        values = run.value(param)
        if not values:
            blank = empty_figure(f"{param} is not present in this run")
            return blank, empty_figure(""), ui.note("")

        faults = {}
        if not run.segments.empty and "fault" in run.segments.columns:
            faults = {str(k): str(v) for k, v in
                      zip(run.segments["segment"], run.segments["fault"]) if str(v)}

        fig = plate_heatmap(
            geom, values, param, sds=run.sd(param), classes=run.classes(),
            faults=faults, vmin=None, vmax=None, robust=True,
            show_labels="on" in (labels or []),
            title=f"{param_label(param)} — {run.label()}",
        )
        fig.update_layout(height=520)

        dist = parameter_distribution(values, param)

        columns = ["segment", "seg_class", "tier", "area_cm2", "cx_mm", "cy_mm",
                   param, f"{param}_sd", "fault", "flags"]
        frame = run.segments[[c for c in columns if c in run.segments.columns]]
        return fig, dist, ui.table(frame, "hm-segment-table", max_rows=200)

    @app.callback(Output("hm-click-title", "children"),
                  Output("hm-click-spectrum", "figure"),
                  Input("hm-map", "clickData"), State("selection", "data"))
    def _clicked(click, selection):
        if not click or not selection:
            return "Click a segment", empty_figure(
                "click a segment on the map to see its spectrum")
        point = click["points"][0]
        segment = point.get("customdata") or point.get("text")
        if isinstance(segment, list):
            segment = segment[0]
        if segment is None:
            return "Click a segment", empty_figure("no segment under the cursor")

        run = store.current_run(selection["kind"], selection["measurement_id"],
                                selection["condition"], selection["plate_key"])
        spectrum = run.spectrum(str(segment))
        if spectrum.empty:
            return f"Segment {segment}", empty_figure(
                "no spectrum stored for this segment (inferred or hardware-bad)")
        curve = dict(name=f"segment {segment}", f=spectrum["freq_hz"],
                     z_re=spectrum["z_re_mohm_cm2"], z_im=spectrum["z_im_mohm_cm2"],
                     z_re_model=spectrum.get("z_re_model_mohm_cm2"),
                     z_im_model=spectrum.get("z_im_model_mohm_cm2"))
        fig = nyquist([curve], title="")
        fig.update_layout(height=320, margin=dict(l=55, r=15, t=15, b=45),
                          showlegend=False)
        return f"Segment {segment}", fig
