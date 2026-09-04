"""A Dash tab that draws the realistic plate for the selected run.

Drop-in for `app/views/`: it follows the same shape as `heatmap.py` --
`layout()` plus `register(app)`, reading `selection` and `app.services.store`
-- so wiring it in is two lines in `app/app.py`:

    from app.views import plate_realistic
    ...
    dcc.Tab(label="Plate (realistic)", value="tab-plate3d"),
    ...
    plate_realistic.register(app)

and one branch in the tab-content callback:

    if tab == "tab-plate3d":
        return plate_realistic.layout()

The colour scale is fixed at robust 2-98 % percentiles, as everywhere else in
this app: one runaway segment stretches a full-range scale until every real
difference on the plate flattens into the same colour.
"""

from __future__ import annotations

from dash import Input, Output, State, dcc, html

from app.data.model import PARAM_META, param_label
from app.services import store
from app.views import common as ui

from app.plates.geometry import PLATE, plate_figure

#: param -> the colour scale it is read with, matching figures.PARAM_META.
_SCALES = {
    "j": "Inferno", "rs": "Viridis", "rp": "Magma", "hfr": "Viridis",
    "coherence": "Cividis", "temperature": "RdYlBu_r",
}


def layout():
    return html.Div([
        ui.panel([
            html.Div([
                html.Div(ui.field("Parameter",
                                  dcc.Dropdown(id="pl-param", options=[], value=None,
                                               clearable=False)),
                         style={"flex": "2 1 260px"}),
                html.Div(ui.field("Side",
                                  dcc.RadioItems(id="pl-side",
                                                 options=[{"label": " cathode · air", "value": "cathode"},
                                                          {"label": " anode · H₂", "value": "anode"}],
                                                 value="cathode",
                                                 labelStyle={"display": "block", "fontSize": "13px"})),
                         style={"flex": "0 1 170px"}),
                html.Div(ui.field("Numbering",
                                  dcc.RadioItems(id="pl-scheme",
                                                 options=[{"label": " as wired", "value": "wired"},
                                                          {"label": " top-left down", "value": "topleft"}],
                                                 value="wired",
                                                 labelStyle={"display": "block", "fontSize": "13px"})),
                         style={"flex": "0 1 180px"}),
                html.Div(ui.field("Overlay",
                                  dcc.Checklist(id="pl-opts",
                                                options=[{"label": " numbers", "value": "num"},
                                                         {"label": " channels", "value": "chan"}],
                                                value=["num", "chan"])),
                         style={"flex": "0 1 150px"}),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
            ui.note("H₂ enters bottom left and leaves top right; air enters bottom "
                    "right and leaves top left — the two gases are in counter-flow, "
                    "so each inlet faces the other's outlet."),
        ]),
        ui.panel([ui.graph("pl-map")]),
        ui.panel([
            html.Div(id="pl-click-title",
                     style={"fontSize": "13px", "fontWeight": 650,
                            "color": ui.COLOURS["text"], "marginBottom": "8px"}),
            html.Div(id="pl-click-body"),
        ]),
    ])


def register(app):

    @app.callback(Output("pl-param", "options"), Output("pl-param", "value"),
                  Input("selection", "data"), State("pl-param", "value"))
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

    @app.callback(Output("pl-map", "figure"),
                  Input("selection", "data"), Input("pl-param", "value"),
                  Input("pl-side", "value"), Input("pl-scheme", "value"),
                  Input("pl-opts", "value"))
    def _map(selection, param, side, scheme, opts):
        from app.services.figures import empty_figure
        if not selection or not param:
            return empty_figure("select an order id and a parameter")
        run = store.current_run(selection["kind"], selection["measurement_id"],
                                selection["condition"], selection["plate_key"])
        raw = run.value(param)
        if not raw:
            return empty_figure(f"{param} is not present in this run")

        # run.value() keys segments as strings; the plate keys them as ints.
        values = {}
        for key, v in raw.items():
            try:
                values[int(key)] = float(v)
            except (TypeError, ValueError):
                continue

        meta = PARAM_META.get(param, {})
        return plate_figure(
            values, label=meta.get("label", param), unit=meta.get("unit", ""),
            colorscale=_SCALES.get(param, meta.get("colorscale", "RdYlBu_r")),
            scheme=scheme, side=side,
            show_numbers="num" in (opts or []), show_channels="chan" in (opts or []),
            title=f"{param_label(param) if param in PARAM_META else param} — {run.label()}")

    @app.callback(Output("pl-click-title", "children"), Output("pl-click-body", "children"),
                  Input("pl-map", "clickData"), State("pl-scheme", "value"))
    def _clicked(click, scheme):
        if not click:
            return "Click a segment", ui.note("click a segment on the plate for its details")
        wired = click["points"][0].get("customdata")
        if isinstance(wired, list):
            wired = wired[0]
        if wired is None:
            return "Click a segment", ui.note("no segment under the cursor")
        seg = PLATE.segments[int(wired)]
        shown = seg.wired if scheme == "wired" else seg.topleft
        cx, cy = seg.centroid_mm
        return f"Segment {shown}", ui.kv_table([
            ("as wired", seg.wired),
            ("top-left down", seg.topleft),
            ("pads", seg.n_pads),
            ("area", f"{seg.area_cm2:.3f} cm²"),
            ("centroid", f"{cx:.1f}, {cy:.1f} mm"),
            ("shape", "rectangle" if len(seg.rings_mm()[0]) == 5 else "staircase"),
        ])
