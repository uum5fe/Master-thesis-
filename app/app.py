"""Local EIS Viewer — a Dash application over the bronze/silver/gold results.

Run locally::

    EIS_RESULTS_ROOT=/path/to/results python -m app.app

or as a Databricks App, where ``app.yaml`` at the repository root supplies the
same environment variables and the platform supplies the port.

Three choices drive everything else and they all live in the sidebar, in one
place, so that nothing in this application knows a personal file path:

* **where** the data is — a Volumes/file-system root, or the datago tables;
* **what format** it is in — finished results (CSV/Parquet) or raw FAMOS
  ``.DAT`` that has yet to be processed;
* **which plate generation** produced it — Gen 1 today, Gen 2 and Gen 3 as
  soon as somebody writes their spec file, with no code change here.
"""

from __future__ import annotations

from dash import Dash, Input, Output, State, dcc, html

from app.data.sources import _condition_sort_key
from app.plates import registry
from app.services import store
from app.settings import SETTINGS
from app.views import common as ui
from app.views import compare, ecm, heatmap, overview, plates, spectra

LOCATIONS = [
    {"label": "Volumes / file system", "value": "volumes"},
    {"label": "datago (Unity Catalog)", "value": "datago"},
]

FORMATS = [
    {"label": "Auto-detect", "value": "auto"},
    {"label": "Processed results — CSV / Parquet", "value": "results"},
    {"label": "Raw recordings — FAMOS .DAT", "value": "famos"},
]

#: Selected (location, format) -> the source kinds the catalogue should offer.
def kinds_for(location: str, fmt: str) -> tuple[str, ...]:
    if location == "datago":
        return ("datago",)
    if fmt == "results":
        return ("results",)
    if fmt == "famos":
        return ("famos",)
    return ("results", "famos")


def sidebar() -> html.Div:
    plate_options = registry.options()
    default_plate = SETTINGS.default_plate or (
        plate_options[0]["value"] if plate_options else None)

    return html.Div([
        html.Div([
            html.Div("Local EIS Viewer", style={"fontSize": "17px", "fontWeight": 700}),
            html.Div("segment-resolved impedance, plate maps and circuit fits",
                     style={"fontSize": "11px", "color": ui.COLOURS["muted"],
                            "marginTop": "2px"}),
        ], style={"marginBottom": "16px"}),

        ui.panel([
            ui.field("Data location",
                     dcc.RadioItems(id="sel-location", options=LOCATIONS,
                                    value="volumes",
                                    labelStyle={"display": "block",
                                                "fontSize": "13px",
                                                "marginBottom": "3px"}),
                     "Volumes reads a mounted path; datago queries the "
                     "measurement metadata tables."),
            ui.field("File format",
                     dcc.Dropdown(id="sel-format", options=FORMATS, value="auto",
                                  clearable=False),
                     "Raw .DAT has to go through the pipeline before it has a "
                     "spectrum to show."),
            ui.field("Order ID (Leepa)",
                     dcc.Dropdown(id="sel-measurement", options=[], value=None,
                                  placeholder="select an order id",
                                  clearable=False)),
            ui.field("Current condition",
                     dcc.Dropdown(id="sel-condition", options=[], value=None,
                                  placeholder="select a condition",
                                  clearable=False)),
            ui.field("Plate generation",
                     dcc.Dropdown(id="sel-plate", options=plate_options,
                                  value=default_plate, clearable=False),
                     "Segment arrangement, areas and known faults come from "
                     "the selected generation's spec."),
            html.Button("Refresh sources", id="btn-refresh", n_clicks=0,
                        style={"width": "100%", "padding": "8px",
                               "background": ui.COLOURS["accent"], "color": "white",
                               "border": "none", "borderRadius": "6px",
                               "fontSize": "13px", "cursor": "pointer"}),
        ]),

        html.Div(id="sidebar-status"),
    ], style={"width": "290px", "minWidth": "290px", "padding": "16px",
              "background": ui.COLOURS["bg"], "height": "100vh",
              "overflowY": "auto", "borderRight": f"1px solid {ui.COLOURS['line']}"})


def build_app() -> Dash:
    app = Dash(__name__, title=SETTINGS.title, suppress_callback_exceptions=True,
               update_title=None)

    app.layout = html.Div([
        dcc.Store(id="selection"),
        dcc.Store(id="refresh-token", data=0),
        sidebar(),
        html.Div([
            dcc.Tabs(id="tabs", value="tab-overview", children=[
                dcc.Tab(label="Overview", value="tab-overview"),
                dcc.Tab(label="Plate map", value="tab-heatmap"),
                dcc.Tab(label="Spectra", value="tab-spectra"),
                dcc.Tab(label="ECM fitting", value="tab-ecm"),
                dcc.Tab(label="Conditions", value="tab-compare"),
                dcc.Tab(label="Plate & sources", value="tab-plates"),
            ]),
            html.Div(id="tab-body", style={"padding": "16px 18px"}),
        ], style={"flex": 1, "height": "100vh", "overflowY": "auto"}),
    ], style={"display": "flex", "fontFamily":
              "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
              "color": ui.COLOURS["text"], "margin": 0})

    register_selection(app)
    for view in (overview, heatmap, spectra, ecm, compare, plates):
        view.register(app)

    @app.callback(Output("tab-body", "children"), Input("tabs", "value"))
    def _render_tab(tab):
        return {
            "tab-overview": overview.layout,
            "tab-heatmap": heatmap.layout,
            "tab-spectra": spectra.layout,
            "tab-ecm": ecm.layout,
            "tab-compare": compare.layout,
            "tab-plates": plates.layout,
        }[tab]()

    return app


def register_selection(app: Dash) -> None:

    @app.callback(Output("refresh-token", "data"),
                  Input("btn-refresh", "n_clicks"), prevent_initial_call=True)
    def _refresh(_n):
        registry.reload()
        return store.bump_generation()

    @app.callback(
        Output("sel-measurement", "options"),
        Output("sel-measurement", "value"),
        Input("sel-location", "value"),
        Input("sel-format", "value"),
        Input("refresh-token", "data"),
        State("sel-measurement", "value"),
    )
    def _measurements(location, fmt, _token, current):
        catalog = store.current_catalog()
        kinds = kinds_for(location, fmt)
        ids = sorted({r.measurement_id for r in catalog.runs if r.kind in kinds})
        options = [{"label": i, "value": i} for i in ids]
        value = current if current in ids else (ids[0] if ids else None)
        return options, value

    @app.callback(
        Output("sel-condition", "options"),
        Output("sel-condition", "value"),
        Input("sel-measurement", "value"),
        Input("sel-location", "value"),
        Input("sel-format", "value"),
        Input("refresh-token", "data"),
        State("sel-condition", "value"),
    )
    def _conditions(measurement_id, location, fmt, _token, current):
        if not measurement_id:
            return [], None
        catalog = store.current_catalog()
        kinds = kinds_for(location, fmt)
        conds = sorted({r.condition for r in catalog.runs
                        if r.measurement_id == measurement_id and r.kind in kinds},
                       key=_condition_sort_key)
        options = [{"label": c, "value": c} for c in conds]
        value = current if current in conds else (conds[0] if conds else None)
        return options, value

    @app.callback(
        Output("selection", "data"),
        Output("sidebar-status", "children"),
        Input("sel-measurement", "value"),
        Input("sel-condition", "value"),
        Input("sel-plate", "value"),
        Input("sel-location", "value"),
        Input("sel-format", "value"),
        Input("refresh-token", "data"),
    )
    def _selection(measurement_id, condition, plate_key, location, fmt, _token):
        catalog = store.current_catalog()
        kinds = kinds_for(location, fmt)
        ref = None
        for kind in kinds:
            ref = catalog.find(measurement_id or "", condition or "", kind)
            if ref is not None:
                break

        selection = {
            "kind": ref.kind if ref else (kinds[0] if kinds else "results"),
            "measurement_id": measurement_id or "",
            "condition": condition or "",
            "plate_key": plate_key or "",
            "location": location, "format": fmt,
            "layout": ref.layout if ref else "",
            "path": ref.path if ref else "",
        }

        messages = list(catalog.messages)
        if not catalog.runs:
            messages.append(
                "No runs discovered. Set EIS_RESULTS_ROOT (finished results) "
                "and/or EIS_FAMOS_ROOT (raw .DAT) and press Refresh.")
        body = [
            ui.kv_table([
                ("source", ref.kind if ref else "—"),
                ("layout", ref.describe() if ref else "—"),
                ("path", ref.path if ref else "—"),
                ("modified", ref.modified_text if ref else "—"),
            ]),
        ]
        status = [ui.panel(body)]
        if messages:
            status.append(ui.warnings_block(messages, "Source notes"))
        return selection, status


app = build_app()
server = app.server                     # gunicorn / Databricks Apps entry point


def banner(host: str, port: int) -> str:
    """What to print at startup, including a URL that can actually be opened.

    Dash does not reliably print its own banner when the reloader is off, and a
    server that starts in silence looks like a server that did not start. The
    bind address is also not the address to visit: binding to 0.0.0.0 is
    required so a container can be reached from outside, but 0.0.0.0 is not a
    destination - so the link shown is always a loopback one.
    """
    url = f"http://127.0.0.1:{port}"
    catalog = store.current_catalog()
    runs = catalog.runs
    orders = sorted({r.measurement_id for r in runs})

    lines = [
        "",
        "=" * 68,
        f"  {SETTINGS.title}",
        "=" * 68,
        f"  Open this link:   {url}",
        f"  (listening on {host}:{port} — press Ctrl+C to stop)",
        "",
    ]
    if runs:
        lines.append(f"  Found {len(runs)} run(s) across {len(orders)} order id(s): "
                     f"{', '.join(orders[:6])}{' …' if len(orders) > 6 else ''}")
    else:
        lines += [
            "  No measurements found. The app will start, but every dropdown",
            "  will be empty. Point it at your data and restart:",
            "",
            "      EIS_RESULTS_ROOT=/path/to/results   finished pipeline output",
            "      EIS_FAMOS_ROOT=/path/to/Famos       raw .DAT recordings",
            "",
            "  EIS_RESULTS_ROOT expects <root>/<order id>/<condition>/{gold,silver}/",
        ]
    for message in catalog.messages:
        lines.append(f"  note: {message}")
    lines += ["", "=" * 68, ""]
    return "\n".join(lines)


def serve(host: str | None = None, port: int | None = None,
          debug: bool | None = None, open_browser: bool = False) -> None:
    """Start the development server, having said where to find it."""
    host = host or SETTINGS.host
    port = int(port or SETTINGS.port)
    debug = SETTINGS.debug if debug is None else debug

    print(banner(host, port), flush=True)
    if open_browser:
        import threading
        import webbrowser
        threading.Timer(1.5, webbrowser.open,
                        args=(f"http://127.0.0.1:{port}",)).start()

    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    serve()
