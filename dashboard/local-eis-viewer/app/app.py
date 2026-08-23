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

import os

from dash import Dash, Input, Output, State, dcc, html

from app.data.sources import _condition_sort_key
from app.plates import registry
from app.services import store
from app.settings import DOTENV_LOADED as SETTINGS_DOTENV
from app.settings import SETTINGS
from app.views import common as ui
from app.views import (ecm, heatmap, overview, reference, signals,
                       spectra)

def location_options() -> list[dict]:
    """Where data can come from — as this deployment is actually configured.

    "Volumes" and "datago" are Databricks vocabulary. On a laptop reading a
    local folder or a network share they are noise, and offering a source that
    cannot work is worse than noise: it invites the reader to pick it and
    wonder why nothing appears. So the file-system option is named after what
    it reads here, and datago is offered only where it is configured.
    """
    on_databricks = bool(SETTINGS.databricks_host or SETTINGS.warehouse_id)
    options = [{
        "label": "Volumes / file system" if on_databricks
                 else "Local or network folder",
        "value": "volumes",
    }]
    if SETTINGS.datago_metadata_table:
        options.append({"label": "datago (Unity Catalog)", "value": "datago"})
    return options


def location_hint(options: list[dict]) -> str:
    if len(options) > 1:
        return ("The file system reads a mounted or network path; datago "
                "queries the measurement metadata tables.")
    roots = SETTINGS.results_roots + SETTINGS.famos_roots
    if roots:
        # A deep share path is longer than the sidebar; the tail identifies the
        # folder, the head rarely does.
        shown = [r if len(r) <= 46 else "…" + r[-45:] for r in roots[:3]]
        if len(roots) > 3:
            shown.append(f"… and {len(roots) - 3} more")
        return "Reading: " + " · ".join(shown)
    return ("No folders configured. Run `python run_dashboard.py --init`, "
            "then set EIS_FAMOS_ROOT and EIS_RESULTS_ROOT in .env.")


#: What can be selected, named for what it *is* rather than for the file
#: extension it happens to use.  "Processed results — CSV / Parquet" and the
#: R2-D2 logger's measurement files are both CSV and are completely different
#: things; sharing one menu entry between them meant choosing "CSV" and being
#: shown a list of FAMOS order ids.
FORMATS = [
    {"label": "Auto-detect", "value": "auto"},
    {"label": "Pipeline results — already evaluated", "value": "results"},
    {"label": "Raw recording — FAMOS .DAT (5 cards)", "value": "famos"},
    {"label": "Raw sweep — R2-D2 CSV logger folder", "value": "csvlog"},
]

#: Formats that hold raw signals rather than finished spectra.  These are the
#: ones the Signals tab can open and the ones that have to be evaluated before
#: a plate map exists.
RAW_FORMATS = ("famos", "csvlog")


#: Selected (location, format) -> the source kinds the catalogue should offer.
def kinds_for(location: str, fmt: str) -> tuple[str, ...]:
    if location == "datago":
        return ("datago",)
    if fmt in ("results", "famos", "csvlog"):
        return (fmt,)
    return ("results", "famos", "csvlog")


def sidebar() -> html.Div:
    plate_options = registry.options()
    locations = location_options()
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
            ui.field("Where the data is",
                     dcc.RadioItems(id="sel-location", options=locations,
                                    value="volumes",
                                    labelStyle={"display": "block",
                                                "fontSize": "13px",
                                                "marginBottom": "3px"},
                                    # One choice is not a choice: keep the
                                    # component so the callbacks still have it,
                                    # but do not draw a radio button nobody can
                                    # move.
                                    style={} if len(locations) > 1
                                    else {"display": "none"}),
                     location_hint(locations)),
            ui.field("File format",
                     dcc.Dropdown(id="sel-format", options=FORMATS, value="auto",
                                  clearable=False),
                     "Pipeline results plot straight away. A raw FAMOS or "
                     "CSV recording has to be evaluated first — open it on "
                     "the Signals tab to see the recording itself."),
            ui.field("Order ID (Leepa)",
                     dcc.Dropdown(id="sel-measurement", options=[], value=None,
                                  placeholder="select an order id",
                                  clearable=False),
                     label_id="label-measurement"),
            ui.field("Current condition",
                     dcc.Dropdown(id="sel-condition", options=[], value=None,
                                  placeholder="select a condition",
                                  clearable=False),
                     label_id="label-condition", field_id="field-condition"),
            ui.field("Plate generation",
                     dcc.Dropdown(id="sel-plate", options=plate_options,
                                  value=default_plate, clearable=False),
                     "Segment arrangement and areas come from the selected "
                     "generation's spec."),
            html.Div(id="plate-status",
                     style={"fontSize": "11px", "marginTop": "-6px",
                            "marginBottom": "12px"}),
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
                dcc.Tab(label="Signals", value="tab-signals"),
                dcc.Tab(label="Whole-cell check", value="tab-reference"),
            ]),
            html.Div(id="tab-body", style={"padding": "16px 18px"}),
        ], style={"flex": 1, "height": "100vh", "overflowY": "auto"}),
    ], style={"display": "flex", "fontFamily":
              "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif",
              "color": ui.COLOURS["text"], "margin": 0})

    register_selection(app)
    for view in (overview, heatmap, spectra, ecm, signals, reference):
        view.register(app)

    @app.callback(Output("tab-body", "children"), Input("tabs", "value"))
    def _render_tab(tab):
        return {
            "tab-overview": overview.layout,
            "tab-heatmap": heatmap.layout,
            "tab-spectra": spectra.layout,
            "tab-ecm": ecm.layout,
            "tab-signals": signals.layout,
            "tab-reference": reference.layout,
        }[tab]()

    return app


def register_selection(app: Dash) -> None:

    @app.callback(Output("refresh-token", "data"),
                  Input("btn-refresh", "n_clicks"), prevent_initial_call=True)
    def _refresh(_n):
        registry.reload()
        return store.bump_generation()

    @app.callback(
        Output("label-measurement", "children"),
        Output("label-condition", "children"),
        Output("field-condition", "style"),
        Input("sel-format", "value"))
    def _relabel(fmt):
        """Each format is identified by different things, so ask for those.

        A FAMOS campaign is an order id at a current setpoint.  A CSV logger
        sweep is a cell at one operating point, and the thing that identifies
        it is the sweep folder, not a current in amps -- so the second field
        is named for what it holds rather than being labelled "Current
        condition" and then filled with folder names.
        """
        shown = {"marginBottom": "12px"}
        if fmt == "csvlog":
            return "Cell (Leepa)", "Sweep folder", shown
        return "Order ID (Leepa)", "Current condition", shown

    @app.callback(Output("plate-status", "children"), Input("sel-plate", "value"))
    def _plate_status(plate_key):
        """The plate self-check, in one line, always visible.

        A map drawn from a spec that does not tile the pad grid looks exactly
        as convincing as one drawn from a spec that does.  This used to live on
        its own tab, which meant it was seen once and then never again; a line
        under the selector is seen every time the plate is chosen.
        """
        if not plate_key:
            return ""
        try:
            geom = registry.get(plate_key)
        except Exception as exc:                      # noqa: BLE001
            return ui.note(f"plate spec unreadable: {exc}")
        check = geom.self_check()
        areas = geom.areas().values()
        ok = bool(check["tiles_exactly"] and check["area_closes"])
        detail = (f"{check['n_segments']} segments · "
                  f"{check['pads_covered']}/{check['pads_total']} pads · "
                  f"{check['area_sum_cm2']:.2f} cm² · "
                  f"{min(areas):.2f}–{max(areas):.2f} cm²")
        if not ok:
            return html.Div(["✗ geometry does not tile — ", detail],
                            style={"color": ui.COLOURS.get("bad", "#a3320b")})
        if not geom.verified:
            return html.Div(["⚠ layout unverified — ", detail],
                            style={"color": ui.COLOURS.get("warn", "#8a6d1f")})
        return html.Div(["✓ ", detail], style={"color": ui.COLOURS["muted"]})

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


def in_databricks_notebook() -> bool:
    return bool(os.environ.get("DATABRICKS_RUNTIME_VERSION"))


def driver_proxy_url(port: int) -> str:
    """The URL that reaches this port when the server runs on a cluster driver.

    Started from a Databricks notebook, the server listens on the *driver node*.
    ``127.0.0.1`` there is the driver's own loopback, not the browser's, so the
    ordinary local link cannot resolve. Databricks exposes driver ports through
    a proxy path instead, which is what has to be printed in that case.

    Returns an empty string when the workspace does not expose the proxy, or
    when the tags this is built from are unavailable - better to print nothing
    than a link that 404s.
    """
    try:
        from pyspark.sql import SparkSession        # type: ignore
        spark = SparkSession.getActiveSession()
        if spark is None:
            return ""
        conf = spark.conf
        org = conf.get("spark.databricks.clusterUsageTags.clusterOwnerOrgId")
        cluster = conf.get("spark.databricks.clusterUsageTags.clusterId")
        host = (os.environ.get("DATABRICKS_HOST")
                or conf.get("spark.databricks.workspaceUrl", ""))
    except Exception:
        return ""
    if not (org and cluster and host):
        return ""
    host = host.replace("https://", "").rstrip("/")
    return f"https://{host}/driver-proxy/o/{org}/{cluster}/{port}/"


def banner(host: str, port: int) -> str:
    """What to print at startup, including a URL that can actually be opened.

    Dash does not reliably print its own banner when the reloader is off, and a
    server that starts in silence looks like a server that did not start. The
    bind address is also not the address to visit either: binding to 0.0.0.0 is
    required so a container can be reached from outside, but 0.0.0.0 is not a
    destination - so the link shown is a loopback one, or, on a cluster driver,
    the proxy path that actually reaches it.
    """
    url = f"http://127.0.0.1:{port}"
    proxy = driver_proxy_url(port) if in_databricks_notebook() else ""
    catalog = store.current_catalog()
    runs = catalog.runs
    orders = sorted({r.measurement_id for r in runs})

    lines = [
        "",
        "=" * 68,
        f"  {SETTINGS.title}",
        "=" * 68,
        f"  Open this link:   {proxy or url}",
        f"  (listening on {host}:{port} — press Ctrl+C to stop)",
        "",
    ]
    if in_databricks_notebook():
        lines += [
            "  Running on the cluster driver, so this cell will stay busy for",
            "  as long as the server is up - that is the server working, not a",
            "  hang. Interrupt the cell to stop it.",
            "",
        ]
        if not proxy:
            lines += [
                "  This workspace did not expose a driver-proxy URL, so the link",
                "  above is the DRIVER's loopback and your browser cannot reach",
                "  it. Deploy as a Databricks App instead - see app.yaml and",
                "  docs/FRONTEND.md - which gives a real, shareable URL.",
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
    if SETTINGS_DOTENV:
        lines.append(f"  settings from: {SETTINGS_DOTENV}")
    elif not runs:
        lines.append("  tip: copy .env.example to .env and put your paths in it")
    lines += ["", "=" * 68, ""]
    return "\n".join(lines)


def quiet_request_log() -> None:
    """Stop logging one line per HTTP request.

    A Dash page is dozens of requests - one per callback, plus a poll every two
    seconds while a background job runs. Every line says 200 and none of them
    says anything, and together they bury the one message that would matter.

    Only the INFO-level access log is suppressed. Warnings and errors from the
    server still print, and so do tracebacks from callbacks, which Flask logs
    through its own logger.
    """
    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


def serve(host: str | None = None, port: int | None = None,
          debug: bool | None = None, open_browser: bool = False) -> None:
    """Start the development server, having said where to find it."""
    host = host or SETTINGS.host
    port = int(port or SETTINGS.port)
    debug = SETTINGS.debug if debug is None else debug

    print(banner(host, port), flush=True)

    if not debug:
        quiet_request_log()

    if open_browser:
        import threading
        import webbrowser
        threading.Timer(1.5, webbrowser.open,
                        args=(f"http://127.0.0.1:{port}",)).start()

    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    serve()
