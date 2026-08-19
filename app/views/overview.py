"""Run overview: is this result trustworthy, and what is in it?

Deliberately the first tab.  A segment-resolved EIS result has several ways of
being quietly wrong - a DC closure that does not close, a plate where most
segments were inferred rather than measured, a card whose acquisition skew
could not be fitted - and all of them are invisible on a pretty heat map.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from dash import Input, Output, State, dcc, html, no_update

from app.services import runner, store
from app.services.figures import empty_figure, plate_heatmap
from app.settings import SETTINGS
from app.views import common as ui

layout_id = "overview"


def layout():
    return html.Div([
        html.Div(id="ov-header"),
        html.Div(id="ov-body"),
    ])


def _closure_tone(deviation: float | None) -> str:
    if deviation is None:
        return "muted"
    return "good" if abs(deviation) <= 5 else ("warn" if abs(deviation) <= 10 else "bad")


def register(app):

    @app.callback(Output("ov-header", "children"), Output("ov-body", "children"),
                  Input("selection", "data"))
    def _render(selection):
        if not selection or not selection.get("measurement_id"):
            return ui.note("Pick an order id in the sidebar."), html.Div()

        kind = selection["kind"]
        if kind == "famos":
            return _famos_header(selection), _famos_body(selection)
        if kind == "datago":
            return _datago_header(selection), _datago_body(selection)

        run = store.current_run(kind, selection["measurement_id"],
                                selection["condition"], selection["plate_key"])
        gold = run.meta.get("gold", {})
        closure = gold.get("dc_closure", {})
        deviation = closure.get("deviation_pct")
        tiers = gold.get("tiers", {})

        header = ui.stat_row([
            ("Order / condition", run.label()),
            ("Segments measured",
             f"{gold.get('n_measured', len(run.segment_names))}"
             f"/{gold.get('n_total', len(run.segment_names))}"),
            ("Inferred", str(gold.get("n_inferred", "—"))),
            ("Hardware-bad", str(gold.get("n_bad", "—"))),
            ("DC closure",
             "—" if deviation is None else f"{deviation:+.1f} %",
             _closure_tone(deviation)),
            ("Quality tiers",
             " / ".join(f"{k}:{v}" for k, v in sorted(tiers.items())) or "—"),
        ])

        blocks = [ui.warnings_block(run.warnings)]

        stats_rows = []
        for param in ("R_ohmic", "R_ct", "R_mt", "R_pol", "j_dc"):
            block = gold.get(param)
            if not isinstance(block, dict):
                continue
            stats_rows.append({
                "parameter": param,
                "mean": block.get("mean"), "sd": block.get("sd"),
                "min": block.get("min"), "max": block.get("max"),
                "spread (max/min)": block.get("spread"),
                "n used": block.get("n_used"),
            })

        left = [
            ui.panel([
                html.Div("Plate-level statistics",
                         style={"fontWeight": 650, "marginBottom": "8px"}),
                ui.note("From the pipeline's own manifest, in SI units "
                        "(Ω·cm², A/cm²) as the pipeline wrote them. "
                        "A large spread is the signal worth chasing: it means "
                        "the plate is not behaving as one cell."),
                ui.table(pd.DataFrame(stats_rows), "ov-stats", height="240px"),
            ]) if stats_rows else html.Div(),
            ui.panel([
                html.Div("DC closure", style={"fontWeight": 650, "marginBottom": "6px"}),
                ui.note("The measured segment currents, extrapolated over the "
                        "unmeasured area, against the load setpoint. This is "
                        "the one end-to-end check on the current calibration."),
                ui.kv_table([(k, _fmt(v)) for k, v in closure.items()])
                if closure else ui.note("not reported by this run"),
            ]),
        ]

        skew = run.meta.get("skew_table")
        right = [
            ui.panel([
                html.Div("Acquisition skew per card",
                         style={"fontWeight": 650, "marginBottom": "6px"}),
                ui.note("A multiplexed converter samples channel slot p at "
                        "p/(n_ch·fs). Left uncorrected this is tens of degrees "
                        "of phase error at the top of the band, which fans the "
                        "high-frequency arcs apart."),
                ui.table(pd.DataFrame(skew), "ov-skew", height="240px")
                if skew else ui.note("no skew table in this run"),
            ]),
            ui.panel([
                html.Div("Segment quality", style={"fontWeight": 650,
                                                   "marginBottom": "6px"}),
                dcc.Graph(figure=_class_figure(run), config=ui.GRAPH_CONFIG),
            ]),
        ]

        body = html.Div([
            html.Div(blocks),
            html.Div([
                html.Div(left, style={"flex": "1 1 420px", "minWidth": "360px"}),
                html.Div(right, style={"flex": "1 1 420px", "minWidth": "360px"}),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ])
        return header, body


    # -- running the pipeline over a raw selection --------------------------

    @app.callback(Output("ov-run-job", "data"), Output("ov-run-poll", "disabled"),
                  Input("ov-run-pipeline", "n_clicks"), State("selection", "data"),
                  prevent_initial_call=True)
    def _start(_n, selection):
        from app.plates import registry
        ref = store.find_ref("famos", selection["measurement_id"],
                             selection["condition"])
        if ref is None:
            return None, True
        geom = registry.get(selection["plate_key"])
        job_id = store.JOBS.submit(
            f"pipeline · {ref.measurement_id}/{ref.condition}",
            runner.run_famos, ref, geom, SETTINGS)
        return job_id, False

    @app.callback(Output("ov-run-status", "children"),
                  Output("ov-run-poll", "disabled", allow_duplicate=True),
                  Output("refresh-token", "data", allow_duplicate=True),
                  Output("sel-format", "value", allow_duplicate=True),
                  Input("ov-run-poll", "n_intervals"), State("ov-run-job", "data"),
                  State("refresh-token", "data"),
                  prevent_initial_call=True)
    def _watch(_n, job_id, token):
        job = store.JOBS.get(job_id or "")
        if job is None:
            return "", True, no_update, no_update
        if job.state == "running":
            detail = f" — {job.message}" if job.message else ""
            return ui.note(job.status_text() + detail), False, no_update, no_update
        if job.state == "failed":
            return (ui.warnings_block([job.error], "Pipeline run failed"), True,
                    no_update, no_update)
        # A finished run should not need the reader to press Refresh and change
        # the format selector to discover its own results: rescan, and switch
        # the picker to the processed output that now exists.
        return (ui.note(f"{job.status_text()}. Results are in {job.result}.",
                        "good"),
                True, store.bump_generation(), "results")


def _class_figure(run):
    from app.plates import registry
    try:
        geom = registry.get(run.plate_key)
    except KeyError:
        return empty_figure("select a plate generation")
    if run.segments is None or run.segments.empty:
        return empty_figure("no per-segment data in this run")
    classes = run.classes()
    numeric = {"measured": 2.0, "inferred": 1.0, "bad": 0.0}
    values = {s: numeric.get(c, 1.0) for s, c in classes.items()}
    faults = ({str(k): str(v) for k, v in
               zip(run.segments["segment"], run.segments.get("fault", ""))}
              if "fault" in run.segments.columns else {})
    fig = plate_heatmap(geom, values, "seg_class", classes=classes, faults=faults,
                        vmin=0, vmax=2, show_labels=True,
                        title="measured (dark) · inferred (hatched) · bad (crossed)")
    fig.update_layout(height=380)
    return fig


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.4g}"
    return value


# ---------------------------------------------------------------------------
# raw and catalogue-only selections
# ---------------------------------------------------------------------------

def _famos_header(selection):
    ref = store.find_ref("famos", selection["measurement_id"], selection["condition"])
    n_files = len(ref.files) if ref else 0
    return ui.stat_row([
        ("Order / condition",
         f"{selection['measurement_id']} / {selection['condition']}"),
        ("Raw card files", str(n_files)),
        ("Status", "not processed yet", "warn"),
    ])


def _famos_body(selection):
    ref = store.find_ref("famos", selection["measurement_id"], selection["condition"])
    if ref is None:
        return ui.note("selection no longer present; press Refresh")
    problems = runner.preflight(ref, SETTINGS)
    notes = runner.warnings_for(SETTINGS)
    files = pd.DataFrame({"file": [Path(f).name for f in ref.files],
                          "path": list(ref.files)})
    destination = runner.output_dir(ref, SETTINGS)
    command = runner.build_command(ref, destination, SETTINGS)

    return html.Div([
        ui.panel([
            html.Div("Raw FAMOS recordings",
                     style={"fontWeight": 650, "marginBottom": "6px"}),
            ui.note("These are recordings, not results. Bronze/silver/gold has "
                    "to run over them before there is a spectrum to plot — "
                    "minutes of CPU over gigabytes — so it runs as a "
                    "background job in its own process."),
            ui.table(files, "ov-famos-files", height="200px"),
            ui.kv_table([
                ("pipeline", str(runner.pipeline_dir(SETTINGS))),
                ("writes to", str(destination)),
            ]),
        ]),
        ui.warnings_block(problems, "Cannot run") if problems else
        ui.panel([
            html.Button("Run pipeline on this selection", id="ov-run-pipeline",
                        n_clicks=0,
                        style={"padding": "8px 14px", "border": "none",
                               "borderRadius": "6px", "cursor": "pointer",
                               "background": ui.COLOURS["accent"], "color": "white"}),
            ui.note("Equivalent command line, if you would rather run it "
                    "outside the app:", "muted"),
            html.Pre(" ".join(command[1:]),
                     style={"fontSize": "10.5px", "whiteSpace": "pre-wrap",
                            "background": ui.COLOURS["bg"], "padding": "8px",
                            "borderRadius": "5px", "margin": "4px 0 0 0"}),
            html.Div(id="ov-run-status", style={"marginTop": "10px"}),
            dcc.Interval(id="ov-run-poll", interval=2000, disabled=True),
            dcc.Store(id="ov-run-job"),
        ]),
        ui.warnings_block(notes, "Before you start") if notes else html.Div(),
    ])


def _datago_header(selection):
    return ui.stat_row([
        ("Order", selection["measurement_id"]),
        ("Source", "datago metadata"),
        ("Status", "catalogue entry", "muted"),
    ])


def _datago_body(_selection):
    return ui.panel([
        html.Div("datago catalogue entry",
                 style={"fontWeight": 650, "marginBottom": "6px"}),
        ui.note("datago lists which orders were measured. To see spectra, "
                "point the format selector at processed results, or run the "
                "pipeline over the order's recordings and let the output land "
                "in the results root."),
    ])
