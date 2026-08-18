"""Plate generations and sources: what the app is looking at, and is it sound.

Two jobs.  First, the integrity check on a plate spec - do its segments tile
the pad grid exactly once, and do their areas add up to the active area?  A
Gen-2 spec with a typo produces a heat map that looks entirely plausible and is
wrong, so the check is one click away rather than buried in a test suite.
Second, the configuration panel: every path the app reads, named, so that
"it cannot find my measurement" is answerable without reading the code.
"""

from __future__ import annotations

import pandas as pd
from dash import Input, Output, dcc, html

from app.plates import registry
from app.services import store
from app.services.figures import empty_figure, plate_heatmap
from app.settings import SETTINGS
from app.views import common as ui

ADD_GENERATION = """\
A new plate generation is a JSON file, not a code change.

1. Copy app/plates/specs/_gen2_template.json to gen2_<name>.json — inside the
   repository, or into any directory named by EIS_PLATE_SPEC_DIR (a Volumes
   path works, so a colleague can add one without a deployment).
2. Describe the pad grid (pad_w_mm, pad_h_mm, n_cols, n_rows) and then either
   the strips/bands construction or one explicit rectangle per segment.
3. Press Refresh sources. The generation appears in the sidebar dropdown and
   the check below tells you whether its segments tile the plate exactly once.
"""


def layout():
    return html.Div([
        html.Div([
            html.Div([
                ui.panel([
                    html.Div("Plate generation", style={"fontWeight": 650,
                                                        "marginBottom": "6px"}),
                    html.Div(id="pl-summary"),
                ]),
                ui.panel([
                    html.Div("Geometry check", style={"fontWeight": 650,
                                                      "marginBottom": "6px"}),
                    ui.note("Segments must cover every pad exactly once and "
                            "their areas must add up to the active area. A map "
                            "drawn from a spec that fails this looks plausible "
                            "and is wrong."),
                    html.Div(id="pl-check"),
                ]),
                ui.panel([
                    html.Div("Adding a generation", style={"fontWeight": 650,
                                                           "marginBottom": "6px"}),
                    html.Pre(ADD_GENERATION,
                             style={"fontSize": "11px", "whiteSpace": "pre-wrap",
                                    "lineHeight": "1.55", "margin": 0,
                                    "color": ui.COLOURS["muted"]}),
                ]),
            ], style={"flex": "2 1 380px", "minWidth": "340px"}),
            html.Div([ui.panel([ui.graph("pl-layout")])],
                     style={"flex": "3 1 520px", "minWidth": "440px"}),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),

        ui.panel([
            html.Div("Configured sources", style={"fontWeight": 650,
                                                  "marginBottom": "6px"}),
            ui.note("Every path comes from an environment variable, so the same "
                    "image runs on a laptop and as a Databricks App without a "
                    "personal path anywhere in the code."),
            ui.kv_table(SETTINGS.summary()),
        ]),
        ui.panel([
            html.Div("Discovered runs", style={"fontWeight": 650,
                                               "marginBottom": "6px"}),
            html.Div(id="pl-catalog"),
        ]),
    ])


def register(app):

    @app.callback(Output("pl-summary", "children"), Output("pl-check", "children"),
                  Output("pl-layout", "figure"), Output("pl-catalog", "children"),
                  Input("selection", "data"))
    def _render(selection):
        catalog = store.current_catalog()
        runs = pd.DataFrame([{
            "source": r.kind, "order id": r.measurement_id, "condition": r.condition,
            "layout": r.layout, "files": len(r.files) or "",
            "modified": r.modified_text, "path": r.path,
        } for r in catalog.runs])
        catalog_table = ui.table(runs, "pl-catalog-table", max_rows=300, height="320px")

        key = (selection or {}).get("plate_key") or ""
        try:
            geom = registry.get(key)
        except KeyError:
            available = ", ".join(sorted(registry.available())) or "none"
            message = f"no plate generation selected (available: {available})"
            return ui.note(message), ui.note(""), empty_figure(message), catalog_table

        summary = ui.kv_table([
            ("key", geom.key), ("name", geom.name),
            ("segments", geom.n_segments),
            ("pad grid", f"{geom.n_cols} × {geom.n_rows} of "
                         f"{geom.pad_w_mm} × {geom.pad_h_mm} mm"),
            ("active area", f"{geom.plate_w_mm:.1f} × {geom.plate_h_mm:.1f} mm "
                            f"= {geom.cell_area_cm2:.2f} cm²"),
            ("segment areas", f"{min(geom.areas().values()):.3f} … "
                              f"{max(geom.areas().values()):.3f} cm²"),
            ("flow axis", geom.flow_axis),
            ("known faults", ", ".join(sorted(geom.known_bad)) or "none"),
            ("spec file", geom.source),
        ])

        check = geom.self_check()
        ok = check["tiles_exactly"] and check["area_closes"]
        rows = [
            ("tiling", "exact" if check["tiles_exactly"] else "BROKEN"),
            ("pads covered", f"{check['pads_covered']} / {check['pads_total']}"),
            ("area sum", f"{check['area_sum_cm2']:.4f} / "
                         f"{check['cell_area_cm2']:.4f} cm²"),
        ]
        if check["uncovered_pads"]:
            rows.append(("uncovered pads (col,row)",
                         ", ".join(f"{c},{r}" for c, r in check["uncovered_pads"][:12])
                         + (" …" if len(check["uncovered_pads"]) > 12 else "")))
        if check["overlapping_pads"]:
            rows.append(("overlapping pads",
                         ", ".join(list(check["overlapping_pads"])[:12])))
        check_block = html.Div([
            html.Div("PASS" if ok else "FAIL",
                     style={"fontWeight": 700, "marginBottom": "6px",
                            "color": ui.COLOURS["good"] if ok else ui.COLOURS["bad"]}),
            ui.kv_table(rows),
            html.Div(geom.notes, style={"fontSize": "11px", "marginTop": "8px",
                                        "color": ui.COLOURS["muted"],
                                        "lineHeight": "1.5"}) if geom.notes else html.Div(),
        ])

        fig = plate_heatmap(geom, geom.areas(), "area_cm2",
                            title=f"{geom.name} — segment areas [cm²]")
        fig.update_layout(height=470)
        return summary, check_block, fig, catalog_table
