"""How far did each segment get, and what stopped it?

Two questions this answers, both of which used to be unanswerable from the
outputs even though the evidence was computed:

  * "my impedance stops at 400 Hz"  -- which of the nine gates in silver
    removed the points above that, per segment.
  * "segment 33 has no spectrum"    -- whether it was never wired, or was
    measured and then rejected, and by what.

It reads `segment_reach.csv` and `point_rejections.csv`, which the pipeline
writes beside the silver tables. A run made before those existed will not have
them, and this tab says so rather than showing an empty screen.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from app.plates import registry
from app.services import store
from app.services.figures import TEMPLATE, empty_figure, plate_heatmap
from app.views import common as ui

#: Gate -> how it should read on screen. Ordered worst-understood first.
REASON_LABEL = {
    "no_channel": "never wired",
    "snr": "SNR below the gate",
    "uncertainty": "uncertainty too large",
    "cycles": "too few cycles in the dwell",
    "magnitude": "|Z| implausible",
    "zmag_outlier": "local |Z| outlier",
    "passivity": "Re Z negative above the passivity gate",
    "thd": "harmonic distortion",
    "drift": "amplitude drifted",
    "outside_band": "outside the configured band",
    "not_finite": "fit did not converge",
}

_HINT = (
    "This run has no coverage report. It is written by the pipeline as "
    "segment_reach.csv and point_rejections.csv beside the silver tables, so "
    "re-run the evaluation to produce it."
)


def layout():
    return html.Div([
        ui.panel([
            ui.section_title("How far each segment got, and what stopped it"),
            ui.note("Nine gates run in sequence when a spectrum is cleaned. "
                    "The gate that removed the most points ABOVE a segment's "
                    "highest surviving frequency is the one limiting its "
                    "bandwidth — that is what is reported here, rather than "
                    "the most common reason overall."),
            html.Div(id="cv-status", style={"marginTop": "8px"}),
        ]),
        ui.panel([
            ui.section_title("Highest frequency reached, per segment"),
            ui.graph("cv-map", height="500px"),
        ]),
        html.Div([
            html.Div([ui.panel([
                ui.section_title("What removed the points"),
                ui.graph("cv-reasons", height="320px"),
            ])], style={"flex": "1 1 420px", "minWidth": "360px"}),
            html.Div([ui.panel([
                ui.section_title("Where in frequency they were removed"),
                ui.note("Each gate against frequency. A gate that only bites "
                        "at the top of the band is a bandwidth limit; one "
                        "spread across the band is a different problem."),
                ui.graph("cv-vs-freq", height="320px"),
            ])], style={"flex": "1 1 420px", "minWidth": "360px"}),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ui.panel([
            ui.section_title("Segment by segment"),
            html.Div(id="cv-table"),
        ]),
    ])


def _run_dir(selection) -> Path | None:
    run = store.current_catalog().find(
        selection.get("measurement_id", ""), selection.get("condition", ""),
        selection.get("kind", "results"))
    if run is None or not run.path:
        return None
    base = Path(run.path)
    for folder in (base, base / "silver", base.parent, base.parent.parent):
        try:
            if (folder / "segment_reach.csv").is_file():
                return folder
        except OSError:
            continue
    return None


def _read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _f(value, default=float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def render(selection):
    blank = empty_figure("no coverage report")
    if not selection:
        return ui.note(""), blank, blank, blank, None

    folder = _run_dir(selection)
    if folder is None:
        return (ui.warnings_block([_HINT], "No coverage report"),
                blank, blank, blank, None)

    reach = _read(folder / "segment_reach.csv")
    points = _read(folder / "point_rejections.csv")
    if not reach:
        return (ui.warnings_block([_HINT], "No coverage report"),
                blank, blank, blank, None)

    geom = registry.get(selection.get("plate_key") or registry.default_key())

    # --- the map: how high each segment reached ---------------------------
    f_max = {r["segment"]: _f(r["f_max_hz"]) for r in reach}
    n_missing = sum(1 for v in f_max.values() if not np.isfinite(v))
    fig = plate_heatmap(
        geom, f_max, param="coverage_f_max", robust=True,
        title="Highest frequency with a surviving point [Hz] — "
              f"{len(f_max) - n_missing} of {len(f_max)} segments evaluated")

    # --- what removed them -------------------------------------------------
    tally: dict[str, int] = {}
    for p in points:
        if p.get("kept") == "1":
            continue
        tally[p.get("reason", "")] = tally.get(p.get("reason", ""), 0) + 1
    for r in reach:
        if r["blocked_by"] == "no_channel":
            tally["no_channel"] = tally.get("no_channel", 0) + 1
    order = sorted(tally, key=lambda k: -tally[k])
    reasons = go.Figure(go.Bar(
        x=[tally[k] for k in order],
        y=[REASON_LABEL.get(k, k) for k in order],
        orientation="h", marker_color="#c0392b",
        hovertemplate="%{y}<br>%{x} point(s)<extra></extra>"))
    reasons.update_layout(template=TEMPLATE, xaxis_title="points removed",
                          yaxis=dict(autorange="reversed"),
                          margin=dict(l=200, r=20, t=20, b=44))

    # --- and where in frequency -------------------------------------------
    vs_f = go.Figure()
    for k in order:
        fs = [_f(p["freq_hz"]) for p in points
              if p.get("kept") != "1" and p.get("reason") == k]
        if not fs:
            continue
        vs_f.add_trace(go.Box(x=fs, name=REASON_LABEL.get(k, k),
                              boxpoints=False, orientation="h"))
    kept_f = [_f(p["freq_hz"]) for p in points if p.get("kept") == "1"]
    if kept_f:
        vs_f.add_trace(go.Box(x=kept_f, name="kept", boxpoints=False,
                              orientation="h", marker_color="#2e9e5b"))
    vs_f.update_layout(template=TEMPLATE, xaxis_type="log",
                       xaxis_title="f [Hz]", showlegend=False,
                       margin=dict(l=200, r=20, t=20, b=44))

    # --- the table ---------------------------------------------------------
    import pandas as pd
    frame = pd.DataFrame([{
        "segment": r["segment"],
        "points kept": f"{r['n_kept']} / {r['n_points']}",
        "band [Hz]": ("—" if not np.isfinite(_f(r["f_max_hz"]))
                      else f"{_f(r['f_min_hz']):.3g} – {_f(r['f_max_hz']):.4g}"),
        "dropped above": r["n_above_f_max"],
        "limited by": REASON_LABEL.get(r["blocked_by"], r["blocked_by"] or "—"),
        "why": r["explanation"],
    } for r in reach])

    notes = []
    unwired = [r["segment"] for r in reach if r["blocked_by"] == "no_channel"]
    if unwired:
        notes.append(
            f"Segments {', '.join(unwired)} have no ADC channel on any card "
            "file. They were never recorded, so there is nothing to evaluate "
            "— this is a wiring fact, not a measurement result.")
    dropped = [r["segment"] for r in reach
               if r["verdict"].startswith("dropped") ]
    if dropped:
        notes.append(
            f"Segments {', '.join(dropped)} were recorded but had too few "
            "points survive the gates to be modelled. The per-point reasons "
            "are in the table and in point_rejections.csv.")
    status = ui.note(f"{len(reach)} segments · {len(points)} recorded points")
    if notes:
        status = html.Div([status, ui.warnings_block(notes, "Not evaluated")])

    return status, fig, reasons, vs_f, ui.table(frame, "cv-table-inner",
                                                height="320px")


def register(app):

    @app.callback(Output("cv-status", "children"), Output("cv-map", "figure"),
                  Output("cv-reasons", "figure"), Output("cv-vs-freq", "figure"),
                  Output("cv-table", "children"),
                  Input("selection", "data"))
    def _show(selection):
        return render(selection)
