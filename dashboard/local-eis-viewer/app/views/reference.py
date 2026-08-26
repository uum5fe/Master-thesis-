"""The whole-cell reference: does the local map add up to the real cell?

The 72 segments sit in parallel across one cell voltage, so aggregated they
must reproduce what an ordinary whole-cell EIS of the same cell measured at the
same operating point.  That is the only end-to-end check the method has -- it
tests the calibration, the geometry, the synchronisation and the chain response
at once, against an instrument that shares none of them.

Agreement here does NOT mean the local map is uninformative.  The aggregate is
a harmonic, area-weighted mean, so it is dominated by the LOW-impedance
segments: a flooded segment has high Z, contributes little admittance, and
barely moves the cell curve.  That is the mathematical statement of why an
integral measurement hides local faults, and why the local map is worth having.
What agreement means is that the local map is correctly SCALED.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from app.services import store
from app.settings import SETTINGS
from app.plates import registry
from app.services.figures import SERIES_COLOURS, TEMPLATE, empty_figure
from app.views import common as ui


#: One colour per condition, so the same current keeps its colour in both
#: plots and the reference/local pair reads as one series. Taken from the
#: shared, colour-vision-checked order rather than chosen here.
_PALETTE = SERIES_COLOURS


def _pipeline_on_path() -> None:
    import sys
    from app.services.runner import pipeline_dir
    for candidate in (pipeline_dir(), Path(__file__).resolve().parents[2]):
        text = str(candidate)
        if candidate.is_dir() and text not in sys.path:
            sys.path.insert(0, text)


def layout():
    return html.Div([
        ui.panel([
            ui.section_title("Local EIS, aggregated to the cell, against a whole-cell "
                     "sweep"),
            ui.note(
                "The segments are in parallel across one cell voltage, so "
                "admittances add: Z_cell = A_cell / Σ(A_s / Z_s). Gamry writes "
                "whole-cell ohms; multiplying by the cell area puts both on "
                "the same Ω·cm² axis. Only the band both instruments actually "
                "covered is compared — nothing is extrapolated."),
            html.Div(id="rf-status", style={"marginTop": "8px"}),
        ]),
        ui.panel([
            ui.section_title("Nyquist, per condition"),
            ui.graph("rf-nyquist"),
        ]),
        ui.panel([
            ui.section_title("What the difference looks like"),
            ui.note(
                "A flat magnitude offset is an area or a common shunt error. A "
                "phase error growing with frequency is uncorrected chain "
                "response or acquisition skew. Agreement at high frequency "
                "with drift at low frequency means the operating point moved "
                "between the two recordings."),
            ui.graph("rf-residual"),
        ]),
        ui.panel([
            ui.section_title("Per condition"),
            html.Div(id="rf-table"),
        ]),
    ])


def _comparison_dir(selection) -> Path | None:
    """Where the pipeline wrote `gamry_comparison.csv` for this run.

    `RunRef.path` is the run directory -- the one holding `gold/` and
    `silver/` -- and the pipeline writes the comparison into `cfg.out_dir`,
    which is that same directory. A campaign-level run writes it one or two
    levels up instead, so those are searched too, nearest first.
    """
    run = store.current_catalog().find(
        selection.get("measurement_id", ""), selection.get("condition", ""),
        selection.get("kind", "results"))
    if run is None or not run.path:
        return None
    base = Path(run.path)
    for folder in (base, base.parent, base.parent.parent):
        try:
            if (folder / "gamry_comparison.csv").is_file():
                return folder
        except OSError:
            continue
    return None


def _read_curves(folder: Path) -> dict[str, dict]:
    """The per-frequency comparison, grouped by condition."""
    import csv

    path = folder / "gamry_comparison_curves.csv"
    if not path.is_file():
        return {}
    out: dict[str, dict] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            d = out.setdefault(row["condition"],
                               {"f": [], "zl": [], "zr": [], "dphi": []})
            d["f"].append(float(row["freq_hz"]))
            d["zl"].append(complex(float(row["z_re_local_mohm_cm2"]),
                                   float(row["z_im_local_mohm_cm2"])))
            d["zr"].append(complex(float(row["z_re_ref_mohm_cm2"]),
                                   float(row["z_im_ref_mohm_cm2"])))
            d["dphi"].append(float(row["phase_diff_deg"]))
    return {k: {kk: np.array(vv) for kk, vv in v.items()} for k, v in out.items()}


def _read_summary(folder: Path) -> list[dict]:
    import csv

    path = folder / "gamry_comparison.csv"
    if not path.is_file():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


_HINT = (
    "No whole-cell comparison was written for this run. Re-run the pipeline "
    "with --gamry pointing at the folder of Gamry .DTA sweeps for this cell "
    "(and, if you have it, --bench-log for the MF4, so the operating point at "
    "each sweep is reported next to the result)."
)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
#
# A module-level function rather than a closure inside `register`, so it can
# be called directly by a test. The version of this tab that shipped reached
# for a RunRef field that does not exist; every test passed, because the only
# code that could raise was unreachable from outside the Dash app.

def _live_comparison(selection):
    """Compute the comparison now, from what is already on disk.

    The pipeline writes gamry_comparison.csv only when it was given --gamry.
    A run made without it still has everything needed -- silver wrote
    cell_aggregate.csv, and the sweeps are wherever EIS_GAMRY_ROOT points --
    so there is no reason to make someone re-process gigabytes to see a
    comparison that takes a fraction of a second to compute.

    Returns the same (rows, curves) shape the CSV path produces, so the
    drawing below does not care which way the numbers arrived.
    """
    _pipeline_on_path()
    import gamry_compare as GC

    run = store.current_catalog().find(
        selection.get("measurement_id", ""), selection.get("condition", ""),
        selection.get("kind", "results"))
    if run is None or not run.path:
        return [], {}, ""

    base = Path(run.path)
    aggregate = None
    for candidate in (base / "silver" / "cell_aggregate.csv",
                      base / "cell_aggregate.csv"):
        if candidate.is_file():
            aggregate = candidate
            break
    if aggregate is None:
        return [], {}, ("This run has no cell_aggregate.csv, so there is "
                        "nothing to compare. It is written by silver.")

    roots = [base, base.parent, base.parent.parent]
    roots += SETTINGS.resolved_gamry_roots()
    sweeps = []
    for root in roots:
        try:
            sweeps = GC.find_cell_sweeps(root)
        except OSError:
            continue
        if sweeps:
            break
    if not sweeps:
        return [], {}, ("No whole-cell Gamry .DTA sweeps found. Set "
                        "EIS_GAMRY_ROOT to the folder holding them.")

    condition = selection.get("condition", "")
    match = next((sw for sw in sweeps if sw.condition == condition), None)
    if match is None:
        have = ", ".join(sorted({sw.condition for sw in sweeps})) or "none"
        return [], {}, (f"No sweep at {condition}. The sweeps found cover: "
                        f"{have}.")

    geom = registry.get(selection.get("plate") or registry.default_key())
    area = float(sum(geom.areas().values()))
    freq, Z = GC.read_cell_aggregate(aggregate)
    comparison = GC.compare(freq, Z, match, area)
    if not comparison.n_points:
        return [], {}, ("The local band and the sweep do not overlap, so "
                        "there is nothing to compare.")

    rows = [{k: ("" if v is None else v)
             for k, v in comparison.summary().items()}]
    curves = {comparison.condition: {
        "f": comparison.freq,
        "zl": comparison.Z_local * 1e3,
        "zr": comparison.Z_ref * 1e3,
        "dphi": comparison.phase_diff_deg,
    }}
    return rows, curves, ""


def render(selection):
    blank = empty_figure("nothing to compare")
    if not selection:
        return ui.note(""), blank, blank, None

    folder = _comparison_dir(selection)
    rows = _read_summary(folder) if folder else []
    curves = _read_curves(folder) if folder else {}
    computed_now = False

    if not rows or not curves:
        # Nothing was written for this run. Compute it rather than asking for
        # a re-processing run that would produce the same numbers.
        rows, curves, problem = _live_comparison(selection)
        computed_now = bool(rows)
        if not rows:
            return (ui.warnings_block([problem or _HINT],
                                      "No reference comparison"),
                    blank, blank, None)

    order = [r["condition"] for r in rows]

    from plotly.subplots import make_subplots

    nyq = go.Figure()
    res = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.09,
                        subplot_titles=("magnitude difference",
                                        "phase difference"))
    for i, cond in enumerate(order):
        d = curves.get(cond)
        if d is None:
            continue
        colour = _PALETTE[i % len(_PALETTE)]
        nyq.add_trace(go.Scatter(
            x=d["zr"].real, y=-d["zr"].imag, mode="lines+markers",
            name=f"{cond} · whole cell",
            line=dict(color=colour, dash="dot"), marker=dict(size=4)))
        nyq.add_trace(go.Scatter(
            x=d["zl"].real, y=-d["zl"].imag, mode="lines+markers",
            name=f"{cond} · local", line=dict(color=colour),
            marker=dict(size=5, symbol="square")))
        # Per cent and degrees are different quantities. They share the
        # frequency axis and nothing else, so they get a row each rather than
        # two scales in one frame, where a crossing would read as a
        # relationship between them.
        rel = 100 * (np.abs(d["zl"]) / np.abs(d["zr"]) - 1.0)
        res.add_trace(go.Scatter(x=d["f"], y=rel, mode="lines+markers",
                                 name=cond, legendgroup=cond,
                                 line=dict(color=colour, width=2),
                                 marker=dict(size=5)), row=1, col=1)
        res.add_trace(go.Scatter(x=d["f"], y=d["dphi"], mode="lines+markers",
                                 name=cond, legendgroup=cond, showlegend=False,
                                 line=dict(color=colour, width=2),
                                 marker=dict(size=5, symbol="square")),
                      row=2, col=1)

    nyq.update_layout(
        template=TEMPLATE,
        xaxis_title="Z′ [mΩ·cm²]", yaxis_title="−Z″ [mΩ·cm²]",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        title="dotted = whole cell (Gamry) · solid = local, aggregated")
    res.update_xaxes(type="log", row=1, col=1)
    res.update_xaxes(type="log", title_text="f [Hz]", row=2, col=1)
    res.update_yaxes(title_text="local − ref  [%]", row=1, col=1)
    res.update_yaxes(title_text="local − ref  [°]", row=2, col=1)
    for row in (1, 2):
        res.add_hline(y=0, line=dict(color="#b9bfc7", width=1), row=row, col=1)
    res.update_layout(template=TEMPLATE, height=460,
                      title="How the two instruments differ")
    notes = [f"{r['condition']}: {r['notes']}" for r in rows
             if r.get("notes")]
    source = ("computed now, from this run's cell_aggregate.csv and the "
              "configured Gamry sweeps"
              if computed_now else f"reference folder {folder.name}")
    status = ui.note(f"{len(rows)} condition(s) compared · {source}")
    if notes:
        status = html.Div([status,
                           ui.warnings_block(notes, "Worth knowing")])

    import pandas as pd
    table = pd.DataFrame([{
        "condition": r["condition"],
        "points": r["n_points"],
        "band [Hz]": f"{float(r['f_lo_hz']):.3g} – {float(r['f_hi_hz']):.4g}",
        **_hfr_cells(r),
        "Δ|Z| [%]": _fmt(r["mag_rel_median_pct"], 1),
        "rms [%]": _fmt(r["rms_rel_pct"], 1),
        "max Δφ [°]": _fmt(r["phase_diff_max_deg"], 2),
    } for r in rows])
    return status, nyq, res, ui.table(table, "rf-summary", height="220px")


def register(app):

    @app.callback(Output("rf-status", "children"),
                  Output("rf-nyquist", "figure"),
                  Output("rf-residual", "figure"),
                  Output("rf-table", "children"),
                  Input("selection", "data"))
    def _show(selection):
        return render(selection)

def _hfr_cells(row: dict) -> dict:
    """The three HFR columns, decided together rather than one at a time.

    Some delivered Gamry sweeps stop at 1.2-3.0 kHz, below where the cell
    crosses the real axis, so one side or both has no measured intercept and
    the pipeline also writes an extrapolation of the top of the band.  Showing
    that is better than three dashes, but it has to be marked, hence the tilde.

    The whole row switches together, and that is the point: when only the
    local side is missing, mixing its extrapolation against the reference's
    MEASURED intercept would charge the extrapolation's bias entirely to the
    local side and read as a real disagreement.  The pipeline's ΔHFR fallback
    extrapolates both sides over the same band, so the two values shown beside
    it must be the same two it divided.  Old comparison CSVs have no fit
    column at all, so this falls back to dashes -- hence the .get().
    """
    measured = (_fmt(row.get("hfr_local_mohm_cm2")),
                _fmt(row.get("hfr_ref_mohm_cm2")),
                _fmt(row.get("hfr_rel_pct"), 1))
    if "—" not in measured:
        local, ref, rel = measured
    else:
        def fit(key, digits=2):
            text = _fmt(row.get(key), digits)
            return f"~{text}" if text != "—" else "—"
        local = fit("hfr_local_fit_mohm_cm2")
        ref = fit("hfr_ref_fit_mohm_cm2")
        rel = fit("hfr_fit_rel_pct", 1)
    return {"HFR local": local, "HFR ref": ref, "ΔHFR [%]": rel}


def _fmt(value: str, digits: int = 2) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(v) else f"{v:.{digits}f}"
