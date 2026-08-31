"""Two operating points of the same plate, side by side along the gas paths.

WHAT THIS TAB IS FOR
--------------------
At 45 A the product water is a small perturbation on the gas that was fed in.
At 450 A it is an order of magnitude larger, and the cathode goes from
"humidified gas carrying a little product water" to "gas carrying more water
than it can hold".  That transition is local: it starts at the OXYGEN OUTLET
and works back up the channel.

WHY THE GAS PATH AND NOT x
--------------------------
The four ports are at the corners --

    top-left  O2 out                        H2 out  top-right
    bottom-left  H2 in                       O2 in  bottom-right

-- so hydrogen crosses the plate bottom-left to top-right and oxygen crosses
it bottom-right to top-left.  The two run in opposite directions.  A profile
against x is therefore correct for one gas and mirrored for the other, and
the corner where flooding begins ends up drawn at the dry end.

WHAT TO READ
------------
`R_mt` is the quantity that should move.  Liquid water in the pores blocks
the gas and lifts the low-frequency arc; R_ohmic moves much less and often
the other way, because a wetter membrane conducts better.  So flooding is a
LARGE R_mt ratio CONCENTRATED at the oxygen outlet.  A uniform rise across
the whole plate is not flooding -- it is the cell being pushed further up its
polarisation curve, which happens to every segment at once.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from dash import Input, Output, dcc, html

from app.plates import registry
from app.services import store
from app.services.figures import empty_figure, plate_heatmap
from app.views import common as ui

#: Quantities worth comparing, and how each reads best.  A resistance as a
#: ratio, because the statement is "twice as much"; a temperature as a
#: difference, because the statement is "eight degrees hotter".
QUANTITIES = [
    {"label": "R_mt — mass transport", "value": "R_mt"},
    {"label": "R_ohmic", "value": "R_ohmic"},
    {"label": "R_ct — charge transfer", "value": "R_ct"},
    {"label": "R_pol — polarisation", "value": "R_pol"},
    {"label": "j_dc — current density", "value": "j_dc"},
]


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
            html.Div([
                html.Div(ui.field(
                    "Compare against",
                    dcc.Dropdown(id="cmp-other", options=[], value=None,
                                 clearable=False),
                    "The other condition of the same measurement. The run "
                    "chosen in the sidebar is the reference."),
                    style={"flex": "1 1 220px"}),
                html.Div(ui.field(
                    "Quantity",
                    dcc.Dropdown(id="cmp-param", options=QUANTITIES,
                                 value="R_mt", clearable=False),
                    "R_mt is where liquid water shows up: it blocks the gas "
                    "and lifts the low-frequency arc."),
                    style={"flex": "1 1 220px"}),
                html.Div(ui.field(
                    "Along which gas",
                    dcc.RadioItems(
                        id="cmp-gas",
                        options=[{"label": " oxygen (cathode)", "value": "O2"},
                                 {"label": " hydrogen (anode)", "value": "H2"}],
                        value="O2",
                        labelStyle={"display": "block", "marginBottom": "2px"}),
                    "Water is produced at the cathode, so the oxygen path is "
                    "the one the water balance runs along."),
                    style={"flex": "1 1 200px"}),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
            html.Div(id="cmp-status", style={"marginTop": "10px"}),
        ]),
        ui.panel([
            ui.section_title("Ratio across the plate"),
            ui.graph("cmp-map", height="480px"),
        ]),
        ui.panel([
            ui.section_title("Along the gas path"),
            ui.note("Each segment against its position along the chosen "
                    "gas path, 0 at the inlet corner and 1 at the outlet."),
            ui.graph("cmp-profile", height="320px"),
        ]),
        ui.panel([
            ui.section_title("Spectra, the two conditions overlaid"),
            ui.note("The plate median at each frequency, so the shape of the "
                    "change is visible without 72 curves."),
            ui.graph("cmp-spectra", height="360px"),
        ]),
    ])


def _load(selection, condition):
    geom_key = selection.get("plate_key") or registry.default_key()
    return store.current_run(selection.get("kind", "results"),
                             selection.get("measurement_id", ""),
                             condition, geom_key)


def register(app):

    @app.callback(Output("cmp-other", "options"), Output("cmp-other", "value"),
                  Input("selection", "data"))
    def _options(selection):
        if not selection:
            return [], None
        catalog = store.current_catalog()
        here = selection.get("condition", "")
        others = [c for c in catalog.conditions(
            selection.get("measurement_id", ""), selection.get("kind", "")) 
            if c != here]
        # Prefer the largest current available: that is the interesting
        # comparison, and the one this tab exists for.
        def _amps(name):
            digits = "".join(ch for ch in name if ch.isdigit())
            return int(digits) if digits else -1
        default = max(others, key=_amps) if others else None
        return [{"label": c, "value": c} for c in others], default

    @app.callback(Output("cmp-map", "figure"), Output("cmp-profile", "figure"),
                  Output("cmp-spectra", "figure"), Output("cmp-status", "children"),
                  Input("selection", "data"), Input("cmp-other", "value"),
                  Input("cmp-param", "value"), Input("cmp-gas", "value"))
    def _draw(selection, other, param, gas):
        blank = empty_figure("nothing to draw")
        if not selection or not other:
            return (blank, blank, blank,
                    ui.note("Pick a second condition to compare against."))
        return render(selection, other, param or "R_mt", gas or "O2")


def render(selection, other, param="R_mt", gas="O2"):
    import plotly.graph_objects as go

    blank = empty_figure("nothing to draw")
    _pipeline_on_path()
    import plate_conditions as PC

    geom = registry.get(selection.get("plate_key") or registry.default_key())
    here = selection.get("condition", "")
    try:
        run_a = _load(selection, here)
        run_b = _load(selection, other)
    except Exception as exc:                                # noqa: BLE001
        return blank, blank, blank, ui.warnings_block(
            [f"could not load both conditions: {exc}"], "Nothing to compare")

    a, b = run_a.value(param), run_b.value(param)
    shared = sorted(set(a) & set(b), key=lambda s: int(s) if s.isdigit() else 0)
    if not shared:
        return blank, blank, blank, ui.warnings_block(
            [f"{here} and {other} share no segment carrying {param}"],
            "Nothing to compare")

    ratio = {s: (b[s] / a[s]) for s in shared
             if np.isfinite(a[s]) and np.isfinite(b[s]) and a[s] != 0}
    if not ratio:
        return blank, blank, blank, ui.warnings_block(
            [f"{param} is zero or missing on every shared segment"],
            "Nothing to compare")

    fig = plate_heatmap(
        geom, ratio, param=f"cmp_{param}", robust=True,
        title=f"{param}: {other} / {here}  —  ratio over {len(ratio)} segments")

    # ---- profile along the gas path ------------------------------------
    cent = geom.centroids()
    xi = PC.flow_coordinate(cent, gas)
    inlet_corner, outlet_corner = PC.gas_path(gas=gas)
    xs = [xi[s] for s in ratio]
    ys = [ratio[s] for s in ratio]
    prof = go.Figure(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=8, color=ys, colorscale="RdBu_r",
                    line=dict(width=.6, color="rgba(255,255,255,.9)")),
        text=[f"segment {s}" for s in ratio],
        hovertemplate="%{text}<br>%{x:.2f} along the " + gas
                      + " path<br>ratio %{y:.2f}<extra></extra>"))
    prof.add_hline(y=1.0, line=dict(color="#888", width=1, dash="dot"),
                   annotation_text="no change", annotation_position="top left")
    prof.update_layout(
        xaxis_title=(f"along the {gas} path — 0 at the {inlet_corner} inlet, "
                     f"1 at the {outlet_corner} outlet"),
        yaxis_title=f"{param}  {other} / {here}",
        margin=dict(l=60, r=20, t=30, b=50), showlegend=False)

    # ---- spectra, plate median -----------------------------------------
    spec = go.Figure()
    for run, name, colour in ((run_a, here, "#2c7fb8"),
                              (run_b, other, "#d95f02")):
        frame = run.spectra
        if frame is None or frame.empty:
            continue
        grouped = frame.groupby("freq_hz")
        z_re = grouped["z_real"].median() if "z_real" in frame else None
        z_im = grouped["z_imag"].median() if "z_imag" in frame else None
        if z_re is None or z_im is None:
            continue
        spec.add_trace(go.Scatter(
            x=z_re.values, y=-z_im.values, mode="lines+markers", name=name,
            line=dict(color=colour, width=2), marker=dict(size=5),
            hovertemplate=(name + "<br>Z' %{x:.4g}<br>-Z'' %{y:.4g}"
                           "<extra></extra>")))
    spec.update_layout(
        xaxis_title="Z' [mΩ·cm²]", yaxis_title="-Z'' [mΩ·cm²]",
        margin=dict(l=60, r=20, t=30, b=50),
        yaxis=dict(scaleanchor="x", scaleratio=1))

    # ---- the verdict ----------------------------------------------------
    xs_arr, ys_arr = np.array(xs), np.array(ys)
    inlet_third = ys_arr[xs_arr <= 0.33]
    outlet_third = ys_arr[xs_arr >= 0.67]
    lines = [f"{len(ratio)} shared segments; {param} compared as "
             f"{other} / {here}."]
    if inlet_third.size >= 2 and outlet_third.size >= 2:
        med_in = float(np.median(inlet_third))
        med_out = float(np.median(outlet_third))
        rank = float(np.corrcoef(np.argsort(np.argsort(xs_arr)),
                                 np.argsort(np.argsort(ys_arr)))[0, 1])
        lines.append(
            f"inlet third {med_in:.2f}×, outlet third {med_out:.2f}× "
            f"(rank correlation with the {gas} path {rank:+.2f}).")
        if param == "R_mt":
            if med_in > 0 and med_out / med_in > 1.2 and rank > 0.3:
                lines.append(
                    "Concentrated at the outlet — that is what liquid water "
                    "looks like: it accumulates where the gas has already "
                    "collected the most vapour.")
            else:
                lines.append(
                    "A roughly uniform rise. Every resistance grows with "
                    "current, so this on its own is the whole cell working "
                    "harder rather than local flooding.")
    return fig, prof, spec, ui.note(" ".join(lines))
