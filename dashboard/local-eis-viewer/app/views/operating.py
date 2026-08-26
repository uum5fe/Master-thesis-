"""Temperature, pressure and humidity across the plate.

The bench measures the operating point at the PORTS. The plate is 252 mm long,
and what happens between those ports is most of what makes a local EIS map
interesting: the cathode dries at the inlet and floods at the outlet, and the
impedance follows. This tab puts the two on the same geometry.

The three fields are NOT equally measured, and that difference is shown on the
map rather than left in a docstring -- a modelled field drawn in the same
colours as a measured one invites conclusions the data cannot support.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from dash import Input, Output, dcc, html

from app.plates import registry
from app.services import store
from app.settings import SETTINGS
from app.services.figures import empty_figure, plate_heatmap
from app.views import common as ui

FIELDS = [
    {"label": "Temperature", "value": "temperature"},
    {"label": "Pressure", "value": "pressure"},
    {"label": "Relative humidity", "value": "humidity"},
]

#: How each provenance should read on screen, and how loudly.
_TONE = {
    "measured": ("Measured", "ok"),
    "interpolated": ("Interpolated between measurements", "warn"),
    "modelled": ("Modelled", "warn"),
    "unavailable": ("Not available", "bad"),
}


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
                    "Field",
                    dcc.Dropdown(id="op-field", options=FIELDS,
                                 value="temperature", clearable=False),
                    "Which operating quantity to draw over the plate."),
                    style={"flex": "1 1 240px"}),
                html.Div(ui.field(
                    "Gas inlet",
                    dcc.RadioItems(
                        id="op-inlet",
                        options=[{"label": " left (x = 0)", "value": "min"},
                                 {"label": " right (x = max)", "value": "max"}],
                        value="min",
                        labelStyle={"display": "block", "marginBottom": "2px"}),
                    "Nothing in the drawing or the bench log states which end "
                    "the gas enters. Getting it backwards mirrors every field, "
                    "so it is a choice you make rather than one inferred for "
                    "you."),
                    style={"flex": "1 1 200px"}),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
            html.Div(id="op-status", style={"marginTop": "10px"}),
        ]),
        ui.panel([ui.graph("op-map", height="520px")]),
        ui.panel([
            ui.section_title("Along the flow path"),
            ui.note("Every segment against its position along the gas path. "
                    "The spread at one position is the spread across the "
                    "plate's width."),
            ui.graph("op-profile", height="300px"),
        ]),
        ui.panel([
            ui.section_title("The operating point these came from"),
            html.Div(id="op-ports"),
        ]),
    ])


def _search_roots(run) -> list[Path]:
    """Where to look for the bench log and the reference sweeps, in order.

    Beside the run first, because that is the common case, and then the
    CONFIGURED Gamry root. That second half is not optional: the sweeps and
    the MF4 usually live on a different branch of the share from the results
    -- .../Lokale_EIS/EIS_Daten_Gamry_Tom against
    .../Lokale_EIS/Daten/EIS_Results/<order>/<condition> -- and no amount of
    walking up from the run reaches them. Looking only near the run is why
    this tab reported "no bench log" while EIS_GAMRY_ROOT was set correctly.
    """
    roots: list[Path] = []
    if run is not None and run.path:
        base = Path(run.path)
        roots += [base, base.parent, base.parent.parent]
    roots += SETTINGS.resolved_gamry_roots()

    seen: set[str] = set()
    out: list[Path] = []
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


def _bench_state(selection):
    """The bench log reading at this run's condition, if there is one."""
    _pipeline_on_path()
    import gamry_compare as GC

    run = store.current_catalog().find(
        selection.get("measurement_id", ""), selection.get("condition", ""),
        selection.get("kind", "results"))
    if run is None or not run.path:
        return None, None, "no run selected"

    roots = _search_roots(run)
    log_path = None
    for folder in roots:
        try:
            log_path = GC.find_bench_log(folder)
        except OSError:
            continue
        if log_path is not None:
            break
    if log_path is None:
        looked = "\n".join(f"      {r}" for r in roots)
        return None, None, (
            "No bench log (.mf4) found. Set EIS_GAMRY_ROOT to the folder "
            "holding the MF4 recorded with this campaign, or put it beside "
            "the results. Looked in:\n" + looked)

    try:
        bench = GC.read_bench_log(log_path)
    except ImportError:
        # Name the interpreter. This machine has a Store placeholder, a Python
        # Install Manager shim and possibly a .venv on PATH; "pip install
        # asammdf" is ambiguous between them and installing into the wrong one
        # leaves this message on screen unchanged.
        import sys
        return None, None, (
            "Reading the bench log needs the asammdf package. Install it into "
            "the interpreter running this app:\n\n"
            f'    "{sys.executable}" -m pip install asammdf\n\n'
            "Everything else works without it; only the operating fields need "
            f"it. Found the log at {log_path}.")
    except Exception as exc:                                # noqa: BLE001
        return None, None, f"bench log unreadable: {exc}"

    sweeps = []
    for folder in roots:                      # the same ordered list
        try:
            sweeps = GC.find_cell_sweeps(folder)
        except OSError:
            continue
        if sweeps:
            break
    cond = selection.get("condition", "")
    match = next((s for s in sweeps if s.condition == cond and s.started), None)
    if match is None:
        match = next((s for s in sweeps if s.started), None)
    if match is None:
        return None, log_path, (
            "The bench log was found but nothing says WHEN this condition was "
            "recorded. A Gamry .dta beside it supplies that timestamp.")
    return bench.state_at(match.started), log_path, ""


def fields_for(selection, inlet: str = "min"):
    """The three per-segment fields for the selected run."""
    _pipeline_on_path()
    import plate_conditions as PC

    state, log_path, problem = _bench_state(selection)
    if state is None:
        return None, None, problem

    geom = registry.get(selection.get("plate") or registry.default_key())
    run = None
    try:
        run = store.current_run(selection.get("kind", "results"),
                                selection.get("measurement_id", ""),
                                selection.get("condition", ""), geom.key)
    except Exception:                                       # noqa: BLE001
        run = None

    plate_t = {k: v for k, v in state.items() if k.lower().startswith("temp")}

    # The measured per-segment current density, when the run has one. Using it
    # is the difference between a humidity field informed by the current
    # distribution that was actually measured and one that assumes the very
    # uniformity a local EIS exists to disprove.
    j_dc = None
    frame = getattr(run, "segments", None) if run is not None else None
    if frame is not None and "j_dc" in getattr(frame, "columns", []):
        j_dc = {str(r.segment): float(r.j_dc) for r in frame.itertuples()
                if np.isfinite(getattr(r, "j_dc", np.nan))}
        j_dc = j_dc or None

    ports = PC.port_state_from_bench(state, plate_t)
    fields = PC.condition_fields(
        geom.centroids(), geom.areas(), ports,
        geom.temp_sensor_x_mm, j_dc=j_dc,
        axis=geom.flow_axis, inlet=inlet)
    return fields, (state, log_path), ""


def _provenance_block(field):
    key = field.provenance.split(" ")[0]
    title, tone = _TONE.get(key, ("", "warn"))
    lines = [field.provenance]
    lines += list(field.notes)
    if tone == "ok":
        return ui.note(" · ".join(lines))
    return ui.warnings_block(lines, title)


def register(app):

    @app.callback(Output("op-map", "figure"), Output("op-profile", "figure"),
                  Output("op-status", "children"), Output("op-ports", "children"),
                  Input("selection", "data"), Input("op-field", "value"),
                  Input("op-inlet", "value"))
    def _draw(selection, which, inlet):
        blank = empty_figure("nothing to draw")
        if not selection:
            return blank, blank, ui.note(""), None
        return render(selection, which or "temperature", inlet or "min")


def render(selection, which="temperature", inlet="min"):
    import plotly.graph_objects as go

    blank = empty_figure("nothing to draw")
    fields, meta, problem = fields_for(selection, inlet)
    if fields is None:
        return blank, blank, ui.warnings_block([problem], "No operating data"), None

    field = fields[which]
    geom = registry.get(selection.get("plate") or registry.default_key())

    if field.provenance == "unavailable":
        return (empty_figure(f"{field.name} is not available for this run"),
                blank, _provenance_block(field), _ports_table(meta))

    label = {"temperature": "T", "pressure": "p", "humidity": "RH"}[which]
    fig = plate_heatmap(
        geom, field.values, param=f"op_{which}", robust=True,
        title=f"{field.name.capitalize()} across the plate "
              f"[{field.unit}] — {field.provenance}")

    # profile along the flow path
    i = 0 if geom.flow_axis == "x" else 1
    cent = geom.centroids()
    pos = {k: c[i] for k, c in cent.items()}
    lo, hi = min(pos.values()), max(pos.values())
    xs = [(pos[k] - lo) if inlet == "min" else (hi - pos[k]) for k in field.values]
    ys = [field.values[k] for k in field.values]
    # Same ramp as the map above, so a colour means the same thing in both
    # panels and the eye can carry a segment from one to the other. A
    # different scale for the same quantity would break that for decoration.
    from app.data.model import PARAM_META
    scale = PARAM_META.get(f"op_{which}", {}).get("colorscale", "Viridis")
    prof = go.Figure(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=8, color=ys, colorscale=scale,
                    line=dict(width=.6, color="rgba(255,255,255,.9)")),
        text=[f"segment {k}" for k in field.values],
        hovertemplate="%{text}<br>%{x:.0f} mm from inlet<br>"
                      "%{y:.4g} " + field.unit + "<extra></extra>"))
    if which == "humidity":
        prof.add_hline(y=100, line=dict(color="#c0392b", width=1, dash="dot"),
                       annotation_text="saturation", annotation_position="top left")
    prof.update_layout(
        xaxis_title="distance from the gas inlet [mm]",
        yaxis_title=f"{label} [{field.unit}]",
        margin=dict(l=60, r=20, t=30, b=50), showlegend=False)

    return fig, prof, _provenance_block(field), _ports_table(meta)


def _ports_table(meta):
    if not meta:
        return None
    import pandas as pd
    _pipeline_on_path()
    import gamry_compare as GC

    state, log_path = meta
    rows = [{"channel": ch, "quantity": GC.BENCH_CHANNELS[ch],
             "value": ("—" if not np.isfinite(state[ch]) else f"{state[ch]:.4g}")}
            for ch in GC.BENCH_CHANNELS if ch in state]
    return html.Div([
        ui.note(f"read from {Path(log_path).name} at t = "
                f"{state.get('t_rel_s', float('nan')):.0f} s into the recording"),
        ui.table(pd.DataFrame(rows), "op-ports-table", height="260px"),
    ])
