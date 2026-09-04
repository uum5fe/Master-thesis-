"""Segments that do not behave like the ones touching them.

A plate has a real spatial gradient: reactant is consumed along the channel,
so a segment at the outlet legitimately carries a higher R_p than one at the
inlet. Ranking segments against the PLATE mean therefore flags the whole
outlet, which is physics rather than a fault, and hides the segment that is
genuinely wrong because its neighbourhood is already extreme.

This tab compares each segment against the ring of segments that physically
touch it. The gradient cancels -- a segment's neighbours share its position
in it -- and what survives is local. Validated on a synthetic plate carrying
only a gradient and one planted fault: nothing flagged for the gradient, the
planted segment flagged at z = +9.7.

The spectrum panel is the diagnostic half. WHERE in frequency a segment
departs from its ring says what kind of departure it is:

    offset at every frequency   -> contact or lead resistance
    high-frequency only         -> ohmic, membrane hydration
    mid-band arc bigger         -> kinetics, catalyst, poisoning
    low-frequency tail only     -> transport, flooding, starvation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

from app.data.model import PARAM_META, param_label
from app.services import store
from app.services.figures import bode, empty_figure, nyquist
from app.views import common as ui

#: Iglewicz & Hoaglin's published cut for the modified z-score. Offered as a
#: control because "how strange is strange" is the reader's call, not the
#: code's, but the default is the reference value rather than a tuned one.
Z_CHOICES = [2.5, 3.0, 3.5, 4.0, 5.0]


def _pipeline():
    """`neighbours` lives with the pipeline, not the frontend.

    Imported lazily and by hand so the tab degrades to a message instead of
    taking the whole app down when the pipeline folder is not on the path --
    which is the normal state of a viewer-only deployment.
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent.parent / "local_eis"
    if root.is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import neighbours
    return neighbours


def layout():
    return html.Div([
        ui.panel([
            html.Div([
                html.Div(ui.field(
                    "Parameters to test",
                    dcc.Dropdown(id="an-params", options=[], value=[],
                                 multi=True, clearable=False)),
                    style={"flex": "3 1 320px"}),
                html.Div(ui.field(
                    "Flag beyond |z|",
                    dcc.Dropdown(id="an-z",
                                 options=[{"label": f"{z:g}", "value": z}
                                          for z in Z_CHOICES],
                                 value=3.5, clearable=False)),
                    style={"flex": "0 1 140px"}),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
            ui.note("Each segment is compared with the ring of segments that "
                    "share a pad edge with it, using a modified z-score "
                    "(median and MAD). Comparing against the plate mean "
                    "instead would flag the whole outlet, which is the "
                    "gradient rather than a fault."),
        ]),
        ui.panel([html.Div(id="an-summary")]),
        ui.panel([
            ui.section_title("Ranked findings",
                             "strongest departure from the ring first"),
            html.Div(id="an-table"),
        ]),
        ui.panel([
            ui.section_title("Where in the spectrum the difference sits",
                             "the segment against the median of its own ring"),
            html.Div(ui.field(
                "Segment",
                dcc.Dropdown(id="an-seg", options=[], value=None,
                             clearable=False)),
                style={"maxWidth": "260px"}),
            ui.graph("an-dev"),
        ]),
        ui.panel([
            ui.section_title("The segment and its neighbours",
                             "so the verdict can be read, not trusted"),
            ui.graph("an-nyquist"),
            ui.graph("an-bode"),
        ]),
    ])


def _run_of(selection):
    if not selection or not selection.get("measurement_id"):
        return None
    return store.current_run(selection["kind"], selection["measurement_id"],
                             selection["condition"], selection["plate_key"])


def _spectra_of(run, names):
    """{segment: (freq, Z)} in mOhm.cm2, straight from the viewer's table."""
    out = {}
    for name in names:
        sub = run.spectrum(name)
        if sub.empty:
            continue
        f = sub["freq_hz"].to_numpy(float)
        z = (sub["z_re_mohm_cm2"].to_numpy(float)
             + 1j * sub["z_im_mohm_cm2"].to_numpy(float))
        out[str(name)] = (f, z)
    return out


def register(app):

    @app.callback(Output("an-params", "options"), Output("an-params", "value"),
                  Input("selection", "data"), State("an-params", "value"))
    def _params(selection, current):
        run = _run_of(selection)
        if run is None:
            return [], []
        available = run.mappable_params()
        options = [{"label": param_label(p) if p in PARAM_META else p,
                    "value": p} for p in available]
        keep = [p for p in (current or []) if p in available]
        return options, keep or available[:3]

    @app.callback(Output("an-summary", "children"), Output("an-table", "children"),
                  Output("an-seg", "options"), Output("an-seg", "value"),
                  Input("selection", "data"), Input("an-params", "value"),
                  Input("an-z", "value"), State("an-seg", "value"))
    def _findings(selection, params, z_cut, current_seg):
        run = _run_of(selection)
        if run is None or not params:
            return (ui.note("select a run and at least one parameter"),
                    None, [], None)
        try:
            nb = _pipeline()
        except Exception as exc:                         # pragma: no cover
            return (ui.warnings_block(
                [f"the neighbour analysis needs the pipeline modules: {exc}"],
                title="Not available"), None, [], None)

        result = nb.analyse({p: run.value(p) for p in params},
                            z_cut=float(z_cut))
        rows = result["rows"]
        flagged = result["by_segment"]

        rings = [len(v) for v in result["adjacency"].values()]
        stats = ui.stat_row([
            ui.stat("Segments flagged", str(len(flagged)),
                    "warn" if flagged else "text"),
            ui.stat("Findings", str(len(rows))),
            ui.stat("Parameters tested", str(len(params))),
            ui.stat("Ring size", f"{min(rings)}–{max(rings)}"),
        ])
        if not rows:
            body = html.Div([
                stats,
                ui.note(f"No segment departs from its own neighbours by more "
                        f"than |z| = {z_cut:g} in "
                        f"{', '.join(params)}. That is a result: the plate "
                        f"varies smoothly, and what variation there is, is "
                        f"shared with the neighbourhood."),
            ])
            return body, None, [], None

        frame = pd.DataFrame(rows)[
            ["segment", "param", "value", "ring_median", "ring_sigma", "z",
             "direction", "n_ring", "neighbours", "note"]]
        segs = sorted(flagged, key=lambda s: int(s) if s.isdigit() else 0)
        options = [{"label": f"segment {s}  ({', '.join(flagged[s])})",
                    "value": s} for s in segs]
        value = current_seg if current_seg in flagged else segs[0]
        return (stats, ui.table(frame, table_id="an-tbl", max_rows=200),
                options, value)

    @app.callback(Output("an-dev", "figure"),
                  Input("selection", "data"), Input("an-seg", "value"))
    def _deviation(selection, segment):
        run = _run_of(selection)
        if run is None or not segment:
            return empty_figure("pick a flagged segment")
        try:
            nb = _pipeline()
        except Exception as exc:                         # pragma: no cover
            return empty_figure(str(exc))

        adj = nb.adjacency()
        wanted = {str(segment)} | {str(n) for n in adj.get(str(segment), ())}
        spectra = _spectra_of(run, [n for n in run.segment_names
                                    if str(n) in wanted])
        if str(segment) not in spectra:
            return empty_figure(f"segment {segment} has no spectrum in this run")
        found = [s for s in nb.spectrum_outliers(spectra, adj=adj)
                 if s.segment == str(segment)]
        if not found:
            return empty_figure(
                f"segment {segment} shares too few frequencies with its "
                f"neighbours to compare — the cards carry different schedules")
        sf = found[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sf.freq_hz, y=sf.dev_db, mode="lines+markers",
                                 name="|Z| vs ring", line={"width": 2}))
        fig.add_trace(go.Scatter(x=sf.freq_hz, y=sf.dev_phase_deg,
                                 mode="lines+markers", name="phase vs ring",
                                 yaxis="y2", line={"width": 1, "dash": "dot"}))
        fig.add_hline(y=0, line_width=1, line_dash="dash", opacity=0.5)
        f_worst, d_worst = sf.worst()
        fig.update_layout(
            title=(f"Segment {segment} against the median of its "
                   f"{sf.n_ring} neighbours ({', '.join(sf.ring)})<br>"
                   f"<sub>{sf.band_summary()};  worst {d_worst:+.1f} dB at "
                   f"{f_worst:.3g} Hz</sub>"),
            xaxis={"title": "f [Hz]", "type": "log"},
            yaxis={"title": "|Z| deviation [dB]"},
            yaxis2={"title": "phase deviation [deg]", "overlaying": "y",
                    "side": "right"},
            margin={"l": 60, "r": 60, "t": 70, "b": 50}, height=420,
            legend={"orientation": "h", "y": -0.2})
        return fig

    @app.callback(Output("an-nyquist", "figure"), Output("an-bode", "figure"),
                  Input("selection", "data"), Input("an-seg", "value"))
    def _curves(selection, segment):
        run = _run_of(selection)
        if run is None or not segment:
            return empty_figure("pick a segment"), empty_figure("")
        try:
            adj = _pipeline().adjacency()
        except Exception as exc:                         # pragma: no cover
            return empty_figure(str(exc)), empty_figure("")

        ring = sorted(adj.get(str(segment), ()),
                      key=lambda s: int(s) if s.isdigit() else 0)
        curves = []
        for name in [str(segment)] + ring:
            sub = run.spectrum(name)
            if sub.empty:
                continue
            curves.append({
                "name": (f"segment {name}" if name == str(segment)
                         else f"neighbour {name}"),
                "f": sub["freq_hz"].to_numpy(float),
                "z": (sub["z_re_mohm_cm2"].to_numpy(float)
                      + 1j * sub["z_im_mohm_cm2"].to_numpy(float)),
                "width": 3 if name == str(segment) else 1,
                "dash": None if name == str(segment) else "dot",
            })
        if not curves:
            return (empty_figure(f"no spectra for segment {segment} or its ring"),
                    empty_figure(""))
        title = f"Segment {segment} and its {len(curves) - 1} neighbours"
        return (nyquist(curves, title=f"Nyquist — {title}"),
                bode(curves, title=f"Bode — {title}"))
