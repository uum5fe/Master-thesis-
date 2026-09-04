"""The recording itself: dwells, extracted tones, and the timing between channels.

Every other tab shows a result. This one shows the measurement, because the
three things most likely to be wrong about a local-EIS number are invisible in
a Nyquist plot:

* **Was the dwell found in the right place?** The pipeline locates each
  frequency step from a demodulated envelope and then shrinks the window while
  the SNR improves. If that window straddles two steps, the fit sees a mixture
  and reports a residual that has nothing to do with the noise.

* **Is the tone actually there?** A step whose segment response never rose
  above the noise still produces a phasor, and the phasor still produces an
  impedance. The amplitude and SNR of the extracted tone say whether that
  number means anything.

* **Are the two channels simultaneous?** The impedance is a RATIO of the
  segment channel to the cell voltage, so any delay between them lands
  directly in the phase: `φ = 2π·f·τ`. At 4.5 kHz a single 100 µs sample is
  137°. This tab measures that delay and shows it before and after the
  correction the pipeline applies.

The plots are drawn from the pipeline's own readers and estimators — the same
`detect_schedule`, the same `fit3` — so what is on screen is what the
evaluation did, not a re-implementation that might agree by luck.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

from app.services import store
from app.services.figures import TEMPLATE, empty_figure
from app.views import common as ui

#: Reading a whole card to draw an overview is wasteful and slow — a 2.5 M
#: sample channel is 20 MB and the envelope of every tenth sample looks the
#: same on a 900 px plot. The zoomed panels always use full resolution.
OVERVIEW_STRIDE = 25

MAX_DWELLS = 60


def layout():
    return html.Div([
        ui.panel([
            html.Div([
                html.Div(ui.field("Segment",
                                  dcc.Dropdown(id="sg-segment", options=[],
                                               value=None, clearable=False,
                                               placeholder="pick a segment"),
                                  "Only segments actually wired to a card "
                                  "appear here."),
                         style={"flex": "1 1 200px"}),
                html.Div(ui.field("Frequency step",
                                  dcc.Dropdown(id="sg-step", options=[], value=None,
                                               clearable=False,
                                               placeholder="pick a dwell"),
                                  "The dwells the pipeline found in this "
                                  "recording, not a list it was given."),
                         style={"flex": "1 1 220px"}),
                html.Div(ui.field("Cycles to draw",
                                  dcc.Slider(id="sg-cycles", min=2, max=20, step=1,
                                             value=6,
                                             marks={2: "2", 6: "6", 12: "12", 20: "20"}),
                                  "How much of the dwell the zoomed plots show."),
                         style={"flex": "2 1 300px"}),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
            html.Div(id="sg-status", style={"marginTop": "8px"}),
            dcc.Store(id="sg-schedule"),
        ]),

        ui.panel([
            ui.section_title("1 · The recording, and where the dwells are"),
            ui.note("Envelope of the cell-voltage reference over the whole "
                    "record. Each shaded band is one frequency step as the "
                    "pipeline located it — grown from the demodulated envelope, "
                    "then trimmed while the fit residual kept improving. Gaps "
                    "between bands are settling time, not lost data."),
            ui.graph("sg-overview"),
        ]),

        html.Div([
            html.Div([ui.panel([
                ui.section_title("2 · One dwell, and the tone taken out of it"),
                ui.note("Raw samples with the fitted sine on top. The fit is a "
                        "least-squares sine at the known frequency — the "
                        "optimum estimator, and the same one the pipeline uses. "
                        "If the samples wander away from the curve, the dwell "
                        "window is wrong or the step was not stationary."),
                ui.graph("sg-dwell"),
            ])], style={"flex": "1 1 480px", "minWidth": "420px"}),
            html.Div([ui.panel([
                ui.section_title("3 · Are the two channels simultaneous?"),
                ui.note("Both channels normalised to unit amplitude and drawn "
                        "over a few cycles. Any horizontal shift between them "
                        "is acquisition skew, and it goes straight into the "
                        "impedance phase. The dashed curve is the segment after "
                        "the pipeline's de-skew."),
                ui.graph("sg-sync"),
            ])], style={"flex": "1 1 480px", "minWidth": "420px"}),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),

        ui.panel([
            ui.section_title("4 · Every tone extracted from this recording"),
            ui.note("One row per dwell. `cycles` under about three means the "
                    "phasor came from a fraction of a period and should not be "
                    "trusted whatever its SNR; `SNR` is the tone against the "
                    "residual after it is removed."),
            html.Div(id="sg-tones"),
        ]),
    ])


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def _pipeline_on_path() -> None:
    """Put the bundled pipeline directory on ``sys.path``.

    Its modules import each other flatly -- ``import csv_source`` -- because
    that is how they run on Databricks, where the notebook drops one folder on
    the path. Importing them as ``local_eis.x`` therefore half-works: the top
    module resolves and its first internal import does not. Adding the folder
    itself is what makes the same files run unmodified in both places, which is
    the point of bundling them rather than forking them.
    """
    import sys
    from pathlib import Path
    from app.services.runner import pipeline_dir
    for candidate in (pipeline_dir(), Path(__file__).resolve().parents[2]):
        text = str(candidate)
        if candidate.is_dir() and text not in sys.path:
            sys.path.insert(0, text)


def _famos_files(selection) -> list[str]:
    catalog = store.current_catalog()
    ref = catalog.find(selection.get("measurement_id", ""),
                       selection.get("condition", ""), "famos")
    return list(ref.files) if ref else []


def _open_card(path: str):
    from pathlib import Path
    _pipeline_on_path()
    from eis_local import FamosFile
    return FamosFile(Path(path))


def _card_for_segment(files: list[str], segment: str):
    """The card that carries this segment, plus its reference channel.

    A segment lives on exactly one card, and the cell voltage it must be
    compared against is the copy recorded ON THAT CARD -- pairing across cards
    would put the whole inter-card offset into the ratio.
    """
    for path in files:
        try:
            fam = _open_card(path)
        except Exception:                             # noqa: BLE001
            continue
        if segment in fam.names and fam.uc_names:
            return fam, fam.uc_names[0]
    return None, ""


def _segment_index(files: list[str]) -> dict[str, str]:
    """segment name -> the card file that holds it."""
    out: dict[str, str] = {}
    for path in files:
        try:
            fam = _open_card(path)
        except Exception:                             # noqa: BLE001
            continue
        for name in fam.segment_names:
            out.setdefault(name, path)
    return out


def _csv_ref(selection):
    catalog = store.current_catalog()
    return catalog.find(selection.get("measurement_id", ""),
                        selection.get("condition", ""), "csvlog")


def _csv_points(ref):
    """Read a sweep folder, and locate the burst and the tone in every file.

    A CSV point file is one frequency, so what plays the part of a "dwell" here
    is the burst inside that file -- the delivered ones carry a quarter of a
    second of lead-in and four tenths of lead-out with no excitation at all.
    The same function the evaluation uses does the locating, so this shows what
    the pipeline did rather than an approximation of it.
    """
    from pathlib import Path
    _pipeline_on_path()
    import csv_source as C
    import csv_pipeline as P
    from config import DEFAULT

    class _Quiet:
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def debug(self, *a, **k): pass

    cfg = DEFAULT.replace(f_min_hz=0.5, f_max_hz=1e6)
    out = []
    for path in ref.files:
        try:
            m = C.read_r2d2(path)
            res = P.r2d2_point(m, cfg, _Quiet())
        except Exception as exc:                      # noqa: BLE001
            out.append({"path": path, "ok": False, "reason": str(exc)})
            continue
        if not res.get("ok"):
            out.append({"path": path, "ok": False,
                        "reason": res.get("reason", "no excitation")})
            continue
        z = res["zone"]
        out.append({
            "path": path, "ok": True,
            "name": Path(path).stem,
            "fs": res["fs_hz"], "n_rows": res["n_rows"],
            "burst": res["burst"],
            "f_alias": res["f_alias_hz"], "f_true": res["f_hz"],
            "zone": z["zone"], "undersampled": bool(z["undersampled"]),
            "R": z["R"], "R_base": z["R_baseband"],
            "conjugated": bool(res["conjugated"]),
            "uc_spread": res["uc_pair_spread"],
        })
    return out


def _schedule(fam, ref_name: str) -> list[dict]:
    """The dwells of the sweep, filtered the way the evaluation filters them.

    `detect_schedule` returns candidates, and the abrupt end of a dwell splatters
    enough energy to raise a few of them out of nothing.  Bronze keeps only the
    steps that pass `Step.valid(min_snr_db)` and evaluates those; offering the
    rejects here would put dwells in the dropdown that no impedance was ever
    computed from, which is exactly what this tab exists not to do.
    """
    _pipeline_on_path()
    from eis_local import detect_schedule
    from config import DEFAULT
    min_snr = float(getattr(DEFAULT, "min_snr_db", 8.0))
    ref = fam.channel(ref_name)
    steps = detect_schedule(ref, fam.fs, min_snr_db=min_snr, verbose=False)
    steps = [s for s in steps if s.valid(min_snr)]
    steps.sort(key=lambda s: s.start)
    return [{"freq": float(s.freq), "start": int(s.start), "stop": int(s.stop),
             "amp": float(s.amp), "snr_db": float(s.snr_db)}
            for s in steps][:MAX_DWELLS]


# ---------------------------------------------------------------------------
# callbacks
# ---------------------------------------------------------------------------


def register(app):

    @app.callback(Output("sg-segment", "options"), Output("sg-segment", "value"),
                  Output("sg-status", "children"),
                  Input("selection", "data"), State("sg-segment", "value"))
    def _segments(selection, current):
        if not selection:
            return [], None, ui.note("")
        if selection.get("kind") == "csvlog":
            ref = _csv_ref(selection)
            if ref is None or not ref.files:
                return [], None, ui.note("no point files for this sweep")
            _pipeline_on_path()
            import csv_source as C
            try:
                m = C.read_r2d2(ref.files[0])
            except Exception as exc:                  # noqa: BLE001
                return [], None, ui.note(f"unreadable: {exc}")
            names = m.segments
            options = [{"label": f"segment {n}", "value": n} for n in names]
            value = current if current in names else (names[0] if names else None)
            meta = ref.detail.get("metadata", {})
            note = ui.note(
                f"{len(ref.files)} frequency point(s) · {len(names)} segments · "
                f"{m.summary()['n_channels']} channels scanned over "
                f"{m.summary()['scan_span_us']} µs "
                f"({100*m.summary()['scan_fraction_of_sample']:.0f} % of a sample) · "
                f"coefficients {meta.get('coefficients', '?')}")
            return options, value, note
        if selection.get("kind") != "famos":
            return [], None, ui.warnings_block(
                ["This tab reads a raw recording. Set 'File format' to "
                 "'Raw recording — FAMOS .DAT' or 'Raw sweep — R2-D2 CSV "
                 "logger folder'; a finished pipeline result no longer "
                 "contains the samples."],
                "Nothing to draw")
        files = _famos_files(selection)
        if not files:
            return [], None, ui.note("no card files for this selection")
        index = _segment_index(files)
        names = sorted(index, key=lambda n: int(n) if n.isdigit() else 1 << 30)
        options = [{"label": f"segment {n}", "value": n} for n in names]
        value = current if current in index else (names[0] if names else None)
        note = ui.note(f"{len(files)} card file(s), {len(names)} wired segments")
        return options, value, note

    @app.callback(Output("sg-schedule", "data"),
                  Output("sg-step", "options"), Output("sg-step", "value"),
                  Input("selection", "data"), Input("sg-segment", "value"),
                  State("sg-step", "value"))
    def _steps(selection, segment, current):
        if not selection or not segment:
            return None, [], None
        if selection.get("kind") == "csvlog":
            ref = _csv_ref(selection)
            if ref is None:
                return None, [], None
            points = _csv_points(ref)
            data = {"source": "csvlog", "points": points}
            options = []
            for i, p in enumerate(points):
                if p.get("ok"):
                    tag = " ⚠ alias" if p["undersampled"] else ""
                    options.append({"label": f"{p['f_true']:.4g} Hz{tag}",
                                    "value": i})
                else:
                    options.append({"label": f"{p['path'].split('/')[-1]} — "
                                             f"no excitation", "value": i})
            usable = [o["value"] for o, p in zip(options, points) if p.get("ok")]
            value = current if current in usable else (usable[0] if usable else None)
            return data, options, value
        if selection.get("kind") != "famos":
            return None, [], None
        files = _famos_files(selection)
        fam, ref_name = _card_for_segment(files, segment)
        if fam is None:
            return None, [], None
        try:
            steps = _schedule(fam, ref_name)
        except Exception:                             # noqa: BLE001
            return None, [], None
        data = {"path": str(fam.path), "ref": ref_name, "fs": fam.fs,
                "n": int(fam.n_samples), "steps": steps}
        options = [{"label": f"{s['freq']:.3g} Hz", "value": i}
                   for i, s in enumerate(steps)]
        value = current if isinstance(current, int) and current < len(steps) else (
            len(steps) // 2 if steps else None)
        return data, options, value

    @app.callback(Output("sg-overview", "figure"),
                  Input("sg-schedule", "data"), Input("sg-step", "value"))
    def _overview(data, step_index):
        if data and data.get("source") == "csvlog":
            return _csv_overview(data, step_index)
        if not data or not data.get("steps"):
            return empty_figure("pick a raw recording and a segment")
        fam = _open_card(data["path"])
        ref = fam.channel(data["ref"])[::OVERVIEW_STRIDE]
        t = np.arange(ref.size) * OVERVIEW_STRIDE / data["fs"]

        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=t, y=ref, mode="lines", name=data["ref"],
            line=dict(width=0.7, color="#1f6feb"),
            hovertemplate="t = %{x:.2f} s<br>%{y:.4f} V<extra></extra>"))

        for i, s in enumerate(data["steps"]):
            selected = (i == step_index)
            # annotation_text=None does NOT mean "no annotation": Plotly then
            # draws its placeholder, which is where the "new text" labels came
            # from. The annotation has to be omitted entirely instead -- and
            # every dwell gets its frequency, not just the selected one, since
            # reading off which frequencies were found is the whole job of
            # this plot.
            fig.add_vrect(
                x0=s["start"] / data["fs"], x1=s["stop"] / data["fs"],
                fillcolor="#c0392b" if selected else "#5b6470",
                opacity=0.30 if selected else 0.10, line_width=0,
                annotation=dict(
                    text=f"{s['freq']:.4g}",
                    font=dict(size=10 if selected else 9,
                              color="#c0392b" if selected else "#5b6470"),
                    textangle=-90, yanchor="top"),
                annotation_position="top left")

        fig.update_layout(
            template=TEMPLATE, height=260,
            margin=dict(l=54, r=16, t=28, b=40),
            xaxis_title="time [s]", yaxis_title=f"{data['ref']} [V]",
            title=dict(text=f"{len(data['steps'])} dwells found · "
                            f"{data['n']/data['fs']:.1f} s at "
                            f"{data['fs']:.0f} Hz", font=dict(size=12)),
            showlegend=False)
        return fig

    @app.callback(Output("sg-dwell", "figure"), Output("sg-sync", "figure"),
                  Output("sg-tones", "children"),
                  Input("sg-schedule", "data"), Input("sg-step", "value"),
                  Input("sg-segment", "value"), Input("sg-cycles", "value"))
    def _dwell(data, step_index, segment, n_cycles):
        if not data or step_index is None or not segment:
            blank = empty_figure("pick a dwell")
            return blank, empty_figure(""), ui.note("")
        if data.get("source") == "csvlog":
            return _csv_dwell(data, step_index, segment, n_cycles)

        _pipeline_on_path()
        from utils import fit3

        fam = _open_card(data["path"])
        fs = data["fs"]
        steps = data["steps"]
        s = steps[int(step_index)]
        f0, a, b = s["freq"], s["start"], s["stop"]

        ref = fam.channel(data["ref"])[a:b]
        seg = fam.channel(segment)[a:b]
        t = np.arange(ref.size) / fs

        A_ref, r_ref, snr_ref = fit3(ref, fs, f0)
        A_seg, r_seg, snr_seg = fit3(seg, fs, f0)

        # ---- panel 2: the dwell and its fitted sine ------------------------
        n_show = int(min(ref.size, max(1, n_cycles or 6) * fs / max(f0, 1e-9)))
        n_show = max(n_show, 32)
        ts = t[:n_show]
        w = 2 * np.pi * f0

        def model(A, y):
            return np.real(A * np.exp(1j * w * ts)) + float(np.mean(y))

        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=ts, y=ref[:n_show], mode="markers",
                                   name=f"{data['ref']} samples",
                                   marker=dict(size=3, color="#1f6feb")))
        fig.add_trace(go.Scatter(x=ts, y=model(A_ref, ref), mode="lines",
                                 name="fitted sine",
                                 line=dict(width=1.6, color="#0b3d91")))
        fig.update_layout(
            template=TEMPLATE, height=300,
            margin=dict(l=56, r=16, t=30, b=42),
            xaxis_title="time within the dwell [s]",
            yaxis_title=f"{data['ref']} [V]",
            title=dict(text=f"{f0:.4g} Hz · dwell {(b-a)/fs:.2f} s · "
                            f"{f0*(b-a)/fs:.0f} cycles · SNR {snr_ref:.1f} dB",
                       font=dict(size=12)),
            legend=dict(orientation="h", y=1.14, x=0, font=dict(size=10)))

        # ---- panel 3: the timing between the two channels ------------------
        # The phase difference between the two fitted phasors IS the impedance
        # phase plus any acquisition skew. The skew part is what the structural
        # model removes; showing the delay in microseconds as well as degrees
        # is what makes it comparable against the sample interval.
        dphi = float(np.angle(A_seg / A_ref)) if A_ref != 0 else float("nan")
        tau = dphi / (2 * np.pi * f0) if f0 else float("nan")
        slot = (fam.position(segment) - fam.position(data["ref"])) / (fam.n_ch * fs)

        sync = go.Figure()
        norm_ref = np.real(A_ref * np.exp(1j * w * ts)) / max(abs(A_ref), 1e-30)
        norm_seg = np.real(A_seg * np.exp(1j * w * ts)) / max(abs(A_seg), 1e-30)
        norm_fix = np.real(A_seg * np.exp(-1j * w * slot) * np.exp(1j * w * ts)) \
            / max(abs(A_seg), 1e-30)
        sync.add_trace(go.Scatter(x=ts * 1e3, y=norm_ref, mode="lines",
                                  name="cell voltage",
                                  line=dict(width=2, color="#1f6feb")))
        sync.add_trace(go.Scatter(x=ts * 1e3, y=norm_seg, mode="lines",
                                  name=f"segment {segment}, as recorded",
                                  line=dict(width=2, color="#c0392b")))
        sync.add_trace(go.Scatter(x=ts * 1e3, y=norm_fix, mode="lines",
                                  name="segment, de-skewed",
                                  line=dict(width=1.6, color="#207d4a",
                                            dash="dash")))
        sync.update_layout(
            template=TEMPLATE, height=300,
            margin=dict(l=56, r=16, t=30, b=42),
            xaxis_title="time [ms]", yaxis_title="normalised amplitude",
            title=dict(text=f"measured phase difference {np.degrees(dphi):+.1f}° "
                            f"= {tau*1e6:+.1f} µs · "
                            f"structural slot skew {slot*1e6:+.1f} µs · "
                            f"one sample = {1e6/fs:.0f} µs",
                       font=dict(size=12)),
            legend=dict(orientation="h", y=1.14, x=0, font=dict(size=10)))

        # ---- panel 4: every tone -------------------------------------------
        import pandas as pd
        rows = []
        for i, st in enumerate(steps):
            n = st["stop"] - st["start"]
            seg_i = fam.channel(segment)[st["start"]:st["stop"]]
            A_i, _r, snr_i = fit3(seg_i, fs, st["freq"])
            rows.append({
                "#": i,
                "f [Hz]": round(st["freq"], 4),
                "dwell [s]": round(n / fs, 3),
                "cycles": round(st["freq"] * n / fs, 1),
                "ref amp [V]": f"{st['amp']:.3e}",
                "ref SNR [dB]": round(st["snr_db"], 1),
                "seg amp [V]": f"{abs(A_i):.3e}",
                "seg SNR [dB]": round(snr_i, 1),
                "selected": "◀" if i == int(step_index) else "",
            })
        table = ui.table(pd.DataFrame(rows), "sg-tone-table", height="260px")
        return fig, sync, table


# ---------------------------------------------------------------------------
# the CSV logger: a burst per file, and a printed channel scan
# ---------------------------------------------------------------------------


def _csv_overview(data, step_index):
    """Where the excitation actually is inside each point file.

    The x axis is the sweep, not time: one column per frequency point, drawn
    at its ANALOGUE frequency. A point whose recorded tone is a fold of
    something above Nyquist is marked, because the frequency written on the
    axis is then a reconstruction rather than a reading.
    """
    points = [p for p in data.get("points", []) if p.get("ok")]
    if not points:
        return empty_figure("no point file in this sweep carries an excitation")

    fig = go.Figure()
    f = [p["f_true"] for p in points]
    frac = [100 * p["burst"]["fraction"] for p in points]
    colours = ["#c0392b" if p["undersampled"] else "#1f6feb" for p in points]
    if isinstance(step_index, int) and 0 <= step_index < len(data["points"]):
        chosen = data["points"][step_index]
        colours = ["#207d4a" if p is chosen else c
                   for p, c in zip(points, colours)]

    fig.add_trace(go.Bar(
        x=f, y=frac, marker_color=colours, width=[0.16 * v for v in f],
        customdata=[[p["f_alias"], p["burst"]["peak_over_floor"],
                     p["burst"]["t0_s"], p["burst"]["t1_s"], p["zone"]]
                    for p in points],
        hovertemplate=("analogue %{x:.4g} Hz<br>recorded %{customdata[0]:.4g} Hz"
                       " (zone %{customdata[4]})<br>burst %{y:.0f} %% of the file"
                       "<br>%{customdata[2]:.2f}–%{customdata[3]:.2f} s"
                       "<br>peak/floor %{customdata[1]:.0f}<extra></extra>")))
    n_alias = sum(1 for p in points if p["undersampled"])
    fig.update_layout(
        template=TEMPLATE, height=260, margin=dict(l=56, r=16, t=30, b=44),
        xaxis=dict(title="analogue frequency [Hz]", type="log"),
        yaxis=dict(title="excitation burst<br>[% of the file]", range=[0, 100]),
        title=dict(text=f"{len(points)} usable point(s) · "
                        f"{n_alias} above Nyquist (red) · "
                        f"green is the selected point",
                   font=dict(size=12)),
        showlegend=False)
    return fig


def _csv_dwell(data, step_index, segment, n_cycles):
    from pathlib import Path
    _pipeline_on_path()
    import numpy as np
    import pandas as pd
    import csv_source as C
    from utils import fit3

    point = data["points"][int(step_index)]
    if not point.get("ok"):
        msg = point.get("reason", "this point file carries no excitation")
        return empty_figure(msg), empty_figure(""), ui.note(msg)

    m = C.read_r2d2(point["path"])
    fs = point["fs"]
    i0, i1 = point["burst"]["i0"], point["burst"]["i1"]
    f_alias = point["f_alias"]
    f_true = point["f_true"]

    seg = m.u_seg[segment][i0:i1]
    uc_hi = m.aux.get("uc2")
    uc_lo = m.aux.get("uc1")
    if uc_hi is None:
        key = sorted(m.aux)[0] if m.aux else None
        uc_hi = m.aux[key] if key else seg
        uc_lo = None
    ref = (uc_hi[i0:i1] - uc_lo[i0:i1]) if uc_lo is not None else uc_hi[i0:i1]
    t = np.arange(seg.size) / fs

    A_ref, _r, snr_ref = fit3(ref, fs, f_alias)
    A_seg, _r2, snr_seg = fit3(seg, fs, f_alias)

    n_show = int(min(seg.size, max(1, n_cycles or 6) * fs / max(f_alias, 1e-9)))
    n_show = max(n_show, 32)
    ts = t[:n_show]
    w = 2 * np.pi * f_alias

    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=ts * 1e3, y=seg[:n_show], mode="markers",
                               name=f"segment {segment} samples",
                               marker=dict(size=3, color="#c0392b")))
    fig.add_trace(go.Scatter(
        x=ts * 1e3,
        y=np.real(A_seg * np.exp(1j * w * ts)) + float(np.mean(seg)),
        mode="lines", name="fitted sine",
        line=dict(width=1.6, color="#7b1d10")))
    fig.update_layout(
        template=TEMPLATE, height=300, margin=dict(l=60, r=16, t=30, b=42),
        xaxis_title="time within the burst [ms]",
        yaxis_title=f"segment {segment} [A/cm²]",
        title=dict(text=f"recorded {f_alias:.4g} Hz · burst "
                        f"{point['burst']['t0_s']:.2f}–{point['burst']['t1_s']:.2f} s "
                        f"({100*point['burst']['fraction']:.0f} % of the file) · "
                        f"SNR {snr_seg:.1f} dB", font=dict(size=12)),
        legend=dict(orientation="h", y=1.14, x=0, font=dict(size=10)))

    # ---- the scan skew, which is printed rather than fitted ---------------
    tau_seg = m.timeshifts.get(f"s{segment}", 0.0)
    tau_ref = m.timeshifts.get("uc2", 0.0)
    d_tau = tau_ref - tau_seg
    deg_true = 360.0 * f_true * d_tau
    deg_alias = 360.0 * f_alias * d_tau

    sync = go.Figure()
    nr = np.real(A_ref * np.exp(1j * w * ts)) / max(abs(A_ref), 1e-30)
    ns = np.real(A_seg * np.exp(1j * w * ts)) / max(abs(A_seg), 1e-30)
    nf = np.real(A_seg * np.exp(-1j * 2 * np.pi * f_true * (tau_seg - tau_ref))
                 * np.exp(1j * w * ts)) / max(abs(A_seg), 1e-30)
    sync.add_trace(go.Scatter(x=ts * 1e3, y=nr, mode="lines",
                              name="cell voltage (uc2 − uc1)",
                              line=dict(width=2, color="#1f6feb")))
    sync.add_trace(go.Scatter(x=ts * 1e3, y=ns, mode="lines",
                              name=f"segment {segment}, as recorded",
                              line=dict(width=2, color="#c0392b")))
    sync.add_trace(go.Scatter(x=ts * 1e3, y=nf, mode="lines",
                              name="segment, de-skewed",
                              line=dict(width=1.6, color="#207d4a", dash="dash")))
    sync.update_layout(
        template=TEMPLATE, height=300, margin=dict(l=60, r=16, t=30, b=42),
        xaxis_title="time [ms]", yaxis_title="normalised amplitude",
        title=dict(text=f"printed scan offset {d_tau*1e6:+.1f} µs "
                        f"= {deg_true:+.0f}° at the analogue {f_true:.4g} Hz "
                        f"({deg_alias:+.0f}° at the recorded {f_alias:.4g} Hz) · "
                        f"one sample = {1e6/fs:.1f} µs",
                   font=dict(size=12)),
        legend=dict(orientation="h", y=1.14, x=0, font=dict(size=10)))

    rows = []
    for i, pt in enumerate(data["points"]):
        if not pt.get("ok"):
            rows.append({"#": i, "point": Path(pt["path"]).stem,
                         "status": pt.get("reason", "")[:60]})
            continue
        rows.append({
            "#": i, "point": pt["name"],
            "recorded [Hz]": round(pt["f_alias"], 4),
            "analogue [Hz]": round(pt["f_true"], 4),
            "zone": pt["zone"],
            "above Nyquist": "yes" if pt["undersampled"] else "no",
            "burst [%]": round(100 * pt["burst"]["fraction"], 0),
            "peak/floor": round(pt["burst"]["peak_over_floor"], 0),
            "phase R": round(pt["R"], 3),
            "uc spread [%]": round(100 * pt["uc_spread"], 1),
            "selected": "◀" if i == int(step_index) else "",
        })
    return fig, sync, ui.table(pd.DataFrame(rows), "sg-tone-table", height="260px")
