"""The dashboard.  ``streamlit run run.py``.

Six pages over one results root:

  Setup        what .env resolved to, and what is wrong with it
  Run          launch the pipeline for one or more conditions
  Plate map    per-segment parameters over the plate geometry
  Spectra      Nyquist / Bode, per segment and whole-cell
  Diagnostics  card alignment, quality tiers, rejected points
  Files        every artifact the run wrote, with downloads
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import pipeline, results, theme
from .settings import DOTENV_LOADED, Settings
from .settings import load as load_settings

PAGES = ["Setup", "Run", "Plate map", "Spectra", "Diagnostics", "Files"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _mode() -> str:
    try:
        return "dark" if st.get_option("theme.base") == "dark" else "light"
    except Exception:
        return "light"


def _severity_box(sev: str, setting: str, msg: str) -> None:
    (st.error if sev == "error" else st.warning)(f"**{setting}** -- {msg}")


def _fmt(v, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--"
    return f"{v:,.{nd}g}"


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def page_setup(stg: Settings) -> None:
    st.subheader("Configuration")
    st.caption(
        f"Read from `{DOTENV_LOADED or '(no .env found -- environment only)'}`. "
        "Anything already exported in the environment wins over the file."
    )
    problems = stg.problems()
    errs = [p for p in problems if p[0] == "error"]
    if not problems:
        st.success("Every configured path exists.")
    else:
        for sev, name, msg in problems:
            _severity_box(sev, name, msg)
    if errs:
        st.info(
            "Fix these in `.env` (copy `.env.example` if you have not yet). "
            "The two that matter are **EIS_DAT_DIR** -- where the .DAT "
            "recordings are read from -- and **EIS_OUT_DIR** -- where results "
            "are written."
        )

    st.dataframe(pd.DataFrame(stg.as_rows()), width="stretch",
                 hide_index=True)

    st.subheader("What is in the input directory")
    conds = pipeline.discover_conditions(stg.dat_dir)
    leepa = stg.leepa or pipeline.discover_leepa(stg.dat_dir)
    if conds:
        c1, c2 = st.columns(2)
        c1.metric("conditions found", len(conds), help=", ".join(conds))
        c2.metric("plate / order id", leepa or "unknown")
        st.write("Conditions: " + ", ".join(f"`{c}`" for c in conds))
    else:
        st.info("No .DAT files with a recognisable current setpoint in the "
                "name were found. The dashboard can still browse results "
                "already under EIS_OUT_DIR.")

    st.subheader("Results already present")
    runs = results.discover(stg.out_dir)
    if runs:
        st.dataframe(pd.DataFrame([
            {"run": r.label, "path": str(r.path),
             "gold": "yes" if r.has(results.PLATE_SUMMARY) else "no"}
            for r in runs]), width="stretch", hide_index=True)
    else:
        st.caption(f"Nothing under `{stg.out_dir}` yet.")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def page_run(stg: Settings) -> None:
    if stg.read_only:
        st.info("EIS_READ_ONLY is set -- this deployment browses results only.")
        return
    errs = [p for p in stg.problems() if p[0] == "error"]
    if errs:
        st.error("Configuration has errors; fix them on the Setup page first.")
        for sev, name, msg in errs:
            _severity_box(sev, name, msg)
        return

    found = pipeline.discover_conditions(stg.dat_dir)
    leepa = stg.leepa or pipeline.discover_leepa(stg.dat_dir)
    default = ([c for c in stg.conditions if c in found]
               or found[:1]) if "ALL" not in stg.conditions else found

    st.subheader("What to run")
    c1, c2 = st.columns([2, 1])
    conditions = c1.multiselect("Conditions", found, default=default)
    stop_after = c2.selectbox("Stop after", ["gold", "silver", "bronze"])

    with st.expander("Algorithm and band"):
        a1, a2, a3 = st.columns(3)
        skew = a1.selectbox("Skew model", ["structural", "per_card", "none"])
        phasor = a2.selectbox("Phasor fit", ["joint7", "independent"])
        preset = a3.selectbox("Gate preset", ["default", "permissive", "strict"])
        b1, b2 = st.columns(2)
        use_band = b1.checkbox("Override band", value=False)
        f_min = f_max = None
        if use_band:
            f_min = b1.number_input("f_min (Hz)", value=0.15, format="%.4f")
            f_max = b2.number_input("f_max (Hz)", value=4500.0, format="%.1f")
        o1, o2, o3, o4 = st.columns(4)
        drt = o1.checkbox("DRT", value=True,
                          help="off loses the R_ct / R_mt process split")
        spatial = o2.checkbox("Infer missing", value=True)
        png = o3.checkbox("PNG maps", value=True)
        html = o4.checkbox("HTML maps", value=True)

    specs = [pipeline.RunSpec(condition=c, stop_after=stop_after, skew=skew,
                              phasor=phasor, preset=preset, f_min=f_min,
                              f_max=f_max, drt=drt, spatial=spatial,
                              png=png, html=html) for c in conditions]
    if specs:
        st.caption("Command for the first condition:")
        st.code(" ".join(pipeline.build_command(stg, specs[0], leepa)),
                language="bash")

    queue = st.session_state.setdefault("queue", [])
    launcher: pipeline.Launcher | None = st.session_state.get("launcher")

    go_col, stop_col = st.columns([1, 1])
    if go_col.button("Run", type="primary",
                     disabled=not specs or (launcher is not None
                                            and launcher.running)):
        st.session_state["queue"] = list(specs)
        st.session_state["launcher"] = None
        st.rerun()
    if stop_col.button("Stop", disabled=launcher is None
                       or not launcher.running):
        launcher.stop()
        st.session_state["queue"] = []

    # start the next queued run when nothing is in flight
    if (launcher is None or not launcher.running) and queue:
        if launcher is None or launcher.returncode == 0:
            spec = queue.pop(0)
            out = pipeline.out_dir_for(stg, spec, leepa)
            launcher = pipeline.Launcher(
                pipeline.build_command(stg, spec, leepa),
                cwd=stg.pipeline_dir, log_path=out / "run.log")
            launcher.start()
            st.session_state["launcher"] = launcher
            st.session_state["queue"] = queue
        else:
            st.error(f"Previous condition exited {launcher.returncode}; "
                     f"{len(queue)} queued run(s) cancelled.")
            st.session_state["queue"] = []

    if launcher is not None:
        st.subheader("Log")
        if launcher.running:
            st.caption("running...")
        elif launcher.returncode == 0:
            st.success("finished")
        else:
            st.error(f"exited with code {launcher.returncode}")
        st.code(launcher.text() or "(waiting for output)", language="text")
        if launcher.running:
            # Streamlit has no push channel, so the log advances when the
            # script reruns.  A button is an honest manual poll; a timed
            # auto-rerun would also restart every widget on the page.
            st.button("Refresh log", key="poll")
            st.caption("The pipeline runs in a subprocess -- leaving this page "
                       "or closing the tab does not stop it. Press Refresh "
                       "log, or come back to this page, to see progress.")


# ---------------------------------------------------------------------------
# Plate map
# ---------------------------------------------------------------------------


#: Parameters whose slow-tau bucket can be outside the measured band.
_BAND_LIMITED_PARAMS = {"R_mt"}


def plate_figure(df: pd.DataFrame, param: str, mode: str) -> go.Figure:
    """Segment centroids coloured by one parameter.

    Squares at the segment centroids rather than an interpolated surface: the
    plate has 72 discrete segments and smoothing between them would draw
    gradients the measurement never resolved.
    """
    v = pd.to_numeric(df.get(param), errors="coerce")
    cls = df.get("class", pd.Series("measured", index=df.index)).astype(str)
    limited = (results.band_limited_mt(df) if param in _BAND_LIMITED_PARAMS
               else pd.Series(False, index=df.index))

    fig = go.Figure()
    # one trace per state so every state gets a legend entry and a symbol
    have = v.notna() & (cls == "measured") & ~limited
    if have.any():
        d = df[have]
        fig.add_trace(go.Scatter(
            x=d["cx_mm"], y=d["cy_mm"], mode="markers", name="measured",
            marker=dict(size=22, symbol="square", color=v[have],
                        colorscale=theme.plotly_colorscale(),
                        colorbar=dict(title=param, thickness=14, x=1.02,
                                      len=0.9, y=0.5),
                        line=dict(width=2, color=theme.SURFACE[mode])),
            customdata=np.stack([d["segment"], v[have]], axis=-1),
            hovertemplate=("segment %{customdata[0]:.0f}<br>"
                           + param + " %{customdata[1]:.4g}<extra></extra>")))
    for key, mask in (("band_limited", limited),
                      ("inferred", (cls == "inferred") & ~limited),
                      ("bad", (cls == "bad") & ~limited)):
        if not mask.any():
            continue
        spec = theme.SEGMENT_STATE[key]
        d = df[mask]
        fig.add_trace(go.Scatter(
            x=d["cx_mm"], y=d["cy_mm"], mode="markers", name=spec["label"],
            marker=dict(size=20, symbol=spec["symbol"], color=spec["color"],
                        line=dict(width=2, color=theme.SURFACE[mode])),
            customdata=d[["segment"]],
            hovertemplate=("segment %{customdata[0]:.0f}<br>"
                           + spec["label"] + "<extra></extra>")))

    # Legend below the plot, not beside it: the colorbar already owns the
    # right-hand gutter and the two overlap there.
    fig.update_layout(**theme.layout(
        mode, title=f"{param} over the plate", height=600,
        xaxis_title="x (mm)", yaxis_title="y (mm)", showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        margin=dict(l=60, r=110, t=48, b=90)))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def page_plate(run: results.Run | None, mode: str) -> None:
    if run is None:
        st.info("Pick a run in the sidebar.")
        return
    df = results.plate_summary(run)
    if df.empty:
        st.warning(f"No `{results.PLATE_SUMMARY}` under `{run.path}` -- "
                   f"the run may have stopped before gold.")
        return

    params = results.parameter_columns(df)
    if not params:
        st.warning("plate_summary.csv has no numeric parameter columns.")
        return
    pref = [p for p in ("R_ohmic", "R_ct", "R_mt", "R_pol", "j_dc") if p in params]
    param = st.selectbox("Parameter", params,
                         index=params.index(pref[0]) if pref else 0)

    st.plotly_chart(plate_figure(df, param, mode), width="stretch")

    _plate_stats(df, param)
    if param in _BAND_LIMITED_PARAMS:
        _mass_transport_note(df)

    with st.expander("Table view"):
        cols = ["segment", "class", "tier", param, param + "_sd", "fault"]
        st.dataframe(df[[c for c in cols if c in df.columns]],
                     width="stretch", hide_index=True)


def _plate_stats(df: pd.DataFrame, param: str) -> None:
    v = pd.to_numeric(df.get(param), errors="coerce")
    meas = v[(df.get("class") == "measured") & v.notna() & (v > 0)]
    c = st.columns(5)
    c[0].metric("segments", len(df))
    c[1].metric("measured", int((df.get("class") == "measured").sum()))
    c[2].metric("median", _fmt(meas.median()) if len(meas) else "--")
    c[3].metric("spread (max/min)",
                _fmt(meas.max() / meas.min(), 3) if len(meas) and meas.min() > 0
                else "--")
    c[4].metric("missing", int(v.isna().sum()))


def _mass_transport_note(df: pd.DataFrame) -> None:
    """Explain a zero or blank R_mt instead of letting it be read as data."""
    limited = results.band_limited_mt(df)
    zeros = pd.to_numeric(df.get("R_mt"), errors="coerce").eq(0.0)
    if "tau_max" not in df.columns:
        if zeros.any():
            st.warning(
                f"**{int(zeros.sum())} segment(s) report R_mt exactly 0.** "
                "This run predates the `tau_max` column, so the dashboard "
                "cannot tell a measured zero from a bucket the band never "
                "reached. Re-run gold to get the distinction."
            )
        return
    if not limited.any():
        return
    segs = ", ".join(str(int(s)) for s in df.loc[limited, "segment"])
    st.warning(
        f"**{int(limited.sum())} segment(s) have no mass-transport estimate "
        f"at all:** {segs}.\n\n"
        "R_mt is the sum of the DRT over tau >= 10 ms. The tau grid stops at "
        "`1/(2*pi*f_min)`, so that bucket only exists for a segment that kept "
        "a point at or below **15.9 Hz**. These segments did not, so the "
        "bucket is empty -- not small, *absent*. They are drawn as "
        "band-limited rather than as zero, because a zero here would mean "
        "'no mass transport', which is not what was measured.\n\n"
        "This is a property of the segment's low-frequency SNR, not of the "
        "operating point, which is why the same segments come out this way "
        "at every condition."
    )


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------


def _spectra_frame(run: results.Run) -> pd.DataFrame:
    for rel in ("silver/impedance.csv", "bronze/raw_spectra.csv"):
        df = results.read_csv(run, rel)
        if not df.empty:
            df.attrs["source"] = rel
            return df
    return pd.DataFrame()


def _cols(df: pd.DataFrame, *names: str) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def page_spectra(run: results.Run | None, mode: str) -> None:
    if run is None:
        st.info("Pick a run in the sidebar.")
        return
    df = _spectra_frame(run)
    if df.empty:
        st.warning("No per-segment spectra found "
                   "(silver/impedance.csv or bronze/raw_spectra.csv).")
        return
    st.caption(f"source: `{df.attrs.get('source', '')}`")

    c_seg = _cols(df, "segment", "seg")
    c_f = _cols(df, "freq_hz", "f_hz", "freq", "frequency_hz")
    c_re = _cols(df, "z_re", "zre", "re_z", "z_real_mohm_cm2", "z_re_mohm_cm2")
    c_im = _cols(df, "z_im", "zim", "im_z", "z_imag_mohm_cm2", "z_im_mohm_cm2")
    if not all((c_f, c_re, c_im)):
        st.warning(f"Could not find frequency / Re Z / Im Z columns among: "
                   f"{', '.join(df.columns)}")
        st.dataframe(df.head(50), width="stretch")
        return

    segs = (sorted(df[c_seg].dropna().unique(), key=lambda s: int(float(s)))
            if c_seg else [])
    picked = st.multiselect(
        "Segments (max 3 overlaid -- past three the colours stop being "
        "separable under colour-vision deficiency)",
        segs, default=segs[:1], max_selections=theme.SCATTER_SLOTS)

    kind = st.radio("View", ["Nyquist", "Bode |Z|", "Bode phase"],
                    horizontal=True)
    pal = theme.CATEGORICAL[mode]
    fig = go.Figure()
    for i, s in enumerate(picked or segs[:1]):
        d = df[df[c_seg] == s] if c_seg else df
        d = d.sort_values(c_f)
        re, im, f = d[c_re].to_numpy(), d[c_im].to_numpy(), d[c_f].to_numpy()
        color = pal[i % theme.SCATTER_SLOTS]
        if kind == "Nyquist":
            fig.add_trace(go.Scatter(
                x=re, y=-im, mode="lines+markers", name=f"segment {s}",
                line=dict(width=2, color=color),
                marker=dict(size=8, color=color,
                            line=dict(width=2, color=theme.SURFACE[mode])),
                customdata=f,
                hovertemplate=("%{customdata:.4g} Hz<br>Re %{x:.4g}"
                               "<br>-Im %{y:.4g}<extra></extra>")))
        elif kind == "Bode |Z|":
            fig.add_trace(go.Scatter(
                x=f, y=np.hypot(re, im), mode="lines+markers",
                name=f"segment {s}", line=dict(width=2, color=color),
                marker=dict(size=8, color=color),
                hovertemplate="%{x:.4g} Hz<br>|Z| %{y:.4g}<extra></extra>"))
        else:
            fig.add_trace(go.Scatter(
                x=f, y=np.degrees(np.arctan2(im, re)), mode="lines+markers",
                name=f"segment {s}", line=dict(width=2, color=color),
                marker=dict(size=8, color=color),
                hovertemplate="%{x:.4g} Hz<br>%{y:.2f} deg<extra></extra>"))

    if kind == "Nyquist":
        lay = theme.layout(mode, title="Nyquist", height=560,
                           xaxis_title="Re Z", yaxis_title="-Im Z")
        fig.update_layout(**lay)
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    else:
        fig.update_layout(**theme.layout(
            mode, title=kind, height=520, xaxis_title="frequency (Hz)",
            yaxis_title="|Z|" if "|Z|" in kind else "phase (deg)"))
        fig.update_xaxes(type="log")
        if "|Z|" in kind:
            fig.update_yaxes(type="log")
    # A single series needs no legend box -- the title names it.
    fig.update_layout(showlegend=len(picked) > 1)
    st.plotly_chart(fig, width="stretch")

    agg = results.read_csv(run, "silver/cell_aggregate.csv")
    if not agg.empty:
        with st.expander("Whole-cell aggregate"):
            st.dataframe(agg, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def page_diagnostics(run: results.Run | None, mode: str) -> None:
    if run is None:
        st.info("Pick a run in the sidebar.")
        return

    st.subheader("Card alignment")
    bm = results.read_json(run, "bronze/bronze_manifest.json")
    lag_s = bm.get("card_lag_s") or {}
    corr = bm.get("card_lag_corr") or {}
    if lag_s:
        rows = [{"card": k, "lag (s)": v, "peak |r|": corr.get(k)}
                for k, v in lag_s.items()]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(
            "A card left on its own clock reads every dwell window at the "
            "wrong instant. Two cards returning nearly the same lag is "
            "evidence the lag is real even when the correlation peak is weak."
        )
    else:
        st.caption("No card_lag_s in bronze_manifest.json.")

    st.subheader("Quality tiers")
    gm = results.read_json(run, "gold/gold_manifest.json")
    tiers = gm.get("tiers") or {}
    if tiers:
        c = st.columns(len(tiers))
        for i, (k, v) in enumerate(sorted(tiers.items())):
            c[i].metric(f"tier {k}", v)
    stat_cols = st.columns(4)
    for i, k in enumerate(("n_total", "n_measured", "n_inferred", "n_bad")):
        if k in gm:
            stat_cols[i].metric(k.replace("n_", ""), gm[k])

    st.subheader("Plausibility")
    pl = results.read_csv(run, "gold/plausibility.csv")
    if pl.empty:
        pl = results.read_csv(run, "plausibility.csv")
    if not pl.empty:
        st.dataframe(pl, width="stretch", hide_index=True)
    else:
        st.caption("No plausibility report in this run.")

    st.subheader("Rejected points")
    rej = results.read_csv(run, "silver/point_rejections.csv")
    if rej.empty:
        st.caption("No point_rejections.csv.")
        return
    reason = _cols(rej, "reason", "cause", "gate")
    if reason:
        counts = (rej[reason].value_counts().rename_axis("reason")
                  .reset_index(name="points"))
        fig = go.Figure(go.Bar(
            x=counts["points"], y=counts["reason"], orientation="h",
            marker=dict(color=theme.CATEGORICAL[mode][0],
                        cornerradius=4),
            hovertemplate="%{y}<br>%{x} points<extra></extra>"))
        fig.update_layout(**theme.layout(
            mode, title="Why points were dropped", height=80 + 30 * len(counts),
            xaxis_title="points", yaxis_title="", showlegend=False))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(counts, width="stretch", hide_index=True)
    with st.expander("All rejected points"):
        st.dataframe(rej, width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def page_files(run: results.Run | None) -> None:
    if run is None:
        st.info("Pick a run in the sidebar.")
        return
    st.caption(f"`{run.path}`")
    files = sorted(p for p in run.path.rglob("*") if p.is_file())
    if not files:
        st.warning("This run directory is empty.")
        return
    rows = [{"file": str(p.relative_to(run.path)),
             "size (KB)": round(p.stat().st_size / 1024, 1)} for p in files]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    pngs = [p for p in files if p.suffix.lower() == ".png"]
    if pngs:
        st.subheader("Maps")
        pick = st.selectbox("Image", [str(p.relative_to(run.path))
                                      for p in pngs])
        st.image(str(run.path / pick), width="stretch")

    st.subheader("Download")
    rel = st.selectbox("File", [r["file"] for r in rows], key="dl")
    p = run.path / rel
    st.download_button(f"Download {Path(rel).name}", p.read_bytes(),
                       file_name=Path(rel).name)

    log = results.log_text(run)
    if log:
        with st.expander("Run log"):
            st.code(log, language="text")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Local EIS", layout="wide",
                       page_icon="\N{HIGH VOLTAGE SIGN}")
    stg = load_settings()
    mode = _mode()

    st.sidebar.title("Local EIS")
    page = st.sidebar.radio("Page", PAGES, label_visibility="collapsed")

    runs = results.discover(stg.out_dir)
    run: results.Run | None = None
    if runs:
        labels = [f"{r.label}" for r in runs]
        idx = st.sidebar.selectbox("Run", range(len(runs)),
                                   format_func=lambda i: labels[i])
        run = runs[idx]
        st.sidebar.caption(str(run.path))
    else:
        st.sidebar.caption(f"No results under {stg.out_dir}")

    if st.sidebar.button("Rescan results"):
        st.rerun()
    st.sidebar.divider()
    st.sidebar.caption(f".env: {DOTENV_LOADED or 'not found'}")

    st.title(f"{page}"
             + (f"  --  {run.label}" if run and page not in ("Setup", "Run")
                else ""))

    if page == "Setup":
        page_setup(stg)
    elif page == "Run":
        page_run(stg)
    elif page == "Plate map":
        page_plate(run, mode)
    elif page == "Spectra":
        page_spectra(run, mode)
    elif page == "Diagnostics":
        page_diagnostics(run, mode)
    else:
        page_files(run)
