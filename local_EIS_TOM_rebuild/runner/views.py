"""
views.py
========
The plots of the runner notebook, built from a finished run folder. Pure
functions of files on disk -- no widgets, no Spark -- so each one can be
tested, and called from a script, without Databricks.

    nyquist_figure        every segment + the Gamry sweep of that run
    median_figure         plate median, card medians, aggregate, Gamry
    frequency_response    ex-situ amplifiers + in-situ response (images)
    bench_figure_whole    the WHOLE bench .mf4, every channel, sweeps marked
    ecm_fits / ecm_viewer standard ECM and DRT-informed ECM, per spectrum

Units everywhere: mOhm*cm2, Hz.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import settings as S

GAMRY_COLOUR = "black"
AGG_COLOUR = "#c0392b"
MEDIAN_COLOUR = "#1f4e9c"
CARD_COLOURS = ("#2a9d8f", "#e9c46a", "#f4a261", "#8e44ad", "#e76f51",
                "#457b9d")
STD_COLOUR = "#111111"
DRT_COLOUR = "#d35400"


# ---------------------------------------------------------------------------
# reading a run folder
# ---------------------------------------------------------------------------
def spectra(run_dir) -> pd.DataFrame | None:
    for sub in ("silver", "csv"):
        p = Path(run_dir) / sub / "spectra_clean.csv"
        if p.is_file():
            return pd.read_csv(p)
    return None


def aggregate(run_dir) -> pd.DataFrame | None:
    for sub in ("silver", "csv"):
        p = Path(run_dir) / sub / "cell_aggregate.csv"
        if p.is_file():
            return pd.read_csv(p).sort_values("freq_hz")
    return None


def gamry_of_run(run_dir):
    """(DataFrame freq_hz/z_re/z_im in mOhm*cm2, sweep name) of the whole-cell
    sweep THIS run was compared against -- the one picked by time for its
    test (gamry_compare.select_sweep), written by the run itself. Never a
    sweep looked up again by current alone."""
    run_dir = Path(run_dir)
    p = run_dir / "gamry_reference_full.csv"
    if not p.is_file():
        return None, ""
    g = pd.read_csv(p).rename(columns={"z_re_ref_mohm_cm2": "z_re_mohm_cm2",
                                       "z_im_ref_mohm_cm2": "z_im_mohm_cm2"})
    name = ""
    c = run_dir / "gamry_comparison.csv"
    if c.is_file():
        try:
            name = str(pd.read_csv(c)["reference"].iloc[0])
        except Exception:                                   # noqa: BLE001
            name = ""
    return g[["freq_hz", "z_re_mohm_cm2", "z_im_mohm_cm2"]], name


def segment_cards(run_dir) -> dict:
    """{segment: 'Karte_n'} from bronze/segment_meta.csv."""
    p = Path(run_dir) / "bronze" / "segment_meta.csv"
    if not p.is_file():
        return {}
    m = pd.read_csv(p)
    return {str(s): str(c).split("_")[-2] + "_" + str(c).split("_")[-1]
            if "Karte" in str(c) else str(c)
            for s, c in zip(m["segment"], m["card"])}


# ---------------------------------------------------------------------------
# Nyquist / Bode of every segment
# ---------------------------------------------------------------------------
def nyquist_figure(run_dir, title: str, colour: str = S.NYQUIST_COLOUR):
    import nyquist
    df = spectra(run_dir)
    if df is None:
        return None
    rec_p = Path(run_dir) / "silver" / "spectra_reconstructed.csv"
    rec = pd.read_csv(rec_p) if rec_p.is_file() else None
    g, gname = gamry_of_run(run_dir)
    return nyquist.figure(df, title=title, colour=colour, rebuilt=rec,
                          gamry=g, gamry_name=f"Gamry {gname}".strip())


# ---------------------------------------------------------------------------
# median Nyquist: plate median, card medians, aggregate, Gamry
# ---------------------------------------------------------------------------
def _median_by_freq(d: pd.DataFrame) -> pd.DataFrame:
    """Median of Z' and Z'' over segments at each frequency (frequencies
    seen by at least half of the segments)."""
    n_seg = d["segment"].nunique()
    g = d.groupby("freq_hz").agg(re=("z_re_mohm_cm2", "median"),
                                 im=("z_im_mohm_cm2", "median"),
                                 n=("segment", "nunique")).reset_index()
    return g[g["n"] >= max(1, n_seg // 2)]


def median_figure(run_dir, title: str):
    """Nyquist, |Z| and phase: the Gamry sweep (black), the aggregate of the
    segments in parallel (red), the plate median over segments (blue) and
    the median of each card (thin)."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    df = spectra(run_dir)
    if df is None:
        return None
    df = df[np.isfinite(df["z_re_mohm_cm2"]) & np.isfinite(df["z_im_mohm_cm2"])]
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.06,
                        subplot_titles=["Nyquist", "|Z|(f) Bode", "Phase(f)"])

    def add(f, Z, name, color, width, mode="lines+markers", dash=None,
            size=4, group=None):
        o = np.argsort(f)
        f, Z = np.asarray(f, float)[o], np.asarray(Z, complex)[o]
        kw = dict(mode=mode, legendgroup=group or name,
                  line=dict(color=color, width=width, dash=dash),
                  marker=dict(size=size, color=color), customdata=f)
        fig.add_trace(go.Scatter(
            x=Z.real, y=-Z.imag, name=name, hovertemplate=(
                f"<b>{name}</b><br>f = %{{customdata:.5g}} Hz<br>"
                "Z' = %{x:.1f}<br>-Z'' = %{y:.1f} mΩ·cm²<extra></extra>"),
            **kw), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=f, y=np.abs(Z), name=name, showlegend=False, hovertemplate=(
                f"<b>{name}</b><br>f = %{{x:.5g}} Hz<br>|Z| = %{{y:.1f}}"
                "<extra></extra>"), **kw), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=f, y=np.degrees(np.angle(Z)), name=name, showlegend=False,
            hovertemplate=(f"<b>{name}</b><br>f = %{{x:.5g}} Hz<br>"
                           "phase = %{y:.1f}°<extra></extra>"), **kw),
            row=1, col=3)

    cards = segment_cards(run_dir)
    if cards:
        df = df.assign(card=df["segment"].astype(str).map(cards))
        for i, (c, d) in enumerate(sorted(df.groupby("card"))):
            m = _median_by_freq(d)
            add(m["freq_hz"], m["re"] + 1j * m["im"],
                f"{c} median ({d['segment'].nunique()} seg)",
                CARD_COLOURS[i % len(CARD_COLOURS)], 1.4, size=3)
    m = _median_by_freq(df)
    add(m["freq_hz"], m["re"] + 1j * m["im"],
        f"Plate median ({df['segment'].nunique()} segments)", MEDIAN_COLOUR,
        3.0, size=5)
    a = aggregate(run_dir)
    if a is not None and len(a):
        add(a["freq_hz"], a["z_re_mohm_cm2"] + 1j * a["z_im_mohm_cm2"],
            "Aggregate (segments in parallel)", AGG_COLOUR, 3.0, size=5)
    g, gname = gamry_of_run(run_dir)
    if g is not None and len(g):
        add(g["freq_hz"], g["z_re_mohm_cm2"] + 1j * g["z_im_mohm_cm2"],
            f"Gamry {gname}".strip(), GAMRY_COLOUR, 2.5, size=4)
    fig.update_xaxes(title_text="Z' [mΩ·cm²]", row=1, col=1)
    fig.update_yaxes(title_text="-Z'' [mΩ·cm²]", scaleanchor="x",
                     scaleratio=1, row=1, col=1)
    fig.update_xaxes(title_text="f [Hz]", type="log", row=1, col=2)
    fig.update_yaxes(title_text="|Z| [mΩ·cm²]", type="log", row=1, col=2)
    fig.update_xaxes(title_text="f [Hz]", type="log", row=1, col=3)
    fig.update_yaxes(title_text="Phase [°]", row=1, col=3)
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    fig.update_layout(title=f"<b>{title}</b>", height=550, width=1500,
                      paper_bgcolor="white", plot_bgcolor="white",
                      margin=dict(t=80, b=50, r=270), hovermode="closest",
                      legend=dict(font=dict(size=10), y=0.5,
                                  yanchor="middle"))
    return fig


# ---------------------------------------------------------------------------
# frequency response of the measuring chain
# ---------------------------------------------------------------------------
def frequency_response(run_dir, gain_file=None, bode_dir=None,
                       scratch=None) -> dict:
    """{'exsitu': png or None, 'insitu': png or None, 'checks': [...]}.
    Each run writes RUN_DIR/frequency_response/; a saved run that predates
    it is evaluated here into `scratch`."""
    import frequency_response as fr
    run_dir = Path(run_dir)
    d = run_dir / fr.OUT
    summ = d / "summary.json"
    if summ.is_file():
        s = json.loads(summ.read_text())
    else:
        d = Path(scratch or d)
        d.mkdir(parents=True, exist_ok=True)
        s = fr.run(run_dir, gain_file=gain_file, bode_dir=bode_dir,
                   out_dir=d, f_top=4000.0,
                   title=f"In-situ response, {run_dir.name}")
    return {"exsitu": (d / "freq_response_exsitu.png"
                       if (d / "freq_response_exsitu.png").is_file() else None),
            "insitu": (d / "freq_response_insitu.png"
                       if (d / "freq_response_insitu.png").is_file() else None),
            "checks": [str(c) for c in fr.checks(s)]}


# ---------------------------------------------------------------------------
# the whole bench log
# ---------------------------------------------------------------------------
def channels_from_values(vals: dict, max_points: int = S.MF4_MAX_POINTS,
                         include_others: bool = True) -> list:
    """bench_plots.Channel list from {name: (t_s, v)} (datago), whole time."""
    import bench_plots as bp
    groups = bp.CHANNEL_GROUPS
    known = {n: (g, lab, unit) for g, ch in groups.items() for n, lab, unit in ch}
    order = [n for ch in groups.values() for n, _, _ in ch if n in vals]
    if include_others:
        order += sorted((n for n in vals if n not in known), key=str.lower)
    out = []
    for n in order:
        t = np.asarray(vals[n][0], float)
        v = np.asarray(vals[n][1], float)
        o = np.argsort(t, kind="stable")
        tv = bp._clean(t[o], v[o])
        if tv is None or tv[0].size < 2:
            continue
        t, v = bp.decimate(*tv, max_points=max_points)
        g, lab, unit = known.get(n, (bp.OTHER_GROUP, n, ""))
        out.append(bp.Channel(n, lab, unit, g, t, v))
    return out


def bench_figure_whole(channels: list, title: str, sweeps=(),
                       reference_lines=None):
    """The whole bench recording, every channel (parameter drop-down of
    bench_plots), with each Gamry sweep shaded and labelled.

    sweeps: [(label, start_s, end_s)] on the bench log's time axis."""
    import bench_plots as bp
    fig = bp.bench_figure(channels, title=title, start="Electrical",
                          reference_lines=reference_lines, height=680)
    for i, (lab, a, b) in enumerate(sweeps):
        fig.add_vrect(x0=a / 60.0, x1=max(b, a + 20) / 60.0,
                      fillcolor="#ffcf56" if i % 2 else "#9ecae1",
                      opacity=0.25, line_width=0, layer="below",
                      annotation_text=lab, annotation_position="top left",
                      annotation_font_size=9)
    fig.update_xaxes(title_text="time since the start of the bench log [min]")
    return fig


# ---------------------------------------------------------------------------
# ECM: standard (all segments, the circuit of the maps) and DRT-informed
# ---------------------------------------------------------------------------
def _clean(f, Z, s=None):
    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    ok = np.isfinite(f) & (f > 0) & np.isfinite(Z.real) & np.isfinite(Z.imag)
    o = np.argsort(f[ok])
    out = [f[ok][o], Z[ok][o]]
    if s is not None:
        out.append(np.asarray(s, float)[ok][o])
    return out


def fit_standard(f, Z, sigma=None) -> dict:
    """R_ohmic + L + two ZARCs, complex NLLS (standard_eis.fit_spectrum, the
    fit behind R_ct / R_mt on the maps). Z in mOhm*cm2."""
    import standard_eis as se
    import csv_pipeline as cp
    r = se.choose_n_arcs(f, Z / 1000.0, sigma)
    if not r.get("ok"):
        return {"ok": False, "reason": r.get("reason", "no fit")}
    p = r["params"]
    v = [p["Rs"], p["L"]]
    for k in range(r["n_arcs"]):
        v += [p[f"R{k+1}"], p[f"tau{k+1}"], p[f"n{k+1}"]]
    fs = np.logspace(np.log10(f.min()), np.log10(f.max()), S.ECM_SMOOTH_POINTS)
    Zf = 1000.0 * cp.z_model(np.asarray(v), f, r["n_arcs"])
    return {"ok": True, "n_arcs": int(r["n_arcs"]),
            "Rs": 1000 * p["Rs"], "L": 1000 * p["L"],
            "arcs": [[1000 * p[f"R{k+1}"], p[f"tau{k+1}"], p[f"n{k+1}"]]
                     for k in range(r["n_arcs"])],
            "R_pol": 1000 * float(sum(p[f"R{k+1}"] for k in range(r["n_arcs"]))),
            "rms_pct": float(r.get("rms_pct", np.nan)),
            "verdict": str(r.get("verdict", "")),
            "f": f.tolist(), "res_pct": (100 * np.abs(Z - Zf) / np.abs(Z)).tolist(),
            "fs": fs.tolist(),
            "Zs": [[float(z.real), float(z.imag)] for z in
                   1000.0 * cp.z_model(np.asarray(v), fs, r["n_arcs"])]}


def fit_drt(f, Z, sigma=None, rs_fixed=None) -> dict:
    """DRT (Tikhonov, NNLS) -> its peaks -> the DRT-informed ECM
    (ecm_drt.fit_drt_ecm: fast/slow arc, AICc, up to ECM_DRT_F_MAX_HZ, Rs
    held at the pipeline's R_ohmic when given). Z in mOhm*cm2."""
    import ecm_drt
    keep = Z.real > 0
    if keep.sum() < 10:
        return {"ok": False, "reason": f"{keep.sum()} points with Z' > 0"}
    drt = ecm_drt.drt_tikhonov(f[keep], Z[keep], lam=1e-3)
    peaks = ecm_drt.drt_peaks(drt)
    r = ecm_drt.fit_drt_ecm(
        f, Z, sigma, peaks=peaks,
        rs_fixed=(float(rs_fixed) if rs_fixed is not None
                  and np.isfinite(rs_fixed) else None),
        f_max_fit_hz=S.ECM_DRT_F_MAX_HZ, fast_element=S.ECM_DRT_FAST)
    out = {"drt_tau": drt["tau"].tolist(), "drt_gamma": drt["gamma"].tolist(),
           "peaks": peaks}
    if not r.get("ok"):
        return {**out, "ok": False, "reason": r.get("reason", "no fit")}
    fs = np.logspace(np.log10(r["freq"].min()), np.log10(r["freq"].max()),
                     S.ECM_SMOOTH_POINTS)
    Zs = ecm_drt.model_curve(r, fs)
    return {**out, "ok": True, "n_arcs": int(r["n_arcs"]),
            "Rs": float(r["Rs"]), "Rs_fixed": bool(r["Rs_fixed"]),
            "L": float(r["L"]), "R_fast": float(r.get("R_fast", np.nan)),
            "tau_fast": float(r.get("tau_fast", np.nan)),
            "n_fast": float(r.get("n_fast", np.nan)),
            "R_slow": float(r.get("R_slow", np.nan)),
            "tau_slow": float(r.get("tau_slow", np.nan)),
            "n_slow": float(r.get("n_slow", np.nan)),
            "R_pol": float(r["R_pol"]), "clean": bool(r["clean"]),
            "fast_element": r.get("fast_element", ""),
            "rms_pct": 100 * float(r["rms_rel_resid"]),
            "flags": list(r["flags"]),
            "f": r["freq"].tolist(), "res_pct": np.asarray(r["res_pct"]).tolist(),
            "fs": fs.tolist(),
            "Zs": [[float(z.real), float(z.imag)] for z in Zs]}


def ecm_fits(run_dir, refit: bool = False, say=print) -> dict:
    """{'Aggregate' | segment: {'f','Z', 'std': fit, 'drt': fit}} for one
    run, saved as RUN_DIR/ecm/ecm_fits.json and read back from there while
    it is newer than the spectra."""
    import standard_eis as se
    run_dir = Path(run_dir)
    out_p = run_dir / "ecm" / "ecm_fits.json"
    sp_p = next((Path(run_dir) / s / "spectra_clean.csv" for s in ("silver", "csv")
                 if (Path(run_dir) / s / "spectra_clean.csv").is_file()), None)
    if sp_p is None:
        return {}
    if (out_p.is_file() and not refit
            and out_p.stat().st_mtime >= sp_p.stat().st_mtime):
        try:
            return json.loads(out_p.read_text())
        except Exception:                                   # noqa: BLE001
            pass
    t0 = time.time()
    df = pd.read_csv(sp_p)
    rs_map = {}
    ss = run_dir / "silver" / "segments_summary.csv"
    if S.ECM_DRT_FIX_RS and ss.is_file():
        t = pd.read_csv(ss)
        rs_map = dict(zip(t["segment"].astype(str), t["R_ohmic_mohm_cm2"]))
    res = {}
    a = aggregate(run_dir)
    spectra_in = []
    if a is not None and len(a):
        spectra_in.append(("Aggregate", a["freq_hz"],
                           a["z_re_mohm_cm2"] + 1j * a["z_im_mohm_cm2"], None))
    for seg in sorted(df["segment"].astype(str).unique(), key=lambda x: int(x)):
        d = df[df["segment"].astype(str) == seg]
        spectra_in.append((seg, d["freq_hz"],
                           d["z_re_mohm_cm2"] + 1j * d["z_im_mohm_cm2"],
                           d["sigma_rel"] if "sigma_rel" in d else None))
    for key, f, Z, s in spectra_in:
        cl = _clean(f, Z, s)
        f_, Z_ = cl[0], cl[1]
        s_ = cl[2] if len(cl) > 2 else None
        if f_.size < 8:
            continue
        entry = {"f": f_.tolist(), "Z": [[float(z.real), float(z.imag)] for z in Z_]}
        try:
            entry["std"] = fit_standard(f_, Z_, s_)
        except Exception as exc:                            # noqa: BLE001
            entry["std"] = {"ok": False, "reason": str(exc)}
        rs = (rs_map.get(key) if key != "Aggregate"
              else se.hf_intercept(f_, Z_)[0] if S.ECM_DRT_FIX_RS else None)
        try:
            entry["drt"] = fit_drt(f_, Z_, s_, rs_fixed=rs)
        except Exception as exc:                            # noqa: BLE001
            entry["drt"] = {"ok": False, "reason": str(exc)}
        res[key] = entry
    out_p.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_p.write_text(json.dumps(res))
        _write_ecm_table(res, run_dir / "ecm" / "ecm_parameters.csv")
    except OSError:
        pass
    n_std = sum(1 for e in res.values() if e["std"].get("ok"))
    n_drt = sum(1 for e in res.values() if e["drt"].get("ok"))
    say(f"    {run_dir.name}: {len(res)} spectra fitted in "
        f"{time.time() - t0:.0f} s (standard ECM {n_std}, DRT-informed {n_drt})")
    return res


def _write_ecm_table(res: dict, path) -> None:
    rows = []
    for k, e in res.items():
        s, d = e.get("std", {}), e.get("drt", {})
        row = {"spectrum": k}
        if s.get("ok"):
            row.update({"std_Rs": s["Rs"], "std_R_pol": s["R_pol"],
                        "std_n_arcs": s["n_arcs"], "std_rms_pct": s["rms_pct"]})
            for i, (R, tau, n) in enumerate(s["arcs"], start=1):
                row.update({f"std_R{i}": R, f"std_tau{i}_ms": 1e3 * tau,
                            f"std_n{i}": n})
        if d.get("ok"):
            row.update({"drt_Rs": d["Rs"], "drt_R_fast": d["R_fast"],
                        "drt_tau_fast_ms": 1e3 * d["tau_fast"],
                        "drt_n_fast": d["n_fast"], "drt_R_slow": d["R_slow"],
                        "drt_tau_slow_ms": 1e3 * d["tau_slow"],
                        "drt_n_slow": d["n_slow"], "drt_R_pol": d["R_pol"],
                        "drt_rms_pct": d["rms_pct"], "drt_clean": d["clean"],
                        "drt_flags": "; ".join(d.get("flags", []))})
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _Z(pairs):
    a = np.asarray(pairs, float).reshape(-1, 2)
    return a[:, 0] + 1j * a[:, 1]


def ecm_figure(entry: dict, title: str):
    """2 x 2: Nyquist, DRT gamma(tau), |Z|(f), residual. Every trace carries
    meta = 'data' | 'std' | 'drt', which the Fit drop-down switches."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.08,
                        vertical_spacing=0.14,
                        subplot_titles=["Nyquist", "DRT  γ(τ)",
                                        "|Z|(f)", "Residual |ΔZ|/|Z|"])
    f = np.asarray(entry["f"], float)
    Z = _Z(entry["Z"])

    def tr(trace, row, col, meta):
        trace.meta = meta
        fig.add_trace(trace, row=row, col=col)

    hover = ("f = %{customdata:.4g} Hz<br>Z' = %{x:.1f}<br>"
             "-Z'' = %{y:.1f} mΩ·cm²<extra>measured</extra>")
    tr(go.Scatter(x=Z.real, y=-Z.imag, mode="markers", name="measured",
                  marker=dict(size=6, color="#7f8c8d"), customdata=f,
                  hovertemplate=hover), 1, 1, "data")
    tr(go.Scatter(x=f, y=np.abs(Z), mode="markers", showlegend=False,
                  name="measured", marker=dict(size=5, color="#7f8c8d")),
       2, 1, "data")
    sub = []
    for key, colour, label in (("std", STD_COLOUR, "Standard ECM"),
                               ("drt", DRT_COLOUR, "DRT-informed ECM")):
        e = entry.get(key) or {}
        if key == "drt" and e.get("drt_tau"):
            tau = np.asarray(e["drt_tau"], float)
            tr(go.Scatter(x=tau, y=e["drt_gamma"], mode="lines",
                          name="DRT γ(τ)", line=dict(color=colour, width=2),
                          hovertemplate="τ = %{x:.3g} s<br>γ = %{y:.3g}"
                                        "<extra>DRT</extra>"), 1, 2, "drt")
            for pk in e.get("peaks", [])[:4]:
                tr(go.Scatter(x=[pk["tau"]], y=[pk["gamma"]], mode="markers",
                              showlegend=False, name="peak",
                              marker=dict(size=9, color=colour,
                                          symbol="triangle-down"),
                              hovertemplate=(f"peak f = {pk['f_peak']:.3g} Hz"
                                             f"<br>R = {pk['R']:.1f} mΩ·cm²"
                                             "<extra></extra>")), 1, 2, "drt")
        if not e.get("ok"):
            sub.append(f"{label}: not fitted ({e.get('reason', '?')})")
            continue
        Zs = _Z(e["Zs"])
        fs = np.asarray(e["fs"], float)
        if key == "std":
            arcs = " · ".join(f"R{i}={R:.1f} τ{i}={1e3*t:.3g} ms n{i}={n:.2f}"
                              for i, (R, t, n) in enumerate(e["arcs"], 1))
            txt = (f"{label}: Rs={e['Rs']:.1f} · {arcs} · R_pol={e['R_pol']:.1f}"
                   f" · rms {e['rms_pct']:.1f} %")
        else:
            txt = (f"{label}: Rs={e['Rs']:.1f}{' (fixed)' if e['Rs_fixed'] else ''}"
                   f" · R_fast={e['R_fast']:.1f} τ={1e3*e['tau_fast']:.3g} ms"
                   f" n={e['n_fast']:.2f}"
                   + (f" · R_slow={e['R_slow']:.1f} τ={1e3*e['tau_slow']:.3g} ms"
                      f" n={e['n_slow']:.2f}" if e["n_arcs"] > 1 else "")
                   + f" · R_pol={e['R_pol']:.1f} · rms {e['rms_pct']:.1f} %"
                   + ("" if e["clean"] else " (not clean)"))
        sub.append(txt)
        tr(go.Scatter(x=Zs.real, y=-Zs.imag, mode="lines", name=label,
                      line=dict(color=colour, width=2.5),
                      hovertemplate=f"{label}<extra></extra>"), 1, 1, key)
        tr(go.Scatter(x=fs, y=np.abs(Zs), mode="lines", showlegend=False,
                      name=label, line=dict(color=colour, width=2.5)),
           2, 1, key)
        tr(go.Scatter(x=e["f"], y=e["res_pct"], mode="lines+markers",
                      showlegend=False, name=label,
                      marker=dict(size=4, color=colour),
                      line=dict(width=1.2, color=colour),
                      hovertemplate=(f"{label}<br>f = %{{x:.4g}} Hz<br>"
                                     "|res| = %{y:.2f} %<extra></extra>")),
           2, 2, key)
    fig.add_hline(y=2, line=dict(color="green", dash="dash", width=1), row=2, col=2)
    fig.add_hline(y=5, line=dict(color="orange", dash="dash", width=1), row=2, col=2)
    fig.update_xaxes(title_text="Z' [mΩ·cm²]", row=1, col=1)
    fig.update_yaxes(title_text="-Z'' [mΩ·cm²]", scaleanchor="x",
                     scaleratio=1, row=1, col=1)
    fig.update_xaxes(title_text="τ [s]", type="log", row=1, col=2)
    fig.update_yaxes(title_text="γ [mΩ·cm²]", row=1, col=2)
    fig.update_xaxes(title_text="f [Hz]", type="log", row=2, col=1)
    fig.update_yaxes(title_text="|Z| [mΩ·cm²]", type="log", row=2, col=1)
    fig.update_xaxes(title_text="f [Hz]", type="log", row=2, col=2)
    fig.update_yaxes(title_text="|ΔZ|/|Z| [%]", rangemode="tozero", row=2, col=2)
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    fig.update_layout(
        title=(f"<b>{title}</b><br><span style='font-size:11px;color:#555'>"
               + "<br>".join(sub) + "</span>"),
        height=860, width=1400, paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(t=150, r=200), hovermode="closest",
        legend=dict(font=dict(size=10), y=0.5, yanchor="middle"))
    return fig


FIT_OPTIONS = ("Standard ECM + DRT-informed", "Standard ECM", "DRT-informed ECM")

_ECM_RENDER = """
  function render(key, P, plot, box) {
    var parts = key.split(SEP);
    var fit = parts[2];
    var it = P.data[parts[0] + SEP + parts[1]];
    if (!it) return false;
    box.innerHTML = "";
    var show = {"data": true,
                "std": fit !== "DRT-informed ECM",
                "drt": fit !== "Standard ECM"};
    var data = it.fig.data.map(function(t) {
      var c = Object.assign({}, t);
      c.visible = show[t.meta] ? true : false;
      return c;
    });
    Plotly.react(plot, data, it.fig.layout, {responsive: true});
    return true;
  }
"""


def ecm_viewer(figs: dict, title: str = "") -> str:
    """{(condition, spectrum): figure} -> Condition / Spectrum / Fit viewer.
    The Fit drop-down only switches traces, so the payload holds each figure
    once."""
    import viewers
    items = {f"{c}{viewers.SEP}{s}": viewers._payload(f)
             for (c, s), f in figs.items() if f is not None}
    if not items:
        return "<p><i>no ECM fits to show</i></p>"
    conds = list(dict.fromkeys(c for c, _s in figs))
    specs = list(dict.fromkeys(s for _c, s in figs))
    return viewers._shell(["Condition", "Spectrum", "Fit"],
                          [conds, specs, list(FIT_OPTIONS)],
                          viewers._pack({"data": items}), _ECM_RENDER,
                          title=title)


def ecm_png(entry: dict, path, title: str) -> Path | None:
    """Static Nyquist + residual of both fits (the thesis figure)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = np.asarray(entry["f"], float)
    Z = _Z(entry["Z"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    ax[0].plot(Z.real, -Z.imag, "o", ms=4, color="#7f8c8d", label="measured")
    ax[1].loglog(f, np.abs(Z), "o", ms=3, color="#7f8c8d")
    for key, colour, label in (("std", STD_COLOUR, "Standard ECM"),
                               ("drt", DRT_COLOUR, "DRT-informed ECM")):
        e = entry.get(key) or {}
        if not e.get("ok"):
            continue
        Zs = _Z(e["Zs"])
        ax[0].plot(Zs.real, -Zs.imag, "-", color=colour, lw=2,
                   label=f"{label} (R_pol {e['R_pol']:.0f})")
        ax[1].loglog(e["fs"], np.abs(Zs), "-", color=colour, lw=2)
        ax[2].semilogx(e["f"], e["res_pct"], ".-", color=colour, lw=1)
    ax[0].set_xlabel("Z' [mΩ·cm²]"); ax[0].set_ylabel("−Z'' [mΩ·cm²]")
    ax[0].set_aspect("equal", adjustable="datalim"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("f [Hz]"); ax[1].set_ylabel("|Z| [mΩ·cm²]")
    ax[2].set_xlabel("f [Hz]"); ax[2].set_ylabel("|ΔZ|/|Z| [%]")
    ax[2].axhline(2, color="green", ls="--", lw=0.8)
    ax[2].axhline(5, color="orange", ls="--", lw=0.8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
