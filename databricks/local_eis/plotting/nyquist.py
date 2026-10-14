#!/usr/bin/env python3
"""
nyquist.py
==========
The per-segment Nyquist / Bode plot, built ONCE and used everywhere: the
runner shows it (interactive, Plotly) and the run saves the same plot
(nyquist.html, and a PNG drawn the same way), so what is saved is what was
looked at.

COLOUR = POSITION ALONG THE AIR PATH
------------------------------------
`colour="flow"` (the default) colours every segment by where it sits between
the cathode inlet (0, dark blue) and the cathode outlet (1, dark red) on the
jet scale, with a colour bar -- the way the bench's MATLAB plots do. A
gradient along the channel (drying at the inlet, flooding towards the
outlet) then shows as a colour trend through the arcs instead of a tangle of
72 unrelated colours. `colour="segment"` gives the old hue per segment.

The inlet end comes from plate_style (config.COOLANT_INLET_END / the port
layout), so a mirrored view or a different plate does not flip the meaning.

THE GAMRY IS DRAWN AS RECORDED
------------------------------
`gamry=` adds the condition's whole-cell Gamry sweep as a thick black line
in every panel (Nyquist, |Z|, phase), so each segment can be held against it.
It is every point of the .DTA at the frequency the file gives, in the file's
order of frequency -- no band limit, no interpolation onto the local grid,
no inductance removed. `gamry_frame()` reads it; `comparison_figure()` draws
the aggregate of the segments against it, and nothing else.

    import nyquist
    g = nyquist.gamry_frame("V26_092_HFR_104_CurrVal_450.dta")   # exact points
    fig = nyquist.figure(spectra_df, title="2612030 / 450A", gamry=g)
    nyquist.save_png(spectra_df, "nyquist.png", title="...", gamry=g)
    fig = nyquist.comparison_figure(aggregate_df, g, title="450A")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

COLOURS = ("flow", "segment")

#: the whole cell's active area: the Gamry's ohm -> mohm.cm2
A_CELL_CM2 = 304.92
GAMRY_COLOUR = "black"
AGGREGATE_COLOUR = "#c0392b"
#: frequencies in hover labels: 5 significant digits, so 0.4783 Hz reads
#: 0.47832 and 2390.6 Hz reads 2390.6 -- the value in the file, not a
#: rounded one
F_FMT = "%{customdata:.5g}"


def gamry_frame(src, area_cm2: float = A_CELL_CM2):
    """The whole-cell Gamry sweep as a DataFrame (freq_hz, z_re_mohm_cm2,
    z_im_mohm_cm2), every point exactly as in the file, ascending f.

    src: a .DTA path, a gamry_compare.CellSweep, a run folder (its
    gamry_reference_full.csv) or a DataFrame already in that form."""
    import pandas as pd
    if src is None:
        return None
    if isinstance(src, pd.DataFrame):
        df = src
    elif hasattr(src, "Z_ohm"):
        z = np.asarray(src.Z_ohm, complex) * area_cm2 * 1e3
        df = pd.DataFrame({"freq_hz": np.asarray(src.freq, float),
                           "z_re_mohm_cm2": z.real, "z_im_mohm_cm2": z.imag})
    else:
        p = Path(src)
        if p.is_dir():
            p = p / "gamry_reference_full.csv"
            if not p.is_file():
                return None
        if p.suffix.lower() == ".csv":
            d = pd.read_csv(p)
            df = pd.DataFrame({"freq_hz": d["freq_hz"],
                               "z_re_mohm_cm2": d["z_re_ref_mohm_cm2"],
                               "z_im_mohm_cm2": d["z_im_ref_mohm_cm2"]})
        else:
            import gamry_compare
            return gamry_frame(gamry_compare.read_cell_sweep(p), area_cm2)
    df = df[["freq_hz", "z_re_mohm_cm2", "z_im_mohm_cm2"]].astype(float)
    df = df[np.isfinite(df).all(axis=1)]
    return df.sort_values("freq_hz").reset_index(drop=True)


def find_gamry(gamry_dir, condition, order_id=None, version=None,
               area_cm2: float = A_CELL_CM2):
    """gamry_frame of the condition's sweep in `gamry_dir` (the same lookup
    the pipeline uses: this order's build only), or None."""
    if not gamry_dir or not condition:
        return None
    try:
        import gamry_compare
        hits = [s for s in gamry_compare.find_cell_sweeps(
                    gamry_dir, order_id=order_id or None,
                    version=version or None)
                if s.condition.upper() == str(condition).upper()]
    except Exception:                                       # noqa: BLE001
        return None
    return gamry_frame(hits[0], area_cm2) if hits else None


def _add_gamry(fig, g, name: str, width: float = 3.5) -> None:
    """The Gamry, black, in all three panels; one legend entry."""
    import plotly.graph_objects as go
    f = g["freq_hz"].to_numpy(float)
    Z = g["z_re_mohm_cm2"].to_numpy(float) + 1j * g["z_im_mohm_cm2"].to_numpy(float)
    hover = (f"<b>{name}</b><br>f = {F_FMT} Hz<br>Z' = %{{x:.2f}}<br>"
             "-Z'' = %{y:.2f}<extra></extra>")
    style = dict(mode="lines+markers", legendgroup="gamry",
                 line=dict(width=width, color=GAMRY_COLOUR),
                 marker=dict(size=6, color=GAMRY_COLOUR, symbol="diamond"))
    fig.add_trace(go.Scatter(x=Z.real, y=-Z.imag, name=name, customdata=f,
                             hovertemplate=hover, **style), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=f, y=np.abs(Z), name=name, showlegend=False, customdata=f,
        hovertemplate=f"<b>{name}</b><br>f = {F_FMT} Hz<br>|Z| = %{{y:.2f}}"
                      "<extra></extra>", **style), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=f, y=np.degrees(np.angle(Z)), name=name, showlegend=False,
        customdata=f, hovertemplate=f"<b>{name}</b><br>f = {F_FMT} Hz<br>"
                                    "phase = %{y:.2f}°<extra></extra>",
        **style), row=1, col=3)


def air_inlet_x() -> float:
    """x (plate mm) of the cathode (air) inlet end."""
    import plate_style as style
    import r2d2_geometry as geom
    for end, streams in style.end_streams().items():
        if any(lab.upper().startswith("AIR IN") for lab, _r, _c in streams):
            return 0.0 if end == "x0" else float(geom.PLATE_W_MM)
    return float(geom.PLATE_W_MM)


def flow_position(segments) -> dict[str, float]:
    """0 at the cathode inlet .. 1 at the cathode outlet, per segment."""
    import r2d2_geometry as geom
    W, x_in = float(geom.PLATE_W_MM), air_inlet_x()
    out = {}
    for s in segments:
        g = geom.SEGMENTS.get(str(s))
        if g is not None:
            out[str(s)] = float(np.clip(abs(g.cx_mm - x_in) / W, 0, 1))
    return out


def _jet(t: float) -> tuple[float, float, float]:
    import plate_style as style
    r, g, b, _a = style.mpl_cmap("jet")(float(t))
    return r, g, b


def colours(segments, colour: str = "flow") -> dict[str, str]:
    segs = [str(s) for s in segments]
    if colour == "flow":
        pos = flow_position(segs)
        return {s: "rgb({:.0f},{:.0f},{:.0f})".format(
            *(255 * c for c in _jet(pos.get(s, 0.5)))) for s in segs}
    n = max(1, len(segs))
    return {s: f"hsl({int(i * 360 / n)}, 70%, 50%)" for i, s in enumerate(segs)}


def _seg_key(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        return 10 ** 6


def figure(df, title: str = "", colour: str = "flow", rebuilt=None,
           note: str = "", height: int = 550, width: int = 1500,
           gamry=None, gamry_name: str = "Gamry whole cell"):
    """Nyquist + |Z| + phase of every segment (one subplot each).

    df       silver/spectra_clean.csv as a DataFrame (segment, freq_hz,
             z_re_mohm_cm2, z_im_mohm_cm2)
    rebuilt  silver/spectra_reconstructed.csv (dotted), optional
    gamry    the condition's whole-cell sweep (gamry_frame), drawn black in
             all three panels with every point of the file, optional
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plate_style as style

    segs = sorted(df["segment"].astype(str).unique(), key=_seg_key)
    col = colours(segs, colour)
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.06,
                        subplot_titles=["Nyquist", "|Z|(f) Bode", "Phase(f)"])
    for s in segs:
        d = df[df["segment"].astype(str) == s].sort_values("freq_hz")
        f = d["freq_hz"].to_numpy(float)
        zr = d["z_re_mohm_cm2"].to_numpy(float)
        zi = d["z_im_mohm_cm2"].to_numpy(float)
        ok = np.isfinite(f) & np.isfinite(zr) & np.isfinite(zi)
        if ok.sum() < 3:
            continue
        f, zr, zi = f[ok], zr[ok], zi[ok]
        c, name = col[s], f"Seg {s}"
        fig.add_trace(go.Scatter(
            x=zr, y=-zi, mode="markers+lines", name=name, legendgroup=name,
            marker=dict(size=4, color=c), line=dict(width=1, color=c),
            customdata=f, hovertemplate=(
                f"<b>Seg {s}</b><br>f = {F_FMT} Hz<br>"
                "Z' = %{x:.1f} mΩ·cm²<br>-Z'' = %{y:.1f} mΩ·cm²<extra></extra>")),
            row=1, col=1)
        Z = zr + 1j * zi
        fig.add_trace(go.Scatter(
            x=f, y=np.abs(Z), mode="lines", name=name, legendgroup=name,
            showlegend=False, line=dict(width=1, color=c), customdata=f,
            hovertemplate=(f"<b>Seg {s}</b><br>f = {F_FMT} Hz<br>"
                           "|Z| = %{y:.1f}<extra></extra>")), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=f, y=np.degrees(np.angle(Z)), mode="lines", name=name,
            legendgroup=name, showlegend=False, line=dict(width=1, color=c),
            customdata=f, hovertemplate=(f"<b>Seg {s}</b><br>f = {F_FMT} Hz<br>"
                                         "phase = %{y:.1f}°<extra></extra>")),
            row=1, col=3)
    n_rec = 0
    if rebuilt is not None and len(rebuilt):
        rs = sorted(rebuilt["segment"].astype(str).unique(), key=_seg_key)
        n_rec = len(rs)
        for j, s in enumerate(rs):
            d = rebuilt[rebuilt["segment"].astype(str) == s].sort_values("freq_hz")
            zr, zi = d["z_re_mohm_cm2"].to_numpy(float), d["z_im_mohm_cm2"].to_numpy(float)
            f = d["freq_hz"].to_numpy(float)
            ok = np.isfinite(zr) & np.isfinite(zi)
            if ok.sum() < 3:
                continue
            don = str(d["donors"].iloc[0]) if "donors" in d else ""
            fig.add_trace(go.Scatter(
                x=zr[ok], y=-zi[ok], mode="lines", legendgroup="rebuilt",
                name="rebuilt from neighbours" if j == 0 else f"Seg {s} (rebuilt)",
                showlegend=(j == 0), line=dict(width=1.6, color="#444", dash="dot"),
                customdata=f[ok], hovertemplate=(
                    f"<b>Seg {s}</b> — REBUILT {don}<br>f = {F_FMT} Hz"
                    "<br>Z' = %{x:.1f}<br>-Z'' = %{y:.1f}<extra></extra>")),
                row=1, col=1)
    g = gamry_frame(gamry)
    if g is not None and len(g):
        # on top of the segments, so it can be read against any one of them
        _add_gamry(fig, g, gamry_name)
    if colour == "flow":
        # the colour bar: position along the air path
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", showlegend=False,
            hoverinfo="skip", marker=dict(
                colorscale=style.plotly_colorscale(), cmin=0, cmax=1,
                color=[0], showscale=True, colorbar=dict(
                    title=dict(text="air path", side="right"),
                    tickvals=[0, 1], ticktext=["cathode<br>inlet",
                                               "cathode<br>outlet"],
                    len=0.75, thickness=14, x=1.0, xanchor="left"))),
            row=1, col=1)
    fig.update_xaxes(title_text="Z' [mΩ·cm²]", row=1, col=1)
    fig.update_yaxes(title_text="-Z'' [mΩ·cm²]", row=1, col=1)
    fig.update_xaxes(title_text="f [Hz]", type="log", row=1, col=2)
    fig.update_yaxes(title_text="|Z| [mΩ·cm²]", type="log", row=1, col=2)
    fig.update_xaxes(title_text="f [Hz]", type="log", row=1, col=3)
    fig.update_yaxes(title_text="Phase [°]", row=1, col=3)
    sub = ("colour: position along the air path, cathode inlet → outlet"
           if colour == "flow" else "colour: one per segment")
    fig.update_layout(
        title=(f"<b>{title} ({len(segs)} measured"
               + (f" + {n_rec} rebuilt" if n_rec else "") + " segments"
               + (f" + {gamry_name}" if g is not None and len(g) else "")
               + ")</b>"
               + f"<br><span style='font-size:11px;color:#888'>{sub}"
               + (f" — {note}" if note else "") + "</span>"),
        height=height, width=width, paper_bgcolor="white",
        plot_bgcolor="white", margin=dict(t=80, b=50, r=250),
        hovermode="closest", legend=dict(font=dict(size=9), y=0.5,
                                         yanchor="middle"))
    return fig


def save_png(df, path, title: str = "", colour: str = "flow", rebuilt=None,
             dpi: int = 160, gamry=None,
             gamry_name: str = "Gamry whole cell") -> Path:
    """The Nyquist panel of `figure`, as a PNG: same data, same colours."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    import plate_style as style

    segs = sorted(df["segment"].astype(str).unique(), key=_seg_key)
    pos = flow_position(segs)
    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    n = max(1, len(segs))
    for i, s in enumerate(segs):
        d = df[df["segment"].astype(str) == s].sort_values("freq_hz")
        zr = d["z_re_mohm_cm2"].to_numpy(float)
        zi = d["z_im_mohm_cm2"].to_numpy(float)
        ok = np.isfinite(zr) & np.isfinite(zi)
        if ok.sum() < 3:
            continue
        c = (_jet(pos.get(s, 0.5)) if colour == "flow"
             else plt.get_cmap("hsv")(i / n))
        ax.plot(zr[ok], -zi[ok], "-", lw=0.9, color=c, alpha=0.9)
    if rebuilt is not None and len(rebuilt):
        for s in rebuilt["segment"].astype(str).unique():
            d = rebuilt[rebuilt["segment"].astype(str) == s].sort_values("freq_hz")
            ax.plot(d["z_re_mohm_cm2"], -d["z_im_mohm_cm2"], ":", lw=1.2,
                    color="0.3")
    g = gamry_frame(gamry)
    if g is not None and len(g):
        ax.plot(g["z_re_mohm_cm2"], -g["z_im_mohm_cm2"], "D-", ms=3.5,
                lw=2.4, color=GAMRY_COLOUR, label=gamry_name, zorder=5)
        ax.legend(loc="upper left", fontsize=9)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel("Re(Z) in mΩ·cm²")
    ax.set_ylabel("-Im(Z) in mΩ·cm²")
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25)
    if colour == "flow":
        sm = ScalarMappable(cmap=style.mpl_cmap("jet"), norm=Normalize(0, 1))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, ticks=[0, 1])
        cb.ax.set_yticklabels(["cathode\ninlet", "cathode\noutlet"])
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def save_run(run_dir, title: str = "", colour: str = "flow",
             out_dir=None, gamry=None) -> dict:
    """nyquist.png + nyquist.html for one finished run (FAMOS or CSV).

    gamry: the whole-cell sweep to draw with the segments; by default the
    run's own gamry_reference_full.csv, when the run has one."""
    import pandas as pd
    run = Path(run_dir)
    sp = run / "silver" / "spectra_clean.csv"
    if not sp.is_file():
        return {}
    df = pd.read_csv(sp)
    rp = run / "silver" / "spectra_reconstructed.csv"
    rec = pd.read_csv(rp) if rp.is_file() else None
    out = Path(out_dir) if out_dir else run / "gold"
    out.mkdir(parents=True, exist_ok=True)
    g = gamry_frame(gamry if gamry is not None else run)
    png = save_png(df, out / "nyquist.png", title=title, colour=colour,
                   rebuilt=rec, gamry=g)
    html = out / "nyquist.html"
    figure(df, title=title, colour=colour, rebuilt=rec, gamry=g).write_html(
        str(html), include_plotlyjs="cdn")
    return {"png": str(png), "html": str(html)}


def comparison_figure(aggregate, gamry, title: str = "",
                      aggregate_name: str = "Aggregate of the segments",
                      gamry_name: str = "Gamry whole cell", note: str = "",
                      height: int = 550, width: int = 1500):
    """The aggregate of the segments against the whole-cell Gamry: Nyquist,
    |Z| and phase, those two curves and nothing else.

    aggregate  silver/cell_aggregate.csv (or the standard pipeline's
               aggregate.csv): freq_hz, z_re_mohm_cm2, z_im_mohm_cm2
    gamry      gamry_frame() input -- every point of the sweep, as recorded
    Neither curve is scaled, cut to a band or corrected here.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.06,
                        subplot_titles=["Nyquist", "|Z|(f) Bode", "Phase(f)"])
    n_agg = 0
    if aggregate is not None and len(aggregate):
        a = aggregate.sort_values("freq_hz")
        f = a["freq_hz"].to_numpy(float)
        Z = (a["z_re_mohm_cm2"].to_numpy(float)
             + 1j * a["z_im_mohm_cm2"].to_numpy(float))
        ok = np.isfinite(f) & np.isfinite(Z)
        f, Z = f[ok], Z[ok]
        n_agg = int(ok.sum())
        style = dict(mode="lines+markers", legendgroup="agg",
                     line=dict(width=3, color=AGGREGATE_COLOUR),
                     marker=dict(size=5, color=AGGREGATE_COLOUR))
        fig.add_trace(go.Scatter(
            x=Z.real, y=-Z.imag, name=aggregate_name, customdata=f,
            hovertemplate=(f"<b>{aggregate_name}</b><br>f = {F_FMT} Hz<br>"
                           "Z' = %{x:.2f}<br>-Z'' = %{y:.2f}<extra></extra>"),
            **style), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=f, y=np.abs(Z), showlegend=False, customdata=f,
            hovertemplate=(f"<b>{aggregate_name}</b><br>f = {F_FMT} Hz<br>"
                           "|Z| = %{y:.2f}<extra></extra>"), **style),
            row=1, col=2)
        fig.add_trace(go.Scatter(
            x=f, y=np.degrees(np.angle(Z)), showlegend=False, customdata=f,
            hovertemplate=(f"<b>{aggregate_name}</b><br>f = {F_FMT} Hz<br>"
                           "phase = %{y:.2f}°<extra></extra>"), **style),
            row=1, col=3)
    g = gamry_frame(gamry)
    if g is not None and len(g):
        _add_gamry(fig, g, gamry_name)
    fig.update_xaxes(title_text="Z' [mΩ·cm²]", row=1, col=1)
    fig.update_yaxes(title_text="-Z'' [mΩ·cm²]", row=1, col=1)
    fig.update_xaxes(title_text="f [Hz]", type="log", row=1, col=2)
    fig.update_yaxes(title_text="|Z| [mΩ·cm²]", type="log", row=1, col=2)
    fig.update_xaxes(title_text="f [Hz]", type="log", row=1, col=3)
    fig.update_yaxes(title_text="Phase [°]", row=1, col=3)
    sub = (f"aggregate: {n_agg} frequencies · Gamry: "
           f"{0 if g is None else len(g)} points, every one in the file")
    fig.update_layout(
        title=(f"<b>{title}</b><br><span style='font-size:11px;color:#888'>"
               f"{sub}" + (f" — {note}" if note else "") + "</span>"),
        height=height, width=width, paper_bgcolor="white",
        plot_bgcolor="white", margin=dict(t=80, b=50, r=250),
        hovermode="closest", legend=dict(font=dict(size=10), y=0.5,
                                         yanchor="middle"))
    return fig


def comparison_png(aggregate, gamry, path, title: str = "",
                   dpi: int = 150) -> Path:
    """`comparison_figure` as a PNG (Nyquist, |Z|, phase in one row)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    g = gamry_frame(gamry)
    if aggregate is not None and len(aggregate):
        a = aggregate.sort_values("freq_hz")
        f = a["freq_hz"].to_numpy(float)
        Z = (a["z_re_mohm_cm2"].to_numpy(float)
             + 1j * a["z_im_mohm_cm2"].to_numpy(float))
        kw = dict(color=AGGREGATE_COLOUR, lw=2, ms=4,
                  label="Aggregate of the segments")
        ax[0].plot(Z.real, -Z.imag, "o-", **kw)
        ax[1].loglog(f, np.abs(Z), "o-", **kw)
        ax[2].semilogx(f, np.degrees(np.angle(Z)), "o-", **kw)
    if g is not None and len(g):
        f = g["freq_hz"].to_numpy(float)
        Z = g["z_re_mohm_cm2"].to_numpy(float) + 1j * g["z_im_mohm_cm2"].to_numpy(float)
        kw = dict(color=GAMRY_COLOUR, lw=2.4, ms=4, label="Gamry whole cell")
        ax[0].plot(Z.real, -Z.imag, "D-", **kw)
        ax[1].loglog(f, np.abs(Z), "D-", **kw)
        ax[2].semilogx(f, np.degrees(np.angle(Z)), "D-", **kw)
    ax[0].axhline(0, color="0.6", lw=0.6)
    ax[0].set(xlabel="Z' [mΩ·cm²]", ylabel="-Z'' [mΩ·cm²]", title="Nyquist")
    ax[0].set_aspect("equal", adjustable="datalim")
    ax[1].set(xlabel="f [Hz]", ylabel="|Z| [mΩ·cm²]", title="|Z|")
    ax[2].set(xlabel="f [Hz]", ylabel="phase [°]", title="Phase")
    for a_ in ax:
        a_.grid(alpha=0.25)
    ax[0].legend(fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
