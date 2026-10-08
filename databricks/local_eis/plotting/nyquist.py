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

    import nyquist
    fig = nyquist.figure(spectra_df, title="2612030 / 450A")      # plotly
    nyquist.save_png(spectra_df, "nyquist.png", title="...")       # same plot
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

COLOURS = ("flow", "segment")


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
           note: str = "", height: int = 550, width: int = 1500):
    """Nyquist + |Z| + phase of every segment (one subplot each).

    df       silver/spectra_clean.csv as a DataFrame (segment, freq_hz,
             z_re_mohm_cm2, z_im_mohm_cm2)
    rebuilt  silver/spectra_reconstructed.csv (dotted), optional
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
                f"<b>Seg {s}</b><br>f = %{{customdata:.2f}} Hz<br>"
                "Z' = %{x:.1f} mΩ·cm²<br>-Z'' = %{y:.1f} mΩ·cm²<extra></extra>")),
            row=1, col=1)
        Z = zr + 1j * zi
        fig.add_trace(go.Scatter(x=f, y=np.abs(Z), mode="lines", name=name,
                                 legendgroup=name, showlegend=False,
                                 line=dict(width=1, color=c)), row=1, col=2)
        fig.add_trace(go.Scatter(x=f, y=np.degrees(np.angle(Z)), mode="lines",
                                 name=name, legendgroup=name, showlegend=False,
                                 line=dict(width=1, color=c)), row=1, col=3)
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
                    f"<b>Seg {s}</b> — REBUILT {don}<br>f = %{{customdata:.2f}} Hz"
                    "<br>Z' = %{x:.1f}<br>-Z'' = %{y:.1f}<extra></extra>")),
                row=1, col=1)
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
               + (f" + {n_rec} rebuilt" if n_rec else "") + " segments)</b>"
               + f"<br><span style='font-size:11px;color:#888'>{sub}"
               + (f" — {note}" if note else "") + "</span>"),
        height=height, width=width, paper_bgcolor="white",
        plot_bgcolor="white", margin=dict(t=80, b=50, r=250),
        hovermode="closest", legend=dict(font=dict(size=9), y=0.5,
                                         yanchor="middle"))
    return fig


def save_png(df, path, title: str = "", colour: str = "flow", rebuilt=None,
             dpi: int = 160) -> Path:
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
             out_dir=None) -> dict:
    """nyquist.png + nyquist.html for one finished run (FAMOS or CSV)."""
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
    png = save_png(df, out / "nyquist.png", title=title, colour=colour,
                   rebuilt=rec)
    html = out / "nyquist.html"
    figure(df, title=title, colour=colour, rebuilt=rec).write_html(
        str(html), include_plotlyjs="cdn")
    return {"png": str(png), "html": str(html)}
