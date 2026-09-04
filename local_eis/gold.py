#!/usr/bin/env python3
"""
gold.py  --  LAYER 3 of 3:  the plate as a field
================================================

Gold answers the question the whole rig exists to answer: WHERE on the plate
is something wrong, and what is it.

THREE THINGS HAPPEN HERE
------------------------
1. PROCESS SPLIT.  One resistance map is ambiguous -- a hot segment could be
   dry, flooded or starved, and those want opposite corrections.  Splitting
   the fitted relaxation-time distribution into buckets turns one map into
   three that mean different things: ohmic (membrane and contact), charge
   transfer, and mass transport.  Drying raises the first, flooding the
   third.

2. EVERY SEGMENT GETS A VALUE.  A segment with no calibration row, a dead
   channel or a failed fit used to vanish from the map.  A hole in a heat map
   is not neutral: the eye reads it as a cold spot, and an interpolated
   colour field will happily paint straight through it.  Here the plate is
   modelled as a smooth field and the missing segments are filled from the
   posterior -- then drawn with hatching and a wider error bar, so an
   inferred value can never be mistaken for a measured one.

3. FAULT LABELS.  Ratios to the plate median, not absolute thresholds, so the
   rules survive a change of operating point.

WHY A FIELD MODEL IS THE RIGHT THING, NOT A CONVENIENCE
------------------------------------------------------
Neighbouring segments share gas composition, membrane hydration and clamping
pressure, so their properties are genuinely correlated -- which is why the
inlet-to-outlet gradient in the legacy maps was believable while the 9x
segment-to-segment scatter was not.  A Gaussian process over the segment
centroids encodes that correlation explicitly, with a LONGER correlation
length along the flow than across it, because that is how the physics is
laid out.  A noisy segment is then pulled towards its neighbours in
proportion to its own uncertainty, and a clean one is left alone.

The honesty constraint: the model is refused if too much of the plate would
be guess rather than measurement (`max_inferred_fraction`), and every
inferred segment is flagged in the CSV as well as hatched on the map.

REFERENCES
----------
A. Hakenjos, C. Hebling, J. Power Sources 145 (2005) 307  -- segmented plate,
    HF resistance as a membrane-hydration map
I. A. Schneider et al., Electrochem. Commun. 7 (2005) 1393 -- locally resolved
    EIS with neutron radiography; flooded vs dry signatures
B. Guo et al., IEEE Trans. Instrum. Meas. 74 (2025) 3543212 -- impedance-field
    view and image-based fault classification
C. E. Rasmussen, C. K. I. Williams, Gaussian Processes for Machine Learning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:                      # package layout: core/ holds the science modules
    import core           # noqa: F401
except ImportError:       # flat layout (Databricks, notebooks): already there
    pass
import r2d2_geometry as geom

import utils
from config import (Config, DEFAULT, COLORS, SEGMENT_CLASS_STYLE, PARAM_META,
                    FAULT_RULES, PLATE_W_CM, PLATE_H_CM, PLATE_W_MM,
                    PLATE_H_MM, FLOW_CHANNEL_Y_MM, N_COLS, N_ROWS,
                    PAD_W_MM, PAD_H_MM, KNOWN_BAD_SEGMENTS)
from silver import SilverRun, SilverSpectrum


# ===========================================================================
# 1. Containers
# ===========================================================================


@dataclass
class SegmentRecord:
    """One segment on the finished plate, measured or inferred."""

    segment: str
    cx_mm: float
    cy_mm: float
    area_cm2: float
    cls: str                       # "measured" | "inferred" | "bad"
    tier: str                      # A/B/C from silver, or D when inferred
    values: dict[str, float] = field(default_factory=dict)
    sd: dict[str, float] = field(default_factory=dict)
    fault: str = ""
    flags: list[str] = field(default_factory=list)


@dataclass
class GoldRun:
    records: dict[str, SegmentRecord]
    fields: dict[str, dict]        # parameter -> {"value": {...}, "sd": {...}}
    stats: dict
    cell_freq: np.ndarray
    Z_cell: np.ndarray

    def measured(self) -> list[str]:
        return [s for s, r in self.records.items() if r.cls == "measured"]

    def inferred(self) -> list[str]:
        return [s for s, r in self.records.items() if r.cls != "measured"]


# ===========================================================================
# 2. Process split from the relaxation-time distribution
# ===========================================================================


def split_processes(sp: SilverSpectrum, cfg: Config) -> dict[str, float]:
    """Bucket the DRT by time constant into ohmic / charge-transfer / transport.

    The boundaries live in the config, not here, because a different MEA
    moves them.  With no DRT available the polarisation resistance is
    reported whole rather than split into numbers that were not measured.
    """
    out = {"R_ohmic": sp.R_ohmic, "R_pol": sp.R_pol,
           "R_ct": np.nan, "R_mt": np.nan}
    if sp.gamma.size == 0 or sp.tau_grid.size != sp.gamma.size:
        return out
    tau, g = sp.tau_grid, sp.gamma
    fast = tau < cfg.tau_split_ohmic_s
    mid = (tau >= cfg.tau_split_ohmic_s) & (tau < cfg.tau_split_kinetic_s)
    slow = tau >= cfg.tau_split_kinetic_s

    # R_ohmic is NOT modified here.  An earlier version folded the fastest
    # DRT bucket into it, on the argument that a relaxation faster than
    # 0.1 ms is indistinguishable from series resistance in this band.  The
    # argument is sound but the arithmetic is not: those bucket elements sit
    # exactly where the design matrix is degenerate with the constant column,
    # so their individual values are unconstrained even when their sum with
    # R_inf is well determined.  Adding them produced segment values spanning
    # eleven orders of magnitude on the end-to-end test.
    #
    # R_ohmic therefore stays as silver measured it, from the data at the top
    # of the band, and the fast bucket is reported beside it as a diagnostic.
    out["R_hf_extra"] = float(max(g[fast].sum(), 0.0))
    out["R_ct"] = float(max(g[mid].sum(), 0.0))
    out["R_mt"] = float(max(g[slow].sum(), 0.0))
    return out


def collect_parameters(sr: SilverRun, cfg: Config
                       ) -> tuple[dict[str, dict[str, float]],
                                  dict[str, dict[str, float]]]:
    """parameter -> {segment: value}, and the matching uncertainties."""
    vals: dict[str, dict[str, float]] = {}
    sds: dict[str, dict[str, float]] = {}

    def put(p, s, v, sd=np.nan):
        vals.setdefault(p, {})[s] = float(v)
        sds.setdefault(p, {})[s] = float(sd)

    for s, sp in sr.spectra.items():
        proc = split_processes(sp, cfg)
        # A non-positive intercept is a failed fit, not a small measurement.
        # Writing it through would put a negative resistance on the map and
        # into the plate statistics.  Handing it to the field as MISSING is
        # both more honest and more useful: the segment still gets a number,
        # it comes from its neighbours, and the flag from silver plus the
        # hatched outline say where it came from.
        r_oh = proc["R_ohmic"]
        put("R_ohmic", s, r_oh if (np.isfinite(r_oh) and r_oh > 0) else np.nan,
            sp.R_ohmic_sd)
        put("R_hf_extra", s, proc.get("R_hf_extra", np.nan))
        put("R_ct", s, proc["R_ct"])
        put("R_mt", s, proc["R_mt"])
        put("R_pol", s, sp.R_pol)
        put("j_dc", s, sp.j_dc)
        put("T_degC", s, sp.T_degC)
        put("tau_peak", s, sp.tau_peak)
        put("sigma_rel", s, sp.R_ohmic_sd / abs(sp.R_ohmic)
            if sp.R_ohmic else np.nan)
        put("hf_closure", s, sp.hf_closure)
        # |Z| and phase at the reference frequency
        f = sp.freq
        if len(f):
            i = int(np.argmin(np.abs(f - 100.0)))
            if abs(f[i] - 100.0) <= 0.5 * 100.0:
                put("Z_mag_100Hz", s, abs(sp.Z_model[i]))
                put("phase_100Hz", s, np.degrees(np.angle(sp.Z_model[i])))
    return vals, sds


# ===========================================================================
# 3. The spatial field
# ===========================================================================


def _kernel(X1: np.ndarray, X2: np.ndarray, lx: float, ly: float,
            kind: str = "matern52") -> np.ndarray:
    """Anisotropic kernel on plate coordinates, in mm.

    Anisotropy is not a tuning knob: the gas travels along x, so two segments
    a long way apart down the channel are far less alike than two the same
    distance apart across it.  `lx > ly` states that.
    """
    dx = (X1[:, 0][:, None] - X2[:, 0][None, :]) / lx
    dy = (X1[:, 1][:, None] - X2[:, 1][None, :]) / ly
    r = np.sqrt(dx ** 2 + dy ** 2)
    if kind == "se":
        return np.exp(-0.5 * r ** 2)
    s5 = np.sqrt(5.0) * r
    return (1.0 + s5 + 5.0 / 3.0 * r ** 2) * np.exp(-s5)


def fit_field(values: dict[str, float], sds: dict[str, float],
              cfg: Config) -> tuple[dict[str, float], dict[str, float], dict]:
    """Gaussian-process regression over the segment centroids.

    Returns (mean, sd) for EVERY segment, measured or not.

    Measured segments come back close to their own value when they are
    precise and pulled towards their neighbours when they are not; that is
    the whole point of using the observation variance rather than a single
    smoothing parameter.  Unmeasured segments come back as pure prediction,
    with a standard deviation that grows with distance from the nearest
    measurement -- so a segment in the middle of a measured region is
    inferred confidently, and one in a gap is not.
    """
    obs = [s for s, v in values.items() if np.isfinite(v)]
    if len(obs) < 4:
        return dict(values), dict(sds), {"ok": False, "note": "too few points"}

    cen = geom.centroids()
    Xo = np.array([cen[s] for s in obs], float)
    yo = np.array([values[s] for s in obs], float)

    mu = float(np.mean(yo))
    amp = float(np.std(yo)) or 1.0
    y = (yo - mu) / amp

    # observation noise: the segment's own uncertainty, floored by a nugget
    n_obs = np.array([sds.get(s, np.nan) for s in obs], float) / amp
    n_obs = np.where(np.isfinite(n_obs) & (n_obs > 0), n_obs, cfg.spatial_nugget)
    n_obs = np.clip(n_obs, cfg.spatial_nugget, 5.0)

    K = _kernel(Xo, Xo, cfg.spatial_len_x_mm, cfg.spatial_len_y_mm,
                cfg.spatial_kernel)
    A = K + np.diag(n_obs ** 2) + 1e-8 * np.eye(len(obs))

    allseg = sorted(geom.SEGMENTS, key=int)
    Xa = np.array([cen[s] for s in allseg], float)
    Ks = _kernel(Xa, Xo, cfg.spatial_len_x_mm, cfg.spatial_len_y_mm,
                 cfg.spatial_kernel)
    try:
        L = np.linalg.cholesky(A)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        m = Ks @ alpha
        V = np.linalg.solve(L, Ks.T)
        var = np.clip(1.0 - np.sum(V ** 2, axis=0), 0.0, None)
    except np.linalg.LinAlgError:
        return dict(values), dict(sds), {"ok": False, "note": "singular"}

    mean = {s: float(m[i] * amp + mu) for i, s in enumerate(allseg)}
    sd = {s: float(np.sqrt(var[i]) * amp) for i, s in enumerate(allseg)}

    # A measured segment keeps its own number.  The field is used to FILL, not
    # to overwrite: smoothing a good measurement towards its neighbours would
    # erase exactly the local anomaly the plate was built to find.
    for s in obs:
        mean[s] = float(values[s])
        if np.isfinite(sds.get(s, np.nan)):
            sd[s] = float(sds[s])

    return mean, sd, {"ok": True, "n_obs": len(obs),
                      "n_inferred": len(allseg) - len(obs)}


# ===========================================================================
# 4. Fault labelling
# ===========================================================================


def label_faults(records: dict[str, SegmentRecord]) -> None:
    """Ratio-to-median rules, applied in place."""
    def med(p):
        v = [r.values.get(p, np.nan) for r in records.values()
             if r.cls == "measured"]
        v = [x for x in v if np.isfinite(x) and x > 0]
        return float(np.median(v)) if v else np.nan

    ref = {p: med(p) for p in ("R_ohmic", "R_ct", "R_mt")}
    for r in records.values():
        if r.cls != "measured":
            continue
        ratios = {}
        for p, m in ref.items():
            v = r.values.get(p, np.nan)
            ratios[p + "_ratio"] = v / m if (np.isfinite(v) and m) else np.nan
        hits = []
        for name, rule in FAULT_RULES.items():
            ok = True
            for key, (lo, hi) in rule.items():
                x = ratios.get(key, np.nan)
                if not np.isfinite(x):
                    ok = False
                    break
                if lo is not None and x < lo:
                    ok = False
                if hi is not None and x > hi:
                    ok = False
            if ok:
                hits.append(name)
        r.fault = "|".join(hits)


# ===========================================================================
# 5. Plate drawing
# ===========================================================================


def _plate_background(ax, cfg: Config) -> None:
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0, 0), PLATE_W_CM, PLATE_H_CM,
                           facecolor=COLORS["plate"],
                           edgecolor=COLORS["plate_edge"], lw=2.5, zorder=1))
    for c in range(N_COLS):
        for r in range(N_ROWS):
            vx = (c * PAD_W_MM + PAD_W_MM / 2) / 10.0
            vy = (r * PAD_H_MM + PAD_H_MM / 2) / 10.0
            ax.add_patch(Rectangle((vx - 0.25, vy - 0.27), 0.50, 0.54,
                                   facecolor=COLORS["plate"],
                                   edgecolor=COLORS["via"], lw=0.4, zorder=2))
    for y_mm in FLOW_CHANNEL_Y_MM:
        ax.plot([0, PLATE_W_CM], [y_mm / 10.0, y_mm / 10.0],
                c=COLORS["channel_line"], lw=0.8, alpha=0.5, zorder=3)


def plate_heatmap(records: dict[str, SegmentRecord], param: str, cfg: Config,
                  title_extra: str = ""):
    """Static plate map with every segment drawn, measured or inferred."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    meta = PARAM_META.get(param, dict(label=param, unit="", scale=1.0,
                                      cmap=cfg.heatmap_colormap))
    scale = meta.get("scale", 1.0)
    vals = {s: r.values.get(param, np.nan) * scale
            for s, r in records.items()}
    finite = np.array([v for v in vals.values() if np.isfinite(v)])
    if finite.size == 0:
        return None

    span = float(np.ptp(finite)) or max(abs(float(np.mean(finite))) * 0.06, 1e-6)
    norm = Normalize(vmin=finite.min() - 0.03 * span,
                     vmax=finite.max() + 0.03 * span)
    cmap = plt.get_cmap(meta.get("cmap", cfg.heatmap_colormap))

    fig, ax = plt.subplots(figsize=(18, 8.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_aspect("equal")
    _plate_background(ax, cfg)

    for s, r in records.items():
        v = vals.get(s, np.nan)
        g = geom.SEGMENTS[s]
        x0, y0 = g.x0_mm / 10.0, g.y0_mm / 10.0
        w, h = g.w_mm / 10.0, g.h_mm / 10.0
        style = SEGMENT_CLASS_STYLE.get(r.cls, SEGMENT_CLASS_STYLE["measured"])
        face = cmap(norm(v)) if np.isfinite(v) else (0.6, 0.6, 0.6, 1.0)
        edge = (COLORS["measured_edge"] if r.cls == "measured"
                else COLORS["bad_edge"] if r.cls == "bad"
                else COLORS["inferred_edge"])
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor=face,
                               alpha=style["alpha"], edgecolor=edge,
                               lw=style["linewidth"], hatch=style["hatch"],
                               zorder=5))
        big = int(s) <= 36
        txt = f"{s}\n{v:.1f}" if np.isfinite(v) else s
        ax.text(g.cx_mm / 10.0, g.cy_mm / 10.0, txt, ha="center", va="center",
                fontsize=(8.5 if big else 6.5), color="white",
                fontweight="bold", zorder=8)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.015)
    unit = meta.get("unit", "")
    cb.set_label(f"{meta.get('label', param)}" + (f"  [{unit}]" if unit else ""),
                 fontsize=12, labelpad=10)

    n_inf = sum(1 for r in records.values() if r.cls != "measured")
    ax.set_xlim(-0.1, PLATE_W_CM + 0.1)
    ax.set_ylim(-0.1, PLATE_H_CM + 0.1)
    ax.axis("off")
    ax.set_title(f"{meta.get('label', param)}{title_extra}\n"
                 f"{len(records) - n_inf} measured, {n_inf} inferred "
                 f"(hatched) - gas flows left to right",
                 fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig


def plate_heatmap_interactive(records: dict[str, SegmentRecord], param: str,
                              cfg: Config):
    """Plotly version: same map, with per-segment hover carrying the caveats."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    meta = PARAM_META.get(param, dict(label=param, unit="", scale=1.0,
                                      cmap=cfg.heatmap_colormap))
    scale = meta.get("scale", 1.0)
    fig = go.Figure()
    xs, ys, zs, texts = [], [], [], []
    for s, r in records.items():
        g = geom.SEGMENTS[s]
        v = r.values.get(param, np.nan) * scale
        sd = r.sd.get(param, np.nan) * scale
        xs.append(g.cx_mm / 10.0)
        ys.append(g.cy_mm / 10.0)
        zs.append(v)
        texts.append(
            f"<b>segment {s}</b><br>"
            f"{meta.get('label', param)} = {v:.3g} {meta.get('unit','')}"
            + (f" &plusmn; {sd:.2g}" if np.isfinite(sd) else "")
            + f"<br>class: {r.cls} (tier {r.tier})"
            + f"<br>area: {r.area_cm2:.3f} cm&sup2;"
            + (f"<br>fault: {r.fault}" if r.fault else "")
            + (f"<br>flags: {', '.join(r.flags)}" if r.flags else "")
        )
    sizes = [26 if int(s) <= 36 else 16 for s in records]
    symbols = ["square" if r.cls == "measured" else "square-open"
               for r in records.values()]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text",
        marker=dict(size=sizes, color=zs, colorscale="RdYlBu_r",
                    symbol=symbols, line=dict(width=1.5, color="white"),
                    colorbar=dict(title=f"{meta.get('label',param)}<br>"
                                        f"{meta.get('unit','')}")),
        text=[s for s in records], textposition="middle center",
        textfont=dict(size=8, color="white"),
        hovertext=texts, hoverinfo="text", name=param))
    fig.update_layout(
        title=f"{meta.get('label', param)} - open squares are inferred, "
              f"not measured",
        xaxis=dict(title="x [cm] (gas flow ->)", range=[-0.3, PLATE_W_CM + 0.3],
                   constrain="domain"),
        yaxis=dict(title="y [cm]", range=[-0.3, PLATE_H_CM + 0.3],
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor="white", height=620, width=1400)
    return fig


def nyquist_figure(sr: SilverRun, cfg: Config):
    """Every segment, measured points and model curve, plus the cell curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
    segs = sorted(sr.spectra, key=int)
    cm = plt.get_cmap("viridis")
    for i, s in enumerate(segs):
        sp = sr.spectra[s]
        c = cm(i / max(1, len(segs) - 1))
        a = 0.9 if sp.tier == "A" else 0.5 if sp.tier == "B" else 0.25
        zr, zc = sp.Z_corr * 1000, sp.Z_model * 1000
        ax[0].plot(np.real(zr), -np.imag(zr), ".", ms=2, color=c, alpha=.25 * a)
        ax[0].plot(np.real(zc), -np.imag(zc), "-", lw=1.1, color=c, alpha=a)
        ax[1].plot(sp.freq, np.abs(zc), "-", lw=1.0, color=c, alpha=a)
        ax[2].plot(sp.freq, np.degrees(np.angle(zc)), "-", lw=1.0, color=c,
                   alpha=a)
    if len(sr.cell_freq):
        zc = sr.Z_cell * 1000
        ax[0].plot(np.real(zc), -np.imag(zc), "k--d", ms=4, lw=2, zorder=9,
                   label="cell: area-weighted harmonic mean")
        ax[1].plot(sr.cell_freq, np.abs(zc), "k--", lw=2, zorder=9)
        ax[2].plot(sr.cell_freq, np.degrees(np.angle(zc)), "k--", lw=2, zorder=9)
        ax[0].legend(fontsize=8)
    ax[0].set(xlabel="Z' [mOhm cm2]", ylabel="-Z'' [mOhm cm2]",
              title="Nyquist - dots measured, lines model")
    ax[0].axhline(0, color=".6", lw=.7, ls=":")
    ax[0].set_aspect("equal", adjustable="datalim")
    ax[1].set(xscale="log", yscale="log", xlabel="f [Hz]",
              ylabel="|Z| [mOhm cm2]", title="Bode magnitude")
    ax[2].set(xscale="log", xlabel="f [Hz]", ylabel="phase [deg]",
              title="Bode phase", ylim=(-90, 30))
    ax[2].axhline(0, color=".6", lw=.7, ls=":")
    for a_ in ax:
        a_.grid(alpha=.25, which="both")
    fig.suptitle(f"Local EIS, {len(segs)} segments, opacity by quality tier",
                 fontweight="bold")
    fig.tight_layout()
    return fig


# ===========================================================================
# 6. Entry point
# ===========================================================================


def run(sr: SilverRun, cfg: Config = DEFAULT, log=None) -> GoldRun:
    log = log or utils.get_logger(cfg.verbose)
    utils.banner("GOLD  --  the plate as a field", log)

    utils.section("scalars per segment", log)
    vals, sds = collect_parameters(sr, cfg)
    log.info(f"  {len(vals)} parameters over {len(sr.spectra)} measured segments")

    # ---- spatial completion -----------------------------------------------
    utils.section("spatial field and completion", log)
    allseg = sorted(geom.SEGMENTS, key=int)
    measured = set(sr.spectra)
    n_missing = len(allseg) - len(measured)
    frac = n_missing / len(allseg)
    do_infer = (cfg.spatial_enable and cfg.infer_missing_segments
                and frac <= cfg.max_inferred_fraction)
    if not do_infer and n_missing:
        log.warning(f"  {n_missing}/{len(allseg)} segments unmeasured "
                    f"({100*frac:.0f} %) - inference "
                    + ("disabled by config" if not cfg.infer_missing_segments
                       else f"REFUSED: above max_inferred_fraction "
                            f"({100*cfg.max_inferred_fraction:.0f} %). "
                            f"A map that is mostly guess is worse than no map."))

    fields: dict[str, dict] = {}
    for p, v in vals.items():
        if do_infer:
            m, s, info = fit_field(v, sds.get(p, {}), cfg)
        else:
            m, s, info = dict(v), dict(sds.get(p, {})), {"ok": False}
        fields[p] = {"value": m, "sd": s, "info": info}
    if do_infer:
        info = fields.get("R_ohmic", {}).get("info", {})
        if info.get("ok"):
            log.info(f"  GP field fitted on {info['n_obs']} segments, "
                     f"{info['n_inferred']} inferred "
                     f"(len_x={cfg.spatial_len_x_mm:.0f} mm along flow, "
                     f"len_y={cfg.spatial_len_y_mm:.0f} mm across)")

    # ---- assemble records --------------------------------------------------
    cen = geom.centroids()
    _areas = utils.segment_areas(cfg)
    records: dict[str, SegmentRecord] = {}
    for s in allseg:
        g = geom.SEGMENTS[s]
        if s in measured:
            sp = sr.spectra[s]
            flags = list(sp.flags)
            bad_primary = not (np.isfinite(sp.R_ohmic) and sp.R_ohmic > 0)
            cls = "inferred" if bad_primary else "measured"
            tier = "D" if bad_primary else sp.tier
        elif s in KNOWN_BAD_SEGMENTS:
            cls, tier = "bad", "D"
            flags = [f"hardware: {KNOWN_BAD_SEGMENTS[s]}"]
        else:
            cls, tier = "inferred", "D"
            flags = ["no channel or fit failed; value from the spatial field"]
        rec = SegmentRecord(segment=s, cx_mm=cen[s][0], cy_mm=cen[s][1],
                            area_cm2=_areas.get(s, g.area_cm2),
                            cls=cls, tier=tier,
                            flags=flags)
        for p, d in fields.items():
            v = d["value"].get(s, np.nan)
            if np.isfinite(v):
                rec.values[p] = float(v)
                rec.sd[p] = float(d["sd"].get(s, np.nan))
        records[s] = rec

    label_faults(records)
    n_fault = sum(1 for r in records.values() if r.fault)
    log.info(f"  {len(measured)} measured, "
             f"{sum(1 for r in records.values() if r.cls=='inferred')} inferred, "
             f"{sum(1 for r in records.values() if r.cls=='bad')} hardware-bad")
    if n_fault:
        by = {}
        for r in records.values():
            for f_ in (r.fault.split("|") if r.fault else []):
                by[f_] = by.get(f_, 0) + 1
        log.info("  fault signatures: "
                 + ", ".join(f"{k}={v}" for k, v in sorted(by.items())))
    else:
        log.info("  no segment matched a fault signature")

    stats = {
        "n_total": len(allseg), "n_measured": len(measured),
        "n_inferred": sum(1 for r in records.values() if r.cls == "inferred"),
        "n_bad": sum(1 for r in records.values() if r.cls == "bad"),
        "inferred_fraction": frac,
        "tiers": sr.tiers(),
        "dc_closure": sr.dc_closure,
    }
    for p in ("R_ohmic", "R_ct", "R_mt", "j_dc"):
        v = [r.values.get(p, np.nan) for r in records.values()
             if r.cls == "measured"]
        # Resistances and current densities are positive by construction, so a
        # non-positive value is a failed fit, not a small measurement.  Letting
        # one through makes max/min meaningless -- it produced a reported
        # "spread" of 8e10 before this guard existed.
        v = np.array([x for x in v if np.isfinite(x) and x > 0])
        if v.size:
            stats[p] = {"mean": float(v.mean()), "sd": float(v.std()),
                        "min": float(v.min()), "max": float(v.max()),
                        "spread": float(v.max() / v.min()),
                        "n_used": int(v.size)}
    return GoldRun(records=records, fields=fields, stats=stats,
                   cell_freq=sr.cell_freq, Z_cell=sr.Z_cell)


def save(gr: GoldRun, sr: SilverRun, cfg: Config, log=None) -> Path:
    log = log or utils.get_logger(cfg.verbose)
    out = Path(cfg.out_dir) / "gold"
    out.mkdir(parents=True, exist_ok=True)

    params = sorted({p for r in gr.records.values() for p in r.values})
    rows = []
    for s in sorted(gr.records, key=int):
        r = gr.records[s]
        row = {"segment": s, "class": r.cls, "tier": r.tier,
               "cx_mm": round(r.cx_mm, 2), "cy_mm": round(r.cy_mm, 2),
               "area_cm2": round(r.area_cm2, 5), "fault": r.fault,
               "flags": ";".join(r.flags)}
        for p in params:
            v = r.values.get(p, np.nan)
            sd = r.sd.get(p, np.nan)
            sc = PARAM_META.get(p, {}).get("scale", 1.0)
            row[p] = round(v * sc, 5) if np.isfinite(v) else ""
            row[p + "_sd"] = round(sd * sc, 5) if np.isfinite(sd) else ""
        rows.append(row)
    utils.write_table(out / "plate_summary.csv", rows)

    if cfg.write_png:
        for p in cfg.heatmap_params:
            if p not in params:
                continue
            fig = plate_heatmap(gr.records, p, cfg)
            if fig is None:
                continue
            import matplotlib.pyplot as plt
            fig.savefig(out / f"map_{p}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        fig = nyquist_figure(sr, cfg)
        import matplotlib.pyplot as plt
        fig.savefig(out / "nyquist.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if cfg.write_html:
        for p in cfg.heatmap_params:
            if p not in params:
                continue
            f = plate_heatmap_interactive(gr.records, p, cfg)
            if f is not None:
                f.write_html(str(out / f"map_{p}.html"),
                             include_plotlyjs="cdn")

    utils.write_json(out / "gold_manifest.json", gr.stats)
    log.info(f"  gold written to {out}")
    return out
