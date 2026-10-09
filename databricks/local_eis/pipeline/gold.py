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

import r2d2_geometry as geom

import utils
import plate_stats
from config import (Config, DEFAULT, PARAM_META, FAULT_RULES, PLATE_W_MM,
                    KNOWN_BAD_SEGMENTS)
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
    cls: str     # measured | substituted | inferred | excluded | bad
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
    # A NEGATIVE BUCKET IS NOT ZERO RESISTANCE, IT IS A FAILED FIT.
    # max(sum, 0) reported a hard 0.0 whenever the unconstrained DRT had
    # cancelled a real arc with negative weight -- and 0.0 reads as a
    # measurement ("this segment has no mass transport") when it is the
    # opposite, a sign that the fit did not resolve the process. With
    # drt_nonneg on (the default) gamma cannot be negative and this never
    # fires; if it is switched off, the bucket is reported as unavailable
    # rather than as zero.
    def _bucket(mask) -> float:
        total = float(g[mask].sum())
        if total < 0:
            return float("nan")
        return total

    out["R_hf_extra"] = _bucket(fast)
    out["R_ct"] = _bucket(mid)
    out["R_mt"] = _bucket(slow)
    return out


def spatial_trends(records: dict, params, plate_w_mm: float = PLATE_W_MM
                   ) -> list[dict]:
    """Is a pattern on the map a gradient, or is it the segment geometry?

    A segmented plate does not have segments of one size. On this one the
    areas span 1.36 to 8.47 cm2, a factor of 6.2, and the SMALL ones are the
    staircase segments along the two ends. So "high at both ends" and "high
    on the small segments" are the same set of segments, and a map cannot
    tell them apart by eye.

    That matters because the two have opposite meanings. A hydration
    gradient is the measurement working. An area-correlated offset is a
    systematic -- contact pressure at the plate edge, lateral current
    spreading into the periphery, or an area that disagrees between the
    calibration and the geometry -- and it is not a property of the cell.

    Three correlations are reported per field, plus the PARTIAL correlation
    with area once the along-plate shape is removed. On RO2612030 at 450 A
    the measured numbers are r(x) = -0.11, r(|x-centre|) = +0.65 and
    r(area) = -0.65, with partials of +0.10 and -0.08: the flow-direction
    gradient is absent, the centre-low pattern is real, and it is
    indistinguishable from an area effect in this geometry. Nothing here
    decides which it is -- that needs a second condition or a second plate --
    but a run that does not report it invites the wrong conclusion.
    """
    out = []
    for p in params:
        segs = [s for s, r in records.items()
                if r.cls == "measured" and np.isfinite(r.values.get(p, np.nan))]
        if len(segs) < 8:
            continue
        x = np.array([records[s].cx_mm for s in segs], float)
        y = np.array([records[s].cy_mm for s in segs], float)
        a = np.array([records[s].area_cm2 for s in segs], float)
        v = np.array([records[s].values[p] for s in segs], float)
        u = np.abs(x - 0.5 * plate_w_mm)

        # A FIELD THAT DOES NOT VARY HAS NO TREND, AND SAYS SO.
        # Correlating against a constant is 0/0: numpy returns whatever the
        # floating-point dust happens to be, which came out as r = +0.88 on a
        # field that was identical on every segment. The scale to compare the
        # spread against is the field's own magnitude, not an absolute number,
        # because these fields range from microseconds to hundreds of
        # milliohms.
        scale = max(float(np.mean(np.abs(v))), 1e-30)

        def _r(q, w):
            if np.std(q) <= 1e-9 * max(float(np.mean(np.abs(q))), 1e-30):
                return float("nan")
            if np.std(w) <= 1e-9 * max(float(np.mean(np.abs(w))), 1e-30):
                return float("nan")
            return float(np.corrcoef(q, w)[0, 1])

        def _partial(q, control):
            """correlation of q with v after removing a linear trend in control"""
            if np.std(v) <= 1e-9 * scale:
                return float("nan")
            if np.std(control) == 0:
                return _r(q, v)
            b = np.polyfit(control, v, 1)
            resid = v - np.polyval(b, control)
            if np.std(resid) <= 1e-9 * scale:
                return float("nan")       # the control explains it entirely
            return _r(q, resid)

        row = {"param": p, "n": len(segs),
               "r_along_flow": round(_r(x, v), 3),
               "r_across": round(_r(y, v), 3),
               "r_edge": round(_r(u, v), 3),
               "r_area": round(_r(a, v), 3),
               "r_area_partial": round(_partial(a, u), 3),
               "r_edge_partial": round(_partial(u, a), 3),
               "area_span": round(float(a.max() / a.min()), 2)}
        out.append(row)
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
        # the current-chain lag found (and, in "correct" mode, removed) for
        # this segment -- mapped so a wiring pattern is visible as one
        put("chain_tau_us", s, 1e6 * getattr(sp, "chain_tau_est", np.nan))
        # Re Z at ONE common frequency. R_ohmic is Re Z at the top of each
        # segment's kept band, so it moves with how far that band reaches
        # (on 2612030 the plate median ran 59..69 mOhm*cm2 over four currents
        # for that reason alone, while Re Z at 1 kHz agreed to 2 %). This one
        # compares segments and conditions like for like.
        put("ReZ_1kHz", s, re_at(sp.freq, sp.Z_corr, 1000.0))
        # |Z| and phase at the reference frequency
        f = sp.freq
        if len(f):
            i = int(np.argmin(np.abs(f - 100.0)))
            if abs(f[i] - 100.0) <= 0.5 * 100.0:
                put("Z_mag_100Hz", s, abs(sp.Z_model[i]))
                put("phase_100Hz", s, np.degrees(np.angle(sp.Z_model[i])))
    return vals, sds


def re_at(freq, Z, f0: float) -> float:
    """Re Z at f0 from the kept points, interpolated in log f; nan when f0
    is outside them or the nearest points are more than a third of a decade
    away (no extrapolation)."""
    f = np.asarray(freq, float)
    z = np.asarray(Z, complex)
    ok = np.isfinite(f) & np.isfinite(z) & (f > 0)
    f, z = f[ok], z[ok]
    if f.size < 2 or not (f.min() <= f0 <= f.max()):
        return float("nan")
    o = np.argsort(f)
    f, z = f[o], z[o]
    k = int(np.searchsorted(f, f0))
    k = min(max(k, 1), f.size - 1)
    if np.log10(f[k] / f[k - 1]) > 1.0 / 3.0:
        return float("nan")
    return float(np.interp(np.log(f0), np.log(f[[k - 1, k]]),
                           z.real[[k - 1, k]]))


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


def plate_heatmap(records: dict[str, SegmentRecord], param: str, cfg: Config,
                  title_extra: str = ""):
    """Static plate map with every segment drawn, measured or inferred.

    Drawn by plate_figure, so map_<param>.png and plate_<param>.png are the
    same drawing: true segment outlines (this used to draw bounding boxes,
    which overlap on the staircase edge segments), pad row 1 at the top (this
    used to draw the plate upside down relative to every other map), the
    plate_style colours (plotly Jet), value large and number small, and the
    gas and coolant ports on their ends.
    """
    import plate_figure

    meta = PARAM_META.get(param, dict(label=param, unit="", scale=1.0))
    scale = meta.get("scale", 1.0)
    vals = {s: r.values.get(param, np.nan) * scale
            for s, r in records.items()}
    if not any(np.isfinite(v) for v in vals.values()):
        return None
    dec = plate_figure.FIELDS.get(param, (None, None, None, None))[3]
    if dec is None:
        dec = 3 if abs(np.nanmedian(list(vals.values()))) < 10 else 1
    return plate_figure.draw_flow_plate(
        vals, param, classes={s: r.cls for s, r in records.items()},
        title=(f"{cfg.leepa or ''} / {cfg.condition or ''}".strip(" /")
               + title_extra),
        label=meta.get("label", param), unit=meta.get("unit", ""),
        decimals=dec)


def plate_heatmap_interactive(records: dict[str, SegmentRecord], param: str,
                              cfg: Config):
    """Plotly version: same map, with per-segment hover carrying the caveats.

    plate_plotly draws the real segment shapes in the same colours and view
    as the PNG; hover anywhere on a segment for its value, sd, class, tier,
    area, fault and flags.
    """
    try:
        import plate_plotly
    except ImportError:
        return None
    meta = PARAM_META.get(param, dict(label=param, unit="", scale=1.0))
    scale = meta.get("scale", 1.0)
    notes = {}
    for s, r in records.items():
        notes[s] = (f"tier {r.tier} · area {r.area_cm2:.3f} cm\u00b2"
                    + (f"<br>fault: {r.fault}" if r.fault else "")
                    + (f"<br>flags: {', '.join(r.flags)}" if r.flags else ""))
    vals = {s: r.values.get(param, np.nan) * scale for s, r in records.items()}
    if not any(np.isfinite(v) for v in vals.values()):
        return None
    fd = plate_plotly.Field(
        key=param, label=meta.get("label", param), unit=meta.get("unit", ""),
        values=vals, classes={s: r.cls for s, r in records.items()},
        sd={s: r.sd.get(param, np.nan) * scale for s, r in records.items()},
        notes=notes,
        decimals=3 if abs(np.nanmedian(list(vals.values()))) < 10 else 1)
    return plate_plotly.interactive_plate(
        [fd], title=f"{cfg.leepa or ''} / {cfg.condition or ''}".strip(" /"),
        subtitle="hatched / value* = rebuilt, not measured")


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

    # AN EXCLUDED SEGMENT MUST NOT COME BACK AS AN INFERRED ONE.
    # `fit_field` returns a value for EVERY segment, so a segment the operator
    # excluded would otherwise be handed a Gaussian-process value from its
    # neighbours, labelled "inferred", and drawn on every heat map. That is
    # the opposite of what excluding it meant: the request was to leave it
    # out of the evaluation, not to replace its measurement with a guess.
    excluded = {str(x) for x in getattr(cfg, "exclude_segments", ()) or ()}
    # Segments silver rebuilt from their neighbours. They are NOT measured and
    # they are not GP-inferred either -- they come from the ring that touches
    # them, which is a different and more local statement, so they get their
    # own class and carry their donors.
    rebuilt = dict(getattr(sr, "filled", {}) or {})
    _areas = utils.segment_areas(cfg)
    fields: dict[str, dict] = {}
    for p, v in vals.items():
        v = dict(v)
        # EVERY scalar, not just R_ohmic.
        # The first version filled only the HF intercept, because that one can
        # be read straight off the rebuilt curve. R_ct and R_mt cannot: they
        # come from the DRT, which is not linear in Z, so there is no rebuilt
        # spectrum to read them from. The result was a plate where the HFR map
        # had 72 segments and the charge-transfer and mass-transport maps had
        # 67 -- the same run, the same request, three different plates.
        #
        # They are taken from the DONORS instead, area-weighted, which is what
        # "use the neighbouring segments" means for a scalar and is the same
        # rule the spectrum reconstruction uses. For R_ohmic the two agree to
        # within the per-point weights anyway, because the top-band estimator
        # is itself a weighted mean of Re Z and the reconstruction is a
        # weighted mean of Z.
        for seg, sp in rebuilt.items():
            donor_vals, donor_w = [], []
            for d in getattr(sp, "donors", []):
                dv = v.get(d, np.nan)
                if np.isfinite(dv):
                    donor_vals.append(float(dv))
                    donor_w.append(float(_areas.get(d, 1.0)))
            if donor_vals:
                w = np.asarray(donor_w, float)
                v.setdefault(seg, float(np.sum(w * np.asarray(donor_vals))
                                        / np.sum(w)))
        if do_infer:
            m, s, info = fit_field(v, sds.get(p, {}), cfg)
        else:
            m, s, info = dict(v), dict(sds.get(p, {})), {"ok": False}
        for x in excluded:
            m.pop(x, None)
            s.pop(x, None)
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
        elif s in rebuilt:
            cls, tier = "substituted", "D"
            _d = rebuilt[s]
            flags = [f"rebuilt from neighbours {' '.join(_d.donors)}"
                     + (f" ({_d.hops} hops)" if _d.hops > 1 else "")
                     + "; an estimate, not a measurement"]
        elif s in excluded:
            # Said plainly, and distinctly from "the fit failed": a reader
            # comparing two runs needs to see that this segment was taken out
            # on purpose, not that it stopped working.
            cls, tier = "excluded", "D"
            flags = ["excluded by configuration (exclude_segments); "
                     "not measured, not inferred"]
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

    # ---- is the pattern on the map a gradient or the geometry? ------------
    trends = spatial_trends(records, list(fields))
    if trends:
        utils.section("spatial trends (gradient, or segment geometry?)", log)
        log.info("  field         n   r(flow) r(edge) r(area)  partial area")
        for t in trends:
            log.info(f"  {t['param']:<12} {t['n']:3d}   {t['r_along_flow']:+6.3f} "
                     f" {t['r_edge']:+6.3f} {t['r_area']:+6.3f}   "
                     f"{t['r_area_partial']:+6.3f}")
        worst = max(trends, key=lambda t: abs(t["r_area"]))
        if abs(worst["r_area"]) >= 0.4:
            log.warning(
                f"  {worst['param']} correlates with segment AREA at "
                f"r = {worst['r_area']:+.2f} (areas span "
                f"{worst['area_span']:.1f}x on this plate). The small "
                f"segments are the ones along the two ends, so an edge "
                f"pattern and an area effect are the SAME set of segments "
                f"here and this map cannot separate them. Before reading it "
                f"as hydration, check it against a second condition: a "
                f"hydration gradient moves with current and stoichiometry, "
                f"a geometric systematic does not.")

    n_fault = sum(1 for r in records.values() if r.fault)
    log.info(f"  {len(measured)} measured, "
             f"{sum(1 for r in records.values() if r.cls=='inferred')} inferred, "
             f"{sum(1 for r in records.values() if r.cls=='substituted')} "
             f"rebuilt from neighbours, "
             f"{sum(1 for r in records.values() if r.cls=='excluded')} excluded, "
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
        "n_excluded": sum(1 for r in records.values() if r.cls == "excluded"),
        "n_substituted": sum(1 for r in records.values()
                             if r.cls == "substituted"),
        "spatial_trends": trends,
        "substituted_segments": {s: list(sp.donors)
                                 for s, sp in sorted(rebuilt.items())},
        "excluded_segments": sorted(excluded, key=lambda x: int(x)
                                    if str(x).isdigit() else 0),
        "n_bad": sum(1 for r in records.values() if r.cls == "bad"),
        "inferred_fraction": frac,
        "tiers": sr.tiers(),
        "dc_closure": sr.dc_closure,
    }
    # Plate statistics over the 36 tile segments (plate_stats): segment
    # 1..36 stands for its whole 5x5-pad tile and the edge segments 37..72
    # inside the tiles are not counted, so every tile weighs the same area.
    # Resistances and current densities are positive by construction, so a
    # non-positive value is a failed fit, not a small measurement.  Letting
    # one through makes max/min meaningless -- it produced a reported
    # "spread" of 8e10 before this guard existed.
    cls = {s: r.cls for s, r in records.items()}
    for p in ("R_ohmic", "ReZ_1kHz", "R_ct", "R_mt", "R_pol", "j_dc"):
        st = plate_stats.tile_stats(
            {s: r.values.get(p, np.nan) for s, r in records.items()}, cls,
            positive=True)
        if st["n"]:
            stats[p] = {"mean": st["mean"], "median": st["median"],
                        "sd": st["sd"], "min": st["min"], "max": st["max"],
                        "spread": st["max"] / st["min"],
                        "n_used": st["n"], "missing_tiles": st["missing"],
                        "basis": st["basis"]}
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
    # plate mean / median per parameter over the 36 tile segments
    plate_stats.write_statistics(out / "plate_summary.csv")

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
        # plate maps in the viewer's layout (ports, flow arrows, sensors),
        # fixed colour scale per parameter -- plate_figure.py
        try:
            import plate_figure
            import matplotlib.pyplot as plt
            cls = {s: r.cls for s, r in gr.records.items()}
            for p in ("R_ohmic", "R_ct", "R_mt", "R_pol"):
                if p not in params:
                    continue
                sc = PARAM_META.get(p, {}).get("scale", 1.0)
                vals = {s: r.values.get(p, np.nan) * sc
                        for s, r in gr.records.items()}
                for rd in plate_figure.renders(
                        getattr(cfg, "heatmap_style", "segments")):
                    fig = plate_figure.draw_flow_plate(
                        vals, p, classes=cls, render=rd,
                        title=f"{cfg.leepa or ''} / {cfg.condition or ''}"
                        .strip(" /"))
                    fig.savefig(out / f"plate_{p}{plate_figure.suffix(rd)}.png",
                                dpi=200, bbox_inches="tight",
                                facecolor=fig.get_facecolor())
                    plt.close(fig)
        except Exception as exc:                            # noqa: BLE001
            log.info(f"  plate figures skipped: {type(exc).__name__}: {exc}")
        # the Nyquist the runner shows, coloured cathode inlet -> outlet, as
        # PNG and interactive HTML (plotting/nyquist.py); silver has written
        # spectra_clean.csv by now
        try:
            import nyquist as _nyq
            _nyq.save_run(cfg.out_dir, out_dir=out,
                          title=f"{cfg.leepa or ''} / {cfg.condition or ''}"
                                .strip(" /"))
        except Exception as exc:                            # noqa: BLE001
            log.info(f"  nyquist figure skipped: {type(exc).__name__}: {exc}")

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
