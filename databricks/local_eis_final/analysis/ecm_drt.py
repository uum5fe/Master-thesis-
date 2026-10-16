#!/usr/bin/env python3
"""
ecm_drt.py
==========
DRT-informed equivalent-circuit fit (the notebook's "method B"), revised for
the thesis. Model:

    Z(f) = Rs + jwL + fast arc [+ slow arc]
    arc  = ZARC  R / (1 + (jw tau)^n)          (default)
         | TLM   catalyst-layer transmission line for the FAST arc:
                 sqrt(R_ion Z_i) coth(sqrt(R_ion / Z_i)),
                 Z_i = R / (1 + (jw tau)^n)      (option, `fast_element`)

What changed against the cell it replaces, point by point:

1. Arcs are SORTED by tau after the fit and reported as `fast` and `slow`,
   so the same column means the same process in every condition.
2. Weighting by the measurement uncertainty sigma = sigma_rel * |Z| (the
   CRLB the pipeline propagates). Only then is chi2_nu a chi-square. Without
   sigma the fit is weighted by 1/|Z| and its figure of merit is called what
   it is: `rms_rel_resid`, the RMS relative residual -- chi2_nu is NaN.
3. One arc or two, chosen by AICc, instead of always two.
4. A stricter "clean": besides converging with no parameter on a bound, no
   arc may carry < 1 % of R_pol, and every tau must lie inside the fitted
   band, 1/(2 pi f_max) .. 1/(2 pi f_min).
5. The top band no longer drives the fit: points above `f_max_fit_hz`
   (1 kHz) are left out, and Rs is held at the pipeline's own R_ohmic when
   given (`rs_fixed`). A low n of the fast arc (< 0.7) is flagged; the TLM
   fast element is offered for that case and competes by AICc.
6. Points on odd mains harmonics (50, 150, 250 ... Hz, +-1.5 %) are dropped
   and listed (`dropped_mains`).
7. `summarize` reports MEDIANS (and IQR) over ALL fitted segments, with the
   clean count beside them -- not means over the clean subset.

Units: whatever Z is given in (the notebook uses mOhm*cm2); tau in s.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# settings (the notebook can override them per call)
# ---------------------------------------------------------------------------
F_MAX_FIT_HZ = 1000.0          # point 5: top band left out of the fit
MAINS_HZ = 50.0                # point 6
MAINS_ODD_ONLY = True          # 50, 150, 250, ... (3-phase rectifier lines)
MAINS_TOL = 0.015              # +-1.5 % around each harmonic
MAINS_MAX_ORDER = 5            # 50, 150, 250 Hz; higher orders lie every
                               # 50 Hz and would hit any step above ~1.5 kHz
MIN_ARC_FRAC = 0.01            # point 4: an arc below 1 % of R_pol is noise
LOW_N_FAST = 0.70              # point 5: flag, and try the TLM fast element
L_MAX = 1e-1                   # series L bound, Z-units * s (mOhm*cm2*s)


# ---------------------------------------------------------------------------
# elements
# ---------------------------------------------------------------------------
def z_zarc(w, R, tau, n):
    return R / (1.0 + (1j * w * tau) ** n)


def z_tlm(w, R_ion, R, tau, n):
    """Finite catalyst-layer transmission line with a ZARC interface."""
    zi = z_zarc(w, R, tau, n)
    zi = np.where(np.abs(zi) < 1e-30, 1e-30, zi)
    g = np.sqrt(R_ion / zi)
    # coth with overflow guard: coth(x) -> 1 for large |x|
    coth = np.where(np.abs(g) > 40, 1.0 + 0j, 1.0 / np.tanh(g))
    return np.sqrt(R_ion * zi) * coth


def _unpack(p, layout):
    """layout = ('zarc',) | ('zarc','zarc') | ('tlm','zarc'); p after Rs, L."""
    arcs, i = [], 0
    for el in layout:
        if el == "tlm":
            arcs.append(("tlm", p[i], p[i + 1], p[i + 2], p[i + 3]))
            i += 4
        else:
            arcs.append(("zarc", None, p[i], p[i + 1], p[i + 2]))
            i += 3
    return arcs


def z_model(Rs, L, p_arcs, layout, freq):
    w = 2 * np.pi * np.asarray(freq, float)
    Z = Rs + 1j * w * L + 0j
    for kind, R_ion, R, tau, n in _unpack(p_arcs, layout):
        Z = Z + (z_tlm(w, R_ion, R, tau, n) if kind == "tlm"
                 else z_zarc(w, R, tau, n))
    return Z


# ---------------------------------------------------------------------------
# point selection
# ---------------------------------------------------------------------------
def mains_mask(freq, mains_hz=MAINS_HZ, odd_only=MAINS_ODD_ONLY, tol=MAINS_TOL,
               max_order=MAINS_MAX_ORDER):
    """True where a point sits on a low mains harmonic (to be dropped)."""
    f = np.asarray(freq, float)
    k = np.maximum(np.rint(f / mains_hz), 1)
    on = (np.abs(f / (k * mains_hz) - 1.0) <= tol) & (k <= max_order)
    if odd_only:
        on &= (k % 2 == 1)
    return on


# ---------------------------------------------------------------------------
# starting values
# ---------------------------------------------------------------------------
def _start(freq, Z, Rs, n_arcs, peaks=None):
    o = np.argsort(freq)
    f, Zs = freq[o], Z[o]
    R_tot = max(float(Zs[0].real) - Rs, 1e-6 * max(abs(Zs[0]), 1e-12))
    im = -Zs.imag
    tau_pk = 1.0 / (2 * np.pi * f[int(np.argmax(im))]) if im.size else 1e-2
    pk = sorted(peaks or [], key=lambda q: q["tau"])
    if n_arcs == 1:
        t = pk[0]["tau"] if len(pk) == 1 else tau_pk
        return [(R_tot, t, 0.85)]
    if len(pk) >= 2:
        a, b = pk[0], pk[-1]
        return [(max(a["R"], 0.05 * R_tot), a["tau"], 0.85),
                (max(b["R"], 0.05 * R_tot), b["tau"], 0.85)]
    return [(0.5 * R_tot, tau_pk / 10, 0.85), (0.5 * R_tot, tau_pk * 10, 0.85)]


# ---------------------------------------------------------------------------
# one fit
# ---------------------------------------------------------------------------
def _fit_layout(f, Z, s, layout, Rs0, rs_fixed, peaks, tau_lo, tau_hi):
    from scipy.optimize import least_squares

    n_arcs = len(layout)
    st = _start(f, Z, Rs0, n_arcs, peaks)
    p0, lo, hi = [], [], []
    if rs_fixed is None:
        p0 += [Rs0]; lo += [0.0]; hi += [np.inf]
    p0 += [1e-9]; lo += [0.0]; hi += [L_MAX]                   # L
    for el, (R, t, n) in zip(layout, st):
        if el == "tlm":
            p0 += [0.3 * R]; lo += [0.0]; hi += [np.inf]       # R_ion
        p0 += [R, t, n]
        lo += [0.0, 1e-7, 0.3]
        hi += [np.inf, 1e4, 1.0]
    p0 = np.clip(np.asarray(p0, float), np.array(lo) + 1e-15, np.array(hi))
    is_n = np.zeros(len(p0), bool)
    j = (0 if rs_fixed is not None else 1) + 1
    for el in layout:
        j += 1 if el == "tlm" else 0
        is_n[j + 2] = True
        j += 3

    def split(p):
        if rs_fixed is None:
            return p[0], p[1], p[2:]
        return rs_fixed, p[0], p[1:]

    def resid(p):
        Rs, L, pa = split(p)
        d = z_model(Rs, L, pa, layout, f) - Z
        return np.concatenate([d.real / s, d.imag / s])

    r = least_squares(resid, p0, bounds=(lo, hi), x_scale="jac",
                      max_nfev=20000)
    Rs, L, pa = split(r.x)
    nfree = len(r.x)
    at_bound = [i for i, (v, a, b) in enumerate(zip(r.x, lo, hi))
                if (v <= a * 1.001 + 1e-15 and a > 0)
                or (np.isfinite(b) and v >= b * 0.999 and not is_n[i])]
    return r, Rs, L, pa, nfree, at_bound


def fit_drt_ecm(freq, Z, sigma_rel=None, peaks=None, rs_fixed=None,
                f_max_fit_hz=F_MAX_FIT_HZ, drop_mains=True,
                fast_element="auto", max_arcs=2) -> dict:
    """Fit one spectrum. Returns a dict (see module docstring).

    peaks        DRT peaks [{'R', 'tau'}, ...] for starting values (optional)
    rs_fixed     hold Rs at this value (the pipeline's R_ohmic), or None
    fast_element "zarc" | "tlm" | "auto" (TLM tried when n_fast < 0.7)
    """
    freq = np.asarray(freq, float)
    Z = np.asarray(Z, complex)
    ok = np.isfinite(freq) & np.isfinite(Z.real) & np.isfinite(Z.imag) & (freq > 0)
    if sigma_rel is not None:
        sigma_rel = np.asarray(sigma_rel, float)
        ok &= np.isfinite(sigma_rel)
    in_band = freq <= f_max_fit_hz if f_max_fit_hz else np.ones_like(freq, bool)
    mains = mains_mask(freq) if drop_mains else np.zeros_like(freq, bool)
    use = ok & in_band & ~mains
    dropped_mains = [float(x) for x in freq[ok & in_band & mains]]
    f, Zf = freq[use], Z[use]
    if f.size < 8:
        return {"ok": False, "reason": f"{f.size} usable points below "
                                       f"{f_max_fit_hz:g} Hz"}

    if sigma_rel is not None and np.all(np.isfinite(sigma_rel[use])):
        s = np.clip(sigma_rel[use], 1e-4, 1.0) * np.abs(Zf)
        weight = "sigma"
    else:
        s = np.abs(Zf)
        weight = "modulus"
    s = np.maximum(s, 1e-15)

    Rs0 = float(rs_fixed) if rs_fixed is not None else max(
        float(Zf[np.argmax(f)].real) - 0.0, 0.0)
    tau_lo, tau_hi = 1 / (2 * np.pi * f.max()), 1 / (2 * np.pi * f.min())

    layouts = [("zarc",)] + ([("zarc", "zarc")] if max_arcs >= 2 else [])
    if fast_element == "tlm":
        layouts = [("tlm",)] + ([("tlm", "zarc")] if max_arcs >= 2 else [])
    cands = []

    def run(layout):
        try:
            r, Rs, L, pa, nfree, ab = _fit_layout(f, Zf, s, layout, Rs0,
                                                  rs_fixed, peaks, tau_lo, tau_hi)
        except Exception as exc:                            # noqa: BLE001
            return {"ok": False, "reason": f"{layout}: {exc}"}
        m = 2 * f.size
        rss = float(np.sum(r.fun ** 2))
        aicc = m * np.log(max(rss / m, 1e-300)) + 2 * nfree
        if m - nfree - 1 > 0:
            aicc += 2 * nfree * (nfree + 1) / (m - nfree - 1)
        return {"ok": True, "layout": layout, "res": r, "Rs": float(Rs),
                "L": float(L), "pa": np.asarray(pa, float), "nfree": nfree,
                "at_bound": ab, "rss": rss, "aicc": float(aicc)}

    for lay in layouts:
        c = run(lay)
        if c["ok"]:
            cands.append(c)
    if not cands:
        return {"ok": False, "reason": "no layout converged"}
    best = min(cands, key=lambda c: c["aicc"])

    # point 5: a low-n fast ZARC -> let the TLM compete
    if fast_element == "auto":
        arcs = sorted(_unpack(best["pa"], best["layout"]), key=lambda a: a[3])
        if arcs and arcs[0][0] == "zarc" and arcs[0][4] < LOW_N_FAST:
            lay = ("tlm",) + best["layout"][1:]
            c = run(lay)
            if c["ok"]:
                cands.append(c)
                if c["aicc"] < best["aicc"]:
                    best = c
    return _report(best, f, Zf, s, weight, dropped_mains, rs_fixed,
                   f_max_fit_hz, tau_lo, tau_hi, cands)


def _report(b, f, Zf, s, weight, dropped_mains, rs_fixed, f_max_fit_hz,
            tau_lo, tau_hi, cands):
    arcs = sorted(_unpack(b["pa"], b["layout"]), key=lambda a: a[3])  # point 1
    R_pol = float(sum(a[2] for a in arcs))
    Zm = z_model(b["Rs"], b["L"], b["pa"], b["layout"], f)
    rel = np.abs(Zf - Zm) / np.abs(Zf)
    m = 2 * f.size
    dof = max(m - b["nfree"], 1)
    flags = []
    if b["at_bound"]:
        flags.append("parameter on bound")
    for name, a in zip(("fast", "slow"), arcs):
        if R_pol > 0 and a[2] < MIN_ARC_FRAC * R_pol:
            flags.append(f"{name} arc < {100 * MIN_ARC_FRAC:.0f} % of R_pol")
        if not (tau_lo <= a[3] <= tau_hi):
            flags.append(f"{name} tau outside band")
    if arcs and arcs[0][0] == "zarc" and arcs[0][4] < LOW_N_FAST:
        flags.append(f"n_fast < {LOW_N_FAST}")
    # notes: reported, but not a reason to call the fit unclean
    notes = [f"n_{nm} = 1 (ideal RC)" for nm, a in zip(("fast", "slow"), arcs)
             if a[4] >= 0.999]
    if weight == "sigma":
        chi2_nu = float(b["rss"] / dof)
        if not 0.2 <= chi2_nu <= 5:
            notes.append(f"chi2_nu {chi2_nu:.3g}: residuals exceed the CRLB "
                         f"sigma (model error or sigma underestimated)")
    else:
        chi2_nu = float("nan")

    out = {
        "ok": True, "success": bool(b["res"].success),
        "clean": bool(b["res"].success) and not flags,
        "flags": flags, "notes": notes,
        "n_arcs": len(arcs),
        "fast_element": arcs[0][0] if arcs else "",
        "layouts_tried": [("+".join(c["layout"]), round(c["aicc"], 2)) for c in cands],
        "weight": weight,
        "chi2_nu": chi2_nu,                 # only meaningful with sigma
        "rms_rel_resid": float(np.sqrt(np.mean(rel ** 2))),
        "aicc": b["aicc"],
        "Rs": b["Rs"], "Rs_fixed": rs_fixed is not None, "L": b["L"],
        "R_pol": R_pol,
        "f_fit_hz": (float(f.min()), float(f.max())),
        "n_points": int(f.size),
        "dropped_mains_hz": dropped_mains,
        "freq": f, "Z_meas": Zf, "Z_fit": Zm,
        "res_pct": 100 * rel,
        "_model": (b["Rs"], b["L"], b["pa"], b["layout"]),
    }
    for name, a in zip(("fast", "slow"), arcs):
        kind, R_ion, R, tau, n = a
        out[f"R_{name}"], out[f"tau_{name}"], out[f"n_{name}"] = float(R), float(tau), float(n)
        if kind == "tlm":
            out[f"R_ion_{name}"] = float(R_ion)
    if len(arcs) == 1:
        out["R_slow"] = 0.0
        out["tau_slow"] = out["n_slow"] = float("nan")
    return out


def model_curve(fit: dict, freq) -> np.ndarray:
    """The fitted model on any frequency grid (e.g. a smooth line)."""
    Rs, L, pa, layout = fit["_model"]
    return z_model(Rs, L, pa, layout, freq)


# ---------------------------------------------------------------------------
# point 7: medians over all fitted segments
# ---------------------------------------------------------------------------
SUMMARY_KEYS = ("Rs", "R_fast", "tau_fast", "n_fast", "R_slow", "tau_slow",
                "n_slow", "R_pol", "rms_rel_resid", "chi2_nu")


def summarize(fits: dict) -> dict:
    """{key: (median, q25, q75, n)} over ALL fitted segments + counts."""
    good = [r for r in fits.values() if r.get("ok")]
    out = {"n_fitted": len(good),
           "n_clean": sum(1 for r in good if r["clean"]),
           "n_two_arc": sum(1 for r in good if r["n_arcs"] == 2),
           "n_tlm": sum(1 for r in good if r.get("fast_element") == "tlm")}
    for k in SUMMARY_KEYS:
        v = np.array([r.get(k, np.nan) for r in good], float)
        v = v[np.isfinite(v)]
        out[k] = ((float(np.median(v)), float(np.percentile(v, 25)),
                   float(np.percentile(v, 75)), int(v.size)) if v.size
                  else (np.nan, np.nan, np.nan, 0))
    return out


def arcs_from_params(params: dict, n_arcs: int) -> dict:
    """Method A (csv_pipeline) params -> the same fast/slow naming."""
    arcs = sorted(((params.get(f"R{k+1}", np.nan), params.get(f"tau{k+1}", np.nan),
                    params.get(f"n{k+1}", np.nan)) for k in range(n_arcs)),
                  key=lambda a: a[1])
    out = {"Rs": params.get("Rs", np.nan),
           "R_pol": float(sum(a[0] for a in arcs))}
    for name, a in zip(("fast", "slow"), arcs):
        out[f"R_{name}"], out[f"tau_{name}"], out[f"n_{name}"] = a
    if n_arcs == 1:
        out["R_slow"], out["tau_slow"], out["n_slow"] = 0.0, np.nan, np.nan
    return out
