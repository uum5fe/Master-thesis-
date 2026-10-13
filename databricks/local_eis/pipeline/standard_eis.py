"""
standard_eis.py
===============
R_ohmic, R_ct, R_mt and R_pol the standard way: an equivalent-circuit fit.

THE MODEL (the textbook PEM fuel-cell circuit)
----------------------------------------------
    Z(f) = R_ohmic + jωL + R_ct / (1 + (jωτ_ct)^n_ct) + R_mt / (1 + (jωτ_mt)^n_mt)

    R_ohmic   membrane + contact resistance (the high-frequency intercept)
    L         series inductance of cables and plate
    R_ct      charge-transfer arc (the faster one, τ_ct < τ_mt)
    R_mt      mass-transport arc (the slower one)
    n         CPE exponent of each arc (1 = ideal RC, < 1 = depressed arc)
    R_pol     = R_ct + R_mt

Each arc is an R in parallel with a constant-phase element (a "ZARC"),
written with its time constant τ = (R·Q)^(1/n). Its peak frequency is
f = 1/(2π·τ).

THE FIT (complex non-linear least squares, as ZView / Gamry Echem Analyst)
-------------------------------------------------------------------------
* Residuals: real and imaginary parts, each divided by |Z_measured| (modulus
  weighting -- every point counts by its relative error).
* Bounds: all R ≥ 0, L ≥ 0, 0.5 ≤ n ≤ 1, τ_mt at least 3× τ_ct (the arcs
  cannot swap or merge), and both arcs peak inside the measured band.
* Only the capacitive part is fitted: the low-frequency inductive loop at
  high current (Im Z > 0 below ~1 Hz) has no element in this circuit.
* Several starting points read off the spectrum (where −Im Z peaks, how the
  polarisation splits); the best fit is kept.
* Standard errors from the Jacobian at the optimum.

Also reported, model-free:
    R_lf       Re Z at the lowest measured frequency
    R_pol_lf   R_lf − R_ohmic  (the graphical polarisation resistance)

Units: whatever Z comes in (the pipeline uses Ω·cm²). No DRT is involved.
"""

from __future__ import annotations

import numpy as np

PARAM_NAMES = ("Rs", "L", "R1", "tau1", "n1", "R2", "tau2", "n2")
MIN_TAU_RATIO_DEC = np.log10(3.0)      # τ_mt ≥ 3·τ_ct
N_MIN, N_MAX = 0.5, 1.0
L_MAX = 1e-3                           # Ω·cm²·s -- generous; real L ~1e-7


def zarc(w, R, tau, n):
    """R ‖ CPE, written with its time constant."""
    return R / (1.0 + (1j * w * tau) ** n)


def z_model(p, freq) -> np.ndarray:
    """Z of R_ohmic + jωL + two ZARCs; p = (Rs, L, R1, tau1, n1, R2, tau2, n2)."""
    w = 2 * np.pi * np.asarray(freq, float)
    Rs, L, R1, t1, n1, R2, t2, n2 = p
    return Rs + 1j * w * L + zarc(w, R1, t1, n1) + zarc(w, R2, t2, n2)


def _to_p(x):
    """Fit vector -> parameters. x = (Rs, L, R1, log10 τ1, n1, R2, Δlog10 τ, n2)."""
    Rs, L, R1, lt1, n1, R2, d, n2 = x
    return np.array([Rs, L, R1, 10 ** lt1, n1, R2, 10 ** (lt1 + d), n2])


def graphical(freq, Z) -> dict:
    """Model-free numbers read straight off the spectrum."""
    f = np.asarray(freq, float)
    Z = np.asarray(Z, complex)
    o = np.argsort(f)
    f, Z = f[o], Z[o]
    out = {"R_lf": float(Z.real[0]) if f.size else np.nan,
           "f_lf_hz": float(f[0]) if f.size else np.nan,
           "R_hf_intercept": np.nan}
    # high-frequency real-axis crossing, only if the band reaches it
    for i in range(f.size - 1, 0, -1):
        a, b = Z.imag[i - 1], Z.imag[i]
        if f[i - 1] >= 100 and a < 0 <= b:
            w = a / (a - b)
            out["R_hf_intercept"] = float(Z.real[i - 1]
                                          + w * (Z.real[i] - Z.real[i - 1]))
            break
    return out


def hf_intercept(freq, Z, n_top: int = 6) -> tuple[float, str]:
    """R_ohmic from the high-frequency end, model-free.

    The high-frequency real-axis intercept: where Im Z crosses zero, if the
    band reaches it; otherwise Re Z extrapolated to Im Z = 0 along a straight
    line through the top `n_top` points (Re Z against Im Z). The second case
    is the usual one with the band ending at 2.4 kHz.
    Returns (value, how).
    """
    g = graphical(freq, Z)
    if np.isfinite(g["R_hf_intercept"]) and g["R_hf_intercept"] > 0:
        return g["R_hf_intercept"], "crossing"
    f = np.asarray(freq, float)
    Z = np.asarray(Z, complex)
    o = np.argsort(f)[-n_top:]
    x, y = Z.imag[o], Z.real[o]
    if o.size < 3 or np.ptp(x) <= 0:
        return float(np.real(Z[np.argmax(f)])), "top_point"
    b, a = np.polyfit(x, y, 1)                  # Re = a + b·Im
    R = float(a)
    lo, hi = float(np.min(y)) * 0.7, float(np.max(y)) * 1.05
    if not np.isfinite(R) or R <= 0 or not lo <= R <= hi:
        return float(np.real(Z[np.argmax(f)])), "top_point"
    return R, "extrapolated"


def capacitive_band(freq, Z) -> np.ndarray:
    """Mask of the points the circuit describes: everything above the
    low-frequency inductive loop. At high current the spectrum turns
    inductive below ~1 Hz (Im Z > 0); the two-arc circuit has no element for
    that loop, so those lowest points are left out of the fit (they are still
    in R_lf). Standard practice."""
    f = np.asarray(freq, float)
    Z = np.asarray(Z, complex)
    o = np.argsort(f)
    keep = np.ones(f.size, bool)
    for i in o:                       # from the lowest frequency upwards
        if -Z.imag[i] > 0:
            break
        keep[i] = False
    return keep


def _starts(f, Z):
    """Starting vectors read off the spectrum."""
    o = np.argsort(f)
    fs, Zs = f[o], Z[o]
    top = Zs[-3:] if Zs.size >= 3 else Zs
    Rs0 = max(float(np.min(top.real)), 1e-6)
    Rp0 = max(float(Zs.real[0]) - Rs0, 1e-5)
    im = -Zs.imag
    f_pk = fs[int(np.argmax(im))] if im.size else np.sqrt(fs[0] * fs[-1])
    # every arc peaks INSIDE the measured band: an arc placed above f_max
    # would trade off against R_ohmic (the band ends before the real axis)
    lt_lo = np.log10(1 / (2 * np.pi * fs[-1]))
    lt_hi = np.log10(1 / (2 * np.pi * fs[0]))
    starts = []
    for share in (0.3, 0.6, 0.85):
        for f_ct in (f_pk, 10 * f_pk, 100 * f_pk):
            lt1 = float(np.clip(np.log10(1 / (2 * np.pi * f_ct)), lt_lo,
                                lt_hi - MIN_TAU_RATIO_DEC - 0.1))
            for d in (1.0, 2.0):
                starts.append([Rs0, 1e-8, share * Rp0, lt1, 0.85,
                               (1 - share) * Rp0, d, 0.85])
    return starts, (lt_lo, lt_hi)


def fit_spectrum(freq, Z, sigma_rel=None, rs_fixed="hf") -> dict:
    """Fit the standard circuit to one spectrum.

    `sigma_rel` is accepted for interface compatibility and not used: the
    standard fit weights by |Z| (modulus weighting).

    rs_fixed="hf" (default): R_ohmic is first read from the high-frequency
    end (hf_intercept) and held fixed while the two arcs are fitted. With
    the band ending below the real-axis crossing, a free R_ohmic trades off
    against an arc starting above the band (R_ohmic fell to ~30 instead of
    ~58 mOhm*cm2 on 10 % of the 450 A segments); fixing it from the data is
    the standard two-step remedy. rs_fixed=None fits it free; a number
    fixes it to that value.

    Returns a dict:
        ok, params{Rs,L,R1,tau1,n1,R2,tau2,n2}, stderr{...}, n_arcs (2),
        n_points, weight, chi2_nu, rms_pct (relative residual),
        R_ohmic, R_ct, R_mt, R_pol (+ *_sd), f_ct_hz, f_mt_hz, tau_peak,
        R_lf, R_pol_lf, R_hf_intercept, verdict, Z_fit (at the data).
    """
    from scipy.optimize import least_squares

    f = np.asarray(freq, float)
    Z = np.asarray(Z, complex)
    ok = np.isfinite(f) & (f > 0) & np.isfinite(Z.real) & np.isfinite(Z.imag) & (np.abs(Z) > 0)
    f, Z = f[ok], Z[ok]
    g = graphical(f, Z)                       # from ALL points, loop included
    cap = capacitive_band(f, Z)
    n_loop = int((~cap).sum())
    f, Z = f[cap], Z[cap]
    if f.size < 8:
        return {"ok": False, "reason": f"{f.size} usable points (need 8)"}
    s = np.abs(Z)

    if rs_fixed == "hf":
        rs_val, rs_how = hf_intercept(f, Z)
    elif rs_fixed is None:
        rs_val, rs_how = np.nan, "fitted"
    else:
        rs_val, rs_how = float(rs_fixed), "given"
    starts, (lt_lo, lt_hi) = _starts(f, Z)
    if np.isfinite(rs_val):
        for x0 in starts:
            x0[0] = rs_val
            x0[2] = max(x0[2], 1e-6)
    # τ_mt = τ_ct·10^d must stay inside the band too: d ≤ lt_hi − lt_lo
    lo = [0.0, 0.0, 0.0, lt_lo, N_MIN, 0.0, MIN_TAU_RATIO_DEC, N_MIN]
    d_max = max(lt_hi - lt_lo, MIN_TAU_RATIO_DEC + 0.1)
    hi = [np.inf, L_MAX, np.inf, lt_hi - MIN_TAU_RATIO_DEC, N_MAX, np.inf,
          d_max, N_MAX]
    if np.isfinite(rs_val):                       # pin R_ohmic
        lo[0], hi[0] = rs_val * (1 - 1e-9), rs_val * (1 + 1e-9) + 1e-15

    def resid(x):
        d = z_model(_to_p(x), f) - Z
        return np.concatenate([d.real / s, d.imag / s])

    best = None
    for x0 in starts:
        x0 = np.clip(np.asarray(x0, float), np.array(lo) + 1e-12,
                     np.array(hi) - 1e-12)
        try:
            r = least_squares(resid, x0, bounds=(lo, hi), max_nfev=4000,
                              x_scale="jac")
        except Exception:                                  # noqa: BLE001
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return {"ok": False, "reason": "no start converged"}

    x = best.x
    p = _to_p(x)
    m, npar = 2 * f.size, len(x)
    dof = max(m - npar, 1)
    chi2_nu = float(2.0 * best.cost / dof)
    try:
        cov = np.linalg.pinv(best.jac.T @ best.jac) * chi2_nu
    except np.linalg.LinAlgError:
        cov = np.full((npar, npar), np.nan)
    var = np.clip(np.diag(cov), 0, None)
    se_x = np.sqrt(var)
    # standard errors in parameter space (delta method for the log taus)
    se = np.array([se_x[0], se_x[1], se_x[2],
                   p[3] * np.log(10) * se_x[3], se_x[4], se_x[5],
                   p[6] * np.log(10) * np.sqrt(var[3] + var[6] + 2 * cov[3, 6]),
                   se_x[7]])
    R_pol_sd = float(np.sqrt(max(var[2] + var[5] + 2 * cov[2, 5], 0.0)))

    Z_fit = z_model(p, f)
    rel = np.abs(Z_fit - Z) / s
    rms_pct = float(100 * np.sqrt(np.mean(rel ** 2)))
    Rs, L, R1, t1, n1, R2, t2, n2 = (float(v) for v in p)
    return {
        "ok": True,
        "params": dict(zip(PARAM_NAMES, (float(v) for v in p))),
        "stderr": dict(zip(PARAM_NAMES, (float(v) for v in se))),
        "n_arcs": 2,
        "n_points": int(f.size),
        "n_lf_loop_points": n_loop,
        "weight": "modulus",
        "chi2_nu": chi2_nu,
        "rms_pct": rms_pct,
        "R_ohmic": Rs,
        "R_ohmic_sd": (float(se[0]) if rs_how == "fitted" else
                       float(np.sqrt(np.mean(np.abs(Z_fit - Z)[np.argsort(f)[-6:]] ** 2)))),
        "R_ohmic_method": rs_how,
        "R_ct": R1, "R_ct_sd": float(se[2]),
        "R_mt": R2, "R_mt_sd": float(se[5]),
        "R_pol": R1 + R2, "R_pol_sd": R_pol_sd,
        "L": L,
        "f_ct_hz": float(1 / (2 * np.pi * t1)),
        "f_mt_hz": float(1 / (2 * np.pi * t2)),
        "tau_peak": t1,
        "R_lf": g["R_lf"],
        "R_pol_lf": g["R_lf"] - Rs,
        "R_hf_intercept": g["R_hf_intercept"],
        "verdict": ("good" if rms_pct <= 3 else
                    "acceptable" if rms_pct <= 8 else "poor"),
        "freq": f,
        "Z_fit": Z_fit,
    }


def choose_n_arcs(freq, Z, sigma_rel=None) -> dict:
    """Drop-in for csv_pipeline.choose_n_arcs: always the standard two-arc
    circuit, so the notebook's ECM views show the same fit the maps use."""
    return fit_spectrum(freq, Z, sigma_rel)
