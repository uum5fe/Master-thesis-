#!/usr/bin/env python3
"""
eis_measurement_model.py
========================
The stage that turns a scattered local-EIS spectrum into a clean one, without
any reference instrument and without cosmetic smoothing.

THE IDEA
--------
A spectrum that satisfies linearity, causality, stability and finiteness can be
written EXACTLY as a series of Voigt (RC) elements plus a series resistance and
a series inductance:

    Z(w) = R0 + jwL + SUM_k  R_k / (1 + jw tau_k)

with the tau_k FIXED and log-spaced.  This is Boukamp's lin-KK model, and the
measurement-model idea of Agarwal, Orazem & Garcia-Rubio: fit it, and the fit is
the best KK-compliant description of the data.  Three things then follow, and
all three are exactly what the raw plate spectra need:

1. FILTERING.  The model spans the manifold of physically realisable spectra.
   Random measurement noise is not on that manifold, so projecting the data onto
   it removes noise while preserving every real feature.  This is a projection,
   not a smoothing kernel -- it cannot invent an arc and it cannot flatten a
   real one, which a moving average or a spline would.

2. Rs WITHOUT EXTRAPOLATION BY EYE.  Every Voigt term vanishes as w -> inf, so

       R0 = Z(inf) = the high-frequency real-axis intercept

   is a FITTED PARAMETER constrained by every point in the spectrum, not the
   real part of one noisy point at f_max.  On the plate data this is the single
   largest reduction in Rs scatter, because at high frequency the cell-voltage
   reference is only ~1.5 mV and one point carries all the noise.

3. DELAY AND INDUCTANCE, SEPARATED.  An acquisition skew multiplies Z by
   exp(-jw dt): an ALL-PASS, which the Voigt+L basis cannot represent because
   that basis is minimum-phase.  So scanning dt and minimising the fit residual
   MEASURES the skew.  To first order a delay looks like a negative inductance
   (Z e^{-jw dt} ~ Z - jw dt Z, and at high frequency Z -> R0, giving
   -jw (dt R0)), so the two are degenerate at the top of the band alone.  They
   separate over a wide band because the delay term scales with |Z| -- large at
   low frequency -- while jwL does not.  Fit both together over the whole band.

WHAT THIS DOES NOT DO
---------------------
It does not make a bad measurement good.  If a segment's spectrum is not
KK-compliant, the residual stays large and the segment should be dropped, not
smoothed.  The residual is returned for exactly that purpose.

References
----------
Boukamp, J. Electrochem. Soc. 142 (1995) 1885           -- the lin-KK model
Schoenleber, Klotz, Ivers-Tiffee, Electrochim. Acta 131 (2014) 20  -- mu
Agarwal, Orazem, Garcia-Rubio, J. Electrochem. Soc. 139 (1992) 1917 and
    142 (1995) 4159                                     -- measurement models
Orazem & Tribollet, Electrochemical Impedance Spectroscopy, Wiley 2017
"""

from __future__ import annotations

import numpy as np

__all__ = ["fit_kk_model", "fit_delay_and_model", "fit_shared_delay",
           "clean_segment", "aggregate_asr"]


# ---------------------------------------------------------------------------
# 1. The linear model
# ---------------------------------------------------------------------------


def _design(freq: np.ndarray, taus: np.ndarray, with_L: bool) -> np.ndarray:
    """Complex design matrix [1, (jw), 1/(1+jw tau_k) ...]."""
    w = 2 * np.pi * freq
    cols = [np.ones_like(w, dtype=complex)]
    if with_L:
        cols.append(1j * w)
    for t in taus:
        cols.append(1.0 / (1.0 + 1j * w * t))
    return np.column_stack(cols)


def _weighted_lstsq(A: np.ndarray, Z: np.ndarray, w: np.ndarray,
                    ridge: float = 0.0, n_free: int = 1):
    """Least squares on the complex system, stacked into real form.

    Minimising sum_i w_i^2 |Z_i - (A p)_i|^2 with real coefficients p.
    Solved by SVD (lstsq) rather than the normal equations: the constant
    column sits next to a jw column whose magnitude spans four decades, and
    squaring that spread in A^T A costs half the available precision.
    """
    Ar = np.vstack([np.real(A) * w[:, None], np.imag(A) * w[:, None]])
    br = np.concatenate([np.real(Z) * w, np.imag(Z) * w])
    if ridge > 0:
        scale = ridge * np.sqrt(np.mean(np.sum(Ar ** 2, axis=0)))
        P = np.zeros((A.shape[1] - n_free, A.shape[1]))
        P[:, n_free:] = np.eye(A.shape[1] - n_free) * scale
        Ar = np.vstack([Ar, P])
        br = np.concatenate([br, np.zeros(P.shape[0])])
    p, *_ = np.linalg.lstsq(Ar, br, rcond=None)
    return p


def _mu(R: np.ndarray) -> float:
    """Schoenleber's mu: 1 - sum|R_neg| / sum R_pos.  Falls towards 0 as the
    RC set degenerates into fitting noise."""
    pos = R[R > 0].sum()
    neg = -R[R < 0].sum()
    return float(1.0 - neg / pos) if pos > 0 else 0.0


def fit_kk_model(freq, Z, sigma_rel=None, M=None, mu_crit=0.85,
                 with_L=None, tau_pad=0.0, ridge=1e-3) -> dict:
    """Fit  R0 + jwL + sum R_k/(1+jw tau_k)  with fixed log-spaced tau.

    Parameters
    ----------
    freq, Z      : the spectrum (Z in ohm*cm^2 here, but any unit works)
    sigma_rel    : per-point RELATIVE standard deviation, e.g. from the
                   sine-fit SNR.  None -> proportional weighting 1/|Z|, which
                   is the standard choice when the error structure is unknown.
    M            : number of RC elements; None -> chosen by the mu-criterion
    with_L       : None -> add the inductance column only if the data actually
                   shows an inductive tail (Im Z > 0 at the top of the band).
                   Forcing it onto a purely capacitive spectrum is what drives
                   the spurious negative R_k that corrupt mu.
    tau_pad      : widen the tau range beyond [1/w_max, 1/w_min], in decades.
                   KEEP THIS AT 0.  Padding is tempting -- it would let the
                   model describe processes just outside the window -- but a
                   tau far above 1/w_min makes 1/(1+jw tau) ~ 1 at every
                   measured frequency, i.e. a column identical to the constant
                   column.  R0 then trades against that element without
                   changing the fit, and the extracted Rs diverges.  With
                   tau_pad = 0 the slowest element still has w_min*tau = 1 and
                   the fastest w_max*tau = 1, so every column is distinguishable
                   inside the band.
    ridge        : Tikhonov weight on the R_k only (never on R0 or L),
                   relative to the mean column norm.  Suppresses the sign-
                   alternating pairs that appear once M exceeds what the data
                   supports, without biasing the intercept.

    Returns dict with Z_model, R0 (= Rs), L, Rp, residuals and diagnostics.

    NOTE ON WHAT Rs MEANS HERE.  R0 is Z(w -> inf) under the assumption that no
    relaxation faster than 1/w_max exists -- the same assumption every lin-KK
    implementation makes, and the reason the tau range stops at 1/w_max.  It is
    an extrapolation, but one constrained by every point in the spectrum rather
    than by the single noisiest one.
    """
    freq = np.asarray(freq, float)
    Z = np.asarray(Z, complex)
    ok = np.isfinite(freq) & (freq > 0) & np.isfinite(Z.real) & np.isfinite(Z.imag)
    freq, Z = freq[ok], Z[ok]
    if sigma_rel is not None:
        sigma_rel = np.asarray(sigma_rel, float)[ok]
    o = np.argsort(freq)
    freq, Z = freq[o], Z[o]
    if sigma_rel is not None:
        sigma_rel = sigma_rel[o]

    n = len(freq)
    if n < 6:
        return {"ok": False, "note": "too few points", "n": n}

    # weights: 1/(|Z| * sigma_rel) -> relative residuals, noisy points count less
    w = 1.0 / np.abs(Z)
    if sigma_rel is not None:
        s = np.clip(sigma_rel, 1e-4, 1.0)
        w = w / s

    if with_L is None:
        hi = Z[freq >= np.percentile(freq, 80)]
        with_L = bool(np.mean(np.imag(hi)) > 0)

    lo_tau = np.log10(1.0 / (2 * np.pi * freq.max())) - tau_pad
    hi_tau = np.log10(1.0 / (2 * np.pi * freq.min())) + tau_pad

    n_dec = max(1.0, np.log10(freq.max() / freq.min()))
    M_min = int(np.clip(round(4 * n_dec), 4, max(4, n - 2)))
    M_max = int(min(2 * n, 40))

    best = None
    for m in ([M] if M else range(4, M_max + 1)):
        taus = np.logspace(lo_tau, hi_tau, m)
        A = _design(freq, taus, with_L)
        p = _weighted_lstsq(A, Z, w, ridge=ridge, n_free=2 if with_L else 1)
        R = p[2:] if with_L else p[1:]
        mu = _mu(R)
        best = (m, taus, A, p, mu)
        if M is None and mu < mu_crit and m >= M_min:
            break

    m, taus, A, p, mu = best
    Z_model = A @ p
    R0 = float(p[0])
    L = float(p[1]) if with_L else 0.0
    Rk = p[2:] if with_L else p[1:]

    res = (Z - Z_model) / np.abs(Z)
    return {
        "ok": True,
        "freq": freq, "Z": Z, "Z_model": Z_model,
        "R0": R0, "Rs": R0, "L": L,
        "Rp": float(np.sum(Rk)),          # polarisation resolved in-band
        "R_k": Rk, "tau": taus, "M": m, "mu": mu, "with_L": bool(with_L),
        "res_re": np.real(res), "res_im": np.imag(res),
        "res_max": float(np.max(np.abs(np.concatenate([np.real(res), np.imag(res)])))),
        "res_rms": float(np.sqrt(np.mean(np.abs(res) ** 2))),
        "n": n,
    }


# ---------------------------------------------------------------------------
# 2. Delay, fitted jointly with the model
# ---------------------------------------------------------------------------


def fit_delay_and_model(freq, Z, sigma_rel=None, dt_range=200e-6,
                        n_grid=161, refine=True, **kw) -> dict:
    """Find the acquisition delay by minimising the KK-model residual.

    A delay is an all-pass; the Voigt+L basis is minimum-phase and cannot
    absorb one.  So the weighted residual of the fit, as a function of an
    applied correction exp(+j w dt), has a minimum at the true delay.  This is
    the same argument as the Z-HIT delay estimator, expressed directly in the
    KK basis, which is better conditioned inside a limited band.

    A coarse scan followed by a golden-section-style refinement; the cost is
    smooth in dt so this is safe.

    Returns the model dict of the best fit, with 'dt' added.
    """
    freq = np.asarray(freq, float)
    Z = np.asarray(Z, complex)
    w = 2 * np.pi * freq

    def cost(dt):
        r = fit_kk_model(freq, Z * np.exp(1j * w * dt), sigma_rel, **kw)
        return (r["res_rms"] if r.get("ok") else np.inf), r

    grid = np.linspace(-dt_range, dt_range, n_grid)
    costs = np.array([cost(dt)[0] for dt in grid])
    k = int(np.argmin(costs))
    dt_best = float(grid[k])

    if refine:                                   # local bisection refinement
        step = (grid[1] - grid[0]) / 2
        for _ in range(12):
            cands = [dt_best - step, dt_best + step]
            vals = [cost(d)[0] for d in cands]
            if min(vals) < costs[k]:
                j = int(np.argmin(vals))
                dt_best, costs[k] = cands[j], vals[j]
            step /= 2.0

    c0, _ = cost(0.0)
    _, best = cost(dt_best)
    best["dt"] = dt_best
    best["dt_us"] = dt_best * 1e6
    best["cost_at_zero"] = float(c0)
    best["cost_gain"] = float((c0 - best["res_rms"]) / c0) if c0 > 0 else 0.0
    best["phase_at_fmax_deg"] = float(abs(np.degrees(2 * np.pi * freq.max() * dt_best)))
    return best


def fit_shared_delay(spectra: dict, dt_range: float = 200e-6, n_grid: int = 161,
                     M_scan: int = 12, verbose: bool = True) -> dict:
    """ONE delay for a whole card, fitted from all its segments at once.

    A single spectrum does not determine the delay well.  With a few percent of
    noise on 40 points the cost surface around the true delay is flatter than
    the noise, and the per-segment estimate wanders by tens of microseconds --
    which then shows up as exactly the high-frequency phase fan-out that makes
    a local-EIS Nyquist plot unreadable.

    But the skew is a property of the ACQUISITION, not of the segment.  Summing
    the normalised cost curves of all segments on a card turns a shallow
    minimum into a sharp one, and the estimate improves as sqrt(n_segments).

    A fixed, modest M is used during the scan: the model order does not need to
    be optimal to locate the minimum, and re-running the mu-criterion inside a
    161-point scan for every segment would be a hundred times slower.

    spectra : {name: (freq, Z, sigma_rel or None)}
    """
    grid = np.linspace(-dt_range, dt_range, n_grid)
    total = np.zeros(n_grid)
    per_seg, n_used = {}, 0

    for name, item in spectra.items():
        f, Z = item[0], item[1]
        s_rel = item[2] if len(item) > 2 else None
        f = np.asarray(f, float)
        Z = np.asarray(Z, complex)
        if len(f) < 8:
            continue
        w = 2 * np.pi * f
        c = np.empty(n_grid)
        for i, dt in enumerate(grid):
            r = fit_kk_model(f, Z * np.exp(1j * w * dt), s_rel, M=M_scan)
            c[i] = r["res_rms"] if r.get("ok") else np.inf
        if not np.all(np.isfinite(c)) or c.max() <= 0:
            continue
        per_seg[name] = float(grid[int(np.argmin(c))])
        total += c / c.max()            # normalise: segments differ in |Z|
        n_used += 1

    if n_used == 0:
        return {"dt": 0.0, "n_segments": 0, "applied": False}

    k = int(np.argmin(total))
    dt = float(grid[k])
    zero = total[n_grid // 2]
    gain = float((zero - total[k]) / zero) if zero > 0 else 0.0
    spread = float(np.std(list(per_seg.values())))

    out = {"dt": dt, "dt_us": dt * 1e6, "n_segments": n_used,
           "cost_gain": gain, "per_segment_spread_us": spread * 1e6,
           "per_segment": per_seg, "grid": grid, "cost": total,
           "applied": bool(gain > 0.02)}
    if verbose:
        print(f"    shared delay {dt*1e6:+.1f} us from {n_used} segments "
              f"(cost -{100*gain:.0f} %, per-segment spread "
              f"{spread*1e6:.1f} us)")
    return out


# ---------------------------------------------------------------------------
# 3. One segment, end to end
# ---------------------------------------------------------------------------


def clean_segment(freq, Z, snr_db=None, *, deskew=True, remove_L=True,
                  dt_fixed=None, min_snr_db=6.0, max_res=0.05,
                  asr_bounds=(1.0, 500.0), dt_range=200e-6) -> dict:
    """Full cleaning chain for one segment.  Z in ohm*cm^2.

    1. drop points below the SNR floor and outside physical ASR bounds
    2. fit delay + KK model (or apply a delay supplied from the card median)
    3. optionally remove the fitted series inductance from the OUTPUT curve,
       so the Nyquist closes onto the real axis instead of hooking upward --
       jwL is cable and plate, not electrochemistry
    4. return both the corrected raw points and the KK-projected curve

    Verdict:
      'ok'        residual small, use it
      'noisy'     residual large -- KK-inconsistent, do not trust the shape
      'rejected'  too few usable points
    """
    freq = np.asarray(freq, float)
    Z = np.asarray(Z, complex)
    keep = np.isfinite(freq) & np.isfinite(Z.real) & np.isfinite(Z.imag)

    sigma_rel = None
    if snr_db is not None:
        snr_db = np.asarray(snr_db, float)
        keep &= np.isfinite(snr_db) & (snr_db >= min_snr_db)
        with np.errstate(over="ignore"):
            sigma_rel = 10.0 ** (-np.clip(snr_db, 0, 80) / 20.0)

    Zasr = np.real(Z) * 1000.0
    keep &= (Zasr > 0) & (Zasr >= asr_bounds[0]) & (Zasr <= asr_bounds[1])

    n_drop = int((~keep).sum())
    f, z = freq[keep], Z[keep]
    s = sigma_rel[keep] if sigma_rel is not None else None
    if len(f) < 8:
        return {"verdict": "rejected", "n_used": len(f), "n_dropped": n_drop}

    if dt_fixed is not None:
        m = fit_kk_model(f, z * np.exp(1j * 2 * np.pi * f * dt_fixed), s)
        m["dt"] = float(dt_fixed)
        m["dt_us"] = dt_fixed * 1e6
    elif deskew:
        m = fit_delay_and_model(f, z, s, dt_range=dt_range)
    else:
        m = fit_kk_model(f, z, s)
        m["dt"] = 0.0
        m["dt_us"] = 0.0
    if not m.get("ok"):
        return {"verdict": "rejected", "n_used": len(f), "n_dropped": n_drop}

    w = 2 * np.pi * m["freq"]
    Z_corr = m["Z"]                                   # deskewed raw points
    Z_out = m["Z_model"].copy()                       # KK projection
    if remove_L and m["L"] != 0.0:
        Z_corr = Z_corr - 1j * w * m["L"]
        Z_out = Z_out - 1j * w * m["L"]

    m.update({
        "verdict": "ok" if m["res_rms"] <= max_res else "noisy",
        "Z_corrected": Z_corr,
        "Z_clean": Z_out,
        "n_used": len(f), "n_dropped": n_drop,
    })
    return m


# ---------------------------------------------------------------------------
# 4. Cell-level aggregate (replaces the Gamry curve)
# ---------------------------------------------------------------------------


def aggregate_asr(freq, Z_by_seg: dict, areas: dict, A_cell: float) -> np.ndarray:
    """Z_cell^ASR = A_cell / sum_s (A_s / Z_s^ASR).

    Segments sit in parallel across one cell voltage, so the cell spectrum is
    the AREA-WEIGHTED HARMONIC MEAN of the local ones -- not the arithmetic
    mean.  Being harmonic it is dominated by the low-impedance segments, which
    is the mathematical statement of why an integral measurement hides local
    faults: a flooded segment has high Z, contributes little admittance and
    barely moves the cell curve.

    This curve is what a cell-level instrument would measure.  It is the
    replacement for the Gamry overlay: when a reference exists it is a
    validation, and when it does not, it is still the correct cell spectrum.
    """
    Y = np.zeros(len(freq), dtype=complex)
    A_used = 0.0
    for s, z in Z_by_seg.items():
        a = areas.get(s)
        if a is None or z is None or len(z) != len(freq):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            Y += np.where(np.isfinite(z) & (z != 0), a / z, 0.0)
        A_used += a
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(Y != 0, A_used / Y, np.nan)


# ---------------------------------------------------------------------------
# self-test: does it actually reduce scatter?
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(3)
    f = np.logspace(0, np.log10(4000), 40)
    w = 2 * np.pi * f

    Rs_true, Rs_naive, Rs_model = [], [], []
    for k in range(40):
        Rs = 0.060 + 0.004 * rng.standard_normal()        # ohm cm^2, 60 mOhm cm2
        Rct = 0.30 + 0.03 * rng.standard_normal()
        tau = 0.02 * (1 + 0.1 * rng.standard_normal())
        L = 2e-7
        Z = Rs + 1j * w * L + Rct / (1 + 1j * w * tau) ** 0.85
        Z = Z * np.exp(-1j * w * 55e-6)                   # acquisition skew
        Z = Z * (1 + 0.04 * (rng.standard_normal(len(f))
                             + 1j * rng.standard_normal(len(f))))   # 4 % noise

        Rs_true.append(Rs * 1000)
        Rs_naive.append(np.real(Z[np.argmax(f)]) * 1000)  # one HF point
        r = clean_segment(f, Z, snr_db=np.full(len(f), 28.0))
        Rs_model.append(r["Rs"] * 1000 if r["verdict"] != "rejected" else np.nan)

    Rs_true = np.array(Rs_true)
    Rs_naive = np.array(Rs_naive)
    Rs_model = np.array(Rs_model)

    def report(name, v):
        e = v - Rs_true
        print(f"  {name:22s} mean {np.nanmean(v):7.2f}  sd {np.nanstd(v):6.2f}"
              f"   bias {np.nanmean(e):+6.2f}  rms error {np.sqrt(np.nanmean(e**2)):6.2f}"
              f"   spread {np.nanmax(v)/np.nanmin(v):5.2f}x")

    print("Rs across 40 synthetic segments  [mOhm cm2], true sd = "
          f"{Rs_true.std():.2f}")
    report("true", Rs_true)
    report("Re Z at f_max", Rs_naive)
    report("KK model R0", Rs_model)
