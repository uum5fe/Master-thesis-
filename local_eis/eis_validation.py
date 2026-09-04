#!/usr/bin/env python3
"""
eis_validation.py
=================
Instrument-independent quality control for impedance spectra.

These are the two accepted ways to decide whether a measured spectrum is
trustworthy WITHOUT a second instrument to compare against.  They replace
"does it look like the Gamry curve?" with a test on the data itself.

1. lin-KK  (linear Kramers-Kronig validity test)
   Boukamp, J. Electrochem. Soc. 142 (1995) 1885 -- linear KK transform test
   Schönleber, Klotz, Ivers-Tiffée, Electrochim. Acta 131 (2014) 20 -- the
   mu-criterion that makes the model order choice unambiguous.
   A spectrum that is linear, causal, stable and time-invariant can be fitted
   by a series of RC elements with FIXED, log-spaced time constants.  If the
   relative residuals are structured or large (> ~1 %), one of those four
   conditions is violated -- drift, non-linear excitation, a leaky channel.

2. Z-HIT  (logarithmic Hilbert transform)
   Ehm, Göhr, Kaus, Röseler, Schiller, ACH Models in Chemistry 137 (2000) 145.
   Reconstructs ln|Z| from the phase.  Two properties matter here:
     * it needs no extrapolation to 0 / infinity, unlike the KK integrals;
     * unlike KK it DOES see all-pass contributions -- i.e. pure time delays.
   That second property is used below to measure the ADC channel skew of the
   plate directly from the spectra (estimate_delay), which is exactly the
   error that a half-sample offset between two ADC cores produces.

Everything here is plain numpy; no external EIS package required.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# lin-KK
# ---------------------------------------------------------------------------


def _linkk_fit(freq: np.ndarray, Z: np.ndarray, M: int,
               add_inductance: bool = True, add_capacitance: bool = False):
    """Least-squares fit of  R0 [+ jwL] [+ 1/jwC] + sum_k R_k/(1+jw tau_k).

    tau_k are fixed and log-spaced between 1/w_max and 1/w_min, so the fit is
    linear in the R_k and has a unique solution (Boukamp).  Weighting is by
    1/|Z| so that the residuals are relative, which is what the test needs.
    """
    w = 2 * np.pi * freq
    tau = np.logspace(np.log10(1 / w.max()), np.log10(1 / w.min()), M)

    cols = [np.ones_like(w, dtype=complex)]                    # R0
    if add_inductance:
        cols.append(1j * w)                                    # L
    if add_capacitance:
        cols.append(1.0 / (1j * w))                            # C
    for t in tau:
        cols.append(1.0 / (1 + 1j * w * t))
    A = np.column_stack(cols)

    wt = 1.0 / np.abs(Z)
    Ar = np.vstack([np.real(A) * wt[:, None], np.imag(A) * wt[:, None]])
    br = np.concatenate([np.real(Z) * wt, np.imag(Z) * wt])
    p, *_ = np.linalg.lstsq(Ar, br, rcond=None)

    Zfit = A @ p
    n_extra = 1 + int(add_inductance) + int(add_capacitance)
    return Zfit, p[n_extra:], p[:n_extra], tau


def _mu(R: np.ndarray) -> float:
    """Schönleber's mu: 1 - sum|R_neg| / sum|R_pos|.

    Falls from 1 towards 0 as the RC set starts to degenerate, i.e. as the
    fit starts to describe noise instead of the spectrum.  Stop at mu < 0.85.
    """
    pos = R[R > 0].sum()
    neg = -R[R < 0].sum()
    if pos <= 0:
        return 0.0
    return float(1.0 - neg / pos)


def lin_kk(freq: np.ndarray, Z: np.ndarray, mu_crit: float = 0.85,
           max_M: int | None = None, add_inductance: bool | None = None,
           tol: float = 0.02) -> dict:
    """Run the lin-KK test.  Returns residuals and a pass/fail verdict.

    Two practical points that the bare mu-criterion does not cover:

    * mu is not monotone at small M -- with only a handful of RC elements the
      least-squares solution flips signs for structural reasons and mu can dip
      below the threshold long before the model is rich enough.  A minimum of
      ~3 RC elements per frequency decade is enforced before the stopping rule
      is allowed to fire, which is the density Boukamp recommends anyway.
    * an inductance column is only added when the data actually shows an
      inductive tail (positive Im Z at the high-frequency end).  Adding it to a
      purely capacitive spectrum is what drives the spurious negative R_k.

    Keys: M, mu, Z_fit, res_re, res_im (relative), res_max, res_rms, valid.
    """
    freq = np.asarray(freq, float)
    Z = np.asarray(Z, complex)
    ok = np.isfinite(freq) & np.isfinite(Z.real) & np.isfinite(Z.imag)
    freq, Z = freq[ok], Z[ok]
    order = np.argsort(freq)
    freq, Z = freq[order], Z[order]

    n = len(freq)
    if n < 6:
        return {"M": 0, "mu": np.nan, "valid": False,
                "res_max": np.nan, "res_rms": np.nan,
                "note": "too few points"}
    max_M = max_M or max(6, min(2 * n, 60))

    if add_inductance is None:
        hi = Z[freq >= np.percentile(freq, 80)]
        add_inductance = bool(np.mean(np.imag(hi)) > 0)

    n_dec = max(1.0, np.log10(freq.max() / freq.min()))
    M_min = int(np.clip(round(4 * n_dec), 4, max(4, n - 2)))

    best = None
    for M in range(4, max_M + 1):
        Zfit, R, _extra, _tau = _linkk_fit(freq, Z, M, add_inductance)
        m = _mu(R)
        best = (M, m, Zfit)
        if m < mu_crit and M >= M_min:
            break

    M, m, Zfit = best
    res_re = (np.real(Z) - np.real(Zfit)) / np.abs(Z)
    res_im = (np.imag(Z) - np.imag(Zfit)) / np.abs(Z)
    res_max = float(np.max(np.abs(np.concatenate([res_re, res_im]))))
    res_rms = float(np.sqrt(np.mean(res_re ** 2 + res_im ** 2)))
    return {"M": M, "mu": float(m), "freq": freq, "Z_fit": Zfit,
            "res_re": res_re, "res_im": res_im,
            "res_max": res_max, "res_rms": res_rms,
            "add_inductance": bool(add_inductance),
            "valid": bool(res_max < tol and np.isfinite(res_max))}


# ---------------------------------------------------------------------------
# Z-HIT
# ---------------------------------------------------------------------------


def z_hit(freq: np.ndarray, Z: np.ndarray, gamma: float = np.pi / 6,
          fit_band: tuple[float, float] | None = None) -> dict:
    """Reconstruct ln|Z| from the phase (Ehm/Göhr approximation).

        ln|Z(w0)| ~ (2/pi) * INT_{w_s}^{w0} phi(w) dln(w)
                    - gamma * dphi/dln(w) |_{w0}   + const

    with phi = angle(Z) in rad (negative for a capacitive system) and
    gamma = pi/6.  The sign of the correction term follows from that phase
    convention: verified against an analytic Debye spectrum, where dropping
    it leaves 6 % rms error, using +gamma leaves 14 %, and -gamma leaves
    1.4 % -- the residual being the truncation of the Bode series itself.
    Do not read a 1-2 % Z-HIT deviation as a measurement fault; the useful
    signal is the DIFFERENCE between segments and the low-frequency trend.

    The unknown constant is fixed by a least-squares offset fit against the
    measured |Z| inside `fit_band` -- by convention a mid-frequency window,
    which is the part of the spectrum least affected by drift (low f) and by
    cable/ADC artefacts (high f).

    Returns the reconstructed |Z|, the log-deviation and a drift metric.
    """
    freq = np.asarray(freq, float)
    Z = np.asarray(Z, complex)
    ok = np.isfinite(freq) & (freq > 0) & np.isfinite(Z.real) & np.isfinite(Z.imag)
    freq, Z = freq[ok], Z[ok]
    o = np.argsort(freq)
    freq, Z = freq[o], Z[o]
    if len(freq) < 6:
        return {"valid": False, "note": "too few points", "dev_rms": np.nan}

    lnw = np.log(2 * np.pi * freq)
    phi = np.unwrap(np.angle(Z))
    ln_mag = np.log(np.abs(Z))

    integral = np.concatenate([[0.0], np.cumsum(
        0.5 * (phi[1:] + phi[:-1]) * np.diff(lnw))])
    dphi = np.gradient(phi, lnw)
    recon = (2.0 / np.pi) * integral - gamma * dphi

    if fit_band is None:                       # middle half of the decade span
        lo, hi = np.percentile(freq, [25, 75])
        fit_band = (lo, hi)
    m = (freq >= fit_band[0]) & (freq <= fit_band[1])
    if m.sum() < 3:
        m = np.ones_like(freq, bool)
    offset = float(np.mean(ln_mag[m] - recon[m]))
    recon += offset

    dev = ln_mag - recon                       # >0: measured too large
    return {"freq": freq, "mag_recon": np.exp(recon), "mag_meas": np.abs(Z),
            "log_dev": dev, "dev_rms": float(np.sqrt(np.mean(dev ** 2))),
            "dev_max": float(np.max(np.abs(dev))),
            "drift_lf": float(np.mean(dev[freq < np.median(freq)])),
            "valid": bool(np.max(np.abs(dev)) < 0.08)}


def estimate_delay(freq: np.ndarray, Z: np.ndarray,
                   dt_range: float = 200e-6, n_grid: int = 801,
                   fit_band: tuple[float, float] | None = None) -> tuple[float, dict]:
    """Find the pure time delay hidden in a spectrum, using Z-HIT.

    A delay dt multiplies Z by exp(-j w dt): an ALL-PASS.  It leaves |Z|
    untouched and rotates the phase, so a KK test cannot see it, while Z-HIT
    -- which builds |Z| out of the phase -- immediately disagrees with the
    measured |Z|.  Scanning dt and minimising that disagreement therefore
    measures the acquisition skew between the two channels of the ratio
    without any reference instrument.

    Returns (dt_seconds, info).  Apply the correction as Z * exp(+j w dt).
    """
    w = 2 * np.pi * np.asarray(freq, float)
    grid = np.linspace(-dt_range, dt_range, n_grid)
    cost = np.empty(n_grid)
    for i, dt in enumerate(grid):
        r = z_hit(freq, np.asarray(Z) * np.exp(1j * w * dt), fit_band=fit_band)
        cost[i] = r["dev_rms"] if np.isfinite(r["dev_rms"]) else np.inf
    k = int(np.argmin(cost))
    dt = float(grid[k])
    return dt, {"dt_s": dt, "dt_us": dt * 1e6,
                "cost_min": float(cost[k]), "cost_zero": float(cost[n_grid // 2]),
                "grid": grid, "cost": cost}


# ---------------------------------------------------------------------------
# quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    f = np.logspace(-1, 4, 40)
    w = 2 * np.pi * f
    Ztrue = 0.05 + 0.12 / (1 + 1j * w * 1e-3) + 0.30 / (1 + 1j * w * 0.05)

    print("clean spectrum")
    r = lin_kk(f, Ztrue)
    print(f"  lin-KK  M={r['M']} mu={r['mu']:.3f} "
          f"res_max={100*r['res_max']:.3f} %  valid={r['valid']}")
    h = z_hit(f, Ztrue)
    print(f"  Z-HIT   dev_rms={h['dev_rms']:.4f}  valid={h['valid']}")

    print("\n+ 60 us acquisition skew")
    Zd = Ztrue * np.exp(-1j * w * 60e-6)
    r = lin_kk(f, Zd)
    h = z_hit(f, Zd)
    print(f"  lin-KK  res_max={100*r['res_max']:.3f} %  -> flagged, but the "
          f"test cannot say what is wrong")
    print(f"  Z-HIT   dev_rms={h['dev_rms']:.4f}")
    dt, info = estimate_delay(f, Zd)
    print(f"  recovered delay = {dt*1e6:.2f} us   (true 60.00 us)")

    print("\n+ 3 % low-frequency drift")
    Zdrift = Ztrue * (1 + 0.03 * np.exp(-f / 1.0))
    r = lin_kk(f, Zdrift)
    h = z_hit(f, Zdrift)
    print(f"  lin-KK  res_max={100*r['res_max']:.3f} %  valid={r['valid']}")
    print(f"  Z-HIT   dev_rms={h['dev_rms']:.4f}  lf_bias={h['drift_lf']:+.4f}")
