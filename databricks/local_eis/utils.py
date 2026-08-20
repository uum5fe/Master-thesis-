#!/usr/bin/env python3
"""
utils.py
========
Everything shared by more than one layer: estimators, uncertainty, small
numerics, logging and IO.  No layer defines its own copy of any of this.

WHAT IS NEW HERE (and why)
--------------------------
1. `fit7_joint` -- the SEVEN-PARAMETER two-channel sine fit.

   The old pipeline estimated the impedance by running two independent
   three-parameter fits, one on the cell-voltage reference and one on the
   segment, and dividing the phasors.  Each fit assumed the frequency it was
   handed.  Ramos & Serra (Measurement 41 (2008) 135) showed that when two
   channels are known to share a frequency -- which is exactly the case here,
   because one generator drives both -- estimating that common frequency from
   BOTH records and both amplitudes at once reduces the standard deviation of
   the PHASE DIFFERENCE by about half relative to two independent
   four-parameter fits, and yields a smaller Cramer-Rao bound on both the
   frequency and the phase difference (Ramos, Janeiro & Radil, Measurement 42
   (2009) 1370).

   The phase difference IS the impedance phase.  Halving its standard
   deviation is therefore a direct, first-principles halving of the noise on
   arg Z -- and arg Z is where the whole high-frequency defect lives, because
   |Z| is nearly flat up there while the phase carries all of the information
   and all of the skew error.

2. `crlb_phasor` -- honest per-point uncertainty.

   The old code weighted the KK fit by `10 ** (-snr/20)`, a monotone function
   of SNR with no units and no derivation.  Rife & Boorstyn (IEEE Trans. Inf.
   Theory 20 (1974) 591) give the actual Cramer-Rao bounds for a single tone
   in white Gaussian noise.  For N samples at signal-to-noise ratio
   gamma = A^2/(2 sigma^2):

       var(A)/A^2  >=  1 / (N * gamma)
       var(phi)    >=  1 / (N * gamma)          (phase, known frequency)
       var(phi)    >=  ~ 6 / (N * gamma)        (phase, frequency also fitted)

   Those are real variances in real units, they propagate correctly through
   the ratio Z = K U / u, and they are what the measurement model should be
   weighted by.  Using them means a 2 dB point and a 30 dB point are combined
   with the weight the physics says, not the weight a rule of thumb says.

3. `unwrap_phase_monotone`, `all_pass_residual` -- the tools the skew
   correction needs, kept out of silver.py so they can be unit-tested.

Everything is plain numpy/scipy.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ===========================================================================
# 1. LOGGING
# ===========================================================================

_LOGGER_NAME = "eis"
_RULE = "\u2550" * 75


def get_logger(verbose: bool = True) -> logging.Logger:
    log = logging.getLogger(_LOGGER_NAME)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(h)
    log.setLevel(logging.INFO if verbose else logging.WARNING)
    log.propagate = False
    return log


def banner(text: str, log: logging.Logger | None = None) -> None:
    log = log or get_logger()
    log.info(f"\n{_RULE}\n  {text}\n{_RULE}")


def section(text: str, log: logging.Logger | None = None) -> None:
    (log or get_logger()).info(f"\n\u2500\u2500\u2500 {text} " + "\u2500" * max(0, 60 - len(text)))


@contextmanager
def timed(text: str, log: logging.Logger | None = None):
    log = log or get_logger()
    t0 = time.time()
    yield
    log.info(f"    [{text}: {time.time() - t0:.1f} s]")


# ===========================================================================
# 2. SINE FITTING  (IEEE Std 1057-2017)
# ===========================================================================


def _design3(t: np.ndarray, f: float, ramp: bool) -> np.ndarray:
    """[cos, sin, 1] and optionally a centred ramp column.

    The ramp is not cosmetic.  An operating cell drifts over the tens of
    seconds a low-frequency dwell takes.  A ramp is broadband; its leakage
    into the fundamental biases the phasor and inflates the stationarity
    metric until every low-frequency step is thrown out.  Least squares
    projects it out exactly, which a high-pass filter cannot do without
    distorting the phase of the very points at issue.
    """
    w = 2 * np.pi * f
    cols = [np.cos(w * t), np.sin(w * t), np.ones_like(t)]
    if ramp:
        cols.append(t - t.mean())
    return np.column_stack(cols)


def fit3(y: np.ndarray, fs: float, f: float, detrend: bool = True
         ) -> tuple[complex, float, float]:
    """Three-parameter fit  y ~ a cos(wt) + b sin(wt) + c  at KNOWN f.

    Returns (phasor, residual_rms, snr_db), phasor under exp(+j w t), so
    y = Re{A e^{jwt}} + c and A = a - j b.

    This is the least-squares optimum.  A lock-in is the same estimator only
    when the window holds a whole number of cycles, which a stepped sweep
    never guarantees.
    """
    y = np.asarray(y, float)
    n = len(y)
    if n < 8:
        return complex("nan"), np.nan, np.nan
    t = np.arange(n) / fs
    D = _design3(t, f, detrend)
    p, *_ = np.linalg.lstsq(D, y, rcond=None)
    resid = y - D @ p
    A = complex(p[0], -p[1])
    r_rms = float(np.sqrt(np.mean(resid ** 2)))
    sig = abs(A) / np.sqrt(2.0)
    snr = 20 * np.log10(sig / r_rms) if r_rms > 0 else np.inf
    return A, r_rms, float(snr)


def fit4(y: np.ndarray, fs: float, f0: float, n_iter: int = 12,
         detrend: bool = True) -> tuple[float, complex, float, float]:
    """Four-parameter fit: frequency solved for as well (IEEE Std 1057).

    Each iteration linearises around the current omega by adding the column
    t*(-a sin + b cos), solves for the increment and updates.  The increment
    is clamped to 5 % so a bad start value cannot walk the fit onto a
    neighbouring step of the sweep.

    Returns (f_hat, phasor, residual_rms, snr_db).
    """
    y = np.asarray(y, float)
    n = len(y)
    if n < 16:
        return f0, complex("nan"), np.nan, np.nan
    t = np.arange(n) / fs
    w = 2 * np.pi * f0
    a, b = 1.0, 0.0
    for _ in range(n_iter):
        c_, s_ = np.cos(w * t), np.sin(w * t)
        cols = [c_, s_, np.ones(n)]
        if detrend:
            cols.append(t - t.mean())
        cols.append(t * (-a * s_ + b * c_))
        D = np.column_stack(cols)
        p, *_ = np.linalg.lstsq(D, y, rcond=None)
        a, b = p[0], p[1]
        dw = float(np.clip(p[-1], -0.05 * w, 0.05 * w))
        w_new = w + dw
        if w_new <= 0:
            break
        conv = abs(dw) / w < 1e-9
        w = w_new
        if conv:
            break
    f_hat = w / (2 * np.pi)
    A, r_rms, snr = fit3(y, fs, f_hat, detrend)
    return float(f_hat), A, r_rms, snr


@dataclass
class JointFit:
    """Result of the seven-parameter two-channel fit."""
    freq: float
    A_ref: complex          # phasor of the reference channel (cell voltage)
    A_sig: complex          # phasor of the signal channel (segment)
    snr_ref_db: float
    snr_sig_db: float
    n: int
    iters: int
    converged: bool

    @property
    def ratio(self) -> complex:
        """A_ref / A_sig -- the quantity the impedance is built from."""
        return self.A_ref / self.A_sig if self.A_sig != 0 else complex("nan")

    @property
    def phase_diff_rad(self) -> float:
        return float(np.angle(self.ratio))


def fit7_joint(y_ref: np.ndarray, y_sig: np.ndarray, fs: float, f0: float,
               n_iter: int = 12, tol: float = 1e-9, detrend: bool = True
               ) -> JointFit:
    """Seven-parameter sine fit of two channels sharing ONE frequency.

    Model, for k in {ref, sig}:

        y_k[i] ~ A_k cos(w t_i) + B_k sin(w t_i) + C_k  [+ D_k (t_i - tbar)]

    with a single w common to both.  Stacking the two records into one
    least-squares problem and adding the linearisation column for dw gives a
    2N-row system whose solution updates all seven (nine with ramps)
    parameters at once.

    WHY THIS MATTERS HERE
    ---------------------
    One generator drives the cell, so both channels genuinely share a
    frequency.  Fitting them separately throws that constraint away and lets
    the two estimated frequencies differ; the difference does not cancel in
    the ratio, it shows up as extra variance on arg(A_ref/A_sig).  Ramos &
    Serra report ~50 % lower standard deviation on the phase difference, and
    Ramos et al. (2009) show a smaller Cramer-Rao bound on both the frequency
    and the phase difference than two independent four-parameter fits.  Since
    arg Z = arg(A_ref) - arg(A_sig), that is a direct halving of the noise on
    the impedance phase -- the quantity that carries the high-frequency arc.

    The estimator degrades gracefully: if one channel is pure noise the other
    still drives w, which is precisely the case for a weak segment at the top
    of the band.
    """
    y_ref = np.asarray(y_ref, float)
    y_sig = np.asarray(y_sig, float)
    n = min(len(y_ref), len(y_sig))
    if n < 16:
        return JointFit(f0, complex("nan"), complex("nan"),
                        np.nan, np.nan, n, 0, False)
    y_ref, y_sig = y_ref[:n], y_sig[:n]
    t = np.arange(n) / fs
    tc = t - t.mean()
    w = 2 * np.pi * f0
    zeros = np.zeros(n)

    a_r, b_r, a_s, b_s = 1.0, 0.0, 1.0, 0.0
    converged = False
    it = 0
    for it in range(1, n_iter + 1):
        c_, s_ = np.cos(w * t), np.sin(w * t)
        # per-channel columns are block-diagonal; the dw column is shared
        base_r = [c_, s_, np.ones(n)] + ([tc] if detrend else [])
        base_s = [c_, s_, np.ones(n)] + ([tc] if detrend else [])
        nb = len(base_r)
        top = np.column_stack(base_r + [zeros] * nb
                              + [t * (-a_r * s_ + b_r * c_)])
        bot = np.column_stack([zeros] * nb + base_s
                              + [t * (-a_s * s_ + b_s * c_)])
        D = np.vstack([top, bot])
        rhs = np.concatenate([y_ref, y_sig])
        p, *_ = np.linalg.lstsq(D, rhs, rcond=None)
        a_r, b_r = p[0], p[1]
        a_s, b_s = p[nb], p[nb + 1]
        dw = float(np.clip(p[-1], -0.05 * w, 0.05 * w))
        w_new = w + dw
        if w_new <= 0:
            break
        rel = abs(dw) / w
        w = w_new
        if rel < tol:
            converged = True
            break

    f_hat = w / (2 * np.pi)
    # final three-parameter fits at the agreed frequency give clean residuals
    A_r, _, snr_r = fit3(y_ref, fs, f_hat, detrend)
    A_s, _, snr_s = fit3(y_sig, fs, f_hat, detrend)
    return JointFit(float(f_hat), A_r, A_s, float(snr_r), float(snr_s),
                    n, it, converged)


def harmonic_distortion(y: np.ndarray, fs: float, f: float) -> float:
    """THD from the 2nd and 3rd harmonic -- the linearity check.

    A galvanostatic EIS excitation must stay in the linear regime.  If the
    amplitude is too large for the local operating point the response clips
    into harmonics and the fundamental-only estimate silently becomes a
    describing function instead of an impedance.
    Giner-Sanz, Ortega & Perez-Herranz, Electrochim. Acta 186 (2015) 598.
    """
    y = np.asarray(y, float)
    n = len(y)
    if n < 32 or 3 * f > 0.45 * fs:
        return np.nan
    t = np.arange(n) / fs
    cols = [np.ones(n), t - t.mean()]
    for k in (1, 2, 3):
        cols += [np.cos(2 * np.pi * k * f * t), np.sin(2 * np.pi * k * f * t)]
    D = np.column_stack(cols)
    p, *_ = np.linalg.lstsq(D, y, rcond=None)
    a1 = np.hypot(p[2], p[3])
    a2 = np.hypot(p[4], p[5])
    a3 = np.hypot(p[6], p[7])
    return float(np.hypot(a2, a3) / a1) if a1 > 0 else np.nan


def stationarity(y: np.ndarray, fs: float, f: float, n_sub: int = 3) -> float:
    """Spread of the phasor over `n_sub` sub-windows -- the time-invariance
    check.  Each sub-fit uses its own t=0, so the phasors are rotated back to
    a common origin before comparison; otherwise a perfectly stationary sine
    looks like 100 % drift.
    """
    y = np.asarray(y, float)
    m = len(y) // n_sub
    if m < max(int(3 * fs / f), 16):
        return np.nan
    ph = []
    for i in range(n_sub):
        A, _, _ = fit3(y[i * m:(i + 1) * m], fs, f)
        ph.append(A * np.exp(-2j * np.pi * f * (i * m) / fs))
    ph = np.asarray(ph)
    if not np.all(np.isfinite(ph)):
        return np.nan
    mu = np.mean(ph)
    return float(np.max(np.abs(ph - mu)) / abs(mu)) if abs(mu) > 0 else np.nan


# ===========================================================================
# 3. UNCERTAINTY  (Rife & Boorstyn 1974)
# ===========================================================================


def crlb_phasor(snr_db: float, n: int, freq_known: bool = True
                ) -> tuple[float, float]:
    """Cramer-Rao lower bounds for a single tone in white Gaussian noise.

    Returns (sigma_amp_rel, sigma_phase_rad): the relative standard deviation
    of the amplitude and the absolute standard deviation of the phase.

    With gamma the linear SNR (signal power over noise power) and N samples,
    Rife & Boorstyn give

        var(A)/A^2 >= 1 / (N gamma)
        var(phi)   >= 1 / (N gamma)        frequency known
        var(phi)   >= 6 / (N gamma)        frequency estimated too

    The factor 6 is the well-known penalty for having to spend information on
    the frequency; it is another way of seeing why the joint two-channel fit
    pays -- the frequency is estimated once from 2N samples rather than twice
    from N.

    These bounds are attained by the least-squares sine fit at moderate SNR,
    which is why using them as weights is legitimate rather than optimistic.
    """
    if not np.isfinite(snr_db) or n < 4:
        return np.nan, np.nan
    gamma = 10.0 ** (snr_db / 10.0)
    denom = max(n * gamma, 1e-12)
    k = 1.0 if freq_known else 6.0
    return float(np.sqrt(1.0 / denom)), float(np.sqrt(k / denom))


def sigma_rel_from_snr(snr_db, n=None, model: str = "crlb",
                       floor: float = 1e-3, ceiling: float = 1.0
                       ) -> np.ndarray:
    """Per-point relative standard deviation of |Z|, for KK weighting.

    model="crlb"    physically derived (needs n, the samples in the dwell)
    model="snr"     the legacy 10^(-SNR/20); kept only for A/B comparison
    model="uniform" every point equal

    The ratio Z = K U/u combines two independent phasors, so the relative
    variances add:  (sigma_Z/|Z|)^2 = (sigma_U/|U|)^2 + (sigma_u/|u|)^2.
    Callers pass the *combined* SNR; utils does not guess it for them.
    """
    snr_db = np.atleast_1d(np.asarray(snr_db, float))
    if model == "uniform":
        out = np.full_like(snr_db, 0.05)
    elif model == "snr":
        out = 10.0 ** (-np.clip(snr_db, 0, 80) / 20.0)
    else:
        if n is None:
            n = 1024
        n_arr = np.broadcast_to(np.asarray(n, float), snr_db.shape)
        gamma = 10.0 ** (snr_db / 10.0)
        out = np.sqrt(1.0 / np.maximum(n_arr * gamma, 1e-12))
    return np.clip(np.nan_to_num(out, nan=ceiling), floor, ceiling)


def zmag_outliers(freq: np.ndarray, Z: np.ndarray, n_mad: float = 4.0,
                  win: int = 5) -> np.ndarray:
    """Flag points whose |Z| departs from the segment's own local trend.

    WHY THIS AND NOT A TIGHTER SNR GATE
    -----------------------------------
    The failure being targeted is a phasor ratio that has blown up because
    its denominator went to noise: |Z| jumps by a factor of several while the
    neighbouring frequencies stay put.  An SNR gate removes those points only
    by removing everything weak, which at the top of the band means removing
    the points that carry the ohmic resistance -- exactly the ones worth
    keeping.

    The magnitude of an electrochemical impedance is smooth in log-frequency:
    it is the sum of a resistance and a few relaxations, none of which can
    produce a spike at one frequency.  So a point far from its own local
    median is an outlier regardless of its SNR, and a weak point that lies on
    the trend is credible regardless of how weak it is.  The test is applied
    per segment, in log|Z|, with a median absolute deviation scale, so it is
    unaffected by the overall level of the segment and by up to half the
    window being bad.

    Returns a boolean mask, True where the point is an outlier.
    """
    freq = np.asarray(freq, float)
    Z = np.asarray(Z, complex)
    n = len(freq)
    bad = np.zeros(n, bool)
    ok = np.isfinite(freq) & (freq > 0) & np.isfinite(np.abs(Z)) & (np.abs(Z) > 0)
    if ok.sum() < max(5, win):
        return bad
    order = np.argsort(freq)
    idx = order[ok[order]]
    y = np.log10(np.abs(Z[idx]))
    m = len(y)
    half = max(1, win // 2)
    resid = np.zeros(m)
    for i in range(m):
        a, b = max(0, i - half), min(m, i + half + 1)
        nb = np.concatenate([y[a:i], y[i + 1:b]])
        resid[i] = y[i] - np.median(nb) if len(nb) else 0.0
    mad = np.median(np.abs(resid - np.median(resid)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        scale = np.std(resid) or 1.0
    # a floor on the scale: on a clean spectrum the residuals are tiny and
    # any scatter at all would otherwise be called an outlier
    scale = max(scale, 0.02)          # 0.02 decade ~ 5 % in |Z|
    bad[idx] = np.abs(resid) > n_mad * scale
    return bad


def combine_snr_db(snr_a, snr_b):
    """SNR of a ratio of two independent phasors, in dB.

    Relative variances add, so 1/gamma_tot = 1/gamma_a + 1/gamma_b: the
    combined SNR is the harmonic mean, dominated by the WORSE channel.  That
    is the correct behaviour -- at the top of the band the cell-voltage
    reference is only a millivolt or two, and it, not the segment, limits the
    measurement.
    """
    ga = 10.0 ** (np.asarray(snr_a, float) / 10.0)
    gb = 10.0 ** (np.asarray(snr_b, float) / 10.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = 1.0 / (1.0 / ga + 1.0 / gb)
    return 10.0 * np.log10(np.where(np.isfinite(g) & (g > 0), g, 1e-12))


# ===========================================================================
# 4. PHASE / ALL-PASS HELPERS  (skew correction)
# ===========================================================================


def unwrap_phase_monotone(freq: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Phase of Z, unwrapped in order of increasing frequency."""
    o = np.argsort(freq)
    ph = np.unwrap(np.angle(np.asarray(Z)[o]))
    out = np.empty_like(ph)
    out[o] = ph
    return out


def apply_delay(freq: np.ndarray, Z: np.ndarray, dt: float) -> np.ndarray:
    """Undo an acquisition skew of `dt` seconds.

    A skew multiplies the measured Z by exp(-j w dt), so the correction is
    exp(+j w dt).  It is an ALL-PASS: |Z| is untouched, only the phase moves,
    linearly in frequency.  That is why it is invisible in a Bode magnitude
    plot and why it destroys the top of a Nyquist arc.
    """
    return np.asarray(Z) * np.exp(2j * np.pi * np.asarray(freq) * dt)


def all_pass_residual(freq: np.ndarray, Z: np.ndarray) -> float:
    """How much all-pass content the spectrum still carries.

    A minimum-phase (Voigt + L) description cannot represent a pure delay, so
    the leftover linear-in-frequency slope of the phase is a direct readout of
    the uncorrected skew.  Fit  phi(f) ~ phi0 + s*f  over the top decade and
    return |s| in rad/Hz; multiply by 2*pi to get seconds.
    """
    freq = np.asarray(freq, float)
    ok = np.isfinite(freq) & (freq > 0) & np.isfinite(Z)
    if ok.sum() < 4:
        return np.nan
    f, z = freq[ok], np.asarray(Z)[ok]
    hi = f >= f.max() / 10.0
    if hi.sum() < 3:
        hi = np.ones_like(f, bool)
    ph = unwrap_phase_monotone(f[hi], z[hi])
    s = np.polyfit(f[hi], ph, 1)[0]
    return float(abs(s) / (2 * np.pi))


def signed_delay_from_phase(freq: np.ndarray, Z: np.ndarray,
                            decades: float = 1.0) -> tuple[float, float]:
    """Delay from the high-frequency phase slope, with its sign and quality.

    A measured spectrum carrying an acquisition skew is Z_true * exp(-j w dt),
    so the excess phase is exactly -2*pi*f*dt: LINEAR IN f, with no curvature.
    An unclosed high-frequency arc also bends the phase, but an arc is smooth
    in LOG f and therefore curved in f.  Fitting a straight line in f over the
    top decade and reporting how well it fits separates the two, which a
    Kramers-Kronig residual cannot do -- the residual has essentially no
    curvature along the common-delay direction (0.4 % over 40 us), which is
    why fitting dt0 against it returned -154 us on hardware that had none.

    Returns (dt seconds, r2) where r2 is the coefficient of determination of
    the straight-line fit.  A high r2 means the excess phase really is a
    delay; a low one means it is arc curvature and must not be removed.
    """
    freq = np.asarray(freq, float)
    Z = np.asarray(Z)
    ok = np.isfinite(freq) & (freq > 0) & np.isfinite(Z)
    if ok.sum() < 4:
        return np.nan, np.nan
    f, z = freq[ok], Z[ok]
    hi = f >= f.max() / (10.0 ** decades)
    if hi.sum() < 4:
        return np.nan, np.nan
    fh = f[hi]
    ph = unwrap_phase_monotone(fh, z[hi])
    p = np.polyfit(fh, ph, 1)
    resid = ph - np.polyval(p, fh)
    ss_tot = float(np.sum((ph - ph.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 0.0
    # phase = -2 pi f dt  ->  slope = -2 pi dt
    return float(-p[0] / (2 * np.pi)), float(r2)


def hf_phase_slope(freq: np.ndarray, Z: np.ndarray, decades: float = 1.0
                   ) -> float:
    """d(phase)/d(ln f) over the top `decades` of the band, in rad.

    For a passive, minimum-phase electrochemical impedance the phase must
    approach 0 from BELOW as w -> infinity (capacitive), or rise smoothly into
    a small inductive tail from cabling.  A large positive phase that keeps
    growing is not electrochemistry; it is an uncorrected delay.  This number
    is the anchor silver.py uses to break the delay/inductance degeneracy that
    a KK residual alone cannot resolve.
    """
    freq = np.asarray(freq, float)
    ok = np.isfinite(freq) & (freq > 0) & np.isfinite(Z)
    if ok.sum() < 4:
        return np.nan
    f, z = freq[ok], np.asarray(Z)[ok]
    hi = f >= f.max() / (10.0 ** decades)
    if hi.sum() < 3:
        return np.nan
    ph = unwrap_phase_monotone(f[hi], z[hi])
    return float(np.polyfit(np.log(f[hi]), ph, 1)[0])


# ===========================================================================
# 5. SMALL NUMERICS
# ===========================================================================


def moving_mean(x: np.ndarray, win: int) -> np.ndarray:
    c = np.concatenate([[0.0 + 0j], np.cumsum(x)])
    return (c[win:] - c[:-win]) / win


def runs_of_true(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs of a boolean array as (start, stop) pairs."""
    mask = np.asarray(mask, bool)
    if not mask.any():
        return []
    d = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    stops = list(np.flatnonzero(d == -1) + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        stops.append(len(mask))
    return list(zip(starts, stops))


def geometric_grid_fit(freqs, tol: float = 0.01) -> dict:
    """Recover a sweep's geometric grid  f_k = f0 * r^-k  from its
    frequencies alone.

    Every stepped-sine sweep worth the name is log-spaced, with r fixed by the
    requested points per decade.  That structure is a property of the DATA,
    not of an instrument file, so recovering it costs nothing in independence
    -- and membership of the grid is far stronger evidence that a step is real
    than the SNR of that one step.  A spurious peak lands on a grid to within
    `tol` only by chance; agreement across dozens of steps is overwhelming.

    Method: the ratios of adjacent frequencies are r^m for integer m >= 1
    (m > 1 where a step was missed).  Seed with the smallest observed ratio,
    assign integer indices, then refit ln f = ln f0 - k ln r by least squares.
    """
    fr = np.array(sorted({float(f) for f in freqs if np.isfinite(f) and f > 0},
                         reverse=True))
    if len(fr) < 6:
        return {"ok": False, "note": "too few steps", "n_on_grid": 0}

    ln = np.log(fr)
    d = -np.diff(ln)
    d = d[(d > 1e-6) & np.isfinite(d)]
    if len(d) < 3:
        return {"ok": False, "note": "no usable ratios", "n_on_grid": 0}

    seed = float(np.median(d[d <= np.percentile(d, 40)]))
    best = None
    # try the seed and integer subdivisions of it: a missed step makes the
    # smallest observed gap a multiple of the true one
    for div in (1, 2, 3):
        lr = seed / div
        if lr <= 1e-6:
            continue
        k = np.round((ln[0] - ln) / lr)
        if len(np.unique(k)) < 4:
            continue
        # refit  ln f = ln f0 - k lr
        Amat = np.column_stack([np.ones_like(k), -k])
        sol, *_ = np.linalg.lstsq(Amat, ln, rcond=None)
        ln_f0, lr_fit = sol
        if lr_fit <= 1e-6:
            continue
        pred = ln_f0 - k * lr_fit
        rel = np.abs(np.expm1(ln - pred))
        n_on = int(np.sum(rel <= tol))
        score = (n_on, -float(np.median(rel)))
        if best is None or score > best[0]:
            best = (score, ln_f0, lr_fit, rel, n_on)

    if best is None:
        return {"ok": False, "note": "no consistent grid", "n_on_grid": 0}

    _, ln_f0, lr_fit, rel, n_on = best
    ratio = float(np.exp(lr_fit))
    return {
        "ok": n_on >= max(6, int(0.6 * len(fr))),
        "ratio": ratio,
        "ppd": float(1.0 / np.log10(ratio)) if ratio > 1 else np.nan,
        "f0": float(np.exp(ln_f0)),
        "n_on_grid": n_on,
        "n_total": len(fr),
        "rel_resid": float(np.median(rel)),
        "on_grid_mask": {float(f): bool(r <= tol) for f, r in zip(fr, rel)},
    }


def interp_complex(f_new: np.ndarray, f: np.ndarray, Z: np.ndarray
                   ) -> np.ndarray:
    """Log-frequency interpolation of a complex spectrum.

    Interpolating Re and Im separately on a LOG frequency axis is the right
    thing: an impedance is smooth in ln f, not in f, so a linear-in-f
    interpolation over a decade-wide gap bends the arc.
    """
    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    ok = np.isfinite(f) & (f > 0) & np.isfinite(Z.real) & np.isfinite(Z.imag)
    if ok.sum() < 2:
        return np.full(len(np.atleast_1d(f_new)), np.nan + 0j)
    o = np.argsort(f[ok])
    lf, z = np.log(f[ok][o]), Z[ok][o]
    lx = np.log(np.clip(np.atleast_1d(f_new), f[ok].min(), f[ok].max()))
    return (np.interp(lx, lf, z.real) + 1j * np.interp(lx, lf, z.imag))


def nan_safe_stats(x) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0, mean=np.nan, std=np.nan, cv=np.nan,
                    min=np.nan, max=np.nan, median=np.nan)
    m = float(np.mean(x))
    s = float(np.std(x))
    return dict(n=int(x.size), mean=m, std=s,
                cv=100.0 * s / m if m else np.nan,
                min=float(x.min()), max=float(x.max()),
                median=float(np.median(x)))


# ===========================================================================
# 6. IO
# ===========================================================================


def read_pairs(path) -> list[tuple[float, float]]:
    """Read a two-column calibration file ('c0;c1' per line)."""
    out = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p for p in re.split(r"[;,\t]", line) if p.strip()]
        if len(parts) < 2:
            continue
        try:
            out.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return out


def load_gain(path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Ex-situ frequency response of the measuring chain.

    CSV: segment,freq_hz,gain_real,gain_imag  (gain normalised to 1 at low
    frequency).  Segment "0" or "all" gives a single shared response.

    This is the one correction the old pipeline documented as missing.  The
    plate amplifiers roll off by about -3 dB near 20 kHz; the analog
    anti-alias filter in front of the ADC contributes a frequency-dependent
    phase that is NOT a pure delay unless the filter is Bessel (Smith,
    "Anti-Aliasing Filters for Digital DA Systems").  In the ratio
    U_cell/u_segment the two chains' responses cancel only to the extent that
    they match; component tolerances leave a residual that lands squarely in
    the decade where Rs lives.  Deconvolving the measured response removes it.
    """
    data: dict[str, list[tuple[float, complex]]] = {}
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("segment"):
            continue
        p = [q for q in re.split(r"[;,\t]", line) if q.strip()]
        if len(p) < 4:
            continue
        try:
            data.setdefault(p[0].strip(), []).append(
                (float(p[1]), complex(float(p[2]), float(p[3]))))
        except ValueError:
            continue
    out = {}
    for k, v in data.items():
        v.sort()
        out[k] = (np.array([x[0] for x in v]),
                  np.array([x[1] for x in v], dtype=complex))
    return out


def gain_at(gain: dict, seg: str, freq: np.ndarray) -> np.ndarray:
    """Interpolate the chain response for one segment onto `freq`.

    Magnitude is interpolated in log-log and phase after unwrapping, because
    an amplifier response is a smooth function of ln f, not of f.
    """
    freq = np.atleast_1d(np.asarray(freq, float))
    if not gain:
        return np.ones_like(freq, dtype=complex)
    key = seg if seg in gain else ("all" if "all" in gain else
                                   ("0" if "0" in gain else None))
    if key is None:
        return np.ones_like(freq, dtype=complex)
    f0, g0 = gain[key]
    lf, lg = np.log(f0), np.log(np.abs(g0))
    ph = np.unwrap(np.angle(g0))
    x = np.log(np.clip(freq, f0.min(), f0.max()))
    return np.exp(np.interp(x, lf, lg)) * np.exp(1j * np.interp(x, lf, ph))


def write_json(path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, complex):
            return {"re": o.real, "im": o.imag}
        if isinstance(o, (set, frozenset)):
            return sorted(o)
        if isinstance(o, Path):
            return str(o)
        return str(o)

    path.write_text(json.dumps(obj, indent=2, default=default))
    return path


def write_table(path, rows: list[dict], columns: list[str] | None = None
                ) -> Path:
    """Minimal CSV writer -- no pandas dependency in the core path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return path
    cols = columns or list(rows[0].keys())
    lines = [",".join(cols)]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                vals.append(f"{v:.6g}" if np.isfinite(v) else "")
            else:
                vals.append(str(v))
        lines.append(",".join(vals))
    path.write_text("\n".join(lines) + "\n")
    return path


# ===========================================================================
# 7. SELF-TEST
# ===========================================================================


def _selftest(seed: int = 0) -> int:
    """Synthetic checks of the estimators.  Run: python utils.py"""
    rng = np.random.default_rng(seed)
    fails = 0
    log = get_logger()
    banner("utils.py self-test")

    # --- fit3 recovers a known phasor -------------------------------------
    fs, f, n = 10_000.0, 137.0, 20_000
    t = np.arange(n) / fs
    A_true = 0.3 * np.exp(1j * 0.7)
    y = np.real(A_true * np.exp(2j * np.pi * f * t)) + 0.01 * rng.normal(size=n)
    A, _, snr = fit3(y, fs, f)
    err = abs(A - A_true) / abs(A_true)
    ok = err < 1e-2
    fails += not ok
    log.info(f"  fit3        : |dA|/|A| = {err:.2e}  SNR={snr:5.1f} dB   "
             f"{'PASS' if ok else 'FAIL'}")

    # --- joint7 beats two independent fits on the PHASE DIFFERENCE --------
    # This is the central claim of the new estimator, so it is tested, not
    # asserted.  Half a sample of skew is introduced exactly as Ramos & Serra
    # simulate an interleaving ADC.
    n = 2048
    f = 1500.0                      # high in the band: where it matters
    trials = 300
    e_ind, e_joint = [], []
    for _ in range(trials):
        phi = rng.uniform(-np.pi, np.pi)
        dphi = rng.uniform(-1.0, 1.0)
        t = np.arange(n) / fs
        # The quantity under test is arg(A_ref / A_sig), which is what the
        # impedance is built from.  Giving the SIGNAL channel a phase of
        # (phi - dphi) makes that quantity exactly +dphi, so the error below
        # is a true error and not 2*dphi from a sign slip.
        yr = 0.02 * np.cos(2 * np.pi * f * t + phi) + 0.004 * rng.normal(size=n)
        ys = 0.20 * np.cos(2 * np.pi * f * t + phi - dphi) + 0.004 * rng.normal(size=n)
        # independent four-parameter fits, one per channel
        _, Ar, _, _ = fit4(yr, fs, f * 1.001)
        _, As, _, _ = fit4(ys, fs, f * 1.001)
        e_ind.append(np.angle(Ar / As) - dphi)
        # joint seven-parameter fit
        jf = fit7_joint(yr, ys, fs, f * 1.001)
        e_joint.append(jf.phase_diff_rad - dphi)
    # Each entry is an INDEPENDENT trial error, not a phase sequence, so it
    # must be wrapped into (-pi, pi] individually.  np.unwrap here would chain
    # unrelated trials together and manufacture 2*pi jumps.
    _wrap = lambda e: np.angle(np.exp(1j * np.asarray(e, float)))
    s_ind = float(np.std(_wrap(e_ind)))
    s_joint = float(np.std(_wrap(e_joint)))
    gain = s_ind / max(s_joint, 1e-12)
    ok = gain > 1.2
    fails += not ok
    log.info(f"  joint7      : sigma(dphi) {np.degrees(s_ind):.4f} deg -> "
             f"{np.degrees(s_joint):.4f} deg  ({gain:.2f}x better)   "
             f"{'PASS' if ok else 'FAIL'}")

    # --- CRLB is a lower bound the fit approaches -------------------------
    sa, sp = crlb_phasor(20.0, 2048, freq_known=True)
    ok = 0 < sp < 0.05 and 0 < sa < 0.05
    fails += not ok
    log.info(f"  crlb        : sigma_A/A={sa:.2e}  sigma_phi={sp:.2e} rad   "
             f"{'PASS' if ok else 'FAIL'}")

    # --- geometric grid recovery ------------------------------------------
    f0, r = 3000.0, 10 ** (1 / 12)
    true_f = f0 * r ** -np.arange(40)
    noisy = true_f * (1 + 0.002 * rng.normal(size=40))
    g = geometric_grid_fit(noisy)
    ok = g["ok"] and abs(g["ppd"] - 12) < 0.6
    fails += not ok
    log.info(f"  grid fit    : ppd={g.get('ppd', float('nan')):.2f} "
             f"(true 12), on-grid {g['n_on_grid']}/{g.get('n_total', 0)}   "
             f"{'PASS' if ok else 'FAIL'}")

    # --- delay is all-pass -------------------------------------------------
    fq = np.logspace(0, 3.5, 60)
    Z = 0.06 + 0.25 / (1 + 2j * np.pi * fq * 1e-3)
    Zd = apply_delay(fq, Z, -55e-6)
    ok = np.allclose(np.abs(Z), np.abs(Zd), rtol=1e-12)
    fails += not ok
    log.info(f"  apply_delay : |Z| unchanged   {'PASS' if ok else 'FAIL'}")
    rec = all_pass_residual(fq, Zd / Z)
    ok = abs(rec - 55e-6) < 5e-6
    fails += not ok
    log.info(f"  all_pass    : recovered {rec*1e6:.1f} us (true 55.0)   "
             f"{'PASS' if ok else 'FAIL'}")

    log.info(f"\n  {'ALL PASS' if not fails else str(fails) + ' FAILURES'}")
    return fails


if __name__ == "__main__":
    raise SystemExit(_selftest())


# ---------------------------------------------------------------------------
# Segment areas for a given Config  (single resolution point)
# ---------------------------------------------------------------------------


def segment_areas(cfg) -> dict[str, float]:
    """The area table this run should use.

    Precedence: --areas CSV  >  --equal-areas  >  true plate geometry.
    Every layer must go through here, otherwise silver and gold can end up
    weighting the same plate differently -- which is how an aggregate stops
    matching the map it is drawn next to.
    """
    import r2d2_geometry as geom
    if getattr(cfg, "areas_file", None):
        return geom.areas_from_csv(cfg.areas_file)
    if getattr(cfg, "equal_areas", False):
        return geom.equal_areas()
    return geom.areas()
