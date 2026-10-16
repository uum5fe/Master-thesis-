#!/usr/bin/env python3
"""
tone_estimation.py
==================
Card-level excitation-tone estimation for the local-EIS pipeline.

WHAT THIS REPLACES AND WHY
--------------------------
`utils.fit7_joint` re-estimates the frequency for EVERY (reference, segment)
pair with an IEEE-1057 Gauss-Newton iteration started at the ladder rung.
Three measured problems (bench/run_bench.py, results in the review doc):

  1. Convergence radius.  Gauss-Newton on the sine frequency only converges
     when the start value is within about half a DFT bin (1/(2T)).  The
     ladder rung is off by ~0.09 % (ladder_snap.py docstring), which at the
     top of the band is 0.7-1.0 bin for a 0.24 s dwell.  From there fit4 /
     fit7 land off the tone in ~30 % of trials at ANY SNR.
  2. Start phasor (a, b) = (1, 0).  The first linearisation column is scaled
     by the true amplitude and rotated by the true phase, so step 1 is
     arbitrary (then clamped to 5 % = tens of bins at kHz).
  3. One frequency per segment.  All channels of a card share one ADC clock,
     so they share ONE frequency.  Estimating it per pair lets weak segments
     wander (median spread 6.5 Hz on one card/step in the benchmark) and
     throws away the other 15 channels' information.

The replacement is the multichannel concentrated maximum-likelihood
estimator ("variable projection", Golub & Pereyra): amplitudes, offsets and
ramps of every channel are linear and are projected out exactly, leaving a
1-D cost in f

        J(f) = sum_k  log RSS_k(f)

which is the exact ML for K records sharing f with independent, unknown noise
variances (each channel self-weights by its own SNR -- the weighting that
Radil & Ramos, Measurement 43 (2010) 1228, show the unweighted 7-parameter
fit lacks).  J is minimised by a coarse grid inside +-1 bin followed by a
bounded Brent search: no derivative, no start phasor, no step clamp, no
divergence.  With no prior it is initialised from the peak of the summed,
SNR-normalised Hann periodogram (multichannel interpolated DFT).

Phasors are then plain three-parameter (+ramp) least squares at that single
frequency -- linear, closed-form, identical frequency for every channel.

`stationarity_test` replaces the fixed 0.25 drift threshold with a
CRLB-normalised chi-square test, so noise alone no longer fails the gate.

Plain numpy/scipy.  Run `python tone_estimation.py` for the self-test.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2


# ---------------------------------------------------------------------------
# linear part
# ---------------------------------------------------------------------------
def _design(n: int, fs: float, f: float, ramp: bool = True) -> np.ndarray:
    t = np.arange(n) / fs
    w = 2.0 * np.pi * f
    cols = [np.cos(w * t), np.sin(w * t), np.ones(n)]
    if ramp:
        cols.append(t - t.mean())
    return np.column_stack(cols)


def phasors_at(Y, fs: float, f: float, ramp: bool = True):
    """Three-parameter (+ramp) LS of every row of Y at ONE frequency.

    Returns (phasor[k], residual_rms[k], snr_db[k]); phasor under exp(+jwt),
    same convention and SNR definition as utils.fit3.
    """
    Y = np.atleast_2d(np.asarray(Y, float))
    n = Y.shape[1]
    D = _design(n, fs, f, ramp)
    P, *_ = np.linalg.lstsq(D, Y.T, rcond=None)
    R = Y.T - D @ P
    A = P[0] - 1j * P[1]
    r_rms = np.sqrt(np.mean(R ** 2, axis=0))
    with np.errstate(divide="ignore"):
        snr = 20.0 * np.log10((np.abs(A) / np.sqrt(2.0)) / r_rms)
    return A, r_rms, snr


def _rss(Y: np.ndarray, fs: float, f: float, ramp: bool) -> np.ndarray:
    Q, _ = np.linalg.qr(_design(Y.shape[1], fs, f, ramp))
    proj = Q.T @ Y.T
    return np.sum(Y * Y, axis=1) - np.sum(proj * proj, axis=0)


# ---------------------------------------------------------------------------
# frequency
# ---------------------------------------------------------------------------
@dataclass
class ToneEstimate:
    freq: float
    f_init: float
    n_channels: int
    n_eval: int
    crlb_hz: float           # common-frequency CRLB from the fitted SNRs
    at_bound: bool           # Brent stopped on the search-interval edge


def _hann(n):
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def _periodogram_peak(Y, fs, f_lo=None, f_hi=None):
    """Multichannel IpDFT: SNR-normalised summed Hann periodogram, 2-point
    Hann interpolation (Grandke 1983)."""
    n = Y.shape[1]
    S = np.sum(np.abs(np.fft.rfft(Y * _hann(n), axis=1)) ** 2
               / np.maximum(np.var(Y, axis=1, keepdims=True), 1e-300), axis=0)
    k0 = 2 if f_lo is None else max(2, int(np.floor(f_lo * n / fs)))
    k1 = len(S) - 3 if f_hi is None else min(len(S) - 3, int(np.ceil(f_hi * n / fs)))
    if k1 <= k0:
        return np.nan
    k = k0 + int(np.argmax(S[k0:k1 + 1]))
    up, dn = np.sqrt(S[k + 1] / S[k]), np.sqrt(S[k - 1] / S[k])
    d = (2 * up - 1) / (up + 1) if up >= dn else -(2 * dn - 1) / (dn + 1)
    return (k + d) * fs / n


def card_frequency(Y, fs: float, f_prior: float | None = None,
                   search_bins: float = 1.0, ramp: bool = True,
                   prior_band_bins: float = 3.0, xtol_rel: float = 1e-10
                   ) -> ToneEstimate:
    """One frequency for all channels of one card in one dwell window.

    Y        (K, N) -- the reference(s) and every segment channel on the card,
             cut to the SAME sample window.
    f_prior  optional ladder rung / schedule frequency.  It is only used to
             restrict the periodogram search to +-prior_band_bins; the
             estimate itself never starts from it.
    """
    Y = np.atleast_2d(np.asarray(Y, float))
    Y = Y[np.all(np.isfinite(Y), axis=1)]
    K, n = Y.shape
    T = n / fs
    Y = Y - Y.mean(axis=1, keepdims=True)
    if f_prior is None:
        f0 = _periodogram_peak(Y, fs)
    else:
        f0 = _periodogram_peak(Y, fs, f_prior - prior_band_bins / T,
                               f_prior + prior_band_bins / T)
        if not np.isfinite(f0):
            f0 = f_prior
    lo, hi = max(f0 - search_bins / T, 1e-12), f0 + search_bins / T
    n_eval = [0]

    def cost(f):
        n_eval[0] += 1
        return float(np.sum(np.log(np.maximum(_rss(Y, fs, f, ramp), 1e-300))))

    grid = np.linspace(lo, hi, 9)
    c = np.array([cost(g) for g in grid])
    i = int(np.argmin(c))
    a, b = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(cost, bounds=(a, b), method="bounded",
                          options={"xatol": xtol_rel * f0, "maxiter": 80})
    f_hat = float(res.x)
    _, _, snr_db = phasors_at(Y, fs, f_hat, ramp)
    rho = np.sum(10.0 ** (np.clip(snr_db, -200, 200) / 10.0))
    crlb = float(np.sqrt(12.0 / ((2 * np.pi) ** 2 * max(rho, 1e-300)
                                 * n * (n * n - 1.0))) * fs)
    # minimum on the edge of the +-search_bins interval: the tone is not
    # where the periodogram said -- flag it instead of trusting it
    at_bound = bool(i in (0, len(grid) - 1)
                    and min(abs(f_hat - lo), abs(hi - f_hat)) < 1e-3 / T)
    return ToneEstimate(f_hat, float(f0), K, n_eval[0], crlb, at_bound)


# ---------------------------------------------------------------------------
# drop-in fix for utils.fit4 (kept for the schedule detector)
# ---------------------------------------------------------------------------
def fit4(y, fs: float, f0: float, n_iter: int = 12, detrend: bool = True):
    """IEEE-1057 four-parameter fit with the two defects removed.

    * (a, b) start from a three-parameter fit at f0, not (1, 0);
    * the step is clamped to half a bin (1/(2T)), not 5 % of f -- outside
      that radius the linearisation is invalid and the fit should fall back
      to `card_frequency` rather than jump across lobes.
    Returns (f_hat, phasor, residual_rms, snr_db) like utils.fit4.
    """
    y = np.asarray(y, float)
    n = len(y)
    if n < 16:
        return f0, complex("nan"), np.nan, np.nan
    t = np.arange(n) / fs
    tc = t - t.mean()
    w = 2.0 * np.pi * f0
    half_bin = np.pi * fs / n                      # 2*pi * 1/(2T)
    p, *_ = np.linalg.lstsq(_design(n, fs, f0, detrend), y, rcond=None)
    a, b = p[0], p[1]
    for _ in range(n_iter):
        c_, s_ = np.cos(w * t), np.sin(w * t)
        cols = [c_, s_, np.ones(n)] + ([tc] if detrend else [])
        cols.append(tc * (-a * s_ + b * c_))
        p, *_ = np.linalg.lstsq(np.column_stack(cols), y, rcond=None)
        a, b = p[0], p[1]
        dw = float(np.clip(p[-1], -half_bin, half_bin))
        w += dw
        if abs(dw) < 1e-10 * w:
            break
    f_hat = w / (2.0 * np.pi)
    A, r, s = phasors_at(y, fs, f_hat, detrend)
    return float(f_hat), complex(A[0]), float(r[0]), float(s[0])


# ---------------------------------------------------------------------------
# stationarity: a test, not a threshold
# ---------------------------------------------------------------------------
def stationarity_test(y, fs: float, f: float, n_sub: int = 3,
                      alpha: float = 1e-3, ramp: bool = True):
    """Chi-square test of phasor constancy across n_sub sub-windows.

    Each sub-window phasor has per-component variance ~ 2 sigma^2 / m (m
    samples, sigma^2 from the full-window residual).  Under stationarity

        Q = sum_i |P_i - P_mean|^2 / (2 sigma^2 / m)  ~  chi2(2 (n_sub - 1)).

    Returns (drift, p_value): drift is the old effect size
    (max |P_i - P_mean| / |P_mean|), p_value the probability of a Q this
    large from noise alone.  Recommended gate: reject only if
    p_value < alpha AND drift > max_drift -- significant AND material.
    """
    y = np.asarray(y, float)
    m = len(y) // n_sub
    if m < max(int(3 * fs / f), 16):
        return np.nan, np.nan
    _, r_full, _ = phasors_at(y, fs, f, ramp)
    sigma2 = float(r_full[0] ** 2)
    P = []
    for i in range(n_sub):
        A, _, _ = phasors_at(y[i * m:(i + 1) * m], fs, f, ramp)
        P.append(A[0] * np.exp(-2j * np.pi * f * (i * m) / fs))
    P = np.asarray(P)
    mu = P.mean()
    drift = float(np.max(np.abs(P - mu)) / abs(mu)) if abs(mu) > 0 else np.nan
    Q = float(np.sum(np.abs(P - mu) ** 2) / (2.0 * sigma2 / m)) if sigma2 > 0 else np.inf
    return drift, float(chi2.sf(Q, 2 * (n_sub - 1)))


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def _selftest(seed: int = 1) -> int:
    rng = np.random.default_rng(seed)
    fails = 0
    fs, f, T = 10_000.0, 3162.28, 0.24
    n = int(fs * T)
    t = np.arange(n) / fs

    # 1. card frequency reaches the CRLB, from a prior 0.09 % off
    err, crlb = [], []
    for _ in range(200):
        ph = rng.uniform(-np.pi, np.pi)
        Y = [2e-3 * np.cos(2 * np.pi * f * t + ph) + 2e-3 * rng.normal(size=n)]
        for s in range(16):
            Y.append(5e-3 * np.cos(2 * np.pi * f * t + ph - 0.2)
                     + 5e-3 * 10 ** (rng.uniform(0, 1.5)) * rng.normal(size=n))
        est = card_frequency(np.array(Y), fs, f_prior=f * 1.0009)
        err.append(est.freq - f); crlb.append(est.crlb_hz)
    eff = np.sqrt(np.mean(np.square(err))) / np.median(crlb)
    ok = eff < 1.3
    fails += not ok
    print(f"card_frequency : RMSE/CRLB = {eff:.2f}          {'PASS' if ok else 'FAIL'}")

    # 2. fixed fit4 converges from 0.2 % off with amplitude 20 (old one fails)
    y = 20 * np.cos(2 * np.pi * 1000 * t + 1.0) + 2 * rng.normal(size=n)
    fh, *_ = fit4(y, fs, 1002.0)
    ok = abs(fh - 1000.0) < 0.05
    fails += not ok
    print(f"fit4 (fixed)   : f error {fh - 1000:+.4f} Hz        {'PASS' if ok else 'FAIL'}")

    # 3. stationarity: noise alone must not fail; a real 2x amplitude step must
    fa = 0
    for _ in range(300):
        y = np.cos(2 * np.pi * f * t) + np.sqrt(1 / (2 * 10 ** -1.5)) * rng.normal(size=n)
        d, p = stationarity_test(y, fs, f)
        fa += (p < 1e-3) and (d > 0.25)
    env = np.where(t < T / 2, 1.0, 2.0)
    y = env * np.cos(2 * np.pi * f * t) + 0.3 * rng.normal(size=n)
    d, p = stationarity_test(y, fs, f)
    ok = fa / 300 < 0.01 and p < 1e-3 and d > 0.25
    fails += not ok
    print(f"stationarity   : false reject {100 * fa / 300:.1f} % at -15 dB, "
          f"step detected p={p:.1e}  {'PASS' if ok else 'FAIL'}")
    return fails


if __name__ == "__main__":
    raise SystemExit(_selftest())
