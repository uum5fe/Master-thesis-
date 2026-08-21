"""Sub-sample delay estimation between two channels.

The whole pipeline rests on one observation: every DAQ card records its own
copy of the *same* physical cell voltage.  The relative timing of two cards is
therefore directly measurable from those two copies, with no assumption about
the electrochemistry of the cell.  That is what this module does.

Sign convention used everywhere
-------------------------------
``estimate_delay(x_ref, x)`` returns ``tau`` such that

    x(t)  ~=  x_ref(t - tau)

i.e. a **positive tau means x lags x_ref** (x is the later signal).  To bring
``x`` onto the reference time base, advance it by ``tau``
(:func:`eis.sync.resample.fractional_delay_fft` with ``-tau``).

Two estimators are combined:

1. **GCC-PHAT** - generalised cross-correlation with phase transform.  Whitens
   the cross-spectrum so the correlation peak is a sharp impulse regardless of
   the excitation's spectral shape.  Unambiguous over the full record, robust,
   accurate to a fraction of a sample after parabolic interpolation.
2. **Cross-spectral phase slope** - the delay is read from the slope of the
   unwrapped cross-spectrum phase, weighted by ``gamma^2/(1-gamma^2)`` (the
   correct inverse-variance weight for a phase estimate).  Precise to far
   below one sample, but only usable after the coarse estimate has removed the
   bulk of the delay - otherwise the phase wraps and the unwrapping is
   meaningless.

Step 1 makes step 2 safe.  Fitting ``arctan2`` output directly, without the
de-rotation, silently fails for anything beyond about half a sample of skew.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import coherence as sp_coherence
from scipy.signal import csd as sp_csd


@dataclass
class DelayEstimate:
    """Result of a delay measurement between two channels."""

    tau_s: float                 #: delay of the test channel w.r.t. reference
    tau_ci_s: float              #: 1-sigma uncertainty on tau
    tau_coarse_s: float          #: GCC-PHAT estimate alone
    coherence_median: float      #: median gamma^2 in the fit band
    intercept_rad: float         #: constant phase term; ~0 for a pure delay
    n_bins: int                  #: bins that entered the phase-slope fit
    method: str                  #: "phat+phase_slope" | "phat" | "failed"
    ok: bool = True
    note: str = ""

    @property
    def tau_ns(self) -> float:
        return self.tau_s * 1e9

    def phase_error_deg_at(self, f_hz: float) -> float:
        """Phase error this delay would cause at ``f_hz``."""
        return 360.0 * f_hz * self.tau_s


# ---------------------------------------------------------------------------
# 1. Coarse: generalised cross-correlation with phase transform
# ---------------------------------------------------------------------------

def gcc_phat(
    x_ref: np.ndarray,
    x: np.ndarray,
    fs: float,
    band_hz: tuple[float, float] | None = None,
    max_lag_s: float | None = 0.1,
    snr_factor: float = 10.0,
    prior_s: float = 0.0,
) -> tuple[float, float]:
    """Return ``(tau_s, peak_sharpness)`` from a PHAT-weighted correlation.

    ``peak_sharpness`` is the ratio of the highest to the second-highest
    correlation peak (peaks closer than one main lobe apart are treated as
    one).  Values near 1 mean the delay is ambiguous.

    Two details matter for EIS excitations and are easy to get wrong:

    * **Whitening must be restricted to bins that carry signal.**  A plain
      phase transform sets *every* bin to unit magnitude, so with a multisine -
      seventeen tones among tens of thousands of empty bins - the correlation
      is dominated by noise-only bins and locks onto a random peak with a
      confident-looking amplitude.  Bins are therefore kept only when the
      cross-power stands clear of the noise floor, estimated robustly as the
      in-band median.
    * **A periodic or comb-like excitation makes the delay ambiguous.**  A
      multisine's whitened correlation has strong sidelobes, and a short block
      can lock onto one of them.  ``max_lag_s`` bounds the search, and
      ``prior_s`` centres that window on an already-known estimate - which is
      how the block-wise drift fit avoids branch errors.
    """
    n = int(min(len(x_ref), len(x)))
    a = np.asarray(x_ref[:n], dtype=np.float64)
    b = np.asarray(x[:n], dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()

    nfft = 1 << int(np.ceil(np.log2(2 * n)))          # zero-pad: linear, not circular
    A = np.fft.rfft(a, nfft)
    B = np.fft.rfft(b, nfft)
    R = np.conj(A) * B

    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    in_band = (
        np.ones(len(freqs), bool) if band_hz is None
        else (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    )

    # Keep only bins whose cross-power rises clearly above the in-band noise
    # floor, then whiten those.  For broadband excitation nearly everything is
    # kept and this reduces to classic PHAT.
    power = np.abs(A) * np.abs(B)
    floor = np.median(power[in_band]) if in_band.any() else 0.0
    keep = in_band & (power > snr_factor * floor)
    if keep.sum() < 8:                       # broadband or very low SNR
        keep = in_band

    mag = np.abs(R)
    W = np.zeros_like(R)
    np.divide(R, mag, out=W, where=(mag > 0) & keep)
    W[~keep] = 0.0

    cc = np.fft.fftshift(np.fft.irfft(W, nfft))
    zero = nfft // 2

    centre = zero + int(round(prior_s * fs))
    span = nfft if max_lag_s is None else int(np.ceil(max_lag_s * fs))
    lo, hi = max(0, centre - span), min(nfft, centre + span + 1)
    if hi <= lo:
        lo, hi = 0, nfft
    segment = np.abs(cc[lo:hi])
    peak = int(np.argmax(segment)) + lo

    # Parabolic interpolation of the peak for a sub-sample estimate.
    if 0 < peak < nfft - 1:
        y0, y1, y2 = cc[peak - 1], cc[peak], cc[peak + 1]
        denom = y0 - 2 * y1 + y2
        delta = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5)) if denom else 0.0
    else:
        delta = 0.0

    # Ambiguity: how much does the best peak beat the best rival outside its
    # own main lobe?
    guard = max(4, int(0.002 * fs))
    masked = segment.copy()
    p_local = peak - lo
    masked[max(0, p_local - guard): p_local + guard + 1] = 0.0
    rival = float(masked.max()) if masked.size else 0.0
    best = float(abs(cc[peak]))
    sharpness = best / rival if rival > 0 else float("inf")
    return (peak + delta - zero) / fs, sharpness


# ---------------------------------------------------------------------------
# 2. Fine: cross-spectral phase slope
# ---------------------------------------------------------------------------

def _weighted_linfit(
    x: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[float, float, float]:
    """Weighted least squares ``y = a*x + b``; returns ``(a, b, sigma_a)``."""
    A = np.column_stack([x, np.ones_like(x)])
    W = w / w.sum()
    ATA = A.T @ (W[:, None] * A)
    ATy = A.T @ (W * y)
    coeffs = np.linalg.solve(ATA, ATy)
    residual = y - A @ coeffs
    dof = max(len(x) - 2, 1)
    var = float((W * residual**2).sum()) * len(x) / dof
    cov = np.linalg.inv(ATA) * var / len(x)
    return float(coeffs[0]), float(coeffs[1]), float(np.sqrt(max(cov[0, 0], 0.0)))


def phase_slope_delay(
    x_ref: np.ndarray,
    x: np.ndarray,
    fs: float,
    tau_prior_s: float = 0.0,
    band_hz: tuple[float, float] = (20.0, 4000.0),
    nperseg: int = 32_768,
    min_coherence: float = 0.90,
    min_bins: int = 16,
) -> DelayEstimate:
    """Refine ``tau_prior_s`` from the slope of the cross-spectrum phase."""
    n = int(min(len(x_ref), len(x)))
    nperseg = int(min(nperseg, n))
    a, b = np.asarray(x_ref[:n], float), np.asarray(x[:n], float)

    f, Pxy = sp_csd(a, b, fs=fs, nperseg=nperseg, detrend="constant")
    _, g2 = sp_coherence(a, b, fs=fs, nperseg=nperseg, detrend="constant")
    g2 = np.clip(g2, 0.0, 1.0 - 1e-12)

    mask = (f >= band_hz[0]) & (f <= band_hz[1]) & (g2 >= min_coherence)
    if mask.sum() < min_bins:
        return DelayEstimate(
            tau_s=tau_prior_s, tau_ci_s=float("inf"),
            tau_coarse_s=tau_prior_s,
            coherence_median=float(np.median(g2[(f >= band_hz[0]) & (f <= band_hz[1])]))
            if np.any((f >= band_hz[0]) & (f <= band_hz[1])) else 0.0,
            intercept_rad=float("nan"), n_bins=int(mask.sum()),
            method="phat", ok=False,
            note=f"only {int(mask.sum())} bins above gamma^2={min_coherence}; "
                 f"phase-slope refinement skipped",
        )

    fm, Pm, gm = f[mask], Pxy[mask], g2[mask]

    # De-rotate by the prior BEFORE unwrapping: the residual phase is then
    # small and monotone, so np.unwrap cannot pick the wrong branch.
    phi = np.unwrap(np.angle(Pm * np.exp(1j * 2 * np.pi * fm * tau_prior_s)))
    weights = gm / (1.0 - gm)                       # 1/var of a phase estimate

    slope, intercept, sigma = _weighted_linfit(2 * np.pi * fm, phi, weights)
    tau = tau_prior_s - slope

    return DelayEstimate(
        tau_s=float(tau), tau_ci_s=float(sigma), tau_coarse_s=float(tau_prior_s),
        coherence_median=float(np.median(gm)), intercept_rad=float(intercept),
        n_bins=int(mask.sum()), method="phat+phase_slope", ok=True,
    )


# ---------------------------------------------------------------------------
# Combined estimator
# ---------------------------------------------------------------------------

def estimate_delay(
    x_ref: np.ndarray,
    x: np.ndarray,
    fs: float,
    band_hz: tuple[float, float] = (20.0, 4000.0),
    nperseg: int = 32_768,
    min_coherence: float = 0.90,
    min_bins: int = 16,
    max_intercept_rad: float = 0.05,
    max_lag_s: float | None = 0.1,
    min_sharpness: float = 1.5,
    prior_s: float = 0.0,
) -> DelayEstimate:
    """Measure the delay of ``x`` relative to ``x_ref``.

    Returns a :class:`DelayEstimate`; ``tau_s > 0`` means ``x`` lags.
    """
    tau_coarse, sharpness = gcc_phat(
        x_ref, x, fs, band_hz=band_hz, max_lag_s=max_lag_s, prior_s=prior_s
    )
    # Remove the whole-sample part by slicing, then estimate the remainder.
    # A cross-spectrum measures delay through fixed index windows, so once the
    # lag is a noticeable fraction of the FFT window the two windows no longer
    # cover the same stretch of signal and the estimate picks up a bias that
    # grows with the lag.  Slicing is exact and free, and leaves the fine stage
    # with a sub-sample residual, where it is unbiased.
    #
    # The loop matters because the coarse stage is the weak link: with a
    # comb-like excitation it can settle on a correlation sidelobe tens of
    # samples away.  The phase slope still finds the right answer from densely
    # spaced bins, so feeding it back and re-slicing converges in one or two
    # passes.
    n = int(min(len(x_ref), len(x)))
    estimate, whole = tau_coarse, None
    est = None
    for _ in range(4):
        new_whole = int(np.round(estimate * fs))
        if new_whole == whole and est is not None:
            break
        whole = new_whole
        if whole > 0:
            ref_aligned, test_aligned = x_ref[: n - whole], x[whole:n]
        elif whole < 0:
            ref_aligned, test_aligned = x_ref[-whole:n], x[: n + whole]
        else:
            ref_aligned, test_aligned = x_ref[:n], x[:n]
        if len(ref_aligned) < 2 * min_bins:
            break
        est = phase_slope_delay(
            ref_aligned, test_aligned, fs, tau_prior_s=estimate - whole / fs,
            band_hz=band_hz, nperseg=min(nperseg, len(ref_aligned)),
            min_coherence=min_coherence, min_bins=min_bins,
        )
        est.tau_s += whole / fs
        if not est.ok:
            break
        estimate = est.tau_s

    if est is None:
        est = phase_slope_delay(
            x_ref, x, fs, tau_prior_s=tau_coarse, band_hz=band_hz,
            nperseg=nperseg, min_coherence=min_coherence, min_bins=min_bins,
        )
    est.tau_coarse_s = tau_coarse

    if sharpness < min_sharpness:
        est.ok = False
        est.note = (
            f"{est.note + '; ' if est.note else ''}"
            f"ambiguous correlation peak (best/rival = {sharpness:.2f}) - the "
            f"two channels may not carry a common signal, or the excitation is "
            f"periodic within the search window"
        ).strip()

    # A pure delay produces a straight line through the origin.  A significant
    # constant term means the channels differ by more than a time shift (an
    # analog filter mismatch, or a polarity inversion at pi rad).
    if est.ok and np.isfinite(est.intercept_rad):
        if abs(est.intercept_rad) > max_intercept_rad:
            est.ok = False
            est.note = (
                f"{est.note + '; ' if est.note else ''}"
                f"phase intercept {est.intercept_rad:+.3f} rad exceeds "
                f"{max_intercept_rad} rad - not a pure delay "
                f"(filter mismatch or inverted polarity?)"
            ).strip()
    return est


def closure_residual(taus: dict[tuple[int, int], float]) -> dict[tuple[int, int, int], float]:
    """Triplet closure check on pairwise delays.

    For any three cards the measured delays must satisfy
    ``tau_ab + tau_bc + tau_ca = 0``.  The residual is a free consistency test
    that needs no ground truth: a non-zero closure proves at least one of the
    three estimates is wrong.
    """
    cards = sorted({c for pair in taus for c in pair})
    out: dict[tuple[int, int, int], float] = {}

    def get(i: int, j: int) -> float | None:
        if (i, j) in taus:
            return taus[(i, j)]
        if (j, i) in taus:
            return -taus[(j, i)]
        return None

    for ai in range(len(cards)):
        for bi in range(ai + 1, len(cards)):
            for ci in range(bi + 1, len(cards)):
                a, b, c = cards[ai], cards[bi], cards[ci]
                parts = [get(a, b), get(b, c), get(c, a)]
                if any(p is None for p in parts):
                    continue
                out[(a, b, c)] = float(sum(parts))  # type: ignore[arg-type]
    return out
