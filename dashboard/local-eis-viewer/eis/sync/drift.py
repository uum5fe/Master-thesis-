"""Clock-drift detection between DAQ cards.

Cards that are not fed a shared sample clock run at slightly different rates.
The skew is then not a constant: it grows linearly through the record.  A
single constant delay correction cannot fix that, and the error is large -
10 ppm over a 120 s record is 1.2 ms, more than twelve samples at 10 kHz, or
several complete phase wraps at 3.8 kHz.  Worse, once the drift exceeds a
fraction of a sample *within one analysis window* it destroys coherence, and
no amount of post-hoc phase rotation can recover it.

The test is simple: measure the delay in successive blocks and regress it
against block time.  The intercept is the static offset, the slope is the
rate at which the delay grows.

Sign conventions
----------------
``delay_slope_ppm`` is ``d(tau)/dt`` expressed in ppm, with ``tau`` following
the convention of :mod:`eis.sync.skew` (positive = the test card lags).  It is
exactly the value to hand to :func:`eis.sync.resample.advance_affine` together
with ``tau0_s`` in order to undo the drift.

``clock_rate_ppm`` is the physical interpretation of the same number: a card
whose sample clock runs *fast* collects samples early, so its apparent delay
*shrinks* with time, giving ``clock_rate_ppm = -delay_slope_ppm``.  Use it for
reporting; use ``delay_slope_ppm`` for correcting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from eis.sync.skew import DelayEstimate, estimate_delay


@dataclass
class DriftEstimate:
    tau0_s: float                 #: delay extrapolated to t = 0 [s]
    delay_slope_ppm: float        #: d(tau)/dt [ppm]; pass to advance_affine
    delay_slope_sigma_ppm: float
    r_squared: float
    t_mid_s: float = 0.0          #: centre of the fitted time span
    tau_mid_s: float = 0.0        #: delay at t_mid (best-determined point)
    block_times_s: np.ndarray = field(default_factory=lambda: np.array([]))
    block_taus_s: np.ndarray = field(default_factory=lambda: np.array([]))
    n_blocks_used: int = 0
    ok: bool = True
    note: str = ""

    @property
    def clock_rate_ppm(self) -> float:
        """Fractional sample-clock error of the test card, for reporting."""
        return -self.delay_slope_ppm

    def tau_at(self, t_s):
        """Delay predicted at time ``t_s`` [s]."""
        return self.tau0_s + (self.delay_slope_ppm * 1e-6) * np.asarray(t_s, float)

    def max_excursion_s(self, duration_s: float) -> float:
        """How much the delay changes over a record of ``duration_s``."""
        return abs(self.delay_slope_ppm * 1e-6) * duration_s

    def significant(self, duration_s: float, f_max_hz: float,
                    cycle_budget: float = 0.01) -> bool:
        """True when the drift must be corrected by resampling.

        Criterion: the phase error the drift causes at the top of the analysis
        band, expressed in cycles.
        """
        return self.max_excursion_s(duration_s) * f_max_hz > cycle_budget

    def intra_window_smear_s(self, window_s: float) -> float:
        """Delay change within one analysis window - the coherence killer."""
        return abs(self.delay_slope_ppm * 1e-6) * window_s


def estimate_drift(
    x_ref: np.ndarray,
    x: np.ndarray,
    fs: float,
    n_blocks: int = 8,
    band_hz: tuple[float, float] = (20.0, 4000.0),
    min_coherence: float = 0.90,
    nperseg: int | None = None,
    max_lag_s: float = 0.1,
    block_search_s: float = 2e-3,
) -> DriftEstimate:
    """Regress block-wise delay against time to separate offset from rate.

    Two safeguards make this usable on real, comb-like excitations:

    * The whole-record delay is measured first and used as a **prior** that
      centres each block's correlation search.  Without it a short block can
      lock onto a correlation sidelobe and return a delay wrong by tens of
      samples - not noise, a different branch entirely.
    * The regression is **robust** (Theil-Sen median slope, then a weighted
      refit on the inliers).  A minority of branch errors would otherwise drag
      an ordinary least-squares slope anywhere it liked.
    """
    n = int(min(len(x_ref), len(x)))
    n_blocks = max(int(n_blocks), 3)
    block = n // n_blocks
    if block < 4096:
        return DriftEstimate(
            tau0_s=0.0, delay_slope_ppm=0.0, delay_slope_sigma_ppm=float("inf"),
            r_squared=0.0, ok=False,
            note=f"record too short for {n_blocks} drift blocks",
        )
    seg_nperseg = int(nperseg or min(1 << max(int(np.log2(block // 4)), 10), 1 << 15))

    # Welch keeps only whole hops, so a block whose length is not a multiple of
    # the hop is analysed over a span whose centre sits *before* the nominal
    # block centre.  Under drift the delay depends on time, so that offset
    # becomes a bias identical in every block - invisible in the slope, but a
    # direct error on the intercept (1 us at 8 ppm with the default sizes).
    # Trimming each block to a whole number of hops removes it exactly.
    hop = seg_nperseg // 2
    block_used = seg_nperseg + hop * ((block - seg_nperseg) // hop)
    block_used = max(block_used, seg_nperseg)

    # Whole-record delay: the prior that keeps every block on the right branch.
    prior = estimate_delay(
        x_ref, x, fs, band_hz=band_hz, nperseg=min(1 << 15, n),
        min_coherence=min_coherence, min_bins=8, max_intercept_rad=np.inf,
        max_lag_s=max_lag_s,
    )
    prior_s = prior.tau_s if np.isfinite(prior.tau_s) else 0.0

    times, taus, sigmas = [], [], []
    for k in range(n_blocks):
        lo = k * block
        hi = lo + block_used
        est: DelayEstimate = estimate_delay(
            x_ref[lo:hi], x[lo:hi], fs, band_hz=band_hz,
            nperseg=seg_nperseg, min_coherence=min_coherence, min_bins=8,
            max_intercept_rad=np.inf,      # judged globally, not per block
            max_lag_s=block_search_s, prior_s=prior_s,
        )
        if not np.isfinite(est.tau_s):
            continue
        times.append((lo + hi) / 2.0 / fs)
        taus.append(est.tau_s)
        sigmas.append(est.tau_ci_s if np.isfinite(est.tau_ci_s) and est.tau_ci_s > 0 else 1.0)

    if len(times) < 3:
        return DriftEstimate(
            tau0_s=float(np.median(taus)) if taus else 0.0,
            delay_slope_ppm=0.0, delay_slope_sigma_ppm=float("inf"), r_squared=0.0,
            block_times_s=np.asarray(times), block_taus_s=np.asarray(taus),
            n_blocks_used=len(times), ok=False,
            note="too few usable blocks to separate drift from offset",
        )

    t_all = np.asarray(times)
    y_all = np.asarray(taus)
    s_all = np.asarray(sigmas)

    # --- robust pass: Theil-Sen median slope, then MAD-based inlier mask ---
    pair_slopes = [
        (y_all[j] - y_all[i]) / (t_all[j] - t_all[i])
        for i in range(len(t_all)) for j in range(i + 1, len(t_all))
        if t_all[j] != t_all[i]
    ]
    inliers = np.ones(len(t_all), dtype=bool)
    n_outliers = 0
    if pair_slopes:
        slope_ts = float(np.median(pair_slopes))
        intercept_ts = float(np.median(y_all - slope_ts * t_all))
        residual_ts = y_all - (slope_ts * t_all + intercept_ts)
        mad = float(np.median(np.abs(residual_ts - np.median(residual_ts)))) * 1.4826
        if mad > 0:
            inliers = np.abs(residual_ts) <= max(4.0 * mad, 5e-9)
        if inliers.sum() < 3:                      # keep the closest three
            inliers = np.zeros(len(t_all), bool)
            inliers[np.argsort(np.abs(residual_ts))[:3]] = True
        n_outliers = int((~inliers).sum())

    t, y = t_all[inliers], y_all[inliers]
    w = 1.0 / np.maximum(s_all[inliers], 1e-15) ** 2
    w /= w.sum()

    # Centre the abscissa: the slope and the mid-point delay are then
    # uncorrelated, so tau_mid is reported with its true (small) uncertainty
    # instead of inheriting the extrapolation error of the t=0 intercept.
    t_mid = float((w * t).sum())
    tc = t - t_mid
    A = np.column_stack([tc, np.ones_like(tc)])
    ATA = A.T @ (w[:, None] * A)
    coeffs = np.linalg.solve(ATA, A.T @ (w * y))
    slope, tau_mid = float(coeffs[0]), float(coeffs[1])

    residual = y - A @ coeffs
    dof = max(len(t) - 2, 1)
    var = float((w * residual**2).sum()) * len(t) / dof
    cov = np.linalg.inv(ATA) * var / len(t)
    slope_sigma = float(np.sqrt(max(cov[0, 0], 0.0)))

    y_mean = float((w * y).sum())
    ss_tot = float((w * (y - y_mean) ** 2).sum())
    r2 = 1.0 - float((w * residual**2).sum()) / ss_tot if ss_tot > 0 else 0.0

    return DriftEstimate(
        tau0_s=tau_mid - slope * t_mid,
        delay_slope_ppm=slope * 1e6,
        delay_slope_sigma_ppm=slope_sigma * 1e6,
        r_squared=float(r2),
        t_mid_s=t_mid, tau_mid_s=tau_mid,
        block_times_s=t_all, block_taus_s=y_all, n_blocks_used=int(inliers.sum()),
        ok=True,
        note=(
            f"{n_outliers}/{len(t_all)} blocks rejected as correlation "
            f"branch outliers" if n_outliers else ""
        ),
    )
