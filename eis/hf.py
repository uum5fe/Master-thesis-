"""High-frequency accuracy for locally-resolved impedance.

Why the high-frequency end needs its own module
-----------------------------------------------
Below a few hundred hertz the measured local impedance is dominated by the
cell.  Above it, it is dominated by the *instrument*, and by three terms in
particular - each of which produces a phase error that grows with frequency
and none of which shows up in the magnitude:

=====================================  ==========================  ===========
term                                   size at 3.8 kHz             handled by
=====================================  ==========================  ===========
via-shunt loop inductance L/R          0.5 nH over 25 uOhm = 20 us  §1
                                       -> **26 deg**
anti-alias filter mismatch             2 % of a 4 kHz corner
                                       -> ~0.5 deg                  §1
residual channel skew                  73 ns -> 0.1 deg             §2
=====================================  ==========================  ===========

The first of these is the one that matters and the one that a pipeline built
around timing alone will never find.  A via shunt is a few tens of microohms;
its own loop inductance therefore has a time constant in the *tens of
microseconds*, three orders of magnitude larger than the nanosecond timing
budget the synchronisation stage works so hard to meet.  Treating the shunt
transfer function ``H(T)`` as a real number - which is what a calibration file
of ``c0;c1`` pairs implies - puts tens of degrees of phase error onto every
segment at the top of the band.  That is exactly the region where the Nyquist
arc crosses the real axis, so it corrupts the ohmic resistance, the inductive
tail, and the visual impression that the data is "fine at high frequency".

The four sections below are the correction chain, in application order.

1. Complex instrument response
------------------------------
Model the shunt as ``H(f) = H_dc * (1 + j*w*tau_shunt)`` and the acquisition
front end as an analog anti-alias filter.  Since ``Z = U/I`` and
``I = V_shunt/H_dc``, an unmodelled phase in ``H`` lands directly on ``Z``:

    Z_true(f) = Z_meas(f) * (1 + j*w*tau_shunt) * exp(-j*w*tau_residual)
                          * H_aa(f; segment) / H_aa(f; cell voltage)

Only the *mismatch* between the two channels' filters survives, which is why a
shared front end costs nothing and a mismatched one is worth about half a
degree per per-cent of corner-frequency tolerance.

2. Identifying the shared residual delay: inductance-resistance decorrelation
-----------------------------------------------------------------------------
A delay and a series inductance are degenerate over a finite band.  Rotating a
spectrum by ``exp(j*w*eps)`` adds ``eps*Rs`` to the apparent inductance:

    Im(Z_meas)/w  ~=  L_true + eps * Rs           (at HF, where |Z| ~= Rs)

so no single spectrum can separate them - and a Kramers-Kronig fit that
carries a ``j*w*L`` term cannot either, because that term absorbs the delay
exactly.  This is the reason an unbounded "KK-optimal delay" search quietly
eats the cell's real inductance.

A segmented cell breaks the degeneracy, because it is not one spectrum but
eighty measured through *one* instrument:

* the residual delay ``eps`` is a property of the acquisition card and is
  therefore **common** to every segment on it;
* the wiring inductance ``L_k`` is a property of one segment's current path
  and is therefore **individual**.

Fit ``(Rs_k, L_k)`` on the high-frequency band of every segment and regress
``L_k`` on ``Rs_k`` across the card.  Because ``L_k(eps) = L_k + eps*Rs_k``,
a non-zero delay shows up as a *spurious proportionality between the fitted
inductance and the fitted ohmic resistance*, and the regression slope is the
delay:

    eps  =  slope of  L_k  against  Rs_k

in closed form, from quantities that are themselves closed-form weighted least
squares.  The identifying assumption is stated rather than hidden: the lead
inductance of a segment is uncorrelated with the local membrane resistance of
that segment - the first is set by via geometry, the second by hydration.  It
is also testable, and the function reports the test: after correction the
correlation must be gone and the scatter of ``L_k`` must shrink.

The estimator needs the ohmic resistance to actually vary across the plate.
When it does not, the regression is ill-conditioned and the identification is
skipped and said so, instead of returning a confident number from noise.

3. High-frequency resistance
----------------------------
``Rs`` is then read from a weighted ``Rs + j*w*L + B*(j*w)^-n`` fit over the
top decade rather than from the median of the highest real parts.  The median
is biased by the inductive tail it does not model; the fit removes the tail,
carries the residual arc, and returns ``Rs`` with a standard error rather than
a bare number.

4. In-plane crosstalk
---------------------
A segment collects some of its neighbours' current through the in-plane
conductivity of the diffusion medium, so the measured admittances are a
spatially smoothed version of the true ones.  With a mixing matrix ``M`` built
from the segment adjacency, ``Y_meas = M @ Y_true``, and a regularised inverse
sharpens the map.  ``alpha = 0`` (the default) disables it: the mixing fraction
has to come from a characterisation of the plate, and inventing one would
trade a known blur for an unknown artefact.

References for the physics this module corrects
-----------------------------------------------
* In-plane conduction of the porous transport layer distorts spatially
  resolved current and impedance measurements; the fix is a mixing matrix
  computed from transport-layer properties and inverted (§4).
  https://iopscience.iop.org/article/10.1149/1945-7111/ae60aa
* Inductive artefacts in the upper frequency range originate in the
  measurement wiring, and the high-frequency real-axis intercept is only the
  sum of ohmic resistances once that inductance is subtracted (§3).
  https://www.biologic.net/documents/eis-precautions-electrochemistry-battery-application-note-5/
* Instrument artefacts in impedance spectra are estimable and correctable as a
  complex response rather than a scalar (§1).
  https://www.nature.com/articles/s41598-020-80468-x
* Local high-frequency resistance distribution is the primary spatially
  resolved observable, which is why §3 is the number that reaches the map.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11647849/
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# 1. Complex instrument response
# ---------------------------------------------------------------------------

@dataclass
class InstrumentResponse:
    """Everything between the physical impedance and the recorded ratio.

    All members default to "no correction", and every applied term is reported
    in :attr:`terms` so that an output row states which corrections it carries
    rather than leaving the reader to infer it from the configuration.
    """

    #: Residual delay of the current channel relative to the voltage channel
    #: [s].  Positive means the current was recorded later.
    delay_s: float = 0.0
    #: ``L_shunt / R_shunt`` of the via shunt [s].
    shunt_tau_s: float = 0.0
    #: Analog anti-alias filter shared by both channels.
    aa_order: int = 0
    aa_corner_hz: float = 0.0
    #: Fractional corner mismatch of the segment channel relative to the
    #: cell-voltage channel.
    aa_corner_mismatch: float = 0.0

    @property
    def active(self) -> bool:
        return bool(
            self.delay_s or self.shunt_tau_s
            or (self.aa_order and self.aa_corner_hz and self.aa_corner_mismatch)
        )

    @property
    def terms(self) -> list[str]:
        out: list[str] = []
        if self.shunt_tau_s:
            out.append(f"shunt_tau={self.shunt_tau_s * 1e6:.2f}us")
        if self.delay_s:
            out.append(f"delay={self.delay_s * 1e9:.1f}ns")
        if self.aa_order and self.aa_corner_hz and self.aa_corner_mismatch:
            out.append(
                f"aa={self.aa_order}@{self.aa_corner_hz:.0f}Hz"
                f"({self.aa_corner_mismatch:+.1%})"
            )
        return out

    def factor(self, f: np.ndarray) -> np.ndarray:
        """Multiplicative correction ``C(f)`` with ``Z_true = Z_meas * C(f)``."""
        f = np.asarray(f, float)
        w = 2.0 * np.pi * f
        C = np.ones(len(f), dtype=complex)
        if self.shunt_tau_s:
            C = C * (1.0 + 1j * w * self.shunt_tau_s)
        if self.delay_s:
            C = C * np.exp(-1j * w * self.delay_s)
        if self.aa_order and self.aa_corner_hz and self.aa_corner_mismatch:
            C = C * (
                butterworth_response(f, self.aa_order, self.aa_corner_hz
                                     * (1.0 + self.aa_corner_mismatch))
                / butterworth_response(f, self.aa_order, self.aa_corner_hz)
            )
        return C

    def phase_deg_at(self, f_hz: float) -> float:
        """Phase this response contributes at ``f_hz`` - the honest headline."""
        return float(np.degrees(np.angle(self.factor(np.array([f_hz]))[0])))


def butterworth_response(f: np.ndarray, order: int, corner_hz: float) -> np.ndarray:
    """Complex response of an analog Butterworth low-pass.

    Written from the pole locations rather than by calling a filter designer,
    so it stays exact for non-integer corner ratios and needs no state.
    """
    f = np.asarray(f, float)
    if order <= 0 or corner_hz <= 0:
        return np.ones(len(f), dtype=complex)
    s = 1j * (f / corner_hz)
    k = np.arange(1, order + 1)
    poles = np.exp(1j * np.pi * (2 * k + order - 1) / (2 * order))
    denominator = np.prod(s[:, None] - poles[None, :], axis=1)
    return np.prod(-poles) / denominator


def shunt_time_constant(
    shunt_h_v_cm2_per_a: float, segment_area_cm2: float, inductance_nh: float
) -> float:
    """``tau = L/R`` of one via shunt [s].

    ``H`` is quoted as a voltage per current *density*, so the resistance the
    inductance competes with is ``R = H / A``.  Getting that factor wrong is
    the difference between a 5 us and a 20 us time constant - between six and
    twenty-six degrees at the top of the band.
    """
    if shunt_h_v_cm2_per_a <= 0 or segment_area_cm2 <= 0 or inductance_nh <= 0:
        return 0.0
    resistance = shunt_h_v_cm2_per_a / segment_area_cm2          # Ohm
    return float(inductance_nh * 1e-9 / resistance)


# ---------------------------------------------------------------------------
# 2. High-frequency Rs + L fit
# ---------------------------------------------------------------------------

@dataclass
class HFRFit:
    """Weighted ``Rs + jwL (+ residual arc)`` fit of the high-frequency end."""

    rs_ohm: float = float("nan")
    rs_sigma_ohm: float = float("nan")
    l_h: float = float("nan")
    l_sigma_h: float = float("nan")
    arc_amplitude: float = 0.0
    n_points: int = 0
    f_min_hz: float = float("nan")
    f_max_hz: float = float("nan")
    chi2_reduced: float = float("nan")
    ok: bool = False
    note: str = ""

    @property
    def crossover_hz(self) -> float:
        """Frequency where the fitted arc turns inductive: ``-Im(Z) = 0``."""
        if not np.isfinite(self.l_h) or self.l_h <= 0 or self.arc_amplitude <= 0:
            return float("nan")
        # arc: Im = -B*w^-n*sin(n*pi/2);  inductor: Im = +w*L.  Equal at:
        n = _ARC_EXPONENT
        w = (self.arc_amplitude * np.sin(n * np.pi / 2) / self.l_h) ** (1.0 / (1.0 + n))
        return float(w / (2 * np.pi))


#: Exponent of the residual-arc term in the HFR fit.  A depressed arc seen far
#: above its own peak frequency behaves as ``(jw)^-n``; 0.85 is the middle of
#: the range PEM cathode arcs actually show, and the fitted ``Rs`` is
#: insensitive to it at the 0.1 % level because the term is small there.
_ARC_EXPONENT = 0.85


def fit_hf_resistance(
    f: np.ndarray,
    Z: np.ndarray,
    sigma: np.ndarray | None = None,
    band_decades: float = 1.0,
    min_points: int = 4,
    arc_min_points: int = 8,
) -> HFRFit:
    """Ohmic resistance from the top of the band, with the inductance removed.

    The model is linear in its parameters::

        Z(f) = Rs + j*w*L + B * (j*w)^-n

    so the fit is one weighted least-squares solve with an exact covariance -
    no optimiser, no starting guess, no chance of a local minimum.  ``B``
    carries the tail of the kinetic arc that is still present at the top of the
    band; without it that tail leaks into ``Rs`` and biases the plate map
    everywhere the arc is large, which is precisely where the interesting
    segments are.
    """
    f = np.asarray(f, float)
    Z = np.asarray(Z, complex)
    order = np.argsort(f)
    f, Z = f[order], Z[order]
    if sigma is not None:
        sigma = np.asarray(sigma, float)[order]

    if len(f) < min_points:
        return HFRFit(n_points=len(f), note=f"only {len(f)} points in the spectrum")

    f_max = float(f[-1])
    f_lo = f_max / (10.0 ** max(band_decades, 0.1))
    mask = f >= f_lo
    if mask.sum() < min_points:                 # widen until it fits
        mask = np.zeros(len(f), bool)
        mask[-min_points:] = True

    fm, Zm = f[mask], Z[mask]
    w = 2.0 * np.pi * fm
    weights = (
        1.0 / np.maximum(sigma[mask], 1e-15) if sigma is not None
        else 1.0 / np.abs(Zm)
    )

    use_arc = mask.sum() >= arc_min_points
    n = _ARC_EXPONENT
    columns = [
        np.concatenate([np.ones(len(fm)), np.zeros(len(fm))]),          # Rs
        np.concatenate([np.zeros(len(fm)), w]),                          # jwL
    ]
    if use_arc:
        arc = (1j * w) ** (-n)
        columns.append(np.concatenate([arc.real, arc.imag]))             # B
    A = np.column_stack(columns)
    b = np.concatenate([Zm.real, Zm.imag])
    W = np.concatenate([weights, weights])

    try:
        Aw = A * W[:, None]
        ATA = Aw.T @ Aw
        coeffs = np.linalg.solve(ATA, Aw.T @ (b * W))
        residual = (A @ coeffs - b) * W
        dof = max(len(b) - A.shape[1], 1)
        chi2_red = float((residual**2).sum() / dof)
        cov = np.linalg.inv(ATA) * chi2_red
        errors = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        return HFRFit(n_points=int(mask.sum()), note="high-frequency fit is singular")

    return HFRFit(
        rs_ohm=float(coeffs[0]), rs_sigma_ohm=float(errors[0]),
        l_h=float(coeffs[1]), l_sigma_h=float(errors[1]),
        arc_amplitude=float(coeffs[2]) if use_arc else 0.0,
        n_points=int(mask.sum()), f_min_hz=float(fm[0]), f_max_hz=f_max,
        chi2_reduced=chi2_red, ok=True,
        note="" if use_arc else "residual-arc term omitted: too few points",
    )


# ---------------------------------------------------------------------------
# 3. Pooled identification of the shared residual delay
# ---------------------------------------------------------------------------

@dataclass
class CommonModeResult:
    """Outcome of the inductance-resistance decorrelation."""

    delay_s: float = 0.0
    delay_sigma_s: float = float("nan")
    n_segments: int = 0
    #: Correlation between the fitted ``L`` and ``Rs`` before and after.  The
    #: first is the evidence, the second is the check.
    correlation_before: float = float("nan")
    correlation_after: float = float("nan")
    #: Spread of the fitted inductance before and after [H].  A real delay
    #: makes this shrink; noise does not.
    l_spread_before_h: float = float("nan")
    l_spread_after_h: float = float("nan")
    rs_spread_ohm: float = float("nan")
    clipped: bool = False
    applied: bool = False
    ok: bool = False
    note: str = ""

    def phase_deg_at(self, f_hz: float) -> float:
        return float(360.0 * f_hz * self.delay_s)


def _theil_sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    slopes = [
        (y[j] - y[i]) / (x[j] - x[i])
        for i in range(len(x)) for j in range(i + 1, len(x))
        if x[j] != x[i]
    ]
    return float(np.median(slopes)) if slopes else 0.0


def identify_common_delay(
    fits: dict[int, HFRFit],
    bound_s: float,
    min_segments: int = 4,
    min_rs_spread_ratio: float = 3.0,
) -> CommonModeResult:
    """Recover the delay shared by a group of segments from their HF fits.

    ``fits`` maps segment number to the result of :func:`fit_hf_resistance` for
    the segments that share one acquisition card - the group over which the
    residual delay is genuinely common.

    The estimator is the slope of ``L_k`` against ``Rs_k``.  It is computed
    robustly (Theil-Sen, then a weighted refit on the inliers) because one
    segment with a broken shunt would otherwise set the correction for the
    whole card.
    """
    usable = {
        s: fit for s, fit in fits.items()
        if fit.ok and np.isfinite(fit.rs_ohm) and np.isfinite(fit.l_h)
        and fit.rs_ohm > 0 and np.isfinite(fit.rs_sigma_ohm)
    }
    result = CommonModeResult(n_segments=len(usable))
    if len(usable) < min_segments:
        result.note = (
            f"{len(usable)} usable segments in this group; pooling needs "
            f"at least {min_segments} to beat a single-spectrum fit"
        )
        return result

    rs = np.array([f.rs_ohm for f in usable.values()], float)
    ll = np.array([f.l_h for f in usable.values()], float)
    rs_sigma = np.array([f.rs_sigma_ohm for f in usable.values()], float)
    l_sigma = np.array([max(f.l_sigma_h, 1e-30) for f in usable.values()], float)

    # The regression needs leverage: Rs has to vary by more than its own
    # measurement error, otherwise the slope is the ratio of two noises.
    rs_spread = float(np.std(rs))
    rs_noise = float(np.median(rs_sigma))
    result.rs_spread_ohm = rs_spread
    if rs_spread < min_rs_spread_ratio * max(rs_noise, 1e-30):
        result.note = (
            f"ohmic resistance varies by {rs_spread * 1e6:.1f} uOhm across the "
            f"group, only {rs_spread / max(rs_noise, 1e-30):.1f}x its own "
            f"uncertainty; the delay is not identifiable from this plate and "
            f"no correction is invented"
        )
        return result

    def correlation(a: np.ndarray, b: np.ndarray) -> float:
        if np.std(a) <= 0 or np.std(b) <= 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    result.correlation_before = correlation(rs, ll)
    result.l_spread_before_h = float(np.std(ll))

    # --- robust slope -----------------------------------------------------
    slope_ts = _theil_sen_slope(rs, ll)
    intercept_ts = float(np.median(ll - slope_ts * rs))
    residual = ll - (slope_ts * rs + intercept_ts)
    mad = float(np.median(np.abs(residual - np.median(residual)))) * 1.4826
    inliers = (
        np.abs(residual) <= 4.0 * mad if mad > 0 else np.ones(len(rs), bool)
    )
    if inliers.sum() < min_segments:
        inliers = np.zeros(len(rs), bool)
        inliers[np.argsort(np.abs(residual))[:min_segments]] = True

    x, y = rs[inliers], ll[inliers]
    weight = 1.0 / l_sigma[inliers] ** 2
    weight = weight / weight.sum()
    x_mean = float((weight * x).sum())
    xc = x - x_mean
    A = np.column_stack([xc, np.ones_like(xc)])
    ATA = A.T @ (weight[:, None] * A)
    try:
        coeffs = np.linalg.solve(ATA, A.T @ (weight * y))
    except np.linalg.LinAlgError:
        result.note = "inductance-resistance regression is singular"
        return result
    slope = float(coeffs[0])

    fit_residual = y - A @ coeffs
    dof = max(len(x) - 2, 1)
    var = float((weight * fit_residual**2).sum()) * len(x) / dof
    slope_sigma = float(np.sqrt(max(np.linalg.inv(ATA)[0, 0] * var / len(x), 0.0)))

    # slope has units H/Ohm = s: it *is* the delay.
    delay = slope
    result.delay_sigma_s = slope_sigma
    result.clipped = abs(delay) > abs(bound_s)
    if result.clipped:
        delay = float(np.sign(delay) * abs(bound_s))
    result.delay_s = float(delay)

    corrected = ll - delay * rs
    result.correlation_after = correlation(rs, corrected)
    result.l_spread_after_h = float(np.std(corrected))
    result.ok = True

    notes = [
        f"delay {result.delay_s * 1e9:+.0f} ns "
        f"+/- {result.delay_sigma_s * 1e9:.0f} ns from {int(inliers.sum())}/"
        f"{len(rs)} segments; L-vs-Rs correlation "
        f"{result.correlation_before:+.2f} -> {result.correlation_after:+.2f}"
    ]
    if result.clipped:
        notes.append(
            f"clipped to the {abs(bound_s) * 1e9:.0f} ns bound taken from the "
            f"measured skew - a larger value would be absorbing real inductance"
        )
    if result.l_spread_after_h > result.l_spread_before_h:
        notes.append(
            "the correction widened the inductance scatter, so the apparent "
            "correlation was probably noise; treat it as unconfirmed"
        )
    result.note = "; ".join(notes)
    return result


def pooled_kk_delay(
    spectra: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    bound_s: float,
    n_grid: int = 21,
    max_segments: int = 8,
    max_elements: int = 20,
    max_inductance_h: float = 50e-9,
    rs_reference_ohm: float = 0.0,
) -> CommonModeResult:
    """Shared delay from the pooled Kramers-Kronig residual.

    The fallback for when :func:`identify_common_delay` has no leverage - most
    often a card timed from a segment proxy, where the delay is uncertain to
    microseconds rather than nanoseconds.

    Getting this right turns on one point, and both obvious formulations get it
    wrong.  Rotating a spectrum by ``exp(-jw*tau)`` changes its apparent series
    inductance by ``-tau*Rs``, so:

    * fit the Voigt basis **with** a free ``jwL`` term and that term absorbs the
      rotation exactly - the residual is blind and the scan reports nothing;
    * fit it **without** the term and the basis, which is purely capacitive,
      cannot represent the cell's own inductance either - so the scan rotates
      until the *real* inductance is cancelled and reports a delay several
      times too large.

    What actually identifies the delay is a **bound on the inductance**.  A
    segment's lead inductance is a matter of via geometry and is a few
    nanohenries; a 6 us residual delay on a 14 mOhm segment looks like 84 nH.
    Constraining ``L`` to what the wiring can physically be therefore leaves
    the delay - and only the delay - to explain the rest.  ``max_inductance_h``
    is that prior, and it has to be a real one.

    Pooling then does the statistics: the delay is common to the card, so it
    moves every segment's residual together, while the individual inductances
    differ in size and sign and average out.

    The search bound still comes from the measured timing.  Both constraints
    matter and neither substitutes for the other.

    **What the estimate is worth.**  Whatever inductance the bound still allows
    is inductance the scan can trade against delay, so the answer is uncertain
    by ``max_inductance_h / Rs`` - about 3 us for a 50 nH bound on a 15 mOhm
    segment - and that is reported in ``delay_sigma_s`` rather than glossed.
    This is a coarse instrument: it is worth using on a card timed from a
    segment proxy, where the delay is unknown to ten microseconds, and worth
    nothing on a card timed from its own cell voltage to nanoseconds.  Pass
    ``rs_reference_ohm`` so the uncertainty can be stated in seconds.
    """
    from eis.validate import lin_kk

    chosen = list(spectra)[:max_segments]
    result = CommonModeResult(n_segments=len(chosen))
    if not chosen or bound_s <= 0:
        result.note = "no spectra or no bound for a pooled KK delay scan"
        return result

    grid = np.linspace(-abs(bound_s), abs(bound_s), max(n_grid, 3))
    scores = np.full(len(grid), np.inf)
    for i, tau in enumerate(grid):
        total, used = 0.0, 0
        # Score the *same* rotation the caller will apply - going through
        # InstrumentResponse rather than writing the exponential out again is
        # what keeps the two from drifting apart in sign.
        probe = InstrumentResponse(delay_s=float(tau))
        for segment in chosen:
            f, Z, sigma = spectra[segment]
            if len(f) < 6:
                continue
            fit = lin_kk(
                f, Z * probe.factor(f), sigma, max_elements=max_elements,
                with_inductance=True, max_inductance=max_inductance_h,
                max_sigma=0.0,
            )
            total += fit.rms_residual_pct
            used += 1
        if used:
            scores[i] = total / used
    if not np.isfinite(scores).any():
        result.note = "pooled KK scan produced no finite residual"
        return result

    best = int(np.argmin(scores))
    delay = float(grid[best])
    # Parabolic refinement between the grid points, which is worth doing: the
    # grid is coarse by design and the residual is smooth in tau.
    if 0 < best < len(grid) - 1 and np.all(np.isfinite(scores[best - 1:best + 2])):
        y0, y1, y2 = scores[best - 1:best + 2]
        denominator = y0 - 2 * y1 + y2
        if denominator > 0:
            step = 0.5 * (y0 - y2) / denominator
            delay += float(np.clip(step, -1.0, 1.0)) * (grid[1] - grid[0])

    baseline = float(scores[int(np.argmin(np.abs(grid)))])
    improvement = baseline - float(scores[best])
    result.delay_s = float(np.clip(delay, -abs(bound_s), abs(bound_s)))
    result.clipped = abs(delay) >= abs(bound_s)

    # The uncertainty has two parts and the second dominates: the grid step,
    # and the delay that the still-permitted inductance could have been.
    inductance_equivalent = (
        max_inductance_h / rs_reference_ohm if rs_reference_ohm > 0 else 0.0
    )
    result.delay_sigma_s = float(np.hypot(
        abs(grid[1] - grid[0]), inductance_equivalent
    )) if improvement > 0 else float("inf")
    result.ok = improvement > 0.05 * max(baseline, 1e-12)
    result.note = (
        f"pooled KK scan over {len(chosen)} segments: delay "
        f"{result.delay_s * 1e9:+.0f} ns "
        f"+/- {result.delay_sigma_s * 1e9:.0f} ns, mean residual "
        f"{baseline:.2f}% -> {scores[best]:.2f}%"
        + (f"; the +/- is dominated by the {max_inductance_h * 1e9:.0f} nH "
           f"inductance the bound still allows, which no scan over a finite "
           f"band can separate from a delay"
           if inductance_equivalent > abs(grid[1] - grid[0]) else "")
        + ("" if result.ok else "; improvement too small to act on")
    )
    return result


def delay_from_reference(
    f: np.ndarray,
    Z_plate: np.ndarray,
    Z_reference: np.ndarray,
    band_hz: tuple[float, float] = (200.0, 4000.0),
) -> CommonModeResult:
    """Shared delay from an independent cell-level spectrum.

    When a reference instrument measured the whole cell over the same band, the
    summed segment admittance must reproduce it.  The phase difference between
    the two is then a direct measurement of the common-mode delay, with no
    identifying assumption at all - this is the strongest anchor available and
    is preferred over :func:`identify_common_delay` whenever the reference
    exists.
    """
    f = np.asarray(f, float)
    mask = (f >= band_hz[0]) & (f <= band_hz[1])
    if mask.sum() < 4:
        return CommonModeResult(
            note=f"only {int(mask.sum())} points inside {band_hz} Hz; "
                 f"the reference cannot fix a delay"
        )
    fm = f[mask]
    delta = np.unwrap(
        np.angle(np.asarray(Z_plate, complex)[mask]
                 / np.asarray(Z_reference, complex)[mask])
    )
    slope, _ = np.polyfit(2 * np.pi * fm, delta, 1)
    residual = delta - np.polyval([slope, np.mean(delta - slope * 2 * np.pi * fm)],
                                  2 * np.pi * fm)
    sigma = float(np.std(residual) / max(np.std(2 * np.pi * fm), 1e-12)
                  / np.sqrt(max(len(fm) - 2, 1)))
    return CommonModeResult(
        delay_s=float(slope), delay_sigma_s=sigma, n_segments=0, ok=True,
        note=f"delay {slope * 1e9:+.0f} ns measured against the reference "
             f"instrument over {len(fm)} points",
    )


# ---------------------------------------------------------------------------
# 4. In-plane crosstalk
# ---------------------------------------------------------------------------

@dataclass
class CrosstalkModel:
    """Mixing of neighbouring segment currents through the diffusion medium."""

    segments: list[int] = field(default_factory=list)
    matrix: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    alpha: float = 0.0
    regularisation: float = 1e-3
    mean_neighbours: float = 0.0
    note: str = ""

    @property
    def active(self) -> bool:
        return self.alpha > 0 and self.matrix.size > 0


def build_crosstalk_model(
    segments: list[int],
    coords: dict[int, tuple[float, float, float, float]],
    alpha: float,
    regularisation: float = 1e-3,
    neighbour_scale: float = 1.6,
) -> CrosstalkModel:
    """Mixing matrix ``M`` with ``Y_measured = M @ Y_true``.

    Each segment keeps ``1 - alpha`` of its own current and receives ``alpha``
    spread evenly over its geometric neighbours.  Neighbours are the segments
    whose centres are within ``neighbour_scale`` times the segment pitch, which
    picks up the four edge-sharing neighbours of a rectangular layout without
    needing the layout to be declared.
    """
    ordered = [s for s in sorted(segments) if s in coords]
    n = len(ordered)
    if n < 2 or alpha <= 0:
        return CrosstalkModel(
            segments=ordered, matrix=np.eye(n), alpha=0.0,
            regularisation=regularisation,
            note="crosstalk correction disabled" if alpha <= 0
                 else "too few located segments for a neighbour graph",
        )

    centres = np.array([coords[s][:2] for s in ordered], float)
    halves = np.array([coords[s][2:] for s in ordered], float)
    pitch = float(np.median(2.0 * halves.max(axis=1)))
    distance = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2)
    adjacency = (distance > 0) & (distance <= neighbour_scale * pitch)

    M = np.eye(n) * (1.0 - alpha)
    degree = adjacency.sum(axis=1)
    for i in range(n):
        if degree[i]:
            M[i, adjacency[i]] += alpha / degree[i]
        else:
            M[i, i] = 1.0                       # isolated segment keeps its own
    return CrosstalkModel(
        segments=ordered, matrix=M, alpha=float(alpha),
        regularisation=regularisation,
        mean_neighbours=float(degree.mean()),
        note=f"alpha={alpha:.3f} over a graph averaging {degree.mean():.1f} "
             f"neighbours per segment",
    )


def deconvolve_crosstalk(
    model: CrosstalkModel, admittance: np.ndarray
) -> tuple[np.ndarray, float]:
    """Invert the mixing.  ``admittance`` is ``(n_segments, n_frequencies)``.

    Returns the deconvolved admittance and the median relative change, which is
    the number that says whether the correction did anything worth doing.
    """
    Y = np.asarray(admittance, complex)
    if not model.active or Y.shape[0] != model.matrix.shape[0]:
        return Y, 0.0
    M = model.matrix
    lam = model.regularisation
    normal = M.T @ M + lam * np.eye(M.shape[0])
    corrected = np.linalg.solve(normal, M.T @ Y)
    change = np.abs(corrected - Y) / np.maximum(np.abs(Y), 1e-30)
    return corrected, float(np.median(change))


# ---------------------------------------------------------------------------
# Uncertainty from the correction chain
# ---------------------------------------------------------------------------

def inflate_sigma_for_timing(
    sigma_rel: np.ndarray, f: np.ndarray, tau_sigma_s: float
) -> np.ndarray:
    """Fold a timing uncertainty into the per-point relative uncertainty.

    ``sigma_rel`` doubles as the phase uncertainty in radians (Bendat-Piersol
    gives the same expression for both), so an unknown delay of ``sigma_tau``
    adds ``2*pi*f*sigma_tau`` in quadrature.  Without this a segment on a card
    timed from a proxy - good to microseconds, not nanoseconds - reports the
    same error bar at 3.8 kHz as one timed from its own cell voltage, which is
    off by three orders of magnitude.
    """
    sigma_rel = np.asarray(sigma_rel, float)
    if not np.isfinite(tau_sigma_s) or tau_sigma_s <= 0:
        return sigma_rel
    timing = 2.0 * np.pi * np.asarray(f, float) * abs(tau_sigma_s)
    return np.sqrt(sigma_rel**2 + timing**2)
