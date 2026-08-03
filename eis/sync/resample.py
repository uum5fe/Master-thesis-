"""Applying a measured delay to a signal.

Alignment is done **in the time domain, before any FFT**.  Rotating the final
impedance by ``exp(-j*2*pi*f*tau)`` fixes the phase but leaves three problems
untouched: the coherence lost because misaligned samples were averaged inside
each analysis window, the tone detection that runs on that coherence, and a
delay that changes during the record, which no constant rotation can express.
"""

from __future__ import annotations

import numpy as np
from scipy.special import i0


def integer_and_fractional(tau_s: float, fs: float) -> tuple[int, float]:
    """Split a delay into an integer sample count and a remainder in [-0.5, 0.5)."""
    samples = tau_s * fs
    whole = int(np.round(samples))
    return whole, float(samples - whole)


# ---------------------------------------------------------------------------
# Constant delay: exact band-limited shift via an FFT phase ramp
# ---------------------------------------------------------------------------

def fractional_delay_fft(x: np.ndarray, tau_s: float, fs: float) -> np.ndarray:
    """Delay ``x`` by ``tau_s`` seconds (negative advances it).

    Implemented as a phase ramp on the analytic spectrum, which is the exact
    band-limited interpolator for a periodic extension of the record.  Because
    the extension is periodic, ``|tau|`` should be small relative to the record
    - use it for the sub-sample remainder and handle whole samples by slicing.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spectrum = spectrum * np.exp(-1j * 2 * np.pi * freqs * tau_s)
    if n % 2 == 0:                    # keep the Nyquist bin real
        spectrum[-1] = spectrum[-1].real
    return np.fft.irfft(spectrum, n)


# ---------------------------------------------------------------------------
# Time-varying delay: polyphase fractional-delay filter bank
# ---------------------------------------------------------------------------

def _fractional_delay_bank(
    n_phases: int = 512, taps: int = 31, beta: float = 8.6, cutoff: float = 0.92
) -> np.ndarray:
    """Kaiser-windowed sinc interpolators for delays ``k/n_phases``, k in [0, 1).

    ``taps`` is odd so that the base delay ``(taps-1)/2`` is an integer and the
    filter can be anchored on a sample index exactly.
    """
    if taps % 2 == 0:
        raise ValueError("taps must be odd")
    m = np.arange(taps)
    phases = np.arange(n_phases) / n_phases
    delay = (taps - 1) / 2.0 + phases                     # (n_phases,)
    t = m[None, :] - delay[:, None]                       # (n_phases, taps)

    h = cutoff * np.sinc(cutoff * t)
    arg = 1.0 - (2.0 * t / (taps - 1)) ** 2
    window = np.where(arg > 0, i0(beta * np.sqrt(np.clip(arg, 0, None))) / i0(beta), 0.0)
    h = h * window
    return h / h.sum(axis=1, keepdims=True)


_BANK: np.ndarray | None = None
_BANK_TAPS = 63
#: The phase quantisation is the accuracy floor of the interpolator: rounding
#: to the nearest of N phases leaves a timing error of 1/(2N) samples, which at
#: 0.8*Nyquist costs a relative amplitude error of about pi*0.8/(2N).  4096
#: phases put that below 3e-4, well under the measurement noise.
_BANK_PHASES = 4096


def _bank() -> np.ndarray:
    global _BANK
    if _BANK is None:
        _BANK = _fractional_delay_bank(_BANK_PHASES, _BANK_TAPS)
    return _BANK


def resample_at(x: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Band-limited interpolation of ``x`` at arbitrary fractional ``positions``.

    ``positions`` are in units of input samples.  Positions outside the record
    are clamped to the edges, and the returned array reports them via NaN-free
    edge extension - callers trim the invalid margin.
    """
    x = np.asarray(x, dtype=np.float64)
    pos = np.asarray(positions, dtype=np.float64)
    bank = _bank()
    taps, n_phases = _BANK_TAPS, _BANK_PHASES
    half = (taps - 1) // 2

    base = np.floor(pos).astype(np.int64)
    frac = pos - base
    # Round (not truncate) to the nearest phase; truncation would leave a
    # systematic half-step timing bias.
    phase = np.rint(frac * n_phases).astype(np.int64)
    carry = phase >= n_phases
    phase[carry] -= n_phases
    base = base + carry

    # padded[j] == x[j - half], so anchoring the filter at index `base` of
    # `padded` evaluates x at `base + frac`.  Positions outside the record are
    # clamped; callers trim that margin with `valid_span`.
    padded = np.pad(x, (half, half + 1), mode="edge")
    start = np.clip(base, 0, len(x) - 1)

    out = np.zeros(len(pos), dtype=np.float64)
    for m in range(taps):
        out += bank[phase, m] * padded[start + m]
    return out


def advance_affine(
    x: np.ndarray, fs: float, tau0_s: float, rate_ppm: float = 0.0
) -> np.ndarray:
    """Undo an affine delay ``tau(t) = tau0 + (rate_ppm*1e-6) * t``.

    Produces ``y[n] = x(t_n + tau(t_n))`` with ``t_n = n/fs``, i.e. the signal
    resampled onto the reference card's time base.  Handles the static offset
    and the clock-rate error in a single band-limited interpolation, so no
    intermediate resampling stage is needed.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    eps = rate_ppm * 1e-6
    idx = np.arange(n, dtype=np.float64)
    positions = idx * (1.0 + eps) + tau0_s * fs
    if abs(eps) < 1e-15 and abs(tau0_s * fs) < 1e-9:
        return x.copy()
    return resample_at(x, positions)


def valid_span(n: int, fs: float, tau0_s: float, rate_ppm: float = 0.0,
               guard_samples: int = 32) -> tuple[int, int]:
    """Index range of ``advance_affine`` output that used only real samples."""
    eps = rate_ppm * 1e-6
    shift = tau0_s * fs
    lo = int(np.ceil(max(0.0, -shift) / (1.0 + eps))) + guard_samples
    hi_pos = n - 1 - guard_samples
    hi = int(np.floor((hi_pos - shift) / (1.0 + eps)))
    return max(lo, 0), max(min(hi, n), 0)
