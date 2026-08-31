"""Impedance estimation from synchronised time series.

Local transfer impedance
------------------------
In a segmented cell every segment sees the *same* cell voltage while carrying
its own current, so the quantity of interest is the local transfer impedance

    Z_k(f) = U_cell(f) / I_k(f)

estimated as a transfer function with the segment current as input and the
cell voltage as output.  The segments are electrically in parallel, which
gives the whole-plate consistency check used in :mod:`eis.validate`.

What this module does differently from a plain Welch ratio
----------------------------------------------------------
* **Ensemble-level coherence gating.**  Windows contaminated by a load
  transient, a purge event or a dropout are rejected *before* averaging,
  instead of discarding frequency points afterwards.  A 120 s fuel-cell record
  is not stationary; one bad window otherwise poisons the whole average and no
  output-side gate can undo it.
* **Unbiased estimator.**  ``H1 = S_iu/S_ii`` is biased low by noise on the
  current - its expectation is ``Z * gamma^2`` in the noisy-input limit, so at
  gamma^2 = 0.9 it under-reads |Z| by 10 %.  The default here is the total
  least-squares estimator ``Hv``, with H1 and H2 reported as bracketing bounds.
* **Leakage-free synchronous DFT.**  When the excitation is a designed
  multisine, analysing an integer number of base periods with a rectangular
  window puts each tone in exactly one bin: no scalloping, no picket-fence
  error, no window correction.
* **Per-point uncertainty.**  Every returned point carries a standard
  deviation derived from its coherence and the number of retained windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import filtfilt, get_window, iirnotch


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class SpectralResult:
    """Impedance spectrum of one segment."""

    f: np.ndarray                 #: frequency [Hz]
    Z: np.ndarray                 #: complex impedance [Ohm]
    coherence: np.ndarray         #: gamma^2 in [0, 1]
    sigma_rel: np.ndarray         #: relative 1-sigma on |Z| (= sigma_phi in rad)
    n_windows_total: int = 0
    n_windows_used: int = 0
    n_eff: float = 0.0            #: effective independent averages
    estimator: str = "hv"
    method: str = "welch"
    Z_h1: np.ndarray | None = None
    Z_h2: np.ndarray | None = None
    rejected_reason: dict[str, int] = field(default_factory=dict)
    note: str = ""

    @property
    def sigma_real(self) -> np.ndarray:
        return np.abs(self.Z) * self.sigma_rel

    @property
    def sigma_imag(self) -> np.ndarray:
        return np.abs(self.Z) * self.sigma_rel

    def band(self, f_min: float, f_max: float) -> "SpectralResult":
        m = (self.f >= f_min) & (self.f <= f_max)
        return self._mask(m)

    def gate(self, min_coherence: float) -> "SpectralResult":
        return self._mask(self.coherence >= min_coherence)

    def _mask(self, m: np.ndarray) -> "SpectralResult":
        return SpectralResult(
            f=self.f[m], Z=self.Z[m], coherence=self.coherence[m],
            sigma_rel=self.sigma_rel[m],
            n_windows_total=self.n_windows_total, n_windows_used=self.n_windows_used,
            n_eff=self.n_eff, estimator=self.estimator, method=self.method,
            Z_h1=None if self.Z_h1 is None else self.Z_h1[m],
            Z_h2=None if self.Z_h2 is None else self.Z_h2[m],
            rejected_reason=dict(self.rejected_reason), note=self.note,
        )


# ---------------------------------------------------------------------------
# Pre-filtering
# ---------------------------------------------------------------------------

def apply_notch(
    x: np.ndarray, fs: float, freqs: list[float], q: float = 30.0
) -> np.ndarray:
    """Zero-phase IIR notches at mains harmonics.

    ``filtfilt`` is used so the filter contributes **no** phase shift - a
    causal notch would inject exactly the kind of frequency-dependent delay the
    synchronisation stage works to remove.
    """
    out = np.asarray(x, float).copy()
    for f0 in freqs:
        if 0 < f0 < fs / 2:
            b, a = iirnotch(f0, q, fs)
            out = filtfilt(b, a, out)
    return out


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def _detrend(block: np.ndarray, how: str) -> np.ndarray:
    if how == "constant":
        return block - block.mean(axis=-1, keepdims=True)
    if how == "linear":
        n = block.shape[-1]
        t = np.arange(n, dtype=float)
        t -= t.mean()
        denom = float((t * t).sum())
        slope = (block * t).sum(axis=-1, keepdims=True) / denom
        return block - block.mean(axis=-1, keepdims=True) - slope * t
    return block


def _frame(x: np.ndarray, nperseg: int, hop: int) -> np.ndarray:
    """Zero-copy view of ``x`` as overlapping frames ``(n_windows, nperseg)``."""
    n = len(x)
    if n < nperseg:
        raise ValueError(f"record of {n} samples is shorter than nperseg={nperseg}")
    count = 1 + (n - nperseg) // hop
    stride = x.strides[0]
    return np.lib.stride_tricks.as_strided(
        x, shape=(count, nperseg), strides=(hop * stride, stride), writeable=False
    )


def _overlap_correlation(win: np.ndarray, hop: int) -> float:
    """Sum of squared overlap correlations, for the effective-average count.

    Overlapping Welch windows are not independent.  The variance of the average
    is reduced by ``n_eff = L / (1 + 2*sum_k rho_k^2)`` rather than ``L``;
    ignoring this over-states the precision by a few per cent.
    """
    denom = float((win * win).sum())
    if denom <= 0:
        return 0.0
    total = 0.0
    k = hop
    while k < len(win):
        total += (float((win[:-k] * win[k:]).sum()) / denom) ** 2
        k += hop
    return total


def window_spectra(
    x: np.ndarray, fs: float, nperseg: int, noverlap: int,
    window: str = "hann", detrend: str = "linear",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-window one-sided FFTs.  Returns ``(freqs, X[L, F], n_eff_factor)``."""
    hop = nperseg - noverlap
    win = get_window(window, nperseg, fftbins=True) if window != "boxcar" else np.ones(nperseg)
    frames = np.array(_frame(np.ascontiguousarray(x, dtype=np.float64), nperseg, hop))
    frames = _detrend(frames, detrend) * win
    X = np.fft.rfft(frames, axis=-1)
    freqs = np.fft.rfftfreq(nperseg, 1.0 / fs)
    return freqs, X, 1.0 + 2.0 * _overlap_correlation(win, hop)


# ---------------------------------------------------------------------------
# Ensemble gating
# ---------------------------------------------------------------------------

def gate_windows(
    X: np.ndarray, Y: np.ndarray, band: np.ndarray,
    amplitude_mad_k: float = 4.0, phase_outlier_rad: float = 0.5,
    max_rejected_fraction: float = 0.5,
) -> tuple[np.ndarray, dict[str, int], str]:
    """Reject contaminated windows before averaging.

    Returns ``(keep_mask, reasons, note)``.
    """
    n = X.shape[0]
    keep = np.ones(n, dtype=bool)
    reasons: dict[str, int] = {}
    if n < 4:
        return keep, reasons, "too few windows to gate"

    # 1. Signal level on both channels.  The input level catches a dropout in
    #    the excitation; the *output* level catches a disturbance that enters
    #    on the response side only - a load transient or a purge event, which
    #    leaves the current spectrum untouched and would otherwise sail
    #    straight through an input-only gate.
    for name, spectrum in (("excitation_level", X), ("response_level", Y)):
        level = np.sqrt(np.mean(np.abs(spectrum[:, band]) ** 2, axis=1))
        med = float(np.median(level))
        mad = float(np.median(np.abs(level - med))) * 1.4826
        if mad <= 0:
            continue
        bad = np.abs(level - med) > amplitude_mad_k * mad
        reasons[name] = int((bad & keep).sum())
        keep &= ~bad

    # 2. Phase consistency: a transient rotates the transfer function of the
    #    window it lands in, which shows up as an outlying band-average phase.
    cross = np.conj(X[:, band]) * Y[:, band]
    weight = np.abs(X[:, band]) ** 2
    resultant = (cross * weight).sum(axis=1)
    phase = np.angle(resultant)
    ref = np.angle(resultant[keep].sum()) if keep.any() else np.angle(resultant.sum())
    delta = np.abs(np.angle(np.exp(1j * (phase - ref))))
    bad = delta > phase_outlier_rad
    reasons["phase_outlier"] = int((bad & keep).sum())
    keep &= ~bad

    note = ""
    if keep.sum() < n * (1.0 - max_rejected_fraction):
        note = (
            f"gate wanted to reject {n - int(keep.sum())}/{n} windows "
            f"(> {max_rejected_fraction:.0%}); gating disabled and the record "
            f"flagged as non-stationary rather than silently trimmed"
        )
        keep = np.ones(n, dtype=bool)
        reasons = {"gate_disabled": n}
    if keep.sum() < 2:
        keep = np.ones(n, dtype=bool)
        reasons = {"gate_disabled": n}
    return keep, {k: v for k, v in reasons.items() if v}, note


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def transfer_estimators(
    Sxx: np.ndarray, Syy: np.ndarray, Sxy: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(H1, H2, Hv, gamma2)`` for input x, output y.

    With ``Sxy = <conj(X) Y>``:

    * ``H1 = Sxy/Sxx``        - biased **low** by noise on the input.  Its
      magnitude is under-read by a factor gamma^2 in the noisy-input limit, so
      at gamma^2 = 0.9 it reports |Z| about 10 % small.
    * ``H2 = Syy/conj(Sxy)``  - biased **high** by noise on the output, by the
      same factor.  ``|H2/H1| = 1/gamma^2`` exactly, so the pair genuinely
      brackets the truth and the width of that bracket *is* the noise
      diagnostic.
    * ``Hv``                  - the scale-invariant total-least-squares
      solution: ``sqrt(Syy/Sxx)`` carrying the phase of ``Sxy``, which is the
      geometric mean of H1 and H2.

    Why Hv is written this way rather than as the eigenvector of the raw
    spectral matrix: total least squares is **not scale invariant**, and the
    two channels here are a current in amps and a voltage in volts whose
    spectra differ by orders of magnitude.  The eigenvector of the
    unnormalised matrix collapses onto whichever channel carries the larger
    numbers - in this pipeline the current - so a naive "unbiased" estimator
    silently returns H1 and the bias it was meant to remove stays put.
    Normalising both channels to unit power first removes the arbitrary scale,
    and the eigenvector of the normalised matrix is the expression above.

    Bias, stated plainly: Hv is unbiased when the two channels carry *equal
    noise-to-signal ratios*.  With noise on the input alone it under-reads by
    gamma rather than gamma^2 - better than H1, but not exact.  Without
    knowing the noise ratio no estimator can do better, which is why H1 and H2
    are returned alongside as bounds.
    """
    eps = np.finfo(float).tiny
    Sxx_r = np.real(Sxx)
    Syy_r = np.real(Syy)

    H1 = Sxy / (Sxx_r + eps)
    H2 = Syy_r / (np.conj(Sxy) + eps)
    Hv = np.sqrt(np.maximum(Syy_r, 0.0) / (Sxx_r + eps)) * np.exp(1j * np.angle(Sxy))

    gamma2 = np.clip(np.abs(Sxy) ** 2 / (Sxx_r * Syy_r + eps), 0.0, 1.0)
    return H1, H2, Hv, gamma2


def relative_uncertainty(gamma2: np.ndarray, n_eff: float) -> np.ndarray:
    """Bendat-Piersol random error of a transfer-function estimate.

    ``sigma_|H|/|H| = sigma_phi[rad] = sqrt((1-gamma^2)/(2 gamma^2 n_eff))``.
    """
    g = np.clip(gamma2, 1e-9, 1.0 - 1e-12)
    return np.sqrt((1.0 - g) / (2.0 * g * max(n_eff, 1.0)))



# ---------------------------------------------------------------------------
# Stepped multisine
# ---------------------------------------------------------------------------
#
# WHY WELCH IS THE WRONG DEFAULT FOR THIS MEASUREMENT
# ---------------------------------------------------
# Welch averages periodograms over the WHOLE record.  That is the right thing
# for a stationary broadband excitation and the wrong thing for one that
# steps: each tone is present for its own slice of the record and absent for
# the rest, so
#
#   * its estimate is diluted by the fraction of the record it was off for,
#     and the dilution is worst for the tones with the shortest dwells --
#     which, on a sweep that dwells a fixed number of CYCLES, are the
#     high-frequency ones; and
#   * the coherence that is supposed to GATE the result is diluted by the
#     same factor, so the gate discards the tones that were on for the
#     shortest time rather than the ones that are least trustworthy.
#
# Both effects push the same way and both worsen with frequency, which is
# what turns "the spectrum stops around a kilohertz" into something that
# looks like a property of the cell rather than of the estimator.
#
# The estimator below fits the tones of each dwell jointly, over that dwell
# only.  Nothing is averaged across a stretch where the tone was not applied,
# and the residual left by the joint fit is noise rather than the other tones
# of the group.


def detect_dwells(x: np.ndarray, fs: float, frame_s: float = 0.02,
                  change_tol: float = 0.25, silence_factor: float = 0.15,
                  min_dwell_s: float = 0.01, trim_frac: float = 0.08
                  ) -> list[tuple[int, int]]:
    """Intervals of constant excitation content, found without a tone list.

    Segmenting on SILENCE does not work: consecutive steps often abut with
    no gap, and a record that is mostly excitation offers no quiet stretch
    from which to estimate what quiet looks like.  So the primary signal is
    a change of CONTENT -- normalised frame spectra are direction vectors,
    and a step change rotates them.  Normalising also means that a change of
    level alone, which is what the amplitude taper of a designed multisine
    produces, is not mistaken for a change of content.

    The comparison is between the mean of several frames either side of a
    candidate boundary rather than between adjacent frames: a step change is
    sustained and survives the averaging, while frame-to-frame jitter does
    not.
    """
    x = np.asarray(x, float)
    n = len(x)
    nper = max(64, 1 << int(np.round(np.log2(max(frame_s * fs, 64)))))
    if n < 4 * nper:
        return [(0, n)]
    step = nper // 2
    starts = np.arange(0, n - nper + 1, step)
    win = np.hanning(nper)

    spectra = np.empty((len(starts), nper // 2 + 1))
    rms = np.empty(len(starts))
    for i, a in enumerate(starts):
        seg = x[a:a + nper] - x[a:a + nper].mean()
        rms[i] = np.sqrt(np.mean(seg ** 2))
        spectra[i] = np.abs(np.fft.rfft(seg * win))

    norm = np.linalg.norm(spectra, axis=1)
    unit = spectra / np.where(norm > 0, norm, 1.0)[:, None]
    W = max(2, min(int(round(0.05 / (step / fs))) or 2, max(2, len(starts) // 8)))
    dist = np.zeros(len(starts))
    for i in range(W, len(starts) - W):
        before, after = unit[i - W:i].mean(axis=0), unit[i:i + W].mean(axis=0)
        nb, na = np.linalg.norm(before), np.linalg.norm(after)
        if nb > 0 and na > 0:
            dist[i] = 1.0 - float(np.dot(before, after) / (nb * na))

    cuts = [0]
    for i in range(1, len(dist) - 1):
        if (dist[i] >= change_tol and dist[i] >= dist[i - 1]
                and dist[i] >= dist[i + 1] and i - cuts[-1] >= W):
            cuts.append(i)
    cuts.append(len(starts))

    spans = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        s0, s1 = int(starts[lo]), int(min(n, starts[hi - 1] + nper))
        if s1 - s0 >= max(min_dwell_s * fs, 2 * nper):
            spans.append((s0, s1, float(np.median(rms[lo:hi]))))
    if not spans:
        return [(0, n)]

    loud = float(np.median([lvl for _a, _b, lvl in spans]))
    out = []
    for s0, s1, lvl in spans:
        if lvl < silence_factor * loud:
            continue                        # a gap between steps, not a dwell
        cut = int(trim_frac * (s1 - s0))
        out.append((s0 + cut, s1 - cut))
    return out or [(0, n)]


def _local_floor(mag: np.ndarray, n_bands: int = 160) -> np.ndarray:
    """Band medians as a local noise floor, interpolated back.

    Uniform bands, not logarithmic: a log grid's low-frequency bands are only
    a few hertz wide, so a band can hold one tone and nothing else, its
    median becomes that tone, and the tone is then not detected.
    """
    n = len(mag)
    if n < 16:
        return np.full(n, float(np.median(mag)) if n else 0.0)
    n_bands = int(max(4, min(n_bands, n // 16)))
    edges = np.linspace(0, n, n_bands + 1).astype(int)
    centres = 0.5 * (edges[:-1] + edges[1:])
    values = np.array([np.median(mag[a:b]) if b > a else np.nan
                       for a, b in zip(edges[:-1], edges[1:])])
    ok = np.isfinite(values)
    if ok.sum() < 2:
        return np.full(n, float(np.median(mag)))
    return np.interp(np.arange(n), centres[ok], values[ok])


def tones_in_window(y: np.ndarray, fs: float, f_lo: float, f_hi: float,
                    peak_db: float = 12.0, max_tones: int = 96,
                    oversample: int = 8) -> np.ndarray:
    """Frequencies standing clear of the local noise floor in one window.

    The threshold also scales with the window's cell count: the magnitude of
    a white-noise spectrum is Rayleigh, so a fixed "12 dB over the floor" is
    a false-alarm rate per bin, and a long window has a great many bins.
    """
    y = np.asarray(y, float)
    n = len(y)
    if n < 64:
        return np.empty(0)
    y = y - y.mean()
    m = 1 << int(np.ceil(np.log2(max(n * oversample, 256))))
    spec = np.abs(np.fft.rfft(y * np.hanning(n), m))
    freqs = np.fft.rfftfreq(m, 1.0 / fs)
    band = (freqs >= f_lo) & (freqs <= min(f_hi, 0.5 * fs))
    if band.sum() < 8:
        return np.empty(0)
    s, f = spec[band], freqs[band]

    n_cells = max(8, n // 2)
    p_fa = min(0.5, 0.01 / n_cells)
    auto_db = 20.0 * np.log10(np.sqrt(-2.0 * np.log(p_fa))
                              / np.sqrt(2.0 * np.log(2.0)))
    thr = _local_floor(s) * 10 ** (max(peak_db, auto_db) / 20.0)

    guard_hz = 6.0 * fs / n                 # ~1.5 Hann main lobes
    idx = [k for k in range(1, len(s) - 1)
           if s[k] >= thr[k] and s[k] >= s[k - 1] and s[k] >= s[k + 1]]
    idx.sort(key=lambda k: -s[k])
    kept: list[int] = []
    for k in idx:
        if all(abs(f[k] - f[j]) > guard_hz for j in kept):
            kept.append(k)
        if len(kept) >= max_tones:
            break
    if not kept:
        return np.empty(0)
    out = []
    for k in kept:                          # sub-bin parabolic interpolation
        y0, y1, y2 = float(s[k - 1]), float(s[k]), float(s[k + 1])
        den = y0 - 2.0 * y1 + y2
        d = float(np.clip(0.5 * (y0 - y2) / den, -0.5, 0.5)) if den else 0.0
        out.append(float(f[k] + d * (f[1] - f[0])))
    return np.sort(np.array(out))


def _tone_phasors(y: np.ndarray, fs: float, freqs: np.ndarray,
                  detrend: str = "linear") -> tuple[np.ndarray, float]:
    """Least-squares phasors of several SIMULTANEOUS tones, and the residual.

    Fitted together.  One at a time leaves the other tones of the group in
    each one's residual, which makes the apparent noise a count of the group
    size, -10*log10(N-1) dB, rather than a measurement of anything.
    """
    y = np.asarray(y, float)
    n = len(y)
    t = np.arange(n) / fs
    cols: list[np.ndarray] = []
    for f in freqs:
        w = 2 * np.pi * f
        cols += [np.cos(w * t), np.sin(w * t)]
    cols.append(np.ones(n))
    if detrend == "linear":
        cols.append(t - t.mean())
    D = np.column_stack(cols)
    p, *_ = np.linalg.lstsq(D, y, rcond=None)
    resid = y - D @ p
    dof = max(n - D.shape[1], 1)
    sigma = float(np.sqrt(float(np.dot(resid, resid)) / dof))
    phasors = np.array([complex(p[2 * i], -p[2 * i + 1])
                        for i in range(len(freqs))], dtype=complex)
    return phasors, sigma


def impedance_stepped_multisine(
    current: np.ndarray, voltage: np.ndarray, fs: float,
    dwells: list[tuple[int, int]] | None = None,
    tones_hz: list[float] | None = None,
    f_min: float = 1.0, f_max: float = 4000.0,
    peak_db: float = 12.0, detrend: str = "linear",
) -> SpectralResult:
    """Z = U/I at the tones of each dwell, fitted per dwell.

    `dwells` and `tones_hz` are optional: when they are not given the
    schedule is read off the excitation itself.  For this measurement that
    is a property of the data, and requiring it to be restated in a config
    file is how a correct answer comes to depend on a setting nobody
    remembers to change.

    UNCERTAINTY.  A stepped multisine gives one window per dwell, so there
    is no ensemble to take a variance over.  It is propagated from the two
    joint-fit residuals instead, which is a better basis anyway: it is the
    noise on THIS tone in THIS dwell rather than the scatter of a quantity
    across windows in which different tones were present.

    The reported "coherence" is the equivalent quantity for a single fit --
    1 / (1 + 1/SNR) per channel, combined -- so a caller that gates on
    coherence keeps working and gates on something meaningful.
    """
    current = np.asarray(current, float)
    voltage = np.asarray(voltage, float)
    n = min(len(current), len(voltage))
    current, voltage = current[:n], voltage[:n]

    spans = dwells if dwells is not None else detect_dwells(voltage, fs)
    f_hi = min(f_max, 0.45 * fs)

    fs_out: list[float] = []
    zs: list[complex] = []
    coh: list[float] = []
    sig: list[float] = []
    for a, b in spans:
        yv, yi = voltage[a:b], current[a:b]
        if len(yv) < 64:
            continue
        freqs = (np.asarray(tones_hz, float) if tones_hz is not None
                 else tones_in_window(yv, fs, f_min, f_hi, peak_db=peak_db))
        freqs = freqs[(freqs >= f_min) & (freqs <= f_hi)] if freqs.size else freqs
        if freqs.size == 0:
            continue
        if 2 * len(freqs) + 2 > len(yv) // 2:
            continue                        # window too short for the group
        ph_v, sigma_v = _tone_phasors(yv, fs, freqs, detrend)
        ph_i, sigma_i = _tone_phasors(yi, fs, freqs, detrend)
        for k, f in enumerate(freqs):
            Av, Ai = ph_v[k], ph_i[k]
            if not (np.isfinite(Av) and np.isfinite(Ai)) or Ai == 0:
                continue
            # Per-tone SNR against the noise the JOINT fit left behind.
            snr_v = (abs(Av) ** 2 / 2.0) / max(sigma_v ** 2, 1e-300)
            snr_i = (abs(Ai) ** 2 / 2.0) / max(sigma_i ** 2, 1e-300)
            gamma = 1.0 / ((1.0 + 1.0 / snr_v) * (1.0 + 1.0 / snr_i))
            # Relative error on a ratio adds in quadrature.
            rel = float(np.sqrt(1.0 / max(snr_v, 1e-300)
                                + 1.0 / max(snr_i, 1e-300)))
            fs_out.append(float(f))
            zs.append(complex(Av / Ai))
            coh.append(float(np.clip(gamma, 0.0, 1.0)))
            sig.append(rel)

    if not fs_out:
        return SpectralResult(
            f=np.empty(0), Z=np.empty(0, complex), coherence=np.empty(0),
            sigma_rel=np.empty(0), n_windows_total=len(spans),
            n_windows_used=0, method="stepped_multisine",
            note="no tones stood clear of the noise in any dwell")

    order = np.argsort(np.asarray(fs_out))
    return SpectralResult(
        f=np.asarray(fs_out)[order],
        Z=np.asarray(zs, complex)[order],
        coherence=np.asarray(coh)[order],
        sigma_rel=np.asarray(sig)[order],
        n_windows_total=len(spans), n_windows_used=len(spans),
        n_eff=float(len(spans)), estimator="joint-tone-fit",
        method="stepped_multisine",
        note=f"{len(spans)} dwell(s); tones fitted jointly within each")


# ---------------------------------------------------------------------------
# Welch path
# ---------------------------------------------------------------------------

def impedance_welch(
    current: np.ndarray, voltage: np.ndarray, fs: float,
    nperseg: int = 8192, noverlap: int | None = None, window: str = "hann",
    detrend: str = "linear", f_min: float = 1.0, f_max: float = 4000.0,
    estimator: str = "hv", gate: bool = True,
    amplitude_mad_k: float = 4.0, phase_outlier_rad: float = 0.5,
    max_rejected_fraction: float = 0.5,
) -> SpectralResult:
    """Coherence-gated Welch estimate of ``Z = U/I``."""
    noverlap = nperseg // 2 if noverlap is None else noverlap
    n = min(len(current), len(voltage))
    freqs, X, overlap_factor = window_spectra(
        current[:n], fs, nperseg, noverlap, window, detrend
    )
    _, Y, _ = window_spectra(voltage[:n], fs, nperseg, noverlap, window, detrend)

    band = (freqs >= f_min) & (freqs <= f_max)
    if band.sum() == 0:
        raise ValueError(f"no FFT bins inside [{f_min}, {f_max}] Hz")

    if gate:
        keep, reasons, note = gate_windows(
            X, Y, band, amplitude_mad_k, phase_outlier_rad, max_rejected_fraction
        )
    else:
        keep, reasons, note = np.ones(X.shape[0], bool), {}, ""

    Xk, Yk = X[keep][:, band], Y[keep][:, band]
    Sxx = np.mean(np.abs(Xk) ** 2, axis=0)
    Syy = np.mean(np.abs(Yk) ** 2, axis=0)
    Sxy = np.mean(np.conj(Xk) * Yk, axis=0)

    H1, H2, Hv, gamma2 = transfer_estimators(Sxx, Syy, Sxy)
    n_eff = int(keep.sum()) / overlap_factor
    Z = {"h1": H1, "h2": H2, "hv": Hv}[estimator]

    return SpectralResult(
        f=freqs[band], Z=Z, coherence=gamma2,
        sigma_rel=relative_uncertainty(gamma2, n_eff),
        n_windows_total=int(X.shape[0]), n_windows_used=int(keep.sum()),
        n_eff=float(n_eff), estimator=estimator, method="welch",
        Z_h1=H1, Z_h2=H2, rejected_reason=reasons, note=note,
    )


# ---------------------------------------------------------------------------
# Synchronous (multisine) path
# ---------------------------------------------------------------------------

def synchronous_window_length(
    fs: float, base_frequency_hz: float, periods: int
) -> tuple[int, float]:
    """Samples per analysis window, and how far that is from a whole number."""
    exact = periods * fs / base_frequency_hz
    n = int(round(exact))
    return n, abs(exact - n)


def impedance_synchronous(
    current: np.ndarray, voltage: np.ndarray, fs: float,
    base_frequency_hz: float, tones_hz: list[float], periods: int = 4,
    estimator: str = "hv", gate: bool = True, detrend: str = "linear",
    amplitude_mad_k: float = 4.0, phase_outlier_rad: float = 0.5,
    max_rejected_fraction: float = 0.5,
) -> SpectralResult:
    """Leakage-free estimate at the designed multisine tones.

    Uses a rectangular window over an integer number of base periods, so each
    tone falls exactly on a DFT bin.  Windows are non-overlapping and therefore
    genuinely independent, which makes the scatter across them a direct,
    assumption-free uncertainty estimate.
    """
    nperseg, samples_error = synchronous_window_length(fs, base_frequency_hz, periods)
    # The tolerance is in *samples*, not relative: even a fraction of a sample
    # of window-length error moves the tones off their bins and reintroduces
    # exactly the leakage this path exists to avoid.
    if samples_error > 0.01:
        raise ValueError(
            f"{periods} periods of {base_frequency_hz} Hz at {fs} Hz is "
            f"{periods * fs / base_frequency_hz:.4f} samples - not an integer "
            f"(off by {samples_error:.3f} samples). Synchronous analysis needs "
            f"fs/f0 rational; use method='welch', or pick periods so that "
            f"periods*fs/f0 is a whole number."
        )

    n = min(len(current), len(voltage))
    if n < 2 * nperseg:
        raise ValueError(
            f"record holds {n/nperseg:.1f} analysis windows; at least 2 are "
            f"needed for a coherence estimate"
        )
    freqs, X, _ = window_spectra(current[:n], fs, nperseg, 0, "boxcar", detrend)
    _, Y, _ = window_spectra(voltage[:n], fs, nperseg, 0, "boxcar", detrend)

    df = fs / nperseg
    bins, kept_tones, off_bin = [], [], []
    for f_tone in tones_hz:
        exact = f_tone / df
        k = int(round(exact))
        if k <= 0 or k >= X.shape[1]:
            continue
        if abs(exact - k) > 0.01:
            off_bin.append(f_tone)
            continue
        bins.append(k)
        kept_tones.append(k * df)
    if not bins:
        raise ValueError("none of the configured tones landed on a DFT bin")
    bins = np.asarray(bins)

    band = np.zeros(X.shape[1], dtype=bool)
    band[bins] = True
    if gate:
        keep, reasons, note = gate_windows(
            X, Y, band, amplitude_mad_k, phase_outlier_rad, max_rejected_fraction
        )
    else:
        keep, reasons, note = np.ones(X.shape[0], bool), {}, ""
    if off_bin:
        note = (note + "; " if note else "") + (
            f"{len(off_bin)} configured tone(s) are not on a bin and were "
            f"dropped: {off_bin[:5]}"
        )

    Xk, Yk = X[keep][:, bins], Y[keep][:, bins]
    Sxx = np.mean(np.abs(Xk) ** 2, axis=0)
    Syy = np.mean(np.abs(Yk) ** 2, axis=0)
    Sxy = np.mean(np.conj(Xk) * Yk, axis=0)

    H1, H2, Hv, gamma2 = transfer_estimators(Sxx, Syy, Sxy)
    n_eff = float(keep.sum())          # non-overlapping: no correlation penalty
    Z = {"h1": H1, "h2": H2, "hv": Hv}[estimator]

    return SpectralResult(
        f=np.asarray(kept_tones), Z=Z, coherence=gamma2,
        sigma_rel=relative_uncertainty(gamma2, n_eff),
        n_windows_total=int(X.shape[0]), n_windows_used=int(keep.sum()),
        n_eff=n_eff, estimator=estimator, method="synchronous",
        Z_h1=H1, Z_h2=H2, rejected_reason=reasons, note=note,
    )


# ---------------------------------------------------------------------------
# Tone discovery (verification / fallback)
# ---------------------------------------------------------------------------

def detect_tones(
    f: np.ndarray, coherence: np.ndarray, threshold: float = 0.95, gap_hz: float = 3.0
) -> np.ndarray:
    """Cluster coherence peaks into tone centres.

    Use this to *verify* that the designed excitation is where it should be, or
    as a fallback when the excitation recipe is unknown - not as the primary
    source of the analysis frequencies, which should be configuration.
    """
    mask = coherence > threshold
    picked = f[mask]
    if picked.size == 0:
        return np.array([])
    centres, start = [], 0
    for i in range(1, len(picked)):
        if picked[i] - picked[i - 1] > gap_hz:
            centres.append(float(np.mean(picked[start:i])))
            start = i
    centres.append(float(np.mean(picked[start:])))
    return np.asarray(centres)
