#!/usr/bin/env python3
"""
eis_local.py
============
Local electrochemical impedance spectroscopy on the Bosch R2-D2 72-segment
plate -- evaluated WITHOUT any potentiostat file.

    python eis_local.py --dat ./measurement_folder \
                        --curr-cal curr.csv --temp-cal temp.csv \
                        --current 45 --out ./results

WHAT MAKES THIS INDEPENDENT OF THE GAMRY
----------------------------------------
The previous pipeline needed the .dta file for three things.  Each is
replaced by something the plate or the data itself provides:

  Gamry gave                     replaced by
  ---------------------------------------------------------------------
  the list of sweep frequencies  blind schedule detection: a log-spaced
                                 demodulation scan finds the ridge, then the
                                 IEEE Std 1057 four-parameter sine fit
                                 recovers the exact frequency, amplitude,
                                 phase and offset of every step, plus a
                                 residual that is a real uncertainty.
  the expected excitation        per-step SNR, harmonic distortion (THD ->
  amplitude, as a QC gate        linearity) and amplitude stationarity across
                                 sub-windows (-> time invariance), then
                                 lin-KK and Z-HIT on the finished spectrum.
  the scale factor H, solved     the ex-situ "Abgleich" calibration of the
  from the current setpoint      plate:  j_s = u_s / (c0 + 0.001*c1*T),
                                 c0,c1 per segment from curr.csv, T from the
                                 four plate sensors via temp.csv.  This is an
                                 absolute, traceable scale that owes nothing
                                 to the cell current.
  the reference Nyquist curve    the aggregate of the local spectra,
                                 Z_cell = A_cell / sum_s (A_s / Z_s),
                                 which is what the cell-level instrument
                                 would measure if it were connected.

The total cell current therefore stops being an input and becomes an
INDEPENDENT CHECK:  sum_s j_s * A_s  must equal it.

WHY THE AREAS MATTER
--------------------
Segment areas span 0.678 .. 8.47 cm^2 (see r2d2_geometry.py).  A single
segment's ASR is area-free, but the current closure, every area-weighted
average and the parallel aggregation above are not.  Areas come from
r2d2_geometry, or from --areas file.csv to override them.

PIPELINE
--------
  1. read FAMOS files, discover cards, segments, UC and temperature channels
  2. plate temperature from temp.csv, interpolated to each segment's centroid
  3. blind detection of the sweep schedule on the UC reference channel
  4. per-step 3-parameter sine fit of U_cell and of every segment voltage
  5. u_s -> j_s with the per-segment calibration (and optional dynamic gain)
  6. Z_s = U_cell / j_s   [ohm*cm^2]
  7. acquisition skew from Z-HIT (a delay is an all-pass; KK cannot see it)
  8. lin-KK and Z-HIT validation per segment
  9. DC current closure, aggregation, maps, CSV/JSON export

Requires: numpy, matplotlib.  r2d2_geometry.py and eis_validation.py must sit
next to this file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

import r2d2_geometry as geom
from eis_validation import estimate_delay, lin_kk, z_hit

T_FALLBACK_C = 58.4          # used only if the temperature channels are unusable


# ===========================================================================
# 1. FAMOS reader
# ===========================================================================


def parse_start_time(text: str) -> datetime | None:
    """Absolute trigger time from the ``|NT`` key: ``|NT,len,d,m,y,h,m,s;``.

    This is the only statement a card makes about WHEN it started, and it is
    what makes "were these five cards recording the same event?" answerable
    before anything is evaluated.  Its resolution is one second and its
    accuracy is the operator pressing start, so it is never precise enough to
    ALIGN with -- that is measured from the shared reference channel.  What it
    is good for is the sanity check the measured alignment cannot make on its
    own: cross-correlation happily returns a confident lag for two cards that
    recorded different runs an hour apart, because it only ever sees sample
    indices.  The header stamps are what catch that.
    """
    match = re.search(
        r"\|NT,\d+,(\d+),(\d+),(\d+),(\d+),(\d+),([\d.]+)\s*;", text)
    if match is None:
        return None
    day, month, year, hour, minute = (int(match.group(i)) for i in range(1, 6))
    seconds = float(match.group(6))
    try:
        return datetime(year, month, day, hour, minute,
                        int(seconds), int(round((seconds % 1) * 1e6)))
    except ValueError:
        return None


#: How far into a file to look for the |CS payload key before giving up.
#: The header carries one name per channel, so a card with many channels has
#: a long one; 1 MiB is far past anything imc writes and still a trivial read
#: over a network share.
_MAX_HEADER_BYTES = 1 << 20

#: The keys the reader needs, and what each one carries.
_HEADER_KEYS = {
    "CD": ("sample interval", r"\|CD,\d+,([\d.eE+-]+)"),
    "CR": ("channel count", r"\|CR,\d+,(\d+)"),
    "CP": ("channel names", r"\|CP,([^;]*);"),
    "CS": ("payload start", r"\|CS,(\d+),(\d+),"),
}


def read_famos_header(path) -> tuple[str, int]:
    """Header text and the byte offset just past ``|CS,n,m,``.

    THE READ GROWS UNTIL IT FINDS THE PAYLOAD KEY
    ---------------------------------------------
    This used to be a flat ``read(8192)``.  Eight kilobytes is enough for the
    cards this was written against and not enough in general: the header
    carries one ``7,32,<name>`` entry per channel plus whatever else the
    exporting tool wrote, so a card with many channels pushes ``|CS`` past
    the fixed window.  The keys are then not found, and the only thing said
    about it is "incomplete FAMOS header" -- which points at the file rather
    than at the reader, and is indistinguishable from the file genuinely
    being some other format.

    Reads no more than it must: it stops at ``|CS``, which is the start of
    the payload, so a 6 GB card still costs one small read.
    """
    size = os.path.getsize(path)
    chunk = 65_536
    with open(path, "rb") as fh:
        while True:
            raw = fh.read(chunk)
            text = raw.decode("latin-1", errors="replace")
            match = re.search(_HEADER_KEYS["CS"][1], text)
            if match is not None:
                return text[: match.end()], match.end()
            if len(raw) < chunk or chunk >= min(size, _MAX_HEADER_BYTES):
                return text, -1
            chunk *= 2
            fh.seek(0)


def _header_diagnosis(path, text: str, scanned: int) -> str:
    """Say which keys were found and which were not, and what to do next.

    "incomplete FAMOS header" names the symptom and nothing else. The two
    causes need opposite responses -- a header longer than the reader looked,
    versus a file that is not FAMOS at all -- and which one it is follows
    directly from WHICH keys were present.
    """
    found, missing = [], []
    for key, (what, pattern) in _HEADER_KEYS.items():
        (found if re.search(pattern, text) else missing).append(f"|{key} ({what})")
    head = open(path, "rb").read(32)
    lines = [
        f"{Path(path).name}: could not read the FAMOS header.",
        f"  scanned:  {scanned} bytes",
        f"  found:    {', '.join(found) if found else 'none of the keys'}",
        f"  missing:  {', '.join(missing)}",
        f"  first 32 bytes: {head.hex(' ')}",
        f"  as text       : {head.decode('latin-1', 'replace')!r}",
    ]
    if found:
        lines.append(
            "  Some keys are present, so this IS a FAMOS file whose header is "
            "laid out differently than the reader expects. Send the two lines "
            "above with the missing key names.")
    else:
        lines.append(
            "  No FAMOS key was found at all, so this is very likely not a "
            "FAMOS file. Establish what it is before writing a reader for it:")
        lines.append(f'      python identify_file.py "{path}"')
    return "\n".join(lines)


@dataclass
class FamosFile:
    """imc FAMOS binary reader.

    Two traps this class handles:

    * The |CS block declares a byte count that, in these recordings, is exactly
      half of what is actually present.  Trusting it silently discards the
      second half of the measurement -- which is where the low-frequency steps
      live.  The sample count is derived from the real file size instead.
    * The header must be decoded as latin-1; channel names contain umlauts.

    ``start_time`` is the ``|NT`` trigger stamp, or None when the dialect does
    not carry one.  Nothing downstream aligns on it -- see
    :func:`parse_start_time` for why -- but bronze compares it against the
    measured lags, which is what turns "these cards disagree" from a silent
    wrong answer into a refusal.
    """

    path: Path
    fs: float = field(init=False)
    names: list[str] = field(init=False)
    n_ch: int = field(init=False)
    n_samples: int = field(init=False)
    offset: int = field(init=False)
    start_time: "datetime | None" = field(init=False)

    def __post_init__(self):
        self.path = Path(self.path)
        raw, payload_at = read_famos_header(self.path)
        self.start_time = parse_start_time(raw)

        m_cd = re.search(_HEADER_KEYS["CD"][1], raw)
        m_cr = re.search(_HEADER_KEYS["CR"][1], raw)
        m_cp = re.search(_HEADER_KEYS["CP"][1], raw)
        m_cs = re.search(_HEADER_KEYS["CS"][1], raw)
        if payload_at < 0 or not all((m_cd, m_cr, m_cp, m_cs)):
            raise ValueError(_header_diagnosis(self.path, raw, len(raw)))

        self.fs = 1.0 / float(m_cd.group(1))
        self.n_ch = int(m_cr.group(1))
        self.names = [n.strip() for n in re.findall(r"7,32,([^,;]*)", m_cp.group(1))]
        self.offset = m_cs.end()

        declared = int(m_cs.group(2))
        available = self.path.stat().st_size - self.offset
        self.n_samples = available // (4 * self.n_ch)

        if abs(available - declared) > 64:
            print(f"  [{self.path.name}] |CS declares {declared} bytes, "
                  f"{available} present. Using the full file: "
                  f"{self.n_samples} samples = {self.n_samples/self.fs:.1f} s")
        if len(self.names) != self.n_ch:
            print(f"  [{self.path.name}] WARNING: {len(self.names)} names for "
                  f"{self.n_ch} channels")

    def channel(self, name: str, start: int = 0,
                stop: int | None = None) -> np.ndarray:
        """One channel, optionally only the samples in [start, stop).

        THE RANGE IS APPLIED TO THE MEMMAP, NOT TO THE RESULT.  Slicing
        afterwards -- `fam.channel(c)[:n]` -- reads the whole file first and
        then throws most of it away, because materialising the column faults
        in every page that holds it.  The samples are interleaved across
        channels, so that is the entire recording.

        It costs nothing noticeable on local disk, which is why it survived,
        and it is the difference between seconds and many minutes over SMB:
        a caller that wants twenty seconds of a five-minute recording to
        decide what the excitation is should move twenty seconds of it
        across the network, not five minutes of it per card.
        """
        if name not in self.names:
            raise KeyError(f"{name!r} not in {self.path.name}: {self.names}")
        stop = self.n_samples if stop is None else min(int(stop), self.n_samples)
        start = max(0, int(start))
        if stop <= start:
            return np.empty(0, dtype=np.float64)
        mm = np.memmap(self.path, dtype="<f4", mode="r",
                       offset=self.offset, shape=(self.n_samples, self.n_ch))
        return np.asarray(mm[start:stop, self.names.index(name)],
                          dtype=np.float64)

    def position(self, name: str) -> int:
        """Channel index in the header = ADC acquisition order."""
        return self.names.index(name)

    @property
    def segment_names(self) -> list[str]:
        return [n for n in self.names if re.fullmatch(r"\d+", n)]

    @property
    def uc_names(self) -> list[str]:
        return [n for n in self.names if n.upper().startswith("UC")]

    @property
    def temp_names(self) -> list[str]:
        return [n for n in self.names if n.lower().startswith("temp")]


# ===========================================================================
# 2. Plate calibration  (the "Abgleich")
# ===========================================================================


@dataclass
class PlateCalibration:
    """j_s [A/cm^2] = u_s [V] / (c0 + 0.001 * c1 * T[degC])   per segment
       T [degC]     = (u_temp [V] - c0) / c1                  per sensor

    curr.csv: 72 lines "c0;c1", line n belongs to segment n.
    temp.csv:  4 lines "c0;c1", line k belongs to temp<k>.

    The segment area is already inside c0 and c1: every pad carries its own
    via, so a 25-pad segment has 25 vias in parallel and R_via * A_seg is
    roughly invariant.  That is why c0 scatters by only ~15 % across the
    plate although the areas span a factor of 12.5 -- and it is why the
    calibration returns a current DENSITY, not a current.
    """

    seg_c0: dict[str, float]
    seg_c1: dict[str, float]
    temp_c0: dict[str, float]
    temp_c1: dict[str, float]

    @classmethod
    def load(cls, curr_path=None, temp_path=None) -> "PlateCalibration":
        seg_c0, seg_c1, t_c0, t_c1 = {}, {}, {}, {}
        if curr_path:
            rows = _read_pairs(curr_path)
            for i, (a, b) in enumerate(rows, start=1):
                seg_c0[str(i)], seg_c1[str(i)] = a, b
            if len(rows) != geom.N_SEGMENTS:
                print(f"  WARNING: {curr_path} has {len(rows)} rows, expected "
                      f"{geom.N_SEGMENTS}")
        if temp_path:
            for i, (a, b) in enumerate(_read_pairs(temp_path), start=1):
                t_c0[f"temp{i}"], t_c1[f"temp{i}"] = a, b
        return cls(seg_c0, seg_c1, t_c0, t_c1)

    @property
    def has_current_cal(self) -> bool:
        return bool(self.seg_c0)

    def K(self, seg: str, T_C: float) -> float:
        """Transfer coefficient in V/(A/cm^2) at temperature T."""
        return self.seg_c0[seg] + 1e-3 * self.seg_c1[seg] * T_C

    def temperature(self, sensor: str, u_volt: float) -> float:
        return (u_volt - self.temp_c0[sensor]) / self.temp_c1[sensor]


def _read_pairs(path) -> list[tuple[float, float]]:
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
    """Optional dynamic response of the measuring chain, from the ex-situ
    frequency-response characterisation.

    CSV columns: segment,freq_hz,gain_real,gain_imag   (gain normalised to 1
    at low frequency).  Z is divided by G so that the roll-off of the
    amplifier -- about -3 dB near 20 kHz on this plate -- does not end up in
    the electrochemistry.  Segment "0" or "all" may be used for a single
    shared response.
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


# ===========================================================================
# 3. Sine fitting  (IEEE Std 1057)
# ===========================================================================


def fit3(y: np.ndarray, fs: float, f: float) -> tuple[complex, float, float]:
    """Three-parameter fit  y ~ a*cos(wt) + b*sin(wt) + c  at known f.

    Returns (phasor, residual_rms, snr_db) with the phasor under the
    exp(+j w t) convention, i.e. y = Re{A e^{jwt}} + c and A = a - j b.
    This is the least-squares optimum; a lock-in is the same estimator only
    when the window holds a whole number of cycles, which a stepped sweep
    never guarantees.
    """
    n = len(y)
    if n < 8:
        return complex("nan"), np.nan, np.nan
    t = np.arange(n) / fs
    w = 2 * np.pi * f
    D = np.column_stack([np.cos(w * t), np.sin(w * t), np.ones(n)])
    p, *_ = np.linalg.lstsq(D, y, rcond=None)
    resid = y - D @ p
    A = complex(p[0], -p[1])
    r_rms = float(np.sqrt(np.mean(resid ** 2)))
    sig = abs(A) / np.sqrt(2.0)
    snr = 20 * np.log10(sig / r_rms) if r_rms > 0 else np.inf
    return A, r_rms, float(snr)


def fit4(y: np.ndarray, fs: float, f0: float, n_iter: int = 12
         ) -> tuple[float, complex, float, float]:
    """Four-parameter fit: also solves for the frequency (IEEE Std 1057).

    Each iteration linearises the model around the current omega by adding the
    column  t * (-a sin(wt) + b cos(wt)), solves for the increment, and
    updates.  Converges in a few steps when the start value is within a few
    percent, which the demodulation scan guarantees.

    Returns (f_hat, phasor, residual_rms, snr_db).
    """
    n = len(y)
    if n < 16:
        return f0, complex("nan"), np.nan, np.nan
    t = np.arange(n) / fs
    w = 2 * np.pi * f0
    a, b = 1.0, 0.0
    for _ in range(n_iter):
        c_, s_ = np.cos(w * t), np.sin(w * t)
        D = np.column_stack([c_, s_, np.ones(n), t * (-a * s_ + b * c_)])
        p, *_ = np.linalg.lstsq(D, y, rcond=None)
        a, b = p[0], p[1]
        dw = p[3]
        # a step larger than a few percent means the start value was wrong;
        # clamp instead of letting the iteration run away to another peak
        dw = float(np.clip(dw, -0.05 * w, 0.05 * w))
        w_new = w + dw
        if w_new <= 0:
            break
        conv = abs(dw) / w < 1e-9
        w = w_new
        if conv:
            break
    f_hat = w / (2 * np.pi)
    A, r_rms, snr = fit3(y, fs, f_hat)
    return float(f_hat), A, r_rms, snr


def dominant_frequency(y: np.ndarray, fs: float, f_lo: float, f_hi: float,
                       oversample: int = 16) -> float:
    """Frequency of the largest spectral peak of ``y`` inside [f_lo, f_hi].

    WHY THE SINE FIT NEEDS THIS IN FRONT OF IT
    ------------------------------------------
    :func:`fit4` is a Gauss-Newton iteration on the frequency, so it finds
    the nearest optimum, not the best one.  Its basin is about the width of
    the DFT main lobe -- roughly +/- 1/(2*N) in relative frequency for a
    window holding N cycles, so +/- 2.6 % on a twenty-cycle dwell.  The
    detection grid is twelve points per decade, which is 21 % apart, so a
    true tone can sit 10 % from every grid point.  Outside the basin the
    iteration does not fail: it converges on a sidelobe near where it
    started and reports a confident frequency that is wrong by several
    percent, with an SNR of about -14 dB.  Measured on a 100 kHz synthetic
    sweep, that is what happened to every step above 1 kHz that the scan had
    already located correctly -- the window was right, sitting on the right
    dwell with 19 cycles in it, and the fit still came back with noise.

    A periodogram peak has no basin.  One zero-padded FFT over the window
    puts the seed inside the main lobe by construction, and fit4 then does
    what it is good at: refining a start value that is already close.

    The Hann window matters here: whatever the dwell finder leaves of the
    neighbouring steps sits 20-30 % away in frequency, and a rectangular
    window's sidelobes at that distance are comparable to a weak tone's main
    lobe.
    """
    y = np.asarray(y, float)
    n = len(y)
    if n < 16:
        return float("nan")
    y = y - y.mean()
    m = 1 << int(np.ceil(np.log2(max(n * oversample, 64))))
    spec = np.abs(np.fft.rfft(y * np.hanning(n), m))
    freqs = np.fft.rfftfreq(m, 1.0 / fs)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    if not band.any():
        return float("nan")
    return float(freqs[band][int(np.argmax(spec[band]))])


def fit_multitone(y: np.ndarray, fs: float, freqs, drift: bool = True
                  ) -> tuple[np.ndarray, float, np.ndarray, dict]:
    """Fit every tone that is present AT THE SAME TIME, in one least squares.

    WHY A STEPPED MULTISINE CANNOT USE THE SINGLE-TONE FIT
    -----------------------------------------------------
    :func:`fit3` fits one sinusoid and calls everything left over "residual".
    That is the right reading when one frequency is applied at a time.  When
    a dwell carries a GROUP of tones together, the other tones of the group
    are in that residual, and the SNR it reports is not a measurement of
    noise -- it is a measurement of how many tones are in the group:

        SNR_reported  ~  -10 * log10(N - 1)   dB

    for N equal-amplitude tones, whatever the actual noise.  Eight tones
    gives -8.5 dB, twenty gives -12.8, forty gives -15.9.  Measured on a
    synthetic eight-tone dwell whose true noise floor is -40 dB, fit3
    reports -8.5 dB on every tone.

    Two consequences, and the second is the expensive one:

    * A quality gate set against that number is not gating on quality.
      Lowering it to -30 dB does not "let the weak high-frequency tones in";
      it lets everything in, because the number never described the tones in
      the first place.
    * The SNR is used downstream as a WEIGHT.  Every tone in a group gets
      the same fictitious value, so the genuinely good points and the
      genuinely bad ones are weighted identically in the KK fit and in the
      aggregate.

    The amplitudes themselves come out right -- the tones are near-orthogonal
    over a long window -- which is why this hides so well: the spectrum looks
    reasonable and only the uncertainties are wrong.

    Fitting the group jointly puts the other tones in the model where they
    belong, so what is left really is noise.

    Returns ``(phasors, resid_rms, snr_db, info)`` with phasors under the
    same exp(+j w t) convention as :func:`fit3`.
    """
    y = np.asarray(y, float)
    freqs = np.asarray(list(freqs), float)
    n = len(y)
    n_par = 2 * len(freqs) + (2 if drift else 1)
    if n < max(16, 2 * n_par) or len(freqs) == 0:
        nan = np.full(len(freqs), np.nan)
        return (nan.astype(complex), np.nan, nan,
                {"ok": False, "note": "window too short for the group"})

    t = np.arange(n) / fs
    cols = []
    for f in freqs:
        w = 2 * np.pi * f
        cols += [np.cos(w * t), np.sin(w * t)]
    cols.append(np.ones(n))
    if drift:
        # A slow baseline wander inside a dwell is otherwise absorbed by the
        # lowest tone of the group, which is where it does the most harm.
        cols.append(t - t.mean())
    D = np.column_stack(cols)

    # RESOLVABILITY.  Two tones closer than about 1/T are not separable over
    # this window: the fit will still return numbers, and they will be a
    # near-arbitrary split of one peak between two columns. Say so rather
    # than reporting the split as two measurements.
    T = n / fs
    order = np.sort(freqs)
    min_gap = float(np.min(np.diff(order))) if len(order) > 1 else np.inf
    resolvable = bool(min_gap * T >= 1.0)

    p, *_ = np.linalg.lstsq(D, y, rcond=None)
    resid = y - D @ p
    ssr = float(np.dot(resid, resid))
    r_rms = float(np.sqrt(ssr / n))
    # Unbiased noise variance: the fit has already spent n_par degrees of
    # freedom, and with a large group that correction is not cosmetic.
    dof = max(n - n_par, 1)
    sigma2 = ssr / dof

    phasors = np.array([complex(p[2 * i], -p[2 * i + 1])
                        for i in range(len(freqs))], dtype=complex)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Power SNR of each tone against the noise the joint fit actually
        # left behind, rather than against its neighbours.
        snr = 10.0 * np.log10((np.abs(phasors) ** 2 / 2.0) / sigma2)
    cond = float(np.linalg.cond(D))
    return phasors, r_rms, snr, {
        "ok": True, "resolvable": resolvable, "min_gap_hz": min_gap,
        "window_s": T, "cond": cond, "n_par": n_par,
        "sigma": float(np.sqrt(sigma2)),
    }


def harmonic_distortion(y: np.ndarray, fs: float, f: float) -> float:
    """THD from the 2nd and 3rd harmonic; the linearity check.

    A galvanostatic EIS excitation must stay in the linear regime.  If the
    amplitude is too large for the local operating point the response clips
    into harmonics, and the fundamental-only estimate silently becomes a
    describing function instead of an impedance.
    """
    n = len(y)
    if n < 32 or 3 * f > 0.45 * fs:
        return np.nan
    t = np.arange(n) / fs
    cols = [np.ones(n)]
    for k in (1, 2, 3):
        cols += [np.cos(2 * np.pi * k * f * t), np.sin(2 * np.pi * k * f * t)]
    D = np.column_stack(cols)
    p, *_ = np.linalg.lstsq(D, y, rcond=None)
    a1 = np.hypot(p[1], p[2])
    a2 = np.hypot(p[3], p[4])
    a3 = np.hypot(p[5], p[6])
    return float(np.hypot(a2, a3) / a1) if a1 > 0 else np.nan


# ===========================================================================
# 4. Blind detection of the sweep schedule
# ===========================================================================


def _moving_mean(x: np.ndarray, win: int) -> np.ndarray:
    c = np.concatenate([[0.0 + 0j], np.cumsum(x)])
    return (c[win:] - c[:-win]) / win


def demod_envelope(x: np.ndarray, fs: float, f: float, n_cyc: float = 5.0,
                   min_win_s: float = 0.0, max_win_s: float = 6.0,
                   min_win_samples: int = 16):
    """|complex amplitude at f| as a function of window start time.

    The window is rounded to a whole number of cycles so that DC and the
    neighbouring steps fall into the nulls of the rectangular window.

    THE FLOOR IS IN SAMPLES, NOT IN SECONDS
    ---------------------------------------
    `min_win_s` used to be 0.04 s, and that one number is why this detector
    stopped finding tones somewhere around a kilohertz.

    A sweep generator dwells for a fixed number of CYCLES, so the dwell is
    n/f seconds and shrinks as the frequency rises: twenty cycles is 200 ms
    at 100 Hz and 5 ms at 4 kHz.  A 40 ms floor on the analysis window is
    therefore harmless at the bottom of the band and, above about 500 Hz,
    makes the window LONGER THAN THE DWELL IT IS MEASURING.  Past that point
    the window averages the tone together with the silence and the
    neighbouring steps on either side of it, so the demodulated amplitude
    falls off as dwell/window -- about -9 dB at 1 kHz and -18 dB at 4 kHz on
    a twenty-cycle sweep -- and the `peak >= 4*floor` test that locates the
    dwell stops firing.

    Nothing about that failure looks like a windowing problem from the
    outside.  It looks like weak high-frequency excitation, which is why the
    natural response is to turn the SNR gate down; and turning it down does
    not help, because the dilution scales the peak and the background
    together.  Measured on a 16-step synthetic sweep at 100 kHz with the
    gate already at -30 dB: 5 of 16 steps missed, every one of them above
    900 Hz, every one of them a frequency where the floor exceeded the dwell.

    What the floor was protecting against is a window of too few SAMPLES to
    average, and that is what it now says: `min_win_samples`.  The window
    stays n_cyc periods long at every frequency, which is what the docstring
    above always claimed it was.
    """
    n = len(x)
    spc = fs / f                                   # samples per cycle
    want_s = n_cyc / f
    if min_win_s > 0:
        want_s = max(want_s, min_win_s)
    win = int(round(min(want_s, max_win_s) * fs))
    win = int(max(min_win_samples, min(win, n // 3)))
    # snap to a whole number of cycles so that DC and the neighbouring steps
    # land in the nulls of the rectangular window.  Rounding samples-per-cycle
    # to an integer first would ruin this near the top of the band, where spc
    # is only a few samples.
    cyc = max(1, int(round(win / spc)))
    win = int(max(8, min(int(round(cyc * spc)), n // 3)))
    t = np.arange(n) / fs
    y = (x - x.mean()) * np.exp(-2j * np.pi * f * t)
    env = 2.0 * np.abs(_moving_mean(y, win))
    return env, win


def dwell_window(x: np.ndarray, fs: float, f: float, centre: int,
                 frac: float = 0.7, trim: float = 0.1) -> tuple[int, int]:
    """Contiguous interval around `centre` where the response at f is above
    `frac` of its peak: the interval in which that frequency was applied.

    Growing the window this way instead of assuming a dwell time is what
    keeps the low-frequency steps -- where the Gamry dwells for tens of
    seconds -- from being evaluated over a fraction of their data.
    """
    # Six cycles, and no floor in seconds -- see demod_envelope. A 20 ms
    # floor here is the same bug one step smaller: above ~300 Hz on a
    # twenty-cycle sweep it makes the interval-finding window wider than the
    # interval it is trying to find, so the dwell comes back stretched into
    # its neighbours and polish_window has to claw it back.
    env, win = demod_envelope(x, fs, f, n_cyc=6.0, max_win_s=6.0)
    if len(env) < 4:
        return 0, len(x)
    k = int(np.clip(centre - win // 2, 0, len(env) - 1))
    lo_k = max(0, k - int(2 * fs / f))
    hi_k = min(len(env) - 1, k + int(2 * fs / f))
    k = lo_k + int(np.argmax(env[lo_k:hi_k + 1]))
    thr = frac * env[k]
    i = k
    while i > 0 and env[i - 1] >= thr:
        i -= 1
    j = k
    while j < len(env) - 1 and env[j + 1] >= thr:
        j += 1
    start, stop = i, min(len(x), j + win)
    n = stop - start
    cut = int(trim * n)
    start, stop = start + cut, stop - cut
    # never shorter than three periods
    if stop - start < 3 * fs / f:
        stop = min(len(x), start + int(3 * fs / f) + 1)
    return int(start), int(stop)


def polish_window(x: np.ndarray, fs: float, f: float, a: int, b: int,
                  iters: int = 5) -> tuple[int, int]:
    """Trim the window until the fit residual stops improving.

    Neighbouring steps of the sweep are only ~30 % apart in frequency, so the
    skirt of one bleeds into the envelope of the next and the dwell interval
    comes out slightly too generous.  A leftover slice of the adjacent step is
    full-amplitude signal sitting in the residual, which costs far more SNR
    than the samples gained.  Shrinking while the SNR rises removes it.
    """
    best_snr = fit3(x[a:b], fs, f)[2]
    for _ in range(iters):
        n = b - a
        cut = max(int(0.06 * n), int(fs / f / 4), 4)
        improved = False
        for aa, bb in ((a + cut, b), (a, b - cut), (a + cut, b - cut)):
            if bb - aa < max(3 * fs / f, 32):
                continue
            snr = fit3(x[aa:bb], fs, f)[2]
            if np.isfinite(snr) and snr > best_snr + 0.5:
                a, b, best_snr, improved = aa, bb, snr, True
        if not improved:
            break
    return int(a), int(b)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs of a boolean array as (start, stop) pairs."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(),
                    np.flatnonzero(d == -1).tolist()))


def fill_gaps(ref: np.ndarray, fs: float, steps: list[Step],
              min_snr_db: float = 8.0, max_rounds: int = 3) -> list[Step]:
    """Recover steps that the log grid stepped over.

    The scan grid and the sweep schedule are both geometric but with different
    ratios, so a grid point can sit up to half a step away from a real one and
    the fit then lands in the wrong basin.  Rather than making the grid ever
    finer, look at what is left: any stretch of the record not claimed by an
    accepted step is either idle or an undetected dwell.  Inside such a
    stretch the excitation is a single tone, so its FFT peak gives the
    frequency directly -- no grid, no prior list.
    """
    n = len(ref)
    for _ in range(max_rounds):
        good = [s for s in steps if s.valid(min_snr_db)]
        mask = np.zeros(n, dtype=bool)
        for s in good:
            mask[s.start:s.stop] = True
        added = 0
        for g0, g1 in _runs(~mask):
            dur = (g1 - g0) / fs
            if dur < 0.05 or g1 - g0 < 64:
                continue
            x = ref[g0:g1].astype(float)
            x = x - x.mean()
            X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
            fr = np.fft.rfftfreq(len(x), 1.0 / fs)
            sel = (fr > max(3.0 / dur, 0.02)) & (fr < 0.40 * fs)
            if not sel.any():
                continue
            k = int(np.argmax(np.where(sel, X, 0.0)))
            if X[k] < 4.0 * np.median(X[sel]):
                continue                       # idle stretch, no tone
            f1, _A, _r, _snr = fit4(x, fs, float(fr[k]))
            if not np.isfinite(f1) or f1 <= 0 or f1 > 0.40 * fs:
                continue
            a, b = dwell_window(x, fs, f1, (g1 - g0) // 2)
            if b - a < 32:
                continue
            a, b = polish_window(x, fs, f1, a, b)
            f2, A2, _r2, snr2 = fit4(x[a:b], fs, f1)
            if not (0.95 * f1 <= f2 <= 1.05 * f1):
                f2 = f1
            st = Step(freq=float(f2), start=g0 + a, stop=g0 + b,
                      amp=float(abs(A2)), snr_db=float(snr2),
                      thd=harmonic_distortion(x[a:b], fs, f2),
                      stationarity=_stationarity(ref, fs, f2, g0 + a, g0 + b))
            if st.valid(min_snr_db):
                steps.append(st)
                added += 1
        if not added:
            break
    return steps


def _dedupe(steps: list[Step], rel_tol: float = 0.02,
            overlap_tol: float = 0.5) -> list[Step]:
    """One physical step must yield one entry.

    Two criteria, because either alone leaves duplicates: nearly equal
    frequency, and -- for a partial window that landed on the same dwell with a
    slightly wrong frequency -- substantial overlap in time.  The higher SNR
    wins, which is always the more complete window.
    """
    out: list[Step] = []
    for s in sorted(steps, key=lambda q: q.freq):
        if out and abs(s.freq / out[-1].freq - 1.0) < rel_tol:
            if s.snr_db > out[-1].snr_db:
                out[-1] = s
        else:
            out.append(s)

    keep: list[Step] = []
    for s in sorted(out, key=lambda q: -q.snr_db):
        clash = False
        for t in keep:
            ov = min(s.stop, t.stop) - max(s.start, t.start)
            if ov > overlap_tol * min(s.stop - s.start, t.stop - t.start):
                clash = True
                break
        if not clash:
            keep.append(s)
    return sorted(keep, key=lambda q: q.freq)


@dataclass
class Step:
    freq: float
    start: int
    stop: int
    amp: float           # |U_cell| phasor magnitude, V
    snr_db: float
    thd: float
    stationarity: float  # spread of the phasor over three sub-windows

    def valid(self, min_snr=8.0, max_thd=0.10, max_drift=0.15) -> bool:
        return (np.isfinite(self.snr_db) and self.snr_db >= min_snr
                and (not np.isfinite(self.thd) or self.thd <= max_thd)
                and (not np.isfinite(self.stationarity)
                     or self.stationarity <= max_drift))


def _stationarity(x, fs, f, start, stop) -> float:
    n = (stop - start) // 3
    if n < int(3 * fs / f) or n < 16:
        return np.nan
    # each sub-fit uses its own t=0, so the phasors must be rotated back to a
    # common origin before they can be compared -- otherwise a perfectly
    # stationary sine looks like 100 % drift
    ph = np.array([fit3(x[start + i * n:start + (i + 1) * n], fs, f)[0]
                   * np.exp(-2j * np.pi * f * (i * n) / fs) for i in range(3)])
    if not np.all(np.isfinite(ph)):
        return np.nan
    m = np.mean(ph)
    return float(np.max(np.abs(ph - m)) / abs(m)) if abs(m) > 0 else np.nan


def detect_schedule(ref: np.ndarray, fs: float, ppd: int = 12,
                    f_lo: float | None = None, f_hi: float | None = None,
                    min_snr_db: float = 8.0, verbose: bool = True
                    ) -> list[Step]:
    """Find every stepped-sine dwell in the record without being told the
    frequency list.

    Stage 1  scan a log grid of trial frequencies; for each, the largest
             demodulated amplitude over the record locates a candidate dwell.
    Stage 2  a four-parameter sine fit on that window recovers the true
             frequency -- the grid only has to land in its basin.
    Stage 3  grow the window to the actual dwell, refit, score.
    Stage 4  collapse grid points that converged onto the same frequency.

    Frequencies above 0.4*fs are never tried: the plate samples at 10 kHz
    while a sweep may go to 30 kHz, and an aliased 30046.88 Hz point folds to
    46.88 Hz, where it would masquerade as a genuine low-frequency step.
    """
    n = len(ref)
    T = n / fs
    f_hi = float(f_hi or 0.35 * fs)   # 3 samples/cycle is already marginal
    f_lo = float(f_lo or max(4.0 / T, 0.02))
    if f_hi <= f_lo:
        return []
    n_grid = int(np.ceil(ppd * np.log10(f_hi / f_lo))) + 1
    grid = np.logspace(np.log10(f_lo), np.log10(f_hi), max(n_grid, 4))
    #: One grid step, as a ratio. The spectral seed is bracketed by it: a
    #: candidate found by this grid point belongs to it or to a neighbour,
    #: and the neighbours have their own turn.
    grid_ratio = float(grid[1] / grid[0]) if len(grid) > 1 else 1.3

    raw: list[Step] = []
    for f in grid:
        env, win = demod_envelope(ref, fs, f)
        if len(env) < 4:
            continue
        peak, floor = float(np.max(env)), float(np.median(env))
        if not np.isfinite(peak) or peak < 4.0 * floor:
            continue                       # nothing stands out at this frequency
        k = int(np.argmax(env)) + win // 2   # centre of the best coarse window

        # The window MUST come from the data, not from a rule of thumb: a
        # window sized as n_cyc/f can straddle two steps of the sweep, and the
        # fit then sees a mixture and reports a residual that has nothing to do
        # with the noise.  Locate the dwell first, fit second.
        a, b = dwell_window(ref, fs, f, k)
        if b - a < 32:
            continue
        # SEED THE SINE FIT SPECTRALLY, NOT FROM THE GRID POINT.  The grid is
        # 21 % coarse and fit4's basin is a couple of percent; see
        # dominant_frequency for what that cost above 1 kHz.  The bracket is
        # one grid step either side, which is the range this candidate could
        # legitimately belong to.
        f_seed = dominant_frequency(ref[a:b], fs, f / grid_ratio, f * grid_ratio)
        if not np.isfinite(f_seed):
            f_seed = f
        f1, A1, _r1, snr1 = fit4(ref[a:b], fs, f_seed)
        if not np.isfinite(f1) or not (0.80 * f <= f1 <= 1.25 * f):
            continue                       # ran away to a different step
        a, b = dwell_window(ref, fs, f1, (a + b) // 2)
        if b - a < 32:
            continue
        a, b = polish_window(ref, fs, f1, a, b)
        f2, A2, _r2, snr2 = fit4(ref[a:b], fs, f1)
        if not (0.97 * f1 <= f2 <= 1.03 * f1):
            f2, A2, snr2 = f1, A1, snr1
        raw.append(Step(freq=float(f2), start=a, stop=b, amp=float(abs(A2)),
                        snr_db=float(snr2),
                        thd=harmonic_distortion(ref[a:b], fs, f2),
                        stationarity=_stationarity(ref, fs, f2, a, b)))

    # collapse duplicates: several grid points converge on the same step
    steps = _dedupe(raw)
    # second pass: anything the grid stepped over shows up as an unclaimed
    # stretch of the record
    steps = _dedupe(fill_gaps(ref, fs, steps, min_snr_db))
    steps = [s for s in steps if f_lo * 0.8 <= s.freq <= f_hi]

    if verbose:
        good = [s for s in steps if s.valid(min_snr_db)]
        print(f"  schedule: {len(steps)} distinct steps found "
              f"({steps[0].freq:.3f} .. {steps[-1].freq:.1f} Hz), "
              f"{len(good)} pass QC" if steps else "  schedule: nothing found")
    return steps


# ===========================================================================
# 4b. Stepped multisine
# ===========================================================================
#
# A stepped multisine applies a GROUP of frequencies at once and steps from
# group to group.  Neither of the two estimators usually reached for suits
# it, and both fail quietly:
#
#   Welch over the whole record   averages every group together.  Each tone
#                                 is present for its own slice of the record,
#                                 so its estimate is diluted by the fraction
#                                 of time it was off, and the coherence that
#                                 is supposed to gate the result is diluted
#                                 with it.
#
#   stepped-sine dwell detection  assumes one frequency per dwell.  Its
#                                 window finder maximises a SINGLE-tone fit,
#                                 and inside a multisine that never improves,
#                                 so the window collapses to its floor of a
#                                 few periods.  Measured on a 15-tone
#                                 synthetic in three groups: 4 of 15 tones
#                                 found, with windows a millisecond long.
#
# What the excitation actually is decides the estimator, so this module
# derives it from the record: segment the record in TIME first, without
# reference to any frequency, then ask each segment which tones it contains,
# then fit those tones together.


def detect_dwells(x: np.ndarray, fs: float, frame_s: float = 0.02,
                  change_tol: float = 0.25, silence_factor: float = 0.15,
                  min_dwell_s: float = 0.01, trim_frac: float = 0.08
                  ) -> list[tuple[int, int]]:
    """Segment the record into intervals of constant excitation content.

    THE SEGMENTATION IS SPECTRAL, NOT ENERGETIC
    -------------------------------------------
    The obvious way to find the steps is to look for the silence between
    them, and it does not work.  Consecutive steps of a sweep frequently
    abut with no gap at all, and even when there is one it can be shorter
    than the analysis frame.  Worse, a silence-first design has to estimate
    "what does quiet look like" from the record, and a record that is mostly
    excitation has no quiet to measure: a low percentile of the frame RMS
    then lands inside the signal, every frame falls below the threshold, and
    the whole record comes back as a single dwell -- which is the one answer
    that looks plausible and loses every step boundary.

    So the primary signal is a CHANGE OF CONTENT.  Normalised frame spectra
    are direction vectors; two frames of the same group are nearly parallel
    and a step change rotates them.  This works whether or not there is a
    gap, and it uses no assumption about how much of the record is silent.
    Normalising also means a change of LEVEL alone -- the amplitude taper a
    designed multisine usually has -- is not mistaken for a change of
    content.

    Silence is then just a segment like any other, dropped afterwards by
    comparing its level against the median of the segments.  That comparison
    is well posed however little silence there is.

    Trimming the ends of each interval matters more than it looks.  The
    first and last frames of a dwell straddle the transition and hold both
    groups; left in, they put a full-amplitude tone from the neighbour into
    the residual of the joint fit, which is exactly the noise estimate the
    whole approach exists to get right.
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
        seg = x[a:a + nper]
        seg = seg - seg.mean()
        rms[i] = np.sqrt(np.mean(seg ** 2))
        spectra[i] = np.abs(np.fft.rfft(seg * win))

    norm = np.linalg.norm(spectra, axis=1)
    unit = spectra / np.where(norm > 0, norm, 1.0)[:, None]

    # BEFORE-AND-AFTER, NOT FRAME-TO-FRAME.  Comparing each frame with its
    # predecessor makes the statistic as noisy as one frame, and a threshold
    # low enough to catch a real step then also fires on the jitter inside
    # one: a three-group record came back as twelve dwells. Averaging W
    # frames either side of the candidate boundary is the standard
    # change-point form and suppresses that jitter by sqrt(W) while leaving
    # a genuine step change untouched -- a step is sustained, which is
    # exactly what the average is sensitive to and the jitter is not.
    W = max(2, int(round(0.05 / (step / fs))) if step > 0 else 2)
    W = min(W, max(2, len(starts) // 8))
    dist = np.zeros(len(starts))
    for i in range(W, len(starts) - W):
        before = unit[i - W:i].mean(axis=0)
        after = unit[i:i + W].mean(axis=0)
        nb, na = np.linalg.norm(before), np.linalg.norm(after)
        if nb > 0 and na > 0:
            dist[i] = 1.0 - float(np.dot(before, after) / (nb * na))

    # Local maxima of the rotation, not every frame above the threshold: a
    # transition spans several overlapping frames and would otherwise be
    # reported as several boundaries a frame apart.
    cuts = [0]
    for i in range(1, len(dist) - 1):
        if dist[i] >= change_tol and dist[i] >= dist[i - 1] \
                and dist[i] >= dist[i + 1] and i - cuts[-1] >= W:
            cuts.append(i)
    cuts.append(len(starts))

    spans = []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        s0 = int(starts[lo])
        s1 = int(min(n, starts[hi - 1] + nper))
        if s1 - s0 >= max(min_dwell_s * fs, 2 * nper):
            spans.append((s0, s1, float(np.median(rms[lo:hi]))))
    if not spans:
        return [(0, n)]

    loud = float(np.median([lvl for _s0, _s1, lvl in spans]))
    out = []
    for s0, s1, lvl in spans:
        if lvl < silence_factor * loud:
            continue                       # a gap between steps, not a dwell
        cut = int(trim_frac * (s1 - s0))
        out.append((s0 + cut, s1 - cut))
    return out or [(0, n)]


def _local_floor(mag: np.ndarray, n_bands: int = 160) -> np.ndarray:
    """A local noise floor: band medians, interpolated back over the band.

    A single global median is the wrong reference when the spectrum spans
    decades of level.  On a stepped multisine the quiet high-frequency end
    drags the global median far below the skirts of the strong low-frequency
    tones, so those skirts clear any fixed threshold and are reported as
    tones -- 81 spurious ones out of 96 on a 15-tone synthetic, all of them
    sidebands hugging a real peak.

    THE BANDS ARE UNIFORM IN BIN INDEX, NOT LOGARITHMIC.  A log grid sounds
    right for a spectrum read on a log axis, and it is wrong here: its bands
    at the bottom of the range are only a few hertz wide, so a band can
    contain one tone and nothing else.  Its median is then the tone, the
    threshold sits 12 dB above the tone, and the tone is not detected --
    which is precisely what happened to the 10 Hz and 20 Hz members of a
    low-frequency group.  A noise floor has to be estimated from a
    neighbourhood that is mostly noise, and uniform bands guarantee every
    band holds many resolution cells.

    Band medians rather than a sliding window: the zero-padded spectrum runs
    to hundreds of thousands of bins, and a sliding view over it at the width
    a noise floor needs is tens of gigabytes.  A floor is smooth by
    construction, so it needs a value per neighbourhood, not per bin.
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


def _false_alarm_db(n_cells: int, p_total: float = 0.01) -> float:
    """Threshold over the noise median that expects `p_total` false peaks.

    The magnitude of a white-noise spectrum is Rayleigh, so a fixed "12 dB
    over the floor" is not a fixed false-alarm rate -- it is a rate per bin,
    and a long window has a lot of bins.  A two-second dwell at 100 kHz
    holds ~10^5 independent cells, at which 12 dB expects one or two noise
    peaks to be promoted to tones, and that is exactly what showed up: a
    lone spurious 42.7 kHz "tone" in a dwell whose real content stopped at
    80 Hz.  Scaling the threshold with the cell count keeps the expected
    number of those near zero however long the window is.
    """
    n_cells = max(8, int(n_cells))
    p = min(0.5, p_total / n_cells)
    sigma = float(np.sqrt(-2.0 * np.log(p)))       # Rayleigh tail
    median = float(np.sqrt(2.0 * np.log(2.0)))     # Rayleigh median / sigma
    return float(20.0 * np.log10(sigma / median))


def tones_in_window(y: np.ndarray, fs: float, f_lo: float, f_hi: float,
                    peak_db: float = 12.0, max_tones: int = 96,
                    oversample: int = 8) -> np.ndarray:
    """Every frequency standing clear of its LOCAL noise floor in a window.

    Peaks are thinned to one per main lobe -- two maxima closer than the
    window's own resolution are one tone -- and returned off the zero-padded
    grid, because a tone need not sit on a DFT bin.
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

    floor = _local_floor(s)
    # The stricter of the caller's floor and what the window length demands.
    db = max(peak_db, _false_alarm_db(max(8, n // 2)))
    thr = floor * 10 ** (db / 20.0)

    # ONE AND A HALF HANN MAIN LOBES.  At exactly one lobe (4*fs/n) the
    # shoulders of a strong tone sit just outside the guard and each becomes
    # its own "tone": a four-tone group came back as twelve, every extra one
    # a sideband at the main-lobe edge of a real peak.
    guard_hz = 6.0 * fs / n
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

    # SUB-BIN INTERPOLATION.  The zero-padded grid still quantises the peak,
    # and the leftover frequency error does not stay in the frequency: the
    # joint fit models each tone at the frequency it is given, so a tone
    # offset by df sheds about 2*pi*df*T/sqrt(3) of its own amplitude into
    # the residual -- which is the noise estimate. On a 0.28 s dwell a 0.2 Hz
    # error costs ~5 dB of apparent SNR, so the tones look worse than they
    # are and the weights that follow are wrong. A parabola through the peak
    # and its two neighbours removes most of it for three multiplications.
    out = []
    for k in kept:
        y0, y1, y2 = float(s[k - 1]), float(s[k]), float(s[k + 1])
        denom = y0 - 2.0 * y1 + y2
        delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
        out.append(float(f[k] + delta * (f[1] - f[0])))
    return np.sort(np.array(out))


def tone_occupancy(ref: np.ndarray, fs: float, f_lo: float, f_hi: float,
                   max_tones: int = 24, n_grid: int = 240,
                   on_frac: float = 0.4) -> dict:
    """When is each strong tone on, and how many are on at once?

    The defining difference between a sweep and a multisine is SIMULTANEITY,
    so that is what this measures, rather than anything that stands in for
    it.  For each strong tone in the record, its demodulated envelope says
    when it was present; laying those on a common time grid says how many
    were present together.

    Measuring it directly matters because the obvious proxy -- segment the
    record, then count the tones in a segment -- inherits every weakness of
    the segmentation.  On a 1 Hz-to-3 kHz sweep the segmenter's frames are
    far too short to resolve the low-frequency steps, so it merged eighteen
    dwells into two and the tone count then reported a plain stepped sine as
    a multisine with four tones per dwell.  Co-occurrence needs no
    segmentation and is not fooled by that.
    """
    x = np.asarray(ref, float)
    x = x - x.mean()
    n = len(x)
    if n < 256:
        return {"ok": False, "tones_at_once": 0.0, "n_tones": 0}

    freqs = tones_in_window(x, fs, f_lo, f_hi, max_tones=max_tones)
    if freqs.size == 0:
        return {"ok": False, "tones_at_once": 0.0, "n_tones": 0}

    grid = np.linspace(0, n - 1, min(n_grid, n)).astype(int)
    on = np.zeros((len(freqs), len(grid)), bool)
    for i, f in enumerate(freqs):
        env, win = demod_envelope(x, fs, float(f))
        if len(env) < 4:
            continue
        centres = np.clip(grid - win // 2, 0, len(env) - 1)
        e = env[centres]
        peak = float(np.max(e)) if e.size else 0.0
        if peak <= 0:
            continue
        on[i] = e >= on_frac * peak

    live = on.sum(axis=0)
    busy = live[live > 0]
    at_once = float(np.median(busy)) if busy.size else 0.0
    return {"ok": True, "tones_at_once": at_once, "n_tones": int(freqs.size),
            "freqs": freqs}


def classify_excitation(ref: np.ndarray, fs: float, f_lo: float = 0.15,
                        f_hi: float | None = None) -> dict:
    """Stepped sine or stepped multisine?  Decided from the record.

    This is not a stylistic choice between two estimators -- they answer
    different questions and each is badly wrong about the other's data:

    * The stepped-sine detector maximises a SINGLE-tone fit to find each
      dwell.  Inside a multisine that fit never improves, because what
      limits it is the other tones, so the window shrinks to its floor of a
      few periods.  Measured on a 15-tone synthetic in three groups: 4 of 15
      tones found, with windows about a millisecond long.
    * The multisine path fits a whole group at once.  Handed a stepped sine
      it is not merely wasteful: if the segmentation then merges several
      steps into one "dwell", it will fit tones jointly that were never on
      together, over a window most of them were absent for.

    So the test is on SIMULTANEITY itself (:func:`tone_occupancy`): how many
    strong tones are on at the same time.  One is a sweep.  Several is a
    multisine.
    """
    f_hi = float(f_hi or 0.45 * fs)
    occ = tone_occupancy(ref, fs, f_lo, f_hi)
    at_once = float(occ.get("tones_at_once", 0.0))
    kind = "multisine" if at_once >= 1.5 else "stepped"
    dwells = detect_dwells(ref, fs) if kind == "multisine" else []
    return {"kind": kind, "tones_at_once": at_once,
            "n_tones": int(occ.get("n_tones", 0)),
            "n_dwells": len(dwells)}


def detect_multisine_schedule(ref: np.ndarray, fs: float,
                              f_lo: float = 0.15, f_hi: float | None = None,
                              peak_db: float = 12.0,
                              verbose: bool = True) -> list[Step]:
    """Blind schedule for a stepped multisine.

    Returns one :class:`Step` per (dwell, tone) pair, so several Steps share
    a start/stop when they were applied together.  That overlap is the record
    of simultaneity, and :func:`group_simultaneous` reads it back so the
    group can be fitted jointly rather than one tone at a time.
    """
    f_hi = float(f_hi or 0.45 * fs)
    dwells = detect_dwells(ref, fs)
    steps: list[Step] = []
    for a, b in dwells:
        y = ref[a:b]
        freqs = tones_in_window(y, fs, f_lo, f_hi, peak_db=peak_db)
        if freqs.size == 0:
            continue
        phasors, _r, snr, info = fit_multitone(y, fs, freqs)
        if not info.get("ok"):
            continue
        for f, A, sn in zip(freqs, phasors, snr):
            if not np.isfinite(A):
                continue
            steps.append(Step(
                freq=float(f), start=int(a), stop=int(b), amp=float(abs(A)),
                snr_db=float(sn),
                thd=harmonic_distortion(y, fs, float(f)),
                stationarity=_stationarity(ref, fs, float(f), a, b)))
    steps.sort(key=lambda s: s.freq)
    if verbose:
        print(f"  multisine schedule: {len(dwells)} dwell(s), "
              f"{len(steps)} tone(s) "
              + (f"({steps[0].freq:.3f} .. {steps[-1].freq:.1f} Hz)"
                 if steps else "(none)"))
    return steps


def group_simultaneous(steps: list[Step], overlap: float = 0.5
                       ) -> list[list[int]]:
    """Indices of steps that share a dwell, i.e. were applied together.

    Simultaneity is not something to configure -- it is already in the
    windows.  Two tones whose dwell windows coincide were on at the same
    time; two tones from different steps of a sweep were not.  Grouping on
    the windows therefore handles a stepped sine (every group of size one),
    a single multisine over the whole record (one group of everything) and a
    stepped multisine (the general case) without being told which it is.
    """
    groups: list[list[int]] = []
    for i, st in enumerate(steps):
        placed = False
        for g in groups:
            ref = steps[g[0]]
            lo = max(ref.start, st.start)
            hi = min(ref.stop, st.stop)
            shared = max(0, hi - lo)
            shortest = min(ref.stop - ref.start, st.stop - st.start)
            if shortest > 0 and shared / shortest >= overlap:
                g.append(i)
                placed = True
                break
        if not placed:
            groups.append([i])
    return groups


# ===========================================================================
# 5. Impedance
# ===========================================================================


def segment_spectrum(fam: FamosFile, uc_name: str, seg: str, steps: list[Step],
                     K: float, gain: dict, uc_gain: dict
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z_s(f) in ohm*cm^2, with per-step SNR.

        u_s  -> j_s = u_s / K(T)          [A/cm^2]     (Abgleich)
        Z_s  = U_cell / j_s = K * U_cell / u_s          [ohm*cm^2]

    The segment area does not appear: the calibration already returns a
    current density.  What the area is needed for is everything that sums
    across segments -- see aggregate() and dc_closure().
    """
    uc = fam.channel(uc_name)
    sg = fam.channel(seg)
    fs = fam.fs
    freq = np.array([s.freq for s in steps])
    Z = np.empty(len(steps), dtype=complex)
    snr = np.empty(len(steps))

    for i, st in enumerate(steps):
        Auc, _r1, s1 = fit3(uc[st.start:st.stop], fs, st.freq)
        Aseg, _r2, s2 = fit3(sg[st.start:st.stop], fs, st.freq)
        Z[i] = K * Auc / Aseg if Aseg != 0 else complex("nan")
        snr[i] = min(s1, s2)

    Z = Z / gain_at(gain, seg, freq) * gain_at(uc_gain, "all", freq)

    # Both channels answer the same imposed current, so the raw ratio can come
    # out with a global sign; Re(Z) of an electrochemical cell is positive.
    if np.isfinite(np.real(np.nanmean(Z))) and np.real(np.nanmean(Z)) < 0:
        Z = -Z
    return freq, Z, snr


def deskew(freq: np.ndarray, Z: dict[str, np.ndarray], fam: FamosFile,
           uc_name: str, dt_range: float = 200e-6, verbose: bool = True
           ) -> tuple[dict[str, np.ndarray], dict]:
    """Remove the acquisition skew between the reference and the segment
    channels, measured with Z-HIT.

    Alternating channels are converted by different ADC cores.  A fixed offset
    between them is a pure delay: it multiplies Z by exp(-j w dt), leaves |Z|
    untouched and rotates the phase.  At 10 kHz sampling, 50 us is 54 degrees
    at 3 kHz, which destroys the high-frequency arc.

    A delay is an ALL-PASS, so it cannot be found by requiring internal
    consistency of the arc shape -- the older parity-group heuristic was
    guessing.  Z-HIT rebuilds |Z| out of the phase, so a delay makes the
    reconstruction disagree with the measured |Z|.

    Two safeguards, because a wrong correction is worse than none:

    * the cost curves of all segments are summed and ONE delay is fitted for
      the whole card.  A single spectrum gives a shallow minimum; dozens of
      them give a sharp one, and the skew is a property of the hardware, not
      of the segment.
    * the correction is only applied if the band actually resolves it -- the
      delay must turn the phase by more than about 10 degrees at the highest
      measured frequency -- and if the summed cost improves materially over
      dt = 0.  Otherwise the result is reported and dt is left at zero.

    The parity of the channel index is reported as a cross-check, never
    assumed.
    """
    p_uc = fam.position(uc_name)
    grid = np.linspace(-dt_range, dt_range, 401)
    total = np.zeros_like(grid)
    per_seg: dict[str, float] = {}
    n_used = 0

    for sg, z in Z.items():
        m = np.isfinite(z)
        if m.sum() < 8:
            continue
        _dt, info = estimate_delay(freq[m], z[m], dt_range=dt_range, n_grid=401)
        c = np.asarray(info["cost"], float)
        if not np.all(np.isfinite(c)) or c.max() <= 0:
            continue
        total += c / c.max()                 # normalise: segments differ in |Z|
        per_seg[sg] = float(info["dt_s"])
        n_used += 1

    info: dict = {"per_segment_dt_us": {k: v * 1e6 for k, v in per_seg.items()},
                  "n_segments_used": n_used}
    if n_used == 0:
        return Z, info | {"applied": False, "reason": "no usable spectra"}

    k = int(np.argmin(total))
    dt = float(grid[k])
    gain = (total[len(grid) // 2] - total[k]) / max(total[len(grid) // 2], 1e-30)
    phase_deg = abs(np.degrees(2 * np.pi * float(np.max(freq)) * dt))
    resolvable = phase_deg > 10.0 and gain > 0.10

    same = [v for s_, v in per_seg.items() if fam.position(s_) % 2 == p_uc % 2]
    opp = [v for s_, v in per_seg.items() if fam.position(s_) % 2 != p_uc % 2]
    info.update({
        "dt_s": dt, "dt_us": dt * 1e6, "dt_samples": dt * fam.fs,
        "cost_improvement": float(gain),
        "phase_at_fmax_deg": float(phase_deg),
        "spread_us": float(np.std(list(per_seg.values())) * 1e6),
        "same_parity_median_us": float(np.median(same) * 1e6) if same else None,
        "opposite_parity_median_us": float(np.median(opp) * 1e6) if opp else None,
        "applied": bool(resolvable),
    })

    if verbose:
        print(f"  skew (Z-HIT, {n_used} spectra): {dt*1e6:+.1f} us "
              f"({dt*fam.fs:+.3f} samples), {phase_deg:.1f} deg at "
              f"{np.max(freq):.0f} Hz, cost -{100*gain:.0f} %")
        if same and opp:
            print(f"    per-segment median: same parity as {uc_name} "
                  f"{np.median(same)*1e6:+.1f} us, opposite "
                  f"{np.median(opp)*1e6:+.1f} us")
        if not resolvable:
            print("    NOT APPLIED: the measured band cannot resolve a delay "
                  "this small. Extend the sweep towards fs/3 if the "
                  "high-frequency arc matters.")

    if not resolvable:
        return Z, info
    return ({s_: z * np.exp(1j * 2 * np.pi * freq * dt) for s_, z in Z.items()},
            info)


# ===========================================================================
# 6. DC closure and aggregation -- this is where the areas enter
# ===========================================================================


def dc_closure(j_dc: dict[str, float], areas: dict[str, float],
               i_setpoint: float | None) -> dict:
    """sum_s j_s * A_s  against the known cell current.

    With the Abgleich calibration this is a genuine, independent validation of
    the whole chain: via resistances, amplifier gains, areas and acquisition.
    Nothing here was tuned to make it come out right.
    """
    segs = list(j_dc)
    A = {s: areas[s] for s in segs}
    I = {s: j_dc[s] * A[s] for s in segs}
    A_meas = sum(A.values())
    I_meas = sum(I.values())

    jv = np.array([j_dc[s] for s in segs])
    Av = np.array([A[s] for s in segs])
    out = {
        "n_measured": len(segs),
        "n_expected": geom.N_SEGMENTS,
        "partial": len(segs) < geom.N_SEGMENTS,
        "area_measured_cm2": A_meas,
        "area_total_cm2": geom.A_CELL_CM2,
        "I_measured_A": I_meas,
        "j_area_weighted_mean_A_cm2": float(np.sum(jv * Av) / np.sum(Av)),
        "j_unweighted_mean_A_cm2": float(jv.mean()),
        "j_min_A_cm2": float(jv.min()),
        "j_max_A_cm2": float(jv.max()),
        "j_cv_pct": float(100 * jv.std() / abs(jv.mean())) if jv.mean() else np.nan,
    }
    if len(segs) < geom.N_SEGMENTS:
        out["I_extrapolated_A"] = I_meas * geom.A_CELL_CM2 / A_meas
    if i_setpoint:
        I_cmp = out.get("I_extrapolated_A", I_meas)
        out["i_setpoint_A"] = i_setpoint
        out["deviation_pct"] = 100.0 * (I_cmp / i_setpoint - 1.0)
        out["j_cell_average_A_cm2"] = i_setpoint / geom.A_CELL_CM2
    return out


def aggregate(freq: np.ndarray, Z: dict[str, np.ndarray],
              areas: dict[str, float]) -> np.ndarray:
    """Cell-level spectrum predicted from the local ones, in ohm*cm^2.

        every segment sees the same U_cell, so they are in parallel:
            I_cell = sum_s j_s A_s = U_cell * sum_s (A_s / Z_s)
            Z_cell = U_cell / I_cell         -> ASR = A_cell / sum(A_s / Z_s)

    An unweighted mean of the local spectra is NOT this quantity, and with
    areas spanning a factor of 12.5 the difference is large.  This curve is
    what a cell-level instrument would measure; when one is available it is a
    validation, and when it is not, it is still the correct cell spectrum.
    """
    Y = np.zeros(len(freq), dtype=complex)
    A_used = 0.0
    for s, z in Z.items():
        a = areas[s]
        with np.errstate(divide="ignore", invalid="ignore"):
            Y = Y + np.where(np.isfinite(z) & (z != 0), a / z, 0)
        A_used += a
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(Y != 0, A_used / Y, np.nan)


# ===========================================================================
# 7. Plots
# ===========================================================================


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_spectra(freq, Z: dict[str, np.ndarray], Z_cell, title, path):
    plt = _mpl()
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    cm = plt.get_cmap("viridis")
    keys = sorted(Z, key=int)
    o = np.argsort(freq)
    for i, s in enumerate(keys):
        z = Z[s][o]
        c = cm(i / max(1, len(keys) - 1))
        ax[0].plot(np.real(z), -np.imag(z), "-o", ms=2.5, lw=.8, color=c, alpha=.8)
        ax[1].plot(freq[o], np.abs(z), "-", lw=.8, color=c, alpha=.8)
        ax[2].plot(freq[o], np.degrees(np.angle(z)), "-", lw=.8, color=c, alpha=.8)
    if Z_cell is not None:
        zc = Z_cell[o]
        ax[0].plot(np.real(zc), -np.imag(zc), "k--d", ms=4, lw=1.8, zorder=9,
                   label="cell, aggregated from local (area-weighted)")
        ax[1].plot(freq[o], np.abs(zc), "k--", lw=1.8, zorder=9)
        ax[2].plot(freq[o], np.degrees(np.angle(zc)), "k--", lw=1.8, zorder=9)
        ax[0].legend(fontsize=8)
    ax[0].set(xlabel="Z'  [mΩ·cm²]", ylabel="−Z''  [mΩ·cm²]", title="Nyquist")
    ax[0].axhline(0, color=".6", lw=.7, ls=":")
    ax[0].set_aspect("equal", adjustable="datalim")
    ax[1].set(xscale="log", yscale="log", xlabel="f [Hz]",
              ylabel="|Z| [mΩ·cm²]", title="Bode magnitude")
    ax[2].set(xscale="log", xlabel="f [Hz]", ylabel="phase [°]",
              title="Bode phase", ylim=(-90, 30))
    ax[2].axhline(0, color=".6", lw=.7, ls=":")
    for a in ax:
        a.grid(alpha=.25, which="both")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_schedule(steps: list[Step], fs: float, path):
    plt = _mpl()
    f = np.array([s.freq for s in steps])
    t0 = np.array([s.start for s in steps]) / fs
    dur = np.array([s.stop - s.start for s in steps]) / fs
    ok = np.array([s.valid() for s in steps])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    ax[0].barh(f, dur, left=t0, height=f * 0.35, color=np.where(ok, "tab:green", "tab:red"))
    ax[0].set(yscale="log", xlabel="t [s]", ylabel="f [Hz]",
              title="detected dwell of every step")
    ax[1].semilogx(f[ok], [s.snr_db for s, k in zip(steps, ok) if k], "o",
                   color="tab:green", label="pass")
    if (~ok).any():
        ax[1].semilogx(f[~ok], [s.snr_db for s, k in zip(steps, ok) if not k],
                       "x", color="tab:red", ms=8, label="fail")
    ax[1].axhline(8, color="k", lw=.8, ls="--")
    ax[1].set(xlabel="f [Hz]", ylabel="SNR [dB]", title="per-step SNR")
    ax[1].legend(fontsize=8)
    ax[2].semilogx(f, [s.thd * 100 for s in steps], "o", color="tab:blue",
                   label="THD (linearity)")
    ax[2].semilogx(f, [s.stationarity * 100 for s in steps], "s",
                   color="tab:orange", ms=4, label="drift over window")
    ax[2].axhline(10, color="k", lw=.8, ls="--")
    ax[2].set(xlabel="f [Hz]", ylabel="[%]", title="linearity and stationarity")
    ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=.3, which="both")
    fig.suptitle("Sweep schedule recovered from the data alone", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_validation(val: dict, path):
    plt = _mpl()
    segs = sorted(val, key=int)
    kk = [100 * val[s]["linkk_res_max"] for s in segs]
    zh = [100 * val[s]["zhit_dev_rms"] for s in segs]
    x = np.arange(len(segs))
    fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    ax[0].bar(x, kk, color=["tab:green" if v < 2 else "tab:red" for v in kk])
    ax[0].axhline(2, color="k", lw=.8, ls="--")
    ax[0].set(ylabel="lin-KK max residual [%]",
              title="Validity of every local spectrum — no reference instrument used")
    ax[1].bar(x, zh, color=["tab:green" if v < 5 else "tab:red" for v in zh])
    ax[1].axhline(5, color="k", lw=.8, ls="--")
    ax[1].set(ylabel="Z-HIT rms deviation [%]", xlabel="segment")
    ax[1].set_xticks(x[::2])
    ax[1].set_xticklabels([segs[i] for i in range(0, len(segs), 2)], fontsize=7)
    for a in ax:
        a.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# 8. Driver
# ===========================================================================


def evaluate(dat_dir, out_dir, curr_cal=None, temp_cal=None, gain_file=None,
             uc_gain_file=None, areas_file=None, i_setpoint=None,
             exclude_segments=(), ppd=12, min_snr=8.0, deskew_on=True):
    dat_dir, out_dir = Path(dat_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude = {str(s) for s in exclude_segments}

    areas = geom.areas_from_csv(areas_file) if areas_file else geom.areas()
    geom.self_check(verbose=False)
    print(f"geometry: {geom.N_SEGMENTS} segments, "
          f"{min(areas.values()):.3f}..{max(areas.values()):.3f} cm^2, "
          f"total {sum(areas.values()):.2f} cm^2")

    cal = PlateCalibration.load(curr_cal, temp_cal)
    if not cal.has_current_cal:
        raise SystemExit(
            "no --curr-cal given. The Abgleich file (curr.csv, 72 lines "
            "'c0;c1') is what makes this evaluation absolute and independent "
            "of any potentiostat; without it there is no traceable scale.")
    gain = load_gain(gain_file) if gain_file else {}
    uc_gain = load_gain(uc_gain_file) if uc_gain_file else {}
    if not gain:
        print("  note: no --gain file. The amplifier roll-off (about -3 dB "
              "near 20 kHz on this plate) stays in the result; treat the "
              "top decade as uncorrected.")

    files = sorted(dat_dir.glob("*.DAT")) + sorted(dat_dir.glob("*.dat"))
    if not files:
        raise SystemExit(f"no .DAT files in {dat_dir}")

    Z_all: dict[str, np.ndarray] = {}
    snr_all: dict[str, np.ndarray] = {}
    dc_u: dict[str, float] = {}
    seg_T: dict[str, float] = {}
    card_info: dict[str, dict] = {}
    freq_ref = None
    steps_ref = None

    for fp in files:
        fam = FamosFile(fp)
        card = fp.stem.split("_")[-1]
        print(f"\n=== {fp.name}  ({fam.fs:.0f} Hz, {fam.n_ch} ch, "
              f"{fam.n_samples/fam.fs:.1f} s)")
        print(f"  segments: {fam.segment_names}")
        if not fam.uc_names:
            print("  no UC reference channel -- skipped")
            continue

        # ---- temperature -------------------------------------------------
        sensor_T = {}
        for tn in fam.temp_names:
            key = tn.lower()
            if key in cal.temp_c0:
                u = float(np.mean(fam.channel(tn)[::10]))
                T = cal.temperature(key, u)
                if 0.0 < T < 120.0:
                    sensor_T[key] = T
                else:
                    print(f"  {tn}: u={u:.3f} V -> T={T:.1f} degC implausible "
                          f"(AC-coupled channel?), ignored")
        if sensor_T:
            print("  temperature: " + ", ".join(f"{k}={v:.1f}" for k, v in sensor_T.items()))
            T_seg = geom.segment_temperatures(sensor_T)
        else:
            print(f"  temperature: no usable sensor, falling back to "
                  f"{T_FALLBACK_C} degC (K varies ~0.3 %/K, so this is a "
                  f"small but real systematic)")
            T_seg = {s: T_FALLBACK_C for s in areas}

        # ---- reference channel and schedule -------------------------------
        uc_name = max(fam.uc_names, key=lambda c: float(np.std(fam.channel(c)[::10])))
        print(f"  reference: {uc_name}")
        steps = detect_schedule(fam.channel(uc_name), fam.fs, ppd=ppd,
                                min_snr_db=min_snr)
        good = [s for s in steps if s.valid(min_snr)]
        if not good:
            print("  no usable steps -- skipped")
            continue
        if steps_ref is None:
            steps_ref = steps
            plot_schedule(steps, fam.fs, out_dir / "schedule.png")

        # ---- per segment ---------------------------------------------------
        Zc, snrc = {}, {}
        for sg in fam.segment_names:
            if sg in exclude or sg not in cal.seg_c0:
                continue
            K = cal.K(sg, T_seg.get(sg, T_FALLBACK_C))
            try:
                f_v, z, sn = segment_spectrum(fam, uc_name, sg, good, K, gain, uc_gain)
            except Exception as e:                       # noqa: BLE001
                print(f"  segment {sg}: {e}")
                continue
            Zc[sg], snrc[sg] = z, sn
            seg_T[sg] = T_seg.get(sg, T_FALLBACK_C)
            dc_u[sg] = float(np.mean(fam.channel(sg)[::10]))
        freq = np.array([s.freq for s in good])

        if Zc and deskew_on:
            Zc, info = deskew(freq, Zc, fam, uc_name)
            card_info[card] = info

        if freq_ref is None:
            freq_ref = freq
        for k, v in Zc.items():
            if len(freq) == len(freq_ref) and np.allclose(freq, freq_ref):
                Z_all[k], snr_all[k] = v, snrc[k]
            else:                                 # cards may differ slightly
                o = np.argsort(freq)
                Z_all[k] = (np.interp(freq_ref, freq[o], np.real(v)[o])
                            + 1j * np.interp(freq_ref, freq[o], np.imag(v)[o]))
                snr_all[k] = np.interp(freq_ref, freq[o], snrc[k][o])

    if not Z_all:
        raise SystemExit("no impedance computed")

    # ---- validation --------------------------------------------------------
    print("\n=== validation (lin-KK + Z-HIT, no reference instrument)")
    val = {}
    for s, z in Z_all.items():
        m = np.isfinite(z)
        if m.sum() < 8:
            continue
        kk = lin_kk(freq_ref[m], z[m])
        zh = z_hit(freq_ref[m], z[m])
        val[s] = {"linkk_M": kk.get("M"), "linkk_mu": kk.get("mu"),
                  "linkk_res_max": kk.get("res_max"),
                  "linkk_valid": kk.get("valid"),
                  "zhit_dev_rms": zh.get("dev_rms"),
                  "zhit_drift_lf": zh.get("drift_lf"),
                  "zhit_valid": zh.get("valid"),
                  "snr_median_db": float(np.nanmedian(snr_all[s]))}
    n_kk = sum(1 for v in val.values() if v["linkk_valid"])
    n_zh = sum(1 for v in val.values() if v["zhit_valid"])
    print(f"  lin-KK pass: {n_kk}/{len(val)}   Z-HIT pass: {n_zh}/{len(val)}")
    worst = sorted(val, key=lambda s: -val[s]["linkk_res_max"])[:5]
    print("  worst lin-KK residual: " +
          ", ".join(f"seg {s} {100*val[s]['linkk_res_max']:.1f} %" for s in worst))

    # ---- current density and closure ---------------------------------------
    j_dc = {s: dc_u[s] / cal.K(s, seg_T[s]) for s in Z_all if s in dc_u}
    chk = dc_closure(j_dc, areas, i_setpoint)
    print("\n=== DC current density (from the Abgleich, not fitted to anything)")
    for k in ("area_measured_cm2", "I_measured_A", "I_extrapolated_A",
              "i_setpoint_A", "deviation_pct", "j_area_weighted_mean_A_cm2",
              "j_min_A_cm2", "j_max_A_cm2", "j_cv_pct"):
        if k in chk:
            print(f"  {k:32s} {chk[k]:.4f}" if isinstance(chk[k], float)
                  else f"  {k:32s} {chk[k]}")

    # ---- aggregate ----------------------------------------------------------
    Z_cell = aggregate(freq_ref, Z_all, areas)
    hf = int(np.argmax(freq_ref))
    Rs = {s: float(np.real(z[hf])) for s, z in Z_all.items()}
    rsv = np.array(list(Rs.values()))
    print(f"\n  Rs (Re Z at {freq_ref[hf]:.0f} Hz) = {1000*rsv.mean():.1f} "
          f"+/- {1000*rsv.std():.1f} mOhm*cm2 "
          f"(spread {100*rsv.std()/rsv.mean():.1f} %) over {len(rsv)} segments")
    print(f"  aggregated cell ASR at that frequency = "
          f"{1000*np.real(Z_cell[hf]):.1f} mOhm*cm2")

    # ---- export -------------------------------------------------------------
    with (out_dir / "impedance.csv").open("w") as fh:
        fh.write("segment,area_cm2,freq_hz,z_real_mohm_cm2,z_imag_mohm_cm2,snr_db\n")
        for s in sorted(Z_all, key=int):
            for f0, z, sn in zip(freq_ref, Z_all[s], snr_all[s]):
                fh.write(f"{s},{areas[s]:.5f},{f0:.6f},{1000*np.real(z):.6f},"
                         f"{1000*np.imag(z):.6f},{sn:.2f}\n")
    with (out_dir / "cell_aggregate.csv").open("w") as fh:
        fh.write("freq_hz,z_real_mohm_cm2,z_imag_mohm_cm2\n")
        for f0, z in zip(freq_ref, Z_cell):
            fh.write(f"{f0:.6f},{1000*np.real(z):.6f},{1000*np.imag(z):.6f}\n")
    with (out_dir / "segments_summary.csv").open("w") as fh:
        fh.write("segment,area_cm2,cx_mm,cy_mm,T_degC,u_dc_V,j_dc_A_cm2,"
                 "I_dc_A,Rs_mohm_cm2,linkk_res_pct,zhit_dev_pct,snr_db\n")
        for s in sorted(Z_all, key=int):
            g = geom.SEGMENTS[s]
            v = val.get(s, {})
            fh.write(f"{s},{areas[s]:.5f},{g.cx_mm:.2f},{g.cy_mm:.2f},"
                     f"{seg_T.get(s, float('nan')):.2f},{dc_u.get(s, float('nan')):.6f},"
                     f"{j_dc.get(s, float('nan')):.6f},"
                     f"{j_dc.get(s, float('nan'))*areas[s]:.6f},"
                     f"{1000*Rs[s]:.4f},{100*v.get('linkk_res_max', float('nan')):.3f},"
                     f"{100*v.get('zhit_dev_rms', float('nan')):.3f},"
                     f"{v.get('snr_median_db', float('nan')):.1f}\n")

    summary = {
        "gamry_used": False,
        "geometry": {"total_area_cm2": sum(areas.values()),
                     "area_min_cm2": min(areas.values()),
                     "area_max_cm2": max(areas.values()),
                     "source": areas_file or "r2d2_geometry.py"},
        "n_steps_used": int(len(freq_ref)),
        "f_min_hz": float(freq_ref.min()), "f_max_hz": float(freq_ref.max()),
        "cards": card_info,
        "dc_check": chk,
        "validation": val,
        "Rs_ohm_cm2": Rs,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # ---- plots --------------------------------------------------------------
    plot_spectra(freq_ref, {k: v * 1000 for k, v in Z_all.items()},
                 Z_cell * 1000,
                 f"Local EIS — {len(Z_all)} segments — no potentiostat reference",
                 out_dir / "local_eis.png")
    plot_validation(val, out_dir / "qc_validation.png")
    geom.plot_map(out_dir / "map_current_density.png", j_dc,
                  label="j  [A/cm$^2$]", title="DC current density")
    geom.plot_map(out_dir / "map_Rs.png", {k: 1000 * v for k, v in Rs.items()},
                  label="Rs  [mΩ·cm$^2$]",
                  title=f"High-frequency resistance at {freq_ref[hf]:.0f} Hz")
    print(f"\nwritten to {out_dir}")
    return summary


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dat", required=True, help="folder with the .DAT files")
    ap.add_argument("--out", default="./results")
    ap.add_argument("--curr-cal", required=True,
                    help="Abgleich file curr.csv (72 lines 'c0;c1')")
    ap.add_argument("--temp-cal", help="temp.csv (4 lines 'c0;c1')")
    ap.add_argument("--gain", help="segment frequency response, "
                                   "'segment,freq,re,im'")
    ap.add_argument("--uc-gain", help="same for the cell-voltage channel")
    ap.add_argument("--areas", help="override segment areas, 'segment,area_cm2'")
    ap.add_argument("--current", type=float, default=None,
                    help="total DC cell current [A] -- used ONLY as an "
                         "independent check, never to scale the result")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--ppd", type=int, default=12,
                    help="trial frequencies per decade in the blind scan")
    ap.add_argument("--min-snr", type=float, default=8.0)
    ap.add_argument("--no-deskew", action="store_true")
    a = ap.parse_args()
    evaluate(a.dat, a.out, curr_cal=a.curr_cal, temp_cal=a.temp_cal,
             gain_file=a.gain, uc_gain_file=a.uc_gain, areas_file=a.areas,
             i_setpoint=a.current, exclude_segments=a.exclude, ppd=a.ppd,
             min_snr=a.min_snr, deskew_on=not a.no_deskew)


if __name__ == "__main__":
    main()
