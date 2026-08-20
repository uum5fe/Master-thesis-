#!/usr/bin/env python3
"""
csv_pipeline.py  --  local EIS from the CSV logger
==================================================

The evaluation path for `cfg.source_format == "csv"`.  It produces the same
products as bronze/silver/gold -- per-segment spectra, a cell aggregate, ECM
parameters, Nyquist plots and plate heat maps -- from a completely different
starting point, and it does NOT reuse the FAMOS stages.  This module explains
why in the places where the difference is a decision rather than an
accident.

WHAT IS DELIBERATELY NOT DONE HERE
----------------------------------
No inter-card synchronisation, no clock-drift regression, no triplet closure,
no `skew_model="structural"` fit.  Those exist because five Dewetron cards
free-run against each other and the offset has to be *measured* on every run.
A CSV logger writes one row per instant for the whole plate.  There is no
second clock, so there is no delay to estimate -- and an estimator pointed at
a quantity that does not exist returns noise with a plausible-looking
uncertainty, which is worse than not running it.

The one skew that CAN exist on this rig is a channel *scan*: if the logger
walks the channel list within a row, the segment and the cell voltage in one
row are separated by a known fraction of the row period.  That is handled
analytically from `scan_rate_hz`, not fitted (`apply_scan_skew`).  Left
unset, no correction is applied and the manifest records the assumption.

WHAT REPLACES IT
----------------
Three things the FAMOS path does not need and this one does:

  1. **Phasors on the recorded timestamps.**  A logger's sample interval
     jitters.  Resampling onto a uniform grid to make an FFT legal injects
     interpolation error where the phase matters most; a least-squares sine
     fit needs the sample times to be *known*, not uniform.
     `csv_source.fit_phasor` picks the uniform-grid fit only when the
     measured jitter is below 0.1 %.

  2. **Quantisation in the error budget.**  The file is text.  The printed
     resolution is a uniform quantiser of step q, contributing variance
     q^2/12 on top of the electronic noise.  It is recovered from the data
     (`csv_source.quantisation_step`) and added in quadrature, so a
     six-decimal export of a 10 mV shunt signal is visible as an error bar
     rather than as suspiciously clean data.

  3. **Excitation discovery that does not assume a schedule.**  The tones may
     be a simultaneous multisine, a stepped sweep, or a list the operator
     already knows.  All three are supported and the choice is reported.

WHAT IS SHARED WITH THE FAMOS PATH, AND MUST BE
-----------------------------------------------
The physics and the hardware description, because those belong to the plate
and not to the file format:

    j_s = u_s / K(T)          Abgleich, eis_local.PlateCalibration
    Z_s = U_cell / j_s        area-free: K already returns a current density
    Z_cell = A / sum(A_s/Z_s) segments in parallel
    areas, centroids, maps    r2d2_geometry, selected by cfg.plate

That last one is why `cfg.plate` matters here exactly as much as it does for
FAMOS: a gen2 recording read with the gen1 map still runs, still fills every
segment, and still draws a heat map -- of the wrong plate.

THE CHAIN RESPONSE IS NOT OPTIONAL AT THE TOP OF THE BAND
---------------------------------------------------------
`cfg.gain_file` (see gamry_dta.py) removes the measuring chain's own roll-off.
Measured on both plates it is -11 deg of phase at 4.5 kHz, rising to -24 deg
at 10 kHz.  If the CSV logger runs to higher frequency than the FAMOS rig --
and a logger that writes one row per instant usually can -- this correction
stops being a refinement and becomes the difference between an R_omega and a
number.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import csv_source
import utils

# ===========================================================================
# 1. Excitation discovery
# ===========================================================================


def _uniform_view(t, y, uniform, fs):
    """`y` on a uniform grid -- FOR PEAK DETECTION ONLY.

    Detection needs to know *which* frequencies are present, to a fraction of
    a bin.  Interpolating a jittered record onto a uniform grid costs a small
    amplitude error and a phase error that grows with frequency -- neither of
    which moves a peak's location.  So the interpolation is safe here and
    nowhere else: the phasors that become Z are always fitted on the recorded
    timestamps (csv_source.fit_phasors_multi), never on this view.
    """
    y = np.asarray(y, float)
    if uniform or t is None:
        return y, fs
    n = t.size
    tu = np.linspace(t[0], t[-1], n)
    return np.interp(tu, t, y), (n - 1) / (t[-1] - t[0])


def _peaks(y, fs, f_lo, f_hi, snr_db, dyn_range_db: float = 40.0
           ) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima of the windowed amplitude spectrum, above TWO thresholds.

    The floor is the MEDIAN of the in-band magnitude, not the mean: an
    excitation occupies a handful of bins out of tens of thousands, so the
    median is the noise by construction and is not dragged up by the signal
    the way a mean is.

    But a floor alone is not enough, and the failure is instructive.  On a
    clean record the noise floor approaches the numerical one, so "10 dB
    above the floor" admits the Hann window's own sidelobes -- 600 spurious
    "tones" on a twelve-tone multisine, every one of them a real feature of
    the spectrum and none of them excitation.  The second gate is a dynamic
    range below the strongest peak: a designed multisine spreads its tones
    over perhaps 20 dB, so anything 40 dB down is leakage.
    """
    y = np.asarray(y, float)
    n = y.size
    if n < 64:
        return np.array([]), np.array([])
    w = np.hanning(n)
    Y = np.abs(np.fft.rfft((y - y.mean()) * w)) * 2.0 / np.sum(w)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    band = (f >= f_lo) & (f <= f_hi)
    if band.sum() < 8:
        return np.array([]), np.array([])
    floor = float(np.median(Y[band])) + 1e-30
    thr = floor * 10.0 ** (max(snr_db, 10.0) / 20.0)
    thr = max(thr, float(np.max(Y[band])) * 10.0 ** (-dyn_range_db / 20.0))

    idx = np.flatnonzero(band & (Y > thr))
    if idx.size == 0:
        return np.array([]), np.array([])
    # keep only local maxima, and merge bins belonging to the same tone
    keep = [i for i in idx
            if Y[i] >= Y[max(0, i - 2):min(Y.size, i + 3)].max()]
    out_f, out_a = [], []
    for i in keep:
        if out_f and (f[i] - out_f[-1]) < max(3.0 * (f[1] - f[0]),
                                              0.005 * f[i]):
            if Y[i] > out_a[-1]:
                out_f[-1], out_a[-1] = f[i], Y[i]
            continue
        out_f.append(f[i])
        out_a.append(Y[i])
    return np.asarray(out_f), np.asarray(out_a)


def detect_tones(m: csv_source.CsvMeasurement, cfg, log) -> dict:
    """Find the excitation frequencies in the cell-voltage record.

    Strategy, in order:
      * `cfg` carries an explicit tone list        -> use it, verify amplitude
      * the record is one block (multisine)        -> peak-pick a periodogram
      * the record is a stepped sweep              -> segment into dwells

    Returns {"mode", "tones": [...], "windows": [(i0, i1, f), ...]}.
    `windows` is empty for a multisine: the whole record is one window and
    every tone is fitted over all of it, which is exactly the leakage-free
    case a designed multisine is chosen for.
    """
    ref = m.u_cell
    if ref is None:
        # No cell-voltage channel: fall back to the loudest segment, which
        # carries the same imposed excitation.  Say so -- the reference
        # choice changes what the phase is measured against.
        key = max(m.u_seg, key=lambda k: float(np.nanstd(m.u_seg[k])))
        ref = m.u_seg[key]
        log.warning(f"  no cell-voltage column; using segment {key} as the "
                    f"excitation reference. Z will be a RATIO OF SEGMENTS, "
                    f"not an impedance, until a u_cell column is supplied.")

    reg = csv_source.regularity(m.t)
    fs = m.fs_nominal
    t = m.t if m.t is not None else np.arange(ref.size) / fs

    explicit = getattr(cfg, "csv_tones", None)
    if explicit:
        tones = sorted(float(x) for x in explicit)
        return {"mode": "explicit", "tones": tones, "windows": [],
                "reference": "u_cell" if m.u_cell is not None else "segment"}

    f_lo = max(cfg.f_min_hz, 3.0 / max(t[-1] - t[0], 1e-9))
    f_hi = min(cfg.f_max_hz, 0.45 * fs)
    if not np.isfinite(f_hi) or f_hi <= f_lo:
        raise ValueError(
            f"empty analysis band: f_min={f_lo:.4g} Hz, f_max={f_hi:.4g} Hz, "
            f"fs={fs:.4g} Hz, record {t[-1]-t[0]:.3g} s. A tone needs at "
            f"least three cycles inside the record and must sit below "
            f"0.45*fs.")

    yu, fsu = _uniform_view(t, ref, reg["uniform"], fs)
    tones_f, tones_a = _peaks(yu, fsu, f_lo, f_hi, cfg.min_snr_db)

    # MULTISINE OR SWEEP?  Not decidable from the count of peaks -- a stepped
    # sweep visits every frequency too, so the spectrum of the whole record
    # looks much the same.  What separates them is whether the tones are
    # SIMULTANEOUS.  Split the record into quarters: a multisine shows the
    # same tone set in every quarter, a sweep shows a different one in each.
    simultaneous = 0
    if tones_f.size >= 2:
        q = yu.size // 4
        seen = np.zeros(tones_f.size, int)
        for b in range(4):
            pf, _ = _peaks(yu[b * q:(b + 1) * q], fsu, f_lo, f_hi,
                           cfg.min_snr_db)
            if pf.size:
                for i, f in enumerate(tones_f):
                    if np.min(np.abs(pf - f)) < 0.02 * f:
                        seen[i] += 1
        simultaneous = int(np.sum(seen >= 3))

    if tones_f.size >= 4 and simultaneous >= 0.6 * tones_f.size:
        # Refine each tone with a 4-parameter fit so the reported frequency
        # is the tone and not the nearest FFT bin -- an off-bin tone costs
        # amplitude and phase in any estimator that assumes it is on-bin.
        tones = []
        for f0 in tones_f:
            if reg["uniform"]:
                f_hat, _A, _r, _snr = utils.fit4(ref, fs, float(f0))
                if abs(f_hat - f0) > 0.05 * f0:
                    f_hat = float(f0)
            else:
                f_hat = float(f0)
            tones.append(float(f_hat))
        tones = sorted(set(np.round(tones, 6)))
        return {"mode": "multisine", "tones": tones, "windows": [],
                "n_peaks": int(tones_f.size),
                "n_simultaneous": simultaneous,
                "reference": "u_cell" if m.u_cell is not None else "segment"}

    win = _detect_sweep_windows(t, yu, fsu, reg["uniform"], f_lo, f_hi, cfg)
    if win:
        return {"mode": "stepped_sweep", "tones": [w[2] for w in win],
                "windows": win,
                "reference": "u_cell" if m.u_cell is not None else "segment"}

    raise ValueError(
        "no excitation found. The cell-voltage record shows no tone more "
        f"than {cfg.min_snr_db:.0f} dB above its own noise floor between "
        f"{f_lo:.4g} and {f_hi:.4g} Hz, and no stepped dwells. Either the "
        "band is wrong, the reference column is not the excited one, or the "
        "file is a rest-phase recording. Pass the tone list explicitly with "
        "cfg.csv_tones to bypass detection.")


def _detect_sweep_windows(t, y, fs, uniform, f_lo, f_hi, cfg):
    """Split a stepped sweep into (start, stop, freq) dwells.

    Delegated to `eis_local.detect_schedule`, which already does this job on
    the FAMOS records and does it well: a log grid of trial frequencies, a
    demodulated envelope to locate each dwell, a four-parameter fit to
    recover the true frequency, then the window grown to the real dwell and
    polished until the residual stops improving.

    Reusing it is deliberate.  Finding a dwell is a property of the
    excitation, not of the file format -- the same load bank drives both
    rigs, so the same detector applies.  What does NOT carry over is
    everything downstream of the dwell: card alignment, per-card skew,
    drift.  Sharing the part that is genuinely common and duplicating
    nothing else is the whole point of keeping the two paths separate.

    The indices come back on the sample grid, and `_uniform_view` preserves
    the sample count, so they index the original record unchanged.
    """
    del t, uniform          # detection runs on the uniform view (see caller)
    try:
        import eis_local
        steps = eis_local.detect_schedule(
            np.asarray(y, float), fs, ppd=cfg.ppd, f_lo=f_lo, f_hi=f_hi,
            min_snr_db=cfg.min_snr_db, verbose=False)
    except Exception:                                # noqa: BLE001
        return []
    out = []
    for s in steps:
        if not s.valid(cfg.min_snr_db):
            continue
        if s.freq * (s.stop - s.start) / fs < cfg.min_cycles_per_dwell:
            continue
        out.append((int(s.start), int(s.stop), float(s.freq)))
    return sorted(out, key=lambda w: w[2])


# ===========================================================================
# 2. Impedance
# ===========================================================================


@dataclass
class SegmentSpectrum:
    segment: str
    freq: np.ndarray
    Z: np.ndarray                 # ohm*cm2
    sigma_rel: np.ndarray         # relative sd of |Z|, per point
    snr_db: np.ndarray
    n_used: np.ndarray            # samples behind each phasor
    flags: list[str] = field(default_factory=list)
    T_C: float = float("nan")
    K: float = float("nan")
    j_dc: float = float("nan")


def apply_scan_skew(freq: np.ndarray, Z: np.ndarray, seg_slot: int,
                    ref_slot: int, scan_rate_hz: float) -> np.ndarray:
    """Remove a KNOWN channel-scan offset.

    If the logger samples channel k at k/scan_rate after the row starts, the
    segment lags the reference by (seg_slot - ref_slot)/scan_rate.  A pure
    delay dt multiplies Z by exp(-j w dt), so the correction is a
    multiplication by exp(+j w dt) -- all-pass, invisible in |Z|, linear in
    phase.

    This is deliberately not fitted.  It is a property of the acquisition
    program, known exactly when the program is known, and fitting a parameter
    that is already known trades a certainty for an estimate.
    """
    if not scan_rate_hz or not np.isfinite(scan_rate_hz):
        return Z
    dt = (seg_slot - ref_slot) / float(scan_rate_hz)
    return Z * np.exp(1j * 2 * np.pi * freq * dt)


def segment_spectra(m: csv_source.CsvMeasurement, sched: dict, cal, T_seg: dict,
                    cfg, log) -> dict[str, SegmentSpectrum]:
    """Z_s(f) = U_cell(f) / j_s(f) for every segment, with uncertainty.

    The area never enters: the Abgleich returns a current density, so Z is
    already an area-specific resistance.  What the area is for is everything
    that sums across segments -- the cell aggregate and the DC closure.
    """
    reg = csv_source.regularity(m.t)
    fs = m.fs_nominal
    t = m.t if m.t is not None else np.arange(
        next(iter(m.u_seg.values())).size) / fs
    if m.t is None:
        log.warning("  no time column: rows assumed uniformly spaced at "
                    f"{fs:.4g} Hz. If the logger jitters, the phase at the "
                    f"top of the band is wrong and nothing here can tell.")

    ref = m.u_cell
    if ref is None:
        ref = m.u_seg[max(m.u_seg, key=lambda k: float(np.nanstd(m.u_seg[k])))]

    # A window is (i0, i1, [tones fitted jointly over it]).  A stepped sweep
    # gives one tone per window; a multisine gives one window carrying every
    # tone -- and those tones MUST be fitted together, see
    # csv_source.fit_phasors_multi.
    if sched["windows"]:
        windows = [(i0, i1, [f]) for (i0, i1, f) in sched["windows"]]
    else:
        windows = [(0, len(ref), list(sched["tones"]))]

    # Quantisation floor, once per channel: it is a property of the export.
    q_ref = csv_source.quantisation_step(ref)
    q_seg = {k: csv_source.quantisation_step(v) for k, v in m.u_seg.items()}

    gain = utils.load_gain(cfg.gain_file) if cfg.gain_file else {}
    scan = float(getattr(cfg, "csv_scan_rate_hz", 0.0) or 0.0)
    slots = dict(getattr(cfg, "csv_channel_slots", ()) or ())

    out: dict[str, SegmentSpectrum] = {}
    for seg, u in m.u_seg.items():
        T = float(T_seg.get(seg, np.nan))
        try:
            K = cal.K(seg, T) if cal.has_current_cal else np.nan
        except KeyError:
            K = np.nan
        if not np.isfinite(K):
            # No calibration row: the spectrum would be in volts per volt.
            # Keep the segment, flag it, and let gold decide -- deleting it
            # would leave a hole in the map that reads as a cold spot.
            K = np.nan

        # Relative sd of one phasor: the residual is the noise (which the
        # joint fit makes true), and the text quantiser adds q^2/12 on top.
        # var(a) = var(b) = 2 sigma^2 / N for a sine fit, so the relative sd
        # of the amplitude is sigma*sqrt(2/N)/|A|.
        def _rel(A, r_rms, q, nn):
            var = r_rms ** 2 + (q ** 2) / 12.0
            amp = max(abs(A), 1e-30)
            return float(np.sqrt(2.0 * max(var, 0.0) / max(nn, 1)) / amp)

        f_list, Z_list, s_list, n_list = [], [], [], []
        for (i0, i1, tones) in windows:
            yr, ys = ref[i0:i1], u[i0:i1]
            tw = t[i0:i1]
            n = i1 - i0
            span = float(tw[-1] - tw[0]) if n > 1 else 0.0
            use = [f for f in tones
                   if f * span >= cfg.min_cycles_per_dwell]
            if not use:
                continue
            Ar, rr, _ = csv_source.fit_phasors_multi(tw, yr, use)
            As, rs, _ = csv_source.fit_phasors_multi(tw, ys, use)
            for kf, f in enumerate(use):
                a_r, a_s = Ar[kf], As[kf]
                if not (np.isfinite(abs(a_r)) and np.isfinite(abs(a_s))) \
                        or a_s == 0:
                    continue
                # Z = U_cell / j_s with j_s = u_s / K  ->  Z = K U / u
                Z = K * a_r / a_s                       # ohm*cm2
                sig = float(np.hypot(_rel(a_r, rr, q_ref, n),
                                     _rel(a_s, rs, q_seg.get(seg, 0.0), n)))
                f_list.append(f)
                Z_list.append(Z)
                s_list.append(sig)
                n_list.append(n)

        if len(f_list) < cfg.min_points_per_spectrum:
            log.debug(f"    segment {seg}: only {len(f_list)} usable points")
            if not f_list:
                continue

        freq = np.asarray(f_list, float)
        Z = np.asarray(Z_list, complex)
        sig = np.asarray(s_list, float)
        nn = np.asarray(n_list, int)
        o = np.argsort(freq)
        freq, Z, sig, nn = freq[o], Z[o], sig[o], nn[o]

        flags: list[str] = []
        if not np.isfinite(K):
            flags.append("no_calibration_row")

        # chain response (see gamry_dta for the sign convention)
        if gain:
            Z = Z / utils.gain_at(gain, seg, freq)
            flags.append("chain_response_removed")

        # known channel-scan skew, if the acquisition program is known
        if scan and seg in slots:
            Z = apply_scan_skew(freq, Z, int(slots[seg]),
                                int(slots.get("u_cell", 0)), scan)
            flags.append("scan_skew_removed")

        # Global sign: both channels answer the same imposed current, so the
        # raw ratio can come out inverted.  Decide on the LOWEST decade,
        # weighted by SNR -- up top a few degrees of residual phase can flip
        # the sign of Re Z, down there |Z| is largest and a delay cannot
        # rotate a 1 Hz point.
        lo = freq <= max(freq.min() * 10, np.median(freq))
        if lo.sum() >= 2 and np.nansum(np.real(Z[lo]) / np.maximum(sig[lo], 1e-9)) < 0:
            Z = -Z
            flags.append("polarity_flipped")

        # |Z| outliers against the segment's own local trend.  A physical
        # spectrum is smooth in log f, so a point far from the trend is bad
        # whatever its SNR, and a weak point on the trend is credible however
        # weak.  This is the gate that removes runaways without removing the
        # top decade.
        bad = utils.zmag_outliers(freq, Z, n_mad=cfg.zmag_outlier_mad,
                                  win=cfg.zmag_outlier_win)
        # passivity, but only above the frequency where a genuinely negative
        # Re Z is physics rather than error (Schneider et al., ECS Trans.
        # 25(1) 937) -- see config.passivity_gate_min_hz
        nonpass = (freq > cfg.passivity_gate_min_hz) & (np.real(Z) <= 0)
        weak = sig > cfg.sigma_rel_max
        keep = ~(bad | nonpass | weak)
        if bad.any():
            flags.append(f"dropped_{int(bad.sum())}_zmag_outlier")
        if nonpass.any():
            flags.append(f"dropped_{int(nonpass.sum())}_non_passive")
        if weak.any():
            flags.append(f"dropped_{int(weak.sum())}_uncertain")
        if keep.sum() < cfg.min_points_per_spectrum:
            flags.append("below_min_points")

        j_dc = float(np.nanmean(u) / K) if np.isfinite(K) else float("nan")
        out[seg] = SegmentSpectrum(seg, freq[keep], Z[keep], sig[keep],
                                   20 * np.log10(1.0 / np.maximum(sig[keep], 1e-12)),
                                   nn[keep], flags, T, K, j_dc)
    return out


def from_frequency_file(m: csv_source.CsvMeasurement, cfg, log
                        ) -> dict[str, SegmentSpectrum]:
    """A CSV that already holds spectra: no phasor estimation at all.

    The only work is units.  The pipeline speaks ohm*cm2 everywhere, so a
    file in ohms has to be multiplied by the segment area -- which is the one
    place on this path where the plate selection changes a NUMBER and not
    just a picture.
    """
    areas = utils.segment_areas(cfg)
    out: dict[str, SegmentSpectrum] = {}
    for seg, (f, Z) in m.spectra.items():
        flags = [f"unit_in={m.z_unit}"]
        if m.z_unit == "mohm_cm2":
            Z = Z / 1000.0
        elif m.z_unit == "ohm":
            A = areas.get(seg)
            if A is None:
                log.warning(f"  segment {seg} not on plate {cfg.plate}; skipped")
                continue
            Z = Z * A
            flags.append(f"area_applied={A:.4f}cm2")
        sig = np.full(f.shape, 0.02)      # no uncertainty in the file
        flags.append("sigma_assumed_2pc")
        out[seg] = SegmentSpectrum(seg, f, Z, sig,
                                   np.full(f.shape, np.nan),
                                   np.full(f.shape, 0, dtype=int), flags)
    return out


# ===========================================================================
# 3. Equivalent circuit
# ===========================================================================


def z_zarc(w, R, tau, n):
    """One ZARC (R in parallel with a CPE), parameterised by (R, tau, n).

    Z = R / (1 + (j w tau)^n)

    Parameterising by tau rather than by the CPE admittance Y0 is what makes
    the fit well conditioned.  With Y0 the three parameters of an arc trade
    against each other over orders of magnitude and the optimiser walks along
    a curved valley; with tau each arc has a location, a size and a shape,
    and the two arcs of a PEMFC separate because their taus differ by
    decades.  The two forms are related by Y0 = tau^n / R.
    """
    return R / (1.0 + (1j * w * tau) ** n)


def z_model(p, freq, n_arcs=2):
    """Rs + jwL + sum of ZARCs.

    The series inductance is fitted rather than removed beforehand because on
    this rig it is degenerate with a common delay (see config.fit_common_delay
    for the algebra) -- only the sum is observable, so exactly one of the two
    may be free, and L is the one with a physical prior.
    """
    w = 2 * np.pi * np.asarray(freq, float)
    Z = p[0] + 1j * w * p[1]
    for k in range(n_arcs):
        R, tau, n = p[2 + 3 * k: 5 + 3 * k]
        Z = Z + z_zarc(w, R, tau, n)
    return Z


def ecm_start(freq, Z, n_arcs=2):
    """Starting values read off the spectrum, not guessed.

    Rs from the high-frequency real part, the total polarisation from the
    low-frequency intercept, and each arc's tau from where -Im Z peaks --
    split at the geometric mean of the band when there is only one visible
    peak, which is the honest way to start a two-arc fit on a spectrum that
    shows one and a half.
    """
    o = np.argsort(freq)
    f, Zs = freq[o], Z[o]
    Rs = max(float(np.real(Zs[-1])), 1e-6)
    Rp = max(float(np.real(Zs[0])) - Rs, 1e-5)
    im = -np.imag(Zs)
    tau_peak = 1.0 / (2 * np.pi * f[int(np.argmax(im))]) if im.size else 1.0
    p = [Rs, 1e-7]
    if n_arcs == 1:
        p += [Rp, tau_peak, 0.85]
    else:
        p += [0.5 * Rp, tau_peak / 10.0, 0.85]
        p += [0.5 * Rp, tau_peak * 10.0, 0.85]
    return p


def fit_ecm(freq, Z, sigma_rel=None, n_arcs=2, weight="sigma"):
    """Weighted complex non-linear least squares with parameter intervals.

    weight="sigma"    1/sigma per point, from the propagated CRLB.  This is
                      the right weighting when sigma is known, and it is
                      known here: the phasor fit reports it.
    weight="modulus"  1/|Z|, the conventional choice when it is not.

    Returns a dict with the parameters, their standard errors from the
    Jacobian at the optimum, chi2_nu, and the derived quantities the maps
    want (R_ohmic, R_ct, R_mt, R_pol).  A fit whose chi2_nu is far from 1 is
    reported as such rather than silently accepted: with real sigmas, chi2_nu
    IS the statement about whether the circuit describes the data.
    """
    from scipy.optimize import least_squares

    freq = np.asarray(freq, float)
    Z = np.asarray(Z, complex)
    ok = np.isfinite(freq) & np.isfinite(Z.real) & np.isfinite(Z.imag)
    freq, Z = freq[ok], Z[ok]
    if sigma_rel is not None:
        sigma_rel = np.asarray(sigma_rel, float)[ok]
    npar = 2 + 3 * n_arcs
    # Each frequency contributes TWO observations, Re and Im, and they are
    # not redundant -- that is what makes a complex fit identifiable from
    # half as many points as a real one would need.  Require four degrees of
    # freedom so chi2_nu means something.
    if 2 * freq.size < npar + 4:
        return {"ok": False,
                "reason": f"{freq.size} points ({2*freq.size} observations) "
                          f"for {npar} parameters"}

    if weight == "sigma" and sigma_rel is not None and np.all(np.isfinite(sigma_rel)):
        s = np.clip(sigma_rel, 1e-3, 1.0) * np.abs(Z)
    else:
        s = np.abs(Z)
        weight = "modulus"
    s = np.maximum(s, 1e-12)

    p0 = ecm_start(freq, Z, n_arcs)
    lo = [0.0, 0.0] + [0.0, 1e-6, 0.3] * n_arcs
    hi = [np.inf, 1e-3] + [np.inf, 1e4, 1.0] * n_arcs
    p0 = np.clip(p0, np.array(lo) + 1e-12, np.array(hi))

    def resid(p):
        d = z_model(p, freq, n_arcs) - Z
        return np.concatenate([d.real / s, d.imag / s])

    try:
        r = least_squares(resid, p0, bounds=(lo, hi), max_nfev=20000,
                          x_scale="jac")
    except Exception as exc:                       # noqa: BLE001
        return {"ok": False, "reason": f"optimiser failed: {exc}"}

    m = 2 * freq.size
    dof = max(m - npar, 1)
    chi2_nu = float(2.0 * r.cost / dof)

    # Standard errors from the Gauss-Newton approximation to the covariance.
    try:
        J = r.jac
        cov = np.linalg.pinv(J.T @ J)
        if weight == "modulus":
            cov = cov * chi2_nu      # scale by the fit when sigma is relative
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(npar, np.nan)

    p = r.x
    names = ["Rs", "L"] + sum(([f"R{k+1}", f"tau{k+1}", f"n{k+1}"]
                               for k in range(n_arcs)), [])
    arcs = sorted(((p[2 + 3 * k], p[3 + 3 * k], p[4 + 3 * k])
                   for k in range(n_arcs)), key=lambda a: a[1])
    R_ct = float(arcs[0][0]) if arcs else float("nan")
    R_mt = float(sum(a[0] for a in arcs[1:])) if len(arcs) > 1 else 0.0

    return {
        "ok": True,
        "params": {k: float(v) for k, v in zip(names, p)},
        "stderr": {k: float(v) for k, v in zip(names, se)},
        "n_arcs": n_arcs,
        "n_points": int(freq.size),
        "weight": weight,
        "chi2_nu": chi2_nu,
        "R_ohmic": float(p[0]),
        "R_ct": R_ct,
        "R_mt": R_mt,
        "R_pol": float(sum(a[0] for a in arcs)),
        "tau_peak": float(arcs[0][1]) if arcs else float("nan"),
        "verdict": ("good" if 0.2 <= chi2_nu <= 5 else
                    "underfit" if chi2_nu > 5 else "overfit/sigma too large"),
    }


def choose_n_arcs(freq, Z, sigma_rel=None) -> dict:
    """Fit one arc and two, keep the one the data supports.

    Selection by the corrected Akaike criterion rather than by chi2: adding
    an arc always lowers chi2, and a two-arc fit to a one-arc spectrum splits
    a single relaxation into two coincident halves whose individual R and tau
    then mean nothing, while the SUM still looks fine.  AICc charges the
    three extra parameters and stops that.
    """
    best = None
    for k in (1, 2):
        r = fit_ecm(freq, Z, sigma_rel, n_arcs=k)
        if not r.get("ok"):
            continue
        n = 2 * r["n_points"]
        p = 2 + 3 * k
        rss = r["chi2_nu"] * max(n - p, 1)
        aicc = n * np.log(max(rss / n, 1e-300)) + 2 * p
        if n - p - 1 > 0:
            aicc += 2 * p * (p + 1) / (n - p - 1)
        r["aicc"] = float(aicc)
        if best is None or aicc < best["aicc"]:
            best = r
    return best or {"ok": False, "reason": "no fit converged"}


# ===========================================================================
# 4. Aggregation and validation
# ===========================================================================


def cell_aggregate(spectra: dict[str, SegmentSpectrum], areas: dict,
                   a_cell: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Z_cell = A_cell / sum_s (A_s / Z_s) -- the segments are in parallel.

    Evaluated on the frequency grid shared by the most segments, and only
    where at least half of the measured area contributes, because an
    aggregate assembled from a different subset at every frequency is not a
    spectrum of anything.
    """
    if not spectra:
        return np.array([]), np.array([]), {}
    counts: dict[float, int] = {}
    for sp in spectra.values():
        for f in np.round(sp.freq, 6):
            counts[float(f)] = counts.get(float(f), 0) + 1
    if not counts:
        return np.array([]), np.array([]), {}
    n_max = max(counts.values())
    grid = np.array(sorted(f for f, c in counts.items() if c >= 0.5 * n_max))

    Y = np.zeros(grid.shape, complex)
    A_used = np.zeros(grid.shape)
    for seg, sp in spectra.items():
        A = areas.get(seg)
        if A is None or sp.freq.size < 2:
            continue
        Zi = utils.interp_complex(grid, sp.freq, sp.Z)
        ok = np.isfinite(Zi.real) & np.isfinite(Zi.imag) & (np.abs(Zi) > 0)
        Y[ok] += A / Zi[ok]
        A_used[ok] += A
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = np.where(np.abs(Y) > 0, a_cell / Y, np.nan)
    info = {"n_freq": int(grid.size),
            "area_used_cm2": float(np.nanmedian(A_used)),
            "area_total_cm2": float(a_cell)}
    return grid, Z, info


def validate(sp: SegmentSpectrum, cfg) -> dict:
    """Linear Kramers-Kronig residuals -- is the spectrum a spectrum?"""
    if sp.freq.size < 6:
        return {"ok": False, "reason": f"{sp.freq.size} points, lin-KK needs 6"}
    try:
        import eis_validation
        r = eis_validation.lin_kk(sp.freq, sp.Z, mu_crit=cfg.mu_crit,
                                  tol=cfg.kk_tol)
    except Exception as exc:                        # noqa: BLE001
        return {"ok": False, "reason": str(exc)}
    return {"ok": bool(r.get("valid")),
            "res_max": float(r.get("res_max", np.nan)),
            "res_rms": float(r.get("res_rms", np.nan)),
            "mu": r.get("mu"), "M": r.get("M")}


# ===========================================================================
# 5. Outputs
# ===========================================================================


def write_outputs(spectra, ecm, agg, cfg, log, extra: dict) -> dict:
    import r2d2_geometry as geom

    out = Path(cfg.out_dir)
    (out / "csv").mkdir(parents=True, exist_ok=True)
    areas = utils.segment_areas(cfg)

    rows = []
    for seg in sorted(spectra, key=int):
        sp = spectra[seg]
        for f, Z, s in zip(sp.freq, sp.Z, sp.sigma_rel):
            rows.append({
                "segment": int(seg), "freq_hz": f,
                "z_re_mohm_cm2": 1000 * Z.real, "z_im_mohm_cm2": 1000 * Z.imag,
                "z_mag_mohm_cm2": 1000 * abs(Z),
                "phase_deg": np.degrees(np.angle(Z)),
                "sigma_rel": s, "T_C": sp.T_C, "area_cm2": areas.get(seg),
                "flags": ";".join(sp.flags),
            })
    utils.write_table(out / "csv" / "spectra_clean.csv", rows)

    erows = []
    for seg in sorted(ecm, key=int):
        r = ecm[seg]
        if not r.get("ok"):
            erows.append({"segment": int(seg), "ok": False,
                          "reason": r.get("reason", "")})
            continue
        d = {"segment": int(seg), "ok": True, "n_arcs": r["n_arcs"],
             "n_points": r["n_points"], "chi2_nu": r["chi2_nu"],
             "verdict": r["verdict"],
             "R_ohmic_mohm_cm2": 1000 * r["R_ohmic"],
             "R_ct_mohm_cm2": 1000 * r["R_ct"],
             "R_mt_mohm_cm2": 1000 * r["R_mt"],
             "R_pol_mohm_cm2": 1000 * r["R_pol"],
             "tau_peak_s": r["tau_peak"]}
        for k, v in r["params"].items():
            d[k] = v
            d[k + "_se"] = r["stderr"].get(k, float("nan"))
        erows.append(d)
    utils.write_table(out / "csv" / "ecm_parameters.csv", erows)

    if agg[0].size:
        utils.write_table(out / "csv" / "cell_aggregate.csv", [
            {"freq_hz": f, "z_re_mohm_cm2": 1000 * Z.real,
             "z_im_mohm_cm2": 1000 * Z.imag}
            for f, Z in zip(agg[0], agg[1])])

    maps = {}
    if cfg.write_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for param, key, unit in (("R_ohmic", "R_ohmic", "mΩ·cm²"),
                                 ("R_ct", "R_ct", "mΩ·cm²"),
                                 ("R_mt", "R_mt", "mΩ·cm²"),
                                 ("R_pol", "R_pol", "mΩ·cm²")):
            vals = {s: 1000 * r[key] for s, r in ecm.items()
                    if r.get("ok") and np.isfinite(r[key])}
            if len(vals) < 3:
                continue
            p = geom.plot_map(out / f"map_{param}.png", vals,
                              label=f"{param}  [{unit}]",
                              title=f"{geom.ACTIVE_PLATE.title} — {param} "
                                    f"({len(vals)} segments, CSV source)")
            maps[param] = str(p)

        # Nyquist, all segments plus the cell aggregate
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        for seg in sorted(spectra, key=int):
            sp = spectra[seg]
            ax.plot(1000 * sp.Z.real, -1000 * sp.Z.imag, lw=0.8, alpha=0.55)
        if agg[0].size:
            ax.plot(1000 * agg[1].real, -1000 * agg[1].imag, "k-", lw=2.4,
                    label="cell aggregate")
            ax.legend()
        ax.set(xlabel="Z'  [mΩ·cm²]",
               ylabel="-Z''  [mΩ·cm²]",
               title=f"{geom.ACTIVE_PLATE.title} — local EIS from CSV "
                     f"({len(spectra)} segments)")

        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        fig.savefig(out / "nyquist.png", dpi=150)
        plt.close(fig)
        maps["nyquist"] = str(out / "nyquist.png")

    manifest = {"config": cfg.to_dict(), "plate": geom.ACTIVE_PLATE.key,
                "source": "csv", "figures": maps,
                "n_segments": len(spectra),
                "n_ecm_ok": sum(1 for r in ecm.values() if r.get("ok")),
                **extra}
    utils.write_json(out / "run_manifest.json", manifest)
    return manifest


# ===========================================================================
# 6. Entry point
# ===========================================================================


def run_csv(cfg, stop_after: str = "gold") -> dict:
    """The CSV path, end to end."""
    import eis_local
    import r2d2_geometry as geom

    log = utils.get_logger(cfg.verbose)
    t0 = time.time()
    plate = geom.use_plate(cfg.plate)
    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    utils.banner("R2-D2 LOCAL EIS — CSV SOURCE", log)
    log.info(f"  plate : {plate.title}")
    log.info(f"  in    : {cfg.csv_path}")
    log.info(f"  out   : {cfg.out_dir}")

    if not cfg.csv_path:
        raise ValueError("cfg.csv_path is not set; the CSV path needs a file "
                         "or a folder to read.")

    m = csv_source.read(cfg.csv_path, cfg.csv_dialect)
    log.info(f"  read  : {m.dialect} / {m.kind}, "
             f"{len(m.segments)} segments")
    for k, v in m.summary().items():
        log.debug(f"          {k} = {v}")

    cal = eis_local.PlateCalibration.load(cfg.curr_cal, cfg.temp_cal)

    # --- temperatures ------------------------------------------------------
    T_seg: dict[str, float] = {}
    if m.kind == "time" and m.temps and cal.temp_c0:
        sensor_T = {}
        for name, volts in m.temps.items():
            try:
                sensor_T[name] = float(cal.temperature(
                    name, float(np.nanmean(volts))))
            except KeyError:
                continue
        if sensor_T:
            T_seg = geom.segment_temperatures(sensor_T)
            log.info("  temps : " + ", ".join(
                f"{k}={v:.2f} C" for k, v in sorted(sensor_T.items())))
    if not T_seg:
        import config as _c
        T_seg = {s: _c.T_FALLBACK_C for s in geom.SEGMENTS}
        # Only a warning on the time-domain path: that is where the Abgleich
        # K(T) is evaluated and a wrong T becomes a wrong Z.  A file that is
        # already a spectrum never touches K.
        if m.kind == "time":
            log.warning(f"  temps : none usable, falling back to "
                        f"{_c.T_FALLBACK_C} C for every segment. Copper TCR "
                        f"is 0.42 %/K, so a 10 K error is 4.2 % on every R.")

    # --- spectra -----------------------------------------------------------
    if m.kind == "frequency":
        sched = {"mode": "from_file", "tones": [], "windows": []}
        spectra = from_frequency_file(m, cfg, log)
    else:
        if not cal.has_current_cal:
            raise ValueError(
                "a time-domain CSV needs cfg.curr_cal: without the Abgleich "
                "the shunt voltage cannot be turned into a current density "
                "and Z has no absolute scale.")
        sched = detect_tones(m, cfg, log)
        log.info(f"  tones : {sched['mode']}, {len(sched['tones'])} "
                 f"frequencies "
                 f"{min(sched['tones']):.4g} .. {max(sched['tones']):.4g} Hz"
                 if sched["tones"] else "  tones : none")
        spectra = segment_spectra(m, sched, cal, T_seg, cfg, log)

    spectra = {k: v for k, v in spectra.items()
               if k not in cfg.exclude_segments and v.freq.size >= 3}
    log.info(f"  Z     : {len(spectra)} segments with a spectrum")
    if stop_after == "bronze":
        return {"stage": "bronze", "n_segments": len(spectra),
                "schedule": sched}

    # --- validation --------------------------------------------------------
    kk = {seg: validate(sp, cfg) for seg, sp in spectra.items()}
    n_kk = sum(1 for v in kk.values() if v.get("ok"))
    log.info(f"  KK    : {n_kk}/{len(kk)} segments inside "
             f"{100*cfg.kk_tol:.0f} % residual")
    if stop_after == "silver":
        return {"stage": "silver", "n_segments": len(spectra), "kk": kk}

    # --- ECM + aggregate ---------------------------------------------------
    ecm = {}
    for seg, sp in spectra.items():
        ecm[seg] = choose_n_arcs(sp.freq, sp.Z, sp.sigma_rel)
    n_ok = sum(1 for r in ecm.values() if r.get("ok"))
    if n_ok:
        rs = np.array([1000 * r["R_ohmic"] for r in ecm.values()
                       if r.get("ok")])
        log.info(f"  ECM   : {n_ok}/{len(ecm)} fitted, "
                 f"R_ohmic {np.median(rs):.1f} +/- {np.std(rs):.1f} "
                 f"mΩ·cm²")

    areas = utils.segment_areas(cfg)
    agg = cell_aggregate(spectra, areas, geom.A_CELL_CM2)

    manifest = write_outputs(
        spectra, ecm, agg, cfg, log,
        extra={"schedule": {k: v for k, v in sched.items() if k != "windows"},
               "reader": m.summary(),
               "kk_pass": n_kk, "kk_total": len(kk),
               "cell_aggregate": agg[2],
               "elapsed_s": round(time.time() - t0, 2)})
    utils.banner("DONE", log)
    log.info(f"  {time.time()-t0:.1f} s -> {cfg.out_dir}")
    return manifest


if __name__ == "__main__":
    import argparse
    from config import DEFAULT

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("csv")
    p.add_argument("--plate", choices=["gen1", "gen2"], default="gen1")
    p.add_argument("--dialect", default="auto")
    p.add_argument("--curr-cal", default=None)
    p.add_argument("--temp-cal", default=None)
    p.add_argument("--gain", default=None)
    p.add_argument("--out", default="./results_csv")
    a = p.parse_args()

    cfg = DEFAULT.replace(
        source_format="csv", csv_path=Path(a.csv), csv_dialect=a.dialect,
        plate=a.plate, out_dir=Path(a.out),
        curr_cal=Path(a.curr_cal) if a.curr_cal else None,
        temp_cal=Path(a.temp_cal) if a.temp_cal else None,
        gain_file=Path(a.gain) if a.gain else None)
    print(json.dumps(run_csv(cfg), indent=2, default=str)[:4000])
