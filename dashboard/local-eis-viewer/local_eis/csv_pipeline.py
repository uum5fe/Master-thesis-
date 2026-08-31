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
    f_hi = cfg.f_hi(fs)
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
# 1b. The R2-D2 logger: one file per frequency, one scan per row
# ===========================================================================
# A point file gives 80 channels sampled one after the other inside each row,
# with the acquisition instant of every channel printed in the `timeshifts`
# header row.  Two things follow, and the second is the one that bites.
#
# 1. THE ROWS ARE NOT SIMULTANEOUS SAMPLES.  Segment 1 is taken at 0 us and
#    the cell-voltage taps at 79-82 us.  The impedance is the ratio of those
#    two channels, so 80 us of skew sits directly in it: 29 degrees at 1 kHz,
#    a full quadrant near the top of the band.  Correcting it needs no fit --
#    the delays are printed in the file.  `deskew` applies them.
#
# 2. THE SCAN IS ALSO A MEASUREMENT OF THE EXCITATION FREQUENCY, and it does
#    not have to agree with the tone the record appears to contain.  Across
#    one row the 80 channels sample the analogue waveform at 1.1 us intervals,
#    so the phase difference between channel k and channel 0 is
#    2*pi*f_analogue*tau_k -- at the TRUE frequency of the signal, whatever
#    the row-rate DFT reports.  Compare the two and undersampling becomes
#    visible instead of invisible.
#
#    On the delivered p1.csv the record contains a clean tone at 923.08 Hz
#    against fs = 11001.10 Hz.  Taken at face value, 923 Hz would rotate the
#    phase by 0.33 deg/us and produce 29 degrees across the 87 us scan.  The
#    measured rotation is ~307 degrees, an order of magnitude more, and
#    de-skewing at 923 Hz leaves the 72 segment phasors scattered (resultant
#    R = 0.14, i.e. no common phase at all) while de-skewing at fs - 923 Hz
#    collapses them (R = 0.84).  The analogue tone is near 10 kHz; the 923 Hz
#    in the record is its alias, folded by a sampler running at 11 kHz with
#    nothing in front of it that stops 10 kHz.
#
#    That is a property of the rig, not of this code, and it is reported
#    rather than silently worked around: `nyquist_zone` returns the frequency
#    the scan implies, the resultant at each candidate, and a verdict.  Supply
#    the sweep's own frequency list through `cfg.csv_tones` and it is used and
#    checked instead of inferred -- the scan resolves f only to about
#    1/scan_span = 11 kHz, which is the same ambiguity as the aliasing itself,
#    so it can identify the Nyquist zone but not replace knowing it.


def deskew(A: complex, tau: float, freq: float, conjugated: bool = False
           ) -> complex:
    """Move one channel's phasor back to the row's nominal instant.

    A channel sampled tau seconds late carries an extra phase +w*tau, so the
    correction is a multiplication by exp(-j w tau).  `freq` must be the
    ANALOGUE frequency: for an undersampled point that is the true tone, not
    the alias the DFT sees.

    When the point is undersampled into an odd Nyquist zone the alias is also
    conjugated -- sampling f = fs - f_a produces cos(w_a t - phi) rather than
    cos(w_a t + phi) -- so the measured phasor is conj(A_true)*exp(-j w tau)
    and recovering A_true takes a conjugation as well.
    """
    if conjugated:
        return np.conj(A) * np.exp(-1j * 2 * np.pi * freq * tau)
    return A * np.exp(-1j * 2 * np.pi * freq * tau)


def nyquist_zone(phasors: np.ndarray, taus: np.ndarray, f_alias: float,
                 fs: float, max_zone: int = 4) -> dict:
    """Which analogue frequency does the channel scan say this point was at?

    All segments of one plate answer the same imposed current, so once the
    scan skew is removed their phasors must share a phase to within the real
    spread of the local impedance.  The resultant length

        R(f) = | sum_k  u_k exp(-j 2 pi f tau_k) | / N ,   u_k = A_k/|A_k|

    is therefore a direct score for a candidate analogue frequency.  Every
    candidate consistent with the observed alias is tried -- f_alias itself,
    then m*fs +- f_alias with the conjugation each zone implies -- and the
    best is returned with the runner-up, because the scan spans less than one
    sample period and so resolves f only to about fs.  That is enough to tell
    a baseband point from an undersampled one; it is not enough to pick
    between neighbouring zones on its own.
    """
    u = phasors / np.maximum(np.abs(phasors), 1e-300)

    def score(f, conj_):
        v = np.conj(u) if conj_ else u
        return float(abs(np.sum(v * np.exp(-1j * 2 * np.pi * f * taus))) / u.size)

    cands = [(float(f_alias), False, 0)]
    for m in range(1, max_zone + 1):
        cands.append((m * fs - f_alias, True, m))     # odd zone: conjugated
        cands.append((m * fs + f_alias, False, m))    # even zone: not
    scored = sorted(((score(f, c), f, c, m) for f, c, m in cands),
                    reverse=True)
    best_R, best_f, best_c, best_m = scored[0]
    base_R = score(float(f_alias), False)

    return {
        "f_alias_hz": float(f_alias),
        "f_analogue_hz": best_f,
        "conjugated": bool(best_c),
        "zone": int(best_m),
        "R": best_R,
        "R_baseband": base_R,
        "runner_up_hz": scored[1][1] if len(scored) > 1 else float("nan"),
        "runner_up_R": scored[1][0] if len(scored) > 1 else float("nan"),
        "undersampled": bool(best_m > 0 and best_R > base_R + 0.15),
        "candidates": [{"f_hz": round(f, 4), "zone": m, "conjugated": c,
                        "R": round(r, 4)} for r, f, c, m in scored[:5]],
    }


def r2d2_point(m, cfg, log, f_true: float | None = None) -> dict:
    """One point file -> one frequency and one phasor per channel.

    Every channel is fitted separately at the alias frequency and then moved
    back to the row instant with its own printed delay.  The cell voltage is
    formed as a DIFFERENCE OF PHASORS afterwards, never as a difference of
    samples: uc1 and uc2 are themselves 1.1 us apart, and subtracting them
    row-wise would bake that into the reference before anything could remove
    it.
    """
    t = m.t
    fs = m.fs_nominal
    n = t.size

    seg_keys = m.segments
    aux_keys = sorted(m.aux)

    # THE TONE IS FOUND ON THE SEGMENTS, NOT ON ONE CHANNEL.
    # The obvious choice -- peak-pick the loudest cell-voltage tap -- is
    # fragile, and the delivered file shows why: uc1 carries a 3488 Hz
    # component larger than the excitation at 923 Hz, and on a short record
    # the picker locks onto it.  The excitation is the one thing that drives
    # all 72 segments coherently, so averaging the amplitude spectra over the
    # segments reinforces it and averages down anything that is not common.
    # On the delivered file that turns a wrong answer into the right one and
    # costs one FFT per channel.
    f_lo = max(cfg.f_min_hz, 3.0 / max(t[-1] - t[0], 1e-9))
    f_hi = cfg.f_hi(fs)

    n_fft = t.size
    win = np.hanning(n_fft)
    wsum = float(np.sum(win))
    acc = None
    for k in seg_keys:
        y = m.u_seg[k]
        Y = np.abs(np.fft.rfft((y - y.mean()) * win)) * 2.0 / wsum
        acc = Y / max(float(np.std(y)), 1e-30) if acc is None \
            else acc + Y / max(float(np.std(y)), 1e-30)
    if acc is None:
        return {"ok": False, "reason": "no segment channels"}
    mean_spec = acc / len(seg_keys)

    fgrid = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    band = (fgrid >= f_lo) & (fgrid <= f_hi)
    if band.sum() < 8:
        return {"ok": False, "reason": f"band {f_lo:.3g}..{f_hi:.3g} Hz holds "
                                       f"fewer than 8 bins"}
    f0 = float(fgrid[band][int(np.argmax(mean_spec[band]))])

    # IS THERE AN EXCITATION AT ALL?  A point file can be all lead-in -- the
    # delivered one carries 0.25 s of it -- and a persistent instrument
    # artefact sits at 999 Hz at about 6x the noise floor.  The excitation is
    # 380x.  Refusing a point with nothing in it is better than reporting an
    # impedance measured against a switching artefact.
    peak_ratio = float(np.max(mean_spec[band]) / (np.median(mean_spec[band]) + 1e-30))
    if peak_ratio < 10.0:
        return {"ok": False,
                "reason": f"no excitation: the strongest common tone in the "
                          f"segments is only {peak_ratio:.1f}x the noise "
                          f"floor ({f0:.1f} Hz). A point file that is all "
                          f"lead-in looks like this."}

    # The phase reference is still a cell-voltage tap -- the impedance is
    # measured against the cell, not against another segment.
    ref_key = max(aux_keys, key=lambda k: float(np.std(m.aux[k]))) if aux_keys \
        else max(seg_keys, key=lambda k: float(np.std(m.u_seg[k])))
    ref = m.aux[ref_key] if ref_key in m.aux else m.u_seg[ref_key]

    f_alias, _A, _r, _snr = utils.fit4(ref, fs, f0)
    if not (0.98 * f0 <= f_alias <= 1.02 * f0):
        f_alias = f0            # the refit ran off to a neighbouring feature

    # THE POINT FILE IS A BURST, NOT A CONTINUOUS RECORDING.
    # The delivered p1.csv runs 1.765 s and the excitation is only present
    # from 0.25 s to 1.35 s -- a quarter of a second of lead-in, four tenths
    # of lead-out, and in both of them a persistent 999 Hz artefact that is
    # 6x the noise floor while the excitation is 380x.  Fitting the whole
    # record inflates the residual, and therefore sigma, by the silent
    # fraction; worse, a file cut inside the lead-in would be "analysed"
    # against the artefact.  So find the burst first.  `eis_local` already
    # locates a dwell from a demodulated envelope and then shrinks it while
    # the SNR improves -- the same job on the same kind of signal, so it is
    # reused rather than rewritten.
    #
    # The envelope is taken on the COMPOSITE of all segments, normalised
    # channel by channel: the excitation is the one thing common to all 72,
    # so the composite has ~sqrt(72) times the SNR of any single channel and
    # the burst edges are unambiguous in it.
    comp = np.zeros(t.size)
    for k in seg_keys:
        y = m.u_seg[k]
        comp += (y - y.mean()) / max(float(np.std(y)), 1e-30)
    comp /= len(seg_keys)

    i0, i1 = 0, t.size
    try:
        import eis_local
        env, wlen = eis_local.demod_envelope(comp, fs, f_alias)
        # Compare the peak against a LOW PERCENTILE, not the median: on the
        # delivered file the burst is 62 % of the record, so the median of
        # the envelope sits inside the burst and a median test would conclude
        # there is nothing to window to.
        floor = float(np.percentile(env, 10)) if env.size > 4 else 0.0
        if env.size > 4 and float(np.max(env)) > 3.0 * floor:
            centre = int(np.argmax(env)) + wlen // 2
            i0, i1 = eis_local.dwell_window(comp, fs, f_alias, centre)
            i0, i1 = eis_local.polish_window(comp, fs, f_alias, i0, i1)
    except Exception:                                # noqa: BLE001
        i0, i1 = 0, t.size
    if (i1 - i0) < max(32, cfg.min_cycles_per_dwell * fs / max(f_alias, 1e-9)):
        i0, i1 = 0, t.size                           # burst not found; use all
    burst = {"i0": int(i0), "i1": int(i1),
             "found": bool(i1 - i0 < t.size),
             "peak_over_floor": round(peak_ratio, 1),
             "fraction": round((i1 - i0) / t.size, 4),
             "t0_s": round(float(t[i0]), 4),
             "t1_s": round(float(t[min(i1, t.size) - 1]), 4)}

    # Re-estimate the frequency inside the burst, where it is not diluted by
    # the silent stretches.
    f_b, _A, _r, _snr = utils.fit4(comp[i0:i1], fs, f_alias)
    if 0.98 * f_alias <= f_b <= 1.02 * f_alias:
        f_alias = f_b

    A_raw, r_rms, taus = {}, {}, {}
    for key, arr in list(m.u_seg.items()) + [(k, m.aux[k]) for k in aux_keys]:
        col = f"s{key}" if key in m.u_seg else key
        a, rr, _s = utils.fit3(arr[i0:i1], fs, f_alias)
        A_raw[key], r_rms[key] = a, rr
        taus[key] = m.timeshifts.get(col, 0.0)

    seg_A = np.array([A_raw[k] for k in seg_keys])
    seg_tau = np.array([taus[k] for k in seg_keys])
    zone = nyquist_zone(seg_A, seg_tau, f_alias, fs)

    if f_true is not None:
        # An operator-supplied frequency wins, but it is checked: if the scan
        # disagrees, the list and the file are not describing the same point.
        f_ana = float(f_true)
        k = int(round(f_ana / fs))
        conj_ = bool(k > 0 and (f_ana < k * fs))
        zone["f_given_hz"] = f_ana
        zone["agrees_with_scan"] = bool(
            abs(f_ana - zone["f_analogue_hz"]) < 0.25 * fs)
    else:
        f_ana = zone["f_analogue_hz"]
        conj_ = zone["conjugated"]

    A = {k: deskew(A_raw[k], taus[k], f_ana, conj_) for k in A_raw}

    # Cell voltage: uc1,uc3 sit near 0 V and uc2,uc4 near the cell potential,
    # so (uc2-uc1) and (uc4-uc3) are two independent measurements of the same
    # cell.  Both are kept: their disagreement is a lead-placement diagnostic
    # that no single-tap reading can show.
    pairs = []
    for lo, hi in (("uc1", "uc2"), ("uc3", "uc4")):
        if lo in A and hi in A:
            pairs.append(A[hi] - A[lo])
    if not pairs:
        pairs = [A[k] for k in aux_keys[:1]] or [None]
    if pairs[0] is None:
        return {"ok": False, "reason": "no cell-voltage channel"}
    U = np.mean(pairs, axis=0)

    spread = (abs(abs(pairs[1]) / abs(pairs[0]) - 1.0)
              if len(pairs) == 2 and abs(pairs[0]) > 0 else 0.0)

    return {"ok": True, "f_hz": f_ana, "f_alias_hz": f_alias, "fs_hz": fs,
            "n": int(i1 - i0), "n_rows": n, "burst": burst,
            "U": U, "A": A, "r_rms": r_rms, "zone": zone,
            "uc_pair_spread": float(spread), "ref_channel": ref_key,
            "conjugated": conj_}


def sweep_grid_check(freqs) -> dict:
    """Do the recovered frequencies form a real sweep?

    This is the check that turns a per-point inference into a sweep-level
    fact, and it is the strongest evidence available that an undersampled
    point was unfolded into the right Nyquist zone.

    An EIS sweep is generated on a logarithmic grid -- almost always an
    integer number of points per decade.  So the recovered ANALOGUE
    frequencies must form a geometric progression, while the aliases they
    were folded from must not: folding is `|f - k*fs|`, which scrambles a
    geometric sequence into an arbitrary one.  Fit the points per decade,
    report the residual, and the two hypotheses separate without any appeal
    to the phase ramp.

    Measured on the delivered p1/p2/p3: the recovered 10078.02 / 8015.53 /
    6328.05 Hz sit on a 10-points-per-decade grid and match the Gamry
    PTSPERDEC=10 frequencies read out of the Abgleich bode files to a
    CONSTANT -11.5 ppm -- which is the sample-rate estimate, not a
    disagreement.  The aliases 923.09 / 2985.57 / 4673.05 have consecutive
    ratios of 3.23 and 1.57 and are not a sweep at all.
    """
    f = np.sort(np.asarray([x for x in freqs if x and np.isfinite(x)], float))
    out = {"n": int(f.size)}
    if f.size < 3:
        out["ok"] = None
        out["reason"] = "need at least three points to test a progression"
        return out

    # Fit log10(f) against the point index; the slope is 1/ppd.
    k = np.arange(f.size, dtype=float)
    slope, icept = np.polyfit(k, np.log10(f), 1)
    ppd = 1.0 / slope if slope else np.inf
    resid_ppm = 1e6 * (np.log(10.0)) * (np.log10(f) - (slope * k + icept))
    out.update(
        points_per_decade=round(float(abs(ppd)), 3),
        ppd_nearest_integer=int(round(abs(ppd))),
        residual_ppm_max=round(float(np.max(np.abs(resid_ppm))), 1),
        f_min_hz=round(float(f.min()), 4),
        f_max_hz=round(float(f.max()), 4),
    )
    # A real generated grid lands within a few hundred ppm; a scrambled set
    # of aliases misses by percent.
    out["ok"] = bool(out["residual_ppm_max"] < 5000
                     and abs(abs(ppd) - round(abs(ppd))) < 0.15)
    return out


def r2d2_sweep_spectra(points: list, cfg, log) -> tuple[dict, dict]:
    """Assemble the per-segment spectra from a folder of point files."""
    tones = list(cfg.csv_tones) if cfg.csv_tones else []
    if tones and len(tones) != len(points):
        log.warning(f"  csv_tones has {len(tones)} entries for "
                    f"{len(points)} point files; the list is used in file "
                    f"order and the extra entries are ignored.")

    per_seg: dict[str, list] = {}
    report = {"points": [], "n_undersampled": 0, "n_failed": 0}

    for i, m in enumerate(points):
        f_given = tones[i] if i < len(tones) else None
        res = r2d2_point(m, cfg, log, f_true=f_given)
        name = m.meta.get("point", f"p{i+1}")
        if not res.get("ok"):
            log.warning(f"  {name}: {res.get('reason')}")
            report["n_failed"] += 1
            report["points"].append({"point": name, "ok": False,
                                     "reason": res.get("reason")})
            continue

        z = res["zone"]
        if z["undersampled"] and f_given is None:
            report["n_undersampled"] += 1
        log.info(f"  {name}: alias {res['f_alias_hz']:9.3f} Hz -> analogue "
                 f"{res['f_hz']:10.3f} Hz  (zone {z['zone']}, R={z['R']:.3f} "
                 f"vs baseband {z['R_baseband']:.3f})"
                 + ("  UNDERSAMPLED" if z["undersampled"] else ""))

        U = res["U"]
        f = res["f_hz"]
        for seg in m.segments:
            a = res["A"][seg]
            if a == 0 or not np.isfinite(abs(a)):
                continue
            # m.seg_unit == "A/cm2": the logger already applied the Abgleich,
            # so the ratio is directly an area-specific impedance and K must
            # NOT be applied again.
            Z = U / a
            n = res["n"]
            sig = float(np.hypot(
                np.sqrt(2.0 / n) * res["r_rms"].get("uc2", 0.0) / max(abs(U), 1e-30),
                np.sqrt(2.0 / n) * res["r_rms"][seg] / max(abs(a), 1e-30)))
            per_seg.setdefault(seg, []).append((f, Z, sig, n))

        report["points"].append({
            "point": name, "ok": True, "f_alias_hz": round(res["f_alias_hz"], 4),
            "f_analogue_hz": round(res["f_hz"], 4), "zone": z["zone"],
            "R": round(z["R"], 4), "R_baseband": round(z["R_baseband"], 4),
            "undersampled": z["undersampled"],
            "uc_pair_spread": round(res["uc_pair_spread"], 4),
            "n_samples": res["n"], "fs_hz": round(res["fs_hz"], 4)})

    # DC operating point and plate temperature, from the first point file --
    # the sweep is one steady state, so one file is enough and averaging over
    # all of them would hide a drift rather than show it.
    m0 = points[0]
    j_dc = {k: float(np.mean(v)) for k, v in m0.u_seg.items()}
    T_seg: dict[str, float] = {}
    if m0.temps and m0.temp_unit == "degC":
        import r2d2_geometry as _g
        sensor_T = {k: float(np.mean(v)) for k, v in m0.temps.items()}
        T_seg = _g.segment_temperatures(sensor_T)
        report["temps_C"] = {k: round(v, 3) for k, v in sensor_T.items()}
    areas0 = utils.segment_areas(cfg)
    report["dc"] = {
        "j_median_A_cm2": round(float(np.median(list(j_dc.values()))), 4),
        "j_min_A_cm2": round(float(min(j_dc.values())), 4),
        "j_max_A_cm2": round(float(max(j_dc.values())), 4),
        "I_closure_A": round(float(sum(j_dc[k] * areas0[k]
                                       for k in j_dc if k in areas0)), 2),
        "note": "sum(j*A) over the measured segments, on the selected plate",
    }
    spreads = [p["uc_pair_spread"] for p in report["points"]
               if p.get("ok") and p.get("uc_pair_spread") is not None]
    if spreads and max(spreads) > 0.10:
        report["uc_pair_warning"] = round(float(max(spreads)), 4)
        log.warning(
            f"  UC    : the two cell-voltage pairs (uc2-uc1 and uc4-uc3) "
            f"disagree by up to {100*max(spreads):.0f} % in amplitude. They "
            f"measure the same cell, so this is a lead-placement or contact "
            f"difference, and it sets a floor on how well any single "
            f"reference defines Z. Their mean is used.")

    # Sweep-level cross-check on the recovered frequencies.
    got = [p["f_analogue_hz"] for p in report["points"] if p.get("ok")]
    alias = [p["f_alias_hz"] for p in report["points"] if p.get("ok")]
    report["grid"] = {"analogue": sweep_grid_check(got),
                      "alias": sweep_grid_check(alias)}
    ga, gl = report["grid"]["analogue"], report["grid"]["alias"]
    if ga.get("ok") is not None:
        log.info(f"  grid  : recovered frequencies sit on a "
                 f"{ga['points_per_decade']:.2f} points/decade grid "
                 f"(residual {ga['residual_ppm_max']:.0f} ppm) -> "
                 f"{'a real sweep' if ga['ok'] else 'NOT a clean progression'}")
        if ga.get("ok") and not gl.get("ok"):
            log.info(f"          the aliases in the files do not "
                     f"({gl['points_per_decade']:.2f} p/dec, "
                     f"{gl['residual_ppm_max']:.0f} ppm) — independent "
                     f"confirmation that the unfolding is correct")

    spectra: dict[str, SegmentSpectrum] = {}
    for seg, rows in per_seg.items():
        rows.sort(key=lambda r: r[0])
        f = np.array([r[0] for r in rows])
        Z = np.array([r[1] for r in rows])
        s = np.array([r[2] for r in rows])
        nn = np.array([r[3] for r in rows])
        flags = ["seg_unit=A/cm2", "timeshift_deskew"]
        if any(p.get("undersampled") for p in report["points"] if p.get("ok")):
            flags.append("contains_undersampled_points")
        # The plate answers one imposed current, so a global sign flip is
        # possible; decide it on the lowest decade where |Z| is largest and a
        # residual delay cannot rotate the point.
        lo = f <= max(f.min() * 10, np.median(f))
        if lo.sum() >= 1 and np.nansum(np.real(Z[lo]) / np.maximum(s[lo], 1e-9)) < 0:
            Z = -Z
            flags.append("polarity_flipped")
        spectra[seg] = SegmentSpectrum(
            seg, f, Z, s, 20 * np.log10(1.0 / np.maximum(s, 1e-12)), nn, flags,
            T_C=float(T_seg.get(seg, np.nan)), K=float("nan"),
            j_dc=float(j_dc.get(seg, np.nan)))
    return spectra, report


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
    # THE SAME LAYOUT THE FAMOS PATH WRITES, not a parallel one.
    #
    # This module's own docstring promises "the same products as
    # bronze/silver/gold", and it did produce them -- into `csv/`, where
    # nothing looks for them. The viewer decides a folder is a result by
    # finding `silver/spectra_clean.csv` or `gold/plate_summary.csv` (see
    # app/data/loaders.py detect_layout), so a CSV run completed, wrote every
    # number correctly, and was then invisible: the campaign never appeared
    # in the results source at all. Same products means the same paths.
    #
    # `csv/` is still written, because it is what this path has always
    # written and something outside this repo may read it.
    (out / "csv").mkdir(parents=True, exist_ok=True)
    (out / "silver").mkdir(parents=True, exist_ok=True)
    (out / "gold").mkdir(parents=True, exist_ok=True)
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
    utils.write_table(out / "silver" / "spectra_clean.csv", rows)

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

    # gold/plate_summary.csv is the per-segment scalar table every map on the
    # plate-map and coverage tabs reads. The FAMOS path builds it in gold.py;
    # here the same quantities come straight off the ECM fit, so it is
    # assembled rather than recomputed. Units match gold's: ohm.cm2, not
    # mohm.cm2, because that is what the reader expects to divide by.
    summary = []
    for seg in sorted(spectra, key=int):
        r = ecm.get(seg, {})
        ok = bool(r.get("ok"))
        sp = spectra[seg]
        summary.append({
            "segment": int(seg),
            "area_cm2": areas.get(seg, float("nan")),
            "R_ohmic": r["R_ohmic"] if ok else float("nan"),
            "R_ct": r["R_ct"] if ok else float("nan"),
            "R_mt": r["R_mt"] if ok else float("nan"),
            "R_pol": r["R_pol"] if ok else float("nan"),
            "tau_peak": r["tau_peak"] if ok else float("nan"),
            "chi2_nu": r.get("chi2_nu", float("nan")),
            "j_dc": getattr(sp, "j_dc", float("nan")),
            "T_C": sp.T_C,
            "n_points": len(sp.freq),
            "measured": 1,
            "inferred": 0,
            "tier": r.get("verdict", "") if ok else "",
            "flags": ";".join(sp.flags),
        })
    utils.write_table(out / "gold" / "plate_summary.csv", summary)

    if agg[0].size:
        cell_rows = [{"freq_hz": f, "z_re_mohm_cm2": 1000 * Z.real,
                      "z_im_mohm_cm2": 1000 * Z.imag}
                     for f, Z in zip(agg[0], agg[1])]
        utils.write_table(out / "csv" / "cell_aggregate.csv", cell_rows)
        # Also where the whole-cell check and the viewer look for it.
        utils.write_table(out / "silver" / "cell_aggregate.csv", cell_rows)

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

        # A sweep too short to fit anything still says something: |Z| and the
        # phase at the one measured frequency are a plate map in their own
        # right, and they are what a single point file is for.
        n_f = int(np.median([sp.freq.size for sp in spectra.values()])) \
            if spectra else 0
        if n_f < 6 and spectra:
            f_ref = float(np.median([sp.freq[0] for sp in spectra.values()]))
            zmag = {s: 1000 * float(np.abs(sp.Z[0])) for s, sp in spectra.items()}
            zphi = {s: float(np.degrees(np.angle(sp.Z[0])))
                    for s, sp in spectra.items()}
            for nm, val, unit in (("Z_mag", zmag, "mΩ·cm²"),
                                  ("Z_phase", zphi, "°")):
                p = geom.plot_map(out / f"map_{nm}_{f_ref:.0f}Hz.png", val,
                                  label=f"|Z|  [{unit}]" if nm == "Z_mag"
                                        else f"arg Z  [{unit}]",
                                  title=f"{geom.ACTIVE_PLATE.title} — {nm} at "
                                        f"{f_ref:.1f} Hz ({len(val)} segments)")
                maps[f"{nm}_at_f"] = str(p)
            jdc = {s: sp.j_dc for s, sp in spectra.items()
                   if np.isfinite(sp.j_dc)}
            if len(jdc) > 3:
                maps["j_dc"] = str(geom.plot_map(
                    out / "map_j_dc.png", jdc, label="j  [A/cm²]",
                    title=f"{geom.ACTIVE_PLATE.title} — DC current density"))

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

    dialect = (csv_source.detect_dialect(cfg.csv_path)
               if cfg.csv_dialect in ("auto", "", None) else cfg.csv_dialect)

    # ---- the R2-D2 logger: a folder of one-frequency point files ----------
    if dialect in ("r2d2", "r2d2_sweep"):
        points = (csv_source.read_r2d2_sweep(cfg.csv_path)
                  if dialect == "r2d2_sweep"
                  else [csv_source.read_r2d2(cfg.csv_path)])
        md = points[0].meta.get("metadata", {})
        log.info(f"  read  : R2-D2 logger, {len(points)} frequency point(s), "
                 f"{len(points[0].segments)} segments")
        if md:
            log.info("  file  : " + ", ".join(
                f"{k}={v}" for k, v in md.items() if k != "notes"))
            if md.get("coefficients"):
                log.info(f"  cal   : the logger already applied coefficient "
                         f"set '{md['coefficients']}' — the s columns are a "
                         f"current density and curr.csv is NOT applied again")
        sp0 = points[0].summary()
        log.info(f"  scan  : {sp0.get('n_channels', '?')} channels over "
                 f"{sp0.get('scan_span_us', '?')} us "
                 f"({100*sp0.get('scan_fraction_of_sample', 0):.0f} % of a "
                 f"{1e6/sp0['fs_hz']:.2f} us sample period), step "
                 f"{sp0.get('scan_step_us', '?')} us")

        spectra, r2d2_report = r2d2_sweep_spectra(points, cfg, log)
        if r2d2_report["n_undersampled"]:
            log.warning(
                f"  ALIAS : {r2d2_report['n_undersampled']} of "
                f"{len(points)} points are undersampled — the tone in the "
                f"record is a fold of a higher analogue frequency. The "
                f"channel scan measures the true one and it has been used, "
                f"but the scan resolves f only to about fs, so pass the "
                f"sweep's own frequency list in cfg.csv_tones to remove the "
                f"inference.")
        return _finish_csv(spectra, {"mode": "r2d2_sweep",
                                     "tones": sorted({round(p["f_analogue_hz"], 4)
                                                      for p in r2d2_report["points"]
                                                      if p.get("ok")}),
                                     "windows": []},
                           points[0], cfg, log, t0, stop_after,
                           extra={"r2d2": r2d2_report, "metadata": md})

    m = csv_source.read(cfg.csv_path, dialect)
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

    return _finish_csv(spectra, sched, m, cfg, log, t0, stop_after)


def _finish_csv(spectra, sched, m, cfg, log, t0, stop_after, extra=None):
    """Validation, ECM, aggregation and output — shared by every CSV route.

    Everything above this point differs between a generic time-domain file, a
    ready-made spectrum and an R2-D2 sweep.  Everything below it is the same
    physics on the same data structure, so it lives in one place.
    """
    import r2d2_geometry as geom

    spectra = {k: v for k, v in spectra.items()
               if k not in cfg.exclude_segments and v.freq.size >= 1}
    n_f = int(np.median([sp.freq.size for sp in spectra.values()])) if spectra else 0
    log.info(f"  Z     : {len(spectra)} segments, {n_f} frequencies each")
    if spectra and n_f < 6:
        # Not an error: a single point file is a legitimate thing to look at,
        # and the per-frequency table and maps below are exactly what it is
        # good for.  But say plainly which stages cannot run, rather than
        # reporting zero segments and letting it look like a read failure.
        log.warning(
            f"  {n_f} frequency point(s) per segment. lin-KK needs 6 and an "
            f"ECM needs at least 5, so those stages are skipped; the "
            f"per-frequency table and the |Z| maps are still written. Point "
            f"cfg.csv_path at the whole sweep folder to get spectra.")
    if stop_after == "bronze":
        return {"stage": "bronze", "n_segments": len(spectra),
                "schedule": sched, **(extra or {})}

    # --- validation --------------------------------------------------------
    kk = {seg: validate(sp, cfg) for seg, sp in spectra.items()}
    n_kk = sum(1 for v in kk.values() if v.get("ok"))
    log.info(f"  KK    : {n_kk}/{len(kk)} segments inside "
             f"{100*cfg.kk_tol:.0f} % residual")
    if stop_after == "silver":
        return {"stage": "silver", "n_segments": len(spectra), "kk": kk,
                **(extra or {})}

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
               "elapsed_s": round(time.time() - t0, 2),
               **(extra or {})})
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
