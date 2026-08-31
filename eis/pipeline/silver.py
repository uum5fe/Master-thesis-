"""Silver tier: aligned signals -> impedance spectra that are measurements.

Silver turns time series into ``Z(f)`` and then does the three things that
separate a spectrum from a plot: it attaches a per-point uncertainty, it
corrects the instrument rather than the data, and it writes down a verdict for
every segment instead of quietly discarding the awkward ones.

Order of operations
-------------------
The corrections have to be applied in a fixed order, and each one has to be
traceable to something that was measured:

===  =================================  ===========================  ==========
 #   step                               constant comes from          domain
===  =================================  ===========================  ==========
 1   integer + fractional alignment     bronze, measured             time
 2   clock-rate resampling              bronze, measured             time
 3   per-channel delay table            sync-calibration bench run   time
 4   via-shunt complex response         shunt loop inductance        frequency
 5   anti-alias filter mismatch         hardware specification       frequency
 6   shared residual delay              pooled across the card       frequency
 7   in-plane crosstalk                 plate characterisation       spatial
===  =================================  ===========================  ==========

Steps 1-3 happened in bronze.  Steps 4-7 happen here, and steps 4-6 are the
high-frequency chain documented in :mod:`eis.hf`: the via shunt is a complex
element whose ``L/R`` time constant is tens of microseconds, the anti-alias
filters of two channels are never identical, and whatever delay survives all of
that is identified from the *ensemble* of segments rather than from any one of
them.

Every segment survives
----------------------
A segment that fails a check gets a ``status``, a list of ``flags`` and a
``quality`` score in [0, 1].  Frequency points below the coherence gate are
marked rather than deleted, so all segments share one frequency grid - which is
what makes a frequency-resolved plate map, the whole-plate admittance sum and
the crosstalk deconvolution possible at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from eis import hf
from eis.pipeline.bronze import (
    STATUS_CHANNEL_FAULT, STATUS_NO_CALIBRATION, STATUS_OK,
    STATUS_TIMING_UNVERIFIED, BronzeCondition, SegmentSignal, SyncReport,
)
from eis.pipeline.config import PipelineConfig
from eis.pipeline.utils import clamp01, common_grid, safe_median
from eis.spectra import (
    SpectralResult, detect_tones, impedance_multiresolution,
    impedance_synchronous, impedance_welch,
)
from eis.validate import (
    AdmittanceSumResult, KKResult, StationarityResult, admittance_sum, lin_kk,
    stationarity_split_half,
)

STATUS_LOW_COHERENCE = "low_coherence"
STATUS_KK_FAIL = "kk_fail"
STATUS_NONSTATIONARY = "nonstationary"
STATUS_INACTIVE = "inactive"

#: Worst status wins.  The order is severity, not alphabet.
_STATUS_ORDER = [
    STATUS_OK, STATUS_KK_FAIL, STATUS_NONSTATIONARY, STATUS_LOW_COHERENCE,
    STATUS_NO_CALIBRATION, STATUS_TIMING_UNVERIFIED, STATUS_CHANNEL_FAULT,
    STATUS_INACTIVE,
]


def _worst(statuses: list[str]) -> str:
    known = [s for s in statuses if s in _STATUS_ORDER]
    if not known:
        return STATUS_OK
    return max(known, key=_STATUS_ORDER.index)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class SegmentSpectrum:
    """One segment's impedance, its verdict, and why."""

    segment: int
    card: int
    channel: str
    uc_channel: str
    uc_from_card: int
    #: Points that passed the coherence gate - what fitting and plotting use.
    spectrum: SpectralResult
    #: Every point, gate verdict included, on the grid every segment shares.
    spectrum_all: SpectralResult
    temperature_c: float = float("nan")
    shunt_H: float = 1.0
    physical_units: bool = True
    timing_sigma_s: float = 0.0
    instrument: hf.InstrumentResponse = field(
        default_factory=hf.InstrumentResponse
    )
    hfr: hf.HFRFit = field(default_factory=hf.HFRFit)
    kk: KKResult | None = None
    stationarity: StationarityResult | None = None
    status: str = STATUS_OK
    flags: list[str] = field(default_factory=list)
    quality: float = 0.0
    active: bool = True
    note: str = ""
    #: Filled in by the gold tier; kept on the same record so that a segment is
    #: one object from ingest to publication rather than three that have to be
    #: joined by number.
    ecm: object | None = None
    ecm_all: dict = field(default_factory=dict)

    @property
    def median_coherence(self) -> float:
        return safe_median(self.spectrum.coherence)

    @property
    def median_sigma_rel(self) -> float:
        return safe_median(self.spectrum.sigma_rel)


@dataclass
class SilverCondition:
    """Everything Silver knows about one operating point."""

    measurement_id: str
    condition: str
    fs: float
    duration_s: float
    sync: SyncReport
    segments: dict[int, SegmentSpectrum] = field(default_factory=dict)
    frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    common_mode: dict[int, hf.CommonModeResult] = field(default_factory=dict)
    crosstalk: hf.CrosstalkModel | None = None
    crosstalk_change: float = 0.0
    plate: AdmittanceSumResult | None = None
    tone_check: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def active_segments(self) -> list[int]:
        return [s for s, r in sorted(self.segments.items()) if r.active]

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.segments.values():
            counts[r.status] = counts.get(r.status, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Spectral estimation
# ---------------------------------------------------------------------------

def _choose_method(cfg: PipelineConfig) -> str:
    scfg = cfg.spectral
    if scfg.method in ("welch", "synchronous", "multiresolution"):
        return scfg.method
    if scfg.base_frequency_hz and scfg.excitation_tones_hz:
        return "synchronous"
    return "multiresolution"


def estimate_spectrum(
    signal: SegmentSignal, cfg: PipelineConfig, fs: float,
    method: str, f_max: float,
) -> SpectralResult:
    """Run the configured spectral estimator on one segment."""
    scfg = cfg.spectral
    if method == "synchronous":
        return impedance_synchronous(
            signal.current_a, signal.voltage_v, fs,
            base_frequency_hz=float(scfg.base_frequency_hz),
            tones_hz=list(scfg.excitation_tones_hz or []),
            periods=scfg.periods_per_window, estimator=scfg.estimator,
            gate=scfg.gate_windows, detrend=scfg.detrend,
            amplitude_mad_k=scfg.amplitude_mad_k,
            phase_outlier_rad=scfg.phase_outlier_rad,
            max_rejected_fraction=scfg.max_rejected_fraction,
        )
    if method == "multiresolution":
        return impedance_multiresolution(
            signal.current_a, signal.voltage_v, fs,
            f_min=scfg.f_min_hz, f_max=f_max,
            min_periods=scfg.min_periods_per_window,
            nperseg_min=scfg.nperseg_min, nperseg_max=scfg.nperseg,
            bands_per_decade=scfg.bands_per_decade, window=scfg.window,
            detrend=scfg.detrend, estimator=scfg.estimator, gate=scfg.gate_windows,
            amplitude_mad_k=scfg.amplitude_mad_k,
            phase_outlier_rad=scfg.phase_outlier_rad,
            max_rejected_fraction=scfg.max_rejected_fraction,
        )
    return impedance_welch(
        signal.current_a, signal.voltage_v, fs, nperseg=scfg.nperseg,
        noverlap=scfg.noverlap, window=scfg.window, detrend=scfg.detrend,
        f_min=scfg.f_min_hz, f_max=f_max, estimator=scfg.estimator,
        gate=scfg.gate_windows, amplitude_mad_k=scfg.amplitude_mad_k,
        phase_outlier_rad=scfg.phase_outlier_rad,
        max_rejected_fraction=scfg.max_rejected_fraction,
    )


# ---------------------------------------------------------------------------
# Correction application
# ---------------------------------------------------------------------------

def _apply_factor(spectrum: SpectralResult, factor: np.ndarray) -> SpectralResult:
    """Multiply every impedance variant a spectrum carries by ``factor``."""
    out = spectrum.with_impedance(spectrum.Z * factor)
    if out.Z_h1 is not None:
        out.Z_h1 = out.Z_h1 * factor
    if out.Z_h2 is not None:
        out.Z_h2 = out.Z_h2 * factor
    return out


def _shunt_response(
    cfg: PipelineConfig, segment: int, shunt_H: float
) -> hf.InstrumentResponse:
    """The complex part of the acquisition chain that is known in advance."""
    hcfg = cfg.hf
    inductance_nh = hcfg.shunt_inductance_nh_per_segment.get(
        segment, hcfg.shunt_inductance_nh
    )
    tau = hf.shunt_time_constant(
        shunt_H, cfg.geometry.segment_area_cm2, inductance_nh
    ) if hcfg.enabled else 0.0
    return hf.InstrumentResponse(
        shunt_tau_s=tau,
        aa_order=hcfg.aa_order if hcfg.enabled else 0,
        aa_corner_hz=hcfg.aa_corner_hz,
        aa_corner_mismatch=hcfg.aa_corner_mismatch,
    )


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

def score_quality(
    record: SegmentSpectrum, cfg: PipelineConfig, f_max: float
) -> float:
    """Combine coherence, uncertainty, KK residual and timing into [0, 1].

    A single number is not a substitute for the four it summarises - all of
    them reach the output table - but a plate map needs one axis to shade by,
    and shading by an explicit score beats dropping segments by an implicit
    one.
    """
    qcfg = cfg.quality
    terms: list[tuple[float, float]] = []

    coherence = record.median_coherence
    if np.isfinite(coherence):
        terms.append((qcfg.w_coherence, clamp01(coherence)))

    sigma = record.median_sigma_rel
    if np.isfinite(sigma):
        terms.append((qcfg.w_uncertainty,
                      clamp01(1.0 - sigma / max(qcfg.sigma_reference, 1e-9))))

    if record.kk is not None and np.isfinite(record.kk.max_residual_pct):
        terms.append((qcfg.w_kk, clamp01(
            1.0 - record.kk.max_residual_pct / max(qcfg.kk_reference_pct, 1e-9)
        )))

    phase_error = 2.0 * np.pi * f_max * max(record.timing_sigma_s, 0.0)
    terms.append((qcfg.w_timing, clamp01(1.0 - phase_error / 0.1)))

    total = sum(w for w, _ in terms)
    if total <= 0:
        return 0.0
    return float(sum(w * v for w, v in terms) / total)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_silver(
    bronze: BronzeCondition,
    cfg: PipelineConfig,
    reference_spectrum: tuple[np.ndarray, np.ndarray] | None = None,
    log=print,
) -> SilverCondition:
    """Spectra, uncertainty, the high-frequency chain and the verdicts."""
    scfg = cfg.spectral
    f_max = min(scfg.f_max_hz, 0.45 * bronze.fs)
    method = _choose_method(cfg)
    log(f"\n  impedance: {method}, estimator={scfg.estimator}, "
        f"band=[{scfg.f_min_hz}, {f_max:.0f}] Hz")

    silver = SilverCondition(
        measurement_id=bronze.measurement_id, condition=bronze.condition,
        fs=bronze.fs, duration_s=bronze.duration_s, sync=bronze.sync,
    )

    # ---- pass 1: spectra, uncertainty, known instrument response ---------
    halves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for signal in bronze.segments():
        flags: list[str] = []
        try:
            spectrum = estimate_spectrum(signal, cfg, bronze.fs, method, f_max)
        except Exception as exc:
            silver.segments[signal.segment] = _dead_segment(signal, exc)
            continue

        # Timing uncertainty is part of the phase uncertainty, not a footnote.
        # It is systematic, so it is kept separate from the random part that
        # the fits are weighted by - see SpectralResult.sigma_rel_random.
        spectrum.sigma_rel_random = spectrum.sigma_rel.copy()
        if scfg.propagate_timing_uncertainty and signal.timing_sigma_s > 0:
            spectrum.sigma_rel = hf.inflate_sigma_for_timing(
                spectrum.sigma_rel, spectrum.f, signal.timing_sigma_s
            )
            flags.append("timing_uncertainty_propagated")

        response = _shunt_response(cfg, signal.segment, signal.shunt_H)
        if response.active:
            spectrum = _apply_factor(spectrum, response.factor(spectrum.f))
            flags.append("instrument_response")

        spectrum = spectrum.mark(scfg.min_coherence)
        record = SegmentSpectrum(
            segment=signal.segment, card=signal.card, channel=signal.channel,
            uc_channel=signal.uc_channel, uc_from_card=signal.uc_from_card,
            spectrum=spectrum.usable(), spectrum_all=spectrum,
            temperature_c=signal.temperature_c, shunt_H=signal.shunt_H,
            physical_units=signal.physical_units,
            timing_sigma_s=signal.timing_sigma_s, instrument=response,
            flags=flags, note=signal.note,
        )
        record.status = signal.status
        silver.segments[signal.segment] = record

        if cfg.validation.run_stationarity:
            halves[signal.segment] = _split_half_impedance(
                signal, cfg, bronze.fs, method, f_max
            )

    # release the memory-mapped files now that every channel has been read
    for inv in bronze.inventories.values():
        inv.famos = None                                  # type: ignore[assignment]

    if not silver.segments:
        raise RuntimeError("no segment produced a spectrum")

    silver.frequencies = common_grid(
        r.spectrum_all.f for r in silver.segments.values()
    )
    log(f"  spectra: {len(silver.segments)} segments on a shared grid of "
        f"{len(silver.frequencies)} frequencies "
        f"({sum(r.spectrum_all.n_used for r in silver.segments.values())} "
        f"points above gamma^2 = {scfg.min_coherence})")

    if any(r.instrument.active for r in silver.segments.values()):
        example = next(r for r in silver.segments.values() if r.instrument.active)
        log(f"  instrument response applied [{', '.join(example.instrument.terms)}]"
            f" -> {example.instrument.phase_deg_at(f_max):+.2f} deg at "
            f"{f_max:.0f} Hz")

    # ---- pass 2: high-frequency fits ------------------------------------
    _refit_hfr(silver, cfg)

    # ---- pass 3: the shared residual delay ------------------------------
    if cfg.hf.enabled and cfg.hf.identify_common_mode:
        if reference_spectrum is not None:
            _apply_reference_anchor(silver, cfg, reference_spectrum, f_max, log)
        _identify_and_apply_common_mode(silver, cfg, f_max, log)

    # ---- pass 4: in-plane crosstalk -------------------------------------
    if cfg.hf.enabled and cfg.hf.crosstalk_alpha > 0:
        _deconvolve_crosstalk(silver, cfg, log)

    # ---- validation ------------------------------------------------------
    if cfg.validation.run_lin_kk:
        for record in silver.segments.values():
            if len(record.spectrum.f) < 6:
                continue
            record.kk = lin_kk(
                record.spectrum.f, record.spectrum.Z,
                sigma=record.spectrum.sigma_random,
                sigma_total=record.spectrum.sigma_real,
                max_elements=cfg.validation.kk_max_elements,
                mu_target=cfg.validation.kk_mu_target,
                max_residual_pct=cfg.validation.kk_max_residual_pct,
                max_sigma=cfg.validation.kk_max_sigma,
            )
        passed = sum(1 for r in silver.segments.values() if r.kk and r.kk.passed)
        shapes: dict[str, int] = {}
        for r in silver.segments.values():
            if r.kk:
                shapes[r.kk.shape_class] = shapes.get(r.kk.shape_class, 0) + 1
        log(f"  Kramers-Kronig: {passed}/{len(silver.segments)} pass "
            f"(<= {cfg.validation.kk_max_residual_pct}% residual); "
            f"residual shapes: {shapes}")

    if cfg.validation.run_stationarity and halves:
        drifting = 0
        for segment, (Z_first, Z_second) in halves.items():
            record = silver.segments.get(segment)
            if record is None or Z_first is None:
                continue
            record.stationarity = stationarity_split_half(
                Z_first, Z_second, cfg.validation.stationarity_tolerance
            )
            drifting += not record.stationarity.passed
        log(f"  stationarity: {len(halves) - drifting}/{len(halves)} segments "
            f"agree between the two halves of the record within "
            f"{cfg.validation.stationarity_tolerance:.0%}")

    # ---- tone verification -----------------------------------------------
    if scfg.verify_tones and scfg.excitation_tones_hz:
        silver.tone_check = _verify_tones(silver, list(scfg.excitation_tones_hz))
        log(f"  excitation: {silver.tone_check}")

    # ---- status, flags, quality ------------------------------------------
    _classify(silver, cfg, f_max)
    counts = silver.status_counts()
    active = len(silver.active_segments)
    log(f"  segments: {active}/{len(silver.segments)} active; "
        f"{', '.join(f'{v} {k}' for k, v in sorted(counts.items()))}")

    # ---- whole-plate identity --------------------------------------------
    if cfg.validation.run_admittance_sum and silver.active_segments:
        silver.plate = admittance_sum(
            {
                s: (silver.segments[s].spectrum.f, silver.segments[s].spectrum.Z)
                for s in silver.active_segments
                if silver.segments[s].physical_units
            },
            cfg.geometry.segment_area_cm2, cfg.geometry.cell_area_cm2,
            reference_spectrum,
        )
        if silver.plate is not None and silver.plate.n_segments:
            note = f"  plate admittance sum: {silver.plate.n_segments} segments"
            if silver.plate.median_relative_difference is not None:
                note += (f"; median deviation from the reference instrument "
                         f"{silver.plate.median_relative_difference:.1%}")
            log(note)
    return silver


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def _dead_segment(signal: SegmentSignal, exc: Exception) -> SegmentSpectrum:
    """A channel that could not produce a spectrum at all - still a row."""
    empty = SpectralResult(
        f=np.array([]), Z=np.array([], complex), coherence=np.array([]),
        sigma_rel=np.array([]), used=np.array([], bool),
    )
    return SegmentSpectrum(
        segment=signal.segment, card=signal.card, channel=signal.channel,
        uc_channel=signal.uc_channel, uc_from_card=signal.uc_from_card,
        spectrum=empty, spectrum_all=empty, temperature_c=signal.temperature_c,
        shunt_H=signal.shunt_H, physical_units=signal.physical_units,
        timing_sigma_s=signal.timing_sigma_s,
        status=STATUS_CHANNEL_FAULT, active=False, quality=0.0,
        flags=["spectrum_failed"],
        note=f"{type(exc).__name__}: {exc}",
    )


def _split_half_impedance(
    signal: SegmentSignal, cfg: PipelineConfig, fs: float,
    method: str, f_max: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Impedance of the first and second half of the record, on one grid.

    Kramers-Kronig assumes a time-invariant system.  A cell that dries out or
    floods during the record violates that premise, and the KK residual alone
    does not always reveal it - so it is tested directly.
    """
    n = min(len(signal.current_a), len(signal.voltage_v)) // 2
    if n < cfg.spectral.nperseg_min * 2:
        return None, None
    out: list[np.ndarray | None] = []
    for lo, hi in ((0, n), (n, 2 * n)):
        half = SegmentSignal(
            segment=signal.segment, card=signal.card, channel=signal.channel,
            uc_from_card=signal.uc_from_card, uc_channel=signal.uc_channel,
            current_a=signal.current_a[lo:hi], voltage_v=signal.voltage_v[lo:hi],
            temperature_c=signal.temperature_c, shunt_H=signal.shunt_H,
            timing_sigma_s=signal.timing_sigma_s,
        )
        try:
            out.append(estimate_spectrum(half, cfg, fs, method, f_max).Z)
        except Exception:
            return None, None
    return out[0], out[1]


def _refit_hfr(silver: SilverCondition, cfg: PipelineConfig) -> None:
    """(Re)fit ``Rs + jwL + arc`` on the top of every segment's band."""
    for record in silver.segments.values():
        spectrum = record.spectrum
        if len(spectrum.f) < cfg.hf.hfr_min_points:
            record.hfr = hf.HFRFit(
                n_points=len(spectrum.f),
                note="too few usable points for a high-frequency fit",
            )
            continue
        record.hfr = hf.fit_hf_resistance(
            spectrum.f, spectrum.Z, spectrum.sigma_random,
            band_decades=cfg.hf.hfr_band_decades,
            min_points=cfg.hf.hfr_min_points,
            arc_min_points=cfg.hf.hfr_arc_min_points,
        )


def _apply_delay(
    silver: SilverCondition, cfg: PipelineConfig,
    result: hf.CommonModeResult, segments: list[int], flag: str,
) -> None:
    """Rotate a group of segments by an identified shared delay."""
    response = hf.InstrumentResponse(delay_s=result.delay_s)
    for segment in segments:
        record = silver.segments.get(segment)
        if record is None or len(record.spectrum_all.f) == 0:
            continue
        record.spectrum_all = _apply_factor(
            record.spectrum_all, response.factor(record.spectrum_all.f)
        )
        record.spectrum = record.spectrum_all.usable()
        record.instrument.delay_s += result.delay_s
        # The identification has its own uncertainty; it belongs in the error
        # bars, not in a footnote.
        record.timing_sigma_s = float(np.hypot(
            record.timing_sigma_s,
            result.delay_sigma_s if np.isfinite(result.delay_sigma_s) else 0.0,
        ))
        if flag not in record.flags:
            record.flags.append(flag)
    result.applied = True
    _refit_hfr(silver, cfg)


def _apply_reference_anchor(
    silver: SilverCondition, cfg: PipelineConfig,
    reference_spectrum: tuple[np.ndarray, np.ndarray], f_max: float, log,
) -> None:
    """Take the shared delay from an independent cell-level measurement.

    This is the strongest anchor there is and it is tried first, because it
    involves no identifying assumption at all: the segments are electrically in
    parallel, so their summed admittance *must* reproduce what a potentiostat
    measured across the whole cell, and any phase difference between the two is
    a delay in the segment chain.  It corrects the part that is common to every
    card, which is exactly the part no per-card method can see.
    """
    usable = [
        s for s in silver.active_segments if silver.segments[s].physical_units
    ]
    if len(usable) < 2:
        return
    plate = admittance_sum(
        {s: (silver.segments[s].spectrum.f, silver.segments[s].spectrum.Z)
         for s in usable},
        cfg.geometry.segment_area_cm2, cfg.geometry.cell_area_cm2,
        reference_spectrum,
    )
    if plate.Z_reference is None or len(plate.f) < 4:
        log("    reference anchor unavailable: the reference spectrum does not "
            "overlap the segment frequency grid")
        return

    band = (cfg.hf.common_mode_band_hz[0],
            min(cfg.hf.common_mode_band_hz[1], f_max))
    result = hf.delay_from_reference(plate.f, plate.Z_plate, plate.Z_reference, band)
    bound = max(cfg.hf.delay_bound_s, cfg.hf.delay_bound_floor_s)
    if not result.ok or abs(result.delay_s) > bound:
        result.clipped = result.ok
        log(f"    reference anchor not applied ({result.note})")
        silver.common_mode[-1] = result
        return
    silver.common_mode[-1] = result          # card -1 == "the whole plate"
    _apply_delay(silver, cfg, result, sorted(silver.segments), "reference_anchor")
    log(f"    reference anchor: {result.note} "
        f"({result.phase_deg_at(f_max):+.2f} deg at {f_max:.0f} Hz)")


def _identify_and_apply_common_mode(
    silver: SilverCondition, cfg: PipelineConfig, f_max: float, log
) -> None:
    """Find the delay shared by each card's segments and take it out.

    The bound comes from the synchronisation stage, never from the search: over
    a finite band a delay is degenerate with an inductance, so an unbounded
    identification would happily absorb the cell's real inductance and report a
    beautifully flat Nyquist arc that is wrong.
    """
    hcfg = cfg.hf
    by_card: dict[int, dict[int, hf.HFRFit]] = {}
    for segment, record in silver.segments.items():
        if record.status in (STATUS_CHANNEL_FAULT, STATUS_INACTIVE):
            continue
        by_card.setdefault(record.card, {})[segment] = record.hfr

    for card, fits in sorted(by_card.items()):
        sync = silver.sync.cards.get(card)
        if hcfg.delay_bound_s > 0:
            bound = hcfg.delay_bound_s
        else:
            measured = sync.timing_sigma_s if sync is not None else 0.0
            bound = max(3.0 * measured, hcfg.delay_bound_floor_s)

        result = hf.identify_common_delay(
            fits, bound_s=bound,
            min_segments=hcfg.min_segments_for_pooling,
        )
        method = "decorrelation"

        # A real shared delay removes the L-vs-Rs correlation and tightens the
        # inductance scatter.  If the estimate does neither, it was noise.
        accepted = result.ok and result.delay_s != 0.0 and (
            abs(result.correlation_after) < abs(result.correlation_before)
            and result.l_spread_after_h <= result.l_spread_before_h
        )
        if result.ok and not accepted:
            result.note += (
                "; rejected because it did not reduce both the L-vs-Rs "
                "correlation and the inductance scatter"
            )

        # Fallback for a card whose timing is coarse to begin with - a proxy
        # measurement is good to microseconds, and a microsecond is tens of
        # degrees at the top of the band, so there is something worth finding.
        coarse = (
            sync is not None
            and sync.timing_sigma_s
            > hcfg.kk_fallback_min_sigma_ratio * cfg.sync.residual_tolerance_s
        )
        if not accepted and hcfg.kk_fallback and coarse:
            spectra = {
                s: (
                    silver.segments[s].spectrum.f,
                    silver.segments[s].spectrum.Z,
                    silver.segments[s].spectrum.sigma_random,
                )
                for s in sorted(fits)
                if len(silver.segments[s].spectrum.f) >= 6
            }
            rs_reference = safe_median(
                (fit.rs_ohm for fit in fits.values() if fit.ok), 0.0
            )
            fallback = hf.pooled_kk_delay(
                spectra, bound_s=bound, n_grid=hcfg.kk_fallback_grid,
                max_segments=hcfg.kk_fallback_segments,
                max_elements=cfg.validation.kk_max_elements,
                max_inductance_h=hcfg.kk_fallback_max_inductance_nh * 1e-9,
                rs_reference_ohm=rs_reference if np.isfinite(rs_reference) else 0.0,
            )
            fallback.note = (
                f"decorrelation unusable ({result.note or 'no estimate'}); "
                + fallback.note
            )
            result, method, accepted = fallback, "pooled_kk", fallback.ok

        silver.common_mode[card] = result
        if not accepted or result.delay_s == 0.0:
            log(f"    card {card}: common-mode delay not applied "
                f"({result.note or 'no estimate'})")
            continue

        _apply_delay(
            silver, cfg, result,
            [s for s, r in silver.segments.items() if r.card == card],
            f"common_mode_delay:{method}",
        )
        log(f"    card {card}: {result.note} "
            f"({result.phase_deg_at(f_max):+.2f} deg at {f_max:.0f} Hz)")


def _deconvolve_crosstalk(
    silver: SilverCondition, cfg: PipelineConfig, log
) -> None:
    """Undo the spatial smoothing that in-plane conduction imposes."""
    grid = silver.frequencies
    usable = [
        s for s in sorted(silver.segments)
        if silver.segments[s].physical_units
        and len(silver.segments[s].spectrum_all.f) == len(grid) and len(grid)
    ]
    if len(usable) < 4:
        log("    crosstalk deconvolution skipped: too few segments on the "
            "shared frequency grid")
        return

    model = hf.build_crosstalk_model(
        usable, cfg.geometry.segment_coords, cfg.hf.crosstalk_alpha,
        cfg.hf.crosstalk_regularisation,
    )
    silver.crosstalk = model
    if not model.active:
        log(f"    crosstalk deconvolution skipped: {model.note}")
        return

    Y = np.array([1.0 / silver.segments[s].spectrum_all.Z for s in model.segments])
    corrected, change = hf.deconvolve_crosstalk(model, Y)
    silver.crosstalk_change = change
    for i, segment in enumerate(model.segments):
        record = silver.segments[segment]
        record.spectrum_all = record.spectrum_all.with_impedance(1.0 / corrected[i])
        record.spectrum = record.spectrum_all.usable()
        record.flags.append("crosstalk_deconvolved")
    log(f"    crosstalk deconvolution: {model.note}; median change "
        f"{change:.2%}")
    _refit_hfr(silver, cfg)


def _verify_tones(silver: SilverCondition, designed: list[float]) -> str:
    """Check that the excitation is where the configuration says it is."""
    best = max(
        silver.segments.values(),
        key=lambda r: safe_median(r.spectrum_all.coherence, 0.0),
        default=None,
    )
    if best is None or len(best.spectrum_all.f) == 0:
        return "no spectrum to verify the tones against"
    found = detect_tones(best.spectrum_all.f, best.spectrum_all.coherence)
    if found.size == 0:
        return (
            f"no coherence peaks found; the {len(designed)} configured tones "
            f"could not be confirmed in the data"
        )
    matched = sum(
        1 for t in designed
        if np.min(np.abs(found - t)) <= max(0.02 * t, 1.0)
    )
    return (
        f"{matched}/{len(designed)} configured tones confirmed among "
        f"{found.size} coherence peaks"
    )


def _classify(silver: SilverCondition, cfg: PipelineConfig, f_max: float) -> None:
    """Write every segment's verdict into the data."""
    qcfg = cfg.quality
    for record in silver.segments.values():
        statuses = [record.status]
        n_used = record.spectrum_all.n_used
        if n_used < qcfg.active_min_points:
            record.active = False
            statuses.append(STATUS_INACTIVE)
        elif n_used < cfg.spectral.min_points_per_segment:
            statuses.append(STATUS_LOW_COHERENCE)
            record.flags.append(
                f"only {n_used} of {len(record.spectrum_all.f)} points reach "
                f"gamma^2 >= {cfg.spectral.min_coherence}"
            )
        if record.kk is not None and not record.kk.passed:
            statuses.append(STATUS_KK_FAIL)
            record.flags.append(f"kk_{record.kk.shape_class}")
        if record.stationarity is not None and not record.stationarity.passed:
            statuses.append(STATUS_NONSTATIONARY)
        record.status = _worst(statuses)
        record.quality = score_quality(record, cfg, f_max)

    if not qcfg.keep_all_segments:
        for segment in [s for s, r in silver.segments.items() if not r.active]:
            del silver.segments[segment]
        silver.notes.append(
            "keep_all_segments is off: inactive segments were removed from the "
            "result and their reasons are no longer queryable"
        )


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def impedance_frame(
    silver: SilverCondition, segment_area_cm2: float,
    provenance: dict[str, str] | None = None, only_used: bool = False,
) -> pd.DataFrame:
    """One row per (segment, frequency) - every segment, every point."""
    asr = segment_area_cm2 * 1e3                     # Ohm -> mOhm*cm^2
    rows = []
    for segment, r in sorted(silver.segments.items()):
        s = r.spectrum_all
        used = s.used_mask
        for i in range(len(s.f)):
            if only_used and not used[i]:
                continue
            z = s.Z[i]
            rows.append({
                "measurement_id": silver.measurement_id,
                "condition": silver.condition,
                "segment": segment,
                "card": r.card,
                "uc_channel": r.uc_channel,
                "frequency_hz": float(s.f[i]),
                "z_real_ohm": float(z.real),
                "z_imag_ohm": float(z.imag),
                "z_mod_ohm": float(abs(z)),
                "z_phase_deg": float(np.degrees(np.angle(z))),
                "z_real_mohm_cm2": float(z.real) * asr,
                "z_imag_mohm_cm2": float(z.imag) * asr,
                "coherence": float(s.coherence[i]),
                "sigma_rel": float(s.sigma_rel[i]),
                "sigma_real_ohm": float(abs(z) * s.sigma_rel[i]),
                "sigma_imag_ohm": float(abs(z) * s.sigma_rel[i]),
                "n_eff": float(s.n_eff_f[i]) if s.n_eff_f is not None else s.n_eff,
                "nperseg": float(s.nperseg_f[i]) if s.nperseg_f is not None else np.nan,
                "used": bool(used[i]),
                "segment_status": r.status,
                "segment_quality": r.quality,
                "physical_units": r.physical_units,
                "estimator": s.estimator,
                "method": s.method,
                **(provenance or {}),
            })
    return pd.DataFrame(rows)


def segment_frame(
    silver: SilverCondition, segment_area_cm2: float,
    provenance: dict[str, str] | None = None,
) -> pd.DataFrame:
    """One row per segment: verdict, high-frequency fit, corrections applied."""
    asr = segment_area_cm2 * 1e3
    rows = []
    for segment, r in sorted(silver.segments.items()):
        row: dict = {
            "measurement_id": silver.measurement_id,
            "condition": silver.condition,
            "segment": segment,
            "card": r.card,
            "channel": r.channel,
            "uc_channel": r.uc_channel,
            "uc_from_card": r.uc_from_card,
            "status": r.status,
            "active": r.active,
            "quality": r.quality,
            "flags": ",".join(r.flags),
            "physical_units": r.physical_units,
            "n_points": len(r.spectrum_all.f),
            "n_points_used": r.spectrum_all.n_used,
            "n_windows_used": r.spectrum_all.n_windows_used,
            "n_windows_total": r.spectrum_all.n_windows_total,
            "median_coherence": r.median_coherence,
            "median_sigma_rel": r.median_sigma_rel,
            "temperature_c": r.temperature_c,
            "shunt_H_V_cm2_per_A": r.shunt_H,
            "timing_sigma_ns": r.timing_sigma_s * 1e9,
            "instrument_terms": ",".join(r.instrument.terms),
            "instrument_phase_deg_at_fmax": (
                r.instrument.phase_deg_at(float(r.spectrum_all.f.max()))
                if len(r.spectrum_all.f) else np.nan
            ),
            "rs_hf_mohm_cm2": r.hfr.rs_ohm * asr,
            "rs_hf_sigma_mohm_cm2": r.hfr.rs_sigma_ohm * asr,
            "hf_inductance_nh": r.hfr.l_h * 1e9,
            "hf_inductance_sigma_nh": r.hfr.l_sigma_h * 1e9,
            # Where the fitted arc turns inductive - the frequency the Nyquist
            # plot crosses the real axis.  A segment whose crossover sits far
            # from the plate median has either a wiring fault or a residual
            # phase error, and the two look identical on a magnitude plot.
            "hf_crossover_hz": r.hfr.crossover_hz,
            "hf_fit_chi2_reduced": r.hfr.chi2_reduced,
            "hf_fit_points": r.hfr.n_points,
            "note": r.note,
        }
        if r.kk is not None:
            row |= {
                "kk_pass": r.kk.passed, "kk_elements": r.kk.n_elements,
                "kk_mu": r.kk.mu, "kk_max_residual_pct": r.kk.max_residual_pct,
                "kk_rms_residual_pct": r.kk.rms_residual_pct,
                "kk_max_normalised_residual": r.kk.max_normalised_residual,
                "kk_shape": r.kk.shape_class,
            }
        if r.stationarity is not None:
            row |= {
                "stationarity_pass": r.stationarity.passed,
                "stationarity_median_diff": r.stationarity.median_relative_difference,
            }
        row |= provenance or {}
        rows.append(row)
    return pd.DataFrame(rows)


def common_mode_frame(silver: SilverCondition) -> pd.DataFrame:
    """One row per card: what the pooled identification found, and whether it
    survived its own consistency check."""
    rows = []
    for card, r in sorted(silver.common_mode.items()):
        rows.append({
            "measurement_id": silver.measurement_id,
            "condition": silver.condition,
            "card": card,
            "delay_ns": r.delay_s * 1e9,
            "delay_sigma_ns": r.delay_sigma_s * 1e9,
            "n_segments": r.n_segments,
            "l_vs_rs_correlation_before": r.correlation_before,
            "l_vs_rs_correlation_after": r.correlation_after,
            "l_spread_before_nh": r.l_spread_before_h * 1e9,
            "l_spread_after_nh": r.l_spread_after_h * 1e9,
            "rs_spread_uohm": r.rs_spread_ohm * 1e6,
            "clipped_to_bound": r.clipped,
            "applied": r.applied,
            "ok": r.ok,
            "note": r.note,
        })
    return pd.DataFrame(rows)
