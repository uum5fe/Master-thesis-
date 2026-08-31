"""Bronze tier: raw FAMOS files -> provably aligned, physically scaled signals.

What this tier promises
-----------------------
Every sample handed to Silver is on a common time base, that time base was
*measured* rather than assumed, and the residual timing error is reported with
each card so the tiers above can widen their error bars instead of pretending.

Why synchronisation is measured even though it often cancels
------------------------------------------------------------
Each card records its own copy of the cell voltage, and the local impedance is
a *ratio* formed from two channels.  If a segment is paired with the cell
voltage recorded on its own card, a bulk timing offset of that card cancels
exactly - it appears in numerator and denominator alike.  That is why the
``same_card`` pairing is the accurate default.

It does not make synchronisation optional:

1. The cancellation has to be **verified**.  If the two cell-voltage copies do
   not agree after alignment, the assumption that they measure the same node -
   or that channels within a card are sampled simultaneously - is false, and
   every impedance on that card is suspect.
2. **Channel-to-channel skew inside a card does not cancel.**  It sits
   directly between the segment current and the cell voltage.
3. When a card's own cell-voltage channel is dead - a real failure mode - the
   fallback is another card's copy, and then the full inter-card alignment is
   what makes the result valid rather than wrong by 137 degrees per sample.
4. The measured skew and clock drift are hardware diagnostics worth reporting
   in their own right.

Nothing is dropped here
-----------------------
A card whose timing could not be established used to take its segments out of
the result entirely.  It no longer does: those segments are marked
``timing_unverified`` and carry a timing uncertainty of their own, which Silver
folds into the per-point phase uncertainty.  A segment with no shunt
calibration is scaled by unity and marked ``no_calibration`` instead of
vanishing.  The quality decision then lives in the data, where it can be
queried, rather than in a set of segment numbers that never reached the table.

Memory
------
Segment currents are produced one at a time by :meth:`BronzeCondition.segments`
and never held together: eighty channels of a two-minute record at 10 kHz is
three quarters of a gigabyte, and there is no reason to pay it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from eis.calibrate import (
    ShuntCalibration, TemperatureCalibration, TemperatureField,
    build_temperature_field, sensor_voltage_to_celsius,
    shunt_voltage_to_current_density,
)
from eis.io.famos import FamosFile, classify_channel, open_famos
from eis.pipeline.config import PipelineConfig
from eis.pipeline.utils import ac_rms, align_channel
from eis.spectra import apply_notch
from eis.sync.drift import DriftEstimate, estimate_drift
from eis.sync.resample import valid_span
from eis.sync.skew import DelayEstimate, closure_residual, estimate_delay

#: Statuses this tier can assign.  Silver may narrow them further but never
#: removes a segment.
STATUS_OK = "ok"
STATUS_NO_CALIBRATION = "no_calibration"
STATUS_TIMING_UNVERIFIED = "timing_unverified"
STATUS_CHANNEL_FAULT = "channel_fault"


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

@dataclass
class CardInventory:
    card: int
    path: str
    famos: FamosFile
    segment_channels: dict[int, str] = field(default_factory=dict)
    uc_channels: list[str] = field(default_factory=list)
    temp_channels: list[str] = field(default_factory=list)
    other_channels: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return self.famos.n_samples


def inventory_card(path: str, card: int) -> CardInventory:
    """Open one card's file and sort its channels into roles."""
    famos = open_famos(path, card=card)
    inv = CardInventory(card=card, path=path, famos=famos)
    for name in famos.channel_names:
        role, key = classify_channel(name)
        if role == "segment":
            inv.segment_channels[int(key)] = name
        elif role == "cell_voltage":
            inv.uc_channels.append(name)
        elif role == "temperature":
            inv.temp_channels.append(name)
        else:
            inv.other_channels.append(name)
    return inv


# ---------------------------------------------------------------------------
# Synchronisation report
# ---------------------------------------------------------------------------

@dataclass
class CardSync:
    card: int
    uc_channel: str | None
    uc_rms_v: float = 0.0
    healthy_uc: bool = True
    tau_s: float = 0.0
    tau_sigma_s: float = float("nan")
    delay_slope_ppm: float = 0.0
    delay_slope_sigma_ppm: float = float("nan")
    drift_corrected: bool = False
    coherence: float = float("nan")
    intercept_rad: float = float("nan")
    metadata_offset_s: float | None = None
    residual_tau_s: float | None = None
    method: str = "reference"
    ok: bool = True
    note: str = ""

    @property
    def timing_sigma_s(self) -> float:
        """Uncertainty to propagate into the impedance phase.

        Once the alignment has been verified, the honest figure is the measured
        residual, not the uncertainty of the estimate that produced it.  When
        it could not be verified the estimate's own sigma stands, and when the
        timing failed outright the whole measured delay is uncertain.
        """
        if not self.ok:
            return max(abs(self.tau_s), self.tau_sigma_s if
                       np.isfinite(self.tau_sigma_s) else 0.0, 1e-6)
        candidates = [self.tau_sigma_s]
        if self.residual_tau_s is not None:
            candidates.append(abs(self.residual_tau_s))
        finite = [c for c in candidates if np.isfinite(c)]
        return float(max(finite)) if finite else 0.0


@dataclass
class SyncReport:
    reference_card: int
    fs: float
    cards: dict[int, CardSync] = field(default_factory=dict)
    closure_s: dict[tuple[int, int, int], float] = field(default_factory=dict)
    alignment_applied: bool = False
    passed: bool = True
    notes: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for card, s in sorted(self.cards.items()):
            rows.append({
                "card": card,
                "uc_channel": s.uc_channel,
                "uc_rms_mV": s.uc_rms_v * 1e3,
                "uc_healthy": s.healthy_uc,
                "tau_ns": s.tau_s * 1e9,
                "tau_sigma_ns": s.tau_sigma_s * 1e9,
                "timing_sigma_ns": s.timing_sigma_s * 1e9,
                "delay_slope_ppm": s.delay_slope_ppm,
                "clock_rate_ppm": -s.delay_slope_ppm,
                "drift_corrected": s.drift_corrected,
                "coherence": s.coherence,
                "phase_intercept_rad": s.intercept_rad,
                "metadata_offset_ms": (
                    None if s.metadata_offset_s is None else s.metadata_offset_s * 1e3
                ),
                "residual_tau_ns": (
                    None if s.residual_tau_s is None else s.residual_tau_s * 1e9
                ),
                "ok": s.ok,
                "note": s.note,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage 1: measure the synchronisation
# ---------------------------------------------------------------------------

def _deembed_shunt(trace: np.ndarray, fs: float, tau_s: float) -> np.ndarray:
    """Divide a via-shunt's ``1 + jw*tau`` out of a segment trace.

    Needed only on the timing path.  When a card's cell-voltage channel is dead
    the delay has to be measured against a *segment* channel instead, and a
    segment channel carries the shunt's own phase - tens of microseconds of
    apparent delay, which is the same order as the accuracy the proxy claims.
    Left in, it does not look like an error; it looks like the card is late.
    """
    if tau_s <= 0:
        return trace
    n = len(trace)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    spectrum = np.fft.rfft(np.asarray(trace, float))
    return np.fft.irfft(spectrum / (1.0 + 1j * 2 * np.pi * freqs * tau_s), n)


def measure_sync(
    inventories: dict[int, CardInventory], cfg: PipelineConfig,
    shunt: ShuntCalibration | None = None,
) -> tuple[SyncReport, dict[int, np.ndarray]]:
    """Measure inter-card skew and drift from the shared cell-voltage channel.

    Returns the report and the (unaligned) cell-voltage trace of each card,
    which the caller reuses so the files are not read twice.

    ``shunt`` is optional and used for one thing only: when a card falls back
    to a segment channel as its timing proxy, the shunt's own phase is removed
    from that channel first, so the proxy measures a delay rather than a delay
    plus an inductance.
    """
    fs = float(np.median([inv.famos.fs for inv in inventories.values()]))
    scfg = cfg.sync

    # --- pick the healthiest cell-voltage channel on each card ------------
    uc_traces: dict[int, np.ndarray] = {}
    uc_names: dict[int, str] = {}
    uc_rms: dict[int, float] = {}
    for card, inv in sorted(inventories.items()):
        best_name, best_rms, best_trace = None, 0.0, None
        for name in inv.uc_channels:
            trace = inv.famos.channel(name)
            rms = ac_rms(trace)
            if rms > best_rms:
                best_name, best_rms, best_trace = name, rms, trace
        if best_name is not None:
            uc_names[card] = best_name
            uc_rms[card] = best_rms
            uc_traces[card] = best_trace              # type: ignore[assignment]

    if not uc_traces:
        raise RuntimeError(
            "no card carries a cell-voltage (UC*) channel; inter-card timing "
            "cannot be measured and no local impedance can be formed"
        )

    healthy = {c for c, r in uc_rms.items() if r >= scfg.uc_min_rms_v}
    reference = max(healthy or set(uc_rms), key=lambda c: uc_rms[c])

    report = SyncReport(reference_card=reference, fs=fs)
    n_common = min(len(t) for t in uc_traces.values())
    ref_trace = uc_traces[reference][:n_common]
    duration = n_common / fs

    band = (scfg.band_hz[0], min(scfg.band_hz[1], 0.45 * fs))

    for card, inv in sorted(inventories.items()):
        sync = CardSync(
            card=card, uc_channel=uc_names.get(card),
            uc_rms_v=uc_rms.get(card, 0.0),
            healthy_uc=card in healthy,
        )
        # Absolute start times give an independent, coarse cross-check.
        t_ref = inventories[reference].famos.header.start_time
        t_this = inv.famos.header.start_time
        if t_ref is not None and t_this is not None:
            sync.metadata_offset_s = (t_this - t_ref).total_seconds()

        if card == reference:
            sync.note = "reference card"
            sync.tau_sigma_s = 0.0
            report.cards[card] = sync
            continue

        # Normally the card's own cell-voltage copy is the timing probe.  When
        # that channel is dead the excitation is still present on every segment
        # channel, so the strongest segment can stand in.  It resolves the
        # whole-sample alignment - the term worth 137 degrees per sample - but
        # not the nanosecond budget, because a segment's own impedance phase
        # looks exactly like a small delay.  The uncertainty says so.
        proxy = False
        if card in uc_traces and sync.healthy_uc:
            trace = uc_traces[card][:n_common]
        elif scfg.allow_segment_proxy and inv.segment_channels:
            best_name, best_segment, best_rms, best_trace = None, None, 0.0, None
            for segment, name in inv.segment_channels.items():
                candidate = inv.famos.channel(name)
                rms = ac_rms(candidate)
                if rms > best_rms:
                    best_name, best_segment = name, segment
                    best_rms, best_trace = rms, candidate
            if best_trace is None:
                sync.ok = False
                sync.note = "no usable timing channel on this card"
                report.cards[card] = sync
                continue
            trace = best_trace[:n_common]
            proxy = True
            sync.method = "segment_proxy"
            sync.note = (
                f"cell voltage dead ({sync.uc_rms_v*1e3:.2f} mV rms); timing "
                f"taken from segment channel {best_name}, good to about "
                f"{scfg.segment_proxy_sigma_s*1e6:.0f} us only"
            )
            # De-embed the shunt before timing off it: a via shunt's L/R is
            # tens of microseconds, which the delay estimator would otherwise
            # read as the card being that much late.
            tau_shunt = 0.0
            if cfg.hf.enabled and shunt is not None and best_segment is not None \
                    and shunt.has(best_segment):
                from eis.hf import shunt_time_constant

                tau_shunt = shunt_time_constant(
                    shunt.H(best_segment, cfg.calibration.temperature_fallback_c),
                    cfg.geometry.segment_area_cm2,
                    cfg.hf.shunt_inductance_nh_per_segment.get(
                        best_segment, cfg.hf.shunt_inductance_nh
                    ),
                )
            if tau_shunt > 0:
                trace = _deembed_shunt(trace, fs, tau_shunt)
                sync.note += (
                    f"; the shunt's {tau_shunt*1e6:.1f} us time constant was "
                    f"divided out of the proxy first"
                )
        else:
            sync.ok = False
            sync.note = "no cell-voltage channel: timing cannot be measured"
            report.cards[card] = sync
            continue
        drift: DriftEstimate = estimate_drift(
            ref_trace, trace, fs, n_blocks=scfg.n_drift_blocks,
            band_hz=band, min_coherence=scfg.min_coherence,
        )
        delay: DelayEstimate = estimate_delay(
            ref_trace, trace, fs, band_hz=band, nperseg=scfg.nperseg,
            min_coherence=scfg.min_coherence, min_bins=scfg.min_bins,
            max_intercept_rad=np.inf if proxy else scfg.max_intercept_rad,
            max_lag_s=scfg.max_lag_s,
        )

        use_drift = (
            drift.ok
            and abs(drift.delay_slope_ppm) > scfg.drift_correction_threshold_ppm
            and abs(drift.delay_slope_ppm) > 3 * drift.delay_slope_sigma_ppm
        )
        if use_drift:
            sync.tau_s = drift.tau0_s
            sync.tau_sigma_s = drift.delay_slope_sigma_ppm * 1e-6 * duration
            sync.delay_slope_ppm = drift.delay_slope_ppm
            sync.delay_slope_sigma_ppm = drift.delay_slope_sigma_ppm
            sync.drift_corrected = True
            sync.method = "drift_affine"
            smear = drift.intra_window_smear_s(cfg.spectral.nperseg / fs)
            sync.note = (
                f"clock rate {drift.clock_rate_ppm:+.2f} ppm; delay moves "
                f"{drift.max_excursion_s(duration)*1e6:.1f} us over the record "
                f"({smear*1e9:.0f} ns within one analysis window)"
            )
        else:
            sync.tau_s = delay.tau_s
            sync.tau_sigma_s = delay.tau_ci_s
            sync.method = delay.method
            if drift.ok:
                sync.delay_slope_sigma_ppm = drift.delay_slope_sigma_ppm
                sync.note = f"drift {drift.delay_slope_ppm:+.3f} ppm not significant"

        # Closed-loop refinement.  A single open-loop estimate leaves a small
        # bias - a drifting card in particular, where each block's delay is
        # measured through a slightly smeared cross-spectrum.  Applying the
        # correction and re-measuring drives the residual to the noise floor,
        # and costs one extra resampling of a single channel.
        if not proxy and delay.ok:
            for _ in range(3):
                corrected = align_channel(
                    trace, fs, sync.tau_s, sync.delay_slope_ppm
                )
                a, b = valid_span(n_common, fs, sync.tau_s, sync.delay_slope_ppm)
                if b - a < scfg.nperseg * 2:
                    break
                residual = estimate_delay(
                    ref_trace[a:b], corrected[a:b], fs, band_hz=band,
                    nperseg=min(scfg.nperseg, b - a),
                    min_coherence=scfg.min_coherence, min_bins=scfg.min_bins,
                    max_intercept_rad=np.inf, max_lag_s=scfg.max_lag_s,
                )
                if not residual.ok or not np.isfinite(residual.tau_s):
                    break
                if abs(residual.tau_s) <= 0.2 * scfg.residual_tolerance_s:
                    break
                sync.tau_s += residual.tau_s

        sync.coherence = delay.coherence_median
        sync.intercept_rad = delay.intercept_rad
        sync.ok = delay.ok
        if proxy:
            sync.method = "segment_proxy"
            sync.tau_sigma_s = max(sync.tau_sigma_s, scfg.segment_proxy_sigma_s)
            # The proxy's phase-slope refinement is contaminated by the
            # segment's own impedance, so keep the coarse correlation result.
            sync.tau_s = delay.tau_coarse_s if delay.ok else sync.tau_s
            sync.ok = np.isfinite(sync.tau_s)
        if delay.note:
            sync.note = (sync.note + "; " if sync.note else "") + delay.note
        if not sync.ok:
            sync.tau_s, sync.delay_slope_ppm, sync.drift_corrected = 0.0, 0.0, False
            sync.note = (sync.note + "; " if sync.note else "") + (
                "timing unknown - this card's segments are processed on an "
                "unverified time base and marked accordingly, with the whole "
                "delay carried as uncertainty"
            )
        report.cards[card] = sync

    # --- triplet closure --------------------------------------------------
    # Only cards whose own cell-voltage copy was usable take part: the closure
    # compares three measurements of the same physical node, and a card timed
    # from a segment proxy is not measuring that node.
    closure_cards = {
        c for c, s in report.cards.items()
        if s.ok and s.healthy_uc and c in uc_traces
    }
    # Closure must compare like with like: a drifting card's delay is a
    # function of time, so every pair is evaluated at the record centre, which
    # is also what a plain whole-record estimate returns.
    t_mid = duration / 2.0
    pairwise = {
        (reference, c): s.tau_s + s.delay_slope_ppm * 1e-6 * t_mid
        for c, s in report.cards.items()
        if c != reference and c in closure_cards
    }
    if len(pairwise) >= 2:
        cards = [c for (_, c) in pairwise]
        for i in range(len(cards)):
            for j in range(i + 1, len(cards)):
                a, b = cards[i], cards[j]
                est = estimate_delay(
                    uc_traces[a][:n_common], uc_traces[b][:n_common], fs,
                    band_hz=band, nperseg=scfg.nperseg,
                    min_coherence=scfg.min_coherence, min_bins=scfg.min_bins,
                    max_intercept_rad=np.inf, max_lag_s=scfg.max_lag_s,
                )
                pairwise[(a, b)] = est.tau_s
        report.closure_s = closure_residual(pairwise)
        worst = max((abs(v) for v in report.closure_s.values()), default=0.0)
        if worst > scfg.closure_tolerance_s:
            report.passed = False
            report.notes.append(
                f"triplet closure residual {worst*1e9:.0f} ns exceeds "
                f"{scfg.closure_tolerance_s*1e9:.0f} ns - at least one delay "
                f"estimate is inconsistent"
            )

    if any(not s.ok for s in report.cards.values()):
        report.passed = False
        report.notes.append("one or more cards failed the delay measurement")
    return report, uc_traces


# ---------------------------------------------------------------------------
# Bronze product
# ---------------------------------------------------------------------------

@dataclass
class SegmentSignal:
    """One segment's current, on the reference time base, in amps."""

    segment: int
    card: int
    channel: str
    uc_from_card: int
    uc_channel: str
    current_a: np.ndarray
    voltage_v: np.ndarray
    temperature_c: float
    shunt_H: float
    timing_sigma_s: float
    status: str = STATUS_OK
    physical_units: bool = True
    note: str = ""


@dataclass
class BronzeCondition:
    """The aligned, scaled dataset one operating point produces."""

    measurement_id: str
    condition: str
    fs: float
    duration_s: float
    span: tuple[int, int]
    sync: SyncReport
    temperature: TemperatureField
    inventories: dict[int, CardInventory] = field(default_factory=dict)
    uc_source: dict[int, int] = field(default_factory=dict)
    voltage: dict[int, np.ndarray] = field(default_factory=dict)
    all_segments: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: Filled in by :meth:`segments` as it walks the plate.
    _plan: dict[int, dict] = field(default_factory=dict)

    @property
    def n_cards(self) -> int:
        return len(self.inventories)

    def segments(self) -> Iterator[SegmentSignal]:
        """Yield every segment's aligned current, one at a time."""
        for segment in self.all_segments:
            plan = self._plan.get(segment)
            if plan is None:
                continue
            yield _materialise(self, segment, plan)

    def to_frame(self) -> pd.DataFrame:
        """Per-segment ingest record - what bronze knows before any spectrum."""
        rows = []
        for segment in self.all_segments:
            plan = self._plan.get(segment, {})
            rows.append({
                "measurement_id": self.measurement_id,
                "condition": self.condition,
                "segment": segment,
                "card": plan.get("card"),
                "channel": plan.get("channel"),
                "uc_from_card": plan.get("uc_card"),
                "uc_channel": plan.get("uc_channel"),
                "bronze_status": plan.get("status", STATUS_CHANNEL_FAULT),
                "physical_units": plan.get("physical_units", False),
                "temperature_c": plan.get("temperature_c"),
                "shunt_H_V_cm2_per_A": plan.get("shunt_H"),
                "channel_tau_ns": plan.get("tau") * 1e9 if plan.get("tau") is not None
                else None,
                "timing_sigma_ns": plan.get("timing_sigma_s", np.nan) * 1e9,
                "note": plan.get("note", ""),
            })
        return pd.DataFrame(rows)


def _materialise(bronze: BronzeCondition, segment: int, plan: dict) -> SegmentSignal:
    """Read, align and scale one segment channel."""
    inv = bronze.inventories[plan["card"]]
    lo, hi = bronze.span
    n_common = plan["n_common"]
    raw = inv.famos.channel(plan["channel"])[:n_common]
    if plan["polarity"] != 1:
        raw = raw * plan["polarity"]
    if abs(plan["tau"]) > 1e-12 or abs(plan["slope_ppm"]) > 1e-6:
        raw = align_channel(raw, bronze.fs, plan["tau"], plan["slope_ppm"])
    raw = raw[lo:hi]
    if plan["notch"]:
        raw = apply_notch(raw, bronze.fs, plan["notch"], plan["notch_q"])

    if plan["shunt"] is not None:
        density = shunt_voltage_to_current_density(
            raw, segment, plan["shunt"], plan["temperature_c"]
        )
    else:
        density = raw                               # shunt volts, not amps
    current = density * plan["segment_area_cm2"]

    return SegmentSignal(
        segment=segment, card=plan["card"], channel=plan["channel"],
        uc_from_card=plan["uc_card"], uc_channel=plan["uc_channel"],
        current_a=current, voltage_v=bronze.voltage[plan["uc_card"]],
        temperature_c=plan["temperature_c"], shunt_H=plan["shunt_H"],
        timing_sigma_s=plan["timing_sigma_s"], status=plan["status"],
        physical_units=plan["physical_units"], note=plan.get("note", ""),
    )


# ---------------------------------------------------------------------------
# Stage 2: build the aligned dataset
# ---------------------------------------------------------------------------

def run_bronze(
    cfg: PipelineConfig,
    condition: str,
    card_files: dict[int, str],
    shunt: ShuntCalibration | None = None,
    temp_cal: TemperatureCalibration | None = None,
    log=print,
) -> BronzeCondition:
    """Read one operating point and produce the aligned, scaled dataset."""
    inventories = {c: inventory_card(p, c) for c, p in sorted(card_files.items())}
    for card, inv in inventories.items():
        log(f"  card {card}: {len(inv.segment_channels)} segments, "
            f"UC={inv.uc_channels or '-'}, temp={len(inv.temp_channels)}, "
            f"{inv.n_samples:,} samples @ {inv.famos.fs:.0f} Hz")

    fs_values = {inv.famos.fs for inv in inventories.values()}
    tolerance = cfg.acquisition.fs_tolerance_ppm * 1e-6 * max(fs_values)
    if max(fs_values) - min(fs_values) > tolerance:
        raise RuntimeError(f"cards disagree on the sampling rate: {sorted(fs_values)}")
    fs = float(np.median(list(fs_values)))

    # ---- synchronisation ----------------------------------------------
    sync, uc_traces = measure_sync(inventories, cfg, shunt)
    log(f"\n  synchronisation (reference card {sync.reference_card}):")
    for card, s in sorted(sync.cards.items()):
        marker = "ok " if s.ok else "FAIL"
        log(f"    [{marker}] card {card}: tau = {s.tau_s*1e9:+9.1f} ns "
            f"+/- {s.tau_sigma_s*1e9:6.1f} ns | "
            f"{s.delay_slope_ppm:+7.3f} ppm | gamma2 = {s.coherence:.4f}"
            + (f" | {s.note}" if s.note else ""))
    for triplet, residual in sync.closure_s.items():
        log(f"    closure {triplet}: {residual*1e9:+.1f} ns")

    # ---- pick the cell voltage for each card ---------------------------
    strategy = cfg.sync.uc_strategy
    uc_source: dict[int, int] = {}
    for card, s in sync.cards.items():
        if strategy == "reference":
            uc_source[card] = sync.reference_card
        elif strategy == "same_card":
            uc_source[card] = card if s.uc_channel else sync.reference_card
        else:                                        # auto
            uc_source[card] = (
                card if (s.uc_channel and s.healthy_uc) else sync.reference_card
            )
    cross_card = {c for c, src in uc_source.items() if src != c}
    sync.alignment_applied = bool(cross_card)
    if cross_card:
        log(f"\n  cards paired with another card's cell voltage: "
            f"{sorted(cross_card)} -> full time-base alignment applied")
    else:
        log("\n  every card uses its own cell-voltage copy: bulk inter-card "
            "skew cancels in the ratio (verified below)")

    # ---- per-channel timing model ---------------------------------------
    # Every channel has an intrinsic delay made of its card's bulk offset plus
    # its own offset within the card.  What matters for an impedance is the
    # *difference* between the segment channel and the cell-voltage channel it
    # is divided by, so that difference is what gets applied - to the segment
    # only, leaving the cell voltage as the local time reference.  When both
    # sit on the same card with no channel table, the difference is zero and
    # the bulk card offset cancels exactly, as it should.
    delay_table = cfg.acquisition.channel_delay_table
    slope_of = {
        c: (s.delay_slope_ppm if s.drift_corrected else 0.0)
        for c, s in sync.cards.items()
    }

    def tau_total(card: int, channel: str) -> float:
        return sync.cards[card].tau_s + delay_table.get(card, {}).get(channel, 0.0)

    shifts: list[tuple[float, float]] = []
    for card, inv in inventories.items():
        src = uc_source[card]
        uc_name = sync.cards[src].uc_channel or ""
        tau_uc, slope_uc = tau_total(src, uc_name), slope_of[src]
        for channel in inv.segment_channels.values():
            shifts.append(
                (tau_total(card, channel) - tau_uc, slope_of[card] - slope_uc)
            )
    for card, s in sync.cards.items():                # UC traces for verification
        shifts.append((s.tau_s, slope_of[card]))

    # ---- common valid sample span --------------------------------------
    n_common = min(inv.n_samples for inv in inventories.values())
    lo, hi = 0, n_common
    for tau, slope in shifts:
        if abs(tau) > 1e-12 or abs(slope) > 1e-6:
            a, b = valid_span(n_common, fs, tau, slope)
            lo, hi = max(lo, a), min(hi, b)
    if hi - lo < cfg.spectral.nperseg_min * 4:
        raise RuntimeError(
            f"only {hi - lo} samples survive alignment trimming; "
            f"need at least {cfg.spectral.nperseg_min * 4}"
        )
    duration = (hi - lo) / fs
    log(f"  common analysis span: samples [{lo}, {hi}) = {duration:.1f} s")

    notch = cfg.spectral.notch_freqs_hz if cfg.spectral.apply_notch else []

    def prepare(trace: np.ndarray, tau: float, slope: float) -> np.ndarray:
        if abs(tau) > 1e-12 or abs(slope) > 1e-6:
            trace = align_channel(trace, fs, tau, slope)
        trace = trace[lo:hi]
        if notch:
            trace = apply_notch(trace, fs, notch, cfg.spectral.notch_q)
        return trace

    # ---- cell-voltage traces --------------------------------------------
    # Two versions: one left on each card's own time base (the denominator for
    # that card's segments) and one brought onto the reference base (used only
    # to verify that the alignment actually worked).
    local_uc = {
        card: prepare(uc_traces[card][:n_common], 0.0, 0.0)
        for card in set(uc_source.values()) if card in uc_traces
    }
    verify_uc = {
        card: prepare(uc_traces[card][:n_common], sync.cards[card].tau_s, slope_of[card])
        for card in uc_traces
    }

    # ---- residual-skew verification --------------------------------------
    # This is the check that turns "we corrected the timing" into "we measured
    # that the timing is now correct".
    for card, s in sync.cards.items():
        if card == sync.reference_card or card not in verify_uc:
            continue
        if not s.healthy_uc:
            # The channel that would do the verifying is the one that failed.
            s.note = (s.note + "; " if s.note else "") + (
                "alignment cannot be verified: this card has no working "
                "cell-voltage channel to compare against the reference"
            )
            log(f"    verify card {card}: not possible (dead cell-voltage channel)")
            continue
        check = estimate_delay(
            verify_uc[sync.reference_card], verify_uc[card], fs,
            band_hz=(cfg.sync.band_hz[0], min(cfg.sync.band_hz[1], 0.45 * fs)),
            nperseg=min(cfg.sync.nperseg, hi - lo), min_coherence=cfg.sync.min_coherence,
            min_bins=cfg.sync.min_bins, max_intercept_rad=np.inf,
        )
        s.residual_tau_s = check.tau_s
        # A proxy-derived delay can only be held to the accuracy it claims.
        budget = cfg.sync.residual_tolerance_s
        if s.method == "segment_proxy":
            budget = max(budget, 3.0 * s.tau_sigma_s)
        within = abs(check.tau_s) <= budget
        if not within:
            s.ok = False
            sync.passed = False
            s.note = (s.note + "; " if s.note else "") + (
                f"residual skew {check.tau_s*1e9:+.0f} ns after alignment "
                f"exceeds the {budget*1e9:.0f} ns budget"
            )
        log(f"    verify card {card}: residual = {check.tau_s*1e9:+7.1f} ns "
            f"({'within' if within else 'OVER'} the {budget*1e9:.0f} ns budget)")

    # ---- temperature ----------------------------------------------------
    sensor_readings: dict[str, float] = {}
    if temp_cal is not None:
        index = 0
        for card, inv in sorted(inventories.items()):
            for name in inv.temp_channels:
                if index < len(temp_cal.c0):
                    trace = inv.famos.channel(name)[lo:hi]
                    sensor_readings[f"card{card}:{name}"] = sensor_voltage_to_celsius(
                        trace, index, temp_cal
                    )
                index += 1

    all_segments = sorted({s for inv in inventories.values() for s in inv.segment_channels})
    if cfg.calibration.use_measured_temperature and sensor_readings:
        temperature = build_temperature_field(
            sensor_readings, None, cfg.geometry.segment_coords, all_segments,
            cfg.calibration.temperature_fallback_c,
            tuple(cfg.calibration.temperature_valid_range_c),
        )
    else:
        temperature = TemperatureField(
            per_segment_c={s: cfg.calibration.temperature_fallback_c for s in all_segments},
            fallback_c=cfg.calibration.temperature_fallback_c, degraded=True,
            note="measured temperature disabled in the configuration"
            if not cfg.calibration.use_measured_temperature
            else "no temperature calibration file supplied",
        )
    log(f"  temperature: {temperature.mean_c:.1f} degC "
        f"({'DEGRADED - ' if temperature.degraded else ''}{temperature.note})")

    bronze = BronzeCondition(
        measurement_id=cfg.measurement_id, condition=condition, fs=fs,
        duration_s=duration, span=(lo, hi), sync=sync, temperature=temperature,
        inventories=inventories, uc_source=uc_source, voltage=local_uc,
        all_segments=all_segments,
    )

    # ---- per-segment plan ------------------------------------------------
    polarity_table = cfg.acquisition.polarity
    counts: dict[str, int] = {}
    for card, inv in sorted(inventories.items()):
        src = uc_source[card]
        uc_name = sync.cards[src].uc_channel or ""
        tau_uc, slope_uc = tau_total(src, uc_name), slope_of[src]
        timing_sigma = float(np.hypot(
            sync.cards[card].timing_sigma_s, sync.cards[src].timing_sigma_s
        ))
        card_ok = sync.cards[card].ok and sync.cards[src].ok

        for segment, channel in sorted(inv.segment_channels.items()):
            status, note, physical = STATUS_OK, "", True
            if not card_ok:
                status = STATUS_TIMING_UNVERIFIED
                note = (
                    f"card {card} time base not verified "
                    f"({sync.cards[card].note or 'no detail'}); the impedance is "
                    f"still computed and its phase uncertainty widened to cover "
                    f"the unknown delay"
                )
            temperature_c = temperature.at(segment)
            if shunt is not None and shunt.has(segment):
                shunt_H = shunt.H(segment, temperature_c)
                shunt_obj: ShuntCalibration | None = shunt
            else:
                shunt_H, shunt_obj, physical = 1.0, None, False
                if status == STATUS_OK:
                    status = STATUS_NO_CALIBRATION
                note = (note + "; " if note else "") + (
                    "no via-shunt calibration for this segment; the impedance "
                    "is in shunt volts per amp, not ohms"
                )
            counts[status] = counts.get(status, 0) + 1
            bronze._plan[segment] = {
                "card": card, "channel": channel, "uc_card": src,
                "uc_channel": sync.cards[src].uc_channel or "?",
                "tau": tau_total(card, channel) - tau_uc,
                "slope_ppm": slope_of[card] - slope_uc,
                "polarity": polarity_table.get(card, {}).get(
                    channel, cfg.acquisition.default_polarity
                ),
                "notch": notch, "notch_q": cfg.spectral.notch_q,
                "shunt": shunt_obj, "shunt_H": shunt_H,
                "temperature_c": temperature_c,
                "segment_area_cm2": cfg.geometry.segment_area_cm2,
                "timing_sigma_s": timing_sigma,
                "status": status, "physical_units": physical,
                "note": note, "n_common": n_common,
            }

    log(f"  segments ingested: {len(bronze._plan)} "
        f"({', '.join(f'{v} {k}' for k, v in sorted(counts.items()))})")
    return bronze
