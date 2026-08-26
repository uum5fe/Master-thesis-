"""End-to-end pipeline: FAMOS files -> synchronised signals -> Z(f) -> models.

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

So the pipeline always measures, always reports, and aligns whenever a segment
is paired with a cell voltage from another card.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from eis import __version__
from eis.calibrate import (
    ShuntCalibration, TemperatureCalibration, TemperatureField,
    build_temperature_field, load_shunt_calibration, load_temperature_calibration,
)
from eis.config import PipelineConfig
from eis.io.famos import FamosFile, classify_channel, discover_files, open_famos
from eis.model.ecm import ECMFit, select_model
from eis.spectra import (
    SpectralResult, apply_notch, impedance_synchronous, impedance_welch,
)
from eis.sync.drift import DriftEstimate, estimate_drift
from eis.sync.resample import (
    advance_affine, fractional_delay_fft, integer_and_fractional, valid_span,
)
from eis.sync.skew import DelayEstimate, closure_residual, estimate_delay
from eis.validate import KKResult, admittance_sum, lin_kk, stationarity_split_half


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
# Segment result
# ---------------------------------------------------------------------------

@dataclass
class SegmentResult:
    segment: int
    card: int
    uc_channel: str
    uc_from_card: int
    spectrum: SpectralResult
    temperature_c: float
    shunt_H: float
    rs_hf_ohm: float
    status: str = "ok"
    reason: str = ""
    kk: KKResult | None = None
    ecm: ECMFit | None = None
    ecm_all: dict[str, ECMFit] = field(default_factory=dict)


@dataclass
class ConditionResult:
    measurement_id: str
    condition: str
    fs: float
    duration_s: float
    sync: SyncReport
    temperature: TemperatureField
    segments: dict[int, SegmentResult] = field(default_factory=dict)
    rejected: dict[int, str] = field(default_factory=dict)
    plate: object | None = None
    provenance: dict[str, str] = field(default_factory=dict)
    elapsed_s: float = 0.0

    # -- tables ----------------------------------------------------------
    def impedance_frame(self, segment_area_cm2: float) -> pd.DataFrame:
        rows = []
        for seg, r in sorted(self.segments.items()):
            s = r.spectrum
            asr = segment_area_cm2 * 1e3            # Ohm -> mOhm*cm^2
            for i in range(len(s.f)):
                z = s.Z[i]
                rows.append({
                    "measurement_id": self.measurement_id,
                    "condition": self.condition,
                    "segment": seg,
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
                    "n_eff": s.n_eff,
                    "estimator": s.estimator,
                    "method": s.method,
                    **self.provenance,
                })
        return pd.DataFrame(rows)

    def segment_frame(self, segment_area_cm2: float) -> pd.DataFrame:
        rows = []
        asr = segment_area_cm2 * 1e3
        for seg, r in sorted(self.segments.items()):
            row = {
                "measurement_id": self.measurement_id,
                "condition": self.condition,
                "segment": seg,
                "card": r.card,
                "uc_channel": r.uc_channel,
                "uc_from_card": r.uc_from_card,
                "status": r.status,
                "reason": r.reason,
                "n_points": len(r.spectrum.f),
                "n_windows_used": r.spectrum.n_windows_used,
                "n_windows_total": r.spectrum.n_windows_total,
                "median_coherence": float(np.median(r.spectrum.coherence))
                if len(r.spectrum.coherence) else np.nan,
                "median_sigma_rel": float(np.median(r.spectrum.sigma_rel))
                if len(r.spectrum.sigma_rel) else np.nan,
                "temperature_c": r.temperature_c,
                "shunt_H_V_cm2_per_A": r.shunt_H,
                "rs_hf_mohm_cm2": r.rs_hf_ohm * asr,
            }
            if r.kk is not None:
                row |= {
                    "kk_pass": r.kk.passed, "kk_elements": r.kk.n_elements,
                    "kk_mu": r.kk.mu, "kk_max_residual_pct": r.kk.max_residual_pct,
                    "kk_rms_residual_pct": r.kk.rms_residual_pct,
                    "kk_shape": r.kk.shape_class,
                }
            if r.ecm is not None:
                row |= {
                    "ecm_model": r.ecm.model,
                    "ecm_chi2_reduced": r.ecm.chi2_reduced,
                    "ecm_aicc": r.ecm.aicc,
                    "ecm_poorly_determined": ",".join(r.ecm.poorly_determined),
                }
                for name, value in r.ecm.params.items():
                    scale = asr if name.startswith(("Rs", "Rct", "Aw")) else 1.0
                    unit = "_mohm_cm2" if scale != 1.0 else ""
                    row[f"ecm_{name}{unit}"] = value * scale
                    row[f"ecm_{name}{unit}_sigma"] = r.ecm.sigma.get(name, np.nan) * scale
                row["ecm_rp_mohm_cm2"] = r.ecm.r_polarisation * asr
            rows.append(row)
        for seg, reason in sorted(self.rejected.items()):
            rows.append({
                "measurement_id": self.measurement_id, "condition": self.condition,
                "segment": seg, "status": "rejected", "reason": reason,
            })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _git_sha(default: str = "unknown") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True, timeout=5,
        ).strip()
    except Exception:
        return default


def _ac_rms(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean((x - x.mean()) ** 2)))


def align_channel(
    x: np.ndarray, fs: float, tau_s: float, delay_slope_ppm: float = 0.0
) -> np.ndarray:
    """Bring ``x`` onto the reference time base.

    ``tau_s`` is the measured delay of this channel (positive = it lags), so
    the channel has to be advanced by that amount.  A constant offset is
    applied as an exact integer shift plus an FFT phase ramp; a drifting one
    goes through the polyphase resampler.
    """
    if abs(delay_slope_ppm) > 1e-6:
        return advance_affine(x, fs, tau_s, delay_slope_ppm)
    whole, frac = integer_and_fractional(tau_s, fs)
    out = np.asarray(x, float)
    if whole:                                   # advance == take later samples
        out = np.roll(out, -whole)
        if whole > 0:
            out[-whole:] = out[-whole - 1]
        else:
            out[:-whole] = out[-whole]
    if abs(frac) > 1e-6:
        out = fractional_delay_fft(out, -frac / fs, fs)
    return out


# ---------------------------------------------------------------------------
# Stage 1: measure the synchronisation
# ---------------------------------------------------------------------------

def measure_sync(
    inventories: dict[int, CardInventory], cfg: PipelineConfig
) -> tuple[SyncReport, dict[int, np.ndarray]]:
    """Measure inter-card skew and drift from the shared cell-voltage channel.

    Returns the report and the (unaligned) cell-voltage trace of each card,
    which the caller reuses so the files are not read twice.
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
            rms = _ac_rms(trace)
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
            best_name, best_rms, best_trace = None, 0.0, None
            for name in inv.segment_channels.values():
                candidate = inv.famos.channel(name)
                rms = _ac_rms(candidate)
                if rms > best_rms:
                    best_name, best_rms, best_trace = name, rms, candidate
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
                "timing unknown - this card's segments are rejected rather "
                "than processed with an unverified time base"
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
# Stage 2: process one condition
# ---------------------------------------------------------------------------

def run_condition(
    cfg: PipelineConfig,
    condition: str,
    card_files: dict[int, str],
    shunt: ShuntCalibration | None = None,
    temp_cal: TemperatureCalibration | None = None,
    reference_spectrum: tuple[np.ndarray, np.ndarray] | None = None,
    verbose: bool = True,
) -> ConditionResult:
    """Full processing of one operating point."""
    t_start = time.time()
    log = print if verbose else (lambda *a, **k: None)

    log(f"\n{'=' * 72}\n  {cfg.measurement_id} / {condition}\n{'=' * 72}")

    # ---- inventory ----------------------------------------------------
    inventories = {c: inventory_card(p, c) for c, p in sorted(card_files.items())}
    for card, inv in inventories.items():
        log(f"  card {card}: {len(inv.segment_channels)} segments, "
            f"UC={inv.uc_channels or '-'}, temp={len(inv.temp_channels)}, "
            f"{inv.n_samples:,} samples @ {inv.famos.fs:.0f} Hz")

    fs_values = {inv.famos.fs for inv in inventories.values()}
    if max(fs_values) - min(fs_values) > cfg.acquisition.fs_tolerance_ppm * 1e-6 * max(fs_values):
        raise RuntimeError(f"cards disagree on the sampling rate: {sorted(fs_values)}")
    fs = float(np.median(list(fs_values)))

    # ---- synchronisation ----------------------------------------------
    sync, uc_traces = measure_sync(inventories, cfg)
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
            uc_source[card] = card if (s.uc_channel and s.healthy_uc) else sync.reference_card
    cross_card = {c for c, src in uc_source.items() if src != c}
    sync.alignment_applied = bool(cross_card)
    if cross_card:
        log(f"\n  cards paired with another card's cell voltage: "
            f"{sorted(cross_card)} -> full time-base alignment applied")
    else:
        log("\n  every card uses its own cell-voltage copy: bulk inter-card "
            "skew cancels in the ratio (verified above)")

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
            shifts.append((tau_total(card, channel) - tau_uc, slope_of[card] - slope_uc))
    for card, s in sync.cards.items():                # UC traces for verification
        shifts.append((s.tau_s, slope_of[card]))

    # ---- common valid sample span --------------------------------------
    n_common = min(inv.n_samples for inv in inventories.values())
    lo, hi = 0, n_common
    for tau, slope in shifts:
        if abs(tau) > 1e-12 or abs(slope) > 1e-6:
            a, b = valid_span(n_common, fs, tau, slope)
            lo, hi = max(lo, a), min(hi, b)
    if hi - lo < cfg.spectral.nperseg * 2:
        raise RuntimeError(
            f"only {hi - lo} samples survive alignment trimming; "
            f"need at least {cfg.spectral.nperseg * 2}"
        )
    duration = (hi - lo) / fs
    log(f"  common analysis span: samples [{lo}, {hi}) = {duration:.1f} s")

    def prepare(trace: np.ndarray, tau: float, slope: float) -> np.ndarray:
        if abs(tau) > 1e-12 or abs(slope) > 1e-6:
            trace = align_channel(trace, fs, tau, slope)
        trace = trace[lo:hi]
        if cfg.spectral.apply_notch and cfg.spectral.notch_freqs_hz:
            trace = apply_notch(
                trace, fs, cfg.spectral.notch_freqs_hz, cfg.spectral.notch_q
            )
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
                    volts = float(np.mean(inv.famos.channel(name)[lo:hi]))
                    sensor_readings[f"card{card}:{name}"] = temp_cal.to_celsius(index, volts)
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

    result = ConditionResult(
        measurement_id=cfg.measurement_id, condition=condition, fs=fs,
        duration_s=duration, sync=sync, temperature=temperature,
        provenance={
            "pipeline_version": __version__,
            "param_hash": cfg.param_hash,
            "git_sha": _git_sha(),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "channel_delay_table": cfg.acquisition.channel_delay_table_version,
        },
    )

    # ---- per-segment impedance -----------------------------------------
    scfg = cfg.spectral
    f_max = min(scfg.f_max_hz, 0.45 * fs)
    use_synchronous = scfg.method == "synchronous" or (
        scfg.method == "auto" and scfg.base_frequency_hz and scfg.excitation_tones_hz
    )
    log(f"\n  impedance: {'synchronous DFT' if use_synchronous else 'coherence-gated Welch'}"
        f", estimator={scfg.estimator}, band=[{scfg.f_min_hz}, {f_max:.0f}] Hz")

    polarity_table = cfg.acquisition.polarity

    for card, inv in sorted(inventories.items()):
        src = uc_source[card]
        uc_name = sync.cards[src].uc_channel or ""
        tau_uc, slope_uc = tau_total(src, uc_name), slope_of[src]

        # Refuse to produce impedances on an unverified time base: a wrong
        # alignment does not look wrong, it looks like a different spectrum.
        if not sync.cards[card].ok or not sync.cards[src].ok:
            for segment in inv.segment_channels:
                result.rejected[segment] = (
                    f"card {card} timing not established "
                    f"({sync.cards[card].note or 'no detail'})"
                )
            continue
        voltage = local_uc[src]

        for segment, channel in sorted(inv.segment_channels.items()):
            try:
                if shunt is not None and not shunt.has(segment):
                    result.rejected[segment] = "no shunt calibration for this segment"
                    continue

                raw = inv.famos.channel(channel)[:n_common]

                # polarity from the wiring map, not from a vote on the data
                sign = polarity_table.get(card, {}).get(
                    channel, cfg.acquisition.default_polarity
                )
                if sign != 1:
                    raw = raw * sign

                # Delay of this channel relative to the cell-voltage channel it
                # is divided by.  Channel-level skew inside a card does NOT
                # cancel in the ratio, so it is applied even when the segment
                # uses its own card's cell voltage.
                raw = prepare(
                    raw,
                    tau_total(card, channel) - tau_uc,
                    slope_of[card] - slope_uc,
                )

                # shunt voltage -> local current density -> segment current
                t_seg = temperature.at(segment)
                H = shunt.H(segment, t_seg) if shunt is not None else 1.0
                current_density = raw / H                     # A/cm^2
                current = current_density * cfg.geometry.segment_area_cm2   # A

                if use_synchronous:
                    spectrum = impedance_synchronous(
                        current, voltage, fs,
                        base_frequency_hz=float(scfg.base_frequency_hz),
                        tones_hz=list(scfg.excitation_tones_hz or []),
                        periods=scfg.periods_per_window, estimator=scfg.estimator,
                        gate=scfg.gate_windows, detrend=scfg.detrend,
                        amplitude_mad_k=scfg.amplitude_mad_k,
                        phase_outlier_rad=scfg.phase_outlier_rad,
                        max_rejected_fraction=scfg.max_rejected_fraction,
                    )
                else:
                    spectrum = impedance_welch(
                        current, voltage, fs, nperseg=scfg.nperseg,
                        noverlap=scfg.noverlap, window=scfg.window,
                        detrend=scfg.detrend, f_min=scfg.f_min_hz, f_max=f_max,
                        estimator=scfg.estimator, gate=scfg.gate_windows,
                        amplitude_mad_k=scfg.amplitude_mad_k,
                        phase_outlier_rad=scfg.phase_outlier_rad,
                        max_rejected_fraction=scfg.max_rejected_fraction,
                    )

                gated = spectrum.gate(scfg.min_coherence)
                if len(gated.f) < scfg.min_points_per_segment:
                    result.rejected[segment] = (
                        f"only {len(gated.f)} of {len(spectrum.f)} points reach "
                        f"gamma^2 >= {scfg.min_coherence}"
                    )
                    continue

                n_hf = max(2, len(gated.f) // 5)
                rs_hf = float(np.median(np.sort(gated.Z.real[np.argsort(gated.f)][-n_hf:])))

                result.segments[segment] = SegmentResult(
                    segment=segment, card=card,
                    uc_channel=sync.cards[src].uc_channel or "?", uc_from_card=src,
                    spectrum=gated, temperature_c=t_seg, shunt_H=H, rs_hf_ohm=rs_hf,
                )
            except Exception as exc:                       # keep going
                result.rejected[segment] = f"{type(exc).__name__}: {exc}"

        del inv.famos                                       # release the memmap

    log(f"  segments: {len(result.segments)} usable, {len(result.rejected)} rejected")

    # ---- validation ------------------------------------------------------
    if cfg.validation.run_lin_kk:
        for r in result.segments.values():
            r.kk = lin_kk(
                r.spectrum.f, r.spectrum.Z, r.spectrum.sigma_real,
                max_elements=cfg.validation.kk_max_elements,
                mu_target=cfg.validation.kk_mu_target,
                max_residual_pct=cfg.validation.kk_max_residual_pct,
            )
        passed = sum(1 for r in result.segments.values() if r.kk and r.kk.passed)
        shapes: dict[str, int] = {}
        for r in result.segments.values():
            if r.kk:
                shapes[r.kk.shape_class] = shapes.get(r.kk.shape_class, 0) + 1
        log(f"  Kramers-Kronig: {passed}/{len(result.segments)} pass "
            f"(<= {cfg.validation.kk_max_residual_pct}% residual); "
            f"residual shapes: {shapes}")

    if cfg.validation.run_admittance_sum and result.segments:
        result.plate = admittance_sum(
            {s: (r.spectrum.f, r.spectrum.Z) for s, r in result.segments.items()},
            cfg.geometry.segment_area_cm2, cfg.geometry.cell_area_cm2,
            reference_spectrum,
        )
        if result.plate is not None and getattr(result.plate, "n_segments", 0):
            note = f"  plate admittance sum: {result.plate.n_segments} segments"
            if result.plate.median_relative_difference is not None:
                note += (f"; median deviation from the reference instrument "
                         f"{result.plate.median_relative_difference:.1%}")
            log(note)

    # ---- modelling --------------------------------------------------------
    if cfg.model.run_ecm:
        fitted = 0
        for r in result.segments.values():
            best, allfits = select_model(
                r.spectrum.f, r.spectrum.Z, cfg.model.models, r.spectrum.sigma_real,
                n_starts=cfg.model.n_starts, max_nfev=cfg.model.max_nfev,
                max_relative_sigma=cfg.model.max_relative_sigma,
            )
            r.ecm, r.ecm_all = best, allfits
            fitted += best is not None
        chosen: dict[str, int] = {}
        for r in result.segments.values():
            if r.ecm:
                chosen[r.ecm.model] = chosen.get(r.ecm.model, 0) + 1
        log(f"  ECM: {fitted}/{len(result.segments)} fitted; selected {chosen}")

    result.elapsed_s = time.time() - t_start
    log(f"  done in {result.elapsed_s:.1f} s")
    return result


# ---------------------------------------------------------------------------
# Stage 3: batch over conditions
# ---------------------------------------------------------------------------

def run_measurement(
    cfg: PipelineConfig,
    reference_spectra: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    write: bool = True,
    verbose: bool = True,
) -> dict[str, ConditionResult]:
    """Discover, process and persist every requested condition."""
    log = print if verbose else (lambda *a, **k: None)

    discovered = discover_files(
        cfg.raw_dir, cfg.acquisition.discovery_regex, cfg.measurement_id or None
    )
    if not discovered:
        raise FileNotFoundError(
            f"no files under {cfg.raw_dir} match {cfg.acquisition.discovery_regex!r}"
        )
    conditions = cfg.conditions or sorted(discovered)
    log(f"discovered conditions: {sorted(discovered)}  ->  processing {conditions}")

    shunt = (
        load_shunt_calibration(cfg.calibration.shunt_csv)
        if cfg.calibration.shunt_csv else None
    )
    temp_cal = (
        load_temperature_calibration(cfg.calibration.temperature_csv)
        if cfg.calibration.temperature_csv else None
    )
    if shunt is None:
        log("WARNING: no shunt calibration configured - impedances will be in "
            "shunt volts per amp, not physical ohms")

    results: dict[str, ConditionResult] = {}
    for condition in conditions:
        if condition not in discovered:
            log(f"  skipping {condition}: no files")
            continue
        results[condition] = run_condition(
            cfg, condition, discovered[condition], shunt, temp_cal,
            (reference_spectra or {}).get(condition), verbose=verbose,
        )

    if write and results:
        write_outputs(cfg, results, verbose=verbose)
    return results


def write_outputs(
    cfg: PipelineConfig, results: dict[str, ConditionResult], verbose: bool = True
) -> dict[str, Path]:
    """Persist impedance, per-segment and synchronisation tables."""
    log = print if verbose else (lambda *a, **k: None)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    area = cfg.geometry.segment_area_cm2

    impedance = pd.concat(
        [r.impedance_frame(area) for r in results.values()], ignore_index=True
    )
    segments = pd.concat(
        [r.segment_frame(area) for r in results.values()], ignore_index=True
    )
    sync = pd.concat(
        [r.sync.to_frame().assign(condition=c) for c, r in results.items()],
        ignore_index=True,
    )

    paths: dict[str, Path] = {}
    for name, frame in (
        ("impedance", impedance), ("segments", segments), ("sync", sync)
    ):
        path = out / f"{name}.parquet"
        try:
            frame.to_parquet(path, index=False)
        except Exception:                            # no pyarrow -> CSV
            path = out / f"{name}.csv"
            frame.to_csv(path, index=False)
        paths[name] = path
        log(f"  wrote {path}  ({len(frame):,} rows)")

    config_path = out / "config.yaml"
    try:
        cfg.save(config_path)
        paths["config"] = config_path
    except Exception:
        pass
    return paths
