"""Configuration objects for the EIS pipeline.

Every numeric constant that influences a result lives here, is serialisable to
YAML, and contributes to :meth:`PipelineConfig.param_hash` so that any stored
spectrum can be traced back to the exact parameters that produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Acquisition / hardware
# ---------------------------------------------------------------------------

@dataclass
class AcquisitionConfig:
    """Describes how the raw files are laid out and what the hardware does."""

    #: Nominal sampling rate [Hz]. Verified against the FAMOS header; a
    #: mismatch larger than ``fs_tolerance_ppm`` is a hard error.
    fs_nominal: float = 10_000.0
    fs_tolerance_ppm: float = 1000.0

    #: ``str.format`` template used to locate the file of one card.
    #: Available fields: measurement_id, condition, card, test.
    filename_template: str = (
        "Leepa_{measurement_id}_Current_{condition}_Test_{test}_Karte_{card}.DAT"
    )
    test_index: str = "01"

    #: Regex used for auto-discovery when no explicit card list is given.
    #: Must expose named groups ``condition`` and ``card``.
    discovery_regex: str = (
        r"Leepa_(?P<measurement_id>[^_]+)_Current_(?P<condition>.+?)"
        r"_Test_(?P<test>\d+)_Karte_(?P<card>\d+)\.DAT"
    )

    #: Per-(card, channel-name) constant delay [s] relative to that card's own
    #: time base, from a dedicated sync-calibration measurement.  Structure:
    #: ``{card: {channel_name: seconds}}``.  Empty means "simultaneous
    #: sampling assumed"; the assumption is recorded in the output.
    channel_delay_table: dict[int, dict[str, float]] = field(default_factory=dict)
    channel_delay_table_version: str = "assumed-simultaneous"

    #: Wiring polarity per (card, channel).  ``-1`` inverts the raw signal.
    #: Resolves the sign of Z once, from the wiring, instead of by a per-run
    #: majority vote over the data.
    polarity: dict[int, dict[str, int]] = field(default_factory=dict)
    default_polarity: int = 1


# ---------------------------------------------------------------------------
# Cell geometry and calibration
# ---------------------------------------------------------------------------

@dataclass
class GeometryConfig:
    """Plate and segment geometry."""

    segment_area_cm2: float = 4.235
    cell_area_cm2: float = 304.92
    plate_w_mm: float = 252.0
    plate_h_mm: float = 121.0
    n_segments: int = 80

    #: ``{segment_number: (x_cm, y_cm, half_width_cm, half_height_cm)}``.
    segment_coords: dict[int, tuple[float, float, float, float]] = field(
        default_factory=dict
    )

    @property
    def plate_w_cm(self) -> float:
        return self.plate_w_mm / 10.0

    @property
    def plate_h_cm(self) -> float:
        return self.plate_h_mm / 10.0


@dataclass
class CalibrationConfig:
    """Via-shunt and temperature-sensor calibration."""

    #: CSV, ``;``-separated, no header, one row per segment: ``c0;c1``.
    #: Shunt transfer function  H(T) = c0 + 1e-3 * c1 * T   [V*cm^2/A].
    shunt_csv: str | None = None
    #: CSV, ``;``-separated, no header, one row per temperature sensor:
    #: ``c0;c1``.   T [degC] = (V - c0) / c1.
    temperature_csv: str | None = None

    #: Used only when no usable temperature channel exists.  Flagged in the
    #: output as a degraded mode rather than silently applied.
    temperature_fallback_c: float = 58.4
    temperature_valid_range_c: tuple[float, float] = (0.0, 200.0)

    #: If True, evaluate H at the temperature measured for each segment
    #: (sensors interpolated over the plate).  If False, use the fallback for
    #: every segment.  ``True`` is the accurate setting.
    use_measured_temperature: bool = True


# ---------------------------------------------------------------------------
# Synchronisation
# ---------------------------------------------------------------------------

@dataclass
class SyncConfig:
    """Inter-card synchronisation parameters.

    The delay between cards is *measured* from the copy of the cell voltage
    that each card records, not assumed.
    """

    #: Frequency band used for the cross-spectral phase-slope fit [Hz].
    #: Must sit where the shared reference signal has high coherence.
    band_hz: tuple[float, float] = (20.0, 4000.0)
    #: Window length for the skew cross-spectrum.  Long windows give a finer
    #: phase-slope fit; must stay well below the record length.
    nperseg: int = 32_768
    #: Only bins above this coherence enter the phase-slope fit.
    min_coherence: float = 0.90
    #: Minimum number of usable bins for the fit to be trusted.
    min_bins: int = 16

    #: Number of blocks used to test for clock drift.
    n_drift_blocks: int = 8
    #: Resample a card when its estimated rate error exceeds this [ppm].
    drift_correction_threshold_ppm: float = 0.5

    #: Accept the alignment when the residual skew is below this [s].
    #: 73 ns == 0.1 deg at 3.8 kHz.
    residual_tolerance_s: float = 100e-9
    #: Triplet closure |t_ab + t_bc + t_ca| must stay below this [s].
    closure_tolerance_s: float = 200e-9

    #: A non-zero intercept in the phase-slope fit means the two channels
    #: differ by more than a pure delay (e.g. a filter mismatch).
    max_intercept_rad: float = 0.05

    #: Largest skew the coarse search will consider [s].  A periodic
    #: excitation makes the delay ambiguous by whole periods, so the search is
    #: bounded to physically plausible values (trigger jitter, cable delay).
    max_lag_s: float = 0.1

    #: When a card's cell-voltage channel is dead, fall back to its strongest
    #: segment channel as a timing proxy.  That recovers the integer-sample
    #: alignment but not nanosecond accuracy, because the segment's own
    #: impedance phase is indistinguishable from a small delay.
    allow_segment_proxy: bool = True
    #: Uncertainty assigned to a proxy-derived delay [s].
    segment_proxy_sigma_s: float = 10e-6

    #: ``same_card``  - pair each segment with the cell voltage recorded on
    #:                  its own card (immune to inter-card skew by
    #:                  construction; the accurate default).
    #: ``reference``  - use one global cell-voltage channel for all segments;
    #:                  requires and exercises the full alignment.
    #: ``auto``       - same_card, falling back to reference for cards whose
    #:                  own cell-voltage channel fails the health check.
    uc_strategy: str = "auto"
    #: A cell-voltage channel below this AC RMS is considered dead [V].
    uc_min_rms_v: float = 2e-3


# ---------------------------------------------------------------------------
# Spectral estimation
# ---------------------------------------------------------------------------

@dataclass
class SpectralConfig:
    """Impedance estimation parameters."""

    #: ``welch``              - coherence-gated Welch over the whole band.
    #: ``synchronous``        - leakage-free DFT at designed multisine tones.
    #: ``stepped_multisine``  - tones of each dwell fitted jointly, per dwell.
    #: ``auto``               - synchronous when ``excitation_tones_hz`` and
    #:                          ``base_frequency_hz`` are given, else
    #:                          stepped_multisine.
    #:
    #: AUTO NO LONGER FALLS BACK TO WELCH.  It used to, whenever the tone
    #: list was not configured -- which is the normal case, because the
    #: schedule is a property of the recording and not something anybody
    #: restates in a config file.  So the default estimator for a stepped
    #: excitation was the one estimator that assumes a stationary one:
    #: Welch averages over the whole record, and a tone that was applied for
    #: a tenth of it is diluted by that factor, along with the coherence
    #: that is supposed to gate it. Both effects worsen with frequency on a
    #: constant-cycles sweep, which is exactly how "the spectrum stops
    #: around a kilohertz" comes to look like a property of the cell.
    #: Welch is still available; it is no longer what you get by accident.
    method: str = "auto"

    nperseg: int = 8192
    noverlap: int | None = None          # None -> nperseg // 2
    window: str = "hann"
    detrend: str = "linear"

    #: Designed multisine.  When known, these are the analysis frequencies -
    #: they are configuration, not something to rediscover from the data.
    base_frequency_hz: float | None = None
    excitation_tones_hz: list[float] | None = None
    #: Number of base periods per analysis window in synchronous mode.
    periods_per_window: int = 4

    #: Transfer-function estimator.  ``hv`` (total least squares) is unbiased
    #: when both signals carry noise; ``h1`` is biased low by input noise.
    estimator: str = "hv"

    f_min_hz: float = 1.0
    f_max_hz: float = 4000.0

    #: dB above the local noise floor for a tone to be claimed inside a
    #: dwell, when the tone list is discovered rather than configured. The
    #: window length imposes a stricter floor automatically.
    tone_peak_db: float = 12.0

    # --- coherence-gated Welch (ensemble-level rejection) ---
    gate_windows: bool = True
    #: Reject a window whose broadband excitation level deviates from the
    #: ensemble median by more than this many median-absolute-deviations.
    amplitude_mad_k: float = 4.0
    #: Reject a window whose transfer-function phase deviates from the
    #: ensemble circular mean by more than this [rad], band-averaged.
    phase_outlier_rad: float = 0.5
    #: Never discard more than this fraction of windows; if the gate wants to,
    #: the record is flagged as unreliable instead.
    max_rejected_fraction: float = 0.5

    # --- output quality gate ---
    min_coherence: float = 0.70
    min_points_per_segment: int = 5

    #: Notch filters for mains interference [Hz]; empty disables.
    notch_freqs_hz: list[float] = field(default_factory=lambda: [50.0, 100.0, 150.0])
    notch_q: float = 30.0
    apply_notch: bool = True


# ---------------------------------------------------------------------------
# Validation and modelling
# ---------------------------------------------------------------------------

@dataclass
class ValidationConfig:
    """Kramers-Kronig and consistency checks."""

    run_lin_kk: bool = True
    kk_max_elements: int = 30
    #: Schoenleber mu-criterion stopping value.
    kk_mu_target: float = 0.85
    #: A segment passes when its max |residual| stays below this [%].
    kk_max_residual_pct: float = 1.0

    run_stationarity: bool = True
    #: Split-half |Z| may differ by at most this fraction to pass.
    stationarity_tolerance: float = 0.05

    run_admittance_sum: bool = True


@dataclass
class ModelConfig:
    """Equivalent-circuit fitting."""

    run_ecm: bool = True
    #: Models tried, best selected by corrected AIC.
    models: list[str] = field(
        default_factory=lambda: ["R_L", "R_L_1RQ", "R_L_2RQ"]
    )
    n_starts: int = 6
    max_nfev: int = 5000
    #: Reject a fitted parameter whose relative standard error exceeds this.
    max_relative_sigma: float = 0.30


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    measurement_id: str = ""
    conditions: list[str] = field(default_factory=list)
    raw_dir: str = "."
    output_dir: str = "./out"

    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    spectral: SpectralConfig = field(default_factory=SpectralConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def param_hash(self) -> str:
        """Stable hash of every parameter that can change a result.

        Excludes paths and the measurement identity, which locate data but do
        not alter the computation.
        """
        payload = self.to_dict()
        for key in ("measurement_id", "conditions", "raw_dir", "output_dir"):
            payload.pop(key, None)
        blob = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def save(self, path: str | Path) -> None:
        import yaml

        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


def _merge(base: Any, override: dict[str, Any]) -> Any:
    """Recursively apply a plain dict onto a dataclass instance."""
    for key, value in override.items():
        if not hasattr(base, key):
            raise KeyError(f"unknown configuration key: {key!r}")
        current = getattr(base, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(base, key, value)
    return base


def load_config(path: str | Path | None = None, **overrides: Any) -> PipelineConfig:
    """Build a :class:`PipelineConfig` from an optional YAML file plus kwargs."""
    cfg = PipelineConfig()
    if path is not None:
        import yaml

        data = yaml.safe_load(Path(path).read_text()) or {}
        _merge(cfg, data)
    if overrides:
        _merge(cfg, overrides)
    return cfg
