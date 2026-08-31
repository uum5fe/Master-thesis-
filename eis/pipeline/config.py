"""Configuration objects for the EIS pipeline.

Every numeric constant that influences a result lives here, is serialisable to
YAML, and contributes to :meth:`PipelineConfig.param_hash` so that any stored
spectrum can be traced back to the exact parameters that produced it.

The sections map onto the three processing tiers:

======================================  ====================================
section                                 consumed by
======================================  ====================================
``acquisition``, ``geometry``, ``sync``  :mod:`eis.pipeline.bronze`
``calibration``                          bronze (scaling) and silver (units)
``spectral``, ``hf``, ``validation``     :mod:`eis.pipeline.silver`
``quality``                              silver (status) and gold (masking)
``model``, ``report``                    :mod:`eis.pipeline.gold`
======================================  ====================================
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

    #: ``welch``           - coherence-gated Welch over the whole band, one
    #:                       window length for every frequency.
    #: ``multiresolution`` - coherence-gated Welch with a window length chosen
    #:                       per octave band: long where the period is long,
    #:                       short (and therefore many times averaged) at the
    #:                       top of the band.  This is what makes the
    #:                       high-frequency end of the arc measurable rather
    #:                       than merely present.
    #: ``synchronous``     - leakage-free DFT at designed multisine tones.
    #: ``auto``            - synchronous when ``excitation_tones_hz`` and
    #:                       ``base_frequency_hz`` are given, else
    #:                       multiresolution.
    method: str = "auto"

    nperseg: int = 8192
    noverlap: int | None = None          # None -> nperseg // 2
    window: str = "hann"
    detrend: str = "linear"

    # --- multi-resolution Welch ---
    #: Each analysis band gets the shortest window that still holds this many
    #: periods of its *lowest* frequency.  Eight periods keeps the leakage of a
    #: Hann window negligible while maximising the number of averages.
    min_periods_per_window: int = 8
    #: Never go below this window length, whatever the frequency.
    nperseg_min: int = 256
    #: Analysis bands per decade of frequency.
    bands_per_decade: int = 3

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
    #: Points below this coherence are *marked unused*, not deleted, so every
    #: segment keeps the same frequency grid and nothing disappears silently.
    min_coherence: float = 0.70
    min_points_per_segment: int = 5

    #: Fold the measured timing uncertainty into the per-point phase
    #: uncertainty: ``sigma_phi^2 += (2*pi*f*sigma_tau)^2``.  Without this a
    #: card timed from a segment proxy reports the same error bar at 3.8 kHz as
    #: one timed from its own cell voltage, which is not true.
    propagate_timing_uncertainty: bool = True

    #: Notch filters for mains interference [Hz]; empty disables.
    notch_freqs_hz: list[float] = field(default_factory=lambda: [50.0, 100.0, 150.0])
    notch_q: float = 30.0
    apply_notch: bool = True

    #: Verify that the designed tones actually carry the excitation, by
    #: clustering coherence peaks and comparing with ``excitation_tones_hz``.
    verify_tones: bool = True


# ---------------------------------------------------------------------------
# High-frequency accuracy
# ---------------------------------------------------------------------------

@dataclass
class HFConfig:
    """The high-frequency accuracy chain (:mod:`eis.hf`).

    Above roughly 500 Hz the measured local impedance stops being dominated by
    the cell and starts being dominated by the *instrument*: the via-shunt's
    own loop inductance, the anti-alias filter, and any residual channel skew.
    All three produce a phase error that grows with frequency and none of them
    is visible in the magnitude, which is why an uncorrected spectrum looks
    perfectly plausible right up to the point where the Nyquist arc bends the
    wrong way.
    """

    enabled: bool = True

    # --- 1. via-shunt as a complex element -------------------------------
    #: Loop inductance of one segment's via shunt [nH].  With a shunt
    #: resistance of tens of microohms even a fraction of a nanohenry gives a
    #: time constant ``tau = L/R`` in the tens of microseconds, i.e. tens of
    #: degrees at the top of the band.  This is the single largest unmodelled
    #: high-frequency error in a segmented-cell measurement; ``0`` disables the
    #: correction and records that it was not applied.
    shunt_inductance_nh: float = 0.0
    #: Per-segment override, ``{segment: nH}``.
    shunt_inductance_nh_per_segment: dict[int, float] = field(default_factory=dict)

    # --- 2. anti-alias filter --------------------------------------------
    #: Order and corner of the acquisition anti-alias filter.  Only the
    #: *mismatch* between the segment and cell-voltage channels matters, so
    #: this is applied as a ratio and is a no-op when both channels share a
    #: filter.  ``aa_order = 0`` disables.
    aa_order: int = 0
    aa_corner_hz: float = 0.0
    #: Fractional corner-frequency mismatch between a segment channel and the
    #: cell-voltage channel.  A 2 % tolerance on a 4 kHz corner is worth about
    #: half a degree at 3.8 kHz.
    aa_corner_mismatch: float = 0.0

    # --- 3. pooled common-mode identification -----------------------------
    #: Identify the residual phase term that is *shared* by every segment on a
    #: card by minimising the pooled Kramers-Kronig residual.  Pooling is what
    #: makes this work: a delay is degenerate with an inductance for a single
    #: spectrum, but the delay is common to the card while the inductance
    #: varies from segment to segment, so the shared term is identifiable from
    #: the ensemble even though it is not identifiable from any one member.
    identify_common_mode: bool = True
    #: Band used for the identification [Hz]; the term being identified only
    #: has leverage where ``2*pi*f*tau`` is appreciable.
    common_mode_band_hz: tuple[float, float] = (200.0, 4000.0)
    #: Search half-width for the residual delay [s].  ``0`` means "derive it
    #: from the measured skew uncertainty", which is the honest default - an
    #: unbounded search eats the cell's real inductance.
    delay_bound_s: float = 0.0
    #: Floor for that derived bound [s].
    delay_bound_floor_s: float = 300e-9
    #: Search half-width for the shunt time constant [s].
    shunt_tau_bound_s: float = 5e-5
    #: Grid points per searched dimension.
    n_grid: int = 41
    #: Below this many usable segments the pooling has no statistical
    #: advantage and the identification is skipped rather than done badly.
    min_segments_for_pooling: int = 4

    #: When the decorrelation has no leverage - most often a card timed from a
    #: segment proxy, where the delay is uncertain to microseconds - fall back
    #: to minimising the *pooled* Kramers-Kronig residual over the card's
    #: segments, still bounded by the measured timing uncertainty.
    kk_fallback: bool = True
    kk_fallback_segments: int = 8
    kk_fallback_grid: int = 21
    #: Largest series inductance the wiring can plausibly have [nH].  This is
    #: what makes the scan identify anything at all: an unbounded inductance
    #: absorbs the delay exactly.  It also sets the answer's uncertainty, since
    #: whatever inductance is still allowed could have been delay instead.
    kk_fallback_max_inductance_nh: float = 50.0
    #: Only attempt the fallback when the card's timing uncertainty is at least
    #: this many times the residual-skew budget; a well-timed card has nothing
    #: for it to find and the scan would only add noise.
    kk_fallback_min_sigma_ratio: float = 10.0

    # --- 4. high-frequency resistance ------------------------------------
    #: ``Rs`` is taken from a weighted ``Rs + jwL`` fit over the top
    #: ``hfr_band_decades`` of the spectrum rather than from the median of the
    #: highest real parts, which is biased by the inductive tail.
    hfr_band_decades: float = 1.0
    hfr_min_points: int = 4
    #: Include a residual-arc term in the HFR fit when this many points are
    #: available; it stops the tail of the kinetic arc leaking into ``Rs``.
    hfr_arc_min_points: int = 8

    # --- 5. in-plane crosstalk -------------------------------------------
    #: Fraction of a segment's current that is collected by its neighbours
    #: through the in-plane conductivity of the diffusion medium.  The measured
    #: admittances are a spatially smoothed version of the true ones; inverting
    #: the mixing matrix sharpens the map.  ``0`` disables.  Non-zero values
    #: must come from a characterisation of the plate, not from taste.
    crosstalk_alpha: float = 0.0
    #: Tikhonov regularisation on the inverse; the mixing matrix is close to
    #: singular for large alpha and an unregularised inverse amplifies noise.
    crosstalk_regularisation: float = 1e-3


# ---------------------------------------------------------------------------
# Segment quality
# ---------------------------------------------------------------------------

@dataclass
class QualityConfig:
    """How a segment is judged - and why nothing is thrown away.

    Deleting a segment moves a quality decision out of the data and into
    whoever happens to read the plot.  Every segment therefore reaches the
    output tables carrying a machine-written ``status`` and a ``quality``
    score, and every plate map draws all of them, with the poor ones visibly
    marked instead of missing.
    """

    #: Keep every segment in the result, classified rather than rejected.
    keep_all_segments: bool = True
    #: A segment is *active* once it has this many usable frequency points.
    active_min_points: int = 3

    #: Quality score weights; they are normalised, so only ratios matter.
    w_coherence: float = 1.0
    w_uncertainty: float = 1.0
    w_kk: float = 1.0
    w_timing: float = 1.0

    #: Relative uncertainty at which the uncertainty term of the score
    #: reaches zero.
    sigma_reference: float = 0.10
    #: Kramers-Kronig residual [%] at which the KK term reaches zero.
    kk_reference_pct: float = 3.0
    #: Segments at or above this score are drawn solid on the plate maps.
    good_quality: float = 0.60


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
    #: A point is a violation only when its residual exceeds *both* this
    #: percentage and ``kk_max_sigma`` times its own uncertainty.  The pair is
    #: what stops an honestly imprecise segment from being declared non-causal
    #: and an over-precise one from passing on a systematic error.
    kk_max_residual_pct: float = 1.0
    kk_max_sigma: float = 3.0

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
    #: Fit segments whose status is not ``ok`` as well.  They are fitted, the
    #: fit is flagged, and the reader decides - which is the whole point of
    #: keeping every segment.
    fit_all_active_segments: bool = True


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

@dataclass
class ReportConfig:
    """Figures, tables and the interactive plate view."""

    write_figures: bool = True
    #: Single self-contained HTML file: plate heat map on the left, the
    #: clicked segment's spectrum on the right, a parameter selector and a
    #: frequency slider.  No server, no external libraries.
    write_dashboard: bool = True

    #: Scalars offered by the heat-map selector.  ``z_mod@f``, ``z_real@f``,
    #: ``neg_z_imag@f`` and ``phase@f`` are frequency-resolved and driven by
    #: the slider.
    heatmap_parameters: list[str] = field(
        default_factory=lambda: [
            "rs_hf", "rp", "quality", "coherence", "sigma_rel",
            "kk_max_residual_pct", "z_mod@f", "neg_z_imag@f", "phase@f",
        ]
    )
    #: Frequencies offered by the slider; empty means "every frequency in the
    #: common grid", which is the useful default.
    heatmap_frequencies_hz: list[float] = field(default_factory=list)
    #: Maximum segments drawn on the spectra overview figure.
    max_spectra_per_figure: int = 12


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
    hf: HFConfig = field(default_factory=HFConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

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
