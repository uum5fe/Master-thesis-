#!/usr/bin/env python3
"""
config.py
=========
Every tunable number in the pipeline, in one place.

WHY A CONFIG MODULE
-------------------
The previous code base carried the same constants in four files that had
drifted apart: `eis_local.py`, `eis_plate_pipeline.py`,
`eis_local_all_in_one(2).py` and the Databricks notebook each had their own
`f_hi`, their own SNR gate and their own copy of `detect_schedule`.  The
notebook then loaded the all-in-one copy by `SourceFileLoader` *and* imported
the module copy, so two different `detect_schedule` implementations were live
in the same session.  That is not a style problem, it is a correctness
problem: a result could not be reproduced from the file names alone.

Here there is exactly one definition of each number, one dataclass that
carries them, and one place to override them (CLI, environment, or a JSON
file).  Nothing downstream defines a default of its own.

USAGE
-----
    from config import Config, DEFAULT
    cfg = Config.from_cli()                 # argparse -> Config
    cfg = DEFAULT.replace(f_max_hz=2000.0)  # programmatic override
    cfg = Config.from_json("run.json")
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, replace, asdict, fields
from pathlib import Path

# ===========================================================================
# 1. PHYSICAL PLATE CONSTANTS  (Bosch R2-D2 72-segment measuring plate)
# ===========================================================================
# These are geometry, not choices.  They mirror r2d2_geometry.py, which stays
# the single source of truth for per-segment areas and centroids; the values
# repeated here are only the ones the pipeline needs before geometry is
# imported (plotting extents, sanity checks).

PAD_W_MM = 5.60                     # pad pitch in x; 45 of them = 252.0 mm
PAD_H_MM = 6.05                     # pad pitch in y; 20 of them = 121.0 mm
N_COLS = 45
N_ROWS = 20
PLATE_W_MM = PAD_W_MM * N_COLS      # 252.0
PLATE_H_MM = PAD_H_MM * N_ROWS      # 121.0
PLATE_W_CM = PLATE_W_MM / 10.0      # 25.20
PLATE_H_CM = PLATE_H_MM / 10.0      # 12.10
A_CELL_CM2 = PLATE_W_MM * PLATE_H_MM / 100.0    # 304.92
N_SEGMENTS = 72

# The four plate temperature sensors sit at these x positions and span the
# full height, so a 1-D interpolation along the flow direction is the honest
# reading of them (see r2d2_geometry.temperature_at).
TEMP_SENSOR_X_MM = {"temp1": 0.0, "temp2": 84.0, "temp3": 168.0, "temp4": 252.0}

# Used only if every temperature channel fails (DC-blocked inputs).
T_FALLBACK_C = 58.4

# Flow direction on the plate.  The gas enters at x = 0 and leaves at
# x = PLATE_W_MM, so x is the "along-channel" coordinate and y is "across".
# The spatial model in gold.py uses this to make its kernel anisotropic:
# neighbouring segments across the flow are far more alike than segments a
# long way down it.
FLOW_AXIS = "x"

# Flow-channel separator lines (mm in y), for the plate drawing only.
FLOW_CHANNEL_Y_MM = (30.25, 60.50, 90.75)


# ===========================================================================
# 2. ACQUISITION
# ===========================================================================

# Dewesoft SIRIUS DualCoreADC, 24-bit delta-sigma, 5 cards x 16 channels.
# Each card records 12-15 segment channels plus at least one UC (cell voltage)
# reference.  65 of the 72 segments are wired; the remaining 7 have no channel
# at all and can only ever be *inferred* (gold.py does this explicitly, and
# flags them as inferred rather than pretending they were measured).
N_CARDS = 5
CHANNELS_PER_CARD = 16
NOMINAL_FS_HZ = 10_000.0

# Segments with a known hardware fault, as segment -> reason.  Anything listed
# here is skipped in bronze and drawn on the map as "bad" rather than measured.
#
# THIS IS DELIBERATELY EMPTY, and it should stay empty unless a fault has been
# demonstrated on data that this pipeline produced.
#
# It used to contain 33 and 59, with the reason "open via / no response above
# noise in every campaign".  No measurement in this repository ever established
# that, and the exclusion made it unfalsifiable: bronze skips an excluded
# segment BEFORE reading its channel (see bronze.py), so those two were never
# evaluated, no evidence could ever accumulate against the claim, and gold
# printed "hardware: open via" from this dict alone.  Every later campaign
# inherited a verdict that nothing had re-tested.
#
# A segment that really has no response does not need to be declared: it
# arrives as a low SNR, a wide posterior and a fault flag on THAT run, which
# is a statement about that recording rather than a permanent property of the
# plate.  That is the honest failure mode, and it is the one the pipeline
# already implements.
KNOWN_BAD_SEGMENTS: dict[str, str] = {}

# FAMOS file naming.  Kept as a template so a different campaign is a config
# change, not a code change.
#: Both naming conventions in use, tried in order.  The dashboard has known
#: about the RO form for a long time; the pipeline knew only the Leepa one, and
#: when a name matched neither it fell back to "every .DAT in the folder" --
#: which on a campaign directory means every CONDITION, so asking for 45A read
#: 45A, 60A, 150A and 450A, four times the data, at every stage.  A run that
#: should take minutes never finished.
FAMOS_PATTERNS = (
    "Leepa_{leepa}_Current_{cond}_Test_{test}_Karte_*.DAT",
    "RO{leepa}-*_Current_{cond}_Test_{test}_Karte_*.DAT",
    "{leepa}_Current_{cond}_Test_{test}_Karte_*.DAT",
)
FAMOS_PATTERN = FAMOS_PATTERNS[0]
FAMOS_TEST_ID = "01"


# ===========================================================================
# 3. THE CONFIG OBJECT
# ===========================================================================


@dataclass(frozen=True)
class Config:
    """One run of the pipeline.

    Frozen on purpose: a config that mutates halfway through a run is how the
    old notebook ended up with two different `f_hi` values in one figure.
    Use `.replace(...)` to derive a variant.
    """

    # ---- plate ------------------------------------------------------------
    # WHICH PHYSICAL PLATE THE RECORDING CAME FROM.  "gen1" is the green
    # Kashyyyk plate, "gen2" the blue Naboo plate.  Both have 72 segments on
    # the same 45x20 pad grid, but the pads belonging to segments 37..72 were
    # re-cut between the two revisions, so the areas and the centroids of
    # those segments differ.  Selecting the wrong plate does not fail: it
    # silently attributes each channel to the wrong piece of hardware.  See
    # r2d2_geometry for the maps and the reconstruction argument.
    plate: str = "gen1"

    # ---- source format ----------------------------------------------------
    # "famos"  multi-card imc FAMOS .DAT recordings -> bronze/silver/gold
    # "csv"    the newer single-file CSV logger     -> csv_pipeline
    # The two are not variants of one reader.  FAMOS gives five free-running
    # cards that must be cross-correlated onto a common clock before any
    # phase is meaningful; the CSV logger writes one row per instant for the
    # whole plate, so there is no inter-card skew to measure and the entire
    # synchronisation stage is not merely unnecessary but undefined.  Nothing
    # is shared below the geometry and the Abgleich.
    source_format: str = "famos"
    csv_path: Path | None = None     # file, or folder of CSV measurements
    csv_dialect: str = "auto"        # see csv_source.detect_dialect

    # Excitation frequencies, when the operator already knows them.  Empty
    # means "discover them from the cell-voltage record", which is the
    # default because a designed multisine and a stepped sweep need different
    # windows and the file says which it is.  Giving the list explicitly
    # removes the only guess on the CSV path.
    csv_tones: tuple[float, ...] = ()

    # CHANNEL SCAN, not sampling.  If the logger walks the channel list
    # inside a row, channel k is sampled k/csv_scan_rate_hz after the row
    # starts, so the segment and the cell voltage in one row are NOT
    # simultaneous.  That offset is a pure delay and is removed analytically
    # -- but only if the acquisition program is known.  0 means "the rows are
    # simultaneous samples", which is recorded in the manifest as the
    # assumption it is.
    csv_scan_rate_hz: float = 0.0
    # segment name -> slot index in the scan, plus "u_cell" for the reference.
    csv_channel_slots: tuple[tuple[str, int], ...] = ()

    # ---- paths ------------------------------------------------------------
    dat_dir: Path = Path(".")
    out_dir: Path = Path("./results")
    curr_cal: Path | None = None     # per-segment "Abgleich": c0;c1 x 72
    temp_cal: Path | None = None     # per-sensor  "Abgleich": c0;c1 x 4
    gain_file: Path | None = None    # ex-situ chain response: seg,f,re,im
    areas_file: Path | None = None   # optional per-segment area override CSV

    # WHOLE-CELL REFERENCE.  A folder of Gamry .DTA sweeps of the same cell at
    # the same operating points, and optionally the bench MF4 log beside them.
    # The 72 segments in parallel must reproduce this; it is the one check that
    # tests the calibration, the geometry, the synchronisation and the chain
    # response at once, against an instrument that shares none of them.
    gamry_dir: Path | None = None
    bench_log: Path | None = None    # ASAM MDF4; defaults to one in gamry_dir

    # EQUAL-AREA MODE.  The plate's true segment areas span 0.678..8.470
    # cm^2, a factor of 12.5.  Setting this replaces them all with
    # A_CELL/72 = 4.235 cm^2.  Local ASR is area-free and does not move,
    # but DC closure, the area-weighted aggregate and every plate average
    # DO -- so this is a deliberate simplification, not a correction, and
    # it is recorded in the manifest whenever it is on.
    equal_areas: bool = False

    # ---- what to process --------------------------------------------------
    leepa: str = ""                  # order id, e.g. "2611976"
    condition: str = "ALL"           # e.g. "45A", "60A", "150A", or "ALL"
    test_id: str = FAMOS_TEST_ID
    i_setpoint_a: float | None = None   # only ever printed as a cross-check

    # ---- frequency band ---------------------------------------------------
    # f_max defaults to 0.45*fs, which is ~2.2 samples per cycle.  A sine fit
    # at a KNOWN frequency is perfectly well conditioned there; what degrades
    # above ~0.4*fs is the blind frequency SEARCH, not the estimate.  Since
    # bronze.py now finds the schedule once, globally, and silver.py only ever
    # fits at known frequencies, the usable ceiling is set by signal amplitude
    # rather than by the search.
    #
    # THE CEILING WAS THE CONFIG CONSTANT, NOT NYQUIST.  A real card header
    # from the campaign reads dx = 1.0e-5 s, i.e. fs = 100 kHz on 16
    # channels, so f_hi(fs) = min(f_max_hz, 0.45*fs) = min(4500, 45000) =
    # 4500 Hz: the converter had 45 kHz of headroom it was never asked for.
    # On RO2612025-01 card 4 (fs = 50 kHz) the Gamry band recorded in the
    # file runs to 23.9 kHz while the pipeline stopped at 7.47 Hz.  Raising
    # this is NECESSARY BUT NOT SUFFICIENT -- on its own it took that card
    # from 11 recovered steps to 13, with the same 7.47 Hz top.  What
    # recovers the band is hf_schedule (see hf_use_ensemble below); this
    # constant only stops binding before it gets the chance.
    f_min_hz: float = 0.15
    f_max_hz: float = 30000.0
    f_hi_frac_fs: float = 0.45       # detection ceiling as a fraction of fs
    ppd: int = 12                    # points per decade of the detection grid

    # ---- schedule detection (bronze) --------------------------------------
    # The consensus schedule is built from ALL reference channels on ALL cards
    # at once.  `min_ref_channels` is how many must agree before a step is
    # accepted; `grid_tol` is the relative tolerance for geometric-grid
    # membership.
    min_ref_channels: int = 2
    grid_tol: float = 0.01
    align_cards: bool = True         # cross-correlate cards onto a common t0
    # The Dewetron cards are ARMED SEPARATELY.  Measured on the 45 A set:
    # card 3 starts 5.712 s after card 1, cards 4/5 about 2.54 s after.
    # A 2 s ceiling silently fails to find the largest real offset, so the
    # search window must be generous -- a wrong lag is caught by the
    # correlation peak height, not by clipping the search.
    align_max_lag_s: float = 12.0
    # A lag is accepted on how far its correlation peak stands above the rest
    # of the curve, not on the peak's absolute height.  Height alone does not
    # separate right from wrong here: on the 45 A set the known-correct
    # near-zero lag of card 2 scored |r| = 0.269 while cards 3/4/5 scored
    # 0.276-0.278 for lags that reproduce this file's own recorded true
    # offsets (5.712 / 2.536 / 2.549 s) to four decimals.  The old 0.30
    # threshold refused all four, and because a refused lag was still applied
    # one way in consensus_schedule, that cost every segment on cards 3-5.
    # The prominence threshold is set from the two distributions, not by
    # taste.  Measured on a 60 s synthetic sweep at this fs (see
    # test_card_alignment.py): unrelated noise peaks at prominence 8.5 over
    # 30 seeds, while a correct lag at the field data's own |r| ~ 0.27 scores
    # ~250.  25 sits three times above the noise ceiling and ten times below
    # the real signal, so it is not a close call in either direction.
    align_min_corr: float = 0.05        # absolute floor against pure garbage
    align_min_prominence: float = 25.0  # robust sigma above the background

    # ---- high-frequency schedule recovery (bronze, hf_schedule.py) --------
    # The blind detector used to be run on the card's REFERENCE channel, the
    # UC* cell-voltage channel with the largest standard deviation.  The
    # sweep is galvanostatic, so the amplitude arriving there is
    # |i_ac| * |Z_cell(f)|, and |Z_cell| falls by an order of magnitude from
    # the bottom of the band to its ~45 mOhm*cm2 minimum near 8 kHz.  The
    # detector was being asked to find a tone exactly where the cell had
    # removed it -- which no value of min_snr_db can undo.
    #
    # The SEGMENT channels measure current density, and current is what the
    # sweep imposes, so their tone amplitude is flat in frequency.  Stacking
    # the ~14 of them on a card adds the tone coherently and the noise in
    # power.  Measured on RO2612025-01 card 4 at 45 A: +11.2 dB narrowband
    # over UC2 above 1 kHz, and the recovered band went 11 -> 21 steps
    # (0.478 .. 189 Hz) on the stack alone, 42 steps (0.478 .. 18.9 kHz)
    # with the ladder extension below.
    #
    # STACK PER CARD, NOT ACROSS THE PLATE.  Pooling all five cards scored
    # WORSE than one card on the synthetic (23/26 against 26/26): the cards
    # are not on a common time base until estimate_card_lags has run, and
    # the residual sub-sample offsets plus the per-slot multiplexer skew make
    # the sum partially destructive at the top of the band.  Each card gets
    # its own stack and consensus_schedule does the cross-card vote it
    # already does.
    hf_use_ensemble: bool = True     # detect on the stacked segment ensemble
    hf_ladder_extend: bool = True    # predict-and-verify the missing rungs
    # Generators are asked for round numbers of points per decade; a free fit
    # is not.  On card 4 a ladder fitted on the fifteen steps below 12 Hz
    # returned 10.059 points/decade, and that 0.13 % error in r compounds
    # with the rung index: checked blind against fifteen tones observed
    # between 946 Hz and 24 kHz, the prediction error grew monotonically from
    # +3.4 % to +5.8 %, so the extension found noise.  Snapping to 10 brings
    # the same blind prediction to -0.9 .. +1.0 %, and 27 of 32 predicted
    # rungs then verify.
    hf_ladder_snap_ppd: bool = True  # snap the fitted spacing to an integer
    hf_ladder_tol: float = 0.02      # relative window for ladder membership
    # OFF BY DEFAULT.  Pruning is the only part of the ensemble path that can
    # REMOVE a step the old pipeline would have kept, so it is the only part
    # that can make a run worse -- and it did, on real 45 A data: a band that
    # reached 550 Hz came back reaching 375 Hz.  A detection is a measurement;
    # the ladder is a model fitted on a handful of low-frequency steps and
    # extrapolated upward, and at the top of the band, where its extrapolation
    # error is largest, the model is the one more likely to be wrong.  Left
    # off, the ensemble path is purely additive.  Turn it on for a record with
    # a continuous interferer the detector keeps latching onto -- the one case
    # ladder membership handles and an SNR gate provably cannot -- and read
    # off_ladder_hz in the manifest to see what it took.
    hf_ladder_prune: bool = False    # drop detections that miss the ladder
    # Averaging SEGMENT impedances across cards is wrong -- they are
    # different segments.  Averaging the five UC channels is not: they are
    # five measurements of one cell voltage, with uncorrelated front-end
    # noise, and the reference is the weak phasor now that detection has
    # moved off it.  Worth ~7 dB exactly where it is weakest.
    hf_pool_reference: bool = True   # inverse-variance mean of A_uc across cards

    # ---- per-step quality gates -------------------------------------------
    # A step that lies on the sweep's own geometric grid is a real step: a
    # geometric progression is not something noise produces.  For those, SNR
    # stops being a membership test and becomes a WEIGHT in the KK fit.
    # Off-grid candidates still face the full gate.
    min_snr_db: float = 5.0          # gate for OFF-grid steps
    snr_floor_db: float = -3.0       # absolute floor even for on-grid steps

    # THE GATE THAT ACTUALLY MATTERS.
    # Raw SNR is the wrong quantity to threshold on, because a long dwell
    # beats down noise: at -3 dB per sample with 20000 samples the phasor is
    # still good to ~1 %, while the same -3 dB over 60 samples is worthless.
    # What decides usability is the PROPAGATED relative uncertainty, which
    # already folds in the dwell length through the Cramer-Rao bound.
    #
    # Without this gate the grid-membership rescue of on-grid steps admits
    # noise: a high-frequency point whose segment phasor is pure noise gives
    # Z = K*A_ref/A_seg with A_seg ~ 0, so |Z| blows up and the phase is
    # random.  That is what produced the diverging high-frequency tails and
    # the positive phase excursions -- note that acquisition skew CANNOT do
    # this, being all-pass and therefore unable to change |Z| at all.
    # Relaxed from 0.35 once the targeted |Z|-outlier gate below took over
    # the job.  At 0.35 this removed the whole top decade of the band along
    # with the noise, which loses the ohmic resistance -- the point of the
    # measurement.  It is now a backstop, not the main filter.
    sigma_rel_max: float = 0.60      # drop a point above 60 % relative sd

    # The gate that actually removes runaway points, without removing weak
    # ones.  |Z| is smooth in log-frequency for any physical cell, so a point
    # far from its own segment's local trend is an outlier whatever its SNR,
    # and a weak point lying on the trend is credible however weak.
    zmag_outlier_mad: float = 4.5    # deviations from the local median
    zmag_outlier_win: int = 7

    # A phasor from a fraction of a cycle is not a measurement.  Guards the
    # low-frequency end, where a 0.15 Hz step needs ~20 s to give 3 cycles.
    min_cycles_per_dwell: float = 3.0

    # PASSIVITY IS NOT A SAFE ASSUMPTION AT LOW FREQUENCY.
    # A segmented plate with all segments tied to one cell voltage and the
    # excitation applied globally measures Z_k, not the localised Z_loc,k.
    # Schneider et al., ECS Trans. 25(1) 937 (2009) show that Z_k can have a
    # genuinely NEGATIVE real part at low frequency: an ac perturbation of
    # the whole cell changes oxygen (or fuel, or water) consumption upstream,
    # those concentration oscillations travel down the channel, and the
    # outlet segment sees an induced polarisation eta_up.  Once
    # |K| = |eta_up|/|eta_mod| exceeds unity the local ac current falls out
    # of phase with the modulation and the local polarisation resistance goes
    # negative.  The relation is Z_k = Z_loc,k / (1 - K).
    #
    # This is real physics and a diagnostic of down-the-channel starvation,
    # not an artefact.  The effect is confined to low frequency, where the
    # channel transport time constant lives, so the positive-real-part gate
    # is applied only ABOVE this frequency.  Below it, only the magnitude
    # bound applies, and a negative real part is kept and flagged.
    #
    # Set to 0.0 to police passivity over the whole band (correct only if the
    # excitation is applied segment-by-segment).
    passivity_gate_min_hz: float = 1.0

    # DO NOT FIT THE COMMON-MODE DELAY.  It is not merely hard to identify --
    # it is DEGENERATE with the series inductance, exactly and by
    # construction.  At high frequency
    #       Z_true * exp(-j w dt)  ~  (R0 + j w L) * (1 - j w dt) ...
    # so a delay dt and an inductance L = -R0*dt produce the same spectrum.
    # Measured: L = 200 nH with dt = 0 and L = 0 with dt = -3.33 us both give
    # a high-frequency phase-slope delay of -9.71 us, agreeing to three
    # decimals.  Only the SUM (L + R0*dt) is observable.
    #
    # Since the measurement model already fits and removes a series
    # inductance, any common delay is absorbed there and nothing is lost by
    # setting dt0 = 0.  Fitting both is asking the data for two numbers when
    # it contains one, and that is what produced the -154.6 us on card 1 --
    # a value with no physical meaning that rotated the top of the band by
    # 206 degrees and destroyed every segment on that card.
    #
    # The DIFFERENTIAL delay is a different matter entirely: it varies from
    # segment to segment within a card, no single inductance can mimic it,
    # and it is well determined.  That is still fitted.
    #
    # Set True only when an independent calibration (e.g. a short-circuit
    # recording) pins the inductance separately.
    fit_common_delay: bool = False

    # Which converter architecture to assume for the DIFFERENTIAL skew.
    #   "auto"    fit both and keep the lower Kramers-Kronig residual
    #   "slot"    one converter walking the channel list, step 1/(n_ch*fs)
    #   "parity"  two cores alternating, step 1/(2*fs), depends on odd/even
    #
    # "auto" is the honest default when the hardware is unknown, but its
    # margin is small -- 1 to 2 percent of the fit cost on synthetic data --
    # so it can and does pick wrong.  If the acquisition card documentation
    # says which architecture it uses, SET IT HERE.  A known answer beats a
    # weakly discriminated fit, and the consequences of guessing wrong are
    # large: the two nominal steps differ by a factor of ten at 16 channels.
    skew_basis: str = "auto"

    # PRESETS.  Config.preset("permissive") loosens every gate to roughly the
    # philosophy of a coherence-threshold pipeline: keep the point unless it
    # is obviously broken, and let the reader judge.  Use it to compare
    # like-for-like against another evaluation, or to see what the strict
    # gates removed before deciding whether they were right to.
    #
    # The trade is real and runs both ways.  Strict gates can remove weak but
    # genuine points, which is what makes a plot look sparse.  Loose gates
    # keep points whose phasor ratio is dominated by noise, which is what
    # makes a plot look complete while being partly fiction.  Neither
    # setting is "correct" -- the honest procedure is to run both and read
    # the flags column to see which points differ and why.
    preset_name: str = "default"
    max_thd: float = 0.10            # linearity (Giner-Sanz 2015)
    max_drift: float = 0.25          # amplitude stationarity across sub-windows

    # ---- phasor estimation (silver) ---------------------------------------
    # "joint7" is the seven-parameter two-channel sine fit of Ramos & Serra
    # (Measurement 41 (2008) 135): one shared frequency estimated from BOTH
    # records, which halves the standard deviation of the phase DIFFERENCE
    # relative to two independent four-parameter fits -- and the phase
    # difference is exactly the impedance phase.  "independent" reproduces the
    # old behaviour and is kept only for A/B comparison.
    phasor_method: str = "joint7"    # {"joint7", "independent"}
    joint7_max_iter: int = 12
    joint7_tol: float = 1e-9

    # ---- acquisition skew (silver) ----------------------------------------
    # A skew dt multiplies Z by exp(-j w dt): an all-pass, invisible in |Z|,
    # linear in phase.  At 10 kHz half a sample is 50 us = 54 deg at 3 kHz,
    # which is the entire high-frequency defect in the old plots.
    #
    # skew_model:
    #   "structural"  dt_seg = dt0_card + k_card * (pos_seg - pos_uc)
    #                 Two parameters per card constrained by ~13 segments,
    #                 instead of one free delay per card or per segment.
    #                 The channel index comes from the FAMOS header, which IS
    #                 the ADC acquisition order.
    #   "per_card"    one free delay per card (old behaviour)
    #   "none"        no correction
    skew_model: str = "structural"
    skew_dt_range_s: float = 300e-6
    skew_n_grid: int = 241
    skew_min_decades: float = 1.5    # below this the band cannot resolve dt
    hf_anchor: bool = True           # enforce passive/minimum-phase HF asymptote
    hf_anchor_top_decade: float = 1.0   # decades from f_max used as the anchor

    # ---- measurement model / KK (silver) ----------------------------------
    mu_crit: float = 0.85            # Schoenleber, Klotz, Ivers-Tiffee (2014)
    kk_tol: float = 0.02             # residual gate, 2 %
    kk_ridge: float = 1e-3
    remove_inductance: bool = True   # cable + plate L is not electrochemistry
    min_points_per_spectrum: int = 8

    # ---- uncertainty ------------------------------------------------------
    # Per-point sigma from the Cramer-Rao bound for a single tone in white
    # noise (Rife & Boorstyn, IEEE Trans. Inf. Theory 20 (1974) 591) rather
    # than the old ad-hoc 10^(-SNR/20).  See utils.crlb_phasor.
    uncertainty_model: str = "crlb"  # {"crlb", "snr", "uniform"}
    sigma_rel_floor: float = 1e-3
    sigma_rel_ceiling: float = 0.60   # cap for weighting, not a gate

    # ---- DRT / process resolution (gold) ----------------------------------
    # Splitting the spectrum by relaxation time is what turns one ambiguous
    # |Z| map into three maps that mean different things physically.  The
    # boundaries below are the conventional PEMFC split; they are config, not
    # code, because a different MEA moves them.
    drt_enable: bool = True
    drt_method: str = "gp"           # {"ridge", "gp"}  -- gp = GP-DRT
    drt_n_tau_per_decade: int = 10
    drt_lambda: float = 1e-3         # ridge weight, ignored when method="gp"

    # Gaussian-process prior on the DRT.  Because the model is linear in
    # (R_inf, L, gamma), a Gaussian prior gives a CLOSED-FORM Gaussian
    # posterior -- so R_inf arrives with a standard deviation instead of as a
    # bare number, and the posterior can be evaluated above f_max.  That is
    # the only honest route to the high-frequency intercept when the sampling
    # rate stops the band below it.  Liu & Ciucci, Electrochim. Acta 331
    # (2020) 135316.
    #
    # The prior is what makes tau-padding safe.  Unregularised lin-KK
    # diverges when tau extends past 1/w_min, because 1/(1+jw tau) -> 1
    # becomes degenerate with the constant column and R_inf trades against it
    # freely.  The kernel penalises exactly that runaway, so the model can
    # describe relaxations just outside the window instead of pretending they
    # do not exist.
    # Asymmetric on purpose.  Measured column correlation with the constant
    # column: 1.0000 at the fast end (tau << 1/w_max), 0.47 at the slow end.
    # So the fast end is where R_inf runs away and the slow end is safe.  A
    # SMALL fast pad is kept deliberately: R_inf means "everything faster than
    # the band", so when a real relaxation sits just above f_max the intercept
    # is genuinely ambiguous, and letting the model carry a little fast
    # content turns that hidden bias into an honest widening of the posterior.
    drt_tau_pad_fast: float = 0.0    # decades below 1/w_max: MUST stay 0
    drt_tau_pad_slow: float = 1.0    # decades above 1/w_min
    drt_tau_pad_decades: float = 0.0  # legacy alias, read only as a fallback
    drt_optimise_hypers: bool = True    # maximise the marginal likelihood
    drt_length_scale_init: float = 0.7  # decades of log10(tau)
    drt_sigma_f_init: float = 1.0       # prior amplitude, relative to |Z|
    drt_sigma_n_init: float = 1.0       # noise scale multiplier
    drt_extrapolate_decades: float = 0.7  # how far above f_max to evaluate
    tau_split_ohmic_s: float = 1e-4      # tau below this -> R_ohmic bucket
    tau_split_kinetic_s: float = 1e-2    # tau below this -> R_ct, above -> R_mt

    # ---- spatial model (gold) ---------------------------------------------
    # The plate field is smooth: neighbouring segments share gas composition,
    # membrane hydration and clamping pressure.  Modelling that explicitly is
    # what lets every one of the 72 segments carry a value -- measured ones
    # from data, unwired and failed ones from the posterior, each with its own
    # credible interval and each flagged for what it is.
    spatial_enable: bool = True
    spatial_kernel: str = "matern52"
    spatial_len_x_mm: float = 70.0   # along flow: long correlation length
    spatial_len_y_mm: float = 35.0   # across flow: shorter
    spatial_nugget: float = 0.05     # relative
    spatial_dc_closure: bool = True  # constrain sum(j_s * A_s) to the setpoint
    infer_missing_segments: bool = True
    max_inferred_fraction: float = 0.60   # refuse to draw a map that is mostly guess

    # ---- output -----------------------------------------------------------
    write_csv: bool = True
    write_json: bool = True
    write_html: bool = True
    write_png: bool = True
    heatmap_params: tuple[str, ...] = (
        "R_ohmic", "R_ct", "R_mt", "Z_mag_100Hz", "phase_100Hz", "j_dc",
    )
    heatmap_colormap: str = "RdYlBu_r"
    verbose: bool = True
    report_steps: bool = False       # print the full per-step table

    # ---- derived ----------------------------------------------------------
    # Segments to skip entirely.  Empty by default: excluding a segment before
    # its data is seen removes the only evidence that could ever overturn the
    # exclusion.  Use --exclude for a genuine, current hardware fault.
    exclude_segments: frozenset[str] = field(
        default_factory=lambda: frozenset(KNOWN_BAD_SEGMENTS)
    )

    # -- helpers ------------------------------------------------------------

    def replace(self, **kw) -> "Config":
        return replace(self, **kw)

    def famos_pattern(self, cond: str | None = None) -> str:
        """The first pattern, kept for messages and for backward compatibility."""
        return self.famos_patterns(cond)[0]

    def famos_patterns(self, cond: str | None = None) -> list[str]:
        """Every filename convention this campaign might use, in order."""
        return [
            p.format(
                leepa=self.leepa or "*",
                cond=cond or (self.condition
                              if self.condition != "ALL" else "*"),
                test=self.test_id,
            )
            for p in FAMOS_PATTERNS
        ]

    def preset(self, name: str) -> "Config":
        """Return a copy with a named gate preset applied."""
        if name in ("default", "", None):
            return self
        if name == "permissive":
            return self.replace(
                preset_name="permissive",
                sigma_rel_max=1.5,          # was 0.60
                min_cycles_per_dwell=1.0,   # was 3.0
                zmag_outlier_mad=8.0,       # was 4.5
                min_points_per_spectrum=4,  # was 8
                min_snr_db=0.0,             # was 5.0
                snr_floor_db=-12.0,         # was -3.0
                max_thd=0.5,
                max_drift=0.5,
            )
        if name == "strict":
            return self.replace(
                preset_name="strict",
                sigma_rel_max=0.30,
                min_cycles_per_dwell=4.0,
                zmag_outlier_mad=3.5,
                min_points_per_spectrum=10,
            )
        raise ValueError(f"unknown preset {name!r}")

    def f_hi(self, fs: float) -> float:
        """Detection ceiling for a given sampling rate."""
        return min(self.f_max_hz, self.f_hi_frac_fs * fs)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, (set, frozenset)):
                d[k] = sorted(v)
            elif isinstance(v, tuple):
                d[k] = list(v)
        return d

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_json(cls, path) -> "Config":
        raw = json.loads(Path(path).read_text())
        return cls._coerce(raw)

    @classmethod
    def from_env(cls, prefix: str = "EIS_") -> "Config":
        raw = {}
        for f in fields(cls):
            key = prefix + f.name.upper()
            if key in os.environ:
                raw[f.name] = os.environ[key]
        return cls._coerce(raw)

    @classmethod
    def _coerce(cls, raw: dict) -> "Config":
        """Turn strings/lists from JSON, env or argparse into typed fields."""
        kw = {}
        types = {f.name: f.type for f in fields(cls)}
        for k, v in raw.items():
            if k not in types or v is None:
                continue
            t = str(types[k])
            if "Path" in t:
                kw[k] = Path(v)
            elif "frozenset" in t:
                kw[k] = frozenset(str(s) for s in v)
            elif "tuple" in t:
                kw[k] = tuple(v)
            elif "bool" in t:
                kw[k] = v if isinstance(v, bool) else str(v).lower() in (
                    "1", "true", "yes", "on")
            elif "int" in t:
                kw[k] = int(v)
            elif "float" in t:
                kw[k] = float(v)
            else:
                kw[k] = v
        return cls(**kw)

    @classmethod
    def build_argparser(cls) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(
            prog="eis-pipeline",
            description="Local EIS on the R2-D2 72-segment plate. "
                        "Bronze -> Silver -> Gold. No reference instrument.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        g = p.add_argument_group("hardware / source")
        g.add_argument("--plate", choices=["gen1", "gen2"], default="gen1",
                       help="gen1 = green/Kashyyyk, gen2 = blue/Naboo")
        g.add_argument("--source", dest="source_format",
                       choices=["famos", "csv"], default="famos",
                       help="measurement file format")
        g.add_argument("--csv", dest="csv_path", type=Path,
                       help="CSV measurement file or folder (--source csv)")
        g.add_argument("--csv-dialect", dest="csv_dialect", default="auto",
                       help="force a CSV layout instead of auto-detecting")

        g = p.add_argument_group("paths")
        g.add_argument("--dat", dest="dat_dir", type=Path, required=False,
                       help="folder of FAMOS .DAT files")
        g.add_argument("--out", dest="out_dir", type=Path, default=Path("./results"))
        g.add_argument("--curr-cal", dest="curr_cal", type=Path,
                       help="per-segment Abgleich c0;c1 (REQUIRED: the only "
                            "absolute scale once the potentiostat is gone)")
        g.add_argument("--temp-cal", dest="temp_cal", type=Path)
        g.add_argument("--gamry", dest="gamry_dir", type=Path,
                       help="folder of whole-cell Gamry .DTA sweeps to "
                            "compare the aggregated local result against")
        g.add_argument("--bench-log", dest="bench_log", type=Path,
                       help="ASAM MDF4 bench log, to report the operating "
                            "point at each reference sweep")
        g.add_argument("--gain", dest="gain_file", type=Path,
                       help="ex-situ chain response segment,freq,re,im")
        g.add_argument("--exclude", dest="exclude_segments", default=None,
                       type=lambda v: frozenset(
                           x.strip() for x in v.split(",") if x.strip()),
                       help="comma-separated segments to skip entirely, for a "
                            "demonstrated hardware fault. Empty by default: an "
                            "excluded segment is never measured, so the "
                            "exclusion can never be disproved")
        g.add_argument("--areas", dest="areas_file", type=Path)
        g.add_argument("--equal-areas", dest="equal_areas",
                       action="store_true",
                       help="treat every segment as A_cell/72 = 4.235 cm2")
        g.add_argument("--config", type=Path, help="JSON config; CLI overrides it")

        g = p.add_argument_group("selection")
        g.add_argument("--leepa", default="")
        g.add_argument("--condition", default="ALL")
        g.add_argument("--current", dest="i_setpoint_a", type=float,
                       help="setpoint in A; only ever printed as a check")

        g = p.add_argument_group("band")
        g.add_argument("--f-min", dest="f_min_hz", type=float, default=0.15)
        g.add_argument("--f-max", dest="f_max_hz", type=float,
                       default=30000.0)
        g.add_argument("--ppd", type=int, default=12)

        g = p.add_argument_group("estimation")
        g.add_argument("--phasor", dest="phasor_method",
                       choices=["joint7", "independent"], default="joint7")
        g.add_argument("--skew", dest="skew_model",
                       choices=["structural", "per_card", "none"],
                       default="structural")
        g.add_argument("--no-hf-anchor", dest="hf_anchor", action="store_false")
        g.add_argument("--uncertainty", dest="uncertainty_model",
                       choices=["crlb", "snr", "uniform"], default="crlb")

        g = p.add_argument_group("gold")
        g.add_argument("--no-drt", dest="drt_enable", action="store_false")
        g.add_argument("--drt-method", choices=["ridge", "gp"], default="ridge")
        g.add_argument("--no-spatial", dest="spatial_enable", action="store_false")
        g.add_argument("--no-infer", dest="infer_missing_segments",
                       action="store_false")

        g = p.add_argument_group("output")
        g.add_argument("--quiet", dest="verbose", action="store_false")
        g.add_argument("--report", dest="report_steps", action="store_true")
        g.add_argument("--no-html", dest="write_html", action="store_false")
        return p

    @classmethod
    def from_cli(cls, argv=None) -> "Config":
        """Precedence: CLI > --config JSON > environment > dataclass default."""
        ns = cls.build_argparser().parse_args(argv)
        raw: dict = {}
        base = cls.from_env()
        raw.update(base.to_dict())
        if getattr(ns, "config", None):
            raw.update(json.loads(Path(ns.config).read_text()))
        given = {k: v for k, v in vars(ns).items()
                 if v is not None and k != "config"}
        raw.update(given)
        return cls._coerce(raw)


DEFAULT = Config()


# ===========================================================================
# 4. PRESENTATION CONSTANTS
# ===========================================================================
# Kept here rather than in gold.py so a figure restyle never touches analysis
# code.

COLORS = {
    "plate": "#C89A2E",          # gold plate
    "plate_edge": "#8B6910",
    "via": "#A07818",
    "channel_line": "#6A4A00",
    "measured_edge": "#FFFFFF",
    "inferred_edge": "#7A7A7A",
    "bad_edge": "#B03030",
    "cell": "#111111",
    "pass": "#2CA02C",
    "fail": "#D62728",
    "grid": "#DDDDDD",
}

# How each segment class is drawn on a heat map.  A segment that was never
# wired must not look like a segment that measured a low value.
SEGMENT_CLASS_STYLE = {
    "measured": dict(alpha=0.92, linewidth=1.6, hatch=None),
    "inferred": dict(alpha=0.45, linewidth=1.0, hatch="///"),
    "bad":      dict(alpha=0.25, linewidth=1.0, hatch="xxx"),
}

# Units and human labels for every scalar the gold layer can map.
PARAM_META = {
    "R_ohmic":     dict(label="R\u03a9 (HF intercept)", unit="m\u03a9\u00b7cm\u00b2",
                        scale=1000.0, cmap="RdYlBu_r"),
    "R_ct":        dict(label="R_ct (charge transfer)", unit="m\u03a9\u00b7cm\u00b2",
                        scale=1000.0, cmap="RdYlBu_r"),
    "R_mt":        dict(label="R_mt (mass transport)", unit="m\u03a9\u00b7cm\u00b2",
                        scale=1000.0, cmap="RdYlBu_r"),
    "R_pol":       dict(label="R_pol (total polarisation)", unit="m\u03a9\u00b7cm\u00b2",
                        scale=1000.0, cmap="RdYlBu_r"),
    "Z_mag_100Hz": dict(label="|Z| @ 100 Hz", unit="m\u03a9\u00b7cm\u00b2",
                        scale=1000.0, cmap="RdYlBu_r"),
    "phase_100Hz": dict(label="Phase @ 100 Hz", unit="\u00b0",
                        scale=1.0, cmap="RdYlBu_r"),
    "j_dc":        dict(label="DC current density", unit="A/cm\u00b2",
                        scale=1.0, cmap="viridis"),
    "tau_peak":    dict(label="Dominant relaxation time", unit="s",
                        scale=1.0, cmap="plasma"),
    "sigma_rel":   dict(label="Relative uncertainty", unit="%",
                        scale=100.0, cmap="Greys"),
}

# Fault signatures used by the gold layer to label a segment.  Thresholds are
# relative to the plate median, so they travel between operating points.
# Sources: Hakenjos & Hebling, J. Power Sources 145 (2005) 307; Schneider et
# al., Electrochem. Commun. 7 (2005) 1393; Guo et al., IEEE TIM 74 (2025).
FAULT_RULES = {
    "drying":            dict(R_ohmic_ratio=(1.25, None), R_mt_ratio=(None, 1.3)),
    "flooding":          dict(R_mt_ratio=(1.6, None), R_ohmic_ratio=(None, 1.15)),
    "starvation":        dict(R_ct_ratio=(1.5, None), R_mt_ratio=(1.4, None)),
    "contact_loss":      dict(R_ohmic_ratio=(1.6, None), R_ct_ratio=(None, 1.2)),
}


if __name__ == "__main__":
    cfg = DEFAULT
    print("R2-D2 EIS pipeline configuration")
    print(f"  plate       : {PLATE_W_MM} x {PLATE_H_MM} mm, "
          f"{A_CELL_CM2:.2f} cm2, {N_SEGMENTS} segments")
    print(f"  acquisition : {N_CARDS} cards x {CHANNELS_PER_CARD} ch @ "
          f"{NOMINAL_FS_HZ:.0f} Hz")
    print(f"  known bad   : {', '.join(sorted(KNOWN_BAD_SEGMENTS))}")
    print(f"  band        : {cfg.f_min_hz} .. {cfg.f_max_hz} Hz, ppd={cfg.ppd}")
    print(f"  phasor      : {cfg.phasor_method}   skew: {cfg.skew_model}   "
          f"uncertainty: {cfg.uncertainty_model}")
    print(f"  gold        : drt={cfg.drt_method if cfg.drt_enable else 'off'}, "
          f"spatial={'on' if cfg.spatial_enable else 'off'}")
