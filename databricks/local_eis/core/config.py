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

# WHICH WAY THE GASES ACTUALLY GO.
# The maps used to be titled "gas flows left to right", which is true of one
# gas and false of the other: on this bench the plate is COUNTER-FLOW -- H2
# enters at the bottom left and leaves top right, air enters at the bottom
# right and leaves top left. The distinction is not cosmetic. Under co-flow
# both inlets are at the same end, so a hydration gradient is monotonic along
# the plate; under counter-flow each gas is driest at its own end, so the
# membrane can be driest at BOTH ends and wettest in the middle. A reader
# told "left to right" will look for a monotonic trend that counter-flow has
# no reason to produce.
#
# Set to "co" (both inlets at x = 0) or "counter" for the rebuild being
# evaluated; the maps and the trend diagnostic annotate themselves from it.
FLOW_ARRANGEMENT = "counter"
# These describe the plate in PLATE coordinates (x = 0 at pad column 1). The
# maps print plate_style.flow_note() instead, which also follows the view
# switch below.
FLOW_DESCRIPTION = {
    "counter": "counter-flow: H2 left to right, air right to left",
    "co": "co-flow: both gases left to right",
    "unknown": "flow arrangement not recorded",
}

# WHICH END THE COOLANT ENTERS: "x0" (pad column 1, the H2-inlet / air-outlet
# end) or "xW" (pad column 45, the air-inlet / H2-outlet end).  Each end
# carries a column of three ports on the maps: gas outlet above the coolant,
# gas inlet below it -- x0: AIR OUT / COOLANT / H2 IN, xW: H2 OUT / COOLANT /
# AIR IN.
#
# "xW" -- coolant in with the air inlet, coolant co-flowing with the air --
# is read off the data, not a drawing:
#   * bench log 2611976 (MF4): at t ~ 1.5 min the coolant inlet T_Si_CL dips
#     by ~6 K. The ANODE outlet T_So_A dips with it (~1.5 K, same instant);
#     the cathode outlet T_So_C does not move. The H2 outlet port therefore
#     sits at the coolant-inlet end, and on a counter-flow plate that is the
#     air-inlet end. (The step at ~23 min is the gas inlet heaters: each
#     outlet follows its own inlet, so it says nothing about position.)
#   * FAMOS plate sensors (temp1 at x = 0 ... temp4 at x = 252): warmest at
#     x = 252 at 45 A and 60 A (+0.8..1.1 K), flat at 150 A. Coolant warming
#     from x = 0 would make that gradient GROW with current; coolant entering
#     at x = 252, next to the hot inlet air (T_Si_C runs 10-20 K above the
#     plate), cancels it as the reaction heat grows -- which is what 150 A
#     shows.
# Neither is a drawing; confirm against the manifold drawing when there is
# one. A coolant-inlet temperature step settles it: the sensor that responds
# first is at the coolant inlet.
COOLANT_INLET_END = "xW"

# HOW THE PLATE IS DRAWN.  False: as the pad map is numbered, pad column 1 on
# the left -- O2 out top left, H2 out top right, H2 in bottom left, air in
# bottom right; with COOLANT_INLET_END = "xW", coolant out left, in right.
# True: seen from the other side, mirrored left <-> right.
#
# Either way the whole plate is drawn together -- segments, numbers, ports,
# coolant, sensors -- so every port stays next to the segments it really
# feeds. Only the picture changes; every coordinate in the CSVs (cx_mm,
# cy_mm) stays in plate coordinates.
PLATE_VIEW_MIRRORED = False


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
NOMINAL_FS_HZ = 25_000.0

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
    "Leepa_RO{leepa}_Current_{cond}_Test_{test}_Karte_*.DAT",
    "Leepa_RO{leepa}_Current_{cond}_Test_*_Karte_*.DAT",
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

    # A spectra_dir whose x axis runs from the other end of the plate: every
    # segment is re-assigned to its left-right mirror before evaluation
    # (csv_source.mirror_x). Off by default -- see the comparison with the
    # FAMOS evaluation in docs/csv_vs_famos.md before switching it on.
    csv_mirror_x: bool = False

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

    # CURRENT-CHAIN LAG, MEASURED ON EVERY RUN (channel_lag.py).
    # Each segment's shunt / trace / amplifier can lag the cell voltage by a
    # first-order time constant; on 2612030 up to ~110 us, which at the
    # 1.2 kHz top of band is 37 deg and an R_ohmic ~20 mOhm*cm2 low. Silver
    # fits tau per segment against the plate median and removes it before
    # R_ohmic is read. "report" fits and records without changing Z; "off"
    # skips the stage. A tau is only used if the phase really follows
    # -atan(w tau) (rms residual <= channel_lag_max_resid_deg) -- a card read
    # at the wrong time does not, and is flagged instead of "corrected".
    # Calibrated on 2612030 45/60/150 A and 2611976 45 A: good channels fit
    # to 0.2-4 deg, lags -100..+180 us.
    # THE GAMRY CLOCK (gamry_sync.py). The FAMOS cards record the cell's
    # response to the Gamry's own sweep, so the .dta for this condition (found
    # under gamry_dir like the whole-cell comparison does) knows the exact
    # frequency and the time of every step. "report": measure the clock offset
    # and list misplaced / missing steps (bronze/gamry_sync.csv). "frequency":
    # also use the Gamry's exact frequencies, and apply a refused card lag
    # that the Gamry clock corroborates. "guide": also re-locate misplaced and
    # missing windows inside their Gamry slot, each verified by a CFAR test.
    # On 2612030 / 60 A: offset 99.24 s, spread 0.44 s; 1.5-4.7 kHz misplaced.
    gamry_sync: str = "report"             # off | report | frequency | guide
    channel_lag: str = "correct"           # off | report | correct
    channel_lag_f_lo_hz: float = 50.0      # below: no lag is visible
    channel_lag_min_points: int = 4
    # smaller: left alone. 1 us, not 5: with the band at 3.8 kHz a 5 us
    # delay is 7 deg at the top, and a delay fit is good to ~1 us there
    channel_lag_min_us: float = 1.0
    channel_lag_max_us: float = 250.0      # larger: not a chain lag
    channel_lag_max_resid_deg: float = 10.0
    channel_lag_iterations: int = 3        # correct, re-reference, refit
    channel_lag_model: str = "delay"       # delay | first_order
    # CARD VOLTAGE GAIN, MEASURED ON EVERY RUN (card_gain.py).
    # Z = K * A_UC / A_seg, and with the reference pool off every card divides
    # by its OWN UC channel: a card whose voltage chain reads 3 % high puts
    # +3 % on every segment it carries. All cards record the same cell voltage
    # on UC2, so each card's UC phasor against the median card IS its gain.
    # "report" measures and writes bronze/card_reference.csv; "correct" also
    # divides each card's Z by that gain when it is flat over the band (low vs
    # high band within card_gain_flat_pct) and within card_gain_max_pct --
    # larger is a different tap or a wiring fault, and is reported instead.
    card_gain: str = "report"              # off | report | correct
    # SERIES RESISTANCE BETWEEN THE UC TAP AND THE REFERENCE'S SENSE LEADS.
    # Every card's UC2 and the Gamry's Vdc measure the cell voltage at
    # different points. On 2612030 UC2 - Vdc = -1.6 mV - I x 44 uOhm over
    # 45..450 A (residual 0.1 mV): the UC2 taps include 44 uOhm the Gamry's do
    # not, i.e. 13.4 mOhm*cm2 over the 304.9 cm2 plate, in EVERY segment's Z.
    # dc_closure.py measures it (needs the Gamry .dta files and two or more
    # conditions). Set it here to subtract it from Re Z in silver and put the
    # maps on the Gamry's reference plane; 0 leaves Z as the UC taps see it.
    uc_series_mohm_cm2: float = 0.0
    # FREQUENCY RESPONSE OF THE MEASURING CHAIN (frequency_response.py).
    # After every run: the ex-situ amplifier response G_d per segment (from
    # gain_file, or from the Abgleich bode/ folder when given) and the
    # in-situ response of each segment against the plate median, before and
    # after the timing correction. Writes RUN_DIR/frequency_response/ and two
    # plausibility checks; never changes Z. "off" skips it.
    freq_response: str = "report"          # off | report
    # STYLE OF THE STATIC PLATE MAPS (plate_figure.py). "segments": every
    # segment filled with its own value and labelled; "interpolated": 2D
    # linear interpolation between the segment centres, a square on every
    # measured segment, no labels (parula, soft gloss) -- the layout of the
    # bench's MATLAB maps, written as plate_<param>_interp.png; "both".
    heatmap_style: str = "interpolated"    # segments | interpolated | both
    abgleich_bode_dir: Path | None = None
    card_gain_min_snr_db: float = 20.0     # steps below: not used
    card_gain_min_steps: int = 5
    card_gain_tol_pct: float = 1.0         # within: "ok"
    card_gain_flat_pct: float = 2.0        # low vs high band, else "not flat"
    card_gain_max_pct: float = 10.0        # beyond: not a gain tolerance

    # WHOLE-CELL REFERENCE.  A folder of Gamry .DTA sweeps of the same cell at
    # the same operating points, and optionally the bench MF4 log beside them.
    # The 72 segments in parallel must reproduce this; it is the one check that
    # tests the calibration, the geometry, the synchronisation and the chain
    # response at once, against an instrument that shares none of them.
    gamry_dir: Path | None = None
    bench_log: Path | None = None    # ASAM MDF4; defaults to one in gamry_dir

    # WHICH CAMPAIGN'S SWEEPS, WHEN THE FOLDER HOLDS SEVERAL.
    # A whole-cell sweep is named "V26_092_HFR_101_CurrVal_45.dta": it carries
    # the BUILD TOKEN and no order number at all. The measurement file for the
    # same cell, "..._RO2612030-01_V26_092_lokale_EIS_6_Boxen_3.mf4", carries
    # both, which is what lets an order be resolved to a build and a build to
    # its sweeps.
    #
    # Leave it empty and a shared Gamry folder is read whole. That is not a
    # loud failure: sweeps are keyed by current, every campaign has a 45 A,
    # and the last file read simply replaces the earlier one. The comparison
    # then runs this cell's local aggregate against another cell's reference
    # with nothing visibly wrong. Set it whenever the folder is shared.
    gamry_version: str = ""          # e.g. "V26_092"

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
    # THE TOP OF THE BAND SILVER MODELS, as a fraction of the card's fs.
    # Detection may go higher (f_hi_frac_fs); the impedance may not. On
    # 2612030 (fs = 25 kHz) the five cards agree as pure delays up to 3.8 kHz
    # (1-2 deg apart) and part above it: 10 deg at 4.7 kHz, 24 at 5.9 kHz,
    # with the plate spectrum turning inductive. With f_max_hz = 250000 the
    # band ran to 9.5 kHz and R_ohmic, read at its top, came out 38-43
    # mOhm*cm2 -- below even the Gamry. 0.16 * 25 kHz = 4 kHz. Raise it only
    # for hardware whose cards stay phase-coherent higher; 0 disables it.
    coherent_f_max_frac_fs: float = 0.16
    f_hi_frac_fs: float = 0.45       # detection ceiling as a fraction of fs
    # FULL BAND: evaluate every step of the sweep the card can resolve, up to
    # just below its Nyquist frequency (full_band_frac_fs * fs = 12.25 kHz at
    # 25 kHz), instead of stopping at the phase-coherent limit above. On
    # 2612030 the Gamry swept 30 kHz .. 0.3 Hz: 4733, 5928, 7547, 9516 and
    # 11953 Hz are added; 15.0, 19.0, 23.9 and 30.0 kHz lie above fs/2, alias
    # onto other frequencies and cannot be measured at 25 kHz. What changes
    # with it (silver):
    #   * R_ohmic is the measured high-frequency intercept (Im Z = 0, the
    #     Gamry and bench-tool definition), not the top-band mean -- the top
    #     of a full band is the inductive branch, where that mean read
    #     38-43 mOhm*cm2 on 2612030;
    #   * the in-situ channel-lag fit stays below the coherent limit, where
    #     the chain IS a delay; above it the cards part (10 deg at 4.7 kHz,
    #     24 deg at 5.9 kHz) and those points are kept, flagged by the
    #     Kramers-Kronig residual, not used to estimate a delay.
    full_band: bool = False
    full_band_frac_fs: float = 0.49
    # "auto": intercept with full_band, top-band mean otherwise;
    # "intercept" / "topband" force one.
    r_ohmic_method: str = "auto"
    ppd: int = 12                    # points per decade of the detection grid

    # ---- schedule detection (bronze) --------------------------------------
    # The consensus schedule is built from ALL reference channels on ALL cards
    # at once.  `min_ref_channels` is how many must agree before a step is
    # accepted; `grid_tol` is the relative tolerance for geometric-grid
    # membership.
    min_ref_channels: int = 2
    grid_tol: float = 0.01
    # THE REFERENCE CHANNEL IS NAMED, NOT DISCOVERED.
    # One cell-voltage line is fanned out to every Dewetron card, on UC2, and
    # the remaining UC inputs are unconnected or carry something else. The
    # code used to take the UC channel with the largest standard deviation on
    # each card independently, which is free to land on a DIFFERENT channel
    # per card -- and an unconnected input is exactly the kind of channel an
    # argmax on std likes, because a floating input is noisy. Cross-
    # correlating one card's UC2 against another's UC1 then returns the lag
    # of two unrelated signals, which can pass both the prominence gate and
    # the absolute floor while meaning nothing, and every dwell window on
    # that card lands on the wrong tone. Naming the channel makes the wiring
    # an input to the pipeline instead of something it re-guesses per file.
    # Set this (or EIS_REF_CHANNEL, or --ref-channel) if a campaign was wired
    # to a different line; empty string restores the old per-card argmax.
    ref_channel: str = "UC2"
    align_cards: bool = True         # cross-correlate cards onto a common t0
    # The Dewetron cards are ARMED SEPARATELY.  Measured on the 45 A set:
    # card 3 starts 5.712 s after card 1, cards 4/5 about 2.54 s after.
    # A 2 s ceiling silently fails to find the largest real offset, so the
    # search window must be generous -- a wrong lag is caught by the
    # correlation peak height, not by clipping the search.
    # 2612030 / 150 A: cards 1 and 2 started 18.1 s after the others (the
    # Gamry clock measured it on 26 and 16 steps). With a 12 s ceiling the
    # true peak was outside the search, a spurious one at 8.6 s (corr 0.08)
    # was found and refused, and both cards ran on the wrong windows. The
    # records are ~280 s long, so 30 s still leaves most of them overlapping.
    align_max_lag_s: float = 30.0
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
    # MEASURED SEPARATION, NOT A GUESS.  On RO2612030 the correct lags score
    # |r| = 0.980-0.998 on every healthy card at 45 A and 450 A.  At 150 A
    # cards 1 and 2 score |r| = 0.083 -- a dead cell-voltage reference -- yet
    # 0.083 cleared the old 0.05 floor, so an 8.63 s lag was ACCEPTED and
    # applied.  The consensus schedule then carried windows from two
    # incompatible time bases 17.7 s apart (= 2 x 8.63 s), 13 of its 45 steps
    # sat out of time order, and the 75-189 Hz block survived only on the two
    # shifted cards.  Real and false lags differ here by a factor of twelve;
    # 0.5 sits halfway between them in the log and nothing on this campaign
    # falls in between.
    # ---- card alignment filter band ---------------------------------------
    # The alignment cross-correlation runs on a band-passed copy of the
    # reference. This band exists ONLY to estimate the lag -- it never limits
    # the reported impedance spectrum -- but it was hard-coded at 0.5-300 Hz
    # in bronze.py, so a sensitivity study meant editing the module and no
    # run recorded which band produced its lags. Both limits are clamped to
    # 0.45*fs per card, so a band above a slow card's Nyquist degrades to
    # that card's usable top rather than zeroing its whole spectrum.
    align_f_lo_hz: float = 0.5
    align_f_hi_hz: float = 300.0
    # THE PROMINENCE GUARD IS A TIME, NOT A SAMPLE COUNT.
    # The guard excludes the peak's own shoulders from the background the
    # peak is scored against. The correlation peak of a band-limited signal
    # is about 1/align_f_lo_hz wide, i.e. 2 s at 0.5 Hz -- a duration. The
    # old fixed 5000 samples is 0.5 s on a 10 kHz card and 0.05 s on a
    # 100 kHz one, so the same recording scored differently for no reason but
    # its sample rate, and on the fast cards most of the shoulder was counted
    # as background, which UNDERSTATES a real peak.
    #
    # 2.0 s (= 1/align_f_lo_hz) is measured, not assumed. On the 60 s
    # synthetic of test_card_alignment.py, sweeping the guard from 0.05 s to
    # 4 s (see test_prominence_guard.py):
    #
    #     guard      null max (30 seeds)     correct 5.7121 s lag
    #     0.05 s          8.53                      252
    #     0.20 s          8.53                      258
    #     2.00 s          8.51                      345
    #     4.00 s          8.47                      417
    #
    # The null does not move -- noise has no shoulders, so widening the
    # excluded region changes nothing about its background -- while a real
    # peak climbs, because its own shoulders stop being averaged into the
    # background it is scored against. Separation goes from ~30x to ~41x at
    # no cost, and because only real peaks move, nothing that passed
    # align_min_prominence before can fail it now.
    #
    # WHAT THIS DOES NOT FIX. The null ceiling stays at ~8.5, so the 15.0
    # gate below still clears it by 1.8x, not the 2x that
    # test_card_alignment.py::test_noise_alone_produces_no_prominent_peak
    # demands -- that test fails for that reason and the failure is real.
    # The gate was cut from 25 to 15 because RO2612030's genuine alignments
    # scored 21-22 at a 0.2 s guard; those same peaks should score ~1.3x
    # higher at 2.0 s, which would leave room to put the gate back up. That
    # is a threshold to re-derive from a re-run of the field data, not to
    # guess at here.
    align_guard_s: float = 2.0
    align_min_corr: float = 0.50        # absolute floor against pure garbage
    align_min_prominence: float = 15.0  # robust sigma above the background
    # NOTE: was 25.0 but that refused clearly-correct alignments on 25 kHz
    # recordings (RO2612030: prominence 21-22, |r| > 0.994).  15.0 is still
    # 3.5x above the noise floor (worst genuinely-bad alignment was 4.1 on
    # RO2612025) while accepting these valid results.

    # ---- corroboration between cards --------------------------------------
    # A weak peak that a SECOND card independently reproduces is a different
    # claim from a weak peak alone: the cards are armed in groups, so two of
    # them sharing a trigger genuinely share an offset, while noise does not
    # put two independent correlations 2 ms apart on an 8.6 s lag. A card
    # whose prominence falls between align_corroborate_min_prominence and
    # align_min_prominence is accepted if another card agrees to within
    # align_agree_tol_s.
    #
    # THIS RELAXES PROMINENCE ONLY, NEVER align_min_corr. The pair this rule
    # was written from -- RO2612030 at 150 A, cards 1 and 2, +215634 and
    # +215687 samples, 53 samples apart -- scored |r| = 0.083 on a dead
    # reference, and the 0.50 floor still refuses it. Corroboration buys a
    # card past a ragged peak, not past a dead channel.
    #
    # 20 ms is the agreement window because it is an order of magnitude above
    # the 2.1 ms that two genuinely co-triggered cards differed by, and three
    # orders below the offsets being confirmed. 5.0 is the corroboration
    # floor because 4.1 is the worst genuinely-bad alignment measured on this
    # campaign (RO2612025); below that, agreement proves nothing.
    # Set align_corroborate_min_prominence above align_min_prominence to
    # disable the mechanism.
    align_agree_tol_s: float = 0.02
    align_corroborate_min_prominence: float = 5.0

    # ---- alignment diagnostics (report-only, never a gate) -----------------
    # A constant lag corrects a different start TIME. It does not test that
    # two cards kept the same sample RATE: at 20 ppm over a 300 s record the
    # two slide 6 ms apart, a quarter of a 25 ms dwell, so the windows at one
    # end of the sweep walk off their tone while the other end looks perfect.
    # The lag is re-estimated in blocks, each compared over the SAME physical
    # interval, and the slope of lag against time is reported in ppm.
    #
    # These numbers are recorded, logged and never used to refuse a lag. A
    # diagnostic that becomes a gate the day it is written is a gate whose
    # threshold was never checked against a distribution; per the rollout
    # plan these run in report-only mode until the ppm and closure
    # distributions have been seen across 45/60/150/450 A.
    align_drift_blocks: int = 5          # < 3 disables the diagnostic
    align_drift_half_window_s: float = 0.050
    align_max_clock_ppm: float = 20.0

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
    # OFF BY DEFAULT, AND THIS IS A REVERSAL.
    # It was on. On RO2612025-01 at 150 A, where the cell voltage really has
    # collapsed, detecting on the ensemble is the difference between a band
    # and no band. On RO2611976-01 at 45 A and 60 A it made a working result
    # WORSE: the stack is a more sensitive detector, so on a record that is
    # mostly not swept -- and that one is 252 s carrying about 23 s of sweep
    # -- it finds more candidates in the idle stretches too, and the schedule
    # inflates. Silver then has more junk to gate than signal to keep, and
    # the Nyquist that came out was a zigzag where the old path drew clean
    # arcs.
    #
    # A change that can make a good result bad has to be opted into, not out
    # of. Off, this whole module is inert and the pipeline is exactly the one
    # that produced those arcs. Turn it on per campaign, with --ensemble, and
    # compare.
    hf_use_ensemble: bool = False    # detect on the stacked segment ensemble
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

    # ---- consensus ladder snap -------------------------------------------
    # After the cross-card consensus, replace each step frequency by the exact
    # rung of the sweep's own geometric ladder.  The detector's frequency
    # estimate is good to ~0.5 % on a clean dwell but drifts to >1 % where the
    # window was mis-cut, and the sine fit is evaluated AT THE REPORTED
    # FREQUENCY: 1.12 % at 596.99 Hz over the 0.2498 s dwell is 1.67 cycles of
    # phase slip, measured at -16.5 dB on cards 1, 4 and 5 of RO2612030 while
    # the same step passed on cards 2 and 3.  Snapping also collapses the
    # duplicate detections that share a dwell window: 71 -> 45 steps at 45 A
    # and 77 -> 45 at 450 A, both recovering 10 points/decade independently.
    ladder_snap: bool = True
    ladder_snap_ppd: int | None = None   # None = recover it from the data

    # ---- dwell-window sanity ----------------------------------------------
    # A stepped sweep visits its rungs in order, so the window start time is
    # monotonic in frequency.  A step that breaks that order, or whose dwell
    # is a small fraction of the local dwell, did not come from the sweep --
    # the detector latched onto something else and the ladder snap carried it
    # through, because the snap corrects frequencies and not windows.  On
    # RO2612030 at 450 A the 1501.05 Hz rung claimed a window at t = 246.9 s,
    # 132 s after both neighbours and outside the swept part of the record,
    # with a 0.036 s dwell against a local median of 0.240 s; all 68 segments
    # rejected it.  The repair interpolates the window in log-frequency from
    # the steps that do sit in order.  It is a PREDICTION, not a measurement:
    # the quality gates still decide whether a real tone is found there.
    window_sanity: bool = True
    window_min_dwell_frac: float = 0.40
    # A few stray windows are a detector slip. Half the schedule out of order
    # is a second time base -- a card lag accepted on a correlation that
    # should have refused it -- and repairing those windows would hide the
    # fault. Above this fraction the repair refuses and says so.
    window_max_repair_frac: float = 0.25

    # ONE STRETCH OF RECORD HOLDS ONE TONE (gamry_sync.confine_windows,
    # gamry_sync.separate_windows). With the Gamry .dta available, every step
    # longer than window_min_interval_s gets its window from the Gamry step
    # interval, minus window_guard_s at both ends (stamps have 1 s resolution)
    # and window_settle_periods periods of settling at the start; then no two
    # windows may overlap. On RO2612030 the blind detector let low-frequency
    # windows run 4-10 s into the neighbouring step.
    # gamry_sync = "guide": an "ok" window whose segment channels show less
    # than this median SNR at its frequency holds no tone and is searched for
    # in its Gamry slot like a misplaced one (None: never). Normal windows
    # read 6..25 dB, an empty one about -30 dB.
    window_min_snr_db: float | None = -3.0
    # Blind frequencies (no Gamry): each step's frequency is re-estimated
    # once on the segment channels within +-freq_refine_pct of the ladder
    # value (the Gamry rounds its frequencies up to ~1.1 % off the ideal
    # 10/decade grid), and every fit -- pooled reference and segments --
    # then uses that one frequency (fit_at_exact_frequency).
    # silver: reject a point whose segment SNR is this far below the 75th
    # percentile of the 5 steps on either side (None: off). Correct windows on 2612030
    # read 17-24 dB; windows on the neighbouring step 0-4 dB.
    off_tone_drop_db: float | None = 12.0
    # full band: remove each card's phase offset against the plate above the
    # coherent limit (silver.card_hf_phase); |Z| is untouched
    card_hf_phase: bool = True
    freq_refine: bool = True
    freq_refine_pct: float = 1.5
    fit_at_exact_frequency: bool = True
    window_confine: bool = True
    window_min_interval_s: float = 3.0
    window_guard_s: float = 0.5
    window_settle_periods: float = 0.5
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
    # Off for the same reason: it replaces each card's own reference phasor,
    # and it reports ref_slot = 0 to silver's skew model on the strength of a
    # rotation this has not been validated against field data.
    hf_pool_reference: bool = False  # inverse-variance mean of A_uc across cards

    # ---- per-step quality gates -------------------------------------------
    # A step that lies on the sweep's own geometric grid is a real step: a
    # geometric progression is not something noise produces.  For those, SNR
    # stops being a membership test and becomes a WEIGHT in the KK fit.
    # Off-grid candidates still face the full gate.
    min_snr_db: float = 10.0          # gate for OFF-grid steps
    snr_floor_db: float = -3.0       # absolute floor even for on-grid steps

    # SILVER'S OWN SNR BACKSTOP, SEPARATE FROM THE TWO ABOVE.
    # `min_snr_db` is overloaded: bronze uses it for blind detection, for the
    # basis of the grid fit, and for the polarity decision, where a strict
    # value is right.  Silver used the SAME number as a per-point membership
    # test, where it is wrong, because raw SNR does not decide whether a
    # phasor is usable -- N*gamma does (Rife & Boorstyn), and the dwell N
    # spans three orders of magnitude across one sweep.  Measured on
    # RO2612030: a 10 % phasor needs -37 dB at 0.20 Hz, -20 dB at 95 Hz and
    # -18 dB at 377 Hz.  No single number is right at both ends, and at
    # min_snr_db = 5 dB the frequencies with full 68/68 coverage stop at
    # 47.5 Hz (45 A) and 3.7 Hz (450 A) -- which is why the heat maps above
    # the low-frequency arc were unreadable.
    #
    # sigma_rel_max below is the physically correct gate and already folds in
    # the dwell length.  It is bimodal on real data: 2918 of 4828 points at
    # 45 A sit below sigma_rel = 0.10 and only 102 fall in 0.03..0.10, so the
    # threshold sits in a genuine valley rather than on a slope.  These two
    # are left far below where they bind, as a backstop against pathological
    # points, not as the main filter.  Set them to 5.0 / -3.0 to reproduce
    # the old behaviour exactly.
    # gamma >= 0 IN THE DRT. A distribution of relaxation times is a sum of
    # RC elements of a passive network, so it cannot be negative. The
    # unconstrained posterior does go negative on noisy data, and not
    # slightly: it describes the low-frequency arc with excess weight at
    # mid tau and cancels it with negative weight at slow tau. The slow
    # bucket then sums negative, gold clamped that to a hard 0.0, and the
    # plate reported "no mass transport" everywhere while R_ct absorbed the
    # difference. On a synthetic carrying R_ct = 40 and R_mt = 30 mOhm*cm2,
    # the unconstrained fit returns R_mt = 15 and R_ct = 46 at zero noise and
    # R_pol biased -37 % at 8 % noise; constrained it returns 30.6 and 38.2,
    # with R_pol within 3.5 % and the same fit to the data. Turn this off
    # only to reproduce the old behaviour for comparison.
    drt_nonneg: bool = True

    # -40 as in the script that produced the clean 45 A spectra. At -20 the
    # weak off-grid top-of-band points of a LOW-current condition (45 A: a
    # tenth of the 450 A excitation) are cut here even though sigma_rel_max
    # -- the gate that is meant to decide -- would keep them.
    silver_snr_gate_db: float = -40.0    # silver, OFF-grid points
    silver_snr_floor_db: float = -40.0   # silver, ON-grid points

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
    #
    # BACK TO FALSE. It had been flipped to True while this note still said
    # not to. The script that drew clean 45 A arcs up to ~1 kHz on RO2612030
    # ran with False; the same code with True lost everything above ~90 Hz at
    # 45 A and scattered 450 A, which is what a wrong dt0 rotating the top of
    # the band into the non-passive half-plane looks like.
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
    skew_basis: str = "slot"

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
    max_thd: float = 0.10
              # linearity (Giner-Sanz 2015)
    max_drift: float = 0.25          # amplitude stationarity across sub-windows

    # ---- phasor estimation (silver) ---------------------------------------
    # "joint7" is the seven-parameter two-channel sine fit of Ramos & Serra
    # (Measurement 41 (2008) 135): one shared frequency estimated from BOTH
    # records, which halves the standard deviation of the phase DIFFERENCE
    # relative to two independent four-parameter fits -- and the phase
    # difference is exactly the impedance phase.  "independent" reproduces the
    # old behaviour and is kept only for A/B comparison.
    # "card_ml": one frequency per (card, step) from ALL channels of the card
    # (tone_estimation.card_frequency), then linear LS phasors at it.
    phasor_method: str = "card_ml"   # {"card_ml", "joint7", "independent"}
    drift_alpha: float = 1e-3        # stationarity is rejected only if p < alpha
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
    # Subtract the fitted series inductance jwL from the output spectrum?
    # Off: on 2612030 the fitted L is 0 on every segment at 45, 60 and 450 A,
    # and non-zero only where channel distortion is left above ~2 kHz (cards
    # 1-2 at 150 A, 1.2-1.3 uH). Subtracting it rotated those spectra by up
    # to 35 deg at 3.8 kHz -- removing an artefact as if it were a cable. L is
    # still fitted and reported per segment (L_nH); set True to subtract it.
    remove_inductance: bool = False
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
        "R_ohmic", "ReZ_1kHz", "R_ct", "R_mt", "Z_mag_100Hz", "phase_100Hz",
        "j_dc",
    )
    heatmap_colormap: str = "plotly_jet"   # plate_style.JET_STOPS
    verbose: bool = True
    report_steps: bool = False       # print the full per-step table

    # ---- derived ----------------------------------------------------------
    # Segments to skip entirely.  Empty by default: excluding a segment before
    # its data is seen removes the only evidence that could ever overturn the
    # exclusion.  Use --exclude for a genuine, current hardware fault.
    exclude_segments: frozenset[str] = field(
        default_factory=lambda: frozenset(KNOWN_BAD_SEGMENTS)
    )

    # ---- reconstructing a segment from the ones around it ------------------
    # THREE DIFFERENT THINGS, KEPT APART ON PURPOSE.
    #
    #   exclude_segments      the segment is gone. No measurement, no value,
    #                         no place in the aggregate. Its area is not
    #                         counted. Use when the segment must play no part.
    #
    #   substitute_segments   the segment's OWN measurement is not trusted,
    #                         but the plate still has that area and it still
    #                         conducts. The measurement is discarded and a
    #                         value is reconstructed from the measured
    #                         segments touching it, so the aggregate covers
    #                         the whole plate and the map has no hole. The
    #                         donors are recorded per segment.
    #
    #   fill_missing_from_neighbours
    #                         the same reconstruction, applied to every
    #                         segment that has no spectrum at all -- an
    #                         unwired channel, or one every gate rejected.
    #
    # The reconstruction is an area-weighted mean of the neighbours'
    # area-specific spectra (neighbours.fill_spectrum). It is an estimate and
    # it is labelled as one everywhere it appears: class "substituted" in
    # gold, its own CSV in silver, and the donor list in both. It is never
    # mixed into spectra_clean.csv, which stays measurements only.
    substitute_segments: frozenset[str] = frozenset()
    fill_missing_from_neighbours: bool = False
    #: How far the search for measured neighbours may widen. 1 = the ring
    #: that shares an edge. 2 allows one more hop when that ring is itself
    #: unmeasured, which is segment 33's situation on RO2612030 (its ring is
    #: 61, 62, 67, 68 and two of those are missing).
    fill_max_hops: int = 2

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
        """Ceiling of the BLIND step search for a given sampling rate."""
        return min(self.f_max_hz, self.f_hi_frac_fs * fs)

    def f_known_hi(self, fs: float) -> float:
        """Ceiling for steps whose frequency is KNOWN (the Gamry's): a sine
        fit at a known frequency stays well conditioned up to just below
        Nyquist, so with full_band it goes to full_band_frac_fs * fs."""
        if self.full_band:
            return min(self.f_max_hz, self.full_band_frac_fs * fs)
        return self.f_hi(fs)

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
                # A STRING IS ONE VALUE, NOT A SEQUENCE OF CHARACTERS.
                # The environment hands everything over as text, so
                # EIS_EXCLUDE_SEGMENTS="33,59" was iterated character by
                # character into {'3', '5', '9', ','} -- which excludes
                # segments 3, 5 and 9 and leaves 33 and 59 in. Every element
                # of that set is a plausible segment number, so nothing
                # downstream could notice.
                if isinstance(v, str):
                    v = [x for x in v.replace(";", ",").split(",") if x.strip()]
                kw[k] = frozenset(str(s).strip() for s in v)
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
        g.add_argument("--gamry-version", dest="gamry_version", default=None,
                       help="build token of the sweeps that belong to this "
                            "cell, e.g. V26_092. Required when the Gamry "
                            "folder holds more than one campaign, because a "
                            ".dta name carries no order number")
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
        g.add_argument("--substitute", dest="substitute_segments", default=None,
                       type=lambda v: frozenset(
                           x.strip() for x in v.split(",") if x.strip()),
                       help="segments whose own measurement is discarded and "
                            "rebuilt from their measured neighbours, e.g. 33")
        g.add_argument("--fill-gaps", dest="fill_missing_from_neighbours",
                       action="store_true", default=None,
                       help="rebuild every segment that has no spectrum from "
                            "its measured neighbours, so the aggregate covers "
                            "the whole plate")
        g.add_argument("--equal-areas", dest="equal_areas",
                       action="store_true",
                       help="treat every segment as A_cell/72 = 4.235 cm2")
        g.add_argument("--config", type=Path, help="JSON config; CLI overrides it")

        g = p.add_argument_group("selection")
        g.add_argument("--leepa", default="")
        g.add_argument("--condition", default="ALL")
        g.add_argument("--current", dest="i_setpoint_a", type=float,
                       help="setpoint in A; only ever printed as a check")

        # default=None, not "UC2": from_cli layers argparse OVER the env and
        # the JSON config, so a flag that always carries a value would make
        # EIS_REF_CHANNEL and --config unsettable. The real default lives on
        # the dataclass field.
        g.add_argument("--ref-channel", dest="ref_channel", default=None,
                       help="UC channel carrying the shared cell-voltage "
                            "reference on every card (default: UC2); pass an "
                            "empty string to fall back to the loudest UC "
                            "channel on each card")

        g = p.add_argument_group("band")
        g.add_argument("--f-min", dest="f_min_hz", type=float, default=0.15)
        g.add_argument("--f-max", dest="f_max_hz", type=float,
                       default=30000.0)
        g.add_argument("--ppd", type=int, default=12)

        g = p.add_argument_group("estimation")
        # default=None so the dataclass default ("card_ml") is not silently
        # overridden by an argparse default when running from the CLI
        g.add_argument("--phasor", dest="phasor_method",
                       choices=["card_ml", "joint7", "independent"],
                       default=None)
        g.add_argument("--drift-alpha", dest="drift_alpha", type=float,
                       default=None,
                       help="card_ml only: stationarity is rejected only if "
                            "the chi-square p-value is below this "
                            "(default 1e-3)")
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
    # A segment the operator excluded is not a segment that failed, and it is
    # not one whose value was inferred. It carries no value at all, so it is
    # drawn blank -- distinguishable at a glance from a measured neighbour and
    # from a guessed one.
    "excluded": dict(alpha=0.15, linewidth=1.0, hatch="..."),
    # rebuilt from the ring that touches it: coloured, but visibly not solid
    "substituted": dict(alpha=0.55, linewidth=1.2, hatch="\\\\"),
}

# Units and human labels for every scalar the gold layer can map.
PARAM_META = {
    "R_ohmic":     dict(label="R\u03a9 (HF intercept)", unit="m\u03a9\u00b7cm\u00b2",
                        scale=1000.0, cmap="plotly_jet"),
    "R_ct":        dict(label="R_ct (charge transfer)", unit="m\u03a9\u00b7cm\u00b2",
                        scale=1000.0, cmap="plotly_jet"),
    "R_mt":        dict(label="R_mt (mass transport)", unit="m\u03a9\u00b7cm\u00b2",
                        scale=1000.0, cmap="plotly_jet"),
    "R_pol":       dict(label="R_pol (total polarisation)", unit="m\u03a9\u00b7cm\u00b2",
                        scale=1000.0, cmap="plotly_jet"),
    "ReZ_1kHz":    dict(label="Re Z @ 1 kHz (band-independent HF)",
                        unit="m\u03a9\u00b7cm\u00b2", scale=1000.0,
                        cmap="plotly_jet"),
    "Z_mag_100Hz": dict(label="|Z| @ 100 Hz", unit="m\u03a9\u00b7cm\u00b2",
                        scale=1000.0, cmap="plotly_jet"),
    "phase_100Hz": dict(label="Phase @ 100 Hz", unit="\u00b0",
                        scale=1.0, cmap="plotly_jet"),
    "j_dc":        dict(label="DC current density", unit="A/cm\u00b2",
                        scale=1.0, cmap="plotly_jet"),
    "chain_tau_us": dict(label="Current-chain lag \u03c4", unit="\u00b5s",
                         scale=1.0, cmap="plotly_jet"),
    "tau_peak":    dict(label="Dominant relaxation time", unit="s",
                        scale=1.0, cmap="plotly_jet"),
    "sigma_rel":   dict(label="Relative uncertainty", unit="%",
                        scale=100.0, cmap="plotly_jet"),
}

# ---- FIXED HEAT-MAP COLOUR SCALES -----------------------------------------
# >>> CHANGE THE HEAT-MAP SCALING HERE <<<
#
# One fixed (vmin, vmax) per resistance, in DISPLAY units (mOhm*cm2), used
# for EVERY measurement condition. With a per-run min..max (or 5..95 %)
# scale the same colour meant 70 mOhm*cm2 at 45 A and 170 at 450 A, so two
# conditions could not be compared by eye. A fixed scale makes a colour mean
# one value on every map.
#
# Chosen from order 2612030 (45 / 60 / 150 / 450 A, all 72 segments, default
# settings), rounded outward to neat numbers:
#
#             observed over all four conditions      fixed scale
#   R_ohmic   45 ..  77                              40 ..  80
#   R_ct      37 .. 209                              25 .. 225
#   R_mt       9 .. 273                               0 .. 300
#   R_pol     62 .. 419                              50 .. 450
#
# A value outside its range is still drawn: in the end colour, with its true
# number printed on the segment and an arrow on the colour bar.
#
# The notebook can override without editing this file, e.g.
#     config.HEATMAP_LIMITS["R_ct"] = (20.0, 250.0)
#     config.HEATMAP_FIXED_SCALE = False    # back to automatic per-run scaling
HEATMAP_FIXED_SCALE = True
HEATMAP_LIMITS = {
    "R_ohmic": (40.0, 80.0),     # HFR / Rs
    "ReZ_1kHz": (45.0, 85.0),    # Re Z at 1 kHz: R_ohmic plus the open arc
    "R_ct":    (25.0, 225.0),    # charge transfer
    "R_mt":    (0.0, 300.0),     # mass transport
    "R_pol":   (50.0, 450.0),    # total polarisation = R_ct + R_mt (+ R_hf_extra)
}
# other names the same quantities are drawn under (ECM cell, CSV path)
HEATMAP_ALIASES = {"Rs": "R_ohmic", "Rs (ECM)": "R_ohmic", "HFR": "R_ohmic",
                   "R_ct (ECM)": "R_ct", "R_mt (ECM)": "R_mt",
                   "R_pol (ECM)": "R_pol"}


def heatmap_limits(param: str) -> tuple[float, float] | None:
    """Fixed (vmin, vmax) in display units for `param`, or None = automatic."""
    if not HEATMAP_FIXED_SCALE:
        return None
    lim = HEATMAP_LIMITS.get(HEATMAP_ALIASES.get(param, param))
    return (float(lim[0]), float(lim[1])) if lim is not None else None


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