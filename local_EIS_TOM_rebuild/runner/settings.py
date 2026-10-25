"""
settings.py
===========
Every fixed setting of the runner notebook, in one place.

The notebook itself only creates the session and calls the plots; what used
to be scattered over ~1 300 lines of its cells (paths, datago tables, source
switches, the fixed evaluation, the cache layout, the plot sizes) is here.
Edit this file -- not the notebook -- when a path moves or a setting changes.
"""

from __future__ import annotations

from pathlib import Path

#: Shown in every run's identity file and printed by the setup cell.
RUNNER_VERSION = "v35.1"

# ---------------------------------------------------------------------------
# where things are
# ---------------------------------------------------------------------------
EV_ROOT = Path("/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev")
FAMOS_ROOT = EV_ROOT / "Famos"              # the FAMOS card .DAT files
GAMRY_ROOT = EV_ROOT / "Gamry"              # .dta sweeps and the bench .mf4
EIS_CONFIG_ROOT = EV_ROOT / "EIS_config"    # curr/temp/chain gain per plate
CSV_ROOT = EV_ROOT / "csv_files"            # bench-tool CSV measurements
RESULTS_ROOT = EV_ROOT / "EIS_Results"      # the saved results (the "cache")

#: Folders searched for an order's .dta sweeps, in order ({ev}, {leepa}).
GAMRY_SEARCH_ROOT_TEMPLATES = (
    "{ev}/RO{leepa}_Gamry",
    "{ev}/{leepa}_Gamry",
    "{ev}/Gamry",
)
#: Folders whose file names tie an order to its build token (V26_xxx).
GAMRY_VERSION_SOURCES = (GAMRY_ROOT, EV_ROOT, FAMOS_ROOT)
#: Last resort when no file name states the build: {'2612030': 'V26_092'}.
PLATE_VERSION_OVERRIDE: dict[str, str] = {}

# ---------------------------------------------------------------------------
# datago
# ---------------------------------------------------------------------------
DATAGO_META_TBL = "ps_xplatform_dev.rvadvtec_ops.datago_advtec_metadata"
DATAGO_FILES_TBL = "ps_xplatform_dev.rvadvtec_ops.datago_advtec_metafiles_delta"
DATAGO_GP_TBL = "ps_xplatform_dev.rvadvtec_ops.datago_advtec_generalproperties"
DATAGO_VAL_TBL = "ps_xplatform_dev.rvadvtec_ops.datago_advtec_values_delta"
DATAGO_LOCAL_TZ = "Europe/Berlin"      # the Gamry's STARTTIME is local time
DATAGO_MIN_SWEEP_POINTS = 10           # fewer: take the Volume .dta instead

#: DATAGO_FIRST  True : the file kinds switched on below come from datago,
#:                      with the Volume as fallback
#:               False: nothing from datago, everything from the Volume
#: The three switches below choose per file kind (all need DATAGO_FIRST = True).
DATAGO_FIRST = True
GAMRY_FROM_DATAGO = True      # the Gamry .dta sweeps
BENCH_FROM_DATAGO = True      # the bench .mf4 (MF4 plot)
#: The FAMOS card recordings. True: each card of the run is rebuilt from
#: datago and checked before use -- one signal per channel, and the same
#: samples as the Volume .DAT when that exists. A card that fails (datago's
#: import of FAMOS-v2 / DASYLab 14 cards returns UC2 under every channel
#: name, RO2612085 450 A Karte_3) is read from the Volume instead. The
#: channel (multiplexer slot) order, which datago does not keep, is taken
#: from a .DAT of the same card on the Volume. Slower than the Volume:
#: ~30-60 s per card.
FAMOS_FROM_DATAGO = True
#: A card whose datago copy is rejected (or missing) is read from the Volume
#: instead -- that card only; the others stay from datago. The Volume copy is
#: found by order, condition, test and card number anywhere under FAMOS_ROOT
#: (three folders deep), whatever its naming convention. False: such a card
#: is left out.
FAMOS_VOLUME_FALLBACK = True
#: Where datago keeps the ORIGINAL uploaded .DAT files, if it does. A card is
#: first taken from its original upload (exact, any FAMOS version -- the only
#: datago source that holds a FAMOS-v2 card's segment channels, because the
#: datago rows of such a card hold UC2 under every name). Empty: found
#: automatically from any path-like column of the datago file record
#: (metafiles_delta), from its bucket + folder columns, and by the file's
#: name under FAMOS_ROOT (three levels down). Or a template, e.g.
#:   "abfss://landing@<account>.dfs.core.windows.net/datago/{raw_file_name}"
#:   "abfss://{bucket}@<account>.dfs.core.windows.net/{folder}/{raw_file_name}"
#:   "/Volumes/<catalog>/<schema>/<volume>/{order}/{raw_file_name}"
#: with {raw_file_name}, {order} (e.g. RO2612086) and every column of the
#: file record: {file_id}, {bucket}, {folder}, {source}, ... A run that
#: cannot find the file prints the record's bucket / folder / source.
DATAGO_RAW_FILE_TEMPLATE = ""

# ---------------------------------------------------------------------------
# the evaluation (fixed; part of every run's identity)
# ---------------------------------------------------------------------------
F_MIN_HZ = 0.15
F_MAX_HZ = 30000.0            # detection ceiling; the band ends at 0.49*fs
FULL_BAND = True              # False: the 'trusted' band, 0.15 Hz - 1.2 kHz
MIN_SNR_DB = -10.0            # validated on 2612087 (Test012 / Test0112)
EVALUATION_MODE = "default"   # 'default' | 'permissive' | 'strict'
CHANNEL_LAG = "correct"       # 'correct' | 'report' | 'off'
                              #   per-segment chain lag, removed or only written
GAMRY_SYNC = "guide"          # 'guide' | 'frequency' | 'report' | 'off'
                              #   guide: Gamry's exact frequencies + windows
                              #   re-located in the Gamry slot; frequency: only
                              #   the exact frequencies; report: only
                              #   bronze/gamry_sync.csv, nothing changed.
                              #   Without guide the top steps (>3 kHz) are
                              #   placed blind and most of them are lost
GAMRY_MATCH = "file"          # 'file' | 'time'
                              #   which Gamry sweep belongs to the run: file =
                              #   order / build / current / test from the file
                              #   name, time stamps not used; time = only a
                              #   sweep within 30 min of the FAMOS stamps.
                              #   Synchronisation itself never uses stamps:
                              #   cards by UC2, Gamry by its frequency steps
CARD_GAIN = "correct"         # 'correct' | 'report' | 'off'
                              #   each card's UC2 gain against the others,
                              #   divided out of Z or only written
UC_SERIES_MOHM_CM2 = 0.0
FREQ_RESPONSE = "report"      # 'report' | 'off' (never changes Z)
HEATMAP_STYLE = "interpolated"
CSV_MIRROR_X = False
#: 'auto': apply <RESULTS_ROOT>/<order>/hf_response/hf_response.csv when it
#: exists; '' never; a path: that file.
HF_RESPONSE = "auto"
#: R.build_hf_response(): 'segment' gives channels that roll off (|H| more
#: than 25 % down at the top, the same at every current -- interface boxes
#: r5 / r6 on 2612087) their own magnitude, every other channel the card's;
#: 'card' is the card magnitude for all (the files built before v35).
HF_RESPONSE_MAG = "segment"

# ---------------------------------------------------------------------------
# step detection and timing at high frequency (part of every run's identity)
# ---------------------------------------------------------------------------
# The values of the validated 2612087 runs (150A_Test012, 450A_Test0112):
# 52-53 of 66-67 segments reach 11.95 kHz. Before v35.1 these lived only in
# core/config.py, so a package update reset them to the library defaults
# (report / none / False / False / 0.45 / -3: the band then ends at 1.2-3 kHz).
SKEW_MODEL = "structural"     # 'structural' | 'parity' | ... | 'none'
                              #   the multiplexer's slot delay; 'none' leaves
                              #   20-35 us across a card = 86-151 deg at 12 kHz
HF_USE_ENSEMBLE = True        # detect the steps on the stacked segment
                              #   channels (+10 dB), not on UC2 alone
HF_LADDER_PRUNE = True        # drop detections that miss the 10/decade ladder
F_HI_FRAC_FS = 0.49           # detection ceiling, fraction of fs (12.25 kHz)
WINDOW_MIN_SNR_DB = -4.0      # analysis-window SNR floor at the top steps

# ---------------------------------------------------------------------------
# saved results: one folder per order / condition / test, named in order
# ---------------------------------------------------------------------------
#   EIS_Results/2612086/60A_Test02/            bronze/ silver/ gold/ ...
#   EIS_Results/2612086/60A_Test02/plots/      the plots of that condition
#   EIS_Results/2612086/plots/                 the drop-down viewers
# gen2 results go to <order>_gen2/, a plate mounted rotated 180 deg to
# <order>_rot180/ (<order>_gen2_rot180/), a non-default evaluation mode to
# <condition>_Test<nn>_<mode>/.
RUN_MODES = ("cache", "rerun")   # rerun recomputes AND overwrites the saved one

# ---------------------------------------------------------------------------
# tables: the silver / gold layer as Delta tables (runner/tables.py)
# ---------------------------------------------------------------------------
#   <TABLE_SCHEMA>.<TABLE_PREFIX>runs, ..._gold_segments, ..._silver_spectra, ...
# one row set per run, keyed by run_key = '<order folder>/<condition folder>'
# (the folder under EIS_Results); publishing a run replaces its rows.
TABLE_SCHEMA = "ps_xplatform_dev.rvadvtec_dev"   # catalog.schema (needs CREATE TABLE)
TABLE_PREFIX = "local_eis_"
PUBLISH_TABLES = True            # 01 Process publishes every run it saves
#: where "02 Local EIS - Results" reads a run from:
#:   'tables'  the Delta tables (<prefix>run_files holds every file of the
#:             run; it is restored into the scratch folder and shown from
#:             there); a run only in a folder (never published) is read
#:             from its folder
#:   'folders' the saved folders under RESULTS_ROOT only (as before)
RESULTS_SOURCE = "tables"
#: 01 Process, Run mode = cache: a run whose folder is gone from
#: RESULTS_ROOT but is published is restored from the tables into its
#: folder instead of evaluated again
CACHE_FROM_TABLES = True

# ---------------------------------------------------------------------------
# the automatic job (03 Local EIS - Auto, runner/auto.py)
# ---------------------------------------------------------------------------
#: orders the job looks at: () = every order with FAMOS data (datago and
#: Volume), or a list ("2612090", "2612091")
AUTO_ORDERS: tuple = ()
#: conditions evaluated per job execution (each takes 5-25 min); the rest
#: follow at the next execution
AUTO_MAX_RUNS = 8
#: plate per order when the file names do not say it (bench .mf4 names carry
#: "Naboo" = gen2 / "Kashyyyk" = gen1); otherwise gen1
AUTO_PLATE: dict[str, str] = {}
#: orders mounted turned 180 deg: {"2612095": "rot180"}; otherwise standard
AUTO_ORIENTATION: dict[str, str] = {}
#: True: a run published with OTHER settings (another digest) is evaluated
#: again; False: only runs that were never published
AUTO_RERUN_CHANGED = True

# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
PANEL_HEIGHT = 560
PANEL_WIDTH = 1150
NYQUIST_COLOUR = "flow"          # 'flow' (inlet -> outlet) | 'segment'
MF4_INCLUDE_ALL_CHANNELS = True  # every channel of the bench log, not only
                                 # the groups of bench_plots.CHANNEL_GROUPS
MF4_MAX_POINTS = 4000            # per channel; min/max kept, spikes survive
ECM_DRT_F_MAX_HZ = 1000.0        # the DRT-informed fit stops here
ECM_DRT_FIX_RS = True            # ... and holds Rs at the pipeline's R_ohmic
ECM_DRT_FAST = "auto"            # 'zarc' | 'tlm' | 'auto'
ECM_SMOOTH_POINTS = 160          # points of each drawn model curve
