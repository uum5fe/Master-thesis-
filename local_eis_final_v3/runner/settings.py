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
RUNNER_VERSION = "v34"

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
FAMOS_FROM_DATAGO = False
#: Where datago keeps the ORIGINAL uploaded .DAT files, if it does. A card is
#: first taken from its original upload (exact, any FAMOS version -- the only
#: datago source that holds a FAMOS-v2 card's segment channels, because the
#: datago rows of such a card hold UC2 under every name). Empty: found
#: automatically from any path-like column of the datago file record
#: (metafiles_delta). Or a template, e.g.
#:   "abfss://landing@<account>.dfs.core.windows.net/datago/{raw_file_name}"
#:   "/Volumes/<catalog>/<schema>/<volume>/{order}/{raw_file_name}"
#: with {raw_file_name}, {file_id}, {order} (e.g. RO2612086).
DATAGO_RAW_FILE_TEMPLATE = ""

# ---------------------------------------------------------------------------
# the evaluation (fixed; part of every run's identity)
# ---------------------------------------------------------------------------
F_MIN_HZ = 0.15
F_MAX_HZ = 30000.0            # detection ceiling; the band ends at 0.49*fs
FULL_BAND = True              # False: the 'trusted' band, 0.15 Hz - 1.2 kHz
MIN_SNR_DB = 0.0
EVALUATION_MODE = "default"   # 'default' | 'permissive' | 'strict'
CHANNEL_LAG = "correct"       # 'correct' | 'report' | 'off'
                              #   per-segment chain lag, removed or only written
GAMRY_SYNC = "guide"          # 'guide' | 'frequency' | 'report' | 'off'
                              #   guide: Gamry's exact frequencies + windows
                              #   re-located in the Gamry slot; frequency: only
                              #   the exact frequencies; report: only
                              #   bronze/gamry_sync.csv, nothing changed
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
