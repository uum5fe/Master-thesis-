# Databricks notebook source
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup: sys.path + module imports
# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL EIS PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════
# Notebook entry point for the modular bronze/silver/gold pipeline.
# Selects measurement by Order ID (Leepa) from datago, runs the full
# pipeline, and displays Nyquist + heatmap results inline.
#
# Modules: config.py, utils.py, bronze.py, silver.py, gold.py, csv_source.py,
#          csv_pipeline.py, gamry_dta.py, abgleich.py, r2d2_geometry.py
# ═══════════════════════════════════════════════════════════════════════════════
import sys
from pathlib import Path
 
# ─── Find the module directory: the folder this notebook sits in ───
# Hard-coding a workspace path means the notebook only runs for whoever
# uploaded it, and breaks silently when the folder is renamed.  The modules
# are always beside the notebook, so ask Databricks where the notebook is and
# work from there.  `_PIPELINE_DIR_OVERRIDE` is the escape hatch for the case
# where they are deliberately kept somewhere else.
_PIPELINE_DIR_OVERRIDE = ''      # e.g. '/Workspace/Users/you@bosch.com/Local_EIS_pipeline'
 
 
def _find_pipeline_dir():
    if _PIPELINE_DIR_OVERRIDE:
        return _PIPELINE_DIR_OVERRIDE
    try:
        nb = (dbutils.notebook.entry_point.getDbutils().notebook()
              .getContext().notebookPath().get())
        cand = '/Workspace' + str(Path(nb).parent)
        if (Path(cand) / 'config.py').exists():
            return cand
    except Exception:
        pass
    # Repos, a local checkout, or anything else: fall back to the cwd and to
    # the historical location, and say which one was used.
    for cand in (str(Path.cwd()),
                 '/Workspace/Users/uum5fe@bosch.com/Local_EIS_pipeline',
                 '/Workspace/Users/uum5fe@bosch.com/Local_EIS_fixed/Local_EIS_fixed'):
        if (Path(cand) / 'config.py').exists():
            return cand
    raise FileNotFoundError(
        "cannot find the pipeline modules. config.py should sit in the same "
        "folder as this notebook; if it does not, set "
        "_PIPELINE_DIR_OVERRIDE at the top of this cell.")
 
 
_PIPELINE_DIR = _find_pipeline_dir()
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)
 
# ─── Shim the missing 'core' package ───
# bronze.py, silver.py and gold.py do `import core` to put a core directory on
# sys.path.  There is no core/ subdirectory; the modules it was meant to
# provide (r2d2_geometry, eis_local) sit alongside them, so point the shim at
# the pipeline directory itself.
import types
_core_mod = types.ModuleType('core')
_core_mod.__path__ = [_PIPELINE_DIR]
sys.modules['core'] = _core_mod
 
import importlib
importlib.invalidate_caches()
import numpy as np
import pandas as pd
 
# Import pipeline modules
import config
import utils
import eis_local
import bronze
import silver
import gold
import main as pipeline_main
import r2d2_geometry as geom
import csv_source
import csv_pipeline
import gamry_dta
import gamry_compare
import abgleich
import ladder_snap
import eis_measurement_model
import figure_panels
import plate_maps
 
# Force reload during development
for mod in [config, utils, eis_local, bronze, silver, gold, pipeline_main,
            geom, csv_source, csv_pipeline, gamry_dta, gamry_compare, abgleich,
            ladder_snap, eis_measurement_model, figure_panels, plate_maps]:
    importlib.reload(mod)

# ─── ONE PLOT PER FIGURE ───
# Every spectrum cell below still builds its multi-panel figure (Nyquist, |Z|,
# phase side by side, or a grid of segments) and hands it to show_fig(), which
# draws each panel as its own full-width figure, one below the other. Set
# PLOTS_ONE_BY_ONE = False to get the old side-by-side rows back.
PLOTS_ONE_BY_ONE = True
PANEL_HEIGHT = 560        # px, per single plot
PANEL_WIDTH = 1150        # px; None = fill the cell width


def show_fig(fig, html=False):
    """Show a (possibly multi-panel) Plotly figure, one panel at a time."""
    figs = (figure_panels.split_subplots(fig, panel_height=PANEL_HEIGHT,
                                         panel_width=PANEL_WIDTH)
            if PLOTS_ONE_BY_ONE else [fig])
    for f in figs:
        if html:
            displayHTML(f.to_html(full_html=False, include_plotlyjs='cdn'))
        else:
            f.show()
 
from config import Config, DEFAULT
 
print("═" * 75)
print("  LOCAL EIS PIPELINE — Notebook Runner")
print(f"  Modules loaded from: {_PIPELINE_DIR}")
for _k, _p in geom.PLATES.items():
    _chk = geom.self_check(verbose=False, plate_name=_k)
    print(f"  Plate {_k:5s} ({_p.colour}/{_p.name}): {_chk['n_segments']} "
          f"segments, {_chk['area_min_cm2']:.2f}–{_chk['area_max_cm2']:.2f} cm², "
          f"{'OK' if not _chk['problems'] else 'PROBLEM: ' + '; '.join(_chk['problems'])}")
print("═" * 75)

# COMMAND ----------

# DBTITLE 1,Widgets: Order ID, Condition, Band, SNR, Stop-After
# ═══════════════════════════════════════════════════════════════════════════════
# WIDGETS  —  Order ID, Condition, Band, SNR gate, Pipeline stage
# ═══════════════════════════════════════════════════════════════════════════════
 
# ─── Discover available Order IDs from datago + Volumes ───
_META_TBL = 'ps_xplatform_dev.rvadvtec_ops.datago_advtec_metadata'
_GP_TBL = 'ps_xplatform_dev.rvadvtec_ops.datago_advtec_generalproperties'
 
try:
    _rows = spark.sql(f"""
        SELECT DISTINCT m.orderId
        FROM {_META_TBL} m
        JOIN {_GP_TBL} gp ON m.measurement_id = gp.measurement_id
        WHERE gp.measurement_type = 'GALVEIS'
        ORDER BY m.orderId
    """).collect()
    AVAILABLE_ORDERS = [r['orderId'].replace('RO', '') for r in _rows if r['orderId']]
except Exception:
    AVAILABLE_ORDERS = ['2611976']
 
# Also scan Volumes for Leepa IDs not in datago (e.g. 2612025)
try:
    import re as _re
    _vol_root = Path('/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev/Famos')
    _vol_ids = set()
    for _f in _vol_root.glob('*_Current_*_Karte_*.DAT'):
        _m = _re.search(r'(?:Leepa_|RO)(\d{7})', _f.name)
        if _m:
            _vol_ids.add(_m.group(1))
    _new = sorted(_vol_ids - set(AVAILABLE_ORDERS))
    if _new:
        AVAILABLE_ORDERS = sorted(set(AVAILABLE_ORDERS) | _vol_ids)
        print(f"  +{len(_new)} Leepa IDs from Volumes (not in datago): {_new}")
except Exception:
    pass
 
# ─── Fixed paths ───
FAMOS_ROOT = Path('/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev/Famos')
_EV_ROOT = Path('/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev')

# ═══════════════════════════════════════════════════════════════════════════
#  WHICH ORDER, DECIDED BEFORE ANYTHING THAT DEPENDS ON IT
# ═══════════════════════════════════════════════════════════════════════════
# Everything below this point -- the condition list, the Gamry folders, the
# build token -- is a property of ONE order, so the order has to be known
# first. It was being read at the bottom of the cell instead, which on a
# fresh kernel meant LEEPA did not exist yet when those blocks ran.
#
# That did not fail loudly, which is worse: the condition discovery sits in a
# try/except, so the NameError was swallowed and CONDITIONS silently fell
# back to the hard-coded ['450A', '60A', '45A', '150A'] -- the dropdown was
# never actually reading the disk on the first run of a session. It worked on
# the SECOND run, because by then LEEPA was left over from the first.
#
# The widget is therefore created here, read here, and the order-dependent
# discovery follows it.
def _w(name, default=''):
    try:
        return dbutils.widgets.get(name)
    except Exception:
        return default


_default = '2612025' if '2612025' in AVAILABLE_ORDERS else AVAILABLE_ORDERS[-1]
try:
    dbutils.widgets.dropdown('leepa_id', _default, AVAILABLE_ORDERS,
                             'Order ID (Leepa)')
except Exception:
    pass
LEEPA = _w('leepa_id', _default)

def condition_sort_key(cond):
    """'45A' < '60A' < '150A' < '450A': by the number, then by the text."""
    import re as _re_k
    m = _re_k.match(r'\s*([0-9]+(?:\.[0-9]+)?)', str(cond))
    return (float(m.group(1)) if m else float('inf'), str(cond))


def parse_conditions(raw, available):
    """The multi-select value -> the conditions to evaluate.

    Databricks hands a multiselect back as one comma-separated string
    ("45A,450A"). Empty, or anything containing ALL, means every condition
    on disk. The result is in current order whatever order they were
    clicked in, and a name that is not on disk is kept (and reported by the
    run cell as "nothing found") rather than silently dropped.
    """
    picked = [x.strip() for x in str(raw or '').split(',') if x.strip()]
    if not picked or any(x.upper() == 'ALL' for x in picked):
        return list(available)
    out = []
    for c in picked:
        if c not in out:
            out.append(c)
    return sorted(out, key=condition_sort_key)


# Discover conditions from filenames on disk for the SELECTED Leepa
# This handles both Leepa_{id} and Leepa_RO{id} naming conventions
try:
    import re as _re2
    _all_conds = set()
    for _pat in [f'Leepa_{LEEPA}_Current_*_Karte_*.DAT',
                 f'Leepa_RO{LEEPA}_Current_*_Karte_*.DAT',
                 f'RO{LEEPA}-*_Current_*_Karte_*.DAT']:
        for _f in FAMOS_ROOT.glob(_pat):
            _cm = _re2.search(r'_Current_([^_]+)_', _f.name)
            if _cm:
                _all_conds.add(_cm.group(1))
    # Ordered by current (45A, 60A, 150A, 450A), not as strings -- the
    # plots come out one condition after another in this order.
    CONDITIONS = (sorted(_all_conds, key=condition_sort_key) if _all_conds
                  else ['450A', '60A', '45A', '150A'])
    if not _all_conds:
        print(f"  no FAMOS files found for order {LEEPA} under {FAMOS_ROOT}; "
              f"condition list is the hard-coded fallback")
except Exception as _ce:                                   # noqa: BLE001
    # Say WHY. A silent fallback here is indistinguishable from a plate that
    # genuinely has these four conditions, and it is how a broken discovery
    # went unnoticed.
    print(f"  condition discovery failed ({type(_ce).__name__}: {_ce}); "
          f"falling back to the hard-coded list")
    CONDITIONS = ['450A', '60A', '45A', '150A']
 
# ═══════════════════════════════════════════════════════════════════════════
#  GAMRY REFERENCE FILES — THE ONE BLOCK TO EDIT IF THE PATHS EVER MOVE
# ═══════════════════════════════════════════════════════════════════════════
# A whole-cell sweep is named        V26_092_HFR_101_CurrVal_45.dta
# and the measurement file for the
# same cell is named                 ..._RO2612030-01_V26_092_lokale_EIS_...mf4
#
# The sweep carries the BUILD TOKEN and no order number; the measurement file
# carries both. So the chain is
#
#     Leepa 2612030  ->  build V26_092  ->  V26_092_*.dta
#
# and the middle step is read off the measurement filenames rather than
# hard-coded. Selecting a different order in the dashboard therefore selects
# a different build and a different set of sweeps, with no table to maintain.
#
# Why this matters more than it looks: a shared Gamry folder holds several
# campaigns, the sweeps are keyed by CURRENT downstream, and every campaign
# has a 45 A. Reading the folder whole does not fail loudly -- the last file
# read replaces the earlier one, and the cell is then compared against another
# cell's reference with nothing visibly wrong.

#: Folders searched for .dta sweeps, in order. ADD NEW LOCATIONS HERE.
#: Written as TEMPLATES rather than as finished paths: a list built once from
#: whatever LEEPA happened to hold is wrong twice over -- it cannot be built
#: before the order is known, and it silently keeps the old order's folders
#: if the widget changes and only the later cells are re-run.
GAMRY_SEARCH_ROOT_TEMPLATES = [
    '{ev}/RO{leepa}_Gamry',       # per-order folder, if there is one
    '{ev}/{leepa}_Gamry',
    '{ev}/Gamry',                 # the shared folder
]


def gamry_search_roots(leepa):
    return [Path(t.format(ev=_EV_ROOT, leepa=leepa))
            for t in GAMRY_SEARCH_ROOT_TEMPLATES]

#: Folders searched for the measurement files that carry order AND build.
GAMRY_VERSION_SOURCES = [
    _EV_ROOT / 'Gamry',
    _EV_ROOT,
    FAMOS_ROOT,
]

#: Last resort when no filename states the build: '2612030': 'V26_092'.
#: Prefer fixing the filenames; a table here is a fact that can go stale.
PLATE_VERSION_OVERRIDE: dict[str, str] = {}


def plate_version(leepa, quiet=True):
    """The build token for this order, e.g. 'V26_092', or None.

    Read from any measurement filename that names BOTH the order and a build.
    """
    leepa = str(leepa)
    if leepa in PLATE_VERSION_OVERRIDE:
        return PLATE_VERSION_OVERRIDE[leepa]
    digits = ''.join(ch for ch in leepa if ch.isdigit())
    seen: dict[str, int] = {}
    for root in GAMRY_VERSION_SOURCES:
        try:
            if not Path(root).is_dir():
                continue
            for f in Path(root).iterdir():
                if digits not in f.name:
                    continue
                v = gamry_compare.version_of(f.name)
                if v:
                    seen[v] = seen.get(v, 0) + 1
        except Exception:                                  # noqa: BLE001
            continue
    if not seen:
        return None
    if len(seen) > 1 and not quiet:
        # Two builds naming one order is a question about the data, not
        # something to average away.
        print(f"  WARNING: order {leepa} appears with more than one build "
              f"{dict(seen)}; using the most frequent. Set "
              f"PLATE_VERSION_OVERRIDE['{leepa}'] to choose.")
    return max(seen, key=lambda v: seen[v])


def gamry_root_for(leepa):
    """First folder from the search roots that holds any .dta, or None."""
    for root in gamry_search_roots(leepa):
        try:
            r = Path(root)
            if r.is_dir() and (any(r.glob('*.dta')) or any(r.glob('*.DTA'))):
                return r
        except Exception:                                  # noqa: BLE001
            continue
    return None


def gamry_files(leepa, version=None):
    """The .dta sweeps that belong to this order. Never another campaign's.

    A file naming a different build is excluded. A file naming no build is
    kept only when nothing names this one -- the same last-resort rule
    gamry_compare uses for order numbers, because a folder holding exactly
    one campaign has no reason to spell the build out.
    """
    root = gamry_root_for(leepa)
    if root is None:
        return [], None
    version = version or plate_version(leepa)
    files = sorted(list(root.glob('*.dta')) + list(root.glob('*.DTA')))
    files = [f for f in files if '_raw' not in f.stem.lower()]

    if not version:
        # No build for this order. That is survivable in a folder holding ONE
        # campaign and not survivable in a folder holding several: the sweeps
        # are keyed by current downstream and every campaign has a 45 A, so
        # handing back a mix means one of them silently wins. Refuse instead.
        builds = {gamry_compare.version_of(f.name) for f in files}
        builds.discard(None)
        if len(builds) > 1:
            print(f"  REFUSING to pick Gamry sweeps for order {leepa}: "
                  f"{root} holds {len(builds)} campaigns "
                  f"({', '.join(sorted(builds))}) and nothing names the build "
                  f"for this order.\n"
                  f"    Fix: add the order's measurement .mf4 (the one named "
                  f"..._RO{leepa}-01_V26_xxx_...) to GAMRY_VERSION_SOURCES, "
                  f"or set PLATE_VERSION_OVERRIDE['{leepa}'] = 'V26_xxx' in "
                  f"the setup cell.")
            return [], None
        return files, (builds.pop() if builds else None)

    named = [f for f in files
             if gamry_compare.names_version(f.name, version) is True]
    if named:
        return named, version
    # Nothing names this build. Files that name ANOTHER one are already out;
    # what is left is silent and may legitimately be this campaign's.
    silent = [f for f in files
              if gamry_compare.names_version(f.name, version) is None]
    if silent:
        print(f"  no sweep names build {version}; falling back to "
              f"{len(silent)} file(s) that name no build at all")
    return silent, version


# Gamry reference: auto-discover per Leepa (may not exist for every order)
try:
    GAMRY_ROOT = gamry_root_for(LEEPA)
    GAMRY_VERSION = plate_version(LEEPA, quiet=False) or ''
except Exception:                                          # noqa: BLE001
    GAMRY_ROOT, GAMRY_VERSION = None, ''
 
# ─── Remove old widgets that are no longer needed ───
# 'condition' was a single-choice dropdown; it is replaced by the
# 'conditions' multi-select below (a widget cannot change type in place).
for _old in ('plate', 'csv_path', 'csv_dialect', 'csv_tones',
             'gain_file', 'gamry_dir', 'bench_log', 'condition'):
    try:
        dbutils.widgets.remove(_old)
    except Exception:
        pass
 
# ─── Create the remaining widgets ───
# leepa_id is NOT here: it is created above, before the discovery that needs
# to know which order is selected. The condition dropdown is here because its
# choices come from that discovery.
try:
    # MULTI-SELECT: tick e.g. 45A and 450A to evaluate only those two.
    # ALL (or nothing ticked) evaluates every condition found on disk.
    dbutils.widgets.multiselect('conditions', 'ALL', ['ALL'] + CONDITIONS,
                                'Conditions (multi-select)')
    # PARAMETER PROFILE -- see the RECOMMENDED_PARAMS block below.
    #   recommended  evaluation mode, SNR gate and band are FIXED to the
    #                values that give clean spectra; the three widgets for
    #                them are ignored (and the printout says so)
    #   custom       the Evaluation mode / Min SNR / F min / F max widgets
    #                decide, exactly as before
    dbutils.widgets.dropdown('param_profile', 'recommended',
                             ['recommended', 'custom'], 'Parameter profile')
    dbutils.widgets.text('f_min_hz', '0.15', 'F min (Hz)')
    dbutils.widgets.text('f_max_hz', '2000.0', 'F max (Hz)')
    dbutils.widgets.dropdown('min_snr_db', '5',
                             ['-30', '-20', '-10', '-3', '0', '3', '5', '8',
                              '10'],
                             'Min SNR (dB)')
    dbutils.widgets.dropdown('source_format', 'famos',
                             ['famos', 'csv'], 'Measurement file format')
    # stop_after:
    #   bronze = raw phasor extraction only (fastest, for debugging)
    #   silver = + de-skew + measurement model + lin-KK validation
    #   gold   = + spatial field, plate maps, plausibility (full run)
    dbutils.widgets.dropdown('stop_after', 'gold',
                             ['bronze', 'silver', 'gold'], 'Stop after')
    # EVALUATION MODE IS A CHOICE, NOT A DEFAULT.
    # This notebook used to relax snr_floor_db to -80, max_drift to 2.5 and
    # max_thd to 1.0 inside the config it built, so every run it produced was
    # an exploratory run wearing the pipeline's default clothes. Those
    # settings are legitimate for looking at a noisy top-of-band; they are
    # not legitimate as the number that goes in a thesis without being named.
    # 'default' now means the shipped gates, and anything looser has to be
    # asked for here and is carried into the cache path and the manifest.
    dbutils.widgets.dropdown('evaluation_mode', 'default',
                             ['default', 'permissive', 'strict'],
                             'Evaluation mode')
    # Segments to leave out of the WHOLE evaluation, comma separated, e.g.
    # "33" or "33,59". An excluded segment is skipped in bronze before its
    # channel is read, so it has no spectrum, no scalars, no place in the cell
    # aggregate and no inferred value on any map -- and the run says so rather
    # than leaving a silent hole. Empty is the default on purpose: excluding a
    # segment before its data is seen removes the only evidence that could
    # ever overturn the exclusion.
    dbutils.widgets.text('exclude_segments', '', 'Exclude segments (e.g. 33)')
    # Segments whose OWN measurement is discarded and rebuilt from the
    # segments touching them. Different from excluding: the plate still has
    # that area and it still conducts, so it stays in the aggregate and on
    # the map -- as an estimate, labelled, with its donors recorded.
    dbutils.widgets.text('substitute_segments', '',
                         'Rebuild from neighbours (e.g. 33)')
    # The same reconstruction for every segment that has no spectrum at all.
    # Off by default: it changes the cell aggregate, so it is a choice that
    # has to be made deliberately and it is recorded in the cache key.
    dbutils.widgets.dropdown('fill_gaps', 'no', ['no', 'yes'],
                             'Fill unmeasured segments from neighbours')
except Exception:
    pass
 
 
# _w is defined above, with the order resolution it serves.

def _widget(*names, default=''):
    """First widget that exists, else a notebook global, else the default.

    Written this way because widget names have differed between versions of
    this runner and a NameError in a display cell kills the whole cell.
    Defined HERE, once, rather than re-pasted in each display cell: the copies
    are what let those cells drift apart about which run they were showing.
    """
    for n in names:
        try:
            v = dbutils.widgets.get(n)
            if v not in (None, ''):
                return v
        except Exception:
            pass
        g = globals().get(n) or globals().get(n.upper())
        if g not in (None, ''):
            return g
    return default
 
 
# ─── Read widget values ───
PLATE = 'gen1'  # always gen1 for FAMOS
SOURCE_FORMAT = _w('source_format', 'famos')
CSV_PATH = ''
CSV_DIALECT = 'auto'
CSV_TONES = ()
GAIN_FILE = ''
GAMRY_DIR = str(GAMRY_ROOT) if GAMRY_ROOT else ''
# The build token that ties this order to its sweeps. It travels into the
# Config, so the pipeline's own whole-cell comparison filters on it too --
# not just the display cells below.
GAMRY_VERSION = globals().get('GAMRY_VERSION', '')
BENCH_LOG = ''
# LEEPA was resolved at the top of this cell, before the discovery that uses
# it. Re-read here only so that changing the widget and re-running from this
# point still picks the change up.
LEEPA = _w('leepa_id', _default)
SELECTED_CONDITIONS = parse_conditions(_w('conditions', 'ALL'), CONDITIONS)
COND_FILTER = ('ALL' if SELECTED_CONDITIONS == list(CONDITIONS)
               else ', '.join(SELECTED_CONDITIONS))
F_MIN = float(_w('f_min_hz', '0.15'))
F_MAX = float(_w('f_max_hz', '2000.0'))
MIN_SNR_DB = float(_w('min_snr_db', '5'))
STOP_AFTER = _w('stop_after', 'gold')
EVALUATION_MODE = _w('evaluation_mode', 'default')

# ═══════════════════════════════════════════════════════════════════════════
#  RECOMMENDED PARAMETERS -- WHY THE 150 A / 450 A SPECTRA WERE SCATTERED
# ═══════════════════════════════════════════════════════════════════════════
# The scattered Nyquist plots (points flung to -Z'' = -90 .. +110, spikes at
# 1, 3, 10 and 20 Hz on one group of segments, phase diving to -80 deg above
# 1 kHz) were read from  <cond>/mode_permissive/snr_0.0_<digest>/  -- the run used
# Evaluation mode = permissive and Min SNR = 0 dB, f_max = 4500 Hz. The clean
# arcs of the earlier run came from the shipped gates. What each setting did:
#
#   permissive preset   sigma_rel_max 0.60 -> 1.5   keeps phasors whose
#                                                   propagated uncertainty
#                                                   is 150 % of |Z| -- noise
#                       zmag_outlier_mad 4.5 -> 8   lets the single-frequency
#                                                   |Z| spikes through
#                       max_thd / max_drift -> 0.5  keeps distorted and
#                                                   non-stationary dwells
#                       min_cycles 3 -> 1           one-cycle phasors at the
#                                                   low-frequency end
#   Min SNR 0 dB        bronze uses this number to accept OFF-grid steps,
#                       to pick the basis of the grid fit and to decide the
#                       polarity. At 0 dB it admits steps the pipeline's own
#                       default (10 dB) and the version that drew the clean
#                       arcs (5 dB) refuse. Silver's per-point backstop is a
#                       separate setting and is not affected.
#   f_max 4500 Hz       the rungs above ~2 kHz come out capacitive (phase
#                       -40 .. -80 deg) on most segments AND in the area-
#                       weighted aggregate, while the Gamry sweep of the same
#                       cell reads about -10 deg there. That is not the cell;
#                       the clean plot never showed anything above 2 kHz.
#
# The 'recommended' profile pins these three to the values below. Pick
# 'custom' in the Parameter profile widget to explore anything else -- the
# mode and settings are still part of the cache key, so an exploratory run
# never overwrites a recommended one.
RECOMMENDED_PARAMS = dict(evaluation_mode='default', min_snr_db=5.0,
                          f_min_hz=0.15, f_max_hz=2000.0)
PARAM_PROFILE = _w('param_profile', 'recommended')
if PARAM_PROFILE == 'recommended':
    _now = dict(evaluation_mode=EVALUATION_MODE, min_snr_db=MIN_SNR_DB,
                f_min_hz=F_MIN, f_max_hz=F_MAX)
    _ignored = [f'{k}={v:g}' if isinstance(v, float) else f'{k}={v}'
                for k, v in _now.items() if v != RECOMMENDED_PARAMS[k]]
    EVALUATION_MODE = RECOMMENDED_PARAMS['evaluation_mode']
    MIN_SNR_DB = float(RECOMMENDED_PARAMS['min_snr_db'])
    F_MIN = float(RECOMMENDED_PARAMS['f_min_hz'])
    F_MAX = float(RECOMMENDED_PARAMS['f_max_hz'])
    if _ignored:
        print(f"  Parameter profile 'recommended': widget value(s) "
              f"{', '.join(_ignored)} ignored. Set the profile to 'custom' "
              f"to use them.")
def _seg_list(name):
    return frozenset(x.strip() for x in
                     _w(name, '').replace(';', ',').split(',') if x.strip())


EXCLUDE_SEGMENTS = _seg_list('exclude_segments')
SUBSTITUTE_SEGMENTS = _seg_list('substitute_segments')
FILL_GAPS = _w('fill_gaps', 'no') == 'yes'
 
# Select the plate for the whole session
_plate = geom.use_plate(PLATE)
 
print(f"  Leepa:     {LEEPA}")
print(f"  Condition: {COND_FILTER}   -> {', '.join(SELECTED_CONDITIONS)}")
print(f"  Profile:   {PARAM_PROFILE}   (mode {EVALUATION_MODE})")
print(f"  Band:      {F_MIN} – {F_MAX} Hz")
print(f"  Min SNR:   {MIN_SNR_DB} dB")
print(f"  Stop:      {STOP_AFTER}")
print(f"  Format:    {SOURCE_FORMAT}")
print(f"  Plate:     {_plate.title}")

# COMMAND ----------

# DBTITLE 1,Plate map: confirm the numbering before trusting any heat map
# ═══════════════════════════════════════════════════════════════════════════════
# Draw the selected plate.  Compare it against the coordinate drawing once,
# at the start of a campaign — this is the cheapest way to catch the one
# mistake that produces a plausible-looking wrong answer: a gen2 recording
# evaluated with the gen1 map.
#
# gen1 (green / Kashyyyk): segments 37..72 are six full-height edge strips.
# gen2 (blue  / Naboo)   : the edge segments are re-cut and interleaved into
#                          the wide strips at the top and bottom of the plate,
#                          so 49, 51, 55 and 57 sit along the TOP edge and
#                          52, 54, 58 and 60 along the bottom.
# Segments 1..36 keep their positions on both, but NOT their areas: the
# strips that gained a top/bottom edge segment lost pad rows to it.
# ═══════════════════════════════════════════════════════════════════════════════
import os, tempfile
from IPython.display import display, Image as IPImage
 
_map_png = Path(tempfile.gettempdir()) / f'plate_{PLATE}_{os.getuid()}.png'
geom.plot_map(_map_png)
print(f"  {_plate.title} — reconstructed from {_plate.drawing}")
_chk = geom.self_check(verbose=False)
print(f"  {_chk['n_segments']} segments, {_chk['pads_covered_once']}/900 pads "
      f"covered exactly once, area sum {_chk['area_sum_cm2']:.2f} cm², "
      f"areas {_chk['area_min_cm2']:.3f}–{_chk['area_max_cm2']:.3f} cm²")
if _chk['problems']:
    print('  PROBLEMS: ' + '; '.join(_chk['problems']))
display(IPImage(filename=str(_map_png)))
 
# Where a gen1 segment number lands on the gen2 plate, for orientation only —
# the two plates have no one-to-one segment correspondence, the edge segments
# were re-cut rather than renamed.
if PLATE == 'gen2':
    _ren = geom.renumbering('gen1', 'gen2')
    print("\n  gen1 segment -> the gen2 segment covering its centre "
          "(orientation only, NOT a data conversion):")
    print('  ' + ', '.join(f'{k}->{v}' for k, v in
                           sorted(_ren.items(), key=lambda kv: int(kv[0]))
                           if k != v))

# COMMAND ----------

# DBTITLE 1,Chain response: build the gain file from the Abgleich bode sweeps
# ═══════════════════════════════════════════════════════════════════════════════
# The Abgleich delivery carries, next to curr.csv/temp.csv, a bode/ folder of
# per-segment Gamry sweeps: the current-measurement chain swept 1 Hz–100 kHz
# at 500 mA rms with no DC bias.  Measured on both plates it is flat to 1 kHz
# and then rolls off — -11° at 4.5 kHz, -24° at 10 kHz.  4500 Hz is the top of
# the default analysis band, so this is the same order as the acquisition skew
# the pipeline works hard to remove, and unlike a skew it moves |Z| too.
#
# Set ABGLEICH_DIR to the folder holding coefficients/ and bode/, run this
# cell once, and paste the resulting path into the 'Chain-response CSV' widget.
# ═══════════════════════════════════════════════════════════════════════════════
ABGLEICH_DIR = ''      # e.g. '/Volumes/.../R2D2_green_Kashyyyk/Abgleichdaten/Kashyyyk'
 
if ABGLEICH_DIR:
    _ab = Path(ABGLEICH_DIR)
    _sweeps = gamry_dta.read_bode_folder(_ab / 'bode')
    print(f"  {len(_sweeps)} segment sweeps")
    for _row in gamry_dta.chain_summary(_sweeps):
        print(f"    {_row['freq_hz']:9.0f} Hz  |H| = {_row['mag_median']:.4f}  "
              f"arg H = {_row['phase_deg_median']:+7.2f}°  "
              f"(p5–p95 spread {_row['phase_deg_spread']:.2f}°)")
 
    _curr = _ab / 'coefficients' / 'curr.csv'
    _chk_g = gamry_dta.cross_check_abgleich(_sweeps, _curr)
    print(f"  cross-check vs curr.csv: r = {_chk_g.get('corr', float('nan')):+.4f}, "
          f"ratio spread {100*_chk_g.get('ratio_cv', float('nan')):.1f} %  "
          f"→ {'consistent' if _chk_g['ok'] else 'SUSPECT'}")
    if not _chk_g['ok']:
        print('  ' + _chk_g['reason'])
        print('  Writing the index-free plate median instead, which still '
              'removes the common roll-off.')
 
    _gain_out = Path(tempfile.gettempdir()) / f'chain_gain_{PLATE}.csv'
    gamry_dta.write_gain_csv(_sweeps, _gain_out,
                             curr_csv=_curr, shared=not _chk_g['ok'])
    print(f"  written: {_gain_out}   ← paste this into the widget")
 
    # And check the DC calibration itself while we are here.
    _rep = abgleich.verify(_ab, _curr, _ab / 'coefficients' / 'temp.csv')
    print(f"\n  Abgleich: {_rep['n_steps']} temperature steps "
          f"{_rep['temps_C']}, linearity r² ≥ {_rep['linearity_r2_min']:.6f}")
    print(f"  copper TCR {_rep['tcr_percent_per_K']['median']:.3f} %/K "
          f"({_rep['tcr_percent_per_K']['min']:.3f}–"
          f"{_rep['tcr_percent_per_K']['max']:.3f})")
    _ia = _rep.get('implied_area', {})
    print(f"  R(T)/K(T) = {_ia.get('median_cm2', float('nan')):.4f} cm², "
          f"constant to {100*_ia.get('cv', float('nan')):.2f} % across segments"
          f"{'' if not _ia.get('outliers') else '  — outliers: ' + ', '.join(_ia['outliers'])}")
else:
    print("  ABGLEICH_DIR is empty — skipping. Set it to build a chain-response "
          "file; without one the top decade of the band carries -11° of "
          "uncorrected phase.")

# COMMAND ----------

# DBTITLE 1,Datago Source: discover + read FAMOS waveforms from Delta table
# ═══════════════════════════════════════════════════════════════════════════════
# DATAGO SOURCE: FamosFile-compatible reader from Delta table
#
# Replaces the Volumes-based file reader with a datago query backend.
# The pipeline (bronze.py) calls FamosFile(path) to get waveform data.
# This cell provides DatagoFamosFile that has the SAME interface but
# reads from ps_xplatform_dev.rvadvtec_ops.datago_advtec_values_delta.
#
# PERFORMANCE NOTE:
#   Each card = 16 channels × 2.5M samples = 40M rows from Delta.
#   ~30-60s per card vs ~5s from Volumes binary. Use Volumes when available.
#   Toggle via DATA_SOURCE widget below.
# ═══════════════════════════════════════════════════════════════════════════════
import numpy as np
import time
from pathlib import Path
from dataclasses import dataclass, field
 
_VAL_TBL = 'ps_xplatform_dev.rvadvtec_ops.datago_advtec_values_delta'
_META_TBL = 'ps_xplatform_dev.rvadvtec_ops.datago_advtec_metadata'
_GP_TBL = 'ps_xplatform_dev.rvadvtec_ops.datago_advtec_generalproperties'
 
# ─── Data source selection ───
# Set to 'datago' to read from Delta table, 'volumes' for fast binary
DATA_SOURCE = 'datago'  # <-- SWITCH HERE
 
print(f"  Data source: {DATA_SOURCE}")
 
 
# ─── Discover FAMOS file_ids from datago ───
def discover_famos_file_ids(leepa: str) -> dict:
    """Find FAMOS card recordings in datago for a Leepa.
    
    Returns: {condition: [file_id_1, ..., file_id_5]} (one per card)
    """
    # Find NULL measurement_type entries (FAMOS files have no type set)
    df = spark.sql(f"""
        SELECT DISTINCT gp.file_id, gp.measurementBegin
        FROM {_META_TBL} m
        JOIN {_GP_TBL} gp ON m.measurement_id = gp.measurement_id
        WHERE m.orderId = 'RO{leepa}'
          AND gp.measurement_type IS NULL
        ORDER BY gp.measurementBegin
    """).toPandas()
    
    if df.empty:
        return {}
    
    # Exclude TOM bench file (first one, usually much earlier timestamp)
    # TOM bench has 80+ channels; FAMOS cards have exactly 16
    # Heuristic: group by minute, groups of 5 = card sets
    df['minute'] = df['measurementBegin'].dt.floor('min')
    groups = df.groupby('minute')['file_id'].apply(list).to_dict()
    
    # Filter: only groups with exactly 5 files (= 5 cards per condition)
    card_groups = {k: v for k, v in groups.items() if len(v) == 5}
    
    # Map to conditions by order (same order as Volumes naming)
    # Try to determine condition from Volumes filenames if available
    conditions_ordered = []
    famos_root = Path('/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev/Famos')
    try:
        vol_files = sorted(famos_root.glob(f'Leepa_{leepa}_Current_*_Karte_1.DAT'))
        conditions_ordered = [f.name.split('_')[3] for f in vol_files]
    except Exception:
        pass
    
    if not conditions_ordered:
        conditions_ordered = ['150A', '450A', '45A', '60A']  # default order
    
    result = {}
    for idx, (ts, file_ids) in enumerate(sorted(card_groups.items())):
        cond = conditions_ordered[idx] if idx < len(conditions_ordered) else f'Cond-{idx+1}'
        result[cond] = file_ids
    
    return result
 
 
# ─── DatagoFamosFile: drop-in replacement for FamosFile ───
class DatagoFamosFile:
    """FamosFile-compatible reader that pulls waveform data from datago.
    
    Interface matches eis_local.FamosFile:
        .path, .fs, .n_ch, .n_samples, .names,
        .segment_names, .uc_names, .temp_names,
        .channel(name) -> np.ndarray
    """
    
    def __init__(self, file_id: str, card_label: str = 'datago'):
        self.file_id = file_id
        self.path = Path(f'/datago/{card_label}.DAT')  # fake path for compatibility
        self._channels = {}  # lazy-loaded
        self._metadata_loaded = False
        self._load_metadata()
    
    def _load_metadata(self):
        """Load channel list and sample counts (lightweight query)."""
        df = spark.sql(f"""
            SELECT channel, COUNT(*) as n_pts,
                   MIN(CAST(time_value AS DOUBLE)) as t_min,
                   MAX(CAST(time_value AS DOUBLE)) as t_max
            FROM {_VAL_TBL}
            WHERE file_id = '{self.file_id}'
            GROUP BY channel
            ORDER BY channel
        """).toPandas()
        
        self.names = df['channel'].tolist()
        self.n_ch = len(self.names)
        
        # Determine sampling rate from first channel
        if not df.empty:
            row0 = df.iloc[0]
            duration = row0['t_max'] - row0['t_min']
            self.n_samples = int(row0['n_pts'])
            self.fs = round(self.n_samples / duration) if duration > 0 else 10000
        else:
            self.n_samples = 0
            self.fs = 10000
        
        # Classify channels
        self.segment_names = [n for n in self.names if n.isdigit()]
        self.uc_names = [n for n in self.names if n.startswith('UC')]
        self.temp_names = [n for n in self.names if n.startswith('Temp')]
        
        # Acquisition slot positions (channel index = multiplexing order)
        # In real FAMOS, this is the binary header order. Here we approximate
        # using the standard R2D2 card layout: UC first, then segments, then Temp
        _ordered = self.uc_names + self.segment_names + self.temp_names
        self._position_map = {name: idx for idx, name in enumerate(_ordered)}
        self.positions = _ordered
        self._metadata_loaded = True
    
    def position(self, channel_name: str) -> int:
        """Return acquisition slot index for a channel (0-based)."""
        return self._position_map.get(channel_name, 0)
    
    def channel(self, name: str) -> np.ndarray:
        """Get waveform data for a channel (loads from datago on first access)."""
        if name not in self._channels:
            self._load_channel(name)
        return self._channels[name]
    
    def _load_channel(self, name: str):
        """Query datago for a single channel's waveform."""
        df = spark.sql(f"""
            SELECT CAST(time_value AS DOUBLE) as t,
                   CAST(value AS DOUBLE) as v
            FROM {_VAL_TBL}
            WHERE file_id = '{self.file_id}'
              AND channel = '{name}'
            ORDER BY t
        """).toPandas()
        self._channels[name] = df['v'].values
    
    def load_all_channels(self):
        """Bulk-load all channels at once (more efficient than one-by-one)."""
        t0 = time.time()
        df = spark.sql(f"""
            SELECT channel,
                   CAST(time_value AS DOUBLE) as t,
                   CAST(value AS DOUBLE) as v
            FROM {_VAL_TBL}
            WHERE file_id = '{self.file_id}'
            ORDER BY channel, t
        """).toPandas()
        
        for ch_name, grp in df.groupby('channel'):
            self._channels[ch_name] = grp['v'].values
        
        dt = time.time() - t0
        print(f"    [{self.path.stem}] loaded {self.n_ch} channels, "
              f"{self.n_samples:,} pts/ch in {dt:.1f}s")
    
    def __getitem__(self, name: str) -> np.ndarray:
        """Array-style access: fam['UC2'] -> waveform."""
        return self.channel(name)
 
 
# ─── Discover for current Leepa ───
DATAGO_FAMOS_MAP = discover_famos_file_ids(LEEPA)
 
if DATAGO_FAMOS_MAP:
    print(f"\n  Datago FAMOS files for Leepa {LEEPA}:")
    for cond, fids in sorted(DATAGO_FAMOS_MAP.items()):
        print(f"    {cond}: {len(fids)} cards ({fids[0][:12]}...)")
    print(f"  Total: {sum(len(v) for v in DATAGO_FAMOS_MAP.values())} files")
else:
    print(f"  No FAMOS data in datago for Leepa {LEEPA}")
 
# ─── Monkey-patch bronze.py to use datago when selected ───
if DATA_SOURCE == 'datago' and DATAGO_FAMOS_MAP:
    import bronze as _bronze_mod
    _orig_FamosFile = _bronze_mod.FamosFile  # keep reference to original
    
    # Build a lookup: condition+card_index -> file_id
    _DATAGO_CARD_LOOKUP = {}
    for cond, fids in DATAGO_FAMOS_MAP.items():
        for card_idx, fid in enumerate(fids, start=1):
            _DATAGO_CARD_LOOKUP[(cond, card_idx)] = fid
    
    # Create a wrapper that intercepts FamosFile(path) calls
    class _FamosFileDatagoShim:
        """Intercepts FamosFile(path) and routes to datago if file_id known."""
        def __new__(cls, path, *args, **kwargs):
            path = Path(path)
            # Try to extract condition + card from filename
            # Expected: Leepa_2611976_Current_60A_Test_01_Karte_1.DAT
            name = path.name
            parts = name.split('_')
            try:
                cond = parts[3]           # e.g. '60A'
                card = int(parts[-1].replace('.DAT', ''))  # e.g. 1
                key = (cond, card)
                if key in _DATAGO_CARD_LOOKUP:
                    fid = _DATAGO_CARD_LOOKUP[key]
                    reader = DatagoFamosFile(fid, card_label=f'Karte_{card}')
                    reader.load_all_channels()  # pre-fetch everything
                    return reader
            except (IndexError, ValueError):
                pass
            # Fall back to original file-based reader
            return _orig_FamosFile(path, *args, **kwargs)
    
    _bronze_mod.FamosFile = _FamosFileDatagoShim
    print(f"\n   Bronze patched: FamosFile now reads from datago")
    print(f"    (Note: ~30-60s per card due to Delta row scan)")
else:
    print(f"\n  Using Volumes path (fast binary read)")

# COMMAND ----------

# DBTITLE 1,FAMOS v1/v2 — now handled by eis_local, not patched in here
# ═══════════════════════════════════════════════════════════════════════════════
# The v2 adapter that used to live in this cell has moved into
# eis_local.FamosFile, which now dispatches on what the file CONTAINS and
# checks that the parse means something before returning it.
#
# That check is the reason it moved. The adapter here dispatched by catching
# ValueError from the v1 reader -- and the v1 reader DOES NOT RAISE on a v2
# file. Handed one it returns, without complaint, zero channel names, a
# channel count read out of a calibration field, and a sample rate of
# 0.0625 Hz. Nothing downstream can tell. See test_famos_dialects.py, which
# writes real v2 bytes and asserts exactly that failure.
#
# The data offset is now verified against the byte count |CS declares,
# instead of counting a fixed number of commas: a 0x2C byte inside a float64
# sample is indistinguishable from a delimiter, and one comma too many landed
# inside the data and shifted every channel by two samples.
# ═══════════════════════════════════════════════════════════════════════════════
print('  FAMOS v1/v2 handled by eis_local.FamosFile (no patch needed)')

# COMMAND ----------

# DBTITLE 1,Run Dashboard: Volume cache inventory + run plan (cache vs fresh rerun)
# ═══════════════════════════════════════════════════════════════════════════════
# RUN DASHBOARD — decide, and SHOW, where each condition's numbers come from
#
# A pipeline run costs 5-10 minutes per condition, so results are copied to a
# UC Volume and reused.  That reuse used to be governed by
#
#       FORCE_RERUN = True      # set True to ignore cache and re-run
#
# a bare literal two hundred lines into the run cell.  Left at True it threw
# away a good cache on every execution; flipped to False and forgotten, it
# quietly served month-old numbers from a superseded pipeline while the
# notebook printed plots as if they were fresh.  Neither state is visible
# from the output, which is the actual problem: you cannot tell, from a
# Nyquist plot, whether the spectrum behind it was computed today.
#
# So the choice is a widget, the consequence is printed before anything runs,
# and the cache is inventoried: what is stored, how old it is, which stages
# it holds, and which pipeline version wrote it.
#
#   Run mode         cache      reuse a cached condition, run the rest
#                    rerun      ignore the cache and recompute everything
#                    cache-only reuse what is cached, SKIP the rest
#                               (no cluster time, no source files needed)
#   Write to cache   yes        a fresh run overwrites its cache entry
#                    no         a fresh run leaves the cache untouched
#                               -- a trial run that cannot clobber good data
#
# Everything below reads these two widgets.  cache_plan() is the single place
# the decision is made; the run cell calls the same function, so the plan on
# screen is the plan that executes.
# ═══════════════════════════════════════════════════════════════════════════════
import os, shutil, tempfile, json, time
import numpy as np
import pandas as pd
from pathlib import Path
 
_CACHE_VOL = Path('/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev/EIS_Results')
 
# Use a user-specific temp base to avoid permission conflicts on shared cluster
_TMP_BASE = Path(tempfile.gettempdir()) / f'eis_{os.getuid()}'
_TMP_BASE.mkdir(parents=True, exist_ok=True)
 
try:
    dbutils.widgets.dropdown('run_mode', 'cache',
                             ['cache', 'rerun', 'cache-only'],
                             'Run mode')
    dbutils.widgets.dropdown('cache_write', 'yes', ['yes', 'no'],
                             'Write results to cache')
except Exception:
    pass
 
RUN_MODE = _w('run_mode', 'cache')
CACHE_WRITE = _w('cache_write', 'yes') == 'yes'
 
# Kept as a name because older cells and notes refer to it; it is now derived
# rather than typed, and nothing reads it that does not go through
# cache_plan() first.
FORCE_RERUN = (RUN_MODE == 'rerun')
 
 
# ─── Cache identity ───
# A CACHE KEY MUST NAME EVERY SETTING THAT CHANGED THE NUMBERS.
# The key used to be the SNR widget alone, so a permissive run and a default
# run of the same condition wrote to the same directory and whichever ran
# last was served to both. Drift, THD, consensus, the frequency band and
# every alignment setting were invisible to the key while being perfectly
# visible in the results.
#
# The key is now the evaluation mode plus a digest of the settings that
# decide what survives. Caches written before this change live under the old
# path and are simply not found -- which is the correct outcome, because
# nothing recorded what produced them.
_CACHE_IDENTITY_KEYS = (
    'f_min_hz', 'f_max_hz', 'ppd', 'grid_tol',
    'min_snr_db', 'snr_floor_db', 'max_thd', 'max_drift',
    'sigma_rel_max', 'min_cycles_per_dwell', 'zmag_outlier_mad',
    'min_points_per_spectrum',
    # The build decides WHICH whole-cell sweep the aggregate is compared
    # against, and that comparison is written into the manifest. Two builds
    # are two different references, so they are two different results.
    'gamry_version',
    # Excluding a segment changes the cell aggregate, the area weighting and
    # every plate map. A run with segment 33 and a run without it are two
    # different results and must not share a cache entry.
    'exclude_segments',
    # Reconstruction changes the aggregate and the maps, so it changes the
    # result and therefore the cache entry.
    'substitute_segments', 'fill_missing_from_neighbours', 'fill_max_hops',
    'drt_nonneg',
    'ref_channel', 'align_cards', 'align_max_lag_s',
    'align_min_corr', 'align_min_prominence', 'align_f_lo_hz',
    'align_f_hi_hz', 'align_guard_s', 'align_agree_tol_s',
    'align_corroborate_min_prominence',
    'ladder_snap', 'window_sanity', 'hf_use_ensemble',
    'phasor_method', 'skew_model', 'uncertainty_model',
)
# min_ref_channels is deliberately NOT in the key: it is set from the number
# of cards the condition actually has, which is a property of the data, not
# an operator choice, and folding it in would give the same measurement two
# different cache entries for no reason.


def _run_identity(mode=None, f_min=None, f_max=None, snr=None):
    """Short digest of the settings that decide what a run keeps."""
    import hashlib as _hashlib
    mode = EVALUATION_MODE if mode is None else mode
    base = DEFAULT.replace(
        f_min_hz=F_MIN if f_min is None else f_min,
        f_max_hz=F_MAX if f_max is None else f_max,
        min_snr_db=MIN_SNR_DB if snr is None else float(snr),
        gamry_version=globals().get('GAMRY_VERSION', ''),
        exclude_segments=globals().get('EXCLUDE_SEGMENTS', frozenset()),
        substitute_segments=globals().get('SUBSTITUTE_SEGMENTS', frozenset()),
        fill_missing_from_neighbours=globals().get('FILL_GAPS', False))
    if mode and mode != 'default':
        base = base.preset(mode)
    # A set has no order, and json's default=str would spell the SAME
    # exclusion differently from one session to the next -- a cache key that
    # changes when nothing did, so every run misses and re-computes. Sets are
    # normalised to sorted lists, and Paths to strings.
    def _norm(v):
        if isinstance(v, (set, frozenset)):
            return sorted(str(x) for x in v)
        if isinstance(v, (tuple, list)):
            return [str(x) for x in v]
        return v

    payload = {k: _norm(getattr(base, k, None)) for k in _CACHE_IDENTITY_KEYS}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return _hashlib.sha256(blob.encode('utf-8')).hexdigest()[:10]


# ─── Cache paths ───
def _cache_dir(leepa, cond, snr=None, mode=None):
    base = _CACHE_VOL / leepa / cond
    if snr is None:
        return base
    mode = EVALUATION_MODE if mode is None else mode
    return base / f'mode_{mode}' / f'snr_{snr}_{_run_identity(mode=mode, snr=snr)}'
 
 
def _cache_exists(leepa, cond, snr=None):
    """True if usable cached results exist on the Volume.
 
    "Usable" means silver spectra, the minimum any display cell needs.  A
    directory holding only a manifest is a failed run, not a cache hit.
    """
    cd = _cache_dir(leepa, cond, snr)
    if not cd.exists():
        return False
    for sub in ('silver', 'csv'):
        if (cd / sub / 'spectra_clean.csv').exists():
            return True
    return False
 
 
def _save_to_cache(out_dir, leepa, cond, snr=None):
    """Copy pipeline output to Volume for persistence across sessions."""
    src = Path(out_dir)
    dst = _cache_dir(leepa, cond, snr)
    dst.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for stage in ('bronze', 'silver', 'gold', 'csv'):
        stage_src = src / stage
        if not stage_src.exists():
            continue
        stage_dst = dst / stage
        stage_dst.mkdir(parents=True, exist_ok=True)
        for f in stage_src.iterdir():
            if f.is_file():
                shutil.copy2(str(f), str(stage_dst / f.name))
                n_copied += 1
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(str(f), str(dst / f.name))
            n_copied += 1
    return n_copied
 
 
def _load_from_cache(leepa, cond, snr=None):
    """Return a PIPELINE_RESULTS-compatible dict pointing to the cached dir.
 
    The manifest is read back off the Volume rather than left empty, so a
    cached condition reports the same summary numbers as a fresh one and the
    summary cells below do not have to special-case it.
    """
    cd = _cache_dir(leepa, cond, snr)
    man = {}
    mp = cd / 'run_manifest.json'
    if mp.exists():
        try:
            man = json.loads(mp.read_text())
        except Exception:                                  # noqa: BLE001
            man = {}
    return {'manifest': man, 'cfg': None, 'out_dir': cd, 'cached': True}
 
 
# ─── The decision, in one place ───
def cache_plan(leepa, cond, snr=None, mode=None):
    """What happens to this condition: ('load'|'run'|'skip', why)."""
    mode = RUN_MODE if mode is None else mode
    have = _cache_exists(leepa, cond, snr)
    if mode == 'rerun':
        return 'run', ('recompute (cache present, will be '
                       + ('overwritten)' if CACHE_WRITE else 'left alone)')
                       if have else 'recompute (nothing cached)')
    if have:
        return 'load', 'cached'
    if mode == 'cache-only':
        return 'skip', 'nothing cached, and mode is cache-only'
    return 'run', 'nothing cached'
 
 
# ─── What is actually in a cache entry ───
def _entry_info(cd: Path) -> dict:
    """Describe one cache directory without trusting it to be complete."""
    info = {'stages': [], 'n_segments': None, 'n_freq': None,
            'n_ecm_ok': None, 'age_days': None, 'size_mb': 0.0,
            'source': None, 'elapsed_s': None, 'f_lo': None, 'f_hi': None}
    if not cd.exists():
        return info
 
    newest = 0.0
    for f in cd.rglob('*'):
        if f.is_file():
            try:
                st = f.stat()
            except OSError:
                continue
            info['size_mb'] += st.st_size / 1e6
            newest = max(newest, st.st_mtime)
    if newest:
        info['age_days'] = (time.time() - newest) / 86400.0
 
    for stage in ('bronze', 'silver', 'gold', 'csv'):
        if (cd / stage).exists() and any((cd / stage).iterdir()):
            info['stages'].append(stage)
 
    for sub in ('silver', 'csv'):
        sp = cd / sub / 'spectra_clean.csv'
        if sp.exists():
            try:
                d = pd.read_csv(sp, usecols=['segment', 'freq_hz'])
                info['n_segments'] = int(d['segment'].nunique())
                info['n_freq'] = int(d['freq_hz'].nunique())
                info['f_lo'] = float(d['freq_hz'].min())
                info['f_hi'] = float(d['freq_hz'].max())
            except Exception:                              # noqa: BLE001
                pass
            break
 
    ep = cd / 'csv' / 'ecm_parameters.csv'
    if ep.exists():
        try:
            de = pd.read_csv(ep)
            info['n_ecm_ok'] = int(de['ok'].astype(str).str.lower()
                                   .isin(['true', '1']).sum())
        except Exception:                                  # noqa: BLE001
            pass
 
    mp = cd / 'run_manifest.json'
    if mp.exists():
        try:
            m = json.loads(mp.read_text())
            info['source'] = m.get('source', 'famos')
            info['elapsed_s'] = m.get('elapsed_s')
            if info['n_ecm_ok'] is None:
                info['n_ecm_ok'] = m.get('n_ecm_ok')
            g = m.get('stages', {}).get('gold', {})
            if info['n_segments'] is None and g:
                info['n_segments'] = g.get('n_measured')
        except Exception:                                  # noqa: BLE001
            pass
    return info
 
 
def _cache_conditions(leepa) -> list:
    """Every (condition, snr_tag, dir) cached under this Leepa."""
    root = _CACHE_VOL / leepa
    out = []
    if not root.exists():
        return out
    for cdir in sorted(root.iterdir()):
        if not cdir.is_dir():
            continue
        snr_subs = [d for d in sorted(cdir.iterdir())
                    if d.is_dir() and d.name.startswith('snr_')]
        if snr_subs:
            for d in snr_subs:
                out.append((cdir.name, d.name[4:], d))
        else:
            out.append((cdir.name, None, cdir))
    return out
 
 
# ─── ONE PLACE THAT ANSWERS "WHERE ARE THIS RUN'S RESULTS?" ──────────────
# Every display cell below used to answer this for itself, by pasting a path
# shape. That is why the heat map could show 60 A while the widget said
# 450 A: nothing it read was tied to the condition that was selected, and
# nothing it read was tied to the cache layout that the run cell writes.
#
# The two functions below are the only correct answers, and each display cell
# now calls them instead of rebuilding a path.

def selected_conditions():
    """The conditions the widgets asked for -- NOT whatever is on the Volume.

    A display cell that iterates the cache directory shows every condition
    ever run for this plate, in whatever order the filesystem returns them.
    With one figure per condition the LAST one drawn is the one on screen,
    which is how selecting 450 A produces a 60 A heat map: 60 A simply sorts
    last among 150A, 450A, 45A, 60A.
    """
    try:
        return list(_conditions_to_run)
    except NameError:
        pass
    return parse_conditions(_w('conditions', 'ALL'), CONDITIONS)


def result_dir(leepa, cond, snr=None, mode=None, prefer_live=True):
    """(directory, provenance) for one condition, newest evidence first.

    Order, and why:
      1. What THIS kernel just computed. A fresh run's output is the most
         recent thing there is, and it may not have reached the Volume at all
         if cache writing is off. The old cells consulted it only when the
         run-mode widget said 'rerun', so a normal run displayed the previous
         run's cache.
      2. The cache entry for the CURRENT identity -- mode and settings digest
         included, via _cache_dir, so this cannot drift from what the run
         cell writes.
      3. Older layouts (snr_<v>/ then the unversioned tree), clearly marked.
         They are readable but they are not evidence about the current
         settings, and the provenance string says so in words the display
         cells print.
    """
    snr = MIN_SNR_DB if snr is None else snr
    mode = EVALUATION_MODE if mode is None else mode

    def _usable(d):
        d = Path(d)
        return d.is_dir() and any((d / sub).is_dir()
                                  for sub in ('gold', 'silver', 'csv'))

    if prefer_live:
        try:
            pr = PIPELINE_RESULTS.get(cond)
        except NameError:
            pr = None
        if pr and not pr.get('cached') and _usable(pr['out_dir']):
            return Path(pr['out_dir']), 'this session\'s run'

    d = _cache_dir(leepa, cond, snr, mode)
    if _usable(d):
        return d, f'cache: mode_{mode}/{Path(d).name}'

    base = _CACHE_VOL / str(leepa) / str(cond)
    for name in _legacy_snr_names(snr):
        if _usable(base / name):
            return base / name, (f'LEGACY cache {name}/ -- written before the '
                                 f'cache key included the evaluation mode and '
                                 f'the settings digest, so it may not match '
                                 f'the settings shown above')
    if _usable(base):
        return base, ('LEGACY unversioned cache -- belongs to whichever run '
                      'wrote last, at unknown settings')
    try:
        tmp = _TMP_BASE / str(leepa) / str(cond)
        if _usable(tmp):
            return tmp, 'local /tmp output from an earlier run in this session'
    except NameError:
        pass
    return None, 'not found'


def _legacy_snr_names(snr):
    """The snr_<value> spellings the old cache layout used."""
    out, text = [], str(snr).strip()
    if text:
        out.append(f'snr_{text}')
    try:
        f = float(text)
        for spelling in (repr(f), f'{f:g}'):
            if f'snr_{spelling}' not in out:
                out.append(f'snr_{spelling}')
    except (TypeError, ValueError):
        pass
    return out


def describe_source(cond, d, prov):
    """One line naming what is about to be plotted, and how old it is."""
    if d is None:
        return (f'  {cond}: NO RESULTS for the current selection '
                f'(mode {EVALUATION_MODE}, SNR {MIN_SNR_DB:g}). '
                f'Run the pipeline cell for this condition.')
    newest = max((f.stat().st_mtime for f in Path(d).rglob('*.csv')),
                 default=0.0)
    age_h = (time.time() - newest) / 3600.0 if newest else float('nan')
    age = ('unknown age' if not newest else
           f'{age_h*60:.0f} min old' if age_h < 1 else
           f'{age_h:.1f} h old' if age_h < 48 else f'{age_h/24:.0f} d old')
    flag = '   <-- NOT the current settings' if prov.startswith('LEGACY') else ''
    return f'  {cond}: {prov}, {age}{flag}\n    {d}'


# ─── Which conditions this execution covers ───
# A CSV measurement is one file, so it is one "condition" -- named after the
# file rather than after a current setpoint, because the file is what
# identifies it.
if SOURCE_FORMAT == 'csv':
    _conditions_to_run = [Path(CSV_PATH).stem or 'csv']
else:
    # The multi-select, already parsed: e.g. ['45A', '450A'], in current order.
    _conditions_to_run = list(SELECTED_CONDITIONS)
 
RUN_PLAN = {c: cache_plan(LEEPA, c, MIN_SNR_DB) for c in _conditions_to_run}
 
 
# ─── Render ───
def _esc(x):
    return (str(x).replace('&', '&amp;').replace('<', '&lt;')
            .replace('>', '&gt;'))
 
 
def _fmt(v, spec='', dash='—'):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return dash
    return format(v, spec) if spec else str(v)
 
 
def _age_cell(days):
    """Age, coloured.  Stale cache is the failure mode this panel exists for."""
    if days is None:
        return f'<td style="{_TD}color:#999">—</td>'
    if days < 1:
        txt, col = f'{days*24:.1f} h', '#1a7f37'
    elif days < 30:
        txt, col = f'{days:.0f} d', '#1a7f37' if days < 7 else '#9a6700'
    else:
        txt, col = f'{days:.0f} d', '#cf222e'
    return f'<td style="{_TD}color:{col};font-weight:600">{txt}</td>'
 
 
_TD = ('padding:6px 10px;border-bottom:1px solid #eaeef2;'
       'font-variant-numeric:tabular-nums;')
_TH = ('padding:7px 10px;text-align:left;font-size:11px;'
       'text-transform:uppercase;letter-spacing:.04em;color:#57606a;'
       'border-bottom:2px solid #d0d7de;font-weight:600;')
_BADGE = ('display:inline-block;padding:2px 9px;border-radius:10px;'
          'font-size:11px;font-weight:700;letter-spacing:.02em;')
_ACTION_STYLE = {
    'load': (_BADGE + 'background:#dafbe1;color:#0a5c26', 'LOAD FROM CACHE'),
    'run':  (_BADGE + 'background:#fff1e5;color:#9a4f00', 'RUN PIPELINE'),
    'skip': (_BADGE + 'background:#f0f1f3;color:#57606a', 'SKIP'),
}
 
_n_load = sum(1 for a, _ in RUN_PLAN.values() if a == 'load')
_n_run = sum(1 for a, _ in RUN_PLAN.values() if a == 'run')
_n_skip = sum(1 for a, _ in RUN_PLAN.values() if a == 'skip')
 
_mode_note = {
    'cache': 'cached conditions are reused; the rest are computed',
    'rerun': 'the cache is ignored and every condition is recomputed',
    'cache-only': 'only cached conditions are shown; nothing is computed',
}.get(RUN_MODE, '')
 
_h = [f'''
<div style="font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;
            max-width:1180px;color:#1f2328">
  <div style="display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;
              border-bottom:2px solid #1f2328;padding-bottom:8px;margin-bottom:4px">
    <span style="font-size:19px;font-weight:700">Run plan</span>
    <span style="font-size:13px;color:#57606a">
      Leepa <b>{_esc(LEEPA)}</b> &middot; {_esc(COND_FILTER)} &middot;
      {_esc(SOURCE_FORMAT)} &middot; {F_MIN:g}&ndash;{F_MAX:g} Hz &middot;
      SNR &ge; {MIN_SNR_DB:g} dB &middot; stop after {_esc(STOP_AFTER)}
    </span>
  </div>
  <div style="margin:10px 0 14px;font-size:13px">
    <span style="{_BADGE}background:#ddf4ff;color:#0550ae">MODE: {_esc(RUN_MODE.upper())}</span>
    <span style="color:#57606a;margin-left:8px">{_esc(_mode_note)}</span>
    <span style="margin-left:14px;color:#57606a">write to cache:
      <b style="color:{'#1a7f37' if CACHE_WRITE else '#cf222e'}">
      {'yes' if CACHE_WRITE else 'no'}</b></span>
  </div>
  <div style="font-size:13px;margin-bottom:10px">
    <b>{_n_load}</b> to load &middot; <b>{_n_run}</b> to run
    {f'&middot; <b>{_n_skip}</b> skipped' if _n_skip else ''}
    &middot; est. cluster time
    <b>{'~0 min' if not _n_run else f'~{5*_n_run}&ndash;{10*_n_run} min'}</b>
  </div>
  <table style="border-collapse:collapse;width:100%;font-size:13px">
    <tr>
      <th style="{_TH}">Condition</th><th style="{_TH}">Action</th>
      <th style="{_TH}">Why</th><th style="{_TH}">Age</th>
      <th style="{_TH}">Seg</th><th style="{_TH}">Freq</th>
      <th style="{_TH}">Band [Hz]</th><th style="{_TH}">ECM ok</th>
      <th style="{_TH}">Stages</th><th style="{_TH}">MB</th>
    </tr>''']
 
for cond in _conditions_to_run:
    action, why = RUN_PLAN[cond]
    style, label = _ACTION_STYLE[action]
    cd = _cache_dir(LEEPA, cond, MIN_SNR_DB)
    inf = _entry_info(cd) if cd.exists() else _entry_info(Path('/nonexistent'))
    band = ('—' if inf['f_lo'] is None
            else f"{inf['f_lo']:.2f}&ndash;{inf['f_hi']:.0f}")
    _h.append(f'''
    <tr>
      <td style="{_TD}font-weight:600">{_esc(cond)}</td>
      <td style="{_TD}"><span style="{style}">{label}</span></td>
      <td style="{_TD}color:#57606a">{_esc(why)}</td>
      {_age_cell(inf['age_days'])}
      <td style="{_TD}">{_fmt(inf['n_segments'])}</td>
      <td style="{_TD}">{_fmt(inf['n_freq'])}</td>
      <td style="{_TD}">{band}</td>
      <td style="{_TD}">{_fmt(inf['n_ecm_ok'])}</td>
      <td style="{_TD}color:#57606a">{'&middot;'.join(inf['stages']) or '—'}</td>
      <td style="{_TD}">{inf['size_mb']:.1f}</td>
    </tr>''')
 
_h.append('</table>')
 
# ─── Everything else cached under this Leepa (other SNR gates, other bands) ───
_others = [(c, s, d) for c, s, d in _cache_conditions(LEEPA)
           if not (c in _conditions_to_run
                   and s == (str(MIN_SNR_DB) if MIN_SNR_DB is not None else None))]
if _others:
    _h.append(f'''
  <div style="margin-top:22px;font-size:14px;font-weight:700">
    Also cached under Leepa {_esc(LEEPA)}
    <span style="font-weight:400;color:#57606a;font-size:12px">
      &mdash; other SNR gates and conditions this run does not touch.
      The SNR gate is part of the cache key: change it and you get a
      different entry, not a stale one.</span>
  </div>
  <table style="border-collapse:collapse;width:100%;font-size:13px;margin-top:6px">
    <tr><th style="{_TH}">Condition</th><th style="{_TH}">SNR gate</th>
        <th style="{_TH}">Age</th><th style="{_TH}">Seg</th>
        <th style="{_TH}">Stages</th><th style="{_TH}">MB</th></tr>''')
    for c, s, d in _others:
        inf = _entry_info(d)
        _h.append(f'''
    <tr><td style="{_TD}">{_esc(c)}</td>
        <td style="{_TD}">{_esc(s) if s is not None else '—'}</td>
        {_age_cell(inf['age_days'])}
        <td style="{_TD}">{_fmt(inf['n_segments'])}</td>
        <td style="{_TD}color:#57606a">{'&middot;'.join(inf['stages']) or '—'}</td>
        <td style="{_TD}">{inf['size_mb']:.1f}</td></tr>''')
    _h.append('</table>')
 
# ─── Other Leepa IDs on the Volume ───
try:
    _other_leepa = sorted(d.name for d in _CACHE_VOL.iterdir()
                          if d.is_dir() and d.name != LEEPA)
except Exception:                                          # noqa: BLE001
    _other_leepa = []
if _other_leepa:
    _h.append(f'''
  <div style="margin-top:18px;font-size:12px;color:#57606a">
    Other Leepa IDs cached on the Volume:
    {_esc(', '.join(_other_leepa))}
    &mdash; switch the Order ID widget to load one without re-running.
  </div>''')
 
_h.append(f'''
  <div style="margin-top:16px;font-size:12px;color:#57606a">
    Cache root: <code>{_esc(_CACHE_VOL)}</code><br>
    Scratch:&nbsp;&nbsp;&nbsp;&nbsp; <code>{_esc(_TMP_BASE)}</code>
    (per-user, cleared when the cluster restarts)
  </div>
</div>''')
 
try:
    displayHTML('\n'.join(_h))
except Exception:                                          # noqa: BLE001
    # No displayHTML (plain python, or a job run): the plan still has to be
    # legible, because it is what decides whether anything runs at all.
    print(f"\n  RUN PLAN — Leepa {LEEPA}, mode={RUN_MODE}, "
          f"write_cache={CACHE_WRITE}")
    for cond in _conditions_to_run:
        action, why = RUN_PLAN[cond]
        print(f"    {cond:>12s}  {action.upper():<5s}  {why}")
 
# Count the entries that actually exist, not the conditions being run: a
# condition with nothing cached overwrites nothing.
_n_clobber = sum(1 for c in _conditions_to_run
                 if RUN_PLAN[c][0] == 'run' and _cache_exists(LEEPA, c, MIN_SNR_DB))
if _n_clobber and RUN_MODE == 'rerun' and CACHE_WRITE:
    print(f"  ⚠ rerun + write: {_n_clobber} existing cache entr"
          f"{'y' if _n_clobber == 1 else 'ies'} for Leepa {LEEPA} will be "
          f"overwritten.  Set 'Write results to cache' to no for a trial run.")
if RUN_MODE == 'cache-only' and _n_skip:
    print(f"  ⚠ cache-only: {_n_skip} condition(s) have nothing cached and "
          f"will be missing from every plot below.")
 

# COMMAND ----------

# DBTITLE 1,Build Config + Run Pipeline (with persistent Volume cache)
 
# ═══════════════════════════════════════════════════════════════════════════════
# RUN PIPELINE PER CONDITION (fixes the cross-condition deduplication bug)
#
# When condition=ALL, bronze.py merges all files and deduplicates segments
# ACROSS conditions (keeping best-SNR only). This is wrong: each current
# setpoint (45A, 60A, 150A, 450A) is a SEPARATE EIS experiment.
#
# Fix: iterate conditions individually, producing separate results per condition.
# ═══════════════════════════════════════════════════════════════════════════════
# Re-imported rather than inherited from the dashboard cell, so this cell
# still runs on its own after a "Clear state".
import os, shutil, tempfile, gc
 
 
def spectra_csv(out_dir):
    """Where this run's per-segment spectra ended up.
 
    The FAMOS path writes silver/spectra_clean.csv; the CSV path writes
    csv/spectra_clean.csv.  Same columns, same units (mΩ·cm²), different
    stage name — so every display cell asks here rather than hard-coding a
    stage that only exists on one of the two routes.
    """
    out_dir = Path(out_dir)
    for sub in ('silver', 'csv'):
        p = out_dir / sub / 'spectra_clean.csv'
        if p.exists():
            return p
    return None
 
 
def maps_dir(out_dir):
    """Where the plate maps and summary tables ended up."""
    out_dir = Path(out_dir)
    return out_dir / 'gold' if (out_dir / 'gold').exists() else out_dir
 
 
_DAT_DIR = FAMOS_ROOT
_CURR_CAL = Path('/Workspace/Users/uum5fe@bosch.com/curr.csv')
_TEMP_CAL = Path('/Workspace/Users/uum5fe@bosch.com/temp.csv')
 
# _CACHE_VOL, _TMP_BASE, _cache_dir, _cache_exists, _save_to_cache,
# _load_from_cache, cache_plan, RUN_MODE, CACHE_WRITE, RUN_PLAN and
# _conditions_to_run all come from the Run Dashboard cell above.  They live
# there so that the cache policy is decided and DISPLAYED in one place
# instead of being a literal buried in this cell.
 
print(f"  DAT dir:    {_DAT_DIR}")
print(f"  Curr cal:   {_CURR_CAL}")
print(f"  Gamry ref:  {GAMRY_DIR or '(none - whole-cell check skipped)'}")
print(f"  Gamry build:{' ' + GAMRY_VERSION if GAMRY_VERSION else ''}"
      + ('' if GAMRY_VERSION else ' (UNKNOWN — if that folder holds more than '
                                 'one campaign, the sweeps may be another '
                                 "cell's)"))
print(f"  Conditions: {_conditions_to_run}")
print(f"  Band:       {F_MIN} – {F_MAX} Hz")
print(f"  Stop after: {STOP_AFTER}")
# The mode decides the gates AND the cache entry, so it is printed with the
# rest of the run identity rather than left to be inferred from the results.
print(f"  Rebuilt:    "
      + (", ".join(sorted(SUBSTITUTE_SEGMENTS, key=lambda x: int(x)
                          if x.isdigit() else 0))
         + "   <-- own measurement discarded, value from neighbours"
         if SUBSTITUTE_SEGMENTS else "(none)")
      + ("   + every unmeasured segment" if FILL_GAPS else ""))
print(f"  Excluded:   "
      + (", ".join(sorted(EXCLUDE_SEGMENTS, key=lambda x: int(x)
                          if x.isdigit() else 0))
         + "   <-- left out of every stage" if EXCLUDE_SEGMENTS else "(none)"))
print(f"  Eval mode:  {EVALUATION_MODE}"
      + ("" if EVALUATION_MODE == 'default'
         else "   <-- NOT the pipeline defaults; exploratory"))
print(f"  Run id:     {_run_identity()}   (cache key: settings digest)\n")
 
# ─── Run pipeline once per condition (or load from cache) ───
PIPELINE_RESULTS = {}  # {condition_str: manifest_dict}
 
for cond in _conditions_to_run:
    # ── Same decision the dashboard displayed, taken from the same function
    #    so the two cannot drift apart ──
    _action, _why = cache_plan(LEEPA, cond, MIN_SNR_DB)
 
    if _action == 'skip':
        print(f"  – {cond}: SKIPPED ({_why})")
        PIPELINE_RESULTS[cond] = None
        continue
 
    if _action == 'load':
        PIPELINE_RESULTS[cond] = _load_from_cache(LEEPA, cond, MIN_SNR_DB)
        _cd = _cache_dir(LEEPA, cond, MIN_SNR_DB)
        _sp = spectra_csv(_cd)
        _n = 0
        if _sp and _sp.exists():
            import pandas as _pd
            _n = _pd.read_csv(_sp)['segment'].nunique()
        _inf = _entry_info(_cd)
        _age = ('unknown age' if _inf['age_days'] is None
                else f"{_inf['age_days']*24:.1f} h old" if _inf['age_days'] < 1
                else f"{_inf['age_days']:.0f} d old")
        # Say how old it is every time.  A cached number that is not marked
        # as cached is the whole reason this path needed a dashboard.
        print(f"   {cond}: FROM CACHE ({_n} segments, {_age}) — "
              f"pipeline not run")
        print(f"    {_cd}")
        continue
 
    # ── Count the cards for THIS condition, before the run ───────────────
    # The consensus requirement is relaxed below only for a genuine
    # single-card condition, so the count has to come from the files that
    # were selected -- not from a discovery that failed, which would make
    # "nothing matched" indistinguishable from "one card".
    _selected_files = []
    if SOURCE_FORMAT == 'famos':
        try:
            _selected_files = bronze.discover_files(
                DEFAULT.replace(dat_dir=_DAT_DIR, leepa=LEEPA, condition=cond))
        except SystemExit as _de:
            print(f"   {cond}: file discovery found nothing ({_de})")
            _selected_files = []
        print(f"   {cond}: {len(_selected_files)} card file(s) selected")

    _out_dir = _TMP_BASE / LEEPA / cond
    # Force-clean if stale directory with wrong permissions exists
    if _out_dir.exists():
        try:
            # Test write access
            (_out_dir / '.writetest').touch()
            (_out_dir / '.writetest').unlink()
        except PermissionError:
            shutil.rmtree(_out_dir, ignore_errors=True)
    _out_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = DEFAULT.replace(
        plate='gen1',  # hardcode string; PLATE var can be corrupted by module reload
        source_format=SOURCE_FORMAT,
        csv_path=Path(CSV_PATH) if CSV_PATH else None,
        csv_dialect=CSV_DIALECT,
        csv_tones=CSV_TONES,
        gain_file=Path(GAIN_FILE) if GAIN_FILE else None,
        gamry_dir=Path(GAMRY_DIR) if GAMRY_DIR else None,
        gamry_version=GAMRY_VERSION,
        bench_log=Path(BENCH_LOG) if BENCH_LOG else None,
        dat_dir=_DAT_DIR,
        out_dir=_out_dir,
        curr_cal=_CURR_CAL,
        temp_cal=_TEMP_CAL,
        leepa=LEEPA,
        condition=cond,
        f_min_hz=F_MIN,
        f_max_hz=F_MAX,
        write_png=True,
        write_html=True,
        infer_missing_segments=False,  # Don't infer unmeasured segments (36,66,70,71)
        min_snr_db=MIN_SNR_DB,   
        # A five-card FAMOS campaign must keep REAL consensus: a step seen by
        # one card only is carried by grid membership, not by lowering the
        # vote. min_ref_channels=1 is restored below, explicitly and out
        # loud, when the run genuinely has one card.
        min_ref_channels=2,
        exclude_segments=EXCLUDE_SEGMENTS,
        substitute_segments=SUBSTITUTE_SEGMENTS,
        fill_missing_from_neighbours=FILL_GAPS,
    )

    # ── Gate preset: named, not smuggled ──────────────────────────────────
    # Everything that relaxes a quality gate now comes from Config.preset(),
    # so the mode is a single word that travels into the cache path, the
    # manifest and the printout below. Nothing here silently edits a
    # threshold the pipeline documents elsewhere.
    if EVALUATION_MODE != 'default':
        cfg = cfg.preset(EVALUATION_MODE)
        print(f"\n  EVALUATION MODE: {EVALUATION_MODE.upper()} — gates are "
              f"NOT the pipeline defaults. These results are exploratory and "
              f"are cached separately from the default-mode results.")

    # ── Single-card exception, stated rather than assumed ─────────────────
    # Counted from the files actually selected, never inferred from a failed
    # discovery: "no files matched" must not read as "one card".
    _n_cards = len(_selected_files) if _selected_files else 0
    if SOURCE_FORMAT == 'famos' and _n_cards == 1:
        cfg = cfg.replace(min_ref_channels=1)
        print(f"  Consensus reduced to 1 reference channel: this condition "
              f"has exactly one card ({Path(_selected_files[0]).name}). "
              f"Cross-card agreement is unavailable, so every step rests on "
              f"grid membership alone.")
    elif SOURCE_FORMAT == 'famos' and _n_cards == 0:
        print("  WARNING: no FAMOS files were counted for this condition; "
              "consensus settings left at the default rather than relaxed.")
 
    print(f"\n{'═'*75}")
    print(f"  {'FILE' if SOURCE_FORMAT == 'csv' else 'CONDITION'}: {cond}"
          f"   [{PLATE}]")
    print(f"{'═'*75}")
    
    try:
        manifest = pipeline_main.run_pipeline(cfg, stop_after=STOP_AFTER)
        PIPELINE_RESULTS[cond] = {
            'manifest': manifest,
            'cfg': cfg,
            'out_dir': _out_dir,
        }
        # ── Persist to Volume ──
        if CACHE_WRITE:
            try:
                _nc = _save_to_cache(_out_dir, LEEPA, cond, MIN_SNR_DB)
                print(f"   Saved {_nc} files to Volume cache: "
                      f"{_cache_dir(LEEPA, cond, MIN_SNR_DB)}")
            except Exception as _ce:                       # noqa: BLE001
                print(f"   Cache save failed (results still in /tmp): {_ce}")
        else:
            print(f"   Cache write disabled — results stay in {_out_dir} "
                  f"and are lost when the cluster restarts")
        # The FAMOS manifest reports per stage; the CSV manifest is flat.
        gs = manifest.get('stages', {}).get('gold', {})
        if gs:
            n_meas = gs.get('n_measured', '?')
            n_total = gs.get('n_total', '?')
            r_ohmic = gs.get('R_ohmic', {})
            if r_ohmic:
                print(f"  {cond}: {n_meas}/{n_total} segments | "
                      f"Rs = {1000*r_ohmic['mean']:.1f} ± {1000*r_ohmic['sd']:.1f} mΩ·cm²")
            else:
                print(f"  {cond}: {n_meas}/{n_total} segments")
        else:
            sch = manifest.get('schedule', {})
            print(f"   {cond}: {manifest.get('n_segments', '?')} segments, "
                  f"{manifest.get('n_ecm_ok', '?')} ECM fits, "
                  f"KK {manifest.get('kk_pass', '?')}/{manifest.get('kk_total', '?')}"
                  f" | excitation: {sch.get('mode', '?')}, "
                  f"{len(sch.get('tones', []))} tones")
    except Exception as e:
        print(f"   {cond}: FAILED — {e}")
        PIPELINE_RESULTS[cond] = None
    # Release memory between conditions to prevent driver OOM
    gc.collect()
    spark.catalog.clearCache()
 
_n_ok = len([v for v in PIPELINE_RESULTS.values() if v])
_n_cached = len([v for v in PIPELINE_RESULTS.values()
                 if v and v.get('cached')])
print(f"\n{'═'*75}")
print(f"  PIPELINE COMPLETE — {_n_ok} condition(s) available "
      f"({_n_cached} from cache, {_n_ok - _n_cached} freshly computed)")
if _n_cached and RUN_MODE != 'rerun':
    print(f"  Set the Run mode widget to 'rerun' to recompute the cached ones.")
print(f"{'═'*75}")
 

# COMMAND ----------

# DBTITLE 1,Display Nyquist (3-panel: Nyquist + Bode |Z| + Phase)
# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE NYQUIST + BODE PER CONDITION
# Same style as EIS Analysis Gold Table (Plotly 3-panel)
# ═══════════════════════════════════════════════════════════════════════════════
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, Image as IPImage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 
SEG_AREA_CM2 = 4.235  # fallback, used for ASR conversion
 
# ── The display rule, in one place and stated in the figure ──────────────
# OFF. Every finite point silver accepted is drawn as an ordinary point of
# its segment's colour -- no grey crosses, nothing set aside.
#
# The rule it replaces (1 <= Z' <= 500 mOhm*cm2, f <= 2 kHz, no inductive
# below 1 kHz) was a viewing convenience with opinions in it: it deletes a
# genuinely bad segment rather than showing it as bad, hides the band
# hf_schedule exists to recover, and removes low-frequency inductive
# behaviour, which on a fuel cell is a finding. Silver's nine gates have
# already decided what is a measurement; a second, softer opinion on top of
# them belongs to whoever is looking, not to the plot.
#
# Set it True to get the rule back, with the excluded points drawn as grey
# crosses and counted in the caption rather than dropped.
DISPLAY_FILTER = False
DISPLAY_RULE_TEXT = ("display rule: 1 ≤ Z′ ≤ 500 mΩ·cm², f ≤ 2 kHz, "
                     "no inductive below 1 kHz")

for cond, pr in PIPELINE_RESULTS.items():
    if pr is None:
        continue
    out_dir = pr['out_dir']
    _n_hidden_total = 0
    _n_finite_total = 0
    _n_rebuilt = 0

    # Load silver spectra for this condition
    _spectra_path = spectra_csv(out_dir)
    if not _spectra_path.exists():
        # Fallback: show the gold nyquist.png if available
        _nyq = maps_dir(out_dir) / 'nyquist.png'
        if _nyq.exists():
            print(f"\n  {cond}: showing gold/nyquist.png")
            display(IPImage(filename=str(_nyq)))
        else:
            print(f"  {cond}: no spectra found")
        continue
    
    # Read silver spectra CSV
    df = pd.read_csv(_spectra_path)
    segments = sorted(df['segment'].unique())
    n_seg = len(segments)
    
    
    # Build interactive Plotly 3-panel (Nyquist + Bode |Z| + Phase)
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['Nyquist', '|Z|(f) Bode', 'Phase(f)'],
        horizontal_spacing=0.06)
    
    # Color palette (hue-spaced)
    colors = [f'hsl({int(i*360/n_seg)}, 70%, 50%)' for i in range(n_seg)]
    
    for i, seg in enumerate(segments):
        sd = df[df['segment'] == seg].sort_values('freq_hz')
        f = sd['freq_hz'].values
        zr = sd['z_re_mohm_cm2'].values  # already in mΩ·cm² from silver
        zi = sd['z_im_mohm_cm2'].values
        
        # ── A DISPLAY FILTER IS NOT AN ACCEPTANCE GATE ──────────────────
        # Everything in spectra_clean.csv has already passed silver's nine
        # gates; whatever this cell removes on top is a VIEWING choice, and
        # the four rules below are opinionated ones. "Z' between 1 and
        # 500 mΩ·cm²" deletes a genuinely bad segment rather than showing it
        # as bad. "f <= 2 kHz" hides exactly the band this campaign spent
        # hf_schedule recovering. "No inductive below 1 kHz" removes real
        # low-frequency inductive behaviour, which on a fuel cell is a
        # finding, not an artefact.
        #
        # So the points are not dropped any more, they are SPLIT: what the
        # filter keeps is drawn as a line, what it hides is drawn as grey
        # crosses in the same figure, and the count of hidden points is
        # printed per condition. A plot that looks clean because the ugly
        # points were deleted is the failure mode this exists to prevent.
        finite = np.isfinite(zr) & np.isfinite(zi) & np.isfinite(f)

        shown = finite.copy()
        if DISPLAY_FILTER:
            shown &= (zr > 0)
            shown &= (zr >= 1.0) & (zr <= 500.0)
            shown &= (f <= 2000.0)
            cap = f <= 1000.0
            shown &= ~(cap & (-zi < -5.0))
        hidden = finite & ~shown
        _n_hidden_total += int(hidden.sum())
        _n_finite_total += int(finite.sum())

        if shown.sum() < 3 and hidden.sum() < 3:
            continue

        clr = colors[i]
        seg_name = f'Seg {seg}'

        # Nyquist — show frequency on hover
        fig.add_trace(go.Scatter(
            x=zr[shown], y=-zi[shown],
            mode='markers+lines', marker=dict(size=4, color=clr),
            line=dict(width=1, color=clr),
            name=seg_name, legendgroup=seg_name, showlegend=True,
            hovertemplate=(f'<b>Seg {seg}</b><br>'
                           'f = %{customdata:.2f} Hz<br>'
                           "Z' = %{x:.1f} mΩ·cm²<br>"
                           "-Z'' = %{y:.1f} mΩ·cm²<extra></extra>"),
            customdata=f[shown],
        ), row=1, col=1)

        # the points the display rule removed: visible, greyed, never silent
        if hidden.any():
            fig.add_trace(go.Scatter(
                x=zr[hidden], y=-zi[hidden],
                mode='markers',
                marker=dict(size=7, symbol='x', color='rgba(120,120,120,0.55)'),
                name=f'{seg_name} hidden by display filter',
                legendgroup=seg_name, showlegend=False,
                hovertemplate=(f'<b>Seg {seg}</b> (hidden by display rule)<br>'
                               'f = %{customdata:.2f} Hz<br>'
                               "Z' = %{x:.1f} mΩ·cm²<br>"
                               "-Z'' = %{y:.1f} mΩ·cm²<extra></extra>"),
                customdata=f[hidden],
            ), row=1, col=1)

        f, zr, zi = f[shown], zr[shown], zi[shown]
        if len(f) < 3:
            continue
        
        # Bode |Z|
        fig.add_trace(go.Scatter(
            x=f, y=np.abs(zr + 1j * zi),
            mode='lines', line=dict(width=1, color=clr),
            name=seg_name, legendgroup=seg_name, showlegend=False,
            hovertemplate=(f'<b>Seg {seg}</b><br>'
                           'f = %{x:.2f} Hz<br>'
                           '|Z| = %{y:.1f} mΩ·cm²<extra></extra>'),
        ), row=1, col=2)
        
        # Phase
        fig.add_trace(go.Scatter(
            x=f, y=np.degrees(np.angle(zr + 1j * zi)),
            mode='lines', line=dict(width=1, color=clr),
            name=seg_name, legendgroup=seg_name, showlegend=False,
            hovertemplate=(f'<b>Seg {seg}</b><br>'
                           'f = %{x:.2f} Hz<br>'
                           'Phase = %{y:.1f}°<extra></extra>'),
        ), row=1, col=3)
    
    # Segments rebuilt from their neighbours, dotted so they cannot be
    # mistaken for measurements. They live in their own file because
    # spectra_clean.csv is measurements only -- and leaving them off this plot
    # meant the run said "67 segments" while the maps and the aggregate beside
    # it were built from 72.
    _rec_path = Path(out_dir) / 'silver' / 'spectra_reconstructed.csv'
    if _rec_path.exists():
        _rec = pd.read_csv(_rec_path)
        _n_rebuilt = _rec['segment'].nunique()
        for _j, _seg in enumerate(sorted(_rec['segment'].unique(),
                                         key=lambda x: int(x))):
            _rd = _rec[_rec['segment'] == _seg].sort_values('freq_hz')
            _zr, _zi = _rd['z_re_mohm_cm2'].values, _rd['z_im_mohm_cm2'].values
            _f = _rd['freq_hz'].values
            _ok = np.isfinite(_zr) & np.isfinite(_zi)
            if _ok.sum() < 3:
                continue
            _don = str(_rd['donors'].iloc[0])
            fig.add_trace(go.Scatter(
                x=_zr[_ok], y=-_zi[_ok], mode='lines',
                line=dict(width=1.6, color='#444', dash='dot'),
                name=f'Seg {_seg} (rebuilt)', legendgroup='rebuilt',
                showlegend=(_j == 0), customdata=_f[_ok],
                hovertemplate=(f'<b>Seg {_seg}</b> \u2014 REBUILT from {_don}<br>'
                               'f = %{customdata:.2f} Hz<br>'
                               "Z' = %{x:.1f}<br>-Z'' = %{y:.1f}<extra></extra>"),
            ), row=1, col=1)
            _Z = _zr[_ok] + 1j * _zi[_ok]
            fig.add_trace(go.Scatter(x=_f[_ok], y=np.abs(_Z), mode='lines',
                line=dict(width=1.6, color='#444', dash='dot'),
                legendgroup='rebuilt', showlegend=False), row=1, col=2)
            fig.add_trace(go.Scatter(x=_f[_ok], y=np.degrees(np.angle(_Z)),
                mode='lines', line=dict(width=1.6, color='#444', dash='dot'),
                legendgroup='rebuilt', showlegend=False), row=1, col=3)

    fig.update_xaxes(title_text="Z' [mΩ·cm²]", row=1, col=1)
    fig.update_yaxes(title_text="-Z'' [mΩ·cm²]", row=1, col=1)
    fig.update_xaxes(title_text="f [Hz]", type="log", row=1, col=2)
    fig.update_yaxes(title_text="|Z| [mΩ·cm²]", type="log", row=1, col=2)
    fig.update_xaxes(title_text="f [Hz]", type="log", row=1, col=3)
    fig.update_yaxes(title_text="Phase [°]", row=1, col=3)
    
    _hidden_note = (
        f'<br><span style="font-size:11px;color:#888">'
        f'{DISPLAY_RULE_TEXT} — {_n_hidden_total} of {_n_finite_total} '
        f'silver-accepted points shown as grey × (hidden by the rule, not '
        f'rejected by the pipeline)</span>'
        if DISPLAY_FILTER and _n_hidden_total else
        '<br><span style="font-size:11px;color:#888">every finite '
        'silver-accepted point shown</span>')
    fig.update_layout(
        title=f'<b>Local EIS — Leepa {LEEPA}, {cond} ({n_seg} measured'
              + (f' + {_n_rebuilt} rebuilt' if _n_rebuilt else '')
              + ' segments)</b>'
              + _hidden_note,
        height=550, width=1500,
        paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(t=80, b=50, r=250),
        hovermode='closest',
        legend=dict(
            font=dict(size=9),
            itemclick='toggle', itemdoubleclick='toggleothers',
            y=0.5, yanchor='middle',
        ),
    )
    show_fig(fig)          # one plot per figure, stacked
    print(f"  {cond}: {n_seg} segments plotted"
          + (f" + {_n_rebuilt} rebuilt from neighbours" if _n_rebuilt else "")
          + (f", all {_n_finite_total} silver-accepted points shown"
             if not _n_hidden_total else
             f", {_n_finite_total - _n_hidden_total} of {_n_finite_total} "
             f"points inside the display rule "
             f"({_n_hidden_total} drawn as grey \u00d7)"))

# COMMAND ----------

# DBTITLE 1,Gamry vs Pipeline Overlay (Volume-based)
# ═══════════════════════════════════════════════════════════════════════════════
# GAMRY vs PIPELINE OVERLAY (Volume-based)
# Loads Gamry from DTA files, pipeline from Volume cache.
# No datago dependency, no kernel state needed.
# ═══════════════════════════════════════════════════════════════════════════════
import re, os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
 
try:
    _LEEPA = LEEPA
except NameError:
    _LEEPA = dbutils.widgets.get('leepa_id')
try:
    _COND = dbutils.widgets.get('conditions')
except Exception:
    _COND = 'ALL'
 
A_CELL_CM2 = 304.92
# Auto-detect Gamry folder for current Leepa
# Priority: per-Leepa folder > common Gamry folder (files matching Leepa ID)
_VOL_BASE = '/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev'
# The paths live in the setup cell (GAMRY_SEARCH_ROOTS) -- one place to edit.
# This cell used to carry its own copy of the folder list and then read every
# .dta in whichever folder it found, keyed by current: with a shared folder
# holding several campaigns, each key was overwritten by whichever file sorted
# last, so the overlay could compare this cell against another cell's sweep.
_GAMRY_VOL = gamry_root_for(_LEEPA) or Path(f'{_VOL_BASE}/Gamry')
_CACHE_VOL = Path('/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev/EIS_Results')
 
 
# ─── 1. Gamry DTA parser ───
def _parse_dta(fp):
    text = fp.read_text(encoding='latin-1')
    m = re.search(r'ZCURVE\s+TABLE\s*\n([^\n]*\n){2}', text)
    if not m:
        return None
    rows = []
    for ln in text[m.end():].strip().split('\n'):
        ln = ln.strip()
        if not ln or not ln[0].isdigit():
            if ln.startswith('STOPABORT'):
                break
            continue
        p = ln.replace(',', '.').split('\t')
        if len(p) >= 8:
            rows.append((float(p[2]), float(p[3]), float(p[4]), float(p[6])))
    if not rows:
        return None
    f, zr, zi, zm = zip(*rows)
    return dict(freq=np.array(f), Zreal=np.array(zr),
                Zimag=np.array(zi), Zmod=np.array(zm), name=fp.stem)
 
 
# ─── 2. Load the sweeps that belong to THIS order ───
_GAMRY_FILES, _GAMRY_BUILD = gamry_files(_LEEPA)
print(f"  Gamry source: {_GAMRY_VOL}")
print(f"  Order {_LEEPA} -> build {_GAMRY_BUILD or 'UNKNOWN'}"
      f"   ({len(_GAMRY_FILES)} sweep file(s) selected)")
if not _GAMRY_BUILD:
    print("    No build token found for this order. Every .dta that does not "
          "name a build is being read; if that folder holds more than one "
          "campaign, set PLATE_VERSION_OVERRIDE in the setup cell.")

_gamry = {}
for fp in _GAMRY_FILES:
    sp = _parse_dta(fp)
    if not sp:
        continue
    k = A_CELL_CM2 * 1e3  # ohm -> mohm.cm2
    # Condition from CurrVal_{number} in the filename -> e.g. "60A"
    _cm = re.search(r'CurrVal_(\d+)', fp.stem)
    _gamry_key = (_cm.group(1) + 'A') if _cm else sp['name']
    if _gamry_key in _gamry:
        # Two files for one current, after filtering by build, is a real
        # ambiguity rather than something to resolve by sort order.
        print(f"    WARNING: {_gamry_key} claimed by more than one file "
              f"({fp.name}); keeping the first and ignoring this one")
        continue
    _gamry[_gamry_key] = pd.DataFrame({
        'freq_hz': sp['freq'],
        'z_re': sp['Zreal'] * k,
        'z_im': sp['Zimag'] * k,
    }).sort_values('freq_hz').reset_index(drop=True)
    print(f"    {_gamry_key}: {len(sp['freq'])} pts  ({fp.name})")
 
 
# ─── 2b. Load Gamry from MF4 files (if no DTA found) ───
if not _gamry:
    try:
        from asammdf import MDF
        # Search for MF4 files matching this Leepa ID (any naming convention)
        _mf4_files = sorted(_GAMRY_VOL.glob(f'*{_LEEPA}*.mf4')) or sorted(_GAMRY_VOL.glob(f'*RO{_LEEPA}*.mf4'))
        if not _mf4_files:
            _mf4_files = sorted(_GAMRY_VOL.glob('*HFR*CurrVal*.mf4'))
        for fp in _mf4_files:
            # Try multiple naming patterns for condition extraction
            m = re.search(r'CurrVal_(\d+)', fp.stem)
            if m:
                cond_key = m.group(1) + 'A'
            else:
                # Fallback: use filename as-is for the Gamry key
                cond_key = fp.stem
            mdf = MDF(str(fp))
            freq = mdf.get('freq').samples
            zreal = mdf.get('zreal').samples
            zimag = mdf.get('zimag').samples
            mdf.close()
            if len(freq) < 3:
                print(f"    {fp.name}: only {len(freq)} pts — skipped")
                continue
            k = A_CELL_CM2 * 1e3  # ohm -> mohm.cm2
            _gamry[cond_key] = pd.DataFrame({
                'freq_hz': freq,
                'z_re': zreal * k,
                'z_im': zimag * k,
            }).sort_values('freq_hz').reset_index(drop=True)
            print(f"    {cond_key}: {len(freq)} pts from {fp.name} (MF4)")
    except ImportError:
        print("    asammdf not installed — run: %pip install asammdf")
    except Exception as e:
        print(f"    MF4 read error: {e}")
 
 
# ─── 3. Load pipeline spectra from Volume cache ───
def _load_pipeline(leepa, cond):
    """Per-segment and aggregate spectra for ONE condition, newest first.

    This used to try `<leepa>/<cond>/` on the Volume before anything else --
    the unversioned tree, which belongs to whichever run wrote last at
    whatever settings. It therefore drew a Gamry overlay against a run the
    operator had not selected and could not identify. It now asks
    result_dir(), the same resolver the heat maps and the run cell use, so
    the overlay and the maps can never disagree about which run they show.
    """
    d, prov = result_dir(leepa, cond)
    if d is None:
        return None, None, None, prov
    for sub in ('silver', 'csv'):
        sp = Path(d) / sub / 'spectra_clean.csv'
        if not sp.exists():
            continue
        seg = pd.read_csv(sp)
        # RECONSTRUCTED SEGMENTS BELONG ON THIS PLOT TOO.
        # They are kept out of spectra_clean.csv on purpose -- that file is
        # measurements only -- but leaving them off the overlay meant the
        # aggregate was drawn over 72 segments while the segment lines behind
        # it were 67, and the missing five were exactly the ones the operator
        # had asked to rebuild. They are read from their own file and marked
        # so nothing can mistake one for a measurement.
        rec_p = Path(d) / 'silver' / 'spectra_reconstructed.csv'
        rec = pd.read_csv(rec_p) if rec_p.exists() else None

        agg_p = Path(d) / 'silver' / 'cell_aggregate.csv'
        agg = pd.read_csv(agg_p) if agg_p.exists() else None
        cov = None
        if agg is not None:
            if 'area_coverage' in agg.columns:
                cov = float(np.nanmedian(agg['area_coverage']))
            agg = pd.DataFrame({
                'freq_hz': agg['freq_hz'],
                'z_re': agg['z_re_mohm_cm2'],
                'z_im': agg['z_im_mohm_cm2'],
            }).sort_values('freq_hz').reset_index(drop=True)
            if np.nanmedian(agg['z_im']) > 0:
                agg['z_im'] = -agg['z_im']
        return seg, agg, str(d), f'{prov}|{rec_p.name if rec is not None else ""}', rec, cov
    return None, None, None, f'{prov} (no spectra_clean.csv in it)', None, None
 
 
# ─── 4. Plot overlay for each condition ───
# The condition list comes from the widgets, through the same function the
# heat maps use, rather than from a literal list of every condition ever
# measured -- which is how a cell "for the selected condition" ended up
# drawing four.
try:
    _conds = selected_conditions()
except NameError:
    _conds = parse_conditions(_COND, ['45A', '60A', '150A', '450A'])
 
for cond in _conds:
    seg_df, agg_df, src, _prov, rec_df, _cov = _load_pipeline(_LEEPA, cond)
    _prov = _prov.split('|')[0]
    if seg_df is None:
        print(f"  {cond}: no pipeline results found ({_prov})")
        continue
    try:
        print(describe_source(cond, Path(src), _prov))
    except NameError:
        print(f"  {cond}: {_prov}  {src}")
 
    # Match Gamry condition
    gamry_df = _gamry.get(cond, None)
    LIKE_FOR_LIKE = True          # set False to see the uncorrected overlay
    _I_err, _L, _lo, _hi = 0.0, 0.0, float('nan'), float('nan')
    if LIKE_FOR_LIKE:
        # (1) shunt-calibration scale, from DC closure — NOT fitted to the Gamry
        _sp = {'45A':45., '60A':60., '150A':150., '450A':450.}.get(cond)
        _ss = Path(src) / 'silver' / 'segments_summary.csv'
        if _sp and _ss.exists() and agg_df is not None:
            _s = pd.read_csv(_ss)
            if {'j_dc_A_cm2', 'area_cm2'} <= set(_s.columns):
                _A = _s['area_cm2'].sum()
                _I = (_s['j_dc_A_cm2'] * _s['area_cm2']).sum() * A_CELL_CM2 / _A
                _I_err = _I / _sp - 1.0
                agg_df = agg_df.copy()
                agg_df[['z_re', 'z_im']] *= (1.0 + _I_err)

        # (2) strip the Gamry's OWN series inductance so both curves are L-free
        if gamry_df is not None:
            _fg = gamry_df['freq_hz'].values
            _k  = _fg >= _fg.max() / 10.0          # top decade
            if _k.sum() >= 4:
                _w = 2 * np.pi * _fg[_k]
                _L = float(np.sum(_w * gamry_df['z_im'].values[_k]) / np.sum(_w * _w))
                gamry_df = gamry_df.copy()
                gamry_df['z_im'] = gamry_df['z_im'] - 2 * np.pi * _fg * _L

        # (3) restrict both curves to the overlapping band
        if gamry_df is not None and agg_df is not None and len(agg_df) > 3:
            _lo = max(agg_df['freq_hz'].min(), gamry_df['freq_hz'].min())
            _hi = min(agg_df['freq_hz'].max(), gamry_df['freq_hz'].max())
            gamry_df = gamry_df[(gamry_df['freq_hz'] >= _lo) &
                                (gamry_df['freq_hz'] <= _hi)]
        print(f"  {cond}: I_err {100*_I_err:+.1f}% applied | "
              f"Gamry L {1e6*_L:.0f} nH.cm2 removed | band {_lo:.2f}-{_hi:.1f} Hz")
 
    fig = make_subplots(rows=1, cols=3,
        subplot_titles=['Nyquist', '|Z|(f) Bode', 'Phase(f)'],
        horizontal_spacing=0.06)
 
    segments = sorted(seg_df['segment'].unique())
    n_seg = len(segments)
    colors = [f'hsl({int(i*360/n_seg)}, 60%, 55%)' for i in range(n_seg)]
 
    # Per-segment spectra (thin)
    for i, seg in enumerate(segments):
        sd = seg_df[seg_df['segment'] == seg].sort_values('freq_hz')
        f = sd['freq_hz'].values
        zr = sd['z_re_mohm_cm2'].values
        zi = sd['z_im_mohm_cm2'].values
        ok = np.isfinite(zr) & np.isfinite(zi) & (zr > 0) & (zr <= 500)
        f, zr, zi = f[ok], zr[ok], zi[ok]
        if len(f) < 3:
            continue
        clr = colors[i]
        fig.add_trace(go.Scatter(
            x=zr, y=-zi, mode='lines', line=dict(width=0.8, color=clr),
            opacity=0.4, name='Segments' if i == 0 else f'Seg {seg}',
            legendgroup='seg', showlegend=(i == 0),
            hovertemplate=f'Seg {seg}<br>f=%{{customdata:.1f}} Hz<br>'
                f"Z'=%{{x:.1f}}<br>-Z''=%{{y:.1f}}<extra></extra>",
            customdata=f,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=f, y=np.abs(zr + 1j*zi), mode='lines',
            line=dict(width=0.8, color=clr), opacity=0.4,
            legendgroup='seg', showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=f, y=np.degrees(np.angle(zr + 1j*zi)),
            mode='lines', line=dict(width=0.8, color=clr), opacity=0.4,
            legendgroup='seg', showlegend=False), row=1, col=3)
 
    # Reconstructed segments (dashed, so they read as estimates at a glance)
    if rec_df is not None and len(rec_df):
        for j, seg in enumerate(sorted(rec_df['segment'].unique(),
                                       key=lambda x: int(x))):
            rd = rec_df[rec_df['segment'] == seg].sort_values('freq_hz')
            zr = rd['z_re_mohm_cm2'].values
            zi = rd['z_im_mohm_cm2'].values
            f = rd['freq_hz'].values
            ok = np.isfinite(zr) & np.isfinite(zi)
            if ok.sum() < 3:
                continue
            donors = str(rd['donors'].iloc[0])
            fig.add_trace(go.Scatter(
                x=zr[ok], y=-zi[ok], mode='lines',
                line=dict(width=1.6, color='#444', dash='dot'),
                opacity=0.8,
                name='Rebuilt from neighbours' if j == 0 else f'Seg {seg} (rebuilt)',
                legendgroup='rebuilt', showlegend=(j == 0),
                customdata=f[ok],
                hovertemplate=(f'<b>Seg {seg}</b> — REBUILT from {donors}<br>'
                               'f=%{customdata:.1f} Hz<br>'
                               "Z'=%{x:.1f}<br>-Z''=%{y:.1f}<extra></extra>"),
            ), row=1, col=1)
            Z = zr[ok] + 1j * zi[ok]
            fig.add_trace(go.Scatter(x=f[ok], y=np.abs(Z), mode='lines',
                line=dict(width=1.6, color='#444', dash='dot'), opacity=0.8,
                legendgroup='rebuilt', showlegend=False), row=1, col=2)
            fig.add_trace(go.Scatter(x=f[ok], y=np.degrees(np.angle(Z)),
                mode='lines', line=dict(width=1.6, color='#444', dash='dot'),
                opacity=0.8, legendgroup='rebuilt', showlegend=False), row=1, col=3)

    # Cell aggregate (thick red)
    if agg_df is not None and len(agg_df) > 3:
        Z_agg = agg_df['z_re'].values + 1j * agg_df['z_im'].values
        fig.add_trace(go.Scatter(
            x=agg_df['z_re'], y=-agg_df['z_im'],
            mode='lines+markers', line=dict(width=3, color='#c0392b'),
            marker=dict(size=5, color='#c0392b'),
            name=f'Pipeline aggregate ({n_seg} seg)', legendgroup='agg',
            customdata=agg_df['freq_hz'].values,
            hovertemplate="Aggregate<br>f=%{customdata:.1f} Hz<br>"
                "Z'=%{x:.1f}<br>-Z''=%{y:.1f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=agg_df['freq_hz'], y=np.abs(Z_agg),
            mode='lines+markers', line=dict(width=3, color='#c0392b'),
            marker=dict(size=4, color='#c0392b'),
            legendgroup='agg', showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=agg_df['freq_hz'], y=np.degrees(np.angle(Z_agg)),
            mode='lines+markers', line=dict(width=3, color='#c0392b'),
            marker=dict(size=4, color='#c0392b'),
            legendgroup='agg', showlegend=False), row=1, col=3)
 
    # Gamry reference (thick black)
    if gamry_df is not None:
        Z_g = gamry_df['z_re'].values + 1j * gamry_df['z_im'].values
        fig.add_trace(go.Scatter(
            x=gamry_df['z_re'], y=-gamry_df['z_im'],
            mode='lines+markers', line=dict(width=3.5, color='black'),
            marker=dict(size=6, color='black', symbol='diamond'),
            name=f'Gamry {cond}', legendgroup='gamry',
            customdata=gamry_df['freq_hz'].values,
            hovertemplate=f"Gamry {cond}<br>f=%{{customdata:.1f}} Hz<br>"
                "Z'=%{x:.1f}<br>-Z''=%{y:.1f}<extra></extra>",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=gamry_df['freq_hz'], y=np.abs(Z_g),
            mode='lines+markers', line=dict(width=3.5, color='black'),
            marker=dict(size=5, color='black', symbol='diamond'),
            legendgroup='gamry', showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=gamry_df['freq_hz'], y=np.degrees(np.angle(Z_g)),
            mode='lines+markers', line=dict(width=3.5, color='black'),
            marker=dict(size=5, color='black', symbol='diamond'),
            legendgroup='gamry', showlegend=False), row=1, col=3)
 
    fig.update_xaxes(title_text="Z' [mohm.cm2]", row=1, col=1)
    fig.update_yaxes(title_text="-Z'' [mohm.cm2]", row=1, col=1)
    fig.update_xaxes(title_text="f [Hz]", type="log", row=1, col=2)
    fig.update_yaxes(title_text="|Z| [mohm.cm2]", type="log", row=1, col=2)
    fig.update_xaxes(title_text="f [Hz]", type="log", row=1, col=3)
    fig.update_yaxes(title_text="Phase [deg]", row=1, col=3)
 
    _gstr = f' + Gamry {cond}' if gamry_df is not None else ''
    _nrec = 0 if rec_df is None else rec_df['segment'].nunique()
    # THE AGGREGATE'S COVERAGE IS PART OF WHAT IT MEANS.
    # Compared against a whole-cell Gamry sweep, an aggregate over 96 % of the
    # plate is not the same quantity as the instrument's, and the difference
    # is exactly the missing 4 %. Say which it is, on the figure.
    _covstr = ('' if _cov is None else
               f' — aggregate covers {100*_cov:.0f} % of the plate area'
               + ('' if _cov > 0.995 else
                  '; the Gamry measures 100 %, so this comparison is '
                  'short by the difference'))
    fig.update_layout(
        title=f'<b>Leepa {_LEEPA} / {cond}: {n_seg} measured'
              + (f' + {_nrec} rebuilt' if _nrec else '')
              + f' segments{_gstr}</b><br>'
              f'<sup>Source: {_prov} — {src}{_covstr}</sup>',
        height=580, width=1550,
        paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(t=100, b=55, r=280), hovermode='closest',
        legend=dict(font=dict(size=10), y=0.5, yanchor='middle'))
    show_fig(fig)          # one plot per figure, stacked
 
    print(f"\n  {cond}: {n_seg} measured"
          + (f" + {_nrec} rebuilt from neighbours" if _nrec else "")
          + f" segments from {src}")
    if _cov is not None:
        print(f"  aggregate covers {100*_cov:.1f} % of the plate area")
    if gamry_df is not None:
        print(f"  Gamry: {len(gamry_df)} pts")

# COMMAND ----------

# ── Gamry vs local-aggregate: is the ASR the same quantity? ──────────────────
import numpy as np, pandas as pd
from pathlib import Path
import gamry_dta
from config import A_CELL_CM2
 
# THREE HARD-CODED VALUES USED TO LIVE HERE, AND EACH ONE WAS A TRAP.
#   LEEPA = '2612030'      -- and it OVERWROTE the notebook's LEEPA global, so
#                             every later cell silently changed plate.
#   /tmp/eis_1003          -- another user's scratch directory. On any other
#                             cluster login this is simply someone else's run.
#   SNR_TAG = 'snr_-20.0'  -- a cache folder from one particular sweep, read
#                             no matter which SNR the widget said.
# All three now come from the widgets, through the same resolver the heat maps
# and the overlay use, so this table describes the run that is selected.
_ASR_LEEPA = str(_widget('leepa_id', 'LEEPA', 'leepa'))

# The filenames used to be written out here, V26_092 and all -- one campaign's
# sweeps, pasted into a cell that runs for whichever order is selected. Pick a
# different order and this table still read RO2612030's references.
# They are discovered now, from the build token that order resolves to.
# GAMRY_DIR is deliberately NOT reassigned: the run cell reads that global to
# build its Config, and a display cell that overwrites it changes the run.
_ASR_GAMRY_DIR = gamry_root_for(_ASR_LEEPA) or Path(
    '/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev/Gamry')
_ASR_FILES, _ASR_BUILD = gamry_files(_ASR_LEEPA)

DTA = {}
for _f in _ASR_FILES:
    _m = re.search(r'CurrVal_(\d+(?:[.,]\d+)?)', _f.stem)
    if _m:
        DTA.setdefault(_m.group(1).replace(',', '.') + 'A', _f.name)
# The setpoint is in the key, so it does not need a second table that can
# disagree with the first.
SETPOINT = {k: float(k[:-1]) for k in DTA}

print(f"  order {_ASR_LEEPA} -> build {_ASR_BUILD or 'UNKNOWN'} -> "
      f"{len(DTA)} sweep(s) in {_ASR_GAMRY_DIR}")
for _k in sorted(DTA, key=lambda c: SETPOINT[c]):
    print(f"    {_k:>6}: {DTA[_k]}")
if not DTA:
    print("    none found -- check GAMRY_SEARCH_ROOTS in the setup cell")

_ASR_PROV = {}


def find_silver(cond):
    """The silver directory for this condition, or None. See result_dir()."""
    d, prov = result_dir(_ASR_LEEPA, cond)
    _ASR_PROV[cond] = prov
    if d is None:
        return None
    sv = Path(d) / 'silver'
    return sv if (sv / 'cell_aggregate.csv').exists() else None
 
def decompose(f, Zg, Zl):
    """Zl = a*Zg + R + jwL  with a, R, L real — separates the three causes."""
    w = 2*np.pi*f; wt = 1.0/np.abs(Zg)
    A = np.vstack([np.column_stack([Zg.real, np.ones_like(f), np.zeros_like(f)])*wt[:,None],
                   np.column_stack([Zg.imag, np.zeros_like(f), w            ])*wt[:,None]])
    y = np.concatenate([Zl.real*wt, Zl.imag*wt])
    p,*_ = np.linalg.lstsq(A, y, rcond=None)
    res = Zl - (p[0]*Zg + p[1] + 1j*w*p[2])
    return p, 100*np.median(np.abs(res)/np.abs(Zl))
 
print(f"A_CELL_CM2 = {A_CELL_CM2:.2f} cm2   (plate footprint)\n")
hdr = (f"{'cond':>5} {'A_used':>7} {'n_f':>4} {'band [Hz]':>17} {'scale a':>8} "
       f"{'implied A':>10} {'R_ser mΩcm²':>11} {'L_ser nH':>11} {'resid':>7} {'I_err':>7}")
print(hdr); print('-'*len(hdr))
 
for cond, fn in DTA.items():
    sv = find_silver(cond)
    if sv is None:
        print(f"{cond:>5}   no cell_aggregate.csv "
              f"({_ASR_PROV.get(cond, 'not found')})")
        continue
    dta = _ASR_GAMRY_DIR/fn
    if not dta.exists():
        print(f"{cond:>5}   missing {dta}"); continue
 
    loc = pd.read_csv(sv/'cell_aggregate.csv').sort_values('freq_hz')
    f_l = loc.freq_hz.values
    Z_l = (loc.z_re_mohm_cm2.values + 1j*loc.z_im_mohm_cm2.values)/1e3      # ohm.cm2
 
    sw  = gamry_dta.read_dta(dta).sorted()          # field is .Z, in OHM
    f_g, Z_g = np.asarray(sw.freq,float), np.asarray(sw.Z,complex)
 
    lo, hi = max(f_l.min(), f_g.min()), min(f_l.max(), f_g.max())
    m = (f_l >= lo) & (f_l <= hi)
    if m.sum() < 8:
        print(f"{cond:>5}   only {m.sum()} overlapping points"); continue
    f, Zl = f_l[m], Z_l[m]
    Zg = (np.interp(np.log(f), np.log(f_g), Z_g.real)
          + 1j*np.interp(np.log(f), np.log(f_g), Z_g.imag)) * A_CELL_CM2    # <- area under test
 
    p, resid = decompose(f, Zg, Zl)
    seg = pd.read_csv(sv/'segments_summary.csv')
    A_used = float(seg.area_cm2.sum())
    I_full = float((seg.j_dc_A_cm2*seg.area_cm2).sum()) * A_CELL_CM2 / A_used
    print(f"{cond:>5} {A_used:7.1f} {len(f):4d} {lo:7.2f}-{hi:8.1f} {p[0]:8.4f} "
          f"{p[0]*A_CELL_CM2:10.1f} {1e3*p[1]:+9.2f} {1e9*p[2]:+11.4g} "
          f"{resid:6.1f}% {100*(I_full/SETPOINT[cond]-1):+6.1f}%")
    
    
COMMON = (0.30, 700.0)        # a band all four conditions cover
BANDS  = [(0.2,1),(1,10),(10,100),(100,1000),(1000,5000)]
 
print(f"{'cond':>5} | " + "  ".join(f"{a:g}-{b:g}Hz" for a,b in BANDS)
      + "   ||   a(common)   R_ser   resid")
for cond, fn in DTA.items():
    sv = find_silver(cond)
    if sv is None: print(f"{cond:>5}  (no data)"); continue
    loc = pd.read_csv(sv/'cell_aggregate.csv').sort_values('freq_hz')
    f_l = loc.freq_hz.values
    Z_l = (loc.z_re_mohm_cm2.values + 1j*loc.z_im_mohm_cm2.values)/1e3
    sw = gamry_dta.read_dta(_ASR_GAMRY_DIR/fn).sorted()
    f_g, Z_g = np.asarray(sw.freq,float), np.asarray(sw.Z,complex)
    lo, hi = max(f_l.min(), f_g.min()), min(f_l.max(), f_g.max())
    m = (f_l>=lo)&(f_l<=hi); f, Zl = f_l[m], Z_l[m]
    Zg = (np.interp(np.log(f), np.log(f_g), Z_g.real)
          + 1j*np.interp(np.log(f), np.log(f_g), Z_g.imag)) * A_CELL_CM2
    r = np.abs(Zl)/np.abs(Zg)
    cells = [f"{np.median(r[(f>=a)&(f<b)]):9.3f}" if ((f>=a)&(f<b)).sum() else f"{'--':>9}"
             for a,b in BANDS]
    k = (f>=COMMON[0])&(f<=COMMON[1])
    if k.sum() >= 8:
        p, res = decompose(f[k], Zg[k], Zl[k])
        tail = f"  ||  {p[0]:8.4f}  {1e3*p[1]:+7.2f}  {res:4.1f}%"
    else:
        tail = f"  ||  only {k.sum()} pts in the common band"
    print(f"{cond:>5} | " + " ".join(cells) + tail)

# COMMAND ----------

# DBTITLE 1,Interactive Plate Heatmaps (plate_viewer)
# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE PLATE HEATMAPS — for the condition the widgets actually selected
#
# THIS CELL USED TO IGNORE THE CONDITION WIDGET.
# It listed every <leepa>/<cond> directory on the Volume and drew a map for
# each, so selecting 450 A still produced a 60 A heat map: with one figure per
# condition, the last one drawn is the one you end up looking at, and 60A
# sorts last among 150A, 450A, 45A, 60A. The title said 60A and was telling
# the truth -- about a condition nobody had asked for.
#
# It also pasted its own copy of the cache layout (<cond>/snr_<v>/), which is
# not the layout the run cell writes any more, and it preferred the Volume
# cache over the run this kernel had just finished unless the run-mode widget
# happened to say 'rerun'.
#
# All three are now one call to result_dir() from the Run Dashboard cell:
# this session's run first, then the cache entry for the CURRENT mode and
# settings digest, then older layouts -- each labelled, with its age printed.
# Nothing is drawn for a condition that was not selected, and a selected
# condition with no results says so instead of a neighbour's map appearing in
# its place.
# ═══════════════════════════════════════════════════════════════════════════════
import sys, html
import numpy as np
import pandas as pd
from pathlib import Path
from IPython.display import display

_map_dir = '/Workspace/Users/uum5fe@bosch.com/master_thesis/map'
if _map_dir not in sys.path:
    sys.path.insert(0, _map_dir)

# The interactive viewer lives outside this folder. It is optional: the
# static maps below are drawn by plate_maps (beside this notebook), which
# needs nothing but r2d2_geometry.
try:
    from plate_viewer import write_html, Field
except Exception as _pv_err:                               # noqa: BLE001
    write_html = Field = None
    print(f'  plate_viewer not importable ({type(_pv_err).__name__}); '
          f'interactive viewer skipped, static maps still drawn')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_CACHE_VOL = Path('/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev/EIS_Results')

_FIELD_DEFS = [
    ('R_ohmic',      'HFR (Rs)',                   'mohm.cm2', 'magma',   1),
    ('R_ct',         'R_ct (charge transfer)',     'mohm.cm2', 'inferno', 1),
    ('R_mt',         'R_mt (mass transport)',      'mohm.cm2', 'magma',   1),
    ('R_pol',        'R_pol (total polarisation)', 'mohm.cm2', 'thermal', 1),
    ('j_dc',         'Current density',            'A/cm2',    'cividis', 3),
    ('Z_mag_100Hz',  '|Z| at 100 Hz',              'mohm.cm2', 'viridis', 1),
    ('phase_100Hz',  'Phase at 100 Hz',            'deg',      'humid',   1),
]


# ── widgets ───────────────────────────────────────────────────────────────────
# _widget, result_dir, selected_conditions and describe_source all live in the
# widgets / Run Dashboard cells now. This cell used to carry its own copy of
# the first one and its own idea of where results live, which is precisely how
# it drifted away from the run cell and started showing another condition.
_LEEPA = str(_widget('leepa_id', 'LEEPA', 'leepa'))
# The SNR PRINTED must be the SNR LOOKED UP, or the heading is decoration.
# Both are MIN_SNR_DB, the value the run cell built its config from.
_SNR = f'{MIN_SNR_DB:g}'

# _snr_dir_names, _find_gold_csv and _available_snrs used to live here. They
# encoded the cache layout a second time, and a second copy of a path shape is
# a copy that goes stale: the run cell now writes
# <cond>/mode_<mode>/snr_<v>_<digest>/, which none of them knew about. The one
# implementation is _cache_dir(), reached through result_dir().


# ── which conditions, and which CSV for each ─────────────────────────────────
# ONLY the conditions the widgets asked for. Never "everything on the Volume".
_COND_GOLD: dict[str, tuple[Path, str]] = {}
_MISSING: list[str] = []

print(f'  plate {_LEEPA}   SNR {_SNR or "(not set)"}   '
      f'mode {EVALUATION_MODE}   selected: {", ".join(selected_conditions())}')

for cond in selected_conditions():
    _d, _prov = result_dir(_LEEPA, cond)
    print(describe_source(cond, _d, _prov))
    if _d is None:
        _MISSING.append(cond)
        continue
    _csv = None
    for _rel in ('gold/plate_summary.csv', 'plate_summary.csv'):
        if (_d / _rel).is_file():
            _csv = _d / _rel
            break
    if _csv is None:
        _hits = sorted(Path(_d).rglob('plate_summary.csv'))
        _csv = _hits[0] if _hits else None
    if _csv is None:
        print(f'    no plate_summary.csv here -- gold did not run, or '
              f'stop_after was set below gold')
        _MISSING.append(cond)
        continue
    _COND_GOLD[cond] = (_csv, _prov)

if _MISSING:
    print(f'\n  No map drawn for: {", ".join(_MISSING)}. Nothing else is '
          f'shown in their place -- a map from a condition you did not select '
          f'is worse than no map.')
if not _COND_GOLD:
    print('  Run the pipeline cell for this condition, mode and SNR first.')


for cond, (gold_csv, prov) in sorted(_COND_GOLD.items()):
    df = pd.read_csv(gold_csv)
    measured = df[df['class'] == 'measured'] if 'class' in df.columns else df

    print(f'\n{"="*75}')
    print(f'  PLATE MAPS — {_LEEPA} / {cond} / SNR {_SNR or "(default)"}'
          f' / mode {EVALUATION_MODE}'
          f'   ({len(measured)}/{len(df)} measured)')
    print(f'  read: {prov}')
    print(f'  file: {gold_csv}  (mtime {gold_csv.stat().st_mtime:.0f})')
    print(f'{"="*75}')
    if prov.startswith('LEGACY'):
        print('  WARNING: this came from an older cache layout. It was NOT '
              'written by a run at the mode and settings selected above, and '
              'nothing recorded what settings did produce it. Re-run this '
              'condition before using these numbers.')

    # Outputs go INSIDE the snapshot that produced them. Writing them to the
    # shared <cond>/gold/ is why every SNR overwrote the previous one's PNGs.
    _out_dir = gold_csv.parent
    _cols = set(df.columns)

    fields = []
    for col, label, unit, ramp, dec in _FIELD_DEFS:
        if col not in _cols:
            continue
        vals = {int(r['segment']): float(r[col]) for _, r in df.iterrows()
                if pd.notna(r[col]) and np.isfinite(r[col])}
        if not vals:
            # A column that is entirely blank is not "zero everywhere"; say so
            # rather than dropping it silently. R_mt does this on any plate
            # whose segments lost their sub-16 Hz points.
            print(f'  {col}: no finite values on any segment -- not mapped')
            continue
        if Field is not None:
            fields.append(Field(col, label, unit, ramp, dec, vals))

    if fields and write_html is not None:
        _html_out = _out_dir / 'plate_interactive.html'
        _html_out.parent.mkdir(parents=True, exist_ok=True)
        # The condition belongs in the figure itself. A screenshot of a heat
        # map outlives the cell output it was printed under, and this title
        # is the only thing that travels with it.
        write_html(fields, _html_out,
                   title=f'{_LEEPA} / {cond} / SNR {_SNR or "default"}',
                   subtitle=f'{len(measured)} segments measured | plate gen1'
                            f' | mode {EVALUATION_MODE} | {prov}')
        print(f'  Interactive viewer: {_html_out}')

        # html.escape handles & BEFORE " -- escaping only the quote corrupts
        # every existing entity in the document.
        _doc = html.escape(_html_out.read_text(encoding='utf-8'), quote=True)
        displayHTML(f'<iframe srcdoc="{_doc}" width="100%" height="700" '
                    f'style="border:none;"></iframe>')

    # ── Static maps, one per parameter, VALUE PRINTED IN EVERY SEGMENT ──
    # All resistance maps share one look: the 'magma' ramp the mass-transport
    # map always had, and a colour scale over the 5th..95th percentile of the
    # plate. On a min..max scale one low segment (e.g. 50 against a plate of
    # 60-70 mOhm.cm2) stretches the bar and the rest of the plate collapses
    # into a few shades -- the uniformity pattern is in the numbers but not
    # in the colours, which is how the HFR map looked. Values outside the
    # percentile range keep the end colour, are printed as measured, and the
    # colour bar gets an arrow for them.
    _classes = ({str(int(r['segment'])): str(r['class'])
                 for _, r in df.iterrows()} if 'class' in _cols else {})
    _STATIC = [('R_ohmic', 'magma'), ('R_ct', 'magma'), ('R_mt', 'magma'),
               ('R_pol', 'magma')]
    _defs = {d[0]: d for d in _FIELD_DEFS}
    for col, ramp in _STATIC:
        if col not in _cols or col not in _defs:
            continue
        _, label, unit, _ramp_unused, dec = _defs[col]
        vals = {int(r['segment']): float(r[col]) for _, r in df.iterrows()
                if pd.notna(r[col]) and np.isfinite(r[col])}
        if not vals:
            continue
        fig, ax = plate_maps.draw_value_map(
            vals, label=label, unit=unit.replace('mohm.cm2', 'mΩ·cm²'),
            cmap=ramp, decimals=dec, classes=_classes,
            title=(f'{_LEEPA} / {cond} / SNR {_SNR or "default"} / '
                   f'{EVALUATION_MODE} — {label}'))
        _png_out = _out_dir / f'plate_{col}.png'
        fig.savefig(str(_png_out), dpi=200, bbox_inches='tight')
        display(fig)
        plt.close(fig)
        print(f'  {col}: {_png_out}')


# COMMAND ----------

# DBTITLE 1,ECM Fitting: Rs + L + (Rct1 || CPE1) + (Rct2 || CPE2)
# ═══════════════════════════════════════════════════════════════════════════════
# ECM FIT — every segment and the whole-cell aggregate, in ONE figure
#
# Model:  Z(f) = Rs + jwL + sum_k  R_k / (1 + (jw*tau_k)^n_k)
#
# Parameterised by tau, not by the CPE admittance Y0.  The two forms are the
# same circuit (Y0 = tau^n / R), but with Y0 the three parameters of an arc
# trade against each other over six decades and the optimiser walks a curved
# valley to a false optimum that still reports success=True.  With tau each
# arc has a location, a size and a shape, and two arcs whose taus differ by
# decades actually separate.
#
# The arc count is chosen by AICc rather than fixed at two: fitting two arcs
# to a one-arc spectrum splits one relaxation into two coincident halves
# whose individual R and tau mean nothing while their SUM still looks fine.
#
# This is csv_pipeline.choose_n_arcs — the fitter the pipeline itself uses,
# so the notebook and gold/plate_summary.csv report the same numbers.
# ═══════════════════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
 
import csv_pipeline as _cp
 
_Z_MODEL = _cp.z_model
_CHOOSE  = _cp.choose_n_arcs
 
 
def _p_vector(res):
    """params dict -> the flat vector z_model expects."""
    p = res["params"]
    v = [p["Rs"], p["L"]]
    for k in range(res["n_arcs"]):
        v += [p[f"R{k+1}"], p[f"tau{k+1}"], p[f"n{k+1}"]]
    return np.asarray(v, float)
 
 
def _fit_one(f_hz, Z_mohm, sigma_rel=None):
    """Fit one spectrum given in mOhm.cm2.  Returns everything the plots need."""
    f = np.asarray(f_hz, float)
    Z = np.asarray(Z_mohm, complex)
    keep = np.isfinite(f) & np.isfinite(Z.real) & np.isfinite(Z.imag) & (f > 0)
    f, Z = f[keep], Z[keep]
    if sigma_rel is not None:
        sigma_rel = np.asarray(sigma_rel, float)[keep]
    if f.size < 6:
        return {"ok": False, "reason": f"{f.size} usable points"}
 
    r = _CHOOSE(f, Z / 1000.0, sigma_rel)          # fitter works in Ohm.cm2
    if not r.get("ok"):
        return r
 
    pv = _p_vector(r)
    r["freq"] = f
    r["Z_meas"] = Z
    r["Z_fit"] = 1000.0 * _Z_MODEL(pv, f, r["n_arcs"])
    # Smooth curve for the Nyquist line: the measured grid is too coarse to
    # draw an arc, and a polyline through 25 points is not the model.
    f_s = np.logspace(np.log10(f.min()), np.log10(f.max()), 400)
    r["freq_smooth"] = f_s
    r["Z_fit_smooth"] = 1000.0 * _Z_MODEL(pv, f_s, r["n_arcs"])
    r["res_pct"] = 100.0 * np.abs(Z - r["Z_fit"]) / np.abs(Z)
    return r
 
 
# ─── Fit every segment + the aggregate, per condition ───
ECM_ALL = {}        # {cond: {'segments': {seg: res}, 'aggregate': res|None}}
 
for cond, pr in PIPELINE_RESULTS.items():
    if pr is None:
        continue
    out_dir = Path(pr['out_dir'])
    sp_path = spectra_csv(out_dir)
    if sp_path is None or not Path(sp_path).exists():
        print(f"  {cond}: no spectra_clean.csv -> skip ECM")
        continue
 
    df = pd.read_csv(sp_path)
    seg_fits, failed = {}, {}
    for seg in sorted(df['segment'].unique(), key=int):
        sd = df[df['segment'] == seg].sort_values('freq_hz')
        sig = sd['sigma_rel'].values if 'sigma_rel' in sd.columns else None
        r = _fit_one(sd['freq_hz'].values,
                     sd['z_re_mohm_cm2'].values + 1j * sd['z_im_mohm_cm2'].values,
                     sig)
        if r.get("ok"):
            seg_fits[int(seg)] = r
        else:
            failed[int(seg)] = r.get("reason", "?")
 
    # ── whole-cell aggregate: Z_cell = A_cell / sum_s (A_s / Z_s) ──
    agg_fit = None
    for _cand in (out_dir / 'silver' / 'cell_aggregate.csv',
                  out_dir / 'csv' / 'cell_aggregate.csv'):
        if _cand.exists():
            ad = pd.read_csv(_cand).sort_values('freq_hz')
            agg_fit = _fit_one(ad['freq_hz'].values,
                               ad['z_re_mohm_cm2'].values + 1j * ad['z_im_mohm_cm2'].values)
            if not agg_fit.get("ok"):
                print(f"  {cond}: aggregate fit failed — {agg_fit.get('reason')}")
                agg_fit = None
            break
    else:
        print(f"  {cond}: no cell_aggregate.csv — run at least the silver stage")
 
    ECM_ALL[cond] = {'segments': seg_fits, 'aggregate': agg_fit}
 
    n1 = sum(1 for r in seg_fits.values() if r['n_arcs'] == 1)
    good = [r for r in seg_fits.values() if r['verdict'] == 'good']
    print(f"  {cond}: {len(seg_fits)}/{len(seg_fits)+len(failed)} segments fitted "
          f"({n1} one-arc, {len(seg_fits)-n1} two-arc), "
          f"{len(good)} with chi2_nu in [0.2, 5]")
    if failed:
        print(f"      not fitted: {failed}")

# COMMAND ----------

# DBTITLE 1,ECM Visualization: Nyquist Overlay + Parameter Heatmaps
# ═══════════════════════════════════════════════════════════════════════════════
# ONE figure per condition.  The dropdown picks what is drawn:
#   "All segments"  — every segment overlaid, colour-graded by Rs
#   "Aggregate"     — the whole-cell parallel sum and its own overall ECM fit
#   "Seg N"         — one segment, its fit, and its residual trace
# Left panel: Nyquist (measured markers + fitted line).
# Right panel: |dZ|/|Z| against frequency, with the 2 % and 5 % guides.
# ═══════════════════════════════════════════════════════════════════════════════
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
 
try:
    import plotly.express as _px
    _SEQ = _px.colors.sequential.Viridis
except Exception:
    _SEQ = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']
 
 
def _grade(v, lo, hi):
    """Pick a colour from the sequential ramp for value v."""
    if not np.isfinite(v) or hi <= lo:
        return _SEQ[len(_SEQ) // 2]
    t = min(max((v - lo) / (hi - lo), 0.0), 1.0)
    return _SEQ[int(round(t * (len(_SEQ) - 1)))]
 
 
for cond, blob in ECM_ALL.items():
    seg_fits, agg_fit = blob['segments'], blob['aggregate']
    if not seg_fits and agg_fit is None:
        continue
 
    segs = sorted(seg_fits)
    rs_all = [seg_fits[s]['params']['Rs'] * 1000 for s in segs]
    rs_lo, rs_hi = (np.percentile(rs_all, [5, 95]) if rs_all else (0, 1))
 
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.56, 0.44],
        subplot_titles=['Nyquist — measured vs ECM', 'Residual |ΔZ|/|Z|'],
        horizontal_spacing=0.09)
 
    # trace_owner[i] = 'seg:<n>' | 'agg' — used to build the dropdown masks
    owner = []
 
    def _add(o, tr, row, col):
        fig.add_trace(tr, row=row, col=col)
        owner.append(o)
 
    for s in segs:
        r = seg_fits[s]
        clr = _grade(r['params']['Rs'] * 1000, rs_lo, rs_hi)
        tag = f"{r['n_arcs']} arc, χ²ν={r['chi2_nu']:.2f}"
        _add(f'seg:{s}', go.Scatter(
            x=r['Z_meas'].real, y=-r['Z_meas'].imag, mode='markers',
            marker=dict(size=5, color=clr, opacity=0.75,
                        line=dict(width=0.4, color='#333')),
            name=f'Seg {s}', legendgroup=f'seg{s}', showlegend=True,
            hovertemplate=(f'<b>Seg {s}</b> ({tag})<br>'
                           "Z'=%{x:.1f}<br>-Z''=%{y:.1f} mΩ·cm²<extra></extra>")),
            1, 1)
        _add(f'seg:{s}', go.Scatter(
            x=r['Z_fit_smooth'].real, y=-r['Z_fit_smooth'].imag, mode='lines',
            line=dict(color=clr, width=1.6),
            name=f'Seg {s} fit', legendgroup=f'seg{s}', showlegend=False,
            hoverinfo='skip'), 1, 1)
        _add(f'seg:{s}', go.Scatter(
            x=r['freq'], y=r['res_pct'], mode='lines+markers',
            marker=dict(size=3, color=clr), line=dict(width=1, color=clr),
            name=f'Seg {s} res', legendgroup=f'seg{s}', showlegend=False,
            hovertemplate=f'Seg {s}<br>f=%{{x:.3g}} Hz<br>|res|=%{{y:.2f}} %<extra></extra>'),
            1, 2)
 
    if agg_fit is not None:
        a = agg_fit
        atag = (f"{a['n_arcs']} arc, χ²ν={a['chi2_nu']:.2f}, "
                f"Rs={a['params']['Rs']*1000:.1f} mΩ·cm²")
        _add('agg', go.Scatter(
            x=a['Z_meas'].real, y=-a['Z_meas'].imag, mode='markers',
            marker=dict(size=8, color='#c0392b', symbol='circle-open',
                        line=dict(width=2)),
            name='Aggregate (whole cell)', legendgroup='agg',
            hovertemplate=("<b>Aggregate</b><br>Z'=%{x:.1f}<br>"
                           "-Z''=%{y:.1f} mΩ·cm²<extra></extra>")), 1, 1)
        _add('agg', go.Scatter(
            x=a['Z_fit_smooth'].real, y=-a['Z_fit_smooth'].imag, mode='lines',
            line=dict(color='#111', width=3),
            name=f'Overall ECM fit — {atag}', legendgroup='agg',
            hoverinfo='skip'), 1, 1)
        _add('agg', go.Scatter(
            x=a['freq'], y=a['res_pct'], mode='lines+markers',
            marker=dict(size=5, color='#c0392b'),
            line=dict(width=2, color='#c0392b'),
            name='Aggregate res', legendgroup='agg', showlegend=False,
            hovertemplate='Aggregate<br>f=%{x:.3g} Hz<br>|res|=%{y:.2f} %<extra></extra>'),
            1, 2)
 
    owner = np.array(owner)
 
    # ─── Dropdown ───
    def _mask(pred):
        return [bool(pred(o)) for o in owner]
 
    buttons = [
        dict(label=f'All segments ({len(segs)})', method='update',
             args=[{'visible': _mask(lambda o: o.startswith('seg:'))},
                   {'title.text': f'<b>ECM fit — all {len(segs)} segments — '
                                  f'Leepa {LEEPA}, {cond}</b>'}]),
        dict(label='All segments + aggregate', method='update',
             args=[{'visible': _mask(lambda o: True)},
                   {'title.text': f'<b>ECM fit — all segments + whole-cell '
                                  f'aggregate — Leepa {LEEPA}, {cond}</b>'}]),
    ]
    if agg_fit is not None:
        a = agg_fit
        p = a['params']
        arcs = ', '.join(f"R{k+1}={p[f'R{k+1}']*1000:.1f} τ{k+1}={p[f'tau{k+1}']*1e3:.3g} ms "
                         f"n{k+1}={p[f'n{k+1}']:.2f}" for k in range(a['n_arcs']))
        buttons.append(dict(
            label='Aggregate only (overall fit)', method='update',
            args=[{'visible': _mask(lambda o: o == 'agg')},
                  {'title.text': f'<b>Overall ECM fit — whole-cell aggregate — '
                                 f'Leepa {LEEPA}, {cond}</b><br>'
                                 f'<sup>Rs={p["Rs"]*1000:.1f} mΩ·cm², {arcs}, '
                                 f'χ²ν={a["chi2_nu"]:.2f} ({a["verdict"]})</sup>'}]))
    for s in segs:
        r = seg_fits[s]
        p = r['params']
        arcs = ', '.join(f"R{k+1}={p[f'R{k+1}']*1000:.1f} mΩ·cm² "
                         f"τ{k+1}={p[f'tau{k+1}']*1e3:.3g} ms n{k+1}={p[f'n{k+1}']:.2f}"
                         for k in range(r['n_arcs']))
        buttons.append(dict(
            label=f'Seg {s}', method='update',
            args=[{'visible': _mask(lambda o, s=s: o == f'seg:{s}')},
                  {'title.text': f'<b>ECM fit — segment {s} — Leepa {LEEPA}, '
                                 f'{cond}</b><br><sup>Rs={p["Rs"]*1000:.1f} mΩ·cm², '
                                 f'{arcs}, χ²ν={r["chi2_nu"]:.2f} ({r["verdict"]})</sup>'}]))
 
    fig.add_hline(y=2, line=dict(color='green', dash='dash', width=1), row=1, col=2)
    fig.add_hline(y=5, line=dict(color='orange', dash='dash', width=1), row=1, col=2)
 
    fig.update_xaxes(title_text="Z' [mΩ·cm²]", showgrid=True, gridcolor='#eee',
                     row=1, col=1)
    fig.update_yaxes(title_text="-Z'' [mΩ·cm²]", showgrid=True, gridcolor='#eee',
                     scaleanchor='x', scaleratio=1, row=1, col=1)
    fig.update_xaxes(title_text='f [Hz]', type='log', showgrid=True,
                     gridcolor='#eee', row=1, col=2)
    fig.update_yaxes(title_text='|ΔZ|/|Z| [%]', rangemode='tozero',
                     showgrid=True, gridcolor='#eee', row=1, col=2)
 
    fig.update_layout(
        title=dict(text=f'<b>ECM fit — all {len(segs)} segments — '
                        f'Leepa {LEEPA}, {cond}</b>'),
        height=680, width=1500, paper_bgcolor='white', plot_bgcolor='white',
        hovermode='closest', margin=dict(t=130, r=250),
        legend=dict(font=dict(size=9), y=0.5, yanchor='middle',
                    tracegroupgap=0),
        updatemenus=[dict(buttons=buttons, direction='down', showactive=True,
                          x=0.0, xanchor='left', y=1.16, yanchor='top',
                          bgcolor='white', bordercolor='#999')])
 
    # start on "All segments"
    for i, o in enumerate(owner):
        fig.data[i].visible = o.startswith('seg:')
 
    show_fig(fig, html=True)   # one plot per figure, stacked

# COMMAND ----------

# DBTITLE 1,Overall ECM fit on the aggregate impedance (Nyquist + Bode + residual)
# ═══════════════════════════════════════════════════════════════════════════════
# The whole-cell aggregate with ONE overall ECM fit through it, in the same
# 3-panel form as the Gamry validation cell so the two can be read side by
# side.  The per-segment median is drawn beside it: an ECM fitted to the
# aggregate is NOT the average of the segment ECMs — the segments combine in
# parallel, so a spread in Rs pulls the aggregate below the mean — and seeing
# the two together is the point.
# ═══════════════════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
 
AGG_ECM = {}      # {cond: params dict in mOhm.cm2}
 
for cond, blob in ECM_ALL.items():
    a = blob['aggregate']
    if a is None:
        print(f"  {cond}: no aggregate fit to show")
        continue
 
    p = a['params']
    Zm, Zf = a['Z_meas'], a['Z_fit']
    Zs, fs = a['Z_fit_smooth'], a['freq_smooth']
    f = a['freq']
 
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.075,
                        subplot_titles=['Nyquist', '|Z|(f)', 'Residual |ΔZ|/|Z|'])
    MEAS, FIT = '#c0392b', '#111111'
 
    fig.add_trace(go.Scatter(x=Zm.real, y=-Zm.imag, mode='markers',
        marker=dict(size=7, color=MEAS), name='Aggregate (measured)',
        hovertemplate="Z'=%{x:.1f}<br>-Z''=%{y:.1f} mΩ·cm²<extra></extra>"),
        row=1, col=1)
    fig.add_trace(go.Scatter(x=Zs.real, y=-Zs.imag, mode='lines',
        line=dict(color=FIT, width=2.5), name='Overall ECM fit',
        hoverinfo='skip'), row=1, col=1)
    # Rs and Rs+R_pol markers: where the model says the arc starts and ends
    _rs, _rpol = p['Rs'] * 1000, a['R_pol'] * 1000
    fig.add_trace(go.Scatter(x=[_rs, _rs + _rpol], y=[0, 0], mode='markers',
        marker=dict(symbol='star', size=13, color='#1f77b4'),
        name=f'Rs={_rs:.1f} · Rs+Rpol={_rs+_rpol:.1f}'), row=1, col=1)
 
    # |Z| Bode, measured points against the fitted curve
    fig.add_trace(go.Scatter(x=f, y=np.abs(Zm), mode='markers',
        marker=dict(size=6, color=MEAS), showlegend=False,
        hovertemplate='f=%{x:.3g} Hz<br>|Z|=%{y:.1f} mΩ·cm²<extra></extra>'),
        row=1, col=2)
    fig.add_trace(go.Scatter(x=fs, y=np.abs(Zs), mode='lines',
        line=dict(color=FIT, width=2.5), showlegend=False,
        hoverinfo='skip'), row=1, col=2)
    fig.update_xaxes(title_text='f [Hz]', type='log', row=1, col=2)
    fig.update_yaxes(title_text='|Z| [mΩ·cm²]', type='log', row=1, col=2)
 
    fig.add_trace(go.Scatter(x=f, y=a['res_pct'], mode='lines+markers',
        marker=dict(size=5, color=MEAS), line=dict(width=1.5, color=MEAS),
        showlegend=False,
        hovertemplate='f=%{x:.3g} Hz<br>|res|=%{y:.2f} %<extra></extra>'),
        row=1, col=3)
    fig.add_hline(y=2, line=dict(color='green', dash='dash', width=1), row=1, col=3)
    fig.add_hline(y=5, line=dict(color='orange', dash='dash', width=1), row=1, col=3)
 
    fig.update_xaxes(title_text="Z' [mΩ·cm²]", row=1, col=1)
    fig.update_yaxes(title_text="-Z'' [mΩ·cm²]", scaleanchor='x', scaleratio=1,
                     row=1, col=1)
    fig.update_xaxes(title_text='f [Hz]', type='log', row=1, col=3)
    fig.update_yaxes(title_text='|ΔZ|/|Z| [%]', rangemode='tozero', row=1, col=3)
    fig.update_xaxes(showgrid=True, gridcolor='#eee')
    fig.update_yaxes(showgrid=True, gridcolor='#eee')
 
    arcs = ' · '.join(f"R{k+1}={p[f'R{k+1}']*1000:.1f} mΩ·cm² "
                      f"τ{k+1}={p[f'tau{k+1}']*1e3:.3g} ms n{k+1}={p[f'n{k+1}']:.2f}"
                      for k in range(a['n_arcs']))
    fig.update_layout(
        title=(f'<b>Overall ECM fit — whole-cell aggregate — Leepa {LEEPA}, {cond}</b>'
               f'<br><sup>Rs={_rs:.1f} · L={p["L"]*1e3:.3g} mΩ·cm²·ms · {arcs}'
               f' · R_pol={_rpol:.1f} mΩ·cm²'
               f' · χ²ν={a["chi2_nu"]:.2f} ({a["verdict"]})'
               f' · {a["n_arcs"]} arc(s) chosen by AICc</sup>'),
        height=520, width=1550, paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(t=110, r=260), hovermode='closest',
        legend=dict(font=dict(size=10), y=0.5, yanchor='middle'))
    show_fig(fig, html=True)   # one plot per figure, stacked
 
    AGG_ECM[cond] = {'Rs_mohm_cm2': _rs, 'R_pol_mohm_cm2': _rpol,
                     'R_ct_mohm_cm2': a['R_ct'] * 1000,
                     'R_mt_mohm_cm2': a['R_mt'] * 1000,
                     'n_arcs': a['n_arcs'], 'chi2_nu': a['chi2_nu'],
                     'verdict': a['verdict'],
                     'max_res_pct': float(a['res_pct'].max())}

# COMMAND ----------

# DBTITLE 1,ECM parameter tables — per segment, and aggregate vs segment median
for cond, blob in ECM_ALL.items():
    seg_fits, agg = blob['segments'], blob['aggregate']
    if not seg_fits:
        print(f"  {cond}: no segment fits")
        continue
 
    rows = []
    for s in sorted(seg_fits):
        r = seg_fits[s]
        p = r['params']
        se = r['stderr']
        d = {'segment': s, 'n_arcs': r['n_arcs'], 'chi2_nu': r['chi2_nu'],
             'verdict': r['verdict'],
             'Rs_mohm_cm2': p['Rs'] * 1000, 'Rs_se': se['Rs'] * 1000,
             'L_mohm_cm2_s': p['L'],
             'R_ct_mohm_cm2': r['R_ct'] * 1000,
             'R_mt_mohm_cm2': r['R_mt'] * 1000,
             'R_pol_mohm_cm2': r['R_pol'] * 1000,
             'tau_peak_ms': r['tau_peak'] * 1e3,
             'max_res_pct': float(r['res_pct'].max())}
        for k in range(r['n_arcs']):
            d[f'R{k+1}_mohm_cm2'] = p[f'R{k+1}'] * 1000
            d[f'tau{k+1}_ms'] = p[f'tau{k+1}'] * 1e3
            d[f'n{k+1}'] = p[f'n{k+1}']
        rows.append(d)
 
    df_p = pd.DataFrame(rows)
    print(f"\n{'═'*78}")
    print(f"  ECM PARAMETERS — Leepa {LEEPA}, {cond}   ({len(df_p)} segments)")
    print(f"  Rs + jwL + sum_k R_k/(1+(jw·τ_k)^n_k),  arc count by AICc, σ-weighted")
    print(f"{'═'*78}")
    for c, lbl in (('Rs_mohm_cm2', 'Rs    '), ('R_ct_mohm_cm2', 'R_ct  '),
                   ('R_mt_mohm_cm2', 'R_mt  '), ('R_pol_mohm_cm2', 'R_pol ')):
        v = df_p[c]
        print(f"  {lbl} median {v.median():8.2f}   IQR "
              f"{v.quantile(.25):7.2f}–{v.quantile(.75):7.2f}   mΩ·cm²")
    print(f"  χ²ν    median {df_p['chi2_nu'].median():.2f}   "
          f"good/underfit/overfit = "
          f"{(df_p.verdict=='good').sum()}/{(df_p.verdict=='underfit').sum()}/"
          f"{(df_p.verdict.str.startswith('overfit')).sum()}")
    print(f"  worst residual: {df_p['max_res_pct'].max():.2f} % "
          f"(segment {int(df_p.loc[df_p['max_res_pct'].idxmax(), 'segment'])})")
 
    if agg is not None:
        print(f"\n  {'─'*74}")
        print(f"  {'':22s}{'aggregate fit':>16s}{'segment median':>18s}")
        for lbl, ak, ck in (('Rs   [mΩ·cm²]', 'Rs', 'Rs_mohm_cm2'),
                            ('R_ct [mΩ·cm²]', None, 'R_ct_mohm_cm2'),
                            ('R_pol[mΩ·cm²]', None, 'R_pol_mohm_cm2')):
            av = (agg['params']['Rs'] * 1000 if ak == 'Rs'
                  else agg['R_ct'] * 1000 if ck == 'R_ct_mohm_cm2'
                  else agg['R_pol'] * 1000)
            print(f"  {lbl:22s}{av:16.2f}{df_p[ck].median():18.2f}")
        print(f"  {'χ²ν':22s}{agg['chi2_nu']:16.2f}{df_p['chi2_nu'].median():18.2f}")
        print("  The aggregate is the parallel sum, not the mean: a spread in Rs")
        print("  pulls it BELOW the segment median.  A large gap is a real")
        print("  statement about heterogeneity, not a fitting error.")
 
    display(df_p.round(4))

# COMMAND ----------

# DBTITLE 1,ECM parameter plate maps (Rs, R_ct, R_pol from the fit above)
# ═══════════════════════════════════════════════════════════════════════════════
# The same ECM parameters, laid out on the plate.  Driven off ECM_ALL, so the
# maps and the Nyquist overlays can never disagree about what was fitted.
#
# Drawn with plate_maps: the real segment outlines, the fitted VALUE printed
# inside every segment, and the same 'magma' ramp over the 5th..95th
# percentile as the pipeline's own maps above -- so an ECM Rs map and a
# pipeline HFR map can be read side by side. (This cell used to draw one dot
# per segment centroid on an RdYlGn ramp, which showed neither the segment
# shapes nor the numbers.)
# ═══════════════════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
 
for cond, blob in ECM_ALL.items():
    seg_fits = blob['segments']
    if not seg_fits:
        continue
 
    _maps = [
        ('Rs (ECM)',    {s: r['params']['Rs'] * 1000 for s, r in seg_fits.items()}),
        ('R_ct (ECM)',  {s: r['R_ct'] * 1000        for s, r in seg_fits.items()}),
        ('R_pol (ECM)', {s: r['R_pol'] * 1000       for s, r in seg_fits.items()}),
    ]
 
    for param_name, param_map in _maps:
        param_map = {str(k): v for k, v in param_map.items()
                     if str(k) in geom.SEGMENTS and np.isfinite(v)}
        if len(param_map) < 3:
            print(f"  {cond}: too few segments with coordinates for {param_name}")
            continue
        fig_h, ax = plate_maps.draw_value_map(
            param_map, label=param_name, unit='mΩ·cm²', cmap='magma',
            decimals=1,
            title=f'{param_name} — Leepa {LEEPA}, {cond} '
                  f'({len(param_map)} segments)')
        display(fig_h)
        plt.close(fig_h)
 
    print(f"  {cond}: ECM plate maps rendered")

# COMMAND ----------

# DBTITLE 1,Summary Table (plate_summary.csv)
import json
 
print(f"{'═'*75}")
print(f"  PIPELINE SUMMARY — Leepa {LEEPA}")
print(f"{'═'*75}\n")
 
for cond, pr in PIPELINE_RESULTS.items():
    if pr is None:
        print(f"  {cond}: FAILED")
        continue
    
    out_dir = pr['out_dir']
    _manifest_path = out_dir / 'run_manifest.json'
    
    if _manifest_path.exists():
        with open(_manifest_path) as f:
            _m = json.load(f)
        gs = _m.get('stages', {}).get('gold', {})
        elapsed = _m.get('elapsed_s', 0)
        print(f"  {cond}:  plate {_m.get('plate', '?')}, "
              f"source {_m.get('source', 'famos')}")
        if gs:
            print(f"    Measured:  {gs.get('n_measured', '?')}/"
                  f"{gs.get('n_total', '?')} segments")
            print(f"    Inferred:  {gs.get('n_inferred', '?')}")
            if 'R_ohmic' in gs:
                r = gs['R_ohmic']
                print(f"    R_ohmic:   {1000*r['mean']:.1f} ± {1000*r['sd']:.1f} mΩ·cm² (spread {r['spread']:.2f}x)")
        else:
            print(f"    Segments:  {_m.get('n_segments', '?')} with a spectrum")
            print(f"    ECM:       {_m.get('n_ecm_ok', '?')} converged")
            print(f"    lin-KK:    {_m.get('kk_pass', '?')}/{_m.get('kk_total', '?')} pass")
            _sch = _m.get('schedule', {})
            if _sch:
                print(f"    Excitation:{_sch.get('mode', '?')}, "
                      f"{len(_sch.get('tones', []))} tones")
        print(f"    Time:      {elapsed:.1f} s")
    
    # Show plate_summary.csv if available
    _summary_path = maps_dir(out_dir) / 'plate_summary.csv'
    if _summary_path.exists():
        df_s = pd.read_csv(_summary_path)
        print(f"    Segments:  {len(df_s)} in summary table")
    print()

# COMMAND ----------

# DBTITLE 1,Gamry Validation: whole-cell reference vs pipeline aggregate (from datago)
# ═══════════════════════════════════════════════════════════════════════════════
# GAMRY VALIDATION (from datago)
#
# WHAT IS COMPARED
#   The Gamry measures the WHOLE cell with one pair of leads.  The right
#   pipeline-side counterpart is silver/cell_aggregate.csv, which combines the
#   66 segments by the parallel rule
#         Z_cell = A_cell / sum_s ( A_s / Z_s )
#   Comparing a SINGLE segment against the Gamry is meaningless: a segment is
#   0.7-8.5 cm2 of a 304.92 cm2 cell.
#
# UNITS
#   Gamry writes Z in OHMS for the whole cell:
#         Z[mOhm*cm2] = Z[ohm] * A_CELL_CM2 * 1000
#   The pipeline is ALREADY in Ohm*cm2, because the Abgleich returns a current
#   DENSITY (K has units V/(A/cm^2)).  Do not apply an area factor twice.
#
# DATA SOURCE
#   datago tables:
#     ps_xplatform_dev.rvadvtec_ops.datago_advtec_metadata
#     ps_xplatform_dev.rvadvtec_ops.datago_advtec_generalproperties
#     ps_xplatform_dev.rvadvtec_ops.datago_advtec_values_delta
# ═══════════════════════════════════════════════════════════════════════════════
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
 
try:
    A_CELL_CM2 = float(geom.A_CELL_CM2)
except NameError:
    A_CELL_CM2 = 304.92
 
# ─── Datago tables ───
_META_TBL = 'ps_xplatform_dev.rvadvtec_ops.datago_advtec_metadata'
_GP_TBL = 'ps_xplatform_dev.rvadvtec_ops.datago_advtec_generalproperties'
_VAL_TBL = 'ps_xplatform_dev.rvadvtec_ops.datago_advtec_values_delta'
 
# ─── Ensure LEEPA is available ───
try:
    LEEPA
except NameError:
    LEEPA = dbutils.widgets.get('leepa_id')
 
# ─── Find Gamry EIS file_ids for this Leepa ───
print(f"  Querying datago for Gamry GALVEIS measurements (Leepa {LEEPA})...")
 
_gamry_df = spark.sql(f"""
    SELECT gp.file_id, m.orderId, m.measurement_id,
           gp.measurement_type, gp.measurementBegin
    FROM {_META_TBL} m
    JOIN {_GP_TBL} gp ON m.measurement_id = gp.measurement_id
    WHERE gp.measurement_type = 'GALVEIS'
      AND m.orderId = 'RO{LEEPA}'
    ORDER BY gp.measurementBegin
""").toPandas()
 
print(f"  Found {len(_gamry_df)} Gamry measurement(s) for Leepa {LEEPA}")
if not _gamry_df.empty:
    for _, row in _gamry_df.iterrows():
        print(f"    file_id={row['file_id'][:12]}...  begin={row['measurementBegin']}")
 
 
def _load_gamry_from_datago(file_id, area_cm2=A_CELL_CM2):
    """Load Gamry EIS spectrum from datago values table.
    
    Channels: freq [Hz], zreal [Ω], zimag [Ω], zmod [Ω], zphz [deg], idc [A], vdc [V]
    Returns DataFrame with Z in mΩ·cm².
    """
    df = spark.sql(f"""
        SELECT channel, CAST(time_value AS DOUBLE) AS pt,
               CAST(value AS DOUBLE) AS value
        FROM {_VAL_TBL}
        WHERE file_id = '{file_id}'
          AND channel IN ('freq', 'zreal', 'zimag', 'zmod', 'zphz', 'idc', 'vdc')
    """).toPandas()
    
    if df.empty:
        raise ValueError(f"No data found for file_id={file_id}")
    
    # Pivot: rows = point index (pt), columns = channel
    pivot = df.pivot_table(index='pt', columns='channel', values='value', aggfunc='first')
    pivot = pivot.sort_values('freq').reset_index(drop=True)
    
    # Convert Ω → mΩ·cm²
    k = area_cm2 * 1000.0
    out = pd.DataFrame({
        'freq_hz': pivot['freq'].values,
        'z_re': pivot['zreal'].values * k,
        'z_im': pivot['zimag'].values * k,
        'z_mod': pivot['zmod'].values * k if 'zmod' in pivot else np.abs(pivot['zreal'] + 1j * pivot['zimag']) * k,
        'z_phz_deg': pivot['zphz'].values if 'zphz' in pivot else np.degrees(np.angle(pivot['zreal'] + 1j * pivot['zimag'])),
        'i_dc': pivot['idc'].values if 'idc' in pivot else np.nan,
        'v_dc': pivot['vdc'].values if 'vdc' in pivot else np.nan,
    })
    return out.dropna(subset=['freq_hz', 'z_re', 'z_im']).sort_values('freq_hz').reset_index(drop=True)
 
 
def _gamry_r_ohmic(g):
    """HF intercept by interpolating the Zimag = 0 crossing -> (R, f)."""
    d = g.sort_values('freq_hz', ascending=False).reset_index(drop=True)
    zi, zr, f = d.z_im.values, d.z_re.values, d.freq_hz.values
    k = np.where(np.diff(np.sign(zi)))[0]
    if not len(k):
        return float('nan'), float('nan')
    j = k[0]
    t = -zi[j] / (zi[j + 1] - zi[j])
    return zr[j] + t * (zr[j + 1] - zr[j]), f[j] * (f[j + 1] / f[j]) ** t
 
 
def _load_pipeline(out_dir):
    """silver/cell_aggregate.csv -> freq_hz, z_re, z_im in mΩ·cm²."""
    p = Path(out_dir) / 'silver' / 'cell_aggregate.csv'
    if not p.exists():
        return None
    d = pd.read_csv(p)
    out = pd.DataFrame({'freq_hz': d.freq_hz,
                        'z_re': d.z_re_mohm_cm2,
                        'z_im': d.z_im_mohm_cm2})
    if np.nanmedian(out.z_im) > 0:        # normalise to Gamry sign convention
        out['z_im'] = -out['z_im']
    return out.sort_values('freq_hz').reset_index(drop=True)
 
 
def _pipeline_rs(out_dir):
    """Mean/sd of the per-segment R_ohmic, in mOhm.cm2.
 
    Two things were wrong here and they cancelled into a number that looked
    like an answer.
 
    1. The FAMOS path writes a `class` column ('measured'/'inferred');
       csv_pipeline writes integer `measured`/`inferred` columns instead.
       Selecting on `class` raised KeyError on every CSV run.
    2. plate_summary.csv holds R_ohmic in OHM.cm2 (gold.py's unit), while
       everything it is compared against on this plot -- the Gamry trace, the
       aggregate, rs_g -- is in mOHM.cm2.  Returning it unconverted made
       R_ohmic_dev_pct read about -99.9 % on every condition.
    """
    p = maps_dir(out_dir) / 'plate_summary.csv'
    if not p.exists():
        return (float('nan'), float('nan'))
    d = pd.read_csv(p)
    if 'class' in d.columns:
        m = d[d['class'] == 'measured']
    elif 'measured' in d.columns:
        m = d[d['measured'].astype(float) > 0]
    else:
        m = d
    r = pd.to_numeric(m['R_ohmic'], errors='coerce').dropna()
    if r.empty:
        return (float('nan'), float('nan'))
    return float(r.mean()) * 1000.0, float(r.std()) * 1000.0
 
 
# ─── Load Gamry spectra from datago (one per file_id) ───
GAMRY_SPECTRA = {}  # {condition_label: DataFrame}
 
if _gamry_df.empty:
    print(f"  No Gamry data in datago for Leepa {LEEPA}.")
    print(f"  Available via local path? Check /Workspace/Users/uum5fe@bosch.com/gamry/")
else:
    for idx, row in _gamry_df.iterrows():
        file_id = row['file_id']
        try:
            gspec = _load_gamry_from_datago(file_id)
            # Try to determine condition from Vdc or current (heuristic)
            vdc_mean = gspec['v_dc'].mean() if 'v_dc' in gspec else np.nan
            idc_mean = gspec['i_dc'].mean() if 'i_dc' in gspec else np.nan
            label = f"Cond-{idx+1}"
            if np.isfinite(idc_mean) and abs(idc_mean) > 1:
                # Map DC current to condition label
                for c in ['45A', '60A', '150A', '450A']:
                    if abs(idc_mean - float(c.replace('A', ''))) < float(c.replace('A', '')) * 0.3:
                        label = c
                        break
            GAMRY_SPECTRA[label] = gspec
            print(f"    Loaded {label}: {len(gspec)} pts, "
                  f"{gspec.freq_hz.min():.2f}–{gspec.freq_hz.max():.0f} Hz, "
                  f"Idc={idc_mean:.1f}A, Vdc={vdc_mean:.3f}V")
        except Exception as e:
            print(f"    file_id={file_id[:12]}... FAILED: {e}")
 
print(f"\n  Gamry conditions loaded: {sorted(GAMRY_SPECTRA.keys()) or 'NONE'}")
 
# ─── Match pipeline conditions to Gamry conditions ───
try:
    _pipeline_avail = PIPELINE_RESULTS
except NameError:
    _pipeline_avail = {}
    print("  PIPELINE_RESULTS not available (run Cell 3 first for comparison).")
 
_matched = [c for c in _pipeline_avail
            if _pipeline_avail.get(c) and c in GAMRY_SPECTRA]
print(f"  Conditions with both pipeline + Gamry: {_matched}")
 
if not _matched and GAMRY_SPECTRA and _pipeline_avail:
    # If condition labels don't match exactly, try pairing by order
    print("  (No exact condition match — showing all Gamry spectra for visual comparison)")
    _matched_fallback = True
else:
    _matched_fallback = False
 
# ─── Compare + plot ───
VALIDATION = {}
_to_plot = _matched if _matched else []
 
for cond in _to_plot:
    g = GAMRY_SPECTRA[cond]
    p = _load_pipeline(_pipeline_avail[cond]['out_dir'])
    if p is None or len(p) < 3:
        print(f'  {cond}: no cell_aggregate.csv — run at least the silver stage')
        continue
    rs_p, sd_p = _pipeline_rs(_pipeline_avail[cond]['out_dir'])
    rs_g, f_x = _gamry_r_ohmic(g)
 
    # Agreement in the OVERLAP band only
    lo = max(g.freq_hz.min(), p.freq_hz.min())
    hi = min(g.freq_hz.max(), p.freq_hz.max())
    m = (p.freq_hz >= lo) & (p.freq_hz <= hi)
    lg = np.log10(g.freq_hz.values)
    gr = np.interp(np.log10(p.freq_hz[m]), lg, g.z_re.values)
    gi = np.interp(np.log10(p.freq_hz[m]), lg, g.z_im.values)
    zp = p.z_re[m].values + 1j * p.z_im[m].values
    zg = gr + 1j * gi
    rel = 100.0 * np.abs(zp - zg) / np.abs(zg)
 
    VALIDATION[cond] = dict(
        n_overlap=int(m.sum()), f_lo_hz=lo, f_hi_hz=hi,
        median_dev_pct=float(np.median(rel)), p95_dev_pct=float(np.percentile(rel, 95)),
        gamry_R_ohmic=rs_g, gamry_crossing_hz=f_x,
        pipeline_R_ohmic=rs_p, pipeline_R_ohmic_sd=sd_p,
        R_ohmic_dev_pct=100.0 * (rs_p - rs_g) / rs_g if rs_g != 0 else np.nan,
        pipeline_f_max_hz=float(p.freq_hz.max()),
        crossing_covered=bool(f_x <= p.freq_hz.max()),
    )
 
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=['Nyquist', '|Z|(f) Bode', 'Phase(f)'],
                        horizontal_spacing=0.07)
    GA, PI = '#2b2b2b', '#c0392b'
 
    def _add(col, x, y, name, color, dash=None, mode='lines+markers', ht=''):
        fig.add_trace(go.Scatter(
            x=x, y=y, mode=mode, name=name, legendgroup=name,
            showlegend=(col == 1),
            line=dict(width=1.8, color=color, dash=dash),
            marker=dict(size=4, color=color), hovertemplate=ht), row=1, col=col)
 
    _add(1, g.z_re, -g.z_im, 'Gamry (whole cell)', GA,
         ht="Gamry<br>Z'=%{x:.1f}<br>-Z''=%{y:.1f} mΩ·cm²<extra></extra>")
    _add(1, p.z_re, -p.z_im, 'Pipeline (66-seg aggregate)', PI,
         ht="Pipeline<br>Z'=%{x:.1f}<br>-Z''=%{y:.1f} mΩ·cm²<extra></extra>")
    fig.add_trace(go.Scatter(x=[rs_g], y=[0], mode='markers',
                             marker=dict(symbol='star', size=15, color='#1f77b4'),
                             name=f'Gamry R_Ω = {rs_g:.1f}', legendgroup='rg'),
                  row=1, col=1)
    if np.isfinite(rs_p):
        fig.add_trace(go.Scatter(x=[rs_p], y=[0], mode='markers',
                                 marker=dict(symbol='star', size=15, color=PI),
                                 name=f'Pipeline R_Ω = {rs_p:.1f}', legendgroup='rp'),
                      row=1, col=1)
 
    for col, yg, yp, lbl in (
            (2, np.abs(g.z_re + 1j * g.z_im), np.abs(p.z_re + 1j * p.z_im), '|Z|'),
            (3, np.degrees(np.angle(g.z_re + 1j * g.z_im)),
                np.degrees(np.angle(p.z_re + 1j * p.z_im)), 'Phase')):
        _add(col, g.freq_hz, yg, 'Gamry (whole cell)', GA, mode='lines',
             ht=f'Gamry<br>f=%{{x:.2f}} Hz<br>{lbl}=%{{y:.1f}}<extra></extra>')
        _add(col, p.freq_hz, yp, 'Pipeline (66-seg aggregate)', PI,
             ht=f'Pipeline<br>f=%{{x:.2f}} Hz<br>{lbl}=%{{y:.1f}}<extra></extra>')
        # Shade the band the pipeline never sees
        fig.add_vrect(x0=float(p.freq_hz.max()), x1=float(g.freq_hz.max()),
                      fillcolor=PI, opacity=0.07, line_width=0, row=1, col=col)
        if np.isfinite(f_x):
            fig.add_vline(x=f_x, line=dict(color='#1f77b4', width=1.2, dash='dash'),
                          row=1, col=col)
 
    fig.update_xaxes(title_text="Z' [mΩ·cm²]", row=1, col=1)
    fig.update_yaxes(title_text="-Z'' [mΩ·cm²]", row=1, col=1)
    for c, t, ty in ((2, '|Z| [mΩ·cm²]', 'log'), (3, 'Phase [°]', 'linear')):
        fig.update_xaxes(title_text='f [Hz]', type='log', row=1, col=c)
        fig.update_yaxes(title_text=t, type=ty, row=1, col=c)
 
    v = VALIDATION[cond]
    fig.update_layout(
        title=(f'<b>Gamry validation — Leepa {LEEPA}, {cond}</b>'
               f'<br><sup>overlap {v["f_lo_hz"]:.2f}–{v["f_hi_hz"]:.0f} Hz, '
               f'{v["n_overlap"]} pts · median |ΔZ| = {v["median_dev_pct"]:.2f} % · '
               f'R_Ω dev {v["R_ohmic_dev_pct"]:+.1f} % · '
               f'shaded = band the pipeline never sees, dashed = arc closes</sup>'),
        height=560, width=1500, paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(t=105, b=55, r=260), hovermode='closest',
        legend=dict(font=dict(size=10), y=0.5, yanchor='middle'))
    show_fig(fig, html=True)   # one plot per figure, stacked
 
# ─── Summary table ───
if VALIDATION:
    _sum = pd.DataFrame(VALIDATION).T
    _sum.index.name = 'condition'
    display(_sum[['n_overlap', 'median_dev_pct', 'p95_dev_pct',
                  'gamry_R_ohmic', 'pipeline_R_ohmic', 'R_ohmic_dev_pct',
                  'gamry_crossing_hz', 'pipeline_f_max_hz', 'crossing_covered']]
            .round(2))
    print('\n  median_dev_pct   — spectrum agreement inside the shared band.')
    print('  R_ohmic_dev_pct  — HF intercept agreement.  Expect this to be large')
    print('                     wherever crossing_covered is False: the Gamry sees')
    print('                     the Zimag=0 crossing and the pipeline extrapolates')
    print('                     to it from below.')
elif GAMRY_SPECTRA and not _matched:
    # Show Gamry spectra standalone (no pipeline match)
    print('\n  No matching pipeline condition — showing Gamry spectra standalone:')
    for label, g in GAMRY_SPECTRA.items():
        rs_g, f_x = _gamry_r_ohmic(g)
        rs_str = f'{rs_g:.1f}' if np.isfinite(rs_g) else 'N/A'
        fx_str = f'{f_x:.0f}' if np.isfinite(f_x) else 'N/A'
        print(f'    {label}: Rs={rs_str} mΩ·cm² @ {fx_str} Hz, '
              f'{len(g)} pts ({g.freq_hz.min():.2f}–{g.freq_hz.max():.0f} Hz)')
else:
    print('  Nothing to compare — no Gamry data found in datago for this Leepa.')

# COMMAND ----------

# DBTITLE 1,Lin-KK: Linear Kramers-Kronig Validation
# ═══════════════════════════════════════════════════════════════════════════════
# LINEAR KRAMERS-KRONIG VALIDATION (Boukamp 1995 / Schoenleber 2014)
#
# Gate test: is this spectrum causal, linear, time-invariant?  A spectrum that
# fails KK cannot be interpreted by an ECM or DRT.  Uses a Voigt ladder with
# FIXED time constants so the fit is LINEAR in R_k; the residual measures ONLY
# the data's departure from KK compliance, never the model flexibility.
#
# The M (number of RC elements) is chosen by the Schoenleber over-fitting
# criterion: mu = 1 - sum(R_k<0)/sum(R_k>0) drops below 0.85 once over-fitting.
# ═══════════════════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
 
# ─── Lin-KK Implementation ───
def _voigt_basis(freq, taus):
    """Complex design matrix for Voigt ladder with fixed time constants."""
    w = 2 * np.pi * freq[:, None]
    return 1.0 / (1.0 + 1j * w * taus[None, :])
 
 
def lin_kk(freq, Z, M=None, add_inductance=True, c_crit=0.85):
    """Linear KK test. Returns dict with residuals and fitted model.
    
    Parameters
    ----------
    freq : Hz, ascending
    Z    : complex impedance (any unit)
    M    : number of RC elements (None = auto via Schoenleber criterion)
    add_inductance : include series jwL term (needed for fuel-cell data)
    c_crit : mu threshold for over-fitting detection (default 0.85)
    """
    freq = np.asarray(freq, float)
    Z = np.asarray(Z, complex)
    n = len(freq)
 
    def _fit(M):
        taus = np.logspace(np.log10(1 / (2 * np.pi * freq.max())),
                           np.log10(1 / (2 * np.pi * freq.min())), M)
        A = _voigt_basis(freq, taus)
        cols = [A.real, np.ones((n, 1))]
        colsi = [A.imag, np.zeros((n, 1))]
        if add_inductance:
            cols.append(np.zeros((n, 1)))
            colsi.append((2 * np.pi * freq)[:, None])
        X = np.vstack([np.hstack(cols), np.hstack(colsi)])
        y = np.concatenate([Z.real, Z.imag])
        wgt = np.concatenate([1.0 / np.abs(Z), 1.0 / np.abs(Z)])
        p, *_ = np.linalg.lstsq(X * wgt[:, None], y * wgt, rcond=None)
        Zf = X @ p
        Zfit = Zf[:n] + 1j * Zf[n:]
        Rk = p[:M]
        pos = Rk[Rk > 0].sum()
        neg = -Rk[Rk < 0].sum()
        mu = 1.0 - neg / pos if pos > 0 else 0.0
        return p, Zfit, taus, mu
 
    if M is None:
        M_grid = list(range(3, min(n - 2, 60)))
        mus = []
        fits = {}
        for m in M_grid:
            fits[m] = _fit(m)
            mus.append(fits[m][3])
        mus = np.asarray(mus)
        # Find peak mu then first drop below c_crit
        start = max(0, int(np.argmax(np.where(np.arange(len(mus)) >= 2, mus, -np.inf))))
        chosen = None
        for i in range(start + 1, len(M_grid)):
            if mus[i] < c_crit:
                chosen = M_grid[i]
                break
        M = chosen if chosen else M_grid[min(start, len(M_grid) - 1)]
 
    p, Zfit, taus, mu = _fit(M)
    res_re = (Z.real - Zfit.real) / np.abs(Z)
    res_im = (Z.imag - Zfit.imag) / np.abs(Z)
    return {
        "M": M, "mu": mu, "taus": taus, "params": p, "Z_fit": Zfit,
        "res_re": res_re, "res_im": res_im,
        "res_re_pct": 100 * res_re, "res_im_pct": 100 * res_im,
        "chi2": float(np.sum(res_re**2 + res_im**2)),
        "max_abs_res_pct": float(100 * np.max(np.abs(res_re + 1j * res_im))),
        "rms_res_pct": float(100 * np.sqrt(np.mean(res_re**2 + res_im**2))),
    }
 
 
# ─── Run Lin-KK on silver spectra from pipeline results ───
KK_RESULTS = {}  # {cond: {seg: kk_dict}}
 
for cond, pr in PIPELINE_RESULTS.items():
    if pr is None:
        continue
    spectra_path = spectra_csv(pr['out_dir'])
    if not spectra_path.exists():
        print(f"  {cond}: no spectra_clean.csv")
        continue
 
    df = pd.read_csv(spectra_path)
    segments = sorted(df['segment'].unique())
    KK_RESULTS[cond] = {}
 
    for seg in segments:
        sd = df[df['segment'] == seg].sort_values('freq_hz')
        f = sd['freq_hz'].values
        zr = sd['z_re_mohm_cm2'].values
        zi = sd['z_im_mohm_cm2'].values
        Z = zr + 1j * zi
 
        # Basic quality filter
        keep = np.isfinite(zr) & np.isfinite(zi) & (zr > 0)
        if keep.sum() < 10:
            continue
        f, Z = f[keep], Z[keep]
 
        kk = lin_kk(f, Z)
        KK_RESULTS[cond][seg] = kk
 
    n_pass = sum(1 for v in KK_RESULTS[cond].values() if v['rms_res_pct'] < 2.0)
    print(f"  {cond}: {len(KK_RESULTS[cond])} segments tested, "
          f"{n_pass} pass (<2% RMS residual)")
 
# ─── Plot: KK diagnostic (Plotly interactive) ───
for cond, kk_cond in KK_RESULTS.items():
    if not kk_cond:
        continue
 
    all_segs = sorted(kk_cond.keys(), key=lambda x: int(x))
    step = max(1, len(all_segs) // 6)
    show_segs = all_segs[::step][:6]
    n_show = len(show_segs)
    n_cols = min(n_show, 3)
    n_row_groups = (n_show + n_cols - 1) // n_cols
    n_rows = 2 * n_row_groups
 
    titles = []
    for i, seg in enumerate(show_segs):
        kk = kk_cond[seg]
        titles.append(f'Seg {seg} Nyquist (M={kk["M"]}, μ={kk["mu"]:.2f})')
    for i, seg in enumerate(show_segs):
        titles.append(f'Seg {seg} Residuals (RMS={kk_cond[seg]["rms_res_pct"]:.2f}%)')
    # Interleave: row1=nyquist, row2=residual for each row group
    ordered_titles = []
    for rg in range(n_row_groups):
        for c in range(n_cols):
            idx = rg * n_cols + c
            ordered_titles.append(titles[idx] if idx < n_show else '')
        for c in range(n_cols):
            idx = rg * n_cols + c
            ordered_titles.append(titles[n_show + idx] if idx < n_show else '')
 
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=ordered_titles,
                        vertical_spacing=0.08, horizontal_spacing=0.06)
 
    spectra_path = spectra_csv(PIPELINE_RESULTS[cond]['out_dir'])
    df = pd.read_csv(spectra_path)
    _colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
 
    for idx, seg in enumerate(show_segs):
        rg = idx // n_cols
        col = idx % n_cols + 1
        row_nyq = rg * 2 + 1
        row_res = rg * 2 + 2
        kk = kk_cond[seg]
 
        sd = df[df['segment'] == seg].sort_values('freq_hz')
        f = sd['freq_hz'].values
        zr = sd['z_re_mohm_cm2'].values
        zi = sd['z_im_mohm_cm2'].values
        keep = np.isfinite(zr) & np.isfinite(zi) & (zr > 0)
        Z_data = (zr + 1j * zi)[keep]
        Zf = kk['Z_fit']
 
        # Nyquist
        fig.add_trace(go.Scatter(x=Z_data.real, y=-Z_data.imag, mode='markers',
            marker=dict(size=4, color=_colors[idx % 6], opacity=0.6),
            name=f'Data s{seg}', showlegend=(idx == 0),
            hovertemplate="Z'=%{x:.1f}<br>-Z''=%{y:.1f}<extra>Data</extra>"),
            row=row_nyq, col=col)
        fig.add_trace(go.Scatter(x=Zf.real, y=-Zf.imag, mode='lines',
            line=dict(color='red', width=2),
            name=f'KK fit', showlegend=(idx == 0),
            hovertemplate="Z'=%{x:.1f}<br>-Z''=%{y:.1f}<extra>KK fit</extra>"),
            row=row_nyq, col=col)
 
        # Residuals
        f_plot = f[keep]
        fig.add_trace(go.Scatter(x=f_plot, y=kk['res_re_pct'], mode='markers',
            marker=dict(size=3, color='#1f77b4'), name='ΔRe', showlegend=(idx == 0)),
            row=row_res, col=col)
        fig.add_trace(go.Scatter(x=f_plot, y=kk['res_im_pct'], mode='markers',
            marker=dict(size=3, color='#d62728'), name='ΔIm', showlegend=(idx == 0)),
            row=row_res, col=col)
        # ±2% threshold lines
        fig.add_hline(y=2, line=dict(color='grey', dash='dash', width=0.8), row=row_res, col=col)
        fig.add_hline(y=-2, line=dict(color='grey', dash='dash', width=0.8), row=row_res, col=col)
        fig.add_hline(y=0, line=dict(color='black', width=0.5), row=row_res, col=col)
        fig.update_xaxes(type='log', row=row_res, col=col)
        fig.update_yaxes(range=[-8, 8], row=row_res, col=col)
 
    fig.update_layout(title=f'<b>Lin-KK Validation — {cond} (Leepa {LEEPA})</b>',
                      height=450 * n_row_groups, width=1400,
                      paper_bgcolor='white', plot_bgcolor='white',
                      showlegend=True, hovermode='closest')
    fig.update_xaxes(showgrid=True, gridcolor='#eee')
    fig.update_yaxes(showgrid=True, gridcolor='#eee')
    show_fig(fig, html=True)   # one plot per figure, stacked

# COMMAND ----------

# DBTITLE 1,DRT: Distribution of Relaxation Times (Tikhonov)
# ═══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTION OF RELAXATION TIMES — Tikhonov Ridge Regression
#
# Deconvolves gamma(ln tau) from the impedance spectrum:
#   Z(f) = R_inf + jωL + ∫ gamma(ln τ) / (1 + jωτ) d(ln τ)
#
# The inversion is ill-posed (small data errors blow up in gamma), so a
# smoothness penalty λ·||γ||² is added.  Lambda is a genuine trade-off:
# too small → oscillations; too large → real peaks merge.
#
# WHY DRT: tells you HOW MANY processes exist and at WHAT timescales,
# before choosing an ECM model order.  Avoids the failure mode of copying
# a circuit from a paper without verification.
#
# References:
#   Wan, Saccoccio, Chen, Ciucci, Electrochim. Acta 184, 483 (2015)
#   Ivers-Tiffee & Weber, J. Ceram. Soc. Japan 125, 193 (2017)
# ═══════════════════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import nnls
from pathlib import Path
 
 
# ─── DRT Implementation ───
def _voigt_basis(freq, taus):
    w = 2 * np.pi * freq[:, None]
    return 1.0 / (1.0 + 1j * w * taus[None, :])
 
 
def drt_tikhonov(freq, Z, lam=1e-3, n_tau_per_decade=10, extend_decades=0.0,
                 use="imag", nonneg=True, fit_L=True, fit_Rinf=False):
    """Deconvolve gamma(ln tau) from impedance spectrum.
    
    Parameters
    ----------
    freq : Hz
    Z    : complex impedance
    lam  : regularization parameter (smoothness penalty)
    use  : 'imag' (default, least affected by unmodelled Rs), 'real', 'both'
    nonneg : enforce gamma >= 0 (physical for passive systems)
    fit_L  : include series inductance jωL
    """
    freq = np.asarray(freq, float)
    Z = np.asarray(Z, complex)
    w = 2 * np.pi * freq
 
    lo = np.log10(1 / (2 * np.pi * freq.max())) - extend_decades
    hi = np.log10(1 / (2 * np.pi * freq.min())) + extend_decades
    n_tau = int(np.ceil((hi - lo) * n_tau_per_decade)) + 1
    taus = np.logspace(lo, hi, n_tau)
    ln_tau = np.log(taus)
 
    A = _voigt_basis(freq, taus)
    d_ln_tau = np.mean(np.diff(ln_tau))
    A = A * d_ln_tau  # quadrature weight
 
    blocks_r, blocks_i = [A.real], [A.imag]
    extra = 0
    if fit_Rinf:
        blocks_r.append(np.ones((len(freq), 1)))
        blocks_i.append(np.zeros((len(freq), 1)))
        extra += 1
    if fit_L:
        blocks_r.append(np.zeros((len(freq), 1)))
        blocks_i.append(w[:, None])
        extra += 1
    Xr, Xi = np.hstack(blocks_r), np.hstack(blocks_i)
 
    if use == "imag":
        X, y = Xi, Z.imag
    elif use == "real":
        X, y = Xr, Z.real
    else:
        X, y = np.vstack([Xr, Xi]), np.concatenate([Z.real, Z.imag])
 
    # First-difference smoothness penalty on gamma only
    npar = X.shape[1]
    D = np.zeros((n_tau - 1, npar))
    for k in range(n_tau - 1):
        D[k, k], D[k, k + 1] = -1.0, 1.0
 
    # Scale penalty against data term
    s_dat = np.linalg.norm(X, 2)
    s_pen = np.linalg.norm(D, 2)
    scale = s_dat / max(s_pen, 1e-30)
    Xa = np.vstack([X, np.sqrt(lam) * scale * D])
    ya = np.concatenate([y, np.zeros(n_tau - 1)])
 
    if nonneg:
        if fit_L:
            Xa_ext = np.hstack([Xa, -Xa[:, -1:]])
            sol, _ = nnls(Xa_ext, ya)
            p = np.concatenate([sol[:npar - 1], [sol[npar - 1] - sol[npar]]])
        else:
            p, _ = nnls(Xa, ya)
    else:
        p, *_ = np.linalg.lstsq(Xa, ya, rcond=None)
 
    gamma = p[:n_tau]
    out = {"tau": taus, "gamma": gamma, "lam": lam,
           "R_pol": float(np.sum(gamma) * d_ln_tau),
           "Z_fit": (Xr @ p) + 1j * (Xi @ p)}
 
    if not fit_Rinf:
        out["R_inf"] = float(np.median(Z.real - (Xr @ p)))
    j = n_tau
    if fit_Rinf:
        out["R_inf"] = float(p[j]); j += 1
    if fit_L:
        out["L"] = float(p[j])
    return out
 
 
def drt_peaks(res, min_rel_height=0.05):
    """Find peak timescales and their areas (= resistance of each process)."""
    g, tau = res["gamma"], res["tau"]
    if not np.any(g > 0):
        return []
    thr = min_rel_height * g.max()
    peaks = []
    d_ln = np.mean(np.diff(np.log(tau)))
    for k in range(1, len(g) - 1):
        if g[k] > thr and g[k] >= g[k - 1] and g[k] > g[k + 1]:
            lo_idx = k
            while lo_idx > 0 and g[lo_idx - 1] < g[lo_idx]:
                lo_idx -= 1
            hi_idx = k
            while hi_idx < len(g) - 1 and g[hi_idx + 1] < g[hi_idx]:
                hi_idx += 1
            peaks.append({"tau": float(tau[k]),
                          "f_peak": float(1 / (2 * np.pi * tau[k])),
                          "gamma": float(g[k]),
                          "R": float(np.sum(g[lo_idx:hi_idx + 1]) * d_ln)})
    return sorted(peaks, key=lambda p: -p["f_peak"])
 
 
# ─── Run DRT on silver spectra ───
DRT_RESULTS = {}  # {cond: {seg: drt_dict}}
 
for cond, pr in PIPELINE_RESULTS.items():
    if pr is None:
        continue
    spectra_path = spectra_csv(pr['out_dir'])
    if not spectra_path.exists():
        continue
 
    df = pd.read_csv(spectra_path)
    segments = sorted(df['segment'].unique())
    DRT_RESULTS[cond] = {}
 
    for seg in segments:
        sd = df[df['segment'] == seg].sort_values('freq_hz')
        f = sd['freq_hz'].values
        zr = sd['z_re_mohm_cm2'].values
        zi = sd['z_im_mohm_cm2'].values
        Z = zr + 1j * zi
 
        keep = np.isfinite(zr) & np.isfinite(zi) & (zr > 0)
        if keep.sum() < 10:
            continue
        f, Z = f[keep], Z[keep]
 
        # Run DRT with multiple lambda to show sensitivity
        drt = drt_tikhonov(f, Z, lam=1e-3)
        peaks = drt_peaks(drt)
        drt['peaks'] = peaks
        DRT_RESULTS[cond][seg] = drt
 
    # Summary
    n_segs = len(DRT_RESULTS[cond])
    n_2peak = sum(1 for v in DRT_RESULTS[cond].values() if len(v.get('peaks', [])) >= 2)
    print(f"  {cond}: {n_segs} segments, {n_2peak} show ≥2 DRT peaks")
 
# ─── Plot: DRT gamma(tau) (Plotly interactive) ───
_drt_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
 
for cond, drt_cond in DRT_RESULTS.items():
    if not drt_cond:
        continue
 
    all_segs = sorted(drt_cond.keys(), key=lambda x: int(x))
    step = max(1, len(all_segs) // 6)
    show_segs = all_segs[::step][:6]
    n_show = len(show_segs)
    n_cols = min(n_show, 3)
    n_rows = (n_show + n_cols - 1) // n_cols
 
    titles = [f'Seg {s} — R_pol={drt_cond[s]["R_pol"]:.1f}, {len(drt_cond[s].get("peaks",[]))} peaks'
              for s in show_segs]
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles,
                        horizontal_spacing=0.06, vertical_spacing=0.12)
 
    for idx, seg in enumerate(show_segs):
        r, c = idx // n_cols + 1, idx % n_cols + 1
        drt = drt_cond[seg]
        tau_ms = drt['tau'] * 1e3
        gamma = drt['gamma']
        peaks = drt.get('peaks', [])
 
        # Filled area + line
        fig.add_trace(go.Scatter(x=tau_ms, y=gamma, mode='lines',
            fill='tozeroy', fillcolor=f'rgba(70,130,180,0.25)',
            line=dict(color='steelblue', width=2), name=f'Seg {seg}',
            showlegend=(idx == 0),
            hovertemplate='τ=%{x:.3f} ms<br>γ=%{y:.2f}<extra></extra>'),
            row=r, col=c)
 
        # Peak markers
        for pk in peaks:
            fig.add_trace(go.Scatter(x=[pk['tau']*1e3], y=[pk['gamma']],
                mode='markers+text', marker=dict(size=8, color='red', symbol='diamond'),
                text=[f"{pk['f_peak']:.0f} Hz"], textposition='top center',
                textfont=dict(size=9, color='red'), showlegend=False),
                row=r, col=c)
 
        fig.update_xaxes(type='log', title_text='τ [ms]', row=r, col=c)
        fig.update_yaxes(title_text='γ(ln τ) [mΩ·cm²]', row=r, col=c)
 
    fig.update_layout(title=f'<b>DRT — Distribution of Relaxation Times — {cond} (Leepa {LEEPA})</b>',
                      height=350 * n_rows, width=1400,
                      paper_bgcolor='white', plot_bgcolor='white', hovermode='closest')
    fig.update_xaxes(showgrid=True, gridcolor='#eee')
    fig.update_yaxes(showgrid=True, gridcolor='#eee')
    show_fig(fig, html=True)   # one plot per figure, stacked
 
# ─── Lambda sensitivity plot (Plotly interactive) ───
for cond, drt_cond in DRT_RESULTS.items():
    if not drt_cond:
        continue
    demo_seg = None
    for seg in sorted(drt_cond.keys(), key=lambda x: int(x)):
        if len(drt_cond[seg].get('peaks', [])) >= 2:
            demo_seg = seg
            break
    if demo_seg is None:
        demo_seg = sorted(drt_cond.keys(), key=lambda x: int(x))[0]
 
    spectra_path = spectra_csv(PIPELINE_RESULTS[cond]['out_dir'])
    df = pd.read_csv(spectra_path)
    sd = df[df['segment'] == demo_seg].sort_values('freq_hz')
    f = sd['freq_hz'].values
    zr = sd['z_re_mohm_cm2'].values
    zi = sd['z_im_mohm_cm2'].values
    Z = (zr + 1j * zi)
    keep = np.isfinite(zr) & np.isfinite(zi) & (zr > 0)
    f, Z = f[keep], Z[keep]
 
    lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    fig = go.Figure()
    for lam in lambdas:
        d = drt_tikhonov(f, Z, lam=lam)
        fig.add_trace(go.Scatter(x=d['tau'] * 1e3, y=d['gamma'], mode='lines',
            name=f'λ={lam:.0e}', hovertemplate='τ=%{x:.3f} ms<br>γ=%{y:.2f}<extra></extra>'))
 
    fig.update_xaxes(type='log', title_text='τ [ms]')
    fig.update_yaxes(title_text='γ(ln τ) [mΩ·cm²]')
    fig.update_layout(title=f'<b>DRT λ Sensitivity — {cond}, Seg {demo_seg} (Leepa {LEEPA})</b>',
                      height=450, width=900,
                      paper_bgcolor='white', plot_bgcolor='white',
                      hovermode='x unified')
    fig.update_xaxes(showgrid=True, gridcolor='#eee')
    fig.update_yaxes(showgrid=True, gridcolor='#eee')
    show_fig(fig, html=True)   # one plot per figure, stacked
    break  # only show for first condition

# COMMAND ----------

# DBTITLE 1,ECM Fit: L + Rs + (Rct1||CPE1) + (Rct2||CPE2) — DRT-informed


# COMMAND ----------

# DBTITLE 1,ECM Fit: L + Rs + (Rct1||CPE1) + (Rct2||CPE2) — DRT-informed
 
# ═══════════════════════════════════════════════════════════════════════════════
# ECM FIT: DRT-INFORMED EQUIVALENT CIRCUIT MODEL
#
# Model: Z(f) = jωL + Rs + Rct1/(1 + Rct1·Y01·(jω)^n1) + Rct2/(1 + Rct2·Y02·(jω)^n2)
#
# Starting values are derived from the DRT peaks (Metrohm AN-EIS-007 recipe):
#   Rs  = HF intercept (min Z')
#   Rct = area under DRT peak (= process resistance)
#   tau = peak position → C = tau/R, then Y0 = C^n / R^(1-n)
#
# This avoids arbitrary initial guesses that cause convergence failures.
#
# References:
#   Metrohm Application Note EIS-007
#   Liu & Ciucci, Electrochim. Acta 331, 135316 (2020)
# ═══════════════════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import least_squares
from pathlib import Path
 
 
# ─── ECM Model Functions ───
# PARAMETERISED BY tau, NOT BY Y0.
#
# The circuit is the same either way -- a ZARC is R in parallel with a CPE
# whichever coordinate you name it in, and the two forms are related by
#
#       Y0 = tau^n / R          equivalently    tau = (R * Y0)^(1/n)
#
# but the two forms do not FIT the same.  In (R, Y0, n) coordinates the three
# parameters of an arc trade against each other over six decades: shrink R,
# grow Y0, and the arc barely moves.  least_squares walks that curved valley,
# stops when the step gets small, and returns success=True at a point where
# R1 is wrong by two orders of magnitude and n1 has been pushed onto its
# bound.  The Nyquist overlay still looks fine, because the SUM of the arcs
# is well determined even when the split between them is not -- which is
# exactly what makes the failure so easy to miss.
#
# In (R, tau, n) coordinates each arc has a location (tau), a size (R) and a
# shape (n), those three are close to orthogonal, and two arcs whose taus
# differ by decades actually separate.  This also lets the DRT peaks feed
# straight in: a DRT peak IS a (R, tau) pair, so no conversion is needed and
# there is no formula left to get wrong.
#
# THE BUG THIS REPLACES: the previous version set Y1 = 1/(R*tau) instead of
# tau^n/R.  For R = 25 mohm.cm2 and tau = 0.3 ms that is 133 instead of
# 2.7e-05 -- wrong by a factor of 5 million -- so every fit started far
# outside the physical region and converged to a false optimum while
# reporting success.
def z_zarc(w, R, tau, n):
    """One ZARC: Z = R / (1 + (j*w*tau)^n)."""
    return R / (1.0 + (1j * w * tau) ** n)
 
 
def z_model(p, freq):
    """L + Rs + (Rct1 || CPE1) + (Rct2 || CPE2), in (R, tau, n) coordinates."""
    L, Rs, R1, tau1, n1, R2, tau2, n2 = p
    w = 2 * np.pi * np.asarray(freq, float)
    return 1j * w * L + Rs + z_zarc(w, R1, tau1, n1) + z_zarc(w, R2, tau2, n2)
 
 
def y0_from_tau(R, tau, n):
    """Report the CPE admittance for anyone who wants the Y0 form."""
    return tau ** n / R if R > 0 else float("nan")
 
 
def ecm_starting_values(freq, Z, drt_res=None):
    """DRT-informed starting values, as (L, Rs, R1, tau1, n1, R2, tau2, n2)."""
    freq = np.asarray(freq, float)
    Z = np.asarray(Z, complex)
    o = np.argsort(freq)
    f, Zs = freq[o], Z[o]
    # Rs is the HIGH-FREQUENCY real part, not min(Re Z).  With an inductive
    # tail the smallest Re Z can sit anywhere in the sweep, and on a noisy
    # spectrum min() picks the noisiest point by construction.
    Rs = max(float(Zs[-1].real), 1e-6)
    R_tot = max(float(Zs[0].real) - Rs, 1e-5)
 
    if drt_res is not None:
        pk = drt_peaks(drt_res)[:2]
        if len(pk) == 2:
            hi_f, lo_f = pk[0], pk[1]          # drt_peaks sorts by -f_peak
            return np.array([1e-9, Rs,
                             max(hi_f["R"], 1e-3), max(hi_f["tau"], 1e-6), 0.9,
                             max(lo_f["R"], 1e-3), max(lo_f["tau"], 1e-6), 0.9])
        if len(pk) == 1:
            t = max(pk[0]["tau"], 1e-6)
            return np.array([1e-9, Rs, 0.5 * R_tot, t / 3.0, 0.9,
                             0.5 * R_tot, t * 3.0, 0.9])
 
    # Fallback: no DRT.  Put the two taus a decade either side of where
    # -Im Z peaks, which is the honest start for a spectrum showing one and
    # a half arcs.
    tau_peak = 1.0 / (2 * np.pi * f[int(np.argmax(-Zs.imag))])
    return np.array([1e-9, Rs, 0.5 * R_tot, tau_peak / 10.0, 0.9,
                     0.5 * R_tot, tau_peak * 10.0, 0.9])
 
 
def ecm_fit(freq, Z, p0=None, drt_res=None, weight="modulus"):
    """Complex NLLS fit.  Returns params, Z_fit, chi2_nu, residuals."""
    freq = np.asarray(freq, float)
    Z = np.asarray(Z, complex)
    if p0 is None:
        p0 = ecm_starting_values(freq, Z, drt_res)
    wgt = 1.0 / np.maximum(np.abs(Z), 1e-12) if weight == "modulus" \
        else np.ones(len(Z))
 
    def resid(p):
        Zm = z_model(p, freq)
        return np.concatenate([(Z.real - Zm.real) * wgt,
                               (Z.imag - Zm.imag) * wgt])
 
    #      L      Rs    R1     tau1   n1    R2     tau2   n2
    lo = np.array([0.0, 0.0, 1e-6, 1e-7, 0.3, 1e-6, 1e-7, 0.3])
    hi = np.array([1e3, 1e6, 1e6, 1e4, 1.0, 1e6, 1e4, 1.0])
    r = least_squares(resid, np.clip(p0, lo + 1e-12, hi),
                      bounds=(lo, hi), method="trf", x_scale="jac",
                      max_nfev=20000)
    Zf = z_model(r.x, freq)
    names = ["L", "Rs", "R1", "tau1", "n1", "R2", "tau2", "n2"]
    par = dict(zip(names, r.x))
    par["Y01"] = y0_from_tau(par["R1"], par["tau1"], par["n1"])
    par["Y02"] = y0_from_tau(par["R2"], par["tau2"], par["n2"])
 
    # A parameter that has been pushed onto its bound is not a fitted value;
    # say so rather than reporting it as one.
    at_bound = [n for n, v, a, b in zip(names, r.x, lo, hi)
                if v <= a * 1.001 + 1e-12 or v >= b * 0.999]
    dof = max(2 * freq.size - len(names), 1)
 
    return {"params": par, "x": r.x, "Z_fit": Zf,
            "chi2": float(np.sum(r.fun ** 2)),
            "chi2_nu": float(2.0 * r.cost / dof),
            "res_pct": 100 * np.abs(Z - Zf) / np.abs(Z),
            "at_bound": at_bound,
            # success means "converged AND no parameter is sitting on a
            # bound", because a fit that ran out of room is not a fit.
            "success": bool(r.success) and not at_bound}
 
 
# ─── Run ECM fit on silver spectra (using DRT for starting values) ───
ECM_FIT_RESULTS = {}  # {cond: {seg: ecm_dict}}
 
for cond, pr in PIPELINE_RESULTS.items():
    if pr is None:
        continue
    spectra_path = spectra_csv(pr['out_dir'])
    if not spectra_path.exists():
        continue
 
    df = pd.read_csv(spectra_path)
    segments = sorted(df['segment'].unique())
    ECM_FIT_RESULTS[cond] = {}
    _errors, _skipped = {}, []
 
    for seg in segments:
        sd = df[df['segment'] == seg].sort_values('freq_hz')
        f = sd['freq_hz'].values
        zr = sd['z_re_mohm_cm2'].values
        zi = sd['z_im_mohm_cm2'].values
        Z = zr + 1j * zi
 
        keep = np.isfinite(zr) & np.isfinite(zi) & (zr > 0)
        if keep.sum() < 10:
            _skipped.append(int(seg))
            continue
        f, Z = f[keep], Z[keep]
 
        # Get DRT for this segment (if available)
        drt_res = DRT_RESULTS.get(cond, {}).get(seg, None)
 
        try:
            ecm = ecm_fit(f, Z, drt_res=drt_res)
            ecm['freq'] = f
            ecm['Z_meas'] = Z
            ECM_FIT_RESULTS[cond][seg] = ecm
        except Exception as exc:                       # noqa: BLE001
            # A swallowed exception here is how a plate quietly loses half
            # its segments: the count still looks plausible and nothing says
            # which ones went missing or why.
            _errors[int(seg)] = f"{type(exc).__name__}: {exc}"
 
    n_ok = sum(1 for v in ECM_FIT_RESULTS[cond].values() if v['success'])
    n_tot = len(ECM_FIT_RESULTS[cond])
    _bounded = {s: v['at_bound'] for s, v in ECM_FIT_RESULTS[cond].items()
                if v['at_bound']}
    rs_vals = [v['params']['Rs'] for v in ECM_FIT_RESULTS[cond].values() if v['success']]
    if rs_vals:
        print(f"  {cond}: {n_ok}/{n_tot} clean | "
              f"Rs = {np.mean(rs_vals):.1f} ± {np.std(rs_vals):.1f} mΩ·cm²")
    else:
        print(f"  {cond}: {n_ok}/{n_tot} clean")
    if _bounded:
        print(f"      {len(_bounded)} fits with a parameter on its bound "
              f"(not trustworthy): {dict(list(_bounded.items())[:6])}"
              f"{' ...' if len(_bounded) > 6 else ''}")
    if _errors:
        print(f"      {len(_errors)} segments raised: "
              f"{dict(list(_errors.items())[:4])}"
              f"{' ...' if len(_errors) > 4 else ''}")
    if _skipped:
        print(f"      {len(_skipped)} segments skipped (<10 usable points): "
              f"{_skipped[:12]}{' ...' if len(_skipped) > 12 else ''}")
 
# ─── Plot: Nyquist overlay + Residuals (Plotly interactive) ───
for cond, ecm_cond in ECM_FIT_RESULTS.items():
    if not ecm_cond:
        continue
 
    all_segs = sorted(ecm_cond.keys(), key=lambda x: int(x))
    step = max(1, len(all_segs) // 6)
    show_segs = all_segs[::step][:6]
    n_show = len(show_segs)
    n_cols = min(n_show, 3)
    n_rows = (n_show + n_cols - 1) // n_cols
 
    # --- Nyquist overlay ---
    titles_nyq = []
    for seg in show_segs:
        ecm = ecm_cond[seg]
        s = '✓' if ecm['success'] else ('⚠ ' + ','.join(ecm['at_bound'])
                                         if ecm['at_bound'] else '✗')
        titles_nyq.append(f'Seg {seg} {s} (χ²ν={ecm["chi2_nu"]:.3f})')
 
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles_nyq,
                        horizontal_spacing=0.06, vertical_spacing=0.12)
 
    for idx, seg in enumerate(show_segs):
        r, c = idx // n_cols + 1, idx % n_cols + 1
        ecm = ecm_cond[seg]
        Z_meas, Z_fit = ecm['Z_meas'], ecm['Z_fit']
        p = ecm['params']
 
        fig.add_trace(go.Scatter(x=Z_meas.real, y=-Z_meas.imag, mode='markers',
            marker=dict(size=4, color='steelblue', opacity=0.5),
            name='Data', showlegend=(idx == 0),
            hovertemplate=f"Seg {seg}<br>Z'=%{{x:.1f}}<br>-Z''=%{{y:.1f}}<extra>Data</extra>"),
            row=r, col=c)
        fig.add_trace(go.Scatter(x=Z_fit.real, y=-Z_fit.imag, mode='lines',
            line=dict(color='red', width=2.5),
            name='ECM fit', showlegend=(idx == 0),
            hovertemplate=f"Seg {seg}<br>Z'=%{{x:.1f}}<br>-Z''=%{{y:.1f}}<extra>ECM</extra>"),
            row=r, col=c)
 
        # Parameter annotation
        _xref = 'x domain' if idx == 0 else f'x{idx+1} domain'
        _yref = 'y domain' if idx == 0 else f'y{idx+1} domain'
        fig.add_annotation(text=f"Rs={p['Rs']:.1f} R1={p['R1']:.1f} R2={p['R2']:.1f}",
            xref=_xref, yref=_yref,
            x=0.98, y=0.95, showarrow=False, font=dict(size=9),
            xanchor='right', yanchor='top')
 
        fig.update_xaxes(title_text="Z' [mΩ·cm²]", row=r, col=c)
        _xanchor = 'x' if idx == 0 else f'x{idx+1}'
        fig.update_yaxes(title_text="-Z'' [mΩ·cm²]", scaleanchor=_xanchor, row=r, col=c)
 
    fig.update_layout(title=f'<b>ECM Fit — L+Rs+(Rct1||CPE1)+(Rct2||CPE2) — {cond} (Leepa {LEEPA})</b>',
                      height=400 * n_rows, width=1400,
                      paper_bgcolor='white', plot_bgcolor='white', hovermode='closest')
    fig.update_xaxes(showgrid=True, gridcolor='#eee')
    fig.update_yaxes(showgrid=True, gridcolor='#eee')
    show_fig(fig, html=True)   # one plot per figure, stacked
 
    # --- Residual plot ---
    titles_res = [f'Seg {s} — max {ecm_cond[s]["res_pct"].max():.1f}%' for s in show_segs]
    fig2 = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles_res,
                         horizontal_spacing=0.06, vertical_spacing=0.12)
 
    for idx, seg in enumerate(show_segs):
        r, c = idx // n_cols + 1, idx % n_cols + 1
        ecm = ecm_cond[seg]
        fig2.add_trace(go.Scatter(x=ecm['freq'], y=ecm['res_pct'], mode='lines+markers',
            marker=dict(size=3, color='steelblue'), line=dict(width=1, color='steelblue'),
            name=f'Seg {seg}', showlegend=False,
            hovertemplate='f=%{x:.1f} Hz<br>|res|=%{y:.2f}%<extra></extra>'),
            row=r, col=c)
        fig2.add_hline(y=2, line=dict(color='green', dash='dash', width=1), row=r, col=c)
        fig2.add_hline(y=5, line=dict(color='orange', dash='dash', width=1), row=r, col=c)
        fig2.update_xaxes(type='log', title_text='f [Hz]', row=r, col=c)
        fig2.update_yaxes(title_text='|Residual| [%]', rangemode='tozero', row=r, col=c)
 
    fig2.update_layout(title=f'<b>ECM Residuals |ΔZ|/|Z| — {cond} (Leepa {LEEPA})</b>',
                       height=350 * n_rows, width=1400,
                       paper_bgcolor='white', plot_bgcolor='white', hovermode='x unified')
    fig2.update_xaxes(showgrid=True, gridcolor='#eee')
    fig2.update_yaxes(showgrid=True, gridcolor='#eee')
    show_fig(fig2, html=True)  # one plot per figure, stacked
 
# ─── Parameter summary table ───
for cond, ecm_cond in ECM_FIT_RESULTS.items():
    if not ecm_cond:
        continue
    rows = []
    for seg, ecm in sorted(ecm_cond.items(), key=lambda x: int(x[0])):
        if ecm['success']:
            p = ecm['params']
            # L is in mOhm.cm2 * s here, because Z is in mOhm.cm2 -- it is an
            # area-specific inductance, NOT henries, so it cannot be printed
            # as nH.  (The previous version multiplied by 1e9 and called the
            # column L_nH, which is off by the cell area and by 1e-3.)
            rows.append({'segment': int(seg), 'Rs': p['Rs'],
                         'L_mohm_cm2_s': p['L'],
                         'Rct1': p['R1'], 'tau1_ms': p['tau1']*1e3,
                         'Y01': p['Y01'], 'n1': p['n1'],
                         'Rct2': p['R2'], 'tau2_ms': p['tau2']*1e3,
                         'Y02': p['Y02'], 'n2': p['n2'],
                         'chi2_nu': ecm['chi2_nu']})
    df_params = pd.DataFrame(rows)
    if df_params.empty:
        # .describe() on a frame with no columns raises, so the old code
        # turned "nothing converged" into a traceback instead of a message.
        print(f"\n  ECM Parameters — {cond}: no clean fit on any segment "
              f"({len(ecm_cond)} attempted).")
        continue
    print(f"\n  ECM Parameters — {cond}  ({len(df_params)}/{len(ecm_cond)} clean):")
    print(f"  {'─'*70}")
    print(f"  Rs:   {df_params['Rs'].mean():.2f} ± {df_params['Rs'].std():.2f} mΩ·cm²")
    print(f"  Rct1: {df_params['Rct1'].mean():.2f} ± {df_params['Rct1'].std():.2f} mΩ·cm²")
    print(f"  Rct2: {df_params['Rct2'].mean():.2f} ± {df_params['Rct2'].std():.2f} mΩ·cm²")
    print(f"  n1:   {df_params['n1'].mean():.3f} ± {df_params['n1'].std():.3f}")
    print(f"  n2:   {df_params['n2'].mean():.3f} ± {df_params['n2'].std():.3f}")
    print(f"  χ²ν:  {df_params['chi2_nu'].median():.4f} (median)")
    display(df_params.round(6))

# COMMAND ----------

import re
import numpy as np, pandas as pd
from pathlib import Path
import gamry_dta
from config import A_CELL_CM2

# Second copy of the same hard-coded table, removed for the same reason: it
# named one campaign's files and one folder, whichever order was selected.
# _ASR_GAMRY_DIR, DTA and SETPOINT come from the cell above, which resolves
# them from the order through its build token.
_ASR_GAMRY_DIR = globals().get('_ASR_GAMRY_DIR') or (
    gamry_root_for(_widget('leepa_id', 'LEEPA', 'leepa'))
    or Path('/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev/Gamry'))
if not globals().get('DTA'):
    _files, _build = gamry_files(_widget('leepa_id', 'LEEPA', 'leepa'))
    DTA = {}
    for _f in _files:
        _m = re.search(r'CurrVal_(\d+(?:[.,]\d+)?)', _f.stem)
        if _m:
            DTA.setdefault(_m.group(1).replace(',', '.') + 'A', _f.name)
    SETPOINT = {k: float(k[:-1]) for k in DTA}
MIN_COV  = 0.80        # area fraction a frequency must have to be trusted

def gamry_series_L(f, Z, top_decade=1.0):
    k = f >= f.max()/10**top_decade
    if k.sum() < 4: return 0.0
    w = 2*np.pi*f[k]
    return float(np.sum(w*Z.imag[k])/np.sum(w*w))

def hfr_band(f, Z, frac=0.5):
    """HF side of the arc: from where -Z'' falls to `frac` of its peak, to f_max.
    This is the region that determines the ohmic intercept."""
    y = -Z.imag; k = int(np.argmax(y)); thr = frac*y[k]; j = k
    while j < len(f)-1 and y[j+1] > thr: j += 1
    return f[j], f[-1]

def r_ohmic_topband(f, Z, decades=1/3):
    """Mean Re Z over the top third of a decade — the estimator silver uses,
    applied identically to both sides."""
    k = f >= f.max()*10**(-decades)
    if k.sum() < 2: k = f >= f.max()*10**(-1.0)
    return float(np.mean(Z.real[k])), int(k.sum())

def interp_log(f_new, f, Z):
    o = np.argsort(f)
    return (np.interp(np.log(f_new), np.log(f[o]), Z[o].real)
            + 1j*np.interp(np.log(f_new), np.log(f[o]), Z[o].imag))

for cond, fn in DTA.items():
    sv = find_silver(cond)                       # from the earlier script
    if sv is None or not (_ASR_GAMRY_DIR/fn).exists():
        print(f"{cond}: missing data"); continue

    ca  = pd.read_csv(sv/'cell_aggregate.csv').sort_values('freq_hz')
    seg = pd.read_csv(sv/'segments_summary.csv')
    sp  = pd.read_csv(sv/'spectra_clean.csv')
    area  = seg.set_index('segment').area_cm2
    A_tot = float(area.sum())

    # --- per-frequency area coverage --------------------------------------
    covmap = {}
    for f0, g in sp.groupby('freq_hz'):
        Z = g.z_re_mohm_cm2.values + 1j*g.z_im_mohm_cm2.values
        ok = np.isfinite(Z) & (Z != 0)
        covmap[round(float(f0), 6)] = g.segment.map(area).values[ok].sum()/A_tot

    f_l = ca.freq_hz.values
    Z_l = (ca.z_re_mohm_cm2.values + 1j*ca.z_im_mohm_cm2.values)/1e3
    cov = np.array([covmap.get(round(float(x), 6), np.nan) for x in f_l])

    # --- like-for-like corrections ----------------------------------------
    I_full = float((seg.j_dc_A_cm2*seg.area_cm2).sum())*A_CELL_CM2/A_tot
    I_err  = I_full/SETPOINT[cond] - 1.0
    Z_l    = Z_l * (1.0 + I_err)

    sw  = gamry_dta.read_dta(_ASR_GAMRY_DIR/fn).sorted()
    f_g = np.asarray(sw.freq, float); Z_g = np.asarray(sw.Z, complex)*A_CELL_CM2
    Z_g = Z_g - 1j*2*np.pi*f_g*gamry_series_L(f_g, Z_g)

    lo, hi = max(f_l.min(), f_g.min()), min(f_l.max(), f_g.max())
    m = (f_l >= lo) & (f_l <= hi)
    f, Zl, cv = f_l[m], Z_l[m], cov[m]
    Zg = interp_log(f, f_g, Z_g)

    a, b = hfr_band(f, Zg)
    hk   = (f >= a) & (f <= b)
    hk_c = hk & (cv >= MIN_COV)

    print(f"\n════ {cond}  (I_err {100*I_err:+.1f}% applied, band {lo:.2f}-{hi:.0f} Hz) ════")
    for nm, k in [("all frequencies", np.ones_like(f, bool)),
                  ("LF   < 1 Hz",  f < 1),
                  ("mid  1-100 Hz", (f>=1)&(f<100)),
                  (f"HFR  {a:.0f}-{b:.0f} Hz", hk),
                  (f"HFR, coverage>={100*MIN_COV:.0f}%", hk_c)]:
        if k.sum() == 0: continue
        d  = 100*(np.abs(Zl[k])-np.abs(Zg[k]))/np.abs(Zg[k])
        dr = 100*(Zl[k].real-Zg[k].real)/Zg[k].real
        print(f"   {nm:26s} n={k.sum():3d}  |Z| dev {np.median(d):+6.1f}% "
              f"(p95 {np.percentile(np.abs(d),95):5.1f}%)   Re Z dev {np.median(dr):+6.1f}%")

    RsA,nA = r_ohmic_topband(f[hk_c] if hk_c.sum()>3 else f, Zl[hk_c] if hk_c.sum()>3 else Zl)
    RsB,_  = r_ohmic_topband(f[hk_c] if hk_c.sum()>3 else f, Zg[hk_c] if hk_c.sum()>3 else Zg)
    print(f"   R_s  local {1e3*RsA:6.1f}  vs Gamry {1e3*RsB:6.1f} mohm.cm2  "
          f"-> {100*(RsA/RsB-1):+.1f}%   (from {nA} points)")

    print(f"   frequencies below {100*MIN_COV:.0f}% coverage in the HFR band: "
          f"{int((hk & (cv < MIN_COV)).sum())} of {int(hk.sum())}")