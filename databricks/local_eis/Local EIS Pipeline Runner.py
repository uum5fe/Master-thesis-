# Databricks notebook source
# fresh Python for a clean module reload -- skipped when the extras notebook
# runs this one with %run (it sets _EIS_NO_RESTART first)
if not globals().get('_EIS_NO_RESTART'):
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
# Folder layout (eis_paths.py puts every folder on sys.path):
#   core/      config, utils, plate geometry      pipeline/  bronze -> silver -> gold
#   checks/    plausibility + per-run checks       analysis/  ECM / DRT, polcurves
#   plotting/  maps, viewers, figure helpers       readers/   Gamry, Abgleich, CSV
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
        if (Path(cand) / 'eis_paths.py').exists():
            return cand
    except Exception:
        pass
    # Repos, a local checkout, or anything else: fall back to the cwd and to
    # the historical location, and say which one was used.
    for cand in (str(Path.cwd()),
                 '/Workspace/Users/uum5fe@bosch.com/Local_EIS_pipeline',
                 '/Workspace/Users/uum5fe@bosch.com/Local_EIS_fixed/Local_EIS_fixed'):
        if (Path(cand) / 'eis_paths.py').exists():
            return cand
    raise FileNotFoundError(
        "cannot find the pipeline modules. eis_paths.py should sit in the same "
        "folder as this notebook; if it does not, set "
        "_PIPELINE_DIR_OVERRIDE at the top of this cell.")
 
 
_PIPELINE_DIR = _find_pipeline_dir()
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)
 
import eis_paths            # core/, pipeline/, checks/, ... on sys.path
 
import importlib
importlib.invalidate_caches()
import numpy as np
import pandas as pd
 
# Import pipeline modules
import config
import utils
import eis_local
import gamry_sync
import channel_lag
import card_gain
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
import tone_estimation
import eis_measurement_model
import figure_panels
import plate_style
import plate_maps
import plate_figure
import plate_plotly
import bench_plots
import polcurve
import ecm_drt
import plausibility
import dc_closure
import segment_scale
import frequency_response
import via_resistance
import viewers
 
# Force reload during development (plate_style before the maps that use it)
for mod in [config, utils, eis_local, gamry_sync, channel_lag, card_gain,
            bronze, silver, gold, plausibility, dc_closure, segment_scale,
            frequency_response, via_resistance, viewers, pipeline_main,
            geom, csv_source, csv_pipeline, gamry_dta, gamry_compare, abgleich,
            ladder_snap, tone_estimation, eis_measurement_model,
            figure_panels, plate_style, plate_maps, plate_figure, plate_plotly,
            bench_plots, polcurve, ecm_drt]:
    importlib.reload(mod)

# ─── ONE PLOT PER FIGURE ───
# Every spectrum cell below still builds its multi-panel figure (Nyquist, |Z|,
# phase side by side, or a grid of segments) and hands it to show_fig(), which
# draws each panel as its own full-width figure, one below the other. Set
# PLOTS_ONE_BY_ONE = False to get the old side-by-side rows back.
PLOTS_ONE_BY_ONE = True
PANEL_HEIGHT = 560        # px, per single plot
PANEL_WIDTH = 1150        # px; None = fill the cell width


def show_by_condition(figs, title='', note=''):
    """{condition: figure} -> ONE plot with Condition (and Plot) drop-downs."""
    displayHTML(viewers.by_condition(figs, title=title, note=note,
                                     panel_height=PANEL_HEIGHT,
                                     panel_width=PANEL_WIDTH))


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
    dbutils.widgets.text('f_max_hz', '4500.0', 'F max (Hz)')
    dbutils.widgets.dropdown('min_snr_db', '0',
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
# The per-segment chain response, built ONCE by the "Chain response" cell from
# the Abgleich bode/ sweeps and kept beside curr.csv. If it exists it is used
# on every run (bronze divides each segment's Z by it); set GAIN_FILE = '' to
# switch it off, or to another path to override it.
CHAIN_GAIN_DEFAULT = Path('/Workspace/Users/uum5fe@bosch.com') / f'chain_gain_{PLATE}.csv'
GAIN_FILE = (globals().get('GAIN_FILE')
             or (str(CHAIN_GAIN_DEFAULT) if CHAIN_GAIN_DEFAULT.is_file() else ''))
print(f"  chain response: {GAIN_FILE or 'NONE -- run the Chain response cell once'}")
# Current-chain lag per segment (channel_lag.py), measured on every run:
# 'correct' removes it before R_ohmic is read, 'report' only records it,
# 'off' skips the stage. See silver/channel_lag.csv and the "channel lag"
# plausibility check after each run.
CHANNEL_LAG = 'correct'
# Gamry clock (gamry_sync.py): the FAMOS cards record the Gamry's own sweep,
# so its .dta gives every step's exact frequency and time. 'guide' uses the
# exact frequencies, applies a refused card lag the Gamry corroborates, and
# re-locates misplaced high-frequency windows (verified by a CFAR test);
# 'frequency' does the first two; 'report' only writes bronze/gamry_sync.csv.
GAMRY_SYNC = 'guide'
# Card voltage gain (card_gain.py): every card records the same cell voltage
# on UC2, and each card's Z divides by ITS OWN UC2 -- so a card whose voltage
# chain reads 3 % high puts +3 % on all its segments. 'report' measures each
# card's UC2 against the median card (bronze/card_reference.csv, needs the
# .DAT files), 'correct' also divides that card's Z by it when the gain is
# flat over the band and under 10 %, 'off' skips it.
CARD_GAIN = 'report'
# Series resistance between the FAMOS UC taps and the Gamry's sense leads,
# in mOhm*cm2, subtracted from every segment's Re Z (config.uc_series_mohm_cm2).
# The "Card voltage gain and DC current closure" cell measures it from UC2 DC
# against the Gamry Vdc over all conditions: 13.4 on 2612030. 0 = Z as the UC
# taps see it (the default); 13.4 = on the Gamry's reference plane.
UC_SERIES_MOHM_CM2 = 0.0
# Frequency response of the measuring chain (frequency_response.py), after
# every run: the ex-situ amplifier response G_d of each segment (from the
# Abgleich bode/ sweeps when ABGLEICH_DIR is set in the "Chain response" cell,
# otherwise from GAIN_FILE) and the in-situ response of each segment against
# the plate median, before and after the timing correction. Writes
# RUN_DIR/frequency_response/ and two plausibility checks; it never changes Z.
# 'off' skips it. See the "Frequency response" cell after the runs.
FREQ_RESPONSE = 'report'
# Static plate maps: 'segments' (each segment filled and labelled),
# 'interpolated' (2D linear interpolation between segment centres, squares
# on the measured segments, no numbers, parula with a soft gloss -- like the
# bench's MATLAB maps; plate_<param>_interp.png), or 'both'.
HEATMAP_STYLE = 'both'


def _bode_dir():
    """The Abgleich bode/ folder, when the "Chain response" cell has one."""
    _ab = globals().get('ABGLEICH_DIR') or ''
    _b = Path(_ab) / 'bode' if _ab else None
    return _b if _b is not None and _b.is_dir() else None


# The current-closure check compares the segment currents with the bench's
# measured I_S at the Gamry sweep; without a readable bench log it uses the
# current in the condition name (45A -> 45 A). Nothing to set here.
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
F_MAX = float(_w('f_max_hz', '4500.0'))
MIN_SNR_DB = float(_w('min_snr_db', '0'))
STOP_AFTER = _w('stop_after', 'gold')
EVALUATION_MODE = _w('evaluation_mode', 'default')

# ═══════════════════════════════════════════════════════════════════════════
#  RECOMMENDED PARAMETERS = THE SETTINGS OF THE SCRIPT THAT DREW CLEAN ARCS
# ═══════════════════════════════════════════════════════════════════════════
# Reference: the earlier notebook that gave clean 45 A arcs up to ~1 kHz and
# clean 150 A / 450 A spectra on RO2612030. Its bronze/silver/gold code is
# identical to this folder's; what differs is settings only:
#
#                               clean script    scattered run
#   evaluation mode             default         permissive
#   Min SNR (bronze)            0 dB            0 dB  (then 5 dB, see below)
#   f_max                       4500 Hz         4500 Hz
#   fit_common_delay (config)   False           True
#   silver_snr_gate_db (config) -40 dB          -20 dB
#
# The two config values are restored in config.py (and are now part of the
# cache key). 'permissive' loosens sigma_rel_max to 1.5, zmag_outlier_mad to
# 8 and thd/drift to 0.5, which keeps noise and spikes -- it is not used here.
#
# A 5 dB SNR gate was tried in between and was WRONG for 45 A: bronze uses
# min_snr_db to accept off-grid steps and to decide which steps the grid fit
# rests on. The excitation at 45 A is about a tenth of the 450 A one, so the
# steps above ~90 Hz sit below 5 dB at 45 A and were never detected -- the
# 45 A spectra stopped at ~90 Hz while 450 A still reached ~900 Hz. 0 dB is
# what the clean script used; silver's uncertainty gate (sigma_rel_max 0.60)
# and |Z| outlier gate are what keep noise out, per point.
#
# 'recommended' pins these widget values; 'custom' in the Parameter profile
# widget uses the widgets instead. The mode and all these settings are in the
# cache key, so different settings never share a cache entry.
RECOMMENDED_PARAMS = dict(evaluation_mode='default', min_snr_db=0.0,
                          f_min_hz=0.15, f_max_hz=4500.0)
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

# DBTITLE 1,Chain response: build the gain file from the Abgleich bode sweeps
# ═══════════════════════════════════════════════════════════════════════════════
# The Abgleich delivery carries, next to curr.csv/temp.csv, a bode/ folder of
# per-segment Gamry sweeps: the current-measurement chain swept 1 Hz–100 kHz
# at 500 mA rms with no DC bias.  Measured on both plates it is flat to 1 kHz
# and then rolls off — -11° at 4.5 kHz, -24° at 10 kHz.  4500 Hz is the top of
# the default analysis band, so this is the same order as the acquisition skew
# the pipeline works hard to remove, and unlike a skew it moves |Z| too.
#
# Set ABGLEICH_DIR to the folder holding coefficients/ and bode/ and run this
# cell ONCE. The file is written beside curr.csv (CHAIN_GAIN_DEFAULT) and
# GAIN_FILE is set to it, here and -- because the settings cell looks for it
# -- in every later session. Nothing to paste.
#
# WHAT IT FIXES AND WHAT IT DOES NOT. It removes the chain roll-off every
# segment shares (-2.5 deg at 1 kHz, -11 deg at 4.5 kHz). The per-segment
# lags the in-situ stage measures (channel_lag.py, up to ~100 us on 2612030)
# are another matter: on the delivered sweeps the segments differ by only
# ~2 deg at 4.5 kHz, i.e. ~1 us. The comparison printed below says, on your
# data, how much of the in-situ lag the ex-situ chain explains; whatever it
# does not explain stays with the in-situ stage (CHANNEL_LAG = 'correct').
# ═══════════════════════════════════════════════════════════════════════════════
# THE FOLDER TO PUT HERE is the plate's calibration ("Abgleich") delivery --
# the one that holds BOTH of these sub-folders:
#
#     <ABGLEICH_DIR>/
#         coefficients/curr.csv        72 rows "c0;c1"  (the same kind of file
#         coefficients/temp.csv         4 rows           as your curr.csv)
#         bode/<name>_100kHz_1Hz_500mA_#1.DTA   one Gamry sweep per segment,
#         bode/<name>_100kHz_1Hz_500mA_#2.DTA   #1 .. #72 (the "_Raw.DTA"
#         ...                                   files beside them are ignored)
#         Step1_<T>Grad.csv, Step2_...          (the DC calibration steps)
#
# e.g. '/Volumes/.../R2D2_green_Kashyyyk/Abgleichdaten/Kashyyyk' for the gen1
# (green / Kashyyyk) plate. Point it at the folder ABOVE bode/, not at bode/.
# Left empty, the roots below are searched and every match is listed.
ABGLEICH_DIR = ''
ABGLEICH_SEARCH_ROOTS = [
    '/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev',
    '/Workspace/Users/uum5fe@bosch.com',
]

if not ABGLEICH_DIR:
    _cand = gamry_dta.find_abgleich_dirs(ABGLEICH_SEARCH_ROOTS)
    if _cand:
        print(f"  ABGLEICH_DIR is empty; found {len(_cand)} candidate(s) "
              f"(a bode/ with per-segment #n.DTA sweeps):")
        for _c in _cand:
            _ok = (_c / 'coefficients' / 'curr.csv').is_file()
            _n = len([p for p in (_c / 'bode').glob('*.DTA')
                      if not p.stem.endswith('_Raw')])
            print(f"    {_c}   ({_n} sweeps, coefficients/curr.csv "
                  f"{'present' if _ok else 'MISSING'})")
        _ready = [c for c in _cand
                  if (c / 'coefficients' / 'curr.csv').is_file()]
        if len(_ready) == 1:
            ABGLEICH_DIR = str(_ready[0])
            print(f"  -> using the only complete one: {ABGLEICH_DIR}")
        else:
            print("  -> more than one (or none complete): copy the right path "
                  "into ABGLEICH_DIR above and re-run this cell. For gen1 it "
                  "is the green / Kashyyyk delivery.")

if ABGLEICH_DIR:
    import csv
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
 
    _gain_out = Path(CHAIN_GAIN_DEFAULT)
    gamry_dta.write_gain_csv(_sweeps, _gain_out,
                             curr_csv=_curr, shared=not _chk_g['ok'])
    GAIN_FILE = str(_gain_out)
    print(f"  written: {_gain_out}  -> GAIN_FILE is set; every run from now "
          f"on applies it")

    # per segment: the chain's own time constant, to set beside the in-situ
    # lag of a finished run (silver/channel_lag.csv, CHANNEL_LAG stage)
    _btau = gamry_dta.chain_tau(_sweeps)
    if _btau:
        _bt = np.array(list(_btau.values())) * 1e6
        print(f"  ex-situ chain tau over {len(_bt)} segments: median "
              f"{np.median(_bt):+.1f} us, spread (sd) {np.std(_bt):.1f} us")
        _lag_csvs = sorted(Path('/Volumes/ps_xplatform_dev/rvadvtec_dev/'
                                'ev_rvadvtec_dev/EIS_Results').rglob(
                                    'silver/channel_lag.csv'))[-1:] \
            if Path('/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev/'
                    'EIS_Results').exists() else []
        for _lc in _lag_csvs:
            _ins = {r['segment']: float(r['tau_us']) * 1e-6
                    for r in csv.DictReader(open(_lc))
                    if r.get('tau_us') not in ('', None)}
            _c = gamry_dta.compare_chain_tau(_btau, _ins)
            if _c.get('ok'):
                print(f"  vs in-situ lag of {_lc.parent.parent.name}: "
                      f"r = {_c['r']:+.2f}, in-situ spread "
                      f"{_c['sd_insitu_us']:.0f} us, {100*_c['explained']:.0f} % "
                      f"explained by the ex-situ chain")
 
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
    print("  ABGLEICH_DIR is empty and no Abgleich delivery (a folder with "
          "bode/*_#<n>.DTA and coefficients/curr.csv) was found under "
          f"{ABGLEICH_SEARCH_ROOTS} -- skipping. Ask for / upload the plate's "
          "calibration delivery and set ABGLEICH_DIR to it; without it the "
          "top decade of the band carries -11 deg of uncorrected phase "
          "(the in-situ CHANNEL_LAG stage still runs).")

# COMMAND ----------

# DBTITLE 1,Datago Source: discover + read FAMOS waveforms from Delta table
# MAGIC %skip
# MAGIC # ═══════════════════════════════════════════════════════════════════════════════
# MAGIC # DATAGO SOURCE: FamosFile-compatible reader from Delta table
# MAGIC #
# MAGIC # Replaces the Volumes-based file reader with a datago query backend.
# MAGIC # The pipeline (bronze.py) calls FamosFile(path) to get waveform data.
# MAGIC # This cell provides DatagoFamosFile that has the SAME interface but
# MAGIC # reads from ps_xplatform_dev.rvadvtec_ops.datago_advtec_values_delta.
# MAGIC #
# MAGIC # PERFORMANCE NOTE:
# MAGIC #   Each card = 16 channels × 2.5M samples = 40M rows from Delta.
# MAGIC #   ~30-60s per card vs ~5s from Volumes binary. Use Volumes when available.
# MAGIC #   Toggle via DATA_SOURCE widget below.
# MAGIC # ═══════════════════════════════════════════════════════════════════════════════
# MAGIC import numpy as np
# MAGIC import time
# MAGIC from pathlib import Path
# MAGIC from dataclasses import dataclass, field
# MAGIC  
# MAGIC _VAL_TBL = 'ps_xplatform_dev.rvadvtec_ops.datago_advtec_values_delta'
# MAGIC _META_TBL = 'ps_xplatform_dev.rvadvtec_ops.datago_advtec_metadata'
# MAGIC _GP_TBL = 'ps_xplatform_dev.rvadvtec_ops.datago_advtec_generalproperties'
# MAGIC  
# MAGIC # ─── Data source selection ───
# MAGIC # Set to 'datago' to read from Delta table, 'volumes' for fast binary
# MAGIC DATA_SOURCE = 'datago'  # <-- SWITCH HERE
# MAGIC  
# MAGIC print(f"  Data source: {DATA_SOURCE}")
# MAGIC  
# MAGIC  
# MAGIC # ─── Discover FAMOS file_ids from datago ───
# MAGIC def discover_famos_file_ids(leepa: str) -> dict:
# MAGIC     """Find FAMOS card recordings in datago for a Leepa.
# MAGIC     
# MAGIC     Returns: {condition: [file_id_1, ..., file_id_5]} (one per card)
# MAGIC     """
# MAGIC     # Find NULL measurement_type entries (FAMOS files have no type set)
# MAGIC     df = spark.sql(f"""
# MAGIC         SELECT DISTINCT gp.file_id, gp.measurementBegin
# MAGIC         FROM {_META_TBL} m
# MAGIC         JOIN {_GP_TBL} gp ON m.measurement_id = gp.measurement_id
# MAGIC         WHERE m.orderId = 'RO{leepa}'
# MAGIC           AND gp.measurement_type IS NULL
# MAGIC         ORDER BY gp.measurementBegin
# MAGIC     """).toPandas()
# MAGIC     
# MAGIC     if df.empty:
# MAGIC         return {}
# MAGIC     
# MAGIC     # Exclude TOM bench file (first one, usually much earlier timestamp)
# MAGIC     # TOM bench has 80+ channels; FAMOS cards have exactly 16
# MAGIC     # Heuristic: group by minute, groups of 5 = card sets
# MAGIC     df['minute'] = df['measurementBegin'].dt.floor('min')
# MAGIC     groups = df.groupby('minute')['file_id'].apply(list).to_dict()
# MAGIC     
# MAGIC     # Filter: only groups with exactly 5 files (= 5 cards per condition)
# MAGIC     card_groups = {k: v for k, v in groups.items() if len(v) == 5}
# MAGIC     
# MAGIC     # Map to conditions by order (same order as Volumes naming)
# MAGIC     # Try to determine condition from Volumes filenames if available
# MAGIC     conditions_ordered = []
# MAGIC     famos_root = Path('/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev/Famos')
# MAGIC     try:
# MAGIC         vol_files = sorted(famos_root.glob(f'Leepa_{leepa}_Current_*_Karte_1.DAT'))
# MAGIC         conditions_ordered = [f.name.split('_')[3] for f in vol_files]
# MAGIC     except Exception:
# MAGIC         pass
# MAGIC     
# MAGIC     if not conditions_ordered:
# MAGIC         conditions_ordered = ['150A', '450A', '45A', '60A']  # default order
# MAGIC     
# MAGIC     result = {}
# MAGIC     for idx, (ts, file_ids) in enumerate(sorted(card_groups.items())):
# MAGIC         cond = conditions_ordered[idx] if idx < len(conditions_ordered) else f'Cond-{idx+1}'
# MAGIC         result[cond] = file_ids
# MAGIC     
# MAGIC     return result
# MAGIC  
# MAGIC  
# MAGIC # ─── DatagoFamosFile: drop-in replacement for FamosFile ───
# MAGIC class DatagoFamosFile:
# MAGIC     """FamosFile-compatible reader that pulls waveform data from datago.
# MAGIC     
# MAGIC     Interface matches eis_local.FamosFile:
# MAGIC         .path, .fs, .n_ch, .n_samples, .names,
# MAGIC         .segment_names, .uc_names, .temp_names,
# MAGIC         .channel(name) -> np.ndarray
# MAGIC     """
# MAGIC     
# MAGIC     def __init__(self, file_id: str, card_label: str = 'datago'):
# MAGIC         self.file_id = file_id
# MAGIC         self.path = Path(f'/datago/{card_label}.DAT')  # fake path for compatibility
# MAGIC         self._channels = {}  # lazy-loaded
# MAGIC         self._metadata_loaded = False
# MAGIC         self._load_metadata()
# MAGIC     
# MAGIC     def _load_metadata(self):
# MAGIC         """Load channel list and sample counts (lightweight query)."""
# MAGIC         df = spark.sql(f"""
# MAGIC             SELECT channel, COUNT(*) as n_pts,
# MAGIC                    MIN(CAST(time_value AS DOUBLE)) as t_min,
# MAGIC                    MAX(CAST(time_value AS DOUBLE)) as t_max
# MAGIC             FROM {_VAL_TBL}
# MAGIC             WHERE file_id = '{self.file_id}'
# MAGIC             GROUP BY channel
# MAGIC             ORDER BY channel
# MAGIC         """).toPandas()
# MAGIC         
# MAGIC         self.names = df['channel'].tolist()
# MAGIC         self.n_ch = len(self.names)
# MAGIC         
# MAGIC         # Determine sampling rate from first channel
# MAGIC         if not df.empty:
# MAGIC             row0 = df.iloc[0]
# MAGIC             duration = row0['t_max'] - row0['t_min']
# MAGIC             self.n_samples = int(row0['n_pts'])
# MAGIC             self.fs = round(self.n_samples / duration) if duration > 0 else 10000
# MAGIC         else:
# MAGIC             self.n_samples = 0
# MAGIC             self.fs = 10000
# MAGIC         
# MAGIC         # Classify channels
# MAGIC         self.segment_names = [n for n in self.names if n.isdigit()]
# MAGIC         self.uc_names = [n for n in self.names if n.startswith('UC')]
# MAGIC         self.temp_names = [n for n in self.names if n.startswith('Temp')]
# MAGIC         
# MAGIC         # Acquisition slot positions (channel index = multiplexing order)
# MAGIC         # In real FAMOS, this is the binary header order. Here we approximate
# MAGIC         # using the standard R2D2 card layout: UC first, then segments, then Temp
# MAGIC         _ordered = self.uc_names + self.segment_names + self.temp_names
# MAGIC         self._position_map = {name: idx for idx, name in enumerate(_ordered)}
# MAGIC         self.positions = _ordered
# MAGIC         self._metadata_loaded = True
# MAGIC     
# MAGIC     def position(self, channel_name: str) -> int:
# MAGIC         """Return acquisition slot index for a channel (0-based)."""
# MAGIC         return self._position_map.get(channel_name, 0)
# MAGIC     
# MAGIC     def channel(self, name: str) -> np.ndarray:
# MAGIC         """Get waveform data for a channel (loads from datago on first access)."""
# MAGIC         if name not in self._channels:
# MAGIC             self._load_channel(name)
# MAGIC         return self._channels[name]
# MAGIC     
# MAGIC     def _load_channel(self, name: str):
# MAGIC         """Query datago for a single channel's waveform."""
# MAGIC         df = spark.sql(f"""
# MAGIC             SELECT CAST(time_value AS DOUBLE) as t,
# MAGIC                    CAST(value AS DOUBLE) as v
# MAGIC             FROM {_VAL_TBL}
# MAGIC             WHERE file_id = '{self.file_id}'
# MAGIC               AND channel = '{name}'
# MAGIC             ORDER BY t
# MAGIC         """).toPandas()
# MAGIC         self._channels[name] = df['v'].values
# MAGIC     
# MAGIC     def load_all_channels(self):
# MAGIC         """Bulk-load all channels at once (more efficient than one-by-one)."""
# MAGIC         t0 = time.time()
# MAGIC         df = spark.sql(f"""
# MAGIC             SELECT channel,
# MAGIC                    CAST(time_value AS DOUBLE) as t,
# MAGIC                    CAST(value AS DOUBLE) as v
# MAGIC             FROM {_VAL_TBL}
# MAGIC             WHERE file_id = '{self.file_id}'
# MAGIC             ORDER BY channel, t
# MAGIC         """).toPandas()
# MAGIC         
# MAGIC         for ch_name, grp in df.groupby('channel'):
# MAGIC             self._channels[ch_name] = grp['v'].values
# MAGIC         
# MAGIC         dt = time.time() - t0
# MAGIC         print(f"    [{self.path.stem}] loaded {self.n_ch} channels, "
# MAGIC               f"{self.n_samples:,} pts/ch in {dt:.1f}s")
# MAGIC     
# MAGIC     def __getitem__(self, name: str) -> np.ndarray:
# MAGIC         """Array-style access: fam['UC2'] -> waveform."""
# MAGIC         return self.channel(name)
# MAGIC  
# MAGIC  
# MAGIC # ─── Discover for current Leepa ───
# MAGIC DATAGO_FAMOS_MAP = discover_famos_file_ids(LEEPA)
# MAGIC  
# MAGIC if DATAGO_FAMOS_MAP:
# MAGIC     print(f"\n  Datago FAMOS files for Leepa {LEEPA}:")
# MAGIC     for cond, fids in sorted(DATAGO_FAMOS_MAP.items()):
# MAGIC         print(f"    {cond}: {len(fids)} cards ({fids[0][:12]}...)")
# MAGIC     print(f"  Total: {sum(len(v) for v in DATAGO_FAMOS_MAP.values())} files")
# MAGIC else:
# MAGIC     print(f"  No FAMOS data in datago for Leepa {LEEPA}")
# MAGIC  
# MAGIC # ─── Monkey-patch bronze.py to use datago when selected ───
# MAGIC if DATA_SOURCE == 'datago' and DATAGO_FAMOS_MAP:
# MAGIC     import bronze as _bronze_mod
# MAGIC     _orig_FamosFile = _bronze_mod.FamosFile  # keep reference to original
# MAGIC     
# MAGIC     # Build a lookup: condition+card_index -> file_id
# MAGIC     _DATAGO_CARD_LOOKUP = {}
# MAGIC     for cond, fids in DATAGO_FAMOS_MAP.items():
# MAGIC         for card_idx, fid in enumerate(fids, start=1):
# MAGIC             _DATAGO_CARD_LOOKUP[(cond, card_idx)] = fid
# MAGIC     
# MAGIC     # Create a wrapper that intercepts FamosFile(path) calls
# MAGIC     class _FamosFileDatagoShim:
# MAGIC         """Intercepts FamosFile(path) and routes to datago if file_id known."""
# MAGIC         def __new__(cls, path, *args, **kwargs):
# MAGIC             path = Path(path)
# MAGIC             # Try to extract condition + card from filename
# MAGIC             # Expected: Leepa_2611976_Current_60A_Test_01_Karte_1.DAT
# MAGIC             name = path.name
# MAGIC             parts = name.split('_')
# MAGIC             try:
# MAGIC                 cond = parts[3]           # e.g. '60A'
# MAGIC                 card = int(parts[-1].replace('.DAT', ''))  # e.g. 1
# MAGIC                 key = (cond, card)
# MAGIC                 if key in _DATAGO_CARD_LOOKUP:
# MAGIC                     fid = _DATAGO_CARD_LOOKUP[key]
# MAGIC                     reader = DatagoFamosFile(fid, card_label=f'Karte_{card}')
# MAGIC                     reader.load_all_channels()  # pre-fetch everything
# MAGIC                     return reader
# MAGIC             except (IndexError, ValueError):
# MAGIC                 pass
# MAGIC             # Fall back to original file-based reader
# MAGIC             return _orig_FamosFile(path, *args, **kwargs)
# MAGIC     
# MAGIC     _bronze_mod.FamosFile = _FamosFileDatagoShim
# MAGIC     print(f"\n   Bronze patched: FamosFile now reads from datago")
# MAGIC     print(f"    (Note: ~30-60s per card due to Delta row scan)")
# MAGIC else:
# MAGIC     print(f"\n  Using Volumes path (fast binary read)")

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
    'phasor_method', 'drift_alpha', 'skew_model', 'uncertainty_model',
    # Both changed between the clean script and the scattered run without
    # changing the key, so a stale entry would have been served as current.
    'fit_common_delay', 'silver_snr_gate_db', 'silver_snr_floor_db',
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
        fill_missing_from_neighbours=globals().get('FILL_GAPS', False),
        channel_lag=globals().get('CHANNEL_LAG', 'correct'),
        gamry_sync=globals().get('GAMRY_SYNC', 'guide'),
        card_gain=globals().get('CARD_GAIN', 'report'),
        uc_series_mohm_cm2=float(globals().get('UC_SERIES_MOHM_CM2', 0.0)))
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
        channel_lag=CHANNEL_LAG,
        gamry_sync=GAMRY_SYNC,
        card_gain=CARD_GAIN,
        uc_series_mohm_cm2=float(UC_SERIES_MOHM_CM2),
        freq_response=FREQ_RESPONSE,
        abgleich_bode_dir=_bode_dir(),
        heatmap_style=HEATMAP_STYLE,
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

# DBTITLE 1,Card voltage gain and DC current closure (all conditions)
# ═══════════════════════════════════════════════════════════════════════════════
# 1. CARD VOLTAGE GAIN -- per condition, from bronze/card_reference.csv:
#    each card's UC2 against the median of the cards. All cards record the
#    same cell voltage, so anything but 0 % is that card's voltage chain, and
#    it sits on every segment of the card (each card divides by its own UC2).
# 2. CARD IMPEDANCE FACTOR -- per condition, from silver/card_factors.csv:
#    the card's |Z| against the plate in four frequency bands and its j_dc.
#    A scale error is the same in every band with j_dc off the other way.
# 3. DC CURRENT CLOSURE across every condition run (45 / 60 / 150 / 450 A):
#    the plate current from the segments against the bench current, fitted as
#    I_plate = a * I_bench + b. a - 1 is the current-scale error, and it is in
#    every impedance too; b is a zero offset, which is not.
# ═══════════════════════════════════════════════════════════════════════════════
importlib.reload(card_gain)
importlib.reload(dc_closure)
importlib.reload(segment_scale)

_cg_runs = [Path(pr['out_dir']) for pr in PIPELINE_RESULTS.values()
            if pr and pr.get('out_dir')]
for _rd in _cg_runs:
    print(f"\n── {_rd.name} ──")
    _cr = _rd / 'bronze' / 'card_reference.csv'
    if _cr.is_file():
        _t = pd.read_csv(_cr)
        _t['card'] = _t['card'].map(card_gain.short)
        print("  UC channels, card against the median card "
              "(gain_pct: what Z carries if 'applied' is 1):")
        print(_t[['channel', 'card', 'used_for_Z', 'gain_pct', 'flat_pct',
                  'dt_us', 'dc_V', 'applied', 'status']]
              .to_string(index=False))
    else:
        print("  no bronze/card_reference.csv -- this result predates the UC "
              "check or came from the cache; re-run with Run mode 'rerun'")
    _cf = _rd / 'silver' / 'card_factors.csv'
    if _cf.is_file():
        _t = pd.read_csv(_cf)
        _t['card'] = _t['card'].map(card_gain.short)
        print("  |Z| per card against the plate [%], by band, and j_dc [%]:")
        print(_t.to_string(index=False))

if _cg_runs:
    _bench = Path(BENCH_LOG) if BENCH_LOG else None
    if _bench is None and GAMRY_DIR:
        try:
            _bench = gamry_compare.find_bench_log(
                GAMRY_DIR, order_id=LEEPA, version=GAMRY_VERSION or None)
        except Exception:                                   # noqa: BLE001
            _bench = None
    _dcc = dc_closure.analyse(_cg_runs, bench_log=_bench,
                              gamry_dir=GAMRY_DIR or None)
    _dcc_dir = _TMP_BASE / LEEPA / 'dc_closure'
    _dcc_dir.mkdir(parents=True, exist_ok=True)
    utils.write_table(_dcc_dir / 'dc_closure.csv', _dcc['rows'])
    utils.write_table(_dcc_dir / 'dc_closure_cards.csv', _dcc['cards'])
    dc_closure.plot(_dcc, _dcc_dir / 'dc_closure.png')
    print(f"\n{'═'*75}\n  DC CURRENT CLOSURE across {len(_cg_runs)} condition(s)"
          f"\n{'═'*75}")
    _t = pd.DataFrame(_dcc['rows'])[['condition', 'i_ref_A', 'i_ref_source',
                                     'i_full_A', 'dev_pct']]
    print(_t.round(2).to_string(index=False))
    if 'scale' in _dcc:
        print(f"\n  I_plate = {_dcc['scale']:.4f} * I_bench "
              f"{_dcc['offset_A']:+.2f} A  -> current scale "
              f"{_dcc['scale_pct']:+.2f} % (in every impedance as "
              f"{-_dcc['scale_pct']:+.2f} %), zero offset "
              f"{_dcc['offset_A']:+.2f} A (not in the impedance)")
    dc_closure.print_sense(_dcc)
    _ct = pd.DataFrame(_dcc['cards'])
    if not _ct.empty:
        print("\n  card current density against the plate [%] -- a K error on "
              "a card is the same at every current:")
        _pv = _ct.pivot_table(index='card', columns='condition',
                              values='j_vs_plate_pct')
        _pv = _pv[sorted(_pv.columns, key=card_gain.current_of)]
        print(_pv.round(2).to_string())
    if (_dcc_dir / 'dc_closure.png').is_file():
        display(IPImage(filename=str(_dcc_dir / 'dc_closure.png')))

    # 4. PER-SEGMENT SCALE (segment_scale.py): is a segment's |Z| off by one
    #    constant factor at every frequency AND every current? That is a
    #    scale error in its current path (K in situ, or the area it really
    #    collects from), not the cell -- and on 2612030 it is most of the
    #    neighbour-to-neighbour HFR scatter. Written as a gain file, NOT
    #    applied: trace it first (swap two channels' cables).
    _ss_runs = [r for r in _cg_runs
                if (r / 'silver' / 'spectra_clean.csv').is_file()]
    if len(_ss_runs) >= 2:
        _ssr = segment_scale.analyse(_ss_runs)
        print(f"\n{'═'*75}\n  PER-SEGMENT SCALE FACTOR across "
              f"{len(_ss_runs)} condition(s)\n{'═'*75}")
        segment_scale.report(_ssr, log=lambda m: print(m))
        _ssr['table'].to_csv(_dcc_dir / 'segment_scale.csv')
        _ss_gain = segment_scale.write_gain(
            _dcc_dir / 'segment_scale_gain.csv', _ssr['table'].scale,
            note=', '.join(_ssr['conditions']))
        print(f"  factors as a gain file (NOT applied): {_ss_gain}")
        print("  to test it on these results without the .DAT files:\n"
              "    pipeline_main.reevaluate(<run_dir>, out_dir=<new_dir>, "
              "gain_file=<that file>)")
    print(f"\n  written to {_dcc_dir}")
 

# COMMAND ----------

# DBTITLE 1,Frequency response of the measuring chain (ex-situ + in-situ, all conditions)
# ═══════════════════════════════════════════════════════════════════════════════
# Z = K * U_cell / u_seg is a ratio of two measured signals, so whatever the
# measuring chain does to either signal lands in Z -- hardest at the top of
# the band, where a phase error rotates the points that set the HFR.
#
# 1. EX-SITU G_d(f) = j_sigma / j_r of each segment's current amplifier, from
#    the Abgleich bode/ sweeps (ABGLEICH_DIR in the "Chain response" cell) or
#    from GAIN_FILE. Absolute, but it sees only the amplifiers.
# 2. IN-SITU Z_s / plate median of each segment during the measurement,
#    before (chain file divided out, no timing correction) and after (mux
#    de-skew + channel lag). Relative -- what all segments share cancels --
#    but it sees the whole chain: cards, multiplexer slots, scale factors.
#
# Each run already wrote RUN_DIR/frequency_response/ (FREQ_RESPONSE =
# 'report'); results from the cache that predate it are computed here.
# Nothing in this cell changes Z.
# ═══════════════════════════════════════════════════════════════════════════════
importlib.reload(frequency_response)

_fr_runs = [Path(pr['out_dir']) for pr in PIPELINE_RESULTS.values()
            if pr and pr.get('out_dir')]
_fr_rows, _fr_imgs, _fr_ex = [], {}, None
for _rd in _fr_runs:
    print(f"\n{'═'*75}\n  {_rd.name}\n{'═'*75}")
    _frd = _rd / frequency_response.OUT
    _sj = _frd / 'summary.json'
    if _sj.is_file():
        _frs = json.loads(_sj.read_text())
        frequency_response.report(_frs, say=print)
    else:
        # a read-only cache folder gets its results beside the other tmp output
        try:
            _frd.mkdir(parents=True, exist_ok=True)
            (_frd / '.w').touch()
            (_frd / '.w').unlink()
        except OSError:
            _frd = _TMP_BASE / LEEPA / 'frequency_response' / _rd.name
        _frs = frequency_response.run(
            _rd, gain_file=GAIN_FILE or None, bode_dir=_bode_dir(),
            out_dir=_frd, f_top=min(float(F_MAX), 4000.0),
            title=f"In-situ response, {_rd.name}")
    for _c in frequency_response.checks(_frs):
        print(f"  {_c}")
    if _fr_ex is None and (_frd / 'freq_response_exsitu.png').is_file():
        _fr_ex = _frd / 'freq_response_exsitu.png'
    if (_frd / 'freq_response_insitu.png').is_file():
        _fr_imgs[_rd.name] = viewers.img_html(_frd / 'freq_response_insitu.png')
    _ins = _frs.get('insitu', {})
    if _ins.get('available'):
        _k = f"{_ins['f_top_hz']:g}"
        _row = {'run': _rd.name}
        for _st in ('raw', 'clean'):
            _r = _ins['stages'].get(_st, {}).get(_k)
            if _r:
                _row[f'phase_sd_{_st}_deg'] = round(_r['phase_deg_sd'], 1)
        for _c in _ins.get('card_residual', []):
            _row[f"{_c['card']}_step_pct"] = _c['mag_step_hf_pct']
        _fr_rows.append(_row)

if _fr_rows:
    print(f"\n{'═'*75}\n  IN-SITU, ALL CONDITIONS: phase spread at the top of the "
          f"band (sd over segments) and |H| step per card above "
          f"{frequency_response.F_HF:g} Hz\n{'═'*75}")
    print(pd.DataFrame(_fr_rows).to_string(index=False))
    print("  A card step that is the same at every current is the hardware; "
          "one that changes with load is the cell.")

# one plot: the ex-situ amplifier response, or the in-situ response of the
# condition chosen in the drop-down
_fr_items = {}
if _fr_ex is not None:
    _fr_items[('Ex-situ amplifiers (Abgleich)',)] = viewers.img_html(_fr_ex)
for _c, _h in _fr_imgs.items():
    _fr_items[(f'In-situ, {_c}',)] = _h
if _fr_items:
    displayHTML(viewers.selector_html(_fr_items, ['Frequency response']))

# COMMAND ----------

# DBTITLE 1,Via resistance vs temperature per card (Abgleich DC calibration)
# ═══════════════════════════════════════════════════════════════════════════════
# Each segment's current is the voltage drop over its vias, amplified. The
# Abgleich DC calibration (Step1_20Grad .. Step5_90Grad, Step6_20Grad back)
# gives, per segment, the slope k = du/di at 20..90 degC: proportional to the
# via resistance. Plotted per card, with the operating temperatures of the
# runs below shaded.
#
# Why it matters for the current density: j = u / K(T), so a via whose real
# temperature differs from the one the pipeline assumes for it (interpolated
# from T1..T4) shifts j by -alpha per kelvin (alpha ~ 0.39 %/K, copper) and
# every impedance of that segment by +alpha per kelvin.
# Needs ABGLEICH_DIR (set in the "Chain response" cell).
# ═══════════════════════════════════════════════════════════════════════════════
importlib.reload(via_resistance)

_ab = globals().get('ABGLEICH_DIR') or ''
if not _ab or not Path(_ab).is_dir():
    print("  ABGLEICH_DIR is not set -- run the 'Chain response' cell first "
          "(it finds the Abgleich folder), or set ABGLEICH_DIR to the folder "
          "that holds Step1_20Grad.csv and coefficients/.")
else:
    _via_runs = [Path(pr['out_dir']) for pr in PIPELINE_RESULTS.values()
                 if pr and pr.get('out_dir')]
    _via_out = _TMP_BASE / LEEPA / 'via_resistance'
    _via = via_resistance.run(_ab, _via_runs, out_dir=_via_out)
    for _png in ('via_vs_temperature.png', 'via_alpha.png'):
        if (_via_out / _png).is_file():
            display(IPImage(filename=str(_via_out / _png)))
    print(f"  written to {_via_out}")


# COMMAND ----------

# DBTITLE 1,Nyquist / Bode — Condition and View drop-downs
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

_nyq_figs = {}
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
            _nyq_figs[cond] = viewers.img_html(_nyq)
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
    _nyq_figs[cond] = fig
    print(f"  {cond}: {n_seg} segments plotted"
          + (f" + {_n_rebuilt} rebuilt from neighbours" if _n_rebuilt else "")
          + (f", all {_n_finite_total} silver-accepted points shown"
             if not _n_hidden_total else
             f", {_n_finite_total - _n_hidden_total} of {_n_finite_total} "
             f"points inside the display rule "
             f"({_n_hidden_total} drawn as grey \u00d7)"))

# ONE plot: pick the condition and the view (Nyquist, |Z|, phase)
show_by_condition(_nyq_figs, title=f'Local EIS spectra — Leepa {LEEPA}')

# COMMAND ----------

# DBTITLE 1,Gamry vs pipeline aggregate — Condition and View drop-downs
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
 
_ov_figs = {}
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
    # The comparison uses only the band BOTH instruments covered, so the
    # Gamry curve used to stop where the local band stops (~4 kHz) although
    # the sweep goes to 30 kHz. GAMRY_FULL_RANGE draws the rest of the sweep
    # as a separate dashed trace -- shown, not compared.
    GAMRY_FULL_RANGE = globals().get('GAMRY_FULL_RANGE', True)
    gamry_full = None
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
        gamry_full = gamry_df
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
 
    # Gamry outside the local band (dashed grey): the rest of the sweep, to
    # its own top frequency (30 kHz), with the same corrections applied
    if GAMRY_FULL_RANGE and gamry_full is None:
        gamry_full = gamry_df
    if GAMRY_FULL_RANGE and gamry_full is not None and gamry_df is not None \
            and len(gamry_full) > len(gamry_df):
        _in = gamry_full['freq_hz'].between(gamry_df['freq_hz'].min(),
                                            gamry_df['freq_hz'].max())
        _ext = gamry_full.copy()
        _ext.loc[_in & ~_ext['freq_hz'].isin([gamry_df['freq_hz'].min(),
                                              gamry_df['freq_hz'].max()]),
                 ['z_re', 'z_im']] = np.nan
        Z_e = _ext['z_re'].values + 1j * _ext['z_im'].values
        _fmax = gamry_full['freq_hz'].max()
        fig.add_trace(go.Scatter(
            x=_ext['z_re'], y=-_ext['z_im'], mode='lines+markers',
            line=dict(width=2, color='grey', dash='dash'),
            marker=dict(size=4, color='grey', symbol='diamond-open'),
            name=f'Gamry {cond} outside local band (to {_fmax/1e3:.0f} kHz)',
            legendgroup='gamry_full', connectgaps=False,
            customdata=_ext['freq_hz'].values,
            hovertemplate=f"Gamry {cond} (not compared)<br>f=%{{customdata:.0f}} Hz"
                "<br>Z'=%{x:.1f}<br>-Z''=%{y:.1f}<extra></extra>",
        ), row=1, col=1)
        for _col, _y in ((2, np.abs(Z_e)), (3, np.degrees(np.angle(Z_e)))):
            fig.add_trace(go.Scatter(x=_ext['freq_hz'], y=_y, mode='lines+markers',
                line=dict(width=2, color='grey', dash='dash'),
                marker=dict(size=3, color='grey'), connectgaps=False,
                legendgroup='gamry_full', showlegend=False), row=1, col=_col)

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
    _ov_figs[cond] = fig
 
    print(f"\n  {cond}: {n_seg} measured"
          + (f" + {_nrec} rebuilt from neighbours" if _nrec else "")
          + f" segments from {src}")
    if _cov is not None:
        print(f"  aggregate covers {100*_cov:.1f} % of the plate area")
    if gamry_df is not None:
        print(f"  Gamry: {len(gamry_df)} pts")

# ONE plot: pick the condition and the view (Nyquist, |Z|, phase)
show_by_condition(_ov_figs, title=f'Gamry vs pipeline aggregate — Leepa {_LEEPA}')

# COMMAND ----------
# DBTITLE 1,Plate heat maps — Condition / Parameter / View
# ═══════════════════════════════════════════════════════════════════════════════
# PLATE HEAT MAPS — ONE plot, chosen from three drop-downs:
#   Condition   the conditions the widgets selected (never others)
#   Parameter   HFR, Re Z at 1 kHz, R_ct, R_mt, R_pol, current density,
#               |Z| and phase at 100 Hz, temperature, chain lag
#   View        "2D spatial (interpolated)": linear interpolation between the
#               segment centres, a square on every measured segment, parula
#               with a soft gloss -- the bench's MATLAB layout;
#               "Values per segment": every segment filled, value printed
# Hover any square / segment for its number and value. Colour scales are the
# fixed per-parameter scales of config.HEATMAP_LIMITS.
#
# WRITE_STATIC_MAPS writes the thesis PNGs (plate_<param>.png and
# plate_<param>_interp.png, HEATMAP_STYLE) into each run's gold folder
# WITHOUT displaying them.
# ═══════════════════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WRITE_STATIC_MAPS = True
_SNR = f'{MIN_SNR_DB:g}'

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


_title = f'{_LEEPA} / SNR {_SNR or "default"}'
displayHTML(viewers.plate_maps({c: csv for c, (csv, _p) in sorted(_COND_GOLD.items())},
                               title=_title))

if WRITE_STATIC_MAPS:
    for cond, (gold_csv, prov) in sorted(_COND_GOLD.items()):
        _flds = viewers.plate_fields(gold_csv)
        _classes = _flds[0].classes if _flds else {}
        for fd in _flds:
            for _rd in plate_figure.renders(HEATMAP_STYLE):
                fig = plate_figure.draw_flow_plate(
                    fd.values, fd.key, classes=_classes, label=fd.label,
                    unit=fd.unit, decimals=fd.decimals, render=_rd,
                    title=f'{_LEEPA} / {cond} / SNR {_SNR or "default"}')
                fig.savefig(str(gold_csv.parent / f'plate_{fd.key}{plate_figure.suffix(_rd)}.png'),
                            dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
                plt.close(fig)
        print(f'  {cond}: static maps written to {gold_csv.parent}')

# COMMAND ----------

# DBTITLE 1,MF4 Test-Stand Parameters: all channels for the selected order
# ═══════════════════════════════════════════════════════════════════════════════
# MF4 TEST-STAND PARAMETERS — every recorded channel for the selected order
#
# Reads the .mf4 measurement file(s) from the Gamry Volume, filters by the
# selected Leepa order ID, and draws ONE interactive figure (bench_plots.py):
#
#   * the "parameter" dropdown (top right) picks what is drawn: a single
#     parameter, e.g. "Temperature · Coolant inlet (T_Si_CL)", or a whole
#     group that shares a unit, e.g. "Temperature — all [°C]"
#       1. Electrical  (voltage, current, power, HFR)
#       2. Temperature (inlet/outlet anode, cathode, coolant)
#       3. Flow rates  (H2, air, N2, coolant)
#       4. Pressure    (inlet/outlet anode, cathode, coolant)
#       5. Humidity    (RH anode/cathode, dew points)
#       6. Other channels in the file (MF4_INCLUDE_ALL_CHANNELS)
#   * the legend toggles: click an entry to hide/show it, double-click to see
#     it alone
#   * move the cursor over the plot to read every visible value at that time;
#     drag the range slider under it to zoom in time
#
# The FAMOS plate temperature sensors (temp1..temp4, from bronze for each
# selected condition) are drawn as dotted reference lines on every °C view,
# so the plate temperature can be read against the bench's coolant and gas
# temperatures directly. temp1 sits at x = 0 (H2-inlet / air-outlet end),
# temp4 at x = 252 mm (air-inlet and coolant-inlet end) -- see
# config.COOLANT_INLET_END.
# ═══════════════════════════════════════════════════════════════════════════════
try:
    import asammdf  # noqa: F401
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'asammdf'])

import json
from pathlib import Path

import bench_plots

_MF4_DIR = Path('/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev/Gamry')
_LEEPA_MF4 = str(_widget('leepa_id', 'LEEPA', 'leepa'))
MF4_INCLUDE_ALL_CHANNELS = True   # False: only the channels in the groups above
MF4_MAX_POINTS = 3000             # per channel; min/max kept, spikes survive

# ── Find MF4 files for the selected order ─────────────────────────────────────
_mf4_files = sorted(
    [f for f in _MF4_DIR.glob('*.mf4') if f'RO{_LEEPA_MF4}' in f.name]
    + [f for f in _MF4_DIR.glob('*.MF4') if f'RO{_LEEPA_MF4}' in f.name]
)

if not _mf4_files:
    print(f'  No MF4 files found for order {_LEEPA_MF4} in {_MF4_DIR}')
else:
    print(f'  Order {_LEEPA_MF4}: {len(_mf4_files)} MF4 file(s)')
    for _f in _mf4_files:
        print(f'    {_f.name}  ({_f.stat().st_size / 1e6:.1f} MB)')

    # Use the LARGEST file (the main measurement, not Anfang/Ende snapshots)
    _main_mf4 = max(_mf4_files, key=lambda f: f.stat().st_size)
    print(f'\n  Reading: {_main_mf4.name}')
    _t0 = bench_plots.mf4_start_time(_main_mf4)

    _chans = bench_plots.read_mf4(_main_mf4,
                                  include_others=MF4_INCLUDE_ALL_CHANNELS,
                                  max_points=MF4_MAX_POINTS)
    _by_group = {}
    for _c in _chans:
        _by_group.setdefault(_c.group, []).append(_c.name)
    for _g, _names in _by_group.items():
        print(f'  {_g}: {len(_names)} channel(s)')
    _wanted = {n for _g in bench_plots.CHANNEL_GROUPS.values() for n, _, _ in _g}
    _absent = sorted(_wanted - {c.name for c in _chans})
    if _absent:
        print(f'  not in this file: {", ".join(_absent)}')

    # ── FAMOS plate sensors for the selected condition(s), as °C references ──
    _refs = []
    for _cond in selected_conditions():
        try:
            _d, _ = result_dir(_LEEPA_MF4, _cond)
            _man = Path(_d) / 'bronze' / 'bronze_manifest.json' if _d else None
            _sens = (json.loads(_man.read_text()).get('sensor_T_degC', {})
                     if _man is not None and _man.is_file() else {})
        except Exception as _e:                                # noqa: BLE001
            _sens = {}
        for _k in sorted(_sens):
            _x = geom.TEMP_SENSOR_X_MM.get(_k)
            _refs.append((f'FAMOS {_k} ({_x:.0f} mm) — {_cond}'
                          if _x is not None else f'FAMOS {_k} — {_cond}',
                          '°C', float(_sens[_k])))
    if _refs:
        print(f'  FAMOS plate sensors overlaid: {len(_refs)} line(s) '
              f'({", ".join(selected_conditions())})')

    _fig = bench_plots.bench_figure(
        _chans, start='Temperature', reference_lines=_refs,
        title=(f'{_LEEPA_MF4} — {_main_mf4.name}'
               + (f' (start {_t0:%Y-%m-%d %H:%M:%S})' if _t0 else '')))
    displayHTML(_fig.to_html(full_html=False, include_plotlyjs='cdn'))
    print(f'\n  Done — {_main_mf4.name}: {len(_chans)} channels; pick one in '
          f'the "parameter" dropdown, click legend entries to toggle them')

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

# DBTITLE 1,ECM fit per segment — Condition and View drop-downs
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
 
 
_ecm_figs = {}
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
 
    _ecm_figs[cond] = fig

show_by_condition(_ecm_figs, title=f'ECM fit per segment — Leepa {LEEPA}')

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
 
_ecm_agg_figs = {}
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
    _ecm_agg_figs[cond] = fig
 
    AGG_ECM[cond] = {'Rs_mohm_cm2': _rs, 'R_pol_mohm_cm2': _rpol,
                     'R_ct_mohm_cm2': a['R_ct'] * 1000,
                     'R_mt_mohm_cm2': a['R_mt'] * 1000,
                     'n_arcs': a['n_arcs'], 'chi2_nu': a['chi2_nu'],
                     'verdict': a['verdict'],
                     'max_res_pct': float(a['res_pct'].max())}

show_by_condition(_ecm_agg_figs, title=f'ECM fit of the aggregate — Leepa {LEEPA}')

# COMMAND ----------

# DBTITLE 1,ECM parameter tables — per segment, and aggregate vs segment median
_ecm_tables = {}
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
 
    _ecm_tables[cond] = viewers.table_html(df_p)

if _ecm_tables:
    displayHTML(viewers.selector_html({(c,): h for c, h in _ecm_tables.items()},
                                      ['Condition'], title='ECM parameters per segment'))

# COMMAND ----------

# DBTITLE 1,ECM parameter plate maps (Rs, R_ct, R_pol from the fit above)
# ═══════════════════════════════════════════════════════════════════════════════
# The same ECM parameters, laid out on the plate.  Driven off ECM_ALL, so the
# maps and the Nyquist overlays can never disagree about what was fitted.
#
# Drawn with plate_figure (the viewer's plate layout with gas and coolant
# ports and flow arrows): the real segment outlines, the fitted VALUE printed
# inside every segment, and the same Jet colours (plate_style) on the same
# FIXED scale as the pipeline's own maps above (config.HEATMAP_LIMITS) -- so an
# ECM Rs map and a pipeline HFR map can be read side by side. (This cell used to draw one dot
# per segment centroid on an RdYlGn ramp, which showed neither the segment
# shapes nor the numbers.)
# ═══════════════════════════════════════════════════════════════════════════════
import numpy as np
import plate_plotly

# Condition / Parameter / View drop-downs, as for the pipeline maps; the
# colour scale is the pipeline's fixed scale for the same quantity
# (Rs -> R_ohmic, R_ct, R_pol), so the two can be read side by side.
_ecm_maps = {}
for cond, blob in ECM_ALL.items():
    seg_fits = blob['segments']
    if not seg_fits:
        continue
    _flds = []
    for key, label, fn in (
            ('R_ohmic', 'Rs (ECM)',    lambda r: r['params']['Rs'] * 1000),
            ('R_ct',    'R_ct (ECM)',  lambda r: r['R_ct'] * 1000),
            ('R_pol',   'R_pol (ECM)', lambda r: r['R_pol'] * 1000)):
        vals = {str(s): fn(r) for s, r in seg_fits.items()}
        vals = {s: v for s, v in vals.items()
                if s in geom.SEGMENTS and np.isfinite(v)}
        if len(vals) >= 3:
            _flds.append(plate_plotly.Field(key, label, 'mΩ·cm²', vals))
    if _flds:
        _ecm_maps[cond] = _flds
if _ecm_maps:
    displayHTML(viewers.plate_maps(_ecm_maps, title=f'Leepa {LEEPA} — ECM'))

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

# DBTITLE 1,Polarisation curve — FAMOS (local EIS, whole plate + flow bands)
# ═══════════════════════════════════════════════════════════════════════════════
# POLARISATION CURVE FROM THE FAMOS RECORDINGS
#
# One point per load condition that was run above:
#   voltage  = DC level of the cell-voltage channel (POLCURVE_REF_CHANNEL,
#              default UC2) on every card, median over the cards
#   current  = the measured segment currents of that run
#              (gold_manifest.json -> dc_closure), per cm2 of measured area
# Because the plate is segmented, the same voltage also gives a LOCAL curve
# per flow band -- air outlet / middle / air inlet -- from plate_summary.csv.
# That is the part a whole-cell polcurve cannot show.
#
# If the reference channel is AC-coupled (a DC level near 0 V), the voltage
# check below says so. Enter the cell voltages by hand in POLCURVE_V_OVERRIDE
# (in V), e.g. from the bench log (U_S) or the Gamry cell below.
# Drawing and reading: polcurve.py.
# ═══════════════════════════════════════════════════════════════════════════════
import importlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from IPython.display import display
import polcurve
importlib.reload(polcurve)

POLCURVE_REF_CHANNEL = 'UC2'
POLCURVE_V_OVERRIDE = {}          # e.g. {'45A': 0.862, '60A': 0.845} in V
POLCURVE_BANDS = True             # also draw the three flow-band curves
POLCURVE_LABEL_AT = (0.1, 1.5)    # A/cm2 where the voltage is printed

_pc_runs = {}
for cond, pr in PIPELINE_RESULTS.items():
    if pr is None:
        continue
    try:
        _pc_files = bronze.discover_files(
            DEFAULT.replace(dat_dir=_DAT_DIR, leepa=LEEPA, condition=cond))
    except SystemExit:
        _pc_files = []
    _pc_runs[cond] = (_pc_files, Path(pr['out_dir']))

if not _pc_runs:
    print('  no pipeline results in this session -- run the pipeline cell first')
else:
    PC_FAMOS, _pc_rows = polcurve.famos_polcurves(
        _pc_runs, ref_channel=POLCURVE_REF_CHANNEL, bands=POLCURVE_BANDS,
        label=f'FAMOS {LEEPA}', v_override=POLCURVE_V_OVERRIDE)
    _pc_df = pd.DataFrame(_pc_rows)
    display(_pc_df.round(4))

    # A cell voltage outside 0.2..1.2 V is not a cell voltage: the channel is
    # AC-coupled, scaled, or not the cell-voltage monitor.
    _bad = _pc_df[~_pc_df.v_cell_mV.between(200, 1200)]
    if len(_bad):
        print(f"  WARNING: {POLCURVE_REF_CHANNEL} gives "
              + ", ".join(f"{r.condition} = {r.v_cell_mV:.0f} mV"
                          if np.isfinite(r.v_cell_mV)
                          else f"{r.condition} = no FAMOS file found"
                          for r in _bad.itertuples())
              + " -- not a plausible cell voltage. This channel probably "
                "carries no DC level; fill POLCURVE_V_OVERRIDE (in V) from "
                "the bench log or the Gamry cell below and re-run this cell.")
    else:
        fig = polcurve.plot_polcurves(
            PC_FAMOS, title=f'Polcurve {LEEPA} — FAMOS',
            label_at=POLCURVE_LABEL_AT)
        _pc_out = _TMP_BASE / LEEPA / 'polcurve_famos.png'
        _pc_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(_pc_out, dpi=200, bbox_inches='tight')
        _pc_df.to_csv(_pc_out.with_suffix('.csv'), index=False)
        display(fig)
        plt.close(fig)
        print(f'  saved: {_pc_out}  (+ .csv)')

# COMMAND ----------

# DBTITLE 1,Polarisation curve — Gamry (whole cell, Vdc of every sweep)
# ═══════════════════════════════════════════════════════════════════════════════
# POLARISATION CURVE FROM THE GAMRY SWEEPS
#
# One point per .dta sweep of this order (the files the Gamry validation
# uses, found by gamry_files):
#   voltage  = the Vdc column of the ZCURVE table (median over the sweep)
#   current  = the set point in the file name (..._CurrVal_150.dta), because
#              the sweeps run with IDCREQ = 0 -- the bench holds the DC, the
#              potentiostat adds only the AC -- so Idc reads ~0. A measured
#              Idc is used instead whenever it carries the load.
#   HFR      = the sweep's own high-frequency intercept (extrapolated when the
#              sweep stops while still capacitive), for the iR-free curve.
#
# Overlays: reference curves from CSV in POLCURVE_REFERENCE_CSVS (columns
# "current density [A/cm2]" and "voltage [mV or V]"), and the FAMOS plate
# curve from the cell above.
# ═══════════════════════════════════════════════════════════════════════════════
import importlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
import config
import polcurve
importlib.reload(polcurve)

POLCURVE_N_CELLS = 1              # divide Vdc by this for a short stack
POLCURVE_SHOW_IR_FREE = True      # V + j*HFR, dashed
POLCURVE_OVERLAY_FAMOS = True     # plate curve from the FAMOS cell above
POLCURVE_REFERENCE_CSVS = []      # [('/Volumes/.../PC_03_C3.1_Ref.csv', 'PC_03_C3.1_Ref_2511414_HTL')]
POLCURVE_LABEL_AT = (0.1, 1.5)

_pc_dta, _pc_build = gamry_files(LEEPA)
if not _pc_dta:
    print(f'  no Gamry .dta sweeps found for {LEEPA} -- check GAMRY_SEARCH_ROOTS')
else:
    PC_GAMRY, _pcg_rows = polcurve.gamry_polcurve(
        _pc_dta, label=f'Gamry {LEEPA}', area_cm2=config.A_CELL_CM2,
        n_cells=POLCURVE_N_CELLS)
    _pcg_df = pd.DataFrame(_pcg_rows)
    print(f'  {len(_pc_dta)} sweep(s), build {_pc_build or "UNKNOWN"}, '
          f'area {config.A_CELL_CM2:.2f} cm2')
    display(_pcg_df.round(4))

    _curves = [PC_GAMRY]
    if POLCURVE_SHOW_IR_FREE and np.isfinite(
            PC_GAMRY.extra.get('hfr_mohm_cm2', np.array([np.nan]))).all():
        _irf = PC_GAMRY.ir_free()
        _irf.show_labels = False
        _curves.append(_irf)
    if POLCURVE_OVERLAY_FAMOS and 'PC_FAMOS' in globals() and PC_FAMOS:
        import dataclasses as _dc
        # own colour: the FAMOS cell draws its plate curve in the first
        # series colour, which is Gamry's here
        _curves.append(_dc.replace(PC_FAMOS[0], color=polcurve.PC_COLORS[2],
                                   dashed=True))
    for _p, _lab in POLCURVE_REFERENCE_CSVS:
        _curves.append(polcurve.load_curve_csv(_p, _lab))

    if not np.size(PC_GAMRY.j):
        print('  no usable point: no Vdc column or no set point in the file names')
    else:
        fig = polcurve.plot_polcurves(
            _curves, title=f'Polcurve {LEEPA} — Gamry',
            label_at=POLCURVE_LABEL_AT)
        _pcg_out = _TMP_BASE / LEEPA / 'polcurve_gamry.png'
        _pcg_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(_pcg_out, dpi=200, bbox_inches='tight')
        _pcg_df.to_csv(_pcg_out.with_suffix('.csv'), index=False)
        display(fig)
        plt.close(fig)
        print(f'  saved: {_pcg_out}  (+ .csv)')
