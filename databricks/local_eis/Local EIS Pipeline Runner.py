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
import abgleich

# Force reload during development
for mod in [config, utils, eis_local, bronze, silver, gold, pipeline_main,
            geom, csv_source, csv_pipeline, gamry_dta, abgleich]:
    importlib.reload(mod)

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
    CONDITIONS = sorted(_all_conds) if _all_conds else ['450A', '60A', '45A', '150A']
except Exception:
    CONDITIONS = ['450A', '60A', '45A', '150A']

# Gamry reference: auto-discover per Leepa (may not exist for every order)
try:
    _gamry_candidates = sorted(_EV_ROOT.glob(f'*{LEEPA}*Gamry*')) + sorted(_EV_ROOT.glob(f'*Gamry*{LEEPA}*'))
    GAMRY_ROOT = _gamry_candidates[0] if _gamry_candidates else None
except Exception:
    GAMRY_ROOT = None

# ─── Remove old widgets that are no longer needed ───
for _old in ('plate', 'csv_path', 'csv_dialect', 'csv_tones',
             'gain_file', 'gamry_dir', 'bench_log'):
    try:
        dbutils.widgets.remove(_old)
    except Exception:
        pass

# ─── Create widgets (only the ones that matter) ───
_default = '2612025' if '2612025' in AVAILABLE_ORDERS else AVAILABLE_ORDERS[-1]
try:
    dbutils.widgets.dropdown('leepa_id', _default, AVAILABLE_ORDERS, 'Order ID (Leepa)')
    dbutils.widgets.dropdown('condition', 'ALL', ['ALL'] + CONDITIONS, 'Condition')
    dbutils.widgets.text('f_min_hz', '0.15', 'F min (Hz)')
    dbutils.widgets.text('f_max_hz', '4500.0', 'F max (Hz)')
    dbutils.widgets.dropdown('min_snr_db', '0',
                             ['-30','-20', '-10', '-3', '0', '3', '5', '8'],
                             'Min SNR (dB)')
    dbutils.widgets.dropdown('source_format', 'famos',
                             ['famos', 'csv'], 'Measurement file format')
    # stop_after:
    #   bronze = raw phasor extraction only (fastest, for debugging)
    #   silver = + de-skew + measurement model + lin-KK validation
    #   gold   = + spatial field, plate maps, plausibility (full run)
    dbutils.widgets.dropdown('stop_after', 'gold',
                             ['bronze', 'silver', 'gold'], 'Stop after')
except Exception:
    pass


def _w(name, default=''):
    try:
        return dbutils.widgets.get(name)
    except Exception:
        return default


# ─── Read widget values ───
PLATE = 'gen1'  # always gen1 for FAMOS
SOURCE_FORMAT = _w('source_format', 'famos')
CSV_PATH = ''
CSV_DIALECT = 'auto'
CSV_TONES = ()
GAIN_FILE = ''
GAMRY_DIR = str(GAMRY_ROOT) if GAMRY_ROOT else ''
BENCH_LOG = ''
LEEPA = _w('leepa_id', _default)
COND_FILTER = _w('condition', 'ALL')
F_MIN = float(_w('f_min_hz', '0.15'))
F_MAX = float(_w('f_max_hz', '4500.0'))
MIN_SNR_DB = float(_w('min_snr_db', '0'))
STOP_AFTER = _w('stop_after', 'gold')

# Select the plate for the whole session
_plate = geom.use_plate(PLATE)

print(f"  Leepa:     {LEEPA}")
print(f"  Condition: {COND_FILTER}")
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
    print(f"\n  ✓ Bronze patched: FamosFile now reads from datago")
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

# Use a user-specific temp base to avoid permission conflicts on shared cluster
_TMP_BASE = Path(tempfile.gettempdir()) / f'eis_{os.getuid()}'
_TMP_BASE.mkdir(parents=True, exist_ok=True)

# ─── Persistent Volume cache ───
# Pipeline results are expensive (5-10 min per condition).  After a successful
# run the bronze/silver/gold CSVs are copied to a UC Volume so that every
# future session can skip the pipeline and go straight to plotting.
_CACHE_VOL = Path('/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev/EIS_Results')
FORCE_RERUN = True          # set True to ignore cache and re-run from scratch


def _cache_dir(leepa, cond, snr=None):
    base = _CACHE_VOL / leepa / cond
    if snr is not None:
        return base / f'snr_{snr}'
    return base


def _cache_exists(leepa, cond, snr=None):
    """True if usable cached results exist on the Volume."""
    cd = _cache_dir(leepa, cond, snr)
    if not cd.exists():
        return False
    # Check for silver spectra (the minimum needed for Nyquist plots)
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
    # Also copy any top-level files (manifests, PNGs)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(str(f), str(dst / f.name))
            n_copied += 1
    return n_copied


def _load_from_cache(leepa, cond, snr=None):
    """Return a PIPELINE_RESULTS-compatible dict pointing to the cached dir."""
    cd = _cache_dir(leepa, cond, snr)
    return {
        'manifest': {},    # no live manifest, but plots only need out_dir
        'cfg': None,
        'out_dir': cd,
        'cached': True,
    }

# Determine which conditions to run.  A CSV measurement is one file, so it is
# one "condition" -- named after the file rather than after a current
# setpoint, because the file is what identifies it.
if SOURCE_FORMAT == 'csv':
    _conditions_to_run = [Path(CSV_PATH).stem or 'csv']
elif COND_FILTER == 'ALL':
    _conditions_to_run = CONDITIONS  # e.g. ['150A', '450A', '45A', '60A']
else:
    _conditions_to_run = [COND_FILTER]

print(f"  DAT dir:    {_DAT_DIR}")
print(f"  Curr cal:   {_CURR_CAL}")
print(f"  Gamry ref:  {GAMRY_DIR or '(none - whole-cell check skipped)'}")
print(f"  Conditions: {_conditions_to_run}")
print(f"  Band:       {F_MIN} – {F_MAX} Hz")
print(f"  Stop after: {STOP_AFTER}\n")

# ─── Run pipeline once per condition (or load from cache) ───
PIPELINE_RESULTS = {}  # {condition_str: manifest_dict}

for cond in _conditions_to_run:
    # ── Check cache first ──
    if not FORCE_RERUN and _cache_exists(LEEPA, cond, MIN_SNR_DB):
        PIPELINE_RESULTS[cond] = _load_from_cache(LEEPA, cond, MIN_SNR_DB)
        _cd = _cache_dir(LEEPA, cond, MIN_SNR_DB)
        _sp = spectra_csv(_cd)
        _n = 0
        if _sp and _sp.exists():
            import pandas as _pd
            _n = _pd.read_csv(_sp)['segment'].nunique()
        print(f"  ✓ {cond}: CACHED on Volume ({_n} segments) — skipping pipeline")
        print(f"    {_cd}")
        continue

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
        min_ref_channels=1,  # allow single-card measurements (e.g. 2612025)
        min_snr_db=MIN_SNR_DB,    # from widget (default 0 dB; raise to 5-8 for cleaner plots)
        align_min_prominence=15.0, # lowered from 25: 25 kHz cards score 21-23 prominence
        snr_floor_db=-40.0,        # lowered from -3: 25 kHz cards have -20 to -35 dB SNR above 120 Hz
        max_drift=2.5,             # lowered from 0.25: HF dwells on short records are not stationary
        max_thd=1.0,               # loosened from 0.10: allow noisy HF points through for display
    )

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
        try:
            _nc = _save_to_cache(_out_dir, LEEPA, cond, MIN_SNR_DB)
            print(f"  💾 Saved {_nc} files to Volume cache: {_cache_dir(LEEPA, cond, MIN_SNR_DB)}")
        except Exception as _ce:
            print(f"  ⚠ Cache save failed (results still in /tmp): {_ce}")
        # The FAMOS manifest reports per stage; the CSV manifest is flat.
        gs = manifest.get('stages', {}).get('gold', {})
        if gs:
            n_meas = gs.get('n_measured', '?')
            n_total = gs.get('n_total', '?')
            r_ohmic = gs.get('R_ohmic', {})
            if r_ohmic:
                print(f"  ✓ {cond}: {n_meas}/{n_total} segments | "
                      f"Rs = {1000*r_ohmic['mean']:.1f} ± {1000*r_ohmic['sd']:.1f} mΩ·cm²")
            else:
                print(f"  ✓ {cond}: {n_meas}/{n_total} segments")
        else:
            sch = manifest.get('schedule', {})
            print(f"  ✓ {cond}: {manifest.get('n_segments', '?')} segments, "
                  f"{manifest.get('n_ecm_ok', '?')} ECM fits, "
                  f"KK {manifest.get('kk_pass', '?')}/{manifest.get('kk_total', '?')}"
                  f" | excitation: {sch.get('mode', '?')}, "
                  f"{len(sch.get('tones', []))} tones")
    except Exception as e:
        print(f"  ✗ {cond}: FAILED — {e}")
        PIPELINE_RESULTS[cond] = None
    # Release memory between conditions to prevent driver OOM
    gc.collect()
    spark.catalog.clearCache()

print(f"\n{'═'*75}")
print(f"  PIPELINE COMPLETE — {len([v for v in PIPELINE_RESULTS.values() if v])} conditions processed")
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

for cond, pr in PIPELINE_RESULTS.items():
    if pr is None:
        continue
    out_dir = pr['out_dir']
    
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
        
        # Physical cleaning (same as old Gold Table pipeline):
        # Reject non-physical points (Z' < 0 at HF = phase artifact)
        keep = np.isfinite(zr) & np.isfinite(zi)
        keep &= (zr > 0)                          # must be positive real
        keep &= (zr >= 1.0) & (zr <= 500.0)       # plausible range mΩ·cm²
        keep &= (f <= 2000.0)                      # display up to 2 kHz
        cap = f <= 1000.0
        keep &= ~(cap & (-zi < -5.0))              # no inductive below 1kHz
        f, zr, zi = f[keep], zr[keep], zi[keep]
        
        if len(f) < 3:
            continue
        
        clr = colors[i]
        seg_name = f'Seg {seg}'
        
        # Nyquist — show frequency on hover
        fig.add_trace(go.Scatter(
            x=zr, y=-zi,
            mode='markers+lines', marker=dict(size=4, color=clr),
            line=dict(width=1, color=clr),
            name=seg_name, legendgroup=seg_name, showlegend=True,
            hovertemplate=(f'<b>Seg {seg}</b><br>'
                           'f = %{customdata:.2f} Hz<br>'
                           "Z' = %{x:.1f} mΩ·cm²<br>"
                           "-Z'' = %{y:.1f} mΩ·cm²<extra></extra>"),
            customdata=f,
        ), row=1, col=1)
        
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
    
    fig.update_xaxes(title_text="Z' [mΩ·cm²]", row=1, col=1)
    fig.update_yaxes(title_text="-Z'' [mΩ·cm²]", row=1, col=1)
    fig.update_xaxes(title_text="f [Hz]", type="log", row=1, col=2)
    fig.update_yaxes(title_text="|Z| [mΩ·cm²]", type="log", row=1, col=2)
    fig.update_xaxes(title_text="f [Hz]", type="log", row=1, col=3)
    fig.update_yaxes(title_text="Phase [°]", row=1, col=3)
    
    fig.update_layout(
        title=f'<b>Local EIS — Leepa {LEEPA}, {cond} ({n_seg} segments)</b>',
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
    fig.show()
    print(f"  {cond}: {n_seg} segments plotted")

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
    _COND = dbutils.widgets.get('condition')
except Exception:
    _COND = '450A'

A_CELL_CM2 = 304.92
# Auto-detect Gamry folder for current Leepa
# Priority: per-Leepa folder > common Gamry folder (files matching Leepa ID)
_VOL_BASE = '/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev'
_GAMRY_CANDIDATES = [
    Path(f'{_VOL_BASE}/RO{_LEEPA}_Gamry'),      # per-Leepa folder
    Path(f'{_VOL_BASE}/{_LEEPA}_Gamry'),
    Path(f'{_VOL_BASE}/Gamry'),                  # shared folder for all Leepas
]
_GAMRY_VOL = next((p for p in _GAMRY_CANDIDATES if p.exists()), Path(f'{_VOL_BASE}/Gamry'))
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


# ─── 2. Load Gamry for all conditions ───
print(f"  Gamry source: {_GAMRY_VOL}")
_gamry = {}
for fp in sorted(list(_GAMRY_VOL.glob('*.DTA')) + list(_GAMRY_VOL.glob('*.dta'))):
    if '_Raw' in fp.stem:
        continue
    sp = _parse_dta(fp)
    if sp:
        k = A_CELL_CM2 * 1e3  # ohm -> mohm.cm2
        # Extract condition from CurrVal_{number} in filename → e.g. "60A"
        _cm = re.search(r'CurrVal_(\d+)', fp.stem)
        _gamry_key = (_cm.group(1) + 'A') if _cm else sp['name']
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
    candidates = [_CACHE_VOL / leepa / cond]
    try:
        pr = PIPELINE_RESULTS.get(cond)
        if pr:
            candidates.insert(0, Path(pr['out_dir']))
    except NameError:
        pass
    candidates.append(Path(f'/tmp/eis_{os.getuid()}/{leepa}/{cond}'))
    for d in candidates:
        for sub in ('silver', 'csv'):
            p = d / sub / 'spectra_clean.csv'
            if p.exists():
                seg = pd.read_csv(p)
                agg_p = d / 'silver' / 'cell_aggregate.csv'
                agg = pd.read_csv(agg_p) if agg_p.exists() else None
                if agg is not None:
                    agg = pd.DataFrame({
                        'freq_hz': agg['freq_hz'],
                        'z_re': agg['z_re_mohm_cm2'],
                        'z_im': agg['z_im_mohm_cm2'],
                    }).sort_values('freq_hz').reset_index(drop=True)
                    if np.nanmedian(agg['z_im']) > 0:
                        agg['z_im'] = -agg['z_im']
                return seg, agg, str(d)
    return None, None, None


# ─── 4. Plot overlay for each condition ───
_conds = [_COND] if _COND != 'ALL' else ['45A', '60A', '150A', '450A']

for cond in _conds:
    seg_df, agg_df, src = _load_pipeline(_LEEPA, cond)
    if seg_df is None:
        print(f"  {cond}: no pipeline results found")
        continue

    # Match Gamry condition
    gamry_df = _gamry.get(cond, None)

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
    fig.update_layout(
        title=f'<b>Leepa {_LEEPA} / {cond}: {n_seg} segments{_gstr}</b><br>'
              f'<sup>Source: {src}</sup>',
        height=580, width=1550,
        paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(t=100, b=55, r=280), hovermode='closest',
        legend=dict(font=dict(size=10), y=0.5, yanchor='middle'))
    fig.show()

    print(f"\n  {cond}: {n_seg} segments from {src}")
    if gamry_df is not None:
        print(f"  Gamry: {len(gamry_df)} pts")

# COMMAND ----------

# DBTITLE 1,Interactive Plate Heatmaps (plate_viewer)
# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE PLATE HEATMAPS — realistic segment geometry
#
# Uses the plate_viewer module from /master_thesis/map/ which renders the
# true pad-level segment outlines, flow ports, channel texture, and dual
# numbering.  Produces:
#   1. Self-contained interactive HTML (field switching, hover, click-to-pin)
#   2. Inline matplotlib (static, for the notebook scroll)
# ═══════════════════════════════════════════════════════════════════════════════
import sys, os
import numpy as np
import pandas as pd
from pathlib import Path
from IPython.display import display, HTML as IPHTML

# Ensure map/ module is importable
_map_dir = '/Workspace/Users/uum5fe@bosch.com/master_thesis/map'
if _map_dir not in sys.path:
    sys.path.insert(0, _map_dir)

from plate_model import PLATE, draw_plate
from plate_viewer import write_html, Field

#_CACHE_VOL = Path('/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev/EIS_Results')

# Field definitions: (csv_col, label, unit, colormap, decimals)
_FIELD_DEFS = [
    ('R_ohmic',      'HFR (Rs)',            'mohm.cm2', 'viridis',  1),
    ('R_ct',         'R_ct (charge transfer)', 'mohm.cm2', 'inferno',  1),
    ('R_mt',         'R_mt (mass transport)', 'mohm.cm2', 'magma',    1),
    ('R_pol',        'R_pol (total polarisation)', 'mohm.cm2', 'thermal', 1),
    ('j_dc',         'Current density',     'A/cm2',    'cividis',  3),
    ('Z_mag_100Hz',  '|Z| at 100 Hz',       'mohm.cm2', 'viridis',  1),
    ('phase_100Hz',  'Phase at 100 Hz',      'deg',      'humid',    1),
]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Resolve condition → gold CSV paths (live pipeline OR Volume cache) ──
try:
    _LEEPA = LEEPA
except NameError:
    _LEEPA = dbutils.widgets.get('leepa_id')

_COND_GOLD: dict[str, Path] = {}  # {cond: path to plate_summary.csv}

# 1. Live pipeline results (if kernel still has them)
try:
    for cond, pr in PIPELINE_RESULTS.items():
        if pr is None:
            continue
        p = Path(pr['out_dir']) / 'gold' / 'plate_summary.csv'
        if p.exists():
            _COND_GOLD[cond] = p
except NameError:
    pass  # kernel restarted — no PIPELINE_RESULTS

# 2. Fill gaps from Volume cache
_cache_leepa = _CACHE_VOL / _LEEPA
if _cache_leepa.exists():
    for cd in sorted(_cache_leepa.iterdir()):
        if cd.name in _COND_GOLD:
            continue
        p = cd / 'gold' / 'plate_summary.csv'
        if p.exists():
            _COND_GOLD[cd.name] = p

if not _COND_GOLD:
    print("  No gold results found (pipeline nor cache). Run cell 8 first.")

for cond, gold_csv in sorted(_COND_GOLD.items()):
    df = pd.read_csv(gold_csv)
    measured = df[df['class'] == 'measured']

    print(f"\n{'='*75}")
    print(f"  PLATE MAPS — {cond}  ({len(measured)}/{len(df)} measured)")
    print(f"{'='*75}")

    # ── Build Field objects for the interactive viewer ──
    _cols = set(df.columns)  # compute once, avoid repeated Analyze RPCs
    fields = []
    for col, label, unit, ramp, dec in _FIELD_DEFS:
        if col not in _cols:
            continue
        vals = {int(r['segment']): r[col]
                for _, r in df.iterrows()
                if pd.notna(r[col]) and np.isfinite(r[col])}
        if not vals:
            continue
        fields.append(Field(col, label, unit, ramp, dec, vals))

    # ── Write interactive HTML to Volume cache ──
    if fields:
        _html_out = _CACHE_VOL / _LEEPA / cond / 'gold' / 'plate_interactive.html'
        _html_out.parent.mkdir(parents=True, exist_ok=True)
        write_html(fields, _html_out,
                   title=f"{_LEEPA} / {cond}",
                   subtitle=f"{len(measured)} segments measured | plate gen1")
        print(f"  Interactive viewer: {_html_out}")

        # Show the HTML inline
        html_content = _html_out.read_text(encoding='utf-8')
        # Wrap in an iframe to keep it self-contained
        displayHTML(f'<iframe srcdoc="{html_content.replace(chr(34), "&quot;")}" '
                    f'width="100%" height="700" style="border:none;"></iframe>')

    # ── Static matplotlib for quick scroll ──
    for col, label, unit, ramp, dec in _FIELD_DEFS[:3]:  # R_ohmic, R_ct, R_mt
        if col not in _cols:
            continue
        vals = {int(r['segment']): r[col]
                for _, r in df.iterrows()
                if pd.notna(r[col]) and np.isfinite(r[col])}
        if not vals:
            continue
        fig, ax = draw_plate(vals, label=label, unit=unit, cmap=ramp)
        ax.set_title(f'{_LEEPA} / {cond} — {label}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        # Save to Volume
        _png_out = _CACHE_VOL / _LEEPA / cond / 'gold' / f'plate_{col}.png'
        fig.savefig(str(_png_out), dpi=200, bbox_inches='tight')
        display(fig)
        plt.close(fig)
        print(f"  {col}: saved to {_png_out}")

# COMMAND ----------

# DBTITLE 1,ECM Fit (all segments + aggregate): tau-parameterised, AICc arc count
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

# DBTITLE 1,ECM Visualisation: one figure, dropdown over all segments + aggregate
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

    displayHTML(fig.to_html(full_html=False, include_plotlyjs='cdn'))

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
    displayHTML(fig.to_html(full_html=False, include_plotlyjs='cdn'))

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
# ═══════════════════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

_seg_coords = {int(s): (c.cx_mm, c.cy_mm) for s, c in geom.SEGMENTS.items()}

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
        param_map = {k: v for k, v in param_map.items()
                     if k in _seg_coords and np.isfinite(v)}
        if len(param_map) < 3:
            print(f"  {cond}: too few segments with coordinates for {param_name}")
            continue

        segs = sorted(param_map)
        xs = [_seg_coords[s][0] for s in segs]
        ys = [_seg_coords[s][1] for s in segs]
        vals = [param_map[s] for s in segs]

        fig_h, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.set_facecolor('#f5f5f5')
        vmin, vmax = np.percentile(vals, 5), np.percentile(vals, 95)
        if vmax <= vmin:
            vmin, vmax = min(vals), max(vals) + 1e-9
        sc = ax.scatter(xs, ys, c=vals, cmap='RdYlGn_r',
                        norm=Normalize(vmin=vmin, vmax=vmax),
                        s=180, edgecolors='k', linewidths=0.5, zorder=3)
        for s, x, y in zip(segs, xs, ys):
            ax.annotate(f'{s}', (x, y), ha='center', va='center',
                        fontsize=6, fontweight='bold', zorder=4)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label(f'{param_name} [mΩ·cm²]')
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_title(f'{param_name} — Leepa {LEEPA}, {cond} ({len(segs)} segments)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        plt.close(fig_h)

    print(f"  {cond}: ECM plate maps rendered")

# COMMAND ----------

# DBTITLE 1,Summary Table (plate_summary.csv)
# ─── Summary per condition ───
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
    displayHTML(fig.to_html(full_html=False, include_plotlyjs='cdn'))

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
    displayHTML(fig.to_html(full_html=False, include_plotlyjs='cdn'))

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
    displayHTML(fig.to_html(full_html=False, include_plotlyjs='cdn'))

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
    displayHTML(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    break  # only show for first condition

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
    displayHTML(fig.to_html(full_html=False, include_plotlyjs='cdn'))

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
    displayHTML(fig2.to_html(full_html=False, include_plotlyjs='cdn'))

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

# DBTITLE 1,Gamry HFR: 3-Panel Nyquist + Bode (all conditions)
# ═══════════════════════════════════════════════════════════════════════════════
# GAMRY HFR — Interactive 3-Panel: Nyquist + Bode |Z| + Phase (all conditions)
# Same Plotly style as Cell 9 (pipeline Nyquist display)
# ═══════════════════════════════════════════════════════════════════════════════
import re
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

GAMRY_DIR = Path('/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev/RO2612025_Gamry')


def parse_gamry_dta(filepath):
    """Parse a Gamry .DTA file and return a dict with freq, Zreal, Zimag, Zmod, Zphz."""
    text = filepath.read_text(encoding='latin-1')
    match = re.search(r'ZCURVE\s+TABLE\s*\n([^\n]*\n){2}', text)
    if not match:
        raise ValueError(f"No ZCURVE TABLE found in {filepath.name}")
    data_text = text[match.end():]
    rows = []
    for line in data_text.strip().split('\n'):
        line = line.strip()
        if not line or not line[0].isdigit():
            if line.startswith('STOPABORT'):
                break
            continue
        line = line.replace(',', '.')
        parts = line.split('\t')
        if len(parts) >= 8:
            rows.append({
                'Freq':  float(parts[2]),
                'Zreal': float(parts[3]),
                'Zimag': float(parts[4]),
                'Zmod':  float(parts[6]),
                'Zphz':  float(parts[7]),
            })
    if not rows:
        raise ValueError(f"No data rows parsed from {filepath.name}")
    freq  = np.array([r['Freq']  for r in rows])
    zreal = np.array([r['Zreal'] for r in rows])
    zimag = np.array([r['Zimag'] for r in rows])
    zmod  = np.array([r['Zmod']  for r in rows])
    zphz  = np.array([r['Zphz']  for r in rows])
    return dict(freq=freq, Zreal=zreal, Zimag=zimag, Zmod=zmod, Zphz=zphz,
                name=filepath.stem, n_pts=len(rows),
                f_min=freq.min(), f_max=freq.max())


# ─── Parse all non-Raw .DTA files ───
dta_files = sorted(GAMRY_DIR.glob('*.DTA'))
processed = [f for f in dta_files if '_Raw' not in f.stem]

spectra = {}
for f in processed:
    try:
        sp = parse_gamry_dta(f)
        spectra[sp['name']] = sp
        print(f"  {sp['name']:15s}: {sp['n_pts']} pts, "
              f"{sp['f_min']:.2f}\u2013{sp['f_max']:.0f} Hz, "
              f"|Z| = {sp['Zmod'].min()*1e3:.3f}\u2013{sp['Zmod'].max()*1e3:.3f} m\u03a9")
    except Exception as e:
        print(f"  {f.stem}: FAILED \u2014 {e}")

print(f"\n  {len(spectra)} conditions parsed")

# ─── Plotly 3-panel per condition (matching Cell 9 style) ───
# Convert \u03a9 \u2192 m\u03a9 for display (Gamry raw is in \u03a9)
TO_MOHM = 1e3

cond_names = sorted(spectra.keys())
cond_colors = {
    '45A':        '#e74c3c',
    '60A':        '#e67e22',
    '150A':       '#2ecc71',
    '300A_10kHz': '#3498db',
    '450A':       '#9b59b6',
}
fallback_colors = [f'hsl({int(i*360/len(cond_names))}, 70%, 50%)'
                   for i in range(len(cond_names))]

# ─── Panel A: All conditions overlaid ───
fig = make_subplots(
    rows=1, cols=3,
    subplot_titles=['Nyquist', '|Z|(f) Bode', 'Phase(f)'],
    horizontal_spacing=0.06)

for idx, name in enumerate(cond_names):
    sp = spectra[name]
    f   = sp['freq']
    zr  = sp['Zreal'] * TO_MOHM
    zi  = sp['Zimag'] * TO_MOHM
    zmod = sp['Zmod'] * TO_MOHM
    zphz = sp['Zphz']
    clr = cond_colors.get(name, fallback_colors[idx])

    # Nyquist
    fig.add_trace(go.Scatter(
        x=zr, y=-zi,
        mode='markers+lines', marker=dict(size=4, color=clr),
        line=dict(width=1.2, color=clr),
        name=name, legendgroup=name, showlegend=True,
        hovertemplate=(f'<b>{name}</b><br>'
                       'f = %{customdata:.2f} Hz<br>'
                       "Z' = %{x:.4f} m\u03a9<br>"
                       "-Z'' = %{y:.4f} m\u03a9<extra></extra>"),
        customdata=f,
    ), row=1, col=1)

    # Bode |Z|
    fig.add_trace(go.Scatter(
        x=f, y=zmod,
        mode='lines+markers', marker=dict(size=3, color=clr),
        line=dict(width=1.2, color=clr),
        name=name, legendgroup=name, showlegend=False,
        hovertemplate=(f'<b>{name}</b><br>'
                       'f = %{x:.2f} Hz<br>'
                       '|Z| = %{y:.4f} m\u03a9<extra></extra>'),
    ), row=1, col=2)

    # Phase
    fig.add_trace(go.Scatter(
        x=f, y=zphz,
        mode='lines+markers', marker=dict(size=3, color=clr),
        line=dict(width=1.2, color=clr),
        name=name, legendgroup=name, showlegend=False,
        hovertemplate=(f'<b>{name}</b><br>'
                       'f = %{x:.2f} Hz<br>'
                       'Phase = %{y:.1f}\u00b0<extra></extra>'),
    ), row=1, col=3)

fig.update_xaxes(title_text="Z' [m\u03a9]", row=1, col=1)
fig.update_yaxes(title_text="-Z'' [m\u03a9]", row=1, col=1)
fig.update_xaxes(title_text="f [Hz]", type="log", row=1, col=2)
fig.update_yaxes(title_text="|Z| [m\u03a9]", type="log", row=1, col=2)
fig.update_xaxes(title_text="f [Hz]", type="log", row=1, col=3)
fig.update_yaxes(title_text="Phase [\u00b0]", row=1, col=3)

fig.update_layout(
    title=f'<b>Gamry Reference 3000 \u2014 All Conditions (RO2612025)</b>',
    height=550, width=1500,
    paper_bgcolor='white', plot_bgcolor='white',
    margin=dict(t=80, b=50, r=250),
    hovermode='closest',
    legend=dict(
        font=dict(size=10),
        itemclick='toggle', itemdoubleclick='toggleothers',
        y=0.5, yanchor='middle',
    ),
)
fig.show()

# ─── Panel B: one figure per condition ───
for idx, name in enumerate(cond_names):
    sp = spectra[name]
    f   = sp['freq']
    zr  = sp['Zreal'] * TO_MOHM
    zi  = sp['Zimag'] * TO_MOHM
    zmod = sp['Zmod'] * TO_MOHM
    zphz = sp['Zphz']
    clr = cond_colors.get(name, fallback_colors[idx])

    fig2 = make_subplots(
        rows=1, cols=3,
        subplot_titles=['Nyquist', '|Z|(f) Bode', 'Phase(f)'],
        horizontal_spacing=0.06)

    fig2.add_trace(go.Scatter(
        x=zr, y=-zi,
        mode='markers+lines', marker=dict(size=5, color=clr),
        line=dict(width=1.5, color=clr),
        name=name, showlegend=True,
        hovertemplate=(f'<b>{name}</b><br>'
                       'f = %{customdata:.2f} Hz<br>'
                       "Z' = %{x:.4f} m\u03a9<br>"
                       "-Z'' = %{y:.4f} m\u03a9<extra></extra>"),
        customdata=f,
    ), row=1, col=1)

    fig2.add_trace(go.Scatter(
        x=f, y=zmod,
        mode='lines+markers', marker=dict(size=4, color=clr),
        line=dict(width=1.5, color=clr),
        name=name, showlegend=False,
        hovertemplate=(f'<b>{name}</b><br>'
                       'f = %{x:.2f} Hz<br>'
                       '|Z| = %{y:.4f} m\u03a9<extra></extra>'),
    ), row=1, col=2)

    fig2.add_trace(go.Scatter(
        x=f, y=zphz,
        mode='lines+markers', marker=dict(size=4, color=clr),
        line=dict(width=1.5, color=clr),
        name=name, showlegend=False,
        hovertemplate=(f'<b>{name}</b><br>'
                       'f = %{x:.2f} Hz<br>'
                       'Phase = %{y:.1f}\u00b0<extra></extra>'),
    ), row=1, col=3)

    # HFR marker: x-axis intercept (min |Zimag|)
    idx_hfr = np.argmin(np.abs(zi))
    fig2.add_trace(go.Scatter(
        x=[zr[idx_hfr]], y=[-zi[idx_hfr]],
        mode='markers', marker=dict(size=12, color='red', symbol='x'),
        name=f'HFR = {zr[idx_hfr]:.4f} m\u03a9',
        hovertemplate=f'HFR = {zr[idx_hfr]:.4f} m\u03a9 @ {f[idx_hfr]:.1f} Hz<extra></extra>',
    ), row=1, col=1)

    fig2.update_xaxes(title_text="Z' [m\u03a9]", row=1, col=1)
    fig2.update_yaxes(title_text="-Z'' [m\u03a9]", row=1, col=1)
    fig2.update_xaxes(title_text="f [Hz]", type="log", row=1, col=2)
    fig2.update_yaxes(title_text="|Z| [m\u03a9]", type="log", row=1, col=2)
    fig2.update_xaxes(title_text="f [Hz]", type="log", row=1, col=3)
    fig2.update_yaxes(title_text="Phase [\u00b0]", row=1, col=3)

    fig2.update_layout(
        title=f'<b>Gamry {name} \u2014 {sp["n_pts"]} pts, '
              f'{sp["f_min"]:.2f}\u2013{sp["f_max"]:.0f} Hz</b>',
        height=500, width=1500,
        paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(t=80, b=50, r=200),
        hovermode='closest',
    )
    fig2.show()

# ─── Summary ───
print("\n" + "\u2550" * 75)
# The unit label is built OUTSIDE the f-string: an escape sequence inside
# the expression part of an f-string is a SyntaxError before Python 3.12,
# and Databricks runtimes are 3.10/3.11.
_u = "m\u03a9"
print(f"  {'Condition':15s}  {'R_hfr (' + _u + ')':>10s}  "
      f"{'|Z|_min (' + _u + ')':>12s}  {'|Z|_max (' + _u + ')':>12s}  "
      f"{'f_range (Hz)':>15s}")
print("\u2500" * 75)
for name in cond_names:
    sp = spectra[name]
    idx_hfr = np.argmin(np.abs(sp['Zimag']))
    r_hfr = sp['Zreal'][idx_hfr] * TO_MOHM
    print(f"  {name:15s}  {r_hfr:10.4f}  {sp['Zmod'].min()*TO_MOHM:12.4f}  "
          f"{sp['Zmod'].max()*TO_MOHM:12.4f}  {sp['f_min']:.2f}\u2013{sp['f_max']:.0f}")
print("\u2550" * 75)