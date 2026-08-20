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

# DBTITLE 1,Widgets: Order ID + Condition (from datago)
# ─── Discover available Order IDs from datago (same as old pipeline) ───
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

# Discover conditions for the FAMOS Volumes path
FAMOS_ROOT = Path('/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev/Famos')
try:
    _cond_dirs = sorted({p.name.split('_')[3] for p in FAMOS_ROOT.glob('Leepa_*_Current_*_Test_*_Karte_*.DAT')})
    CONDITIONS = _cond_dirs if _cond_dirs else ['450A', '60A', '45A', '150A']
except Exception:
    CONDITIONS = ['450A', '60A', '45A', '150A']

# ═══════════════════════════════════════════════════════════════════════════════
# THE TWO WIDGETS THAT CHANGE WHAT IS BEING MEASURED
#
#   Plate           gen1 = green / Kashyyyk,  gen2 = blue / Naboo.
#                   Same 45x20 pad grid, same 72 segments, DIFFERENT grouping
#                   of pads into segments 37..72.  Choosing wrong does not
#                   fail: it draws the right numbers on the wrong squares.
#
#   Source format   famos = the five-card imc .DAT recordings, which need the
#                   inter-card synchronisation stage;
#                   csv   = the newer single-file logger, which does not have
#                   a second clock and therefore must not run it.  The CSV
#                   route is a separate pipeline (csv_pipeline.py), not a
#                   second reader in front of the same one.
#
# For the R2-D2 logger, point 'CSV file / folder' at the SWEEP FOLDER -- the
# one holding metadata.csv and p1.csv, p2.csv, ...  Each point file carries a
# single frequency, so the spectrum only exists once they are read together;
# pointing at one file gives you that one frequency's plate maps and nothing
# to fit.
#
# There is no FAMOS recording of the gen2 plate yet, so "gen2 + famos" is
# rejected below rather than silently producing an empty run.
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Create widgets ───
_default = '2611976' if '2611976' in AVAILABLE_ORDERS else AVAILABLE_ORDERS[-1]
try:
    dbutils.widgets.dropdown('plate', 'gen1',
                             ['gen1', 'gen2'], 'Plate generation')
    dbutils.widgets.dropdown('source_format', 'famos',
                             ['famos', 'csv'], 'Measurement file format')
    dbutils.widgets.text('csv_path', '', 'CSV file / folder (csv only)')
    dbutils.widgets.dropdown(
        'csv_dialect', 'auto',
        ['auto', 'r2d2', 'r2d2_sweep', 'records', 'wide_time', 'long_time',
         'freq', 'gamry'],
        'CSV layout (csv only)')
    dbutils.widgets.text('csv_tones', '', 'CSV tones, comma separated (blank = detect)')
    dbutils.widgets.dropdown('leepa_id', _default, AVAILABLE_ORDERS, 'Order ID (Leepa)')
    dbutils.widgets.dropdown('condition', 'ALL', ['ALL'] + CONDITIONS, 'Condition')
    dbutils.widgets.text('f_min_hz', '0.15', 'F min (Hz)')
    dbutils.widgets.text('f_max_hz', '4500.0', 'F max (Hz)')
    dbutils.widgets.dropdown('stop_after', 'gold', ['bronze', 'silver', 'gold'], 'Stop after')
    dbutils.widgets.text('gain_file', '', 'Chain-response CSV (optional)')
except Exception:
    pass


def _w(name, default=''):
    try:
        return dbutils.widgets.get(name)
    except Exception:
        return default


# ─── Read widget values ───
PLATE = _w('plate', 'gen1')
SOURCE_FORMAT = _w('source_format', 'famos')
CSV_PATH = _w('csv_path').strip()
CSV_DIALECT = _w('csv_dialect', 'auto')
CSV_TONES = tuple(float(x) for x in _w('csv_tones').replace(';', ',').split(',')
                  if x.strip())
GAIN_FILE = _w('gain_file').strip()
LEEPA = _w('leepa_id', _default)
COND_FILTER = _w('condition', 'ALL')
F_MIN = float(_w('f_min_hz', '0.15'))
F_MAX = float(_w('f_max_hz', '4500.0'))
STOP_AFTER = _w('stop_after', 'gold')

# Select the plate for the whole session BEFORE anything reads geom.SEGMENTS.
_plate = geom.use_plate(PLATE)

if SOURCE_FORMAT == 'famos' and PLATE == 'gen2':
    raise ValueError(
        "There is no FAMOS recording of the gen2 (blue / Naboo) plate. "
        "Set 'Measurement file format' to csv and give a CSV path, or switch "
        "the plate back to gen1. Running gen2 against the gen1 .DAT files "
        "would map every channel onto the wrong piece of hardware.")
if SOURCE_FORMAT == 'csv' and not CSV_PATH:
    raise ValueError("Measurement file format is 'csv' but 'CSV file / folder' "
                     "is empty.")

print(f"  Plate:     {_plate.title}")
print(f"  Source:    {SOURCE_FORMAT}")
if SOURCE_FORMAT == 'csv':
    print(f"  CSV:       {CSV_PATH}  (layout: {CSV_DIALECT})")
    print(f"  Tones:     {'detect from the record' if not CSV_TONES else CSV_TONES}")
else:
    print(f"  Leepa:     {LEEPA}")
    print(f"  Condition: {COND_FILTER}")
    print(f"  Orders available: {AVAILABLE_ORDERS}")
print(f"  Band:      {F_MIN} – {F_MAX} Hz")
print(f"  Stop:      {STOP_AFTER}")
_gain_note = GAIN_FILE if GAIN_FILE else (
    "none — the measuring chain rolls off to -11° at 4.5 kHz and -24° at "
    "10 kHz; build one with gamry_dta.py from the Abgleich bode/ folder")
print(f"  Chain resp: {_gain_note}")

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
import tempfile
from IPython.display import display, Image as IPImage

_map_png = Path(tempfile.gettempdir()) / f'plate_{PLATE}.png'
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

# DBTITLE 1,Build Config + Run Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
# RUN PIPELINE PER CONDITION (fixes the cross-condition deduplication bug)
#
# When condition=ALL, bronze.py merges all files and deduplicates segments
# ACROSS conditions (keeping best-SNR only). This is wrong: each current
# setpoint (45A, 60A, 150A, 450A) is a SEPARATE EIS experiment.
#
# Fix: iterate conditions individually, producing separate results per condition.
# ═══════════════════════════════════════════════════════════════════════════════
import os, shutil, tempfile


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
    return spectra_csv(out_dir)


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
print(f"  Conditions: {_conditions_to_run}")
print(f"  Band:       {F_MIN} – {F_MAX} Hz")
print(f"  Stop after: {STOP_AFTER}\n")

# ─── Run pipeline once per condition ───
PIPELINE_RESULTS = {}  # {condition_str: manifest_dict}

for cond in _conditions_to_run:
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
        plate=PLATE,
        source_format=SOURCE_FORMAT,
        csv_path=Path(CSV_PATH) if CSV_PATH else None,
        csv_dialect=CSV_DIALECT,
        csv_tones=CSV_TONES,
        gain_file=Path(GAIN_FILE) if GAIN_FILE else None,
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

# DBTITLE 1,Nyquist Comparison: Gamry (whole-cell) vs Local Pipeline Segments
# ═══════════════════════════════════════════════════════════════════════════════
# NYQUIST COMPARISON: Gamry whole-cell vs Local Pipeline segments
#
# Overlay plot showing:
#   - Individual segment spectra (thin coloured lines) from pipeline silver
#   - Pipeline cell aggregate (thick red) = area-weighted parallel combination
#   - Gamry Reference 3000 (thick black) = independent whole-cell measurement
#
# This is the key validation: does the 66-segment parallel sum reproduce
# the single-pair-of-leads Gamry measurement?
# ═══════════════════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ─── Ensure variables are available after kernel restart ───
try:
    LEEPA
except NameError:
    LEEPA = dbutils.widgets.get('leepa_id')

try:
    COND_FILTER = dbutils.widgets.get('condition')
except Exception:
    COND_FILTER = '60A'

A_CELL_CM2 = 304.92

_META_TBL = 'ps_xplatform_dev.rvadvtec_ops.datago_advtec_metadata'
_GP_TBL = 'ps_xplatform_dev.rvadvtec_ops.datago_advtec_generalproperties'
_VAL_TBL = 'ps_xplatform_dev.rvadvtec_ops.datago_advtec_values_delta'

# ─── 1. Load Gamry spectrum from datago ───
def load_gamry_from_datago(leepa, cond_filter=None):
    """Load Gamry GALVEIS spectrum(s) from datago for this Leepa."""
    df_meta = spark.sql(f"""
        SELECT DISTINCT gp.file_id, gp.measurementBegin
        FROM {_META_TBL} m
        JOIN {_GP_TBL} gp ON m.measurement_id = gp.measurement_id
        WHERE m.orderId = 'RO{leepa}'
          AND gp.measurement_type = 'GALVEIS'
        ORDER BY gp.measurementBegin
    """).toPandas()
    
    if df_meta.empty:
        return {}
    
    # Get unique file_ids (each = one Gamry spectrum)
    file_ids = df_meta['file_id'].unique().tolist()
    
    spectra = {}
    for idx, fid in enumerate(file_ids):
        df = spark.sql(f"""
            SELECT channel, CAST(time_value AS DOUBLE) AS pt,
                   CAST(value AS DOUBLE) AS value
            FROM {_VAL_TBL}
            WHERE file_id = '{fid}'
              AND channel IN ('freq', 'zreal', 'zimag', 'zmod', 'zphz', 'idc', 'vdc')
        """).toPandas()
        
        if df.empty:
            continue
        
        pivot = df.pivot_table(index='pt', columns='channel', values='value', aggfunc='first')
        pivot = pivot.sort_values('freq').reset_index(drop=True)
        
        k = A_CELL_CM2 * 1000.0  # Ω → mΩ·cm²
        spec = pd.DataFrame({
            'freq_hz': pivot['freq'].values,
            'z_re': pivot['zreal'].values * k,
            'z_im': pivot['zimag'].values * k,
            'i_dc': pivot.get('idc', pd.Series([np.nan])).values,
        })
        spec = spec.dropna(subset=['freq_hz', 'z_re', 'z_im']).sort_values('freq_hz').reset_index(drop=True)
        
        # Label by DC current
        idc_mean = np.nanmean(spec['i_dc']) if 'i_dc' in spec else np.nan
        label = f'Gamry-{idx+1}'
        if np.isfinite(idc_mean) and abs(idc_mean) > 1:
            for c in ['45A', '60A', '150A', '450A']:
                if abs(idc_mean - float(c.replace('A', ''))) < float(c.replace('A', '')) * 0.3:
                    label = c
                    break
        spectra[label] = spec
    
    return spectra


# ─── 2. Load pipeline silver spectra + cell aggregate ───
def load_pipeline_spectra(leepa, cond):
    """Load per-segment spectra and cell aggregate from pipeline output."""
    # Try workspace path first (persistent), then /tmp (ephemeral)
    out_dir = Path(f'/Workspace/Users/uum5fe@bosch.com/eis_results/{leepa}/{cond}')
    if not out_dir.exists():
        out_dir = Path(f'/tmp/eis_results/{leepa}/{cond}')
    
    # Per-segment spectra
    seg_path = spectra_csv(out_dir)
    seg_df = pd.read_csv(seg_path) if seg_path.exists() else None
    
    # Cell aggregate
    agg_path = out_dir / 'silver' / 'cell_aggregate.csv'
    agg_df = None
    if agg_path.exists():
        agg = pd.read_csv(agg_path)
        agg_df = pd.DataFrame({
            'freq_hz': agg['freq_hz'],
            'z_re': agg['z_re_mohm_cm2'],
            'z_im': agg['z_im_mohm_cm2'],
        })
        # Normalise sign convention (Gamry: -Im for capacitive)
        if np.nanmedian(agg_df['z_im']) > 0:
            agg_df['z_im'] = -agg_df['z_im']
        agg_df = agg_df.sort_values('freq_hz').reset_index(drop=True)
    
    return seg_df, agg_df


print(f"  Loading Gamry data for Leepa {LEEPA} from datago...")
GAMRY_SPECTRA = load_gamry_from_datago(LEEPA)
print(f"  Found {len(GAMRY_SPECTRA)} Gamry spectrum(s): {list(GAMRY_SPECTRA.keys())}")

# Determine conditions to plot
_conds = [COND_FILTER] if COND_FILTER != 'ALL' else ['60A', '150A', '45A', '450A']

for cond in _conds:
    seg_df, agg_df = load_pipeline_spectra(LEEPA, cond)
    
    if seg_df is None:
        print(f"  {cond}: no pipeline spectra found — run pipeline first")
        continue
    
    # Find matching Gamry
    gamry = GAMRY_SPECTRA.get(cond)
    if gamry is None:
        # Try first available
        gamry_keys = list(GAMRY_SPECTRA.keys())
        gamry = GAMRY_SPECTRA[gamry_keys[0]] if gamry_keys else None
        gamry_label = gamry_keys[0] if gamry_keys else None
    else:
        gamry_label = cond
    
    # ─── Build 3-panel Plotly figure ───
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['<b>Nyquist</b>', '<b>|Z|(f) Bode</b>', '<b>Phase(f)</b>'],
        horizontal_spacing=0.06)
    
    segments = sorted(seg_df['segment'].unique())
    n_seg = len(segments)
    colors = [f'hsl({int(i*360/n_seg)}, 60%, 55%)' for i in range(n_seg)]
    
    # Plot per-segment spectra (thin, semi-transparent)
    for i, seg in enumerate(segments):
        sd = seg_df[seg_df['segment'] == seg].sort_values('freq_hz')
        f = sd['freq_hz'].values
        zr = sd['z_re_mohm_cm2'].values
        zi = sd['z_im_mohm_cm2'].values
        
        # Physical cleaning
        keep = np.isfinite(zr) & np.isfinite(zi) & (zr > 0) & (zr <= 500)
        f, zr, zi = f[keep], zr[keep], zi[keep]
        if len(f) < 3:
            continue
        
        clr = colors[i]
        seg_name = f'Seg {seg}'
        show = (i == 0)  # only first in legend to avoid clutter
        
        fig.add_trace(go.Scatter(
            x=zr, y=-zi, mode='lines', line=dict(width=0.8, color=clr),
            opacity=0.4, name='Pipeline segments' if i == 0 else seg_name,
            legendgroup='segments', showlegend=show,
            hovertemplate=f'Seg {seg}<br>f=%{{customdata:.1f}} Hz<br>'
                          f"Z'=%{{x:.1f}}<br>-Z''=%{{y:.1f}}<extra></extra>",
            customdata=f,
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=f, y=np.abs(zr + 1j * zi), mode='lines',
            line=dict(width=0.8, color=clr), opacity=0.4,
            legendgroup='segments', showlegend=False,
        ), row=1, col=2)
        
        fig.add_trace(go.Scatter(
            x=f, y=np.degrees(np.angle(zr + 1j * zi)), mode='lines',
            line=dict(width=0.8, color=clr), opacity=0.4,
            legendgroup='segments', showlegend=False,
        ), row=1, col=3)
    
    # Plot cell aggregate (thick red)
    if agg_df is not None and len(agg_df) > 3:
        fig.add_trace(go.Scatter(
            x=agg_df['z_re'], y=-agg_df['z_im'],
            mode='lines+markers', line=dict(width=3, color='#c0392b'),
            marker=dict(size=5, color='#c0392b'),
            name='Pipeline aggregate (66 seg)', legendgroup='agg',
            hovertemplate="Aggregate<br>f=%{customdata:.1f} Hz<br>"
                          "Z'=%{x:.1f}<br>-Z''=%{y:.1f} mΩ·cm²<extra></extra>",
            customdata=agg_df['freq_hz'].values,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=agg_df['freq_hz'], y=np.abs(agg_df['z_re'] + 1j * agg_df['z_im']),
            mode='lines+markers', line=dict(width=3, color='#c0392b'),
            marker=dict(size=4, color='#c0392b'),
            legendgroup='agg', showlegend=False,
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=agg_df['freq_hz'],
            y=np.degrees(np.angle(agg_df['z_re'] + 1j * agg_df['z_im'])),
            mode='lines+markers', line=dict(width=3, color='#c0392b'),
            marker=dict(size=4, color='#c0392b'),
            legendgroup='agg', showlegend=False,
        ), row=1, col=3)
    
    # Plot Gamry (thick black, prominent)
    if gamry is not None and len(gamry) > 3:
        fig.add_trace(go.Scatter(
            x=gamry['z_re'], y=-gamry['z_im'],
            mode='lines+markers', line=dict(width=3.5, color='#1a1a1a'),
            marker=dict(size=6, color='#1a1a1a', symbol='diamond'),
            name=f'Gamry Reference 3000 ({gamry_label})', legendgroup='gamry',
            hovertemplate="Gamry<br>f=%{customdata:.1f} Hz<br>"
                          "Z'=%{x:.1f}<br>-Z''=%{y:.1f} mΩ·cm²<extra></extra>",
            customdata=gamry['freq_hz'].values,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=gamry['freq_hz'], y=np.abs(gamry['z_re'] + 1j * gamry['z_im']),
            mode='lines+markers', line=dict(width=3.5, color='#1a1a1a'),
            marker=dict(size=5, color='#1a1a1a', symbol='diamond'),
            legendgroup='gamry', showlegend=False,
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=gamry['freq_hz'],
            y=np.degrees(np.angle(gamry['z_re'] + 1j * gamry['z_im'])),
            mode='lines+markers', line=dict(width=3.5, color='#1a1a1a'),
            marker=dict(size=5, color='#1a1a1a', symbol='diamond'),
            legendgroup='gamry', showlegend=False,
        ), row=1, col=3)
    
    # ─── Layout ───
    fig.update_xaxes(title_text="Z' [mΩ·cm²]", row=1, col=1)
    fig.update_yaxes(title_text="-Z'' [mΩ·cm²]", row=1, col=1)
    fig.update_xaxes(title_text="f [Hz]", type="log", row=1, col=2)
    fig.update_yaxes(title_text="|Z| [mΩ·cm²]", type="log", row=1, col=2)
    fig.update_xaxes(title_text="f [Hz]", type="log", row=1, col=3)
    fig.update_yaxes(title_text="Phase [°]", row=1, col=3)
    
    _title = (f'<b>Gamry vs Local Pipeline — Leepa {LEEPA}, {cond}</b><br>'
              f'<sup>{n_seg} segments (thin) | Pipeline aggregate (red) | '
              f'Gamry Reference 3000 (black diamonds)</sup>')
    
    fig.update_layout(
        title=dict(text=_title, font=dict(size=13)),
        height=580, width=1550,
        paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(t=100, b=55, r=280),
        hovermode='closest',
        legend=dict(
            font=dict(size=10), y=0.5, yanchor='middle',
            itemclick='toggle', itemdoubleclick='toggleothers'),
    )
    for r in range(1, 4):
        fig.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.07)', row=1, col=r)
        fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.07)', row=1, col=r)
    
    displayHTML(fig.to_html(full_html=False, include_plotlyjs='cdn'))
    
    # ─── Print summary ───
    print(f"\n  {cond}: {n_seg} segments plotted")
    if gamry is not None:
        # R_ohmic intercept (Gamry)
        d = gamry.sort_values('freq_hz', ascending=False).reset_index(drop=True)
        zi_g, zr_g = d['z_im'].values, d['z_re'].values
        k = np.where(np.diff(np.sign(zi_g)))[0]
        if len(k):
            j = k[0]
            t = -zi_g[j] / (zi_g[j+1] - zi_g[j])
            rs_g = zr_g[j] + t * (zr_g[j+1] - zr_g[j])
            print(f"  Gamry R_Ω = {rs_g:.1f} mΩ·cm² (HF intercept)")
    if agg_df is not None:
        d2 = agg_df.sort_values('freq_hz', ascending=False).reset_index(drop=True)
        zi_p, zr_p = d2['z_im'].values, d2['z_re'].values
        k2 = np.where(np.diff(np.sign(zi_p)))[0]
        if len(k2):
            j2 = k2[0]
            t2 = -zi_p[j2] / (zi_p[j2+1] - zi_p[j2])
            rs_p = zr_p[j2] + t2 * (zr_p[j2+1] - zr_p[j2])
            print(f"  Pipeline R_Ω = {rs_p:.1f} mΩ·cm² (cell aggregate HF intercept)")
            if 'rs_g' in dir():
                print(f"  ΔR_Ω = {rs_p - rs_g:+.1f} mΩ·cm² ({100*(rs_p-rs_g)/rs_g:+.1f}%)")

# COMMAND ----------

# DBTITLE 1,Display Plate Heatmaps (R_ohmic, R_ct, R_mt)
# ═══════════════════════════════════════════════════════════════════════════════
# PLATE HEATMAPS PER CONDITION (Gold-plate style, matching old pipeline)
# Shows: R_ohmic PNG + interactive HTML for each condition
# ═══════════════════════════════════════════════════════════════════════════════
_heatmap_params = ['R_ohmic', 'R_ct', 'R_mt']

for cond, pr in PIPELINE_RESULTS.items():
    if pr is None:
        continue
    _gold_dir = maps_dir(pr['out_dir'])
    
    print(f"\n{'═'*75}")
    print(f"  HEATMAPS — {cond}")
    print(f"{'═'*75}")
    
    for param in _heatmap_params:
        png_path = _gold_dir / f'map_{param}.png'
        if png_path.exists():
            print(f"\n  {param}:")
            display(IPImage(filename=str(png_path)))
        else:
            print(f"  {param}: not generated")
    
    # Interactive HTML (Plotly with per-segment hover)
    for param in _heatmap_params:
        html_path = _gold_dir / f'map_{param}.html'
        if html_path.exists():
            with open(html_path) as f:
                displayHTML(f.read())

# COMMAND ----------

# DBTITLE 1,ECM Fitting: Rs + L + (Rct1 || CPE1) + (Rct2 || CPE2)
# ═══════════════════════════════════════════════════════════════════════════════
# ECM FIT: Two-Arc + Inductance Model
# Model: Z(f) = Rs + jωL + Rct1/(1+Rct1·Y01·(jω)^n1) + Rct2/(1+Rct2·Y02·(jω)^n2)
#
# Same approach as EIS Analysis Gold Table (eis_lockin.py fit_ecm).
# Input:  silver spectra (mΩ·cm²)
# Output: ECM parameters per segment per condition
# ═══════════════════════════════════════════════════════════════════════════════
from scipy.optimize import least_squares

# ─── ECM model (inlined from eis_lockin.py) ───
def Z_ecm(f, Rs, L, Rct1, Y01, n1, Rct2, Y02, n2):
    """Two-arc + inductance ECM: Rs + jωL + Z_arc1 + Z_arc2."""
    w = 2 * np.pi * f
    Zl = 1j * w * L
    Z1 = Rct1 / (1.0 + Rct1 * Y01 * (1j * w) ** n1)
    Z2 = Rct2 / (1.0 + Rct2 * Y02 * (1j * w) ** n2)
    return Rs + Zl + Z1 + Z2


def fit_ecm(f, Z, p0=None):
    """Fit two-arc + L ECM model. Z in Ω·cm² (ASR)."""
    if p0 is None:
        Rs0 = float(np.real(Z[np.argmax(f)]))
        Rct0 = max(float(np.real(Z[np.argmin(f)]) - Rs0) / 2, 0.001)
        p0 = [Rs0, 1e-7, Rct0, 0.01, 0.8, Rct0, 0.1, 0.8]

    def residual(p):
        Zm = Z_ecm(f, *p)
        Z_abs = np.abs(Z)
        Z_abs[Z_abs < 1e-12] = 1e-12  # avoid division by zero
        dr = (np.real(Z) - np.real(Zm)) / Z_abs
        di = (np.imag(Z) - np.imag(Zm)) / Z_abs
        return np.concatenate([dr, di])

    lb = [0, 0, 0, 1e-6, 0.3, 0, 1e-6, 0.3]
    ub = [np.inf, 1e-4, np.inf, 100, 1.0, np.inf, 100, 1.0]

    try:
        res = least_squares(residual, p0, bounds=(lb, ub), max_nfev=5000)
        return res.x, res.cost
    except Exception:
        return p0, np.inf


# ─── Run ECM fit for each condition ───
ECM_RESULTS = {}  # {cond: {seg: {params...}}}

for cond, pr in PIPELINE_RESULTS.items():
    if pr is None:
        continue
    out_dir = pr['out_dir']
    
    _spectra_path = spectra_csv(out_dir)
    if not _spectra_path.exists():
        print(f"  {cond}: no spectra → skip ECM")
        continue

    # The CSV pipeline already fitted the ECM, and fitted it better: the
    # number of arcs is chosen by AICc instead of fixed at two, the residuals
    # are weighted by the propagated per-point sigma instead of by |Z|, and
    # every parameter comes back with a standard error and a chi2_nu that
    # says whether the circuit describes the data.  Show that table when it
    # exists rather than re-fitting worse.
    _pipeline_ecm = out_dir / 'csv' / 'ecm_parameters.csv'
    if _pipeline_ecm.exists():
        _df_e = pd.read_csv(_pipeline_ecm)
        print(f"\n{'═'*75}")
        print(f"  ECM FIT (from csv_pipeline) — {cond}")
        print(f"  Model: Rs + jωL + ZARC(s), arc count chosen by AICc, "
              f"σ-weighted")
        print(f"{'═'*75}")
        display(_df_e.round(5))
        ECM_RESULTS[cond] = {}
        continue

    df = pd.read_csv(_spectra_path)
    segments = sorted(df['segment'].unique())

    ecm_cond = {}
    failed = []
    
    for seg in segments:
        sd = df[df['segment'] == seg].sort_values('freq_hz')
        f = sd['freq_hz'].values
        zr = sd['z_re_mohm_cm2'].values
        zi = sd['z_im_mohm_cm2'].values
        
        # Physical cleaning (same as Nyquist cell)
        keep = np.isfinite(zr) & np.isfinite(zi) & (zr > 0)
        keep &= (zr >= 1.0) & (zr <= 500.0)
        f, zr, zi = f[keep], zr[keep], zi[keep]
        
        if len(f) < 5:
            failed.append(seg)
            continue
        
        # Convert mΩ·cm² → Ω·cm² for fitter (divide by 1000)
        Z_complex = (zr + 1j * zi) / 1000.0
        
        pars, cost = fit_ecm(f, Z_complex)
        
        if cost == np.inf or not np.isfinite(pars[0]):
            failed.append(seg)
        else:
            ecm_cond[seg] = {
                'Rs': pars[0], 'L': pars[1],
                'Rct1': pars[2], 'Y01': pars[3], 'n1': pars[4],
                'Rct2': pars[5], 'Y02': pars[6], 'n2': pars[7],
                'chi2': cost,
                'freq': f,
                'Z_meas': (zr + 1j * zi),  # keep in mΩ·cm² for plotting
            }
    
    ECM_RESULTS[cond] = ecm_cond
    
    # ─── Summary ───
    if ecm_cond:
        rows = []
        for seg, p in sorted(ecm_cond.items(), key=lambda x: int(x[0])):
            rows.append({
                'segment': int(seg),
                'Rs_mOhm_cm2': p['Rs'] * 1000,      # Ω·cm² → mΩ·cm²
                'Rct1_mOhm_cm2': p['Rct1'] * 1000,
                'Rct2_mOhm_cm2': p['Rct2'] * 1000,
                'CPE1_Y0': p['Y01'],
                'CPE1_n': p['n1'],
                'CPE2_Y0': p['Y02'],
                'CPE2_n': p['n2'],
                'chi2': p['chi2'],
            })
        
        ecm_df = pd.DataFrame(rows)
        
        print(f"\n{'═'*75}")
        print(f"  ECM FIT — Leepa {LEEPA}, {cond}")
        print(f"  Model: Rs + jωL + (Rct1||CPE1) + (Rct2||CPE2)")
        print(f"  Fitted: {len(ecm_cond)} segments | Failed: {len(failed)}")
        print(f"{'═'*75}")
        print(f"  Rs:   {ecm_df['Rs_mOhm_cm2'].median():.2f} ± {ecm_df['Rs_mOhm_cm2'].std():.2f} mΩ·cm² (median ± std)")
        print(f"  Rct1: {ecm_df['Rct1_mOhm_cm2'].median():.2f} ± {ecm_df['Rct1_mOhm_cm2'].std():.2f} mΩ·cm²")
        print(f"  Rct2: {ecm_df['Rct2_mOhm_cm2'].median():.2f} ± {ecm_df['Rct2_mOhm_cm2'].std():.2f} mΩ·cm²")
        print(f"  CPE1 n: {ecm_df['CPE1_n'].median():.3f} | CPE2 n: {ecm_df['CPE2_n'].median():.3f}")
        print(f"  χ² (median): {ecm_df['chi2'].median():.4f}")
        if failed:
            print(f"  Failed segments: {failed}")
        print()
        display(ecm_df.round(4))
    else:
        print(f"  {cond}: all segments failed ECM fitting.")

# COMMAND ----------

# DBTITLE 1,ECM Visualization: Nyquist Overlay + Parameter Heatmaps
# ═══════════════════════════════════════════════════════════════════════════════
# ECM VISUALIZATION: Nyquist overlay (measured vs fitted) per condition
# Shows 6 representative segments + residuals
# ═══════════════════════════════════════════════════════════════════════════════
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_ECM_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for cond, ecm_cond in ECM_RESULTS.items():
    if not ecm_cond:
        continue
    
    # Select up to 6 representative segments (evenly spaced)
    all_segs = sorted(ecm_cond.keys(), key=lambda x: int(x))
    step = max(1, len(all_segs) // 6)
    show_segs = all_segs[::step][:6]
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[f'Seg {s}' for s in show_segs],
        horizontal_spacing=0.08, vertical_spacing=0.15)
    
    for idx, seg in enumerate(show_segs):
        row = idx // 3 + 1
        col = idx % 3 + 1
        p = ecm_cond[seg]
        
        f_hz = p['freq']
        Z_meas = p['Z_meas']  # in mΩ·cm²
        Z_fit = Z_ecm(f_hz, p['Rs'], p['L'], p['Rct1'], p['Y01'], p['n1'],
                       p['Rct2'], p['Y02'], p['n2']) * 1000  # Ω·cm² → mΩ·cm²
        
        clr = _ECM_COLORS[idx % len(_ECM_COLORS)]
        
        # Measured points
        fig.add_trace(go.Scatter(
            x=Z_meas.real, y=-Z_meas.imag,
            mode='markers', marker=dict(size=5, color=clr),
            name='Measured', showlegend=(idx == 0),
            legendgroup='meas',
        ), row=row, col=col)
        
        # ECM fit line
        fig.add_trace(go.Scatter(
            x=Z_fit.real, y=-Z_fit.imag,
            mode='lines', line=dict(color='black', width=2, dash='dash'),
            name='ECM Fit', showlegend=(idx == 0),
            legendgroup='fit',
        ), row=row, col=col)
        
        fig.update_xaxes(title_text="Z' [mΩ·cm²]" if row == 2 else "",
                         showgrid=True, row=row, col=col)
        fig.update_yaxes(title_text="-Z'' [mΩ·cm²]" if col == 1 else "",
                         showgrid=True, row=row, col=col)
    
    fig.update_layout(
        title=f'<b>ECM Fit Quality — Leepa {LEEPA}, {cond}</b>',
        height=600, width=1200,
        paper_bgcolor='white', plot_bgcolor='white',
        margin=dict(t=80))
    fig.show()
    
    # ─── Parameter heatmaps (R_ohmic and Rct from ECM) ───
    # Build segment → value maps for plate visualization
    rs_map = {int(seg): p['Rs'] * 1000 for seg, p in ecm_cond.items()}
    rct_map = {int(seg): p['Rct1'] * 1000 for seg, p in ecm_cond.items()}
    
    # Filter to segments with known coordinates
    seg_coords = {int(s): (c.cx_mm, c.cy_mm) for s, c in geom.SEGMENTS.items()}
    rs_map = {k: v for k, v in rs_map.items() if k in seg_coords}
    rct_map = {k: v for k, v in rct_map.items() if k in seg_coords}
    
    if not rs_map:
        print(f"  {cond}: no coordinate data for heatmaps")
        continue
    
    # Scatter-style plate heatmap using matplotlib
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm
    
    for param_name, param_map, unit in [
        ('Rs (ECM)', rs_map, 'mΩ·cm²'),
        ('Rct1 (ECM)', rct_map, 'mΩ·cm²'),
    ]:
        fig_h, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.set_facecolor('#f5f5f5')
        
        segs = sorted(param_map.keys())
        xs = [seg_coords[s][0] for s in segs]
        ys = [seg_coords[s][1] for s in segs]
        vals = [param_map[s] for s in segs]
        
        vmin, vmax = np.percentile(vals, 5), np.percentile(vals, 95)
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        sc = ax.scatter(xs, ys, c=vals, cmap='RdYlGn_r', norm=norm,
                        s=180, edgecolors='k', linewidths=0.5, zorder=3)
        
        for s, x, y, v in zip(segs, xs, ys, vals):
            ax.annotate(f'{s}', (x, y), ha='center', va='center',
                        fontsize=6, fontweight='bold', zorder=4)
        
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label(f'{param_name} [{unit}]')
        
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('y [mm]')
        ax.set_title(f'{param_name} — Leepa {LEEPA}, {cond} ({len(segs)} segments)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        plt.close(fig_h)
    
    print(f"  {cond}: ECM heatmaps rendered")

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
    p = maps_dir(out_dir) / 'plate_summary.csv'
    if not p.exists():
        return (float('nan'), float('nan'))
    d = pd.read_csv(p)
    m = d[d['class'] == 'measured']
    return float(m.R_ohmic.mean()), float(m.R_ohmic.std())


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
def z_cpe(w, Y0, n):
    """CPE impedance: Z_CPE = 1 / (Y0 · (jω)^n)."""
    return 1.0 / (Y0 * (1j * w) ** n)


def z_model(p, freq):
    """L + Rs + (Rct1 || CPE1) + (Rct2 || CPE2)."""
    L, Rs, R1, Y1, n1, R2, Y2, n2 = p
    w = 2 * np.pi * freq
    Z_par = lambda R, Y, n: 1.0 / (1.0 / R + 1.0 / z_cpe(w, Y, n))
    return 1j * w * L + Rs + Z_par(R1, Y1, n1) + Z_par(R2, Y2, n2)


def ecm_starting_values(freq, Z, drt_res=None):
    """DRT-informed starting values for ECM fit."""
    freq = np.asarray(freq, float)
    Z = np.asarray(Z, complex)
    Rs = float(np.min(Z.real))
    R_tot = float(np.max(Z.real) - Rs)

    if drt_res is not None:
        pk = drt_peaks(drt_res)[:2]
        if len(pk) == 2:
            h, l = pk[0], pk[1]
            R1, R2 = h["R"], l["R"]
            Y1 = 1 / (R1 * h["tau"]) if R1 * h["tau"] > 0 else 0.01
            Y2 = 1 / (R2 * l["tau"]) if R2 * l["tau"] > 0 else 0.1
            return np.array([1e-9, Rs, max(R1, 1e-3), Y1, 0.9,
                             max(R2, 1e-3), Y2, 0.9])

    # Fallback: no DRT, split arc in two
    k = int(np.argmax(-Z.imag))
    f_top = freq[k]
    R1 = R2 = R_tot / 2
    C = 1.0 / (2 * np.pi * f_top * max(R_tot, 1e-6))
    return np.array([1e-9, Rs, max(R1, 1e-3), C, 0.9,
                     max(R2, 1e-3), C * 10, 0.9])


def ecm_fit(freq, Z, p0=None, drt_res=None, weight="modulus"):
    """Complex NLLS fit. Returns params, Z_fit, chi2, residuals."""
    freq = np.asarray(freq, float)
    Z = np.asarray(Z, complex)
    if p0 is None:
        p0 = ecm_starting_values(freq, Z, drt_res)
    wgt = 1.0 / np.abs(Z) if weight == "modulus" else np.ones(len(Z))

    def resid(p):
        Zm = z_model(p, freq)
        return np.concatenate([(Z.real - Zm.real) * wgt,
                               (Z.imag - Zm.imag) * wgt])

    lo = np.array([0, 0, 1e-6, 1e-12, 0.3, 1e-6, 1e-12, 0.3])
    hi = np.array([1e3, 1e6, 1e6, 1e6, 1.0, 1e6, 1e6, 1.0])
    r = least_squares(resid, np.clip(p0, lo * 1.001, hi * 0.999),
                      bounds=(lo, hi), method="trf", max_nfev=20000)
    Zf = z_model(r.x, freq)
    names = ["L", "Rs", "R1", "Y1", "n1", "R2", "Y2", "n2"]
    return {"params": dict(zip(names, r.x)), "x": r.x, "Z_fit": Zf,
            "chi2": float(np.sum(r.fun ** 2)),
            "res_pct": 100 * np.abs(Z - Zf) / np.abs(Z),
            "success": bool(r.success)}


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

        # Get DRT for this segment (if available)
        drt_res = DRT_RESULTS.get(cond, {}).get(seg, None)

        try:
            ecm = ecm_fit(f, Z, drt_res=drt_res)
            ecm['freq'] = f
            ecm['Z_meas'] = Z
            ECM_FIT_RESULTS[cond][seg] = ecm
        except Exception as e:
            pass  # skip segments that fail to converge

    n_ok = sum(1 for v in ECM_FIT_RESULTS[cond].values() if v['success'])
    n_tot = len(ECM_FIT_RESULTS[cond])
    # Extract Rs statistics
    rs_vals = [v['params']['Rs'] for v in ECM_FIT_RESULTS[cond].values() if v['success']]
    if rs_vals:
        print(f"  {cond}: {n_ok}/{n_tot} converged | "
              f"Rs = {np.mean(rs_vals):.1f} ± {np.std(rs_vals):.1f} mΩ·cm²")
    else:
        print(f"  {cond}: {n_ok}/{n_tot} converged")

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
        s = '✓' if ecm['success'] else '✗'
        titles_nyq.append(f'Seg {seg} {s} (χ²={ecm["chi2"]:.3f})')

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
            rows.append({'segment': int(seg), 'Rs': p['Rs'], 'L_nH': p['L']*1e9,
                         'Rct1': p['R1'], 'Y01': p['Y1'], 'n1': p['n1'],
                         'Rct2': p['R2'], 'Y02': p['Y2'], 'n2': p['n2'],
                         'chi2': ecm['chi2']})
    df_params = pd.DataFrame(rows)
    print(f"\n  ECM Parameters — {cond}:")
    print(f"  {'─'*70}")
    print(f"  Rs:   {df_params['Rs'].mean():.2f} ± {df_params['Rs'].std():.2f} mΩ·cm²")
    print(f"  Rct1: {df_params['Rct1'].mean():.2f} ± {df_params['Rct1'].std():.2f} mΩ·cm²")
    print(f"  Rct2: {df_params['Rct2'].mean():.2f} ± {df_params['Rct2'].std():.2f} mΩ·cm²")
    print(f"  n1:   {df_params['n1'].mean():.3f} ± {df_params['n1'].std():.3f}")
    print(f"  n2:   {df_params['n2'].mean():.3f} ± {df_params['n2'].std():.3f}")
    print(f"  χ²:   {df_params['chi2'].median():.4f} (median)")
    display(df_params.describe().round(4))