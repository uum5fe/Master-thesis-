# Databricks notebook source
# fresh Python for a clean module reload
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Local EIS — STANDARD evaluation from the raw signals (separate pipeline)
# ═══════════════════════════════════════════════════════════════════════════════
# A SEPARATE pipeline from "Local EIS Pipeline Runner". It evaluates the raw
# FAMOS waveforms the textbook way, end to end (pipeline/std_pipeline.py):
#
#   1. read the cards (datago first, the Volumes .DAT files otherwise) and
#      the Gamry .DTA of the condition = the list of excitation frequencies
#   2. locate every frequency step on each card with a lock-in, edges where
#      the signal stops following that frequency's sine
#   3. sine fit at the known frequency over whole periods (1st period dropped)
#   4. Z = K(T) * U / u_seg, then only KNOWN corrections: the Abgleich chain
#      calibration and the ADC slot sampling delay -- nothing fitted to data
#   5. keep points with >= 3 periods and SNR >= 3 dB
#   6. R_ohmic (high-frequency intercept), R_ct, R_mt, R_pol from the
#      standard circuit R_ohmic + L + (R_ct||CPE) + (R_mt||CPE), plus j, T
#   7. the MEASURED segments in parallel vs the Gamry whole cell
#   8. tables, Nyquist / Bode / phase, one 2D interpolated map per parameter
#
# Results: /Volumes/.../EIS_Results_standard/<order>/<condition>/
# ═══════════════════════════════════════════════════════════════════════════════
import sys
import importlib
from pathlib import Path

_PIPELINE_DIR_OVERRIDE = ''      # e.g. '/Workspace/Users/you@bosch.com/local_eis'


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
    if (Path.cwd() / 'eis_paths.py').exists():
        return str(Path.cwd())
    raise FileNotFoundError('eis_paths.py must sit beside this notebook '
                            '(or set _PIPELINE_DIR_OVERRIDE)')


_PIPELINE_DIR = _find_pipeline_dir()
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)
import eis_paths  # noqa: F401
import numpy as np
import pandas as pd
import r2d2_geometry as geom
import standard_eis
import std_pipeline
import datago_source
import nyquist
import viewers
import plate_figure
for _m in (geom, standard_eis, std_pipeline, datago_source, nyquist, viewers, plate_figure):
    importlib.reload(_m)
print(f"  modules from {_PIPELINE_DIR}")

# ─── where things are (same locations as the main runner) ───
_EV_ROOT = Path('/Volumes/ps_xplatform_dev/rvadvtec_dev/ev_rvadvtec_dev')
FAMOS_ROOT = _EV_ROOT / 'Famos'
RESULTS_ROOT = _EV_ROOT / 'EIS_Results_standard'
GAMRY_ROOTS = ['{ev}/RO{leepa}_Gamry', '{ev}/{leepa}_Gamry', '{ev}/Gamry']
GAMRY_VERSION = ''               # e.g. 'V26_092' when a folder holds several builds
CURR_CAL = Path('/Workspace/Users/uum5fe@bosch.com/curr.csv')
TEMP_CAL = Path('/Workspace/Users/uum5fe@bosch.com/temp.csv')
GAIN_FILE = Path('/Workspace/Users/uum5fe@bosch.com/chain_gain_gen1.csv')
F_MIN_HZ, F_MAX_HZ = 0.15, 2400.0

# COMMAND ----------

# DBTITLE 1,Widgets: order, conditions, write results
import re as _re
_orders = set()
for _p in FAMOS_ROOT.glob('*_Current_*_Karte_*.DAT'):
    _m = _re.search(r'(?:Leepa_|RO)(\d{7})', _p.name)
    if _m:
        _orders.add(_m.group(1))
try:
    _df = spark.sql(f"""SELECT DISTINCT m.orderId FROM {datago_source.META_TBL} m
                       JOIN {datago_source.GP_TBL} gp ON m.measurement_id = gp.measurement_id
                       WHERE gp.measurement_type = 'GALVEIS'""").toPandas()
    _orders |= {str(o).replace('RO', '') for o in _df.orderId}
except Exception as _e:
    print(f"  datago order list not available ({type(_e).__name__}) -- Volumes only")
ORDERS = sorted(_orders) or ['2612030']
dbutils.widgets.dropdown('order', '2612030' if '2612030' in ORDERS else ORDERS[-1],
                         ORDERS, 'Order ID (Leepa)')
LEEPA = dbutils.widgets.get('order')
_conds = sorted({_re.search(r'_Current_([^_]+)_', p.name).group(1)
                 for p in FAMOS_ROOT.glob(f'*{LEEPA}*_Current_*_Karte_1.DAT')}
                or {'45A', '60A', '150A', '450A'},
                key=lambda c: float(_re.sub(r'[^\d.]', '', c) or 0))
dbutils.widgets.multiselect('conditions', 'ALL', ['ALL'] + _conds, 'Conditions')
dbutils.widgets.dropdown('write_results', 'yes', ['yes', 'no'], 'Write results to the Volume')
_sel = dbutils.widgets.get('conditions').split(',')
CONDITIONS = _conds if (not _sel or 'ALL' in _sel) else [c for c in _conds if c in _sel]
WRITE = dbutils.widgets.get('write_results') == 'yes'
print(f"  order {LEEPA} · conditions {', '.join(CONDITIONS)} · band {F_MIN_HZ} Hz – "
      f"{F_MAX_HZ/1000:g} kHz · results {'Volume' if WRITE else '/tmp only'}")

# COMMAND ----------

# DBTITLE 1,Data source: datago first, Volumes .DAT otherwise
try:
    DATAGO_MAP = datago_source.discover_famos_file_ids(spark, LEEPA, FAMOS_ROOT)
except Exception as _e:
    print(f"  datago lookup failed ({type(_e).__name__}: {_e}) -> Volumes")
    DATAGO_MAP = {}


def source_for(cond):
    """(files, reader, label) for one condition."""
    if cond in DATAGO_MAP:
        ids = DATAGO_MAP[cond]
        files = [f'Karte_{i+1}' for i in range(len(ids))]
        lookup = dict(zip(files, ids))
        return files, (lambda lab: datago_source.DatagoFamosFile(spark, lookup[str(lab)], str(lab))), 'datago'
    files = std_pipeline.discover_files(FAMOS_ROOT, LEEPA, cond)
    return files, None, 'Volumes'


for _c in CONDITIONS:
    _f, _r, _src = source_for(_c)
    print(f"  {_c}: {len(_f)} cards from {_src}")

# COMMAND ----------

# DBTITLE 1,Run the standard evaluation
import tempfile
STD_RESULTS = {}
_gamry_dirs = [Path(t.format(ev=_EV_ROOT, leepa=LEEPA)) for t in GAMRY_ROOTS]
_gamry_dir = next((d for d in _gamry_dirs if d.exists() and
                   (list(d.glob('*.dta')) + list(d.glob('*.DTA')))), None)
print(f"  Gamry folder: {_gamry_dir}")
for cond in CONDITIONS:
    files, reader, src = source_for(cond)
    if not files:
        print(f"  {cond}: no card files -- skipped")
        continue
    out = (RESULTS_ROOT if WRITE else Path(tempfile.gettempdir()) / 'eis_std') / LEEPA / cond
    cfg = std_pipeline.StdConfig(
        leepa=LEEPA, condition=cond, files=tuple(files), reader=reader,
        gamry_dir=_gamry_dir, gamry_version=GAMRY_VERSION,
        curr_cal=CURR_CAL, temp_cal=TEMP_CAL,
        gain_file=GAIN_FILE if GAIN_FILE.exists() else None,
        out_dir=out, f_min_hz=F_MIN_HZ, f_max_hz=F_MAX_HZ)
    print(f"\n── {cond} ({src}) ──")
    try:
        STD_RESULTS[cond] = std_pipeline.run(cfg)
    except SystemExit as e:
        print(f"  {cond}: {e}")
    except Exception as e:
        print(f"  {cond}: FAILED {type(e).__name__}: {e}")

# COMMAND ----------

# DBTITLE 1,Nyquist / Bode / phase of the measured spectra + Gamry (black) — Condition and Plot drop-downs
# every segment, coloured inlet -> outlet, and the condition's whole-cell
# Gamry sweep in black in all three views: every point of the .dta at the
# frequency the file gives, no band limit, nothing interpolated or removed
_figs = {c: nyquist.figure(r['spectra'][r['spectra'].kept],
                           title=f'{LEEPA} / {c} — standard evaluation (measured)',
                           gamry=r.get('gamry_full'), gamry_name=f'Gamry {c}')
         for c, r in STD_RESULTS.items()}
displayHTML(viewers.by_condition(_figs, title=f'Leepa {LEEPA} — measured spectra',
                                 panel_height=560, panel_width=1150))

# COMMAND ----------

# DBTITLE 1,Plate maps — 2D interpolated, every parameter
displayHTML(viewers.plate_maps({c: Path(r['out_dir']) / 'parameters.csv'
                                for c, r in STD_RESULTS.items()},
                               title=f'Leepa {LEEPA} — standard evaluation'))

# COMMAND ----------

# DBTITLE 1,Parameters and plate statistics (36 tiles)
for c, r in STD_RESULTS.items():
    P = r['parameters']
    m = P[P['class'] == 'measured']
    print(f"\n{c}: {len(m)} segments measured · fit verdicts "
          f"{m.fit_verdict.value_counts().to_dict()}")
    st = pd.read_csv(Path(r['out_dir']) / 'plate_statistics.csv')
    display(st[['parameter', 'unit', 'n_tiles', 'mean', 'median', 'sd', 'cv_pct', 'min', 'max']])
    display(m[['segment', 'R_ohmic', 'R_ohmic_how', 'R_ct', 'R_mt', 'R_pol', 'R_pol_lf',
               'f_ct_hz', 'f_mt_hz', 'fit_rms_pct', 'kk_res_pct', 'ReZ_1kHz', 'j_dc',
               'T_degC', 'n_points']].round(3))

# COMMAND ----------

# DBTITLE 1,Measured segments in parallel vs the Gamry whole cell — Condition and Plot drop-downs
# two curves only: the aggregate of the measured segments (red) and the
# Gamry (black, every point of the file). The per-frequency differences are
# in gamry_comparison.csv; the whole sweep as read is in gamry_full.csv.
displayHTML(viewers.by_condition(
    {c: nyquist.comparison_figure(
            r['aggregate'], r.get('gamry_full'),
            title=f'{LEEPA} / {c} — aggregate of the measured segments vs Gamry {c}',
            gamry_name=f'Gamry {c}')
     for c, r in STD_RESULTS.items()},
    title=f'Leepa {LEEPA} — aggregate of the measured spectra vs Gamry',
    panel_height=560, panel_width=1150))

# COMMAND ----------

# DBTITLE 1,Where each step was found (schedule per card)
for c, r in STD_RESULTS.items():
    s = r['schedule']
    print(f"{c}: {int(s.fit_start.notna().sum())}/{len(s)} card-steps evaluated; "
          f"not used: {s[s.fit_start.isna()].groupby('note').size().to_dict()}")
