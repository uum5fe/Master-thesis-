import sys, time, pickle, os, json
sys.path.insert(0,'/home/claude/work/Local_EIS_updated'); os.chdir('/home/claude/work/Local_EIS_updated')
from pathlib import Path
import numpy as np, utils, bronze
from config import Config, DEFAULT
from eis_local import PlateCalibration
cfg = DEFAULT.replace(dat_dir=Path('/home/claude/dat'), out_dir=Path('/home/claude/results'),
                      curr_cal=Path('/home/claude/curr.csv'), temp_cal=Path('/home/claude/temp.csv'),
                      condition='45A', i_setpoint_a=45.0)
log = utils.get_logger(True); CK = Path('/home/claude/ck'); CK.mkdir(exist_ok=True)
files = bronze.discover_files(cfg)
def ck(name, fn):
    p = CK/f'{name}.pkl'
    if p.exists(): return pickle.load(open(p,'rb'))
    t=time.time(); v=fn(); print(f'  [{name}: {time.time()-t:.1f} s]'); sys.stdout.flush()
    pickle.dump(v, open(p,'wb')); return v
channels, cards = ck('inv', lambda: bronze.inventory_channels(files, cfg, log))
cal = PlateCalibration.load(cfg.curr_cal, cfg.temp_cal)
lags = ck('lags', lambda: bronze.estimate_card_lags(files, cards, cfg, log))
T_seg, sensor_T = ck('temp', lambda: bronze.plate_temperatures(files, cards, cal, cfg, log))
schedule, grid = ck('sched', lambda: bronze.consensus_schedule(files, cards, cfg, log, lags=lags))
print(f'  schedule: {len(schedule)} steps, grid ok={grid.get("ok")}')
spectra = {}
for fp in files:
    info = lags.get(fp.stem, {}); shift = int(info.get('lag',0)) if info.get('applied') else 0
    got = ck(f'card_{fp.stem[-1]}', lambda fp=fp, shift=shift:
             bronze.process_card(fp, cal, schedule, grid, cfg, log, T_seg=T_seg, lag=shift))
    spectra.update(got)
run_obj = bronze.BronzeRun(schedule=schedule, channels=channels, spectra=spectra, cards=cards,
    grid=grid, config_digest=bronze._digest([json.dumps(cfg.to_dict(), sort_keys=True)]),
    input_digest=bronze._digest([f"{p.name}:{p.stat().st_size}" for p in files]),
    n_files=len(files), lags=lags, sensor_T=sensor_T)
pickle.dump(run_obj, open(CK/'bronze.pkl','wb'))
print('BRONZE OK:', len(spectra), 'segments,', len(schedule), 'steps')
