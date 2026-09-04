import sys, os, pickle, time
sys.path.insert(0,'/home/claude/work/Local_EIS_updated'); os.chdir('/home/claude/work/Local_EIS_updated')
from pathlib import Path
import numpy as np, utils, silver
from config import DEFAULT
cfg = DEFAULT.replace(dat_dir=Path('/home/claude/dat'), out_dir=Path(sys.argv[1] if len(sys.argv)>1 else '/home/claude/results'),
                      curr_cal=Path('/home/claude/curr.csv'), temp_cal=Path('/home/claude/temp.csv'),
                      condition='45A', i_setpoint_a=45.0, equal_areas=(len(sys.argv)>2 and sys.argv[2]=='equal'))
log = utils.get_logger(True)
br = pickle.load(open('/home/claude/ck/bronze.pkl','rb'))
t=time.time(); sr = silver.run(br, cfg, log); print('silver %.1f s'%(time.time()-t))
silver.save(sr, cfg, log)
pickle.dump(sr, open('/home/claude/ck/silver_%s.pkl'%('equal' if cfg.equal_areas else 'true'),'wb'))
print('SILVER:', len(sr.spectra), 'spectra;', sr.tiers())
