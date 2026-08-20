import sys, os, pickle, time
sys.path.insert(0,'/home/claude/work/Local_EIS_updated'); os.chdir('/home/claude/work/Local_EIS_updated')
from pathlib import Path
import numpy as np, utils, gold
from config import DEFAULT
out = Path(sys.argv[1])
cfg = DEFAULT.replace(dat_dir=Path('/home/claude/dat'), out_dir=out,
                      curr_cal=Path('/home/claude/curr.csv'), temp_cal=Path('/home/claude/temp.csv'),
                      condition='45A', i_setpoint_a=45.0,
                      equal_areas=(len(sys.argv)>2 and sys.argv[2]=='equal'))
log = utils.get_logger(True)
sr = pickle.load(open('/home/claude/ck/silver.pkl','rb'))
t=time.time(); gr = gold.run(sr, cfg, log); gold.save(gr, sr, cfg, log)
print('gold %.1f s'%(time.time()-t))
st=gr.stats
print('GOLD:', st['n_measured'],'measured,',st['n_inferred'],'inferred,',st['n_bad'],'bad')
if 'R_ohmic' in st:
    r=st['R_ohmic']; print('R_ohmic %.1f +/- %.1f mOhm cm2, spread %.2fx'%(1000*r['mean'],1000*r['sd'],r['spread']))
