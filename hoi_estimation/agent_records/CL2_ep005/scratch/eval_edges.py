import sys; sys.path.insert(0,'/work/scratch')
from edge_score import *
T=d['rec']['T_world_mesh'].astype(np.float64); Tc=cam_from_world(d['T_wc'],T)
idx=list(range(0,1003,10))
res={'base':score(Tc,idx)}
for fn_ in sys.argv[1:]:
    res[fn_]=score(np.load(fn_),idx)
names=list(res)
print('frames   '+'  '.join(names))
for a in range(0,1003,100):
    sl=slice(a//10,(a+100)//10); print(a, '  '.join(f'{np.nanmean(res[n][sl]):.3f}' for n in names))
print('ALL', '  '.join(f'{np.nanmean(res[n]):.3f}' for n in names))
