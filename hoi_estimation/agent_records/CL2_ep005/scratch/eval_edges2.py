import sys; sys.path.insert(0,'/work/scratch')
from edge_score import *
T=d['rec']['T_world_mesh'].astype(np.float64); Tc=cam_from_world(d['T_wc'],T)
idx=list(range(int(sys.argv[1]),1003,10))
res={'base':score(Tc,idx)}
for fn_ in sys.argv[2:]:
    res[fn_.split('/')[-1]]=score(np.load(fn_),idx)
names=list(res)
print('frames   '+'  '.join(names))
for a in range(0,1003,100):
    sl=[k for k,i in enumerate(idx) if a<=i<a+100]; print(a, '  '.join(f'{np.nanmean(res[n][sl]):.3f}' for n in names))
print('ALL', '  '.join(f'{np.nanmean(res[n]):.3f}' for n in names))
