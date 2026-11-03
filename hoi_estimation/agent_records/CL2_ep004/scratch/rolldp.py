import sys; sys.path.insert(0,'/work/scratch')
from common import *
d=np.load('/work/scratch/rollmap.npz'); S=d['S']; valid=d['valid']; Q=d['Q']; angs=d['angs']
k=decode_roll(S,valid,[(0,110),(706,802)],cap=float(sys.argv[1]))
phi=np.degrees(angs[k])
print('static1 %.0f static2 %.0f'%(phi[50],phi[750]))
print(' '.join('%d:%d'%(i,phi[i]) for i in range(110,710,10)))
dphi=np.abs((np.diff(phi)+180)%360-180); print('roll step med %.1f p95 %.1f max %.1f'%(np.median(dphi[110:706]),np.percentile(dphi[110:706],95),dphi.max()))
