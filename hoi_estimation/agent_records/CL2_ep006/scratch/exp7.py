import sys, time; sys.path.insert(0,'/work/outputs')
from hoi_fit import *
torch.manual_seed(0)
S=setup(); pm=S['pm']
E=Evidence(S); sr=SilRenderer(decimate(S['mesh']),S['K'],S['Twc'],scale=0.5)
t0=time.time()
G=grid_costs(sr,E,pm,S['par_base'],log=print)
np.save('grid_v2.npy',G); print('time',time.time()-t0)
