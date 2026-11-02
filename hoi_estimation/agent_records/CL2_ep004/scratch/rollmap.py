import sys; sys.path.insert(0,'/work/scratch')
from common import *
from bat_lib import Renderer
ep,A=get_ep(); Rf=Renderer(ep.mesh)
T=np.load('/work/scratch/run1/object_poses_world.npz')['T_world_mesh']; Tc=ep.Tcw[None]@T
Q=transported_frames(Tc)
import time; t0=time.time()
angs,S,valid=roll_scores(ep,Rf,Tc,Q,list(range(ep.F)))
print('time',time.time()-t0)
np.savez('/work/scratch/rollmap.npz',angs=angs,S=S,valid=valid,Q=Q)
# relative roll of current trajectory w.r.t Q
cur=np.degrees(np.arctan2(np.einsum('fi,fi->f',Q[:,:,1],Tc[:,:3,0]),np.einsum('fi,fi->f',Q[:,:,0],Tc[:,:3,0])))%360
for i in range(0,803,10):
    s=S[i]; k=np.argmin(s); print(i,'best %3d  min %.2f mean %.2f  contrast %.2f  cur %3d'%(np.degrees(angs[k]),s.min(),s.mean(),s.mean()-s.min(),cur[i]))
