import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoi_lib import *
inp='/work/inputs'
cam,K,Twc,rec,plane=load_inputs(inp)
mesh=load_mesh(inp)
n,d=plane[:3],plane[3]
T=rec['T_world_mesh'].astype(np.float64)
v=np.asarray(mesh.vertices)
F=len(T)
# which mesh axis is "up": average over frames of R^T (-n)
up_mesh=np.array([T[i,:3,:3].T@(-n) for i in range(F)])
print('mean up in mesh coords', up_mesh.mean(0), np.linalg.norm(up_mesh.mean(0)))
u0=up_mesh.mean(0); u0/=np.linalg.norm(u0)
ang=np.degrees(np.arccos(np.clip(up_mesh@u0,-1,1)))
h=np.array([ (v@T[i,:3,:3].T+T[i,:3,3])@n+d for i in range(0,F)])  # (F,N)
low=np.percentile(h,0.5,axis=1); hmin=h.min(1)
for i in range(0,F,20): print(i,'tilt %.1f'%ang[i],'lowest %.3f p0.5 %.3f'%(hmin[i],low[i]))
