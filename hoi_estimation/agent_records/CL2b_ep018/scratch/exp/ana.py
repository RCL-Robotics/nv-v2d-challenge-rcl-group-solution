import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, json
from hoi_lib import *
from scipy.spatial.transform import Rotation as Rot
D=load_inputs('/work/inputs'); mesh=load_mesh('/work/inputs/object_mesh.glb')
V=np.asarray(mesh.vertices)[::20]
Tb=D['rec']['T_world_mesh'].astype(float)
Tf=D['T_wc'][None]@np.load(sys.argv[1] if len(sys.argv)>1 else 'fit_all.npz')['T']
Tcb=np.linalg.inv(D['T_wc'])[None]@Tb; Tcf=np.linalg.inv(D['T_wc'])[None]@Tf
n,d=D['plane'][:3],D['plane'][3]
def yawtilt(T):
    X=T[:,:3,0]; Z=T[:,:3,2]; Xp=X-(X@n)[:,None]*n
    return np.degrees(np.arctan2(Xp[:,2],Xp[:,0])), np.degrees(np.arccos(np.abs(Z@n)))
def floor(T): return np.array([(V@t[:3,:3].T+t[:3,3])@n+d for t in T]).min(1)
yb,tb=yawtilt(Tb); yf,tf=yawtilt(Tf); fb=floor(Tb); ff=floor(Tf)
def add(Ta,Tb_):
    return np.array([np.linalg.norm((V@a[:3,:3].T+a[:3,3])-(V@b[:3,:3].T+b[:3,3]),axis=1).mean() for a,b in zip(Ta,Tb_)])
a=add(Tb,Tf)
phases={'S1':(0,252),'M1':(252,462),'S2':(462,659),'M2':(659,720),'S3':(720,858)}
for k,(s,e) in phases.items():
    sl=slice(s,e)
    print(k, 'yaw b %.1f f %.1f'%(np.median(yb[sl]),np.median(yf[sl])), 'tilt b %.1f f %.1f'%(np.median(tb[sl]),np.median(tf[sl])),
      'floor b %.3f f %.3f'%(np.median(fb[sl]),np.median(ff[sl])), 'z b %.3f f %.3f'%(np.median(Tcb[sl,2,3]),np.median(Tcf[sl,2,3])),
      'ADD(b,f) med %.3f'%np.median(a[sl]), 'std pos f', np.round(Tf[sl,:3,3].std(0),3), 'b',np.round(Tb[sl,:3,3].std(0),3))
if len(sys.argv)>2:
    for i in range(int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])):
        print(i,'yaw %.1f %.1f tilt %.1f %.1f floor %.3f %.3f z %.3f %.3f add %.3f'%(yb[i],yf[i],tb[i],tf[i],fb[i],ff[i],Tcb[i,2,3],Tcf[i,2,3],a[i]))
