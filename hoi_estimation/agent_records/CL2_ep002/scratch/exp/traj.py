import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoilib import load_inputs
K,Twc,rec,cam=load_inputs('/work/inputs')
key=sys.argv[1] if len(sys.argv)>1 else 'T_world_mesh'
if key in rec.files and key!='T_world_mesh':
    T=Twc@rec[key].astype(np.float64)
else: T=rec['T_world_mesh'].astype(np.float64)
ms=np.load('/work/scratch/maskstats.npz')
Tcw=np.linalg.inv(Twc)
for i in range(0,980,int(sys.argv[2]) if len(sys.argv)>2 else 10):
    R=T[i,:3,:3]; t=T[i,:3,3]
    up=R[:,1]  # mesh y in world
    tilt=np.degrees(np.arccos(np.clip(-up[1],-1,1)))  # angle between mesh y and world up(-y)... sign unknown
    x=R[:,0]; yaw=np.degrees(np.arctan2(x[2],x[0]))
    tc=(Tcw@T[i])[:3,3]
    dR=np.degrees(np.arccos(np.clip((np.trace(T[i-1,:3,:3].T@R)-1)/2,-1,1))) if i>0 else 0
    print(f"{i:4d} pos {t[0]:6.3f} {t[1]:6.3f} {t[2]:6.3f} zc {tc[2]:5.2f} tilt {tilt:6.1f} yaw {yaw:7.1f} dR {dR:5.1f} area {ms['area'][i]}")
