import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from hoilib import load_inputs
K,Twc,rec,cam=load_inputs('/work/inputs')
def yawtilt(T):
    R=T[:,:3,:3]; up=R[:,:,1]; x=R[:,:,0]
    tilt=np.degrees(np.arccos(np.clip(-up[:,1],-1,1)))
    # yaw: project mesh x onto plane perpendicular to world up
    yaw=np.degrees(np.arctan2(x[:,2],x[:,0]))
    return yaw,tilt
srcs={'base':rec['T_world_mesh'].astype(np.float64),'fp':Twc@rec['T_cam_mesh_foundationpose'].astype(np.float64),'cc':Twc@rec['T_cam_mesh_coconet'].astype(np.float64)}
for a in sys.argv[1:]:
    srcs[a.split('/')[-1]]=np.load(a)['T_world_mesh']
fig,ax=plt.subplots(4,1,figsize=(18,14),sharex=True)
ms=np.load('/work/scratch/maskstats.npz')
for k,T in srcs.items():
    y,t=yawtilt(T); ax[0].plot(y,'.',ms=2,label=k); ax[1].plot(t,'.',ms=2,label=k)
    Tc=np.linalg.inv(Twc)@T; ax[2].plot(Tc[:,2,3],label=k)
ax[3].plot(ms['area']); 
for a in ax: a.grid(); a.legend()
ax[0].set_ylabel('yaw'); ax[1].set_ylabel('tilt'); ax[2].set_ylabel('zcam'); ax[2].set_ylim(1.5,6)
ax[0].set_xticks(range(0,980,20)); plt.tight_layout(); plt.savefig('/work/scratch/vis/yawplot.png',dpi=60)
