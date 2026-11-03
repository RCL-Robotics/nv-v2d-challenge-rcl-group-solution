import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from hoi_lib import *
inp='/work/inputs'
cam,K,Twc,rec,plane=load_inputs(inp); mesh=load_mesh(inp); o=Twc[:3,3]
fm=FloorModel(mesh,plane,np.array([-0.28,-0.02,-0.96]),h0=0.014)
fig,ax=plt.subplots(4,1,figsize=(16,13),sharex=True)
for p,lab in [(None,'base')]+[(x,x.split('/')[-1]) for x in sys.argv[1:]]:
    T=rec['T_world_mesh'].astype(np.float64) if p is None else np.load(p)['T_world_mesh']
    P=np.array([fm.from_pose(t,o) for t in T]); P[:,0]=np.unwrap(P[:,0])
    ax[0].plot(np.degrees(P[:,0]),label=lab,lw=0.8); ax[1].plot(P[:,1],lw=0.8); ax[2].plot(P[:,2],lw=0.8)
    z=(np.linalg.inv(Twc)@T)[:,2,3]; ax[3].plot(z,lw=0.8)
ax[0].legend(); ax[0].set_ylabel('yaw deg'); ax[1].set_ylabel('a'); ax[2].set_ylabel('b'); ax[3].set_ylabel('cam z')
for a in ax: a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('/work/scratch/frames/series.png',dpi=60)
