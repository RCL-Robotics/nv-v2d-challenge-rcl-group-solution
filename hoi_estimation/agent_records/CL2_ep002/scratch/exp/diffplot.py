import sys; sys.path.insert(0,'/work/outputs')
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from hoilib import *
K,Twc,rec,cam=load_inputs('/work/inputs')
kp=rec['human_keypoints_world'].astype(np.float64)
def hand_frames(kp,base):
    h=kp[:,base:base+21]; wr=h[:,20]; mcp=h[:,[3,7,11,15,19]]
    x=mcp.mean(1)-wr; x/=np.linalg.norm(x,axis=1,keepdims=True)
    a=mcp[:,0]-mcp[:,4]; z=np.cross(x,a); z/=np.linalg.norm(z,axis=1,keepdims=True); y=np.cross(z,x)
    return np.stack([x,y,z],-1), wr
def yaw(R): x=R[:,:,0]; return np.degrees(np.arctan2(x[:,2],x[:,0]))
srcs={'base':rec['T_world_mesh'].astype(np.float64),'fp':Twc@rec['T_cam_mesh_foundationpose'].astype(np.float64),'cc':Twc@rec['T_cam_mesh_coconet'].astype(np.float64)}
fig,ax=plt.subplots(3,1,figsize=(18,12),sharex=True)
for hi,base in enumerate([21,42]):
    Rh,wr=hand_frames(kp,base); hy=np.degrees(np.arctan2(Rh[:,2,0],Rh[:,0,0]))
    ax[2].plot(hy,label=f'handyaw {base}')
    for k,T in srcs.items():
        d=((yaw(T[:,:3,:3])-hy+180)%360)-180
        ax[hi].plot(d,'.',ms=3,label=k)
for k,T in srcs.items(): ax[2].plot(yaw(T[:,:3,:3]),'.',ms=2,label=k)
for a in ax: a.grid(); a.legend()
ax[0].set_title('objyaw - R hand yaw'); ax[1].set_title('objyaw - L hand yaw')
ax[0].set_xticks(range(0,980,20)); plt.tight_layout(); plt.savefig('/work/scratch/vis/diff.png',dpi=55)
