import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoi_lib import *
K,Twc,Kp,W,H=load_camera('/work/inputs')
r=np.load('/work/inputs/reconstruction.npz')
Twm=r['T_world_mesh'].astype(np.float64)
Tcm=np.linalg.inv(Twc)@Twm
fp=r['T_cam_mesh_foundationpose'].astype(np.float64); cc=r['T_cam_mesh_coconet'].astype(np.float64)
Rw=Twm[:,:3,:3]
# which mesh axis is aligned with world up (-y)
up=np.array([0,-1,0.])
ax=np.einsum('fij,i->fj',Rw,up)  # mesh-frame coords of world-up
print('mesh-frame up vector at frames:'); 
for i in range(0,553,25): 
    print(i, np.round(ax[i],3), 'ang0 %.1f'%rot_angle_deg(Rw[0],Rw[i]), 'step %.2f'%(rot_angle_deg(Rw[i-1],Rw[i]) if i else 0),
      'tcam',np.round(Tcm[i,:3,3],3),'fp',np.round(fp[i,:3,3],3),'cc',np.round(cc[i,:3,3],3),
      'R(base vs fp) %.1f'%rot_angle_deg(Tcm[i,:3,:3],fp[i,:3,:3]),'R(base vs cc) %.1f'%rot_angle_deg(Tcm[i,:3,:3],cc[i,:3,:3]))
st=np.array([0]+[rot_angle_deg(Rw[i-1],Rw[i]) for i in range(1,553)])
print('big steps', [(i,round(s,1)) for i,s in enumerate(st) if s>3])
print('contact', np.round(r['contact_prob_right_left'][::25],2).tolist())
