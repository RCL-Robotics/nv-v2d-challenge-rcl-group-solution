import numpy as np, json
from scipy.spatial.transform import Rotation as Rot
r=np.load('/work/inputs/reconstruction.npz'); c=json.load(open('/work/inputs/camera.json'))
Twc=np.array(c['T_world_cam']); T=r['T_world_mesh'].astype(float)
Tc=np.linalg.inv(Twc)@T
fp=r['T_cam_mesh_foundationpose'].astype(float); cc=r['T_cam_mesh_coconet'].astype(float)
def ang(A,B): return np.degrees(Rot.from_matrix(A[:3,:3].T@B[:3,:3]).magnitude())
F=len(T)
for t in range(0,F,6):
    print(t, 'z=%.3f'%Tc[t,2,3], 'xyz_w=',np.round(T[t,:3,3],3), 'dR=%.1f'%ang(T[t-1],T[t]) if t else '', 'R_vs0=%.0f'%ang(T[0],T[t]), 'bFP=%.0f'%ang(Tc[t],fp[t]), 'bCC=%.0f'%ang(Tc[t],cc[t]), 'zfp=%.2f zcc=%.2f'%(fp[t,2,3],cc[t,2,3]), 'contact',np.round(r['contact_prob_right_left'][t],2))
