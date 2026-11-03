import numpy as np, json
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz')
T=r['T_world_mesh'].astype(np.float64)
cam=json.load(open('/work/inputs/camera.json')); Twc=np.array(cam['T_world_cam'])
Tc=np.linalg.inv(Twc)@T
fp=r['T_cam_mesh_foundationpose'].astype(np.float64); cc=r['T_cam_mesh_coconet'].astype(np.float64)
cp=r['contact_prob_right_left']
def ang(A,B): 
    return np.degrees(np.linalg.norm(R.from_matrix(A[...,:3,:3].transpose(0,2,1)@B[...,:3,:3]).as_rotvec(),axis=-1))
dR=ang(T[:-1],T[1:]); dt=np.linalg.norm(T[1:,:3,3]-T[:-1,:3,3],axis=1)
for i in range(0,715,5):
    print(i, 'pos',np.round(T[i,:3,3],3),'camz %.3f'%Tc[i,2,3], 'fpz %.3f ccz %.3f'%(fp[i,2,3],cc[i,2,3]), 'dR %.1f dt %.3f'%(dR[min(i,713)],dt[min(i,713)]), 'cont',np.round(cp[i],2),
      'up',np.round(T[i,:3,:3]@np.array([0,0,1.]),2), 'R-fp %.1f R-cc %.1f'%(ang(Tc[i:i+1],fp[i:i+1])[0],ang(Tc[i:i+1],cc[i:i+1])[0]))
