import numpy as np, json
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz'); T=r['T_world_mesh'].astype(np.float64)
Rw=R.from_matrix(T[:,:3,:3])
w=(Rw[:-1].inv()*Rw[1:]).as_rotvec()  # body-frame increments
wd=np.degrees(np.linalg.norm(w,axis=1))
acc=np.degrees(np.linalg.norm(w[1:]-w[:-1],axis=1))
for a in range(165,600,15):
    print(a, 'mean |w| %.2f deg/f, mean |dw| %.2f'%(wd[a:a+15].mean(), acc[a:a+15].mean()), ' max |w| %.1f'%wd[a:a+15].max())
