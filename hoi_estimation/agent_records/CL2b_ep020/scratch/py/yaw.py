import sys; sys.path.insert(0,'/work/outputs')
import numpy as np
from hoi_lib import *
K,Twc,Kp,W,H=load_camera('/work/inputs')
r=np.load('/work/inputs/reconstruction.npz')
Twm=r['T_world_mesh'].astype(np.float64)
Rw=Twm[:,:3,:3]
# yaw: direction of mesh x axis projected to floor plane (world xz)
def yaw(R):
    a=R[:,:,0]; return np.degrees(np.arctan2(a[:,2],a[:,0]))
y=np.unwrap(np.radians(yaw(Rw))); y=np.degrees(y)
fp=np.linalg.inv(Twc)[None]  # cam->... we want world rot of fp: Twc@Tcm
for name,T in [('base',Twm),('fp',Twc@r['T_cam_mesh_foundationpose']),('cc',Twc@r['T_cam_mesh_coconet'])]:
    yy=np.degrees(np.unwrap(np.radians(yaw(T[:,:3,:3]))))
    print(name,' '.join('%d:%.0f'%(i,yy[i]-yy[0]) for i in range(0,553,10)))
