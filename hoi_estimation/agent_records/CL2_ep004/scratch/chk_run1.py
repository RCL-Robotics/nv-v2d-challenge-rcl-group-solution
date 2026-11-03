import sys; sys.path.insert(0,'/work/scratch')
from common import *
from bat_lib import rotmat_angle
ep=Episode('/work/inputs')
T=np.load('/work/scratch/run2/object_poses_world.npz')['T_world_mesh']; Tc=ep.Tcw[None]@T
d=np.load('/work/scratch/run1/validation_details.npz')
for i in [0,750]:
    print(i,'axis vs base %.1f deg'%np.degrees(np.arccos(np.clip(Tc[i,:3,2]@ep.Tc_base[i,:3,2],-1,1))), 'rot vs base %.1f'%rotmat_angle(Tc[i,:3,:3],ep.Tc_base[i,:3,:3]))
