import sys; sys.path.insert(0,'/work/outputs')
from trajectory import *
import numpy as np, json, h5py, time
rr=np.load('/work/inputs/reconstruction.npz'); Tw=rr['T_world_mesh'].astype(np.float64)
Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']); Tc0=np.linalg.inv(Twc)@Tw
f=h5py.File('/work/inputs/masks.h5'); O=f['object']; P=f['person']
F=len(Tw)
t0=time.time()
s_end,e_start=detect_static_phases(O,P)
print('static start [0,%d)  end [%d,%d)'%(s_end,e_start,F), time.time()-t0)
vis=np.load('vis.npy'); occ=vis[:,2]/np.maximum(vis[:,1],1)
w=np.clip(1-occ,0.02,1)**2
TA=static_pose_cam(Tc0[:s_end]); TB=static_pose_cam(Tc0[e_start:])
fixed={k:TA for k in range(s_end)}; fixed.update({k:TB for k in range(e_start,F)})
lam=float(sys.argv[1]) if len(sys.argv)>1 else 30
Ts=smooth_se3(Tc0,w,fixed=fixed,lam_rot=lam,lam_tr=lam)
np.save(f'draft1_cam_{lam}.npy',Ts)
from scipy.spatial.transform import Rotation as R
dr=np.degrees(np.linalg.norm(R.from_matrix(np.einsum('nji,njk->nik',Tc0[:,:3,:3],Ts[:,:3,:3])).as_rotvec(),axis=1))
dt=np.linalg.norm(Ts[:,:3,3]-Tc0[:,:3,3],axis=1)
for a,b in [(0,s_end),(s_end,e_start),(e_start,F),(430,510),(380,405)]:
    print('[%d,%d) mean dR %.2f deg, mean dt %.3f m, max dR %.1f'%(a,b,dr[a:b].mean(),dt[a:b].mean(),dr[a:b].max()))
print(time.time()-t0)
