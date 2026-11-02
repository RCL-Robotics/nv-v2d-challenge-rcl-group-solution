import sys; sys.path.insert(0,'/work/outputs')
from trajectory import *
import numpy as np, json
from scipy.spatial.transform import Rotation as R
rr=np.load('/work/inputs/reconstruction.npz'); Tw=rr['T_world_mesh'].astype(np.float64)
Twc=np.array(json.load(open('/work/inputs/camera.json'))['T_world_cam']); Tc0=np.linalg.inv(Twc)@Tw
vis=np.load('vis.npy'); occ=vis[:,2]/np.maximum(vis[:,1],1); w=np.clip(1-occ,0.02,1)**2
a,b=171,591
sub=Tc0[a:b]; ww=w[a:b].copy()
ho=np.zeros(len(sub),bool); ho[2::5]=True; ho&=ww>0.8
ww2=ww.copy(); ww2[ho]=0
for lam in [1,3,10,30,100]:
    Ts=smooth_se3(sub,ww2,lam_rot=lam,lam_tr=lam,iters=1500)
    dr=np.degrees(np.linalg.norm(R.from_matrix(np.einsum('nji,njk->nik',sub[:,:3,:3],Ts[:,:3,:3])).as_rotvec(),axis=1))
    dt=np.linalg.norm(Ts[:,:3,3]-sub[:,:3,3],axis=1)
    print('lam',lam,'heldout rot err %.2f deg  trans err %.2f cm | train rot %.2f trans %.2f'%(dr[ho].mean(),100*dt[ho].mean(),dr[~ho&(ww>0.8)].mean(),100*dt[~ho&(ww>0.8)].mean()))
