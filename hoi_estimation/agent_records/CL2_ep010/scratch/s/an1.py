import numpy as np, json
from scipy.spatial.transform import Rotation as R
r=np.load('/work/inputs/reconstruction.npz')
T=r['T_world_mesh'].astype(np.float64)
cam=json.load(open('/work/inputs/camera.json')); Twc=np.array(cam['T_world_cam'])
Tc=np.linalg.inv(Twc)@T
F=len(T)
rv=R.from_matrix(T[:,:3,:3])
d0=(rv*rv[0].inv()).magnitude()*180/np.pi
dstep=np.r_[0,(rv[1:]*rv[:-1].inv()).magnitude()*180/np.pi]
tstep=np.r_[0,np.linalg.norm(np.diff(T[:,:3,3],axis=0),axis=1)]
fp=r['T_cam_mesh_foundationpose'].astype(np.float64); cc=r['T_cam_mesh_coconet'].astype(np.float64)
rf=R.from_matrix(fp[:,:3,:3]); rc=R.from_matrix(cc[:,:3,:3]); rb=R.from_matrix(Tc[:,:3,:3])
dfb=(rf*rb.inv()).magnitude()*180/np.pi; dcb=(rc*rb.inv()).magnitude()*180/np.pi
cp=r['contact_prob_right_left']
for i in range(0,F,10):
    print(i, f"rot0 {d0[i]:6.1f} step {dstep[i]:5.1f} tstep {tstep[i]*100:5.1f}cm  tw {np.round(T[i,:3,3],3)} zc {Tc[i,2,3]:.3f} zfp {fp[i,2,3]:.3f} zcc {cc[i,2,3]:.3f} dFP {dfb[i]:5.1f} dCC {dcb[i]:5.1f} cp {np.round(cp[i],2)}")
print('max steps', np.argsort(dstep)[-20:], np.sort(dstep)[-20:])
