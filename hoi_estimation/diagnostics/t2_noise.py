# Diagnostic (uses GT): structure of the Tier 2 object noise (rotation / translation error, autocorrelation).
import os, numpy as np, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hoi_extraction", "scripts"))
from eval_object_pose import quat_wxyz_to_R
from scipy.spatial.transform import Rotation as Rot
V=os.environ["V2D_ROOT"]
def ac(x,l): x=x-x.mean(0); return float((x[:-l]*x[l:]).sum()/(x*x).sum())
for i in range(0,30,3):
    r=np.load(f"{V}/ref_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float); t=np.load(f"{V}/tier2_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float)
    ok=(np.linalg.norm(r[:,3:],axis=1)>.5)&(np.linalg.norm(t[:,3:],axis=1)>.5); r,t=r[ok],t[ok]
    Rr,Rt=quat_wxyz_to_R(r[:,3:]),quat_wxyz_to_R(t[:,3:])
    E=Rot.from_matrix(np.einsum("tji,tjk->tik",Rr,Rt)).as_rotvec()  # object-frame rotation error
    dt=t[:,:3]-r[:,:3]; ang=np.degrees(np.linalg.norm(E,axis=1))
    # frame-to-frame motion of GT vs noise
    gstep=np.degrees(np.linalg.norm(Rot.from_matrix(np.einsum("tji,tjk->tik",Rr[:-1],Rr[1:])).as_rotvec(),axis=1))
    print(i, f"rot err mean {ang.mean():.1f} med {np.median(ang):.1f} | mean rotvec(deg) {np.degrees(E.mean(0)).round(1)} | ac1 {ac(E,1):.2f} ac10 {ac(E,10):.2f} ac60 {ac(E,60):.2f} | trans err cm mean {np.linalg.norm(dt,axis=1).mean()*100:.1f} bias {np.linalg.norm(dt.mean(0))*100:.1f} ac1 {ac(dt,1):.2f} ac10 {ac(dt,10):.2f} ac60 {ac(dt,60):.2f} | gt step {gstep.mean():.2f}deg")
