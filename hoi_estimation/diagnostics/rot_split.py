# Diagnostic (GT): split the fused rotation error on accepted frames into a constant world-frame part (camera rotation error)
# and the residual.
import json, sys, numpy as np
sys.path.insert(0, "/home/hanzhou/Projects/v2d/hoi_extraction/scripts")
from eval_object_pose import quat_wxyz_to_R
V="/mnt/nas/Group/hanzhou/v2d"
def proj(M): u,_,vt=np.linalg.svd(M); return u@np.diag([1,1,np.linalg.det(u@vt)])@vt
def ang(A,B): return np.degrees(np.arccos(np.clip((np.trace(np.einsum("...ji,...jk->...ik",A,B),axis1=-2,axis2=-1)-1)/2,-1,1)))
C=[]
for i in range(30):
    f=np.load(f"{V}/experiments/tier2_refine/'+__import__("os").environ.get("RUN","abl_45_5_1_0")+'/episode_{i:06d}.npz"); ref=np.load(f"{V}/ref_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float)
    n=min(len(ref),len(f["accepted"])); g=f["accepted"][:n]&(np.linalg.norm(ref[:n,3:],axis=1)>.5)
    if g.sum()<10: continue
    R=f["T_world_mesh"][:n][g][:,:3,:3]; Rr=quat_wxyz_to_R(ref[:n][g][:,3:])
    D=proj(np.einsum("tij,tkj->ik",Rr,R))  # constant world rotation  Rr ~ D R
    tot=np.median(ang(Rr,R)); res=np.median(ang(Rr,D[None]@R)); cst=ang(D,np.eye(3))
    C.append((tot,cst,res)); print(i, f"total {tot:.1f}  constant {cst:.1f}  residual {res:.1f}")
print("MEAN", np.mean(C,0).round(2))
