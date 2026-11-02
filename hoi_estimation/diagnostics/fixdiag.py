# diagnostic (GT): compare the frame-0 camera fix of v3 with the oracle constant camera correction
import numpy as np, sys
sys.path.insert(0,'/home/hanzhou/Projects/v2d/hoi_extraction/scripts')
from eval_object_pose import quat_wxyz_to_R
V='/mnt/nas/Group/hanzhou/v2d'; E=f'{V}/experiments/tier2_refine'
def proj(M): u,_,vt=np.linalg.svd(M); d=np.sign(np.linalg.det(u@vt)); u[:,-1]*=d; return u@vt
def ang(A,B): return np.degrees(np.arccos(np.clip((np.trace(A.T@B)-1)/2,-1,1)))
for i in range(30):
    a=np.load(f'{E}/v3_cf0/episode_{i:06d}.npz'); b=np.load(f'{E}/v3_cf1/episode_{i:06d}.npz')
    ref=np.load(f'{V}/ref_poses/episode_{i:06d}.npz')['pose_xyz_wxyz'].astype(float); n=min(len(ref),len(a['visible']))
    g=a['accepted'][:n]&(np.linalg.norm(ref[:n,3:],axis=1)>.5); Rg=quat_wxyz_to_R(ref[:n][g][:,3:])
    Ra=a['T_world_mesh'][:n][g][:,:3,:3]; Rb=b['T_world_mesh'][:n][g][:,:3,:3]
    Q=proj(np.einsum('tij,tkj->ik',Rg,Ra)); D=proj(np.einsum('tij,tkj->ik',Rb,Ra))
    res=np.median([ang(Q@x,y) for x,y in zip(Ra,Rg)])
    print(f'{i:2d} oracle {ang(Q,np.eye(3)):5.1f}  fix {ang(D,np.eye(3)):5.1f}  fix-vs-oracle {ang(Q,D):5.1f}  resid-after-oracle {res:4.1f}')
