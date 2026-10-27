# Diagnostic (uses GT): rotation error of our camera-frame rotations after the best constant camera rotation (oracle),
# vs camera rotation from positions only (Tier 2 object translations, Umeyama sim3), vs Tier 2.
import json, sys, numpy as np
sys.path.insert(0, "/home/hanzhou/Projects/v2d/hoi_extraction/scripts")
from eval_object_pose import quat_wxyz_to_R, umeyama
from scipy.spatial.transform import Rotation as Rot
V="/mnt/nas/Group/hanzhou/v2d"; DS=f"{V}/dataset/track_2/tier_1_multiview_caption"; REL=f"{V}/release/tier3_s1_20261020/reconstructed_data"
meta={d["episode_index"]:d for d in map(json.loads,open(f"{DS}/meta/episodes_metadata.jsonl"))}
def opus(i): return f"{V}/agent_runs/{'CL2' if i<=10 else 'CL2b' if i<=21 else 'CL2c'}_ep{i:03d}/outputs/object_poses_world.npz"
def proj(M): u,_,vt=np.linalg.svd(M); return u@np.diag([1,1,np.linalg.det(u@vt)])@vt
def err(A,B): return np.degrees(np.linalg.norm(Rot.from_matrix(np.einsum("tji,tjk->tik",A,B)).as_rotvec(),axis=1))
acc=[]
for i in range(30):
    sid=meta[i]["sequence_id"]; Wi=np.linalg.inv(np.array(json.load(open(f"{REL}/{sid}/camera.json"))["T_world_cam"]))
    t2=np.load(f"{V}/tier2_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float); ref=np.load(f"{V}/ref_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float)
    Tc=Wi[None]@np.load(opus(i))["T_world_mesh"]; n=min(len(t2),len(ref),len(Tc))
    g=(np.linalg.norm(t2[:n,3:],axis=1)>.5)&(np.linalg.norm(ref[:n,3:],axis=1)>.5)
    Rr=quat_wxyz_to_R(ref[:n][g][:,3:]); R2=quat_wxyz_to_R(t2[:n][g][:,3:]); Rc_=Tc[:n][g][:,:3,:3]; pc=Tc[:n][g][:,:3,3]
    Ror=proj(np.einsum("tij,tkj->ik",Rr,Rc_))                   # oracle constant camera rotation
    s,Rp,tp=umeyama(pc,t2[:n][g][:,:3],True) if False else (None,None,None)
    # positions-only sim3 (camera rotation from the Tier 2 translation track)
    a=pc-pc.mean(0); b=t2[:n][g][:,:3]-t2[:n][g][:,:3].mean(0); U,S,Vt=np.linalg.svd(b.T@a); Rp=U@np.diag([1,1,np.linalg.det(U@Vt)])@Vt
    spread=np.sqrt(np.linalg.eigvalsh(np.cov(b.T)))*100
    r=[np.median(err(Rr,R2)), np.median(err(Rr,Ror[None]@Rc_)), np.median(err(Rr,Rp[None]@Rc_)), np.degrees(np.linalg.norm(Rot.from_matrix(Ror.T@Rp).as_rotvec()))]
    acc.append(r); print(i, meta[i]["object"][:14].ljust(14), "t2 %.0f | opus w/ oracle camR %.0f | opus w/ position-fit camR %.0f (camR err %.1f) | track spread cm %s"%(*r, spread.round(0)))
print("MEAN", np.mean(acc,0).round(1))
