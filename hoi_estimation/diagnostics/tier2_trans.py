# Diagnostic (uses GT): released Tier 2 noisy trajectory, raw and translation-only AUC, same code as oracle_trans.py.
import sys, json, numpy as np, trimesh
sys.path.insert(0, "/home/hanzhou/Projects/v2d/hoi_extraction/scripts")
from eval_object_pose import metrics, quat_wxyz_to_R, MPPE_OFFSETS
V="/mnt/nas/Group/hanzhou/v2d"; DS=f"{V}/dataset/track_2/tier_1_multiview_caption"
meta={d["episode_index"]:d for d in map(json.loads,open(f"{DS}/meta/episodes_metadata.jsonl"))}
res={}
for i in range(30):
    m=meta[i]; ref=np.load(f"{V}/ref_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float)
    t2=np.load(f"{V}/tier2_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float)
    n=min(len(t2),len(ref)); k=(np.linalg.norm(ref[:n,3:],axis=1)>0.5)&(np.linalg.norm(t2[:n,3:],axis=1)>0.5)
    np.random.seed(0); v=np.asarray(trimesh.load(f"{DS}/{m['mesh']}",force="scene",process=False).to_geometry().sample(1000))
    Rr=quat_wxyz_to_R(ref[:n][k][:,3:]); tr=ref[:n][k][:,:3]; R2=quat_wxyz_to_R(t2[:n][k][:,3:]); t=t2[:n][k][:,:3]
    pe=(t[:,None]+np.einsum("tij,kj->tki",R2,MPPE_OFFSETS)).reshape(-1,3); pg=(tr[:,None]+np.einsum("tij,kj->tki",Rr,MPPE_OFFSETS)).reshape(-1,3)
    d=(pg-pe).mean(0); raw=metrics(R2,t,Rr,tr,v)["AUC"]; at=metrics(R2,t+d,Rr,tr,v)["AUC"]
    res[i]=(raw,at); print(i, m["object"], f"raw {raw:.3f} trans-only {at:.3f} shift {np.linalg.norm(d)*100:.1f}cm", flush=True)
json.dump(res, open(sys.argv[1],"w"))
