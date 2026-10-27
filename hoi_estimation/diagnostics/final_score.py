# Diagnostic (uses GT): raw and translation-only ADD AUC for s1, DeepSeek T1, Opus v2 (CL2/CL2b/CL2c) on all 30 episodes.
import sys, os, json, numpy as np, trimesh
sys.path.insert(0, "/home/hanzhou/Projects/v2d/hoi_extraction/scripts")
from eval_object_pose import metrics, quat_wxyz_to_R, MPPE_OFFSETS
V="/mnt/nas/Group/hanzhou/v2d"; DS=f"{V}/dataset/track_2/tier_1_multiview_caption"; REL=f"{V}/release/tier3_s1_20261020/reconstructed_data"
meta={d["episode_index"]:d for d in map(json.loads,open(f"{DS}/meta/episodes_metadata.jsonl"))}
def opus(i): return f"{V}/agent_runs/{'CL2' if i<=10 else 'CL2b' if i<=21 else 'CL2c'}_ep{i:03d}/outputs/object_poses_world.npz"
res={}
for i in range(30):
    m=meta[i]; ref=np.load(f"{V}/ref_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float); ok=np.linalg.norm(ref[:,3:],axis=1)>0.5
    np.random.seed(0); v=np.asarray(trimesh.load(f"{DS}/{m['mesh']}",force="scene",process=False).to_geometry().sample(1000))
    r={"object":m["object"]}
    for lab,p in (("s1",None),("T1",f"{V}/agent_runs/T1_ds_ep{i:03d}/outputs/object_poses_world.npz"),("opus",opus(i))):
        if p and not os.path.isfile(p): r[lab]=None; continue
        T=np.load(f"{REL}/{m['sequence_id']}/poses.npy") if p is None else np.load(p)["T_world_mesh"]
        n=min(len(T),len(ref)); k=ok[:n]; Rr=quat_wxyz_to_R(ref[:n][k][:,3:]); tr=ref[:n][k][:,:3]; Tk=T[:n][k]
        pe=(Tk[:,None,:3,3]+np.einsum("tij,kj->tki",Tk[:,:3,:3],MPPE_OFFSETS)).reshape(-1,3); pg=(tr[:,None]+np.einsum("tij,kj->tki",Rr,MPPE_OFFSETS)).reshape(-1,3)
        d=(pg-pe).mean(0); r[lab]=[metrics(Tk[:,:3,:3],Tk[:,:3,3],Rr,tr,v)["AUC"], metrics(Tk[:,:3,:3],Tk[:,:3,3]+d,Rr,tr,v)["AUC"]]
    res[i]=r; print(i, r, flush=True)
json.dump(res, open(sys.argv[1],"w"))
