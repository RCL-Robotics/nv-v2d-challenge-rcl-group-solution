# Diagnostic (GT): position variants on top of fuse (flips, no interp): Tier 2 position vs Tier 2 projected onto our viewing ray
# (keep Tier 2 range from the camera, take the direction from our image-side estimate) vs our mapped position.
import json, sys, numpy as np, trimesh
sys.path.insert(0, "/home/hanzhou/Projects/v2d/hoi_extraction/scripts")
from eval_object_pose import metrics, quat_wxyz_to_R
V="/mnt/nas/Group/hanzhou/v2d"; DS=f"{V}/dataset/track_2/tier_1_multiview_caption"; REL=f"{V}/release/tier3_s1_20261020/reconstructed_data"
meta={d["episode_index"]:d for d in map(json.loads,open(f"{DS}/meta/episodes_metadata.jsonl"))}; cams=json.load(open(f"{V}/logs/t2r_cam_human.json"))
def opus(i): return f"{V}/agent_runs/{'CL2' if i<=10 else 'CL2b' if i<=21 else 'CL2c'}_ep{i:03d}/outputs/object_poses_world.npz"
A=[]
for i in range(30):
    sid=meta[i]["sequence_id"]; c=cams[str(i)]; s,Rc,tc=c["s"],np.array(c["R"]),np.array(c["t"])
    Wi=np.linalg.inv(np.array(json.load(open(f"{REL}/{sid}/camera.json"))["T_world_cam"]))
    f=np.load(f"{V}/experiments/tier2_refine/abl_45_5_1_0/episode_{i:06d}.npz"); T=f["T_world_mesh"]; vis=f["visible"]
    Tc=Wi[None]@np.load(opus(i))["T_world_mesh"]; ref=np.load(f"{V}/ref_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float)
    n=min(len(T),len(Tc),len(ref)); g=vis[:n]&(np.linalg.norm(ref[:n,3:],axis=1)>.5)
    p2=T[:n][g][:,:3,3]; pm=s*Tc[:n][g][:,:3,3]@Rc.T+tc; d=(pm-tc)/np.linalg.norm(pm-tc,axis=1,keepdims=True)
    pray=tc+d*((p2-tc)*d).sum(1,keepdims=True)
    Rr=quat_wxyz_to_R(ref[:n][g][:,3:]); pr=ref[:n][g][:,:3]; R=T[:n][g][:,:3,:3]
    np.random.seed(0); v=np.asarray(trimesh.load(f"{DS}/{meta[i]['mesh']}",force="scene",process=False).to_geometry().sample(1000))
    r=[metrics(R,p,Rr,pr,v)["AUC"] for p in (p2,pray,pm)]+[np.median(np.linalg.norm(p-pr,axis=1))*100 for p in (p2,pray,pm)]
    A.append(r); print(i, " ".join(f"{x:.3f}" for x in r[:3]), " pos cm", " ".join(f"{x:.1f}" for x in r[3:]))
A=np.array(A); print("MEAN t2pos / ray / mapped:", A.mean(0).round(3))
