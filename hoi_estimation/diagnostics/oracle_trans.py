# Diagnostic (uses GT): oracle with a TRANSLATION-only world shift (camera rotation kept) and the rigid oracle's camera change.
import sys, os, json, numpy as np, trimesh
sys.path.insert(0, "/home/hanzhou/Projects/v2d/hoi_extraction/scripts")
from eval_object_pose import metrics, umeyama, quat_wxyz_to_R, MPPE_OFFSETS
V="/mnt/nas/Group/hanzhou/v2d"; DS=f"{V}/dataset/track_2/tier_1_multiview_caption"; REL=f"{V}/release/tier3_s1_20261020/reconstructed_data"
PFX=os.environ.get('PFX','CL1')
meta={d["episode_index"]:d for d in map(json.loads,open(f"{DS}/meta/episodes_metadata.jsonl"))}
for i in map(int, sys.argv[1].split(",")):
    m=meta[i]; W=np.array(json.load(open(f"{REL}/{m['sequence_id']}/camera.json"))["T_world_cam"]); Wi=np.linalg.inv(W)
    ref=np.load(f"{V}/ref_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float); ok=np.linalg.norm(ref[:,3:],axis=1)>0.5
    np.random.seed(0); v=np.asarray(trimesh.load(f"{DS}/{m['mesh']}",force="scene",process=False).to_geometry().sample(1000))
    out=[]
    for lab,T in (("s1",np.load(f"{REL}/{m['sequence_id']}/poses.npy")),("opus",np.load(f"{V}/agent_runs/{PFX}_ep{i:03d}/outputs/object_poses_world.npz")["T_world_mesh"])):
        n=min(len(T),len(ref)); k=ok[:n]; Rr=quat_wxyz_to_R(ref[:n][k][:,3:]); tr=ref[:n][k][:,:3]; Tk=T[:n][k]
        pe=(Tk[:,None,:3,3]+np.einsum("tij,kj->tki",Tk[:,:3,:3],MPPE_OFFSETS)).reshape(-1,3); pg=(tr[:,None]+np.einsum("tij,kj->tki",Rr,MPPE_OFFSETS)).reshape(-1,3)
        d=(pg-pe).mean(0); at=metrics(Tk[:,:3,:3],Tk[:,:3,3]+d,Rr,tr,v)["AUC"]
        Tc=Wi[None]@Tk; pc=(Tc[:,None,:3,3]+np.einsum("tij,kj->tki",Tc[:,:3,:3],MPPE_OFFSETS)).reshape(-1,3); _,R,t=umeyama(pc,pg,False)
        ang=np.degrees(np.arccos(np.clip((np.trace(R.T@W[:3,:3])-1)/2,-1,1)))
        out.append(f"{lab} raw {metrics(Tk[:,:3,:3],Tk[:,:3,3],Rr,tr,v)['AUC']:.3f} trans-only {at:.3f} (shift {np.linalg.norm(d)*100:.1f}cm) | rigid cam change {ang:.1f}deg {np.linalg.norm(t-W[:3,3])*100:.0f}cm")
    print(i, m["object"], " || ".join(out), flush=True)
