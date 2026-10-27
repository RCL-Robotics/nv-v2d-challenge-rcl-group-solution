# Diagnostic (uses GT): camera-frame depth and rotation of s1 / Opus vs GT seen from each method's oracle camera, ep10.
import sys, json, numpy as np
sys.path.insert(0, "/home/hanzhou/Projects/v2d/hoi_extraction/scripts")
from eval_object_pose import umeyama, quat_wxyz_to_R, MPPE_OFFSETS
V="/mnt/nas/Group/hanzhou/v2d"; DS=f"{V}/dataset/track_2/tier_1_multiview_caption"; REL=f"{V}/release/tier3_s1_20261020/reconstructed_data"
i=int(sys.argv[1]); m={d["episode_index"]:d for d in map(json.loads,open(f"{DS}/meta/episodes_metadata.jsonl"))}[i]
W=np.array(json.load(open(f"{REL}/{m['sequence_id']}/camera.json"))["T_world_cam"]); Wi=np.linalg.inv(W)
ref=np.load(f"{V}/ref_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float); ok=np.linalg.norm(ref[:,3:],axis=1)>0.5
G=np.tile(np.eye(4),(len(ref),1,1)); G[ok,:3,:3]=quat_wxyz_to_R(ref[ok][:,3:]); G[:,:3,3]=ref[:,:3]
for lab,T in (("s1",np.load(f"{REL}/{m['sequence_id']}/poses.npy")),("opus",np.load(f"{V}/agent_runs/CL1_ep{i:03d}/outputs/object_poses_world.npz")["T_world_mesh"])):
    n=min(len(T),len(G)); k=ok[:n]; Tc=Wi[None]@T[:n]
    src=(Tc[k][:,None,:3,3]+np.einsum("tij,kj->tki",Tc[k][:,:3,:3],MPPE_OFFSETS)).reshape(-1,3); dst=(G[:n][k][:,None,:3,3]+np.einsum("tij,kj->tki",G[:n][k][:,:3,:3],MPPE_OFFSETS)).reshape(-1,3)
    _,R,t=umeyama(src,dst,False); M=np.eye(4); M[:3,:3],M[:3,3]=R,t; Gc=np.linalg.inv(M)[None]@G[:n]
    d=Tc[k][:,:3,3]-Gc[k][:,:3,3]; z=d[:,2]; xy=np.linalg.norm(d[:,:2],axis=1)
    ra=np.degrees(np.arccos(np.clip((np.einsum("tij,tij->t",Tc[k][:,:3,:3],Gc[k][:,:3,:3])-1)/2,-1,1)))
    cam=np.degrees(np.arccos(np.clip((np.trace(M[:3,:3].T@W[:3,:3])-1)/2,-1,1)))
    print(f"{lab:5s} est-vs-GT(in its oracle cam): depth err mean {z.mean()*100:+.1f} sd {z.std()*100:.1f} cm | lateral {np.median(xy)*100:.1f} cm | rot med {np.median(ra):.1f} deg | oracle cam differs from W by {cam:.2f} deg, {np.linalg.norm(M[:3,3]-W[:3,3])*100:.1f} cm")
    q=np.quantile(np.arange(k.sum()),[0,.25,.5,.75,1]).astype(int)
    print("      depth err by quarter (cm):", " ".join(f"{z[a:b].mean()*100:+.1f}" for a,b in zip(q[:-1],q[1:])), "| rot by quarter (deg):", " ".join(f"{np.median(ra[a:b]):.1f}" for a,b in zip(q[:-1],q[1:])))
