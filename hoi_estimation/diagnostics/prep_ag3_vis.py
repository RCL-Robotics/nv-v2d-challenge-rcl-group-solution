import sys, json, numpy as np, torch, trimesh
sys.path.insert(0, "/home/hanzhou/Projects/v2d/hoi_extraction/scripts")
from eval_official import bundle_path
from scipy.spatial.transform import Rotation as Rot
V2D = "/mnt/nas/Group/hanzhou/v2d"; DS = f"{V2D}/dataset/track_2/tier_1_multiview_caption"
REL = f"{V2D}/release/tier3_s1_20261020/reconstructed_data"
O = f"{V2D}/agent_runs/vis/ag3"
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{DS}/meta/episodes_metadata.jsonl"))}
al = json.load(open(f"{V2D}/logs/JC2s1_shared_f820_pp_align.json"))
for idx in map(int, sys.argv[1:]):
    ep, m = f"episode_{idx:06d}", meta[idx]
    S = np.asarray(trimesh.bounds.oriented_bounds(trimesh.load(f"{DS}/{m['mesh']}", force="scene", process=False).to_geometry())[0])
    W = np.asarray(al[str(idx)]["sim"])
    s1 = np.load(f"{REL}/{m['sequence_id']}/poses.npy")
    b = torch.load(bundle_path("AG3_agent_rot_K", ep), map_location="cpu", weights_only=False)
    ag = W[None] @ np.asarray(b["pr"]["pose_abs"], np.float64) @ S
    ref = np.load(f"{V2D}/ref_poses/{ep}.npz")["pose_xyz_wxyz"].astype(float)  # visualisation only
    gt = np.full((len(ref), 4, 4), np.nan); ok = np.linalg.norm(ref[:, 3:], axis=1) > 0.5
    gt[ok] = np.eye(4); gt[ok, :3, :3] = Rot.from_quat(ref[ok][:, [4, 5, 6, 3]]).as_matrix(); gt[ok, :3, 3] = ref[ok, :3]
    for k, T in [("s1", s1), ("ag3", ag), ("gt", gt)]:
        np.savez(f"{O}/ep{idx:03d}_{k}.npz", T_world_mesh=T)
    json.dump(json.load(open(f"{REL}/{m['sequence_id']}/camera.json")), open(f"{O}/ep{idx:03d}_camera.json", "w"))
    print(idx, len(s1), len(ag), len(gt), np.abs(ag[:, :3, 3] - s1[:, :3, 3]).max())
