"""DIAGNOSTIC (GT human/object): object-vs-body relative depth ratio at each CARI4D stage (in -> pr_initial -> pr)."""
import json, sys
import numpy as np, torch, trimesh
sys.path.insert(0, "/workspace/v2d_cari4d/lib/cari4d"); sys.path.insert(0, "/home/hanzhou/Projects/v2d/hoi_extraction/scripts")
from eval_official import bundle_path
from lib_mhr.mhr_layer import MHRLayer
from tools.render_mhr_wild_inference import _params
import soma
V2D = "/mnt/nas/Group/hanzhou/v2d"; ds = f"{V2D}/dataset/track_2/tier_1_multiview_caption"
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{ds}/meta/episodes_metadata.jsonl"))}
al = json.load(open(f"{V2D}/logs/JC1_joint_camera_align.json"))
dev = torch.device("cuda")
mhr = MHRLayer.from_mhr_assets(mhr_assets_root=f"{V2D}/weights/sam3d_body", device="cuda")
sl = soma.SOMALayer(identity_model_type="mhr", device="cuda")
src = sys.argv[2] if len(sys.argv) > 2 else None
ff = json.load(open(f"{V2D}/experiments/FF2_feet_ref/feet_floor.json"))
for idx in [int(x) for x in sys.argv[1].split(",")]:
    ep = f"episode_{idx:06d}"
    b = torch.load(bundle_path(src or ff[str(idx)]["src"], ep), map_location="cpu", weights_only=False)
    g = np.load(f"{V2D}/diag_only_ref_human/{ep}.npz"); fr = g["frame"]; fr = fr[fr < len(b["pr"]["pose_abs"])]
    W = np.asarray(al[str(idx)]["sim"]); Rw, C = W[:3, :3], W[:3, 3]
    n = len(fr)
    with torch.no_grad():
        gv = sl(torch.tensor(g["pose"][:n]).float().view(n, 77, 3).to(dev), torch.tensor(g["identity"][:n]).float().to(dev),
                scale_params=torch.tensor(g["scale"][:n]).float().to(dev), transl=torch.tensor(g["transl"][:n]).float().to(dev)).vertices.cpu().numpy().astype(np.float64) @ np.diag([1.0, -1.0, -1.0])
    gb = np.linalg.norm(((gv - C) @ Rw).mean(1), axis=1)
    glb = trimesh.load(f"{ds}/{meta[idx]['mesh']}", force="scene", process=False).dump(concatenate=True)
    S = np.asarray(trimesh.bounds.oriented_bounds(glb)[0])
    ref = np.load(f"{V2D}/ref_poses/{ep}.npz")["pose_xyz_wxyz"].astype(np.float64)[fr]
    vis = np.linalg.norm(ref[:, 3:], axis=1) > 0.5
    go = np.linalg.norm((ref[:, :3] - C) @ Rw, axis=1)
    line = f"{idx:2d} {meta[idx]['object'][:10]:10s}"
    for st in ("in", "pr_initial", "pr"):
        d = b[st]
        P = np.asarray(d["pose_abs"], np.float64)
        try:
            with torch.no_grad():
                hv = mhr.mhr_forward(_params(d, fr, dev)).vertices.cpu().numpy().astype(np.float64)
        except Exception as e:
            line += f" | {st}: err {type(e).__name__}"; continue
        hb = np.linalg.norm(hv.mean(1), axis=1)
        ob = np.linalg.norm((P[fr] @ S)[:, :3, 3], axis=1)
        rb, ro = hb / gb, ob / go
        rel = (ro / rb)[vis]
        line += f" | {st}: body {np.median(rb):.3f} obj {np.median(ro[vis]):.3f} rel {np.median(rel):.3f}"
    print(line, flush=True)
