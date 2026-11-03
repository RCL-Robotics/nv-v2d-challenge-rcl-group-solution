"""DIAGNOSTIC ONLY (uses full GT human): depth ratio ours/GT along the camera ray for human body parts vs object."""
import json, os, sys
import numpy as np, torch, trimesh
sys.path.insert(0, "/workspace/v2d_cari4d/lib/cari4d")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hoi_extraction", "scripts"))
from eval_object_pose import quat_wxyz_to_R
from eval_official import bundle_path
from lib_mhr.mhr_layer import MHRLayer
from tools.render_mhr_wild_inference import _params
import soma

V2D = os.environ["V2D_ROOT"]
ds = f"{V2D}/dataset/track_2/tier_1_multiview_caption"
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{ds}/meta/episodes_metadata.jsonl"))}
al = json.load(open(f"{V2D}/logs/pnp_forE1.json"))
ff = json.load(open(f"{V2D}/experiments/FF2_feet_ref/feet_floor.json"))
dev = torch.device("cuda")
mhr = MHRLayer.from_mhr_assets(mhr_assets_root=f"{V2D}/weights/sam3d_body", device="cuda")
sl = soma.SOMALayer(identity_model_type="mhr", device="cuda")
eps = [int(x) for x in sys.argv[1].split(",")]
print("ep obj        | ratio ours/GT (median, [p10,p90]):  body   feet   hands  object | corr(obj,hand) corr(obj,body)")
res = {}
for idx in eps:
    ep = f"episode_{idx:06d}"
    b = torch.load(bundle_path("FF2_feet_ref", ep), map_location="cpu", weights_only=False)
    bs = torch.load(bundle_path(ff[str(idx)]["src"], ep), map_location="cpu", weights_only=False)
    P = np.asarray(b["pr"]["pose_abs"], np.float64); Ps = np.asarray(bs["pr"]["pose_abs"], np.float64)
    f = np.einsum("ti,ti->t", P[:, :3, 3], Ps[:, :3, 3]) / np.einsum("ti,ti->t", Ps[:, :3, 3], Ps[:, :3, 3])
    g = np.load(f"{V2D}/diag_only_ref_human/{ep}.npz")
    fr = g["frame"]; fr = fr[fr < len(P)]
    W = np.asarray(al[str(idx)]["sim"]); s = np.cbrt(np.linalg.det(W[:3, :3])); Rw = W[:3, :3] / s; C = W[:3, 3]
    with torch.no_grad():
        ov = mhr.mhr_forward(_params(b["pr"], fr, dev)).vertices.cpu().numpy().astype(np.float64) * f[fr, None, None]
        n = len(fr)
        gv = sl(torch.tensor(g["pose"][:n]).float().view(n, 77, 3).to(dev), torch.tensor(g["identity"][:n]).float().to(dev),
                scale_params=torch.tensor(g["scale"][:n]).float().to(dev), transl=torch.tensor(g["transl"][:n]).float().to(dev)).vertices.cpu().numpy().astype(np.float64) @ np.diag([1.0, -1.0, -1.0])
    gv_c = (gv - C) @ Rw / s  # GT human in our camera frame
    # parts (by GT geometry at frame 0): feet = lowest 3% vertices along world up; hands = vertices of the two extremities farthest from the body axis
    from scipy.spatial import cKDTree
    kp = mhr.mhr_forward(_params(b["pr"], fr[:1], dev)).keypoints.cpu().numpy()[0].astype(np.float64)
    o0 = ov[0] / f[fr[0]]
    hands_o = np.linalg.norm(o0[:, None] - kp[None, 21:63], axis=-1).min(1) < 0.06
    hgt = (ov[0] @ Rw.T * s + C) @ np.array([0, -1.0, 0])
    feet_o = hgt < np.percentile(hgt, 3)
    tree = cKDTree(gv_c[0])
    _, jh = tree.query(ov[0][hands_o]); _, jf = tree.query(ov[0][feet_o])
    def ratio(mo, jg):
        return np.linalg.norm(ov[:, mo].mean(1), axis=1) / np.linalg.norm(gv_c[:, np.unique(jg)].mean(1), axis=1)
    rb = np.linalg.norm(ov.mean(1), axis=1) / np.linalg.norm(gv_c.mean(1), axis=1)
    rf, rh = ratio(feet_o, jf), ratio(hands_o, jh)
    ref = np.load(f"{V2D}/ref_poses/{ep}.npz")["pose_xyz_wxyz"].astype(np.float64)[fr]
    glb = trimesh.load(f"{ds}/{meta[idx]['mesh']}", force="scene", process=False).dump(concatenate=True)
    S = np.asarray(trimesh.bounds.oriented_bounds(glb)[0])
    tc = (P[fr] @ S)[:, :3, 3]
    vis = np.linalg.norm(ref[:, 3:], axis=1) > 0.5
    ro = np.where(vis, np.linalg.norm(tc, axis=1) / np.maximum(np.linalg.norm((ref[:, :3] - C) @ Rw / s, axis=1), 1e-6), np.nan)
    q = lambda r: f"{np.nanmedian(r):.3f}[{np.nanpercentile(r,10):.2f},{np.nanpercentile(r,90):.2f}]"
    m = vis & np.isfinite(ro)
    print(f"{idx:2d} {meta[idx]['object'][:10]:10s} | {q(rb)} {q(rf)} {q(rh)} {q(ro)} | {np.corrcoef(ro[m], rh[m])[0,1]:+.2f} {np.corrcoef(ro[m], rb[m])[0,1]:+.2f}", flush=True)
    res[idx] = {"body": rb.tolist(), "feet": rf.tolist(), "hands": rh.tolist(), "obj": ro.tolist(), "frames": fr.tolist()}
json.dump(res, open(os.environ.get("OUT", "/dev/null"), "w"))
