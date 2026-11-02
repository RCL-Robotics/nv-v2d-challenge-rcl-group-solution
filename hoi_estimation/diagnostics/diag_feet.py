"""DIAGNOSTIC ONLY: lowest-vertex height above the released plane, ours (after FF2 factor) vs GT, per frame."""
import json, sys, numpy as np, torch
sys.path.insert(0, "/workspace/v2d_cari4d/lib/cari4d"); sys.path.insert(0, "/home/hanzhou/Projects/v2d/hoi_extraction/scripts")
from eval_official import bundle_path
from lib_mhr.mhr_layer import MHRLayer
from tools.render_mhr_wild_inference import _params
import soma
V2D = "/mnt/nas/Group/hanzhou/v2d"; ds = f"{V2D}/dataset/track_2/tier_1_multiview_caption"
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{ds}/meta/episodes_metadata.jsonl"))}
al = json.load(open(f"{V2D}/logs/pnp_forE1.json")); ff = json.load(open(f"{V2D}/experiments/FF2_feet_ref/feet_floor.json"))
dev = torch.device("cuda"); mhr = MHRLayer.from_mhr_assets(mhr_assets_root=f"{V2D}/weights/sam3d_body", device="cuda"); sl = soma.SOMALayer(identity_model_type="mhr", device="cuda")
for idx in map(int, sys.argv[1].split(",")):
    ep = f"episode_{idx:06d}"
    b = torch.load(bundle_path("FF2_feet_ref", ep), map_location="cpu", weights_only=False); bs = torch.load(bundle_path(ff[str(idx)]["src"], ep), map_location="cpu", weights_only=False)
    P = np.asarray(b["pr"]["pose_abs"], np.float64); Ps = np.asarray(bs["pr"]["pose_abs"], np.float64)
    f = np.einsum("ti,ti->t", P[:, :3, 3], Ps[:, :3, 3]) / np.einsum("ti,ti->t", Ps[:, :3, 3], Ps[:, :3, 3])
    g = np.load(f"{V2D}/diag_only_ref_human/{ep}.npz"); fr = g["frame"]; fr = fr[fr < len(P)]; n = len(fr)
    W = np.asarray(al[str(idx)]["sim"]); s = np.cbrt(np.linalg.det(W[:3, :3])); Rw = W[:3, :3] / s; C = W[:3, 3]
    gp = np.asarray(json.load(open(f"{ds}/{meta[idx]['ground_plane']}"))["plane"]); nn_, d = gp[:3] / np.linalg.norm(gp[:3]), gp[3] / np.linalg.norm(gp[:3])
    if nn_ @ C + d < 0: nn_, d = -nn_, -d
    with torch.no_grad():
        ov = mhr.mhr_forward(_params(b["pr"], fr, dev)).vertices.cpu().numpy().astype(np.float64) * f[fr, None, None]
        gv = sl(torch.tensor(g["pose"][:n]).float().view(n, 77, 3).to(dev), torch.tensor(g["identity"][:n]).float().to(dev), scale_params=torch.tensor(g["scale"][:n]).float().to(dev), transl=torch.tensor(g["transl"][:n]).float().to(dev)).vertices.cpu().numpy().astype(np.float64) @ np.diag([1.0, -1.0, -1.0])
    ow = s * ov @ Rw.T + C
    ho = (ow @ nn_ + d).min(1); hg = (gv @ nn_ + d).min(1)
    # depth of the lowest point (ours, camera) and GT lowest point in our camera
    zo = np.linalg.norm(ov[np.arange(n), (ow @ nn_ + d).argmin(1)], axis=1)
    gc = (gv - C) @ Rw / s; zg = np.linalg.norm(gc[np.arange(n), (gv @ nn_ + d).argmin(1)], axis=1)
    print(f"ep {idx}: camera height {nn_ @ C + d:.3f}  f_t median {np.median(f):.3f}")
    for i in range(0, n, max(1, n // 14)):
        print(f"  f{fr[i]:4d} h_ours {ho[i]*100:5.1f}cm h_gt {hg[i]*100:5.1f}cm  z_lowest ours {zo[i]:.2f} gt {zg[i]:.2f}  f_t {f[fr[i]]:.3f}")
