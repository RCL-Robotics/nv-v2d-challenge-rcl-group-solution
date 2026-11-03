#!/usr/bin/env python3
"""Refine the human-fitted camera (cam_from_human.py) with the object track. No GT.
Joint similarity fit  x_world = s R x_cam + t  over
  * human: nearest-neighbour correspondences between our MHR human (camera frame) and the Tier 2 SOMA human (world) on
    NF frames, trimmed; each frame's body points share a total weight of 1;
  * object: our camera-frame object position (Opus trajectory, mesh origin) <-> Tier 2 object position on every visible
    frame, trimmed; weight 1 per frame.
Usage: cam_joint.py <cam_human.json> <out.json>"""
import os
import json, sys, numpy as np, torch
from scipy.spatial import cKDTree
import soma
V = os.environ["V2D_ROOT"]; FLIP = np.diag([1.0, -1.0, -1.0]); NP = 1500; TRIM = 0.8
REL = f"{V}/release/tier3_s1_20261020/reconstructed_data"
eps_meta = {e["episode_index"]: e for e in json.load(open(f"{V}/release/tier3_s1_20261020/episodes.json"))}
def opus(i): return f"{V}/agent_runs/{'CL2' if i <= 10 else 'CL2b' if i <= 21 else 'CL2c'}_ep{i:03d}/outputs/object_poses_world.npz"
def sim3(a, b, w):
    w = w / w.sum(); ma, mb = w @ a, w @ b; A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd((B * w[:, None]).T @ A); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt; s = (S * np.diag(D)).sum() / (w @ (A ** 2).sum(1)); return s, R, mb - s * R @ ma
cams = json.load(open(sys.argv[1])); out = {}
layer = soma.SOMALayer(identity_model_type="mhr", device="cuda")
for i in range(30):
    c = cams[str(i)]; s, R, t = c["s"], np.array(c["R"]), np.array(c["t"]); fr = np.array(c["frames"])
    h = np.load(f"{V}/tier2_human/episode_{i:06d}.npz"); rec = np.load(f"{V}/agent_runs/T1_ds_ep{i:03d}/inputs/reconstruction.npz")
    Wi = np.linalg.inv(np.array(json.load(open(f"{REL}/{eps_meta[i]['sequence_id']}/camera.json"))["T_world_cam"]))
    rng = np.random.default_rng(0)
    with torch.no_grad():
        o = layer(torch.tensor(h["pose"][fr]).float().view(-1, 77, 3).cuda(), torch.tensor(h["identity"][fr]).float().cuda(),
                  scale_params=torch.tensor(h["scale"][fr]).float().cuda(), transl=torch.tensor(h["transl"][fr]).float().cuda())
        Vw = o.vertices.cpu().numpy().astype(np.float64) @ FLIP.T
    hv = rec["human_vertices_world"]
    src = [(hv[f] @ Wi[:3, :3].T + Wi[:3, 3])[rng.choice(hv.shape[1], NP, replace=False)] for f in fr]
    dst = [Vw[j][rng.choice(Vw.shape[1], 4000, replace=False)] for j in range(len(fr))]; trees = [cKDTree(d) for d in dst]
    t2 = np.load(f"{V}/tier2_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float); Tc = Wi[None] @ np.load(opus(i))["T_world_mesh"]
    n = min(len(t2), len(Tc)); vis = np.linalg.norm(t2[:n, 3:], axis=1) > .5
    po, qo = Tc[:n][vis][:, :3, 3], t2[:n][vis][:, :3]
    for _ in range(30):
        P, Q = [], []
        for f in range(len(src)):
            x = s * src[f] @ R.T + t; _, j = trees[f].query(x); P.append(src[f]); Q.append(dst[f][j])
        P, Q = np.concatenate(P), np.concatenate(Q)
        rh = np.linalg.norm(s * P @ R.T + t - Q, axis=1); kh = rh <= np.quantile(rh, TRIM)
        ro = np.linalg.norm(s * po @ R.T + t - qo, axis=1); ko = ro <= np.quantile(ro, TRIM)
        a = np.concatenate([P[kh], po[ko]]); b = np.concatenate([Q[kh], qo[ko]])
        # equal total weight for the two tracks: human 1 per sampled frame, object NF / (#object frames kept) per frame
        w = np.concatenate([np.full(kh.sum(), len(src) / kh.sum()), np.full(ko.sum(), len(src) / max(ko.sum(), 1))])
        s, R, t = sim3(a, b, w)
    out[str(i)] = {"s": s, "R": R.tolist(), "t": t.tolist(), "rms_h_m": float(np.sqrt((rh[kh] ** 2).mean())), "rms_o_m": float(np.sqrt((ro[ko] ** 2).mean())),
                   "frames": fr.tolist()}
    print(i, f"s {s:.3f} rms human {out[str(i)]['rms_h_m']*100:.1f} cm object {out[str(i)]['rms_o_m']*100:.1f} cm", flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
