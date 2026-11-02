#!/usr/bin/env python3
"""Estimate the static exo camera of each episode in the Tier 2 world frame from the Tier 2 human (noisy SOMA-X, world)
and our reconstructed human (MHR, camera frame; inputs/reconstruction.npz human_vertices_world mapped back with the s1
camera). No GT. Multi-frame trimmed ICP with a similarity transform  x_world = s R x_cam + t  (s absorbs our depth
scale); correspondences are nearest neighbours within the same frame (the two meshes have different topology).
Initialisation: similarity fit of per-frame body centroids, plus the same rotated 180 deg about the body-up axis; the
lower trimmed residual wins. Runs in the v2d_cari4d image (GPU for the SOMA decoder).
Usage: cam_from_human.py <out.json> [eps]"""
import json, sys, numpy as np, torch
from scipy.spatial import cKDTree
import soma
V = "/mnt/nas/Group/hanzhou/v2d"; FLIP = np.diag([1.0, -1.0, -1.0]); NF = 40; NP = 1500; TRIM = 0.8
REL = f"{V}/release/tier3_s1_20261020/reconstructed_data"
eps_meta = {e["episode_index"]: e for e in json.load(open(f"{V}/release/tier3_s1_20261020/episodes.json"))}

def sim3(a, b, w=None):
    w = np.ones(len(a)) if w is None else w; w = w / w.sum()
    ma, mb = w @ a, w @ b; A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd((B * w[:, None]).T @ A); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt; s = (S * np.diag(D)).sum() / (w @ (A ** 2).sum(1)); return s, R, mb - s * R @ ma

def icp(src, dst_trees, dst, s, R, t, it=40):
    for _ in range(it):
        P, Q = [], []
        for f in range(len(src)):
            x = s * src[f] @ R.T + t; d, j = dst_trees[f].query(x); P.append(src[f]); Q.append(dst[f][j])
        P, Q = np.concatenate(P), np.concatenate(Q); r = np.linalg.norm(s * P @ R.T + t - Q, axis=1)
        k = r <= np.quantile(r, TRIM); s, R, t = sim3(P[k], Q[k])
    r = np.linalg.norm(s * P @ R.T + t - Q, axis=1); return s, R, t, float(np.sqrt((r[r <= np.quantile(r, TRIM)] ** 2).mean()))

def main():
    out = sys.argv[1]; eps = list(map(int, sys.argv[2].split(","))) if len(sys.argv) > 2 else range(30)
    layer = soma.SOMALayer(identity_model_type="mhr", device="cuda"); res = {}
    for i in eps:
        h = np.load(f"{V}/tier2_human/episode_{i:06d}.npz"); F = len(h["pose"])
        rec = np.load(f"{V}/agent_runs/T1_ds_ep{i:03d}/inputs/reconstruction.npz")
        W = np.array(json.load(open(f"{REL}/{eps_meta[i]['sequence_id']}/camera.json"))["T_world_cam"]); Wi = np.linalg.inv(W)
        hv = rec["human_vertices_world"]; F = min(F, len(hv))
        fr = np.linspace(0, F - 1, NF).astype(int); rng = np.random.default_rng(0)
        with torch.no_grad():
            o = layer(torch.tensor(h["pose"][fr]).float().view(-1, 77, 3).cuda(), torch.tensor(h["identity"][fr]).float().cuda(),
                      scale_params=torch.tensor(h["scale"][fr]).float().cuda(), transl=torch.tensor(h["transl"][fr]).float().cuda())
            Vw = o.vertices.cpu().numpy().astype(np.float64) @ FLIP.T
        src = [(hv[f] @ Wi[:3, :3].T + Wi[:3, 3])[rng.choice(hv.shape[1], NP, replace=False)] for f in fr]
        ok = [np.isfinite(x).all() and np.abs(x).max() < 50 for x in src]
        fr = fr[ok]; src = [x for x, k in zip(src, ok) if k]; dst = [Vw[j][rng.choice(Vw.shape[1], 4000, replace=False)] for j in np.where(ok)[0]]
        trees = [cKDTree(d) for d in dst]
        s0, R0, t0 = sim3(np.array([x.mean(0) for x in src]), np.array([d.mean(0) for d in dst]))
        best = None
        for flip in (False, True):
            R = R0
            if flip:  # 180 deg about the world body-up axis (mean direction feet->head of the Tier 2 bodies)
                up = np.mean([d[d[:, 1].argmin()] - d.mean(0) for d in dst], 0); up /= np.linalg.norm(up)
                K = np.array([[0, -up[2], up[1]], [up[2], 0, -up[0]], [-up[1], up[0], 0]]); R = (np.eye(3) + 2 * K @ K) @ R0
            c_src = np.mean([x.mean(0) for x in src], 0); c_dst = np.mean([d.mean(0) for d in dst], 0)
            r = icp(src, trees, dst, s0, R, c_dst - s0 * R @ c_src)
            if best is None or r[3] < best[3]: best = r
        s, R, t, rms = best
        # per-frame residual spread as a reliability signal
        res[str(i)] = {"s": s, "R": R.tolist(), "t": t.tolist(), "rms_m": rms, "frames": fr.tolist(), "centroid_init_scale": s0}
        print(i, eps_meta[i]["object"], f"s {s:.3f} rms {rms*100:.1f} cm (centroid-init scale {s0:.2f})", flush=True)
        json.dump(res, open(out, "w"), indent=1)

if __name__ == "__main__":
    main()
