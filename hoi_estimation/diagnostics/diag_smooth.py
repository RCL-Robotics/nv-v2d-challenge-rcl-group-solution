"""DIAGNOSTIC: how much of JC1's object error is per-frame noise vs per-episode bias, and what temporal filters buy."""
import os, json, sys
import numpy as np, torch, trimesh
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hoi_extraction", "scripts"))
from eval_object_pose import metrics, quat_wxyz_to_R
V2D = os.environ["V2D_ROOT"]; ds = f"{V2D}/dataset/track_2/tier_1_multiview_caption"
meta = {d["episode_index"]: d for d in map(json.loads, open(f"{ds}/meta/episodes_metadata.jsonl"))}
al = json.load(open(f"{V2D}/logs/JC1_joint_camera_align.json"))
print("ep obj        | raw  | med15 med45 | dep15 dep45 | static | bias-free(oracle) | pos_cm raw, bias_cm, resid_cm, speed_cm/f")
for idx in range(30):
    ep = f"episode_{idx:06d}"
    b = torch.load(f"{V2D}/experiments/JC1_joint_camera/{ep}/coconet.pth", map_location="cpu", weights_only=False)
    P = np.asarray(b["pr"]["pose_abs"], np.float64); F = len(P)
    glb = trimesh.load(f"{ds}/{meta[idx]['mesh']}", force="scene", process=False).dump(concatenate=True)
    S = np.asarray(trimesh.bounds.oriented_bounds(glb)[0]); verts = np.asarray(glb.sample(1000))
    W = np.asarray(al[str(idx)]["sim"]); Rw, C = W[:3, :3], W[:3, 3]
    T = P @ S; tc = T[:, :3, 3]; Rc = T[:, :3, :3]
    ref = np.load(f"{V2D}/ref_poses/{ep}.npz")["pose_xyz_wxyz"].astype(np.float64)
    n = min(F, len(ref)); ok = (np.linalg.norm(ref[:n, 3:], axis=1) > 0.5)
    Rr, tr = quat_wxyz_to_R(ref[:n][ok][:, 3:]), ref[:n][ok][:, :3]
    def ev(tcam, Rcam=Rc):
        tw = tcam @ Rw.T + C; Rwo = Rw[None] @ Rcam
        return metrics(Rwo[:n][ok], tw[:n][ok], Rr, tr, verts)
    out = [ev(tc)]
    for w in (15, 45):
        out.append(ev(median_filter(tc, size=(w, 1), mode="nearest")))
    dep = np.linalg.norm(tc, axis=1); u = tc / dep[:, None]
    for w in (15, 45):
        out.append(ev(u * median_filter(dep, size=w, mode="nearest")[:, None]))
    # static: frames with low contact prob -> world-static segments get segment median (in camera frame, static cam)
    cp = 1 / (1 + np.exp(-np.asarray(b["pr"]["contact_logits"], np.float64))).max(1)
    st = median_filter((cp < 0.5).astype(int), size=9) > 0
    ts = tc.copy(); i = 0
    while i < F:
        if st[i]:
            j = i
            while j < F and st[j]: j += 1
            if j - i >= 10: ts[i:j] = np.median(tc[i:j], axis=0)
            i = j
        else: i += 1
    out.append(ev(ts))
    tw = tc @ Rw.T + C; bias = np.median(tw[:n][ok] - tr, axis=0)
    tw_ob = tw - bias
    out.append(ev((tw_ob - C) @ Rw))
    speed = np.median(np.linalg.norm(np.diff(tw, axis=0), axis=1)) * 100
    resid = np.median(np.linalg.norm(tw[:n][ok] - tr - bias, axis=1)) * 100
    print(f"{idx:2d} {meta[idx]['object'][:10]:10s} | " + " ".join(f"{o['AUC']:.3f}" for o in out) +
          f" | {out[0]['pos_cm']:.1f} {np.linalg.norm(bias)*100:.1f} {resid:.1f} {speed:.2f} static%{st.mean()*100:.0f}", flush=True)
