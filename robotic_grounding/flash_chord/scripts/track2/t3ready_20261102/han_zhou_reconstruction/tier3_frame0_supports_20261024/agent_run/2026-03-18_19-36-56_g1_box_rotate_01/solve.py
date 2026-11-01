"""Recover the 6-DoF trajectory of the object in inputs/video.mp4 (episode 5, big_black_box).

Method (see REPORT.md):
  1. phase segmentation from the SAM2 object masks (static = object silhouette unchanged over +-10 frames),
  2. joint render-and-compare optimisation of ALL frames at once (nvdiffrast, crop-based):
       silhouette term (object mask, person-occluded pixels ignored) + internal box-crease-edge chamfer term
       (Canny edges of the frame), static phases share ONE pose, acceleration penalty everywhere and a
       motion-adaptive velocity penalty (strong when the masks say the object does not move),
     initialised from the baseline (T_world_mesh),
  3. depth re-anchoring: camera-frame positions are scaled along the viewing rays with a smooth per-frame ratio
     (constant within each static phase) so that each phase's depth matches the baseline's (median z),
  4. independent physical depth cues are computed and reported; a shift is applied only if >=2 agree (default 0).

Run: python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
"""
import argparse
import json
import os
import sys
import time

import cv2
import h5py
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoi_lib import Renderer, SilMesh, cam_from_world, load_inputs, load_mesh, rot_angle  # noqa: E402
from traj_opt import TrajOpt, make_crops, make_edge_maps  # noqa: E402

# ---------------- constants (chosen on this episode, see REPORT.md) ----------------
STATIC_IOU_THR = 0.98   # mask IoU (person pixels excluded) with frames i-10 and i+10 => object static at i
STATIC_MIN_LEN = 20     # minimum static run length (frames)
CROP_S = 192            # silhouette crop resolution
EDGE_E = 320            # edge-map crop resolution
LAM_ACC = 10.0          # acceleration penalty (selected with held-out-frame silhouette IoU: 10 > 50 > 250)
LAM_VEL = 100.0         # velocity penalty, scaled by the per-frame "static-ness" from the masks
EDGE_MU = 0.01          # weight of the crease-edge chamfer term (held-out-frame edge error improved 2.15->1.91 px)
ITERS1, ITERS2 = 150, 100
RATIO_SMOOTH = 3e6      # smoothness of the depth re-anchoring ratio (2nd differences of log ratio; ~40-frame scale)


def log(*a):
    print(*a, flush=True)


def read_gray_frames(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    return np.stack(out)


def static_runs(om_h, pm_h, k=10, thr=STATIC_IOU_THR, min_len=STATIC_MIN_LEN):
    """Static phases: runs of frames whose object mask (excluding person pixels) matches frames i-k and i+k."""
    F = len(om_h)

    def iou(a, b):
        v = ~pm_h[a] & ~pm_h[b]
        return (om_h[a] & om_h[b] & v).sum() / max(((om_h[a] | om_h[b]) & v).sum(), 1)

    s = np.array([min(iou(i, max(i - k, 0)), iou(i, min(i + k, F - 1))) for i in range(F)])
    st = s >= thr
    runs, i = [], 0
    while i < F:
        if st[i]:
            j = i
            while j + 1 < F and st[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs, s


def build_nodes(Tc, runs):
    F = len(Tc)
    node_of = np.zeros(F, int)
    init, n, i = [], 0, 0
    starts = {a: b for a, b in runs}
    while i < F:
        if i in starts:
            a, b = i, starts[i]
            Rm = Rot.from_matrix(Tc[a:b + 1, :3, :3]).mean().as_matrix()
            tt = Tc[a:b + 1, :3, 3]
            xy = np.median(tt[:, :2] / tt[:, 2:], 0)
            z = np.median(tt[:, 2])
            Ti = np.eye(4)
            Ti[:3, :3] = Rm
            Ti[:3, 3] = [xy[0] * z, xy[1] * z, z]
            init.append(Ti)
            node_of[a:b + 1] = n
            i = b + 1
        else:
            init.append(Tc[i])
            node_of[i] = n
            i += 1
        n += 1
    return np.stack(init), node_of


def phases_from_runs(runs, F):
    ph, prev = [], 0
    for a, b in runs:
        if a > prev:
            ph.append(("moving", prev, a - 1))
        ph.append(("static", a, b))
        prev = b + 1
    if prev < F:
        ph.append(("moving", prev, F - 1))
    return ph


def reanchor_depth(Tc_new, Tc_base, phases, node_of):
    """Scale camera-frame translations along the viewing ray (image position unchanged) by a smooth ratio rho(t),
    constant inside static phases, fitted to log(z_base/z_new) (~40-frame smoothing scale). The trajectory thus
    keeps the baseline's low-frequency depth (its calibrated placement, per phase) and our relative motion."""
    F = len(Tc_new)
    r = np.log(Tc_base[:, 2, 3] / Tc_new[:, 2, 3])
    # unknowns: one per node
    N = node_of.max() + 1
    A = np.zeros((F, N))
    A[np.arange(F), node_of] = 1
    D = np.zeros((N - 2, N))
    for k in range(N - 2):
        D[k, k:k + 3] = [1, -2, 1]
    x = np.linalg.solve(A.T @ A + RATIO_SMOOTH * D.T @ D, A.T @ r)
    rho = np.exp(x[node_of])
    out = Tc_new.copy()
    out[:, :3, 3] *= rho[:, None]
    return out, rho


def depth_cues(Tc, d, mesh, phases, rec):
    """Suggested depth shift (cm, along the viewing ray, + = farther) per independent physical cue."""
    T_wc = d["T_wc"]
    n, dd = d["plane"][:3], d["plane"][3]
    V = np.asarray(mesh.vertices)[::10]
    gp = json.load(open(os.path.join(d["inp"], "ground_plane.json")))
    floor_level = gp["foot_plane_dist_stats"]["median"]
    cues = {}
    # (1) floor: lowest object point vs floor level (object always touches the floor in this episode)
    Tw = T_wc[None] @ Tc
    h = (np.einsum("fij,nj->fni", Tw[:, :3, :3], V) + Tw[:, None, :3, 3]) @ n + dd
    hmin = h.min(1)
    Tp = Tc.copy()
    Tp[:, :3, 3] *= 1.01
    Twp = T_wc[None] @ Tp
    hp = ((np.einsum("fij,nj->fni", Twp[:, :3, :3], V) + Twp[:, None, :3, 3]) @ n + dd).min(1)
    dh_ds = (hp - hmin) / (0.01 * np.linalg.norm(Tc[:, :3, 3], axis=1))  # height change per metre along ray
    s_floor = (floor_level - hmin) / dh_ds
    # (2) hand contact: shift that brings the reconstructed hand keypoints onto the object surface
    kp = rec["human_keypoints_world"].astype(np.float64)
    cp = rec["contact_prob_right_left"]
    Tcw = np.linalg.inv(T_wc)
    from scipy.spatial import cKDTree
    tree = cKDTree(np.asarray(mesh.vertices)[::5])
    s_hand = np.full(len(Tc), np.nan)
    for i in range(len(Tc)):
        hands = []
        if cp[i, 0] > 0.5:
            hands.append(kp[i, 21:42])
        if cp[i, 1] > 0.5:
            hands.append(kp[i, 42:63])
        if not hands:
            continue
        H = np.concatenate(hands) @ Tcw[:3, :3].T + Tcw[:3, 3]
        ray = Tc[i, :3, 3] / np.linalg.norm(Tc[i, :3, 3])
        best, bs = 1e9, 0.0
        for s in np.arange(-0.15, 0.1501, 0.01):
            P = (H - Tc[i, :3, 3] - s * ray) @ Tc[i, :3, :3]
            dist, _ = tree.query(P)
            c = np.sort(dist)[: max(3, len(dist) // 4)].mean()  # closest quarter of hand points
            if c < best:
                best, bs = c, s
        s_hand[i] = bs
    for kind, a, b in phases:
        key = f"{kind} {a}-{b}"
        cues[key] = dict(floor_cm=float(np.median(s_floor[a:b + 1]) * 100),
                         hand_cm=float(np.nanmedian(s_hand[a:b + 1]) * 100) if np.isfinite(s_hand[a:b + 1]).any() else None)
    allc = dict(floor_cm=float(np.median(s_floor) * 100), hand_cm=float(np.nanmedian(s_hand) * 100))
    return cues, allc


def decide_shift(allc, sil_cm):
    """Apply a global shift only if >=2 independent cues agree within 2 cm and the shift exceeds their spread."""
    vals = [v for v in [allc["floor_cm"], allc["hand_cm"], sil_cm] if v is not None and np.isfinite(v)]
    best = 0.0
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            spread = abs(vals[i] - vals[j])
            m = (vals[i] + vals[j]) / 2
            if spread <= 2.0 and abs(m) > spread and abs(m) > abs(best):
                best = m
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--out", default="/work/outputs/object_poses_world.npz")
    ap.add_argument("--holdout", action="store_true", help="validation: drop image evidence of every 5th moving frame")
    args = ap.parse_args()
    t_start = time.time()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True

    d = load_inputs(args.inputs)
    d["inp"] = args.inputs
    rec = d["rec"]
    mesh = load_mesh(args.inputs)
    R = Renderer(mesh)
    SM = SilMesh(mesh, glctx=R.glctx)
    T_base = rec["T_world_mesh"].astype(np.float64)
    Tc_base = cam_from_world(d["T_wc"], T_base)
    F = len(Tc_base)

    # ---- masks / frames
    f5 = h5py.File(os.path.join(args.inputs, "masks.h5"), "r")
    OM = f5["object"][:]
    PM = f5["person"][:]
    gray = read_gray_frames(os.path.join(args.inputs, "video.mp4"))
    assert len(gray) == F
    log(f"loaded {F} frames  {time.time() - t_start:.1f}s")

    # ---- 1. phases
    runs, stat_iou = static_runs(OM[:, ::2, ::2], PM[:, ::2, ::2])
    phases = phases_from_runs(runs, F)
    log("phases:", phases)

    # ---- 2. joint optimisation
    init, node_of = build_nodes(Tc_base, runs)
    N = len(init)
    crops, om, vm = make_crops(Tc_base, lambda i: OM[i], lambda i: PM[i], d["K"], d["H"], d["W"], R, S=CROP_S)
    dts, pms = make_edge_maps(crops, lambda i: gray[i], lambda i: PM[i], E=EDGE_E)
    log(f"crops/edges ready {time.time() - t_start:.1f}s")

    # velocity weights between consecutive nodes: "static-ness" of the frame pair from the mask IoU signal
    first_frame = np.array([np.where(node_of == k)[0][0] for k in range(N)])
    last_frame = np.array([np.where(node_of == k)[0][-1] for k in range(N)])
    s_pair = np.minimum(stat_iou[last_frame[:-1]], stat_iou[first_frame[1:]])
    wv = np.clip((s_pair - 0.95) / 0.03, 0, 1) ** 2
    wv_t = torch.tensor(wv, dtype=torch.float32, device="cuda")

    def smooth(Rn, t):
        aR = Rn[:-2] - 2 * Rn[1:-1] + Rn[2:]
        at = t[:-2] - 2 * t[1:-1] + t[2:]
        vR = ((Rn[1:] - Rn[:-1]) ** 2).sum((1, 2))
        vt = ((t[1:] - t[:-1]) ** 2).sum(1)
        return LAM_ACC * ((aR ** 2).sum() + (at ** 2).sum()) + LAM_VEL * (wv_t * (vR + vt)).sum()

    cnt = np.bincount(node_of)[node_of]
    fw = np.ones(F) / cnt ** 0.5  # a static phase counts like sqrt(n) frames
    ho = np.zeros(F, bool)
    if args.holdout:
        mv = np.where(cnt == 1)[0]
        ho[mv[2::5]] = True
        fw[ho] = 0
    opt = TrajOpt(SM, d["K"], init, node_of, crops, om, vm, CROP_S)
    opt.run(ITERS1, fw, smooth, log=log)
    log(f"stage 1 done {time.time() - t_start:.1f}s")

    Sobb, _ = trimesh.bounds.oriented_bounds(mesh)
    Robb = np.linalg.inv(Sobb)[:3, :3]
    v = SM.v.cpu().numpy()
    fc = SM.f.cpu().numpy().astype(int)
    fnm = np.cross(v[fc[:, 1]] - v[fc[:, 0]], v[fc[:, 2]] - v[fc[:, 0]])
    fnm /= np.linalg.norm(fnm, axis=1, keepdims=True) + 1e-12
    fn_obb = torch.tensor(np.argmax(fnm @ np.concatenate([Robb, -Robb], 1), 1) + 1, device="cuda")
    edge = dict(dts=torch.tensor(dts, device="cuda"), pms=torch.tensor(pms, device="cuda"), E=EDGE_E,
                fn_obb=fn_obb, mu=EDGE_MU)
    opt.run(ITERS2, fw, smooth, lr=1e-3, log=log, edge=edge, edge_w=fw)
    log(f"stage 2 done {time.time() - t_start:.1f}s")
    Tc_opt = opt.frame_T().detach().cpu().numpy().astype(np.float64)
    iou_opt = opt.ious()

    # ---- 3. depth re-anchoring to the baseline, per phase
    Tc_new, ratio = reanchor_depth(Tc_opt, Tc_base, phases, node_of)

    # ---- 4. physical depth cues (reported; shift applied only on agreement of >=2 cues)
    cues, allc = depth_cues(Tc_new, d, mesh, phases, rec)
    sil_cm = float(np.median((Tc_opt[:, 2, 3] - Tc_new[:, 2, 3]) / Tc_new[:, 2, 3] * np.linalg.norm(Tc_new[:, :3, 3], axis=1)) * 100)
    shift = decide_shift(allc, sil_cm)
    if shift != 0.0:
        ray = Tc_new[:, :3, 3] / np.linalg.norm(Tc_new[:, :3, 3], axis=1, keepdims=True)
        Tc_new[:, :3, 3] += ray * shift / 100.0
    log("depth cues (cm, + = farther):", allc, "silhouette:", round(sil_cm, 2), "applied:", shift)

    T_out = d["T_wc"][None] @ Tc_new
    T_out[:, 3] = [0, 0, 0, 1]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, T_world_mesh=T_out.astype(np.float64))

    diag = dict(phases=phases, runs=runs, cues_per_phase=cues, cues_all=allc, silhouette_scale_cue_cm=sil_cm,
                applied_shift_cm=shift, runtime_s=time.time() - t_start,
                iou_crop_opt_mean=float(iou_opt.mean()), holdout=bool(args.holdout))
    np.savez(os.path.splitext(args.out)[0] + "_diag.npz", Tc_opt=Tc_opt, ratio=ratio, stat_iou=stat_iou,
             holdout_mask=ho, iou_crop_opt=iou_opt, node_of=node_of)
    json.dump(diag, open(os.path.splitext(args.out)[0] + "_diag.json", "w"), indent=1)
    log(f"saved {args.out}  total {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
