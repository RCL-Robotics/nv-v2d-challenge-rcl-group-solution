"""Trajectory-level re-estimation of the spray-bottle pose for every frame.

python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz

Pipeline (see REPORT.md):
  1. phase segmentation from the SAM2 object mask (bottom-centre stationarity): static S1 (table),
     carried, static S2 (floor);
  2. static phases: one pose per phase. Rotation = chordal mean of the baseline, made upright w.r.t. the
     floor normal (bottle resting on its flat bottom), yaw about the bottle axis re-estimated by
     render-and-compare on the temporal-median image of the phase (silhouette + texture). Translation =
     baseline median (camera frame);
  3. carried phase: 24 yaw hypotheses (15 deg steps about the bottle axis) per frame, scored by texture
     agreement (blurred textured render vs. image, standardised colour correlation inside the visible mask);
     Viterbi decoding with a rotation-change transition cost and the static poses as boundary
     conditions; frames without a usable mask (occlusion by the body) are bridged by SLERP;
     temporal smoothing of rotation and translation;
  4. depth re-anchoring: per phase, positions are scaled along the viewing rays so that the median camera
     depth equals the baseline median depth of that phase (keeps the calibrated placement).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as Fnn
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.ndimage import gaussian_filter1d, median_filter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoi_lib import (Renderer, FrameData, load_inputs, load_mesh, frame_box, project, rot_mean,
                     rot_about_axis, rot_angle_deg, refine_batch)

torch.manual_seed(0)
np.random.seed(0)
DEV = "cuda"
YAWS = np.arange(0, 360, int(os.environ.get("HOI_YAW_STEP", 10)))
TILT_DEG = float(os.environ.get("HOI_TILT", 5))  # out-of-plane tilt hypotheses (camera x/y axes)

# ---------------------------------------------------------------- parameters (all automatic criteria)
STATIC_TOL_PX = 2.0      # mask bottom-centre within this many px of the phase median => object at rest
MIN_MASK_PX = 400        # frames with fewer visible object pixels carry no usable image evidence
RELIABLE_IOU = 0.45      # baseline silhouette IoU below this => baseline pose of that frame untrusted
LAMBDA_SQ = float(os.environ.get("HOI_LAMBDA_SQ", 5e-4))         # Viterbi transition cost: LAMBDA_SQ * angle_deg^2 / frame_gap (random-walk prior on
                         # rotation: large changes are cheap only across long gaps, e.g. occlusions)
W_TEX = 1.0              # weight of texture correlation in the Viterbi data term
W_IOU = 0.5              # weight of silhouette IoU in the Viterbi data term
SMOOTH_SIGMA_R = 1.5     # temporal Gaussian smoothing (frames) of rotations in the carried phase
SMOOTH_SIGMA_T = 2.0     # same for translations
BLEND_FRAMES = 20        # boundary blending of the carried translation into the static positions (lift-off)
BLEND_FRAMES_END = 30    # same at the end (covers the hand-guided placement onto the floor)


def log(*a):
    print(*a, flush=True)


def blur_t(x, s):
    k = int(3 * s) * 2 + 1
    g = torch.exp(-((torch.arange(k, device=x.device) - k // 2) ** 2) / (2 * s * s))
    g = g / g.sum()
    x = x.permute(0, 3, 1, 2)
    C = x.shape[1]
    x = Fnn.conv2d(Fnn.pad(x, (k // 2, k // 2, 0, 0), mode="replicate"), g.view(1, 1, 1, k).repeat(C, 1, 1, 1), groups=C)
    x = Fnn.conv2d(Fnn.pad(x, (0, 0, k // 2, k // 2), mode="replicate"), g.view(1, 1, k, 1).repeat(C, 1, 1, 1), groups=C)
    return x.permute(0, 2, 3, 1)


def evidence(Rd, K, box, hw, Tcs, img, om, pm, blur_px=1.2):
    """Texture correlation and person-aware IoU for a batch of camera-frame poses (B,4,4)."""
    Tt = torch.tensor(Tcs, device=DEV, dtype=torch.float32)
    rgb, m, _ = Rd.render(Tt, K, box, hw)
    sc = hw[0] / (box[2] - box[0])
    mf = m.float()[..., None]
    rb = blur_t(rgb * mf, blur_px * sc) / blur_t(mf, blur_px * sc).clamp_min(1e-3)
    O = torch.tensor(om, device=DEV)
    P = torch.tensor(pm, device=DEV) & ~O
    I = torch.tensor(img, device=DEV)[None]
    ov = (m & O).float()[..., None]
    cnt = ov.sum((1, 2)).clamp_min(1)

    def z(x):
        mu = (x * ov).sum((1, 2)) / cnt
        xc = (x - mu[:, None, None]) * ov
        return xc / ((xc ** 2).sum((1, 2)) / cnt + 1e-4).sqrt()[:, None, None]

    tex = ((z(rb) * z(I.expand_as(rb))).sum((1, 2, 3)) / cnt[:, 0] / 3).cpu().numpy()
    inter = (m & O).flatten(1).sum(1).float()
    union = (O | (m & ~P)).flatten(1).sum(1).float()
    iou = (inter / union.clamp_min(1)).cpu().numpy()
    return tex, iou


def hypotheses(Rb):
    """Rotation hypotheses around a baseline camera-frame rotation: yaw about the bottle axis (x) optional
    out-of-plane tilts about the camera x / y axes."""
    tilts = [np.eye(3)]
    if TILT_DEG > 0:
        for ax in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]):
            tilts.append(rot_about_axis(ax, np.radians(TILT_DEG)))
    return np.stack([Tl @ Rb @ rot_about_axis([0, 0, 1], np.radians(a)) for Tl in tilts for a in YAWS])


def make_upright(Rm, up):
    """Minimal rotation of Rm so that its z axis (bottle axis, pointing to the trigger) equals `up`."""
    a = Rm[:, 2]
    v = np.cross(a, up)
    s, c = np.linalg.norm(v), a @ up
    if s < 1e-9:
        return Rm
    Rfix = R.from_rotvec(v / s * np.arctan2(s, c)).as_matrix()
    return Rfix @ Rm


def detect_phases(om):
    F = om.shape[0]
    b = np.full((F, 2), np.nan)
    area = np.zeros(F)
    for i in range(F):
        ys, xs = np.nonzero(om[i])
        area[i] = len(xs)
        if len(xs):
            y1 = ys.max()
            sel = ys >= y1 - 4
            b[i] = [xs[sel].mean(), y1]
    # the episode starts and ends with the object at rest (verified in the video: on the desk, then on
    # the floor); the rest positions are the medians of the first / last 60 frames.
    s1 = np.nanmedian(b[:60], 0)
    s2 = np.nanmedian(b[-60:], 0)
    d1 = np.linalg.norm(b - s1, axis=1)
    d2 = np.linalg.norm(b - s2, axis=1)
    e1 = 0
    while e1 + 1 < F and d1[e1 + 1] <= STATIC_TOL_PX:
        e1 += 1
    s2i = F - 1
    while s2i - 1 > e1 and (d2[s2i - 1] <= STATIC_TOL_PX):
        s2i -= 1
    return e1, s2i, area, b


def median_image(fd, frames, box, hw):
    cr = [fd.crop(i, box, hw) for i in frames]
    med = np.median(np.stack([c[0] for c in cr]), 0).astype(np.float32)
    om = np.mean(np.stack([c[1] for c in cr]), 0) > 0.5
    pm = np.mean(np.stack([c[2] for c in cr]), 0) > 0.5
    return med, om, pm


def static_pose(Rd, fd, K, Tc_base, frames, up_cam, refine_yaw):
    """One pose for a static phase. Returns camera-frame 4x4 and diagnostics."""
    Rm = make_upright(rot_mean(Tc_base[frames, :3, :3]), up_cam)
    t = np.median(Tc_base[frames, :3, 3], 0)
    X = np.eye(4)
    X[:3, :3], X[:3, 3] = Rm, t
    diag = {}
    if refine_yaw:
        box = frame_box(K, X, scale=1.4)
        hw = (256, 256)
        med, om, pm = median_image(fd, frames, box, hw)
        # deterministic forward-only search of the yaw about the (upright) bottle axis on the median image
        yaws = np.arange(0, 360, 2)
        H = np.repeat(X[None], len(yaws), 0)
        for k, a in enumerate(yaws):
            H[k, :3, :3] = Rm @ rot_about_axis([0, 0, 1], np.radians(a))
        tex, iou = evidence(Rd, K, box, hw, H, med, om, pm)
        sc = W_TEX * tex + W_IOU * iou
        kb = int(np.argmax(sc))
        far = np.abs((yaws - yaws[kb] + 180) % 360 - 180) > 45
        diag = dict(yaw_change_deg=float((yaws[kb] + 180) % 360 - 180), score_best=float(sc[kb]),
                    score_at_baseline=float(sc[0]), best_score_beyond_45deg=float(sc[far].max()))
        X[:3, :3] = H[kb, :3, :3]
    return X, diag


LTIPS = [42, 46, 50, 54, 58]  # MHR70 left-hand finger tips (the task: object handled with the left hand)
CUE_AGREE_CM = 2.0


def cue_floor(X, Twc, plane, verts):
    """Camera-depth shift (cm) that puts the lowest mesh point of camera-frame pose X on the floor plane."""
    n, d = plane[:3], plane[3]
    sgn = 1 if n[1] < 0 else -1
    best = None
    for dz in np.arange(-0.40, 0.4001, 0.002):
        Y = Twc @ np.r_[np.c_[X[:3, :3], X[:3, 3] * (X[2, 3] + dz) / X[2, 3]], [[0, 0, 0, 1]]]
        low = (((verts @ Y[:3, :3].T + Y[:3, 3]) @ n + d) * sgn).min()
        if best is None or abs(low) < abs(best[1]):
            best = (dz, low)
    return best[0] * 100


def cue_hand(Tc, frames, kp_cam):
    """Median camera-depth difference (cm) between the left finger-tip centroid and the trigger head."""
    out = [kp_cam[i, LTIPS].mean(0)[2] - (Tc[i, :3, :3] @ np.array([0, 0, 0.12]) + Tc[i, :3, 3])[2] for i in frames]
    return float(np.median(out) * 100)


def cue_size(Rd, fd, K, X, frames):
    """Camera-depth shift (cm) maximising the silhouette IoU on the phase-median image (metric size)."""
    box = frame_box(K, X, scale=1.4)
    hw = (256, 256)
    med, om, pm = median_image(fd, frames, box, hw)
    zs = np.arange(-0.30, 0.3001, 0.01)
    H = np.stack([np.r_[np.c_[X[:3, :3], X[:3, 3] * (X[2, 3] + dz) / X[2, 3]], [[0, 0, 0, 1]]] for dz in zs])
    _, iou = evidence(Rd, K, box, hw, H, med, om, pm)
    return float(zs[np.argmax(iou)] * 100)


CUE_SIGMA_CM = {"floor_contact": 3.0, "hand_contact": 4.0, "metric_size": 5.0}  # rough reliabilities; the
# floor plane is a fit with a few cm tolerance, finger tips vs. grasp point a few cm, silhouette size is biased
# towards the camera by mask bleeding.


def decide_shift(cues):
    """Task rule: change the depth only if two independent cues agree within CUE_AGREE_CM and the correction is
    larger than their disagreement. If so, apply the reliability-weighted mean of all cues; else 0."""
    names = list(cues)
    ok = []
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            ca, cb = cues[names[a]], cues[names[b]]
            if abs(ca - cb) <= CUE_AGREE_CM and abs(0.5 * (ca + cb)) > abs(ca - cb):
                ok.append((names[a], names[b]))
    if not ok:
        return 0.0, None
    w = {k: 1.0 / CUE_SIGMA_CM[k] ** 2 for k in names}
    return float(sum(w[k] * cues[k] for k in names) / sum(w.values())), ok


def quat_smooth(Rs, sigma, fixed=None):
    q = R.from_matrix(Rs).as_quat()
    for i in range(1, len(q)):
        if q[i] @ q[i - 1] < 0:
            q[i] = -q[i]
    qs = gaussian_filter1d(q, sigma, axis=0, mode="nearest")
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    out = R.from_quat(qs).as_matrix()
    if fixed is not None:
        out[fixed] = Rs[fixed]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--out", default="/work/outputs/object_poses_world.npz")
    args = ap.parse_args()
    t0 = time.time()
    inp = args.inputs
    cam, K, Twc, rec = load_inputs(inp)
    Tcw = np.linalg.inv(Twc)
    Tw_base = rec["T_world_mesh"].astype(np.float64)
    F = Tw_base.shape[0]
    Tc_base = Tcw[None] @ Tw_base
    plane = np.array(json.load(open(f"{inp}/ground_plane.json"))["plane"], np.float64)
    n = plane[:3] / np.linalg.norm(plane[:3])
    up_w = n if n[1] < 0 else -n  # world is y-down: "up" has negative y
    up_cam = Tcw[:3, :3] @ up_w

    Rd = Renderer(load_mesh(inp))
    fd = FrameData(inp)
    om_all = fd.om
    e1, s2, area, bottom = detect_phases(om_all)
    log(f"phases: S1 static [0,{e1}], carried [{e1 + 1},{s2 - 1}], S2 static [{s2},{F - 1}]")
    S1 = np.arange(0, e1 + 1)
    S2 = np.arange(s2, F)
    C = np.arange(e1 + 1, s2)

    # ------------------------------------------------------------ static phases
    X1, d1 = static_pose(Rd, fd, K, Tc_base, S1, up_cam, refine_yaw=True)
    log("S1 static pose:", d1)
    # S2 (floor, dim, small): the texture is not discriminative there; the yaw of the baseline was checked
    # with the trigger-head silhouette on the median image (side view, lever on the left; REPORT.md), so
    # only the upright constraint is applied.
    X2, d2 = static_pose(Rd, fd, K, Tc_base, S2, up_cam, refine_yaw=False)

    # ------------------------------------------------------------ carried phase: evidence
    hw = (192, 192)
    nC = len(C)
    nH = len(hypotheses(np.eye(3)))
    TEX = np.zeros((nC, nH))
    IOU = np.zeros((nC, nH))
    base_iou = np.zeros(nC)
    reliable = np.zeros(nC, bool)
    for j, i in enumerate(C):
        B = Tc_base[i]
        box = frame_box(K, B, scale=1.3)
        img, om, pm = fd.crop(i, box, hw)
        if area[i] < MIN_MASK_PX:
            continue
        Hr = hypotheses(B[:3, :3])
        H = np.repeat(B[None], len(Hr), 0)
        H[:, :3, :3] = Hr
        tex, iou = evidence(Rd, K, box, hw, H, img, om, pm)
        TEX[j], IOU[j] = tex, iou
        base_iou[j] = iou[0]
        reliable[j] = iou.max() >= RELIABLE_IOU
    log(f"carried: {reliable.sum()}/{nC} frames with usable evidence ({time.time() - t0:.0f}s)")

    # ------------------------------------------------------------ Viterbi over yaw hypotheses
    idx = np.where(reliable)[0]
    Rh = np.stack([hypotheses(Tc_base[C[j], :3, :3]) for j in idx])  # (n, nH, 3, 3)
    unary = -(W_TEX * TEX[idx] + W_IOU * IOU[idx])
    if os.environ.get("HOI_HOLDOUT"):  # development only: hide the evidence of odd frames (validation)
        unary[C[idx] % 2 == 1] = 0.0
    nS = nH
    # boundary: transition from the S1 pose into the first reliable carried frame
    fr = C[idx]
    tcost = lambda ang, gap: LAMBDA_SQ * ang ** 2 / gap
    cost = unary[0] + tcost(rot_angle_deg(X1[:3, :3][None], Rh[0]), fr[0] - e1)
    back = np.zeros((len(idx), nS), int)
    for n_ in range(1, len(idx)):
        A = rot_angle_deg(Rh[n_ - 1][:, None], Rh[n_][None, :])  # (prev, cur)
        tot = cost[:, None] + tcost(A, fr[n_] - fr[n_ - 1])
        back[n_] = np.argmin(tot, 0)
        cost = tot[back[n_], np.arange(nS)] + unary[n_]
    cost = cost + tcost(rot_angle_deg(X2[:3, :3][None], Rh[-1]), s2 - fr[-1])
    path = np.zeros(len(idx), int)
    path[-1] = int(np.argmin(cost))
    for n_ in range(len(idx) - 1, 0, -1):
        path[n_ - 1] = back[n_, path[n_]]
    yaw_sel = YAWS[path % len(YAWS)]
    log("Viterbi yaw offsets (deg) w.r.t. baseline, run-length:",
        [(int(C[idx[a]]), int(yaw_sel[a])) for a in range(len(idx)) if a == 0 or yaw_sel[a] != yaw_sel[a - 1]])

    # ------------------------------------------------------------ assemble rotations
    Rall = np.zeros((F, 3, 3))
    Rall[S1] = X1[:3, :3]
    Rall[S2] = X2[:3, :3]
    key_t = [e1] + [int(C[j]) for j in idx] + [s2]
    key_R = [X1[:3, :3]] + [Rh[a, path[a]] for a in range(len(idx))] + [X2[:3, :3]]
    sl = Slerp(key_t, R.from_matrix(np.stack(key_R)))
    Rall[C] = sl(C).as_matrix()
    fixed = np.zeros(F, bool)
    fixed[S1] = fixed[S2] = True
    Rs = quat_smooth(Rall, SMOOTH_SIGMA_R, fixed=fixed)

    # ------------------------------------------------------------ translations (camera frame)
    tc = Tc_base[:, :3, 3].copy()
    tall = tc.copy()
    tall[S1] = X1[:3, 3]
    tall[S2] = X2[:3, 3]
    # carried: robust smoothing of the baseline (median filter removes isolated jumps, then Gaussian)
    tcar = median_filter(tc, size=(5, 1), mode="nearest")
    tall[C] = tcar[C]
    ts = gaussian_filter1d(tall, SMOOTH_SIGMA_T, axis=0, mode="nearest")
    ts[S1] = X1[:3, 3]
    ts[S2] = X2[:3, 3]

    # ------------------------------------------------------------ depth re-anchoring per phase
    for ph in (S1, C, S2):
        zb = np.median(Tc_base[ph, 2, 3])
        zr = np.median(ts[ph, 2])
        ts[ph] *= zb / zr  # scaling along the viewing rays keeps the image position

    # ------------------------------------------------------------ independent physical depth cues (static)
    kp_w = rec["human_keypoints_world"].astype(np.float64)
    kp_cam = kp_w @ Tcw[:3, :3].T + Tcw[:3, 3]
    cp = rec["contact_prob_right_left"]
    grasp = np.array([i for i in C if cp[i].max() > 0.5 and area[i] >= MIN_MASK_PX])
    Tc_tmp = np.tile(np.eye(4), (F, 1, 1))
    Tc_tmp[:, :3, :3], Tc_tmp[:, :3, 3] = Rs, ts
    verts = Rd.verts_np[::20].astype(np.float64)
    depth_report = {}
    for name, ph, near in (("S1", S1, grasp[grasp <= e1 + 15]), ("S2", S2, grasp[grasp >= s2 - 25])):
        X = Tc_tmp[ph[len(ph) // 2]]
        cues = {"metric_size": cue_size(Rd, fd, K, X, ph)}
        if len(near):
            cues["hand_contact"] = cue_hand(Tc_tmp, near, kp_cam)
        if name == "S2":  # resting on the floor (task: placed on the ground)
            cues["floor_contact"] = cue_floor(X, Twc, plane, verts)
        shift, pair = decide_shift(cues)
        depth_report[name] = dict(cues_cm=cues, applied_cm=shift, agreeing=pair)
        log(f"depth cues {name}: {cues} -> applied {shift:+.1f} cm {pair}")
        if shift != 0.0:
            ts[ph] *= (ts[ph, 2:3] + shift / 100) / ts[ph, 2:3]

    # continuity with the rest poses: the object leaves / reaches the static position, so the carried-phase
    # translation offset at the phase boundaries is blended out linearly over BLEND_FRAMES frames.
    nC_ = len(C)
    off_s = ts[e1] - ts[C[0]]
    off_e = ts[s2] - ts[C[-1]]
    j = np.arange(nC_)
    ws = np.clip(1 - j / BLEND_FRAMES, 0, 1)[:, None]
    we = np.clip(1 - (nC_ - 1 - j) / BLEND_FRAMES_END, 0, 1)[:, None]
    ts[C] += ws * off_s + we * off_e

    Tc_out = np.tile(np.eye(4), (F, 1, 1))
    Tc_out[:, :3, :3] = Rs
    Tc_out[:, :3, 3] = ts
    Tw_out = Twc[None] @ Tc_out
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, T_world_mesh=Tw_out.astype(np.float64),
             phases=np.array([e1, s2]), viterbi_frames=C[idx], viterbi_yaw=yaw_sel,
             tex=TEX, iou=IOU, depth_report=json.dumps(depth_report))
    log(f"saved {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
