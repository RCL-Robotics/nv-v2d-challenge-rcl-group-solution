#!/usr/bin/env python
"""Trajectory-level 6-DoF pose recovery for one monocular HOI video (white desk episode).

Method (see REPORT.md):
  1. Phase segmentation from the video: per-frame image change inside the (dilated) object mask, with
     person pixels excluded.  Long runs without change = static phases (the desk stands untouched / is held
     without moving / lies on the floor).  Everything else = moving frames.
  2. Joint render-and-compare optimisation of the whole trajectory (nvdiffrast silhouettes vs. SAM2 object
     masks, person pixels ignored).  Each static phase shares ONE pose (evidence pooled over all its frames);
     every moving frame has its own pose.  Terms:
        - silhouette L2 (anti-aliased render vs. mask) on non-person pixels,
        - temporal smoothness (2nd differences of the object centre and of the rotation matrix),
        - non-penetration of the released floor plane (the desk is tipped over and must not sink into the floor);
          this term only moves the object along its viewing ray (its gradient reaches the log-depth only),
        - depth anchoring: per phase group, the median camera-frame depth of the object centre equals the
          baseline's median (the baseline placement is calibrated against the human body and floor).
     Pose parameters: rotation increment about the object centre, image-plane offset and log-depth of the centre.
     Initialised from the baseline (T_world_mesh of reconstruction.npz); static phases from the rotation-mean /
     median of the baseline over the phase.
  3. Exact per-group median-depth re-anchoring along the viewing rays (camera frame), map back to world.
Deterministic (fixed seeds, fixed iteration counts, deterministic data order).
"""
import argparse, json, os, sys, time
import numpy as np
import cv2, h5py, trimesh, torch
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot

# ----------------------------------------------------------------------------------------------- constants
HALF = 0.5                 # optimisation resolution (768x576)
MOTION_LAG = 3             # frames between compared images for the change detector
MOTION_THR = 2.0           # mean abs grey-level change (0..255) inside the object region => moving.
                           # Noise level when nothing moves is 0.6-1.5 (checked on the whole video).
MIN_STATIC = 20            # a static phase must last at least this many frames
STATIC_SUBSAMPLE = 2       # use every 2nd frame of a static phase for its silhouette term
SIG_T = 0.002              # smoothness scale for 2nd difference of the centre (m / frame^2)
SIG_R = 0.006              # smoothness scale for 2nd difference of rotation matrices
LAM_SMOOTH = 0.01          # weight of the smoothness term (chosen with held-out-frame IoU, see REPORT.md)
FLOOR_TOL = 0.0            # lowest desk point may not go below the released floor plane (m); the true frame-0
                           # human soles are 1.4 cm above that plane, so 0 is a lenient bound.
SIG_FLOOR = 0.003
SIG_CONTACT = 0.005       # ablation only (--floor_mode contact): equal lowest-point height within a phase group
FLOOR_START = 0.3         # floor term is ramped in after the silhouettes have converged (30%..55% of iters)
SIG_ANCHOR = 0.002         # depth-anchor softness (m) during optimisation (exact re-anchoring afterwards)
ITERS = 300
LR = 3e-3
CHUNK = 24


def so3_exp(w):
    th = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-9)
    k = w / th
    Kx = torch.zeros(w.shape[:-1] + (3, 3), device=w.device)
    Kx[..., 0, 1] = -k[..., 2]; Kx[..., 0, 2] = k[..., 1]; Kx[..., 1, 0] = k[..., 2]
    Kx[..., 1, 2] = -k[..., 0]; Kx[..., 2, 0] = -k[..., 1]; Kx[..., 2, 1] = k[..., 0]
    th = th[..., None]
    I = torch.eye(3, device=w.device).expand_as(Kx)
    return I + torch.sin(th) * Kx + (1 - torch.cos(th)) * (Kx @ Kx)


class Silhouette:
    def __init__(self, mesh, K, H, W):
        self.v = torch.tensor(mesh.vertices, dtype=torch.float32, device='cuda')
        self.f = torch.tensor(mesh.faces, dtype=torch.int32, device='cuda')
        self.glctx = dr.RasterizeCudaContext()
        self.H, self.W = H, W
        near, far = 0.05, 20.0
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        self.P = torch.tensor([[2 * fx / W, 0, (2 * cx + 1) / W - 1, 0], [0, 2 * fy / H, (2 * cy + 1) / H - 1, 0],
                               [0, 0, (far + near) / (far - near), -2 * far * near / (far - near)], [0, 0, 1, 0]],
                              dtype=torch.float32, device='cuda')

    def __call__(self, R, t):
        vc = torch.einsum('bij,nj->bni', R, self.v) + t[:, None]
        vh = torch.cat([vc, torch.ones_like(vc[..., :1])], -1)
        clip = torch.einsum('ij,bnj->bni', self.P, vh).contiguous()
        rast, _ = dr.rasterize(self.glctx, clip, self.f, (self.H, self.W))
        m = (rast[..., 3:4] > 0).float()
        return dr.antialias(m, rast, clip, self.f)[..., 0]


def read_video_gray_half(path):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(cv2.resize(fr, (768, 576), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
    return np.array(out)


def load_masks_half(path, F):
    f = h5py.File(path, 'r')
    OM, PM = f['object'], f['person']
    om = np.zeros((F, 576, 768), np.float32); pm = np.zeros((F, 576, 768), bool)
    for i in range(F):
        om[i] = cv2.resize(OM[i].astype(np.float32), (768, 576), interpolation=cv2.INTER_AREA)
        pm[i] = cv2.resize(PM[i].astype(np.uint8), (768, 576), interpolation=cv2.INTER_NEAREST) > 0
    return om, pm


def segment_phases(gray, om, pm):
    """Static runs: no image change inside the object region (person pixels excluded)."""
    F = len(gray)
    k9 = np.ones((9, 9), np.uint8); k15 = np.ones((15, 15), np.uint8)
    omb = om > 0.5
    d = np.zeros(F)
    g = gray.astype(np.float32)
    for i in range(MOTION_LAG, F):
        reg = (cv2.dilate(omb[i].astype(np.uint8), k9) > 0) | (cv2.dilate(omb[i - MOTION_LAG].astype(np.uint8), k9) > 0)
        p = cv2.dilate((pm[i] | pm[i - MOTION_LAG]).astype(np.uint8), k15) > 0
        v = reg & ~p
        d[i] = np.abs(g[i] - g[i - MOTION_LAG])[v].mean() if v.sum() > 200 else 99.0
    moving = np.zeros(F, bool)
    for i in range(MOTION_LAG, F):
        if d[i] > MOTION_THR:
            moving[i - MOTION_LAG:i + 1] = True
    runs, s = [], None
    for i in range(F + 1):
        mv = True if i == F else moving[i]
        if not mv and s is None:
            s = i
        if mv and s is not None:
            if i - s >= MIN_STATIC:
                runs.append((s, i - 1))
            s = None
    return runs, d


def mean_pose(Ts):
    T = np.eye(4)
    T[:3, :3] = Rot.from_matrix(Ts[:, :3, :3]).mean().as_matrix()
    T[:3, 3] = np.median(Ts[:, :3, 3], 0)
    return T


def anchor_groups(F, runs):
    """Phase groups for depth anchoring: everything up to the start of the last moving block before the final
    static phase is split at the starts/ends of the long moving block (the main manipulation)."""
    # boundaries: gaps between static runs; the longest gap is the main manipulation phase
    gaps = [(runs[k][1] + 1, runs[k + 1][0] - 1) for k in range(len(runs) - 1)]
    if not gaps:
        return [(0, F - 1)]
    a, b = max(gaps, key=lambda g: g[1] - g[0])
    groups = [(0, a - 1), (a, b), (b + 1, F - 1)]
    return [g for g in groups if g[1] >= g[0]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='/work/inputs')
    ap.add_argument('--out', default='/work/outputs/object_poses_world.npz')
    ap.add_argument('--lam', type=float, default=LAM_SMOOTH)
    ap.add_argument('--iters', type=int, default=ITERS)
    ap.add_argument('--no_floor', action='store_true')
    ap.add_argument('--floor_mode', default='nonpen', choices=['contact', 'nonpen'])
    ap.add_argument('--floor_full_grad', action='store_true')
    ap.add_argument('--sig_contact', type=float, default=SIG_CONTACT)
    ap.add_argument('--no_static', action='store_true', help='ablation: no shared static poses')
    ap.add_argument('--heldout', default='', help="ablation: 'odd' or 'blocks' -> those frames get no data term")
    ap.add_argument('--debug_dir', default='')
    ap.add_argument('--free_groups', default='', help='ablation: comma list of anchor-group indices left free')
    args = ap.parse_args()
    t_start = time.time()
    np.random.seed(0); torch.manual_seed(0)
    inp = args.inputs

    cam = json.load(open(os.path.join(inp, 'camera.json')))
    K = np.array(cam['K']); Twc = np.array(cam['T_world_cam']); Tcw = np.linalg.inv(Twc)
    plane = np.array(json.load(open(os.path.join(inp, 'ground_plane.json')))['plane'])
    mesh = trimesh.load(os.path.join(inp, 'object_mesh.glb'), force='scene', process=False).to_geometry()
    rec = np.load(os.path.join(inp, 'reconstruction.npz'))
    Tb_w = rec['T_world_mesh'].astype(np.float64)
    F = len(Tb_w)
    Tb = Tcw[None] @ Tb_w                       # baseline, camera frame
    c_np = mesh.bounds.mean(0)                  # object centre (mesh frame)
    pb = Tb[:, :3, :3] @ c_np + Tb[:, :3, 3]    # baseline centre in camera frame

    # ------------------------------------------------------------------ data
    gray = read_video_gray_half(os.path.join(inp, 'video.mp4'))
    assert len(gray) == F, (len(gray), F)
    om, pm = load_masks_half(os.path.join(inp, 'masks.h5'), F)
    runs, dchange = segment_phases(gray, om, pm)
    groups = anchor_groups(F, runs)
    print('static runs:', runs, '\nanchor groups:', groups, flush=True)
    if args.no_static:
        runs = []

    # slots: one per static run, one per moving frame
    slot = -np.ones(F, int); init = []
    for (a, b) in runs:
        slot[a:b + 1] = len(init); init.append(mean_pose(Tb[a:b + 1]))
    for i in range(F):
        if slot[i] < 0:
            slot[i] = len(init); init.append(Tb[i])
    init = np.array(init); S = len(init)

    # data frames
    use = np.zeros(F, bool)
    in_static = np.zeros(F, bool)
    for (a, b) in runs:
        in_static[a:b + 1] = True
        use[a:b + 1:STATIC_SUBSAMPLE] = True
    use[~in_static] = True
    if args.heldout == 'odd':
        use &= (np.arange(F) % 2 == 0)
    elif args.heldout == 'blocks':
        use &= ~((np.arange(F) // 5) % 4 == 2)       # 5 of every 20 frames held out
    dfr = np.where(use)[0]

    # validity: dilated person mask is "don't care"
    kd = np.ones((7, 7), np.uint8)
    valid = np.array([~(cv2.dilate(pm[i].astype(np.uint8), kd) > 0) for i in range(F)])

    Kh = K.copy(); Kh[:2] *= HALF; Kh[0, 2] = (K[0, 2] + 0.5) * HALF - 0.5; Kh[1, 2] = (K[1, 2] + 0.5) * HALF - 0.5
    sil = Silhouette(mesh, Kh, 576, 768)
    dev = 'cuda'
    M_all = torch.tensor(om, device=dev)            # (F,H,W) ~ 1.4 GB fp32
    V_all = torch.tensor(valid, device=dev)
    area = (M_all * V_all).sum((1, 2)).clamp_min(50.0)

    T0 = torch.tensor(init, dtype=torch.float32, device=dev)
    c = torch.tensor(c_np, dtype=torch.float32, device=dev)
    # pose parameters per slot: rotation increment w (about the object centre), image-plane offset of the centre
    # (normalised coordinates) duv, and log-depth increment dlz.  Separating the depth along the viewing ray
    # makes the poorly observed direction explicit (and lets the floor term act on it alone).
    w = torch.zeros(S, 3, device=dev, requires_grad=True)
    duv = torch.zeros(S, 2, device=dev, requires_grad=True)
    dlz = torch.zeros(S, 1, device=dev, requires_grad=True)
    p0 = T0[:, :3, :3] @ c + T0[:, :3, 3]
    z0 = p0[:, 2:3]; xy0 = p0[:, :2] / z0
    grp = np.zeros(F, int)
    for g, (a, b) in enumerate(groups):
        grp[a:b + 1] = g
    grp_t = torch.tensor(grp, device=dev)
    h0 = torch.zeros(len(groups), device=dev, requires_grad=True)
    opt = torch.optim.Adam([{'params': [w, h0], 'lr': LR}, {'params': [duv, dlz], 'lr': LR / 3}], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters, eta_min=LR * 0.05)
    slot_t = torch.tensor(slot, device=dev)
    # floor in camera frame: n_c . x + d_c = 0
    n_w, d_w = plane[:3], plane[3]
    n_c = Twc[:3, :3].T @ n_w; d_c = n_w @ Twc[:3, 3] + d_w
    n_ct = torch.tensor(n_c, dtype=torch.float32, device=dev); d_ct = float(d_c)
    vsub = torch.tensor(mesh.vertices[::25], dtype=torch.float32, device=dev)
    zb_med = [float(np.median(pb[a:b + 1, 2])) for (a, b) in groups]

    def slot_centres():
        z = z0 * torch.exp(dlz)
        return torch.cat([(xy0 + duv) * z, z], 1)

    def slot_poses():
        Rn = so3_exp(w) @ T0[:, :3, :3]
        tn = slot_centres() - Rn @ c                      # rotation about the object centre
        return Rn, tn

    for it in range(args.iters):
        opt.zero_grad()
        Ld = 0.0
        for k in range(0, len(dfr), CHUNK):
            ii = torch.tensor(dfr[k:k + CHUNK], device=dev)
            Rn, tn = slot_poses()
            s_ = slot_t[ii]
            A = sil(Rn[s_], tn[s_])
            l = ((((A - M_all[ii]) ** 2) * V_all[ii]).sum((1, 2)) / area[ii]).sum()
            l.backward(); Ld += l.item()
        Rn, tn = slot_poses()
        Rf, tf = Rn[slot_t], tn[slot_t]
        p = Rf @ c + tf
        d2p = p[2:] - 2 * p[1:-1] + p[:-2]
        d2R = Rf[2:] - 2 * Rf[1:-1] + Rf[:-2]
        Ls = args.lam * ((d2p ** 2).sum() / SIG_T ** 2 + (d2R ** 2).sum() / SIG_R ** 2 / 2)
        La = 0.0
        for g, (a, b) in enumerate(groups):
            if str(g) in args.free_groups.split(','):
                continue
            La = La + (torch.median(p[a:b + 1, 2]) - zb_med[g]) ** 2 / SIG_ANCHOR ** 2
        Lf = torch.zeros((), device=dev)
        if not args.no_floor:
            if args.floor_full_grad:
                hv = torch.einsum('sij,nj->sni', Rn, vsub) + tn[:, None]
            else:
                # floor evidence only moves the object along its viewing ray (rotation / image position come
                # from the images): gradient flows through the log-depth only
                pc = slot_centres().detach()
                sc = torch.exp(dlz - dlz.detach())
                hv = torch.einsum('sij,nj->sni', Rn.detach(), vsub - c) + (pc * sc)[:, None]
            h = (hv @ n_ct + d_ct).min(1).values            # lowest point height above plane, per slot
            ramp = min(1.0, max(0.0, (it - FLOOR_START * args.iters) / (0.25 * args.iters)))
            if it == int(FLOOR_START * args.iters):
                with torch.no_grad():
                    hf0 = h[slot_t]
                    for g, (a, b) in enumerate(groups):
                        h0[g] = torch.median(hf0[a:b + 1])
            if args.floor_mode == 'nonpen':
                Lf = ramp * (torch.relu(FLOOR_TOL - h) ** 2).sum() / SIG_FLOOR ** 2
            else:
                # contact: the desk is supported by the floor in every frame (it is tipped over, never carried),
                # so the height of its lowest point above the floor plane is the same in all frames of a phase
                # group (h0_g free per group -> only the relative depth profile is constrained, not the placement).
                hf = h[slot_t]
                Lf = ramp * ((hf - h0[grp_t]) ** 2).sum() / args.sig_contact ** 2
        (Ls + La + Lf).backward()
        opt.step(); sched.step()
        if it % 50 == 0 or it == args.iters - 1:
            print(f'it {it:4d} data {Ld:8.3f} smooth {Ls.item():8.3f} anchor {float(La):7.3f} floor {Lf.item():7.3f}'
                  f'  ({time.time() - t_start:.0f}s)', flush=True)

    with torch.no_grad():
        Rn, tn = slot_poses()
        Rn = Rn.double().cpu().numpy(); tn = tn.double().cpu().numpy()
    T = np.tile(np.eye(4), (F, 1, 1))
    T[:, :3, :3] = Rn[slot]; T[:, :3, 3] = tn[slot]
    # re-orthonormalise
    U, _, Vt = np.linalg.svd(T[:, :3, :3]); T[:, :3, :3] = U @ Vt

    # exact depth re-anchoring per group: scale centre along its viewing ray
    p = T[:, :3, :3] @ c_np + T[:, :3, 3]
    for g, (a, b) in enumerate(groups):
        if str(g) in args.free_groups.split(','):
            continue
        s = zb_med[g] / np.median(p[a:b + 1, 2])
        print(f'group {a}-{b}: residual depth scale {s:.5f}')
        q = p[a:b + 1] * s
        T[a:b + 1, :3, 3] = q - T[a:b + 1, :3, :3] @ c_np
    Tw = Twc[None] @ T
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, T_world_mesh=Tw.astype(np.float64))
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        np.savez(os.path.join(args.debug_dir, 'debug.npz'), runs=np.array(runs), groups=np.array(groups),
                 dchange=dchange, use=use, slot=slot)
    print(f'done in {time.time() - t_start:.0f}s -> {args.out}; peak GPU mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB')


if __name__ == '__main__':
    main()
