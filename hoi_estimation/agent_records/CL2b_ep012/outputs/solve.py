#!/usr/bin/env python
"""Trajectory-level re-estimation of the 6-DoF object pose for one HOI video.

Pipeline (all estimates are recomputed from the inputs; nothing per-frame is embedded):
  1. Phase segmentation from the video: static-A (object untouched / not yet moving), moving M, static-B.
     Criterion: mean absolute image difference inside the object's region (person pixels excluded)
     relative to the first / last frame.
  2. Per-frame render-and-compare: nvdiffrast silhouette of the exact mesh vs. the SAM2 object mask
     (pixels covered by the person are ignored), 6-DoF Adam optimisation started from the baseline.
     The silhouette gives very precise *relative* poses but its absolute depth (size) is biased, so
     depth is re-anchored to the baseline afterwards.
  3. Symmetry check: the mesh is ~symmetric under a 180 deg turn about its thin (y) axis.  Both the
     baseline branch and the flipped branch are fitted per frame and scored on a held-out cue (dark-pixel
     map of the image, which sees the openings of the pallet).  The flipped branch is adopted only if it
     wins clearly in the static phases (where evidence can be pooled).
  4. Static phases: one pose per phase, jointly fitted to all frames of the phase.
     Moving phase: per-frame fits smoothed in time (Gaussian, quaternions / positions).
  5. Depth re-anchoring (camera frame, along the viewing ray of the object centre):
       static phases: median camera z == baseline median z of the phase;
       moving phase : default = baseline median z of the phase; replaced by the floor-contact /
                      size-constancy estimate only if those two independent cues agree within 2 cm
                      and the correction is larger than their disagreement.
  6. Diagnostics (validation table) written next to the output as JSON.

Run:  python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
"""
import argparse, json, os, time
import numpy as np, cv2, h5py, torch, trimesh
import nvdiffrast.torch as dr
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot

dev = 'cuda'
CROP = 320            # crop size (full-res pixels) for rendering; the object is <= ~190 px in this video
FIT_ITERS = 150       # Adam iterations per frame batch
BATCH = 50            # frames per GPU batch (keeps GPU memory << 9 GB)
SMOOTH_SIGMA = 2.0    # temporal Gaussian sigma (frames) for the moving phase
RAMP = 30             # frames over which the depth scale blends between phases


# ----------------------------------------------------------------------------- geometry helpers
def rotdist(R1, R2):
    R = np.einsum('...ji,...jk->...ik', R1, R2)
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def rot_mean(Rs):
    U, _, Vt = np.linalg.svd(np.sum(Rs, 0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    return R


def so3_exp_b(w):
    th = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-9)
    k = w / th
    z = torch.zeros_like(k[:, 0])
    Kx = torch.stack([z, -k[:, 2], k[:, 1], k[:, 2], z, -k[:, 0], -k[:, 1], k[:, 0], z], -1).view(-1, 3, 3)
    th = th[..., None]
    return torch.eye(3, device=w.device)[None] + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


def symmetry_transform(mesh, seed=0):
    """Rigid transform S (mesh coords) mapping the mesh onto itself after a 180 deg turn about its
    thin axis (y), refined by ICP (the scan is only approximately symmetric)."""
    import open3d as o3d
    P = trimesh.sample.sample_surface(mesh, 100000, seed=seed)[0]
    c = (mesh.bounds[0] + mesh.bounds[1]) / 2
    F = np.eye(4); F[:3, :3] = np.diag([-1., 1, -1]); C = np.eye(4); C[:3, 3] = c
    S0 = C @ F @ np.linalg.inv(C)
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P @ S0[:3, :3].T + S0[:3, 3]))
    est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    reg = o3d.pipelines.registration.registration_icp(tgt, src, 0.02, np.eye(4), est)
    reg = o3d.pipelines.registration.registration_icp(tgt, src, 0.01, reg.transformation, est)
    return reg.transformation @ S0


# ----------------------------------------------------------------------------- renderer / fitter
class Renderer:
    def __init__(self, mesh, W, H):
        self.ctx = dr.RasterizeCudaContext()
        self.v = torch.tensor(mesh.vertices, dtype=torch.float32, device=dev)
        self.vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        self.f = torch.tensor(mesh.faces, dtype=torch.int32, device=dev)
        self.W, self.H = W, H

    def proj(self, K, n=0.05, fa=50.):
        W, H = self.W, self.H
        P = np.zeros((4, 4))
        P[0, 0] = 2 * K[0, 0] / W; P[0, 2] = 2 * (K[0, 2] + 0.5) / W - 1
        P[1, 1] = 2 * K[1, 1] / H; P[1, 2] = 2 * (K[1, 2] + 0.5) / H - 1
        P[2, 2] = (fa + n) / (fa - n); P[2, 3] = -2 * fa * n / (fa - n); P[3, 2] = 1
        return P

    def soft_mask(self, Pb, T):
        vc = torch.einsum('bij,nj->bni', T, self.vh)
        clip = torch.einsum('bij,bnj->bni', Pb, vc).contiguous()
        rast, _ = dr.rasterize(self.ctx, clip, self.f, (self.H, self.W))
        m = (rast[..., 3:] > 0).float()
        return dr.antialias(m, rast, clip, self.f)[..., 0]


def compose(T0, w, dt):
    T = torch.zeros_like(T0)
    T[:, :3, :3] = so3_exp_b(w) @ T0[:, :3, :3]
    T[:, :3, 3] = T0[:, :3, 3] + dt
    T[:, 3, 3] = 1
    return T


def fit_batch(R, T0, Ks, target, weight, iters, static=False, lr_r=0.01, lr_t=0.004):
    """Silhouette fit. static=True: one shared pose offset for the whole batch (static phase)."""
    T0 = torch.tensor(T0, dtype=torch.float32, device=dev)
    B = T0.shape[0]
    Pb = torch.tensor(np.stack([R.proj(k) for k in Ks]), dtype=torch.float32, device=dev)
    tgt = torch.as_tensor(target, dtype=torch.float32, device=dev)
    wt = torch.as_tensor(weight, dtype=torch.float32, device=dev)
    n = 1 if static else B
    w = torch.zeros(n, 3, device=dev, requires_grad=True)
    dt = torch.zeros(n, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{'params': [w], 'lr': lr_r}, {'params': [dt], 'lr': lr_t}])
    for _ in range(iters):
        m = R.soft_mask(Pb, compose(T0, w.expand(B, 3), dt.expand(B, 3)))
        loss = (((m - tgt) ** 2 * wt).sum((1, 2)) / wt.sum((1, 2)).clamp_min(1)).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        T = compose(T0, w.expand(B, 3), dt.expand(B, 3))
        m = R.soft_mask(Pb, T)
        per = ((m - tgt) ** 2 * wt).sum((1, 2)) / wt.sum((1, 2)).clamp_min(1)
    return T.cpu().numpy().astype(np.float64), per.cpu().numpy(), m.cpu().numpy()


def crop_K(K, cx, cy):
    x0, y0 = int(cx) - CROP // 2, int(cy) - CROP // 2
    Kc = K.copy(); Kc[0, 2] -= x0; Kc[1, 2] -= y0
    return Kc, x0, y0


def crop_img(a, x0, y0):
    H, W = a.shape[:2]
    out = np.zeros((CROP, CROP) + a.shape[2:], a.dtype)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + CROP, W), min(y0 + CROP, H)
    out[ya - y0:yb - y0, xa - x0:xb - x0] = a[ya:yb, xa:xb]
    return out


def dark_prob(img_bgr, thr=38., s=5.):
    """Probability that a pixel belongs to the (black) pallet; sees its openings, unlike the SAM mask."""
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return 1 / (1 + np.exp(-(thr - g) / s))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='/work/inputs')
    ap.add_argument('--out', default='/work/outputs/object_poses_world.npz')
    args = ap.parse_args()
    t_start = time.time()
    np.random.seed(0); torch.manual_seed(0)
    I = args.inputs

    mesh = trimesh.load(f'{I}/object_mesh.glb', force='scene', process=False).to_geometry()
    cam = json.load(open(f'{I}/camera.json'))
    K = np.array(cam['K']); Twc = np.array(cam['T_world_cam']); Tcw = np.linalg.inv(Twc)
    plane = np.array(json.load(open(f'{I}/ground_plane.json'))['plane'])
    rec = np.load(f'{I}/reconstruction.npz')
    Tw_base = rec['T_world_mesh'].astype(np.float64)
    Tc_base = Tcw[None] @ Tw_base
    F = len(Tw_base)
    ctr = (mesh.bounds[0] + mesh.bounds[1]) / 2          # bbox centre of the mesh
    centre = lambda T: T[:, :3, :3] @ ctr + T[:, :3, 3]   # object centre in the pose's frame

    with h5py.File(f'{I}/masks.h5') as f:
        OM = f['object'][:]; PM = f['person'][:]

    # ---- video: full-res crops around the object + half-res grey frames for phase detection
    cap = cv2.VideoCapture(f'{I}/video.mp4')
    Ks, x0s, y0s, IMC, GREY = [], [], [], [], []
    for i in range(F):
        ok, fr = cap.read()
        assert ok, f'video ended at frame {i}'
        ys, xs = np.nonzero(OM[i])
        if len(xs) == 0:  # no mask: centre crop on the baseline projection
            p = centre(Tc_base[i:i + 1])[0]; xs, ys = [K[0, 0] * p[0] / p[2] + K[0, 2]], [K[1, 1] * p[1] / p[2] + K[1, 2]]
        kc, x0, y0 = crop_K(K, np.mean(xs), np.mean(ys))
        Ks.append(kc); x0s.append(x0); y0s.append(y0)
        IMC.append(crop_img(fr, x0, y0))
        GREY.append(cv2.cvtColor(cv2.resize(fr, (fr.shape[1] // 2, fr.shape[0] // 2)), cv2.COLOR_BGR2GRAY).astype(np.float32))
    GREY = np.stack(GREY)
    print('[%.0fs] video read' % (time.time() - t_start))

    # ---- 1. phases
    om_h = OM[:, ::2, ::2]; pm_h = PM[:, ::2, ::2]

    # motion energy: mean |I(i+2) - I(i-2)| inside the (dilated) object masks, person pixels excluded
    e = np.zeros(F)
    for i in range(F):
        a_, b_ = max(i - 2, 0), min(i + 2, F - 1)
        reg = cv2.dilate((om_h[a_] | om_h[b_]).astype(np.uint8), np.ones((9, 9))) > 0
        v = reg & ~(cv2.dilate((pm_h[a_] | pm_h[b_]).astype(np.uint8), np.ones((7, 7))) > 0)
        e[i] = np.abs(GREY[b_] - GREY[a_])[v].mean() if v.sum() > 50 else np.inf
    # static-A ends 2 frames before the first run of 3 frames whose energy exceeds 1.6x the energy level
    # of the first 60 frames (the object is untouched there); static-B starts 3 frames after the last
    # such run (threshold from the last 60 frames).  Verified by eye: lift starts ~196, put down ~582.
    thrA = 1.6 * np.median(e[:60]); thrB = 1.6 * np.median(e[-60:])
    i0 = next(i for i in range(F - 2) if (e[i:i + 3] > thrA).all())
    a_end = i0 + 1
    i1 = next(i for i in range(F - 1, 2, -1) if (e[i - 2:i + 1] > thrB).all())
    b_start = i1 + 3
    phases = {'A': (0, a_end), 'M': (a_end + 1, b_start - 1), 'B': (b_start, F - 1)}
    print('phases', phases, 'thrA %.2f thrB %.2f' % (thrA, thrB))
    sA = np.arange(0, a_end + 1); sM = np.arange(a_end + 1, b_start); sB = np.arange(b_start, F)

    # ---- 2. per-frame silhouette fit
    tg = np.stack([crop_img(OM[i], x0s[i], y0s[i]) for i in range(F)]).astype(np.float32)
    wt = np.stack([1 - crop_img(cv2.dilate(PM[i].astype(np.uint8), np.ones((5, 5))), x0s[i], y0s[i])
                   for i in range(F)]).astype(np.float32)
    DP = np.stack([dark_prob(im) for im in IMC]).astype(np.float32)
    roi = np.stack([cv2.dilate(t.astype(np.uint8), np.ones((21, 21))) for t in tg]).astype(np.float32) * wt
    R = Renderer(mesh, CROP, CROP)

    def fit_all(T0, iters):
        Ts, Ls = [], []
        for s in range(0, F, BATCH):
            sl = slice(s, s + BATCH)
            T, per, _ = fit_batch(R, T0[sl], Ks[sl], tg[sl], wt[sl], iters)
            Ts.append(T); Ls.append(per)
        return np.concatenate(Ts), np.concatenate(Ls)

    def eval_all(T):
        """held-out dark-map loss, SAM silhouette loss and IoU (person pixels excluded)"""
        Ld, Ls, IoU = [], [], []
        for s in range(0, F, BATCH):
            sl = slice(s, s + BATCH)
            _, pd, mm = fit_batch(R, T[sl], Ks[sl], DP[sl], roi[sl], 0)
            _, ps, _ = fit_batch(R, T[sl], Ks[sl], tg[sl], wt[sl], 0)
            a = (mm > 0.5) & (wt[sl] > 0); b = (tg[sl] > 0) & (wt[sl] > 0)
            Ld.append(pd); Ls.append(ps); IoU.append((a & b).sum((1, 2)) / np.maximum((a | b).sum((1, 2)), 1))
        return np.concatenate(Ld), np.concatenate(Ls), np.concatenate(IoU)

    T_fit, L_fit = fit_all(Tc_base, FIT_ITERS)
    print('[%.0fs] per-frame fit' % (time.time() - t_start))

    # ---- 3. symmetry branch (180 deg about the thin axis), scored on the held-out dark map
    S = symmetry_transform(mesh)
    T_sym, L_sym = fit_all(T_fit @ S, FIT_ITERS)
    Ld_fit = eval_all(T_fit)[0]; Ld_sym = eval_all(T_sym)[0]
    sym_votes = {}
    for k, sl in (('A', sA), ('M', sM), ('B', sB)):
        sym_votes[k] = dict(frac_base_better_dark=float((Ld_fit[sl] < Ld_sym[sl]).mean()),
                            frac_base_better_sam=float((L_fit[sl] < L_sym[sl]).mean()),
                            mean_dark_base=float(Ld_fit[sl].mean()), mean_dark_sym=float(Ld_sym[sl].mean()))
    # A rigid trajectory cannot switch branch without a visible 180 deg spin (none in this video:
    # the per-frame rotation is smooth), so the choice is global.  Flip only if BOTH static phases
    # prefer the flipped branch on the held-out cue in >70 % of their frames.
    flip = sym_votes['A']['frac_base_better_dark'] < 0.3 and sym_votes['B']['frac_base_better_dark'] < 0.3
    if flip:
        T_fit, L_fit = T_sym, L_sym
    print('[%.0fs] symmetry' % (time.time() - t_start))
    print('symmetry votes', json.dumps(sym_votes), 'flip', flip)

    # ---- 4a. static phases: one pose per phase, jointly fitted to all its frames
    T_rel = T_fit.copy()
    static_pose = {}
    for k, sl in (('A', sA), ('B', sB)):
        T0 = np.eye(4); T0[:3, :3] = rot_mean(T_fit[sl, :3, :3])
        T0[:3, 3] = np.median(centre(T_fit[sl]), 0) - T0[:3, :3] @ ctr
        sub = sl[::2]
        T_s, _, _ = fit_batch(R, np.repeat(T0[None], len(sub), 0), [Ks[i] for i in sub], tg[sub], wt[sub],
                              FIT_ITERS, static=True, lr_r=0.005, lr_t=0.002)
        static_pose[k] = T_s[0]
        T_rel[sl] = T_s[0]

    # ---- 4b. moving phase: temporal smoothing (static poses act as boundary context)
    q = Rot.from_matrix(T_rel[:, :3, :3]).as_quat()
    for i in range(1, F):
        if q[i] @ q[i - 1] < 0:
            q[i] *= -1
    p = centre(T_rel)
    qs = gaussian_filter1d(q, SMOOTH_SIGMA, axis=0, mode='nearest')
    ps = gaussian_filter1d(p, SMOOTH_SIGMA, axis=0, mode='nearest')
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    T_sm = T_rel.copy()
    Rs = Rot.from_quat(qs[sM]).as_matrix()
    T_sm[sM, :3, :3] = Rs
    T_sm[sM, :3, 3] = ps[sM] - Rs @ ctr

    print('[%.0fs] static + smoothing' % (time.time() - t_start))
    # ---- 5. depth re-anchoring along the viewing ray of the object centre
    zb = centre(Tc_base)[:, 2]; zr = centre(T_sm)[:, 2]
    s_phase = {k: float(np.median(zb[sl]) / np.median(zr[sl])) for k, sl in (('A', sA), ('M', sM), ('B', sB))}

    def apply_scale(T, s):
        T = T.copy(); c = centre(T)
        T[:, :3, 3] = c * np.asarray(s)[:, None] - T[:, :3, :3] @ ctr
        return T

    HULL = mesh.convex_hull.vertices

    def floor_min(Tc):  # signed distance of the lowest mesh point above the released floor plane
        V = HULL
        Tw = Twc[None] @ Tc
        Vw = np.einsum('bij,nj->bni', Tw[:, :3, :3], V) + Tw[:, None, :3, 3]
        return (Vw @ plane[:3] + plane[3]).min(1)

    # cue 1 (floor contact): the pallet is never lifted clear of the floor (it is tipped over an edge),
    # so in every frame its lowest point should sit at the floor height h0 observed in the static
    # phases (anchored to the baseline there).  Solve per frame for the scale giving that height.
    s_static = np.ones(F); s_static[sA] = s_phase['A']; s_static[sB] = s_phase['B']
    h0 = float(np.median(floor_min(apply_scale(T_sm, s_static)[np.r_[sA, sB]])))
    grid = np.linspace(0.9, 1.2, 301)
    hm = np.stack([floor_min(apply_scale(T_sm[sM], np.full(len(sM), g))) for g in grid], 1)  # (M, G)
    s_floor = np.array([np.interp(h0, row[::-1], grid[::-1]) for row in hm])
    # cue 2 (silhouette size constancy): the silhouette-to-metric scale calibrated by the baseline in
    # the two static phases, interpolated linearly in time across the moving phase
    ramp = np.interp(sM, [sM[0] - 1, sM[-1] + 1], [s_phase['A'], s_phase['B']])
    zM = np.median(zb[sM])
    shift_floor = (np.median(s_floor) - s_phase['M']) * np.median(zr[sM]) * 100
    shift_size = (np.median(ramp) - s_phase['M']) * np.median(zr[sM]) * 100
    # cue 3 (reported only, not independent of the baseline): contact with the reconstructed hands
    kp = rec['human_keypoints_world']; cp = rec['contact_prob_right_left']
    from scipy.spatial import cKDTree
    tree = cKDTree(mesh.vertices)
    s_hand = []
    for i in sM[::5]:
        hands = [kp[i, 21:42]] * int(cp[i, 0] > .5) + [kp[i, 42:63]] * int(cp[i, 1] > .5)
        if not hands: continue
        best = []
        for g in grid[::3]:
            Tm = np.linalg.inv(Twc @ apply_scale(T_sm[i:i + 1], [g])[0])  # world -> mesh
            best.append(sum(tree.query(h @ Tm[:3, :3].T + Tm[:3, 3])[0].min() for h in hands))
        s_hand.append(grid[::3][int(np.argmin(best))])
    shift_hand = (np.median(s_hand) - s_phase['M']) * np.median(zr[sM]) * 100 if s_hand else float('nan')
    disagreement = abs(shift_floor - shift_size)
    correction = 0.5 * (shift_floor + shift_size)
    # Deviate from the baseline depth of the moving phase only if the two independent cues agree
    # within 2 cm and the correction exceeds their disagreement (task rule); otherwise 0.
    apply_corr = disagreement <= 2.0 and abs(correction) > disagreement
    z_target = np.median(zb[sM]) + (correction / 100 if apply_corr else 0.0)
    # moving-phase scale: size-constancy ramp between the static phases, offset so that the phase
    # median camera depth equals z_target; the offset is tapered to 0 over RAMP frames at both phase
    # ends so the trajectory stays continuous with the static phases
    taper = np.clip(np.minimum(sM - sM[0] + 1, sM[-1] - sM + 1) / RAMP, 0, 1)
    s_t = s_static.copy()
    off = 0.0
    for _ in range(50):  # fixed-point iteration: median depth -> z_target
        off += (z_target - np.median(zr[sM] * (ramp + off * taper))) / np.median(zr[sM])
    s_t[sM] = ramp + off * taper
    T_final_c = apply_scale(T_sm, s_t)
    depth_cues = dict(shift_floor_cm=float(shift_floor), shift_size_constancy_cm=float(shift_size),
                      shift_hand_contact_cm_reported_only=float(shift_hand), disagreement_cm=float(disagreement),
                      applied=bool(apply_corr),
                      applied_median_shift_cm=float((np.median(zr[sM] * s_t[sM]) - zM) * 100),
                      s_phase=s_phase, h0_floor_m=h0, s_floor_median=float(np.median(s_floor)))
    print('[%.0fs] depth' % (time.time() - t_start))
    print('depth cues', json.dumps(depth_cues))

    T_world = Twc[None] @ T_final_c
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, T_world_mesh=T_world.astype(np.float64))

    # ---- 6. validation table
    Ld_b, Ls_b, iou_b = eval_all(Tc_base)
    Ld_r, Ls_r, iou_r = eval_all(T_final_c)
    fl_b, fl_r = floor_min(Tc_base), floor_min(T_final_c)
    zf = centre(T_final_c)[:, 2]

    def accel(T):
        c = centre(Twc[None] @ T)
        a = np.linalg.norm(c[2:] - 2 * c[1:-1] + c[:-2], axis=1) * 100
        dR = np.r_[rotdist(T[1:, :3, :3], T[:-1, :3, :3])]
        return a, np.abs(np.diff(dR))
    ab, rb = accel(Tc_base); ar, rr = accel(T_final_c)
    table = {}
    for k, sl in (('A', sA), ('M', sM), ('B', sB), ('all', np.arange(F))):
        s2 = sl[(sl >= 1) & (sl <= F - 2)] - 1
        table[k] = dict(frames=[int(sl[0]), int(sl[-1])],
                        iou_base=float(iou_b[sl].mean()), iou_ours=float(iou_r[sl].mean()),
                        darkloss_base=float(Ld_b[sl].mean()), darkloss_ours=float(Ld_r[sl].mean()),
                        floor_pen_base_cm=float(np.percentile(fl_b[sl], 5) * 100), floor_pen_ours_cm=float(np.percentile(fl_r[sl], 5) * 100),
                        accel_base_cm=float(ab[s2].mean()), accel_ours_cm=float(ar[s2].mean()),
                        rot_jerk_base_deg=float(rb[s2[s2 < len(rb)]].mean()), rot_jerk_ours_deg=float(rr[s2[s2 < len(rr)]].mean()),
                        dz_mean_cm=float((zf[sl] - zb[sl]).mean() * 100), dz_median_cm=float(np.median(zf[sl] - zb[sl]) * 100),
                        rot_change_vs_base_mean_deg=float(rotdist(T_final_c[sl, :3, :3], Tc_base[sl, :3, :3]).mean()),
                        pos_change_vs_base_mean_cm=float(np.linalg.norm(centre(T_final_c[sl]) - centre(Tc_base[sl]), axis=1).mean() * 100))
    # static-phase self-consistency of the raw per-frame fits vs the baseline
    for k, sl in (('A', sA), ('B', sB)):
        table[k]['spread_base'] = dict(z_std_cm=float(centre(Tc_base[sl])[:, 2].std() * 100),
                                       rot_dev_deg=float(rotdist(Tc_base[sl, :3, :3], rot_mean(Tc_base[sl, :3, :3])[None]).mean()))
        table[k]['spread_perframe_fit'] = dict(z_std_cm=float(centre(T_fit[sl])[:, 2].std() * 100),
                                               rot_dev_deg=float(rotdist(T_fit[sl, :3, :3], rot_mean(T_fit[sl, :3, :3])[None]).mean()))
    diag = dict(phases={k: [int(a), int(b)] for k, (a, b) in phases.items()}, symmetry=sym_votes, flip=bool(flip),
                depth_cues=depth_cues, table=table, runtime_s=time.time() - t_start)
    with open(os.path.splitext(args.out)[0] + '_diagnostics.json', 'w') as f:
        json.dump(diag, f, indent=1)
    print(json.dumps(table, indent=1))
    print('done in %.1f s' % (time.time() - t_start))


if __name__ == '__main__':
    main()
