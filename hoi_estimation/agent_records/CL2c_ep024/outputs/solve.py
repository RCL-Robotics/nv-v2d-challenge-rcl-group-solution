#!/usr/bin/env python
"""Trajectory-level re-estimation of the 6-DoF object pose for one HOI video.

Method (see REPORT.md for the reasoning and validation):
  1. Phase segmentation from the SAM2 object masks: an initial static phase S1 (object untouched / resting on
     the floor) and a final static phase S2 are the maximal prefix / suffix whose object mask agrees
     (IoU >= STATIC_IOU, pixels covered by the person ignored) with the median mask of the first / last
     REF_LEN frames. Everything in between is the "carried" phase.
  2. Silhouette model: the textured mesh is rasterised with nvdiffrast (differentiable, anti-aliased) and
     dilated by MASK_BLOAT_PX pixels, because the SAM2 masks of this video are systematically ~2-3 px too
     large on every side (verified on frame 0: the person mask is ~4.5% larger than the projection of the
     true frame-0 human, but centred on it). Loss = 1 - soft IoU over pixels not covered by the person.
  3. Static phases: ONE pose per phase, fitted to all its frames jointly (init: chordal mean of baseline).
  4. Carried phase: all poses fitted jointly with the image term + second-order smoothness on rotation and
     translation, with the two static poses as fixed boundary conditions (init: baseline).
  5. Depth (distance along the viewing ray) is NOT taken from the silhouette. It is decoded from the
     baseline: constant = median baseline camera-z in each static phase, and a robust (Huber-IRLS)
     second-difference-smoothed baseline z in the carried phase, continuous with the static constants.
     Each fitted pose is then moved along its own viewing ray to that depth (image position kept).
     Depth shift w.r.t. the baseline applied on top of this: 0 (see REPORT.md, cue analysis).
  6. Map back to the world with T_world_cam.
Fixed seeds, no randomness, no network access. nvdiffrast's backward pass uses GPU atomic adds, so repeated runs are
not bit-identical; measured run-to-run spread is 0.35 deg / 0.09 cm mean (see REPORT.md).
"""
import argparse, json, os, sys, time
import numpy as np
import h5py, torch, trimesh
import torch.nn.functional as Fn
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot

# ---------------- constants (chosen on this episode; reasons in REPORT.md) ----------------
STATIC_IOU = 0.94      # mask IoU vs. phase reference above which the object counts as not moved
REF_LEN = 30           # frames used to build the start / end reference masks
MASK_BLOAT_PX = 2      # SAM2 masks are ~2-3 px too large per side (frame-0 true human check)
SIL_FACES = 20000      # decimated mesh for silhouettes
STATIC_ITERS = 500
CARRIED_ITERS = 600
LAMBDA_ROT = 1e-4      # weight of rotation 2nd-difference penalty (chosen by held-out-frame IoU)
LAMBDA_TRANS = 1e-3    # weight of translation 2nd-difference penalty (chosen by held-out-frame IoU)
Z_SMOOTH = 200.0       # 2nd-difference weight for decoding the carried-phase depth from the baseline
PRIOR_ROT = 0.1        # weak pull of every rotation toward its init (baseline) -> unique optimum along directions the
                       # silhouette cannot see (also makes runs reproducible despite non-deterministic GPU atomics)
PRIOR_TRANS = 1.0      # same for translation (units of the internal parameters)
HUBER_Z = 0.02         # m, Huber scale of baseline-depth residuals
CHUNK = 128
dev = 'cuda'


def seed_all(s=0):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def so3_exp(w):
    th = w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = w / th
    Kx = torch.zeros(w.shape[:-1] + (3, 3), device=w.device, dtype=w.dtype)
    Kx[..., 0, 1] = -k[..., 2]; Kx[..., 0, 2] = k[..., 1]
    Kx[..., 1, 0] = k[..., 2]; Kx[..., 1, 2] = -k[..., 0]
    Kx[..., 2, 0] = -k[..., 1]; Kx[..., 2, 1] = k[..., 0]
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(Kx)
    s = torch.sin(th)[..., None]; c = torch.cos(th)[..., None]
    return I + s * Kx + (1 - c) * (Kx @ Kx)


def pose_from(T0, w, t):
    R = so3_exp(w + 1e-9) @ T0[..., :3, :3]
    tt = T0[..., :3, 3] + t
    top = torch.cat([R, tt[..., None]], -1)
    bot = torch.zeros(top.shape[:-2] + (1, 4), device=dev); bot[..., 0, 3] = 1
    return torch.cat([top, bot], -2)


def mean_pose(Ts):
    T = np.eye(4)
    T[:3, :3] = Rot.from_matrix(Ts[:, :3, :3]).mean().as_matrix()
    T[:3, 3] = np.median(Ts[:, :3, 3], 0)
    return T


class SilRenderer:
    def __init__(self, mesh_path, K, crop):
        import open3d as o3d
        m = trimesh.load(mesh_path, force='scene', process=False).to_geometry()
        om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
        om = om.merge_close_vertices(1e-6).simplify_quadric_decimation(SIL_FACES)
        self.v = torch.tensor(np.asarray(om.vertices), dtype=torch.float32, device=dev)
        self.f = torch.tensor(np.asarray(om.triangles), dtype=torch.int32, device=dev)
        self.ctx = dr.RasterizeCudaContext()
        x0, y0, W, H = crop
        self.W, self.H = W, H
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2] - x0, K[1, 2] - y0
        n, f = 0.05, 20.
        self.P = torch.tensor([[2 * fx / W, 0, 2 * (cx + 0.5) / W - 1, 0],
                               [0, 2 * fy / H, 2 * (cy + 0.5) / H - 1, 0],
                               [0, 0, (f + n) / (f - n), -2 * f * n / (f - n)],
                               [0, 0, 1, 0]], dtype=torch.float32, device=dev)

    def __call__(self, Tcm):
        vc = Tcm[:, None, :3, :3].matmul(self.v[None, :, :, None])[..., 0] + Tcm[:, None, :3, 3]
        clip = (torch.cat([vc, torch.ones_like(vc[..., :1])], -1) @ self.P.T).contiguous()
        rast, _ = dr.rasterize(self.ctx, clip, self.f, (self.H, self.W))
        sil = dr.antialias((rast[..., 3:4] > 0).float().contiguous(), rast, clip, self.f)[..., 0]
        if MASK_BLOAT_PX > 0:
            # soft dilation by MASK_BLOAT_PX px: 1 - prod_{neighbours}(1 - sil)  (deterministic conv, unlike max-pool backward)
            r = MASK_BLOAT_PX
            yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            ker = torch.tensor((xx ** 2 + yy ** 2 <= r * r + 0.5).astype(np.float32), device=dev)[None, None]
            lg = torch.log((1 - sil).clamp(min=1e-4))[:, None]
            sil = 1 - torch.exp(Fn.conv2d(lg, ker, padding=r))[:, 0]
        return sil


def soft_iou(sil, tg, valid):
    inter = (sil * tg * valid).sum((-1, -2)); uni = ((sil + tg - sil * tg) * valid).sum((-1, -2))
    return inter / uni.clamp(min=1)


def segment_phases(om, pm):
    F = len(om)
    valid = ~(pm & ~om)

    def score(ref):
        inter = (om & ref[None] & valid).sum((1, 2)); uni = ((om | ref[None]) & valid).sum((1, 2))
        return inter / np.maximum(uni, 1)
    s_start = score(np.mean(om[:REF_LEN], 0) > 0.5)
    s_end = score(np.mean(om[F - REF_LEN:], 0) > 0.5)
    bad = np.nonzero(s_start < STATIC_IOU)[0]
    a = int(bad[0]) if len(bad) else F
    bad = np.nonzero(s_end < STATIC_IOU)[0]
    b = int(bad[-1]) + 1 if len(bad) else 0
    return (0, a), (b, F), s_start, s_end


def fit_static(R, T0, tg, valid, iters):
    T0t = torch.tensor(T0, device=dev, dtype=torch.float32)
    w = torch.zeros(3, device=dev, requires_grad=True); t = torch.zeros(3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w, t], lr=0.003)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=1e-5)
    tsc = torch.tensor([1, 1, 5.], device=dev)
    n = tg.shape[0]
    for it in range(iters):
        opt.zero_grad()
        T = pose_from(T0t, w, t * tsc)
        for c in range(0, n, CHUNK):
            sil = R(T[None].expand(min(CHUNK, n - c), 4, 4).contiguous())
            l = (1 - soft_iou(sil, tg[c:c + CHUNK], valid[c:c + CHUNK])).sum() / n
            l.backward(retain_graph=True)
        (PRIOR_ROT * (w ** 2).sum() + PRIOR_TRANS * (t ** 2).sum()).backward()
        opt.step(); sched.step()
    with torch.no_grad():
        return pose_from(T0t, w, t * tsc).cpu().numpy().astype(np.float64)


def fit_carried(R, T_init, T_left, T_right, tg, valid, iters):
    """T_init (N,4,4) for the carried frames; T_left / T_right fixed static poses before / after."""
    N = len(T_init)
    T0t = torch.tensor(T_init, device=dev, dtype=torch.float32)
    TL = torch.tensor(T_left, device=dev, dtype=torch.float32)[None]
    TR = torch.tensor(T_right, device=dev, dtype=torch.float32)[None]
    w = torch.zeros(N, 3, device=dev, requires_grad=True); t = torch.zeros(N, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w, t], lr=0.003)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=1e-5)
    tsc = torch.tensor([1, 1, 5.], device=dev)
    for it in range(iters):
        opt.zero_grad()
        Tall = torch.cat([TL, TL, pose_from(T0t, w, t * tsc), TR, TR], 0)
        Rm = Tall[:, :3, :3]; tr = Tall[:, :3, 3]
        acc_t = tr[2:] - 2 * tr[1:-1] + tr[:-2]
        pred = Rm[1:-1] @ Rm[:-2].transpose(1, 2) @ Rm[1:-1]
        acc_r = (Rm[2:] - pred).flatten(1)
        lsm = LAMBDA_ROT * 1e3 * (acc_r ** 2).sum() + LAMBDA_TRANS * (1e5 * (acc_t[:, :2] ** 2).sum() + 1e4 * (acc_t[:, 2] ** 2).sum())
        lsm = lsm + PRIOR_ROT * (w ** 2).sum() + PRIOR_TRANS * (t ** 2).sum()
        lsm.backward()
        for c in range(0, N, CHUNK):
            T = pose_from(T0t[c:c + CHUNK], w[c:c + CHUNK], t[c:c + CHUNK] * tsc)
            sil = R(T.contiguous())
            l = (1 - soft_iou(sil, tg[c:c + CHUNK], valid[c:c + CHUNK])).sum()
            l.backward()
        opt.step(); sched.step()
    with torch.no_grad():
        return pose_from(T0t, w, t * tsc).cpu().numpy().astype(np.float64)


def decode_depth(zb, S1, S2):
    """robust smooth depth: constant (baseline median) in static phases, Huber + 2nd-difference smoothing in between."""
    F = len(zb)
    z1 = float(np.median(zb[S1[0]:S1[1]])); z2 = float(np.median(zb[S2[0]:S2[1]]))
    a, b = S1[1], S2[0]
    n = b - a
    if n <= 0:
        z = np.empty(F); z[:S1[1]] = z1; z[S2[0]:] = z2; return z, z1, z2
    # unknowns z[a..b-1]; fixed: z[a-2],z[a-1]=z1 ; z[b],z[b+1]=z2
    ext = n + 4
    D = np.zeros((ext - 2, ext))
    for i in range(ext - 2):
        D[i, i], D[i, i + 1], D[i, i + 2] = 1, -2, 1
    Du, Dk = D[:, 2:2 + n], np.concatenate([D[:, :2], D[:, 2 + n:]], 1)
    zk = np.array([z1, z1, z2, z2])
    y = zb[a:b]
    wts = np.ones(n)
    z = y.copy()
    for _ in range(20):
        A = np.diag(wts) + Z_SMOOTH * Du.T @ Du
        rhs = wts * y - Z_SMOOTH * Du.T @ (Dk @ zk)
        z = np.linalg.solve(A, rhs)
        r = np.abs(z - y)
        wts = np.where(r <= HUBER_Z, 1.0, HUBER_Z / np.maximum(r, 1e-9))
    out = np.empty(F); out[:S1[1]] = z1; out[S2[0]:] = z2; out[a:b] = z
    return out, z1, z2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='/work/inputs')
    ap.add_argument('--out', default='/work/outputs/object_poses_world.npz')
    ap.add_argument('--diag', default=None, help='optional path for a diagnostics npz')
    ap.add_argument('--holdout', type=int, default=0, help='validation only: ignore the image term of odd frames')
    args = ap.parse_args()
    t0 = time.time()
    seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    inp = args.inputs
    cam = json.load(open(f'{inp}/camera.json'))
    K = np.array(cam['K'], float); Twc = np.array(cam['T_world_cam'], float); Tcw = np.linalg.inv(Twc)
    rec = np.load(f'{inp}/reconstruction.npz')
    Tw_base = rec['T_world_mesh'].astype(np.float64)
    Tc_base = Tcw[None] @ Tw_base
    F = len(Tw_base)

    with h5py.File(f'{inp}/masks.h5') as f:
        om_full = f['object'][:]; pm_full = f['person'][:]
    # crop window around all object masks (+ margin)
    ys, xs = np.nonzero(om_full.any(0))
    m = 60
    x0, x1 = max(0, xs.min() - m), min(om_full.shape[2], xs.max() + m)
    y0, y1 = max(0, ys.min() - m), min(om_full.shape[1], ys.max() + m)
    crop = (int(x0), int(y0), int(x1 - x0), int(y1 - y0))
    om = om_full[:, y0:y1, x0:x1]; pm = pm_full[:, y0:y1, x0:x1]
    del om_full, pm_full
    valid = ~(pm & ~om)
    S1, S2, s_start, s_end = segment_phases(om, pm)
    print(f'phases: S1 {S1}  carried [{S1[1]},{S2[0]})  S2 {S2}  crop {crop}', flush=True)

    R = SilRenderer(f'{inp}/object_mesh.glb', K, crop)
    tg = torch.tensor(om, device=dev, dtype=torch.float32)
    va = torch.tensor(valid, device=dev, dtype=torch.float32)
    va_fit = va.clone()
    if args.holdout:
        va_fit[1::2] = 0  # held-out frames contribute nothing to the fit

    # ---- static phases
    T_S = []
    for (a, b) in [S1, S2]:
        T0 = mean_pose(Tc_base[a:b])
        T_S.append(fit_static(R, T0, tg[a:b], va_fit[a:b], STATIC_ITERS))
    print('static fits done %.1fs' % (time.time() - t0), flush=True)
    # ---- carried phase
    a, b = S1[1], S2[0]
    T_fit = np.empty((F, 4, 4))
    T_fit[S1[0]:S1[1]] = T_S[0]; T_fit[S2[0]:S2[1]] = T_S[1]
    if b > a:
        T_fit[a:b] = fit_carried(R, Tc_base[a:b], T_S[0], T_S[1], tg[a:b], va_fit[a:b], CARRIED_ITERS)
    print('carried fit done %.1fs' % (time.time() - t0), flush=True)

    # ---- image-space quality (diagnostic)
    with torch.no_grad():
        iou_fit, iou_base = [], []
        for c in range(0, F, CHUNK):
            for T, lst in [(T_fit, iou_fit), (Tc_base, iou_base)]:
                sil = R(torch.tensor(T[c:c + CHUNK], device=dev, dtype=torch.float32))
                lst.append(soft_iou((sil > 0.5).float(), tg[c:c + CHUNK], va[c:c + CHUNK]).cpu().numpy())
        iou_fit = np.concatenate(iou_fit); iou_base = np.concatenate(iou_base)

    # ---- depth: decoded from the baseline, poses moved along their own viewing ray
    zb = Tc_base[:, 2, 3]
    z_dec, z1, z2 = decode_depth(zb, S1, S2)
    DEPTH_SHIFT = 0.0  # independent physical cues disagree (REPORT.md) -> keep the baseline depth
    z_dec = z_dec + DEPTH_SHIFT
    T_final_c = T_fit.copy()
    T_final_c[:, :3, 3] *= (z_dec / T_fit[:, 2, 3])[:, None]
    T_world = Twc[None] @ T_final_c
    T_world[:, 3] = [0, 0, 0, 1]
    # orthonormalise
    for i in range(F):
        U, _, Vt = np.linalg.svd(T_world[i, :3, :3]); T_world[i, :3, :3] = U @ Vt
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, T_world_mesh=T_world.astype(np.float64))
    if args.diag:
        np.savez(args.diag, T_fit=T_fit, T_final_c=T_final_c, iou_fit=iou_fit, iou_base=iou_base, z_dec=z_dec,
                 S1=np.array(S1), S2=np.array(S2), s_start=s_start, s_end=s_end, crop=np.array(crop))
    print(f'IoU (dilated render vs SAM mask) baseline {iou_base.mean():.4f}  fit {iou_fit.mean():.4f}')
    print(f'wrote {args.out}  ({F} frames)  total {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
