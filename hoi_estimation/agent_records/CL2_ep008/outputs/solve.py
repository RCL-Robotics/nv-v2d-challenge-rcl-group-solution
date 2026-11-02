#!/usr/bin/env python
"""Trajectory-level re-estimation of the object pose for one HOI video.

Method (see REPORT.md):
 1. Phase segmentation from the SAM2 object/person masks: static-A (chair untouched / resting),
    moving (pushed), static-B (resting at the table).  A frame is "same as reference" when the
    object-mask IoU with the first (last) frame, restricted to pixels not covered by the person, is high.
 2. Render-and-compare silhouette fitting with nvdiffrast (exact textured mesh, multi-scale L2 on
    the soft silhouette, person pixels ignored):
      - static phases: ONE pose fitted jointly to all (sub-sampled) frames of the phase;
      - moving phase: per-frame fit initialised from the baseline, then temporal smoothing
        (Gaussian kernel regression on SE(3); depth along the ray smoothed more strongly).
    A weak prior keeps the chair upright w.r.t. the released floor plane (legs plane || floor).
 3. Depth re-anchoring: silhouette fitting has a systematic depth (scale) bias, so camera-frame
    positions are scaled along the viewing rays such that the median camera depth of each phase
    equals the baseline's median depth over the same frames (scale interpolated through the moving phase).
 4. Frames are mapped back to world with T_world_cam.
Deterministic: fixed seeds, no randomness in the optimisation (Adam from fixed inits).
"""
import argparse, json, time, os, sys
import numpy as np, torch, h5py, cv2, trimesh, roma
import torch.nn.functional as Fn
import nvdiffrast.torch as dr

torch.manual_seed(0); np.random.seed(0)
torch.backends.cudnn.deterministic = True

# ----------------------------------------------------------------------------------- constants
S = 512                 # render crop size (full-res pixels)
PERSON_DILATE = 7       # px; person pixels (dilated) are ignored in all silhouette terms
FIT_ITERS = 300         # Adam iterations for silhouette fits
FIT_LR = 3e-3
LAM_UP = 1.0            # weak upright prior (silhouettes alone prefer ~1.5 deg forward pitch w.r.t. the
                        # released floor normal; exact upright costs 1.3% IoU -> compromise, see REPORT)
STATIC_STRIDE = 3       # sub-sampling of frames used for the pooled static fits
SIG_ROT, SIG_LAT, SIG_Z = 2.5, 2.5, 6.0   # temporal smoothing (frames) in the moving phase; chosen by
                                          # held-out (even->odd frame) silhouette IoU, depth is IoU-blind
                                          # and noisier, hence the larger kernel.
SEG_SMOOTH = 10         # a phase boundary needs this many consecutive frames past the IoU threshold
SEG_DROP = 0.04         # threshold = median IoU of the first/last 60 frames minus this


# ----------------------------------------------------------------------------------- io
def load_inputs(inp):
    cam = json.load(open(f'{inp}/camera.json'))
    K = np.array(cam['K'], dtype=np.float64); Twc = np.array(cam['T_world_cam'], dtype=np.float64)
    rec = np.load(f'{inp}/reconstruction.npz')
    Tw = rec['T_world_mesh'].astype(np.float64)
    pl = json.load(open(f'{inp}/ground_plane.json'))['plane']
    mesh = trimesh.load(f'{inp}/object_mesh.glb', force='scene', process=False).to_geometry()
    return K, Twc, Tw, rec, np.array(pl, dtype=np.float64), mesh


def leg_plane_normal(mesh):
    """'up' direction of the chair in mesh coordinates = normal of the plane through the 4 leg tips.
    Leg tips: lowest vertex (max mesh z; mesh +z points down) in each xy-quadrant of the lower part."""
    V = np.asarray(mesh.vertices)
    low = V[V[:, 2] > V[:, 2].min() + 0.8 * (V[:, 2].max() - V[:, 2].min())]
    cx, cy = np.median(low[:, 0]), np.median(low[:, 1])
    tips = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            q = low[(np.sign(low[:, 0] - cx) == sx) & (np.sign(low[:, 1] - cy) == sy)]
            if len(q): tips.append(q[np.argmax(q[:, 2])])
    tips = np.array(tips)
    n = np.linalg.svd(tips - tips.mean(0))[2][2]
    if n[2] > 0: n = -n
    return n, tips


# ----------------------------------------------------------------------------------- rendering
class CropRenderer:
    def __init__(self, mesh, dev='cuda'):
        self.dev = dev
        self.v = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=dev)
        self.f = torch.tensor(np.asarray(mesh.faces), dtype=torch.int32, device=dev)
        self.glctx = dr.RasterizeCudaContext()
        self.c = self.v.mean(0)

    def render(self, T, Kt, boxes):
        """soft silhouettes (B,S,S) of poses T (B,4,4, camera-from-mesh) in crops with top-left boxes (B,2)."""
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        pc = torch.einsum('bij,nj->bni', T, vh)[..., :3]
        X, Y, Z = pc[..., 0], pc[..., 1], pc[..., 2]
        cx = Kt[0, 2] - boxes[:, 0:1]; cy = Kt[1, 2] - boxes[:, 1:2]
        u = Kt[0, 0] * X + cx * Z; v = Kt[1, 1] * Y + cy * Z
        n, f = 0.05, 20.
        clip = torch.stack([(2 * u + Z) / S - Z, (2 * v + Z) / S - Z,
                            Z * (f + n) / (f - n) - 2 * f * n / (f - n), Z], -1).contiguous()
        rast, _ = dr.rasterize(self.glctx, clip, self.f, (S, S))
        m, _ = dr.interpolate(torch.ones_like(clip[..., :1]), rast, self.f)
        m = dr.antialias(m, rast, clip, self.f)
        return m[..., 0]


def ms_loss(r, m, w, scales=(8, 4, 2, 1)):
    tot = 0
    for s in scales:
        if s > 1:
            ws = Fn.avg_pool2d(w[:, None], s)[:, 0]
            rs = Fn.avg_pool2d(r[:, None] * w[:, None], s)[:, 0] / ws.clamp(min=1e-3)
            ms = Fn.avg_pool2d(m[:, None] * w[:, None], s)[:, 0] / ws.clamp(min=1e-3)
            ws = (ws > 0.5).float()
        else:
            rs, ms, ws = r, m, w
        tot = tot + ((rs - ms) ** 2 * ws).sum((1, 2)) / ws.sum((1, 2)).clamp(min=1)
    return tot / len(scales)


def iou(r, m, w):
    r = (r > 0.5).float()
    return (r * m * w).sum((1, 2)) / ((r + m - r * m) * w).sum((1, 2)).clamp(min=1)


class MaskData:
    """object masks / visibility weights cropped around the object, for a list of frames"""
    def __init__(self, h5, idxs, dev='cuda'):
        ker = np.ones((PERSON_DILATE, PERSON_DILATE), np.uint8)
        M, W, B = [], [], []
        for i in idxs:
            o = h5['object'][i]; p = h5['person'][i]
            pd = cv2.dilate(p.astype(np.uint8), ker) > 0
            ys, xs = np.nonzero(o)
            cx, cy = ((xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2) if len(xs) else (768, 576)
            x0 = int(np.clip(cx - S // 2, 0, o.shape[1] - S)); y0 = int(np.clip(cy - S // 2, 0, o.shape[0] - S))
            M.append(o[y0:y0 + S, x0:x0 + S]); W.append(~pd[y0:y0 + S, x0:x0 + S]); B.append((x0, y0))
        self.idxs = list(idxs)
        self.M = torch.tensor(np.stack(M), dtype=torch.float32, device=dev)
        self.W = torch.tensor(np.stack(W), dtype=torch.float32, device=dev)
        self.B = torch.tensor(np.array(B), dtype=torch.float32, device=dev)


def fit(R, Kt, data, Tinit, pidx, up_cam=None, leg_n=None, iters=FIT_ITERS, chunk=48):
    """Fit P poses (Tinit (P,4,4)); data frame j uses pose pidx[j]. Rotation is perturbed about the
    object centre. Returns fitted poses and per-data-frame IoU."""
    dev = 'cuda'
    P = len(Tinit)
    R0 = torch.tensor(Tinit[:, :3, :3], dtype=torch.float32, device=dev)
    t0 = torch.tensor(Tinit[:, :3, 3], dtype=torch.float32, device=dev)
    w = torch.zeros(P, 3, device=dev, requires_grad=True); dt = torch.zeros(P, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([w, dt], lr=FIT_LR)
    pi = torch.tensor(pidx, dtype=torch.long, device=dev)
    nd = len(pidx)
    if up_cam is not None:
        upc = torch.tensor(up_cam, dtype=torch.float32, device=dev); lg = torch.tensor(leg_n, dtype=torch.float32, device=dev)
    Rc = (R0 @ R.c[:, None])[..., 0]
    def build():
        Rw = roma.rotvec_to_rotmat(w)
        T = torch.zeros(P, 4, 4, device=dev); T[:, :3, :3] = Rw @ R0
        T[:, :3, 3] = Rc - (Rw @ Rc[..., None])[..., 0] + t0 + dt; T[:, 3, 3] = 1
        return T
    for it in range(iters):
        opt.zero_grad()
        T = build()
        Tl = T.detach().requires_grad_(True)
        for k in range(0, nd, chunk):
            r = R.render(Tl[pi[k:k + chunk]], Kt, data.B[k:k + chunk])
            (ms_loss(r, data.M[k:k + chunk], data.W[k:k + chunk]).sum() / nd).backward()
        loss = (T * Tl.grad).sum()
        if up_cam is not None and LAM_UP > 0:
            loss = loss + LAM_UP * (((T[:, :3, :3] @ lg) - upc) ** 2).sum(1).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        T = build(); ious = []
        for k in range(0, nd, chunk):
            r = R.render(T[pi[k:k + chunk]], Kt, data.B[k:k + chunk])
            ious.append(iou(r, data.M[k:k + chunk], data.W[k:k + chunk]))
    return T.cpu().double().numpy(), torch.cat(ious).cpu().numpy()


def eval_iou(R, Kt, data, Tcams, chunk=64):
    out = []
    with torch.no_grad():
        for k in range(0, len(data.idxs), chunk):
            T = torch.tensor(Tcams[k:k + chunk], dtype=torch.float32, device='cuda')
            out.append(iou(R.render(T, Kt, data.B[k:k + chunk]), data.M[k:k + chunk], data.W[k:k + chunk]))
    return torch.cat(out).cpu().numpy()


# ----------------------------------------------------------------------------------- pose utils
def chordal_mean(Rs, w=None):
    w = np.ones(len(Rs)) if w is None else w
    U, _, Vt = np.linalg.svd((w[:, None, None] * Rs).sum(0))
    return U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt


def rotdeg(Ra, Rb):
    Rr = np.swapaxes(Ra, -1, -2) @ Rb
    return np.degrees(np.arccos(np.clip((np.trace(Rr, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def smooth_traj(T, sig_rot, sig_lat, sig_z, wts=None):
    """Gaussian kernel smoothing of a camera-frame pose sequence (one pose per frame).
    lateral position (x/z, y/z) and depth z are smoothed separately."""
    F = len(T); t = np.arange(F); wts = np.ones(F) if wts is None else wts
    out = np.zeros_like(T); out[:, 3, 3] = 1
    xz = T[:, 0, 3] / T[:, 2, 3]; yz = T[:, 1, 3] / T[:, 2, 3]; z = T[:, 2, 3]
    for k in range(F):
        def kw(s):
            g = np.exp(-0.5 * ((t - k) / s) ** 2) * wts; return g / g.sum()
        wr, wl, wz = kw(sig_rot), kw(sig_lat), kw(sig_z)
        out[k, :3, :3] = chordal_mean(T[:, :3, :3], wr)
        zz = (wz * z).sum()
        out[k, :3, 3] = np.array([(wl * xz).sum() * zz, (wl * yz).sum() * zz, zz])
    return out


# ----------------------------------------------------------------------------------- phases
def segment_phases(h5, F):
    """returns (a, b): static-A = [0, a), moving = [a, b), static-B = [b, F)."""
    O0 = h5['object'][0][::2, ::2]; O1 = h5['object'][F - 1][::2, ::2]
    iA = np.zeros(F); iB = np.zeros(F)
    for s in range(0, F, 32):
        O = h5['object'][s:s + 32][:, ::2, ::2]; P = h5['person'][s:s + 32][:, ::2, ::2]
        for j in range(O.shape[0]):
            v = ~P[j]
            iA[s + j] = (O[j] & O0 & v).sum() / max(((O[j] | O0) & v).sum(), 1)
            iB[s + j] = (O[j] & O1 & v).sum() / max(((O[j] | O1) & v).sum(), 1)
    thA = np.median(iA[:60]) - SEG_DROP; thB = np.median(iB[-60:]) - SEG_DROP
    # occlusion by the person can only lower these IoUs, so we look for runs of HIGH IoU:
    # static-B starts at the first run of SEG_SMOOTH consecutive frames matching the last frame;
    # static-A ends after the last such run (before b) matching frame 0.
    b = next(t for t in range(F - SEG_SMOOTH) if np.all(iB[t:t + SEG_SMOOTH] >= thB))
    a = max(t for t in range(SEG_SMOOTH, b + 1) if np.all(iA[t - SEG_SMOOTH:t] >= thA))
    return a, b, iA, iB


# ----------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='/work/inputs')
    ap.add_argument('--out', default='/work/outputs/object_poses_world.npz')
    ap.add_argument('--debug_dir', default=None, help='optional: dump intermediates for the report')
    args = ap.parse_args()
    t_start = time.time()
    K, Twc, Tw, rec, plane, mesh = load_inputs(args.inputs)
    F = Tw.shape[0]
    Tcw = np.linalg.inv(Twc)
    Tbase = Tcw[None] @ Tw                                  # baseline, camera frame
    h5 = h5py.File(f'{args.inputs}/masks.h5', 'r')
    leg_n, _ = leg_plane_normal(mesh)
    up_cam = Tcw[:3, :3] @ plane[:3]                        # floor normal ("up", world is y-down) in camera frame

    # 1. phases
    a, b, iA, iB = segment_phases(h5, F)
    print(f'phases: static-A [0,{a})  moving [{a},{b})  static-B [{b},{F})', flush=True)

    R = CropRenderer(mesh); Kt = torch.tensor(K, dtype=torch.float32, device='cuda')

    # 2a. static phases: one pose each, pooled over the phase
    static = {}
    for name, (s0, s1) in (('A', (0, a)), ('B', (b, F))):
        idx = list(range(s0, s1, STATIC_STRIDE))
        d = MaskData(h5, idx)
        T0 = np.eye(4); T0[:3, :3] = chordal_mean(Tbase[s0:s1, :3, :3]); T0[:3, 3] = np.median(Tbase[s0:s1, :3, 3], 0)
        Tf, ious = fit(R, Kt, d, T0[None], np.zeros(len(idx), int), up_cam, leg_n)
        static[name] = Tf[0]
        print(f'static {name}: pooled IoU {ious.mean():.4f}  baseline IoU {eval_iou(R, Kt, d, Tbase[idx]).mean():.4f}', flush=True)
        del d

    # 2b. moving phase: per-frame fits from baseline init
    midx = list(range(a, b))
    Tm = np.zeros((len(midx), 4, 4)); im = np.zeros(len(midx))
    for k in range(0, len(midx), 96):
        sub = midx[k:k + 96]
        d = MaskData(h5, sub)
        Tm[k:k + len(sub)], im[k:k + len(sub)] = fit(R, Kt, d, Tbase[sub], np.arange(len(sub)), up_cam, leg_n)
        del d
    torch.cuda.empty_cache()
    print(f'moving per-frame IoU {im.mean():.4f}', flush=True)

    # 3. assemble full sequence (static poses repeated) and smooth through the moving phase.
    # robust weights: frames whose fit is poor (low IoU) count less in the smoothing.
    Tseq = np.concatenate([np.repeat(static['A'][None], a, 0), Tm, np.repeat(static['B'][None], F - b, 0)])
    wts = np.ones(F); wts[a:b] = np.clip((im - 0.5) / 0.3, 0.05, 1.0)
    Ts = smooth_traj(Tseq, SIG_ROT, SIG_LAT, SIG_Z, wts)
    Ts[:a] = static['A']; Ts[b:] = static['B']

    # 4. depth re-anchoring along the viewing rays (per phase; scale interpolated through the moving phase)
    def ratio(sl): return np.median(Tbase[sl, 2, 3]) / np.median(Ts[sl, 2, 3])
    sA, sB, sM = ratio(slice(0, a)), ratio(slice(b, F)), ratio(slice(a, b))
    # moving phase: linear interpolation sA->sB (continuity) plus a bump c*ramp (zero at both ends)
    # with c chosen such that the moving-phase median depth equals the baseline's.
    s = np.ones(F); s[:a] = sA; s[b:] = sB
    tm = np.arange(a, b)
    lin = sA + (sB - sA) * (tm - a + 0.5) / (b - a)
    ramp = np.clip(np.minimum(tm - a + 1, b - tm) / 15.0, 0, 1)
    zb = np.median(Tbase[a:b, 2, 3]); lo, hi = -0.2, 0.2
    for _ in range(60):
        c = 0.5 * (lo + hi)
        if np.median(Ts[a:b, 2, 3] * (lin + c * ramp)) < zb: lo = c
        else: hi = c
    s[a:b] = lin + c * ramp
    print(f'depth scales: A {sA:.4f}  moving {sM:.4f}  B {sB:.4f}', flush=True)
    Tfin = Ts.copy(); Tfin[:, :3, 3] *= s[:, None]

    Tworld = Twc[None] @ Tfin
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, T_world_mesh=Tworld.astype(np.float64))
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        np.savez(f'{args.debug_dir}/debug.npz', a=a, b=b, iA=iA, iB=iB, Tm=Tm, im=im, TA=static['A'], TB=static['B'],
                 Ts=Ts, s=s, Tfin=Tfin)
    print(f'done in {time.time() - t_start:.1f}s -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
