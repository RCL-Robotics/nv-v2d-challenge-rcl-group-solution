#!/usr/bin/env python
"""Trajectory-level 6-DoF pose recovery for the big red bowl (episode 9).

Pipeline (all estimates recomputed from the inputs; deterministic):
  1. Phase segmentation from the SAM2 object-mask bounding boxes (static = the bbox edges do not move).
  2. Per-frame silhouette render-and-compare (nvdiffrast) from the baseline initialisation: image position,
     depth and axis (tilt) of the bowl; reliability from visibility (person occlusion) and IoU.
  3. Static phases: one pose per phase, fitted to all masks of the phase jointly (pooled evidence).
  4. Moving phases: robust temporal fusion of silhouette fits, rigid hand attachment (image position of the
     reconstructed hands + smooth offset) and the baseline, followed by local-linear smoothing.
  5. Yaw about the (rotationally symmetric, untextured) bowl axis is not observable in the image; it is
     propagated by rigid attachment: constant while the bowl rests, rotating with the person's body/hands
     while carried; anchored to the baseline's yaw at the first static phase (frame-0 registration).
  6. Depth re-anchoring: per phase the median camera depth is set to the baseline median times (1+s), where s
     is a global relative correction applied only if independent physical cues agree (see depth_cues()).
"""
import argparse, json, os, sys, time
import numpy as np

T0 = time.time()
SEED = 0


def log(*a):
    print('[%7.1fs]' % (time.time() - T0), *a, flush=True)


# ----------------------------------------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------------------------------------
def rodrigues(axis, ang):
    axis = np.asarray(axis, float); axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def min_rot(a, b):
    """smallest rotation taking unit vector a to unit vector b"""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    c = np.cross(a, b); s = np.linalg.norm(c); d = float(a @ b)
    if s < 1e-9:
        return np.eye(3)
    return rodrigues(c / s, np.arctan2(s, d))


def rot_angle_deg(R1, R2):
    c = (np.trace(np.swapaxes(R1, -1, -2) @ R2, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def project(K, p):
    q = p @ K.T
    return q[..., :2] / q[..., 2:3]


def unproject(K, uv, z):
    return np.stack([(uv[..., 0] - K[0, 2]) / K[0, 0] * z, (uv[..., 1] - K[1, 2]) / K[1, 1] * z, z], -1)


def loclin_smooth(x, y, w, sigma, F=None, iters=3, c_huber=None, min_w=1e-3):
    """Robust local-linear Gaussian smoother evaluated on the integer grid 0..F-1.
    x (N,) sample times (may repeat: several measurement sources), y (N,D), w (N,) -> (F,D)"""
    y = np.asarray(y, float); w = np.asarray(w, float)
    if y.ndim == 1:
        y = y[:, None]
    if F is None:
        F = int(x.max()) + 1
    o = np.argsort(x, kind='stable'); x = np.asarray(x, float)[o]; y = y[o]; w = w[o]
    good = np.isfinite(y).all(1); y = np.where(good[:, None], y, 0); w = w * good
    out = np.full((F, y.shape[1]), np.nan); rw = np.ones(len(x)); R = 4 * sigma
    for it in range(iters):
        ww = w * rw
        for i in range(F):
            lo, hi = np.searchsorted(x, i - R), np.searchsorted(x, i + R, side='right')
            dx = x[lo:hi] - i
            k = np.exp(-0.5 * (dx / sigma) ** 2) * ww[lo:hi]
            S0, S1, S2 = k.sum(), (k * dx).sum(), (k * dx * dx).sum()
            if S0 < min_w:
                continue
            T0_ = (k[:, None] * y[lo:hi]).sum(0); T1 = (k[:, None] * dx[:, None] * y[lo:hi]).sum(0)
            det = S0 * S2 - S1 * S1
            one_sided = abs(S1 / S0) > 0.7 * sigma         # support only on one side: no linear extrapolation
            out[i] = T0_ / S0 if (det < 1e-6 * S0 * S0 or one_sided) else (S2 * T0_ - S1 * T1) / det
        if c_huber is None:
            break
        xi = np.clip(np.round(x).astype(int), 0, F - 1)
        r = np.linalg.norm(y - out[xi], axis=1); r[~np.isfinite(r)] = 0
        rw = np.where(r <= c_huber, 1.0, c_huber / np.maximum(r, 1e-12))
    bad = ~np.isfinite(out).all(1)
    if bad.any() and (~bad).any():                     # fill frames without support by linear interpolation
        g = np.arange(F)
        for c in range(out.shape[1]):
            out[bad, c] = np.interp(g[bad], g[~bad], out[~bad, c])
    return out


# ----------------------------------------------------------------------------------------------------------
# rendering (nvdiffrast)
# ----------------------------------------------------------------------------------------------------------
class SilRenderer:
    def __init__(self, verts, faces, K):
        import torch, nvdiffrast.torch as dr
        self.torch = torch; self.dr = dr
        self.dev = 'cuda'
        self.ctx = dr.RasterizeCudaContext()
        import open3d as o3d
        uq, inv = np.unique(np.round(verts, 7), axis=0, return_inverse=True)
        # silhouettes only: quadric decimation to 8k faces (deterministic; max deviation ~1 mm = 0.3 px at 3 m)
        m3 = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(uq), o3d.utility.Vector3iVector(inv.reshape(-1)[faces]))
        m3 = m3.simplify_quadric_decimation(8000)
        self.v = torch.tensor(np.asarray(m3.vertices), dtype=torch.float32, device=self.dev)
        self.f = torch.tensor(np.asarray(m3.triangles), dtype=torch.int32, device=self.dev)
        self.K = K

    def axang(self, w):
        torch = self.torch
        th = torch.sqrt((w * w).sum(-1, keepdim=True) + 1e-12); k = w / th
        Km = torch.zeros(w.shape[0], 3, 3, device=w.device, dtype=w.dtype)
        Km[:, 0, 1] = -k[:, 2]; Km[:, 0, 2] = k[:, 1]; Km[:, 1, 0] = k[:, 2]
        Km[:, 1, 2] = -k[:, 0]; Km[:, 2, 0] = -k[:, 1]; Km[:, 2, 1] = k[:, 0]
        th = th[..., None]
        return torch.eye(3, device=w.device)[None] + torch.sin(th) * Km + (1 - torch.cos(th)) * (Km @ Km)

    def render(self, R, t, xy0, S):
        torch, dr, K = self.torch, self.dr, self.K
        P = torch.einsum('bij,vj->bvi', R, self.v) + t[:, None, :]
        Z = P[..., 2].clamp(min=1e-3)
        u = K[0, 0] * P[..., 0] / Z + K[0, 2] - xy0[:, None, 0]
        v = K[1, 1] * P[..., 1] / Z + K[1, 2] - xy0[:, None, 1]
        n, f = 0.1, 50.
        clip = torch.stack([((2 * u + 1) / S - 1) * Z, ((2 * v + 1) / S - 1) * Z,
                            (f + n) / (f - n) * Z - 2 * f * n / (f - n), Z], -1).contiguous()
        rast, _ = dr.rasterize(self.ctx, clip, self.f, (S, S))
        a = dr.antialias((rast[..., 3:] > 0).float(), rast, clip, self.f)
        return a[..., 0]

    def blur(self, x, sigma):
        torch = self.torch
        if sigma <= 0:
            return x
        k = int(3 * sigma) * 2 + 1
        xs = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
        g = torch.exp(-xs ** 2 / (2 * sigma ** 2)); g = g / g.sum()
        x = torch.nn.functional.conv2d(x[:, None], g.view(1, 1, 1, k), padding=(0, k // 2))
        x = torch.nn.functional.conv2d(x, g.view(1, 1, k, 1), padding=(k // 2, 0))
        return x[:, 0]

    def fit(self, R0, uv0, z0, obj, valid, xy0, S, iters=(30, 30, 40), sigmas=(4, 2, 0.7),
            fit_rot=True, fit_z=True, axis_only=False):
        """Silhouette fit by Adam on central finite-difference gradients of a blurred-silhouette loss.
        Only forward rasterisation is used, so the result is bit-reproducible (nvdiffrast's backward uses atomics).
        Parameters per frame: image position (u,v), log depth, and two tilt angles about axes perpendicular to the
        bowl's symmetry axis (spin about the axis does not change the silhouette and is kept fixed).
        Returns R (B,3,3), t (B,3), iou (B,)"""
        torch = self.torch
        B = R0.shape[0]
        R0t = torch.tensor(R0, dtype=torch.float32, device=self.dev)
        ax = R0t[:, :, 1]
        tmp = torch.where((ax[:, :1].abs() < 0.9), torch.tensor([1., 0, 0], device=self.dev).expand(B, 3),
                          torch.tensor([0, 0, 1.], device=self.dev).expand(B, 3))
        e1 = torch.linalg.cross(ax, tmp); e1 = e1 / e1.norm(dim=-1, keepdim=True); e2 = torch.linalg.cross(ax, e1)
        Kt = torch.tensor(self.K, dtype=torch.float32, device=self.dev)
        xy = torch.tensor(xy0, dtype=torch.float32, device=self.dev)
        O = torch.tensor(obj, dtype=torch.float32, device=self.dev)
        V = torch.tensor(valid, dtype=torch.float32, device=self.dev)
        dims = [0, 1] + ([2] if fit_z else []) + ([3, 4] if fit_rot else [])
        p0 = torch.zeros(B, 5, device=self.dev)
        p0[:, 0:2] = torch.tensor(uv0, dtype=torch.float32, device=self.dev)
        p0[:, 2] = torch.tensor(np.log(z0), dtype=torch.float32, device=self.dev)
        p = p0.clone().requires_grad_(True)
        lr = torch.tensor([0.5, 0.5, 0.003, 0.01, 0.01], device=self.dev)
        opt = torch.optim.Adam([{'params': [p], 'lr': 1.0}])

        def poses(P):                                   # P (N,5) with frame index repeat -> R,t
            n = P.shape[0] // B
            w = P[:, 3:4] * e1.repeat(n, 1) + P[:, 4:5] * e2.repeat(n, 1)
            R = self.axang(w) @ R0t.repeat(n, 1, 1)
            z = torch.exp(P[:, 2])
            t = torch.stack([(P[:, 0] - Kt[0, 2]) / Kt[0, 0] * z, (P[:, 1] - Kt[1, 2]) / Kt[1, 1] * z, z], -1)
            return R, t

        def render_bin(R, t, xyr):
            dr = self.dr
            P = torch.einsum('bij,vj->bvi', R, self.v) + t[:, None, :]
            Z = P[..., 2].clamp(min=1e-3)
            u = self.K[0, 0] * P[..., 0] / Z + self.K[0, 2] - xyr[:, None, 0]
            v = self.K[1, 1] * P[..., 1] / Z + self.K[1, 2] - xyr[:, None, 1]
            n_, f_ = 0.1, 50.
            clip = torch.stack([((2 * u + 1) / S - 1) * Z, ((2 * v + 1) / S - 1) * Z,
                                (f_ + n_) / (f_ - n_) * Z - 2 * f_ * n_ / (f_ - n_), Z], -1).contiguous()
            out = []
            for c in range(0, clip.shape[0], 600):
                rast, _ = dr.rasterize(self.ctx, clip[c:c + 600], self.f, (S, S))
                out.append((rast[..., 3] > 0).float())
            return torch.cat(out)

        norm = O.sum((1, 2)) + 50
        for n_it, sg in zip(iters, sigmas):
            Ob = self.blur(O, sg)
            hs = torch.tensor([max(0.5, sg / 2), max(0.5, sg / 2), 0.004 * max(1., sg / 2), 0.006 * max(1., sg / 2),
                               0.006 * max(1., sg / 2)], device=self.dev)
            for _ in range(n_it):
                with torch.no_grad():
                    Ps = [p.detach()]
                    for d in dims:
                        for sgn in (1, -1):
                            q = p.detach().clone(); q[:, d] += sgn * hs[d]; Ps.append(q)
                    Pall = torch.cat(Ps)
                    R, t = poses(Pall)
                    a = render_bin(R, t, xy.repeat(len(Ps), 1))
                    L = (((self.blur(a, sg) - Ob.repeat(len(Ps), 1, 1)) ** 2) * V.repeat(len(Ps), 1, 1)).sum((1, 2))
                    L = L.view(len(Ps), B) / norm[None]
                    g = torch.zeros(B, 5, device=self.dev)
                    for k_, d in enumerate(dims):
                        g[:, d] = (L[1 + 2 * k_] - L[2 + 2 * k_]) / (2 * hs[d])
                opt.zero_grad()
                p.grad = g * lr / lr                       # plain gradient; per-parameter step sizes via lr below
                # Adam with per-dimension learning rates: scale the update after the step
                p_before = p.detach().clone()
                opt.step()
                with torch.no_grad():
                    p.copy_(p_before + (p.detach() - p_before) * lr)
        with torch.no_grad():
            R, t = poses(p.detach())
            a = render_bin(R, t, xy) > 0.5
            ob = O > 0.5; vb = V > 0.5
            iou = (a & ob & vb).sum((1, 2)).float() / ((a | ob) & vb).sum((1, 2)).clamp(min=1).float()
        return R.cpu().numpy().astype(float), t.cpu().numpy().astype(float), iou.cpu().numpy().astype(float)

    def iou(self, R, t, obj, valid, xy0, S):
        torch = self.torch
        with torch.no_grad():
            a = self.render(torch.tensor(R, dtype=torch.float32, device=self.dev),
                            torch.tensor(t, dtype=torch.float32, device=self.dev),
                            torch.tensor(xy0, dtype=torch.float32, device=self.dev), S) > 0.5
            a = a.cpu().numpy()
        ob = obj > 0.5; vb = valid > 0.5
        return (a & ob & vb).sum((1, 2)) / np.maximum(((a | ob) & vb).sum((1, 2)), 1), a


# ----------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='/work/inputs')
    ap.add_argument('--out', default='/work/outputs/object_poses_world.npz')
    ap.add_argument('--diag', default=None, help='optional: write diagnostics npz here')
    ap.add_argument('--holdout', type=int, default=-1, help='validation only: ignore mask evidence of frames with i %% 5 == holdout')
    ap.add_argument('--cache', default=None, help='development only: cache of the per-frame fits (not used for the final run)')
    args = ap.parse_args()
    import torch, h5py, trimesh, cv2
    np.random.seed(SEED); torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    inp = args.inputs

    # ---------------- inputs ----------------
    rec = np.load(os.path.join(inp, 'reconstruction.npz'))
    T_base = rec['T_world_mesh'].astype(np.float64)
    F = len(T_base)
    cam = json.load(open(os.path.join(inp, 'camera.json')))
    K = np.array(cam['K'], float); Twc = np.array(cam['T_world_cam'], float); Tcw = np.linalg.inv(Twc)
    plane = np.array(json.load(open(os.path.join(inp, 'ground_plane.json')))['plane'], float)
    n_floor, d_floor = plane[:3], plane[3]          # n.x+d > 0 above the floor (camera side)
    up_w = n_floor / np.linalg.norm(n_floor)
    if (Twc[:3, 3] @ n_floor + d_floor) < 0:
        up_w = -up_w
    up_c = Tcw[:3, :3] @ up_w
    mesh = trimesh.load(os.path.join(inp, 'object_mesh.glb'), force='scene', process=False).to_geometry()
    V = np.asarray(mesh.vertices, float); Fc = np.asarray(mesh.faces)
    Vs = V[::7]
    Tc_base = Tcw[None] @ T_base                       # camera-frame baseline
    z_base = Tc_base[:, 2, 3]
    uv_base = project(K, Tc_base[:, :3, 3])
    hm = h5py.File(os.path.join(inp, 'masks.h5'), 'r')
    Om, Pm = hm['object'], hm['person']
    kp_w = rec['human_keypoints_world'].astype(float)
    kp_c = kp_w @ Tcw[:3, :3].T + Tcw[:3, 3]
    # brightness (HSV value) of the video around the baseline object position: the person's clothes are black,
    # skin is bright -> used to tell hands (possible occluders) from the torso behind a carried bowl
    CB = 384
    vcrop = np.zeros((F, CB, CB), np.uint8); vorg = np.zeros((F, 2), int)
    cap = cv2.VideoCapture(os.path.join(inp, 'video.mp4'))
    for i in range(F):
        ok, im = cap.read()
        assert ok, 'video ended early'
        cx0 = int(np.clip(uv_base[i, 0] - CB / 2, 0, im.shape[1] - CB)); cy0 = int(np.clip(uv_base[i, 1] - CB / 2, 0, im.shape[0] - CB))
        vcrop[i] = im[cy0:cy0 + CB, cx0:cx0 + CB].max(2); vorg[i] = (cx0, cy0)
    cap.release()

    def bright(i, x0, y0, S):
        a, b = x0 - vorg[i, 0], y0 - vorg[i, 1]
        out = np.ones((S, S), bool)
        ys, xs = slice(max(0, b), min(CB, b + S)), slice(max(0, a), min(CB, a + S))
        out[ys.start - b:ys.stop - b, xs.start - a:xs.stop - a] = vcrop[i, ys, xs] >= 45
        return out
    log('loaded', F, 'frames')

    # ---------------- 1. phases from mask bboxes ----------------
    bb = np.zeros((F, 4)); area = np.zeros(F)
    for i in range(F):
        o = Om[i]
        ys, xs = np.nonzero(o)
        area[i] = len(xs)
        bb[i] = (xs.min(), xs.max(), ys.min(), ys.max()) if len(xs) else np.nan
    W = 4
    mscore = np.full(F, np.inf)
    for t in range(F):
        seg = bb[max(0, t - W):t + W + 1]
        if np.isfinite(seg).all():
            mscore[t] = np.median(seg.max(0) - seg.min(0))
    st = mscore <= 1.0                                 # bbox edges (median over the 4) move <= 1 px in 9 frames
    runs, t = [], 0
    while t < F:
        if st[t]:
            s0 = t
            while t < F and st[t]: t += 1
            runs.append([s0, t])
        else:
            t += 1
    runs = [r for r in runs if r[1] - r[0] >= 10]
    merged = []
    for r in runs:                                      # merge runs separated by short gaps at the same place
        if merged and r[0] - merged[-1][1] <= 12 and \
                np.median(np.abs(np.median(bb[r[0]:r[1]], 0) - np.median(bb[merged[-1][0]:merged[-1][1]], 0))) <= 2:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    static = [r for r in merged if r[1] - r[0] >= 25]
    # extend first/last static phase to the episode ends (the object is untouched at the very start/end)
    if static and static[0][0] <= 5: static[0][0] = 0
    if static and static[-1][1] >= F - 5: static[-1][1] = F
    log('static phases', static)
    # By eye (contact sheet): 0-190 bowl on the chair (hands touch it from ~160 but it only lifts at ~191);
    # ~191-670 lifted, carried with both hands, put on the floor; ~670-710 resting on the floor, hand on it;
    # ~712-760 slid under the chair; ~760-869 resting under the chair untouched (hand leaves at ~786).
    phase_id = np.full(F, -1)
    for k, (a, b) in enumerate(static):
        phase_id[a:b] = k
    on_floor = []
    for k, (a, b) in enumerate(static):
        h = np.median([((V @ T_base[i, :3, :3].T + T_base[i, :3, 3]) @ n_floor + d_floor).min() for i in range(a, b, 5)])
        on_floor.append(abs(h) < 0.12)                  # bowl bottom within 12 cm of the floor plane
    log('static on floor', on_floor)

    # ---------------- 2. per-frame silhouette fits ----------------
    rend = SilRenderer(V, Fc, K)
    S = 192
    ker = np.ones((5, 5), np.uint8)
    if args.cache and os.path.exists(args.cache):
        c = np.load(args.cache); fitR, fitt, fitiou, vis, iou_base, xy0s, side_vis = [c[k] for k in ['fitR', 'fitt', 'fitiou', 'vis', 'iou_base', 'xy0s', 'side_vis']]
    else:
        fitR = np.zeros((F, 3, 3)); fitt = np.zeros((F, 3)); fitiou = np.zeros(F); vis = np.zeros(F); side_vis = np.zeros(F)
        iou_base = np.zeros(F); xy0s = np.zeros((F, 2), int)
        crops = {}
        B = 48
        for s0 in range(0, F, B):
            ids = list(range(s0, min(F, s0 + B)))
            xy0 = np.array([[int(np.clip(uv_base[i, 0] - S / 2, 0, 1536 - S)), int(np.clip(uv_base[i, 1] - S / 2, 0, 1152 - S))] for i in ids])
            O_, V_ = [], []
            for k, i in enumerate(ids):
                x0, y0 = xy0[k]
                o = Om[i, y0:y0 + S, x0:x0 + S]; p = Pm[i, y0:y0 + S, x0:x0 + S]
                ign = occluder_region(p, bright(i, x0, y0, S), i, kp_c, z_base[i])
                O_.append(o); V_.append(~ign | o)
                crops[i] = (o, ign)
            O_ = np.array(O_, np.float32); V_ = np.array(V_, np.float32)
            R, tt, iou = rend.fit(Tc_base[ids, :3, :3], uv_base[ids], z_base[ids], O_, V_, xy0, S)
            ib, _ = rend.iou(Tc_base[ids, :3, :3], Tc_base[ids, :3, 3], O_, V_, xy0, S)
            _, a = rend.iou(R, tt, O_, V_, xy0, S)
            for k, i in enumerate(ids):
                o, ign = crops[i]
                hid = ign & ~o
                vis[i] = (a[k] & ~hid).sum() / max(a[k].sum(), 1)   # fraction of the silhouette not hidden by the person
                cols = np.nonzero(a[k].any(0))[0]
                if len(cols):                                       # are the left/right extremes (width -> depth) visible?
                    L = a[k][:, cols[0]:cols[0] + 6]; Rr = a[k][:, cols[-1] - 5:cols[-1] + 1]
                    side_vis[i] = min((L & ~hid[:, cols[0]:cols[0] + 6]).sum() / max(L.sum(), 1),
                                      (Rr & ~hid[:, cols[-1] - 5:cols[-1] + 1]).sum() / max(Rr.sum(), 1))
            fitR[ids] = R; fitt[ids] = tt; fitiou[ids] = iou; iou_base[ids] = ib; xy0s[ids] = xy0
        if args.cache:
            np.savez(args.cache, fitR=fitR, fitt=fitt, fitiou=fitiou, vis=vis, iou_base=iou_base, xy0s=xy0s, side_vis=side_vis)
    log('per-frame fits done; mean IoU baseline %.3f fit %.3f' % (iou_base.mean(), fitiou.mean()))
    uv_fit = project(K, fitt); z_fit = fitt[:, 2]; ax_fit = fitR[:, :, 1]
    # reliability of a per-frame silhouette fit
    w_fit = np.clip((vis - 0.25) / 0.5, 0, 1) * np.clip((fitiou - 0.8) / 0.15, 0, 1)
    # depth from the silhouette needs (nearly) the whole outline: hands covering the rim tips are tolerated
    w_dep = w_fit * np.clip((vis - 0.75) / 0.2, 0, 1)
    held = (np.arange(F) % 5 == args.holdout) if args.holdout >= 0 else np.zeros(F, bool)
    w_fit = w_fit * ~held; w_dep = w_dep * ~held

    # ---------------- 3. depth cues & global correction ----------------
    cues = depth_cues(inp, rec, Tcw, K, V, n_floor, d_floor, T_base, Tc_base, z_fit, w_dep, static, on_floor, kp_c, z_base)
    s_glob = cues['applied_rel']
    log('depth cues', json.dumps({k: v for k, v in cues.items() if k != 'detail'}))

    # ---------------- 4. static phases: pooled fits ----------------
    stat_pose = []
    for k, (a, b) in enumerate(static):
        ids = [i for i in range(a, b, 2) if not held[i]]
        i0 = ids[len(ids) // 2]
        z_t = np.median(z_base[a:b]) * (1 + s_glob)    # depth anchored to the baseline (+ global correction)
        uv0 = np.median(uv_fit[a:b][w_fit[a:b] > 0.5], 0) if (w_fit[a:b] > 0.5).sum() > 5 else np.median(uv_base[a:b], 0)
        x0, y0 = int(uv0[0] - S / 2), int(uv0[1] - S / 2)
        Os, Vs_ = [], []
        for i in ids:
            o = Om[i, y0:y0 + S, x0:x0 + S]; p = Pm[i, y0:y0 + S, x0:x0 + S]
            ign = occluder_region(p, bright(i, x0, y0, S), i, kp_c, z_base[i])
            Os.append(o); Vs_.append(~ign | o)
        Os = np.array(Os, np.float32); Vs_ = np.array(Vs_, np.float32)
        Omean = (Os * Vs_).sum(0, keepdims=True) / np.maximum(Vs_.sum(0, keepdims=True), 1)
        Vmean = (Vs_.mean(0, keepdims=True) > 0.5).astype(np.float32)
        R0 = Tc_base[i0:i0 + 1, :3, :3].copy()
        if on_floor[k]:
            # resting on the floor: the bowl's flat foot sits on the floor, so its axis is the floor normal
            R0[0] = min_rot(R0[0][:, 1], up_c) @ R0[0]
        R, tt, iou = rend.fit(R0, uv0[None], np.array([z_t]), Omean, Vmean, np.array([[x0, y0]]), S,
                              iters=(60, 60, 80), fit_z=False, fit_rot=not on_floor[k])
        stat_pose.append((R[0], tt[0]))
        log('static %d [%d,%d): z %.3f (base med %.3f) iou %.3f tilt %.1f deg' % (k, a, b, tt[0, 2], np.median(z_base[a:b]), iou[0],
            np.degrees(np.arccos(np.clip(R[0][:, 1] @ up_c, -1, 1)))))

    # ---------------- 5. moving frames: fused, smoothed position and axis ----------------
    x = np.arange(F, dtype=float)
    BIG = 50.0
    # rigid hand attachment: image position of the mid-point between the reconstructed hands + smooth offset
    hmid = (kp_c[:, 21:42].mean(1) + kp_c[:, 42:63].mean(1)) / 2
    uv_hand = project(K, hmid)
    moving = phase_id < 0
    off = loclin_smooth(x, uv_fit - uv_hand, w_fit * moving, sigma=15, F=F)
    uv_hand_pred = uv_hand + off
    # baseline depth, corrected by a smooth ratio to the silhouette depth (shape of baseline in gaps)
    ratio = loclin_smooth(x, np.log(z_fit / z_base), w_dep * moving, sigma=20, F=F, c_huber=0.03)[:, 0]
    # measurements
    stat_uv = np.zeros((F, 2)); stat_z = np.zeros(F); stat_ax = np.zeros((F, 3)); is_stat = np.zeros(F, bool)
    for k, (a, b) in enumerate(static):
        R, tt = stat_pose[k]
        stat_uv[a:b] = project(K, tt[None])[0]; stat_z[a:b] = tt[2]; stat_ax[a:b] = R[:, 1]; is_stat[a:b] = True
    w_hand = 0.25 * moving
    w_b = 0.05 * np.ones(F)
    ws = w_fit * moving
    # image position
    X3 = np.concatenate([x, x, x, x])
    uv_s = loclin_smooth(X3, np.concatenate([uv_fit, uv_hand_pred, uv_base, stat_uv]),
                         np.concatenate([ws, w_hand, w_b, BIG * is_stat]), sigma=2.5, F=F, c_huber=6.0)
    # depth (log z)
    zb_corr = np.log(z_base) + ratio
    lz_s = loclin_smooth(np.concatenate([x, x, x]), np.concatenate([np.log(z_fit), zb_corr, np.log(np.where(is_stat, stat_z, 1))]),
                         np.concatenate([w_dep * moving, 0.2 * np.ones(F), BIG * is_stat]), sigma=6, F=F, c_huber=0.04)[:, 0]
    # axis
    ax_s = loclin_smooth(np.concatenate([x, x, x]), np.concatenate([ax_fit, Tc_base[:, :3, 1], stat_ax]),
                         np.concatenate([ws, 0.1 * np.ones(F), BIG * is_stat]), sigma=5, F=F, c_huber=0.15)
    ax_s /= np.linalg.norm(ax_s, axis=1, keepdims=True)
    for k in range(len(static) - 1):
        if on_floor[k] and on_floor[k + 1]:
            # moving between two rests on the floor = sliding on the floor (task text; verified by eye): upright
            ax_s[static[k][1]:static[k + 1][0]] = up_c
    z_s = np.exp(lz_s)
    for k, (a, b) in enumerate(static):            # exact static poses
        uv_s[a:b] = stat_uv[a:b]; z_s[a:b] = stat_z[a:b]; ax_s[a:b] = stat_ax[a:b]

    # ---------------- 6. depth anchoring per phase (moving phases) ----------------
    seg = segments(F, static)
    # frames where the bowl is on the floor: on-floor static phases and slides between two of them
    floor_fr = np.zeros(F, bool)
    for k, (a, b) in enumerate(static):
        if on_floor[k]:
            floor_fr[a:b] = True
            if k + 1 < len(static) and on_floor[k + 1]:
                floor_fr[b:static[k + 1][0]] = True
    if floor_fr.any():
        # on the floor the bowl's foot lies on ONE plane parallel to the released floor plane (resting and sliding):
        # its centre is the viewing ray through the fitted image position intersected with that plane. The plane's
        # offset h is NOT snapped to the released floor; it is chosen so that the median depth over all floor frames
        # equals the baseline's median depth * (1+s) (same anchoring rule as everywhere else).
        n_c = up_c; d_c = (Twc[:3, 3] @ n_floor + d_floor) * np.sign(n_floor @ up_w)   # height of camera centre
        foot = -(V @ np.array([0, 1., 0])).min()                # mesh origin height above its foot when upright
        ray = unproject(K, uv_s[floor_fr], np.ones(floor_fr.sum()))
        target = np.median(z_base[floor_fr]) * (1 + s_glob)

        def zs_for(h):
            lam = (h + foot - d_c) / (ray @ n_c)
            return lam
        lo, hi = -0.5, 0.5
        for _ in range(60):
            h = (lo + hi) / 2
            if np.median(zs_for(h)) > target: lo = h
            else: hi = h
        z_s[floor_fr] = zs_for(h)
        log('floor frames: common foot height above released plane %.1f cm, median z %.3f (base %.3f)' %
            (100 * h, np.median(z_s[floor_fr]), np.median(z_base[floor_fr])))
    for (a, b, kind) in seg:
        if kind == 'moving' and not floor_fr[a:b].all():
            # median depth = baseline median * (1+s) over the segment; then a short linear blend at both ends so the
            # depth is continuous with the (already anchored) neighbouring static poses
            kfac = np.median(z_base[a:b]) * (1 + s_glob) / np.median(z_s[a:b])
            z_s[a:b] *= kfac
            L = min(20, (b - a) // 3); tt_ = np.arange(a, b)
            corr = np.zeros(b - a)
            if a > 0:
                corr += np.log(z_s[a - 1] / z_s[a]) * np.clip(1 - (tt_ - a) / L, 0, 1)
            if b < F:
                corr += np.log(z_s[b] / z_s[b - 1]) * np.clip(1 - (b - 1 - tt_) / L, 0, 1)
            z_s[a:b] *= np.exp(corr)
            log('moving [%d,%d): depth anchor factor %.4f, end blends %.1f / %.1f cm' % (a, b, kfac,
                100 * (z_s[a - 1] - z_s[a] / np.exp(corr[0])) if a > 0 else 0, 100 * (z_s[b] - z_s[b - 1] / np.exp(corr[-1])) if b < F else 0))
    # ---------------- 6b. second pass for carried frames: depth fixed, refit image position and tilt ----------------
    mv = np.nonzero((phase_id < 0) & ~floor_fr)[0]
    if len(mv):
        uv2 = np.zeros((F, 2)); ax2 = np.zeros((F, 3)); iou2 = np.zeros(F); vis2 = np.zeros(F)
        for s0 in range(0, len(mv), 48):
            ids = mv[s0:s0 + 48]
            xy0 = np.array([[int(np.clip(uv_s[i, 0] - S / 2, 0, 1536 - S)), int(np.clip(uv_s[i, 1] - S / 2, 0, 1152 - S))] for i in ids])
            O_, V_, H_ = [], [], []
            for k, i in enumerate(ids):
                x0, y0 = xy0[k]
                o = Om[i, y0:y0 + S, x0:x0 + S]; p = Pm[i, y0:y0 + S, x0:x0 + S]
                ign = occluder_region(p, bright(i, x0, y0, S), i, kp_c, z_s[i])
                O_.append(o); V_.append(~ign | o); H_.append(ign & ~o)
            O_ = np.array(O_, np.float32); V_ = np.array(V_, np.float32)
            R0 = np.array([min_rot(fitR[i][:, 1], ax_s[i]) @ fitR[i] for i in ids])
            R, tt, iou = rend.fit(R0, uv_s[ids], z_s[ids], O_, V_, xy0, S, iters=(20, 20, 30), fit_z=False)
            _, a = rend.iou(R, tt, O_, V_, xy0, S)
            for k, i in enumerate(ids):
                vis2[i] = (a[k] & ~H_[k]).sum() / max(a[k].sum(), 1)
            uv2[ids] = project(K, tt); ax2[ids] = R[:, :, 1]; iou2[ids] = iou
        w2 = np.zeros(F)
        w2[mv] = np.clip((vis2[mv] - 0.1) / 0.4, 0, 1) * np.clip((iou2[mv] - 0.7) / 0.25, 0, 1)
        w2 = w2 * ~held
        prior = np.ones(F) * 0.1
        uv_n = loclin_smooth(np.concatenate([x, x, x]), np.concatenate([uv2, uv_s, stat_uv]),
                             np.concatenate([w2, prior, BIG * is_stat]), sigma=2.5, F=F, c_huber=5.0)
        uv_s[mv] = uv_n[mv]
        ax_n = loclin_smooth(np.concatenate([x, x, x]), np.concatenate([ax2, ax_s, stat_ax]),
                             np.concatenate([w2, prior, BIG * is_stat]), sigma=5, F=F, c_huber=0.15)
        ax_n /= np.linalg.norm(ax_n, axis=1, keepdims=True)
        ax_s[mv] = ax_n[mv]
        log('pass 2 (depth fixed) on %d carried frames: mean IoU %.3f, mean vis %.2f' % (len(mv), iou2[mv].mean(), vis2[mv].mean()))
    t_c = unproject(K, uv_s, z_s)

    # ---------------- 7. rotation: axis + propagated yaw ----------------
    dpsi = body_yaw_delta(kp_w, up_w, static, F)
    # yaw anchor: the baseline's (frame-0 registration) yaw in the first static phase, tilt from the pooled fit
    Rb0 = T_base[static[0][0] if static else 0, :3, :3]
    R0w = min_rot(Rb0[:, 1], Twc[:3, :3] @ stat_pose[0][0][:, 1]) @ Rb0 if static else Rb0
    T_out = np.zeros((F, 4, 4)); T_out[:, 3, 3] = 1
    for i in range(F):
        R1 = rodrigues(up_w, dpsi[i]) @ R0w                  # yaw propagated about the world up axis
        a_w = Twc[:3, :3] @ ax_s[i]
        Rw = min_rot(R1[:, 1], a_w) @ R1                     # tilt so the bowl axis matches the estimate
        T_out[i, :3, :3] = Rw
        T_out[i, :3, 3] = Twc[:3, :3] @ t_c[i] + Twc[:3, 3]
    np.savez(args.out, T_world_mesh=T_out.astype(np.float64))
    log('wrote', args.out, T_out.shape, '| peak torch GPU memory %.2f GB' % (torch.cuda.max_memory_allocated() / 1e9))
    if args.diag:
        np.savez(args.diag, fitR=fitR, fitt=fitt, fitiou=fitiou, iou_base=iou_base, vis=vis, w_fit=w_fit, w_dep=w_dep, side_vis=side_vis,
                 xy0=xy0s, static=np.array(static), dpsi=dpsi, uv_s=uv_s, z_s=z_s, ax_s=ax_s,
                 uv_hand_pred=uv_hand_pred, ratio=ratio, cues=json.dumps(cues))


def occluder_region(p, bright, i, kp_c, z_obj):
    """Pixels of the crop where the person may hide the bowl (ignored by the silhouette loss).
    If the bowl is clearly in front of the torso (carried in front of the body), only the (bright, skin-coloured)
    hands/forearms can occlude it, while the person's black clothes around it are known background (negatives);
    otherwise the whole (dilated) person mask is ignored."""
    import cv2
    pd = cv2.dilate(p.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    z_torso = np.median(kp_c[i, [5, 6, 9, 10], 2])
    if not (z_obj < z_torso - 0.12):
        return pd
    br = cv2.dilate(bright.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return pd & br


def segments(F, static):
    seg, t = [], 0
    for a, b in static:
        if a > t: seg.append((t, a, 'moving'))
        seg.append((a, b, 'static')); t = b
    if t < F: seg.append((t, F, 'moving'))
    return seg


def body_yaw_delta(kp_w, up_w, static, F):
    """Yaw change of the bowl about world up. Zero until the end of the first static phase (bowl untouched or only
    touched), then follows the person's body/hand yaw while carried, frozen once the bowl rests again."""
    e1 = np.cross(up_w, [0, 0, 1.]); e1 /= np.linalg.norm(e1); e2 = np.cross(up_w, e1)

    def yaw(v):
        return np.arctan2(v @ e2, v @ e1)
    sh = np.unwrap(yaw(kp_w[:, 5] - kp_w[:, 6]))          # left - right shoulder
    wr = np.unwrap(yaw(kp_w[:, 62] - kp_w[:, 41]))        # left - right wrist
    wr = wr + 2 * np.pi * np.round(np.median(sh - wr) / (2 * np.pi))
    psi = 0.5 * (sh + wr)
    psi = loclin_smooth(np.arange(F, dtype=float), psi[:, None], np.ones(F), sigma=5, F=F, c_huber=0.3)[:, 0]
    if not static:
        return np.zeros(F)
    t_lift = static[0][1]                                  # first frame after the first static phase
    t_rest = static[1][0] if len(static) > 1 else F        # bowl put down again
    d = np.zeros(F)
    d[t_lift:t_rest] = psi[t_lift:t_rest] - psi[t_lift]
    d[t_rest:] = psi[t_rest - 1] - psi[t_lift]
    return d


def depth_cues(inp, rec, Tcw, K, V, n_floor, d_floor, T_base, Tc_base, z_fit, w_fit, static, on_floor, kp_c, z_base):
    """Independent cues for a relative depth (scale along the viewing rays) correction of the baseline.
    Returns suggested relative shifts (fraction of depth) and the applied one."""
    out = {}
    # (a) reconstructed human vs. the true frame-0 reference human (the baseline object depth was calibrated against
    #     the reconstructed human): ratio of median camera depths at frame 0
    ref = np.load(os.path.join(inp, 'frame0_reference_human.npz'))['vertices_world']
    zr = np.median(ref @ Tcw[2, :3] + Tcw[2, 3])
    zh = np.median(rec['human_vertices_world'][0].astype(float) @ Tcw[2, :3] + Tcw[2, 3])
    out['human_ref_vs_recon'] = zr / zh - 1
    # (b) metric mesh size vs image size: silhouette depth in the static phases (many frames, unoccluded)
    rs = []
    for (a, b) in static:
        m = w_fit[a:b] > 0.7
        if m.sum() > 10:
            rs.append(np.median(z_fit[a:b][m] / z_base[a:b][m]) - 1)
    out['silhouette_static'] = float(np.median(rs)) if rs else np.nan
    # (c) untouched bowl resting on the floor: scale making its lowest point touch the floor plane
    fl = []
    for (a, b), of in zip(static, on_floor):
        if not of:
            continue
        ks = []
        for i in range(a, b, 5):
            P = V[::10] @ Tc_base[i, :3, :3].T
            lo, hi = 0.8, 1.2
            for _ in range(40):
                mid = (lo + hi) / 2
                W = (P + mid * Tc_base[i, :3, 3]) @ np.linalg.inv(Tcw)[:3, :3].T + np.linalg.inv(Tcw)[:3, 3]
                if (W @ n_floor + d_floor).min() > 0: lo = mid
                else: hi = mid
            ks.append(mid - 1)
        fl.append(np.median(ks))
    out['floor_contact'] = float(np.median(fl)) if fl else np.nan
    # (d) hand contact at grasp: reconstructed hand mid-point depth vs. bowl depth at the end of the first static
    #     phase (both hands hold the rim from the side; mid-point ~ bowl axis depth)
    if static:
        b = static[0][1]
        hmid = (kp_c[:, 21:42].mean(1) + kp_c[:, 42:63].mean(1)) / 2
        sl = slice(max(0, b - 20), b + 10)
        out['hand_contact_grasp'] = float(np.median(hmid[sl, 2] / z_base[sl]) - 1)
    vals = {k: v for k, v in out.items() if np.isfinite(v)}
    zref = float(np.median(z_base))
    # decision rule: apply the median of the cues if at least two independent cues agree within 2 cm (at the
    # median baseline depth) and the correction is larger than their spread; otherwise 0
    names = list(vals); best = None
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = abs(vals[names[i]] - vals[names[j]]) * zref
            if d <= 0.02:
                c = 0.5 * (vals[names[i]] + vals[names[j]])
                if abs(c) * zref > d and (best is None or d < best[0]):
                    best = (d, c, names[i], names[j])
    if best is not None:
        agree = [v for v in vals.values() if abs(v - best[1]) * zref <= 0.02]
        applied = float(np.median(agree))
    else:
        applied = 0.0
    out['applied_rel'] = applied
    out['cm_at_median_depth'] = {k: round(v * zref * 100, 2) for k, v in vals.items()}
    out['applied_cm_at_median_depth'] = round(applied * zref * 100, 2)
    return out


if __name__ == '__main__':
    main()
