#!/usr/bin/env python
"""Trajectory-level 6-DoF pose recovery for one monocular HOI video (squeegee episode).

Pipeline (all estimates recomputed from the inputs; no per-frame results embedded):
  1. per-frame evidence: SAM2 object/person masks (+ orange pixels the masks miss) -> truncated distance maps,
     occlusion map (person), sampled object / orange pixels.
  2. hypotheses per frame: baseline pose x {same end, end-to-end flip} x 8 roll angles about the handle axis
     (+ depth-tilt reflections), each refined by point-based render-and-compare (silhouette chamfer both ways,
     orange-part chamfer with back-face culling, floor penetration, hand proximity, weak depth prior).
  3. appearance score per hypothesis: NCC between the rendered visible-orange coverage of the head bracket
     (nvdiffrast, z-buffered, so the bracket hole / face is visible) and the image's orange score.
  4. static phases (untouched object at start / end, detected from mask stability) -> ONE pose per phase fitted
     on pooled evidence of all its frames.
  5. global decoding: Viterbi over hypotheses (unary = geometric cost + appearance cost; transition = rotation +
     translation change), then a propagation pass (re-register every frame from its decoded neighbours) and a
     second Viterbi.
  6. temporal smoothing + per-frame refinement with a prior towards the smoothed trajectory.
  7. depth re-anchoring: per phase, camera-frame object-centre positions are scaled along their viewing rays so
     that the median camera depth equals the baseline's median depth over the same frames.
"""
import argparse, json, os, sys, time
import numpy as np, cv2, h5py, torch, trimesh
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot
from scipy.ndimage import gaussian_filter1d

dev = 'cuda'
W, H = 1536, 1152
TRUNC = 60  # px truncation of distance maps
SEED = 0

# ----------------------------------------------------------------------------------------------- evidence
def orange_mask(bgr):
    # squeegee orange: hue 3-11, strongly saturated (skin: hue 13-16, sat ~135 -> excluded); checked on frames
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return (hsv[..., 0] >= 3) & (hsv[..., 0] <= 11) & (hsv[..., 1] >= 145) & (hsv[..., 2] >= 85)

def img_orange_score(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return np.clip(1 - np.abs(h - 7.5) / 5, 0, 1) * np.clip((s - 90) / 80, 0, 1) * np.clip((v - 50) / 50, 0, 1)

def frame_features(bgr, om, pm, n_mask=1500, n_or=400, seed=0):
    rng = np.random.default_rng(seed)
    # orange evidence anywhere within ~250 px of the SAM2 object mask (SAM2 often misses the raised head), not on person
    near = cv2.dilate(om.astype(np.uint8), np.ones((501, 501), np.uint8)) > 0 if om.any() else np.ones_like(om)
    o = orange_mask(bgr) & near & ~(pm & ~om)
    o = cv2.morphologyEx(o.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    om = om | (cv2.dilate(o.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    def dt(m):
        if m.sum() == 0: return np.full(m.shape, TRUNC, np.uint8)
        d = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5)
        return np.minimum(d, TRUNC).astype(np.uint8)
    def samp(m, n):
        ys, xs = np.nonzero(m)
        if len(xs) > n:
            k = rng.choice(len(xs), n, replace=False); xs, ys = xs[k], ys[k]
        return np.stack([xs, ys], 1).astype(np.float32)
    occ = pm & ~om
    return dict(dtm=dt(om), dto=dt(o), occ=occ, mpts=samp(om, n_mask), opts=samp(o, n_or), small=om[::4, ::4].copy())

def _feat_worker(a):
    i, bgr, om, pm = a
    return i, frame_features(bgr, om, pm, seed=i)

class FrameBank:
    def __init__(self, inp, nproc=16):
        import multiprocessing as mp
        cap = cv2.VideoCapture(f'{inp}/video.mp4')
        h5 = h5py.File(f'{inp}/masks.h5', 'r')
        F = h5['object'].shape[0]; self.F = F
        self.dtm = np.zeros((F, H, W), np.uint8); self.dto = np.zeros((F, H, W), np.uint8)
        self.occ = np.zeros((F, H, W), bool); self.small = np.zeros((F, H // 4, W // 4), bool)
        self.mpts = [None] * F; self.opts = [None] * F
        def gen():
            for i in range(F):
                ok, bgr = cap.read(); assert ok
                yield i, bgr, h5['object'][i], h5['person'][i]
        with mp.get_context('fork').Pool(nproc) as pool:
            for i, ft in pool.imap(_feat_worker, gen(), chunksize=4):
                self.dtm[i] = ft['dtm']; self.dto[i] = ft['dto']; self.occ[i] = ft['occ']; self.small[i] = ft['small']
                self.mpts[i] = ft['mpts']; self.opts[i] = ft['opts']

def read_frames(inp):
    cap = cv2.VideoCapture(f'{inp}/video.mp4'); out = []
    while True:
        ok, im = cap.read()
        if not ok: break
        out.append(im)
    return out

# ----------------------------------------------------------------------------------------------- geometry
def so3_exp(w):
    th = w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = w / th
    Kx = torch.zeros(w.shape[:-1] + (3, 3), device=w.device, dtype=w.dtype)
    Kx[..., 0, 1] = -k[..., 2]; Kx[..., 0, 2] = k[..., 1]; Kx[..., 1, 0] = k[..., 2]
    Kx[..., 1, 2] = -k[..., 0]; Kx[..., 2, 0] = -k[..., 1]; Kx[..., 2, 1] = k[..., 0]
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(Kx)
    s = torch.sin(th)[..., None]; c = torch.cos(th)[..., None]
    return I + s * Kx + (1 - c) * Kx @ Kx

def rot_angle_deg(R1, R2):
    tr = np.einsum('...ji,...ji->...', R1, R2)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))

def Rz(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])

ENDFLIP = np.diag([1., -1, -1])  # 180 deg about mesh x through the centre: swaps head and handle end

def pad_pts(lst, n):
    out = np.zeros((len(lst), n, 2), np.float32); msk = np.zeros((len(lst), n), np.float32)
    for k, a in enumerate(lst):
        a = a[:n]; out[k, :len(a)] = a; msk[k, :len(a)] = 1
    return torch.tensor(out, device=dev), torch.tensor(msk, device=dev)

# ----------------------------------------------------------------------------------------------- problem
class Problem:
    def __init__(self, inp, bank):
        self.bank = bank
        cam = json.load(open(f'{inp}/camera.json')); rec = np.load(f'{inp}/reconstruction.npz')
        gp = json.load(open(f'{inp}/ground_plane.json'))
        self.K = np.array(cam['K']); self.Twc = np.array(cam['T_world_cam']); self.Tcw = np.linalg.inv(self.Twc)
        self.rec = rec; self.F = rec['T_world_mesh'].shape[0]
        mesh = trimesh.load(f'{inp}/object_mesh.glb', force='scene', process=False).to_geometry()
        self.mesh = mesh
        self.c_mesh = mesh.bounds.mean(0)
        pts, orange, nrm = self.sample_points(mesh)
        rng = np.random.default_rng(SEED)
        self.Kt = torch.tensor(self.K, dtype=torch.float32, device=dev)
        self.P = torch.tensor(pts - self.c_mesh, dtype=torch.float32, device=dev)
        self.Psub = self.P[rng.choice(len(pts), 700, replace=False)]
        self.Por = self.P[torch.tensor(orange, device=dev)]; self.Nor = torch.tensor(nrm[orange], dtype=torch.float32, device=dev)
        hp = np.stack([np.zeros(40), np.zeros(40), np.linspace(-0.40, 1.0, 40)], 1) - self.c_mesh
        self.handle = torch.tensor(hp, dtype=torch.float32, device=dev)
        n_w = np.array(gp['plane'][:3]); d_w = gp['plane'][3]
        self.n_c_np = self.Twc[:3, :3].T @ n_w; self.d_c = float(d_w + n_w @ self.Twc[:3, 3])
        self.n_c = torch.tensor(self.n_c_np, dtype=torch.float32, device=dev)
        kp = rec['human_keypoints_world'].astype(np.float64)
        hands_w = np.stack([kp[:, 21:42].mean(1), kp[:, 42:63].mean(1)], 1)
        self.hands_c = np.einsum('ij,fkj->fki', self.Tcw[:3, :3], hands_w) + self.Tcw[:3, 3]
        self.contact = rec['contact_prob_right_left'].astype(np.float64)
        self.T_base_c = self.Tcw @ rec['T_world_mesh'].astype(np.float64)
        self.Rb, self.tb = self.T_to_Rt(self.T_base_c)

    def sample_points(self, m, n=4000):
        rng = np.random.default_rng(SEED)
        v = np.asarray(m.vertices); f = np.asarray(m.faces); uv = np.asarray(m.visual.uv)
        tex = np.asarray(m.visual.material.baseColorTexture.convert('RGB'))
        area = trimesh.triangles.area(v[f]); fi = rng.choice(len(f), n, p=area / area.sum())
        b = rng.random((n, 2)); s = b.sum(1) > 1; b[s] = 1 - b[s]
        bc = np.stack([1 - b.sum(1), b[:, 0], b[:, 1]], 1)
        pts = np.einsum('nk,nkj->nj', bc, v[f[fi]]); uvs = np.einsum('nk,nkj->nj', bc, uv[f[fi]])
        th, tw = tex.shape[:2]
        px = np.clip((uvs[:, 0] * (tw - 1)).round().astype(int), 0, tw - 1)
        py = np.clip(((1 - uvs[:, 1]) * (th - 1)).round().astype(int), 0, th - 1)
        hsv = cv2.cvtColor(tex[py, px][None].astype(np.uint8), cv2.COLOR_RGB2HSV)[0]
        orange = (hsv[:, 0] >= 3) & (hsv[:, 0] <= 18) & (hsv[:, 1] > 110) & (hsv[:, 2] > 80)
        return pts, orange, np.asarray(m.face_normals)[fi]

    def T_to_Rt(self, T):
        T = np.asarray(T, np.float64); R = T[..., :3, :3]
        return R, np.einsum('...ij,j->...i', R, self.c_mesh) + T[..., :3, 3]
    def Rt_to_T(self, R, t):
        T = np.zeros(R.shape[:-2] + (4, 4)); T[..., :3, :3] = R
        T[..., :3, 3] = t - np.einsum('...ij,j->...i', R, self.c_mesh); T[..., 3, 3] = 1; return T

    def tilt_reflect(self, R, t):
        """reflect the handle axis across the plane perpendicular to the viewing ray (depth-tilt ambiguity)"""
        a = R[:, :, 2]; v = t / np.linalg.norm(t, axis=1, keepdims=True)
        a2 = a - 2 * (a * v).sum(1, keepdims=True) * v
        out = []
        for i in range(len(R)):
            rv = np.cross(a[i], a2[i]); s = np.linalg.norm(rv)
            if s < 1e-8: out.append(R[i]); continue
            out.append(Rot.from_rotvec(rv / s * np.arctan2(s, a[i] @ a2[i])).as_matrix() @ R[i])
        return np.array(out)

    # -- evidence terms; R (B,G,3,3), t (B,G,3) map centred mesh points to camera
    def maps(self, idx):
        b = self.bank
        return torch.stack([torch.tensor(np.asarray(a[idx]), device=dev).half() for a in (b.dtm, b.dto, b.occ)], 1)
    def project(self, R, t, X):
        Xc = torch.einsum('bgij,nj->bgni', R, X) + t[:, :, None]
        z = Xc[..., 2].clamp(min=0.05)
        return torch.stack([self.Kt[0, 0] * Xc[..., 0] / z + self.Kt[0, 2], self.Kt[1, 1] * Xc[..., 1] / z + self.Kt[1, 2]], -1), Xc
    def sample(self, maps, uv):
        B, G, N, _ = uv.shape
        g = torch.stack([uv[..., 0] / (W - 1) * 2 - 1, uv[..., 1] / (H - 1) * 2 - 1], -1).reshape(B, G * N, 1, 2).to(maps.dtype)
        s = torch.nn.functional.grid_sample(maps, g, mode='bilinear', padding_mode='border', align_corners=True)[..., 0].float()
        s = s.reshape(B, -1, G, N).permute(0, 2, 1, 3)
        inside = ((uv[..., 0] >= 0) & (uv[..., 0] <= W - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= H - 1)).float()
        return s, inside
    def costs(self, idx, R, t, ev, zref, sig=10.):
        maps, mp, mpn, op, opn = ev
        rho = lambda d: d ** 2 / (d ** 2 + sig ** 2)
        B, G = R.shape[:2]; ct = {}
        uv, Xc = self.project(R, t, self.P)
        s, ins = self.sample(maps, uv)
        w = ins * (1 - s[:, :, 2])                          # visible & not hidden by the person
        ct['m2i'] = (rho(s[:, :, 0]) * w).sum(-1) / w.sum(-1).clamp(min=1)
        ct['outside'] = 1 - ins.mean(-1)
        uvs, _ = self.project(R, t, self.Psub)
        M = mp.shape[1]
        d = torch.cdist(mp[:, None].expand(B, G, M, 2).reshape(B * G, M, 2), uvs.reshape(B * G, -1, 2)).min(-1).values.reshape(B, G, M)
        ct['i2m'] = (rho(d) * mpn[:, None]).sum(-1) / mpn[:, None].sum(-1).clamp(min=1)
        uvo, Xo = self.project(R, t, self.Por)
        face = ((torch.einsum('bgij,nj->bgni', R, self.Nor) * (-Xo)).sum(-1) > 0).float()
        so, inso = self.sample(maps, uvo)
        wo = face * inso * (1 - so[:, :, 2])
        ct['o2i'] = (rho(so[:, :, 1]) * wo).sum(-1) / wo.sum(-1).clamp(min=1)
        Mo = op.shape[1]
        do = torch.cdist(op[:, None].expand(B, G, Mo, 2).reshape(B * G, Mo, 2), uvo.reshape(B * G, -1, 2)).reshape(B, G, Mo, -1) + (1 - face)[:, :, None] * 1e3
        ct['i2o'] = (rho(do.min(-1).values) * opn[:, None]).sum(-1) / opn[:, None].sum(-1).clamp(min=1)
        h = Xc @ self.n_c + self.d_c                         # height above the released floor (camera frame)
        ct['floor'] = (torch.relu(-h - 0.03) ** 2).mean(-1) * 1e3
        Xh = torch.einsum('bgij,nj->bgni', R, self.handle) + t[:, :, None]
        hc = torch.tensor(self.hands_c[idx], dtype=torch.float32, device=dev)
        dh = torch.cdist(hc[:, None].expand(B, G, 2, 3).reshape(B * G, 2, 3), Xh.reshape(B * G, -1, 3)).min(-1).values.reshape(B, G, 2)
        cp = torch.tensor(self.contact[idx], dtype=torch.float32, device=dev)[:, None]
        ct['hand'] = (cp * (dh ** 2 / (dh ** 2 + 0.15 ** 2))).sum(-1)
        ct['depth'] = ((t[..., 2] - zref) / 0.15) ** 2
        return ct
    def evidence(self, idx):
        idx = np.asarray(idx)
        mp, mpn = pad_pts([self.bank.mpts[i] for i in idx], 600)
        op, opn = pad_pts([self.bank.opts[i] for i in idx], 300)
        return self.maps(idx), mp, mpn, op, opn

W_DEF = dict(m2i=0.5, i2m=1.0, o2i=0.5, i2o=0.5, outside=0.5, floor=1.0, hand=0.3, depth=0.02)
def total(ct, w=W_DEF):
    return sum(w.get(k, 0) * v for k, v in ct.items())

def optimize(prob, idx, R0, t0, zref, steps=80, sig_sched=(20., 8.), lr=(0.03, 0.02), prior=None):
    """batched local refinement. R0 (B,G,3,3), t0 (B,G,3). prior=(Rp (B,3,3), tp (B,3), w) pulls to a trajectory"""
    idx = np.asarray(idx); ev = prob.evidence(idx)
    R0t = torch.tensor(R0, dtype=torch.float32, device=dev); t0t = torch.tensor(t0, dtype=torch.float32, device=dev)
    zr = torch.tensor(zref, dtype=torch.float32, device=dev)
    om = torch.zeros(R0t.shape[:2] + (3,), device=dev, requires_grad=True)
    dt = torch.zeros(R0t.shape[:2] + (3,), device=dev, requires_grad=True)
    opt = torch.optim.Adam([{'params': [om], 'lr': lr[0]}, {'params': [dt], 'lr': lr[1]}])
    if prior is not None:
        Rp = torch.tensor(prior[0], dtype=torch.float32, device=dev)[:, None]; tp = torch.tensor(prior[1], dtype=torch.float32, device=dev)[:, None]
    def loss_fn(R, t, sig):
        ct = prob.costs(idx, R, t, ev, zr, sig=sig); L = total(ct)
        if prior is not None:
            cosang = ((torch.einsum('bgji,bgji->bg', R, Rp.expand_as(R)) - 1) / 2).clamp(-1, 1)
            ct['prior'] = (1 - cosang) / (1 - np.cos(np.radians(5))) + ((t - tp) ** 2).sum(-1) / 0.03 ** 2
            L = L + prior[2] * ct['prior']
        return L, ct
    for k in range(steps):
        sig = sig_sched[0] * (sig_sched[1] / sig_sched[0]) ** (k / max(1, steps - 1))
        R = so3_exp(om) @ R0t; t = t0t + dt
        L, _ = loss_fn(R, t, sig)
        opt.zero_grad(); L.sum().backward(); opt.step()
    with torch.no_grad():
        R = so3_exp(om) @ R0t; t = t0t + dt
        ct = prob.costs(idx, R, t, ev, zr, sig=sig_sched[1]); L = total(ct)
    return R.cpu().numpy().astype(np.float64), t.cpu().numpy().astype(np.float64), L.cpu().numpy(), {k: v.cpu().numpy() for k, v in ct.items()}

def evaluate(prob, idx, R, t, zref, sig=8.):
    idx = np.asarray(idx)
    with torch.no_grad():
        ct = prob.costs(idx, torch.tensor(R, dtype=torch.float32, device=dev), torch.tensor(t, dtype=torch.float32, device=dev),
                        prob.evidence(idx), torch.tensor(zref, dtype=torch.float32, device=dev), sig=sig)
        return total(ct).cpu().numpy(), {k: v.cpu().numpy() for k, v in ct.items()}

def run_batched(prob, frames, R0, t0, zref, bs=16, prior=None, **kw):
    out = [optimize(prob, frames[s:s + bs], R0[s:s + bs], t0[s:s + bs], zref[s:s + bs],
                    prior=None if prior is None else (prior[0][s:s + bs], prior[1][s:s + bs], prior[2]), **kw)
           for s in range(0, len(frames), bs)]
    return (np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out]), np.concatenate([o[2] for o in out]),
            {k: np.concatenate([o[3][k] for o in out]) for k in out[0][3]})

# ----------------------------------------------------------------------------------------------- appearance
class OrangeRenderer:
    """renders the visible orange coverage of the textured mesh (z-buffered) in small crops"""
    SS = 2
    def __init__(self, prob):
        m = prob.mesh
        self.v = torch.tensor(np.asarray(m.vertices, np.float32), device=dev); self.f = torch.tensor(np.asarray(m.faces, np.int32), device=dev)
        uv = np.asarray(m.visual.uv, np.float32).copy(); uv[:, 1] = 1 - uv[:, 1]; self.uv = torch.tensor(uv, device=dev)
        tex = np.asarray(m.visual.material.baseColorTexture.convert('RGB'))
        hsv = cv2.cvtColor(tex, cv2.COLOR_RGB2HSV)
        o = ((hsv[..., 0] >= 3) & (hsv[..., 0] <= 13) & (hsv[..., 1] >= 120)).astype(np.float32)
        self.otex = torch.tensor(o[..., None], device=dev)[None].contiguous()
        self.glctx = dr.RasterizeCudaContext(); self.K = prob.K
    def render_crops(self, T, x0, y0, s):
        B = len(T); SS = self.SS; S = s * SS; K = self.K
        fx, fy, cx, cy = K[0, 0] * SS, K[1, 1] * SS, K[0, 2] * SS, K[1, 2] * SS
        n, fa = 0.05, 20.
        P = torch.zeros(B, 4, 4, device=dev)
        P[:, 0, 0] = 2 * fx / S; P[:, 0, 2] = torch.tensor(2 * (cx - x0 * SS) / S - 1, dtype=torch.float32, device=dev)
        P[:, 1, 1] = 2 * fy / S; P[:, 1, 2] = torch.tensor(2 * (cy - y0 * SS) / S - 1, dtype=torch.float32, device=dev)
        P[:, 2, 2] = (fa + n) / (fa - n); P[:, 2, 3] = -2 * fa * n / (fa - n); P[:, 3, 2] = 1
        Tt = torch.tensor(T, dtype=torch.float32, device=dev)
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        clip = torch.einsum('bij,bjk,nk->bni', P, Tt, vh) * torch.tensor([1, -1, 1, 1], device=dev, dtype=torch.float32)
        rast, _ = dr.rasterize(self.glctx, clip.contiguous(), self.f, (S, S))
        rast = torch.flip(rast, [1])
        texc, _ = dr.interpolate(self.uv[None], rast, self.f)
        col = dr.texture(self.otex, texc, filter_mode='linear')[..., 0] * (rast[..., 3] > 0)
        return torch.nn.functional.avg_pool2d(col[:, None], SS)[:, 0]
    def ncc(self, Ts, im, min_orange=10.):
        """per hypothesis NCC(rendered orange coverage, image orange score) in a window around the head bracket"""
        Ts = np.asarray(Ts); out = np.full(len(Ts), np.nan); K = self.K
        p = np.einsum('bij,j->bi', Ts[:, :3, :3], np.array([0, 0, -0.475])) + Ts[:, :3, 3]
        ok = p[:, 2] > 0.3
        if not ok.any(): return out
        half = int(np.clip(0.12 * K[0, 0] / np.median(p[ok, 2]), 12, 80)); s = 2 * half
        u = (K @ p.T).T; c = u[:, :2] / np.maximum(u[:, 2:], 1e-3)
        x0 = (c[:, 0] - half).astype(int); y0 = (c[:, 1] - half).astype(int)
        ok &= (x0 >= 0) & (y0 >= 0) & (x0 + s <= W) & (y0 + s <= H)
        idx = np.nonzero(ok)[0]
        if len(idx) == 0: return out
        so = np.stack([img_orange_score(im[y0[k]:y0[k] + s, x0[k]:x0[k] + s]) for k in idx])
        r = self.render_crops(Ts[idx], x0[idx], y0[idx], s).cpu().numpy()
        for j, k in enumerate(idx):
            if so[j].sum() < min_orange: continue
            a = r[j] - r[j].mean(); b = so[j] - so[j].mean()
            out[k] = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))
        return out

def ncc_all(prob, orr, frames_img, R, t, frames):
    N = np.full(R.shape[:2], np.nan)
    for k, i in enumerate(frames):
        N[k] = orr.ncc(prob.Rt_to_T(R[k], t[k]), frames_img[i])
    return N

# ----------------------------------------------------------------------------------------------- decoding
ALPHA, BETA, NCC_MISSING = 20., 8., 0.6   # unary = ALPHA*geometric + BETA*(1-NCC); NCC unavailable -> 1-NCC := 0.6
TH0, TC0 = 8., 0.06                          # transition: deg / 8 + metres / 0.06

def unary(L, N):
    return ALPHA * L + BETA * np.where(np.isnan(N), NCC_MISSING, 1 - N)

def viterbi(R, t, U):
    F, G = U.shape
    acc = U[0].copy(); bp = np.zeros((F, G), int)
    for f in range(1, F):
        Tc = rot_angle_deg(R[f - 1][:, None], R[f][None, :]) / TH0 + np.linalg.norm(t[f - 1][:, None] - t[f][None, :], axis=-1) / TC0
        tot = acc[:, None] + Tc
        bp[f] = tot.argmin(0); acc = tot.min(0) + U[f]
    path = np.zeros(F, int); path[-1] = acc.argmin()
    for f in range(F - 1, 0, -1): path[f - 1] = bp[f, path[f]]
    return path

def static_phases(bank, thr=0.75):
    """untouched phases at the start / end: object mask overlaps the first / last frame's mask with IoU > thr"""
    F = bank.F; m0 = bank.small[0]; m1 = bank.small[F - 1]
    iou = lambda a, b: (a & b).sum() / max(1, (a | b).sum())
    a = 0
    while a < F and iou(bank.small[a], m0) > thr: a += 1
    b = F - 1
    while b > 0 and iou(bank.small[b], m1) > thr: b -= 1
    return (0, a), (b + 1, F)

def fit_static(prob, fr, R0s, t0s, zref, steps=120):
    """one pose explaining all frames fr (pooled evidence), from several initialisations"""
    fr = np.asarray(fr); ev = prob.evidence(fr); B = len(fr); G = len(R0s)
    R0t = torch.tensor(np.asarray(R0s), dtype=torch.float32, device=dev); t0t = torch.tensor(np.asarray(t0s), dtype=torch.float32, device=dev)
    zr = torch.tensor(zref, dtype=torch.float32, device=dev)
    om = torch.zeros(G, 3, device=dev, requires_grad=True); dt = torch.zeros(G, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{'params': [om], 'lr': 0.02}, {'params': [dt], 'lr': 0.01}])
    for k in range(steps + 1):
        sig = 20. * (8. / 20.) ** (min(k, steps) / steps)
        R = so3_exp(om) @ R0t; t = t0t + dt
        ct = prob.costs(fr, R[None].expand(B, G, 3, 3), t[None].expand(B, G, 3), ev, zr, sig=sig)
        L = total(ct).mean(0)
        if k == steps: break
        opt.zero_grad(); L.sum().backward(); opt.step()
    return R.detach().cpu().numpy().astype(np.float64), t.detach().cpu().numpy().astype(np.float64), L.detach().cpu().numpy()

def smooth_traj(R, t, seg, sigma=1.5):
    """Gaussian smoothing of positions and rotations (quaternion, sign-aligned) inside segment [a,b)"""
    a, b = seg; R = R.copy(); t = t.copy()
    if b - a < 3: return R, t
    q = Rot.from_matrix(R[a:b]).as_quat()
    for k in range(1, len(q)):
        if q[k] @ q[k - 1] < 0: q[k] = -q[k]
    qs = gaussian_filter1d(q, sigma, axis=0, mode='nearest'); qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    R[a:b] = Rot.from_quat(qs).as_matrix(); t[a:b] = gaussian_filter1d(t[a:b], sigma, axis=0, mode='nearest')
    return R, t

# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='/work/inputs'); ap.add_argument('--out', default='/work/outputs/object_poses_world.npz')
    ap.add_argument('--diag', default=None, help='optional: save intermediate diagnostics npz here')
    ap.add_argument('--cache', default=None, help='development only: reuse stage-1 hypotheses from this npz')
    args = ap.parse_args()
    np.random.seed(SEED); torch.manual_seed(SEED)
    T0 = time.time(); log = lambda *a: print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)

    bank = FrameBank(args.inputs); log('features')
    imgs = read_frames(args.inputs)
    prob = Problem(args.inputs, bank); F = prob.F; Rb, tb = prob.Rb, prob.tb
    orr = OrangeRenderer(prob)
    allf = np.arange(F)

    # ---- 1. hypotheses: {same end, end flip} x 8 rolls about the handle axis (+ tilt reflections of even rolls)
    R0 = []; t0 = []
    for e in [np.eye(3), ENDFLIP]:
        for k in range(8):
            R = Rb @ e @ Rz(k * np.pi / 4); R0.append(R); t0.append(tb)
            if k % 2 == 0: R0.append(prob.tilt_reflect(R, tb)); t0.append(tb)
    R0 = np.stack(R0, 1); t0 = np.stack(t0, 1); zref = tb[:, 2:3]
    if args.cache and os.path.exists(args.cache):
        c = np.load(args.cache); Rc, tc, Lc, Nc = c['Rc'], c['tc'], c['Lc'], c['Nc']; log('stage 1 loaded from dev cache')
    else:
        Rc, tc, Lc, _ = run_batched(prob, allf, R0, t0, zref); log('hypotheses refined', Rc.shape)
        Nc = ncc_all(prob, orr, imgs, Rc, tc, allf); log('appearance scored')
        if args.cache: np.savez(args.cache, Rc=Rc, tc=tc, Lc=Lc, Nc=Nc)

    # ---- 2. static phases: one pose per phase from pooled evidence
    phases = static_phases(bank); log('static phases', phases)
    static_pose = []
    for (a, b) in phases:
        if b - a < 10: static_pose.append(None); continue
        sub = np.linspace(a, b - 1, min(24, b - a)).round().astype(int)
        U = unary(Lc[a:b], Nc[a:b])
        # initialisations: per hypothesis, the candidate at the phase frame where it scores best
        inits = []
        for g in np.argsort(np.median(U, 0))[:6]:
            f = a + U[:, g].argmin(); inits.append((Rc[f, g], tc[f, g]))
        Rs, ts, Ls = fit_static(prob, sub, [x[0] for x in inits], [x[1] for x in inits], np.full((len(sub), 1), np.median(tb[a:b, 2])))
        Ns = np.array([np.nanmean(ncc_all(prob, orr, imgs, np.repeat(Rs[None, g:g + 1], len(sub), 0), np.repeat(ts[None, g:g + 1], len(sub), 0), sub))
                       for g in range(len(Rs))])
        score = ALPHA * Ls + BETA * np.where(np.isnan(Ns), NCC_MISSING, 1 - Ns)
        g = score.argmin(); static_pose.append((Rs[g], ts[g]))
        log(f'static phase {a}-{b}: geo {np.round(Ls, 3)} ncc {np.round(Ns, 3)} score {np.round(score, 2)} '
            f'angle-to-baseline {np.round([np.median(rot_angle_deg(Rs[k][None], Rb[a:b])) for k in range(len(Rs))], 0)} -> pick {g}')
    is_static = np.zeros(F, bool)
    for (a, b), sp in zip(phases, static_pose):
        if sp is not None: is_static[a:b] = True

    def add_static(R, t, U):
        R = np.concatenate([R, R[:, :1]], 1); t = np.concatenate([t, t[:, :1]], 1); U = np.concatenate([U, np.full((F, 1), 1e6)], 1)
        for (a, b), sp in zip(phases, static_pose):
            if sp is None: continue
            R[a:b, -1] = sp[0]; t[a:b, -1] = sp[1]; U[a:b, :-1] = 1e6; U[a:b, -1] = 0
        return R, t, U

    # ---- 3. global decoding
    Ra, ta, Ua = add_static(Rc, tc, unary(Lc, Nc))
    path = viterbi(Ra, ta, Ua)
    Rp = Ra[allf, path]; tp = ta[allf, path]; log('viterbi 1')

    # ---- 4. propagation: re-register every moving frame from its decoded neighbours, decode again
    mv = np.nonzero(~is_static)[0]
    offs = [-3, -1, 1, 3]
    R0p = np.stack([Rp[np.clip(mv + o, 0, F - 1)] for o in offs], 1); t0p = np.stack([tp[np.clip(mv + o, 0, F - 1)] for o in offs], 1)
    Rq, tq, Lq, _ = run_batched(prob, mv, R0p, t0p, tb[mv, 2:3], steps=60, sig_sched=(10., 6.))
    Nq = ncc_all(prob, orr, imgs, Rq, tq, mv)
    G2 = Ra.shape[1] + len(offs)
    Rb2 = np.concatenate([Ra, np.repeat(Ra[:, :1], len(offs), 1)], 1); tb2 = np.concatenate([ta, np.repeat(ta[:, :1], len(offs), 1)], 1)
    Ub2 = np.concatenate([Ua, np.full((F, len(offs)), 1e6)], 1)
    Rb2[mv, -len(offs):] = Rq; tb2[mv, -len(offs):] = tq; Ub2[mv, -len(offs):] = unary(Lq, Nq)
    path2 = viterbi(Rb2, tb2, Ub2)
    Rp = Rb2[allf, path2]; tp = tb2[allf, path2]; log('propagation + viterbi 2')

    # ---- 5. smoothing + refinement with a trajectory prior (moving frames only)
    segs = []; a = None
    for f in range(F + 1):
        if f < F and not is_static[f]:
            a = f if a is None else a
        elif a is not None: segs.append((a, f)); a = None
    Rs_, ts_ = Rp.copy(), tp.copy()
    for sg in segs: Rs_, ts_ = smooth_traj(Rs_, ts_, sg, sigma=2.0)
    Rr, tr, Lr, _ = run_batched(prob, mv, Rs_[mv][:, None], ts_[mv][:, None], tb[mv, 2:3], steps=60, sig_sched=(8., 6.), lr=(0.01, 0.005),
                                prior=(Rs_[mv], ts_[mv], 0.05))
    Rf, tf = Rp.copy(), tp.copy(); Rf[mv] = Rr[:, 0]; tf[mv] = tr[:, 0]
    for sg in segs: Rf, tf = smooth_traj(Rf, tf, sg, sigma=1.0)
    log('refined + smoothed')

    # ---- 6. depth re-anchoring: camera-frame object-centre positions are scaled along their viewing rays so that the
    # median camera depth equals the baseline's median over the same frames. Anchor units: the static phases, and the
    # runs of the moving phase where our orientation agrees with the baseline's (< 45 deg; there the baseline's
    # per-frame depth correction refers to the same physical configuration). Where the baseline orientation is wrong
    # (end-to-end flipped or lost), its depth is not a usable reference: the scale is interpolated in time between
    # the neighbouring anchor units, which keeps the trajectory continuous with them.
    agree = rot_angle_deg(Rf, Rb) < 45.
    agree = np.round(gaussian_filter1d(agree.astype(float), 7)).astype(bool)   # remove flicker
    units = [(a, b) for (a, b), sp in zip(phases, static_pose) if sp is not None]
    for (a0, b0) in segs:
        f = a0
        while f < b0:
            g = f
            while g < b0 and agree[g] == agree[f]: g += 1
            if agree[f] and g - f >= 30: units.append((f, g))
            f = g
    units.sort()
    scale = np.full(F, np.nan); depth_scale = {}
    for (a, b) in units:
        s = np.median(tb[a:b, 2]) / np.median(tf[a:b, 2]); scale[a:b] = s; depth_scale[f'{a}-{b}'] = float(s)
    known = np.nonzero(~np.isnan(scale))[0]
    scale = np.interp(np.arange(F), known, scale[known])
    sm = gaussian_filter1d(scale, 3.0)
    for (a, b) in units:            # keep the anchor units exact (static phases stay constant)
        sm[a:b] = scale[a:b] if (a, b) in [p for p in phases] else sm[a:b]
    tf = tf * sm[:, None]
    log('depth anchoring units/scales', depth_scale)

    T_cam = prob.Rt_to_T(Rf, tf)
    T_world = np.einsum('ij,fjk->fik', prob.Twc, T_cam)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, T_world_mesh=T_world.astype(np.float64))
    log('saved', args.out, T_world.shape)
    if args.diag:
        np.savez(args.diag, Rc=Rc, tc=tc, Lc=Lc, Nc=Nc, path=path, path2=path2, Rp=Rp, tp=tp, Rf=Rf, tf=tf, is_static=is_static,
                 phases=np.array(phases), segs=np.array(segs), depth_scale=json.dumps(depth_scale), scale=sm)

if __name__ == '__main__':
    main()
