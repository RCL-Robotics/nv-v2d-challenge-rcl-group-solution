#!/usr/bin/env python
"""Trajectory-level 6-DoF pose recovery of the mug in one monocular HOI video.

Method (see REPORT.md):
  1. Phase segmentation from the SAM2 object mask: the mug is "on the table" when the bottom row of its mask is at
     the table rest level; maximal runs of such frames are REST phases (pose constant), the rest are LIFT phases.
  2. REST phases: the mug stands upright on the table (mesh bottom normal along gravity of the released floor
     plane).  One pose per phase = (azimuth phi of the handle, image position, depth).  phi is chosen from a
     grid by pooled render-and-compare evidence over all frames of the phase (grey-level NCC of the textured
     render vs. image, hand-occluder pixels excluded) plus a prior from the grasping hand (the handle is in the
     hand).  The image position is fitted to the body silhouette (handle pixels ignored, SAM2 misses the dark
     handle).  Depth: all rest phases lie on ONE table plane parallel to the floor; the table height is calibrated
     so that the median rest depth equals the median baseline rest depth (the baseline depth is human/floor
     calibrated, but noisy per phase; the plane makes it consistent across phases).
  3. LIFT phases: the mug moves rigidly with the grasping hand.  Initialise every lift frame by propagating the
     adjacent rest poses with the rigid motion of the 21 hand keypoints (Kabsch), forward from the lift-off pose and
     backward from the set-down pose, blended over time.  Then jointly refine all poses of the lift with a body
     silhouette term, a hand-attachment prior (pose in the hand frame ~ propagated), depth prior and SE(3)
     second-difference smoothness, with the neighbouring rest poses as fixed end points.
Deterministic: fixed seeds, no randomness in the optimisation.
"""
import argparse, json, os, sys, time
import numpy as np, h5py, cv2, torch, torch.nn.functional as Fn, trimesh
import nvdiffrast.torch as dr

torch.manual_seed(0); np.random.seed(0)
torch.backends.cudnn.deterministic = True
DEV = 'cuda'
S_WIN = 128          # render / compare window (full-res pixels)

# ----------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------
class Data:
    def __init__(self, inp):
        cam = json.load(open(os.path.join(inp, 'camera.json')))
        self.K = np.array(cam['K'], float); self.Twc = np.array(cam['T_world_cam'], float); self.Tcw = np.linalg.inv(self.Twc)
        gp = json.load(open(os.path.join(inp, 'ground_plane.json')))
        self.plane = np.array(gp['plane'], float)                    # n.x+d = height above floor (n points up, world y-down)
        self.down_w = -self.plane[:3] / np.linalg.norm(self.plane[:3])
        self.r = np.load(os.path.join(inp, 'reconstruction.npz'))
        self.Tb = self.r['T_world_mesh'].astype(np.float64)
        self.kp = self.r['human_keypoints_world'].astype(np.float64)
        self.F = len(self.Tb)
        self.h5 = h5py.File(os.path.join(inp, 'masks.h5'), 'r')
        om = self.h5['object']
        self.bb = np.zeros((self.F, 4), int)
        self.area = np.zeros(self.F)
        for i in range(self.F):
            m = om[i]; ys, xs = np.nonzero(m)
            self.area[i] = len(xs)
            self.bb[i] = [xs.min(), ys.min(), xs.max(), ys.max()] if len(xs) else [-1, -1, -1, -1]
        # fill frames without a mask (none in this episode, but be safe)
        bad = self.bb[:, 0] < 0
        if bad.any():
            good = np.nonzero(~bad)[0]
            for i in np.nonzero(bad)[0]: self.bb[i] = self.bb[good[np.argmin(np.abs(good - i))]]
        # region of interest of the video that contains the object in every frame
        m = S_WIN
        self.X0 = max(0, self.bb[:, 0].min() - m); self.Y0 = max(0, self.bb[:, 1].min() - m)
        X1 = min(1536, self.bb[:, 2].max() + m); Y1 = min(1152, self.bb[:, 3].max() + m)
        cap = cv2.VideoCapture(os.path.join(inp, 'video.mp4'))
        crops = []
        while True:
            ok, fr = cap.read()
            if not ok: break
            crops.append(fr[self.Y0:Y1, self.X0:X1, ::-1].copy())   # RGB
        assert len(crops) == self.F, (len(crops), self.F)
        self.crops = np.stack(crops)
        self.om_roi = np.stack([om[i, self.Y0:Y1, self.X0:X1] for i in range(self.F)])
        self.pm_roi = np.stack([self.h5['person'][i, self.Y0:Y1, self.X0:X1] for i in range(self.F)])

    def win_center(self, i):
        b = self.bb[i]; return int((b[0] + b[2]) // 2 - S_WIN // 2), int((b[1] + b[3]) // 2 - S_WIN // 2)

    def cam_kp(self, i):
        return (self.Tcw[:3, :3] @ self.kp[i].T).T + self.Tcw[:3, 3]

    def frame(self, i, x0, y0, obj_z):
        """image / object mask / occluder mask (person pixels close to a hand that is in front of the object)"""
        S = S_WIN; xa, ya = x0 - self.X0, y0 - self.Y0
        img = self.crops[i, ya:ya + S, xa:xa + S].astype(np.float32) / 255.
        om = self.om_roi[i, ya:ya + S, xa:xa + S]; pm = self.pm_roi[i, ya:ya + S, xa:xa + S]
        p = self.cam_kp(i); uv = (self.K @ p.T).T; uv = uv[:, :2] / uv[:, 2:]
        near = np.zeros((S, S), np.uint8)
        for j in range(21, 63):
            if p[j, 2] > obj_z + 0.06: continue
            cv2.circle(near, (int(round(uv[j, 0] - x0)), int(round(uv[j, 1] - y0))), 14, 1, -1)
        occ = pm & (near > 0) & ~om
        occ = (cv2.dilate(occ.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~om
        return img, om.astype(np.float32), occ.astype(np.float32)

    def batch(self, idx, x0s, y0s, zs):
        I, M, O = [], [], []
        for i, x0, y0, z in zip(idx, x0s, y0s, zs):
            a, b, c = self.frame(i, x0, y0, z); I.append(a); M.append(b); O.append(c)
        t = lambda a: torch.tensor(np.array(a), dtype=torch.float32, device=DEV)
        return t(I), t(M), t(O)

# ----------------------------------------------------------------------------------------------------------------
# mesh, geometry of the mug
# ----------------------------------------------------------------------------------------------------------------
class Mug:
    def __init__(self, path):
        mesh = trimesh.load(path, force='scene', process=False).to_geometry()
        v = np.asarray(mesh.vertices, np.float64); f = np.asarray(mesh.faces, np.int64)
        self.v_file = v
        # bottom face normal (mesh +z side is the bottom, lid at -z)
        b = v[v[:, 2] > v[:, 2].max() - 0.002]; c = b - b.mean(0); w, U = np.linalg.eigh(c.T @ c)
        self.nb = U[:, 0] * np.sign(U[2, 0])
        # body axis: circle fits to horizontal slices on the handle-free side (done once, constants below)
        self.axc = np.array([-0.0101, 0.0023, 0.012])          # point on the body axis (rotation centre)
        ad = np.array([-0.0093, -0.0339, 0.9994]); ad /= np.linalg.norm(ad)
        fc = v[f].mean(1)
        pa = np.c_[-0.0100 + ad[0] * fc[:, 2] / ad[2], 0.0027 + ad[1] * fc[:, 2] / ad[2]]
        rad = np.hypot(fc[:, 0] - pa[:, 0], fc[:, 1] - pa[:, 1])
        self.handle_faces = (rad > 0.048) & (fc[:, 2] > -0.075) & (fc[:, 2] < 0.04)
        hv = v[np.unique(f[self.handle_faces])]; self.handle_c = hv.mean(0)
        h = self.handle_c - self.axc; h = h - (h @ self.nb) * self.nb; self.hdir = h / np.linalg.norm(h)
        self.bottom = self.axc + (v[:, 2].max() - self.axc[2]) / self.nb[2] * self.nb   # axis point on the bottom plane
        rng = np.random.default_rng(0); self.pts = v[rng.choice(len(v), 3000, replace=False)]
        # renderer buffers (unique positions, uv per corner)
        uq, inv = np.unique(np.round(v, 7), axis=0, return_inverse=True); inv = inv.reshape(-1)
        self.glctx = dr.RasterizeCudaContext()
        self.tv = torch.tensor(uq - self.axc, dtype=torch.float32, device=DEV)   # vertices relative to rotation centre
        self.tf = torch.tensor(inv[f], dtype=torch.int32, device=DEV)
        self.tfuv = torch.tensor(f, dtype=torch.int32, device=DEV)
        self.tuv = torch.tensor(np.asarray(mesh.visual.uv, np.float32), device=DEV)
        tex = np.asarray(mesh.visual.material.baseColorTexture.convert('RGB'), np.float32) / 255.
        self.ttex = torch.tensor(tex[::-1].copy(), device=DEV)[None]
        self.lut = torch.zeros(len(f) + 1, dtype=torch.bool, device=DEV); self.lut[1:] = torch.tensor(self.handle_faces, device=DEV)

    # pose helpers: pose = (R_cam (3x3), p_cam = camera position of the rotation centre axc)
    def T_from(self, R, p):
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p - R @ self.axc; return T
    def Rp_from(self, T):
        return T[:3, :3], T[:3, :3] @ self.axc + T[:3, 3]

    def render(self, R, p, x0, y0, K, S=S_WIN, ss=1, rgb=True):
        """R (B,3,3), p (B,3) torch; x0,y0 (B,) window offsets (full-res px). Returns soft/hard masks (+rgb)."""
        Kx = K.copy()
        if ss > 1:
            Kx[:2] *= ss; Kx[0, 2] = K[0, 2] * ss + (ss - 1) / 2; Kx[1, 2] = K[1, 2] * ss + (ss - 1) / 2
        Sx = S * ss
        pc = torch.einsum('bij,nj->bni', R, self.tv) + p[:, None]
        z = pc[..., 2]
        u = Kx[0, 0] * pc[..., 0] / z + (Kx[0, 2] - x0[:, None] * ss) + 0.5
        v = Kx[1, 1] * pc[..., 1] / z + (Kx[1, 2] - y0[:, None] * ss) + 0.5
        n, f_ = 0.05, 20.
        clip = torch.stack([(2 * u / Sx - 1) * z, (2 * v / Sx - 1) * z, z * (f_ + n) / (f_ - n) - 2 * f_ * n / (f_ - n), z], -1).contiguous()
        rast, _ = dr.rasterize(self.glctx, clip, self.tf, (Sx, Sx))
        tid = rast[..., 3].long()
        mask = (tid > 0).float()[..., None]; hmask = self.lut[tid].float()[..., None]
        ch = [mask, hmask]
        if rgb:
            uvi, _ = dr.interpolate(self.tuv[None].contiguous(), rast, self.tfuv)
            col = dr.texture(self.ttex.expand(R.shape[0], -1, -1, -1).contiguous(), uvi, filter_mode='linear')
            ch = [col * mask] + ch
        out = dr.antialias(torch.cat(ch, -1).contiguous(), rast, clip, self.tf)
        o = {'sil': out[..., -2], 'hsil': out[..., -1], 'hard': mask[..., 0], 'hhard': hmask[..., 0]}
        if rgb: o['rgb'] = out[..., :3]
        if ss > 1:
            for k in list(o):
                x = o[k]
                o[k] = Fn.avg_pool2d(x[:, None], ss)[:, 0] if x.dim() == 3 else Fn.avg_pool2d(x.permute(0, 3, 1, 2), ss).permute(0, 2, 3, 1)
        return o

# ----------------------------------------------------------------------------------------------------------------
# small math helpers
# ----------------------------------------------------------------------------------------------------------------
def so3_exp(w):
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-9); k = w / th
    Kx = torch.zeros(w.shape[:-1] + (3, 3), device=w.device, dtype=w.dtype)
    Kx[..., 0, 1] = -k[..., 2]; Kx[..., 0, 2] = k[..., 1]; Kx[..., 1, 0] = k[..., 2]
    Kx[..., 1, 2] = -k[..., 0]; Kx[..., 2, 0] = -k[..., 1]; Kx[..., 2, 1] = k[..., 0]
    th = th[..., None]; I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(Kx)
    return I + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx

def rot_angle(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2; return np.degrees(np.arccos(np.clip(c, -1, 1)))

def blur(x, s):
    if s <= 0: return x
    k = int(3 * s) * 2 + 1; g = torch.exp(-((torch.arange(k, device=x.device) - k // 2).float() ** 2) / (2 * s * s)); g = g / g.sum()
    x = x[:, None]; x = Fn.conv2d(x, g.view(1, 1, 1, k), padding=(0, k // 2)); x = Fn.conv2d(x, g.view(1, 1, k, 1), padding=(k // 2, 0)); return x[:, 0]

def ncc(a, b, w):
    n = w.sum((1, 2)).clamp_min(1); ma = (a * w).sum((1, 2)) / n; mb = (b * w).sum((1, 2)) / n
    da = (a - ma[:, None, None]) * w; db = (b - mb[:, None, None]) * w
    return (da * db).sum((1, 2)) / torch.sqrt((da * da).sum((1, 2)) * (db * db).sum((1, 2)) + 1e-9)

def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0); H = (A - ca).T @ (B - cb); U, S, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]); R = Vt.T @ D @ U.T
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = cb - R @ ca; return T

def slerp_R(Ra, Rb, w):
    from scipy.spatial.transform import Rotation as Rot
    d = Rot.from_matrix(Ra.T @ Rb).as_rotvec(); return Ra @ Rot.from_rotvec(w * d).as_matrix()

def align_vec(a, b):
    v = np.cross(a, b); s = np.linalg.norm(v); c = a @ b
    if s < 1e-12: return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]); return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)

# ----------------------------------------------------------------------------------------------------------------
# 1. phases
# ----------------------------------------------------------------------------------------------------------------
def segment(D, min_rest=12, shrink=2):
    bot = D.bb[:, 3].astype(float)
    vals, cnt = np.unique(bot, return_counts=True)
    level = vals[np.argmax(cnt)]                          # table rest level (mode of the mask bottom row)
    on = bot >= level - 2.5                               # bottom row within 2.5 px of the rest level -> on the table
    # close 1-2 frame gaps
    on_f = on.copy()
    for i in range(1, D.F - 1):
        if not on[i] and on[i - 1] and on[i + 1]: on_f[i] = True
    runs = []; i = 0
    while i < D.F:
        if on_f[i]:
            j = i
            while j + 1 < D.F and on_f[j + 1]: j += 1
            if j - i + 1 >= min_rest: runs.append([i, j])
            i = j + 1
        else: i += 1
    # shrink rest runs at lift boundaries: lift-off / set-down frames already tilt while touching the table
    rests = []
    for a, b in runs:
        a2 = a + shrink if a > 0 else a; b2 = b - shrink if b < D.F - 1 else b
        rests.append((a2, b2))
    lifts = []
    for k in range(len(rests) - 1):
        lifts.append((rests[k][1] + 1, rests[k + 1][0] - 1))
    if rests[0][0] > 0: lifts.insert(0, (0, rests[0][0] - 1))
    if rests[-1][1] < D.F - 1: lifts.append((rests[-1][1] + 1, D.F - 1))
    return level, rests, lifts

# ----------------------------------------------------------------------------------------------------------------
# 2. rest phases
# ----------------------------------------------------------------------------------------------------------------
def horiz_frame(D):
    d = D.down_w; fwd = D.Twc[:3, 2] - (D.Twc[:3, 2] @ d) * d; fwd /= np.linalg.norm(fwd); right = np.cross(d, fwd)
    return d, fwd, right

def upright_Rw(M, D, phi):
    """world rotation: mesh bottom normal -> gravity, handle direction -> azimuth phi (deg; 0 = away from the camera,
    -90 = image left, +90 = image right, +-180 = towards the camera)"""
    d, fwd, right = horiz_frame(D)
    hw = np.cos(np.radians(phi)) * fwd + np.sin(np.radians(phi)) * right
    a = np.stack([M.nb, M.hdir, np.cross(M.nb, M.hdir)], 1); b = np.stack([d, hw, np.cross(d, hw)], 1)
    return b @ a.T

def phi_of(M, D, Rw):
    d, fwd, right = horiz_frame(D); hw = Rw @ M.hdir; return np.degrees(np.arctan2(hw @ right, hw @ fwd))

def depth_on_table(M, D, Rw, ray_c, h):
    """camera depth z of the rotation centre on pixel ray ray_c (z=1 normalised) s.t. mug bottom is at height h"""
    n = D.plane[:3]; Rcw = D.Twc[:3, :3]; tcw = D.Twc[:3, 3]
    off = Rw @ (M.bottom - M.axc)                               # world offset centre -> bottom
    # height(z) = n.(Rcw (z ray) + tcw + off) + d
    a = n @ (Rcw @ ray_c); b = n @ (tcw + off) + D.plane[3]
    return (h - b) / a

def grasp_hand(D, i, axis_w):
    """centroid of the hand closest to the mug axis (world) and its distance"""
    best = None
    for idx in (range(21, 42), range(42, 63)):
        c = D.kp[i, list(idx)].mean(0); dd = np.linalg.norm(c - axis_w)
        if best is None or dd < best[1]: best = (c, dd)
    return best

def rest_phase_solve(D, M, frames_all, z_guess, PHI, log):
    frames = frames_all[::max(1, len(frames_all) // 24)]
    bb = D.bb[frames]
    x0 = int(np.median((bb[:, 0] + bb[:, 2]) / 2)) - S_WIN // 2; y0 = int(np.median((bb[:, 1] + bb[:, 3]) / 2)) - S_WIN // 2
    # initial centre: baseline median pixel position at depth z_guess
    pcs = np.array([M.Rp_from(D.Tcw @ D.Tb[i])[1] for i in frames]); uv = np.median((D.K @ pcs.T).T[:, :2] / pcs[:, 2:], 0)
    ray = np.linalg.inv(D.K) @ np.array([uv[0], uv[1], 1.]); pc = ray * z_guess
    I, Mk, O = D.batch(frames, [x0] * len(frames), [y0] * len(frames), [z_guess] * len(frames))
    W = 1 - O
    Rw = np.array([upright_Rw(M, D, p) for p in PHI]); Rc = np.einsum('ij,njk->nik', D.Tcw[:3, :3], Rw)
    ss = 2; A = len(PHI)
    Rt = torch.tensor(Rc, dtype=torch.float32, device=DEV); pt = torch.tensor(np.tile(pc, (A, 1)), dtype=torch.float32, device=DEV)
    xs = torch.full((A,), float(x0), device=DEV); ys = torch.full((A,), float(y0), device=DEV)
    # hypotheses rendered once at 2x super-sampling; integer shifts at 2x = half-pixel search
    with torch.no_grad():
        Hs = {k: [] for k in ['rgb', 'hard', 'hhard']}
        # render at 2x without pooling
        Kx = D.K.copy(); Kx[:2] *= ss; Kx[0, 2] = D.K[0, 2] * ss + 0.5; Kx[1, 2] = D.K[1, 2] * ss + 0.5
        for a in range(0, A, 24):
            o = M.render(Rt[a:a + 24], pt[a:a + 24], xs[a:a + 24] * ss, ys[a:a + 24] * ss, Kx, S=S_WIN * ss, ss=1)
            for k in Hs: Hs[k].append(o[k])
        Hs = {k: torch.cat(v) for k, v in Hs.items()}
        body = Hs['hard'] * (1 - Hs['hhard']); hz = Fn.max_pool2d(Hs['hhard'][:, None], 5, 1, 2)[:, 0]
        pool = lambda x: Fn.avg_pool2d(x[:, None], ss)[:, 0] if x.dim() == 3 else Fn.avg_pool2d(x.permute(0, 3, 1, 2), ss).permute(0, 2, 3, 1)
        rng = list(range(-16, 17)); nr = len(rng)
        res = torch.zeros(A, nr, nr, device=DEV)
        for iy, dy in enumerate(rng):
            for ix, dx in enumerate(rng):
                bs = pool(torch.roll(body, (dy, dx), (1, 2))); hzs = pool(torch.roll(hz, (dy, dx), (1, 2)))
                w = W[None] * (1 - hzs[:, None])
                inter = (bs[:, None] * Mk[None] * w).sum((2, 3)); uni = ((bs[:, None] + Mk[None] - bs[:, None] * Mk[None]) * w).sum((2, 3))
                res[:, iy, ix] = (inter / uni).mean(1)
        flat = res.view(A, -1); bi = flat.argmax(1)
        Rg = torch.tensor(rng, device=DEV); dys = Rg[bi // nr]; dxs = Rg[bi % nr]
        nc = []
        for k in range(A):
            rgb = pool(torch.roll(Hs['rgb'][k:k + 1], (int(dys[k]), int(dxs[k])), (1, 2))); hard = pool(torch.roll(Hs['hard'][k:k + 1], (int(dys[k]), int(dxs[k])), (1, 2)))
            nc.append(ncc(I.mean(-1), rgb.mean(-1).expand(len(frames), -1, -1), (hard > 0.5).float() * W).cpu().numpy())
        nc = np.array(nc)                                                   # A x frames
    # hand prior: frames of the phase (and the 8 frames around it) in which a hand grasps the mug
    axis_w = (D.Twc @ np.r_[pc, 1])[:3]
    d, fwd, right = horiz_frame(D)
    hphis = []
    for i in range(max(0, frames_all[0] - 8), min(D.F, frames_all[-1] + 9)):
        c, dist = grasp_hand(D, i, axis_w)
        if dist < 0.12:
            v = c - axis_w; v = v - (v @ d) * d; hphis.append(np.degrees(np.arctan2(v @ right, v @ fwd)))
    ncm = nc.mean(1)
    # circular smoothing of the NCC curve (+-1 grid step)
    ncs = (np.roll(ncm, 2) + 2 * np.roll(ncm, 1) + 3 * ncm + 2 * np.roll(ncm, -1) + np.roll(ncm, -2)) / 9   # +-10 deg
    score = ncs / 0.02                                                     # NCC differences of 0.02 ~ 1 nat
    if len(hphis) >= 5:
        hm = np.degrees(np.arctan2(np.mean(np.sin(np.radians(hphis))), np.mean(np.cos(np.radians(hphis)))))
        dphi = (PHI - hm + 180) % 360 - 180
        score = score - 0.5 * (dphi / 35.) ** 2
    else: hm = np.nan
    k = int(np.argmax(score)); phi = PHI[k]
    o2 = np.argsort(-score)
    log(f'  rest {frames_all[0]}-{frames_all[-1]}: phi={phi:.0f} (ncc-only best {PHI[np.argmax(ncs)]:.0f}, hand {hm:.0f} from {len(hphis)} frames) '
        f'ncc={ncs[k]:.3f} score margin to next mode={score[k]-max([score[j] for j in range(A) if abs(((PHI[j]-phi+180)%360)-180)>40]):.2f}')
    # sub-pixel position: continuous refinement of a shared (du,dv) for the chosen phi (body silhouette)
    R1 = Rt[k:k + 1].expand(len(frames), -1, -1)
    duv = torch.tensor([[float(dxs[k]) / ss, float(dys[k]) / ss]], device=DEV, requires_grad=True)
    opt = torch.optim.Adam([duv], lr=0.05)
    p0 = torch.tensor(pc, dtype=torch.float32, device=DEV)
    xsF = torch.full((len(frames),), float(x0), device=DEV); ysF = torch.full((len(frames),), float(y0), device=DEV)
    for it in range(60):
        s = 1.0 if it < 30 else 0.5
        u = D.K[0, 0] * p0[0] / p0[2] + D.K[0, 2] + duv[0, 0]; v = D.K[1, 1] * p0[1] / p0[2] + D.K[1, 2] + duv[0, 1]
        p = torch.stack([(u - D.K[0, 2]) * p0[2] / D.K[0, 0], (v - D.K[1, 2]) * p0[2] / D.K[1, 1], p0[2]])[None].expand(len(frames), -1)
        o = M.render(R1, p, xsF, ysF, D.K, ss=2, rgb=False)
        Wb = W * (1 - (blur(o['hhard'], 1.5) > 0.02).float())
        loss = ((blur(o['sil'], s) - blur(Mk, s)) ** 2 * Wb).sum() / Mk.sum()
        opt.zero_grad(); loss.backward(); opt.step()
    uvf = np.array([float(u), float(v)])
    with torch.no_grad():
        o = M.render(R1, p, xsF, ysF, D.K, ss=2, rgb=False); h = (o['hard'] > 0.5).float()
        iouF = ((h * Mk * W).sum((1, 2)) / ((h + Mk - h * Mk) * W).sum((1, 2))).mean().item()
    return dict(phi=phi, uv=uvf, Rw=Rw[k], ncc_curve=ncs, score=score, hand_phi=hm, iou=iouF, frames=frames_all)

# ----------------------------------------------------------------------------------------------------------------
# 3. lift phases
# ----------------------------------------------------------------------------------------------------------------
def hand_of(D, i, T_mug_w):
    ax = (T_mug_w @ np.r_[MUG.axc, 1])[:3]
    dR = np.linalg.norm(D.kp[i, 21:42].mean(0) - ax); dL = np.linalg.norm(D.kp[i, 42:63].mean(0) - ax)
    return list(range(21, 42)) if dR <= dL else list(range(42, 63))

def lift_init(D, M, a, b, Tstart, Tend):
    """propagate rest poses through the lift with the rigid motion of the grasping hand"""
    n = b - a + 1
    hs = hand_of(D, a, Tstart); he = hand_of(D, b, Tend)
    fw = []; bw = []
    for i in range(a, b + 1):
        fw.append(kabsch(D.kp[a - 1, hs], D.kp[i, hs]) @ Tstart)
        bw.append(kabsch(D.kp[b + 1, he], D.kp[i, he]) @ Tend)
    out = []
    for k, i in enumerate(range(a, b + 1)):
        w = (k + 1) / (n + 1)
        R = slerp_R(fw[k][:3, :3], bw[k][:3, :3], w)
        pf = fw[k][:3, :3] @ M.axc + fw[k][:3, 3]; pb = bw[k][:3, :3] @ M.axc + bw[k][:3, 3]
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = (1 - w) * pf + w * pb - R @ M.axc
        out.append(T)
    # consistency: forward-propagated end vs set-down rest pose
    Tf_end = kabsch(D.kp[a - 1, hs], D.kp[b + 1, hs]) @ Tstart
    cons = (rot_angle(Tf_end[:3, :3], Tend[:3, :3]), np.linalg.norm(Tf_end[:3, :3] @ M.axc + Tf_end[:3, 3] - (Tend[:3, :3] @ M.axc + Tend[:3, 3])))
    return np.array(out), np.array(fw), np.array(bw), cons

def lift_refine(D, M, a, b, T_init_w, T_prev_w, T_next_w, iters=300, w_sil=10., w_prior_r=0.05, w_prior_t=20., w_smooth=1.0, log=print):
    """joint refinement of all lift frames a..b (camera frame).  T_prev_w / T_next_w: fixed neighbours."""
    n = b - a + 1
    Tc = np.einsum('ij,njk->nik', D.Tcw, T_init_w)
    R0 = torch.tensor(Tc[:, :3, :3], dtype=torch.float32, device=DEV)
    P0 = torch.tensor(np.einsum('nij,j->ni', Tc[:, :3, :3], M.axc) + Tc[:, :3, 3], dtype=torch.float32, device=DEV)
    Tpc = D.Tcw @ T_prev_w; Tnc = D.Tcw @ T_next_w
    Rp = torch.tensor(Tpc[:3, :3], dtype=torch.float32, device=DEV); Pp = torch.tensor(Tpc[:3, :3] @ M.axc + Tpc[:3, 3], dtype=torch.float32, device=DEV)
    Rn = torch.tensor(Tnc[:3, :3], dtype=torch.float32, device=DEV); Pn = torch.tensor(Tnc[:3, :3] @ M.axc + Tnc[:3, 3], dtype=torch.float32, device=DEV)
    idx = list(range(a, b + 1))
    # windows follow the mask
    x0s = []; y0s = []
    for i in idx:
        x0, y0 = D.win_center(i); x0s.append(x0); y0s.append(y0)
    I, Mk, O = D.batch(idx, x0s, y0s, P0[:, 2].cpu().numpy())
    W = 1 - O
    xs = torch.tensor(x0s, dtype=torch.float32, device=DEV); ys = torch.tensor(y0s, dtype=torch.float32, device=DEV)
    w = torch.zeros(n, 3, device=DEV, requires_grad=True); duv = torch.zeros(n, 2, device=DEV, requires_grad=True)
    dz = torch.zeros(n, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([{'params': [w], 'lr': 0.01}, {'params': [duv], 'lr': 0.3}, {'params': [dz], 'lr': 0.002}])
    msum = Mk.sum((1, 2)).clamp_min(50)
    fx, fy, cx, cy = [float(v) for v in (D.K[0, 0], D.K[1, 1], D.K[0, 2], D.K[1, 2])]
    u0 = fx * P0[:, 0] / P0[:, 2] + cx; v0 = fy * P0[:, 1] / P0[:, 2] + cy
    def pose():
        z = P0[:, 2] + dz
        p = torch.stack([(u0 + duv[:, 0] - cx) * z / fx, (v0 + duv[:, 1] - cy) * z / fy, z], -1)
        return so3_exp(w) @ R0, p
    # stage A: per-frame centroid pre-alignment of the body silhouette (robust to large init offsets)
    with torch.no_grad():
        yy, xx = torch.meshgrid(torch.arange(S_WIN, device=DEV, dtype=torch.float32), torch.arange(S_WIN, device=DEV, dtype=torch.float32), indexing='ij')
        for _ in range(4):
            R, p = pose()
            o = M.render(R, p, xs, ys, D.K, rgb=False)
            hz = (blur(o['hhard'], 1.5) > 0.02).float(); Wb = W * (1 - hz)
            br = o['hard'] * Wb; bm = Mk * Wb
            nr = br.sum((1, 2)).clamp_min(1); nm = bm.sum((1, 2)).clamp_min(1)
            d_u = (bm * xx).sum((1, 2)) / nm - (br * xx).sum((1, 2)) / nr
            d_v = (bm * yy).sum((1, 2)) / nm - (br * yy).sum((1, 2)) / nr
            ok = (nm > 100) & (nr > 100)
            duv[:, 0] += torch.where(ok, d_u, torch.zeros_like(d_u)); duv[:, 1] += torch.where(ok, d_v, torch.zeros_like(d_v))
    for it in range(iters):
        s = 4.0 if it < iters * 0.25 else (2.0 if it < iters * 0.5 else (1.0 if it < iters * 0.8 else 0.5))
        R, p = pose()
        o = M.render(R, p, xs, ys, D.K, rgb=False)
        Wb = W * (1 - (blur(o['hhard'], 1.5) > 0.02).float())
        l_sil = ((blur(o['sil'], s) - blur(Mk, s)) ** 2 * Wb).sum((1, 2)) / msum
        # priors to the hand-propagated initialisation: rotation (sigma ~ 25 deg), depth (sigma ~ 4 cm),
        # image position only very weakly (sigma ~ 30 px) - the silhouette determines it
        l_pr = w_prior_r * (w ** 2).sum(1) + w_prior_t * dz ** 2 + 1e-4 * (duv ** 2).sum(1)
        Ra = torch.cat([Rp[None], R, Rn[None]]); Pa = torch.cat([Pp[None], p, Pn[None]])
        l_sm = w_smooth * (((Ra[2:] - 2 * Ra[1:-1] + Ra[:-2]) ** 2).sum((1, 2)) + 400. * ((Pa[2:] - 2 * Pa[1:-1] + Pa[:-2]) ** 2).sum(1))
        loss = w_sil * l_sil.sum() + l_pr.sum() + l_sm.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0 or it == iters - 1:
            log(f'    it {it} sil {l_sil.mean().item():.4f} prior {l_pr.mean().item():.4f} smooth {l_sm.mean().item():.4f}')
    with torch.no_grad():
        R, p = pose()
        # re-orthonormalise
        U, _, Vt = torch.linalg.svd(R); R = U @ Vt
    Rn_ = R.cpu().numpy().astype(np.float64); pn = p.cpu().numpy().astype(np.float64)
    out = np.array([D.Twc @ M.T_from(Rn_[k], pn[k]) for k in range(n)])
    return out


def handle_to_hand(D, M, i, T_w):
    """rotate the pose about the mug body axis so that the handle points to the grasping hand"""
    R = T_w[:3, :3]; ax_w = R @ M.nb; c_w = R @ M.axc + T_w[:3, 3]
    hand = D.kp[i, hand_of(D, i, T_w)].mean(0)
    v = hand - c_w; v = v - (v @ ax_w) * ax_w
    if np.linalg.norm(v) < 1e-6: return T_w
    v /= np.linalg.norm(v); h = R @ M.hdir; h = h - (h @ ax_w) * ax_w; h /= np.linalg.norm(h)
    ang = np.arctan2(np.cross(h, v) @ ax_w, h @ v)
    from scipy.spatial.transform import Rotation as Rot
    Rn = Rot.from_rotvec(ax_w * ang).as_matrix() @ R
    T = np.eye(4); T[:3, :3] = Rn; T[:3, 3] = c_w - Rn @ M.axc; return T

def mirror_tilt(D, M, i, T_w):
    """silhouette-ambiguous twin: reflect the body axis about the plane orthogonal to the viewing ray"""
    Tc = D.Tcw @ T_w; R = Tc[:3, :3]; c = R @ M.axc + Tc[:3, 3]; r = c / np.linalg.norm(c)
    a = R @ M.nb; a2 = a - 2 * (a @ r) * r
    Rn = align_vec(a, a2) @ R
    T = np.eye(4); T[:3, :3] = Rn; T[:3, 3] = c - Rn @ M.axc
    return handle_to_hand(D, M, i, D.Twc @ T)

def frame_scores(D, M, T_w, frames):
    ev = evaluate(D, M, T_w, frames)
    return ev['iou_body'] + 0.5 * ev['ncc']

def viterbi(cost_unary, Ts, M, w_rot=1.0 / 100., w_tr=1.0 / 0.0004):
    """cost_unary (K,n); Ts (K,n,4,4). transition cost: (dR/10deg)^2 + (dt/2cm)^2 between consecutive frames"""
    K, n = cost_unary.shape
    acc = cost_unary[:, 0].copy(); back = np.zeros((K, n), int)
    for t in range(1, n):
        tr = np.zeros((K, K))
        for a in range(K):
            for b in range(K):
                pa = Ts[a, t - 1, :3, :3] @ M.axc + Ts[a, t - 1, :3, 3]; pb = Ts[b, t, :3, :3] @ M.axc + Ts[b, t, :3, 3]
                tr[a, b] = w_rot * rot_angle(Ts[a, t - 1, :3, :3], Ts[b, t, :3, :3]) ** 2 + w_tr * np.sum((pa - pb) ** 2)
        tot = acc[:, None] + tr
        back[:, t] = np.argmin(tot, 0); acc = tot[back[:, t], np.arange(K)] + cost_unary[:, t]
    path = [int(np.argmin(acc))]
    for t in range(n - 1, 0, -1): path.append(back[path[-1], t])
    return np.array(path[::-1])

# ----------------------------------------------------------------------------------------------------------------
# validation metrics (no ground truth)
# ----------------------------------------------------------------------------------------------------------------
def evaluate(D, M, T_w, frames):
    """per-frame body IoU (handle ignored), full IoU, NCC vs. the SAM2 mask / image"""
    res = {'iou_body': [], 'iou_full': [], 'ncc': []}
    for c in range(0, len(frames), 64):
        idx = frames[c:c + 64]
        Tc = np.einsum('ij,njk->nik', D.Tcw, T_w[idx])
        R = torch.tensor(Tc[:, :3, :3], dtype=torch.float32, device=DEV)
        p = torch.tensor(np.einsum('nij,j->ni', Tc[:, :3, :3], M.axc) + Tc[:, :3, 3], dtype=torch.float32, device=DEV)
        x0s, y0s = zip(*[D.win_center(i) for i in idx])
        I, Mk, O = D.batch(idx, x0s, y0s, p[:, 2].cpu().numpy())
        W = 1 - O
        with torch.no_grad():
            o = M.render(R, p, torch.tensor(x0s, dtype=torch.float32, device=DEV), torch.tensor(y0s, dtype=torch.float32, device=DEV), D.K)
            h = (o['hard'] > 0.5).float(); hb = h * (1 - (blur(o['hhard'], 1.5) > 0.02).float())
            Wb = W * (1 - (blur(o['hhard'], 1.5) > 0.02).float())
            res['iou_full'] += ((h * Mk * W).sum((1, 2)) / ((h + Mk - h * Mk) * W).sum((1, 2)).clamp_min(1)).tolist()
            res['iou_body'] += ((hb * Mk * Wb).sum((1, 2)) / ((hb + Mk - hb * Mk) * Wb).sum((1, 2)).clamp_min(1)).tolist()
            res['ncc'] += ncc(I.mean(-1), o['rgb'].mean(-1), h * W).tolist()
    return {k: np.array(v) for k, v in res.items()}

def smoothness(M, T_w):
    p = T_w[:, :3, :3] @ M.axc + T_w[:, :3, 3]
    dr_ = np.array([rot_angle(T_w[i, :3, :3], T_w[i + 1, :3, :3]) for i in range(len(T_w) - 1)])
    dt = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return dr_, dt

# ----------------------------------------------------------------------------------------------------------------
MUG = None
def main():
    global MUG
    ap = argparse.ArgumentParser(); ap.add_argument('--inputs', default='/work/inputs'); ap.add_argument('--out', default='/work/outputs/object_poses_world.npz')
    ap.add_argument('--report_json', default=None)
    args = ap.parse_args()
    t0 = time.time()
    logs = []
    def log(s): print(s, flush=True); logs.append(s)
    D = Data(args.inputs); M = Mug(os.path.join(args.inputs, 'object_mesh.glb')); MUG = M
    log(f'loaded {D.F} frames in {time.time()-t0:.1f}s')
    level, rests, lifts = segment(D)
    log(f'rest level row {level}; rest phases {rests}; lift phases {lifts}')

    # ---- depth model: table plane height so that the median rest depth equals the baseline's median rest depth
    rest_frames = np.concatenate([np.arange(a, b + 1) for a, b in rests])
    zb = np.array([M.Rp_from(D.Tcw @ D.Tb[i])[1][2] for i in rest_frames]); z_med = float(np.median(zb))
    log(f'baseline median rest depth {z_med:.3f} (per phase: ' + ', '.join(f'{np.median([M.Rp_from(D.Tcw@D.Tb[i])[1][2] for i in range(a,b+1)]):.3f}' for a, b in rests) + ')')

    PHI = np.arange(-180, 180, 5.)
    rest_sol = []
    for a, b in rests:
        rest_sol.append(rest_phase_solve(D, M, list(range(a, b + 1)), z_med, PHI, log))
    # table height: for every rest phase, height of the mug bottom when placed at depth z_med on its pixel ray
    hs = []; ws = []
    for rs in rest_sol:
        ray = np.linalg.inv(D.K) @ np.array([rs['uv'][0], rs['uv'][1], 1.])
        pw = (D.Twc @ np.r_[ray * z_med, 1])[:3] + rs['Rw'] @ (M.bottom - M.axc)
        hs.append(D.plane[:3] @ pw + D.plane[3]); ws.append(len(rs['frames']))
    h_table = float(np.average(hs, weights=ws))
    log(f'table height above floor {h_table:.4f} m (per-phase at z_med: {np.round(hs,4).tolist()})')
    T_out = D.Tb.copy()
    for rs in rest_sol:
        ray = np.linalg.inv(D.K) @ np.array([rs['uv'][0], rs['uv'][1], 1.])
        z = depth_on_table(M, D, rs['Rw'], ray, h_table)
        pc = ray * z; Rc = D.Tcw[:3, :3] @ rs['Rw']
        T = D.Twc @ M.T_from(Rc, pc)
        rs['T'] = T; rs['z'] = z
        a, b = rs['frames'][0], rs['frames'][-1]
        T_out[a:b + 1] = T
        log(f'  rest {a}-{b}: phi {rs["phi"]:.0f} uv {rs["uv"].round(2)} depth {z:.3f} (baseline {np.median([M.Rp_from(D.Tcw@D.Tb[i])[1][2] for i in range(a,b+1)]):.3f}) iou {rs["iou"]:.3f}')

    # ---- adaptive rest/lift boundaries: rest frames next to a lift whose static-pose body IoU drops clearly below
    # the phase median (lift-off already started / set-down not finished) are handed to the lift solver
    new_rests = []
    for rs in rest_sol:
        a, b = rs['frames'][0], rs['frames'][-1]
        ev = evaluate(D, M, T_out, list(range(a, b + 1)))['iou_body']; med = np.median(ev); thr = med - 0.05
        a2, b2 = a, b
        if a > 0:
            while a2 < a + 10 and ev[a2 - a] < thr: a2 += 1
        if b < D.F - 1:
            while b2 > b - 10 and ev[b2 - a] < thr: b2 -= 1
        if (a2, b2) != (a, b): log(f'  rest {a}-{b} -> {a2}-{b2} (static-pose IoU below {thr:.3f} at the boundary)')
        T_out[a:a2] = D.Tb[a:a2]; T_out[b2 + 1:b + 1] = D.Tb[b2 + 1:b + 1]
        rs['frames'] = list(range(a2, b2 + 1)); new_rests.append((a2, b2))
    rests = new_rests
    lifts = [(rests[k][1] + 1, rests[k + 1][0] - 1) for k in range(len(rests) - 1)]
    if rests[0][0] > 0: lifts.insert(0, (0, rests[0][0] - 1))
    if rests[-1][1] < D.F - 1: lifts.append((rests[-1][1] + 1, D.F - 1))
    log(f'final rest phases {rests}; lift phases {lifts}')

    # ---- lifts
    rest_of_end = {rs['frames'][-1]: rs for rs in rest_sol}; rest_of_start = {rs['frames'][0]: rs for rs in rest_sol}
    for a, b in lifts:
        prev = rest_of_end.get(a - 1); nxt = rest_of_start.get(b + 1)
        if prev is None or nxt is None:
            log(f'  lift {a}-{b}: no rest neighbour on both sides -> baseline'); continue
        Tinit, fw, bw, cons = lift_init(D, M, a, b, prev['T'], nxt['T'])
        log(f'  lift {a}-{b}: hand-propagated end vs set-down pose: rot {cons[0]:.1f} deg, trans {cons[1]*100:.1f} cm')
        idx = list(range(a, b + 1))
        # candidate initialisations (whole-lift sequences)
        Tbase_fix = np.array([handle_to_hand(D, M, i, D.Tb[i]) for i in idx])
        # baseline depth is noisy per frame: put the baseline candidate at the hand-propagated depth along its own ray
        for k, i in enumerate(idx):
            Rc_, pc_ = M.Rp_from(D.Tcw @ Tbase_fix[k]); zt = M.Rp_from(D.Tcw @ Tinit[k])[1][2]
            Tbase_fix[k] = D.Twc @ M.T_from(Rc_, pc_ / pc_[2] * zt)
        cands = {'hand': Tinit, 'hand_mirror': np.array([mirror_tilt(D, M, i, Tinit[k]) for k, i in enumerate(idx)]),
                 'base': Tbase_fix, 'base_mirror': np.array([mirror_tilt(D, M, i, Tbase_fix[k]) for k, i in enumerate(idx)])}
        names = list(cands); refined = []; costs = []
        for nm in names:
            Tr = lift_refine(D, M, a, b, cands[nm], prev['T'], nxt['T'], log=lambda *x: None)
            sc = frame_scores(D, M, np.concatenate([D.Tb[:a], Tr, D.Tb[b + 1:]]), idx)
            # hand consistency: rotation away from the hand-propagated pose costs a little (sigma 30 deg)
            dh = np.array([rot_angle(Tr[k, :3, :3], Tinit[k, :3, :3]) for k in range(len(idx))])
            refined.append(Tr); costs.append(-sc / 0.03 + 0.5 * (dh / 30.) ** 2)
            log(f'    cand {nm:12s}: mean score {sc.mean():.3f}  mean dev from hand-prop {dh.mean():.1f} deg')
        refined = np.array(refined); costs = np.array(costs)
        path = viterbi(costs, refined, M)
        log(f'    viterbi choice counts: ' + ', '.join(f'{names[k]}={int((path==k).sum())}' for k in range(len(names))))
        Tdec = refined[path, np.arange(len(idx))]
        T_out[a:b + 1] = lift_refine(D, M, a, b, Tdec, prev['T'], nxt['T'], log=log)
    np.savez(args.out, T_world_mesh=T_out.astype(np.float64))
    log(f'saved {args.out}  total time {time.time()-t0:.1f}s')

    # ---- validation table
    ev_b = evaluate(D, M, D.Tb, list(range(D.F))); ev_o = evaluate(D, M, T_out, list(range(D.F)))
    drb, dtb = smoothness(M, D.Tb); dro, dto = smoothness(M, T_out)
    rows = []
    for name, (a, b) in [(f'rest {a}-{b}', (a, b)) for a, b in rests] + [(f'lift {a}-{b}', (a, b)) for a, b in lifts]:
        sl = slice(a, b + 1); sl2 = slice(a, b)
        rows.append([name, ev_b['iou_body'][sl].mean(), ev_o['iou_body'][sl].mean(), ev_b['iou_full'][sl].mean(), ev_o['iou_full'][sl].mean(),
                     ev_b['ncc'][sl].mean(), ev_o['ncc'][sl].mean(), np.median(drb[sl2]) if b > a else 0, np.median(dro[sl2]) if b > a else 0])
    log('phase | IoU_body base/ours | IoU_full base/ours | NCC base/ours | median frame-to-frame rot (deg) base/ours')
    for r in rows: log(f'{r[0]:>12} | {r[1]:.3f} / {r[2]:.3f} | {r[3]:.3f} / {r[4]:.3f} | {r[5]:.3f} / {r[6]:.3f} | {r[7]:.1f} / {r[8]:.1f}')
    log(f'ALL          | {ev_b["iou_body"].mean():.3f} / {ev_o["iou_body"].mean():.3f} | {ev_b["iou_full"].mean():.3f} / {ev_o["iou_full"].mean():.3f} | {ev_b["ncc"].mean():.3f} / {ev_o["ncc"].mean():.3f}')
    if args.report_json:
        json.dump({'logs': logs, 'rests': rests, 'lifts': lifts, 'rest_phi': [r['phi'] for r in rest_sol], 'h_table': h_table,
                   'iou_body_base': ev_b['iou_body'].tolist(), 'iou_body_ours': ev_o['iou_body'].tolist(), 'ncc_base': ev_b['ncc'].tolist(), 'ncc_ours': ev_o['ncc'].tolist(),
                   'iou_full_base': ev_b['iou_full'].tolist(), 'iou_full_ours': ev_o['iou_full'].tolist(),
                   'max_gpu_mem_gb': torch.cuda.max_memory_allocated() / 1e9, 'time_s': time.time() - t0}, open(args.report_json, 'w'))
    log(f'max GPU memory {torch.cuda.max_memory_allocated()/1e9:.2f} GB; total {time.time()-t0:.1f}s')
    open(os.path.join(os.path.dirname(os.path.abspath(args.out)), 'solve_log.txt'), 'w').write('\n'.join(logs) + '\n')

if __name__ == '__main__':
    main()
