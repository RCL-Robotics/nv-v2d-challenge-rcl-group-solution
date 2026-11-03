#!/usr/bin/env python
"""Trajectory-level 6-DoF re-estimation of the object in one HOI video.

Pipeline (see REPORT.md for the reasoning and the validation):
  1. phase segmentation from the video (object-region frame differences, person pixels excluded)
  2. static phases: ONE pose per phase from a global render-and-compare search (silhouette + texture NCC,
     textured mesh rendered with nvdiffrast) pooled over several frames of the phase
  3. moving phases: SAM2 mask repair, forward / backward render-and-compare tracking from the static anchors plus
     the baseline pose and its 3 box-symmetry flips as hypotheses, Viterbi decoding over time,
     interpolation over frames without image evidence, light temporal smoothing
  4. depth re-anchoring: camera-frame positions are scaled along the viewing rays so that every static phase has
     the baseline's median camera depth (moving phases: factor interpolated between neighbouring static phases)
Deterministic: fixed seeds, fixed hypothesis sets; run OFFLINE.
"""
import argparse, json, os, sys, time
import numpy as np, torch, cv2, h5py, trimesh
import torch.nn.functional as Fnn
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot, Slerp
from scipy.ndimage import median_filter

dev = 'cuda'
torch.manual_seed(0); np.random.seed(0)
torch.backends.cudnn.deterministic = True

# ----------------------------------------------------------------------------- constants
MOTION_LAG = 3            # frame lag of the object-region difference signal
MOTION_STATIC_THR = 3.0   # mean abs grey-level difference below which the object is considered static
MIN_STATIC_LEN = 20       # shortest static phase (frames)
STATIC_N_FRAMES = 4       # frames pooled per static phase in the global search
CROP_OUT = 160            # render-and-compare crop resolution (px)
SIL_UNRELIABLE = 0.45
RAMP = 10                 # frames over which the moving-phase depth correction ramps in     # silhouette loss above which a moving-phase frame has no usable image evidence


# ----------------------------------------------------------------------------- geometry helpers
def rotvec_to_mat(w):
    th = w.norm(dim=-1, keepdim=True).clamp(min=1e-8); k = w / th
    Kx = torch.zeros(w.shape[:-1] + (3, 3), device=w.device, dtype=w.dtype)
    Kx[..., 0, 1] = -k[..., 2]; Kx[..., 0, 2] = k[..., 1]; Kx[..., 1, 0] = k[..., 2]
    Kx[..., 1, 2] = -k[..., 0]; Kx[..., 2, 0] = -k[..., 1]; Kx[..., 2, 1] = k[..., 0]
    th = th[..., None]; I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(Kx)
    return I + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


def ortho(R):
    U, _, Vt = np.linalg.svd(R); D = np.diag([1, 1, np.linalg.det(U @ Vt)]); return U @ D @ Vt


def T44(R, t):
    T = np.eye(4); T[:3, :3] = ortho(R); T[:3, 3] = t; return T


def rdist(A, B):
    return np.degrees(Rot.from_matrix(ortho(A[:3, :3].T @ B[:3, :3])).magnitude())


# 180-degree flips about the three box axes (the box's silhouette symmetries)
FLIPS = np.stack([np.eye(3), np.diag([1., -1, -1]), np.diag([-1., 1, -1]), np.diag([-1., -1, 1])])


# ----------------------------------------------------------------------------- renderer / losses
class RC:
    def __init__(self, mesh, tex_scale=0.5):
        self.glctx = dr.RasterizeCudaContext()
        self.v = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=dev)
        self.f = torch.tensor(np.asarray(mesh.faces), dtype=torch.int32, device=dev)
        uv = np.asarray(mesh.visual.uv).astype(np.float32).copy(); uv[:, 1] = 1 - uv[:, 1]
        self.uv = torch.tensor(uv, device=dev)
        tex = np.asarray(mesh.visual.material.baseColorTexture.convert('RGB'))
        tex = cv2.resize(tex, None, fx=tex_scale, fy=tex_scale, interpolation=cv2.INTER_AREA)
        self.tex = torch.tensor(tex[..., ::-1].copy(), dtype=torch.float32, device=dev)[None] / 255.

    def render(self, R, t, Kc, w, h, color=True):
        pc = torch.einsum('bij,nj->bni', R, self.v) + t[:, None]
        Z = pc[..., 2].clamp(min=1e-3)
        u = Kc[0, 0] * pc[..., 0] / Z + Kc[0, 2]; vv = Kc[1, 1] * pc[..., 1] / Z + Kc[1, 2]
        nx = 2 * (u + 0.5) / w - 1; ny = 2 * (vv + 0.5) / h - 1
        n_, f_ = 0.05, 30.
        zc = Z * (f_ + n_) / (f_ - n_) - 2 * f_ * n_ / (f_ - n_)
        cl = torch.stack([nx * Z, ny * Z, zc, Z], -1).contiguous()
        rast, _ = dr.rasterize(self.glctx, cl, self.f, (h, w))
        hard = (rast[..., 3:4] > 0).float()
        alpha = dr.antialias(hard, rast, cl, self.f)[..., 0]
        out = {'alpha': alpha, 'hard': hard[..., 0]}
        if color:
            uvi, _ = dr.interpolate(self.uv[None], rast, self.f)
            out['color'] = dr.texture(self.tex, uvi, filter_mode='linear') * hard
        return out


def crop_params(cx, cy, size, out):
    sc = out / size; x0 = cx - size / 2; y0 = cy - size / 2
    return np.array([[sc, 0, (-x0 + 0.5) * sc - 0.5], [0, sc, (-y0 + 0.5) * sc - 0.5]])


def crop_K(K, M):
    Kc = K.copy().astype(np.float64); Kc[0, 0] *= M[0, 0]; Kc[1, 1] *= M[1, 1]
    Kc[0, 2] = K[0, 2] * M[0, 0] + M[0, 2]; Kc[1, 2] = K[1, 2] * M[1, 1] + M[1, 2]; return Kc


def gblur(x, s):
    k = int(2 * round(3 * s) + 1); g = torch.exp(-(torch.arange(k, device=x.device) - k // 2) ** 2 / (2 * s * s)); g = g / g.sum()
    C = x.shape[1]
    x = Fnn.conv2d(Fnn.pad(x, (k // 2, k // 2, 0, 0), mode='replicate'), g.view(1, 1, 1, k).repeat(C, 1, 1, 1), groups=C)
    return Fnn.conv2d(Fnn.pad(x, (0, 0, k // 2, k // 2), mode='replicate'), g.view(1, 1, k, 1).repeat(C, 1, 1, 1), groups=C)


class FrameObs:
    """square crop of one frame: image, object mask, valid mask (pixels not covered by the person)"""
    def __init__(self, img, om, pm, K, cx, cy, size, out=CROP_OUT, w_fp=1.0):
        M = crop_params(cx, cy, size, out); self.out = out; self.w_fp = w_fp
        self.Kc = torch.tensor(crop_K(K, M), dtype=torch.float32, device=dev)
        wa = lambda a, fl=cv2.INTER_LINEAR: cv2.warpAffine(a, M, (out, out), flags=fl, borderMode=cv2.BORDER_CONSTANT)
        im = wa(img).astype(np.float32) / 255.
        o = wa(om.astype(np.uint8) * 255).astype(np.float32) / 255.
        p = cv2.dilate(wa((pm & ~om).astype(np.uint8) * 255).astype(np.float32) / 255., np.ones((5, 5)))
        self.img = torch.tensor(im, device=dev).permute(2, 0, 1)[None]
        self.om = torch.tensor(o, device=dev)[None]; self.valid = torch.tensor(1 - p, device=dev)[None]


def losses(rc, obs, R, t, blur=1.0, w_tex=1.0):
    o = rc.render(R, t, obs.Kc, obs.out, obs.out)
    a = o['alpha']; v = obs.valid; m = obs.om
    sb = max(blur, 0.5) * 2
    ab = gblur(a[:, None], sb)[:, 0]; mb = gblur(m[:, None], sb)[:, 0]
    inter = (ab * mb * v).sum((1, 2)); uni = (mb * v).sum((1, 2)) + obs.w_fp * (ab * (1 - mb) * v).sum((1, 2))
    l_sil = 1 - inter / uni.clamp(min=1)
    col = o['color'].permute(0, 3, 1, 2); hard = o['hard'][:, None]
    reg = hard * v[:, None]
    rb = gblur(col, blur) / gblur(hard, blur).clamp(min=1e-3)
    ib = gblur(obs.img, blur).expand_as(rb)
    n = reg.sum((2, 3)).clamp(min=1)

    def stdz(x):
        mu = (x * reg).sum((2, 3)) / n; sd = (((x - mu[..., None, None]) ** 2 * reg).sum((2, 3)) / n).clamp(min=1e-6).sqrt()
        return (x - mu[..., None, None]) / sd[..., None, None]
    l_tex = 1 - ((stdz(rb) * stdz(ib) * reg).sum((2, 3)) / n).mean(1)
    return l_sil + w_tex * l_tex, l_sil, l_tex


def optimize(rc, obs_list, R0, t0, iters=150, lr_r=0.02, lr_t=0.0005, blurs=(3.0, 2.0, 1.0), w_tex=0.0,
             z_ref=None, w_z=0.0, w_rot=0.0):
    """Adam on (rotation vector, relative translation) of B hypotheses; one pose shared by all obs."""
    B = R0.shape[0]
    w = torch.zeros(B, 3, device=dev, requires_grad=True); d = torch.zeros(B, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{'params': [w], 'lr': lr_r}, {'params': [d], 'lr': lr_t}])
    per = iters // len(blurs)
    for it in range(iters):
        bl = blurs[min(it // per, len(blurs) - 1)]
        R = rotvec_to_mat(w) @ R0; t = t0 + d * t0[:, 2:3]
        L = 0
        if z_ref is not None: L = L + w_z * ((t[:, 2] - z_ref) / 0.05) ** 2
        if w_rot > 0: L = L + w_rot * (w ** 2).sum(1) / np.radians(10) ** 2
        wt = w_tex * min(1.0, it / (0.3 * iters))
        for ob in obs_list: L = L + losses(rc, ob, R, t, blur=bl, w_tex=wt)[0]
        opt.zero_grad(); L.sum().backward(); opt.step()
    with torch.no_grad():
        return rotvec_to_mat(w) @ R0, t0 + d * t0[:, 2:3]


def score(rc, obs_list, R, t, blur=1.0):
    R = torch.as_tensor(R if torch.is_tensor(R) else np.asarray(R), dtype=torch.float32, device=dev)
    t = torch.as_tensor(t if torch.is_tensor(t) else np.asarray(t), dtype=torch.float32, device=dev)
    outs = []
    with torch.no_grad():
        for i in range(0, len(R), 64):
            acc = 0
            for ob in obs_list: acc = acc + torch.stack(losses(rc, ob, R[i:i + 64], t[i:i + 64], blur=blur), 1)
            outs.append(acc / len(obs_list))
    return torch.cat(outs).cpu().numpy()   # columns: total, silhouette, texture


def global_search(rc, obs_list, t_center, n=2000, topk=16, iters=120, extra_R=None):
    Rs = Rot.random(n, random_state=0).as_matrix()
    if extra_R is not None: Rs = np.concatenate([extra_R, Rs])
    tt = np.broadcast_to(t_center, (len(Rs), 3))
    sc = score(rc, obs_list, Rs, tt, blur=3.0)[:, 1]          # coarse: silhouette only
    sel = []
    for i in np.argsort(sc):
        if all(rdist(Rs[i], Rs[j]) > 15 for j in sel): sel.append(i)
        if len(sel) >= topk: break
    R, t = optimize(rc, obs_list, torch.tensor(Rs[sel], dtype=torch.float32, device=dev),
                    torch.tensor(tt[sel].copy(), dtype=torch.float32, device=dev), iters=iters)
    R = R.cpu().numpy(); t = t.cpu().numpy()
    Rf = np.concatenate([R @ F for F in FLIPS]); tf = np.concatenate([t] * 4)   # texture decides among symmetric poses
    s = score(rc, obs_list, Rf, tf)
    o = np.argsort(s[:, 0])
    return Rf[o], tf[o], s[o]


# ----------------------------------------------------------------------------- data
class Data:
    def __init__(self, inp):
        c = json.load(open(f'{inp}/camera.json'))
        self.K = np.array(c['K']); self.Twc = np.array(c['T_world_cam'])
        rec = np.load(f'{inp}/reconstruction.npz')
        self.Tw_base = rec['T_world_mesh'].astype(np.float64)
        self.Tc_base = np.linalg.inv(self.Twc)[None] @ self.Tw_base
        self.mesh = trimesh.load(f'{inp}/object_mesh.glb', force='scene', process=False).to_geometry()
        self.diam = float(np.linalg.norm(self.mesh.extents))
        hf = h5py.File(f'{inp}/masks.h5', 'r')
        self.om = hf['object'][:]; self.pm = hf['person'][:]
        cap = cv2.VideoCapture(f'{inp}/video.mp4'); fr = []
        while True:
            ok, im = cap.read()
            if not ok: break
            fr.append(im)
        self.frames = np.stack(fr); self.F = len(self.frames)
        assert self.F == len(self.Tw_base) == len(self.om)


# ----------------------------------------------------------------------------- 1. phases
def motion_signal(D):
    F = D.F; d = np.full(F, np.nan)
    small = lambda a, fl: cv2.resize(a, (768, 576), interpolation=fl)
    gr = [small(cv2.cvtColor(D.frames[t], cv2.COLOR_BGR2GRAY), cv2.INTER_AREA).astype(np.float32) for t in range(F)]
    om = [small(D.om[t].astype(np.uint8), cv2.INTER_NEAREST) > 0 for t in range(F)]
    pm = [small(D.pm[t].astype(np.uint8), cv2.INTER_NEAREST) > 0 for t in range(F)]
    k = MOTION_LAG
    for t in range(k, F):
        m = (om[t] | om[t - k]) & ~pm[t] & ~pm[t - k]
        m = cv2.dilate(m.astype(np.uint8), np.ones((5, 5))) > 0
        m &= ~(cv2.dilate((pm[t] | pm[t - k]).astype(np.uint8), np.ones((7, 7))) > 0)
        if m.sum() >= 50: d[t] = np.abs(gr[t] - gr[t - k])[m].mean()
    return d


def segment_phases(d, F):
    """static phases = maximal runs where the (lag-compensated) motion signal stays below threshold."""
    dd = np.where(np.isnan(d), np.inf, d)      # object invisible (fully covered by the person) -> moving
    dd[:MOTION_LAG] = 0.0                       # no lagged difference available for the first frames
    moving = np.zeros(F, bool)
    for t in range(F):
        if dd[t] > MOTION_STATIC_THR: moving[max(t - MOTION_LAG, 0):t + 1] = True   # a change at t involves t-k..t
    moving[0] = moving[0] and moving[1]
    segs = []; t = 0
    while t < F:
        s = t
        while t < F and moving[t] == moving[s]: t += 1
        segs.append([s, t - 1, 'moving' if moving[s] else 'static'])
    # short static runs inside motion are treated as motion (pauses); merge neighbours
    for sgm in segs:
        if sgm[2] == 'static' and sgm[1] - sgm[0] + 1 < MIN_STATIC_LEN: sgm[2] = 'moving'
    out = []
    for sgm in segs:
        if out and out[-1][2] == sgm[2]: out[-1][1] = sgm[1]
        else: out.append(list(sgm))
    return out


# ----------------------------------------------------------------------------- 2. static phases
def make_obs(D, t, Tpred, om=None, w_fp=1.0):
    om = D.om[t] if om is None else om
    ys, xs = np.nonzero(om)
    p = D.K @ Tpred[:3, 3]; pc = p[:2] / p[2]; dpx = D.K[0, 0] * D.diam / Tpred[2, 3]
    if len(xs) > 30:
        cx = 0.5 * (xs.min() + xs.max()); cy = 0.5 * (ys.min() + ys.max()); ext = max(np.ptp(xs), np.ptp(ys))
        if np.hypot(cx - pc[0], cy - pc[1]) > dpx: cx, cy = pc
        size = max(ext * 1.6, dpx * 1.2)
    else:
        cx, cy = pc; size = dpx * 1.5
    return FrameObs(D.frames[t], om, D.pm[t], D.K, cx, cy, size, w_fp=w_fp)


def solve_static(D, rc, a, b):
    fs = np.arange(a, b + 1)
    area = D.om[fs].sum((1, 2)).astype(float)
    occ = []
    for i, t in enumerate(fs):
        y0, y1, x0, x1 = bbox(D.om[t])
        occ.append(D.pm[t][max(0, y0 - 10):y1 + 10, max(0, x0 - 10):x1 + 10].mean() if area[i] > 0 else 1.0)
    occ = np.array(occ)
    bad = occ + 10 * (area < 0.8 * np.median(area))
    good = fs[np.argsort(bad, kind='stable')[:max(STATIC_N_FRAMES, len(fs) // 4)]]
    sel = np.sort(good)[np.linspace(0, len(good) - 1, STATIC_N_FRAMES).astype(int)]
    obs = []
    for t in sel:
        ys, xs = np.nonzero(D.om[t])
        obs.append(FrameObs(D.frames[t], D.om[t], D.pm[t], D.K, xs.mean(), ys.mean(), max(np.ptp(xs), np.ptp(ys)) * 1.6, out=192))
    zc = np.median(D.Tc_base[sel, 2, 3])
    ys, xs = np.nonzero(D.om[sel[0]]); ray = np.linalg.inv(D.K) @ [xs.mean(), ys.mean(), 1]
    R, t, S = global_search(rc, obs, ray / ray[2] * zc, extra_R=D.Tc_base[sel[0], :3, :3][None] @ FLIPS)
    return T44(R[0], t[0]), S[0], sel, (R, t, S)


def bbox(m):
    ys, xs = np.nonzero(m)
    if len(xs) == 0: return (0, 0, 0, 0)
    return ys.min(), ys.max(), xs.min(), xs.max()


# ----------------------------------------------------------------------------- 3. moving phases
def build_sam2():
    import hydra
    from hydra import compose
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    d = '/weights/sam2'
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(config_dir=d, job_name="sam2", version_base=None):
        cfg = compose(config_name='sam2.1_hiera_l.yaml')
    OmegaConf.resolve(cfg); model = instantiate(cfg.model, _recursive_=True)
    model.load_state_dict(torch.load(f'{d}/sam2.1_hiera_large.pt', map_location='cpu', weights_only=False)['model'], strict=True)
    return SAM2ImagePredictor(model.cuda().eval())


def repair_mask(pred, im, om, z, f):
    """SAM2 re-segmentation prompted by the existing (often partial) object mask; accepted only if the area is
    plausible for the metric box at depth z, it contains the old mask and is compact."""
    ys, xs = np.nonzero(om)
    if len(xs) < 20: return om, False
    cx, cy = np.median(xs), np.median(ys); s = max(np.ptp(xs), np.ptp(ys), 20)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        pred.set_image(np.ascontiguousarray(im[..., ::-1]))
        box = np.array([cx - s * 1.2, cy - s * 1.2, cx + s * 1.2, cy + s * 1.2])
        masks, scores, _ = pred.predict(point_coords=np.array([[cx, cy]]), point_labels=np.array([1]),
                                        box=box[None], multimask_output=True)
    k2 = (f / z) ** 2
    for k in np.argsort(-scores, kind='stable'):
        m = masks[k] > 0; a = m.sum()
        if a < 0.05 * k2 or a > 0.22 * k2: continue          # projected area of a 34x20x40 cm box: 0.07-0.2 m^2
        if (m & om).sum() < 0.7 * om.sum(): continue
        yy, xx = np.nonzero(m)
        if a / ((np.ptp(xx) + 1) * (np.ptp(yy) + 1)) < 0.35: continue
        return m | om, True
    return om, False


OPT = dict(iters=30, blurs=(2.0, 1.0), w_tex=0.2, w_z=1.0, w_rot=0.02)


def opt_frame(D, rc, obs, inits, zr):
    R0 = torch.tensor(np.stack([T[:3, :3] for T in inits]), dtype=torch.float32, device=dev)
    t0 = torch.tensor(np.stack([T[:3, 3] for T in inits]), dtype=torch.float32, device=dev)
    R, tt = optimize(rc, [obs], R0, t0, z_ref=torch.tensor(zr, dtype=torch.float32, device=dev), **OPT)
    S = score(rc, [obs], R, tt)
    return [T44(R[k].cpu().numpy(), tt[k].cpu().numpy()) for k in range(len(inits))], S


def chain(D, rc, frames, Tstart, zr, masks):
    out = {}; prev = Tstart; prev2 = None
    for t in frames:
        obs = make_obs(D, t, prev, masks[t], w_fp=0.6); inits = [prev]
        if prev2 is not None:                     # constant-velocity prediction as a second hypothesis
            dR = prev[:3, :3] @ prev2[:3, :3].T; inits.append(T44(dR @ prev[:3, :3], 2 * prev[:3, 3] - prev2[:3, 3]))
        Ts, S = opt_frame(D, rc, obs, inits, zr[t]); k = int(np.argmin(S[:, 0])); Tn = Ts[k]
        if not np.isfinite(Tn).all(): Tn = prev.copy()
        out[t] = (Tn, S[k]); prev2 = prev; prev = Tn
    return out


def base_cands(D, rc, frames, zr, masks):
    out = {}
    for t in frames:
        Tb = D.Tc_base[t].copy(); Tb[:3, 3] *= zr[t] / Tb[2, 3]
        obs = make_obs(D, t, Tb, masks[t], w_fp=0.6)
        Ts, S = opt_frame(D, rc, obs, [T44(Tb[:3, :3] @ F, Tb[:3, 3]) for F in FLIPS], zr[t])
        out[t] = list(zip(Ts, S))
    return out


def viterbi(cands, unary, Tstart, Tend, w_u=4.0, s_r=10.0, s_t=0.05):
    def pair(A, B): return rdist(A, B) / s_r + np.linalg.norm(A[:3, 3] - B[:3, 3]) / s_t
    n = len(cands)
    cost = [np.array([pair(Tstart, c) for c in cands[0]]) + w_u * unary[0]]; back = []
    for t in range(1, n):
        P = np.array([[pair(p, c) for p in cands[t - 1]] for c in cands[t]])
        tot = P + cost[-1][None]; bi = tot.argmin(1); back.append(bi)
        cost.append(tot[np.arange(len(bi)), bi] + w_u * unary[t])
    k = int((cost[-1] + np.array([pair(c, Tend) for c in cands[-1]])).argmin()); path = [k]
    for t in range(n - 2, -1, -1): k = int(back[t][k]); path.append(k)
    return path[::-1]


def interp_pose(T0, T1, w):
    R = Slerp([0, 1], Rot.from_matrix(np.stack([ortho(T0[:3, :3]), ortho(T1[:3, :3])])))([w]).as_matrix()[0]
    return T44(R, (1 - w) * T0[:3, 3] + w * T1[:3, 3])


def smooth_poses(Ts, sigma=1.5):
    """Gaussian smoothing of a pose sequence (rotations via local tangent-space averaging)."""
    n = len(Ts); out = []
    r = int(3 * sigma)
    for i in range(n):
        idx = np.arange(max(0, i - r), min(n, i + r + 1)); w = np.exp(-(idx - i) ** 2 / (2 * sigma ** 2)); w /= w.sum()
        Ri = Ts[i][:3, :3]
        rv = np.array([Rot.from_matrix(ortho(Ri.T @ Ts[j][:3, :3])).as_rotvec() for j in idx])
        R = Ri @ Rot.from_rotvec((w[:, None] * rv).sum(0)).as_matrix()
        out.append(T44(R, (w[:, None] * np.array([Ts[j][:3, 3] for j in idx])).sum(0)))
    return out


def solve_moving(D, rc, a, b, Ts, Te, masks, log):
    zb = median_filter(D.Tc_base[:, 2, 3], size=15, mode='nearest')
    ta = max(a - 1, 0); tb = min(b + 1, D.F - 1)
    s0 = Ts[2, 3] / zb[ta]; s1 = Te[2, 3] / zb[tb]
    zr = {t: zb[t] * ((1 - (t - ta) / max(tb - ta, 1)) * s0 + (t - ta) / max(tb - ta, 1) * s1) for t in range(a, b + 1)}
    fw = chain(D, rc, range(a, b + 1), Ts, zr, masks)
    bw = chain(D, rc, range(b, a - 1, -1), Te, zr, masks)
    bc = base_cands(D, rc, range(a, b + 1), zr, masks)
    cands = []; un = []; sil = []
    for t in range(a, b + 1):
        cs = [fw[t], bw[t]] + bc[t]
        cands.append([c[0] for c in cs]); un.append(np.array([c[1][0] for c in cs])); sil.append(np.array([c[1][1] for c in cs]))
    path = viterbi(cands, un, Ts, Te)
    sel = [cands[i][k] for i, k in enumerate(path)]
    rel = np.array([sil[i][k] < SIL_UNRELIABLE for i, k in enumerate(path)])
    log['moving'][f'{a}-{b}'] = dict(path=''.join('fb0123'[k] for k in path), reliable=int(rel.sum()), n=len(path),
                                      unary=[float(un[i][k]) for i, k in enumerate(path)])
    # frames without usable evidence: interpolate between reliable neighbours (anchors at both ends)
    keys = [a - 1] + [a + i for i in range(len(sel)) if rel[i]] + [b + 1]
    vals = [Ts] + [sel[i] for i in range(len(sel)) if rel[i]] + [Te]
    out = list(sel)
    for i in range(len(sel)):
        if rel[i]: continue
        t = a + i; j = np.searchsorted(keys, t)
        out[i] = interp_pose(vals[j - 1], vals[j], (t - keys[j - 1]) / (keys[j] - keys[j - 1]))
    out = smooth_poses([Ts] + out + [Te])[1:-1]
    return out, dict(fw=fw, bw=bw, bc=bc, path=path, rel=rel)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='/work/inputs'); ap.add_argument('--out', default='/work/outputs/object_poses_world.npz')
    ap.add_argument('--debug_dump', default=None, help='optional: also dump intermediate results (development only)')
    ap.add_argument('--dev_cache', default=None, help='development only: cache static poses / repaired masks here')
    args = ap.parse_args()
    T0 = time.time(); log = {'moving': {}}
    D = Data(args.inputs); F = D.F
    print(f'loaded {F} frames in {time.time() - T0:.0f}s', flush=True)
    mesh = D.mesh; rc = RC(mesh)

    # 1. phases
    d = motion_signal(D)
    phases = segment_phases(d, F)
    print('phases:', phases, flush=True)
    log['phases'] = phases

    # 2. static phases
    Tc = np.array(D.Tc_base)              # fallback = baseline
    static_T = {}
    for (a, b, kind) in phases:
        if kind != 'static': continue
        cf = f'{args.dev_cache}/static_{a}_{b}.npy' if args.dev_cache else None
        if cf and os.path.exists(cf): T, S, sel = np.load(cf, allow_pickle=True)
        else:
            T, S, sel, _ = solve_static(D, rc, a, b)
            if cf: os.makedirs(args.dev_cache, exist_ok=True); np.save(cf, np.array([T, S, sel], dtype=object), allow_pickle=True)
        Tb = D.Tc_base[sel[0]]
        static_T[(a, b)] = T
        print(f'static {a}-{b}: frames {sel.tolist()} loss {S.round(3)} rot vs baseline {rdist(T, Tb):.1f} deg  ({time.time() - T0:.0f}s)', flush=True)
        log[f'static_{a}_{b}'] = dict(frames=sel.tolist(), loss=S.tolist(), rot_vs_base=rdist(T, Tb))
        Tc[a:b + 1] = T

    # 3. moving phases (anchored on the neighbouring static poses)
    pred = build_sam2() if not args.dev_cache else None
    if args.dev_cache and not all(os.path.exists(f'{args.dev_cache}/masks_{a}_{b}.npy') for (a, b, k) in phases if k == 'moving'):
        pred = build_sam2()
    dump = {}
    for i, (a, b, kind) in enumerate(phases):
        if kind != 'moving': continue
        prevS = [p for p in phases[:i] if p[2] == 'static']; nextS = [p for p in phases[i + 1:] if p[2] == 'static']
        Ts = static_T[tuple(prevS[-1][:2])] if prevS else D.Tc_base[a]
        Te = static_T[tuple(nextS[0][:2])] if nextS else D.Tc_base[b]
        masks = {}; nrep = 0
        cf = f'{args.dev_cache}/masks_{a}_{b}.npy' if args.dev_cache else None
        if cf and os.path.exists(cf):
            masks = {t: np.unpackbits(v)[:D.om[0].size].reshape(D.om[0].shape).astype(bool) for t, v in np.load(cf, allow_pickle=True).item().items()}
        else:
            for t in range(a, b + 1):
                masks[t], ok = repair_mask(pred, D.frames[t], D.om[t], D.Tc_base[t, 2, 3], D.K[0, 0]); nrep += ok
            if cf: np.save(cf, {t: np.packbits(m) for t, m in masks.items()}, allow_pickle=True)
        out, info = solve_moving(D, rc, a, b, Ts, Te, masks, log)
        for k, t in enumerate(range(a, b + 1)): Tc[t] = out[k]
        dump[(a, b)] = info
        print(f'moving {a}-{b}: repaired masks {nrep}, path {log["moving"][f"{a}-{b}"]["path"]} ({time.time() - T0:.0f}s)', flush=True)
    del pred; torch.cuda.empty_cache()

    # 4. depth re-anchoring to the baseline (static phases: median baseline z; moving: interpolated factor)
    fac = np.full(F, np.nan)
    for (a, b, kind) in phases:
        if kind == 'static': fac[a:b + 1] = np.median(D.Tc_base[a:b + 1, 2, 3]) / Tc[a, 2, 3]
    idx = np.where(~np.isnan(fac))[0]
    fac = np.interp(np.arange(F), idx, fac[idx])
    # moving phases: additionally match the baseline's median depth of the phase with a factor that ramps in over
    # RAMP frames from both ends (keeps the trajectory continuous with the static anchors)
    for (a, b, kind) in phases:
        if kind != 'moving': continue
        n = b - a + 1; L = max(1, min(RAMP, n // 4)); ts = np.arange(a, b + 1)
        bump = np.clip(np.minimum(ts - a + 1, b - ts + 1) / L, 0, 1)
        zb_med = np.median(D.Tc_base[a:b + 1, 2, 3]); g = 1.0
        for _ in range(20):
            z = Tc[a:b + 1, 2, 3] * fac[a:b + 1] * (1 + (g - 1) * bump)
            g *= zb_med / np.median(z)
        fac[a:b + 1] *= 1 + (g - 1) * bump
    Tc_final = Tc.copy(); Tc_final[:, :3, 3] *= fac[:, None]
    for (a, b, kind) in phases:
        dz = (Tc_final[a:b + 1, 2, 3] - D.Tc_base[a:b + 1, 2, 3]) * 100
        print(f'{kind} {a}-{b}: depth factor {fac[a:b+1].mean():.4f}; z - z_base mean {dz.mean():.1f} cm median {np.median(dz):.1f} cm')
    Tw = D.Twc[None] @ Tc_final
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, T_world_mesh=Tw.astype(np.float64))
    log['depth_factor'] = fac.tolist(); log['runtime_s'] = time.time() - T0
    json.dump(log, open(os.path.splitext(args.out)[0] + '_log.json', 'w'), indent=1, default=str)
    if args.debug_dump:
        np.save(args.debug_dump, dict(Tc_unanchored=Tc, phases=phases, motion=d, moving=dump, static=static_T), allow_pickle=True)
    print(f'done in {time.time() - T0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
