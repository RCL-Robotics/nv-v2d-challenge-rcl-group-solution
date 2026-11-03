#!/usr/bin/env python
"""Trajectory-level 6-DoF pose recovery for one monocular HOI episode (light_blue_book, episode 11).

Usage:  python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz

Pipeline (see REPORT.md for the reasoning and validation):
  1. decode video, read SAM2 masks, cut a fixed 160x160 crop per frame around the object;
  2. static scene occluders: median background -> table-top mask (bright, unsaturated component under the
     object in the first rest phase), table-top height from the baseline rest-on-table poses, table leg columns
     (constants read off the background image);
  3. phase segmentation from image motion inside the object region (rest vs. carried);
  4. rest phases: ONE pose per phase = chordal rotation mean + median translation of the baseline frames
     (the baseline orientation wins the pooled texture test against all 180-degree flips in every rest phase);
  5. carried phases: per-frame 2-D shift search of the baseline silhouette against the observed mask (fixes the
     gross lag of the per-frame tracker), then a joint optimisation of all poses of the segment with
     silhouette + texture (high-pass NCC) data terms, robust prior to the baseline, constant-velocity smoothness
     on SE(3), and the adjacent rest-phase poses as fixed anchors; depth along the viewing ray is tied to the
     baseline (sigma 3 cm) because the object-only depth cues are weak and disagree (REPORT.md, Depth).
"""
import argparse, json, os, time
import numpy as np, cv2, h5py, torch, trimesh
import torch.nn.functional as Fn
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot
from scipy.ndimage import median_filter, binary_opening, binary_closing

torch.manual_seed(0); np.random.seed(0)
torch.backends.cudnn.deterministic = True
DEV = 'cuda'
S = 160            # crop size (px)
SS = 2             # render supersampling

# Table leg columns (x0, x1, y0, y1) in full-image pixels, read off the median background image by eye
# (the legs are too low-contrast against the carpet to segment by threshold). Pixels inside are treated as
# "unknown" (possible occluder) in the silhouette term; the book passes near/behind the front-left leg.
LEG_RECTS = [(678, 709, 664, 908), (847, 875, 646, 852)]


# ----------------------------------------------------------------------------------------------- utilities
def load_mesh(inp):
    m = trimesh.load(f'{inp}/object_mesh.glb', force='scene', process=False).to_geometry()
    tex = np.asarray(m.visual.material.baseColorTexture.convert('RGB'), np.float32) / 255.
    return (np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int32),
            np.asarray(m.visual.uv, np.float32), tex)


def gauss_k(sig):
    r = int(3 * sig + 0.5); x = torch.arange(-r, r + 1, dtype=torch.float32)
    g = torch.exp(-x ** 2 / (2 * sig ** 2)); return g / g.sum()


def blur(x, sig):
    g = gauss_k(sig).to(x.device); r = len(g) // 2
    x = Fn.conv2d(Fn.pad(x[:, None], (r, r, 0, 0), mode='replicate'), g[None, None, None, :])
    x = Fn.conv2d(Fn.pad(x, (0, 0, r, r), mode='replicate'), g[None, None, :, None]); return x[:, 0]


def nblur(x, w, sig): return blur(x * w, sig) / blur(w, sig).clamp(min=1e-4)


def ncc(a, b, w):
    n = w.sum((1, 2)).clamp(min=1)
    a0 = a - (a * w).sum((1, 2))[:, None, None] / n[:, None, None]
    b0 = b - (b * w).sum((1, 2))[:, None, None] / n[:, None, None]
    return (a0 * b0 * w).sum((1, 2)) / torch.sqrt((a0 * a0 * w).sum((1, 2)) * (b0 * b0 * w).sum((1, 2)) + 1e-9)


def so3_exp(w):
    th = w.norm(dim=-1, keepdim=True).clamp(min=1e-8); k = w / th
    Kx = torch.zeros(w.shape[:-1] + (3, 3), device=w.device)
    Kx[..., 0, 1] = -k[..., 2]; Kx[..., 0, 2] = k[..., 1]; Kx[..., 1, 0] = k[..., 2]
    Kx[..., 1, 2] = -k[..., 0]; Kx[..., 2, 0] = -k[..., 1]; Kx[..., 2, 1] = k[..., 0]
    s = torch.sin(th)[..., None]; c = torch.cos(th)[..., None]
    return torch.eye(3, device=w.device) + s * Kx + (1 - c) * Kx @ Kx


def vee_skew(M):
    return torch.stack([M[..., 2, 1] - M[..., 1, 2], M[..., 0, 2] - M[..., 2, 0], M[..., 1, 0] - M[..., 0, 1]], -1) / 2


def rot_mean(Rs): return Rot.from_matrix(Rs).mean().as_matrix()


# ----------------------------------------------------------------------------------------------- evidence model
class Evidence:
    """Per-frame crops + differentiable silhouette / texture scoring with static occluders."""

    def __init__(self, K, Twc, mesh, X0, Y0, IM, OM, PM, TZ, LG, TOP):
        self.K, self.Twc = K, Twc
        v, f, uv, tex = mesh
        self.glctx = dr.RasterizeCudaContext()
        self.v = torch.tensor(v, device=DEV); self.f = torch.tensor(f, device=DEV)
        uvt = torch.tensor(uv, device=DEV); uvt[:, 1] = 1 - uvt[:, 1]; self.uv = uvt.contiguous()
        self.tex = torch.tensor(tex, device=DEV)[None].contiguous()
        self.X0, self.Y0 = X0, Y0
        self.IM = torch.tensor(IM, device=DEV).float() / 255; self.G = self.IM.mean(-1)
        self.OM = torch.tensor(OM, device=DEV); self.PM = torch.tensor(PM, device=DEV)
        self.TZ = torch.tensor(TZ, device=DEV); self.LG = torch.tensor(LG, device=DEV)
        self.TOP = torch.tensor(TOP, device=DEV)

    def proj(self, idx):
        Sz = S * SS; K = self.K; n, fa = 0.05, 20.
        P = np.zeros((len(idx), 4, 4), np.float32)
        for i, fr in enumerate(idx):
            fx, fy = K[0, 0] * SS, K[1, 1] * SS
            cx = (K[0, 2] - self.X0[fr]) * SS + (SS - 1) / 2; cy = (K[1, 2] - self.Y0[fr]) * SS + (SS - 1) / 2
            P[i] = [[2 * fx / Sz, 0, 2 * cx / Sz - 1, 0], [0, 2 * fy / Sz, 2 * cy / Sz - 1, 0],
                    [0, 0, (fa + n) / (fa - n), -2 * fa * n / (fa - n)], [0, 0, 1, 0]]
        return torch.tensor(P, device=DEV)

    def render(self, Tc, idx, rgb=True):
        Sz = S * SS
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        vc = torch.einsum('bij,nj->bni', Tc.float(), vh)
        clip = torch.einsum('bij,bnj->bni', self.proj(idx), vc).contiguous()
        rast, _ = dr.rasterize(self.glctx, clip, self.f, (Sz, Sz))
        msk = (rast[..., 3:4] > 0).float()
        zc, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, self.f)
        out = [msk]
        if rgb:
            uvi, _ = dr.interpolate(self.uv[None], rast, self.f)
            col = dr.texture(self.tex, uvi, filter_mode='linear')
            out.append(col.mean(-1, keepdim=True) * msk)
        out = dr.antialias(torch.cat(out, -1).contiguous(), rast, clip, self.f)
        out = Fn.avg_pool2d(out.permute(0, 3, 1, 2), SS)
        self._z = Fn.max_pool2d((zc * msk).detach().permute(0, 3, 1, 2), SS)[:, 0]
        m = out[:, 0]; g = out[:, 1] / m.clamp(min=1e-4) if rgb else None
        return m, g

    def losses(self, Tc, idx, w_app=1.0, rgb=True):
        idx = list(idx); it = torch.tensor(idx, device=DEV)
        m, g = self.render(Tc, idx, rgb=rgb)
        om = self.OM[it].float()
        # occluded / unknown pixels: person, table legs, table top in front of the rendered surface
        occ = self.PM[it] | self.LG[it] | ((self._z > self.TZ[it] + 0.02) & (self._z > 0))
        valid = (~occ).float(); self._valid = valid
        inter = (m * om * valid).sum((1, 2)); uni = ((m + om - m * om) * valid).sum((1, 2)).clamp(min=1)
        iou = inter / uni
        mb = blur(m, 3.); ob = blur(om, 3.)
        lb = (((mb - ob) ** 2) * valid).sum((1, 2)) / ((ob * valid).sum((1, 2)).clamp(min=20))
        if not rgb: return {'iou': iou, 'lb': lb, 'loss': (1 - iou) + lb}
        inner = (-Fn.max_pool2d(-(m > 0.99).float()[:, None], 5, 1, 2)[:, 0]) * valid * om
        I = self.G[it]
        hp = ncc(g - nblur(g, inner, 1.5), I - nblur(I, inner, 1.5), inner)
        hp = torch.where(inner.sum((1, 2)) > 30, hp, torch.zeros_like(hp))
        return {'iou': iou, 'lb': lb, 'ncc': hp, 'loss': (1 - iou) + lb + w_app * (1 - hp)}

    def coarse_app(self, Tc, idx):
        """Validation score (not optimised directly): band-pass + raw NCC of rendered texture vs image."""
        it = torch.tensor(list(idx), device=DEV)
        with torch.no_grad():
            self.losses(Tc, idx, rgb=False); valid = self._valid
            m, g = self.render(Tc, idx, rgb=True)
            inner = (-Fn.max_pool2d(-(m > 0.99).float()[:, None], 5, 1, 2)[:, 0]) * valid * self.OM[it].float()
            I = self.G[it]
            s1 = ncc(nblur(g, inner, 1.5) - nblur(g, inner, 5.0), nblur(I, inner, 1.5) - nblur(I, inner, 5.0), inner)
            s2 = ncc(g, I, inner)
        return ((s1 + s2) / 2).cpu().numpy()

    def shift_search(self, Tc, idx, maxs=48):
        """Best 2-D image shift of the rendered silhouette vs. the observed mask (table top = don't care)."""
        it = torch.tensor(idx, device=DEV)
        with torch.no_grad():
            m, _ = self.render(torch.tensor(Tc, device=DEV), idx, rgb=False); m = (m > 0.5).float()
            om = self.OM[it].float(); occ = (self.PM[it] | self.LG[it]).float(); top = self.TOP[it].float()
            bg = (1 - om) * (1 - occ) * (1 - top)
            P = S + 2 * maxs
            def corr(a, b):
                A = torch.fft.rfft2(Fn.pad(a, (0, P - S, 0, P - S))); B = torch.fft.rfft2(Fn.pad(b, (0, P - S, 0, P - S)))
                c = torch.fft.irfft2(torch.conj(A) * B, s=(P, P))
                ds = torch.arange(-maxs, maxs + 1, device=DEV) % P
                return c[:, ds][:, :, ds]
            ov = corr(m, om); fp = corr(m, bg)
            iou = ov / (om.sum((1, 2))[:, None, None] + fp).clamp(min=1)
            best = iou.flatten(1).argmax(1); dy = best // (2 * maxs + 1) - maxs; dx = best % (2 * maxs + 1) - maxs
            i0 = iou[:, maxs, maxs]; i1 = iou.flatten(1).max(1).values
        return torch.stack([dx, dy], 1).cpu().numpy(), i0.cpu().numpy(), i1.cpu().numpy()


def seg_optimize(E, Tinit, Tprior, idx, anchor_l=None, anchor_r=None, iters=150, lr_r=0.01, lr_t=0.003,
                 s_acc_t=0.006, s_acc_r=0.03, c_r=np.radians(15), c_t=0.05, w_prior=1.0, s_depth=0.03,
                 w_app=1.0, chunk=80, has_ev=None):
    """Joint optimisation of all poses of one carried segment (camera frame).
    data: silhouette IoU + blurred-mask L2 + texture NCC (frames with a usable mask only);
    smoothness: translation and rotation acceleration (constant-velocity prior), anchored at the rest poses;
    prior: robust (Geman-McClure) pull to the baseline rotation/lateral position; Gaussian on depth along ray."""
    N = len(idx)
    T0 = torch.tensor(Tinit, device=DEV, dtype=torch.float32); TP = torch.tensor(Tprior, device=DEV, dtype=torch.float32)
    w = torch.zeros(N, 3, device=DEV, requires_grad=True); t = torch.zeros(N, 3, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([{'params': [w], 'lr': lr_r}, {'params': [t], 'lr': lr_t}])
    ev = torch.tensor(has_ev, device=DEV, dtype=torch.float32)
    AL = torch.tensor(anchor_l, device=DEV, dtype=torch.float32)[None] if anchor_l is not None else None
    AR = torch.tensor(anchor_r, device=DEV, dtype=torch.float32)[None] if anchor_r is not None else None
    ray = TP[:, :3, 3] / TP[:, :3, 3].norm(dim=-1, keepdim=True)

    def build(): return so3_exp(w) @ T0[:, :3, :3], T0[:, :3, 3] + t

    for _ in range(iters):
        opt.zero_grad()
        R, tt = build()
        Rs, ts = [R], [tt]
        if AL is not None: Rs = [AL[:, :3, :3]] + Rs; ts = [AL[:, :3, 3]] + ts
        if AR is not None: Rs = Rs + [AR[:, :3, :3]]; ts = ts + [AR[:, :3, 3]]
        Ra = torch.cat(Rs); ta = torch.cat(ts)
        rel = vee_skew(Ra[:-1].transpose(1, 2) @ Ra[1:])
        L_s = (((rel[1:] - rel[:-1]) / s_acc_r) ** 2).sum() + (((ta[2:] - 2 * ta[1:-1] + ta[:-2]) / s_acc_t) ** 2).sum()
        ang = torch.acos((((TP[:, :3, :3] * R).sum((1, 2)) - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6))
        dtp = tt - TP[:, :3, 3]; dpar = (dtp * ray).sum(-1); dlat = (dtp - dpar[:, None] * ray).norm(dim=-1)
        L_p = w_prior * ((ang ** 2 / (ang ** 2 + c_r ** 2)) + (dlat ** 2 / (dlat ** 2 + c_t ** 2))).sum() \
            + ((dpar / s_depth) ** 2).sum()
        (L_s + L_p).backward()
        for s in range(0, N, chunk):
            sl = slice(s, min(s + chunk, N))
            R2, t2 = build()
            T2 = torch.eye(4, device=DEV).repeat(sl.stop - sl.start, 1, 1); T2[:, :3, :3] = R2[sl]; T2[:, :3, 3] = t2[sl]
            L = E.losses(T2, idx[sl], w_app=w_app, rgb=w_app > 0)
            (L['loss'] * ev[sl]).sum().backward()
        opt.step()
    with torch.no_grad():
        R, tt = build(); T = torch.eye(4, device=DEV).repeat(N, 1, 1); T[:, :3, :3] = R; T[:, :3, 3] = tt
    return T.cpu().numpy().astype(np.float64)


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='/work/inputs')
    ap.add_argument('--out', default='/work/outputs/object_poses_world.npz')
    args = ap.parse_args(); inp = args.inputs
    t_start = time.time()
    cam = json.load(open(f'{inp}/camera.json')); K = np.array(cam['K']); Twc = np.array(cam['T_world_cam']); Tcw = np.linalg.inv(Twc)
    W, H = cam['width'], cam['height']
    gp = json.load(open(f'{inp}/ground_plane.json')); n_fl = np.array(gp['plane'][:3]); d_fl = gp['plane'][3]
    rec = np.load(f'{inp}/reconstruction.npz')
    Tw_base = rec['T_world_mesh'].astype(np.float64); F = len(Tw_base)
    Tb = np.einsum('ij,njk->nik', Tcw, Tw_base)            # baseline, camera frame
    mesh = load_mesh(inp)

    # ---- 1. video, masks, crops, motion
    cap = cv2.VideoCapture(f'{inp}/video.mp4')
    hm = h5py.File(f'{inp}/masks.h5', 'r')
    X0 = np.zeros(F, int); Y0 = np.zeros(F, int)
    IM = np.zeros((F, S, S, 3), np.uint8); OM = np.zeros((F, S, S), bool); PM = np.zeros((F, S, S), bool)
    oarea = np.zeros(F); d5 = np.full(F, np.nan)
    bg_samples = []; ring = {}
    for fr in range(F):
        ok, bgr = cap.read(); assert ok, fr
        img = bgr[..., ::-1]
        om = hm['object'][fr]; pm = hm['person'][fr]
        ys, xs = np.nonzero(om); oarea[fr] = len(xs)
        p = K @ Tb[fr, :3, 3]; pc = p[:2] / p[2]
        c = np.array([xs.mean(), ys.mean()]) if len(xs) > 100 else pc
        x0 = int(np.clip(c[0] - S / 2, 0, W - S)); y0 = int(np.clip(c[1] - S / 2, 0, H - S)); X0[fr], Y0[fr] = x0, y0
        IM[fr] = img[y0:y0 + S, x0:x0 + S]; OM[fr] = om[y0:y0 + S, x0:x0 + S]; PM[fr] = pm[y0:y0 + S, x0:x0 + S]
        gray = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY).astype(np.float32)
        ring[fr] = (gray, om, pm)
        if fr - 5 in ring:   # mean abs. intensity change over 5 frames inside the object region, person excluded
            g0, o0, p0 = ring[fr - 5]
            reg = cv2.dilate((om | o0).astype(np.uint8), np.ones((5, 5))) > 0
            reg &= ~pm & ~p0
            if reg.sum() > 50: d5[fr] = np.abs(gray - g0)[reg].mean()
        ring.pop(fr - 6, None)
        if fr % 15 == 0:
            b = img.astype(np.float32).copy()
            b[cv2.dilate((om | pm).astype(np.uint8), np.ones((15, 15))) > 0] = np.nan; bg_samples.append(b)
    print(f'[{time.time() - t_start:.0f}s] decoded {F} frames')

    # ---- 2. static occluders: table top (plane) and leg columns
    bgimg = np.nanmedian(np.stack(bg_samples), 0); del bg_samples
    gimg = bgimg.mean(-1); sat = bgimg.max(-1) - bgimg.min(-1)
    th = cv2.morphologyEx(((gimg > 185) & (sat < 35)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3)))
    _, lab, _, _ = cv2.connectedComponentsWithStats(th)
    # seed: pixels just below the object mask in frame 0 (the book initially rests on the table)
    om0 = hm['object'][0]; ys, xs = np.nonzero(om0)
    seeds = lab[np.clip(ys.max() + np.arange(3, 25), 0, H - 1), int(np.median(xs))]
    seed = np.bincount(seeds[seeds > 0]).argmax()
    top = cv2.morphologyEx((lab == seed).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7)))
    kd = np.zeros((9, 1), np.uint8); kd[4:] = 1
    top = cv2.dilate(top, kd) > 0          # include the front edge strip of the table top
    legs = np.zeros((H, W), bool)
    for x0_, x1_, y0_, y1_ in LEG_RECTS: legs[y0_:y1_, x0_:x1_] = True

    # ---- 3. phase segmentation (rest phases = sustained low image motion of the object)
    d5f = np.nan_to_num(d5, nan=99.); d5f[:5] = d5f[5]
    st = median_filter(d5f, 9, mode='nearest') < 6.5
    pad = 30   # edge padding so the morphology does not erode the first/last phase
    st = np.r_[np.repeat(st[0], pad), st, np.repeat(st[-1], pad)]
    st = binary_opening(binary_closing(st, np.ones(9)), np.ones(25))[pad:-pad]
    phases = []; i = 0
    while i < F:
        if st[i]:
            j = i
            while j < F and st[j]: j += 1
            phases.append((i, j - 1)); i = j
        else: i += 1
    # lowest mesh point above the floor, per rest phase (baseline) -> resting on floor or on table?
    vs = mesh[0][::200].astype(np.float64)
    low = np.array([((vs @ Tw_base[f, :3, :3].T + Tw_base[f, :3, 3]) @ n_fl + d_fl).min() for f in range(F)])
    rest = []
    for a, b in phases:
        h = np.median(low[a:b + 1])
        # a rest phase must be supported: on the floor (< 8 cm) or on the table top (0.6-1.0 m), and the
        # object must be held still for >= 1 s.  The final 'held behind the table edge' stretch fails this
        # (the book is held by both hands below the table-top level) and stays a moving segment.
        if b - a + 1 >= 30 and (h < 0.08 or 0.6 < h < 1.0) and (h < 0.08 or abs(h - np.median(low[phases[0][0]:phases[0][1] + 1])) < 0.05):
            rest.append((a, b, h))
    print('rest phases', [(a, b, round(h, 3)) for a, b, h in rest])
    table_h = np.median([h for a, b, h in rest if h > 0.3])
    print(f'table-top height (baseline rest-on-table poses) {table_h:.3f} m')
    # table-top depth map (camera z) from the plane at table_h above the floor
    nC = Twc[:3, :3].T @ n_fl; dC = n_fl @ Twc[:3, 3] + d_fl - table_h
    yy, xx = np.mgrid[0:H, 0:W]
    rays = np.stack([(xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1], np.ones((H, W))], -1)
    z = -dC / (rays @ nC); z[(z < 0) | ~top] = 1e9
    TZ = np.stack([z[Y0[f]:Y0[f] + S, X0[f]:X0[f] + S] for f in range(F)]).astype(np.float32)
    LG = np.stack([legs[Y0[f]:Y0[f] + S, X0[f]:X0[f] + S] for f in range(F)])
    TOP = np.stack([top[Y0[f]:Y0[f] + S, X0[f]:X0[f] + S] for f in range(F)])
    E = Evidence(K, Twc, mesh, X0, Y0, IM, OM, PM, TZ, LG, TOP)

    # ---- 4. rest phases: one pose each
    out = Tb.copy(); rest_pose = {}
    for a, b, h in rest:
        T = np.eye(4); T[:3, :3] = rot_mean(Tb[a:b + 1, :3, :3]); T[:3, 3] = np.median(Tb[a:b + 1, :3, 3], 0)
        out[a:b + 1] = T; rest_pose[(a, b)] = T

    # ---- 5. carried segments
    has_ev = (oarea > 100).astype(float)
    sh = np.zeros((F, 4))
    for s in range(0, F, 100):
        idx = list(range(s, min(s + 100, F))); d, i0, i1 = E.shift_search(Tb[idx], idx); sh[idx] = np.c_[d, i0, i1]
    Tinit = Tb.copy()
    for f in np.nonzero(sh[:, 3] - sh[:, 2] > 0.1)[0]:     # gross misplacement only (IoU gain > 0.1)
        zf = Tinit[f, 2, 3]; Tinit[f, 0, 3] += sh[f, 0] * zf / K[0, 0]; Tinit[f, 1, 3] += sh[f, 1] * zf / K[1, 1]
    print(f'[{time.time() - t_start:.0f}s] shift-corrected frames: {int((sh[:, 3] - sh[:, 2] > 0.1).sum())}')
    bounds = [(-1, -1)] + [(a, b) for a, b, _ in rest] + [(F, F)]
    for k in range(len(bounds) - 1):
        a = bounds[k][1] + 1; b = bounds[k + 1][0] - 1
        if b < a: continue
        idx = list(range(a, b + 1))
        AL = rest_pose.get(bounds[k]); AR = rest_pose.get(bounds[k + 1])
        out[idx] = seg_optimize(E, Tinit[idx], Tinit[idx], idx, anchor_l=AL, anchor_r=AR, has_ev=has_ev[idx])
        print(f'[{time.time() - t_start:.0f}s] segment {a}-{b} optimised')

    # ---- 6. diagnostics + output
    Tw = np.einsum('ij,njk->nik', Twc, out)
    for name, TT in [('baseline', Tb), ('result', out)]:
        iou = np.zeros(F); app = np.zeros(F)
        for s in range(0, F, 60):
            idx = list(range(s, min(s + 60, F)))
            with torch.no_grad(): iou[idx] = E.losses(torch.tensor(TT[idx], device=DEV), idx, rgb=False)['iou'].cpu().numpy()
            app[idx] = E.coarse_app(torch.tensor(TT[idx], device=DEV), idx)
        e = has_ev > 0
        print(f'{name}: mean IoU {iou[e].mean():.3f}  mean texture score {app[e].mean():.3f}')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, T_world_mesh=Tw.astype(np.float64))
    print(f'wrote {args.out}  ({F} frames)  total {time.time() - t_start:.0f}s')


if __name__ == '__main__':
    main()
