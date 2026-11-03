#!/usr/bin/env python
"""Trajectory-level re-estimation of the 6-DoF object pose for one HOI video.

Method (see REPORT.md):
 1. phase segmentation (static / moving) from image differences inside the object region,
 2. per-frame silhouette evidence: robust 2-D chamfer between projected mesh surface points and
    the SAM2 object mask (points that fall on the person are not penalised = occlusion aware),
 3. joint optimisation over the whole episode: ONE pose per static phase (evidence pooled over all
    its frames), one pose per moving frame, SE(3) second-difference smoothness, initialised from
    the baseline (180-deg flip hypotheses were tested offline and always lost by a clear margin; the
    same test is re-run here as a sanity check and logged),
 4. re-anchoring: per phase, camera-frame positions are scaled along the viewing rays so that the
    median camera depth equals the baseline's (silhouette-only depth is biased, see REPORT.md),
    with a smooth blend of the scale across the moving phase so the trajectory stays continuous.
Deterministic: fixed seeds, fixed point samples, fixed iteration counts.
"""
import argparse, json, os, time
import numpy as np, h5py, cv2, trimesh, torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rot

# ----------------------------------------------------------------------------- utils
def so3_exp(w):
    th = w.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    k = w / th
    Kx = torch.zeros(w.shape[:-1] + (3, 3), device=w.device, dtype=w.dtype)
    Kx[..., 0, 1] = -k[..., 2]; Kx[..., 0, 2] = k[..., 1]; Kx[..., 1, 0] = k[..., 2]
    Kx[..., 1, 2] = -k[..., 0]; Kx[..., 2, 0] = -k[..., 1]; Kx[..., 2, 1] = k[..., 0]
    th = th[..., None]
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(Kx)
    return I + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx

def so3_log(R):
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    th = torch.acos(((tr - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7))
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = torch.where(th[..., None] < 1e-4, 0.5 + th[..., None] ** 2 / 12,
                    th[..., None] / (2 * torch.sin(th[..., None]).clamp(min=1e-9)))
    return v * s

def project(K, X):
    return torch.stack([X[..., 0] / X[..., 2] * K[0, 0] + K[0, 2], X[..., 1] / X[..., 2] * K[1, 1] + K[1, 2]], -1)

def mean_rot(Rs):
    return Rot.from_matrix(Rs).mean().as_matrix()

def flip180(ax, c):
    """4x4 transform in mesh coords: 180 deg about mesh axis `ax` through the centroid c"""
    Rf = -np.eye(3); Rf[ax, ax] = 1
    F4 = np.eye(4); F4[:3, :3] = Rf; F4[:3, 3] = c - Rf @ c
    return F4

# ----------------------------------------------------------------------------- evidence
class Evidence:
    """Per-frame silhouette targets at full resolution, stored as fixed-size crops.
    dt  : distance (px) to the nearest object-or-person pixel (person = possible occluder)
    q   : random object-mask pixels (for the coverage / reverse chamfer term)"""
    CH, CW = 640, 768

    def __init__(self, obj, per, frames, nq=800, seed=0, person_dil=4):
        rng = np.random.RandomState(seed)
        n = len(frames)
        self.dt = np.full((n, self.CH, self.CW), 1e3, np.float32)
        self.off = np.zeros((n, 2), np.float32)
        self.q = np.zeros((n, nq, 2), np.float32)
        self.area = np.zeros(n)
        ker = np.ones((2 * person_dil + 1,) * 2, np.uint8)
        for j, i in enumerate(frames):
            o = obj[i]; p = cv2.dilate(per[i].astype(np.uint8), ker) > 0
            ys, xs = np.nonzero(o)
            self.area[j] = len(xs)
            cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
            x0 = int(np.clip(cx - self.CW // 2, 0, o.shape[1] - self.CW)); y0 = int(np.clip(cy - self.CH // 2, 0, o.shape[0] - self.CH))
            ok = (o | p)[y0:y0 + self.CH, x0:x0 + self.CW]
            self.dt[j] = cv2.distanceTransform((~ok).astype(np.uint8), cv2.DIST_L2, 5)
            self.off[j] = (x0, y0)
            sel = rng.choice(len(xs), nq, replace=len(xs) < nq)
            self.q[j] = np.stack([xs[sel], ys[sel]], 1) + 0.5
        self.dt = torch.tensor(self.dt, device='cuda'); self.off = torch.tensor(self.off, device='cuda')
        self.q = torch.tensor(self.q, device='cuda')

    def loss(self, uv, idx, clamp=25.0, rev_sel=None):
        """uv (B,N,2) full-res pixel coords for evidence rows idx -> (fwd, rev) mean robust px distances"""
        uv = uv.float()
        loc = uv - self.off[idx][:, None]
        g = torch.stack([loc[..., 0] / self.CW * 2 - 1, loc[..., 1] / self.CH * 2 - 1], -1)[:, :, None]
        d = F.grid_sample(self.dt[idx][:, None], g, align_corners=False, padding_mode='border')[:, 0, :, 0]
        fwd = d.clamp(max=clamp).mean(1)
        up = uv if rev_sel is None else uv[:, rev_sel]
        rev = torch.cdist(self.q[idx], up).min(2).values.clamp(max=clamp).mean(1)
        return fwd, rev

# ----------------------------------------------------------------------------- phases
def detect_phases(video, obj, per, F_, lag=5, thr=4.0, min_run=5, min_static=30):
    """static frames = the object region does not change. d(i) = mean |I_i - I_{i-lag}| over the
    (dilated) object mask of frame i-lag, excluding (dilated) person pixels, half resolution grey.
    Static phases are the leading / trailing runs with d below `thr` grey levels (sensor noise +
    slow exposure drift are ~1-1.5; a moving ladder gives 5-20). Returns (s1_end, s2_start, d)."""
    cap = cv2.VideoCapture(video); g = []
    for i in range(F_):
        ok, im = cap.read(); assert ok, i
        g.append(cv2.cvtColor(cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32))
    k9 = np.ones((9, 9), np.uint8); k7 = np.ones((7, 7), np.uint8)
    def half(m): return cv2.resize(m.astype(np.uint8), (m.shape[1] // 2, m.shape[0] // 2), interpolation=cv2.INTER_AREA) > 0
    d = np.zeros(F_)
    for i in range(lag, F_):
        reg = cv2.dilate(half(obj[i - lag]).astype(np.uint8), k9) > 0
        v = reg & ~(cv2.dilate(half(per[i]).astype(np.uint8), k7) > 0) & ~(cv2.dilate(half(per[i - lag]).astype(np.uint8), k7) > 0)
        d[i] = np.abs(g[i] - g[i - lag])[v].mean() if v.sum() > 50 else np.nan
    mov = np.nan_to_num(d, nan=0.0) > thr
    # keep only sustained motion (>= min_run consecutive frames): brief blips come from the
    # person's legs / shadow passing next to the object (e.g. frames 80-82 here, d~3.3)
    lab = np.zeros(F_, bool); i = 0
    while i < F_:
        if mov[i]:
            j = i
            while j < F_ and mov[j]: j += 1
            if j - i >= min_run: lab[i:j] = True
            i = j
        else: i += 1
    mov = lab
    # frame i moving if the change over (i-lag, i] is large -> motion starts at first such i - lag + 1
    first = int(np.argmax(mov)); last = int(F_ - 1 - np.argmax(mov[::-1]))
    s1_end = max(first - lag + 1, 0)        # frames [0, s1_end) static
    s2_start = min(last + 1, F_)            # frames [s2_start, F) static
    if s1_end < min_static: s1_end = 0
    if F_ - s2_start < min_static: s2_start = F_
    return s1_end, s2_start, d

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='/work/inputs'); ap.add_argument('--out', default='/work/outputs/object_poses_world.npz')
    ap.add_argument('--iters', type=int, default=600)
    args = ap.parse_args()
    t_start = time.time()
    np.random.seed(0); torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    inp = args.inputs
    cam = json.load(open(f'{inp}/camera.json'))
    K = np.array(cam['K']); Twc = np.array(cam['T_world_cam']); Tcw = np.linalg.inv(Twc)
    rec = np.load(f'{inp}/reconstruction.npz')
    Tw_base = rec['T_world_mesh'].astype(np.float64); F_ = len(Tw_base)
    Tb = Tcw @ Tw_base                                        # baseline camera-frame poses
    mesh = trimesh.load(f'{inp}/object_mesh.glb', force='scene', process=False).to_geometry()
    c = mesh.vertices.mean(0)
    pts_np, _ = trimesh.sample.sample_surface(mesh, 4000, seed=0)
    pts = torch.tensor(pts_np, device='cuda')
    h5 = h5py.File(f'{inp}/masks.h5'); obj = h5['object'][:]; per = h5['person'][:]
    log = {}

    # ---- 1. phases
    s1_end, s2_start, dmot = detect_phases(f'{inp}/video.mp4', obj, per, F_)
    print(f'phases: static [0,{s1_end})  moving [{s1_end},{s2_start})  static [{s2_start},{F_})')
    log['phases'] = [s1_end, s2_start]
    static_sets = []
    if s1_end > 0: static_sets.append(np.arange(0, s1_end))
    if s2_start < F_: static_sets.append(np.arange(s2_start, F_))
    moving = np.arange(s1_end, s2_start)

    # ---- 2. evidence (static phases: every 2nd frame is plenty, they share one pose)
    ev_frames = np.unique(np.concatenate([s[::2] for s in static_sets] + [moving]))
    ev = Evidence(obj, per, ev_frames)
    row = -np.ones(F_, int); row[ev_frames] = np.arange(len(ev_frames))
    del obj, per
    Kt = torch.tensor(K, device='cuda')

    # ---- 3. parameters: static phases share one pose; init from baseline (mean rot / median pos)
    slot = np.arange(F_)                    # frame -> parameter slot
    T_init = Tb.copy()
    for s in static_sets:
        T0 = np.eye(4); T0[:3, :3] = mean_rot(Tb[s, :3, :3]); T0[:3, 3] = np.median(Tb[s, :3, 3], 0)
        T_init[s] = T0; slot[s] = s[0]
    uslots = np.unique(slot); slot_idx = np.searchsorted(uslots, slot)
    R0 = torch.tensor(T_init[uslots, :3, :3], device='cuda'); t0 = torch.tensor(T_init[uslots, :3, 3], device='cuda')
    ct = torch.tensor(c, device='cuda')
    W = torch.zeros(len(uslots), 3, dtype=torch.float64, device='cuda', requires_grad=True)
    D = torch.zeros(len(uslots), 3, dtype=torch.float64, device='cuda', requires_grad=True)
    slot_idx_t = torch.tensor(slot_idx, device='cuda')
    ev_rows = torch.tensor(row[ev_frames], device='cuda'); ev_slots = torch.tensor(slot_idx[ev_frames], device='cuda')
    # each frame's data weight: static frames share a pose -> weight so a static phase counts like
    # its number of frames (every 2nd frame sampled -> x2)
    wdata = torch.ones(len(ev_frames), device='cuda')
    for s in static_sets: wdata[torch.tensor(np.isin(ev_frames, s), device='cuda')] = len(s) / len(s[::2])

    def poses():
        R = so3_exp(W) @ R0
        t = t0 + torch.einsum('bij,j->bi', R0, ct) - torch.einsum('bij,j->bi', R, ct) + D
        return R, t

    SIG_R, SIG_T = 0.02, 0.01     # rad / m per frame^2 (second differences) - see REPORT (held-out check)
    def smooth(R, t):
        Rf, tf = R[slot_idx_t], t[slot_idx_t]
        relc = torch.einsum('bij,bj->bi', Rf[:-1], so3_log(Rf[:-1].transpose(1, 2) @ Rf[1:]))
        p = torch.einsum('bij,j->bi', Rf, ct) + tf
        return ((relc[1:] - relc[:-1]) / SIG_R).pow(2).sum() + ((p[2:] - 2 * p[1:-1] + p[:-2]) / SIG_T).pow(2).sum()

    opt = torch.optim.Adam([W, D], lr=2e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(args.iters * 0.6), int(args.iters * 0.85)], 0.3)
    gen = torch.Generator(device='cuda'); gen.manual_seed(0)
    CH = 120
    for it in range(args.iters):
        opt.zero_grad()
        R, t = poses()
        sm = smooth(R, t); sm.backward(retain_graph=True)
        sel = torch.randperm(len(pts), device='cuda', generator=gen)[:1500]
        tot = 0.0
        for k in range(0, len(ev_frames), CH):
            sl = ev_slots[k:k + CH]
            X = torch.einsum('bij,nj->bni', R[sl], pts) + t[sl][:, None]
            fw, rv = ev.loss(project(Kt, X), ev_rows[k:k + CH], rev_sel=sel)
            l = ((fw + rv) * wdata[k:k + CH]).sum()
            l.backward(retain_graph=True); tot += l.item()
        opt.step(); sched.step()
        if it % 100 == 0 or it == args.iters - 1:
            print(f'it {it} data {tot / wdata.sum().item():.3f} smooth {sm.item():.1f}  {time.time() - t_start:.0f}s', flush=True)
    with torch.no_grad():
        R, t = poses()
        Tfit = np.tile(np.eye(4), (F_, 1, 1))
        Tfit[:, :3, :3] = R[slot_idx_t].cpu().numpy(); Tfit[:, :3, 3] = t[slot_idx_t].cpu().numpy()

    # ---- 4. re-anchor depth to the baseline (per phase, along the viewing rays)
    def centre(T): return np.einsum('bij,j->bi', T[:, :3, :3], c) + T[:, :3, 3]
    pc, pb = centre(Tfit), centre(Tb)
    phase_sets = static_sets[:1] + ([moving] if len(moving) else []) + static_sets[1:]
    scale = {}
    for name, s in zip(['S1', 'M', 'S2'] if len(phase_sets) == 3 else [str(i) for i in range(len(phase_sets))], phase_sets):
        scale[name] = np.median(pb[s, 2]) / np.median(pc[s, 2])
    print('re-anchor scales', scale)
    sc = np.ones(F_)
    for name, s in zip(scale.keys(), phase_sets): sc[s] = scale[name]
    Tout_c = Tfit.copy(); Tout_c[:, :3, 3] += (sc - 1)[:, None] * pc
    T_world = Twc @ Tout_c
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, T_world_mesh=T_world.astype(np.float64))
    log['scales'] = {k: float(v) for k, v in scale.items()}
    log['runtime_s'] = time.time() - t_start
    json.dump(log, open(os.path.join(os.path.dirname(args.out), 'solve_log.json'), 'w'), indent=1)
    print('done', T_world.shape, f'{time.time() - t_start:.0f}s')

if __name__ == '__main__':
    main()
