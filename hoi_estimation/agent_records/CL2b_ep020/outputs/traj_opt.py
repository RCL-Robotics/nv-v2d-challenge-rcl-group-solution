"""Joint trajectory optimisation of an upright object (yaw + image position + depth) over a whole video."""
import numpy as np
import torch
from hoi_lib import DiffFitter, rotz_t


def runs(mask):
    out, a = [], 0
    for i in range(1, len(mask) + 1):
        if i == len(mask) or mask[i] != mask[a]:
            out.append([a, i - 1, bool(mask[a])]); a = i
    return out


def segment_phases(d1, thr=0.8, min_static=15, min_moving=4, pad=6):
    """d1: per-frame mean abs frame difference on visible object pixels.
    Returns list of (start, end, is_moving)."""
    from scipy.ndimage import median_filter
    s = median_filter(d1, 5, mode="nearest")
    mv = s > thr
    changed = True
    while changed:  # remove short runs (short moving blips -> static first, then short static gaps -> moving)
        changed = False
        for a, b, m in runs(mv):
            if m and b - a + 1 < min_moving:
                mv[a:b + 1] = False; changed = True; break
        if changed:
            continue
        for a, b, m in runs(mv):
            if (not m) and b - a + 1 < min_static and a > 0 and b < len(mv) - 1:
                mv[a:b + 1] = True; changed = True; break
    # pad moving phases: motion onset/offset is gradual, a per-frame segment can represent a still object
    # but a static segment cannot represent a moving one
    mv2 = mv.copy()
    for a, b, m in runs(mv):
        if m:
            mv2[max(0, a - pad):min(len(mv), b + pad + 1)] = True
    return [tuple(r) for r in runs(mv2)]


class TrajectoryOptimizer:
    """Per-unit parameters (theta[rad], u[px], v[px], z[m]); a unit is a static phase or a single moving frame.
    Pose: R_cam_mesh = Rcw @ B @ Rz(theta), t_cam = z * K^-1 [u, v, 1]."""

    def __init__(self, renderer, K, Rcw, B, crop, frames_rgb, oms, pms, phases, init_theta, init_uvz,
                 dev="cuda"):
        self.dev = dev
        self.D = DiffFitter(renderer, K, crop=crop)
        self.D.load_frames(frames_rgb, oms, pms)
        self.F = len(frames_rgb)
        unit_of = np.zeros(self.F, np.int64)
        units = []
        for a, b, mv in phases:
            if mv:
                for i in range(a, b + 1):
                    unit_of[i] = len(units); units.append((i, i, False))
            else:
                unit_of[a:b + 1] = len(units); units.append((a, b, True))
        self.units = units
        self.unit_of = torch.tensor(unit_of, device=dev)
        U = len(units)
        th0 = np.zeros(U); uvz0 = np.zeros((U, 3))
        for k, (a, b, st) in enumerate(units):
            th0[k] = np.median(init_theta[a:b + 1])
            uvz0[k] = np.median(init_uvz[a:b + 1], 0)
        self.th0 = torch.tensor(th0, device=dev, dtype=torch.float32)
        self.uvz0 = torch.tensor(uvz0, device=dev, dtype=torch.float32)
        # offsets in natural units: deg, px, px, cm
        self.raw = torch.zeros(U, 4, device=dev, requires_grad=True)
        self.step = torch.tensor([0.5, 0.15, 0.15, 0.5], device=dev)  # Adam step size per dim (deg, px, px, cm)
        self.Bc = torch.tensor(Rcw @ B, device=dev, dtype=torch.float32)
        self.Kinv = torch.tensor(np.linalg.inv(K), device=dev, dtype=torch.float32)

    @property
    def off(self):
        return self.raw * self.step

    def unit_params(self):
        th = self.th0 + torch.deg2rad(self.off[:, 0])
        uvz = self.uvz0 + self.off[:, 1:] * torch.tensor([1.0, 1.0, 0.01], device=self.dev)
        return th, uvz

    def pose(self, th, uvz):
        R = self.Bc[None] @ rotz_t(th)
        ray = torch.cat([uvz[:, :2], torch.ones_like(uvz[:, :1])], 1) @ self.Kinv.T
        t = ray * uvz[:, 2:3]
        return R, t

    def frame_params(self):
        th, uvz = self.unit_params()
        return th[self.unit_of], uvz[self.unit_of]

    def smooth_loss(self, lam):
        th, uvz = self.frame_params()
        P = torch.stack([torch.rad2deg(th), uvz[:, 0], uvz[:, 1], uvz[:, 2] * 100], 1)  # deg, px, px, cm
        acc = P[2:] - 2 * P[1:-1] + P[:-2]
        vel = P[1:] - P[:-1]
        l = torch.tensor(lam["acc"], device=self.dev)
        lv = torch.tensor(lam["vel"], device=self.dev)
        return (acc ** 2 * l).sum() + (vel ** 2 * lv).sum()

    def data_terms(self, idx, w_ncc, grad=True):
        th, uvz = self.frame_params()
        th, uvz = th[idx], uvz[idx]
        R, t = self.pose(th, uvz)
        iou, _ = self.D.losses(R, t, idx, w_ncc=0)
        # appearance term must not move the depth (texture scale is biased by blur): detach z
        uvz_d = torch.cat([uvz[:, :2], uvz[:, 2:].detach()], 1)
        R2, t2 = self.pose(th, uvz_d)
        _, ncc = self.D.losses(R2, t2, idx)
        return iou, ncc

    def optimize(self, iters=200, lr=(0.5, 0.15, 0.15, 0.5), w_ncc=0.5, lam=None, chunk=96, frame_w=None,
                 fix_z=False, log=print):
        lam = lam or {"acc": [2e-3, 2e-2, 2e-2, 2e-2], "vel": [0, 0, 0, 0]}
        new = torch.tensor(lr, device=self.dev)
        with torch.no_grad():
            self.raw *= self.step / new  # keep current offsets when changing the step scale
        self.step = new
        opt = torch.optim.Adam([self.raw], lr=1.0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=0.05)
        fw = torch.ones(self.F, device=self.dev) if frame_w is None else torch.tensor(frame_w, device=self.dev).float()
        for it in range(iters):
            opt.zero_grad()
            tot_iou = tot_ncc = 0.0
            for s in range(0, self.F, chunk):
                idx = torch.arange(s, min(self.F, s + chunk), device=self.dev)
                iou, ncc = self.data_terms(idx, w_ncc)
                loss = -((iou + w_ncc * ncc) * fw[idx]).sum()
                loss.backward()
                tot_iou += iou.sum().item(); tot_ncc += ncc.sum().item()
            sl = self.smooth_loss(lam)
            sl.backward()
            if fix_z:
                with torch.no_grad():
                    self.raw.grad[:, 3] = 0
            opt.step(); sched.step()
            if it % 25 == 0 or it == iters - 1:
                log(f"it {it:4d} iou {tot_iou / self.F:.4f} ncc {tot_ncc / self.F:.4f} smooth {sl.item():.3f}")
        return self

    def eval_frames(self, w_ncc=0.5, chunk=96):
        out_i, out_n = [], []
        with torch.no_grad():
            for s in range(0, self.F, chunk):
                idx = torch.arange(s, min(self.F, s + chunk), device=self.dev)
                iou, ncc = self.data_terms(idx, w_ncc)
                out_i.append(iou); out_n.append(ncc)
        return torch.cat(out_i).cpu().numpy(), torch.cat(out_n).cpu().numpy()

    def result(self):
        with torch.no_grad():
            th, uvz = self.frame_params()
            R, t = self.pose(th, uvz)
        T = np.tile(np.eye(4), (self.F, 1, 1))
        T[:, :3, :3] = R.cpu().numpy(); T[:, :3, 3] = t.cpu().numpy()
        return T, th.cpu().numpy(), uvz.cpu().numpy()
