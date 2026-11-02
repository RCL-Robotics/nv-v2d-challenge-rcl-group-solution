"""Pose models constrained by contact with the table plane."""
import numpy as np
import torch
import torch.nn.functional as Fnn
from fitcore import so3_exp, color_loss
from hyp import soft_iou_loss, rot_a_to_b


def axis_rot_t(axis, ang):
    """axis (3,) torch unit, ang (B,) -> (B,3,3)"""
    return so3_exp(axis[None, :] * ang[:, None])


class PlaneModel:
    """Object resting with mesh facet normal `facet_n` (mesh frame, outward) on plane n.x + d = 0 (camera frame,
    n pointing up/out of the table). Free per-frame params: yaw psi about n, in-plane position (a, b)."""

    def __init__(self, verts, facet_n, n, d, dev="cuda"):
        facet_n = facet_n / np.linalg.norm(facet_n)
        n = n / np.linalg.norm(n)
        self.s0 = float(np.max(verts @ facet_n))  # support of the facet
        self.R0 = rot_a_to_b(facet_n, -n)
        e1 = np.cross(n, [0, 0, 1.0]); e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        self.n, self.d, self.e1, self.e2 = n, d, e1, e2
        T = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)
        self.nt, self.R0t, self.e1t, self.e2t = T(n), T(self.R0), T(e1), T(e2)
        self.dev = dev

    def pose_t(self, psi, a, b):
        R = axis_rot_t(self.nt, psi) @ self.R0t[None]
        t = (self.s0 - self.d) * self.nt[None] + a[:, None] * self.e1t[None] + b[:, None] * self.e2t[None]
        return R, t

    def pose(self, psi, a, b):
        R, t = self.pose_t(*[torch.tensor(np.atleast_1d(x), dtype=torch.float32, device=self.dev) for x in (psi, a, b)])
        return R.cpu().numpy().astype(np.float64), t.cpu().numpy().astype(np.float64)

    def from_pose(self, R, t):
        """Project arbitrary poses onto the model: yaw from the facet-aligned rotation, (a,b) from the centre."""
        psi = []
        for Ri in R:
            # rotation that would map R0 to Ri, measured about n
            M = Ri @ self.R0.T
            x = M @ self.e1
            x = x - (x @ self.n) * self.n
            psi.append(np.arctan2(x @ self.e2, x @ self.e1))
        return np.array(psi), t @ self.e1, t @ self.e2


def load_frames(data, frames, hw, dev, keys=("mask", "person", "rgb")):
    full = data["mask"].shape[1:3]
    out = {}
    for k in keys:
        x = torch.tensor(data[k][frames], device=dev)
        if tuple(hw) != tuple(full):
            x = Fnn.interpolate(x[:, None] if x.dim() == 3 else x.permute(0, 3, 1, 2), size=hw, mode="area")
            x = x[:, 0] if x.shape[1] == 1 else x.permute(0, 2, 3, 1)
        out[k] = x.contiguous()
    return out


def fit_plane_model(fit, model, frames, data, psi0, a0, b0, sigma=None, iters=80, lr=0.01, w_col=0.1, hw=(160, 160),
                    w_smooth=0.0, opt_sigma=False):
    dev = fit.dev
    P = fit.proj_mats(data["roi"][frames], hw)
    D = load_frames(data, frames, hw, dev)
    valid = 1 - D["person"] * (1 - D["mask"])
    T = lambda x: torch.tensor(np.asarray(x, np.float64), dtype=torch.float32, device=dev).clone().requires_grad_(True)
    psi, a, b = T(psi0), T(a0), T(b0)
    sg = T(1.0 if sigma is None else sigma)
    params = [{"params": [psi], "lr": lr}, {"params": [a, b], "lr": lr * 0.3}]
    if opt_sigma:
        params.append({"params": [sg], "lr": 0.003})
    opt = torch.optim.Adam(params)
    use_rgb = w_col > 0
    for it in range(iters + 1):
        R, t = model.pose_t(psi, a, b)
        r, col, _ = fit.render(R, t, P, hw, rgb=use_rgb or it == iters, sigma=sg)
        sil = soft_iou_loss(r, D["mask"], valid)
        cl = color_loss(col, r, D["rgb"], D["mask"], valid) if col is not None else torch.zeros_like(sil)
        if it == iters:
            break
        loss = (sil + w_col * cl).sum()
        if w_smooth > 0 and len(frames) > 2:
            loss = loss + w_smooth * (((psi[2:] - 2 * psi[1:-1] + psi[:-2]) ** 2).sum()
                                      + 100 * ((a[2:] - 2 * a[1:-1] + a[:-2]) ** 2).sum()
                                      + 100 * ((b[2:] - 2 * b[1:-1] + b[:-2]) ** 2).sum())
        opt.zero_grad()
        loss.backward()
        opt.step()
    g = lambda x: x.detach().cpu().numpy().astype(np.float64)
    return g(psi), g(a), g(b), g(sil), g(cl), float(sg.item())
