"""Orientation hypotheses and per-frame silhouette fitting."""
import numpy as np
import torch
import torch.nn.functional as Fnn
from fitcore import so3_exp, color_loss

SOLE_N = np.array([-0.9821, 0.0797, 0.1705])   # largest convex-hull facet (the soleplate), mesh frame
HEEL_N = np.array([0.1392, -0.0064, -0.9902])  # heel-rest facet (iron stood up), mesh frame


def rot_a_to_b(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = a @ b
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        p = np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0])
        p /= np.linalg.norm(p)
        return 2 * np.outer(p, p) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def axis_angle(axis, ang):
    import cv2
    axis = np.asarray(axis, float)
    return cv2.Rodrigues((axis / np.linalg.norm(axis) * ang).reshape(3, 1))[0]


def rest_rotations(up_c, down_mesh, yaws):
    """Rotations (camera<-mesh) that put mesh direction `down_mesh` along -up_c, for each yaw about up_c."""
    R0 = rot_a_to_b(down_mesh, -up_c)
    return np.array([axis_angle(up_c, y) @ R0 for y in yaws])


def hypothesis_set(up_c, n_yaw=72, tilts=(0.0, 0.25, 0.5, 0.75, 1.0)):
    yaws = np.linspace(0, 2 * np.pi, n_yaw, endpoint=False)
    Rs, tags = [], []
    for t in tilts:
        d = (1 - t) * SOLE_N + t * HEEL_N
        for k, R in enumerate(rest_rotations(up_c, d, yaws)):
            Rs.append(R)
            tags.append((t, yaws[k]))
    return np.array(Rs), tags


def soft_iou_loss(r, m, valid):
    inter = (valid * r * m).sum((1, 2))
    union = (valid * (r + m - r * m)).sum((1, 2)).clamp_min(1e-6)
    return 1 - inter / union


def init_translation(fit, R, roi_box, K, mesh_v, center_mesh):
    """Place mesh so its projected centre is at the mask-box centre and its projected extent matches the box."""
    x0, y0, x1, y1 = roi_box
    u, v = (x0 + x1) / 2, (y0 + y1) / 2
    size_px = max(x1 - x0, y1 - y0)
    out = []
    for Rm in R:
        p = mesh_v @ Rm.T
        ext = max(np.ptp(p[:, 0]), np.ptp(p[:, 1]))
        z = K[0, 0] * ext / max(size_px, 5)
        ray = np.linalg.inv(K) @ [u, v, 1]
        c = Rm @ center_mesh
        out.append(ray * z - c)
    return np.array(out)


def fit_batch(fit, R0, t0, frames, data, iters=60, lr_t=0.01, lr_r=0.02, opt_rot=True, w_col=0.0,
              extra_loss=None, hw=None):
    """Optimise B poses (R0 (B,3,3), t0 (B,3)) to frames (list of frame ids, len B) by soft IoU on silhouettes.
    Returns R, t, final loss per item."""
    dev = fit.dev
    full = data["mask"].shape[1:3]
    hw = full if hw is None else hw
    P = fit.proj_mats(data["roi"][frames], hw)
    def ld(k):
        x = torch.tensor(data[k][frames], device=dev)
        if tuple(hw) != tuple(full):
            x = Fnn.interpolate(x[:, None] if x.dim() == 3 else x.permute(0, 3, 1, 2), size=hw, mode="area")
            x = x[:, 0] if x.shape[1] == 1 else x.permute(0, 2, 3, 1)
        return x.contiguous()
    m = ld("mask")
    pers = ld("person")
    img = ld("rgb") if w_col > 0 else None
    valid = 1 - pers * (1 - m)
    Rb = torch.tensor(R0, dtype=torch.float32, device=dev)
    t = torch.tensor(t0, dtype=torch.float32, device=dev).clone().requires_grad_(True)
    w = torch.zeros(len(frames), 3, device=dev, requires_grad=True)
    params = [{"params": [t], "lr": lr_t}]
    if opt_rot:
        params.append({"params": [w], "lr": lr_r})
    opt = torch.optim.Adam(params)
    for it in range(iters):
        Rm = so3_exp(w) @ Rb if opt_rot else Rb
        r, col, _ = fit.render(Rm, t, P, hw, rgb=w_col > 0)
        loss = soft_iou_loss(r, m, valid)
        if w_col > 0:
            loss = loss + w_col * color_loss(col, r, img, m, valid)
        tot = loss.sum()
        if extra_loss is not None:
            tot = tot + extra_loss(Rm, t)
        opt.zero_grad()
        tot.backward()
        opt.step()
    with torch.no_grad():
        Rm = so3_exp(w) @ Rb if opt_rot else Rb
        r, col, _ = fit.render(Rm, t, P, hw, rgb=True)
        if img is None:
            img = ld("rgb")
        sil = soft_iou_loss(r, m, valid)
        cl = color_loss(col, r, img, m, valid)
    return (Rm.detach().cpu().numpy().astype(np.float64), t.detach().cpu().numpy().astype(np.float64),
            sil.cpu().numpy(), cl.cpu().numpy())
