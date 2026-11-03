"""Per-frame multi-hypothesis render-and-compare fitting of the bat pose (camera frame)."""
import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation

RES = 192


def rot_a_to_b(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        if c > 0: return np.eye(3)
        p = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        p = p - (p @ a) * a; p /= np.linalg.norm(p)
        return Rotation.from_rotvec(np.pi * p).as_matrix()
    s = np.linalg.norm(v)
    return Rotation.from_rotvec(v / s * np.arctan2(s, c)).as_matrix()


FLIP_X = np.diag([1.0, -1.0, -1.0])  # 180 deg about mesh x: swaps barrel/handle ends


def mirror_tilt(T):
    """Reflect the bat's long axis about the plane perpendicular to the viewing ray (same 2D image direction,
    opposite out-of-plane tilt)."""
    R = T[:3, :3]; t = T[:3, 3]
    a = R[:, 2]; r = t / np.linalg.norm(t)
    a2 = a - 2 * (a @ r) * r
    T2 = T.copy(); T2[:3, :3] = rot_a_to_b(a, a2) @ R
    return T2


def variants(T):
    out = [T]
    Tf = T.copy(); Tf[:3, :3] = T[:3, :3] @ FLIP_X; out.append(Tf)
    out.append(mirror_tilt(T)); out.append(mirror_tilt(Tf))
    return out


def convert_K(Tp, Kp, K, zscale):
    """Pose estimated for intrinsics Kp -> same pixel position for intrinsics K, depth scaled by zscale."""
    t = Tp[:3, 3]; u = Kp @ t; u = u / u[2]
    ray = np.linalg.inv(K) @ u
    t2 = ray / ray[2] * t[2] * zscale
    T = np.eye(4); T[:3, :3] = rot_a_to_b(t, t2) @ Tp[:3, :3]; T[:3, 3] = t2
    return T


def crop_window(mask, center_hint, lproj):
    ys, xs = np.nonzero(mask)
    if len(xs) > 30:
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ext = max(x1 - x0, y1 - y0)
    else:
        cx, cy = center_hint; ext = 0
    side = float(np.clip(max(1.3 * ext + 30, 2.0 * lproj), 120, 900))
    return cx - side / 2, cy - side / 2, side


def extend_mask(img, om, pm, H_rng=(95, 125), S_min=60, dil=2, bridge=50):
    """SAM2 masks often miss the dark-blue handle. Add saturated-blue pixels (bat colours; HSV hue 95-125,
    S>=60) that are not on the person and are connected to the mask, allowing connections through person pixels
    within `bridge` px of the mask (the gripping hand separates barrel and handle)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    B = (hsv[..., 0] >= H_rng[0]) & (hsv[..., 0] <= H_rng[1]) & (hsv[..., 1] >= S_min) & ~pm
    near = cv2.dilate(om.astype(np.uint8), np.ones((2 * bridge + 1, 2 * bridge + 1), np.uint8)) > 0
    U = (B | om | (pm & near)).astype(np.uint8)
    Ud = cv2.dilate(U, np.ones((2 * dil + 1, 2 * dil + 1), np.uint8))
    n, lab = cv2.connectedComponents(Ud)
    keep = np.unique(lab[om & (Ud > 0)]); keep = keep[keep > 0]
    return om | (B & np.isin(lab, keep))


def prep_frame(img, omask, pmask, win):
    """Crop+resize image/masks to RES x RES. Returns dict of float arrays."""
    x0, y0, side = win
    s = RES / side
    A = np.array([[s, 0, -x0 * s], [0, s, -y0 * s]], dtype=np.float64)
    I = cv2.warpAffine(img, A, (RES, RES), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
    M = cv2.warpAffine(omask.astype(np.float32), A, (RES, RES), flags=cv2.INTER_LINEAR)
    P = cv2.warpAffine(pmask.astype(np.float32), A, (RES, RES), flags=cv2.INTER_LINEAR)
    Mb = (M > 0.5).astype(np.uint8)
    # distance (in crop pixels) to object-or-person region: rendered pixels far from it are penalised
    tgt = ((M > 0.5) | (P > 0.5)).astype(np.uint8)
    if tgt.sum() == 0:
        D = np.zeros((RES, RES), np.float32)
    else:
        D = cv2.distanceTransform(1 - tgt, cv2.DIST_L2, 5).astype(np.float32)
    return dict(I=I[..., ::-1].astype(np.float32) / 255.0, M=M, P=P, D=D / RES)


def crop_K(K, win):
    x0, y0, side = win
    s = RES / side
    Kc = K.copy(); Kc[0, 2] -= x0; Kc[1, 2] -= y0; Kc[:2] *= s
    return Kc


def hat(w):
    O = torch.zeros(w.shape[0], 3, 3, device=w.device, dtype=w.dtype)
    O[:, 0, 1] = -w[:, 2]; O[:, 0, 2] = w[:, 1]; O[:, 1, 0] = w[:, 2]
    O[:, 1, 2] = -w[:, 0]; O[:, 2, 0] = -w[:, 1]; O[:, 2, 1] = w[:, 0]
    return O


def expm(w):
    return torch.linalg.matrix_exp(hat(w))


def losses(FR, R, t, Kc, data, w_col=1.0, w_dt=2.0):
    S, C, _ = FR.render(R, t, Kc, RES)
    M, P, D, I = data["M"], data["P"], data["D"], data["I"]
    V = 1 - P * (1 - M)
    inter = (S * M).sum((1, 2))
    union = (V * (S + M - S * M)).sum((1, 2)) + 1e-6
    l_iou = 1 - inter / union
    l_dt = (S * D).sum((1, 2)) / (S.sum((1, 2)) + 1.0)
    Wt = (S * M)[..., None]
    g = (Wt * I * C).sum((1, 2)) / ((Wt * C * C).sum((1, 2)) + 1e-6)  # (B,3) per-channel gain
    res = (Wt * (I - g[:, None, None, :] * C).abs()).sum((1, 2, 3)) / (3 * Wt.sum((1, 2, 3)) + 1e-6)
    tot = l_iou + w_dt * l_dt + w_col * res
    return tot, l_iou, res, l_dt


def fit_batch(FR, K, T0, wins, datas, iters=100, w_roll=0.5, dev="cuda", lr=(0.01, 0.003, 0.003), w_col=0.0):
    """T0 (B,4,4) camera-frame inits; wins/datas per element (frame data repeated per hypothesis).
    Returns refined poses and loss terms."""
    B = len(T0)
    Kc = np.stack([crop_K(K, w) for w in wins])
    Kt = torch.tensor(K, dtype=torch.float32, device=dev)
    R0 = torch.tensor(T0[:, :3, :3], dtype=torch.float32, device=dev)
    t0 = torch.tensor(T0[:, :3, 3], dtype=torch.float32, device=dev)
    u0 = (Kt @ t0.T).T; u0 = u0[:, :2] / u0[:, 2:]
    z0 = t0[:, 2]
    data = {k: torch.tensor(np.stack([d[k] for d in datas]), device=dev) for k in ["M", "P", "D", "I"]}
    w = torch.zeros(B, 3, device=dev, requires_grad=True)
    du = torch.zeros(B, 2, device=dev, requires_grad=True)
    dz = torch.zeros(B, device=dev, requires_grad=True)
    side = torch.tensor([wi[2] for wi in wins], dtype=torch.float32, device=dev)
    opt = torch.optim.Adam([{"params": [w], "lr": lr[0]}, {"params": [du], "lr": lr[1]}, {"params": [dz], "lr": lr[2]}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=0.0)
    Kinv = torch.linalg.inv(Kt)

    def pose():
        R = R0 @ expm(w)
        u = u0 + du * side[:, None]  # du in units of crop side
        z = z0 * torch.exp(dz)
        ray = (Kinv @ torch.cat([u, torch.ones_like(u[:, :1])], 1).T).T
        return R, ray * z[:, None]

    for it in range(iters):
        R, t = pose()
        tot, *_ = losses(FR, R, t, Kc, data, w_col=w_col)
        loss = tot.sum() + w_roll * (w[:, 2] ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        R, t = pose()
        tot, l_iou, l_col, l_dt = losses(FR, R, t, Kc, data)
    T = np.tile(np.eye(4), (B, 1, 1))
    T[:, :3, :3] = R.cpu().numpy(); T[:, :3, 3] = t.cpu().numpy()
    return T, tot.cpu().numpy(), l_iou.cpu().numpy(), l_col.cpu().numpy(), l_dt.cpu().numpy()


def score_batch(FR, K, T, wins, datas, dev="cuda"):
    Kc = np.stack([crop_K(K, w) for w in wins])
    data = {k: torch.tensor(np.stack([d[k] for d in datas]), device=dev) for k in ["M", "P", "D", "I"]}
    with torch.no_grad():
        R = torch.tensor(T[:, :3, :3], dtype=torch.float32, device=dev)
        t = torch.tensor(T[:, :3, 3], dtype=torch.float32, device=dev)
        out = losses(FR, R, t, Kc, data)
    return [o.cpu().numpy() for o in out]
