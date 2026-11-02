"""Phase segmentation and robust temporal smoothing of object poses."""
import numpy as np, torch
from scipy.spatial.transform import Rotation as Rot

def chordal_mean(Rs, w=None):
    w = np.ones(len(Rs)) if w is None else np.asarray(w)
    M = (Rs * w[:, None, None]).sum(0)
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt

def static_pose_cam(Tc):
    """One camera-frame pose for a static phase: chordal mean rotation; translation = median depth along the
    median viewing ray (keeps the baseline's median image position and median depth)."""
    Rm = chordal_mean(Tc[:, :3, :3])
    t = Tc[:, :3, 3]; z = np.median(t[:, 2]); u = np.median(t[:, 0] / t[:, 2]); v = np.median(t[:, 1] / t[:, 2])
    T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = [u * z, v * z, z]
    return T

def mask_iou_vis(a, b, vis):
    a = a & vis; b = b & vis
    return (a & b).sum() / max((a | b).sum(), 1)

def detect_static_phases(obj_masks, person_masks, thr=0.85, ref_n=10):
    """Start phase: longest prefix whose object mask (outside the person) overlaps the mask of the first frames with
    IoU >= thr; end phase: same with the last frames. Returns (start_end_exclusive, end_start)."""
    F = len(obj_masks)
    ref0 = np.median(np.stack([obj_masks[i] for i in range(ref_n)]), 0) > 0.5
    refE = np.median(np.stack([obj_masks[F - 1 - i] for i in range(ref_n)]), 0) > 0.5
    s_end = 0
    for i in range(F):
        if mask_iou_vis(obj_masks[i], ref0, ~person_masks[i]) < thr: break
        s_end = i + 1
    e_start = F
    for i in range(F - 1, -1, -1):
        if mask_iou_vis(obj_masks[i], refE, ~person_masks[i]) < thr: break
        e_start = i
    return s_end, e_start

def smooth_se3(Tobs, w, fixed=None, lam_rot=30.0, lam_tr=30.0, huber_deg=6.0, huber_m=0.04, iters=1500, seed=0):
    """Robust smoothing of a pose sequence (camera frame).
    minimise sum_t w_t huber(|log(R_obs^T R_t)|) + lam * sum_t |acc_t|^2 (rotation, translation separately)
    fixed: dict frame -> 4x4 pose held exactly (static phases). Returns (F,4,4)."""
    torch.manual_seed(seed)
    F = len(Tobs)
    R0 = torch.tensor(Tobs[:, :3, :3], dtype=torch.float64); t0 = torch.tensor(Tobs[:, :3, 3], dtype=torch.float64)
    wt = torch.tensor(w, dtype=torch.float64)
    fix = np.zeros(F, bool); Rf = R0.clone(); tf = t0.clone()
    if fixed:
        for k, T in fixed.items():
            fix[k] = True; Rf[k] = torch.tensor(T[:3, :3]); tf[k] = torch.tensor(T[:3, 3])
    fixt = torch.tensor(fix)
    d = torch.zeros(F, 3, dtype=torch.float64, requires_grad=True)
    p = tf.clone().requires_grad_(True)
    opt = torch.optim.Adam([d, p], lr=3e-3)
    hr = np.radians(huber_deg)
    def expm(v):
        th = v.norm(dim=-1, keepdim=True).clamp(min=1e-12); k = v / th
        K = torch.zeros(v.shape[0], 3, 3, dtype=v.dtype)
        K[:, 0, 1] = -k[:, 2]; K[:, 0, 2] = k[:, 1]; K[:, 1, 0] = k[:, 2]; K[:, 1, 2] = -k[:, 0]; K[:, 2, 0] = -k[:, 1]; K[:, 2, 1] = k[:, 0]
        s = torch.sin(th)[..., None]; c = torch.cos(th)[..., None]
        return torch.eye(3, dtype=v.dtype)[None] + s * K + (1 - c) * K @ K
    def logm(Rm):
        c = ((Rm.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
        th = torch.acos(c)
        v = torch.stack([Rm[:, 2, 1] - Rm[:, 1, 2], Rm[:, 0, 2] - Rm[:, 2, 0], Rm[:, 1, 0] - Rm[:, 0, 1]], -1)
        return v * (th / (2 * torch.sin(th).clamp(min=1e-9)))[:, None]
    def huber(x, s):
        return torch.where(x < s, 0.5 * x ** 2, s * (x - 0.5 * s))
    for it in range(iters):
        R = torch.where(fixt[:, None, None], Rf, Rf @ expm(d))
        t = torch.where(fixt[:, None], tf, p)
        er = logm(R0.transpose(1, 2) @ R).norm(dim=-1)
        et = (t - t0).norm(dim=-1)
        data = (wt * huber(er, hr)).sum() / hr ** 2 + (wt * huber(et, huber_m)).sum() / huber_m ** 2
        vel = logm(R[:-1].transpose(1, 2) @ R[1:])           # body-frame angular increments
        acc_r = ((vel[1:] - vel[:-1]) ** 2).sum() / hr ** 2
        acc_t = (((t[2:] - 2 * t[1:-1] + t[:-2])) ** 2).sum() / huber_m ** 2
        loss = data + lam_rot * acc_r + lam_tr * acc_t
        opt.zero_grad(); loss.backward(); opt.step()
    R = torch.where(fixt[:, None, None], Rf, Rf @ expm(d)).detach().numpy(); t = torch.where(fixt[:, None], tf, p).detach().numpy()
    T = np.tile(np.eye(4), (F, 1, 1)); T[:, :3, :3] = R; T[:, :3, 3] = t
    return T
