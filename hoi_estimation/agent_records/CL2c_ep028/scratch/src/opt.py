import numpy as np, torch, pickle, json
from core import *
from scipy.spatial.transform import Rotation as Rot
INP = '/work/inputs'
cam, rec, gp = load_inputs(INP)
K = np.array(cam['K']); Kp = np.array(cam['K_pipeline']); Twc = np.array(cam['T_world_cam']); Tcw = np.linalg.inv(Twc)
F = rec['T_world_mesh'].shape[0]
pts, orange, nrm, mesh = sample_mesh_points(INP)
c_mesh = mesh.bounds.mean(0)
class Bank: pass
def load_bank_cache():
    b = Bank()
    b.dtm = np.load('/work/scratch/fb_dtm.npy', mmap_mode='r'); b.dto = np.load('/work/scratch/fb_dto.npy', mmap_mode='r'); b.occ = np.load('/work/scratch/fb_occ.npy', mmap_mode='r')
    d = pickle.load(open('/work/scratch/fb_pts.pkl', 'rb')); b.__dict__.update(d); return b
# plane in camera frame: n_c . x + d_c = 0 (positive = above floor)
n_w = np.array(gp['plane'][:3]); d_w = gp['plane'][3]
n_c = Twc[:3, :3].T @ n_w; d_c = d_w + n_w @ Twc[:3, 3]
kp = rec['human_keypoints_world'].astype(np.float64)
hands_w = np.stack([kp[:, 21:42].mean(1), kp[:, 42:63].mean(1)], 1)  # F,2,3
hands_c = np.einsum('ij,fkj->fki', Tcw[:3, :3], hands_w) + Tcw[:3, 3]
contact = rec['contact_prob_right_left'].astype(np.float64)
handle_pts = np.stack([np.zeros(40), np.zeros(40), np.linspace(-0.40, 1.0, 40)], 1) - c_mesh  # handle axis samples (centred)

def T_to_Rt(T):
    T = np.asarray(T, np.float64); R = T[..., :3, :3]; t = np.einsum('...ij,j->...i', R, c_mesh) + T[..., :3, 3]
    return R, t
def Rt_to_T(R, t):
    T = np.zeros(R.shape[:-2] + (4, 4)); T[..., :3, :3] = R; T[..., :3, 3] = t - np.einsum('...ij,j->...i', R, c_mesh); T[..., 3, 3] = 1; return T

T_base_c = Tcw @ rec['T_world_mesh'].astype(np.float64)
Rb, tb = T_to_Rt(T_base_c)
def fp_rescaled():
    """FoundationPose camera-frame poses, centre moved along its K_pipeline ray onto the K ray at baseline centre depth"""
    R, t = T_to_Rt(rec['T_cam_mesh_foundationpose'])
    uv = (Kp @ t.T).T; uv = uv[:, :2] / uv[:, 2:]
    ray = (np.linalg.inv(K) @ np.c_[uv, np.ones(len(uv))].T).T
    return R, ray * tb[:, 2:3]

def flips():
    return [np.eye(3), np.diag([1., -1, -1]), np.diag([-1., 1, -1]), np.diag([-1., -1, 1])]  # about mesh x, y, z (centre)

def tilt_reflect(R, t):
    """reflect the handle axis across plane perpendicular to the viewing ray (depth-tilt ambiguity)"""
    a = R[:, :, 2]; v = t / np.linalg.norm(t, axis=1, keepdims=True)
    a2 = a - 2 * (a * v).sum(1, keepdims=True) * v
    out = []
    for i in range(len(R)):
        rv = np.cross(a[i], a2[i]); s = np.linalg.norm(rv); cth = a[i] @ a2[i]
        if s < 1e-8: out.append(R[i]); continue
        ang = np.arctan2(s, cth); out.append(Rot.from_rotvec(rv / s * ang).as_matrix() @ R[i])
    return np.array(out)

class Problem:
    def __init__(self, bank):
        self.bank = bank
        self.sc = Scorer(bank, pts, orange, nrm, K, c_mesh)
        self.Pall = self.sc.P
        self.handle = torch.tensor(handle_pts, dtype=torch.float32, device=dev)
        self.n_c = torch.tensor(n_c, dtype=torch.float32, device=dev); self.d_c = float(d_c)
    def costs(self, idx, R, t, maps, mp, mpn, op, opn, zref, sig=10., w=None):
        w = w or {}
        ct = self.sc.image_terms(R, t, maps, mp, mpn, op, opn, sig=sig)
        # floor penetration
        Xc = torch.einsum('bgij,nj->bgni', R, self.Pall) + t[:, :, None]
        h = Xc @ self.n_c + self.d_c
        ct['floor'] = (torch.relu(-h - 0.03) ** 2).mean(-1) * 1e3
        # hand: distance hand centre -> handle axis
        Xh = torch.einsum('bgij,nj->bgni', R, self.handle) + t[:, :, None]
        hc = torch.tensor(hands_c[idx], dtype=torch.float32, device=dev)
        B, G = R.shape[:2]
        dh = torch.cdist(hc[:, None].expand(B, G, 2, 3).reshape(B * G, 2, 3), Xh.reshape(B * G, -1, 3)).min(-1).values.reshape(B, G, 2)
        cp = torch.tensor(contact[idx], dtype=torch.float32, device=dev)[:, None]
        ct['hand'] = (cp * (dh ** 2 / (dh ** 2 + 0.15 ** 2))).sum(-1)
        ct['depth'] = ((t[..., 2] - zref) / 0.15) ** 2
        return ct
    def total(self, ct, w):
        return sum(w.get(k, 0) * v for k, v in ct.items())

W_DEF = dict(m2i=0.5, i2m=1.0, o2i=0.5, i2o=0.5, outside=0.5, floor=1.0, hand=0.3, depth=0.02)

def optimize(prob, idx, R0, t0, zref, steps=120, sig_sched=(20., 8.), lr=(0.03, 0.02), w=W_DEF, fix_depth=False):
    idx = np.asarray(idx)
    maps = prob.sc.maps(idx)
    mp, mpn = pad_pts([prob.bank.mpts[i] for i in idx], 600)
    op, opn = pad_pts([prob.bank.opts[i] for i in idx], 300)
    R0t = torch.tensor(R0, dtype=torch.float32, device=dev); t0t = torch.tensor(t0, dtype=torch.float32, device=dev)
    zr = torch.tensor(zref, dtype=torch.float32, device=dev)
    om = torch.zeros(R0t.shape[:2] + (3,), device=dev, requires_grad=True); dt = torch.zeros(R0t.shape[:2] + (3,), device=dev, requires_grad=True)
    opt = torch.optim.Adam([{'params': [om], 'lr': lr[0]}, {'params': [dt], 'lr': lr[1]}])
    for k in range(steps):
        sig = sig_sched[0] * (sig_sched[1] / sig_sched[0]) ** (k / max(1, steps - 1))
        R = so3_exp(om) @ R0t; t = t0t + dt
        ct = prob.costs(idx, R, t, maps, mp, mpn, op, opn, zr, sig=sig)
        L = prob.total(ct, w)
        opt.zero_grad(); L.sum().backward(); opt.step()
    with torch.no_grad():
        R = so3_exp(om) @ R0t; t = t0t + dt
        ct = prob.costs(idx, R, t, maps, mp, mpn, op, opn, zr, sig=sig_sched[1])
        L = prob.total(ct, w)
    return R.cpu().numpy().astype(np.float64), t.cpu().numpy().astype(np.float64), L.cpu().numpy(), {k: v.cpu().numpy() for k, v in ct.items()}

def evaluate(prob, idx, R, t, zref, sig=8., w=W_DEF):
    idx = np.asarray(idx)
    with torch.no_grad():
        maps = prob.sc.maps(idx)
        mp, mpn = pad_pts([prob.bank.mpts[i] for i in idx], 600)
        op, opn = pad_pts([prob.bank.opts[i] for i in idx], 300)
        ct = prob.costs(idx, torch.tensor(R, dtype=torch.float32, device=dev), torch.tensor(t, dtype=torch.float32, device=dev), maps, mp, mpn, op, opn, torch.tensor(zref, dtype=torch.float32, device=dev), sig=sig)
        return prob.total(ct, w).cpu().numpy(), {k: v.cpu().numpy() for k, v in ct.items()}
