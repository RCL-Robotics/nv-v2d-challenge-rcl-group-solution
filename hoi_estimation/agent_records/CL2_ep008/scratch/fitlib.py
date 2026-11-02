import numpy as np, torch, h5py, nvdiffrast.torch as dr
from common import *
from render import Renderer
import roma

def so3_exp(w):
    return roma.rotvec_to_rotmat(w)

class CropRenderer(Renderer):
    """renders into a crop window [x0,y0,w,h] of the full-res image."""
    def render_crop(self, T, K, box):
        x0, y0, w, h = box
        Kc = K.clone(); Kc[0, 2] -= x0; Kc[1, 2] -= y0
        self.W, self.H = w, h
        clip, Z = self.clip(T, Kc)
        rast, _ = dr.rasterize(self.glctx, clip, self.f, (h, w))
        ones = torch.ones_like(clip[..., :1])
        m, _ = dr.interpolate(ones, rast, self.f)
        m = dr.antialias(m, rast, clip, self.f)
        return m[..., 0], rast

def load_masks(idxs):
    f = h5py.File(f'{INP}/masks.h5', 'r')
    O = np.stack([f['object'][i] for i in idxs]); P = np.stack([f['person'][i] for i in idxs])
    return O, P

def pose_from_params(R0, t0, c, w, dt):
    # R0 (B,3,3), t0 (B,3), c (3,) mesh center; rotation perturbation about the object's center, in camera frame
    Rw = so3_exp(w)
    R = Rw @ R0
    t = (Rw @ (R0 @ c[:, None]))[..., 0] * -1 + (R0 @ c[:, None])[..., 0] + t0 + dt
    T = torch.zeros(R.shape[0], 4, 4, device=R.device); T[:, :3, :3] = R; T[:, :3, 3] = t; T[:, 3, 3] = 1
    return T

def ms_loss(r, m, w, scales=(8, 4, 2, 1)):
    # r,m,w: (B,H,W). multi-scale weighted L2
    import torch.nn.functional as Fn
    tot = 0
    for s in scales:
        if s > 1:
            rs = Fn.avg_pool2d(r[:, None] * w[:, None], s)[:, 0]; ms = Fn.avg_pool2d(m[:, None] * w[:, None], s)[:, 0]
            ws = Fn.avg_pool2d(w[:, None], s)[:, 0]
            rs = rs / ws.clamp(min=1e-3); ms = ms / ws.clamp(min=1e-3); ws = (ws > 0.5).float()
        else:
            rs, ms, ws = r, m, w
        tot = tot + ((rs - ms) ** 2 * ws).sum((1, 2)) / ws.sum((1, 2)).clamp(min=1)
    return tot / len(scales)

def soft_iou(r, m, w):
    i = (r * m * w).sum((1, 2)); u = ((r + m - r * m) * w).sum((1, 2))
    return i / u.clamp(min=1)
