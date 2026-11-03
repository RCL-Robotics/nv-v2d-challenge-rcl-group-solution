import numpy as np, torch, nvdiffrast.torch as dr, h5py, cv2, json
import sys; sys.path.insert(0, '/work/scratch')
from common import *

DEV = 'cuda'
_glctx = None
def ctx():
    global _glctx
    if _glctx is None:
        _glctx = dr.RasterizeCudaContext()
    return _glctx

class Mesh:
    def __init__(self):
        m = mesh()
        v = np.asarray(m.vertices); f = np.asarray(m.faces)
        # merge duplicate vertices for silhouette
        uq, inv = np.unique(v.round(7), axis=0, return_inverse=True)
        self.v = torch.tensor(uq, dtype=torch.float32, device=DEV)
        self.f = torch.tensor(inv.reshape(-1)[f], dtype=torch.int32, device=DEV)

def axang_to_R(w):
    # w (B,3) -> (B,3,3), differentiable
    th = torch.sqrt((w * w).sum(-1, keepdim=True) + 1e-12)
    k = w / th
    K = torch.zeros(w.shape[0], 3, 3, device=w.device, dtype=w.dtype)
    K[:, 0, 1] = -k[:, 2]; K[:, 0, 2] = k[:, 1]; K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]; K[:, 2, 0] = -k[:, 1]; K[:, 2, 1] = k[:, 0]
    th = th[..., None]
    return torch.eye(3, device=w.device, dtype=w.dtype)[None] + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)

def render_sil(M, R, t, K, x0y0, S, n=0.1, f=50.):
    """R (B,3,3), t (B,3) camera-from-mesh; K 3x3 np; x0y0 (B,2) crop origins; S crop size -> alpha (B,S,S)"""
    B = R.shape[0]
    P = torch.einsum('bij,vj->bvi', R, M.v) + t[:, None, :]
    Z = P[..., 2].clamp(min=1e-3)
    u = K[0, 0] * P[..., 0] / Z + K[0, 2] - x0y0[:, None, 0]
    v = K[1, 1] * P[..., 1] / Z + K[1, 2] - x0y0[:, None, 1]
    xn = (2 * u + 1) / S - 1; yn = (2 * v + 1) / S - 1
    a = (f + n) / (f - n); b = -2 * f * n / (f - n)
    clip = torch.stack([xn * Z, yn * Z, a * Z + b, Z], -1).contiguous()
    rast, _ = dr.rasterize(ctx(), clip, M.f, (S, S))
    alpha = (rast[..., 3:] > 0).float()
    alpha = dr.antialias(alpha, rast, clip, M.f)
    return alpha[..., 0], rast

class Masks:
    def __init__(self):
        self.h = h5py.File(INP + '/masks.h5', 'r')
    def crop(self, i, x0, y0, S):
        o = self.h['object'][i, y0:y0 + S, x0:x0 + S]
        p = self.h['person'][i, y0:y0 + S, x0:x0 + S]
        return o, p

def read_frames(idx):
    idx = sorted(set(idx)); out = {}
    cap = cv2.VideoCapture(INP + '/video.mp4'); i = 0; j = 0
    while j < len(idx):
        ok, im = cap.read()
        if not ok: break
        if i == idx[j]:
            out[i] = im; j += 1
        i += 1
    return out

def gblur(x, sigma):
    if sigma <= 0: return x
    k = int(3 * sigma) * 2 + 1
    xs = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-xs ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    x = x[:, None]
    x = torch.nn.functional.conv2d(x, g.view(1, 1, 1, k), padding=(0, k // 2))
    x = torch.nn.functional.conv2d(x, g.view(1, 1, k, 1), padding=(k // 2, 0))
    return x[:, 0]

def fit_sil(M, K, R0, uv0, z0, obj, valid, x0y0, S, iters=(60, 60, 60), sigmas=(4, 2, 0.7), fit_rot=True, fit_z=True, lr=None):
    """R0 (B,3,3) np, uv0 (B,2), z0 (B,), obj/valid (B,S,S) float tensors. Returns R,uv,z,loss"""
    B = R0.shape[0]
    R0t = torch.tensor(R0, dtype=torch.float32, device=DEV)
    w = torch.zeros(B, 3, device=DEV, requires_grad=True)
    uv = torch.tensor(uv0, dtype=torch.float32, device=DEV).clone().requires_grad_(True)
    lz = torch.tensor(np.log(z0), dtype=torch.float32, device=DEV).clone().requires_grad_(True)
    Kt = torch.tensor(K, dtype=torch.float32, device=DEV)
    xy = torch.tensor(x0y0, dtype=torch.float32, device=DEV)
    params = [{'params': [uv], 'lr': 0.5}]
    if fit_z: params.append({'params': [lz], 'lr': 0.003})
    if fit_rot: params.append({'params': [w], 'lr': 0.01})
    opt = torch.optim.Adam(params)
    def pose():
        z = torch.exp(lz)
        t = torch.stack([(uv[:, 0] - Kt[0, 2]) / Kt[0, 0] * z, (uv[:, 1] - Kt[1, 2]) / Kt[1, 1] * z, z], -1)
        R = axang_to_R(w) @ R0t
        return R, t
    for it_n, sg in zip(iters, sigmas):
        ob = gblur(obj, sg); 
        for it in range(it_n):
            R, t = pose()
            a, _ = render_sil(M, R, t, K, xy, S)
            ab = gblur(a, sg)
            l = (((ab - ob) ** 2) * valid).sum((1, 2)) / (obj.sum((1, 2)) + 50)
            opt.zero_grad(); l.sum().backward(); opt.step()
    with torch.no_grad():
        R, t = pose()
        a, _ = render_sil(M, R, t, K, xy, S)
        ab = a > 0.5; ob = obj > 0.5; vb = valid > 0.5
        iou = ((ab & ob & vb).sum((1, 2)).float() / ((ab | ob) & vb).sum((1, 2)).clamp(min=1).float())
    return R.detach().cpu().numpy().astype(float), t.detach().cpu().numpy().astype(float), iou.cpu().numpy(), a.detach().cpu().numpy()
