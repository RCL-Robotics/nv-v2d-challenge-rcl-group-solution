# Joint silhouette refinement of all frames with temporal smoothness (experiment)
import sys, time; sys.path.insert(0, '/work/outputs')
from common import *
import torch.nn.functional as Fn
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--init', default='base'); ap.add_argument('--out', default='/work/scratch/opt1.npz')
ap.add_argument('--iters', type=int, default=60); ap.add_argument('--ws', type=float, default=1.0)
ap.add_argument('--frames', default='0:993'); ap.add_argument('--static', default='0:98,316:390,937:993')
ap.add_argument('--lr', type=float, default=3e-3)
a = ap.parse_args()
torch.manual_seed(0)
dev = 'cuda'
D = load_inputs('/work/inputs'); mesh = load_mesh('/work/inputs')
Tw = D['rec']['T_world_mesh'].astype(np.float64) if a.init == 'base' else np.load(a.init)['T_world_mesh']
Tc0 = np.linalg.inv(D['Twc']) @ Tw
F = len(Tw)
s0, s1 = map(int, a.frames.split(':'))
fr = np.arange(s0, s1)
M = torch.tensor(np.load('/work/scratch/mobj_half.npy')[s0:s1], device=dev)
P = torch.tensor(np.load('/work/scratch/mper_half.npy')[s0:s1], device=dev)
K = D['K'].copy(); K[:2] *= 0.5
R = Renderer(mesh, K, 576, 768)
c = 0.5 * (mesh['v'].min(0) + mesh['v'].max(0))
R.v = R.v - torch.tensor(c, dtype=torch.float32, device=dev)  # centred verts
# units: static phases share one pose
unit = np.arange(len(fr))
statics = [tuple(map(int, s.split(':'))) for s in a.static.split(',')] if a.static else []
nu = len(fr)
R0 = Tc0[fr, :3, :3].copy(); p0 = (Tc0[fr, :3, :3] @ c) + Tc0[fr, :3, 3]
for (u0, u1) in statics:
    idx = np.where((fr >= u0) & (fr < u1))[0]
    if len(idx) == 0: continue
    # chordal mean rotation + median position
    U, _, Vt = np.linalg.svd(R0[idx].sum(0)); Rm = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    R0[idx] = Rm; p0[idx] = np.median(p0[idx], 0)
    unit[idx] = idx[0]
R0t = torch.tensor(R0, dtype=torch.float32, device=dev); p0t = torch.tensor(p0, dtype=torch.float32, device=dev)
unit_t = torch.tensor(unit, device=dev)
om = torch.zeros(nu, 3, device=dev, requires_grad=True)
dp = torch.zeros(nu, 3, device=dev, requires_grad=True)
opt = torch.optim.Adam([{'params': [om], 'lr': a.lr}, {'params': [dp], 'lr': a.lr * 0.5}])


def rodrigues(w):
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = w / th
    Kx = torch.zeros(w.shape[0], 3, 3, device=w.device)
    Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    Kx[:, 1, 2], Kx[:, 2, 0], Kx[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    th = th[..., None]
    return torch.eye(3, device=w.device) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


def poses():
    Rf = rodrigues(om[unit_t]) @ R0t
    pf = p0t + dp[unit_t]
    return Rf, pf


def sil_loss(idx, Rf, pf):
    B = len(idx)
    T = torch.zeros(B, 4, 4, device=dev); T[:, :3, :3] = Rf[idx]; T[:, :3, 3] = pf[idx]; T[:, 3, 3] = 1
    vh = torch.cat([R.v, torch.ones_like(R.v[:, :1])], 1)
    vc = torch.einsum('bij,nj->bni', T, vh)
    clip = torch.einsum('ij,bnj->bni', R.proj(R.near, R.far), vc).contiguous()
    rast, _ = dr.rasterize(R.glctx, clip, R.f, (R.H, R.W))
    soft = dr.antialias((rast[..., 3:] > 0).float(), rast, clip, R.f)[..., 0]
    m = M[idx].float(); valid = 1 - (P[idx] & ~M[idx]).float()
    l = (((soft - m) ** 2) * valid).sum((1, 2)) / (m.sum((1, 2)) + 100)
    # coarse term for wider basin
    s4 = Fn.avg_pool2d((soft * valid)[:, None], 8); m4 = Fn.avg_pool2d((m * valid)[:, None], 8)
    l = l + 4 * ((s4 - m4) ** 2).sum((1, 2, 3)) / (m.sum((1, 2)) / 64 + 2)
    return l


def smooth_loss(Rf, pf):
    d2R = Rf[2:] - 2 * Rf[1:-1] + Rf[:-2]
    d2p = pf[2:] - 2 * pf[1:-1] + pf[:-2]
    return (d2R ** 2).sum() * 1.0 + (d2p ** 2).sum() / (0.39 ** 2)  # p in units of table radius


t0 = time.time()
n = len(fr); bs = 48
for it in range(a.iters):
    perm = torch.randperm(n, device=dev)
    tot = 0
    for b in range(0, n, bs):
        idx = perm[b:b + bs]
        Rf, pf = poses()
        ls = sil_loss(idx, Rf, pf).sum()
        sm = smooth_loss(Rf, pf) * a.ws * 1e3 * len(idx) / n
        loss = ls + sm
        opt.zero_grad(); loss.backward(); opt.step()
        tot += ls.item()
    if it % 10 == 0 or it == a.iters - 1:
        print(it, 'sil', tot / n, 'smooth', smooth_loss(*poses()).item(), '%.0fs' % (time.time() - t0), flush=True)
with torch.no_grad():
    Rf, pf = poses()
Rf = Rf.cpu().numpy().astype(np.float64); pf = pf.cpu().numpy().astype(np.float64)
Tc = Tc0.copy()
Tc[fr, :3, :3] = Rf; Tc[fr, :3, 3] = pf - Rf @ c
np.savez(a.out, T_world_mesh=D['Twc'] @ Tc)
print('done', time.time() - t0)
