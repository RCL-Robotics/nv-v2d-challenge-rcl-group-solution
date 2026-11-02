# refine one static pose per spin hypothesis over a set of frames; compare final silhouette fit
import sys; sys.path.insert(0, '/work/outputs')
from common import *
import torch.nn.functional as Fn
poses = sys.argv[1]; frames = eval(sys.argv[2])
D = load_inputs('/work/inputs'); mesh = load_mesh('/work/inputs')
Tw = D['rec']['T_world_mesh'].astype(np.float64) if poses == 'base' else np.load(poses)['T_world_mesh']
Tc = np.linalg.inv(D['Twc']) @ Tw
import h5py
h = h5py.File('/work/inputs/masks.h5')
M = torch.tensor(np.stack([h['object'][f] for f in frames]), device='cuda')
P = torch.tensor(np.stack([h['person'][f] for f in frames]), device='cuda')
R = Renderer(mesh, D['K'], 1152, 1536)
c = 0.5 * (mesh['v'].min(0) + mesh['v'].max(0))
R.v = R.v - torch.tensor(c, dtype=torch.float32, device='cuda')
ax_p = np.array([0.0007, 0.0137, 0.0])
valid = (~(P & ~M)).float(); m = M.float()


def spinT(T, deg):
    S = np.eye(4); S[:3, :3] = axis_angle([0, 0, 1], np.radians(deg)); S[:3, 3] = ax_p - S[:3, :3] @ ax_p
    return T @ S


def rodrigues(w):
    th = w.norm().clamp_min(1e-8); k = w / th
    Kx = torch.zeros(3, 3, device=w.device)
    Kx[0, 1], Kx[0, 2], Kx[1, 0], Kx[1, 2], Kx[2, 0], Kx[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    return torch.eye(3, device=w.device) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


# median pose of the frames as start
U, _, Vt = np.linalg.svd(Tc[frames, :3, :3].sum(0)); Rm = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
T0 = np.eye(4); T0[:3, :3] = Rm; T0[:3, 3] = np.median(Tc[frames, :3, 3], 0)
RES = {}
for g in [0, 72, 144, -144, -72]:
    Ti = spinT(T0, g)
    R0 = torch.tensor(Ti[:3, :3], dtype=torch.float32, device='cuda')
    p0 = torch.tensor(Ti[:3, :3] @ c + Ti[:3, 3], dtype=torch.float32, device='cuda')
    om = torch.zeros(3, device='cuda', requires_grad=True); dp = torch.zeros(3, device='cuda', requires_grad=True)
    opt = torch.optim.Adam([om, dp], lr=2e-3)
    for it in range(int(__import__("os").environ.get("IT","150"))):
        Rr = rodrigues(om) @ R0; pp = p0 + dp
        T = torch.eye(4, device='cuda'); T = T.clone(); T[:3, :3] = Rr; T[:3, 3] = pp
        vh = torch.cat([R.v, torch.ones_like(R.v[:, :1])], 1)
        vc = (T @ vh.T).T[None]
        clip = torch.einsum('ij,bnj->bni', R.proj(R.near, R.far), vc).contiguous()
        rast, _ = dr.rasterize(R.glctx, clip, R.f, (R.H, R.W))
        soft = dr.antialias((rast[..., 3:] > 0).float(), rast, clip, R.f)[..., 0]
        s = soft.expand_as(m)
        loss = (((s - m) ** 2) * valid).sum() / m.sum()
        s8 = Fn.avg_pool2d((s * valid)[:, None], 8); m8 = Fn.avg_pool2d((m * valid)[:, None], 8)
        loss = loss + 4 * ((s8 - m8) ** 2).sum() / (m.sum() / 64)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        r = (soft[0] > 0.5)[None] & (valid > 0)
        iou = ((r & M).sum((1, 2)).float() / (r | M).sum((1, 2)).float()).mean().item()
    Tf = np.eye(4); Tf[:3, :3] = Rr.detach().cpu().numpy(); Tf[:3, 3] = pp.detach().cpu().numpy() - Tf[:3, :3] @ c
    RES[g] = Tf
    print('hyp %4d  loss %.4f  iou %.4f  drot %.2f deg  dp %s cm' % (g, loss.item(), iou, np.degrees(om.norm().item()), np.round(dp.detach().cpu().numpy() * 100, 1)), flush=True)

f0 = frames[len(frames) // 2]
Tall = np.repeat((D['Twc'] @ RES[0])[None], 993, 0)
for i, g in enumerate([0, 72, 144, -144, -72]):
    Tall[i] = D['Twc'] @ RES[g]
np.savez('/work/scratch/hyp_static.npz', T_world_mesh=Tall, frame=f0)
