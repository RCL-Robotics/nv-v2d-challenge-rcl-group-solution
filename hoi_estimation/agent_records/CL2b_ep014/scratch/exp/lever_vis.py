import sys; sys.path.insert(0, '/work/outputs')
from common import *
import h5py
poses = sys.argv[1]; frames = eval(sys.argv[2])
D = load_inputs('/work/inputs'); mesh = load_mesh('/work/inputs')
Tw = D['rec']['T_world_mesh'].astype(np.float64) if poses == 'base' else np.load(poses)['T_world_mesh']
Tc = np.linalg.inv(D['Twc']) @ Tw
R = Renderer(mesh, D['K'], 1152, 1536)
v = mesh['v']; f = mesh['f']; fc = v[f].mean(1)
r = np.hypot(fc[:, 0] - 0.006, fc[:, 1] - 0.004); ang = np.degrees(np.arctan2(fc[:, 1] - 0.004, fc[:, 0] - 0.006))
lever = (r > 0.12) & (r < 0.37) & (fc[:, 2] > -0.06) & (fc[:, 2] < 0.12) & ((ang > 135) | (ang < -170))
lev_t = torch.tensor(np.r_[False, lever], device='cuda')
ax_p = np.array([0.0039, 0.0117, 0.0])
def spinT(T, deg):
    S = np.eye(4); S[:3, :3] = axis_angle([0, 0, 1], np.radians(deg)); S[:3, 3] = ax_p - S[:3, :3] @ ax_p
    return T @ S
h = h5py.File('/work/inputs/masks.h5')
for fr in frames:
    T = torch.tensor(np.stack([spinT(Tc[fr], s) for s in [0, 72, 144, -144, -72]]), dtype=torch.float32, device='cuda')
    vh = torch.cat([R.v, torch.ones_like(R.v[:, :1])], 1)
    clip = torch.einsum('ij,bnj->bni', R.proj(R.near, R.far), torch.einsum('bij,nj->bni', T, vh)).contiguous()
    rast, _ = dr.rasterize(R.glctx, clip, R.f, (R.H, R.W))
    lm = lev_t[rast[..., 3].long()].cpu().numpy()
    m = h['object'][fr]; p = h['person'][fr]
    out = []
    for i in range(5):
        l = lm[i] & ~p
        out.append('%5d:%4.2f' % (l.sum(), (l & m).sum() / max(l.sum(), 1)))
    print(fr, ' '.join(out))
