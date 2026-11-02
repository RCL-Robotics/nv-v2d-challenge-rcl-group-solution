import sys; sys.path.insert(0, '/work/outputs')
from common import *
import h5py
frames = eval(sys.argv[1])
D = load_inputs('/work/inputs'); mesh = load_mesh('/work/inputs')
H = np.linalg.inv(D['Twc']) @ np.load('/work/scratch/hyp_static.npz')['T_world_mesh'][:5]
R = Renderer(mesh, D['K'], 1152, 1536)
v = mesh['v']; f = mesh['f']; fc = v[f].mean(1)
r = np.hypot(fc[:, 0] - 0.006, fc[:, 1] - 0.004); ang = np.degrees(np.arctan2(fc[:, 1] - 0.004, fc[:, 0] - 0.006))
cast = (r > 0.24) & (fc[:, 2] > 0.588)
lever = (r > 0.05) & (r < 0.37) & (fc[:, 2] > -0.07) & (fc[:, 2] < 0.12) & ((ang > 135) | (ang < -170))
ign_t = torch.tensor(np.r_[False, cast | lever], device='cuda')
T = torch.tensor(H, dtype=torch.float32, device='cuda')
vh = torch.cat([R.v, torch.ones_like(R.v[:, :1])], 1)
clip = torch.einsum('ij,bnj->bni', R.proj(R.near, R.far), torch.einsum('bij,nj->bni', T, vh)).contiguous()
rast, _ = dr.rasterize(R.glctx, clip, R.f, (R.H, R.W))
full = rast[..., 3] > 0
ign = ign_t[rast[..., 3].long()].any(0).float()[None, None]
ign = torch.nn.functional.max_pool2d(ign, 15, 1, 7)[0, 0] > 0
h = h5py.File('/work/inputs/masks.h5')
acc = np.zeros(5)
for fr in frames:
    m = torch.tensor(h['object'][fr], device='cuda'); p = torch.tensor(h['person'][fr], device='cuda')
    valid = ~(p & ~m) & ~ign
    o = full & valid; mm = m & valid
    acc += ((o & mm).sum((1, 2)) / (o | mm).sum((1, 2))).cpu().numpy()
print('IoU w/o casters+lever:', dict(zip([0, 72, 144, -144, -72], np.round(acc / len(frames), 4))))
