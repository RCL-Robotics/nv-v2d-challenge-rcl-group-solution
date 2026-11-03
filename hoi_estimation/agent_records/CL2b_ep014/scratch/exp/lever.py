# lever-darkness score per spin hypothesis
import sys; sys.path.insert(0, '/work/outputs')
from common import *
import h5py
poses = sys.argv[1]; frames = eval(sys.argv[2])
spins = [float(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [0, 72, 144, -144, -72]
D = load_inputs('/work/inputs'); mesh = load_mesh('/work/inputs')
Tw = D['rec']['T_world_mesh'].astype(np.float64) if poses == 'base' else np.load(poses)['T_world_mesh']
Tc = np.linalg.inv(D['Twc']) @ Tw
R = Renderer(mesh, D['K'], 1152, 1536)
v = mesh['v']; fcs = mesh['f']
c = np.array([0.006, 0.004]); fc = v[fcs].mean(1)
r = np.hypot(*(fc[:, :2] - c).T); ang = np.degrees(np.arctan2(fc[:, 1] - c[1], fc[:, 0] - c[0]))
lever = (r > 0.07) & (r < 0.37) & (fc[:, 2] > -0.065) & (fc[:, 2] < 0.12) & ((ang > 135) | (ang < -170))
board = (fc[:, 2] < -0.075) & (fc[:, 2] > -0.09) & (r > 0.12) & (r < 0.33)
print('lever faces', lever.sum(), 'board faces', board.sum())
lev_t = torch.tensor(np.r_[False, lever], device='cuda'); brd_t = torch.tensor(np.r_[False, board], device='cuda')
h = h5py.File('/work/inputs/masks.h5')
cap = cv2.VideoCapture('/work/inputs/video.mp4')
ax_p = np.array([0.006, 0.004, 0.0])


def spinT(T, deg):
    S = np.eye(4); S[:3, :3] = axis_angle([0, 0, 1], np.radians(deg)); S[:3, 3] = ax_p - S[:3, :3] @ ax_p
    return T @ S


tot = np.zeros(len(spins)); res = []
for f in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
    g = torch.tensor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), device='cuda').float()
    pm = torch.tensor(h['person'][f], device='cuda')
    Ts = np.stack([spinT(Tc[f], s) for s in spins])
    T = torch.tensor(Ts, dtype=torch.float32, device='cuda')
    vh = torch.cat([R.v, torch.ones_like(R.v[:, :1])], 1)
    clip = torch.einsum('ij,bnj->bni', R.proj(R.near, R.far), torch.einsum('bij,nj->bni', T, vh)).contiguous()
    rast, _ = dr.rasterize(R.glctx, clip, R.f, (R.H, R.W))
    tid = rast[..., 3].long()
    sc = []
    for i in range(len(spins)):
        lm = lev_t[tid[i]] & ~pm; bm = brd_t[tid[i]] & ~pm
        if lm.sum() < 30 or bm.sum() < 100:
            sc.append(np.nan); continue
        sc.append((g[bm].median() - g[lm].mean()).item())
    sc = np.array(sc); res.append(sc)
    print(f, np.round(sc, 1), flush=True)
res = np.array(res)
print('mean', dict(zip(spins, np.round(np.nanmean(res, 0), 1))), 'wins', dict(zip(spins, np.bincount(np.nanargmax(np.nan_to_num(res, nan=-999), 1), minlength=len(spins)))))
