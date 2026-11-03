# plate/dark-region agreement score per spin hypothesis on the underside board
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
r = np.hypot(*(fc[:, :2] - c).T)
board = (fc[:, 2] > -0.095) & (fc[:, 2] < -0.06) & (r < 0.36)
brd_t = torch.tensor(np.r_[False, board], device='cuda')
h = h5py.File('/work/inputs/masks.h5')
cap = cv2.VideoCapture('/work/inputs/video.mp4')
ax_p = np.array([0.006, 0.004, 0.0])


def spinT(T, deg):
    S = np.eye(4); S[:3, :3] = axis_angle([0, 0, 1], np.radians(deg)); S[:3, 3] = ax_p - S[:3, :3] @ ax_p
    return T @ S


res = []
for f in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
    g = torch.tensor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), device='cuda').float()
    pm = torch.tensor(h['person'][f], device='cuda')
    o = R.render(np.stack([spinT(Tc[f], s) for s in spins]))
    T = torch.tensor(np.stack([spinT(Tc[f], s) for s in spins]), dtype=torch.float32, device='cuda')
    vh = torch.cat([R.v, torch.ones_like(R.v[:, :1])], 1)
    clip = torch.einsum('ij,bnj->bni', R.proj(R.near, R.far), torch.einsum('bij,nj->bni', T, vh)).contiguous()
    rast, _ = dr.rasterize(R.glctx, clip, R.f, (R.H, R.W))
    tid = rast[..., 3].long()
    rg = o['rgb'].mean(-1) * 255
    sc = []
    for i in range(len(spins)):
        bm = brd_t[tid[i]] & ~pm
        if bm.sum() < 500:
            sc.append(np.nan); continue
        gr = g[bm]; gq = rg[i][bm]
        dr_ = gr < 0.45 * gr.median(); dq = gq < 0.45 * gq.median()
        sc.append(((dr_ & dq).sum() / ((dr_ | dq).sum() + 1)).item())
    sc = np.array(sc); res.append(sc)
    print(f, np.round(sc, 3), flush=True)
res = np.array(res)
print('mean', dict(zip(spins, np.round(np.nanmean(res, 0), 3))), 'wins', dict(zip(spins, np.bincount(np.nanargmax(np.nan_to_num(res, nan=-999), 1), minlength=len(spins)))))
