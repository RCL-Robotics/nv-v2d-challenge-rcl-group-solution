import sys; sys.path.insert(0, '/work/outputs')
from common import *
import h5py
poses = sys.argv[1]; f = int(sys.argv[2]); spins=[float(x) for x in sys.argv[3].split(',')]; out=sys.argv[4]
D = load_inputs('/work/inputs'); mesh = load_mesh('/work/inputs')
Tw = D['rec']['T_world_mesh'].astype(np.float64) if poses == 'base' else np.load(poses)['T_world_mesh']
Tc = np.linalg.inv(D['Twc']) @ Tw
R = Renderer(mesh, D['K'], 1152, 1536)
v = mesh['v']; fcs = mesh['f']
c = np.array([0.006, 0.004]); fc = v[fcs].mean(1)
r = np.hypot(*(fc[:, :2] - c).T); ang = np.degrees(np.arctan2(fc[:, 1] - c[1], fc[:, 0] - c[0]))
lever = (r > 0.07) & (r < 0.37) & (fc[:, 2] > -0.065) & (fc[:, 2] < 0.12) & ((ang > 135) | (ang < -170))
lev_t = torch.tensor(np.r_[False, lever], device='cuda')
cap = cv2.VideoCapture('/work/inputs/video.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
ax_p = np.array([0.006, 0.004, 0.0])
def spinT(T, deg):
    S = np.eye(4); S[:3, :3] = axis_angle([0, 0, 1], np.radians(deg)); S[:3, 3] = ax_p - S[:3, :3] @ ax_p
    return T @ S
m = h5py.File('/work/inputs/masks.h5')['object'][f]
ys, xs = np.nonzero(m); y0, y1, x0, x1 = ys.min()-30, ys.max()+30, xs.min()-30, xs.max()+30
tiles=[]
for s in spins:
    T = torch.tensor(spinT(Tc[f], s)[None], dtype=torch.float32, device='cuda')
    vh = torch.cat([R.v, torch.ones_like(R.v[:, :1])], 1)
    clip = torch.einsum('ij,bnj->bni', R.proj(R.near, R.far), torch.einsum('bij,nj->bni', T, vh)).contiguous()
    rast, _ = dr.rasterize(R.glctx, clip, R.f, (R.H, R.W))
    lm = lev_t[rast[0, ..., 3].long()].cpu().numpy()
    a = im.copy(); a[lm] = (0.5 * a[lm] + [0, 0, 127]).astype(np.uint8)
    a = a[max(y0,0):y1, max(x0,0):x1].copy(); cv2.putText(a, str(s), (5, 25), 0, 0.8, (0, 255, 255), 2)
    tiles.append(a)
cv2.imwrite(out, np.concatenate(tiles, 1))
