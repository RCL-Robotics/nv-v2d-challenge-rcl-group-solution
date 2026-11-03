# board-underside restricted appearance scores per spin hypothesis
import sys; sys.path.insert(0, '/work/outputs')
from common import *
from match import Matcher, crop_box, to_tensor_crop
import h5py
poses = sys.argv[1]; frames = eval(sys.argv[2])
spins = [0, 72, 144, -144, -72]
D = load_inputs('/work/inputs'); mesh = load_mesh('/work/inputs')
Tw = D['rec']['T_world_mesh'].astype(np.float64) if poses == 'base' else np.load(poses)['T_world_mesh']
Tc = np.linalg.inv(D['Twc']) @ Tw
R = Renderer(mesh, D['K'], 1152, 1536)
v = mesh['v']; fcs = mesh['f']
c = np.array([0.006, 0.004]); fc = v[fcs].mean(1)
r = np.hypot(*(fc[:, :2] - c).T)
board = (fc[:, 2] > -0.095) & (fc[:, 2] < 0.12) & (r > 0.03) & (r < 0.37)  # underside incl. plate+lever
brd_t = torch.tensor(np.r_[False, board], device='cuda')
mt = Matcher('superpoint')
h = h5py.File('/work/inputs/masks.h5')
cap = cv2.VideoCapture('/work/inputs/video.mp4')
ax_p = np.array([0.006, 0.004, 0.0])


def spinT(T, deg):
    S = np.eye(4); S[:3, :3] = axis_angle([0, 0, 1], np.radians(deg)); S[:3, 3] = ax_p - S[:3, :3] @ ax_p
    return T @ S


A1, A2 = [], []
for f in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read(); im = im[:, :, ::-1].copy()
    gblur = cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 3)
    pm = h['person'][f]; om = h['object'][f]
    Ts = np.stack([spinT(Tc[f], s) for s in spins])
    o = R.render(Ts)
    T = torch.tensor(Ts, dtype=torch.float32, device='cuda')
    vh = torch.cat([R.v, torch.ones_like(R.v[:, :1])], 1)
    clip = torch.einsum('ij,bnj->bni', R.proj(R.near, R.far), torch.einsum('bij,nj->bni', T, vh)).contiguous()
    rast, _ = dr.rasterize(R.glctx, clip, R.f, (R.H, R.W))
    bms = brd_t[rast[..., 3].long()].cpu().numpy()
    rgbs = o['rgb'].cpu().numpy()
    s1, s2 = [], []
    for i in range(len(spins)):
        bm = bms[i] & ~pm & om
        if bm.sum() < 2000:
            s1.append(np.nan); s2.append(np.nan); continue
        gq = cv2.GaussianBlur((rgbs[i].mean(-1) * 255).astype(np.float32), (0, 0), 3)
        a = gblur[bm]; b = gq[bm]
        s1.append(np.corrcoef(a, b)[0, 1])
        # matches restricted to board region
        keep = cv2.dilate(bm.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        box = crop_box(bm)
        tr, A = to_tensor_crop((im.astype(np.float32) / 255) * keep[..., None], box, 512)
        tq, _ = to_tensor_crop(rgbs[i].astype(np.float32) * keep[..., None], box, 512)
        k0, k1, _ = mt.match(tq, tr)
        d = np.linalg.norm(k0 - k1, axis=1) / A[0, 0]
        s2.append((d < 8).sum())
    A1.append(s1); A2.append(s2)
    print(f, 'ncc', np.round(s1, 2), 'match', s2, flush=True)
A1 = np.array(A1); A2 = np.array(A2, float)
for nm, A in [('ncc', A1), ('match', A2)]:
    print(nm, 'mean', dict(zip(spins, np.round(np.nanmean(A, 0), 3))), 'wins', dict(zip(spins, np.bincount(np.nanargmax(np.nan_to_num(A, nan=-999), 1), minlength=5))))
