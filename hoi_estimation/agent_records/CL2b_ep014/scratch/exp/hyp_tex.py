# for refined static hypotheses (hyp_static.npz rows 0..4 = spins 0,72,144,-144,-72), pooled top texture correlation
import sys; sys.path.insert(0, '/work/outputs')
from common import *
from rectify import *
import h5py
frames = eval(sys.argv[1]); which = sys.argv[2] if len(sys.argv) > 2 else 'top'
D = load_inputs('/work/inputs'); mesh = load_mesh('/work/inputs')
H = np.linalg.inv(D['Twc']) @ np.load('/work/scratch/hyp_static.npz')['T_world_mesh'][:5]
K = D['K']; R = Renderer(mesh, K, 1152, 1536)
h = h5py.File('/work/inputs/masks.h5'); Mo = h['object']; Mp = h['person']
exec(open('/work/scratch/exp/frame_tex.py').read().split("ref = np.load")[0].split("angs = np.arange")[1].join(["angs = np.arange", ""]) if False else "")
n = 400
yy, xx = np.mgrid[:n, :n]; rad = np.hypot(xx - n / 2 + 0.5, yy - n / 2 + 0.5) * 0.002
ann = (rad > 0.05) & (rad < 0.34)
angs = np.arange(-180, 180, 2)
def prep(img, w):
    g = img[..., :3].mean(-1); m = (w > 0) & ann
    g = np.where(m, g, 0)
    b = cv2.GaussianBlur(g, (0, 0), 8); bw = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 8)
    hp = np.where(m, g - b / np.maximum(bw, 1e-3), 0)
    hp = cv2.GaussianBlur(hp, (0, 0), 1.5) / np.maximum(cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5), 1e-3)
    m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return hp, m
def corr(a, ma, b, mb):
    m = ma & mb
    if m.sum() < 2000: return np.nan
    x = a[m] - a[m].mean(); y = b[m] - b[m].mean()
    return (x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum())
ref = np.load('/work/scratch/rect_ref_%s.npy' % which)
refs = [prep(rotate_grid(ref[..., :3], s), rotate_grid(ref[..., 3], s)) for s in angs]
z0, sign = (-0.115, -1) if which == 'top' else (-0.092, 1)
P = plane_grid(z0)
cap = cv2.VideoCapture('/work/inputs/video.mp4')
ims = []
for fr in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, im = cap.read(); ims.append(im[:, :, ::-1].astype(np.float32) / 255)
for gi, g in enumerate([0, 72, 144, -144, -72]):
    T = H[gi]; od = R.render(T[None], rgb=False, depth=True); dep = od['depth'][0].cpu().numpy()
    acc = 0; wacc = 0
    for k, fr in enumerate(frames):
        vm = cv2.erode((Mo[fr] & ~Mp[fr]).astype(np.float32), np.ones((5, 5), np.uint8))
        rr, w = rectify(ims[k], T, K, P, depth=dep, valid_mask=vm, normal_sign=sign, min_facing=0.08)
        if w.sum() < 2000: continue
        gg = (rr * w[..., None]).sum((0, 1)) / w.sum(); acc = acc + rr / gg * w[..., None]; wacc = wacc + w
    a_, ma = prep(acc / np.maximum(wacc, 1e-6)[..., None], wacc)
    cs = np.array([corr(a_, ma, *rf) for rf in refs]); i = np.nanargmax(cs)
    print('hyp %4d: corr at 0: %.3f   peak %4d %.3f' % (g, cs[list(angs).index(0)], angs[i], cs[i]))
