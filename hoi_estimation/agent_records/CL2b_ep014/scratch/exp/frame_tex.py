# per-frame texture spin evidence (single-frame rectification), top or under
import sys; sys.path.insert(0, '/work/outputs')
from common import *
from rectify import *
import h5py
poses = sys.argv[1]; which = sys.argv[2]; frames = eval(sys.argv[3]); pool = int(sys.argv[4]) if len(sys.argv) > 4 else 1
D = load_inputs('/work/inputs'); mesh = load_mesh('/work/inputs')
Tw = D['rec']['T_world_mesh'].astype(np.float64) if poses == 'base' else np.load(poses)['T_world_mesh']
Tc = np.linalg.inv(D['Twc']) @ Tw
K = D['K']; R = Renderer(mesh, K, 1152, 1536)
h = h5py.File('/work/inputs/masks.h5'); Mo = h['object']; Mp = h['person']
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
gi = [list(angs).index(g) for g in [0, 72, 144, -144, -72]]
for f in frames:
    acc = 0; wacc = 0
    for fr in range(f, f + pool):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, im = cap.read(); im = im[:, :, ::-1].astype(np.float32) / 255
        od = R.render(Tc[fr][None], rgb=False, depth=True)
        vm = cv2.erode((Mo[fr] & ~Mp[fr]).astype(np.float32), np.ones((5, 5), np.uint8))
        rr, w = rectify(im, Tc[fr], K, P, depth=od['depth'][0].cpu().numpy(), valid_mask=vm, normal_sign=sign, min_facing=float(__import__("os").environ.get("MINF","0.1")))
        if w.sum() < 2000: continue
        g = (rr * w[..., None]).sum((0, 1)) / w.sum()
        acc = acc + rr / g * w[..., None]; wacc = wacc + w
    if np.isscalar(wacc):
        print(f, 'no view'); continue
    a_, ma = prep(acc / np.maximum(wacc, 1e-6)[..., None], wacc)
    cs = np.array([corr(a_, ma, *rf) for rf in refs])
    if np.all(np.isnan(cs)):
        print(f, 'no overlap'); continue
    i = np.nanargmax(cs)
    print(f, 'peak %4d %.3f | grid 0:%.2f 72:%.2f 144:%.2f -144:%.2f -72:%.2f' % ((angs[i], cs[i]) + tuple(cs[gi])), flush=True)
