# sliding-window pooled texture spin evidence (top & underside) over the episode
import sys; sys.path.insert(0, '/work/outputs')
from common import *
from rectify import *
import h5py
poses = sys.argv[1]; W = int(sys.argv[2]); out = sys.argv[3]
D = load_inputs('/work/inputs'); mesh = load_mesh('/work/inputs')
Tw = D['rec']['T_world_mesh'].astype(np.float64) if poses == 'base' else np.load(poses)['T_world_mesh']
Tc = np.linalg.inv(D['Twc']) @ Tw
K = D['K']
R = Renderer(mesh, K, 1152, 1536)
V = np.load('/work/scratch/vid_half.npy', mmap_mode='r')
h = h5py.File('/work/inputs/masks.h5')
Mo = h['object']; Mp = h['person']
n = 400
yy, xx = np.mgrid[:n, :n]; rad = np.hypot(xx - n / 2 + 0.5, yy - n / 2 + 0.5) * 0.002
ann = (rad > 0.05) & (rad < 0.34)
angs = np.arange(-180, 180, 2)


def prep(img, w):
    g = img[..., :3].mean(-1)
    m = (w > 0) & ann
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


refs = {}
for which in ['top', 'under']:
    ref = np.load('/work/scratch/rect_ref_%s.npy' % which)
    refs[which] = [prep(rotate_grid(ref[..., :3], s), rotate_grid(ref[..., 3], s)) for s in angs]
cap = cv2.VideoCapture('/work/inputs/video.mp4')
frames = []
while True:
    ok, im = cap.read()
    if not ok: break
    frames.append(None)
F = len(Tc)
res = []
for which, z0, sign in [('top', -0.115, -1), ('under', -0.092, 1)]:
    P = plane_grid(z0)
    for a in range(0, F, W):
        acc = 0; wacc = 0
        for fr in range(a, min(a + W, F), 2):
            # full-res frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, im = cap.read(); im = im[:, :, ::-1].astype(np.float32) / 255
            od = R.render(Tc[fr][None], rgb=False, depth=True)
            vm = cv2.erode((Mo[fr] & ~Mp[fr]).astype(np.float32), np.ones((5, 5), np.uint8))
            rr, w = rectify(im, Tc[fr], K, P, depth=od['depth'][0].cpu().numpy(), valid_mask=vm, normal_sign=sign, min_facing=0.1)
            if w.sum() < 2000: continue
            g = (rr * w[..., None]).sum((0, 1)) / w.sum()
            acc = acc + rr / g * w[..., None]; wacc = wacc + w
        if np.isscalar(wacc):
            continue
        img = acc / np.maximum(wacc, 1e-6)[..., None]
        a_, ma = prep(img, wacc)
        cs = np.array([corr(a_, ma, *rf) for rf in refs[which]])
        if np.all(np.isnan(cs)): continue
        i = np.nanargmax(cs)
        # second peak at least 20 deg away
        d = np.abs(((angs - angs[i]) + 180) % 360 - 180)
        cs2 = np.where(d > 20, cs, -9); j = np.nanargmax(cs2)
        res.append((which, a, int(angs[i]), cs[i], int(angs[j]), cs[j]))
        print(which, a, a + W, 'peak %4d %.3f  second %4d %.3f' % (angs[i], cs[i], angs[j], cs[j]), flush=True)
        np.save('%s_%s_%d.npy' % (out, which, a), cs)
