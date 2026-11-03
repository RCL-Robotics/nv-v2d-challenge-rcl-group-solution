import sys; sys.path.insert(0, '/work/outputs')
from rectify import *
import numpy as np, cv2
which = sys.argv[1]; files = sys.argv[2].split(',')
ref = np.load('/work/scratch/rect_ref_%s.npy' % which)
n = ref.shape[0]
yy, xx = np.mgrid[:n, :n]; rr = np.hypot(xx - n / 2 + 0.5, yy - n / 2 + 0.5) * 0.002
ann = (rr > 0.05) & (rr < 0.34)


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
    if m.sum() < 500: return np.nan
    x = a[m] - a[m].mean(); y = b[m] - b[m].mean()
    return (x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum())


angs = np.arange(-180, 180, 2)
if files[0] == 'selftest':
    real = rotate_grid(ref, 30); files = files[1:]
    a, ma = prep(real[..., :3], real[..., 3])
    cs = [corr(a, ma, *prep(rotate_grid(ref[..., :3], s), rotate_grid(ref[..., 3], s))) for s in angs]
    print('selftest peak', angs[np.nanargmax(cs)])
for fn in files:
    real = np.load(fn)
    a, ma = prep(real[..., :3], real[..., 3])
    cs = np.array([corr(a, ma, *prep(rotate_grid(ref[..., :3], s), rotate_grid(ref[..., 3], s))) for s in angs])
    top = np.argsort(-cs)[:5]
    grid = {g: round(float(cs[angs == g][0]) if g in angs else np.nan, 3) for g in [0, 72, 144, -144, -72]}
    print(fn.split('/')[-1], 'peaks', [(int(angs[i]), round(float(cs[i]), 3)) for i in top], 'grid', grid)
