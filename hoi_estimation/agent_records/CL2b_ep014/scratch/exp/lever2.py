# lever polyline vs. real dark-thin-line map (distance transform) per spin hypothesis
import sys; sys.path.insert(0, '/work/outputs')
from common import *
import h5py
poses = sys.argv[1]; frames = eval(sys.argv[2]); dbg = len(sys.argv) > 3
spins = [0, 72, 144, -144, -72]
D = load_inputs('/work/inputs'); mesh = load_mesh('/work/inputs')
Tw = D['rec']['T_world_mesh'].astype(np.float64) if poses == 'base' else np.load(poses)['T_world_mesh']
Tc = np.linalg.inv(D['Twc']) @ Tw
R = Renderer(mesh, D['K'], 1152, 1536)
K = D['K']
c = np.array([0.006, 0.004])
rr = np.linspace(0.16, 0.31, 16); th = np.radians(161)
lever_pts = np.c_[c[0] + rr * np.cos(th), c[1] + rr * np.sin(th), np.interp(rr, [0.05, 0.31], [-0.06, 0.02])]
h = h5py.File('/work/inputs/masks.h5')
cap = cv2.VideoCapture('/work/inputs/video.mp4')
ax_p = np.array([0.006, 0.004, 0.0])


def spinT(T, deg):
    S = np.eye(4); S[:3, :3] = axis_angle([0, 0, 1], np.radians(deg)); S[:3, 3] = ax_p - S[:3, :3] @ ax_p
    return T @ S


res = []
for f in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    om = h['object'][f]; pm = h['person'][f]
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    med = np.median(g[om]) if om.sum() else 128
    line = (bh > 25) & (g < 0.6 * med) & om
    line = cv2.morphologyEx(line.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    dt = cv2.distanceTransform((1 - line).astype(np.uint8), cv2.DIST_L2, 5)
    Ts = [spinT(Tc[f], s) for s in spins]
    o = R.render(np.stack(Ts), rgb=False, depth=True)
    deps = o['depth'].cpu().numpy()
    sc = []
    for i, T in enumerate(Ts):
        P = lever_pts @ T[:3, :3].T + T[:3, 3]
        q = P @ K.T; u = q[:, 0] / q[:, 2]; v = q[:, 1] / q[:, 2]
        ui = np.clip(u.round().astype(int), 0, 1535); vi = np.clip(v.round().astype(int), 0, 1151)
        vis = (np.abs(deps[i][vi, ui] - P[:, 2]) < 0.03) & ~pm[vi, ui] & (u > 0) & (u < 1536) & (v > 0) & (v < 1152)
        if vis.sum() < 6:
            sc.append(np.nan); continue
        sc.append(np.mean(np.exp(-dt[vi[vis], ui[vis]] ** 2 / (2 * 12 ** 2))))
    sc = np.array(sc); res.append(sc)
    print(f, np.round(sc, 2), 'linepx', int(line.sum()), flush=True)
    if dbg:
        a = im.copy(); a[line > 0] = (0, 0, 255)
        for i, T in enumerate(Ts):
            P = lever_pts @ T[:3, :3].T + T[:3, 3]; q = P @ K.T
            for k in range(len(q)): cv2.circle(a, (int(q[k, 0] / q[k, 2]), int(q[k, 1] / q[k, 2])), 2, [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)][i], -1)
        ys, xs = np.nonzero(om); cv2.imwrite('/work/scratch/frames/l2_%d.png' % f, a[ys.min():ys.max(), xs.min():xs.max()])
res = np.array(res)
print('mean', dict(zip(spins, np.round(np.nanmean(res, 0), 3))), 'wins', dict(zip(spins, np.bincount(np.nanargmax(np.nan_to_num(res, nan=-999), 1), minlength=5))))
