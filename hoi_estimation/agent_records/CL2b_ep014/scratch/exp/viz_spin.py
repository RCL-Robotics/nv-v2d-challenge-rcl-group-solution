import sys; sys.path.insert(0, '/work/outputs')
from common import *
import h5py
poses = sys.argv[1]; frames = [int(x) for x in sys.argv[2].split(',')]; out = sys.argv[3]
spins = [float(x) for x in sys.argv[4].split(',')] if len(sys.argv) > 4 else [0, 72, 144, -144, -72]
D = load_inputs('/work/inputs'); mesh = load_mesh('/work/inputs')
Tw = D['rec']['T_world_mesh'].astype(np.float64) if poses == 'base' else np.load(poses)['T_world_mesh']
Tc = np.linalg.inv(D['Twc']) @ Tw
R = Renderer(mesh, D['K'], 1152, 1536)
h = h5py.File('/work/inputs/masks.h5')
cap = cv2.VideoCapture('/work/inputs/video.mp4')
ax_p = np.array([0.006, 0.004, 0.0])


def spinT(T, deg):
    S = np.eye(4); S[:3, :3] = axis_angle([0, 0, 1], np.radians(deg)); S[:3, 3] = ax_p - S[:3, :3] @ ax_p
    return T @ S


rows = []
for f in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read(); im = im[:, :, ::-1].astype(np.float32) / 255
    m = h['object'][f]
    ys, xs = np.nonzero(m); cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    s = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 0.6) + 10
    y0, x0 = max(cy - s, 0), max(cx - s, 0)
    tl = [cv2.resize(im[y0:cy + s, x0:cx + s], (300, 300))]
    o = R.render(np.stack([spinT(Tc[f], d) for d in spins]))
    for i in range(len(spins)):
        rgb = o['rgb'][i].cpu().numpy(); mm = o['mask'][i].cpu().numpy()
        c = im * 0.25; c[mm] = rgb[mm]
        c = cv2.resize(c[y0:cy + s, x0:cx + s], (300, 300))
        cv2.putText(c, '%d' % spins[i], (5, 20), 0, 0.6, (1, 0, 0), 2)
        tl.append(c)
    t = np.concatenate(tl, 1)
    cv2.putText(t, str(f), (5, 290), 0, 0.6, (1, 0, 0), 2)
    rows.append(t)
cv2.imwrite(out, (np.concatenate(rows, 0)[:, :, ::-1] * 255).astype(np.uint8))
