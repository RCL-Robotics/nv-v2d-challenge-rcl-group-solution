# per-frame lever angle on the board plane (mesh frame of the given poses)
import sys; sys.path.insert(0, '/work/outputs')
from common import *
import h5py
poses = sys.argv[1]; frames = eval(sys.argv[2])
D = load_inputs('/work/inputs'); mesh = load_mesh('/work/inputs')
Tw = D['rec']['T_world_mesh'].astype(np.float64) if poses == 'base' else np.load(poses)['T_world_mesh']
Tc = np.linalg.inv(D['Twc']) @ Tw
R = Renderer(mesh, D['K'], 1152, 1536)
K = D['K']; Ki = np.linalg.inv(K)
v = mesh['v']; fcs = mesh['f']
c = np.array([0.006, 0.004]); fc = v[fcs].mean(1)
r = np.hypot(*(fc[:, :2] - c).T)
board = (fc[:, 2] > -0.095) & (fc[:, 2] < -0.06) & (r < 0.37)
brd_t = torch.tensor(np.r_[False, board], device='cuda')
h = h5py.File('/work/inputs/masks.h5')
cap = cv2.VideoCapture('/work/inputs/video.mp4')
ZB = -0.08
out = []
CH = []
for f in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, im = cap.read()
    T = Tc[f]
    # underside normal (+z mesh) must face camera
    n = T[:3, 2]; p0 = T[:3, :3] @ np.array([c[0], c[1], ZB]) + T[:3, 3]
    facing = -(n @ p0) / np.linalg.norm(p0)
    if facing < 0.25:
        out.append((f, np.nan, 0, facing)); continue
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    om = h['object'][f]; pm = h['person'][f]
    Tt = torch.tensor(T[None], dtype=torch.float32, device='cuda')
    vh = torch.cat([R.v, torch.ones_like(R.v[:, :1])], 1)
    clip = torch.einsum('ij,bnj->bni', R.proj(R.near, R.far), torch.einsum('bij,nj->bni', Tt, vh)).contiguous()
    rast, _ = dr.rasterize(R.glctx, clip, R.f, (R.H, R.W))
    bm = brd_t[rast[0, ..., 3].long()].cpu().numpy()
    bm = cv2.dilate(bm.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & om & ~pm
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    med = np.median(g[bm]) if bm.sum() > 100 else 128
    line = (bh > 18) & (g < 0.7 * med) & bm
    ys, xs = np.nonzero(line)
    if len(ys) < 20:
        out.append((f, np.nan, 0, facing)); continue
    rays = np.c_[xs, ys, np.ones(len(xs))] @ Ki.T
    # intersect with board plane: n.(X - p0) = 0
    tt = (n @ p0) / (rays @ n)
    X = rays * tt[:, None]
    Xm = (X - T[:3, 3]) @ T[:3, :3]
    rad = np.hypot(Xm[:, 0] - c[0], Xm[:, 1] - c[1]); ang = np.degrees(np.arctan2(Xm[:, 1] - c[1], Xm[:, 0] - c[0]))
    sel = (rad > 0.13) & (rad < 0.33)
    if sel.sum() < 15:
        out.append((f, np.nan, int(sel.sum()), facing)); continue
    hst, e = np.histogram(ang[sel], bins=72, range=(-180, 180))
    hs = hst + np.roll(hst, 1) + np.roll(hst, -1)
    k = np.argmax(hs)
    # refine: circular mean around peak
    a0 = e[k] + 2.5
    d = (ang[sel] - a0 + 180) % 360 - 180
    m = np.abs(d) < 12
    peak = a0 + d[m].mean()
    out.append((f, (peak + 180) % 360 - 180, int(m.sum()), facing))
    # chamfer score of 3D mesh lever polyline under the 5 spin hypotheses
    det = np.zeros(g.shape, np.uint8); det[ys[sel], xs[sel]] = 1
    dt = cv2.distanceTransform(1 - det, cv2.DIST_L2, 5)
    rr = np.linspace(0.16, 0.31, 16); th = np.radians(161)
    LP = np.c_[c[0] + rr * np.cos(th), c[1] + rr * np.sin(th), np.interp(rr, [0.05, 0.31], [-0.06, 0.02])]
    scs = []
    for s_ in [0, 72, 144, -144, -72]:
        Rz = axis_angle([0, 0, 1], np.radians(s_)); ap = np.array([0.006, 0.004, 0])
        Pm = (LP - ap) @ Rz.T + ap
        P = Pm @ T[:3, :3].T + T[:3, 3]; q = P @ K.T
        ui = np.clip((q[:, 0] / q[:, 2]).round().astype(int), 0, 1535); vi = np.clip((q[:, 1] / q[:, 2]).round().astype(int), 0, 1151)
        scs.append(np.mean(np.exp(-dt[vi, ui] ** 2 / (2 * 15 ** 2))))
    CH.append([f] + scs)
    if len(sys.argv) > 3:
        a = im.copy(); a[line] = (0, 0, 255)
        yy, xx = ys[sel][np.abs(((ang[sel] - peak + 180) % 360) - 180) < 12], xs[sel][np.abs(((ang[sel] - peak + 180) % 360) - 180) < 12]
        a[yy, xx] = (0, 255, 0)
        # mesh lever at 161 deg for reference, projected with this pose
        for rr_ in np.linspace(0.13, 0.33, 12):
            P = T[:3, :3] @ np.array([c[0] + rr_ * np.cos(np.radians(161)), c[1] + rr_ * np.sin(np.radians(161)), ZB]) + T[:3, 3]
            q = K @ P; cv2.circle(a, (int(q[0] / q[2]), int(q[1] / q[2])), 2, (255, 0, 0), -1)
        y_, x_ = np.nonzero(om); cv2.imwrite('/work/scratch/frames/la_%d.png' % f, a[y_.min():y_.max(), x_.min():x_.max()])
CH = np.array(CH)
for r_ in CH: print('chamfer', int(r_[0]), np.round(r_[1:], 2), [0, 72, 144, -144, -72][int(np.argmax(r_[1:]))])
for o in out:
    print(o[0], 'lever_ang %.1f' % o[1], 'n', o[2], 'facing %.2f' % o[3], ' spin_corr %.1f' % (((o[1] - 161) + 180) % 360 - 180 if o[1] == o[1] else np.nan))
np.save('/work/scratch/lever_angle_%s.npy' % ('base' if poses == 'base' else 'x'), np.array([(o[0], o[1], o[2], o[3]) for o in out]))
