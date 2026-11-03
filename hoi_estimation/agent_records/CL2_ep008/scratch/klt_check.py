"""Independent check of relative motion: KLT tracks on the object between frames t and t+D;
predicted displacement from a pose sequence (render depth at t, back-project, move rigidly, project)."""
import sys, numpy as np, torch, h5py, cv2
sys.path.insert(0, '/work/outputs')
from solve import load_inputs
sys.path.insert(0, '/work/scratch')
from render import Renderer

D = 3
def main(paths, s0, s1, step=2):
    K, Twc, Tw, rec, plane, mesh = load_inputs('/work/inputs')
    Tcw = np.linalg.inv(Twc)
    seqs = {'base': Tcw[None] @ Tw}
    for p in paths: seqs[p] = Tcw[None] @ np.load(p)['T_world_mesh']
    h5 = h5py.File('/work/inputs/masks.h5', 'r')
    cap = cv2.VideoCapture('/work/inputs/video.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES, s0)
    grays = {}
    for f in range(s0, s1 + D + 1):
        ok, im = cap.read(); grays[f] = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    R = Renderer(1536, 1152)
    errs = {k: [] for k in seqs}
    ker = np.ones((9, 9), np.uint8)
    for t in range(s0, s1, step):
        o = h5['object'][t]; p = h5['person'][t]; p2 = h5['person'][t + D]
        m = cv2.erode(o.astype(np.uint8), np.ones((5, 5), np.uint8)) & ~cv2.dilate(p.astype(np.uint8), ker).astype(bool) & ~cv2.dilate(p2.astype(np.uint8), ker).astype(bool)
        pts = cv2.goodFeaturesToTrack(grays[t], 300, 0.01, 5, mask=m.astype(np.uint8) * 255)
        if pts is None or len(pts) < 5: continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(grays[t], grays[t + D], pts, None, winSize=(21, 21), maxLevel=3)
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(grays[t + D], grays[t], p1, None, winSize=(21, 21), maxLevel=3)
        good = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(pb - pts, axis=2)[:, 0] < 0.5)
        x0 = pts[good, 0]; x1 = p1[good, 0]
        if len(x0) < 5: continue
        for k, T in seqs.items():
            dep = R.render(T[t:t + 1], K, rgb=False)['depth'][0].cpu().numpy()
            ui = np.clip(np.round(x0).astype(int), 0, [1535, 1151])
            z = dep[ui[:, 1], ui[:, 0]]
            ok = z > 0
            X = np.linalg.inv(K) @ np.vstack([x0[ok].T, np.ones(ok.sum())]) * z[ok]
            M = np.linalg.inv(T[t]) @ np.vstack([X, np.ones(ok.sum())])
            Y = (T[t + D] @ M)[:3]
            y = (K @ Y)[:2] / (K @ Y)[2]
            e = np.linalg.norm(y.T - x1[ok], axis=1)
            errs[k].append((t, np.median(e), ok.mean(), len(e)))
    for k, v in errs.items():
        v = np.array(v)
        print(f'{k:40s} frames {len(v)}  median-of-median px err {np.median(v[:, 1]):.3f}  mean {v[:, 1].mean():.3f}  on-object frac {v[:, 2].mean():.2f}')
    return errs

if __name__ == '__main__':
    main(sys.argv[3:], int(sys.argv[1]), int(sys.argv[2]))
