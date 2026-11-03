"""Validation measures that do not need ground truth."""
import numpy as np, cv2, torch, h5py
from hoi_render import Renderer


def read_gray_frames(video_path):
    cap = cv2.VideoCapture(video_path); out = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    return out


class FlowCheck:
    """Tracks texture points on the object (LK, forward-backward checked) between frames t and t+lag and
    measures how well a pose trajectory explains their motion: points at t are lifted to 3D with the rendered
    depth of pose T_t, moved rigidly with T_{t+lag} T_t^-1 and projected. Returns median reprojection error (px)."""
    def __init__(self, mesh, K, Twc, grays, masks_path, lag=4, step=2, dilate_person=15):
        self.R = Renderer(mesh, grays[0].shape[0], grays[0].shape[1])
        self.K = np.asarray(K); self.Tcw = np.linalg.inv(Twc); self.g = grays; self.lag = lag
        f = h5py.File(masks_path, "r"); O, P = f["object"], f["person"]
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_person, dilate_person))
        self.tracks = {}
        for t in range(0, len(grays) - lag, step):
            m = O[t] & ~cv2.dilate(P[t].astype(np.uint8), k).astype(bool) & ~cv2.dilate(P[t + lag].astype(np.uint8), k).astype(bool)
            m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8))
            pts = cv2.goodFeaturesToTrack(grays[t], 300, 0.01, 5, mask=m)
            if pts is None or len(pts) < 8: continue
            p1, st, _ = cv2.calcOpticalFlowPyrLK(grays[t], grays[t + lag], pts, None, winSize=(15, 15), maxLevel=3)
            p0b, st2, _ = cv2.calcOpticalFlowPyrLK(grays[t + lag], grays[t], p1, None, winSize=(15, 15), maxLevel=3)
            ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(pts - p0b, axis=2)[:, 0] < 0.5)
            if ok.sum() >= 8: self.tracks[t] = (pts[ok, 0], p1[ok, 0])

    def errors(self, Tw):
        """Tw (F,4,4) world poses. Returns dict t -> median error (px), and the image motion magnitude."""
        out = {}
        K = self.K
        for t, (p0, p1) in self.tracks.items():
            Tc0 = self.Tcw @ Tw[t]; Tc1 = self.Tcw @ Tw[t + self.lag]
            _, mk, dep = self.R.render(Tc0, K)
            dep = dep[0].cpu().numpy(); mk = mk[0].cpu().numpy()
            u, v = np.round(p0[:, 0]).astype(int), np.round(p0[:, 1]).astype(int)
            on = mk[v, u]
            if on.sum() < 5: continue
            z = dep[v[on], u[on]]
            X = np.stack([(p0[on, 0] - K[0, 2]) / K[0, 0] * z, (p0[on, 1] - K[1, 2]) / K[1, 1] * z, z], 1)
            M = Tc1 @ np.linalg.inv(Tc0)
            Y = X @ M[:3, :3].T + M[:3, 3]
            q = np.stack([K[0, 0] * Y[:, 0] / Y[:, 2] + K[0, 2], K[1, 1] * Y[:, 1] / Y[:, 2] + K[1, 2]], 1)
            e = np.linalg.norm(q - p1[on], axis=1)
            out[t] = (np.median(e), np.median(np.linalg.norm(p1[on] - p0[on], axis=1)), on.mean())
        return out


def mask_iou(sr, E, Tw, dev="cuda"):
    """Per-frame silhouette agreement with the SAM2 mask (un-occluded pixels), best of full / screen-only."""
    out = np.zeros(len(Tw))
    with torch.no_grad():
        for c0 in range(0, len(Tw), 100):
            T = torch.tensor(Tw[c0:c0 + 100], dtype=torch.float32, device=dev)
            o = sr.render(T)
            for j in range(len(T)):
                M, V, I = E.frame(c0 + j)
                Mb = (M > 0.5).float()
                best = 0
                for A in (o[j, ..., 0], o[j, ..., 1]):
                    Ab = (A > 0.5).float()
                    inter = (V * Mb * Ab).sum(); uni = (V * ((Mb + Ab) > 0).float()).sum()
                    best = max(best, (inter / uni.clamp(min=1)).item())
                out[c0 + j] = best
    return out
