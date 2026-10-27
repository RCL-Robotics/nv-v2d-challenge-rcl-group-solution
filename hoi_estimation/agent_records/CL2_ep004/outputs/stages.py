"""Pipeline stages used by solve.py (kept importable so experiments and the final run share code)."""
import json
import numpy as np
import cv2
import h5py
import torch
from bat_lib import load_inputs, load_mesh, simplified_mesh, FastRenderer
from fitter import (RES, crop_window, prep_frame, convert_K, variants, fit_batch, score_batch, crop_K, extend_mask)

BAT_LEN = 0.543
MOGE_RAW = "/pipeline/depth/moge2/raw.npy"


class Episode:
    def __init__(self, inp):
        self.inp = inp
        self.K, self.Twc, rec, self.plane = load_inputs(inp)
        self.Kp = np.array(json.load(open(f"{inp}/camera.json"))["K_pipeline"], dtype=np.float64)
        self.Tw_base = rec["T_world_mesh"].astype(np.float64)
        self.F = len(self.Tw_base)
        self.Tcw = np.linalg.inv(self.Twc)
        self.Tc_base = self.Tcw[None] @ self.Tw_base
        self.fp = rec["T_cam_mesh_foundationpose"].astype(np.float64)
        self.cc = rec["T_cam_mesh_coconet"].astype(np.float64)
        self.kp_world = rec["human_keypoints_world"].astype(np.float64)
        self.hv_world = rec["human_vertices_world"]
        self.contact = rec["contact_prob_right_left"].astype(np.float64)
        # depth scale of the pipeline's camera-frame estimates relative to the (calibrated) baseline
        self.zs_fp = float(np.median(self.Tc_base[:, 2, 3] / self.fp[:, 2, 3]))
        self.zs_cc = float(np.median(self.Tc_base[:, 2, 3] / self.cc[:, 2, 3]))
        self.mesh = load_mesh(inp)

    def fast_renderer(self):
        vs, fs, cs = simplified_mesh(self.mesh)
        return FastRenderer(vs, fs, cs)

    def load_frames(self):
        """Decode the video once; build the per-frame crop windows and crop data."""
        f = h5py.File(f"{self.inp}/masks.h5", "r")
        cap = cv2.VideoCapture(f"{self.inp}/video.mp4")
        raw = np.load(MOGE_RAW, mmap_mode="r")  # MoGe-2 metric depth (uint16 mm), only used for relative depth
        self.wins, self.datas = [], []
        self.mask_area = np.zeros(self.F); self.mask_cent = np.zeros((self.F, 2)); self.ext_area = np.zeros(self.F)
        self.mask_small = []  # quarter-res masks for phase analysis
        for i in range(self.F):
            ok, img = cap.read()
            assert ok, i
            om = f["object"][i]; pm = f["person"][i]
            T = self.Tc_base[i]
            hint = self.K @ T[:3, 3]; hint = hint[:2] / hint[2]
            win = crop_window(om, hint, BAT_LEN * self.K[0, 0] / T[2, 3])
            x0, y0, side = win  # extend the mask only inside a margin around the crop window (speed)
            xa, ya = int(max(0, x0 - 60)), int(max(0, y0 - 60))
            xb, yb = int(min(om.shape[1], x0 + side + 60)), int(min(om.shape[0], y0 + side + 60))
            om_ext = om.copy()
            if om.sum() > 0:
                om_ext[ya:yb, xa:xb] = extend_mask(img[ya:yb, xa:xb], om[ya:yb, xa:xb], pm[ya:yb, xa:xb])
            self.ext_area[i] = om_ext.sum()
            d = prep_frame(img, om_ext, pm, win)
            x0, y0, side = win; sc = RES / side
            Aff = np.array([[sc, 0, -x0 * sc], [0, sc, -y0 * sc]])
            d["Zm"] = cv2.warpAffine(raw[i].astype(np.float32) / 1000.0, Aff, (RES, RES), flags=cv2.INTER_NEAREST)
            self.wins.append(win); self.datas.append(d)
            self.mask_area[i] = om.sum()
            ys, xs = np.nonzero(om)
            self.mask_cent[i] = (xs.mean(), ys.mean()) if len(xs) else (np.nan, np.nan)
            self.mask_small.append((cv2.resize(om.astype(np.uint8), (384, 288), interpolation=cv2.INTER_AREA) > 0,
                                    cv2.resize(pm.astype(np.uint8), (384, 288), interpolation=cv2.INTER_AREA) > 0))
        self.mask_small_o = np.stack([m[0] for m in self.mask_small]); self.mask_small_p = np.stack([m[1] for m in self.mask_small])
        del self.mask_small

    def init_hypotheses(self, i):
        out = []
        for T in [self.Tc_base[i], convert_K(self.fp[i], self.Kp, self.K, self.zs_fp),
                  convert_K(self.cc[i], self.Kp, self.K, self.zs_cc)]:
            out += variants(T)
        return np.stack(out)


def fit_frames(ep, FR, frame_ids, inits_per_frame, iters=80, chunk=96):
    """Refine a list of inits for each frame. inits_per_frame: list of (n_i,4,4). Returns list of dicts."""
    jobs = [(i, k) for i, ins in zip(frame_ids, inits_per_frame) for k in range(len(ins))]
    allT = np.concatenate(inits_per_frame, 0)
    resT = np.zeros_like(allT); sc = np.zeros((len(allT), 4))
    for s in range(0, len(jobs), chunk):
        js = jobs[s:s + chunk]
        T0 = allT[s:s + chunk]
        wins = [ep.wins[i] for i, _ in js]; datas = [ep.datas[i] for i, _ in js]
        T, tot, li, lc, ld = fit_batch(FR, ep.K, T0, wins, datas, iters=iters)
        resT[s:s + chunk] = T
        sc[s:s + chunk] = np.stack([tot, li, lc, ld], 1)
    out = []; p = 0
    for i, ins in zip(frame_ids, inits_per_frame):
        n = len(ins)
        out.append(dict(T=resT[p:p + n], sc=sc[p:p + n])); p += n
    return out


def score_frames(ep, FR, frame_ids, T_per_frame, chunk=192):
    jobs = [(i, k) for i, Ts in zip(frame_ids, T_per_frame) for k in range(len(Ts))]
    allT = np.concatenate(T_per_frame, 0)
    sc = np.zeros((len(allT), 4))
    for s in range(0, len(jobs), chunk):
        js = jobs[s:s + chunk]
        out = score_batch(FR, ep.K, allT[s:s + chunk], [ep.wins[i] for i, _ in js], [ep.datas[i] for i, _ in js])
        sc[s:s + chunk] = np.stack(out, 1)
    res = []; p = 0
    for Ts in T_per_frame:
        res.append(sc[p:p + len(Ts)]); p += len(Ts)
    return res


def detect_static_phases(mask_o, mask_p, thr=0.9):
    """Static phases at the start/end: the object mask (person-occluded pixels ignored) stays the same as in the
    first / last frame. Returns (end of first static phase, start of last static phase), inclusive indices."""
    F = len(mask_o)

    def iou_ref(ref):
        r = np.zeros(F)
        for i in range(F):
            v = ~mask_p[i]
            a = mask_o[i] & v; b = ref & v
            r[i] = (a & b).sum() / max((a | b).sum(), 1)
        return r
    i0 = iou_ref(mask_o[0]); i1 = iou_ref(mask_o[-1])
    s1 = 0
    while s1 + 1 < F and i0[s1 + 1] >= thr: s1 += 1
    s2 = F - 1
    while s2 - 1 >= 0 and i1[s2 - 1] >= thr: s2 -= 1
    return s1, s2, i0, i1


def unary_cost(sc, w_col=1.0):
    return sc[:, 1] + 2.0 * sc[:, 3] + w_col * sc[:, 2]


def huber(x, d):
    a = np.abs(x)
    return np.where(a < d, 0.5 * a ** 2 / d, a - 0.5 * d)


def pair_cost(Ta, Tb, ang_scale=10.0, t_scale=0.05, z_w=0.3):
    """(na,nb) transition costs between candidate sets of consecutive frames (camera frame)."""
    aa = Ta[:, :3, 2]; ab = Tb[:, :3, 2]
    ang = np.degrees(np.arccos(np.clip(aa @ ab.T, -1, 1)))
    xa = Ta[:, :3, 0]; xb = Tb[:, :3, 0]
    roll = np.degrees(np.arccos(np.clip(xa @ xb.T, -1, 1)))
    d = Ta[:, None, :3, 3] - Tb[None, :, :3, 3]
    d[..., 2] *= z_w
    dt = np.linalg.norm(d, axis=-1)
    return 2.0 * huber(ang / ang_scale, 1.5) + 0.1 * huber(roll / 30.0, 1.0) + 2.0 * huber(dt / t_scale, 1.5)


def viterbi(cands, unary, lam_u=10.0, **kw):
    """cands: list of (n_i,4,4); unary: list of (n_i,). Returns chosen index per frame."""
    F = len(cands)
    acc = lam_u * unary[0]; back = []
    for i in range(1, F):
        P = pair_cost(cands[i - 1], cands[i], **kw)
        tot = acc[:, None] + P
        b = tot.argmin(0); back.append(b)
        acc = tot[b, np.arange(len(b))] + lam_u * unary[i]
    k = [int(acc.argmin())]
    for b in back[::-1]:
        k.append(int(b[k[-1]]))
    return np.array(k[::-1])


def depth_scores(ep, FR, frame_ids, T_per_frame, chunk=192, dev="cuda"):
    """Shift-invariant depth-profile residual (m): median |(Z_moge - Z_render) - median(.)| over eroded object
    pixels covered by the rendered bat. Discriminates the out-of-plane tilt sign (mirror) hypotheses."""
    jobs = [(i, k) for i, Ts in zip(frame_ids, T_per_frame) for k in range(len(Ts))]
    allT = np.concatenate(T_per_frame, 0)
    out = np.full(len(allT), np.nan)
    ero = {}
    for s in range(0, len(jobs), chunk):
        js = jobs[s:s + chunk]; T = allT[s:s + chunk]
        Kc = np.stack([crop_K(ep.K, ep.wins[i]) for i, _ in js])
        with torch.no_grad():
            S, _, Z = FR.render(torch.tensor(T[:, :3, :3], dtype=torch.float32, device=dev),
                                torch.tensor(T[:, :3, 3], dtype=torch.float32, device=dev), Kc, RES)
        S = S.cpu().numpy(); Z = Z.cpu().numpy()
        for b, (i, _) in enumerate(js):
            if i not in ero:
                ero[i] = cv2.erode((ep.datas[i]["M"] > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            Zm = ep.datas[i]["Zm"]
            sel = (S[b] > 0.99) & ero[i] & (Zm > 0)
            if sel.sum() >= 30:
                r = Zm[sel] - Z[b][sel]
                out[s + b] = np.median(np.abs(r - np.median(r)))
    res = []; p = 0
    for Ts in T_per_frame:
        res.append(out[p:p + len(Ts)]); p += len(Ts)
    return res


def floor_heights(ep, Tc, verts, z_target):
    """Min height above the released floor plane of mesh points, after moving the pose along its viewing ray to
    depth z_target (baseline depth), in metres. Tc (n,4,4)."""
    T = Tc.copy()
    T[:, :3, 3] *= (z_target / T[:, 2, 3])[:, None]
    Tw = ep.Twc[None] @ T
    P = np.einsum("nij,vj->nvi", Tw[:, :3, :3], verts) + Tw[:, None, :3, 3]
    h = P @ ep.plane[:3] + ep.plane[3]
    return h.min(1)


def smooth_baseline_z(ep, win=15):
    from scipy.ndimage import median_filter
    return median_filter(ep.Tc_base[:, 2, 3], size=win, mode="nearest")


def full_unary(ep, sc, dres, fh, w_d=3.0, w_f=1.0):
    u = unary_cost(sc)
    d = np.nan_to_num(np.clip(dres, 0, 0.15), nan=0.0)
    # penetration of the floor beyond 10 cm (evaluated at the baseline depth; the plane fit and the baseline depth
    # have several-cm tolerance; the term only has to reject grossly impossible hypotheses such as mirror flips)
    pen = np.maximum(0.0, -fh - 0.10) / 0.05
    return u + w_d * d + w_f * pen


def mean_pose(Ts):
    """Chordal mean of rotations + mean translation."""
    U, _, Vt = np.linalg.svd(Ts[:, :3, :3].sum(0))
    R = U @ Vt
    if np.linalg.det(R) < 0: U[:, -1] *= -1; R = U @ Vt
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = Ts[:, :3, 3].mean(0)
    return T


def joint_refine(ep, FR, T_init, static_ranges, iters=150, chunk=96, sig_t=0.004, sig_R=0.03, sig_roll=0.035,
                 lr=(0.003, 0.001, 0.001), dev="cuda", w_data=None):
    """Jointly refine all frame poses (camera frame): per-frame silhouette terms + second-difference smoothness on
    translation and rotation + first-difference penalty on the roll rate about the bat axis (roll is nearly
    unobservable in the image). Frames inside static_ranges [(a,b) inclusive] keep their (pooled) pose fixed."""
    from fitter import losses, expm
    F = len(T_init)
    fixed = np.zeros(F, bool)
    for a, b in static_ranges:
        fixed[a:b + 1] = True
    Kt = torch.tensor(ep.K, dtype=torch.float32, device=dev); Kinv = torch.linalg.inv(Kt)
    R0 = torch.tensor(T_init[:, :3, :3], dtype=torch.float32, device=dev)
    t0 = torch.tensor(T_init[:, :3, 3], dtype=torch.float32, device=dev)
    u0 = (Kt @ t0.T).T; u0 = u0[:, :2] / u0[:, 2:]; z0 = t0[:, 2]
    w = torch.zeros(F, 3, device=dev, requires_grad=True)
    du = torch.zeros(F, 2, device=dev, requires_grad=True)  # pixels/100
    dz = torch.zeros(F, device=dev, requires_grad=True)
    free = torch.tensor(~fixed, dtype=torch.float32, device=dev)
    opt = torch.optim.Adam([{"params": [w], "lr": lr[0]}, {"params": [du], "lr": lr[1]}, {"params": [dz], "lr": lr[2]}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=0.0)
    fw = np.ones(F) if w_data is None else np.asarray(w_data, dtype=np.float64)
    Kc_all = np.stack([crop_K(ep.K, ep.wins[i]) for i in range(F)])
    ids_free = np.nonzero(~fixed)[0]

    def poses():
        R = R0 @ expm(w * free[:, None])
        u = u0 + du * free[:, None] * 100.0
        z = z0 * torch.exp(dz * free)
        ray = (Kinv @ torch.cat([u, torch.ones_like(u[:, :1])], 1).T).T
        return R, ray * z[:, None]

    chunks = [ids_free[s:s + chunk] for s in range(0, len(ids_free), chunk)]
    data_t = [{k: torch.tensor(np.stack([ep.datas[i][k] for i in ch]), device=dev) for k in ["M", "P", "D", "I"]}
              for ch in chunks]
    for it in range(iters):
        opt.zero_grad()
        for ch, dt in zip(chunks, data_t):
            R, t = poses()
            tot, *_ = losses(FR, R[ch], t[ch], Kc_all[ch], dt, w_col=0.0)
            (tot * torch.tensor(fw[ch], dtype=torch.float32, device=dev)).sum().backward()
        R, t = poses()
        a2t = t[:-2] - 2 * t[1:-1] + t[2:]
        a2R = R[:-2] - 2 * R[1:-1] + R[2:]
        rel = R[:-1].transpose(1, 2) @ R[1:]
        roll = 0.5 * (rel[:, 1, 0] - rel[:, 0, 1])
        sm = (a2t ** 2).sum() / sig_t ** 2 + (a2R ** 2).sum() / sig_R ** 2 + (roll ** 2).sum() / sig_roll ** 2
        sm.backward()
        opt.step(); sched.step()
    with torch.no_grad():
        R, t = poses()
    T = np.tile(np.eye(4), (F, 1, 1))
    T[:, :3, :3] = R.cpu().numpy(); T[:, :3, 3] = t.cpu().numpy()
    return T


BAT_CENTER = np.array([0.0, 0.0, 0.0616])  # mesh-frame centre of the bat's bounding box (on the long axis)


def baseline_center_depth(ep):
    """Camera-frame depth of the bat centre under the baseline poses."""
    c = ep.Tc_base[:, :3, :3] @ BAT_CENTER + ep.Tc_base[:, :3, 3]
    return c[:, 2]


def anchor_depth(T, z_target):
    """Move each pose along the viewing ray through the bat centre so that the centre's depth equals z_target
    (keeps the image position of the object and the rotation)."""
    T = T.copy()
    p = T[:, :3, :3] @ BAT_CENTER + T[:, :3, 3]
    s = z_target / p[:, 2]
    T[:, :3, 3] = p * s[:, None] - T[:, :3, :3] @ BAT_CENTER
    return T


def depth_targets(ep, s1, s2, med=9, sigma=2.0):
    """Per-frame target depth: baseline centre depth, median+Gaussian smoothed in the moving phase, and the
    phase median in the two static phases."""
    from scipy.ndimage import median_filter, gaussian_filter1d
    zc = baseline_center_depth(ep)
    zt = gaussian_filter1d(median_filter(zc, med, mode="nearest"), sigma, mode="nearest")
    zt[:s1 + 1] = np.median(zc[:s1 + 1]); zt[s2:] = np.median(zc[s2:])
    return zt


def transported_frames(Tc):
    """Minimal-twist (parallel transport) rotation sequence along the bat axis sequence of Tc."""
    from fitter import rot_a_to_b
    Q = np.zeros((len(Tc), 3, 3)); Q[0] = Tc[0, :3, :3]
    for i in range(1, len(Tc)):
        Q[i] = rot_a_to_b(Q[i - 1][:, 2], Tc[i, :3, 2]) @ Q[i - 1]
    return Q


def roll_scores(ep, Rfull, Tc, Q, frames, n_ang=24, bs=4):
    """Texture evidence for the roll about the bat axis: for each frame and roll angle phi (pose Q_i Rz(phi), same
    centre), negative correlation of image luminance with rendered texture luminance on object pixels."""
    from scipy.spatial.transform import Rotation
    angs = np.arange(n_ang) * 2 * np.pi / n_ang
    Rz = Rotation.from_euler("z", angs).as_matrix()
    out = np.zeros((len(frames), n_ang)); valid = np.zeros(len(frames), bool); sil = np.zeros((len(frames), n_ang))
    for fi, i in enumerate(frames):
        Ts = np.tile(np.eye(4), (n_ang, 1, 1))
        Ts[:, :3, :3] = Q[i][None] @ Rz
        c = Tc[i, :3, :3] @ np.array([0, 0, 0.0616]) + Tc[i, :3, 3]
        Ts[:, :3, 3] = c - Ts[:, :3, :3] @ np.array([0, 0, 0.0616])
        m, _, rgb = Rfull.render(Ts, crop_K(ep.K, ep.wins[i]), RES, RES, color=True)
        m = m.cpu().numpy() > 0; rgb = rgb.cpu().numpy()
        M = ep.datas[i]["M"] > 0.5; I = ep.datas[i]["I"]
        V = ~((ep.datas[i]["P"] > 0.5) & ~M)  # person pixels not on the object: don't care
        inter = (m & M[None]).sum((1, 2)); union = ((m | M[None]) & V[None]).sum((1, 2))
        sil[fi] = 1 - inter / np.maximum(union, 1)  # the scanned bat is slightly bent: roll changes the outline
        for k in range(n_ang):
            W = m[k] & M
            if W.sum() < 40:
                continue
            cl = rgb[k][W].mean(1); xl = I[W].mean(1)
            cl = cl - cl.mean(); xl = xl - xl.mean()
            out[fi, k] = -(cl * xl).sum() / np.sqrt((cl ** 2).sum() * (xl ** 2).sum() + 1e-9)
            valid[fi] = True
    return angs, out, valid, sil


def decode_roll(S, valid, static_ranges, sil=None, lam=15.0, lam_sil=60.0, sig_deg=8.0, cap=1000.0):
    """DP over roll bins. S (F,n) texture costs, sil (F,n) silhouette IoU losses; frames in a static range must
    keep the same roll; elsewhere roll changes cost (dphi/sig)^2."""
    F, n = S.shape
    U = lam * (S - S.min(1, keepdims=True))
    if sil is not None:
        U = U + lam_sil * (sil - sil.min(1, keepdims=True))
    U = np.where(valid[:, None], U, 0.0)
    step = 360.0 / n
    d = np.abs(np.arange(n)[:, None] - np.arange(n)[None]) * step
    d = np.minimum(d, 360 - d)
    P = np.minimum((d / sig_deg) ** 2, cap)
    Pstat = np.where(d == 0, 0.0, 1e6)
    in_static = np.zeros(F, bool)
    for a, b in static_ranges:
        in_static[a + 1:b + 1] = True  # transition (i-1 -> i) inside a static phase
    acc = U[0].copy(); back = np.zeros((F, n), int)
    for i in range(1, F):
        tot = acc[:, None] + (Pstat if in_static[i] else P)
        back[i] = tot.argmin(0)
        acc = tot[back[i], np.arange(n)] + U[i]
    k = np.zeros(F, int); k[-1] = acc.argmin()
    for i in range(F - 1, 0, -1):
        k[i - 1] = back[i, k[i]]
    return k


def apply_roll(Tc, Q, phi):
    from scipy.spatial.transform import Rotation
    T = Tc.copy()
    c = Tc[:, :3, :3] @ np.array([0, 0, 0.0616]) + Tc[:, :3, 3]
    T[:, :3, :3] = Q @ Rotation.from_euler("z", phi).as_matrix()
    T[:, :3, 3] = c - T[:, :3, :3] @ np.array([0, 0, 0.0616])
    return T
