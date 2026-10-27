#!/usr/bin/env python
"""Trajectory-level re-estimation of the skateboard pose for one monocular HOI video.

Method (see REPORT.md):
  1. Read video + SAM2 masks; crop around the object.
  2. Segment the episode into static / moving phases from the object-mask change between frames
     (pixels covered by the person are ignored).
  3. Hypotheses per phase: baseline rotation and its 180-deg flips about the mesh x / y / z axes.
     Each is refined locally by render-and-compare (nvdiffrast) and scored by silhouette IoU (outside the
     person mask) + texture NCC. The best pooled hypothesis per phase is kept.
  4. Board-on-floor model: the board stays upright (as in the baseline) at a fixed height above the
     released floor plane; each frame is (yaw about the floor normal, 2-D position in the floor plane).
     Static phases: one pose, pooled render-and-compare search over the frames of the phase.
     Moving phases: per-frame coarse-to-fine search, then a weighted least-squares smoother
     (2nd-difference penalty) that is continuous with the static poses.
  5. Depth re-anchoring: scale camera-frame positions along the viewing rays so that the episode median
     camera depth equals the baseline's (no absolute depth change; see REPORT.md for the cue analysis).
Deterministic: no random numbers are used; grid searches only.
"""
import argparse, json, os, sys, time
import numpy as np
import cv2, h5py, trimesh, torch
import torch.nn.functional as Fn
import nvdiffrast.torch as dr

torch.backends.cudnn.deterministic = True
torch.manual_seed(0); np.random.seed(0)

# ----------------------------------------------------------------------------- constants
PAD = 40                 # crop padding around the object mask (px)
SS = 2                   # render super-sampling for anti-aliased silhouettes / texture
W_NCC = 1.0              # weight of texture NCC relative to silhouette IoU in the score
MOTION_STEP = 3          # frame gap for the mask-change motion signal
MOTION_THR = 0.06        # mask change (1-IoU over 3 frames) above which the object is moving;
                         # static-phase noise is 0.00-0.05 (checked on the whole episode)
MIN_STATIC = 15          # minimum length (frames) of a static phase
MOVE_DILATE = 4          # frames added on each side of a moving segment (transition frames are
                         # estimated per frame, not forced to the static pose)
SMOOTH_LAMBDA = (3.0, 3.0, 3.0, 3.0, 3.0)     # 2nd-difference weights for (yaw, a, b, pitch, roll) relative
                                                # to data terms; chosen by held-out-frame validation (REPORT.md)
TILT_PENALTY = 0.003     # score penalty per degree of tilt (score units: IoU + NCC)
HYP_STRIDE = 20          # frame stride for the hypothesis (flip) test (margins are large: >= 0.035 per frame
                         # pooled in every phase in a stride-10 run, so stride 20 saves GPU time)
STATIC_STRIDE = 3        # frame stride for pooled static-phase search
# coarse-to-fine grids: (yaw offsets [deg], planar offsets [m])
GRIDS = [(np.arange(-12, 12.1, 3.0), np.arange(-0.045, 0.0451, 0.015)),
         (np.arange(-2, 2.1, 1.0), np.arange(-0.010, 0.0101, 0.005)),
         (np.arange(-0.5, 0.51, 0.5), np.arange(-0.0025, 0.00251, 0.0025))]


# ----------------------------------------------------------------------------- geometry helpers
def rodr(axis, ang):
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


class FloorModel:
    """Pose = Rot(n, yaw) @ R_ref, position = p0 + a*e1 + b*e2 (p0 on the plane at height h0)."""
    def __init__(self, plane, R_ref, p_ref, h0):
        n = np.asarray(plane[:3], float); d = float(plane[3])
        s = np.linalg.norm(n); n, d = n / s, d / s
        self.n, self.d = n, d
        e1 = np.cross(n, [0, 0, 1.0]); self.e1 = e1 / np.linalg.norm(e1); self.e2 = np.cross(n, self.e1)
        self.R_ref = R_ref
        self.p0 = p_ref - (p_ref @ n + d - h0) * n           # reference point at height h0
        self.vs = None
    def set_mesh(self, verts):
        self.vs = verts                                     # mesh-frame vertices (subsampled)
        self.u = self.R_ref.T @ self.n                      # floor normal (up) in mesh frame
        self.low0 = (verts @ self.u).min()
    def pose(self, yaw, a, b, pitch=0.0, roll=0.0):
        """pitch / roll: tilt about the mesh x / z axes; the height is adjusted so that the lowest point of the
        mesh keeps its height (the board pivots on the wheels that stay on the floor)"""
        T = np.eye(4)
        Rl = rodr(np.array([1.0, 0, 0]), pitch) @ rodr(np.array([0, 0, 1.0]), roll)
        T[:3, :3] = rodr(self.n, yaw) @ self.R_ref @ Rl
        dh = 0.0
        if (pitch != 0 or roll != 0) and self.vs is not None:
            dh = self.low0 - ((self.vs @ Rl.T) @ self.u).min()
        T[:3, 3] = self.p0 + a * self.e1 + b * self.e2 + dh * self.n
        return T
    def coords(self, T):
        """(yaw, a, b) of an (approximately upright) pose; yaw measured relative to R_ref."""
        Rrel = T[:3, :3] @ self.R_ref.T
        x = Rrel @ self.e1
        yaw = np.arctan2(x @ self.e2, x @ self.e1)
        dp = T[:3, 3] - self.p0
        return np.array([yaw, dp @ self.e1, dp @ self.e2])


# ----------------------------------------------------------------------------- renderer / scorer
class Renderer:
    def __init__(self, mesh, K, dev='cuda'):
        self.dev = dev
        self.v = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=dev)
        self.f = torch.tensor(np.asarray(mesh.faces), dtype=torch.int32, device=dev)
        uv = np.asarray(mesh.visual.uv).copy(); uv[:, 1] = 1 - uv[:, 1]
        self.uv = torch.tensor(uv, dtype=torch.float32, device=dev)
        tex = np.asarray(mesh.visual.material.baseColorTexture.convert('RGB')).astype(np.float32) / 255.
        self.tex = torch.tensor(tex, device=dev)[None].contiguous()
        self.glctx = dr.RasterizeCudaContext()
        self.K = K

    def render(self, Tc, crop, res):
        x0, y0, w, h = crop
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2] - x0, self.K[1, 2] - y0
        n_, f_ = 0.05, 20.0
        P = torch.tensor([[2 * fx / w, 0, (2 * cx - w) / w, 0], [0, 2 * fy / h, (2 * cy - h) / h, 0],
                          [0, 0, (f_ + n_) / (f_ - n_), -2 * f_ * n_ / (f_ - n_)], [0, 0, 1, 0]],
                         dtype=torch.float32, device=self.dev)
        T = torch.as_tensor(Tc, dtype=torch.float32, device=self.dev)
        vc = self.v @ T[:, :3, :3].transpose(1, 2) + T[:, None, :3, 3]
        clip = torch.cat([vc, torch.ones_like(vc[..., :1])], -1) @ P.T
        rast, _ = dr.rasterize(self.glctx, clip.contiguous(), self.f, resolution=list(res))
        uvi, _ = dr.interpolate(self.uv[None], rast, self.f)
        col = dr.texture(self.tex, uvi, filter_mode='linear')
        mask = (rast[..., 3] > 0).float()
        return col * mask[..., None], mask


class Scorer:
    def __init__(self, R, crops):
        self.R = R; self.crops = crops

    @torch.no_grad()
    def score(self, i, Tc, chunk=150):
        c = self.crops[i]; x0, y0, w, h = c['box']; dev = self.R.dev
        g = torch.tensor(c['gray'], device=dev)
        o = torch.tensor(c['obj'], device=dev).float()
        v = 1 - torch.tensor(c['per'], device=dev).float()
        out = []
        for s in range(0, len(Tc), chunk):
            col, mk = self.R.render(Tc[s:s + chunk], c['box'], (h * SS, w * SS))
            gr = Fn.avg_pool2d(col.mean(-1)[:, None], SS)[:, 0]
            mk = Fn.avg_pool2d(mk[:, None], SS)[:, 0]
            inter = (mk * o * v).sum((1, 2)); uni = ((mk + o - mk * o) * v).sum((1, 2))
            iou = inter / uni.clamp(min=1)
            Wt = (mk > 0.99).float() * v; nW = Wt.sum((1, 2)).clamp(min=1)
            a = (gr - ((gr * Wt).sum((1, 2)) / nW)[:, None, None]) * Wt
            b = (g - ((g * Wt).sum((1, 2)) / nW)[:, None, None]) * Wt
            ncc = (a * b).sum((1, 2)) / torch.sqrt((a * a).sum((1, 2)) * (b * b).sum((1, 2)) + 1e-9)
            ncc = torch.where(nW > 30, ncc, torch.zeros_like(ncc))
            out.append(torch.stack([iou, ncc], 1))
        return torch.cat(out).cpu().numpy().astype(np.float64)


# ----------------------------------------------------------------------------- search
def local_search(score_fn, model, x0, grids=GRIDS, tilt=(0.0, 0.0)):
    """coarse-to-fine grid search of (yaw, a, b) around x0; score_fn(list of world poses) -> (B,) scores"""
    best = np.array(x0, float); bs = None
    for ys, ab in grids:
        C = np.array([(best[0] + np.radians(y), best[1] + a, best[2] + b) for y in ys for a in ab for b in ab])
        s = score_fn([model.pose(*c, *tilt) for c in C])
        k = int(np.argmax(s)); best = C[k]; bs = s[k]
    return best, bs


def segment_phases(obj, per, F):
    """returns list of (start, end_exclusive, 'static'|'moving')"""
    O = obj[:, ::4, ::4]; P = per[:, ::4, ::4]
    mot = np.zeros(F)
    for i in range(F):
        j = max(i - MOTION_STEP, 0); vis = ~(P[i] | P[j])
        a = O[i] & vis; b = O[j] & vis
        mot[i] = 1 - (a & b).sum() / max((a | b).sum(), 1)
    # moving frame i: change between i-3 and i -> frames i-3..i are moving
    mv = np.zeros(F, bool)
    for i in np.nonzero(mot > MOTION_THR)[0]:
        mv[max(i - MOTION_STEP, 0):i + 1] = True
    # close short static gaps, then dilate moving segments
    lab = mv.copy(); i = 0
    while i < F:
        if not lab[i]:
            j = i
            while j < F and not lab[j]: j += 1
            if (j - i) < MIN_STATIC and i > 0 and j < F: lab[i:j] = True
            i = j
        else: i += 1
    dil = lab.copy()
    for i in np.nonzero(lab)[0]: dil[max(i - MOVE_DILATE, 0):min(i + MOVE_DILATE + 1, F)] = True
    phases = []; i = 0
    while i < F:
        j = i
        while j < F and dil[j] == dil[i]: j += 1
        phases.append((i, j, 'moving' if dil[i] else 'static')); i = j
    return phases, mot


def tilt_search(score_fn, model, x):
    """grid search of (pitch, roll) for fixed yaw; the planar position (a, b) is re-fitted for every tilt
    (the pivot about the wheels moves the board in the image). A small penalty per degree keeps the board
    flat unless the image clearly prefers a tilt."""
    best_t = np.zeros(2); best_x = np.array(x, float); bs = None
    ab = np.arange(-0.01, 0.0101, 0.005)
    # pitch (nose/tail lift) is not searched: in a development run with pitch in [-4, 4] deg it was selected
    # (|pitch| <= 1 deg) in only 5 of 187 moving frames; dropped to save GPU time. Roll is searched.
    for ps, rs in [(np.array([0.0]), np.arange(-6, 6.1, 2.0)), (np.array([0.0]), np.arange(-1, 1.1, 1.0))]:
        C = [(best_t[0] + np.radians(p), best_t[1] + np.radians(r), best_x[1] + da, best_x[2] + db)
             for p in ps for r in rs for da in ab for db in ab]
        C = np.array(C)
        sc = score_fn([model.pose(best_x[0], c[2], c[3], c[0], c[1]) for c in C]) - TILT_PENALTY * np.degrees(np.abs(C[:, :2]).sum(1))
        k = int(np.argmax(sc)); best_t = C[k, :2]; best_x = np.array([best_x[0], C[k, 2], C[k, 3]]); bs = sc[k]
    return best_t, best_x, bs


def smooth_trajectory(F, phases, meas, wts, static_pose, lam):
    """weighted LS: per-frame variables, data terms in moving phases, hard-ish equality to the phase pose in
    static phases (large weight), 2nd-difference penalty everywhere. meas (F,3) unwrapped yaw,a,b."""
    from scipy.sparse import lil_matrix, csr_matrix
    from scipy.sparse.linalg import spsolve
    D = meas.shape[1]; X = np.zeros((F, D))
    for d in range(D):
        A = lil_matrix((F, F)); rhs = np.zeros(F)
        for (s, e, kind) in phases:
            for t in range(s, e):
                if kind == 'static':
                    w = 1e4; m = static_pose[(s, e)][d]
                else:
                    w = wts[t]; m = meas[t, d]
                if w > 0:
                    A[t, t] += w; rhs[t] += w * m
        for t in range(1, F - 1):
            idx = [t - 1, t, t + 1]; c = [1, -2, 1]
            for p in range(3):
                for q in range(3):
                    A[idx[p], idx[q]] += lam[d] * c[p] * c[q]
        X[:, d] = spsolve(csr_matrix(A), rhs)
    return X


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='/work/inputs')
    ap.add_argument('--out', default='/work/outputs/object_poses_world.npz')
    ap.add_argument('--diag', default=None, help='optional path to save diagnostics (npz)')
    ap.add_argument('--lam', type=float, nargs=5, default=SMOOTH_LAMBDA, help='(experiments only) smoothing weights')
    ap.add_argument('--no_tilt', action='store_true', help='(experiments only) keep the board flat in moving phases')
    ap.add_argument('--holdout', type=int, default=0, help='(validation only) drop data term of every k-th moving frame')
    args = ap.parse_args()
    t_start = time.time()
    I = args.inputs
    cam = json.load(open(f'{I}/camera.json'))
    K = np.array(cam['K'], float); Twc = np.array(cam['T_world_cam'], float); Tcw = np.linalg.inv(Twc)
    plane = np.array(json.load(open(f'{I}/ground_plane.json'))['plane'], float)
    rec = np.load(f'{I}/reconstruction.npz')
    Tb = rec['T_world_mesh'].astype(np.float64); F = len(Tb)
    mesh = trimesh.load(f'{I}/object_mesh.glb', force='scene', process=False).to_geometry()
    mh = h5py.File(f'{I}/masks.h5', 'r')
    print(f'[load] F={F}', flush=True)
    obj = mh['object'][:]; per = mh['person'][:]

    # ---- crops (object mask bbox, or baseline projection if the mask is empty)
    corners = trimesh.bounds.corners(mesh.bounds)
    cap = cv2.VideoCapture(f'{I}/video.mp4'); crops = []
    for i in range(F):
        ok, fr = cap.read(); assert ok, f'video ended at {i}'
        ys, xs = np.nonzero(obj[i])
        pc = (Tcw @ Tb[i])[:3]; pts = corners @ pc[:, :3].T + pc[:, 3]; uv = pts[:, :2] / pts[:, 2:] * [K[0, 0], K[1, 1]] + K[:2, 2]
        if len(xs) > 50:
            x0, x1, y0, y1 = min(xs.min(), uv[:, 0].min()), max(xs.max(), uv[:, 0].max()), min(ys.min(), uv[:, 1].min()), max(ys.max(), uv[:, 1].max())
        else:
            x0, x1, y0, y1 = uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max()
        x0 = int(max(x0 - PAD, 0)); y0 = int(max(y0 - PAD, 0)); x1 = int(min(x1 + PAD, fr.shape[1])); y1 = int(min(y1 + PAD, fr.shape[0]))
        g = fr[y0:y1, x0:x1].astype(np.float32).mean(-1) / 255.
        crops.append(dict(box=(x0, y0, x1 - x0, y1 - y0), gray=g, obj=obj[i, y0:y1, x0:x1].copy(),
                          per=per[i, y0:y1, x0:x1].copy(), n_obj=len(xs)))
    cap.release()

    # ---- phases
    phases, mot = segment_phases(obj, per, F)
    del obj, per
    print('[phases]', phases, flush=True)

    R = Renderer(mesh, K); S = Scorer(R, crops)
    def score_fn_for(i):
        def f(poses):
            Tc = np.stack([Tcw @ P for P in poses])
            sc = S.score(i, Tc)
            return sc[:, 0] + W_NCC * sc[:, 1]
        return f

    # ---- floor model: baseline is upright; fixed height h0 = median baseline height of the mesh origin
    nrm = plane[:3] / np.linalg.norm(plane[:3]); dpl = plane[3] / np.linalg.norm(plane[:3])
    h0 = float(np.median(Tb[:, :3, 3] @ nrm + dpl))
    tilt = np.degrees(np.arccos(np.clip(np.abs(Tb[:, :3, 1] @ nrm), 0, 1)))
    print(f'[floor] h0={h0:.4f} m, baseline tilt of mesh y vs floor normal: max {tilt.max():.2f} deg', flush=True)

    # ---- hypotheses: baseline rotation composed with 180-deg flips about the mesh axes (about mesh origin)
    HYP = {'id': np.eye(4), 'flip_y': np.diag([-1, 1, -1, 1.]), 'flip_x': np.diag([1, -1, -1, 1.]), 'flip_z': np.diag([-1, -1, 1, 1.])}
    def model_for(Href):
        Tr = Tb[0] @ Href
        return FloorModel(plane, Tr[:3, :3], Tb[0][:3, 3], h0)
    hyp_choice = {}; hyp_table = {}
    for (s, e, kind) in phases:
        frs = [t for t in range(s, e, HYP_STRIDE) if crops[t]['n_obj'] > 100] or [s]
        tot = {}
        for hn, H in HYP.items():
            M = model_for(H)
            acc = []
            for t in frs:
                x0 = M.coords(Tb[t] @ H)
                _, bs = local_search(score_fn_for(t), M, x0, GRIDS[:2])
                acc.append(bs)
            tot[hn] = float(np.mean(acc))
        hyp_table[(s, e)] = tot
        # the trajectory model keeps the board upright on the floor, so only yaw-type hypotheses (identity or
        # 180 deg about the mesh y = floor-normal axis) can be used; x/z flips (wheels up) are scored for the report
        hyp_choice[(s, e)] = max(['id', 'flip_y'], key=tot.get)
        if max(tot, key=tot.get) not in ('id', 'flip_y'):
            print(f'[hyp] WARNING: wheels-up hypothesis scored best in {s}-{e}', flush=True)
        print(f'[hyp] {s}-{e} {kind}: ' + ' '.join(f'{k}={v:.3f}' for k, v in tot.items()) + f' -> {hyp_choice[(s, e)]}', flush=True)

    # one reference model (flip of the first phase); per-phase flips enter through the initial yaw
    H0 = HYP[hyp_choice[phases[0][:2]]]
    M = model_for(H0)
    M.set_mesh(np.asarray(mesh.vertices)[::10])
    def init_coords(t, ph):
        Tt = Tb[t] @ HYP[hyp_choice[ph]]
        return M.coords(Tt)

    # ---- static phases: one pose each, pooled search
    static_pose = {}
    for (s, e, kind) in phases:
        if kind != 'static': continue
        frs = [t for t in range(s, e, STATIC_STRIDE) if crops[t]['n_obj'] > 100]
        xs = np.array([init_coords(t, (s, e)) for t in range(s, e)])
        x0 = np.array([np.angle(np.mean(np.exp(1j * xs[:, 0]))), np.median(xs[:, 1]), np.median(xs[:, 2])])
        def pooled(poses, frs=frs):
            tot = np.zeros(len(poses))
            for t in frs: tot += score_fn_for(t)(poses)
            return tot / len(frs)
        best, bs = local_search(pooled, M, x0)
        static_pose[(s, e)] = best
        print(f'[static] {s}-{e}: yaw {np.degrees(best[0]):.2f} a {best[1]:.4f} b {best[2]:.4f} score {bs:.3f}', flush=True)

    # ---- moving phases: per-frame search around the (flip-corrected) baseline
    meas = np.full((F, 5), np.nan); wts = np.zeros(F); fscore = np.full(F, np.nan)
    for (s, e, kind) in phases:
        if kind != 'moving': continue
        for t in range(s, e):
            if crops[t]['n_obj'] < 100: continue
            x0 = init_coords(t, (s, e))
            sf = score_fn_for(t)
            best, bs = local_search(sf, M, x0)
            tl = np.zeros(2)
            if not args.no_tilt:
                tl, best, _ = tilt_search(sf, M, best)
                best, bs = local_search(sf, M, best, GRIDS[1:], tilt=tuple(tl))
            meas[t] = [*best, *tl]; fscore[t] = bs
            wts[t] = max(bs - 0.8, 0.05)     # weight: confident frames (IoU+NCC high) count more
        print(f'[moving] {s}-{e} done ({time.time() - t_start:.0f}s)', flush=True)

    # ---- unwrap yaw consistently along time (static poses + measurements), then smooth
    seq = np.full(F, np.nan)
    for (s, e, kind) in phases:
        seq[s:e] = static_pose[(s, e)][0] if kind == 'static' else meas[s:e, 0]
    valid = ~np.isnan(seq); idx = np.nonzero(valid)[0]
    unw = np.unwrap(seq[idx]); seq[idx] = unw
    meas_u = meas.copy(); meas_u[:, 0] = seq
    sp_u = {}
    for (s, e, kind) in phases:
        if kind == 'static':
            sp_u[(s, e)] = np.array([seq[s], *static_pose[(s, e)][1:], 0.0, 0.0])
    meas_u[np.isnan(meas_u)] = 0.0
    if args.holdout:
        for (s, e, kind) in phases:
            if kind == 'moving': wts[s + 2:e - 2:args.holdout] = 0.0
    X = smooth_trajectory(F, phases, meas_u, wts, sp_u, args.lam)
    T_out = np.stack([M.pose(*X[t]) for t in range(F)])

    # ---- depth re-anchoring along the viewing rays (episode median camera z = baseline's)
    Tc_out = Tcw[None] @ T_out; Tc_b = Tcw[None] @ Tb
    scale = np.median(Tc_b[:, 2, 3]) / np.median(Tc_out[:, 2, 3])
    Tc_out[:, :3, 3] *= scale
    T_out = Twc[None] @ Tc_out
    print(f'[anchor] depth scale along rays {scale:.4f}', flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, T_world_mesh=T_out.astype(np.float64))
    if args.diag:
        np.savez(args.diag, phases=np.array([(s, e, k == 'moving') for s, e, k in phases]), mot=mot, meas=meas,
                 fscore=fscore, X=X, scale=scale, h0=h0,
                 hyp=json.dumps({f'{k[0]}-{k[1]}': v for k, v in hyp_table.items()}),
                 hyp_choice=json.dumps({f'{k[0]}-{k[1]}': v for k, v in hyp_choice.items()}))
    print(f'[done] {args.out}  {time.time() - t_start:.0f}s', flush=True)


if __name__ == '__main__':
    main()
