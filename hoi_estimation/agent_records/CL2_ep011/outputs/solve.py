#!/usr/bin/env python
"""Trajectory-level refinement of the 6-DoF object poses of one HOI video (episode 11, light-blue book).

usage: python solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
Everything is recomputed from the inputs (video, SAM2 masks, baseline reconstruction, mesh, camera);
deterministic (fixed seeds, no random restarts). ~6 min on one GPU, < 3 GB GPU memory.

 1. baseline poses -> camera frame. The baseline rotations are kept as prior (their 180-deg symmetry choice was
    verified by eye at frames 0, 430, 520, 700: spine text / front-cover title visible as rendered).
 2. static background (temporal median) -> table-top mask; used as an occluder (unknown pixels) whenever the book
    is not resting fully visible on the table (it passes under / behind the table 4 times).
 3. phase segmentation: static = object-mask IoU with frames i+-5 >= 0.85 AND photometric stillness (mean grey
    difference to frames i+-5 in the object region, person excluded) < 6 grey levels; runs >= 20 frames.
 4. initial trajectory: static phases -> robust mean rotation / median image position / median baseline depth;
    moving frames -> baseline; depth along the ray = robustly smoothed baseline depth (continuous at boundaries).
 5. joint optimisation of rotation + image position of all 901 frames (depth fixed from 4): chamfer silhouette
    term against the SAM2 mask (occluded pixels ignored), Cauchy priors to the baseline rotation / position,
    second-difference smoothness while moving, first-difference (near-constant) inside static phases.
 6. texture fine alignment: shading-normalised texture fit of the textured mesh render for +-5 px lateral shifts;
    pooled over each static phase, temporally smoothed in moving phases (corrects ~1-3 px silhouette bias).
 7. depth re-anchoring per phase to the baseline's median camera z (tapered at phase boundaries in moving phases).
 8. writes T_world_mesh (F,4,4) float64 + *_diag.json / *_diag.npz diagnostics next to the output.
"""
import argparse, json, os, time
import numpy as np, cv2, h5py, torch, trimesh
import nvdiffrast.torch as dr
import torch.nn.functional as Fn
from scipy.ndimage import median_filter, gaussian_filter1d

torch.manual_seed(0); np.random.seed(0)
DEV = 'cuda'

# ----------------------------------------------------------------------------- constants
STATIC_IOU = 0.85        # mask IoU with frames i-5 and i+5 for a frame to count as static
STATIC_MIN_LEN = 20      # minimum static run length (frames)
STATIC_PHOTO = 6.0       # photometric stillness threshold (grey levels, see static_phases)
CROP = 112               # crop resolution for silhouette rendering
MIN_VIS_PX = 60          # frames whose visible object mask is smaller get no silhouette term
# optimisation weights; per-frame data term = log(1 + (L/DATA_C)^2), L = chamfer silhouette distance (crop px)
DATA_C = 3.0
N_SURF = 1500            # mesh surface samples
N_MASKPX = 300           # object-mask pixels sampled per frame for the coverage term
W_ROT_PRIOR, ROT_C = 0.2, np.radians(10.0)      # Cauchy prior to baseline rotation (scale 10 deg)
W_UV_PRIOR, UV_C = 0.02, 8.0                    # Cauchy prior to baseline image position (scale 8 px)
W_ACC_R = 30.0                                  # chordal second difference of rotations
W_ACC_UV = 0.01                                 # second difference of image position (px^2)
W_VEL_R_STATIC = 300.0                          # first difference of rotation inside static runs
W_VEL_UV_STATIC = 0.5                           # first difference of image position inside static runs (px^2)
ITERS = 400
TEX_R = 5                # texture alignment search radius (px)
TEX_MIN_PX = 300         # minimum visible object pixels for a texture score
TEX_SMOOTH = 4.0         # temporal smoothing (frames) of score maps in moving phases


def so3_exp(w):
    Kx = torch.zeros(w.shape[:-1] + (3, 3), device=w.device, dtype=w.dtype)
    Kx[..., 0, 1] = -w[..., 2]; Kx[..., 0, 2] = w[..., 1]; Kx[..., 1, 0] = w[..., 2]
    Kx[..., 1, 2] = -w[..., 0]; Kx[..., 2, 0] = -w[..., 1]; Kx[..., 2, 1] = w[..., 0]
    return torch.linalg.matrix_exp(Kx)


def rot_mean(Rs, iters=10):
    """robust (Weiszfeld-like) chordal mean of rotations"""
    w = np.ones(len(Rs))
    for _ in range(iters):
        M = (w[:, None, None] * Rs).sum(0)
        U, _, Vt = np.linalg.svd(M); R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
        d = np.linalg.norm(Rs - R[None], axis=(1, 2)); w = 1 / np.maximum(d, 1e-3)
    return R


def rotdiff_deg(Ra, Rb):
    c = (np.trace(np.swapaxes(Ra, -1, -2) @ Rb, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def read_video(path, step=15, grey=False):
    """returns (number of frames, every step-th frame[, all frames as half-resolution grey uint8])"""
    cap = cv2.VideoCapture(path); frames = []; greys = []; n = 0
    while True:
        ok, im = cap.read()
        if not ok: break
        if n % step == 0: frames.append(im)
        if grey: greys.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[::2, ::2])
        n += 1
    return (n, frames, np.stack(greys)) if grey else (n, frames)


def table_mask(frames):
    """table-top: bright, low saturation connected component of the median background image that
    contains the image point (780,630) (centre of the table top in this static camera, checked by eye)"""
    bg = np.median(np.stack(frames), 0).astype(np.uint8)
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] < 45) & (hsv[..., 2] > 185)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    seed = lab[630, 780]
    tm = (lab == seed).astype(np.uint8) if seed > 0 else np.zeros_like(m)
    tm = cv2.morphologyEx(tm, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    tm = cv2.dilate(tm, np.ones((5, 5), np.uint8))   # include the table's front edge band
    return tm > 0


def static_phases(obj_ds, per_ds, grey):
    """static frames: (a) object mask IoU with frames i-5 and i+5 >= STATIC_IOU (4x down-sampled masks) and
    (b) photometric stillness: mean |grey(i) - grey(i+-5)| over the (dilated) object mask without person
    pixels < STATIC_PHOTO grey levels (half-res frames). (b) catches 1-5 px slides that (a) misses."""
    N = len(obj_ds)
    def iou(a, b):
        u = (a | b).sum(); return (a & b).sum() / u if u else 0.0
    s = np.array([min(iou(obj_ds[i], obj_ds[max(i - 5, 0)]), iou(obj_ds[i], obj_ds[min(i + 5, N - 1)]))
                  for i in range(N)])
    H2, W2 = grey.shape[1:]
    ph = np.full(N, np.inf)
    for i in range(N):
        o = cv2.resize(obj_ds[i].astype(np.uint8), (W2, H2), interpolation=cv2.INTER_NEAREST)
        if o.sum() < 50: continue
        o = cv2.dilate(o, np.ones((5, 5), np.uint8)) > 0
        vals = []
        for j in (i - 5, i + 5):
            if j < 0 or j >= N: continue
            p = cv2.resize((per_ds[i] | per_ds[j]).astype(np.uint8), (W2, H2), interpolation=cv2.INTER_NEAREST) > 0
            m = o & ~p
            if m.sum() < 30: continue
            vals.append(np.abs(grey[i][m].astype(np.float32) - grey[j][m]).mean())
        if vals: ph[i] = max(vals)
    s = np.where(ph < STATIC_PHOTO, s, 0.0)
    st = s >= STATIC_IOU
    # close 1-3 frame gaps, then keep long runs
    st2 = st.copy()
    for i in range(N):
        if not st[i]:
            l = st[max(0, i - 3):i].any(); r = st[i + 1:i + 4].any(); st2[i] = l and r
    runs = []; i = 0
    while i < N:
        if st2[i]:
            j = i
            while j + 1 < N and st2[j + 1]: j += 1
            if j - i + 1 >= STATIC_MIN_LEN: runs.append((i, j + 1))
            i = j + 1
        else: i += 1
    return runs, s


class SilRenderer:
    def __init__(self, mesh, K):
        import open3d as o3d
        om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
                                       o3d.utility.Vector3iVector(np.asarray(mesh.faces)))
        om = om.merge_close_vertices(1e-5).simplify_quadric_decimation(3000)
        self.v = torch.tensor(np.asarray(om.vertices), dtype=torch.float32, device=DEV)
        self.f = torch.tensor(np.asarray(om.triangles), dtype=torch.int32, device=DEV)
        self.ctx = dr.RasterizeCudaContext(); self.K = K

    def render(self, R, t, boxes):
        """R (B,3,3), t (B,3) camera frame; boxes (B,4) tensor x0,y0,side -> soft masks (B,CROP,CROP)"""
        K = self.K; S = CROP
        s = S / boxes[:, 2]
        fx = K[0, 0] * s; fy = K[1, 1] * s; cx = (K[0, 2] - boxes[:, 0]) * s; cy = (K[1, 2] - boxes[:, 1]) * s
        vc = torch.einsum('bij,nj->bni', R, self.v) + t[:, None]
        x, y, z = vc[..., 0], vc[..., 1], vc[..., 2]
        n, f = 0.05, 20.
        clip = torch.stack([2 * fx[:, None] / S * x + (2 * cx[:, None] / S - 1) * z,
                            2 * fy[:, None] / S * y + (2 * cy[:, None] / S - 1) * z,
                            z * (f + n) / (f - n) - 2 * f * n / (f - n), z], -1).contiguous()
        rast, _ = dr.rasterize(self.ctx, clip, self.f, (S, S))
        m = (rast[..., 3:4] > 0).float()
        m = dr.antialias(m, rast, clip, self.f)
        return m[..., 0].clamp(0, 1)


class TexRenderer:
    """textured full-resolution mesh renderer (albedo) + per-pixel box-face label"""
    def __init__(self, mesh, K):
        self.v = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32, device=DEV)
        self.f = torch.tensor(np.asarray(mesh.faces), dtype=torch.int32, device=DEV)
        uv = np.asarray(mesh.visual.uv).copy(); uv[:, 1] = 1 - uv[:, 1]
        self.uv = torch.tensor(uv, dtype=torch.float32, device=DEV)
        tex = np.asarray(mesh.visual.material.baseColorTexture.convert('RGB')).astype(np.float32) / 255.
        self.tex = torch.tensor(tex, device=DEV)[None].contiguous()
        n = mesh.face_normals; ax = np.abs(n).argmax(1); sg = np.sign(n[np.arange(len(n)), ax])
        self.flab = torch.tensor(np.concatenate([[6], ax * 2 + (sg > 0)]), device=DEV)
        self.ctx = dr.RasterizeCudaContext(); self.K = K

    def render(self, T, box, S):
        """T (B,4,4) camera<-mesh, box=(x0,y0,side) -> grey albedo (B,S,S), face label (B,S,S) (6 = background)"""
        x0, y0, side = box; K = self.K; s = S / side
        fx, fy, cx, cy = K[0, 0] * s, K[1, 1] * s, (K[0, 2] - x0) * s, (K[1, 2] - y0) * s
        vc = torch.einsum('bij,nj->bni', T[:, :3, :3], self.v) + T[:, None, :3, 3]
        x, y, z = vc[..., 0], vc[..., 1], vc[..., 2]; n, f = 0.05, 20.
        clip = torch.stack([2 * fx / S * x + (2 * cx / S - 1) * z, 2 * fy / S * y + (2 * cy / S - 1) * z,
                            z * (f + n) / (f - n) - 2 * f * n / (f - n), z], -1).contiguous()
        rast, _ = dr.rasterize(self.ctx, clip, self.f, (S, S))
        texc, _ = dr.interpolate(self.uv[None], rast, self.f)
        rgb = dr.texture(self.tex, texc, filter_mode='linear')
        return rgb.mean(-1), self.flab[rast[..., 3].long()]


def gauss_blur(x, sigma):
    k = int(2 * round(2 * sigma) + 1); g = torch.exp(-(torch.arange(k, device=x.device) - k // 2) ** 2 / (2 * sigma ** 2))
    g = (g / g.sum()).float(); x = x[:, None]
    x = Fn.conv2d(Fn.pad(x, (k // 2, k // 2, 0, 0), mode='replicate'), g.view(1, 1, 1, k))
    x = Fn.conv2d(Fn.pad(x, (0, 0, k // 2, k // 2), mode='replicate'), g.view(1, 1, k, 1))
    return x[:, 0]


def texture_scores(T_cam, K, mesh, video, H5, R=TEX_R, S=128):
    """for every frame: score map (2R+1,2R+1) of the shading-normalised texture fit (R^2 of grey levels with a
    separate gain/offset per box face) of the textured render shifted by (dy,dx) image pixels (lateral translation
    at constant depth) against the video frame; person pixels are ignored. NaN where the object is barely visible."""
    tr = TexRenderer(mesh, K); N = len(T_cam)
    corners = np.array(trimesh.bounds.corners(mesh.bounds))
    maps = np.full((N, 2 * R + 1, 2 * R + 1), np.nan)
    sh = np.array([(dy, dx) for dy in range(-R, R + 1) for dx in range(-R, R + 1)], float)
    cap = cv2.VideoCapture(video)
    for i in range(N):
        ok, img = cap.read()
        om = H5['object'][i]
        if om.sum() < TEX_MIN_PX: continue
        pm = H5['person'][i]
        T = T_cam[i]; pc = corners @ T[:3, :3].T + T[:3, 3]; pu = (K @ pc.T).T; pu = pu[:, :2] / pu[:, 2:]
        c = pu.mean(0); side = 1.3 * (pu.max(0) - pu.min(0)).max() + 40; box = (c[0] - side / 2, c[1] - side / 2, side)
        s = S / side; M = np.float32([[s, 0, -box[0] * s], [0, s, -box[1] * s]])
        p = cv2.warpAffine(pm.astype(np.float32), M, (S, S)) > 0.5
        im = cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_AREA).astype(np.float32).mean(-1) / 255
        Ts = np.repeat(T[None], len(sh), 0)
        Ts[:, 0, 3] += sh[:, 1] * T[2, 3] / K[0, 0]; Ts[:, 1, 3] += sh[:, 0] * T[2, 3] / K[1, 1]
        with torch.no_grad():
            gr, lab = tr.render(torch.tensor(Ts, dtype=torch.float32, device=DEV), box, S)
            gr = gauss_blur(gr, 1.0)
            gi = gauss_blur(torch.tensor(im, device=DEV)[None], 0.7)[0]
            keep = torch.tensor(~p, device=DEV)[None]
            rs = torch.zeros(len(sh), device=DEV); vs = torch.zeros(len(sh), device=DEV)
            for fl in range(6):
                sel = ((lab == fl) & keep).float(); cnt = sel.sum((1, 2))
                ok = cnt >= 20; cc = cnt.clamp_min(1)
                mx = (gr * sel).sum((1, 2)) / cc; my = (gi[None] * sel).sum((1, 2)) / cc
                x = (gr - mx[:, None, None]) * sel; y = (gi[None] - my[:, None, None]) * sel
                sxy = (x * y).sum((1, 2)); sxx = (x * x).sum((1, 2)); syy = (y * y).sum((1, 2))
                g = (sxy / (sxx + 1e-9)).clamp_min(0)
                rs += torch.where(ok, syy - 2 * g * sxy + g * g * sxx, 0); vs += torch.where(ok, syy, 0)
            sc = (1 - rs / vs.clamp_min(1e-9)).cpu().numpy()
        maps[i] = sc.reshape(2 * R + 1, 2 * R + 1)
    return maps


def peak2d(m):
    """argmax with separable quadratic sub-pixel refinement; returns (dy, dx) relative to the map centre"""
    R = m.shape[0] // 2
    a, b = np.unravel_index(np.nanargmax(m), m.shape); off = []
    for (k, axis) in ((a, 0), (b, 1)):
        line = m[:, b] if axis == 0 else m[a, :]
        d = 0.0
        if 0 < k < len(line) - 1:
            den = line[k - 1] - 2 * line[k] + line[k + 1]
            if den < 0: d = float(np.clip(0.5 * (line[k - 1] - line[k + 1]) / den, -0.5, 0.5))
        off.append(k + d - R)
    return np.array(off)


def phase_list(runs, N):
    bounds = []; prev = 0
    for a, b in runs:
        if a > prev: bounds.append((prev, a, 'moving'))
        bounds.append((a, b, 'static')); prev = b
    if prev < N: bounds.append((prev, N, 'moving'))
    return bounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='/work/inputs')
    ap.add_argument('--out', default='/work/outputs/object_poses_world.npz')
    ap.add_argument('--iters', type=int, default=ITERS)
    ap.add_argument('--holdout', type=int, default=0)
    ap.add_argument('--no_texture', action='store_true')
    ap.add_argument('--cache', default=None, help='development only: crop cache file')
    args = ap.parse_args()
    t_start = time.time()
    inp = args.inputs
    cam = json.load(open(f'{inp}/camera.json'))
    K = np.array(cam['K'], dtype=np.float64); Twc = np.array(cam['T_world_cam'], dtype=np.float64)
    rec = np.load(f'{inp}/reconstruction.npz')
    Tw_base = rec['T_world_mesh'].astype(np.float64); N = len(Tw_base)
    Tc_base = np.linalg.inv(Twc)[None] @ Tw_base
    mesh = trimesh.load(f'{inp}/object_mesh.glb', force='scene', process=False).to_geometry()
    ctr = mesh.bounds.mean(0)                       # mesh-file bbox centre: the tracked point
    Rb = Tc_base[:, :3, :3]
    pc_b = (Rb @ ctr) + Tc_base[:, :3, 3]           # camera-frame centre (baseline)
    zb = pc_b[:, 2]
    uvb = (K @ pc_b.T).T; uvb = uvb[:, :2] / uvb[:, 2:]

    print('reading video / masks', flush=True)
    nvid, frames, grey = read_video(f'{inp}/video.mp4', grey=True); assert nvid == N, (nvid, N)
    H5 = h5py.File(f'{inp}/masks.h5', 'r')
    tmask = table_mask(frames)
    obj_ds = np.stack([H5['object'][i][::4, ::4] for i in range(N)])
    per_ds = np.stack([H5['person'][i][::4, ::4] for i in range(N)])
    runs, stat_score = static_phases(obj_ds, per_ds, grey); del grey
    print('static runs', runs, flush=True)

    # ---- projected extent of the book (for crop sizes)
    corners = np.array(trimesh.bounds.corners(mesh.bounds))
    pc = np.einsum('bij,nj->bni', Rb, corners) + Tc_base[:, None, :3, 3]
    pu = np.einsum('ij,bnj->bni', K, pc); pu = pu[..., :2] / pu[..., 2:]
    ext = (pu.max(1) - pu.min(1)).max(1)

    # ---- per-frame crops of masks (object / valid) centred at the mask (or baseline) location
    boxes = np.zeros((N, 3)); OM = np.zeros((N, CROP, CROP), np.float32); VA = np.zeros_like(OM)
    vis_px = np.zeros(N)
    static_id = -np.ones(N, int)
    for k, (a, b) in enumerate(runs): static_id[a:b] = k
    # a static phase "rests on the table" if the object is fully visible there and lies on the
    # table-top region of the image -> table pixels are then valid background, otherwise occluder
    on_table = np.zeros(N, bool)
    for k, (a, b) in enumerate(runs):
        fr = range(a, b, 5)
        area = np.median([obj_ds[i].sum() * 16 for i in fr]); exp_area = np.median([ext[i] ** 2 * 0.45 for i in fr])
        # static and (almost) fully visible -> table pixels around it are plain background
        if area > 0.5 * exp_area: on_table[a:b] = True
    if args.cache and os.path.exists(args.cache):        # development only: reuse crops
        cz = np.load(args.cache); boxes, OM, VA, vis_px = cz['boxes'], cz['OM'], cz['VA'], cz['vis_px']
    else:
        for i in range(N):
            om = H5['object'][i]; pm = H5['person'][i]
            ys, xs = np.nonzero(om)
            if len(xs) >= MIN_VIS_PX:
                cx, cy = 0.5 * (xs.min() + xs.max()), 0.5 * (ys.min() + ys.max())
                # keep the crop large enough for the full (possibly occluded) object
                cx = 0.5 * (cx + uvb[i, 0]); cy = 0.5 * (cy + uvb[i, 1])
            else:
                cx, cy = uvb[i]
            side = 1.3 * ext[i] + 40
            x0, y0 = cx - side / 2, cy - side / 2
            boxes[i] = (x0, y0, side)
            s = CROP / side
            M = np.float32([[s, 0, -x0 * s], [0, s, -y0 * s]])
            o = cv2.warpAffine(om.astype(np.float32), M, (CROP, CROP), flags=cv2.INTER_AREA)
            p = cv2.warpAffine(pm.astype(np.float32), M, (CROP, CROP), flags=cv2.INTER_AREA)
            occ = p
            if not on_table[i]:
                tb = cv2.warpAffine(tmask.astype(np.float32), M, (CROP, CROP), flags=cv2.INTER_AREA)
                occ = np.maximum(occ, tb)
            # out-of-image pixels are unknown as well
            inside = cv2.warpAffine(np.ones(om.shape, np.float32), M, (CROP, CROP), flags=cv2.INTER_AREA)
            OM[i] = o; VA[i] = np.clip(1 - np.clip(occ - o, 0, 1), 0, 1) * (inside > 0.5)
            vis_px[i] = len(xs)
        if args.cache: np.savez(args.cache, boxes=boxes, OM=OM, VA=VA, vis_px=vis_px)
    wdata = (vis_px >= MIN_VIS_PX).astype(np.float32)
    # chamfer data: distance transform outside the object mask, sampled object-mask pixels
    DT = np.zeros_like(OM); MP = np.zeros((N, N_MASKPX, 2), np.float32); MPw = np.zeros((N, N_MASKPX), np.float32)
    rs = np.random.RandomState(0)
    for i in range(N):
        o = OM[i] > 0.5
        if o.sum() < 5: wdata[i] = 0; continue
        DT[i] = cv2.distanceTransform((~o).astype(np.uint8), cv2.DIST_L2, 5)
        ys, xs = np.nonzero(o); sel = rs.choice(len(xs), min(N_MASKPX, len(xs)), replace=False)
        MP[i, :len(sel)] = np.stack([xs[sel], ys[sel]], 1) + 0.5; MPw[i, :len(sel)] = 1
    print('crops done %.1fs' % (time.time() - t_start), flush=True)

    # ---- initial trajectory: per-frame parameters; static phases start from one robust-mean pose
    # robust smoothing of baseline depth for moving frames: running median (15) + gaussian (sigma 3)
    # static frames are replaced by their phase median first so that the smoothed depth is continuous at
    # the phase boundaries
    zseq = zb.copy()
    for a, b in runs: zseq[a:b] = np.median(zb[a:b])
    zs = gaussian_filter1d(median_filter(zseq, size=15, mode='nearest'), 3, mode='nearest')
    R_init = Rb.copy(); uv_init = uvb.copy(); zinit = zs.copy()
    for a, b in runs:
        R_init[a:b] = rot_mean(Rb[a:b]); uv_init[a:b] = np.median(uvb[a:b], 0); zinit[a:b] = np.median(zb[a:b])
    is_static = static_id >= 0
    if args.holdout > 0:      # validation mode: drop the silhouette term on every k-th frame
        wdata[np.arange(N) % args.holdout == args.holdout // 2] = 0
    # ---- optimisation
    sr = SilRenderer(mesh, K)
    surf, _ = trimesh.sample.sample_surface_even(mesh, N_SURF, seed=0)
    tt = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=DEV)
    R0 = tt(R_init); UV0 = tt(uv_init)
    w = torch.zeros(N, 3, device=DEV, requires_grad=True); duv = torch.zeros(N, 2, device=DEV, requires_grad=True)
    z_t = tt(zinit); ctr_t = tt(ctr)
    st_pair = tt((static_id[1:] == static_id[:-1]) & (static_id[1:] >= 0))   # consecutive frames in one static run
    mv_trip = 1 - tt(is_static[1:-1] & is_static[:-2] & is_static[2:])
    Rb_t = tt(Rb); uvb_t = tt(uvb); OM_t = tt(OM); VA_t = tt(VA); W_t = tt(wdata); box_t = tt(boxes)
    Kinv = tt(np.linalg.inv(K)); Kt = tt(K); surf_t = tt(surf)
    DT_t = tt(DT)[:, None]; MP_t = tt(MP); MPw_t = tt(MPw)
    opt = torch.optim.Adam([w, duv], lr=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.iters, eta_min=0.0005)
    CH = 128

    def poses():
        R = so3_exp(w * 0.2) @ R0                    # (n,3,3)
        uv = UV0 + duv * 50.0
        Rf = R; uvf = uv
        ray = torch.cat([uvf, torch.ones_like(uvf[:, :1])], 1) @ Kinv.T
        pcen = ray * z_t[:, None]                     # camera-frame bbox centre
        tf = pcen - (Rf @ ctr_t)
        return Rf, tf, uvf

    def sil_loss(Rf, tf, idx):
        """chamfer silhouette distance (crop px): projected surface samples falling on valid pixels must lie
        inside the object mask, and every visible object-mask pixel must be covered by a projected sample"""
        pc = torch.einsum('bij,pj->bpi', Rf[idx], surf_t) + tf[idx][:, None]
        q = pc @ Kt.T; q = q[..., :2] / q[..., 2:3]
        b = box_t[idx]; q = (q - b[:, None, :2]) * (CROP / b[:, 2])[:, None, None]
        g = (q / CROP * 2 - 1)[:, None]
        d = Fn.grid_sample(DT_t[idx], g, align_corners=False, padding_mode='border')[:, 0, 0]
        v = Fn.grid_sample(VA_t[idx][:, None], g, align_corners=False, padding_mode='zeros')[:, 0, 0]
        Lout = (d * v).sum(1) / (v.sum(1) + 1)
        cov = torch.cdist(MP_t[idx], q).min(2).values
        Lcov = (cov * MPw_t[idx]).sum(1) / (MPw_t[idx].sum(1) + 1e-6)
        L = Lout + Lcov
        return torch.log1p((L / DATA_C) ** 2) * W_t[idx]

    idx_all = torch.arange(N, device=DEV)
    for it in range(args.iters):
        opt.zero_grad()
        Rf, tf, uvf = poses()
        # priors / smoothness on the full sequence (cheap)
        dR = torch.linalg.norm(Rf - Rb_t, dim=(1, 2)) / (2 * np.sqrt(2))      # = sin(theta/2)
        th = 2 * torch.asin(dR.clamp(max=1.0))
        Lr = W_ROT_PRIOR * torch.log1p((th / ROT_C) ** 2).sum()
        du = torch.linalg.norm(uvf - uvb_t, dim=1)
        Lu = W_UV_PRIOR * torch.log1p((du / UV_C) ** 2).sum()
        accR = (((Rf[2:] - 2 * Rf[1:-1] + Rf[:-2]) ** 2).sum((1, 2)) * mv_trip).sum()
        accU = (((uvf[2:] - 2 * uvf[1:-1] + uvf[:-2]) ** 2).sum(1) * mv_trip).sum()
        velR = (((Rf[1:] - Rf[:-1]) ** 2).sum((1, 2)) * st_pair).sum()
        velU = (((uvf[1:] - uvf[:-1]) ** 2).sum(1) * st_pair).sum()
        Lreg = Lr + Lu + W_ACC_R * accR + W_ACC_UV * accU + W_VEL_R_STATIC * velR + W_VEL_UV_STATIC * velU
        Lreg.backward()
        Ls = 0.0
        for c in range(0, N, CH):
            Rf, tf, _ = poses()
            l = sil_loss(Rf, tf, idx_all[c:c + CH]).sum()
            l.backward(); Ls += float(l)
        opt.step(); sched.step()
        if it % 50 == 0 or it == args.iters - 1:
            if os.environ.get('DEBUG_FRAME'):
                fi = int(os.environ['DEBUG_FRAME'])
                with torch.no_grad():
                    Rf, tf, uvf = poses(); ls = sil_loss(Rf, tf, idx_all[fi:fi + 1])
                print('   dbg frame', fi, uvf[fi].cpu().numpy(), float(ls), 'grad uv', duv.grad[fi].cpu().numpy())
            print('it %d sil %.3f rot %.3f uv %.3f accR %.4f accU %.2f  (%.0fs)' % (
                it, Ls, float(Lr), float(Lu), float(accR), float(accU), time.time() - t_start), flush=True)

    with torch.no_grad():
        Rf, tf, uvf = poses()
        Rf = Rf.double().cpu().numpy(); tf = tf.double().cpu().numpy()
        # per-frame silhouette IoU (valid region) for diagnostics: ours vs baseline
        def ious(R, t):
            out = np.zeros(N)
            for c in range(0, N, CH):
                ix = idx_all[c:c + CH]
                m = sr.render(tt(R[c:c + CH]), tt(t[c:c + CH]), box_t[ix])
                m = (m > 0.5).float(); om = (OM_t[ix] > 0.5).float(); va = VA_t[ix]
                inter = (m * om * va).sum((1, 2)); uni = ((m + om - m * om) * va).sum((1, 2))
                out[c:c + CH] = (inter / uni.clamp_min(1)).cpu().numpy()
            return out
        iou_new = ious(Rf, tf); iou_base = ious(Rb, Tc_base[:, :3, 3])

    T_cam = np.tile(np.eye(4), (N, 1, 1)); T_cam[:, :3, :3] = Rf; T_cam[:, :3, 3] = tf

    # ---- texture fine alignment (lateral image shift at constant depth); the silhouette of the SAM2 mask is
    # biased by ~1-3 px (shadows, blur), the texture of the book is not
    shifts = np.zeros((N, 2)); tex_info = []
    if not args.no_texture:
        maps = texture_scores(T_cam, K, mesh, f'{inp}/video.mp4', H5)
        have = ~np.isnan(maps[:, 0, 0])
        mapz = np.where(have[:, None, None], maps, 0.0)
        for a, b, kind in phase_list(runs, N):
            if kind == 'static':
                if have[a:b].sum() < 3: continue
                pm = mapz[a:b][have[a:b]].mean(0)
                shifts[a:b] = peak2d(pm); tex_info.append((a, b, shifts[a].tolist(), int(have[a:b].sum())))
        is_st = np.zeros(N, bool)
        for a, b in runs: is_st[a:b] = True
        known = is_st.copy()
        for i in range(N):
            if is_st[i]: continue
            # temporally smoothed score maps (normalised Gaussian weights over frames that have a map)
            j0, j1 = max(0, i - int(3 * TEX_SMOOTH)), min(N, i + int(3 * TEX_SMOOTH) + 1)
            w = np.exp(-0.5 * ((np.arange(j0, j1) - i) / TEX_SMOOTH) ** 2) * have[j0:j1]
            if w.sum() < 1.0: continue
            shifts[i] = peak2d((w[:, None, None] * mapz[j0:j1]).sum(0) / w.sum()); known[i] = True
        # frames without texture evidence (object hidden): interpolate the shift from the neighbours
        for c in range(2): shifts[~known, c] = np.interp(np.nonzero(~known)[0], np.nonzero(known)[0], shifts[known, c])
        sm = gaussian_filter1d(shifts, 2.0, axis=0, mode='nearest')
        shifts[~is_st] = sm[~is_st]
        T_cam[:, 0, 3] += shifts[:, 1] * T_cam[:, 2, 3] / K[0, 0]
        T_cam[:, 1, 3] += shifts[:, 0] * T_cam[:, 2, 3] / K[1, 1]
        print('texture shifts (static phases):', tex_info, flush=True)

    # ---- depth re-anchoring per phase (static runs and the moving segments between them): scale the camera-frame
    # positions along the viewing rays so that the phase's median z equals the baseline median z. Static phases:
    # one factor. Moving phases: factor 1 + A * taper(t) (taper ramps from 0 at a boundary with a static phase to
    # 1 within RAMP frames) so that no jump is created at the boundaries; A solved so the median matches exactly.
    RAMP = 10
    bounds = phase_list(runs, N)
    for a, b, kind in bounds:
        pc_new = (T_cam[a:b, :3, :3] @ ctr) + T_cam[a:b, :3, 3]
        target = np.median(zb[a:b])
        if kind == 'static' or b - a < 3 * RAMP:
            fac = np.full(b - a, target / np.median(pc_new[:, 2]))
        else:
            k = np.arange(b - a, dtype=float)
            tap = np.ones(b - a)
            if a > 0: tap = np.minimum(tap, (k + 1) / RAMP)
            if b < N: tap = np.minimum(tap, (b - a - k) / RAMP)
            tap = np.clip(tap, 0, 1)
            lo, hi = -0.5, 0.5
            for _ in range(60):                      # bisection on A (median is monotone in A)
                A = 0.5 * (lo + hi)
                if np.median(pc_new[:, 2] * (1 + A * tap)) < target: lo = A
                else: hi = A
            fac = 1 + A * tap
        T_cam[a:b, :3, 3] += (fac[:, None] - 1) * pc_new
    T_world = Twc[None] @ T_cam
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, T_world_mesh=T_world.astype(np.float64))

    # ---- diagnostics
    pc_new = (T_cam[:, :3, :3] @ ctr) + T_cam[:, :3, 3]
    diag = dict(static_runs=runs, phases=[(int(a), int(b), k) for a, b, k in bounds],
                on_table=[int(x) for x in np.nonzero(np.diff(np.r_[0, on_table.astype(int), 0]))[0]],
                runtime_s=time.time() - t_start, per_phase=[])
    for a, b, kind in bounds:
        vis = wdata[a:b] > 0
        dz = (pc_new[a:b, 2] - zb[a:b]) * 100
        diag['per_phase'].append(dict(
            a=int(a), b=int(b), kind=kind,
            iou_base=float(np.mean(iou_base[a:b][vis])) if vis.any() else None,
            iou_new=float(np.mean(iou_new[a:b][vis])) if vis.any() else None,
            dz_mean_cm=float(dz.mean()), dz_median_cm=float(np.median(dz)),
            rot_change_deg_median=float(np.median(rotdiff_deg(T_cam[a:b, :3, :3], Rb[a:b]))),
            rot_change_deg_max=float(np.max(rotdiff_deg(T_cam[a:b, :3, :3], Rb[a:b]))),
            uv_shift_px_median=float(np.median(np.linalg.norm(
                (lambda p: p[:, :2] / p[:, 2:])((K @ pc_new[a:b].T).T) - uvb[a:b], axis=1)))))
    np.savez(os.path.splitext(args.out)[0] + '_diag.npz', tex_shifts=shifts, iou_new=iou_new, iou_base=iou_base, wdata=wdata,
             stat_score=stat_score, on_table=on_table)
    json.dump(diag, open(os.path.splitext(args.out)[0] + '_diag.json', 'w'), indent=1)
    print(json.dumps(diag['per_phase'], indent=1))
    print('done in %.1fs' % (time.time() - t_start))


if __name__ == '__main__':
    main()
