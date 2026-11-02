"""Shared helpers: data loading, nvdiffrast rendering of the textured mesh."""
import json, numpy as np, torch, trimesh, cv2, h5py
import nvdiffrast.torch as dr

W, H = 1536, 1152

def load_mesh(path):
    m = trimesh.load(path, force="scene", process=False).to_geometry()
    v = np.asarray(m.vertices, np.float32); f = np.asarray(m.faces, np.int32)
    uv = np.asarray(m.visual.uv, np.float32)
    mat = m.visual.material
    tex = getattr(mat, 'baseColorTexture', None) or getattr(mat, 'image', None)
    tex = np.asarray(tex.convert('RGB'), np.float32) / 255.0
    return m, v, f, uv, tex

def load_inputs(inp):
    cam = json.load(open(f'{inp}/camera.json'))
    K = np.array(cam['K'], np.float64); Twc = np.array(cam['T_world_cam'], np.float64)
    rec = np.load(f'{inp}/reconstruction.npz')
    return cam, K, Twc, rec

def read_video(path, every=1):
    cap = cv2.VideoCapture(path); fr = []
    while True:
        ok, im = cap.read()
        if not ok: break
        fr.append(im)
    return fr

def proj_from_K(K, w, h, n=0.05, f=10.0):
    # OpenCV camera -> clip; nvdiffrast clip y up, we flip so row 0 = image top
    P = np.zeros((4, 4), np.float32)
    P[0, 0] = 2 * K[0, 0] / w; P[0, 2] = 2 * K[0, 2] / w - 1
    P[1, 1] = 2 * K[1, 1] / h; P[1, 2] = 2 * K[1, 2] / h - 1
    P[2, 2] = (f + n) / (f - n); P[2, 3] = -2 * f * n / (f - n)
    P[3, 2] = 1
    return P

class Renderer:
    def __init__(self, v, f, uv, tex, device='cuda'):
        self.glctx = dr.RasterizeCudaContext()
        self.v = torch.tensor(v, device=device); self.f = torch.tensor(f, device=device)
        self.uv = torch.tensor(uv, device=device)
        self.uv[:, 1] = 1 - self.uv[:, 1]
        self.tex = torch.tensor(tex, device=device)[None].contiguous()
        self.dev = device

    def render(self, T_cam_mesh, K, w, h, crop=None):
        """T_cam_mesh (B,4,4) torch or np. crop=(x0,y0) window offset; w,h window size.
        returns rgb (B,h,w,3), mask (B,h,w), depth (B,h,w) (z in camera, 0 bg)"""
        T = torch.as_tensor(np.asarray(T_cam_mesh) if not torch.is_tensor(T_cam_mesh) else T_cam_mesh,
                            dtype=torch.float32, device=self.dev)
        Kc = np.array(K, np.float64).copy()
        if crop is not None:
            Kc = np.repeat(Kc[None], T.shape[0], 0) if Kc.ndim == 2 else Kc.copy()
            Kc[:, 0, 2] -= crop[:, 0]; Kc[:, 1, 2] -= crop[:, 1]
        if Kc.ndim == 2: Kc = np.repeat(Kc[None], T.shape[0], 0)
        P = torch.tensor(np.stack([proj_from_K(k, w, h) for k in Kc]), device=self.dev)
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        vc = torch.einsum('bij,nj->bni', T, vh)
        clip = torch.einsum('bij,bnj->bni', P, vc)
        rast, _ = dr.rasterize(self.glctx, clip.contiguous(), self.f, (h, w))
        texc, _ = dr.interpolate(self.uv[None], rast, self.f)
        rgb = dr.texture(self.tex, texc, filter_mode='linear')
        mask = rast[..., 3] > 0
        z, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, self.f)
        return rgb, mask, z[..., 0] * mask

def body_axis(v):
    """Fit the mug body's symmetry axis (mesh coords): returns point c0 and unit dir u (towards lid, -z)."""
    v = v.astype(np.float64); cs = []
    for zq in np.linspace(-0.06, 0.105, 18):
        s = v[np.abs(v[:, 2] - zq) < 0.004]; s = s[s[:, 0] < 0.02]   # drop handle side (+x)
        A = np.c_[2 * s[:, 0], 2 * s[:, 1], np.ones(len(s))]; b = (s[:, :2] ** 2).sum(1)
        c = np.linalg.lstsq(A, b, rcond=None)[0]; cs.append([c[0], c[1], zq])
    P = np.array(cs); c0 = P.mean(0); u = np.linalg.svd(P - c0)[2][0]
    if u[2] > 0: u = -u
    return c0, u

def rot_about(u, ang):
    """Rodrigues rotation matrices (N,3,3) about unit axis u by angles ang (N,) radians."""
    ang = np.atleast_1d(ang); Kx = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3)[None] + np.sin(ang)[:, None, None] * Kx[None] + (1 - np.cos(ang))[:, None, None] * (Kx @ Kx)[None]

def spin(T, c0, u, ang):
    """T (4,4) mesh->X. Returns (N,4,4) poses T @ [rotation about the mesh body axis by ang]."""
    Rs = rot_about(u, ang); out = np.repeat(T[None], len(Rs), 0).astype(np.float64)
    for k, Rk in enumerate(Rs):
        M = np.eye(4); M[:3, :3] = Rk; M[:3, 3] = c0 - Rk @ c0
        out[k] = T @ M
    return out

class Crops:
    """Per-frame fixed-size crops around the SAM2 object mask (image, object mask, person mask)."""
    def __init__(self, inp, size=160, frames=None):
        fr = read_video(f'{inp}/video.mp4'); F = len(fr)
        self.size = size; s = size // 2
        idx = range(F) if frames is None else frames
        f = h5py.File(f'{inp}/masks.h5', 'r')
        self.img, self.om, self.pm, self.off = [], [], [], []
        for i in idx:
            om = f['object'][i]; pm = f['person'][i]
            ys, xs = np.nonzero(om)
            cx, cy = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2)) if len(xs) else (W // 2, H // 2)
            x0 = int(np.clip(cx - s, 0, W - size)); y0 = int(np.clip(cy - s, 0, H - size))
            sl = (slice(y0, y0 + size), slice(x0, x0 + size))
            self.img.append(fr[i][sl][..., ::-1].astype(np.float32) / 255.0)
            self.om.append(om[sl]); self.pm.append(pm[sl]); self.off.append([x0, y0])
        self.img = np.stack(self.img); self.om = np.stack(self.om); self.pm = np.stack(self.pm)
        self.off = np.array(self.off, np.float64)

def axis_angle_to_R(w):
    th = torch.linalg.norm(w, dim=-1, keepdim=True).clamp_min(1e-9)
    k = w / th; Kx = torch.zeros(w.shape[:-1] + (3, 3), device=w.device, dtype=w.dtype)
    Kx[..., 0, 1] = -k[..., 2]; Kx[..., 0, 2] = k[..., 1]; Kx[..., 1, 0] = k[..., 2]
    Kx[..., 1, 2] = -k[..., 0]; Kx[..., 2, 0] = -k[..., 1]; Kx[..., 2, 1] = k[..., 0]
    s = torch.sin(th)[..., None]; c = torch.cos(th)[..., None]
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(Kx)
    return I + s * Kx + (1 - c) * (Kx @ Kx)

def gray(x):
    return x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114

def blur(g, k=3):
    return torch.nn.functional.avg_pool2d(g[:, None], k, 1, k // 2, count_include_pad=False)[:, 0]

def local_norm(g, k=7):
    """high-pass + local contrast normalisation of (B,h,w) gray images"""
    x = g[:, None]; pad = k // 2
    mu = torch.nn.functional.avg_pool2d(x, k, 1, pad, count_include_pad=False)
    var = torch.nn.functional.avg_pool2d(x * x, k, 1, pad, count_include_pad=False) - mu * mu
    return ((x - mu) / (var.clamp_min(0).sqrt() + 0.02))[:, 0]

class Fitter:
    """Render-and-compare refinement of pose hypotheses on frame crops.
    Each hypothesis h has its own 6-DoF delta (rotation about the body centre, camera-frame translation)
    and is evaluated on a list of frames (all frames of a static phase share one pose)."""
    def __init__(self, R, K, crops, c0, up=2, dev='cuda'):
        self.R = R; self.K = K; self.up = up; self.dev = dev; self.c0 = c0
        self.S = crops['img'].shape[1]; self.crops = crops

    def batch(self, fr):
        up = self.up; S = self.S
        img = torch.tensor(self.crops['img'][fr], device=self.dev).float() / 255.0
        om = torch.tensor(self.crops['om'][fr], device=self.dev).float()
        pm = torch.tensor(self.crops['pm'][fr], device=self.dev).float()
        img = torch.nn.functional.interpolate(img.permute(0, 3, 1, 2), scale_factor=up, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        om = torch.nn.functional.interpolate(om[:, None], scale_factor=up, mode='nearest')[:, 0]
        pm = torch.nn.functional.interpolate(pm[:, None], scale_factor=up, mode='nearest')[:, 0]
        valid = 1 - pm * (1 - om)
        return img, om, valid, local_norm(gray(img))

    def poses(self, T0, w, dt):
        """T0 (N,4,4) torch, w,dt (N,3) -> (N,4,4)"""
        Rw = axis_angle_to_R(w); R0 = T0[:, :3, :3]; t0 = T0[:, :3, 3]
        c = torch.tensor(self.c0, device=self.dev, dtype=T0.dtype)
        Rn = Rw @ R0; ccam = R0 @ c + t0
        tn = ccam + dt - Rn @ c
        T = torch.zeros_like(T0); T[:, :3, :3] = Rn; T[:, :3, 3] = tn; T[:, 3, 3] = 1
        return T

    def render_pairs(self, T, fr):
        up = self.up; S = self.S * up
        Ku = self.K.copy(); Ku[:2] *= up
        off = self.crops['off'][fr] * up
        P = torch.tensor(np.stack([proj_from_K(np.array([[Ku[0, 0], 0, Ku[0, 2] - o[0]], [0, Ku[1, 1], Ku[1, 2] - o[1]], [0, 0, 1]]), S, S) for o in off]), device=self.dev, dtype=torch.float32)
        R = self.R
        vh = torch.cat([R.v, torch.ones_like(R.v[:, :1])], 1)
        vc = torch.einsum('bij,nj->bni', T.float(), vh)
        clip = torch.einsum('bij,bnj->bni', P, vc).contiguous()
        rast, _ = dr.rasterize(R.glctx, clip, R.f, (S, S))
        texc, _ = dr.interpolate(R.uv[None], rast, R.f)
        rgb = dr.texture(R.tex, texc, filter_mode='linear')
        m = (rast[..., 3:4] > 0).float()
        sil = dr.antialias(m, rast, clip, R.f)[..., 0]
        return rgb, m[..., 0], sil

    def losses(self, T, fr, data):
        img, om, valid, ln = data
        rgb, m, sil = self.render_pairs(T, fr)
        inter = (sil * om * valid).sum((1, 2)); uni = ((sil + om - sil * om) * valid).sum((1, 2))
        iou = inter / uni.clamp_min(1)
        # appearance: NCC of (lightly blurred) gray over the rendered, unoccluded region (not restricted to the
        # SAM mask, which often misses the dark handle against the dark shirt)
        reg = m * valid
        lr = blur(gray(rgb)); lb = blur(gray(img))
        n = reg.sum((1, 2)).clamp_min(1)
        am = (lr * reg).sum((1, 2)) / n; bm = (lb * reg).sum((1, 2)) / n
        a = (lr - am[:, None, None]) * reg; b = (lb - bm[:, None, None]) * reg
        ncc = (a * b).sum((1, 2)) / ((a * a).sum((1, 2)) * (b * b).sum((1, 2))).clamp_min(1e-10).sqrt()
        return iou, ncc

    def fit(self, T0, hyp_frames, iters=60, lr_r=0.02, lr_t=0.003, w_ncc=0.5, z_prior=None, w_z=0.0, chunk=256, free_z=False, free_rot=True):
        """T0 (Hn,4,4) np initial poses, hyp_frames: list of frame-index lists (one per hypothesis).
        Frame-dependent T0 allowed via T0 of shape (P,4,4) matched to pairs -> not used here."""
        Hn = len(T0)
        pairs_h = np.concatenate([[h] * len(fl) for h, fl in enumerate(hyp_frames)]).astype(int)
        pairs_f = np.concatenate([list(fl) for fl in hyp_frames]).astype(int)
        T0t = torch.tensor(T0, device=self.dev, dtype=torch.float32)
        w = torch.zeros(Hn, 3, device=self.dev, requires_grad=True)
        dt = torch.zeros(Hn, 3, device=self.dev, requires_grad=True)
        opt = torch.optim.Adam([{'params': [w], 'lr': lr_r}, {'params': [dt], 'lr': lr_t}])
        cnt = torch.tensor(np.bincount(pairs_h, minlength=Hn), device=self.dev).float()
        for it in range(iters + 1):
            tot_iou = torch.zeros(Hn, device=self.dev); tot_ncc = torch.zeros(Hn, device=self.dev)
            if it < iters: opt.zero_grad()
            for s in range(0, len(pairs_h), chunk):
                ph = pairs_h[s:s + chunk]; pf = pairs_f[s:s + chunk]
                data = self.batch(pf)
                zm = torch.tensor([1., 1., 1. if free_z else 0.], device=self.dev)
                T = self.poses(T0t[ph], w[ph] * (1. if free_rot else 0.), dt[ph] * zm)
                iou, ncc = self.losses(T, pf, data)
                phi = torch.tensor(ph, device=self.dev)
                loss_pair = -(iou + w_ncc * ncc) / cnt[phi]
                if z_prior is not None:
                    loss_pair = loss_pair + w_z * (dt[ph, 2] ** 2) / cnt[phi]
                if it < iters: loss_pair.sum().backward()
                tot_iou.index_add_(0, phi, iou.detach() / cnt[phi]); tot_ncc.index_add_(0, phi, ncc.detach() / cnt[phi])
            if it < iters: opt.step()
        with torch.no_grad():
            zm = torch.tensor([1., 1., 1. if free_z else 0.], device=self.dev)
            Tf = self.poses(T0t, w * (1. if free_rot else 0.), dt * zm).double().cpu().numpy()
        return Tf, tot_iou.cpu().numpy(), tot_ncc.cpu().numpy()

def logo_texture(v, f, uv, tex, c0, u):
    """Label texture (H,W) float: 1 on the white 'contigo' logo texels of the lower body, 0 elsewhere.
    The logo is found geometrically: white texels on faces at axial height h in [-0.075,-0.052] m below the
    body centre and azimuth 75..140 deg (mesh frame, 0 = +x side, where the handle is ~19 deg); outside this
    window the white texels on the lower body are long vertical specular streaks baked into the scan."""
    e1 = np.array([1., 0, 0]); e1 = e1 - u * (e1 @ u); e1 /= np.linalg.norm(e1); e2 = np.cross(u, e1)
    fc = v[f].mean(1).astype(np.float64); d = fc - c0; h = d @ u
    ph = np.degrees(np.arctan2(d @ e2, d @ e1))
    sel = (h > -0.075) & (h < -0.052) & (ph > 75) & (ph < 140)
    Ht, Wt = tex.shape[:2]
    m = np.zeros((Ht, Wt), np.uint8)
    for tri in f[sel]:
        p = np.stack([uv[tri, 0] * (Wt - 1), (1 - uv[tri, 1]) * (Ht - 1)], 1).round().astype(np.int32)
        cv2.fillConvexPoly(m, p, 1)
    hsv = cv2.cvtColor((tex * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32) / 255
    white = (hsv[..., 2] > 0.55) & (hsv[..., 1] < 0.35)
    return (m > 0) & white

def bright_detail(gray_img):
    """bright, vertically short structures (text), suppressing long vertical specular streaks: gray - vertical opening"""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    op = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, k)
    return np.clip(gray_img - op, 0, None)

def mask_tracks(inp):
    """per frame: bottom row of the object mask, lower-body centre column (bottom 25 rows; the handle is
    higher up), and bbox"""
    f = h5py.File(f'{inp}/masks.h5', 'r'); out = []
    for i in range(f['object'].shape[0]):
        m = f['object'][i]; ys, xs = np.nonzero(m)
        if len(ys) == 0: out.append([np.nan] * 4); continue
        yb = ys.max(); low = m[yb - 25:yb + 1]; yy, xx = np.nonzero(low)
        out.append([yb, xx.mean(), (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2])
    return np.array(out, np.float64)

def static_phases(tr, min_len=15, tol_y=1.5, tol_x=0.8, win=3):
    """Static (object resting, constant pose) segments: mask bottom at the table level and lower-body
    centre not moving. Returns list of (start, end_exclusive)."""
    F = len(tr); yb = tr[:, 0]; xc = tr[:, 1]
    vals, cnt = np.unique(yb[~np.isnan(yb)], return_counts=True)
    # table level: most common bottom row (the object rests on the table most of the time); allow
    # small differences between placements by testing proximity to the per-frame running median
    st = np.zeros(F, bool)
    for t in range(F):
        a, b = max(0, t - win), min(F, t + win + 1)
        st[t] = (np.nanmax(yb[a:b]) - np.nanmin(yb[a:b]) <= tol_y) and (np.nanmax(xc[a:b]) - np.nanmin(xc[a:b]) <= tol_x) \
            and abs(yb[t] - vals[np.argmax(cnt)]) <= 4
    segs = []; t = 0
    while t < F:
        if st[t]:
            s = t
            while t < F and st[t]: t += 1
            if t - s >= min_len: segs.append((s, t))
        else: t += 1
    # merge segments separated by short gaps (brief mask jitter) if the object did not move between them
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] <= 6 and abs(np.nanmedian(xc[s[0]:s[1]]) - np.nanmedian(xc[merged[-1][0]:merged[-1][1]])) < 1.0:
            merged[-1] = (merged[-1][0], s[1])
        else: merged.append(s)
    return merged

def fit_track(Fi, T0, frames, iters=80, lr_r=0.01, lr_t=0.002, w_ncc=0.3, w_smooth=50.0, w_prior=1.0,
              w_prior_t=0.0, fix_ends=True, chunk=128, data_mask=None):
    """Joint refinement of a pose sequence (one pose per frame) on its frames: maximise silhouette IoU and
    appearance NCC; penalise second differences of rotation/translation deltas (smooth motion), deviation of
    the rotation delta from the initial (prior) poses; depth along z kept fixed (re-anchored later).
    fix_ends: first/last poses keep their initial values (boundary conditions from static phases)."""
    N = len(T0); dev = Fi.dev
    T0t = torch.tensor(T0, device=dev, dtype=torch.float32)
    w = torch.zeros(N, 3, device=dev, requires_grad=True); dt = torch.zeros(N, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{'params': [w], 'lr': lr_r}, {'params': [dt], 'lr': lr_t}])
    zm = torch.tensor([1., 1., 0.], device=dev)
    em = torch.ones(N, 1, device=dev)
    if fix_ends: em[0] = 0; em[-1] = 0
    frames = np.asarray(frames)
    hist = []
    for it in range(iters + 1):
        if it < iters: opt.zero_grad()
        tot_i = torch.zeros(N, device=dev); tot_n = torch.zeros(N, device=dev)
        for s in range(0, N, chunk):
            idx = np.arange(s, min(N, s + chunk))
            data = Fi.batch(frames[idx])
            T = Fi.poses(T0t[idx], (w * em)[idx], (dt * em * zm)[idx])
            iou, ncc = Fi.losses(T, frames[idx], data)
            dm = torch.ones(len(idx), device=dev) if data_mask is None else torch.tensor(data_mask[idx], device=dev).float()
            loss = -((iou + w_ncc * ncc) * dm).sum()
            if it < iters: loss.backward()
            tot_i[idx] = iou.detach(); tot_n[idx] = ncc.detach()
        if it < iters:
            we = w * em; de = dt * em * zm
            reg = w_prior * (we ** 2).sum() * 100 + w_prior_t * (de ** 2).sum() * 1e4
            if N >= 3:
                reg = reg + w_smooth * (((we[2:] - 2 * we[1:-1] + we[:-2]) ** 2).sum() * 100
                                        + ((de[2:] - 2 * de[1:-1] + de[:-2]) ** 2).sum() * 1e4)
            reg.backward(); opt.step()
        hist.append(float(tot_i.mean()))
    with torch.no_grad():
        Tf = Fi.poses(T0t, w * em, dt * em * zm).double().cpu().numpy()
    return Tf, tot_i.cpu().numpy(), tot_n.cpu().numpy(), hist

def geodesic_deg(Ra, Rb):
    c = (np.einsum('...ij,...ij->...', Ra, Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))

def yaw_match(T, Rref, c0, u, grid=np.radians(np.arange(0, 360, 2))):
    """spin T about the mesh body axis to the angle whose rotation is closest to Rref"""
    S = spin(T, c0, u, grid)
    d = geodesic_deg(S[:, :3, :3], Rref[None])
    return S[np.argmin(d)], d.min()

def iou_scores(R, K, crops, T, frames, maxs=3, chunk=96):
    """IoU of rendered silhouette with object mask over unoccluded pixels, best over small 2-D shifts"""
    out = np.zeros(len(T)); S = crops['om'].shape[1]
    for s in range(0, len(T), chunk):
        sl = slice(s, s + chunk); fr = np.asarray(frames)[sl]
        _, mk, _ = R.render(T[sl], K, S, S, crop=crops['off'][fr])
        om = torch.tensor(crops['om'][fr], device='cuda').float(); pm = torch.tensor(crops['pm'][fr], device='cuda').float()
        valid = 1 - pm * (1 - om); mk = mk.float(); best = torch.zeros(len(fr), device='cuda')
        for dy in range(-maxs, maxs + 1):
            for dx in range(-maxs, maxs + 1):
                r = torch.roll(mk, (dy, dx), (1, 2))
                best = torch.maximum(best, (r * om * valid).sum((1, 2)) / ((r + om - r * om) * valid).sum((1, 2)).clamp_min(1))
        out[sl] = best.cpu().numpy()
    return out

def viterbi(data_cost, trans_cost_fn, start_cost=None, end_cost=None):
    """data_cost (T,C); trans_cost_fn(t) -> (C,C) cost from candidates at t-1 to t"""
    Tn, C = data_cost.shape
    acc = data_cost[0] + (0 if start_cost is None else start_cost); bp = np.zeros((Tn, C), int)
    for t in range(1, Tn):
        tot = acc[:, None] + trans_cost_fn(t)
        bp[t] = np.argmin(tot, 0); acc = tot[bp[t], np.arange(C)] + data_cost[t]
    if end_cost is not None: acc = acc + end_cost
    path = np.zeros(Tn, int); path[-1] = np.argmin(acc)
    for t in range(Tn - 1, 0, -1): path[t - 1] = bp[t, path[t]]
    return path

def handle_az(T):
    """azimuth (deg) of the mesh +x (handle side) direction in the camera x-z plane: atan2(z, x)"""
    hx = T[..., :3, 0]
    return np.degrees(np.arctan2(hx[..., 2], hx[..., 0]))

def measure_logo_az(RL, K, T_align, c0, u, img_rgb, om, off, sc=4, step=2):
    """Yaw of a resting mug from the position of the white 'contigo' logo across the lower body.
    T_align: camera pose giving the silhouette alignment (any yaw). The mug is spun about its axis on a grid;
    for each yaw the rendered logo's centre is expressed as a fraction of the body width on its row, and
    compared to the fraction measured in the image (bright, vertically short detail = text, inside the
    observed object mask). Returns (az_deg, observed_fraction, residual) or None if no logo is visible."""
    az0 = handle_az(T_align); targets = np.arange(0, 360, step)
    Ts = spin(T_align, c0, u, np.radians(targets - az0))
    K4 = K.copy(); K4[:2] *= sc; S = img_rgb.shape[0] * sc
    o = np.repeat((off * sc)[None], len(Ts), 0)
    lab, mk, _ = RL.render(Ts, K4, S, S, crop=o)
    lab = lab[..., 0].cpu().numpy() > 0.5; mk = mk.cpu().numpy()
    rfr = np.full(len(Ts), np.nan); rows = []
    for q in range(len(Ts)):
        ys, xs = np.nonzero(lab[q])
        if len(xs) < 30: continue
        row = int(np.median(ys)); cols = np.nonzero(mk[q][row])[0]
        if len(cols) < 2: continue
        rfr[q] = (xs.mean() - cols.min()) / (cols.max() - cols.min()); rows.append(ys)
    if not rows: return None
    ys_all = np.concatenate(rows); r0, r1 = np.percentile(ys_all, [5, 95]).astype(int)
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    bd = cv2.resize(bright_detail(g), None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
    omu = cv2.resize(om.astype(np.uint8), None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
    prof = (bd[r0:r1 + 1] * omu[r0:r1 + 1]).sum(0)
    cols = np.nonzero(omu[(r0 + r1) // 2])[0]
    if len(cols) < 10: return None
    lft, rgt = cols.min(), cols.max()
    inner = np.zeros_like(prof); inner[lft + 6:rgt - 5] = prof[lft + 6:rgt - 5]
    pk = int(np.argmax(inner))
    if inner[pk] <= 0: return None
    # contrast check: logo blob must stand out from the rest of the band
    if inner[pk] < 3 * np.median(inner[lft + 6:rgt - 5]) + 1e-6: return None
    hm = inner > 0.5 * inner[pk]; a = pk; b = pk
    while a > 0 and hm[a - 1]: a -= 1
    while b < len(hm) - 1 and hm[b + 1]: b += 1
    cx = (np.arange(a, b + 1) * inner[a:b + 1]).sum() / inner[a:b + 1].sum()
    ofr = (cx - lft) / (rgt - lft)
    d = np.abs(rfr - ofr)
    if np.all(np.isnan(d)): return None
    q = int(np.nanargmin(d))
    return float(targets[q]), float(ofr), float(d[q])

def table_cue_z(depth_png, om, pm, K_moge, K, p_bottom, n_up, rng, iters=200):
    """Depth cue from the support surface: RANSAC plane on MoGe-2 (metric-aligned) depth of the table-top
    pixels around/below the resting object (excluding object and person), accepted only if the plane
    normal is within 12 deg of the floor normal; returns z (camera) of the intersection of the viewing ray
    of the object's bottom centre with that plane, or None."""
    d = cv2.imdecode(depth_png, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    ys, xs = np.nonzero(om)
    if len(ys) == 0: return None
    yb = ys.max(); xc = int(xs.mean())
    Y, X = np.mgrid[yb + 3:min(yb + 60, d.shape[0]), max(xc - 100, 0):min(xc + 100, d.shape[1])]
    ok = ~pm[Y, X] & ~om[Y, X] & (d[Y, X] > 0)
    if ok.sum() < 300: return None
    z = d[Y, X][ok]; P = np.stack([(X[ok] - K_moge[0, 2]) / K_moge[0, 0] * z, (Y[ok] - K_moge[1, 2]) / K_moge[1, 1] * z, z], 1)
    best = None
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0]); nn = np.linalg.norm(n)
        if nn < 1e-9: continue
        n /= nn; dd = -n @ s[0]; inl = np.abs(P @ n + dd) < 0.01
        if best is None or inl.sum() > best[0]: best = (inl.sum(), inl)
    inl = best[1]; Q = P[inl]; c = Q.mean(0); n = np.linalg.svd(Q - c)[2][2]
    if abs(n @ n_up) < np.cos(np.radians(12)) or inl.mean() < 0.5: return None
    pix = K @ p_bottom; pix = pix[:2] / pix[2]
    ray = np.array([(pix[0] - K_moge[0, 2]) / K_moge[0, 0], (pix[1] - K_moge[1, 2]) / K_moge[1, 1], 1.0])
    den = n @ ray
    if abs(den) < 1e-6: return None
    return float((n @ c) / den)

def hand_cue_shift(T_seq, frames, kp_cam, c0, hv, shifts=np.arange(-0.3, 0.3001, 0.005)):
    """Depth cue from hand contact: shift (m, camera z of the body centre, along the viewing rays) that minimises
    the mean distance between the handle centre and the centroid of the nearest reconstructed hand over the
    given contact frames. returns (shift, residual_m)"""
    cen = np.einsum('nij,j->ni', T_seq[:, :3, :3], c0) + T_seq[:, :3, 3]
    hc = np.einsum('nij,j->ni', T_seq[:, :3, :3], hv) + T_seq[:, :3, 3]
    hands = np.stack([kp_cam[frames, 21:42].mean(1), kp_cam[frames, 42:63].mean(1)], 1)
    hsel = hands[np.arange(len(frames)), np.argmin(np.linalg.norm(hands - cen[:, None], axis=2), 1)]
    best = None
    for s in shifts:
        sc = (cen[:, 2] + s) / cen[:, 2]
        e = np.linalg.norm(hc + cen * (sc - 1)[:, None] - hsel, axis=1).mean()
        if best is None or e < best[1]: best = (float(s), float(e))
    return best
