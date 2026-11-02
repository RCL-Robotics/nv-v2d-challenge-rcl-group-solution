"""Shared helpers: data loading, nvdiffrast textured rendering in image crops."""
import json, numpy as np, torch, trimesh, h5py, cv2
import nvdiffrast.torch as dr

DEV = 'cuda'
W_CHAM = 1.0
PERSON_NEG = 0.5


def load_inputs(inp):
    cam = json.load(open(f'{inp}/camera.json'))
    K = np.array(cam['K'], np.float64)
    Twc = np.array(cam['T_world_cam'], np.float64)
    rec = np.load(f'{inp}/reconstruction.npz')
    return K, Twc, rec


def load_video(path, scale=1.0):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if scale != 1.0:
            f = cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        frames.append(f[:, :, ::-1].copy())  # RGB
    return frames


class MeshRenderer:
    def __init__(self, glb):
        m = trimesh.load(glb, force='scene', process=False).to_geometry()
        self.v = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=DEV)
        self.f = torch.tensor(np.asarray(m.faces), dtype=torch.int32, device=DEV)
        uv = np.asarray(m.visual.uv, np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        self.uv = torch.tensor(uv, device=DEV)
        tex = np.asarray(m.visual.material.baseColorTexture.convert('RGB'), np.float32) / 255.0
        self.tex = torch.tensor(tex, device=DEV)[None].contiguous()
        self.verts_np = np.asarray(m.vertices)
        fn = np.asarray(m.face_normals, np.float32)
        vn = np.zeros((len(m.vertices), 3), np.float32)
        vn[np.asarray(m.faces).reshape(-1)] = np.repeat(fn, 3, 0)
        self.vn = torch.tensor(vn, device=DEV)
        self.ctx = dr.RasterizeCudaContext()

    def render(self, T_cam, K, crop, size, with_tex=True):
        """T_cam (B,4,4) torch/np mesh->camera; crop=(x0,y0,w,h) in image px; size=(W,H) output.
        Returns rgb (B,H,W,3), mask (B,H,W), depth (B,H,W) (0 where empty)."""
        if not torch.is_tensor(T_cam):
            T_cam = torch.tensor(np.asarray(T_cam), dtype=torch.float32, device=DEV)
        B = T_cam.shape[0]
        x0, y0, w, h = crop
        W, H = size
        sx, sy = W / w, H / h
        fx, fy, cx, cy = K[0, 0] * sx, K[1, 1] * sy, (K[0, 2] - x0) * sx, (K[1, 2] - y0) * sy
        n, fa = 0.05, 20.0
        # OpenCV camera -> clip
        P = torch.tensor([[2 * fx / W, 0, 2 * cx / W - 1, 0],
                          [0, 2 * fy / H, 2 * cy / H - 1, 0],
                          [0, 0, (fa + n) / (fa - n), -2 * fa * n / (fa - n)],
                          [0, 0, 1, 0]], dtype=torch.float32, device=DEV)
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        vc = torch.einsum('bij,nj->bni', T_cam, vh)
        clip = torch.einsum('ij,bnj->bni', P, vc)
        # nvdiffrast: y up in clip -> image row 0 at bottom; we flip after
        rast, _ = dr.rasterize(self.ctx, clip.contiguous(), self.f, (H, W))
        mask = (rast[..., 3] > 0)
        depth, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, self.f)
        out_rgb = None
        if with_tex:
            uvi, _ = dr.interpolate(self.uv[None], rast, self.f)
            out_rgb = dr.texture(self.tex.expand(B, -1, -1, -1).contiguous(), uvi, filter_mode='linear')
            out_rgb = out_rgb * mask[..., None]
        depth = depth[..., 0] * mask
        # clip y = +y_cam*.. ; nvdiffrast row 0 corresponds to clip y=-1 -> image top for OpenCV y-down. keep as is.
        return out_rgb, mask, depth


def mesh_to_cam(Twc, Tw):
    return np.linalg.inv(Twc)[None] @ Tw


def cam_to_world(Twc, Tc):
    return Twc[None] @ Tc


# ---------------------------------------------------------------- geometry helpers
def so3_exp(w):
    """w (B,3) torch -> (B,3,3)"""
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    k = w / th
    Kx = torch.zeros(w.shape[0], 3, 3, device=w.device, dtype=w.dtype)
    Kx[:, 0, 1], Kx[:, 0, 2] = -k[:, 2], k[:, 1]
    Kx[:, 1, 0], Kx[:, 1, 2] = k[:, 2], -k[:, 0]
    Kx[:, 2, 0], Kx[:, 2, 1] = -k[:, 1], k[:, 0]
    th = th[..., None]
    I = torch.eye(3, device=w.device, dtype=w.dtype)[None]
    return I + torch.sin(th) * Kx + (1 - torch.cos(th)) * (Kx @ Kx)


def rot_angle_deg(Ra, Rb):
    c = (np.trace(np.swapaxes(Ra, -1, -2) @ Rb, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def flip_T(axis):
    """180 deg rotation about mesh axis (0=x,1=y,2=z) through the mesh origin."""
    T = np.eye(4)
    d = -np.ones(3); d[axis] = 1
    T[:3, :3] = np.diag(d)
    return T


SYM_FLIPS = [np.eye(4), flip_T(0), flip_T(1), flip_T(2)]


# ---------------------------------------------------------------- observations
class Obs:
    """Per-frame cropped observations for render-and-compare."""

    def __init__(self, frames, OM, PM, bg_depth, K, crop_wh=(192, 144), hand_uv=None, hand_r=14):
        self.frames, self.OM, self.PM, self.bg, self.K = frames, OM, PM, bg_depth, K
        self.cw, self.ch = crop_wh
        self.hand_uv, self.hand_r = hand_uv, hand_r   # (F,J,2) projected hand keypoints or None

    def crop_for(self, i, center):
        cx, cy = center
        x0 = int(round(cx - self.cw / 2)); y0 = int(round(cy - self.ch / 2))
        return (x0, y0, self.cw, self.ch)

    def get(self, i, crop):
        x0, y0, w, h = crop
        H, W = self.OM.shape[1:]
        def cut(a, fill=0):
            out = np.full((h, w) + a.shape[2:], fill, a.dtype)
            ys, ye, xs, xe = max(y0, 0), min(y0 + h, H), max(x0, 0), min(x0 + w, W)
            if ye > ys and xe > xs:
                out[ys - y0:ye - y0, xs - x0:xe - x0] = a[ys:ye, xs:xe]
            return out
        img = cut(self.frames[i]).astype(np.float32) / 255.0
        om = cut(self.OM[i]); pm = cut(self.PM[i]); bg = cut(self.bg, 100.0)
        # person pixels: 1 = treated as occluder (ignored). Without hand info every person pixel is an occluder;
        # with hand keypoints only person pixels near the hands are occluders, the rest of the body counts as
        # partial negative evidence (value PERSON_NEG in [0,1): weight of 'object absent' there is 1-value).
        pm = pm.astype(np.float32)
        if self.hand_uv is not None:
            hm = np.zeros((h, w), np.uint8)
            for u, v in self.hand_uv[i]:
                cv2.circle(hm, (int(round(u - x0)), int(round(v - y0))), self.hand_r, 1, -1)
            pm = np.where(hm > 0, pm, pm * PERSON_NEG)
        return img, om, pm, bg


def batch_obs(obs, idx, crops):
    imgs, oms, pms, bgs = zip(*[obs.get(i, c) for i, c in zip(idx, crops)])
    dts = [np.minimum(cv2.distanceTransform((~o).astype(np.uint8), cv2.DIST_L2, 3), 30.0) if o.any()
           else np.full(o.shape, 30.0, np.float32) for o in oms]
    t = lambda a: torch.tensor(np.stack(a), device=DEV)
    return t(imgs), t(oms).float(), t(np.stack(pms)).float(), t(bgs).float(), t(dts).float()


def blur(x, k=5, s=1.0):
    """x (B,H,W,C) gaussian blur"""
    g = torch.exp(-(torch.arange(k, device=x.device) - k // 2) ** 2 / (2 * s * s)); g = g / g.sum()
    B, H, W, C = x.shape
    y = x.permute(0, 3, 1, 2).reshape(B * C, 1, H, W)
    y = torch.nn.functional.conv2d(y, g.view(1, 1, 1, k), padding=(0, k // 2))
    y = torch.nn.functional.conv2d(y, g.view(1, 1, k, 1), padding=(k // 2, 0))
    return y.view(B, C, H, W).permute(0, 2, 3, 1)


def render_soft(R, T_cam, K, crops, size, ss=2):
    """Batched render with per-item crop. Returns rgb (B,H,W,3), soft mask (B,H,W), depth (B,H,W)."""
    B = T_cam.shape[0]
    W, H = size
    rgbs, ms, ds = [], [], []
    # group by identical crop is unnecessary: build per-item projection matrices
    n, fa = 0.05, 20.0
    Ps = []
    for c in crops:
        x0, y0, w, h = c
        sx, sy = W * ss / w, H * ss / h
        fx, fy, cx, cy = K[0, 0] * sx, K[1, 1] * sy, (K[0, 2] - x0) * sx, (K[1, 2] - y0) * sy
        Ws, Hs = W * ss, H * ss
        Ps.append([[2 * fx / Ws, 0, 2 * cx / Ws - 1, 0], [0, 2 * fy / Hs, 2 * cy / Hs - 1, 0],
                   [0, 0, (fa + n) / (fa - n), -2 * fa * n / (fa - n)], [0, 0, 1, 0]])
    P = torch.tensor(Ps, dtype=torch.float32, device=DEV)
    vh = torch.cat([R.v, torch.ones_like(R.v[:, :1])], 1)
    vc = torch.einsum('bij,nj->bni', T_cam, vh)
    clip = torch.einsum('bij,bnj->bni', P, vc).contiguous()
    rast, _ = dr.rasterize(R.ctx, clip, R.f, (H * ss, W * ss))
    uvi, _ = dr.interpolate(R.uv[None], rast, R.f)
    rgb = dr.texture(R.tex.expand(B, -1, -1, -1).contiguous(), uvi, filter_mode='linear')
    m = (rast[..., 3:4] > 0).float()
    depth, _ = dr.interpolate(vc[..., 2:3].contiguous(), rast, R.f)
    nc = torch.einsum('bij,nj->bni', T_cam[:, :3, :3], R.vn).contiguous()
    nrm, _ = dr.interpolate(nc, rast, R.f)
    feat = torch.cat([rgb * m, m, depth * m, nrm * m], -1)
    feat = dr.antialias(feat.contiguous(), rast, clip, R.f)
    if ss > 1:
        feat = torch.nn.functional.avg_pool2d(feat.permute(0, 3, 1, 2), ss).permute(0, 2, 3, 1)
    m = feat[..., 3]
    mm = m[..., None].clamp_min(1e-4)
    rgb = feat[..., :3] / mm
    depth = feat[..., 4] / m.clamp_min(1e-4)
    nrm = feat[..., 5:8] / mm
    return rgb, m, depth, nrm


def gray(x):
    return x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114


def eval_terms(rgb, m, depth, nrm, img, om, pm, bg, dtm, occ_margin=0.05):
    """Per-item losses. Pixels covered by the person (and not object) are ignored; rendered pixels
    lying behind the static background depth (+margin) are treated as occluded (e.g. by the table).
    Photometric score: R^2 of regressing the (blurred) image grey value on albedo*[1,n_x,n_y,n_z] and 1
    (first-order lighting with unknown gain/offset), i.e. texture agreement invariant to shading."""
    occ_bg = (depth > bg + occ_margin).float() * (1 - om)
    valid = (1 - pm * (1 - om)) * (1 - occ_bg)
    mv = m * valid
    inter = (mv * om).sum((1, 2)); union = (mv + om - mv * om).sum((1, 2))
    sil = ((mv - om * valid) ** 2).sum((1, 2)) / om.sum((1, 2)).clamp_min(50)
    siou = inter / union.clamp_min(1)
    # chamfer-like term: rendered visible pixels far from the observed mask (gives a pull when not overlapping)
    cham = (mv * dtm).sum((1, 2)) / mv.sum((1, 2)).clamp_min(1) / 30.0 * (om.sum((1, 2)) > 20).float()
    w = (m > 0.5).float() * om * (1 - pm)
    gi = gray(blur(img, 5, 1.0))
    bm = blur(m[..., None], 5, 1.0)[..., 0].clamp_min(1e-3)
    al = gray(blur(rgb * m[..., None], 5, 1.0)) / bm
    nb = blur(nrm * m[..., None], 5, 1.0) / bm[..., None]
    X = torch.stack([al, al * nb[..., 0], al * nb[..., 1], al * nb[..., 2], torch.ones_like(al)], -1)  # B,H,W,5
    Xw = X * w[..., None]
    A = torch.einsum('bhwi,bhwj->bij', Xw, X) + 1e-4 * torch.eye(5, device=X.device)
    rhs = torch.einsum('bhwi,bhw->bi', Xw, gi)
    coef = torch.linalg.solve(A, rhs)
    pred = torch.einsum('bhwi,bi->bhw', X, coef)
    n = w.sum((1, 2)).clamp_min(1)
    mi = (gi * w).sum((1, 2)) / n
    var = (((gi - mi[:, None, None]) ** 2) * w).sum((1, 2)).clamp_min(1e-6)
    res = (((gi - pred) ** 2) * w).sum((1, 2))
    r2 = 1 - res / var
    return dict(sil=sil, siou=siou, ncc=r2, nov=n, cham=cham)


class PoseFitter:
    def __init__(self, R, obs, K, size=(192, 144)):
        self.R, self.obs, self.K, self.size = R, obs, K, size

    def crops_for(self, T, idx):
        cs = []
        for T_, i in zip(T, idx):
            p = self.K @ T_[:3, 3]
            cs.append(self.obs.crop_for(i, (p[0] / p[2], p[1] / p[2])))
        return cs

    def fit(self, T0, idx, pid=None, iters=60, lr_r=0.02, lr_t=0.004, w_photo=0.3, fix_z=True,
            T_prior=None, w_prior=0.0, crops=None, flat_n=None, w_flat=0.0):
        """T0 (P,4,4) initial camera-frame poses; idx: frame index per item; pid: pose id per item
        (default: item k uses pose k). Returns optimised poses (P,4,4) and final per-item terms."""
        P = len(T0)
        if pid is None:
            pid = np.arange(P)
        pid_t = torch.tensor(pid, device=DEV)
        T0t = torch.tensor(np.asarray(T0), dtype=torch.float32, device=DEV)
        if crops is None:
            crops = self.crops_for(np.asarray(T0)[pid], idx)
        ob = batch_obs(self.obs, idx, crops)
        w = torch.zeros(P, 3, device=DEV, requires_grad=True)
        dt = torch.zeros(P, 3, device=DEV, requires_grad=True)
        opt = torch.optim.Adam([{'params': [w], 'lr': lr_r}, {'params': [dt], 'lr': lr_t}])
        zmask = torch.tensor([1., 1., 0. if fix_z else 1.], device=DEV)
        cnt = torch.bincount(pid_t, minlength=P).float()

        def make(w, dt):
            T = T0t.clone()
            T = torch.cat([torch.cat([T0t[:, :3, :3] @ so3_exp(w), (T0t[:, :3, 3] + dt * zmask)[..., None]], 2),
                           T0t[:, 3:]], 1)
            return T
        for it in range(iters):
            T = make(w, dt)
            rgb, m, d, nr = render_soft(self.R, T[pid_t], self.K, crops, self.size)
            terms = eval_terms(rgb, m, d, nr, *ob)
            li = terms['sil'] + w_photo * (1 - terms['ncc']) + W_CHAM * terms['cham']
            loss = (li / cnt[pid_t]).sum()
            if T_prior is not None and w_prior > 0:
                # prior towards the initial rotation (T0); translation only very weakly
                loss = loss + w_prior * ((w ** 2).sum() + 1.0 * (dt ** 2).sum())
            if flat_n is not None and w_flat > 0:
                # object resting on a support plane: thin axis (mesh y) parallel to the plane normal
                nt = torch.tensor(flat_n, dtype=torch.float32, device=DEV)
                c = (T[:, :3, 1] * nt).sum(-1)
                loss = loss + w_flat * (1 - c.abs()).sum()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            T = make(w, dt)
            rgb, m, d, nr = render_soft(self.R, T[pid_t], self.K, crops, self.size)
            terms = eval_terms(rgb, m, d, nr, *ob)
        return T.detach().cpu().numpy().astype(np.float64), {k: v.cpu().numpy() for k, v in terms.items()}

    def score(self, T, idx, crops=None):
        with torch.no_grad():
            T = np.asarray(T)
            if crops is None:
                crops = self.crops_for(T, idx)
            ob = batch_obs(self.obs, idx, crops)
            Tt = torch.tensor(T, dtype=torch.float32, device=DEV)
            rgb, m, d, nr = render_soft(self.R, Tt, self.K, crops, self.size)
            terms = eval_terms(rgb, m, d, nr, *ob)
        return {k: v.cpu().numpy() for k, v in terms.items()}


def mean_pose(Ts):
    """Chordal L2 mean rotation + median translation of (N,4,4)."""
    M = Ts[:, :3, :3].sum(0)
    U, _, Vt = np.linalg.svd(M)
    Rm = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = np.median(Ts[:, :3, 3], 0)
    return T


def pick_frames(PM, OM, frames, n):
    """Pick up to n frames of a phase, preferring frames where the object is least covered by the person."""
    frames = np.asarray(frames)
    occ = np.array([(PM[i] & cv2.dilate(OM[i].astype(np.uint8), np.ones((9, 9))).astype(bool)).sum()
                    / max(OM[i].sum(), 1) for i in frames])
    order = np.argsort(occ, kind='stable')
    keep = np.sort(frames[order[:max(n * 2, 1)]])
    if len(keep) > n:
        keep = keep[np.linspace(0, len(keep) - 1, n).round().astype(int)]
    return keep


def rot_grid(step_deg, half_deg):
    from scipy.spatial.transform import Rotation as Rr
    v = np.arange(-half_deg, half_deg + 1e-6, step_deg)
    g = np.stack(np.meshgrid(v, v, v, indexing='ij'), -1).reshape(-1, 3)
    return Rr.from_rotvec(np.radians(g)).as_matrix()


def axis_angle(axis, ang):
    a = np.asarray(axis, float); a = a / np.linalg.norm(a)
    Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * Kx @ Kx


def flatten_pose(T, n, c):
    """Minimal rotation (about the mesh centroid c) making the mesh y axis parallel to +-n."""
    y = T[:3, 1]; n = np.asarray(n, float); n = n / np.linalg.norm(n)
    if y @ n < 0:
        n = -n
    ax = np.cross(y, n); s_ = np.linalg.norm(ax)
    if s_ < 1e-9:
        return T.copy()
    Rf = axis_angle(ax / s_, np.arctan2(s_, y @ n))
    T2 = T.copy(); T2[:3, :3] = Rf @ T[:3, :3]
    T2[:3, 3] = T[:3, 3] + T[:3, :3] @ c - T2[:3, :3] @ c
    return T2


def solve_static(fit, T0, frames, n_frames=16, n_grid_frames=4, w_photo=0.3, iters=100, topk=2,
                 grid=(7.5, 15.0), PM=None, OM=None, flips=(0, 1, 2, 3), flat_n=None, w_flat=20.0,
                 yaw_half=20.0, yaw_step=2.5):
    """One constant camera-frame pose for a static phase. For each of the 4 box-symmetry flips of T0:
    coarse rotation grid around it (silhouette+texture on a few frames), then gradient refinement of the
    best candidates on n_frames frames. Returns list (per flip) of (T, mean loss, terms)."""
    fr = pick_frames(PM, OM, frames, n_frames)
    frg = pick_frames(PM, OM, frames, n_grid_frames)
    p = fit.K @ T0[:3, 3]
    crop = fit.obs.crop_for(0, (p[0] / p[2], p[1] / p[2]))
    G = rot_grid(*grid)
    out = []
    c = fit.R.verts_np.mean(0)
    for k in flips:
        Tk = T0 @ SYM_FLIPS[k]
        cands = []
        if flat_n is not None:
            Tk = flatten_pose(Tk, flat_n, c)
            for ang in np.arange(-yaw_half, yaw_half + 1e-6, yaw_step):
                g = axis_angle(flat_n, np.radians(ang))
                T = Tk.copy()
                T[:3, :3] = g @ Tk[:3, :3]
                T[:3, 3] = Tk[:3, 3] + Tk[:3, :3] @ c - T[:3, :3] @ c
                cands.append(T)
        else:
          for g in G:
            T = Tk.copy()
            T[:3, :3] = Tk[:3, :3] @ g
            T[:3, 3] = Tk[:3, 3] + Tk[:3, :3] @ c - T[:3, :3] @ c  # rotate about mesh centroid
            cands.append(T)
        cands = np.stack(cands)
        losses = np.zeros(len(cands))
        for i in frg:
            for a in range(0, len(cands), 64):
                s = fit.score(cands[a:a + 64], [i] * len(cands[a:a + 64]), crops=[crop] * len(cands[a:a + 64]))
                losses[a:a + 64] += s['sil'] + w_photo * (1 - s['ncc'])
        best = np.argsort(losses)[:topk]
        res = []
        for b in best:
            Tf, terms = fit.fit(cands[b][None], fr, pid=np.zeros(len(fr), int), iters=iters, w_photo=w_photo,
                                crops=[crop] * len(fr), flat_n=flat_n, w_flat=w_flat if flat_n is not None else 0)
            L = terms['sil'] + w_photo * (1 - terms['ncc'])
            res.append((L.mean(), Tf[0], terms, L))
        res.sort(key=lambda r: r[0])
        out.append(res[0])
    return out, fr


def fit_traj(fit, T_init, idx, iters=60, w_photo=0.3, lam_acc_r=30.0, lam_acc_t=3000.0, lam_prior_r=0.5,
             lam_prior_t=0.0, img_mask=None, chunk=32, lr_r=0.01, lr_t=0.002, fixed_ends=None, z_free=False):
    """Joint optimisation of a trajectory segment: per-frame image terms (silhouette + texture) plus
    second-difference smoothness on rotation (chordal) and translation, plus a weak prior towards T_init.
    img_mask: optional bool per frame (False -> frame contributes no image term, e.g. held-out frames).
    fixed_ends: optional (T_before, T_after) poses adjacent to the segment (e.g. static phase poses)."""
    N = len(idx)
    T0 = torch.tensor(np.asarray(T_init), dtype=torch.float32, device=DEV)
    w = torch.zeros(N, 3, device=DEV, requires_grad=True)
    dt = torch.zeros(N, 3, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([{'params': [w], 'lr': lr_r}, {'params': [dt], 'lr': lr_t}])
    zmask = torch.tensor([1., 1., 1. if z_free else 0.], device=DEV)
    crops = fit.crops_for(np.asarray(T_init), idx)
    if img_mask is None:
        img_mask = np.ones(N, bool)
    chunks = [np.arange(s, min(s + chunk, N)) for s in range(0, N, chunk)]
    obs_c = [batch_obs(fit.obs, [idx[j] for j in c], [crops[j] for j in c]) for c in chunks]
    ends = None
    if fixed_ends is not None:
        ends = [None if e is None else torch.tensor(e, dtype=torch.float32, device=DEV) for e in fixed_ends]

    def poses():
        Rm = T0[:, :3, :3] @ so3_exp(w)
        t = T0[:, :3, 3] + dt * zmask
        return Rm, t
    for it in range(iters):
        opt.zero_grad()
        for c, ob in zip(chunks, obs_c):
            Rm, t = poses()
            T = torch.cat([torch.cat([Rm[c], t[c][..., None]], 2), T0[c, 3:]], 1)
            rgb, m, d, nr = render_soft(fit.R, T, fit.K, [crops[j] for j in c], fit.size)
            terms = eval_terms(rgb, m, d, nr, *ob)
            li = (terms['sil'] + w_photo * (1 - terms['ncc']) + W_CHAM * terms['cham']) * torch.tensor(img_mask[c], device=DEV).float()
            li.sum().backward()
        Rm, t = poses()
        if ends is not None:
            if ends[0] is not None:
                Rm = torch.cat([ends[0][None, :3, :3], Rm]); t = torch.cat([ends[0][None, :3, 3], t])
            if ends[1] is not None:
                Rm = torch.cat([Rm, ends[1][None, :3, :3]]); t = torch.cat([t, ends[1][None, :3, 3]])
        reg = 0
        if len(Rm) >= 3:
            reg = lam_acc_r * ((Rm[2:] - 2 * Rm[1:-1] + Rm[:-2]) ** 2).sum() + \
                  lam_acc_t * ((t[2:] - 2 * t[1:-1] + t[:-2]) ** 2).sum()
        reg = reg + lam_prior_r * (w ** 2).sum() + lam_prior_t * (dt ** 2).sum()
        reg.backward()
        opt.step()
    with torch.no_grad():
        Rm, t = poses()
        T = torch.cat([torch.cat([Rm, t[..., None]], 2), T0[:, 3:]], 1)
    return T.cpu().numpy().astype(np.float64)


def score_traj(fit, T, idx, bs=64):
    out = {}
    for s in range(0, len(idx), bs):
        sc = fit.score(T[s:s + bs], list(idx[s:s + bs]))
        for k, v in sc.items():
            out.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def shift_search(fit, T, idx, max_shift=40, bs=32, w_cham=1.0):
    """For each frame, find the image-plane shift (pixels) of the rendered silhouette of pose T that best
    explains the object mask (same loss as eval_terms' silhouette + chamfer, evaluated for all integer shifts
    by correlation). Returns shifted poses (moved parallel to the image plane at constant z), loss at the
    best shift, and loss at zero shift."""
    import torch.nn.functional as Fn
    T = np.asarray(T)
    out_T = T.copy(); Lbest = np.zeros(len(T)); L0 = np.zeros(len(T)); shifts = np.zeros((len(T), 2))
    S = max_shift
    for s in range(0, len(T), bs):
        ii = list(idx[s:s + bs]); Ts = T[s:s + bs]
        crops = fit.crops_for(Ts, ii)
        img, om, pm, bg, dtm = batch_obs(fit.obs, ii, crops)
        with torch.no_grad():
            _, m, d, _ = render_soft(fit.R, torch.tensor(Ts, dtype=torch.float32, device=DEV), fit.K, crops, fit.size)
        H, W = m.shape[1:]
        for j in range(len(ii)):
            mj = m[j]
            if mj.sum() < 5 or om[j].sum() < 20:
                continue
            z = float(Ts[j, 2, 3])
            occ = ((z > bg[j] + 0.05).float()) * (1 - om[j])
            valid = (1 - pm[j] * (1 - om[j])) * (1 - occ)
            ys, xs = torch.nonzero(mj > 0.01, as_tuple=True)
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
            k = mj[y0:y1, x0:x1]
            pad = lambda a: Fn.pad(a[None, None], (S, S, S, S))
            A = Fn.conv2d(pad(valid), (k * k)[None, None])[0, 0]
            B = Fn.conv2d(pad(valid * om[j]), k[None, None])[0, 0]
            Cm = Fn.conv2d(pad(dtm[j] * valid), k[None, None])[0, 0]
            Kv = Fn.conv2d(pad(valid), k[None, None])[0, 0]
            C = (valid * om[j]).sum()
            # index of zero shift: top-left at (y0, x0) in unpadded coords -> (y0+S, x0+S) in padded
            dy = torch.arange(-S, S + 1, device=DEV); dx = torch.arange(-S, S + 1, device=DEV)
            YY = (y0 + S + dy)[:, None].clamp(0, A.shape[0] - 1); XX = (x0 + S + dx)[None, :].clamp(0, A.shape[1] - 1)
            sil = (A[YY, XX] - 2 * B[YY, XX] + C) / max(float(om[j].sum()), 50.0)
            cham = Cm[YY, XX] / Kv[YY, XX].clamp_min(1) / 30.0
            L = sil + w_cham * cham + 0.002 * torch.sqrt(dy[:, None] ** 2 + dx[None, :] ** 2).float()
            b = int(torch.argmin(L))
            by, bx = b // L.shape[1], b % L.shape[1]
            L0[s + j] = float(L[S, S]); Lbest[s + j] = float(L[by, bx])
            du, dv = float(dx[bx]), float(dy[by])
            shifts[s + j] = (du, dv)
            out_T[s + j, 0, 3] += du * z / fit.K[0, 0]
            out_T[s + j, 1, 3] += dv * z / fit.K[1, 1]
    return out_T, Lbest, L0, shifts


def viterbi(unary, trans):
    """unary (N,C); trans(t) -> (C,C) cost from candidate at t-1 to t. Returns best path."""
    N, C = unary.shape
    cost = unary[0].copy(); back = np.zeros((N, C), int)
    for t in range(1, N):
        tot = cost[:, None] + trans(t)
        back[t] = tot.argmin(0)
        cost = tot.min(0) + unary[t]
    path = np.zeros(N, int); path[-1] = cost.argmin()
    for t in range(N - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def smooth_series(y, w, lam, groups=None):
    """Weighted 2nd-difference smoothing: min sum w (x-y)^2 + lam sum (x[t+1]-2x[t]+x[t-1])^2.
    groups: list of (a,b) index ranges forced to a constant value (static phases)."""
    import scipy.sparse as sp, scipy.sparse.linalg as spl
    N = len(y)
    # parametrise x = P u, where frames of a static group share one variable
    var = np.arange(N)
    if groups:
        for a, b in groups:
            var[a:b + 1] = var[a]
    uniq, var = np.unique(var, return_inverse=True)
    P = sp.csr_matrix((np.ones(N), (np.arange(N), var)), shape=(N, len(uniq)))
    D = sp.diags([np.ones(N - 2), -2 * np.ones(N - 2), np.ones(N - 2)], [0, 1, 2], shape=(N - 2, N))
    Wm = sp.diags(w)
    A = P.T @ (Wm + lam * D.T @ D) @ P
    u = spl.spsolve(A.tocsc(), P.T @ (w * y))
    return P @ u
