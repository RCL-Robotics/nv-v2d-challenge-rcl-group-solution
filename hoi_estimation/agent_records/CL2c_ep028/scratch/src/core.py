import numpy as np, cv2, h5py, json, torch, trimesh
dev = 'cuda'
W, H = 1536, 1152
TRUNC = 60  # px truncation of distance maps

def load_inputs(inp):
    cam = json.load(open(f'{inp}/camera.json'))
    rec = np.load(f'{inp}/reconstruction.npz')
    gp = json.load(open(f'{inp}/ground_plane.json'))
    return cam, rec, gp

def sample_mesh_points(inp, n=4000, seed=0):
    m = trimesh.load(f'{inp}/object_mesh.glb', force='scene', process=False).to_geometry()
    rng = np.random.default_rng(seed)
    v = np.asarray(m.vertices); f = np.asarray(m.faces); uv = np.asarray(m.visual.uv)
    tex = np.asarray(m.visual.material.baseColorTexture.convert('RGB'))
    area = trimesh.triangles.area(v[f]); p = area / area.sum()
    fi = rng.choice(len(f), n, p=p)
    b = rng.random((n, 2)); s = b.sum(1) > 1; b[s] = 1 - b[s]
    bc = np.stack([1 - b.sum(1), b[:, 0], b[:, 1]], 1)
    pts = np.einsum('nk,nkj->nj', bc, v[f[fi]])
    uvs = np.einsum('nk,nkj->nj', bc, uv[f[fi]])
    th, tw = tex.shape[:2]
    px = np.clip((uvs[:, 0] * (tw - 1)).round().astype(int), 0, tw - 1)
    py = np.clip(((1 - uvs[:, 1]) * (th - 1)).round().astype(int), 0, th - 1)
    col = tex[py, px]
    hsv = cv2.cvtColor(col[None].astype(np.uint8), cv2.COLOR_RGB2HSV)[0]
    orange = (hsv[:, 0] >= 3) & (hsv[:, 0] <= 18) & (hsv[:, 1] > 110) & (hsv[:, 2] > 80)
    nrm = np.asarray(m.face_normals)[fi]
    return pts, orange, nrm, m

def orange_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # object orange: hue 3-11, strongly saturated (skin: hue 13-16, sat ~135 -> excluded)
    return (hsv[..., 0] >= 3) & (hsv[..., 0] <= 11) & (hsv[..., 1] >= 145) & (hsv[..., 2] >= 85)

def frame_features(bgr, om, pm, n_mask=1500, n_or=400, seed=0):
    rng = np.random.default_rng(seed)
    # orange evidence: anywhere within ~250 px of the SAM2 object mask (SAM2 often misses the head), not on the person
    near = cv2.dilate(om.astype(np.uint8), np.ones((501, 501), np.uint8)) > 0 if om.any() else np.ones_like(om)
    o = orange_mask(bgr) & near & ~(pm & ~om)
    o = cv2.morphologyEx(o.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    om = om | (cv2.dilate(o.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    def dt(m):
        if m.sum() == 0: return np.full(m.shape, TRUNC, np.uint8)
        d = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5)
        return np.minimum(d, TRUNC).astype(np.uint8)
    def samp(m, n):
        ys, xs = np.nonzero(m)
        if len(xs) > n:
            k = rng.choice(len(xs), n, replace=False); xs, ys = xs[k], ys[k]
        return np.stack([xs, ys], 1).astype(np.float32)
    occ = pm & ~om
    return dict(dtm=dt(om), dto=dt(o), occ=occ, mpts=samp(om, n_mask), opts=samp(o, n_or), area=int(om.sum()), oarea=int(o.sum()))

def _feat_worker(a):
    i, bgr, om, pm = a
    return i, frame_features(bgr, om, pm, seed=i)

class FrameBank:
    """precomputed per-frame evidence (distance maps, occlusion, sampled mask / orange pixels)"""
    def __init__(self, inp, nproc=16):
        import multiprocessing as mp
        cap = cv2.VideoCapture(f'{inp}/video.mp4')
        h5 = h5py.File(f'{inp}/masks.h5', 'r')
        F = h5['object'].shape[0]
        self.F = F
        self.dtm = np.zeros((F, H, W), np.uint8); self.dto = np.zeros((F, H, W), np.uint8)
        self.occ = np.zeros((F, H, W), bool)
        self.mpts = [None] * F; self.opts = [None] * F; self.area = np.zeros(F, int); self.oarea = np.zeros(F, int)
        def gen():
            for i in range(F):
                ok, bgr = cap.read(); assert ok
                yield i, bgr, h5['object'][i], h5['person'][i]
        with mp.get_context('fork').Pool(nproc) as pool:
            for i, ft in pool.imap(_feat_worker, gen(), chunksize=4):
                self.dtm[i] = ft['dtm']; self.dto[i] = ft['dto']; self.occ[i] = ft['occ']
                self.mpts[i] = ft['mpts']; self.opts[i] = ft['opts']; self.area[i] = ft['area']; self.oarea[i] = ft['oarea']

# ---------------- geometry helpers (torch) ----------------
def so3_exp(w):
    th = w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = w / th
    Kx = torch.zeros(w.shape[:-1] + (3, 3), device=w.device, dtype=w.dtype)
    Kx[..., 0, 1] = -k[..., 2]; Kx[..., 0, 2] = k[..., 1]; Kx[..., 1, 0] = k[..., 2]
    Kx[..., 1, 2] = -k[..., 0]; Kx[..., 2, 0] = -k[..., 1]; Kx[..., 2, 1] = k[..., 0]
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(Kx)
    s = torch.sin(th)[..., None]; c = torch.cos(th)[..., None]
    return I + s * Kx + (1 - c) * Kx @ Kx

def rot_angle_deg(R1, R2):
    """angle between rotations, numpy (...,3,3)"""
    tr = np.einsum('...ji,...ji->...', R1, R2)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))

class Scorer:
    """point-based image + physics evidence for poses. Pose = (R (3x3), t (3,)) mapping (X - c) to camera."""
    def __init__(self, bank, pts, orange, normals, K, c, n_model=700, seed=0):
        self.b = bank
        rng = np.random.default_rng(seed)
        self.K = torch.tensor(K, dtype=torch.float32, device=dev)
        self.c = c
        P = torch.tensor(pts - c, dtype=torch.float32, device=dev)
        self.P = P; self.Nrm = torch.tensor(normals, dtype=torch.float32, device=dev)
        self.orange = torch.tensor(orange, device=dev)
        sub = rng.choice(len(pts), n_model, replace=False)
        self.Psub = P[sub]
        self.Por = P[self.orange]; self.Nor = self.Nrm[self.orange]
    def maps(self, idx):
        dtm = torch.tensor(np.asarray(self.b.dtm[idx]), device=dev).half()
        dto = torch.tensor(np.asarray(self.b.dto[idx]), device=dev).half()
        occ = torch.tensor(np.asarray(self.b.occ[idx]), device=dev).half()
        return torch.stack([dtm, dto, occ], 1)  # B,3,H,W
    def project(self, R, t, X):
        # R (B,G,3,3), t (B,G,3) -> uv (B,G,N,2)
        Xc = torch.einsum('bgij,nj->bgni', R, X) + t[:, :, None]
        z = Xc[..., 2].clamp(min=0.05)
        u = self.K[0, 0] * Xc[..., 0] / z + self.K[0, 2]
        v = self.K[1, 1] * Xc[..., 1] / z + self.K[1, 2]
        return torch.stack([u, v], -1), Xc
    def sample(self, maps, uv):
        B, G, N, _ = uv.shape
        g = torch.stack([uv[..., 0] / (W - 1) * 2 - 1, uv[..., 1] / (H - 1) * 2 - 1], -1).reshape(B, G * N, 1, 2).to(maps.dtype)
        s = torch.nn.functional.grid_sample(maps, g, mode='bilinear', padding_mode='border', align_corners=True)[..., 0].float()
        s = s.reshape(B, -1, G, N).permute(0, 2, 1, 3)  # B,G,C,N
        inside = ((uv[..., 0] >= 0) & (uv[..., 0] <= W - 1) & (uv[..., 1] >= 0) & (uv[..., 1] <= H - 1)).float()
        return s, inside  # B,G,C,N ; B,G,N
    def image_terms(self, R, t, maps, mp, mpn, op, opn, sig=10.):
        rho = lambda d: d ** 2 / (d ** 2 + sig ** 2)
        B, G = R.shape[:2]
        uv, Xc = self.project(R, t, self.P)
        s, ins = self.sample(maps, uv)
        dm, occ = s[:, :, 0], s[:, :, 2]
        w = ins * (1 - occ)
        m2i = (rho(dm) * w).sum(-1) / w.sum(-1).clamp(min=1)
        outside = 1 - ins.mean(-1)
        uvs, _ = self.project(R, t, self.Psub)
        M = mp.shape[1]
        d = torch.cdist(mp[:, None].expand(B, G, M, 2).reshape(B * G, M, 2), uvs.reshape(B * G, -1, 2)).min(-1).values.reshape(B, G, M)
        i2m = (rho(d) * mpn[:, None]).sum(-1) / mpn[:, None].sum(-1).clamp(min=1)
        uvo, Xo = self.project(R, t, self.Por)
        nrm = torch.einsum('bgij,nj->bgni', R, self.Nor)
        face = ((nrm * (-Xo)).sum(-1) > 0).float()
        so, inso = self.sample(maps, uvo)
        wo = face * inso * (1 - so[:, :, 2])
        o2i = (rho(so[:, :, 1]) * wo).sum(-1) / wo.sum(-1).clamp(min=1)
        Mo = op.shape[1]
        do = torch.cdist(op[:, None].expand(B, G, Mo, 2).reshape(B * G, Mo, 2), uvo.reshape(B * G, -1, 2)).reshape(B, G, Mo, -1) + (1 - face)[:, :, None] * 1e3
        i2o = (rho(do.min(-1).values) * opn[:, None]).sum(-1) / opn[:, None].sum(-1).clamp(min=1)
        return dict(m2i=m2i, i2m=i2m, o2i=o2i, i2o=i2o, outside=outside, wvis=w.mean(-1))

def pad_pts(lst, n=None):
    n = n or max(1, max(len(a) for a in lst))
    out = np.zeros((len(lst), n, 2), np.float32); msk = np.zeros((len(lst), n), np.float32)
    for k, a in enumerate(lst):
        a = a[:n]; out[k, :len(a)] = a; msk[k, :len(a)] = 1
    return torch.tensor(out, device=dev), torch.tensor(msk, device=dev)
