"""Planar (table-top) pose model + differentiable silhouette fitting."""
import numpy as np, torch, json, h5py, cv2
import nvdiffrast.torch as dr

CROP = (640, 380, 512, 416)  # x0,y0,w,h full-res crop that contains the monitor in every frame (checked on masks)

def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def rot_axis(axis, ang):
    axis = axis / np.linalg.norm(axis)
    K = skew(axis)
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K

def rot_a_to_b(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = a @ b
    if np.linalg.norm(v) < 1e-9: return np.eye(3)
    K = skew(v)
    return np.eye(3) + K + K @ K * (1 / (1 + c))

class PlanarModel:
    """T_world_mesh(theta,a,b) = [Rot(up,theta) R0 | p(a,b) - Rot R0 c]  with c the base-contact point (mesh),
    p on the table plane up.x + d_t = 0  (up = world up unit vector)."""
    def __init__(self, up, d_t, R0, c, origin):
        self.up = up / np.linalg.norm(up); self.d_t = d_t; self.R0 = R0; self.c = c
        # in-plane basis
        e1 = np.cross(self.up, [0, 0, 1.0]); e1 /= np.linalg.norm(e1)
        e2 = np.cross(self.up, e1)
        self.e1, self.e2 = e1, e2
        self.o = origin - (origin @ self.up + d_t) * self.up  # project onto plane

    def pose(self, th, a, b):
        R = rot_axis(self.up, th) @ self.R0
        p = self.o + a * self.e1 + b * self.e2
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p - R @ self.c
        return T

    def params_from_pose(self, T):
        R = T[:3, :3]
        # yaw: rotation about up that best aligns R0 to R
        M = R @ self.R0.T
        # project M onto rotations about up: angle from the in-plane components
        x = M @ self.e1
        th = np.arctan2(x @ self.e2, x @ self.e1)
        p = T[:3, :3] @ self.c + T[:3, 3]
        return th, (p - self.o) @ self.e1, (p - self.o) @ self.e2

    def torch_pose(self, th, a, b, dev):
        """batched torch version; th,a,b (B,) tensors -> (B,4,4) world poses"""
        up = torch.tensor(self.up, dtype=torch.float32, device=dev)
        K = torch.tensor(skew(self.up), dtype=torch.float32, device=dev)
        I = torch.eye(3, device=dev)
        Rz = I[None] + torch.sin(th)[:, None, None] * K[None] + (1 - torch.cos(th))[:, None, None] * (K @ K)[None]
        R = Rz @ torch.tensor(self.R0, dtype=torch.float32, device=dev)[None]
        o = torch.tensor(self.o, dtype=torch.float32, device=dev)
        e1 = torch.tensor(self.e1, dtype=torch.float32, device=dev); e2 = torch.tensor(self.e2, dtype=torch.float32, device=dev)
        c = torch.tensor(self.c, dtype=torch.float32, device=dev)
        p = o[None] + a[:, None] * e1[None] + b[:, None] * e2[None]
        t = p - (R @ c)
        T = torch.zeros(th.shape[0], 4, 4, device=dev); T[:, :3, :3] = R; T[:, :3, 3] = t; T[:, 3, 3] = 1
        return T


class SilRenderer:
    """Renders soft silhouettes of the full mesh and of the screen panel in the fixed crop."""
    def __init__(self, vf, K, T_world_cam, scale=0.5, dev="cuda"):
        self.dev = dev
        self.glctx = dr.RasterizeCudaContext()
        v, f = vf
        self.v = torch.tensor(v, dtype=torch.float32, device=dev)
        self.f = torch.tensor(f, dtype=torch.int32, device=dev)
        # part label per vertex: 1 = screen panel (x<0.09 top part, y<0.045 excludes stand mount), 0 = stand/base
        lab = ((v[:, 0] < 0.09) & (v[:, 1] < 0.045)).astype(np.float32)
        # front-of-screen label: panel vertices on the front side of the panel mid-plane whose normal faces forward
        import trimesh
        vn = trimesh.Trimesh(v, f, process=False).vertex_normals
        pan = lab > 0
        c = v[pan].mean(0); _, _, vt = np.linalg.svd(v[pan] - c)
        nf = vt[2] if vt[2][1] < 0 else -vt[2]  # screen faces mesh -y
        front = (pan & (vn @ nf > 0.7) & ((v - c) @ nf > 0)).astype(np.float32)
        self.attr = torch.tensor(np.stack([lab, front], 1), device=dev)
        self.T_cam_world = torch.tensor(np.linalg.inv(T_world_cam), dtype=torch.float32, device=dev)
        self.set_scale(K, scale)

    def set_scale(self, K, scale):
        x0, y0, w, h = CROP
        self.scale = scale
        self.rw, self.rh = int(round(w * scale)), int(round(h * scale))
        fx, fy = K[0][0] * scale, K[1][1] * scale
        cx, cy = (K[0][2] - x0) * scale, (K[1][2] - y0) * scale
        n, fa = 0.05, 20.0
        self.P = torch.tensor([[2 * fx / self.rw, 0, 2 * cx / self.rw - 1, 0],
                               [0, 2 * fy / self.rh, 2 * cy / self.rh - 1, 0],
                               [0, 0, (fa + n) / (fa - n), -2 * fa * n / (fa - n)],
                               [0, 0, 1, 0]], dtype=torch.float32, device=self.dev)

    def render(self, T_world_mesh):
        T = self.T_cam_world[None] @ T_world_mesh
        vh = torch.cat([self.v, torch.ones_like(self.v[:, :1])], 1)
        vc = torch.einsum("bij,nj->bni", T, vh)
        clip = torch.einsum("ij,bnj->bni", self.P, vc).contiguous()
        rast, _ = dr.rasterize(self.glctx, clip, self.f, (self.rh, self.rw))
        lab, _ = dr.interpolate(self.attr[None].expand(T.shape[0], -1, -1).contiguous(), rast, self.f)
        cov = (rast[..., 3:4] > 0).float()
        out = torch.cat([cov, lab * cov], -1)
        out = dr.antialias(out.contiguous(), rast, clip, self.f)
        return out  # (B,h,w,3): full, screen, front

def load_crop_masks(masks_path, scale, frames=None):
    x0, y0, w, h = CROP
    f = h5py.File(masks_path, "r")
    O, P = f["object"], f["person"]
    idx = range(O.shape[0]) if frames is None else frames
    rw, rh = int(round(w * scale)), int(round(h * scale))
    mo, mp = [], []
    for i in idx:
        o = O[i, y0:y0 + h, x0:x0 + w].astype(np.float32)
        p = P[i, y0:y0 + h, x0:x0 + w].astype(np.float32)
        mo.append(cv2.resize(o, (rw, rh), interpolation=cv2.INTER_AREA))
        mp.append(cv2.resize(p, (rw, rh), interpolation=cv2.INTER_AREA))
    return np.stack(mo), np.stack(mp)


def setup(inputs="/work/inputs"):
    """Load inputs and build the planar table-top model from the baseline + floor plane."""
    import trimesh
    S = {}
    cam = json.load(open(f"{inputs}/camera.json")); S["K"] = np.array(cam["K"]); S["Twc"] = np.array(cam["T_world_cam"])
    pl = json.load(open(f"{inputs}/ground_plane.json"))["plane"]
    up = np.array(pl[:3]); up /= np.linalg.norm(up)  # plane normal points to world up (-y)
    S["up"] = up; S["floor_d"] = pl[3] / np.linalg.norm(pl[:3])
    r = np.load(f"{inputs}/reconstruction.npz")
    Tb = r["T_world_mesh"].astype(np.float64); S["Tb"] = Tb; S["rec"] = r
    mesh = trimesh.load(f"{inputs}/object_mesh.glb", force="scene", process=False).to_geometry(); S["mesh"] = mesh
    v = np.asarray(mesh.vertices)
    # resting face of the base: the convex-hull face whose normal is closest to mesh +x (monitor 'down').
    hull = mesh.convex_hull
    fn, ar = hull.face_normals, hull.area_faces
    sel = fn[:, 0] > np.cos(np.radians(8))  # hull faces forming the flat bottom of the base
    nr = (fn[sel] * ar[sel, None]).sum(0); nr /= np.linalg.norm(nr)
    hmax = (v @ nr).max()
    c = (hull.triangles_center[sel] * ar[sel, None]).sum(0) / ar[sel].sum()
    c = c - (c @ nr - hmax) * nr  # base-contact centre on the rest plane
    S["nr"], S["c"] = nr, c
    pc = np.einsum("fij,j->fi", Tb, np.r_[c, 1])[:, :3]
    dn = np.einsum("fij,j->fi", Tb[:, :3, :3], nr)
    tilt = np.degrees(np.arccos(np.clip(dn @ (-up), -1, 1)))
    S["tilt_base"] = tilt
    good = tilt < 10  # baseline frames that are physically plausible (monitor upright on the table)
    # drop short 'good' islands inside failure segments (a frame can be upright by chance with a wrong yaw)
    idx = np.arange(len(good)); lab = np.cumsum(np.r_[1, np.diff(good.astype(int)) != 0])
    for l in np.unique(lab):
        run = lab == l
        if good[run].all() and run.sum() < 5: good[run] = False
    S["good"] = good
    d_t = -np.median(pc[good] @ up)  # table plane height from the baseline's contact points
    i0 = np.nonzero(good)[0][0]
    R0 = rot_a_to_b(Tb[i0, :3, :3] @ nr, -up) @ Tb[i0, :3, :3]
    pm = PlanarModel(up, d_t, R0, c, pc[i0]); S["pm"] = pm
    par = np.array([pm.params_from_pose(T) for T in Tb])
    idx = np.arange(len(Tb))
    par[good, 0] = np.unwrap(par[good, 0])  # unwrap over plausible frames only
    par_i = par.copy()
    for k in range(3):
        par_i[~good, k] = np.interp(idx[~good], idx[good], par[good, k])
    S["par_base_raw"], S["par_base"] = par, par_i
    return S


def decimate(mesh, n_faces=8000):
    """Merged + quadric-decimated copy of the mesh for fast silhouette rendering (deterministic)."""
    import trimesh, open3d as o3d
    mm = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces), process=True)
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mm.vertices), o3d.utility.Vector3iVector(mm.faces))
    d = om.simplify_quadric_decimation(n_faces)
    return np.asarray(d.vertices), np.asarray(d.triangles)


def read_video_gray_crops(video_path, scale):
    """Grayscale crops (F,h,w) float32 in [0,255] at the silhouette-render resolution."""
    x0, y0, w, h = CROP
    rw, rh = int(round(w * scale)), int(round(h * scale))
    cap = cv2.VideoCapture(video_path); out = []
    while True:
        ok, fr = cap.read()
        if not ok: break
        g = cv2.cvtColor(fr[y0:y0 + h, x0:x0 + w], cv2.COLOR_BGR2GRAY).astype(np.float32)
        out.append(cv2.resize(g, (rw, rh), interpolation=cv2.INTER_AREA))
    return np.stack(out)


def occluder_maps(S, MP, scale, z_obj, margin=0.10, radius_px=12):
    """Person pixels that are in FRONT of the object: person-mask pixels whose splatted reconstructed-human
    depth is < object depth + margin. Person pixels behind the object are kept as evidence (the object
    cannot be there, otherwise SAM2 would have labelled them object)."""
    x0, y0, w, h = CROP
    rw, rh = MP.shape[2], MP.shape[1]
    K = S["K"]; Tcw = np.linalg.inv(S["Twc"])
    hv = S["rec"]["human_vertices_world"]
    occ = np.zeros_like(MP)
    rad = max(1, int(round(radius_px * scale)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
    for i in range(len(MP)):
        pc = hv[i] @ Tcw[:3, :3].T + Tcw[:3, 3]
        u = (K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2] - x0) * scale
        v = (K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2] - y0) * scale
        near = pc[:, 2] < z_obj[i] + margin
        m = np.zeros((rh, rw), np.uint8)
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        ok = near & (ui >= 0) & (ui < rw) & (vi >= 0) & (vi < rh)
        m[vi[ok], ui[ok]] = 1
        m = cv2.dilate(m, ker)
        occ[i] = (MP[i] > 0.5) * m
    return occ


class Evidence:
    """Per-frame image evidence and a batched cost for many pose hypotheses of one frame."""
    def __init__(self, S, scale=0.5, dev="cuda", inputs="/work/inputs"):
        self.dev = dev; self.scale = scale
        self.MO, self.MP = load_crop_masks(f"{inputs}/masks.h5", scale)
        self.G = read_video_gray_crops(f"{inputs}/video.mp4", scale)
        Tcw = np.linalg.inv(S["Twc"])
        z_obj = (np.einsum("ij,fjk->fik", Tcw, S["Tb"]))[:, 2, 3]
        z_obj = np.where(S["good"], z_obj, np.median(z_obj))
        self.OCC = occluder_maps(S, self.MP, scale, z_obj)
        # brightness normalised: white screen (~170-200) -> 1, dark back of the monitor (<60) -> 0
        self.IN = np.clip((self.G - 60.0) / 80.0, 0, 1)

    def frame(self, i):
        t = lambda x: torch.tensor(x, device=self.dev)
        M = t(self.MO[i]); V = 1 - t(self.OCC[i]); I = t(self.IN[i])
        return M, V, I

    two_sided = True

    @staticmethod
    def cost(out, M, V, I, w_fb=0.5, blur=0.0):
        Af, As, Afr = out[..., 0], out[..., 1], out[..., 2]
        if blur > 0.5:
            k = int(blur * 3) * 2 + 1
            g = torch.exp(-(torch.arange(k, device=M.device) - k // 2) ** 2 / (2 * blur ** 2)); g = g / g.sum()
            bl = lambda x: torch.nn.functional.conv2d(torch.nn.functional.conv2d(x[:, None], g[None, None, None, :], padding=(0, k // 2)), g[None, None, :, None], padding=(k // 2, 0))[:, 0]
            Af, As, M2 = bl(Af), bl(As), bl(M[None])[0]
        else:
            M2 = M
        norm = (V * M).sum() + 1.0
        if Evidence.two_sided:
            # SAM2 masks sometimes contain the whole monitor, sometimes only the screen panel (stand/base dropped):
            # two-sided silhouette error against both explanations, keep the better one per hypothesis
            e_full = (V * (M2 - Af).abs()).sum((1, 2)) / norm
            e_scr = (V * (M2 - As).abs()).sum((1, 2)) / norm
            sil = torch.minimum(e_full, e_scr)
        else:
            sil = ((V * M2 * (1 - Af)).sum((1, 2)) + (V * (1 - M2) * As).sum((1, 2))) / norm
        # front/back: rendered screen-front pixels should be bright, other screen-panel pixels dark
        W = V * M
        fb = ((W * Afr * (1 - I)).sum((1, 2)) + (W * (As - Afr).clamp(min=0) * I).sum((1, 2))) / ((W * As).sum((1, 2)) + 1.0)
        return sil + w_fb * fb, sil, fb


def viterbi_yaw(D, sig_deg, step_deg):
    """D (F,N) data cost on a circular yaw grid; sig_deg (F,) allowed yaw change per frame (deg).
    Transition cost ((dtheta)/sig)^2 / 2. Returns state index per frame."""
    F_, N = D.shape
    k = np.arange(N); dd = np.minimum(np.abs(k[:, None] - k[None]), N - np.abs(k[:, None] - k[None])) * step_deg
    acc = D[0].copy(); back = np.zeros((F_, N), int)
    for t in range(1, F_):
        tr = 0.5 * (dd / sig_deg[t]) ** 2
        tot = acc[:, None] + tr  # from i to j
        back[t] = np.argmin(tot, 0)
        acc = tot[back[t], k] + D[t]
    path = np.zeros(F_, int); path[-1] = np.argmin(acc)
    for t in range(F_ - 1, 0, -1): path[t - 1] = back[t, path[t]]
    return path


class FreeDepthModel:
    """Upright pose (yaw about world up, from PlanarModel.R0) + free translation of the base-contact point,
    parametrised in the camera frame as p_cam = z0*exp(ls) * [x0+dx, y0+dy, 1]."""
    def __init__(self, pm, Twc):
        self.pm = pm; self.Twc = Twc; self.Tcw = np.linalg.inv(Twc)

    def contact_cam(self, T_world_mesh):
        p = T_world_mesh[:3, :3] @ self.pm.c + T_world_mesh[:3, 3]
        return self.Tcw[:3, :3] @ p + self.Tcw[:3, 3]

    def torch_pose(self, th, x, y, z, dev):
        """th (B,), x,y normalised image coords of the contact point, z its camera depth -> (B,4,4) world"""
        pm = self.pm
        K = torch.tensor(skew(pm.up), dtype=torch.float32, device=dev)
        I = torch.eye(3, device=dev)
        Rz = I[None] + torch.sin(th)[:, None, None] * K[None] + (1 - torch.cos(th))[:, None, None] * (K @ K)[None]
        R = Rz @ torch.tensor(pm.R0, dtype=torch.float32, device=dev)[None]
        pc = torch.stack([x * z, y * z, z], -1)
        Twc = torch.tensor(self.Twc, dtype=torch.float32, device=dev)
        pw = pc @ Twc[:3, :3].T + Twc[:3, 3]
        c = torch.tensor(pm.c, dtype=torch.float32, device=dev)
        t = pw - R @ c
        T = torch.zeros(th.shape[0], 4, 4, device=dev); T[:, :3, :3] = R; T[:, :3, 3] = t; T[:, 3, 3] = 1
        return T

    def pose(self, th, x, y, z):
        R = rot_axis(self.pm.up, th) @ self.pm.R0
        pw = self.Twc[:3, :3] @ np.array([x * z, y * z, z]) + self.Twc[:3, 3]
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pw - R @ self.pm.c
        return T

    def params_from_pose(self, T):
        th = self.pm.params_from_pose(T)[0]
        pc = self.contact_cam(T)
        return th, pc[0] / pc[2], pc[1] / pc[2], pc[2]


def fit_frames(sr, E, fm, frames, th_init, xyz_init, iters=60, fit_th=False, lr_th=0.02, blur0=3.0, dev="cuda"):
    """Batched local fits: for each frame, NH hypotheses (th_init (n,NH)); optimises translation (and yaw if fit_th).
    xyz_init (n,3). Returns th (n,NH), xyz (n,NH,3), cost (n,NH,3)."""
    n, NH = th_init.shape
    TH = np.zeros((n, NH)); XYZ = np.zeros((n, NH, 3)); CO = np.zeros((n, NH, 3))
    for k, fi in enumerate(frames):
        M, V, I = E.frame(fi)
        th = torch.tensor(th_init[k], dtype=torch.float32, device=dev).requires_grad_(fit_th)
        x = torch.full((NH,), xyz_init[k, 0], device=dev, requires_grad=True)
        y = torch.full((NH,), xyz_init[k, 1], device=dev, requires_grad=True)
        ls = torch.zeros(NH, device=dev, requires_grad=True)
        groups = [{"params": [x, y], "lr": 0.002}, {"params": [ls], "lr": 0.006}]
        if fit_th: groups.append({"params": [th], "lr": lr_th})
        opt = torch.optim.Adam(groups)
        for it in range(iters):
            z = xyz_init[k, 2] * torch.exp(ls)
            o = sr.render(fm.torch_pose(th, x, y, z, dev))
            L, _, _ = Evidence.cost(o, M, V, I, blur=max(blur0 * (1 - it / (0.6 * iters)), 0))
            opt.zero_grad(); L.sum().backward(); opt.step()
        with torch.no_grad():
            z = xyz_init[k, 2] * torch.exp(ls)
            o = sr.render(fm.torch_pose(th, x, y, z, dev))
            L, sil, fb = Evidence.cost(o, M, V, I)
        TH[k] = th.detach().cpu().numpy()
        XYZ[k] = torch.stack([x, y, z], -1).detach().cpu().numpy()
        CO[k] = torch.stack([L, sil, fb], -1).cpu().numpy()
    return TH, XYZ, CO


def motion_score(G, MO, MP, lag=5):
    """Mean abs grey change over un-occluded object pixels between t and t-lag (static object -> ~0.8)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    Pd = np.stack([cv2.dilate((p > 0.3).astype(np.uint8), k) for p in MP])
    Od = np.stack([cv2.dilate((o > 0.5).astype(np.uint8), k) for o in MO])
    F_ = len(G); m = np.full(F_, np.nan)
    for t in range(lag, F_):
        s = t - lag
        w = ((Od[t] | Od[s]) > 0) & (Pd[t] == 0) & (Pd[s] == 0)
        if w.sum() > 200: m[t] = np.mean(np.abs(G[t][w] - G[s][w]))
    ok = ~np.isnan(m); m[~ok] = np.interp(np.nonzero(~ok)[0], np.nonzero(ok)[0], m[ok])
    return m


def joint_refine(sr, E, pm, th0, a0, b0, seg, iters=150, w_th=40.0, w_ab=400.0, chunk=140, dev="cuda", log=None):
    """Joint optimisation of all frames. seg (F,) int: frames with the same seg id >= 0 share one pose (static phase);
    seg == -1 frames have their own pose. Second-difference smoothness on the per-frame sequence.
    A single global table-height offset dh absorbs a global scale mismatch."""
    F_ = len(th0)
    pid = np.zeros(F_, int); nxt = 0; seen = {}
    for t in range(F_):
        if seg[t] >= 0:
            if seg[t] not in seen: seen[seg[t]] = nxt; nxt += 1
            pid[t] = seen[seg[t]]
        else:
            pid[t] = nxt; nxt += 1
    P = nxt
    def init(x):
        v = np.zeros(P)
        for p in range(P): v[p] = np.median(x[pid == p]) if (pid == p).sum() > 1 else x[pid == p][0]
        return torch.tensor(v, dtype=torch.float32, device=dev)
    th = init(np.unwrap(th0)).requires_grad_(); a = init(a0).requires_grad_(); b = init(b0).requires_grad_()
    dh = torch.zeros(1, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{"params": [th], "lr": 0.004}, {"params": [a, b], "lr": 0.002}, {"params": [dh], "lr": 0.001}])
    pid_t = torch.tensor(pid, device=dev)
    up = torch.tensor(pm.up, dtype=torch.float32, device=dev)
    frames_M = [E.frame(i) for i in range(F_)]
    for it in range(iters):
        opt.zero_grad()
        blur = max(2.0 * (1 - it / (0.5 * iters)), 0)
        tot = 0.0
        for c0 in range(0, F_, chunk):
            idx = torch.arange(c0, min(F_, c0 + chunk), device=dev)
            p = pid_t[idx]
            T = pm.torch_pose(th[p], a[p], b[p], dev)
            T = T.clone(); T[:, :3, 3] = T[:, :3, 3] - dh * up  # move the table plane along up
            o = sr.render(T)
            Ls = []
            for j, i in enumerate(idx.tolist()):
                M, V, I = frames_M[i]
                L, _, _ = Evidence.cost(o[j:j + 1], M, V, I, blur=blur)
                Ls.append(L)
            L = torch.cat(Ls).sum()
            L.backward(); tot += L.item()
        # smoothness on the per-frame sequence (shared params give zero differences inside static phases)
        thf, af, bf = th[pid_t], a[pid_t], b[pid_t]
        d2 = lambda x: x[2:] - 2 * x[1:-1] + x[:-2]
        R = w_th * (d2(thf) ** 2).sum() + w_ab * ((d2(af) ** 2).sum() + (d2(bf) ** 2).sum())
        R.backward()
        opt.step()
        if log is not None and (it % 25 == 0 or it == iters - 1):
            log(f"refine it {it} data {tot:.3f} smooth {R.item():.3f} dh {dh.item()*100:.2f}cm")
    thf, af, bf = th[pid_t], a[pid_t], b[pid_t]
    return thf.detach().cpu().numpy(), af.detach().cpu().numpy(), bf.detach().cpu().numpy(), dh.item()


def grid_costs(sr, E, pm, par_init, NG=120, iters=30, dev="cuda", log=None):
    """For every frame and every yaw on a circular grid (NG values), optimise the table-plane position (a,b)
    of the silhouette+front/back cost. Returns (F,NG,5): total, sil, fb, a, b."""
    F_ = len(par_init); ths = np.arange(NG) * 2 * np.pi / NG
    out = np.zeros((F_, NG, 5), np.float32)
    th = torch.tensor(ths, dtype=torch.float32, device=dev)
    for fi in range(F_):
        M, V, I = E.frame(fi)
        a = torch.full((NG,), float(par_init[fi, 1]), device=dev, requires_grad=True)
        b = torch.full((NG,), float(par_init[fi, 2]), device=dev, requires_grad=True)
        opt = torch.optim.Adam([a, b], lr=0.004)
        for it in range(iters):
            o = sr.render(pm.torch_pose(th, a, b, dev))
            L, _, _ = Evidence.cost(o, M, V, I, blur=max(3.0 * (1 - it / (0.6 * iters)), 0))
            opt.zero_grad(); L.sum().backward(); opt.step()
        with torch.no_grad():
            o = sr.render(pm.torch_pose(th, a, b, dev)); L, sil, fb = Evidence.cost(o, M, V, I)
        out[fi] = np.stack([L.cpu().numpy(), sil.cpu().numpy(), fb.cpu().numpy(),
                            a.detach().cpu().numpy(), b.detach().cpu().numpy()], 1)
        if log is not None and fi % 100 == 0: log(f"grid frame {fi}")
    return out


def detect_phases(motion, contact_prob, thr=1.3):
    """Static phase 1: from the start until the object first moves (image motion score > thr).
    Static phase 2: after the last hand contact, from the first frame where the motion score is back below thr.
    Returns seg (F,) with 0 / 1 for the two static phases and -1 for moving frames."""
    F_ = len(motion); seg = -np.ones(F_, int)
    mv = np.nonzero(motion > thr)[0]; mv = mv[mv > 5]
    e1 = mv[0] if len(mv) else F_
    seg[:e1] = 0
    cp = np.nonzero(contact_prob > 0.5)[0]
    if len(cp):
        lc = cp.max(); q = np.nonzero(motion[lc:] < thr)[0]
        if len(q): seg[lc + q[0]:] = 1
    return seg


def subgrid(curve, k):
    """Parabolic refinement of a circular cost curve around index k; returns fractional index."""
    N = len(curve); l, c, r = curve[(k - 1) % N], curve[k], curve[(k + 1) % N]
    den = l - 2 * c + r
    if den <= 1e-9: return float(k)
    return k + float(np.clip(0.5 * (l - r) / den, -0.5, 0.5))


def smooth_with_anchors(y, fixed_mask, fixed_val, lam, w=None):
    """argmin_x sum w (x-y)^2 + lam sum (x[t+1]-2x[t]+x[t-1])^2, with x[fixed]=fixed_val (dense solve, F~500)."""
    F_ = len(y); w = np.ones(F_) if w is None else w
    D = np.zeros((F_ - 2, F_))
    for t in range(F_ - 2): D[t, t:t + 3] = [1, -2, 1]
    A = np.diag(w) + lam * D.T @ D; rhs = w * y
    free = ~fixed_mask
    A_ff = A[np.ix_(free, free)]; rhs_f = rhs[free] - A[np.ix_(free, fixed_mask)] @ fixed_val[fixed_mask]
    x = fixed_val.copy().astype(float); x[free] = np.linalg.solve(A_ff, rhs_f)
    return x


def decode_trajectory(G, seg, sig_move_deg=4.0, sig_static_deg=0.3, lam_th=3.0, lam_ab=100.0, w_data=10.0,
                      prior_th=None, prior_valid=None, prior_w=0.0, prior_sig_deg=10.0, prior_cap_deg=30.0):
    """Global decoding of yaw over time + translation smoothing. G: grid costs (F,NG,5).
    Optional soft prior towards a reference yaw (the baseline) on frames prior_valid: it only matters where the
    image cost curve is flat (ambiguous near-frontal views); capped so that it cannot override clear evidence."""
    F_, NG, _ = G.shape; step = 360.0 / NG
    D = np.minimum(G[:, :, 0], 1.0) * w_data
    if prior_th is not None and prior_w > 0:
        grid = np.arange(NG) * step
        dd = (grid[None] - np.degrees(prior_th)[:, None] + 180) % 360 - 180
        pen = prior_w * np.minimum(dd ** 2, prior_cap_deg ** 2) / prior_sig_deg ** 2
        D = D + pen * prior_valid[:, None]
    Dp = D.copy()
    for s in np.unique(seg[seg >= 0]):  # static phases: pooled evidence (mean cost curve over the phase)
        Dp[seg == s] = D[seg == s].mean(0)
    sig = np.where(seg >= 0, sig_static_deg, sig_move_deg)
    path = viterbi_yaw(Dp, sig, step)
    # sub-grid yaw, unwrapped
    thd = np.array([subgrid(Dp[t], path[t]) for t in range(F_)]) * step
    thd = np.degrees(np.unwrap(np.radians(thd)))
    # translation at the chosen yaw (linear interpolation between neighbouring grid yaws)
    k0 = np.floor(thd / step).astype(int) % NG; k1 = (k0 + 1) % NG; f = (thd / step) - np.floor(thd / step)
    ab = (1 - f)[:, None] * G[np.arange(F_), k0, 3:5] + f[:, None] * G[np.arange(F_), k1, 3:5]
    fixed = seg >= 0
    th_fix = thd.copy(); ab_fix = ab.copy()
    for s in np.unique(seg[seg >= 0]):
        m = seg == s
        th_fix[m] = np.median(thd[m]); ab_fix[m] = np.median(ab[m], 0)
    # robust pre-filter of the moving-phase translations (median of 5) then quadratic smoothing
    from scipy.ndimage import median_filter
    ab_m = np.stack([median_filter(ab[:, k], 5, mode="nearest") for k in range(2)], 1)
    th_s = smooth_with_anchors(thd, fixed, th_fix, lam_th)
    a_s = smooth_with_anchors(ab_m[:, 0], fixed, ab_fix[:, 0], lam_ab)
    b_s = smooth_with_anchors(ab_m[:, 1], fixed, ab_fix[:, 1], lam_ab)
    return np.radians(th_s), a_s, b_s, path, np.radians(thd), ab


def reanchor_depth(T_new, T_base, valid_base, seg, Twc, sigma=30.0):
    """Move each pose along its camera viewing ray (rotation unchanged) so that the camera-frame depth matches
    the baseline's: per static phase the median depth ratio, in the moving phase a heavily smoothed ratio
    (Gaussian sigma frames, computed over plausible baseline frames only), so the shape of the trajectory is ours
    and the overall placement is the baseline's."""
    from scipy.ndimage import gaussian_filter1d
    Tcw = np.linalg.inv(Twc)
    zc = lambda T: np.einsum("ij,fjk->fik", Tcw, T)[:, 2, 3]
    zn, zb = zc(T_new), zc(T_base)
    ratio = zb / zn
    F_ = len(T_new); r = np.ones(F_)
    ok = valid_base.copy()
    idx = np.arange(F_)
    rr = ratio.copy(); rr[~ok] = np.interp(idx[~ok], idx[ok], ratio[ok])
    r = gaussian_filter1d(rr, sigma, mode="nearest")
    for s in np.unique(seg[seg >= 0]):
        m = seg == s
        r[m] = np.median(ratio[m & ok])
    # moving frames: blend so the ratio is continuous with the static phases at the boundaries
    mv = seg < 0
    if mv.any():
        r_mv = gaussian_filter1d(rr, sigma, mode="nearest")
        fixed = ~mv
        r = smooth_with_anchors(np.where(mv, r_mv, r), fixed, r, lam=50.0)
    out = T_new.copy()
    for i in range(F_):
        Tc = Tcw @ T_new[i]
        Tc[:3, 3] *= r[i]
        out[i] = Twc @ Tc
    return out, r
