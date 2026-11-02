"""Shared helpers: data loading, nvdiffrast renderer for the textured object mesh, pose utilities."""
import json
import os

import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot

H, W = 1152, 1536


def load_inputs(inp):
    cam = json.load(open(os.path.join(inp, "camera.json")))
    K = np.array(cam["K"], float)
    Twc = np.array(cam["T_world_cam"], float)
    rec = np.load(os.path.join(inp, "reconstruction.npz"))
    plane = np.array(json.load(open(os.path.join(inp, "ground_plane.json")))["plane"], float)
    return dict(K=K, Twc=Twc, Kp=np.array(cam["K_pipeline"], float), rec=rec, plane=plane)


def load_mesh(inp):
    m = trimesh.load(os.path.join(inp, "object_mesh.glb"), force="scene", process=False).to_geometry()
    return m


class Renderer:
    def __init__(self, mesh, device="cuda", tex_max=2048):
        self.dev = device
        self.glctx = dr.RasterizeCudaContext()
        v = np.asarray(mesh.vertices, np.float32)
        f = np.asarray(mesh.faces, np.int32)
        uv = np.asarray(mesh.visual.uv, np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        tex = mesh.visual.material.baseColorTexture.convert("RGB")
        s = tex_max / max(tex.size)
        if s < 1:
            tex = tex.resize((int(tex.size[0] * s), int(tex.size[1] * s)))
        self.v = torch.tensor(v, device=device)
        self.f = torch.tensor(f, device=device)
        self.uv = torch.tensor(uv, device=device)
        self.tex = torch.tensor(np.asarray(tex, np.float32) / 255.0, device=device)[None].contiguous()
        self.verts_np = v

    def render(self, Tcm, K, crop, res, near=0.05, far=20.0, need_rgb=True):
        """Tcm: (B,4,4) cam<-mesh (torch or np). crop=(x0,y0,w,h) in full-image pixels, res=(rw,rh).
        Returns dict mask (B,rh,rw), depth (B,rh,rw) metric z (0 where empty), rgb (B,rh,rw,3)."""
        if not torch.is_tensor(Tcm):
            Tcm = torch.tensor(np.asarray(Tcm), dtype=torch.float32, device=self.dev)
        Tcm = Tcm.float()
        B = Tcm.shape[0]
        x0, y0, cw, ch = crop
        rw, rh = res
        pc = torch.einsum("bij,nj->bni", Tcm[:, :3, :3], self.v) + Tcm[:, None, :3, 3]
        Z = pc[..., 2].clamp(min=1e-4)
        u = K[0][0] * pc[..., 0] / Z + K[0][2]
        vv = K[1][1] * pc[..., 1] / Z + K[1][2]
        xn = 2 * (u - x0) / cw - 1
        yn = 2 * (vv - y0) / ch - 1
        zn = (Z - near) / (far - near) * 2 - 1
        clip = torch.stack([xn * Z, yn * Z, zn * Z, Z], -1).contiguous()
        rast, _ = dr.rasterize(self.glctx, clip, self.f, (rh, rw))
        mask = rast[..., 3] > 0
        out = {"mask": mask}
        d, _ = dr.interpolate(Z[..., None].contiguous(), rast, self.f)
        out["depth"] = d[..., 0] * mask
        if need_rgb:
            uvi, _ = dr.interpolate(self.uv[None].expand(B, -1, -1).contiguous(), rast, self.f)
            col = dr.texture(self.tex.expand(B, -1, -1, -1).contiguous(), uvi, filter_mode="linear")
            out["rgb"] = col * mask[..., None]
        out["rast"] = rast
        return out


def rot_angle_deg(Ra, Rb):
    R = Ra @ np.swapaxes(Rb, -1, -2)
    c = (np.trace(R, axis1=-2, axis2=-1) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def add_dist(T1, T2, pts):
    """mean distance between pts under poses T1 and T2 (broadcast over leading dims)."""
    p1 = np.einsum("...ij,nj->...ni", T1[..., :3, :3], pts) + T1[..., None, :3, 3]
    p2 = np.einsum("...ij,nj->...ni", T2[..., :3, :3], pts) + T2[..., None, :3, 3]
    return np.linalg.norm(p1 - p2, axis=-1).mean(-1)


# ---------------------------------------------------------------------------------------------
# Frame data (crops) and render-and-compare


def read_video(path, frames_needed=None):
    import cv2
    cap = cv2.VideoCapture(path)
    out = {}
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if frames_needed is None or i in frames_needed:
            out[i] = f[:, :, ::-1].copy()
        i += 1
    return out


def make_crop(objmask, center_uv=None, min_side=90, pad=1.7):
    """square crop around the object mask bbox (or a given centre)."""
    ys, xs = np.nonzero(objmask)
    if len(xs) > 30:
        u = (xs.min() + xs.max()) / 2
        v = (ys.min() + ys.max()) / 2
        side = max(xs.max() - xs.min(), ys.max() - ys.min()) * pad
    else:
        u, v = center_uv
        side = 0
    side = max(side, min_side)
    return (float(u - side / 2), float(v - side / 2), float(side), float(side))


def crop_resize(img, crop, res, interp=None):
    """crop (x0,y0,w,h) from img with zero padding, resized to res x res."""
    import cv2
    x0, y0, cw, ch = crop
    sx = res / cw
    sy = res / ch
    M = np.array([[sx, 0, -x0 * sx], [0, sy, -y0 * sy]], np.float32)
    if interp is None:
        interp = cv2.INTER_AREA if sx < 1 else cv2.INTER_LINEAR
    if img.dtype == bool:
        o = cv2.warpAffine(img.astype(np.float32), M, (res, res), flags=cv2.INTER_LINEAR)
        return o
    return cv2.warpAffine(img, M, (res, res), flags=cv2.INTER_LINEAR)


class FrameData:
    """GPU tensors for a set of frames: image crops, object / person mask crops."""

    def __init__(self, frames, imgs, objm, perm, crops, res, device="cuda"):
        self.frames = list(frames)
        self.crops = crops
        self.res = res
        im = np.stack([crop_resize(imgs[f], c, res) for f, c in zip(frames, crops)]).astype(np.float32) / 255
        om = np.stack([crop_resize(objm[f], c, res) for f, c in zip(frames, crops)])
        pm = np.stack([crop_resize(perm[f], c, res) for f, c in zip(frames, crops)])
        self.img = torch.tensor(im, device=device)
        self.obj = torch.tensor(om, device=device)
        self.per = torch.tensor(pm, device=device)
        # person pixels that are not object pixels are treated as occluders (ignored)
        self.valid = (1 - torch.clamp(self.per - self.obj, 0, 1))


def gauss_blur(x, sigma):
    """x (B,C,H,W)"""
    if sigma <= 0.3:
        return x
    k = int(2 * round(3 * sigma) + 1)
    t = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-0.5 * (t / sigma) ** 2)
    g = g / g.sum()
    C = x.shape[1]
    x = torch.nn.functional.conv2d(x, g.view(1, 1, 1, k).expand(C, 1, 1, k), padding=(0, k // 2), groups=C)
    x = torch.nn.functional.conv2d(x, g.view(1, 1, k, 1).expand(C, 1, k, 1), padding=(k // 2, 0), groups=C)
    return x


def render_compare(R, Tc, fd, idx, K, sigma=1.0, w_sil=1.0, w_ph=1.0, photo="ncc", lncc_win=3.0):
    """Tc (B,4,4) torch cam<-mesh, idx (B,) frame indices into fd. Returns loss per item (B,) and parts."""
    B = Tc.shape[0]
    res = fd.res
    losses = []
    out_sil, out_ph = [], []
    # render each crop: crops may differ per item; do it in one batch by per-item projection
    x0 = torch.tensor([fd.crops[i][0] for i in idx], device=Tc.device)
    y0 = torch.tensor([fd.crops[i][1] for i in idx], device=Tc.device)
    cs = torch.tensor([fd.crops[i][2] for i in idx], device=Tc.device)
    pc = torch.einsum("bij,nj->bni", Tc[:, :3, :3], R.v) + Tc[:, None, :3, 3]
    Z = pc[..., 2].clamp(min=1e-3)
    u = K[0][0] * pc[..., 0] / Z + K[0][2]
    v = K[1][1] * pc[..., 1] / Z + K[1][2]
    xn = 2 * (u - x0[:, None]) / cs[:, None] - 1
    yn = 2 * (v - y0[:, None]) / cs[:, None] - 1
    near, far = 0.05, 20.0
    zn = (Z - near) / (far - near) * 2 - 1
    clip = torch.stack([xn * Z, yn * Z, zn * Z, Z], -1).contiguous()
    rast, _ = dr.rasterize(R.glctx, clip, R.f, (res, res))
    uvi, _ = dr.interpolate(R.uv[None].expand(B, -1, -1).contiguous(), rast, R.f)
    col = dr.texture(R.tex.expand(B, -1, -1, -1).contiguous(), uvi, filter_mode="linear")
    m = (rast[..., 3:4] > 0).float()
    rgba = torch.cat([col * m, m], -1)
    rgba = dr.antialias(rgba.contiguous(), rast, clip, R.f)
    A = rgba[..., 3]
    col = rgba[..., :3]
    I = fd.img[idx]
    O = fd.obj[idx]
    V = fd.valid[idx]
    # silhouette term
    sil = ((A - O) ** 2 * V).sum((1, 2)) / (V.sum((1, 2)) + 1)
    # normalise sil by object size so that it is ~1-IoU like
    sil = ((A - O) ** 2 * V).sum((1, 2)) / ((O * V).sum((1, 2)) + (A * V).sum((1, 2)) + 10)
    # photometric: per-channel gain/bias fit inside overlap
    with torch.no_grad():
        Mo = (A > 0.5).float() * (O > 0.5).float() * V
        n = Mo.sum((1, 2))[:, None] + 1
        rm = (col * Mo[..., None]).sum((1, 2)) / n
        im = (I * Mo[..., None]).sum((1, 2)) / n
        rv = (((col - rm[:, None, None]) ** 2) * Mo[..., None]).sum((1, 2)) / n
        cv = (((col - rm[:, None, None]) * (I - im[:, None, None])) * Mo[..., None]).sum((1, 2)) / n
        g = (cv + 0.01 * 1.0) / (rv + 0.01)
        g = g.clamp(0.3, 3.0)
        b = im - g * rm
    colc = col * g[:, None, None] + b[:, None, None] * A[..., None]
    C = colc + (1 - A[..., None]) * I
    Cb = gauss_blur(C.permute(0, 3, 1, 2), sigma)
    Ib = gauss_blur(I.permute(0, 3, 1, 2), sigma)
    roi = gauss_blur(torch.maximum(A, O)[:, None], 3.0)[:, 0].clamp(0, 1) * V
    roi = (roi > 0.02).float() * V
    if photo == "l1":
        ph = ((Cb - Ib).abs().sum(1) * roi).sum((1, 2)) / (roi.sum((1, 2)) + 1)
    else:
        # local normalised cross-correlation of colour (invariant to per-face shading)
        ws = lncc_win
        mu_c = gauss_blur(Cb, ws)
        mu_i = gauss_blur(Ib, ws)
        vc = gauss_blur(Cb * Cb, ws) - mu_c ** 2
        vi = gauss_blur(Ib * Ib, ws) - mu_i ** 2
        cc = gauss_blur(Cb * Ib, ws) - mu_c * mu_i
        ncc = cc / torch.sqrt(vc.clamp(min=0) * vi.clamp(min=0) + 1e-5)
        ncc = ncc.mean(1)
        reg = (A.detach() > 0.5).float() * V
        ph = ((1 - ncc) * reg).sum((1, 2)) / (reg.sum((1, 2)) + 1)
    loss = w_sil * sil + w_ph * ph
    return loss, sil, ph, A


def so3_exp(w):
    th = w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    k = w / th
    Kx = torch.zeros(w.shape[:-1] + (3, 3), device=w.device, dtype=w.dtype)
    Kx[..., 0, 1] = -k[..., 2]
    Kx[..., 0, 2] = k[..., 1]
    Kx[..., 1, 0] = k[..., 2]
    Kx[..., 1, 2] = -k[..., 0]
    Kx[..., 2, 0] = -k[..., 1]
    Kx[..., 2, 1] = k[..., 0]
    th = th[..., None]
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(Kx)
    return I + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


def apply_delta(T0, w, t):
    """rotate about the object's own origin (in camera axes), then translate (camera frame)."""
    Rm = so3_exp(w) @ T0[:, :3, :3]
    T = torch.zeros_like(T0)
    T[:, :3, :3] = Rm
    T[:, :3, 3] = T0[:, :3, 3] + t
    T[:, 3, 3] = 1
    return T


def refine_poses(R, fd, K, T0, groups, iters=150, sigmas=(4.0, 2.0, 1.0), lr_r=0.01, lr_t=0.003,
                 w_sil=1.0, w_ph=0.5, fix_depth=False, photo="ncc", rot_axis=None, plane_n=None, depth_lr_scale=1.0):
    """Render-and-compare refinement of H pose hypotheses.
    T0 (H,4,4) torch cam<-mesh init poses. groups: list (len H) of lists of fd-indices that the
    hypothesis must explain jointly (one static pose for all of them).
    fix_depth: translation only perpendicular to the viewing ray.
    rot_axis (H,3) camera-frame unit axis: rotation restricted to this axis (planar motion).
    plane_n (H,3): translation restricted to the plane perpendicular to plane_n.
    Returns refined T (H,4,4), loss (H,), silhouette (H,), photometric (H,)."""
    dev = T0.device
    Hn = T0.shape[0]
    item_h = torch.tensor([h for h in range(Hn) for _ in groups[h]], device=dev)
    item_f = [f for h in range(Hn) for f in groups[h]]
    nper = torch.tensor([len(g) for g in groups], device=dev, dtype=torch.float32)
    w = torch.zeros(Hn, 3, device=dev, requires_grad=True)
    t = torch.zeros(Hn, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{"params": [w], "lr": lr_r}, {"params": [t], "lr": lr_t}])
    ray = T0[:, :3, 3] / T0[:, :3, 3].norm(dim=-1, keepdim=True)

    def cur_T():
        ww = w
        if rot_axis is not None:
            ww = (w * rot_axis).sum(-1, keepdim=True) * rot_axis
        tt = t
        if fix_depth:
            tt = t - (t * ray).sum(-1, keepdim=True) * ray
        elif depth_lr_scale != 1.0:
            along = (t * ray).sum(-1, keepdim=True)
            tt = t - along * ray + depth_lr_scale * along * ray
        if plane_n is not None:
            tt = tt - (tt * plane_n).sum(-1, keepdim=True) * plane_n
        return apply_delta(T0, ww, tt)

    per_stage = max(1, iters // len(sigmas))
    nstep = per_stage * len(sigmas)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * k / nstep)))
    for s_i, sig in enumerate(sigmas):
        for it in range(per_stage):
            T = cur_T()
            loss, _, _, _ = render_compare(R, T[item_h], fd, item_f, K, sigma=sig, w_sil=w_sil, w_ph=w_ph,
                                           photo=photo)
            lh = torch.zeros(Hn, device=dev).index_add(0, item_h, loss) / nper
            opt.zero_grad()
            lh.sum().backward()
            opt.step()
            sched.step()
    with torch.no_grad():
        T = cur_T()
        loss, sil, ph, _ = render_compare(R, T[item_h], fd, item_f, K, sigma=sigmas[-1], w_sil=w_sil,
                                          w_ph=w_ph, photo=photo)
        lh = torch.zeros(Hn, device=dev).index_add(0, item_h, loss) / nper
        sh = torch.zeros(Hn, device=dev).index_add(0, item_h, sil) / nper
        ph_ = torch.zeros(Hn, device=dev).index_add(0, item_h, ph) / nper
    return T.detach(), lh, sh, ph_


def score_poses(R, fd, K, T, idx, sigma=1.0, w_sil=1.0, w_ph=0.5, chunk=64):
    """loss of poses T (B,4,4) on fd frames idx (B,), no optimisation."""
    out = []
    with torch.no_grad():
        for k in range(0, T.shape[0], chunk):
            l, s, p, _ = render_compare(R, T[k:k + chunk], fd, idx[k:k + chunk], K, sigma=sigma, w_sil=w_sil, w_ph=w_ph)
            out.append(torch.stack([l, s, p], -1))
    return torch.cat(out)


# ---------------------------------------------------------------------------------------------
# Pose helpers


def move_to_depth(T, z):
    """slide pose(s) along the viewing ray of the object origin so that camera z == z."""
    T = T.copy()
    s = z / T[..., 2, 3]
    T[..., :3, 3] = T[..., :3, 3] * np.asarray(s)[..., None]
    return T


def kabsch(P, Q):
    """rigid T (4,4) minimising |T P - Q|, P,Q (N,3)."""
    mp, mq = P.mean(0), Q.mean(0)
    Hm = (P - mp).T @ (Q - mq)
    U, S, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Dm = np.diag([1, 1, d])
    Rm = Vt.T @ Dm @ U.T
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = mq - Rm @ mp
    return T


def multi_track(R, fd, K, tracks, iters=25, sigmas=(2.0, 1.0), hand_delta=None, planar_n=None, z_sched=None):
    """Frame-to-frame render-and-compare tracking of several tracks in one batch.
    tracks: list of dicts {frames: [fd indices in tracking order], T0: start pose (4,4) cam<-mesh,
    frame_ids: [video frame ids], planar: bool}
    hand_delta: dict (f_prev, f) -> (4,4) camera-frame rigid motion of the carrying hand (optional)
    z_sched: dict video frame -> camera depth to put the pose at before refinement (optional; then depth fixed)
    Returns per track list of (T (4,4), loss, sil, photo) in tracking order."""
    dev = "cuda"
    nT = len(tracks)
    L = max(len(t["frames"]) for t in tracks)
    prev = [t["T0"].copy() for t in tracks]
    prev2 = [t["T0"].copy() for t in tracks]
    out = [[] for _ in tracks]
    for k in range(L):
        T0s, groups, owner, axes, pns = [], [], [], [], []
        for ti, tr in enumerate(tracks):
            if k >= len(tr["frames"]):
                continue
            j = tr["frames"][k]
            f = tr["frame_ids"][k]
            fprev = tr["frame_ids"][k - 1] if k > 0 else None
            hyps = [prev[ti], (prev[ti] @ np.linalg.inv(prev2[ti])) @ prev[ti]]
            if hand_delta is not None and fprev is not None and (fprev, f) in hand_delta:
                hd = hand_delta[(fprev, f)]
                hyps.append(hd @ prev[ti])
            for h in hyps:
                h = h.copy()
                if z_sched is not None:
                    h = move_to_depth(h, z_sched[f])
                T0s.append(h)
                groups.append([j])
                owner.append(ti)
                if tr.get("planar"):
                    axes.append(planar_n)
                    pns.append(planar_n)
                else:
                    axes.append(None)
                    pns.append(None)
        T0t = torch.tensor(np.stack(T0s), dtype=torch.float32, device=dev)
        planar_any = any(a is not None for a in axes)
        if planar_any:
            ax = torch.tensor(np.stack([a if a is not None else np.zeros(3) for a in axes]), dtype=torch.float32,
                              device=dev)
            # non-planar items: rot_axis None -> emulate with full rotation by separate call
        if planar_any and not all(a is not None for a in axes):
            raise ValueError("mixing planar and 6dof tracks in one batch is not supported")
        T, l, s, p = refine_poses(R, fd, K, T0t, groups, iters=iters, sigmas=sigmas,
                                  rot_axis=ax if planar_any else None, plane_n=ax if planar_any else None,
                                  fix_depth=(z_sched is not None) and not planar_any)
        T = T.cpu().numpy().astype(np.float64)
        l = l.cpu().numpy()
        s = s.cpu().numpy()
        p = p.cpu().numpy()
        owner = np.array(owner)
        for ti in range(nT):
            idx = np.nonzero(owner == ti)[0]
            if len(idx) == 0:
                continue
            b = idx[np.argmin(l[idx])]
            out[ti].append((T[b], float(l[b]), float(s[b]), float(p[b])))
            prev2[ti] = prev[ti]
            prev[ti] = T[b]
    return out
