#!/usr/bin/env python
"""Trajectory-level 6-DoF pose recovery of the watering can (episode 24).

Method (see REPORT.md for the reasoning and validation):
  1. Per-frame evidence: a sub-pixel "green-ness" map of the object computed from the video (Lab a* channel;
     the can is saturated green, floor/person are grey/black) in a 160x160 window around the SAM2 object mask.
     Occluders are only person pixels near the projected hand keypoints of the reconstructed human (the legs
     are BEHIND the can, so other person pixels are treated as background).
  2. Phase segmentation from the video: static phase A = leading frames whose green map still matches frame 0
     (IoU >= 0.93), static phase B = trailing frames matching the last frame; in between the can is moved.
  3. Static phases: ONE pose per phase, fitted by differentiable silhouette rendering (nvdiffrast) pooled over
     all frames of the phase.
  4. Moving phase: joint optimisation of all poses (silhouette term + 2nd-order SE(3) smoothness), anchored with
     zero velocity to the static poses at both ends, initialised from the baseline (which is on the correct
     branch of the near mirror-symmetric silhouette ambiguity).
  5. Depth along the viewing ray: the silhouette is ~9% larger than the metric mesh at the floor-consistent depth
     (a size/calibration mismatch, consistent in both static phases), so absolute depth is NOT taken from the
     silhouette. Static phases: mean of (a) floor contact (lowest mesh point 1 cm below the released plane, the
     level of the true frame-0 human's soles) and (b) the baseline's median depth over the phase. Moving phase:
     robustly smoothed baseline depth (calibrated by the pipeline against human + floor), joined continuously
     to the static depths. Rotation and the image position of the object come from the image fit.
"""
import argparse, json, time, os
import numpy as np
import torch, torch.nn.functional as F
import cv2, h5py, trimesh
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import brentq
import scipy.sparse as sp
import scipy.sparse.linalg as spla

WIN = 160          # crop window (px) around the object
UP = 2             # render / compare at 2x crop resolution
STATIC_IOU = 0.93  # green-map IoU to first/last frame that defines the static phases
FLOOR_OFFSET = -0.01  # resting level of the can bottom w.r.t. released plane: true frame-0 human soles reach
                      # -1..-2.5 cm below the plane (checked), so the physical floor is ~1 cm below it.
HAND_DILATE = 25   # px; occluder region = person pixels within the dilated convex hull of hand keypoints


def seed_all(s=0):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


# ----------------------------------------------------------------------------------------------- geometry utils
def so3_exp(w):
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    k = w / th
    Kx = torch.zeros(w.shape[:-1] + (3, 3), device=w.device)
    Kx[..., 0, 1] = -k[..., 2]; Kx[..., 0, 2] = k[..., 1]; Kx[..., 1, 0] = k[..., 2]
    Kx[..., 1, 2] = -k[..., 0]; Kx[..., 2, 0] = -k[..., 1]; Kx[..., 2, 1] = k[..., 0]
    th = th[..., None]
    I = torch.eye(3, device=w.device).expand_as(Kx)
    return I + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx


def so3_log_batch(Rm):
    cos = ((Rm[:, 0, 0] + Rm[:, 1, 1] + Rm[:, 2, 2] - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    th = torch.acos(cos)
    v = torch.stack([Rm[:, 2, 1] - Rm[:, 1, 2], Rm[:, 0, 2] - Rm[:, 2, 0], Rm[:, 1, 0] - Rm[:, 0, 1]], -1)
    return v * (th / (2 * torch.sin(th).clamp_min(1e-6)))[:, None]


def apply_delta(T0, w, p):
    """rotate about the object origin (camera frame) by exp(w); p = (du, dv, dlogz) of the origin's ray/depth"""
    T = T0.clone()
    T[:, :3, :3] = so3_exp(w) @ T0[:, :3, :3]
    z0 = T0[:, 2, 3]; u0 = T0[:, 0, 3] / z0; v0 = T0[:, 1, 3] / z0
    z = z0 * torch.exp(p[:, 2]); u = u0 + p[:, 0]; v = v0 + p[:, 1]
    T[:, :3, 3] = torch.stack([u * z, v * z, z], -1)
    return T


def mean_pose(Ts):
    T = np.eye(4)
    T[:3, :3] = Rot.from_matrix(Ts[:, :3, :3]).mean().as_matrix()
    T[:3, 3] = np.median(Ts[:, :3, 3], 0)
    return T


def rot_angle(A, B):
    return np.degrees(Rot.from_matrix(A[:3, :3].T @ B[:3, :3]).magnitude())


# ----------------------------------------------------------------------------------------------- mesh / renderer
def decimated_mesh(mesh, n_faces=8000):
    import open3d as o3d
    mm = mesh.copy(); mm.merge_vertices(merge_tex=True, merge_norm=True)
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(mm.vertices)),
                                   o3d.utility.Vector3iVector(np.asarray(mm.faces)))
    om = om.simplify_quadric_decimation(n_faces)
    return np.asarray(om.vertices), np.asarray(om.triangles)


class SilRenderer:
    def __init__(self, V, Fc):
        self.glctx = dr.RasterizeCudaContext()
        self.v = torch.tensor(V, dtype=torch.float32, device='cuda')
        self.f = torch.tensor(Fc, dtype=torch.int32, device='cuda')

    def sil(self, Tcam, K, X0, blur=1.0):
        """soft silhouettes (B, WIN*UP, WIN*UP) of the crops with top-left corners X0 (B,2)"""
        H = W = WIN * UP
        vc = self.v @ Tcam[:, :3, :3].transpose(1, 2) + Tcam[:, None, :3, 3]
        fx, fy = K[0, 0] * UP, K[1, 1] * UP
        cx = torch.tensor((K[0, 2] + 0.5 - X0[:, 0]) * UP - 0.5, device='cuda', dtype=torch.float32)[:, None]
        cy = torch.tensor((K[1, 2] + 0.5 - X0[:, 1]) * UP - 0.5, device='cuda', dtype=torch.float32)[:, None]
        n, f = 0.05, 20.
        x, y, z = vc[..., 0], vc[..., 1], vc[..., 2]
        pos = torch.stack([2 * fx * x / W + (2 * (cx + 0.5) / W - 1) * z,
                           2 * fy * y / H + (2 * (cy + 0.5) / H - 1) * z,
                           z * (f + n) / (f - n) - 2 * f * n / (f - n), z], -1).contiguous()
        rast, _ = dr.rasterize(self.glctx, pos, self.f, resolution=[H, W])
        m = (rast[..., 3:4] > 0).float()
        s = dr.antialias(m, rast, pos, self.f)[..., 0]
        if blur > 0:   # the video is slightly blurred; symmetric blur does not move the 50% contour
            k = int(3 * blur * UP) * 2 + 1
            xs = torch.arange(k, device='cuda') - k // 2
            g = torch.exp(-xs ** 2 / (2 * (blur * UP) ** 2)); g = g / g.sum()
            s = F.conv2d(s[:, None], g.view(1, 1, 1, k), padding=(0, k // 2))
            s = F.conv2d(s, g.view(1, 1, k, 1), padding=(k // 2, 0))[:, 0]
        return s


# ----------------------------------------------------------------------------------------------- observations
def green_prob(img_rgb):
    a = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)[..., 1] - 128
    return np.clip((-a - 5) / 16, 0, 1)   # object a* ~ -27..-31, floor/person ~ -2


def build_observations(inp, K, Tcw, kp_world):
    mf = h5py.File(os.path.join(inp, 'masks.h5'), 'r')
    F_ = mf['object'].shape[0]
    cap = cv2.VideoCapture(os.path.join(inp, 'video.mp4'))
    G = np.zeros((F_, WIN, WIN), np.float32); Wt = np.ones((F_, WIN, WIN), np.float32)
    X0 = np.zeros((F_, 2), int)
    last = (768, 576)
    for i in range(F_):
        ok, fr = cap.read()
        assert ok, f'video ended at frame {i}'
        om = mf['object'][i]
        ys, xs = np.nonzero(om)
        cx, cy = (int(np.median(xs)), int(np.median(ys))) if len(ys) else last
        last = (cx, cy)
        y0 = int(np.clip(cy - WIN // 2, 0, fr.shape[0] - WIN)); x0 = int(np.clip(cx - WIN // 2, 0, fr.shape[1] - WIN))
        X0[i] = (x0, y0)
        img = fr[y0:y0 + WIN, x0:x0 + WIN, ::-1].copy()
        g = green_prob(img); G[i] = g
        pm = mf['person'][i, y0:y0 + WIN, x0:x0 + WIN]
        hand = np.zeros((WIN, WIN), np.uint8)
        for js in (range(21, 42), range(42, 63)):
            P = kp_world[i, list(js)] @ Tcw[:3, :3].T + Tcw[:3, 3]
            uv = P @ K.T; uv = uv[:, :2] / uv[:, 2:] - np.array([x0, y0])
            cv2.fillConvexPoly(hand, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 1)
        hand = cv2.dilate(hand, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (HAND_DILATE, HAND_DILATE))) > 0
        occ = pm & (g < 0.5) & hand
        occ = cv2.dilate(occ.astype(np.uint8), np.ones((3, 3))) > 0
        Wt[i] = (~occ).astype(np.float32)
    return G, Wt, X0


class Obs:
    def __init__(self, G, Wt, X0, idx):
        self.X0 = X0[idx]
        g = torch.tensor(G[idx], device='cuda'); w = torch.tensor(Wt[idx], device='cuda')
        self.G = F.interpolate(g[:, None], scale_factor=UP, mode='bilinear', align_corners=False)[:, 0]
        self.W = F.interpolate(w[:, None], scale_factor=UP, mode='nearest')[:, 0]


def sil_loss(R, Tc, K, o):
    s = R.sil(Tc, K, o.X0)
    return (((s - o.G) ** 2) * o.W).sum((1, 2)) / UP ** 2


# ----------------------------------------------------------------------------------------------- fitting
def fit_shared(R, K, o, Tinit, iters=500, lr_w=0.02, lr_p=0.01):
    T0 = torch.as_tensor(Tinit, dtype=torch.float32, device='cuda')[None]; B = len(o.X0)
    w = torch.zeros(1, 3, device='cuda', requires_grad=True); p = torch.zeros(1, 3, device='cuda', requires_grad=True)
    opt = torch.optim.Adam([{'params': [w], 'lr': lr_w}, {'params': [p], 'lr': lr_p}])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    for _ in range(iters):
        L = sil_loss(R, apply_delta(T0, w, p).expand(B, 4, 4), K, o).mean()
        opt.zero_grad(); L.backward(); opt.step(); sch.step()
    with torch.no_grad():
        Tc = apply_delta(T0, w, p); L = sil_loss(R, Tc.expand(B, 4, 4), K, o)
    return Tc[0].cpu().numpy().astype(np.float64), L.cpu().numpy()


def fit_traj(R, K, G, Wt, X0, idx, Tinit, A0, A1, iters=400, lr_w=0.01, lr_p=0.005,
             lam_r=300., lam_t=1e5, lam_z=1e4, chunk=128):
    T0 = torch.as_tensor(Tinit, dtype=torch.float32, device='cuda'); B = T0.shape[0]
    w = torch.zeros(B, 3, device='cuda', requires_grad=True); p = torch.zeros(B, 3, device='cuda', requires_grad=True)
    opt = torch.optim.Adam([{'params': [w], 'lr': lr_w}, {'params': [p], 'lr': lr_p}])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    chunks = [(list(range(s, min(s + chunk, B))), Obs(G, Wt, X0, idx[s:s + chunk])) for s in range(0, B, chunk)]
    A0 = torch.as_tensor(A0, dtype=torch.float32, device='cuda')[None]
    A1 = torch.as_tensor(A1, dtype=torch.float32, device='cuda')[None]
    for it in range(iters):
        opt.zero_grad()
        for sel, o in chunks:
            sil_loss(R, apply_delta(T0[sel], w[sel], p[sel]), K, o).sum().backward()
        S = torch.cat([A0, A0, apply_delta(T0, w, p), A1, A1], 0)   # duplicated anchors => zero velocity
        Rr = S[:, :3, :3]
        relc = (Rr[:-1] @ so3_log_batch(Rr[:-1].transpose(1, 2) @ Rr[1:])[..., None])[..., 0]
        d2r = relc[1:] - relc[:-1]
        tt = S[:, :3, 3]; zz = tt[:, 2:].detach()
        xyz = torch.cat([tt[:, :2] / tt[:, 2:] * zz, torch.log(tt[:, 2:]) * zz], 1)
        d2t = xyz[2:] - 2 * xyz[1:-1] + xyz[:-2]
        Ls = lam_r * (d2r ** 2).sum() + lam_t * (d2t[:, :2] ** 2).sum() + lam_z * (d2t[:, 2] ** 2).sum()
        Ls.backward(); opt.step(); sch.step()
    with torch.no_grad():
        Tc = apply_delta(T0, w, p)
        L = torch.cat([sil_loss(R, Tc[sel], K, o) for sel, o in chunks])
    return Tc.cpu().numpy().astype(np.float64), L.cpu().numpy()


def smooth_depth(z_obs, fixed, lam=2000., huber=0.02, iters=10):
    """robust 1-D smoothing: min sum rho(z - z_obs) + lam * sum (d2 z)^2, with z fixed where `fixed` is not nan"""
    n = len(z_obs)
    D2 = sp.diags([np.ones(n - 2), -2 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2], shape=(n - 2, n))
    free = np.isnan(fixed)
    z = np.where(free, z_obs, fixed)
    for _ in range(iters):
        r = z - z_obs
        wdat = np.where(np.abs(r) < huber, 1.0, huber / np.maximum(np.abs(r), 1e-9)) * free
        Big = 1e6 * (~free)
        A = sp.diags(wdat + Big) + lam * (D2.T @ D2)
        b = wdat * z_obs + Big * np.nan_to_num(fixed)
        z = spla.spsolve(A.tocsc(), b)
    return z


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='/work/inputs')
    ap.add_argument('--out', default='/work/outputs/object_poses_world.npz')
    args = ap.parse_args()
    seed_all(0)
    t_start = time.time()
    inp = args.inputs
    cam = json.load(open(os.path.join(inp, 'camera.json')))
    K = np.array(cam['K']); Twc = np.array(cam['T_world_cam']); Tcw = np.linalg.inv(Twc)
    gp = json.load(open(os.path.join(inp, 'ground_plane.json')))['plane']; n_pl = np.array(gp[:3]); d_pl = gp[3]
    rec = np.load(os.path.join(inp, 'reconstruction.npz'))
    Tw_base = rec['T_world_mesh'].astype(np.float64)
    Fn = len(Tw_base)
    Tb = np.array([Tcw @ x for x in Tw_base])          # baseline in camera frame (K of camera.json)
    mesh = trimesh.load(os.path.join(inp, 'object_mesh.glb'), force='scene', process=False).to_geometry()
    Vfull = np.asarray(mesh.vertices)

    print('[1] observations ...', flush=True)
    G, Wt, X0 = build_observations(inp, K, Tcw, rec['human_keypoints_world'])
    print(f'    done {time.time() - t_start:.0f}s', flush=True)

    # ---------------- phase segmentation (green-map IoU in image coordinates to first / last frame)
    def iou_to(ref):
        out = np.zeros(Fn)
        for i in range(Fn):
            a = np.zeros((1152 + WIN, 1536 + WIN), bool); b = np.zeros_like(a); w = np.ones(a.shape, np.float32)
            x, y = X0[ref]; a[y:y + WIN, x:x + WIN] = G[ref] > 0.5
            x, y = X0[i]; b[y:y + WIN, x:x + WIN] = G[i] > 0.5; w[y:y + WIN, x:x + WIN] = Wt[i]
            out[i] = ((a & b) * w).sum() / max(((a | b) * w).sum(), 1)
        return np.convolve(out, np.ones(5) / 5, mode='same')
    iou0, iou1 = iou_to(0), iou_to(Fn - 1)
    a_end = int(np.argmax(iou0[2:-2] < STATIC_IOU)) + 2           # first frame of motion
    b_start = Fn - 2 - int(np.argmax(iou1[::-1][2:-2] < STATIC_IOU))  # first frame of final rest
    print(f'[2] phases: static A [0,{a_end}), moving [{a_end},{b_start}), static B [{b_start},{Fn})', flush=True)

    R = SilRenderer(*decimated_mesh(mesh))

    # ---------------- static phases: one pose each
    def floor_min(Tc):
        Tw = Twc @ Tc
        return ((Vfull[::10] @ Tw[:3, :3].T + Tw[:3, 3]) @ n_pl + d_pl).min()
    static = {}
    for name, (a, b) in {'A': (0, a_end), 'B': (b_start, Fn)}.items():
        idx = list(range(a, b, 3))
        Tsil, L = fit_shared(R, K, Obs(G, Wt, X0, idx), mean_pose(Tb[a:b]))
        lam_floor = brentq(lambda l: floor_min(np.block([[Tsil[:3, :3], l * Tsil[:3, 3:]], [0, 0, 0, 1]])) - FLOOR_OFFSET, 0.7, 1.6)
        z_floor = lam_floor * Tsil[2, 3]; z_base = np.median(Tb[a:b, 2, 3])
        z_fin = 0.5 * (z_floor + z_base)
        T = Tsil.copy(); T[:3, 3] *= z_fin / Tsil[2, 3]
        static[name] = dict(T=T, Tsil=Tsil, kappa=z_fin / Tsil[2, 3], z_floor=z_floor, z_base=z_base, loss=L.mean())
        print(f'    static {name}: sil-depth {Tsil[2,3]:.3f}  floor-depth {z_floor:.3f}  baseline-depth {z_base:.3f}'
              f'  -> {z_fin:.3f}  rot change vs baseline mean {rot_angle(mean_pose(Tb[a:b]), T):.1f} deg', flush=True)
    kappa = 0.5 * (static['A']['kappa'] + static['B']['kappa'])
    print(f'    silhouette->world depth factor kappa = {kappa:.3f}', flush=True)

    # ---------------- moving phase: joint trajectory (rotation + image position), silhouette scale
    print('[3] trajectory fit ...', flush=True)
    idx = list(range(a_end, b_start))
    init = Tb[a_end:b_start].copy(); init[:, :3, 3] /= kappa
    Tmov, Lmov = fit_traj(R, K, G, Wt, X0, idx, init, static['A']['Tsil'], static['B']['Tsil'])

    # ---------------- depth: static constants, moving = robustly smoothed baseline depth, continuous at ends
    fixed = np.full(Fn, np.nan)
    fixed[:a_end] = static['A']['T'][2, 3]; fixed[b_start:] = static['B']['T'][2, 3]
    z_fin = smooth_depth(Tb[:, 2, 3], fixed)

    Tc_out = np.zeros((Fn, 4, 4))
    Tc_out[:a_end] = static['A']['T']; Tc_out[b_start:] = static['B']['T']
    for k, i in enumerate(idx):
        T = Tmov[k].copy(); T[:3, 3] *= z_fin[i] / T[2, 3]     # keep ray direction, set depth
        Tc_out[i] = T
    Tw_out = np.array([Twc @ T for T in Tc_out]).astype(np.float64)
    assert np.isfinite(Tw_out).all() and Tw_out.shape == (Fn, 4, 4)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, T_world_mesh=Tw_out)
    np.savez(os.path.join(os.path.dirname(os.path.abspath(args.out)), 'solve_diagnostics.npz'),
             a_end=a_end, b_start=b_start, iou0=iou0, iou1=iou1, kappa=kappa, z_final=z_fin, sil_loss_moving=Lmov,
             **{f'static_{k}_{kk}': v for k, d in static.items() for kk, v in d.items()})
    print(f'wrote {args.out}  total {time.time() - t_start:.0f}s', flush=True)


if __name__ == '__main__':
    main()
