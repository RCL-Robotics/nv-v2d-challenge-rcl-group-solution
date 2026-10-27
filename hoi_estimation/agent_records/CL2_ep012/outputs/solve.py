#!/usr/bin/env python
"""Trajectory-level re-estimation of the object pose for one HOI video.

Pipeline (see REPORT.md):
  1. phase segmentation from the SAM2 object masks (static / manipulated / static);
  2. global yaw-symmetry check (pallet is ~180 deg symmetric about its thin axis): both hypotheses are refined
     with the same silhouette-only optimiser, the flip is adopted only if it wins clearly everywhere;
  3. static phases: ONE pose per phase, fitted jointly to all frames (person pixels excluded);
  4. manipulated phase: joint render-and-compare optimisation of all frames with SE(3) smoothness terms and the
     static poses as fixed boundary conditions;
  5. depth re-anchoring: camera-frame positions are moved along the viewing rays so that the depth equals the
     baseline's (per static phase: baseline median z; manipulated phase: temporally smoothed baseline z,
     ramped to be continuous with the static phases).
Deterministic; uses only /work/inputs (+ installed python packages). GPU memory < 2 GB.
"""
import argparse, json, os, sys, time
import numpy as np, torch, h5py, trimesh, cv2
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoilib import Renderer
from refine import rotvec_to_mat, render_soft, render_soft_multi, pool

# ----------------------------------------------------------------------------------------------- constants
STATIC_IOU_THR = 0.975   # object-mask IoU (person pixels excluded) w.r.t. first/last frame above which the object
                         # is considered not moved. Verified by eye: lift-off at ~frame 201, set down at ~frame 579.
RUN = 8                  # frames the IoU must stay below the threshold to count as motion
C_CROP, S_UP = 320, 2    # crop size (native px) around the object, render up-sampling factor
FLIP_MARGIN = 0.005      # mean IoU gain the yaw-flipped hypothesis must reach in EVERY test group to be adopted
Z_SMOOTH_SIGMA = 6.0     # frames, Gaussian smoothing of the baseline depth in the manipulated phase
W_ROT2, W_T2 = 4.0, 400.0  # weights of 2nd-difference smoothness (rotation matrix [Frobenius], translation [m])
W_PRIOR_R, W_PRIOR_T = 2.0, 20.0  # weak prior towards the baseline (rotvec [rad], translation [m])


def log(*a):
    print(*a, flush=True)


def seed_all(s=0):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True


# ----------------------------------------------------------------------------------------------- geometry utils
def mean_rotation(Rs):
    return Rot.from_matrix(Rs).mean().as_matrix()


def symmetry_transform(mesh):
    """Rigid transform (mesh frame) mapping the mesh onto itself by ~180 deg about its thin (y) axis (ICP fit)."""
    import open3d as o3d
    p = trimesh.sample.sample_surface(mesh, 60000, seed=0)[0]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
    ext = mesh.extents; ax = int(np.argmin(ext))  # thin axis
    D = -np.eye(3); D[ax, ax] = 1
    cen = mesh.bounds.mean(0)
    T0 = np.eye(4); T0[:3, :3] = D; T0[:3, 3] = cen - D @ cen
    reg = o3d.pipelines.registration.registration_icp(
        pc, pc, 0.03, T0, o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
    return np.asarray(reg.transformation), reg.inlier_rmse


def rescale_to_depth(Tc, z):
    """Move the object along its viewing ray so that camera-frame depth equals z (image position unchanged)."""
    T = Tc.copy(); T[:3, 3] *= z / Tc[2, 3]; return T


# ----------------------------------------------------------------------------------------------- data
class Masks:
    def __init__(self, path):
        h = h5py.File(path, "r"); self.o = h["object"]; self.p = h["person"]; self.F = self.o.shape[0]

    def crop(self, f, x0, y0, C):
        o = np.zeros((C, C), bool); p = np.ones((C, C), bool)
        H, W = self.o.shape[1:]
        ys, xs = slice(max(0, y0), min(H, y0 + C)), slice(max(0, x0), min(W, x0 + C))
        if ys.stop > ys.start and xs.stop > xs.start:
            o[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = self.o[f, ys, xs]
            p[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0] = self.p[f, ys, xs]
        return o, p


def crop_K(K, x0, y0, S):
    Kc = K.copy(); Kc[0, 0] *= S; Kc[1, 1] *= S
    Kc[0, 2] = (K[0, 2] - x0 + 0.5) * S - 0.5; Kc[1, 2] = (K[1, 2] - y0 + 0.5) * S - 0.5
    return Kc


def crop_origin(K, t):
    c = K @ t; u, v = c[0] / c[2], c[1] / c[2]
    return int(round(u - C_CROP / 2)), int(round(v - C_CROP / 2))


# ----------------------------------------------------------------------------------------------- phases
def segment_phases(masks):
    F = masks.F
    o = masks.o[:, ::2, ::2]; p = masks.p[:, ::2, ::2]
    def iou(a, b, v):
        return (a & b & v).sum() / max(1, ((a | b) & v).sum())
    i0 = np.array([iou(o[f], o[0], ~(p[f] | p[0])) for f in range(F)])
    i1 = np.array([iou(o[f], o[F - 1], ~(p[f] | p[F - 1])) for f in range(F)])
    def first_sustained(x):  # first index from which x stays below the threshold for RUN frames (ignores brief
        for k in range(len(x) - RUN):  # dips caused by mask noise / the person's feet next to the object)
            if (x[k:k + RUN] < STATIC_IOU_THR).all():
                return k
        return len(x)
    a = first_sustained(i0) - 1          # last static frame of phase 1 (+1 frame safety margin)
    b = F - first_sustained(i1[::-1])    # first static frame of the final phase
    return a, b, i0, i1


# ----------------------------------------------------------------------------------------------- losses
def sil_loss(ms, mo, valid):
    return (((ms - mo) ** 2) * valid).sum((-1, -2)) / ((torch.maximum(ms, mo) * valid).sum((-1, -2)) + 1.0)


def hard_iou(ms, mo, valid):
    hm = (ms > 0.5).float()
    return ((hm * mo * valid).sum((-1, -2)) / (((hm + mo) > 0).float() * valid).sum((-1, -2)).clamp_min(1))


def apply_delta(T0, w, t):
    T = T0.clone()
    T[:, :3, :3] = rotvec_to_mat(w) @ T0[:, :3, :3]
    T[:, :3, 3] = T0[:, :3, 3] + t
    return T


def fit_static(r, masks, frames, T_init, K, iters=300, dev="cuda"):
    """One pose for all `frames`: silhouette agreement with the visibility-weighted mean object mask.
    Depth is free here (only the rotation / image position are kept; depth is re-anchored later)."""
    x0, y0 = crop_origin(K, T_init[:3, 3])
    acc_o = np.zeros((C_CROP, C_CROP)); acc_v = np.zeros((C_CROP, C_CROP))
    for f in frames:
        o, p = masks.crop(f, x0, y0, C_CROP); v = ~p
        acc_o += o * v; acc_v += v
    mo = torch.tensor(acc_o / np.maximum(acc_v, 1), dtype=torch.float32, device=dev)[None]
    valid = torch.tensor(acc_v / len(frames), dtype=torch.float32, device=dev)[None]
    Kc = crop_K(K, x0, y0, S_UP)
    T0 = torch.tensor(T_init[None], dtype=torch.float32, device=dev)
    w = torch.zeros(1, 3, device=dev, requires_grad=True); t = torch.zeros(1, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{"params": [w], "lr": 0.003}, {"params": [t], "lr": 0.002}])
    for it in range(iters):
        ms, _ = render_soft(r, apply_delta(T0, w, t), Kc); ms = pool(ms, S_UP)
        loss = sil_loss(ms, mo, valid).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if it == iters // 2:
            for g in opt.param_groups: g["lr"] *= 0.3
    with torch.no_grad():
        T = apply_delta(T0, w, t)
        ms, _ = render_soft(r, T, Kc); ms = pool(ms, S_UP)
        i = hard_iou(ms, mo, (valid > 0.5).float()).item()
    return T[0].cpu().numpy().astype(float), i


class FrameBatch:
    """Cropped masks + crop intrinsics for a list of frames (crop centred on an initial pose)."""
    def __init__(self, masks, frames, Tinit, K, dev="cuda"):
        self.frames = frames; mo, va, Ks = [], [], []
        for f, T in zip(frames, Tinit):
            x0, y0 = crop_origin(K, T[:3, 3]); o, p = masks.crop(f, x0, y0, C_CROP)
            mo.append(o); va.append(~p); Ks.append(crop_K(K, x0, y0, S_UP))
        self.mo = torch.tensor(np.stack(mo), dtype=torch.float32, device=dev)
        self.va = torch.tensor(np.stack(va), dtype=torch.float32, device=dev)
        self.Ks = Ks


def render_batch_crops(r, T, Ks):
    """Render each pose with its own crop intrinsics (nvdiffrast batch shares the projection, so loop over
    groups); returns pooled soft masks."""
    return pool(render_soft_multi(r, T, Ks), S_UP)


def fit_sequence(r, fb, T_init, T_prev2, T_next2, iters=150, chunk=24, dev="cuda", tag=""):
    """Joint optimisation of all frames of a moving segment.
    loss = sum_f silhouette_f + smoothness (2nd differences of R and t, incl. fixed boundary poses)
           + weak prior to the initial poses.  Depth (z) is free; it is re-anchored afterwards."""
    N = len(fb.frames)
    T0 = torch.tensor(T_init, dtype=torch.float32, device=dev)
    w = torch.zeros(N, 3, device=dev, requires_grad=True); t = torch.zeros(N, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{"params": [w], "lr": 0.004}, {"params": [t], "lr": 0.002}])
    Bp = torch.tensor(np.stack(T_prev2), dtype=torch.float32, device=dev)
    Bn = torch.tensor(np.stack(T_next2), dtype=torch.float32, device=dev)
    for it in range(iters):
        opt.zero_grad()
        tot_sil = 0.0
        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            T = apply_delta(T0[s:e], w[s:e], t[s:e])
            ms = render_batch_crops(r, T, fb.Ks[s:e])
            l = sil_loss(ms, fb.mo[s:e], fb.va[s:e]).sum()
            l.backward(); tot_sil += l.item()
        T = apply_delta(T0, w, t)
        seq = torch.cat([Bp, T, Bn], 0)
        Rs, ts = seq[:, :3, :3], seq[:, :3, 3]
        d2R = Rs[2:] - 2 * Rs[1:-1] + Rs[:-2]; d2t = ts[2:] - 2 * ts[1:-1] + ts[:-2]
        # depth (z) noise is handled by re-anchoring; smoothness on x,y and rotation only
        reg = W_ROT2 * (d2R ** 2).sum() * 100 + W_T2 * (d2t[:, :2] ** 2).sum() * 100 \
            + W_PRIOR_R * (w ** 2).sum() + W_PRIOR_T * (t[:, :2] ** 2).sum()
        reg.backward()
        opt.step()
        if it == int(iters * 0.6):
            for g in opt.param_groups: g["lr"] *= 0.3
        if it % 50 == 0 or it == iters - 1:
            log(f"  [{tag}] it {it} sil {tot_sil / N:.4f} reg {reg.item():.4f}")
    with torch.no_grad():
        T = apply_delta(T0, w, t)
        ious = []
        for s in range(0, N, chunk):
            e = min(N, s + chunk)
            ms = render_batch_crops(r, T[s:e], fb.Ks[s:e])
            ious.append(hard_iou(ms, fb.mo[s:e], fb.va[s:e]))
    return T.cpu().numpy().astype(float), torch.cat(ious).cpu().numpy()


def per_frame_iou(r, fb, T, chunk=24, dev="cuda"):
    Tt = torch.tensor(T, dtype=torch.float32, device=dev); out = []
    with torch.no_grad():
        for s in range(0, len(fb.frames), chunk):
            e = min(len(fb.frames), s + chunk)
            ms = render_batch_crops(r, Tt[s:e], fb.Ks[s:e])
            out.append(hard_iou(ms, fb.mo[s:e], fb.va[s:e]))
    return torch.cat(out).cpu().numpy()


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--out", default="/work/outputs/object_poses_world.npz")
    args = ap.parse_args()
    t_start = time.time(); seed_all(0)
    inp = args.inputs
    cam = json.load(open(f"{inp}/camera.json"))
    K = np.array(cam["K"], float); Twc = np.array(cam["T_world_cam"], float); Tcw = np.linalg.inv(Twc)
    rec = np.load(f"{inp}/reconstruction.npz")
    Tw_base = rec["T_world_mesh"].astype(np.float64)
    F = Tw_base.shape[0]
    Tc_base = np.einsum("ij,fjk->fik", Tcw, Tw_base)
    mesh = trimesh.load(f"{inp}/object_mesh.glb", force="scene", process=False).to_geometry()
    masks = Masks(f"{inp}/masks.h5")
    r = Renderer(mesh, C_CROP * S_UP, C_CROP * S_UP, tex_max=512)

    # ---- 1. phases
    a, b, i0, i1 = segment_phases(masks)
    S1 = list(range(0, a + 1)); M = list(range(a + 1, b)); S2 = list(range(b, F))
    log(f"phases: static [0,{a}]  manipulated [{a + 1},{b - 1}]  static [{b},{F - 1}]")

    # ---- 2. static poses (baseline yaw), initialised from robust baseline averages
    def static_init(fr, Tc):
        T = np.eye(4); T[:3, :3] = mean_rotation(Tc[fr, :3, :3]); T[:3, 3] = np.median(Tc[fr, :3, 3], 0)
        return T
    Tsym, sym_rmse = symmetry_transform(mesh)
    log(f"yaw symmetry transform: ICP rmse {sym_rmse * 1000:.1f} mm")

    def fit_all_static(Tc):
        out = {}
        for name, fr in (("S1", S1), ("S2", S2)):
            T, i = fit_static(r, masks, fr[::2], static_init(fr, Tc), K)
            out[name] = (T, i)
        return out

    st_A = fit_all_static(Tc_base)
    Tc_flip = np.einsum("fij,jk->fik", Tc_base, Tsym)
    st_B = fit_all_static(Tc_flip)

    # ---- 3. yaw-flip test: identical silhouette-only optimisation for both labellings
    #         groups: static 1, static 2, manipulated (every 10th frame, per-frame fits)
    mid = M[::10]
    fbA = FrameBatch(masks, mid, Tc_base[mid], K)
    TA_m, iouA_m = fit_sequence(r, fbA, Tc_base[mid], [Tc_base[mid[0]]] * 2, [Tc_base[mid[-1]]] * 2,
                                iters=80, tag="flipA")
    TB_m, iouB_m = fit_sequence(r, fbA, Tc_flip[mid], [Tc_flip[mid[0]]] * 2, [Tc_flip[mid[-1]]] * 2,
                                iters=80, tag="flipB")
    gains = [st_B["S1"][1] - st_A["S1"][1], st_B["S2"][1] - st_A["S2"][1], float(np.mean(iouB_m - iouA_m))]
    flip = all(g > FLIP_MARGIN for g in gains)
    log(f"yaw test IoU gain of flipped labelling (S1, S2, M): {np.round(gains, 4)} -> flip={flip}")
    Tc_ref = Tc_flip if flip else Tc_base
    st = st_B if flip else st_A

    # ---- 4. static phases: one pose, depth = baseline median depth in the phase (moved along the ray)
    Tc_out = np.zeros_like(Tc_base)
    zS = {}
    for name, fr in (("S1", S1), ("S2", S2)):
        z = float(np.median(Tc_base[fr, 2, 3])); zS[name] = z
        T = rescale_to_depth(st[name][0], z)
        Tc_out[fr] = T
        log(f"{name}: joint IoU {st[name][1]:.4f}, depth {z:.3f}")

    # ---- 5. manipulated phase: joint optimisation with smoothness and static boundary poses
    TS1, TS2 = Tc_out[S1[-1]], Tc_out[S2[0]]
    # initial depth profile: smoothed baseline z, ramped to be continuous with the static phases
    zb = gaussian_filter1d(Tc_base[:, 2, 3], Z_SMOOTH_SIGMA, mode="nearest")
    zm = zb[M].copy()
    alpha = np.linspace(0, 1, len(M) + 2)[1:-1]
    zm += (1 - alpha) * (zS["S1"] - zm[0]) + alpha * (zS["S2"] - zm[-1])
    # initial poses: baseline rotation (relative motion smoothed lightly), translation rescaled to zm
    Tm0 = np.stack([rescale_to_depth(Tc_ref[f], z) for f, z in zip(M, zm)])
    fb = FrameBatch(masks, M, Tm0, K)
    iou_init = per_frame_iou(r, fb, Tm0)
    Tm, iou_m = fit_sequence(r, fb, Tm0, [TS1, TS1], [TS2, TS2], iters=150, tag="manip")
    log(f"manipulated: mean IoU {iou_init.mean():.4f} -> {iou_m.mean():.4f}")
    # re-anchor depth: along the ray, to the smooth continuous baseline depth profile
    Tm = np.stack([rescale_to_depth(T, z) for T, z in zip(Tm, zm)])
    Tc_out[M] = Tm

    # ---- 6. output in world frame
    Tw_out = np.einsum("ij,fjk->fik", Twc, Tc_out)
    Tw_out[:, 3] = [0, 0, 0, 1]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, T_world_mesh=Tw_out.astype(np.float64),
             phases=np.array([0, a, a + 1, b - 1, b, F - 1]), yaw_flipped=np.array(flip),
             yaw_test_gains=np.array(gains))
    for name, fr in (("S1", S1), ("M", M), ("S2", S2)):
        dz = (Tc_out[fr, 2, 3] - Tc_base[fr, 2, 3]) * 100
        log(f"{name}: depth diff vs baseline mean {dz.mean():+.2f} cm median {np.median(dz):+.2f} cm")
    log(f"saved {args.out}  ({time.time() - t_start:.1f} s)")


if __name__ == "__main__":
    main()
