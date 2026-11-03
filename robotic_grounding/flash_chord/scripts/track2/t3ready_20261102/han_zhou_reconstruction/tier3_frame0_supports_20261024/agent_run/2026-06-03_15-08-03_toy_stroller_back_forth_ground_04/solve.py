#!/usr/bin/env python
"""Floor-constrained, silhouette-registered, temporally smoothed 6-DoF trajectory of the toy stroller.

Method (see REPORT.md):
  1. Floor model: the stroller rolls / rests on the floor on its convex-hull support face (the wheel contact
     polygon). Its world pose is then (yaw about gravity, 2D position on the floor); roll/pitch and height above
     the released floor plane are constant. The support face is chosen automatically as the largest hull face
     whose normal is within 5 deg of the baseline's average "down" direction in mesh coordinates; the support
     height h0 is the median over the episode of the baseline's support height.
  2. Initialisation: every baseline pose is projected onto this model (minimal tilt correction, the bbox centre
     is kept on its viewing ray).
  3. Per-frame render-and-compare (nvdiffrast): yaw + floor position are optimised to maximise the soft IoU
     between the rendered silhouette (dilated by 3 px to model the dilated, hole-filled SAM2 masks) and the SAM2
     object mask; pixels covered by the person (and not by the object) are ignored.
  4. 180-deg yaw-flip diagnostic on a frame subset; a segment would only be flipped if the flipped hypothesis is
     clearly better over a long run of frames (it is not, for this episode).
  5. Phases: static at the start / end, detected automatically from the registered trajectory; each static phase
     gets ONE pose (robust median over its frames). Moving frames: Hampel outlier rejection + visibility-weighted
     Gaussian smoothing.
  6. Depth anchoring: one global factor scales the camera-frame translations along their viewing rays so that
     the median per-frame camera-depth difference to the baseline is zero (keeps the baseline's placement).
"""
import argparse
import json
import os
import time

import cv2
import h5py
import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr
from scipy.spatial.transform import Rotation as Rot

# ----------------------------------------------------------------------------------------------- constants
RES = 256            # crop resolution (crops are ~1:1 with the image unless the object is larger than ~180 px)
SIL_DILATE = 3       # px; SAM2 masks are ~3 px fatter than the true silhouette (chosen on static frames 0-95,
                     # where it gave the best IoU among {0,2,3,4,6} px)
ITERS = 120          # Adam iterations of the per-frame registration
BATCH = 48
STATIC_POS_TOL = 0.01    # m; static phase = registered position within 1 cm ...
STATIC_ROT_TOL = 3.0     # deg; ... and yaw within 3 deg of the phase reference
SMOOTH_SIGMA = 2.0       # frames; Gaussian smoothing of moving frames
FLIP_STEP = 6            # flip diagnostic on every 6th frame
FLIP_MIN_RUN = 30        # frames; a flip would need a run this long ...
FLIP_MARGIN = 0.05       # ... with mean IoU-loss improvement larger than this


# ----------------------------------------------------------------------------------------------- utilities
def load_inputs(inp):
    cam = json.load(open(os.path.join(inp, "camera.json")))
    K = np.array(cam["K"], np.float64)
    Twc = np.array(cam["T_world_cam"], np.float64)
    rec = np.load(os.path.join(inp, "reconstruction.npz"))
    rec = {k: rec[k] for k in rec.files}
    plane = np.array(json.load(open(os.path.join(inp, "ground_plane.json")))["plane"], np.float64)
    return cam, K, Twc, rec, plane


def load_mesh(inp):
    return trimesh.load(os.path.join(inp, "object_mesh.glb"), force="scene", process=False).to_geometry()


def rot_angle_deg(Ra, Rb):
    return np.degrees(Rot.from_matrix(Ra.T @ Rb).magnitude())


class Renderer:
    def __init__(self, mesh, device="cuda"):
        self.dev = device
        self.glctx = dr.RasterizeCudaContext(device=device)
        self.v = torch.tensor(np.asarray(mesh.vertices, np.float32), device=device)
        self.f = torch.tensor(np.asarray(mesh.faces, np.int32), device=device)


def proj_batch(Kc, W, H, near=0.05, far=20.0):
    """OpenCV intrinsics (B,3,3) -> clip-space projection (B,4,4); image row 0 maps to NDC y=-1."""
    B = Kc.shape[0]
    P = torch.zeros(B, 4, 4, device=Kc.device, dtype=torch.float32)
    P[:, 0, 0] = 2 * Kc[:, 0, 0] / W
    P[:, 0, 2] = (2 * (Kc[:, 0, 2] + 0.5) - W) / W
    P[:, 1, 1] = 2 * Kc[:, 1, 1] / H
    P[:, 1, 2] = (2 * (Kc[:, 1, 2] + 0.5) - H) / H
    P[:, 2, 2] = (far + near) / (far - near)
    P[:, 2, 3] = -2 * far * near / (far - near)
    P[:, 3, 2] = 1
    return P


def render_alpha(R, T, Kc, res):
    """Differentiable anti-aliased silhouette (B,res,res) of the mesh under cam-from-mesh poses T (B,4,4)."""
    vh = torch.cat([R.v, torch.ones_like(R.v[:, :1])], 1)
    vc = torch.einsum("bij,nj->bni", T, vh)
    clip = torch.einsum("bij,bnj->bni", proj_batch(Kc, res, res), vc).contiguous()
    rast, _ = dr.rasterize(R.glctx, clip, R.f, resolution=[res, res])
    mask = (rast[..., 3:4] > 0).float()
    return dr.antialias(mask, rast, clip, R.f)[..., 0]


def axis_angle_torch(axis, ang):
    k = axis
    Kx = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=torch.float32, device=ang.device)
    I = torch.eye(3, device=ang.device)
    s, c = torch.sin(ang)[:, None, None], torch.cos(ang)[:, None, None]
    return I + s * Kx + (1 - c) * (Kx @ Kx)


class FloorModel:
    """Object resting on its convex-hull support face on a plane parallel to the released floor plane, at height
    h0 above it. Pose = yaw theta about gravity and position (a,b) along in-plane axes e1,e2."""

    def __init__(self, mesh, plane, down_hint, h0=0.0):
        v = np.asarray(mesh.vertices, np.float64)
        hull = trimesh.convex.convex_hull(v)
        fn = hull.face_normals
        cand = np.where(fn @ down_hint > np.cos(np.radians(5)))[0]
        f = cand[np.argmax(hull.area_faces[cand])]
        self.dm = fn[f]
        self.s = float((v @ self.dm).max())
        nn = np.linalg.norm(plane[:3])
        self.n, self.d = plane[:3] / nn, plane[3] / nn
        self.g = -self.n
        ax = np.cross(self.dm, self.g); sn = np.linalg.norm(ax)
        self.B = Rot.from_rotvec(ax / sn * np.arctan2(sn, self.dm @ self.g)).as_matrix()   # B @ dm = g
        e1 = np.array([1.0, 0, 0]); e1 -= (e1 @ self.n) * self.n; e1 /= np.linalg.norm(e1)
        self.e1, self.e2 = e1, np.cross(self.n, e1)
        self.h0 = h0
        self.c = 0.5 * (v.min(0) + v.max(0))      # bbox centre (mesh frame)

    def support_height(self, T):
        """Height of the support face above the plane for a world pose T (exact for tilt-free poses)."""
        return float(self.n @ T[:3, 3] + self.d - (self.dm @ (T[:3, :3].T @ self.g)) * self.s)

    def t0(self, h0=None):
        h0 = self.h0 if h0 is None else h0
        return self.n * (h0 + self.s - self.d)

    def pose(self, theta, a, b, h0=None):
        T = np.eye(4)
        T[:3, :3] = Rot.from_rotvec(self.g * theta).as_matrix() @ self.B
        T[:3, 3] = self.t0(h0) + a * self.e1 + b * self.e2
        return T

    def pose_torch(self, theta, a, b):
        dev = theta.device
        f32 = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)
        R = axis_angle_torch(f32(self.g), theta) @ f32(self.B)
        t = f32(self.t0()) + a[:, None] * f32(self.e1) + b[:, None] * f32(self.e2)
        T = torch.zeros(theta.shape[0], 4, 4, device=dev)
        T[:, :3, :3] = R; T[:, :3, 3] = t; T[:, 3, 3] = 1
        return T

    def from_pose(self, T, cam_o):
        Rb = T[:3, :3]
        u = Rb @ self.dm
        ax = np.cross(u, self.g); sn = np.linalg.norm(ax)
        Rc = np.eye(3) if sn < 1e-12 else Rot.from_rotvec(ax / sn * np.arctan2(sn, u @ self.g)).as_matrix()
        Rn = Rc @ Rb
        theta = float(Rot.from_matrix(Rn @ self.B.T).as_rotvec() @ self.g)
        pc = Rb @ self.c + T[:3, 3]; r = pc - cam_o; r /= np.linalg.norm(r)
        lam = (self.h0 + self.s - self.d - self.n @ cam_o + self.n @ (Rn @ self.c)) / (self.n @ r)
        t = cam_o + lam * r - Rn @ self.c - self.t0()
        return theta, float(t @ self.e1), float(t @ self.e2)

    def yaw_flip(self, theta, a, b, dth=np.pi):
        """Rotate by dth about the vertical axis through the bbox centre (keeps the object's location)."""
        P0 = self.pose(theta, a, b); P1 = self.pose(theta + dth, a, b)
        dt = (P0[:3, :3] @ self.c) - (P1[:3, :3] @ self.c)
        return theta + dth, a + dt @ self.e1, b + dt @ self.e2


# ----------------------------------------------------------------------------------------------- registration
def build_crops(mf, ids, poses_cam, K, mesh_v, chunk=64):
    """Crop object mask and validity (not person, or object) around each frame's object.
    Returns crops, crop intrinsics and the full-image visible object area per frame."""
    M, V, Kc, area = [], [], [], []
    vs = mesh_v[::50]
    ids = list(ids)
    cache = {}
    for j, i in enumerate(ids):
        if i not in cache:   # read masks in contiguous chunks (much faster than per-frame h5 reads)
            cache = {}
            lo_i = i; hi_i = min(i + chunk, mf["object"].shape[0])
            ob, pe = mf["object"][lo_i:hi_i], mf["person"][lo_i:hi_i]
            for k in range(hi_i - lo_i):
                cache[lo_i + k] = (ob[k], pe[k])
        om, pm = cache[i]
        area.append(float(om.sum()))
        Pc = poses_cam[j]
        pv = vs @ Pc[:3, :3].T + Pc[:3, 3]
        uv = pv[:, :2] / pv[:, 2:] * K[[0, 1], [0, 1]] + K[:2, 2]
        lo, hi = uv.min(0), uv.max(0)
        ys, xs = np.nonzero(om)
        if len(xs) > 50:
            lo = np.minimum(lo, [xs.min(), ys.min()]); hi = np.maximum(hi, [xs.max(), ys.max()])
        c = (lo + hi) / 2
        S = int(max(RES, 1.4 * (hi - lo).max()))
        x0, y0 = int(c[0] - S / 2), int(c[1] - S / 2)
        r = RES / S
        Kc.append([[K[0, 0] * r, 0, (K[0, 2] - x0 + 0.5) * r - 0.5], [0, K[1, 1] * r, (K[1, 2] - y0 + 0.5) * r - 0.5], [0, 0, 1]])

        def crop(a):
            pad = np.zeros((S, S), np.float32)
            xa, ya = max(0, x0), max(0, y0); xb, yb = min(a.shape[1], x0 + S), min(a.shape[0], y0 + S)
            if xb > xa and yb > ya:
                pad[ya - y0:yb - y0, xa - x0:xb - x0] = a[ya:yb, xa:xb]
            return cv2.resize(pad, (RES, RES), interpolation=cv2.INTER_AREA)
        M.append(crop(om.astype(np.float32)))
        V.append(crop(((~pm) | om).astype(np.float32)))   # outside the image -> 0 (invalid)
    return np.stack(M), np.stack(V), np.array(Kc, np.float32), np.array(area)


def sil_loss(R, fm, Tcw_t, th, a, b, Kc, M, V):
    Tc = Tcw_t @ fm.pose_torch(th, a, b)
    al = render_alpha(R, Tc, Kc, RES)
    if SIL_DILATE > 0:
        al = torch.nn.functional.max_pool2d(al[:, None], 2 * SIL_DILATE + 1, 1, SIL_DILATE)[:, 0]
    inter = (al * M * V).sum((1, 2)); uni = ((al + M - al * M) * V).sum((1, 2))
    return 1 - inter / uni.clamp(min=1)


def register(R, fm, Tcw, M, V, Kc, init, iters=ITERS):
    """Per-frame optimisation of (theta,a,b). Returns params (N,3) and final loss (N,)."""
    dev = "cuda"
    Tcw_t = torch.tensor(Tcw, dtype=torch.float32, device=dev)
    out, losses = [], []
    for s0 in range(0, len(init), BATCH):
        sl = slice(s0, min(len(init), s0 + BATCH))
        Mt = torch.tensor(M[sl], device=dev); Vt = torch.tensor(V[sl], device=dev); Kt = torch.tensor(Kc[sl], device=dev)
        th = torch.tensor(init[sl, 0], dtype=torch.float32, device=dev, requires_grad=True)
        a = torch.tensor(init[sl, 1], dtype=torch.float32, device=dev, requires_grad=True)
        b = torch.tensor(init[sl, 2], dtype=torch.float32, device=dev, requires_grad=True)
        opt = torch.optim.Adam([{"params": [th], "lr": 0.02}, {"params": [a, b], "lr": 0.005}])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=0.1 * 0.005)
        for _ in range(iters):
            L = sil_loss(R, fm, Tcw_t, th, a, b, Kt, Mt, Vt)
            opt.zero_grad(); L.sum().backward(); opt.step(); sched.step()
        with torch.no_grad():
            L = sil_loss(R, fm, Tcw_t, th, a, b, Kt, Mt, Vt)
        out.append(torch.stack([th, a, b], 1).detach().cpu().numpy().astype(np.float64))
        losses.append(L.cpu().numpy())
    return np.concatenate(out), np.concatenate(losses)


def eval_loss(R, fm, Tcw, M, V, Kc, params):
    dev = "cuda"
    Tcw_t = torch.tensor(Tcw, dtype=torch.float32, device=dev)
    out = []
    with torch.no_grad():
        for s0 in range(0, len(params), BATCH):
            sl = slice(s0, min(len(params), s0 + BATCH))
            p = torch.tensor(params[sl], dtype=torch.float32, device=dev)
            out.append(sil_loss(R, fm, Tcw_t, p[:, 0], p[:, 1], p[:, 2], torch.tensor(Kc[sl], device=dev),
                                torch.tensor(M[sl], device=dev), torch.tensor(V[sl], device=dev)).cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------------------------- temporal
def detect_static(P, from_start=True, n_ref=30):
    """Length of the static run at the start (or end) of the registered trajectory P (F,3)."""
    Q = P if from_start else P[::-1]
    ref = np.median(Q[:n_ref], 0)
    dev_pos = np.linalg.norm(Q[:, 1:] - ref[1:], axis=1)
    dev_rot = np.degrees(np.abs(Q[:, 0] - ref[0]))
    moving = (dev_pos > STATIC_POS_TOL) | (dev_rot > STATIC_ROT_TOL)
    # first frame from which the object is moving for 5 consecutive frames
    for t in range(len(Q) - 5):
        if moving[t:t + 5].all():
            return t
    return len(Q)


def gaussian_smooth(x, w, sigma):
    F = len(x)
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    num = np.convolve(np.pad(x * w, r, mode="edge"), k, "valid")
    den = np.convolve(np.pad(w, r, mode="edge"), k, "valid")
    return num / np.maximum(den, 1e-9)


def hampel(x, win=7, nsig=3.0, floor=0.0):
    F = len(x); bad = np.zeros(F, bool)
    for t in range(F):
        a, b = max(0, t - win), min(F, t + win + 1)
        med = np.median(x[a:b]); mad = 1.4826 * np.median(np.abs(x[a:b] - med))
        bad[t] = abs(x[t] - med) > nsig * max(mad, floor)
    return bad


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--out", default="/work/outputs/object_poses_world.npz")
    ap.add_argument("--diag", default=None, help="optional path to save diagnostics (npz)")
    args = ap.parse_args()
    t_start = time.time()
    np.random.seed(0); torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    cam, K, Twc, rec, plane = load_inputs(args.inputs)
    mesh = load_mesh(args.inputs)
    mv = np.asarray(mesh.vertices, np.float64)
    Tcw = np.linalg.inv(Twc); o = Twc[:3, 3]
    Tb = rec["T_world_mesh"].astype(np.float64)
    F = len(Tb)

    # 1. floor model
    nrm = plane[:3] / np.linalg.norm(plane[:3]); g = -nrm
    down_hint = np.mean([T[:3, :3].T @ g for T in Tb], 0); down_hint /= np.linalg.norm(down_hint)
    fm = FloorModel(mesh, plane, down_hint)
    tilt_b = np.array([np.degrees(np.arccos(np.clip((T[:3, :3] @ fm.dm) @ fm.g, -1, 1))) for T in Tb])
    hb = np.array([fm.support_height(T) for T in Tb])
    fm.h0 = float(np.median(hb))
    print(f"[floor] support normal (mesh) {np.round(fm.dm, 3)}, angle to baseline mean down "
          f"{np.degrees(np.arccos(fm.dm @ down_hint)):.1f} deg; baseline tilt median {np.median(tilt_b):.1f} deg; "
          f"h0 = {fm.h0 * 100:.1f} cm (baseline support height p10/p90 {np.percentile(hb, 10) * 100:.1f}/{np.percentile(hb, 90) * 100:.1f} cm)")

    # 2. init from baseline
    init = np.array([fm.from_pose(T, o) for T in Tb])
    init[:, 0] = np.unwrap(init[:, 0])

    # 3. per-frame registration
    with h5py.File(os.path.join(args.inputs, "masks.h5"), "r") as mf:
        M, V, Kc, vis_area = build_crops(mf, range(F), [Tcw @ fm.pose(*p) for p in init], K, mv)
    print(f"[crops] {time.time() - t_start:.0f}s")
    R = Renderer(mesh)
    reg, loss = register(R, fm, Tcw, M, V, Kc, init)
    loss_init = eval_loss(R, fm, Tcw, M, V, Kc, init)
    reg[:, 0] = np.unwrap(reg[:, 0])
    print(f"[register] {time.time() - t_start:.0f}s; mean sil loss init {loss_init.mean():.3f} -> {loss.mean():.3f}")

    # 4. 180-deg yaw flip diagnostic (about the vertical axis through the bbox centre)
    fids = np.arange(0, F, FLIP_STEP)
    finit = np.array([fm.yaw_flip(*reg[i]) for i in fids])
    freg, floss = register(R, fm, Tcw, M[fids], V[fids], Kc[fids], finit)
    gain = loss[fids] - floss            # >0: flipped hypothesis fits better
    flip_segments = []
    run = 0
    for j, gj in enumerate(gain):
        run = run + 1 if gj > 0 else 0
        if run * FLIP_STEP >= FLIP_MIN_RUN:
            a = j - run + 1
            if gain[a:j + 1].mean() > FLIP_MARGIN:
                flip_segments.append((int(fids[a]), int(fids[j])))
    print(f"[flip] flipped better on {np.mean(gain > 0) * 100:.0f}% of {len(fids)} tested frames "
          f"(mean gain where better {gain[gain > 0].mean() if (gain > 0).any() else 0:.3f}); "
          f"segments meeting the adoption rule: {flip_segments}")
    for (a, b) in flip_segments:   # adopt flip on that segment (not triggered for this episode)
        for i in range(a, b + 1):
            reg[i] = np.array(fm.yaw_flip(*reg[i]))
        reg[:, 0] = np.unwrap(reg[:, 0])

    # 5. phases
    s0 = detect_static(reg, True)
    s1 = F - detect_static(reg, False)
    print(f"[phases] static start: 0..{s0 - 1}, static end: {s1}..{F - 1}")
    P = reg.copy()
    # visibility weight: visible mask area relative to the phase-robust expected area
    w = np.clip(vis_area / np.maximum(gaussian_smooth(vis_area, np.ones(F), 30), 1), 0.05, 1.0) * np.exp(-((loss / 0.35) ** 2))
    # moving frames: outlier rejection and smoothing
    bad = np.zeros(F, bool)
    for k, fl in zip(range(3), [np.radians(1.0), 0.005, 0.005]):
        bad |= hampel(P[:, k], 7, 3.0, fl)
    w_s = w * (~bad)
    for k in range(3):
        P[:, k] = gaussian_smooth(P[:, k], w_s, SMOOTH_SIGMA)
    # static phases: one pose each (median of the registered per-frame estimates)
    if s0 > 0:
        P[:s0] = np.median(reg[:s0], 0)
    if s1 < F:
        P[s1:] = np.median(reg[s1:], 0)
    Tw = np.array([fm.pose(*p) for p in P])

    # 6. global depth anchoring to the baseline (along viewing rays, camera frame). One factor for the whole
    #    episode (the floor model ties the phases together physically). It is chosen so that the median over
    #    frames of the per-frame camera-depth ratio baseline/ours is 1, i.e. the median per-frame depth
    #    difference to the baseline is ~0 (the robust form of "keep the baseline's mean depth"; matching only
    #    median(z) would be decided by the few frames near the median depth and here shifts the trajectory by
    #    +5 cm on average). Independent depth cues are only reported (validate.py), not applied.
    Tc_ours = Tcw @ Tw; Tc_base = Tcw @ Tb
    scale = np.median(Tc_base[:, 2, 3] / Tc_ours[:, 2, 3])
    Tc_ours[:, :3, 3] *= scale
    T_final = Twc @ Tc_ours
    print(f"[anchor] global ray scale {scale:.4f} (median z base {np.median(Tc_base[:, 2, 3]):.3f} m)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, T_world_mesh=T_final.astype(np.float64))
    print(f"[done] wrote {args.out} ({T_final.shape}) in {time.time() - t_start:.0f}s; "
          f"peak torch GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    if args.diag:
        np.savez(args.diag, init=init, reg=reg, loss=loss, loss_init=loss_init, P=P, w=w, bad=bad, s0=s0, s1=s1,
                 fids=fids, flip_gain=gain, scale=scale, h0=fm.h0, vis_area=vis_area)


if __name__ == "__main__":
    main()
