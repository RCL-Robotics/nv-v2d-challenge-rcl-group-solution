#!/usr/bin/env python
"""Trajectory-level 6-DoF pose recovery of the toy stroller (episode 23).

Method (see REPORT.md):
  1. The object is a rigid 3-wheeled stroller that stays on the floor the whole episode (rolled / pushed).
     Its physically stable support face is taken from the mesh convex hull (largest hull face whose outward
     normal is close to the baseline's median "down" direction).  Every pose is parameterised as
     upright-on-floor:  R_t = Rot(n_floor, yaw_t) @ R0,  support plane at a constant height c above the
     released floor plane, and the floor position (a_t, b_t) of the support point below the centroid.
  2. Static phases (object not moving) are detected from the video (frame differences inside the object mask,
     person pixels excluded) and share one pose.
  3. Per-frame evidence: a sharp object probability map (colour model of the stroller vs. surrounding carpet,
     learned from the baseline render ∩ SAM mask, restricted to the dilated SAM object mask), person pixels
     ignored.  All poses are optimised jointly by differentiable silhouette rendering (nvdiffrast) with
     second-order temporal smoothness on (yaw, a, b).
  4. Depth re-anchoring: camera-frame positions are scaled along the viewing rays so that the median
     camera-frame depth equals the baseline's median depth (see ANCHOR_MODE).
"""
import argparse
import json
import os
import sys
import time

import cv2
import h5py
import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoi_render import MeshRenderer  # noqa: E402
import nvdiffrast.torch as dr  # noqa: E402

SEED = 0
CROP = 160            # crop resolution for render-and-compare
STATIC_START_THR = 13.0   # mean abs grey diff vs. frame 0 inside object mask (person excluded)
STATIC_END_THR = 10.0     # same vs. last frame
DEV = "cuda"


def log(*a):
    print("[solve]", *a, flush=True)


# ----------------------------------------------------------------------------- geometry helpers
def rodrigues(axis, ang):
    axis = axis / np.linalg.norm(axis)
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * Kx @ Kx


def rot_between(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); s = np.linalg.norm(v); c = a @ b
    if s < 1e-12:
        return np.eye(3)
    return rodrigues(v / s, np.arctan2(s, c))


class Inputs:
    def __init__(self, root):
        self.root = root
        cam = json.load(open(f"{root}/camera.json"))
        self.K = np.array(cam["K"], dtype=np.float64)
        self.Twc = np.array(cam["T_world_cam"], dtype=np.float64)
        self.Tcw = np.linalg.inv(self.Twc)
        self.H, self.W = cam["height"], cam["width"]
        pl = json.load(open(f"{root}/ground_plane.json"))["plane"]
        self.n = np.array(pl[:3], dtype=np.float64)   # points up (world is y-down)
        self.d = float(pl[3])
        rec = np.load(f"{root}/reconstruction.npz")
        self.Tbase = rec["T_world_mesh"].astype(np.float64)
        self.contact = rec["contact_prob_right_left"].astype(np.float64)
        self.F = self.Tbase.shape[0]
        self.masks = h5py.File(f"{root}/masks.h5", "r")
        self.mesh_path = f"{root}/object_mesh.glb"
        self.video = f"{root}/video.mp4"


class FloorModel:
    """Upright-on-floor parameterisation  (yaw, a, b) + global height c."""

    def __init__(self, inp, mesh, Tbase):
        n = inp.n
        V = np.asarray(mesh.vertices)
        # median floor-up direction expressed in mesh coordinates (baseline), used only to pick the hull face
        upm = np.einsum("fji,j->fi", Tbase[:, :3, :3], n)
        u0 = np.median(upm, 0); u0 /= np.linalg.norm(u0)
        hull = mesh.convex_hull
        ok = (hull.face_normals @ -u0) > 0.97
        k = np.argmax(hull.area_faces * ok)
        self.u_m = -hull.face_normals[k]                    # mesh-frame up of the resting pose
        self.h_low = (V @ self.u_m).min()                   # support plane offset along u_m
        ctr = V.mean(0)
        self.q_m = ctr - (ctr @ self.u_m - self.h_low) * self.u_m   # support point below the centroid
        self.R0 = rot_between(self.u_m, n)
        e1 = np.cross(n, [0, 0, 1.0]); e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        self.e1, self.e2, self.n = e1, e2, n
        self.P0 = -inp.d * n                                # a point on the floor plane
        self.tilt_base = np.degrees(np.arccos(np.clip(upm @ self.u_m, -1, 1)))
        self.ref_dir = self.R0 @ np.array([0, 0, 1.0])      # yaw reference: mesh z-axis after R0 (horizontal-ish)
        self.ref_dir -= (self.ref_dir @ n) * n; self.ref_dir /= np.linalg.norm(self.ref_dir)

    def yaw_of(self, R):
        """yaw such that Rot(n,yaw) R0 best matches R (project R R0^-1 to rotation about n)."""
        M = R @ self.R0.T
        v = M @ self.ref_dir
        v -= (v @ self.n) * self.n
        r = self.ref_dir
        return np.arctan2(np.cross(r, v) @ self.n, r @ v)

    def support_height(self, T):
        """height above the floor plane of the support point q_m under pose T (F,4,4)."""
        q = np.einsum("fij,j->fi", T[:, :3, :3], self.q_m) + T[:, :3, 3]
        return q @ self.n + (-(self.P0 @ self.n))

    def from_baseline(self, inp, T, c):
        """Init (yaw, a, b): yaw from T's rotation; floor position by sliding the support point along the
        camera ray (keeps its image position) to height c."""
        C = inp.Twc[:3, 3]
        yaw = np.array([self.yaw_of(T[i, :3, :3]) for i in range(len(T))])
        yaw = np.unwrap(yaw)
        q = np.einsum("fij,j->fi", T[:, :3, :3], self.q_m) + T[:, :3, 3]
        ray = q - C
        # want n.(C + s ray) + d = c  -> s
        s = (c - inp.d - C @ self.n) / (ray @ self.n)
        p = C + s[:, None] * ray
        a = (p - self.P0) @ self.e1
        b = (p - self.P0) @ self.e2
        return yaw, a, b

    # torch version for optimisation
    def to_T_torch(self, yaw, a, b, c):
        dev = yaw.device
        n = torch.tensor(self.n, dtype=torch.float32, device=dev)
        Kx = torch.tensor([[0, -self.n[2], self.n[1]], [self.n[2], 0, -self.n[0]], [-self.n[1], self.n[0], 0]],
                          dtype=torch.float32, device=dev)
        s, co = torch.sin(yaw)[:, None, None], torch.cos(yaw)[:, None, None]
        Rn = torch.eye(3, device=dev)[None] + s * Kx[None] + (1 - co) * (Kx @ Kx)[None]
        R = Rn @ torch.tensor(self.R0, dtype=torch.float32, device=dev)[None]
        P0 = torch.tensor(self.P0, dtype=torch.float32, device=dev)
        e1 = torch.tensor(self.e1, dtype=torch.float32, device=dev)
        e2 = torch.tensor(self.e2, dtype=torch.float32, device=dev)
        p = P0[None] + a[:, None] * e1[None] + b[:, None] * e2[None] + c * n[None]
        t = p - (R @ torch.tensor(self.q_m, dtype=torch.float32, device=dev))
        T = torch.zeros(len(yaw), 4, 4, device=dev)
        T[:, :3, :3] = R; T[:, :3, 3] = t; T[:, 3, 3] = 1
        return T

    def to_T(self, yaw, a, b, c):
        out = np.zeros((len(yaw), 4, 4))
        for i in range(len(yaw)):
            R = rodrigues(self.n, yaw[i]) @ self.R0
            p = self.P0 + a[i] * self.e1 + b[i] * self.e2 + c * self.n
            out[i, :3, :3] = R; out[i, :3, 3] = p - R @ self.q_m; out[i, 3, 3] = 1
        return out


# ----------------------------------------------------------------------------- evidence
def read_video(path, F):
    cap = cv2.VideoCapture(path)
    for i in range(F):
        ok, fr = cap.read()
        assert ok, f"video ended at {i}"
        yield i, fr


LAB_BINS = (16, 32, 32)


def lab_idx(img):
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.int32)
    return (L[..., 0] * LAB_BINS[0] // 256, L[..., 1] * LAB_BINS[1] // 256, L[..., 2] * LAB_BINS[2] // 256)


def render_full_masks(rend, inp, T_world, ids, bs=16):
    out = {}
    for k in range(0, len(ids), bs):
        sub = ids[k:k + bs]
        Tc = np.stack([inp.Tcw @ T_world[i] for i in sub])
        o = rend.render(Tc, inp.K, inp.H, inp.W, want_rgb=False)
        m = o["mask"].cpu().numpy()
        for j, i in enumerate(sub):
            out[i] = m[j]
    return out


def build_evidence(inp, rend, Tinit):
    """One pass over the video: colour model, static-phase statistics, per-frame crops of the target
    probability map and the valid (non-person) weight."""
    F, H, W = inp.F, inp.H, inp.W
    train = set(range(0, F, 5))
    hf = np.zeros(LAB_BINS); hb = np.zeros(LAB_BINS)
    base_masks = {}
    ids = list(range(F))
    for k in range(0, F, 64):
        base_masks.update(render_full_masks(rend, inp, Tinit, ids[k:k + 64]))
        # keep only what we need (bbox) to save memory
        for i in ids[k:k + 64]:
            m = base_masks[i]
            ys, xs = np.nonzero(m)
            base_masks[i] = (m if i in train else None,
                             (xs.min(), xs.max(), ys.min(), ys.max()) if len(xs) else None)
    grey = []
    obj0 = inp.masks["object"][0][::2, ::2]; objE = inp.masks["object"][F - 1][::2, ::2]
    frames_small = {}
    boxes = np.zeros((F, 4))
    cache = {}
    k15 = np.ones((15, 15), np.uint8); k61 = np.ones((61, 61), np.uint8)
    for i, fr in read_video(inp.video, F):
        om = inp.masks["object"][i]; pm = inp.masks["person"][i]
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)[::2, ::2].astype(np.float32)
        grey.append((g, pm[::2, ::2]))
        if i in train:
            m = base_masks[i][0]
            fg = m & om & ~pm
            ring = cv2.dilate(om.astype(np.uint8), k61).astype(bool) & ~cv2.dilate(om.astype(np.uint8), k15).astype(bool) & ~pm
            li = lab_idx(fr)
            np.add.at(hf, tuple(x[fg] for x in li), 1)
            np.add.at(hb, tuple(x[ring] for x in li), 1)
        # crop box: union of SAM mask bbox and init render bbox
        ys, xs = np.nonzero(om)
        bb = base_masks[i][1]
        x0s, x1s, y0s, y1s = [], [], [], []
        if len(xs):
            x0s.append(xs.min()); x1s.append(xs.max()); y0s.append(ys.min()); y1s.append(ys.max())
        if bb is not None:
            x0s.append(bb[0]); x1s.append(bb[1]); y0s.append(bb[2]); y1s.append(bb[3])
        cx, cy = (min(x0s) + max(x1s)) / 2, (min(y0s) + max(y1s)) / 2
        side = 1.25 * max(max(x1s) - min(x0s), max(y1s) - min(y0s)) + 20
        boxes[i] = [cx - side / 2, cy - side / 2, side, side]
        cache[i] = (fr, om, pm)
        if len(cache) >= 64 or i == F - 1:
            frames_small.update(_crop_batch(cache, boxes))
            cache = {}
    hf /= hf.sum(); hb /= hb.sum()
    P = (hf + 1e-6) / (hf + hb + 2e-6)
    # second pass on the stored crops: colour probability (crops keep the raw colour)
    tgt = np.zeros((F, CROP, CROP), np.float32); val = np.zeros((F, CROP, CROP), np.float32)
    for i in range(F):
        img, om, pm = frames_small[i]
        p = P[lab_idx(img)].astype(np.float32)
        omd = cv2.dilate(om, np.ones((5, 5), np.uint8)).astype(np.float32)
        tgt[i] = cv2.GaussianBlur(p * omd, (3, 3), 0)
        val[i] = 1.0 - pm.astype(np.float32)
    # static statistics
    d0 = np.full(F, np.nan); dE = np.full(F, np.nan)
    g0 = grey[0][0]; gE = grey[F - 1][0]
    for i, (g, pm) in enumerate(grey):
        v = obj0 & ~pm
        if v.sum() > 100:
            d0[i] = np.abs(g[v] - g0[v]).mean()
        v = objE & ~pm
        if v.sum() > 100:
            dE[i] = np.abs(g[v] - gE[v]).mean()
    return dict(tgt=tgt, val=val, boxes=boxes, d0=d0, dE=dE, colP=P)


def _crop_batch(cache, boxes):
    out = {}
    for i, (fr, om, pm) in cache.items():
        x0, y0, s, _ = boxes[i]
        A = np.array([[CROP / s, 0, -x0 * CROP / s], [0, CROP / s, -y0 * CROP / s]], np.float32)
        img = cv2.warpAffine(fr, A, (CROP, CROP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        omc = cv2.warpAffine(om.astype(np.uint8), A, (CROP, CROP), flags=cv2.INTER_NEAREST)
        pmo = (pm & ~om).astype(np.uint8)
        pmc = cv2.warpAffine(pmo, A, (CROP, CROP), flags=cv2.INTER_NEAREST)
        out[i] = (img, omc, pmc)
    return out


# ----------------------------------------------------------------------------- differentiable silhouettes
def crop_proj(K, boxes, near=0.05, far=20.0):
    """Per-frame OpenGL projection for square crops boxes (B,4)=[x0,y0,s,s] rendered at CROP x CROP."""
    B = len(boxes)
    P = np.zeros((B, 4, 4), np.float32)
    for j, (x0, y0, s, _) in enumerate(boxes):
        sc = CROP / s
        fx, fy = K[0, 0] * sc, K[1, 1] * sc
        cx, cy = (K[0, 2] - x0 + 0.5) * sc, (K[1, 2] - y0 + 0.5) * sc
        P[j] = [[2 * fx / CROP, 0, 2 * cx / CROP - 1, 0], [0, 2 * fy / CROP, 2 * cy / CROP - 1, 0],
                [0, 0, (far + near) / (far - near), -2 * far * near / (far - near)], [0, 0, 1, 0]]
    return P


def render_sil(rend, T_cam, P):
    """Antialiased (differentiable) silhouettes. T_cam (B,4,4) torch, P (B,4,4) torch."""
    vh = torch.cat([rend.verts, torch.ones_like(rend.verts[:, :1])], 1)
    vc = torch.einsum("bij,nj->bni", T_cam, vh)
    clip = torch.einsum("bij,bnj->bni", P, vc).contiguous()
    rast, _ = dr.rasterize(rend.glctx, clip, rend.faces, resolution=[CROP, CROP])
    m = (rast[..., 3:] > 0).float()
    m = dr.antialias(m, rast, clip, rend.faces)
    return m[..., 0]


def seg_iou_loss(sil, tgt, val):
    inter = (sil * tgt * val).sum((1, 2))
    union = ((sil + tgt - sil * tgt) * val).sum((1, 2))
    return 1 - inter / union.clamp(min=1.0)


def optimise(rend, fm, inp, ev, yaw0, a0, b0, c, group, iters=300, lam_acc=1.0, lr=(4e-3, 2e-3),
             frames=None, bs=128, verbose=True):
    """Joint optimisation of per-group (yaw, a, b). group[t] = parameter index of frame t (static phases share
    one index). Returns per-frame yaw, a, b and per-frame final IoU."""
    F = inp.F
    frames = np.arange(F) if frames is None else np.asarray(frames)
    G = group.max() + 1
    # group init: mean of member frames (circular mean not needed: yaw unwrapped & close)
    def gmean(x):
        s = np.zeros(G); cnt = np.zeros(G)
        np.add.at(s, group, x); np.add.at(cnt, group, 1)
        return s / cnt
    gy = torch.tensor(gmean(yaw0), dtype=torch.float32, device=DEV, requires_grad=True)
    ga = torch.tensor(gmean(a0), dtype=torch.float32, device=DEV, requires_grad=True)
    gb = torch.tensor(gmean(b0), dtype=torch.float32, device=DEV, requires_grad=True)
    gidx = torch.tensor(group, device=DEV)
    opt = torch.optim.Adam([{"params": [gy], "lr": lr[0]}, {"params": [ga, gb], "lr": lr[1]}])
    Tcw = torch.tensor(inp.Tcw, dtype=torch.float32, device=DEV)
    P_all = torch.tensor(crop_proj(inp.K, ev["boxes"]), device=DEV)
    tgt = torch.tensor(ev["tgt"], device=DEV); val = torch.tensor(ev["val"], device=DEV)
    LEVER = 0.25  # m, converts yaw (rad) into a displacement at the object's extent
    for it in range(iters):
        opt.zero_grad()
        tot = 0.0
        perm = frames
        for k in range(0, len(perm), bs):
            sub = torch.tensor(perm[k:k + bs], device=DEV)
            gi = gidx[sub]
            T = fm.to_T_torch(gy[gi], ga[gi], gb[gi], c)
            sil = render_sil(rend, Tcw[None] @ T, P_all[sub])
            l = seg_iou_loss(sil, tgt[sub], val[sub]).sum()
            l.backward()
            tot += l.item()
        # temporal smoothness on the per-frame sequence
        y, a, b = gy[gidx], ga[gidx], gb[gidx]
        acc = lambda x: x[2:] - 2 * x[1:-1] + x[:-2]
        ls = lam_acc * ((acc(y) * LEVER / 0.002) ** 2 + (acc(a) / 0.002) ** 2 + (acc(b) / 0.002) ** 2).mean() * F
        ls.backward()
        opt.step()
        if verbose and (it % 50 == 0 or it == iters - 1):
            log(f"it {it} data {tot / len(perm):.4f} smooth {ls.item() / F:.4f}")
    with torch.no_grad():
        y, a, b = gy[gidx].cpu().numpy().astype(np.float64), ga[gidx].cpu().numpy().astype(np.float64), \
            gb[gidx].cpu().numpy().astype(np.float64)
    return y, a, b
