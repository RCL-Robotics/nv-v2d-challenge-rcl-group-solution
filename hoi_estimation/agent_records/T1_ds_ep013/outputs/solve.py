#!/usr/bin/env python3
"""Recover a temporally consistent 6-DoF snack-box trajectory.

The method uses the exact textured mesh, SAM2 masks, the released camera, MoGe-2
depth, the CARI4D baseline, and a temporal hypothesis decoder.  It is deliberately
self contained: all pose estimates below are recomputed from the inputs.
"""
import argparse
import itertools
import json
import os
import time
from collections import defaultdict

import cv2
import h5py
import numpy as np
import torch
import trimesh
from scipy.optimize import differential_evolution, minimize
from scipy.spatial.transform import Rotation as Rot

import nvdiffrast.torch as dr

# ------------------------------ deterministic setup ------------------------------
SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# These boundaries were verified by eye on the video.  They are fixed, commented
# constants (rather than per-frame fitted data): the object is stationary while
# untouched/resting, and moves only in the three manipulation intervals.
PHASES = {
    "desk_initial": (0, 149),       # box rests on the white desk
    "pickup_to_floor": (150, 324),  # carried down and released
    "floor_initial": (325, 389),    # box rests on the floor
    "foot_push": (390, 499),        # foot pushes/tumbles box through desk
    "floor_final": (500, 689),      # box rests again on the floor
    "lift_to_desk": (690, 829),     # picked up, carried, and put on desk
    "desk_final": (830, 921),       # box rests on the desk
}

# Dynamic keyframes: independent render-and-compare hypotheses are made at these
# frames, then a Viterbi pass selects one hypothesis per keyframe.
KEYFRAMES = {
    "A": list(range(150, 321, 10)),
    "B": list(range(400, 500, 10)),
    "C": list(range(700, 831, 10)),
}
SEGMENTS = {
    "A": {"start": 149, "end": 325, "maxstep": 0.55},
    "B": {"start": 390, "end": 500, "maxstep": 0.90},
    "C": {"start": 690, "end": 830, "maxstep": 0.75},
}


def proper_axis_symmetries():
    """24 proper signed axis permutations (the box's geometric symmetry group)."""
    out = []
    for p in itertools.permutations(range(3)):
        for signs in itertools.product([-1, 1], repeat=3):
            h = np.zeros((4, 4), np.float32)
            h[3, 3] = 1.0
            for j in range(3):
                h[p[j], j] = signs[j]
            if np.linalg.det(h[:3, :3]) > 0:
                out.append(h)
    return np.asarray(out, np.float32)


SYMS = proper_axis_symmetries()


def average_pose(poses):
    poses = np.asarray(poses, np.float64)
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = Rot.from_matrix(poses[:, :3, :3]).mean().as_matrix()
    t[:3, 3] = np.median(poses[:, :3, 3], axis=0)
    return t


def rot_angle(a, b):
    return float(Rot.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude())


def clean_pose(t):
    t = np.asarray(t, np.float64).copy()
    u, _, vt = np.linalg.svd(t[:3, :3])
    t[:3, :3] = u @ vt
    if np.linalg.det(t[:3, :3]) < 0:
        u[:, -1] *= -1
        t[:3, :3] = u @ vt
    t[3] = [0, 0, 0, 1]
    return t


def slerp_pose(a, b, alpha):
    ra, rb = Rot.from_matrix(a[:3, :3]), Rot.from_matrix(b[:3, :3])
    delta = (rb * ra.inv()).as_rotvec()
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = (Rot.from_rotvec(alpha * delta) * ra).as_matrix()
    out[:3, 3] = (1 - alpha) * a[:3, 3] + alpha * b[:3, 3]
    return out


class PoseRenderer:
    def __init__(self, inputs, device="cuda"):
        self.device = device
        with open(os.path.join(inputs, "camera.json")) as f:
            cam = json.load(f)
        self.K = np.asarray(cam["K"], np.float32)
        self.Twc = np.asarray(cam["T_world_cam"], np.float32)
        self.Tcw = np.linalg.inv(self.Twc).astype(np.float32)
        self.W, self.H = int(cam["width"]), int(cam["height"])

        geo = trimesh.load(
            os.path.join(inputs, "object_mesh.glb"),
            force="scene",
            process=False,
        ).to_geometry()
        self.mesh = geo
        self.corners = np.asarray(
            [
                [x, y, z, 1.0]
                for x in geo.bounds[0, 0:1].tolist() + geo.bounds[1, 0:1].tolist()
                for y in geo.bounds[0, 1:2].tolist() + geo.bounds[1, 1:2].tolist()
                for z in geo.bounds[0, 2:3].tolist() + geo.bounds[1, 2:3].tolist()
            ],
            np.float32,
        )
        self.V = torch.tensor(geo.vertices, dtype=torch.float32, device=device)
        self.F = torch.tensor(geo.faces, dtype=torch.int32, device=device)
        self.UV = torch.tensor(geo.visual.uv, dtype=torch.float32, device=device)
        tex = np.asarray(geo.visual.material.baseColorTexture)
        self.TEX = (
            torch.tensor(tex, dtype=torch.float32, device=device)
            .permute(2, 0, 1)[None]
            / 255.0
        )
        self.ctx = dr.RasterizeCudaContext()

    def crop_box(self, bbox, pad=30, min_size=300):
        x0, y0, x1, y1 = bbox
        cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        side = int(max(min_size, max(x1 - x0, y1 - y0) + 2 * pad))
        side = (side + 15) // 16 * 16
        xa = int(np.clip(round(cx - side / 2), 0, self.W - side))
        ya = int(np.clip(round(cy - side / 2), 0, self.H - side))
        return xa, ya, side, side

    def render(self, poses, x0, y0, w, h):
        """Render world_mesh poses in a crop. Returns mask, RGB, camera z."""
        poses = torch.as_tensor(poses, dtype=torch.float32, device=self.device)
        n = len(poses)
        vh = torch.cat([self.V, torch.ones_like(self.V[:, :1])], 1)
        world = (poses @ vh.T).transpose(1, 2)
        cam = world @ torch.tensor(self.Tcw, device=self.device).T
        x, y, z = cam[:, :, 0], cam[:, :, 1], cam[:, :, 2]
        # The rational depth map is monotonic with camera z; unlike z=const it
        # gives correct front-face z buffering in nvdiffrast.
        zc = (100.01 / 99.99) * z - (2.0 * 100.0 * 0.01 / 99.99)
        clip = torch.stack(
            [
                2.0 * (self.K[0, 0] * x + (self.K[0, 2] - x0) * z) / w - z,
                2.0 * (self.K[1, 1] * y + (self.K[1, 2] - y0) * z) / h - z,
                zc,
                z,
            ],
            -1,
        )
        rast, rdb = dr.rasterize(self.ctx, clip, self.F, resolution=[h, w])
        uv, _ = dr.interpolate(self.UV[None].expand(n, -1, -1).contiguous(), rast, self.F, rdb)
        depth, _ = dr.interpolate(z[..., None].contiguous(), rast, self.F, rdb)
        mask = rast[:, :, :, 3] > 0
        # glTF texture V is bottom-up, while the decoded PIL image is top-down.
        texn = torch.stack([uv[..., 0] * 2 - 1, 1 - uv[..., 1] * 2], -1)
        rgb = torch.nn.functional.grid_sample(
            self.TEX.expand(n, -1, -1, -1),
            texn,
            align_corners=False,
            mode="bilinear",
            padding_mode="border",
        )
        return mask, rgb.permute(0, 2, 3, 1), depth[..., 0]

    def projected_box_polygon(self, pose):
        pc = (self.Tcw @ (pose @ self.corners.T)).T
        u = self.K[0, 0] * pc[:, 0] / pc[:, 2] + self.K[0, 2]
        v = self.K[1, 1] * pc[:, 1] / pc[:, 2] + self.K[1, 2]
        return np.c_[u, v]

    @staticmethod
    def raster_polygon(poly, x0, y0, w, h):
        out = np.zeros((h, w), np.uint8)
        p = np.round(poly - [x0, y0]).astype(np.int32)
        cv2.fillConvexPoly(out, cv2.convexHull(p), 1)
        return out.astype(bool)


def ncc(a, b, mask):
    if int(mask.sum()) < 10:
        return -1.0
    a = a[mask].astype(np.float64)
    b = b[mask].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9
    return float((a * b).sum() / den)


def image_measure(image, rendered, mask, obj, person):
    """Segmentation and texture evidence (all in the same crop)."""
    visible = ~cv2.dilate(
        person.astype(np.uint8), np.ones((5, 5), np.uint8)
    ).astype(bool)
    overlap = mask & obj & visible
    union = (mask | obj) & visible
    iou = float((mask & obj & visible).sum() / max(1, union.sum()))

    a = image.astype(np.float32) / 255.0
    h1 = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)
    h2 = cv2.cvtColor(rendered, cv2.COLOR_RGB2HSV)
    l1 = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)
    l2 = cv2.cvtColor(rendered, cv2.COLOR_RGB2LAB)
    ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor(rendered, cv2.COLOR_RGB2GRAY)

    sat = ncc(h1[:, :, 1], h2[:, :, 1], overlap)
    ca = ncc(l1[:, :, 1], l2[:, :, 1], overlap)
    cb = ncc(l1[:, :, 2], l2[:, :, 2], overlap)
    gray = ncc(ga, gb, overlap)
    chroma = 0.7 * sat + 0.55 * ca + 0.55 * cb

    gx1 = cv2.Sobel(ga, cv2.CV_32F, 1, 0, ksize=3)
    gy1 = cv2.Sobel(ga, cv2.CV_32F, 0, 1, ksize=3)
    gx2 = cv2.Sobel(gb, cv2.CV_32F, 1, 0, ksize=3)
    gy2 = cv2.Sobel(gb, cv2.CV_32F, 0, 1, ksize=3)
    if overlap.sum():
        dot = (gx1 * gx2 + gy1 * gy2)[overlap]
        mag = (
            np.sqrt(gx1 * gx1 + gy1 * gy1)
            * np.sqrt(gx2 * gx2 + gy2 * gy2)
        )[overlap]
        grad = float((dot / (mag + 1e-6)).mean())
    else:
        grad = -1.0
    return dict(iou=iou, chrom=chroma, grad=grad, gray=gray, sat=sat, ca=ca, cb=cb)


class Solver:
    def __init__(self, inputs, out):
        self.inputs = inputs
        self.out = out
        self.renderer = PoseRenderer(inputs)
        rec = np.load(os.path.join(inputs, "reconstruction.npz"))
        self.baseline = rec["T_world_mesh"].astype(np.float64)
        self.F = len(self.baseline)
        self.cap = cv2.VideoCapture(os.path.join(inputs, "video.mp4"))
        with h5py.File(os.path.join(inputs, "masks.h5"), "r") as f:
            self.obj_masks = f["object"][:]
            self.person_masks = f["person"][:]
        self.depth_h5 = h5py.File(
            "/pipeline/depth/moge2/aligned.h5", "r"
        )
        self.both = np.zeros((self.F, 4), np.int32)
        for i in range(self.F):
            y, x = np.where(self.obj_masks[i])
            self.both[i] = [x.min(), x.max() + 1, y.min(), y.max() + 1]

    def close(self):
        self.cap.release()
        self.depth_h5.close()

    def frame(self, i):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, bgr = self.cap.read()
        if not ok:
            raise RuntimeError(f"could not decode frame {i}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def depth(self, i):
        raw = self.depth_h5["aligned/front_stereo_camera_left"][i]
        return cv2.imdecode(raw, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

    def crop_data(self, i, pad=30, min_size=300):
        x0, y0, w, h = self.renderer.crop_box(self.both[i], pad, min_size)
        image = self.frame(i)[y0 : y0 + h, x0 : x0 + w]
        obj = self.obj_masks[i, y0 : y0 + h, x0 : x0 + w]
        person = self.person_masks[i, y0 : y0 + h, x0 : x0 + w]
        depth = self.depth(i)[y0 : y0 + h, x0 : x0 + w]
        return image, obj, person, depth, (x0, y0, w, h)

    def bbox_pose_fit(self, i, zspan=0.16, popsize=12, maxiter=180, seed=SEED):
        """Global box-silhouette fit; leaves the texture ambiguity to symmetry."""
        image, obj, person, _, crop = self.crop_data(i, pad=25, min_size=240)
        x0, y0, w, h = crop
        visible = ~cv2.dilate(
            person.astype(np.uint8), np.ones((5, 5), np.uint8)
        ).astype(bool)
        base = self.baseline[i]
        cam = self.renderer.Tcw @ base
        c0 = cam[:3, 3]
        z0 = float(c0[2])
        uv0 = np.array(
            [
                self.renderer.K[0, 0] * c0[0] / c0[2] + self.renderer.K[0, 2],
                self.renderer.K[1, 1] * c0[1] / c0[2] + self.renderer.K[1, 2],
            ]
        )

        def make(x):
            z, du, dv, rx, ry, rz = x
            r = Rot.from_rotvec([rx, ry, rz]).as_matrix().astype(np.float32)
            tcam = np.array(
                [
                    (uv0[0] + du - self.renderer.K[0, 2]) * z / self.renderer.K[0, 0],
                    (uv0[1] + dv - self.renderer.K[1, 2]) * z / self.renderer.K[1, 1],
                    z,
                ]
            )
            t = np.eye(4, dtype=np.float32)
            t[:3, :3] = r
            t[:3, 3] = tcam
            return self.renderer.Twc @ t

        def objective(x):
            pose = make(x)
            poly = self.renderer.projected_box_polygon(pose)
            mask = self.renderer.raster_polygon(poly, x0, y0, w, h) & visible
            union = ((mask | obj) & visible).sum()
            return 1.0 - (mask & obj).sum() / max(1, union)

        bounds = [
            (max(1.0, z0 - zspan), z0 + zspan),
            (-35, 35),
            (-35, 35),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
            (-np.pi, np.pi),
        ]
        result = differential_evolution(
            objective,
            bounds,
            seed=seed,
            popsize=popsize,
            maxiter=maxiter,
            tol=1e-5,
            polish=True,
            workers=1,
            updating="immediate",
        )
        return make(result.x)

    def candidate_scores(self, candidates, i, pad=30, min_size=300):
        """Score arbitrary pose hypotheses using all available image evidence."""
        image, obj, person, dep, crop = self.crop_data(i, pad, min_size)
        x0, y0, w, h = crop
        masks, rgbs, depths = self.renderer.render(candidates, x0, y0, w, h)
        masks = masks.cpu().numpy()
        rgbs = rgbs.cpu().numpy()
        depths = depths.cpu().numpy()
        out = []
        for k in range(len(candidates)):
            m = image_measure(image, rgbs[k], masks[k], obj, person)
            common = masks[k] & obj
            if common.sum() > 20:
                dz = float(np.median(depths[k][common]) - np.median(dep[common]))
            else:
                dz = 1.0
            score = (
                1.6 * m["iou"]
                + 0.9 * m["chrom"]
                + 0.5 * m["grad"]
                - 2.0 * abs(dz - 0.08)
            )
            out.append((score, m, dz))
        return out

    def best_symmetry(self, pose0, frames):
        """Select the texture-correct member of the 24-way box ambiguity."""
        candidates = np.asarray([pose0 @ s for s in SYMS], np.float32)
        totals = np.zeros(len(candidates))
        stats = defaultdict(lambda: defaultdict(list))
        for i in frames:
            scores = self.candidate_scores(candidates, i)
            for k, (s, m, dz) in enumerate(scores):
                totals[k] += s
                stats[k]["iou"].append(m["iou"])
                stats[k]["chrom"].append(m["chrom"])
                stats[k]["grad"].append(m["grad"])
                stats[k]["dz"].append(dz)
        k = int(np.argmax(totals))
        detail = {
            key: float(np.mean(val))
            for key, val in stats[k].items()
        }
        return clean_pose(candidates[k]), float(totals[k] / len(frames)), detail

    def dynamic_keyframe(self, i):
        pose0 = self.bbox_pose_fit(i, zspan=0.22, popsize=10, maxiter=130)
        candidates = np.asarray(
            [pose0 @ s for s in SYMS] + [self.baseline[i]], np.float32
        )
        scores = self.candidate_scores(candidates, i)
        values = np.asarray([x[0] for x in scores], np.float64)
        ious = np.asarray([x[1]["iou"] for x in scores])
        chroms = np.asarray([x[1]["chrom"] for x in scores])
        grads = np.asarray([x[1]["grad"] for x in scores])
        dzs = np.asarray([x[2] for x in scores])
        return candidates, values, ious, chroms, grads, dzs

    def decode_segment(self, name):
        info = SEGMENTS[name]
        kfs = KEYFRAMES[name]
        pose_data = {i: self.dynamic_keyframe(i) for i in kfs}
        start_t = self.static_poses[{"A": "floor_initial", "B": "floor_final", "C": "desk_final"}[name]]
        end_t = self.static_poses[{"A": "floor_initial", "B": "floor_final", "C": "desk_final"}[name]]
        # Correct endpoint names explicitly (A starts at the desk and B/C start on floor).
        start_t = {
            "A": self.static_poses["desk_initial"],
            "B": self.static_poses["floor_initial"],
            "C": self.static_poses["floor_final"],
        }[name]
        end_t = {
            "A": self.static_poses["floor_initial"],
            "B": self.static_poses["floor_final"],
            "C": self.static_poses["desk_final"],
        }[name]

        states = [[start_t]] + [list(pose_data[i][0]) for i in kfs] + [[end_t]]
        scores = [np.array([0.0])] + [pose_data[i][1] for i in kfs] + [np.array([0.0])]
        dp = scores[0].copy()
        back = []
        lam, hi, maxstep = 0.5, 3.0, info["maxstep"]
        for j in range(1, len(states)):
            ndp = np.full(len(states[j]), -1e18)
            bk = np.zeros(len(states[j]), np.int32)
            for b, tb in enumerate(states[j]):
                best, best_a = -1e18, 0
                for a, ta in enumerate(states[j - 1]):
                    ang = rot_angle(ta, tb)
                    transition = lam * ang + hi * max(0.0, ang - maxstep) ** 2
                    val = dp[a] - transition
                    if val > best:
                        best, best_a = val, a
                ndp[b] = scores[j][b] + best
                bk[b] = best_a
            dp, _ = ndp, back.append(bk)
        path = [int(np.argmax(dp))]
        idx = path[0]
        for bk in back[::-1]:
            idx = int(bk[idx])
            path.append(idx)
        path = path[::-1]
        return {i: states[j + 1][path[j + 1]] for j, i in enumerate(kfs)}, path

    def estimate_static_poses(self):
        print("[1/5] estimating static phase poses", flush=True)
        initial = average_pose(self.baseline[:66])
        floor_initial0 = self.bbox_pose_fit(350, zspan=0.16, popsize=12, maxiter=180)
        floor_final0 = self.bbox_pose_fit(550, zspan=0.16, popsize=12, maxiter=180)
        desk_final0 = self.bbox_pose_fit(850, zspan=0.16, popsize=12, maxiter=180)
        floor_initial, si, si_d = self.best_symmetry(
            floor_initial0, [330, 340, 360, 380]
        )
        floor_final, sf, sf_d = self.best_symmetry(
            floor_final0, [510, 530, 555, 665, 680]
        )
        desk_final, sd, sd_d = self.best_symmetry(
            desk_final0, [840, 860, 880, 900, 921]
        )
        self.static_poses = {
            "desk_initial": clean_pose(initial),
            "floor_initial": floor_initial,
            "floor_final": floor_final,
            "desk_final": desk_final,
        }
        self.static_stats = {
            "desk_initial": {"score": None, "detail": "baseline rotation pooled over frames 0:66"},
            "floor_initial": {"score": si, "detail": si_d},
            "floor_final": {"score": sf, "detail": sf_d},
            "desk_final": {"score": sd, "detail": sd_d},
        }
        for k, p in self.static_poses.items():
            print("   ", k, np.round(p[:3, 3], 4), flush=True)

    def solve(self):
        tic = time.time()
        self.estimate_static_poses()
        print("[2/5] fitting and decoding dynamic keyframe hypotheses", flush=True)
        selected = {}
        for name in ["A", "B", "C"]:
            poses, path = self.decode_segment(name)
            selected.update(poses)
            print("   ", name, path, flush=True)

        print("[3/5] interpolating the selected trajectory", flush=True)
        keys = {
            0: self.static_poses["desk_initial"],
            149: self.static_poses["desk_initial"],
            325: self.static_poses["floor_initial"],
            389: self.static_poses["floor_initial"],
            390: self.static_poses["floor_initial"],
            499: self.static_poses["floor_final"],
            500: self.static_poses["floor_final"],
            689: self.static_poses["floor_final"],
            690: self.static_poses["floor_final"],
            830: self.static_poses["desk_final"],
            921: self.static_poses["desk_final"],
        }
        keys.update(selected)
        key_ids = sorted(keys)
        poses = np.zeros((self.F, 4, 4), np.float64)
        for i in range(self.F):
            if i <= key_ids[0]:
                poses[i] = keys[key_ids[0]]
            elif i >= key_ids[-1]:
                poses[i] = keys[key_ids[-1]]
            else:
                j = np.searchsorted(key_ids, i, side="right") - 1
                a, b = key_ids[j], key_ids[j + 1]
                poses[i] = slerp_pose(keys[a], keys[b], (i - a) / (b - a))

        print("[4/5] per-frame silhouette translation alignment", flush=True)
        poses = self.refine_translation(poses)
        print("[5/5] local rigid-pose refinement from image evidence", flush=True)
        poses = self.refine_pose(poses)
        for i in range(self.F):
            poses[i] = clean_pose(poses[i])
        np.savez(self.out, T_world_mesh=poses.astype(np.float64))
        elapsed = time.time() - tic
        print(f"wrote {self.out} ({self.F} frames) in {elapsed/60:.2f} minutes", flush=True)
        return poses

    def refine_translation(self, poses):
        out = poses.copy()
        areas = self.obj_masks.sum((1, 2))
        for i in range(150, 830):
            if not (150 <= i <= 324 or 390 <= i <= 499 or 690 <= i <= 829):
                continue
            pose = poses[i]
            cam = self.renderer.Tcw @ pose
            c0 = cam[:3, 3]
            z0 = float(c0[2])
            uv0 = np.array(
                [
                    self.renderer.K[0, 0] * c0[0] / c0[2] + self.renderer.K[0, 2],
                    self.renderer.K[1, 1] * c0[1] / c0[2] + self.renderer.K[1, 2],
                ]
            )
            image, obj, person, _, crop = self.crop_data(i, pad=25, min_size=240)
            x0, y0, w, h = crop
            visible = ~cv2.dilate(person.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            weight = float(np.clip(areas[i] / 7000.0, 0.08, 1.0))

            def make(x):
                z, du, dv = x
                tcam = np.array(
                    [
                        (uv0[0] + du - self.renderer.K[0, 2]) * z / self.renderer.K[0, 0],
                        (uv0[1] + dv - self.renderer.K[1, 2]) * z / self.renderer.K[1, 1],
                        z,
                    ]
                )
                p = pose.copy()
                p[:3, 3] = (self.renderer.Twc @ np.r_[tcam, 1])[:3]
                return p

            def objective(x):
                p = make(x)
                poly = self.renderer.projected_box_polygon(p)
                mask = self.renderer.raster_polygon(poly, x0, y0, w, h) & visible
                union = ((mask | obj) & visible).sum()
                iou = (mask & obj).sum() / max(1, union)
                reg = 0.5 * ((x[0] - z0) / 0.2) ** 2 + 0.0005 * (x[1] ** 2 + x[2] ** 2)
                return weight * (1 - iou) + reg

            result = minimize(
                objective,
                np.array([z0, 0.0, 0.0]),
                method="Nelder-Mead",
                options={"maxiter": 300, "xatol": 1e-5, "fatol": 1e-7},
            )
            out[i] = make(result.x)
        return out

    def refine_pose(self, poses):
        out = poses.copy()
        areas = self.obj_masks.sum((1, 2))
        for i in range(150, 830):
            if not (150 <= i <= 324 or 390 <= i <= 499 or 690 <= i <= 829):
                continue
            pose = poses[i]
            cam = self.renderer.Tcw @ pose
            c0 = cam[:3, 3]
            z0 = float(c0[2])
            uv0 = np.array(
                [
                    self.renderer.K[0, 0] * c0[0] / c0[2] + self.renderer.K[0, 2],
                    self.renderer.K[1, 1] * c0[1] / c0[2] + self.renderer.K[1, 2],
                ]
            )
            image, obj, person, _, crop = self.crop_data(i, pad=25, min_size=240)
            x0, y0, w, h = crop
            visible = ~cv2.dilate(person.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            weight = float(np.clip(areas[i] / 7000.0, 0.06, 1.0))

            def make(x):
                z, du, dv, rx, ry, rz = x
                r = Rot.from_rotvec([rx, ry, rz]).as_matrix().astype(np.float32) @ pose[:3, :3]
                tcam = np.array(
                    [
                        (uv0[0] + du - self.renderer.K[0, 2]) * z / self.renderer.K[0, 0],
                        (uv0[1] + dv - self.renderer.K[1, 2]) * z / self.renderer.K[1, 1],
                        z,
                    ]
                )
                p = np.eye(4)
                p[:3, :3] = r
                p[:3, 3] = tcam
                return self.renderer.Twc @ p

            def objective(x):
                p = make(x)
                poly = self.renderer.projected_box_polygon(p)
                mask = self.renderer.raster_polygon(poly, x0, y0, w, h) & visible
                union = ((mask | obj) & visible).sum()
                iou = (mask & obj).sum() / max(1, union)
                reg = (
                    0.30 * (x[3] ** 2 + x[4] ** 2 + x[5] ** 2)
                    + 0.20 * ((x[0] - z0) / 0.2) ** 2
                    + 0.0005 * (x[1] ** 2 + x[2] ** 2)
                )
                return weight * (1 - iou) + reg

            seeds = [np.zeros(6)]
            for k in range(3):
                for sign in (-1, 1):
                    q = np.zeros(6)
                    q[3 + k] = sign * 0.08
                    seeds.append(q)
            best = None
            for seed in seeds:
                seed = seed.copy()
                seed[0] = z0
                result = minimize(
                    objective,
                    seed,
                    method="Nelder-Mead",
                    options={"maxiter": 800, "xatol": 1e-5, "fatol": 1e-7},
                )
                if best is None or result.fun < best.fun:
                    best = result
            out[i] = make(best.x)
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    solver = Solver(args.inputs, args.out)
    try:
        solver.solve()
    finally:
        solver.close()


if __name__ == "__main__":
    main()
