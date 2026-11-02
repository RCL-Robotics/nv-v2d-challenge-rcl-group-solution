#!/usr/bin/env python
"""Recover the 6-DoF trajectory of the iron in inputs/video.mp4.

Pipeline (see REPORT.md):
  1. SAM2 video segmentation of the iron (the released / pipeline object masks segment the tripod, not the iron).
  2. Crops around the iron; person mask from masks.h5 marks occluded pixels.
  3. Physically constrained render-and-compare: the iron rests on the table top with its soleplate (flat phase) or
     its heel rest (standing phase); per frame only yaw + position on the table plane are free. Keyframe hypothesis
     search over yaw, Viterbi decoding over time, dense refinement, static-phase pooling, temporal smoothing.
     The stand-up transition is fitted with free 6-DoF poses with a table-contact penalty.
  4. Depth anchoring: the whole trajectory is scaled along the viewing rays so that the table plane matches the
     physically estimated table plane (casters on the released floor plane + table rim).
"""
import argparse
import json
import os
import sys
import time
import pickle

import numpy as np
import torch
import cv2
import h5py

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import load_inputs, load_mesh, rot_angle_deg, project  # noqa: E402
from fitcore import Fitter  # noqa: E402
from hyp import SOLE_N, HEEL_N, soft_iou_loss, init_translation, fit_batch  # noqa: E402
from constrained import PlaneModel, fit_plane_model  # noqa: E402
from prep import build, compute_touch  # noqa: E402
from stages import run_all, depth_cues, apply_scale  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

# ---------------------------------------------------------------- constants (verified by eye, see REPORT.md)
# Crop of the video that contains the round table and the iron during the whole episode.
CROP = (520, 380, 1000, 780)  # x0, y0, x1, y1 (full-res pixels)
# SAM2 box prompt on the iron at frame 0 (crop coordinates). Found by eye: the iron is at x 685-757, y 612-657
# in the full frame. (The pipeline's Grounding-DINO prompt picked the tripod behind the table.)
IRON_BOX_F0 = [160, 228, 242, 282]
# Table geometry measured by eye on frame 0 (full-res pixels): floor contact points of the five casters of the
# table base, and the extreme points of the table-top ellipse (left, right, back, front of the top surface).
CASTERS_PX = [(746, 911), (659, 879), (839, 879), (701, 830), (801, 835)]
TABLE_RIM_PX = [(632, 655), (848, 652), (740, 632), (740, 674)]
KEY_STEP = 5


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------- 1. segmentation
def run_sam2(video, F, workdir):
    from v2d.sam2.lib.sam2_utils import build_sam2_video_predictor_low_mem
    fdir = os.path.join(workdir, "crops")
    os.makedirs(fdir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    x0, y0, x1, y1 = CROP
    for i in range(F):
        ok, f = cap.read()
        cv2.imwrite(f"{fdir}/{i:05d}.jpg", f[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 95])
    pred = build_sam2_video_predictor_low_mem(config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
                                              ckpt_path="/weights/sam2/sam2.1_hiera_large.pt")
    M = np.zeros((F, y1 - y0, x1 - x0), bool)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        st = pred.init_state(fdir, offload_video_to_cpu=True, offload_state_to_cpu=True)
        pred.add_new_points_or_box(st, frame_idx=0, obj_id=1, box=IRON_BOX_F0)
        for fi, ids, logits in pred.propagate_in_video(st):
            M[fi] = (logits[0, 0] > 0).cpu().numpy()
    del pred
    torch.cuda.empty_cache()
    return M


# ---------------------------------------------------------------- geometry helpers
def floor_in_cam(Twc, plane, shift=0.0):
    """Floor plane in camera frame, n_c.x + d_c = 0, n_c pointing up. `shift` lowers the floor by that many m."""
    n_c = Twc[:3, :3].T @ plane[:3]
    d_c = plane[3] + plane[:3] @ Twc[:3, 3] + shift
    return n_c, d_c


def table_geometry(K, n_c, d_c):
    """Table-top centre (camera frame) from casters on the floor plane + vertical column + rim ellipse."""
    from scipy.optimize import least_squares
    Ki = np.linalg.inv(K)

    def floorpt(u, v):
        r = Ki @ [u, v, 1]
        return -d_c / (n_c @ r) * r
    cast = np.array([floorpt(*p) for p in CASTERS_PX])
    sol = least_squares(lambda p: np.r_[np.linalg.norm(cast - p[:3], axis=1) - p[3], 10 * (n_c @ p[:3] + d_c)],
                        np.r_[cast.mean(0), 0.35])
    c = sol.x[:3]
    rim = np.array(TABLE_RIM_PX, float)
    e1 = np.cross(n_c, [0, 0, 1.0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n_c, e1)
    ang = np.linspace(0, 2 * np.pi, 720, endpoint=False)

    def res2(p):
        h, r = p
        P = c + h * n_c + r * (np.cos(ang)[:, None] * e1 + np.sin(ang)[:, None] * e2)
        uv = project(K, P)
        return np.r_[uv[:, 0].min() - rim[0, 0], uv[:, 0].max() - rim[1, 0], uv[:, 1].min() - rim[2, 1],
                     uv[:, 1].max() - rim[3, 1]]
    s2 = least_squares(res2, [0.9, 0.4])
    h, r = s2.x
    return dict(center=c + h * n_c, height=h, radius=r, caster_center=c, caster_R=sol.x[3],
                rim_resid_px=s2.fun.tolist())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--out", default="/work/outputs/object_poses_world.npz")
    ap.add_argument("--cache", default=None, help="dev only: directory for stage caches")
    ap.add_argument("--workdir", default="/tmp/iron_solve")
    args = ap.parse_args()
    T0 = time.time()
    inp = args.inputs
    os.makedirs(args.workdir, exist_ok=True)
    cache = args.cache

    def cached(name, fn):
        if cache:
            p = os.path.join(cache, name + ".pkl")
            if os.path.exists(p):
                log("cache hit", name)
                return pickle.load(open(p, "rb"))
            r = fn()
            os.makedirs(cache, exist_ok=True)
            pickle.dump(r, open(p, "wb"))
            return r
        return fn()

    K, Kp, Twc, rec = load_inputs(inp)
    F = json.load(open(f"{inp}/episode.json"))["frames"]
    plane = np.array(json.load(open(f"{inp}/ground_plane.json"))["plane"], np.float64)
    # The true frame-0 human's lowest vertices (shoe soles) lie below the released floor plane; the floor used for
    # the table geometry is lowered so that they touch it (physical cue; 0 if they are above the plane).
    Vref = np.load(f"{inp}/frame0_reference_human.npz")["vertices_world"].astype(np.float64)
    floor_shift = float(max(0.0, -np.min(Vref @ plane[:3] + plane[3])))
    n_c, d_c = floor_in_cam(Twc, plane, floor_shift)
    mesh = load_mesh(f"{inp}/object_mesh.glb")
    V = np.asarray(mesh.vertices, np.float64)
    cm = mesh.bounds.mean(0)
    fit = Fitter(mesh, K)

    # 1. masks
    log("SAM2 segmentation")
    M = cached("sam2", lambda: run_sam2(f"{inp}/video.mp4", F, args.workdir))
    np.savez_compressed(os.path.join(os.path.dirname(args.out), "iron_masks_crop.npz"), mask=M, crop=np.array(CROP))
    x0, y0, x1, y1 = CROP

    def full(i):
        o = np.zeros((1152, 1536), bool)
        o[y0:y1, x0:x1] = M[i]
        return o
    log("crops")
    data = cached("crops", lambda: build(f"{inp}/video.mp4", full, f"{inp}/masks.h5", F))
    data["touch"] = cached("touch", lambda: compute_touch(M, CROP, f"{inp}/masks.h5"))

    STAGES = {}
    table = table_geometry(K, n_c, d_c)
    STAGES["table"] = {k: np.asarray(v).tolist() for k, v in table.items()}
    STAGES["floor_shift_m"] = floor_shift
    res = run_all(fit, data, K, n_c, d_c, V, cm, F, rec, Twc, table, cached, log, STAGES)
    T_fit = res["T_fit"]

    # 4. depth anchoring: scale k along the viewing rays (rotations and image positions unchanged)
    table["rim_px"] = TABLE_RIM_PX
    cues = depth_cues(T_fit, res["sole"], res["X_fit"], res["s_fit"], data, K, Twc, rec, M, CROP, table, F, log)
    phys = [cues[k] for k in ("table_geometry", "moge_table", "right_hand") if k in cues]
    k_final = float(np.median(phys))
    zc = np.median((np.einsum("fij,j->fi", T_fit[:, :3, :3], cm) + T_fit[:, :3, 3])[:, 2])
    cues["median_depth_m"] = {k: float(zc * v) for k, v in cues.items() if isinstance(v, float)}
    cues["k_final"] = k_final
    cues["median_depth_final_m"] = float(zc * k_final)
    STAGES["depth_cues"] = cues
    log("depth cues (median iron depth, m):", cues["median_depth_m"], "-> k = %.4f" % k_final)
    T_cam = apply_scale(T_fit, k_final, cm)
    np.save(os.path.splitext(args.out)[0] + "_Tfit.npy", T_fit)

    T_world = Twc[None] @ T_cam
    np.savez(args.out, T_world_mesh=T_world.astype(np.float64))
    info = {k: v for k, v in STAGES.items() if isinstance(v, (int, float, str, list, dict))}
    info["runtime_s"] = time.time() - T0
    json.dump(info, open(os.path.splitext(args.out)[0] + "_info.json", "w"), indent=1, default=float)
    log("done", args.out, "runtime %.1f s" % (time.time() - T0))


if __name__ == "__main__":
    main()
