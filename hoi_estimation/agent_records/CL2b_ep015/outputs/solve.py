#!/usr/bin/env python
"""Trajectory-level re-estimation of the object pose for one monocular HOI video.

Pipeline (see REPORT.md for the reasoning and validation):
  1. Load the baseline (CARI4D) world poses, map them to the camera frame.
  2. Build per-frame evidence: square crops around the SAM2 object mask (image, object mask, person-free mask).
  3. Per-frame render-and-compare refinement (nvdiffrast) of the yaw about the sign's vertical (mesh z) axis and of
     the image-plane position; tilt and camera depth are kept at the baseline values (tilt/depth from silhouettes
     proved biased for this thin A-frame object, see REPORT.md).
  4. Segment phases automatically from the masks: initial static phase (object untouched on the floor), carried
     phase, final static phase (object back on the floor).
  5. Static phases: ONE pose each (chordal mean rotation of the refined per-frame rotations, median image
     position, baseline median depth).
  6. Carried phase: robust SE(3) smoothing of the refined observations (Huber data term weighted by the visible
     fraction of the object, constant-velocity prior; smoothing strength chosen by hold-out cross-validation),
     hard-tied to the static poses at the phase boundaries; heavily occluded frames are thereby interpolated.
  7. Depth re-anchoring: per phase, scale camera-frame positions along the viewing rays so that the median camera
     depth equals the baseline's median depth over the same frames.
  8. Map back to the world frame and save.
Deterministic: fixed seeds, no randomness in the optimisers' initialisation.
"""
import argparse, json, os, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_mesh, Renderer, load_camera, iter_video
from evidence import Evidence, refine_yaw
from trajectory import chordal_mean, detect_static_phases, smooth_se3
import h5py

# ---- constants (all chosen on this episode with ground-truth-free checks, see REPORT.md) ----
CROP_S = 320            # evidence crop resolution
REF_ITERS = 100         # Adam steps of the per-frame yaw refinement
PHOTO_LAMBDA = 0.3      # weight of the photometric term relative to the silhouette term
TILT_W = 1.0            # very strong prior: tilt stays at the baseline (silhouette tilt is biased, REPORT.md)
STATIC_IOU_THR = 0.9    # mask IoU (outside the person) with the first/last frames to count as "untouched/static"
LAMBDAS = [1.0, 3.0, 10.0, 30.0]  # candidate smoothing strengths, picked by hold-out CV
MIN_W = 0.02            # minimum data weight (fully occluded frames)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def rot_err_deg(A, B):
    M = np.einsum('nji,njk->nik', A[:, :3, :3], B[:, :3, :3])
    c = np.clip((np.trace(M, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(c))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--out", default="/work/outputs/object_poses_world.npz")
    ap.add_argument("--diag", default=None, help="optional path for diagnostics npz")
    a = ap.parse_args()
    t_start = time.time()
    np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

    K, Kp, Twc = load_camera(a.inputs)
    rec = np.load(f"{a.inputs}/reconstruction.npz")
    Tw0 = rec["T_world_mesh"].astype(np.float64)
    F = len(Tw0)
    Tc0 = np.linalg.inv(Twc)[None] @ Tw0                     # baseline camera-frame poses
    mesh = load_mesh(f"{a.inputs}/object_mesh.glb")
    r = Renderer(mesh, K)

    # ---- 2. evidence ----
    log("building evidence crops")
    ev = Evidence(a.inputs, iter_video(f"{a.inputs}/video.mp4"), r, K, Tc0, S=CROP_S)
    vis = np.clip(1.0 - ev.occ, 0.0, 1.0)

    # ---- 3. per-frame yaw / image-position refinement ----
    log("per-frame refinement")
    Tref = np.zeros_like(Tc0); iou0 = np.zeros(F); iou1 = np.zeros(F)
    for k in range(0, F, 60):
        ids = np.arange(k, min(F, k + 60))
        T, iou, _, _, _ = refine_yaw(r, ev, ids, Tc0[ids], iters=REF_ITERS, lam=PHOTO_LAMBDA, tilt_w=TILT_W)
        _, i0, _, _, _ = refine_yaw(r, ev, ids, Tc0[ids], iters=0, lam=PHOTO_LAMBDA, tilt_w=TILT_W)
        Tref[ids] = T; iou1[ids] = iou; iou0[ids] = i0
    # frames without usable evidence (object (almost) invisible) keep the baseline observation
    no_ev = (ev.area < 200) | (vis < 0.1)
    Tref[no_ev] = Tc0[no_ev]
    log("refinement done; mean |dyaw| %.2f deg" % rot_err_deg(Tref, Tc0).mean())

    # ---- 4. phases ----
    f = h5py.File(f"{a.inputs}/masks.h5", "r")
    s_end, e_start = detect_static_phases(f["object"], f["person"], thr=STATIC_IOU_THR)
    log(f"phases: static [0,{s_end})  carried [{s_end},{e_start})  static [{e_start},{F})")

    # ---- 5. static poses ----
    def static_pose(rng):
        w = np.maximum(vis[rng], MIN_W)
        Rm = chordal_mean(Tref[rng, :3, :3], w)
        t = Tref[rng, :3, 3]
        u = np.median(t[:, 0] / t[:, 2]); v = np.median(t[:, 1] / t[:, 2])
        z = np.median(Tc0[rng, 2, 3])                         # baseline depth (kept)
        T = np.eye(4); T[:3, :3] = Rm; T[:3, 3] = [u * z, v * z, z]
        return T
    fixed = {}
    if s_end > 0:
        TA = static_pose(np.arange(0, s_end)); fixed.update({k: TA for k in range(s_end)})
    if e_start < F:
        TB = static_pose(np.arange(e_start, F)); fixed.update({k: TB for k in range(e_start, F)})

    # ---- 6. carried phase smoothing ----
    w = np.maximum(vis ** 2, MIN_W); w[no_ev] = MIN_W
    # hold-out CV for the smoothing strength (carried phase only, well-visible frames)
    car = np.arange(s_end, e_start)
    ho = np.zeros(F, bool); ho[car[2::5]] = True; ho &= (w > 0.8)
    cv = []
    for lam in LAMBDAS:
        w2 = w.copy(); w2[ho] = 0
        Ts = smooth_se3(Tref, w2, fixed=fixed, lam_rot=lam, lam_tr=lam, iters=1500)
        er = rot_err_deg(Ts[ho], Tref[ho]).mean(); et = np.linalg.norm(Ts[ho, :3, 3] - Tref[ho, :3, 3], axis=1).mean()
        cv.append(er / 6.0 + et / 0.04)                       # same normalisation as the smoother's data term
        log(f"CV lam={lam}: held-out rot {er:.2f} deg, trans {100*et:.2f} cm")
    lam = LAMBDAS[int(np.argmin(cv))]
    log(f"chosen lambda {lam}")
    Ts = smooth_se3(Tref, w, fixed=fixed, lam_rot=lam, lam_tr=lam, iters=2500)

    # ---- 7. depth re-anchoring per phase (median camera z = baseline median z) ----
    phases = [(0, s_end), (s_end, e_start), (e_start, F)]
    for p0, p1 in phases:
        if p1 <= p0: continue
        s = np.median(Tc0[p0:p1, 2, 3]) / np.median(Ts[p0:p1, 2, 3])
        Ts[p0:p1, :3, 3] *= s                                  # move along the viewing rays
        log(f"phase [{p0},{p1}) depth scale {s:.5f}")

    # ---- 8. world frame ----
    Tw = Twc[None] @ Ts
    Tw[:, 3] = [0, 0, 0, 1]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, T_world_mesh=Tw.astype(np.float64))
    log(f"saved {a.out}  ({time.time()-t_start:.0f}s)")
    if a.diag:
        np.savez(a.diag, Tc0=Tc0, Tref=Tref, Ts=Ts, w=w, vis=vis, iou0=iou0, iou1=iou1, phases=np.array(phases),
                 lam=lam, cv=np.array(cv))


if __name__ == "__main__":
    main()
