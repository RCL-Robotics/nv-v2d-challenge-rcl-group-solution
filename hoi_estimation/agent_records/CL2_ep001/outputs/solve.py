"""Trajectory-level object pose estimation for one HOI video.

Pipeline (see REPORT.md):
 1. occluder masks: person (SAM2 masks from inputs) + the static bar stool (SAM2 box prompt, constant box)
 2. phase segmentation: start / end static phases from object-mask stability w.r.t. first / last frame
 3. static phases: ONE pose per phase, pooled render-and-compare over the phase from several hypotheses
    (baseline mean rotation, rolls about the long axis, 180 deg flips); free depth gives the silhouette scale bias
 4. carried phase: 8 hypotheses / frame (baseline & FoundationPose rotations x 4 rolls about the long axis),
    each refined by silhouette render-and-compare; scored with silhouette + appearance (redness / brightness)
 5. Viterbi over hypotheses (rotation / translation transition costs) -> initial trajectory
 6. joint refinement of the carried phase: silhouette terms + SE(3) smoothness, anchored to the static poses
 7. depth re-anchoring: camera-frame positions scaled along the viewing rays so that per phase median z equals
    the baseline median z (depth correction applied: 0)
Run: python solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import trimesh
from scipy.ndimage import median_filter, gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load_inputs, load_video, rot_angle  # noqa: E402
from fit import CropFitter, FrameData, project, box_from, refine, refine_grouped, losses, app_score, FloorTerm  # noqa: E402
from traj import viterbi, joint_refine, eval_frames  # noqa: E402

# ---------------------------------------------------------------- constants (with reasons)
# Box (full-res pixels) around the tall bar stool that occludes the vacuum in both rest phases; chosen by eye
# on frame 0 / last frame (static camera, the stool does not move: SAM2 masks of frame 0 and 746 have IoU 0.94).
CHAIR_BOX = np.array([625, 650, 835, 1005])
STATIC_IOU_THR = 0.93      # object-mask IoU w.r.t. the first / last frame to count as "at rest"
RES_HYP = 128              # crop resolution for hypothesis refinement
RES_FIT = 160              # crop resolution for static fit / joint refinement / scoring
SEED = 0
# Floor terms are implemented but disabled: at the baseline depth (which we keep, see REPORT) the floor contact
# and the image evidence disagree by several cm; enforcing contact at fixed depth distorted rotations / image fit
# (static-end mask IoU 0.43 -> 0.37). The floor is therefore only reported as a depth cue (validate.py).
USE_FLOOR = False


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def chair_mask(frames, obj_masks, ids):
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model = build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml", "/weights/sam2/sam2.1_hiera_large.pt", device="cuda")
        pred = SAM2ImagePredictor(model)
        out = np.zeros(frames.shape[1:3], bool)
        for i in ids:
            pred.set_image(frames[i][:, :, ::-1].copy())
            m, s, _ = pred.predict(box=CHAIR_BOX, multimask_output=False)
            out |= (m[0] > 0) & ~obj_masks[i]
        del model, pred
        torch.cuda.empty_cache()
        return out
    except Exception as e:  # pragma: no cover
        log("chair mask failed, continuing without:", e)
        return np.zeros(frames.shape[1:3], bool)


def static_phases(obj, per, thr=STATIC_IOU_THR):
    F = len(obj)
    sm = lambda m: cv2.resize(m.astype(np.uint8), (384, 288), interpolation=cv2.INTER_AREA) > 0
    O = np.stack([sm(o) for o in obj])
    P = np.stack([sm(p) for p in per])

    def prof(r):
        v = np.zeros(F)
        for i in range(F):
            vis = ~(P[i] | P[r])
            a, b = O[i] & vis, O[r] & vis
            v[i] = (a & b).sum() / max((a | b).sum(), 1)
        return median_filter(v, 5, mode="nearest")

    p0, p1 = prof(0), prof(F - 1)
    s1 = 0
    while s1 + 1 < F and p0[s1 + 1] >= thr:
        s1 += 1
    s2 = F - 1
    while s2 - 1 > s1 and p1[s2 - 1] >= thr:
        s2 -= 1
    return s1, s2, p0, p1  # static: [0, s1] and [s2, F-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--out", default="/work/outputs/object_poses_world.npz")
    ap.add_argument("--holdout", type=int, default=0, help="validation: drop image term on every k-th frame")
    args = ap.parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t_start = time.time()
    inp = args.inputs
    cam, K, Twc, rec = load_inputs(inp)
    Tcw = np.linalg.inv(Twc)
    Tb_world = rec["T_world_mesh"].astype(np.float64)
    Tb = Tcw @ Tb_world                                    # baseline, camera frame
    Tfp = rec["T_cam_mesh_foundationpose"].astype(np.float64)
    F = len(Tb)
    mesh_path = os.path.join(inp, "object_mesh.glb")

    log("loading video / masks")
    frames = load_video(inp, scale=1.0)
    import h5py
    with h5py.File(os.path.join(inp, "masks.h5"), "r") as f:
        OBJ = f["object"][:]
        PER = f["person"][:]
    CH = chair_mask(frames, OBJ, [0, F - 1])
    kern = np.ones((7, 7), np.uint8)
    ign = lambda i: cv2.dilate((PER[i] | CH).astype(np.uint8), kern) > 0

    fit = CropFitter(mesh_path, res=RES_FIT)
    corners = trimesh.bounds.corners(fit.mesh.bounds)

    def make_fd(ids, Tinit, R):
        boxes = [box_from(OBJ[i], project(K, T, corners), pad=0.15) if OBJ[i].any()
                 else box_from(np.zeros_like(OBJ[i]), project(K, T, corners), pad=0.15)
                 for i, T in zip(ids, Tinit)]
        return FrameData([frames[i] for i in ids], [OBJ[i] for i in ids], [ign(i) for i in ids], boxes, K, R)

    def U_of(E):
        return 1 - E[:, 0] + 2 * E[:, 1] + E[:, 2] + E[:, 3]

    Rb, pb = fit.to_rp(Tb)
    Rf, _ = fit.to_rp(Tfp)

    # floor: released plane + level of the true frame-0 human soles (signed distance of its lowest vertices);
    # a resting object touches the floor at that level, a carried one must not go below it.
    plane = np.array(json.load(open(os.path.join(inp, "ground_plane.json")))["plane"], np.float64)
    hv = np.load(os.path.join(inp, "frame0_reference_human.npz"))["vertices_world"].astype(np.float64)
    h0 = float(np.percentile(hv @ plane[:3] + plane[3], 0.1))
    fl_probe = FloorTerm(fit, plane, Twc, h0, 1.0, "contact")
    with torch.no_grad():
        dmin_base = fl_probe.dmin(torch.tensor(Rb, dtype=torch.float32, device="cuda"),
                                  torch.tensor(pb, dtype=torch.float32, device="cuda")).cpu().numpy()
    log(f"floor contact level h0 {h0 * 100:.1f} cm; baseline lowest point median {np.median(dmin_base) * 100:.1f} cm")

    # ------------------------------------------------------------ phases
    s1, s2, prof0, prof1 = static_phases(OBJ, PER)
    log(f"static phases: [0,{s1}] and [{s2},{F - 1}]; carried: [{s1 + 1},{s2 - 1}]")

    # ------------------------------------------------------------ static phases: one pose each
    # (a) silhouette scale (mask-size bias): free-depth fit of the baseline mean pose of each static phase.
    #     The masks are slightly bloated (SAM2 person mask vs. the true frame-0 human: ~3-5 % larger), so the
    #     silhouette prefers the object closer; one global factor absorbs this and is discarded at the end.
    sdata = []
    for (a, b) in [(0, s1), (s2, F - 1)]:
        ids = [i for i in range(a, b + 1, 3) if OBJ[i].sum() > 200]
        Rm0 = Rot.from_matrix(Rb[a:b + 1]).mean().as_matrix()
        pm0 = np.median(pb[a:b + 1], 0)
        T0 = fit.to_T(Rm0[None], pm0[None])[0]
        fd = make_fd(ids, [T0] * len(ids), RES_FIT)
        R1, p1, _ = refine_grouped(fit, Rm0[None], pm0[None], fd, np.zeros(len(ids), int), iters=100,
                                   fix_depth=False)
        sdata.append((a, b, ids, Rm0, pm0, fd, p1[0, 2] / pm0[2]))
        log(f"  static [{a},{b}] free-depth scale {p1[0, 2] / pm0[2]:.3f}")
    gscale = float(np.clip(np.mean([d[-1] for d in sdata]), 0.85, 1.0))
    log(f"global silhouette scale (mask-size bias) {gscale:.3f}")
    # (b) one pose per static phase, several hypotheses, depth fixed at the baseline median (x gscale)
    static = []
    for (a, b, ids, Rm0, pm0, fd, _) in sdata:
        # resting-contact term only if the baseline itself has the object on the floor in this phase (< 5 cm)
        rest = abs(np.median(dmin_base[a:b + 1]) - h0) < 0.05
        floor = None if not USE_FLOOR else FloorTerm(fit, plane, Twc, h0, 300.0, "contact") if rest else \
            FloorTerm(fit, plane, Twc, h0, 300.0, "above")
        log(f"  static [{a},{b}] floor term: {('contact' if rest else 'above') if USE_FLOOR else 'off'}")
        hyps = {f"rz{d}": Rm0 @ Rot.from_euler("z", d, degrees=True).as_matrix() for d in [0, -30, 30, 90, 180, 270]}
        hyps["fx"] = Rm0 @ Rot.from_euler("x", 180, degrees=True).as_matrix()
        hyps["fy"] = Rm0 @ Rot.from_euler("y", 180, degrees=True).as_matrix()
        best = None
        for name, Rh in hyps.items():
            R1, p1, _ = refine_grouped(fit, Rh[None], pm0[None], fd, np.zeros(len(ids), int), iters=100,
                                       fix_depth=True, scale=gscale, floor=floor)
            E = eval_frames(fit, fd, np.repeat(R1, len(ids), 0), np.repeat(p1, len(ids), 0), scale=gscale)
            sc = U_of(E).mean()
            log(f"  static [{a},{b}] hyp {name}: U {sc:.3f} iou {E[:, 0].mean():.3f} dR_base {rot_angle(R1[0], Rm0):.1f}")
            if best is None or sc < best[0]:
                best = (sc, name, R1[0], p1[0])
        Eb = eval_frames(fit, fd, Rb[ids], pb[ids], scale=gscale)
        sc, name, Rs, ps = best
        static.append(dict(a=a, b=b, R=Rs, p=ps, name=name, U=sc, U_base=U_of(Eb).mean()))
        log(f"  -> static [{a},{b}] chose {name} U {sc:.3f} (baseline per-frame U {U_of(Eb).mean():.3f})")

    # ------------------------------------------------------------ carried phase: hypotheses
    a, b = s1 + 1, s2 - 1
    ids = list(range(a, b + 1))
    z_s = gaussian_filter1d(median_filter(pb[:, 2], 9, mode="nearest"), 3, mode="nearest")
    p_init = pb.copy()
    p_init[:, 2] = z_s
    H = {}
    for src, Rs_ in [("b", Rb), ("f", Rf)]:
        for d in [0, 90, 180, 270]:
            H[f"{src}{d}"] = Rs_ @ Rot.from_euler("z", d, degrees=True).as_matrix()
    ks = list(H)
    Rh = {k: [] for k in ks}
    Ph = {k: [] for k in ks}
    Eh = {k: [] for k in ks}
    fit_h = CropFitter(mesh_path, res=RES_HYP)
    for c0 in range(0, len(ids), 140):
        cid = ids[c0:c0 + 140]
        fdh = make_fd(cid, Tb[cid], RES_HYP)
        fds = make_fd(cid, Tb[cid], RES_FIT)
        for k in ks:
            R1, p1, _ = refine(fit_h, H[k][cid], p_init[cid], fdh, iters=50, w_col=0.0, fix_depth=True,
                               lr_r=0.03, scale=gscale)
            R1 = np.nan_to_num(R1)
            E = eval_frames(fit, fds, R1, p1, scale=gscale)
            Rh[k].append(R1)
            Ph[k].append(p1)
            Eh[k].append(E)
        log(f"  hypotheses {c0 + len(cid)}/{len(ids)}")
    del fit_h
    R = np.stack([np.concatenate(Rh[k]) for k in ks], 1)
    P = np.stack([np.concatenate(Ph[k]) for k in ks], 1)
    U = np.nan_to_num(np.stack([U_of(np.concatenate(Eh[k])) for k in ks], 1), nan=9.0)
    path = viterbi(U, R, P)
    log("viterbi choices " + str({k: int((path == i).sum()) for i, k in enumerate(ks)}))
    R0 = R[np.arange(len(ids)), path]
    p0 = P[np.arange(len(ids)), path]

    # ------------------------------------------------------------ joint refinement
    fd = make_fd(ids, Tb[ids], RES_FIT)
    area = np.array([OBJ[i].sum() for i in ids], float)
    iw = np.clip(area / (0.3 * np.median(area)), 0, 1)   # tiny / missing masks carry little weight
    if args.holdout:
        iw[::args.holdout] = 0
    sA, sB = static
    Rj, pj = joint_refine(fit, fd, R0, p0, iw, np.repeat(sA["R"][None], 2, 0), np.repeat(sA["p"][None], 2, 0),
                          np.repeat(sB["R"][None], 2, 0), np.repeat(sB["p"][None], 2, 0), iters=150,
                          lam_acc_r=5.0, scale=gscale, floor=FloorTerm(fit, plane, Twc, h0, 300.0, "above") if USE_FLOOR else None, log=log)

    # ------------------------------------------------------------ assemble, depth re-anchoring
    Rall = Rb.copy()
    pall = pb.copy()
    for s in static:
        Rall[s["a"]:s["b"] + 1] = s["R"]
        pall[s["a"]:s["b"] + 1] = s["p"]
    Rall[a:b + 1] = Rj
    pall[a:b + 1] = pj
    phases = [("static_start", 0, s1), ("carried", a, b), ("static_end", s2, F - 1)]
    for name, x0, x1 in phases:
        zb = np.median(pb[x0:x1 + 1, 2])   # baseline median depth of the object centre in this phase
        zo = np.median(pall[x0:x1 + 1, 2])
        pall[x0:x1 + 1] *= zb / zo           # move along the viewing rays only
        log(f"depth re-anchor {name}: factor {zb / zo:.4f}")
    Tc = fit.to_T(Rall, pall)
    Tw = Twc @ Tc
    # report-style per-phase validation numbers (in-sample)
    if args.holdout:
        m = iw == 0
        Ej = eval_frames(fit, fd, Rj, pj, scale=gscale)
        Eb = eval_frames(fit, fd, Rb[a:b + 1], pb[a:b + 1], scale=gscale)
        Ev = eval_frames(fit, fd, R0, p0, scale=gscale)
        log(f"HOLDOUT ({m.sum()} frames): iou base {Eb[m, 0].mean():.3f} viterbi(in-sample) {Ev[m, 0].mean():.3f} "
            f"joint {Ej[m, 0].mean():.3f} | U base {U_of(Eb)[m].mean():.3f} joint {U_of(Ej)[m].mean():.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, T_world_mesh=Tw.astype(np.float64))
    np.savez(os.path.splitext(args.out)[0] + "_diag.npz", phases=np.array([[0, s1], [a, b], [s2, F - 1]]),
             gscale=gscale, prof0=prof0, prof1=prof1, path=path, U=U, ks=np.array(ks),
             static_names=np.array([s["name"] for s in static]))
    log(f"done in {time.time() - t_start:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
