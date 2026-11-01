#!/usr/bin/env python
"""Trajectory-level 6-DoF object pose recovery for one monocular HOI video (light-blue book episode).

Pipeline (see REPORT.md):
 1. Load video, SAM2 masks, baseline reconstruction, MoGe-2 aligned depth (static background depth).
 2. Phase segmentation (static on table / floor vs. carried), boundaries verified by eye, see PHASES.
 3. Camera-frame depth target: baseline z, robustly smoothed, constant inside static phases.
 4. Static phases: ONE constant pose per phase, fitted by render-and-compare (silhouette + shading-invariant
    texture term) jointly over many frames of the phase (coarse rotation grid + gradient refinement).
 5. Moving phases: per-frame candidates (baseline / baseline shifted to best explain the mask), Viterbi
    decoding, then joint trajectory optimisation (image terms + SE(3) smoothness + weak baseline prior)
    with the adjacent static poses as boundary conditions.
 6. Depth re-anchoring: per phase, camera-frame positions are scaled along the viewing rays so that the
    median z equals the baseline's median z over the same frames. Map back to world.
"""
import argparse, os, sys, time, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hoi_lib as H

torch.manual_seed(0); np.random.seed(0)
torch.backends.cudnn.deterministic = True

# Phase boundaries (inclusive frame ranges). Static phases were found from the frame-difference signal inside
# the object region (mean |I_t - I_{t-3}| over object-mask pixels not covered by the person; < ~3 grey levels
# = static) and the mask centroid / bbox stability, and then verified by eye on fixed-position crops:
#   A  0-79    book lies untouched on the table (person reaches for it ~50-80, lifts at ~83)
#   B  212-303 book lies on the floor under the table (bbox bottom settles at 212, hands rest on it, lift ~305)
#   C  405-466 book lies on the table again (lifted ~470)
#   D1 656-669 book on the floor; at 670-692 the hands slide/rotate it on the floor
#   D2 695-747 book on the floor after the slide (lifted ~750)
# Everything else is carried / manipulated (at the end, 830-900, the book is held behind the far table edge,
# partly occluded by the table top, and slowly lowered - not static).
# trajectory optimisation weights (chosen with the --holdout validation, see REPORT.md)
W_PHOTO_MOVING = 1.0   # moving phases: weight of the texture term relative to the silhouette term
LAM_PRIOR_R = 1.0      # moving phases: prior towards the decoded initial rotation (baseline)
STATIC_W_PRIOR = 4.0   # static phases: prior towards the pooled baseline rotation (~0.03 loss per 5 deg)
STATIC_W_FLAT = 5.0    # static phases: soft 'lies flat on the support plane' term
STATIC_YAW_STARTS = (-4.0, 0.0, 4.0)
STATIC = [('A', 0, 79), ('B', 212, 303), ('C', 405, 466), ('D1', 656, 669), ('D2', 695, 747)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='/work/inputs')
    ap.add_argument('--out', default='/work/outputs/object_poses_world.npz')
    ap.add_argument('--pipeline', default='/pipeline')
    ap.add_argument('--diag', default=None, help='optional path for diagnostics npz')
    ap.add_argument('--holdout', action='store_true',
                    help='validation mode: odd frames of moving segments get no image term (scored afterwards)')
    ap.add_argument('--lam_prior_r', type=float, default=LAM_PRIOR_R)
    ap.add_argument('--w_photo', type=float, default=W_PHOTO_MOVING)
    args = ap.parse_args()
    t_start = time.time()
    K, Twc, rec = H.load_inputs(args.inputs)
    Tw_base = rec['T_world_mesh'].astype(np.float64)
    F = len(Tw_base)
    Tb = H.mesh_to_cam(Twc, Tw_base)
    frames = H.load_video(f'{args.inputs}/video.mp4')
    assert len(frames) == F
    import h5py, cv2
    with h5py.File(f'{args.inputs}/masks.h5') as f:
        OM = f['object'][:]; PM = f['person'][:]
    # static background depth (median of MoGe-2 aligned depth over the episode): used only as an occluder
    # (table top) test, rendered object pixels behind it are not penalised
    with h5py.File(f'{args.pipeline}/depth/moge2/aligned.h5') as fa:
        ds = [cv2.imdecode(np.frombuffer(fa['aligned/front_stereo_camera_left'][i], np.uint8), cv2.IMREAD_UNCHANGED)
              for i in range(0, F, 10)]
    bg = (np.median(np.stack(ds), 0) / 1000.0).astype(np.float32)
    del ds
    print(f'loaded inputs in {time.time() - t_start:.1f}s', flush=True)

    # projected hand keypoints (MHR right hand 21-41, left hand 42-62) -> person pixels near the hands are
    # treated as occluders of the object, the rest of the body as (partial) evidence of object absence
    kp = rec['human_keypoints_world'][:, 21:63].astype(np.float64)
    kc = np.einsum('ij,fkj->fki', np.linalg.inv(Twc)[:3, :3], kp) + np.linalg.inv(Twc)[:3, 3]
    hand_uv = np.einsum('ij,fkj->fki', K, kc)
    hand_uv = hand_uv[..., :2] / hand_uv[..., 2:3]
    R = H.MeshRenderer(f'{args.inputs}/object_mesh.glb')
    obs = H.Obs(frames, OM, PM, bg, K, hand_uv=hand_uv)
    # second scorer with the plain criterion (all person pixels ignored) used for the static accept/reject test
    fit_plain = H.PoseFitter(R, H.Obs(frames, OM, PM, bg, K), K)
    # support-plane normal in the camera frame (released floor plane; the table top is assumed parallel to it -
    # the baseline's static table poses are flat w.r.t. the floor normal to <1 deg)
    plane = np.array(json.load(open(f'{args.inputs}/ground_plane.json'))['plane'])
    n_cam = np.linalg.inv(Twc)[:3, :3] @ plane[:3]
    fit = H.PoseFitter(R, obs, K)

    is_static = np.zeros(F, bool)
    for _, a, b in STATIC:
        is_static[a:b + 1] = True

    # ---- 3. depth target: smoothed baseline camera z (robust: median prefilter), constant in static phases
    zb = Tb[:, 2, 3]
    import scipy.ndimage as ndi
    zmed = ndi.median_filter(zb, size=9, mode='nearest')
    zt = H.smooth_series(zmed, np.ones(F), lam=20.0, groups=[(a, b) for _, a, b in STATIC])
    T_init = Tb.copy()
    T_init[:, :3, 3] *= (zt / zb)[:, None]          # move along the viewing ray to the smoothed depth

    T_out = T_init.copy()
    diag = {}
    # ---- 4. static phases
    for name, a, b in STATIC:
        # pooled baseline pose of the phase (chordal mean rotation, median position), made flat on the support
        # plane, then refined jointly on up to 16 least-occluded frames with priors towards it
        T_pool = H.mean_pose(T_init[a:b + 1])
        T_pool[:3, 3] *= zt[a] / T_pool[2, 3]
        T0 = H.flatten_pose(T_pool, n_cam, R.verts_np.mean(0))
        # multi-start (yaw -4/0/+4 deg about the support normal), small steps: the GPU rasteriser gradients are
        # not bit-deterministic, a single start occasionally ends in a poor local minimum
        fr = H.pick_frames(PM, OM, range(a, b + 1), 12)
        p = K @ T0[:3, 3]
        crop = obs.crop_for(0, (p[0] / p[2], p[1] / p[2]))
        c0 = R.verts_np.mean(0)
        starts = []
        for ang in STATIC_YAW_STARTS:
            g = H.axis_angle(n_cam, np.radians(ang))
            Tsd = T0.copy(); Tsd[:3, :3] = g @ T0[:3, :3]
            Tsd[:3, 3] = T0[:3, 3] + T0[:3, :3] @ c0 - Tsd[:3, :3] @ c0
            starts.append(Tsd)
        nS = len(starts)
        Tf, terms = fit.fit(np.stack(starts), np.tile(fr, nS), pid=np.repeat(np.arange(nS), len(fr)), iters=120,
                            lr_r=0.01, lr_t=0.002, crops=[crop] * (len(fr) * nS), T_prior=True,
                            w_prior=STATIC_W_PRIOR, flat_n=n_cam, w_flat=STATIC_W_FLAT)
        Ls = (terms['sil'] + 0.3 * (1 - terms['ncc']) + H.W_CHAM * terms['cham']).reshape(nS, len(fr)).mean(1)
        Ts = Tf[int(np.argmin(Ls))]
        Ts[:3, 3] *= zt[a] / Ts[2, 3]
        # accept the refined pose only if it explains ALL frames of the phase better than the pooled baseline
        # pose (chordal-mean rotation, median position; not flattened) under the plain criterion: IoU gain > 0.01 and texture R^2 not lower by more than 0.01
        allf = np.arange(a, b + 1)
        allf = allf[OM[allf].reshape(len(allf), -1).sum(1) > 50]
        s_fit = H.score_traj(fit_plain, np.repeat(Ts[None], len(allf), 0), allf)
        s_b = H.score_traj(fit_plain, np.repeat(T_pool[None], len(allf), 0), allf)
        accept = (s_fit['siou'].mean() - s_b['siou'].mean() > 0.01) and (s_fit['ncc'].mean() - s_b['ncc'].mean() > -0.01)
        print(f'   static {name}: pooled baseline IoU {s_b["siou"].mean():.3f} R2 {s_b["ncc"].mean():.3f} | refined IoU '
              f'{s_fit["siou"].mean():.3f} R2 {s_fit["ncc"].mean():.3f} -> {"refined" if accept else "pooled baseline"}')
        if not accept:      # fall back to the pooled baseline pose (not flattened)
            Ts = T_pool
        T_out[a:b + 1] = Ts
        diag[f'static_{name}'] = Ts
        print(f'static {name} {a}-{b}: start losses {np.round(Ls, 3)} rot change vs pooled baseline '
              f'{H.rot_angle_deg(T_pool[:3, :3], Ts[:3, :3]):.1f} deg  [{time.time() - t_start:.0f}s]', flush=True)

    # ---- 5. moving segments
    segs = []
    t = 0
    while t < F:
        if is_static[t]:
            t += 1; continue
        s = t
        while t < F and not is_static[t]:
            t += 1
        segs.append((s, t - 1))
    for a, b in segs:
        idx = np.arange(a, b + 1)
        Tc0 = T_init[idx]
        Tsh, Lb, L0, sh = H.shift_search(fit, Tc0, idx)
        if args.holdout:   # held-out frames must not see their own image: use the neighbours' shift
            for j in range(1, len(idx), 2):
                nb = [k for k in (j - 1, j + 1) if k < len(idx)]
                du, dv = sh[nb].mean(0)
                Tsh[j] = Tc0[j]; Tsh[j, 0, 3] += du * Tc0[j, 2, 3] / K[0, 0]; Tsh[j, 1, 3] += dv * Tc0[j, 2, 3] / K[1, 1]
                L0[j] = Lb[j] = 0.0
        cands = np.stack([Tc0, Tsh], 1)                       # (N,2,4,4)
        unary = np.stack([L0, Lb], 1)
        def trans(tt):
            d = np.linalg.norm(cands[tt][None, :, :3, 3] - cands[tt - 1][:, None, :3, 3], axis=-1)
            return (d / 0.05) ** 2
        path = H.viterbi(unary, trans)
        Tsel = cands[np.arange(len(idx)), path]
        ends = (T_out[a - 1] if a > 0 else None, T_out[b + 1] if b + 1 < F else None)
        img_mask = (np.arange(len(idx)) % 2 == 0) if args.holdout else None
        Tr = H.fit_traj(fit, Tsel, idx, iters=120, lr_r=0.02, lr_t=0.003, lam_acc_r=10.0, lam_acc_t=1000.0,
                        lam_prior_r=args.lam_prior_r, fixed_ends=ends, img_mask=img_mask, w_photo=args.w_photo)
        T_out[idx] = Tr
        print(f'moving {a}-{b}: shifted {path.mean():.2f} of frames, mean rot change vs baseline '
              f'{H.rot_angle_deg(Tb[idx, :3, :3], Tr[:, :3, :3]).mean():.1f} deg  [{time.time() - t_start:.0f}s]',
              flush=True)

    # ---- 6. depth re-anchoring per phase (static phases and moving segments)
    phases = [(n, a, b) for n, a, b in STATIC] + [(f'M{k}', a, b) for k, (a, b) in enumerate(segs)]
    for n, a, b in phases:
        s = np.median(Tb[a:b + 1, 2, 3]) / np.median(T_out[a:b + 1, 2, 3])
        T_out[a:b + 1, :3, 3] *= s
    Tw = H.cam_to_world(Twc, T_out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, T_world_mesh=Tw.astype(np.float64))
    if args.diag:
        np.savez(args.diag, T_cam=T_out, phases=np.array([(a, b) for _, a, b in phases]),
                 names=np.array([n for n, _, _ in phases]), **diag)
    print(f'done in {time.time() - t_start:.0f}s -> {args.out}; peak GPU memory '
          f'{torch.cuda.max_memory_allocated() / 1e9:.2f} GB')


if __name__ == '__main__':
    main()
