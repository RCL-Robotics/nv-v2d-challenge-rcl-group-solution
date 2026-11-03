"""Trajectory-level refinement of the object pose for one monocular HOI video.

python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz [--validate]

Method (see REPORT.md):
 1. phase segmentation (static / moving) from photometric stability of the object region;
 2. global symmetry-flip hypothesis check (rotations about the mesh's principal axes) by mask IoU;
 3. joint render-and-compare fit of ALL frames (nvdiffrast silhouette soft-IoU vs. SAM2 object mask,
    person-occluded pixels ignored), one shared pose per static phase, per-frame poses in moving phases
    tied by second-difference smoothness, weak priors toward the baseline, tilt prior (feet on floor)
    for static phases;
 4. depth re-anchoring: per phase, camera-frame centres are scaled along their viewing rays so that the
    median depth equals the baseline's median depth over the same frames (baseline placement kept).
--validate additionally runs the same fit with every odd frame held out (weight 0) and reports held-out IoU.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_inputs, load_mesh, load_masks, read_video, cam_poses_from_world, \
    world_poses_from_cam, rot_angles_deg, H, W
from fit import SilFitter
from stages import static_segments, chordal_mean, rot_about_mesh_axis, depth_anchor
import cv2

# ---------------- constants (chosen on this episode, see REPORT.md) ----------------
SCALE = 0.5            # fitting resolution (768x576)
ITERS = 200
LR_R, LR_UV, LR_Z = 0.004, 0.0015, 0.001   # rad, normalised image units, log-depth per step
W_SMOOTH_R, W_SMOOTH_T = 100.0, 1000.0     # second-difference smoothness (rotation matrix entries, metres)
W_PRIOR_R, W_PRIOR_Z = 1.0, 10.0           # weak priors toward the baseline (per parameter group)
W_TILT_STATIC, W_TILT_MOVING = 2.0, 0.5    # per frame; feet-plane normal || floor normal (static: resting on floor)
FLIP_MARGIN = 0.05     # a global flip hypothesis must beat the baseline by this mean IoU to be adopted


def mesh_foot_normal(mesh):
    """Normal of the plane through the lowest point of each of the 4 legs (mesh frame, pointing to the
    feet = 'down' when upright; mesh +z is the stool's downward axis)."""
    v = np.asarray(mesh.vertices)
    c = (v.min(0) + v.max(0)) / 2
    feet = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            m = ((v[:, 0] - c[0]) * sx > 0.1) & ((v[:, 1] - c[1]) * sy > 0.1)
            q = v[m]
            feet.append(q[np.argmax(q[:, 2])])
    feet = np.array(feet)
    A = np.c_[feet[:, :2], np.ones(4)]
    coef = np.linalg.lstsq(A, feet[:, 2], rcond=None)[0]
    n = np.array([-coef[0], -coef[1], 1.0])
    return n / np.linalg.norm(n)


def build_groups(segs, Rb, pb):
    F = len(Rb)
    group = np.zeros(F, int); R0 = []; p0 = []; gstatic = []
    for a, b, st in segs:
        if st:
            group[a:b] = len(R0); R0.append(chordal_mean(Rb[a:b])); p0.append(np.median(pb[a:b], 0)); gstatic.append(True)
        else:
            for i in range(a, b):
                group[i] = len(R0); R0.append(Rb[i]); p0.append(pb[i]); gstatic.append(False)
    return group, np.array(R0), np.array(p0), np.array(gstatic)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='/work/inputs')
    ap.add_argument('--out', default='/work/outputs/object_poses_world.npz')
    ap.add_argument('--validate', action='store_true')
    args = ap.parse_args()
    t_start = time.time()
    np.random.seed(0); torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True

    K, Kp, Twc, rec, gp = load_inputs(args.inputs)
    mesh = load_mesh(args.inputs)
    Tw_base = rec['T_world_mesh'].astype(np.float64)
    F = len(Tw_base)
    Tb = cam_poses_from_world(Twc, Tw_base)

    # ---- data at fitting resolution
    Ks = K.copy(); Ks[:2] *= SCALE
    w, h = int(W * SCALE), int(H * SCALE)
    obj, per = load_masks(args.inputs, SCALE)
    valid = ~(per & ~obj)
    frames = read_video(os.path.join(args.inputs, 'video.mp4'), scale=SCALE)
    gray = np.stack([cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (5, 5), 0) for f in frames])
    del frames
    assert len(gray) == F

    # ---- 1. phases
    st, segs, pscore = static_segments(gray, obj, per)
    print('phases:', [(a, b, 'static' if s else 'moving') for a, b, s in segs])

    fz = SilFitter(mesh, Ks, w, h)
    Rb, pb = fz.split(Tb)

    # ---- 2. global flip hypotheses (rotation about the mesh centre), scored by mask IoU on every 4th frame
    sub = np.arange(0, F, 4)
    hyps = {'none': np.eye(3)}
    for ax in 'xyz':
        hyps[f'{ax}180'] = rot_about_mesh_axis(ax, 180)
    hyps['z90'] = rot_about_mesh_axis('z', 90); hyps['z270'] = rot_about_mesh_axis('z', 270)
    flip_iou = {k: float(fz.iou(Rb[sub] @ Rf, pb[sub], obj[sub], valid[sub]).mean()) for k, Rf in hyps.items()}
    print('flip hypothesis IoU:', {k: round(v, 4) for k, v in flip_iou.items()})
    best = max(flip_iou, key=flip_iou.get)
    flip = 'none' if flip_iou[best] < flip_iou['none'] + FLIP_MARGIN else best
    Rinit = Rb @ hyps[flip]
    print('adopted flip:', flip)

    # ---- 3. joint fit
    n_down_cam = -(Twc[:3, :3].T @ np.asarray(gp['plane'][:3]))  # floor normal in camera, pointing DOWN
    fn = mesh_foot_normal(mesh)
    group, R0, p0, gstatic = build_groups(segs, Rinit, pb)
    gsize = np.bincount(group, minlength=len(R0))
    w_tilt = np.where(gstatic, W_TILT_STATIC, W_TILT_MOVING) * gsize
    cfg = dict(iters=ITERS, lr_r=LR_R, lr_t=LR_UV, lr_z=LR_Z, w_smooth_r=W_SMOOTH_R, w_smooth_t=W_SMOOTH_T,
               w_prior_r=W_PRIOR_R, w_prior_z=W_PRIOR_Z, tilt_axis=fn, tilt_target=n_down_cam, w_tilt=w_tilt)

    def run(weights=None):
        Rg, pg, _ = fz.fit(R0, p0, group, obj, valid, weights=weights, **cfg)
        R = Rg[group]; p = pg[group]
        # ---- 4. depth re-anchoring per phase (move along viewing rays, keep image position)
        pa = p.copy(); facs = []
        for a, b, s in segs:
            pa[a:b], f = depth_anchor(p[a:b], pb[a:b]); facs.append(f)
        return R, p, pa, facs

    t0 = time.time()
    R, p_raw, p, facs = run()
    print('joint fit %.1fs' % (time.time() - t0))
    Tc = fz.join(R, p)
    Tw = world_poses_from_cam(Twc, Tc)
    np.savez(args.out, T_world_mesh=Tw.astype(np.float64))
    print('saved', args.out, Tw.shape)

    # ---------------- diagnostics / validation table ----------------
    ib = fz.iou(Rb, pb, obj, valid); io = fz.iou(R, p, obj, valid)
    lines = []
    def tilt_deg(Rc):
        return np.degrees(np.arccos(np.clip((Rc @ fn) @ n_down_cam, -1, 1)))
    hold = None
    if args.validate:
        wts = np.ones(F); wts[1::2] = 0
        Rv, _, pv, _ = run(wts)
        hold = fz.iou(Rv, pv, obj, valid)
    hdr = '| phase | frames | type | IoU base | IoU ours | held-out IoU base | held-out IoU ours | rot change med/max (deg) | depth-anchor factor | dz ours-base mean/median (cm) | frame-to-frame rot jitter base/ours (deg, median) | tilt vs floor base/ours (deg) |'
    lines.append(hdr); lines.append('|' + '---|' * 12)
    for k, (a, b, s) in enumerate(segs):
        fr = np.arange(a, b); odd = fr[fr % 2 == 1]
        ang = rot_angles_deg(Rb[a:b], R[a:b])
        dz = (Tc[a:b, 2, 3] - Tb[a:b, 2, 3]) * 100
        jb = np.median(rot_angles_deg(Rb[a:b - 1], Rb[a + 1:b])) if b - a > 1 else 0
        jo = np.median(rot_angles_deg(R[a:b - 1], R[a + 1:b])) if b - a > 1 else 0
        ho = '%.3f' % hold[odd].mean() if hold is not None else '-'
        lines.append(f'| {k} | {a}-{b - 1} | {"static" if s else "moving"} | {ib[a:b].mean():.3f} | {io[a:b].mean():.3f} | '
                     f'{ib[odd].mean():.3f} | {ho} | {np.median(ang):.2f}/{ang.max():.2f} | {facs[k]:.4f} | '
                     f'{dz.mean():.2f}/{np.median(dz):.2f} | {jb:.2f}/{jo:.2f} | '
                     f'{np.median(tilt_deg(Rb[a:b])):.2f}/{np.median(tilt_deg(R[a:b])):.2f} |')
    dz = (Tc[:, 2, 3] - Tb[:, 2, 3]) * 100
    ho = '%.3f' % hold[1::2].mean() if hold is not None else '-'
    lines.append(f'| all | 0-{F - 1} | | {ib.mean():.3f} | {io.mean():.3f} | {ib[1::2].mean():.3f} | {ho} | '
                 f'{np.median(rot_angles_deg(Rb, R)):.2f}/{rot_angles_deg(Rb, R).max():.2f} | | {dz.mean():.2f}/{np.median(dz):.2f} | | |')
    table = '\n'.join(lines)
    print(table)
    diag = dict(phases=[(int(a), int(b), bool(s)) for a, b, s in segs], flip_iou=flip_iou, flip=flip,
                depth_factors=[float(f) for f in facs], runtime_s=time.time() - t_start,
                raw_dz_cm_median=float(np.median((p_raw[:, 2] - pb[:, 2]) * 100)))
    base = os.path.splitext(args.out)[0]
    open(base + '_validation.md', 'w').write(table + '\n')
    json.dump(diag, open(base + '_diag.json', 'w'), indent=1)
    np.savez(base + '_perframe.npz', iou_base=ib, iou_ours=io, iou_holdout=hold if hold is not None else np.zeros(0),
             photo_score=pscore)
    print('total %.1fs' % (time.time() - t_start))


if __name__ == '__main__':
    main()
