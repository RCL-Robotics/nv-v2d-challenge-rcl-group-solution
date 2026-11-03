"""Trajectory-level re-estimation of the object (contigo coffee mug) pose for every frame.

Pipeline (see REPORT.md):
 1. phases from the SAM2 object mask: static (mug resting on the table, constant pose) vs carried.
 2. static phases: temporal-median image/mask of the phase -> render-and-compare alignment (tilt, image
    position, scale) from several yaw initialisations -> yaw about the mug axis from the position of the
    'contigo' logo across the lower body -> depth re-anchored to the baseline median depth of the phase.
 3. carried phases: hand-rigid propagation of the static poses with the reconstructed grasping hand
    (forward from lift-off, backward from set-down, slerp-blended) -> per-frame candidates (hand-rigid pose,
    baseline / FoundationPose / CoCoNet rotations spun about the mug axis to the hand-rigid yaw) ->
    Viterbi (silhouette IoU data term + prior to hand-rigid + rotation-change transition) -> temporal
    smoothing -> depth re-anchored to the baseline per segment.
Run: python solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
"""
import argparse, os, sys, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from scipy.spatial.transform import Rotation as Rot, Slerp
from lib import *

torch.manual_seed(0); np.random.seed(0)


def rigid(A, B):
    """least-squares rigid transform (4x4) mapping point set A onto B"""
    ca, cb = A.mean(0), B.mean(0); a = A - ca; b = B - cb
    U, S, Vt = np.linalg.svd(a.T @ b); d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T; T = np.eye(4); T[:3, :3] = R; T[:3, 3] = cb - R @ ca
    return T


def upright(T, c0, u, n_up):
    """rotate T minimally (about the body centre) so that the mug axis (towards the lid) is along n_up"""
    a = T[:3, :3] @ u; ax = np.cross(a, n_up); s = np.linalg.norm(ax)
    out = T.copy()
    if s < 1e-9: return out
    ang = np.arctan2(s, a @ n_up); Rm = rot_about(ax / s, [ang])[0]
    out[:3, :3] = Rm @ T[:3, :3]; cc = T[:3, :3] @ c0 + T[:3, 3]; out[:3, 3] = cc - out[:3, :3] @ c0
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--inputs', default='/work/inputs')
    ap.add_argument('--out', default='/work/outputs/object_poses_world.npz')
    ap.add_argument('--debug_dir', default=None)
    args = ap.parse_args(); t_start = time.time(); inp = args.inputs
    log = {}

    m, v, f, uv, tex = load_mesh(f'{inp}/object_mesh.glb')
    cam, K, Twc, rec = load_inputs(inp); Tcw = np.linalg.inv(Twc)
    c0, u = body_axis(v)
    B = Tcw[None] @ rec['T_world_mesh'].astype(np.float64)          # baseline, camera frame
    FP = rec['T_cam_mesh_foundationpose'].astype(np.float64); CO = rec['T_cam_mesh_coconet'].astype(np.float64)
    kp = np.einsum('ij,fkj->fki', Tcw[:3, :3], rec['human_keypoints_world'].astype(np.float64)) + Tcw[:3, 3]
    F = len(B)
    gp = json.load(open(f'{inp}/ground_plane.json'))['plane']
    n_up = Tcw[:3, :3] @ np.array(gp[:3]); n_up /= np.linalg.norm(n_up)
    if n_up[1] > 0: n_up = -n_up          # 'up' = against gravity (camera y is down)

    def centre(T): return T[..., :3, :3] @ c0 + T[..., :3, 3]

    def set_centre(T, p):
        T = T.copy(); T[:3, 3] = p - T[:3, :3] @ c0; return T

    def anchor(Ts, z_ref):
        """move the body centres along their viewing rays so that median z equals z_ref"""
        Ts = Ts.copy(); cz = np.median(centre(Ts)[:, 2]); s = z_ref / cz
        for i in range(len(Ts)): Ts[i] = set_centre(Ts[i], centre(Ts[i]) * s)
        return Ts

    # ---------------- data ----------------
    crops = Crops(inp, 160)
    C = dict(img=(crops.img * 255).astype(np.uint8), om=crops.om, pm=crops.pm, off=crops.off)
    R = Renderer(v, f, uv, tex)
    L = logo_texture(v, f, uv, tex, c0, u).astype(np.float32)
    RL = Renderer(v, f, uv, np.repeat(L[..., None], 3, 2))

    # ---------------- 1. phases ----------------
    tr = mask_tracks(inp)
    phases = static_phases(tr)
    log['static_phases'] = phases
    print('static phases', phases)

    # ---------------- 2. static poses ----------------
    frames_all = read_video(f'{inp}/video.mp4')
    f5 = h5py.File(f'{inp}/masks.h5', 'r')
    med = dict(img=[], om=[], pm=[], off=[])
    for (a, b) in phases:
        x0, y0 = np.median(crops.off[a:b], 0).astype(int)
        ims, oks, oms = [], [], []
        for i in range(a, b):
            om = f5['object'][i][y0:y0 + 160, x0:x0 + 160]; pm = f5['person'][i][y0:y0 + 160, x0:x0 + 160]
            ok = ~(pm & ~om)
            ims.append(np.where(ok[..., None], frames_all[i][y0:y0 + 160, x0:x0 + 160].astype(np.float32), np.nan))
            oks.append(ok); oms.append(om & ok)
        with np.errstate(all='ignore'):
            md = np.nanmedian(np.stack(ims), 0)
        oks = np.stack(oks); frac = np.stack(oms).sum(0) / np.maximum(oks.sum(0), 1)
        med['img'].append(np.nan_to_num(md[..., ::-1]).astype(np.uint8)); med['om'].append(frac > 0.5)
        med['pm'].append(oks.sum(0) == 0); med['off'].append([x0, y0])
    MD = {k: np.array(vv) for k, vv in med.items()}; MD['off'] = MD['off'].astype(np.float64)
    del frames_all
    Fi_med = Fitter(R, K, MD, c0)
    T_static = []; log['static'] = []
    for k, (a, b) in enumerate(phases):
        Tm = B[(a + b) // 2].copy(); Tm = set_centre(Tm, np.median(centre(B[a:b]), 0))
        Tu = upright(Tm, c0, u, n_up)
        inits = spin(Tu, c0, u, np.radians(np.arange(0, 360, 20)))
        Tf, iou, ncc = Fi_med.fit(inits, [[k]] * len(inits), iters=80, free_z=True)
        tilt = np.degrees(np.arccos(np.clip(np.einsum('nij,j->ni', Tf[:, :3, :3], u) @ n_up, -1, 1)))
        score = iou - 1.0 * (tilt > 10)
        T_al = Tf[int(np.argmax(score))]
        res = measure_logo_az(RL, K, T_al, c0, u, MD['img'][k], MD['om'][k], MD['off'][k])
        if res is not None:
            az, ofr, resid = res
            T_k = spin(T_al, c0, u, np.radians([az - handle_az(T_al)]))[0]
        else:
            az, ofr, resid = None, None, None; T_k = T_al
        # translation-only realignment after the yaw change (z free for the image alignment)
        T_k, _, _ = Fi_med.fit(T_k[None], [[k]], iters=40, free_z=True, free_rot=False)
        T_k = T_k[0]
        z_ref = np.median(centre(B[a:b])[:, 2])
        dz_img = centre(T_k)[2] - z_ref
        T_k = set_centre(T_k, centre(T_k) * z_ref / centre(T_k)[2])
        T_static.append(T_k)
        log['static'].append(dict(phase=[a, b], logo_az=az, logo_frac=ofr, logo_resid=resid,
                                  tilt_deg=float(np.degrees(np.arccos(np.clip((T_k[:3, :3] @ u) @ n_up, -1, 1)))),
                                  image_fit_dz_cm=float(dz_img * 100), align_iou=float(iou.max()),
                                  baseline_az_median=float(np.median(handle_az(B[a:b])))))
        print('static', (a, b), log['static'][-1])

    out = B.copy()
    for k, (a, b) in enumerate(phases): out[a:b] = T_static[k]

    # ---------------- 3. carried segments ----------------
    Fi = Fitter(R, K, C, c0)
    hands = [slice(21, 42), slice(42, 63)]
    segs = []
    if phases[0][0] > 0: segs.append((None, 0))
    for k in range(len(phases) - 1): segs.append((k, k + 1))
    if phases[-1][1] < F: segs.append((len(phases) - 1, None))
    log['carry'] = []
    for (ka, kb) in segs:
        a = phases[ka][1] - 1 if ka is not None else 0           # last static frame (anchor) or start
        b = phases[kb][0] if kb is not None else F - 1           # first static frame (anchor) or end
        fr = np.arange(a, b + 1); N = len(fr)
        # hand-rigid prediction
        P = np.zeros((N, 4, 4))
        ref_t = a if ka is not None else b; ref_T = T_static[ka] if ka is not None else T_static[kb]
        hs = min(hands, key=lambda q: np.linalg.norm(kp[ref_t, q].mean(0) - centre(ref_T)))
        for j, t in enumerate(fr):
            Tf_ = rigid(kp[a, hs], kp[t, hs]) @ T_static[ka] if ka is not None else None
            Tb_ = rigid(kp[b, hs], kp[t, hs]) @ T_static[kb] if kb is not None else None
            if Tf_ is None: P[j] = Tb_; continue
            if Tb_ is None: P[j] = Tf_; continue
            w = (t - a) / max(b - a, 1)
            Rm = Slerp([0, 1], Rot.from_matrix([Tf_[:3, :3], Tb_[:3, :3]]))(w).as_matrix()
            P[j] = np.eye(4); P[j][:3, :3] = Rm
            P[j] = set_centre(P[j], (1 - w) * centre(Tf_) + w * centre(Tb_))
        # candidates
        cands = []
        for j, t in enumerate(fr):
            cs = [P[j]]
            for src in (B, FP, CO):
                Ty, _ = yaw_match(src[t], P[j][:3, :3], c0, u)
                cs.append(set_centre(Ty, centre(B[t])))
            cands.append(np.stack(cs))
        cands = np.stack(cands); Cn = cands.shape[1]
        iou = iou_scores(R, K, C, cands.reshape(-1, 4, 4), np.repeat(fr, Cn)).reshape(N, Cn)
        prior = geodesic_deg(cands[:, :, :3, :3], P[:, None, :3, :3])
        data = -20 * iou + prior / 15.0
        start = None; end = None
        if ka is not None: start = np.where(np.arange(Cn) == 0, 0, 1e3)   # anchored to the static pose
        if kb is not None: end = np.where(np.arange(Cn) == 0, 0, 1e3)
        trans = lambda t: geodesic_deg(cands[t - 1][:, None, :3, :3], cands[t][None, :, :3, :3]) / 5.0
        path = viterbi(data, trans, start, end)
        seq = cands[np.arange(N), path]
        # temporal smoothing (rotation: windowed quaternion mean; centre: windowed mean), ends kept
        q = Rot.from_matrix(seq[:, :3, :3]).as_quat()
        for j in range(1, N): q[j] *= np.sign(q[j] @ q[j - 1]) or 1
        p = centre(seq); hw = 3; sm = seq.copy()
        for j in range(N):
            lo, hi = max(0, j - hw), min(N, j + hw + 1)
            qm = q[lo:hi].mean(0); qm /= np.linalg.norm(qm)
            sm[j] = np.eye(4); sm[j][:3, :3] = Rot.from_quat(qm).as_matrix(); sm[j] = set_centre(sm[j], p[lo:hi].mean(0))
        if ka is not None: sm[0] = seq[0]
        if kb is not None: sm[-1] = seq[-1]
        # joint image refinement of the carried sequence (silhouette + appearance, smooth, depth fixed,
        # end poses = static poses kept). Validated on held-out (odd) frames: IoU +0.014..0.024, NCC +0.06..0.08
        fix = (ka is not None) or (kb is not None)
        sm, _, _, _ = fit_track(Fi, sm, fr, iters=150, lr_r=0.02, lr_t=0.002, w_smooth=5.0, w_prior=0.01, fix_ends=fix)
        if ka is not None: sm[0] = seq[0]
        if kb is not None: sm[-1] = seq[-1]
        # depth re-anchoring of the interior frames to the baseline over the same frames
        inner = slice(1 if ka is not None else 0, N - 1 if kb is not None else N)
        if len(fr[inner]) > 0:
            sm[inner] = anchor(sm[inner], np.median(centre(B[fr[inner]])[:, 2]))
            out[fr[inner]] = sm[inner]
        log['carry'].append(dict(frames=[int(a), int(b)], hand='R' if hs.start == 21 else 'L',
                                 choice_counts=np.bincount(path, minlength=Cn).tolist(),
                                 iou_mean_candidates=iou.mean(0).round(3).tolist(),
                                 iou_mean_chosen=float(iou[np.arange(N), path].mean())))
        print('carry', log['carry'][-1])

    # ---------------- 4. depth (along the viewing ray) ----------------
    # Default: depth of every phase = baseline median depth over the same frames (done above).
    # A static phase is corrected only if two independent physical cues agree within 2.5 cm and the
    # correction is larger than their disagreement:
    #   table : support-surface cue, RANSAC plane of the MoGe-2 metric depth of the table top around the mug,
    #           intersected with the viewing ray of the mug's bottom centre (reliable if >=10 frames, IQR<=4 cm)
    #   hand  : hand-contact cue, depth shift minimising handle-centre <-> grasping-hand distance over the
    #           contact frames at the phase boundaries (reliable if >=15 frames, residual <= 4 cm)
    #   rel   : same table spot as an already corrected phase (mask bottom row within 3 px, lower-body column
    #           within 10 px) -> relative depth from the free-depth image fits of the two phases (the absolute
    #           object-size depth is biased, the difference between phases is not)
    Kp = np.array(cam['K_pipeline']); rng = np.random.default_rng(0)
    hb = ((v - c0) @ u).min(); bot = c0 + u * hb; hv = v[v[:, 0] > 0.035].mean(0)
    A = h5py.File('/pipeline/depth/moge2/aligned.h5', 'r')['aligned/front_stereo_camera_left']
    cpm = rec['contact_prob_right_left'].max(1)
    cues = []
    for k, (a, b) in enumerate(phases):
        T = T_static[k]; zc = centre(T)[2]; pb = T[:3, :3] @ bot + T[:3, 3]
        zs = []
        for i in range(a, b, 2):
            z = table_cue_z(A[i], f5['object'][i], f5['person'][i], Kp, K, pb, n_up, rng)
            if z is not None: zs.append(z + (zc - pb[2]))
        tab = None
        if len(zs) >= 10 and np.subtract(*np.percentile(zs, [75, 25])) <= 0.04: tab = float(np.median(zs) - zc)
        frc = np.array([i for i in range(max(0, a - 10), min(F, b + 10)) if cpm[i] > 0.5], int)
        hand = None
        if len(frc) >= 15:
            Tseq = np.stack([out[i] for i in frc]); hs_, hr_ = hand_cue_shift(Tseq, frc, kp, c0, hv)
            if hr_ <= 0.04: hand = hs_
        cues.append(dict(table=tab, table_n=len(zs), table_iqr=float(np.subtract(*np.percentile(zs, [75, 25]))) if zs else None,
                         hand=hand, hand_n=int(len(frc)), z_anchor=float(zc),
                         z_img=float(zc + log['static'][k]['image_fit_dz_cm'] / 100), rel=None, applied=0.0, rule=''))
    decided = {}
    for k, c in enumerate(cues):
        if c['table'] is not None and c['hand'] is not None:
            dis = abs(c['table'] - c['hand']); mean = (c['table'] + c['hand']) / 2
            if dis <= 0.025 and abs(mean) > dis:
                decided[k] = mean; c['rule'] = 'table+hand'
    changed = True
    while changed:
        changed = False
        for k, c in enumerate(cues):
            if k in decided or c['rule'] == 'kept': continue
            nbs = [j for j in decided if abs(np.nanmedian(tr[phases[j][0]:phases[j][1], 0]) - np.nanmedian(tr[phases[k][0]:phases[k][1], 0])) <= 3
                   and abs(np.nanmedian(tr[phases[j][0]:phases[j][1], 1]) - np.nanmedian(tr[phases[k][0]:phases[k][1], 1])) <= 10]
            if not nbs: continue
            j = min(nbs, key=lambda q: abs(q - k))
            c['rel'] = float(cues[j]['z_anchor'] + decided[j] + (c['z_img'] - cues[j]['z_img']) - c['z_anchor'])
            vals = {n: c[n] for n in ('table', 'hand', 'rel') if c[n] is not None}
            pairs = [(abs(vals[x] - vals[y]), x, y) for i_, x in enumerate(vals) for y in list(vals)[i_ + 1:]]
            ok = [(d_, x, y) for d_, x, y in pairs if d_ <= 0.025 and abs((vals[x] + vals[y]) / 2) > d_]
            if ok:
                d_, x, y = min(ok); decided[k] = (vals[x] + vals[y]) / 2; c['rule'] = f'{x}+{y}'
            else:
                c['rule'] = 'kept'
            changed = True
    corr_static = np.array([decided.get(k, 0.0) for k in range(len(phases))])
    for k, c in enumerate(cues):
        c['applied'] = float(corr_static[k]); c['rule'] = c['rule'] or 'kept'
    # per-frame correction: static phases constant, carried segments linearly interpolated between phases
    key_t = np.array([(a + b - 1) / 2 for a, b in phases]); corr = np.zeros(F)
    for k, (a, b) in enumerate(phases): corr[a:b] = corr_static[k]
    for t in range(F):
        if any(a <= t < b for a, b in phases): continue
        prev = [k for k, (a, b) in enumerate(phases) if b <= t]; nxt = [k for k, (a, b) in enumerate(phases) if a > t]
        if prev and nxt:
            ka_, kb_ = prev[-1], nxt[0]; ta, tb = phases[ka_][1] - 1, phases[kb_][0]
            w = (t - ta) / (tb - ta); corr[t] = (1 - w) * corr_static[ka_] + w * corr_static[kb_]
        else: corr[t] = corr_static[prev[-1]] if prev else corr_static[nxt[0]]
    for t in range(F):
        cz = centre(out[t]); out[t] = set_centre(out[t], cz * (cz[2] + corr[t]) / cz[2])
    # report: hand cue inside carried segments after correction (should be ~0 if consistent)
    for c_ in log['carry']:
        a, b = c_['frames']; frc = np.array([i for i in range(a + 1, b) if cpm[i] > 0.5], int)
        if len(frc) >= 10:
            hs_, hr_ = hand_cue_shift(out[frc], frc, kp, c0, hv); c_['hand_cue_after_cm'] = 100 * hs_; c_['hand_resid_cm'] = 100 * hr_
    log['depth_cues'] = cues
    print('depth cues'); [print(' ', phases[k], {kk: (round(vv * 100, 1) if isinstance(vv, float) and kk in ('table', 'hand', 'rel', 'applied', 'table_iqr') else vv) for kk, vv in c.items()}) for k, c in enumerate(cues)]

    Tw = Twc[None] @ out
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, T_world_mesh=Tw.astype(np.float64))
    log['runtime_s'] = time.time() - t_start
    log['torch_max_mem_GB'] = torch.cuda.max_memory_allocated() / 1e9
    with open(os.path.splitext(args.out)[0] + '_log.json', 'w') as fh: json.dump(log, fh, indent=1, default=float)
    print('done in %.1fs' % log['runtime_s'])


if __name__ == '__main__':
    main()
