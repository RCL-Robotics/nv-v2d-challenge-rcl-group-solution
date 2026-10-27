"""Recover the 6-DoF trajectory of the cane in inputs/video.mp4.

Pipeline (see REPORT.md):
 1. per-frame multi-hypothesis silhouette registration (baseline, its 180/90 deg flips, FoundationPose)
 2. static-phase detection from the SAM2 masks; one shared pose per static phase (joint fit over its frames,
    candidates ranked by silhouette + photometric evidence)
 3. Viterbi decoding over hypotheses (silhouette + photometric data cost, rotation/translation transition cost)
 4. propagation: re-fit every frame from its decoded neighbours, re-decode
 5. temporal smoothing + re-fit with a prior to the smoothed trajectory
 6. depth re-anchoring to the baseline (median camera z per phase), map back to world.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2  # noqa: E402
import h5py  # noqa: E402
import torch  # noqa: E402

from common import load_camera, load_mesh, rot_angle, rotvec_to_R, R_to_rotvec, Renderer, read_video  # noqa
from fit import CENTER, FrameData, fit_poses, sample_points  # noqa: E402
from decode import viterbi, centres  # noqa: E402
from photo import photo_scores  # noqa: E402

SEED = 0
CHUNK = 12          # frames per GPU batch
N_PTS = 1200        # surface samples for the silhouette chamfer
N_PIX = 800         # mask pixels per frame
W_SIL, W_PHOTO = 1 / 0.25, 1 / 0.02  # data-cost weights in the decoding (loss units -> cost)
SIG_ROT, SIG_TR = 6.0, 0.04          # transition scales (deg, m per frame)


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def flipM(axis, ang=np.pi):
    v = np.zeros(3)
    v[axis] = ang
    R = rotvec_to_R(v)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = CENTER - R @ CENTER
    return M


HYPS = [np.eye(4), flipM(0), flipM(1), flipM(2), flipM(2, np.pi / 2), flipM(2, -np.pi / 2)]


class Masks:
    def __init__(self, path):
        self.f = h5py.File(path, 'r')
        self.F = self.f['object'].shape[0]
        self.cache = {}

    def get(self, i):
        return self.f['object'][i], self.f['person'][i]

    def data(self, frames):
        frames = list(frames)
        lo, hi = min(frames), max(frames) + 1
        ob = self.f['object'][lo:hi]
        pe = self.f['person'][lo:hi]
        return {i: FrameData(ob[i - lo], pe[i - lo], n_pix=N_PIX, seed=SEED + i) for i in frames}


# ------------------------------------------------------------------ static phases
def detect_static(masks, min_len=30):
    """Static phases from the SAM2 object masks: maximal runs of frames whose mask lies inside the (dilated)
    mask of a reference frame, while the visible mask is not missing un-occluded parts of the reference."""
    F = masks.F
    o4 = masks.f['object'][:, ::4, ::4]
    st = np.zeros(F, bool)
    for i in range(F):
        j = max(0, i - 3)
        R = cv2.dilate(o4[j].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        st[i] = (o4[i] & ~R).sum() / max(1, o4[i].sum()) < 0.02 and o4[i].sum() > 0
    runs, i = [], 0
    while i < F:
        if st[i]:
            j = i
            while j + 1 < F and st[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    phases = []
    for a, b in runs:
        ref = (a + b) // 2
        ob_r, pe_r = masks.get(ref)
        Rd = cv2.dilate(ob_r.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0

        def ok(k):
            ob, pe = masks.get(k)
            if ob.sum() == 0:
                return True  # fully occluded: cannot contradict
            out = (ob & ~Rd).sum() / ob.sum()
            missing = (ob_r & ~ob & ~pe).sum() / max(1, ob_r.sum())
            return out < 0.02 and missing < 0.12
        s = ref
        while s - 1 >= 0 and ok(s - 1):
            s -= 1
        e = ref
        while e + 1 < F and ok(e + 1):
            e += 1
        if phases and s <= phases[-1][1] + 1:
            # same object placement as the previous phase (e.g. person walking in front): merge
            phases[-1] = (phases[-1][0], max(e, phases[-1][1]))
        else:
            phases.append((s, e))
    # merge consecutive phases separated by occlusion if their reference masks coincide
    merged = []
    for s, e in phases:
        if merged:
            ps, pe_ = merged[-1]
            a, _ = masks.get((ps + pe_) // 2)
            b, _ = masks.get((s + e) // 2)
            iou = (a & b).sum() / max(1, (a | b).sum())
            gap_ok = all(masks.get(k)[0].sum() == 0 or
                         ((masks.get(k)[0] & ~(cv2.dilate(a.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)).sum()
                          / masks.get(k)[0].sum() < 0.01) for k in range(pe_ + 1, s))
            if iou > 0.8 and gap_ok:
                merged[-1] = (ps, e)
                continue
        merged.append((s, e))
    return merged


# ------------------------------------------------------------------ stages
def stage_hyps(K, Tb, Tfp, masks, pts):
    F = len(Tb)
    NH = len(HYPS) + 1
    out = np.zeros((F, NH, 4, 4))
    loss = np.zeros((F, NH))
    for s in range(0, F, CHUNK):
        fr = list(range(s, min(F, s + CHUNK)))
        data = masks.data(fr)
        T0, fid = [], []
        for i in fr:
            for h in HYPS:
                T0.append(Tb[i] @ h)
                fid.append(i)
            T0.append(Tfp[i])
            fid.append(i)
        T, L = fit_poses(K, np.array(T0), fid, data, pts, iters=120)
        out[fr] = T.reshape(len(fr), NH, 4, 4)
        loss[fr] = L['loss'].reshape(len(fr), NH)
    return out, loss


def stage_static(K, phases, cands, closs, masks, pts, R, video):
    """One pose per static phase: joint silhouette fit over 16 frames from distinct per-frame candidates,
    then rank by (silhouette + photometric) summed over the frames."""
    res = []
    for a, b in phases:
        fr = np.unique(np.linspace(a, b, 16).round().astype(int))
        fr = [i for i in fr if masks.get(i)[0].sum() > 200]
        data = masks.data(fr)
        U = []
        for i in np.unique(np.linspace(a, b, 12).round().astype(int)):
            for h in np.argsort(closs[i])[:3]:
                c = cands[i][h]
                if all(rot_angle(c[:3, :3], u[:3, :3]) > 4 for u in U):
                    U.append(c)
        Tg, sil = [], []
        for g0 in range(0, len(U), 3):  # 3 shared poses x len(fr) frames per GPU batch (memory)
            T0, fid, sh = [], [], []
            for g, u in enumerate(U[g0:g0 + 3]):
                for i in fr:
                    T0.append(u)
                    fid.append(i)
                    sh.append(g)
            T_, L = fit_poses(K, np.array(T0), fid, data, pts, iters=150, share=sh)
            Tg.append(T_)
            sil.append(L['loss'].reshape(len(T_), len(fr)).mean(1))
        Tg, sil = np.concatenate(Tg), np.concatenate(sil)
        ph = np.zeros(len(U))
        for i in fr:
            ob, pe = masks.get(i)
            ph += photo_scores(R, K, video[i], ob, pe, Tg) / len(fr)
        cost = sil * W_SIL + ph * W_PHOTO
        o = np.argsort(cost)
        log(f'  static phase {a}-{b}: {len(U)} candidates; best cost {cost[o[0]]:.2f} (sil {sil[o[0]]:.3f} '
            f'photo {ph[o[0]]:.3f}), runner-up {cost[o[1]]:.2f} at {rot_angle(Tg[o[1], :3, :3], Tg[o[0], :3, :3]):.0f} deg')
        res.append(dict(phase=(a, b), T=Tg[o[0]], cost=cost[o], margin=float(cost[o[1]] - cost[o[0]])))
    return res


def score_photo(K, R, video, masks, cands):
    F = len(cands)
    ph = []
    for i in range(F):
        ob, pe = masks.get(i)
        ph.append(photo_scores(R, K, video[i], ob, pe, cands[i]))
    return ph


def propagate(K, path_T, masks, pts, offsets=(-2, -1, 1, 2)):
    F = len(path_T)
    newT, newL = [[] for _ in range(F)], [[] for _ in range(F)]
    for s in range(0, F, CHUNK):
        fr = list(range(s, min(F, s + CHUNK)))
        data = masks.data(fr)
        T0, fid = [], []
        for i in fr:
            for o in offsets:
                j = min(F - 1, max(0, i + o))
                T0.append(path_T[j])
                fid.append(i)
        T, L = fit_poses(K, np.array(T0), fid, data, pts, iters=100)
        for k, i in enumerate(fid):
            newT[i].append(T[k])
            newL[i].append(L['loss'][k])
    return [np.array(t) for t in newT], [np.array(l) for l in newL]


# ------------------------------------------------------------------ smoothing on SE(3)
def smooth_traj(T, weights, sigma=2.0, static_mask=None, sigma_t=None, range_deg=None):
    """Weighted Gaussian smoothing of rotations (via local rotation vectors around each frame) and centres.
    range_deg: bilateral range kernel on rotations (neighbours rotated by >> range_deg get little weight, so
    genuine fast twists are not smeared). sigma_t: separate temporal sigma for the centres."""
    sigma_t = sigma if sigma_t is None else sigma_t
    F = len(T)
    rad = int(3 * max(sigma, sigma_t))
    c = centres(T)
    out = T.copy()
    for i in range(F):
        if static_mask is not None and static_mask[i]:
            continue
        js = np.arange(max(0, i - rad), min(F, i + rad + 1))
        if static_mask is not None:
            js = js[~static_mask[js]] if (~static_mask[js]).any() else js
        Ri = T[i, :3, :3]
        rv = np.array([R_to_rotvec(Ri.T @ T[j, :3, :3]) for j in js])
        w = np.exp(-0.5 * ((js - i) / sigma) ** 2) * weights[js]
        if range_deg is not None:
            w = w * np.exp(-0.5 * (np.degrees(np.linalg.norm(rv, axis=1)) / range_deg) ** 2)
        w = w / w.sum()
        wt = np.exp(-0.5 * ((js - i) / sigma_t) ** 2) * weights[js]
        wt = wt / wt.sum()
        Rn = Ri @ rotvec_to_R((w[:, None] * rv).sum(0))
        cn = (wt[:, None] * c[js]).sum(0)
        out[i, :3, :3] = Rn
        out[i, :3, 3] = cn - Rn @ CENTER
    return out


def refine(K, P, static, area, masks, pts):
    F = len(P)
    # confidence: visible mask area relative to the typical unoccluded area
    wconf = np.clip(area / np.percentile(area, 75), 0.02, 1.0)
    for rnd in range(2):
        S = smooth_traj(P, wconf, sigma=2.0, static_mask=static)
        mov = [i for i in range(F) if not static[i] and area[i] > 0]
        newP = S.copy()
        for s in range(0, len(mov), 4 * CHUNK):
            fr = mov[s:s + 4 * CHUNK]
            data = masks.data(fr)
            T, L = fit_poses(K, S[fr], fr, data, pts, iters=80, prior=S[fr], w_prior=(0.02, 0.02))
            newP[fr] = T
        P = newP
    return smooth_traj(P, wconf, sigma=1.0, static_mask=static)


def smooth_depth(P, area, static, sigma=3.0):
    from scipy.ndimage import gaussian_filter1d
    F = len(P)
    c = centres(P)
    w = np.clip(area / np.percentile(area, 75), 0.02, 1.0)
    z = c[:, 2]
    zs = gaussian_filter1d(z * w, sigma, mode='nearest') / gaussian_filter1d(w, sigma, mode='nearest')
    out = P.copy()
    for i in range(F):
        if static[i]:
            continue
        s = zs[i] / z[i]
        out[i, :3, 3] = s * c[i] - P[i, :3, :3] @ CENTER  # move the centre along its viewing ray
    return out


# feet of the quad base in mesh coordinates (lowest vertex of each foot; the feet plane is the mesh xy-plane)
FEET = np.array([[-0.0883, 0.0723, -0.3779], [0.0868, 0.0712, -0.3779],
                 [-0.0898, -0.0503, -0.3775], [0.0872, -0.0506, -0.3779]])
GRIP = np.array([0.026, -0.001, 0.509])  # mean of the handle vertices (mesh z > 0.49)


def scale_along_rays(P, idx, s):
    out = P.copy()
    out[idx, :3, 3] *= s  # scaling the whole camera-frame translation keeps the image position of the origin;
    # rotation unchanged -> the object is moved along the viewing ray of the mesh origin
    return out


def anchor_depth(P, Tb, phases, static, Twc, plane, rec, mesh):
    """Per phase: target median camera z. Default = baseline median z. Deviate if two independent cues
    agree within 2 cm and the correction exceeds their disagreement:
      static phase : (a) metric mesh size vs image (our joint silhouette fit) and (b) untouched base resting
                     on the released floor plane (ray scaling that puts the 4 feet on the plane);
      moving phase : (a) silhouette size (our median z) and (b) hand contact: the reconstructed hand
                     (mean of its 21 keypoints) holds the grip -> median depth offset hand - grip."""
    F = len(P)
    out = P.copy()
    cues = []
    Tcw = np.linalg.inv(Twc)
    # phase list: static phases + moving segments between them
    segs = []
    prev = 0
    for a, b in phases:
        if a > prev:
            segs.append((prev, a - 1, 'moving'))
        segs.append((a, b, 'static'))
        prev = b + 1
    if prev < F:
        segs.append((prev, F - 1, 'moving'))
    kp = rec['human_keypoints_world']
    kpc = kp @ Tcw[:3, :3].T + Tcw[:3, 3]
    hands = np.stack([kpc[:, 21:42].mean(1), kpc[:, 42:63].mean(1)], 1)  # (F,2,3) right, left
    for a, b, kind in segs:
        sl = slice(a, b + 1)
        z_base = float(np.median(Tb[sl, 2, 3]))
        z_sil = float(np.median(P[sl, 2, 3]))
        cue_a = z_sil - z_base
        if kind == 'static':
            T = P[a]
            fc = FEET @ T[:3, :3].T + T[:3, 3]
            s = -(plane[:3] @ Twc[:3, 3] + plane[3]) / ((Twc[:3, :3] @ fc.T).T @ plane[:3])
            cue_b = float(T[2, 3] * s.mean() - z_base)
            name_b = 'floor_contact'
        else:
            g = np.einsum('fij,j->fi', P[sl, :3, :3], GRIP) + P[sl, :3, 3]
            d = np.linalg.norm(hands[sl] - g[:, None], axis=-1)
            k = d.argmin(1)
            hz = hands[sl][np.arange(len(k)), k, 2]
            held = d.min(1) < 0.15
            if held.sum() < 20:
                cue_b = None
            else:
                resid = float(np.median(hz[held] - g[held, 2]))
                cue_b = cue_a + resid
            name_b = 'hand_contact'
        apply = cue_b is not None and abs(cue_a - cue_b) < 0.02 and abs(cue_a) > abs(cue_a - cue_b) \
            and np.sign(cue_a) == np.sign(cue_b)
        shift = 0.5 * (cue_a + cue_b) if apply else 0.0
        z_target = z_base + shift
        out = scale_along_rays(out, np.arange(a, b + 1), z_target / z_sil)
        cues.append(dict(phase=[int(a), int(b)], kind=kind, z_base_median=z_base, z_ours_median=z_sil,
                         cue_silhouette_shift_cm=100 * cue_a,
                         **{f'cue_{name_b}_shift_cm': None if cue_b is None else 100 * cue_b},
                         applied_shift_cm=100 * shift))
    return out, cues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='/work/inputs')
    ap.add_argument('--out', default='/work/outputs/object_poses_world.npz')
    ap.add_argument('--cache', default=None, help='(development only) directory to cache stage results')
    args = ap.parse_args()
    t_start = time.time()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.set_per_process_memory_fraction(min(1.0, 8.0 * 2**30 / torch.cuda.get_device_properties(0).total_memory))
    inp = args.inputs
    K, Twc, Kp = load_camera(inp)
    Tcw = np.linalg.inv(Twc)
    rec = np.load(f'{inp}/reconstruction.npz')
    Tb = Tcw @ rec['T_world_mesh']  # baseline, camera frame
    F = len(Tb)
    Tfp = rec['T_cam_mesh_foundationpose'].copy()
    Tfp[:, :3, 3] *= (Tb[:, 2, 3] / Tfp[:, 2, 3])[:, None]  # FP rotation, baseline depth
    mesh = load_mesh(inp)[4]
    pts = sample_points(mesh, N_PTS, seed=SEED)
    masks = Masks(f'{inp}/masks.h5')
    log('reading video')
    video = read_video(f'{inp}/video.mp4')
    R = Renderer(inp)

    def cached(name, fn):
        if args.cache:
            p = os.path.join(args.cache, name + '.npy')
            if os.path.exists(p):
                return np.load(p, allow_pickle=True).item()
            r = fn()
            np.save(p, r, allow_pickle=True)
            return r
        return fn()

    # 1. hypotheses
    log('stage 1: per-frame hypotheses')
    s1 = cached('s1', lambda: dict(zip(['T', 'L'], stage_hyps(K, Tb, Tfp, masks, pts))))
    cands = [s1['T'][i] for i in range(F)]
    closs = [s1['L'][i] for i in range(F)]

    # 2. static phases
    phases = detect_static(masks)
    log('static phases:', phases)
    static = np.zeros(F, bool)
    for a, b in phases:
        static[a:b + 1] = True
    st = cached('s2', lambda: {'res': stage_static(K, phases, cands, closs, masks, pts, R, video)})['res']

    # 3. decode
    log('stage 3: photometric scoring + decoding')
    ph = cached('s3', lambda: {'ph': score_photo(K, R, video, masks, cands)})['ph']

    def decode(cands, closs, ph):
        cs, dc = [], []
        for i in range(F):
            cs.append(cands[i])
            dc.append(closs[i] * W_SIL + ph[i] * W_PHOTO)
        for r in st:
            a, b = r['phase']
            for i in range(a, b + 1):
                cs[i] = r['T'][None]
                dc[i] = np.zeros(1)
        path = viterbi(cs, dc, sig_rot=SIG_ROT, sig_tr=SIG_TR)
        return np.array([cs[i][path[i]] for i in range(F)]), path

    P1, _ = decode(cands, closs, ph)

    # 4. propagation rounds
    for rnd in range(2):
        log(f'stage 4: propagation round {rnd}')
        def prop():
            nT, nL = propagate(K, P1, masks, pts)
            nph = score_photo(K, R, video, masks, nT)
            return {'T': nT, 'L': nL, 'ph': nph}
        pr = cached(f's4_{rnd}', prop)
        cands = [np.concatenate([cands[i], pr['T'][i]]) for i in range(F)]
        closs = [np.concatenate([closs[i], pr['L'][i]]) for i in range(F)]
        ph = [np.concatenate([ph[i], pr['ph'][i]]) for i in range(F)]
        P1, path = decode(cands, closs, ph)

    if args.cache:
        np.save(os.path.join(args.cache, 'P1.npy'), P1)

    # 5. smoothing + re-fit with a prior to the smoothed trajectory (moving frames only)
    log('stage 5: smoothing / refinement')
    area = np.array([masks.f['object'][i, ::4, ::4].sum() * 16 for i in range(F)], float)
    P2 = cached('s5', lambda: {'P': refine(K, P1, static, area, masks, pts)})['P']
    # 5b. depth smoothing along the viewing rays (silhouette depth is the noisiest coordinate; the image
    # position is kept)
    P3 = smooth_depth(P2, area, static, sigma=3.0)

    # 6. depth anchoring per phase (default: baseline median z; deviate only with two agreeing cues)
    log('stage 6: depth anchoring')
    plane = np.array(json.load(open(f'{inp}/ground_plane.json'))['plane'], float)
    out, cues = anchor_depth(P3, Tb, phases, static, Twc, plane, rec, mesh)
    for c in cues:
        log('  ', json.dumps(c))
    Tw = Twc @ out
    np.savez(args.out, T_world_mesh=Tw.astype(np.float64))
    json.dump({'static_phases': [list(map(int, p)) for p in phases], 'depth_cues': cues,
               'static_fit': [dict(phase=list(map(int, r['phase'])), margin=r['margin']) for r in st]},
              open(os.path.splitext(args.out)[0] + '_info.json', 'w'), indent=1)
    log(f'done in {time.time() - t_start:.0f}s')


if __name__ == '__main__':
    main()
