#!/usr/bin/env python
"""Trajectory-level 6-DoF pose recovery of the cyan bottle (episode 25).

Pipeline (all estimates recomputed from the inputs; deterministic):
  0. per-frame crops of image / SAM2 object mask / person mask
  1. phase segmentation (static-at-start, carried, static-at-end) from the object-mask centroid track
  2. per-frame hypotheses: baseline and FoundationPose rotations and their end-over-end flips, refined by
     render-and-compare (nvdiffrast, silhouette + appearance features)
  3. hypothesis expansion: distinct axis directions, their toward/away tilt mirrors, x 24 spins about the
     bottle's long axis (the spin is only observable through the blue strap near the cap)
  4. global decoding over time with Viterbi (rotation/translation change penalties, much stronger in static phases)
  5. joint continuous refinement of all poses: image terms + second-order temporal smoothness on SE(3);
     one shared pose per static phase
  6. depth re-anchoring to the baseline: camera-frame positions scaled along the viewing rays so that the
     median depth matches the baseline (per static phase, and in a sliding window for the carried phase)
"""
import argparse, json, os, sys, time, pickle
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoi_lib import Renderer, Refiner, FrameData, Rz, rot_angle, so3_exp, dev

# ----------------------------------------------------------------------------- constants
SEED = 0
NSPIN = 24                  # spin hypotheses about the long (mesh z) axis, 15 deg apart
AXIS_DEDUP_DEG = 12.0       # axis hypotheses closer than this are merged
LAM_R = 0.004               # Viterbi: cost per degree of rotation change between consecutive frames
LAM_T = 1.0                 # Viterbi: cost per metre of translation change
STATIC_MULT = 20.0          # Viterbi: transition penalties multiplied inside static phases
STATIC_TOL_PX = 2.5         # static phase: (median-filtered) mask centroid within this many px of the rest position
DEPTH_WIN = 61              # sliding window (frames) for depth re-anchoring during the carried phase
FX = np.diag([1., -1., -1.])  # 180 deg about mesh x: end-over-end flip of the bottle


def log(*a):
    print('[%7.1fs]' % (time.time() - T0), *a, flush=True)


def tt(a):
    return torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)


def segment_phases(D):
    """static phase at the start and the end of the episode from the object-mask centroid track"""
    from scipy.ndimage import median_filter
    c = np.stack([median_filter(D.cent[:, 0], 5, mode='nearest'), median_filter(D.cent[:, 1], 5, mode='nearest')], 1)
    F = len(c)
    rest0 = np.median(c[:15], 0); rest1 = np.median(c[-15:], 0)
    d0 = np.linalg.norm(c - rest0, axis=1); d1 = np.linalg.norm(c - rest1, axis=1)
    s0 = int(np.argmax(d0 > STATIC_TOL_PX))           # first frame that left the start rest position
    s1 = F - int(np.argmax(d1[::-1] > STATIC_TOL_PX))  # first frame of the final rest
    # 2-frame safety margins: lift-off / set-down frames are treated as moving
    return (0, max(s0 - 2, 1)), (min(s1 + 2, F - 1), F)


def viterbi(Rs, ts, un, static):
    F = len(Rs); back = []; acc = un[0].copy()
    for i in range(1, F):
        A = rot_angle(Rs[i - 1], Rs[i])
        Td = np.linalg.norm(ts[i - 1][:, None] - ts[i][None], axis=-1)
        m = STATIC_MULT if (static[i] and static[i - 1]) else 1.0
        tot = acc[:, None] + m * (LAM_R * A + LAM_T * Td)
        b = np.argmin(tot, 0); back.append(b)
        acc = tot[b, np.arange(len(b))] + un[i]
    path = [int(np.argmin(acc))]
    for b in back[::-1]:
        path.append(int(b[path[-1]]))
    return path[::-1]


def tilt_mirror(R, t):
    """rotate the pose about the object centre so that the camera-z component of the bottle axis changes sign
    (the toward/away ambiguity of an elongated object's silhouette)"""
    u = R[:, 2]; v = u * np.array([1, 1, -1.])
    ax = np.cross(u, v); s = np.linalg.norm(ax); c = u @ v
    if s < 1e-6:
        return None
    ax = ax / s; ang = np.arctan2(s, c)
    Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    Q = np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * Kx @ Kx
    return Q @ R


def main():
    global T0
    T0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='/work/inputs')
    ap.add_argument('--out', default='/work/outputs/object_poses_world.npz')
    ap.add_argument('--cache', default=None, help='(development only) directory to cache stages')
    args = ap.parse_args()
    np.random.seed(SEED); torch.manual_seed(SEED)
    inp = args.inputs
    cam = json.load(open(f'{inp}/camera.json'))
    K = np.array(cam['K']); Twc = np.array(cam['T_world_cam']); Tcw = np.linalg.inv(Twc)
    rec = np.load(f'{inp}/reconstruction.npz')
    Tb = Tcw[None] @ rec['T_world_mesh'].astype(np.float64)       # baseline, camera frame
    Tf = rec['T_cam_mesh_foundationpose'].astype(np.float64)
    F = Tb.shape[0]

    def cached(name, fn):
        if args.cache:
            p = os.path.join(args.cache, name + '.pkl')
            if os.path.exists(p):
                return pickle.load(open(p, 'rb'))
            r = fn(); pickle.dump(r, open(p, 'wb'), protocol=4); return r
        return fn()

    # ------------------------------------------------------------------ 0. data
    D = cached('data', lambda: FrameData(f'{inp}/video.mp4', f'{inp}/masks.h5'))
    assert D.F == F
    log('data loaded', F, 'frames')
    R = Renderer(f'{inp}/object_mesh.glb')
    RF = Refiner(R, K, D.S)

    # ------------------------------------------------------------------ 1. phases
    ph0, ph1 = segment_phases(D)
    static = np.zeros(F, bool); static[ph0[0]:ph0[1]] = True; static[ph1[0]:ph1[1]] = True
    phases = [('static_start', ph0[0], ph0[1]), ('carried', ph0[1], ph1[0]), ('static_end', ph1[0], ph1[1])]
    log('phases', phases)

    # ------------------------------------------------------------------ 2. refined hypotheses
    def stage1():
        jobs = []
        for i in range(F):
            for Rs in [Tb[i, :3, :3], Tb[i, :3, :3] @ FX, Tf[i, :3, :3], Tf[i, :3, :3] @ FX]:
                jobs.append((i, Rs, Tb[i, :3, 3]))
        out = []
        BS = 192
        for s in range(0, len(jobs), BS):
            J = jobs[s:s + BS]
            Rm, t, L = RF.optimize(tt([j[1] for j in J]), tt([j[2] for j in J]), *D.batch([j[0] for j in J]), iters=40)
            u = RF.unary(*L.T).cpu().numpy()
            for k, j in enumerate(J):
                out.append((j[0], Rm[k].cpu().numpy().astype(np.float64), t[k].cpu().numpy().astype(np.float64), float(u[k])))
        return out
    s1 = cached('s1', stage1)
    log('stage 1 (refined hypotheses) done', len(s1))

    # ------------------------------------------------------------------ 3. axis hypotheses x tilt mirrors x spins
    def stage2():
        by = [[] for _ in range(F)]
        for x in s1:
            by[x[0]].append(x)
        hyps = []
        for i in range(F):
            cands = []
            for c in sorted(by[i], key=lambda x: x[3]):
                cands.append((c[1], c[2]))
                Rm = tilt_mirror(c[1], c[2])
                if Rm is not None:
                    cands.append((Rm, c[2]))
            keep = []
            for c in cands:
                if all(np.degrees(np.arccos(np.clip(c[0][:, 2] @ k[0][:, 2], -1, 1))) > AXIS_DEDUP_DEG for k in keep):
                    keep.append(c)
            hyps.append(keep[:6])
        SP = np.array([Rz(a) for a in np.arange(NSPIN) * 360. / NSPIN])
        jobs = [(i, h[0] @ SP[a], h[1]) for i in range(F) for h in hyps[i] for a in range(NSPIN)]
        U = []
        BS = 384
        for s in range(0, len(jobs), BS):
            J = jobs[s:s + BS]
            L = RF.evaluate(tt([j[1] for j in J]), tt([j[2] for j in J]), *D.batch([j[0] for j in J]))
            U.append(RF.unary(*L.T).cpu().numpy())
        U = np.concatenate(U)
        Rs = [[] for _ in range(F)]; ts = [[] for _ in range(F)]; un = [[] for _ in range(F)]
        for j, u in zip(jobs, U):
            Rs[j[0]].append(j[1]); ts[j[0]].append(j[2]); un[j[0]].append(u)
        return [np.array(x) for x in Rs], [np.array(x) for x in ts], [np.array(x) for x in un]
    Rs, ts, un = cached('s2', stage2)
    log('stage 2 (spin sweep) done', sum(len(x) for x in un), 'states')

    # ------------------------------------------------------------------ 4. Viterbi
    path = viterbi(Rs, ts, un, static)
    R0 = np.array([Rs[i][k] for i, k in enumerate(path)]); t0 = np.array([ts[i][k] for i, k in enumerate(path)])
    log('viterbi done')

    # ------------------------------------------------------------------ 5. joint refinement
    Rfin, tfin, diag = joint_refine(RF, D, R0, t0, phases)
    log('joint refinement done')

    # ------------------------------------------------------------------ 6. depth re-anchoring to the baseline
    zb = Tb[:, 2, 3]
    scale = np.ones(F)
    for name, a, b in phases:
        if name.startswith('static'):
            scale[a:b] = np.median(zb[a:b]) / np.median(tfin[a:b, 2])
    a, b = phases[1][1], phases[1][2]
    h = DEPTH_WIN // 2
    for i in range(a, b):
        lo, hi = max(a, i - h), min(b, i + h + 1)
        scale[i] = np.median(zb[lo:hi]) / np.median(tfin[lo:hi, 2])
    tanch = tfin * scale[:, None]      # move along the viewing ray: image position unchanged
    Tc = np.tile(np.eye(4), (F, 1, 1)); Tc[:, :3, :3] = Rfin; Tc[:, :3, 3] = tanch
    Tw = Twc[None] @ Tc
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, T_world_mesh=Tw.astype(np.float64))
    # diagnostics for the report
    dz = (tanch[:, 2] - zb) * 100
    diag.update(phases=phases, depth_scale=scale.tolist(),
                dz_cm={n: (float(dz[a:b].mean()), float(np.median(dz[a:b]))) for n, a, b in phases},
                path_unary=[float(un[i][k]) for i, k in enumerate(path)])
    np.savez(os.path.splitext(args.out)[0] + '_diag.npz', R_viterbi=R0, t_viterbi=t0, R_refined=Rfin,
             t_refined=tfin, t_anchored=tanch, static=static)
    json.dump(diag, open(os.path.splitext(args.out)[0] + '_diag.json', 'w'))
    log('saved', args.out, '| peak GPU memory %.2f GB' % (torch.cuda.max_memory_allocated() / 1e9))


def joint_refine(RF, D, R0, t0, phases, iters=60, lam_r=1.0, lam_t=500.0, chunk=128):
    """all frames jointly: sum of per-frame image unaries + second-order smoothness; static phases share one pose"""
    F = len(R0)
    # parameterisation: per frame a local rotation increment and translation increment; static phases map all
    # their frames to a single parameter row (shared pose, initialised from the phase's median frame)
    pid = np.arange(F)
    Rinit = R0.copy(); tinit = t0.copy()
    for name, a, b in phases:
        if name.startswith('static'):
            m = (a + b) // 2
            pid[a:b] = m
    Rinit_t = tt(Rinit); tinit_t = tt(tinit); pid_t = torch.tensor(pid, device=dev)
    w = torch.zeros(F, 3, device=dev, requires_grad=True)
    dt = torch.zeros(F, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{'params': [w], 'lr': 0.01}, {'params': [dt], 'lr': 0.003}])
    moving = torch.ones(F, device=dev)
    for name, a, b in phases:
        if name.startswith('static'):
            moving[a:b] = 0

    def poses(idx):
        p = pid_t[idx]
        return Rinit_t[p] @ so3_exp(w[p]), tinit_t[p] + dt[p]
    allidx = torch.arange(F, device=dev)
    for it in range(iters):
        opt.zero_grad()
        tot = 0.
        for s in range(0, F, chunk):
            idx = allidx[s:s + chunk]
            Rm, t = poses(idx)
            L = RF.losses(Rm, t, *D.batch(idx.cpu().numpy()))
            loss = RF.unary(*L).sum()
            loss.backward(); tot += loss.item()
        Rm, t = poses(allidx)
        acc_r = ((Rm[:-2] - 2 * Rm[1:-1] + Rm[2:]) ** 2).sum((1, 2))
        acc_t = ((t[:-2] - 2 * t[1:-1] + t[2:]) ** 2).sum(1)
        # depth (z) is weakly observed: smooth it more
        acc_z = (t[:-2, 2] - 2 * t[1:-1, 2] + t[2:, 2]) ** 2
        sm = (lam_r * acc_r + lam_t * acc_t + 4 * lam_t * acc_z).sum()
        sm.backward()
        opt.step()
    with torch.no_grad():
        Rm, t = poses(allidx)
        U = []
        for s in range(0, F, chunk):
            idx = allidx[s:s + chunk]
            U.append(torch.stack(RF.losses(Rm[s:s + chunk], t[s:s + chunk], *D.batch(idx.cpu().numpy())), -1))
        U = torch.cat(U).cpu().numpy()
    return Rm.cpu().numpy().astype(np.float64), t.cpu().numpy().astype(np.float64), {'final_losses': U.tolist()}


if __name__ == '__main__':
    main()
