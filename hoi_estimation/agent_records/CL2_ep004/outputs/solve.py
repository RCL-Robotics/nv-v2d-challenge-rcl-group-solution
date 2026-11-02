"""Recover the 6-DoF trajectory of the bat in episode video.mp4.

python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz

Method (see REPORT.md):
 1. per frame: 12 hypotheses (baseline / FoundationPose / CoCoNet x {as is, end-to-end flip, out-of-plane mirror,
    both}) refined by render-and-compare against the object silhouette (SAM2 mask extended with handle-coloured
    pixels; person pixels = don't care);
 2. per-hypothesis evidence: silhouette IoU + distance-to-mask, colour agreement (light barrel / dark handle), MoGe-2
    relative depth profile along the bat, floor penetration;
 3. static phases (mask unchanged w.r.t. first/last frame) get one pooled pose; Viterbi decoding of the moving phase
    with rotation/translation transition costs; a propagation pass adds neighbour poses as hypotheses;
 4. joint refinement of all poses with image terms + SE(3) smoothness (static poses fixed);
 5. depth re-anchoring: the bat centre is moved along its viewing ray to the (smoothed) baseline depth.
"""
import argparse
import pickle
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stages import (Episode, fit_frames, score_frames, depth_scores, floor_heights, full_unary, pair_cost, viterbi,
                    detect_static_phases, smooth_baseline_z, joint_refine, depth_targets, anchor_depth, mean_pose)
from stages import transported_frames, roll_scores, decode_roll, apply_roll, BAT_CENTER
from bat_lib import simplified_mesh, FastRenderer, Renderer


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def unaries(ep, FR, verts, ids, Ts, scs, zb):
    dr = depth_scores(ep, FR, ids, Ts)
    return [full_unary(ep, sc, d, floor_heights(ep, T, verts, zb[i])) for i, T, sc, d in zip(ids, Ts, scs, dr)]


def static_pose(ep, FR, verts, frames, cands, zb, n_samp=12):
    """One pose for a static phase: every candidate of the sampled frames is scored on all sampled frames
    (pooled evidence), the best is refined on each sampled frame and the refined poses are averaged."""
    samp = frames[::max(1, len(frames) // n_samp)]
    pool = np.concatenate([cands[i] for i in samp])
    tot = np.zeros(len(pool))
    for i in samp:
        sc = score_frames(ep, FR, [i], [pool])[0]
        tot += unaries(ep, FR, verts, [i], [pool], [sc], zb)[0]
    best = pool[np.argmin(tot)]
    ref = fit_frames(ep, FR, samp, [best[None]] * len(samp))
    return mean_pose(np.stack([r["T"][0] for r in ref])), tot.min() / len(samp)


def static_fallback(ep, FR, T, a, b, zt):
    """Static phase [a,b]: compare our pooled pose with the baseline's phase pose (chordal mean, axis/position)
    carrying OUR decoded roll, both at the anchored depth. The baseline alternative is adopted only if it is better
    on both pooled image checks (silhouette IoU loss and MoGe depth-profile residual)."""
    from fitter import rot_a_to_b
    fr = list(range(a, b + 1))
    Pb = mean_pose(ep.Tc_base[a:b + 1])
    Rb = rot_a_to_b(T[a, :3, 2], Pb[:3, 2]) @ T[a, :3, :3]  # our roll, baseline axis
    Pb[:3, :3] = Rb
    c_b = mean_pose(ep.Tc_base[a:b + 1])
    Pb[:3, 3] = c_b[:3, :3] @ BAT_CENTER + c_b[:3, 3] - Rb @ BAT_CENTER  # same bat centre as the baseline
    cand = anchor_depth(np.stack([T[a], Pb]), np.full(2, zt[a]))
    samp = fr[::max(1, len(fr) // 12)]
    sil = np.zeros(2); dp = np.zeros(2)
    for i in samp:
        sil += score_frames(ep, FR, [i], [cand])[0][:, 1]
        dp += np.nan_to_num(depth_scores(ep, FR, [i], [cand])[0])
    use_base = bool(sil[1] < sil[0] and dp[1] < dp[0])
    log(f"static {a}-{b}: ours sil {sil[0] / len(samp):.3f} depth {dp[0] / len(samp):.4f} | baseline-with-our-roll "
        f"sil {sil[1] / len(samp):.3f} depth {dp[1] / len(samp):.4f} -> {'baseline' if use_base else 'ours'}")
    if use_base:
        # keep the solver's (silhouette) depth convention for the following steps; re-anchored at the end anyway
        Pb = anchor_depth(Pb[None], np.array([(T[a, :3, :3] @ BAT_CENTER + T[a, :3, 3])[2]]))[0]
        T = T.copy(); T[a:b + 1] = Pb
    return T


def decode(ep, FR, verts, cands, un, s1, s2, P1, P2):
    mid = list(range(s1 + 1, s2))
    cm = [cands[i] for i in mid]; um = [un[i].copy() for i in mid]
    # boundary conditions: continuity with the static poses
    um[0] = um[0] + pair_cost(P1[None], cm[0])[0] / 10.0
    um[-1] = um[-1] + pair_cost(cm[-1], P2[None])[:, 0] / 10.0
    k = viterbi(cm, um)
    T = np.zeros((ep.F, 4, 4)); T[:s1 + 1] = P1; T[s2:] = P2
    for j, i in enumerate(mid):
        T[i] = cm[j][k[j]]
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--out", default="/work/outputs/object_poses_world.npz")
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--cache-dir", default=None, help="experiments only: cache/reuse frame data and stage A")
    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    t_start = time.time()

    ep = Episode(args.inputs)
    vs, fs, cs = simplified_mesh(ep.mesh)
    FR = FastRenderer(vs, fs, cs)
    cache = None if args.cache_dir is None else os.path.join(args.cache_dir, "stageA.pkl")
    if cache is not None and os.path.exists(cache):
        import pickle
        ep.__dict__.update(pickle.load(open(cache, "rb"))["ep"])
        log("loaded cached frame data")
    else:
        log("decoding video, building crops / extended masks / MoGe crops")
        ep.load_frames()
    F = ep.F
    s1, s2, _, _ = detect_static_phases(ep.mask_small_o, ep.mask_small_p)
    log(f"static phases: 0-{s1}, {s2}-{F - 1}")
    zb = smooth_baseline_z(ep)

    # ---- stage A: multi-hypothesis per-frame fitting
    t0 = time.time()
    ids = list(range(F))
    if cache is not None and os.path.exists(cache):
        res = pickle.load(open(cache, "rb"))["res"]
    else:
        res = fit_frames(ep, FR, ids, [ep.init_hypotheses(i) for i in ids])
        if cache is not None:
            import pickle
            keys = ["wins", "datas", "mask_area", "mask_cent", "ext_area", "mask_small_o", "mask_small_p"]
            pickle.dump(dict(ep={k: ep.__dict__[k] for k in keys}, res=res), open(cache, "wb"), protocol=4)
    cands = [r["T"] for r in res]
    un = unaries(ep, FR, vs, ids, cands, [r["sc"] for r in res], zb)
    log(f"stage A done ({time.time() - t0:.0f}s)")

    # ---- static phases: pooled poses
    P1, c1 = static_pose(ep, FR, vs, list(range(0, s1 + 1)), cands, zb)
    P2, c2 = static_pose(ep, FR, vs, list(range(s2, F)), cands, zb)
    log(f"static poses: pooled cost {c1:.3f} / {c2:.3f}")

    # ---- Viterbi decoding, then a propagation pass (decoded neighbours as new hypotheses), decode again
    T = decode(ep, FR, vs, cands, un, s1, s2, P1, P2)
    mid = list(range(s1 + 1, s2))
    pr = fit_frames(ep, FR, mid, [np.stack([T[i - 1], T[i + 1]]) for i in mid])
    newc = [r["T"] for r in pr]
    nu = unaries(ep, FR, vs, mid, newc, [r["sc"] for r in pr], zb)
    for j, i in enumerate(mid):
        cands[i] = np.concatenate([cands[i], newc[j]]); un[i] = np.concatenate([un[i], nu[j]])
    T = decode(ep, FR, vs, cands, un, s1, s2, P1, P2)
    log("decoding done")

    # ---- roll about the bat axis (silhouette-invariant): texture evidence for 36 roll angles per frame relative to
    # a minimal-twist reference frame, decoded with DP (constant in static phases, smooth otherwise)
    Rfull = Renderer(ep.mesh)
    Q = transported_frames(T)
    angs, S, valid, sil = roll_scores(ep, Rfull, T, Q, ids, n_ang=36)
    kr = decode_roll(S, valid, [(0, s1), (s2, F - 1)], sil=sil)
    T = apply_roll(T, Q, angs[kr])
    del Rfull; torch.cuda.empty_cache()
    log("roll decoding done")
    if args.cache_dir is not None:
        np.savez(os.path.join(args.cache_dir, "decoded.npz"), T=T, S=S, valid=valid, kr=kr, sil=sil)

    # ---- static phases: fall back to the baseline's pose (with our roll) where it explains the images better
    zt = depth_targets(ep, s1, s2)
    T = static_fallback(ep, FR, T, 0, s1, zt)
    T = static_fallback(ep, FR, T, s2, F - 1, zt)

    # ---- joint refinement (moving phase; static poses fixed). Frames with tiny visible masks rely more on
    # smoothness.
    w_data = np.clip(ep.ext_area / 400.0, 0.2, 1.0)
    J = joint_refine(ep, FR, T, [(0, s1), (s2, F - 1)], w_data=w_data)
    log("joint refinement done")

    # ---- depth re-anchoring to the baseline (bat centre moved along its viewing ray)
    Tc = anchor_depth(J, zt)
    Tw = ep.Twc[None] @ Tc
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, T_world_mesh=Tw.astype(np.float64))
    log(f"saved {args.out}  total {time.time() - t_start:.0f}s")

    if not args.no_validate:
        from validate import evaluate, format_table
        phases = {"static-1": (0, s1), "moving": (s1 + 1, s2 - 1), "static-2": (s2, F - 1), "all": (0, F - 1)}
        rows = {}
        rows["baseline"], _ = evaluate(ep, FR, ep.Tc_base, phases, vs)
        rows["decoded (before joint)"], _ = evaluate(ep, FR, anchor_depth(T, zt), phases, vs)
        rows["final"], det = evaluate(ep, FR, Tc, phases, vs)
        tab = format_table(rows)
        print(tab)
        out_dir = os.path.dirname(os.path.abspath(args.out))
        with open(os.path.join(out_dir, "validation_table.md"), "w") as f:
            f.write(tab + "\n")
        np.savez(os.path.join(out_dir, "validation_details.npz"), **det, static=np.array([s1, s2]))
        log(f"validation done, total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
