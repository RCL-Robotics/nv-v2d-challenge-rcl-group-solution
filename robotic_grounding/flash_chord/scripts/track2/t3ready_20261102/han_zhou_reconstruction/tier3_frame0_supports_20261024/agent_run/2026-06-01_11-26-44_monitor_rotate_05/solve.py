#!/usr/bin/env python
"""Episode 6 (monitor rotated on a round table): full 6-DoF object trajectory.

Method (see REPORT.md):
  1. Physical model: the monitor stands on the table the whole time (verified in the video: the person only pushes
     and turns the base), so its pose is upright on the floor-parallel table plane: yaw about the floor normal plus a
     2-D position on the plane. The table height is the baseline's median base-contact height.
  2. Per-frame evidence on a 3-deg yaw grid: soft silhouette agreement with the SAM2 mask (two-sided, best of
     'mask = whole monitor' / 'mask = screen panel only', person pixels in front of the object ignored using the
     reconstructed human depth) + a front/back term (the real screen is covered by white paper -> bright, the back
     is dark). For every yaw the table-plane position is optimised.
  3. Phases from an image-motion score: static 1 (until the monitor first moves), moving, static 2 (after the last
     hand contact once the image is still). Static phases pool the evidence of all their frames into one pose.
  4. Viterbi over yaw (smooth rotation while moving, constant when static) with a weak, capped prior towards the
     baseline yaw (only decisive where the image cost is flat, i.e. near-frontal views), sub-grid refinement,
     quadratic smoothing; translations smoothed the same way.
  5. Depth re-anchoring: every pose is moved along its viewing ray so that the camera-frame depth follows the
     baseline (median per static phase, heavily smoothed ratio while moving). Rotations and image positions are ours.
Deterministic: fixed seeds, no randomness in the optimisation (Adam from fixed inits).
"""
import argparse, os, sys, time, json
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoi_fit import (setup, Evidence, SilRenderer, decimate, grid_costs, motion_score, detect_phases,
                     decode_trajectory, reanchor_depth)

PRIOR_W = 0.3  # weight of the baseline-yaw prior; chosen with the held-out optical-flow check (REPORT.md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--out", default="/work/outputs/object_poses_world.npz")
    args = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(0); np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    log = lambda s: print(f"[{time.time() - t0:7.1f}s] {s}", flush=True)

    S = setup(args.inputs); pm = S["pm"]; F_ = len(S["Tb"])
    log(f"frames {F_}; table plane d={pm.d_t:.3f}; baseline implausible (tilted) frames: {int((~S['good']).sum())}")
    E = Evidence(S, scale=0.5, inputs=args.inputs)
    sr = SilRenderer(decimate(S["mesh"]), S["K"], S["Twc"], scale=0.5)
    log("evidence ready")

    G = grid_costs(sr, E, pm, S["par_base"], NG=120, iters=30, log=log)
    log("yaw-grid costs done")

    m = motion_score(E.G, E.MO, E.MP)
    seg = detect_phases(m, S["rec"]["contact_prob_right_left"].max(1))
    for s in (0, 1):
        idx = np.nonzero(seg == s)[0]
        if len(idx): log(f"static phase {s}: frames {idx[0]}-{idx[-1]}")

    th, a, b, path, th_raw, ab_raw = decode_trajectory(
        G, seg, prior_th=S["par_base_raw"][:, 0], prior_valid=S["good"].astype(float), prior_w=PRIOR_W)
    T = np.stack([pm.pose(th[i], a[i], b[i]) for i in range(F_)])
    T_final, ratio = reanchor_depth(T, S["Tb"], S["good"], seg, S["Twc"])
    log(f"depth re-anchoring ratios: min {ratio.min():.3f} max {ratio.max():.3f}")

    assert T_final.shape == (F_, 4, 4) and np.isfinite(T_final).all()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, T_world_mesh=T_final.astype(np.float64))
    diag = os.path.join(os.path.dirname(os.path.abspath(args.out)), "solve_diagnostics.npz")
    np.savez(diag, grid=G, seg=seg, motion=m, yaw=th, yaw_raw=th_raw, a=a, b=b, ratio=ratio,
             T_before_reanchor=T)
    log(f"wrote {args.out} (and {diag})")


if __name__ == "__main__":
    main()
