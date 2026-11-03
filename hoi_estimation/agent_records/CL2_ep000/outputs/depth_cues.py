#!/usr/bin/env python
"""Independent physical cues for a depth (along-the-viewing-ray) correction of a trajectory. Reporting only;
solve.py applies none of them (shift 0) unless two independent cues agree (see REPORT.md).

For each phase and each candidate shift s (cm, + = farther from the camera; every pose is moved along its
own viewing ray by s) we evaluate:
  floor : static phases only - the lowest mesh point should lie at the floor. The released plane is a fit;
          the TRUE frame-0 human's lowest vertex lies 2.9 cm below it, so the floor level is taken as -2.9 cm.
          Reported: the shift that puts the median lowest point at -2.9 cm (and at 0 cm, plane taken literally).
  hand  : contact frames (contact prob > 0.5) - median distance from the closest hand keypoint of the
          reconstructed human to the mesh surface; reported: the shift minimising it.
  size  : the shift implied by metric mesh size vs image size (render-and-compare with free depth),
          i.e. median(z_fit) - median(z_result) per phase, from solve.py's cached per-frame fits.
usage: python depth_cues.py --result object_poses_world.npz --a_end A --b_start B [--fits debug.npz]
"""
import argparse
import os
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoilib import load_inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default="/work/inputs")
    ap.add_argument("--result", required=True)
    ap.add_argument("--a_end", type=int, required=True)
    ap.add_argument("--b_start", type=int, required=True)
    ap.add_argument("--fits", default=None)
    args = ap.parse_args()
    K, Twc, rec, mesh, plane = load_inputs(args.inputs)
    Tcw = np.linalg.inv(Twc)
    T = np.einsum("ij,fjk->fik", Tcw, np.load(args.result)["T_world_mesh"])
    F = len(T)
    n, d = plane[:3], plane[3]
    V = np.asarray(mesh.vertices)
    surf, _ = trimesh.sample.sample_surface_even(mesh, 20000, seed=0)
    tree = cKDTree(surf)
    kp = rec["human_keypoints_world"].astype(float)
    cp = rec["contact_prob_right_left"]
    shifts = np.arange(-15, 16, 1.0) / 100
    fits = np.load(args.fits) if args.fits else None
    phases = {"A static": np.arange(0, args.a_end + 1), "moving": np.arange(args.a_end + 1, args.b_start),
              "B static": np.arange(args.b_start, F)}

    def shifted(Tc, s):
        T2 = Tc.copy()
        T2[:3, 3] = Tc[:3, 3] * (1 + s / Tc[2, 3])
        return Twc @ T2

    for pn, ph in phases.items():
        out = [pn]
        if "static" in pn:
            low = []
            for s in shifts:
                low.append(np.median([((V @ shifted(T[i], s)[:3, :3].T + shifted(T[i], s)[:3, 3]) @ n + d).min()
                                      for i in ph[::10]]))
            low = np.array(low)
            s_true = shifts[np.argmin(np.abs(low + 0.029))]
            s_plane = shifts[np.argmin(np.abs(low))]
            out.append(f"floor: lowest pt {low[np.argmin(np.abs(shifts))] * 100:+.1f} cm at s=0; "
                       f"shift to floor(-2.9cm) {s_true * 100:+.1f} cm, to plane {s_plane * 100:+.1f} cm")
        hd = []
        frames = [i for i in ph[::3] if cp[i].max() > 0.5]
        if frames:
            for s in shifts:
                v = []
                for i in frames:
                    Ti = np.linalg.inv(shifted(T[i], s))
                    for h, (lo, hi) in enumerate([(21, 42), (42, 63)]):
                        if cp[i, h] > 0.5:
                            p = kp[i, lo:hi] @ Ti[:3, :3].T + Ti[:3, 3]
                            v.append(tree.query(p)[0].min())
                hd.append(np.median(v))
            hd = np.array(hd)
            out.append(f"hand: {len(frames)} frames, dist {hd[np.argmin(np.abs(shifts))] * 100:.2f} cm at s=0, "
                       f"best shift {shifts[np.argmin(hd)] * 100:+.1f} cm ({hd.min() * 100:.2f} cm)")
        if fits is not None:
            zf = fits["Tf"][ph, 2, 3]
            out.append(f"size: render-and-compare depth shift {(np.median(zf) - np.median(T[ph, 2, 3])) * 100:+.1f} cm")
        print(" | ".join(out))


if __name__ == "__main__":
    main()
