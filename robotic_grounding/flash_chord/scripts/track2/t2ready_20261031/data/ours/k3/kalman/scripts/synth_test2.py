"""SYNTHETIC self-test of the K2 rest-physics anchors (sign conventions, association) on a box resting on a floor.
No dataset reference data: a simulated box (trimesh primitive) rests on stable faces with random yaw, is flipped between
rests, and gets the d1 random-walk bias. Compares none / K1 / K2 errors vs the simulated truth."""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent))
from kfilter import Episode, min_rotvec  # noqa: E402
from kcommon import KOUT, noise_sigmas, static_mask  # noqa: E402
from d2_mesh_grasp_physics import stable_poses  # noqa: E402

PRM = json.loads((KOUT / "design/frozen_params.json").read_text())
UP = np.array([0.0, -1.0, 0.0]); E0 = 1.4                       # floor: up.x + e = 0  (y = 1.4 m, OpenCV-like y down)


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def main():
    box = trimesh.creation.box(extents=[0.40, 0.20, 0.10])
    sp = stable_poses(box)
    out = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        T = 900
        segs = [(0, 150, None), (150, 300, "m"), (300, 450, None), (450, 560, "m"), (560, 700, None), (700, 800, "m"), (800, T, None)]
        rots, p = [], np.zeros((T, 3))
        def rest_pose():
            k = rng.integers(len(sp["h_m"]))
            Rk = R.from_rotvec(min_rotvec(sp["n"][k], -UP))
            Ry = R.from_rotvec(UP * rng.uniform(0, 2 * np.pi))
            Rw = Ry * Rk
            xy = rng.normal(size=3) * 50; xy -= (xy @ UP) * UP
            pos = xy + UP * (100 * (sp["h_m"][k] - E0) + PRM["height_mu_cm"])     # up.p = 100(h - e) + mu
            return Rw, pos
        cur_r, cur_p = rest_pose()
        for a, b, kind in segs:
            if kind is None:
                for t in range(a, b):
                    rots.append(cur_r); p[t] = cur_p
            else:
                nr, npos = rest_pose()
                dr = (nr * cur_r.inv()).as_rotvec()
                for i, t in enumerate(range(a, b)):
                    s = smoothstep((i + 1) / (b - a))
                    lift = UP * 30 * np.sin(np.pi * s)
                    rots.append(R.from_rotvec(dr * s) * cur_r); p[t] = cur_p + (npos - cur_p) * s + lift
                cur_r, cur_p = nr, npos
        Rt = R.concatenate(rots)
        bth = np.cumsum(rng.normal(size=(T, 3)) * np.radians(PRM["q_br_deg"]), 0); bth[0] = 0
        bp = np.cumsum(rng.normal(size=(T, 3)) * PRM["q_bp_cm"], 0); bp[0] = 0
        Rm = R.from_rotvec(bth) * Rt; pm = p + bp
        spg, srg = noise_sigmas(pm, Rm)
        st = static_mask(pm, Rm, spg, np.radians(srg))
        E = object.__new__(Episode)
        E.prm = PRM; E.p, E.rot = pm, Rm; E.T = T; E.vis = np.ones(T, bool)
        inc = (Rm[1:] * Rm[:-1].inv()).as_rotvec(); E.phi = np.vstack([np.zeros(3), np.cumsum(inc, 0)])
        E.rest = st; E.rest_raw = st; E.up, E.e = UP, E0; E.floor_only = True
        E.dmin = np.full((2, T), 1e3); E.sp_n, E.sp_h, E.sp_prob = sp["n"], sp["h_m"], sp["prob"]
        E.qa, E.qal = np.median(list(PRM["q_a_cm2_s3"].values())), np.median(list(PRM["q_al_rad2_s3"].values()))
        meas1 = E.base_meas(zv=True)
        X1, P1, _, _ = E.run(meas1)
        anc, diag = E.anchors(X1, P1)
        meas2 = [list(m) for m in meas1]
        for t, mm in anc:
            meas2[t].append(mm)
        X2, _, _, _ = E.run(meas2)
        def err(X):
            ra = R.from_rotvec(-X[:, 15:18]) * Rm
            return float(np.degrees((ra.inv() * Rt).magnitude()).mean()), float(np.linalg.norm(X[:, 0:3] - p, axis=1).mean())
        r = {"seed": seed, "none": (float(np.degrees((Rm.inv() * Rt).magnitude()).mean()), float(np.linalg.norm(pm - p, axis=1).mean())),
             "K1": err(X1), "K2": err(X2), **diag}
        out.append(r)
        print(r, flush=True)
    for k in ("none", "K1", "K2"):
        print(k, "mean rot err %.2f deg, pos err %.2f cm" % (np.mean([o[k][0] for o in out]), np.mean([o[k][1] for o in out])))
    (KOUT / "design/synth_test2.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
