"""SYNTHETIC self-test of K3 hand coupling (on top of the K2 box world of synth_test2). No dataset reference data.
During every move the box is held rigidly by a simulated hand; the hand is observed through a simulated root bias
(rotation about a Hips pivot + translation, d1 noisy-only steps). Compares none / K1 / K2 / K3 vs the simulated truth."""
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
        moving = np.zeros(T, bool)
        for a, b, kind in segs:
            if kind is None:
                for t in range(a, b):
                    rots.append(cur_r); p[t] = cur_p
            else:
                moving[a:b] = True
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
        # simulated hand: rigid grasp during moves, observed through a root bias about a moving Hips pivot
        hips = p + np.array([40.0, -30.0, 20.0])
        R_oh = R.from_rotvec(rng.normal(size=3)); r_oh = np.array([0.0, 0.0, 15.0])
        Rh_true = Rt * R_oh; ph_true = p + Rt.apply(r_oh)
        dth = np.cumsum(rng.normal(size=(T, 3)) * np.radians(PRM["q_root_rot_deg"]), 0); dth[0] = 0
        dtr = np.cumsum(rng.normal(size=(T, 3)) * PRM["q_root_trans_cm"], 0); dtr[0] = 0
        Rd = R.from_rotvec(dth)
        hp_meas = hips + Rd.apply(ph_true - hips) + dtr
        hr_meas = Rd * Rh_true
        hands = ({0: hp_meas, 1: hp_meas}, {0: hr_meas, 1: hr_meas}, hips)
        E.side = np.where(moving, 0, -1)
        _, X1, _ = E.variant("K1")
        _, X2, diag = E.variant("K2")
        _, X3, d3 = E.variant("K3", hands=hands)
        diag.update(d3)
        def err(X):
            ra = R.from_rotvec(-X[:, 15:18]) * Rm
            return float(np.degrees((ra.inv() * Rt).magnitude()).mean()), float(np.linalg.norm(X[:, 0:3] - p, axis=1).mean())
        r = {"seed": seed, "none": (float(np.degrees((Rm.inv() * Rt).magnitude()).mean()), float(np.linalg.norm(pm - p, axis=1).mean())),
             "K1": err(X1), "K2": err(X2), "K3": err(X3), **diag}
        out.append(r)
        print(r, flush=True)
    for k in ("none", "K1", "K2", "K3"):
        print(k, "mean rot err %.2f deg, pos err %.2f cm" % (np.mean([o[k][0] for o in out]), np.mean([o[k][1] for o in out])))
    (KOUT / "design/synth_test3.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
