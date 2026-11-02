"""SYNTHETIC self-test of the K0/K1 machinery and the ZUPT v1 baseline (no dataset reference data at all).
A simulated rest-move-rest object trajectory gets a world-left random-walk bias with the d1 noisy-only steps; we check
that K1 reduces the error vs the simulated truth and compare it to ZUPT v1 (filter_apply.zupt, copied verbatim)."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent))
from kfilter import Episode  # noqa: E402
from kcommon import KOUT, noise_sigmas, static_mask  # noqa: E402

PRM = json.loads((KOUT / "design/frozen_params.json").read_text())


def zupt(p_cm, rot, static):
    T = len(p_cm)
    corr_p = np.zeros(3); corr_r = R.identity()
    pf = np.empty_like(p_cm); rf = [None] * T
    for t in range(T):
        if t > 0 and static[t] and static[t - 1]:
            corr_p = corr_p + (p_cm[t] - p_cm[t - 1])
            corr_r = (rot[t] * rot[t - 1].inv()) * corr_r
        pf[t] = p_cm[t] - corr_p
        rf[t] = corr_r.inv() * rot[t]
    return pf, R.concatenate(rf)


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def simulate(seed, T=900):
    rng = np.random.default_rng(seed)
    segs = [(0, 150, None), (150, 300, "m"), (300, 450, None), (450, 560, "m"), (560, 700, None), (700, 800, "m"), (800, T, None)]
    rv = np.zeros((T, 3)); p = np.zeros((T, 3))
    cur_r = R.from_rotvec(rng.normal(size=3)); cur_p = np.array([0, 50.0, 200.0])
    rots = []
    for a, b, k in segs:
        if k is None:
            for t in range(a, b):
                rots.append(cur_r); p[t] = cur_p
        else:
            ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
            ang = np.radians(rng.uniform(30, 120)); dp = rng.normal(size=3) * 30
            for i, t in enumerate(range(a, b)):
                s = smoothstep((i + 1) / (b - a))
                rots.append(R.from_rotvec(ax * ang * s) * cur_r); p[t] = cur_p + dp * s
            cur_r = R.from_rotvec(ax * ang) * cur_r; cur_p = cur_p + dp
    Rt = R.concatenate(rots)
    bth = np.cumsum(rng.normal(size=(T, 3)) * np.radians(PRM["q_br_deg"]), 0); bth[0] = 0
    bp = np.cumsum(rng.normal(size=(T, 3)) * PRM["q_bp_cm"], 0); bp[0] = 0
    Rm = R.from_rotvec(bth) * Rt
    pm = p + bp
    return p, Rt, pm, Rm


def err(pa, ra, p, Rt):
    return float(np.degrees((ra.inv() * Rt).magnitude()).mean()), float(np.linalg.norm(pa - p, axis=1).mean())


def main():
    out = []
    for seed in range(20):
        p, Rt, pm, Rm = simulate(seed)
        sp, sr = noise_sigmas(pm, Rm)
        st = static_mask(pm, Rm, sp, np.radians(sr))
        E = object.__new__(Episode)
        E.prm = PRM; E.p, E.rot = pm, Rm; E.T = len(pm); E.vis = np.ones(E.T, bool)
        inc = (Rm[1:] * Rm[:-1].inv()).as_rotvec(); E.phi = np.vstack([np.zeros(3), np.cumsum(inc, 0)])
        E.rest = st; E.rest_raw = st
        E.qa, E.qal = np.median(list(PRM["q_a_cm2_s3"].values())), np.median(list(PRM["q_al_rad2_s3"].values()))
        r = {"seed": seed, "none": err(pm, Rm, p, Rt)}
        for v, zv in (("K0", False), ("K1", True)):
            X, _, _, _ = E.run(E.base_meas(zv=zv))
            r[v] = err(X[:, 0:3], R.from_rotvec(-X[:, 15:18]) * Rm, p, Rt)
        pz, rz = zupt(pm, Rm, st)
        r["zupt"] = err(pz, rz, p, Rt)
        r["rest_frac"] = float(st.mean())
        out.append(r)
        print(r, flush=True)
    for k in ("none", "K0", "K1", "zupt"):
        print(k, "mean rot err %.2f deg, pos err %.2f cm" % (np.mean([o[k][0] for o in out]), np.mean([o[k][1] for o in out])))
    (KOUT / "design/synth_test.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
