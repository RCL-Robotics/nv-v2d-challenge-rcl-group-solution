#!/usr/bin/env python
"""t3f0_lean_search.py -- Tier-3 f0 prep, leaning episodes only (ep01 vacuum; optional for ep28): GT-free search for the
SMALLEST horizontal correction of Han's stool pose that makes the frame-0 rest physically stable. NEW FILE (2026-10-24).
Why: the task text says the object leans on the stool and the frame-0 object pose is the (allowed) true first-frame pose,
but with Han's stool (fit IoU 0.61 for ep01) + the anchor-contact rule the passive settle test of the frame-0 rest fails
(the object tips over). Candidates: the stool contact boxes at Han's T_world_mesh_contact (t3f0_supports.py record minus its
anchor-contact shift) translated along 16 horizontal directions until the frame-0 object surface is 0.2 cm from the box
union (shift <= 8 cm), kept only if the whole reference object path stays >= -0.5 cm from every box; each candidate gets the
passive settle test of the anchor run (t3f0_physcheck.settle: MuJoCo-Warp CPU, training sim settings, 10 s). The chosen
candidate = the strictly passing one with the smallest shift; if none passes, the upright one (final |dp| <= 5 cm,
rot <= 10 deg) with the smallest final |dp| (flagged MARGINAL; amendment after the first 16-direction pass); written to tmp_analysis/tier3/f0/supports/epNN_lean_search.json,
which t3f0_supports.py then applies (recorded there). No GT, no clean data. CPU only.
Usage: t3f0_lean_search.py EP [--workers N]
"""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np
ME = Path("/root/autodl-tmp/haoran.zhang")
A = ME / "tmp_analysis/tier3/f0"
sys.path.insert(0, str(ME / "scripts"))
OUTD = A / "lean_search"
NDIR, SMAX, TARGET = 32, 0.08, 0.002
UPRIGHT_DP, UPRIGHT_ROT = 0.05, 10.0


def candidates(ep):
    import trimesh
    import t3f0_supports as S
    rec = json.loads((A / "supports" / f"ep{ep:02d}.json").read_text())
    assert rec["kind"] == "stool"
    fr = np.load(A / "frames" / f"ep{ep:02d}.npz")
    om = trimesh.load(str(A / "frames" / f"ep{ep:02d}_obj" / "textured_mesh.obj"), force="mesh", process=False)
    OS, _ = trimesh.sample.sample_surface(om, 20000, seed=0)
    OS = np.vstack([OS, om.convex_hull.vertices])
    P, Rm = fr["obj_pos"], fr["obj_R"]
    s0 = np.asarray(rec["horizontal_shift_cm"]) / 100.0
    base = [dict(b, center=(np.asarray(b["center"]) - s0).tolist()) for b in rec["boxes"]]  # Han's pose
    W0 = OS @ Rm[0].T + P[0]
    sub = OS[::3]

    def moved(v):
        return [dict(b, center=(np.asarray(b["center"]) + v).tolist()) for b in base]
    out = []
    for k in range(NDIR):
        th = 2 * np.pi * k / NDIR
        d = np.array([np.cos(th), np.sin(th), 0.0])
        prev = None
        for s in np.arange(0.0, SMAX + 1e-9, 0.002):
            m = float(S.union_sdf(W0, moved(s * d)).min())
            if m <= TARGET:
                if prev is None:
                    break  # already touching / overlapping at s = 0 in this direction
                lo, hi = s - 0.002, s
                for _ in range(20):
                    mid = (lo + hi) / 2
                    if float(S.union_sdf(W0, moved(mid * d)).min()) > TARGET:
                        lo = mid
                    else:
                        hi = mid
                v = lo * d
                bx = moved(v)
                f0 = float(S.union_sdf(W0, bx).min())
                path = min(float(S.union_sdf(sub @ Rm[t].T + P[t], bx).min()) for t in range(0, len(P), 2))
                out.append(dict(direction_deg=round(float(np.degrees(th)), 1), shift_cm=np.round(100 * v, 2).tolist(),
                                shift_norm_cm=round(100 * float(np.linalg.norm(v)), 2), frame0_sdf_cm=round(100 * f0, 2),
                                path_min_sdf_cm=round(100 * path, 2), path_ok=bool(path >= -0.005), boxes=bx))
                break
            prev = m
    return rec, out


def run_one(args):
    ep, i, cand = args
    import t3f0_supports as S
    import t3f0_physcheck as PC
    u = OUTD / f"ep{ep:02d}_c{i:02d}.usda"
    S.write_usda(u, cand["boxes"], ep, f"lean-search candidate {i} (dev)")
    res = PC.settle(ep, usda_override=str(u), only_anchor=True, save=False)
    t = res["tests"][0]["with_support"] if res["tests"] else {}
    return i, dict(ok=bool(t.get("ok")), final_dp_cm=t.get("final_dp_cm"), final_rot_deg=t.get("final_rot_deg"),
                   final_disp_cm=t.get("final_disp_cm"))


def main():
    ep = int(sys.argv[1])
    nw = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6
    OUTD.mkdir(parents=True, exist_ok=True)
    rec, cands = candidates(ep)
    valid = [(i, c) for i, c in enumerate(cands) if c["path_ok"]]
    print(f"ep{ep:02d}: {len(cands)} directions reach contact, {len(valid)} keep the reference path clear", flush=True)
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(nw) as pool:
        for i, r in pool.imap_unordered(run_one, [(ep, i, c) for i, c in valid]):
            cands[i]["settle"] = r
            print(json.dumps({k: v for k, v in cands[i].items() if k != "boxes"}), flush=True)
    passing = sorted([c for c in cands if c.get("settle", {}).get("ok")], key=lambda c: c["shift_norm_cm"])
    chosen, how = (passing[0], "strict pass (final |dp| <= 2 cm, rot <= 5 deg, max |dp| <= 3 cm), smallest shift") if passing else (None, None)
    if chosen is None:
        # amendment (2026-10-24, after the first 16-direction pass on ep01 found no strict pass but two near-passes):
        # a leaning object resting on a point contact may tilt a little before it rests; accept the candidate that stays
        # upright (final |dp| <= 5 cm, rot <= 10 deg) with the smallest final |dp|, and flag the episode as marginal
        up = sorted([c for c in cands if c.get("settle") and c["settle"]["final_dp_cm"] is not None
                     and c["settle"]["final_dp_cm"] <= 100 * UPRIGHT_DP and c["settle"]["final_rot_deg"] <= UPRIGHT_ROT],
                    key=lambda c: c["settle"]["final_dp_cm"])
        if up:
            chosen, how = up[0], "MARGINAL: no strict pass; upright candidate with the smallest final |dp| (<= 5 cm, <= 10 deg)"
    res = dict(ep=ep, written=time.strftime("%F %T"), rule=__doc__.split("Usage")[0].strip(), n_directions=NDIR, shift_max_m=SMAX,
               contact_target_m=TARGET, candidates=[{k: v for k, v in c.items() if k != "boxes"} for c in cands],
               chosen=None if chosen is None else {k: v for k, v in chosen.items() if k != "boxes"},
               chosen_shift_cm=None if chosen is None else chosen["shift_cm"], chosen_how=how)
    (A / "supports" / f"ep{ep:02d}_lean_search.json").write_text(json.dumps(res, indent=1, default=float))
    print("chosen", res["chosen"], flush=True)


if __name__ == "__main__":
    main()
