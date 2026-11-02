#!/usr/bin/env python3
"""t2cons_a1_gate.py: AMENDMENTS.md A1 re-check of fc_runs/submissions/t2_consensus_20261024 (amended PREREG 3.5.2 gate).

For every run of every current candidate job (the 80 jobs t2cons_consensus.py validates: 400 runs), it scores the
rollout against the job's OWN training input exactly as t2cons_consensus.py validate does (same imported functions,
seed 0), and again with the released-mesh vertex sample of seeds 1..9. Then it applies:

  A1-GATE: min_s AUC_s - 0.02 <= eval_AUC <= max_s AUC_s + 0.02 (s = 0..9), for every run
  unchanged: max |dMPPE| <= 0.5 cm; SP-SR equal in >= 95 % of runs; MP-SR equal in >= 95 % of runs (seed 0)

Cross-checks: seed 0 must reproduce scores/own_input_validation.csv (mine_*) exactly; the seed range is compared with
diag/validation_vertex_sensitivity.json (report). No consensus reference exists or is read; nothing is selected here.
Every path goes through the PREREG 5.7 guard of t2cons_consensus.py.

Writes (new files only): scores/own_input_validation_a1.csv, scores/own_input_validation_a1.json
Run: conda_envs/v2dkit/bin/python -I -B scripts/t2cons_a1_gate.py
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import importlib.util  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
spec = importlib.util.spec_from_file_location("t2cons_consensus", ME / "scripts/t2cons_consensus.py")
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

ROOT = C.ROOT
AMEND, AMEND_SHA = ROOT / "AMENDMENTS.md", ROOT / "AMENDMENTS.sha256"
OUT_CSV, OUT_JSON = ROOT / "scores/own_input_validation_a1.csv", ROOT / "scores/own_input_validation_a1.json"
GATE_LINE = "A1-GATE: min_s AUC_s - 0.02 <= eval_AUC <= max_s AUC_s + 0.02 (s = 0..9), for every run"
SEEDS = tuple(range(10))
BAND = 0.02
REPRO_TOL = 1e-12
GATE = {"auc_band_around_seed_range": BAND, "seeds": list(SEEDS), "max_abs_d_mppe": 0.5, "min_sp_equal_frac": 0.95,
        "min_mp_equal_frac": 0.95}  # fmt: skip


def seed_vertices(K, cache: dict, name: str, s: int) -> np.ndarray:
    if (name, s) not in cache:
        v0 = C.vertices(K, name)  # seed 0 via the validated path; checks the mesh sha256 against inputs/TRACKS.sha256
        mesh = K.verts[name][1]
        vs = K.sample_vertices(C.guard(mesh), 500, s)
        if s == 0 and not np.array_equal(vs, v0):
            raise SystemExit(f"{name}: seed-0 sample differs from t2cons_consensus.vertices()")
        cache[(name, s)] = vs
    return cache[(name, s)]


def auc_seed(K, cache, converted, reference, names, s: int) -> float:
    T, B = len(converted) // len(names), len(names)
    achieved = K.op._normalise_quaternions(converted.reshape(T, 1, B, 7), "submission", ValueError)
    ref = K.op._normalise_quaternions(np.asarray(reference, dtype=np.float64).reshape(T, 1, B, 7), "reference", ValueError)
    verts = np.stack([seed_vertices(K, cache, n, s) for n in names])
    ids = np.arange(B, dtype=np.int64)
    return float(K.op.episode_metric("add_auc", achieved, ref, ids, verts, align_initial_pose=K.align))


def main() -> int:
    if not sys.flags.dont_write_bytecode or not sys.flags.isolated:
        print("run with the kit python -I -B")
        return 2
    fatal, info = C.check_pins(need_pins=False)
    for x in info:
        print("  " + x)
    if fatal:
        print("FATAL:\n  " + "\n  ".join(fatal))
        return 2
    amend_sha = C.sha256_file(AMEND)
    if not AMEND_SHA.is_file() or amend_sha != (AMEND_SHA.read_text().split() or [""])[0]:
        print("FATAL: AMENDMENTS.md != AMENDMENTS.sha256")
        return 2
    if GATE_LINE not in AMEND.read_text():
        print("FATAL: AMENDMENTS.md does not state the A1 gate this script implements")
        return 2
    cons_sha = C.sha256_file(ME / "scripts/t2cons_consensus.py")
    st = C.read_json(ROOT / "selftest/selftest.json")
    if st.get("pass") is not True or st.get("script_sha256") != cons_sha:
        print("FATAL: selftest/selftest.json is not a PASS of t2cons_consensus.py")
        return 2
    if OUT_CSV.exists() or OUT_JSON.exists():
        print("FATAL: output exists (new files only)")
        return 2
    val = {(r["job"], int(r["run"])): r for r in C.read_csv(ROOT / "scores/own_input_validation.csv")}
    diag = {}
    for e, v in C.read_json(ROOT / "diag/validation_vertex_sensitivity.json")["episodes"].items():
        for r in v["runs"]:
            diag[(r["job"], int(r["run"]))] = r

    before = C.integrity()
    K = C.load_kit()
    jobs = C.load_jobs()
    evals = C.load_evals(jobs)
    cand = C.candidates(jobs, evals)
    cache: dict = {}
    grids: dict = {}
    rows, problems = [], []
    for job in cand:
        j = jobs[job]
        e = j["episode"]
        if e not in grids:
            grids[e] = C.grid(K, e)
        frames, names = grids[e]
        for k in range(1, C.NRUNS + 1):
            rec = evals[job][k]
            row = {"job": job, "episode": e, "run": k, "try": rec["try"], "input": j["input"], "source": j["source"]}
            try:
                conv, inref, inf = C.rollout_on_grid(K, j, rec, frames, names)
                m0 = C.metrics_vs(K, conv, inref, names)
                aucs = [m0["auc"]] + [auc_seed(K, cache, conv, inref, names, s) for s in SEEDS[1:]]
            except ValueError as err:
                row.update({"status": "failed", "problem": str(err)})
                problems.append(f"{job} run {k}: {err}")
                rows.append(row)
                continue
            ev = {m: rec[m] for m in C.METRICS}
            lo, hi = min(aucs), max(aucs)
            excess = max(lo - ev["auc"], ev["auc"] - hi, 0.0)
            vr = val.get((job, k))
            repro = None if vr is None else max(abs(m0[m] - float(vr[f"mine_{m}"])) for m in C.METRICS)
            dg = diag.get((job, k))
            row.update({
                "status": "ok", "eval_auc": ev["auc"], **{f"auc_s{s}": a for s, a in zip(SEEDS, aucs)},
                "seed_min": lo, "seed_max": hi, "seed_range": hi - lo, "excess_outside_range": excess,
                "band_ok": bool(lo - BAND <= ev["auc"] <= hi + BAND), "inside_range": bool(lo <= ev["auc"] <= hi),
                "d_auc_seed0": m0["auc"] - ev["auc"],
                "eval_mppe": ev["mppe"], "mine_mppe": m0["mppe"], "d_mppe": m0["mppe"] - ev["mppe"],
                "eval_sp_sr": ev["sp_sr"], "mine_sp_sr": m0["sp_sr"], "eval_mp_sr": ev["mp_sr"], "mine_mp_sr": m0["mp_sr"],
                "repro_vs_validation_max_abs": repro,
                "diag_seed_min_diff": None if dg is None else lo - dg["seed_min"],
                "diag_seed_max_diff": None if dg is None else hi - dg["seed_max"],
                "conv_pos_residual_m": inf["conv_pos_residual_m"], "conv_ang_residual_rad": inf["conv_ang_residual_rad"],
            })  # fmt: skip
            rows.append(row)
        print(f"  {job}: {len([r for r in rows if r['job'] == job and r['status'] == 'ok'])}/5 scored", flush=True)
    after = C.integrity()
    diff = C.integrity_diff(before, after)

    ok = [r for r in rows if r["status"] == "ok"]
    band_fail = [r for r in ok if not r["band_ok"]]
    max_excess = max(ok, key=lambda r: r["excess_outside_range"]) if ok else None
    d_mppe = max((abs(r["d_mppe"]) for r in ok), default=None)
    sp_eq = sum(r["mine_sp_sr"] == r["eval_sp_sr"] for r in ok) / len(ok) if ok else 0.0
    mp_eq = sum(r["mine_mp_sr"] == r["eval_mp_sr"] for r in ok) / len(ok) if ok else 0.0
    repro_missing = [f"{r['job']} r{r['run']}" for r in ok if r["repro_vs_validation_max_abs"] is None]
    repro_max = max((r["repro_vs_validation_max_abs"] for r in ok if r["repro_vs_validation_max_abs"] is not None),
                    default=None)  # fmt: skip
    repro_ok = not repro_missing and repro_max is not None and repro_max <= REPRO_TOL
    diag_max = max((max(abs(r["diag_seed_min_diff"]), abs(r["diag_seed_max_diff"])) for r in ok
                    if r["diag_seed_min_diff"] is not None), default=None)  # fmt: skip
    parts = {
        "auc_band": bool(ok) and not band_fail,
        "mppe": d_mppe is not None and d_mppe <= GATE["max_abs_d_mppe"],
        "sp_equal": sp_eq >= GATE["min_sp_equal_frac"],
        "mp_equal": mp_eq >= GATE["min_mp_equal_frac"],
        "no_problem_runs": not problems and len(ok) == len(rows) == len(cand) * C.NRUNS,
        "seed0_reproduces_validation": repro_ok,
        "kit_stock_unchanged": not diff,
    }
    by_input = {}
    for kind in sorted({r["input"] for r in ok}):
        xs = [r for r in ok if r["input"] == kind]
        by_input[kind] = {"n": len(xs), "n_inside_range": sum(r["inside_range"] for r in xs),
                          "max_excess_outside_range": max(r["excess_outside_range"] for r in xs),
                          "max_abs_d_auc_seed0": max(abs(r["d_auc_seed0"]) for r in xs),
                          "max_seed_range": max(r["seed_range"] for r in xs)}  # fmt: skip
    worst = sorted(ok, key=lambda r: -r["excess_outside_range"])[:10]
    out = {
        "written_at": C.now(), "script_sha256": C.sha256_file(Path(__file__).resolve()),
        "consensus_script_sha256": cons_sha, "amendments_sha256": amend_sha, "prereg": "3.5.2 as amended by AMENDMENTS A1",
        "gate": GATE_LINE, "thresholds": GATE, "pass": all(parts.values()), "parts": parts,
        "n_jobs": len(cand), "n_runs": len(rows), "n_scored": len(ok), "problems": problems,
        "n_band_fail": len(band_fail), "band_fail_runs": [f"{r['job']} r{r['run']}" for r in band_fail],
        "n_inside_seed_range": sum(r["inside_range"] for r in ok),
        "max_excess_outside_range": None if max_excess is None else {
            "job": max_excess["job"], "run": max_excess["run"], "excess": max_excess["excess_outside_range"],
            "eval_auc": max_excess["eval_auc"], "seed_min": max_excess["seed_min"], "seed_max": max_excess["seed_max"]},
        "max_seed_range": max((r["seed_range"] for r in ok), default=None),
        "max_abs_d_auc_seed0": max((abs(r["d_auc_seed0"]) for r in ok), default=None),
        "n_abs_d_auc_seed0_gt_0.02": sum(abs(r["d_auc_seed0"]) > 0.02 for r in ok),
        "max_abs_d_mppe": d_mppe, "sp_equal_frac": sp_eq, "mp_equal_frac": mp_eq,
        "seed0_vs_own_input_validation_csv_max_abs": repro_max, "seed0_repro_missing": repro_missing,
        "diag_seed_range_max_abs_diff": diag_max,
        "max_conv_pos_residual_m": max((r["conv_pos_residual_m"] for r in ok), default=None),
        "max_conv_ang_residual_rad": max((r["conv_ang_residual_rad"] for r in ok), default=None),
        "by_input": by_input,
        "worst_excess_runs": [{k: r[k] for k in ("job", "run", "eval_auc", "seed_min", "seed_max", "excess_outside_range",
                                                "d_auc_seed0")} for r in worst],
        "vertex_meshes": {n: {"file": str(v[1]), "sha256": C.sha256_file(v[1])} for n, v in sorted(K.verts.items())},
        "integrity_diff": diff[:50],
        "note": "own-input check only (each run vs its OWN training input); seeds 1..9 used only for this gate; "
                "no consensus reference exists or is read; nothing is selected",
    }  # fmt: skip
    cols = ["job", "episode", "run", "try", "input", "source", "status", "problem", "eval_auc"] + \
        [f"auc_s{s}" for s in SEEDS] + \
        ["seed_min", "seed_max", "seed_range", "excess_outside_range", "band_ok", "inside_range", "d_auc_seed0",
         "eval_mppe", "mine_mppe", "d_mppe", "eval_sp_sr", "mine_sp_sr", "eval_mp_sr", "mine_mp_sr",
         "repro_vs_validation_max_abs", "diag_seed_min_diff", "diag_seed_max_diff", "conv_pos_residual_m",
         "conv_ang_residual_rad"]  # fmt: skip
    C.write_text(OUT_CSV, C.csv_text(rows, cols))
    out["csv_sha256"] = C.sha256_file(OUT_CSV)
    C.write_text(OUT_JSON, C.dump(out))
    print(json.dumps({k: out[k] for k in ("pass", "parts", "n_runs", "n_scored", "n_band_fail", "n_inside_seed_range",
                                          "max_excess_outside_range", "max_seed_range", "max_abs_d_mppe",
                                          "sp_equal_frac", "mp_equal_frac", "seed0_vs_own_input_validation_csv_max_abs",
                                          "diag_seed_range_max_abs_diff")}, indent=1))  # fmt: skip
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
