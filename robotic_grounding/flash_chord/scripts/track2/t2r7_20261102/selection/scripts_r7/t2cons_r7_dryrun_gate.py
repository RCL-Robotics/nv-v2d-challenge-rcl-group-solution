#!/usr/bin/env python3
"""t2cons_r7_dryrun_gate.py: plumbing check of PREREG_R7 3.6 / 5.1 on the two round-7 dry-run evaluations, OWN INPUT ONLY.

For each dryrun/eval_<phase>_*/ folder it re-scores the dry-run rollout against the job's OWN input reference with the
pinned round-1 functions (t2cons_consensus.py e1b943b1: load_kit, grid, rollout_on_grid, metrics_vs; t2cons_a1_gate.py
a29c997b: auc_seed, vertex seeds 0..9), exactly as the A1.3 validation gate will, and reports whether the evaluator's
own-input values reproduce (AUC inside [min_s - 0.02, max_s + 0.02], |dMPPE| <= 0.5 cm, SP-SR and MP-SR equal). No
consensus reference, no clean file and no packer is opened; the values decide nothing (plumbing evidence only).

Writes (new file only): dryrun/gate_dryrun_r7.json
Run: conda_envs/v2dkit/bin/python -I -B scripts/t2cons_r7_dryrun_gate.py
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
ROOT = ME / "fc_runs/submissions/t2_consensus_r7_20261102"
CONS, CONS_SHA = ME / "scripts/t2cons_consensus.py", "e1b943b10fd1db13ab6f78e5bde7b3c6e6c48f7daeee6230c32c463d14ac41b4"
GATE, GATE_SHA = ME / "scripts/t2cons_a1_gate.py", "a29c997b7271e9299db109e97af5210f528484719210bb58685523b0ae0eb16b"
OUT = ROOT / "dryrun/gate_dryrun_r7.json"
EXTRA_FORBIDDEN = ("best_weights", "seed_best", "remote_reps", "report_r", "composites", "per_episode_zx", "readme",
                   "train_rl.log", "t2_zx07_20261028", "tier2_eval.csv", "tier2_runs.csv", "summary.txt",
                   "tier2_best_scan", "eval_gen_local")  # fmt: skip


def _load(name: str, path: Path, want: str):
    src = path.read_bytes()
    got = hashlib.sha256(src).hexdigest()
    if got != want:
        raise SystemExit(f"{path} sha256 {got} != pinned {want}")
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)  # noqa: S102 (verified pinned bytes)
    return mod


C = _load("t2cons_consensus_pinned_r7dry", CONS, CONS_SHA)
C.FORBIDDEN = tuple(C.FORBIDDEN) + EXTRA_FORBIDDEN
G = _load("t2cons_a1_gate_pinned_r7dry", GATE, GATE_SHA)  # its own import of t2cons_consensus.py: same verified file
if hashlib.sha256(CONS.read_bytes()).hexdigest() != CONS_SHA:
    raise SystemExit("t2cons_consensus.py changed while loading")


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    K = C.load_kit()
    pool = {r["job"]: r for r in json.loads(C.guard(ROOT / "inputs_r7/pool_r7.json").read_text())["rows"]}
    incs = {x["job"]: x for x in json.loads(C.guard(ROOT / "inputs_r7/incumbents_r7.json").read_text())["incumbents"].values()}
    out = {"written_at": C.now(), "script_sha256": C.sha256_file(Path(__file__).resolve()), "consensus_sha256": CONS_SHA,
           "gate_sha256": GATE_SHA, "note": "OWN-INPUT re-score of the dry-run rollouts only (PREREG_R7 3.6 / 5.1); "
           "plumbing evidence, decides nothing; no consensus reference, no clean file", "runs": []}  # fmt: skip
    ok_all = True
    for d in sorted((ROOT / "dryrun").glob("eval_*_*/")):
        res = json.loads(C.guard(d / "result.json").read_text())
        job = res["job"]
        rr = json.loads(C.guard(d / job / "run1" / "run.json").read_text())
        src = pool.get(job) or incs.get(job)
        kind = src["input"]
        e = int(src["episode"])
        exp = res["export"]
        jd = {"job": job, "episode": e, "export_sha256": exp["sha256"], "input_reference": str(C.INPUT_REF[kind])}
        rec = {"parquet": rr["results_parquet"]}
        frames, names = C.grid(K, e)
        converted, inref, info = C.rollout_on_grid(K, jd, rec, frames, names)
        m0 = C.metrics_vs(K, converted, inref, names)
        cache: dict = {}
        aucs = [G.auc_seed(K, cache, converted, inref, names, s) for s in G.SEEDS]
        ev = rr["metrics"]
        lo, hi = min(aucs) - G.BAND, max(aucs) + G.BAND
        checks = {"auc_in_band": lo <= ev["auc"] <= hi, "abs_d_mppe_le_0.5": abs(m0["mppe"] - ev["mppe"]) <= 0.5,
                  "sp_equal": m0["sp_sr"] == ev["sp_sr"], "mp_equal": m0["mp_sr"] == ev["mp_sr"],
                  "seed0_reproduces_metrics_vs": abs(aucs[0] - m0["auc"]) <= G.REPRO_TOL}  # fmt: skip
        ok = all(checks.values())
        ok_all &= ok
        out["runs"].append({"dir": str(d), "phase": res["phase"], "job": job, "episode": e, "input": kind,
                            "input_reference": jd["input_reference"],
                            "input_reference_file": str(C.ep_file(Path(jd["input_reference"]), e)),
                            "rollout": rec["parquet"], "rollout_sha256": C.sha256_file(rec["parquet"]),
                            "export_sha256": exp["sha256"], "evaluator_own_input": ev, "rescored_own_input_seed0": m0,
                            "auc_seeds_0_9": aucs, "band": [lo, hi], "conversion": info, "checks": checks, "pass": ok})  # fmt: skip
        print(f"{job} ({res['phase']}, {kind}): evaluator AUC {ev['auc']:.4f} vs re-scored seeds [{min(aucs):.4f}, "
              f"{max(aucs):.4f}] -> band [{lo:.4f}, {hi:.4f}]; MPPE {ev['mppe']:.3f} vs {m0['mppe']:.3f}; SP {ev['sp_sr']} "
              f"vs {m0['sp_sr']}; MP {ev['mp_sr']} vs {m0['mp_sr']}; conversion residual {info['conv_pos_residual_m']:.2e} m "
              f"{info['conv_ang_residual_rad']:.2e} rad: {'PASS' if ok else 'FAIL'}")
    out["pass"] = ok_all and len(out["runs"]) == 2
    tmp = OUT.with_name(OUT.name + ".tmp")
    tmp.write_text(json.dumps(out, indent=1, default=str) + "\n")
    os.replace(tmp, OUT)
    os.chmod(OUT, 0o444)
    print(f"gate dry run: {'PASS' if out['pass'] else 'FAIL'} -> {OUT}")
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
