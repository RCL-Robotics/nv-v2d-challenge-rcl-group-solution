#!/usr/bin/env python3
"""k3only_choose.py: step 1 of fc_runs/submissions/t2_k3only_20261102 (Tier 2, K3-only file).

Own-input only. Reads ONLY the snapshot copies in inputs/ (own_input_evals.csv, summary_own_input.json,
manifest.json, manifest_stageB.jsonl, VERDICT_k3full.json). Refuses to open any path containing
"LOCAL_NOT_FOR_UPLOAD", "clean" or "packed_".

1. K3FULL composition = summary_own_input.json composition_jobs["K3FULL"] (the 2026-10-24 files' own definition:
   PREREG.md section 4 "K3FULL | PREREG_v3, as VERDICT.json gives it; PENDING maps to raw and is flagged").
   Re-derived independently from VERDICT_k3full.json + manifest rows and required to match.
2. Trial k = run k (its ok try) of every episode's K3FULL job, k = 1..5.
   T_k = mean over the 30 episodes of own-input AUC of run k. M = mean of T_1..T_5.
   Chosen trial = argmin_k |T_k - M|, ties to the lower k (exact floats, no rounding).
3. Writes CHOICE.json, CHOICE.md (new files only); the caller writes CHOICE.sha256.

Run: <kit python> -I -B k3only_choose.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import csv  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

W = Path("/root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_k3only_20261102")
INP = W / "inputs"
FORBIDDEN = ("local_not_for_upload", "clean", "packed_")
GATE_FAIL = [0, 5, 15, 19, 22, 26, 28]


def guard(p) -> Path:
    for s in (str(p), os.path.realpath(str(p))):
        low = s.lower()
        hit = [t for t in FORBIDDEN if t in low]
        if hit:
            raise PermissionError(f"refusing {p} (contains {hit})")
    return Path(p)


def sha(p) -> str:
    h = hashlib.sha256()
    with guard(p).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rj(p):
    return json.loads(guard(p).read_text())


def write_new(p: Path, text: str) -> None:
    p = guard(p)
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, p)


def main() -> int:
    if not sys.flags.dont_write_bytecode or not sys.flags.isolated:
        print("run with python -I -B")
        return 2
    problems: list[str] = []
    summ = rj(INP / "summary_own_input.json")
    comp = {int(k): v for k, v in summ["composition_jobs"]["K3FULL"].items()}
    if sorted(comp) != list(range(30)):
        raise SystemExit("K3FULL does not list 30 episodes")

    # independent re-derivation from VERDICT + manifest (PREREG.md 1: CHOSEN K3FIX = the K3fix run's final.json,
    # RAW = raw final.json, PENDING -> raw final.json)
    verdict = rj(INP / "VERDICT_k3full.json")
    rows = rj(INP / "manifest.json")["rows"]
    for line in guard(INP / "manifest_stageB.jsonl").read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    byjob = {r["job"]: r for r in rows}
    derived, vchoice = {}, {}
    for x in verdict["episodes"]:
        e = int(x["episode"])
        vchoice[e] = x["choice"]
        if x["choice"] == "K3FIX":
            run_dir = "/root/autodl-tmp/haoran.zhang/fc_runs/" + x["k3fix_run"]
            c = [r for r in rows if r["episode"] == e and r["run_dir"] == run_dir and r["iteration"] == x["final_iteration"]
                 and r["designation"] == "final.json"]
        else:
            c = [r for r in rows if r["episode"] == e and r["family"] == "raw" and r["designation"] == "final.json"]
        if len(c) != 1:
            problems.append(f"ep{e:02d}: re-derivation found {len(c)} candidate jobs")
            continue
        derived[e] = c[0]["job"]
    for e in range(30):
        if derived.get(e) != comp[e]:
            problems.append(f"ep{e:02d}: summary K3FULL {comp[e]} != re-derived {derived.get(e)}")

    # own-input evaluations
    evals: dict[str, dict[int, dict]] = {}
    with guard(INP / "own_input_evals.csv").open(newline="") as fp:
        for r in csv.DictReader(fp):
            if r["status"] != "ok":
                continue
            j, k = r["job"], int(r["run"])
            if k in evals.setdefault(j, {}):
                problems.append(f"{j} run {k}: more than one ok row")
            evals[j][k] = r
    per_ep = {}
    for e in range(30):
        j = comp[e]
        runs = evals.get(j, {})
        if sorted(runs) != [1, 2, 3, 4, 5]:
            problems.append(f"ep{e:02d} {j}: ok runs {sorted(runs)} != 1..5")
        m = byjob[j]
        per_ep[e] = {"job": j, "family": m["family"], "input": m["root_kind"], "verdict_choice": vchoice.get(e),
                     "iteration": m["iteration"], "run_dir": m["run_dir"],
                     "training_sha256": m["training_sha256"], "parquet_sha256": m["parquet_sha256"],
                     "runs": {k: {"auc": float(runs[k]["auc"]), "sp_sr": float(runs[k]["sp_sr"]),
                                  "mp_sr": float(runs[k]["mp_sr"]), "mppe": float(runs[k]["mppe"]),
                                  "try": int(runs[k]["try"]), "export_sha256_16": runs[k]["export_sha256"]}
                              for k in sorted(runs)}}
        # cross-check against the summary JSON's own per-episode values
        sp = summ["compositions"]["K3FULL"]["per_episode"][str(e)]
        for k in range(1, 6):
            a1 = per_ep[e]["runs"].get(k, {}).get("auc")
            a2 = sp["runs"].get(str(k), {}).get("auc")
            if a1 is None or a2 is None or abs(a1 - a2) > 1e-9:
                problems.append(f"ep{e:02d} run {k}: csv auc {a1} != summary {a2}")
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        return 1

    trials = {}
    for k in range(1, 6):
        t = {m: sum(per_ep[e]["runs"][k][m] for e in range(30)) / 30 for m in ("auc", "sp_sr", "mp_sr", "mppe")}
        trials[k] = t
    mean = {m: sum(trials[k][m] for k in range(1, 6)) / 5 for m in ("auc", "sp_sr", "mp_sr", "mppe")}
    dist = {k: abs(trials[k]["auc"] - mean["auc"]) for k in range(1, 6)}
    chosen = min(range(1, 6), key=lambda k: (dist[k], k))
    n_k3 = sum(1 for e in range(30) if per_ep[e]["input"] in ("k3fix", "k3"))
    n_raw = sum(1 for e in range(30) if per_ep[e]["input"] == "raw")
    rec = {
        "written_at": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S CST"),
        "script_sha256": sha(Path(__file__).resolve()),
        "rule": "trial k = run k of every episode's K3FULL job; T_k = 30-episode mean own-input AUC; chosen = "
                "argmin |T_k - mean_k T_k|, ties to the lower k",
        "own_input_only": True,
        "inputs": {p.name: sha(p) for p in sorted(INP.iterdir()) if p.is_file()},
        "composition": "K3FULL (t2_best5_20261024 PREREG section 4; summary_own_input.json composition_jobs)",
        "n_k3fix_or_k3": n_k3, "n_raw": n_raw,
        "gate_fail_raw": GATE_FAIL,
        "pending_to_raw": [e for e in range(30) if vchoice.get(e) == "PENDING"],
        "episodes": {f"{e:02d}": per_ep[e] for e in range(30)},
        "trials_own_input": {str(k): trials[k] for k in range(1, 6)},
        "mean_of_trials_own_input": mean,
        "abs_dist_to_mean_auc": {str(k): dist[k] for k in range(1, 6)},
        "chosen_trial": chosen,
        "chosen_trial_own_input": trials[chosen],
    }
    write_new(W / "CHOICE.json", json.dumps(rec, indent=1) + "\n")
    L = ["# Tier 2 K3-only file: trial choice (own-input only)", "",
         f"Written {rec['written_at']} by `tools/k3only_choose.py` (sha256 {rec['script_sha256']}), BEFORE any clean "
         "number for this file was opened. Inputs: snapshot copies in `inputs/` (sha256 in CHOICE.json and "
         "INPUTS_SNAPSHOT.sha256); no clean / LOCAL_NOT_FOR_UPLOAD / packed file was read.", "",
         "**Composition (fixed, not chosen here):** K3FULL of `t2_best5_20261024` (PREREG section 4: PREREG_v3 as "
         "VERDICT.json gives it; PENDING maps to raw). "
         f"{n_k3} episodes on our K3fix / K3 easy-clip runs (final.json), {n_raw} on our raw C3r2_tier2 runs (final.json): "
         f"gate-FAIL {', '.join(f'{e:02d}' for e in GATE_FAIL)}, plus ep04 (gate PASS, but its K3fix run never finished: "
         "VERDICT PENDING -> raw). No G2, nothing of Zuxing's.", "",
         "**Rule:** trial k = run k (its ok try) of every episode's job; T_k = mean over the 30 episodes of own-input "
         "AUC; chosen = the trial whose T_k is closest to the mean of T_1..T_5, ties to the lower k.", "",
         "| trial | own AUC | own SP-SR | own MP-SR | own MPPE (cm) | abs(T_k - mean) |", "|---:|---:|---:|---:|---:|---:|"]
    for k in range(1, 6):
        t = trials[k]
        L.append(f"| {k}{' (chosen)' if k == chosen else ''} | {t['auc']:.6f} | {t['sp_sr']:.4f} | {t['mp_sr']:.4f} | "
                 f"{t['mppe']:.3f} | {dist[k]:.6f} |")
    L.append(f"| mean | {mean['auc']:.6f} | {mean['sp_sr']:.4f} | {mean['mp_sr']:.4f} | {mean['mppe']:.3f} | |")
    L += ["", f"**CHOSEN TRIAL: {chosen}** (own-input AUC {trials[chosen]['auc']:.6f}, SP {trials[chosen]['sp_sr']:.4f}, "
          f"MP {trials[chosen]['mp_sr']:.4f}, MPPE {trials[chosen]['mppe']:.3f} cm).", "",
          "Own-input composites mix references (raw / K3fix / K3); they are not leaderboard numbers.", "",
          "| ep | job | input | VERDICT | run-chosen own AUC | 5-run own AUC mean |", "|---:|---|---|---|---:|---:|"]
    for e in range(30):
        p = per_ep[e]
        mu = sum(p["runs"][k]["auc"] for k in range(1, 6)) / 5
        L.append(f"| {e:02d} | {p['job']} | {p['input']} | {p['verdict_choice']} | {p['runs'][chosen]['auc']:.4f} | {mu:.4f} |")
    write_new(W / "CHOICE.md", "\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
