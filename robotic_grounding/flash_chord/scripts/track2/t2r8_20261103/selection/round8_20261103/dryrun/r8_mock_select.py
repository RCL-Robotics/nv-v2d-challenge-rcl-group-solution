"""r8_mock_select.py: mock harness for r8_select.py (PREREG_R8 PINS 1.6). Runs cmd_select_s, cmd_select_f (with and
without finalists) and cmd_keep_r7 of r8_select.py with every round-8 input replaced by SYNTHETIC values and every write
captured in memory: nothing is written on disk. The round-7 records are the real frozen ones (incumbents, round-7
phase-S results of two real reuse7 / barred7 jobs, round-7 phase-F runs). Prints a summary; exit 0 iff every check holds.

  conda_envs/v2dkit/bin/python -I -B dryrun/r8_mock_select.py r8_select.py
"""
import json
import sys
import types
from pathlib import Path

import numpy as np

SRC = Path(sys.argv[1]).resolve()
mod = types.ModuleType("r8s_mock")
mod.__file__ = str(SRC)
sys.modules["r8s_mock"] = mod
exec(compile(SRC.read_bytes(), str(SRC), "exec"), mod.__dict__)
M, C = mod, mod.C
WRITTEN: dict[str, str] = {}
CHECKS: list[tuple[str, bool]] = []


def check(name, ok):
    CHECKS.append((name, bool(ok)))
    print(("ok   " if ok else "FAIL ") + name, flush=True)


def write_new(p, text):
    if str(p) in WRITTEN:
        raise SystemExit(f"refusing to overwrite {p}")
    WRITTEN[str(p)] = text


class AnyHash(dict):
    def __missing__(self, k):
        return "9" * 64


M.write_new = write_new
M.sha = lambda p: "f" * 64
M.require = lambda ph, stage: True
M.r8_docs = lambda require_lock=True: ([], ["mock"])
orig_read_json = C.read_json
MEM: dict[str, object] = {}
C.read_json = lambda p: MEM[str(p)] if str(p) in MEM else orig_read_json(p)
orig_hash_list = C.hash_list
C.hash_list = lambda p: AnyHash() if Path(p) == SRC else orig_hash_list(p)  # only the mock round-8 rollout list
M.SEL_S8 = SRC  # an existing file (hashes are mocked); hash_lines finds no line in it
M.R7C.PH["F8"]["rollouts"] = SRC
rng = np.random.default_rng(20261103)
incs = M.incumbents8()  # real round-7 records
check("30 incumbents = round-7 S7", len(incs) == 30)

# ---------------------------------------------------------------- phase S: synthetic round-8 screens + 2 real round-7 jobs
pool, jobs, evals, jst, flagged = {}, {}, {}, {}, set()
for e in range(30):
    n = 0 if e in (10, 11, 12) else int(rng.integers(1, 4))
    for i in range(n):
        job = f"ep{e:02d}_mock{i}"
        pool[job] = {"job": job, "episode": e, "role": "screen", "sha256": "a" * 64}
        jobs[job] = {"job": job, "episode": e, "export_status": "ok", "input": "g2", "owner": "zuxing", "machine": "5090",
                     "tree": "b39b69b", "iteration": 1000 * (i + 1), "training_sha256": "b" * 64, "pool_role": "screen",
                     "export_sha256": f"{e:02d}{i}" + "c" * 61, "export_file": f"/mock/{job}", "role": "screen"}
        base = incs[e]["m_P_frozen"]
        bump = 0.03 if i == 0 and e % 3 == 0 else -0.02
        aucs = [float(np.clip(base + bump + rng.normal(0, 0.003), 0, 1)) for _ in range(3)]
        sp = 0.0 if (e == 21 and i == 0) else 1.0
        evals[job] = {k: {"auc": 0.5, "sp_sr": sp, "mp_sr": sp, "mppe": 1.0, "try": 1, "parquet": f"/r/{job}/{k}"} for k in (1, 2, 3)}
        jst[job] = {"n": 3, "runs": [1, 2, 3], "auc": aucs, "mppe": [1.0] * 3, "sp_sr_stats": {"mean": 1.0}, "mp_sr_stats": {"mean": 1.0}}
pool["ep09_z5g2_180000"] = {"job": "ep09_z5g2_180000", "episode": 9, "role": "reuse7", "sha256": "d" * 64}
pool["ep02_z5q_180000"] = {"job": "ep02_z5q_180000", "episode": 2, "role": "barred7", "sha256": "e" * 64}
pool["ep22_z5k3fix_90000"] = {"job": "ep22_z5k3fix_90000", "episode": 22, "role": "barred7", "sha256": "e" * 64}
M.scored_inputs8 = lambda ph: ({"jobs": jst}, {"validation_failed_jobs": []}, jobs, evals, {}, flagged)
M.pool_rows = lambda: pool
rc = M.cmd_select_s()
check("select S rc 0", rc == 0)
sel = json.loads(WRITTEN[str(M.ROOT / "selection_s8.json")])
final = json.loads(WRITTEN[str(M.FINAL)])
fins = {x["episode"]: x for x in final["finalists"]}
check("episodes 10, 11, 12 (no candidate) are not evaluated and have no finalist",
      all(not sel["episodes"][e]["evaluated"] and e not in fins for e in (10, 11, 12)))
check("barred candidates are never finalists", all(x["job"] not in ("ep02_z5q_180000", "ep22_z5k3fix_90000") for x in final["finalists"]))
check("every finalist clears m_P_frozen + 0.01 and is S-eligible",
      all(x["cons_S_mean"] >= x["m_P_frozen"] + 0.01 and x["floor_S"] >= 2 for x in final["finalists"]))
ep09 = next(x for x in sel["episodes"] if x["episode"] == 9)
check("the real reuse7 job is a round-8 candidate of ep09 with its round-7 phase-S record",
      any(r["job"] == "ep09_z5g2_180000" and r["source"].startswith("round-7") for r in ep09["candidates"]))
check("SELECTION_S8.sha256 written last", list(WRITTEN)[-1] == str(M.SEL_S8) or str(M.ROOT / "SELECTION_S8.sha256") in WRITTEN)
print("finalists:", [(x["episode"], x["job"], round(x["cons_S_mean"], 4), round(x["bar_s"], 4)) for x in final["finalists"]])

# ---------------------------------------------------------------- phase F: synthetic finalist runs
MEM[str(M.FINAL)] = final
MEM[str(M.ROOT / "selection_s8.json")] = sel
jobsF, evalsF, scoresF = {}, {}, {}
for e, x in fins.items():
    job = x["job"]
    jobsF[job] = {"job": job, "episode": e, "role": "finalist", "input": "g2", "owner": "zuxing", "machine": "5090",
                  "input_reference": "/mock/ref", "export_sha256": x["export_sha256"], "export_file": x["export_file"],
                  "training_checkpoint": "/mock/ck", "source_checkpoint": "/mock/src", "iteration": x["iteration"]}
    nruns = 7 if e == 0 else 10  # ep00: a deadline-stopped finalist (7 of 10 runs)
    evalsF[job] = {}
    for k in range(1, nruns + 1):
        evalsF[job][k] = {"auc": 0.6 + 0.001 * k, "sp_sr": 1.0, "mp_sr": 1.0, "mppe": 2.0, "try": 1, "parquet": f"/f/{job}/{k}"}
        scoresF[(job, k)] = {"auc": str(incs[e]["bar"] + (-0.005 if e == 3 else 0.02)), "sp_sr": "1.0", "mp_sr": "1.0", "mppe": "3.0"}
M.scored_inputs8 = lambda ph: ({}, {}, jobsF, evalsF, scoresF, set())
rc = M.cmd_select_f()
check("select F rc 0", rc == 0)
s8 = json.loads(WRITTEN[str(M.ROOT / "SELECTION_R8.json")])
rep = set(s8["replaced_episodes"])
want = {f"{e:02d}" for e in fins if e not in (0, 3)}
check("replaced = finalists with 10 good runs and m_F >= bar (ep00 deadline-stopped and ep03 below the bar are not)", rep == want)
check("a kept episode's S8 job is its round-7 pick with round-7 runs",
      all(s8["compositions"]["S8"][f"{e:02d}"] == incs[e]["job"] and s8["compositions"]["S8_run_source"][f"{e:02d}"] == "r7F"
          for e in range(30) if f"{e:02d}" not in rep))
check("R8-CLOSEST exists and its file has 30 episodes", s8["r8_closest_trial"] is not None and len(s8["r8_closest_file"]) == 30)
check("SELECTION_R8.sha256 written", str(M.SEL_R8) in WRITTEN)
print("replaced:", sorted(rep), "R8-CLOSEST trial", s8["r8_closest_trial"])

# ---------------------------------------------------------------- phase F without finalists, and keep-r7 (real round-7 runs)
for no_fin in (True, False):
    WRITTEN.clear()
    if no_fin:
        MEM[str(M.FINAL)] = {**final, "finalists": []}
        rc = M.cmd_select_f()
        label = "select F without finalists"
    else:
        rc = M.cmd_keep_r7(types.SimpleNamespace(reason="mock: phase S not complete by its cut-off"))
        label = "keep-r7"
    s8 = json.loads(WRITTEN[str(M.ROOT / "SELECTION_R8.json")])
    check(f"{label}: rc 0, no replacement, S8 == S7, closest trial == round 7's trial 2 (real round-7 runs)",
          rc == 0 and not s8["replaced_episodes"] and s8["r8_closest_trial"] == 2
          and all(s8["compositions"]["S8"][f"{e:02d}"] == incs[e]["job"] for e in range(30)))
# ---------------------------------------------------------------- keep-r7 while an evaluation driver is alive (PREREG_R8 7.6)
WRITTEN.clear()
orig_running, orig_pid = M.driver_running, M.pidfile_live
M.driver_running = lambda: ["pid 1: python -I -B r8_eval.py run --phase screen (mock)"]
rc = M.cmd_keep_r7(types.SimpleNamespace(reason="mock: must be refused"))
check("keep-r7 refused while an r8_eval.py run process is alive (nothing written)", rc == 2 and not WRITTEN)
M.driver_running, M.pidfile_live = (lambda: []), (lambda: "driver.pid 1 (mock)")
rc = M.cmd_keep_r7(types.SimpleNamespace(reason="mock: must be refused"))
check("keep-r7 refused while driver.pid names a live r8_eval.py (nothing written)", rc == 2 and not WRITTEN)
M.driver_running, M.pidfile_live = orig_running, orig_pid
check("no live r8_eval.py on this host now; a missing / stale driver.pid does not count",
      M.driver_running() == [] and M.pidfile_live() is None)
bad = [n for n, ok in CHECKS if not ok]
print(f"mock: {len(CHECKS) - len(bad)}/{len(CHECKS)} checks ok; files that would have been written: "
      f"{sorted(Path(k).name for k in WRITTEN)}")
sys.exit(0 if not bad else 1)
