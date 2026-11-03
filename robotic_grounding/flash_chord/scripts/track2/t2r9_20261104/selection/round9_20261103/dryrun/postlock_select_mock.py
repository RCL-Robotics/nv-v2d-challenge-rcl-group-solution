#!/usr/bin/env python3
"""postlock_select_mock.py: post-lock check of r9_select.py's phase-S selection and phase-F decision on the dry-run pool
(PREREG_R9 5.3-5.6). The 34 screen rows get synthetic round-9 phase-S runs; reuse8 / reuse7 / barred rows use their real
recorded phase-S records through round 8's / round 7's loaders. Every write is captured in memory; nothing is written.

  conda_envs/v2dkit/bin/python -I -B dryrun/postlock_select_mock.py <r9_select.py> <pool_dryrun_*.json>
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

SEL, POOLJ = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
S = types.ModuleType("r9_select_postlock_mock")
S.__file__ = str(SEL)
sys.modules[S.__name__] = S
exec(compile(SEL.read_bytes(), str(SEL), "exec"), S.__dict__)  # noqa: S102
WRITES: dict[str, str] = {}
S.write_new = lambda p, text: WRITES.__setitem__(str(p), text)
_SHA = S.sha
S.sha = lambda p: _SHA(p) if Path(p).is_file() else "(not written: mock)"
S.frozen_hashes = lambda names: "".join(f"(mock)  {n}\n" for n in names)
S.require = lambda ph, stage: True
RES = []


def check(name, ok, detail=""):
    RES.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  -- {str(detail)[:400]}"), flush=True)


pool_rows = json.loads(POOLJ.read_text())["pool"]
pool = {p["job"]: p for p in pool_rows}
S.pool_rows = lambda: pool
incs = S.incumbents9()
screen = sorted(j for j, p in pool.items() if p["role"] == "screen")
# synthetic phase S: ep10's first screen job clearly above its bar_S; ep20's report file just below; the rest far below
hi = next(j for j in screen if pool[j]["episode"] == 10)
lo = next(j for j in screen if pool[j]["episode"] == 20 and "zxrep" in j)
cons, jobs, evals, jst = {}, {}, {}, {}
for j in screen:
    e = pool[j]["episode"]
    v = incs[e]["m_P_frozen"] + (0.06 if j == hi else (0.0099 if j == lo else -0.2))
    cons[j] = [v, v, v]
    jobs[j] = {"job": j, "episode": e, "input": pool[j]["input"], "owner": pool[j]["owner"], "machine": pool[j]["machine"],
               "tree": pool[j]["tree"], "iteration": pool[j]["iteration"], "export_status": "ok", "training_sha256": "t" * 64,
               "export_sha256": "x" * 64, "export_file": f"/mock/{j}"}  # fmt: skip
    evals[j] = {k: {"auc": 0.5, "sp_sr": 1.0, "mp_sr": 1.0, "mppe": 1.0, "try": 1, "parquet": f"/mock/{j}/{k}"} for k in (1, 2, 3)}
    jst[j] = {"n": 3, "runs": [1, 2, 3], "auc": cons[j], "mppe": [1.0, 1.0, 1.0]}
S.scored_inputs9 = lambda ph: ({"jobs": jst}, {"validation_failed_jobs": []}, jobs, evals, {}, set())
rc = S.cmd_select_s()
fin = json.loads(WRITES.get(str(S.ROOT / "final_r9.json"), "{}"))
sel = json.loads(WRITES.get(str(S.ROOT / "selection_s9.json"), "{}"))
check("select S runs on the dry-run pool (34 synthetic screen jobs + real reuse8 / reuse7 / barred records)", rc == 0 and sel, rc)
fj = [x["job"] for x in fin.get("finalists", [])]
check(f"the synthetic candidate above bar_S ({hi}) is ep10's finalist; 0.0099 above m_P_frozen ({lo}) is not", hi in fj and lo not in fj, fj)
ev = {int(x["episode"]): x for x in sel.get("episodes", [])}
check("evaluated episodes = the 28 with a screen / reuse8 / reuse7 candidate (not 16, 19)",
      sorted(e for e, x in ev.items() if x["evaluated"]) == [e for e in range(30) if e not in (16, 19)],
      sorted(e for e, x in ev.items() if x["evaluated"]))
barred_rows = [r for x in sel.get("episodes", []) for r in x["candidates"] if r.get("barred")]
check("barred rows with recorded phase-S results are listed and never finalists",
      all(not r["qualifies"] for r in barred_rows) and not set(fj) & {r["job"] for r in barred_rows}, [r["job"] for r in barred_rows])
reused = [r for x in sel.get("episodes", []) for r in x["candidates"] if r["pool_role"] in ("reuse8", "reuse7")]
check("all 23 reuse8 / reuse7 rows are judged on their recorded phase-S records",
      len(reused) == 23 and all(r["source"] in ("round-8 phase S (reused)", "round-7 phase S (reused)") for r in reused), len(reused))
other = [x for x in fin.get("finalists", []) if x["job"] != hi]
check("any other finalist is a recorded reuse8 / reuse7 candidate at or above its episode's bar_S",
      all(x["pool_role"] in ("reuse8", "reuse7") and x["cons_S_mean"] >= x["bar_s"] for x in other),
      [(x["job"], x["pool_role"]) for x in other])
print("finalists (mock):", [(x["episode"], x["job"], x["pool_role"]) for x in fin.get("finalists", [])])
# phase F on the mock finalists: the synthetic one replaces, recorded ones do not (mock runs below bar)
WRITES_S = dict(WRITES)
WRITES.clear()
S.C.read_json = (lambda orig: (lambda p: json.loads(WRITES_S[str(p)]) if str(p) in WRITES_S else orig(p)))(S.C.read_json)
fjobs = {}
fev, fsc = {}, {}
for x in fin.get("finalists", []):
    j = x["job"]
    fjobs[j] = {**jobs.get(j, {}), "job": j, "episode": x["episode"], "input": x["input"], "owner": x["owner"],
                "machine": x["machine"], "input_reference": "/mock/ref", "export_sha256": x["export_sha256"],
                "export_file": x["export_file"], "training_checkpoint": "/mock/ck", "source_checkpoint": "/mock/src",
                "iteration": x["iteration"]}  # fmt: skip
    bar = incs[x["episode"]]["bar"]
    v = bar + 0.02 if j == hi else bar - 0.05
    fev[j] = {k: {"auc": 0.5, "sp_sr": 1.0, "mp_sr": 1.0, "mppe": 1.0, "try": 1, "parquet": f"/mock/f/{j}/{k}"} for k in range(1, 11)}
    for k in range(1, 11):
        fsc[(j, k)] = {"auc": v, "sp_sr": 1.0, "mp_sr": 1.0, "mppe": 1.0}
S.scored_inputs9 = lambda ph: ({}, {}, fjobs, fev, fsc, set())
S.SEL_S9 = Path(sys.argv[1])  # an existing file stands in for the phase-S lock (mock only)
orig_rl = S.rollout_lists
S.rollout_lists = lambda: {**orig_rl(), "r9F": {f"/mock/f/{j}/{k}": "1" * 64 for j in fjobs for k in range(1, 11)}}
rc = S.cmd_select_f()
r9 = json.loads(WRITES.get(str(S.ROOT / "SELECTION_R9.json"), "{}"))
check("select F: the synthetic finalist replaces ep10; recorded finalists below their bars do not; R9-CLOSEST exists",
      rc == 0 and r9.get("replaced_episodes") == ["10"] and r9.get("r9_closest_trial") is not None, (rc, r9.get("replaced_episodes")))
check("the S9 composite: ep10 from round-9 runs, ep13 from round-8 runs, the other 28 from round-7 runs",
      r9["compositions"]["S9_run_source"]["10"] == "r9F" and r9["compositions"]["S9_run_source"]["13"] == "r8F"
      and sum(v == "r7F" for v in r9["compositions"]["S9_run_source"].values()) == 28)
print(f"postlock select mock: {sum(RES)}/{len(RES)} PASS -> {'PASS' if all(RES) else 'FAIL'}")
sys.exit(0 if all(RES) else 1)
