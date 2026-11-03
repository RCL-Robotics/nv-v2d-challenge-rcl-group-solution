#!/usr/bin/env python3
"""r9_mock_select.py: mock harness for r9_select.py (build step B2; PREREG_R9 10). Nothing is written: every write of
r9_select.py is captured in memory. Real inputs: round 8's records through R8_RECORDS_R9.json and round 8's / round 7's
verified loaders. Synthetic inputs: round-9 phase-S / phase-F jobs and runs.

  conda_envs/v2dkit/bin/python -I -B dryrun/r9_mock_select.py <r9_select.py>
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

SEL = Path(sys.argv[1]).resolve()
src = SEL.read_bytes()
S = types.ModuleType("r9_select_mock")
S.__file__ = str(SEL)
sys.modules[S.__name__] = S
exec(compile(src, str(SEL), "exec"), S.__dict__)  # noqa: S102

WRITES: dict[str, str] = {}
S.write_new = lambda p, text: WRITES.__setitem__(str(p), text)
_SHA = S.sha
S.sha = lambda p: _SHA(p) if Path(p).is_file() else "(not written: mock)"
S.frozen_hashes = lambda names: "".join(f"(mock)  {n}\n" for n in names)  # the frozen inputs do not exist before the freeze
RES = []


def check(name, ok, detail=""):
    RES.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  -- {str(detail)[:300]}"), flush=True)


incs = S.incumbents9()
check("30 incumbents; ep13 = the round-8 replacement with round-8 phase-F runs; the rest round-7 F",
      incs[13]["job"] == "ep13_w312runinws_90000" and incs[13]["phase_f_source"] == "r8F"
      and all(incs[e]["phase_f_source"] == "r7F" for e in range(30) if e != 13), [incs[e]["phase_f_source"] for e in range(30)])
check("ep13 bar = m_P_frozen + 0.01 (m_P_fresh == m_P_frozen: the same 10 runs)",
      incs[13]["m_P_fresh"] == incs[13]["m_P_frozen"] and abs(incs[13]["bar"] - (incs[13]["m_P_frozen"] + 0.01)) < 1e-15)

# keep-r8 / no-change path: R9-CLOSEST must be round 8's trial 7 (PREREG_R9 5.6 consistency check)
runinfo = S.base_runinfo(incs)
comp = {e: f"{incs[e]['phase_f_source']}:{incs[e]['job']}" for e in range(30)}
rows, t = S.closest9(comp, runinfo)
check("no change: the S8 composite from the recorded runs gives trial 7 (the R8 file's trial)", t == 7, (t, rows))
WRITES.clear()
eps = [{"episode": e, "incumbent": incs[e]["job"], "incumbent_input": incs[e]["input"], "m_P_frozen": incs[e]["m_P_frozen"],
        "m_P_fresh": incs[e]["m_P_fresh"], "bar": incs[e]["bar"], "evaluated": False, "finalist": None, "changed": False,
        "pick": incs[e]["job"], "pick_status": "round-8 pick kept", "why_not": ["mock"]} for e in range(30)]
rc = S.finish_f(eps, comp, runinfo, {}, "mock keep-r8")
names = sorted(Path(k).name for k in WRITES)
check("keep-r8 path: finish_f writes SELECTION_R9.json / .md / episodes / trials / SELECTION_R9.sha256 (last), trial 7",
      rc == 0 and names == sorted(["SELECTION_R9.json", "selection_r9_episodes.csv", "summary_r9_trials.csv", "selection_r9.md",
                                   "SELECTION_R9.sha256"]) and '"r9_closest_trial": 7' in WRITES[str(S.ROOT / "SELECTION_R9.json")],
      names)
sel = WRITES.get(str(S.ROOT / "SELECTION_R9.json"), "")
check("keep-r8 file: every episode's rollout is the recorded run 7 of its round-8 pick (round-7 rollouts; ep13 round-8)",
      '"source": "round-8 phase F (round-8 pick kept)"' in sel and sel.count('"source": "round-7 phase F (round-8 pick kept)"') == 29)
check("records state the basis: selection uses only own-input and consensus-B scores",
      json.loads(WRITES[str(S.ROOT / "SELECTION_R9.json")]).get("basis") == "own-input and consensus-B scores only"
      and "Selection uses only own-input and consensus-B scores" in WRITES[str(S.ROOT / "selection_r9.md")])

# a replacement in one episode: R9-CLOSEST over a round-7 / round-8 / round-9 composite
e0 = 10
fin = {k: {"ok": True, "flagged": False, "try": 1, "parquet": f"/mock/r9/{k}", "own_auc": 0.9, "own_sp": 1.0, "own_mp": 1.0,
           "own_mppe": 1.0, "cons_auc": incs[e0]["bar"] + 0.05, "cons_sp": 1.0, "cons_mp": 1.0, "cons_mppe": 1.0} for k in range(1, 11)}
d = S.decide_episode_f9(incs[e0]["m_P_frozen"], incs[e0]["phase_f_runs"], "ep10_mock_1", fin)
check("decide F against the recorded round-7 runs: m_F = bar + 0.05 replaces", d["changed"], d)
d2 = S.decide_episode_f9(incs[13]["m_P_frozen"], incs[13]["phase_f_runs"], "ep13_mock", {k: {**v, "cons_auc": incs[13]["bar"] - 1e-9} for k, v in fin.items()})
check("decide F against the round-8 replacement's recorded runs: m_F just below its bar does not replace", not d2["changed"], d2)
comp2 = dict(comp)
comp2[e0] = "r9F:ep10_mock_1"
runinfo2 = {**runinfo, "r9F:ep10_mock_1": fin}
rows2, t2 = S.closest9(comp2, runinfo2)
check("a replacement changes the composite: trials stay complete and a closest trial exists", t2 is not None and all(r["complete"] for r in rows2), (t2,))
WRITES.clear()
eps2 = [dict(x) for x in eps]
eps2[e0].update(changed=True, pick="ep10_mock_1", pick_status="replaced", pick_input="g2", pick_owner="zuxing",
                pick_machine="5090", pick_input_reference="/mock/ref", pick_export_sha256="0" * 64, pick_export_file="/mock/x")
orig = S.rollout_lists
S.rollout_lists = lambda: {**orig(), "r9F": {f"/mock/r9/{k}": "1" * 64 for k in range(1, 11)}}
rc = S.finish_f(eps2, comp2, runinfo2, {}, None)
S.rollout_lists = orig
sel2 = json.loads(WRITES[str(S.ROOT / "SELECTION_R9.json")])
check("replacement path: SELECTION_R9 written with the replaced episode and its round-9 rollout",
      rc == 0 and sel2["replaced_episodes"] == ["10"] and sel2["r9_closest_file"]["10"]["source"] == "round-9 phase F (replacement)"
      and sel2["compositions"]["S9_run_source"]["10"] == "r9F", sel2.get("replaced_episodes"))

# recorded phase-S records of reuse8 / reuse7 candidates load through round 8's / round 7's verified loaders
rec = S.load_r8records()["phase_s_candidates"]
c8 = sorted(c for c, v in rec.items() if v["round"] == "r8")[0]
c7 = sorted(c for c, v in rec.items() if v["round"] == "r7")[0]
pool = {c8: {"role": "reuse8", "episode": rec[c8]["episode"]}, c7: {"role": "reuse7", "episode": rec[c7]["episode"]}}
info, srcs = S.earlier_s_info(pool, {c8, c7})
r8row, r7row = S.R7C.s_row(c8, info[c8]), S.R7C.s_row(c7, info[c7])
check("reuse8 record: round 8's phase-S record reproduces the stored eligibility / mean",
      r8row["eligible"] == rec[c8]["s_eligible"] and r8row.get("cons_mean") == rec[c8]["cons_mean"], (r8row, rec[c8]))
check("reuse7 record: round 7's phase-S record reproduces the stored eligibility / mean",
      r7row["eligible"] == rec[c7]["s_eligible"] and r7row.get("cons_mean") == rec[c7]["cons_mean"], (r7row, rec[c7]))
check("keep-r8 refused while an r9_eval.py run is alive (driver_running reads /proc; none now)", S.driver_running() == [])
n_ok = sum(ok for _, ok in RES)
print(f"mock: {n_ok}/{len(RES)} checks PASS -> {'PASS' if n_ok == len(RES) else 'FAIL'}")
sys.exit(0 if n_ok == len(RES) else 1)
