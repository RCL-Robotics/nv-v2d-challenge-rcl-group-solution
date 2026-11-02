"""Mock harness (synthetic values only): exercises cmd_select_s / cmd_select_f of t2cons_r7_consensus.py with every
file read / write patched to in-memory synthetic records. Reads no real score, writes nothing on disk."""
import json
import sys
import types
from pathlib import Path

import numpy as np

SRC = Path(sys.argv[1])
mod = types.ModuleType("r7c_mock")
mod.__file__ = str(SRC)
sys.modules["r7c_mock"] = mod
exec(compile(SRC.read_bytes(), str(SRC), "exec"), mod.__dict__)
M = mod
C = M.C
rng = np.random.default_rng(7)
WRITTEN = {}


def write_new(p, text, mode=0o444):
    p = Path(p)
    if str(p) in WRITTEN:
        raise SystemExit(f"refusing to overwrite {p}")
    WRITTEN[str(p)] = text


M.write_new = write_new
M.raw_sha = lambda p: "f" * 64
M.require = lambda ph, stage: True

# ---------------------------------------------------------------- phase S synthetic data
incs = {}
for e, (job, m6, n, kind, rnd) in M.INC_TABLE.items():
    incs[e] = {"job": job, "episode": e, "input": kind, "m_P_frozen": m6}
refs = {e: {"job": job, "episode": e, "input": kind} for e, (job, kind, pre) in M.REF_TABLE.items()}
pool, jobs, evals, jst, flagged = {}, {}, {}, {}, set()
excluded = []
for e in range(30):
    n = int(rng.integers(0, 6)) if e not in (12, 13, 16) else 4
    if e == 16:  # a STRICT episode with no S-eligible candidate -> fallback
        n = 3
    for i in range(n):
        job = f"ep{e:02d}_c{i}"
        if e == 12 and i == 0:
            job = "ep12_raw_37575"
        if e == 13 and i == 0:
            job = "ep13_raw_44922"
        role = "reuse" if i % 2 else "screen"
        if job in M.BARRED:
            role = "reuse"
        pool[job] = {"job": job, "episode": e, "role": role}
        if job in M.BARRED:
            pool[job]["r6_phase_f_verdict"] = {"r6_incumbent": M.BARRED[job][1], "verdict": "did not replace the incumbent"}
        jobs[job] = {"job": job, "episode": e, "export_status": "ok" if rng.random() > 0.05 else "missing", "input": "raw",
                     "owner": "zuxing" if i % 3 else "haoran", "machine": "5090" if i % 3 == 2 else "pro6000", "pool_role": role,
                     "run_label": "lab", "iteration": 1000 * (i + 1), "tree": "b39b69b", "run_dir": f"/x/{e}", "pool_sha256": "a" * 64,
                     "training_sha256": "b" * 64, "export_sha256": f"{e:02d}{i}" + "c" * 61, "export_file": f"/x/{job}.safetensors",
                     "role": role}
        base = incs[e]["m_P_frozen"] if e in incs else 0.5
        nr = 3 if rng.random() > 0.05 else 2
        sp = 1.0 if (e != 16 and rng.random() > 0.2) else 0.0
        evals[job] = {k: {"auc": float(rng.uniform()), "sp_sr": sp, "mp_sr": sp, "mppe": 1.0, "try": 1, "parquet": f"/r/{job}/{k}"}
                      for k in range(1, nr + 1)}
        aucs = [float(np.clip(base + rng.normal(0.02, 0.03), 0, 1)) for _ in range(nr)]
        if job in M.BARRED:
            aucs = [0.99] * 3
            evals[job] = {k: {"auc": 0.9, "sp_sr": 1.0, "mp_sr": 1.0, "mppe": 1.0, "try": 1, "parquet": f"/r/{job}/{k}"} for k in (1, 2, 3)}
        if rng.random() < 0.05:
            flagged.add((job, 1))
        jst[job] = {"n": len(aucs), "runs": list(range(1, len(aucs) + 1)), "auc": aucs, "mppe": [1.0] * len(aucs),
                    "sp_sr_stats": {"mean": 1.0}, "mp_sr_stats": {"mean": 1.0}}
# STRICT episodes need >= 1 candidate with 3 ok unflagged runs (else halt): make sure
for e in M.STRICT:
    job = f"ep{e:02d}_safe"
    pool[job] = {"job": job, "episode": e, "role": "screen"}
    jobs[job] = dict(next(iter(jobs.values())), job=job, episode=e, export_status="ok", pool_role="screen", role="screen",
                     export_sha256=f"{e:02d}s" + "d" * 61)
    evals[job] = {k: {"auc": 0.5, "sp_sr": 1.0, "mp_sr": 1.0, "mppe": 1.0, "try": 1, "parquet": f"/r/{job}/{k}"} for k in (1, 2, 3)}
    jst[job] = {"n": 3, "runs": [1, 2, 3], "auc": [0.4, 0.41, 0.42], "mppe": [1.0] * 3, "sp_sr_stats": {"mean": 1.0}, "mp_sr_stats": {"mean": 1.0}}
js = {"jobs": jst, "reuse_reproduction": {"pass": True}}
g = {"validation_failed_jobs": excluded}
M.scored_inputs = lambda ph: (js, g, jobs, evals, {}, flagged)
M.load_incumbents = lambda: incs
M.pool_rows = lambda: pool
rc = M.cmd_select_s()
print("select S rc", rc)
final = json.loads(WRITTEN[str(M.FINAL)])
print("finalists:", [(x["episode"], x["job"], x["rank"], x["selection_mode"]) for x in final["finalists"]])
assert all(x["job"] not in M.BARRED for x in final["finalists"])
print(open("/dev/null").read() or WRITTEN[str(M.ROOT / "selection_s7.txt")][:3000])

# ---------------------------------------------------------------- phase F synthetic data
jobsF, evalsF, scoresF, flaggedF = {}, {}, {}, set()
def addjob(job, e, role, base, floor_p=0.9, nmiss=0):
    jobsF[job] = {"job": job, "episode": e, "role": role, "input": "raw", "owner": "zuxing", "machine": "5090", "run_dir": "/x",
                  "run_label": "lab", "iteration": 5, "pool_role": role, "input_reference": "/ref", "export_sha256": job + "e" * 10,
                  "export_file": f"/x/{job}"}
    evalsF[job] = {}
    for k in range(1, 11 - nmiss):
        sp = 1.0 if rng.random() < floor_p else 0.0
        evalsF[job][k] = {"auc": float(rng.uniform()), "sp_sr": sp, "mp_sr": sp, "mppe": 2.0, "try": 1, "parquet": f"/f/{job}/{k}"}
        scoresF[(job, k)] = {"auc": str(base + rng.normal(0, 0.01)), "sp_sr": "1.0", "mp_sr": "1.0", "mppe": "3.0"}
for e, x in incs.items():
    addjob(x["job"], e, "incumbent", x["m_P_frozen"])
for e, x in refs.items():
    addjob(x["job"], e, "r6ref", 0.6)
for x in final["finalists"]:
    addjob(x["job"], x["episode"], "finalist", (x["m_P_frozen"] or 0.5) + 0.05, floor_p=0.5 if x["episode"] == 19 else 0.95)
rollouts = {r["parquet"]: "9" * 64 for j in evalsF for r in evalsF[j].values()}
s6 = {f"{e:02d}": (incs[e]["job"] if e in incs else refs[e]["job"]) for e in range(30)}
sel6 = {"compositions": {"S6": s6}}
orig_read_json = C.read_json
def rj(p):
    if Path(p) == M.FINAL:
        return final
    if Path(p) == Path("SEL6"):
        return sel6
    return orig_read_json(p)
C.read_json = rj
M.r6file = lambda key: Path("SEL6")
C.hash_list = lambda p: rollouts
M.scored_inputs = lambda ph: ({}, {}, jobsF, evalsF, scoresF, flaggedF)
M.load_refs6 = lambda: refs
rc = M.cmd_select_f()
print("select F rc", rc)
sel = json.loads(WRITTEN[str(M.ROOT / "selection_r7.json")])
print("S7", sel["compositions"]["S7"])
print("closest", sel["r7_closest_trial"], "replaced", sel["replaced_episodes"], "unconf", sel["strict_unconfirmed_episodes"])
print(WRITTEN[str(M.ROOT / "summary_r7.md")][-2500:])
print("SEL_R7 written:", str(M.SEL_R7) in WRITTEN, "files:", sorted(Path(k).name for k in WRITTEN))
