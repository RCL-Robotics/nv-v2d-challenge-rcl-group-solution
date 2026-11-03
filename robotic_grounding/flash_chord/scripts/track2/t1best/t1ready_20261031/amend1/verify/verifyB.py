"""Independent verification B (provenance, commit, validator) of the amend1 Tier-1 file. Read-only except OUT dir."""
import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
KIT = ME / "v2d_submission_kit_src/v2d_submission_kit"
sys.path.insert(0, str(KIT))
from v2dlb import report_schema as rs  # noqa: E402

U = ME / "fc_runs/T1improve/topup"
A = U / "amend1"
OUT = A / "verifyB"
CSV = A / "files/trial05_SUBMITTABLE_W3H_closest_to_mean/T1improve_W3H_tier1_trial05_SUBMITTABLE_closest_to_mean.csv"
CLAIM_SHA = "53cfe9bfa5a2487f3502035e535ea5fabcbff0cab0a578fbea514108aa0f3f77"
CLAIM_TRIAL = 5
URL = "https://github.com/Beat-in-our-hearts/nv-v2d-challenge-codebase/commit/0c7516eda6f90deacb1536aabbce98beba5194ca"
EVAL_SCRIPT = "160d24f5fbeb70ccd569ca4192056f1892b2a9f2999066db9d8279461648fdf1"
EVAL_CODE = "b1b3ebe3ae4629d1687a83f420149e3f6fdfc154d70acad55c12fc8fa69811a1"
EP28 = "497ae3b3a785fc9283fcdda2c019020608ed8c41472f4229ac780f4450081375"
CHANGED = {1: "ep01", 5: "ep05", 10: "ep10", 14: "ep14", 15: "ep15", 19: "ep19"}
B20 = ME / "fc_runs/submissions/t1_best20_20261026"
TRACK = "track_2_tier1"

report = {"checks": {}, "problems": [], "info": {}}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def prob(msg):
    report["problems"].append(msg)


# ---------- 0. file identity
csv_sha = sha(CSV)
report["info"]["csv_sha256"] = csv_sha
if csv_sha != CLAIM_SHA:
    prob(f"csv sha {csv_sha} != claimed {CLAIM_SHA}")

# ---------- 1. provenance columns
df = pd.read_csv(CSV, dtype=str, keep_default_na=False)
report["info"]["columns"] = list(df.columns)
report["info"]["rows"] = len(df)
c1 = {}
for col in ["eval_script_sha256", "eval_code_sha256", "checkpoint_sha256", "packer_script_sha256",
            "uploader_script_sha256", "code_commit_url"]:
    if col not in df.columns:
        prob(f"missing column {col}")
        continue
    raw = df[col]
    stripped = raw.str.strip()
    if (raw != stripped).any():
        prob(f"{col} has whitespace-padded values")
    if (stripped == "").any():
        prob(f"{col} has empty values")
    c1[col] = sorted(stripped.unique().tolist())
c1_ok = True
if c1.get("code_commit_url") != [URL]:
    c1_ok = False
    prob(f"code_commit_url values {c1.get('code_commit_url')}")
if c1.get("eval_script_sha256") != [EVAL_SCRIPT]:
    c1_ok = False
    prob(f"eval_script_sha256 values {c1.get('eval_script_sha256')}")
if c1.get("eval_code_sha256") != [EVAL_CODE]:
    c1_ok = False
    prob(f"eval_code_sha256 values {c1.get('eval_code_sha256')}")
for col in ["packer_script_sha256", "uploader_script_sha256"]:
    if len(c1.get(col, [])) != 1:
        c1_ok = False
        prob(f"{col} not uniform {c1.get(col)}")
# packer / uploader digests vs the kit files on disk (informational + consistency)
report["info"]["kit_pack_submission_sha256"] = sha(KIT / "tools/pack_submission.py")
report["info"]["kit_v2d_submit_sha256"] = sha(KIT / "submit/v2d_submit.py")
report["info"]["kit_evaluate_policy_sha256"] = sha(KIT / "flash_chord/scripts/evaluate_policy.py")
report["info"]["stock_evaluate_policy_sha256"] = sha(
    ME / "code/video_to_data_8c778f9/robotic_grounding/flash_chord/scripts/evaluate_policy.py")
ep_str = df["row_id"].str.split("/").str[1]
if not ep_str.str.match(r"^e\d{6}$").all():
    prob("row_id episode token malformed")
df["_ep"] = ep_str.str[1:].astype(int)
per_ep = df.groupby("_ep")["checkpoint_sha256"].agg(lambda s: sorted(set(s.str.strip())))
csv_ck = {}
for e in range(30):
    v = per_ep.get(e)
    if v is None or len(v) != 1:
        c1_ok = False
        prob(f"episode {e}: checkpoint shas {v}")
    else:
        csv_ck[e] = v[0]
if sorted(per_ep.index.tolist()) != list(range(30)):
    c1_ok = False
    prob(f"episodes present {sorted(per_ep.index.tolist())}")
report["info"]["csv_distinct_checkpoints"] = int(df["checkpoint_sha256"].nunique())
report["checks"]["1_provenance"] = {"ok": c1_ok, "uniform_values": c1, "n_episodes": int(per_ep.shape[0])}

# ---------- 2. expected checkpoint per episode
pins = {}
for line in (ME / "fc_runs/T1improve/confirm/EXPORT_PINS.md").read_text().splitlines():
    m = re.match(r"^\| (\S+) \| \S+ \| `([0-9a-f]{64})` \| `([^`]+)` \| `([0-9a-f]{64})` \|", line)
    if m:
        pins[m.group(1)] = {"file": m.group(3), "sha": m.group(4)}
dec = json.loads((U / "decisions.json").read_text())
winners = {}
for e, it in CHANGED.items():
    d = dec["items"][it]
    winners[e] = d["winner"]
    if d["episode"] != e:
        prob(f"decisions {it} episode {d['episode']} != {e}")
    if not d.get("keep_new"):
        prob(f"decisions {it} keep_new false")
    if d["eval_file_sha256"] != pins[d["winner"]]["sha"]:
        prob(f"decisions {it} sha != EXPORT_PINS for {d['winner']}")
ep28d = dec["items"]["ep28alt"]
report["info"]["decisions_ep28alt"] = {"winner": ep28d["winner"], "keep_new": ep28d["keep_new"], "why": ep28d["why"]}
if ep28d["winner"] != "ep28-W3H_83000" or pins["ep28-W3H_83000"]["sha"] != EP28:
    prob("ep28 W3H pin mismatch")
report["info"]["winners"] = winners
# no other changed items in decisions?
report["info"]["decision_items"] = sorted(dec["items"].keys())

sc_path = U / f"scores/all7/t{CLAIM_TRIAL:02d}/score.json"
sc = json.loads(sc_path.read_text())
if sc.get("trial") != CLAIM_TRIAL or sc.get("config") != "all7" or not sc.get("complete"):
    prob(f"score.json header trial={sc.get('trial')} config={sc.get('config')} complete={sc.get('complete')}")
expected = {}
c2_rows = {}
c2_ok = True
for e in range(30):
    src = sc["sources"][str(e)]
    if e == 28:
        exp = EP28
        basis = "ep28-W3H_83000 (fixed)"
    elif e in CHANGED:
        exp = pins[winners[e]]["sha"]
        basis = f"EXPORT_PINS {winners[e]}"
    else:
        exp = src["eval_file_sha256"]
        basis = f"score.json sources kind={src['kind']} name={src['name']}"
        if src["kind"] != "B20":
            c2_ok = False
            prob(f"episode {e}: unchanged episode source kind {src['kind']}")
        # independent cross-check with the t1best20 export record + file bytes
        exj = json.loads((B20 / f"exports/ep{e:02d}/export.json").read_text())
        fsha = sha(exj["file"])
        if not (exj["sha256"] == exj.get("destination_sha256", exj["sha256"]) == fsha == exp):
            c2_ok = False
            prob(f"episode {e}: t1best20 export {exj['sha256']} / file {fsha} != score.json {exp}")
    if e in CHANGED or e == 28:
        if src["eval_file_sha256"] != exp:
            c2_ok = False
            prob(f"episode {e}: score.json source sha {src['eval_file_sha256']} != expected {exp}")
        name = winners.get(e, "ep28-W3H_83000")
        if not src["name"].startswith(name.split("_")[0]):
            prob(f"episode {e}: score.json source name {src['name']} vs {name}")
        # file bytes of the pinned evaluated file
        pf = pins[name]["file"].replace("T1I/", str(ME / "fc_runs/T1improve") + "/")
        fsha = sha(pf)
        if fsha != exp:
            c2_ok = False
            prob(f"episode {e}: pinned file {pf} hashes {fsha} != {exp}")
    expected[e] = exp
    got = csv_ck.get(e)
    ok = got == exp
    if not ok:
        c2_ok = False
        prob(f"episode {e}: csv checkpoint {got} != expected {exp} ({basis})")
    c2_rows[e] = {"csv": got, "expected": exp, "basis": basis, "score_src_name": src["name"], "ok": ok}

# results packed = the selected trial's
res_dir = U / f"scores/all7/t{CLAIM_TRIAL:02d}/results"
res_ok = True
res_detail = {}
files = sorted(p.name for p in res_dir.iterdir())
if len(files) != 60:
    res_ok = False
    prob(f"results dir has {len(files)} files")
for e in range(30):
    pq = res_dir / f"episode_{e:06d}.parquet"
    js = json.loads((res_dir / f"episode_{e:06d}.json").read_text())
    pq_sha = sha(pq)
    inp = sc["inputs"][str(e)]
    orig = sc["sources"][str(e)]["results_parquet"]
    orig_sha = sha(orig)
    jok = js.get("checkpoint_sha256") == expected[e]
    ok = pq_sha == inp == orig_sha and jok
    if not ok:
        res_ok = False
        prob(f"episode {e}: parquet {pq_sha} inputs {inp} orig {orig_sha} json_ck_ok {jok}")
    res_detail[e] = {"parquet_sha": pq_sha[:16], "inputs": inp[:16], "orig": orig, "json_ck_ok": jok}
# builder's pack command pointed at this results dir
plog = (CSV.parent / "pack.log").read_text().splitlines()[0]
m = re.search(r"--results (\S+)", plog)
if not m or m.group(1) != str(res_dir):
    res_ok = False
    prob(f"builder pack.log --results {m.group(1) if m else None}")
m2 = re.search(r"--code-commit (\S+)", plog)
if not m2 or m2.group(1) != URL:
    prob("builder pack.log code-commit mismatch")
# content equality with the trial's own LOCAL pack (placeholder commit) on every column but code_commit_url
loc = U / f"scores/all7/t{CLAIM_TRIAL:02d}/all7_trial{CLAIM_TRIAL:02d}_LOCAL_NOT_FOR_UPLOAD.csv"
loc_sha = sha(loc)
if loc_sha != sc["pack"]["csv_sha256"]:
    prob(f"LOCAL csv sha {loc_sha} != score.json pack.csv_sha256")
ldf = pd.read_csv(loc, dtype=str, keep_default_na=False)
cols = [c for c in ldf.columns if c != "code_commit_url"]
same_local = list(ldf.columns) == [c for c in df.columns if c != "_ep"] and ldf[cols].equals(df[cols])
if not same_local:
    res_ok = False
    prob("csv differs from the trial's LOCAL pack outside code_commit_url")
# the pose content matches ONLY this trial among the 20 all7 LOCAL packs
pose_cols = ["pos_x", "pos_y", "pos_z", "quat_x", "quat_y", "quat_z", "quat_w"]
match_trials = []
for t in range(1, 21):
    p = U / f"scores/all7/t{t:02d}/all7_trial{t:02d}_LOCAL_NOT_FOR_UPLOAD.csv"
    if not p.exists():
        continue
    o = pd.read_csv(p, dtype=str, keep_default_na=False, usecols=["row_id", *pose_cols, "checkpoint_sha256"])
    if o.equals(df[["row_id", *pose_cols, "checkpoint_sha256"]]):
        match_trials.append(t)
if match_trials != [CLAIM_TRIAL]:
    res_ok = False
    prob(f"pose content matches all7 trials {match_trials}")
report["checks"]["2_checkpoints"] = {"ok": c2_ok, "per_episode": c2_rows}
report["checks"]["2b_results_are_selected_trial"] = {
    "ok": res_ok, "same_as_local_pack_except_commit": same_local, "pose_matches_all7_trials": match_trials,
    "local_csv_sha256": loc_sha, "per_episode": res_detail}

# ---------- 3. kit uploader validate_submission()
config = json.loads((KIT / "config/tracks.json").read_text())
spec = importlib.util.spec_from_file_location("kit_v2d_submit", KIT / "submit/v2d_submit.py")
up = importlib.util.module_from_spec(spec)
spec.loader.exec_module(up)
c3 = {"ok": False}
try:
    frame = up.validate_submission(CSV, TRACK, config)
    eps = frame[rs.ROW_ID_COLUMN].astype(str).str.split("/").str[1].nunique()
    c3 = {"ok": True, "rows": int(len(frame)), "episodes": int(eps), "problems": []}
except SystemExit as err:
    c3 = {"ok": False, "problems": [str(err)]}
    prob(f"validate_submission: {err}")
report["checks"]["3_validate_submission"] = c3

# ---------- extra: selection rule over complete all7 trials (informational)
rows = []
for t in range(1, 21):
    s = json.loads((U / f"scores/all7/t{t:02d}/score.json").read_text())
    if s.get("complete"):
        o = s["set"]["overall_float32"]
        rows.append((t, o["AUC"], o["SP-SR"], o["MP-SR"], o["MPPE"]))


def mode(vals):
    c = Counter(round(v, 9) for v in vals)
    top = max(c.values())
    return max(k for k, n in c.items() if n == top)


msp = mode([r[2] for r in rows])
mmp = mode([r[3] for r in rows])
aucs = [r[1] for r in rows]
mppes = [r[4] for r in rows]
ma, sa = statistics.mean(aucs), statistics.stdev(aucs)
mm, sm = statistics.mean(mppes), statistics.stdev(mppes)
cands = [r for r in rows if round(r[2], 9) == msp and round(r[3], 9) == mmp]


def score(r):
    a = abs(r[1] - ma) / sa if sa > 0 else 0.0
    b = abs(r[4] - mm) / sm if sm > 0 else 0.0
    return a + b


best = min(cands, key=lambda r: (score(r), r[0]))
report["info"]["selection_recompute"] = {
    "n_complete": len(rows), "mode_SP": msp, "mode_MP": mmp, "mean_AUC": ma, "sd_AUC": sa, "mean_MPPE": mm,
    "sd_MPPE": sm, "candidates": [r[0] for r in cands], "selected": best[0],
    "scores": {r[0]: round(score(r), 6) for r in cands},
    "mp_by_trial": {r[0]: r[3] for r in rows}}
if best[0] != CLAIM_TRIAL:
    prob(f"selection recompute picks trial {best[0]} != {CLAIM_TRIAL}")
# set values of the selected trial vs builder's re-score
mc = json.loads((CSV.parent / "metric_code.json").read_text())
report["info"]["selected_trial_set"] = sc["set"]["overall_float32"]

report["ok"] = not report["problems"]
(OUT / "verifyB_report.json").write_text(json.dumps(report, indent=1, default=str) + "\n")
print(json.dumps({"ok": report["ok"], "problems": report["problems"],
                  "c1": report["checks"]["1_provenance"]["ok"], "c2": c2_ok, "c2b": res_ok, "c3": c3,
                  "selection": {k: v for k, v in report["info"]["selection_recompute"].items() if k != "mp_by_trial"},
                  "csv_sha": csv_sha, "match_trials": match_trials,
                  "kit_shas": {k: report["info"][k] for k in report["info"] if k.startswith(("kit_", "stock_"))},
                  "uniform": c1}, indent=1, default=str))
