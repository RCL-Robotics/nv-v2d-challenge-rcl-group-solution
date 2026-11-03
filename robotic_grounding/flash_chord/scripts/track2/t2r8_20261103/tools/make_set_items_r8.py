#!/usr/bin/env python3
"""make_set_items_r8.py: write set/set_items_r8.json and set/set_manifest_r8.tsv of the R8-CLOSEST file (round 8,
trial t* of S8) from frozen records only. Nothing is evaluated, scored or selected here.

Sources (all read only):
  --selection   round 8's SELECTION_R8.json (S8, run sources, R8-CLOSEST trial and its per-episode rollouts / exports)
  --r7-items    round 7's set/set_items_r7.json as committed in the record (t2r7_20261102/set/set_items_r7.json)
  --pack-file   the pack step's pack_r8_placeholder/r8_closest/file.json (what was copied into the packed folder)
  --r8          the round-8 folder (POOL_R8.json, RELOCATION_R8.tsv, exports/, inputs_r8/, zuxing_copies_r8/)
Kept round-7 picks: the round-7 item, with its relative record paths pointing to ../t2r7_20261102/ and its
"evaluated" block replaced by the round-7 phase-F run of trial t*. A replacement: a new item from round 8's pool,
relocation, export, training-commit and run-record copies. Every item's export / rollout sha256 must equal both
SELECTION_R8.json and the pack file. New files only.
Run: <python> -I -B make_set_items_r8.py --selection S --r7-items I --pack-file F --r8 R8 --out-dir DIR
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import copy  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

PH = {"<ME>": "/root/autodl-tmp/haoran.zhang", "<ZX>": "/root/autodl-tmp/zuxing.lu",
      "<ZX5090>": "/mnt/gfs/home/zuxinglu/workspace", "<ZXWS>": "/home/zuxinglu/workspace"}
R7_PREFIXES = ("records/", "selection/", "set/", "train/", "data/", "tools/")
RULE_RELOC = ("only task.parquet and reference.parquet rewritten (as-trained host path -> this server's path of the same "
              "parquet); every tensor byte unchanged")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ph(s: str) -> str:
    for k in ("<ZX5090>", "<ZXWS>", "<ME>", "<ZX>"):
        s = s.replace(PH[k], k)
    return s


def to_r8_relative(obj, changed: list, where: str = ""):
    """Relative record paths of a round-7 item are relative to t2r7_20261102/; point them there from t2r8_20261103/."""
    if isinstance(obj, dict):
        return {k: to_r8_relative(v, changed, f"{where}.{k}") for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_r8_relative(v, changed, f"{where}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, str) and obj.startswith(R7_PREFIXES):
        changed.append(where)
        return "../t2r7_20261102/" + obj
    return obj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("selection", "r7-items", "pack-file", "r8", "out-dir"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    a = ap.parse_args()
    sel = json.loads(a.selection.read_text())
    r7 = json.loads(a.r7_items.read_text())
    pf = json.loads(a.pack_file.read_text())
    R8 = a.r8
    t = int(sel["r8_closest_trial"])
    s8, src, s7 = sel["compositions"]["S8"], sel["compositions"]["S8_run_source"], sel["compositions"]["S7_round7"]
    r7_by = {it["episode"]: it for it in r7["episodes"]}
    pf_by = {r["episode"]: r for r in pf["episodes"]}
    dec = {x["episode"]: x for x in sel["episodes"]}
    pool = {r["job"]: r for r in json.loads((R8 / "POOL_R8.json").read_text())["rows"]}
    with open(R8 / "RELOCATION_R8.tsv", newline="") as fp:
        reloc = {r["job"]: r for r in csv.DictReader(fp, delimiter="\t")}
    with open(R8 / "inputs_r8/ZUXING_CODE_R8.tsv", newline="") as fp:
        zcode = {r["run"]: r for r in csv.DictReader(fp, delimiter="\t")}
    problems, items, changed = [], [], []
    for e in range(30):
        k = f"{e:02d}"
        x = sel["r8_closest_file"][k]
        p = pf_by[e]
        job = s8[k]
        if x["job"] != job or p["job"] != job or p["export_sha256"] != x["export_sha256"] or \
                p["rollout_sha256"] != x["rollout_sha256"] or p["rollout_src"] != x["rollout"] or int(p["run"]) != t:
            problems.append(f"ep{k}: SELECTION_R8 / pack file disagree")
        evaluated = {"round": "round 7 phase F" if src[k] == "r7F" else "round 8 phase F", "trial": t, "run": t,
                     "try": int(x["try"]), "rollout": ph(x["rollout"]), "rollout_sha256": x["rollout_sha256"],
                     "frame_reference": ph(x["frame_reference"]),
                     "frame_reference_file_sha256": p["frame_ref_src_sha256"]}
        if src[k] == "r7F":
            it7 = r7_by[e]
            if it7["job"] != job or it7["export"]["sha256"] != x["export_sha256"] or s7[k] != job:
                problems.append(f"ep{k}: round-7 item {it7['job']} / export != the kept round-7 pick {job}")
            it = to_r8_relative(copy.deepcopy(it7), changed, f"ep{k}")
            it["evaluated"] = evaluated
            it["decision_round8"] = "round-7 pick kept (no round-8 finalist replaced it)"
            it["round7_record"] = (f"../t2r7_20261102/set/set_items_r7.json (episode {e}, job {job}); its 'evaluated' "
                                   f"block there is round 7's trial 2")
            items.append(it)
            continue
        # replacement (round-8 phase F)
        r = pool[job]
        rr = reloc[job]
        exp = json.loads((R8 / "exports" / job / "export.json").read_text())
        d = dec[e]
        run_dir = r["run_dir"]
        zc = zcode.get(run_dir) or {}
        rel_run = run_dir.split("/runs/", 1)[1]  # 2026-11-02-tier2-r8-handover/ws031/...
        copy_dir = f"selection/round8_20261103/zuxing_copies_r8/{rel_run}"
        overrides = [ln[2:].strip() for ln in (R8 / "zuxing_copies_r8" / rel_run / "hydra/.hydra/overrides.yaml")
                     .read_text().splitlines() if ln.startswith("- ")]
        first_line = (R8 / "zuxing_copies_r8" / rel_run / "train_log_first_train_t2_line.txt").read_text().strip()
        if exp.get("sha256") != x["export_sha256"] or exp.get("source_sha256") != r["training_sha256"] or \
                rr["relocated_sha256"] != r["training_sha256"] or rr["original_sha256"] != r["sha256"] or \
                rr.get("status") != "ok" or rr.get("tensors_identical") != "True":  # fmt: skip
            problems.append(f"ep{k}: export / relocation / pool records of {job} disagree")
        if zc.get("commit") != r["commit"] or zc.get("train_t2_line") != first_line or "commit=82837c8" not in first_line:
            problems.append(f"ep{k}: training-commit records of {job} disagree")
        it = {"episode": e, "job": job, "input": r["input"], "owner": r["owner"], "machine": r["machine"],
              "host_class": r.get("host_class"), "decision_round8": "replaced",
              "round8_decision": {"round7_pick": d["incumbent"], "bar": d["bar"], "m_F": d["m_F"],
                                  "finalist_floor": d["floor_count"], "finalist_good_runs": len(d["finalist_good_runs"]),
                                  "basis": sel["basis"]},
              "export": {"file": ph(exp["file"]), "sha256": exp["sha256"],
                         "record": f"selection/round8_20261103/exports/{job}/export.json", "tree": exp["tree"],
                         "code_commit": exp["code_commit"], "exporter": exp["exporter"], "tolerance": exp["tolerance"],
                         "max_rel_diff": exp["max_rel_diff"], "source_sha256": exp["source_sha256"],
                         "stock_verified": exp.get("stock_verified")},
              "evaluated": evaluated,
              "sequence_id": r["sequence_id"], "parquet": ph(r["parquet"]), "parquet_sha256": r["parquet_sha256"],
              "data_root": ph(r["data_root"]),
              "host_parquet": {"path_as_trained": ph(r["parquet_as_trained"]),
                               "sha256_recorded_by_host": None, "matches_server_parquet": None,
                               "source": f"{copy_dir}/MANIFEST.txt ('parquet sha256' line) and "
                                         "selection/round8_20261103/inputs_r8/runs_r8.csv"},
              "run_label": r["run_label"], "iteration": r["iteration"], "run_last_iteration": r["run_last"],
              "slot": r["slot"], "pool_route": r["routes"],
              "checkpoint": {"evaluated_file": ph(rr["relocated_file"]), "name": Path(r["source_checkpoint"]).name,
                             "sha256": rr["relocated_sha256"], "pool_sha256": r["sha256"],
                             "source_checkpoint": ph(r["source_checkpoint"]), "iteration": r["iteration"],
                             "stored_output_dir": ph(r["output_dir_as_trained"]),
                             "relocated": {"record": "selection/round8_20261103/RELOCATION_R8.tsv",
                                           "as_trained_copy": ph(rr["original_copy"]),
                                           "as_trained_sha256": rr["original_sha256"],
                                           "zuxing_sha256sums_line": rr["sha256sums_line"],
                                           "old_parquet_path": ph(rr["old_parquet_path"]),
                                           "new_parquet_path": ph(rr["new_parquet_path"]),
                                           "fields_rewritten": rr["fields_rewritten"],
                                           "tensors_identical": rr["tensors_identical"] == "True",
                                           "data_block_sha256": rr["data_block_sha256"], "rule": RULE_RELOC}},
              "train": {"machine": f"{r['machine']} (Zuxing's {r.get('host_class')} workstation)",
                        "code_commit": "82837c8569c2461f263c28fce406c1b517f40d93",
                        "train_log_first_line": first_line,
                        "code_commit_source": (f"selection/round8_20261103/inputs_r8/ZUXING_CODE_R8.tsv row {rel_run} "
                                               f"({zc.get('commit_source', '').replace(PH['<ZX>'], '<ZX>')}); copy of the "
                                               f"line: {copy_dir}/train_log_first_train_t2_line.txt"),
                        "pushed_heads": "dev.zuxing.hard_start",
                        "code_in_this_history": "82837c8 (dev.zuxing.hard_start; an ancestor of this commit)",
                        "launcher": "not on the team server; not in this commit (README, 'Missing')",
                        "stages": [{"stage": 1, "run": ph(run_dir), "record": f"{copy_dir}/hydra/.hydra",
                                    "record_kind": "hydra", "overrides": [ph(o) for o in overrides],
                                    "final_json": f"{copy_dir}/final.json", "manifest": f"{copy_dir}/MANIFEST.txt",
                                    "sha256sums": f"{copy_dir}/SHA256SUMS"}]}}
        # host parquet record (the hand-over MANIFEST line), as runs_r8.csv records it
        with open(R8 / "inputs_r8/runs_r8.csv", newline="") as fp:
            rrow = next((y for y in csv.DictReader(fp) if y["dir"] == run_dir), None)
        if rrow is None or rrow["server_parquet_sha256"] != r["parquet_sha256"]:
            problems.append(f"ep{k}: runs_r8.csv row of {run_dir} missing or server parquet sha256 differs")
        else:
            it["host_parquet"]["sha256_recorded_by_host"] = rrow["host_parquet_sha256"]
            it["host_parquet"]["matches_server_parquet"] = rrow["host_parquet_matches_server"] == "True"
        items.append(it)
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    out = {"what": f"Track 2 Tier 2 submission set R8-CLOSEST (round 8: trial {t} of the 10 complete trials of S8)",
           "basis": sel["basis"], "selection_r8_json_sha256": sha(a.selection),
           "selection_r8_sha256_file": sha(a.selection.parent / "SELECTION_R8.sha256"),
           "set_items_r7_json_sha256": sha(a.r7_items), "pack_file_json_sha256": sha(a.pack_file),
           "r8_closest_trial": t, "replaced_episodes": sel["replaced_episodes"], "placeholders": PH,
           "paths": "relative paths are relative to robotic_grounding/flash_chord/scripts/track2/t2r8_20261103/",
           "episodes": items}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    p1, p2 = a.out_dir / "set_items_r8.json", a.out_dir / "set_manifest_r8.tsv"
    for p in (p1, p2):
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
    p1.write_text(json.dumps(out, indent=1) + "\n")
    cols = ["episode", "job", "input", "owner", "machine", "decision_round8", "export_sha256", "rollout_round", "trial",
            "run", "try", "rollout_sha256", "frame_reference", "training_commit"]
    with open(p2, "w", newline="") as fp:
        w = csv.writer(fp, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for it in items:
            ev = it["evaluated"]
            w.writerow([it["episode"], it["job"], it["input"], it["owner"], it.get("machine") or "",
                        it["decision_round8"], it["export"]["sha256"], ev["round"], ev["trial"], ev["run"], ev["try"],
                        ev["rollout_sha256"], ev["frame_reference"], it["train"]["code_commit"]])
    print(f"wrote {p1} ({sha(p1)}) and {p2} ({sha(p2)}); {len(items)} episodes; trial {t}; replaced "
          f"{sel['replaced_episodes']}; round-7 relative paths re-pointed: {len(changed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
