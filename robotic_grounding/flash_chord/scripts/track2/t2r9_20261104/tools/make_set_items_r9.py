#!/usr/bin/env python3
"""make_set_items_r9.py: write set/set_items_r9.json and set/set_manifest_r9.tsv of the R9-CLOSEST file (round 9,
trial t* of S9) from frozen records only. Nothing is evaluated, scored or selected here (round 8's
tools/make_set_items_r8.py with round 9's sources).

Sources (all read only):
  --selection   round 9's SELECTION_R9.json (S9, run sources, R9-CLOSEST trial and its per-episode rollouts / exports)
  --r8-items    round 8's set/set_items_r8.json as committed in the record (t2r8_20261103/set/set_items_r8.json)
  --pack-file   the pack step's pack_r9_placeholder/r9_closest/file.json (what was copied into the packed folder)
  --r9          the round-9 folder (POOL_R9.json, RELOCATION_R9.tsv, exports/, inputs_r9/, zuxing_copies_r9/)
Kept round-8 picks: the round-8 item, with its record paths relative to t2r8_20261103/ pointed there from
t2r9_20261104/ (paths into ../t2r7_20261102/ stay valid) and its "evaluated" block replaced by the recorded phase-F run
of trial t* (round 7's for a round-7 pick kept in round 8, round 8's for episode 13). A round-9 replacement: a new item
from round 9's pool, report-file record, relocation, export, training-commit and copy records. Every item's export /
rollout sha256 must equal both SELECTION_R9.json and the pack file. New files only.
Run: <python> -I -B make_set_items_r9.py --selection S --r8-items I --pack-file F --r9 R9 --out-dir DIR
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import copy  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import struct  # noqa: E402
from pathlib import Path  # noqa: E402

PH = {"<ME>": "/root/autodl-tmp/haoran.zhang", "<ZX>": "/root/autodl-tmp/zuxing.lu",
      "<ZX5090>": "/mnt/gfs/home/zuxinglu/workspace", "<ZXWS>": "/home/zuxinglu/workspace"}
R8_PREFIXES = ("selection/", "set/", "pack/", "tools/")
R8_INNER = re.compile(r"(?<![\w./-])(selection/round8_20261103/)")
RULE_RELOC = ("only task.parquet and reference.parquet rewritten (as-trained host path -> this server's path of the same "
              "parquet); every tensor byte unchanged")
COMMIT_82837C8 = "82837c8569c2461f263c28fce406c1b517f40d93"
EVIDENCE_5090 = "runs/2026-11-02-tier2-r8-handover/evidence/5090.txt"


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


def to_r9_relative(obj, changed: list, where: str = ""):
    """Record paths of a round-8 item are relative to t2r8_20261103/; point them there from t2r9_20261104/."""
    if isinstance(obj, dict):
        return {k: to_r9_relative(v, changed, f"{where}.{k}") for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_r9_relative(v, changed, f"{where}[{i}]") for i, v in enumerate(obj)]
    if isinstance(obj, str):
        new = "../t2r8_20261103/" + obj if obj.startswith(R8_PREFIXES) else obj
        new = R8_INNER.sub(r"../t2r8_20261103/\1", new)  # inner mentions (not already behind a '/')
        if new != obj:
            changed.append(where)
        return new
    return obj


def embedded_config(p: Path) -> dict:
    """The training configuration embedded in a learner_state checkpoint (safetensors header, read only)."""
    with open(p, "rb") as fp:
        n = struct.unpack("<Q", fp.read(8))[0]
        head = json.loads(fp.read(n))
    meta = head.get("__metadata__", {})
    c = json.loads(meta["user.resolved_config_json"])
    return {"output_dir": c.get("output_dir"), "task.parquet": c["task"].get("parquet"),
            "training.max_iterations": c["training"].get("max_iterations"), "training.seed": c["training"].get("seed"),
            "training.early_stop_help_zero_tail": c["training"].get("early_stop_help_zero_tail"),
            "scene.world_count": c["scene"].get("world_count"), "scene.support": c["scene"].get("support"),
            "reset.seed": (c.get("reset") or {}).get("seed"), "iteration": int(meta.get("user.iteration", -1)),
            "format": meta.get("format")}  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("selection", "r8-items", "pack-file", "r9", "out-dir"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    a = ap.parse_args()
    sel = json.loads(a.selection.read_text())
    r8 = json.loads(a.r8_items.read_text())
    pf = json.loads(a.pack_file.read_text())
    R9 = a.r9
    t = int(sel["r9_closest_trial"])
    s9, src, s8 = sel["compositions"]["S9"], sel["compositions"]["S9_run_source"], sel["compositions"]["S8_round8"]
    r8_by = {it["episode"]: it for it in r8["episodes"]}
    pf_by = {r["episode"]: r for r in pf["episodes"]}
    dec = {x["episode"]: x for x in sel["episodes"]}
    pool = {r["job"]: r for r in json.loads((R9 / "POOL_R9.json").read_text())["rows"]}
    reports = {x["file"]: x for x in json.loads((R9 / "inputs_r9/reports_r9.json").read_text())["files"]}
    with open(R9 / "RELOCATION_R9.tsv", newline="") as fp:
        reloc = {r["job"]: r for r in csv.DictReader(fp, delimiter="\t")}
    with open(R9 / "inputs_r9/ZUXING_CODE_R9.tsv", newline="") as fp:
        zcode = {r["source"]: r for r in csv.DictReader(fp, delimiter="\t")}
    ev5090 = (R9 / "zuxing_copies_r9" / EVIDENCE_5090).read_text().splitlines()
    problems, items, changed = [], [], []
    for e in range(30):
        k = f"{e:02d}"
        x = sel["r9_closest_file"][k]
        p = pf_by[e]
        job = s9[k]
        if x["job"] != job or p["job"] != job or p["export_sha256"] != x["export_sha256"] or \
                p["rollout_sha256"] != x["rollout_sha256"] or p["rollout_src"] != x["rollout"] or int(p["run"]) != t:
            problems.append(f"ep{k}: SELECTION_R9 / pack file disagree")
        rnd = {"r7F": "round 7 phase F", "r8F": "round 8 phase F", "r9F": "round 9 phase F"}[src[k]]
        evaluated = {"round": rnd, "trial": t, "run": t, "try": int(x["try"]), "rollout": ph(x["rollout"]),
                     "rollout_sha256": x["rollout_sha256"], "frame_reference": ph(x["frame_reference"]),
                     "frame_reference_file_sha256": p["frame_ref_src_sha256"]}
        if src[k] in ("r7F", "r8F"):
            it8 = r8_by[e]
            if it8["job"] != job or it8["export"]["sha256"] != x["export_sha256"] or s8[k] != job:
                problems.append(f"ep{k}: round-8 item {it8['job']} / export != the kept round-8 pick {job}")
            if it8["evaluated"]["round"] != rnd:
                problems.append(f"ep{k}: round-8 item's run source {it8['evaluated']['round']} != {rnd}")
            it = to_r9_relative(copy.deepcopy(it8), changed, f"ep{k}")
            it["evaluated"] = evaluated
            it["decision_round9"] = "round-8 pick kept (no round-9 finalist replaced it)"
            it["round8_record"] = (f"../t2r8_20261103/set/set_items_r8.json (episode {e}, job {job}); its 'evaluated' "
                                   f"block there is round 8's trial 7")
            items.append(it)
            continue
        # replacement (round-9 phase F)
        r = pool[job]
        rr = reloc[job]
        rep = reports[r["report_file"]]
        exp = json.loads((R9 / "exports" / job / "export.json").read_text())
        d = dec[e]
        zc = zcode.get(r["report_file"]) or {}
        copy_rel = r["copy"].split("/zuxing_copies_r9/", 1)[1]  # reports/tier2-*/epNN-tier2-*.safetensors
        emb = embedded_config(R9 / "zuxing_copies_r9" / copy_rel)
        if exp.get("sha256") != x["export_sha256"] or exp.get("source_sha256") != r["training_sha256"] or \
                rr["relocated_sha256"] != r["training_sha256"] or rr["original_sha256"] != r["sha256"] or \
                rr.get("status") != "ok" or rr.get("tensors_identical") != "True" or rep["sha256"] != r["sha256"] or \
                sha(R9 / "zuxing_copies_r9" / copy_rel) != r["sha256"]:  # fmt: skip
            problems.append(f"ep{k}: export / relocation / pool / report records of {job} disagree")
        if zc.get("commit") != r["commit"] or zc.get("train_t2_line") != rep.get("train_t2_line") or \
                "commit=82837c8" not in (zc.get("train_t2_line") or ""):  # fmt: skip
            problems.append(f"ep{k}: training-commit records of {job} disagree")
        if emb["output_dir"] != r["output_dir_as_trained"] or emb["iteration"] != int(r["iteration"]) or \
                emb["training.max_iterations"] != r["max_iterations"] or emb["task.parquet"] != r["parquet_as_trained"]:
            problems.append(f"ep{k}: embedded configuration of {job} != its pool record")
        if not d.get("changed") or d.get("pick") != job or d.get("incumbent") != s8[k]:
            problems.append(f"ep{k}: SELECTION_R9 decision does not replace {s8[k]} by {job}")
        run_rel = r["output_dir_as_trained"].split("/runs/", 1)[1]  # e.g. 2026-10-31-tier2-5090-round2/ep17
        copy_runs = [y for y in rep.get("run_records_here") or []]
        stage = {"stage": 1, "run": ph(r["output_dir_as_trained"]),
                 "embedded_config": {kk: (ph(v) if isinstance(v, str) else v) for kk, v in emb.items()},
                 "embedded_config_source": f"selection/round9_20261103/zuxing_copies_r9/{copy_rel} (not committed; "
                                           f"sha256 {r['sha256']}), header field user.resolved_config_json"}  # fmt: skip
        if copy_runs:
            crel = copy_runs[0].split("/zuxing.lu/runs/", 1)[1]
            hyd = R9 / "zuxing_copies_r9/runs" / crel / "hydra/.hydra/overrides.yaml"
            if hyd.is_file():
                stage["record"] = f"selection/round9_20261103/zuxing_copies_r9/runs/{crel}/hydra/.hydra"
                stage["record_kind"] = "hydra (copy run on the team server, which ends before this checkpoint)"
                stage["overrides"] = [ph(ln[2:].strip()) for ln in hyd.read_text().splitlines() if ln.startswith("- ")]
            stage["copy_run_on_team_server"] = ph(copy_runs[0])
        else:
            stage["record_kind"] = "embedded configuration only (no run record of this run on the team server)"
        host_line = next((ln for ln in ev5090 if ln.startswith(f"{run_rel} | ") and "/data.parquet" in ln), None)
        host_sha = host_line.split(" | ")[1].strip() if host_line else None
        it = {"episode": e, "job": job, "input": r["input"], "owner": r["owner"], "machine": r["machine"],
              "host_class": r.get("host_class"), "decision_round9": "replaced",
              "round9_decision": {"round8_pick": d["incumbent"], "bar": d.get("bar"), "m_F": d.get("m_F"),
                                  "finalist_floor": d.get("floor_count"),
                                  "finalist_good_runs": len(d.get("finalist_good_runs") or []),
                                  "basis": sel["basis"]},
              "export": {"file": ph(exp["file"]), "sha256": exp["sha256"],
                         "record": f"selection/round9_20261103/exports/{job}/export.json", "tree": exp["tree"],
                         "code_commit": exp["code_commit"], "exporter": exp["exporter"], "tolerance": exp["tolerance"],
                         "max_rel_diff": exp["max_rel_diff"], "source_sha256": exp["source_sha256"],
                         "stock_verified": exp.get("stock_verified")},
              "evaluated": evaluated,
              "sequence_id": r["sequence_id"], "parquet": ph(r["parquet"]), "parquet_sha256": r["parquet_sha256"],
              "data_root": ph(r["data_root"]),
              "host_parquet": {"path_as_trained": ph(r["parquet_as_trained"]), "sha256_recorded_by_host": host_sha,
                               "matches_server_parquet": (host_sha == r["parquet_sha256"]) if host_sha else None,
                               "source": (f"selection/round9_20261103/zuxing_copies_r9/{EVIDENCE_5090} ('task.parquet "
                                          f"sha256' section, row {run_rel})") if host_sha else
                                         "not recorded by the host (no hand-over line for this run)"},
              "report_file": {"path": ph(r["report_file"]), "sha256": r["sha256"], "folder": r.get("copy_folder"),
                              "record": "selection/round9_20261103/inputs_r9/reports_r9.json",
                              "mtime": rep.get("mtime")},
              "run_label": r["run_label"], "iteration": r["iteration"], "max_iterations": r["max_iterations"],
              "final_status_at_t_c9": {"final": r["final_at_t_c"], "evidence": r["final_evidence"],
                                       "check_report": r.get("check_report_final"),
                                       "note": ("not the run's final checkpoint: the run had not reached its planned "
                                                "end (max_iterations) at T_c9; Zuxing's hand-over names this file the "
                                                "run's latest checkpoint")},
              "slot": r["slot"], "pool_route": r["routes"],
              "checkpoint": {"evaluated_file": ph(rr["relocated_file"]), "name": Path(r["source_checkpoint"]).name,
                             "sha256": rr["relocated_sha256"], "pool_sha256": r["sha256"],
                             "source_checkpoint": ph(r["source_checkpoint"]), "iteration": r["iteration"],
                             "stored_output_dir": ph(r["output_dir_as_trained"]),
                             "relocated": {"record": "selection/round9_20261103/RELOCATION_R9.tsv",
                                           "as_trained_copy": ph(rr["original_copy"]),
                                           "as_trained_sha256": rr["original_sha256"],
                                           "old_parquet_path": ph(rr["old_parquet_path"]),
                                           "new_parquet_path": ph(rr["new_parquet_path"]),
                                           "fields_rewritten": rr["fields_rewritten"],
                                           "tensors_identical": rr["tensors_identical"] == "True",
                                           "data_block_sha256": rr["data_block_sha256"], "rule": RULE_RELOC}},
              "train": {"machine": f"{r['machine']} (Zuxing's RTX 5090 host)" if r["machine"] == "5090" else r["machine"],
                        "code_commit": COMMIT_82837C8,
                        "train_log_first_line": zc.get("train_t2_line"),
                        "code_commit_source": ("selection/round9_20261103/inputs_r9/ZUXING_CODE_R9.tsv row of "
                                               f"{ph(r['report_file'])} ({ph(zc.get('commit_source', ''))})"),
                        "pushed_heads": "dev.zuxing.hard_start",
                        "code_in_this_history": "82837c8 (dev.zuxing.hard_start; an ancestor of this commit)",
                        "launcher": "not on the team server; not in this commit (README, 'Missing or open')",
                        "stages": [stage]}}
        items.append(it)
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    out = {"what": f"Track 2 Tier 2 submission set R9-CLOSEST (round 9: trial {t} of the 10 complete trials of S9)",
           "basis": sel["basis"], "selection_r9_json_sha256": sha(a.selection),
           "selection_r9_sha256_file": sha(a.selection.parent / "SELECTION_R9.sha256"),
           "set_items_r8_json_sha256": sha(a.r8_items), "pack_file_json_sha256": sha(a.pack_file),
           "r9_closest_trial": t, "replaced_episodes": sel["replaced_episodes"], "placeholders": PH,
           "paths": "relative paths are relative to robotic_grounding/flash_chord/scripts/track2/t2r9_20261104/",
           "episodes": items}
    a.out_dir.mkdir(parents=True, exist_ok=True)
    p1, p2 = a.out_dir / "set_items_r9.json", a.out_dir / "set_manifest_r9.tsv"
    for p in (p1, p2):
        if p.exists():
            raise SystemExit(f"refusing to overwrite {p}")
    p1.write_text(json.dumps(out, indent=1) + "\n")
    cols = ["episode", "job", "input", "owner", "machine", "decision_round9", "export_sha256", "rollout_round", "trial",
            "run", "try", "rollout_sha256", "frame_reference", "training_commit"]
    with open(p2, "w", newline="") as fp:
        w = csv.writer(fp, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for it in items:
            ev = it["evaluated"]
            w.writerow([it["episode"], it["job"], it["input"], it["owner"], it.get("machine") or "",
                        it["decision_round9"], it["export"]["sha256"], ev["round"], ev["trial"], ev["run"], ev["try"],
                        ev["rollout_sha256"], ev["frame_reference"], it["train"]["code_commit"]])
    print(f"wrote {p1} ({sha(p1)}) and {p2} ({sha(p2)}); {len(items)} episodes; trial {t}; replaced "
          f"{sel['replaced_episodes']}; round-8 relative paths re-pointed: {len(changed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
