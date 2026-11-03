#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""stage_t3ready.py: stage the Tier-3 submission record (FINAL_r) for one commit, a child of 41403d3.

Steps (CPU only; reads anything except files whose name marks clean / GT material; writes only under W and BUNDLE):
  copy   read-only copies of Han Zhou's release files that the record needs (code, READMEs, episodes.json, support
         fit records, the 30 opus agent runs): sha256 of the source before, of the copy, and of the source after;
         every row must agree (W/COPIES_t3ready.tsv). Han's folder is never written.
  set    set/set_items.json: the 30 FINAL_r items (19 f0, 11 retry) with every training stage's saved Hydra
         overrides, checkpoint sha256, export record and the packed rollout; extracts each checkpoint's stored resolved
         config (safetensors metadata) to set/checkpoint_config/ and copies the run records (final.json, fork.json,
         export_report.json, Hydra config.yaml / overrides.yaml).
  stage  copy the code and records into BUNDLE byte for byte, with SOURCES.tsv (bundle path, source, sha256, bytes).
         Never staged: any name containing a forbidden marker (LOCAL_NOT_FOR_UPLOAD, clean, VERDICT_clean,
         summary_LOCAL, s4a_, s4b_, clean_stat, report_LOCAL, GT, gt_eval), Git-LFS file types, images, bytecode.
  sums   write BUNDLE/SHA256SUMS (every file but itself).
  scan   hygiene scan of BUNDLE: forbidden names, LFS types, secret / proxy patterns, and lines naming clean / GT
         material with every digit masked (so the reviewer sees no number), for review.

  python -I -B stage_t3ready.py copy|set|stage|sums|scan
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

ME = Path("/root/autodl-tmp/haoran.zhang")
HAN = Path("/root/autodl-tmp/han.zhou")
REL = HAN / "tier3/release"
W = ME / "tmp_analysis/tier3/t3ready_20261102"
HC = W / "han_copies"
COPIES = W / "COPIES_t3ready.tsv"
BUNDLE = ME / "code/v2d_t3ready/robotic_grounding/flash_chord/scripts/track2/t3ready_20261102"
F = ME / "fc_runs/submissions/t3_f0_final"
RA = ME / "tmp_analysis/tier3/f0_retry"
F0A = ME / "tmp_analysis/tier3/f0"
ROOT = ME / "data/human_motion_data_tier3_f0"
RUNS_F0 = ME / "fc_runs/C3r2_tier3_f0"
RUNS_R = ME / "fc_runs/C3r2_tier3_f0_retry"
ROOTS = [(str(ME), "<ME>"), (str(HAN), "<HAN>")]
FORBIDDEN = ("local_not_for_upload", "clean", "summary_local", "s4a_", "s4b_", "report_local", "gt_eval")
LFS_EXT = {".obj", ".glb", ".stl", ".mtl", ".dae", ".usd", ".usda", ".usdc", ".usdz", ".h5", ".hdf5", ".parquet",
           ".npy", ".npz", ".pkl", ".pickle", ".onnx", ".pt", ".safetensors", ".png", ".jpg", ".jpeg", ".webp",
           ".gif", ".mp4", ".tgz", ".tar", ".gz", ".zip", ".pyc", ".pkg"}
FINAL_R_CSV_SHA = "bd73e0e94e168d055ec200cf74eeec9cff945345e9d523f589b6587183f23d52"
FINAL_R_POSE_SHA = "82fef2c7d09326191098df29567fcc6d0b51cb8523ad70331b60eb4535c36303"
FC_R2_COMMIT = "61fa5d31ff8054bf1e5cf5375ef0957e54d2729b"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def ph(s):
    if isinstance(s, str):
        for a, b in ROOTS:
            s = s.replace(a, b)
    return s


def ph_all(x):
    if isinstance(x, dict):
        return {k: ph_all(v) for k, v in x.items()}
    if isinstance(x, list):
        return [ph_all(v) for v in x]
    return ph(x)


def name_forbidden(rel: str) -> bool:
    low = rel.lower()
    if any(m in low for m in FORBIDDEN):
        return True
    return any("gt" == part.lower() or part.startswith("GT") or "_GT" in part or "GT_" in part
               for part in re.split(r"[/.]", rel))


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def st_meta(path: Path) -> dict:
    with open(path, "rb") as fp:
        n = struct.unpack("<Q", fp.read(8))[0]
        return json.loads(fp.read(n)).get("__metadata__", {}) or {}


def stored_config(md: dict) -> dict:
    return json.loads(md.get("user.resolved_config_json") or md["resolved_config_json"])


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n")


# ============================================================================================ copy (Han's files)
def han_sources() -> list[Path]:
    f0 = REL / "tier3_frame0_supports_20261024"
    out = [f0 / "README.md", f0 / "episodes.json", f0 / "supports/README_supports.md", f0 / "supports/index.json",
           f0 / "supports/contact_report.json"]
    out += sorted(f0.glob("supports/*/support.json"))
    out += sorted(f0.glob("reconstructed_data/*/episode.json"))
    for seq in sorted(p.name for p in (f0 / "agent_run").iterdir()):
        out += [f0 / "agent_run" / seq / n for n in ("solve.py", "TOOLS.md", "REPORT.md")]
    op = REL / "tier3_opus_20261021"
    out += [op / "README.md", op / "episodes.json"]
    s1 = REL / "tier3_s1_20261020"
    out += [s1 / "README.md", s1 / "episodes.json"] + sorted((s1 / "code").glob("*.py"))
    for p in out:
        rel = str(p.relative_to(REL))
        if name_forbidden(rel):
            raise SystemExit(f"refusing a forbidden name: {rel}")
        if not p.is_file():
            raise SystemExit(f"missing: {p}")
    return out


def cmd_copy() -> int:
    if COPIES.exists():
        print(f"{COPIES} exists: copy is write-once")
        return 1
    rows = []
    for src in han_sources():
        rel = src.relative_to(REL)
        dst = HC / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        before = sha256(src)
        st = src.stat()
        with open(src, "rb") as fi, open(dst, "wb") as fo:  # plain read; no metadata copy, nothing written at src
            shutil.copyfileobj(fi, fo)
        after = sha256(src)
        cp = sha256(dst)
        ok = before == cp == after
        rows.append({"source": str(src), "copy": str(dst), "bytes": st.st_size,
                     "source_mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
                     "sha256_source_before": before, "sha256_copy": cp, "sha256_source_after": after, "ok": ok,
                     "copied_at": now()})
    # agent_run of the opus release vs the frame-0 release (README: kept as is)
    same = []
    f0a, opa = REL / "tier3_frame0_supports_20261024/agent_run", REL / "tier3_opus_20261021/agent_run"
    for seq in sorted(p.name for p in f0a.iterdir()):
        for n in ("solve.py", "TOOLS.md", "REPORT.md"):
            a, b = f0a / seq / n, opa / seq / n
            same.append(b.is_file() and sha256(a) == sha256(b))
    with open(COPIES, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    note = {"rows": len(rows), "all_ok": all(r["ok"] for r in rows),
            "opus_agent_run_identical_to_frame0_agent_run": f"{sum(same)}/{len(same)}"}
    write_json(W / "COPIES_t3ready.summary.json", note)
    print(json.dumps(note))
    return 0 if note["all_ok"] else 1


# ============================================================================================ set
def load(p: Path) -> dict:
    return json.loads(p.read_text())


def hydra_record(run: Path, tag: str) -> tuple[list[str], str]:
    src = run / "hydra/.hydra"
    dst = BUNDLE / "set/hydra" / tag
    dst.mkdir(parents=True, exist_ok=True)
    for n in ("config.yaml", "overrides.yaml"):
        shutil.copyfile(src / n, dst / n)
    import yaml  # PyYAML (flashchord env)
    ovs = [str(x) for x in yaml.safe_load((src / "overrides.yaml").read_text())]
    return ovs, f"set/hydra/{tag}"


def ckpt_record(path: Path, want_sha: str | None, tag: str) -> dict:
    s = sha256(path)
    if want_sha and s != want_sha:
        raise SystemExit(f"{path}: sha256 {s} != recorded {want_sha}")
    md = st_meta(path)
    cfg = stored_config(md)
    rel = f"set/checkpoint_config/{tag}.resolved_config.json"
    write_json(BUNDLE / rel, cfg)
    return {"name": path.name, "file": ph(str(path)), "sha256": s, "iteration": int(md.get("user.iteration", -1)),
            "stored_config": rel}


def grep_lines(path: Path, *pats: str) -> list[str]:
    out = []
    with open(path, errors="replace") as fp:
        for line in fp:
            if all(x in line for x in pats):
                out.append(ph(line.rstrip("\n")))
    return out


def cmd_set() -> int:
    fj = load(F / "pack/FINAL_r/file.json")
    bj = load(F / "pack/FINAL_r/build.json")
    if bj["csv_sha256"] != FINAL_R_CSV_SHA or bj["pose_columns_sha256"] != FINAL_R_POSE_SHA:
        raise SystemExit("FINAL_r build.json does not name the locked file")
    choices = {int(r["episode"]): r for r in csv.DictReader(open(F / "pack/FINAL_r/choices.csv"))}
    items = []
    for key, ent in sorted(fj["episodes"].items()):
        e = int(key)
        it = load(F / f"items/ep{e:02d}.json")
        run0 = RUNS_F0 / f"ep{e:02d}"
        fin0 = load(run0 / "final.json")
        shutil.copyfile(run0 / "final.json", (BUNDLE / "set/final").mkdir(parents=True, exist_ok=True)
                        or BUNDLE / f"set/final/ep{e:02d}_f0.final.json")
        ov0, rec0 = hydra_record(run0, f"ep{e:02d}_f0")
        stages = []
        if ent["kind"] == "f0":
            ck0 = ckpt_record(Path(it["training_checkpoint"]), it["training_sha256"],
                              f"ep{e:02d}_f0_{Path(it['training_checkpoint']).stem}")
            if fin0["checkpoint"] != Path(it["training_checkpoint"]).name:
                raise SystemExit(f"ep{e:02d}: final.json checkpoint != item")
            stages.append({"stage": "f0", "run_dir": ph(str(run0)), "record": rec0, "record_kind": "hydra",
                           "overrides": ph_all(ov0), "final_json": f"set/final/ep{e:02d}_f0.final.json",
                           "checkpoint": ck0})
            export_file, export_sha = Path(it["eval_file"]), it["eval_sha256"]
            export_report = Path(it["export_report"])
            tol = it.get("export_tolerance")
        else:
            rd = RUNS_R / f"ep{e:02d}"
            fork = load(rd / "fork.json")
            fin = load(rd / "final.json")
            shutil.copyfile(rd / "final.json", BUNDLE / f"set/final/ep{e:02d}_retry.final.json")
            shutil.copyfile(rd / "fork.json", BUNDLE / f"set/final/ep{e:02d}_retry.fork.json")
            src44 = Path(fork["files"]["model"]["src"])
            ck0 = ckpt_record(src44, fork["files"]["model"]["sha256"], f"ep{e:02d}_f0_{src44.stem}")
            stages.append({"stage": "f0", "run_dir": ph(str(run0)), "record": rec0, "record_kind": "hydra",
                           "overrides": ph_all(ov0), "final_json": f"set/final/ep{e:02d}_f0.final.json",
                           "checkpoint": ck0,
                           "note": "f0 final model_44922 + rsi_44922, copied (sha256 checked) into the retry run dir "
                                   "as the resume point (fork.json)"})
            ov1, rec1 = hydra_record(rd, f"ep{e:02d}_retry")
            ckr = ckpt_record(rd / fin["checkpoint"], ent["retry_final_sha256"], f"ep{e:02d}_retry_{Path(fin['checkpoint']).stem}")
            stages.append({"stage": "retry_C90", "run_dir": ph(str(rd)), "record": rec1, "record_kind": "hydra",
                           "overrides": ph_all(ov1), "final_json": f"set/final/ep{e:02d}_retry.final.json",
                           "fork_json": f"set/final/ep{e:02d}_retry.fork.json", "checkpoint": ckr,
                           "recipe": f"evaluation/f0_retry/recipes/ep{e:02d}.json"})
            export_file, export_sha = Path(ent["evaluated_checkpoint"]), ent["eval_sha256"]
            export_report = export_file.parent / "export_report.json"
            tol = 1e-4 if "tol0.0001" in ent["retry_dir"] else None
        last = stages[-1]["checkpoint"]
        if sha256(export_file) != export_sha:
            raise SystemExit(f"ep{e:02d}: evaluated file sha256 mismatch")
        src_sha = st_meta(export_file).get("user.export_source_sha256")
        shutil.copyfile(export_report, (BUNDLE / "set/export").mkdir(parents=True, exist_ok=True)
                        or BUNDLE / f"set/export/ep{e:02d}.export_report.json")
        ch = choices.get(e, {})
        cmds = {"train_launch_f0": grep_lines(F0A / "driver.log", f"[f0] ep{e:02d} launch on", "train_rl.py")}
        if ent["kind"] == "f0":
            cmds["export_and_post_eval_f0"] = grep_lines(F0A / f"eval/ep{e:02d}/eval_all.log", "eval_all.py")
            cmds["rollout_eval"] = grep_lines(Path(ent["json"]).parents[1] / "eval_all.log", "eval_all.py")
        else:
            cmds["train_launch_retry"] = grep_lines(RA / "driver.log", f"[retry] ep{e:02d} LAUNCHED", "train_rl.py")
            cmds["export_and_rollout_eval"] = grep_lines(Path(ent["retry_dir"]) / "eval_all.log", "eval_all.py")
        for k, v in cmds.items():
            if len(v) != 1:
                raise SystemExit(f"ep{e:02d}: {k}: {len(v)} matching lines")
        items.append({
            "episode": e, "object": ent["object"], "sequence_id": it["sequence_id"], "kind": ent["kind"],
            "scene_support": it.get("scene_support"), "mitigated": it.get("mitigated"),
            "train": {"code_commit": FC_R2_COMMIT, "code_tree": ph(it.get("code_tree", "")),
                      "this_commit_path": "robotic_grounding/flash_chord (code files identical to fc_r2 @ 61fa5d3)",
                      "entry": "scripts/train_rl.py (python from conda_envs/flashchord, run in the flash_chord dir)",
                      "stages": stages},
            "checkpoint": last,
            "export": {"file": ph(str(export_file)), "sha256": export_sha, "export_source_sha256": src_sha,
                       "export_source_is_checkpoint": src_sha == last["sha256"], "tolerance": tol,
                       "report": f"set/export/ep{e:02d}.export_report.json"},
            "recorded_commands": cmds,
            "rollout": {"run": ent["run"], "results_json": ph(ent["json"]), "results_json_sha256": ent["json_sha256"],
                        "results_parquet": ph(ent["parquet"]), "results_parquet_sha256": ent["parquet_sha256"],
                        "packed_copy": f"evaluation/t3_f0_final/pack/FINAL_r/results/episode_{e:06d}.json"},
            "input": {"parquet": ph(it["parquet"]), "parquet_sha256": it["parquet_sha256"],
                      "frame_ref": ph(ent["frame_ref"]), "frame_ref_sha256": ent["frame_ref_sha256"],
                      "manifest_sha256": it.get("manifest_sha256")},
            "own_input_of_packed_rollout": {k: ch.get(k) for k in ("own_auc", "own_sp_sr", "own_mp_sr", "own_mppe",
                                                                   "chosen_run", "arm_mean_auc", "decision", "why")},
        })
        if not items[-1]["export"]["export_source_is_checkpoint"]:
            raise SystemExit(f"ep{e:02d}: evaluated file's export source is not the training checkpoint")
    out = {"what": "the 30 items of Tier-3 FINAL_r (t3_f0_final/pack/FINAL_r, csv sha256 " + FINAL_R_CSV_SHA +
                   ", pose columns sha256 " + FINAL_R_POSE_SHA + ")",
           "placeholders": {"<ME>": str(ME), "<HAN>": str(HAN)},
           "built_at": now(), "episodes": items}
    write_json(BUNDLE / "set/set_items.json", out)
    print(f"set: {len(items)} items, stages {sum(len(i['train']['stages']) for i in items)}")
    return 0


# ============================================================================================ stage
def plan() -> list[tuple[str, Path]]:
    """(bundle path, source) for every staged file. Generated files (set/, tools/, README, verification/) are not
    listed here."""
    L: list[tuple[str, Path]] = []

    def add(dst: str, src: Path):
        L.append((dst, src))

    # ---- B. data preparation
    for n in ("t3f0_build_root.py", "t3f0_frames.py", "t3f0_retarget.sh", "t3f0_retarget_all.sh", "t3f0_rcheck.py",
              "t3f0_supports.py", "t3f0_lean_search.py", "t3f0_physcheck.py", "t3f0_prep_table.py", "t3f0_align.py",
              "t3f0_driver.sh", "t3f0_helper.py", "track2_queue_tier2v2_helper.py", "t3opus_helper_v2.py", "fc_env.sh"):
        add(f"data_prep/scripts/{n}", ME / "scripts" / n)
    for n in ("physcheck_chain.sh", "repin_watch.sh", "build_files_f0.sha256", "cfgcheck_f0.json", "prep_f0.json",
              "prep_f0.tsv", "lean_search_ep01.log", "lean_search_ep01_pass1_16dir.log", "supports_build.log",
              "supports_build2.log", "supports_stool.log", "ALL_LAUNCHED"):
        add(f"data_prep/f0/{n}", F0A / n)
    for p in sorted((F0A / "build").iterdir()):
        add(f"data_prep/f0/build/{p.name}", p)
    for p in sorted((F0A / "align").glob("*")):
        if p.is_file() and (p.suffix in (".json", ".tsv", ".sha256", ".sh") or p.name.endswith(".log") or p.name == "v3_done.txt"):
            add(f"data_prep/f0/align/{p.name}", p)
    for p in sorted((F0A / "retarget").iterdir()):
        if p.is_file() and (p.name in ("parquet_lock.json", "run.log") or p.name.endswith(("_rcheck.json", "_retarget.status"))):
            add(f"data_prep/f0/retarget/{p.name}", p)
    for p in sorted((F0A / "supports").glob("*.json")):
        add(f"data_prep/f0/supports/{p.name}", p)
    for p in sorted((F0A / "lean_search").glob("*.json")):
        add(f"data_prep/f0/lean_search/{p.name}", p)
    for p in sorted((F0A / "physcheck").iterdir()):
        if p.is_file() and p.suffix in (".json", ".log"):
            add(f"data_prep/f0/physcheck/{p.name}", p)
    for p in sorted((F0A / "frames").glob("ep??.json")):
        add(f"data_prep/f0/frames/{p.name}", p)
    for p in sorted((F0A / "review").iterdir()):
        if p.is_file():
            add(f"data_prep/f0/review/{p.name}", p)
    for p in sorted((F0A / "dryrun").glob("*.txt")):
        add(f"data_prep/f0/dryrun/{p.name}", p)
    for n in ("READY", "FILES.sha256", "SOURCE_FILES.sha256", "SCENE_SUPPORT.json", "SUPPORT_FILES.sha256",
              "whole_body/soma/track2_manifest.json"):
        add(f"data_prep/data_root/{n}", ROOT / n)

    # ---- B'. the opus_v2 root that the f0 build reads (manifest, the 16 geometry-free stage files) and that the GT-free
    #      checks compare with (t3f0_rcheck G9, t3f0_align): its builder, retarget and gates, and its records
    OE = ME / "tmp_analysis/tier3/opus_easy"
    OROOT = ME / "data/human_motion_data_tier3_opus"
    for n in ("t3opus_build_data.py", "t3opus_retarget.sh", "t3opus_rcheck.py", "t3opus_rcheck_v2.py", "t3opus_helper.py",
              "t3opus_driver.sh", "t3opus_driver_v2.sh"):
        add(f"data_prep/opus_root/scripts/{n}", ME / "scripts" / n)
    for n in ("build_files.sha256", "build_files_v2.sha256"):
        add(f"data_prep/opus_root/opus_easy/{n}", OE / n)
    for n in ("build_report.json", "build_check.json"):
        add(f"data_prep/opus_root/opus_easy/build/{n}", OE / "build" / n)
    for p in sorted((OE / "retarget").iterdir()):
        if p.is_file() and (p.name == "parquet_lock.json" or p.name.endswith("_rcheck.json")):
            add(f"data_prep/opus_root/opus_easy/retarget/{p.name}", p)
    for n in ("READY", "SOURCE_FILES.sha256", "whole_body/soma/track2_manifest.json"):
        add(f"data_prep/opus_root/data_root/{n}", OROOT / n)

    # ---- C. Han Zhou's reconstruction (from the verified copies)
    for row in csv.DictReader(open(COPIES), delimiter="\t"):
        if row["ok"] != "True":
            raise SystemExit(f"copy row not ok: {row['source']}")
        rel = Path(row["copy"]).relative_to(HC)
        add(f"han_zhou_reconstruction/{rel}", Path(row["copy"]))

    # ---- D. evaluation, selection, pack
    for n in ("t3final_driver.py", "t3final_kit.py", "make_prereg.py", "launch.sh", "status.sh", "simtest.py",
              "PREREG.md", "PREREG.sha256", "AMENDMENT_1.md", "AMENDMENT_1.sha256", "AMENDMENT_2.md",
              "AMENDMENT_2.sha256", "own_input_runs.csv", "selection.csv", "status.md",
              "FINAL_r.sha256"):
        add(f"evaluation/t3_f0_final/{n}", F / n)
    for sub in ("items", "selection", "integrity"):
        for p in sorted((F / sub).glob("*.json")):
            add(f"evaluation/t3_f0_final/{sub}/{p.name}", p)
    for n in ("build.json", "build.log", "choices.csv", "choices.md", "expect.json", "file.json", "FILES.sha256",
              "pack_status.json", "validate.json", "validate.log"):
        add(f"evaluation/t3_f0_final/pack/FINAL_r/{n}", F / "pack/FINAL_r" / n)
    for p in sorted((F / "pack/FINAL_r/inputs").iterdir()):
        add(f"evaluation/t3_f0_final/pack/FINAL_r/inputs/{p.name}", p)
    for p in sorted((F / "pack/FINAL_r/results").glob("*.json")):
        add(f"evaluation/t3_f0_final/pack/FINAL_r/results/{p.name}", p)
    for n in ("expect.json", "file.json", "pack_status.json", "validate.json", "validate.log"):
        add(f"evaluation/t3_f0_final/pack/FINAL/{n}", F / "pack/FINAL" / n)
    for n in ("PREREG_retry.md", "PREREG_retry.sha256", "PREREG_v3.md", "PREREG_v3.sha256", "RETRY_FILES.sha256",
              "RETRY_FILES_v3.sha256", "RETRY_GO", "RETRY_GO.template", "RETRY_NOGO.template",
              "RETRY_NOGO.superseded_20261031", "AMENDMENT_1_t3_f0_final.md", "t3retry.py", "t3retry_v3.py",
              "launch.sh", "launch_v3.sh", "t3final_r_build.py", "recipes.sha256", "VERDICT_retry.json",
              "SUBMISSION_CHOICE.json", "pids.tsv"):
        add(f"evaluation/f0_retry/{n}", RA / n)
    for sub in ("recipes", "cfgcheck"):
        for p in sorted((RA / sub).glob("*.json")):
            add(f"evaluation/f0_retry/{sub}/{p.name}", p)
    for p in sorted((RA / "dryrun").rglob("*.json")):
        add(f"evaluation/f0_retry/dryrun/{p.relative_to(RA / 'dryrun')}", p)
    for p in sorted((RA / "v3fix").glob("*.py")):
        add(f"evaluation/f0_retry/v3fix/{p.name}", p)
    for p in sorted((RA / "r3fix").glob("*.py")):
        add(f"evaluation/f0_retry/r3fix/{p.name}", p)
    for p in sorted((RA / "r3fix/v1").iterdir()):
        if p.is_file():
            add(f"evaluation/f0_retry/r3fix/v1/{p.name}", p)
    for e in sorted(int(p.parent.name[2:]) for p in (RA / "eval").glob("ep??/post.json")):
        add(f"evaluation/f0_retry/eval/ep{e:02d}/post.json", RA / f"eval/ep{e:02d}/post.json")
    for n in ("PREREG_f0.md", "PREREG_f0.sha256", "VERDICT.json"):
        add(f"evaluation/f0/{n}", F0A / n)
    for e in range(30):
        add(f"evaluation/f0/eval/ep{e:02d}/post.json", F0A / f"eval/ep{e:02d}/post.json")
    return L


def cmd_stage() -> int:
    rows, bad = [], []
    for dst, src in plan():
        if name_forbidden(dst) or name_forbidden(src.name):
            bad.append(f"forbidden name: {dst} <- {src}")
            continue
        if src.suffix.lower() in LFS_EXT:
            bad.append(f"LFS / binary type: {dst}")
            continue
        if not src.is_file():
            bad.append(f"missing: {src}")
            continue
        d = BUNDLE / dst
        d.parent.mkdir(parents=True, exist_ok=True)
        before = sha256(src)
        shutil.copyfile(src, d)
        if sha256(d) != before or sha256(src) != before:
            bad.append(f"copy mismatch: {src}")
        rec = REL / src.relative_to(HC) if str(src).startswith(str(HC) + "/") else src  # Han rows: the original path
        rows.append((dst, ph(str(rec)), before, src.stat().st_size))
    if bad:
        print("\n".join(bad))
        return 1
    # generated: the copy log with placeholders
    cp = BUNDLE / "han_zhou_reconstruction/COPIES_t3ready.tsv"
    cp.write_text(ph(COPIES.read_text()))
    rows.append(("han_zhou_reconstruction/COPIES_t3ready.tsv", "generated: " + ph(str(COPIES)) + " with placeholders",
                 sha256(cp), cp.stat().st_size))
    rows.sort()
    with open(BUNDLE / "SOURCES.tsv", "w") as fp:
        fp.write("bundle_path\tsource\tsha256\tbytes\n")
        for r in rows:
            fp.write("\t".join(map(str, r)) + "\n")
    print(f"stage: {len(rows)} files staged")
    return 0


def cmd_sums() -> int:
    files = sorted(p for p in BUNDLE.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    with open(BUNDLE / "SHA256SUMS", "w") as fp:
        for p in files:
            fp.write(f"{sha256(p)}  {p.relative_to(BUNDLE)}\n")
    print(f"sums: {len(files)} files")
    return 0


SECRET_RES = [re.compile(x, re.I) for x in (
    r"ghp_[A-Za-z0-9]{20,}", r"github_" r"pat_", r"\bhf_[A-Za-z0-9]{20,}", r"\bsk-[A-Za-z0-9]{20,}", r"AKIA[0-9A-Z]{16}",
    r"api[_-]?key\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{20,}", r"WANDB_API_KEY\s*=\s*['\"]?[0-9a-f]{20,}", r"password\s*[=:]",
    r"token\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{16,}", r"network_" r"turbo", r"\b(?:http|https)_proxy\s*=\s*\S",
    r"KAGGLE_" r"KEY", r"kaggle\.json",
    r"BEGIN (?:RSA|OPENSSH) PRIVATE KEY")]
REVIEW_RE = re.compile(r"clean|\bGT\b|ground.?truth|tier.?1|tier_1|multiview", re.I)


def cmd_scan() -> int:
    out, n_files = [], 0
    for p in sorted(BUNDLE.rglob("*")):
        if not p.is_file():
            continue
        n_files += 1
        rel = str(p.relative_to(BUNDLE))
        if name_forbidden(rel):
            out.append(f"NAME   {rel}")
        if p.suffix.lower() in LFS_EXT:
            out.append(f"LFS    {rel}")
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            out.append(f"BINARY {rel}")
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for r in SECRET_RES:
                if r.search(line):
                    out.append(f"SECRET {rel}:{i}: pattern {r.pattern!r}: {re.sub(r'[0-9]', '#', line.strip())[:160]}")
    review = {}
    for p in sorted(BUNDLE.rglob("*")):
        if not p.is_file() or "han_zhou_reconstruction" in str(p):
            continue
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            continue
        hits = [i for i, line in enumerate(text.splitlines(), 1) if REVIEW_RE.search(line)]
        if hits:
            review[str(p.relative_to(BUNDLE))] = len(hits)
    W.mkdir(parents=True, exist_ok=True)
    rep = {"at": now(), "files": n_files, "problems": out,
           "checks": ["file names: no clean / GT / LOCAL marker (" + ", ".join(FORBIDDEN) + ", GT)",
                      "no Git-LFS / binary type (" + " ".join(sorted(LFS_EXT)) + ")",
                      "no credential / token / proxy-setting pattern (" + str(len(SECRET_RES)) + " patterns)"],
           "lines_naming_clean_gt_tier1_per_file": review,
           "review_note": "per-file counts of lines that name clean / GT / Tier-1 material (rule text, guards, file names); "
                          "reviewed with every digit masked; Han Zhou's copied documents are excluded from this count "
                          "(his READMEs carry GT-based evaluation tables, copied verbatim, quoted nowhere)"}
    write_json(W / "scan.json", rep)
    write_json(BUNDLE / "verification/hygiene_scan.json", rep)
    print("\n".join(out) if out else f"scan: {n_files} files, no forbidden name, no LFS type, no secret pattern")
    print(f"files with lines naming clean / GT / Tier 1 (counts only; see scan.json): {len(review)}")
    return 1 if out else 0


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("copy", "set", "stage", "sums", "scan"):
        print(__doc__)
        return 2
    return {"copy": cmd_copy, "set": cmd_set, "stage": cmd_stage, "sums": cmd_sums, "scan": cmd_scan}[sys.argv[1]]()


if __name__ == "__main__":
    sys.exit(main())
