#!/usr/bin/env python3
"""build_t1ready.py: write the data files of t1best/t1ready_20261031/ into the v2d_t1ready worktree.

The folder records the Tier-1 set `all7` (TOPUP_AMEND1) and its submittable trial 05: for each of the 30 episodes,
the training checkpoint, every training stage (overrides, code, hull cache), the export, and the trial-05 inputs.
Reads only (fc_runs, the worktree's committed item configs); writes only under DEST. Never runs training/eval.

  <flashchord python> -I -B build_t1ready.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shlex  # noqa: E402
import shutil  # noqa: E402
import struct  # noqa: E402
from pathlib import Path  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
FC = ME / "fc_runs"
T = FC / "T1improve"
U = T / "topup"
A = U / "amend1"
C = T / "confirm"
B20 = FC / "submissions/t1_best20_20261026"
REPO = ME / "code/v2d_t1ready"
T1 = REPO / "robotic_grounding/flash_chord/scripts/track2/t1best"
DEST = T1 / "t1ready_20261031"
MOTION = str(ME / "data/human_motion_data")
RUNS = str(FC)
SCRIPTS = ME / "scripts"

# episode -> (arm, selected iteration, base item whose stages precede the arm's fork)
CHANGED = {1: ("ep01-Bc", 162000, "ep01"), 5: ("ep05-B3", 62000, "ep05"), 10: ("ep10-BT", 58000, "ep10"),
           14: ("ep14-R", 92000, "ep14"), 15: ("ep15-B2", 40000, "ep15"), 19: ("ep19-Bc", 135000, "ep19"),
           28: ("ep28-W3H", 83000, "ep28alt")}
TREE_CODE = {"b39b69b": "b39b69be6c2f", "7fbc227": "7fbc2273a92b", "61fa5d3": "61fa5d31ff80 (flash_chord = f2923d61817e)"}
SOURCE_CSV = A / "files/trial05_SUBMITTABLE_W3H_closest_to_mean/T1improve_W3H_tier1_trial05_SUBMITTABLE_closest_to_mean.csv"
SOURCE_SHA = "53cfe9bfa5a2487f3502035e535ea5fabcbff0cab0a578fbea514108aa0f3f77"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ph(s: str) -> str:
    """Replace this machine's roots by the placeholders used in configs/items (<MOTION>, <RUNS>)."""
    return s.replace(MOTION, "<MOTION>").replace(RUNS, "<RUNS>").replace(str(ME), "<ME>")


def meta(path: Path) -> dict:
    with open(path, "rb") as fp:
        n = struct.unpack("<Q", fp.read(8))[0]
        return json.loads(fp.read(n)).get("__metadata__", {}) or {}


def load(p: Path):
    return json.loads(p.read_text())


def main() -> int:
    if DEST.exists():
        raise SystemExit(f"{DEST} exists; not overwriting")
    items = {p.stem: load(p) for p in sorted((T1 / "configs/items").glob("*.json"))}
    with (T1 / "best_set/manifest.tsv").open() as fp:
        b20man = {r["item"]: r for r in csv.DictReader(fp, delimiter="\t")}
    score = load(U / "scores/all7/t05/score.json")
    mc = load(A / "files/trial05_SUBMITTABLE_W3H_closest_to_mean/metric_code.json")
    dec = load(U / "decisions.json")["items"]
    dec_by_ep = {v["episode"]: v for v in dec.values()}
    if sha256(SOURCE_CSV) != SOURCE_SHA:
        raise SystemExit("source CSV sha256 changed")

    episodes, problems = [], []
    for e in range(30):
        src = score["sources"][str(e)]
        if e in CHANGED:
            arm, it, base = CHANGED[e]
            ad = T / "arms" / arm
            fork, launch = load(ad / "fork.json"), load(ad / "launch.json")
            run_dir = sorted((ad / "runs").iterdir())[0]
            final = load(run_dir / "final.json")
            exp = load(C / "exports" / f"{arm}_{it}" / "export.json")
            rep = load(C / "exports" / f"{arm}_{it}" / "try1/export_report.json")
            argv = launch["argv"]
            ovs = [ph(o) for o in argv[argv.index("scripts/train_rl.py") + 1:]]
            base_cfg = items[base]
            stages = [dict(s, lineage_of=base) for s in base_cfg["stages"]]
            last = stages[-1]
            if last["run"] != fork["src"]:
                problems.append(f"ep{e:02d}: fork source {fork['src']} is not the base item's last stage {last['run']}")
            fp_ = fork["fork_pair"]
            stage = {
                "run": str(run_dir.relative_to(FC)),
                "start": f"resume model_{fork['fork']} (fork pair T1improve/arms/{arm}/fork = copy of {fork['src']}/model_{fork['fork']})",
                "end": f"final.json {final['checkpoint'].replace('.safetensors', '')} (max_iterations); selected model_{it}",
                "code": TREE_CODE[fork["tree"]],
                "hull_cache": f"seeded copy T1improve/arms/{arm}/fc_cache ({fork['hull_before']})",
                "overrides": ovs,
                "source": f"t1improve/arms/{arm}/launch.json (argv) and overrides.yaml",
                "arm": arm,
                "launch_overrides": fork["overrides"],
                "inherited_overrides": fork["inherited_overrides"],
                "added_overrides": fork["added_overrides"],
                "fork_iteration": fork["fork"],
                "max_iterations": fork["max_iterations"],
                "fork_pair": {k: {"sha256": v["sha256"], "size": v["size"], "path": ph(v["dest"]), "source": ph(v["source"])}
                              for k, v in fp_.items()},
                "env": {k: ph(v) for k, v in launch["env_set"].items()
                        if k in ("FLASH_CHORD_CACHE_DIR", "HUMAN_MOTION_DATA_DIR", "PYTHONPATH", "OMP_NUM_THREADS")},
            }
            stages.append(stage)
            rsi = run_dir / f"rsi_{it}.safetensors"
            ck = {"run": stage["run"], "iteration": it, "model": f"model_{it}.safetensors", "model_sha256": exp["training_sha256"],
                  "model_size": (run_dir / f"model_{it}.safetensors").stat().st_size,
                  "rsi": rsi.name, "rsi_sha256": sha256(rsi) if rsi.is_file() else None,
                  "rsi_size": rsi.stat().st_size if rsi.is_file() else None,
                  "trained_by": TREE_CODE[fork["tree"]], "hull_cache": stage["hull_cache"]}
            ex = {"root": "RUNS", "file": f"T1improve/confirm/exports/{arm}_{it}/try1/model_{it}.safetensors",
                  "sha256": exp["sha256"], "size": (C / "exports" / f"{arm}_{it}/try1/model_{it}.safetensors").stat().st_size,
                  "max_rel_diff": exp["max_rel_diff"], "exported_by": TREE_CODE[fork["tree"]], "tolerance": exp["tolerance"],
                  "command": ph(exp["command"])}
            d = dec_by_ep[e]
            ev = {"trials20": {"n": d["n_new"], "auc_mean": d["new"]["AUC"]["mean"], "auc_sd": d["new"]["AUC"]["sd"],
                               "auc_min": d["new"]["AUC"]["min"], "auc_max": d["new"]["AUC"]["max"],
                               "sp_pass": d["sp_pass_new"], "mp_pass": d["mp_pass_new"], "mppe_mean_cm": d["new"]["MPPE"]["mean"]},
                  "decision": {"keep_new_8_5": d["keep_new"], "why": d.get("why")}}
            ent = {"episode": e, "object": base_cfg["object"], "item": f"{arm}_{it}", "origin": f"T1improve arm {arm}",
                   "replaces": f"{base} ({base_cfg['source']})", "sequence_id": base_cfg["sequence_id"],
                   "scene_support": fork["scene_support"], "experiment": fork["experiment"], "checkpoint": ck,
                   "export": ex, "evaluation": ev, "stages": stages}
            if rep.get("source_sha256") != ck["model_sha256"] or rep.get("destination_sha256") != ex["sha256"]:
                problems.append(f"ep{e:02d}: export report digests differ")
        else:
            item = f"ep{e:02d}"
            cfg = items[item]
            if not cfg["in_set"]:
                problems.append(f"{item} not in the 2026-10-26 set")
            p, x = cfg["pinned"], cfg["export"]
            ck = {**p}
            ex = {"root": "T1_EXPORTS", **x}
            r = b20man[item]
            ent = {"episode": e, "object": cfg["object"], "item": item, "origin": f"t1best set (2026-10-26): {cfg['source']}",
                   "replaces": None, "sequence_id": cfg["sequence_id"], "scene_support": cfg["scene_support"],
                   "experiment": cfg["experiment"], "checkpoint": ck, "export": ex, "evaluation": cfg["evaluation"],
                   "stages": cfg["stages"]}
            if r["export_sha256"] != x["sha256"]:
                problems.append(f"{item}: best_set manifest export differs")
        if ent["export"]["sha256"] != src["eval_file_sha256"]:
            problems.append(f"ep{e:02d}: export sha {ent['export']['sha256'][:12]} != trial-05 evaluated file {src['eval_file_sha256'][:12]}")
        pe = mc["per_episode_float32"][str(e)]
        ent["trial05"] = {"kind": src["kind"], "name": src["name"], "evaluated_file_sha256": src["eval_file_sha256"],
                          "results_parquet": ph(src["results_parquet"]), "results_parquet_sha256": score["inputs"][str(e)],
                          "seed_signature": src.get("seed_signature"), "evaluator_json_metrics": src.get("json_metrics"),
                          "metric_code_float32": pe}
        ent["n_stages"] = len(ent["stages"])
        episodes.append(ent)
    if problems:
        raise SystemExit("problems: " + "; ".join(problems))

    DEST.mkdir(parents=True)
    out = {"set": "all7", "trial": 5,
           "amendment": "amend1/TOPUP_AMEND1.md (sha256 76143fce2e81bbd2699220b8e158c3aee136516dc56baf029ba5fe0c1c09b843)",
           "placeholders": {"<MOTION>": "derived motion data root (HUMAN_MOTION_DATA_DIR)",
                            "<RUNS>": "run root of the original machine (fc_runs)", "<ME>": "work root of the original machine"},
           "export_roots": {"T1_EXPORTS": "the t1best payload exports/ (README_NEW_MACHINE.md, group B_exports)",
                            "RUNS": "<RUNS> (T1improve confirmation exports)"},
           "episodes": episodes}
    (DEST / "set_items.json").write_text(json.dumps(out, indent=1) + "\n")

    with (DEST / "set_stages.tsv").open("w", newline="") as fp:
        w = csv.writer(fp, delimiter="\t", lineterminator="\n")
        w.writerow(["episode", "item", "stage", "run", "start", "end", "code", "hull_cache", "trained_the_checkpoint", "overrides"])
        for ent in episodes:
            for i, s in enumerate(ent["stages"], 1):
                w.writerow([ent["episode"], ent["item"], i, s["run"], s["start"], s["end"], s["code"], s["hull_cache"],
                            "yes" if i == len(ent["stages"]) else "", shlex.join(s["overrides"])])

    cols = ["episode", "object", "item", "origin", "training_run", "iteration", "training_checkpoint_sha256", "trained_by",
            "n_stages", "evaluated_file_sha256", "exported_by", "export_max_rel_diff", "trials20_auc_mean", "trials20_mppe_mean_cm",
            "trials20_sp_pass", "trials20_mp_pass", "trial05_auc", "trial05_sp", "trial05_mp", "trial05_mppe_cm"]
    with (DEST / "set_manifest.tsv").open("w", newline="") as fp:
        w = csv.writer(fp, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for ent in episodes:
            t20 = ent["evaluation"].get("trials20", {})
            pe = ent["trial05"]["metric_code_float32"]
            w.writerow([ent["episode"], ent["object"], ent["item"], ent["origin"], ent["checkpoint"]["run"], ent["checkpoint"]["iteration"],
                        ent["checkpoint"]["model_sha256"], ent["checkpoint"]["trained_by"], ent["n_stages"], ent["export"]["sha256"],
                        ent["export"]["exported_by"], ent["export"].get("max_rel_diff"),
                        f"{t20.get('auc_mean'):.4f}", f"{t20.get('mppe_mean_cm'):.3f}", t20.get("sp_pass"), t20.get("mp_pass"),
                        f"{pe['AUC']:.6f}", pe["SP-SR"], pe["MP-SR"], f"{pe['MPPE']:.4f}"])

    # records of the amendment and of the source file (copied verbatim)
    am = DEST / "amend1"
    (am / "files_trial05").mkdir(parents=True)
    for f in ("TOPUP_AMEND1.md", "TOPUP_AMEND1.sha256", "amend1_build.py", "selection.json", "files.json", "amend1.log"):
        shutil.copy2(A / f, am / f)
    for f in ("pack.log", "metric_code.json", "validate.json", "validate.log", "expected_checkpoints.json", "score.log"):
        shutil.copy2(A / "files/trial05_SUBMITTABLE_W3H_closest_to_mean" / f, am / "files_trial05" / f)
    (am / "verify").mkdir()
    shutil.copy2(A / "verifyA/rescore_trial05.json", am / "verify/verifyA_rescore_trial05.json")
    for f in ("verifyB.py", "verifyB_report.json", "verifyB.log", "git_check.log"):
        shutil.copy2(A / "verifyB" / f, am / "verify" / f)
    tr = DEST / "trial05"
    tr.mkdir()
    for f in ("score.json", "metric_code.json", "pack.log", "score.log"):
        if (U / "scores/all7/t05" / f).is_file():
            shutil.copy2(U / "scores/all7/t05" / f, tr / f)
    sv = DEST / "server_scripts"
    sv.mkdir()
    for f in ("t1improve_launch.py", "t1improve_launch.sh"):
        shutil.copy2(SCRIPTS / f, sv / f)
    print(f"wrote {DEST}: {len(episodes)} episodes, {sum(e['n_stages'] for e in episodes)} stages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
