"""make_prereg.py: write PREREG.md + PREREG.sha256 of fc_runs/submissions/t3_f0_final from the files as they are now.

Refuses to run once any item, run or selection exists (the protocol must be fixed before any result). A re-freeze before
results (e.g. after a dry-run fix) keeps the previous file as PREREG_superseded_<stamp>.md and says so in the new one.
  <flashchord python> -I -B make_prereg.py [--refreeze "<reason>"]
"""
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
ROOT = ME / "fc_runs/submissions/t3_f0_final"
KIT = ME / "v2d_submission_kit_src/v2d_submission_kit"
STOCK_REPO = ME / "code/video_to_data_8c778f9"
CODE = ME / "code/fc_r2"


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def head(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                          env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).stdout.strip()


def results_exist() -> list[str]:
    out = []
    for sub in ("items", "runs", "selection", "pack"):
        d = ROOT / sub
        if d.is_dir() and any(d.iterdir()):
            out.append(sub)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refreeze", default="", help="reason for replacing an existing PREREG before any result")
    a = ap.parse_args()
    if results_exist():
        print(f"refusing: results exist already ({results_exist()}); write an amendment instead")
        return 1
    prereg, prereg_sha = ROOT / "PREREG.md", ROOT / "PREREG.sha256"
    superseded = ""
    if prereg.exists():
        if not a.refreeze:
            print("PREREG.md exists: pass --refreeze '<reason>' (allowed only before any result)")
            return 1
        old = ROOT / f"PREREG_superseded_{dt.datetime.now():%Y%m%d_%H%M%S}.md"
        os.replace(prereg, old)
        prereg_sha.unlink(missing_ok=True)
        superseded = (f"\n**Re-frozen before any result.** Previous version kept as `{old.name}` (sha256 {sha(old)}). "
                      f"Reason: {a.refreeze}\n")
    f = {
        "t3final_driver.py": ROOT / "t3final_driver.py", "t3final_kit.py": ROOT / "t3final_kit.py",
        "make_prereg.py": ROOT / "make_prereg.py", "launch.sh": ROOT / "launch.sh", "status.sh": ROOT / "status.sh",
        "kit examples/run_policy_evaluation.py": KIT / "examples/run_policy_evaluation.py",
        "kit flash_chord/scripts/evaluate_policy.py": KIT / "flash_chord/scripts/evaluate_policy.py",
        "stock scripts/evaluate_policy.py": STOCK_REPO / "robotic_grounding/flash_chord/scripts/evaluate_policy.py",
        "kit tools/pack_submission.py": KIT / "tools/pack_submission.py",
        "kit submit/v2d_submit.py": KIT / "submit/v2d_submit.py",
        "kit KIT_MANIFEST.json": KIT / "KIT_MANIFEST.json", "kit config/tracks.json": KIT / "config/tracks.json",
        "kit metric_code/track_2/AUC.py": KIT / "metric_code/track_2/AUC.py",
        "kit metric_code/track_2/SP-SR.py": KIT / "metric_code/track_2/SP-SR.py",
        "kit metric_code/track_2/MP-SR.py": KIT / "metric_code/track_2/MP-SR.py",
        "kit metric_code/track_2/MPPE.py": KIT / "metric_code/track_2/MPPE.py",
        "code/fc_r2 scripts/track2/eval_all.py": CODE / "scripts/track2/eval_all.py",
        "code/fc_r2 coordinate/export.py": CODE / "src/flash_chord/training/ppo/coordinate/export.py",
        "tmp_analysis/tier3/f0/PREREG_f0.md": ME / "tmp_analysis/tier3/f0/PREREG_f0.md",
        "tmp_analysis/tier3/opus_easy/PREREG.md": ME / "tmp_analysis/tier3/opus_easy/PREREG.md",
        "tmp_analysis/tier3/opus_easy/PREREG_wave3.md": ME / "tmp_analysis/tier3/opus_easy/PREREG_wave3.md",
        "data/human_motion_data_tier3_f0/READY": ME / "data/human_motion_data_tier3_f0/READY",
        "data/human_motion_data_tier3_f0/FILES.sha256": ME / "data/human_motion_data_tier3_f0/FILES.sha256",
        "f0 track2_manifest.json": ME / "data/human_motion_data_tier3_f0/whole_body/soma/track2_manifest.json",
        "tmp_analysis/tier3/f0/retarget/parquet_lock.json": ME / "tmp_analysis/tier3/f0/retarget/parquet_lock.json",
        "data/human_motion_data_tier3_opus/READY": ME / "data/human_motion_data_tier3_opus/READY",
        "opus track2_manifest.json": ME / "data/human_motion_data_tier3_opus/whole_body/soma/track2_manifest.json",
        "scripts/t3f0_helper.py (post step, running)": ME / "scripts/t3f0_helper.py",
        "scripts/t3f0_driver.sh (running)": ME / "scripts/t3f0_driver.sh",
    }  # fmt: skip
    h = {k: sha(p) for k, p in f.items()}
    stock_head, code_head = head(STOCK_REPO), head(CODE)
    written = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    pins = "\n".join(f"| {k} | `{v}` |" for k, v in h.items())
    text = f"""# PREREG: Tier-3 f0 FINAL (5x own-input official evaluation, selection, pack)

Written {written} CST by `make_prereg.py`, before any evaluation, item, run or selection of this protocol exists.
Folder `fc_runs/submissions/t3_f0_final`. Nothing here is uploaded or pushed.
{superseded}
**Authorization (user, 2026-10-26 ~20:35 CST, verbatim).** "i think it is currently fine to remove tier2 from training
queue since zuxing is fully improve it, i think we need to take care of tier3 right now"

Read as: Tier 3 is the top priority, so build its submission pipeline now. The main session decides when to launch.
Upload, push and the cited commit each need the user's explicit OK.

## Purpose

Build the Tier-3 submission file from the f0 test (`tmp_analysis/tier3/f0/PREREG_f0.md`, sha256
`{h['tmp_analysis/tier3/f0/PREREG_f0.md']}`) under a rule fixed here:
- each episode's eval file is the one PREREG_f0 fixes;
- that file is evaluated 5 times against its own input;
- per episode, the rollout closest to its own 5-run mean is packed.

Reported: the own-input per-run values, mean and spread. LOCAL only: clean scores (report, never a decision).

## 1. Items: which file per episode (= PREREG_f0 "Use in a submission", applied mechanically)

For each episode 00..29, the driver resolves an item once and freezes it (`items/epNN.json`, write-once):
- **f0**: the t3f0 post step is terminal with an evaluated checkpoint: `eval/epNN/post.json` has status `done` or
  `done_score_failed`, eval_status `evaluated` and eval_exit 0. The eval file is post.json `evaluated_checkpoint`, the
  file the post step evaluated (export of `final.json`'s checkpoint at 1e-4, or 2e-4 if recorded).
- **opus_v2**: only when the f0 clip has no evaluated checkpoint (post status `export_refused` / `failed`, or a
  `CRASHED_epNN` marker) AND the opus_v2 post step (`tmp_analysis/tier3/opus_easy/eval/epNN/post.json`) has one. That
  item uses its own opus input, manifest and `frame_references_opus`.
- **none**: neither. The episode has no policy. Then no complete FINAL file can be packed; the user decides (addendum).
- A FAIL by PREREG_f0's pass rule still has an evaluated checkpoint, so it stays f0.
- Clips still training or in their post step are pending. They are picked up automatically every 60 s until all 30
  are settled. Nothing is chosen by any number: only post-step status fields and file checks decide.

**Verification before freezing** (any failure = BLOCKED: logged, not frozen, needs a human):
- the eval file's sha256 must equal post.json, the eval_report.json row, the post step's results JSON and the
  export_report `destination_sha256`;
- the export must be ok, stock-runtime verified (8c778f9, eval_code b1b3ebe3, help scale 0), with max_rel_diff within the
  recorded tolerance;
- the export source must be the run's `final.json` checkpoint (path and sha256);
- the reference parquet must be the retarget-locked f0 parquet (`parquet_lock.json`, path and sha256) of the manifest
  sequence; for opus, the post.json parquet sha256;
- the frame reference file must exist (f0: covered by the write-once root's FILES.sha256).

## 2. Evaluation protocol (5 runs per item)

Run k = 1..5 of an item = one official evaluation of the pinned eval file against its own input:
- `code/fc_r2` (HEAD {code_head}) `scripts/track2/eval_all.py` with `--export no --world-count 1`, the item's own
  `--manifest` and `--iteration`, `HUMAN_MOTION_DATA_DIR` = the item's data root;
- then the kit `examples/run_policy_evaluation.py`, then the kit `flash_chord/scripts/evaluate_policy.py` on the untouched
  stock runtime `code/video_to_data_8c778f9` (HEAD {stock_head}): frame 0, explicit reset, no help;
- the work dir `runs/epNN/rK/tryT/epNN/model_<it>.safetensors` is a symlink to the pinned file.

Scheduling (performance only, never a decision):
- item-major order: episode index, then run;
- GPUs 0,1,2,3,5,6,7, never 4; no MPS GPU; at least 4096+1536 MiB free;
- each eval holds our lane lock plus every existing shared eval-lane lock of its GPU (rolling start, tier2v2, t3f0,
  opus, k3*, t1_best20; read-only, all-or-nothing);
- at most `--lane-max` (2) evaluations of anybody per NUMA eval-core lane; at most `--max-parallel` (3) of ours;
- a 35 s cooldown per GPU after each of our evals;
- `nice -n 10 taskset -c <lane cores>`.

**An evaluation is ok only if all of these hold:**
- exit 0, and eval_all reports "evaluated";
- the results JSON and parquet exist;
- JSON and parquet `checkpoint_sha256` equal the pinned sha256, and the JSON names the pinned file;
- the JSON's reference parquet is the item's own input (path, and the sha256 is unchanged);
- start_frame 0, motion_start_frame 0, world_count 1, reset_mode explicit, no non-finite world;
- the parquet's eval_script (160d24f5) and eval_code (b1b3ebe3) digests match;
- environment_valid holds on every row;
- the pinned file is unchanged afterwards.

A technical failure is retried (at most 3 tries, 5 min apart). A completed evaluation is never re-run, whatever its
score. After 3 failed tries, the run is final as failed.

## 3. Selection rule (fixed here, before any result)

As soon as an item's 5 runs are final, `selection/epNN.json` is frozen (write-once):
- **chosen run** = among the ok runs, the one whose own-input ADD-AUC (results JSON `metrics.add_auc`, the evaluator's
  number against the item's own input) is closest to the mean of those ok runs' own-input ADD-AUC, computed exactly
  (rational arithmetic on the float64 values, so ties are exact); ties go to the lower run index;
- if no run is ok: run 0 = the post step's own official evaluation of the same file (already verified at freezing).

Nothing else enters the choice: no clean number, no SP/MP/MPPE, no other run's file. The post step's evaluation is
reported next to the 5 runs, but it is not a candidate except in the no-ok-run case.

## 4. Pack (after every episode is settled; also `pack` by hand)

- **FINAL**: per episode, the chosen run's results JSON and parquet (byte copies, sha256 verified). This is the only
  file eligible for upload.
- **T1..T5**: run k of every episode, for the report only. A T_k with a failed run is not packed.
- Kit packer `tools/pack_submission.py --track track_2_tier3`, unchanged, run as `v2dkit python -I -B`:
  - `--frame-references` points to a per-file folder of byte copies of each episode's own frame reference
    (f0: `frame_references_f0`; opus: `frame_references_opus`);
  - `--dataset-root data/video_to_data_challenge`, `--eval-script` = the kit evaluate_policy.py;
  - commit URL = placeholder `{'https://github.com/local-placeholder/not-uploaded/commit/' + '0' * 40}`.
- Required:
  - packer exit 0;
  - input-reference residual < 1e-4 m and < 1e-3 rad, so the frame conversion reproduces the policy input;
  - `submit/v2d_submit.py validate_submission()` (offline) passes;
  - provenance holds: uniform evaluator, packer and uploader digests and commit URL, and per episode the checkpoint
    digest equals the item's eval file.
  - The pose-columns sha256 is recorded.
- `repack --commit-url <real /commit/ URL>` re-packs FINAL into `upload/<sha7>/`. The pose columns must equal the LOCAL
  FINAL's. The result is an upload candidate only; the user decides.

## 5. The Tier-3 no-clean rule, and what is report-only

- No decision here (item, run, retry, selection, file, order) uses Tier-1 or clean data or a clean score.
  - The driver reads decision inputs only through a guard that refuses any path naming clean,
    LOCAL_NOT_FOR_UPLOAD, tier_1_multiview_caption or track_2/baseline.
  - Items are frozen from post-step status fields and file checks. Selections are frozen from own-input ADD-AUC only.
- **Clean numbers come from two places:**
  - the kit packer's own cross-check ("locally computed", in `pack_LOCAL_NOT_FOR_UPLOAD.log`);
  - the official metric_code (`t3final_kit.py score` -> `metric_code_LOCAL_NOT_FOR_UPLOAD.json`,
    `summary_LOCAL_NOT_FOR_UPLOAD.*`).
  - Both run only after the selections are frozen. They are report-only and are read by no step of this pipeline.
- **Report-only (noisy side):** own-input AUC, SP-SR, MP-SR, MPPE and CHORD SR per run; the 5-run mean and spread; the
  post step's run 0; GPU and timing; the eval file's resolved-config facts (scene.support, palm threshold, seeds).

## 6. Disclosures

1. **The operator had already seen the clean diagnostics.** Before this protocol was written, the operator (main
   session) had seen the LOCAL clean diagnostics of the 11 clips whose post step was done (00-08, 10, 12). This
   protocol makes no per-clip choice:
   - the item rule is PREREG_f0's, fixed on 2026-10-24 before any f0 training;
   - the number of runs is fixed;
   - the selection is a mechanical function of own-input ADD-AUC;
   - there is no retry list, no recipe change and no order here that depends on any number.

   Any later retry list for failing clips must be derived mechanically from Tier-3-only signals (own-input SP/MP floor,
   help-zero, training stats) under its own PREREG that repeats this disclosure. The agent that built this protocol
   opened no clean file.
2. **Grey areas still to disclose.** PREREG_f0's grey areas still apply to any submission:
   - the frame-0 object pose (user-confirmed allowed, FAQ);
   - supports from Han's video fit (task text as input) plus our four placement rules;
   - the s1 human with the moved object (mitigation by config only);
   - our frame references;
   - Han's GT-based README table, which was read but chose nothing.
3. **Where the run-to-run spread comes from.** The 5 runs differ only by GPU floating-point non-determinism: the
   official path has no selectable seed (t1_best20 audit). Closest-to-mean is used so that the packed rollout reflects
   what an organiser rerun (rule 2.9) is expected to give, not a best-of-5.
4. **Opus fallback episodes.** These would be trained on an earlier reconstruction (opus_v2), as PREREG_f0 already
   allows. Their row is packed with their own frame reference. A mixed f0 / opus file must say so in the submission
   note.

## 7. Failure handling

- **BLOCKED item:** never frozen; a human decides (addendum).
- **No-policy episode ("none"):** the FINAL is not packed (`pack` exits 1 and says why); the user decides (addendum).
- **Kit or stock runtime changed during the run:** the integrity check (file list, sizes, mtimes) warns; the run is
  then reviewed before packing.
- **Stop:** `touch STOP`. Running evals finish and are recorded. Remove STOP and relaunch to resume (`recover`
  re-attaches to or records orphaned evals and never kills them).

## 8. Out of scope here

- the upload, the push, and the content of the cited commit (see `COMMIT_REQUIREMENTS.md`, not frozen);
- any retrain or retry of failing clips;
- an interim safety-net file, which would need every episode to have a rollout. That is impossible before every clip
  has finished training, short of a separately pre-registered intermediate-checkpoint rule.

## 9. Amendments

Any change to the driver, the kit helper, the launch / status scripts or this text needs a new frozen version:
- before any result: `make_prereg.py --refreeze`, which keeps the old file;
- after results: `AMENDMENT_N.md` with its own sha256, the prior results kept, and the change disclosed.

The driver refuses to run unless `PREREG.sha256` matches this file and this file names the current driver and kit
helper sha256 (below).

## 10. Pins (sha256 at freezing)

| file | sha256 |
|---|---|
{pins}

Code: `code/fc_r2` HEAD `{code_head}` (training and export code of every f0 / opus run); stock runtime HEAD
`{stock_head}`.

## 11. Commands

- start / resume: `bash fc_runs/submissions/t3_f0_final/launch.sh`
- status: `bash fc_runs/submissions/t3_f0_final/status.sh`
- stop: `touch fc_runs/submissions/t3_f0_final/STOP`
- by hand (also run automatically at the end): `python -B t3final_driver.py pack`
- `python -B t3final_driver.py repack --commit-url <URL>` (after the user's OK)
"""
    prereg.write_text(text)
    digest = sha(prereg)
    prereg_sha.write_text(f"{digest}  PREREG.md\n")
    os.chmod(prereg, 0o444)
    os.chmod(prereg_sha, 0o444)
    print(f"PREREG.md sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
