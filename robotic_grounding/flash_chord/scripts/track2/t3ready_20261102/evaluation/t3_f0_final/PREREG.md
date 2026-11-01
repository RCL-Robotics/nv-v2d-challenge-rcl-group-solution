# PREREG: Tier-3 f0 FINAL (5x own-input official evaluation, selection, pack)

Written 2026-10-26 21:02 CST by `make_prereg.py`, before any evaluation, item, run or selection of this protocol exists.
Folder `fc_runs/submissions/t3_f0_final`. Nothing here is uploaded or pushed.

**Authorization (user, 2026-10-26 ~20:35 CST, verbatim).** "i think it is currently fine to remove tier2 from training
queue since zuxing is fully improve it, i think we need to take care of tier3 right now"

Read as: Tier 3 is the top priority, so build its submission pipeline now. The main session decides when to launch.
Upload, push and the cited commit each need the user's explicit OK.

## Purpose

Build the Tier-3 submission file from the f0 test (`tmp_analysis/tier3/f0/PREREG_f0.md`, sha256
`82702d85fd119d0ad5165dd8c2d890c87e700906ebffcdb21c56c024e1e8eeec`) under a rule fixed here:
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
- `code/fc_r2` (HEAD 61fa5d31ff8054bf1e5cf5375ef0957e54d2729b) `scripts/track2/eval_all.py` with `--export no --world-count 1`, the item's own
  `--manifest` and `--iteration`, `HUMAN_MOTION_DATA_DIR` = the item's data root;
- then the kit `examples/run_policy_evaluation.py`, then the kit `flash_chord/scripts/evaluate_policy.py` on the untouched
  stock runtime `code/video_to_data_8c778f9` (HEAD 8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef): frame 0, explicit reset, no help;
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
  - commit URL = placeholder `https://github.com/local-placeholder/not-uploaded/commit/0000000000000000000000000000000000000000`.
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
| t3final_driver.py | `22c26adfc44727fa5ab29e1671229ef350c7d1c26a293f97c9f9baeac83b1e08` |
| t3final_kit.py | `1c8f94b96a05934f71a3893d2910702e1e5fc186248fcc4f0ab4ad2451beebc7` |
| make_prereg.py | `758ea69857ccee9cbae17df08c92b92cde6d1c932fe06693d769f49dcbd04b42` |
| launch.sh | `014ad481a3e7996c574766e7c5f2dfdec2e139c0a05e86d838876a2d856bb3c4` |
| status.sh | `bd7d9c50ad93e46973d3233d800c3437d7a3ff729b227a0d4b9e297e2b22bc96` |
| kit examples/run_policy_evaluation.py | `0bcc41f5de246a51f194409b3e5292e04109012cb22a433a8f7ed477df04e55d` |
| kit flash_chord/scripts/evaluate_policy.py | `160d24f5fbeb70ccd569ca4192056f1892b2a9f2999066db9d8279461648fdf1` |
| stock scripts/evaluate_policy.py | `160d24f5fbeb70ccd569ca4192056f1892b2a9f2999066db9d8279461648fdf1` |
| kit tools/pack_submission.py | `47a4589574ff8284ba4376ef31ac5548c8fe87ead753be2d3b43d6c87041e5be` |
| kit submit/v2d_submit.py | `fe5cb1dc86d12abd01eab9dd5dd98b51569b5d8513a2d00d4e79da42a5f11718` |
| kit KIT_MANIFEST.json | `398947a7783490245a47d70ec68d6894ecef7636e712532710ec448dc0ca1c3e` |
| kit config/tracks.json | `5799f424a4cd5453f8a6cfceaeff76e7cc7f8a78ba7d22132108827b1c2a09c7` |
| kit metric_code/track_2/AUC.py | `58b437697c32c86497ff5a306fe586014c3429b0d2c18e1d3094a6f503479339` |
| kit metric_code/track_2/SP-SR.py | `d206042872acd1c1de6120f96966272495fbbd71a50315f49f5192fea1aaa5bf` |
| kit metric_code/track_2/MP-SR.py | `a1d72216eb7745a13ed7db6a55b9efb31492b35149113181a6c42ffc231fff11` |
| kit metric_code/track_2/MPPE.py | `41815bc1e8870fb377036750976dcf135818e029f06096ccef5177090fe5e033` |
| code/fc_r2 scripts/track2/eval_all.py | `657295921e820a464160039ad9a65ba92a683108a99dcb197290ae4147363197` |
| code/fc_r2 coordinate/export.py | `34e3d7f3a7dca4fdd1e7c4d26750c6332abcc1d629f6a9c6c7a965833a22206f` |
| tmp_analysis/tier3/f0/PREREG_f0.md | `82702d85fd119d0ad5165dd8c2d890c87e700906ebffcdb21c56c024e1e8eeec` |
| tmp_analysis/tier3/opus_easy/PREREG.md | `c7a19218eed2f7548e4c9b04b9d76649adcaf58223cd37a9a95ccbde7b86dfa4` |
| tmp_analysis/tier3/opus_easy/PREREG_wave3.md | `2157329f03a5c1c4ee823f48b9848f3e26a4cb711808628243c06e19871cec9e` |
| data/human_motion_data_tier3_f0/READY | `ee58a62deb3a0bb06248d221e1184de751f32c171410fa2616f057c4b3b2e50b` |
| data/human_motion_data_tier3_f0/FILES.sha256 | `f9c94e8550e03223c96e101e155038537efeedda3b5213d96cecc4dcbafaba49` |
| f0 track2_manifest.json | `d496584c98cb4e811c41a12b0afa5992ff02a64dd27cc8e73013304b10d364b5` |
| tmp_analysis/tier3/f0/retarget/parquet_lock.json | `bdb83b0ee34f05a8408a871cba3d8ca2bb4cef6ec0e67ca435a57232b3e0b0d7` |
| data/human_motion_data_tier3_opus/READY | `9c4cb2a96e18cdaa9b8e9d68dd7779107660231d760f2a99a515dc2fea0fff09` |
| opus track2_manifest.json | `fac5a9855dd4074846c6117a3fdd29fd76311bd00283d9937c912d1add261d3e` |
| scripts/t3f0_helper.py (post step, running) | `b907731504074188840ad911794487614dea7511e196e6690e776b1d6993341b` |
| scripts/t3f0_driver.sh (running) | `c861fd3fe44de4d9e75943112a7b86162c5c90fab95255e0c676b9287f547a52` |

Code: `code/fc_r2` HEAD `61fa5d31ff8054bf1e5cf5375ef0957e54d2729b` (training and export code of every f0 / opus run); stock runtime HEAD
`8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`.

## 11. Commands

- start / resume: `bash fc_runs/submissions/t3_f0_final/launch.sh`
- status: `bash fc_runs/submissions/t3_f0_final/status.sh`
- stop: `touch fc_runs/submissions/t3_f0_final/STOP`
- by hand (also run automatically at the end): `python -B t3final_driver.py pack`
- `python -B t3final_driver.py repack --commit-url <URL>` (after the user's OK)
