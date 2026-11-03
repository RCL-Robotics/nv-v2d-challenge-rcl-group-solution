# CONFIRM_RULES: T1improve candidate confirmation, 10 official evaluations per candidate (2026-10-28)

Written 2026-10-28 ~14:30 CST, **before any export or evaluation of this pass**: `confirm/exports/` and `confirm/runs/`
do not exist yet. Folder `/root/autodl-tmp/haoran.zhang/fc_runs/T1improve/confirm` (below: `C`); `T1I` = its parent;
`ME=/root/autodl-tmp/haoran.zhang`. This file is hash-locked in `CONFIRM_RULES.sha256`; the driver refuses to export or
evaluate unless the hash matches and this file names the driver's and the screen scorer's sha256 (section 11).

It applies `PREREG_t1improve.md` **version 2** (sha256 `e418a6489a25951ce2518b9734caeed28b153645d48c7458d7c6b8700dfae79f`),
`PREREG_t1improve_ADD1.md` (`5faa730bc8248135af57060d407e66a2b79ce950240e95e8888acabe0d196244`) and
`PREREG_t1improve_ADD2.md` (`82ec3c75ab594fe993542a2d2923b8c835836616f9a20bc873ae5d09f6f64725`). All three stay unchanged
and binding. The only change made here is the user's (section 1); everything else is a clarification written before
the first read.

## 1. Authority: the user's request, verbatim
- Main session question (2026-10-28 ~14:15 CST): "OK to run the Tier-1 confirmation plus the 20-trial top-up? It's
  about 8 GPU-hours of short official evals on eval lanes, running alongside the DexDuet XHand training without taking
  its training slots."
- **User, verbatim (2026-10-28 ~14:20 CST):** "ok so you can eval all current best seed 10 times and give the seed that
  is the best with the submitable csv?"
- Reading:
  - "eval all current best ... 10 times": every candidate of the ended arms (section 3) gets **10 fresh official
    evaluations**. PREREG v2 8.3 asks for "at least 3" (5 for ep01 / ep25); 10 satisfies "at least", but the number is
    the user's, so it is recorded as a **user-requested deviation** (section 9, D1). All 10 count; none is dropped.
  - "the seed that is the best": the official evaluator has **no selectable seed**. The harness forces
    `env.reset.seed = 42` (stock `harness.py` line 237); the actuator-delay seed (0, group seeds 0-4), object-scale seed
    (0), wrench seed (0) and ADD vertex seed (0 / 0) come from the evaluated file's own config or code constants
    (t1best20 PREREG section 4 audit). Every evaluation of one exported file therefore has the same seed signature; the
    spread between evaluations is GPU floating-point non-determinism. The seed signature is recorded for every
    evaluation (`seeds.tsv`). The "best seed" is reported as the best **trial**, labelled report-only.
  - "with the submitable csv": the submittable file follows PREREG v2 8.6 (the trial closest to the set mean, never a
    best-of trial). A best-trial file is also made, **report only**. Packing, the cited commit and the 20-trial top-up
    are later steps of the same request (section 8); nothing is uploaded and nothing is pushed.
- Out of scope here: uploads, pushes, the evaluator / kit / stock runtime (never modified), MPS, signals, GPU 4
  (section 6), any Tier-2 / Tier-3 decision (nothing from this pass feeds one).

## 2. PREREG v2 section 8, quoted verbatim (binding)

> ## 8. Selection protocol (binding; written before the first read)
> 1. **Scans** (watcher): one official evaluation per 2000-grid checkpoint as above.
> 2. **Candidates per arm**, fixed once the arm has ended: the top-2 scan AUCs among its checkpoints in its last 6000
>    iterations (iteration >= end - 6000) plus its final checkpoint (deduplicated). An arm stopped early by a kill flag
>    (section 10) uses its last checkpoint as the end.
> 3. **Candidate evaluations**: at least **3 fresh official evaluations** of each candidate (5 for ep01-Bc and ep25-B25,
>    the items whose 20-trial AUC sd exceeds 0.005), same protocol as the watcher, with the arm's tree exporter. The scan
>    that nominated a candidate is **not** counted in its mean. (version 2) The watcher's own `runin_final.json`
>    re-evaluations are report-only and do **not** count as candidate evaluations. A checkpoint whose 1e-4 export fails
>    is not a candidate (no looser tolerance); if that is the final checkpoint, the final-checkpoint slot goes to the
>    arm's newest checkpoint whose export passes.
> 4. **A candidate replaces the pinned item only if all hold** (version 2: every AUC / MPPE comparison is evaluator
>    JSON vs evaluator JSON, against the section-2 bar columns; SP / MP rules as PLAN section 7.3):
>    (a) its mean JSON AUC >= the item's 20-trial JSON AUC mean + **0.005** (section 2, "needed");
>    (b) SP-SR passes in **every** candidate evaluation;
>    (c) MP-SR passes in **every** candidate evaluation. ep28 only: a candidate that misses MP in some candidate
>        evaluation may still go to the 20-trial confirmation (step 5) when (a), (b), (d), (e) and (f) hold, and is
>        kept only if MP passes in at least **16/20** of its confirmation trials (PLAN 7.3);
>    (d) JSON MPPE mean <= the item's 20-trial JSON MPPE mean + 0.15 cm (+0.30 cm allowed when the AUC gain in (a) is
>        >= 0.02);
>    (e) no candidate evaluation below the item's 20-trial JSON AUC mean - 0.10 (section 2, "floor");
>    (f) ep01, ep14, ep21 and ep28 candidates also need 0 collapses in a robustness screen with the official evaluator
>        at `evaluation.world_count=64` (PLAN section 7.4; a screen only, never a score; collapse = a world's JSON AUC
>        below the item's 20-trial JSON AUC mean - 0.10; first check on one candidate that world 0 reproduces the
>        official score).
>    Several passing candidates of one item (including ep10-Bc vs ep10-BT): the highest candidate mean AUC wins; ties ->
>    lower MPPE, then the earlier iteration, then the earlier arm in section 2.
> 5. **20-trial top-up of only the changed items** with the t1best20 protocol (`fc_runs/submissions/t1_best20_20261026/
>    PREREG.md` sections 3-5: one exported file per item evaluated 20 times, official invocation, default seeds,
>    lanes and checks unchanged), the item's incumbent evaluated side by side as the alternate. Per changed item the
>    final choice is the one with the higher 20-trial mean AUC unless its MPPE mean is > 0.2 cm worse (PLAN 7.5).
>    Both sides of this comparison are scored the same way (the t1best20 kit metric_code on the packed CSV). Added in
>    version 2: the new item is kept only if its SP and MP pass counts over the 20 trials are not lower than the
>    incumbent's side-by-side counts (an ep28 candidate admitted by the 4(c) exception needs MP >= 16/20); otherwise the
>    incumbent stays.
>    Trial t of the new set = the t1best20 trial t of every unchanged item + top-up trial t of every changed item.
> 6. **Upload file = the trial closest to the set mean**: among trials whose SP-SR and MP-SR equal their 20-trial mode,
>    the one minimising |AUC - mean|/sd_AUC + |MPPE - mean|/sd_MPPE. **Never a best-of trial or a best single scan.**
>    If an interim file was uploaded, the final file must dominate it on all four metrics (PLAN 7.6). Packing, pushing
>    and uploading need the user's OK and are not part of this wave.
> 7. Reporting: the 20-trial mean including collapses, next to the uploaded trial.

ADD1 section 6 (binding for the ADD1 arms in this pass, ep21-B5s2 and ep15-P30), quoted: "Wave-1b arms are more
candidates for their items ... Seeds are just more candidates; there is no averaging over seeds." "Several passing
candidates of one item, across both waves. The highest candidate mean AUC wins. Ties go to the lower MPPE, then the
earlier iteration, then the earlier arm: the v2 section 2 order first, then this section 2's order." "Under 8.4 (a), a
P30 candidate replaces its item only with a JSON AUC gain of at least +0.005." PLAN 10's P-pilot gate and PLAN 7's
"ep19-B5 within +0.1 cm of ep19-Bc" are report only.

## 3. Arms and candidates in this pass (section 8.2, re-derived by `t1confirm_driver.py build`)
The 12 ended arms (final.json present, last checkpoint = max_iterations, no live trainer). The driver re-derives 8.2
from each arm's watcher `eval/evals.csv` (ok scans only; ties in scan AUC -> the earlier iteration) and refuses unless
the result equals the dry pass's list. `candidates.json` as built: sha256
`ae80a6616fef5213df16c66a9bede96610b154c91209a1e947fdfdfa1f6d5653` (29 candidates; it changes only if a final-slot
fallback is appended, section 4).

| arm (wave) | item / bar row | tree | candidates (nominated by) |
|---|---|---|---|
| ep14-R (v2) | ep14 | b39b69b | 92000 (scan top-1 0.83264), 95500 (top-2 0.83220), 96000 (final) |
| ep28-W3H (v2) | ep28 = ep28alt (W2PD 66000) | 7fbc227 | 78000 (top-2 0.82079), 83000 (top-1 0.82218; final) |
| ep21-B5 (v2) | ep21 | b39b69b | 136000 (top-2 0.76318), 138000 (top-1 0.76438), 139500 (final) |
| ep10-Bc (v2) | ep10 | b39b69b | 59500 (top-2 0.88704), 60000 (top-1 0.88744; final) |
| ep10-BT (v2) | ep10 | b39b69b | 54000 (top-2 0.89263), 58000 (top-1 0.89405), 60000 (final) |
| ep19-Bc (v2) | ep19 | b39b69b | 130000 (top-1 0.92466), 135000 (top-2 0.92452; final) |
| ep01-Bc (v2) | ep01 | b39b69b | 161500 (top-2 0.79526), 162000 (top-1 0.79727; final) |
| ep15-B2 (v2) | ep15 | 61fa5d3 | 40000 (top-1 0.98575), 42000 (top-2 0.98479; tie with 44000, earlier wins), 46000 (final) |
| ep05-B3 (v2) | ep05 | b39b69b | 60000 (top-1 0.96529), 62000 (top-2 0.96392), 62500 (final) |
| ep25-B25 (v2) | ep25 | b39b69b | 116000 (top-2 0.75473), 122000 (top-1 0.75618; final) |
| ep21-B5s2 (ADD1) | ep21 | b39b69b | 139000 (top-1 0.76447), 139500 (top-2 0.76337; final) |
| ep15-P30 (ADD1) | ep15 | 61fa5d3 | 40000 (top-1 0.96075), 41000 (top-2 0.95927; final) |

**Not in this pass (still training at 14:27 CST):** ep14-Rs2 (pid 259435, target 96000), ep19-B5 (288824, 135000),
ep10-BTs2 (303228, 60000), ep05-P30 (319195, 62500). They are candidates for a later pass under the same rules; a
winner chosen now can be displaced only by that later pass's own confirmation.

## 4. Exports
- Each candidate is exported **once**, by its arm's tree exporter, exactly the command `scripts/track2/eval_all.py`
  runs: `python -B -m flash_chord.training.ppo.coordinate.export --input <model_N> --output C/exports/<cand>/try<t>/model_N.safetensors
  --stock-runtime <stock> --report .../export_report.json --tolerance 0.0001` (PYTHONPATH = the tree's src, CPU only,
  nice 10 on the eval-core lanes). Checks as t1best20 `check_export` (status ok, tolerance 1e-4, max_rel_diff <= 1e-4,
  stock runtime verified, eval_code digest = the common build, no help scale, source sha256 = the training
  checkpoint, file sha256 = the report).
- **Tries:** up to 2 (the watcher also tries twice); a refusal at 1e-4 counts as a failed try. A candidate whose 2 tries
  both fail is **not a candidate** (8.3, no looser tolerance). If it is the arm's final checkpoint, the final slot goes
  to the newest checkpoint of the arm (by iteration, any grid) not yet a candidate whose export passes, at most 5
  attempts; the driver appends it to `candidates.json` ("final-slot fallback").
- After the exports and before the first evaluation, `EXPORT_PINS.md` (hash-locked in `EXPORT_PINS.sha256`) lists every
  candidate's evaluated file and sha256; the driver refuses to evaluate a file that it does not name.
- The exported files are kept for good: every confirmation evaluation, the top-up trials and the packed file of a
  chosen candidate use that same file.

## 5. The 10 evaluations and the collapse screen
- **One evaluation** (tag `t01` ... `t10`) = the t1best20 official invocation (t1best20 PREREG section 3) of the pinned
  export: `nice -n 10 taskset -c <NUMA eval cores> python -B <tree>/scripts/track2/eval_all.py --arm-dir <work> --arm
  T1confirm --export no --iteration N --episodes <ep> --gpus <g> --jobs-per-gpu 1 --world-count 1 --skip-pack --kit-root
  <kit> --stock-runtime <stock 8c778f9> ...` with `<work>/epNN/model_N.safetensors` a symlink to the pinned file. The
  kit runner then runs the kit `evaluate_policy.py` (160d24f5...) on the untouched stock runtime, frame 0, explicit
  reset, world_count 1. Metrics: the evaluator's results JSON (add_auc, spider_sr_uncentered, maniptrans_sr,
  mean_per_point_error_cm). This is the watcher's evaluation path with the export done once instead of per evaluation
  (section 9, D2).
- Checks (all must hold for an "ok" evaluation): exit 0 and eval_report status "evaluated"; JSON + parquet exist; JSON
  and parquet checkpoint_sha256 = the pinned file and the JSON names it; the Tier-1 sequence; start_frame 0,
  motion_start_frame 0, world_count 1, reset explicit, no non-finite world; parquet eval_script / eval_code digests =
  the common build; environment_valid everywhere; the pinned file unchanged afterwards.
- **A completed evaluation is never re-run, whatever its score.** A technical failure is retried (<= 3 tries, 5 min
  apart); a tag failing 3 times is reported failed. A candidate is judged on its ok evaluations if it has at least 3 (5
  for ep01 / ep25, 8.3) once all 10 tags are final; otherwise it fails.
- Order: trial-major (t01 of all 29 candidates, then t02, ...), then the screens.
- **Collapse screen (8.4 f)**, tag `w64`: the same invocation with `--world-count 64`, once per candidate of ep01, ep14,
  ep21 and ep28 (12 candidates; only the screens of candidates passing (a)-(e) decide anything, the others are
  reported). The 64 worlds share the frame-0 explicit reset; each world draws its own actuator lags (stock
  `runtime/delay.py`: rand_init(seed + group, world + reset_count x 104729)), world 0 gets the official lags. Checks as
  above with world_count 64.
  - Per-world AUC: the evaluator JSON pools all worlds, so each world's AUC is computed from that run's rollout parquet
    by `C/screen_score.py`: the kit's own maths (`v2dlb/object_pose.py`, the transcription of the evaluator's
    `metrics.py` that the Kaggle metric inlines) on one world's slice, with the kit's canonical ADD vertices (released
    visual mesh, 500 points, seed 0) and the evaluator's reference columns. Self-test before this file
    (`C/selftest/compare.out`, 16 world_count=1 trials of ep01 / ep14 / ep21 / ep28alt from t1best20): the scorer equals
    the kit metric_code per-episode AUC exactly (max difference 0.0) and MPPE / SP / MP exactly, and differs from the
    evaluator JSON AUC by at most 0.0024 (the evaluator samples its vertices from the simulation mesh). The rule's
    margin is 0.10, so the scorer stands in for "a world's JSON AUC".
  - **Collapse** = a world whose AUC is below the item's 20-trial JSON AUC mean - 0.10 (the floor, section 7). (f)
    passes with 0 collapses in 64 worlds **and** the world-0 check.
  - **World-0 check** (8.4 f "first check on one candidate that world 0 reproduces the official score"; applied to
    every screen, not only the first): world 0's AUC lies within [min - 0.01, max + 0.01] of the same scorer's AUC over
    that candidate's ok t01-t10 evaluations (at least 3). If it fails, (f) is **not** met (fail closed) and the report
    flags it.
  - Report only: per-world SP / MP failures, world AUC mean / min, pooled AUC vs the run's JSON AUC.

## 6. Lanes, GPUs and safety
- GPUs 0, 1, 2, 3, 5, 6, 7. **GPU 4 is not used**: ADD2 lets only the wave-1b arms that train on GPU 4 evaluate there,
  and this pass has none; GPU 4 also holds 3 trainings now.
- One evaluation per GPU across every driver: each evaluation holds `C/evallock/gpu<g>.lock` plus every existing shared
  lane-lock **file** of that GPU, each inode once (the `queue_t1fix*` and `queue_t1improve` locks of GPUs 0-2 and 5-7
  are links to `queue_runin`'s), all or nothing, never blocking, never created: `fc_runs/queue_runin`, `queue_tier2v2`,
  `queue_t1fix`..`queue_t1fix5`, `queue_t2fix`, `queue_t1improve` (GPU 3's own file), `submissions/t1_best20_20261026`,
  `submissions/t3_f0_final`, `submissions/t2_consensus_r2_20261027`, `submissions/t2_consensus_r3_20261028`,
  `tmp_analysis/tier3/{f0,opus_easy}`, `tmp_analysis/tier2/{k3ca/train,k3fix_pilot,k3full,k3easy}`.
- At most 4 of our evaluations at once, at most 2 per NUMA eval-core lane (48-51,152-155 for GPUs 0-2;
  100-103,204-207 for GPUs 3, 5-7), `nice -n 10`, a 35 s rest of the GPU after each of ours. No MPS GPU; free memory
  >= 4096 + 1536 MiB (w64: 4096 + 8192 MiB). These evaluations take no training slot. The DexDuet XHand runner
  (`ch_runs/xhand_dexduet/run_gpu_xhand.sh`) refuses a GPU only for GPU 4, MPS, or free memory below need + 8192 MiB;
  one evaluation uses about 1.5-3 GB, so it is not blocked.
- **Never signals a process**, its own children included: an evaluation running past 40 min is only logged OVERDUE. No
  arm, trainer, watcher or launcher is touched; nothing of Zuxing's or of the DexDuet workflow is touched.

## 7. Bars (PREREG v2 section 2; unrounded from t1best20 `per_episode.csv`, sha256 `e02cd1f8...`)

| bar row | JSON AUC mean | needed (+0.005) | floor (-0.10) | JSON MPPE mean cm | cap +0.15 | cap +0.30 (gain >= 0.02) | SP / MP pass (20) |
|---|---:|---:|---:|---:|---:|---:|---|
| ep14 | 0.82649 | 0.83149 | 0.72649 | 1.50269 | 1.65269 | 1.80269 | 20 / 20 |
| ep28alt | 0.80733 | 0.81233 | 0.70733 | 3.46553 | 3.61553 | 3.76553 | 20 / 6 |
| ep21 | 0.75477 | 0.75977 | 0.65477 | 1.49727 | 1.64727 | 1.79727 | 20 / 20 |
| ep10 | 0.86753 | 0.87253 | 0.76753 | 1.31243 | 1.46243 | 1.61243 | 20 / 20 |
| ep19 | 0.89505 | 0.90005 | 0.79505 | 1.02312 | 1.17312 | 1.32312 | 20 / 20 |
| ep01 | 0.76031 | 0.76531 | 0.66031 | 3.67288 | 3.82288 | 3.97288 | 19 / 19 |
| ep15 | 0.93697 | 0.94197 | 0.83697 | 0.87628 | 1.02628 | 1.17628 | 20 / 20 |
| ep05 | 0.94797 | 0.95297 | 0.84797 | 0.78635 | 0.93635 | 1.08635 | 20 / 20 |
| ep25 | 0.74901 | 0.75401 | 0.64901 | 2.01295 | 2.16295 | 2.31295 | 20 / 20 |

Comparisons use the unrounded values. "Gain" in (d) = candidate mean JSON AUC - the bar's JSON AUC mean.

## 8. Verdicts, and how the later steps of the request use them
- Per candidate (driver `summary`, `candidates_summary.csv`, `verdicts.json`): QUALIFIES when (a)-(e) hold (+ (f) for
  ep01 / ep14 / ep21 / ep28); QUALIFIES_EP28_MP_EXCEPTION when an ep28 candidate fails only (c) and (a), (b), (d), (e),
  (f) hold (it must then reach MP >= 16/20 in its 20-trial confirmation, else the incumbent stays); otherwise FAIL with
  the failed letters.
- Per item: the qualifier with the highest mean JSON AUC wins (ties: lower MPPE, earlier iteration, earlier arm in the
  v2 then ADD1 order). An item with a winner is a **changed item**.
- **20-trial top-up (8.5), stated now:** the protocol of these 10 evaluations is the t1best20 protocol (one pinned
  export, the same invocation, seeds and checks). So the changed item's t01-t10 here **are trials 01-10** of its 20,
  and the top-up adds t11-t20 of the same file. Trial k of the new set = t1best20 trial k of every unchanged item + trial
  k of every changed item. Caveat, disclosed: trials 01-10 also chose the winner, so its 20-trial mean carries a small
  selection bias (at most the difference between the best and second-best qualifier means, usually far less).
- **Incumbent side of 8.5:** the incumbent's existing 20 t1best20 trials (same protocol, 2026-10-26) are its
  side-by-side record; they are not re-evaluated (section 9, D3). The 8.5 test (higher 20-trial kit metric_code mean AUC,
  MPPE not > 0.2 cm worse, SP / MP pass counts not lower; ep28 exception MP >= 16/20) decides whether the change stays.
- **Files (8.6):** the submittable file = the trial closest to the set mean among trials whose SP-SR and MP-SR equal
  their 20-trial mode, minimising |AUC - mean|/sd_AUC + |MPPE - mean|/sd_MPPE; the best trial is packed as a separate,
  report-only file. Both are packed with the kit packer and the real pushed commit URL, validated by the kit
  uploader's offline check and scored by the kit metric_code; nothing is uploaded.

## 9. Deviations from PREREG v2 (recorded before any evaluation)
- **D1 (user-requested):** 10 fresh official evaluations per candidate instead of "at least 3" (5 for ep01 / ep25).
  Every one of the 10 counts in the mean; the nominating scan and the watcher's re-evaluations still do not count.
- **D2 (export once):** v2 8.3 says "same protocol as the watcher". The watcher exports anew for every evaluation;
  here each candidate is exported once and that file is evaluated 10 times, through the same `eval_all.py` evaluation
  path, exporter, tolerance and stock runtime. This is the t1best20 protocol that 8.5 requires for the top-up, so the
  confirmation evaluations of a winner are reusable as its first 10 trials (section 8).
- **D3 (incumbent not re-evaluated):** 8.5's "incumbent evaluated side by side" is met by the incumbent's existing 20
  trials under the identical protocol, not by 20 fresh evaluations.
- **D4 (world-0 check on every screen):** v2 asks for the check on one candidate; it is applied to every screened
  candidate, fail closed, with the 0.01 tolerance of section 5.
- **D5 (screens for every ep01 / ep14 / ep21 / ep28 candidate):** 12 screens are run, not only for qualifiers, so that
  one pass covers both; only qualifiers' screens decide anything.
- Clarifications (not deviations): ties in scan AUC for 8.2 go to the earlier iteration (only ep15-B2 42000 vs 44000);
  export tries = 2 with refusals counted (section 4); kit per-world AUC stands in for "a world's JSON AUC" (section 5).

## 10. Seeds
`seeds.tsv` records, per evaluation, every seed key read from the evaluated file plus the harness's forced reset seed,
the GPU index and UUID, start / end, the runner command line and the result digests. Expected signature for files
trained with seed 42: `delay=0(+g0..4,0-2) reset=42 scale_seed=0(1.0/1.0) wrench=0 vertex=0/0 ppo=42`; the seed-43 arm
ep21-B5s2 shows `ppo=43` (the training seed; inert at evaluation; its stored reset seed 43 is forced to 42).

## 11. Files (sha256)

| file | sha256 |
|---|---|
| driver `C/t1confirm_driver.py` | `f742f139799ce7b987f4f044ce040bca577fcd7fb71089d3e9cd1d64ca6e336e` |
| per-world scorer `C/screen_score.py` | `33fbd0e5e9cbb606153f0a70e1e49329ddb1ac48d3b67f6f5fb462e42920b525` |
| launcher `C/launch.sh` | `bb52460020129127a8e7217195f6e028b14c6c69f0df816ac36379c37406e47d` |
| `C/candidates.json` (as built, 29 candidates) | `ae80a6616fef5213df16c66a9bede96610b154c91209a1e947fdfdfa1f6d5653` |
| scorer self-test `C/selftest/compare.out` | `8dbdd43b062d382d078109dfccab84ead69f89db4c8f3826564dae4e8d38a614` |
| t1best20 `per_episode.csv` (bars) | `e02cd1f80cc45a83fc80b4f5f0142411473b6db6c0bcb39ae0cd2ab3f7c27adc` |
| tools pinned by the driver's preflight | eval_all.py `657295921e82...` (all three trees), kit runner `0bcc41f5de24...`, kit / stock evaluate_policy.py `160d24f5fbeb...`, exporters b39b69b `04367c56564d...`, 7fbc227 `d9a7d45da58c...`, 61fa5d3 `34e3d7f3a7dc...`, stock HEAD 8c778f9 |

Commands: `bash C/launch.sh --export`; `bash C/launch.sh --dry-run <candidate>` (t01 + w64 of one candidate, kept as
ordinary evaluations); `bash C/launch.sh`; `python -B C/t1confirm_driver.py status | summary`; stop: `touch C/STOP`.
