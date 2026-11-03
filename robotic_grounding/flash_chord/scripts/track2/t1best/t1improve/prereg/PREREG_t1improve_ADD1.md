# PREREG addendum 1: Tier-1 improvement wave 1b (T1improve ADD1, 2026-10-28)

Written 2026-10-28 ~05:05 CST, **before any wave-1b arm is prepared, released or started**: no fork copy, release file,
launch.json, ledger line or process exists for any arm below. Folder `/root/autodl-tmp/haoran.zhang/fc_runs/T1improve`
(below: `T1I`); `ME=/root/autodl-tmp/haoran.zhang`.

This addendum extends `PREREG_t1improve.md` **version 2** (sha256
`e418a6489a25951ce2518b9734caeed28b153645d48c7458d7c6b8700dfae79f`). Version 2 stays unchanged and binding for wave 1;
everything not stated here follows version 2 by section number. This file is hash-locked in
`PREREG_t1improve_ADD1.sha256`. The wave-1b launcher refuses to start anything unless both hashes match, this file names
its own sha256, the 1b wrapper's and the fc_r2 watcher copy's (section 10), and the wave-1 launcher is the pinned,
unedited file.

## 1. Authority, priority and scope
- The main session recorded these user instructions verbatim in release files:
  - "ok make sure all gpu fully used?" and "tier3 tasks is top priority, and then tier2, if no such improvement tasks
    you could run tier1" (`T1I/release/ep05-B3.txt`, 2026-10-27 18:31).
  - "Yes, approve both": the training-only palm/pelvis termination relaxation of ep14-R and ep28-W3H
    (`T1I/release/ep14-R.txt`, 19:52).
- The main session specified the wave-1b arm list on 2026-10-28 ~05:00 CST. This addendum records it.
- Priority: Tier 3 first, then Tier 2 (Zuxing's; none of ours), then wave 1, then **wave 1b**. There is no Tier-3
  retry (`tmp_analysis/tier3/f0_retry/RETRY_NOGO`). Wave 1b never preempts anything.
- Tier 1 may use clean data. Nothing from wave 1b (scores, checkpoints, choices) feeds any Tier-2 or Tier-3 decision.
- Out of scope: uploads, pushes, the evaluator / kit / stock runtime (never modified), GPU 4, MPS, signals.
- Never edited: the running wave-1 launcher (auto pid 756735), PREREG v2, and every file a running trainer or watcher
  executes.
- State at writing (status 04:54 CST): all 10 wave-1 arms RUNNING, nothing pending, gate OPEN.

## 2. Arms (priority order = start order; every arm after every wave-1 arm)

Common to every arm (as v2 section 2):
- `training.resume=true` from a sha-checked copy of the source pair in `T1I/arms/<arm>/fork/`;
- `training.early_stop_help_zero_tail=0`, ABSOLUTE `training.max_iterations`;
- `logging.mode=disabled +logging.local_jsonl=true`;
- the source's experiment, Tier-1 task parquet and `scene.support`, checked against the source's saved overrides
  (sha-pinned);
- `reset.seed` = the arm's seed below. Every source was trained with 42.

| # | arm | item it may replace | code tree | source pair (fc_runs/...) | model sha256 | rsi sha256 | experiment | support | reset.seed | inherited overrides | added overrides | max_iterations (+it) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ep21-B5s2 | ep21 (119500) | b39b69b | C3r2_runin/ep21 @ 119500 | `ca4801dcd3c2838e127c66803886647c055fc2427adc2db03a5113650ad8c6cf` | `be1bd5ffbadd25ee9f4be0b03ccd1d43839f3584ffce315d8c549d4590ceab5f` | g1_recon_body_coord_rsi_r2_runin | false | **43** | - | objective.object_task_rotation_beta=5.0 | 139500 (+20000) |
| 2 | ep14-Rs2 | ep14 (78000) | b39b69b | T1fix/B/ep14 @ 81000 | `6c16cf94e78fffcc45c7c3717da704f0a99b80b1f96d57612db70cba1887a8ae` | `83323fa9f4818f6c2dfd44876f1ae2b6d9f3df742c14ae32e9592323df51a4f5` | g1_recon_body_coord_rsi_r2 | false | **43** | objective.object_task_rotation_beta=2.6 | termination.palm_position_threshold=1.2, termination.palm_orientation_threshold=3.2, termination.pelvis_position_threshold=1.2 | 96000 (+15000) |
| 3 | ep19-B5 | ep19 (108000) | b39b69b | T1fix/B/ep19 @ 110000 | `a433b497254130be426f5ae16afd31f87d4983e7b02ca86d7f15478312b158fc` | `37dc23ed02988522c298c04722692e681e03a980e7af4facbb1276ee7b4e218e` | g1_recon_body_coord_rsi_r2_runin | false | 42 | objective.object_task_rotation_beta=3.0 (replaced) | objective.object_task_rotation_beta=5.0 | 135000 (+25000) |
| 4 | ep10-BTs2 | ep10 (36000) | b39b69b | T1fix/B/ep10 @ 36432 | `dc34e3009039d22aa4e11d83dd04fb27d8a1227ae1a0b177778b4afd9952e442` | `25da0cfecc246573ac611220eada3f3bdeb1b7bf7557651535846b15c5d64bce` | g1_recon_body_coord_rsi_r2 | false | **43** | objective.object_task_rotation_beta=4.5 | termination.object_orientation_threshold=0.6 | 60000 (+23568) |
| 5 | ep05-P30 | ep05 (42500) | b39b69b | C3r2_runin/ep05 @ 42500 | `37371dfcb760cbb15db5c2cc9ca6a566c05cee6d2f3874c08653a4f852ab6704` | `a2f6a2bb9f55ba7a3e1c16bfdbc87e2fdb0c9a63d7b7901eae12b0a4cba7a60e` | g1_recon_body_coord_rsi_r2_runin | false | 42 | - | objective.object_task_position_beta=30.0 | 62500 (+20000) |
| 6 | ep15-P30 | ep15 (20974) | 61fa5d3 | C3r2_wide/ep15 @ 20974 | `16e14bcd737120d0dccb7acc5f5454daf90fd63024eef91264655d98d59f8af7` | `8bffcbd190bad0299b787b8b50afe857eb0f6006725f66d67285fb1ace567987` | g1_recon_body_coord_rsi_r2 | false | 42 | - | objective.object_task_position_beta=30.0 | 41000 (+20026) |

The source saved configs and task parquets are the same files, with the same sha256 pins, as v2 section 2. The
`C3r2_wide/ep15` source has no hydra dir; its wandb-metadata args are pinned as in v2.

### What each arm is
- **ep21-B5s2, ep14-Rs2, ep10-BTs2 (second seeds).** Each is its wave-1 sibling's exact recipe: same source pair,
  overrides and target. Only `reset.seed` changes, to 43.
  - `training.seed = ${reset.seed}` resolves to 43.
  - The checkpoint holds no PRNG-key tensor (114 tensors in `model_119500`; checked), and
    `training/ppo/learner.py` seeds the learner key from `PRNGKey(config.seed)`. So a seed-43 fork of the same pair is
    a distinct stochastic continuation, not a copy.
  - The official evaluation forces `env.reset.seed = 42` (stock `harness.py` line 237, v2 section 9), so the training
    seed changes nothing at evaluation.
  - PLAN action 3 names the ep21 second seed ("Spare slot: B5 with `--seed 43`"). The ep14 and ep10 second seeds come
    from the main session's arm list.
- **ep19-B5.** PLAN action 7's beta-5 variant: the ep19-Bc fork (same pair), rotation beta 3.0 -> 5.0, ending at
  ep19-Bc's 135000.
- **ep05-P30 and ep15-P30 (precision pilot, PLAN action 10).** The settings are taken from PLAN action 10:
  `objective.object_task_position_beta=30`; the experiment value is 10.0; each item's rotation beta is kept (1.0, the
  experiment value, because neither pinned pair overrode it).
  - The setting is config-only in both trees: dataclass field `objectives/recon_body.py` line 80; experiment value at
    `g1_recon_body_coord_rsi_r2.yaml` line 68 and `_runin.yaml` line 28.
  - The exporter strips it: `training/ppo/coordinate/export.py` line 121, `DEV_OBJECTIVE_KEYS`, in b39b69b and in
    fc_r2 61fa5d3.
  - The forks are the pinned pairs of `t1_best20_20261026` (ep05 42500, ep15 20974), the same pairs ep05-B3 and
    ep15-B2 forked.

### Hydra compose (dry run 2026-10-28 05:03 CST, before this file was written)
Every arm was composed in its own tree, with flash_chord imported from that tree. Results:
- The source's saved config re-composes with no drift.
- `training.seed` resolves to the arm's seed.
- `env.termination = ${termination}` and `env.objective = ${objective}`.
- The instantiated termination and objective dataclasses hold exactly the intended values.
- Arm vs source, besides max_iterations / resume / checkpoint / early stop / logging:

| arm | differences from its source |
|---|---|
| ep21-B5s2 | rotation_beta 1.0 -> 5.0; reset.seed 42 -> 43 |
| ep14-Rs2 | palm_position 0.15 -> 1.2; palm_orientation 1.5 -> 3.2; pelvis_position 0.7 -> 1.2; reset.seed 42 -> 43 |
| ep19-B5 | rotation_beta 3.0 -> 5.0 |
| ep10-BTs2 | object_orientation 1.5 -> 0.6; reset.seed 42 -> 43 |
| ep05-P30 | object_task_position_beta 10.0 -> 30.0 |
| ep15-P30 | object_task_position_beta 10.0 -> 30.0 |

The full output is in `T1I/build/dry_run_1b_20261028.txt`.

## 3. Training command, environment, code trees, collision hulls
As v2 sections 3, 4 and 5, with three differences:
- `reset.seed=<the arm's seed>`;
- the marker `T1IMPROVE_WAVE=1b` is added (T1IMPROVE_ARM, T1FIX_ARM, `T1FIX_STAGE=t1improve` and
  `RUNIN_QUEUE=fc_runs/queue_t1improve` are as in wave 1);
- everything lives under `T1I/arms/<arm>/`.

Each arm's `FLASH_CHORD_CACHE_DIR=T1I/arms/<arm>/fc_cache` holds a byte copy of all 29 stock-runtime hull files (listing
digest `4864e7c2eddec2063463a5c7c722e045514316948fa17394c5acca280f82b27d`). The object keys and stock hulls are those
of v2 section 5:

| object | arms | key | stock sha256 (head) |
|---|---|---|---|
| tall_bar_stool | ep21 | 228b2dc97a7f8ce0 | e11854911827 |
| round_table | ep14 | 6c7584c98b6a755d | 6868717638a4 |
| cane | ep19 | 7d05b4ab00818a6f | b1fdceaea89f |
| white_desk | ep10 | b1a56fe315a143bd | e6236a69aaf9 |
| big_black_box | ep05 | c0749211ce6d40ec | a25b504158f4 |
| wet_floor_sign | ep15 | dd4639fc0cde3051 | a14f2b8b9647 |

The v2 section 5 disclosure applies to these arms as to their wave-1 siblings: ep10, ep19, ep05 and ep15 switch at the
fork from their tree's hull set to the evaluator's stock set.

## 4. Watcher protocol (one per arm)
As v2 section 7:
- Script by tree: `scripts/runin_eval_watch.py` for b39b69b arms, `scripts/runin_eval_watch_r2.py` for ep15-P30.
- Environment `RUNIN_QDIR=fc_runs/queue_t1improve`, `RUNIN_RUNS=T1I/arms/<arm>/runs`,
  `RUNIN_EVAL_ROOT=T1I/arms/<arm>/eval`, `RUNIN_WATCH_ITERS=<target>`.
- Arguments `--episodes NN --every 2000 --no-early-stop --lane-max 1`.
- Evaluation: the official evaluator (E2 export 1e-4 with the tree's exporter, stock runtime, frame 0, world_count 1),
  through the shared one-evaluation-per-GPU locks.

`fc_runs/queue_t1improve` is wave 1's queue folder and wave 1b only reads it. Its `jobs.tsv` (sha256
`62a12f19523fdedf766a2c9f262a82f941cc629d6e37f66efee7f712126c4930`) already lists episodes 5, 10, 14, 15, 19 and 21.
Wave 1b shares its `settings.conf` (RESERVE_SLOTS=1), `gpus.conf` (never GPU 4), evallocks and `gpu3_handed_over`.

## 5. Placement: a second launcher instance that defers to wave 1; every arm HELD
**Why a second instance.** The wave-1 launcher's arm list is fixed in its code, and it has no add-arm mechanism. It is
not edited and keeps running.

**The wave-1b launcher.** `scripts/t1improve1b_launch.py` is generated from the wave-1 launcher by
`T1I/build/patch_launcher_1b.py`, which anchors every change and refuses on any mismatch. The changes are:
- the arm list and the release rule;
- its own lock, log, ledger, heartbeat, STOP and PAUSE files;
- deference to wave 1;
- a seed-aware source check and compose check;
- a position-beta compose check;
- read-only use of the queue folder.

It has its own files: `.auto_1b.lock`, `launcher_1b.log`, `launches_1b.tsv`, `heartbeat_1b.log`, `auto_stdout_1b.log`,
`STOP_AUTO_1B`, `PAUSE_1B`. The wave-1 launcher matches only its own arm names, so it ignores wave-1b trainers and
watchers.

**Start conditions.** A wave-1b start needs everything a wave-1 start needs (v2 section 6):
- yield gate (a)-(h);
- the fill rule (at most 3 trainings of anybody per GPU, memory, own-NUMA cores, never GPU 4, no MPS);
- RESERVE_SLOTS = 1 (so wave 1b never takes the last feasible slot);
- the shared GPU-choice locks, held 180 s after a start;
- one start per pass.

Additionally, every one of these must hold:
- **(i) Defer.** Every wave-1 arm has a `launch.json` or failed to start twice. Unreadable state counts as pending.
- **(ii) Hold-off.** At least 600 s since the latest start of either launcher (`launches.tsv` and `launches_1b.tsv`).
- **(iii) No pause.** No `T1I/PAUSE` and no `T1I/PAUSE_1B`.
- **(iv) Release.** `T1I/release/<arm>.txt` exists and is non-empty, written by the main session:
  - for every arm: the confirmation that no Tier-3 resume or revision and no Tier-2 job of Zuxing needs the slot (the
    gate cannot see Zuxing's pending jobs);
  - for **ep14-Rs2**, additionally the user's explicit OK of the training-only palm/pelvis termination relaxation,
    verbatim. The 19:52 OK named ep14-R and ep28-W3H. The main session either quotes a newer OK that covers ep14-Rs2,
    or quotes the 19:52 OK and records why it covers a second seed of the same relaxation on the same task.

**Order.** Strict priority order 1-6. A HELD arm is passed over; once released, it precedes every later arm.

**Not started by the build.** The build runs `init` (arm folders), `prepare` (fork copies, source-config copies, hull
caches; after this file) and `dry`. Starting the loop is `bash scripts/t1improve1b_launch.sh start-auto` from a local
background ssh; pausing is `touch T1I/PAUSE_1B`; stopping the loop is `touch T1I/STOP_AUTO_1B`. Neither launcher ever
signals a process.

**Slots at writing.**
- GPUs 0 and 3 each have one free slot. One of them is the reserve, so at most one wave-1b start fits now.
- Four Tier-3 f0 clips are due to finish ~06:00-10:00 CST today and free four slots. The gate gives any Tier-3 need
  priority first.

## 6. Selection protocol: v2 section 8, unchanged
- **Items.** Wave-1b arms are more candidates for their items: ep21-B5s2 for ep21, ep14-Rs2 for ep14, ep19-B5 for
  ep19, ep10-BTs2 for ep10, ep05-P30 for ep05, ep15-P30 for ep15. Seeds are just more candidates; there is no
  averaging over seeds.
- **Bars.** The bars are v2 section 2's evaluator-JSON numbers:

| item | JSON AUC mean (bar) | needed (bar + 0.005) | floor (bar - 0.10) | JSON MPPE mean cm | MPPE cap (d) cm |
|---|---:|---:|---:|---:|---:|
| ep21 | 0.7548 | 0.7598 | 0.6548 | 1.497 | 1.647 |
| ep14 | 0.8265 | 0.8315 | 0.7265 | 1.503 | 1.653 |
| ep19 | 0.8951 | 0.9001 | 0.7951 | 1.023 | 1.173 |
| ep10 | 0.8675 | 0.8725 | 0.7675 | 1.312 | 1.462 |
| ep05 | 0.9480 | 0.9530 | 0.8480 | 0.786 | 0.936 |
| ep15 | 0.9370 | 0.9420 | 0.8370 | 0.876 | 1.026 |

- **Steps, unchanged.**
  - 8.2: candidates per arm are the top-2 scans in its last 6000 iterations plus its final checkpoint.
  - 8.3: at least 3 fresh official evaluations per candidate (none of these items needs 5). The nominating scan and
    the watcher's `runin_final.json` re-evaluations do not count.
  - 8.4 (a)-(f): ep14 and ep21 candidates also need 0 collapses in the world_count=64 screen.
  - 8.5: 20-trial top-up of changed items only, with the incumbent side by side; SP / MP pass counts not lower.
  - 8.6: the upload file is the trial closest to the set mean, never a best-of.
- **Several passing candidates of one item, across both waves.** The highest candidate mean AUC wins. Ties go to the
  lower MPPE, then the earlier iteration, then the earlier arm: the v2 section 2 order first, then this section 2's
  order.
- **Report only; these change no decision:**
  - PLAN 7, "keep ep19-B5 only if its MPPE is within +0.1 cm of ep19-Bc": reported; 8.4 (d) binds.
  - PLAN 10's P-pilot gate (vs the incumbent's 20-trial mean: MPPE at least 15% lower, AUC no lower than -0.002,
    SP/MP unchanged): reported from the section-8 candidate evaluations, plus the first 2000-grid scan at or after
    +8000 (ep05 52000, ep15 30000).
- **What a P30 candidate needs.** Under 8.4 (a), a P30 candidate replaces its item only with a JSON AUC gain of at least
  +0.005. Admitting an MPPE-only gain, or rolling P30 out to other items, would need a further addendum; neither is
  pre-registered here.

## 7. Early reads and kill flags (report only; v2 section 10 procedure; the launcher never acts)
- **ep21-B5s2 at +8k (127500):** kill flag if ADD over frames 585-677 is still > 12 cm.
- **ep14-Rs2 at +4-6k:** the target read is rsi bins 34..37 fail rate < 0.5 and final tilt < 10 deg. Kill flag at +8k
  (89000) if the final tilt is still > 18 deg.
- **ep10-BTs2 at +6k (42432):** kill flag if > 50% of episodes end by object termination, or kf_committed stalls.
- **ep19-B5, ep05-P30, ep15-P30:** no kill flag.

The stop procedure is v2 section 10's, done by the main session:
1. Write `T1I/arms/<arm>/eval/epNN/stop.json` first.
2. Then send SIGINT to the arm's own trainer: the pid from its `launch.json`, verified by `T1IMPROVE_ARM=<arm>` and
   `T1IMPROVE_WAVE=1b`. Never an MPS client.

## 8. Disclosure: training-only changes
- v2 section 9 applies unchanged to ep14-Rs2 (palm 1.2 m / 3.2 rad, pelvis 1.2 m) and ep10-BTs2 (object orientation
  0.6 rad). These thresholds only decide when a TRAINING episode is cut. The official evaluation sets every
  tracking-failure threshold to 1.0e6 (stock `harness.py` lines 239-242).
- P30 changes only a training reward weight, and the exporter strips `object_task_position_beta`.

## 9. Deviations from PLAN.md (sha256 `1b099ed493f279527bf2083a9960eba790ae628df6d96c03a1da9a1b5d234d64`)
- **Second seeds of ep14-R and ep10-BT:** not in PLAN; from the main session's arm list. (The ep21 B5 second seed is
  PLAN action 3.)
- **ep19-B5:** ends at 135000 (PLAN 7: 140000), like ep19-Bc.
- **P30:**
  - ep05 and ep15 only (PLAN 10: also ep17, ep22, ep09);
  - +20000 / +20026 iterations (PLAN 10: +8000), per the main session's "~+20k" for every arm;
  - no roll-out here.
- **Margin and metric:** the replacement margin is the task's 0.005 in the evaluator-JSON metric, as in v2 section 11.
- **Still not run:** ep14-RB, ep28-W3C, ep21-B25 / ep21-R (code), ep01-LB, ep15-C, ep09-C, ep04 cont5.

## 10. Files (sha256)

| file | sha256 | notes |
|---|---|---|
| wave-1b launcher `scripts/t1improve1b_launch.py` | `ae7fcc754460938baf1d6a9b51b50eaf0e05396e95e9a49e351729eef9cc4d69` | mode 444 |
| wave-1b wrapper `scripts/t1improve1b_launch.sh` | `9c52a8862d90b8fec0b93f2580d5d5fac9d96f322d260afc8c66f4ec1f560d5d` | mode 444; run with `bash` |
| generator `T1I/build/patch_launcher_1b.py` | `a8be5b1b7cf7e9f1db057ae1c4bc5380cd599b0f4f1c6d1e5fa52359d38efcb1` | |
| wave-1 launcher `scripts/t1improve_launch.py` | `ba5146a4ddbd5f2f4b51f5ba4436444f3a30dfaae5ba63d6c75c5cd02eeb3cbd` | pinned, unedited; the 1b launcher refuses if it changes |
| fc_r2 watcher copy `scripts/runin_eval_watch_r2.py` | `5572b07446bfc2537ce69760902381ec9bc8cda8f7a15fc087d7f5acc93b33d8` | |
| watcher `scripts/runin_eval_watch.py` | `61ac5b8a170606d3804a00c927834afe1f8a4d84ac99f6ba594c8b028ab6c5cc` | |
| PREREG v2 `T1I/PREREG_t1improve.md` | `e418a6489a25951ce2518b9734caeed28b153645d48c7458d7c6b8700dfae79f` | |
| queue `fc_runs/queue_t1improve/jobs.tsv` | `62a12f19523fdedf766a2c9f262a82f941cc629d6e37f66efee7f712126c4930` | read only |

Commands:
- `bash scripts/t1improve1b_launch.sh check | dry | gate | status`
- `bash scripts/t1improve1b_launch.sh start-auto` (not started by the build)
- `touch T1I/PAUSE_1B` (pause starts); `touch T1I/STOP_AUTO_1B` (stop the loop)
