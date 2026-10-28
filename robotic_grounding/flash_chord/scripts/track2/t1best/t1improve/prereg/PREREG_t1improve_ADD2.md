# PREREG addendum 2: wave-1b arms may also start on GPU 4 (T1improve ADD2, 2026-10-28)

Written 2026-10-28 ~06:00 CST, **before the GPU-4 instance is enabled**: it has started nothing, and no GPU-4 ledger
line, launch record or process exists. Folder `/root/autodl-tmp/haoran.zhang/fc_runs/T1improve` (below: `T1I`);
`ME=/root/autodl-tmp/haoran.zhang`.

This addendum extends `PREREG_t1improve.md` version 2 (sha256
`e418a6489a25951ce2518b9734caeed28b153645d48c7458d7c6b8700dfae79f`) and `PREREG_t1improve_ADD1.md` (sha256
`5faa730bc8248135af57060d407e66a2b79ce950240e95e8888acabe0d196244`). Both stay unchanged and binding. Everything not
stated here follows them. This file is hash-locked in `PREREG_t1improve_ADD2.sha256`. The GPU-4 instance refuses to
start anything unless all three hashes match, this file names its own sha256, the GPU-4 wrapper's and the fc_r2 watcher
copy's (section 8), and the wave-1 launcher, the wave-1b launcher and the 1b wrapper are the pinned, unedited files.

## 1. Authority
- **User, verbatim (2026-10-28 ~05:42 CST):** "ok you could also use gpu4 since it is emty right now?"
- This lifts the standing rule "never GPU 4" (GPU 4 kept for Zuxing / tests; v2 FILL RULE, ADD1 section 1 "Out of scope:
  ... GPU 4") **for the Tier-1 wave-1b arms only**, and only **while GPU 4 is free of other people's work**. Read as a
  condition:
  - start on GPU 4 only while it holds fewer than 3 trainings of anybody;
  - if a job that is not ours appears on GPU 4 (Zuxing's or anyone's), keep ours running but start no more there, to
    leave room for him.
- Unchanged:
  - the priority (Tier 3, then Tier 2, then Tier-1 wave 1, then wave 1b);
  - the Tier-3 yield gate;
  - every other launcher's "never GPU 4": wave 1, wave 1b's shared queue, the Tier-3 f0 driver and final pipeline;
  - the Tier-2 round-3 PREREG, whose evaluations never use GPU 4.
- Nothing from these arms feeds any Tier-2 or Tier-3 decision.

## 2. Arms, training recipe, selection: unchanged
- The arms are ADD1 section 2's six wave-1b arms, in the same order, with the same release files and the same arm
  folders, fork copies, hull caches, commands and watchers.
- No arm is added and no recipe changes.
- Placement does not change the training: GPU 4 is the same RTX PRO 6000 model, and the trainer environment is identical.
  It adds one marker, `T1IMPROVE_LAUNCHER=1b_g4`, for provenance. `T1IMPROVE_ARM` and `T1IMPROVE_WAVE=1b` are kept, so
  the ADD1 section 7 stop procedure applies unchanged.
- Selection: v2 section 8 / ADD1 section 6, unchanged.

## 3. Placement: a GPU-4-only instance of the wave-1b launcher
**Why a separate instance.** The running launchers are never edited in place:
- `scripts/t1improve1b_launch.py`, auto pid 230552;
- `scripts/t1improve_launch.py`, auto pid 756735.

Both hard-refuse GPU 4 in code.

**How it is built.** `scripts/t1improve1b_g4_launch.py` is generated from the 1b launcher by
`T1I/build/patch_launcher_1b_g4.py`. The generator anchors every change, refuses on any mismatch, and pins the 1b
launcher's sha256.

**Start conditions.** A start needs everything a wave-1b start needs (ADD1 section 5):
- yield gate (a)-(h);
- deference to wave 1;
- release file;
- no `T1I/PAUSE` or `T1I/PAUSE_1B`;
- the shared GPU-choice locks, held 180 s after a start;
- one start per pass.

Also:
1. **GPU 4 only.** No other GPU is ever a candidate.
2. **At most 3 trainings of anybody on GPU 4.** This counts nvidia-smi trainings plus trainers not yet visible, as in
   the v2 fill rule.
3. **Memory rule.** As v2: free ≥ JOB_MB + MIN_FREE_AFTER_MB + the ramp debt of trainings younger than 900 s.
4. **No MPS.**
5. **NUMA-correct cores.** GPU 4 is on NUMA1. Each trainer gets 2 physical cores (+HT) of the NUMA1 training pool
   (52-99 / 156-203). It never gets the eval lane (100-103 / 204-207) and never cross-NUMA cores. Any slice outside
   the pool is refused.
6. **Foreign guard.** No start while any process on GPU 4 is not ours. Two kinds of process are checked:
   - every nvidia-smi compute app on GPU 4;
   - every live process whose environment names GPU 4 alone: `CUDA_VISIBLE_DEVICES` = 4 or GPU 4's UUID, or
     `RUNIN_GPU` / `TIER2V2_GPU` / `TIER2_GPU` = 4.

   "Ours" means one of:
   - a T1improve trainer: `T1IMPROVE_ARM` of wave 1 or 1b, with `RUNIN_QUEUE=fc_runs/queue_t1improve`;
   - an evaluation started by a T1improve watcher: `RUNIN_EVAL` under `T1I/arms/`.

   The first sighting, in any launch pass, writes the latch `T1I/G4_FOREIGN_SEEN` with the evidence. While the latch
   exists:
   - no start happens on GPU 4, and `start-auto` refuses;
   - running arms keep running, and nothing is signalled.

   Only the main session removes the latch, after checking the evidence and telling the user. The guard is deliberately
   strict: any non-T1improve process counts, our own Tier-2/3 work included. None of those use GPU 4.
7. **Reserve unchanged.** RESERVE_SLOTS (= 1, shared `settings.conf`) is computed over the SHARED GPUs, as before. GPU 4
   is not a Tier-3 GPU and adds no Tier-3 slot. A GPU-4 start must still leave at least 1 feasible shared slot (it can
   only take NUMA1 cores from them).
8. **Hold-off: 600 s between any two of our starts.** The instance reads `launches.tsv`, `launches_1b.tsv` and
   `launches_g4.tsv`. Every GPU-4 start is written to `launches_g4.tsv` **and** `launches_1b.tsv`, so the running 1b
   launcher's own hold-off counts it.
9. **Claiming: never a double start.** An arm is taken if any of these holds:
   - it has a `launch.json`;
   - a live trainer carries its `T1IMPROVE_ARM` marker;
   - any python trainer has its `output_dir`;
   - its run dir holds `model_*`, `rsi_*`, `final.json` or `metrics.jsonl`;
   - it has a line in any of the three ledgers;
   - it has 2 failed starts.

   The instance checks this before planning and again under the shared GPU-choice locks. The 1b launcher re-checks
   `launch.json` and its marker-based trainer scan under the same locks. The trainer is visible from the moment it is
   spawned, and `launch.json` follows about 8 s later. Both happen while the GPU-4 instance holds the locks, so the 1b
   launcher sees the arm as launched and skips it.
10. **Own files:** `.auto_g4.lock`, `launcher_g4.log`, `launches_g4.tsv`, `heartbeat_g4.log`, `auto_stdout_g4.log`,
    `STOP_AUTO_G4`, `PAUSE_G4`, `G4_FOREIGN_SEEN`. `prepare` refuses, because the arms belong to the 1b launcher.

## 4. Watchers and the GPU-4 evaluation lock
- **Same watcher per arm as ADD1 section 4:** same script, environment and arguments.
- **Eval lane.** The watcher evaluates on its trainer's GPU, so a GPU-4 arm is evaluated on GPU 4, on the NUMA1 eval lane
  (100-103 / 204-207). After training ends, the final and re-evaluations go to the GPU of its last evaluation (GPU 4), or
  else to a shared-queue GPU.
- **Eval lock.** `fc_runs/queue_t1improve/evallock/gpu4.lock` is a symlink to the shared
  `fc_runs/queue_runin/evallock/gpu4.lock` (one evaluation per GPU across every watcher).
  - `init` wrote it on 2026-10-28 05:55:41.
  - It is the only write to wave 1's queue folder.
  - The wave-1 / 1b launchers check only GPUs 0-3 and 5-7, and `gpus.conf` (never 4) is untouched. So nothing else
    changes for them.
- **Watcher before `launch.json`.** The instance starts the arm's watcher right after the trainer survives its first
  8 s, and before writing `launch.json`. The 1b loop starts watchers only for arms that have a `launch.json` and no
  watcher, so it never starts a second one.
- **Restarts.** The GPU-4 instance starts or restarts watchers only while no wave-1b auto loop runs. While the 1b loop
  runs, it restarts the watchers of every wave-1b arm, GPU-4 arms included, as usual.

## 5. Dry run and simulation (2026-10-28 05:55-06:00 CST, before this file was hash-locked)
- **State.** The 1b launcher started **ep14-Rs2 at 05:50:17 on GPU 0** (pid 259435, its own pass), so ep14-Rs2 is no
  longer a GPU-4 candidate. Released and unclaimed, in order: ep19-B5, ep10-BTs2, ep05-P30, ep15-P30.
- **Gate:** OPEN. GPU 4: 0 trainings, 97251 MiB free, no MPS, no foreign process.
- **GPU-4 instance, next start:** **ep19-B5 on GPU 4, cpus 64,168,65,169** (NUMA1). One feasible shared slot remains
  after it (reserve 1). It would then place:
  - ep10-BTs2 (cpus 76,180,77,181), at least 10 min later;
  - ep05-P30 (cpus 86,190,87,191), at least 10 min after that.

  With 3 trainings, GPU 4 is full. ep15-P30 stays with the shared queue.
- **Simulation** (`T1I/build/g4_skip_sim.py`, read only). It runs the running 1b launcher's own code against a simulated
  GPU-4 start of ep19-B5:
  - the 1b todo becomes [ep10-BTs2, ep05-P30, ep15-P30], so ep19-B5 is skipped;
  - its `arm_state` is RUNNING on GPU 4;
  - its launch pass stops at the hold-off ("last start 1 s ago < 600 s");
  - after the hold-off, its pass takes its next head, ep10-BTs2, on the shared GPUs (it waits for the reserve);
  - **race path:** the 1b pass picked ep19-B5 before the GPU-4 start was visible. Under the locks it logs "ep19-B5:
    launched meanwhile by another launcher process: skipped";
  - its `ensure_watchers` starts no second watcher;
  - **counterfactual:** writing `launch.json` before the watcher would have made it start one.

  Reverse direction: the GPU-4 instance treats ep21-B5s2 and ep14-Rs2 (started by 1b) as taken, and a simulated
  1b-started ep19-B5 as taken too.
- Outputs:
  - `T1I/build/dry_run_g4_20261028.txt` (the full `dry`, written after this file);
  - `T1I/build/g4_skip_sim_20261028.txt`.

## 6. Enabling (not done by the build)
- Start: `bash scripts/t1improve1b_g4_launch.sh start-auto`, from a local background ssh, after the main session's go.
- Pause GPU-4 starts: `touch T1I/PAUSE_G4`. `T1I/PAUSE` or `T1I/PAUSE_1B` also pause it.
- Stop the loop: `touch T1I/STOP_AUTO_G4`. Trainers and watchers keep running.
- Stopping an arm early: ADD1 section 7, unchanged. The trainer pid is in the arm's `launch.json`.

## 7. Disclosure
- The arm a GPU-4 start takes is the same arm the 1b launcher would have started later on a shared GPU. Only the GPU,
  the start time and the core slice differ. Throughput per trainer on GPU 4 is the same as elsewhere (1-3 trainings per
  GPU, v2).
- Evaluations of GPU-4 arms run on GPU 4: about 3 GB, one at a time, under the lock. If a foreign job later appears on
  GPU 4, those evaluations continue there; only training starts stop.

## 8. Files (sha256)

| file | sha256 | notes |
|---|---|---|
| GPU-4 launcher `scripts/t1improve1b_g4_launch.py` | `108d6e3e50ea351e65880873e84cef4d829764d7f7c2ef978cb25859ab7a0b78` | mode 444 |
| GPU-4 wrapper `scripts/t1improve1b_g4_launch.sh` | `f4461eadd2ca6f2f5f25a0801e5828ee2ee97d81904f8dc4752169ed4fcfd791` | mode 444; run with `bash` |
| generator `T1I/build/patch_launcher_1b_g4.py` | `08b541c3fce44aebcee20cb553c2ef6493f07b064b91257c5cbfc63d9dfae93e` | |
| simulation `T1I/build/g4_skip_sim.py` | `7f9142cb3e108e751eb64064e7208e01fce7ededa932d6d0c531e11b60563f9b` | read only |
| wave-1b launcher `scripts/t1improve1b_launch.py` | `ae7fcc754460938baf1d6a9b51b50eaf0e05396e95e9a49e351729eef9cc4d69` | pinned, unedited (auto pid 230552) |
| wave-1b wrapper `scripts/t1improve1b_launch.sh` | `9c52a8862d90b8fec0b93f2580d5d5fac9d96f322d260afc8c66f4ec1f560d5d` | pinned, unedited |
| wave-1 launcher `scripts/t1improve_launch.py` | `ba5146a4ddbd5f2f4b51f5ba4436444f3a30dfaae5ba63d6c75c5cd02eeb3cbd` | pinned, unedited (auto pid 756735) |
| fc_r2 watcher copy `scripts/runin_eval_watch_r2.py` | `5572b07446bfc2537ce69760902381ec9bc8cda8f7a15fc087d7f5acc93b33d8` | |
| watcher `scripts/runin_eval_watch.py` | `61ac5b8a170606d3804a00c927834afe1f8a4d84ac99f6ba594c8b028ab6c5cc` | |
| PREREG v2 | `e418a6489a25951ce2518b9734caeed28b153645d48c7458d7c6b8700dfae79f` | |
| ADD1 | `5faa730bc8248135af57060d407e66a2b79ce950240e95e8888acabe0d196244` | |

Commands:
- `bash scripts/t1improve1b_g4_launch.sh check | dry | gate | status`
- `bash scripts/t1improve1b_g4_launch.sh start-auto` (not started by the build)
- `touch T1I/PAUSE_G4`; `touch T1I/STOP_AUTO_G4`
