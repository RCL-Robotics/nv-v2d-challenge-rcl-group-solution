#!/usr/bin/env python3
"""t1improve_launch.py: Tier-1 improvement wave of 2026-10-27 (fc_runs/T1improve), queued BEHIND Tier 3.

User (verbatim, 2026-10-27 ~17:22 CST, the latest instruction): "why retry, I think no need retry right keep current results
for tier3 queue tier1 first, we will wait first round all tier3 finished then add revisement on tier3". Priority recorded by
the main session: Tier 3 first, then Tier 2 (Zuxing's; none of ours), then this Tier-1 wave. Plan:
tmp_analysis/t1improve_20261027/PLAN.md. Pre-registration: fc_runs/T1improve/PREREG_t1improve.md, hash-locked in
PREREG_t1improve.sha256; launch modes refuse to start anything unless that hash matches AND the PREREG names this file's
sha256 (and the shell wrapper's and the fc_r2 watcher copy's). A NEW file: scripts/t1fix{,2,3,4,5}_launch.py are untouched.

ARMS (priority order = start order; an arm HELD for a release (RELEASE below) is passed over until it is released;
  every other arm strictly in this order)
  ep14-R    T1fix/B/ep14 model/rsi_81000      b39b69b  beta 2.6 (inh.) + palm_pos 1.2, palm_ori 3.2, pelvis_pos 1.2 -> 96000
  ep28-W3H  T1fix/W2PD/ep28 model/rsi_68000   7fbc227  every W2PD override (inh.) + palm_pos 1.2, palm_ori 3.2      -> 83000
  ep21-B5   C3r2_runin/ep21 model/rsi_119500  b39b69b  beta 5.0                                                     -> 139500
  ep10-Bc   T1fix/B/ep10 model/rsi_36432      b39b69b  beta 4.5 (inh.)                                              -> 60000
  ep10-BT   T1fix/B/ep10 model/rsi_36432      b39b69b  beta 4.5 (inh.) + object_orientation_threshold 0.6           -> 60000
  ep19-Bc   T1fix/B/ep19 model/rsi_110000     b39b69b  beta 3.0 (inh.)                                              -> 135000
  ep01-Bc   T1fix/B/ep01 model/rsi_142000     b39b69b  beta 3.0 (inh.)                                              -> 162000
  ep15-B2   C3r2_wide/ep15 model/rsi_20974    61fa5d3  beta 2.0                                                     -> 46000
  ep05-B3   C3r2_runin/ep05 model/rsi_42500   b39b69b  beta 3.1   (filler)                                          -> 62500
  ep25-B25  C3r2_runin/ep25 model/rsi_102000  b39b69b  beta 2.5   (filler)                                          -> 122000
  Every arm: the source's experiment, task.parquet (Tier-1 data root), scene.support and reset.seed 42 (from the source's
  saved config, sha-pinned); training.resume=true from a sha-checked COPY of the pair in arms/<arm>/fork/ (outside the run
  dir); training.early_stop_help_zero_tail=0; ABSOLUTE training.max_iterations; logging.mode=disabled
  +logging.local_jsonl=true; FLASH_CHORD_CACHE_DIR=arms/<arm>/fc_cache holding a byte copy of ALL 29 stock-runtime CoACD
  hulls (code/video_to_data_8c778f9 .cache/convex; the object's key is checked), as T1fix stage 2 seeded its cache.
  Code trees (cwd + PYTHONPATH=<tree>/src): b39b69b = code/nv-v2d_runin/robotic_grounding/flash_chord, 7fbc227 =
  code/nv-v2d_ep28w2/robotic_grounding/flash_chord, 61fa5d3 = code/fc_r2 (each arm resumes in the tree that wrote its pair).

LAYOUT  fc_runs/T1improve/arms/<arm>/{fork/, source/, fc_cache/convex/, runs/epNN/ (= output_dir), eval/ (watcher root),
        fork.json, launch.json, train.log, watcher_starts.tsv, start_failed}; fc_runs/T1improve/{launcher.log,
        launches.tsv, heartbeat.log, PREREG_t1improve.md(.sha256), STOP_AUTO, PAUSE, .auto.lock}; queue folder
        fc_runs/queue_t1improve {jobs.tsv, gpus.conf, settings.conf, eval_reserve_mb, gpu3_handed_over, README,
        evallock/ (gpu0,1,2,5,6,7.lock = symlinks to fc_runs/queue_runin/evallock/gpuN.lock, the shared one-eval-per-GPU
        locks; gpu3.lock our own file: queue_runin has none)}.

YIELD GATE (every start, re-checked under the GPU-choice locks; fail closed on anything unreadable): no start while
  (a) T1improve/PAUSE exists; (b) tmp_analysis/tier3/f0/waiting.json is unreadable or lists a waiting clip;
  (c) tmp_analysis/tier3/f0/ALL_LAUNCHED is absent (some f0 clip is FRESH, held or not); (d) the f0 driver's last HEARTBEAT
  shows a clip in FRESH / NOT_READY / PARQUET_CHANGED / NEEDS_* / RETARGETING, or is older than T3_STALE_S while the driver
  (driver.pid, a t3f0_driver.sh process) is alive; (e) tmp_analysis/tier3/f0_retry/RETRY_GO exists and the retry driver's
  last heartbeat is missing / older than T3_STALE_S / lists a NEW or QUEUED clip; (f) any other
  tmp_analysis/tier3/*/waiting.json written within T3_STALE_S is unreadable or lists a waiting clip (a future Tier-3
  revision driver that follows the f0 convention); (g) fewer than RESERVE_SLOTS (settings.conf; 1 since PREREG v2) feasible
  slots would remain after the start; (h) (PREREG v2) the f0 driver's last HEARTBEAT shows a clip DEAD or CRASHED (no
  live trainer, no final.json: a manual resume would need a slot) and fc_runs/T1improve/t3ack/epNN (the main session's
  note that the clip is not resumed) does not exist.
RELEASE (PREREG v2): ep14-R and ep28-W3H (training-only palm/pelvis termination relaxation: needs the user's explicit OK,
  PLAN section 9 item 3) and the fillers ep05-B3 / ep25-B25 (need the main session's confirmation that no Tier-3 resume
  or revision and no Tier-2 job of Zuxing needs the slot) start only once fc_runs/T1improve/release/<arm>.txt exists
  and is non-empty (written by the main session: the user's verbatim words / the confirmation). Until then they are
  HELD and passed over; once released, an arm precedes every later arm again.
FILL RULE (as scripts/t1fix5_launch.py without the equal share): GPUs of gpus.conf (never 4); no MPS; trainings of
  ANYBODY < 3 (nvidia-smi + not-yet-visible trainers); free >= JOB_MB + MIN_FREE_AFTER_MB + ramp debt of every training
  younger than 900 s; our evallock/gpu<g>.lock present; CORES (2) free physical cores (+HT) of the GPU's own NUMA training
  pool (never the eval-lane cores, never cross-NUMA). Fewest trainings, then most free memory, then lowest index.
  ONE start per pass (POLL_S 60), at least HOLDOFF_S (600 s) after our previous start, under the shared GPU-choice locks
  ch_runs/{seedstudy,f0share05,filler}/.gpulock (opened read-only, never created), held LAUNCH_HOLD_S (180 s) after it.
WATCHERS (one per arm; started right after its trainer, restarted at most 3 times while it has no runin_final.json and no
  eval/STOP): scripts/runin_eval_watch.py (b39b69b exporter), runin_eval_watch_w2.py (7fbc227 exporter, for ep28-W3H) or
  runin_eval_watch_r2.py (fc_r2 61fa5d3 exporter, for ep15-B2; a byte copy of runin_eval_watch.py whose only changed line
  is CODE, written by `init`), with RUNIN_QDIR=fc_runs/queue_t1improve RUNIN_RUNS=arms/<arm>/runs
  RUNIN_EVAL_ROOT=arms/<arm>/eval RUNIN_WATCH_ITERS=<target>, args --episodes NN --every 2000 --no-early-stop --lane-max 1:
  the official evaluator (export 1e-4 with the tree's exporter, stock runtime, frame 0, world_count 1) on the newest 2000-grid
  checkpoint, on the trainer's GPU, nice 10 on the NUMA eval-lane cores; after final.json: the newest checkpoint off-grid,
  then 2 re-evaluations of the best scan -> runin_final.json (report only; the selection protocol is the PREREG's).
  --no-early-stop: a watcher never writes stop.json and never signals. This launcher never signals any process.

SUBCOMMANDS (scripts/t1improve_launch.sh wraps them)
  init / prepare        queue folder, layout, watcher copy / sha-checked fork copies, source-config copies, hull caches
  check                 preflight (pins, copies, caches, queue folder, locks, PREREG, hydra compose of every arm); no writes
  dry                   check + the yield gate + the fill-rule plan for the current GPU view + exact commands; no writes
  gate / status         the yield gate only / per-arm state, evaluations, watchers
  watch-dry ARM         that arm's watcher with --dry-run (writes nothing)
  next                  one launch pass (at most one start) + ensure watchers
  ensure-watchers       start missing watchers of launched arms
  auto                  loop: ensure watchers, one launch pass, heartbeat; ends when every arm has ended and has its
                        runin_final.json (or failed to start twice); STOP: touch fc_runs/T1improve/STOP_AUTO
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ME = Path("/root/autodl-tmp/haoran.zhang")
sys.path.insert(0, str(ME / "scripts"))
import track2_queue_tier2v2_helper as T2H  # noqa: E402  read-only primitives: nvidia-smi, /proc, observe, cpu_slice

T1I = ME / "fc_runs/T1improve"
ARMS_DIR = T1I / "arms"
QDIR = ME / "fc_runs/queue_t1improve"
RUNIN_QDIR = ME / "fc_runs/queue_runin"
SHARED_EVALLOCK = RUNIN_QDIR / "evallock"
FC = ME / "fc_runs"
HMD = ME / "data/human_motion_data"
SOMA = HMD / "whole_body/soma"
PY = ME / "conda_envs/flashchord/bin/python"
PREREG = T1I / "PREREG_t1improve.md"
PREREG_SHA = T1I / "PREREG_t1improve.sha256"
WRAPPER = ME / "scripts/t1improve_launch.sh"
LOCKS = (ME / "ch_runs/seedstudy/.gpulock", ME / "ch_runs/f0share05/.gpulock", ME / "ch_runs/filler/.gpulock")
STOCK_CACHE = ME / "code/video_to_data_8c778f9/robotic_grounding/flash_chord/src/flash_chord/assets/.cache/convex"
STOCK_CACHE_N = 29
STOCK_CACHE_DIGEST = "4864e7c2eddec2063463a5c7c722e045514316948fa17394c5acca280f82b27d"  # sha256 of `sha256sum *.npz`
INCUMBENTS = FC / "submissions/t1_best20_20261026/per_episode.csv"
INCUMBENTS_SHA = "e02cd1f80cc45a83fc80b4f5f0142411473b6db6c0bcb39ae0cd2ab3f7c27adc"
T3ROOT = ME / "tmp_analysis/tier3"
T3F0 = T3ROOT / "f0"
T3RETRY = T3ROOT / "f0_retry"
T3_PRELAUNCH = {"FRESH", "NOT_READY", "PARQUET_CHANGED", "NEEDS_RETARGET", "NEEDS_RCHECK", "RETARGETING"}
RETRY_PENDING = {"NEW", "QUEUED"}
T3_NEEDS_SLOT = {"DEAD", "CRASHED"}  # f0 clips with no live trainer and no final.json (PREREG v2, gate (h))
NEVER_GPUS = {4}
SHARED_GPUS = (0, 1, 2, 3, 5, 6, 7)
DEFAULT_GPUS_CONF = "0:3 1:3 2:3 3:3 5:3 6:3 7:3"
SETTINGS = {"JOB_MB": "16896", "MIN_FREE_AFTER_MB": "20480", "CORES": "2", "LAUNCH_HOLD_S": "180", "HOLDOFF_S": "600",
            "POLL_S": "60", "RESERVE_SLOTS": "1", "T3_STALE_S": "1800"}
RAMP_S = 900.0
LOCK_WAIT_S = 120.0
EVAL_RESERVE_MB = 3072
EVERY = 2000
WATCHER_RESTARTS = 3
START_FAIL_MAX = 2
HEARTBEAT_S = 600.0
HELPER_SHA = {ME / "scripts/runin_helper.py": "99c2a3e7d282ef4f68e6e36797549d2219e6aa1356c66d90483a3a25f3b99bcc",
              ME / "scripts/track2_queue_tier2v2_helper.py": "b75a33c3fe050710178c8690d71213df3891988b3e76196e99a37890a7b45886"}

WATCHER_ORIG = ME / "scripts/runin_eval_watch.py"
WATCHER_ORIG_SHA = "61ac5b8a170606d3804a00c927834afe1f8a4d84ac99f6ba594c8b028ab6c5cc"
WATCHER_W2 = ME / "scripts/runin_eval_watch_w2.py"
WATCHER_W2_SHA = "ac2102b87fe224917b9b008a77697cc22e3082e31d8fe1f1b413eb011f92f5fe"
WATCHER_R2 = ME / "scripts/runin_eval_watch_r2.py"  # written by `init`: the original with ONLY its CODE line changed
WATCHER_LINE_OLD = 'CODE = ME / "code/nv-v2d_runin/robotic_grounding/flash_chord"'
WATCHER_LINE_R2 = ('CODE = ME / "code/fc_r2"  # runin_eval_watch_r2.py: the only changed line (fc_r2 61fa5d3 exporter '
                   'for C3r2-lineage forks; scripts/t1improve_launch.py init)')

_COMMON_FILES = {"scripts/train_rl.py": "eafa5723fab4a3ee565843d6718a8318a91288437d8cc03dc6829ce2a8e703a7",
                 "scripts/track2/eval_all.py": "657295921e820a464160039ad9a65ba92a683108a99dcb197290ae4147363197",
                 "src/flash_chord/assets/mesh.py": "9868910150c90fbebf828b1edde85ed238823a8a44dfbbfd5fd062ad5de3ab1c",
                 "src/flash_chord/configs/termination/recon_body.yaml":
                     "fd039fba9830d6441206888c7c16613795f90f7b8281e53192c24b84cbb6dce1",
                 "src/flash_chord/configs/train.yaml": "fc00ddbabbade7c94c9b889ef78cf6eac2fab7b326b76b73c7ffe8f50363ef43",
                 "src/flash_chord/configs/experiment/g1_recon_body_coord_rsi_r2.yaml":
                     "aa2e31c22fc269d828fc7a96e9100432e52c8327b8688063c1b6e263664091bb"}
_RUNIN_EXP = {"src/flash_chord/configs/experiment/g1_recon_body_coord_rsi_r2_runin.yaml":
              "9ee8e1ba3a263677cf458121d68d35dc22ce208ac2a079a9ac3c637b7ca6a0a1"}
TREES = {
    "b39b69b": {"root": ME / "code/nv-v2d_runin", "code": ME / "code/nv-v2d_runin/robotic_grounding/flash_chord",
                "head": "b39b69be6c2f561c0123c6aeb8cd13f8b9d542e7", "watcher": WATCHER_ORIG, "watcher_sha": WATCHER_ORIG_SHA,
                "files": {**_COMMON_FILES, **_RUNIN_EXP,
                          "src/flash_chord/training/ppo/coordinate/export.py":
                              "04367c56564de9540deb3d8e33adbfddb7800df190361a0576005a1a4ccfd212",
                          "src/flash_chord/objectives/recon_body.py":
                              "2476d0514dd4260a841eaa29e29f0d33b34480ab4fbc78ad1a08faaed4baa5d2"},
                "label": "code/nv-v2d_runin @ b39b69b (Zuxing's rolling start + C3r2; local worktree, read only)"},
    "7fbc227": {"root": ME / "code/nv-v2d_ep28w2", "code": ME / "code/nv-v2d_ep28w2/robotic_grounding/flash_chord",
                "head": "7fbc2273a92baefe652cb11e09c7e0acdd62da00", "watcher": WATCHER_W2, "watcher_sha": WATCHER_W2_SHA,
                "files": {**_COMMON_FILES, **_RUNIN_EXP,
                          "src/flash_chord/training/ppo/coordinate/export.py":
                              "d9a7d45da58c44d658af3ef733061c0462bb04c5867994c74cc9e246017eb585",
                          "src/flash_chord/objectives/recon_body.py":
                              "a4f804c07622241df829caa1bf294d68d3280b06cf4aae318c2edf93cef913c8"},
                "label": "code/nv-v2d_ep28w2 @ 7fbc227 (b39b69b + ep28 wave-2 objective; local only, merged in t1best d1373fb)"},
    "61fa5d3": {"root": ME / "code/fc_r2", "code": ME / "code/fc_r2",
                "head": "61fa5d31ff8054bf1e5cf5375ef0957e54d2729b", "watcher": WATCHER_R2, "watcher_sha": None,
                "files": {**_COMMON_FILES,
                          "src/flash_chord/training/ppo/coordinate/export.py":
                              "34e3d7f3a7dca4fdd1e7c4d26750c6332abcc1d629f6a9c6c7a965833a22206f",
                          "src/flash_chord/objectives/recon_body.py":
                              "2476d0514dd4260a841eaa29e29f0d33b34480ab4fbc78ad1a08faaed4baa5d2"},
                "label": "code/fc_r2 @ 61fa5d3 (C3r2; the tree of the C3r2_wide runs and of the running Tier-3 f0 jobs; read only)"},
}

EPISODES = {  # episode -> (sequence_id, object, CoACD key of the object's collision mesh, sha256 of the stock hull file)
    1: ("2026-05-18_17-06-21_vacuum_left_lean_02", "vacuum", "5367fdef34ef90d3",
        "4cc7c2e1da01482a926ea0a1581bef838d4dc0c16e3cfb7606cd10c0e8a44cf9"),
    5: ("2026-03-18_19-36-56_g1_box_rotate_01", "big_black_box", "c0749211ce6d40ec",
        "a25b504158f4204f96c66ef6e4661077be97ff8df56d854ec73dd7ea3e0fb455"),
    10: ("2026-05-01_11-36-40_white_desk_lower_to_ground_10", "white_desk", "b1a56fe315a143bd",
         "e6236a69aaf97e03a4c65f440ec697f4c602722766db6edd0d8695020f709c91"),
    14: ("2026-05-28_14-44-09_round_table_flip_08", "round_table", "6c7584c98b6a755d",
         "6868717638a4333c86a9a0b8397882772cef8c91c57db768078948decaa97f12"),
    15: ("2026-05-06_11-06-15_wet_floor_sign_pick_place_ground_08", "wet_floor_sign", "dd4639fc0cde3051",
         "a14f2b8b9647c1f7d5ab60f3f4f55f670add79500de1d713e9057dc6424d444d"),
    19: ("2026-05-19_10-19-48_cane_swing_05", "cane", "7d05b4ab00818a6f",
         "b1fdceaea89f5a8d9d55252f90b12ff979bbaaaf494a22f34ccfc6fe9a58744f"),
    21: ("2026-04-29_11-53-57_tall_bar_stool_pull_08", "tall_bar_stool", "228b2dc97a7f8ce0",
         "e11854911827aa48343ed3d72d7ff988113d023ebb390d46921f12e5c3e271a7"),
    25: ("2026-03-20_11-59-57_cyan_water_bottle_rotate_01", "cyan_water_bottle", "15263099f076b724",
         "2b40b5283e43e446529ede7fef10eb035feb7f868a1669967cec518f04885497"),
    28: ("2026-05-19_11-27-07_squeegee_sweep_away_from_you_03_", "squeegee", "0885d534a2fd2869",
         "d8c2ee511734339c076f30b52c52715de38d7210179765d0da8f6266de1f91ca"),
}

_HY = ("hydra/.hydra/config.yaml", "hydra/.hydra/overrides.yaml")
SOURCES = {  # run (relative to fc_runs) -> pinned pair + the saved config files it was trained with
    "T1fix/B/ep14": {"it": 81000, "tree": "b39b69b",
                     "model": "6c16cf94e78fffcc45c7c3717da704f0a99b80b1f96d57612db70cba1887a8ae",
                     "rsi": "83323fa9f4818f6c2dfd44876f1ae2b6d9f3df742c14ae32e9592323df51a4f5",
                     "cfg": dict(zip(_HY, ("1303589055be652b4a00de15da294a3c5513bc55d9b63c1c68bd1283cf36d2bb",
                                           "ab24efde436fc26f3126094519d27dbe14eb8589e1dcd38c66aa01f4c6c6b1cd"))),
                     "hull_before": "stock (T1fix/stage2/fc_cache)"},
    "T1fix/W2PD/ep28": {"it": 68000, "tree": "7fbc227",
                        "model": "b674e5e5ff931672435c85f11e3e42f5ee37968fd3a6774a592d51d1748881fc",
                        "rsi": "d74eb9563407eaed31601a79a9d1a2cb696b0a4bc26483e36e19aeb90b5c944b",
                        "cfg": dict(zip(_HY, ("45bab9da23d3ab9054b28f79983e66127790d7c21f671e8dc4d2b07627606cd2",
                                              "9b933924ff086323ed2eb981fa0840d1f0298a35e92bcf0749468f131923d19d"))),
                        "hull_before": "stock (T1fix/ep28w2/fc_cache)"},
    "C3r2_runin/ep21": {"it": 119500, "tree": "b39b69b",
                        "model": "ca4801dcd3c2838e127c66803886647c055fc2427adc2db03a5113650ad8c6cf",
                        "rsi": "be1bd5ffbadd25ee9f4be0b03ccd1d43839f3584ffce315d8c549d4590ceab5f",
                        "cfg": dict(zip(_HY, ("b618a4652f342bf8f8313a89fcc25e71737846b4b9158b2f4b515175b46098ac",
                                              "8e9b6f915cbc716bcf314b8c69f6d0db99a7b92567b8963f8525627ea4f0415b"))),
                        "hull_before": "tree (nv-v2d_runin: 21 hulls, same set as stock, other bytes)"},
    "T1fix/B/ep10": {"it": 36432, "tree": "b39b69b",
                     "model": "dc34e3009039d22aa4e11d83dd04fb27d8a1227ae1a0b177778b4afd9952e442",
                     "rsi": "25da0cfecc246573ac611220eada3f3bdeb1b7bf7557651535846b15c5d64bce",
                     "cfg": dict(zip(_HY, ("b98ababbac1d17b5dc02a27ff9780fa18a1a1cbc44573cf33b7989ea377b76d8",
                                           "60627db1ac17e5f284a94c3ad9fdc634897a327af67d022c44e4ed9888a11957"))),
                     "hull_before": "tree (nv-v2d_runin: 18 hulls vs stock 17)"},
    "T1fix/B/ep19": {"it": 110000, "tree": "b39b69b",
                     "model": "a433b497254130be426f5ae16afd31f87d4983e7b02ca86d7f15478312b158fc",
                     "rsi": "37dc23ed02988522c298c04722692e681e03a980e7af4facbb1276ee7b4e218e",
                     "cfg": dict(zip(_HY, ("6883df5ae868d9aa5803e1f24de93e14ebc64196766a37396109d00240c93cff",
                                           "118b645d95695b073f4734095bd2ef0fc037b3debeee07bebc06a894fed1cb01"))),
                     "hull_before": "tree (nv-v2d_runin: 11 hulls vs stock 10)"},
    "T1fix/B/ep01": {"it": 142000, "tree": "b39b69b",
                     "model": "774fe75ff731f8ffb59325815ceb377d3f9dbf0ebc5a2f48ef7495e9782f1603",
                     "rsi": "1b666e070d9401ed2dd7e3911d2755c2983999eaf901c14f67f6bb2cfa94b440",
                     "cfg": dict(zip(_HY, ("db4e6da7ce0598e11e7c22be51fe26b70a0ce025fb44da348113bbc452a7c44a",
                                           "218f35fcc686cd2faec3c1cbe8c044d12e30ccfd45b40ab17a0309048035c960"))),
                     "hull_before": "stock (T1fix/stage2/fc_cache)"},
    "C3r2_wide/ep15": {"it": 20974, "tree": "61fa5d3",
                       "model": "16e14bcd737120d0dccb7acc5f5454daf90fd63024eef91264655d98d59f8af7",
                       "rsi": "8bffcbd190bad0299b787b8b50afe857eb0f6006725f66d67285fb1ace567987",
                       "cfg": {"wandb/run-20261012_024833-xg9j2z2a/files/config.yaml":
                               "c86c7f58724bc9de5ed8518e45f1927d7bc3f3ee7fe88274cce0413681c317ba",
                               "wandb/run-20261012_024833-xg9j2z2a/files/wandb-metadata.json":
                               "e7256600d0aec23a888e8384e096af51b2cadd5787773643ce297dcb5877691c"},
                       "hull_before": "tree (fc_r2: 50 hulls vs stock 49)"},
    "C3r2_runin/ep05": {"it": 42500, "tree": "b39b69b",
                        "model": "37371dfcb760cbb15db5c2cc9ca6a566c05cee6d2f3874c08653a4f852ab6704",
                        "rsi": "a2f6a2bb9f55ba7a3e1c16bfdbc87e2fdb0c9a63d7b7901eae12b0a4cba7a60e",
                        "cfg": dict(zip(_HY, ("14af26a77d78db5eec9fc5dd0ee3db52aa7c47ef6df9f44b967d0fa5c4491954",
                                              "92b1b6ed03a1d22d32e6ecd09df6289a3a886399fe413d1b6663ffa8f7c58ee0"))),
                        "hull_before": "tree (nv-v2d_runin: 52 hulls vs stock 53)"},
    "C3r2_runin/ep25": {"it": 102000, "tree": "b39b69b",
                        "model": "ffac8b771ef47e5ad1771d076af16ea55934fdf5017196cbbb4d2a7ce129d39b",
                        "rsi": "1bc0ecc6c3b9f76406708e28ab04c491738d06781e9c82569e8089815abb47de",
                        "cfg": dict(zip(_HY, ("69e47dda87f05555e0f945a76b54c4fa63f213ca7f7c86d982a890c7a94ec513",
                                              "4f2d828eb3695e9ae257c8eb14c72c3534a3a85aee4a222f1d99b9086cbe018d"))),
                        "hull_before": "tree (nv-v2d_runin: 8 hulls, same set as stock, other bytes)"},
}
EP15_ARGS = ["experiment=g1_recon_body_coord_rsi_r2",
             f"task.parquet='{SOMA}/sequence_id=2026-05-06_11-06-15_wet_floor_sign_pick_place_ground_08/robot_name=g1/data.parquet'",
             "reset.seed=42", "training.max_iterations=45000", "scene.support=false"]  # wandb-metadata args minus output/logging

BETA = "objective.object_task_rotation_beta"
EXP = "g1_recon_body_coord_rsi_r2"
EXP_RUNIN = "g1_recon_body_coord_rsi_r2_runin"
RELAX_PALM = ["termination.palm_position_threshold=1.2", "termination.palm_orientation_threshold=3.2"]
W2PD = [f"{BETA}=4.5", "termination.object_orientation_threshold=3.2", "termination.object_position_threshold=0.20",
        "+objective.object_task_position_only_after_release=true", "+objective.no_drop_weight=-50.0",
        "+curriculum.stages.0.objective_weights.object_no_drop=-50.0"]


def _arm(name, ep, item, tree, src, fork, target, exp, support, inherit, extra, role, why):
    return {"name": name, "ep": ep, "item": item, "tree": tree, "src": src, "fork": fork, "target": target, "exp": exp,
            "support": support, "seed": 42, "inherit": inherit, "extra": extra, "role": role, "why": why}


ARMS = [
    _arm("ep14-R", 14, "ep14", "b39b69b", "T1fix/B/ep14", 81000, 96000, EXP, False, [f"{BETA}=2.6"],
         RELAX_PALM + ["termination.pelvis_position_threshold=1.2"], "main",
         "final flip stalls ~22 deg short at the tipping point; training-only palm/pelvis termination relaxation"),
    _arm("ep28-W3H", 28, "ep28alt", "7fbc227", "T1fix/W2PD/ep28", 68000, 83000, EXP, True, list(W2PD), list(RELAX_PALM),
         "main", "hand withdraws early, squeegee slides off the chair; training-only palm termination relaxation"),
    _arm("ep21-B5", 21, "ep21", "b39b69b", "C3r2_runin/ep21", 119500, 139500, EXP_RUNIN, False, [], [f"{BETA}=5.0"],
         "main", "stool released tilted on one leg; rotation weight 5"),
    _arm("ep10-Bc", 10, "ep10", "b39b69b", "T1fix/B/ep10", 36432, 60000, EXP, False, [f"{BETA}=4.5"], [], "main",
         "ep10-B still rising (+0.043 / 10k) when its +15k budget ended; plain continuation"),
    _arm("ep10-BT", 10, "ep10", "b39b69b", "T1fix/B/ep10", 36432, 60000, EXP, False, [f"{BETA}=4.5"],
         ["termination.object_orientation_threshold=0.6"], "main", "ep10-Bc plus a 0.6 rad object-orientation termination"),
    _arm("ep19-Bc", 19, "ep19", "b39b69b", "T1fix/B/ep19", 110000, 135000, EXP_RUNIN, False, [f"{BETA}=3.0"], [], "main",
         "ep19-B still rising (+0.021 / 10k); plain continuation"),
    _arm("ep01-Bc", 1, "ep01", "b39b69b", "T1fix/B/ep01", 142000, 162000, EXP_RUNIN, True, [f"{BETA}=3.0"], [], "main",
         "ep01-B still rising (+0.018 / 10k); continuation (143000 is not used: its export failed 1e-4)"),
    _arm("ep15-B2", 15, "ep15", "61fa5d3", "C3r2_wide/ep15", 20974, 46000, EXP, False, [], [f"{BETA}=2.0"], "main",
         "cut by help_zero_tail at 20974 and never continued; rotation weight 2"),
    _arm("ep05-B3", 5, "ep05", "b39b69b", "C3r2_runin/ep05", 42500, 62500, EXP_RUNIN, False, [], [f"{BETA}=3.1"], "filler",
         "filler: rotation weight 3.1"),
    _arm("ep25-B25", 25, "ep25", "b39b69b", "C3r2_runin/ep25", 102000, 122000, EXP_RUNIN, True, [], [f"{BETA}=2.5"],
         "filler", "filler: rotation weight 2.5"),
]
BY_NAME = {a["name"]: a for a in ARMS}
ORDER = [a["name"] for a in ARMS]
_CONSENT = ("the user's explicit OK of the training-only palm/pelvis termination relaxation (PLAN section 9 item 3), "
            "verbatim")
_SLOT_OK = ("the main session's confirmation that no Tier-3 resume / revision and no Tier-2 job of Zuxing needs the "
            "slot (filler)")
RELEASE_NEEDED = {"ep14-R": _CONSENT, "ep28-W3H": _CONSENT, "ep05-B3": _SLOT_OK, "ep25-B25": _SLOT_OK}  # PREREG v2
STD_KEYS = {"experiment", "task.parquet", "scene.support", "reset.seed", "output_dir", "hydra.run.dir", "logging.mode",
            "logging.local_jsonl", "training.early_stop_help_zero_tail", "training.max_iterations", "training.resume",
            "training.checkpoint"}
ALWAYS_DIFF = {"output_dir", "logging.mode", "logging.local_jsonl", "logging.entity", "logging.project",
               "logging.experiment", "logging.run_name", "logging.upload_checkpoints", "training.max_iterations",
               "training.resume", "training.checkpoint", "training.early_stop_help_zero_tail"}
TERM_KEYS = ("pelvis_position_threshold", "pelvis_orientation_threshold", "palm_position_threshold",
             "palm_orientation_threshold", "object_position_threshold", "object_orientation_threshold")


# ============================================================================================== small helpers
def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ts(text: str) -> float | None:
    try:
        return dt.datetime.strptime(text, "%Y-%m-%d %H:%M:%S").timestamp()
    except (TypeError, ValueError):
        return None


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_new(path: Path, text: str) -> str:
    """Create path with text; an existing identical file is kept, a differing one is never overwritten."""
    if path.is_symlink() or path.exists():
        try:
            if path.read_text() == text:
                return f"kept {path}"
        except OSError:
            pass
        raise SystemExit(f"{path} exists with different content: not overwritten (inspect it by hand)")
    write_atomic(path, text)
    return f"wrote {path}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def node(gpu: int) -> int:
    return 0 if gpu <= 2 else 1


def ep2(ep: int) -> str:
    return f"ep{ep:02d}"


def arm_root(a: dict) -> Path:
    return ARMS_DIR / a["name"]


def run_dir(a: dict) -> Path:
    return arm_root(a) / "runs" / ep2(a["ep"])


def eval_root(a: dict) -> Path:
    return arm_root(a) / "eval"


def fork_dir(a: dict) -> Path:
    return arm_root(a) / "fork"


def cache_root(a: dict) -> Path:
    return arm_root(a) / "fc_cache"


def src_dir(a: dict) -> Path:
    return FC / a["src"]


def pair_files(d: Path, n: int) -> tuple[Path, Path]:
    return d / f"model_{n}.safetensors", d / f"rsi_{n}.safetensors"


def parquet_of(a: dict) -> str:
    return f"{SOMA}/sequence_id={EPISODES[a['ep']][0]}/robot_name=g1/data.parquet"


def override_key(o: str) -> str:
    return o.split("=", 1)[0].lstrip("+~")


def merge_overrides(base: list[str], extra: list[str]) -> list[str]:
    keys = {override_key(o) for o in extra}
    return [o for o in base if override_key(o) not in keys] + list(extra)


def overrides(a: dict) -> list[str]:
    return merge_overrides(a["inherit"], a["extra"])


def same_value(got, want: str) -> bool:
    w = want.strip("'\"")
    if isinstance(got, bool) or w.lower() in ("true", "false"):
        return str(got).lower() == w.lower()
    try:
        return abs(float(got) - float(w)) < 1e-12
    except (TypeError, ValueError):
        return str(got) == w


def tree_of(a: dict) -> dict:
    return TREES[a["tree"]]


def watcher_sha(tree: str) -> str | None:
    if tree == "61fa5d3":
        try:
            return sha_text(watcher_r2_text())
        except (OSError, SystemExit):
            return None
    return TREES[tree]["watcher_sha"]


class Log:
    def __init__(self, write: bool):
        self.write = write

    def __call__(self, msg: str) -> None:
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        if self.write:
            T1I.mkdir(parents=True, exist_ok=True)
            with (T1I / "launcher.log").open("a") as fp:
                fp.write(line + "\n")


def load_settings() -> dict:
    s = dict(SETTINGS)
    try:
        for line in (QDIR / "settings.conf").read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                k, sep, v = line.partition("=")
                if sep and k.strip() in SETTINGS:
                    s[k.strip()] = v.strip()
    except OSError:
        pass
    try:
        out = {k: int(v) for k, v in s.items()}
    except ValueError as error:
        raise SystemExit(f"bad settings {s} ({error})") from None
    if out["CORES"] < 1 or out["JOB_MB"] < 1024 or out["POLL_S"] < 10 or out["RESERVE_SLOTS"] < 0 or out["HOLDOFF_S"] < 0:
        raise SystemExit(f"bad settings {s}")
    return out


def gpus_conf() -> dict[int, int]:
    try:
        text = re.sub(r"#.*", "", (QDIR / "gpus.conf").read_text())
    except OSError:
        text = DEFAULT_GPUS_CONF
    conf = {int(m.group(1)): int(m.group(2)) for m in re.finditer(r"\b([0-7]):([0-9])\b", text)}
    return {g: min(3, n) for g, n in conf.items() if g not in NEVER_GPUS}


def prereg_ok() -> tuple[bool, str]:
    """PREREG sha matches its .sha256 file AND names this launcher's, the wrapper's and the r2 watcher copy's sha256."""
    try:
        want = PREREG_SHA.read_text().split()[0]
        got = sha256(PREREG)
        text = PREREG.read_text()
    except (OSError, IndexError) as error:
        return False, f"{PREREG} / {PREREG_SHA} unreadable ({error.__class__.__name__})"
    if got != want:
        return False, f"{PREREG.name} sha256 {got[:16]}... DIFFERS FROM {PREREG_SHA.name} {want[:16]}..."
    missing = []
    for label, path in (("launcher", Path(__file__).resolve()), ("wrapper", WRAPPER), ("r2 watcher", WATCHER_R2)):
        try:
            h = sha256(path)
        except OSError:
            missing.append(f"{label} unreadable")
            continue
        if h not in text:
            missing.append(f"{label} sha256 {h[:16]}... not named in the PREREG")
    if missing:
        return False, "; ".join(missing)
    return True, f"{PREREG.name} sha256 {got[:16]}... matches; it names the launcher, wrapper and r2 watcher sha256"


# ============================================================================================== sources and copies
def source_overrides(a: dict) -> list[str]:
    """The source run's own hydra overrides (no hydra.*), from the staged copy if present, else the original."""
    if a["src"] == "C3r2_wide/ep15":  # no saved hydra dir: the run's own argv from its (sha-pinned) wandb metadata
        meta = "wandb/run-20261012_024833-xg9j2z2a/files/wandb-metadata.json"
        for base in (arm_root(a) / "source", src_dir(a)):
            p = base / meta
            if p.is_file():
                if sha256(p) != SOURCES[a["src"]]["cfg"][meta]:
                    raise SystemExit(f"{p}: sha256 != pin")
                args = [str(x) for x in (read_json(p) or {}).get("args", [])]
                missing = [x for x in EP15_ARGS if x not in args]
                if missing:
                    raise SystemExit(f"{p}: args lack {missing}")
                return list(EP15_ARGS)
        raise SystemExit(f"{a['name']}: no wandb-metadata.json of {a['src']}")
    from omegaconf import OmegaConf

    for base in (arm_root(a) / "source", src_dir(a)):
        p = base / "hydra/.hydra/overrides.yaml"
        if p.is_file():
            if sha256(p) != SOURCES[a["src"]]["cfg"]["hydra/.hydra/overrides.yaml"]:
                raise SystemExit(f"{p}: sha256 != pin")
            return [str(x) for x in OmegaConf.to_container(OmegaConf.load(p)) if not str(x).startswith("hydra.")]
    raise SystemExit(f"{a['name']}: no saved overrides.yaml of {a['src']}")


def source_problems(a: dict) -> list[str]:
    """The arm's experiment / parquet / support / seed / inherited overrides against the source's saved overrides."""
    ov = source_overrides(a)
    vals = {override_key(o): o.split("=", 1)[1] for o in ov}
    p = []
    if vals.get("experiment") != a["exp"]:
        p.append(f"source experiment {vals.get('experiment')} != {a['exp']}")
    pq = vals.get("task.parquet", "").strip("'\"")
    if pq != parquet_of(a) or not Path(pq).is_file():
        p.append(f"source task.parquet {pq} != the Tier-1 G1 reference {parquet_of(a)} (or missing)")
    if vals.get("scene.support", "").lower() != str(a["support"]).lower():
        p.append(f"source scene.support {vals.get('scene.support')} != {a['support']}")
    if vals.get("reset.seed") != str(a["seed"]):
        p.append(f"source reset.seed {vals.get('reset.seed')} != {a['seed']}")
    for o in a["inherit"]:
        if o not in ov:
            p.append(f"inherited override {o} is not among the source's overrides")
    for o in ov:
        k = override_key(o)
        if k not in STD_KEYS and k != "hydra.run.dir" and o not in a["inherit"] and not k.startswith("logging."):
            p.append(f"source override {o} is neither standard nor inherited")
    return p


def copy_verified(src: Path, dst: Path, expect: str | None = None) -> dict:
    st1 = src.stat()
    h_src = sha256(src)
    if expect and h_src != expect:
        raise SystemExit(f"{src}: sha256 {h_src} != pinned {expect}")
    tmp = dst.with_name(f".{dst.name}.part")
    tmp.unlink(missing_ok=True)
    shutil.copyfile(src, tmp)
    os.chmod(tmp, 0o444)
    h_dst = sha256(tmp)
    st2 = src.stat()
    if h_dst != h_src or (st1.st_size, st1.st_mtime_ns) != (st2.st_size, st2.st_mtime_ns):
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"copy {src} -> {dst} not verified (source changed during the copy?)")
    if dst.exists():
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"{dst} appeared during the copy: not overwritten")
    os.replace(tmp, dst)
    return {"source": str(src), "dest": str(dst), "sha256": h_src, "size": st1.st_size, "copied_at": now()}


def ensure_copy(src: Path, dst: Path, expect: str, log: Log) -> dict:
    """dst must end up byte-identical to the pinned sha256: an existing dst is checked, never overwritten."""
    if dst.exists():
        got = sha256(dst)
        if got != expect:
            raise SystemExit(f"{dst} exists with sha256 {got[:16]}... != pinned {expect[:16]}...: not overwritten")
        return {"dest": str(dst), "sha256": got, "note": "already present, verified"}
    dst.parent.mkdir(parents=True, exist_ok=True)
    rec = copy_verified(src, dst, expect)
    log(f"copied {src} -> {dst} (sha256 {expect[:16]}... verified)")
    return rec


def stock_listing() -> dict[str, str]:
    return {p.name: sha256(p) for p in sorted(STOCK_CACHE.glob("*.npz"))}


def listing_digest(listing: dict[str, str]) -> str:
    return sha_text("".join(f"{h}  {n}\n" for n, h in sorted(listing.items())))


def stock_ok() -> tuple[bool, str, dict]:
    lst = stock_listing()
    d = listing_digest(lst)
    ok = len(lst) == STOCK_CACHE_N and d == STOCK_CACHE_DIGEST
    return ok, f"{STOCK_CACHE}: {len(lst)} hull files, listing digest {d[:16]}... {'== pin' if ok else '!= PIN'}", lst


def prepare_arm(a: dict, log: Log, stock: dict[str, str]) -> dict:
    s = SOURCES[a["src"]]
    n = a["fork"]
    if s["it"] != n:
        raise SystemExit(f"{a['name']}: fork iteration {n} != source pin {s['it']}")
    root = arm_root(a)
    rec_path = root / "fork.json"
    rec = read_json(rec_path) or {}
    if rec and (rec.get("src") != a["src"] or rec.get("fork") != n):
        raise SystemExit(f"{root} belongs to another fork ({rec.get('src')} @ {rec.get('fork')})")
    files = {}
    for kind, src, dst in zip(("model", "rsi"), pair_files(src_dir(a), n), pair_files(fork_dir(a), n)):
        files[kind] = ensure_copy(src, dst, s[kind], log)
    cfgs = {name: ensure_copy(src_dir(a) / name, root / "source" / name, pin, log) for name, pin in s["cfg"].items()}
    key, hsha = EPISODES[a["ep"]][2:]
    if stock.get(f"{key}.npz") != hsha:
        raise SystemExit(f"{a['name']}: stock hull {key}.npz sha256 {stock.get(f'{key}.npz')} != pin {hsha}")
    cache = {}
    for name, h in stock.items():
        cache[name] = ensure_copy(STOCK_CACHE / name, cache_root(a) / "convex" / name, h, log)["sha256"]
    run = run_dir(a)
    if run.is_symlink():
        raise SystemExit(f"{run} is a symlink: refusing")
    run.mkdir(parents=True, exist_ok=True)
    busy = [p.name for p in run.iterdir() if p.name.startswith(("model_", "rsi_")) or p.name in ("final.json", "metrics.jsonl")]
    if busy and not rec.get("launched"):
        raise SystemExit(f"{a['name']}: {run} already holds {busy[:5]}: not a fresh run dir")
    eval_root(a).mkdir(parents=True, exist_ok=True)
    rec.update({"arm": a["name"], "episode": a["ep"], "item": a["item"], "tree": a["tree"], "code": str(tree_of(a)["code"]),
                "src": a["src"], "source_dir": str(src_dir(a)), "fork": n, "max_iterations": a["target"],
                "experiment": a["exp"], "scene_support": a["support"], "reset_seed": a["seed"],
                "inherited_overrides": a["inherit"], "added_overrides": a["extra"], "overrides": overrides(a),
                "fork_pair": files, "source_config": cfgs, "hull_cache": str(cache_root(a)),
                "hull_object_key": key, "hull_files": len(cache), "hull_before": s["hull_before"],
                "prepared_at": rec.get("prepared_at") or now()})
    write_atomic(rec_path, json.dumps(rec, indent=1) + "\n")
    return rec


def fork_ok(a: dict) -> tuple[bool, str]:
    s = SOURCES[a["src"]]
    probs = []
    for kind, p in zip(("model", "rsi"), pair_files(fork_dir(a), a["fork"])):
        if not p.is_file():
            probs.append(f"{p.name} missing")
        elif sha256(p) != s[kind]:
            probs.append(f"{p.name} sha256 != pin")
    return not probs, "; ".join(probs) or f"{fork_dir(a)}: model/rsi_{a['fork']} == pins"


def cache_ok(a: dict, stock: dict[str, str]) -> tuple[bool, str]:
    d = cache_root(a) / "convex"
    try:
        names = sorted(p.name for p in d.iterdir())
    except OSError:
        return False, f"{d} missing"
    extra = [n for n in names if n not in stock]
    probs = [f"missing {n}" for n in stock if n not in names]
    probs += [f"{n} sha256 != stock" for n in names if n in stock and sha256(d / n) != stock[n]]
    if extra:
        probs.append(f"NEW file(s) {extra}: a trainer built a hull the stock cache does not have")
    return not probs, "; ".join(probs) or f"{d}: exactly the {len(stock)} stock hull files (object key {EPISODES[a['ep']][2]})"


def safetensors_keys(path: Path) -> set[str]:
    with open(path, "rb") as fp:
        n = struct.unpack("<Q", fp.read(8))[0]
        return {k for k in json.loads(fp.read(n)) if k != "__metadata__"}


# ============================================================================================== watcher copy (fc_r2)
def watcher_r2_text() -> str:
    if sha256(WATCHER_ORIG) != WATCHER_ORIG_SHA:
        raise SystemExit(f"{WATCHER_ORIG} sha256 != pin (it changed?)")
    text = WATCHER_ORIG.read_text()
    if text.count(WATCHER_LINE_OLD + "\n") != 1:
        raise SystemExit(f"{WATCHER_ORIG}: the CODE line is not unique")
    return text.replace(WATCHER_LINE_OLD + "\n", WATCHER_LINE_R2 + "\n", 1)


def watcher_ok(tree: str) -> tuple[bool, str]:
    t = TREES[tree]
    try:
        got = sha256(t["watcher"])
    except OSError:
        return False, f"{t['watcher']} missing (run init)"
    want = watcher_sha(tree)
    ok = want is not None and got == want
    return ok, f"{t['watcher'].name} sha256 {got[:16]}... {'== expected' if ok else '!= EXPECTED ' + str(want)[:16]}"


# ============================================================================================== commands
def train_argv(a: dict, cpus: str) -> list[str]:
    run = run_dir(a)
    model, _ = pair_files(fork_dir(a), a["fork"])
    return ["taskset", "-c", cpus, str(PY), "scripts/train_rl.py", f"experiment={a['exp']}",
            f"task.parquet='{parquet_of(a)}'", f"scene.support={str(a['support']).lower()}", f"reset.seed={a['seed']}",
            f"output_dir={run}", f"hydra.run.dir={run}/hydra", "logging.mode=disabled", "+logging.local_jsonl=true",
            "training.early_stop_help_zero_tail=0", f"training.max_iterations={a['target']}", "training.resume=true",
            f"training.checkpoint={model}", *overrides(a)]


def train_env_updates(a: dict, gpu: int, nth: int) -> dict:
    sp = ME / "conda_envs/flashchord/lib/python3.11/site-packages/nvidia"
    return {"LD_LIBRARY_PATH": f"{sp}/cu13/lib:{sp}/cudnn/lib:{sp}/nccl/lib:{sp}/nvshmem/lib",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(tree_of(a)["code"] / "src"), "PYTHONUNBUFFERED": "1", "HUMAN_MOTION_DATA_DIR": str(HMD),
            "WANDB_MODE": "disabled", "WANDB_DIR": str(ME / "fc_runs/wandb_local"), "XDG_CACHE_HOME": str(ME / ".cache"),
            "PYTHONPYCACHEPREFIX": str(ME / ".cache/pycache"), "TZ": "Asia/Shanghai", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "RUNIN_QUEUE": str(QDIR), "RUNIN_GPU": str(gpu), "CUDA_VISIBLE_DEVICES": str(gpu),
            "OMP_NUM_THREADS": str(nth), "MKL_NUM_THREADS": str(nth), "OPENBLAS_NUM_THREADS": str(nth),
            "FLASH_CHORD_CACHE_DIR": str(cache_root(a)), "T1FIX_ARM": a["name"], "T1FIX_STAGE": "t1improve",
            "T1IMPROVE_ARM": a["name"]}


DROP_ENV = ("CUDA_MPS_", "WANDB_", "RUNIN_", "TIER2V2_", "TIER2_", "K3EASY", "K3FIX", "K3FULL", "T1FIX_", "T2FIX_",
            "FLASH_CHORD_", "T1IMPROVE_", "T3F0")
DROP_KEYS = {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "OPENBLAS_NUM_THREADS", "PYTHONINSPECT", "PYTHONNOUSERSITE"}


def clean_env() -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(DROP_ENV) and k not in DROP_KEYS}
    env["PATH"] = f"{ME}/conda_envs/flashchord/bin:{env.get('PATH', '/usr/bin:/bin')}"
    return env


def shown_command(a: dict, gpu, cpus, nth) -> str:
    upd = train_env_updates(a, gpu, nth)
    envs = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in upd.items())
    return (f"( cd {tree_of(a)['code']} && exec env PATH={ME}/conda_envs/flashchord/bin:$PATH {envs} "
            f"{shlex.join(train_argv(a, str(cpus)))} ) >> {arm_root(a) / 'train.log'} 2>&1   "
            f"# setsid, nice 0; unset first: CUDA_MPS_* WANDB_* FLASH_CHORD_* RUNIN_* PYTHONHOME PYTHONSTARTUP")


def watcher_args(a: dict) -> list[str]:
    return ["--episodes", str(a["ep"]), "--every", str(EVERY), "--no-early-stop", "--lane-max", "1"]


def watcher_env_updates(a: dict) -> dict:
    return {"RUNIN_QDIR": str(QDIR), "RUNIN_RUNS": str(arm_root(a) / "runs"), "RUNIN_EVAL_ROOT": str(eval_root(a)),
            "RUNIN_WATCH_ITERS": str(a["target"]), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
            "TZ": "Asia/Shanghai"}


def watcher_command(a: dict, dry: bool = False) -> str:
    envs = " ".join(f"{k}={shlex.quote(v)}" for k, v in watcher_env_updates(a).items())
    w = tree_of(a)["watcher"]
    tail = " --dry-run" if dry else f" >> {eval_root(a)}/watcher_stdout.log 2>&1 < /dev/null &"
    return f"cd {ME} && env {envs} {'' if dry else 'setsid nohup '}{PY} -B {w} {shlex.join(watcher_args(a))}{tail}"


# ============================================================================================== processes / GPUs
def output_dir_of(cmd: list[str]) -> str:
    return next((x.split("=", 1)[1] for x in cmd if x.startswith("output_dir=")), "").strip("'\"")


def our_trainers() -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = {}
    for pid in T2H.all_pids():
        cmd = T2H.read_cmdline(pid)
        if not T2H.is_python_trainer(cmd) or not T2H.alive(pid):
            continue
        env = T2H.read_environ(pid)
        name = env.get("T1IMPROVE_ARM")
        if name not in BY_NAME or env.get("RUNIN_QUEUE") != str(QDIR):
            continue
        g = env.get("RUNIN_GPU", "")
        found.setdefault(name, []).append({"pid": pid, "gpu": int(g) if g.isdigit() else None,
                                           "age": T2H.proc_age(pid) or 0.0, "output_dir": output_dir_of(cmd)})
    return found


def our_watchers() -> dict[str, list[int]]:
    roots = {str(eval_root(a)): a["name"] for a in ARMS}
    scripts = {str(t["watcher"]) for t in TREES.values()}
    found: dict[str, list[int]] = {}
    for pid in T2H.all_pids():
        cmd = T2H.read_cmdline(pid)
        if len(cmd) < 3 or not any(x in scripts for x in cmd[1:3]) or not T2H.alive(pid):
            continue
        name = roots.get(T2H.read_environ(pid).get("RUNIN_EVAL_ROOT", ""))
        if name:
            found.setdefault(name, []).append(pid)
    return found


def observe(st: dict) -> dict:
    s = {"MAX_TRAIN_PER_GPU": 3, "JOB_MB": st["JOB_MB"], "MIN_FREE_AFTER_MB": st["MIN_FREE_AFTER_MB"],
         "GPUS": list(range(8)), "UNKNOWN_TRAIN_AGE_S": 1800}
    view, _ = T2H.observe(s, [], time.time())
    mem = {a["pid"]: a["mem"] for a in T2H.smi()["apps"]}
    for v in view.values():
        v["debt"], v["young"] = 0, []
        for pid in v["train_pids"]:
            if (T2H.proc_age(pid) or 0.0) < RAMP_S:
                v["debt"] += max(0, st["JOB_MB"] - mem.get(pid, 0))
                v["young"].append(pid)
    return view


def free_cores(nd: int, exclude: str) -> int:
    cpus, n, _ = T2H.cpu_slice(0 if nd == 0 else 5, exclude, 60, 0)
    return 0 if cpus == "none" else n // 2


def gpu_slots(v: dict, st: dict, extra: int) -> int:
    if not v.get("present") or v.get("mps"):
        return 0
    free = v["free"] - v["debt"] - extra * st["JOB_MB"]
    mem_slots = max(0, (free - st["MIN_FREE_AFTER_MB"]) // st["JOB_MB"])
    return max(0, min(3 - v["trainings"] - extra, mem_slots))


def feasible(view: dict, st: dict, conf: dict, assume: dict, exclude: str) -> tuple[int, dict]:
    per = {0: 0, 1: 0}
    for g in SHARED_GPUS:
        if g in view and g in conf and g not in NEVER_GPUS and (QDIR / "evallock" / f"gpu{g}.lock").exists():
            per[node(g)] += gpu_slots(view[g], st, assume.get(g, 0))
    cores = {nd: free_cores(nd, exclude) for nd in (0, 1)}
    by = {nd: min(per[nd], cores[nd] // st["CORES"]) for nd in (0, 1)}
    return sum(by.values()), {"gpu_slots": per, "free_cores": cores, "feasible": by}


def gpu_check(g: int, view: dict, st: dict, conf: dict, assume: dict, exclude: str):
    if g in NEVER_GPUS:
        return False, "never GPU 4", None, 0
    if g not in conf:
        return False, "not in queue_t1improve/gpus.conf", None, 0
    if not (QDIR / "evallock" / f"gpu{g}.lock").exists():
        return False, f"no {QDIR}/evallock/gpu{g}.lock (run init)", None, 0
    v = view.get(g)
    if not v or not v.get("present"):
        return False, "not listed by nvidia-smi", None, 0
    if v.get("mps"):
        return False, f"MPS ({v['mps']})", None, 0
    n, cap = v["trainings"] + assume.get(g, 0), min(3, conf[g])
    if n >= cap:
        return False, f"full: {n} training(s) of anybody >= cap {cap}", None, 0
    need = st["JOB_MB"] + st["MIN_FREE_AFTER_MB"] + v["debt"] + assume.get(g, 0) * st["JOB_MB"]
    if v["free"] < need:
        return False, f"memory: {v['free']} MiB free < {need}", None, 0
    cpus, nth, note = T2H.cpu_slice(g, exclude, st["CORES"], st["CORES"])
    if cpus == "none" or note or nth < 2 * st["CORES"]:
        return False, f"CPU: no {st['CORES']} free own-NUMA cores ({note or (str(nth // 2) + ' free')})", None, 0
    return True, f"{n} training(s) of anybody -> {n + 1} (cap {cap}); {v['free']} MiB free >= {need}", cpus, nth


def plan_one(name: str, view: dict, st: dict, conf: dict, assume: dict, exclude: str):
    rows = [(g, *gpu_check(g, view, st, conf, assume, exclude)) for g in sorted(conf)]
    cands = [r for r in rows if r[1]]
    if not cands:
        return None, f"{name}: no GPU ({'; '.join(f'gpu{r[0]} {r[2]}' for r in rows)})"
    g, _, why, cpus, nth = min(cands, key=lambda r: (view[r[0]]["trainings"] + assume.get(r[0], 0),
                                                     -(view[r[0]]["free"] - assume.get(r[0], 0) * st["JOB_MB"]), r[0]))
    trial = dict(assume)
    trial[g] = trial.get(g, 0) + 1
    excl2 = f"{exclude},{cpus}" if exclude else cpus
    f_after, _ = feasible(view, st, conf, trial, excl2)
    if f_after < st["RESERVE_SLOTS"]:
        return None, (f"{name}: after a start on gpu{g} {f_after} feasible slot(s) would remain < RESERVE_SLOTS "
                      f"{st['RESERVE_SLOTS']} (held for Tier 3)")
    return {"name": name, "gpu": g, "cpus": cpus, "nth": nth, "why": why, "feasible_after": f_after, "assume": trial,
            "exclude": excl2}, ""


# ============================================================================================== the yield gate
def last_heartbeat(path: Path, marker: str) -> tuple[float | None, str | None]:
    try:
        with open(path, "rb") as fp:
            fp.seek(0, 2)
            size = fp.tell()
            fp.seek(max(0, size - 400_000))
            lines = fp.read().decode(errors="replace").splitlines()
    except OSError:
        return None, None
    for line in reversed(lines):
        if marker in line:
            m = re.match(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]", line)
            return (ts(m.group(1)) if m else None), line
    return None, None


def driver_alive(pidfile: Path, name: str) -> tuple[bool, str]:
    try:
        pid = int(pidfile.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return False, f"{pidfile} unreadable"
    cmd = T2H.read_cmdline(pid)
    ok = T2H.alive(pid) and any(name in x for x in cmd[:4])
    return ok, f"pid {pid} {'alive' if ok else 'not running'} ({' '.join(cmd[:3])[:80]})"


def yield_gate(st: dict) -> dict:
    """Never start while Tier 3 waits for a slot (module doc, YIELD GATE (a)-(f)). Fail closed."""
    reasons, facts = [], []
    t = time.time()
    if (T1I / "PAUSE").exists():
        reasons.append(f"PAUSE file {T1I / 'PAUSE'}")
    w = read_json(T3F0 / "waiting.json")
    if not isinstance(w, dict) or not isinstance(w.get("waiting"), list):
        reasons.append(f"{T3F0 / 'waiting.json'} unreadable (fail closed)")
    else:
        facts.append(f"f0 waiting.json ({w.get('written')}): waiting {w['waiting'] or 'none'}, held "
                     f"{w.get('held_waiting') or 'none'}, all_launched {w.get('all_launched')}")
        if w["waiting"]:
            reasons.append(f"Tier-3 f0 clips waiting for a slot: {w['waiting']}")
    if not (T3F0 / "ALL_LAUNCHED").is_file():
        reasons.append(f"{T3F0 / 'ALL_LAUNCHED'} absent: a Tier-3 f0 clip is still FRESH")
    else:
        facts.append("f0/ALL_LAUNCHED present")
    alive, why = driver_alive(T3F0 / "driver.pid", "t3f0_driver")
    hb_t, hb = last_heartbeat(T3F0 / "heartbeat.log", "HEARTBEAT driver f0")
    if hb is None:
        if alive:
            reasons.append("no f0 HEARTBEAT line while the f0 driver runs (fail closed)")
        facts.append(f"f0 driver {why}; no heartbeat")
    else:
        states = dict(re.findall(r"\bep(\d\d) ([A-Z_]+)\b", hb))
        pre = {e: s for e, s in states.items() if s in T3_PRELAUNCH}
        running = sorted(e for e, s in states.items() if s in ("RUNNING", "STARTING"))
        age = t - hb_t if hb_t else float("inf")
        facts.append(f"f0 driver {why}; last heartbeat {age / 60:.0f} min ago: {len(states)} clips, running "
                     f"{running or 'none'}, not launched {pre or 'none'}")
        if pre:
            reasons.append(f"Tier-3 f0 clips not launched yet (last heartbeat): {pre}")
        dead = {e: s for e, s in states.items() if s in T3_NEEDS_SLOT and not (T1I / "t3ack" / f"ep{e}").exists()}
        if dead:
            reasons.append(f"Tier-3 f0 clips without a live trainer (last heartbeat): {dead}; a resume may need a slot "
                           f"(not resumed: write {T1I / 't3ack'}/epNN)")
        if alive and age > st["T3_STALE_S"]:
            reasons.append(f"f0 heartbeat {age / 60:.0f} min old while the driver runs (fail closed)")
    if (T3RETRY / "RETRY_GO").exists():
        rt, rl = last_heartbeat(T3RETRY / "heartbeat.log", "HEARTBEAT retry")
        if rl is None or rt is None or t - rt > st["T3_STALE_S"]:
            reasons.append("RETRY_GO present and no fresh Tier-3 retry heartbeat: a retry may be pending (fail closed)")
        else:
            rs = dict(re.findall(r"\bep(\d\d):([A-Z_]+)", rl))
            pend = {e: s for e, s in rs.items() if s in RETRY_PENDING}
            facts.append(f"RETRY_GO present; retry heartbeat {(t - rt) / 60:.0f} min ago: {rs or 'no candidate'}")
            if pend:
                reasons.append(f"Tier-3 retries pending: {pend}")
    else:
        facts.append(f"no RETRY_GO ({'RETRY_NOGO present' if (T3RETRY / 'RETRY_NOGO').exists() else 'no RETRY_NOGO'})")
    for f in sorted(T3ROOT.glob("*/waiting.json")):
        if f.parent.name == "f0":
            continue
        try:
            age = t - f.stat().st_mtime
        except OSError:
            continue
        if age > st["T3_STALE_S"]:
            facts.append(f"{f} {age / 3600:.1f} h old (ignored)")
            continue
        d = read_json(f)
        if not isinstance(d, dict) or not isinstance(d.get("waiting"), list):
            reasons.append(f"{f} (fresh) unreadable (fail closed)")
        elif d["waiting"]:
            reasons.append(f"Tier-3 clips waiting in {f}: {d['waiting']}")
    return {"ok": not reasons, "reasons": reasons, "facts": facts, "checked_at": now()}


# ============================================================================================== state
def release_file(a: dict) -> Path:
    return T1I / "release" / f"{a['name']}.txt"


def released(a: dict) -> tuple[bool, str]:
    """PREREG v2 RELEASE: an arm of RELEASE_NEEDED starts only once T1I/release/<arm>.txt exists and is non-empty."""
    need = RELEASE_NEEDED.get(a["name"])
    if need is None:
        return True, "no release needed"
    f = release_file(a)
    try:
        body = f.read_text().strip() if (f.is_file() and not f.is_symlink()) else ""
    except OSError:
        body = ""
    if not body:
        return False, f"HELD until {f} exists (non-empty): {need}"
    return True, f"released by {f}: {body.splitlines()[0][:160]}"


def launch_record(a: dict) -> dict | None:
    return read_json(arm_root(a) / "launch.json")


def final_of(a: dict) -> dict | None:
    d = read_json(run_dir(a) / "final.json")
    return d if isinstance(d, dict) and isinstance(d.get("iteration"), int) and d["iteration"] > a["fork"] else None


def newest_ckpt(run: Path) -> int | None:
    its = [int(m.group(1)) for p in run.glob("model_*.safetensors") if (m := re.fullmatch(r"model_(\d+)\.safetensors", p.name))]
    return max(its) if its else None


def start_failures(a: dict) -> int:
    try:
        return sum(1 for ln in (arm_root(a) / "start_failed").read_text().splitlines() if ln.strip())
    except OSError:
        return 0


def arm_state(a: dict, trainers: dict) -> tuple[str, str]:
    rec = launch_record(a)
    tr = trainers.get(a["name"], [])
    if len(tr) > 1:
        return "DUPLICATE", f"{len(tr)} trainers {[x['pid'] for x in tr]}"
    if tr:
        return "RUNNING", f"pid {tr[0]['pid']} gpu {tr[0]['gpu']}"
    if rec is None:
        if start_failures(a) >= START_FAIL_MAX:
            return "START_FAILED", f"trainer exited at once {start_failures(a)} times ({arm_root(a) / 'start_failed'})"
        ok, why = released(a)
        if not ok:
            return "HELD", why
        return "PENDING", "not launched"
    fin = final_of(a)
    if fin:
        return "ENDED", f"final.json {fin.get('reason')} at {fin.get('iteration')}"
    return "CRASHED", f"trainer gone without final.json (log {arm_root(a) / 'train.log'})"


def eval_rows(a: dict) -> list[dict]:
    try:
        rows = list(csv.DictReader((eval_root(a) / "evals.csv").open()))
    except OSError:
        return []
    return [r for r in rows if r.get("episode", "").strip().lstrip("0") == str(a["ep"]) or r.get("episode") == str(a["ep"])]


def incumbents() -> dict[str, dict]:
    out = {}
    try:
        for r in csv.DictReader(INCUMBENTS.open()):
            out[r["item"]] = r
    except OSError:
        pass
    return out


def last_launch_time() -> float | None:
    try:
        lines = [ln for ln in (T1I / "launches.tsv").read_text().splitlines() if ln.strip()]
    except OSError:
        return None
    return ts(lines[-1].split("\t")[0]) if lines else None


# ============================================================================================== locks
class GpuLocks:
    def __init__(self):
        self.fds = []

    def __enter__(self):
        for p in LOCKS:
            if not p.is_file():
                raise SystemExit(f"GPU-choice lock {p} missing: refusing to launch (never created here)")
            fd = os.open(p, os.O_RDONLY)
            t0 = time.time()
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.time() - t0 > LOCK_WAIT_S:
                        os.close(fd)
                        self.__exit__(None, None, None)
                        raise TimeoutError(f"GPU-choice lock {p} busy for > {LOCK_WAIT_S:.0f} s")
                    time.sleep(2)
            self.fds.append(fd)
        return self

    def __exit__(self, *exc):
        for fd in reversed(self.fds):
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            except OSError:
                pass
        self.fds = []
        return False


# ============================================================================================== launch
_LAST_WAIT: dict[str, str] = {}


def wait_note(log: Log, key: str, msg: str) -> None:
    sig = re.sub(r"\d+", "#", msg)
    if _LAST_WAIT.get(key) != sig:
        _LAST_WAIT[key] = sig
        log(msg)


def start_trainer(a: dict, p: dict, log: Log, stock: dict[str, str]) -> int:
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    if nice != 0:
        raise SystemExit(f"the launcher runs at nice {nice}: trainers would inherit it; run launch modes at nice 0")
    for ok, why in (fork_ok(a), cache_ok(a, stock)):
        if not ok:
            raise SystemExit(f"{a['name']}: {why}")
    argv = train_argv(a, p["cpus"])
    env = clean_env()
    env.update(train_env_updates(a, p["gpu"], p["nth"]))
    lp = arm_root(a) / "train.log"
    with lp.open("a") as fp:
        fp.write(f"[{now()}] t1improve_launch: {a['name']} on gpu {p['gpu']} cpus {p['cpus']}: {shlex.join(argv)}\n")
        fp.flush()
        proc = subprocess.Popen(argv, cwd=tree_of(a)["code"], env=env, stdin=subprocess.DEVNULL, stdout=fp,
                                stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    time.sleep(8)
    if proc.poll() is not None:
        with (arm_root(a) / "start_failed").open("a") as fp:
            fp.write(f"{now()}\tpid {proc.pid}\tcode {proc.returncode}\tgpu {p['gpu']}\tcpus {p['cpus']}\n")
        raise SystemExit(f"{a['name']}: trainer exited at once (code {proc.returncode}; failed start "
                         f"{start_failures(a)}/{START_FAIL_MAX}); see {lp}")
    rec = {"arm": a["name"], "episode": a["ep"], "item": a["item"], "pid": proc.pid, "gpu": p["gpu"], "cpus": p["cpus"],
           "nthreads": p["nth"], "tree": a["tree"], "cwd": str(tree_of(a)["code"]), "fork": a["fork"],
           "max_iterations": a["target"], "experiment": a["exp"], "overrides": overrides(a), "reset_seed": a["seed"],
           "output_dir": str(run_dir(a)), "argv": argv, "env_set": train_env_updates(a, p["gpu"], p["nth"]),
           "launched_at": now(), "plan_reason": p["why"], "launched": True}
    write_atomic(arm_root(a) / "launch.json", json.dumps(rec, indent=1) + "\n")
    fr = read_json(arm_root(a) / "fork.json") or {}
    fr["launched"] = True
    write_atomic(arm_root(a) / "fork.json", json.dumps(fr, indent=1) + "\n")
    with (T1I / "launches.tsv").open("a") as fp:
        fp.write("\t".join(map(str, (now(), a["name"], a["ep"], p["gpu"], p["cpus"], proc.pid, a["fork"], a["target"],
                                     a["tree"], " ".join(overrides(a))))) + "\n")
    log(f"LAUNCHED {a['name']}: pid {proc.pid} gpu {p['gpu']} cpus {p['cpus']} model_{a['fork']} -> {a['target']} "
        f"({a['tree']}, {a['exp']}; {' '.join(overrides(a)) or 'no override'}); log {lp}")
    return proc.pid


def start_watcher(a: dict, log: Log) -> int | None:
    ok, why = watcher_ok(a["tree"])
    if not ok:
        log(f"{a['name']}: watcher not started: {why}")
        return None
    env = clean_env()
    env.update(watcher_env_updates(a))
    eval_root(a).mkdir(parents=True, exist_ok=True)
    w = tree_of(a)["watcher"]
    with (eval_root(a) / "watcher_stdout.log").open("a") as fp:
        proc = subprocess.Popen([str(PY), "-B", str(w), *watcher_args(a)], cwd=ME, env=env, stdin=subprocess.DEVNULL,
                                stdout=fp, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    with (arm_root(a) / "watcher_starts.tsv").open("a") as fp:
        fp.write(f"{now()}\t{proc.pid}\t{w.name}\n")
    log(f"watcher {a['name']} started: pid {proc.pid} ({w.name} {' '.join(watcher_args(a))}; RUNIN_EVAL_ROOT {eval_root(a)})")
    return proc.pid


def watcher_starts(a: dict) -> int:
    try:
        return sum(1 for ln in (arm_root(a) / "watcher_starts.tsv").read_text().splitlines() if ln.strip())
    except OSError:
        return 0


def ensure_watchers(log: Log) -> None:
    ws = our_watchers()
    for a in ARMS:
        if launch_record(a) is None:
            continue
        if (eval_root(a) / ep2(a["ep"]) / "runin_final.json").is_file() or (eval_root(a) / "STOP").exists():
            continue
        if ws.get(a["name"]):
            continue
        if watcher_starts(a) >= 1 + WATCHER_RESTARTS:
            wait_note(log, f"w{a['name']}", f"{a['name']}: watcher gone {watcher_starts(a)} times: not restarted "
                                            f"(inspect {eval_root(a)}/watcher.log)")
            continue
        start_watcher(a, log)


def launch_pass(log: Log, stock: dict[str, str]) -> int:
    """At most ONE start: the first RELEASED arm not yet started (priority order), if the hold-off, the yield gate and
    the fill rule allow it."""
    ok, why = prereg_ok()
    if not ok:
        wait_note(log, "prereg", f"PREREG check failed: {why}; nothing is launched")
        return 0
    st, conf = load_settings(), gpus_conf()
    trainers = our_trainers()
    todo = [n for n in ORDER if launch_record(BY_NAME[n]) is None and n not in trainers
            and start_failures(BY_NAME[n]) < START_FAIL_MAX]
    held = [n for n in todo if not released(BY_NAME[n])[0]]
    if held:
        wait_note(log, "held", f"HELD until fc_runs/T1improve/release/<arm>.txt exists: {held}")
    todo = [n for n in todo if n not in held]
    if not todo:
        return 0
    last = last_launch_time()
    if last is not None and time.time() - last < st["HOLDOFF_S"]:
        wait_note(log, "holdoff", f"hold-off: last start {time.time() - last:.0f} s ago < {st['HOLDOFF_S']} s")
        return 0
    gate = yield_gate(st)
    if not gate["ok"]:
        wait_note(log, "gate", f"YIELD to Tier 3: {'; '.join(gate['reasons'])}")
        return 0
    name = todo[0]
    a = BY_NAME[name]
    prob = source_problems(a)
    if prob:
        raise SystemExit(f"{name}: {'; '.join(prob)}")
    view = observe(st)
    p, why = plan_one(name, view, st, conf, {}, "")
    if p is None:
        wait_note(log, "plan", f"{name} waits (head of the queue): {why}")
        return 0
    try:
        with GpuLocks():
            if launch_record(a) is not None or name in our_trainers():
                log(f"{name}: launched meanwhile by another launcher process: skipped")
                return 0
            if not released(a)[0]:
                log(f"{name}: no longer released: skipped")
                return 0
            st, conf = load_settings(), gpus_conf()
            gate = yield_gate(st)  # re-check under the locks
            if not gate["ok"]:
                wait_note(log, "gate", f"YIELD to Tier 3 (under the locks): {'; '.join(gate['reasons'])}")
                return 0
            view = observe(st)
            p, why = plan_one(name, view, st, conf, {}, "")
            if p is None:
                wait_note(log, "plan", f"{name} no longer fits under the locks: {why}")
                return 0
            _LAST_WAIT.clear()
            log(f"{name} slot decision: gpu{p['gpu']} cpus {p['cpus']}: {p['why']}; {p['feasible_after']} feasible "
                f"slot(s) after it (reserve {st['RESERVE_SLOTS']}); gate: {' | '.join(gate['facts'])}")
            start_trainer(a, p, log, stock)
            start_watcher(a, log)
            time.sleep(st["LAUNCH_HOLD_S"])  # hold the locks: the new job's memory shows up for every launcher
            return 1
    except TimeoutError as error:
        wait_note(log, "locks", f"{name}: {error}; retry next pass")
        return 0


# ============================================================================================== compose check
def flat(d, prefix="") -> dict:
    out = {}
    for k, v in (d.items() if isinstance(d, dict) else enumerate(d)):
        key = f"{prefix}{k}"
        if isinstance(v, (dict, list)) and v:
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


def compose_child(tree: str) -> int:
    """Runs with PYTHONPATH=<tree>/src, cwd=<tree> (no -I): composes every arm of that tree, compares with the source's
    composition (and its saved config), instantiates termination / objective / curriculum. Prints one JSON line."""
    import flash_chord
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    code = TREES[tree]["code"]
    res = {"tree": tree, "flash_chord_file": str(Path(flash_chord.__file__).resolve()), "arms": {}}
    res["import_ok"] = res["flash_chord_file"].startswith(str((code / "src").resolve()) + "/")
    with initialize_config_dir(config_dir=str(code / "src/flash_chord/configs"), version_base="1.3"):
        for a in [x for x in ARMS if x["tree"] == tree]:
            r: dict = {}
            try:
                src_ov = source_overrides(a)
                src_cfg = compose(config_name="train", overrides=src_ov)
                src_flat = flat(OmegaConf.to_container(src_cfg, resolve=False))
                saved = None
                for base in (arm_root(a) / "source", src_dir(a)):
                    if (base / "hydra/.hydra/config.yaml").is_file():
                        saved = flat(OmegaConf.to_container(OmegaConf.load(base / "hydra/.hydra/config.yaml"), resolve=False))
                        break
                if saved is not None:
                    r["source_changed"] = sorted(k for k in set(saved) & set(src_flat) if saved[k] != src_flat[k]
                                                 and k not in ALWAYS_DIFF and not k.startswith("hydra."))
                    r["source_one_side"] = sorted(k for k in set(saved) ^ set(src_flat) if not k.startswith("hydra."))
                else:
                    r["source_changed"], r["source_one_side"] = [], ["(no saved hydra config: C3r2_wide wandb run)"]
                argv = [x for x in train_argv(a, "0")[5:] if not x.startswith("hydra.")]
                cfg = compose(config_name="train", overrides=argv)
                arm_flat = flat(OmegaConf.to_container(cfg, resolve=False))
                diff = {k: [src_flat.get(k, "<absent>"), arm_flat.get(k, "<absent>")]
                        for k in set(src_flat) | set(arm_flat) if src_flat.get(k, "<absent>") != arm_flat.get(k, "<absent>")}
                allowed = ALWAYS_DIFF | {override_key(o) for o in overrides(a)}
                r["diff_vs_source"] = {k: diff[k] for k in sorted(diff)}
                r["unexpected"] = sorted(k for k in diff if k not in allowed and not k.startswith("hydra."))
                want = {override_key(o): o.split("=", 1)[1] for o in overrides(a)}
                want.update({"training.max_iterations": str(a["target"]), "training.resume": "true",
                             "training.checkpoint": str(pair_files(fork_dir(a), a["fork"])[0]),
                             "training.early_stop_help_zero_tail": "0", "scene.support": str(a["support"]).lower(),
                             "reset.seed": str(a["seed"]), "task.parquet": parquet_of(a), "output_dir": str(run_dir(a))})
                r["values"] = {k: arm_flat.get(k, "<absent>") for k in want}
                r["values_bad"] = sorted(k for k, v in want.items() if not same_value(arm_flat.get(k, "<absent>"), v))
                r["training_seed"] = [arm_flat.get("training.seed"), OmegaConf.select(cfg, "training.seed")]
                r["env_refs"] = {k: arm_flat.get(k, "<absent>") for k in ("env.termination", "env.objective",
                                                                          "training.curriculum")}
                src_term = instantiate(src_cfg.termination)
                term = instantiate(cfg.termination)
                exp_term = {k: getattr(src_term, k, None) for k in TERM_KEYS}
                for o in overrides(a):
                    k = override_key(o)
                    if k.startswith("termination.") and k.split(".", 1)[1] in TERM_KEYS:
                        exp_term[k.split(".", 1)[1]] = float(o.split("=", 1)[1])
                r["termination"] = {k: getattr(term, k, None) for k in TERM_KEYS}
                r["termination_source"] = {k: getattr(src_term, k, None) for k in TERM_KEYS}
                r["termination_ok"] = all(r["termination"][k] is not None and abs(float(r["termination"][k]) - float(v)) < 1e-12
                                          for k, v in exp_term.items())
                obj = instantiate(cfg.objective, _convert_="all")
                src_obj = instantiate(src_cfg.objective, _convert_="all")
                beta_want = next((float(o.split("=", 1)[1]) for o in reversed(overrides(a)) if override_key(o) == BETA),
                                 getattr(src_obj, "object_task_rotation_beta", None))
                r["objective"] = {"type": type(obj).__name__,
                                  "object_task_rotation_beta": getattr(obj, "object_task_rotation_beta", None),
                                  "source_rotation_beta": getattr(src_obj, "object_task_rotation_beta", None),
                                  "object_task_weight": getattr(obj, "object_task_weight", None),
                                  "object_task_position_beta": getattr(obj, "object_task_position_beta", None),
                                  "position_only_after_release": getattr(obj, "object_task_position_only_after_release", "n/a"),
                                  "no_drop_weight": getattr(obj, "no_drop_weight", "n/a"),
                                  "n_terms": len(getattr(obj, "term_names", ()) or ())}
                r["objective_ok"] = beta_want is not None and abs(float(r["objective"]["object_task_rotation_beta"]) -
                                                                  float(beta_want)) < 1e-12
                if a["name"] == "ep28-W3H":
                    r["objective_ok"] &= (r["objective"]["position_only_after_release"] is True
                                          and r["objective"]["no_drop_weight"] == -50.0)
                try:
                    cur = instantiate(cfg.curriculum, _convert_="all")
                    stage_w = {}
                    for step in (a["fork"], a["target"]):
                        w = cur.stage_at(step).weights_for(tuple(obj.term_names))
                        stage_w[str(step)] = {"stage": cur.stage_index(step), "object_task": dict(zip(obj.term_names, w)).get("object_task"),
                                              "object_no_drop": dict(zip(obj.term_names, w)).get("object_no_drop")}
                    r["curriculum"] = stage_w
                    r["curriculum_ok"] = True
                except AttributeError as error:  # the tree has no stage_at / weights_for API: report only
                    r["curriculum"] = f"n/a in this tree ({error})"
                    r["curriculum_ok"] = not any(override_key(o).startswith("curriculum.") for o in overrides(a))
                except Exception as error:  # noqa: BLE001
                    r["curriculum"] = f"{type(error).__name__}: {error}"
                    r["curriculum_ok"] = False
            except Exception as error:  # noqa: BLE001
                r["error"] = f"{type(error).__name__}: {error}"
            res["arms"][a["name"]] = r
    print("COMPOSE_JSON " + json.dumps(res, default=str), flush=True)
    return 0


def compose_check(trees: list[str] | None = None) -> tuple[bool, list[str]]:
    lines, ok_all = [], True
    for tree in trees or sorted({a["tree"] for a in ARMS}):
        code = TREES[tree]["code"]
        env = clean_env()
        env.update({"PYTHONPATH": str(code / "src"), "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
                    "XDG_CACHE_HOME": str(ME / ".cache"), "PYTHONPYCACHEPREFIX": str(ME / ".cache/pycache"),
                    "HUMAN_MOTION_DATA_DIR": str(HMD), "WANDB_MODE": "disabled"})
        cmd = [str(PY), "-B", str(Path(__file__).resolve()), "compose-child", "--tree", tree]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env, cwd=str(code))
        except subprocess.TimeoutExpired:
            lines.append(f"FAIL compose {tree}: child timed out")
            ok_all = False
            continue
        line = next((ln for ln in out.stdout.splitlines() if ln.startswith("COMPOSE_JSON ")), None)
        if out.returncode != 0 or line is None:
            lines.append(f"FAIL compose {tree}: child exit {out.returncode}: {out.stderr.strip()[-600:]}")
            ok_all = False
            continue
        res = json.loads(line[len("COMPOSE_JSON "):])
        lines.append(f"compose {tree}: PYTHONPATH={code / 'src'}, flash_chord from {res['flash_chord_file']} "
                     f"({'the tree' if res['import_ok'] else 'NOT THE TREE'})")
        ok_all &= res["import_ok"]
        for name, r in res["arms"].items():
            a = BY_NAME[name]
            if "error" in r:
                lines.append(f"FAIL {name}: {r['error']}")
                ok_all = False
                continue
            seed_ok = r["training_seed"][0] == "${reset.seed}" and r["training_seed"][1] == a["seed"]
            refs_ok = r["env_refs"]["env.termination"] == "${termination}" and r["env_refs"]["env.objective"] == "${objective}"
            ok = (not r["source_changed"] and not r["unexpected"] and not r["values_bad"] and seed_ok and refs_ok
                  and r["termination_ok"] and r["objective_ok"] and r["curriculum_ok"])
            ok_all &= ok
            changed = [f"{k}: {v[0]} -> {v[1]}" for k, v in r["diff_vs_source"].items()
                       if k not in ("output_dir", "training.checkpoint", "task.parquet")]
            lines.append(f"{'ok  ' if ok else 'FAIL'} {name} ({a['tree']}): source saved-config drift "
                         f"{r['source_changed'] or 'none'}; arm vs source {changed}; unexpected {r['unexpected'] or 'none'}; "
                         f"bad values {r['values_bad'] or 'none'}; training.seed {r['training_seed']}; env refs "
                         f"{'ok' if refs_ok else r['env_refs']}")
            lines.append(f"     termination (dataclass) {r['termination']} (source {r['termination_source']}) "
                         f"{'ok' if r['termination_ok'] else 'MISMATCH'}; objective {r['objective']} "
                         f"{'ok' if r['objective_ok'] else 'MISMATCH'}; curriculum {r['curriculum']}")
    return ok_all, lines


# ============================================================================================== init / prepare / check
def jobs_tsv() -> str:
    head = ("# Tier-1 improvement wave episodes for scripts/runin_eval_watch{,_w2,_r2}.py run with "
            "RUNIN_QDIR=fc_runs/queue_t1improve (one watcher per arm, --episodes NN). Written by scripts/t1improve_launch.py init.\n"
            "#order\tepisode\tsequence_id\tobject\tscene_support\n")
    rows, seen = [], set()
    for a in ARMS:
        if a["ep"] in seen:
            continue
        seen.add(a["ep"])
        seq, obj = EPISODES[a["ep"]][:2]
        rows.append(f"{len(rows) + 1}\t{a['ep']}\t{seq}\t{obj}\t{str(a['support']).lower()}\n")
    return head + "".join(rows)


def init(log: Log) -> int:
    QDIR.mkdir(parents=True, exist_ok=True)
    msgs = [write_new(QDIR / "jobs.tsv", jobs_tsv()),
            write_new(QDIR / "gpus.conf", "# Tier-1 improvement wave: GPUs a trainer may use and the per-GPU cap counting "
                      "EVERY training (<= 3; user rule). Never 4. Also the watchers' GPU list for evaluations of ended runs.\n"
                      f"{DEFAULT_GPUS_CONF}\n"),
            write_new(QDIR / "eval_reserve_mb", f"{EVAL_RESERVE_MB}\n"),
            write_new(QDIR / "settings.conf", "# scripts/t1improve_launch.py settings (KEY=VALUE). RESERVE_SLOTS = feasible "
                      "slots left free for Tier 3 after any start; HOLDOFF_S = minimum gap between two of our starts.\n"
                      + "".join(f"{k}={v}\n" for k, v in SETTINGS.items())),
            write_new(QDIR / "gpu3_handed_over", "GPU 3 may be used by this queue's trainers and watchers (it is not the "
                      "rolling-start handover marker of fc_runs/queue_runin; runin_helper.refused() reads <QDIR>/gpu3_handed_over). "
                      "Written by scripts/t1improve_launch.py init.\n"),
            write_new(QDIR / "README", "fc_runs/queue_t1improve: queue folder of the Tier-1 improvement wave "
                      "(scripts/t1improve_launch.py; fc_runs/T1improve/PREREG_t1improve.md). evallock/gpu{0,1,2,5,6,7}.lock are "
                      "symlinks to fc_runs/queue_runin/evallock (one evaluation per GPU across every watcher); gpu3.lock is our "
                      "own file (queue_runin has none). No state.tsv/status.tsv/queue_state.json on purpose.\n")]
    ev = QDIR / "evallock"
    if ev.is_symlink():
        raise SystemExit(f"{ev} is a symlink: expected a directory")
    ev.mkdir(exist_ok=True)
    for g in SHARED_GPUS:
        p = ev / f"gpu{g}.lock"
        if g == 3:
            if p.is_symlink():
                raise SystemExit(f"{p} is a symlink: expected our own file")
            if not p.exists():
                p.touch()
                msgs.append(f"created {p}")
            continue
        target = SHARED_EVALLOCK / f"gpu{g}.lock"
        if not target.is_file():
            raise SystemExit(f"{target} missing: never created here")
        if p.is_symlink():
            if os.readlink(p) != str(target):
                raise SystemExit(f"{p} -> {os.readlink(p)} != {target}")
        elif p.exists():
            raise SystemExit(f"{p} exists and is not a symlink")
        else:
            p.symlink_to(target)
            msgs.append(f"linked {p} -> {target}")
    msgs.append(write_new(WATCHER_R2, watcher_r2_text()))
    for a in ARMS:
        (arm_root(a) / "runs").mkdir(parents=True, exist_ok=True)
        eval_root(a).mkdir(parents=True, exist_ok=True)
    for m in msgs:
        log(m)
    log(f"init done: {QDIR}, {ARMS_DIR}/<arm>/{{runs,eval}}, {WATCHER_R2}")
    return 0


def prepare(log: Log) -> int:
    ok, why, stock = stock_ok()
    if not ok:
        raise SystemExit(why)
    for a in ARMS:
        prob = source_problems(a)
        if prob:
            raise SystemExit(f"{a['name']}: {'; '.join(prob)}")
        prepare_arm(a, log, stock)
        log(f"{a['name']}: prepared (fork pair, source config, {len(stock)} stock hulls, run dir {run_dir(a)})")
    return 0


def git_head(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60,
                              env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def check_rows(with_compose: bool = True) -> list[tuple[str, bool, str]]:
    rows = []

    def add(name, ok, detail):
        rows.append((name, bool(ok), detail))

    add("prereg", *prereg_ok())
    for name, t in TREES.items():
        head = git_head(t["root"])
        add(f"tree {name} head", head == t["head"], f"{t['root']}: HEAD {head} (pin {t['head'][:12]})")
        bad = [f for f, h in t["files"].items() if not (t["code"] / f).is_file() or sha256(t["code"] / f) != h]
        add(f"tree {name} files", not bad, f"{len(t['files'])} pinned files {'match' if not bad else 'DIFFER: ' + str(bad)}")
        add(f"watcher {name}", *watcher_ok(name))
    for p, h in HELPER_SHA.items():
        add(f"helper {p.name}", sha256(p) == h, f"sha256 {'== pin' if sha256(p) == h else '!= PIN'}")
    ok, why, stock = stock_ok()
    add("stock hull cache", ok, why)
    try:
        inc_ok = sha256(INCUMBENTS) == INCUMBENTS_SHA
    except OSError:
        inc_ok = False
    add("incumbents (20-trial)", inc_ok, f"{INCUMBENTS} sha256 {'== pin' if inc_ok else '!= PIN'}")
    for p in LOCKS:
        add(f"gpu-choice lock {p.parent.name}", p.is_file(), str(p))
    add("queue jobs.tsv", (QDIR / "jobs.tsv").is_file() and (QDIR / "jobs.tsv").read_text() == jobs_tsv(), str(QDIR / "jobs.tsv"))
    add("queue gpus.conf", gpus_conf() == {g: 3 for g in SHARED_GPUS}, f"{gpus_conf()}")
    add("queue gpu3_handed_over", (QDIR / "gpu3_handed_over").is_file(), str(QDIR / "gpu3_handed_over"))
    for g in SHARED_GPUS:
        p = QDIR / "evallock" / f"gpu{g}.lock"
        if g == 3:
            add(f"evallock gpu{g}", p.is_file() and not p.is_symlink(), f"{p} (own file)")
        else:
            tgt = os.readlink(p) if p.is_symlink() else None
            add(f"evallock gpu{g}", tgt == str(SHARED_EVALLOCK / f"gpu{g}.lock") and p.exists(), f"{p} -> {tgt}")
    for a in ARMS:
        s = SOURCES[a["src"]]
        bad = [k for k, p in zip(("model", "rsi"), pair_files(src_dir(a), a["fork"])) if not p.is_file() or sha256(p) != s[k]]
        add(f"{a['name']} source pair", not bad, f"{src_dir(a)} model/rsi_{a['fork']} {'== pins' if not bad else 'DIFFER ' + str(bad)}")
        badc = [n for n, h in s["cfg"].items() if not (src_dir(a) / n).is_file() or sha256(src_dir(a) / n) != h]
        add(f"{a['name']} source config", not badc, f"{list(s['cfg'])} {'== pins' if not badc else 'DIFFER ' + str(badc)}")
        prob = source_problems(a)
        add(f"{a['name']} source facts", not prob, "; ".join(prob) or
            f"experiment {a['exp']}, {EPISODES[a['ep']][1]} parquet, support {a['support']}, seed {a['seed']}, "
            f"inherited {a['inherit'] or 'none'}")
        add(f"{a['name']} fork copy", *fork_ok(a))
        if stock:
            add(f"{a['name']} hull cache", *cache_ok(a, stock))
        try:
            keys = safetensors_keys(pair_files(fork_dir(a), a["fork"])[1])
            pend = all(k in keys for k in ("pending.n_iter", "pending.f_iter"))
            add(f"{a['name']} rsi sidecar", pend and any(k.startswith("bins.") for k in keys),
                f"{len(keys)} keys, bins.* {sorted(k[5:] for k in keys if k.startswith('bins.'))}")
        except (OSError, ValueError, struct.error) as error:
            add(f"{a['name']} rsi sidecar", False, f"unreadable ({error.__class__.__name__})")
        run = run_dir(a)
        busy = [p.name for p in run.iterdir() if p.name.startswith(("model_", "rsi_")) or p.name == "final.json"] \
            if run.is_dir() else ["(missing)"]
        add(f"{a['name']} run dir", launch_record(a) is not None or not busy, f"{run}: {busy[:3] or 'fresh'}")
    if with_compose:
        ok, lines = compose_check()
        add("hydra compose (every arm, its own tree)", ok, "\n    " + "\n    ".join(lines))
    return rows


def check(log: Log) -> int:
    rows = check_rows()
    for name, ok, detail in rows:
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
    bad = [n for n, ok, _ in rows if not ok]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} checks ok{'' if not bad else '; FAILED: ' + ', '.join(bad)}")
    return 0 if not bad else 1


# ============================================================================================== dry / gate / status
def dry(log: Log) -> int:
    rc = check(log)
    st, conf = load_settings(), gpus_conf()
    print(f"\n==== YIELD GATE ({now()})")
    gate = yield_gate(st)
    for f in gate["facts"]:
        print(f"  fact: {f}")
    for r in gate["reasons"]:
        print(f"  HOLD: {r}")
    print(f"  decision: {'OPEN (Tier 3 has no waiting clip)' if gate['ok'] else 'HOLD (yield to Tier 3)'}")
    view = observe(st)
    print("\n==== GPU VIEW (trainings of anybody / free MiB / ramp debt / MPS)")
    for g in sorted(view):
        v = view[g]
        print(f"  gpu{g}: {v['trainings']} training(s) {v['train_pids']}, {v['free']} MiB free, debt {v['debt']}, "
              f"MPS {v['mps'] or 'no'}{'  (never used: GPU 4)' if g in NEVER_GPUS else ''}")
    f_now, detail = feasible(view, st, conf, {}, "")
    print(f"  feasible slots now: {f_now} {detail}; settings {st}")
    trainers = our_trainers()
    todo_all = [n for n in ORDER if launch_record(BY_NAME[n]) is None and n not in trainers]
    todo = [n for n in todo_all if released(BY_NAME[n])[0]]
    last = last_launch_time()
    print(f"\n==== PLAN (priority order; one start per pass, >= {st['HOLDOFF_S']} s apart; last start "
          f"{'never' if last is None else f'{time.time() - last:.0f} s ago'})")
    for n in todo_all:
        if n not in todo:
            print(f"  HELD {n}: {released(BY_NAME[n])[1]}")
    assume, exclude = {}, ""
    for i, name in enumerate(todo):
        p, why = plan_one(name, view, st, conf, assume, exclude)
        if p is None:
            print(f"  {i + 1}. {name}: WAITS: {why}")
            for rest in todo[i + 1:]:
                print(f"     {rest}: waits behind {name} (strict order)")
            break
        tag = "NEXT START" if i == 0 else f"then (start {i + 1}, >= {i * st['HOLDOFF_S'] // 60} min later)"
        print(f"  {i + 1}. {name}: {tag} on gpu{p['gpu']} cpus {p['cpus']}: {p['why']}; {p['feasible_after']} feasible after")
        assume, exclude = p["assume"], p["exclude"]
    head = todo[0] if todo else None
    if head:
        p, why = plan_one(head, view, st, conf, {}, "")
        decision = ("WOULD START " + head + f" on gpu{p['gpu']}" if (gate["ok"] and p) else
                    f"NO START ({'gate: ' + '; '.join(gate['reasons']) if not gate['ok'] else why})")
        print(f"\n==== DECISION NOW: {decision}")
        a = BY_NAME[head]
        if p:
            print(f"  command: {shown_command(a, p['gpu'], p['cpus'], p['nth'])}")
            print(f"  watcher: {watcher_command(a)}")
    print("\n==== EVERY ARM (command shape on gpu G, cpus C; watcher)")
    for a in ARMS:
        print(f"  {a['name']}: {shown_command(a, 'G', 'C', 4)}")
        print(f"     watcher: {watcher_command(a)}")
    return rc


def gate_cmd(log: Log) -> int:
    g = yield_gate(load_settings())
    print(json.dumps(g, indent=1))
    return 0 if g["ok"] else 3


def fmt(x, spec=".4f"):
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return "-"


def status(log: Log) -> int:
    trainers, ws, inc = our_trainers(), our_watchers(), incumbents()
    ok, why = prereg_ok()
    print(f"[{now()}] PREREG: {why}")
    g = yield_gate(load_settings())
    print(f"gate: {'OPEN' if g['ok'] else 'HOLD: ' + '; '.join(g['reasons'])}")
    auto = [p for p in T2H.all_pids() if "t1improve_launch.py" in " ".join(T2H.read_cmdline(p)) and "auto" in T2H.read_cmdline(p)]
    print(f"auto loop: {auto or 'not running'}; STOP_AUTO {'present' if (T1I / 'STOP_AUTO').exists() else 'absent'}; "
          f"PAUSE {'present' if (T1I / 'PAUSE').exists() else 'absent'}")
    print(f"{'arm':9} {'state':12} {'detail':24} {'newest':>7} {'evals':>5} {'last scan (it auc sp mp mppe)':34} "
          f"{'best scan':16} {'incumb. 20-tr. JSON':22} watcher")
    for a in ARMS:
        s, d = arm_state(a, trainers)
        rows = [r for r in eval_rows(a) if r.get("status") == "ok"]
        last = rows[-1] if rows else None
        best = max(rows, key=lambda r: (float(r["auc"]), -int(r["iteration"]))) if rows else None
        i = inc.get(a["item"], {})
        fin = read_json(eval_root(a) / ep2(a["ep"]) / "runin_final.json")
        print(f"{a['name']:9} {s:12} {d[:24]:24} {newest_ckpt(run_dir(a)) or '-'!s:>7} {len(rows):>5} "
              f"{(last['iteration'] + ' ' + fmt(last['auc']) + ' ' + fmt(last['sp_sr'], '.0f') + ' ' + fmt(last['mp_sr'], '.0f') + ' ' + fmt(last['mppe'], '.2f')) if last else '-':34} "
              f"{(best['iteration'] + ' ' + fmt(best['auc'])) if best else '-':16} "
              f"{fmt(i.get('json_auc_mean')) + ' (' + i.get('iteration', '?') + ')' if i else '-':22} "
              f"{ws.get(a['name']) or ('final' if fin else '-')}")
    return 0


def watch_dry(name: str) -> int:
    a = BY_NAME[name]
    ok, why = watcher_ok(a["tree"])
    if not ok:
        raise SystemExit(why)
    env = clean_env()
    env.update(watcher_env_updates(a))
    print(f"==== {name}: {watcher_command(a, dry=True)}", flush=True)
    return subprocess.run([str(PY), "-B", str(tree_of(a)["watcher"]), *watcher_args(a), "--dry-run"], cwd=ME, env=env,
                          timeout=900).returncode


# ============================================================================================== auto
def all_settled() -> bool:
    trainers = our_trainers()
    for a in ARMS:
        s, _ = arm_state(a, trainers)
        if s == "START_FAILED":
            continue
        if s != "ENDED" or not (eval_root(a) / ep2(a["ep"]) / "runin_final.json").is_file():
            return False
    return True


def heartbeat(gate: dict) -> None:
    trainers = our_trainers()
    parts = []
    for a in ARMS:
        s, _ = arm_state(a, trainers)
        parts.append(f"{a['name']}:{s}" + (f"@{newest_ckpt(run_dir(a))}" if s in ("RUNNING", "ENDED") else ""))
    line = (f"[{now()}] HEARTBEAT t1improve {os.getpid()} | gate {'OPEN' if gate['ok'] else 'HOLD'} "
            f"({'; '.join(gate['reasons'])[:300] or '-'}) | {' '.join(parts)}")
    with (T1I / "heartbeat.log").open("a") as fp:
        fp.write(line + "\n")


def auto(log: Log) -> int:
    T1I.mkdir(parents=True, exist_ok=True)
    lock = os.open(T1I / ".auto.lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("another auto loop holds fc_runs/T1improve/.auto.lock") from None
    if (T1I / "STOP_AUTO").exists():
        raise SystemExit("STOP_AUTO present: remove it first")
    ok, why = prereg_ok()
    if not ok:
        raise SystemExit(f"PREREG check failed: {why}")
    ok, why, stock = stock_ok()
    if not ok:
        raise SystemExit(why)
    log(f"auto loop start pid {os.getpid()}: order {ORDER}")
    last_beat = 0.0
    while True:
        if (T1I / "STOP_AUTO").exists():
            log("STOP_AUTO: auto loop exits (trainers and watchers keep running)")
            return 0
        st = load_settings()
        try:
            ensure_watchers(log)
        except Exception as error:  # noqa: BLE001
            log(f"ensure_watchers failed: {type(error).__name__}: {error}")
        try:
            launch_pass(log, stock)
        except SystemExit as error:
            log(f"launch pass refused: {error}")
        except Exception as error:  # noqa: BLE001
            log(f"launch pass failed: {type(error).__name__}: {error}")
        if time.time() - last_beat >= HEARTBEAT_S:
            try:
                heartbeat(yield_gate(st))
            except Exception as error:  # noqa: BLE001
                log(f"heartbeat failed: {type(error).__name__}: {error}")
            last_beat = time.time()
        if all_settled():
            log("every arm ended with runin_final.json (or failed to start): auto loop exits")
            return 0
        time.sleep(st["POLL_S"])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["init", "prepare", "check", "dry", "gate", "status", "watch-dry", "next",
                                   "ensure-watchers", "auto", "compose-child"])
    p.add_argument("arm", nargs="?")
    p.add_argument("--tree")
    args = p.parse_args()
    if args.cmd == "compose-child":
        return compose_child(args.tree)
    write = args.cmd in ("init", "prepare", "next", "ensure-watchers", "auto")
    log = Log(write)
    if args.cmd == "init":
        return init(log)
    if args.cmd == "prepare":
        return prepare(log)
    if args.cmd == "check":
        return check(log)
    if args.cmd == "dry":
        return dry(log)
    if args.cmd == "gate":
        return gate_cmd(log)
    if args.cmd == "status":
        return status(log)
    if args.cmd == "watch-dry":
        if args.arm not in BY_NAME:
            raise SystemExit(f"watch-dry needs an arm: {ORDER}")
        return watch_dry(args.arm)
    if args.cmd == "ensure-watchers":
        ok, why = prereg_ok()
        if not ok:
            raise SystemExit(f"PREREG check failed: {why}")
        ensure_watchers(log)
        return 0
    if args.cmd == "next":
        ok, why, stock = stock_ok()
        if not ok:
            raise SystemExit(why)
        n = launch_pass(log, stock)
        ensure_watchers(log)
        print(f"started {n}")
        return 0
    if args.cmd == "auto":
        return auto(log)
    return 2


if __name__ == "__main__":
    sys.exit(main())
