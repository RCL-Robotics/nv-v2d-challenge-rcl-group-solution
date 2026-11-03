# PREREG: Tier-1 improvement wave (T1improve, 2026-10-27)

**Version 2**, amended 2026-10-27 18:26 CST, still **before any arm starts** (no launches.tsv, no launch.json, no t1improve
process). Version 1 (sha256 `e387a3c4750489d822c25d6a71d3e3cce90a4c44958a527c0526c445beeb6a79`) is kept as `T1I/build/PREREG_t1improve.v1.md`; section 13 lists every
change and why.

Written 2026-10-27 ~18:00 CST, after the build (`init`, `prepare`, `check`, `dry`) and **before any arm starts**. Nothing in
this wave has been launched by the build. Folder `/root/autodl-tmp/haoran.zhang/fc_runs/T1improve` (below: `T1I`);
`ME=/root/autodl-tmp/haoran.zhang`. This file is hash-locked in `PREREG_t1improve.sha256`; the launcher refuses to start
anything if the hash differs or if this file does not name the launcher's, the wrapper's and the fc_r2 watcher copy's
sha256 (section 12).

## 1. Authority, priority and scope
- User, 2026-10-27 ~17:22 CST (verbatim, the latest instruction): "why retry, I think no need retry right keep current
  results for tier3 queue tier1 first, we will wait first round all tier3 finished then add revisement on tier3".
- Priority as recorded by the main session: Tier 3 first; then Tier 2 (run by Zuxing; none of ours); then this Tier-1
  wave. No Tier-3 retry: `tmp_analysis/tier3/f0_retry/RETRY_NOGO` (written 17:23, read only here). The first-round
  Tier-3 f0 runs (driver pid 818630) and the Tier-3 final 5x pipeline (pid 927986) keep their slots and lanes; the
  Tier-2 round-2 selection on eval lanes is not touched.
- Plan: `tmp_analysis/t1improve_20261027/PLAN.md` (sha256 `1b099ed493f279527bf2083a9960eba790ae628df6d96c03a1da9a1b5d234d64`), actions 3-9 and 12.
- Tier 1 may use clean data. Nothing from this wave (scores, checkpoints, choices) feeds any Tier-2 or Tier-3 decision.
- Out of scope here: uploads, pushes, the evaluator / kit / stock runtime (never modified), GPU 4, MPS, signals.
- Consent (version 2): PLAN section 9 item 3 lists the training-only palm/pelvis termination relaxation of ep14-R and
  ep28-W3H as a user decision; the 17:22 reply above did not answer it explicitly. Both arms are therefore HELD
  (section 6) and start only after the main session records the user's explicit OK, verbatim, in
  `T1I/release/<arm>.txt`. If the user declines, they stay HELD, never start and are reported as not run; no other arm
  depends on them and no new PREREG is needed.

## 2. Arms (priority order = start order)

All arms: `reset.seed=42`, `training.resume=true` from a sha-checked copy of the source pair in `T1I/arms/<arm>/fork/`
(outside the run dir), `training.early_stop_help_zero_tail=0`, ABSOLUTE `training.max_iterations`,
`logging.mode=disabled +logging.local_jsonl=true`, the source's experiment, task parquet (Tier-1 data root
`ME/data/human_motion_data`) and `scene.support` (verified against the source's saved overrides, sha-pinned).

| # | arm | replaces item (if selected) | code tree | source pair (fc_runs/...) | model sha256 | rsi sha256 | experiment | support | inherited overrides | added overrides | max_iterations (+it) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ep14-R | ep14 (78000) | b39b69b | T1fix/B/ep14 @ 81000 | `6c16cf94e78fffcc45c7c3717da704f0a99b80b1f96d57612db70cba1887a8ae` | `83323fa9f4818f6c2dfd44876f1ae2b6d9f3df742c14ae32e9592323df51a4f5` | g1_recon_body_coord_rsi_r2 | false | objective.object_task_rotation_beta=2.6 | termination.palm_position_threshold=1.2, termination.palm_orientation_threshold=3.2, termination.pelvis_position_threshold=1.2 | 96000 (+15000) |
| 2 | ep28-W3H | ep28 = ep28alt (W2PD 66000) | 7fbc227 | T1fix/W2PD/ep28 @ 68000 | `b674e5e5ff931672435c85f11e3e42f5ee37968fd3a6774a592d51d1748881fc` | `d74eb9563407eaed31601a79a9d1a2cb696b0a4bc26483e36e19aeb90b5c944b` | g1_recon_body_coord_rsi_r2 | true | every W2PD override: objective.object_task_rotation_beta=4.5, termination.object_orientation_threshold=3.2, termination.object_position_threshold=0.20, +objective.object_task_position_only_after_release=true, +objective.no_drop_weight=-50.0, +curriculum.stages.0.objective_weights.object_no_drop=-50.0 | termination.palm_position_threshold=1.2, termination.palm_orientation_threshold=3.2 | 83000 (+15000) |
| 3 | ep21-B5 | ep21 (119500) | b39b69b | C3r2_runin/ep21 @ 119500 | `ca4801dcd3c2838e127c66803886647c055fc2427adc2db03a5113650ad8c6cf` | `be1bd5ffbadd25ee9f4be0b03ccd1d43839f3584ffce315d8c549d4590ceab5f` | g1_recon_body_coord_rsi_r2_runin | false | - | objective.object_task_rotation_beta=5.0 | 139500 (+20000) |
| 4 | ep10-Bc | ep10 (36000) | b39b69b | T1fix/B/ep10 @ 36432 | `dc34e3009039d22aa4e11d83dd04fb27d8a1227ae1a0b177778b4afd9952e442` | `25da0cfecc246573ac611220eada3f3bdeb1b7bf7557651535846b15c5d64bce` | g1_recon_body_coord_rsi_r2 | false | objective.object_task_rotation_beta=4.5 | - | 60000 (+23568) |
| 5 | ep10-BT | ep10 (36000) | b39b69b | T1fix/B/ep10 @ 36432 (own copy) | same as ep10-Bc | same as ep10-Bc | g1_recon_body_coord_rsi_r2 | false | objective.object_task_rotation_beta=4.5 | termination.object_orientation_threshold=0.6 | 60000 (+23568) |
| 6 | ep19-Bc | ep19 (108000) | b39b69b | T1fix/B/ep19 @ 110000 | `a433b497254130be426f5ae16afd31f87d4983e7b02ca86d7f15478312b158fc` | `37dc23ed02988522c298c04722692e681e03a980e7af4facbb1276ee7b4e218e` | g1_recon_body_coord_rsi_r2_runin | false | objective.object_task_rotation_beta=3.0 | - | 135000 (+25000) |
| 7 | ep01-Bc | ep01 (140000) | b39b69b | T1fix/B/ep01 @ 142000 (S2B) | `774fe75ff731f8ffb59325815ceb377d3f9dbf0ebc5a2f48ef7495e9782f1603` | `1b666e070d9401ed2dd7e3911d2755c2983999eaf901c14f67f6bb2cfa94b440` | g1_recon_body_coord_rsi_r2_runin | true | objective.object_task_rotation_beta=3.0 | - | 162000 (+20000) |
| 8 | ep15-B2 | ep15 (20974) | 61fa5d3 | C3r2_wide/ep15 @ 20974 | `16e14bcd737120d0dccb7acc5f5454daf90fd63024eef91264655d98d59f8af7` | `8bffcbd190bad0299b787b8b50afe857eb0f6006725f66d67285fb1ace567987` | g1_recon_body_coord_rsi_r2 | false | - | objective.object_task_rotation_beta=2.0 | 46000 (+25026) |
| 9 | ep05-B3 (filler) | ep05 (42500) | b39b69b | C3r2_runin/ep05 @ 42500 | `37371dfcb760cbb15db5c2cc9ca6a566c05cee6d2f3874c08653a4f852ab6704` | `a2f6a2bb9f55ba7a3e1c16bfdbc87e2fdb0c9a63d7b7901eae12b0a4cba7a60e` | g1_recon_body_coord_rsi_r2_runin | false | - | objective.object_task_rotation_beta=3.1 | 62500 (+20000) |
| 10 | ep25-B25 (filler) | ep25 (102000) | b39b69b | C3r2_runin/ep25 @ 102000 | `ffac8b771ef47e5ad1771d076af16ea55934fdf5017196cbbb4d2a7ce129d39b` | `1bc0ecc6c3b9f76406708e28ab04c491738d06781e9c82569e8089815abb47de` | g1_recon_body_coord_rsi_r2_runin | true | - | objective.object_task_rotation_beta=2.5 | 122000 (+20000) |

Source saved configs (sha256; copied byte-identically into `T1I/arms/<arm>/source/`):
- T1fix/B/ep14 `hydra/.hydra/config.yaml` `1303589055be652b4a00de15da294a3c5513bc55d9b63c1c68bd1283cf36d2bb`, `overrides.yaml` `ab24efde436fc26f3126094519d27dbe14eb8589e1dcd38c66aa01f4c6c6b1cd`
- T1fix/W2PD/ep28 `45bab9da23d3ab9054b28f79983e66127790d7c21f671e8dc4d2b07627606cd2`, `9b933924ff086323ed2eb981fa0840d1f0298a35e92bcf0749468f131923d19d`
- C3r2_runin/ep21 `b618a4652f342bf8f8313a89fcc25e71737846b4b9158b2f4b515175b46098ac`, `8e9b6f915cbc716bcf314b8c69f6d0db99a7b92567b8963f8525627ea4f0415b`
- T1fix/B/ep10 `b98ababbac1d17b5dc02a27ff9780fa18a1a1cbc44573cf33b7989ea377b76d8`, `60627db1ac17e5f284a94c3ad9fdc634897a327af67d022c44e4ed9888a11957`
- T1fix/B/ep19 `6883df5ae868d9aa5803e1f24de93e14ebc64196766a37396109d00240c93cff`, `118b645d95695b073f4734095bd2ef0fc037b3debeee07bebc06a894fed1cb01`
- T1fix/B/ep01 `db4e6da7ce0598e11e7c22be51fe26b70a0ce025fb44da348113bbc452a7c44a`, `218f35fcc686cd2faec3c1cbe8c044d12e30ccfd45b40ab17a0309048035c960`
- C3r2_wide/ep15 (no hydra dir; wandb run xg9j2z2a) `files/config.yaml` `c86c7f58724bc9de5ed8518e45f1927d7bc3f3ee7fe88274cce0413681c317ba`, `files/wandb-metadata.json` `e7256600d0aec23a888e8384e096af51b2cadd5787773643ce297dcb5877691c` (its `args` give the source overrides: experiment=g1_recon_body_coord_rsi_r2, the wet_floor_sign parquet, reset.seed=42, training.max_iterations=45000, scene.support=false)
- C3r2_runin/ep05 `14af26a77d78db5eec9fc5dd0ee3db52aa7c47ef6df9f44b967d0fa5c4491954`, `92b1b6ed03a1d22d32e6ecd09df6289a3a886399fe413d1b6663ffa8f7c58ee0`
- C3r2_runin/ep25 `69e47dda87f05555e0f945a76b54c4fa63f213ca7f7c86d982a890c7a94ec513`, `4f2d828eb3695e9ae257c8eb14c72c3534a3a85aee4a222f1d99b9086cbe018d`

Task parquets (Tier-1 data root, sha256): ep14 `c2daabf33d447229d4e3bafa7d7298cf4d49db40a17aaf7e01c99ba0b8442421`, ep28 `731660a9d95094f88e9b35aaab4d05829e8022160a9b45e47fc140188a5c63dc`,
ep21 `1aaba73ef26926087170068197755e6ae27156bb729a3daa55712dbc37bdbbc2`, ep10 `56742e0fe7994d8b262313385efddcc056d35f908220a431edd1d95bbc0bd7f1`, ep19 `9f91433ee5e838614c11a1abfcfae2bb1bc288a7002d42f3c0d808cedf7a7e92`,
ep01 `a570b5169370dee17ab19e5e9b3cbbbcb39d4d3fee15ba6f32cbcb8cedcae585`, ep15 `e91e9ec5cb5da0d84c087bbad0b195dc43bdeb2483f927a62a9eac15ac9db45e`, ep05 `26ff8eae4cf6dc5457c3aaae918a1ed09d9d2961c4955f278093371e2fc51381`,
ep25 `ef78c662cdd7ca3c8ccc72a47fd3964cb2ebc3f740cfc58648c3b6056fb6dc39`.

Hydra compose (dry run, 2026-10-27 17:55, `check`): every arm composed in its own tree with flash_chord imported from
that tree; the source's saved config re-composes with no drift; the only differences arm vs source are the added
overrides above plus max_iterations / resume / checkpoint / early stop / logging; `training.seed` resolves to 42;
`env.termination = ${termination}`, `env.objective = ${objective}`; the termination and objective dataclasses
(validators ran) hold exactly the intended thresholds and rotation beta; ep28-W3H keeps position-only-after-release
and no_drop_weight -50 (curriculum weight object_no_drop -50 at 68000 and 83000).

Incumbents (the pinned set's 20-trial records, `fc_runs/submissions/t1_best20_20261026/per_episode.csv`, sha256
`e02cd1f80cc45a83fc80b4f5f0142411473b6db6c0bcb39ae0cd2ab3f7c27adc`; manifest_best.json `e07c5bf56e76cdc743140d0d3bac8d0f31873f25272d75ce7efb2fffa3378c84`).

**The selection bar is the evaluator-JSON metric (version 2).** Candidate evaluations follow the watcher protocol and
read the evaluator's results JSON (`add_auc`, `mean_per_point_error_cm`, SP / MP success), so section 8.4 compares
them with the incumbent's `json_*` columns, measured the same way on the same 20 trial files. The kit `metric_code`
score of the packed CSV (`auc_mean`; the set's leaderboard number, t1best20 PREREG section 6) differs from the JSON
mean by a fixed amount per item and is shown for reference only. Comparisons use the unrounded CSV values; the table
rounds. SP / MP pass counts are identical in both columns.

| item | iteration | JSON AUC mean (bar) | JSON AUC sd | needed: bar + 0.005 | floor (e): bar - 0.10 | kit AUC mean (reference) | kit - JSON | SP pass | MP pass | JSON MPPE mean cm | MPPE cap (d) cm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ep14 | 78000 | 0.8265 | 0.0037 | 0.8315 | 0.7265 | 0.8270 | +0.0005 | 20/20 | 20/20 | 1.503 | 1.653 |
| ep28alt (set's ep28) | 66000 | 0.8073 | 0.0028 | 0.8123 | 0.7073 | 0.8074 | +0.0001 | 20/20 | 6/20 | 3.466 | 3.616 |
| ep21 | 119500 | 0.7548 | 0.0011 | 0.7598 | 0.6548 | 0.7548 | +0.0000 | 20/20 | 20/20 | 1.497 | 1.647 |
| ep10 | 36000 | 0.8675 | 0.0014 | 0.8725 | 0.7675 | 0.8693 | +0.0017 | 20/20 | 20/20 | 1.312 | 1.462 |
| ep19 | 108000 | 0.8951 | 0.0020 | 0.9001 | 0.7951 | 0.8957 | +0.0006 | 20/20 | 20/20 | 1.023 | 1.173 |
| ep01 | 140000 | 0.7603 | 0.1223 (one collapse trial) | 0.7653 | 0.6603 | 0.7582 | -0.0021 | 19/20 | 19/20 | 3.673 | 3.823 |
| ep15 | 20974 | 0.9370 | 0.0013 | 0.9420 | 0.8370 | 0.9376 | +0.0006 | 20/20 | 20/20 | 0.876 | 1.026 |
| ep05 | 42500 | 0.9480 | 0.0023 | 0.9530 | 0.8480 | 0.9473 | -0.0006 | 20/20 | 20/20 | 0.786 | 0.936 |
| ep25 | 102000 | 0.7490 | 0.0067 | 0.7540 | 0.6490 | 0.7481 | -0.0009 | 20/20 | 20/20 | 2.013 | 2.163 |

## 3. Training command and environment (every arm)
`taskset -c <2 own-NUMA cores + HT> <ME>/conda_envs/flashchord/bin/python scripts/train_rl.py experiment=<exp>
task.parquet='<ME>/data/human_motion_data/whole_body/soma/sequence_id=<seq>/robot_name=g1/data.parquet'
scene.support=<s> reset.seed=42 output_dir=T1I/arms/<arm>/runs/epNN hydra.run.dir=<output_dir>/hydra logging.mode=disabled
+logging.local_jsonl=true training.early_stop_help_zero_tail=0 training.max_iterations=<target> training.resume=true
training.checkpoint=T1I/arms/<arm>/fork/model_<N>.safetensors <inherited + added overrides>`, cwd = the tree's
flash_chord dir, own session, nice 0, stdout to `T1I/arms/<arm>/train.log`. Environment: PYTHONPATH=<tree>/src,
HUMAN_MOTION_DATA_DIR=<ME>/data/human_motion_data, CUDA_VISIBLE_DEVICES = RUNIN_GPU = the chosen GPU,
OMP/MKL/OPENBLAS threads = 4, XLA_PYTHON_CLIENT_PREALLOCATE=false, WANDB disabled, LD_LIBRARY_PATH = the env's cu13 /
cudnn / nccl / nvshmem libs (as every trainer), FLASH_CHORD_CACHE_DIR=T1I/arms/<arm>/fc_cache, markers
RUNIN_QUEUE=<ME>/fc_runs/queue_t1improve, T1IMPROVE_ARM=<arm>, T1FIX_ARM=<arm> (monitor label), T1FIX_STAGE=t1improve;
CUDA_MPS_*, WANDB_*, FLASH_CHORD_*, RUNIN_*, PYTHONHOME, PYTHONSTARTUP of the launcher's environment are dropped.
Each arm resumes in the tree that wrote its pair, so the RSI sidecar format is the writer's own (checked: 15 keys,
bins.* = counts, last_move, rate, top, win_f, win_n; pending.n_iter / f_iter present).

## 4. Code trees (read only; never edited)
| tree | path (cwd) | HEAD | exporter export.py sha256 | objectives/recon_body.py | watcher |
|---|---|---|---|---|---|
| b39b69b | code/nv-v2d_runin/robotic_grounding/flash_chord | b39b69be6c2f561c0123c6aeb8cd13f8b9d542e7 | 04367c56564de9540deb3d8e33adbfddb7800df190361a0576005a1a4ccfd212 | 2476d0514dd4260a841eaa29e29f0d33b34480ab4fbc78ad1a08faaed4baa5d2 | scripts/runin_eval_watch.py `61ac5b8a170606d3804a00c927834afe1f8a4d84ac99f6ba594c8b028ab6c5cc` |
| 7fbc227 | code/nv-v2d_ep28w2/robotic_grounding/flash_chord | 7fbc2273a92baefe652cb11e09c7e0acdd62da00 | d9a7d45da58c44d658af3ef733061c0462bb04c5867994c74cc9e246017eb585 | a4f804c07622241df829caa1bf294d68d3280b06cf4aae318c2edf93cef913c8 | scripts/runin_eval_watch_w2.py `ac2102b87fe224917b9b008a77697cc22e3082e31d8fe1f1b413eb011f92f5fe` |
| 61fa5d3 | code/fc_r2 | 61fa5d31ff8054bf1e5cf5375ef0957e54d2729b | 34e3d7f3a7dca4fdd1e7c4d26750c6332abcc1d629f6a9c6c7a965833a22206f | 2476d0514dd4260a841eaa29e29f0d33b34480ab4fbc78ad1a08faaed4baa5d2 | scripts/runin_eval_watch_r2.py `5572b07446bfc2537ce69760902381ec9bc8cda8f7a15fc087d7f5acc93b33d8` (byte copy of runin_eval_watch.py with only line 98 `CODE = ME / "code/fc_r2"`; the fc_r2 exporter is the one that exported the pinned ep15, t1best20 PREREG section 2) |

Byte-identical in all three: scripts/train_rl.py `eafa5723...`, scripts/track2/eval_all.py `657295921e82...`,
assets/mesh.py `9868910150c9...`, configs/termination/recon_body.yaml `fd039fba9830...`, configs/train.yaml
`fc00ddbabbad...`, experiment g1_recon_body_coord_rsi_r2.yaml `aa2e31c22fc2...` (runin experiment `9ee8e1ba3a26...`
in b39b69b / 7fbc227). Submission note: a selected ep28-W3H checkpoint needs 7fbc227 in the cited commit (already
merged in dev.haoran.t1best d1373fb); ep15-B2 is exported with fc_r2 61fa5d3 like the pinned ep15.

## 5. Collision hulls (all arms: the evaluator's)
Each arm's `FLASH_CHORD_CACHE_DIR=T1I/arms/<arm>/fc_cache` holds a byte copy of ALL 29 stock-runtime CoACD hull files
(`code/video_to_data_8c778f9/.../assets/.cache/convex`, listing digest `4864e7c2eddec2063463a5c7c722e045514316948fa17394c5acca280f82b27d`), as T1fix stage 2 seeded its cache.
Object keys (computed by the scene code's own path, `T1I/build/cache_keys.py`, output `cache_keys.out` sha256
`87d8730c633f63ae27bc9c1e2cb01961a7a31ae3c33d0c80868b483a281c67ac`) and the hulls each source trained on:

| arm | object | key | stock sha256 (head) | hulls the source trained on |
|---|---|---|---|---|
| ep14-R | round_table | 6c7584c98b6a755d | 6868717638a4 | stock (T1fix/stage2/fc_cache) |
| ep28-W3H | squeegee | 0885d534a2fd2869 | d8c2ee511734 | stock (T1fix/ep28w2/fc_cache) |
| ep21-B5 | tall_bar_stool | 228b2dc97a7f8ce0 | e11854911827 | nv-v2d_runin tree cache: same 21-hull set, other bytes |
| ep10-Bc / BT | white_desk | b1a56fe315a143bd | e6236a69aaf9 | nv-v2d_runin tree cache: **18 hulls vs stock 17** |
| ep19-Bc | cane | 7d05b4ab00818a6f | b1fdceaea89f | nv-v2d_runin tree cache: **11 hulls vs stock 10** |
| ep01-Bc | vacuum | 5367fdef34ef90d3 | 4cc7c2e1da01 | stock (T1fix/stage2/fc_cache) |
| ep15-B2 | wet_floor_sign | dd4639fc0cde3051 | a14f2b8b9647 | fc_r2 tree cache: **50 hulls vs stock 49** |
| ep05-B3 | big_black_box | c0749211ce6d40ec | a25b504158f4 | nv-v2d_runin tree cache: **52 hulls vs stock 53** |
| ep25-B25 | cyan_water_bottle | 15263099f076b724 | 2b40b5283e43 | nv-v2d_runin tree cache: same 8-hull set, other bytes |

Disclosure: ep10, ep19, ep15 and ep05 switch at the fork from their tree's hull set to the evaluator's stock set
(intended: training then sees the evaluation geometry). A new file appearing in an arm's cache after its start means
a trainer built a hull the stock cache lacks; `check` / `status` flag it.

## 6. Placement: Tier 3 first (yield gate), then the fill rule
`scripts/t1improve_launch.py` (`auto`, every 60 s) starts at most ONE arm per pass, always the first
RELEASED arm not yet started, in priority order (RELEASE below), at least HOLDOFF_S = 600 s after our previous start, and only when ALL hold:
- YIELD GATE (fail closed; re-checked under the GPU-choice locks): no `T1I/PAUSE`; `tmp_analysis/tier3/f0/waiting.json`
  readable with an empty `waiting` list; `tmp_analysis/tier3/f0/ALL_LAUNCHED` present; the f0 driver's last HEARTBEAT
  lists no clip in FRESH / NOT_READY / PARQUET_CHANGED / NEEDS_* / RETARGETING and is at most 30 min old while the driver
  runs; if `tmp_analysis/tier3/f0_retry/RETRY_GO` ever appears: a retry heartbeat at most 30 min old with no NEW or QUEUED
  clip; any other `tmp_analysis/tier3/*/waiting.json` written within 30 min readable with an empty `waiting` list (a
  later Tier-3 revision driver that follows the f0 convention); at least RESERVE_SLOTS (settings.conf; **1** since version
  2) feasible slots left after the start; (version 2) the f0 driver's last HEARTBEAT shows no clip DEAD or CRASHED
  (no live trainer and no final.json, so a manual resume would need a slot) unless the main session wrote
  `T1I/t3ack/epNN` (that clip is not resumed).
- FILL RULE: a GPU of `queue_t1improve/gpus.conf` (0,1,2,3,5,6,7; **never 4**); no MPS on it; fewer than 3 trainings of
  ANYBODY on it (nvidia-smi plus trainers not yet visible); free memory >= 16896 + 20480 MiB + the ramp debt of every
  training younger than 900 s; 2 free physical cores (+HT) in the GPU's own NUMA training pool (never the eval-lane
  cores, never cross-NUMA). Choice: fewest trainings, then most free memory, then lowest index. The start happens under
  the shared GPU-choice locks `ch_runs/{seedstudy,f0share05,filler}/.gpulock` (opened read-only, never created; the
  locks the Tier-3 and other drivers take) and they are held 180 s afterwards.
- RELEASE (version 2): ep14-R and ep28-W3H (consent, section 1) and the fillers ep05-B3 and ep25-B25 are HELD until
  the main session writes a non-empty `T1I/release/<arm>.txt`: for ep14-R / ep28-W3H the user's explicit OK,
  verbatim; for a filler the main session's confirmation that no Tier-3 resume or revision and no Tier-2 job of
  Zuxing needs that slot (the gate cannot see Zuxing's pending jobs). A HELD arm is passed over; once released it
  precedes every later arm. With RESERVE_SLOTS = 1 the wave never takes the last feasible slot.
- Dry-run decision at 17:55 CST: gate OPEN (f0 waiting none, ALL_LAUNCHED present, 7 f0 clips running, none
  unlaunched; no RETRY_GO, RETRY_NOGO present); 8 feasible slots (GPUs 3, 5, 6, 7); next start ep14-R on GPU 5; the
  first 8 arms fit GPUs 3/5/6/7 at 10-min spacing; ep05-B3 and ep25-B25 wait for slots the Tier-3 f0 runs free. Version 2: the first
  starts are ep21-B5, ep10-Bc, ep10-BT, ep19-Bc, ep01-Bc, ep15-B2 (>= 10 min apart, one feasible slot always kept
  free); ep14-R / ep28-W3H take the next slot after their release; the fillers start only after theirs.
- Pausing: `touch T1I/PAUSE` (no new start; running arms continue). Stopping the loop: `touch T1I/STOP_AUTO`. The
  launcher never signals any process.

## 7. Watcher protocol (one per arm)
- Script by tree (section 4), started by the launcher right after the arm's trainer; restarted at most 3 times while
  the arm has no `runin_final.json` and no `eval/STOP`. Environment `RUNIN_QDIR=fc_runs/queue_t1improve
  RUNIN_RUNS=T1I/arms/<arm>/runs RUNIN_EVAL_ROOT=T1I/arms/<arm>/eval RUNIN_WATCH_ITERS=<target>`, arguments
  `--episodes NN --every 2000 --no-early-stop --lane-max 1`.
- Each evaluation is the official path: `eval_all.py` of the arm's tree -> E2 export at tolerance 1e-4 with that tree's
  exporter (stock-runtime equivalence check) -> the kit's `run_policy_evaluation.py` + `evaluate_policy.py` on the
  untouched stock runtime `code/video_to_data_8c778f9` (frame-0 start, explicit reset, world_count 1); metrics from
  the evaluator's results JSON (auc = add_auc, sp_sr = spider_sr_uncentered, mp_sr = maniptrans_sr, mppe =
  mean_per_point_error_cm). nice 10, pinned to the NUMA eval-lane cores (48-51,152-155 / 100-103,204-207).
- Lanes: one evaluation per GPU at a time through `queue_t1improve/evallock/gpu<g>.lock`: for GPUs 0,1,2,5,6,7 a
  symlink to the shared `fc_runs/queue_runin/evallock/gpu<g>.lock` (the lock every other watcher / driver honours);
  GPU 3 has no shared runin lock, so ours is our own file (coordinated among our watchers; the Tier-3 final driver
  still counts our evaluate_policy processes per NUMA lane). A running arm is evaluated on its own GPU only; never GPU 4.
- What: the newest stable checkpoint on the 2000 grid (older unevaluated ones are skipped); after final.json the newest
  checkpoint even off-grid; then 2 re-evaluations of the best single scan -> `eval/epNN/runin_final.json` (report only).
  `--no-early-stop`: a watcher never writes stop.json and never signals.

## 8. Selection protocol (binding; written before the first read)
1. **Scans** (watcher): one official evaluation per 2000-grid checkpoint as above.
2. **Candidates per arm**, fixed once the arm has ended: the top-2 scan AUCs among its checkpoints in its last 6000
   iterations (iteration >= end - 6000) plus its final checkpoint (deduplicated). An arm stopped early by a kill flag
   (section 10) uses its last checkpoint as the end.
3. **Candidate evaluations**: at least **3 fresh official evaluations** of each candidate (5 for ep01-Bc and ep25-B25,
   the items whose 20-trial AUC sd exceeds 0.005), same protocol as the watcher, with the arm's tree exporter. The scan
   that nominated a candidate is **not** counted in its mean. (version 2) The watcher's own `runin_final.json`
   re-evaluations are report-only and do **not** count as candidate evaluations. A checkpoint whose 1e-4 export fails
   is not a candidate (no looser tolerance); if that is the final checkpoint, the final-checkpoint slot goes to the
   arm's newest checkpoint whose export passes.
4. **A candidate replaces the pinned item only if all hold** (version 2: every AUC / MPPE comparison is evaluator
   JSON vs evaluator JSON, against the section-2 bar columns; SP / MP rules as PLAN section 7.3):
   (a) its mean JSON AUC >= the item's 20-trial JSON AUC mean + **0.005** (section 2, "needed");
   (b) SP-SR passes in **every** candidate evaluation;
   (c) MP-SR passes in **every** candidate evaluation. ep28 only: a candidate that misses MP in some candidate
       evaluation may still go to the 20-trial confirmation (step 5) when (a), (b), (d), (e) and (f) hold, and is
       kept only if MP passes in at least **16/20** of its confirmation trials (PLAN 7.3);
   (d) JSON MPPE mean <= the item's 20-trial JSON MPPE mean + 0.15 cm (+0.30 cm allowed when the AUC gain in (a) is
       >= 0.02);
   (e) no candidate evaluation below the item's 20-trial JSON AUC mean - 0.10 (section 2, "floor");
   (f) ep01, ep14, ep21 and ep28 candidates also need 0 collapses in a robustness screen with the official evaluator
       at `evaluation.world_count=64` (PLAN section 7.4; a screen only, never a score; collapse = a world's JSON AUC
       below the item's 20-trial JSON AUC mean - 0.10; first check on one candidate that world 0 reproduces the
       official score).
   Several passing candidates of one item (including ep10-Bc vs ep10-BT): the highest candidate mean AUC wins; ties ->
   lower MPPE, then the earlier iteration, then the earlier arm in section 2.
5. **20-trial top-up of only the changed items** with the t1best20 protocol (`fc_runs/submissions/t1_best20_20261026/
   PREREG.md` sections 3-5: one exported file per item evaluated 20 times, official invocation, default seeds,
   lanes and checks unchanged), the item's incumbent evaluated side by side as the alternate. Per changed item the
   final choice is the one with the higher 20-trial mean AUC unless its MPPE mean is > 0.2 cm worse (PLAN 7.5).
   Both sides of this comparison are scored the same way (the t1best20 kit metric_code on the packed CSV). Added in
   version 2: the new item is kept only if its SP and MP pass counts over the 20 trials are not lower than the
   incumbent's side-by-side counts (an ep28 candidate admitted by the 4(c) exception needs MP >= 16/20); otherwise the
   incumbent stays.
   Trial t of the new set = the t1best20 trial t of every unchanged item + top-up trial t of every changed item.
6. **Upload file = the trial closest to the set mean**: among trials whose SP-SR and MP-SR equal their 20-trial mode,
   the one minimising |AUC - mean|/sd_AUC + |MPPE - mean|/sd_MPPE. **Never a best-of trial or a best single scan.**
   If an interim file was uploaded, the final file must dominate it on all four metrics (PLAN 7.6). Packing, pushing
   and uploading need the user's OK and are not part of this wave.
7. Reporting: the 20-trial mean including collapses, next to the uploaded trial.

## 9. Disclosure: the termination relaxations are training-only
ep14-R (palm 1.2 m / 3.2 rad, pelvis 1.2 m), ep28-W3H (palm 1.2 m / 3.2 rad) and ep10-BT (object orientation 0.6
rad) change only when a TRAINING episode is cut. The official evaluation disables every tracking-failure termination:
stock runtime `code/video_to_data_8c778f9/robotic_grounding/flash_chord/src/flash_chord/evaluation/harness.py`
(sha256 `6e61b092e9aa9d00d11c557b8508c09ef990314a3f68b0779bbf97765c85ddc2`):
- lines 57-64: `_RECON_BODY_THRESHOLD_FIELDS` = pelvis / palm / object position and orientation thresholds;
- line 65: `_DISABLED_THRESHOLD = 1.0e6`;
- lines 216-251 `apply_evaluation_composition`, lines 239-242: for every termination block (env and top level) each of
  the six thresholds is set to 1.0e6 ("The tracking-failure terms are disabled so a failed world keeps advancing");
  line 237 forces `env.reset.seed = 42`;
- lines 388-390: the checkpoint's own thresholds are read (`evaluation_thresholds`, `reference_tracking_criteria`)
  BEFORE they are disabled and are used only for the harness's `chord_sr` (lines 489-491; `metrics.py` sha256
  `ef641481aaefd2792cb106df43ede28b0b52223b30f610be4d28286cc4d0209e` line 429 `chord_success`). add_auc (metrics.py 432),
  spider_success (434) and maniptrans_success (435) take no threshold from the checkpoint, and the kit's leaderboard
  metrics use fixed constants (`metric_code/track_2/MP-SR.py` lines 19-23, sha256
  `a1d72216eb7745a13ed7db6a55b9efb31492b35149113181a6c42ffc231fff11`). So the relaxed values change no leaderboard metric;
  they change only the harness's own chord_sr diagnostic, which is not scored.
- Before any submission uses a relaxed-termination checkpoint, one exported file is compared with
  `trials_eval.py export` / `compare_exports.py` (PLAN section 8) and the relaxation is stated in the report. After
  release the robot is no longer required to follow the human in training (the metric is object-only); an earlier note
  called this a grey area, so it is disclosed, not hidden.

## 10. Early reads and kill flags (report only; the main session decides, the launcher never acts)
- ep14-R at +4-6k: rsi bins fail rate 34..37 < 0.5 and final tilt < 10 deg; kill flag at +8k (89000) if the final
  tilt is still > 18 deg.
- ep28-W3H at +5k: palm-termination share < 10% and RSI bins 35-36 failing < 0.3.
- ep21-B5 at +8k (127500): kill flag if ADD over frames 585-677 is still > 12 cm.
- ep10-BT at +6k: kill flag if > 50% of episodes end by object termination or kf_committed stalls.
- To stop an arm early: write `T1I/arms/<arm>/eval/epNN/stop.json` (`{"iteration": <last ckpt>, "stop_reason":
  "kill_flag"}`) FIRST, then the main session sends SIGINT to that arm's own trainer (pid from `launch.json`, verified
  by its T1IMPROVE_ARM marker; never an MPS client); the watcher then evaluates the last checkpoint and finalises.

## 11. Deviations from PLAN.md (recorded before any start)
- ep10-Bc / ep10-BT end at 60000 (PLAN: 66000) and ep19-Bc at 135000 (PLAN: 140000), per the main session's arm list.
- ep15-B2 resumes in its own tree code/fc_r2 @ 61fa5d3 (PLAN: the portable t1best tool); its watcher uses the fc_r2
  exporter (the exporter of the pinned ep15).
- Not in this wave: ep14-RB, ep28-W3C, ep21-B25 / ep21-R (code), ep19-B5, ep01-LB (label bridge), ep15-C, ep09-C, the
  P30 precision pilots, ep04 cont5, the eval-only screen of PLAN action 2.
- The watcher grid is 2000 (PLAN 7.1). Candidates and confirmations follow PLAN section 7 (version 2 restored PLAN
  7.3's SP / MP rule: pass in every candidate evaluation, with ep28's 16/20 MP confirmation alternative), except: the
  replacement margin is the task's 0.005 (not max(0.005, 2 x sd)); the bar is measured in the evaluator-JSON metric on
  both sides (PLAN does not say which; version 2); the side-by-side 20-trial confirmation also requires SP / MP pass
  counts not lower than the incumbent's (an added guard; version 2).
- Version 2 holds ep14-R and ep28-W3H for the user's explicit OK (PLAN 9.3) and the fillers for a main-session slot
  confirmation, sets RESERVE_SLOTS = 1, and holds new starts while an f0 clip is DEAD or CRASHED (section 6).

## 12. Files (sha256)
- launcher `scripts/t1improve_launch.py` `ba5146a4ddbd5f2f4b51f5ba4436444f3a30dfaae5ba63d6c75c5cd02eeb3cbd` (version 2; version 1 `3ee92b9bcf2ac0ecac2c52862f4cbf33054500b5000bb4daae7b74adefbd8075` kept as
  `T1I/build/t1improve_launch.v1.py`; v2 = v1 + RELEASE holds, gate (h) DEAD / CRASHED, RESERVE_SLOTS default 1,
  status shows the JSON bar; written by `T1I/build/patch_launcher_v2.py`)
- wrapper `scripts/t1improve_launch.sh` `1047b819cd710dff79fb814a7b1586774b59751f14994c21c591508cb44df5b0`
- fc_r2 watcher copy `scripts/runin_eval_watch_r2.py` `5572b07446bfc2537ce69760902381ec9bc8cda8f7a15fc087d7f5acc93b33d8`
- watchers `scripts/runin_eval_watch.py` `61ac5b8a170606d3804a00c927834afe1f8a4d84ac99f6ba594c8b028ab6c5cc`, `scripts/runin_eval_watch_w2.py` `ac2102b87fe224917b9b008a77697cc22e3082e31d8fe1f1b413eb011f92f5fe`
- helpers `scripts/runin_helper.py` `99c2a3e7d282ef4f68e6e36797549d2219e6aa1356c66d90483a3a25f3b99bcc`, `scripts/track2_queue_tier2v2_helper.py` `b75a33c3fe050710178c8690d71213df3891988b3e76196e99a37890a7b45886`
- hull-key script `T1I/build/cache_keys.py` `8d2c3daaee465db17ba521380278211711bcdaaf96166983f9bd8a0a1182588b`
- queue folder `fc_runs/queue_t1improve` (jobs.tsv, gpus.conf `0:3 1:3 2:3 3:3 5:3 6:3 7:3`, settings.conf (RESERVE_SLOTS=1 since version 2), eval_reserve_mb 3072,
  gpu3_handed_over, evallock/), written by `init`.

Commands: `bash scripts/t1improve_launch.sh status | gate | dry | check`; start: `bash scripts/t1improve_launch.sh
start-auto` (from a local background ssh); pause `touch fc_runs/T1improve/PAUSE`; stop the loop `touch
fc_runs/T1improve/STOP_AUTO`.

## 13. Amendments (version 2, before any arm started)
Version 1 (sha256 `e387a3c4750489d822c25d6a71d3e3cce90a4c44958a527c0526c445beeb6a79`, 17:58 CST) went through two independent reviews (rules / safety and correctness), both
GO_AFTER_FIXES. Every change below was made before any arm started:
1. Same metric on both sides of the bar (correctness review). Version 1 compared evaluator-JSON candidate means with
   the kit metric_code incumbent mean; the per-item offset (kit - JSON: ep14 +0.0005, ep28alt +0.0001, ep21 +0.0000, ep10 +0.0017, ep19 +0.0006, ep01 -0.0021, ep15 +0.0006, ep05 -0.0006, ep25 -0.0009)
   would have shifted the effective margin per item (ep01 +0.0029 instead of +0.005, ep10 +0.0067). Section 2 now
   gives the JSON bar; section 8.4 (a), (d), (e), (f) compare JSON with JSON.
2. SP / MP rule restored to PLAN 7.3 (correctness review). Version 1's "pass fraction not worse than the 20-trial
   record" would have let an ep28 candidate pass with 1 MP pass in 3 evaluations, and section 11 did not list it.
3. Consent hold for ep14-R and ep28-W3H (rules review; PLAN 9.3): HELD until the user's explicit OK is recorded.
4. Slots for higher priorities (rules review): fillers HELD for a main-session release, RESERVE_SLOTS = 1, and the
   gate holds while an f0 clip is DEAD or CRASHED. Otherwise the wave could take the last free slots that a Tier-3
   resume or revision, or a Tier-2 job of Zuxing (invisible to the gate), needs; arms are never preempted.
5. Clarifications (correctness review, non-blocking): watcher runin_final re-evaluations do not count as candidate
   evaluations; the export-failure rule (8.3); the SP / MP check in the 20-trial confirmation (8.5).
The launcher changed accordingly (section 12). The wrapper and the fc_r2 watcher copy are unchanged. Gate self-test of
version 2: `T1I/build/gate_selftest_v2.py` (the 22 version-1 cases plus cases 22-29), output
`T1I/build/gate_selftest_v2.out`.
