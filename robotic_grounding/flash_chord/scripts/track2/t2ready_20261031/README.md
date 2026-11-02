# Track 2 Tier 2: inputs, training records and selection rounds 0-6

This folder holds the code and records that the Tier-2 submission files of this commit build on: the round-7 file
(`../t2r7_20261102/`) and the K3-only file (`../t2k3only_20261102/`). It covers the Tier-2 input builders, the training
records, and selection rounds 0, 1 and 6, with round 6's set **R6-CLOSEST** (round 6, phase F, trial 9; superseded by
round 7). Selection uses only own-input and consensus-B (noisy) scores.

**Training commit evidence (2026-11-02).** The training commit of every checkpoint of round 6's set is evidenced,
including Zuxing's 6 remote RTX 5090 checkpoints: their training logs, copied from his run copy, start with
`commit=82837c8`. The evidence is summarised in "Training commit evidence" below.

Paths in the records keep three placeholders:
- `<ME>` is `/root/autodl-tmp/haoran.zhang` (Haoran Zhang's workdir on the training server).
- `<ZX>` is `/root/autodl-tmp/zuxing.lu` (Zuxing Lu's workdir on that server).
- `<ZX5090>` is `/mnt/gfs/home/zuxinglu/workspace` (Zuxing's remote RTX 5090 workspace).

## Round 6's set (R6-CLOSEST)

`set/set_items.json` is the authoritative list. `set/set_manifest.tsv` is a one-line-per-episode summary.

The `code_in_this_history` notes of `set/set_items.json` predate this commit. The merge commits they name for bc3b1bd
and 82837c8 (773c645, 8c8106f) are not in this commit's history; bc3b1bd and 82837c8 are its parents. "flash_chord
identical to 61fa5d3" there means its src/, scripts/ and configs. The file is kept unchanged because
`verification/check_set_report.json` records its sha256.

| ep | checkpoint (job) | by | input | training commit |
|---:|---|---|---|---|
| 00 | ep00_zxb_k3fix_60000 | Zuxing (remote 5090) | K3fix | 82837c8 (training log) |
| 01 | ep01_k3fix_final44922 | Haoran | K3fix | 61fa5d3 |
| 02 | ep02_k3fix_final34093 | Haoran | K3fix | 61fa5d3 |
| 03 | ep03_k3fix_final27068 | Haoran | K3fix | 61fa5d3 |
| 04 | ep04_zxg2_421500 | Zuxing (pro6000) | G2 | bc3b1bd (training log) |
| 05 | ep05_zx_k3fix_88000 | Zuxing (pro6000) | K3fix | bc3b1bd (training log) |
| 06 | ep06_k3fix_final28420 | Haoran | K3fix | 61fa5d3 |
| 07 | ep07_k3fix_final18529 | Haoran | K3fix | 61fa5d3 |
| 08 | ep08_k3easy_final17496 | Haoran | K3 | 61fa5d3 |
| 09 | ep09_raw_final39328 | Haoran | raw | 61fa5d3 |
| 10 | ep10_raw_final34267 | Haoran | raw | 61fa5d3 |
| 11 | ep11_raw_24000 | Haoran | raw | 61fa5d3 |
| 12 | ep12_k3fix_final23178 | Haoran | K3fix | 61fa5d3 |
| 13 | ep13_k3fix_final39254 | Haoran | K3fix | 61fa5d3 |
| 14 | ep14_k3fix_final44922 | Haoran | K3fix | 61fa5d3 |
| 15 | ep15_raw_final44922 | Haoran | raw | 61fa5d3 |
| 16 | ep16_zxb_g2_34000 | Zuxing (remote 5090) | G2 | 82837c8 (training log) |
| 17 | ep17_raw_final31807 | Haoran | raw | 61fa5d3 |
| 18 | ep18_k3fix_final18911 | Haoran | K3fix | 61fa5d3 |
| 19 | ep19_zxb_g2hard_248000 | Zuxing (remote 5090) | G2 | 82837c8 (training logs of all 3 stages) |
| 20 | ep20_zxb_k3fix_30000 | Zuxing (remote 5090) | K3fix | 82837c8 (training log) |
| 21 | ep21_k3fix_final44922 | Haoran | K3fix | 61fa5d3 |
| 22 | ep22_zxb_k3fix_78000 | Zuxing (remote 5090) | K3fix | 82837c8 (training log) |
| 23 | ep23_raw_final20023 | Haoran | raw | 61fa5d3 |
| 24 | ep24_raw_final44922 | Haoran | raw | 61fa5d3 |
| 25 | ep25_zxb_k3fix_48000 | Zuxing (remote 5090) | K3fix | 82837c8 (training log) |
| 26 | ep26_zx_k3fix_92000 | Zuxing (pro6000) | K3fix | bc3b1bd (training log) |
| 27 | ep27_k3fix_final20863 | Haoran | K3fix | 61fa5d3 |
| 28 | ep28_zx_g2_102000 | Zuxing (pro6000) | G2 | bc3b1bd (training log) |
| 29 | ep29_k3fix_final17358 | Haoran | K3fix | 61fa5d3 |

Inputs (each derived from the released Tier-2 input):
- **raw** is the released `tier_2_synthetic_noise` episode, converted and retargeted to G1.
- **K3** is raw with the object track replaced by our K3 Kalman/RTS smoother.
- **K3fix** is K3 with the pre-committed rest-test and anchor fix.
- **G2** is raw with the object track replaced by Zuxing's G2 batch MAP smoother, which starts from his variant of K3.

Own-input scores of the file are evaluator values, each policy against its own input. They are not leaderboard
numbers. Trial 9 has AUC 0.7721, SP-SR 0.933, MP-SR 0.933 and MPPE 3.39 cm. The 10-trial mean is 0.7719 / 0.933 /
0.930 / 3.52 cm, and the selection metric (consensus-B AUC) is 0.6567.

## Where the code is

Everything below is in this commit or its history, except where noted. Scripts added on 2026-11-03 from the team
server are listed with source, sha256 and times in `CODE_COPIES.tsv` (each file's mtime predates its first use).

**Training code.** Each stage ran `robotic_grounding/flash_chord/scripts/train_rl.py` with the Hydra overrides in
`set/set_items.json`.
- Haoran's 20 checkpoints were trained in a local tree at 61fa5d3. Its `robotic_grounding/flash_chord` src/, scripts/ and
  configs are identical to f2923d6, the parent of 41403d3 on `dev.haoran.c3r2`, so they are in this history. The
  experiment is `g1_recon_body_coord_rsi_r2` (C3r2).
- Zuxing's 4 checkpoints trained on this server (ep04, 05, 26, 28) used bc3b1bd (`dev.zuxing.speedup`), a parent of
  this commit. The experiment is `g1_recon_body_coord_rsi_r2_runin`. The first line of each training log says
  `commit=bc3b1bd`. The logs are in `records/zuxing_runs/pro6000_logs/`.
- His 6 remote RTX 5090 checkpoints (ep00, 16, 19, 20, 22, 25) used 82837c8 (`dev.zuxing.hard_start`), a parent of
  this commit. The first line of every training log, including all 3 stages of ep19, says `commit=82837c8`. The logs are in
  `records/zuxing_runs/round7_5090_copy/`. ep19 used `g1_recon_body_coord_rsi_r2_runin_hard`; the others used
  `g1_recon_body_coord_rsi_r2_runin`. The hard-start options default to the old behaviour.
  - This tree's Hydra configs (`src/flash_chord/configs`) are identical to 82837c8's.
  - This tree differs from 82837c8 in two source files, `objectives/recon_body.py` and
    `training/ppo/coordinate/export.py`, which come from 7fbc227 (another parent). To retrain these 6 exactly, check out
    82837c8.

**Inputs.**
- `data/ours/raw/`: the Tier-2 retargeting driver. The converter is `robotic_grounding/scripts/track2/track2_to_soma_inputs.py`
  with `--track-root .../track_2/tier_2_synthetic_noise`, and `logs/convert.log` is its output. The driver sources
  `activate.sh`, the environment script of Haoran's workdir (copied 2026-11-03).
- `data/ours/k3/`: the K3 design steps d1 to d3, the filter `kfilter.py` with `kalman_core.py` and `kcommon.py`, and the
  full-frame track and data-root builder `e1_k3_fullframe.py` / `e2_build_eval_inputs.py`, with their pre-registration.
- `data/ours/k3fix/`: the pre-committed fix `rootcause_ep04/k3fix_v1_precommit.py` and its null calibration
  (`rootcause_ep04/s6_null_calibration.py`), plus the builders `k3fix_build.py tracks|gates` and `k3fix_build_data.py`,
  and the build records and data-root `READY`.
- `data/ours/gates/`: `k3full_gates.py`, the per-episode K3fix-or-raw input rule (gates G1 to G4).
- `data/ours/k3ca/`: the K3CA contact-aware track. It is used only in the selection's consensus references; no submitted
  checkpoint trained on it.
- `data/zuxing_g2/`: Zuxing's G2 code, copied from his workdir with sha256 checks. `data/zuxing_g2/README.md` gives
  the provenance.

Zuxing's copies of `kalman_core.py`, `kcommon.py` and `kfilter.py` are byte-identical to ours.

**Training launchers.** Byte-identical copies of the launcher files on the team server. Where a file was rewritten
after a run that is in one of the two files had started, this is said below.
- `train/ours/`:
  - `track2_queue_tier2.sh` (with `track2_queue_helper.py` and `fc_env.sh`) launched the raw `C3r2_tier2` runs that
    started from 2026-10-16 22:18 on. The scheduler session before (2026-10-16 13:23:59 to about 22:18) ran the
    previous version, committed as `track2_queue_tier2.v2_20261016.sh` (the server file
    `scripts/track2_queue_tier2.sh.v2_20261016`, copied 2026-11-03). That session started the raw runs of ep00, 01, 02,
    07, 16, 18 and 20; of these, only ep00's (`ep00_raw_final41043`, K3-only file) is in one of the two files. The
    two versions differ in the MPS clients' `CUDA_VISIBLE_DEVICES`, the `MAX_TRIES` default and logging; the training
    command is the same;
  - `k3easy_driver.sh` with `k3easy_helper.py` the K3 easy-clip runs (ep08, 20, 23);
  - `k3fix_driver.sh` with `k3fix_helper.py` the K3fix pilot (ep03, 07, 18);
  - `k3full_driver.sh` with `k3full_helper.py` the full K3fix arm;
  - `track2_queue_tier2v2_helper.py`: the GPU, CPU-slice and process primitives that the three k3 helpers import
    (copied 2026-11-03).

  The K3fix arm's job list and per-episode rule are in `train/ours/prereg/k3full/` (`PREREG_v3.md`, `jobs.json`,
  `VERDICT.json`); the other drivers' pre-registrations are not included. Every training stage's Hydra overrides are
  recorded (`set/set_items.json`, `records/`), and `tools/print_commands_t2.py` prints the training commands.
- `train/zuxing/`: his launchers (`train_t2.sh` and `run_*.sh`), the evaluation watcher that the `run_*.sh` scripts
  start (`eval/eval_watch.sh` with `eval_watch_helper.py`, copied 2026-11-03; its own-input `metrics.csv` rows
  designated his ep05 88000 and ep26 92000 under round 1's Rule Z, `AMENDMENTS.md` A3) and his bundle notes for the
  remote checkpoints (`bundles/*/README.txt`, edited copies that keep the run paths and inputs;
  `../t2r7_20261102/REDACTIONS.tsv`).
  - `train_t2.sh` is the live file, last written on 2026-10-24 at 05:19:09. It launched his runs started from 05:20 on
    (e.g. the G2 runs of `run_g2_runin.sh`, ep04). His ep05 / ep26 K3fix run (`run_ep05_ep26_runin_k3fix.sh`) started
    at 04:35:10, before that write; the version of `train_t2.sh` it ran is not on the team server
    (`../t2r7_20261102/MISSING_FROM_PUSHED_COMMITS.md`).
  - These are his pro6000 launchers. The RTX 5090 runs used a variant of `train_t2.sh`: its log line has no `cpus=`
    field, and it ran from his `worktrees/dev.zuxing.hard_start`. That variant, his RTX 5090 launchers and his run
    wrapper `launch_run.sh` are not on the team server (`../t2r7_20261102/MISSING_FROM_PUSHED_COMMITS.md`).
  - `run_a5_runin.sh` names `worktrees/dev.zuxing.speedup`, but the remote ep16 and ep25 logs show the hard_start
    worktree and `commit=82837c8`.
- `records/zuxing_runs/`: the training records of his 10 checkpoints in the set, copied read-only with sha256 checks.
  `COPIES_t2final.tsv` lists the source, the copy, and the before and after hashes.
  - `round7_5090_copy/` holds `MANIFEST.tsv` and `SHA256SUMS` from his copy of the remote runs
    (`<ZX>/runs/2026-11-01-round7-5090-ckpts`, made 2026-11-01). For each of the 8 run folders behind the 6 remote
    checkpoints, it also holds `train.log`, `final.json`, `hydra/train_rl.log` and `hydra/.hydra/*.yaml`.
  - `pro6000_logs/` holds the training logs of his 4 checkpoints trained on this server.

**Export.** Each training checkpoint was converted to the stock format with
`python -m flash_chord.training.ppo.coordinate.export` at tolerance 1e-4 (ep04: 2e-4), with the stock-runtime check.
Haoran's checkpoints used the 61fa5d3 tree and Zuxing's used b39b69b. Each record is in `records/exports/`.

**Evaluation.** The official kit evaluator was run as `examples/run_policy_evaluation.py` with
`flash_chord/scripts/evaluate_policy.py`, on the stock runtime nvidia-isaac/video_to_data 8c778f9. It used world count 1
and the default seed 42, one episode per call, against the input parquet stored in each checkpoint. The driver was
`scripts/track2/eval_all.py` (b39b69b tree) under the round drivers.

**Selection.** `selection/scripts/` holds the drivers, scorers and packers of rounds 0, 1 and 6, the two modules of
rounds 2 and 5 that later rounds load (`t2cons_r2_consensus.py`, `t2cons_r5_relocate.py`), round 1's A1 evaluation
driver, launcher and resolver (`t2cons_a1_eval.py`, `t2cons_a1_eval_launch.sh`, `t2cons_a1_resolve.py`), and the
collision-hull check `cache_keys_t2.py` (run by round 0's driver; pinned by PREREG_R6 and PREREG_R7; copied
2026-11-03). Round 0's inventory builder, pinned by its PREREG, is
`selection/round0_t2best5_20261024/inventory/build_inventory.py` (copied 2026-11-03). The kinds of scripts that this
record does not include are listed in `../t2r7_20261102/README.md` ("Record scope").
`selection/round0_*`, `round1_*` and `round6_*` hold those rounds' pre-registration, pins, amendments, manifests,
own-input evaluation tables, consensus scores, selection files and summaries. Of rounds 3 and 5, the record includes
the score files that round 7 pins (consensus scores and job stats in `scores_r3/` and `scores_r5/`), round 5's
`SELECTION_R5.sha256`, which round 7 also pins, and the relocation records of both rounds (`RELOCATION_R3.*`,
`RELOCATION_R5.*`). The other files of `round3_*/` and `round5_*/` are withheld (`../t2r7_20261102/REDACTIONS.tsv`),
among them round 5's `selection_r5.json` and `PREREG_R5.md`, which round 7 also pins (`../t2r7_20261102/README.md`,
"Record scope"). Rounds 2 and 4 are not included (`../t2r7_20261102/REDACTIONS.tsv`).
- The round-6 rule is in `selection/round6_20261031/PREREG_R6.md` (frozen as sha256 0b0fbf9a...; this copy is
  shortened, see `../t2r7_20261102/REDACTIONS.tsv`).
- Phase S screened 250 candidates with 3 runs each. Phase F ran 10 paired trials of the 30 incumbents and 4 finalists.
- ep04 and ep11 changed. The set replaced R5-CLOSEST under PREREG_R6 5.6.
- R6-CLOSEST is the trial whose 30-episode own-input mean AUC is closest to the 10-trial mean, which is trial 9. It is
  not the best trial.
- `SELECTION_R6.sha256` (0689840123a6de7f...) locks `selection_r6.json` (506b601e...).

**Packing.** The kit packer `tools/pack_submission.py --track track_2_tier2 --frame-references <dir>` packed the
trial-9 results. The frame reference of each episode is its own policy input on the released frame grid: the K3fix, K3
or G2 track, or the released raw track. `tools/repack_t2.py` re-packs those same results with this commit's URL.

## Reproduce

1. **Environment and data**: as in `../repro/README.md`, which describes the FlashCHORD env, the stock runtime 8c778f9,
   the kit and Git LFS. Then:
   - `bash robotic_grounding/scripts/track2/get_data.sh <V2D_DATA>` downloads HF `nvidia/video_to_data_challenge` at
     revision 5f68335f.
   - `python robotic_grounding/scripts/track2/track2_to_soma_inputs.py --track-root <V2D_DATA>/track_2/tier_2_synthetic_noise --motion-root <ME>/data/human_motion_data_tier2 --episodes all`
     converts the Tier-2 episodes.
   - `bash data/ours/raw/retarget_tier2.sh "<gpu>" <jobs>` runs `scripts/retarget/soma_to_g1.py --save --fps 30` per episode.
2. **K3**: in `data/ours/k3/kalman/scripts`, run `d1_avail_frames_variograms.py`, `d2_mesh_grasp_physics.py`,
   `d2b_rigid_carry_anchor_fit.py` and `d3_params_ml_dryrun.py` (v2d env, CPU). Then run `../eval/e1_k3_fullframe.py`,
   then `../eval/e2_build_eval_inputs.py` with the kit env. The second writes the K3 data root and the K3 frame
   references. `data/human_motion_data_tier2_k3` is a byte copy of it plus the Tier-2 manifest (`READY`).
3. **K3fix**:
   - `k3fix_build.py tracks`, then `k3fix_build.py gates` (v2d env).
   - `k3fix_build_data.py` (kit env) writes `data/human_motion_data_tier2_k3fix` and `frame_refs_k3fix`.
   - `k3full_gates.py gates` gives the per-episode input rule: K3fix where G1 to G4 pass, raw otherwise.
   - K3CA: `k3ca_build.py tracks|gates`, then `k3ca_build_data.py`.
4. **G2**: see `data/zuxing_g2/README.md`. The steps are `kfix/run_fix.py agile+hgate`, then
   `G2_surface_only/run_all.sh <out>`, then `g2data/build_g2_data.py <root> <episodes>`.
5. **Train, export, evaluate**: `python tools/print_commands_t2.py --me <ME> --zx <ZX>` prints, per episode:
   - every training stage, with its saved overrides and data root;
   - the export command;
   - the evaluator call.

   Use the training commit given for each episode. This tree composes the same configuration (Verification below).
6. **Select**: the round drivers in `selection/scripts` with the records of each round. Round 6 is
   `t2cons_r6_pool.py`, then `t2cons_r6_eval_launch.sh --phase screen|final`, then `t2cons_r6_consensus.py`, then
   `t2cons_r6_pack.py`. The evaluator is not deterministic on GPU. A rerun therefore gives a new set of trials, and the
   rule takes the trial closest to their mean.
7. **Pack**: `tools/repack_t2.py` re-packs recorded results with a commit URL (see its docstring and
   `../t2r7_20261102/README.md`). The CSV it writes is identical to the source pack except for `code_commit_url`.

## Training commit evidence (2026-11-02)

`tools/t2final_20261102.py evidence` wrote `verification/training_commit_evidence.json` (ALL OK). It checked Zuxing's
copy of his remote RTX 5090 runs, `<ZX>/runs/2026-11-01-round7-5090-ckpts`, which he made on 2026-11-01. The text
records it read are the read-only copies in `records/zuxing_runs/`. It read the checkpoints in his folder in place.

**Training commit.** All 8 run folders behind the 6 remote checkpoints give the same commit, 82837c8.

| ep | checkpoint | run in his copy | training log, first line |
|---:|---|---|---|
| 00 | model_60000 | `2026-10-24-tier2-runin-k3fix/ep00` | `commit=82837c8` |
| 22 | model_78000 | `2026-10-24-tier2-runin-k3fix/ep22` | `commit=82837c8` |
| 16 | model_34000 | `2026-10-28-tier2-runin-a5/ep16` | `commit=82837c8` |
| 25 | model_48000 | `2026-10-28-tier2-runin-a5/ep25` | `commit=82837c8` |
| 20 | model_30000 | `2026-10-28-tier2-5090-fill/ep20` (`scene.world_count=2048`) | `commit=82837c8` |
| 19 | model_248000 | `2026-10-25-tier2-runin-hard-g2/{ep19, cont/ep19, cont2/ep19}` | `commit=82837c8` in all 3 |

Four other sources agree with the log line:
- the `commit` column of his `MANIFEST.tsv`;
- the code paths in each log, which are all under `worktrees/dev.zuxing.hard_start`;
- the Hydra `runtime.cwd` of each run;
- `git rev-parse`: 82837c8 resolves uniquely to 82837c8569c2461f263c28fce406c1b517f40d93, the head of
  `dev.zuxing.hard_start`, which is an ancestor of this commit.

**Each checkpoint against its run.**
- The run's `hydra/.hydra/{config,overrides,hydra}.yaml` is byte-identical to the record already committed from his
  bundles. ep19's base stage had no record until now; its saved `overrides.yaml` equals the reconstruction used before.
  It is now `records/hydra/ep19_zxb_g2hard_248000/s1`.
- The run's overrides match its MANIFEST row: experiment, data root, sequence and support.
- The resolved config stored in the checkpoint:
  - equals the run's saved `config.yaml`, resolved, on every key;
  - equals the stored config of every checkpoint the copy keeps for that run, 2 to 19 per run;
  - names the run's `output_dir`.
- The stored iteration lies in the run's range, and the stored environment steps equal iteration times the run's
  steps per iteration. That rate is 98,304, or 49,152 for ep20's 2,048 worlds.
- **sha256.**
  - Every checkpoint the copy keeps matches his `SHA256SUMS`.
  - The copy keeps ep20 model_30000. Its sha256 equals the as-trained file we evaluated, c10003d2....
  - The other 5 exact checkpoints are not in the copy, which keeps only some checkpoints of each run: the last two,
    plus every 10,000th for some runs. Their sha256 is pinned by his bundle `SHA256SUMS` lines and re-hashed on our
    as-trained copies.
  - The relocated file we evaluated has the same tensor bytes as the as-trained file. Only the stored config's
    `task.parquet` / `reference.parquet` differ.

**ep19's three stages.**
- The base run finished at 180,000 (`final.json`).
- `cont` resumed from the base run's `model_180000` and saved up to `model_224000`. It then stopped with "No space left
  on device" at about 62%. The saved `model_224000` is complete: its size matches its header.
- `cont2` resumed from `cont/model_224000`; its overrides and `final.json` `resumed_from` say so, with start iteration
  224000. It ran to 360,000, and model_248000 lies in that range.
- Across stages, the stored configs differ only in `output_dir`, `training.resume`, `training.checkpoint` and
  `training.max_iterations`.

**His 4 pro6000 checkpoints.** The training logs of ep04, ep05, ep26 and ep28 say `commit=bc3b1bd`. The checkpoint in
each of his run folders has the same sha256 as the set's file.

**No robolab checkpoint.** Rounds 2 and 4 relocated and evaluated 5 of his checkpoints from his robolab and ws031
machines. None of them
is in this set, checked by sha256 against every checkpoint, as-trained copy and evaluated file. The set holds checkpoints
from this server (24) and his RTX 5090 (6) only.

**Not checkable.** His launcher prints `git log -1` of the worktree, so uncommitted edits would not show in the log.

## Verification (run on the training server before this commit)

- `tools/check_set_t2.py --files --hash` gave `verification/check_set_report.json`: ALL OK. It was re-run on
  2026-11-02 with the updated `set/set_items.json`.
  - 32 training stages compose with this tree's configs.
  - The 25 stages with a saved Hydra record reproduce its `config.yaml` key by key, all keys, and its `overrides.yaml`.
    That now includes ep19's base stage.
  - For all 30 checkpoints, the resolved composition equals the resolved config stored in the checkpoint, all keys,
    with no new or missing key.
  - For the 6 relocated remote checkpoints, it equals the as-trained copy exactly. The relocated file differs only in
    `task.parquet`, by the recorded path.
  - Every evaluated file names its checkpoint as its export source. The sha256 of every checkpoint, as-trained copy and
    evaluated file matches its pin.
- `verification/reexport_t2.sh` re-exported all 30 checkpoints with this tree's exporter.
  `verification/reexport_summary.tsv` shows 30/30 the same policy as the evaluated files (tensors and metadata).
  This was run on 2026-10-31; the exporter and every other code file are byte-identical in this commit.

## Notes

- **Relocation.** The 6 remote checkpoints store the remote parquet path. Our copies rewrite only `task.parquet` and
  `reference.parquet` in the stored config, to this server's path of the same parquet (sha256 checked). Weights are
  unchanged. The records are `selection/round3_20261028/RELOCATION_R3.*` and `selection/round5_20261029/RELOCATION_R5.*`.
- **Reference bias.** The consensus references (round 1 `refs/B`) include the G2 track only on episodes 1, 4, 9, 19, 21
  and 28, and the K3CA track only on 0, 5, 15, 19, 22, 26 and 28. ep04's winner is a G2 policy.
- **Round 6 timing.** The round-6 pack and runner pins (`PREREG_R6_PINS_c.md`) were installed after the phase-F
  evaluations, but before any phase-F score. The reason was server access outages. No rule changed.
- **Open item.** Whether his worktrees had uncommitted edits when his runs started cannot be checked from our side;
  only Zuxing can confirm it. The training commit of every checkpoint is evidenced from its training log (2026-11-02,
  above).
