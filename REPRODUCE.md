# Reproducing the Track 2 submissions of team MBZUAI-RCL Group (Tier 1, Tier 2, Tier 3)

Repository: `github.com/Beat-in-our-hearts/nv-v2d-challenge-codebase`. Every counted file on the 12 Track-2 boards
(4 metrics x 3 tiers) cites the commit that contains this file:
`https://github.com/Beat-in-our-hearts/nv-v2d-challenge-codebase/commit/<this commit>`. A commit cannot contain its own
URL, so the sha256 of the uploaded files are recorded outside Git, next to the files.

| tier | counted file (before the repack at this commit) | what it is | record folder |
|---|---|---|---|
| 1 | `T1_all7_trial05_closest_to_mean_b1e6597.csv` (`aee58640…`) | set `all7`, trial 05 = the trial closest to the 20-trial mean | `TR/t1best/t1ready_20261031/` |
| 2 | `T2_R9_closest_trial03_1dcee7e.csv` (`3129b378…`) | round 9 (R9-CLOSEST), trial 3 = the trial closest to the 10-trial mean | `TR/t2r9_20261104/` (builds on `TR/t2r8_20261103/`, `TR/t2r7_20261102/`, `TR/t2ready_20261031/`) |
| 3 | `t3_FINAL_r_a7e7d3a.csv` (`ba78a547…`) | `FINAL_r` (AMENDMENT_1 (e), AMENDMENT_2) | `TR/t3ready_20261102/` and Han Zhou's `hoi_estimation/` |

The file uploaded for each tier is the same file re-packed with this commit's URL: every row is equal except
`code_commit_url` (`scripts/final/pack_tier{1,2,3}.sh`). Abbreviations: `FC` = `robotic_grounding/flash_chord`,
`TR` = `FC/scripts/track2`, `<ME>` = the training server's work root `/root/autodl-tmp/haoran.zhang` (the placeholder
in all records), `<ZX>` = Zuxing Lu's work root there, `<KIT>` = the Oct-11 submission kit, `<STOCK>` = the stock
FlashCHORD runtime.

## Quick start: one script per step

All scripts are in `scripts/final/`. Each one takes `--dry-run` (print the exact commands, execute nothing),
`--site FILE` (this machine's paths; copy `scripts/final/site.env.example`) and `--help`. A real run writes its command
list to `$V2D_WORK/logs/<script>_<time>.sh` and runs it with `bash -euo pipefail`. Without `--site`, every path
defaults to the training server's layout, where all record paths are real. `scripts/final/selftest.sh` dry-runs all
of them (one log per call).

| step | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| environments (section 1) | `setup_env.sh` | same | same |
| data (section 3) | `get_data.sh`, then `TR/repro` (retarget) | `get_data.sh`, `build_inputs_tier2.sh` | `build_inputs_tier3.sh` (Han Zhou's runbooks + our f0 prep) |
| train (sections 4-6) | `train_tier1.sh --episode N` (stops each stage at the iteration that is used; `--full-length`; `--check-sources`) | `train_tier2.sh --episode N`; commands printed by `scripts/final/lib/print_commands_r9.py` | `train_tier3.sh --episode N` |
| evaluate | `eval_tier1.sh [--trials K]` | `eval_tier2.sh` (own input) | `eval_tier3.sh` (own input) |
| select | `TR/t1best/t1ready_20261031/amend1/` (20 trials, closest to mean) | `select_tier2.sh` | `select_tier3.sh` |
| pack | `pack_tier1.sh --commit-url URL` | `pack_tier2.sh --commit-url URL` | `pack_tier3.sh --commit-url URL` |

`scripts/final/FINAL_EPISODES.tsv` has one row per counted checkpoint (90 rows): input root and `task.parquet` sha256,
frame-reference sha256, training commits in stage order and their code in this history, training checkpoint sha256,
export commit, evaluated file sha256 (checked against the `checkpoint_sha256` column of the counted CSV) and the
counted rollout's sha256.

---

## 0. What this commit is

This commit adds `REPRODUCE.md`, `scripts/final/`, `environment/` and a short pointer section at the top of the
root `README.md` on top of the merge `ceb0186`, an octopus merge of the three tier lines (first parent Tier 2), in
two commits: `51d5232` (the runbook) and this one (the fixes of its merge test, section 10). No other file changes.
The merge had no conflicts and edited no file: every one of its 10,238 entries equals the blob of one parent.

| tier | line (branch) | head merged | contents |
|---|---|---|---|
| 2 | `dev.haoran.t2clean_20261102` | `1dcee7e` | the round-9 record (`TR/t2r9_20261104/`) on the Tier-2 line |
| 1 | `dev.haoran.t1ready_20261031` | `b1e6597` | the Tier-1 record (`TR/t1best/`) |
| 3 | `dev.hanzhou.hoi_estimation` (Han Zhou, head `c0dbf5b`) + our follow-up `a12bbd0` | `a12bbd0` | Han's reconstruction (`hoi_estimation/`, as-run code at ancestor `48c217a`; later edits are path portability) and the Tier-3 record (`TR/t3ready_20261102/`, which contains `a7e7d3a`) |

Every file of every line is in this commit byte for byte, with one exception: `FC/src` and `FC/tests` exist once,
and this commit carries the Tier-2 line's version (section 2). Commits `a94614a`, `d769f0c`, `6bb0214`, `e624294`,
`2935e78` and `8c8106f` (earlier Tier-2 records) are not in this history.

**Where to look first:** this file; then the record README of each tier (table above). Each record folder has a
`SHA256SUMS` (or lock files) for its own files.

## 1. Environment (`scripts/final/setup_env.sh`; `environment/README.md`)

| item | value |
|---|---|
| Machine (training, export, evaluation, packing) | 8x NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120, 97,887 MiB), driver 580.82.09 (CUDA 13.0); 2x Xeon Platinum 8470Q; Ubuntu 22.04.5 (container), glibc 2.35 (`environment/system.txt`) |
| Other training hosts (Tier 2, Zuxing Lu) | an RTX 5090 workstation and the robolab workstation ws031, with the same code commits (section 7) and the same `uv.lock` |
| `flashchord` env (training, export, evaluation) | conda Python 3.11.17, uv 0.13.0, then the stock runtime's `uv.lock` hash-locked (`TR/repro/env/install_flashchord.sh`): jax 0.10.2 (CUDA 13 wheels), warp-lang 1.14.0, newton 1.2.0, mujoco-warp 3.8.1. Freeze: `environment/flashchord.*` |
| `v2dkit` env (packer, validation) | a uv venv on the flashchord Python with `environment/v2dkit.pip-freeze.txt` (the kit's `requirements.txt` resolved) |
| `v2d` env (retargeting, input builders) | conda Python 3.11.17, torch 2.7.0+cu128 and torchvision 0.22.0+cu128 (from `https://download.pytorch.org/whl/cu128`, the only index with these builds), torchaudio 2.7.0 (PyPI), Isaac Sim 5.1.0 (pypi.nvidia.com), Isaac Lab v2.3.2 (installed with `CONDA_PREFIX` set to this env), numpy 1.26.0 (`TR/repro/env/install_retarget.sh`, pins `isaac-constraints.txt`). Freeze: `environment/v2d.*` |
| `zx_g2_builder` env (Tier-2 G2 inputs, Zuxing Lu) | a uv venv on CPython 3.12.3, 28 packages (`ZX_KIT_PYTHON`; `<ZX>/envs/kit` on the training server). Freeze: `environment/zx_g2_builder.pip-freeze.txt` (recorded read-only on 2026-11-04) |
| Stock runtime (the evaluator runs on it, never on this repository's `FC`) | `nvidia-isaac/video_to_data` @ `8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef` with LFS assets (`TR/repro/env/get_stock_runtime.sh`) |
| Submission kit that packed every counted file | the Oct-11 kit: `KIT_MANIFEST.json` `398947a7…`, `flash_chord/scripts/evaluate_policy.py` `160d24f5…`, `tools/pack_submission.py` `47a45895…`, `submit/v2d_submit.py` `fe5cb1dc…`, `config/tracks.json` `5799f424…`; always run with `-I -B` or `-B` |
| Tier-3 reconstruction (Han Zhou) | apptainer images and weights: `hoi_estimation/reproducibility/README.md` (`environment/weights.sha256` there) |

```bash
bash scripts/final/setup_env.sh --site site.env                 # conda, stock, flashchord, kit, retarget, zxg2, lfs
bash scripts/final/setup_env.sh --site site.env --steps check   # read-only: envs vs environment/*.pip-freeze.txt, kit hashes
```

`env_check.py` (run after each install step and by `--steps check`) fails on any package that differs from the freeze,
is missing or is extra; `--report-only` lists the differences without failing. On the training server the `v2d` env
also holds one unrelated local editable package of another project; the freeze leaves it out and a comment line there
lets `env_check.py` accept it as an extra.

**Git LFS: two sources.** The fork hosts no LFS objects for the upstream files. Clone without smudging, take the
FlashCHORD assets from upstream, and Han Zhou's 41 `hoi_estimation/agent_records/*/outputs/*.{npz,npy}` from the fork:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Beat-in-our-hearts/nv-v2d-challenge-codebase.git v2d
cd v2d && GIT_LFS_SKIP_SMUDGE=1 git checkout <this commit>
git lfs install --local && git config lfs.url https://github.com/nvidia-isaac/video_to_data.git/info/lfs
git lfs pull --include "robotic_grounding/flash_chord/**,robotic_grounding/source/robotic_grounding/robotic_grounding/assets/**,reconstruction/modules/v2d_cari4d/lib/cari4d/lib_mhr/assets/**,reconstruction/modules/v2d_sam3d_body/assets/**"
git -c lfs.url=https://github.com/Beat-in-our-hearts/nv-v2d-challenge-codebase.git/info/lfs lfs pull --include "hoi_estimation/**"
```

The two `reconstruction/modules` patterns cover the 3 runtime LFS files that Han Zhou's `run_all_tier3.sh` checks
before it starts (`lib_mhr/assets/mhr_collision_proxy_4000v.npz`, `lib_mhr/assets/mhr_hand_surface_spec.npz`,
`v2d_sam3d_body/assets/bbox_track.pt`; upstream files, so their objects are in the upstream store).

Training needs the 559 FlashCHORD asset files (`FC/src/flash_chord/assets`, meshes and the SONIC ONNX prior, 415 MB);
export and evaluation do not (the exporter's check and the evaluator use the stock runtime's assets). The assets tree
is the same in all training commits, so `train_tier*.sh --lfs-from <dir>` can fill each worktree from one filled
checkout (sha256 of each file = its pointer's oid).

## 2. Code: one tree, five training trees

Training and export ran on five code trees. All five are ancestors of this commit:

| tree | branch | trained / exported |
|---|---|---|
| `f2923d6` | `dev.haoran.c3r2` (via `41403d3`) | Tier 1 C3r2_wide stages and the ep15-B2 arm; Tier 2 "ours" items (raw, K3, K3fix); all 30 Tier-3 items. The runs used the local repository `fc_r2` @ `61fa5d3`, whose `flash_chord` code equals `f2923d6`'s; `61fa5d3` is not on GitHub |
| `b39b69b` | `dev.zuxing` (rolling start) | Tier 1 rolling-start, T1fix and T1improve stages; exporter of Zuxing's Tier-2 checkpoints. Teammate commit `0868a3f` (ep01, ep04, ep19 stages 1-3) is not on GitHub; its tree equals `b39b69b`'s |
| `7fbc227` | `dev.haoran.ep28w2` | Tier 1 ep28 (W2PD stage, W3H arm) and its export |
| `bc3b1bd` | `dev.zuxing.speedup` | Tier 1 ep04 continuation stages; Tier 2 ep04, ep05, ep26 |
| `82837c8` | `dev.zuxing.hard_start` | Tier 2 ep00, 10, 13, 16, 17, 19, 20, 21, 22, 24, 25, 28 |

`FC/src` in this commit is tree `f15d4ea5`, the Tier-2 line's: the union of the five trees, every later option off by
default. Compared with the tier heads it differs only in Tier 1 (`b1e6597`): `lifecycle/rsi/bins.py`,
`lifecycle/rsi/kernels.py`, `lifecycle/rsi_reset.py` (hard-start sampling, used only by `*_hard` configs); Tier 3
(`a7e7d3a` = `f2923d6`'s src): 9 source files and 2 tests (PSI reset, rolling start, patience, help fix, speedup,
squeegee terms; all off by default). No training tree equals `f15d4ea5`, so **exact retraining runs each stage in a
worktree of its recorded training commit**, which `train_tier*.sh` creates (`$V2D_WORK/worktrees/<sha7>`;
`--code this` uses this checkout instead). The merge test of `ceb0186` (2026-11-04) re-exported one counted item per
tier with this commit's exporter (identical tensors and metadata) and trained one recipe per tier from this commit's
`FC/src` for 600-1,000 iterations (same configuration, same learning curves within run-to-run noise). The
`flash_chord/evaluation` package is identical in this commit, all five training commits and the stock runtime.

## 3. Data

1. **Challenge data** (HF `nvidia/video_to_data_challenge`, revision `5f68335f3acc802033d1e80728c1633197521de8`):
   `scripts/final/get_data.sh` (wraps `robotic_grounding/scripts/track2/get_data.sh`; gated dataset, log in first).
2. **Tier 1:** conversion `robotic_grounding/scripts/track2/track2_to_soma_inputs.py` and G1 retargeting
   `robotic_grounding/scripts/retarget/soma_to_g1.py --save --fps 30` per episode (`robotic_grounding/scripts/track2/retarget_all.sh`;
   steps and timings `TR/repro/README.md`) -> `<ME>/data/human_motion_data`. Retargeting is not verified to be
   bit-deterministic; reuse the derived data where available (`FINAL_EPISODES.tsv` has each parquet's sha256).
3. **Tier 2** (only `track_2/tier_2_synthetic_noise` is read): `scripts/final/build_inputs_tier2.sh` -> raw
   `human_motion_data_tier2`, K3 `human_motion_data_tier2_k3`, K3fix `human_motion_data_tier2_k3fix`, G2 (Zuxing Lu)
   `<ZX>/data/human_motion_data_tier2_g2-2026-10-24`, each with its frame references. The G2 steps run in Zuxing's
   builder env (`ZX_KIT_PYTHON`, checked against `environment/zx_g2_builder.pip-freeze.txt` first). `--steps verify`
   checks the 30 inputs and frame references of the counted file (read-only).
4. **Tier 3** (video, mesh, ground plane, frame-0 human and frame-0 object pose only): `scripts/final/build_inputs_tier3.sh`
   = Han Zhou's reconstruction (`hoi_estimation/reproducibility/README.md`, `s1_runbook.md`, `tier3_runbook.md`,
   `run_all_tier3.sh`), then `TR/t3ready_20261102/data_prep/README_tier3.md` steps 0 and A-F ->
   `<ME>/data/human_motion_data_tier3_f0` and `frame_references_f0`. `--steps verify` checks the 30 locked parquets.

**Absolute paths.** Every training checkpoint and export stores the absolute `task.parquet` path of the machine that
trained it (the evaluator's `--source-root` only rewrites `/workspace` or relative paths). To evaluate the pinned
files elsewhere, make `<ME>` (and `<ZX>`) resolve to the copied data, with a symlink or a container bind mount
(`TR/repro/check_task_paths.py <exports>` lists the roots).

## 4. Tier 1 (record `TR/t1best/t1ready_20261031/README.md`)

1. **Train:** `scripts/final/train_tier1.sh --episode N` runs every stage of the episode's lineage in order (54 stages,
   30 lineages; `set_items.json` `stages`, `set_stages.tsv`) in the worktree of the stage's `code`, with the recorded
   overrides and seed 42. Hull caches: the stage's recorded cache (payload group `F_hull_caches`; `--hull-cache`), else
   the stock runtime's.
   - **Resume sources.** The record's `<RUNS>` is our `fc_runs` (`T1_RUNS`) except `<RUNS>/runin*/epNN`: Zuxing Lu's
     runin chains (ep01 stages 1-3, ep04 stages 1-5, ep19 stages 1-2), which live in
     `<ZX>/runs/2026-10-14-patience-runin` (`T1_ZX_RUNS`; it also holds ep04's counted checkpoint
     `runin_cont4/ep04/model_358000`). Fork stages (T1fix, T1improve) copy the recorded fork pair, sha256-checked
     against the record. The teammate-chain and in-place-restart pairs have no digest in the record; they are checked
     against `scripts/final/lib/resume_sha256.tsv` (taken read-only on 2026-11-04 and labelled as such in the plan).
     Three files in those folders do have a record digest, and match it: `runin_cont2/ep01/model_128000` (ep01's
     stage-4 fork pair), `runin_cont/ep19/model_90000` (ep19's stage-3 fork pair) and ep04's counted
     `model_358000` + `rsi_358000`. `train_tier1.sh --episode N [--stage S] --check-sources` checks all of these in
     place (read-only).
   - **Chains.** In-place restarts (ep00 at 4000, ep20 at 24000) and teammate chains resume in the previous stage's run
     folder. When the previous stage ran in the same replay, it leaves `handoff_N.sha256` for `model_N` + `rsi_N`, and
     the next stage resumes from exactly those files. When it did not (`--stage S` alone, or the replay's previous
     stage ended before N), the stage copies the recorded pair instead (the plan says which).
   - **Stop (default).** A stage whose recorded run was stopped from outside (end empty, `manual_plateau`,
     `auc_confirmed`, "running at selection", in-place restart) or went on past what is used later stops at the
     iteration that is used: the next stage's resume iteration, or the counted checkpoint (31 of the 54 stages; for
     example ep00 stage 1 at 4000 and stage 2 at 28000, ep02 at 44000 instead of the recorded 120,000). Stages that the
     trainer ended itself (`help_zero_tail`) are not cut. The stop is the trainer's own end, `training.max_iterations`,
     and only after `scripts/final/lib/stop_check.py` has shown that it changes nothing else: it composes the stage's
     Hydra configuration with the recorded and with the lowered value (`train_rl.py --cfg job --resolve`, no
     training) and requires every other key to be equal. In the code of all five training trees,
     `training.max_iterations` is only the loop end, the last log/save, `final.json` and the progress bar; no learning
     rate, entropy, curriculum, RSI or help schedule reads it. One config family does: the `*_helpfix` experiments of
     `bc3b1bd`/`82837c8` set `reset.help_final_iteration: ${training.max_iterations}`; no counted Tier-1 stage uses
     them, and if a stage did, the check would fall back to the recorded value. The stopped run's checkpoints record
     `training.max_iterations` = the stop iteration in their embedded configuration. `--full-length` trains every stage
     to its recorded `max_iterations` (the counted checkpoint is saved on the way, every 500 iterations).
   - `--iterations K` (smoke test): scratch stages stop at K, resumed stages at N + K; a chained stage whose previous
     stage is in the same plan resumes from that stage's smoke end.
2. **Export** (CPU, in the worktree of `set_manifest.tsv` `exported_by`, tolerance 1e-4): `eval_tier1.sh --reexport`
   runs it and `TR/t1best/compare_exports.py` (tensors and metadata must equal the evaluated file's).
3. **Evaluate:** `scripts/final/eval_tier1.sh [--trials K]`: the kit evaluator on the stock runtime, world count 1,
   per episode, on the pinned evaluated files (sha256 checked). `TR/t1best/trials_eval.py run --trials 20` is the tool
   that ran the set's 20 trials.
4. **Select:** set `all7` = `amend1/TOPUP_AMEND1.md` (sha256 in `amend1/TOPUP_AMEND1.sha256`; post-hoc amendment,
   disclosed in the record); the file is the trial closest to the 20-trial mean (trial 05).
5. **Pack:** `scripts/final/pack_tier1.sh --commit-url URL` (`repack.py`: kit packer, uploader validation, metric_code,
   row comparison; only `code_commit_url` changes).
6. **Check:** `TR/t1best/t1ready_20261031/check_set.py --records <RUNS> --exports <T1_EXPORTS> --alias ... --hash`.

## 5. Tier 2 (records `TR/t2r9_20261104/README.md`, then `TR/t2r8_20261103/`, `TR/t2r7_20261102/`, `TR/t2ready_20261031/`)

Selection used only own-input scores (the kit evaluator against each policy's own Tier-2-derived input) and the
consensus reference built from Tier-2 tracks (`refs/B`). Nothing in `scripts/final` reads or computes a clean score.

1. **Inputs:** section 3 (raw, K3, K3fix, G2).
2. **Train:** `scripts/final/train_tier2.sh --episode N`: the item's recorded stage (`set_items_r9.json`
   `train.stages[].overrides`) in the worktree of its training commit, on this machine's copy of its input parquet.
   Stages train as recorded (to their recorded `max_iterations`; the counted checkpoint is saved on the way).
   `python3 -I -B scripts/final/lib/print_commands_r9.py [--me ... --zx ...]` prints every item's training, export and
   evaluation commands from `set_items_r9.json`; it replaces `TR/t2ready_20261031/tools/print_commands_t2.py` (round
   6), which stops on the round-8/9 item format (`KeyError: 'file'`). ep10 has no run record on the team server: its
   overrides are rebuilt from the checkpoint's embedded configuration. ep19 and ep25 resume from checkpoints that exist
   only on Zuxing's RTX 5090 host (`ROOT_ZX5090`).
3. **Export:** exporter `b39b69b` for Zuxing's checkpoints, `61fa5d3` (= `f2923d6`) for ours (`eval_tier2.sh --reexport`).
4. **Evaluate:** `scripts/final/eval_tier2.sh`: frame 0, world count 1, explicit reset, no help, on the stock runtime,
   each policy against its own input.
5. **Select:** `scripts/final/select_tier2.sh` verifies the round-9 locks (PREREG_R9, LOCK_R9, POOL, RELOCATION,
   SELECTION) and prints the decision; the drivers are `selection/round9_20261103/r9_*.py` (`r9_launch.sh`).
   R9-CLOSEST = trial 3; round 9 kept 27 of round 8's picks and replaced episodes 10, 17 and 21.
6. **Pack:** `scripts/final/pack_tier2.sh --commit-url URL` (`TR/t2ready_20261031/tools/repack_t2.py` on the packed
   folder `r9_closest/`; only `code_commit_url` changes; pose columns `1380472d…`). The kit packer's clean-reference
   "locally computed" line is withheld by `scripts/final/lib/kitpy_withhold.sh`. The round's own
   `pack/t2cons_r9_pack.py pack-commit` gates on the clean-history build clone and works only there.

## 6. Tier 3 (record `TR/t3ready_20261102/README.md`; reconstruction `hoi_estimation/README.md`)

1. **Reconstruction and data:** `scripts/final/build_inputs_tier3.sh` (section 3). Agent runs are not deterministic
   (`hoi_estimation/REPRODUCTION_REPORT.md`); the frame-0 step is deterministic (re-run byte-identical in 30/30
   episodes from `c0dbf5b`).
2. **Train:** `scripts/final/train_tier3.sh --episode N`: stage f0 (`experiment=g1_recon_body_coord_rsi_r2
   reset.seed=42 training.max_iterations=45000`, PREREG_f0; `scene.support=true` for 01 02 03 06 07 09 11 13 16 17 20 25
   27 28; `termination.palm_position_threshold=0.25` for 01 02 03 15 17 19 20 28), and for the 11 retry items of
   FINAL_r (of the 14 rule-selected retries) stage retry_C90 (resume from the recorded f0 `model_44922` + `rsi_44922`,
   sha256-checked, to 90000; PREREG_v3). Code
   `f2923d6` (= `fc_r2` @ `61fa5d3`). Hull cache: `fc_r2`'s (the convex decomposition is not reproducible in hull
   order; payload).
3. **Export and evaluate:** `scripts/final/eval_tier3.sh [--reexport]` (own f0 input).
4. **Select:** `scripts/final/select_tier3.sh` verifies the locks (PREREG, AMENDMENT_1/2, PREREG_retry/v3, RETRY_FILES,
   recipes), runs `tools/check_set_t3.py --files --hash` and prints FINAL_r's composition: 5 own-input evaluations per
   item, the run closest to the item's mean; retry items replace f0 only if own-input SP/MP improve.
5. **Pack:** `scripts/final/pack_tier3.sh --commit-url URL` (`evaluation/f0_retry/t3final_r_build.py repack`; pose
   columns `82fef2c7…`; writes `<ME>/fc_runs/submissions/t3_f0_final/upload_r/<sha7>/`).
6. **Check:** `TR/t3ready_20261102/tools/check_set_t3.py --files --hash`, `tools/code_identity_t3.sh`.

## 7. Per-episode training and export commits

Training commit `61fa5d3` = code of `f2923d6`; `0868a3f` = tree of `b39b69b` (section 2). Tier-2 rows in bold are round
9's replacements. "palm 0.25" marks the Tier-3 training-only termination mitigation. Full table with digests:
`scripts/final/FINAL_EPISODES.tsv`.

| ep | object | T1 item (all7 trial05) | T1 training commit(s), stage order | T1 export | T2 item (R9) | T2 input | T2 training | T2 export | T3 item (FINAL_r) | T3 training | T3 export |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 00 | tall_bar_stool | ep00 (C3r2_runin/ep00 @28000) | b39b69b | b39b69b | ep00_z5k3fix_90000 | k3fix | 82837c8 | b39b69b | f0 @18910 | 61fa5d3 (f0) | 61fa5d3 |
| 01 | vacuum | ep01-Bc_162000 (T1improve) | 0868a3f -> b39b69b | b39b69b | ep01_k3fix_final44922 | k3fix | 61fa5d3 | 61fa5d3 | f0 @44922 (palm 0.25) | 61fa5d3 (f0) | 61fa5d3 |
| 02 | blue_easter_basket | ep02 (C3r2_runin/ep02 @44000) | b39b69b | b39b69b | ep02_k3fix_final34093 | k3fix | 61fa5d3 | 61fa5d3 | retry @90000 (palm 0.25) | 61fa5d3 (f0+retry_C90) | 61fa5d3 |
| 03 | contigo_coffee_mug | ep03 (C3r2_runin/ep03 @36000) | b39b69b | b39b69b | ep03_k3fix_final27068 | k3fix | 61fa5d3 | 61fa5d3 | f0 @19184 (palm 0.25) | 61fa5d3 (f0) | 61fa5d3 |
| 04 | baseball_bat | ep04 (runin_cont4/ep04 @358000) | 0868a3f -> bc3b1bd | b39b69b | ep04_zxg2_450000 | g2 | bc3b1bd | b39b69b | retry @90000 | 61fa5d3 (f0+retry_C90) | 61fa5d3 |
| 05 | big_black_box | ep05-B3_62000 (T1improve) | b39b69b | b39b69b | ep05_zx_k3fix_88000 | k3fix | bc3b1bd | b39b69b | retry @90000 | 61fa5d3 (f0+retry_C90) | 61fa5d3 |
| 06 | monitor | ep06 (C3r2_runin/ep06 @114000) | b39b69b | b39b69b | ep06_k3fix_final28420 | k3fix | 61fa5d3 | 61fa5d3 | f0 @44922 | 61fa5d3 (f0) | 61fa5d3 |
| 07 | iron | ep07 (C3r2_wide/ep07 @17950) | 61fa5d3 | 61fa5d3 | ep07_k3fix_final18529 | k3fix | 61fa5d3 | 61fa5d3 | f0 @20779 | 61fa5d3 (f0) | 61fa5d3 |
| 08 | skinny_wood_chair | ep08 (C3r2_runin/ep08 @28000) | b39b69b | b39b69b | ep08_k3easy_final17496 | k3 | 61fa5d3 | 61fa5d3 | f0 @17451 | 61fa5d3 (f0) | 61fa5d3 |
| 09 | big_red_bowl | ep09 (C3r2_wide/ep09 @32185) | 61fa5d3 | 61fa5d3 | ep09_raw_final39328 | raw | 61fa5d3 | 61fa5d3 | f0 @44922 | 61fa5d3 (f0) | 61fa5d3 |
| 10 | white_desk | ep10-BT_58000 (T1improve) | 61fa5d3 -> b39b69b | b39b69b | **ep10_zxrepg2_145500** | g2 | 82837c8 | b39b69b | f0 @25976 | 61fa5d3 (f0) | 61fa5d3 |
| 11 | light_blue_book | ep11 (C3r2_runin/ep11 @36000) | b39b69b | b39b69b | ep11_raw_24000 | raw | 61fa5d3 | 61fa5d3 | f0 @44922 | 61fa5d3 (f0) | 61fa5d3 |
| 12 | black_platform | ep12 (C3r2_runin/ep12 @28000) | b39b69b | b39b69b | ep12_k3fix_final23178 | k3fix | 61fa5d3 | 61fa5d3 | f0 @29267 | 61fa5d3 (f0) | 61fa5d3 |
| 13 | snack_box | ep13 (C3r2_runin/ep13 @48000) | b39b69b | b39b69b | ep13_w312runinws_90000 | g2 | 82837c8 | b39b69b | retry @90000 | 61fa5d3 (f0+retry_C90) | 61fa5d3 |
| 14 | round_table | ep14-R_92000 (T1improve) | 61fa5d3 -> b39b69b | b39b69b | ep14_k3fix_final44922 | k3fix | 61fa5d3 | 61fa5d3 | retry @90000 | 61fa5d3 (f0+retry_C90) | 61fa5d3 |
| 15 | wet_floor_sign | ep15-B2_40000 (T1improve) | 61fa5d3 | 61fa5d3 | ep15_raw_final44922 | raw | 61fa5d3 | 61fa5d3 | f0 @39300 (palm 0.25) | 61fa5d3 (f0) | 61fa5d3 |
| 16 | spray_bottle | ep16 (C3r2_wide/ep16 @24000) | 61fa5d3 | 61fa5d3 | ep16_z5a5_120000 | g2 | 82837c8 | b39b69b | retry @90000 | 61fa5d3 (f0+retry_C90) | 61fa5d3 |
| 17 | traffic_cone | ep17 (C3r2_runin/ep17 @44000) | b39b69b | b39b69b | **ep17_zxrepk3fix_74500** | k3fix | 82837c8 | b39b69b | retry @90000 (palm 0.25) | 61fa5d3 (f0+retry_C90) | 61fa5d3 |
| 18 | blue_trash_can | ep18 (C3r2_runin/ep18 @32000) | b39b69b | b39b69b | ep18_k3fix_final18911 | k3fix | 61fa5d3 | 61fa5d3 | f0 @21486 | 61fa5d3 (f0) | 61fa5d3 |
| 19 | cane | ep19-Bc_135000 (T1improve) | 0868a3f -> b39b69b | b39b69b | ep19_z5g2hardc2_360000 | g2 | 82837c8 | b39b69b | retry @90000 (palm 0.25) | 61fa5d3 (f0+retry_C90) | 61fa5d3 |
| 20 | rice_cooker | ep20 (C3r2_runin/ep20 @70000) | b39b69b | b39b69b | ep20_z5fill_90000 | k3fix | 82837c8 | b39b69b | f0 @18433 (palm 0.25) | 61fa5d3 (f0) | 61fa5d3 |
| 21 | tall_bar_stool | ep21 (C3r2_runin/ep21 @119500) | b39b69b | b39b69b | **ep21_zxrepk3fix_80500** | k3fix | 82837c8 | b39b69b | retry @90000 | 61fa5d3 (f0+retry_C90) | 61fa5d3 |
| 22 | step_ladder | ep22 (C3r2_runin/ep22 @36000) | b39b69b | b39b69b | ep22_z5k3fix_89500 | k3fix | 82837c8 | b39b69b | f0 @20827 | 61fa5d3 (f0) | 61fa5d3 |
| 23 | toy_stroller | ep23 (C3r2_runin/ep23 @28000) | b39b69b | b39b69b | ep23_raw_final20023 | raw | 61fa5d3 | 61fa5d3 | retry @90000 | 61fa5d3 (f0+retry_C90) | 61fa5d3 |
| 24 | watering_can | ep24 (C3r2_runin/ep24 @118000) | b39b69b | b39b69b | ep24_z5q_150000 | g2 | 82837c8 | b39b69b | f0 @20449 | 61fa5d3 (f0) | 61fa5d3 |
| 25 | cyan_water_bottle | ep25 (C3r2_runin/ep25 @102000) | b39b69b | b39b69b | ep25_z5a5c_280500 | k3fix | 82837c8 | b39b69b | retry @90000 | 61fa5d3 (f0+retry_C90) | 61fa5d3 |
| 26 | skateboard | ep26 (C3r2_runin/ep26 @28000) | b39b69b | b39b69b | ep26_zx_k3fix_92000 | k3fix | bc3b1bd | b39b69b | f0 @22863 | 61fa5d3 (f0) | 61fa5d3 |
| 27 | woven_basket | ep27 (C3r2_runin/ep27 @41500) | b39b69b | b39b69b | ep27_k3fix_final20863 | k3fix | 61fa5d3 | 61fa5d3 | f0 @34196 | 61fa5d3 (f0) | 61fa5d3 |
| 28 | squeegee | ep28-W3H_83000 (T1improve, AMEND1) | 61fa5d3 -> b39b69b -> 7fbc227 | 7fbc227 | ep28_z5q_180000 | g2 | 82837c8 | b39b69b | f0 @44922 (palm 0.25) | 61fa5d3 (f0) | 61fa5d3 |
| 29 | rolling_luggage | ep29 (T1fix/C/ep29 @30000) | 61fa5d3 -> b39b69b | b39b69b | ep29_k3fix_final17358 | k3fix | 61fa5d3 | 61fa5d3 | f0 @28248 | 61fa5d3 (f0) | 61fa5d3 |

## 8. Re-running only the evaluation and packing of the counted files

This is what the organisers' container check runs: for each tier, the kit evaluator on the stock runtime, world count 1,
each of the 30 evaluated (exported) checkpoints against its own input (`eval_tier{1,2,3}.sh`), then the Oct-11 kit
packer with the tier's frame references, then the uploader's offline validation. The simulator is not
bit-deterministic, so the poses differ slightly from the submitted file and the scores land within the run-to-run
spread recorded in each tier's selection records (section 10 shows it for one episode per tier). Packing a fresh
evaluation gives new pose columns; packing the recorded rollouts (`pack_tier{1,2,3}.sh`) reproduces the submitted rows
exactly except `code_commit_url`.

**Exports are not byte-deterministic.** safetensors writes the `__metadata__` keys in a per-process order, so a
re-export has a new file sha256 even with the same code (tensors, data section and metadata content are identical).
A rollout of a re-export therefore carries the re-export's `checkpoint_sha256`. Check exports by tensors
(`TR/t1best/compare_exports.py`), not by file hash; the submitted CSVs are reproduced from the pinned evaluated files.

## 9. Known gaps

- Tier 2, Zuxing Lu's code that is only committed as our read-only copies (G2 input builder, his launchers), and code
  that is not on the team server (the launchers of his RTX 5090 and ws031 runs, the first version of `train_t2.sh`):
  `TR/t2r7_20261102/MISSING_FROM_PUSHED_COMMITS.md`, `TR/t2r8_20261103/README.md` and `TR/t2r9_20261104/README.md`
  "Missing or open" (5090 run-folder listings for ep10/17/21; ep10's G2 parquet identity on the 5090). Every stage's
  Hydra overrides (or, for ep10, the embedded configuration) and first `train_t2:` line are recorded.
- Tier 2, host data: the 5090 / robolab copies of the input parquets can only be compared by bytes from the team server;
  their array identity with the server parquets is to be confirmed by Zuxing Lu.
- Tier 3: images, weights and intermediates of the reconstruction live on Han Zhou's storage (locations and sha256 in
  `hoi_estimation/reproducibility/README.md`); the agent runs (Claude Code / Codex) are not deterministic.
- Commits `61fa5d3` and `0868a3f` are local / teammate commits and are not on GitHub; tree-equal commits (`f2923d6`,
  `b39b69b`) are ancestors of this commit.
- Training and GPU evaluation are not bit-reproducible; reruns reproduce the procedure and the policy class, not the
  bytes. The convex-decomposition (hull) caches are not in git and a fresh CoACD run can order the hulls differently;
  the recorded caches travel with the payload. The exact checkpoints, exports and rollouts are kept on the training
  server at the paths in each set file (and go into the container: `weights/<tier>/`, `outputs/<tier>/`).
- The builders and drivers embed the training server's absolute paths (`<ME>`, `<ZX>`); run them in that layout
  (bind mount or symlink) or with `restore_layout_t3.py --rewrite` (Tier 3).
- Tier 1: the digests of the teammate-chain and in-place-restart resume pairs (`scripts/final/lib/resume_sha256.tsv`)
  were taken on 2026-11-04 from the files on the training server, not at training time; the set record has none.
- Tier 2: Zuxing's G2 builder env was recorded on 2026-11-04 (after the G2 build), from its live folder.

## 10. Tests of this runbook (2026-11-04, training server, GPUs 5-7, one job per GPU)

### 10.1 This commit: the fixes of the merge test of `51d5232`

The adversarial merge test of `51d5232` (PASS_WITH_NOTES) found: the retarget step could not resolve torch
2.7.0+cu128 (E1); `isaaclab.sh` could install into an active conda env (E2); the 3 runtime LFS files of
`reconstruction/modules` were not pulled (E3); Zuxing's G2 builder env had no freeze (E4); `env_check.py` could not
fail (E5); the `v2d` freeze named a private repository and a local wheel path, and the server's host name appeared
(E6); the Tier-1 teammate chains were looked up in `<ME>/fc_runs` (R1); stages trained far past the counted or next
checkpoint, and in-place restarts resumed from an unchecked file (R2); selftest logs overwrote each other (R3); and
`print_commands_t2.py` stops on the round-8/9 items. This commit fixes all of them (sections 1, 4, 5).

Run on the training server from a fresh clean worktree (docs/ excluded) of the test build `c21b0d4`, which differs
from this commit only in this subsection, a comment in `setup_env.sh` and the name of the per-stage `stop_check`
folder; `REQUIRE_IDLE_GPU=1`, GPUs 5 and 6 (idle), one job per GPU, every output under a scratch `V2D_WORK`. Nothing
was installed; no clean reference was read or scored. The same tests were repeated from a fresh worktree of this
commit.

| test | result |
|---|---|
| `selftest.sh` (syntax of every script, 23 `--dry-run` plans, `print_commands_r9.py`) | 24/24 PASS; one log per call (`NN_<script>_<args>.txt`), so the `eval_tier*` `--trials 2` and `--reexport` logs are both kept (R3) |
| R1: `train_tier1.sh --episode 1/4/19 --stage 2` and `--episode 4 --stage 5`, `--dry-run` then `--check-sources` | resume sources resolve to `T1_ZX_RUNS` (`<ZX>/runs/2026-10-14-patience-runin/runin*/epNN`); all sha256 ok against `resume_sha256.tsv`; ep04 stage 5 also checks the counted `model_358000` + `rsi_358000` against the set record (ok). `--episode 0-29 --check-sources`: 108/108 files ok (24 resume pairs, 30 counted checkpoints with their RSI states). `eval_tier1.sh --episode 4 --reexport` now checks the training checkpoint at `T1_ZX_RUNS` |
| R1: `train_tier1.sh --episode 4 --stage 2 --iterations 30` (worktree `b39b69b`, GPU 5) | no handoff (stage 1 not in the plan), so the recorded pair was copied from `T1_ZX_RUNS` (sha256 ok, digest from `resume_sha256.tsv`); resumed at 44922, stopped at 44952 (`final.json` reason `max_iterations`, 67 s); Hydra `config.yaml` equal to Zuxing's recorded `runin_cont/ep04` in 278 of 281 keys (only `output_dir`, `training.checkpoint` path, `training.max_iterations`); `overrides.yaml` same 11 keys in the same order; metrics finite |
| R2: `stop_check.py` on real stages (CPU, configuration only) | stop allowed, configuration otherwise identical (422-424 keys): ep00 stage 1 -> 4000 and stage 2 -> 28000, ep20 stage 1 -> 24000, ep02 -> 44000, ep01 stage 3 -> 128000, ep29 stage 2 -> 30000 (`b39b69b`), ep04 stage 5 -> 358000 (`bc3b1bd`). Negative control, a `*_helpfix` experiment (`82837c8`, `bc3b1bd`): `reset.help_final_iteration` follows `training.max_iterations`, so the check keeps the recorded 120000 |
| R2: `train_tier1.sh --episode 0 --iterations 30` (in-place restart chain, GPU 6) | stage 1 stopped at 30 through `training.max_iterations` (stop check: nothing else changes) and wrote `handoff_30.sha256`; stage 2 resumed from exactly that `model_30` + `rsi_30` (sha256 checked; `final.json` `start_iteration` 30, `resumed_from` `.../model_30`; the trainer moved stage 1's `final.json` aside) and stopped at 60; `config.yaml` equal to the recorded `C3r2_runin/ep00` run in 278 of 281 keys (the same three); metrics finite |
| `setup_env.sh --steps check` (read-only) | rc 0: flashchord 108, v2dkit 38, v2d 332 (plus the one allowed local editable extra), zx_g2_builder 28 packages, 0 differ / missing / extra; driver 580.82.09; kit and stock evaluator `160d24f5…`, packer, uploader; stock runtime at `8c778f9` |
| E5: `env_check.py` on temporary copies of `flashchord.pip-freeze.txt` (the env itself untouched) | unchanged: rc 0; jax 0.10.2 -> 0.10.1: rc 1 (`--report-only`: rc 0); an added fake package: rc 1 (missing); numpy removed: rc 1 (extra) |
| E1: index pages, nothing installed | `download.pytorch.org/whl/cu128` (reached directly from the training server) lists `torch-2.7.0+cu128`, `torchvision-0.22.0+cu128` and `torchaudio-2.7.0+cu128` for cp311 manylinux_2_28_x86_64; PyPI lists `torchaudio-2.7.0` cp311 x86_64, the build the env has |
| E3: `git lfs ls-files -I <the lfs step's include>` | lists the 3 runtime files (`mhr_collision_proxy_4000v.npz`, `mhr_hand_surface_spec.npz`, `bbox_track.pt`) among 1,041 |
| E6: `git grep` of the new and changed files | no host name, no private repository URL, no local wheel path |

### 10.2 Tests of `51d5232` (before the fixes)

Run on the training server from a clean detached checkout of the test build of `51d5232` (`13eef86`: the same
`scripts/final` code; `REPRODUCE.md`, `scripts/final/README.md`, the `check_sums` helper, the select scripts' lock
checks and the raw-input frame-reference check of `verify_inputs.py` were added or fixed after it and re-tested from
`51d5232`, see the last two rows). `REQUIRE_IDLE_GPU=1`; GPUs 5-7 only, one job per GPU; every output under a scratch
`V2D_WORK`. No clean reference was read or scored in any Tier-2 or Tier-3 test.

| test | result |
|---|---|
| `selftest.sh`: syntax of every script and `--dry-run` of all 18 plans (train and eval of all 30 episodes per tier) | 18/18 PASS |
| `setup_env.sh --steps check` | the three envs equal `environment/*.pip-freeze.txt` (108 / 38 / 333 packages: 0 differ, 0 missing, 0 extra); driver 580.82.09; kit `evaluate_policy.py` = stock = `160d24f5…`, packer `47a45895…`, uploader `fe5cb1dc…`; stock runtime at `8c778f9` |
| `get_data.sh --verify-only` | 313 files, every one downloaded at revision `5f68335f` |
| `build_inputs_tier2.sh --steps verify`, `build_inputs_tier3.sh --steps verify` | 30/30 Tier-2 own-input parquets and frame references, 30/30 Tier-3 f0 parquets and frame references, f0 manifest `d496584c…` and `FILES.sha256` equal the records |
| `select_tier2.sh`, `select_tier3.sh` | record hash files verified (files kept out of git by design are counted, not checked); `check_set_t3.py --files --hash` ALL OK; decisions printed |
| `eval_tier1.sh --episode 28 --trials 2` (pinned file `497ae3b3…`) | AUC 0.8190 / 0.8074, SP-SR 1 / 1, MP-SR 0 / 1, MPPE 3.37 / 2.37 cm. The 20 recorded trials: AUC 0.8034-0.8224, MP pass 15/20, MPPE 1.63-3.61 cm |
| `eval_tier2.sh --episode 17 --trials 2` (pinned `81b57136…`, K3fix own input) | AUC 0.7912 / 0.7906, SP-SR = MP-SR = 1, MPPE 1.80 / 1.82 cm. The 10 recorded round-9 phase-F runs: AUC 0.7810-0.7956, SP = MP = 1, MPPE 1.74-1.89 cm |
| `eval_tier3.sh --episode 7 --trials 2`, then `--trials 4` (pinned `a85f1cf1…`, f0 own input) | AUC 0.9870, 0.9817, 0.9835, 0.9872, 0.9868, 0.9863 (mean 0.9854), SP-SR = MP-SR = 1 in 6/6, MPPE 0.69-0.74 cm. The 6 recorded runs: mean 0.9867 (sd 0.0015), range 0.9839-0.9880, MPPE 0.672-0.726 cm |
| `train_tier1.sh --episode 8 --iterations 30` (C3r2_runin, scratch) | worktree `b39b69b` created, 559 LFS assets filled (sha256 = pointer oid), recorded hull cache (27 hulls); `final.json` max_iterations at 30, `model_30` + `rsi_30`; saved `config.yaml` equal to the recorded run's in 224 keys except `output_dir`, `training.max_iterations`; overrides same keys in the same order; metrics finite |
| `train_tier2.sh --episode 17 --iterations 30` (Zuxing's recipe at `82837c8`, K3fix) | worktree `82837c8`, stock hulls (29); `final.json` at 30; `config.yaml` equal except `output_dir`, `task.parquet` (this server's copy, sha256 `25c13883…`), `training.max_iterations` |
| `train_tier3.sh --episode 7 --stage f0 --iterations 30` | worktree `f2923d6`, `fc_r2` hulls (29); `final.json` at 30; `config.yaml` equal except `output_dir`, `training.max_iterations` (223 keys) |
| resumed stages: `train_tier1.sh --episode 28 --stage 4 --iterations 30` (T1improve arm W3H at `7fbc227`), `train_tier3.sh --episode 2 --stage retry_C90 --iterations 30` | the recorded resume pairs copied with sha256 equal to the record (`b674e5e5…`/`d74eb956…`, `ba0973a6…`/`2121f804…`); runs resumed at 68000 and 44922 and stopped at 68030 and 44952; `config.yaml` equal except `output_dir`, `training.checkpoint` (path), `training.max_iterations` |
| `pack_tier1.sh` with a placeholder URL | kit packer, uploader validation ok, 22,989 rows, metric_code scores equal, only `code_commit_url` differs from `aee58640…` |
| `pack_tier2.sh`, `pack_tier3.sh` | dry run only: the kit packer always computes a clean-reference score, which the team's Tier-2/3 rule keeps out of these tests. The same tools re-packed both files from the merge `ceb0186` earlier on 2026-11-04 (Tier 2 byte-identical to round 9's own placeholder pack; Tier 3 byte-identical to FINAL_r's locked CSV, and only `code_commit_url` differing from the `a7e7d3a` file) |
| `51d5232`, from a fresh clean checkout | all PASS (`6717e04`, which differs from `51d5232` only in this table row): `selftest.sh` 18/18; the read-only real runs exit 0, with every present file listed by the record hash files matching (t2r9 `SHA256SUMS` 511/511, t3ready `SHA256SUMS` 980/980, PREREG / LOCK / SELECTION / AMENDMENT / retry locks; `POOL_R9` 186 of 250 and `RELOCATION_R9` 2 of 16 present, the rest are checkpoint copies kept out of git) and 30/30 Tier-2 frame references; one evaluation per tier: Tier 1 ep28 AUC 0.8154 (SP, MP pass, MPPE 2.56 cm), Tier 2 ep17 0.7902 (SP = MP = 1, 1.81 cm), Tier 3 ep07 0.9884 (SP = MP = 1, 0.69 cm); `train_tier3.sh --episode 7 --stage f0 --iterations 30` stopped by `max_iterations` at 30 with `config.yaml` equal to the recorded run's except `output_dir`, `training.max_iterations`; `pack_tier1.sh` (placeholder URL): validation ok, only `code_commit_url` differs |
| `51d5232`, adversarial merge test (fresh sparse worktree, GPU 6) | `selftest.sh` 18/18; `setup_env.sh --steps check` rc 0 (108 / 38 / 333 packages equal); `eval_tier2.sh --episode 9 --trials 2` on the pinned file `c73692c5…`: AUC 0.78226 / 0.78132, SP = MP = 1, MPPE 1.673 / 1.700 cm; `--reexport` (worktree `f2923d6`, CPU): same policy (72/72 tensors, 11/11 metadata keys), AUC 0.79181 (the 25 recorded own-input runs: mean 0.7757, range 0.6604-0.7917); plans equal the hydra / W&B / embedded records for 17 spot-checked episodes; verdict PASS_WITH_NOTES (findings fixed in 10.1) |
