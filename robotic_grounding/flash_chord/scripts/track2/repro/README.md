# Track 2 Tier 1, method C3r2: one-command reproduction

This folder reproduces our Track 2 Tier-1 submission, method **C3r2** (arm `C3r2_wide`, experiment
`g1_recon_body_coord_rsi_r2`). It covers the challenge's reproducibility rule 2.9: the code commit, the trained
weights (hosted separately, see [Weights and access](#weights-and-access)) and one command that goes from the
competition data to the submission file.

The file to cite is `C3r2_tier1_final_run1.csv` (sha256 `620a341d101d923fd9893b425efa079fbd1eaed1285e46c2da77c0e1c9d03dc6`).
It was produced by one final evaluation run of the 30 pre-registered final checkpoints. It scores AUC 0.7939,
SP-SR 0.867, MP-SR 0.800 and MPPE 12.26 cm in the evaluator's own metrics.

## Quick start

Get the code without Git LFS downloads (this fork hosts no LFS objects), then take the assets from the upstream
repository, as explained in [Git LFS](#git-lfs):

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --no-checkout <this repository's URL> v2d && cd v2d
GIT_LFS_SKIP_SMUDGE=1 git checkout <this commit>
git lfs install --local
git config lfs.url https://github.com/nvidia-isaac/video_to_data.git/info/lfs
git lfs pull --include "robotic_grounding/flash_chord/**,robotic_grounding/source/robotic_grounding/robotic_grounding/assets/**"
```

Then:

```bash
cd robotic_grounding/flash_chord/scripts/track2/repro
cp config.env.example config.env        # set the paths, GPUs and interpreters
bash reproduce.sh --dry-run             # print every command of every step; runs nothing
bash reproduce.sh                       # data -> conversion -> retargeting -> training -> export -> evaluation -> CSV
bash reproduce.sh --steps eval,pack     # only evaluate existing runs and pack
bash reproduce.sh --episodes 0,3 --steps convert,retarget,train   # a trial on two episodes
```

The steps can be selected with `--steps`. They always run in this order:

| Step | Script | What it does | Where it runs |
|---|---|---|---|
| `setup` (optional) | `env/get_stock_runtime.sh`, `env/install_flashchord.sh` | Fetches the stock runtime `nvidia-isaac/video_to_data@8c778f9` (`robotic_grounding/flash_chord` with LFS). Builds the hash-locked flashchord environment. | network |
| `data` | `robotic_grounding/scripts/track2/get_data.sh` | `hf download nvidia/video_to_data_challenge --repo-type dataset --revision 5f68335f3acc802033d1e80728c1633197521de8 --include 'track_2/*'` | network |
| `convert` | `robotic_grounding/scripts/track2/track2_to_soma_inputs.py` | Converts Track 2 LeRobot episodes into `soma_to_g1.py` input folders. Writes `soma_params.npz`, `poses.npy` (invisible object frames interpolated), `ground_plane.json`, the object mesh, `reconstructed_stage/<seq>_support.usda` and `track2_manifest.json`. | CPU, seconds |
| `retarget` | `robotic_grounding/scripts/track2/retarget_all.sh` | Runs `soma_to_g1.py <motion>/whole_body/soma/sequence_id=<seq> --save --fps 30 --motion-root <motion>` per episode, round-robin over the GPUs, at most `RETARGET_JOBS_PER_GPU` (default 4) at a time per GPU. Output: `robot_name=g1/data.parquet` plus the object mesh and URDF in `object/`. | GPU, ~17 min on 8 GPUs |
| `train` | `train_all.sh` | One PPO run per episode with the recorded command (below). | GPU, 10-27 h per run |
| `export` | `final_eval.sh --export-only` → `export_all.sh` | Exports the final checkpoint named by `final.json` to the stock ActorCritic format at tolerance 1e-4, then checks the stored task paths (`check_task_paths.py`). | CPU |
| `eval` | `final_eval.sh` → in-repo `scripts/track2/eval_all.py`, then `summarize.py` | Checks the stock runtime commit, the evaluator digest and the kit manifest (`check_eval_inputs.py`). Runs the kit's evaluator on the stock runtime: one world, one episode per call, one job per GPU. Writes the per-episode table `<work>/eval/C3r2_wide_final.csv`. | GPU, ~6 min on 8 GPUs |
| `pack` | `pack.sh` | Runs the kit's `tools/pack_submission.py --track track_2_tier1`, then `tools/preflight.py --skip-kaggle`. Nothing is uploaded. | CPU |
| `repeats` (optional) | `repeat_eval.sh` | Re-runs the evaluator on the same exported files to measure its spread. Diagnostic only. | GPU |

## Per-episode training command

`train_all.sh` runs this for episode `NN` (sequence `<seq>`, object `<obj>`). The working directory is
`robotic_grounding/flash_chord`, with `PYTHONPATH=robotic_grounding/flash_chord/src` and the flashchord `python`
first on `PATH`:

```
CUDA_VISIBLE_DEVICES=<gpu> python scripts/train_rl.py experiment=g1_recon_body_coord_rsi_r2 \
  "task.parquet='<motion>/whole_body/soma/sequence_id=<seq>/robot_name=g1/data.parquet'" reset.seed=42 \
  output_dir=<work>/runs/C3r2_wide/epNN training.max_iterations=45000 logging.mode=<WANDB_MODE> \
  [logging.entity=<WANDB_ENTITY>] [logging.project=<WANDB_PROJECT>] logging.experiment=C3r2_wide \
  logging.run_name=epNN_<obj> logging.upload_checkpoints=false [scene.support=false]
```

- **`scene.support=false`** is added for floor-only episodes, whose `<seq>_support.usda` contains no `def Cylinder`
  or `def Cube`. In Tier 1 these are episodes 00, 04, 05, 08, 10, 12, 14, 15, 18, 19, 21, 22, 23, 24, 26 and 29.
- **Run end.** A run ends at the Coordinates_hand early stop or at 45,000 iterations. The early stop needs 5,860
  iterations with zero help in every RSI start bin. At either end, the runner writes `final.json`, naming the run's
  final checkpoint. All 30 submitted runs stopped early, between iterations 17,272 and 44,922; 7 of them at
  44,922, the latest the deadline allows.
- **Checkpoint selection.** The checkpoint that `final.json` names is the one evaluated (`queue_helper.py final`,
  `eval_all.py --checkpoint-from-final`). It was fixed before any evaluation; there is no best-of selection.
- **Failures.** A run that stops without `final.json` is started once more, resuming from its newest finite
  checkpoint. A run directory that holds checkpoints but no usable resume point is never started fresh.
- **W&B** is off by default (`logging.mode=disabled`). With `WANDB_MODE=online` plus `WANDB_ENTITY` and
  `WANDB_PROJECT`, the override list is identical to the recorded one. Export `WANDB_API_KEY` in your shell, never
  in a file.

`tests/check_train_commands.py` compares the dry-run commands with the W&B records of the submitted runs:
`args`, program and git commit of all 30 `wandb-metadata.json` files, and the original queue log. It ignores the
paths.

## What actually ran

- **Code.** The checkpoints were trained, exported and evaluated from a local commit, `61fa5d31ff8054bf1e5cf5375ef0957e54d2729b`, which
  W&B records as `git.commit`. That commit was never pushed. Its 11 commits were replayed onto this repository as
  `f2923d61817e61eb37d939e0f75e5b4a7aaa5cbf`, the commit the submitted CSV cites. All 335 tracked files under
  `robotic_grounding/flash_chord` are identical. The only exceptions are 8 documentation GIFs, stored here as LFS
  pointers with identical content. Outside `flash_chord`, f2923d6 changes nothing relative to its base commit 7c0d3b9,
  so `robotic_grounding/scripts/retarget/soma_to_g1.py` is the retargeter that was used. This commit adds tooling
  only. `tests/check_code_unchanged.sh` fails if anything under `flash_chord/src`, `train_rl.py`, the three existing
  `scripts/track2/*.py`, the package metadata, `uv.lock` or `scripts/retarget` differs from f2923d6.
- **Data preparation.** `track2_to_soma_inputs.py` and `replay_check.py` are byte-identical to the files that ran
  (md5 `912f294a7e08441ff74b6defd256e7ad` and `0a22e207261c215d30e79610abf7e657`). Before this commit they were not
  in the repository. `replay_check.py` is an optional Isaac Lab replay check and is not part of the pipeline.
- **Data download.** It fetched both tiers of Track 2 (313 files, 2.1 GB) at revision `5f68335f…`, which every
  file's download metadata records. Only `tier_1_multiview_caption` is read, by the converter and by the packer.
- **Training launcher.** Training was launched by a machine-specific queue. `train_all.sh` reproduces its
  per-episode command, its completion rule (`final.json`) and its single resume retry. The original packed **4 jobs
  per GPU on 8 GPUs under an NVIDIA MPS daemon**, with each job pinned to its own CPU slice. That packing changes
  only speed. `train_all.sh` defaults to one job per GPU (`JOBS_PER_GPU`) and uses neither MPS nor CPU pinning.
  Episode 04's first attempt died with a CUDA context error before its first checkpoint and was started again from
  scratch; every `final.json` has `resumed_from: null`.
- **`scripts/track2/launch.py` is not what ran, and must not be used for this line.** It ports an earlier queue:
  - it treats a run as done only when `model_<iterations>.safetensors` exists, so it does not recognise the
    `final.json` early stop, which all 30 runs used;
  - it names run directories `ep<NN>_<object>`;
  - it adds `+logging.local_jsonl=true`, which the submitted runs did not have.

  It is left unchanged because this commit does not modify existing files.
- **Export.** Each export ran on the CPU with `--tolerance 1e-4`, 4 at a time. The measured `max_rel_diff` was
  8.1e-6 to 6.47e-5, so all 30 were accepted. The exporter's default of 1e-5 would refuse most of them.
- **Evaluation.** It used the official kit's `examples/run_policy_evaluation.py` and its
  `flash_chord/scripts/evaluate_policy.py` (sha256 `160d24f5…`, equal to the stock runtime's), outside MPS, through
  `eval_all.py`. The stock runtime was a clean checkout of `8c778f9`.
- **Packing** used the kit packer exactly as `pack.sh` does, with `--code-commit` set to the f2923d6 URL. Packing
  the existing results again with the kit packer reproduces the submitted CSV byte for byte.
- **Re-packing.** The cited CSV's `code_commit_url` points to f2923d6, which has the same code but not this folder.
  Packing the same results with this commit's URL instead changes only the `code_commit_url` column.

## Compute environment (rule 2.9.c)

| | |
|---|---|
| Machine | 1 node, 8× NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB, sm_120), driver 580.82.09 |
| CPU / RAM | 2× Intel Xeon Platinum 8470Q (208 hardware threads), 1 TB |
| OS | Ubuntu 22.04.5 LTS (container), glibc 2.35 |
| flashchord env | Python 3.11.17 and the 8c778f9 `uv.lock`, installed with uv 0.13.0 and `--require-hashes` (`env/requirements.flashchord.txt`, regenerated and compared by `env/install_flashchord.sh`). It includes jax 0.10.2 (CUDA 13 wheels), warp-lang 1.14.0, newton 1.2.0, mujoco-warp 3.8.1, onnxruntime-gpu 1.28.0 and wandb 0.22.2. |
| retargeting env | Python 3.11, Isaac Sim 5.1.0, Isaac Lab 2.3.2 (rsl_rl), torch 2.7.0+cu128, numpy 1.26.0, pin 3.7.0, pin-pink 4.2.0, py-soma-x 0.2.1, warp-lang 1.18.0, scipy 1.15.3. Built by `env/install_retarget.sh`, with `robotic_grounding/workflow/Dockerfile` as the reference. `env/retarget-env-freeze.txt` records every package of the environment that made the data (no package changed after the retargeting run); the installer reports any difference from it. |
| packer env | Python 3.11.17 venv with the kit's `requirements.txt` |
| Training budget | 30 runs. Each iteration is 4,096 parallel worlds × 24 steps. Each run took 10.1-27.1 h at 4 jobs per GPU; the total is 483 job-hours. Wall clock: 2026-10-12 17:43 to 2026-10-13 21:01 (UTC+8), about 27.3 h. |
| Other steps | Conversion: seconds, CPU. Retargeting: about 17 min on 8 GPUs (all episodes at once, about 4 per GPU). Export: minutes, CPU. Evaluation of 30 episodes: about 6 min on 8 GPUs. |
| Disk | Data 2.1 GB. Derived motion data (`whole_body/`) 1.3 GB: the converter's output about 390 MB, the retargeting output about 850 MB (the 30 G1 reference parquets, 106 MB, plus each episode's `object/` mesh, texture and URDF, 740 MB). Run directories about 47 GB, with checkpoints every 500 iterations. The 30 exported files: 1.3 GB. |

## Environments

- **flashchord.** Build it with `bash env/install_flashchord.sh <env dir> $V2D_RUNTIME` (needs `uv`), and set
  `FC_PYTHON=<env dir>/bin/python`. Every FlashCHORD step sources `env/fc_env.sh`. That puts the env first on
  `PATH`, adds the env's pip-installed CUDA 13 / cuDNN / NCCL / NVSHMEM libraries to `LD_LIBRARY_PATH` (needed by
  onnxruntime-gpu) and sets `XLA_PYTHON_CLIENT_PREALLOCATE=false`. The Docker image in
  `robotic_grounding/flash_chord/docker` (CUDA 13.0.2, `uv sync --frozen`) is an equivalent reference.
- **Stock runtime.** `bash env/get_stock_runtime.sh <dir>` fetches it, then set
  `V2D_RUNTIME=<dir>/robotic_grounding/flash_chord`. It must stay a clean `8c778f9` checkout: the evaluator runs
  against it, never against this repository's `flash_chord`. Running the evaluator writes only git-ignored files
  into it (the convex-decomposition cache, see [Caveats](#caveats)). `final_eval.sh` and `repeat_eval.sh` first run
  `check_eval_inputs.py`: the runtime's HEAD must be the line's `STOCK_RUNTIME_COMMIT`, both copies of
  `evaluate_policy.py` must have the line's `EVAL_SCRIPT_SHA256`, and every kit file must match `KIT_MANIFEST.json`.
  `pack.sh` runs the kit part. `SKIP_INPUT_CHECK=1` skips these checks.
- **Retargeting.** `env/install_retarget.sh` runs inside a Python 3.11 environment that already has Isaac Sim
  5.1.0 and Isaac Lab 2.3.2. It pins the versions that made the data, stages the SOMA-X body model through
  `scripts/setup_soma_assets.py`, and lists every package whose version differs from `env/retarget-env-freeze.txt`.
  Before conversion and retargeting, `env/retarget_env.sh` sources `RETARGET_ACTIVATE` when it is set and pins
  `PIP_CONSTRAINT` to `env/isaac-constraints.txt`. `retarget_all.sh` runs at most `RETARGET_JOBS_PER_GPU` episodes
  (default 4) at a time on each GPU; the original ran all 30 at once on 8 GPUs.
- **Submission kit.** Use the official kit (`V2D_KIT` is the folder with `KIT_MANIFEST.json`), with a Python 3.11
  venv built from its `requirements.txt` (`KIT_PYTHON`). The pipeline runs kit Python only with `-B` /
  `PYTHONDONTWRITEBYTECODE=1`, so the kit stays identical to its manifest.

### Git LFS

This fork does not host LFS objects, so a plain `git clone` with git-lfs installed fails at checkout. Clone with
`GIT_LFS_SKIP_SMUDGE=1` (see [Quick start](#quick-start)), point LFS at the upstream repository and pull two folders:

```bash
git lfs install --local
git config lfs.url https://github.com/nvidia-isaac/video_to_data.git/info/lfs
git lfs pull --include "robotic_grounding/flash_chord/**,robotic_grounding/source/robotic_grounding/robotic_grounding/assets/**"
```

- `robotic_grounding/flash_chord/**` (about 420 MB): the FlashCHORD robot and scene assets, used by training,
  export and evaluation.
- `robotic_grounding/source/robotic_grounding/robotic_grounding/assets/**` (about 430 MB): the G1 + Dex3 URDF
  meshes. `soma_to_g1.py` loads `assets/urdfs/g1/main_with_hand.urdf` with pinocchio, which fails on LFS pointer
  files ("Could not load resource").
- `git lfs pull --include "robotic_grounding/**"` also works (1.6 GB). It adds the motionbricks planner models
  (715 MB), which this pipeline does not use.

Every one of the 1,043 LFS files under `robotic_grounding/` is served by the upstream repository and is identical
(sha256) to the file the runs used. `reproduce.sh` stops before `retarget`, `train` or `export` when the folder that
step needs still holds LFS pointer files. A repository-level `.lfsconfig` would set `lfs.url` automatically. This
commit does not add one, because LFS uploads from this fork would then also go to the upstream repository.

## Configuration

`config.env` (from `config.env.example`) is sourced by `reproduce.sh`. Each variable can also come from the
environment.

| Variable | Meaning |
|---|---|
| `V2D_WORK` | Root of everything written: `runs/`, `eval/`, `submissions/`, `logs/` |
| `V2D_DATA` | Dataset root containing `track_2/`. Also the packer's `--dataset-root`. |
| `V2D_MOTION` | Derived motion data (`HUMAN_MOTION_DATA_DIR`) |
| `V2D_RUNTIME` | Stock runtime `.../robotic_grounding/flash_chord` at 8c778f9 |
| `V2D_KIT` | Extracted submission kit |
| `FC_PYTHON`, `RETARGET_PYTHON`, `RETARGET_ACTIVATE`, `KIT_PYTHON` | Interpreters (see [Environments](#environments)) |
| `GPUS`, `JOBS_PER_GPU` | GPUs, and training jobs per GPU. Evaluation always runs one job per GPU. |
| `RETARGET_JOBS_PER_GPU` | Retargeting jobs at a time per GPU (default 4) |
| `FLASH_CHORD_CACHE_DIR` | Optional: where FlashCHORD reads and writes its convex-decomposition cache (see [Caveats](#caveats)) |
| `SKIP_INPUT_CHECK`, `SKIP_PATH_CHECK` | `1` skips `check_eval_inputs.py` or `check_task_paths.py` |
| `WANDB_MODE`, `WANDB_ENTITY`, `WANDB_PROJECT` | Logging, `disabled` by default |
| `CODE_COMMIT_URL`, `SUBMISSION_CSV` | The packed CSV's commit URL (default: this checkout's pushed HEAD) and its path |

`lines/c3r2_tier1.env` holds the method's fixed values, every one as used by the submitted run:
- experiment, seed, iteration cap and expected floor-only episodes;
- selection rule and export tolerance;
- dataset revision and stock runtime commit;
- the trained and cited commits.

## Caveats

- **Training is not bit-reproducible.** GPU contact simulation and parallel reductions are not deterministic, and
  job packing changes timing. A retrained policy will differ from the submitted weights. Compare scores, not
  digests.
- **The evaluator is not deterministic either.** Four evaluator runs of the *same* 30 exported files gave AUC 0.7939
  (the submitted run), 0.7948, 0.8202 and 0.7957; for example, episode 03 flips between success and failure.
  `repeat_eval.sh` measures this spread. A reproduction from the same weights should land inside it, not exactly on
  0.7939.
- **Cite the single-run file.** A per-episode best of those four runs (AUC 0.8299) exists, but it cannot be
  reproduced by one command, so it is not the file to cite.
- **The packer's cross-check warns.** When packing these results, the kit packer recomputes AUC from the
  trajectories and reports small differences from the evaluator's JSON on 24 episodes (for example episode 02:
  0.9423 vs 0.9426). Its local total is AUC 0.7937, against 0.7939 from the JSON. The leaderboard scores the
  trajectories. The submitted file is byte-identical to this re-pack, so the same applies to it.
- **Export is not byte-deterministic** (safetensors header order). Re-exporting gives new `checkpoint_sha256`
  values; reproducing the CSV's digests needs the exact evaluated files.
- **Retargeting determinism** has not been verified. For an exact reference trajectory, use the same derived motion
  data as the run (see [Weights and access](#weights-and-access)).
- **Convex-decomposition cache.** FlashCHORD builds collision hulls with CoACD and caches them in
  `src/flash_chord/assets/.cache/convex` of the tree that runs (git-ignored), keyed by the mesh and the parameters.
  CoACD is not deterministic. For the same keys, 18 of the 29 cached decompositions differ between the training
  tree and the stock runtime that evaluated the submission, 10 of them in vertex count. A fresh machine builds new
  hulls. To reuse the original ones, host both caches (about 6 MB each) and point `FLASH_CHORD_CACHE_DIR` at a
  directory that holds the right one as `convex/`: the training cache for `train`, the stock runtime's cache for
  `eval` and `repeats`. FlashCHORD and the stock evaluator both read this variable, and `eval_all.py` passes it on
  to the evaluator.
- **Verbatim data scripts.** `track2_to_soma_inputs.py` and `replay_check.py` are kept byte-identical to the files
  that ran (md5 above), so they keep their original one-line SPDX header and a few lines over 120 columns. Do not
  run the `robotic_grounding` pre-commit hooks on them; the license and formatting hooks would rewrite them.

### Absolute task.parquet paths

Every checkpoint stores, in its metadata (`user.resolved_config_json`), the **absolute** path of the episode it was
trained on (`task.parquet`). The evaluator loads the reference motion from that path. In the stock harness
(`flash_chord/evaluation/harness.py`, `_resolve_parquet`, at 8c778f9), `--source-root` rewrites the path only when it
lies under `/workspace` or is relative. The submitted checkpoints store a path from the training machine, so
`--source-root` has no effect and the files evaluate only where that path exists.

Workaround (needed for the hosted weights, not for policies you retrain yourself):
1. Run `python check_task_paths.py <checkpoint dir>`. It prints every stored path and the motion-data root the
   checkpoints expect.
2. Make that root resolve to your derived motion data: a symlink
   (`mkdir -p "$(dirname <root>)" && ln -s "$V2D_MOTION" <root>`) or a container bind mount
   (`-v "$V2D_MOTION":<root>:ro`).
3. Run `final_eval.sh` or `repeat_eval.sh`. Both run this check first and stop if a path is missing; set
   `SKIP_PATH_CHECK=1` to bypass it.

Rewriting the stored path instead would change the files' digests.

## Weights and access

The weights are not in Git. **Hosting them and inviting the organizers are the repository owner's job:**
- Host the 30 exported files that were evaluated (1.3 GB; their sha256 values equal the CSV's `checkpoint_sha256`).
  Optionally also host the raw final checkpoints (`model_N` and `rsi_N` plus `final.json` per episode).
- Host the derived motion data next to them. The evaluator's reference trajectory is our own retarget, and
  retargeting determinism is unverified. At least the retargeting output is needed (each episode's
  `robot_name=g1/` and `object/`, about 850 MB); the whole `whole_body/` tree is 1.3 GB.
- Host both convex-decomposition caches (training tree and stock runtime, about 6 MB each), see
  [Caveats](#caveats).
- Give the organizers' GitHub account read access to this private repository.

To evaluate the hosted exported files directly:
1. Write a map `{"0": "<file>", ..., "29": "<file>"}` with absolute paths.
2. Run `repeat_eval.sh --checkpoints map.json --repeats 1`.
3. Run `pack.sh --results <work>/eval/C3r2_wide_repeats/rep1`.

To re-export from the hosted raw checkpoints instead:
1. Place them as `<work>/runs/C3r2_wide/epNN/`.
2. Run `reproduce.sh --steps export,eval,pack`.

## Files

| File | Purpose |
|---|---|
| `reproduce.sh` | The one command (steps above; `--dry-run`) |
| `config.env.example` | Settings template |
| `lines/c3r2_tier1.env` | The method's fixed values |
| `train_all.sh` | Per-episode training queue with the recorded command |
| `queue_helper.py` | Run-directory checks: `done`, `resume`, `final` (checkpoint selection) |
| `export_all.sh` | Final checkpoint → stock format (tolerance from the line) |
| `final_eval.sh` | View directory, export, path check, `eval_all.py`, summary |
| `summarize.py` | Per-episode table and means from the evaluator's own JSON |
| `check_task_paths.py` | Checks the absolute task paths stored in checkpoints |
| `check_eval_inputs.py` | Checks the stock runtime commit, the evaluator script digest and the kit manifest |
| `repeat_eval.sh` | Evaluator spread on fixed files; also evaluates hosted files directly |
| `pack.sh` | Kit packer and preflight |
| `env/` | Environment builders: `install_flashchord.sh`, `fc_env.sh`, `requirements.flashchord.txt`, `get_stock_runtime.sh`, `install_retarget.sh`, `retarget_env.sh`, `isaac-constraints.txt`, and the record `retarget-env-freeze.txt` |
| `tests/run_checks.sh` | CPU-only checks: syntax, unchanged code, hygiene, support rule, recorded commands |
| `../../../../scripts/track2/` | Data preparation: `get_data.sh`, `track2_to_soma_inputs.py`, `retarget_all.sh`, `replay_check.py` |

To run the checks on CPU (the record paths point at the original run's directory and queue log):

```bash
bash tests/run_checks.sh --motion-root "$V2D_MOTION" --records <runs>/C3r2_wide --queue-log <queue.log>
```

`tests/check_hygiene.sh` checks only generic patterns: home-directory paths, proxy and package-index settings,
credential formats, unexplained 40-hex strings, LFS-tracked and binary files. To also reject your own names, hosts or
mirrors, pass them at check time as a regular expression, for example
`HYGIENE_EXTRA_PATTERNS='name1|name2|mirror\.example' bash tests/run_checks.sh`, rather than writing them into the
repository.
