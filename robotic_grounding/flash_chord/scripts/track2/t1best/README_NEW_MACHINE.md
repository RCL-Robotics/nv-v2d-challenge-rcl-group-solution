# Track 2 Tier 1 best set (2026-10-26): continue the experiments on a new machine

This folder holds everything needed to keep working on the current Tier-1 best set somewhere else:
- the per-item training records of the 31 pinned checkpoints;
- the fork arms of the Tier-1 fix (T1fix) experiments;
- the 20-trial evaluation of the set;
- portable versions of the tools that produced them: fork training, a checkpoint watcher, and repeated official
  evaluation with kit packing and scoring.

Checkpoints, exports, data and hull caches are **not** in Git. They travel as a separate payload, which has its own
manifest (see [Copy the payload](#3-copy-the-payload)). Nothing here uploads anything.

## What this branch is

`dev.haoran.t1best` is built from `dev.haoran.mix` (10a3af0), with these changes:

| Commit | What it brings | Why it is here |
|---|---|---|
| merge of 7fbc2273a92b (`dev.haoran.ep28w2`) | ReconBody `object_task` options, both off by default: position-only reward after the final release, and the `object_no_drop` term. The E2 exporter strips the new fields. | It trained and exported ep28 W2PD `model_66000`, the set's episode 28 |
| merge of bc3b1bdf34f2 (`dev.zuxing.speedup`, which brings bbde02f and 731df7f) | Rollout speedup (results unchanged); success-driven RSI help, off by default (`help_mode=deadline`); an RSI loader that also reads sidecars with `pass_ema`/`fail_ema`; the rolling-start launcher notes | It trained ep04 `runin_cont3`/`runin_cont4` (`model_358000` is in the set). Forking ep04 needs its RSI loader. |
| this folder | configs, records and tools | — |

Both merges were clean, with no conflicts.

**Compatibility checks (run on the original machine, CPU only):**
- **Configs.** Every pinned item's training stage, composed with this tree's Hydra configs, matches the run's saved
  `config.yaml`: 31 of 31, no changed keys, no new keys (`check_items.py --compose --records`).
- **Exporter.** This tree's exporter is byte-identical to 7fbc227's. Compared with b39b69b's, it only adds the
  stripping of the new fields. Re-exports of ep00 (trained at b39b69b), ep07 (61fa5d3) and ep28alt (7fbc227)
  produce the same tensors and metadata as the evaluated files (`compare_exports.py`).
- **Unit tests.** The objective, RSI, rolling-start, help-fix and speedup unit tests pass on CPU. Tests that need a
  GPU were skipped or failed for lack of one. `test_per_group_ppo::test_k3_per_group_functions_match_numpy_loops`
  fails on CPU at 1e-7 float equality; it fails the same way at b39b69b.

## The best set

There is one item per episode; `best_set/manifest.tsv` and `configs/items/<item>.json` hold the details:
- the source run and iteration;
- the sha256 of the training checkpoint, RSI sidecar and export;
- the commits that trained and exported it;
- the hull cache it trained with;
- its previous and 20-trial scores;
- every training stage, with its Hydra overrides (`configs/pinned_stages.tsv`).

**What the items are:**
- **19 rolling-start runs** (`C3r2_runin`), trained at b39b69be6c2f.
- **4 C3r2 runs** (`C3r2_wide` ep07, 09, 15, 16), trained at 61fa5d31ff80. Its `flash_chord` files equal f2923d61817e.
- **6 T1fix forks:**
  - ep10, ep19, ep14 and ep01 use arm B (rotation weight);
  - ep29 uses arm C (plain continuation);
  - ep28alt is W2PD.
- **ep04** is the teammate's `runin_cont4` `model_358000`.
- **ep28** (B `model_58000`) is kept as an alternative. It is not in the set: `best_set/BANK_ep28_RULE.md` chose
  W2PD.

**20 official trials of the same 30 exported files** (`best_set/summary_20trials.md`, `per_trial_20trials.csv`,
`per_item_20trials.csv`): there, ep28 was B and W2PD was scored alongside it.

**Re-packing the 20 trials with W2PD as episode 28** (`trials_eval.py summary` over the same evaluator results;
`best_set/per_trial_20trials_w2pd_repack.csv`) gives:

| | AUC | SP-SR | MP-SR | MPPE cm |
|---|---:|---:|---:|---:|
| Mean of 20 trials | 0.9254 (sd 0.0050) | 0.9967 | 0.9733 | 1.08 |
| Trial 01, the one closest to the mean | 0.9251 | 1.000 | 0.967 | 0.99 |

Trial 01's re-pack equals `hypothetical_set_auc_with_alt` of the original `per_trial.csv` (0.92508).

**Weak items** (20-trial AUC):
- ep25 water bottle 0.748
- ep04 bat 0.751
- ep21 stool 0.755
- ep01 vacuum 0.758; once collapsed to 0.24
- ep28 squeegee 0.807 (W2PD)
- ep14 0.827
- ep29 0.854
- ep10 0.869

## 1. Get the code

The fork hosts no Git LFS objects. Clone without LFS downloads, then take the assets from the public upstream:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --branch dev.haoran.t1best <this repository's URL> v2d && cd v2d
git lfs install --local
git config lfs.url https://github.com/nvidia-isaac/video_to_data.git/info/lfs
git lfs pull --include "robotic_grounding/flash_chord/**"
```

Training needs the FlashCHORD assets (meshes, SONIC encoder/decoder `.onnx`). All of them are identical to the
upstream LFS objects. Retargeting is not needed: the payload carries the retargeted motion data. The full LFS story is
in `../repro/README.md` ("Git LFS").

## 2. Environments

These are the same as `../repro/README.md` ("Environments"):

- **flashchord** (training, export, evaluation, and these tools). Build it with
  `bash ../repro/env/install_flashchord.sh <env dir> <stock runtime flash_chord dir>`; this needs `uv`, and the
  versions are hash-locked from the 8c778f9 `uv.lock`.
  - The original machine had RTX PRO 6000 Blackwell GPUs (sm_120) on driver 580.82.09, with JAX's CUDA 13 wheels.
    Any GPU and driver that run CUDA 13 should work.
  - The tools put the env's pip `nvidia/{cu13,cudnn,nccl,nvshmem}/lib` on `LD_LIBRARY_PATH` themselves, like
    `../repro/env/fc_env.sh`. onnxruntime-gpu needs them.
- **Stock runtime.** Run `bash ../repro/env/get_stock_runtime.sh <dir>`; it fetches nvidia-isaac/video_to_data at
  8c778f9 with LFS. The evaluator runs against it, never against this tree. Then copy the payload's
  `caches/stock_8c778f9/convex/*.npz` into `<dir>/robotic_grounding/flash_chord/src/flash_chord/assets/.cache/convex/`.
  Without them the evaluator rebuilds hulls with CoACD, which is not deterministic, and scores can drift from the
  20-trial numbers.
  - The payload's optional group `H_stock_runtime_optional` is a copy of that `flash_chord` tree, without `.git`.
    Use it only if the fetch is impossible; `get_stock_runtime.sh` is the clean way.
- **Kit.** The payload has the extracted kit (`v2d_submission_kit_src/`). Build a Python 3.11 venv from its
  `requirements.txt` (`KIT_PYTHON`); the packer and `score_csv.py` use it.

## 3. Copy the payload

On the original machine, the payload is described outside Git, in `transfer/t1best_20261026/`:
- `MANIFEST.tsv` lists every file: group, path in the payload, size, sha256, source and note;
- `lists/<group>.txt` holds the paths of each group;
- `payload/` is a tree of symlinks to the source files;
- `copy_example.sh` and `README.txt` explain the copy.

The default groups are about 6 GB:

| Group | Content |
|---|---|
| `A_pinned` | the 31 pinned training checkpoints and their RSI sidecars (`runs/<run>/{model,rsi}_<N>.safetensors`) |
| `B_exports` | the 31 evaluated exports, `exports/<item>/model_<N>.safetensors`, with their export reports |
| `C_run_records` | Hydra records, `final.json`, and the T1fix launch and fork records of every stage |
| `E_fork_origins` | the fork-point pairs of the T1fix stages, at the paths their overrides name |
| `F_hull_caches` | the stock and training-tree caches, and the two seeded T1fix caches |
| `G_motion_data` | the derived motion data. It is **required** and must be byte-identical. |
| `G_challenge_data` | the challenge dataset |
| `G_kit` | the submission kit |
| `R_records` | protocols, bank rules, tables |
| `S_server_scripts` | the original machine-specific scripts, for reference only |

Optional groups:
- `D_metrics`: training curves, 1 GB.
- `H_stock_runtime_optional`.

`X_teammate_ok` is not staged. It holds the ep04 RSI sidecar, which lives only in the teammate's run folder. It is
needed only to resume or fork ep04, and copying it needs his OK.

Copy with `copy_example.sh` after adapting it. It runs `rsync -aL`, because `payload/` holds symlinks. Choose the
groups with `COPY_GROUPS`, not `GROUPS`, which bash reserves:

```bash
COPY_GROUPS="A_pinned B_exports C_run_records E_fork_origins F_hull_caches G_motion_data G_challenge_data G_kit R_records S_server_scripts" \
    DEST=user@newhost:/data/t1best/payload bash copy_example.sh
```

Every pinned pair sits in its real run folder, `runs/<pinned.run>/` of `configs/items/<item>.json`, next to its run
records. The original machine's `T1fix/watch*` folders are symlinks to those runs and are not in the payload.

After copying, check the payload:

```bash
python verify_transfer.py --manifest <payload>/MANIFEST.tsv --root <payload>             # sha256, default groups
python verify_transfer.py --manifest <payload>/MANIFEST.tsv --root <payload> --quick     # sizes only
python verify_transfer.py --manifest <payload>/MANIFEST.tsv --root <payload> --groups D_metrics   # an optional group
```

Without `--groups` it checks the default groups. `--groups all` also checks `D_metrics`, `H_stock_runtime_optional`
and the unstaged `X_teammate_ok` row.

### Absolute task paths stored in the checkpoints

Every training checkpoint and every export stores the absolute `task.parquet` path of the original machine. The
stock evaluator loads that exact path: its `--source-root` only rewrites `/workspace` or relative paths.

1. Find the root those files expect:

   ```bash
   python ../repro/check_task_paths.py <payload>/exports
   ```

2. Make that root resolve to the copied `data/human_motion_data`, with a symlink or a container bind mount. The
   transfer `README.txt` gives the literal path.

New forks store the new machine's path, so this matters only for the transferred files.

## 4. Site settings

```bash
cp site.env.example site.env    # set T1_WORK, T1_RUNS, T1_EXPORTS, V2D_MOTION, V2D_DATA, V2D_RUNTIME, V2D_KIT, FC_PYTHON, KIT_PYTHON
set -a; source site.env; set +a # or pass --site site.env to every tool
```

`site.env` holds machine paths and is ignored by Git (`.gitignore` in this folder).

**Machine-specific options:**
- `GPUS` / `EXCLUDE_GPUS`. The original machine never used GPU 4.
- `REFUSE_MPS=1` skips GPUs that run an MPS server. Set it if training runs under MPS: an evaluation past its
  timeout is stopped as a process group, and a signal to one MPS client can take down every job of that GPU.
- `EVAL_CPUS` pins evaluations to CPU lists per GPU group (NUMA). Quote the value, since it contains `;`.
- `EVAL_LOCKDIRS` holds other drivers' lane locks, to share GPUs fairly.

Run every tool with the flashchord python.

Then run the CPU checks: syntax, configs, hygiene, and the Hydra composition of every item and fork. With
`--records`, the composition is also compared with the saved configs:

```bash
bash tests/run_checks.sh --python "$FC_PYTHON" --compose --records "$T1_RUNS" --payload
```

`--payload` checks that every file a fork or a trial reads is where the configs say: each pinned pair at
`<T1_RUNS>/<pinned.run>/`, each export at `<T1_EXPORTS>/<item>/`, and the fork-origin pair of each T1fix stage.
`python check_items.py --payload --hash` also compares their sha256 with the pins. ep04's RSI sidecar is reported as
a note, not a failure (see `X_teammate_ok`).

## 5. Fork training (T1fix-style continuation)

A fork resumes from a pinned pair. It keeps the following from the item:
- the experiment, task, `scene.support` and `reset.seed`;
- by default, the task-specific overrides of the stage that trained the pinned checkpoint (`fork.inherit`, e.g.
  ep01's rotation beta 3.0, or ep28alt's W2PD terms).

The arm's overrides from `configs/arms.json` are added on top, plus `--override`. The run uses:
- `training.max_iterations = N + --plus` (default 15,000), with the help-zero early stop off;
- W&B off, with a local `metrics.jsonl`;
- by default, a hull cache seeded with the stock runtime's hulls (`--hull-cache stock`, as T1fix stage 2 did).

**Commands:**

```bash
python fork_train.py list                                    # items, pinned iterations, inherited overrides, arms
python fork_train.py plan  ep25 --arm B --param beta=3.0      # what start would do
python fork_train.py check ep25 --arm B --param beta=3.0      # Hydra compose only (CPU)
python fork_train.py start ep25 --arm B --param beta=3.0 --gpu 0 --plus 20000 --name ep25-B3
python fork_train.py status
```

**Arms:**
- `C`: plain.
- `B`: rotation beta (`--param beta=`).
- `T` / `TB`: ep21-style orientation termination 0.65, without or with beta 2.5.
- `OP` / `OP2`: relaxed orientation, position termination 0.20, seed 42 or 43.
- `W2P` / `W2PD`: ep28 wave 2.
- `G`: bat terminations; built but never launched.

To fork a checkpoint that is not the pinned one, for example a fork's own later checkpoint, use
`--from-run <dir> --from-iteration <N>`.

**Phase-B candidates from the 2026-10-26 plan:**
- ep28: a further fix after the final release, or W2PD continued (`start ep28alt --arm C`).
- ep25 and ep21: rotation weight, and orientation termination (`TB`).
- ep01: continue from a checkpoint that does not collapse.

Each fork trains on one GPU. On the original machine, one job was about 16 GiB of GPU memory, run 3-4 per GPU.

## 6. Watch a fork

```bash
python eval_watch.py ep25-B3 --gpu 0                 # newest checkpoint on the 2,000 grid, as it appears
python eval_watch.py ep25-B3 --all --once            # every grid checkpoint now, then exit
```

**What it does:**
- Each evaluation runs `scripts/track2/eval_all.py`: export at tolerance 1e-4 with this tree's exporter, then the kit
  runner and evaluator on the stock runtime (frame 0, one world, no help). It is checked as in the 20-trial driver.
- When the training ends, the best checkpoint (highest single AUC, earliest on ties) gets 2 more evaluations, and
  `eval/watch_final.json` holds their mean.
- The exported files are kept in `eval/work/it<N>/`, so the chosen one can go straight into trials.

## 7. Repeated official evaluation (trials)

```bash
# the whole set, 20 trials, with the B squeegee alongside (as on 2026-10-26)
python trials_eval.py run --name set20 --extra ep28 --trials 20 --max-parallel 4
# only the changed items, with a new export for ep25 and a second candidate for ep21
python trials_eval.py export --out "$T1_WORK/exports_new" "$T1_WORK/forks/ep25-B3/run/model_118000.safetensors"
python trials_eval.py run --name p2_ep25_ep21 --items ep25,ep21 --trials 20 \
    --file ep25="$T1_WORK/exports_new/run_model_118000/model_118000.safetensors" \
    --add ep21tb:ep21=<an ep21 export>
python trials_eval.py status  --name p2_ep25_ep21
python trials_eval.py summary --name p2_ep25_ep21
```

**What one trial is:**
- Trial k of an item is one official evaluation of the same exported file. There is no selectable evaluator seed, so
  trials differ only by GPU nondeterminism; `seeds.tsv` of the 20-trial run records the seeds.
- A finished evaluation is never re-run. A technical failure is retried up to 3 times.

**Packing and scoring:**
- A complete trial is packed by the kit packer, with a placeholder commit URL and `*_LOCAL_NOT_FOR_UPLOAD.csv`
  names, then scored by the kit's metric_code (`score_csv.py`).
- A subset is packed with a restricted `tracks.json` and scored per episode. Its `summary.md` adds a 30-episode
  estimate that takes the other episodes from the best set's 20-trial means. It is an estimate; confirm with a
  full-set run.
- Write the decision rule down before looking at the results, as `best_set/BANK_ep28_RULE.md` did.
- Cite the trial closest to the mean, never the best trial.

## 8. Pack a submission file

Packing for real needs a pushed commit that contains the training and export code of every item. This branch does,
once pushed. Use the kit packer through `../repro/pack.sh`:

```bash
bash ../repro/pack.sh --results "$T1_WORK/trials/<name>/scores/t<KK>/set/results" \
    --out <file>.csv --code-commit https://github.com/<owner>/<repo>/commit/<pushed sha>
```

`pack.sh` checks the kit against its manifest and runs the kit's preflight. Uploading is a separate decision.

## Caveats

- **Training is not bit-reproducible**, and job packing changes timing. Compare scores, not digests.
- **Exports are not byte-deterministic.** Evaluate the transferred exports to stay comparable with the 20-trial
  numbers. Use `compare_exports.py` to check that a re-export is the same policy.
- **Hull caches.**
  - The evaluator uses the stock cache.
  - The training trees' caches differ from it for 18-23 objects: rolling-start runs and T1fix stage 1 (ep10, ep19,
    ep14-C) trained with their tree's cache (`caches/tree_b39b69b`), and the C3r2 runs with `caches/tree_61fa5d3`.
  - T1fix stage 2 and wave 2 trained with stock hulls (`runs/T1fix/{stage2,ep28w2}/fc_cache`). `fork_train.py`
    seeds stock hulls by default.
  - For an exact replay of a stage-1 run, pass `--hull-cache <payload>/caches/tree_b39b69b`.
- **bc3b1bd defaults.**
  - SONIC ONNX Runtime runs skip the host synchronization; the results are designed to be identical, and its tests
    check the rollout bit for bit. `+action.synchronize_ort_runs=true` restores the old behaviour, but it then
    appears in the stored config; check such an export with `trials_eval.py export` before relying on it.
  - The RSI help schedule stays `deadline` unless the `g1_recon_body_coord_rsi_r2_helpfix` experiment is used.
- **ep04.** The pinned model and export are in the payload; its RSI sidecar is not (see `X_teammate_ok`).
- **Tier 2 / Tier 3** are not covered here, and Tier-1 results must not drive Tier-2/3 decisions.

## Files

| File | Purpose |
|---|---|
| `configs/items/<item>.json` | per item: pinned pair and digests, export, code commits, hull cache, fork template, scores, all training stages |
| `configs/pinned_stages.tsv` | the same stages as one table (`<MOTION>` / `<RUNS>` placeholders) |
| `configs/arms.json` | fork arms |
| `best_set/manifest.tsv` | one row per item: in set, source, iteration, digests, commits, scores |
| `best_set/BANK_ep28_RULE.md` (+ `.sha256`) | the pre-declared ep28 choice (W2PD) |
| `best_set/summary_20trials.md`, `per_trial_20trials.csv`, `per_item_20trials.csv` | the 20-trial records (ep28 = B in the set) |
| `best_set/per_trial_20trials_w2pd_repack.csv` | the same 20 trials re-packed with W2PD as episode 28 (`set_auc_with_ep28` = the set with B instead) |
| `fork_train.py`, `eval_watch.py`, `trials_eval.py`, `score_csv.py` | the tools (sections 5-7) |
| `t1common.py` | shared settings, GPU lanes, commands and result checks |
| `check_items.py`, `compare_exports.py`, `verify_transfer.py`, `tests/run_checks.sh` | checks |
| `site.env.example` | settings template |

The tools are portable versions of the server scripts `t1fix_launch.py`, `t1fix2_launch.py`, `t1fix5_launch.py`,
`runin_eval_watch.py`, `t1best20_driver.py` and `t1best20_score.py`. The originals are in the payload's
`server_scripts/` for reference. They hard-code the original machine: its root folder, GPU 4 exclusion, MPS checks,
NUMA lanes, and slot sharing with the Tier-2/3 drivers.
