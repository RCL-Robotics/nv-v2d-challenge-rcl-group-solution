# Track 2 Tier 1: submission record of set `all7`, trial 05 (2026-10-31)

This folder is the code-and-records side of one Tier-1 submission file. The file is the official 30-episode evaluation
of the set `all7`, trial 05, packed by the kit packer with this commit's URL. It is not uploaded by anything here.

- **Set `all7`**: the 2026-10-26 best set (`../best_set/`) with 7 episodes replaced by T1improve checkpoints
  (`../t1improve/`): ep01, ep05, ep10, ep14, ep15, ep19, and ep28 by `TOPUP_AMEND1` (`amend1/`).
- **Trial 05**: the trial closest to the mean of 20 complete official trials of the set (TOPUP_RULES section 6;
  never a best-of choice). The file is the same 22,989 rows as the trial's earlier pack, which cited 0c7516e
  (sha256 `53cfe9bfa5a2487f3502035e535ea5fabcbff0cab0a578fbea514108aa0f3f77`); only `code_commit_url` changes.
  A commit cannot contain its own URL, so the new file's sha256 is recorded outside Git, next to the file.

| | AUC | SP-SR | MP-SR | MPPE cm |
|---|---:|---:|---:|---:|
| trial 05, kit metric_code against the clean Tier-1 reference | 0.930970 | 1.000 | 1.000 | 0.930 |
| set `all7`, mean of 20 trials (sd) | 0.93113 (0.0033) | 0.998 | 0.990 | 0.972 (0.282) |
| report only: best trial (08) | 0.93280 | | | |

**Post-hoc amendment, disclosed.** `amend1/TOPUP_AMEND1.md` (sha256 `76143fce...`) was written after the 20-trial
results were seen, at the user's request. It waives one extra condition (ep28 MP pass >= 16 of 20 trials) that kept
the incumbent W2PD model_66000; W3H 83000 passed MP in 15 of 20. Over the same 20 trials W3H has mean AUC 0.8158 vs
0.8074, MPPE 2.52 vs 3.47 cm, and MP pass 15 vs 6. The set's MP-SR of 1.0 is a property of trial 05, not of every
evaluation: W3H misses MP on ep28 in 5 of 20 trials.

## The 30 checkpoints

One checkpoint per episode. Full records: `set_items.json` (every training stage with its overrides, fork pairs,
hull cache, export, trial-05 inputs), `set_stages.tsv` (54 stages), `set_manifest.tsv` (one row per episode).

| ep | object | item | trained (last stage) at | stages | evaluated file | 20-trial AUC | trial-05 AUC |
|---:|---|---|---|---:|---|---:|---:|
| 0 | tall_bar_stool | ep00 (C3r2_runin 28000) | b39b69b | 2 | `2e3079703b50` | 0.9976 | 0.9977 |
| 1 | vacuum | **ep01-Bc 162000** (T1improve) | b39b69b | 5 | `01c0a3bd6568` | 0.7967 | 0.8002 |
| 2 | blue_easter_basket | ep02 (C3r2_runin 44000) | b39b69b | 1 | `be9833ef63d3` | 0.9776 | 0.9774 |
| 3 | contigo_coffee_mug | ep03 (C3r2_runin 36000) | b39b69b | 1 | `9224384118fe` | 0.9619 | 0.9804 |
| 4 | baseball_bat | ep04 (teammate runin_cont4 358000) | bc3b1bd | 5 | `52042a12944c` | 0.7507 | 0.7504 |
| 5 | big_black_box | **ep05-B3 62000** (T1improve) | b39b69b | 2 | `ed7bca161687` | 0.9630 | 0.9633 |
| 6 | monitor | ep06 (C3r2_runin 114000) | b39b69b | 1 | `842cb67303c0` | 0.9679 | 0.9681 |
| 7 | iron | ep07 (C3r2_wide 17950) | 61fa5d3 | 1 | `a54f0c6ea1c0` | 1.0000 | 1.0000 |
| 8 | skinny_wood_chair | ep08 (C3r2_runin 28000) | b39b69b | 1 | `2f98f4695236` | 0.9673 | 0.9676 |
| 9 | big_red_bowl | ep09 (C3r2_wide 32185) | 61fa5d3 | 1 | `1a4086d09e6d` | 0.9290 | 0.9290 |
| 10 | white_desk | **ep10-BT 58000** (T1improve) | b39b69b | 3 | `725ee5e8f8dd` | 0.8951 | 0.8947 |
| 11 | light_blue_book | ep11 (C3r2_runin 36000) | b39b69b | 1 | `9f60a9cb5943` | 0.9768 | 0.9776 |
| 12 | black_platform | ep12 (C3r2_runin 28000) | b39b69b | 1 | `3b2c52d79496` | 0.9856 | 0.9863 |
| 13 | snack_box | ep13 (C3r2_runin 48000) | b39b69b | 1 | `5cec84bb80d6` | 0.9786 | 0.9583 |
| 14 | round_table | **ep14-R 92000** (T1improve) | b39b69b | 4 | `b01866a8f910` | 0.8343 | 0.8294 |
| 15 | wet_floor_sign | **ep15-B2 40000** (T1improve) | 61fa5d3 | 2 | `8d0e8023a9a9` | 0.9859 | 0.9865 |
| 16 | spray_bottle | ep16 (C3r2_wide 24000) | 61fa5d3 | 1 | `baefdf33e932` | 0.9896 | 0.9932 |
| 17 | traffic_cone | ep17 (C3r2_runin 44000) | b39b69b | 1 | `8cbe742c58cb` | 0.9424 | 0.9313 |
| 18 | blue_trash_can | ep18 (C3r2_runin 32000) | b39b69b | 1 | `ff7e980fc055` | 0.9933 | 0.9940 |
| 19 | cane | **ep19-Bc 135000** (T1improve) | b39b69b | 4 | `07863de3bb9b` | 0.9228 | 0.9183 |
| 20 | rice_cooker | ep20 (C3r2_runin 70000) | b39b69b | 2 | `409796a2703a` | 0.9855 | 0.9852 |
| 21 | tall_bar_stool | ep21 (C3r2_runin 119500) | b39b69b | 1 | `ffc06d795a21` | 0.7548 | 0.7548 |
| 22 | step_ladder | ep22 (C3r2_runin 36000) | b39b69b | 1 | `b86140bb17a4` | 0.9761 | 0.9768 |
| 23 | toy_stroller | ep23 (C3r2_runin 28000) | b39b69b | 1 | `674841934bc5` | 0.9908 | 0.9907 |
| 24 | watering_can | ep24 (C3r2_runin 118000) | b39b69b | 1 | `4c99a3de7b9f` | 0.9994 | 0.9995 |
| 25 | cyan_water_bottle | ep25 (C3r2_runin 102000) | b39b69b | 1 | `fa932e0d4e58` | 0.7481 | 0.7403 |
| 26 | skateboard | ep26 (C3r2_runin 28000) | b39b69b | 1 | `af8d40d74abc` | 0.9975 | 0.9970 |
| 27 | woven_basket | ep27 (C3r2_runin 41500) | b39b69b | 1 | `656d87669772` | 0.9965 | 0.9972 |
| 28 | squeegee | **ep28-W3H 83000** (T1improve, AMEND1) | 7fbc227 | 4 | `497ae3b3a785` | 0.8158 | 0.8211 |
| 29 | rolling_luggage | ep29 (T1fix C 30000) | b39b69b | 2 | `bcb1c5412d2a` | 0.8536 | 0.8628 |

## The code each checkpoint needs, and where it is in this branch

Every training stage of every checkpoint ran on one of these code trees. All of them are in this branch:

| tree | what ran on it | in this branch |
|---|---|---|
| b39b69be6c2f (RSI rolling start) | the rolling-start runs, T1fix stages, 5 T1improve arms (ep01-Bc, ep05-B3, ep10-BT, ep14-R, ep19-Bc) | ancestor (via 10a3af0) |
| 0868a3f4a29a (teammate commit) | ep01, ep04, ep19 stages 1-3 (`runin*`) | not a commit of this history; its tree equals b39b69b's |
| 61fa5d31ff80 (`fc_r2` repo) | the C3r2_wide runs (stage 1 of ep07, 09, 10, 14, 15, 16, 28, 29) and the ep15-B2 arm | its `flash_chord` files equal f2923d61817e, an ancestor (via 41403d3) |
| 7fbc2273a92b (squeegee options) | ep28 W2PD stage and the ep28-W3H arm | merged in f9202c2 |
| bc3b1bdf34f2 (Zuxing Lu, `dev.zuxing.speedup`) | ep04 `runin_cont3` and `runin_cont4` | merged in 773c645 |

`../t1improve/topup/commit_check.json` records that the working trees which trained the T1improve arms (b39b69b,
7fbc227, fc_r2) equal those commits file for file. The arms' hyperparameters are Hydra overrides only
(`../t1improve/arms/<arm>/overrides.yaml`, `launch.json`).

- **Configs.** `check_set.py` composes every stage with this tree's Hydra configs; see [Verification](#verification).
- **Export.** Each evaluated file's exporter is in `set_manifest.tsv` (`exported_by`), always the training tree,
  tolerance 1e-4. This tree's exporter is 7fbc227's. It re-exports all 30 training checkpoints to the same policy as
  the evaluated files (tensors and metadata equal, `compare_exports.py`); see Verification.
- **Retargeting and data.** `../repro/README.md` (data -> conversion -> retargeting -> training -> export ->
  evaluation): `robotic_grounding/scripts/track2/{get_data.sh,track2_to_soma_inputs.py,retarget_all.sh}` and
  `robotic_grounding/scripts/retarget/soma_to_g1.py`. All 30 checkpoints trained on the same derived motion data.
  Retargeting determinism is unverified, so reuse the derived motion data (payload group `G_motion_data`).
- **Evaluation.** The stock runtime nvidia-isaac/video_to_data 8c778f9 and the kit evaluator
  (`flash_chord/scripts/evaluate_policy.py`, sha256 `160d24f5...`), world count 1. The harness forces the reset seed
  42; every (trial, episode) evaluation has the seed signature
  `delay=0(+g0..4,0-2) reset=42 scale_seed=0(1.0/1.0) wrench=0 vertex=0/0 ppo=42`. The spread between trials is GPU
  non-determinism.

## Verification

`check_set.py` (CPU only), run with `--hash` on this tree on the original machine just before it was committed (the
tree differed from the commit only by `verification/` and `SHA256SUMS`). Report: `verification/check_set_report.json`.

| check | result |
|---|---|
| stages that compose with this tree (all 30 lineages, 54 stages) | 54 / 54 |
| A: committed overrides -> compose == the run's saved `config.yaml` (stages with their own record) | 52 / 52 |
| B: the run's saved `overrides.yaml` -> compose == saved `config.yaml`, all keys exact | 52 / 52 |
| D: stage that trained each checkpoint -> resolved compose == `resolved_config_json` stored **in the checkpoint** | 30 / 30 |
| E: each evaluated file's `export_source_sha256` == its training checkpoint; both sha256 == pins | 30 / 30 |
| keys this tree composes that a saved config lacks | none |
| this tree's exporter re-exports each training checkpoint (CPU, tolerance 1e-4) to the same policy as its evaluated file (`compare_exports.py`: tensors and metadata equal) | 30 / 30 |

Notes:
- ep00 and ep20 were restarted in place (stage 2 in the stage-1 folder). That folder's record is stage 2's, so stage 1
  is checked for composition only.
- C (committed overrides == saved overrides.yaml) differs only for the C3r2_wide stage-1 runs: the committed lines use
  `logging.mode=<WANDB_MODE>` and drop the optional W&B entity/project; the runs logged online. No training key differs.

## Reproduce

Environments, payload (checkpoints, exports, run records, hull caches, motion data, kit) and site settings:
`../README_NEW_MACHINE.md` sections 1-4. The T1improve training records (forks, runs) are under `<RUNS>/T1improve/`.

1. **Training**, one stage (`set_stages.tsv` row, or `set_items.json` stage):
   ```bash
   cd robotic_grounding/flash_chord
   PYTHONPATH=$PWD/src HUMAN_MOTION_DATA_DIR=<MOTION> FLASH_CHORD_CACHE_DIR=<hull cache of the stage> \
       python scripts/train_rl.py <the stage's overrides>
   ```
   A T1improve arm resumes from its fork pair (`training.resume=true training.checkpoint=<fork>/model_<N>.safetensors`,
   with the matching `rsi_<N>.safetensors` beside it); the exact argv is `../t1improve/arms/<arm>/launch.json`.
   `../fork_train.py start <item> --from-run <run> --from-iteration <N> --override <...>` builds the same command.
   Training is not bit-reproducible; compare scores, not digests.
2. **Export** (CPU):
   ```bash
   CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONPATH=$PWD/src python -B -m flash_chord.training.ppo.coordinate.export \
       --input <run>/model_<N>.safetensors --output <out>/model_<N>.safetensors \
       --stock-runtime <stock 8c778f9>/robotic_grounding/flash_chord --report <out>/export_report.json --tolerance 0.0001
   python ../compare_exports.py <out>/model_<N>.safetensors <evaluated file>
   ```
3. **One official evaluation** of an exported file (what each trial ran, per episode):
   `python scripts/track2/eval_all.py --export no --world-count 1 --episodes <e> ...` around the kit runner
   `examples/run_policy_evaluation.py --eval-script <kit>/flash_chord/scripts/evaluate_policy.py --runtime-root <stock>
   --world-count 1`. `../trials_eval.py run --trials 20` repeats it for a set.
4. **Pack, validate, score** trial 05 with a commit URL (no evaluation is re-run):
   ```bash
   <kit python> -B repack.py --kit <kit> --dataset-root <data> --results <trial 05 results> \
       --source-csv <trial 05 file citing 0c7516e> --source-scores amend1/files_trial05/metric_code.json \
       --expected amend1/files_trial05/expected_checkpoints.json --scorer <scorer> --validator <validator> \
       --commit-url https://github.com/<owner>/<repo>/commit/<sha> --out-dir <dir> --name <file>.csv
   ```
   It runs the kit packer, the kit uploader's offline `validate_submission()` with provenance checks, the kit
   metric_code scorer, and a row-by-row comparison with the source file. Packing the trial with the source file's own
   URL reproduces it byte for byte (sha256 `53cfe9bf...`).

## Files

| path | content |
|---|---|
| `set_items.json`, `set_stages.tsv`, `set_manifest.tsv` | the 30 checkpoints, their 54 training stages, exports and trial-05 inputs |
| `check_set.py` | the composition and provenance check above |
| `repack.py` | pack + offline validation + metric_code score + row comparison for a commit URL |
| `amend1/` | `TOPUP_AMEND1.md` (+ `.sha256`), `amend1_build.py`, its selection, files record and log; `files_trial05/` (pack log, metric_code scores, validation, expected checkpoints of the file citing 0c7516e); `verify/` (two independent checks of that file) |
| `trial05/` | the trial's score record (`score.json`: per-episode sources, results digests, seeds), metric_code scores, pack log |
| `verification/` | `check_set_report.json`, `reexport_compare.tsv` and `reexport_all.sh` (the re-export check), `build_t1ready.py` (wrote the set tables from the run records) |
| `SHA256SUMS` | sha256 of every other file of this folder |
| `server_scripts/` | the T1improve wave-1 launcher that trained the 7 arms (reference only; it hard-codes the original machine) |
| `../t1improve/` | PREREG v2 + ADD1/ADD2, arm configs, confirmation and 20-trial top-up records and tools |
| `../best_set/`, `../configs/` | the 2026-10-26 set: item configs, pinned stages, 20-trial records |
