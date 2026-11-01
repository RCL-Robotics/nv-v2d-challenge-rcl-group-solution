# Tier-3 data preparation: from Han Zhou's frame-0 release to the training inputs

This folder covers everything between Han's release `tier3_frame0_supports_20261024` and the 30 locked training
parquets, support scenes and frame references that the Tier-3 policies were trained and evaluated on. It is part B of
the submission record (`../README.md`).

**Rules followed throughout.**
- CPU only, except the retarget, which needs one plain CUDA context per episode.
- No clean or ground-truth trajectory is read anywhere. The only exception is the true first-frame object pose that
  is already inside Han's track (grey area 1 in `../README.md`).
- Han's release folder is only read, never written.

## Paths

The scripts embed the training server's paths: `ME=/root/autodl-tmp/haoran.zhang` and Han's release under
`/root/autodl-tmp/han.zhou/tier3/release/`.
- `SOURCES.tsv` (one level up) gives each file's original path.
- `python3 ../tools/restore_layout_t3.py --me <root> [--rewrite]` puts the files back at those paths. With `--rewrite`,
  it also changes the root inside the text files.
- Environments:
  - `conda_envs/v2d`: Isaac Sim 5.1.0, Isaac Lab 2.3.2, robotic_grounding and py-soma-x. Used for the retarget,
    `t3f0_frames.py` and `t3f0_align.py`.
  - `conda_envs/flashchord`: FlashCHORD (this repo's `robotic_grounding/flash_chord`, identical to fc_r2 @ 61fa5d3). Used
    for the build, supports, physical checks, driver and helper.
  - `conda_envs/v2dkit`: the unchanged submission kit. Used for the parquet gates.
- `fc_env.sh` sets the FlashCHORD environment.

**The retarget code is the one in this commit.** It is `robotic_grounding/scripts/retarget/soma_to_g1.py` (git blob
`8207b16312237b1510858e379705ba7ae2069e61`), identical to nv `f2923d61` in `code/nv-v2d-challenge-codebase`, which ran
it (`../verification/code_identity.txt`).

## Rebuild order, commands and expected hashes

All sha256 values below are the ones recorded on the training server. Short forms are prefixes of the full values in
`f0/build/build_files.sha256`, `f0/build_files_f0.sha256`, `f0/align/ALIGN_FILES.sha256` and `../SHA256SUMS`.

### 0. The opus_v2 root (`opus_root/`)

`t3opus_build_data.py` byte-copies Han's `tier3_opus_20261021`. It writes:
- `whole_body/soma/track2_manifest.json` (sha256 `fac5a985…`), built from `episodes.json`;
- `frame_references_opus/`;
- the released geometry-free stage files of the 16 floor-only episodes.

Its READY is `9c4cb2a9…`. Then `t3opus_driver(_v2).sh`, retarget step only, runs per episode:
- `t3opus_retarget.sh`, which calls `soma_to_g1.py <folder> --save --fps 30 --motion-root <root>`, v2d env;
- `t3opus_rcheck(_v2).py`, which writes `opus_easy/retarget/parquet_lock.json` (`e1b42382…`).

The f0 build reads this root in two places:
- step A below: the manifest and the 16 geometry-free stage files;
- the GT-free comparisons: rcheck G9, where the robot columns must equal the opus parquet, and `t3f0_align.py`, where
  the mitigation rule compares f0 with opus_v2.

The opus_v2 policies themselves are not part of this submission.

### A. Inputs and frame references

`flashchord python -I -B scripts/t3f0_build_root.py inputs` (`a6cb943a…`)
- Byte copies of the 30 `reconstructed_data/<seq>/` folders, 7 files each, into
  `data/human_motion_data_tier3_f0/whole_body/soma/sequence_id=<seq>/`, which is `soma_to_g1.py`'s data-folder layout.
  - Checks: camera, ground plane, `soma_params.npz` and mesh are byte-identical to the opus root.
  - The input sha256 are in `data_root/SOURCE_FILES.sha256` (`ce9d9aa1…`).
- `frame_references_f0/episode_NNNNNN.npz`: Han's f0 `poses.npy` in `released_track2_world`, all frames. This is the
  packer's `--frame-references`.
- `whole_body/soma/track2_manifest.json` (`d496584c…`).
- The 16 geometry-free stage files.
- `han_release_docs/`.
- Report: `f0/build/build_report.json` (`8aebef66…`) and `inputs_check.json`.

### B. The retarget frame chain

`v2d python scripts/t3f0_frames.py [--verts 1,2,3,6,7,9,11,13,16,17,20,25,27,28] 0 … 29` (`0109d3d3…`)

It reproduces `soma_to_g1.py`'s static map from OpenCV world to G1 world, without IK. The output goes to
`f0/frames/epNN.{json,npz}`; only the json is committed.

### C. Retarget and gates

`bash scripts/t3f0_retarget_all.sh` (`87bb30ca…`) takes one episode at a time:
- `t3f0_retarget.sh` (`b4bf2e78…`): `soma_to_g1.py <folder> --save --fps 30 --motion-root <root>`, v2d env, clean
  environment, plain CUDA context.
- `t3f0_rcheck.py --ep N` (`b4104672…`, v2dkit): gates G1–G10.
  - G9: the robot side equals the opus parquet.
  - G10: the object equals the step-B frame chain.
- Output: `f0/retarget/epNN_rcheck.json`, `run.log` and the lock **`f0/retarget/parquet_lock.json` (`bdb83b0e…`, 30
  entries)**. Training and evaluation use only locked parquets.

GPU choice is `t3opus_helper_v2.py rplan`, which is operational only.

### D. Support scenes for the 14 support episodes (01 02 03 06 07 09 11 13 16 17 20 25 27 28)

The order on 2026-10-24, from the logs and file times:
1. `flashchord python scripts/t3f0_supports.py all` (`11caa79d…`): first passes at 18:31–18:47 CST
   (`f0/supports_build*.log`, `supports_stool.log`).
2. `t3f0_lean_search.py 1` (`ec52fd66…`) at 19:12–19:15. This is the ep01 settle-verified sideways shift, written to
   `f0/supports/ep01_lean_search.json`.
3. A final `t3f0_supports.py` pass at 19:16, which applied the shift and wrote `SCENE_SUPPORT.json`.

The script writes Cube primitives into `whole_body/reconstructed_stage/<seq>_support.usda`. Inputs:
- Han's `support.json` (`T_world_mesh_contact`, `top_box`) and `support_mesh.glb`;
- our four placement rules (PREREG_f0 "Supports"), recorded in `f0/supports/epNN.json`.

Outputs: `data_root/SCENE_SUPPORT.json` (`bce8e9eb…`) and `data_root/SUPPORT_FILES.sha256` (`5deea2b0…`).

**Physical checks:** `bash f0/physcheck_chain.sh` (`585d1984…`), which runs `t3f0_physcheck.py` (`dec7798f…`)
`rest | human | g1 | settle`, then `summary`. Output: `f0/physcheck/`. `f0/repin_watch.sh` (`37c9dc26…`) only pins CPU
affinity.

### E. Finalize the data root

`flashchord python -I -B scripts/t3f0_build_root.py finalize` writes the write-once root:
- **`FILES.sha256` (`f9c94e85…`, 576 files)**;
- **`READY` (`ee58a62d…`)**.

Then `t3f0_prep_table.py` (`fc2f4799…`) writes `f0/build/prep_table.{json,tsv}` (`61bf2db0…`).

### F. Hand-object gap and the palm mitigation list

1. `v2d python scripts/t3f0_align.py 0 … 29` (`03ea75fc…`) writes `f0/align/epNN.json`.
2. `--summary` writes `summary.{json,tsv}`.
3. The rule writes **`f0/align/mitigation_f0.json` (`2ab767eb…`)**: ep01 02 03 15 17 19 20 28 get
   `termination.palm_position_threshold=0.25` for training.

Pins: `f0/align/ALIGN_FILES.sha256` (`1ccbd296…`) and `MITIGATION_f0_FROZEN.sha256`. Both were frozen 2026-10-24
22:51 CST, before any f0 training.

### G. Training queue (f0)

`bash scripts/t3f0_driver.sh dry`, then `run`:
- driver `c861fd3f…`; helper `t3f0_helper.py` (`b9077315…`), which imports `track2_queue_tier2v2_helper.py`
  (`b75a33c3…`);
- governed by `../evaluation/f0/PREREG_f0.md` (`82702d85…`).

It writes:
- `f0/cfgcheck_f0.json` (`3da056a4…`): Hydra comparison of all 30 against the recipe;
- `f0/prep_f0.{json,tsv}`;
- one training per episode in `fc_runs/C3r2_tier3_f0/epNN`. The exact launch lines are in
  `../set/set_items.json` (`recorded_commands`).

After training, its post step exports `final.json`'s checkpoint at tolerance 1e-4 and runs one official own-input
evaluation (`../evaluation/f0/eval/epNN/post.json`).

The helper also names a LOCAL clean-diagnostic scorer (`scripts/track2_tier2_clean_scores.py`). It is report-only,
never decides anything, and is not part of this record.

## Files here

- `scripts/`: the f0 prep and queue scripts above, `fc_env.sh`, and `t3opus_helper_v2.py` (used only for GPU choice).
- `f0/`: the records and pins of steps A–G under their server names (`tmp_analysis/tier3/f0/`).
  - Not committed: large arrays, `align/dump/`, `physcheck_old/`, the driver and heartbeat logs, and LOCAL
    clean-diagnostic files.
- `data_root/`: the data root's READY, FILES.sha256, SOURCE_FILES.sha256, SCENE_SUPPORT.json, SUPPORT_FILES.sha256 and
  manifest.
  - The data itself is not committed: Han's arrays, our parquets, the support USDA and the frame references. FILES.sha256
    pins every file of it.
- `opus_root/`: step 0's scripts and records.
