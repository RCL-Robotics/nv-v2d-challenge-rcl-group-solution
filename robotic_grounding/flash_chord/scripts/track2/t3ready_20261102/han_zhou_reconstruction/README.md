# Han Zhou's reconstruction: what is here, where the rest is, and what is still missing

The Tier-3 policies were trained on Han Zhou's video reconstruction. Han is on our team.

**Where his code is.** Since 2026-11-02, Han's own branch `dev.hanzhou.hoi_estimation` is part of this record, at the
repo root under [`hoi_estimation/`](../../../../../../hoi_estimation/README.md).
- The code as it ran is `48c217a`, merged unchanged in `ab0d24c`.
- His later commits `6c10a22` … `c0dbf5b` follow it. `c0dbf5b` is the parent of the record update that follows it.
  They add the reproducibility material and make the script paths portable (`../README.md`, history item 4).

It holds his full workflow:
- `hoi_extraction/`: the base pipeline (s1);
- `agent/`: the agent harness, with the records of all 112 agent runs in `agent_records/`;
- `tier2_refine/`: the frame-0 refinement (Tier 3) and the Tier-2 fusion (not used by us);
- `support/`: the support reconstruction;
- `diagnostics/`: GT-only diagnostics;
- `reproducibility/`: the exact commands (`s1_runbook.md`, `tier3_runbook.md`), the end-to-end driver
  `run_all_tier3.sh`, the environment pins and the weight hashes.

**This folder** holds the release documents and the part of his code that he shipped with s1. They were copied from the
training server before the merge, for `232e6b2`. The section **Formerly MISSING** below maps each piece that
`232e6b2` listed as missing to its files in `hoi_estimation/`, and lists what is still missing.

All files here are read-only byte copies, made on 2026-11-02 from
`/root/autodl-tmp/han.zhou/tier3/release/` (`<HAN>/tier3/release/` in `SOURCES.tsv`). For each file,
`COPIES_t3ready.tsv` records the sha256 of the source before copying, of the copy, and of the source after copying.
All 148 rows agree. Han's folder was never written.

## The three releases, in order

| release | what changed | used here |
|---|---|---|
| `tier3_s1_20261020` | First full reconstruction from the monocular exo video: human and object, camera, world alignment. | The human (`soma_params.npz`), camera, ground plane and mesh of every episode come from s1. No later release changed them. |
| `tier3_opus_20261021` | The object track of every episode was re-estimated by a Claude Code (Opus) agent in a sandbox, starting from s1. Human unchanged. | The object track is the base of the frame-0 release. |
| `tier3_frame0_supports_20261024` | opus_v2 plus a frame-0 object-pose correction (variant `opus_v2_frame0_f3_c2_shift`). Also video-fit supports for 14 episodes. | **This is the release our data root was built from.** |

Its README describes the correction steps as: symmetric flip, camera rotation error, object-orientation anchoring, and a
constant translation. The translation is the difference between the frame-0 object pose and the mean of the static
segment. The README states that only the first-frame object pose is used, and no other ground truth.

## What is in this folder

- `tier3_frame0_supports_20261024/`
  - `README.md`, `episodes.json`;
  - `reconstructed_data/<seq>/episode.json` (30 files: per-episode metadata, object and camera variant);
  - `supports/README_supports.md`, `supports/index.json`, `supports/contact_report.json`;
  - `supports/<seq>/support.json` for the 14 support episodes: `T_world_mesh` from the video fit,
    `T_world_mesh_contact`, `top_box`, the IoU of the fit.
  - `agent_run/<seq>/{solve.py, TOOLS.md, REPORT.md}` for all 30 episodes: the opus_v2 agent's per-episode solver, its
    tool list and its report. These 90 files are byte-identical to the opus release's `agent_run/` (checked: 90/90).
- `tier3_opus_20261021/`: `README.md`, `episodes.json`.
- `tier3_s1_20261020/`: `README.md`, `episodes.json`, and `code/`, the copy of part of Han's
  `~/Projects/v2d/hoi_extraction/scripts/` that he shipped with s1:
  - `postproc_joint_camera.py`, `postproc_joint_camera_K.py`: joint camera calibration across the 30 episodes;
  - `postproc_depth_blend.py`: FoundationPose / CoCoNet depth blend;
  - `build_release.py`: writes the release folders;
  - `eval_official.py`: Han's evaluation against the reference, used only for his quality tables. It produced none of
    the inputs. (The other four import only its `bundle_path()` path helper.)
  - All five are the same blobs as `hoi_estimation/hoi_extraction/scripts/<name>` in Han's branch.

**Not copied:**
- binary data: the reconstructed arrays, meshes, `support_mesh.glb`, images. Our data root holds byte copies of the
  inputs we used; their sha256 are in `../data_prep/data_root/SOURCE_FILES.sha256`.
- `tier3_agentT1_20261020`: a single-episode pilot that no submitted episode uses.
- `tier3_ep13_opusmax_20261030`: an ep13 re-run that Han selected with ground truth. It is not used (PREREG_v3 D3).

**GT numbers.** Han's README files contain ground-truth-based quality numbers, for evaluation only. They are copied
verbatim. Nothing in our pipeline reads them, and our documents do not quote them.

## Formerly MISSING: where each piece is now (paths under `hoi_estimation/`)

`232e6b2` listed four pieces as MISSING. All four now have code in `hoi_estimation/`.

**Corrections to the `232e6b2` list (from its verifier).** Part of item 1 was never missing:
- the JC1 / JC2 joint-camera steps (`postproc_joint_camera.py`, `postproc_joint_camera_K.py`), which include the
  principal-point fit;
- the depth blend (`postproc_depth_blend.py`);
- the MHR → SOMA-X conversion with the official `PoseInversion` (`build_release.py`).

All of these were already in `tier3_s1_20261020/code/` in `232e6b2`. What item 1 really lacked is listed below.

### 1. s1: the human and object reconstruction from the exo video (`hoi_extraction/`)

| piece | files | what they do |
|---|---|---|
| CARI4D per-video run | `cluster/run_episode.sh`, `cluster/gen_prompt_one.py`, `cluster/episodes.tsv` | per episode: GroundingDINO prompt boxes (`gen_prompt_one.py`) → SAM2 video masks (`v2d.sam2.lib.video_to_masks`) → mask packing (`v2d.cari4d.lib.pack_masks`) → `v2d.cari4d.lib.run_inference`, which runs MoGe-2 depth, SAM-3D-Body, FoundationPose and CoCoNet (and retries with windowed refinement) |
| batch drivers | `scripts/run_local.sh` (apptainer, local GPUs), `slurm/run_all.sh`, `slurm/smoke_ep0.sh`; the earlier docker path `scripts/00_setup.sh`, `01_gen_prompts.py`, `02_run_episode.sh`, `03_run_all.sh` | run the per-video pipeline over the 30 episodes |
| stage re-runs | `scripts/exp_fp_mode.sh` (FoundationPose + CoCoNet with other flags, e.g. `E1_every_frame "--register-every-frame"`), `scripts/exp_postopt.sh` (contact-guided refinement) | experiment variants written to `experiments/<name>/` |
| container and environment | `singularity/v2d_cari4d.def`, `v2d_gdino.def`, `mini.def`, `10_nvidia_egl.json`; `slurm/build_images*.sh`, `build_gdino_gpu.sh`, `build_test.sh` | build the two images. `v2d_cari4d.def` sets the base image and installs pybind11, eigen, kaolin, pytorch3d, MoGe and TensorRT at fixed versions, plus nvdiffrast (unpinned) and the pip packages |
| `/workspace/v2d_cari4d/lib/cari4d` library | this repo's `reconstruction/modules/v2d_cari4d/lib/cari4d` (in the tree since `main`), copied into the image by `v2d_cari4d.def`; plus `patches/run_foundationpose_mhr_export.py`, bind-mounted over `prep/run_foundationpose_mhr_export.py` | the CARI4D code imported by the s1 scripts (`lib_mhr`, `tools/render_mhr_wild_inference`). The patch returns NaN instead of inf for the centroid distance of a fully hidden object |
| M1 step (`M1_prompt_v2`) | `cluster/gen_prompt_v2.py` | size-consistent GroundingDINO prompt boxes for a CARI4D re-run; `export_inputs.py`, `run_batch_*.sh` and the post-processing use that run where the episode's alignment names `M1` |
| P1 step (`P1_temporal`) | `scripts/postproc_temporal.py` | temporal stabilisation of the CARI4D object poses (rotation and translation outlier rejection, smoothing; no reference data) |
| other post-processing | `scripts/postproc_flip_select.py`, `postproc_yaw_flip.py`, `postproc_floor_depth.py`, `postproc_feet_floor.py`, `postproc_size_depth.py`, `postproc_const_scale.py`, `postproc_contact_row.py`, `postproc_hand.py`, `camera_floor.py`, `estimate_focal.py`, `ground_scale.py`, `vda_depth.py` | symmetry flips, floor and depth constraints, focal and floor estimates |
| first-frame alignment → `logs/pnp_forE1.json` | `scripts/firstframe_align.py` (trimmed ICP of our frame-0 human to the frame-0 reference human), `scripts/firstframe_pnp.py` (PnP + depth scale from the ICP correspondences; writes the `--out` JSON with `sim` / `rigid`) | the initial world alignment. `postproc_joint_camera*.py` read it (`--align`, default `logs/pnp_forE1.json`). `logs/pnp_forE1.json` is a copy of `pnp_merged.json` (= `pnp_baseline.json` updated by `pnp_M1.json`), saved under the name the joint-camera scripts use by default. The name refers to the E1 experiment, which is the source of no released episode (FF2 sources: M1 6, P1_temporal 15, baseline 9) |
| release | `scripts/build_release.py` (already in `232e6b2`) | writes the s1 release; its usage line names the variant `DB_JC2s1_shared_f820_pp` / camera `JC2s1_shared_f820_pp` |

### 2. opus_v2: the agent harness (`agent/`) and the run records (`agent_records/`)

- **Prompt:** `agent/AGENT_TASK_temporal_v2.md`.
  - It is the same blob as the `AGENT_TASK.md` of all 30 release runs.
  - Each run manifest's `task_sha256` (`5d5091be…`) matches it.
  - The earlier prompt versions (`AGENT_TASK.md`, `_camera`, `_depth`, `_temporal`) are also in `agent/`.
- **Sandbox and launcher:** `agent/run_claude.sh`.
  - It runs apptainer `--containall --network none`. Only the workspace, CARI4D code, weights and the episode's
    pipeline outputs are bound.
  - Its only route out is the logging CONNECT proxy `agent/allowlist_proxy.py`.
  - The release runs used `net=open`: every host except huggingface.co / hf.co.
  - The runs used `claude-opus-5-5`, effort high, 90 min.
  - The batch script is `agent/run_batch_CL.sh` (`PFX` / `TASK` from the environment). `run_codex.sh` and
    `run_batch_T1.sh` were used for the earlier Codex runs.
- **Input packing:** `agent/export_inputs.py`. It writes `video.mp4`, `object_mesh.glb`, `ground_plane.json`,
  `task.txt`, `episode.json`, `camera.json`, `reconstruction.npz`, `frame0_reference_human.npz` and `masks.h5`.
  - It ran with `--variant DB_JC2s1_shared_f820_pp --align logs/JC2s1_shared_f820_pp_align.json --camera
    JC2s1_shared_f820_pp` (`reproducibility/tier3_runbook.md`); the code defaults DB1 / JC1 were not used.
  - `reconstruction.npz` holds that variant's object poses, the FoundationPose and CoCoNet poses, the human vertices
    and keypoints, and the contact probabilities.
  - `camera.json`, recovered from each release run's event stream, matches the run manifest's sha256 (30/30). Its
    `T_world_cam` equals the s1 (JC2s1) camera exactly (`../verification/han_followup/from_6c10a22/`).
  - `frame0_reference_human.npz` holds frame-0 vertices only. All 30 release manifests list it (grey area 2 in the
    top-level README).
  - The CL2* runs reused the `T1_ds_ep*` inputs.
- **Output → release:** `agent/build_release_opus.py` (s1 release + each run's `object_poses_world.npz` + its `REPORT.md`,
  `solve.py`, `TOOLS.md`).
- **Which run made each episode.** The run is identified by blob identity. All 90 `agent_run/<seq>/{solve.py,
  REPORT.md, TOOLS.md}` files of this folder equal `agent_records/<run>/outputs/` of:
  - `CL2_ep000` … `CL2_ep010` (episodes 0–10);
  - `CL2b_ep011` … `CL2b_ep021` (11–21);
  - `CL2c_ep022` … `CL2c_ep029` (22–29).

  This is the mapping in `build_release_opus.py`. See `../verification/merge_t3final/agent_run_map.json`.
- **Helper modules.** For all 30 episodes, every local module that `solve.py` imports, directly or through another
  helper, is in the same `outputs/` folder:

  | ep | run | local modules (in `agent_records/<run>/outputs/`) |
  |---|---|---|
  | 00 | CL2_ep000 | `hoilib` |
  | 01 | CL2_ep001 | `fit`, `lib`, `traj` |
  | 02 | CL2_ep002 | `hoilib` |
  | 03 | CL2_ep003 | `lib` |
  | 04 | CL2_ep004 | `bat_lib`, `fitter`, `stages`, `validate` |
  | 05 | CL2_ep005 | `hoi_lib`, `traj_opt` |
  | 06 | CL2_ep006 | `hoi_fit` |
  | 07 | CL2_ep007 | `common`, `constrained`, `fitcore`, `hyp`, `prep`, `stages` (and `v2d.sam2.lib.sam2_utils`, which is this repo's `reconstruction/modules/v2d_sam2/lib/sam2_utils.py`, installed in the CARI4D image) |
  | 11, 16, 18, 25 | CL2b_ep011 / 016 / 018, CL2c_ep025 | `hoi_lib` |
  | 14 | CL2b_ep014 | `common`, `rectify` |
  | 15 | CL2b_ep015 | `common`, `evidence`, `trajectory` |
  | 19 | CL2b_ep019 | `common`, `decode`, `fit`, `photo` |
  | 20 | CL2b_ep020 | `hoi_lib`, `traj_opt` |
  | 21 | CL2b_ep021 | `common`, `fit`, `stages` |
  | 22 | CL2c_ep022 | `render_utils`, `silfit`, `traj_utils` |
  | 27 | CL2c_ep027 | `hoilib` |

  - The other 11 episodes import no local module.
  - Third-party packages are listed with versions in each `TOOLS.md`.
- **Records of each run:** the prompt, `transcript.md`, `events.jsonl.gz` (raw event stream), `run_manifest.txt`
  (container, binds, the sha256 of each input file) and `outputs/`. Index: `agent_records/INDEX.md`.
- **Offline re-run:** `diagnostics/repro.sh` re-runs a delivered `solve.py` with no network and compares the result
  with the delivered track.

### 3. The frame-0 correction (`tier2_refine/t3_frame0.py`)

- **Arguments:** `t3_frame0.py opus <out> 3 2 shift`, i.e. `opus_f3_c2_shift`. The steps are:
  - flip = 3: track the symmetric flip, then add the object-frame anchor only above 15°;
  - camrot = 2: camera rotation fix, rotations only. Over the static start, the chordal mean D of R0 R(t)^T is taken as
    the camera rotation error and applied to the whole track about the camera centre; skipped if D exceeds 15°;
  - trans = shift: add the constant translation p0 − static-start mean.
- **Inputs:** each episode's opus_v2 track and the frame-0 object pose, taken from `ref_poses/<ep>.npz`, frame 0 (grey
  area 1).
- **After writing each trajectory**, the same run scores it against the reference poses as a diagnostic
  (`diag_scores.json`).
- **Publishing:** `support/build_release_combined.py` writes the frame-0 release. It replaces `poses.npy` /
  `object_pose_xyz_wxyz.npy` by this output and labels it `opus_v2_frame0_f3_c2_shift`.
- **GT-blind re-run.** Cut the diagnostic tail of `t3_frame0.py` (from `res = {}` on) and give it a `ref_poses` that
  holds frame 0 only. Our re-run from `c0dbf5b` reproduced the release byte for byte, 30/30
  (`../verification/han_followup/`). Since `c0dbf5b`, `build_release_combined.py` runs without `diag_scores.json`.

### 4. The support fit and the contact shift (`support/`)

Run in this order:
- `common.py`: the support furniture per episode, from the task text, and the prompts;
- `background.py`: temporal median background, person / object occupancy;
- `anchor.py`: the resting frame of the frame-0 track;
- `detect.py`: GroundingDINO on the background, choosing the box nearest the anchor;
- `segment.py`: SAM2 box-prompted mask;
- `fit.py`: the mesh standing on the ground plane; coarse grid over up-axis, yaw and position, then nvdiffrast
  silhouette soft-IoU refinement;
- `contact.py`: per rest frame, the gap between the object's lowest point and the support below it;
- `package.py`: `support.json` with `T_world_mesh`, `T_world_mesh_contact` (the mesh shifted along up by the median
  rest gap), `top_box`;
- `build_release_combined.py`: the frame-0 release.

Also there: `lean_check.py`, `overlay.py`, `sheet.py`, and `score_snap.py` (a GT diagnostic).

- **Which `fit.py`.** The released supports of eps 2 3 6 7 9 11 13 16 17 20 25 27 28 were fitted with the `fit.py` of
  20:28Z. Han reconstructed it as `support/fit_20261023T2028Z.py`. ep1 was fitted with the current `fit.py`. The
  commands are in `reproducibility/tier3_runbook.md` section C.
- `contact.py` ran with `--snap`; the snapped track is not used.
- `fit_overview.jpg` and the release's `contact_report.json` are copies of `sheet.py`'s and `contact.py --snap`'s
  outputs. The release READMEs are hand-written (`reproducibility/release_files/`).

### Formerly still missing

The four items that were still missing after the merge are resolved by Han's later commits. What remains open is listed
in `../README.md` section 6.

**GT material in Han's branch.** It includes GT-based evaluation and diagnostics:
- `diagnostics/`;
- `hoi_extraction/scripts/eval_official.py`, `eval_object_pose.py`, `eval_tier2.py`;
- `agent/score_runs.py`, `eval_agent.py`, `gt_camera_fit.py`;
- the diagnostic halves of `t3_frame0.py` and `support/contact.py --gt`;
- tables in his READMEs;
- GT diagnostic numbers in `reproducibility/` and in his reproduction report;
- `hoi_extraction/scripts/export_reference.py`, which exports all frames of `ref_poses` for the diagnostics. Only frame 0
  enters the frame-0 step.

They are merged as they are. Our pipeline reads none of them, and our documents quote none of their numbers.

The rest of the pipeline is in this commit:
- what we did from the frame-0 release onwards: data root, retarget, supports for the trainer, mitigation, training,
  evaluation, selection and pack (`../data_prep/`, `../set/`, `../evaluation/`);
- the trainer and the retarget (`robotic_grounding/flash_chord`, `robotic_grounding/scripts/retarget/soma_to_g1.py`).
