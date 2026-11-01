# Han Zhou's reconstruction: what is here, and what is still missing

The Tier-3 policies were trained on Han Zhou's video reconstruction. Han is on our team. This folder holds every piece
of his reconstruction code and documentation that exists on the training server. The pieces listed under **MISSING**
exist only on Han's machine. They are needed to rebuild the inputs from the video (rule 2.9.a) and **must be added by a
later commit**.

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
    the inputs.

**Not copied:**
- binary data: the reconstructed arrays, meshes, `support_mesh.glb`, images. Our data root holds byte copies of the
  inputs we used; their sha256 are in `../data_prep/data_root/SOURCE_FILES.sha256`.
- `tier3_agentT1_20261020`: a single-episode pilot that no submitted episode uses.
- `tier3_ep13_opusmax_20261030`: an ep13 re-run that Han selected with ground truth. It is not used (PREREG_v3 D3).

**GT numbers.** Han's README files contain ground-truth-based quality numbers, for evaluation only. They are copied
verbatim. Nothing in our pipeline reads them, and our documents do not quote them.

## MISSING (only Han can supply; needed for rule 2.9.a; to be added by a later commit)

1. **The s1 human and object reconstruction pipeline**, the parts not in `tier3_s1_20261020/code/`:
   - The CARI4D run on each video: GroundingDINO → SAM2 → MoGe-2 → SAM-3D-Body → FoundationPose → CoCoNet, the P1 / M1
     → JC1 camera steps, and the `v2d_cari4d.sif` container.
   - The MHR → SOMA-X conversion with the official SOMA-X `PoseInversion`.
   - **The world alignment.** One rigid transform from the reconstruction to the official world, fitted to the
     **frame-0 reference human**. The camera principal point is also estimated from it. This is grey area 2 in the
     top-level README.
2. **The opus_v2 agent harness.**
   - The prompt and instructions given to each Claude Code (Opus) agent, the sandbox, and the script that ran the 30
     agents.
   - The code that packed each agent's `/work/inputs`: the video, SAM2 `masks.h5`, the object mesh, the camera, the
     ground plane, and `reconstruction.npz` with the human keypoints and contact probabilities.
   - The code that turned each agent's `object_poses_world.npz` into the release.
   - **Per-episode helper modules imported by `solve.py` that are not in the release.** Without them, `solve.py` cannot
     run for these 19 episodes:

     | ep | missing module(s) |
     |---|---|
     | 00, 02, 27 | `hoilib` |
     | 01 | `fit`, `lib`, `traj` (also needs `sam2`) |
     | 03 | `lib` |
     | 04 | `bat_lib`, `fitter`, `stages`, `validate` |
     | 05, 20 | `hoi_lib`, `traj_opt` |
     | 06 | `hoi_fit` |
     | 07 | `common`, `constrained`, `fitcore`, `hyp`, `prep`, `stages`, `v2d` |
     | 11, 16, 18, 25 | `hoi_lib` |
     | 14 | `common`, `rectify` |
     | 15 | `common`, `evidence`, `trajectory` |
     | 19 | `common`, `decode`, `fit`, `photo` |
     | 21 | `common`, `fit`, `stages` |
     | 22 | `render_utils`, `silfit`, `traj_utils` |

     The other 11 episodes (08 09 10 12 13 17 23 24 26 28 29) import only third-party packages: `roma` for 08 and 29;
     `sam2`, `hydra` and `omegaconf` for 13. They still need the agent inputs above.
3. **The frame-0 correction code** (`opus_v2_frame0_f3_c2_shift`). It applies the symmetric flip, the camera rotation,
   the orientation anchor and the constant translation, using the true first-frame object pose (grey area 1). It also
   produced `poses.npy` / `object_pose_xyz_wxyz.npy` of the frame-0 release.
4. **The support fit code**, Han's `~/Projects/v2d/support/`. Its stages are background, anchor, detect, segment, fit,
   contact and package:
   - GroundingDINO detection on a median background image;
   - SAM2 segmentation;
   - a grid search plus an nvdiffrast silhouette-IoU fit of the scanned furniture mesh standing on the ground plane;
   - **the contact shift** behind `T_world_mesh_contact`: a vertical shift of −5 to +2 cm so that his object track rests
     on the support.

   The task text names which piece of furniture is the support (grey area 3).

The rest of the pipeline is in this commit:
- what we did from the frame-0 release onwards: data root, retarget, supports for the trainer, mitigation, training,
  evaluation, selection and pack (`../data_prep/`, `../set/`, `../evaluation/`);
- the trainer and the retarget (`robotic_grounding/flash_chord`, `robotic_grounding/scripts/retarget/soma_to_g1.py`).
