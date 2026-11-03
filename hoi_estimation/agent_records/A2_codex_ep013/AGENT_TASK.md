# Task: improve the 6-DoF object trajectory of one monocular human-object-interaction video

You work inside a container. Your workspace is `/work` (read-write). Write code and results under `/work/outputs`,
put any downloaded tools / repositories / model weights / pip packages under `/work/tools` (or the pip user base
`/work/scratch/pyuser`), and scratch files under `/work/scratch`. Do not modify `/work/inputs`.

## Goal
Estimate the object pose for EVERY frame of `inputs/video.mp4` in the released world frame, as accurately as
possible. Accuracy is measured afterwards by someone else with ADD (mean distance between object mesh points under
your pose and under the true pose; no symmetry handling, no alignment), averaged over thresholds 1-9 cm. Rotation
and translation both matter; a rotation that is 180 deg off about a symmetry-looking axis is a full error.

The existing reconstruction (below) comes from a full CARI4D pipeline and is a strong baseline: a result worse than
it is a failure. Change it only where you have clear evidence, and keep the baseline wherever you are unsure.

## Inputs (`/work/inputs`)
- `video.mp4` - the only camera (static exo camera), 1536x1152.
- `task.txt` - what the person does. `episode.json` - object name, frame count.
- `object_mesh.glb` - the exact object mesh (metres), textured. Load it with
  `trimesh.load(path, force="scene", process=False).to_geometry()`; all poses below map THESE mesh-file
  coordinates to the target frame.
- `camera.json` - `K` (pinhole intrinsics, estimated by MoGe-2), `T_world_cam` (4x4 camera-to-world, estimated from
  the frame-0 reference human and the ground plane). OpenCV conventions: camera x right, y down, z forward; the
  world frame is also y-down (gravity along +y).
- `ground_plane.json` - the released floor plane in world coordinates: n.x + d = 0 with `plane` = [n, d].
- `frame0_reference_human.npz` - `vertices_world`: the true human mesh vertices at frame 0 only.
- `reconstruction.npz` - the existing reconstruction:
  - `T_world_mesh` (F,4,4): current best object poses in world (mesh-file frame).
  - `T_cam_mesh_foundationpose`, `T_cam_mesh_coconet` (F,4,4): intermediate camera-frame estimates (FoundationPose
    tracking, and its refinement by the CARI4D network), mesh-file frame. Their depth scale is not calibrated to
    the world; `T_world_mesh` already includes a per-frame depth correction.
  - `human_vertices_world` (F,V,3, every 4th vertex), `human_keypoints_world` (F,70,3; MHR keypoints, right hand
    21-41, left hand 42-62) of the reconstructed human in world; `contact_prob_right_left` (F,2).
- `masks.h5` - `object` and `person` (F,1152,1536) boolean SAM2 masks.

## Tools you may use
- This container's software (Python, torch + CUDA, nvdiffrast, opencv, trimesh, open3d, ...). The CARI4D pipeline
  modules are installed under `/workspace` (v2d_sam2, v2d_moge, v2d_foundation_pose, v2d_sam3d_body, v2d_cari4d);
  their source and run scripts are also in `/code/reconstruction` (read-only).
- Model weights (read-only) in `/weights`: sam2, foundationpose, sam3d_body, cari4d (CoCoNet), grounding_dino,
  hf_home (MoGe-2). HuggingFace is not reachable; set paths to these local weights.
- `/pipeline` (read-only): this episode's full CARI4D intermediate outputs (depth/moge2 with aligned depth maps and
  intrinsics, foundationpose, mhr_init, export, inference/coconet.pth and refined.pth, work/masks). Note: inside
  these CARI4D files, object poses `pose_abs` are for the mesh re-centred to its oriented bounding box
  (`S, _ = trimesh.bounds.oriented_bounds(mesh)`; mesh-file pose = pose_abs @ S).
- The internet (except HuggingFace): you may install packages with pip (`pip install --user ...`), clone code from
  GitHub, and download other public model weights or tools if they help (e.g. other pose estimators, trackers,
  depth or matching models). Keep everything you download under `/work/tools` or the pip user base.

## Rules (strict)
1. Never use or try to obtain ground truth: do not download the challenge dataset (`nvidia/video_to_data_challenge`
   or any copy/mirror of it, its parquet pose files, Tier 1/Tier 2 trajectories) or any annotation of this video.
   Only the files listed above count as data.
2. GPU: one GPU (already selected), total GPU time under 45 minutes.
3. Reproducibility: the final result must be produced by one deterministic script, run OFFLINE as
   `python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
   (it may use `/pipeline`, `/weights`, `/code` and anything you stored under `/work/tools` or the pip user base,
   but no network and no other scratch files; fix random seeds). Decisions taken by looking at images must be
   encoded as automatic criteria or as explicit, commented constants with the reason.
4. Outputs: `/work/outputs/object_poses_world.npz` with key `T_world_mesh` (F,4,4) float64 for all F frames;
   `/work/outputs/REPORT.md` (what you changed and why, evidence, confidence, run time); and
   `/work/outputs/TOOLS.md` listing every external tool you used or tried: pip packages with versions, git repos with
   commit hashes, downloaded weights with URLs and file sha256, and which of them `solve.py` needs.
5. If you cannot improve on the existing reconstruction, output it unchanged and say so.
