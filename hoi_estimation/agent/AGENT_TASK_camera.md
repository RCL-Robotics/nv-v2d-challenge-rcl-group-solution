# Task: estimate the camera (and with it the world trajectory of the object) of one monocular HOI video

You work inside a container. Your workspace is `/work` (read-write). Write code and results under `/work/outputs`,
put any downloaded tools / repositories / model weights / pip packages under `/work/tools` (or the pip user base
`/work/scratch/pyuser`), and scratch files under `/work/scratch`. Do not modify `/work/inputs`.

## Goal
Estimate the object pose for EVERY frame of `inputs/video.mp4` in the released WORLD frame, as accurately as
possible. Accuracy is measured afterwards by someone else with ADD (mean distance between object mesh points under
your pose and under the true pose, in the world frame; no symmetry handling, no alignment), averaged over thresholds
1-9 cm.

This task is about the CAMERA. The baseline object trajectory was estimated in the camera frame and fits the image
well, but it is mapped to the world with an estimated camera `T_world_cam` (and intrinsics `K`):
`T_world_mesh = T_world_cam @ T_cam_mesh`. Any error of `T_world_cam` moves the whole trajectory in the world, and
it is invisible in the image (the object still projects onto the right pixels). In this kind of data, errors of a few
degrees of camera rotation and 5-30 cm of camera position are typical for the baseline camera, and they are the
largest single source of world-frame error, often larger than the per-frame object errors. A related error is a
near-constant offset of the object's distance from the camera (depth along the ray), which also does not show in the
image. Your job is to estimate the camera as well as the allowed evidence permits, and then re-express the object
trajectory with it (keeping its camera-frame trajectory, i.e. its image fit, unless you have strong evidence).
You may also improve the object trajectory itself, but that is secondary here.

## The camera is fixed for the whole video; what pins it down
The camera is static: one `T_world_cam` (6 DoF) and one `K` for all frames. The world frame is defined by the data
below; the camera has to be placed in it. Independent, allowed evidence includes, for example:
1. **The frame-0 reference human** (`frame0_reference_human.npz`): the exact human mesh in WORLD coordinates at
   frame 0. Its image at frame 0 is known (video frame 0, the person mask, and our reconstructed human). Matching the
   true 3D body to its 2D evidence (silhouette, 2D keypoints you detect, our human's projected keypoints, dense
   correspondences of our reconstructed human, which has the same topology and pose up to errors) constrains all 6 DoF;
   the body is ~1.7 m tall and its depth extent is small, so distance and rotation about the viewing axis are the weak
   directions. The baseline camera was fitted to this human through a few keypoints; you can do much better with the
   full silhouette / dense correspondence.
2. **The released floor plane** (`ground_plane.json`, world coordinates). The floor is visible in the image; the
   MoGe-2 depth maps in `/pipeline/depth/moge2` give metric 3D floor points in the CAMERA frame (up to MoGe's scale /
   shift errors; check them). A plane fitted to camera-frame floor points must coincide with the released world plane
   after the camera transform: this fixes the camera's tilt (2 DoF) and height above the floor (1 DoF) independently of
   the human. Segment the floor yourself (e.g. the largest planar region below the horizon not covered by person or
   object masks, robust plane fit).
3. **Feet on the floor over the episode** (our reconstructed human per frame; contacts with the released plane),
   objects that rest on the floor while untouched (their lowest points must lie on the plane), the exact metric size
   of the mesh vs. its image size.
4. **K**: the given `K` is shared over all episodes of this camera (focal length from MoGe-2, principal point from a
   fit). Changing K changes how depth maps to the image; only change it with clear evidence.
Combine these in one robust estimate (e.g. a joint least-squares over the 6 camera DoF, plus optionally one depth
scale of the object trajectory), with each term weighted by its reliability, and check that the cues agree.
Quantify the disagreement between cues: it is the best uncertainty estimate you have.

## Validate without ground truth
You have no ground truth and must not try to get it. Report, for the baseline camera and yours: reprojection of the
frame-0 reference human onto the frame-0 person mask (IoU, boundary distance), floor-plane residuals of the
camera-frame floor points, feet-to-floor distances over the episode, height of resting objects above the floor, and
the camera change (rotation deg, centre cm) suggested by each cue alone. Adopt a camera change only when several
independent cues agree on it.

## Inputs (`/work/inputs`)
- `video.mp4` - the only camera (static exo camera), 1536x1152.
- `task.txt` - what the person does. `episode.json` - object name, frame count.
- `object_mesh.glb` - the exact object mesh (metres), textured. Load it with
  `trimesh.load(path, force="scene", process=False).to_geometry()`; all poses below map THESE mesh-file
  coordinates to the target frame.
- `camera.json` - `K` (pinhole intrinsics, shared by all episodes of this camera), `T_world_cam` (4x4 camera-to-world,
  the baseline camera, fitted to a few keypoints of the frame-0 reference human and to feet-on-floor). `K_pipeline`:
  the per-episode MoGe-2 K the pipeline's camera-frame estimates (and the MoGe depth maps) were made with. OpenCV
  conventions: camera x right, y down, z forward; the world frame is also y-down (gravity along +y).
- `ground_plane.json` - the released floor plane in world coordinates: n.x + d = 0 with `plane` = [n, d]
  (`foot_plane_dist_stats`: how far the true feet were from it over the dataset, i.e. its tolerance).
- `frame0_reference_human.npz` - `vertices_world`: the true human mesh vertices at frame 0 only (world frame).
- `reconstruction.npz` - the existing reconstruction:
  - `T_world_mesh` (F,4,4): the baseline object poses in world (mesh-file frame), = `T_world_cam @ T_cam_mesh`.
  - `T_cam_mesh_foundationpose`, `T_cam_mesh_coconet` (F,4,4): intermediate camera-frame estimates, mesh-file frame,
    for `K_pipeline`, depth scale not calibrated.
  - `human_vertices_world` (F,V,3, every 4th vertex of the MHR mesh), `human_keypoints_world` (F,70,3; MHR keypoints)
    of the reconstructed human in world (mapped with the baseline camera); `contact_prob_right_left` (F,2).
- `masks.h5` - `object` and `person` (F,1152,1536) boolean SAM2 masks.

## Tools you may use
- This container's software (Python, torch + CUDA, nvdiffrast, opencv, trimesh, open3d, ...). The CARI4D pipeline
  modules are installed under `/workspace` (v2d_sam2, v2d_moge, v2d_foundation_pose, v2d_sam3d_body, v2d_cari4d);
  their source and run scripts are also in `/code/reconstruction` (read-only). The MHR / SOMA body models are there
  too (the reference human and our reconstructed human share the MHR topology; see v2d_sam3d_body).
- Model weights (read-only) in `/weights`: sam2, foundationpose, sam3d_body, cari4d (CoCoNet), grounding_dino,
  hf_home (MoGe-2). HuggingFace is not reachable; set paths to these local weights.
- `/pipeline` (read-only): this episode's CARI4D intermediate outputs (depth/moge2 with raw and aligned depth maps and
  intrinsics, foundationpose, mhr_init, export, inference/coconet.pth and refined.pth, work/masks). Inside these
  files object poses `pose_abs` are for the mesh re-centred to its oriented bounding box
  (`S, _ = trimesh.bounds.oriented_bounds(mesh)`; mesh-file pose = pose_abs @ S) and use `K_pipeline`.
- The internet (except HuggingFace): you may pip-install packages, clone code from GitHub, and download other public
  model weights or tools (e.g. 2D human keypoint detectors, segmentation or depth models). Keep everything you
  download under `/work/tools` or the pip user base.

## Rules (strict)
1. Never use or try to obtain ground truth: do not download the challenge dataset (`nvidia/video_to_data_challenge`
   or any copy/mirror of it, its parquet pose files, Tier 1/Tier 2 trajectories, camera calibrations of the dataset)
   or any annotation of this video. Only the files listed above count as data. The frame-0 reference human may be used
   only for frame 0, as given.
2. GPU: one GPU (already selected), shared with up to five other jobs: keep your GPU memory below 8 GB and total GPU
   time under 45 minutes.
3. Reproducibility: the final result must be produced by one deterministic script, run OFFLINE as
   `python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
   (it may use `/pipeline`, `/weights`, `/code` and anything you stored under `/work/tools` or the pip user base,
   but no network and no other scratch files; fix random seeds). It must also write
   `/work/outputs/camera_estimate.json` with your `K` and `T_world_cam`. Decisions taken by looking at images must be
   encoded as automatic criteria or as explicit, commented constants with the reason. `solve.py` must recompute its
   estimates from the inputs; do not embed per-frame results or fitted camera parameters computed elsewhere.
   Run `solve.py` once at the end and check its output. You have about 80 minutes in total; leave time for that.
4. Outputs: `/work/outputs/object_poses_world.npz` with key `T_world_mesh` (F,4,4) float64 for all F frames;
   `/work/outputs/camera_estimate.json`; `/work/outputs/REPORT.md` (cues used, per-cue camera estimates and their
   agreement, validation table baseline vs. yours, confidence, run time); and `/work/outputs/TOOLS.md` listing every
   external tool you used or tried (pip packages with versions, git repos with commit hashes, downloaded weights with
   URLs and file sha256, and which of them `solve.py` needs).
5. If you cannot improve on the baseline camera, output the baseline unchanged and say so.
