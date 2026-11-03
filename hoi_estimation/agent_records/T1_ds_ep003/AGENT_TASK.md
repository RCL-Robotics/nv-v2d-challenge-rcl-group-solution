# Task: recover the full 6-DoF object trajectory of one monocular human-object-interaction video

You work inside a container. Your workspace is `/work` (read-write). Write code and results under `/work/outputs`,
put any downloaded tools / repositories / model weights / pip packages under `/work/tools` (or the pip user base
`/work/scratch/pyuser`), and scratch files under `/work/scratch`. Do not modify `/work/inputs`.

## Goal
Estimate the object pose for EVERY frame of `inputs/video.mp4` in the released world frame, as accurately as
possible. Accuracy is measured afterwards by someone else with ADD (mean distance between object mesh points under
your pose and under the true pose; no symmetry handling, no alignment), averaged over thresholds 1-9 cm. Rotation
and translation both matter; a rotation that is 180 deg off about a symmetry-looking axis is a full error, and so is
a correct rotation on only half of the frames.

You are not asked to tune an existing pipeline. Think of this as a research problem: you may design your own method,
re-run or replace any component, and use any public tool. The existing reconstruction (below) is one source of
evidence, a reasonable but frame-by-frame estimate; it is the baseline you must beat, and wherever you have no
better evidence you should fall back to it.

## What a frame-by-frame estimate gets wrong (and a trajectory method can fix)
The baseline was produced by a tracker plus a per-frame network. Typical failure modes of such estimates are:
- the whole episode, or a long segment, turned by ~180 deg about a near-symmetry axis of the object (the
  silhouette is almost identical; texture, small asymmetric parts and the hand grasp are not);
- sudden jumps: within a few frames the rotation changes by tens of degrees while the object in the video does not,
  typically during occlusion by the hand/body, motion blur, or when the object is small or leaves the image;
- slow drift during long occlusions, and frames where tracking was lost;
- a near-constant error of the distance from the camera (depth along the viewing ray) over the episode.

A rigid object in a video obeys strong temporal constraints that a per-frame estimate ignores. Use them. Examples
of what we expect a good solution to do (choose and design your own):
1. **Segment the episode into phases** from the video, masks and the human: object at rest (not touched, or resting
   on a surface) - its world pose is CONSTANT over the whole phase; carried / manipulated - it moves rigidly with the
   hand and smoothly in time; transitions. Static phases are very valuable: one pose explains many frames, so evidence
   from all of them can be pooled (e.g. texture over hundreds of frames resolves a symmetry that one frame cannot).
2. **Per-frame evidence that can score many hypotheses**: render the textured mesh (nvdiffrast is installed) and
   compare with the image: silhouette agreement with the object mask (excluding pixels covered by the person),
   edges / texture / colour agreement, consistency with the MoGe-2 depth map; optical flow or feature tracks between
   neighbouring frames (a correct trajectory must explain the image motion of the object's texture).
3. **Several hypotheses per frame, not one**: the baseline, the FoundationPose and CoCoNet estimates, their 180 deg
   symmetry flips about each principal axis of the mesh, poses propagated from neighbouring frames, and fresh
   registrations (e.g. render-and-compare optimisation from several initialisations, or FoundationPose
   re-registration on chosen frames).
4. **Global decoding over time**: choose one hypothesis per frame (or per segment) jointly, e.g. dynamic programming /
   Viterbi with a data cost from (2) and a transition cost that penalises rotation and translation changes (much more
   in static phases), then refine continuously, e.g. a joint optimisation of all poses with image terms plus
   smoothness / constant-velocity terms on SE(3), and rigid hand-object attachment while carried.
5. **Anchors and propagation**: find frames where the pose is unambiguous (object large, little occluded, distinctive
   view, high evidence margin between hypotheses) and propagate forward and backward from them through the ambiguous
   segments, rather than trusting every frame equally.

## Validate without ground truth
You have no ground truth and must not try to get it. Build your own checks and make decisions with them:
measures of how well a trajectory explains the video (held-out frames, mask IoU, photometric / flow residuals,
temporal smoothness, physical plausibility such as no floating or penetrating the floor/human), with a table baseline
vs. your result per phase. Adopt a change only where your checks clearly favour it, and say how confident you are.
Iterate: diagnose the baseline, form a hypothesis, test it on a few frames, measure, and repeat. Work in small,
fast experiments first; you have a fast model, so many short iterations are better than one long attempt.

## Depth
The baseline's distance from the camera was calibrated against the whole human body and the floor; depth from the
object alone (silhouette scale on an object a few dozen pixels wide, with an estimated K) is far less reliable. You
may correct depth, but only when several independent cues agree (hand contact with the reconstructed human, static
object on the released floor plane, MoGe-2 depth, metric mesh size vs. image size), and quantify their disagreement.
Do not snap the object onto the floor plane: it is a fit with several-cm tolerance (see `foot_plane_dist_stats` in
`ground_plane.json`), and the object may rest on other surfaces.

## Inputs (`/work/inputs`)
- `video.mp4` - the only camera (static exo camera), 1536x1152.
- `task.txt` - what the person does. `episode.json` - object name, frame count.
- `object_mesh.glb` - the exact object mesh (metres), textured. Load it with
  `trimesh.load(path, force="scene", process=False).to_geometry()`; all poses below map THESE mesh-file
  coordinates to the target frame.
- `camera.json` - `K` (pinhole intrinsics, shared by all episodes of this camera), `T_world_cam` (4x4 camera-to-world,
  estimated from the frame-0 reference human and the floor). `K_pipeline`: the per-episode MoGe-2 K the pipeline's
  camera-frame estimates were made with. OpenCV conventions: camera x right, y down, z forward; the world frame is also
  y-down (gravity along +y).
- `ground_plane.json` - the released floor plane in world coordinates: n.x + d = 0 with `plane` = [n, d].
- `frame0_reference_human.npz` - `vertices_world`: the true human mesh vertices at frame 0 only.
- `reconstruction.npz` - the existing reconstruction:
  - `T_world_mesh` (F,4,4): the baseline object poses in world (mesh-file frame), consistent with `camera.json`.
  - `T_cam_mesh_foundationpose`, `T_cam_mesh_coconet` (F,4,4): intermediate camera-frame estimates (FoundationPose
    tracking, and its refinement by the CARI4D network), mesh-file frame, for `K_pipeline`. Their depth scale is not
    calibrated to the world; `T_world_mesh` already includes a per-frame depth correction.
  - `human_vertices_world` (F,V,3, every 4th vertex), `human_keypoints_world` (F,70,3; MHR keypoints, right hand
    21-41, left hand 42-62) of the reconstructed human in world; `contact_prob_right_left` (F,2).
- `masks.h5` - `object` and `person` (F,1152,1536) boolean SAM2 masks (may be imperfect; you may recompute them).

## Tools you may use
- This container's software (Python, torch + CUDA, nvdiffrast, opencv, trimesh, open3d, ...). The CARI4D pipeline
  modules are installed under `/workspace` (v2d_sam2, v2d_moge, v2d_foundation_pose, v2d_sam3d_body, v2d_cari4d);
  their source and run scripts are also in `/code/reconstruction` (read-only).
- Model weights (read-only) in `/weights`: sam2, foundationpose, sam3d_body, cari4d (CoCoNet), grounding_dino,
  hf_home (MoGe-2). HuggingFace is not reachable; set paths to these local weights.
- `/pipeline` (read-only): this episode's full CARI4D intermediate outputs (depth/moge2 with aligned depth maps and
  intrinsics, foundationpose, mhr_init, export, inference/coconet.pth and refined.pth, work/masks). Note: inside
  these CARI4D files, object poses `pose_abs` are for the mesh re-centred to its oriented bounding box
  (`S, _ = trimesh.bounds.oriented_bounds(mesh)`; mesh-file pose = pose_abs @ S) and use `K_pipeline`.
- The internet (except HuggingFace): you may install packages with pip (`pip install --user ...`), clone code from
  GitHub, and download other public model weights or tools if they help (e.g. point trackers such as CoTracker /
  TAPIR, optical flow such as RAFT, feature matchers, other pose estimators or depth models). Keep everything you
  download under `/work/tools` or the pip user base.

## Rules (strict)
1. Never use or try to obtain ground truth: do not download the challenge dataset (`nvidia/video_to_data_challenge`
   or any copy/mirror of it, its parquet pose files, Tier 1/Tier 2 trajectories) or any annotation of this video.
   Only the files listed above count as data.
2. GPU: one GPU (already selected), shared with two other jobs: keep your GPU memory below 14 GB (process long
   videos in chunks) and total GPU time under 45 minutes.
3. Reproducibility: the final result must be produced by one deterministic script, run OFFLINE as
   `python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
   (it may use `/pipeline`, `/weights`, `/code` and anything you stored under `/work/tools` or the pip user base,
   but no network and no other scratch files; fix random seeds). Decisions taken by looking at images must be
   encoded as automatic criteria or as explicit, commented constants with the reason (e.g. phase boundaries you
   verified by eye). `solve.py` must recompute its estimates from the inputs; do not embed per-frame results or fitted
   correction tables computed elsewhere. Run `solve.py` once at the end and check its output.
4. Outputs: `/work/outputs/object_poses_world.npz` with key `T_world_mesh` (F,4,4) float64 for all F frames;
   `/work/outputs/REPORT.md` (phases you found, what you changed and why, your validation table baseline vs. result,
   confidence per phase, run time); and `/work/outputs/TOOLS.md` listing every external tool you used or tried: pip
   packages with versions, git repos with commit hashes, downloaded weights with URLs and file sha256, and which of
   them `solve.py` needs.
5. If you cannot improve on the existing reconstruction, output it unchanged and say so.
