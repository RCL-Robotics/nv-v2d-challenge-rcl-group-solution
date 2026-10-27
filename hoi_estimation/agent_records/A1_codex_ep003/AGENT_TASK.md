# Task: improve the 6-DoF object trajectory of one monocular human-object-interaction video

You are working inside an isolated container. Everything you may use is under `/work/inputs` (do not modify it).
Write all your code and results under `/work/outputs` and scratch files under `/work/scratch`.

## Goal
Estimate the object pose for EVERY frame of `inputs/video.mp4` in the released world frame, as accurately as
possible. Accuracy is measured (by someone else, after you finish) with the mean distance between the object mesh
points under your pose and under the true pose (ADD, no symmetry handling, no alignment), averaged over thresholds
1-9 cm. Both rotation and translation errors matter; a rotation that is 180 deg off about a symmetry-looking axis
is a full error.

## Inputs (`/work/inputs`)
- `video.mp4` - the only camera (static exo camera), 1536x1152.
- `task.txt` - what the person does in the video. `episode.json` - object name, frame count.
- `object_mesh.glb` - the exact object mesh (metres). Poses map mesh-file coordinates to world.
- `camera.json` - `K` (pinhole intrinsics, estimated by MoGe-2 from the video), `T_world_cam` (4x4 camera-to-world,
  estimated from the frame-0 reference human and the ground plane). OpenCV conventions: camera x right, y down,
  z forward; the world frame is also y-down.
- `ground_plane.json` - the released floor plane in world coordinates: n.x + d = 0 with `plane` = [n, d].
- `frame0_reference_human.npz` - `vertices_world`: the true human mesh vertices at frame 0 only (released for
  defining the frame; no later frames are available).
- `reconstruction.npz` - an existing automatic reconstruction (CARI4D pipeline + post-processing), which may contain
  errors of any kind (wrong rotation, e.g. flipped about a symmetry-looking axis or wrong handle side; depth/scale
  errors along the camera ray; jitter; tracking failures):
  - `T_world_mesh` (F,4,4): current best object poses in world.
  - `T_cam_mesh_foundationpose`, `T_cam_mesh_coconet` (F,4,4): two intermediate camera-frame estimates
    (FoundationPose tracking, and its refinement by a human-object network). Their depth scale is not calibrated
    to the world; `T_world_mesh` already includes a per-frame depth correction.
  - `human_vertices_world` (F,V,3, every 4th vertex) and `human_keypoints_world` (F,70,3; MHR keypoints, right hand
    21-41, left hand 42-62) of the reconstructed human in world; `contact_prob_right_left` (F,2).
- `masks.h5` - `object` and `person` (F,1152,1536) boolean SAM2 segmentation masks.

## Rules (strict)
1. Do NOT use any ground truth or try to obtain it. Shell commands have no network; do not try to get around this
   (no downloads, no `pip install`). Use only the files in `/work/inputs` and the software already installed in the
   container (Python with numpy, scipy, torch + CUDA, opencv, trimesh, nvdiffrast, ...; code under /workspace may be
   read and used, but there are no model weights).
2. GPU: at most one GPU (already selected), total GPU time under 45 minutes.
3. Reproducibility: your final result must be produced by a single deterministic script
   `/work/outputs/solve.py`, run as
   `python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
   (fixed random seeds, no reliance on scratch files). Any decision you took by looking at images must be encoded in
   the script as an automatic criterion or, if truly manual, as an explicit, commented constant with the reason.
4. Output: `/work/outputs/object_poses_world.npz` with key `T_world_mesh` (F,4,4) float64 for all F frames, plus
   `/work/outputs/REPORT.md` (what you changed and why, what evidence you used, how confident you are, run time).
5. If you cannot improve on the existing reconstruction, output it unchanged and say so.

Suggested approach (optional): render the mesh under the current poses over video frames and compare with the
object mask / image (silhouette IoU, edges); test rotation hypotheses (e.g. symmetry flips, handle side) against the
image evidence over many frames; check physical plausibility (object resting on the table/floor, contact with the
hand when the task says it is held, no interpenetration); smooth over time where the object is static.
