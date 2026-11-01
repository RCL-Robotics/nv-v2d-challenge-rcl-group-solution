# Tools used

No external tools were downloaded. No git repositories were cloned, no pip packages were installed, and no
model weights were fetched. Everything below was already in the container.

## Python packages (pre-installed in the container; Python 3.11.10)
| package | version | used for | needed by `solve.py` |
|---|---|---|---|
| torch | 2.5.1+cu124 | GPU tensors, autograd for pose optimisation | yes |
| nvdiffrast | 0.4.0 | differentiable textured rasterisation of the object mesh (silhouette, texture, normals, depth) | yes |
| numpy | 1.26.3 | everything | yes |
| scipy | 1.16.3 | sparse least squares (depth smoothing), median filter, rotations (diagnostics) | yes |
| opencv-python (cv2) | 4.11.0 | video decoding, distance transforms, PNG depth decoding, debug images | yes |
| trimesh | 5.1.1 | loading `object_mesh.glb` (mesh, UVs, texture) | yes |
| h5py | 3.16.0 | `masks.h5`, `/pipeline/depth/moge2/aligned.h5` | yes |
| Pillow | 10.2.0 | texture image (through trimesh) | yes (indirectly) |

## Data used by `solve.py` (read-only, no network)
- `/work/inputs/*`: video, masks, mesh, camera, ground plane, `reconstruction.npz` (baseline poses, MHR hand
  keypoints).
- `/pipeline/depth/moge2/aligned.h5`: the episode's MoGe-2 aligned depth maps. Used only to build a static
  background-depth map (median over every 10th frame), which tells the renderer where the table occludes the book.

## Tried and not used
- Nothing else was installed or tried. FoundationPose / CoCoNet / SAM2 modules under `/workspace` and `/weights`
  were not re-run. Their existing outputs (`T_cam_mesh_foundationpose`, `T_cam_mesh_coconet`) were only compared
  with the baseline during diagnosis; the baseline equals CoCoNet up to a constant 1.3° offset and a depth
  correction.

## Code written for this task (in `/work/outputs`)
- `solve.py`: the deterministic solver (entry point).
- `hoi_lib.py`: renderer, observation crops, losses, static and trajectory optimisers, shift search, Viterbi,
  smoothing.
- `validate.py`: ground-truth-free validation table (baseline vs result per phase, optional held-out evaluation).
- `depth_cues.py`: independent depth cues (floor contact, MoGe-2 depth, hand keypoints), for reporting only.
