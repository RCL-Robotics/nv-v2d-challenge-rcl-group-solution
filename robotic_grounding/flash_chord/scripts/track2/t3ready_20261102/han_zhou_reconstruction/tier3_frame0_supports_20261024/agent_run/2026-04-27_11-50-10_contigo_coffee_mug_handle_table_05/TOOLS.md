# Tools used

No external tools, repositories, pip packages or model weights were downloaded or installed.
`/work/tools` and the pip user base are empty. Everything ran on the software already in the container.

## Container software used (all needed by `solve.py`)
| package | version | used for |
|---|---|---|
| Python | 3.11.10 | |
| torch | 2.5.1+cu124 (CUDA 12.4) | differentiable render-and-compare fitting (Adam) |
| nvdiffrast | 0.4.0 | rasterising the textured mesh (silhouette, texture, logo-label renders) |
| numpy | 1.26.3 | |
| scipy | 1.16.3 | rotations / slerp (`scipy.spatial.transform`) |
| opencv-python (cv2) | 4.11.0 | video decoding, PNG depth decoding, morphology |
| trimesh | 5.1.1 | mesh/texture loading (`trimesh.load(..., force="scene", process=False).to_geometry()`) |
| h5py | 3.16.0 | masks, MoGe-2 aligned depth |

## Data used by `solve.py` (read-only)
- `/work/inputs/*`: video, masks, mesh, camera, ground plane, `reconstruction.npz` (baseline poses,
  FoundationPose/CoCoNet camera-frame poses, human keypoints, contact probabilities).
- `/pipeline/depth/moge2/aligned.h5`: the episode's MoGe-2 metric-aligned depth maps, already computed by the
  pipeline. Used only for the table-surface depth cue.

No network access is needed. No weights from `/weights` were used (no network was re-run).

## Tried and not used
Nothing external was tried. Silhouette-only, local-normalised-texture, variance-weighted NCC and
edge-chamfer yaw scores were written and tested in-house. They were not decisive (see REPORT.md), and only
the logo-position measurement and the hand cues were adopted.
