# Tools used

No external tools, repositories, pip packages or model weights were downloaded. Everything ran on the software already
in the container plus the read-only pipeline outputs. `/work/tools` is empty.

## Python packages (pre-installed in the container; Python 3.11.10)
| package | version | used for | needed by `solve.py` |
|---|---|---|---|
| torch | 2.5.1+cu124 (CUDA 12.4) | batched optimisation (Adam), GPU tensors | yes |
| nvdiffrast | 0.4.0 | differentiable rasterisation (silhouette, depth, vertex colour, textured renders) | yes |
| opencv-python (cv2) | 4.11.0 | video decoding, crops, HSV colour rule, distance transforms, connected components | yes |
| trimesh | 5.1.1 | loading `object_mesh.glb` (texture, UVs) | yes |
| open3d | 0.20.0 | quadric decimation of the mesh to 6000 faces (fast scoring renderer) | yes |
| scipy | 1.16.3 | rotations, KD-tree, median/Gaussian filters | yes |
| numpy | 1.26.3 | everything | yes |
| h5py | 3.16.0 | reading `masks.h5` | yes |
| Pillow | 10.2.0 | texture image (via trimesh) | yes |

## Data from the existing pipeline (read-only, not downloaded)
| item | path | used for | needed by `solve.py` |
|---|---|---|---|
| MoGe-2 raw depth (uint16 mm), model `Ruicheng/moge-2-vitl-normal` rev `b135031b`, file sha256 `f0b88b94e099a44120bb333fbab70b8b097af0e383c55a91bc17b6b1ebd6ea57` (from `depth_report.json`) | `/pipeline/depth/moge2/raw.npy` | relative depth profile along the bat (shift-invariant), to choose between out-of-plane-mirror hypotheses | yes |

Inputs used from `/work/inputs`: `video.mp4`, `object_mesh.glb`, `camera.json`, `ground_plane.json`,
`reconstruction.npz` (baseline, FoundationPose and CoCoNet poses as hypotheses; baseline depth for re-anchoring;
human hand keypoints for validation only), `masks.h5`.

## Considered but not used
- SAM2.1-large (`/weights/sam2`) for re-segmenting the bat (the SAM2 masks often miss the dark handle). Not used:
  a deterministic colour rule connected to the existing masks recovered the handle in the checked frames without
  extra GPU time.
- Point trackers / optical flow (CoTracker, RAFT): not used (time / GPU budget).
