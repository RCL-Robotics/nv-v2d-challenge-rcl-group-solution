# Tools used

No external tools, repositories, pip packages or model weights were downloaded for this task. Everything ran on
software that was already installed in the container. No network access was used, and HuggingFace was never
contacted.

| tool | version | used for | needed by `solve.py` |
|---|---|---|---|
| Python | 3.11.10 | everything | yes |
| torch | 2.5.1+cu124 (CUDA 12.4) | optimisation (Adam), GPU tensors | yes |
| nvdiffrast | 0.4.0 | differentiable rasterisation (silhouette with antialias, crease labels, textured renders) | yes |
| numpy | 1.26.3 | | yes |
| scipy | 1.16.3 | rotation averaging (`Rotation.mean`), KD-tree for the hand-contact cue | yes |
| opencv-contrib-python / opencv-python | 4.11.0.86 | video decoding, Canny edges, distance transforms, crop warps | yes |
| trimesh | 5.1.1 | mesh loading, oriented bounding box | yes |
| open3d | 0.20.0 | quadric decimation of the mesh (8k faces) for fast silhouette rendering | yes |
| h5py | 3.16.0 | reading `masks.h5` | yes |
| matplotlib | 3.11.2 | diagnostic plots during development only | no |

Inputs `solve.py` reads: `/work/inputs/*` only: `reconstruction.npz` (baseline `T_world_mesh`,
`human_keypoints_world`, `contact_prob_right_left`), `masks.h5`, `video.mp4`, `object_mesh.glb`, `camera.json`
and `ground_plane.json`. It does not need `/pipeline`, `/weights` or `/code`. It reads no scratch files and uses
no network.

Things I looked at but did not use: the CARI4D modules under `/workspace` and `/code`, and the
FoundationPose / CoCoNet estimates in `reconstruction.npz`. CoCoNet's rotation equals the baseline rotation up to
a constant 1.4° camera offset, so it adds no independent rotation evidence. FoundationPose's rotation stays within
2–10° of the baseline. I only used them in the depth analysis.
