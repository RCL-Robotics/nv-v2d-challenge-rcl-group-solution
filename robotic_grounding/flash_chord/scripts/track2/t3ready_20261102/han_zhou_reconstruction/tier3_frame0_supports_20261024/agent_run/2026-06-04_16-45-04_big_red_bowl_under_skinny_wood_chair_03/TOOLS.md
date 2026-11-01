# Tools used

Nothing was downloaded: no pip installs, no git clones, no model weights, no network access. Everything below was
already installed in the container.

| tool | version | used for | needed by `solve.py` |
|---|---|---|---|
| Python | 3.11.10 | everything | yes |
| numpy | 1.26.3 | everything | yes |
| torch (CUDA) | 2.5.1+cu124 | GPU tensors, Adam optimiser | yes |
| nvdiffrast | 0.4.0 | silhouette rasterisation (forward only in the final fits) | yes |
| open3d | 0.20.0 | quadric decimation of the mesh to 8k faces for silhouettes | yes |
| trimesh | 5.1.1 | loading `object_mesh.glb` | yes |
| h5py | 3.16.0 | reading `masks.h5` | yes |
| opencv-python (cv2) | 4.11.0 | video decoding, dilation | yes |
| scipy | 1.16.3 | scratch analysis only | no |
| matplotlib | 3.11.2 | scratch plots only | no |
| ffmpeg (system) | 4.4.2 | contact sheets for visual inspection (scratch only) | no |

Data used by `solve.py` comes only from `/work/inputs`: video, masks, mesh, camera, ground plane,
`reconstruction.npz` (baseline poses, human keypoints) and `frame0_reference_human.npz`. No ground truth and no
pipeline intermediates from `/pipeline` are used. No model weights from `/weights` are used.

Scratch scripts used during development (`/work/scratch`, not needed by `solve.py`):
* `evalv.py` – validation table
* `overlay2.py` – overlays
* `dettest.py` – determinism test
* `fit_all.py`, `static_fit.py`, `vis.py` – early experiments
