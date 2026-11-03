# Tools used

Nothing was downloaded. No pip installs, no git clones, no extra model weights, and no network access at any stage.
Everything below came preinstalled in the container.

| tool | version | used for | needed by `solve.py` |
|---|---|---|---|
| Python | 3.11.10 | everything | yes |
| torch (CUDA) | 2.5.1+cu124 | differentiable silhouette registration (Adam), batched chamfer | yes |
| nvdiffrast | 0.4.0 | rasterising the textured mesh (photometric scores, IoU checks, overlays) | yes |
| numpy | 1.26.3 | | yes |
| scipy | 1.16.3 | rotations (`scipy.spatial.transform`), Gaussian filtering | yes |
| opencv-python | 4.11.0.86 | video decoding, distance transforms, dilation | yes |
| trimesh | 5.1.1 | loading the GLB mesh, surface sampling | yes |
| h5py | 3.16.0 | reading `masks.h5` | yes |
| Pillow | 10.2.0 | texture image (through trimesh) | yes |
| scikit-learn | (container) | one-off clustering of the base's foot vertices during development. The resulting 4 foot points are hard-coded in `solve.py` (`FEET`) as mesh-geometry constants. | no |

Inputs read by `solve.py`: the files in `/work/inputs` only: video, masks, mesh, camera, ground plane, and
`reconstruction.npz` (baseline poses, FoundationPose poses, human hand keypoints). It does not use `/pipeline`,
`/weights` or `/code`. No learned model is run.

Code in `/work/outputs`: `solve.py` (entry point), `common.py` (I/O, renderer, SO(3) helpers), `fit.py`
(silhouette registration), `photo.py` (photometric scoring), `decode.py` (Viterbi), `evaluate.py` (validation
table; not needed by `solve.py`).
