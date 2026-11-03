# Tools used

No external tools were downloaded. No pip packages were installed, no git repositories were cloned, and no
model weights were fetched. `/work/tools` and the pip user base are empty. Everything ran on software
already in the container.

## Container software used by `solve.py` (all needed)
| package | version | use |
|---|---|---|
| python | 3.11 (conda, container) | |
| numpy | 1.26.3 | everything |
| scipy | 1.16.3 | rotations / slerp, Gaussian and median filters |
| torch | 2.5.1+cu124 | GPU tensors, Adam optimisation |
| nvdiffrast | 0.4.0 | differentiable silhouette rendering, textured rendering for yaw scoring |
| roma | 1.6.1 | rotation-vector → matrix inside the optimisation |
| trimesh | 5.1.1 | loading `object_mesh.glb` (geometry, UVs, texture) |
| opencv-python (cv2) | 4.11.0 | table-top colour segmentation (occluder mask) |
| h5py | 3.16.0 | reading `masks.h5` |
| av (PyAV) | 18.0.0 | video decoding |

## Data used by `solve.py`
Only `/work/inputs` (video, masks, mesh, camera, ground plane, `reconstruction.npz`, including its human
keypoints). It does not read `/pipeline`, `/weights` or `/code`.

## Used during analysis only (not needed by `solve.py`)
- `/pipeline/mhr_init/episode_000002.pkl` (initial SAM-3D-Body human, before contact optimisation): checked
  that the hand depth is independent of the object estimate (found ~1 cm agreement with the refined human,
  and noisy as an object-depth cue, so it was not used).
- `/pipeline/depth/moge2/aligned.h5`: inspected only (per-frame scale varies 1.02–1.12, so not used).
- Same container packages as above, plus matplotlib-free OpenCV visualisations.
