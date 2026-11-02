# Tools used

No external tools, repositories, model weights or pip packages were downloaded. Everything below was
already installed in the container (Python 3.11.10).

| tool | version | used for | needed by solve.py |
|---|---|---|---|
| torch | 2.5.1+cu124 | GPU tensors, image scoring | yes |
| nvdiffrast | 0.4.0 | rasterising the textured mesh (render-and-compare) | yes |
| numpy | 1.26.3 | everything | yes |
| scipy | 1.16.3 | rotations / SLERP, Gaussian and median filters | yes |
| trimesh | 5.1.1 | loading `object_mesh.glb` | yes |
| opencv-python | 4.11.0 | video decoding, crops, visualisation | yes |
| h5py | 3.16.0 | reading `masks.h5` | yes |
| ffmpeg (system) | – | frame extraction for visual inspection only | no |

Data used by `solve.py`: only `/work/inputs` (video, masks, mesh, camera, ground plane, reconstruction.npz
incl. human keypoints and contact probabilities). It does not use `/pipeline`, `/weights` or the network.
The pipeline modules in `/workspace` and `/code` were only read (MHR70 keypoint names).
