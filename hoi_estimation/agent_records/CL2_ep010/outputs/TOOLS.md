# Tools used

I downloaded nothing: no pip packages, git repositories or model weights. Everything was already installed in the container,
and nothing was placed in `/work/tools` or the pip user base.

| tool | version | used for | needed by `solve.py` |
|---|---|---|---|
| Python | 3.11.10 | everything | yes |
| PyTorch (CUDA 12.4) | 2.5.1+cu124 | optimisation (Adam), GPU tensors | yes |
| nvdiffrast | 0.4.0 | differentiable silhouette rendering (rasterize + antialias) | yes |
| numpy | 1.26.3 | — | yes |
| scipy | 1.16.3 | rotation averaging (`Rotation.mean`); cKDTree in the analysis scripts | yes (Rotation) |
| opencv-python (cv2) | 4.11.0 | video decoding, resizing, dilation, overlays | yes |
| h5py | 3.16.0 | reading `masks.h5` | yes |
| trimesh | 5.1.1 | loading `object_mesh.glb` | yes |
| Pillow | 10.2.0 | texture access while rendering inspection overlays (analysis only) | no |

`solve.py` uses only files in `/work/inputs`. It does not use `/pipeline`, `/weights` or `/code`, and it needs no network access.
