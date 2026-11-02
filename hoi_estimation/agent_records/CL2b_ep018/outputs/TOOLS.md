# Tools used

No external tools, repositories or model weights were downloaded. Nothing was installed under `/work/tools` or the
pip user base. Everything ran with software that was already in the container.

| tool | version | used for | needed by `solve.py` |
|---|---|---|---|
| Python | 3.11.10 | everything | yes |
| torch (CUDA) | 2.5.1+cu124 | Adam pose optimisation, NCC/IoU on the GPU | yes |
| nvdiffrast | 0.4.0 | differentiable rasterisation of the textured mesh (soft silhouettes via `antialias`, texture rendering) | yes |
| trimesh | 5.1.1 | loading `object_mesh.glb`; surface sampling for a hand-contact diagnostic (scratch only) | yes |
| opencv-python (cv2) | 4.11.0 | video decoding, resizing, overlays (scratch) | yes |
| h5py | 3.16.0 | reading `masks.h5` | yes |
| numpy | 1.26.3 | everything | yes |
| scipy | 1.16.3 | rotation utilities (`scipy.spatial.transform`), KD-tree for the hand diagnostic (scratch only) | yes |
| Pillow | 10.2.0 | texture image (through trimesh) | yes |

Inputs `solve.py` reads: only `/work/inputs/*` (video, masks, mesh, camera, ground plane, reconstruction).
It does not use `/pipeline`, `/weights` or `/code`.

Code: `/work/outputs/solve.py` and `/work/outputs/hoi_lib.py`, which holds the renderer, the optimisers and the
scoring functions.
