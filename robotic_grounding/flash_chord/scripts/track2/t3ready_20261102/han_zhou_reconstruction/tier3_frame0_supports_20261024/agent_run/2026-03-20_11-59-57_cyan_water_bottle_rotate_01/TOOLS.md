# Tools used

No external tools, repositories, model weights or pip packages were downloaded. Nothing was stored under
`/work/tools` or the pip user base. Everything came from the provided container.

| tool | version | source | used by solve.py |
|---|---|---|---|
| Python | 3.11.10 | container | yes |
| torch (CUDA) | 2.5.1+cu124 | container | yes (Adam optimisation, batched losses) |
| nvdiffrast | 0.4.0 | container | yes (differentiable rasterisation, texture, antialias) |
| opencv-python (cv2) | 4.11.0 | container | yes (video decoding) |
| trimesh | 5.1.1 | container | yes (loads `object_mesh.glb`) |
| h5py | 3.16.0 | container | yes (reads `masks.h5`) |
| numpy | 1.26.3 | container | yes |
| scipy | 1.16.3 | container | yes (median filter for phase segmentation) |
| Pillow | 10.2.0 | container (via trimesh texture) | yes |

**Data `solve.py` reads.** It reads only `/work/inputs`:
* `video.mp4`, `masks.h5`, `object_mesh.glb`, `camera.json`;
* `reconstruction.npz` (baseline and FoundationPose poses, used as initial hypotheses and as the depth anchor).

It does not use `/pipeline`, `/weights` or the network.

**Other files in `/work/outputs`.**
* `hoi_lib.py` is the helper module that `solve.py` imports.
* `validate.py` runs the ground-truth-free checks and is not needed by `solve.py`.
* `object_poses_world_diag.{npz,json}` are diagnostics written by `solve.py`.
