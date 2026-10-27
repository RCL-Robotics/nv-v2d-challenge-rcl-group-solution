# Tools used

No external tools, repositories, model weights or pip packages were downloaded. Everything ran on software
already in the container (Python 3.11.10):

| package | version | used by solve.py |
|---|---|---|
| torch (CUDA 12.4) | 2.5.1+cu124 | yes (differentiable render-and-compare, Adam) |
| nvdiffrast | 0.4.0 | yes (rasterisation, texture lookup, antialiasing of the textured mesh) |
| opencv-python | 4.11.0.86 | yes (video decoding, blur, dilation) |
| numpy | 1.26.3 | yes |
| scipy | 1.16.3 | yes (median/gaussian filters, rotations) |
| trimesh | 5.1.1 | yes (mesh + texture loading) |
| h5py | 3.16.0 | yes (masks.h5) |
| Pillow | 10.2.0 | yes (texture image, through trimesh) |

Not used: the CARI4D pipeline modules (`/workspace`, `/code`), the `/weights` models (FoundationPose,
SAM2, SAM3D-body, CoCoNet, MoGe-2) and the `/pipeline` intermediates. I considered and rejected
re-running SAM2/FoundationPose: the supplied masks and the baseline yaw track were already good enough, and the
remaining errors were depth and temporal consistency.

`solve.py` needs only `/work/inputs`, `/work/outputs/solve.py`, `/work/outputs/hoi_lib.py` and
`/work/outputs/traj_opt.py`. It makes no network access.
