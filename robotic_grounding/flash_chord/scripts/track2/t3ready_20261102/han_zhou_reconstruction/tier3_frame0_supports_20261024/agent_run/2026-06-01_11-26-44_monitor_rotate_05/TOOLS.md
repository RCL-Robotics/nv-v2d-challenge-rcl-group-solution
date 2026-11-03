# Tools used

Nothing was downloaded: no pip installs, no git clones, no new weights. `/work/tools` and the pip user base were not used.
Everything below was already installed in the container or provided read-only.

| tool | version / source | used by `solve.py`? | purpose |
|---|---|---|---|
| Python | /opt/conda/bin/python 3.11 | yes | |
| torch | 2.5.1+cu124 | yes | Adam optimisation of silhouette costs on the GPU |
| nvdiffrast | 0.4.0 | yes | rasterising + antialiasing soft silhouettes / part labels; full-res textured renders for validation |
| numpy | 1.26.3 | yes | |
| scipy | 1.16.3 | yes | `ndimage` median / Gaussian filters |
| opencv-python (cv2) | 4.11.0.86 | yes | video decoding, mask resizing, morphology; LK optical flow in `validate.py` |
| trimesh | 5.1.1 | yes | mesh loading, convex hull (base resting face), vertex normals |
| open3d | 0.20.0 | yes | deterministic quadric decimation of the mesh to 8k faces for fast silhouette rendering |
| h5py | 3.16.0 | yes | reading `masks.h5` |
| roma | 1.6.1 | no (scratch experiments only) | rotation vectors in free 6-DoF diagnostic fits |
| matplotlib | 3.11.2 | no (scratch plots only) | |
| SAM-2 (installed `sam2` package) + `/weights/sam2/sam2.1_hiera_large.pt` (sha256 2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318) | provided | no | tried as a box-prompted image predictor to get full-monitor masks (stand + base); rejected because it merged the monitor with the dark-clothed person or dropped the stand |

`solve.py` reads only `--inputs` (video, masks, mesh, camera, plane, reconstruction). It does not use `/pipeline`, `/weights` or `/code`.
It needs the helper modules `hoi_fit.py` and `hoi_render.py` in `/work/outputs`. `validate.py` additionally needs `hoi_validate.py`.
