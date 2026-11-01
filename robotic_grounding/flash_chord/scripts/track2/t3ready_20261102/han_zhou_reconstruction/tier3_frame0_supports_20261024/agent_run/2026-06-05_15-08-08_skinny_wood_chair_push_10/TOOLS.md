# Tools used

No external tools, repositories, pip packages or model weights were downloaded. No network access was used, and nothing was installed under `/work/tools` or the pip user base.

All software came from the provided container (Python 3.11.10):

| Package | Version | Used by `solve.py`? | Purpose |
|---|---|---|---|
| torch | 2.5.1+cu124 | yes | optimisation (Adam), GPU tensors |
| nvdiffrast | 0.4.0 | yes | differentiable silhouette rendering of the mesh |
| roma | 1.6.1 | yes | rotation-vector ↔ rotation-matrix conversion |
| trimesh | 5.1.1 | yes | loading `object_mesh.glb` |
| h5py | 3.16.0 | yes | reading `masks.h5` |
| opencv-python / opencv-contrib-python | 4.11.0.86 | yes (dilating the person mask) | also used in validation: video decoding, KLT tracks |
| numpy | 1.26.3 | yes | |
| scipy | 1.16.3 | no | exploration only |
| scikit-learn | 1.9.1 | no | exploration only (KMeans to locate the leg tips; `solve.py` uses a quadrant rule) |

Models and weights: none were used. The pipeline weights under `/weights` and the CARI4D outputs under `/pipeline` were not needed.

Validation scripts (development only, not needed by `solve.py`) are in `/work/scratch`: `validate.py`, `klt_check.py`, `cv_moving.py`, `smooth.py`, `sbs.py`.
