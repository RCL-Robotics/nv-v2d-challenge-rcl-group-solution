# External tools and dependencies

No packages were downloaded or installed for this task. No model weights were downloaded. The final solver uses no learned depth/pose model and needs no network access.

## Required by `solve.py`

| Tool | Version | Role |
|---|---:|---|
| Python | 3.11.10 | Runtime |
| PyTorch | 2.5.1+cu124 | Tensor/GPU execution |
| nvdiffrast | 0.4.0 | Differentiable full-resolution mesh silhouette rasterization |
| NumPy | 1.26.3 | Array/matrix operations |
| SciPy | 1.16.3 | Powelle optimization, quaternion slerp, Savitzky--Golay filter |
| OpenCV (`opencv-python`) | 4.11.0 | Mask morphology and image utilities |
| h5py | 3.16.0 | Reading SAM2 masks |
| trimesh | 5.1.1 | Loading the exact GLB mesh |

`solve.py` does not require `/pipeline`, `/weights`, or `/code`; it reads only `/work/inputs` and the packages above.

## Used during diagnosis, not required by `solve.py`

| Tool | Version / revision | Role |
|---|---:|---|
| pyrender | 0.1.45 | Initial visual render/overlay sanity checks; replaced by nvdiffrast in the solver |
| Pillow | 10.2.0 | Inspecting the GLB base-color texture atlas |
| CARI4D reconstruction source under `/code/reconstruction` | git `7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80` | Inspected for data semantics; no code from it is called by the solver |

`open3d`, `pytorch3d`, and `scikit-image` were checked for availability but were not used in any final or diagnostic computation.

## Downloaded repositories/weights

None. The provided `/pipeline/depth/moge2/aligned.h5` was read for a depth cross-check only; it is not used by `solve.py`.
