# Tools and dependencies

No packages, repositories, model weights, or other assets were downloaded. No network requests, challenge annotations, ground-truth trajectories, or external video data were used. All software below was preinstalled in the container.

## Required by solve.py

| Tool/package | Version | Use |
| --- | --- | --- |
| CPython | 3.11.10 | Script runtime and standard library |
| NumPy | 1.26.3 | Arrays, mesh/camera transformations, NPZ output |
| SciPy | 1.16.3 | Rotation calculations and nearest mesh-vertex texture lookup |
| OpenCV (`opencv-python`) | 4.11.0.86 | Video decoding, image warping, HSV color extraction, distance transforms |
| trimesh | 5.1.1 | Exact GLB loading, deterministic surface sampling, texture access |
| h5py | 3.16.0 | Read released object/person masks |
| PyTorch | 2.5.1+cu124 | Fixed-distance rotation and image-plane fitting on the selected CUDA GPU |
| nvdiffrast | 0.4.0 | Exact textured mesh rendering and candidate validation |
| Pillow | 10.2.0 | GLB texture decoding through trimesh |

The script uses the supplied `/work/inputs` files, these installed dependencies, and optional verified caches in `/work/tools`. It does not need `/pipeline`, `/weights`, `/code`, scratch files, or the exploratory outputs. With `--recompute`, neither cache is needed. CUDA GPU 0 was the already-selected device (`CUDA_VISIBLE_DEVICES=0`).

## Other tools used or tried

- Bash, coreutils, ripgrep, and `nvidia-smi`: inspect task files, installed software, outputs, and GPU status.
- Python `importlib` / `importlib.metadata`: inspect installed package availability and versions. Availability was checked for Open3D 0.20.0, scikit-image 0.26.0, and pyrender; they were not used in fitting or in `solve.py`.
- The exploratory scripts in this directory used the same packages listed above. Initial untextured silhouette fitting and CUDA `grid_sample` fitting were tried; the final script adds orange texture cues and uses explicit bilinear interpolation to make backward evaluation deterministic.

There are no downloaded-weight URLs/hashes or cloned repository commits to list. The installed packages' source repositories were not cloned or modified.

## Locally generated optimization cache

`/work/tools/cane_initial_candidates.npz` contains the 17 initial image-fitted hypotheses for each supplied frame. It was generated here from the released mesh, video, masks, and baseline by the included solver, with fixed seeds and 220 optimization steps. It is not downloaded data or a model weight. The matching JSON manifest records SHA-256 of every input used and of the cache. The default solve verifies these hashes before using it; `--recompute` rebuilds the initial hypotheses without it. Final render validation, temporal fitting, acceptance criteria, and output transformations are implemented in `solve.py`.

Initial candidate cache SHA-256: `7f1fe70c0ff0cbabd8783775cf6e56545d9a3fa4ed2e8266351c1b1ae85ad89a`.

`/work/tools/cane_validated_hypotheses.npz` contains locally generated fitted poses and full-mesh image-validation metrics, including temporal proposals, from the same solver. `/work/tools/cane_validated_hypotheses.json` records its SHA-256 and all source-input hashes. The default run uses this cache when valid, then reruns trajectory selection and conservative acceptance. `--recompute` ignores both caches and reruns all image fitting and rendering. It requires the same installed CUDA tools; default cached execution performs no GPU fitting/rendering.

Validated hypothesis cache SHA-256: `b3cc8c8d02c7e89c613c3ed4f7115c07e3e197d19151c048ba6cb9826ff6cb27`.
