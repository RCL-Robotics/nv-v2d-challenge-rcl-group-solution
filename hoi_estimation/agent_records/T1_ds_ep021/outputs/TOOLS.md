# Tools used

No external model weights or GitHub repositories were downloaded. `solve.py` runs
offline using the supplied inputs and preinstalled packages.

## Required by `solve.py`

| Tool | Version | Use |
|---|---:|---|
| Python | 3.11 | Runtime |
| PyTorch | 2.5.1+cu124 | GPU tensor math, optimization |
| nvdiffrast | 0.4.0 | Differentiable mesh rasterization / silhouette fitting |
| NumPy | 1.26.3 | Arrays, SE(3) manipulation |
| SciPy | 1.16.3 | Quaternion conversion and Savitzky-Golay smoothing |
| h5py | 3.16.0 | Read supplied SAM2 masks |
| OpenCV (`opencv-python`) | 4.11.0 | Mask dilation/resizing and diagnostics |
| trimesh | 5.1.1 | Load exact GLB mesh and geometry |
| Pillow | 10.2.0 | Installed GLB texture support dependency |

## Used for diagnosis / validation, not required by `solve.py`

| Tool | Version | Use |
|---|---:|---|
| FFmpeg | 4.4.2 | Extract video contact sheets and key frames |
| MoGe-2 aligned depth from `/pipeline/depth/moge2/aligned.h5` | supplied pipeline output | Independent depth residual check only |
| PyRender | 0.1.45 | Availability probe only; final rendering uses nvdiffrast |
| Open3D | 0.20.0 | Availability probe only; not used by the final method |
| ImageIO | 2.38.0 | Availability probe only |

## Downloaded artifacts

None. No network was used during the final solve, and no external checkpoints were
downloaded. The supplied local reconstruction and `/pipeline` MoGe-2 output were used
only as evidence/diagnostics described in `REPORT.md`.

## Git repositories

None cloned or checked out for this solution.

## Reproducibility

`solve.py` uses seed 0 for Python, NumPy, and PyTorch, disables TF32, uses fixed phase
boundaries and fixed optimization schedules, and writes only
`object_poses_world.npz`.
