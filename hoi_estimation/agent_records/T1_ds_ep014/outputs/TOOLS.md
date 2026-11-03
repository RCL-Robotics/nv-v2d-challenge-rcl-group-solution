# Tools and dependencies

No external repositories, model weights, or pretrained checkpoints were
downloaded for this solution. The deterministic solver needs only the supplied
inputs and the preinstalled software below.

| Tool | Version | Used by `solve.py` | Purpose |
|---|---:|---:|---|
| Python | 3.11 | Yes | Runtime |
| PyTorch | 2.5.1+cu124 | Yes | GPU tensors and mesh transforms |
| nvdiffrast | 0.4.0 | Yes | Rasterization for object masks and textured appearance |
| trimesh | 5.1.1 | Yes | GLB mesh and UV/texture loading |
| OpenCV (`cv2`) | 4.11.0 | Yes | Video/mask I/O, resizing, Sobel and NCC preprocessing |
| NumPy | installed with environment | Yes | Numerical arrays and SE(3) trajectory processing |
| SciPy | 1.16.3 | Yes | Savitzky-Golay filtering and rotation utilities |
| h5py | 3.16.0 | Yes | Reading `masks.h5` |
| Pillow (`PIL`) | 10.2.0 | Yes | Accessing the GLB base-color texture |

Additional supplied data inspected during validation, but not required by the
final offline solver:

- `/pipeline/depth/moge2/aligned.h5`: aligned MoGe-2 depth used only for the
  reported depth diagnostic.
- `/pipeline/inference/*.pth` and `/pipeline/foundationpose/*.pkl`: inspected
  to understand the released reconstruction; `solve.py` does not read a new
  pose output from them.

GPU use: one selected CUDA device; final measured solver runtime 274.4 s,
well below the 45-minute budget.

Downloaded tools/weights: none.
