# Tools used

## Final `solve.py` dependencies

All are preinstalled in the container. No downloaded code, model weights, or
network resources are required by the final solver.

| Tool | Version | Use |
|---|---:|---|
| Python | 3.11.10 | Runtime |
| NumPy | 1.26.3 | Array math, transforms |
| SciPy | 1.16.3 | Quaternion/SO(3) utilities and deterministic global/local optimization |
| OpenCV (`opencv-python`) | 4.11.0.86 | Video decoding, resizing, mask operations |
| h5py | 3.16.0 | Reading `masks.h5` |
| trimesh | 5.1.1 | GLB loading, UV texture access, convex hull |
| Pillow | 10.2.0 | Texture access through trimesh |
| PyTorch | 2.5.1+cu124 | GPU rasterization tensors |
| nvdiffrast | 0.4.0 | Differentiable-compatible textured mesh rasterization |
| CUDA / NVIDIA RTX 6000 Ada | driver-provided | GPU rendering |

## Tried during diagnosis, not required by final solver

| Tool | Version | Status |
|---|---:|---|
| pyrender | 0.1.45 | Imported during early renderer checks; replaced by nvdiffrast |
| Open3D | 0.20.0 | Environment availability check only |
| Matplotlib | 3.11.2 | Diagnostic plots only |

## External repositories and weights

- No external git repositories were cloned.
- No public model weights were downloaded.
- The released reconstruction and CARI4D/FoundationPose outputs were inspected as
  baseline evidence, but the final solver recomputes its trajectory from the
  sequence inputs using the exact mesh and released masks. It does not invoke
  CARI4D, FoundationPose, SAM2, MoGe-2, or any other learned model.
