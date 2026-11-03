# Tools used

## Final solver dependencies

`solve.py` requires only:

- Python 3.11.10
- NumPy 1.26.3
- SciPy 1.16.3

It does not need GPU, network access, model weights, `/pipeline`, `/code`, or
scratch files.

## Diagnostic and validation tools

These tools were used to inspect the video, render the mesh, compute mask and
texture-feature diagnostics, and compare candidate trajectories. They are not
required by `solve.py`.

| Tool | Version | Use |
|---|---:|---|
| PyTorch | 2.5.1+cu124 | GPU tensor operations for diagnostic rasterization |
| nvdiffrast | 0.4.0 | Mesh silhouette/attribute rendering and full-resolution IoU checks |
| OpenCV | 4.11.0 | Video decoding, colour-space conversion, blue-handle segmentation |
| h5py | 3.16.0 | Reading released masks and inspecting aligned depth |
| trimesh | 5.1.1 | GLB loading, mesh vertices/faces/UVs |
| Open3D | 0.20.0 | Point-cloud/RANSAC experiments for table/background depth |
| PyTorch3D | 0.7.9 | Attempted textured rendering; not used for a reported decision |
| Pillow | 10.2.0 | Inspecting the released texture atlas |

## Repositories, weights, and downloads

- No repositories were cloned.
- No weights were downloaded.
- No public/challenge annotations or ground-truth data were downloaded.
- The installed `/code/reconstruction` checkout was inspected for context only.
  Its reported HEAD was `7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80`; none of its
  code is imported by `solve.py`.
- The released CARI4D intermediate files under `/pipeline` were inspected as
  inputs/evidence, but `solve.py` does not depend on them.

## Determinism

`solve.py` uses no random numbers. The only numerical filters are deterministic
SciPy rotation averaging and Savitzky-Golay filtering. The output arrays are
written as float64, and every rotation is re-orthonormalized before saving.
