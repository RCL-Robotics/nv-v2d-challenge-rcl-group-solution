# External tools and dependencies

No external code repositories, model weights, or datasets were downloaded for this
task. The solution operates on the released episode inputs and the supplied
reconstruction. The CUDA rasterizer is the only GPU component used by `solve.py`.

## Python environment

| Tool | Version | Role | Needed by `solve.py` |
|---|---:|---|---|
| Python | 3.11.10 | Runtime | Yes |
| PyTorch | 2.5.1+cu124 | CUDA tensors / nvdiffrast bridge | Yes |
| nvdiffrast | 0.4.0 | Differentiable mesh rasterization (masks used in the final script) | Yes |
| NumPy | 1.26.3 | Array / SE(3) computations | Yes |
| SciPy | 1.16.3 | Median/Savitzky-Golay filtering and SO(3) operations | Yes |
| trimesh | 5.1.1 | GLB mesh and texture loading | Yes |
| h5py | 3.16.0 | SAM mask HDF5 loading | Yes |
| OpenCV (`opencv-python`) | 4.11.0.86 | Experimental overlays, contour checks, photometric validation | No |
| Pillow | 10.2.0 | Experimental decoding of aligned depth PNG payloads | No |

## Supplied repositories / weights

The following were used as source evidence or during diagnosis only:

- `/code/reconstruction`: CARI4D reconstruction source, git commit
  `7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80`.
- `/workspace/v2d_cari4d`, `/workspace/v2d_foundation_pose`,
  `/workspace/v2d_moge`, and `/workspace/v2d_sam2`: installed package trees without
  local git metadata.
- `/weights`: read-only FoundationPose, SAM2, MoGe-2, SAM3D-Body, and CoCoNet
  weights supplied in the container. No weight was downloaded and the final script
  does not load them directly.
- `/pipeline`: supplied intermediate CARI4D output. It was inspected during diagnosis
  (depth, masks, reliability diagnostics, refined poses), but `solve.py` needs none
  of its files because the released `reconstruction.npz` and masks already contain
  the required evidence.

## Downloaded artifacts

None. Consequently there are no download URLs or downloaded-weight SHA-256 values.

## Output checksum

- `object_poses_world.npz`
  SHA-256: `311016ab3aeab891e1a31fbff506edd28a2253713b6d00498d07eedc51fe5511`
  (checksum from the final run described in `REPORT.md`; it may differ by harmless
  last-bit CPU/GPU rounding if rerun on a different device.)
