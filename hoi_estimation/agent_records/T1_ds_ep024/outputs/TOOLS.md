# Tools and dependencies

No external repositories were cloned and no model weights were downloaded for
the final solution. The final script depends only on the supplied inputs and
the installed software listed below.

## Used by `solve.py`

| Tool/package | Version | Role |
|---|---:|---|
| PyTorch | 2.5.1+cu124 | SE(3) optimization and differentiable rasterization graphs |
| nvdiffrast | 0.4.0 | Mesh rasterization, UV interpolation, and antialiased silhouette |
| NumPy | 1.26.3 | Array/pose math |
| SciPy | 1.16.3 | Rotation averaging/interpolation and Savitzky–Golay smoothing |
| OpenCV (`opencv-python`) | 4.11.0.86 | Video decoding and ROI handling |
| h5py | 3.16.0 | Reading supplied object/person masks |
| trimesh | 5.1.1 | GLB mesh, UV texture, and convex-hull geometry |
| Pillow | 10.2.0 | Texture image extraction through trimesh |

CUDA 12.4 and the system NVIDIA driver/runtime are also required. No pip
installation, Git checkout, or downloaded weight file was needed.

## Used or tried during analysis, not required by `solve.py`

| Tool/package | Version | Use |
|---|---:|---|
| PyTorch3D | 0.7.9 | Tried as an alternative differentiable silhouette renderer; not used in the final script |
| Open3D | 0.20.0 | Availability check only |
| `/pipeline` CARI4D outputs | supplied | Inspected FoundationPose/CoCoNet metadata and MoGe-2 aligned depth for diagnostics; the final script does not read `/pipeline` |
| `/code/reconstruction` CARI4D source | supplied | Inspected stage definitions and pose-frame conventions |
| FFmpeg/ffprobe | system | Video metadata check |

## Downloaded assets

None. There are therefore no external URLs, repository commit hashes, or
downloaded-weight SHA-256 values to report.

## Reproduction artifact

- `solve.py`: SHA-256 `8d6a04e155a39075fd5a428e94b28896b90031e62131d2bab07ab4e2ac28dddd` (run `sha256sum` after final edits if needed)
- `object_poses_world.npz`: SHA-256 `aec523ac087ab284481fa65a8f460d7272439459b971e712b0195b57b9ebe81c`
