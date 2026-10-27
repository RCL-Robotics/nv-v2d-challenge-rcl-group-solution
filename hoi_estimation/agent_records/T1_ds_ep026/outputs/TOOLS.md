# Tools used

## Runtime dependencies required by `solve.py`

All packages were already installed in the container; nothing was downloaded or pip-installed for the final solution.

| Tool | Version | Role in `solve.py` |
|---|---:|---|
| Python | 3.11 | Runtime |
| PyTorch | 2.5.1+cu124 | GPU tensors, autograd, pose optimization |
| CUDA | 12.4 | GPU backend |
| nvdiffrast | 0.4.0 | Differentiable mesh rasterization and texture sampling |
| NumPy | 1.26.3 | Array and NPZ I/O |
| SciPy | 1.16.3 | Rotation utilities and Savitzky-Golay smoothing |
| OpenCV (`opencv-python`) | 4.11.0.86 | Video frame decoding and image preparation |
| h5py | 3.16.0 | Reading released masks |
| trimesh | 5.1.1 | Loading mesh-file geometry, UVs, and base-color texture |
| Pillow | 10.2.0 | Texture image returned through trimesh |

## Development-only tools checked or used for inspection

| Tool | Version | Use |
|---|---:|---|
| FFmpeg | 4.4.2 | Extract video frames/contact sheets for visual diagnosis |
| PyTorch3D | 0.7.9 | Checked as an alternative renderer; not used by `solve.py` |
| Open3D | 0.20.0 | Checked as an alternative geometry utility; not used by `solve.py` |
| pyrender | 0.1.45 | Checked as an alternative renderer; not used by `solve.py` |
| scikit-image | 0.26.0 | Checked for image metrics; not used by `solve.py` |

## Provided data/tools examined

- `/pipeline` CARI4D intermediate bundle: inspected the baseline/reconstruction reports and refinement metadata to diagnose the tracker output. `solve.py` does not import CARI4D or require `/pipeline`.
- `/workspace/v2d_cari4d` and `/code/reconstruction`: inspected as pipeline documentation/source. No CARI4D model was rerun and no code from these trees is required by `solve.py`.
- `/weights`: not used by the final solution.

## External repositories, downloads, and weights

- Git repositories cloned: **none**.
- Pip packages installed for this task: **none**.
- Model weights downloaded: **none**.
- Network access used during solution development: **none**.

The only object texture/geometry used is the released `inputs/object_mesh.glb`. The only pose source used is the released baseline plus the released video and masks.
