# Tools used

No external code repository, model weights, or data were downloaded.  All
downloads/weight fields below are intentionally recorded as none.  The solution
uses only the files supplied in `/work/inputs` and software already installed in
the container.

## Required by `solve.py`

| tool | version | role |
|---|---:|---|
| Python | 3.11.10 | runtime |
| NumPy | 1.26.3 | pose/mask arrays and interpolation |
| SciPy | 1.16.3 | rotation quaternions and Savitzky-Golay filtering |
| PyTorch | 2.5.1+cu124 | GPU tensor operations |
| nvdiffrast | 0.4.0 | differentiable-free mesh rasterization for mask scoring |
| trimesh | 5.1.1 | loading the supplied textured GLB mesh |
| h5py | 3.16.0 | reading the supplied SAM2 masks |

## Inspected / tried but not required by the final script

| tool | version | status |
|---|---:|---|
| OpenCV (`cv2`) | 4.11.0 | used only for diagnostic overlay montages; not used by `solve.py` |
| Pillow | 10.2.0 | used only for diagnostic texture inspection; not used by `solve.py` |
| pyrender | 0.1.45 | availability checked; not used |
| pytorch3d | 0.7.9 | availability checked; not used |
| Open3D | 0.20.0 | availability checked; not used |

## Git repositories

None cloned or used.

## Downloaded model weights and URLs

None downloaded or used by this solution.  The supplied CARI4D intermediate
reconstruction under `/pipeline` was inspected, but the deterministic solver
uses only `/work/inputs` and installed software.  No HuggingFace or other
network access was used.

## Reproducibility

The required command is:

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

The script fixes the NumPy/Torch random seeds, uses deterministic CUDA settings,
reads no scratch files, and makes no network calls.
