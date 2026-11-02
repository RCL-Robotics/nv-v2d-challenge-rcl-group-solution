# External tools

No external repositories were cloned, no model weights were downloaded, and
no pip packages were installed for this solution.  All software below was
already present in the container.

| Tool/package | Version | Used by `solve.py` | Purpose |
|---|---:|---|---|
| NumPy | 1.26.3 | Yes | Array math, interpolation, NPZ output |
| SciPy | 1.16.3 | Yes | Quaternion/rotation utilities and Gaussian smoothing |
| OpenCV (`opencv-python`) | 4.11.0.86 | Yes | Mask resizing/erosion and image geometry |
| h5py | 3.16.0 | Yes | Reading `masks.h5` |
| PyTorch | 2.5.1+cu124 | Yes | CUDA tensors for rasterization |
| trimesh | 5.1.1 | Yes | Loading the supplied GLB mesh |
| nvdiffrast | 0.4.0 | Yes | Deterministic mesh silhouette rasterization |
| CUDA / PyTorch CUDA runtime | CUDA 12.4 build | Yes | GPU execution |

The supplied CARI4D reconstruction source exists at
`/code/reconstruction` (commit `7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80`),
but `solve.py` does not import or execute it.  The supplied baseline data in
`reconstruction.npz` is the only CARI4D-derived evidence used.

No FoundationPose, CoCoNet, SAM2, MoGe-2, or other external model weights are
invoked by `solve.py`; the released masks and baseline arrays are read as
provided.
