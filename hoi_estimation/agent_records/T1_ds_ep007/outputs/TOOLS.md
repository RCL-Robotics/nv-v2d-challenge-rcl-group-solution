# External tools and weights

No network access, repository clone, or package download was used while
producing the final result. No ground-truth dataset or annotation was accessed.

## Software used by `solve.py`

| Tool | Version | Use |
|---|---|---|
| PyTorch | 2.5.1+cu124 | GPU tensors, pose fitting, deterministic execution |
| nvdiffrast | 0.4.0 | Differentiable rasterization, masks, UV texture sampling and z-buffer |
| SAM2 / `v2d_sam2` | packaged `sam2`; `/workspace` git HEAD `7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80` | Re-segment the iron from a tight frame-0 box prompt |
| trimesh | 5.1.1 | Load the supplied GLB and its UV texture |
| OpenCV (`opencv-python`) | 4.11.0 | Video I/O, mask processing, resizing, colour conversion |
| NumPy | 1.26.3 | Numerical arrays and geometry |
| SciPy | 1.16.3 | Quaternion/SO(3) utilities, median and Savitzky–Golay filtering |
| h5py | 3.16.0 | Read supplied person masks |
| Pillow | 10.2.0 | Texture image handling through trimesh |

## Weights

| File | Size | SHA-256 | Used by |
|---|---:|---|---|
| `/weights/sam2/sam2.1_hiera_large.pt` | 898,083,611 bytes | `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318` | `solve.py` |
| `/weights/sam2/sam2.1_hiera_l.yaml` | 3,798 bytes | `545e4325aa5c19a1615d43c946b07276ed4c57214eacf1437e38fa3d9374f636` | `solve.py` |

These are the local preinstalled SAM2 weights supplied in the container. No
weights were downloaded.

## Diagnostics only (not needed by `solve.py`)

- PyTorch3D 0.7.9 — tried for soft silhouette rendering; not retained because
  the final silhouette/scoring pipeline used nvdiffrast directly.
- matplotlib 3.11.2 — trajectory and mask-area plots during diagnosis.
- FFmpeg/ffprobe 4.4.2 — video metadata and frame extraction checks.
- `nvidia-smi` — GPU memory checks.
