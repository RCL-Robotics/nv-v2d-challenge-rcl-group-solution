# External tools

No package, repository, or model weight was downloaded for the final solve.
Everything below is provided or preinstalled in the container.

## Required by `solve.py`

| Tool | Version / revision | Use |
|---|---|---|
| Python | 3.11.10 | runtime |
| NumPy | 1.26.3 | numerical arrays |
| SciPy | 1.16.3 | rotations, median/Gaussian/Savitzky-Golay filtering |
| OpenCV (`opencv-python`) | 4.11.0 | video/mask/image operations |
| h5py | 3.16.0 | read `masks.h5` |
| trimesh | 5.1.1 | load exact GLB mesh and texture |
| Pillow | installed with trimesh | texture transport |
| PyTorch | 2.5.1+cu124 | GPU rasterization tensors |
| nvdiffrast | 0.4.0, commit `253ac4fcea7de5f396371124af597e6cc957bfae` | differentiable rasterizer used for masks, depth, and textured scoring |

`solve.py` also reads `/pipeline/depth/moge2/aligned.h5` only to initialize
camera depth for the two pooled stationary-pose fits.  It does not need
`/weights` or `/code`.

## Tried but not required by `solve.py`

* NVIDIA CARI4D / FoundationPose wrappers under `/code/reconstruction`
  (repository commit `7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80`).  The
  provided FoundationPose outputs were inspected and used only as the released
  baseline evidence.  A direct FoundationPose scorer experiment was attempted
  but not used by the final deterministic script.
* Provided FoundationPose PyTorch weights (not downloaded here):
  * `/weights/foundationpose/nvlabs_pytorch/2024-01-11-20-02-45/model_best.pth`,
    SHA-256 `81924d384bf5c26c646ee4783104982ae3d1e049c181c36641b6a7aeae494c26`
  * `/weights/foundationpose/nvlabs_pytorch/2023-10-28-18-33-37/model_best.pth`,
    SHA-256 `774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60`

## Reproducibility

The solve fixes all random seeds, runs offline, and writes only the requested
NPZ plus this report and `TOOLS.md`.  No network URLs were used.
