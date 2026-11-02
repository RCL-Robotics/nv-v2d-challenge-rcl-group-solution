# External Tools and Data

## Required by `solve.py`

All of these were already installed in the container; nothing was downloaded.

| Tool | Version | Use |
|---|---:|---|
| Python | 3.11 | Runtime |
| PyTorch | 2.5.1+cu124 | GPU tensors and rendering |
| nvdiffrast | 0.4.0 | Differentiable/soft silhouette rendering for static translation refinement; hard mesh-mask rendering and visible mesh-point sampling |
| OpenCV (`cv2`) | 4.11.0 | Video decode, Lucas-Kanade optical flow, RANSAC PnP and LM refinement |
| NumPy | 1.26.3 | Numerical arrays and linear algebra |
| SciPy | 1.16.3 | Rotation utilities and Savitzky-Golay filtering |
| h5py | 3.16.0 | Reading released masks |
| trimesh | 5.1.1 | Loading the exact textured GLB mesh |

`solve.py` needs only the files in `--inputs` plus the packages above. It does
not read `/pipeline`, `/code`, `/weights`, `/work/scratch`, or the network.

## Used for validation / diagnosis only (not needed by `solve.py`)

| Tool | Version / identity | Use |
|---|---:|---|
| pyrender | 0.1.45 | Textured RGB render checks for long-axis spin at rest and sampled dynamic frames |
| CARI4D source tree | git commit `7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80` | Read-only inspection of the released reconstruction pipeline and render utilities; no CARI4D code is invoked by the final solver |

The released FoundationPose and CoCoNet pose fields inside
`inputs/reconstruction.npz` were used as baseline evidence. No model weights
were downloaded or executed by the final solver.

## Downloaded repositories / weights

None.

- Git repositories cloned: none.
- External weights downloaded: none.
- Downloaded-file SHA-256 entries: none.
