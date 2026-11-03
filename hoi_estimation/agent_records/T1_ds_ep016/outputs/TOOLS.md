# Tools and external dependencies

No model weights or repositories were downloaded for this solution.  The
supplied reconstruction, mesh, masks, camera metadata, and existing installed
software were used.  `solve.py` requires no network access.

## Required by `solve.py`

| Tool | Version / revision | Role | Needed at solve time |
|---|---|---|---|
| Python | 3.11.10 | runtime | yes |
| NumPy | 1.26.3 | arrays, phase construction, SE(3) operations | yes |
| SciPy | 1.16.3 | quaternion rotations, Savitzky-Golay and median filters | yes |
| h5py | 3.16.0 | read supplied `masks.h5` for static-pose refinement | yes |
| trimesh | 5.1.1 | load `object_mesh.glb` and texture | yes |
| PyTorch | 2.5.1+cu124 | GPU tensors for mask rendering | yes |
| nvdiffrast | 0.4.0 | rasterize mesh for static-pose mask fitting | yes |
| Pillow | 10.2.0 | decode GLB base-color texture via trimesh | yes |
| CARI4D `Utils.py` | `/code/reconstruction` git `7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80` | batched nvdiffrast render helper used by the static-pose fitter | yes |
| NVIDIA Warp | 1.17.0 | imported indirectly by CARI4D `Utils.py` | yes (indirect) |
| OpenCV | 4.11.0 | imported indirectly by CARI4D `Utils.py` | yes (indirect) |
| Open3D | 0.20.0 | imported indirectly by CARI4D `Utils.py` | yes (indirect) |
| Kornia | 0.8.3 | optional import inside CARI4D `Utils.py` | not used |

The GPU renderer in `solve.py` uses less than about 1.4 GB and completes the
whole solve in about 8 seconds on the allocated NVIDIA RTX 6000 Ada GPU.

## Used only for offline diagnostics

| Tool | Version | Role |
|---|---|---|
| FFmpeg | 4.4.2-0ubuntu0.22.04.1 | extract video frames for visual inspection and contact sheets |
| OpenCV | 4.11.0 | frame crops, masks, contact sheets, rendered-mask scoring |
| h5py | 3.16.0 | read object/person masks and supplied aligned depth for diagnostics |
| PyTorch + nvdiffrast | 2.5.1+cu124 / 0.4.0 | render textured pose hypotheses and compute mask/photometric diagnostics |
| Matplotlib-compatible NumPy tooling | n/a | numeric summaries and phase tables |

## Probed but not used

These packages were detected while checking the container and were not used in
the reconstruction or final script: `pyrender 0.1.45`, `pytorch3d 0.7.9`,
`torchvision 0.20.1+cu124`, `scikit-image 0.26.3`, `romatch` (not installed),
and `cotracker` (not installed).

## Downloaded resources

None.  No external weights, URLs, git clones, or package installations were
needed.  The supplied read-only weights under `/weights` and the supplied
`/pipeline` outputs were not loaded by `solve.py`; the result is recomputed from
`/work/inputs/reconstruction.npz`, `object_mesh.glb`, `camera.json`, and
`masks.h5`.
