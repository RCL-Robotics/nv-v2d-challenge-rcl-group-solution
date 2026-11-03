# Tools and External Dependencies

No external repositories or model weights were downloaded. All analysis was performed offline on
the provided inputs and the preinstalled CARI4D/FoundationPose/MoGe-2 artifacts. The internet was
not needed.

## Provided software and data inspected or used

| Tool / source | Version or revision | Use | Needed by `solve.py` |
|---|---|---|---|
| Python | 3.11.10 | Runtime | Yes |
| NumPy | 1.26.3 | Array operations, pose composition | Yes |
| SciPy | 1.16.3 | Quaternion rotations, Gaussian filtering | Yes |
| h5py | 3.16.0 | Reading provided SAM2 masks during validation | No |
| OpenCV (`opencv-python`) | 4.11.0 | Video frames, masks, image metrics | No |
| PyTorch | 2.5.1+cu124 | GPU mesh rendering during validation | No |
| nvdiffrast | 0.4.0 | Differentiable mesh rasterization during validation | No |
| trimesh | 5.1.1 | Loading exact GLB mesh and texture | No |
| Pillow | 10.2.0 | Reading mesh texture through trimesh | No |
| FFmpeg | 4.4.2 | Frame extraction and contact sheets | No |
| CARI4D reconstruction source | `/code/reconstruction` git commit `7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80` | Inspected rendering/pose conventions and provided baseline provenance | No |
| `/pipeline` CARI4D outputs | Provided episode data | Baseline diagnostics and metadata | No |
| `/weights` checkpoints | Provided, read-only | Not required for the final solver | No |

## Downloaded tools and weights

None. No git repositories, pip packages, or model weights were downloaded. Therefore there are no
download URLs or downloaded-weight SHA-256 values to report.

## Reproduction environment

The final solver can run with only Python, NumPy, and SciPy:

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

The GPU rendering stack listed above was used only for offline experiments and validation and is not
required by `solve.py`.
