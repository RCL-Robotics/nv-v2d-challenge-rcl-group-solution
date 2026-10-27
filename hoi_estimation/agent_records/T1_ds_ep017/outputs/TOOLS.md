# Tools and dependencies

## Used by the final solver

All packages were already installed in the container. No Python package was installed from the network for the final solver, and no external repository or model weight was downloaded.

| Tool | Version | Role | Needed by `solve.py` |
|---|---:|---|---|
| Python | 3.11 | Runtime | Yes |
| NumPy | 1.26.3 | Arrays, SE(3), trajectory logic | Yes |
| SciPy | 1.16.3 | Powell silhouette fit, SLERP, Savitzky-Golay filtering | Yes |
| OpenCV (`opencv-python`) | 4.11.0 | Video decoding and validation image processing | Yes |
| PyTorch | 2.5.1+cu124 | GPU tensors and rasterizer interface | Yes |
| nvdiffrast | 0.4.0 | Differentiable-compatible mesh rasterization and texture sampling | Yes |
| trimesh | 5.1.1 | GLB scene/mesh and UV texture loading | Yes |
| h5py | 3.16.0 | Reading supplied mask HDF5 | Yes |
| CUDA / NVIDIA driver | CUDA 12.4 runtime, driver 580.178.04 | GPU execution | Yes at runtime |

## Supplied data/tools inspected during analysis

These were used as diagnostic evidence, but the final solver does **not** require any path outside `/work/inputs`:

- `/pipeline/foundationpose/episode_000017.pkl`: FoundationPose tracking diagnostics.
- `/pipeline/inference/coconet.pth` and `/pipeline/inference/refined.pth`: CoCoNet/refinement poses and diagnostics.
- `/pipeline/depth/moge2/aligned.h5`: aligned MoGe-2 metric depth, used to check that a silhouette-only depth correction would be unjustified.
- `/weights`: read-only model weights; no new model inference was run for the final solution.
- `/code/reconstruction`: read-only CARI4D source, inspected for depth decoding conventions.

## Downloaded tools, repositories, and weights

None. No git clone, model download, challenge dataset, annotation, parquet trajectory, or mirror was used.

## Reproduction

```bash
cd /work
python /work/outputs/solve.py \
  --inputs /work/inputs \
  --out /work/outputs/object_poses_world.npz
```

The solver fixes Python, NumPy, PyTorch, and CUDA random seeds and is deterministic on the supplied inputs. It writes `validation.json` next to the requested output, containing the exact validation table used in `REPORT.md`.
