# Tools and provenance

No external repository, package, or model weight was downloaded during this task. The final solver is self-contained and uses only software already present in the container plus the supplied inputs.

## Required by `solve.py`

| Tool | Version / revision | Use | Provenance |
|---|---|---|---|
| Python | 3.11 | Runtime | Container |
| NumPy | 1.26.3 | Array math, NPZ I/O | Installed |
| SciPy | 1.16.3 | Rotation averaging, Powell optimization, Savitzky-Golay filter | Installed |
| h5py | 3.16.0 | Read SAM2 masks | Installed |
| trimesh | 5.1.1 | Load exact GLB mesh | Installed |
| PyTorch | 2.5.1+cu124 | GPU tensor/mesh projection | Installed |
| nvdiffrast | 0.4.0, commit `253ac4fcea7de5f396371124af597e6cc957bfae` | Differentiable/rasterized mesh silhouettes | `/workspace/third_party/nvdiffrast` |
| CUDA runtime / one GPU | Container-provided | Projection and rasterization | Container |

`/usr/bin/time` was not available, so runtime was measured with the shell `date` command.

## Existing reconstruction software and intermediate outputs

The supplied baseline and camera-frame proposals were produced by the container's CARI4D/FoundationPose stack. No such model was rerun for the final solution.

| Tool | Revision / identifier | Role |
|---|---|---|
| CARI4D reconstruction repository | `7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80` | Source of supplied CARI4D pipeline behavior/intermediates |
| CARI4D model repository recorded by the pipeline report | `1f7287ac6fd5f72c30ce2222fb345a3e7d779fc9` | Recorded provenance of the released CoCoNet output |
| FoundationPose | Installed under `/workspace/v2d_foundation_pose`; no Git metadata in the installed copy | Produced the supplied `T_cam_mesh_foundationpose` proposal |
| MoGe-2 | `Ruicheng/moge-2-vitl-normal`, revision `b135031bae30b5ac2ae141a0e68717795ce38340` (from supplied metadata) | Supplied depth/intrinsics, not rerun |
| SAM2 | Supplied masks | Object/person masks, not rerun |

No new model weights were downloaded. No URL or weight SHA is therefore attributable to this solution beyond the supplied pipeline metadata. The exact object mesh in `/work/inputs/object_mesh.glb` is supplied by the task.

## Tried for diagnostics only, not needed by `solve.py`

| Tool | Version | Use |
|---|---|---|
| OpenCV | 4.11.0.86 | Video frame extraction, mask inspection, diagnostic overlays |
| Pillow | 10.2.0 | Diagnostic contact sheets and mesh texture inspection |
| Open3D | 0.20.0 | Exploratory table/floor point-cloud plane fitting; not adopted |
| pyrender | 0.1.45 | Exploratory canonical mesh views; not used by final method |
| Kaolin | revision `06ffb7d955ca26b608c60a9e862327c56b226921` | Present in container and inspected as an available renderer; not used |

## Reproducible command

```bash
python /work/outputs/solve.py \
  --inputs /work/inputs \
  --out /work/outputs/object_poses_world.npz
```

The final run took 216 seconds and used one GPU. `solve.py` does not access the network and does not read `/work/scratch` or precomputed per-frame correction tables.
