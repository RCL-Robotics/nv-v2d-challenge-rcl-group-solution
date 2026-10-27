# Tools and dependencies

No new model weights or repositories were downloaded for the final solution. No network access is used by `solve.py`.

## Required by `solve.py`

| Tool | Version | Use |
|---|---:|---|
| Python | 3.11 | runtime |
| NumPy | 1.26.3 | arrays, linear algebra |
| SciPy | 1.16.3 | rotations, signal filtering |
| PyTorch | 2.5.1+cu124 | CUDA tensors and rasterization interface |
| nvdiffrast | 0.4.0 | exact mesh mask and textured crop rendering |
| OpenCV (`opencv-python`) | 4.11.0 | video I/O, masks, image filtering |
| trimesh | 5.1.1 | GLB mesh loading and vertices/faces/UVs |
| h5py | 3.16.0 | object/person masks |

These packages were preinstalled in the container. `solve.py` imports nvdiffrast directly and does not depend on `/code/reconstruction` or `/workspace`.

## Existing local pipeline used for evidence/validation

| Component | Revision / location | Use in this task |
|---|---|---|
| CARI4D reconstruction repository | git `7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80`, `/code/reconstruction` | Source inspection and exploration of the released CARI4D intermediate formats; not required by `solve.py` |
| CARI4D installed modules | `/workspace/v2d_cari4d` | Utility renderer used only during exploratory comparisons; replaced by self-contained rendering in `solve.py` |
| MoGe-2 aligned depth | `/pipeline/depth/moge2/aligned.h5` | Independent validation check only; not read by `solve.py` |
| Released reconstruction | `/work/inputs/reconstruction.npz` | Baseline poses, human contact probabilities |
| SAM2 masks | `/work/inputs/masks.h5` | Object/person masks |
| Object mesh | `/work/inputs/object_mesh.glb`, SHA-256 `ebc656d5de2c6fb0075acd136503ac936b6c821fbfe05b21286efbc9af7225d6` | Exact textured mesh for all renders |

## Downloaded weights

None.

## Output hash

`object_poses_world.npz`: SHA-256 `d58955d35116f07063fece76f6ce4471808c2ca36b72d548f760e32dd38d0270`.
