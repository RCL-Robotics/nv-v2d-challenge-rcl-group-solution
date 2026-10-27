# Tools

## Used by the final solver

`/work/outputs/solve.py` needs the following installed packages. No downloaded repositories or model weights are required.

| Tool | Version | Use |
|---|---:|---|
| Python | 3.11.10 | Runtime |
| NumPy | 1.26.3 | Array math, SE(3) trajectory composition |
| SciPy | 1.16.3 | Rotation interpolation, differential evolution, robust filters |
| PyTorch | 2.5.1+cu124 | GPU rasterization/back-end tensors |
| nvdiffrast | 0.4.0 | Differentiable textured-mesh rasterization and texture sampling |
| trimesh | 5.1.1 | GLB loading and mesh geometry/texture access |
| OpenCV | 4.11.0 | Video decoding, resizing, Sobel gradients |
| h5py | 3.16.0 | Reading object/person masks |
| Pillow | 10.2.0 | GLB texture decoding through trimesh |

GPU: NVIDIA RTX 6000 Ada Generation. Peak process GPU memory stayed well below the 9 GB limit.

## Availability probes only

These were checked as possible alternatives but were not used to produce the result:

- pyrender 0.1.45
- Open3D 0.20.0
- PyTorch3D 0.7.9
- Kornia 0.8.3

## External code, repositories, and weights

- No Git repositories were cloned.
- No model weights were downloaded.
- No network resources were used by the final solver.
- The solver does not use `/pipeline`, `/weights`, or `/code`; it reads only `/work/inputs` and outputs the requested trajectory.
