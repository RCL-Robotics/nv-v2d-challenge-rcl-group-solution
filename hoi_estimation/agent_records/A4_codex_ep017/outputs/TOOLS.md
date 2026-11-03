# Tools and dependencies

No packages, repositories, or weights were downloaded or installed. No network requests, external annotations, challenge datasets, or ground-truth trajectories were used. All Python packages below were already installed in the container.

| Package | Version | Use | Needed by solve.py |
|---|---|---|---|
| Python | 3.11.10 | Interpreter and standard library | Yes |
| NumPy | 1.26.3 | Geometry, arrays, NPZ input/output | Yes |
| SciPy | 1.16.3 | Rotations, Gaussian filtering, sparse trajectory least squares | Yes |
| OpenCV (opencv-python) | 4.11.0.86 | Occluder mask dilation; video inspection and exploratory convex-hull rasterization | Yes |
| trimesh | 5.1.1 | Exact supplied GLB loading; exploratory convex hull | Yes |
| h5py | 3.16.0 | Supplied segmentation masks | Yes |
| PyTorch | 2.5.1+cu124 | GPU mesh projection, bounded silhouette optimization | Yes |
| nvdiffrast | 0.4.0 | Full supplied mesh rasterization and silhouette derivatives | Yes |
| Pillow | 10.2.0 | Diagnostic video/render contact sheets; GLB texture loading | Indirectly, through trimesh GLB loading |
| matplotlib | 3.11.2 | Exploratory rotation plot | No |

System tools used: bash, cat, ls, rg, head, tail, ps, nvidia-smi. GPU driver: 580.178.04. The environment exposes one selected NVIDIA RTX 6000 Ada GPU to the Python process through CUDA_VISIBLE_DEVICES=0. No second GPU is used.

Package metadata was additionally queried for pyrender 0.1.45 and open3d 0.20.0; neither was imported or used for estimation. They are not required.

Git repositories cloned/tried: none. Downloaded weights and their URLs/SHA256: none. Existing model weights used: none. Pipeline neural models invoked: none. solve.py does not depend on /pipeline, /weights, /code, /work/tools, diagnostic scripts, video contact sheets, or any scratch files. It reads only the supplied reconstruction, camera, exact mesh, and masks, and writes its results beside the requested output.

Reproduce offline:

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

CUDA and the preinstalled nvdiffrast rasterizer are required. NumPy and Torch seeds are fixed to zero, CPU Torch uses one thread, cuDNN benchmarking is disabled, and a fixed cuBLAS workspace configuration is selected. No stochastic model inference or sampling is used.
