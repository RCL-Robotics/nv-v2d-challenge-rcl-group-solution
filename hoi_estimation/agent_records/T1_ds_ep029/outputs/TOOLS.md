# Tools used

All processing is offline. No external repository or model weight was downloaded for this solution.

| tool | version | needed by solve.py | use |
|---|---|---|---|
| Python | 3.11.10 | yes | runtime |
| PyTorch | 2.5.1+cu124 | yes | differentiable rasterisation / Adam |
| nvdiffrast | 0.4.0 | yes | antialiased silhouette rasterisation |
| NumPy | 1.26.3 | yes | array geometry |
| SciPy | 1.16.3 | yes | rotations, smoothing, unwrap |
| trimesh | 5.1.1 | yes | load exact GLB mesh |
| h5py | 3.16.0 | yes | released object masks |
| OpenCV | 4.11.0.86 | yes | mask resizing and image I/O |
| CUDA runtime | 12.4 | yes | GPU rasterisation |

The solution reads the released `/work/inputs` files and the shipped `/work/inputs/reconstruction.npz`; it does not need `/pipeline`, network access, scratch files, or downloaded weights.
