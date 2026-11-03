# Tools used

No external repositories or model weights were downloaded. Everything used is
preinstalled in the provided container. The solver needs the following Python
packages:

| Tool/package | Version | Used by `solve.py`? | Purpose |
|---|---:|---|---|
| Python | 3.11.10 | Yes | Runtime |
| NumPy | 1.26.3 | Yes | Arrays, SE(3), pose trajectory |
| SciPy | 1.16.3 | Yes | Rotation interpolation and Savitzky-Golay filtering (`scipy.spatial.transform`, `scipy.signal`) |
| OpenCV (`opencv-python`) | 4.11.0.86 | Yes | Mask resizing, morphology, distance transforms |
| trimesh | 5.1.1 | Yes | Loading `object_mesh.glb` |
| h5py | 3.16.0 | Yes | Reading object/person masks |
| PyTorch | 2.5.1+cu124 | Yes | Tensor operations and nvdiffrast integration |
| nvdiffrast | 0.4.0 | Yes | Differentiable mask rasterization |
| Pillow | 10.2.0 | Indirectly | GLB texture/image loading through trimesh |
| Matplotlib | 3.11.2 | No (diagnostic only) | Trajectory plots during development |
| ffprobe (FFmpeg) | 4.4.2 | No (diagnostic only) | Video metadata inspection |

No pip installs, git clones, or downloaded weights were used. The provided
CARI4D `/pipeline` outputs were consulted for diagnostic depth validation only;
the final solver does not read them. No network requests are made at solve time.
