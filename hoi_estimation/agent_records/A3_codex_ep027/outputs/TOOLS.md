# Tools and dependencies

All tools below were already installed in the supplied container. No pip installs, git clones, repository downloads, model downloads, or network requests were performed. Consequently there are no downloaded repository commit hashes, weight URLs or weight SHA256 values to report. No model weights are required by `solve.py`.

## Direct tools used

| Tool/package | Version | Use | Required by solve.py |
|---|---|---|---|
| Python | 3.11.10, conda-forge, GCC 13.3.0 | Scripts and standard library | Yes |
| NumPy | 1.26.3 | Pose arrays, coordinate transforms, output | Yes |
| opencv-python | 4.11.0.86 | Distance transforms, mask warping; video inspection | Yes |
| trimesh | 5.1.1 | Load exact GLB mesh and texture | Yes |
| SciPy | 1.16.3 | Powell optimization, temporal filters; exploratory rotation analysis | Yes |
| h5py | 3.16.0 | Read supplied masks | Yes |
| pyrender | 0.1.45 | EGL perspective mesh rasterization | Yes |
| PyOpenGL | 3.1.0 | EGL/OpenGL interface used by pyrender | Yes, transitively |
| Pillow | 10.2.0 | Inspection montages and evidence image; mesh texture loading | Yes, transitively |
| torch | 2.5.1+cu124 | Imported once to check CUDA availability; no inference | No |
| Bash and core command-line utilities | Container-provided | File inspection and launching scripts | No |
| ripgrep | Container-provided | Search for workspace instructions | No |
| nvidia-smi | Driver 580.178.04 | Inspect GPU model/driver | No |

Rendering used one NVIDIA RTX 6000 Ada Generation through EGL. No CARI4D, FoundationPose, MoGe, SAM, or other model was rerun. The supplied reconstruction and segmentation masks are input data, not newly fetched tools.

## Renderer support packages

Preinstalled support dependencies used by the rendering/mesh stack: `pyglet 2.1.16`, `imageio 2.38.0`, `networkx 3.2.1`, `freetype-py 2.5.1`, and `six 1.16.0`. The environment also contains `setuptools 72.1.0`. No package installation or version change is necessary in this container.

`nvdiffrast 0.4.0` was considered and its installed version queried; it was not imported or used. No other rasterizer, pose estimator, downloaded repository, or external model was tried.

## Script requirements

`solve.py` requires NumPy, OpenCV, trimesh, SciPy, h5py, pyrender and its installed EGL/OpenGL/texture dependencies. It reads only four input files: `reconstruction.npz`, `camera.json`, `object_mesh.glb`, and `masks.h5`. It needs no `/work/tools` assets, pip user packages, exploratory outputs, scratch files, video, pipeline checkpoints or external services.

The supplementary `make_evidence.py` additionally reads `video.mp4` and uses Pillow. The other Python scripts in this directory document exploratory checks and are not needed to reproduce the final poses.
