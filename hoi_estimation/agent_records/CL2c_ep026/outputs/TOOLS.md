# Tools used

No external tools, repositories, model weights or pip packages were downloaded or installed. Everything ran on
software that was already in the container. Nothing was put under `/work/tools` or the pip user base.

| tool | version | where it comes from | used by `solve.py` | what it was used for |
|---|---|---|---|---|
| Python | 3.11.10 | container | yes | – |
| PyTorch (CUDA 12.4) | 2.5.1+cu124 | container | yes | GPU tensors for the render-and-compare scorer |
| nvdiffrast | 0.4.0 | container | yes | rasterising the textured object mesh (batched, cropped) |
| OpenCV (opencv-python / contrib) | 4.11.0 | container | yes | reading `video.mp4`; diagnostic image ops |
| trimesh | 5.1.1 | container | yes | loading `object_mesh.glb`, bounds; (dev only) proximity queries for the hand-contact cue |
| h5py | 3.16.0 | container | yes | reading `masks.h5` |
| NumPy | 1.26.3 | container | yes | – |
| SciPy | 1.16.3 | container | yes | sparse least-squares trajectory smoother |
| Pillow | 10.2.0 | container | yes (through trimesh) | texture image |

These were not used: `/weights` (no network was re-run), `/code`, and `/pipeline`. Only the files in `/work/inputs` were
read. No ground truth or dataset copy was accessed.

Development-only scripts (diagnostics, cue analysis, validation) are in `/work/scratch/an/`. `solve.py` does not
need them.
