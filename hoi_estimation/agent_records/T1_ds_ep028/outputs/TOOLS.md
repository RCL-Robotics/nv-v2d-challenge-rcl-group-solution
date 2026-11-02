# Tools Used

No external repositories, model weights, or network downloads were needed for the final solution.

## Runtime environment

| Tool | Version | Used by `solve.py` | Purpose |
|---|---|---:|---|
| Python | 3.11.10 | yes | Runtime |
| NumPy | 1.26.3 | yes | Arrays, linear algebra |
| SciPy | 1.16.3 | yes | Distance transforms, SO(3) utilities |
| OpenCV (`opencv-python`) | 4.11.0 | yes | Video decoding and image operations |
| h5py | 3.16.0 | yes | Reading supplied mask archive |
| PyTorch | 2.5.1+cu124 | yes | GPU rendering and tensor operations |
| trimesh | 5.1.1 | yes | Loading the supplied GLB mesh and UV texture |
| nvdiffrast | 0.4.0 | yes | Rasterized silhouette/texture evidence |
| CUDA | 12.4 | yes | GPU backend |
| FFmpeg | 4.4.2 | no | Frame extraction only during exploratory inspection |

GPU used for the final run: NVIDIA RTX 6000 Ada Generation.

## Existing data/tools consumed

- Supplied inputs in `/work/inputs`: `video.mp4`, `object_mesh.glb`, `camera.json`, `episode.json`, `reconstruction.npz`, and `masks.h5`.
- `/pipeline` was inspected during diagnosis, but the final `solve.py` does not read it. The released `T_world_mesh`, FoundationPose, and CoCoNet summaries were already packaged in `reconstruction.npz`.
- `/weights` was not needed by the final solution.

## Git repositories / downloaded weights

None. No files were downloaded from the internet, and no challenge ground truth or challenge annotation was accessed.
