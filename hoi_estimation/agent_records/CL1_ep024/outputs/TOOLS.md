# Tools

No external tools were downloaded: no pip installs, git clones or downloaded weights; `/work/tools` is unused. Everything
below was already installed in the container.

| tool | version | used for | needed by `solve.py` |
|---|---|---|---|
| Python | 3.11.10 | – | yes |
| PyTorch (CUDA 12.4) | 2.5.1+cu124 | differentiable pose optimisation (Adam) | yes |
| nvdiffrast | 0.4.0 | rasterisation and antialiased soft silhouettes | yes |
| numpy | 1.26.3 | – | yes |
| scipy | 1.16.3 | rotation averaging, brentq (floor contact), sparse robust depth smoothing | yes |
| OpenCV | 4.11.0 | video decoding, Lab colour conversion, morphology | yes |
| trimesh | 5.1.1 | mesh loading, closest-point queries during experiments | yes |
| open3d | 0.20.0 | quadric decimation of the mesh to 8k faces | yes |
| h5py | 3.16.0 | reading `masks.h5` | yes |

Data used by `solve.py`: only the files in `/work/inputs`. These are the video, the SAM2 masks (crop centring and the
person mask), the reconstruction (baseline poses, hand keypoints), the camera, the ground plane and the mesh.

Used during analysis only, not by `solve.py`:
- `/pipeline/depth/moge2/raw.npy` (MoGe-2 depth, as a depth cross-check);
- `/pipeline/depth/moge2/aligned.h5`;
- `frame0_reference_human.npz` (to check the camera/human projection and the floor level).

No model weights from `/weights` were run.
