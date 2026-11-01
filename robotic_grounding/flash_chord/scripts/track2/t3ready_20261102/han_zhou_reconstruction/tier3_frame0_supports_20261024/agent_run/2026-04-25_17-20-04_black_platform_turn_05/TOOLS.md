# Tools used

Nothing was downloaded: no pip packages, git repositories or model weights were added, and `/work/tools` is empty.
Everything below was already installed in the container (Python 3.11.10).

| tool | version | used for | needed by `solve.py` |
|---|---|---|---|
| torch (CUDA) | 2.5.1+cu124 | Adam pose optimisation on the GPU | yes |
| nvdiffrast | 0.4.0 | differentiable rasterisation of the textured mesh (silhouette, antialias); texture rendering during exploration | yes |
| opencv-python (cv2) | 4.11.0 | video decoding, morphology, crops | yes |
| trimesh | 5.1.1 | mesh loading, surface sampling, convex hull | yes |
| open3d | 0.20.0 | ICP to find the mesh's approximate 180° self-symmetry about its thin axis | yes |
| scipy | 1.16.3 | Gaussian temporal smoothing, quaternions, KD-tree (hand-contact cue) | yes |
| h5py | 3.16.0 | reading `masks.h5` | yes |
| numpy | 1.26.3 | everything | yes |

Not used: the CARI4D modules under `/workspace` and `/code`, the weights in `/weights` (FoundationPose, SAM2, MoGe-2,
CoCoNet, ...), the `/pipeline` intermediates, and any network resource. `solve.py` reads only `/work/inputs`.
