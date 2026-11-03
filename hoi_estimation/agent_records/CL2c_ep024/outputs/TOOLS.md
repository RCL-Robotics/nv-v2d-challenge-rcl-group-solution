# Tools used

Nothing was downloaded: no pip installs, no git clones, no extra model weights. `/work/tools` is empty.
Everything below was already installed in the container.

| tool | version | used for | needed by `solve.py` |
|---|---|---|---|
| Python | 3.11.10 | everything | yes |
| PyTorch | 2.5.1+cu124 (CUDA 12.4) | differentiable pose optimisation on the GPU | yes |
| nvdiffrast | 0.4.0 | differentiable rasterisation and anti-aliasing of the object mesh (silhouettes); textured renders for visual checks | yes |
| open3d | 0.20.0 | quadric decimation of the mesh to 20k faces for silhouette rendering | yes |
| trimesh | 5.1.1 | loading `object_mesh.glb` | yes |
| numpy | 1.26.3 | everything | yes |
| scipy | 1.16.3 | rotation averaging (`Rotation.mean`), KD-tree in the validation scripts | yes |
| h5py | 3.16.0 | reading `masks.h5` | yes |
| OpenCV (cv2) | 4.11.0 | frame crops, HSV colour masks, contact sheets and overlays (analysis and validation only) | no |
| ffmpeg | 4.4.2 (Ubuntu) | extracting video frames for visual inspection (analysis only) | no |

Data and models: only the provided inputs (`/work/inputs`) are read: the SAM2 masks, the mesh, the camera,
the baseline reconstruction and the floor plane. The frame-0 reference human and the video were used for
analysis and validation only. Nothing from `/weights` or `/pipeline` was used: no network such as
FoundationPose, SAM2 or MoGe was re-run. HuggingFace and the challenge dataset were not accessed.
