# Tools used

No packages, repositories or weights were downloaded. Everything came from the container. `/work/tools` is empty.

| tool | version / source | used for | needed by solve.py |
|---|---|---|---|
| Python | container (/opt/conda, 3.11) | everything | yes |
| torch | 2.5.1+cu124 (container) | GPU tensors, batched linear algebra | yes |
| nvdiffrast | 0.4.0 (container, /workspace/third_party/nvdiffrast) | rasterisation: silhouette coverage, texture unwrapping, textured renders | yes |
| numpy | 1.26.3 | | yes |
| scipy | 1.16.3 | smoothing filters (gaussian_filter1d, median_filter) | yes |
| opencv-python | 4.11.0 | video decoding, image resampling | yes |
| trimesh | 5.1.1 | loading the GLB mesh | yes |
| open3d | 0.20.0 | quadric decimation of the mesh (6000 faces) for silhouette rendering | yes |
| h5py | 3.16.0 | reading masks.h5 | yes |
| matplotlib | 3.11.2 | diagnostic plots during development only | no |
| Pillow | 10.2.0 | decoding MoGe depth PNGs in the FoundationPose experiment only | no |
| FoundationPose ScorePredictor | container `/workspace/v2d_foundation_pose/lib/FoundationPose`, weights `/weights/foundationpose/nvlabs_pytorch/2024-01-11-20-02-45/model_best.pth` (read-only, provided) | tried as a yaw-hypothesis scorer. Flat scores, so not used | no |
| MoGe-2 aligned depth | `/pipeline/depth/moge2/aligned.h5` (provided) | only in the FoundationPose experiment | no |

`solve.py` reads only `/work/inputs` (video, masks, mesh, camera, ground plane, reconstruction) and needs no network.
