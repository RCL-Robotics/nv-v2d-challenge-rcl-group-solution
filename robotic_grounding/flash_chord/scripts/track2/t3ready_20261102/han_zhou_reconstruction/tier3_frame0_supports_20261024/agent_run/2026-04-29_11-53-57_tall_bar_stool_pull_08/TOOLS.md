# Tools used

No external tools, repositories or model weights were downloaded. No network access is needed.
Everything below was already installed in the container.

| tool | version | used by solve.py | purpose |
|---|---|---|---|
| Python | 3.11.10 | yes | |
| torch (CUDA) | 2.5.1+cu124 | yes | optimisation (Adam), GPU tensors |
| nvdiffrast | 0.4.0 | yes | differentiable rasterisation + antialiasing of the textured object mesh (silhouette render-and-compare) |
| numpy | 1.26.3 | yes | |
| scipy | 1.16.3 | yes | rotations (`scipy.spatial.transform`) |
| opencv-contrib-python | 4.11.0.86 | yes | video decoding, resizing, blur, dilation |
| trimesh | 5.1.1 | yes | loading `object_mesh.glb` |
| h5py | 3.16.0 | yes | reading `masks.h5` |

Inputs that `solve.py` reads: only `/work/inputs` (video, masks, mesh, camera, ground plane, reconstruction).
It does not use `/pipeline`, `/weights` or anything under `/work/tools` (which stays empty).
I did not try any other pretrained models (FoundationPose re-registration, flow, point trackers) in the final method;
I looked at the CARI4D outputs in `/pipeline` only for context (`wild_export.json`, `pipeline_report.json`).
