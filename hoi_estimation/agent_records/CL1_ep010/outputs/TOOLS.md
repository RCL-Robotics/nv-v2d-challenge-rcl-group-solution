# Tools used

No external tools, repositories, model weights or pip packages were downloaded or installed. Everything ran on
software that was already in the container. `/work/tools` is empty.

| tool | version | used for | needed by `solve.py` |
|---|---|---|---|
| Python packages already in the container: torch | 2.5.1+cu124 | GPU tensors, Adam optimisation of the poses | yes |
| nvdiffrast | 0.4.0 | differentiable silhouette / depth / texture rendering of the object mesh | yes |
| opencv-python (cv2) | 4.11.0 | video decoding, mask resizing, MoGe PNG depth decoding, KLT / template matching (validation) | yes |
| trimesh | 5.1.1 | loading `object_mesh.glb`, bounding-box corners, convex hull (analysis) | yes |
| h5py | 3.16.0 | reading `masks.h5` and `/pipeline/depth/moge2/aligned.h5` | yes |
| numpy | 1.26.3 | everything | yes |
| scipy | 1.16.3 | rotation-vector printing in scratch analysis only | no |

Data read by `solve.py`: `/work/inputs/*` (video, masks, mesh, camera, ground plane, frame-0 reference human,
reconstruction) and `/pipeline/depth/moge2/aligned.h5` (the MoGe-2 depth that was already computed; it is only
used as a weak depth cue in the static phases). No network access is needed.

Code: `/work/outputs/solve.py` (main), `/work/outputs/hoi_common.py` (renderer, mask/loss helpers, KLT helpers),
`/work/outputs/validate.py` (ground-truth-free validation table; it is not needed to produce the result).
