# Tools used

No external tools, repositories, model weights or pip packages were downloaded or installed.
Nothing was placed in `/work/tools` or the pip user base. Everything below was already in the container.

## Python packages (container, Python 3.11.10)

| package | version | used by `solve.py` | purpose |
|---|---|---|---|
| torch | 2.5.1+cu124 | yes | optimisation (Adam), GPU tensors |
| nvdiffrast | 0.4.0 | yes | rasterisation of the textured mesh (texture alignment) and silhouettes (diagnostic IoU) |
| opencv-python (cv2) | 4.11.0 | yes | video decoding, crops, distance transforms, background mask |
| trimesh | 5.1.1 | yes | mesh / texture loading, surface sampling |
| open3d | 0.20.0 | yes | quadric decimation of the mesh for fast silhouette rendering |
| scipy | 1.16.3 | yes | median / Gaussian filters (depth smoothing) |
| numpy | 1.26.3 | yes | everything |
| h5py | 3.16.0 | yes | reading `masks.h5` |

## Models, weights, repositories

- None used. The CARI4D modules (`/workspace`, `/code/reconstruction`), `/weights` (SAM2, FoundationPose,
  SAM-3D-Body, CoCoNet, GroundingDINO, MoGe-2) and the `/pipeline` intermediates were **not** used by `solve.py`.
- During exploration I read the `T_cam_mesh_foundationpose` / `T_cam_mesh_coconet` fields of
  `inputs/reconstruction.npz`, only to check whether they differ from the baseline in symmetry. They don't: all
  three agree to within a few degrees.

## Files `solve.py` reads

Only `/work/inputs/*`: `video.mp4`, `masks.h5`, `reconstruction.npz` (`T_world_mesh`), `object_mesh.glb`,
`camera.json`. No network, no scratch files. `validate.py` and `depth_cues.py` (diagnostics only) additionally
read `ground_plane.json` and `reconstruction.npz` (human keypoints, contact probabilities).
