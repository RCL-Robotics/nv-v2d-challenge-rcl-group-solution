# Tools used

No external tools, repositories, model weights or pip packages were downloaded or installed. Everything comes from
the container's existing Python environment (Python 3.11.10). Nothing was written to `/work/tools` or to the pip user base.

| package | version | used for | needed by `solve.py` |
|---|---|---|---|
| torch (CUDA) | 2.5.1+cu124 | batched point-based render-and-compare optimisation (Adam), `grid_sample`, `cdist` | yes |
| nvdiffrast | 0.4.0 | z-buffered rendering of the textured mesh (orange-coverage appearance score, silhouette IoU in `validate.py`, visual checks) | yes |
| opencv-python (cv2) | 4.11.0 | video decoding, HSV colour masks, distance transforms, visualisation | yes |
| trimesh | 5.1.1 | loading `object_mesh.glb`, surface sampling, texture lookup | yes |
| numpy | 1.26.3 | everything | yes |
| scipy | 1.16.3 | rotations (`scipy.spatial.transform`), Gaussian temporal smoothing | yes |
| h5py | 3.16.0 | reading `masks.h5` | yes |

Inputs read by `solve.py`: only `/work/inputs/*` (video, masks, mesh, camera, ground plane, reconstruction.npz, which
holds the baseline poses, hand keypoints and contact probabilities). `/pipeline`, `/weights`, `/code` and the CARI4D
modules were not used: no FoundationPose, SAM2, MoGe or CoCoNet re-runs. The FoundationPose
camera-frame estimates in `reconstruction.npz` were tried as extra hypotheses during development. They added nothing over the
rolled and flipped baseline hypotheses, so the final script does not use them.

Scripts in `/work/outputs`:
- `solve.py`: produces the result. Run it offline as `python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`.
- `validate.py`: the ground-truth-free validation table (imports `solve.py`). It is not needed to produce the result.
