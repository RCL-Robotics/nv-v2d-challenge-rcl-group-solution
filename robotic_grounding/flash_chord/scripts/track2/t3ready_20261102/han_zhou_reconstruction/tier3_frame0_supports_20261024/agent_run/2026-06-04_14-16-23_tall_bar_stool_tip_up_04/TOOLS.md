# Tools used

No external tools, repositories, model weights or pip packages were downloaded or installed. Everything ran
with the software already in the container (`/work/tools` and the pip user base are unused/empty). No
network access is needed by `solve.py`. Neither `/pipeline`, `/weights` nor `/code` is read by `solve.py`: it
uses only `/work/inputs`.

| package (pre-installed in the container) | version | used by | purpose |
|---|---|---|---|
| torch | 2.5.1+cu124 (CUDA 12.4) | solve.py, validate.py | differentiable pose optimisation (Adam), GPU |
| nvdiffrast | 0.4.0 | solve.py, validate.py | rasterising the mesh (silhouette, depth, antialiased silhouette gradients) |
| numpy | 1.26.3 | all | |
| scipy | 1.16.3 | solve.py, validate.py, depth_cues.py | rotations, gaussian_filter1d, brentq, cKDTree |
| opencv-python / opencv-contrib-python | 4.11.0.86 | solve.py, validate.py | Canny edges, distance transforms, morphology |
| trimesh | 5.1.1 | all | loading the textured GLB mesh, surface sampling |
| h5py | 3.16.0 | solve.py, validate.py | reading `masks.h5` |
| decord | 0.6.0 | solve.py, validate.py | decoding `video.mp4` |
| matplotlib | 3.11.2 | development only (plots in /work/scratch) | visual diagnostics, not needed by solve.py |

Considered but not used: point trackers / optical flow (CoTracker, RAFT), FoundationPose re-registration,
MoGe depth on the object. For this video the baseline had no 180-degree flips or tracking losses, so the
work went into render-and-compare refinement and temporal modelling. Re-estimating depth was ruled out on
purpose (see REPORT.md).

Files `solve.py` needs: `/work/outputs/solve.py` and `/work/outputs/hoilib.py` (same directory).
`validate.py` and `depth_cues.py` only produce the report tables.
