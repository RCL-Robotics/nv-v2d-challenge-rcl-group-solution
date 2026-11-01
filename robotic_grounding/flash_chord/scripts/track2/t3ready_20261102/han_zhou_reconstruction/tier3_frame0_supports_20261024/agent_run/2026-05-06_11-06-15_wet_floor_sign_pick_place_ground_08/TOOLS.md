# Tools used

No external tools, repositories, pip packages or model weights were downloaded. No network access was used.
Everything ran on the container's preinstalled software.

| tool | version | used by solve.py | purpose |
|---|---|---|---|
| Python | 3.11.10 | yes | |
| numpy | 1.26.3 | yes | |
| scipy | 1.16.3 | analysis/validation only | rotations (`scipy.spatial.transform`) |
| torch (CUDA 12.4) | 2.5.1+cu124 | yes | optimisation (Adam) for refinement and smoothing |
| nvdiffrast | 0.4.0 | yes | differentiable rasterisation of the textured mesh |
| opencv-python | 4.11.0 | yes | video decoding, crop warping |
| trimesh | 5.1.1 | yes | mesh loading |
| h5py | 3.16.0 | yes | SAM2 masks |

Inputs read by `solve.py`: `/work/inputs/*` only (video, masks, mesh, camera, reconstruction). It does not use
`/pipeline`, `/weights` or `/code`.

Considered but not used: re-running FoundationPose or MoGe from `/weights`, and point trackers or optical flow
such as CoTracker or RAFT. The baseline tracking was already pixel-accurate in visible frames, and the remaining
errors (small yaw biases, occlusion jumps, static-phase jitter) were handled by render-and-compare plus temporal
modelling.
