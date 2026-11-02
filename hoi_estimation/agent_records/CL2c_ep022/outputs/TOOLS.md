# Tools used

No external tools, repositories, pip packages or model weights were downloaded. Nothing was put in `/work/tools`.
No network access was used. No pretrained network weights (from `/weights` or elsewhere) are used: the SAM2 masks,
the human reconstruction and the baseline poses come from the provided inputs.

## Container software used (pre-installed)
| package | version | used by `solve.py` | purpose |
|---|---|---|---|
| torch | 2.5.1+cu124 | yes | optimisation (Adam), GPU tensors |
| nvdiffrast | 0.4.0 | yes | differentiable rasterisation of the textured mesh |
| opencv-python (cv2) | 4.11.0 | yes | video decoding, Canny edges, distance transforms, crop warping |
| numpy | 1.26.3 | yes | |
| scipy | 1.16.3 | yes | rotations (`scipy.spatial.transform`), median filter |
| h5py | 3.16.0 | yes | reading `masks.h5` |
| trimesh | 5.1.1 | yes | loading `object_mesh.glb`, OBB axes (exploration) |
| ffmpeg | 4.4.2 (system) | no | frame extraction for visual inspection during exploration only |

## Project files needed by `solve.py`
`/work/outputs/render_utils.py`, `/work/outputs/silfit.py`, `/work/outputs/traj_utils.py`
(`edgescore.py` and `validate.py` are only used for validation).
