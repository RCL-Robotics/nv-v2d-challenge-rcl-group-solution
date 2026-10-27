# Tools used

No packages, repositories or weights were downloaded. Everything came from the container's software or the
read-only `/weights`, `/pipeline` and `/code` mounts. No network access was used.

## Needed by `solve.py`
| tool | version | role |
|---|---|---|
| Python | 3.11.10 | |
| numpy | 1.26.3 | |
| scipy | 1.16.3 | rotation slerp / rotvec |
| torch (CUDA) | 2.5.1+cu124 | batched optimisation on GPU |
| nvdiffrast | 0.4.0 (container `/workspace/third_party/nvdiffrast`) | differentiable rasterisation of the textured mesh |
| opencv-python (cv2) | 4.11.0 | video decoding, mask morphology |
| trimesh | 5.1.1 | mesh/texture loading |
| h5py | 3.16.0 | SAM2 masks (`inputs/masks.h5`) |

Data read by `solve.py`: only `/work/inputs/*`. It does not read `/pipeline`, `/weights` or `/code`.

## Tried during development, NOT used by `solve.py`
| tool | version / file | what for | outcome |
|---|---|---|---|
| SAM2 (`sam2` package, SAM-2 1.0, installed in container) | `/weights/sam2/sam2.1_hiera_large.pt`, sha256 `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318` | re-segment the mug on 4x upsampled crops with a box prompt, hoping to recover the dark handle the released masks miss | handle recovered only on some frames, plus background leaks; dropped |
| MoGe-2 aligned depth | `/pipeline/depth/moge2/aligned.h5` (pre-computed pipeline output; no model run) | object depth cue | noisy on the small reflective mug (disagrees with the baseline by up to ±13 cm in both directions per phase); used only as a reported cross-check |
