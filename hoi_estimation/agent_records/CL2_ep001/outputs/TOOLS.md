# Tools used

Nothing was downloaded. No pip installs, no git clones, no new weights; `/work/tools` is empty. Everything below was already in the container.

| tool | version / hash | used for | needed by `solve.py` |
|---|---|---|---|
| Python packages: numpy | 1.26.3 | everything | yes |
| torch (CUDA) | 2.5.1+cu124 | optimisation (Adam), GPU tensors | yes |
| nvdiffrast | 0.4.0 | textured, differentiable mesh rendering (silhouette + colour) | yes |
| opencv-python | 4.11.0 | video decoding, mask resizing, crops | yes |
| trimesh | 5.1.1 | mesh / texture loading | yes |
| scipy | 1.16.3 | rotation averaging, filters, distance transforms | yes |
| h5py | 3.16.0 | reading `masks.h5` | yes |
| sam2 (container install, `/opt/conda/lib/python3.11/site-packages/sam2`) | package exposes no version | one box-prompted image segmentation of the static bar stool (an occluder) on the first and last frame | yes (if import fails, solve.py continues without the stool mask) |
| SAM2.1 hiera-large weights | `/weights/sam2/sam2.1_hiera_large.pt`, sha256 `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318` | as above | yes |

Pipeline data that was read but not run: `/pipeline/depth/moge2` was inspected, but MoGe depth is not used in the final method. FoundationPose and CoCoNet were not re-run; their saved poses in `reconstruction.npz` serve as hypotheses.

Own code, all in `/work/outputs`:
- `solve.py`: the full pipeline.
- `lib.py`, `fit.py`, `traj.py`: helpers for rendering, fitting and the trajectory optimisation.
- `validate.py`: builds the validation table.
- `depth_cues.py`: reports the depth cues.
