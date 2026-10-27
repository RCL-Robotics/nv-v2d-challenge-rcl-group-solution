# Tools used

Nothing was downloaded: no pip installs, no git clones, no new weights. Everything below was already in the
container (`/opt/conda`, `/workspace`, `/weights`, `/pipeline`). No network access is needed at run time.

## Python packages (preinstalled, `/opt/conda/bin/python`)
| package | version | used for | needed by `solve.py` |
|---|---|---|---|
| torch | 2.5.1+cu124 (CUDA 12.4) | optimisation, SAM2 | yes |
| nvdiffrast | 0.4.0 | differentiable rendering of the textured mesh (silhouette + colour) | yes |
| SAM-2 (`sam2` package) | 1.0 | video object segmentation of the iron | yes |
| hydra-core | 1.3.7 | SAM2 config loading | yes |
| opencv-python | 4.11.0.86 | video I/O, image ops | yes |
| trimesh | 5.1.1 | mesh loading, convex hull (rest facets) | yes |
| open3d | 0.20.0 | quadric decimation of the mesh for fast silhouette rendering | yes |
| scipy | 1.16.3 | least squares (table geometry), filters, rotations/slerp | yes |
| numpy | 1.26.3 | everything | yes |
| h5py | 3.16.0 | masks.h5, MoGe depth file | yes |

## Container modules (`/workspace`, editable installs; source also in `/code/reconstruction`)
- `v2d.sam2.lib.sam2_utils.build_sam2_video_predictor_low_mem` (module `v2d_sam2`) is the low-memory SAM2 video
  predictor wrapper. `solve.py` uses it.
  `/code/reconstruction` is not a git checkout, so it has no commit hash; the pipeline report lists the CARI4D
  repository commit as 1f7287ac6fd5f72c30ce2222fb345a3e7d779fc9. The `v2d_foundation_pose` and `v2d_cari4d`
  modules were inspected but not run.

## Weights (read-only, `/weights`)
- `/weights/sam2/sam2.1_hiera_large.pt`
  (sha256 `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`), with the packaged config
  `configs/sam2.1/sam2.1_hiera_l.yaml` from the `sam2` package. Used by `solve.py`.
  The original public source is the SAM 2.1 release (https://github.com/facebookresearch/sam2), but this file was
  not downloaded by me.

## Pipeline intermediates (read-only, `/pipeline`)
- `/pipeline/depth/moge2/aligned.h5` contains the MoGe-2 depth maps that CARI4D aligned to the reconstructed
  human. `solve.py` reads them for one of the three depth cues (table-plane depth).

## Tried and not used
- The released `masks.h5` `object` masks and the pipeline masks: they segment the tripod behind the table, not the
  iron. Only the `person` mask from `masks.h5` is used, to mark occluded pixels.
- FoundationPose (`/weights/foundationpose`) and CoCoNet: inspected but not re-run. Render-and-compare with the
  physical constraints was enough, and FoundationPose would also have needed a depth map at the correct scale.
