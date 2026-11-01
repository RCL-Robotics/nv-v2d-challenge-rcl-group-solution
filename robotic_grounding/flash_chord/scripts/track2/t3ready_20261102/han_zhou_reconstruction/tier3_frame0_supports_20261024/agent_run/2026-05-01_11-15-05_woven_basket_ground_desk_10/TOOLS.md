# Tools used

No external packages, repositories or weights were downloaded. Nothing was installed under `/work/tools` or the pip user base. Everything below was already in the container.

## Needed by `solve.py` (and `validate.py`)

| tool | version | use |
|---|---|---|
| Python | 3.11.10 | |
| torch | 2.5.1+cu124 | GPU tensors, Adam for silhouette fitting |
| nvdiffrast | 0.4.0 | rasterisation and antialiased (differentiable) silhouettes, textured rendering |
| trimesh | 5.1.1 | loading `object_mesh.glb`, even surface sampling |
| opencv-python (cv2) | 4.11.0 | video decoding, crop resampling, blurring, morphology |
| numpy | 1.26.3 | |
| scipy | 1.16.3 | temporal filters (`gaussian_filter1d`, `median_filter`); `cKDTree` in `validate.py` |
| h5py | 3.16.0 | `masks.h5` |

Our own code: `solve.py`, `hoilib.py`, `validate.py` (all in `/work/outputs`). `solve.py` needs only `/work/inputs` and these files. It does not read `/pipeline` or `/weights`.

## Used only in experiments (not needed by `solve.py`)

**FoundationPose** (pre-installed CARI4D copy, `/workspace/v2d_cari4d/lib/cari4d/estimater.py`, with `learning/training/predict_score.py` and `predict_pose_refine.py`).
* Used to re-run registration (252 rotation hypotheses, refiner and scorer) on 11 floor frames, 12 desk frames and 25 carried frames. The aim was to see which 180° yaw class its RGB-D scorer prefers.
* Depth input: `/pipeline/depth/moge2/aligned.h5`, read with `prep.mhr_depth_h5.read_metric_depth`.
* Weights (read-only, `/weights/foundationpose/nvlabs_pytorch`):

  | weights | sha256 |
  |---|---|
  | scorer, `2024-01-11-20-02-45/model_best.pth` | `81924d384bf5c26c646ee4783104982ae3d1e049c181c36641b6a7aeae494c26` |
  | refiner, `2023-10-28-18-33-37/model_best.pth` | `774700586ddc435d408fc01c9809c43e151232936369dfbea0f0f964ba471d60` |

* Of the fields in `/pipeline/foundationpose/episode_000027.pkl`, only the tracking-state fields were read (registration on frame 0, then tracking). The values of the `gt_*` and `oracle` fields were never inspected or used (only the list of keys and their shapes was printed).

**open3d** (pre-installed): used once (ICP) to check that the mesh is symmetric under a 180° turn about its vertical axis (mean deviation 2 mm).

The experiment scripts are in `/work/scratch` and are not part of the solution.
