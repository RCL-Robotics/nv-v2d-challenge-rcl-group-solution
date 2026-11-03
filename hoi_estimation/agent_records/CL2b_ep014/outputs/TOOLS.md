# Tools used

## Needed by `solve.py` (all pre‑installed in the container; nothing downloaded is needed)
| package | version | use |
|---|---|---|
| python | 3.11 (container) | |
| torch | 2.5.1+cu124 | optimisation, GPU tensors |
| nvdiffrast | 0.4.0 | rasterisation / soft silhouettes / depth / texture rendering |
| opencv‑python (cv2) | 4.11.0 | video decoding, rectification remap, filters |
| numpy | 1.26.3 | |
| scipy | 1.16.3 | symmetry‑axis fit (Nelder‑Mead, cKDTree), median/gaussian filters |
| trimesh | 5.1.1 | loading `object_mesh.glb` |
| h5py | 3.16.0 | `masks.h5` |
| Pillow | 10.2.0 | texture image (through trimesh) |

Local modules in `/work/outputs`:
* `common.py`: loading and the nvdiffrast renderer.
* `rectify.py`: plane rectification.
* `solve.py`: the pipeline.
* `validate.py`: the validation table. Not needed to produce the result.

Inputs only: `/work/inputs/*`. `/pipeline`, `/weights` and `/code` are not read by `solve.py`.

## Tried during development (NOT used by `solve.py`)
* **LightGlue** — git repo https://github.com/cvg/LightGlue, commit `eb42fee2d71449efb0aa5c10549752b5d75384d8`,
  cloned to `/work/tools/LightGlue`. Used in `match.py` for render‑to‑image matching; abandoned.
* **Weights** downloaded to `/work/tools/weights`. They are symlinked into `/work/tools/torch_home/hub/checkpoints`
  for offline loading.

  | file | URL | sha256 |
  |---|---|---|
  | superpoint_v1.pth | https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_v1.pth | 52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e |
  | superpoint_lightglue.pth | https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth | 6ff7040d0a497fc6639337946d7538dae07428c18f77a067a0b5a960e7cc551a |
  | aliked_lightglue.pth | https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/aliked_lightglue.pth | d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503 |
  | disk_lightglue.pth | https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/disk_lightglue.pth | b5b21d47ea24f2c5e501aec9c91b9716e4c8c3429a4dc1e615c133c4c9378335 |
  | aliked-n16.pth | https://github.com/Shiaoming/ALIKED/raw/main/models/aliked-n16.pth | 5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2 |
* **kornia** 0.8.3 (pre‑installed): checked, not used. The LoFTR weights URL returned 403.
* No pip packages were installed and no other data was downloaded. No ground truth or dataset files were accessed.
