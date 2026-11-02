# Tools and external assets

## Runtime environment

- Python 3.11.10
- numpy 1.26.3
- scipy 1.16.3
- opencv-python 4.11.0
- h5py 3.16.0
- trimesh 5.1.1

`solve.py` needs only the packages above plus the inputs and `/pipeline/depth/moge2/aligned.h5`. It runs offline with fixed seeds.

## Tried / used during development

- Open3D 0.20.0 (`open3d`) - tried for point-cloud surface reconstruction; final script does not need it.
- PyTorch 2.5.1+cu124 and PyTorch3D 0.7.9 - tried for differentiable silhouette rendering; final script does not need them.
- torchvision 0.20.1+cu124 - used for an independent Keypoint R-CNN cross-check.
- Keypoint R-CNN COCO weights: `https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-fc266e95.pth`
  - local path: `/work/tools/torch/hub/checkpoints/keypointrcnn_resnet50_fpn_coco-fc266e95.pth`
  - sha256: `fc266e953d2b302cdcbb9ae66f71f6b0d4649928bf02dc573961e361e4918926`
  - not required by `solve.py`.
- CARI4D source tree `/code/reconstruction` at commit `7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80` - inspected for file conventions; no code imported by the final script.
- No git repositories or pip packages were installed. No model weights beyond the listed Keypoint R-CNN download were used.
