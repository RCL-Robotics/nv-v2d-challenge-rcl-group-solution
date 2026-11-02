# Tools and provenance

No weights, repositories, datasets, or annotations were downloaded for the final solve. All inputs are the released files in `/work/inputs`; camera-frame depth comes from `/pipeline/depth/moge2/aligned.h5`.

## Runtime dependencies used by `solve.py`

- Python 3.11
- numpy 1.26.3
- scipy 1.16.3
- opencv-python 4.11.0
- h5py 3.16.0
- trimesh 5.1.1
- PyTorch 2.5.1+cu124
- nvdiffrast 0.4.0

## Tools inspected or tried during development (not required by `solve.py`)

- ffmpeg 4.4.2-0ubuntu0.22.04.1: used only to extract a visual frame montage during development.
- SAM3D-Body checkpoint at `/weights/sam3d_body/checkpoints/sam-3d-body-dinov3/model.ckpt`: tried for diagnostic 2-D keypoints. Its supporting torchvision 0.20.1+cu124 was used by that diagnostic only. The final solver does not run the model.
- Open3D 0.20.0: imported while checking available tools; not used.
- kornia 0.8.3: availability probe only; not used.
- Availability probes that were not installed and not used: mediapipe, mmpose, ultralytics, onnxruntime, rtmlib.
- CARI4D source under `/workspace` and `/code/reconstruction`: inspected for conventions; no code was copied into the result.
- Git repositories cloned or updated: none.

No public downloads were made. The final script is offline and deterministic.
