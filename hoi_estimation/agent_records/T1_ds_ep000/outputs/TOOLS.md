# Tools used

## Required by `solve.py`

- Python 3.11.10
- NumPy 1.26.3
- SciPy 1.16.3 (`scipy.spatial.transform.Rotation`, `scipy.signal.savgol_filter`)

No network access, downloaded weights, or external repositories are needed by
`solve.py`.

## Used for analysis and validation only

- OpenCV (`opencv-python`) 4.11.0: mask contour, chamfer, Canny-edge, and video-frame diagnostics.
- h5py 3.16.0: read the supplied masks and existing aligned MoGe-2 depth.
- trimesh 5.1.1: load the supplied GLB mesh for rendering and floor-contact checks.
- PyTorch 2.5.1+cu124 and nvdiffrast 0.4.0: differentiable-style silhouette/depth rendering for offline checks.
- FFmpeg/ffprobe 4.4.2: read video metadata.
- Existing CARI4D pipeline aligned MoGe-2 depth under `/pipeline/depth/moge2/`: independent validation/calibration evidence. The final `solve.py` does not require it.

## External downloads and repositories

- No pip packages were installed.
- No GitHub repositories were cloned.
- No model weights were downloaded.
- No external weight URLs or SHA-256 values are therefore associated with the final solver.

The released inputs and pre-existing local CARI4D/FoundationPose/CoCoNet/MoGe-2
artifacts were used only as allowed by the task. No challenge ground-truth data,
annotation, parquet trajectory, or mirror was accessed.
