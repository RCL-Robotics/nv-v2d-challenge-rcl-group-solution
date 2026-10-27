# Tools and reproducibility inventory

No packages, repositories, weights, videos, annotations, or other data were downloaded. No network requests were used. All software below was already installed in the provided container. No challenge data or ground-truth poses were accessed.

## Final solver dependencies

Run: `python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`

The solver is self-contained: it does not import the exploratory scripts or read their outputs. It needs video.mp4, masks.h5, camera.json, object_mesh.glb, reconstruction.npz, and episode.json from the input directory. It reads no pipeline or weight files and needs no scratch files.

| Installed package | Version | Purpose | Needed by solve.py |
|---|---|---|---|
| Python | 3.11.10, conda-forge | Runtime and standard library | Yes |
| numpy | 1.26.3 | Matrices, arrays, NPZ I/O | Yes |
| scipy | 1.16.3 | Least squares, rotations, smoothing, nearest-neighbor texture transfer | Yes |
| trimesh | 5.1.1 | Exact GLB loading, texture colors, mesh simplification | Yes |
| opencv-python | 4.11.0.86 | Video decoding, HSV features, masks, image resampling | Yes |
| h5py | 3.16.0 | Object/person mask input | Yes |
| torch | 2.5.1+cu124 | Forward-only full-mesh validation; GPU optimization tried and replaced | Yes |
| nvdiffrast | 0.4.0 | Forward rasterization; differentiable optimization tried and replaced | Yes |
| fast-simplification | 0.2.0 | Reduced experimental GPU mesh to 12,000 faces | No; exploratory only |
| Pillow | 10.2.0 | Texture loading; inspection contact sheets | Yes for texture loading |
| open3d | 0.20.0 | Availability checked only | No |
| pyrender | 0.1.45 | Availability checked only | No |

PyTorch uses its installed CUDA 12.4 build with NVIDIA driver 580.178.04. Only logical CUDA device 0, as selected by the environment, is used. The solver does not change CUDA_VISIBLE_DEVICES.

## Existing repositories inspected or used

- /workspace/third_party/nvdiffrast: commit 253ac4fcea7de5f396371124af597e6cc957bfae. Existing installed rasterizer; required through its Python package.
- /code/reconstruction: commit 7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80. Read-only inspection, particularly modules/v2d_foundation_pose/lib/FoundationPose/Utils.py, for raster camera conventions. The solver implements its own direct OpenCV-to-clip projection and does not import this repository. Inspected file SHA-256: 8536ac09e876679c3e35690692142128ad9e06a48dbd2cafa904007ac99d7626.

No git repositories were cloned. No weights were downloaded or used, so downloaded-weight URLs and hashes do not apply.

## Other tools

Bash, ripgrep 15.2.0, git, nvidia-smi, sha256sum, and standard file inspection commands. OpenCV line-segment detection was used to inspect the partly hidden blade. Resulting image-plane directions are explicit commented constants in solve.py. They are observations from the supplied RGB, not external 3-D annotations.

Seeds are fixed for Python, NumPy, OpenCV and PyTorch. cuDNN benchmarking and TF32 are disabled. Final fitting uses single-threaded CPU least squares with explicit finite differences, fixed area-stratified mesh samples, and SciPy's Euclidean distance transform. OpenCV's precise distance transform was tried and replaced after separate processes produced small numerical differences. The GPU is used only for forward rasterization and validation. An earlier differentiable GPU optimizer was rejected because repeated executions differed; its backward atomic reductions are not part of the final solver. Repeatability checks are recorded in REPORT.md.
