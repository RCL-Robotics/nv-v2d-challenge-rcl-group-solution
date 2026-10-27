# Tools Used

No external repository was cloned and no model weights or files were downloaded.
All tools below were already installed in the container. The final solver does
not use the network, `/pipeline`, or `/work/scratch`.

## Required by `solve.py`

| Tool | Version | Use |
|---|---:|---|
| Python | 3.11.10 | Runtime |
| NumPy | 1.26.3 | Arrays, linear algebra, deterministic numerical operations |
| SciPy | 1.16.3 | SO(3) utilities, Powell optimization, robust filters, PCHIP interpolation |
| OpenCV (`cv2`) | 4.11.0 | Video decoding, Canny edges, masks/contours, distance transforms |
| trimesh | 5.1.1 | Loads the supplied GLB and exposes exact mesh vertices |
| h5py | 3.16.0 | Reads supplied `masks.h5` |

## Used for experiments/validation, not required by the final solver

| Tool | Version | Use |
|---|---:|---|
| PyTorch | 2.5.1+cu124 | Differentiable textured-render experiments |
| nvdiffrast | 0.4.0 | Rasterized textured-mesh hypothesis rendering and depth/edge experiments; helped reject symmetry flips and quantify depth bias |
| Pillow | 10.2.0 | Contact sheets for visual inspection |
| scikit-image | 0.26.0 | Early photometric/structural score experiments |

## Existing read-only pipeline artifacts inspected

* CARI4D reconstruction and stage reports under `/pipeline`.
* Provided MoGe-2 aligned depth under `/pipeline/depth/moge2/aligned.h5`, used
  only as an independent validation cue. MoGe-2 model id
  `Ruicheng/moge-2-vitl-normal`, revision
  `b135031bae30b5ac2ae141a0e68717795ce38340`, source commit
  `925b8ed835a7a9cdb7578ba15c658a0afc969030`, as recorded in the supplied depth
  report.
* No CARI4D, SAM3D-Body, FoundationPose, or CoCoNet model was rerun.

## Downloaded weights

None. Therefore there are no download URLs or downloaded-file SHA-256 values to
report.

## Supplied input checksums used during validation

| Input | SHA-256 |
|---|---|
| `object_mesh.glb` | `e6d434d16811524b8b3725e32c9ba2781bb878c47962f2e6a25b22e0cc2e4427` |
| `reconstruction.npz` | `b6814ef943adf81d5013b1332e6c3b607e9c80d713b7d70c5a6d0ba602df1e27` |
| `masks.h5` | `b3866396e93476e2789b7366935bbc8cd77b711305c8e225b033c55014dcaffb` |
| `video.mp4` | `9fe5b6797fe91a3a696e70ab86c0500a6e238330cf7916c0fe86c21e37fce3b2` |
