# Tools used

## Runtime / external software

All tools were already installed in the supplied container; nothing was downloaded and no network access was used by `solve.py`.

| Tool | Version | Role | Needed by `solve.py` |
|---|---|---|---|
| Python | 3.11.10 | runtime | Yes |
| NumPy | 1.26.3 | arrays, linear algebra | Yes |
| SciPy | 1.16.3 | rotation means, Savitzky-Golay smoothing | Yes |
| OpenCV (`opencv-python`) | 4.11.0.86 | masks, contours, distance transforms | Yes |
| h5py | 3.16.0 | read supplied SAM2 masks | Yes |
| trimesh | 5.1.1 | load exact GLB mesh | Yes |
| PyTorch | 2.5.1+cu124 | GPU rasterization tensors | Yes |
| nvdiffrast | 0.4.0 | deterministic mesh silhouette renderer | Yes |
| NVIDIA RTX 6000 Ada GPU | 50.9 GB device, CUDA 12.4 | rasterization and fitting | Yes |

## Supplied data and reconstruction code

- `/code/reconstruction` Git commit examined while diagnosing the pipeline: `7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80`. The final solver does not import or require this repository.
- CARI4D-derived inputs used as evidence: `reconstruction.npz` and `masks.h5`. The final solver also reads `object_mesh.glb` and `camera.json` from the supplied inputs.
- Supplied weights under `/weights` and CARI4D intermediates under `/pipeline` were inspected for diagnosis only; the final solver does not load external model weights. In particular, `/pipeline/depth/moge2/aligned.h5` was used for a depth consistency check, but MoGe depth was not used in the final trajectory because its per-pixel residual on the thin object was biased and noisy.

## Downloads / Git repositories / external weights

None. No pip packages, repositories, or model weights were downloaded for this solution. Consequently there are no download URLs or weight hashes to report.

## Output checksums

- `object_poses_world.npz`: `2f9ea95bb31918955a704fee6a76726241fe8fc56c3214157bfed7e074d36a34`
- `solve.py`: `db161665683a2ab1208b70120d9692552c83c7507a29777eba3d0878b24d23c1`
