# Tools and provenance

No packages were installed, repositories cloned, weights downloaded, or network requests made. All software and data were already present in the container. `solve.py` is CPU-only and does not load any learned model weights.

| Tool / Python package | Version | Use | Needed by solve.py |
|---|---|---|---|
| Python | 3.11.10 | Deterministic solver and diagnostic scripts | Yes |
| NumPy | 1.26.3 | Arrays, geometry, NPZ I/O | Yes |
| SciPy | 1.16.3 | Sparse LSQR; exploratory least-squares and KD trees; audit convex hull | Yes |
| opencv-python | 4.11.0.86 (cv2 4.11.0) | Mask dilation; video reading, depth PNG decoding, diagnostic drawings | Yes |
| trimesh | 5.1.1 | Exact GLB mesh loading and convex hull | Yes |
| h5py | 3.16.0 | Released masks; optional pipeline depth audit | Yes |
| Matplotlib | 3.11.2 | Standalone depth comparison chart in evidence.py | No |
| Pillow | 10.2.0 | Existing GLB/image and plotting dependency | Indirect mesh/image dependency |
| NetworkX | 3.2.1 | Existing trimesh dependency | Indirect |
| packaging | 24.1 | Existing library dependency | Indirect |
| PyTorch | 2.5.1+cu124 | Imported to inspect installed version and CUDA availability only | No |
| Bash | 5.1.16 | Shell orchestration | No, beyond invoking Python |
| ripgrep | 15.2.0, revision e89fff89ac | File/source discovery | No |
| Git | Container-installed CLI | Read existing source revision only | No |

Python standard-library modules used include argparse, pathlib, json, os, time, importlib.metadata, sys, shutil, and exploratory pickle. Other ordinary shell utilities were used only for file inspection and moving an exploratory script. No separately acquired binary tools were used.

**Existing source repository inspected:** `/code/reconstruction`, commit `7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80`. Searched existing depth-decoding and body-mesh source for documentation; no modules from this repository are imported by the final solver or audit. No repository was cloned or modified.

**Existing model outputs inspected:** `/pipeline/depth/moge2/aligned.h5`, its depth report and intrinsics metadata, and structural metadata for the existing human initialization. MoGe-2 report identifies model `Ruicheng/moge-2-vitl-normal`, revision `b135031bae30b5ac2ae141a0e68717795ce38340`, and pipeline source commit `925b8ed835a7a9cdb7578ba15c658a0afc969030`. The already-computed maps are used only by the optional audit; `solve.py` does not need them. No inference was run.

**Downloaded weights:** none. Consequently there are no download URLs or downloaded-file SHA256 values to list. Existing `/weights` paths were listed for inspection; no weight file was loaded. There are no new external data dependencies.
