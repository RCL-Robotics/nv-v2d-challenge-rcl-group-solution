# Tools and dependencies

No packages, repositories, or weights were downloaded or installed. No network access was used. All packages below were already installed in the container. The final solver uses only the released inputs and these installed packages; it does not need `/pipeline`, `/weights`, or any scratch files.

| Package | Installed version | Used by `solve.py` | Purpose |
|---|---|---|---|
| numpy | 1.26.3 | Yes | Arrays and transforms |
| scipy | 1.16.3 | Yes | Rotation averaging and interpolation |
| opencv-python | 4.11.0.86 | Yes | Video decoding, SIFT, descriptor matching, RANSAC PnP, masks |
| trimesh | 5.1.1 | Yes | Load exact textured GLB in its original coordinates |
| h5py | 3.16.0 | Yes | Read supplied SAM2 masks |
| torch | 2.5.1+cu124 | Yes | CUDA tensors for rendering |
| nvdiffrast | 0.4.0 | Yes | Rasterize textured mesh and mesh-coordinate images |
| pillow | 10.2.0 | Indirectly | Decode the GLB texture through trimesh |

Python: 3.11.10. GPU: the selected CUDA device 0, NVIDIA RTX 6000 Ada Generation. The solver uses one GPU. CUDA runtime reported by PyTorch: 12.4.

The installed nvdiffrast source is `/workspace/third_party/nvdiffrast`, git commit `253ac4fcea7de5f396371124af597e6cc957bfae` (upstream https://github.com/NVlabs/nvdiffrast). This pre-existing installation is needed by the solver. No other git repositories were used as tools.

Other tools used during inspection: Bash, `rg`, `cat`, `ls`, `tail`, `git rev-parse`, and `nvidia-smi`; Python standard library (`json`, `pathlib`, `time`, `argparse`, `random`, `os`, `importlib.metadata`). They require no downloaded assets. Inspection of `/pipeline` was limited to the available file list and depth metadata/schema; none of those intermediates are needed for the final result.

Exploratory approaches used the same dependencies: projecting the baseline mesh, comparing four half-turn hypotheses, matching SIFT from a rendered-view library, and matching SIFT from per-frame renders. The global rendered-view library was discarded because repeated printed text caused ambiguous matches; the final solver uses per-frame hypotheses and explicit geometric/photometric gates.

Downloaded weights: **none**. Download URLs and SHA-256 hashes: **not applicable**. No additional model is run.
