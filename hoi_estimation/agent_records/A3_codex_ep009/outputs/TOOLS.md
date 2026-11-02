All software was already installed in the container. **No pip package, repository, model weight, or other external data was downloaded or installed.** The final script performs no network access.

| Package/tool | Version | Usage | Required by `solve.py` |
| --- | --- | --- | --- |
| Python | 3.11.10 | Scripts and standard library (`argparse`, `csv`, `json`, `pathlib`, `random`, `time`, `hashlib`, `os`) | Yes |
| NumPy | 1.26.3 | Pose arrays, transforms, masks, output serialization | Yes |
| SciPy | 1.16.3 | Bounded contour optimisation, median filtering, SO(3) means/interpolation | Yes |
| PyTorch | 2.5.1+cu124 | CUDA tensors; also inspected schemas of provided pipeline `.pth` outputs during exploration | Yes |
| nvdiffrast | 0.4.0 | Exact-mesh silhouette rendering; textured rendering for diagnostic heading sweeps | Yes |
| trimesh | 5.1.1 | Exact GLB scene-to-geometry loading, centroid, UVs/normals for diagnostic rendering; inspected its GLTF UV-convention source | Yes |
| opencv-python | 4.11.0.86 | Video decoding, contours, colour/occlusion processing, distance transforms and interpolation | Yes |
| h5py | 3.16.0 | Input mask crops; inspected provided depth-file structure during exploration | Yes |
| Pillow | 10.2.0 | GLB texture loading through trimesh; diagnostic image sheets and texture export | Yes, through trimesh |
| matplotlib | 3.11.2 | Diagnostic plot of baseline heading, opening axis, and rotation steps | No |
| Open3D | 0.20.0 | Installed-version query only; no algorithm or model was executed | No |
| tqdm | 4.66.5 | Installed-version query only | No |
| networkx | 3.2.1 | Installed-version query; also part of the preinstalled trimesh dependency environment | No direct usage |
| setuptools | 72.1.0 | Installed-version query only | No direct usage |
| ninja | 1.11.1.1 | Installed-version query only; no compilation or build was needed | No |
| NVIDIA CUDA runtime/toolkit | 12.4; NVCC 12.4.131 | CUDA rendering; queried the installed compiler version | CUDA runtime yes; compiler no |
| NVIDIA driver / GPU | 580.178.04 / RTX 6000 Ada Generation | The selected CUDA GPU; queried with `nvidia-smi` | Yes |
| GNU bash | 5.1.16 | Workspace commands | No |
| ripgrep | 15.2.0, revision `e89fff89ac` | Workspace/source file discovery | No |
| Git | 2.34.1 | Read the existing source repository commit | No |
| GNU findutils | 4.8.0 | Listed permitted pipeline files | No |
| Standard shell/coreutils (`cat`, `ls`, `mkdir`, `cp`, `du`, `ps`, `head`) | Container-provided | Read files, create output directory, inspect processes and file sizes, retain first-run comparison | No |

The existing read-only source repository `/code/reconstruction` was inspected/listed; its commit is **`7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80`**. It was not cloned or modified, and `solve.py` does not import or execute its pipeline modules. Its supplied FoundationPose/CoCoNet reconstruction is consumed through `inputs/reconstruction.npz`.

No repositories were cloned, and no other Git repository was used. Installed package source inspection is identified by the package versions above. No FoundationPose, CoCoNet, MoGe, SAM, learned matcher, or other model was run. No weights were downloaded, loaded for inference, or needed by the final script; weight URLs and downloaded-file SHA-256 values are consequently not applicable.

Exploratory methods tried with these packages were textured mesh rendering at 30° heading increments; baseline camera/world and temporal-rotation diagnostics; silhouette squared-error translation fitting; contour fitting with the full person mask excluded; and the final skin-aware contour fit. The first two fitting variants and broad heading changes were not used in the final solver. All final code is in `solve.py`; the remaining scripts and diagnostics are optional evidence.
