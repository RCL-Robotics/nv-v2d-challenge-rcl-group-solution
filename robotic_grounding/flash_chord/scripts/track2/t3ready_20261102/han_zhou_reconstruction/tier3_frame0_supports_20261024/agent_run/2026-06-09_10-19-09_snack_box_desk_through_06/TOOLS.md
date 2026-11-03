# Tools used

Nothing was downloaded. No pip packages were installed, no git repositories were cloned and no external weights were fetched.
Everything below was already in the container or in `/weights`.

| tool | version / identity | used for | needed by `solve.py` |
|---|---|---|---|
| Python 3.11 + numpy | numpy 1.26.3 | everything | yes |
| PyTorch | 2.5.1+cu124 | Adam-based render-and-compare optimisation, GPU tensors | yes |
| nvdiffrast | 0.4.0 (preinstalled, `/opt/conda/.../nvdiffrast`) | differentiable rasterisation of the textured object mesh (silhouette + texture) | yes |
| trimesh | 5.1.1 | loading `object_mesh.glb` (mesh, UVs, base-colour texture) | yes |
| opencv-python / opencv-contrib-python | 4.11.0.86 | video decoding, crops / warps, morphology | yes |
| scipy | 1.16.3 | rotations, Slerp, median filter | yes |
| h5py | 3.16.0 | reading `masks.h5` | yes |
| SAM 2 (`sam2` package "SAM-2" 1.0, preinstalled) + hydra-core 1.3.7 / omegaconf 2.3.1 | image predictor `SAM2ImagePredictor` | repairing partial object masks in the moving phases | yes |
| SAM 2.1 Hiera-L weights | `/weights/sam2/sam2.1_hiera_large.pt` sha256 `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318`; config `/weights/sam2/sam2.1_hiera_l.yaml` sha256 `545e4325aa5c19a1615d43c946b07276ed4c57214eacf1437e38fa3d9374f636` (local, read-only) | same | yes |

Also considered but not used: kornia (0.8.3, preinstalled) LoFTR / LightGlue feature matching. Their weights would
have had to be downloaded, and texture-NCC render-and-compare was already enough to tell the symmetric hypotheses
apart. FoundationPose and CoCoNet were not re-run. Their stored outputs are in `reconstruction.npz`, which `solve.py`
reads only through the baseline `T_world_mesh`.

`solve.py` reads only `/work/inputs` and `/weights/sam2`. It does not read `/pipeline`, and it uses no network and no scratch files.
`validate.py` (the validation table in REPORT.md) imports `solve.py` and uses the same tools.
