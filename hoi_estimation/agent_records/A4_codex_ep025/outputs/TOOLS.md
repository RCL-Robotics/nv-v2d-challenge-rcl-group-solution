# Tools and reproducibility

No packages, repositories, model weights, or external data were downloaded. No network calls were made. All processing used the supplied episode files and packages already installed in the container. No ground-truth trajectories, annotations, challenge datasets, or mirrors were accessed.

## Final solver dependencies

The single file `solve.py` reads only these files under its `--inputs` directory: `video.mp4`, `object_mesh.glb`, `camera.json`, `reconstruction.npz`, `masks.h5`, and `episode.json`. It does not read cached candidates, other authored Python modules, scratch files, /pipeline, /weights, or /code. It uses the installed CUDA rasterizer and the one GPU exposed by `CUDA_VISIBLE_DEVICES=0`.

| Package/tool | Installed version | Use | Needed by solve.py |
| --- | --- | --- | --- |
| Python | 3.11.10, conda-forge, GCC 13.3.0 | Script runtime and standard library | Yes |
| NumPy | 1.26.3 | Matrices, poses, arrays, NPZ output | Yes |
| SciPy | 1.16.3 | Rotation matrices, distance transforms, binary dilation | Yes |
| trimesh | 5.1.1 | Exact supplied GLB mesh and UV texture loading | Yes |
| opencv-python | 4.11.0.86 | Video decoding and inspection montages | Yes |
| h5py | 3.16.0 | Supplied SAM2 masks | Yes |
| Pillow | 10.2.0 | Supplied mesh texture image conversion | Yes, through mesh material |
| torch | 2.5.1+cu124 | Forward GPU projection/rasterization arithmetic | Yes |
| nvdiffrast | 0.4.0 | Deterministic forward textured rasterization | Yes |
| imageio | 2.38.0 | Installed image/mesh support dependency; version inspected | Not directly imported |
| networkx | 3.2.1 | Installed trimesh dependency; version inspected | Not directly imported |
| CUDA runtime | 12.4, as reported by PyTorch | GPU computation | Yes |
| NVIDIA driver | 580.178.04 | GPU runtime | Yes |
| ripgrep | 15.2.0, rev e89fff89ac | Locate task instructions and installed source | No |
| GNU bash | 5.1.16 | Development command execution | No |
| GNU sed | 4.8 | Attempted development text edit; invalid regex attempt had no effect | No |
| Git | 2.34.1 | Read source revision hashes | No |

Python standard-library modules used include argparse, pathlib, os, json, time, csv, hashlib, and importlib.metadata. Shell utilities cat, ls, head, pwd, printenv, and nvidia-smi were used for inspection. Client exec_command, apply_patch, write_stdin, and view_image were used to execute commands, create files, wait for completion, and inspect visual evidence.

## Existing source trees inspected

- `/code/reconstruction`: commit `7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80`. Source paths were searched to inspect the existing rasterization workflow. Its pipeline was not rerun and the final solver does not import it.
- Installed `/workspace/v2d_foundation_pose`: source paths searched; this installation has no Git metadata. Not needed by the final solver.
- Installed `/workspace/third_party/nvdiffrast`: commit `253ac4fcea7de5f396371124af597e6cc957bfae`. Package metadata reports a local installation from that directory. Needed as the installed nvdiffrast package; no repository download was made.

## Approaches tried

1. Baseline and FoundationPose textured projection inspection.
2. Differentiable silhouette/material optimization using PyTorch Adam and nvdiffrast. Rejected for the final solver because repeated runs showed small differences from GPU gradient reductions.
3. Deterministic forward-only coordinate search, multiple existing-pose seeds, explicit axial-roll alternatives, conservative evidence gates, and temporal candidate selection. This is the final method.

No pretrained models or weights were loaded. There are therefore no downloaded-weight URLs or SHA256 values to list. Input-file SHA256 values are recorded in `input_sha256.json`; these are the supplied inputs, not external assets. The final report and validation JSON record output reproducibility checks.
