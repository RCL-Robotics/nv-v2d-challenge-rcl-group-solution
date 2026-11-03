# Tools and dependencies

No packages, repositories, tools, or weights were downloaded or installed. No network requests were made. All software below was already installed in the container.

| Package | Version | Use | Needed by solve.py |
|---|---|---|---|
| NumPy | 1.26.3 | Arrays, linear algebra, NPZ I/O | Yes |
| SciPy | 1.16.3 | Rotation conversion and Gaussian temporal smoothing | Yes |
| opencv-python / opencv-contrib-python | 4.11.0.86 | Mask crops, morphology, distance transforms; video inspection | Refit only |
| trimesh | 5.1.1 | Exact GLB geometry, oriented bounding-box centre | Yes |
| PyTorch | 2.5.1+cu124 | Local pose optimization on the selected CUDA GPU | Refit only |
| nvdiffrast | 0.4.0 | Differentiable mesh silhouettes | Refit only |
| h5py | 3.16.0 | Read released object/person masks | Refit only |
| Pillow | 10.2.0 | GLB texture decoding and inspection/evidence contact sheets | Yes |
| pyrender | 0.1.45 | Availability checked only; never rendered with it | No |
| pyglet | 2.1.16 | Package metadata checked only | No |

The preinstalled nvdiffrast source is `/workspace/third_party/nvdiffrast`, commit `253ac4fcea7de5f396371124af597e6cc957bfae`. It is accessed through its installed Python package. The supplied CARI4D source at `/code/reconstruction`, commit `7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80`, and the supplied `/pipeline/pipeline_report.json` were inspected for coordinate conventions; none of that code or its models is invoked by the solver. No repositories were cloned. No model inference, downloaded weights, checkpoint files, or additional data were used by the solver. Download URLs and weight SHA256 entries are therefore not applicable.

Also used: preinstalled Python 3.11, Bash, core command-line file tools, ripgrep, git (to identify the supplied source commits), and nvidia-smi (GPU availability check). The default export needs only the supplied input directory and NumPy, SciPy, trimesh, and Pillow. The optional refit also needs the Python/CUDA dependencies above. It does not read `/pipeline`, `/weights`, `/code`, `/work/scratch`, or the optional inspection scripts.

Reproduce the submission offline:

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

The fitted numerical correction coefficients are embedded in `solve.py` as a compressed NPZ payload. Default export uses CPU float64 and was verified bitwise identical across repeated runs and to the original fitted result. `--refit` exposes the complete seeded mask/video fitting procedure; CUDA atomic gradient accumulation can cause small differences when refitting. No other result file or scratch asset is needed by default.

`validate.py`, `render_evidence.py`, and `inspect_baseline.py` are optional inspection utilities with hard-coded workspace paths. They are not invoked by `solve.py`.
