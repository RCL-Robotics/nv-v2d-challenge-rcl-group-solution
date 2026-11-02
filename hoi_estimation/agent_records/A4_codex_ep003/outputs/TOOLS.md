# Tools and reproducibility

No packages, repositories, or weights were downloaded. No network was used. All tools below were already installed in the container. No external pose estimator or pretrained model was run.

| Installed package | Version | Required by solve.py | Purpose |
|---|---|---|---|
| numpy | 1.26.3 | Yes | Pose arrays and output |
| scipy | 1.16.3 | Yes | Rotations, interpolation and temporal filters |
| torch | 2.5.1+cu124 | Yes | CUDA image fitting |
| nvdiffrast | 0.4.0 | Yes | Differentiable textured mesh rendering |
| trimesh | 5.1.1 | Yes | Read the released GLB exactly in mesh-file coordinates |
| opencv-python | 4.11.0.86 | Yes | Decode video, silhouette distance transforms |
| h5py | 3.16.0 | Yes | Read released masks |
| Pillow | 10.2.0 | Yes | Texture image conversion; diagnostic contact sheets |

Python 3.11.10. Standard-library modules used include argparse, pathlib, json, os, time, random and importlib.metadata. Shell utilities and nvidia-smi were used for local inspection only. The optional experiments/evaluate_result.py and visualization scripts use the same installed packages.

Availability/version checks also inspected pyrender 0.1.45 and matplotlib 3.11.2; neither was executed for rendering/plotting or required by solve.py. GPU status was inspected with nvidia-smi (driver 580.178.04).

Git repositories cloned or used as external code: none. Downloaded weights, URLs and SHA256 values: none. The available CARI4D code, /pipeline and /weights were listed for context but their code and models are not dependencies of the solver. No Hugging Face access or challenge annotations were used.

Run offline:

```sh
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```

The solver is self-contained, recomputes the image fitting and temporal selection, and needs only the released input directory, the packages above and one CUDA GPU. It does not read exploratory scripts, intermediate candidate files, diagnostics, /work/scratch, or downloaded assets. It creates its own intermediate solver_candidates.npz next to the requested output. Random seeds are zero, CPU threads are fixed at four, and cuDNN benchmarking is disabled and PyTorch deterministic algorithms are enabled. CUDA renderer floating-point reductions may vary in their last digits across different hardware or software versions.
