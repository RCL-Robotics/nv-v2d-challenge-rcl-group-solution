# Tools and dependencies

No packages, repositories, models, or weights were downloaded. No network calls were used. All numerical and rendering tools below were already installed in the container.

| Tool / package | Version / revision | Purpose | Required by solve.py |
|---|---|---|---|
| CPython | 3.11.10 | Script execution and standard library | Yes |
| NumPy | 1.26.3 | Pose transforms, arrays, NPZ output | Yes |
| SciPy | 1.16.3 | Temporal median and maximum filters | Yes |
| PyTorch | 2.5.1+cu124 | GPU mesh projection | Yes |
| nvdiffrast | 0.4.0 | Rasterize the supplied exact mesh | Yes |
| trimesh | 5.1.1 | Load GLB without processing; surface centroid | Yes |
| OpenCV / opencv-python | 4.11.0.86 | Mask morphology, registration, diagnostic video decoding | Yes |
| h5py | 3.16.0 | Read released SAM2 object/person masks | Yes |
| Pillow | 10.2.0 | Texture decoding through trimesh | Yes, indirectly |
| NVIDIA driver / CUDA runtime | 580.178.04 / PyTorch CUDA 12.4 | GPU execution on selected CUDA device 0 | Yes |
| nvidia-smi | supplied with NVIDIA driver | Inspect GPU availability | No |
| Git | 2.34.1 | Read installed nvdiffrast revision | No |

Installed nvdiffrast source: `/workspace/third_party/nvdiffrast`, commit
`253ac4fcea7de5f396371124af597e6cc957bfae`.
Upstream: https://github.com/NVlabs/nvdiffrast . No repository was cloned or modified.
The script directs any future PyTorch extension compilation cache to `/work/tools/torch_extensions`; the installed renderer ran without needing to create that cache.

Package metadata for pyrender 0.1.45 and matplotlib 3.11.2 was checked for availability; neither package was imported, executed, nor needed. FoundationPose, CoCoNet/CARI4D, MoGe, SAM2 inference, and other installed model pipelines were not run. The released poses and masks are inputs, not newly estimated by those tools.

Downloaded weights: **none**. Download URLs and weight SHA256 hashes: **not applicable**.

`solve.py` needs only the installed dependencies above and the files in the supplied input directory. It does not read `/pipeline`, `/weights`, scratch files, diagnostic results, or other output scripts. `make_evidence.py` is optional and produces the before/after image using the video and final NPZ. Bash, standard shell utilities, and Python standard-library modules were used for local file management and validation.
