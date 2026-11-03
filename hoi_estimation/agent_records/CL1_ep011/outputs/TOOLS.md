# Tools used

No external tools were downloaded. No pip packages were installed, no git repositories cloned, no model weights
fetched (`/work/tools` is empty). Everything ran on the software that was already in the container.

| tool | version | origin | used for | needed by `solve.py` |
|---|---|---|---|---|
| Python | 3.11.10 | container (`/opt/conda`) | everything | yes |
| PyTorch | 2.5.1+cu124 (CUDA 12.4) | container | optimisation, FFT shift search | yes |
| nvdiffrast | 0.4.0 | container | differentiable textured rendering (mask, texture, depth) | yes |
| OpenCV (cv2) | 4.11.0 | container | video decoding, morphology, connected components | yes |
| trimesh | 5.1.1 | container | loading `object_mesh.glb` (vertices, faces, UVs, texture) | yes |
| h5py | 3.16.0 | container | reading `masks.h5` (and, for analysis only, `/pipeline/depth/moge2/aligned.h5`) | yes (masks only) |
| NumPy | 1.26.3 | container | everything | yes |
| SciPy | 1.16.3 | container | rotation averaging (`Rotation.mean`), median filter, binary morphology | yes |
| ffmpeg | 4.4.2 | container (system) | analysis only: extracted JPEG frames to `/work/scratch/frames` for inspection | no |

Data used: only the files in `/work/inputs`. During analysis I also read the MoGe-2 aligned depth from
`/pipeline/depth/moge2/aligned.h5`, as a depth cross-check (see REPORT.md, Depth). `solve.py` does not use it.
No ground truth or dataset copies were accessed.
