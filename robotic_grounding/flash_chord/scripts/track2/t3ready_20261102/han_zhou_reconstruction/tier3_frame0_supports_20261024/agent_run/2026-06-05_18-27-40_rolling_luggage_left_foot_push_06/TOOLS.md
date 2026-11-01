# Tools used

Nothing was downloaded: no pip installs, no git clones, no external weights. `/work/tools` is empty.
Everything below was already installed in the container (Python 3.11.10).

| tool | version | used for | needed by `solve.py` |
|---|---|---|---|
| torch (CUDA) | 2.5.1+cu124 | optimisation (Adam), GPU tensors | yes |
| nvdiffrast | 0.4.0 | differentiable silhouette rasterisation (`RasterizeCudaContext`, `antialias`) | yes |
| roma | 1.6.1 | rotation-vector to matrix (differentiable) | yes |
| trimesh | 5.1.1 | mesh loading (GLB), vertex merging; signed distance (exploration only) | yes |
| open3d | 0.20.0 | quadric decimation of the body/handle meshes for silhouette rendering | yes |
| opencv-python / contrib | 4.11.0.86 | video decoding, Canny edges, distance transforms (flip test) | yes |
| scipy | 1.16.3 | rotations, median filter / morphology for phase detection | yes |
| h5py | 3.16.0 | reading `masks.h5` | yes |
| numpy | 1.26.3 | everything | yes |

Data used: only the files in `/work/inputs` (`video.mp4`, `masks.h5`, `object_mesh.glb`, `camera.json`,
`ground_plane.json`, `reconstruction.npz`). `frame0_reference_human.npz` was used only during development, as a
check of the floor plane (sole height of the frame-0 human) for the depth-cue analysis in the report. `solve.py`
does not read it. `/pipeline`, `/weights` and `/code` were not needed. No network access is needed.
