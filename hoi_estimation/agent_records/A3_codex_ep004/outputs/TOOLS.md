All software below was already installed in the container. No packages, repositories, datasets, annotations, or model weights were downloaded. No network access was used. No CARI4D model was rerun; only the supplied reconstruction and its FoundationPose intermediate estimates were read.

| Tool | Version | Use | Needed by `solve.py` |
|---|---|---|---|
| Python | 3.11.10 | Execution | Yes |
| NumPy | 1.26.3 | Pose arrays, mask statistics, geometry | Yes |
| SciPy | 1.16.3 | Project corrected rotations onto SO(3) | Yes |
| PyTorch | 2.5.1+cu124 | Batched pose fitting on the selected GPU | Yes |
| nvdiffrast | 0.4.0 | Mesh silhouette rasterization | Yes |
| opencv-python | 4.11.0.86 | Video decoding, masks, appearance checks | Yes |
| h5py | 3.16.0 | Supplied segmentation masks | Yes |
| trimesh | 5.1.1 | Read the supplied GLB, weld vertices, inspect geometry | No; proxy creation and independent validation only |
| Open3D | 0.20.0 | Simplify the rendering mesh; check approximation distances | No; proxy creation only |
| Pillow | 10.2.0 | Contact sheets and final evidence image | No; `make_preview.py` only |

Bash and the preinstalled standard utilities `cat`, `ls`, `rg`, `find`, `head`, `tail`, `sed`, `mv`, `cp`, `ps`, `date`, `du`, and `nvidia-smi` were used for workspace inspection and process monitoring. CUDA runtime/compiler components and native build tools used by nvdiffrast are supplied by the container. The final script places any newly compiled extension cache under `/work/tools/torch_extensions`.

Git repositories cloned or checked out: none. There are no additional repository commit hashes to record. Downloaded weights and their URLs/SHA256: none. The pretrained weights under `/weights` were not used.

The only additional runtime data artifact is `/work/tools/bat_render_mesh.npz`, derived exclusively from the supplied mesh. Its SHA256 is `5814d443720e06b41fe7b60b5d0d833c718477d17fb91f8f0f167d5b03362c67`. It contains 3,002 vertices and 6,000 triangles, in the original mesh-file coordinate frame and metre scale. The optional `build_render_mesh.py` records its complete construction: load the scene as specified in the task, weld duplicated vertices, run Open3D quadric decimation, and discard unreferenced vertices. No pose estimates are cached in this artifact.

The source GLB SHA256 is `5aa2294d698eb370d0d1435963e42022cc1064faa355b547e492ffe893e51cd7`. `solve.py` verifies both checksums. The original reconstruction SHA256 is `7bb731d3abbef29361edd6a5f13fdb544342804e1784267121902618bd963139`.

Runtime dependencies are the first seven package rows above, the proxy NPZ, the selected CUDA GPU, and `video.mp4`, `masks.h5`, `camera.json`, `object_mesh.glb`, and `reconstruction.npz` under the specified input directory. The solver does not read development scratch files, diagnostics from previous runs, `/pipeline`, or the internet. All pose candidates and decisions are recomputed on each invocation.
