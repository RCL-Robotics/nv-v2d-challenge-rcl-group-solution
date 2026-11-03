# Tools and provenance

All tools were already installed in the supplied container. No packages, repositories, datasets, annotations, or model weights were downloaded. No external service or network call was used. The solver operates offline and reads only the supplied inputs; it does not require `/pipeline`, `/weights`, `/code`, `/work/tools`, or scratch files.

| Tool | Version / provenance | Use | Needed by solve.py |
|---|---|---|---|
| Python | 3.11.10, conda-forge | Scripts and standard library | Yes |
| NumPy | 1.26.3 | Pose arrays and numerical operations | Yes |
| SciPy | 1.16.3 | Robust least squares, rotations, temporal smoothing | Yes |
| OpenCV (`opencv-python`) | 4.11.0.86 | Video decoding, SIFT, matching, RANSAC, image processing | Yes |
| PyTorch | 2.5.1+cu124 | GPU tensor operations | Yes |
| nvdiffrast | 0.4.0; supplied checkout `/workspace/third_party/nvdiffrast`, commit `253ac4fcea7de5f396371124af597e6cc957bfae`; upstream https://github.com/NVlabs/nvdiffrast | Render the supplied mesh texture and per-pixel mesh coordinates; exploratory differentiable registration | Yes |
| trimesh | 5.1.1 | Load the exact GLB mesh, merge identical vertices, compute its OBB center | Yes |
| h5py | 3.16.0 | Read supplied object/person masks | Yes (object masks only in final solver) |
| Pillow | 10.2.0 | Supplied texture decoding; inspection montages | Yes, indirectly through trimesh |
| pyrender | 0.1.45 | Installed version inspected and supplied rendering source reviewed; no rendering performed with it | No |
| Bash, standard filesystem utilities, ripgrep, git | Container-provided | Read task/source files, inspect installed-tool provenance, manage output files | No |
| NVIDIA driver / CUDA GPU | Container-provided, selected NVIDIA RTX 6000 Ada Generation; PyTorch CUDA runtime 12.4 | Rasterization on the single selected device | Yes |

No learned estimator or pretrained weights were run. The printed pattern used for matching comes exclusively from the supplied `object_mesh.glb`, and observations come exclusively from `video.mp4` and its released masks.

Exploratory methods tried: textured baseline/intermediate-pose rendering; differentiable texture plus silhouette registration with fixed camera range; SIFT correspondence validation; robust fixed-range feature registration. The silhouette/photometric optimizer was rejected because occlusion and mask-hole errors sometimes improved its objective while worsening independent printed-feature reprojection. Only the feature registration and its conservative acceptance checks are used by `solve.py`.

There are no downloaded-weight URLs or SHA256 hashes to list, and no newly cloned repositories. The only repository commit used directly is the supplied nvdiffrast checkout recorded above.
