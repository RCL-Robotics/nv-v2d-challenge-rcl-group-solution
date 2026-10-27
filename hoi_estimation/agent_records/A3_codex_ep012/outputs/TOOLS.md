# Tools and dependencies

No packages, repositories, model weights, or data were downloaded. No network requests or external pose models were used. All dependencies were already installed in the container. The deterministic `solve.py` requires no GPU, `/pipeline`, `/weights`, `/code`, `/work/scratch`, previous output files, or downloaded files. It applies an explicit, commented table of accepted corrections fitted from the supplied inputs. `fit_refinement.py` retains the full GPU fitting procedure for audit. The table is embedded in the solver, not loaded from a separate cache.

| Tool / package | Version / revision | Use | Needed by solve.py / fitting procedure |
|---|---|---|---|
| Python | 3.11.10, conda-forge, GCC 13.3.0 | Scripts | Yes / Yes |
| NumPy | 1.26.3 | Pose arithmetic, NPZ input/output | Yes / Yes |
| SciPy | 1.16.3 | Rotation arithmetic, temporal median/Gaussian filters | Yes / Yes |
| OpenCV (`opencv-python`) | 4.11.0.86 | Video decoding, masks, Canny edges, distance transforms | No / Yes |
| trimesh | 5.1.1 | Exact GLB loading with `force="scene", process=False).to_geometry()` | No / Yes |
| h5py | 3.16.0 | Released SAM2 masks | No / Yes |
| PyTorch | 2.5.1+cu124 | GPU rasterization inputs, pose gradients, Adam | No / Yes |
| nvdiffrast | 0.4.0; existing source `/workspace/third_party/nvdiffrast`, git `253ac4fcea7de5f396371124af597e6cc957bfae` | Differentiable rasterization and silhouette antialiasing | No / Yes |
| Pillow | 10.2.0 | Diagnostic contact sheets, texture inspection; also used by trimesh to load GLB textures | No / Yes, through trimesh |
| Bash / standard Unix utilities | Container supplied | Launch scripts, inspect files and package metadata | No / No |
| Git | Container supplied | Read revision of existing nvdiffrast installation | No / No |
| NVIDIA driver / CUDA | Driver 580.178.04; torch CUDA runtime 12.4 | One selected RTX 6000 Ada GPU (`CUDA_VISIBLE_DEVICES=0`) | No / Yes |
| `nvidia-smi` | Driver supplied | GPU monitoring | No / No |

The installed versions of pyrender (0.1.45), Open3D (0.20.0), imageio (2.38.0), networkx (3.2.1), lxml (6.1.3), and pyglet (2.1.16) were queried during environment inspection. They were not used for estimation or rendering and are not direct solver dependencies.

No newly cloned git repositories and no downloaded weight files exist for this task, so there are no downloaded-weight URLs or SHA256 hashes to list. The supplied textured object mesh is input data, not a downloaded model weight. Rendering uses geometry only; its texture was inspected in an exploratory diagnostic.

Run offline in the supplied environment:

```bash
python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
```
