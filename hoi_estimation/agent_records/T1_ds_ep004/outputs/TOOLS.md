# Tools and dependencies

## External tools used by `solve.py`

| Tool | Version | Use in final solution |
|---|---:|---|
| Python | 3.11.10 | Runtime |
| NumPy | 1.26.3 | Linear algebra, pose/quaternion processing, pooled observations |
| SciPy | 1.16.3 | Savitzky-Golay filtering, SO(3) rotations, SLERP |
| OpenCV | 4.11.0 | Video decoding, mask resize/dilation/erosion, crop handling |
| h5py | 3.16.0 | Reading the released SAM2 masks |
| PyTorch | 2.5.1+cu124 | GPU tensors for nvdiffrast |
| trimesh | 5.1.1 | Exact GLB mesh and texture loading |
| nvdiffrast | installed in container | Batched silhouette rasterization for translation fitting |
| Pyrender | 0.1.45 | Textured GLB rendering for pooled static-pose fitting |
| PyOpenGL / EGL | installed in container | Headless Pyrender backend |

No external git repository, model weight, or pip package was downloaded for the
final solution. No network access is required.

## Provided data/tools inspected or used as evidence

- CARI4D reconstruction bundle and intermediate FoundationPose/CoCoNet outputs
  supplied in `inputs/reconstruction.npz` and `/pipeline`. The final script uses
  only the released rotations in `reconstruction.npz` as hypotheses; it does not
  rerun FoundationPose, CoCoNet, SAM2, or MoGe-2.
- Provided `object_mesh.glb`, `camera.json`, `ground_plane.json`, `masks.h5`, and
  `video.mp4`.
- Development checks additionally used the provided `/pipeline` rendered crops
  and intermediate pose arrays for diagnosis only; they are not read by
  `solve.py`.

## Reproducibility notes

- GPU: one NVIDIA RTX 6000 Ada Generation, well below the 9 GB process budget.
- Final wall-clock runtime: 1m18s.
- Seeds: NumPy and PyTorch are seeded to zero.
- The output is recomputed from `inputs/` on every run; no scratch candidate table
  or fitted correction array is loaded.
