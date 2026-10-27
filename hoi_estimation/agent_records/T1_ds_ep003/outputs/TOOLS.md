# Tools

The solution is self-contained and requires no downloaded code or weights beyond the container's preinstalled stack.

| Tool | Version | Role | Needed by solve.py |
|---|---|---|---|
| Python | 3.11.10 | runtime | yes |
| PyTorch | 2.5.1+cu124 | tensor/rasterization runtime | yes |
| nvdiffrast | 0.4.0 | differentiable/hard rasterization and texture lookup | yes |
| trimesh | 5.1.1 | exact GLB mesh and UV texture loading | yes |
| NumPy | 1.26.3 | numerical linear algebra | yes |
| SciPy | 1.16.3 | sparse trajectory smoother, rotation utilities | yes |
| OpenCV | 4.11.0 | video decoding | yes |
| h5py | 3.16.0 | mask and depth I/O | yes |
| FoundationPose (NVlabs PyTorch backend) | preinstalled PyTorch weights, runs dated 2024-01-11 and 2023-10-28 | independent re-registration was attempted as a cross-check but was not used in the final deterministic path; supplied FoundationPose poses are used | no |

No external repository, model weight, or network download was introduced. The supplied `/weights` and `/pipeline` assets are read-only inputs to the reconstruction, not downloaded by this solution.
