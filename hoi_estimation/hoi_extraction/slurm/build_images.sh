#!/bin/bash
#SBATCH --job-name=v2d-build-images
#SBATCH --output=/l/users/han.zhou/v2d/logs/%x-%j.out
#SBATCH --partition=cscc-cpu-p
#SBATCH --qos=cscc-cpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00

set -euo pipefail
export SINGULARITY_CACHEDIR=/l/users/han.zhou/v2d/build/cache
export SINGULARITY_TMPDIR=/l/users/han.zhou/v2d/build/tmp
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" /l/users/han.zhou/v2d/images

# def files reference modules/... relative to the reconstruction dir
cd /l/users/han.zhou/v2d/code/reconstruction

echo "=== building v2d_cari4d.sif ==="
singularity build --fakeroot /l/users/han.zhou/v2d/images/v2d_cari4d.sif \
  /l/users/han.zhou/v2d/build/v2d_cari4d.def

echo "=== building v2d_gdino.sif ==="
singularity build --fakeroot /l/users/han.zhou/v2d/images/v2d_gdino.sif \
  /l/users/han.zhou/v2d/build/v2d_gdino.def

echo "=== sanity checks ==="
singularity exec /l/users/han.zhou/v2d/images/v2d_cari4d.sif python -c \
  "import torch, pytorch3d, nvdiffrast, kaolin; print('cari4d imports OK', torch.__version__)"
singularity exec /l/users/han.zhou/v2d/images/v2d_gdino.sif python -c \
  "import groundingdino; print('gdino imports OK')"
echo BUILD_IMAGES_DONE
