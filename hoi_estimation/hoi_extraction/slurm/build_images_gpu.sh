#!/bin/bash
#SBATCH --job-name=v2d-build-images
#SBATCH --output=/l/users/han.zhou/v2d/logs/%x-%j.out
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

set -euo pipefail
# fakeroot build needs chown-capable local tmpfs; Lustre rejects lchown.
BTMP=/tmp/han.zhou/v2d-build
mkdir -p "$BTMP/cache" "$BTMP/tmp"
export SINGULARITY_CACHEDIR=$BTMP/cache
export SINGULARITY_TMPDIR=$BTMP/tmp
mkdir -p /l/users/han.zhou/v2d/images

cd /l/users/han.zhou/v2d/code/reconstruction

echo "=== building v2d_cari4d.sif ==="
singularity build --fakeroot $BTMP/v2d_cari4d.sif \
  /l/users/han.zhou/v2d/build/v2d_cari4d.def
cp $BTMP/v2d_cari4d.sif /l/users/han.zhou/v2d/images/v2d_cari4d.sif

echo "=== building v2d_gdino.sif ==="
singularity build --fakeroot $BTMP/v2d_gdino.sif \
  /l/users/han.zhou/v2d/build/v2d_gdino.def
cp $BTMP/v2d_gdino.sif /l/users/han.zhou/v2d/images/v2d_gdino.sif

echo "=== sanity checks (GPU) ==="
nvidia-smi
singularity exec --nv /l/users/han.zhou/v2d/images/v2d_cari4d.sif python -c \
  "import torch, pytorch3d, nvdiffrast, kaolin; x=torch.randn(8,device='cuda'); print('cari4d imports+cuda OK', torch.__version__, x.sum().item())"
singularity exec --nv /l/users/han.zhou/v2d/images/v2d_gdino.sif python -c \
  "import groundingdino, torch; print('gdino imports OK', torch.cuda.is_available())"
echo BUILD_IMAGES_DONE
