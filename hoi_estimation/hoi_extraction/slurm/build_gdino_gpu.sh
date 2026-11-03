#!/bin/bash
#SBATCH --job-name=v2d-build-gdino
#SBATCH --output=/l/users/han.zhou/v2d/logs/%x-%j.out
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

set -euo pipefail
# fakeroot build needs chown-capable local tmpfs; Lustre rejects lchown.
BTMP=/tmp/han.zhou/v2d-build
mkdir -p "$BTMP/cache" "$BTMP/tmp"
export SINGULARITY_CACHEDIR=$BTMP/cache
export SINGULARITY_TMPDIR=$BTMP/tmp
mkdir -p /l/users/han.zhou/v2d/images

cd /l/users/han.zhou/v2d/code/reconstruction

BASE=/l/users/han.zhou/v2d/images/base_pytorch2.5.1-cu124-devel.sif
if [ ! -s "$BASE" ]; then
  echo "=== pulling base image (with retries; docker CDN resets intermittently) ==="
  for t in 1 2 3 4 5; do
    rm -f $BTMP/base.sif; singularity pull --force $BTMP/base.sif docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel && break
    echo "pull attempt $t failed"
  done
  [ -s $BTMP/base.sif ] || { echo "FATAL: base pull failed"; exit 1; }
  cp $BTMP/base.sif "$BASE"
fi

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
