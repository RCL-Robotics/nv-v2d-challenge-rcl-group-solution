#!/bin/bash
#SBATCH --job-name=v2d-smoke-ep0
#SBATCH --output=/l/users/han.zhou/v2d/logs/%x-%j.out
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=gpu-debug-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --exclude=gpu-05

set -euo pipefail
cd /l/users/han.zhou/v2d
nvidia-smi
# image sanity (gpu-05 GPUs were unusable during the build job, so checks run here)
I=/l/users/han.zhou/v2d/images
singularity exec --nv $I/v2d_cari4d.sif python -c \
  "import torch, pytorch3d, nvdiffrast, kaolin; x=torch.randn(8,device='cuda'); print('cari4d imports+cuda OK', torch.__version__, kaolin.__version__, x.sum().item())"
singularity exec --nv $I/v2d_gdino.sif python -c \
  "import groundingdino, torch; print('gdino imports OK', torch.cuda.is_available())"
# GL checks with the same glvnd fix run_episode.sh uses
gl() { SINGULARITYENV___EGL_VENDOR_LIBRARY_FILENAMES=/l/users/han.zhou/v2d/build/10_nvidia_egl.json \
  singularity exec --nv "$1" bash -c 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH; python -c "$0"' "$2"; }
gl $I/v2d_gdino.sif "import cv2; print('gdino cv2 OK', cv2.__version__)"
gl $I/v2d_cari4d.sif "import cv2, pyrender; r=pyrender.OffscreenRenderer(64,64); print('cari4d cv2 + EGL OK'); r.delete()" \
  || echo "WARN: EGL offscreen check failed; continuing to see whether the pipeline needs it"
bash /l/users/han.zhou/v2d/code/pipeline/run_episode.sh 0
echo SMOKE_EP0_DONE
