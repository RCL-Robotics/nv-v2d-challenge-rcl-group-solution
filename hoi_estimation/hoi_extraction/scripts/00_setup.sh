#!/usr/bin/env bash
# One-time setup: build docker images and download public weights (SAM2, Grounding DINO).
# CARI4D weights are downloaded automatically on first inference run (gated HF repos).
set -euo pipefail
RECON=${V2D_RECON:-/home/hanzhou/Projects/v2d/nv-v2d-challenge-codebase/reconstruction}
WEIGHTS=${V2D_WEIGHTS:-/mnt/nas/Group/hanzhou/v2d/weights}
cd "$RECON" && source .venv/bin/activate

python -m v2d.sam2.docker.build
python -m v2d.cari4d.docker.build
python -m v2d.grounding_dino.docker.build

python -m v2d.sam2.docker.run_download_weights --output_dir "$WEIGHTS/sam2"
python -m v2d.grounding_dino.docker.run_download_weights --output_dir "$WEIGHTS/grounding_dino"
echo "setup done"
