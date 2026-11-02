#!/usr/bin/env bash
# Run the full off-the-shelf HOI extraction (SAM2 masks -> CARI4D) for one episode.
# Usage: 02_run_episode.sh <episode_index>   e.g. 02_run_episode.sh 0
set -euo pipefail

EP=$(printf "%06d" "$1")

DATASET=${V2D_DATASET:-${V2D_ROOT:?set V2D_ROOT to the data root (reproducibility/README.md)}/dataset/track_2/tier_1_multiview_caption}
WORK=${V2D_WORK:-${V2D_ROOT:?set V2D_ROOT to the data root (reproducibility/README.md)}/work}/episode_${EP}
OUT=${V2D_OUT:-${V2D_ROOT:?set V2D_ROOT to the data root (reproducibility/README.md)}/outputs/cari4d}
SAM2_WEIGHTS=${V2D_SAM2_WEIGHTS:-${V2D_ROOT:?set V2D_ROOT to the data root (reproducibility/README.md)}/weights/sam2}
CARI4D_WEIGHTS=${V2D_CARI4D_WEIGHTS:-${V2D_ROOT:?set V2D_ROOT to the data root (reproducibility/README.md)}/weights/cari4d}
RECON=${V2D_RECON:-$(cd "$(dirname "$0")/../../.." && pwd)/reconstruction}

cd "$RECON"
source .venv/bin/activate

VIDEO=$DATASET/videos/chunk-000/observation.images.exo_camera/episode_${EP}.mp4
MESH=$(python - "$DATASET" "$1" <<'EOF'
import json, sys
ds, idx = sys.argv[1], int(sys.argv[2])
for line in open(f"{ds}/meta/episodes_metadata.jsonl"):
    d = json.loads(line)
    if d["episode_index"] == idx:
        print(f"{ds}/{d['mesh']}")
        break
EOF
)
FRAMES=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$VIDEO")

mkdir -p "$WORK" "$OUT"
# CARI4D expects <sequence>.0.color.mp4 naming
ln -sf "$VIDEO" "$WORK/episode_${EP}.0.color.mp4"

echo "== episode $EP  frames=$FRAMES  mesh=$MESH"

# 1. SAM2 video -> masks (prompts.json from 01_gen_prompts.py)
if [ ! -d "$WORK/masks/0" ] || [ ! -d "$WORK/masks/1" ]; then
  python -m v2d.sam2.docker.run_video_to_masks \
    --video_path "$WORK/episode_${EP}.0.color.mp4" \
    --prompts_path "$WORK/prompts.json" \
    --masks_dir "$WORK/masks" \
    --weights_dir "$SAM2_WEIGHTS"
fi

# 2. Pack masks into CARI4D H5
if [ ! -f "$WORK/masks_k0.h5" ]; then
  python -m v2d.cari4d.docker.run_pack_masks \
    --video_path "$WORK/episode_${EP}.0.color.mp4" \
    --human_masks_path "$WORK/masks/0" \
    --object_masks_path "$WORK/masks/1" \
    --output_path "$WORK/masks_k0.h5"
fi

# 3. CARI4D inference (downloads pinned weights on first run; needs HF_TOKEN with
#    access to nvidia/cari4d_commercial + facebook/sam-3d-body-dinov3)
python -m v2d.cari4d.docker.run_inference \
  --video_path "$WORK/episode_${EP}.0.color.mp4" \
  --mask_h5_path "$WORK/masks_k0.h5" \
  --object_mesh_path "$MESH" \
  --weights_path "$CARI4D_WEIGHTS" \
  --output_dir "$OUT" \
  --expected_frames "$FRAMES"

echo "== episode $EP done -> $OUT"
