#!/usr/bin/env bash
# Full HOI extraction for one Track-2 episode on CSCC using singularity images.
# Usage: run_episode.sh <episode_index_int>
set -euo pipefail

IDX=$1
EP=$(printf "%06d" "$IDX")

V2D=${V2D_ROOT:?set V2D_ROOT to the data root}
DS=$V2D/dataset/track_2/tier_1_multiview_caption
# WORK_ROOT/OUT_ROOT let a local run keep heavy per-episode I/O off the shared store
WORK=${WORK_ROOT:-$V2D/work}/episode_$EP
OUT=${OUT_ROOT:-$V2D/outputs/cari4d}
SIF_EXEC=${SIF_EXEC:-singularity}
CARI4D_SIF=$V2D/images/v2d_cari4d.sif
GDINO_SIF=$V2D/images/v2d_gdino.sif

# --nv binds host glvnd libs (libGLdispatch etc., built for host glibc 2.39) which
# break inside the glibc-2.35 images. Prefer the container's own glvnd, keep host
# NVIDIA driver libs from /.singularity.d/libs, and point EGL at the NVIDIA vendor.
sx() {
  local sif=$1; shift
  local m=$V2D/code/reconstruction/modules
  local extra=()
  [ -n "${WORK_ROOT:-}" ] && extra+=(-B "$WORK_ROOT:$WORK_ROOT")
  [ -n "${OUT_ROOT:-}" ] && extra+=(-B "$OUT_ROOT:$OUT_ROOT")
  # LFS assets missing from the image build (fork had no LFS objects); bind the verified copies.
  # Patched FoundationPose export: NaN instead of inf centroid distance for fully hidden objects.
  SINGULARITYENV___EGL_VENDOR_LIBRARY_FILENAMES=$V2D/build/10_nvidia_egl.json \
  $SIF_EXEC exec --nv -B "$V2D:$V2D" "${extra[@]}" \
    -B "$m/v2d_cari4d/lib/cari4d/lib_mhr/assets:/workspace/v2d_cari4d/lib/cari4d/lib_mhr/assets" \
    -B "$m/v2d_sam3d_body/assets:/workspace/v2d_sam3d_body/assets" \
    -B "$V2D/build/patches/run_foundationpose_mhr_export.py:/workspace/v2d_cari4d/lib/cari4d/prep/run_foundationpose_mhr_export.py" \
    "$sif" \
    bash -c 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH; exec "$@"' _ "$@"
}

# episode metadata from episodes.tsv: idx, sequence_id, object, mesh, frames
LINE=$(awk -F'\t' -v i="$IDX" '$1==i' "$V2D/code/pipeline/episodes.tsv")
OBJ=$(echo "$LINE" | cut -f3)
MESH_REL=$(echo "$LINE" | cut -f4)
FRAMES=$(echo "$LINE" | cut -f5)
OBJ_PROMPT=${OBJ//_/ }

VIDEO=$DS/videos/chunk-000/observation.images.exo_camera/episode_$EP.mp4
MESH=$DS/$MESH_REL
mkdir -p "$WORK" "$OUT"
# hard link (not symlink): CARI4D resolve()s paths and requires the *.0.color.mp4 name.
# Never re-copy an existing input: CARI4D stage caches key on its size + mtime.
if [ ! -f "$WORK/episode_$EP.0.color.mp4" ]; then
  ln -f "$VIDEO" "$WORK/episode_$EP.0.color.mp4" 2>/dev/null || cp -p "$VIDEO" "$WORK/episode_$EP.0.color.mp4"
fi

echo "== episode $EP  object=$OBJ  frames=$FRAMES  mesh=$MESH_REL"

# 1. prompts via Grounding DINO
if [ ! -f "$WORK/prompts.json" ]; then
  sx "$GDINO_SIF" \
    python "$V2D/code/pipeline/gen_prompt_one.py" \
      --video_path "$WORK/episode_$EP.0.color.mp4" \
      --object_prompt "$OBJ_PROMPT" \
      --model_dir "$V2D/weights/grounding_dino" \
      --output_path "$WORK/prompts.json"
fi

# 2. SAM2 video -> masks
if [ ! -d "$WORK/masks/0" ] || [ ! -d "$WORK/masks/1" ]; then
  sx "$CARI4D_SIF" \
    python -m v2d.sam2.lib.video_to_masks \
      --video_path "$WORK/episode_$EP.0.color.mp4" \
      --prompts_path "$WORK/prompts.json" \
      --masks_dir "$WORK/masks" \
      --weights_dir "$V2D/weights/sam2"
fi

# 3. pack masks to H5
if [ ! -f "$WORK/masks_k0.h5" ]; then
  sx "$CARI4D_SIF" \
    python -m v2d.cari4d.lib.pack_masks \
      --video_path "$WORK/episode_$EP.0.color.mp4" \
      --human_masks_path "$WORK/masks/0" \
      --object_masks_path "$WORK/masks/1" \
      --output_path "$WORK/masks_k0.h5"
fi

# 4. CARI4D reconstruction (weights pre-staged, skip download).
# Full-clip contact refinement can exceed 48 GB on long clips; if the run fails, retry with
# windowed refinement (finished stages are reused from their markers).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cari4d() {
  sx "$CARI4D_SIF" \
    python -m v2d.cari4d.lib.run_inference \
      --video_path "$WORK/episode_$EP.0.color.mp4" \
      --mask_h5_path "$WORK/masks_k0.h5" \
      --object_mesh_path "$MESH" \
      --weights_path "$V2D/weights" \
      --output_dir "$OUT" \
      --expected_frames "$FRAMES" \
      --skip_weight_download "$@"
}
if ! cari4d; then
  echo "== episode $EP: retrying with --postopt_batch_size ${POSTOPT_FALLBACK_BATCH:-384}"
  cari4d --postopt_batch_size "${POSTOPT_FALLBACK_BATCH:-384}"
fi

echo "== episode $EP done"
