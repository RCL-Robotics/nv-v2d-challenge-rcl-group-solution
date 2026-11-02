#!/usr/bin/env bash
# Re-run CARI4D stage 5 (FoundationPose) + stage 6 (CoCoNet) with a different FoundationPose
# configuration, writing results to an experiment directory (original outputs untouched).
# Env passed into the container (e.g. V2D_FP_REREGISTER=1) reaches FoundationPose.
# Usage: CUDA_VISIBLE_DEVICES=1 exp_fp_mode.sh <exp_name> "<extra fp flags>" <episode idx ...>
#   e.g. exp_fp_mode.sh E1_every_frame "--register-every-frame" 3 13 15
set -uo pipefail
EXP=$1; FP_FLAGS=$2; shift 2
V2D=${V2D_ROOT:?set V2D_ROOT to the data root (reproducibility/README.md)}
RUN=${V2D_RUN:?set V2D_RUN to the local run root of run_local.sh}
SIF=$V2D/images/v2d_cari4d.sif
APPTAINER=${APPTAINER:-apptainer}
W=$V2D/weights
CFG=$(ls $W/cari4d/*/resolved_config.yaml); CKPT=$(ls $W/cari4d/*/step200000.pth)
SRC=/workspace/v2d_cari4d/lib/cari4d
m=$V2D/code/reconstruction/modules

sx() {  # same container setup as run_episode.sh, plus the env run_inference gives its stages
  APPTAINERENV___EGL_VENDOR_LIBRARY_FILENAMES=$V2D/build/10_nvidia_egl.json \
  $APPTAINER exec --nv -B "$V2D:$V2D" -B "$RUN:$RUN" \
    -B "$m/v2d_cari4d/lib/cari4d/lib_mhr/assets:/workspace/v2d_cari4d/lib/cari4d/lib_mhr/assets" \
    -B "$m/v2d_sam3d_body/assets:/workspace/v2d_sam3d_body/assets" \
    -B "$V2D/build/patches/run_foundationpose_mhr_export.py:$SRC/prep/run_foundationpose_mhr_export.py" \
    "$SIF" bash -c 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
      export PYTHONPATH='"$SRC"':/workspace/v2d_sam3d_body/lib:/workspace/v2d_foundation_pose/lib/FoundationPose:${PYTHONPATH:-}
      export HF_HOME='"$W"'/hf_home TORCH_HOME='"$W"'/sam3d_body/torch_home SAM3D_BODY_ROOT=/workspace/v2d_sam3d_body/lib
      export MHR_ASSETS_ROOT='"$W"'/sam3d_body FOUNDATIONPOSE_WEIGHTS_DIR='"$W"'/foundationpose/nvlabs_pytorch
      export OPENCV_IO_ENABLE_OPENEXR=1 PYOPENGL_PLATFORM=egl HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
      cd '"$SRC"' && exec "$@"' _ "$@"
}

for i in "$@"; do
  EP=$(printf "episode_%06d" "$i")
  NAS_EP=${SRC_ROOT:-$V2D/outputs/cari4d}/$EP
  LOC=$RUN/out/$EP
  # SRC_ROOT (e.g. the M1 rerun's NAS output root) already holds valid absolute paths: use it in place
  [ -n "${SRC_ROOT:-}" ] && LOC=$NAS_EP
  OUTD=$V2D/experiments/$EXP/$EP
  TMP=$V2D/experiments/.tmp/$EXP/$EP   # on NAS: the shared local root disk is nearly full
  mkdir -p "$OUTD" "$TMP"
  if [ -s "$OUTD/coconet.pth" ]; then echo "[$EXP] $EP done, skip"; continue; fi
  restored=0
  if [ ! -e "$LOC" ]; then
    # stage inputs reference absolute local paths: expose the read-only NAS outputs at that path
    mkdir -p "$RUN/out" && ln -s "$NAS_EP" "$LOC" && restored=1
  fi
  echo "[$EXP] $EP start $(date '+%F %T')"
  t0=$(date +%s)
  if sx python prep/run_foundationpose_mhr_export.py "$LOC/export/$EP" --depth-root "$LOC/depth/moge2/aligned.h5" \
        --out-file "$TMP/fp.pkl" --camera-id 0 --iteration 5 --max-attempts 5 $FP_FLAGS \
        --no-first-usable-frame-gt-rotation-oracle > "$OUTD/stage5.log" 2>&1; then
    t1=$(date +%s)
    sx python tools/run_mhr_wild_inference.py "$LOC/export/$EP" --depth-h5 "$LOC/depth/moge2/aligned.h5" \
        --mhr-init "$LOC/mhr_init/$EP.pkl" --foundationpose-file "$TMP/fp.pkl" --config "$CFG" --checkpoint "$CKPT" \
        --output "$TMP/coconet.pth" --stride 96 --render-batch-size 32 --crop-workers 8 --crop-buffer-count 2 \
        --input-cache "$TMP/materialized_inputs.h5" --device cuda --offline-supervision-contract > "$OUTD/stage6.log" 2>&1
    t2=$(date +%s)
    cp "$TMP/fp.pkl" "$OUTD/" 2>/dev/null; cp "$TMP/coconet.pth" "$OUTD/" 2>/dev/null
    echo "[$EXP] $EP stage5 $((t1-t0))s stage6 $((t2-t1))s $( [ -s $OUTD/coconet.pth ] && echo OK || echo COCONET_FAILED )"
  else
    echo "[$EXP] $EP STAGE5_FAILED (see $OUTD/stage5.log)"
  fi
  rm -rf "$TMP"
  [ $restored = 1 ] && [ "${KEEP_LOCAL:-0}" != 1 ] && [ -L "$LOC" ] && rm "$LOC"  # remove only our symlink
done
echo "EXP_DONE $EXP"
