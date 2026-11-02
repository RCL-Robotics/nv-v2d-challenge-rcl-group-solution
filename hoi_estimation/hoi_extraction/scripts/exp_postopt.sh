#!/usr/bin/env bash
# Re-run CARI4D stage 7 (contact-guided refinement) on an experiment's CoCoNet bundle with extra
# refinement flags; results go to <v2d>/experiments/<out_exp>/<episode>/refined.pth.
# Usage: CUDA_VISIBLE_DEVICES=1 exp_postopt.sh <src_exp> <out_exp> "<extra postopt flags>" <episode idx ...>
#   e.g. exp_postopt.sh E2_reregister S2_sym_rot "--symmetric-object --optimize-object-rotation" 17 12 8
set -uo pipefail
SRC_EXP=$1; OUT_EXP=$2; FLAGS=$3; shift 3
V2D=${V2D_ROOT:?set V2D_ROOT to the data root (reproducibility/README.md)}
RUN=${V2D_RUN:?set V2D_RUN to the local run root of run_local.sh}
SIF=$V2D/images/v2d_cari4d.sif
APPTAINER=${APPTAINER:-apptainer}
W=$V2D/weights
SRC=/workspace/v2d_cari4d/lib/cari4d
m=$V2D/code/reconstruction/modules

sx() {  # same container setup as exp_fp_mode.sh
  APPTAINERENV___EGL_VENDOR_LIBRARY_FILENAMES=$V2D/build/10_nvidia_egl.json \
  $APPTAINER exec --nv -B "$V2D:$V2D" -B "$RUN:$RUN" \
    -B "$m/v2d_cari4d/lib/cari4d/lib_mhr/assets:/workspace/v2d_cari4d/lib/cari4d/lib_mhr/assets" \
    -B "$m/v2d_sam3d_body/assets:/workspace/v2d_sam3d_body/assets" \
    "$SIF" bash -c 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
      export PYTHONPATH='"$SRC"':/workspace/v2d_sam3d_body/lib:/workspace/v2d_foundation_pose/lib/FoundationPose:${PYTHONPATH:-}
      export HF_HOME='"$W"'/hf_home TORCH_HOME='"$W"'/sam3d_body/torch_home SAM3D_BODY_ROOT=/workspace/v2d_sam3d_body/lib
      export MHR_ASSETS_ROOT='"$W"'/sam3d_body PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      export OPENCV_IO_ENABLE_OPENEXR=1 PYOPENGL_PLATFORM=egl HDF5_USE_FILE_LOCKING=FALSE PYTHONUNBUFFERED=1
      cd '"$SRC"' && exec "$@"' _ "$@"
}

for i in "$@"; do
  EP=$(printf "episode_%06d" "$i")
  LOC=$RUN/out/$EP
  OUTD=$V2D/experiments/$OUT_EXP/$EP
  mkdir -p "$OUTD"
  if [ -s "$OUTD/refined.pth" ]; then echo "[$OUT_EXP] $EP done, skip"; continue; fi
  linked=0
  if [ ! -e "$LOC" ]; then mkdir -p "$RUN/out" && ln -s "$V2D/outputs/cari4d/$EP" "$LOC" && linked=1; fi
  echo "[$OUT_EXP] $EP start $(date '+%F %T')"
  t0=$(date +%s)
  # full clip first; windowed refinement if the clip does not fit in GPU memory
  for bs in 0 384; do
    sx python -m learning.training.mhr_opt_refineout --bundle "$V2D/experiments/$SRC_EXP/$EP/coconet.pth" \
      --object-mesh "$LOC/export/$EP/object_mesh/output_aligned.glb" --out "$OUTD/refined.pth" --mode smplh_parity \
      --num-steps 300 --batch-size $bs --device cuda --postopt-checkpoint "$OUTD/refinement_checkpoint.pth" \
      --mhr-assets-root "$W/sam3d_body" --w-temporal 100.0 --w-human-pose-prior 200.0 \
      --contact-activation-distance-m 0.05 --report-every 10 --diagnostics-every 500 $FLAGS > "$OUTD/stage7_bs$bs.log" 2>&1 && break
  done
  echo "[$OUT_EXP] $EP $(( $(date +%s) - t0 ))s $( [ -s $OUTD/refined.pth ] && echo OK || echo FAILED )"
  [ $linked = 1 ] && [ -L "$LOC" ] && rm "$LOC"
done
echo "EXP_DONE $OUT_EXP"
