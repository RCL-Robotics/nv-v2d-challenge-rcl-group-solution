#!/usr/bin/env bash
# Run Track-2 HOI extraction on this workstation's GPUs with apptainer (unprivileged).
# Heavy per-episode I/O goes to local disk; finished (PASS) episodes are moved to NAS.
# Usage: run_local.sh "<gpu ids, e.g. 0 1>" [episode idx ...]   (default: all in episodes.tsv)
set -uo pipefail
export V2D_ROOT=${V2D_ROOT:?set V2D_ROOT to the data root (reproducibility/README.md)}
export V2D_RUN=${V2D_RUN:?set V2D_RUN to a local scratch directory for the per-episode runs}
export WORK_ROOT=$V2D_RUN/work
export OUT_ROOT=$V2D_RUN/out
export SIF_EXEC=${APPTAINER:-apptainer}
NAS_OUT=$V2D_ROOT/outputs/cari4d
LOGS=$V2D_ROOT/logs
read -ra GPUS <<< "$1"; shift
if [ $# -gt 0 ]; then EPS=("$@"); else mapfile -t EPS < <(cut -f1 "$V2D_ROOT/code/pipeline/episodes.tsv"); fi
mkdir -p "$WORK_ROOT" "$OUT_ROOT" "$NAS_OUT" "$LOGS"

worker() {
  local gpu=$1; shift
  for i in "$@"; do
    local EP; EP=$(printf "%06d" "$i")
    if grep -qs '"verdict": "PASS"' "$NAS_OUT/episode_$EP/pipeline_report.json"; then
      echo "[gpu $gpu] episode $EP already PASS on NAS, skip"; continue
    fi
    echo "[gpu $gpu] episode $EP start $(date '+%F %T')"
    if CUDA_VISIBLE_DEVICES=$gpu bash "$V2D_ROOT/code/pipeline/run_episode.sh" "$i" > "$LOGS/episode_${EP}.local.log" 2>&1 \
       && grep -qs '"verdict": "PASS"' "$OUT_ROOT/episode_$EP/pipeline_report.json"; then
      # move results to NAS, keep masks/prompts alongside, then free local disk
      rsync -a "$OUT_ROOT/episode_$EP/" "$NAS_OUT/episode_$EP/" \
        && rsync -a --exclude='*.0.color.mp4' "$WORK_ROOT/episode_$EP/" "$NAS_OUT/episode_$EP/work/" \
        && rm -rf "$OUT_ROOT/episode_$EP" "$WORK_ROOT/episode_$EP"
      echo "[gpu $gpu] episode $EP OK $(date '+%F %T')"
    else
      echo "[gpu $gpu] episode $EP FAILED $(date '+%F %T') (local files kept for resume)"
    fi
  done
}

pids=()
NG=${#GPUS[@]}
for g in $(seq 0 $((NG-1))); do
  slice=()
  for k in $(seq $g $NG $((${#EPS[@]}-1))); do slice+=("${EPS[$k]}"); done
  worker "${GPUS[$g]}" "${slice[@]}" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo RUN_LOCAL_DONE
