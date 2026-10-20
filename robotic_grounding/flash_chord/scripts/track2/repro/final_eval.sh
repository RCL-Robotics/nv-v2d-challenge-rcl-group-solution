#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Final evaluation of one arm, as for the submitted C3r2 Tier-1 file:
#   0. check the kit and the stock runtime against the line spec (check_eval_inputs.py: runtime commit, evaluator
#      digest, KIT_MANIFEST.json);
#   1. a view directory <eval root>/<ARM>/epNN -> <runs>/<ARM>/epNN (keeps exports and results out of the run dirs);
#   2. export each episode's FINAL checkpoint (final.json) at the line's tolerance (export_all.sh);
#   3. check that the absolute task.parquet path stored in every exported checkpoint exists (check_task_paths.py;
#      the stock harness only remaps paths under /workspace, see README);
#   4. the official kit evaluator on the untouched stock runtime, one episode per call, one job per GPU, outside any
#      MPS daemon, through the in-repo scripts/track2/eval_all.py (--checkpoint-from-final, --skip-pack);
#   5. the per-episode table with averages (summarize.py) -> <eval root>/<ARM>_final.csv.
#
# usage: final_eval.sh [--line FILE] [--runs DIR] [--eval-root DIR] [--episodes all|0,3,...] [--gpus 0,1,...]
#                      [--export-only] [--dry-run]
#   --export-only  stop after steps 0-3
#   env: V2D_WORK V2D_DATA V2D_MOTION V2D_RUNTIME V2D_KIT FC_PYTHON KIT_PYTHON
#        SKIP_INPUT_CHECK=1 (skip step 0)  SKIP_PATH_CHECK=1 (skip step 3)
# Episodes without a valid final.json (or model_<MAX_ITERATIONS>) are reported missing and the rest are evaluated.
# Exits non-zero when an export, the path check, eval_all.py or the summary fails.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); FC=$(cd "$HERE/../../.." && pwd)
abs(){ case $1 in /*|"") printf '%s' "$1";; *) printf '%s' "$PWD/$1";; esac; }   # relative to the caller's cwd
LINE=$HERE/lines/c3r2_tier1.env; RUNS=""; OUTROOT=""; EPISODES=all; DRY=0; GPUS=${GPUS:-0}; EXPORT_ONLY=0
while [ $# -gt 0 ]; do case "$1" in
  --line) LINE=$2; shift 2;; --runs) RUNS=$2; shift 2;; --eval-root) OUTROOT=$2; shift 2;;
  --episodes) EPISODES=$2; shift 2;; --gpus) GPUS=$2; shift 2;; --export-only) EXPORT_ONLY=1; shift;;
  --dry-run) DRY=1; shift;;
  *) echo "unknown argument $1" >&2; exit 2;; esac; done
# shellcheck disable=SC1090
source "$LINE"
RUNS=${RUNS:-${V2D_WORK:?set V2D_WORK or pass --runs}/runs}; OUTROOT=${OUTROOT:-$V2D_WORK/eval}
for v in V2D_DATA V2D_MOTION V2D_RUNTIME V2D_KIT KIT_PYTHON; do [ -n "${!v:-}" ] || { echo "set $v" >&2; exit 2; }; done
RUNS=$(abs "$RUNS"); OUTROOT=$(abs "$OUTROOT")
V2D_DATA=$(abs "$V2D_DATA"); V2D_MOTION=$(abs "$V2D_MOTION"); V2D_RUNTIME=$(abs "$V2D_RUNTIME"); V2D_KIT=$(abs "$V2D_KIT")
case $KIT_PYTHON in */*) KIT_PYTHON=$(abs "$KIT_PYTHON");; esac
export V2D_DATA V2D_MOTION V2D_RUNTIME V2D_KIT KIT_PYTHON
source "$HERE/env/fc_env.sh"
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
export MAX_ITERATIONS
SRC=$RUNS/$ARM; VIEW=$OUTROOT/$ARM; OUT=$OUTROOT/${ARM}_final.csv; LOG=$OUTROOT/${ARM}_final_eval.log
if [ "$EPISODES" = all ]; then EPS=$(seq -w 0 29 | tr '\n' ' ')
else EPS=$(for e in ${EPISODES//,/ }; do printf '%02d ' "$((10#$e))"; done); fi
EPS_CSV=$(for e in $EPS; do echo $((10#$e)); done | paste -sd, -)
run(){ # print (dry run) or run and log a command
  if [ $DRY = 1 ]; then printf '[eval] '; printf '%q ' "$@"; echo; return 0; fi
  "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"
}
log(){ [ $DRY = 1 ] && { echo "[eval] # $*"; return 0; }; echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
[ $DRY = 1 ] || mkdir -p "$VIEW"
if [ "${SKIP_INPUT_CHECK:-0}" != 1 ]; then
  run python -B "$HERE/check_eval_inputs.py" --line "$LINE" --kit "$V2D_KIT" --runtime "$V2D_RUNTIME" \
    || { log "the kit or the stock runtime does not match the line spec (see above; SKIP_INPUT_CHECK=1 skips)"; exit 1; }
fi
finals=""; missing=""
for e in $EPS; do
  if it=$(python "$HERE/queue_helper.py" final "$SRC/ep$e" 2>/dev/null); then finals="$finals $e:$it"; else missing="$missing $e"; fi
  if [ $DRY = 1 ]; then echo "[eval] ln -sfn $SRC/ep$e $VIEW/ep$e"; else ln -sfn "$SRC/ep$e" "$VIEW/ep$e"; fi
done
log "final checkpoints (episode:iteration):${finals:- none yet}; without final.json:${missing:- none} (code $FC)"
if [ $DRY = 1 ]; then bash "$HERE/export_all.sh" "$VIEW" --tolerance "$EXPORT_TOLERANCE" --jobs 4 --episodes "$EPS" --dry-run
else run bash "$HERE/export_all.sh" "$VIEW" --tolerance "$EXPORT_TOLERANCE" --jobs 4 --episodes "$EPS" \
       || { log "export failed (see above and <view>/export/epNN/export.log)"; exit 1; }; fi
if [ "${SKIP_PATH_CHECK:-0}" != 1 ]; then
  if [ $DRY = 1 ]; then run python "$HERE/check_task_paths.py" "$VIEW/export"
  else run python "$HERE/check_task_paths.py" "$VIEW/export" || { log "stored task.parquet paths are missing (see above)"; exit 1; }; fi
fi
[ $EXPORT_ONLY = 1 ] && { log "export done (--export-only)"; exit 0; }
export PYTHONPATH="$FC/src" OMP_NUM_THREADS=4
EVAL=(python scripts/track2/eval_all.py --arm-dir "$VIEW" --arm "$ARM" --checkpoint-from-final --iteration "$MAX_ITERATIONS"
      --episodes "$EPS_CSV" --gpus "$GPUS" --jobs-per-gpu 1 --skip-pack --export-tolerance "$EXPORT_TOLERANCE"
      --results "$VIEW/results" --world-count "$EVAL_WORLD_COUNT" --kit-root "$V2D_KIT" --stock-runtime "$V2D_RUNTIME"
      --dataset-root "$V2D_DATA" --manifest "$V2D_MOTION/whole_body/soma/track2_manifest.json" --code "$FC"
      --python "$(command -v python || echo python)" --pack-python "$KIT_PYTHON")
erc=0
if [ $DRY = 1 ]; then printf '[eval] cd %q && nice -n 10 ' "$FC"; printf '%q ' "${EVAL[@]}"; echo
else ( cd "$FC" && nice -n 10 "${EVAL[@]}" ) >> "$LOG" 2>&1; erc=$?; log "eval_all exit $erc"; fi
run python "$HERE/summarize.py" --arm-dir "$VIEW" --results "$VIEW/results" --out "$OUT" --kit "$V2D_KIT"; src=$?
if [ "$erc" != 0 ] || [ "$src" != 0 ]; then log "FAILED: eval_all exit $erc, summarize exit $src (log $LOG)"; exit 1; fi
log "DONE $OUT"
