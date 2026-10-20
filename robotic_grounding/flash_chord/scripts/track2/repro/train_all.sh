#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Train one FlashCHORD policy per Track 2 episode with the exact per-episode command of the C3r2 run, through a
# simple per-GPU queue (JOBS_PER_GPU slots per GPU; a slot starts the next episode when its job ends).
#
# usage: train_all.sh [--line FILE] [--runs DIR] [--episodes all|0,3,...] [--gpus 0,1,...] [--jobs-per-gpu K] [--dry-run]
#   --line   line spec (default lines/c3r2_tier1.env)       --runs  runs root (default $V2D_WORK/runs)
#   env: V2D_MOTION (HUMAN_MOTION_DATA_DIR with the retargeted episodes and track2_manifest.json)
#        FC_PYTHON (flashchord env python; see env/fc_env.sh)   WANDB_MODE=disabled  WANDB_ENTITY  WANDB_PROJECT
#        OMP_THREADS=4  POLL=30  START_GAP=15
# Per episode NN (sequence <seq>, object <obj>), from robotic_grounding/flash_chord with PYTHONPATH=<that>/src:
#   CUDA_VISIBLE_DEVICES=<gpu> python scripts/train_rl.py experiment=<EXPERIMENT> \
#     "task.parquet='<V2D_MOTION>/whole_body/soma/sequence_id=<seq>/robot_name=g1/data.parquet'" reset.seed=<RESET_SEED> \
#     output_dir=<runs>/<ARM>/epNN training.max_iterations=<MAX_ITERATIONS> logging.mode=<WANDB_MODE> \
#     [logging.entity=<WANDB_ENTITY>] [logging.project=<WANDB_PROJECT>] logging.experiment=<ARM> \
#     logging.run_name=epNN_<obj> logging.upload_checkpoints=false [scene.support=false]
#   scene.support=false when whole_body/reconstructed_stage/<seq>_support.usda has no `def Cylinder` / `def Cube`.
#   The quotes around the parquet path are part of the Hydra override (as recorded).
# A job is finished when <run>/final.json names its checkpoint (early stop or max_iterations; queue_helper.py done).
# A job that stops without one is started once more, resuming from the newest finite model_N (+ rsi_N); a run
# directory that holds checkpoints but no usable resume point is never started fresh (it would overwrite them).
# The original run packed 4 jobs per GPU on 8 GPUs under NVIDIA MPS, each job pinned to its own CPU slice; that
# packing only changes speed. --dry-run prints every command without running anything.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); FC=$(cd "$HERE/../../.." && pwd)   # robotic_grounding/flash_chord
abs(){ case $1 in /*|"") printf '%s' "$1";; *) printf '%s' "$PWD/$1";; esac; }   # relative to the caller's cwd
LINE=$HERE/lines/c3r2_tier1.env; RUNS=""; EPISODES=all; DRY=0
GPUS=${GPUS:-0}; K=${JOBS_PER_GPU:-1}
while [ $# -gt 0 ]; do case "$1" in
  --line) LINE=$2; shift 2;; --runs) RUNS=$2; shift 2;; --episodes) EPISODES=$2; shift 2;;
  --gpus) GPUS=$2; shift 2;; --jobs-per-gpu) K=$2; shift 2;; --dry-run) DRY=1; shift;;
  *) echo "unknown argument $1" >&2; exit 2;; esac; done
# shellcheck disable=SC1090
source "$LINE"
: "${V2D_MOTION:?set V2D_MOTION (derived motion data root)}"
RUNS=${RUNS:-${V2D_WORK:?set V2D_WORK or pass --runs}/runs}
RUNS=$(abs "$RUNS"); V2D_MOTION=$(abs "$V2D_MOTION")
WANDB_MODE=${WANDB_MODE:-disabled}; OMP=${OMP_THREADS:-4}; POLL=${POLL:-30}; START_GAP=${START_GAP:-15}
source "$HERE/env/fc_env.sh"
export PYTHONPATH="$FC/src" PYTHONDONTWRITEBYTECODE=1   # no __pycache__ in the source tree; numerics unchanged
HELPER=$HERE/queue_helper.py; HMD=$V2D_MOTION; ARMDIR=$RUNS/$ARM; QDIR=$ARMDIR/queue
MANIFEST=$HMD/whole_body/soma/track2_manifest.json
PYJSON=$(command -v python3 || command -v python)
log(){ echo "[$(date '+%F %T')] $*"; [ $DRY = 1 ] || echo "[$(date '+%F %T')] $*" >> "$QDIR/queue.log"; }
if [ ! -f "$MANIFEST" ]; then
  [ $DRY = 1 ] || { echo "missing $MANIFEST (run the data steps first)" >&2; exit 1; }
  echo "[train] (dry run) $MANIFEST does not exist yet; per episode NN the command is:"
  echo "[train] cd $FC && env CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=$FC/src WANDB_MODE=$WANDB_MODE python scripts/train_rl.py" \
       "experiment=$EXPERIMENT \"task.parquet='$HMD/whole_body/soma/sequence_id=<seq>/robot_name=g1/data.parquet'\"" \
       "reset.seed=$RESET_SEED output_dir=$ARMDIR/epNN training.max_iterations=$MAX_ITERATIONS logging.mode=$WANDB_MODE" \
       "logging.experiment=$ARM logging.run_name=epNN_<obj> logging.upload_checkpoints=false [scene.support=false]"
  exit 0
fi
# job list "ep seq obj", episode order
mapfile -t JOBS < <("$PYJSON" - "$MANIFEST" "$EPISODES" <<'PY'
import json, sys
manifest, episodes = json.load(open(sys.argv[1])), sys.argv[2]
wanted = None if episodes == "all" else {int(e) for e in episodes.replace(",", " ").split()}
for ep, seq, obj in sorted((int(r["episode_index"]), s, r["object"]) for s, r in manifest.items()):
    if wanted is None or ep in wanted:
        print(f"{ep:02d} {seq} {obj}")
PY
)
[ ${#JOBS[@]} -gt 0 ] || { echo "no episodes selected" >&2; exit 1; }

support_flag(){ # seq -> prints scene.support=false for a floor-only episode
  local sup="$HMD/whole_body/reconstructed_stage/${1}_support.usda"
  { [ -f "$sup" ] && grep -q -E "def (Cylinder|Cube)" "$sup"; } || echo scene.support=false
}
train_args(){ # ep seq obj -> one Hydra argument per line (the recorded order)
  local ep=$1 seq=$2 obj=$3
  printf '%s\n' "experiment=$EXPERIMENT" "task.parquet='$HMD/whole_body/soma/sequence_id=$seq/robot_name=g1/data.parquet'" \
    "reset.seed=$RESET_SEED" "output_dir=$ARMDIR/ep$ep" "training.max_iterations=$MAX_ITERATIONS" "logging.mode=$WANDB_MODE"
  [ -n "${WANDB_ENTITY:-}" ] && printf '%s\n' "logging.entity=$WANDB_ENTITY"
  [ -n "${WANDB_PROJECT:-}" ] && printf '%s\n' "logging.project=$WANDB_PROJECT"
  printf '%s\n' "logging.experiment=$ARM" "logging.run_name=ep${ep}_${obj}" "logging.upload_checkpoints=false"
  support_flag "$seq"
}
job_env(){ # gpu
  printf '%s\n' "CUDA_VISIBLE_DEVICES=$1" "PYTHONPATH=$FC/src" "WANDB_MODE=$WANDB_MODE" \
    "OMP_NUM_THREADS=$OMP" "MKL_NUM_THREADS=$OMP" "OPENBLAS_NUM_THREADS=$OMP"
}
run_job(){ # ep seq obj gpu attempt
  local ep=$1 seq=$2 obj=$3 g=$4 a=$5 out=$ARMDIR/ep$1 last="" hrc=0 notes line
  local args envs resume=()
  mapfile -t args < <(train_args "$ep" "$seq" "$obj"); mapfile -t envs < <(job_env "$g")
  if [ -d "$out" ] && [ $DRY = 0 ]; then
    notes=$(mktemp); last=$(python "$HELPER" resume "$out" --quarantine 2> "$notes"); hrc=$?
    while IFS= read -r line; do log "ep$ep resume check: $line"; done < "$notes"; rm -f "$notes"
    if [ $hrc -ne 0 ] || { [ -z "$last" ] && compgen -G "$out/model_*.safetensors" > /dev/null; }; then
      log "ep$ep NOT started: resume check exit $hrc, resume point '${last:-none}' - $out has checkpoints a fresh run would overwrite; inspect by hand"
      return 97
    fi
  fi
  [ -n "$last" ] && resume=(training.resume=true "training.checkpoint=$last")
  if [ $DRY = 1 ]; then
    printf '[train ep%s gpu%s] ' "$ep" "$g"; printf '%q ' env "${envs[@]}" python scripts/train_rl.py "${args[@]}" ${resume[@]+"${resume[@]}"}; echo
    return 0
  fi
  mkdir -p "$ARMDIR/logs"
  ( cd "$FC" && env "${envs[@]}" python scripts/train_rl.py "${args[@]}" ${resume[@]+"${resume[@]}"} ) >> "$ARMDIR/logs/ep$ep.log" 2>&1
  local rc=$?; printf '%s\tep%s\t%s\tgpu%s\tattempt%s\texit%s\n' "$(date '+%F %T')" "$ep" "$obj" "$g" "$a" "$rc" >> "$QDIR/status.tsv"
  return $rc
}

IFS=, read -r -a GPU_LIST <<< "$GPUS"
SLOTS=(); for k in $(seq 1 "$K"); do for g in "${GPU_LIST[@]}"; do SLOTS+=("$g:$k"); done; done
if [ $DRY = 1 ]; then
  echo "[train] line $LINE: arm $ARM, experiment $EXPERIMENT, ${#JOBS[@]} episodes, ${#SLOTS[@]} slots (gpus $GPUS x $K)"
  echo "[train] cwd for every job: $FC"
  i=0; for j in "${JOBS[@]}"; do
    set -- $j; s=${SLOTS[$(( i % ${#SLOTS[@]} ))]}
    if [ -d "$ARMDIR/ep$1" ] && why=$(python "$HELPER" done "$ARMDIR/ep$1" "$MAX_ITERATIONS" 2>/dev/null); then
      echo "# ep$1 already finished ($why): would be skipped"; fi
    run_job "$1" "$2" "$3" "${s%%:*}" 1; i=$((i + 1))
  done
  exit 0
fi

mkdir -p "$QDIR"
declare -A SLOT_PID SLOT_JOB TRIES
QUEUE=(); for j in "${JOBS[@]}"; do set -- $j
  if why=$(python "$HELPER" done "$ARMDIR/ep$1" "$MAX_ITERATIONS" 2>/dev/null); then log "ep$1 already finished ($why)"; else QUEUE+=("$j"); fi
done
log "train_all: line $LINE arm $ARM experiment $EXPERIMENT iterations $MAX_ITERATIONS gpus [$GPUS] x $K code $FC" \
    "($(git -C "$FC" rev-parse --short HEAD 2>/dev/null)) W&B $WANDB_MODE queued ${#QUEUE[@]}"
while :; do
  for s in "${SLOTS[@]}"; do
    pid=${SLOT_PID[$s]:-}
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"; rc=$?; set -- ${SLOT_JOB[$s]}; ep=$1
      if why=$(python "$HELPER" done "$ARMDIR/ep$ep" "$MAX_ITERATIONS" 2>/dev/null); then log "ep$ep finished ($why)"
      else t=${TRIES[$ep]:-1}
        if [ "$t" -lt 2 ]; then TRIES[$ep]=$((t + 1)); QUEUE=("${SLOT_JOB[$s]}" "${QUEUE[@]}"); log "ep$ep stopped without final.json (exit $rc): requeued (it resumes)"
        else log "ep$ep FAILED twice (exit $rc): giving up"; fi
      fi
      SLOT_PID[$s]=""; SLOT_JOB[$s]=""
    fi
    if [ -z "${SLOT_PID[$s]:-}" ] && [ ${#QUEUE[@]} -gt 0 ]; then
      job=${QUEUE[0]}; QUEUE=("${QUEUE[@]:1}"); set -- $job; g=${s%%:*}
      run_job "$1" "$2" "$3" "$g" "${TRIES[$1]:-1}" & SLOT_PID[$s]=$!; SLOT_JOB[$s]=$job
      log "start ep$1 ($3) on gpu$g slot $s pid $!"; sleep "$START_GAP"
    fi
  done
  busy=0; for s in "${SLOTS[@]}"; do [ -n "${SLOT_PID[$s]:-}" ] && busy=1; done
  [ $busy = 0 ] && [ ${#QUEUE[@]} = 0 ] && break
  sleep "$POLL"
done
n=0; for j in "${JOBS[@]}"; do set -- $j; python "$HELPER" done "$ARMDIR/ep$1" "$MAX_ITERATIONS" > /dev/null 2>&1 && n=$((n + 1)); done
log "TRAIN_DONE: $n/${#JOBS[@]} episodes finished"
[ "$n" = ${#JOBS[@]} ]
