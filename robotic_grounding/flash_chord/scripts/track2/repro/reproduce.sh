#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# One command for the Track 2 Tier-1 C3r2 submission: data -> SOMA inputs -> G1 retargets -> training (one policy per
# episode) -> export of each final checkpoint -> official evaluation -> summary -> packed CSV. See README.md.
#
# usage: reproduce.sh [--config FILE] [--line FILE] [--steps LIST] [--episodes all|0,3,...] [--dry-run]
#   --config    bash file with the settings (default: config.env next to this script); see config.env.example
#   --line      line spec (default lines/c3r2_tier1.env)
#   --steps     comma list, run in this order:  setup data convert retarget train export eval pack repeats
#               default: data,convert,retarget,train,export,eval,pack
#                 setup    fetch the stock runtime (env/get_stock_runtime.sh) and build the flashchord env
#                          (env/install_flashchord.sh); the retargeting env and the kit venv are manual (README)
#                 data     download the dataset (robotic_grounding/scripts/track2/get_data.sh)
#                 convert  Track 2 LeRobot -> SOMA inputs (robotic_grounding/scripts/track2/track2_to_soma_inputs.py)
#                 retarget SOMA-X -> G1 + Dex3 (robotic_grounding/scripts/track2/retarget_all.sh)
#                 train    one training run per episode (train_all.sh)
#                 export   final checkpoint -> stock format, tolerance from the line (final_eval.sh --export-only)
#                 eval     official evaluator + summary table (final_eval.sh)
#                 pack     submission CSV + kit preflight (pack.sh; all episodes only)
#                 repeats  re-run the evaluator on the same exports (repeat_eval.sh; diagnostic)
#   --episodes  a subset for trials (packing needs all 30)
#   --dry-run   print every command of the selected steps, run nothing
# Relative paths in the config or the environment are taken from the current directory. Before retarget, train and
# export, the Git LFS assets these steps load must be real files, not pointers (README, "Git LFS").
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); FC=$(cd "$HERE/../../.." && pwd); RG=$(cd "$FC/.." && pwd)
CONFIG=$HERE/config.env; LINE=$HERE/lines/c3r2_tier1.env; STEPS=data,convert,retarget,train,export,eval,pack
EPISODES=all; DRY=0
while [ $# -gt 0 ]; do case "$1" in
  --config) CONFIG=$2; shift 2;; --line) LINE=$2; shift 2;; --steps) STEPS=$2; shift 2;;
  --episodes) EPISODES=$2; shift 2;; --dry-run) DRY=1; shift;; -h|--help) sed -n '2,24p' "$0"; exit 0;;
  *) echo "unknown argument $1" >&2; exit 2;; esac; done
LINE=$(cd "$(dirname "$LINE")" && pwd)/$(basename "$LINE")
# shellcheck disable=SC1090
if [ -f "$CONFIG" ]; then set -a; source "$CONFIG"; set +a; else echo "# no config file $CONFIG: using the environment"; fi
# shellcheck disable=SC1090
source "$LINE"
for v in V2D_WORK V2D_DATA V2D_MOTION V2D_RUNTIME V2D_KIT FC_PYTHON KIT_PYTHON; do
  [ -n "${!v:-}" ] || { echo "set $v (in $CONFIG or the environment)" >&2; exit 2; }; done
abs(){ case $1 in /*|"") printf '%s' "$1";; *) printf '%s' "$PWD/$1";; esac; }
for v in V2D_WORK V2D_DATA V2D_MOTION V2D_RUNTIME V2D_KIT SUBMISSION_CSV RETARGET_ACTIVATE; do
  [ -n "${!v:-}" ] && printf -v "$v" '%s' "$(abs "${!v}")"; done
for v in FC_PYTHON KIT_PYTHON RETARGET_PYTHON; do   # a bare command name stays as it is
  case ${!v:-} in */*) printf -v "$v" '%s' "$(abs "${!v}")";; esac; done
export V2D_WORK V2D_DATA V2D_MOTION V2D_RUNTIME V2D_KIT FC_PYTHON KIT_PYTHON
export GPUS=${GPUS:-0} JOBS_PER_GPU=${JOBS_PER_GPU:-1} WANDB_MODE=${WANDB_MODE:-disabled}
export RETARGET_PYTHON=${RETARGET_PYTHON:-python} RETARGET_JOBS_PER_GPU=${RETARGET_JOBS_PER_GPU:-4}
DRYF=(); [ $DRY = 1 ] && DRYF=(--dry-run)
ORDER=(setup data convert retarget train export eval pack repeats)
for s in ${STEPS//,/ }; do case " ${ORDER[*]} " in *" $s "*) ;; *) echo "unknown step $s" >&2; exit 2;; esac; done
want(){ case ",$STEPS," in *",$1,"*) return 0;; *) return 1;; esac; }
step(){ echo; echo "========== [$(date '+%F %T')] $* =========="; }
show(){ printf '%s ' "$1"; shift; printf '%q ' "$@"; echo; }   # print a command
do_run(){ local tag=$1; shift; if [ $DRY = 1 ]; then show "$tag" "$@"; else "$@" || { echo "step failed: $*" >&2; exit 1; }; fi; }
need_assets(){ # <step> <dir>...: stop when the Git LFS assets the step loads are still pointer files
  local tag=$1 n; shift
  n=$(find "$@" -type f -size -1024c -exec grep -l -m1 '^version https://git-lfs.github.com/spec/v1' {} + 2>/dev/null | wc -l)
  [ "$n" = 0 ] && return 0
  echo "[$tag] $n Git LFS pointer files under $*: pull the LFS assets first (README.md, \"Git LFS\")" >&2
  [ $DRY = 1 ] || exit 1
}

echo "reproduce.sh: line $LINE_NAME ($LINE) | steps $STEPS | episodes $EPISODES | code $(git -C "$HERE" rev-parse --short HEAD 2>/dev/null) | work $V2D_WORK"
for s in "${ORDER[@]}"; do want "$s" || continue; case $s in
  setup)
    step "setup: stock runtime $STOCK_RUNTIME_COMMIT + flashchord env"
    do_run "[setup]" bash "$HERE/env/get_stock_runtime.sh" "${V2D_RUNTIME%/robotic_grounding/flash_chord}"
    do_run "[setup]" bash "$HERE/env/install_flashchord.sh" "$(dirname "$(dirname "$FC_PYTHON")")" "$V2D_RUNTIME"
    echo "[setup] # retargeting env: env/install_retarget.sh inside an Isaac Sim 5.1 / Isaac Lab 2.3.2 env (or robotic_grounding/workflow/Dockerfile)"
    echo "[setup] # kit venv: python3.11 -m venv <dir> && <dir>/bin/pip install -r \$V2D_KIT/requirements.txt";;
  data)
    step "data: $DATA_REPO@$DATA_REVISION -> $V2D_DATA"
    export DATA_REPO DATA_REVISION
    bash "$RG/scripts/track2/get_data.sh" "$V2D_DATA" ${DRYF[@]+"${DRYF[@]}"} || { echo "data download failed" >&2; exit 1; };;
  convert)
    step "convert: $TIER_DIR -> SOMA inputs in $V2D_MOTION"
    cmd=("$RETARGET_PYTHON" scripts/track2/track2_to_soma_inputs.py --track-root "$V2D_DATA/track_2/$TIER_DIR"
         --motion-root "$V2D_MOTION" --episodes "$EPISODES")
    if [ $DRY = 1 ]; then printf '[convert] cd %q && ' "$RG"; printf '%q ' "${cmd[@]}"; echo
    else ( source "$HERE/env/retarget_env.sh"; cd "$RG" && "${cmd[@]}" ) || { echo "convert failed" >&2; exit 1; }; fi;;
  retarget)
    step "retarget: SOMA-X -> G1 + Dex3 (fps $RETARGET_FPS)"
    need_assets retarget "$RG/source/robotic_grounding/robotic_grounding/assets/urdfs" \
      "$RG/source/robotic_grounding/robotic_grounding/assets/meshes"
    rt=("$RG/scripts/track2/retarget_all.sh" "$V2D_MOTION" --gpus "$GPUS" --jobs-per-gpu "$RETARGET_JOBS_PER_GPU"
        --fps "$RETARGET_FPS" --episodes "$EPISODES" --log-dir "$V2D_WORK/logs/retarget")
    if [ $DRY = 1 ]; then bash "${rt[@]}" --dry-run
    else ( source "$HERE/env/retarget_env.sh"; bash "${rt[@]}" ) || { echo "retarget failed" >&2; exit 1; }; fi;;
  train)
    step "train: $ARM ($EXPERIMENT, $MAX_ITERATIONS iterations max, seed $RESET_SEED)"
    need_assets train "$FC/src/flash_chord/assets"
    bash "$HERE/train_all.sh" --line "$LINE" --runs "$V2D_WORK/runs" --episodes "$EPISODES" --gpus "$GPUS" \
      --jobs-per-gpu "$JOBS_PER_GPU" ${DRYF[@]+"${DRYF[@]}"} || { echo "training incomplete" >&2; exit 1; };;
  export)
    step "export: final checkpoints, tolerance $EXPORT_TOLERANCE"
    need_assets export "$FC/src/flash_chord/assets"
    bash "$HERE/final_eval.sh" --line "$LINE" --runs "$V2D_WORK/runs" --eval-root "$V2D_WORK/eval" --episodes "$EPISODES" \
      --gpus "$GPUS" --export-only ${DRYF[@]+"${DRYF[@]}"} || { echo "export failed" >&2; exit 1; };;
  eval)
    step "eval: official evaluator (world_count $EVAL_WORLD_COUNT) on the stock runtime"
    bash "$HERE/final_eval.sh" --line "$LINE" --runs "$V2D_WORK/runs" --eval-root "$V2D_WORK/eval" --episodes "$EPISODES" \
      --gpus "$GPUS" ${DRYF[@]+"${DRYF[@]}"} || { echo "evaluation failed" >&2; exit 1; };;
  pack)
    step "pack: $TRACK submission CSV"
    if [ "$EPISODES" != all ]; then echo "[pack] skipped: --episodes $EPISODES is a subset (the packer needs every episode)"; continue; fi
    bash "$HERE/pack.sh" --line "$LINE" --results "$V2D_WORK/eval/$ARM/results" ${DRYF[@]+"${DRYF[@]}"} || { echo "pack failed" >&2; exit 1; };;
  repeats)
    step "repeats: evaluator spread on the same exports"
    bash "$HERE/repeat_eval.sh" --line "$LINE" --eval-root "$V2D_WORK/eval" --gpu "${GPUS%%,*}" ${DRYF[@]+"${DRYF[@]}"} \
      || { echo "repeats failed" >&2; exit 1; };;
esac; done
echo; echo "reproduce.sh: done ($STEPS)"
