#!/usr/bin/env bash
# Train one FlashCHORD run on a Tier 2 input on pro6000. Usage:
#   train_t2.sh <experiment> <data root under haoran.zhang/data> <sequence_id> <support true|false> <out> <cpus> [overrides...]
# Code: worktrees/${CODE:-dev.zuxing.speedup}; the teammate's conda env and data are read only; every write goes under
# /root/autodl-tmp/zuxing.lu.
set -uo pipefail
ME=/root/autodl-tmp/haoran.zhang; W=/root/autodl-tmp/zuxing.lu
FC=$W/worktrees/${CODE:-dev.zuxing.speedup}/robotic_grounding/flash_chord
exp=$1; root=$2; seq=$3; support=$4; out=$5; cpus=$6; shift 6
case $root in /*) droot=$root ;; *) droot=$ME/data/$root ;; esac   # an absolute root (e.g. our G2 data) as is
parquet=$droot/whole_body/soma/sequence_id=$seq/robot_name=g1/data.parquet
[ -f "$parquet" ] || { echo "missing $parquet" >&2; exit 2; }
export PATH=$ME/conda_envs/flashchord/bin:$PATH
_SP=$ME/conda_envs/flashchord/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH=$_SP/cu13/lib:$_SP/cudnn/lib:$_SP/nccl/lib:$_SP/nvshmem/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$FC/src
export HUMAN_MOTION_DATA_DIR=$droot WANDB_MODE=disabled WANDB_DIR=$W/.cache/wandb
export XDG_CACHE_HOME=$W/.cache PYTHONPYCACHEPREFIX=$W/.cache/pycache TZ=Asia/Shanghai
mkdir -p "$out"
cd $FC
echo "train_t2: exp=$exp root=$root seq=$seq support=$support cpus=$cpus gpu=${CUDA_VISIBLE_DEVICES:-unset} commit=$(git -c filter.lfs.required=false log -1 --format=%h)"
exec taskset -c "$cpus" python scripts/train_rl.py experiment=$exp "task.parquet='$parquet'" \
  scene.support=$support reset.seed=42 output_dir="$out" hydra.run.dir="$out/hydra" \
  logging.mode=disabled +logging.local_jsonl=true "$@"
