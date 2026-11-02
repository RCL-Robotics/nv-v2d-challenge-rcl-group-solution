#!/bin/bash
#SBATCH --job-name=v2d-hoi-all
#SBATCH --output=/l/users/han.zhou/v2d/logs/%x-%j.out
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=24:00:00
#SBATCH --exclude=gpu-05

set -uo pipefail
V2D=/l/users/han.zhou/v2d
cd "$V2D"
nvidia-smi

# Allocated GPUs (Slurm sets CUDA_VISIBLE_DEVICES). One worker per GPU.
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
NGPU=${#GPUS[@]}
echo "allocated GPUs: ${GPUS[*]}"

# Usage: sbatch run_all.sh <shard> <nshard>
# Account limit is 4 GPUs / 2 GPU jobs, so the 30 episodes are split into 2 shards
# (episode idx % nshard == shard) of 2 GPUs each; 2-GPU jobs schedule much sooner than 4.
SHARD=${1:-0}; NSHARD=${2:-1}
mapfile -t EPS < <(awk -F'\t' -v s="$SHARD" -v n="$NSHARD" '$1 % n == s {print $1}' "$V2D/code/pipeline/episodes.tsv")
echo "shard $SHARD/$NSHARD: ${#EPS[@]} episodes"

worker() {
  local gpu=$1; shift
  local eps=("$@")
  for i in "${eps[@]}"; do
    EP=$(printf "%06d" "$i")
    if grep -qs '"verdict": "PASS"' "$V2D/outputs/cari4d/episode_$EP/pipeline_report.json"; then
      echo "[gpu $gpu] episode $EP already PASS, skip"; continue
    fi
    echo "[gpu $gpu] episode $EP start $(date +%T)"
    CUDA_VISIBLE_DEVICES=$gpu bash "$V2D/code/pipeline/run_episode.sh" "$i" \
      > "$V2D/logs/episode_${EP}.log" 2>&1 \
      && echo "[gpu $gpu] episode $EP OK" || echo "[gpu $gpu] episode $EP FAILED"
  done
}

# round-robin episodes across GPUs
pids=()
for g in $(seq 0 $((NGPU-1))); do
  slice=()
  for k in $(seq $g $NGPU $((${#EPS[@]}-1))); do
    slice+=("${EPS[$k]}")
  done
  worker "${GPUS[$g]}" "${slice[@]}" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo RUN_ALL_DONE
