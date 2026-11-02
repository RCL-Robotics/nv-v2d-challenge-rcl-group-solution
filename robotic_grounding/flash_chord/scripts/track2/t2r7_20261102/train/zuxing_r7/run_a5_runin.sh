#!/usr/bin/env bash
# Tier 2 runs of the high-feasibility list (user request 2026-10-28; at most 3 tasks per pro6000 GPU), pure rolling
# start (g1_recon_body_coord_rsi_r2_runin), 180,000 iterations, early stop off, seed 42, code worktrees/dev.zuxing.speedup:
#   ep16 spray_bottle on the G2 input (data/human_motion_data_tier2_g2-2026-10-24, ep16 added 2026-10-28, build_ep16.json)
#   ep25 cyan_water_bottle and ep14 round_table on the teammate's K3fix input (K3 beats G2 on both: 0.831 vs 0.747,
#   0.802 vs 0.651). All three on one GPU, G (default 6): the teammate filled 1-3 and his short evaluations move between 2, 5 and 6. One evaluation watcher per
# run on its GPU (Tier 2 clean scores; G2 or K3fix frame references). launch_run.sh runs this as G=<g> with --gpus <g>.
set -uo pipefail
W=/root/autodl-tmp/zuxing.lu
G2=$W/data/human_motion_data_tier2_g2-2026-10-24
R=$W/runs/2026-10-28-tier2-runin-a5; L=$W/logs/2026-10-28-tier2-runin-a5
mkdir -p $R $L
cd $W
pids=()
while read gpu ep seq support root cpus; do
  gpu=${G:-$gpu}
  [ $root = G2 ] && root=$G2
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu bash tmp/t2/train_t2.sh g1_recon_body_coord_rsi_r2_runin $root \
    $seq $support $R/ep$ep $cpus training.max_iterations=180000 training.early_stop_help_zero_tail=0 > $L/ep$ep.log 2>&1 &
  pids+=($!); echo "[$(date '+%F %T')] started ep$ep ($root) on GPU $gpu, CPUs $cpus (pid $!)"
done <<'JOBS'
6 16 2026-03-12_15-06-36_spray_bottle_left_desk_ground_02 true G2 88,89,192,193
6 25 2026-03-20_11-59-57_cyan_water_bottle_rotate_01 true human_motion_data_tier2_k3fix 90,91,194,195
6 14 2026-05-28_14-44-09_round_table_flip_08 false human_motion_data_tier2_k3fix 92,93,196,197
JOBS
sleep 30
for x in "6 16 16 G2" "6 25 25 K3fix" "6 14 14 K3fix"; do
  set -- $x; [ -n "${G:-}" ] && set -- $G $2 $3 $4
  [ $4 = G2 ] && FR=(FRAME_REFS=$G2/frame_refs_g2) || FR=()
  env "${FR[@]}" RUNS=$R OUT=$R/eval_ep$2 EPISODES="$3" FRAME_REFS_EPISODES="$3" \
    CODE=$W/worktrees/dev.zuxing.speedup/robotic_grounding/flash_chord CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 \
    taskset -c 94,95,198,199 bash tmp/t2/eval/eval_watch.sh > $L/eval_watch_ep$2.log 2>&1 &
  pids+=($!); echo "[$(date '+%F %T')] started eval watcher ep$2 on GPU $1 (pid $!)"
done
for p in "${pids[@]}"; do wait "$p"; echo "[$(date '+%F %T')] pid $p exited $?"; done
