#!/usr/bin/env bash
# Tier 2 G2 + rolling start for five of the episodes never trained on G2 (user request 2026-11-01; pro6000 GPU 5 takes
# 2 tasks, GPU 6 1, GPU 7 2, beside the teammate's jobs, by the user's decision). Pure rolling start
# (g1_recon_body_coord_rsi_r2_runin), 4,096 worlds, 180,000 iterations, early stop off, seed 42, code
# worktrees/dev.zuxing.speedup; input data/human_motion_data_tier2_g2-2026-10-24 (build_ep00_02_..._29.json);
# scene.support from the teammate's C3r2_tier2 configs. One evaluation watcher per run on its GPU (G2 frame references).
set -uo pipefail
W=/root/autodl-tmp/zuxing.lu
G2=$W/data/human_motion_data_tier2_g2-2026-10-24
R=$W/runs/2026-11-01-tier2-g2-rest; L=$W/logs/2026-11-01-tier2-g2-rest
mkdir -p $R $L
cd $W
pids=()
while read gpu ep seq support cpus; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu bash tmp/t2/train_t2.sh g1_recon_body_coord_rsi_r2_runin $G2 \
    $seq $support $R/ep$ep $cpus training.max_iterations=180000 training.early_stop_help_zero_tail=0 > $L/ep$ep.log 2>&1 &
  pids+=($!); echo "[$(date '+%F %T')] started ep$ep on GPU $gpu, CPUs $cpus (pid $!)"
done <<'JOBS'
5 20 2026-06-08_16-57-34_rice_cooker_clockwise_04 true 76,77,180,181
5 22 2026-05-22_16-52-36_step_ladder_drag_03 false 78,79,182,183
6 26 2026-06-05_16-05-08_skateboard_rotate_ground_10 false 80,81,184,185
7 27 2026-05-01_11-15-05_woven_basket_ground_desk_10 true 88,89,192,193
7 29 2026-06-05_18-27-40_rolling_luggage_left_foot_push_06 false 92,93,196,197
JOBS
sleep 30
for x in "5 20" "5 22" "6 26" "7 27" "7 29"; do
  set -- $x
  FRAME_REFS=$G2/frame_refs_g2 RUNS=$R OUT=$R/eval_ep$2 EPISODES="$((10#$2))" FRAME_REFS_EPISODES="$((10#$2))" \
    CODE=$W/worktrees/dev.zuxing.speedup/robotic_grounding/flash_chord CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 \
    taskset -c 94,95,198,199 bash tmp/t2/eval/eval_watch.sh > $L/eval_watch_ep$2.log 2>&1 &
  pids+=($!); echo "[$(date '+%F %T')] started eval watcher ep$2 on GPU $1 (pid $!)"
done
for p in "${pids[@]}"; do wait "$p"; echo "[$(date '+%F %T')] pid $p exited $?"; done
