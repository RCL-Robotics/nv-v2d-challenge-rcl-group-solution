#!/usr/bin/env bash
# Tier 2 ep04 (baseball_bat) with the pure rolling start (g1_recon_body_coord_rsi_r2_runin) on the teammate's K3fix input
# (data/human_motion_data_tier2_k3fix), 450,000 iterations, early stop off, seed 42 (user request 2026-10-24), on GPU 7
# (free), code worktrees/dev.zuxing.speedup (bc3b1bd: the rolling-start code with the rollout speedup), with its own
# evaluation watcher (tmp/t2/eval: official kit flow + Tier 2 clean-reference scores with the K3fix frame references).
# launch_run.sh runs this with --no-gpu; it pins GPU 7 itself. Logs: logs/2026-10-24-tier2-runin-k3fix-ep04/.
set -uo pipefail
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7
W=/root/autodl-tmp/zuxing.lu
R=$W/runs/2026-10-24-tier2-runin-k3fix-ep04; L=$W/logs/2026-10-24-tier2-runin-k3fix-ep04
mkdir -p $R $L
cd $W
bash tmp/t2/train_t2.sh g1_recon_body_coord_rsi_r2_runin human_motion_data_tier2_k3fix \
  2026-04-24_11-40-52_baseball_bat_rotate_05 false $R/ep04 68-75,172-179 \
  training.max_iterations=450000 training.early_stop_help_zero_tail=0 > $L/ep04.log 2>&1 &
p=$!; echo "[$(date '+%F %T')] started ep04 (pid $p) on GPU $CUDA_VISIBLE_DEVICES"
sleep 30
RUNS=$R OUT=$R/eval EPISODES="4" FRAME_REFS_EPISODES="4" CODE=$W/worktrees/dev.zuxing.speedup/robotic_grounding/flash_chord \
  taskset -c 80-83,184-187 bash tmp/t2/eval/eval_watch.sh > $L/eval_watch.log 2>&1 &
w=$!; echo "[$(date '+%F %T')] started eval watcher (pid $w)"
wait "$p"; echo "[$(date '+%F %T')] training pid $p exited $?"
wait "$w"; echo "[$(date '+%F %T')] eval watcher pid $w exited $?"
