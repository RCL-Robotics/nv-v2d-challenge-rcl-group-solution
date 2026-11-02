#!/usr/bin/env bash
# Tier 2 with the pure rolling start (g1_recon_body_coord_rsi_r2_runin) on the G2 filter's object track (user requests
# 2026-10-24): ep04 450,000 iterations (replaces the K3fix ep04 run, stopped at 12k), ep01 / ep19 / ep28 180,000;
# early stop off, seed 42, code worktrees/dev.zuxing.speedup (bc3b1bd). Input: data/human_motion_data_tier2_g2-2026-10-24
# (tmp/t2/g2data/build_g2_data.py: the raw Tier-2 G1 parquet with only the object columns replaced by the G2 output; the
# human stays raw). GPUs (not 4, at most 4 tasks each): ep04 and ep28 on 7, ep01 on 1, ep19 on 6. One evaluation watcher
# per run on its GPU (tmp/t2/eval: Tier 2 clean scores with the G2 frame references). launch_run.sh runs this with
# --no-gpu. Logs: logs/2026-10-24-tier2-runin-g2/.
set -uo pipefail
W=/root/autodl-tmp/zuxing.lu
D=$W/data/human_motion_data_tier2_g2-2026-10-24
R=$W/runs/2026-10-24-tier2-runin-g2; L=$W/logs/2026-10-24-tier2-runin-g2
mkdir -p $R $L
cd $W
pids=()
while read gpu ep seq support iters cpus; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu bash tmp/t2/train_t2.sh g1_recon_body_coord_rsi_r2_runin $D \
    $seq $support $R/ep$ep $cpus training.max_iterations=$iters training.early_stop_help_zero_tail=0 > $L/ep$ep.log 2>&1 &
  pids+=($!); echo "[$(date '+%F %T')] started ep$ep ($iters iterations) on GPU $gpu, CPUs $cpus (pid $!)"
done <<'JOBS'
7 04 2026-04-24_11-40-52_baseball_bat_rotate_05 false 450000 70,71,174,175
1 01 2026-05-18_17-06-21_vacuum_left_lean_02 true 180000 72,73,176,177
6 19 2026-05-19_10-19-48_cane_swing_05 false 180000 74,75,178,179
7 28 2026-05-19_11-27-07_squeegee_sweep_away_from_you_03_ true 180000 80,81,184,185
JOBS
sleep 30
for x in "7 04 4" "1 01 1" "6 19 19" "7 28 28"; do
  set -- $x
  RUNS=$R OUT=$R/eval_ep$2 EPISODES="$3" FRAME_REFS=$D/frame_refs_g2 FRAME_REFS_EPISODES="$3" \
    CODE=$W/worktrees/dev.zuxing.speedup/robotic_grounding/flash_chord CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 \
    taskset -c 82,83,186,187 bash tmp/t2/eval/eval_watch.sh > $L/eval_watch_ep$2.log 2>&1 &
  pids+=($!); echo "[$(date '+%F %T')] started eval watcher ep$2 on GPU $1 (pid $!)"
done
for p in "${pids[@]}"; do wait "$p"; echo "[$(date '+%F %T')] pid $p exited $?"; done
