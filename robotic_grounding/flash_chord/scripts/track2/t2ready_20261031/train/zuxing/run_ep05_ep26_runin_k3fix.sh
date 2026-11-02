#!/usr/bin/env bash
# Tier 2 ep05 (big_black_box) and ep26 (skateboard) with the pure rolling start (g1_recon_body_coord_rsi_r2_runin) on the
# teammate's K3fix input, 180,000 iterations, early stop off, seed 42 (user request 2026-10-24: not GPU 4, at most 4
# tasks per GPU). GPUs 0 and 2 held 3 tasks each (none of ours); code worktrees/dev.zuxing.speedup (bc3b1bd); one
# evaluation watcher per run on its GPU (tmp/t2/eval: Tier 2 clean scores with the K3fix frame references).
# launch_run.sh runs this with --no-gpu. Logs: logs/2026-10-24-tier2-runin-k3fix-ep05-ep26/.
set -uo pipefail
W=/root/autodl-tmp/zuxing.lu
R=$W/runs/2026-10-24-tier2-runin-k3fix-ep05-ep26; L=$W/logs/2026-10-24-tier2-runin-k3fix-ep05-ep26
mkdir -p $R $L
cd $W
pids=()
while read gpu ep seq cpus; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu bash tmp/t2/train_t2.sh g1_recon_body_coord_rsi_r2_runin \
    human_motion_data_tier2_k3fix $seq false $R/ep$ep $cpus \
    training.max_iterations=180000 training.early_stop_help_zero_tail=0 > $L/ep$ep.log 2>&1 &
  pids+=($!); echo "[$(date '+%F %T')] started ep$ep on GPU $gpu, CPUs $cpus (pid $!)"
done <<'JOBS'
0 05 2026-03-18_19-36-56_g1_box_rotate_01 48-51,152-155
2 26 2026-06-05_16-05-08_skateboard_rotate_ground_10 78,79,84,85,182,183,188,189
JOBS
sleep 30
for x in "0 05 5" "2 26 26"; do
  set -- $x
  RUNS=$R OUT=$R/eval_ep$2 EPISODES="$3" FRAME_REFS_EPISODES="$3" CODE=$W/worktrees/dev.zuxing.speedup/robotic_grounding/flash_chord \
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1 bash tmp/t2/eval/eval_watch.sh > $L/eval_watch_ep$2.log 2>&1 &
  pids+=($!); echo "[$(date '+%F %T')] started eval watcher ep$2 on GPU $1 (pid $!)"
done
for p in "${pids[@]}"; do wait "$p"; echo "[$(date '+%F %T')] pid $p exited $?"; done
