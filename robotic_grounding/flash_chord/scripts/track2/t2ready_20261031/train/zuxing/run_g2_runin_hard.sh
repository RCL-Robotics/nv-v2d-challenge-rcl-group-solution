#!/usr/bin/env bash
# Tier 2 ep01 and ep19 on the G2 filter's object track (data/human_motion_data_tier2_g2-2026-10-24, as run_g2_runin.sh)
# with rolling start + hard-start sampling (g1_recon_body_coord_rsi_r2_runin_hard: 10 % frame 0, 10 % uniform, 80 %
# softmax(hazard / 0.1), cap 15 %, start shift 20, no outcomes during the hold), 180,000 iterations, early stop off,
# seed 42, code worktrees/dev.zuxing.hard_start (82837c8). Both on GPU 4 (user request 2026-10-25). One evaluation
# watcher per run (tmp/t2/eval: Tier 2 clean scores with the G2 frame references; export with the hard_start code).
# launch_run.sh runs this with --no-gpu. Logs: logs/2026-10-25-tier2-runin-hard-g2/.
set -uo pipefail
W=/root/autodl-tmp/zuxing.lu
D=$W/data/human_motion_data_tier2_g2-2026-10-24
R=$W/runs/2026-10-25-tier2-runin-hard-g2; L=$W/logs/2026-10-25-tier2-runin-hard-g2
mkdir -p $R $L
cd $W
echo "[$(date '+%F %T')] code dev.zuxing.hard_start $(git -C worktrees/dev.zuxing.hard_start -c filter.lfs.required=false log -1 --format=%h)"
pids=()
while read gpu ep seq support iters cpus; do
  CODE=dev.zuxing.hard_start CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu bash tmp/t2/train_t2.sh \
    g1_recon_body_coord_rsi_r2_runin_hard $D $seq $support $R/ep$ep $cpus training.max_iterations=$iters \
    training.early_stop_help_zero_tail=0 > $L/ep$ep.log 2>&1 &
  pids+=($!); echo "[$(date '+%F %T')] started ep$ep ($iters iterations) on GPU $gpu, CPUs $cpus (pid $!)"
  sleep 20
done <<'JOBS'
4 01 2026-05-18_17-06-21_vacuum_left_lean_02 true 180000 10-15,114-119
4 19 2026-05-19_10-19-48_cane_swing_05 false 180000 16-21,120-125
JOBS
sleep 20
for x in "01 1" "19 19"; do
  set -- $x
  RUNS=$R OUT=$R/eval_ep$1 EPISODES="$2" FRAME_REFS=$D/frame_refs_g2 FRAME_REFS_EPISODES="$2" \
    CODE=$W/worktrees/dev.zuxing.hard_start/robotic_grounding/flash_chord CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 \
    taskset -c 62-65,166-169 bash tmp/t2/eval/eval_watch.sh > $L/eval_watch_ep$1.log 2>&1 &
  pids+=($!); echo "[$(date '+%F %T')] started eval watcher ep$1 on GPU 4 (pid $!)"
done
for p in "${pids[@]}"; do wait "$p"; echo "[$(date '+%F %T')] pid $p exited $?"; done
