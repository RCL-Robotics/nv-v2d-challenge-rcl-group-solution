#!/usr/bin/env bash
# r9_launch.sh: start the round-9 orchestrator r9_driver.py (PREREG_R9.md section 7) in its own session, or report that it
# already runs. The orchestrator freezes the pool at once (T_c9 = the freeze start), runs phase S / phase F on the GPUs of
# PREREG_R9 3.5 (3, 5, 6, 7 while no compute process is listed; 0, 1, 2, 4 also while only our own ablation jobs are
# listed; plain, one evaluation of ours per GPU) and writes SELECTION_R9. It never commits, pushes, packs, uploads or
# signals anything.
#   bash r9_launch.sh            start (safe to call again: a running orchestrator is reported, not duplicated; refused
#                                while an r9_eval.py evaluation driver of an earlier orchestrator still runs)
# Status: conda_envs/flashchord/bin/python -I -B r9_driver.py status   (or R9_DRIVER_HEARTBEAT.json, r9_driver.log)
# Stop:   touch STOP in the round-9 folder (running evaluations finish; nothing further is selected).
set -euo pipefail
ME=/root/autodl-tmp/haoran.zhang
ROOT=$ME/fc_runs/submissions/t2_consensus_r9_20261103
PY=$ME/conda_envs/flashchord/bin/python
export TZ=Asia/Shanghai PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE CUDA_VISIBLE_DEVICES
cd "$ROOT"
RUNNING=$(pgrep -af "r9_driver.py run" | head -1 || true)
if [ -n "$RUNNING" ]; then
  echo "r9_driver.py already runs: $RUNNING"
  exit 0
fi
EVALS=$(pgrep -af "r9_eval.py run" | grep -v -e "--dry-run" | head -3 || true)
if [ -n "$EVALS" ]; then
  echo "an r9_eval.py evaluation driver still runs (PREREG_R9 7.6): $EVALS" >&2
  echo "wait until it has ended (eval_r9_heartbeat_*.json no longer 'running'), then run this launcher again" >&2
  exit 1
fi
if [ -e "$ROOT/STOP" ]; then
  echo "$ROOT/STOP exists: remove it first, then relaunch" >&2
  exit 1
fi
setsid nohup "$PY" -I -B "$ROOT/r9_driver.py" run >> "$ROOT/r9_driver_stdout.log" 2>&1 < /dev/null &
PID=$!
disown "$PID" || true
echo "started r9_driver.py (pid $PID): log $ROOT/r9_driver.log, heartbeat $ROOT/R9_DRIVER_HEARTBEAT.json, stop: touch $ROOT/STOP"
