#!/usr/bin/env bash
# t2best5_launch.sh: start (or resume) scripts/t2best5_driver.py in its own session (setsid nohup), or dry-run it.
#   bash scripts/t2best5_launch.sh               start / resume in the background
#   bash scripts/t2best5_launch.sh --dry-run     plan only, foreground (writes only ROOT/dryrun and ROOT/preflight)
#   further arguments go to "t2best5_driver.py run", e.g. --gpus 0,1,2,5,6,7 --max-parallel 4
# Stop: touch ROOT/STOP (running jobs finish and are recorded first; rm ROOT/STOP before a restart).
# This script never signals any process.
set -euo pipefail
ME=/root/autodl-tmp/haoran.zhang
ROOT=$ME/fc_runs/submissions/t2_best5_20261024
PYBIN=$ME/conda_envs/flashchord/bin/python
DRIVER=$ME/scripts/t2best5_driver.py
export PYTHONDONTWRITEBYTECODE=1
export TZ=Asia/Shanghai
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
if [ "${1:-}" = "--dry-run" ]; then
  shift
  exec "$PYBIN" -B "$DRIVER" run --dry-run "$@"
fi
if [ -e "$ROOT/STOP" ]; then
  echo "$ROOT/STOP exists: remove it first (rm $ROOT/STOP), then relaunch" >&2
  exit 1
fi
if [ -s "$ROOT/driver.pid" ]; then
  old=$(cat "$ROOT/driver.pid")
  if [ -r "/proc/$old/cmdline" ] && tr '\0' ' ' < "/proc/$old/cmdline" | grep -q "t2best5_driver.py run"; then
    echo "t2best5_driver.py already runs (pid $old); heartbeat $ROOT/heartbeat.json"
    exit 0
  fi
fi
cd "$ROOT"
setsid nohup "$PYBIN" -B "$DRIVER" run "$@" >> "$ROOT/driver_stdout.log" 2>&1 < /dev/null &
echo "started t2best5_driver.py run (pid $!): log $ROOT/driver.log, heartbeat $ROOT/heartbeat.json, stop: touch $ROOT/STOP"
