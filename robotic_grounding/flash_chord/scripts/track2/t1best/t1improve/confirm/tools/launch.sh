#!/usr/bin/env bash
# launch.sh: start (or resume) the T1improve confirmation (t1confirm_driver.py run) in its own session.
#   bash launch.sh                  10 evaluations per candidate + the world_count=64 screens, GPUs 0,1,2,3,5,6,7 (never 4)
#   bash launch.sh --dry-run CAND   t01:CAND and w64:CAND only, on GPU ${DRY_GPU:-1}, one at a time (kept as ordinary evals)
#   bash launch.sh --export         the single export of every candidate (CPU only, 2 workers on the eval-core lanes)
# Log: driver.log (+ driver_stdout.log); pid: driver.pid. Stop: touch STOP (running evaluations finish and are recorded).
# Never signals anything.
set -euo pipefail
ME=/root/autodl-tmp/haoran.zhang
ROOT=$ME/fc_runs/T1improve/confirm
PYBIN=$ME/conda_envs/flashchord/bin/python
DRIVER=$ROOT/t1confirm_driver.py
export PYTHONDONTWRITEBYTECODE=1 TZ=Asia/Shanghai
cd "$ROOT"
if [ "${1:-}" = "--export" ]; then
  setsid nohup "$PYBIN" -B "$DRIVER" export --workers 2 >> "$ROOT/export_stdout.log" 2>&1 < /dev/null &
  echo "started export (pid $!): log $ROOT/export.log"
  exit 0
fi
if [ -e "$ROOT/STOP" ]; then
  echo "$ROOT/STOP exists: remove it first, then relaunch" >&2
  exit 1
fi
if [ -s "$ROOT/driver.pid" ]; then
  old=$(cat "$ROOT/driver.pid")
  if [ -r "/proc/$old/cmdline" ] && tr '\0' ' ' < "/proc/$old/cmdline" | grep -q "t1confirm_driver.py run"; then
    echo "t1confirm_driver.py already runs (pid $old)"
    exit 0
  fi
fi
ARGS=(run --trials 10 --gpus 0,1,2,3,5,6,7 --max-parallel 4 --lane-max 2 --cooldown 35)
if [ "${1:-}" = "--dry-run" ]; then
  CAND=${2:?usage: launch.sh --dry-run <candidate>}
  ARGS=(run --trials 10 --only "t01:$CAND,w64:$CAND" --gpus "${DRY_GPU:-1}" --max-parallel 1 --lane-max 1 --cooldown 35)
fi
setsid nohup "$PYBIN" -B "$DRIVER" "${ARGS[@]}" >> "$ROOT/driver_stdout.log" 2>&1 < /dev/null &
echo $! > "$ROOT/launch.pid"
echo "started t1confirm_driver.py ${ARGS[*]} (pid $!): log $ROOT/driver.log; status: $PYBIN -B $DRIVER status"
