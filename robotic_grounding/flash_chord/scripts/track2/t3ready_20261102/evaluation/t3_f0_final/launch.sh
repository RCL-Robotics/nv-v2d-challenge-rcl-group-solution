#!/usr/bin/env bash
# launch.sh: start (or resume) the Tier-3 f0 FINAL pipeline (t3final_driver.py run) in its own session.
#   bash launch.sh              start / resume: 5 runs per item, progressive over the 30 episodes, GPUs 0,1,2,3,5,6,7
#                               (never 4), at most 3 of our evals, at most 2 evals of anybody per NUMA eval lane; packs
#                               FINAL + T1..T5 when every episode is settled
#   further arguments go to "t3final_driver.py run" (e.g. --max-parallel 2, --no-pack)
# Log: driver.log (+ driver_stdout.log); pid: driver.pid (written by the driver) and launch.pid.
# Stop: touch STOP (running evaluations finish and are recorded; rm STOP before a restart). Never signals anything.
set -euo pipefail
ME=/root/autodl-tmp/haoran.zhang
ROOT=$ME/fc_runs/submissions/t3_f0_final
PYBIN=$ME/conda_envs/flashchord/bin/python
DRIVER=$ROOT/t3final_driver.py
export PYTHONDONTWRITEBYTECODE=1 TZ=Asia/Shanghai
cd "$ROOT"
if [ -e "$ROOT/STOP" ]; then
  echo "$ROOT/STOP exists: remove it first (rm $ROOT/STOP), then relaunch" >&2
  exit 1
fi
if [ -s "$ROOT/driver.pid" ]; then
  old=$(cat "$ROOT/driver.pid")
  if [ -r "/proc/$old/cmdline" ] && tr '\0' ' ' < "/proc/$old/cmdline" | grep -q "t3final_driver.py run"; then
    echo "t3final_driver.py already runs (pid $old); status: bash $ROOT/status.sh"
    exit 0
  fi
fi
ARGS=(run --runs 5 --gpus 0,1,2,3,5,6,7 --max-parallel 3 --lane-max 2 --cooldown 35)
setsid nohup "$PYBIN" -B "$DRIVER" "${ARGS[@]}" "$@" >> "$ROOT/driver_stdout.log" 2>&1 < /dev/null &
echo $! > "$ROOT/launch.pid"
echo "started t3final_driver.py ${ARGS[*]} $* (pid $!): log $ROOT/driver.log; status: bash $ROOT/status.sh; stop: touch $ROOT/STOP"
