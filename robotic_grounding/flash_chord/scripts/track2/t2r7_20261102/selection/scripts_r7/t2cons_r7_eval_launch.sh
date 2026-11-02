#!/usr/bin/env bash
# t2cons_r7_eval_launch.sh: start (or resume) scripts/t2cons_r7_eval.py (round 7, PREREG_R7.md section 3) in its own
# session (setsid nohup ... & disown), or run one of its preparatory modes. Round 6's launcher with the round-7 names,
# GPUs 6 and 7, cap 2 and a 10 s cooldown.
#   bash scripts/t2cons_r7_eval_launch.sh --phase screen                 phase S: 154 new exports + 3 runs each, background
#   bash scripts/t2cons_r7_eval_launch.sh --phase final                  phase F: 10 trials (needs SELECTION_S7.sha256)
#   bash scripts/t2cons_r7_eval_launch.sh --phase screen|final --dry-run       plan only, foreground (ROOT/dryrun, ROOT/preflight)
#   bash scripts/t2cons_r7_eval_launch.sh --phase screen|final --dry-run-eval  ONE official evaluation into ROOT/dryrun,
#                                                       background (screen: ep00_z5k3fix_89500; final: ep01_zxg2_120000)
#   bash scripts/t2cons_r7_eval_launch.sh --phase screen --exports-only  every new screen job's single export, background
# Defaults: --gpus 6,7 --max-parallel 2 --lane-max 2 --cooldown 10. Extra driver flags (e.g. --gpus 6 for a dry run)
# may follow the mode. Status: $PYBIN -I -B scripts/t2cons_r7_eval.py status --phase screen|final
# Stop: touch ROOT/STOP (running jobs finish and are recorded first; rm ROOT/STOP before a restart).
# This script never signals any process. GPUs 6 and 7 only, plain (no CUDA_MPS_*), never MPS.
set -euo pipefail
ME=/root/autodl-tmp/haoran.zhang
ROOT=$ME/fc_runs/submissions/t2_consensus_r7_20261102
PYBIN=$ME/conda_envs/flashchord/bin/python
DRIVER=$ME/scripts/t2cons_r7_eval.py
export PYTHONDONTWRITEBYTECODE=1
export TZ=Asia/Shanghai
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE CUDA_VISIBLE_DEVICES
usage="usage: bash $0 --phase screen|final [--dry-run | --dry-run-eval | --exports-only] [--gpus G,...]"
if [ "${1:-}" != "--phase" ] || [ -z "${2:-}" ]; then echo "$usage" >&2; exit 2; fi
PHASE=$2; shift 2
case "$PHASE" in screen|final) ;; *) echo "$usage" >&2; exit 2 ;; esac
GPUS=6,7
MODE=()
LOGTAG=run_$PHASE
case "${1:-}" in
  --dry-run) shift; MODE=(--dry-run); LOGTAG=dryrun_$PHASE ;;
  --dry-run-eval) shift; MODE=(--dry-run-eval); LOGTAG=dryrun_eval_$PHASE ;;
  --exports-only) shift; MODE=(--exports-only); LOGTAG=exports_only ;;
  "") ;;
  --gpus) ;;
  *) echo "$usage" >&2; exit 2 ;;
esac
if [ "${1:-}" = "--gpus" ]; then
  [ -n "${2:-}" ] || { echo "$usage" >&2; exit 2; }
  GPUS=$2; shift 2
fi
if [ "$#" -gt 0 ]; then echo "$usage" >&2; exit 2; fi
ARGS=(--phase "$PHASE" --gpus "$GPUS" --max-parallel 2 --lane-max 2 --cooldown 10)
cd "$ROOT"
if [ "${MODE[*]:-}" = "--dry-run" ]; then
  exec "$PYBIN" -I -B "$DRIVER" run "${MODE[@]}" "${ARGS[@]}"
fi
if [ -e "$ROOT/STOP" ]; then
  echo "$ROOT/STOP exists: remove it first (rm $ROOT/STOP), then relaunch" >&2
  exit 1
fi
if [ -s "$ROOT/driver.pid" ]; then
  old=$(cat "$ROOT/driver.pid")
  if [ -r "/proc/$old/cmdline" ] && tr '\0' ' ' < "/proc/$old/cmdline" | grep -q "t2cons_r7_eval"; then
    echo "a round-7 driver already runs (pid $old: $(tr '\0' ' ' < /proc/$old/cmdline | cut -c1-140)); status: $PYBIN -I -B $DRIVER status --phase $PHASE"
    exit 0
  fi
fi
setsid nohup "$PYBIN" -I -B "$DRIVER" run "${MODE[@]}" "${ARGS[@]}" >> "$ROOT/eval_r7_driver_stdout_${LOGTAG}.log" 2>&1 < /dev/null &
PID=$!
disown "$PID" 2>/dev/null || true
echo "started t2cons_r7_eval.py run ${MODE[*]:-} ${ARGS[*]} (pid $PID): log $ROOT/eval_r7_driver.log (dry-run eval: ROOT/dryrun/eval_*/driver.log), heartbeat $ROOT/eval_r7_heartbeat_<s7|f7>.json, stdout $ROOT/eval_r7_driver_stdout_${LOGTAG}.log, stop: touch $ROOT/STOP"
