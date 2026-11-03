#!/bin/bash
# t1improve_launch.sh: entry point of the Tier-1 improvement wave of 2026-10-27 (fc_runs/T1improve), queued BEHIND Tier 3
# (user, 2026-10-27: "... keep current results for tier3 queue tier1 first, we will wait first round all tier3 finished
# then add revisement on tier3"). Logic in scripts/t1improve_launch.py (docstring); pre-registration
# fc_runs/T1improve/PREREG_t1improve.md (hash-locked; launch modes refuse to start anything if it changed).
#   bash scripts/t1improve_launch.sh init|prepare           queue folder + layout + fc_r2 watcher copy / fork copies + caches
#   bash scripts/t1improve_launch.sh check|dry|gate|status  preflight / gate + plan for the current GPU view / state (no writes)
#   bash scripts/t1improve_launch.sh watch-dry <arm>        that arm's watcher with --dry-run (writes nothing)
#   bash scripts/t1improve_launch.sh next                   one launch pass (at most one start) + missing watchers
#   bash scripts/t1improve_launch.sh start-auto             background loop (every 60 s: watchers, one launch pass);
#                                                           STOP: touch fc_runs/T1improve/STOP_AUTO; PAUSE launching:
#                                                           touch fc_runs/T1improve/PAUSE (rm to resume)
#   Stop one arm's watcher: touch fc_runs/T1improve/arms/<arm>/eval/STOP (its running evaluation finishes first).
# Planning / checks at nice 19 without a GPU; launch modes (next, auto, ensure-watchers) at nice 0 (a trainer inherits it).
# Nothing here signals a process. Run start-auto from a LOCAL background ssh (a foreground ssh that starts setsid
# processes can hang).
ME=/root/autodl-tmp/haoran.zhang
PY=$ME/conda_envs/flashchord/bin/python
T1I=$ME/fc_runs/T1improve
cd $ME || exit 1
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY RUNIN_QUEUE RUNIN_GPU RUNIN_QDIR
unset RUNIN_RUNS RUNIN_EVAL_ROOT RUNIN_ITERS RUNIN_WATCH_ITERS T1FIX_ARM T1FIX_STAGE T1IMPROVE_ARM FLASH_CHORD_CACHE_DIR
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 TZ=Asia/Shanghai

case ${1:-} in
  start-auto)
    [ -f $T1I/PREREG_t1improve.sha256 ] || { echo "no PREREG_t1improve.sha256: not started" >&2; exit 1; }
    [ -f $T1I/STOP_AUTO ] && { echo "$T1I/STOP_AUTO exists: remove it first" >&2; exit 1; }
    setsid nohup $PY -I -B $ME/scripts/t1improve_launch.py auto >> $T1I/auto_stdout.log 2>&1 < /dev/null &
    echo "auto loop started: pid $! (log $T1I/launcher.log, heartbeat $T1I/heartbeat.log; stop: touch $T1I/STOP_AUTO; pause launching: touch $T1I/PAUSE)" ;;
  next|auto|ensure-watchers)
    exec $PY -I -B $ME/scripts/t1improve_launch.py "$@" ;;
  init|prepare|check|dry|gate|status|watch-dry)
    exec nice -n 19 $PY -I -B $ME/scripts/t1improve_launch.py "$@" ;;
  *) sed -n '2,17p' $0; exit 2 ;;
esac
