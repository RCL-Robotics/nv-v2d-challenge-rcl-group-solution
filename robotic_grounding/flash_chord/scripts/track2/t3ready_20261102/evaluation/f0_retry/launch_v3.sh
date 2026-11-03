#!/bin/bash
# launch_v3.sh: start the Tier-3 retry driver v3 (tmp_analysis/tier3/f0_retry/t3retry_v3.py run; PREREG_v3.md).
# Starting the driver does NOT start any training: a retry trains only when PREREG/file hashes match, RETRY_GO exists
# (written by the main session after the user's approval, naming this PREREG's sha256), tmp_analysis/tier3/f0/ALL_LAUNCHED
# exists with no FRESH f0 clip, a slot is free under the t3f0 fill rule (v3: GPUs 0 1 2 5 6 7 only, < 3 trainings of anybody per GPU before a start), no STOP /
# HOLD_epNN, and the projected end is before 2026-11-02 12:00 CST.  Stop: touch tmp_analysis/tier3/f0_retry/STOP.
ME=/root/autodl-tmp/haoran.zhang; RA=$ME/tmp_analysis/tier3/f0_retry
[ -e $RA/STOP ] && { echo "STOP present: rm $RA/STOP first" >&2; exit 1; }
cd $RA && setsid nohup env -u PYTHONPATH -u CUDA_MPS_PIPE_DIRECTORY -u CUDA_MPS_LOG_DIRECTORY CUDA_VISIBLE_DEVICES= $ME/conda_envs/flashchord/bin/python -I -B $RA/t3retry_v3.py run \
  >> $RA/driver_stdout.log 2>&1 < /dev/null &
echo "t3retry_v3 driver started: pid $! (log $RA/driver.log, heartbeat $RA/heartbeat.log)"
