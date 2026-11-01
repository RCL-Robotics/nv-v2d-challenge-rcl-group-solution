#!/bin/bash
# t3f0_retarget_all.sh: PREP step of the Tier-3 f0 test -- retarget the 30 episodes of data/human_motion_data_tier3_f0
# one at a time (episode order) and gate each with scripts/t3f0_rcheck.py. NEW FILE (2026-10-24).
# Why a re-run (not the opus parquets with the object swapped): soma_to_g1.py feeds the object track into object_* columns,
# hand_contact_active (palm/fingertip-to-object-mesh distance < threshold; used by the recon_body reward) and the
# frame-0 object ground lift; the robot IK never sees the object (rcheck G9 shows the robot side equals the opus parquet).
# GPU choice = scripts/t3opus_helper_v2.py rplan (called read-only: never GPU 4, no MPS, >= 10 GB + 20 GB free, NUMA-1
# GPUs first, 2 own-NUMA cores), under the shared GPU-choice locks opened READ-ONLY for the first 30 s (as driver v2).
# Retarget = scripts/t3f0_retarget.sh in a clean environment (plain CUDA context). Never signals a process.
# STOP: touch tmp_analysis/tier3/f0/retarget/STOP (the running retarget is finished first).
# Usage: setsid nohup bash scripts/t3f0_retarget_all.sh [EP ...] >> tmp_analysis/tier3/f0/retarget/run.log 2>&1 < /dev/null &
set -u
ME=/root/autodl-tmp/haoran.zhang; A=$ME/tmp_analysis/tier3/f0; RDIR=$A/retarget; DATA=$ME/data/human_motion_data_tier3_f0
FCPY=$ME/conda_envs/flashchord/bin/python; KPY=$ME/conda_envs/v2dkit/bin/python
V2HELPER=$ME/scripts/t3opus_helper_v2.py; RETARGET=$ME/scripts/t3f0_retarget.sh; RCHECK=$ME/scripts/t3f0_rcheck.py
CLEANPATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
read -r -a LOCKS <<< "$ME/ch_runs/seedstudy/.gpulock $ME/ch_runs/f0share05/.gpulock $ME/ch_runs/filler/.gpulock"
EPS=${*:-$(seq 0 29)}
mkdir -p $RDIR
exec 9>>$RDIR/.runner.lock; flock -n 9 || { echo "another t3f0_retarget_all holds the lock"; exit 1; }
log(){ echo "[$(date '+%F %T')] $*"; }
release_locks(){ local fd; for fd in 6 7 8; do { flock -u $fd; eval "exec $fd<&-"; } 2>/dev/null; done; return 0; }
take_locks(){
  if [ -f "${LOCKS[0]}" ]; then exec 8<"${LOCKS[0]}" || return 1; flock -w 120 8 || { release_locks; return 1; }; fi
  if [ -f "${LOCKS[1]}" ]; then exec 7<"${LOCKS[1]}" || { release_locks; return 1; }; flock -w 120 7 || { release_locks; return 1; }; fi
  if [ -f "${LOCKS[2]}" ]; then exec 6<"${LOCKS[2]}" || { release_locks; return 1; }; flock -w 120 6 || { release_locks; return 1; }; fi
  return 0; }
rplan(){ ( cd /tmp && env -u PYTHONPATH CUDA_VISIBLE_DEVICES= $FCPY -I -B $V2HELPER rplan 2>/dev/null | tail -1 ); }
seqof(){ $FCPY -I -B -c "import json;m=json.load(open('$DATA/whole_body/soma/track2_manifest.json'));print([k for k,v in m.items() if v['episode_index']==$1][0])"; }
log "t3f0_retarget_all pid $$: episodes $(echo $EPS | paste -sd' ')"
for ep in $EPS; do
  E=$(printf %02d $ep); seq=$(seqof $ep)
  [ -f $RDIR/STOP ] && { log "STOP file: exit before ep$E"; exit 0; }
  if [ -e "$DATA/whole_body/soma/sequence_id=$seq/robot_name=g1/data.parquet" ]; then log "ep$E parquet exists: skipped"; continue; fi
  while :; do
    [ -f $RDIR/STOP ] && { log "STOP file: exit before ep$E"; exit 0; }
    line=$(rplan)
    case $line in OK*) ;; *) log "ep$E waits: $line"; sleep 60; continue;; esac
    take_locks || { log "GPU-choice locks busy: retry"; sleep 30; continue; }
    line=$(rplan)
    case $line in OK*) ;; *) release_locks; log "ep$E waits: $line"; sleep 60; continue;; esac
    read -r tag g uuid cpus nth rest <<< "$line"
    case " 0 1 2 3 5 6 7 " in *" $g "*) ;; *) release_locks; log "ep$E: GPU '$g' not allowed"; sleep 60; continue;; esac
    break
  done
  log "ep$E retarget on gpu$g: $line"
  ( exec env -i HOME=/root USER=root LOGNAME=root SHELL=/bin/bash TERM=dumb PATH=$CLEANPATH \
      bash --noprofile --norc $RETARGET $E $seq $uuid $cpus $nth $RDIR/ep${E}_retarget.log $RDIR/ep${E}_retarget.status ) \
      < /dev/null > $RDIR/ep${E}_retarget.out 2>&1 6<&- 7<&- 8<&- 9>&- &
  rp=$!
  sleep 30; release_locks
  wait $rp; rc=$?
  log "ep$E retarget exit $rc: $(tail -2 $RDIR/ep${E}_retarget.status | paste -sd'|')"
  if [ $rc = 0 ]; then
    for i in $(seq 60); do [ -f $A/frames/ep$E.npz ] && [ -f $A/frames/ep$E.json ] && break; sleep 30; done  # G10 needs t3f0_frames.py
    ( cd /tmp && exec env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= nice -n 10 $KPY -I -B $RCHECK --ep $ep \
        --log $RDIR/ep${E}_retarget.log ) > $RDIR/ep${E}_rcheck.out 2>&1 9>&-
    log "ep$E rcheck exit $?: $(grep -E 'FAIL|all_ok' $RDIR/ep${E}_rcheck.out | paste -sd'|' | cut -c1-400)"
  else
    echo "retarget exit $rc" > $RDIR/RETARGET_FAILED_ep$E; log "ep$E RETARGET FAILED (marker written)"
  fi
done
log "t3f0_retarget_all done"
