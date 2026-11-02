#!/bin/bash
# k3full_driver.sh: the FULL K3fix arm (arm folder C3r2_tier2_k3fix). Pre-registration tmp_analysis/tier2/k3full/PREREG_v3.md
# (sha256 in PREREG_v3.sha256). User 2026-10-22: "so you can train on more tasks right?" -> extend the K3fix training to every
# Tier-2 episode that passes the noisy-only input rule G1-G4 (scripts/k3full_gates.py -> k3full/build/gates_all30.json).
#   Jobs (k3full/jobs.json, launch order = raw C3r2_tier2 final.json help_zero_since ascending, ties by episode, ep04 last):
#     ep11 ep16 ep17 ep06 ep10 ep02 ep12 ep09 ep01 ep13 ep14 ep21 ep24 ep25 ep27 ep29 ep04   (run dir epNN_k3fix_s42)
#   Not trained here: gate FAIL 00 05 15 19 22 26 28 (raw run used); already on the K3fix input: ep03/07/18 (k3fix pilot
#   stage 1) and ep08/20/23 (k3easy, K3 input = K3fix input byte-identically).
#   Recipe: exactly the raw C3r2_tier2 run per episode (code/fc_r2 @61fa5d3, g1_recon_body_coord_rsi_r2, seed 42, 45000,
#   deadline 35156 / zero_ramp 3906 / help-zero tail 5860, scene.support as raw); only the data root
#   (data/human_motion_data_tier2_k3fix), output_dir and logging differ (k3full/cfgcheck.json).
#   ep17 / ep09 / ep04 are the k3fix pilot's stage-2/3 jobs (same names, same argv): at their launch this driver ALSO
#   appends the launch line to tmp_analysis/tier2/k3fix_pilot/pids.tsv, so the running k3fix driver treats them as its own
#   (watch, post-training step, PREREG_v2 stage-2/3 verdict). Without that line it would loop on them after training.
# Fill rule (helper 'plan', re-checked under the shared GPU-choice locks ch_runs/{seedstudy,f0share05,filler}/.gpulock,
#   opened read-only and held 180 s after a launch): never GPU 4, no MPS, < 3 trainings of anybody before the start,
#   >= 16.5 GB + 20 GB free + ramp debt of every young training, 2 free physical cores of the GPU's own NUMA node
#   (track2_queue_tier2v2_helper.cpu_slice, never borrowed), NUMA-1 GPUs first; strictly in order (the next job waits).
#   YIELD: no start while the t3opus driver (run mode) still has a clip of its wave(s) waiting to be launched (Tier 3 first).
#   EQUAL SHARE: no start if the slots left after it would fall below the T1fix launcher's reserve (pending T1fix arms).
# Post-training (background, after final.json): new jobs -> k3full_helper.py post = k3fix_helper's post on this arm's
#   records (export 2e-4 + ONE official own-input evaluation of final.json's checkpoint + LOCAL clean diagnostic with
#   frame_refs_k3fix); ep17/09/04 -> the k3fix driver does it while alive, else k3fix_helper's post via this helper.
#   A trainer that dies without final.json -> k3full/CRASHED_<job>, NO automatic relaunch (the raw run is used).
# Modes:
#   run    preflight (helper check incl. PREREG_v3 sha256), then the loop below
#   adopt  as run but never launches (driver restart while trainers run: rm STOP first)
#   dry    checks, config check, GPU view, yield and share state, job states, plan of every FRESH job (later ones as if
#          the earlier had started; with and without the yield), exact commands, verdict now; starts and writes nothing
# Files: tmp_analysis/tier2/k3full/{driver.log, heartbeat.log, pids.tsv, driver.pid, STOP, CRASHED_<job>, post.log,
#        eval/<job>/..., VERDICT.{json,txt}, VERDICT_clean_LOCAL_NOT_FOR_UPLOAD.json}; runs fc_runs/C3r2_tier2_k3fix/<job>,
#        logs fc_runs/C3r2_tier2_k3fix/logs/<job>.log. The t2watch watcher discovers the run dirs by itself.
# STOP: touch tmp_analysis/tier2/k3full/STOP -> exits at its next loop; trainings and evaluations keep running.
#       Restart: rm STOP; setsid nohup bash scripts/k3full_driver.sh adopt|run >> tmp_analysis/tier2/k3full/driver_stdout.log 2>&1 < /dev/null &
# The driver never signals a process. GPU 4 is never used. No MPS.
# Launch: cd /root/autodl-tmp/haoran.zhang && setsid nohup bash scripts/k3full_driver.sh run >> tmp_analysis/tier2/k3full/driver_stdout.log 2>&1 < /dev/null &
# Dry run: bash scripts/k3full_driver.sh dry
ME=/root/autodl-tmp/haoran.zhang
MODE=${1:-}
case $MODE in run|adopt|dry) [ $# = 1 ] || { echo "usage: bash $0 run|adopt|dry" >&2; exit 2; };; *) echo "usage: bash $0 run|adopt|dry" >&2; exit 2;; esac
K=$ME/tmp_analysis/tier2/k3full; RUNS=$ME/fc_runs/C3r2_tier2_k3fix; CODE=$ME/code/fc_r2
K3FIX_PIDS=$ME/tmp_analysis/tier2/k3fix_pilot/pids.tsv
FCPY=$ME/conda_envs/flashchord/bin/python; HELPER=$ME/scripts/k3full_helper.py
DLOG=$K/driver.log; HB=$K/heartbeat.log; PIDS=$K/pids.tsv; STOPF=$K/STOP
POLL=60; HEARTBEAT_S=600; VERDICT_S=1800; LAUNCH_HOLD=180; LOCK_WAIT=120
read -r -a LOCKS <<< "$ME/ch_runs/seedstudy/.gpulock $ME/ch_runs/f0share05/.gpulock $ME/ch_runs/filler/.gpulock"

# ---- environment of every training: exactly k3fix_driver.sh's (fc_env.sh, PYTHONPATH=<code>/src, no bytecode, caches
#      under ME, PCI order, no MPS, no W&B key); HUMAN_MOTION_DATA_DIR and the GPU are set per job
source $ME/scripts/fc_env.sh
export PYTHONPATH=$CODE/src PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=$ME/.cache/pycache XDG_CACHE_HOME=$ME/.cache
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_VISIBLE_DEVICES WANDB_API_KEY PYTHONHOME PYTHONSTARTUP

hlp(){ env -u PYTHONPATH $FCPY -I -B $HELPER "$@" 2> >(grep -v -e UnsupportedFieldAttributeWarning -e 'warnings.warn(' >&2); }
if [ "$MODE" = dry ]; then log(){ echo "[dry] $*"; }; else log(){ echo "[$(date '+%F %T')] $*" >> $DLOG; }; fi
die(){ log "ABORT: $*"; echo "k3full_driver: $*" >&2
  [ "$MODE" != dry ] && [ "$(cat $K/driver.pid 2>/dev/null)" = "$$" ] && rm -f $K/driver.pid; exit 1; }
declare -A NOTE SEEN
note(){ local sig=${3:-$2}; [ "${NOTE[$1]:-}" = "$sig" ] && return 0; NOTE[$1]=$sig; [ -n "$2" ] && log "$2"; return 0; }

ORDER=$(hlp order | paste -sd' '); NJOBS=$(echo $ORDER | wc -w)
[ "$NJOBS" -gt 0 ] || die "no job order from the helper (jobs.json?)"

# ---- singleton + preflight
if [ "$MODE" != dry ]; then
  mkdir -p $K $RUNS/logs || die "cannot create $K / $RUNS/logs"
  exec 9>>$K/.driver.lock
  flock -n 9 || die "another k3full_driver.sh holds $K/.driver.lock"
  [ -f $STOPF ] && die "STOP file $STOPF present: remove it to start"
  echo $$ > $K/driver.pid
  log "k3full driver $MODE: pid $$; code $CODE ($(git -C $CODE rev-parse --short=7 HEAD 2>/dev/null)); runs $RUNS; $NJOBS jobs in order: $ORDER"
fi
CHECK_OUT=$(hlp check); CHECK_RC=$?
if [ "$MODE" = dry ]; then
  echo "$CHECK_OUT" | sed 's/^/[dry] /'
else
  echo "$CHECK_OUT" | grep -v ' ok ' >> $DLOG; echo "$CHECK_OUT" | tail -1 | sed "s/^/[$(date '+%F %T')] /" >> $DLOG
  [ $CHECK_RC = 0 ] || die "preflight failed (see above): nothing launched"
fi

# ---- GPU-choice locks shared with the other queues, opened READ-ONLY (a missing file is skipped, never created)
release_locks(){ local fd; for fd in 6 7 8; do { flock -u $fd; eval "exec $fd<&-"; } 2>/dev/null; done; return 0; }
take_locks(){
  if [ -f "${LOCKS[0]}" ]; then exec 8<"${LOCKS[0]}" || return 1; flock -w $LOCK_WAIT 8 || { release_locks; return 1; }; fi
  if [ -f "${LOCKS[1]}" ]; then exec 7<"${LOCKS[1]}" || { release_locks; return 1; }; flock -w $LOCK_WAIT 7 || { release_locks; return 1; }; fi
  if [ -f "${LOCKS[2]}" ]; then exec 6<"${LOCKS[2]}" || { release_locks; return 1; }; flock -w $LOCK_WAIT 6 || { release_locks; return 1; }; fi
  return 0; }

# ---- the training command of one job (argv from the helper) and its launch
TRAIN=()
train_cmd(){ mapfile -d '' -t TRAIN < <(hlp cmd --job $1); [ ${#TRAIN[@]} -gt 10 ]; }
DATA=$ME/data/human_motion_data_tier2_k3fix
print_cmd(){ # job gpu cpus nth
  printf '( cd %q && exec setsid nohup env K3FIX=1 K3FIX_JOB=%s K3FULL=1 HUMAN_MOTION_DATA_DIR=%s CUDA_VISIBLE_DEVICES=%s WANDB_MODE=offline OMP_NUM_THREADS=%s MKL_NUM_THREADS=%s OPENBLAS_NUM_THREADS=%s taskset -c %s' \
    "$CODE" $1 "$DATA" $2 $4 $4 $4 $3
  printf ' %q' "${TRAIN[@]}"; echo " ) >> $RUNS/logs/$1.log 2>&1 < /dev/null &"; }
launch(){ # job gpu cpus nth
  local job=$1 g=$2 cpus=$3 nth=$4 rp tp ok nat line
  ok=$(hlp launch-ok --job $job) || { note nl$job "$job NOT launched: $ok"; return 1; }
  train_cmd $job || { log "$job NOT launched: no training command from the helper"; return 1; }
  [ "$(hlp data --job $job)" = "$DATA" ] || { log "$job NOT launched: data root"; return 1; }
  case " 0 1 2 3 5 6 7 " in *" $g "*) ;; *) log "$job NOT launched: GPU '$g' not allowed (never 4)"; return 1;; esac
  nat=$(hlp native --job $job)
  log "$job launch on gpu$g cpus $cpus OMP $nth ($ok): $(print_cmd $job $g $cpus $nth)"
  ( cd $CODE && exec setsid nohup env K3FIX=1 K3FIX_JOB=$job K3FULL=1 HUMAN_MOTION_DATA_DIR=$DATA CUDA_VISIBLE_DEVICES=$g \
      WANDB_MODE=offline OMP_NUM_THREADS=$nth MKL_NUM_THREADS=$nth OPENBLAS_NUM_THREADS=$nth \
      taskset -c $cpus "${TRAIN[@]}" ) >> $RUNS/logs/$job.log 2>&1 < /dev/null 6<&- 7<&- 8<&- 9>&- &
  rp=$!
  line=$(printf '%s\t%s\t%s\t%s\t%s\t%s\tlaunched' $job $rp $g $cpus $nth $(date +%s))
  echo "$line" >> $PIDS
  if [ "$nat" = yes ]; then
    echo "$line" >> $K3FIX_PIDS
    log "$job is a k3fix pilot job (PREREG_v2 stage 2/3): launch line also appended to $K3FIX_PIDS (the k3fix driver watches it as its own)"
  fi
  sleep 10; tp=$(hlp trainer --job $job)
  log "$job launched: pid $rp; trainer pid(s) now: ${tp:-not visible yet}; log $RUNS/logs/$job.log"
  return 0; }
EXCL=""
try_launch(){ # job: plan -> locks -> plan again -> launch -> hold the locks LAUNCH_HOLD s
  local job=$1 line tag g cpus nth rest
  line=$(hlp plan --job $job --exclude "$EXCL")
  case $line in OK*) ;; *) note plan "$job waits: ${line#WAIT }" "$job $(echo "$line" | cut -c1-48)"; return 1;; esac
  if ! take_locks; then note locks "GPU-choice locks busy for > ${LOCK_WAIT}s: retrying"; return 1; fi
  note locks ""
  line=$(hlp plan --job $job --exclude "$EXCL")   # re-check under the locks
  case $line in OK*) ;; *) release_locks; note plan "$job waits: ${line#WAIT }" "$job $(echo "$line" | cut -c1-48)"; return 1;; esac
  note plan ""
  read -r tag g cpus nth rest <<< "$line"
  log "$job slot decision: $line"
  if launch $job $g $cpus $nth; then EXCL="$EXCL,$cpus"; sleep $LAUNCH_HOLD; EXCL=""; fi   # review F1: reset after the hold (trainer is taskset-pinned; cpu_slice sees its cores)   # hold: the new job's memory shows up
  release_locks; return 0; }
launch_post(){ # job detail
  local job=$1
  mkdir -p $K/eval/$job
  ( exec setsid nohup env -u PYTHONPATH $FCPY -I -B $HELPER post --job $job ) >> $K/eval/$job/post_stdout.log 2>&1 < /dev/null 6<&- 7<&- 8<&- 9>&- &
  log "$job training finished ($2): post-training step launched (pid $!)"
  sleep 5; }

# ---- dry run: everything the run would do now; nothing started or written
if [ "$MODE" = dry ]; then
  log "preflight exit $CHECK_RC ($([ $CHECK_RC = 0 ] && echo 'run would pass the preflight' || echo 'run would ABORT'))"
  log "config check (hydra compose, CPU; --no-write):"
  hlp cfgcheck --no-write | grep -E '^== |cfgcheck:' | sed 's/^/[dry]   /'
  log "GPU view now:"; hlp gpus | sed 's/^/[dry]   /'
  log "Tier-3 yield now: $(hlp yield)"
  log "T1fix equal share now: $(hlp share)"
  log "job states now:"
  declare -A JST
  while read -r _ job stt rest; do JST[$job]=$stt; echo "[dry]   JOB $job $stt $rest"; done < <(hlp state)
  log "STOP file: $([ -f $STOPF ] && echo present || echo absent); GPU-choice lock files (read-only, never created): $(for f in "${LOCKS[@]}"; do printf '%s=%s ' "${f#$ME/}" "$([ -f "$f" ] && echo present || echo 'missing(skipped)')"; done)"
  ex=""; assume=""; declare -A ASSUMED; first=1
  for job in $ORDER; do
    if [ "${JST[$job]:-}" = FRESH ]; then
      if [ $first = 1 ]; then log "$job (head of the queue) -> with the yield: $(hlp plan --job $job)"; first=0; fi
      line=$(hlp plan --job $job --exclude "$ex" --assume "$assume" --no-yield)
      log "$job -> fill rule only (yield ignored; earlier jobs assumed started): $line"
      log "$job -> launch guard: $(hlp launch-ok --job $job)"
      read -r tag g cpus nth rest <<< "$line"
      if [ "$tag" != OK ]; then g="<gpu>"; cpus="<cpus>"; nth="<n>"; else ex="$ex,$cpus"; ASSUMED[$g]=$(( ${ASSUMED[$g]:-0} + 1 ))
        assume=$(for k in "${!ASSUMED[@]}"; do printf '%s:%s,' $k ${ASSUMED[$k]}; done); fi
    else
      log "$job: state ${JST[$job]:-?}: not launched by run"; g="<gpu>"; cpus="<cpus>"; nth="<n>"
    fi
    train_cmd $job || { log "$job: no command"; continue; }
    echo "---- $job exact launch command (cwd $CODE; log $RUNS/logs/$job.log)$([ "$(hlp native --job $job)" = yes ] && echo "; + launch line appended to $K3FIX_PIDS"):"
    print_cmd $job $g $cpus $nth
  done
  log "NOTE: in run mode the jobs start one at a time in this order (>= 180 s apart), each planned on the GPU state then"
  echo "     env (all trainings): PATH=${PATH%%:*}:... PYTHONPATH=$PYTHONPATH PYTHONDONTWRITEBYTECODE=$PYTHONDONTWRITEBYTECODE PYTHONPYCACHEPREFIX=$PYTHONPYCACHEPREFIX XDG_CACHE_HOME=$XDG_CACHE_HOME CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE WANDB_DIR=$WANDB_DIR CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-<unset>} WANDB_API_KEY=${WANDB_API_KEY:+<set>}${WANDB_API_KEY:-<unset>} PYTHONUNBUFFERED=$PYTHONUNBUFFERED"
  echo "---- after final.json (every job): ( exec setsid nohup env -u PYTHONPATH $FCPY -I -B $HELPER post --job <job> ) >> $K/eval/<job>/post_stdout.log 2>&1 < /dev/null &"
  echo "     post commands of the first new job (training GPU from pids.tsv; shown for gpu 5):"
  env -u PYTHONPATH $FCPY -I -B - $HELPER <<'PY' 2>/dev/null
import sys, importlib.util
spec = importlib.util.spec_from_file_location("k3fullh", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.setup()
job = next(j for j in m.ORDER if not m.native(j))
w = m.EVROOT / job
print(f"       {job}: " + " ".join(m.F2.eval_command(job, 5, w)))
print(f"       {job} then (CPU): " + " ".join(m.F2.score_command(job, w)))
nat = next(j for j in m.ORDER if m.native(j))
print(f"       {nat} (k3fix pilot job): k3fix_helper.py post --job {nat} -> " + " ".join(m.F.eval_command(nat, 5, m.F.EVROOT / nat)))
PY
  log "verdict now (--no-write):"; hlp verdict --no-write | sed 's/^/[dry]   /'
  log "dry run done: nothing was started or written"
  exit 0
fi

# ---- adopt / run: start state
hlp state | while read -r line; do log "start state: $line"; done
[ "$MODE" = adopt ] && log "adopt mode: no launch is ever made"

# ---- main loop
last_beat=0; last_verdict=$(date +%s)
while :; do
  if [ -f $STOPF ]; then log "STOP file: driver exits; trainings and evaluations keep running (restart: rm STOP, then 'adopt' or 'run')"
    rm -f $K/driver.pid; exit 0; fi
  mapfile -t ST < <(hlp state)
  [ ${#ST[@]} = $NJOBS ] || { note state "state helper returned ${#ST[@]} lines (expected $NJOBS): retrying"; sleep $POLL; continue; }
  note state ""
  settled=0; newterm=0; head_done=0
  for line in "${ST[@]}"; do
    read -r _ job stt rest <<< "$line"
    note st$job "$job $stt $rest" "$stt"
    case $stt in
      FRESH)
        if [ "$MODE" = run ] && [ $head_done = 0 ]; then head_done=1     # strict order: only the first FRESH job is tried
          grep -q "^$job"$'\t' $PIDS 2>/dev/null || try_launch $job
        elif [ "$MODE" = adopt ]; then settled=$((settled + 1)); fi;;
      RUNNING|STARTING|POST_RUNNING|POST_RETRY_WAIT|POST_BY_K3FIX) ;;
      DEAD) hlp crash --job $job --why "$rest" > /dev/null
            log "$job CRASHED: $rest -> marker $K/CRASHED_$job; no automatic relaunch (the raw run is used for this episode)"; newterm=1;;
      CRASHED) settled=$((settled + 1)); [ -z "${SEEN[$job]:-}" ] && { SEEN[$job]=crashed; newterm=1; };;
      TRAINED) launch_post $job "$rest";;
      POST_DONE) settled=$((settled + 1)); [ -z "${SEEN[$job]:-}" ] && { SEEN[$job]=post; newterm=1; };;
      *) note unk$job "$job: unknown state '$stt'";;
    esac
  done
  now=$(date +%s)
  if [ $newterm = 1 ] || [ $((now - last_verdict)) -ge $VERDICT_S ]; then last_verdict=$now
    [ $newterm = 1 ] && log "verdict update:"
    hlp verdict > $K/verdict_last.log 2>&1 || log "verdict refused or failed (exit $?)"
    [ $newterm = 1 ] && head -3 $K/verdict_last.log >> $DLOG; fi
  if [ $((now - last_beat)) -ge $HEARTBEAT_S ]; then last_beat=$now
    echo "[$(date '+%F %T')] HEARTBEAT driver $$ | $(hlp heartbeat)" >> $HB; fi
  [ $settled -ge $NJOBS ] && break
  sleep $POLL
done
log "every job is settled: verdict"
hlp verdict >> $DLOG 2>&1 || log "verdict refused or failed (exit $?)"
echo "[$(date '+%F %T')] DONE driver $$ | $(hlp heartbeat)" >> $HB
rm -f $K/driver.pid
exit 0
