#!/bin/bash
# k3easy_driver.sh: the easy-clip K3 trainability test (user-approved 2026-10-21; pre-registration
# tmp_analysis/tier2/k3easy/PREREG.md, sha256 in PREREG.sha256).
#   C3r2 exactly as the raw-input Tier-2 arm C3r2_tier2 (code/fc_r2 @61fa5d3, experiment g1_recon_body_coord_rsi_r2,
#   seed 42, training.max_iterations 45000, experiment-default RSI deadline 35156 / zero_ramp 3906 / help-zero early
#   stop 5860, scene.support as the raw arm) trained on data/human_motion_data_tier2_k3 (noisy Tier-2 body + K3 object):
#     ep20 rice_cooker  -> GPU 3        ep23 toy_stroller -> GPU 3        ep08 skinny_wood_chair -> GPU 6
#   The ONLY recipe difference from C3r2_tier2 is the data root (task.parquet, HUMAN_MOTION_DATA_DIR). output_dir
#   (fc_runs/C3r2_tier2_k3easy/epNN), hydra.run.dir and logging (offline + metrics.jsonl, experiment name) do not change
#   training: k3easy/cfgcheck.json composes both configs. NOT the stretched curriculum.
#   The stopped K3 queue (fc_runs/queue_tier2v2, STOP + PAUSE) is not touched.
# Modes:
#   run    preflight (scripts/k3easy_helper.py check; PREREG sha256 included), then launch every FRESH episode on its GPU,
#          one at a time, when the slot rule allows it (after the start < 4 trainings of anybody on the GPU, i.e. at most
#          3; no MPS; >= 16.5 GB + 20 GB free plus the ramp debt of our own young trainers), CPU slice = free physical
#          cores (+HT siblings) of the GPU's NUMA node (track2_queue_tier2v2_helper cpu_slice: GPUs 3-7 -> cores 52-99
#          + 156-203, eval lanes excluded, cores any other process holds as a slice excluded), shared among this test's
#          launches still to come (ceil(free / remaining), 2-4 cores each; the other node only if < 2 are free), under the shared
#          GPU-choice locks (opened read-only; held 180 s after a launch). Then waits. A start is never made over existing
#          model_* / rsi_* files, a final.json, a CRASHED marker or an earlier launch of this test.
#   adopt  as run, but never launches: for a driver restart while the trainers run (remove STOP first).
#   dry    prints the checks, the config diff, the GPU view, the slot/CPU decision and the exact commands; starts
#          nothing, writes nothing.
# While waiting (every 60 s):
#   - a trainer that ended with final.json -> post-training step in the background (k3easy_helper.py post: export at
#     2e-4 + ONE official evaluation, eval_all.py --checkpoint-from-final with track2_tier2v2_post.py's arguments, of
#     final.json's checkpoint against its own K3 input on the training GPU, holding k3easy/evallock/gpu<g>.lock and, if it
#     exists, fc_runs/queue_runin/evallock/gpu<g>.lock (read-only, never created); then the local clean scoring with
#     --frame-references kalman/eval/frame_refs_k3 into *_LOCAL_NOT_FOR_UPLOAD files); a technical eval failure is
#     retried once after 10 min;
#   - after each finished post / crash: k3easy_helper.py verdict (refuses when PREREG.sha256 does not match);
#   - a trainer that died without final.json -> k3easy/CRASHED_epNN marker; NO automatic relaunch;
#   - heartbeat line every 10 min in k3easy/heartbeat.log.
#   The driver exits after writing the verdict once every episode is settled (post finished, or crashed).
# Files: tmp_analysis/tier2/k3easy/{driver.log, heartbeat.log, pids.tsv, driver.pid, STOP, CRASHED_epNN, post.log,
#        eval/epNN/{post.json, eval_all.log, results/, export/, *_LOCAL_NOT_FOR_UPLOAD*}, VERDICT.{json,txt},
#        fail_rates_vs_raw.csv, VERDICT_clean_LOCAL_NOT_FOR_UPLOAD.json}; runs fc_runs/C3r2_tier2_k3easy/epNN, training
#        logs fc_runs/C3r2_tier2_k3easy/logs/epNN.log.
# STOP: touch tmp_analysis/tier2/k3easy/STOP -> the driver exits at its next loop; trainings and evaluations keep running.
#       Restart: rm STOP; setsid nohup bash scripts/k3easy_driver.sh adopt >> tmp_analysis/tier2/k3easy/driver_stdout.log 2>&1 < /dev/null &
# The driver never signals a process. GPU 4 is never used. No MPS.
# Launch:  setsid nohup bash scripts/k3easy_driver.sh run > tmp_analysis/tier2/k3easy/driver_stdout.log 2>&1 < /dev/null &
# Dry run: bash scripts/k3easy_driver.sh dry
ME=/root/autodl-tmp/haoran.zhang
MODE=${1:-}
K=$ME/tmp_analysis/tier2/k3easy; RUNS=$ME/fc_runs/C3r2_tier2_k3easy; CODE=$ME/code/fc_r2
DATA=$ME/data/human_motion_data_tier2_k3
FCPY=$ME/conda_envs/flashchord/bin/python; HELPER=$ME/scripts/k3easy_helper.py
DLOG=$K/driver.log; HB=$K/heartbeat.log; PIDS=$K/pids.tsv; STOPF=$K/STOP
POLL=60; HEARTBEAT_S=600; LAUNCH_HOLD=180; LOCK_WAIT=120
ORDER="20 23 08"
declare -A GPU_OF=([20]=3 [23]=3 [08]=6)
read -r -a LOCKS <<< "$ME/ch_runs/seedstudy/.gpulock $ME/ch_runs/f0share05/.gpulock $ME/ch_runs/filler/.gpulock"
case $MODE in run|adopt|dry) ;; *) echo "usage: bash $0 run|adopt|dry" >&2; exit 2;; esac

# ---- environment of every training: exactly track2_queue_tier2v2.sh's (fc_env.sh, PYTHONPATH=<code>/src, no bytecode,
#      caches under ME, PCI order, no MPS, no W&B key); HUMAN_MOTION_DATA_DIR and the GPU are set per job
source $ME/scripts/fc_env.sh
export PYTHONPATH=$CODE/src PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=$ME/.cache/pycache XDG_CACHE_HOME=$ME/.cache
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_VISIBLE_DEVICES WANDB_API_KEY PYTHONHOME PYTHONSTARTUP

hlp(){ env -u PYTHONPATH $FCPY -I -B $HELPER "$@" 2> >(grep -v -e UnsupportedFieldAttributeWarning -e 'warnings.warn(' >&2); }
if [ "$MODE" = dry ]; then log(){ echo "[dry] $*"; }; else log(){ echo "[$(date '+%F %T')] $*" >> $DLOG; }; fi
die(){ log "ABORT: $*"; echo "k3easy_driver: $*" >&2; exit 1; }
declare -A NOTE SEEN
note(){ local sig=${3:-$2}; [ "${NOTE[$1]:-}" = "$sig" ] && return 0; NOTE[$1]=$sig; [ -n "$2" ] && log "$2"; return 0; }
for g in "${GPU_OF[@]}"; do [ "$g" = 4 ] && die "GPU 4 is never used"; done

# ---- singleton + preflight
if [ "$MODE" != dry ]; then
  mkdir -p $K $RUNS/logs || die "cannot create $K / $RUNS/logs"
  exec 9>>$K/.driver.lock
  flock -n 9 || die "another k3easy_driver.sh holds $K/.driver.lock"
  [ -f $STOPF ] && die "STOP file $STOPF present: remove it to start"
  echo $$ > $K/driver.pid
  log "k3easy driver $MODE: pid $$; code $CODE ($(git -C $CODE rev-parse --short=7 HEAD 2>/dev/null)); data $DATA; runs $RUNS"
fi
CHECK_OUT=$(hlp check); CHECK_RC=$?
if [ "$MODE" = dry ]; then
  echo "$CHECK_OUT" | sed 's/^/[dry] /'
else
  echo "$CHECK_OUT" | grep -v ' ok ' >> $DLOG; echo "$CHECK_OUT" | tail -1 | sed "s/^/[$(date '+%F %T')] /" >> $DLOG
  [ $CHECK_RC = 0 ] || die "preflight failed (see above): nothing launched"
fi

# ---- GPU-choice locks shared with the other queues (seedstudy -> f0share05 -> filler, as gpu_filler.sh /
#      track2_queue_tier2v2.sh), opened READ-ONLY (flock needs no write access; a missing file is skipped, never created)
release_locks(){ local fd; for fd in 6 7 8; do { flock -u $fd; eval "exec $fd<&-"; } 2>/dev/null; done; return 0; }
take_locks(){
  if [ -f "${LOCKS[0]}" ]; then exec 8<"${LOCKS[0]}" || return 1; flock -w $LOCK_WAIT 8 || { release_locks; return 1; }; fi
  if [ -f "${LOCKS[1]}" ]; then exec 7<"${LOCKS[1]}" || { release_locks; return 1; }; flock -w $LOCK_WAIT 7 || { release_locks; return 1; }; fi
  if [ -f "${LOCKS[2]}" ]; then exec 6<"${LOCKS[2]}" || { release_locks; return 1; }; flock -w $LOCK_WAIT 6 || { release_locks; return 1; }; fi
  return 0; }

# ---- the training command of one episode (argv from the helper) and its launch
TRAIN=()
train_cmd(){ mapfile -d '' -t TRAIN < <(hlp cmd --ep $1); [ ${#TRAIN[@]} -gt 10 ]; }
print_cmd(){ # ep gpu cpus nth: the exact launch line
  printf '( cd %q && exec setsid nohup env K3EASY=1 K3EASY_EP=%s HUMAN_MOTION_DATA_DIR=%s CUDA_VISIBLE_DEVICES=%s WANDB_MODE=offline OMP_NUM_THREADS=%s MKL_NUM_THREADS=%s OPENBLAS_NUM_THREADS=%s taskset -c %s' \
    "$CODE" $1 "$DATA" $2 $4 $4 $4 $3
  printf ' %q' "${TRAIN[@]}"; echo " ) >> $RUNS/logs/ep$1.log 2>&1 < /dev/null &"; }
launch(){ # ep gpu cpus nth
  local ep=$1 g=$2 cpus=$3 nth=$4 out=$RUNS/ep$1 rp tp
  train_cmd $ep || { log "ep$ep NOT launched: no training command from the helper"; return 1; }
  if compgen -G "$out/model_*.safetensors" > /dev/null || compgen -G "$out/rsi_*.safetensors" > /dev/null || [ -e $out/final.json ]; then
    log "ep$ep NOT launched: $out already has checkpoints or final.json (never a fresh start over them)"; return 1; fi
  [ -e $K/CRASHED_ep$ep ] && { log "ep$ep NOT launched: CRASHED marker present"; return 1; }
  [ "$g" = 4 ] && { log "ep$ep NOT launched: GPU 4"; return 1; }
  log "ep$ep launch on gpu$g cpus $cpus OMP $nth: $(print_cmd $ep $g $cpus $nth)"
  ( cd $CODE && exec setsid nohup env K3EASY=1 K3EASY_EP=$ep HUMAN_MOTION_DATA_DIR=$DATA CUDA_VISIBLE_DEVICES=$g \
      WANDB_MODE=offline OMP_NUM_THREADS=$nth MKL_NUM_THREADS=$nth OPENBLAS_NUM_THREADS=$nth \
      taskset -c $cpus "${TRAIN[@]}" ) >> $RUNS/logs/ep$ep.log 2>&1 < /dev/null 6<&- 7<&- 8<&- 9>&- &
  rp=$!
  printf '%s\t%s\t%s\t%s\t%s\t%s\tlaunched\n' $ep $rp $g $cpus $nth $(date +%s) >> $PIDS
  sleep 10; tp=$(hlp trainer --ep $ep)
  log "ep$ep launched: pid $rp; trainer pid(s) now: ${tp:-not visible yet}; log $RUNS/logs/ep$ep.log"
  return 0; }
EXCL=""
try_launch(){ # ep: plan -> locks -> plan again -> launch -> hold the locks LAUNCH_HOLD s
  local ep=$1 g=${GPU_OF[$1]} line tag cpus nth cnote rest
  line=$(hlp plan --ep $ep --gpu $g --exclude "$EXCL")
  case $line in OK*) ;; *) note plan$ep "ep$ep waits for gpu$g: ${line#WAIT }" "$(echo "$line" | cut -c1-40)"; return 1;; esac
  if ! take_locks; then note locks "GPU-choice locks busy for > ${LOCK_WAIT}s: retrying"; return 1; fi
  note locks ""
  line=$(hlp plan --ep $ep --gpu $g --exclude "$EXCL")   # re-check under the locks
  case $line in OK*) ;; *) release_locks; note plan$ep "ep$ep waits for gpu$g: ${line#WAIT }" "$(echo "$line" | cut -c1-40)"; return 1;; esac
  note plan$ep ""
  read -r tag cpus nth cnote rest <<< "$line"
  log "ep$ep slot decision: $line"
  if launch $ep $g $cpus $nth; then EXCL="$EXCL,$cpus"; sleep $LAUNCH_HOLD; fi   # hold: the new job's memory shows up
  release_locks; return 0; }
launch_post(){ # ep detail
  local ep=$1 g=${GPU_OF[$1]}
  mkdir -p $K/eval/ep$ep
  ( exec setsid nohup env -u PYTHONPATH $FCPY -I -B $HELPER post --ep $ep --gpu $g ) >> $K/eval/ep$ep/post_stdout.log 2>&1 < /dev/null 6<&- 7<&- 8<&- 9>&- &
  log "ep$ep training finished ($2): post-training step launched (pid $!; gpu$g): export 2e-4 -> official eval vs own input -> local clean diagnostic"
  sleep 5; }

# ---- dry run: everything the run would do now, nothing started or written
if [ "$MODE" = dry ]; then
  log "preflight exit $CHECK_RC ($([ $CHECK_RC = 0 ] && echo 'run would start' || echo 'run would ABORT'))"
  log "config check (hydra compose, CPU; --no-write):"
  hlp cfgcheck --no-write | sed 's/^/[dry]   /'
  log "GPU view now:"; hlp gpus | sed 's/^/[dry]   /'
  log "episode states now:"; hlp state | sed 's/^/[dry]   /'
  log "STOP file: $([ -f $STOPF ] && echo present || echo absent); queue_tier2v2 STOP/PAUSE (untouched): $(ls $ME/fc_runs/queue_tier2v2/STOP $ME/fc_runs/queue_tier2v2/PAUSE 2>/dev/null | xargs -n1 basename | paste -sd' ')"
  log "GPU-choice lock files (opened read-only by run, never created): $(for f in "${LOCKS[@]}"; do printf '%s=%s ' "${f#$ME/}" "$([ -f "$f" ] && echo present || echo 'missing(skipped)')"; done)"
  log "eval-lane locks: k3easy/evallock/gpu<g>.lock (own); fc_runs/queue_runin/evallock: gpu3.lock $([ -f $ME/fc_runs/queue_runin/evallock/gpu3.lock ] && echo present || echo 'absent (not held)'), gpu6.lock $([ -f $ME/fc_runs/queue_runin/evallock/gpu6.lock ] && echo 'present (held read-only during our eval)' || echo absent)"
  ex=""; rem=3
  for ep in $ORDER; do
    g=${GPU_OF[$ep]}
    line=$(hlp plan --ep $ep --gpu $g --exclude "$ex" --remaining $rem); rem=$((rem - 1))
    log "ep$ep -> gpu$g: $line"
    read -r tag cpus nth cnote rest <<< "$line"
    if [ "$tag" != OK ]; then cpus="<cpus>"; nth="<n>"; else ex="$ex,$cpus"; fi
    train_cmd $ep || { log "ep$ep: no command"; continue; }
    echo "---- ep$ep exact launch command (cwd $CODE; log $RUNS/logs/ep$ep.log):"
    print_cmd $ep $g $cpus $nth
  done
  log "NOTE: the dry plan counts each GPU as it is now; in run mode the second GPU-3 job is planned after the first is"
  log "      launched (its training then counts, plus its ramp debt of up to 16.5 GB until it has allocated its memory)."
  echo "     env (all trainings): PATH=${PATH%%:*}:... PYTHONPATH=$PYTHONPATH PYTHONDONTWRITEBYTECODE=$PYTHONDONTWRITEBYTECODE PYTHONPYCACHEPREFIX=$PYTHONPYCACHEPREFIX XDG_CACHE_HOME=$XDG_CACHE_HOME CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE WANDB_DIR=$WANDB_DIR CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-<unset>} WANDB_API_KEY=${WANDB_API_KEY:+<set>}${WANDB_API_KEY:-<unset>} PYTHONUNBUFFERED=$PYTHONUNBUFFERED"
  for ep in $ORDER; do
    echo "---- ep$ep after final.json: ( exec setsid nohup env -u PYTHONPATH $FCPY -I -B $HELPER post --ep $ep --gpu ${GPU_OF[$ep]} ) >> $K/eval/ep$ep/post_stdout.log 2>&1 < /dev/null &"
  done
  echo "     post runs (on the training GPU, nice 10, taskset 100-103,204-207):"
  env -u PYTHONPATH $FCPY -I -B - $HELPER <<'PY' 2>/dev/null
import sys, importlib.util
spec = importlib.util.spec_from_file_location("k3h", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
w = m.EVROOT / "ep20"
print("       " + " ".join(m.eval_command(20, 3, w)))
print("       then (CPU): " + " ".join(m.score_command(20, w)))
PY
  log "verdict now (--no-write):"; hlp verdict --no-write | sed 's/^/[dry]   /'
  log "dry run done: nothing was started or written"
  exit 0
fi

# ---- adopt: report running trainers (state finds them by output_dir)
hlp state | while read -r line; do log "start state: $line"; done
[ "$MODE" = adopt ] && log "adopt mode: no launch is ever made; FRESH episodes stay unlaunched"

# ---- main loop
last_beat=0
while :; do
  if [ -f $STOPF ]; then log "STOP file: driver exits; trainings and evaluations keep running (restart: rm STOP, then 'adopt')"
    rm -f $K/driver.pid; exit 0; fi
  mapfile -t ST < <(hlp state)
  [ ${#ST[@]} = 3 ] || { note state "state helper returned ${#ST[@]} lines (expected 3): retrying"; sleep $POLL; continue; }
  note state ""
  settled=0; newterm=0
  for line in "${ST[@]}"; do
    read -r _ ep stt rest <<< "$line"
    note st$ep "ep$ep $stt $rest" "$stt"
    case $stt in
      FRESH)
        if [ "$MODE" = run ] && ! grep -q "^$ep"$'\t' $PIDS 2>/dev/null; then try_launch $ep
        else settled=$((settled + 1)); fi;;   # adopt mode: never launched by this driver
      RUNNING|STARTING|POST_RUNNING|POST_RETRY_WAIT) ;;
      DEAD) hlp crash --ep $ep --why "$rest" > /dev/null
            log "ep$ep CRASHED: $rest -> marker $K/CRASHED_ep$ep; no automatic relaunch"; newterm=1;;
      CRASHED) settled=$((settled + 1)); [ -z "${SEEN[$ep]:-}" ] && { SEEN[$ep]=crashed; newterm=1; };;
      TRAINED) launch_post $ep "$rest";;
      POST_DONE) settled=$((settled + 1)); [ -z "${SEEN[$ep]:-}" ] && { SEEN[$ep]=post; newterm=1; };;
      *) note unk$ep "ep$ep: unknown state '$stt'";;
    esac
  done
  if [ $newterm = 1 ]; then log "verdict update:"; hlp verdict >> $DLOG 2>&1 || log "verdict refused or failed (exit $?)"; fi
  now=$(date +%s)
  if [ $((now - last_beat)) -ge $HEARTBEAT_S ]; then last_beat=$now
    echo "[$(date '+%F %T')] HEARTBEAT driver $$ | $(hlp heartbeat)" >> $HB; fi
  [ $settled -ge 3 ] && break
  sleep $POLL
done
log "every episode is settled: final verdict"
hlp verdict >> $DLOG 2>&1 || log "verdict refused or failed (exit $?)"
echo "[$(date '+%F %T')] DONE driver $$ | $(hlp heartbeat)" >> $HB
rm -f $K/driver.pid
exit 0
