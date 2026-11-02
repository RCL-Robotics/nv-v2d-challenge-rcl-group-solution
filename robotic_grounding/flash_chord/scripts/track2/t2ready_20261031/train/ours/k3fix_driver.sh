#!/bin/bash
# k3fix_driver.sh: the STAGED K3fix pilot (arm C3r2_tier2_k3fix). Pre-registration tmp_analysis/tier2/k3fix_pilot/PREREG_v2.md
# (sha256 in PREREG_v2.sha256; it supersedes PREREG.md, sha256 390d7996, which is kept unchanged). User 2026-10-21: "Yes, run
# Stage 1" (staged plan: easy Tier-1 clips first). Generalises scripts/k3easy_driver.sh (the easy-clip K3 test) to seven jobs
# in three stages with the GPU chosen at launch.
#   C3r2 exactly as the raw-input Tier-2 arm C3r2_tier2 (code/fc_r2 @61fa5d3, experiment g1_recon_body_coord_rsi_r2,
#   max_iterations 45000, experiment-default RSI deadline 35156 / zero_ramp 3906 / help-zero early stop 5860, scene.support
#   as the raw arm), jobs by stage in launch order (run dirs fc_runs/C3r2_tier2_k3fix/<job>):
#     stage 1  ep07_k3fix_s42  ep07 iron on data/human_motion_data_tier2_k3fix (K3fix = frozen K3 here; parquet byte-identical
#                              to data/human_motion_data_tier2_k3), seed 42
#              ep18_k3fix_s42  ep18 blue_trash_can on the K3fix root (= K3, byte-identical), seed 42, scene.support=false as raw
#              ep03_k3fix_s42  ep03 contigo_coffee_mug on the K3fix root (K3fix-changed), seed 42
#              ep03_raw_s7     ep03 on the RAW Tier-2 input data/human_motion_data_tier2, reset.seed=7 (seed-variance control)
#     stage 2  ep17_k3fix_s42  ep09_k3fix_s42   on the K3fix root, seed 42          (later: only by 'run --stage 2')
#     stage 3  ep04_k3fix_s42  on the K3fix root, seed 42, scene.support=false      (later: only by 'run --stage 3'; report only)
#   A K3fix job differs from the raw run only in the data root (task.parquet, HUMAN_MOTION_DATA_DIR); the control only in
#   the seed. output_dir, hydra.run.dir and logging (offline + metrics.jsonl, experiment name) do not change training
#   (k3fix_pilot/cfgcheck_v2.json composes every config). NOT the stretched curriculum. The stopped K3 queue
#   (fc_runs/queue_tier2v2, STOP + PAUSE) and the running easy-clip test (k3easy) are not touched.
# Modes:
#   run --stage S  preflight (scripts/k3fix_helper.py check; PREREG_v2 sha256 included) and stage guard (an earlier stage that
#          is not in S must have been launched), then launch every FRESH job of stage(s) S (S = 1, 2, 3 or a comma list),
#          one at a time, spaced >= 180 s, on the GPU the helper's plan picks NOW: placement tiers GPUs 5/7 first, then 3/6
#          (NUMA 1); GPUs 0-2 (NUMA 0) are left to the Tier-1 fork arms; never GPU 4; no MPS; < 3 trainings of anybody before
#          the start, >= 16.5 GB + 20 GB free plus the ramp debt of this pilot's young trainers, >= 2 free physical cores on
#          the GPU's own NUMA node (never borrowed); fewest trainings, then most free memory. CPU slice = free own-node
#          physical cores (+HT siblings) no other process holds (eval lanes excluded; track2_queue_tier2v2_helper.cpu_slice),
#          2 physical cores per job (the Tier-1 fork launcher's slot size, equal share). Under the shared GPU-choice
#          locks (opened read-only, held 180 s after a launch). A start is never made over existing model_* / rsi_* files,
#          a final.json, a CRASHED marker or an earlier launch of this pilot. FRESH jobs of other stages are DEFERRED (never
#          launched); jobs launched earlier (any stage) are watched as usual.
#   adopt  as run, but never launches (every FRESH job is DEFERRED): for a driver restart while the trainers run (rm STOP first).
#   dry --stage S  prints the checks, the config diffs, the GPU view, the job states, the stage guard, the GPU/CPU plan of the
#          FRESH jobs of S (later jobs planned as if the earlier ones had started), the exact commands of every job and the
#          verdict now; starts nothing, writes nothing.
# Later stages: when the running driver has exited by itself (every watched job settled) just run the next stage; otherwise
#   touch STOP, wait until k3fix_pilot/driver.pid is gone (<= 60 s, or up to 180 s after a launch), rm STOP, then
#   setsid nohup bash scripts/k3fix_driver.sh run --stage 2 >> tmp_analysis/tier2/k3fix_pilot/driver_stdout.log 2>&1 < /dev/null &
#   (the new driver adopts every running job of stage 1 as well). Launching a later stage is the user's decision (PREREG_v2).
# While waiting (every 60 s):
#   - a trainer that ended with final.json -> post-training step in the background (k3fix_helper.py post: export at
#     2e-4 + ONE official evaluation, eval_all.py --checkpoint-from-final, of final.json's checkpoint against its own
#     input on its training GPU, holding k3fix_pilot/evallock/gpu<g>.lock and, if it exists,
#     fc_runs/queue_runin/evallock/gpu<g>.lock (read-only, never created); then the LOCAL clean diagnostic, K3fix jobs
#     with --frame-references k3fix_pilot/frame_refs_k3fix, the raw control with the default raw reference, into
#     *_LOCAL_NOT_FOR_UPLOAD files); a technical eval failure is retried once after 10 min;
#   - after each finished post / crash, and every 30 min: k3fix_helper.py verdict (refuses when PREREG_v2.sha256 does not match);
#   - a trainer that died without final.json -> k3fix_pilot/CRASHED_<job> marker; NO automatic relaunch;
#   - heartbeat line every 10 min in k3fix_pilot/heartbeat.log.
#   The driver exits after the verdict once every watched job is settled (post finished, or crashed); DEFERRED jobs are not waited for.
# Files: tmp_analysis/tier2/k3fix_pilot/{driver.log, heartbeat.log, pids.tsv, driver.pid, STOP, CRASHED_<job>, post.log,
#        eval/<job>/{post.json, eval_all.log, results/, export/, *_LOCAL_NOT_FOR_UPLOAD*}, VERDICT.{json,txt},
#        fail_rates_vs_raw.csv, VERDICT_clean_LOCAL_NOT_FOR_UPLOAD.json, raw_curves/}; runs fc_runs/C3r2_tier2_k3fix/<job>,
#        training logs fc_runs/C3r2_tier2_k3fix/logs/<job>.log.
# STOP: touch tmp_analysis/tier2/k3fix_pilot/STOP -> the driver exits at its next loop; trainings and evaluations keep
#       running. Restart: rm STOP; setsid nohup bash scripts/k3fix_driver.sh adopt >> tmp_analysis/tier2/k3fix_pilot/driver_stdout.log 2>&1 < /dev/null &
# The driver never signals a process. GPU 4 is never used. No MPS.
# Launch stage 1:  setsid nohup bash scripts/k3fix_driver.sh run --stage 1 >> tmp_analysis/tier2/k3fix_pilot/driver_stdout.log 2>&1 < /dev/null &
# Dry run:         bash scripts/k3fix_driver.sh dry --stage 1
ME=/root/autodl-tmp/haoran.zhang
MODE=${1:-}; [ $# -gt 0 ] && shift
STAGES=""; HAVE_STAGE=0
while [ $# -gt 0 ]; do
  case $1 in
    --stage) STAGES=${2:-}; HAVE_STAGE=1; shift; [ $# -gt 0 ] && shift;;
    *) echo "k3fix_driver: unknown argument '$1'" >&2; exit 2;;
  esac
done
USAGE="usage: bash $0 run --stage 1|2|3[,..] | dry --stage 1|2|3[,..] | adopt"
case $MODE in
  run|dry) { [ $HAVE_STAGE = 1 ] && [[ $STAGES =~ ^[123](,[123])*$ ]]; } || { echo "$USAGE" >&2; exit 2; };;
  adopt) [ $HAVE_STAGE = 0 ] || { echo "k3fix_driver: adopt takes no --stage (it never launches)" >&2; exit 2; }; STAGES="";;
  *) echo "$USAGE" >&2; exit 2;;
esac
K=$ME/tmp_analysis/tier2/k3fix_pilot; RUNS=$ME/fc_runs/C3r2_tier2_k3fix; CODE=$ME/code/fc_r2
FCPY=$ME/conda_envs/flashchord/bin/python; HELPER=$ME/scripts/k3fix_helper.py
DLOG=$K/driver.log; HB=$K/heartbeat.log; PIDS=$K/pids.tsv; STOPF=$K/STOP
POLL=60; HEARTBEAT_S=600; VERDICT_S=1800; LAUNCH_HOLD=180; LOCK_WAIT=120
ORDER="ep07_k3fix_s42 ep18_k3fix_s42 ep03_k3fix_s42 ep03_raw_s7 ep17_k3fix_s42 ep09_k3fix_s42 ep04_k3fix_s42"
NJOBS=7
read -r -a LOCKS <<< "$ME/ch_runs/seedstudy/.gpulock $ME/ch_runs/f0share05/.gpulock $ME/ch_runs/filler/.gpulock"

# ---- environment of every training: exactly track2_queue_tier2v2.sh's / k3easy's (fc_env.sh, PYTHONPATH=<code>/src, no
#      bytecode, caches under ME, PCI order, no MPS, no W&B key); HUMAN_MOTION_DATA_DIR and the GPU are set per job
source $ME/scripts/fc_env.sh
export PYTHONPATH=$CODE/src PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=$ME/.cache/pycache XDG_CACHE_HOME=$ME/.cache
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_VISIBLE_DEVICES WANDB_API_KEY PYTHONHOME PYTHONSTARTUP

hlp(){ env -u PYTHONPATH $FCPY -I -B $HELPER "$@" 2> >(grep -v -e UnsupportedFieldAttributeWarning -e 'warnings.warn(' >&2); }
if [ "$MODE" = dry ]; then log(){ echo "[dry] $*"; }; else log(){ echo "[$(date '+%F %T')] $*" >> $DLOG; }; fi
die(){ log "ABORT: $*"; echo "k3fix_driver: $*" >&2
  [ "$MODE" != dry ] && [ "$(cat $K/driver.pid 2>/dev/null)" = "$$" ] && rm -f $K/driver.pid; exit 1; }
declare -A NOTE SEEN
note(){ local sig=${3:-$2}; [ "${NOTE[$1]:-}" = "$sig" ] && return 0; NOTE[$1]=$sig; [ -n "$2" ] && log "$2"; return 0; }

# ---- the jobs this invocation may launch (stage(s) S; none in adopt mode)
SJOBS=""
if [ -n "$STAGES" ]; then SJOBS=$(hlp stage-jobs --stages "$STAGES" | paste -sd' '); [ -n "$SJOBS" ] || die "no job in stage(s) '$STAGES'"; fi

# ---- singleton + preflight + stage guard
if [ "$MODE" != dry ]; then
  mkdir -p $K $RUNS/logs || die "cannot create $K / $RUNS/logs"
  exec 9>>$K/.driver.lock
  flock -n 9 || die "another k3fix_driver.sh holds $K/.driver.lock"
  [ -f $STOPF ] && die "STOP file $STOPF present: remove it to start"
  echo $$ > $K/driver.pid
  log "k3fix driver $MODE${STAGES:+ --stage $STAGES}: pid $$; code $CODE ($(git -C $CODE rev-parse --short=7 HEAD 2>/dev/null)); runs $RUNS; may launch: ${SJOBS:-nothing}"
fi
CHECK_OUT=$(hlp check); CHECK_RC=$?
if [ "$MODE" = dry ]; then
  echo "$CHECK_OUT" | sed 's/^/[dry] /'
else
  echo "$CHECK_OUT" | grep -v ' ok ' >> $DLOG; echo "$CHECK_OUT" | tail -1 | sed "s/^/[$(date '+%F %T')] /" >> $DLOG
  [ $CHECK_RC = 0 ] || die "preflight failed (see above): nothing launched"
fi
GUARD=""; GUARD_RC=0
if [ -n "$STAGES" ]; then GUARD=$(hlp stage-guard --stages "$STAGES"); GUARD_RC=$?; fi
if [ "$MODE" = run ]; then
  log "stage guard: $GUARD"
  [ $GUARD_RC = 0 ] || die "stage guard refused: nothing launched"
  case ",$STAGES," in *,1,*) ;; *) log "earlier-stage verdict at this launch:"; hlp verdict --no-write 2>&1 | sed -n '1,6p' >> $DLOG;; esac
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
print_cmd(){ # job gpu cpus nth data
  printf '( cd %q && exec setsid nohup env K3FIX=1 K3FIX_JOB=%s HUMAN_MOTION_DATA_DIR=%s CUDA_VISIBLE_DEVICES=%s WANDB_MODE=offline OMP_NUM_THREADS=%s MKL_NUM_THREADS=%s OPENBLAS_NUM_THREADS=%s taskset -c %s' \
    "$CODE" $1 "$5" $2 $4 $4 $4 $3
  printf ' %q' "${TRAIN[@]}"; echo " ) >> $RUNS/logs/$1.log 2>&1 < /dev/null &"; }
launch(){ # job gpu cpus nth
  local job=$1 g=$2 cpus=$3 nth=$4 out=$RUNS/$1 rp tp data
  case " $SJOBS " in *" $job "*) ;; *) log "$job NOT launched: not a job of stage(s) '$STAGES'"; return 1;; esac
  train_cmd $job || { log "$job NOT launched: no training command from the helper"; return 1; }
  data=$(hlp data --job $job)
  case $data in $ME/data/human_motion_data_tier2_k3fix|$ME/data/human_motion_data_tier2) ;; *) log "$job NOT launched: data root '$data'"; return 1;; esac
  if compgen -G "$out/model_*.safetensors" > /dev/null || compgen -G "$out/rsi_*.safetensors" > /dev/null || [ -e $out/final.json ]; then
    log "$job NOT launched: $out already has checkpoints or final.json (never a fresh start over them)"; return 1; fi
  [ -e $K/CRASHED_$job ] && { log "$job NOT launched: CRASHED marker present"; return 1; }
  case " 3 5 6 7 " in *" $g "*) ;; *) log "$job NOT launched: GPU '$g' not allowed (placement GPUs 5 7 3 6; never 0-2, never 4)"; return 1;; esac
  log "$job launch on gpu$g cpus $cpus OMP $nth: $(print_cmd $job $g $cpus $nth $data)"
  ( cd $CODE && exec setsid nohup env K3FIX=1 K3FIX_JOB=$job HUMAN_MOTION_DATA_DIR=$data CUDA_VISIBLE_DEVICES=$g \
      WANDB_MODE=offline OMP_NUM_THREADS=$nth MKL_NUM_THREADS=$nth OPENBLAS_NUM_THREADS=$nth \
      taskset -c $cpus "${TRAIN[@]}" ) >> $RUNS/logs/$job.log 2>&1 < /dev/null 6<&- 7<&- 8<&- 9>&- &
  rp=$!
  printf '%s\t%s\t%s\t%s\t%s\t%s\tlaunched\n' $job $rp $g $cpus $nth $(date +%s) >> $PIDS
  sleep 10; tp=$(hlp trainer --job $job)
  log "$job launched: pid $rp; trainer pid(s) now: ${tp:-not visible yet}; log $RUNS/logs/$job.log"
  return 0; }
EXCL=""
try_launch(){ # job: plan -> locks -> plan again -> launch -> hold the locks LAUNCH_HOLD s
  local job=$1 line tag g cpus nth cnote rest
  line=$(hlp plan --job $job --exclude "$EXCL" --stages "$STAGES")
  case $line in OK*) ;; *) note plan$job "$job waits: ${line#WAIT }" "$(echo "$line" | cut -c1-60)"; return 1;; esac
  if ! take_locks; then note locks "GPU-choice locks busy for > ${LOCK_WAIT}s: retrying"; return 1; fi
  note locks ""
  line=$(hlp plan --job $job --exclude "$EXCL" --stages "$STAGES")   # re-check under the locks
  case $line in OK*) ;; *) release_locks; note plan$job "$job waits: ${line#WAIT }" "$(echo "$line" | cut -c1-60)"; return 1;; esac
  note plan$job ""
  read -r tag g cpus nth cnote rest <<< "$line"
  log "$job slot decision: $line"
  if launch $job $g $cpus $nth; then EXCL="$EXCL,$cpus"; sleep $LAUNCH_HOLD; fi   # hold: the new job's memory shows up
  release_locks; return 0; }
launch_post(){ # job detail
  local job=$1
  mkdir -p $K/eval/$job
  ( exec setsid nohup env -u PYTHONPATH $FCPY -I -B $HELPER post --job $job ) >> $K/eval/$job/post_stdout.log 2>&1 < /dev/null 6<&- 7<&- 8<&- 9>&- &
  log "$job training finished ($2): post-training step launched (pid $!): export 2e-4 -> official eval vs own input -> local clean diagnostic"
  sleep 5; }

# ---- dry run: everything the run would do now, nothing started or written
if [ "$MODE" = dry ]; then
  log "preflight exit $CHECK_RC ($([ $CHECK_RC = 0 ] && echo 'run would pass the preflight' || echo 'run would ABORT'))"
  log "stage(s) requested: $STAGES -> may launch: $SJOBS"
  log "stage guard: $GUARD ($([ $GUARD_RC = 0 ] && echo 'run would launch' || echo 'run would ABORT'))"
  log "config check (hydra compose, CPU; --no-write):"
  hlp cfgcheck --no-write | sed 's/^/[dry]   /'
  log "GPU view now:"; hlp gpus | sed 's/^/[dry]   /'
  log "job states now (stage(s) $STAGES requested; FRESH jobs of other stages are DEFERRED):"
  declare -A JST
  while read -r _ job stt rest; do JST[$job]=$stt; echo "[dry]   JOB $job $stt $rest"; done < <(hlp state --stages "$STAGES")
  log "STOP file: $([ -f $STOPF ] && echo present || echo absent); queue_tier2v2 STOP/PAUSE (untouched): $(ls $ME/fc_runs/queue_tier2v2/STOP $ME/fc_runs/queue_tier2v2/PAUSE 2>/dev/null | xargs -n1 basename | paste -sd' ')"
  log "GPU-choice lock files (opened read-only by run, never created): $(for f in "${LOCKS[@]}"; do printf '%s=%s ' "${f#$ME/}" "$([ -f "$f" ] && echo present || echo 'missing(skipped)')"; done)"
  ex=""; rem=0; assume=""
  for job in $SJOBS; do [ "${JST[$job]:-}" = FRESH ] && rem=$((rem + 1)); done
  declare -A ASSUMED
  for job in $ORDER; do
    case " $SJOBS " in
      *" $job "*)
        if [ "${JST[$job]:-}" = FRESH ]; then
          line=$(hlp plan --job $job --exclude "$ex" --remaining $rem --assume "$assume" --stages "$STAGES"); rem=$((rem - 1))
          log "$job -> $line"
          read -r tag g cpus nth cnote rest <<< "$line"
          if [ "$tag" != OK ]; then g="<gpu>"; cpus="<cpus>"; nth="<n>"; else ex="$ex,$cpus"; ASSUMED[$g]=$(( ${ASSUMED[$g]:-0} + 1 ))
            assume=$(for k in "${!ASSUMED[@]}"; do printf '%s:%s,' $k ${ASSUMED[$k]}; done); fi
        else
          log "$job: state ${JST[$job]:-?}: not launched by run (watched only)"; g="<gpu>"; cpus="<cpus>"; nth="<n>"
        fi;;
      *) log "$job: ${JST[$job]:-?} (another stage): not planned now; command shown with placeholders"; g="<gpu>"; cpus="<cpus>"; nth="<n>";;
    esac
    train_cmd $job || { log "$job: no command"; continue; }
    echo "---- $job exact launch command (cwd $CODE; log $RUNS/logs/$job.log):"
    print_cmd $job $g $cpus $nth "$(hlp data --job $job)"
  done
  log "NOTE: in run mode each job is planned when it is launched (>= 180 s after the previous one), on the GPU state then"
  echo "     env (all trainings): PATH=${PATH%%:*}:... PYTHONPATH=$PYTHONPATH PYTHONDONTWRITEBYTECODE=$PYTHONDONTWRITEBYTECODE PYTHONPYCACHEPREFIX=$PYTHONPYCACHEPREFIX XDG_CACHE_HOME=$XDG_CACHE_HOME CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE WANDB_DIR=$WANDB_DIR CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-<unset>} WANDB_API_KEY=${WANDB_API_KEY:+<set>}${WANDB_API_KEY:-<unset>} PYTHONUNBUFFERED=$PYTHONUNBUFFERED"
  for job in $SJOBS; do
    echo "---- $job after final.json: ( exec setsid nohup env -u PYTHONPATH $FCPY -I -B $HELPER post --job $job ) >> $K/eval/$job/post_stdout.log 2>&1 < /dev/null &"
  done
  echo "     post commands (training GPU from pids.tsv; shown here for gpu 5):"
  env -u PYTHONPATH $FCPY -I -B - $HELPER $SJOBS <<'PY' 2>/dev/null
import sys, importlib.util
spec = importlib.util.spec_from_file_location("k3fh", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for job in sys.argv[2:]:
    w = m.EVROOT / job
    print(f"       {job}: " + " ".join(m.eval_command(job, 5, w)))
    print(f"       {job} then (CPU): " + " ".join(m.score_command(job, w)))
PY
  log "verdict now (--no-write):"; hlp verdict --no-write | sed 's/^/[dry]   /'
  log "dry run done: nothing was started or written"
  exit 0
fi

# ---- adopt / run: report the start state (state finds running trainers by output_dir)
hlp state --stages "$STAGES" | while read -r line; do log "start state: $line"; done
[ "$MODE" = adopt ] && log "adopt mode: no launch is ever made; FRESH jobs stay unlaunched (DEFERRED)"

# ---- main loop
last_beat=0; last_verdict=$(date +%s)
while :; do
  if [ -f $STOPF ]; then log "STOP file: driver exits; trainings and evaluations keep running (restart: rm STOP, then 'adopt' or 'run --stage N')"
    rm -f $K/driver.pid; exit 0; fi
  mapfile -t ST < <(hlp state --stages "$STAGES")
  [ ${#ST[@]} = $NJOBS ] || { note state "state helper returned ${#ST[@]} lines (expected $NJOBS): retrying"; sleep $POLL; continue; }
  note state ""
  settled=0; newterm=0
  for line in "${ST[@]}"; do
    read -r _ job stt rest <<< "$line"
    note st$job "$job $stt $rest" "$stt"
    case $stt in
      FRESH)
        if [ "$MODE" = run ] && ! grep -q "^$job"$'\t' $PIDS 2>/dev/null; then try_launch $job
        else settled=$((settled + 1)); fi;;   # adopt mode: never launched by this driver
      DEFERRED) settled=$((settled + 1));;    # another stage: never launched by this driver, not waited for
      RUNNING|STARTING|POST_RUNNING|POST_RETRY_WAIT) ;;
      DEAD) hlp crash --job $job --why "$rest" > /dev/null
            log "$job CRASHED: $rest -> marker $K/CRASHED_$job; no automatic relaunch"; newterm=1;;
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
    [ $newterm = 1 ] && cat $K/verdict_last.log >> $DLOG; fi
  if [ $((now - last_beat)) -ge $HEARTBEAT_S ]; then last_beat=$now
    echo "[$(date '+%F %T')] HEARTBEAT driver $$ | $(hlp heartbeat)" >> $HB; fi
  [ $settled -ge $NJOBS ] && break
  sleep $POLL
done
log "every watched job is settled: verdict"
hlp verdict >> $DLOG 2>&1 || log "verdict refused or failed (exit $?)"
echo "[$(date '+%F %T')] DONE driver $$ | $(hlp heartbeat)" >> $HB
rm -f $K/driver.pid
exit 0
