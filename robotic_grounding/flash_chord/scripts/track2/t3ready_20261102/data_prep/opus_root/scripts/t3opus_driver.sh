#!/bin/bash
# t3opus_driver.sh: the Tier-3 opus easy-clip trainability test (arm C3r2_tier3_opus). Pre-registration
# tmp_analysis/tier3/opus_easy/PREREG.md (sha256 in PREREG.sha256). User 2026-10-22: "try some easier tasks for tier3 first
# on han's data .../tier3_opus_20261021" and "tier3 first". Generalises scripts/k3fix_driver.sh (staged K3fix pilot) with a
# RETARGET stage in front:
#   data   data/human_motion_data_tier3_opus (byte copy of Han Zhou's opus release, READY; frame_references_opus = the opus
#          track in the released frame; floor-only support files; built by scripts/t3opus_build_data.py)
#   wave 1 ep08 skinny_wood_chair, ep23 toy_stroller, ep22 step_ladder        (run --wave 1)
#   wave 2 ep00 tall_bar_stool, ep18 blue_trash_can, ep10 white_desk         (only by 'run --wave 2': the user's decision)
#   Per clip, in order:
#   1. RETARGET (GPU, brief, one at a time): scripts/t3opus_retarget.sh in a clean environment = soma_to_g1.py <folder>
#      --save --fps 30 --motion-root <root> (nv repo @f2923d61, v2d env, plain CUDA context), on the GPU helper 'rplan'
#      picks (never 4, no MPS, >= 10 GB + 20 GB free, 2 own-NUMA cores), then scripts/t3opus_rcheck.py (v2dkit, CPU):
#      parquet valid, schema = the agent_T1 example, 803-style frame counts, finite, one rigid map of the input, packer
#      accepts frame_references_opus and rejects Han's release refs (s1) and the Tier-2 input, inputs unchanged,
#      floor-only. Pass -> parquet sha256 into opus_easy/retarget/parquet_lock.json. Fail -> RETARGET_FAILED_epNN marker,
#      the clip is never trained (no automatic substitute).
#   2. TRAIN: C3r2 exactly as C3r2_wide / C3r2_tier2 (code/fc_r2 @61fa5d3, g1_recon_body_coord_rsi_r2, seed 42, 45000,
#      deadline 35156 / zero_ramp 3906 / help-zero tail 5860, scene.support=false), only the data root differs. Launched
#      when the fill rule allows (helper 'plan': never GPU 4, no MPS, < 3 trainings of anybody before the start,
#      >= 16.5 GB + 20 GB free + ramp debt, 2 free physical cores of the GPU's own NUMA node, NUMA-1 GPUs first), spaced
#      >= 180 s, under the shared GPU-choice locks (opened read-only). Never over model_*/rsi_*/final.json, a CRASHED
#      marker, or an earlier launch of this test; only with a passing retarget lock (helper 'launch-ok').
#   3. POST (after final.json, background): helper 'post' = export at 1e-4 (redone at 2e-4 only if refused, recorded: the
#      Tier-3 pilot's rule) + ONE official evaluation of final.json's checkpoint against its own input (world count 1) on
#      its training GPU, holding opus_easy/evallock/gpu<g>.lock and fc_runs/queue_runin/evallock/gpu<g>.lock if it
#      exists (read-only); then the LOCAL clean diagnostic (kit packer, track_2_tier3, --frame-references
#      frame_references_opus) into *_LOCAL_NOT_FOR_UPLOAD files. One retry after 10 min for a technical failure.
#   4. VERDICT: helper 'verdict' (PREREG rule; refuses if PREREG.sha256 does not match) after each settled clip and every 30 min.
#   A trainer that dies without final.json -> CRASHED_epNN marker, NO automatic relaunch.
# Modes:
#   run --wave W   preflight (helper check), wave guard (wave 2 needs wave 1 launched), then the loop above for the clips of
#                  W; clips of other waves are DEFERRED (never retargeted or launched here).
#   adopt          never retargets or launches; watches running trainings / posts (driver restart: rm STOP first).
#   dry --wave W   checks, cfgcheck, GPU view, clip states, retarget plan + exact retarget/rcheck commands, training plans
#                  (later launches planned as if the earlier had started) + exact training and post commands, verdict now;
#                  self-tests of the retarget check (agent_T1 ep14) and of the LOCAL scoring path (pilot probe ep14).
#                  Starts nothing; writes only under opus_easy/dryrun/.
# STOP: touch tmp_analysis/tier3/opus_easy/STOP -> the driver exits at its next loop (a running retarget is finished first);
#       trainings and evaluations keep running. Restart: rm STOP; setsid nohup bash scripts/t3opus_driver.sh adopt
#       >> tmp_analysis/tier3/opus_easy/driver_stdout.log 2>&1 < /dev/null &   (or 'run --wave W' to continue launching)
# The driver never signals a process. GPU 4 is never used. No MPS.
# Launch wave 1:  setsid nohup bash scripts/t3opus_driver.sh run --wave 1 >> tmp_analysis/tier3/opus_easy/driver_stdout.log 2>&1 < /dev/null &
# Dry run:        bash scripts/t3opus_driver.sh dry --wave 1
ME=/root/autodl-tmp/haoran.zhang
MODE=${1:-}; [ $# -gt 0 ] && shift
WAVES=""; HAVE_WAVE=0
while [ $# -gt 0 ]; do
  case $1 in
    --wave) WAVES=${2:-}; HAVE_WAVE=1; shift; [ $# -gt 0 ] && shift;;
    *) echo "t3opus_driver: unknown argument '$1'" >&2; exit 2;;
  esac
done
USAGE="usage: bash $0 run --wave 1|2[,..] | dry --wave 1|2[,..] | adopt"
case $MODE in
  run|dry) { [ $HAVE_WAVE = 1 ] && [[ $WAVES =~ ^[12](,[12])*$ ]]; } || { echo "$USAGE" >&2; exit 2; };;
  adopt) [ $HAVE_WAVE = 0 ] || { echo "t3opus_driver: adopt takes no --wave (it never launches)" >&2; exit 2; }; WAVES="";;
  *) echo "$USAGE" >&2; exit 2;;
esac
A=$ME/tmp_analysis/tier3/opus_easy; RUNS=$ME/fc_runs/C3r2_tier3_opus; CODE=$ME/code/fc_r2
DATA=$ME/data/human_motion_data_tier3_opus; RDIR=$A/retarget
FCPY=$ME/conda_envs/flashchord/bin/python; KPY=$ME/conda_envs/v2dkit/bin/python; HELPER=$ME/scripts/t3opus_helper.py
RCHECK=$ME/scripts/t3opus_rcheck.py; RETARGET=$ME/scripts/t3opus_retarget.sh
DLOG=$A/driver.log; HB=$A/heartbeat.log; PIDS=$A/pids.tsv; STOPF=$A/STOP
POLL=60; HEARTBEAT_S=600; VERDICT_S=1800; LAUNCH_HOLD=180; LOCK_WAIT=120; RETARGET_HOLD=30
NCLIPS=6
CLEANPATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
read -r -a LOCKS <<< "$ME/ch_runs/seedstudy/.gpulock $ME/ch_runs/f0share05/.gpulock $ME/ch_runs/filler/.gpulock"

# ---- training environment: exactly k3fix_driver.sh's (fc_env.sh, PYTHONPATH=<code>/src, no bytecode, caches under ME,
#      PCI order, no MPS, no W&B key); HUMAN_MOTION_DATA_DIR and the GPU are set per job
source $ME/scripts/fc_env.sh
export PYTHONPATH=$CODE/src PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=$ME/.cache/pycache XDG_CACHE_HOME=$ME/.cache
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_VISIBLE_DEVICES WANDB_API_KEY PYTHONHOME PYTHONSTARTUP

hlp(){ env -u PYTHONPATH $FCPY -I -B $HELPER "$@" 2> >(grep -v -e UnsupportedFieldAttributeWarning -e 'warnings.warn(' >&2); }
if [ "$MODE" = dry ]; then log(){ echo "[dry] $*"; }; else log(){ echo "[$(date '+%F %T')] $*" >> $DLOG; }; fi
die(){ log "ABORT: $*"; echo "t3opus_driver: $*" >&2
  [ "$MODE" != dry ] && [ "$(cat $A/driver.pid 2>/dev/null)" = "$$" ] && rm -f $A/driver.pid; exit 1; }
declare -A NOTE SEEN
note(){ local sig=${3:-$2}; [ "${NOTE[$1]:-}" = "$sig" ] && return 0; NOTE[$1]=$sig; [ -n "$2" ] && log "$2"; return 0; }

WCLIPS=""
if [ -n "$WAVES" ]; then WCLIPS=$(hlp wave-clips --waves "$WAVES" | paste -sd' '); [ -n "$WCLIPS" ] || die "no clip in wave(s) '$WAVES'"; fi

# ---- singleton + preflight + wave guard
if [ "$MODE" != dry ]; then
  mkdir -p $A $RDIR $RUNS/logs || die "cannot create $A / $RUNS/logs"
  exec 9>>$A/.driver.lock
  flock -n 9 || die "another t3opus_driver.sh holds $A/.driver.lock"
  [ -f $STOPF ] && die "STOP file $STOPF present: remove it to start"
  echo $$ > $A/driver.pid
  log "t3opus driver $MODE${WAVES:+ --wave $WAVES}: pid $$; code $CODE ($(git -C $CODE rev-parse --short=7 HEAD 2>/dev/null)); data $DATA; runs $RUNS; clips: ${WCLIPS:-none (adopt)}"
fi
CHECK_OUT=$(hlp check); CHECK_RC=$?
if [ "$MODE" = dry ]; then
  echo "$CHECK_OUT" | sed 's/^/[dry] /'
else
  echo "$CHECK_OUT" | grep -v ' ok ' >> $DLOG; echo "$CHECK_OUT" | tail -1 | sed "s/^/[$(date '+%F %T')] /" >> $DLOG
  [ $CHECK_RC = 0 ] || die "preflight failed (see above): nothing retargeted or launched"
fi
GUARD="ok"; GUARD_RC=0
case ",$WAVES," in
  *,2,*) case ",$WAVES," in *,1,*) ;; *)
     fresh1=$(hlp state --waves 1 | awk '$3 ~ /^(NEEDS_RETARGET|NEEDS_RCHECK|FRESH)$/ && ($2=="08"||$2=="23"||$2=="22")' | wc -l)
     [ "$fresh1" = 0 ] && GUARD="wave 1 launched or settled" || { GUARD="wave 1 has $fresh1 clip(s) not yet launched: run --wave 1 first"; GUARD_RC=1; };;
   esac;;
esac
if [ "$MODE" = run ]; then log "wave guard: $GUARD"; [ $GUARD_RC = 0 ] || die "wave guard refused"; fi

# ---- GPU-choice locks shared with the other queues, opened READ-ONLY (a missing file is skipped, never created)
release_locks(){ local fd; for fd in 6 7 8; do { flock -u $fd; eval "exec $fd<&-"; } 2>/dev/null; done; return 0; }
take_locks(){
  if [ -f "${LOCKS[0]}" ]; then exec 8<"${LOCKS[0]}" || return 1; flock -w $LOCK_WAIT 8 || { release_locks; return 1; }; fi
  if [ -f "${LOCKS[1]}" ]; then exec 7<"${LOCKS[1]}" || { release_locks; return 1; }; flock -w $LOCK_WAIT 7 || { release_locks; return 1; }; fi
  if [ -f "${LOCKS[2]}" ]; then exec 6<"${LOCKS[2]}" || { release_locks; return 1; }; flock -w $LOCK_WAIT 6 || { release_locks; return 1; }; fi
  return 0; }

# ---- retarget + check of one clip (synchronous; the retarget runs in a clean env, plain GPU)
retarget_cmd(){ # ep seq uuid cpus nth
  printf 'env -i HOME=/root USER=root LOGNAME=root SHELL=/bin/bash TERM=dumb PATH=%s bash --noprofile --norc %s %s %s %s %s %s %s %s' \
    $CLEANPATH $RETARGET $1 $2 $3 $4 $5 $RDIR/ep$1_retarget.log $RDIR/ep$1_retarget.status; }
rcheck_cmd(){ printf 'env -u PYTHONPATH %s -I -B %s --ep %s --log %s' $KPY $RCHECK $1 $RDIR/ep$1_retarget.log; }
run_rcheck(){ # ep
  local ep=$1 rc
  ( cd /tmp && exec env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= nice -n 10 $KPY -I -B $RCHECK --ep $ep --log $RDIR/ep${ep}_retarget.log ) \
      > $RDIR/ep${ep}_rcheck.out 2>&1 9>&- 6<&- 7<&- 8<&-
  rc=$?
  grep -E 'FAIL|all_ok' $RDIR/ep${ep}_rcheck.out | sed 's/^/    /' >> $DLOG
  return $rc; }
do_retarget(){ # ep
  local ep=$1 seq line tag g uuid cpus nth rest rp rc
  [ -n "$(hlp trainer --ep $ep)" ] && return 1
  seq=$(hlp seq --ep $ep)
  line=$(hlp rplan --exclude "$EXCL")
  case $line in OK*) ;; *) note rplan$ep "ep$ep retarget waits: ${line#WAIT }" "$(echo "$line" | cut -c1-60)"; return 1;; esac
  if ! take_locks; then note locks "GPU-choice locks busy for > ${LOCK_WAIT}s: retrying"; return 1; fi
  line=$(hlp rplan --exclude "$EXCL")
  case $line in OK*) ;; *) release_locks; note rplan$ep "ep$ep retarget waits: ${line#WAIT }" "$(echo "$line" | cut -c1-60)"; return 1;; esac
  note rplan$ep ""
  read -r tag g uuid cpus nth rest <<< "$line"
  [ "$g" = 4 ] && { release_locks; log "ep$ep retarget NOT started: GPU 4"; return 1; }
  exec 5>>$RDIR/ep$ep.lock
  flock -n 5 || { release_locks; exec 5>&-; log "ep$ep retarget lock busy"; return 1; }
  log "ep$ep retarget on gpu$g: $line"
  log "ep$ep retarget command: $(retarget_cmd $ep $seq $uuid $cpus $nth)"
  ( exec env -i HOME=/root USER=root LOGNAME=root SHELL=/bin/bash TERM=dumb PATH=$CLEANPATH \
      bash --noprofile --norc $RETARGET $ep $seq $uuid $cpus $nth $RDIR/ep${ep}_retarget.log $RDIR/ep${ep}_retarget.status ) \
      < /dev/null > $RDIR/ep${ep}_retarget.out 2>&1 5>&- 6<&- 7<&- 8<&- 9>&- &
  rp=$!
  sleep $RETARGET_HOLD; release_locks
  wait $rp; rc=$?
  log "ep$ep retarget exit $rc: $(tail -3 $RDIR/ep${ep}_retarget.status 2>/dev/null | paste -sd'|')"
  if [ $rc = 0 ]; then
    if run_rcheck $ep; then log "ep$ep retarget check PASSED: parquet locked ($(hlp launch-ok --ep $ep))"
    else hlp rfail --ep $ep --why "rcheck failed (see $RDIR/ep${ep}_rcheck.out / ep${ep}_rcheck.json)" > /dev/null
         log "ep$ep retarget check FAILED -> marker RETARGET_FAILED_ep$ep; never trained"; fi
  else
    hlp rfail --ep $ep --why "retarget exit $rc (see $RDIR/ep${ep}_retarget.status / .log)" > /dev/null
    log "ep$ep retarget FAILED (exit $rc) -> marker RETARGET_FAILED_ep$ep; never trained"
  fi
  flock -u 5; exec 5>&-
  return 0; }

# ---- the training command of one clip (argv from the helper) and its launch
TRAIN=()
train_cmd(){ mapfile -d '' -t TRAIN < <(hlp cmd --ep $1); [ ${#TRAIN[@]} -gt 10 ]; }
print_cmd(){ # ep gpu cpus nth
  printf '( cd %q && exec setsid nohup env T3OPUS=1 T3OPUS_EP=%s HUMAN_MOTION_DATA_DIR=%s CUDA_VISIBLE_DEVICES=%s WANDB_MODE=offline OMP_NUM_THREADS=%s MKL_NUM_THREADS=%s OPENBLAS_NUM_THREADS=%s taskset -c %s' \
    "$CODE" $1 "$DATA" $2 $4 $4 $4 $3
  printf ' %q' "${TRAIN[@]}"; echo " ) >> $RUNS/logs/ep$1.log 2>&1 < /dev/null &"; }
launch(){ # ep gpu cpus nth
  local ep=$1 g=$2 cpus=$3 nth=$4 out=$RUNS/ep$1 rp tp ok
  case " $WCLIPS " in *" $ep "*) ;; *) log "ep$ep NOT launched: not a clip of wave(s) '$WAVES'"; return 1;; esac
  train_cmd $ep || { log "ep$ep NOT launched: no training command from the helper"; return 1; }
  ok=$(hlp launch-ok --ep $ep) || { log "ep$ep NOT launched: retarget lock check: $ok"; return 1; }
  if compgen -G "$out/model_*.safetensors" > /dev/null || compgen -G "$out/rsi_*.safetensors" > /dev/null || [ -e $out/final.json ]; then
    log "ep$ep NOT launched: $out already has checkpoints or final.json (never a fresh start over them)"; return 1; fi
  [ -e $A/CRASHED_ep$ep ] && { log "ep$ep NOT launched: CRASHED marker present"; return 1; }
  case " 0 1 2 3 5 6 7 " in *" $g "*) ;; *) log "ep$ep NOT launched: GPU '$g' not allowed (never 4)"; return 1;; esac
  log "ep$ep launch on gpu$g cpus $cpus OMP $nth ($ok): $(print_cmd $ep $g $cpus $nth)"
  ( cd $CODE && exec setsid nohup env T3OPUS=1 T3OPUS_EP=$ep HUMAN_MOTION_DATA_DIR=$DATA CUDA_VISIBLE_DEVICES=$g \
      WANDB_MODE=offline OMP_NUM_THREADS=$nth MKL_NUM_THREADS=$nth OPENBLAS_NUM_THREADS=$nth \
      taskset -c $cpus "${TRAIN[@]}" ) >> $RUNS/logs/ep$ep.log 2>&1 < /dev/null 5>&- 6<&- 7<&- 8<&- 9>&- &
  rp=$!
  printf '%s\t%s\t%s\t%s\t%s\t%s\tlaunched\n' $ep $rp $g $cpus $nth $(date +%s) >> $PIDS
  sleep 10; tp=$(hlp trainer --ep $ep)
  log "ep$ep launched: pid $rp; trainer pid(s) now: ${tp:-not visible yet}; log $RUNS/logs/ep$ep.log"
  return 0; }
EXCL=""
try_launch(){ # ep: plan -> locks -> plan again -> launch -> hold the locks LAUNCH_HOLD s
  local ep=$1 line tag g cpus nth cnote rest
  line=$(hlp plan --ep $ep --exclude "$EXCL")
  case $line in OK*) ;; *) note plan$ep "ep$ep waits: ${line#WAIT }" "$(echo "$line" | cut -c1-60)"; return 1;; esac
  if ! take_locks; then note locks "GPU-choice locks busy for > ${LOCK_WAIT}s: retrying"; return 1; fi
  note locks ""
  line=$(hlp plan --ep $ep --exclude "$EXCL")
  case $line in OK*) ;; *) release_locks; note plan$ep "ep$ep waits: ${line#WAIT }" "$(echo "$line" | cut -c1-60)"; return 1;; esac
  note plan$ep ""
  read -r tag g cpus nth cnote rest <<< "$line"
  log "ep$ep slot decision: $line"
  if launch $ep $g $cpus $nth; then EXCL="$EXCL,$cpus"; sleep $LAUNCH_HOLD; fi
  release_locks; return 0; }
launch_post(){ # ep detail
  local ep=$1
  mkdir -p $A/eval/ep$ep
  ( exec setsid nohup env -u PYTHONPATH $FCPY -I -B $HELPER post --ep $ep ) >> $A/eval/ep$ep/post_stdout.log 2>&1 < /dev/null 5>&- 6<&- 7<&- 8<&- 9>&- &
  log "ep$ep training finished ($2): post-training step launched (pid $!): export 1e-4 (2e-4 if refused) -> official eval vs own input -> LOCAL clean diagnostic"
  sleep 5; }

# ======================================================================== dry run
if [ "$MODE" = dry ]; then
  DR=$A/dryrun; mkdir -p $DR
  log "preflight exit $CHECK_RC ($([ $CHECK_RC = 0 ] && echo 'run would pass the preflight' || echo 'run would ABORT'))"
  log "wave(s) requested: $WAVES -> clips $WCLIPS; wave guard: $GUARD ($([ $GUARD_RC = 0 ] && echo ok || echo 'run would ABORT'))"
  log "config check (hydra compose, CPU; --no-write): $(hlp cfgcheck --no-write 2>&1 | tail -1)"
  log "GPU view now:"; hlp gpus | sed 's/^/[dry]   /'
  log "clip states now:"
  declare -A CST
  while read -r _ ep stt rest; do CST[$ep]=$stt; echo "[dry]   CLIP $ep $stt $rest"; done < <(hlp state --waves "$WAVES")
  log "STOP file: $([ -f $STOPF ] && echo present || echo absent); GPU-choice lock files (read-only, never created): $(for f in "${LOCKS[@]}"; do printf '%s=%s ' "${f#$ME/}" "$([ -f "$f" ] && echo present || echo 'missing(skipped)')"; done)"
  log "retarget-check self-test on the Tier-3 pilot's agent_T1 ep14 parquet (CPU, writes nothing):"
  ( cd /tmp && env -u PYTHONPATH CUDA_VISIBLE_DEVICES= nice -n 10 $KPY -I -B $RCHECK --selftest ) 2>&1 | sed 's/^/[dry]   /' | cut -c1-200
  log "LOCAL scoring self-test (kit packer, track_2_tier3) on the pilot's probe output (must reproduce tmp_analysis/tier3/s4b_probe_LOCAL_NOT_FOR_UPLOAD.json; no clean number printed here):"
  rm -rf $DR/score_selftest; SST=$(hlp score-selftest --work $DR/score_selftest); log "  exit $?: $SST"
  log "retarget plan now: $(hlp rplan)"
  for ep in $WCLIPS; do
    st=${CST[$ep]:-?}
    read -r tag g uuid cpus nth rest <<< "$(hlp rplan)"; [ "$tag" = OK ] || { uuid="<uuid>"; cpus="<cpus>"; nth="<n>"; }
    echo "---- ep$ep ($st) retarget (cwd /, clean env; then rcheck on CPU):"
    echo "     $(retarget_cmd $ep $(hlp seq --ep $ep) $uuid $cpus $nth)"
    echo "     $(rcheck_cmd $ep)"
  done
  ex=""; assume=""; declare -A ASSUMED
  for ep in $WCLIPS; do
    line=$(hlp plan --ep $ep --exclude "$ex" --assume "$assume")
    log "ep$ep training plan now: $line"
    read -r tag g cpus nth cnote rest <<< "$line"
    if [ "$tag" != OK ]; then g="<gpu>"; cpus="<cpus>"; nth="<n>"; else ex="$ex,$cpus"; ASSUMED[$g]=$(( ${ASSUMED[$g]:-0} + 1 ))
      assume=$(for k in "${!ASSUMED[@]}"; do printf '%s:%s,' $k ${ASSUMED[$k]}; done); fi
    train_cmd $ep || { log "ep$ep: no command"; continue; }
    echo "---- ep$ep exact launch command (cwd $CODE; log $RUNS/logs/ep$ep.log):"
    print_cmd $ep $g $cpus $nth
  done
  echo "     env (all trainings): PATH=${PATH%%:*}:... PYTHONPATH=$PYTHONPATH PYTHONDONTWRITEBYTECODE=$PYTHONDONTWRITEBYTECODE XDG_CACHE_HOME=$XDG_CACHE_HOME CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE WANDB_DIR=$WANDB_DIR CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-<unset>} WANDB_API_KEY=${WANDB_API_KEY:+<set>}${WANDB_API_KEY:-<unset>}"
  for ep in $WCLIPS; do
    echo "---- ep$ep after final.json: ( exec setsid nohup env -u PYTHONPATH $FCPY -I -B $HELPER post --ep $ep ) >> $A/eval/ep$ep/post_stdout.log 2>&1 < /dev/null &"
  done
  echo "     post commands (training GPU from pids.tsv; shown for gpu 0):"
  env -u PYTHONPATH $FCPY -I -B - $HELPER $WCLIPS <<'PY' 2>/dev/null
import sys, importlib.util
spec = importlib.util.spec_from_file_location("t3h", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for ep in sys.argv[2:]:
    w = m.EVROOT / f"ep{int(ep):02d}"
    print(f"       ep{ep}: " + " ".join(m.eval_command(int(ep), 0, w, m.EXPORT_TOLS[0])))
    print(f"       ep{ep} only if refused at 1e-4: the same with --export-tolerance {m.EXPORT_TOLS[1]!r}")
    print(f"       ep{ep} then (CPU, LOCAL): " + " ".join(m.score_command(int(ep), w)))
PY
  log "verdict now (--no-write):"; hlp verdict --no-write | sed 's/^/[dry]   /'
  log "dry run done: nothing was started; files written only under $DR"
  exit 0
fi

# ======================================================================== adopt / run
hlp state --waves "$WAVES" | while read -r line; do log "start state: $line"; done
[ "$MODE" = adopt ] && log "adopt mode: nothing is ever retargeted or launched"

last_beat=0; last_verdict=$(date +%s)
while :; do
  if [ -f $STOPF ]; then log "STOP file: driver exits; trainings and evaluations keep running (restart: rm STOP, then 'adopt' or 'run --wave W')"
    rm -f $A/driver.pid; exit 0; fi
  mapfile -t ST < <(hlp state --waves "$WAVES")
  [ ${#ST[@]} = $NCLIPS ] || { note state "state helper returned ${#ST[@]} lines (expected $NCLIPS): retrying"; sleep $POLL; continue; }
  note state ""
  settled=0; newterm=0
  for line in "${ST[@]}"; do
    read -r _ ep stt rest <<< "$line"
    note st$ep "ep$ep $stt $rest" "$stt"
    case $stt in
      NEEDS_RETARGET)
        if [ "$MODE" = run ]; then do_retarget $ep; else settled=$((settled + 1)); fi;;
      NEEDS_RCHECK)
        if [ "$MODE" = run ] && [ -z "$(pgrep -f "soma_to_g1.py .*sequence_id=.*$DATA" | head -1)" ]; then
          log "ep$ep parquet present without a passing check: running the retarget check"
          if run_rcheck $ep; then log "ep$ep retarget check PASSED"; else hlp rfail --ep $ep --why "rcheck failed" > /dev/null; log "ep$ep retarget check FAILED -> RETARGET_FAILED_ep$ep"; newterm=1; fi
        elif [ "$MODE" != run ]; then settled=$((settled + 1)); fi;;
      RETARGETING) ;;
      RETARGET_FAILED|PARQUET_CHANGED) settled=$((settled + 1)); [ -z "${SEEN[$ep]:-}" ] && { SEEN[$ep]=rfail; newterm=1; };;
      FRESH)
        if [ "$MODE" = run ] && ! grep -q "^$ep"$'\t' $PIDS 2>/dev/null; then try_launch $ep
        else settled=$((settled + 1)); fi;;
      DEFERRED) settled=$((settled + 1));;
      RUNNING|STARTING|POST_RUNNING|POST_RETRY_WAIT) ;;
      DEAD) hlp crash --ep $ep --why "$rest" > /dev/null
            log "ep$ep CRASHED: $rest -> marker $A/CRASHED_ep$ep; no automatic relaunch"; newterm=1;;
      CRASHED) settled=$((settled + 1)); [ -z "${SEEN[$ep]:-}" ] && { SEEN[$ep]=crashed; newterm=1; };;
      TRAINED) launch_post $ep "$rest";;
      POST_DONE) settled=$((settled + 1)); [ -z "${SEEN[$ep]:-}" ] && { SEEN[$ep]=post; newterm=1; };;
      *) note unk$ep "ep$ep: unknown state '$stt'";;
    esac
    [ -f $STOPF ] && break
  done
  now=$(date +%s)
  if [ $newterm = 1 ] || [ $((now - last_verdict)) -ge $VERDICT_S ]; then last_verdict=$now
    [ $newterm = 1 ] && log "verdict update:"
    hlp verdict > $A/verdict_last.log 2>&1 || log "verdict refused or failed (exit $?)"
    [ $newterm = 1 ] && cat $A/verdict_last.log >> $DLOG; fi
  if [ $((now - last_beat)) -ge $HEARTBEAT_S ]; then last_beat=$now
    echo "[$(date '+%F %T')] HEARTBEAT driver $$ | $(hlp heartbeat --waves "${WAVES:-1,2}")" >> $HB; fi
  [ $settled -ge $NCLIPS ] && break
  sleep $POLL
done
log "every clip is settled: verdict"
hlp verdict >> $DLOG 2>&1 || log "verdict refused or failed (exit $?)"
echo "[$(date '+%F %T')] DONE driver $$ | $(hlp heartbeat --waves "${WAVES:-1,2}")" >> $HB
rm -f $A/driver.pid
exit 0
