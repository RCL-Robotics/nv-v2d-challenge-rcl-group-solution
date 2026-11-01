#!/bin/bash
# t3f0_driver.sh: the Tier-3 f0 test (arm C3r2_tier3_f0): C3r2 on Han Zhou's tier3_frame0_supports_20261024 release
# (opus_v2 object track + frame-0 object-pose correction; human unchanged; Han's supports for 14 episodes), ALL 30 episodes.
# NEW FILE (2026-10-24), copied from scripts/t3opus_driver_v2.sh (left untouched: the opus_v2 driver executes it).
# User 2026-10-24: "ran new tier3 based on Han's new released dataset? frame 0 is allowed and we double checked on it. but it
# might have a potential issue that hand is not that align with the objects".
# Pre-registration (hash-locked, never edited): tmp_analysis/tier3/f0/PREREG_f0.md (+ PREREG_f0.sha256).
# Helper scripts/t3f0_helper.py. Data root data/human_motion_data_tier3_f0 (READY write-once; frame_references_f0; all 30
#   parquets retargeted by the prep step and locked in tmp_analysis/tier3/f0/retarget/parquet_lock.json, t3f0_rcheck.py
#   G1-G10): this driver has NO retarget stage; a clip without a passing lock is NOT_READY and never launched.
# ORDER (fixed, PREREG_f0): episode index 00 01 ... 29 (no Tier-1 information).
# Per clip, in order:
#   1. TRAIN: C3r2 exactly as C3r2_tier2 (code/fc_r2 @61fa5d3, g1_recon_body_coord_rsi_r2, seed 42, 45000, deadline 35156 /
#      zero_ramp 3906 / help-zero tail 5860); only the data root, output_dir and logging differ, plus
#      scene.support from the root's SCENE_SUPPORT.json (true for 01 02 03 06 07 09 11 13 16 17 20 25 27 28) and the
#      pre-registered mitigation termination.palm_position_threshold=0.25 for 01 02 03 15 17 19 20 28 (cfgcheck_f0.json).
#      Launched in ORDER when the v2 fill rule allows (helper 'plan': never GPU 4, no MPS, < 3 trainings of anybody before
#      the start, >= 16.5 GB + 20 GB free + ramp debt, 2 free physical cores of the GPU's own NUMA node, NUMA-1 GPUs first),
#      spaced >= 180 s, under the shared GPU-choice locks (opened read-only). Never over model_*/rsi_*/final.json, a CRASHED
#      marker, or an earlier launch of this test (pids.tsv); only with a passing retarget lock.
#   2. POST (after final.json, background): helper 'post' = export at 1e-4 (redone at 2e-4 only if refused, recorded) + ONE
#      official evaluation of final.json's checkpoint against its own f0 input (world count 1) on its training GPU under the
#      eval-lane locks (own + rolling-start / opus_v2 / k3* lanes, read-only); then the LOCAL clean diagnostic into
#      *_LOCAL_NOT_FOR_UPLOAD files. One retry after 10 min.
#   3. VERDICT: helper 'verdict' (refuses if PREREG_f0.md's sha256 does not match) after each settled clip and every 30 min.
#   A trainer that dies without final.json -> CRASHED_epNN marker, NO automatic relaunch.
#   Every pass also writes f0/waiting.json (un-held FRESH clips) and keeps f0/ALL_LAUNCHED present iff no clip is FRESH: the
#   Tier-2 launchers (K3CA, T2fix) do NOT recognise this driver, so keep their PAUSE files until ALL_LAUNCHED appears.
# Modes:
#   run      preflight (helper check) + refuses while a t3opus driver is alive in RUN mode (stop the opus_v2 driver v2 first;
#            'adopt' mode of v2 is fine) -> the loop above
#   adopt    never launches; watches running trainings / posts
#   dry [--ignore-pids P,..]   checks, cfgcheck, prep table, GPU view, clip states, training plans (later launches planned as
#            if the earlier had started; --ignore-pids additionally simulates those processes gone, e.g. the opus_v2
#            trainers the switch plan stops) + exact training and post commands, verdict now, waiting state; self-test of
#            the LOCAL scoring path (pilot probe ep14). Starts nothing; writes only under tmp_analysis/tier3/f0/dryrun/.
# Singleton: tmp_analysis/tier3/f0/.driver.lock.
# STOP: touch tmp_analysis/tier3/f0/STOP -> the driver exits at its next loop; trainings and evaluations keep running.
#       Restart: rm STOP; then 'adopt' or 'run'.
# HOLD: touch tmp_analysis/tier3/f0/HOLD_epNN -> clip NN is not LAUNCHED while the file exists (later clips go ahead);
#       rm it to release the clip at the next pass. No HOLD file exists by default. Changes no rule, order or recipe.
# The driver never signals a process. GPU 4 is never used. No MPS.
# Launch:  cd /root/autodl-tmp/haoran.zhang && setsid nohup bash scripts/t3f0_driver.sh run >> tmp_analysis/tier3/f0/driver_stdout.log 2>&1 < /dev/null &
# Dry run: bash scripts/t3f0_driver.sh dry [--ignore-pids P,..]
ME=/root/autodl-tmp/haoran.zhang
MODE=${1:-}; [ $# -gt 0 ] && shift
IGNORE=""
while [ $# -gt 0 ]; do
  case $1 in
    --ignore-pids) IGNORE=${2:-}; shift; [ $# -gt 0 ] && shift;;
    *) echo "t3f0_driver: unknown argument '$1'" >&2; exit 2;;
  esac
done
USAGE="usage: bash $0 run | adopt | dry [--ignore-pids P,..]"
case $MODE in
  run|adopt) [ -z "$IGNORE" ] || { echo "t3f0_driver: --ignore-pids is for 'dry' only" >&2; exit 2; };;
  dry) [ -z "$IGNORE" ] || [[ $IGNORE =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "$USAGE" >&2; exit 2; };;
  *) echo "$USAGE" >&2; exit 2;;
esac
A=$ME/tmp_analysis/tier3/f0; RUNS=$ME/fc_runs/C3r2_tier3_f0; CODE=$ME/code/fc_r2
DATA=$ME/data/human_motion_data_tier3_f0
FCPY=$ME/conda_envs/flashchord/bin/python; HELPER=$ME/scripts/t3f0_helper.py
DLOG=$A/driver.log; HB=$A/heartbeat.log; PIDS=$A/pids.tsv; STOPF=$A/STOP
held(){ [ -e $A/HOLD_ep$1 ]; }
holds(){ local h; h=$(cd $A 2>/dev/null && ls -1 2>/dev/null | grep '^HOLD_ep[0-9][0-9]$' | paste -sd' '); echo "${h:-none}"; }
POLL=60; HEARTBEAT_S=600; VERDICT_S=1800; LAUNCH_HOLD=180; LOCK_WAIT=120
NCLIPS=30
read -r -a LOCKS <<< "$ME/ch_runs/seedstudy/.gpulock $ME/ch_runs/f0share05/.gpulock $ME/ch_runs/filler/.gpulock"

# ---- training environment: exactly v2's (fc_env.sh, PYTHONPATH=<code>/src, no bytecode, caches under ME, PCI order,
#      no MPS, no W&B key); HUMAN_MOTION_DATA_DIR and the GPU are set per job
source $ME/scripts/fc_env.sh
export PYTHONPATH=$CODE/src PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=$ME/.cache/pycache XDG_CACHE_HOME=$ME/.cache
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_VISIBLE_DEVICES WANDB_API_KEY PYTHONHOME PYTHONSTARTUP

hlp(){ env -u PYTHONPATH CUDA_VISIBLE_DEVICES= $FCPY -I -B $HELPER "$@" 2> >(grep -v -e UnsupportedFieldAttributeWarning -e 'warnings.warn(' >&2); }
if [ "$MODE" = dry ]; then log(){ echo "[dry] $*"; }; else log(){ echo "[$(date '+%F %T')] [f0] $*" >> $DLOG; }; fi
die(){ log "ABORT: $*"; echo "t3f0_driver: $*" >&2
  [ "$MODE" != dry ] && [ "$(cat $A/driver.pid 2>/dev/null)" = "$$" ] && rm -f $A/driver.pid; exit 1; }
declare -A NOTE SEEN
note(){ local sig=${3:-$2}; [ "${NOTE[$1]:-}" = "$sig" ] && return 0; NOTE[$1]=$sig; [ -n "$2" ] && log "$2"; return 0; }

# ---- singleton + preflight + no t3opus driver in run mode
if [ "$MODE" != dry ]; then
  mkdir -p $A $RUNS/logs || die "cannot create $A / $RUNS/logs"
  exec 9>>$A/.driver.lock
  flock -n 9 || die "another t3f0 driver holds $A/.driver.lock"
  [ -f $STOPF ] && die "STOP file $STOPF present: remove it to start"
  echo $$ > $A/driver.pid
  log "t3f0 driver $MODE: pid $$; helper $(sha256sum $HELPER | cut -c1-12); code $CODE ($(git -C $CODE rev-parse --short=7 HEAD 2>/dev/null)); data $DATA; runs $RUNS"
fi
CHECK_OUT=$(hlp check); CHECK_RC=$?
if [ "$MODE" = dry ]; then
  echo "$CHECK_OUT" | sed 's/^/[dry] /'
else
  echo "$CHECK_OUT" | grep -v ' ok ' >> $DLOG; echo "$CHECK_OUT" | tail -1 | sed "s/^/[$(date '+%F %T')] [f0] /" >> $DLOG
  [ $CHECK_RC = 0 ] || die "preflight failed (see above): nothing launched"
fi
DRV_OUT=$(hlp t3-run-drivers); DRV_RC=$?
if [ "$MODE" = run ]; then
  log "Tier-3 drivers in run mode: $(echo "$DRV_OUT" | paste -sd';')"
  [ $DRV_RC = 0 ] || die "a t3opus driver is alive in RUN mode: stop it first (touch tmp_analysis/tier3/opus_easy/STOP and wait for its exit)"
fi

# ---- GPU-choice locks shared with the other queues, opened READ-ONLY (a missing file is skipped, never created)
release_locks(){ local fd; for fd in 6 7 8; do { flock -u $fd; eval "exec $fd<&-"; } 2>/dev/null; done; return 0; }
take_locks(){
  if [ -f "${LOCKS[0]}" ]; then exec 8<"${LOCKS[0]}" || return 1; flock -w $LOCK_WAIT 8 || { release_locks; return 1; }; fi
  if [ -f "${LOCKS[1]}" ]; then exec 7<"${LOCKS[1]}" || { release_locks; return 1; }; flock -w $LOCK_WAIT 7 || { release_locks; return 1; }; fi
  if [ -f "${LOCKS[2]}" ]; then exec 6<"${LOCKS[2]}" || { release_locks; return 1; }; flock -w $LOCK_WAIT 6 || { release_locks; return 1; }; fi
  return 0; }

# ---- the training command of one clip (argv from the helper) and its launch
TRAIN=()
train_cmd(){ mapfile -d '' -t TRAIN < <(hlp cmd --ep $1); [ ${#TRAIN[@]} -gt 10 ]; }
print_cmd(){ # ep gpu cpus nth
  printf '( cd %q && exec setsid nohup env T3F0=1 T3F0_EP=%s HUMAN_MOTION_DATA_DIR=%s CUDA_VISIBLE_DEVICES=%s WANDB_MODE=offline OMP_NUM_THREADS=%s MKL_NUM_THREADS=%s OPENBLAS_NUM_THREADS=%s taskset -c %s' \
    "$CODE" $1 "$DATA" $2 $4 $4 $4 $3
  printf ' %q' "${TRAIN[@]}"; echo " ) >> $RUNS/logs/ep$1.log 2>&1 < /dev/null &"; }
launch(){ # ep gpu cpus nth
  local ep=$1 g=$2 cpus=$3 nth=$4 out=$RUNS/ep$1 rp tp ok
  grep -q "^$ep"$'\t' $PIDS 2>/dev/null && { log "ep$ep NOT launched: already in $PIDS (never a relaunch)"; return 1; }
  train_cmd $ep || { log "ep$ep NOT launched: no training command from the helper"; return 1; }
  ok=$(hlp launch-ok --ep $ep) || { log "ep$ep NOT launched: retarget lock check: $ok"; return 1; }
  if compgen -G "$out/model_*.safetensors" > /dev/null || compgen -G "$out/rsi_*.safetensors" > /dev/null || [ -e $out/final.json ]; then
    log "ep$ep NOT launched: $out already has checkpoints or final.json (never a fresh start over them)"; return 1; fi
  [ -e $A/CRASHED_ep$ep ] && { log "ep$ep NOT launched: CRASHED marker present"; return 1; }
  held $ep && { log "ep$ep NOT launched: HOLD file $A/HOLD_ep$ep present"; return 1; }
  case " 0 1 2 3 5 6 7 " in *" $g "*) ;; *) log "ep$ep NOT launched: GPU '$g' not allowed (never 4)"; return 1;; esac
  log "ep$ep launch on gpu$g cpus $cpus OMP $nth ($ok): $(print_cmd $ep $g $cpus $nth)"
  ( cd $CODE && exec setsid nohup env T3F0=1 T3F0_EP=$ep HUMAN_MOTION_DATA_DIR=$DATA CUDA_VISIBLE_DEVICES=$g \
      WANDB_MODE=offline OMP_NUM_THREADS=$nth MKL_NUM_THREADS=$nth OPENBLAS_NUM_THREADS=$nth \
      taskset -c $cpus "${TRAIN[@]}" ) >> $RUNS/logs/ep$ep.log 2>&1 < /dev/null 5>&- 6<&- 7<&- 8<&- 9>&- &
  rp=$!
  printf '%s\t%s\t%s\t%s\t%s\t%s\tlaunched\n' $ep $rp $g $cpus $nth $(date +%s) >> $PIDS
  sleep 10; tp=$(hlp trainer --ep $ep)
  log "ep$ep launched: pid $rp; trainer pid(s) now: ${tp:-not visible yet}; log $RUNS/logs/ep$ep.log"
  return 0; }
EXCL=""
try_launch(){ # ep: plan -> locks -> plan again -> launch -> hold the locks LAUNCH_HOLD s. Returns 1 when the clip has to
              # WAIT for a GPU (the caller then skips the later FRESH clips of this pass: they would get the same answer)
  local ep=$1 line tag g cpus nth cnote rest
  line=$(hlp plan --ep $ep --exclude "$EXCL")
  case $line in OK*) ;; *) note plan "ep$ep (head of the launch queue) waits: ${line#WAIT }" "$ep $(echo "$line" | cut -c1-60)"; return 1;; esac
  if ! take_locks; then note locks "GPU-choice locks busy for > ${LOCK_WAIT}s: retrying"; return 1; fi
  note locks ""
  line=$(hlp plan --ep $ep --exclude "$EXCL")
  case $line in OK*) ;; *) release_locks; note plan "ep$ep (head of the launch queue) waits: ${line#WAIT }" "$ep $(echo "$line" | cut -c1-60)"; return 1;; esac
  note plan ""
  read -r tag g cpus nth cnote rest <<< "$line"
  log "ep$ep slot decision: $line"
  if launch $ep $g $cpus $nth; then EXCL="$EXCL,$cpus"; sleep $LAUNCH_HOLD; fi
  release_locks; return 0; }
launch_post(){ # ep detail
  local ep=$1
  mkdir -p $A/eval/ep$ep
  ( exec setsid nohup env -u PYTHONPATH CUDA_VISIBLE_DEVICES= $FCPY -I -B $HELPER post --ep $ep ) >> $A/eval/ep$ep/post_stdout.log 2>&1 < /dev/null 5>&- 6<&- 7<&- 8<&- 9>&- &
  log "ep$ep training finished ($2): post-training step launched (pid $!): export 1e-4 (2e-4 if refused) -> official eval vs own input -> LOCAL clean diagnostic"
  sleep 5; }

# ======================================================================== dry run
if [ "$MODE" = dry ]; then
  DR=$A/dryrun; mkdir -p $DR
  IG=${IGNORE:+--ignore-pids $IGNORE}
  log "preflight exit $CHECK_RC ($([ $CHECK_RC = 0 ] && echo 'run would pass the preflight' || echo 'run would ABORT'))"
  log "Tier-3 drivers in run mode now: $(echo "$DRV_OUT" | paste -sd';') -> $([ $DRV_RC = 0 ] && echo "'run' allowed" || echo "'run' would ABORT until the t3opus driver has exited (switch step 1)")"
  log "singleton: $(if flock -n $A/.driver.lock true 2>/dev/null; then echo 'f0/.driver.lock free'; else echo "f0/.driver.lock HELD (pid $(cat $A/driver.pid 2>/dev/null))"; fi)"
  log "config check (hydra compose, CPU; --no-write): $(hlp cfgcheck --no-write 2>&1 | tail -1)"
  log "launch order: $(hlp order | paste -sd' ')"
  log "prep table (frame references, scene, mitigation, flags, lock):"; hlp prep | sed 's/^/[dry]   /'
  log "GPU view now:"; hlp gpus | sed 's/^/[dry]   /'
  [ -n "$IGNORE" ] && { log "GPU view SIMULATED with pids $IGNORE (and children) gone:"; hlp gpus $IG | sed 's/^/[dry]   /'; }
  log "live opus_v2 trainers (switch plan):"; hlp opus-trainers | sed 's/^/[dry]   /' | cut -c1-400
  log "clip states now:"
  declare -A CST
  while read -r _ ep stt rest; do CST[$ep]=$stt; echo "[dry]   CLIP $ep $stt $rest" | cut -c1-260; done < <(hlp state)
  log "pids.tsv: $(cut -f1 $PIDS 2>/dev/null | paste -sd' ')"
  log "HOLD files ($A; a held clip is not launched, later clips go ahead): $(holds)"
  log "STOP file: $([ -f $STOPF ] && echo present || echo absent); GPU-choice lock files (read-only, never created): $(for f in "${LOCKS[@]}"; do printf '%s=%s ' "${f#$ME/}" "$([ -f "$f" ] && echo present || echo 'missing(skipped)')"; done)"
  log "LOCAL scoring self-test (kit packer, track_2_tier3) on the pilot's probe output (must reproduce tmp_analysis/tier3/s4b_probe_LOCAL_NOT_FOR_UPLOAD.json; no clean number printed here):"
  rm -rf $DR/score_selftest; SST=$(hlp score-selftest --work $DR/score_selftest); log "  exit $?: $SST"
  ex=""; assume=""; declare -A ASSUMED; NL=0
  for pass in now ${IGNORE:+simulated}; do
    ex=""; assume=""; unset ASSUMED; declare -A ASSUMED; NL=0
    [ $pass = simulated ] && log "==== training plans SIMULATED after the switch (pids $IGNORE gone; earlier launches of this list assumed started):"
    [ $pass = now ] && log "==== training plans NOW (earlier launches of this list assumed started):"
    for ep in $(hlp order); do
      st=${CST[$ep]:-?}
      case $st in FRESH) ;; *) echo "---- ep$ep ($st): nothing to launch"; continue;; esac
      if held $ep; then echo "---- ep$ep HELD ($A/HOLD_ep$ep present): not launched while it exists; later clips go ahead"; continue; fi
      if [ $pass = simulated ]; then line=$(hlp plan --ep $ep --exclude "$ex" --assume "$assume" $IG); else line=$(hlp plan --ep $ep --exclude "$ex" --assume "$assume"); fi
      NL=$((NL + 1))
      log "launch #$NL ep$ep training plan ($pass): $line"
      read -r tag g cpus nth cnote rest <<< "$line"
      if [ "$tag" != OK ]; then g="<gpu>"; cpus="<cpus>"; nth="<n>"; else ex="$ex,$cpus"; ASSUMED[$g]=$(( ${ASSUMED[$g]:-0} + 1 ))
        assume=$(for k in "${!ASSUMED[@]}"; do printf '%s:%s,' $k ${ASSUMED[$k]}; done); fi
      train_cmd $ep || { log "ep$ep: no command"; continue; }
      if [ $NL -le 3 ] || [ "$tag" = OK ]; then
        echo "---- ep$ep exact launch command ($pass; cwd $CODE; log $RUNS/logs/ep$ep.log):"
        print_cmd $ep $g $cpus $nth
      fi
    done
    log "planned ($pass): ${#ASSUMED[@]} GPU(s) take launches: $(for k in "${!ASSUMED[@]}"; do printf 'gpu%s x%s ' $k ${ASSUMED[$k]}; done); the rest wait for slots in ORDER"
  done
  echo "     env (all trainings): PATH=${PATH%%:*}:... PYTHONPATH=$PYTHONPATH PYTHONDONTWRITEBYTECODE=$PYTHONDONTWRITEBYTECODE XDG_CACHE_HOME=$XDG_CACHE_HOME CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE WANDB_DIR=$WANDB_DIR CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-<unset>} WANDB_API_KEY=${WANDB_API_KEY:+<set>}${WANDB_API_KEY:-<unset>}"
  echo "---- after final.json (any clip): ( exec setsid nohup env -u PYTHONPATH CUDA_VISIBLE_DEVICES= $FCPY -I -B $HELPER post --ep NN ) >> $A/eval/epNN/post_stdout.log 2>&1 < /dev/null &"
  echo "     post commands (training GPU from pids.tsv; shown for gpu 0, ep00 (floor), ep01 (support + mitigated)):"
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES= $FCPY -I -B - $HELPER 00 01 <<'PY' 2>/dev/null
import sys, importlib.util
spec = importlib.util.spec_from_file_location("t3f0h", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
for ep in sys.argv[2:]:
    w = m.EVROOT / f"ep{int(ep):02d}"
    print(f"       ep{ep}: " + " ".join(m.eval_command(int(ep), 0, w, m.EXPORT_TOLS[0])))
    print(f"       ep{ep} only if refused at 1e-4: the same with --export-tolerance {m.EXPORT_TOLS[1]!r}")
    print(f"       ep{ep} then (CPU, LOCAL): " + " ".join(m.score_command(int(ep), w)))
PY
  log "waiting state now (what K3CA / T2fix must yield to; --write only in run mode): $(hlp waiting)"
  log "verdict now (--no-write):"; hlp verdict --no-write | sed 's/^/[dry]   /' | cut -c1-330
  log "dry run done: nothing was started; files written only under $DR"
  exit 0
fi

# ======================================================================== adopt / run
hlp state | while read -r line; do log "start state: $(echo "$line" | cut -c1-240)"; done
[ "$MODE" = adopt ] && log "adopt mode: nothing is ever launched"
log "HOLD files at start ($A): $(holds)"

last_beat=0; last_verdict=$(date +%s)
while :; do
  if [ -f $STOPF ]; then log "STOP file: driver exits; trainings and evaluations keep running (restart: rm STOP, then 'adopt' or 'run')"
    rm -f $A/driver.pid; exit 0; fi
  mapfile -t ST < <(hlp state)
  [ ${#ST[@]} = $NCLIPS ] || { note state "state helper returned ${#ST[@]} lines (expected $NCLIPS): retrying"; sleep $POLL; continue; }
  note state ""
  settled=0; newterm=0; BLOCKED=0
  EXCL=""  # a slice only needs excluding until its trainer (started under taskset) is visible: within a pass (as v2)
  for line in "${ST[@]}"; do
    read -r _ ep stt rest <<< "$line"
    note st$ep "ep$ep $stt $(echo "$rest" | cut -c1-200)" "$stt"
    case $stt in
      FRESH)
        if [ "$MODE" = run ] && ! grep -q "^$ep"$'\t' $PIDS 2>/dev/null; then
          if held $ep; then note hold$ep "ep$ep HELD ($A/HOLD_ep$ep present): not launched; later clips go ahead" held
          else
            [ "${NOTE[hold$ep]:-}" = held ] && { NOTE[hold$ep]=""; log "ep$ep HOLD released (HOLD_ep$ep removed): back in the launch queue"; }
            [ $BLOCKED = 0 ] && { try_launch $ep || BLOCKED=1; }
          fi
        else settled=$((settled + 1)); fi;;
      NOT_READY|PARQUET_CHANGED) settled=$((settled + 1)); [ -z "${SEEN[$ep]:-}" ] && { SEEN[$ep]=notready; newterm=1; };;
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
  hlp waiting --write > /dev/null 2>&1 || note waiting "waiting.json update failed"
  now=$(date +%s)
  if [ $newterm = 1 ] || [ $((now - last_verdict)) -ge $VERDICT_S ]; then last_verdict=$now
    [ $newterm = 1 ] && log "verdict update:"
    hlp verdict > $A/verdict_last.log 2>&1 || log "verdict refused or failed (exit $?)"
    [ $newterm = 1 ] && grep -E '^(all|floor_only|support|mitigated|not_mitigated) ' $A/verdict_last.log >> $DLOG; fi
  if [ $((now - last_beat)) -ge $HEARTBEAT_S ]; then last_beat=$now
    echo "[$(date '+%F %T')] HEARTBEAT driver f0 $$ | $(hlp heartbeat)" >> $HB; fi
  [ $settled -ge $NCLIPS ] && break
  sleep $POLL
done
log "every clip is settled: verdict"
hlp verdict >> $DLOG 2>&1 || log "verdict refused or failed (exit $?)"
echo "[$(date '+%F %T')] DONE driver f0 $$ | $(hlp heartbeat)" >> $HB
rm -f $A/driver.pid
exit 0
