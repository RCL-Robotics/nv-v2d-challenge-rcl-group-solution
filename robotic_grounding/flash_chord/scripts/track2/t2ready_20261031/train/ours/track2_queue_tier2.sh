#!/bin/bash
# Tier-2 training queue: the UNCHANGED C3r2 method (experiment g1_recon_body_coord_rsi_r2, code fc_r2) trained on the
# Tier-2 synthetic-noise input (HUMAN_MOTION_DATA_DIR=$ME/data/human_motion_data_tier2). Adapted from
# track2_queue_v2.sh (round 2): same job model (one single-GPU job per episode, final.json / early stop = finished,
# resume from the newest finite checkpoint, never a fresh start over existing model_* files).
# Differences from v2, all because Tier 2 shares the GPUs with other people's non-MPS Genesis jobs:
#   - data root: DATA (default human_motion_data_tier2) is exported as HUMAN_MOTION_DATA_DIR AFTER fc_env.sh (which
#     hard-sets the Tier-1 root); the Tier-1 root is refused, and all 30 robot_name=g1/data.parquet must exist.
#   - own queue dir QDIR (default $ME/fc_runs/queue_tier2: queue.log, status.tsv, jobs.tsv, gpus.conf, scheduler.pid);
#     a second scheduler on the same QDIR is refused. fc_runs/queue (the Tier-1 queue's) is never touched.
#   - episode order: FIRST (default "07 16 18 20") first, then the other episodes in index order.
#   - GPU slots come from GPUS_CONF (default $QDIR/gpus.conf), re-read every scheduling loop (POLL s). Entries,
#     separated by spaces or newlines ('#' starts a comment):
#        <gpu>:<slots>       plain CUDA contexts (no MPS variables)
#        <gpu>:<slots>:mps   clients of this queue's private MPS daemon for that GPU
#     e.g. "1:2 3:2 7:3:mps". GPU 4 is always refused. Invalid content (bad entry, GPU 4, a GPU twice, no entries)
#     keeps the previous setting and is logged. Fewer slots than running jobs only stops new starts there (no job is
#     ever stopped); '<gpu>:0' = no new starts on that GPU. The file is created from GPUS (default "1:2 3:2") when it
#     does not exist; once it exists, GPUS is ignored.
#   - MPS, per GPU: daemon with CUDA_VISIBLE_DEVICES=<g>, pipe $MPS_ROOT/gpu<g>/pipe, log $MPS_ROOT/gpu<g>/log, started
#     on first need when no daemon serves that pipe (a daemon found there that sees another GPU blocks MPS starts).
#     Its clients get CUDA_VISIBLE_DEVICES=<the GPU's UUID> (accepted under the daemon, tested 2026-10-16 in
#     tmp_analysis/efficiency/crash; the physical index is rejected under MPS, and the daemon-local index 0 used before
#     would silently land on physical GPU 0 if the daemon were missing at CUDA init - with the UUID such a client runs
#     plain on its own GPU), that pipe/log and TIER2_GPU=<g>; no MPS start while the UUID cannot be read from
#     nvidia-smi. Never mixed: no MPS client starts on a GPU where this queue still runs
#     plain Tier-2 jobs, and no plain job where it runs MPS clients (other people's plain jobs are allowed neighbours).
#     The queue never signals a job. A daemon is quit only when its get_client_list is empty and no slot is configured
#     as mps for its GPU (or the queue is done).
#   - each job's GPU and mode are recorded in $QDIR/jobs.tsv and in its environment (TIER2_GPU). After a scheduler
#     restart running jobs are adopted with that GPU (else nvidia-smi; CUDA_VISIBLE_DEVICES only for plain jobs).
#   - GPU-choice locks shared with the other queues (seedstudy/.gpulock -> f0share05/.gpulock -> filler/.gpulock, the
#     order of gpu_filler.sh) are held while a GPU is chosen and HOLD s after each launch, so the new job's memory shows
#     up before anyone else chooses; at most one launch per lock hold.
#   - memory guard: a job starts on a GPU only when it has >= MIN_FREE_MB free plus JOB_MB for each job this queue
#     started there in the last RAMP_S seconds (memory grows during the ~4 min compile); else it waits.
#   - CPU slice per launch: up to MAX_CORES (6) physical cores (+HT siblings, cpu+104) of the GPU's NUMA node (GPU0-2
#     NUMA0 = cores 0-51, GPU3-7 NUMA1 = 52-103) that no process is pinned to as a slice (an affinity of <= 26 physical
#     cores, this queue's own jobs included; whole-socket pins claim nothing); fewer than MIN_CORES (2) free borrows the
#     other node's. POOL0 / POOL1 (cpu lists of physical cores) restrict the nodes' pools.
#   - adopts only running train_rl.py whose output_dir is $RUNS/<arm of ARMS>/epNN; other FlashCHORD runs are ignored.
#   - a stopped job is retried MAX_TRIES times in total (default 5), resuming from its checkpoint.
#   - DRY_RUN=1: print the settings, the job order and the exact command (+ MPS daemon action) of the next job for each
#     free slot (slots filled round-robin over the GPUs); starts nothing, writes nothing, takes no lock. Its slots come
#     from GPUS if set, else GPUS_CONF, else "1:2 3:2".
# Environment parameters (defaults = the Tier-2 round):
#   ARMS="C3r2_tier2:g1_recon_body_coord_rsi_r2"  ITERS=45000  GPUS="1:2 3:2" (seeds gpus.conf)  CODE=$ME/code/fc_r2
#   DATA=$ME/data/human_motion_data_tier2  RUNS=$ME/fc_runs  QDIR=$ME/fc_runs/queue_tier2  GPUS_CONF=$QDIR/gpus.conf
#   EPISODES=all ("0 1 ...")  FIRST="07 16 18 20"  MPS_ROOT=$ME/mps_tier2  LOCK_FILES="<3 lock files>"
#   MIN_FREE_MB=20000 JOB_MB=17000 RAMP_S=600 MAX_TRIES=5 HOLD=180 LOCK_WAIT=120 POLL=30 SEED=42 MAX_CORES=6 MIN_CORES=2
#   WANDB_ENTITY=juicer_test WANDB_PROJECT=v2d-track2-tier2 WANDB_MODE=online (WANDB_API_KEY only via the
#   launcher's environment, never written to disk)
# Change slots while it runs (picked up within POLL s; see queue.log):
#   cd $ME/fc_runs/queue_tier2 && echo '1:2 3:2 7:3:mps' > gpus.conf.new && mv gpus.conf.new gpus.conf
ME=/root/autodl-tmp/haoran.zhang
ARMS=${ARMS:-"C3r2_tier2:g1_recon_body_coord_rsi_r2"}
ITERS=${ITERS:-45000}; CODE=${CODE:-$ME/code/fc_r2}
DATA=${DATA:-$ME/data/human_motion_data_tier2}; SEED=${SEED:-42}; DRY_RUN=${DRY_RUN:-0}
HELPER=$ME/scripts/track2_queue_helper.py; PYI=$ME/conda_envs/v2d/bin/python
RUNS=${RUNS:-$ME/fc_runs}; POLL=${POLL:-30}; EPISODES=${EPISODES:-all}; FIRST=${FIRST:-"07 16 18 20"}
MIN_FREE_MB=${MIN_FREE_MB:-20000}; JOB_MB=${JOB_MB:-17000}; RAMP_S=${RAMP_S:-600}; MAX_TRIES=${MAX_TRIES:-5}
HOLD=${HOLD:-180}; LOCK_WAIT=${LOCK_WAIT:-120}; MAX_CORES=${MAX_CORES:-6}; MIN_CORES=${MIN_CORES:-2}
WANDB_ENTITY=${WANDB_ENTITY:-juicer_test}; WANDB_PROJECT=${WANDB_PROJECT:-v2d-track2-tier2}; WANDB_MODE=${WANDB_MODE:-online}
export WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-600} WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT:-600} WANDB_HTTP_TIMEOUT=${WANDB_HTTP_TIMEOUT:-180}
QDIR=${QDIR:-$ME/fc_runs/queue_tier2}; QLOG=$QDIR/queue.log; STATUS=$QDIR/status.tsv; JSTATE=$QDIR/jobs.tsv
GPUS_CONF=${GPUS_CONF:-$QDIR/gpus.conf}; MPS_ROOT=${MPS_ROOT:-$ME/mps_tier2}
read -r -a LOCKS <<< "${LOCK_FILES:-$ME/ch_runs/seedstudy/.gpulock $ME/ch_runs/f0share05/.gpulock $ME/ch_runs/filler/.gpulock}"
source $ME/scripts/fc_env.sh
export HUMAN_MOTION_DATA_DIR=$DATA PYTHONPATH=$CODE/src
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY   # plain contexts by default; MPS clients get their pipe per command
HMD=$HUMAN_MOTION_DATA_DIR
die(){ echo "track2_queue_tier2: $*" >&2; exit 1; }
if [ "$DRY_RUN" = 1 ]; then log(){ echo "[dry-run] $*"; }; else log(){ echo "[$(date '+%F %T')] $*" | tee -a $QLOG; }; fi
declare -A NOTE
note(){ # <id> <message> [<signature>]: log the message once while its signature (default: the message) stays the
       # same for that id; an empty message clears the id
  local sig=${3:-$2}; [ "${NOTE[$1]:-}" = "$sig" ] && return 0; NOTE[$1]=$sig; [ -n "$2" ] && log "$2"; return 0; }

# ---- gpus.conf: "<gpu>:<slots>" (plain) / "<gpu>:<slots>:mps"; parse_conf sets PC_SLOTS, PC_MODE, PC_CANON or PC_ERR
declare -A PC_SLOTS PC_MODE CONF_SLOTS CONF_MODE
CONF_CANON=""
parse_conf(){
  local text tok g re='^([0-9]):([0-9])(:mps)?$'; local -a toks=()
  PC_SLOTS=(); PC_MODE=(); PC_CANON=""; PC_ERR=""
  text=$(printf '%s\n' "$1" | sed 's/#.*//')
  read -r -d '' -a toks <<< "$text"
  for tok in "${toks[@]}"; do
    [[ $tok =~ $re ]] || { PC_ERR="bad entry '$tok' (use <gpu>:<slots> or <gpu>:<slots>:mps, slots 0-9)"; return 1; }
    g=${BASH_REMATCH[1]}
    [ "$g" = 4 ] && { PC_ERR="GPU 4 is a teammate's and always refused"; return 1; }
    [ "$g" -le 7 ] || { PC_ERR="there is no GPU $g"; return 1; }
    [ -n "${PC_SLOTS[$g]:-}" ] && { PC_ERR="GPU $g listed twice"; return 1; }
    PC_SLOTS[$g]=${BASH_REMATCH[2]}; PC_MODE[$g]=plain; [ -n "${BASH_REMATCH[3]}" ] && PC_MODE[$g]=mps
  done
  [ ${#PC_SLOTS[@]} -gt 0 ] || { PC_ERR="no entries (to pause, give 0 slots, e.g. '1:0 3:0')"; return 1; }
  for g in $(printf '%s\n' "${!PC_SLOTS[@]}" | sort -n); do
    PC_CANON+="$g:${PC_SLOTS[$g]}$([ ${PC_MODE[$g]} = mps ] && echo :mps) "; done
  PC_CANON=${PC_CANON% }; return 0
}
apply_conf(){ local g; CONF_SLOTS=(); CONF_MODE=()
  for g in "${!PC_SLOTS[@]}"; do CONF_SLOTS[$g]=${PC_SLOTS[$g]}; CONF_MODE[$g]=${PC_MODE[$g]}; done; CONF_CANON=$PC_CANON; }
conf_gpus(){ printf '%s\n' "${!CONF_SLOTS[@]}" | sort -n; }
write_conf_file(){ # <canonical entries>
  { cat <<EOF
# Tier-2 queue GPU slots (scripts/track2_queue_tier2.sh), re-read every scheduling loop (~${POLL} s).
# Entries, separated by spaces or newlines:  <gpu>:<slots>  plain    <gpu>:<slots>:mps  this queue's MPS daemon for that GPU
# GPU 4 is always refused. Invalid content keeps the previous setting (see queue.log). Fewer slots than running jobs
# only stops new starts there; no job is ever stopped. Plain and mps never mix on one GPU: a switch takes effect after
# this queue's jobs of the other kind there have ended.
# Edit atomically:  echo '1:2 3:2 7:3:mps' > gpus.conf.new && mv gpus.conf.new gpus.conf
EOF
    echo "$1"; } > "$GPUS_CONF.tmp" && mv "$GPUS_CONF.tmp" "$GPUS_CONF"; }
read_conf(){ # every loop: apply a changed valid gpus.conf; keep the previous setting on invalid / unreadable content
  local text
  if ! text=$(cat "$GPUS_CONF" 2>/dev/null); then note conf "gpus.conf unreadable ($GPUS_CONF): keeping [$CONF_CANON]"; return; fi
  if parse_conf "$text"; then
    note conf ""
    [ "$PC_CANON" = "$CONF_CANON" ] && return
    log "gpus.conf changed: [$CONF_CANON] -> [$PC_CANON]"; apply_conf
    local g; for g in $(conf_gpus); do local n; n=$(count_on $g)
      [ $n -gt ${CONF_SLOTS[$g]} ] && log "gpu$g runs $n jobs of this queue > ${CONF_SLOTS[$g]} slots: no new start there until below (nothing is stopped)"; done
  else note conf "gpus.conf INVALID ($PC_ERR): keeping [$CONF_CANON]"; fi
}

# ---- preflight (before anything is written)
[ "${MPS:-0}" = 0 ] || die "MPS=$MPS: MPS is chosen per GPU now - write '<gpu>:<slots>:mps' in gpus.conf (GPUS for a first start)"
[ "$(readlink -f $HMD)" = "$(readlink -f $ME/data/human_motion_data)" ] && die "DATA is the Tier-1 root; this queue trains on Tier 2"
[ -f $HMD/whole_body/soma/track2_manifest.json ] || die "no $HMD/whole_body/soma/track2_manifest.json"
[ -d $CODE/src/flash_chord ] || die "no FlashCHORD code at $CODE"
[ ${#LOCKS[@]} = 3 ] || die "LOCK_FILES must name 3 lock files (got ${#LOCKS[@]})"
if [ "$DRY_RUN" = 1 ]; then
  if [ -n "${GPUS:-}" ]; then ctext=$GPUS; csrc="GPUS='$GPUS'"
  elif [ -f "$GPUS_CONF" ]; then ctext=$(cat "$GPUS_CONF"); csrc=$GPUS_CONF
  else ctext="1:2 3:2"; csrc="default"; fi
  parse_conf "$ctext" || die "slots from $csrc: $PC_ERR"
else
  mkdir -p $QDIR
  if [ -f $QDIR/scheduler.pid ] && p=$(cat $QDIR/scheduler.pid) && [ "$p" != $$ ] && kill -0 $p 2>/dev/null \
     && grep -qa track2_queue_tier2 /proc/$p/cmdline 2>/dev/null; then die "a Tier-2 scheduler (pid $p) already runs on $QDIR"; fi
  if [ ! -f "$GPUS_CONF" ]; then
    parse_conf "${GPUS:-1:2 3:2}" || die "GPUS='${GPUS:-}': $PC_ERR"
    write_conf_file "$PC_CANON" || die "cannot write $GPUS_CONF"; csrc="$GPUS_CONF (created from GPUS='${GPUS:-1:2 3:2}')"
  else
    parse_conf "$(cat "$GPUS_CONF")" || die "$GPUS_CONF: $PC_ERR"; csrc=$GPUS_CONF
    [ -n "${GPUS:-}" ] && [ "$GPUS" != "$PC_CANON" ] && echo "GPUS='$GPUS' ignored: $GPUS_CONF exists and is authoritative" >> $QLOG
  fi
  echo $$ > $QDIR/scheduler.pid
fi
apply_conf

# ---- MPS daemons of this queue: one per GPU, pipe $MPS_ROOT/gpu<g>/pipe
mps_pipe(){ echo $MPS_ROOT/gpu$1/pipe; }
mps_logdir(){ echo $MPS_ROOT/gpu$1/log; }
declare -A GPU_UUID
gpu_uuid_known(){ # <gpu>: make sure GPU_UUID[<gpu>] holds its UUID (call it in this shell, not in $(...)); 1 = unknown
  [ -n "${GPU_UUID[$1]:-}" ] && return 0
  local u; u=$(timeout 60 nvidia-smi --query-gpu=uuid --format=csv,noheader -i $1 2>/dev/null | tr -d ' \r' | head -1)
  [[ $u =~ ^GPU-[0-9a-fA-F-]+$ ]] || return 1
  GPU_UUID[$1]=$u; return 0; }
mode_env(){ # <gpu> <mode> -> MENV: the CUDA/MPS variables of a job; 1 = an MPS client whose GPU UUID is unknown
  if [ "$2" = mps ]; then [ -n "${GPU_UUID[$1]:-}" ] || return 1
    MENV=(CUDA_VISIBLE_DEVICES=${GPU_UUID[$1]} CUDA_MPS_PIPE_DIRECTORY=$(mps_pipe $1) CUDA_MPS_LOG_DIRECTORY=$(mps_logdir $1))
  else MENV=(CUDA_VISIBLE_DEVICES=$1); fi; }
mps_daemon_info(){ # <pipe dir> -> "pid visible" of the MPS control daemon serving that pipe (nothing if none)
  local p e
  for p in $(pgrep -f '^(/[^ ]*/)?nvidia-cuda-mps-control'); do
    tr '\0' '\n' < /proc/$p/cmdline 2>/dev/null | grep -qx -- '-d' || continue
    e=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null) || continue
    [ "$(echo "$e" | sed -n 's/^CUDA_MPS_PIPE_DIRECTORY=//p')" = "$1" ] || continue
    echo "$p $(echo "$e" | sed -n 's/^CUDA_VISIBLE_DEVICES=//p')"; return 0
  done; return 0; }
mps_ctl(){ # <pipe> <command>: send one command to that daemon
  echo "$2" | env -u WANDB_API_KEY CUDA_MPS_PIPE_DIRECTORY=$1 CUDA_MPS_LOG_DIRECTORY=${1%/pipe}/log timeout 60 nvidia-cuda-mps-control; }
mps_clients(){ # <pipe> -> client pids of every server of that daemon; returns 1 when it cannot tell
  local out s c x
  out=$(mps_ctl $1 get_server_list 2>&1) || return 1
  for s in $out; do [[ $s =~ ^[0-9]+$ ]] || return 1
    c=$(mps_ctl $1 "get_client_list $s" 2>&1) || return 1
    for x in $c; do [[ $x =~ ^[0-9]+$ ]] || return 1; echo $x; done
  done; return 0; }
ensure_mps(){ # <gpu>: make sure this queue's daemon for that GPU runs (start it on first need); 1 = no MPS start possible
  local g=$1 pipe logd dpid dvis i
  pipe=$(mps_pipe $g); logd=$(mps_logdir $g)
  read -r dpid dvis <<< "$(mps_daemon_info $pipe)"
  if [ -n "$dpid" ]; then
    [ "$dvis" = "$g" ] && return 0
    note mps$g "gpu$g: the MPS daemon pid $dpid on $pipe sees CUDA_VISIBLE_DEVICES='$dvis', not $g: no MPS start there"; return 1
  fi
  mkdir -p $pipe $logd
  if ! env -u WANDB_API_KEY CUDA_VISIBLE_DEVICES=$g CUDA_MPS_PIPE_DIRECTORY=$pipe CUDA_MPS_LOG_DIRECTORY=$logd \
       nvidia-cuda-mps-control -d 6>&- 7>&- 8>&-; then note mps$g "gpu$g: cannot start the MPS daemon ($pipe)"; return 1; fi
  for i in 1 2 3 4 5 6 7 8 9 10; do read -r dpid dvis <<< "$(mps_daemon_info $pipe)"; [ -n "$dpid" ] && break; sleep 1; done
  if [ -z "$dpid" ] || [ "$dvis" != "$g" ]; then note mps$g "gpu$g: MPS daemon started but not found on $pipe: no MPS start"; return 1; fi
  note mps$g ""; log "MPS daemon for gpu$g started: pid $dpid, pipe $pipe, log $logd (CUDA_VISIBLE_DEVICES=$g)"; return 0
}
mps_housekeeping(){ # quit a daemon of this queue when no slot is configured as mps for its GPU (or the queue is done),
                    # no job of this queue is recorded as an MPS client there, and its get_client_list is empty
  local d g dpid dvis clients key busy
  for d in $MPS_ROOT/gpu*/pipe; do
    [ -d "$d" ] || continue
    g=${d#$MPS_ROOT/gpu}; g=${g%/pipe}
    read -r dpid dvis <<< "$(mps_daemon_info $d)"; [ -n "$dpid" ] || continue
    if [ "${ALL_DONE:-0}" != 1 ] && [ "${CONF_MODE[$g]:-}" = mps ] && [ "${CONF_SLOTS[$g]:-0}" -gt 0 ]; then continue; fi
    busy=0; for key in "${!J_PID[@]}"; do [ "${J_GPU[$key]}" = "$g" ] && [ "${J_MODE[$key]}" = mps ] && busy=1; done
    [ $busy = 0 ] || continue
    if ! clients=$(mps_clients $d); then note mpsq$g "gpu$g: cannot read the client list of MPS daemon pid $dpid: not quitting it"; continue; fi
    if [ -n "$clients" ]; then note mpsq$g "gpu$g: MPS daemon pid $dpid still has clients ($(echo $clients)): not quitting it"; continue; fi
    mps_ctl $d quit > /dev/null 2>&1
    note mpsq$g ""; log "MPS daemon for gpu$g (pid $dpid) quit: no clients, no mps slot configured for gpu$g"
  done
}

# ---- GPU-choice locks shared with gpu_filler.sh / seedstudy_queue.sh / gsfix_queue.sh (same order)
take_locks(){
  exec 8>>"${LOCKS[0]}" || return 1
  flock -w $LOCK_WAIT 8 || { exec 8>&-; return 1; }
  exec 7>>"${LOCKS[1]}" || { flock -u 8; exec 8>&-; return 1; }
  flock -w $LOCK_WAIT 7 || { exec 7>&-; flock -u 8; exec 8>&-; return 1; }
  exec 6>>"${LOCKS[2]}" || { flock -u 7; flock -u 8; exec 7>&- 8>&-; return 1; }
  flock -w $LOCK_WAIT 6 || { exec 6>&-; flock -u 7; flock -u 8; exec 7>&- 8>&-; return 1; }
  return 0; }
release_locks(){ flock -u 6; flock -u 7; flock -u 8; exec 6>&- 7>&- 8>&-; }

# ---- CPU slice for one new job on GPU g: free physical cores of its NUMA node (+HT siblings)
cpu_slice(){ # <gpu> [extra excluded cpu list] -> "cpus nthreads note" ("none 0 ..." if no core is free)
  $PYI -I - "$1" "${POOL0:-}" "${POOL1:-}" "$MAX_CORES" "$MIN_CORES" "${2:-}" <<'PY'
import os, sys
g, pool0, pool1, cap, floor_, excl = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
def parse(text):
    cpus = set()
    for part in text.split(","):
        if part and part != "none":
            a, _, b = part.partition("-"); cpus.update(range(int(a), int(b or a) + 1))
    return cpus
full, pinned = set(range(208)), {c % 104 for c in parse(excl)}
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        if not open(f"/proc/{pid}/cmdline", "rb").read().strip(b"\0"):
            continue  # kernel thread
        status = open(f"/proc/{pid}/status").read()
    except OSError:
        continue
    allowed = next((l.split(":", 1)[1].strip() for l in status.splitlines() if l.startswith("Cpus_allowed_list:")), "")
    if not allowed or parse(allowed) >= full:
        continue
    cores = {c % 104 for c in parse(allowed)}
    if len(cores) <= 26:  # a slice (e.g. another queue's job or a teammate's 8-core job); a whole-socket pin claims none
        pinned |= cores
node = 0 if g <= 2 else 1
nodes = [set(range(52)), set(range(52, 104))]
pools = [parse(pool0) & nodes[0] if pool0 else nodes[0], parse(pool1) & nodes[1] if pool1 else nodes[1]]
own, other = sorted(pools[node] - pinned), sorted(pools[1 - node] - pinned)
cores, note = own[:cap], ""
if len(cores) < floor_:
    extra = other[: floor_ - len(cores)]
    cores += extra
    note = "cross-NUMA" if extra else ""
if not cores:
    print("none 0 no-free-core")
else:
    print(",".join(f"{c},{c + 104}" for c in cores), 2 * len(cores), note)
PY
}

# ---- jobs: "arm exp ep seq obj" in ARMS priority, then FIRST episodes, then episode index order
mapfile -t JOBS < <($PYI -I - "$HMD" "$ARMS" "$EPISODES" "$FIRST" <<'PY'
import json, sys
hmd, arms, episodes, first = sys.argv[1:5]
m = json.load(open(f"{hmd}/whole_body/soma/track2_manifest.json"))
first = [int(e) for e in first.replace(",", " ").split()]
order = lambda r: (0, first.index(r[0])) if r[0] in first else (1, r[0])
byep = sorted(((r["episode_index"], s, r["object"]) for s, r in m.items()), key=order)
wanted = None if episodes == "all" else {int(e) for e in episodes.replace(",", " ").split()}
for arm_exp in arms.split():
    arm, exp = arm_exp.split(":")
    for ep, s, o in byep:
        if wanted is None or ep in wanted:
            print(f"{arm} {exp} {ep:02d} {s} {o}")
PY
)
[ ${#JOBS[@]} -gt 0 ] || die "no jobs (ARMS=$ARMS EPISODES=$EPISODES)"
for j in "${JOBS[@]}"; do set -- $j; [ -f "$HMD/whole_body/soma/sequence_id=$4/robot_name=g1/data.parquet" ] || die "missing G1 reference for ep$3 ($4) in $HMD"; done

outdir(){ echo "$RUNS/$1/ep$2"; }   # arm ep
done_job(){ python $HELPER done "$(outdir $1 $2)" $ITERS > /dev/null 2>&1; }
done_why(){ python $HELPER done "$(outdir $1 $2)" $ITERS 2>/dev/null; }
support_extra(){ local sup="$HMD/whole_body/reconstructed_stage/${1}_support.usda"   # seq -> "scene.support=false" for floor-only
  { [ -f "$sup" ] && grep -q -E "def (Cylinder|Cube)" "$sup"; } || echo scene.support=false; }
train_cmd(){ # arm exp ep seq obj out -> the training command line (array TRAIN)
  TRAIN=(python scripts/train_rl.py experiment=$2 "task.parquet='$HMD/whole_body/soma/sequence_id=$4/robot_name=g1/data.parquet'"
         reset.seed=$SEED output_dir=$6 training.max_iterations=$ITERS logging.mode=$WANDB_MODE logging.entity=$WANDB_ENTITY
         logging.project=$WANDB_PROJECT logging.experiment=$1 logging.run_name=ep${3}_${5} logging.upload_checkpoints=false
         $(support_extra $4)); }
gpu_free_mb(){ nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $1 2>/dev/null | tr -dc 0-9; }
smi_gpu_of(){ # pid -> physical index of the GPU nvidia-smi lists it on (empty if not listed)
  local uuid; uuid=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null | awk -F', ' -v p=$1 '$1==p {print $2; exit}')
  [ -n "$uuid" ] && nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | awk -F', ' -v u=$uuid '$2==u {print $1; exit}'; }
run_job(){ # arm exp ep seq obj gpu mode attempt cpus nthreads
  local arm=$1 exp=$2 ep=$3 seq=$4 obj=$5 g=$6 mode=$7 a=$8 cpus=$9 nth=${10} out
  out=$(outdir $arm $ep); mkdir -p $RUNS/$arm/logs
  local resume=() last="" hrc=0 notes line; notes=$(mktemp $QDIR/.resume_XXXXXX)
  if [ -d $out ]; then last=$(python $HELPER resume $out --quarantine 2> $notes); hrc=$?; fi   # newest finite model_N (+ rsi_N)
  while IFS= read -r line; do log "$arm/ep$ep resume check: $line"; done < $notes; rm -f $notes
  if [ $hrc -ne 0 ] || { [ -z "$last" ] && compgen -G "$out/model_*.safetensors" > /dev/null; }; then
    log "$arm/ep$ep NOT started (attempt fails, exit 97): resume check exit $hrc, resume point '${last:-none}' - $out has model_* files a fresh run would overwrite, or the helper failed; inspect and fix by hand"
    echo -e "$(date '+%F %T')\t$arm\tep$ep\t$obj\tgpu$g\t$mode\tcpus$cpus\tattempt$a\tresume_check_failed$hrc" >> $STATUS; return 97
  fi
  [ -n "$last" ] && resume+=(training.resume=true "training.checkpoint=$last")
  train_cmd $arm $exp $ep $seq $obj $out
  if ! mode_env $g $mode; then   # never expected: try_start reads the UUID first
    log "$arm/ep$ep NOT started (attempt fails, exit 98): no UUID known for gpu$g (an MPS client names its GPU by UUID)"
    echo -e "$(date '+%F %T')\t$arm\tep$ep\t$obj\tgpu$g\t$mode\tcpus$cpus\tattempt$a\tno_gpu_uuid" >> $STATUS; return 98
  fi
  ( cd $CODE && exec env TIER2_GPU=$g "${MENV[@]}" WANDB_MODE=$WANDB_MODE OMP_NUM_THREADS=$nth MKL_NUM_THREADS=$nth OPENBLAS_NUM_THREADS=$nth \
      taskset -c $cpus "${TRAIN[@]}" "${resume[@]}" ) >> $RUNS/$arm/logs/ep$ep.log 2>&1
  local rc=$?; echo -e "$(date '+%F %T')\t$arm\tep$ep\t$obj\tgpu$g\t$mode\tcpus$cpus\tattempt$a\texit$rc" >> $STATUS; return $rc
}

# ---- this queue's jobs, keyed "arm/ep": J_PID (pid watched: the runner, or the trainer when adopted), J_GPU (physical),
#      J_MODE (plain|mps), J_T0 (start, 0 = adopted), J_ADOPTED, J_JOB (job line), T_PID (train_rl.py pid)
declare -A J_PID J_GPU J_MODE J_T0 J_ADOPTED J_JOB T_PID TRIES
count_on(){ local key n=0; for key in "${!J_PID[@]}"; do [ "${J_GPU[$key]}" = "$1" ] && n=$((n+1)); done; echo $n; }
other_mode_on(){ # <gpu> <mode> -> the other mode if this queue runs jobs of it on that GPU
  local key; for key in "${!J_PID[@]}"; do
    [ "${J_GPU[$key]}" = "$1" ] && [ "${J_MODE[$key]}" != "$2" ] && { echo ${J_MODE[$key]}; return; }; done; }
write_state(){
  [ "$DRY_RUN" = 1 ] && return 0
  local key tp out
  { printf '#key\tpid\tgpu\tmode\tstarted\ttrain_pid\n'
    for key in "${!J_PID[@]}"; do
      tp=${T_PID[$key]:-}
      if [ -z "$tp" ] || ! kill -0 $tp 2>/dev/null; then
        set -- ${J_JOB[$key]}; out=$(outdir $1 $3); tp=$(pgrep -f "scripts/train_rl.py .*output_dir=$out( |\$)" | head -1); T_PID[$key]=$tp; fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$key" "${J_PID[$key]}" "${J_GPU[$key]}" "${J_MODE[$key]}" \
        "$([ "${J_T0[$key]}" = 0 ] && echo adopted || date -d @${J_T0[$key]} '+%F %T')" "${tp:-?}"
    done; } > $JSTATE.tmp && mv $JSTATE.tmp $JSTATE
}

# ---- adopt this queue's own running jobs (output_dir = $RUNS/<arm of ARMS>/epNN) after a scheduler restart
for pid in $(pgrep -f "python scripts/train_rl.py experiment="); do
  cmd=$(tr "\0" " " < /proc/$pid/cmdline 2>/dev/null) || continue
  out=$(echo "$cmd" | grep -o "output_dir=[^ ]*" | cut -d= -f2); [ -n "$out" ] || continue
  arm=$(basename $(dirname $out)); ep=$(basename $out | sed "s/^ep//"); [ "$(dirname $(dirname $out))" = "$RUNS" ] || continue
  job=$(printf "%s\n" "${JOBS[@]}" | awk -v a=$arm -v e=$ep '$1==a && $3==e' | head -1); [ -n "$job" ] || continue
  key="$arm/$ep"; envv=$(tr "\0" "\n" < /proc/$pid/environ 2>/dev/null)
  mode=plain; echo "$envv" | grep -q "^CUDA_MPS_PIPE_DIRECTORY=." && mode=mps
  g=$(echo "$envv" | sed -n "s/^TIER2_GPU=//p"); src=TIER2_GPU
  [ -n "$g" ] || { g=$(awk -F'\t' -v k="$key" '$1==k {print $3}' $JSTATE 2>/dev/null | tail -1); src=jobs.tsv; }
  [ -n "$g" ] || { g=$(smi_gpu_of $pid); src=nvidia-smi; }
  if [ -z "$g" ] && [ $mode = plain ]; then g=$(echo "$envv" | sed -n "s/^CUDA_VISIBLE_DEVICES=//p"); [[ $g =~ ^[0-7]$ ]] || g=""; src=CUDA_VISIBLE_DEVICES; fi
  [ -n "$g" ] || { g="?"; src="nowhere: it holds no slot on any GPU"; }
  J_PID[$key]=$pid; T_PID[$key]=$pid; J_GPU[$key]=$g; J_MODE[$key]=$mode; J_T0[$key]=0; J_ADOPTED[$key]=1; J_JOB[$key]="$job"
  log "adopted running $key pid $pid on gpu$g ($mode; gpu from $src; cpus $(taskset -cp $pid 2>/dev/null | sed 's/.*: //'))"
done
QUEUE=(); for j in "${JOBS[@]}"; do set -- $j; [ -n "${J_PID[$1/$3]:-}" ] && continue; done_job $1 $3 && continue; QUEUE+=("$j"); done
log "queue tier2: ARMS=[$ARMS] ITERS=$ITERS slots [$CONF_CANON] from $csrc CODE=$CODE ($(git -C $CODE rev-parse --short HEAD 2>/dev/null) $(git -C $CODE rev-parse --abbrev-ref HEAD 2>/dev/null)) DATA=$HMD W&B=$WANDB_ENTITY/$WANDB_PROJECT ($WANDB_MODE) adopted=${#J_PID[@]} queued=${#QUEUE[@]} min_free=${MIN_FREE_MB}MB+${JOB_MB}MB/ramping job hold=${HOLD}s max_tries=$MAX_TRIES mps_clients=CUDA_VISIBLE_DEVICES=<gpu uuid>"
log "episode order: $(for j in "${QUEUE[@]}"; do set -- $j; printf 'ep%s ' $3; done)"

if [ "$DRY_RUN" = 1 ]; then
  log "W&B key in environment: $([ -n "${WANDB_API_KEY:-}" ] && echo yes || echo 'no (the launcher passes it)')"
  declare -A FREE_SLOTS; i=0; excl=""
  for g in $(conf_gpus); do
    m=${CONF_MODE[$g]}; n=$(count_on $g); other=$(other_mode_on $g $m); FREE_SLOTS[$g]=$(( ${CONF_SLOTS[$g]} - n ))
    log "gpu$g: ${CONF_SLOTS[$g]} $m slot(s), $n running job(s) of this queue, $(gpu_free_mb $g) MB free now (a start needs >= $MIN_FREE_MB MB + $JOB_MB MB per job of this queue started there < $RAMP_S s ago)"
    [ -n "$other" ] && { log "gpu$g: this queue runs $other jobs there - no $m start until they end (never mix)"; FREE_SLOTS[$g]=0; }
    if [ $m = mps ] && [ ${FREE_SLOTS[$g]} -gt 0 ]; then
      read -r dpid dvis <<< "$(mps_daemon_info $(mps_pipe $g))"
      if [ -z "$dpid" ]; then log "gpu$g: would start the MPS daemon: env -u WANDB_API_KEY CUDA_VISIBLE_DEVICES=$g CUDA_MPS_PIPE_DIRECTORY=$(mps_pipe $g) CUDA_MPS_LOG_DIRECTORY=$(mps_logdir $g) nvidia-cuda-mps-control -d"
      elif [ "$dvis" = $g ]; then log "gpu$g: would join the running MPS daemon pid $dpid ($(mps_pipe $g))"
      else log "gpu$g: MPS daemon pid $dpid on $(mps_pipe $g) sees '$dvis', not $g: MPS starts would be refused"; FREE_SLOTS[$g]=0; fi
    fi
  done
  for r in 1 2 3 4 5 6 7 8 9; do for g in $(conf_gpus); do   # round-robin, as the scheduler prefers the GPU with fewest jobs
    [ ${FREE_SLOTS[$g]} -ge $r ] || continue; [ $i -lt ${#QUEUE[@]} ] || break 2
    m=${CONF_MODE[$g]}
    if [ $m = mps ] && ! gpu_uuid_known $g; then echo "---- gpu$g: its UUID cannot be read from nvidia-smi: no MPS start there"; continue; fi
    set -- ${QUEUE[$i]}; out=$(outdir $1 $3); train_cmd $1 $2 $3 $4 $5 $out; mode_env $g $m
    read -r cpus nth cnote <<< "$(cpu_slice $g "$excl")"; excl="$excl,$cpus"
    res="fresh start"; [ -d $out ] && res="resume check: $(python $HELPER resume $out 2>&1 | tr '\n' ' ')"
    echo "---- job $((i+1))/${#QUEUE[@]}: $1/ep$3 ($5) -> gpu$g ($m); $res; cpus $cpus${cnote:+ ($cnote)}; log $RUNS/$1/logs/ep$3.log"
    printf '( cd %q && exec env TIER2_GPU=%s' "$CODE" $g; printf ' %q' "${MENV[@]}"
    printf ' WANDB_MODE=%s OMP_NUM_THREADS=%s MKL_NUM_THREADS=%s OPENBLAS_NUM_THREADS=%s taskset -c %s' $WANDB_MODE $nth $nth $nth $cpus
    printf ' %q' "${TRAIN[@]}"; echo " ) >> $RUNS/$1/logs/ep$3.log 2>&1"
    echo "     env: HUMAN_MOTION_DATA_DIR=$HUMAN_MOTION_DATA_DIR PYTHONPATH=$PYTHONPATH XLA_PYTHON_CLIENT_PREALLOCATE=$XLA_PYTHON_CLIENT_PREALLOCATE CUDA_MPS_PIPE_DIRECTORY(scheduler)=${CUDA_MPS_PIPE_DIRECTORY:-<unset>} WANDB_DIR=$WANDB_DIR"
    i=$((i+1))
  done; done
  [ $i -lt ${#QUEUE[@]} ] && echo "---- the other $(( ${#QUEUE[@]} - i )) jobs wait for a free slot: $(for j in "${QUEUE[@]:$i}"; do set -- $j; printf 'ep%s ' $3; done)"
  exit 0
fi

# ---- scheduling
reap_jobs(){
  local key pid rc why t arm ep
  for key in "${!J_PID[@]}"; do
    pid=${J_PID[$key]}; kill -0 $pid 2>/dev/null && continue
    rc=0; [ -z "${J_ADOPTED[$key]:-}" ] && { wait $pid; rc=$?; }
    set -- ${J_JOB[$key]}; arm=$1; ep=$3
    if why=$(done_why $arm $ep); then log "$key finished on gpu${J_GPU[$key]} (${J_MODE[$key]}): $why"
    else t=${TRIES[$key]:-1}
      if [ $t -lt $MAX_TRIES ]; then TRIES[$key]=$((t+1)); QUEUE=("${J_JOB[$key]}" "${QUEUE[@]}")
        log "$key stopped on gpu${J_GPU[$key]} without final.json / model_$ITERS (exit $rc), requeued at front (attempt $((t+1))/$MAX_TRIES, will resume)"
      else log "$key FAILED $t times (exit $rc) - giving up"; fi
    fi
    unset "J_PID[$key]" "J_GPU[$key]" "J_MODE[$key]" "J_T0[$key]" "J_ADOPTED[$key]" "J_JOB[$key]" "T_PID[$key]"
  done
}
mem_ok(){ # <gpu> <now>: memory guard -> MEM_FREE, MEM_NEED, MEM_RAMP; status 0 = a job may start there
  local key; MEM_RAMP=0
  for key in "${!J_PID[@]}"; do [ "${J_GPU[$key]}" = $1 ] && [ $(( $2 - ${J_T0[$key]:-0} )) -lt $RAMP_S ] && MEM_RAMP=$((MEM_RAMP+1)); done
  MEM_FREE=$(gpu_free_mb $1); MEM_NEED=$(( MIN_FREE_MB + MEM_RAMP * JOB_MB ))
  [ -n "$MEM_FREE" ] && [ "$MEM_FREE" -ge $MEM_NEED ]; }
try_start(){ # at most one launch: choose a GPU under the shared locks, start the next job, hold the locks HOLD s
  local g m n other dpid dvis cand=() best="" bestn=0 bestfree=0 now key job cpus nth cnote
  now=$(date +%s)
  for g in $(conf_gpus); do
    m=${CONF_MODE[$g]}; n=$(count_on $g)
    [ $n -lt ${CONF_SLOTS[$g]} ] || { note gpu$g ""; continue; }
    other=$(other_mode_on $g $m)
    if [ -n "$other" ]; then note gpu$g "gpu$g is configured $m but this queue still runs $other Tier-2 job(s) there: no start until they end (never mix)"; continue; fi
    if [ $m = mps ]; then
      read -r dpid dvis <<< "$(mps_daemon_info $(mps_pipe $g))"
      if [ -n "$dpid" ] && [ "$dvis" != "$g" ]; then note gpu$g "gpu$g: the MPS daemon pid $dpid on $(mps_pipe $g) sees '$dvis', not $g: no MPS start"; continue; fi
    fi
    if ! mem_ok $g $now; then note gpu$g "gpu$g slot waits: ${MEM_FREE:-?} MB free < $MEM_NEED MB ($MEM_RAMP job(s) of this queue still ramping there)" "mem $MEM_NEED"; continue; fi
    cand+=($g)
  done
  [ ${#cand[@]} -gt 0 ] || return 0
  if ! take_locks; then note locks "GPU-choice locks busy for > ${LOCK_WAIT}s (another queue is choosing a GPU): retrying"; return 0; fi
  note locks ""; now=$(date +%s)
  for g in "${cand[@]}"; do   # re-check under the locks: another queue may have launched meanwhile
    if ! mem_ok $g $now; then note gpu$g "gpu$g slot waits: ${MEM_FREE:-?} MB free < $MEM_NEED MB ($MEM_RAMP job(s) of this queue still ramping there)" "mem $MEM_NEED"; continue; fi
    note gpu$g ""; n=$(count_on $g)
    if [ -z "$best" ] || [ $n -lt $bestn ] || { [ $n -eq $bestn ] && [ $MEM_FREE -gt $bestfree ]; }; then best=$g; bestn=$n; bestfree=$MEM_FREE; fi
  done
  [ -n "$best" ] || { release_locks; return 0; }
  g=$best; m=${CONF_MODE[$g]}
  [ "$g" = 4 ] && { release_locks; return 0; }   # never (parse_conf refuses it too)
  read -r cpus nth cnote <<< "$(cpu_slice $g)"
  if [ -z "$cpus" ] || [ "$cpus" = none ]; then note cpu$g "gpu$g: no free CPU core for a slice: waiting"; release_locks; return 0; fi
  note cpu$g ""
  if [ $m = mps ] && ! gpu_uuid_known $g; then
    note uuid$g "gpu$g: its UUID cannot be read from nvidia-smi: no MPS start (a client must name its GPU by UUID)"; release_locks; return 0; fi
  note uuid$g ""
  if [ $m = mps ] && ! ensure_mps $g; then release_locks; return 0; fi
  job="${QUEUE[0]}"; QUEUE=("${QUEUE[@]:1}"); set -- $job; key="$1/$3"
  run_job $1 $2 $3 $4 $5 $g $m ${TRIES[$key]:-1} $cpus $nth 6>&- 7>&- 8>&- &
  J_PID[$key]=$!; J_GPU[$key]=$g; J_MODE[$key]=$m; J_T0[$key]=$now; J_ADOPTED[$key]=""; J_JOB[$key]="$job"; T_PID[$key]=""
  log "start $key ($5) on gpu$g ($m$([ $m = mps ] && echo ", CUDA_VISIBLE_DEVICES=${GPU_UUID[$g]}")) cpus $cpus${cnote:+ ($cnote)} OMP $nth, runner pid $!, attempt ${TRIES[$key]:-1} (gpu$g had $bestfree MB free and $bestn job(s) of this queue); GPU-choice locks held ${HOLD}s"
  write_state
  sleep $HOLD
  release_locks
}
ALL_DONE=0; last_beat=$(date +%s)
while :; do
  read_conf
  reap_jobs
  mps_housekeeping
  [ ${#QUEUE[@]} -gt 0 ] && try_start
  write_state
  [ ${#J_PID[@]} -eq 0 ] && [ ${#QUEUE[@]} -eq 0 ] && break
  if [ $(( $(date +%s) - last_beat )) -ge 3600 ]; then last_beat=$(date +%s)
    log "HEARTBEAT slots [$CONF_CANON]; running: $(for key in "${!J_PID[@]}"; do printf '%s@gpu%s(%s) ' $key ${J_GPU[$key]} ${J_MODE[$key]}; done); queued ${#QUEUE[@]}"; fi
  sleep $POLL
done
ALL_DONE=1
for arm_exp in $ARMS; do arm=${arm_exp%%:*}; n=0; early=0
  for j in "${JOBS[@]}"; do set -- $j; [ "$1" = "$arm" ] || continue; why=$(done_why $1 $3) && { n=$((n+1)); case "$why" in *"early stop"*) early=$((early+1));; esac; }; done
  log "QUEUE_DONE: $arm $n/$(printf "%s\n" "${JOBS[@]}" | awk -v a=$arm '$1==a' | wc -l) episodes finished ($early by early stop; final.json or model_$ITERS)"; done
mps_housekeeping
rm -f $QDIR/scheduler.pid
exit 0
