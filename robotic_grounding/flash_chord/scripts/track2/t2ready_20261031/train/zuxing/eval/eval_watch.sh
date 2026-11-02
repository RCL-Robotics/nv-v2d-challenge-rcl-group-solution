#!/usr/bin/env bash
# Deployment-condition evaluation watcher for runs/2026-10-14-psi-worst3 (g1_recon_body_coord_psi_r2, ep19/ep04/ep01).
#
# While training runs, evaluates every model_<N>.safetensors with N % EVERY == 0 (and the final checkpoint named by
# each run's final.json) once, through scripts/track2/eval_all.py of dev.zuxing: E2 export (CPU, checked against the
# stock runtime) -> the kit's official evaluator on the stock 8c778f9 runtime (frame-0 start, no VOC help, 1 world).
# Each (episode, N) runs in its own dir $OUT/ep<NN>/it<N>/, whose ep<NN>/model_<N>.safetensors is a symlink to the
# training checkpoint, so nothing is written into the training run dirs. One row per attempt goes to
# $OUT/metrics.csv. Details, the exact command line and the source citations: EVAL_NOTES.md.
#
# Launch with CUDA_VISIBLE_DEVICES set to ONE physical GPU index (e.g. 4); this script does not set it.
#   CUDA_VISIBLE_DEVICES=4 bash eval_watch.sh
#   DRY_RUN=1 CUDA_VISIBLE_DEVICES=4 bash eval_watch.sh   # preflight + scan + print the next command; writes nothing
#   ONCE=1 ...                                            # evaluate what is ready now, then exit
# Stop: touch $OUT/STOP (checked between evaluations and every 10 s while idle; a running eval finishes first).
# Exit: STOP, or every run has final.json and every scheduled checkpoint is finished (ok, or failed twice).
# The eval_watch_helper.py next to this file must be deployed with it.
# Tier 2 copy (tmp/t2/eval, 2026-10-24): after every ok evaluation the kit packer (tools/pack_submission.py
# --track track_2_tier2, through the teammate's track2_tier2_clean_scores.py, copied unchanged) scores the rollout
# against the CLEAN reference, one row per (episode, iteration) in $OUT/tier2_clean.csv; metrics.csv stays
# against the run's own training input. Episodes trained on K3fix: FRAME_REFS_EPISODES (their frame references).

set -uo pipefail

# ------------------------------------------------------------------------------------------------ settings
ROOT=/root/autodl-tmp/zuxing.lu
RUNS=${RUNS:-$ROOT/runs/2026-10-14-psi-worst3}
OUT=${OUT:-$RUNS/eval}
EPISODES=${EPISODES:-"19 4 1"}
EVERY=${EVERY:-2000}               # evaluate model_<k*EVERY> (+ the final checkpoint)
POLL=${POLL:-300}                  # s between scans when nothing is ready
MIN_AGE=${MIN_AGE:-60}             # s a checkpoint must be old (the runner also writes it atomically)
RETRY_DELAY=${RETRY_DELAY:-600}    # s between a failure and its single retry
EVAL_TIMEOUT=${EVAL_TIMEOUT:-1800} # s per eval_all.py call (normal: ~45-65 s); then SIGTERM, SIGKILL 60 s later
FAIL_PAUSE=${FAIL_PAUSE:-1800}     # s pause after 3 consecutive failures (looks system-wide)
EXPORT_TOLERANCE=${EXPORT_TOLERANCE:-1e-4} # exporter default 1e-5 is exceeded by long runs (C3r2_wide it20000: 1.06e-5)
EVAL_CPUS=${EVAL_CPUS:-}           # optional taskset core list, e.g. 100-103,204-207; empty = no pinning

CODE=${CODE:-$ROOT/worktrees/dev.zuxing.help_force_fix/robotic_grounding/flash_chord}  # help_force_fix copy: export with the new code
STOCK=$ROOT/ref_repos/video_to_data_8c778f9/robotic_grounding/flash_chord
KIT=${KIT:-$ROOT/ref_repos/v2d_submission_kit}
HZ=/root/autodl-tmp/haoran.zhang   # read-only
DATASET=$HZ/data/video_to_data_challenge
MANIFEST=$HZ/data/human_motion_data/whole_body/soma/track2_manifest.json
CONDA=$HZ/conda_envs/flashchord
PY=$CONDA/bin/python
BASELINE=$HZ/fc_runs/eval_watch/state.csv
BASELINE_ARM=C3r2_wide
KPY=${KPY:-$ROOT/envs/kit/bin/python}
COMMIT=https://github.com/local-placeholder/not-uploaded/commit/0000000000000000000000000000000000000000
FRAME_REFS=${FRAME_REFS:-$HZ/tmp_analysis/tier2/k3fix_pilot/frame_refs_k3fix}   # read-only
FRAME_REFS_EPISODES=${FRAME_REFS_EPISODES:-}
HERE=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
HELPER=$HERE/eval_watch_helper.py
STOP=$OUT/STOP
CSV=$OUT/metrics.csv

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { printf '[%s] %s\n' "$(ts)" "$*"; }
die() { log "ERROR: $*"; exit 2; }

# ----------------------------------------------------------------------------------------------- environment
# Workspace .env (TZ, XDG_CACHE_HOME, PYTHONPYCACHEPREFIX), then the evaluation environment. LD_LIBRARY_PATH is built
# as haoran.zhang/scripts/fc_env.sh does (pip CUDA 13 / cuDNN libs for onnxruntime-gpu), without sourcing that file.
[[ -f $ROOT/.env ]] && { set -a; . "$ROOT/.env"; set +a; }
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME=$ROOT/.cache
export PYTHONPYCACHEPREFIX=$ROOT/.cache/pycache
export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_PREALLOCATE=false
SP=$CONDA/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH=$SP/cu13/lib:$SP/cudnn/lib:$SP/nccl/lib:$SP/nvshmem/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=$CONDA/bin:$PATH
# eval_all.py needs only the stdlib; it sets PYTHONPATH itself for its children ($CODE/src for the export,
# $STOCK/src for the evaluator). The eval is a plain non-MPS client, as in the teammate's watcher.
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY

# eval_all.py writes --gpus verbatim into the evaluator's CUDA_VISIBLE_DEVICES (it replaces the inherited one), so it
# must be the PHYSICAL index: pass the launcher's CUDA_VISIBLE_DEVICES through (--gpus 0 would run on physical GPU 0).
GPU=${CUDA_VISIBLE_DEVICES:-}

# ------------------------------------------------------------------------------------------------- preflight
preflight() {
  [[ $GPU =~ ^[0-9]+$ ]] || die "CUDA_VISIBLE_DEVICES must be exactly one physical GPU index, got '${GPU}'"
  [[ -x $PY ]] || die "python not found: $PY"
  local f
  for f in "$CODE/scripts/track2/eval_all.py" "$CODE/src/flash_chord/training/ppo/coordinate/export.py" \
           "$STOCK/src/flash_chord/__init__.py" "$STOCK/scripts/evaluate_policy.py" \
           "$KIT/examples/run_policy_evaluation.py" "$KIT/flash_chord/scripts/evaluate_policy.py" \
           "$KIT/tools/pack_submission.py" "$KIT/v2dlb/report_schema.py" "$MANIFEST" "$HELPER"; do
    [[ -f $f ]] || die "missing $f$([[ $f == "$KIT"* ]] && echo " (copy the kit: rsync -a --exclude __pycache__ $HZ/v2d_submission_kit_src/v2d_submission_kit/ $KIT/)")"
  done
  for f in "$DATASET" "$SP/cu13/lib" "$SP/cudnn/lib" "$SP/nccl/lib" "$SP/nvshmem/lib"; do
    [[ -d $f ]] || die "missing directory $f"
  done
  grep -q -- '--export-tolerance' "$CODE/scripts/track2/eval_all.py" || die "eval_all.py has no --export-tolerance"
  local kit_hash stock_hash
  kit_hash=$(sha256sum "$KIT/flash_chord/scripts/evaluate_policy.py" | cut -d' ' -f1)
  stock_hash=$(sha256sum "$STOCK/scripts/evaluate_policy.py" | cut -d' ' -f1)
  [[ $kit_hash == "$stock_hash" ]] || die "kit evaluate_policy.py ($kit_hash) differs from the stock one ($stock_hash)"
}

# One eval_all.py call for (episode, iteration); the command is built in CMD.
build_cmd() {
  local ep=$1 it=$2 work=$3
  CMD=(nice -n 10)
  [[ -n $EVAL_CPUS ]] && CMD+=(taskset -c "$EVAL_CPUS")
  CMD+=(timeout -k 60 "$EVAL_TIMEOUT"
    "$PY" "$CODE/scripts/track2/eval_all.py"
    --arm-dir "$work" --arm C3r2_psi --export yes
    --iteration "$it" --episodes "$ep"
    --gpus "$GPU" --jobs-per-gpu 1 --world-count 1 --export-jobs 1
    --export-tolerance "$EXPORT_TOLERANCE"
    --skip-pack
    --kit-root "$KIT" --stock-runtime "$STOCK" --dataset-root "$DATASET" --manifest "$MANIFEST"
    --results "$work/results" --code "$CODE" --python "$PY")
}

LAST_RESULT=""
evaluate() {
  local ep=$1 it=$2 attempt=$3
  local epd work src log t0 code out
  epd=$(printf 'ep%02d' "$ep")
  work=$OUT/$epd/it$it
  src=$RUNS/$epd/model_$it.safetensors
  log=$work/eval_all.attempt$attempt.log
  mkdir -p "$work/$epd"
  ln -sfn "$src" "$work/$epd/model_$it.safetensors"   # the view: eval_all only reads through this link
  rm -f "$work/eval_report.json"                       # never parse a stale report of an earlier attempt
  build_cmd "$ep" "$it" "$work"
  log "$epd model_$it: attempt $attempt on GPU $GPU -> $work"
  t0=$SECONDS
  {
    printf '[%s] attempt %s:' "$(ts)" "$attempt"; printf ' %q' "${CMD[@]}"; printf '\n'
    (cd "$CODE" && "${CMD[@]}")
  } </dev/null >"$log" 2>&1
  code=$?
  out=$("$PY" -B "$HELPER" record --runs "$RUNS" --out "$OUT" --work "$work" --manifest "$MANIFEST" --kit "$KIT" \
        --episode "$ep" --iteration "$it" --attempt "$attempt" --exit "$code" --seconds $((SECONDS - t0)) \
        --log "$log" 2>&1)
  local res
  res=$(grep -m1 '^RESULT ' <<<"$out" || true)
  if [[ -n $res ]]; then
    LAST_RESULT=$(awk '{print $2}' <<<"$res")
    log "${res#RESULT * }"
  else
    LAST_RESULT=helper_error
    log "helper record failed (exit $code of eval_all; log $log): $out"
  fi
}

# Tier 2 leaderboard metrics: the kit packer against the clean reference, for this one episode (partial config).
clean_score() {
  local ep=$1 it=$2 work=$3 cfg out res fr=()
  [[ " $FRAME_REFS_EPISODES " == *" $ep "* ]] && fr=(--frame-references "$FRAME_REFS")
  cfg=$work/tracks_partial.json; out=$work/tier2_clean_scores.csv
  res=$work/results/$(printf 'episode_%06d.parquet' "$ep")
  [[ -f $res ]] || { log "clean score: no $res"; return; }
  "$KPY" -I -B -c "import json,sys; c=json.load(open(sys.argv[1])); c['tracks']['track_2_tier2']['scored_episodes']=[int(sys.argv[3])]; open(sys.argv[2],'w').write(json.dumps(c,indent=2))" \
    "$KIT/config/tracks.json" "$cfg" "$ep"
  (cd "$KIT" && PYTHONDONTWRITEBYTECODE=1 nice -n 10 "$KPY" -I -B "$HERE/track2_tier2_clean_scores.py" --kit "$KIT" --out "$out" -- \
    --track track_2_tier2 --results "$work/results" --dataset-root "$DATASET" --code-commit "$COMMIT" \
    --eval-script "$KIT/flash_chord/scripts/evaluate_policy.py" --config "$cfg" "${fr[@]}" -o "$work/tier2_LOCAL_NOT_FOR_UPLOAD.csv") \
    >"$work/clean_score.log" 2>&1
  "$KPY" -I -B "$HERE/clean_row.py" "$out" "$ep" "$it" "$OUT/tier2_clean.csv" 2>&1 |
    while IFS= read -r line; do log "ep$ep model_$it clean (Tier 2): $line"; done
}

nap() {  # sleep $1 s, waking early if the STOP file appears
  local left=$1
  while (( left > 0 )); do
    [[ -e $STOP ]] && return
    sleep $(( left < 10 ? left : 10 ))
    left=$(( left - 10 ))
  done
}

scan() {
  "$PY" -B "$HELPER" todo --runs "$RUNS" --out "$OUT" --manifest "$MANIFEST" --episodes "$EPISODES" \
    --every "$EVERY" --min-age "$MIN_AGE" --retry-delay "$RETRY_DELAY" 2>&1
}

final_table() {
  log "ok rows of $CSV (right: $BASELINE_ARM from $BASELINE at the same iteration):"
  "$PY" -B "$HELPER" table --out "$OUT" --baseline "$BASELINE" --baseline-arm "$BASELINE_ARM" 2>&1 |
    while IFS= read -r line; do log "  $line"; done
}

# ------------------------------------------------------------------------------------------------------ main
preflight

if [[ -n ${DRY_RUN:-} ]]; then
  log "DRY_RUN: preflight ok; GPU $GPU; scan of $RUNS:"
  result=$(scan); echo "$result"
  item=$(grep -m1 '^ITEM ' <<<"$result" || true)
  read -r _ ep it _ <<<"${item:-ITEM 19 2000 1}"
  work=$OUT/$(printf 'ep%02d' "$ep")/it$it
  build_cmd "$ep" "$it" "$work"
  log "DRY_RUN: command for ep$ep model_$it (run in $CODE; nothing was written):"
  printf ' %q' "${CMD[@]}"; printf '\n'
  exit 0
fi

mkdir -p "$OUT"
exec 9>"$OUT/.eval_watch.lock"
# Children inherit fd 9, so an eval left running by a killed watcher also keeps the lock until it exits.
flock -n 9 || die "$OUT/.eval_watch.lock is held (another eval_watch.sh, or an eval left by a killed one)"

log "eval watcher start: pid $$, GPU $GPU, episodes [$EPISODES], every $EVERY + final, poll ${POLL}s"
log "runs $RUNS; out $OUT; csv $CSV; stop file $STOP"
log "code $CODE @ $(git -C "$CODE" rev-parse --short HEAD 2>/dev/null || echo '?'); stock $STOCK; kit $KIT"
log "python $PY; export tolerance $EXPORT_TOLERANCE; timeout ${EVAL_TIMEOUT}s"

last_status=""
fail_streak=0
while :; do
  if [[ -e $STOP ]]; then log "stop file $STOP exists: exiting"; break; fi
  result=$(scan)
  rc=$?
  if (( rc != 0 )) || ! grep -q '^SUMMARY ' <<<"$result"; then
    log "scan failed (exit $rc): $result"
    (( rc == 2 )) && die "fix the episode list"
    nap "$POLL"; continue
  fi
  status=$(grep -v '^ITEM ' <<<"$result")
  if [[ $status != "$last_status" ]]; then
    while IFS= read -r line; do log "$line"; done <<<"$status"
    last_status=$status
  fi
  item=$(grep -m1 '^ITEM ' <<<"$result" || true)
  if [[ -n $item ]]; then
    read -r _ ep it attempt <<<"$item"
    evaluate "$ep" "$it" "$attempt"
    [[ $LAST_RESULT == ok ]] && clean_score "$ep" "$it" "$OUT/$(printf 'ep%02d' "$ep")/it$it"
    if [[ $LAST_RESULT == ok || $LAST_RESULT == seq_mismatch ]]; then
      fail_streak=0
    else
      fail_streak=$((fail_streak + 1))
      if (( fail_streak >= 3 )); then
        log "$fail_streak consecutive failures (GPU memory, env, disk?): pausing ${FAIL_PAUSE}s"
        nap "$FAIL_PAUSE"
        fail_streak=0
      fi
    fi
    continue   # rescan at once: new checkpoints may be ready, and the STOP file is checked
  fi
  if grep -q '^SUMMARY complete=1' <<<"$result"; then
    log "every run has final.json and every scheduled checkpoint is finished: exiting"
    break
  fi
  if [[ -n ${ONCE:-} ]]; then log "ONCE: nothing ready now, exiting"; break; fi
  nap "$POLL"
done
final_table
log "eval watcher end"
