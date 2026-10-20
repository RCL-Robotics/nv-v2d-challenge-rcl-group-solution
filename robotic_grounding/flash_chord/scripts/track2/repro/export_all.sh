#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Export every episode of an evaluation view directory at ITS OWN final iteration (<run>/final.json, read by
# queue_helper.py final; without one, model_<MAX_ITERATIONS>) into the stock ActorCritic format, on the CPU, with an
# explicit equivalence tolerance, so eval_all.py --checkpoint-from-final reuses these "ok" exports.
# Why a tolerance: float32 rounding in the exact embedding grows to 1-8e-5 after long training, so the exporter's
# default 1e-5 refuses nearly every export. Tier 1 used 1e-4 (all 30 accepted, max_rel_diff 8.1e-6 .. 6.47e-5).
#
# usage: export_all.sh <view dir> [--tolerance 1e-4] [--jobs 4] [--episodes "00 01 ..."] [--dry-run]
#   <view dir>/epNN -> the episode's run directory (final_eval.sh builds it); writes <view dir>/export/epNN/
#   --episodes  only these (two-digit) episodes; default every <view dir>/epNN
#   env: V2D_RUNTIME (stock runtime .../robotic_grounding/flash_chord)  FC_PYTHON  MAX_ITERATIONS=45000
# Per episode (cwd robotic_grounding/flash_chord, PYTHONPATH=<that>/src, JAX_PLATFORMS=cpu, no GPU):
#   python -m flash_chord.training.ppo.coordinate.export --input <run>/model_N.safetensors \
#     --output <view>/export/epNN/model_N.safetensors --stock-runtime $V2D_RUNTIME \
#     --report <view>/export/epNN/export_report.json --tolerance <tol>
# Exits non-zero when an export fails; an episode without its checkpoint is reported and skipped.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); FC=$(cd "$HERE/../../.." && pwd)
abs(){ case $1 in /*|"") printf '%s' "$1";; *) printf '%s' "$PWD/$1";; esac; }   # relative to the caller's cwd
VIEW=${1:?usage: export_all.sh <view dir> [--tolerance T] [--jobs N] [--dry-run]}; shift
TOL=1e-4; JOBS=4; DRY=0; EPS=""
while [ $# -gt 0 ]; do case "$1" in
  --tolerance) TOL=$2; shift 2;; --jobs) JOBS=$2; shift 2;; --episodes) EPS=$2; shift 2;; --dry-run) DRY=1; shift;;
  *) echo "unknown argument $1" >&2; exit 2;; esac; done
: "${V2D_RUNTIME:?set V2D_RUNTIME (stock runtime robotic_grounding/flash_chord)}"
VIEW=$(abs "$VIEW"); V2D_RUNTIME=$(abs "$V2D_RUNTIME")
ITERS=${MAX_ITERATIONS:-45000}
source "$HERE/env/fc_env.sh"
export PYTHONPATH="$FC/src" JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
one(){ # run dir (a <view>/epNN link)
  local d=$1 ep it out src cmd
  ep=$(basename "$d"); it=$(python "$HERE/queue_helper.py" final "$d" 2>/dev/null) || it=$ITERS
  if [ "$DRY" = 1 ] && [ ! -f "$d/final.json" ]; then it="<N of $ep/final.json>"; fi
  src=$d/model_$it.safetensors; out=$VIEW/export/$ep
  cmd=(python -m flash_chord.training.ppo.coordinate.export --input "$src" --output "$out/model_$it.safetensors"
       --stock-runtime "$V2D_RUNTIME" --report "$out/export_report.json" --tolerance "$TOL")
  if [ "$DRY" = 1 ]; then printf '[export %s] cd %q && ' "$ep" "$FC"; printf '%q ' "${cmd[@]}"; echo; return 0; fi
  [ -f "$src" ] || { echo "$ep: model_$it missing - not exported"; return 0; }
  mkdir -p "$out"
  if [ -f "$out/model_$it.safetensors" ] && python - "$out/export_report.json" "$src" <<'PY'
import hashlib, json, sys
report, source = json.load(open(sys.argv[1])), sys.argv[2]
digest = hashlib.sha256(open(source, "rb").read()).hexdigest()
sys.exit(0 if report.get("status") == "ok" and report.get("source_sha256") == digest else 1)
PY
  then echo "$ep: model_$it already exported (ok)"; return 0; fi
  local rc; ( cd "$FC" && nice -n 10 "${cmd[@]}" ) > "$out/export.log" 2>&1; rc=$?
  echo "$ep: model_$it exit $rc $(grep -o '"max_rel_diff": [0-9.e-]*' "$out/export_report.json" 2>/dev/null | head -1)"
  [ $rc = 0 ] || return 1
}
export -f one; export HERE FC VIEW ITERS TOL DRY V2D_RUNTIME
if [ -n "$EPS" ]; then DIRS=(); for e in $EPS; do DIRS+=("$VIEW/ep$e"); done
else DIRS=("$VIEW"/ep[0-9][0-9]); [ -e "${DIRS[0]}" ] || { echo "no epNN entries in $VIEW" >&2; exit 1; }; fi
if [ "$DRY" = 1 ]; then for d in "${DIRS[@]}"; do one "$d"; done; exit 0; fi
printf '%s\n' "${DIRS[@]}" | xargs -P "$JOBS" -I{} bash -c 'one "$1"' _ {} \
  || { echo "at least one export failed (see <view>/export/epNN/export.log)" >&2; exit 1; }
