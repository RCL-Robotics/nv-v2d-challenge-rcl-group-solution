#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Re-run the official evaluator on the SAME exported checkpoints, to measure the evaluator's own run-to-run spread
# (it is not deterministic: for C3r2 Tier 1 four runs of the same 30 files gave AUC 0.7939, 0.7948, 0.8202, 0.7957).
# Diagnostic only: the submitted file is the single final-evaluation run, never a best-of selection.
#
# usage: repeat_eval.sh [--line FILE] [--eval-root DIR] [--repeats 3] [--gpu 0] [--checkpoints FILE] [--dry-run]
#   reads <eval root>/<ARM>/eval_report.json (written by final_eval.sh) for the evaluated checkpoint of each episode,
#   writes <eval root>/<ARM>_repeats/{checkpoints.json,repK/,repK.log,status.txt}
#   --checkpoints  a JSON map {"<episode index>": "<absolute path of the exported .safetensors>", ...} instead of
#                  eval_report.json, e.g. to evaluate the hosted exported files directly (--repeats 1, then
#                  pack.sh --results <...>/rep1)
#   env: V2D_WORK V2D_RUNTIME V2D_KIT FC_PYTHON  SKIP_PATH_CHECK=1 (skip check_task_paths.py)
#        SKIP_INPUT_CHECK=1 (skip check_eval_inputs.py: runtime commit, evaluator digest, kit manifest)
# Relative paths are taken from the caller's working directory (the runner itself runs inside the kit).
# Each repeat runs the kit's runner directly (cwd = kit, PYTHONPATH = stock runtime src, no bytecode writes); as in
# the original, the repeats run concurrently on one GPU, 5 s apart.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
abs(){ case $1 in /*|"") printf '%s' "$1";; *) printf '%s' "$PWD/$1";; esac; }   # relative to the caller's cwd
LINE=$HERE/lines/c3r2_tier1.env; OUTROOT=""; N=3; GPU=0; DRY=0; GIVEN=""
while [ $# -gt 0 ]; do case "$1" in
  --line) LINE=$2; shift 2;; --eval-root) OUTROOT=$2; shift 2;; --repeats) N=$2; shift 2;;
  --gpu) GPU=$2; shift 2;; --checkpoints) GIVEN=$2; shift 2;; --dry-run) DRY=1; shift;;
  *) echo "unknown argument $1" >&2; exit 2;; esac; done
# shellcheck disable=SC1090
source "$LINE"
OUTROOT=${OUTROOT:-${V2D_WORK:?set V2D_WORK or pass --eval-root}/eval}
: "${V2D_RUNTIME:?set V2D_RUNTIME}" "${V2D_KIT:?set V2D_KIT}"
OUTROOT=$(abs "$OUTROOT"); GIVEN=$(abs "$GIVEN"); V2D_RUNTIME=$(abs "$V2D_RUNTIME"); V2D_KIT=$(abs "$V2D_KIT")
source "$HERE/env/fc_env.sh"
if [ "${SKIP_INPUT_CHECK:-0}" != 1 ]; then
  CHECK=(python -B "$HERE/check_eval_inputs.py" --line "$LINE" --kit "$V2D_KIT" --runtime "$V2D_RUNTIME")
  if [ $DRY = 1 ]; then printf '[repeats] '; printf '%q ' "${CHECK[@]}"; echo
  else "${CHECK[@]}" || { echo "the kit or the stock runtime does not match the line spec (SKIP_INPUT_CHECK=1 skips)" >&2; exit 1; }; fi
fi
REP=$OUTROOT/${ARM}_repeats; MAP=$REP/checkpoints.json; REPORT=$OUTROOT/$ARM/eval_report.json
if [ -n "$GIVEN" ]; then
  MAP=$GIVEN; echo "[repeats] # checkpoint map $MAP (given)"
elif [ $DRY = 0 ]; then
  mkdir -p "$REP"
  python - "$REPORT" "$MAP" <<'PY' || exit 1
import json, sys
report = json.load(open(sys.argv[1]))
ckpts = {str(int(e["episode"])): e["evaluated_checkpoint"] for e in report["episodes"] if e.get("evaluated_checkpoint")}
json.dump(ckpts, open(sys.argv[2], "w"), indent=1)
print(f"{len(ckpts)} checkpoints -> {sys.argv[2]}")
PY
else
  echo "[repeats] # checkpoint map $MAP <- evaluated_checkpoint of every episode in $REPORT"
fi
if [ $DRY = 0 ] && [ "${SKIP_PATH_CHECK:-0}" != 1 ]; then   # the stored task.parquet paths must exist (README)
  mapfile -t FILES < <(python -c 'import json, sys; print("\n".join(json.load(open(sys.argv[1])).values()))' "$MAP")
  python "$HERE/check_task_paths.py" "${FILES[@]}" || exit 1
fi
for k in $(seq 1 "$N"); do
  cmd=(env -u PYTHONHOME -u PYTHONSTARTUP "PYTHONPATH=$V2D_RUNTIME/src" PYTHONDONTWRITEBYTECODE=1
       XLA_PYTHON_CLIENT_PREALLOCATE=false "CUDA_VISIBLE_DEVICES=$GPU" python examples/run_policy_evaluation.py
       --eval-script "$V2D_KIT/flash_chord/scripts/evaluate_policy.py" --runtime-root "$V2D_RUNTIME"
       --world-count "$EVAL_WORLD_COUNT" --checkpoints "$MAP" --results "$REP/rep$k")
  if [ $DRY = 1 ]; then printf '[repeats] cd %q && ' "$V2D_KIT"; printf '%q ' "${cmd[@]}"; echo; continue; fi
  mkdir -p "$REP"; ( cd "$V2D_KIT" && "${cmd[@]}" > "$REP/rep$k.log" 2>&1; echo "rep$k exit $?" >> "$REP/status.txt" ) &
  sleep 5
done
[ $DRY = 1 ] && exit 0
wait; echo ALL_DONE >> "$REP/status.txt"
[ -f "$REPORT" ] || exit 0   # the summary table needs final_eval.sh's eval_report.json
for k in $(seq 1 "$N"); do
  python "$HERE/summarize.py" --arm-dir "$OUTROOT/$ARM" --results "$REP/rep$k" --out "$REP/rep$k.csv" --kit "$V2D_KIT" | tail -2
done
