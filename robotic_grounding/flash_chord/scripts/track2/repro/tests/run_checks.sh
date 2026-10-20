#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# CPU-only checks of the reproduction tooling. Runs nothing on a GPU and writes only to a temporary directory.
#   1. bash -n on every shell script of the tooling; byte-compile every Python file (into the temp dir)
#   2. tests/check_code_unchanged.sh   (training/export/eval code identical to the line's CODE_COMMIT)
#   3. tests/check_hygiene.sh          (no machine paths, names, mirrors, credentials, LFS or binary files)
#   4. with --motion-root: the support rule on the converted data (tests/check_support_rule.py)
#   5. with --records too: train_all.sh --dry-run vs the recorded W&B args and queue log (tests/check_train_commands.py)
# usage: tests/run_checks.sh [--motion-root DIR] [--records DIR] [--queue-log FILE] [--python PY]
#   --records    the original arm run directory (epNN/wandb/run-*/files/wandb-metadata.json)
#   with --records, set WANDB_MODE / WANDB_ENTITY / WANDB_PROJECT to the recorded values for an exact comparison;
#   otherwise the W&B destination keys are left out of the comparison.
#   HYGIENE_EXTRA_PATTERNS: further forbidden strings for tests/check_hygiene.sh (names, hosts), given at check time.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); REPRO=$(cd "$HERE/.." && pwd)
abs(){ case $1 in /*|"") printf '%s' "$1";; *) printf '%s' "$PWD/$1";; esac; }
RG=$(cd "$REPRO/../../../.." && pwd)   # robotic_grounding/
MOTION=""; RECORDS=""; QLOG=""; PY=$(command -v python3 || command -v python)
while [ $# -gt 0 ]; do case "$1" in
  --motion-root) MOTION=$2; shift 2;; --records) RECORDS=$2; shift 2;; --queue-log) QLOG=$2; shift 2;;
  --python) PY=$2; shift 2;; *) echo "unknown argument $1" >&2; exit 2;; esac; done
MOTION=$(abs "$MOTION"); RECORDS=$(abs "$RECORDS"); QLOG=$(abs "$QLOG")
export FC_PYTHON=${FC_PYTHON:-$PY}   # train_all.sh --dry-run puts this interpreter first on PATH
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
fail=0; check(){ local name=$1; shift; echo "--- $name"; if "$@"; then echo "PASS $name"; else echo "FAIL $name"; fail=1; fi; }
syntax(){
  local f rc=0 n=0
  for f in "$REPRO"/*.sh "$REPRO"/env/*.sh "$REPRO"/tests/*.sh "$RG"/scripts/track2/*.sh; do
    bash -n "$f" || { echo "syntax error: $f"; rc=1; }; n=$((n + 1)); done
  echo "bash -n: $n scripts"
  n=0
  for f in "$REPRO"/*.py "$REPRO"/tests/*.py "$RG"/scripts/track2/*.py; do
    "$PY" -c 'import py_compile, sys; py_compile.compile(sys.argv[1], cfile=sys.argv[2], doraise=True)' "$f" "$TMP/$(basename "$f")c" \
      || { echo "compile error: $f"; rc=1; }; n=$((n + 1)); done
  echo "py_compile: $n files"
  return $rc
}
check "syntax" syntax
check "code unchanged" bash "$HERE/check_code_unchanged.sh"
check "hygiene" bash "$HERE/check_hygiene.sh"
if [ -n "$MOTION" ]; then
  check "support rule" "$PY" "$HERE/check_support_rule.py" --motion-root "$MOTION" ${RECORDS:+--records "$RECORDS"}
  if [ -n "$RECORDS" ]; then
    mode=ignore; [ -n "${WANDB_ENTITY:-}" ] && mode=exact
    V2D_MOTION=$MOTION V2D_WORK=$TMP bash "$REPRO/train_all.sh" --runs "$TMP/runs" --dry-run > "$TMP/train_dry_run.txt" 2>&1
    check "training commands" "$PY" "$HERE/check_train_commands.py" --dry-run "$TMP/train_dry_run.txt" \
      --records "$RECORDS" --motion-root "$MOTION" --runs-root "$TMP/runs" --logging "$mode" ${QLOG:+--queue-log "$QLOG"}
  fi
fi
echo; [ $fail = 0 ] && echo "ALL CHECKS PASSED" || echo "SOME CHECKS FAILED"
exit $fail
