#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# CPU-only checks of the t1best tools and configs. Runs nothing on a GPU and writes only to a temporary directory.
#   1. bash -n on the shell scripts; byte-compile every Python file (into the temp dir)
#   2. check_items.py: configs schema, set, digests, placeholders, manifest and stages tables
#   3. ../repro/tests/check_hygiene.sh on every file of this folder (no machine paths, credentials, LFS, binaries)
#   4. with --compose: Hydra composition of every pinned stage and of a plain fork with this tree's configs
#      (needs the flashchord python: --python); with --records DIR also against the runs' saved Hydra configs
# usage: tests/run_checks.sh [--python PY] [--compose] [--records DIR]
#   HYGIENE_EXTRA_PATTERNS: further forbidden strings for check_hygiene.sh, given at check time.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); T1=$(cd "$HERE/.." && pwd)
PY=$(command -v python3 || command -v python); COMPOSE=0; RECORDS=""
while [ $# -gt 0 ]; do case "$1" in
  --python) PY=$2; shift 2;; --compose) COMPOSE=1; shift;; --records) RECORDS=$2; shift 2;;
  *) echo "unknown argument $1" >&2; exit 2;; esac; done
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
fail=0; check(){ local name=$1; shift; echo "--- $name"; if "$@"; then echo "PASS $name"; else echo "FAIL $name"; fail=1; fi; }
syntax(){
  local f rc=0 n=0
  for f in "$T1"/tests/*.sh; do bash -n "$f" || { echo "syntax error: $f"; rc=1; }; n=$((n + 1)); done
  echo "bash -n: $n scripts"; n=0
  for f in "$T1"/*.py; do
    "$PY" -c 'import py_compile, sys; py_compile.compile(sys.argv[1], cfile=sys.argv[2], doraise=True)' "$f" "$TMP/$(basename "$f")c" \
      || { echo "compile error: $f"; rc=1; }; n=$((n + 1)); done
  echo "py_compile: $n files"; return $rc
}
hygiene(){ mapfile -t FILES < <(cd "$T1" && git ls-files --cached --others --exclude-standard -- . | sed "s#^#$T1/#"); bash "$T1/../repro/tests/check_hygiene.sh" "${FILES[@]}"; }
check "syntax" syntax
check "configs" "$PY" -B "$T1/check_items.py"
check "hygiene" hygiene
if [ $COMPOSE = 1 ]; then
  check "compose" "$PY" -B "$T1/check_items.py" --compose --forks ${RECORDS:+--records "$RECORDS"}
fi
echo; [ $fail = 0 ] && echo "ALL CHECKS PASSED" || echo "SOME CHECKS FAILED"
exit $fail
