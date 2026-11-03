#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# selftest.sh: syntax check and --dry-run of every scripts/final/*.sh (no command of a plan is executed; nothing is
# written except the dry-run outputs under --out). Exit 0 iff every script passes.
#   selftest.sh [--site FILE] [--out DIR]       (default out: $V2D_WORK/selftest_<time>)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
usage() { sed -n '3,6p' "$0"; }
final_init "$@"
OUT=""
set -- ${REST[@]+"${REST[@]}"}
while [ $# -gt 0 ]; do case "$1" in --out) OUT=$2; shift ;; *) die "unknown option $1" ;; esac; shift; done
: "${OUT:=$V2D_WORK/selftest_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT"
site=(); [ -n "$SITE" ] && site=(--site "$SITE")
fail=0; n=0
check() {  # check NAME ARGS...
  local name=$1; shift; n=$((n + 1))
  local log="$OUT/${name}.txt"
  if bash "$FINAL_DIR/$name.sh" --dry-run ${site[@]+"${site[@]}"} "$@" > "$log" 2>&1; then
    echo "PASS  $name $*  ($(wc -l < "$log") lines)"
  else
    echo "FAIL  $name $*  (see $log)"; tail -n 5 "$log"; fail=$((fail + 1))
  fi
}
for f in "$FINAL_DIR"/*.sh "$FINAL_LIB"/*.sh; do bash -n "$f" || { echo "SYNTAX $f"; fail=$((fail + 1)); }; done
"$PLAN_PY" -I -B -c 'import ast, sys; [ast.parse(open(f).read(), f) for f in sys.argv[1:]]' "$FINAL_LIB"/*.py \
  && echo "PASS  python syntax (lib/*.py)" || fail=$((fail + 1))
check setup_env
check get_data
check build_inputs_tier2
check build_inputs_tier3
for t in 1 2 3; do
  check eval_tier$t --episode all --trials 2
  check eval_tier$t --episode all --reexport
done
check train_tier1 --episode 0-29
check train_tier2 --episode 0-29
check train_tier3 --episode 0-29
check select_tier2
check select_tier3
for t in 1 2 3; do check pack_tier$t --commit-url "$GITHUB_REPO/commit/0000000000000000000000000000000000000000"; done
echo "selftest: $n dry runs, $fail failure(s); outputs in $OUT"
[ "$fail" = 0 ]
