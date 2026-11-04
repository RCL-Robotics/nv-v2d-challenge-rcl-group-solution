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
check() {  # check NAME ARGS...: one log per call, $OUT/NN_<name>[_<args>].txt (never overwritten by a later call)
  local name=$1; shift; n=$((n + 1))
  local slug; slug=$(printf '%s_' "$@" | tr -c 'A-Za-z0-9_.' '_' | tr -s '_' | cut -c1-60); slug=${slug%_}; slug=${slug#_}
  local log; log="$OUT/$(printf '%02d' "$n")_${name}${slug:+_$slug}.txt"
  [ -e "$log" ] && die "selftest log $log exists already"
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
check setup_env --steps check --report-only
check get_data
check build_inputs_tier2
check build_inputs_tier3
for t in 1 2 3; do
  check eval_tier$t --episode all --trials 2
  check eval_tier$t --episode all --reexport
done
check train_tier1 --episode 0-29
check train_tier1 --episode 0-29 --full-length
check train_tier1 --episode 0-29 --check-sources
check train_tier1 --episode 0 --iterations 30
check train_tier1 --episode 4 --stage 2 --iterations 30
check train_tier2 --episode 0-29
check train_tier3 --episode 0-29
check select_tier2
check select_tier3
for t in 1 2 3; do check pack_tier$t --commit-url "$GITHUB_REPO/commit/0000000000000000000000000000000000000000"; done
# the Tier-2 command printer (reads only set_items_r9.json)
n=$((n + 1)); log="$OUT/$(printf '%02d' "$n")_print_commands_r9.txt"
if "$PLAN_PY" -I -B "$FINAL_LIB/print_commands_r9.py" > "$log" 2>&1; then echo "PASS  print_commands_r9.py  ($(wc -l < "$log") lines)"
else echo "FAIL  print_commands_r9.py  (see $log)"; tail -n 5 "$log"; fail=$((fail + 1)); fi
echo "selftest: $n dry runs, $fail failure(s); outputs in $OUT"
[ "$fail" = 0 ]
