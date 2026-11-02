#!/usr/bin/env bash
# Run the G2_surface_only pipeline on all 30 Tier-2 episodes (6 in parallel, 2 BLAS threads each, nice 10), then
# collect <out>/proxies.csv. usage: run_all.sh <out dir> [--opt key=value ...]   (extra args go to run_episode.py)
set -uo pipefail
OUT=$(realpath -m "$1"); shift
HERE=$(cd "$(dirname "$0")" && pwd)
PY=/root/autodl-tmp/zuxing.lu/envs/kit/bin/python
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$OUT/logs"
echo "run_all start $(date '+%F %T %z') out $OUT extra: $*"
seq 0 29 | xargs -P 6 -I{} sh -c "nice -n 10 $PY -I -B $HERE/run_episode.py {} --out $OUT $* > $OUT/logs/ep\$(printf %02d {}).log 2>&1; echo \"ep{} exit \$? \$(date +%T)\""
$PY -I -B "$HERE/make_csv.py" "$OUT"
echo "run_all end $(date '+%F %T %z')"
