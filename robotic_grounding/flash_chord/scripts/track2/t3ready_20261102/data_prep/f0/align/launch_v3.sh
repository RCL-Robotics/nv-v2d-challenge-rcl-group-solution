#!/bin/bash
# launch_v3.sh: CPU-only GT-free alignment run (t3f0_align.py v3), 3 workers on free NUMA-0 cores, nice 10
ME=/root/autodl-tmp/haoran.zhang; A=$ME/tmp_analysis/tier3/f0/align
cd $A
rm -f $A/ep*.json $A/summary.json $A/summary.tsv
run() { CPUS=$1; shift; LOG=$1; shift
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 nice -n 10 taskset -c $CPUS \
    timeout 3600 $ME/conda_envs/v2d/bin/python -B $ME/scripts/t3f0_align.py --dump "$@" > $A/$LOG 2>&1 < /dev/null & }
run 10-17 v3_a.log 0 1 2 3 4 5 6 7 8 9
run 18-25 v3_b.log 10 11 12 13 14 15 16 17 18 19
run 26-31,114-115 v3_c.log 20 21 22 23 24 25 26 27 28 29
wait
echo "ALL DONE $(date '+%F %T')" > $A/v3_done.txt
