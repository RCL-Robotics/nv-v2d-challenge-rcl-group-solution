#!/bin/bash
# keeps every t3f0 CPU-only check/build process on NUMA-0 cores 10-31 (+HT 114-135), which no trainer uses; sched affinity
# only (taskset -a -p), never a signal. Exits when the physcheck chain has finished.
A=/root/autodl-tmp/haoran.zhang/tmp_analysis/tier3/f0
while ! grep -q done $A/physcheck/chain.log 2>/dev/null; do
  for p in $(pgrep -f "scripts/t3f0_(physcheck|supports|frames)\.py"); do
    cur=$(taskset -cp $p 2>/dev/null | awk -F': ' '{print $2}')
    [ "$cur" = "10-31,114-135" ] || taskset -a -cp 10-31,114-135 $p > /dev/null 2>&1
  done
  sleep 3
done
