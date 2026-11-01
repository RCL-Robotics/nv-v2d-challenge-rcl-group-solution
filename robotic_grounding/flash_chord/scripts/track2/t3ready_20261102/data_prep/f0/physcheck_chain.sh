#!/bin/bash
# t3f0 prep: physical checks of the final 14 support files (runs after the earlier dev pass has exited; never signals)
ME=/root/autodl-tmp/haoran.zhang; A=$ME/tmp_analysis/tier3/f0; FC="$ME/conda_envs/flashchord/bin/python -I -B"
EPS="1 2 3 6 7 9 11 13 16 17 20 25 27 28"
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4
cd /tmp
while pgrep -f "scripts/t3f0_physcheck.py" > /dev/null; do sleep 20; done
mkdir -p $A/physcheck_old && mv $A/physcheck/* $A/physcheck_old/ 2>/dev/null
echo "[$(date '+%F %T')] start" > $A/physcheck/chain.log
( nice -n 10 taskset -c 88-95 $FC $ME/scripts/t3f0_physcheck.py rest $EPS > $A/physcheck/rest.log 2>&1
  nice -n 10 taskset -c 88-95 $FC $ME/scripts/t3f0_physcheck.py settle 1 2 3 6 7 9 11 > $A/physcheck/settle_a.log 2>&1 ) &
( while [ ! -f $A/physcheck/rest_ep28.json ]; do sleep 10; done
  nice -n 10 taskset -c 72-79 $FC $ME/scripts/t3f0_physcheck.py settle 13 16 17 20 25 27 28 > $A/physcheck/settle_b.log 2>&1 ) &
( nice -n 10 taskset -c 96-103 $FC $ME/scripts/t3f0_physcheck.py human $EPS > $A/physcheck/human.log 2>&1 ) &
( while ! grep -q "t3f0_retarget_all done" $A/retarget/run.log; do sleep 30; done
  nice -n 10 taskset -c 64-71 $FC $ME/scripts/t3f0_physcheck.py g1 1 2 3 6 7 9 11 > $A/physcheck/g1_a.log 2>&1 ) &
( while ! grep -q "t3f0_retarget_all done" $A/retarget/run.log; do sleep 30; done
  nice -n 10 taskset -c 56-63 $FC $ME/scripts/t3f0_physcheck.py g1 13 16 17 20 25 27 28 > $A/physcheck/g1_b.log 2>&1 ) &
wait
$FC $ME/scripts/t3f0_physcheck.py summary > $A/physcheck/summary.log 2>&1
echo "[$(date '+%F %T')] done" >> $A/physcheck/chain.log
