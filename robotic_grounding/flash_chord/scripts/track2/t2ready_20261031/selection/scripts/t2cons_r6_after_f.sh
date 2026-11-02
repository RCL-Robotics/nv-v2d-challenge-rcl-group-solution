#!/usr/bin/env bash
# t2cons_r6_after_f.sh: round 6 (PREREG_R6 5.5-6, PREREG_R6_PINS_c 1.3). Waits for the phase-F driver to finish (polls its
# heartbeat every 60 s; never signals anything), then, only if the heartbeat state is "done", runs in order and stops at
# the first non-zero exit: t2cons_r6_consensus.py check / gate / score / select --phase F, then t2cons_r6_pack.py check /
# prepare / pack / validate. CPU only, nice 10. Log: ROOT/after_f_stdout.log (the caller redirects).
set -u
ME=/root/autodl-tmp/haoran.zhang
ROOT=$ME/fc_runs/submissions/t2_consensus_r6_20261031
PY=$ME/conda_envs/v2dkit/bin/python
HB=$ROOT/eval_r6_heartbeat_f6.json
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 TZ=Asia/Shanghai
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP
cd /tmp || exit 1
echo "=== waiting for the phase-F driver $(date '+%F %T')"
while :; do
  if [ -f "$HB" ] && ! grep -q '"state": "running"' "$HB" && [ ! -e "$ROOT/driver.pid" ]; then break; fi
  sleep 60
done
state=$(grep -o '"state": "[^"]*"' "$HB")
echo "=== phase-F driver finished: $state $(date '+%F %T')"
case "$state" in *'"done"'*) ;; *) echo "=== STOP: phase-F state is not done"; exit 1 ;; esac
for c in check gate score select; do
  echo "=== scorer $c --phase F $(date '+%F %T')"
  nice -n 10 "$PY" -I -B "$ME/scripts/t2cons_r6_consensus.py" "$c" --phase F || { echo "=== STOP: scorer $c rc $?"; exit 1; }
done
for c in check prepare pack validate; do
  echo "=== pack $c $(date '+%F %T')"
  nice -n 10 "$PY" -I -B "$ME/scripts/t2cons_r6_pack.py" "$c" || { echo "=== STOP: pack $c rc $?"; exit 1; }
done
echo "=== ALL DONE $(date '+%F %T')"
