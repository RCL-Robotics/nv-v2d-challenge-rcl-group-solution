#!/usr/bin/env bash
# t2cons_r7_after_s.sh: round 7 (PREREG_R7 5-6, PREREG_R7_PINS_c 1). Waits for the phase-S driver to finish (polls its
# heartbeat every 60 s; never signals anything); only if the heartbeat state is "done": t2cons_r7_consensus.py check /
# gate / score / select --phase S (SELECTION_S7.sha256 last); then starts phase F with the pinned launcher
# (scripts/t2cons_r7_eval_launch.sh --phase final: GPUs 6 and 7, plain, setsid nohup); waits for the phase-F driver the same
# way; only if "done": check / gate / score / select --phase F (SELECTION_R7.sha256 last), then t2cons_r7_pack.py check /
# prepare / pack / validate (placeholder pack). Stops at the first non-zero exit. CPU only (scorer / packer), nice 10.
# Decides nothing that the pinned scripts do not. Log: ROOT/after_s_stdout.log (the caller redirects).
set -u
ME=/root/autodl-tmp/haoran.zhang
ROOT=$ME/fc_runs/submissions/t2_consensus_r7_20261102
PY=$ME/conda_envs/v2dkit/bin/python
SC=$ME/scripts/t2cons_r7_consensus.py
PK=$ME/scripts/t2cons_r7_pack.py
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 TZ=Asia/Shanghai
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP
cd /tmp || exit 1

wait_done() {  # $1 heartbeat file
  local hb=$1 state
  echo "=== waiting for $(basename "$hb") $(date '+%F %T')"
  while :; do
    if [ -f "$hb" ] && ! grep -q '"state": "running"' "$hb" && [ ! -e "$ROOT/driver.pid" ]; then break; fi
    sleep 60
  done
  state=$(grep -o '"state": "[^"]*"' "$hb")
  echo "=== driver finished: $state $(date '+%F %T')"
  case "$state" in *'"done"'*) return 0 ;; *) echo "=== STOP: state is not done"; return 1 ;; esac
}

score_phase() {  # $1 S|F
  local c
  for c in check gate score select; do
    echo "=== scorer $c --phase $1 $(date '+%F %T')"
    CUDA_VISIBLE_DEVICES= nice -n 10 "$PY" -I -B "$SC" "$c" --phase "$1" || { echo "=== STOP: scorer $c --phase $1 rc $?"; return 1; }
  done
}

wait_done "$ROOT/eval_r7_heartbeat_s7.json" || exit 1
score_phase S || exit 1
[ -s "$ROOT/SELECTION_S7.sha256" ] || { echo "=== STOP: SELECTION_S7.sha256 missing"; exit 1; }
[ -e "$ROOT/STOP" ] && { echo "=== STOP: ROOT/STOP exists; phase F not started"; exit 1; }
echo "=== launching phase F $(date '+%F %T')"
(cd "$ME" && env -u CUDA_VISIBLE_DEVICES bash "$ME/scripts/t2cons_r7_eval_launch.sh" --phase final) || { echo "=== STOP: launch rc $?"; exit 1; }
sleep 120
wait_done "$ROOT/eval_r7_heartbeat_f7.json" || exit 1
score_phase F || exit 1
for c in check prepare pack validate; do
  echo "=== pack $c $(date '+%F %T')"
  CUDA_VISIBLE_DEVICES= nice -n 10 "$PY" -I -B "$PK" "$c" || { echo "=== STOP: pack $c rc $?"; exit 1; }
done
echo "=== ALL DONE $(date '+%F %T')"
