#!/usr/bin/env bash
# Run the temporal-prompt agent (AGENT_TASK_temporal.md) on many episodes, at most P at a time on GPU 0.
# A run counts as active while its workspace has codex_events.jsonl but no wall_s line in run_manifest.txt, so runs
# started by an earlier invocation are counted too and nothing is started twice. Episodes with an output are skipped.
# Usage: run_batch_T1.sh <P> <ep> [<ep> ...]
set -uo pipefail
P=$1; shift
HERE=$(cd "$(dirname "$0")" && pwd)
V2D=${V2D_ROOT:?set V2D_ROOT to the data root (reproducibility/README.md)}

active() {
  local n=0 w
  for w in $V2D/agent_runs/T1_ds_ep*; do
    [ -f $w/logs/codex_events.jsonl ] && ! grep -q '^wall_s' $w/logs/run_manifest.txt 2>/dev/null && n=$((n + 1))
  done
  echo $n
}

one() {
  local n=$1 e ep ws src pl
  e=$(printf %03d $n); ep=$(printf episode_%06d $n); ws=$V2D/agent_runs/T1_ds_ep$e
  src=$(python3 -c "import json;print(json.load(open('$V2D/logs/JC2s1_shared_f820_pp_align.json'))['$n']['src'])")
  if [ "$src" = M1 ]; then pl=$V2D/experiments/M1_prompt_v2/out/$ep
  elif [ -f $V2D/outputs/cari4d/$ep/inference/coconet.pth ]; then pl=$V2D/outputs/cari4d/$ep
  else pl=$V2D/outputs/cari4d_failed/${ep}_out; fi
  echo "$(date +%T) start $e pipeline=$pl"
  bash $HERE/run_codex.sh $ws high 90 open $pl $HERE/AGENT_TASK_temporal.md
  echo "$(date +%T) end $e $(tail -1 $ws/logs/run_manifest.txt) outputs=$(ls $ws/outputs | tr '\n' ' ')"
}

for n in "$@"; do
  ws=$V2D/agent_runs/T1_ds_ep$(printf %03d $n)
  if [ -f $ws/outputs/object_poses_world.npz ] || [ -f $ws/logs/codex_events.jsonl ]; then echo "skip $n (done or running)"; continue; fi
  until [ "$(active)" -lt "$P" ]; do sleep 30; done
  one $n &
  sleep 20  # let the run create its event log before counting again
done
wait
