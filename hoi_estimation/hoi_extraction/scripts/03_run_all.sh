#!/usr/bin/env bash
# Run HOI extraction for all 30 Track-2 episodes, sequentially (single GPU).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR=${V2D_LOGS:-${V2D_ROOT:-/mnt/nas/Group/hanzhou/v2d}/logs}
mkdir -p "$LOGDIR"
for i in $(seq 0 29); do
  EP=$(printf "%06d" "$i")
  echo "=== episode $EP ==="
  bash "$HERE/02_run_episode.sh" "$i" > "$LOGDIR/episode_${EP}.log" 2>&1 \
    && echo "ok $EP" || echo "FAILED $EP (see $LOGDIR/episode_${EP}.log)"
done
