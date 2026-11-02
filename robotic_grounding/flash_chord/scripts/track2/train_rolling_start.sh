#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Train the C3r2 rolling-start arm (experiment g1_recon_body_coord_rsi_r2_runin) on Track 2 Tier-1 episodes through
# scripts/track2/launch.py (manifest lookup, scene.support rule, resume with the RSI sidecar, K jobs per GPU).
# Recipe and results: scripts/track2/rolling_start.md.
#
# Usage: train_rolling_start.sh <episodes> <gpus> <out dir> [launch.py options...]
#   train_rolling_start.sh 19,4,1 4 /root/autodl-tmp/haoran.zhang/fc_runs/C3r2_runin --jobs-per-gpu 3
#   train_rolling_start.sh all 0,1,2,3 /root/autodl-tmp/haoran.zhang/fc_runs/C3r2_runin --jobs-per-gpu 4 \
#       --wandb-mode online --wandb-entity ENTITY --wandb-project PROJECT
#   ITERATIONS=130000 train_rolling_start.sh 1 4 <same out dir>   # continue a finished run (resumes model_90000)
#   train_rolling_start.sh 19 4 <out dir> --dry-run                 # print the commands only
#
# Environment (all optional):
#   ITERATIONS       training.max_iterations (default 90000; what solved ep19 and ep01, see the doc)
#   EARLY_STOP_TAIL  training.early_stop_help_zero_tail (default 0 = off; the arm's own value is 5860, which stops
#                    at ~44,922 and makes launch.py report the job as failed because model_<ITERATIONS> is missing)
#   EXTRA            more Hydra overrides, one string (do not pass --extra yourself: it would replace these)
#   ARM              logging.experiment / W&B group (default C3r2_runin)
#   FC_ENV           environment script to source (default the pro6000 one: conda env, HUMAN_MOTION_DATA_DIR)
set -eo pipefail
FC=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
FC_ENV=${FC_ENV:-/root/autodl-tmp/haoran.zhang/scripts/fc_env.sh}
if [ -f "$FC_ENV" ]; then
  # shellcheck disable=SC1090
  source "$FC_ENV"
fi
: "${HUMAN_MOTION_DATA_DIR:?set HUMAN_MOTION_DATA_DIR (or FC_ENV to a script that sets it)}"
if [ $# -lt 3 ]; then
  sed -n '8,13p' "$0"
  exit 2
fi
episodes=$1 gpus=$2 out=$3
shift 3
# This checkout is the training code (not FC_ROOT of FC_ENV, which may point to a stock runtime without the arm).
exec python "$FC/scripts/track2/launch.py" \
  --arm "${ARM:-C3r2_runin}" \
  --experiment g1_recon_body_coord_rsi_r2_runin \
  --code "$FC" \
  --episodes "$episodes" \
  --gpus "$gpus" \
  --iterations "${ITERATIONS:-90000}" \
  --out "$out" \
  --extra "training.early_stop_help_zero_tail=${EARLY_STOP_TAIL:-0} ${EXTRA:-}" \
  "$@"
