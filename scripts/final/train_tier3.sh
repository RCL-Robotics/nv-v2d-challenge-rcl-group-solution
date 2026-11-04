#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# train_tier3.sh: retrain the 30 checkpoints of the counted Tier-3 file (record TR/t3ready_20261102/),
# one episode at a time, with the recorded code, overrides, seeds and iterations.
#   stage f0 (45000 iterations from scratch, every episode) and, for the 11 retry items of that file, stage retry_C90 (resume from
#   the recorded f0 model_44922 + rsi_44922 to 90000). Code 61fa5d3 = f2923d6. Inputs: the frame-0 root
#   data/human_motion_data_tier3_f0 (build_inputs_tier3.sh).
# Each stage runs in a git worktree of its recorded training commit ($V2D_WORK/worktrees/<sha7>; all are ancestors
# of this commit), or in this checkout with --code this. Training is GPU non-deterministic: a rerun reproduces the
# recipe and the policy class, not the bytes (REPRODUCE.md section 5).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
usage() { cat <<'EOF'
usage: train_tier3.sh --episode N[,N...] [options] [--dry-run] [--site FILE]
  --episode N[,N...]   episode number(s) of the counted file (required)
  --stage S            only this stage of the lineage (f0 or retry_C90; default: all, in order)
  --code training|this training: a worktree of the stage's recorded training commit (default); this: this checkout
  --runs-root DIR      where the runs go (default $V2D_WORK/runs/tier3)
  --iterations K       smoke test: scratch stages stop at K iterations, resumed stages at <resume iteration> + K
  --full-length        accepted for symmetry with train_tier1.sh: Tier-3 stages always train as recorded
  --check-sources      read-only: sha256 of every resume source in place and of the counted training checkpoint
  --gpu G              GPU index (default: first of $GPUS); REQUIRE_IDLE_GPU=1 refuses a GPU with any compute process
  --cpus LIST          taskset CPU list (optional)
  --hull-cache recorded|stock|none|DIR
                       convex-decomposition cache seeded into <run>/fc_cache (FLASH_CHORD_CACHE_DIR): the recorded one
                       if it is on this machine, else the stock runtime's (default recorded); none = the tree's own
  --lfs-from DIR       fill the worktree's LFS assets from DIR (sha256-checked copies) instead of `git lfs pull`
example: train_tier3.sh --episode 7 --stage f0 --iterations 50 --gpu 5 --dry-run
EOF
}
final_init "$@"
emit train_tier3 plan_py train --tier 3 ${REST[@]+"${REST[@]}"}
