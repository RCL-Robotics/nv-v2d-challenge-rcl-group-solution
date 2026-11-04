#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# train_tier2.sh: retrain the 30 checkpoints of the counted Tier-2 file (R9-CLOSEST, trial 3; record TR/t2r9_20261104/),
# one episode at a time, with the recorded code, overrides, seeds and iterations.
#   one stage per item, on the item's own input root (raw / K3 / K3fix / G2). 15 items are ours (61fa5d3 = f2923d6),
#   15 are Zuxing Lu's (82837c8, bc3b1bd; RTX 5090, ws031 or this server). ep19 and ep25 resume from checkpoints that
#   exist only on Zuxing's RTX 5090 host (set ROOT_ZX5090 to a copy). ep10 has no run record on the team server: its
#   overrides are rebuilt from the checkpoint's embedded configuration.
# Each stage runs in a git worktree of its recorded training commit ($V2D_WORK/worktrees/<sha7>; all are ancestors
# of this commit), or in this checkout with --code this. Training is GPU non-deterministic: a rerun reproduces the
# recipe and the policy class, not the bytes (REPRODUCE.md section 9).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
usage() { cat <<'EOF'
usage: train_tier2.sh --episode N[,N...] [options] [--dry-run] [--site FILE]
  --episode N[,N...]   episode number(s) of the counted file (required)
  --stage S            only this stage of the lineage (1, 2, ...; default: all, in order)
  --code training|this training: a worktree of the stage's recorded training commit (default); this: this checkout
  --runs-root DIR      where the runs go (default $V2D_WORK/runs/tier2)
  --iterations K       smoke test: scratch stages stop at K iterations, resumed stages at <resume iteration> + K
  --full-length        accepted for symmetry with train_tier1.sh: Tier-2 stages always train as recorded
  --check-sources      read-only: sha256 of every resume source in place and of the counted training checkpoint
  --gpu G              GPU index (default: first of $GPUS); REQUIRE_IDLE_GPU=1 refuses a GPU with any compute process
  --cpus LIST          taskset CPU list (optional)
  --hull-cache recorded|stock|none|DIR
                       convex-decomposition cache seeded into <run>/fc_cache (FLASH_CHORD_CACHE_DIR): the recorded one
                       if it is on this machine, else the stock runtime's (default recorded); none = the tree's own
  --lfs-from DIR       fill the worktree's LFS assets from DIR (sha256-checked copies) instead of `git lfs pull`
example: train_tier2.sh --episode 17 --iterations 50 --gpu 6 --dry-run
EOF
}
final_init "$@"
emit train_tier2 plan_py train --tier 2 ${REST[@]+"${REST[@]}"}
