#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# train_tier1.sh: retrain the 30 checkpoints of the counted Tier-1 file (set all7, trial 05; record TR/t1best/t1ready_20261031/),
# one episode at a time, with the recorded code, overrides, seeds and iterations.
#   54 stages in 30 lineages: C3r2_runin / C3r2_wide from scratch, teammate runin chains, T1fix forks and the 7
#   T1improve arms (set_items.json `stages`, set_stages.tsv). Example: --episode 8 is one stage from scratch at b39b69b.
# Resume sources: our runs in T1_RUNS (<ME>/fc_runs), the teammate runin chains (ep01, ep04, ep19) in T1_ZX_RUNS
# (<ZX>/runs/2026-10-14-patience-runin); copies are sha256-checked (set record, else lib/resume_sha256.tsv). In-place
# restarts (ep00, ep20) and teammate chains resume from this replay's own previous stage when it ran (handoff_N.sha256).
# Each stage runs in a git worktree of its recorded training commit ($V2D_WORK/worktrees/<sha7>; all are ancestors
# of this commit), or in this checkout with --code this. Training is GPU non-deterministic: a rerun reproduces the
# recipe and the policy class, not the bytes (REPRODUCE.md section 9).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
usage() { cat <<'EOF'
usage: train_tier1.sh --episode N[,N...] [options] [--dry-run] [--site FILE]
  --episode N[,N...]   episode number(s) of the counted file (required)
  --stage S            only this stage of the lineage (1, 2, ...; default: all, in order)
  --code training|this training: a worktree of the stage's recorded training commit (default); this: this checkout
  --runs-root DIR      where the runs go (default $V2D_WORK/runs/tier1)
  --iterations K       smoke test: scratch stages stop at K iterations, resumed stages at <resume iteration> + K (a
                       chained stage whose previous stage is in the same plan resumes from that stage's smoke end)
  --full-length        train every stage to its recorded max_iterations. Default: a stage whose recorded run was
                       stopped from outside or went on past what is used later (in-place restarts, manual_plateau,
                       "running at selection", fork points, the counted checkpoint) stops at the iteration that is
                       used, through training.max_iterations, only if lib/stop_check.py finds that the resolved
                       configuration is otherwise identical (no schedule reads it); stages that ended by the trainer's
                       own early stop are not cut
  --check-sources      read-only: sha256 of every resume source in place and of the counted training checkpoint
  --gpu G              GPU index (default: first of $GPUS); REQUIRE_IDLE_GPU=1 refuses a GPU with any compute process
  --cpus LIST          taskset CPU list (optional)
  --hull-cache recorded|stock|none|DIR
                       convex-decomposition cache seeded into <run>/fc_cache (FLASH_CHORD_CACHE_DIR): the recorded one
                       if it is on this machine, else the stock runtime's (default recorded); none = the tree's own
  --lfs-from DIR       fill the worktree's LFS assets from DIR (sha256-checked copies) instead of `git lfs pull`
example: train_tier1.sh --episode 8 --iterations 50 --gpu 7 --dry-run
EOF
}
final_init "$@"
emit train_tier1 plan_py train --tier 1 ${REST[@]+"${REST[@]}"}
