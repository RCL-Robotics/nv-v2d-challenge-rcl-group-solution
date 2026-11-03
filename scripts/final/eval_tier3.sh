#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# eval_tier3.sh: N official evaluations ("trials") of the 30 pinned evaluated (exported) checkpoints of the counted Tier-3 file (FINAL_r), with the kit evaluator
# (examples/run_policy_evaluation.py + flash_chord/scripts/evaluate_policy.py 160d24f5 of the Oct-11 kit) on the stock
# runtime video_to_data@8c778f9, through this checkout's scripts/track2/eval_all.py: world count 1, frame 0, explicit
# reset, --export no, one episode per call. Reference = each policy's OWN frame-0 input (data/human_motion_data_tier3_f0). These own-input numbers are not the
#   leaderboard score, which uses a hidden clean reference; this script never reads or computes a clean score.
# Each trial writes <out>/tNN/results/episode_*.{json,parquet} (what the kit packer reads); the summary
# (<out>/summary.tsv) lists AUC, SP-SR, MP-SR and MPPE per episode and trial. GPU simulation is not bit-deterministic,
# so trials differ slightly; the counted file is one recorded trial (pack_tier3.sh re-packs it).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
usage() { cat <<'EOF'
usage: eval_tier3.sh [--episode all|N[,N...]] [--trials K] [options] [--dry-run] [--site FILE]
  --episode            episodes (default all 30)
  --trials K           independent evaluations of each episode (default 1; the selections used 5 to 20)
  --out DIR            output (default $V2D_WORK/eval/tier3)
  --gpu G              GPU index (default: first of $GPUS); REQUIRE_IDLE_GPU=1 refuses a GPU with any compute process
  --cpus LIST          taskset CPU list (optional)
  --reexport           evaluate a fresh CPU export of the training checkpoint instead of the pinned file (the export
                       must be the same policy, compare_exports.py); the export's file sha256 differs from the pinned one
                       because safetensors writes the metadata keys in a random order
  --code training|this with --reexport: exporter of the recorded export commit (worktree, default) or of this checkout
example (ep07): eval_tier3.sh --episode 7 --trials 1 --gpu 6
EOF
}
final_init "$@"
emit eval_tier3 plan_py eval --tier 3 ${REST[@]+"${REST[@]}"}
