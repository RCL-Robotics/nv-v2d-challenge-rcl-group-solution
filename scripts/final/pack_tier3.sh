#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# pack_tier3.sh: pack the counted Tier-3 file (FINAL_r in its records) for a commit URL with the Oct-11 kit packer tools/pack_submission.py
# (47a45895, unchanged; --track track_2_tier3 --code-commit; each episode with its own frame_references_f0 npz), through
# the pinned builder TR/t3ready_20261102/evaluation/f0_retry/t3final_r_build.py repack (AMENDMENT_2 pin 29dc1933...):
# it re-checks the FINAL_r lock and FILES.sha256, copies the 30 stored results / frame references (sha256-checked), runs
# the packer, the input-reference residual check (< 1e-4 m / 1e-3 rad), the uploader's offline validate_submission()
# (t3final_kit.py) and provenance, and writes <ME>/fc_runs/submissions/t3_f0_final/upload_r/<sha7>/ (log appended to
# <ME>/tmp_analysis/tier3/f0_retry/final_r_build.log). No evaluation is re-run; nothing is uploaded. The packer's own
# clean cross-check line goes only to that folder's pack_LOCAL_NOT_FOR_UPLOAD.log (never read by any decision).
# The builder embeds the training server's paths (<ME>); elsewhere restore them first (build_inputs_tier3.sh --steps restore).
# Expected: pose columns 82fef2c7..., and the CSV equal to upload_r/a7e7d3a/t3_final_r_closest_a7e7d3a.csv (ba78a547...)
# in every column except code_commit_url.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
usage() { cat <<'EOF2'
usage: pack_tier3.sh [--commit-url URL] [--dry-run] [--site FILE]
  --commit-url URL   default: the GitHub URL of this checkout's HEAD ($COMMIT_URL); it must be pushed before uploading
EOF2
}
final_init "$@"
set -- ${REST[@]+"${REST[@]}"}
while [ $# -gt 0 ]; do
  case "$1" in
    --commit-url) COMMIT_URL=$2; shift ;;
    *) die "unknown option $1 (see --help)" ;;
  esac; shift
done
B=$TR/t3ready_20261102
F=$ROOT_ME/fc_runs/submissions/t3_f0_final
plan_header "Tier-3 pack for $COMMIT_URL"
add 'check_sha256 "$V2D_KIT/tools/pack_submission.py" 47a4589574ff8284ba4376ef31ac5548c8fe87ead753be2d3b43d6c87041e5be'
add 'check_sha256 "$V2D_KIT/submit/v2d_submit.py" fe5cb1dc86d12abd01eab9dd5dd98b51569b5d8513a2d00d4e79da42a5f11718'
addq check_sha256 "$B/evaluation/f0_retry/t3final_r_build.py" 29dc19339d2e38c96352438a971752332c226ff1a153fc78bdd4e0e413d61057
addq "$FC_PYTHON" -I -B "$B/evaluation/f0_retry/t3final_r_build.py" repack --commit-url "$COMMIT_URL"
add "sha7=$(printf %q "${COMMIT_URL##*/}"); sha7=\${sha7:0:7}"
addq ls -la "$F/upload_r/"
add "note \"packed folder: $F/upload_r/\$sha7 (CSV t3_final_r_closest_\$sha7.csv)\""
printf '%s' "$PLAN" | run_plan pack_tier3
