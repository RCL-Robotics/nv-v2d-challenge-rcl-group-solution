#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# pack_tier2.sh: pack the counted Tier-2 file (trial 3 of the final Tier-2 selection) for a commit URL with the Oct-11 kit packer
# tools/pack_submission.py (47a45895, unchanged; --code-commit; each episode with the frame reference of its own input),
# through TR/t2ready_20261031/tools/repack_t2.py: results and frame references unchanged by packing, the kit uploader's
# offline validate_submission() with provenance (one checkpoint digest per episode = set_items_r9.json), and every row
# equal to the source file's except code_commit_url (pose columns 1380472d...). No evaluation is re-run; nothing is
# uploaded. The kit packer always prints one clean-reference "locally computed" line; it goes through
# lib/kitpy_withhold.sh, which replaces it before anything shows or stores it (Tier-2 no-clean rule).
#   recorded inputs (training server): fc_runs/submissions/t2r9_pack_20261104/pack_r9_placeholder/r9_closest/{results,
#   frame_refs} and the source file submission_r9/T2_R9_closest_trial03_1dcee7e.csv (3129b378...).
# The selection's own pack script (t2r9_20261104/pack/t2cons_r9_pack.py pack-commit) additionally gates on the clean-history
# build clone and a branch-head listing of that clone; use it only there.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
usage() { cat <<'EOF2'
usage: pack_tier2.sh [--commit-url URL] [--out DIR] [--name FILE.csv] [--pack-dir DIR] [--source-csv F] [--dry-run] [--site FILE]
  --commit-url URL   default: the GitHub URL of this checkout's HEAD ($COMMIT_URL); it must be pushed before uploading
  --out DIR          default $V2D_WORK/pack/tier2
EOF2
}
final_init "$@"
P=$ROOT_ME/fc_runs/submissions/t2r9_pack_20261104
PACKDIR=$P/pack_r9_placeholder/r9_closest; SRC=$P/submission_r9/T2_R9_closest_trial03_1dcee7e.csv
OUT=$V2D_WORK/pack/tier2; NAME=""
set -- ${REST[@]+"${REST[@]}"}
while [ $# -gt 0 ]; do
  case "$1" in
    --commit-url) COMMIT_URL=$2; shift ;; --out) OUT=$2; shift ;; --name) NAME=$2; shift ;;
    --pack-dir) PACKDIR=$2; shift ;; --source-csv) SRC=$2; shift ;;
    *) die "unknown option $1 (see --help)" ;;
  esac; shift
done
: "${NAME:=T2_R9_closest_trial03_${COMMIT_URL##*/}.csv}"
plan_header "Tier-2 pack for $COMMIT_URL"
add 'check_sha256 "$V2D_KIT/tools/pack_submission.py" 47a4589574ff8284ba4376ef31ac5548c8fe87ead753be2d3b43d6c87041e5be'
add 'check_sha256 "$V2D_KIT/submit/v2d_submit.py" fe5cb1dc86d12abd01eab9dd5dd98b51569b5d8513a2d00d4e79da42a5f11718'
addq mkdir -p "$OUT"
addq env KIT_PYTHON="$KIT_PYTHON" "$KIT_PYTHON" -I -B "$TR/t2ready_20261031/tools/repack_t2.py" --kit "$V2D_KIT" \
  --dataset-root "$V2D_DATA" --pack-dir "$PACKDIR" --source-csv "$SRC" --items "$TR/t2r9_20261104/set/set_items_r9.json" \
  --commit-url "$COMMIT_URL" --out-dir "$OUT" --name "$NAME" --python "$FINAL_LIB/kitpy_withhold.sh"
add "note \"packed: \$(sha256sum $(printf %q "$OUT/$NAME"))\""
printf '%s' "$PLAN" | run_plan pack_tier2
