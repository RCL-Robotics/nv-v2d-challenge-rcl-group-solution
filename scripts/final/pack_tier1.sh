#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# pack_tier1.sh: pack the counted Tier-1 file (set all7, trial 05) for a commit URL with the Oct-11 kit packer
# tools/pack_submission.py (47a45895, unchanged; --code-commit), then the kit uploader's offline validate_submission()
# with provenance checks, the kit metric_code scorer and a row-by-row comparison with the source file
# (TR/t1best/t1ready_20261031/repack.py). No evaluation is re-run: the trial's 30 recorded rollouts are re-packed, so
# the only column that changes is code_commit_url. Nothing is uploaded.
#   recorded inputs (training server): results T1_RUNS/T1improve/topup/scores/all7/t05/results (60 files);
#   source file  fc_runs/submissions/t1ready_20261031/files/T1_all7_trial05_closest_to_mean_b1e6597.csv (aee58640...)
# --results DIR packs a fresh evaluation instead (eval_tier1.sh output tNN/results): the comparison with the source file
# then reports the pose differences of the new rollouts.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
usage() { cat <<'EOF2'
usage: pack_tier1.sh [--commit-url URL] [--out DIR] [--name FILE.csv] [--results DIR] [--source-csv F] [--source-scores F]
                     [--dry-run] [--site FILE]
  --commit-url URL   default: the GitHub URL of this checkout's HEAD ($COMMIT_URL); it must be pushed before uploading
  --out DIR          default $V2D_WORK/pack/tier1
EOF2
}
final_init "$@"
SUB=$ROOT_ME/fc_runs/submissions/t1ready_20261031/files
RESULTS=$T1_RUNS/T1improve/topup/scores/all7/t05/results; SRC=$SUB/T1_all7_trial05_closest_to_mean_b1e6597.csv
SCORES=$SUB/metric_code.json; OUT=$V2D_WORK/pack/tier1; NAME=""
set -- ${REST[@]+"${REST[@]}"}
while [ $# -gt 0 ]; do
  case "$1" in
    --commit-url) COMMIT_URL=$2; shift ;; --out) OUT=$2; shift ;; --name) NAME=$2; shift ;;
    --results) RESULTS=$2; shift ;; --source-csv) SRC=$2; shift ;; --source-scores) SCORES=$2; shift ;;
    *) die "unknown option $1 (see --help)" ;;
  esac; shift
done
: "${NAME:=T1_all7_trial05_${COMMIT_URL##*/}.csv}"
A=$TR/t1best/t1ready_20261031
plan_header "Tier-1 pack for $COMMIT_URL"
add 'check_sha256 "$V2D_KIT/tools/pack_submission.py" 47a4589574ff8284ba4376ef31ac5548c8fe87ead753be2d3b43d6c87041e5be'
add 'check_sha256 "$V2D_KIT/submit/v2d_submit.py" fe5cb1dc86d12abd01eab9dd5dd98b51569b5d8513a2d00d4e79da42a5f11718'
addq mkdir -p "$OUT"
addq env V2D_KIT="$V2D_KIT" V2D_DATA="$V2D_DATA" "$KIT_PYTHON" -B "$A/repack.py" --kit "$V2D_KIT" --dataset-root "$V2D_DATA" \
  --results "$RESULTS" --source-csv "$SRC" --source-scores "$SCORES" \
  --expected "$A/amend1/files_trial05/expected_checkpoints.json" --scorer "$TR/t1best/score_csv.py" \
  --validator "$TR/t1best/t1improve/topup/tools/t1topup_validate.py" --commit-url "$COMMIT_URL" --out-dir "$OUT" \
  --name "$NAME" --python "$KIT_PYTHON"
add "note \"packed: \$(sha256sum $(printf %q "$OUT/$NAME"))\""
printf '%s' "$PLAN" | run_plan pack_tier1
