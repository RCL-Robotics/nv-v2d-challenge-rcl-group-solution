#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# select_tier3.sh: the selection that chose the counted Tier-3 file (FINAL_r in its records), as a pointer to its pinned scripts and
# records (TR/t3ready_20261102/README.md section 2; evaluation/t3_f0_final/PREREG.md, AMENDMENT_1.md, AMENDMENT_2.md;
# evaluation/f0_retry/PREREG_retry.md, PREREG_v3.md):
#   f0 item per episode: the f0 run's final.json checkpoint, 5 official own-input evaluations, the run closest to the
#   item's own-input mean ADD-AUC; retry items (14 rule-selected episodes, C90 continuation to 90000) replace the f0 item
#   only if own-input SP/MP improve (PREREG_retry acceptance); FINAL_r = AMENDMENT_1 (e), built by t3final_r_build.py.
#   Only own-input scores; no clean reference decides anything.
# A real run is read-only: it verifies the record's SHA256SUMS and the PREREG / AMENDMENT / FINAL_r / retry locks,
# runs check_set_t3.py --files --hash (CPU) and prints the composition. The drivers are printed, not run (they need the
# training server's layout and the evaluation folders, which are not in git).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
usage() { cat <<'EOF2'
usage: select_tier3.sh [--dry-run] [--site FILE]
EOF2
}
final_init "$@"
[ ${#REST[@]} -eq 0 ] || die "unknown option ${REST[0]} (see --help)"
B=$TR/t3ready_20261102
plan_header "Tier-3 final selection: verify the locks, print the composition"
addq B="$B"
add 'check_sums "$B" SHA256SUMS'
add 'check_sums "$B/evaluation/t3_f0_final" PREREG.sha256 AMENDMENT_1.sha256 AMENDMENT_2.sha256 FINAL_r.sha256'
add 'check_sums "$B/evaluation/f0_retry" PREREG_retry.sha256 PREREG_v3.sha256 RETRY_FILES.sha256 RETRY_FILES_v3.sha256 recipes.sha256'
add 'note "builder t3final_r_build.py: $(sha256sum "$B/evaluation/f0_retry/t3final_r_build.py" | cut -c1-16)... (AMENDMENT_2 pin 29dc19339d2e38c9...)"'
add '"$PLAN_PY" -I -B "$B/tools/check_set_t3.py" --files --hash | tail -n 15'
add '"$PLAN_PY" -I -B "$FINAL_LIB/show_selection.py" --tier 3'
add '# ---- the drivers as they ran (server layout <ME>; each refuses unless its pinned PREREG / hash files match):'
add '#   bash <ME>/scripts/t3f0_driver.sh run                      f0: 30 trainings + post (PREREG_f0)'
add '#   bash <ME>/fc_runs/submissions/t3_f0_final/launch.sh      5x own-input evaluation, selection, FINAL_f0 pack (PREREG.md)'
add '#   bash <ME>/tmp_analysis/tier3/f0_retry/launch_v3.sh       retry wave: 14 C90 continuations + 5+5 evaluations (PREREG_v3)'
add '#   $FC_PYTHON -I -B <ME>/tmp_analysis/tier3/f0_retry/t3final_r_build.py when|build    FINAL_r (AMENDMENT_2)'
add '#   (restore them with tools/restore_layout_t3.py; build_inputs_tier3.sh --steps restore)'
printf '%s' "$PLAN" | run_plan select_tier3
