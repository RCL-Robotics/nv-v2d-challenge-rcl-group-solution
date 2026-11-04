#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# select_tier2.sh: the selection that chose the counted Tier-2 file, as a pointer to its pinned scripts and records.
# The final Tier-2 selection, "round 9" in its records (TR/t2r9_20261104/README.md; earlier selection rounds 0-8:
# TR/t2r8_20261103, TR/t2r7_20261102, TR/t2ready_20261031):
#   PREREG_R9.md (hash-locked before the pool freeze and before any evaluation of this selection), the pool rule D2, phase S
#   (3 official own-input runs per candidate), phase F (10 runs per finalist, replacement bar m_F >= max(m_P) + 0.01 on
#   consensus-B AUC), then R9-CLOSEST (the counted file) = the complete trial whose 30-episode mean own-input AUC is closest to the mean of
#   the 10 trials (trial 3; never a best-of). Only own-input and consensus-B scores (refs/B, built from Tier-2 tracks).
# A real run is read-only: it verifies the record's hash locks (SHA256SUMS, LOCK_R9, PREREG_R9, PINS, POOL, RELOCATION,
# SELECTION_S9/R9) and prints the decision. The selection's drivers are printed, not run: they need the training server's
# layout and the full evaluation folders (runs_s/, runs_f/, exports/), which are not in git (README "Record scope").
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
usage() { cat <<'EOF2'
usage: select_tier2.sh [--dry-run] [--site FILE]
EOF2
}
final_init "$@"
[ ${#REST[@]} -eq 0 ] || die "unknown option ${REST[0]} (see --help)"
R9=$TR/t2r9_20261104
SEL=$R9/selection/round9_20261103
plan_header "Tier-2 final selection: verify the locks, print the decision"
addq TRK="$TR"; addq SEL="$SEL"
add 'check_sums "$TRK" t2r9_20261104/SHA256SUMS'
add 'check_sums "$SEL" LOCK_R9.sha256 PREREG_R9.sha256 PREREG_R9_PINS.sha256 POOL_R9.sha256 RELOCATION_R9.sha256 SELECTION_S9.sha256 SELECTION_R9.sha256'
add 'check_sums "$TRK/t2r9_20261104/pack" PACK_R9_PINS.sha256'
add '"$PLAN_PY" -I -B "$FINAL_LIB/show_selection.py" --tier 2'
add '# ---- the selection drivers as they ran (training-server layout <ME>/fc_runs/submissions/t2_consensus_r9_20261103; each refuses'
add '#      unless its pinned PREREG/LOCK files match):'
add '#   bash $SEL/r9_launch.sh                         pool freeze, phase S, phase F, selection (r9_driver.py run)'
add '#   $FC_PYTHON -I -B $SEL/r9_driver.py status       progress; r9_pool.py / r9_eval.py / r9_select.py are its steps'
add '#   $KIT_PYTHON -I -B $TR/t2r9_20261104/pack/t2cons_r9_pack.py check|prepare|pack|validate|pack-commit   (pack_tier2.sh)'
add '# earlier selection rounds: t2r8_20261103/selection/round8_20261103/r8_*.py, t2r7_20261102/selection/scripts_r7/, t2ready_20261031/selection/scripts/'
printf '%s' "$PLAN" | run_plan select_tier2
