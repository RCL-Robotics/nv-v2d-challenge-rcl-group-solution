#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# The reproduction tooling must not change the code that trained, exported and evaluated the submitted checkpoints.
# Fails when any of these paths differs between <base> (default: the line's CODE_COMMIT) and HEAD, or between <base>
# and the working tree: the FlashCHORD sources, the existing track2 scripts, the package metadata and lock files,
# and the retargeting scripts.
# usage: tests/check_code_unchanged.sh [base commit]
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
BASE=${1:-$(source "$HERE/../lines/c3r2_tier1.env" && echo "$CODE_COMMIT")}
ROOT=$(git -C "$HERE" rev-parse --show-toplevel) || exit 2
PATHS=(robotic_grounding/flash_chord/src
       robotic_grounding/flash_chord/scripts/track2/eval_all.py
       robotic_grounding/flash_chord/scripts/track2/launch.py
       robotic_grounding/flash_chord/scripts/track2/record_eval_actions.py
       robotic_grounding/flash_chord/scripts/train_rl.py
       robotic_grounding/flash_chord/pyproject.toml
       robotic_grounding/flash_chord/uv.lock
       robotic_grounding/pyproject.toml
       robotic_grounding/scripts/retarget)
cd "$ROOT" || exit 2
git cat-file -e "$BASE^{commit}" 2>/dev/null || { echo "FAIL: base commit $BASE is not in this repository"; exit 1; }
rc=0
if git diff --quiet "$BASE" HEAD -- "${PATHS[@]}"; then echo "OK: HEAD = $BASE on the protected paths"
else echo "FAIL: HEAD differs from $BASE on:"; git diff --stat "$BASE" HEAD -- "${PATHS[@]}"; rc=1; fi
if git diff --quiet "$BASE" -- "${PATHS[@]}"; then echo "OK: working tree = $BASE on the protected paths"
else echo "FAIL: the working tree differs from $BASE on:"; git diff --stat "$BASE" -- "${PATHS[@]}"; rc=1; fi
exit $rc
