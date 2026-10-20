#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fail when files contain home-directory paths, proxy or package-index settings, or anything that looks like a
# credential, or when a file matches a Git LFS pattern or is binary. Only generic patterns live here.
# usage: tests/check_hygiene.sh [file ...]
#   default: every file added or changed since the line's CODE_COMMIT (committed or not)
#   env: HYGIENE_EXTRA_PATTERNS  an extended regular expression (case-insensitive) of further forbidden strings,
#        e.g. personal or team names, machine names, internal hosts or mirror hosts, given at check time and
#        never written into the repository.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(git -C "$HERE" rev-parse --show-toplevel) || exit 2
cd "$ROOT" || exit 2
if [ $# -gt 0 ]; then FILES=("$@"); else
  # shellcheck disable=SC1091
  BASE=$(source "$HERE/../lines/c3r2_tier1.env" && echo "$CODE_COMMIT")
  mapfile -t FILES < <({ git diff --no-renames --name-only --diff-filter=AM "$BASE" -- .
                         git ls-files --others --exclude-standard; } | sort -u)
fi
# patterns are assembled from pieces so this file does not match itself
P_PATHS="/ro""ot/|/ho""me/[a-z]|/Use""rs/[A-Za-z]"
P_NET="(http|https|all|no)_pro""xy=|(PIP|UV)_INDEX_""URL=|HF_END""POINT=[^ ]"
P_KEYS="hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|github_""pat_|BEGIN [A-Z ]*PRIVATE KEY|AKIA[0-9A-Z]{16}|xox[abpr]-"
P_KEYS="$P_KEYS|(WANDB_API_""KEY|HF_TO""KEN|KAGGLE_""KEY)=[^\"\$ ]"
EXTRA=${HYGIENE_EXTRA_PATTERNS:-}
# 40-hex strings that are not one of the commits named in the line spec could be W&B keys
ALLOWED_SHA="f2923d61817e61eb37d939e0f75e5b4a7aaa5cbf|61fa5d31ff8054bf1e5cf5375ef0957e54d2729b|8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef|5f68335f3acc802033d1e80728c1633197521de8"
rc=0
for f in "${FILES[@]}"; do
  [ -f "$f" ] || continue
  if grep -nIiE "$P_PATHS|$P_NET" "$f"; then echo "FAIL ($f): home-directory path or proxy/index setting"; rc=1; fi
  if [ -n "$EXTRA" ] && grep -nIiE "$EXTRA" "$f"; then echo "FAIL ($f): matches HYGIENE_EXTRA_PATTERNS"; rc=1; fi
  if grep -nIE "$P_KEYS" "$f"; then echo "FAIL ($f): looks like a credential"; rc=1; fi
  if grep -noIE "\b[0-9a-f]{40}\b" "$f" | grep -vE ":($ALLOWED_SHA)$"; then echo "FAIL ($f): unexplained 40-hex string"; rc=1; fi
  if [ -n "$(git check-attr filter -- "$f" | grep ': lfs$')" ]; then echo "FAIL ($f): matches a Git LFS pattern"; rc=1; fi
  if ! grep -qI . "$f" && [ -s "$f" ]; then echo "FAIL ($f): binary file"; rc=1; fi
done
extra_note=""; [ -n "$EXTRA" ] && extra_note=", extra patterns"
[ $rc = 0 ] && echo "OK: ${#FILES[@]} files clean (paths, proxy/index settings, credentials, LFS, binaries$extra_note)"
exit $rc
