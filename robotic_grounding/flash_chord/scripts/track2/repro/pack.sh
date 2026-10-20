#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Pack the evaluator results into the Kaggle submission CSV with the official kit packer, then run the kit's
# preflight. Nothing is uploaded.
#
# usage: pack.sh [--line FILE] [--results DIR] [--out CSV] [--code-commit URL] [--no-preflight] [--dry-run]
#   --results      default <V2D_WORK>/eval/<ARM>/results (final_eval.sh)
#   --out          default $SUBMISSION_CSV, else <V2D_WORK>/submissions/<ARM>_tier1.csv
#   --code-commit  default $CODE_COMMIT_URL, else https://github.com/<owner>/<repo>/commit/<HEAD of this checkout>
#                  taken from the origin remote; HEAD must be committed, clean and pushed (the URL must resolve
#                  for the organizers)
#   env: V2D_WORK V2D_DATA V2D_KIT KIT_PYTHON  SKIP_INPUT_CHECK=1 (skip the kit check)
# Relative paths are taken from the caller's working directory (the packer itself runs inside the kit). Before
# packing, check_eval_inputs.py checks every kit file against KIT_MANIFEST.json.
# The command is the one the submitted file was packed with:
#   <kit python> -B tools/pack_submission.py --track track_2_tier1 --results <results> --dataset-root <V2D_DATA> \
#       --code-commit <URL> -o <csv>
# then  <kit python> -B tools/preflight.py --track track_2_tier1 --results <results> --dataset-root <V2D_DATA> --skip-kaggle
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
abs(){ case $1 in /*|"") printf '%s' "$1";; *) printf '%s' "$PWD/$1";; esac; }   # relative to the caller's cwd
LINE=$HERE/lines/c3r2_tier1.env; RESULTS=""; OUT=""; URL=${CODE_COMMIT_URL:-}; PRE=1; DRY=0
while [ $# -gt 0 ]; do case "$1" in
  --line) LINE=$2; shift 2;; --results) RESULTS=$2; shift 2;; --out) OUT=$2; shift 2;;
  --code-commit) URL=$2; shift 2;; --no-preflight) PRE=0; shift;; --dry-run) DRY=1; shift;;
  *) echo "unknown argument $1" >&2; exit 2;; esac; done
# shellcheck disable=SC1090
source "$LINE"
: "${V2D_DATA:?set V2D_DATA}" "${V2D_KIT:?set V2D_KIT}" "${KIT_PYTHON:?set KIT_PYTHON}"
RESULTS=${RESULTS:-${V2D_WORK:?set V2D_WORK or pass --results}/eval/$ARM/results}
OUT=${OUT:-${SUBMISSION_CSV:-${V2D_WORK:?set V2D_WORK or pass --out}/submissions/${ARM}_tier1.csv}}
RESULTS=$(abs "$RESULTS"); OUT=$(abs "$OUT"); V2D_DATA=$(abs "$V2D_DATA"); V2D_KIT=$(abs "$V2D_KIT")
case $KIT_PYTHON in */*) KIT_PYTHON=$(abs "$KIT_PYTHON");; esac
if [ -z "$URL" ]; then
  repo=$(git -C "$HERE" remote get-url origin 2>/dev/null | sed -E 's#^(https://|git@)github\.com[:/]##; s#\.git$##')
  sha=$(git -C "$HERE" rev-parse HEAD 2>/dev/null)
  [ -n "$repo" ] && [ -n "$sha" ] || { echo "cannot derive the commit URL; pass --code-commit" >&2; exit 2; }
  URL=https://github.com/$repo/commit/$sha
  if [ $DRY = 0 ]; then
    [ -z "$(git -C "$HERE" status --porcelain --untracked-files=no)" ] || { echo "the checkout has uncommitted changes" >&2; exit 2; }
    git -C "$HERE" branch -r --contains "$sha" 2>/dev/null | grep -q . || { echo "HEAD $sha is not on any fetched remote branch: push it first" >&2; exit 2; }
  fi
fi
case "$URL" in https://github.com/*/commit/[0-9a-f]*) ;; *) echo "not a GitHub /commit/<sha> URL: $URL" >&2; exit 2;; esac
PACK=("$KIT_PYTHON" -B tools/pack_submission.py --track "$TRACK" --results "$RESULTS" --dataset-root "$V2D_DATA"
      --code-commit "$URL" -o "$OUT")
PREFLIGHT=("$KIT_PYTHON" -B tools/preflight.py --track "$TRACK" --results "$RESULTS" --dataset-root "$V2D_DATA" --skip-kaggle)
CHECK=("$KIT_PYTHON" -B "$HERE/check_eval_inputs.py" --line "$LINE" --kit "$V2D_KIT")
if [ $DRY = 1 ]; then
  [ "${SKIP_INPUT_CHECK:-0}" = 1 ] || { printf '[pack] '; printf '%q ' "${CHECK[@]}"; echo; }
  printf '[pack] cd %q && PYTHONDONTWRITEBYTECODE=1 ' "$V2D_KIT"; printf '%q ' "${PACK[@]}"; echo
  [ $PRE = 1 ] && { printf '[pack] cd %q && PYTHONDONTWRITEBYTECODE=1 ' "$V2D_KIT"; printf '%q ' "${PREFLIGHT[@]}"; echo; }
  exit 0
fi
if [ "${SKIP_INPUT_CHECK:-0}" != 1 ]; then
  "${CHECK[@]}" || { echo "the kit does not match its manifest or the line spec (SKIP_INPUT_CHECK=1 skips)" >&2; exit 1; }
fi
mkdir -p "$(dirname "$OUT")"
( cd "$V2D_KIT" && PYTHONDONTWRITEBYTECODE=1 "${PACK[@]}" ) || exit 1
echo "packed $OUT  sha256 $(sha256sum "$OUT" | cut -d' ' -f1)  code_commit_url $URL"
if [ $PRE = 1 ]; then ( cd "$V2D_KIT" && PYTHONDONTWRITEBYTECODE=1 "${PREFLIGHT[@]}" ); fi
