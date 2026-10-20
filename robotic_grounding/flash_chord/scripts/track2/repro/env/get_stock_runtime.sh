#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Fetch the stock FlashCHORD runtime that the submission kit evaluates against: nvidia-isaac/video_to_data at
# 8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef, only robotic_grounding/flash_chord, with its Git LFS assets (meshes,
# policies). The evaluator must run on this untouched tree. The pipeline changes no tracked file in it; running the
# evaluator writes git-ignored files only (the convex-decomposition cache in src/flash_chord/assets/.cache, unless
# FLASH_CHORD_CACHE_DIR points elsewhere; see README "Caveats").
#
# usage: get_stock_runtime.sh <dir>      e.g. get_stock_runtime.sh $V2D_WORK/video_to_data_8c778f9
#   then V2D_RUNTIME=<dir>/robotic_grounding/flash_chord
# Needs git and git-lfs. Re-running it on an existing <dir> only re-checks it. It refuses a <dir> that is this
# repository's own checkout or a git checkout of another repository.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DIR=${1:?usage: get_stock_runtime.sh <dir>}
SHA=8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef; URL=https://github.com/nvidia-isaac/video_to_data.git
SUB=robotic_grounding/flash_chord
mkdir -p "$DIR" && DIR=$(cd "$DIR" && pwd -P) || exit 1
SELF=$(git -C "$HERE" rev-parse --show-toplevel 2>/dev/null) && SELF=$(cd "$SELF" && pwd -P)
if [ -n "$SELF" ] && [ "$DIR" = "$SELF" ]; then
  echo "refusing: $DIR is this repository's own checkout, not a separate stock runtime directory" >&2; exit 2
fi
cd "$DIR" || exit 1
if [ -e .git ]; then
  origin=$(git config --get remote.origin.url 2>/dev/null)
  case "$origin" in
    https://github.com/nvidia-isaac/video_to_data|https://github.com/nvidia-isaac/video_to_data.git) ;;
    git@github.com:nvidia-isaac/video_to_data|git@github.com:nvidia-isaac/video_to_data.git) ;;
    *) echo "refusing: $DIR/.git has origin '${origin:-none}', not $URL" >&2; exit 2;;
  esac
else
  git init -q && git remote add origin "$URL" || exit 1
fi
if [ "$(git rev-parse -q --verify HEAD 2>/dev/null)" != "$SHA" ]; then
  for i in 1 2 3 4 5; do GIT_LFS_SKIP_SMUDGE=1 git fetch -q --depth 1 origin "$SHA" && break; echo "fetch attempt $i failed"; sleep 10; done
  git sparse-checkout init --cone && git sparse-checkout set "$SUB" || exit 1
  GIT_LFS_SKIP_SMUDGE=1 git checkout -q --detach "$SHA" || exit 1
fi
git lfs install --local > /dev/null && git lfs pull --include "$SUB/**" || { echo "git lfs pull failed" >&2; exit 1; }
ptrs=$(grep -rl -m1 "^version https://git-lfs.github.com/spec/v1" "$SUB" 2>/dev/null | wc -l)
head=$(git rev-parse HEAD); dirty=$(git status --porcelain | wc -l)
echo "HEAD=$head  LFS pointer files left: $ptrs  modified files: $dirty"
[ "$head" = "$SHA" ] && [ "$ptrs" = 0 ] && [ "$dirty" = 0 ] && [ -f "$SUB/scripts/evaluate_policy.py" ] || {
  echo "the stock runtime in $DIR is not a clean $SHA with LFS assets" >&2; exit 1; }
echo "V2D_RUNTIME=$DIR/$SUB"
