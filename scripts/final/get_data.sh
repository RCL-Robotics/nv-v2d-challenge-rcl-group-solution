#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# get_data.sh: the challenge data, Hugging Face dataset nvidia/video_to_data_challenge at the revision every counted
# file used (5f68335f3acc802033d1e80728c1633197521de8), into $V2D_DATA (the directory that contains track_2/; the kit
# packer's --dataset-root). It wraps robotic_grounding/scripts/track2/get_data.sh (unchanged). The dataset is gated:
# log in first (`hf auth login`) or export HF_TOKEN; nothing here reads or stores a token.
#   Tier 1 reads track_2/tier_1_multiview_caption and Tier 2 only track_2/tier_2_synthetic_noise. Tier 3's sources
#   (video, mesh, ground plane, frame-0 human and object pose) come with the Tier-3 reconstruction pipeline (build_inputs_tier3.sh).
# The derived inputs (G1 retargets) are built by TR/repro (Tier 1), build_inputs_tier2.sh and build_inputs_tier3.sh.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
usage() { cat <<'EOF'
usage: get_data.sh [--include 'track_2/*'] [--mirror cn] [--verify-only] [--dry-run] [--site FILE]
  --include GLOB   what to download (default 'track_2/*': both public tiers, 313 files, 2.1 GB)
  --mirror cn      optional: HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 (used on the training server)
  --verify-only    only count the files under $V2D_DATA/track_2 and list the recorded download revisions
needs: the huggingface_hub CLI `hf` (HF_CLI=... to use another binary)
EOF
}
final_init "$@"
INCLUDE='track_2/*'; MIRROR=""; VERIFY=0
set -- ${REST[@]+"${REST[@]}"}
while [ $# -gt 0 ]; do
  case "$1" in
    --include) INCLUDE=$2; shift ;; --mirror) MIRROR=$2; shift ;; --verify-only) VERIFY=1 ;;
    *) die "unknown option $1 (see --help)" ;;
  esac; shift
done
plan_header "challenge data into $V2D_DATA"
if [ "$VERIFY" = 0 ]; then
  [ "$MIRROR" = cn ] && add 'export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1'
  addq env DATA_REVISION=5f68335f3acc802033d1e80728c1633197521de8 DATA_INCLUDE="$INCLUDE" \
    bash "$REPO/robotic_grounding/scripts/track2/get_data.sh" "$V2D_DATA"
fi
add 'n=$(find "$V2D_DATA/track_2" -type f | wc -l); note "files under $V2D_DATA/track_2: $n (both public tiers at 5f68335f: 313)"'
add 'for d in tier_1_multiview_caption tier_2_synthetic_noise; do note "$d: $(find "$V2D_DATA/track_2/$d" -type f | wc -l) files"; done'
add 'meta=$(find "$V2D_DATA/.cache/huggingface/download" -name "*.metadata" | head -400 | xargs -r head -q -n1 | sort | uniq -c)'
add 'note "download revisions recorded by hf: ${meta:-none (no hf download metadata here)}"'
printf '%s' "$PLAN" | run_plan get_data
