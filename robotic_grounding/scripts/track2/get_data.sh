#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Download the V2D challenge Track 2 data from Hugging Face (dataset nvidia/video_to_data_challenge).
#
# usage: get_data.sh <dataset root> [--dry-run]
#   env: DATA_REPO=nvidia/video_to_data_challenge
#        DATA_REVISION=5f68335f3acc802033d1e80728c1633197521de8   (the revision every file of the C3r2 run came from)
#        DATA_INCLUDE='track_2/*'      (what the C3r2 run downloaded: both tiers, 313 files, 2.1 GB;
#                                       Tier 1 alone is 'track_2/tier_1_multiview_caption/*')
#        HF_CLI=hf  ATTEMPTS=30  MAX_WORKERS=4
# The dataset is gated: log in first (`hf auth login`) or export HF_TOKEN in your shell. This script never
# reads, prints or stores a token. Behind a slow network, set HF_ENDPOINT to a mirror and HF_HUB_DISABLE_XET=1
# yourself (optional; the original run used both because the Xet backend refused the mirror).
# The same <dataset root> (the directory that contains track_2/) is later passed to the packer as --dataset-root.
set -u
ROOT=${1:?usage: get_data.sh <dataset root> [--dry-run]}; DRY=${2:-}
DATA_REPO=${DATA_REPO:-nvidia/video_to_data_challenge}
DATA_REVISION=${DATA_REVISION:-5f68335f3acc802033d1e80728c1633197521de8}
DATA_INCLUDE=${DATA_INCLUDE:-track_2/*}
HF_CLI=${HF_CLI:-hf}; ATTEMPTS=${ATTEMPTS:-30}; MAX_WORKERS=${MAX_WORKERS:-4}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-60} HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-30}
CMD=("$HF_CLI" download "$DATA_REPO" --repo-type dataset --revision "$DATA_REVISION" --include "$DATA_INCLUDE"
     --local-dir "$ROOT" --max-workers "$MAX_WORKERS")
if [ "$DRY" = --dry-run ]; then printf '[data] '; printf '%q ' "${CMD[@]}"; echo; exit 0; fi
mkdir -p "$ROOT"
for i in $(seq 1 "$ATTEMPTS"); do
  echo "[$(date +%T)] download attempt $i"
  "${CMD[@]}" && { echo DATA_DONE; break; }
  find "$ROOT/.cache" -name "*.incomplete" -size 0 -delete 2>/dev/null
  [ "$i" = "$ATTEMPTS" ] && { echo "download failed after $ATTEMPTS attempts" >&2; exit 1; }
  sleep 10
done
echo "files under $ROOT/track_2: $(find "$ROOT/track_2" -type f | wc -l) (both tiers: 313)  size: $(du -sh "$ROOT/track_2" | cut -f1)"
# every downloaded file records the revision it came from (hf download metadata)
revs=$(find "$ROOT/.cache/huggingface/download" -name "*.metadata" -exec head -1 {} \; 2>/dev/null | sort | uniq -c)
echo "revisions in the download metadata: ${revs:-none}"
