#!/usr/bin/env bash
# Build side-by-side preview videos (left: input video, right: CARI4D reconstruction) for
# every PASS episode on NAS. CARI4D's *_reconstruction.mp4 already has this layout; we
# downscale it to 1920x720 and add episode/object labels. Skips videos already made.
# Usage: make_videos.sh            (one pass)
#        make_videos.sh --watch    (repeat every 10 min until the local run + retry finish)
set -uo pipefail
V2D=${V2D_ROOT:-/mnt/nas/Group/hanzhou/v2d}
OUT=$V2D/videos
FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
mkdir -p "$OUT"

one_pass() {
  for d in "$V2D"/outputs/cari4d/episode_*; do
    [ -d "$d" ] || continue
    local ep=${d##*_} idx obj src dst
    grep -qs '"verdict": "PASS"' "$d/pipeline_report.json" || continue
    src=$(ls "$d"/inference/*_reconstruction.mp4 2>/dev/null | head -1); [ -n "$src" ] || continue
    idx=$((10#$ep))
    obj=$(awk -F'\t' -v i="$idx" '$1==i{print $3}' "$V2D/code/pipeline/episodes.tsv")
    dst=$OUT/episode_${ep}_${obj}.mp4
    [ -s "$dst" ] && continue
    ffmpeg -v error -y -i "$src" -vf "scale=1920:720,\
drawtext=fontfile=$FONT:text='episode $ep  |  $obj':x=20:y=20:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8,\
drawtext=fontfile=$FONT:text='Input':x=20:y=h-50:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8,\
drawtext=fontfile=$FONT:text='Reconstruction':x=w/2+20:y=h-50:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8" \
      -c:v libx264 -crf 23 -preset medium -pix_fmt yuv420p -movflags +faststart "$dst.tmp.mp4" \
      && mv "$dst.tmp.mp4" "$dst" && echo "made $dst"
  done
}

if [ "${1:-}" = "--watch" ]; then
  while true; do
    one_pass
    grep -qs RUN_LOCAL_DONE "$V2D/logs/run_local_retry.log" && { one_pass; echo "watch done $(date '+%F %T')"; break; }
    sleep 600
  done
else
  one_pass
fi
