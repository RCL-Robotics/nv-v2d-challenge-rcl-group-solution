#!/usr/bin/env bash
# Re-export every training checkpoint of the set with the v2d_t1ready tree's exporter (CPU only) and compare each with
# the evaluated file (compare_exports.py). Writes only under $W/reexport. Never uses a GPU.
set -u
ME=/root/autodl-tmp/haoran.zhang
W=$ME/fc_runs/submissions/t1ready_20261031
FCD=$ME/code/v2d_t1ready/robotic_grounding/flash_chord
PY=$ME/conda_envs/flashchord/bin/python
$PY -I -B - "$W/check_set_pre_hash.json" > $W/reexport/pairs.tsv <<'PYX'
import json, sys
r = json.load(open(sys.argv[1]))
for er in r["episodes"]:
    s = er["stages"][-1]
    print(f"{er['episode']}\t{er['item']}\t{s['checkpoint_file']}\t{s['export_file']}")
PYX
one() {
  ep=$1 item=$2 ck=$3 ev=$4
  O=$W/reexport/$(printf 'ep%02d' $ep)_$item; mkdir -p $O
  [ -f $O/compare.txt ] && return 0
  cd $FCD && env CUDA_VISIBLE_DEVICES='' PYTHONPATH=$FCD/src JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    OMP_NUM_THREADS=2 HUMAN_MOTION_DATA_DIR=$ME/data/human_motion_data TMPDIR=$W/tmp nice -n 15 \
    $PY -B -m flash_chord.training.ppo.coordinate.export --input "$ck" --output $O/$(basename "$ck") \
    --stock-runtime $ME/code/video_to_data_8c778f9/robotic_grounding/flash_chord --report $O/export_report.json --tolerance 0.0001 > $O/export.log 2>&1
  echo "export_exit $?" > $O/compare.txt
  $PY -I -B $FCD/scripts/track2/t1best/compare_exports.py $O/$(basename "$ck") "$ev" >> $O/compare.txt 2>&1; echo "compare_exit $?" >> $O/compare.txt
  rm -f $O/$(basename "$ck")   # keep only reports; the evaluated file stays the pinned one
}
export -f one; export W FCD PY ME
cat $W/reexport/pairs.tsv | xargs -P 2 -L 1 bash -c 'one "$0" "$1" "$2" "$3"'
for d in $W/reexport/ep*_*; do printf "%s\t%s\n" "$(basename $d)" "$(tr '\n' ' ' < $d/compare.txt)"; done > $W/reexport/summary.tsv
