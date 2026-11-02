#!/usr/bin/env bash
# Re-export every training checkpoint of the Tier-2 set R6-CLOSEST with the v2d_t2ready tree's exporter (CPU only,
# nice 15, eval-lane cores) at the tolerance recorded for its evaluated file, and compare the result with the evaluated
# file (t1best/compare_exports.py: tensors + parsed metadata). Writes only under $W/reexport. Never uses a GPU.
set -u
ME=/root/autodl-tmp/haoran.zhang
W=$ME/fc_runs/submissions/t2ready_20261031
FCD=$ME/code/v2d_t2ready/robotic_grounding/flash_chord
PY=$ME/conda_envs/flashchord/bin/python
$PY -I -B - "$FCD/scripts/track2/t2ready_20261031/set/set_items.json" > $W/reexport/pairs.tsv <<'PYX'
import json, sys
d = json.load(open(sys.argv[1])); roots = d["placeholders"]
def real(s):
    for k, v in roots.items():
        s = s.replace(k, v)
    return s
for e in d["episodes"]:
    print(f"{e['episode']}\t{e['job']}\t{real(e['checkpoint']['file'])}\t{real(e['export']['file'])}\t{e['export']['tolerance']}")
PYX
one() {
  ep=$1 job=$2 ck=$3 ev=$4 tol=$5
  O=$W/reexport/$job; mkdir -p $O
  [ -f $O/compare.txt ] && return 0
  cd $FCD && env CUDA_VISIBLE_DEVICES='' PYTHONPATH=$FCD/src JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 XLA_PYTHON_CLIENT_PREALLOCATE=false \
    OMP_NUM_THREADS=2 TMPDIR=$W/tmp nice -n 15 taskset -c 48-51,152-155 \
    $PY -B -m flash_chord.training.ppo.coordinate.export --input "$ck" --output $O/$(basename "$ck") \
    --stock-runtime $ME/code/video_to_data_8c778f9/robotic_grounding/flash_chord --report $O/export_report.json --tolerance $tol > $O/export.log 2>&1
  echo "export_exit $?" > $O/compare.txt.tmp
  $PY -I -B $FCD/scripts/track2/t1best/compare_exports.py $O/$(basename "$ck") "$ev" >> $O/compare.txt.tmp 2>&1; echo "compare_exit $?" >> $O/compare.txt.tmp
  mv $O/compare.txt.tmp $O/compare.txt
  rm -f $O/$(basename "$ck")   # keep only reports; the evaluated file stays the pinned one
}
export -f one; export W FCD PY ME
cat $W/reexport/pairs.tsv | xargs -P 2 -L 1 bash -c 'one "$0" "$1" "$2" "$3" "$4"'
for d in $W/reexport/ep*_*; do printf "%s\t%s\n" "$(basename $d)" "$(tr '\n' ' ' < $d/compare.txt)"; done > $W/reexport/summary.tsv
echo REEXPORT_DONE
