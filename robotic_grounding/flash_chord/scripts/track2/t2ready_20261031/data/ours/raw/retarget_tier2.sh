#!/bin/bash
# Retarget all Tier-2 (synthetic noise) episodes SOMA-X -> G1 into human_motion_data_tier2.
# Same command as Tier 1 (TRACK2_PIPELINE_GUIDE.md section 4), different --motion-root.
# Usage: retarget_tier2.sh "<gpu list, e.g. 5>" <parallel jobs>
ME=/root/autodl-tmp/haoran.zhang; HMD=$ME/data/human_motion_data_tier2; LOG=$ME/tmp_analysis/tier2/retarget_logs
GPUS=(${1:-5}); PAR=${2:-6}
mkdir -p $LOG
source $ME/activate.sh >/dev/null 2>&1; unset http_proxy https_proxy CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
export HUMAN_MOTION_DATA_DIR=$HMD OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
cd $ME/code/nv-v2d-challenge-codebase/robotic_grounding
SEQS=$(python -c "import json; m=json.load(open('$HMD/whole_body/soma/track2_manifest.json')); print(' '.join(sorted(m, key=lambda k: -m[k]['frames'])))")
i=0; todo=""
for s in $SEQS; do [ -f "$HMD/whole_body/soma/sequence_id=$s/robot_name=g1/data.parquet" ] || todo="$todo $s"; done
echo "[$(date +%T)] to retarget: $(echo $todo | wc -w) on GPUs ${GPUS[*]} x$PAR"
run1(){ local s=$1 g=$2; local t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$g nice -n 10 python scripts/retarget/soma_to_g1.py $HMD/whole_body/soma/sequence_id=$s \
     --save --fps 30 --motion-root $HMD > $LOG/$s.log 2>&1
  echo "[$(date +%T)] $s gpu=$g exit=$? $(( $(date +%s)-t0 ))s"; }
for s in $todo; do
  while [ $(jobs -rp | wc -l) -ge $PAR ]; do sleep 2; done
  run1 $s ${GPUS[$((i % ${#GPUS[@]}))]} & i=$((i+1)); sleep 3
done
wait
echo "[$(date +%T)] RETARGET_TIER2_DONE"
