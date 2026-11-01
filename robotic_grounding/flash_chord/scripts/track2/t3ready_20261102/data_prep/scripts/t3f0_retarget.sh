#!/bin/bash
# t3f0_retarget.sh EP SEQ GPU_UUID CPUS NTH LOG STATUS: retarget ONE Tier-3 f0 episode SOMA-X -> G1. NEW FILE (2026-10-24),
# a copy of scripts/t3opus_retarget.sh (left untouched) with only the data root (data/human_motion_data_tier3_f0) and the
# marker variables (T3F0_RETARGET/T3F0_EP) changed: scripts/retarget/soma_to_g1.py <data_folder> --save --fps 30
# --motion-root <root>, nv repo robotic_grounding @ f2923d61 (clean tree), v2d env from activate.sh, clean environment
# (env -i ... bash --noprofile --norc), plain CUDA context (never MPS), nice 10, taskset CPUS. Refuses when the parquet
# exists already or the GPU has an MPS server. Called by scripts/t3f0_retarget_all.sh.
set -u
EP=$1; SEQ=$2; GPU_UUID=$3; CPUS=$4; NTH=$5; LOG=$6; ST=$7
ME=/root/autodl-tmp/haoran.zhang; DATA=$ME/data/human_motion_data_tier3_f0
NV=$ME/code/nv-v2d-challenge-codebase/robotic_grounding; NV_PIN=f2923d61817e61eb37d939e0f75e5b4a7aaa5cbf
DF=$DATA/whole_body/soma/sequence_id=$SEQ
echo "start $(date '+%F %T') ep$EP $SEQ gpu $GPU_UUID cpus $CPUS threads $NTH" > $ST
[ -d "$DF" ] || { echo "REFUSED: no data folder $DF" >> $ST; exit 3; }
[ -e "$DF/robot_name=g1/data.parquet" ] && { echo "REFUSED: parquet exists (never retarget over it)" >> $ST; exit 3; }
[ -e "$DATA/READY" ] && { echo "REFUSED: $DATA/READY exists (write-once root)" >> $ST; exit 3; }
if nvidia-smi --query-compute-apps=gpu_uuid,process_name --format=csv,noheader | grep "$GPU_UUID" | grep -qi mps; then
  echo "REFUSED: GPU $GPU_UUID has an MPS server" >> $ST; exit 3; fi
source $ME/activate.sh >/dev/null 2>&1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY PYTHONPATH PYTHONHOME PYTHONSTARTUP
for v in $(env | grep -o '^CUDA_MPS_[A-Z_]*'); do unset $v; done
export HUMAN_MOTION_DATA_DIR=$DATA OMP_NUM_THREADS=$NTH MKL_NUM_THREADS=$NTH OPENBLAS_NUM_THREADS=$NTH PYTHONDONTWRITEBYTECODE=1 \
  T3F0_RETARGET=1 T3F0_EP=$EP
PY=$(which python)
[ "$PY" = $ME/conda_envs/v2d/bin/python ] || { echo "REFUSED: python is $PY, not the v2d env" >> $ST; exit 3; }
cd $NV || { echo "REFUSED: no $NV" >> $ST; exit 3; }
HEAD=$(git rev-parse HEAD 2>/dev/null); DIRTY=$(GIT_OPTIONAL_LOCKS=0 git status --porcelain --untracked-files=no 2>/dev/null | wc -l)
echo "python $PY; XDG_CACHE_HOME=${XDG_CACHE_HOME:-unset}; nv HEAD $HEAD tracked-modified $DIRTY; MPS vars $(env | grep -c '^CUDA_MPS_')" >> $ST
{ [ "$HEAD" = $NV_PIN ] && [ "$DIRTY" = 0 ]; } || { echo "REFUSED: nv repo not at $NV_PIN or tracked changes" >> $ST; exit 3; }
t0=$(date +%s)
CUDA_VISIBLE_DEVICES=$GPU_UUID nice -n 10 taskset -c $CPUS timeout -k 60 1800 \
  python scripts/retarget/soma_to_g1.py $DF --save --fps 30 --motion-root $DATA > $LOG 2>&1
rc=$?
echo "exit $rc after $(( $(date +%s) - t0 )) s at $(date '+%F %T')" >> $ST
echo "DONE" >> $ST
exit $rc
