#!/usr/bin/env bash
# Offline reproduction: run the agent's solve.py with no network, output to scratch, compare with the delivered npz.
set -uo pipefail
ws=$1; n=$2; V2D=${V2D_ROOT:?set V2D_ROOT to the data root (reproducibility/README.md)}; REPO=$(cd "$(dirname "$0")/../.." && pwd); A=${APPTAINER:-apptainer}
ep=$(printf episode_%06d $n)
src=$(python3 -c "import json;print(json.load(open('$V2D/logs/JC2s1_shared_f820_pp_align.json'))['$n']['src'])")
if [ "$src" = M1 ]; then pl=$V2D/experiments/M1_prompt_v2/out/$ep
elif [ -f $V2D/outputs/cari4d/$ep/inference/coconet.pth ]; then pl=$V2D/outputs/cari4d/$ep
else pl=$V2D/outputs/cari4d_failed/${ep}_out; fi
mkdir -p $ws/scratch/repro
export APPTAINERENV_CUDA_VISIBLE_DEVICES=0 APPTAINERENV_PYTHONUSERBASE=/work/scratch/pyuser APPTAINERENV_HF_HUB_OFFLINE=1 APPTAINERENV_HF_HOME=/weights/hf_home APPTAINERENV_TORCH_HOME=/weights/sam3d_body/torch_home
s=$(date +%s)
$A exec --nv --containall --net --network none --home /work/scratch --pwd /work/outputs -B $ws:/work -B $REPO:/code:ro -B $V2D/weights:/weights:ro -B $pl:/pipeline:ro $V2D/images/v2d_cari4d.sif \
  python /work/outputs/solve.py --inputs /work/inputs --out /work/scratch/repro/object_poses_world.npz > $ws/scratch/repro/log.txt 2>&1; echo "exit $? $(( $(date +%s)-s ))s"
python3 -c "
import numpy as np;a=np.load('$ws/outputs/object_poses_world.npz')['T_world_mesh'];b=np.load('$ws/scratch/repro/object_poses_world.npz')['T_world_mesh']
print('shape',a.shape,b.shape,'max trans diff mm',np.abs(a[:,:3,3]-b[:,:3,3]).max()*1000,'max rot elem diff',np.abs(a[:,:3,:3]-b[:,:3,:3]).max())"
