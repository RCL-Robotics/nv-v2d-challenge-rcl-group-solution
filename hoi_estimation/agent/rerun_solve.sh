#!/usr/bin/env bash
# Re-run the released Opus v2 agent's final solver for one episode, without the agent (no LLM, no network), in the same
# sandbox layout as run_claude.sh: /work = workspace, /pipeline = the episode's pipeline folder, /weights read-only.
# The solver and the files next to it come from agent_records/<run>/outputs/ of this checkout (run CL2 for episodes
# 0-10, CL2b 11-21, CL2c 22-29, as in build_release_opus.py); the inputs are the exported agent inputs
# $V2D_ROOT/agent_runs/T1_ds_ep<eee>/inputs (export_inputs.py, tier3_runbook.md A1).
# Usage: V2D_ROOT=<data root> rerun_solve.sh <episode> <out_dir>    -> <out_dir>/<run>/outputs/object_poses_world.npz
# Env: APPTAINER (default apptainer), GPU (default 0).
set -euo pipefail
n=$1; OUT=$2
HERE=$(cd "$(dirname "$0")" && pwd); REC=$HERE/../agent_records
V2D=${V2D_ROOT:?set V2D_ROOT to the data root (reproducibility/README.md)}; A=${APPTAINER:-apptainer}; GPU=${GPU:-0}
e=$(printf %03d $n); ep=$(printf episode_%06d $n)
run=$([ $n -le 10 ] && echo CL2 || { [ $n -le 21 ] && echo CL2b || echo CL2c; })_ep$e
# pipeline folder: same rule as run_batch_CL.sh
src=$(python3 -c "import json;print(json.load(open('$V2D/logs/JC2s1_shared_f820_pp_align.json'))['$n']['src'])")
if [ "$src" = M1 ]; then pl=$V2D/experiments/M1_prompt_v2/out/$ep
elif [ -f $V2D/outputs/cari4d/$ep/inference/coconet.pth ]; then pl=$V2D/outputs/cari4d/$ep
else pl=$V2D/outputs/cari4d_failed/${ep}_out; fi
W=$OUT/$run; rm -rf "$W"; mkdir -p "$W/outputs" "$W/scratch"
cp -rL "$V2D/agent_runs/T1_ds_ep$e/inputs" "$W/"
cp -r "$REC/$run/outputs/." "$W/outputs/"; rm -f "$W/outputs/object_poses_world.npz"   # released result: kept in agent_records
APPTAINERENV_CUDA_VISIBLE_DEVICES=$GPU APPTAINERENV_HF_HOME=/weights/hf_home APPTAINERENV_HF_HUB_OFFLINE=1 \
APPTAINERENV_TORCH_HOME=/weights/sam3d_body/torch_home APPTAINERENV_PYTHONUSERBASE=/work/scratch/pyuser \
  $A exec --nv --containall --home /work/scratch --pwd /work -B "$W:/work" -B "$(readlink -f "$V2D/weights"):/weights:ro" \
    -B "$(readlink -f "$pl"):/pipeline:ro" "$V2D/images/v2d_cari4d.sif" \
    python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz > "$W/solve_stdout.txt" 2>&1
python3 - "$W/outputs/object_poses_world.npz" "$REC/$run/outputs/object_poses_world.npz" <<'PY'
import sys, numpy as np
a, b = (np.load(p)["T_world_mesh"].astype(float) for p in sys.argv[1:])
print(f"{sys.argv[1]}: frames {len(a)}, vs released max |dT| rot {np.abs(a[:, :3, :3] - b[:, :3, :3]).max():.2e}, "
      f"trans {np.abs(a[:, :3, 3] - b[:, :3, 3]).max() * 100:.3f} cm")
PY
