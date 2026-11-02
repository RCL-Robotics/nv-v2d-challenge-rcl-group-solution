#!/usr/bin/env bash
# Tier 2 refinement: fit the exo camera in the Tier 2 world from the Tier 2 human (+ object track), map our image-side
# object rotations (Opus agent, s1) into it, and fuse them with the Tier 2 trajectory. No GT is used to build the result
# (fuse_v2.py also prints a GT diagnostic). GPU only for the SOMA decoder (a few minutes in total).
set -euo pipefail
V=${V2D_ROOT:?set V2D_ROOT to the data root (reproducibility/README.md)}; H=$(cd "$(dirname "$0")" && pwd); A=${APPTAINER:-apptainer}
RUN="$A exec --nv -B $V -B $H/.. $V/images/v2d_cari4d.sif python"
export APPTAINERENV_CUDA_VISIBLE_DEVICES=0
${T2_PY:-python3} $H/extract_t2_human.py $V/dataset/track_2/tier_2_synthetic_noise $V/tier2_human
$RUN $H/cam_from_human.py $V/logs/t2r_cam_human.json
$RUN $H/cam_joint.py $V/logs/t2r_cam_human.json $V/logs/t2r_cam_joint_bal.json
$RUN $H/fuse_v2.py $V/logs/t2r_cam_joint_bal.json $V/experiments/tier2_refine/jointbal_45_5_1_0 45 5 1 0   # tau, window, flips, interp
