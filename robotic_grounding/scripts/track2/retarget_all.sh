#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Retarget every converted Track 2 episode from SOMA-X to the G1 + Dex3 robot with the in-repo
# scripts/retarget/soma_to_g1.py, in parallel: episodes are spread round-robin over the GPUs and each GPU runs at most
# --jobs-per-gpu of them at a time (the SOMA model runs on the GPU; all 30 episodes on one GPU run out of memory).
# The C3r2 data was made with all episodes at once on 8 GPUs, i.e. about 4 per GPU, the default cap.
# Episodes that already have robot_name=g1/data.parquet are skipped.
#
# usage: retarget_all.sh <motion root> [--gpus 0,1,...] [--jobs-per-gpu 4] [--fps 30] [--episodes all|0,3,...]
#                        [--log-dir DIR] [--dry-run]
#   <motion root>  HUMAN_MOTION_DATA_DIR written by track2_to_soma_inputs.py (it holds whole_body/soma/track2_manifest.json)
#   --log-dir      one log per episode; default $V2D_WORK/logs/retarget, else ./logs/retarget
#   env: RETARGET_PYTHON=python (an Isaac Sim 5.1 / Isaac Lab 2.3.2 env with robotic_grounding installed, see
#        flash_chord/scripts/track2/repro/env/install_retarget.sh)  OMP_THREADS=4
#        RETARGET_JOBS_PER_GPU, RETARGET_FPS: defaults for --jobs-per-gpu and --fps
# Per episode it runs exactly what the C3r2 data was made with (cwd robotic_grounding):
#   CUDA_VISIBLE_DEVICES=<i % #gpus> python scripts/retarget/soma_to_g1.py <motion root>/whole_body/soma/sequence_id=<seq> \
#       --save --fps 30 --motion-root <motion root>
# Output: <motion root>/whole_body/soma/sequence_id=<seq>/robot_name=g1/data.parquet plus object/ (mesh and URDF).
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); RG=$(cd "$HERE/../.." && pwd)   # robotic_grounding/
abs(){ case $1 in /*|"") printf '%s' "$1";; *) printf '%s' "$PWD/$1";; esac; }   # relative to the caller's cwd
HMD=${1:?usage: retarget_all.sh <motion root> [--gpus LIST] [--jobs-per-gpu N] [--fps N] [--episodes all|LIST] [--log-dir DIR] [--dry-run]}; shift
GPUS=0; EPISODES=all; LOGDIR=""; DRY=0; JPG=${RETARGET_JOBS_PER_GPU:-4}; FPS=${RETARGET_FPS:-30}
while [ $# -gt 0 ]; do case "$1" in
  --gpus) GPUS=$2; shift 2;; --jobs-per-gpu) JPG=$2; shift 2;; --fps) FPS=$2; shift 2;;
  --episodes) EPISODES=$2; shift 2;; --log-dir) LOGDIR=$2; shift 2;;
  --dry-run) DRY=1; shift;; *) echo "unknown argument $1" >&2; exit 2;; esac; done
[ "$JPG" -ge 1 ] 2>/dev/null || { echo "--jobs-per-gpu must be a positive integer" >&2; exit 2; }
PY=${RETARGET_PYTHON:-python}; LOGDIR=${LOGDIR:-${V2D_WORK:-.}/logs/retarget}
HMD=$(abs "$HMD"); LOGDIR=$(abs "$LOGDIR"); case $PY in */*) PY=$(abs "$PY");; esac
PYJSON=$(command -v python3 || command -v python || echo "$PY")   # any Python 3 reads the manifest
export OMP_NUM_THREADS=${OMP_THREADS:-4} MKL_NUM_THREADS=${OMP_THREADS:-4} OPENBLAS_NUM_THREADS=${OMP_THREADS:-4}
MANIFEST=$HMD/whole_body/soma/track2_manifest.json
if [ ! -f "$MANIFEST" ]; then
  [ $DRY = 1 ] || { echo "missing $MANIFEST (run track2_to_soma_inputs.py first)" >&2; exit 1; }
  echo "[retarget] (dry run) $MANIFEST does not exist yet: one soma_to_g1.py call per converted episode, as below"
  printf '[retarget] cd %q && CUDA_VISIBLE_DEVICES=<gpu> %q scripts/retarget/soma_to_g1.py %q --save --fps %q --motion-root %q\n' \
    "$RG" "$PY" "$HMD/whole_body/soma/sequence_id=<seq>" "$FPS" "$HMD"
  exit 0
fi
# sequences in episode order (the manifest maps sequence_id -> {episode_index, object, ...})
SEQS=$("$PYJSON" - "$MANIFEST" "$EPISODES" <<'PY'
import json, sys
manifest, episodes = json.load(open(sys.argv[1])), sys.argv[2]
wanted = None if episodes == "all" else {int(e) for e in episodes.replace(",", " ").split()}
for ep, seq in sorted((int(r["episode_index"]), s) for s, r in manifest.items()):
    if wanted is None or ep in wanted:
        print(seq)
PY
) || exit 1
IFS=, read -r -a GPU_LIST <<< "$GPUS"
todo=(); for s in $SEQS; do
  if [ -f "$HMD/whole_body/soma/sequence_id=$s/robot_name=g1/data.parquet" ]; then
    [ $DRY = 1 ] && { echo "# $s already retargeted: would be skipped"; todo+=("$s"); }   # the dry run lists every command
  else todo+=("$s"); fi
done
if [ $DRY = 1 ]; then echo "[retarget] dry run: ${#todo[@]} episodes, gpus $GPUS, at most $JPG per GPU, logs in $LOGDIR"
else echo "[$(date +%T)] to retarget: ${#todo[@]} (already done: $(( $(echo $SEQS | wc -w) - ${#todo[@]} ))), gpus $GPUS, at most $JPG per GPU"; fi
run1(){ # seq gpu
  local cmd=(env "CUDA_VISIBLE_DEVICES=$2" "$PY" scripts/retarget/soma_to_g1.py "$HMD/whole_body/soma/sequence_id=$1"
             --save --fps "$FPS" --motion-root "$HMD")
  if [ "$DRY" = 1 ]; then printf '[retarget] cd %q && ' "$RG"; printf '%q ' "${cmd[@]}"; echo; return 0; fi
  local t0 rc; t0=$(date +%s); ( cd "$RG" && "${cmd[@]}" ) > "$LOGDIR/$1.log" 2>&1; rc=$?
  echo "$1 gpu$2 exit=$rc $(( $(date +%s) - t0 ))s"
}
if [ $DRY = 1 ]; then
  i=0; for s in "${todo[@]}"; do run1 "$s" "${GPU_LIST[$(( i % ${#GPU_LIST[@]} ))]}"; i=$((i + 1)); done
  exit 0
fi
mkdir -p "$LOGDIR"
export -f run1; export RG PY HMD FPS LOGDIR DRY
for gi in "${!GPU_LIST[@]}"; do   # one queue per GPU (round-robin assignment), at most JPG episodes at a time on it
  mine=(); for i in "${!todo[@]}"; do [ $(( i % ${#GPU_LIST[@]} )) = "$gi" ] && mine+=("${todo[$i]}"); done
  [ ${#mine[@]} -gt 0 ] || continue
  printf '%s\n' "${mine[@]}" | xargs -P "$JPG" -I{} bash -c 'run1 "$1" "$2"' _ {} "${GPU_LIST[$gi]}" &
done
wait
missing=0; for s in $SEQS; do [ -f "$HMD/whole_body/soma/sequence_id=$s/robot_name=g1/data.parquet" ] || { echo "MISSING: $s (see $LOGDIR/$s.log)"; missing=$((missing + 1)); }; done
echo "[$(date +%T)] RETARGET_ALL_DONE ($missing missing)"; [ $missing = 0 ]
