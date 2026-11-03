#!/usr/bin/env bash
# Re-run single steps of the Tier 3 chain from this checkout and compare each result with the released file.
# Every step writes under a new name, so the released files are only read:
#   $V2D_ROOT/experiments/R_<variant>/, $V2D_ROOT/logs/R_<variant>_align.json,
#   $V2D_ROOT/experiments/t3_frame0/R_opus_f3_c2_shift/, $V2D_ROOT/repro_out/
# Usage: V2D_ROOT=<data root, writable> bash check_steps.sh [step ...]
#   steps (default: all, in this order): p1 ff2 jc2s1 db release export solve frame0 opus combined fit supportpkg
# Env:
#   SIF_EXEC     apptainer binary (default: apptainer)
#   GPU          GPU index for the GPU steps (default 0). The images are built for sm_80-sm_90 (Ampere/Ada/Hopper).
#   HOST_PY      python for db / opus / combined; these steps ran on the host, not in the image. Bit-identical
#                results need the host versions of the original runs: python 3.12.13, numpy 2.5.2, torch 2.10.0,
#                scipy 1.17.0, trimesh 4.11.1 (environment/pip_freeze_host_venv.txt). Default: python3.
#   EXTRA_BINDS  extra apptainer -B options, e.g. when $V2D_ROOT holds symlinks into another directory.
# Results are appended to $V2D_ROOT/repro_out/check_results.txt.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$HERE"); SC=$REPO/hoi_extraction/scripts
N=${V2D_ROOT:?set V2D_ROOT to the data root}; export V2D_ROOT=$N APPTAINERENV_V2D_ROOT=$N
A=${SIF_EXEC:-apptainer}; SIF=$N/images/v2d_cari4d.sif; GPU=${GPU:-0}; PY=${HOST_PY:-python3}
read -ra XB <<< "${EXTRA_BINDS:-}"; BB=(-B "$N" -B "$REPO" "${XB[@]}")
O=$N/repro_out; mkdir -p "$O"; LOG=$O/check_results.txt
ep() { printf "episode_%06d" "$1"; }
PW=()   # optional --pwd for the image calls
RUN()   { $A exec "${BB[@]}" "${PW[@]}" "$SIF" python "$@"; }                               # CPU, image
RUNG()  { APPTAINERENV_CUDA_VISIBLE_DEVICES=$GPU $A exec --nv "${BB[@]}" "${PW[@]}" "$SIF" "$@"; }   # GPU, image
RUNGP() { RUNG bash -c 'export PYTHONPATH=/workspace/v2d_cari4d/lib/cari4d:/workspace/v2d_sam3d_body/lib:$PYTHONPATH
  export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH; exec "$@"' _ "$@"; }  # GPU, image, SOMA/MHR libs
say() { echo "$*" | tee -a "$LOG"; }
# cmpf <label> <rerun file> <released file>: byte compare; on a difference print the per-leaf differences
cmpf() {
  if cmp -s "$2" "$3"; then say "$1 BYTE-IDENTICAL"
  else say "$1 DIFFERS"; case $2 in *.pth|*.npz) RUN "$HERE/compare_bundles.py" "$2" "$3" 2>&1 | grep -v INFO | tee -a "$LOG";; esac; fi
}
# cmptree <label> <rerun dir> <released dir> [find args]: byte compare of every file of the released dir
cmptree() {
  local l=$1 a=$2 b=$3 t=0 s=0 f; shift 3; local diff=()
  while IFS= read -r f; do t=$((t + 1)); if cmp -s "$a/$f" "$b/$f"; then s=$((s + 1)); else diff+=("$f"); fi
  done < <(cd "$b" && find . -type f "$@" | sort)
  say "$l $s/$t files byte-identical${diff:+; differ: ${diff[*]}}"
}
# cmpjson <label> <rerun json> <released json> <keys...>: max |diff| of every numeric field of the given entries
cmpjson() {
  local l=$1 a=$2 b=$3; shift 3
  python3 - "$a" "$b" "$@" <<'PY' | sed "s/^/$l /" | tee -a "$LOG"
import json, sys
import numpy as np
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
for k in sys.argv[3:]:
    out = []
    for f in a[k]:
        if f == "bundle": continue  # absolute path of the machine that ran it
        try: out.append(f"{f} {np.abs(np.array(a[k][f], float) - np.array(b[k][f], float)).max():.2g}")
        except (TypeError, ValueError): out.append(f"{f} {'same' if a[k][f] == b[k][f] else 'DIFFERS'}")
    print(k, "max|d|:", ", ".join(out))
PY
}

step_p1() {      # P1_temporal, CPU, image (episodes with an E2 source: 2, and 25 = the failed baseline)
  (cd "$SC" && RUN postproc_temporal.py 2 25 --out-exp R_P1_temporal)
  for e in 2 25; do cmpf "p1 $(ep $e)" "$N/experiments/R_P1_temporal/$(ep $e)/coconet.pth" "$N/experiments/P1_temporal/$(ep $e)/coconet.pth"; done
}
step_ff2() {     # FF2_feet_ref, GPU (SOMA decode), image; 3 = M1 source, 25 = P1 source
  (cd "$SC" && RUNGP python postproc_feet_floor.py 3 25 --use-sim --ref-foot-height --align default="$N/logs/pnp_merged.json" --out-exp R_FF2_feet_ref)
  for e in 3 25; do cmpf "ff2 $(ep $e)" "$N/experiments/R_FF2_feet_ref/$(ep $e)/coconet.pth" "$N/experiments/FF2_feet_ref/$(ep $e)/coconet.pth"; done
  cmpjson "ff2 feet_floor.json" "$N/experiments/R_FF2_feet_ref/feet_floor.json" "$N/experiments/FF2_feet_ref/feet_floor.json" 3 25
}
step_jc2s1() {   # JC2s1_shared_f820_pp, GPU, image; 0 = CIAI baseline, 25 = P1 source
  (cd "$SC" && RUNG python postproc_joint_camera_K.py 0 25 --out-exp R_JC2s1_shared_f820_pp --sig-logf 1e-4 --sig-pp 0.01 --K-prior 820 758.1 560.5)
  for e in 0 25; do cmpf "jc2s1 $(ep $e)" "$N/experiments/R_JC2s1_shared_f820_pp/$(ep $e)/coconet.pth" "$N/experiments/JC2s1_shared_f820_pp/$(ep $e)/coconet.pth"; done
  cmpjson "jc2s1 align.json" "$N/logs/R_JC2s1_shared_f820_pp_align.json" "$N/logs/JC2s1_shared_f820_pp_align.json" 0 25
}
step_db() {      # DB_JC2s1_shared_f820_pp, CPU, host python (from the released JC2s1 bundles)
  (cd "$SC" && $PY postproc_depth_blend.py 0 25 --src JC2s1_shared_f820_pp --out-exp R_DB_JC2s1_shared_f820_pp)
  for e in 0 25; do cmpf "db $(ep $e)" "$N/experiments/R_DB_JC2s1_shared_f820_pp/$(ep $e)/coconet.pth" "$N/experiments/DB_JC2s1_shared_f820_pp/$(ep $e)/coconet.pth"; done
  cmpjson "db depth_blend.json" "$N/experiments/R_DB_JC2s1_shared_f820_pp/depth_blend.json" "$N/experiments/DB_JC2s1_shared_f820_pp/depth_blend.json" 0 25
}
step_release() { # build_release.py (s1), GPU (SOMA pose inversion), image
  rm -rf "$O/tier3_s1_eps0_25"; (cd "$SC" && RUNG python build_release.py "$O/tier3_s1_eps0_25" --episodes 0 25)
  local s; for s in $(ls "$O/tier3_s1_eps0_25/reconstructed_data"); do
    cmptree "release $s" "$O/tier3_s1_eps0_25/reconstructed_data/$s" "$N/release/tier3_s1_20261020/reconstructed_data/$s"
    cmpf "release $s/soma_params.npz" "$O/tier3_s1_eps0_25/reconstructed_data/$s/soma_params.npz" "$N/release/tier3_s1_20261020/reconstructed_data/$s/soma_params.npz"
  done
}
step_export() {  # agent/export_inputs.py, GPU, image; 25 = masks from outputs/cari4d_failed
  rm -rf "$O/T1_ds_ep025"; (cd "$REPO/agent" && RUNG python3 export_inputs.py 25 --ws "$O/T1_ds_ep025" --variant DB_JC2s1_shared_f820_pp \
    --align "$N/logs/JC2s1_shared_f820_pp_align.json" --camera JC2s1_shared_f820_pp)
  local f; for f in $(ls "$N/agent_runs/T1_ds_ep025/inputs"); do cmpf "export ep025 $f" "$O/T1_ds_ep025/inputs/$f" "$N/agent_runs/T1_ds_ep025/inputs/$f"; done
}
step_solve() {   # the released agent's solve.py (CL2_ep002), GPU, image, same sandbox as run_claude.sh (minus network)
  local W=$O/ws_CL2_ep002; rm -rf "$W"; mkdir -p "$W/outputs" "$W/scratch"
  cp -rL "$N/agent_runs/T1_ds_ep002/inputs" "$W/"; cp "$N/agent_runs/CL2_ep002/outputs/"{solve.py,hoilib.py} "$W/outputs/"
  APPTAINERENV_CUDA_VISIBLE_DEVICES=$GPU APPTAINERENV_HF_HOME=/weights/hf_home APPTAINERENV_HF_HUB_OFFLINE=1 \
  APPTAINERENV_TORCH_HOME=/weights/sam3d_body/torch_home APPTAINERENV_PYTHONUSERBASE=/work/scratch/pyuser \
    $A exec --nv --containall --home /work/scratch --pwd /work -B "$W:/work" -B "$(readlink -f "$N/weights"):/weights:ro" \
      -B "$(readlink -f "$N/outputs/cari4d/episode_000002"):/pipeline:ro" "$SIF" \
      python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz
  cmpf "solve CL2_ep002 object_poses_world.npz" "$W/outputs/object_poses_world.npz" "$N/agent_runs/CL2_ep002/outputs/object_poses_world.npz"
}
step_frame0() {  # tier2_refine/t3_frame0.py opus 3 2 shift, CPU, image, all 30 episodes
  (cd "$N" && RUN "$REPO/tier2_refine/t3_frame0.py" opus experiments/t3_frame0/R_opus_f3_c2_shift 3 2 shift)
  cmptree "frame0" "$N/experiments/t3_frame0/R_opus_f3_c2_shift" "$N/experiments/t3_frame0/opus_f3_c2_shift" -name "episode_*.npz"
}
step_opus() {    # agent/build_release_opus.py, host python; README.md is hand-written (release_files/)
  rm -rf "$O/tier3_opus"; (cd "$REPO/agent" && $PY build_release_opus.py "$O/tier3_opus" > /dev/null)
  cp "$HERE/release_files/tier3_opus_20261021_README.md" "$O/tier3_opus/README.md"
  cmptree "opus release" "$O/tier3_opus" "$N/release/tier3_opus_20261021"
}
step_combined() { # support/build_release_combined.py, host python; README.md is hand-written (release_files/)
  rm -rf "$O/tier3_frame0_supports"; (cd "$REPO/support" && $PY build_release_combined.py "$O/tier3_frame0_supports" > /dev/null)
  cp "$HERE/release_files/tier3_frame0_supports_20261024_README.md" "$O/tier3_frame0_supports/README.md"
  cmptree "combined release" "$O/tier3_frame0_supports" "$N/release/tier3_frame0_supports_20261024"
}
step_fit() {     # support fit on a copy of support/: 2 and 28 with the legacy fit, 1 with the current fit.py
  local SO=$O/support_check; rm -rf "$SO"; cp -r "$N/support" "$SO"; export APPTAINERENV_V2D_SUPPORT_OUT=$SO
  PW=(--pwd "$REPO/support")
  RUNG python fit_20261023T2028Z.py 2,28
  RUNG python fit.py 1
  PW=(); unset APPTAINERENV_V2D_SUPPORT_OUT
  local i; for i in 1 2 28; do python3 - "$SO" "$N/support" "$i" <<'PY' | tee -a "$LOG"
import json, sys
import numpy as np
a, b = (json.load(open(f"{d}/ep{int(sys.argv[3]):03d}/support.json")) for d in sys.argv[1:3])
d = np.abs(np.array(a["T_world_mesh"]) - np.array(b["T_world_mesh"])).max()
other = [k for k in a if k != "mesh" and a[k] != b.get(k)]
print(f"fit ep{int(sys.argv[3]):03d} T_world_mesh max|d| {d:.3g}; iou {a['iou']} vs {b['iou']}; other fields "
      + ("identical (mesh: absolute path of the run)" if not other else "DIFFER: " + ",".join(other)))
PY
  done
}
step_supportpkg() { # contact.py --snap + package.py + the two copied files, on the copy left by step_fit (or a fresh one)
  local SO=$O/support_check; [ -d "$SO" ] || cp -r "$N/support" "$SO"; export APPTAINERENV_V2D_SUPPORT_OUT=$SO
  rm -rf "$O/tier3_supports"
  PW=(--pwd "$REPO/support")
  RUN contact.py --snap > /dev/null && RUN package.py "$O/tier3_supports"
  PW=(); unset APPTAINERENV_V2D_SUPPORT_OUT
  cp "$SO/vis/sheet_fits.jpg" "$O/tier3_supports/fit_overview.jpg"; cp "$SO/contact_report.json" "$O/tier3_supports/"
  cp "$HERE/release_files/tier3_supports_20261024_README.md" "$O/tier3_supports/README.md"
  cmptree "supports release" "$O/tier3_supports" "$N/release/tier3_supports_20261024"
}

STEPS=("$@"); [ ${#STEPS[@]} = 0 ] && STEPS=(p1 ff2 jc2s1 db release export solve frame0 opus combined fit supportpkg)
say "=== check_steps.sh $(date -u +%FT%TZ) repo ${REPO_COMMIT:-$(cd "$REPO" && git rev-parse --short HEAD 2>/dev/null || echo "(no git; set REPO_COMMIT)")} steps: ${STEPS[*]}"
for s in "${STEPS[@]}"; do say "--- $s"; t0=$(date +%s); "step_$s"; say "--- $s done in $(( $(date +%s) - t0 )) s"; done
