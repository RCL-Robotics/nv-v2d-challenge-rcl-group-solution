#!/usr/bin/env bash
# End-to-end Tier 3 run, from the official dataset to the combined release, with the parameters of the submission.
# Every stage writes into a fresh data root ($V2D_ROOT); the dataset, weights and images are only read.
#
# Usage:
#   V2D_ROOT=<new, empty, writable dir>  DATASET=<.../track_2/tier_1_multiview_caption>  WEIGHTS=<weights dir>
#   IMAGES=<dir with v2d_cari4d.sif, v2d_gdino.sif>  V2D_RUN=<local scratch dir>  REF_PY=<python with pandas+pyarrow (may be "env X=.. python3")>
#   bash run_all_tier3.sh [stage ...]
# Stages (default: all, in this order):
#   setup baseline e2 p1 m1 ffw pnp ff2 jc2s1 db release export agents opus frame0 support combined
# Env:
#   EPISODES   space-separated episode indices (default 0..29)
#   GPU        GPU index (default 0); APPTAINER  apptainer binary (default: apptainer on PATH)
#   AGENTS     llm (default): run the Claude Code agents (agent/run_batch_CL.sh; needs the claude CLI and a token,
#              see agent/run_claude.sh); replay: re-run the delivered solvers of agent_records/ instead (no LLM)
#   AGENT_P    agents run at the same time (default 3)
#   HOST_PY    python for postproc_depth_blend / build_release_opus / build_release_combined. Unset: run them in
#              the CARI4D image (float-level differences in the depth blend). For bit-identical results use the
#              host versions in environment/pip_freeze_host_venv.txt.
# Requires the Git LFS files of reconstruction/modules (git lfs pull) and an Ampere/Ada/Hopper GPU.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(dirname "$HERE"); S=$REPO/hoi_extraction/scripts
N=${V2D_ROOT:?set V2D_ROOT}; DATASET=${DATASET:?set DATASET}; WEIGHTS=${WEIGHTS:?set WEIGHTS}; IMAGES=${IMAGES:?set IMAGES}
export V2D_ROOT=$N V2D_RUN=${V2D_RUN:?set V2D_RUN} APPTAINERENV_V2D_ROOT=$N
A=${APPTAINER:-apptainer}; export APPTAINER=$A SIF_EXEC=$A; GPU=${GPU:-0}
read -ra EPS <<< "${EPISODES:-$(seq -s ' ' 0 29)}"
CSV=$(IFS=,; echo "${EPS[*]}")
SIF=$N/images/v2d_cari4d.sif; GSIF=$N/images/v2d_gdino.sif
# the dataset, weights and images are symlinked into the root; bind their real locations read-only in every call
RD=$(readlink -f "$DATASET"); RW=$(readlink -f "$WEIGHTS"); RI=$(readlink -f "$IMAGES")
export APPTAINER_BIND="${APPTAINER_BIND:+$APPTAINER_BIND,}$RD:$RD:ro,$RW:$RW:ro,$RI:$RI:ro"
B=(-B "$N" -B "$REPO")
RUN()   { $A exec "${B[@]}" "$SIF" python "$@"; }                                           # CPU, image
RUNG()  { APPTAINERENV_CUDA_VISIBLE_DEVICES=$GPU $A exec --nv "${B[@]}" "$@"; }             # GPU, any image
RUNC()  { RUNG "$SIF" bash -c 'export PYTHONPATH=/workspace/v2d_cari4d/lib/cari4d:/workspace/v2d_sam3d_body/lib:$PYTHONPATH
  export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH; exec "$@"' _ "$@"; }    # GPU, SOMA/MHR libs
HPY()   { if [ -n "${HOST_PY:-}" ]; then "$HOST_PY" "$@"; else RUN "$@"; fi; }
inter() { local x out=(); for x in $1; do [[ " ${EPS[*]} " == *" $x "* ]] && out+=("$x"); done; echo "${out[*]:-}"; }
ep6()   { printf episode_%06d "$1"; }
log()   { echo "[$(date -u +%FT%TZ)] $*"; }
# episode lists of the submission (s1_runbook.md)
E2_EPS=$(inter "1 2 3 4 6 7 8 9 11 12 13 14 15 16 17 18 19 25 27 28")
M1_EPS=$(inter "3 6 12 14 16 21")

st_setup() {
  for f in "$REPO"/../reconstruction/modules/v2d_cari4d/lib/cari4d/lib_mhr/assets/* "$REPO"/../reconstruction/modules/v2d_sam3d_body/assets/*; do
    head -c 40 "$f" | grep -q "git-lfs" && { echo "Git LFS file not fetched: $f (run git lfs pull)"; exit 1; }
  done
  local bad; bad=$(find -L "$RD" "$RW" "$RI" ! -readable 2>/dev/null | head -5)
  [ -n "$bad" ] && { echo "files not readable by $(id -un) (fix the permissions first):"; echo "$bad"; exit 1; }
  [ -z "${REF_PY:-}" ] && { echo "set REF_PY (python with pandas + pyarrow, for export_reference.py)"; exit 1; }
  local RP; read -ra RP <<< "$REF_PY"
  mkdir -p "$N"/{dataset/track_2,code/pipeline,code/reconstruction,build/patches,logs,outputs,experiments,agent_runs,release}
  ln -sfn "$RD" "$N/dataset/track_2/tier_1_multiview_caption"; ln -sfn "$RW" "$N/weights"; ln -sfn "$RI" "$N/images"
  ln -sfn "$(cd "$REPO/../reconstruction/modules" && pwd)" "$N/code/reconstruction/modules"
  cp "$REPO"/hoi_extraction/cluster/{run_episode.sh,gen_prompt_one.py,gen_prompt_v2.py,episodes.tsv} "$N/code/pipeline/"
  cp "$REPO/hoi_extraction/singularity/10_nvidia_egl.json" "$N/build/"
  cp "$REPO/hoi_extraction/patches/run_foundationpose_mhr_export.py" "$N/build/patches/"
  RUN -c "
import json,numpy as np,trimesh
ds='$N/dataset/track_2/tier_1_multiview_caption'; info=json.load(open(ds+'/meta/info.json'))
out={}
for o in info['objects']:
    g=trimesh.load(ds+'/'+o['mesh'],force='scene',process=False).dump(concatenate=True)
    out[o['name']]={'diag_m':float(np.linalg.norm(g.extents)),'prompt':o['prompt']}
json.dump(out,open('$N/code/pipeline/object_info.json','w'),indent=1); print('object_info', len(out))"
  "${RP[@]}" "$S/export_reference.py" --out "$N"     # ref_human_f0 (frame 0) and ref_poses (frame 0 used; all frames for scoring)
}

st_baseline() {  # CARI4D stages 0-8 per episode (run_local.sh moves PASS episodes to outputs/cari4d)
  bash "$S/run_local.sh" "$GPU" "${EPS[@]}"
}

st_e2() {  # FoundationPose re-registration; uses $V2D_RUN/out/<ep> for an episode whose baseline did not pass
  [ -z "$E2_EPS" ] && return
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$GPU V2D_FP_REREGISTER=1 bash "$S/exp_fp_mode.sh" E2_reregister "--register-first-then-track" $E2_EPS
  # failed baselines (e.g. episode 25): keep their stage 0-5 outputs next to the others, as in the submission
  local i e
  for i in "${EPS[@]}"; do e=$(ep6 "$i")
    if [ ! -f "$N/outputs/cari4d/$e/inference/coconet.pth" ] && [ -d "$V2D_RUN/out/$e" ] && [ ! -L "$V2D_RUN/out/$e" ]; then
      mkdir -p "$N/outputs/cari4d_failed"
      rsync -a "$V2D_RUN/out/$e/" "$N/outputs/cari4d_failed/${e}_out/" && rsync -a "$V2D_RUN/work/$e/" "$N/outputs/cari4d_failed/${e}_work/"
      log "$e: baseline did not pass, stage 0-5 outputs kept in outputs/cari4d_failed"
    fi
  done
}

st_p1() {
  [ -z "$E2_EPS" ] && return
  # shellcheck disable=SC2086
  RUN "$S/postproc_temporal.py" $E2_EPS --out-exp P1_temporal
}

st_m1() {  # re-prompted CARI4D (GroundingDINO prompt with depth and person proximity) for episodes 3 6 12 14 16 21
  [ -z "$M1_EPS" ] && return
  local X=$N/experiments/M1_prompt_v2 i e obj diag desc D
  mkdir -p "$X/work" "$X/out" "$N/experiments/M1"
  for i in $M1_EPS; do e=$(printf %06d "$i")
    obj=$(python3 -c "import json;[print(d['object']) for d in map(json.loads,open('$N/dataset/track_2/tier_1_multiview_caption/meta/episodes_metadata.jsonl')) if d['episode_index']==$i]")
    diag=$(python3 -c "import json;print(json.load(open('$N/code/pipeline/object_info.json'))['$obj']['diag_m'])")
    desc=$(python3 -c "import json;print(json.load(open('$N/code/pipeline/object_info.json'))['$obj']['prompt'])")
    D=$N/outputs/cari4d/episode_$e/depth/moge2; mkdir -p "$X/work/episode_$e"
    [ -f "$X/work/episode_$e/prompts.json" ] || RUNG "$GSIF" bash -c 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH; exec "$@"' _ \
      python "$N/code/pipeline/gen_prompt_v2.py" --video_path "$N/dataset/track_2/tier_1_multiview_caption/videos/chunk-000/observation.images.exo_camera/episode_$e.mp4" \
        --object_prompt "${obj//_/ }" --object_description "$desc" --object_diag_m "$diag" \
        --raw_depth "$D/raw.npy" --intrinsics "$D/intrinsics.pkl" --model_dir "$N/weights/grounding_dino" --output_path "$X/work/episode_$e/prompts.json"
    [ "${M1_PROMPT_ONLY:-0}" = 1 ] && continue
    log "M1 episode $e start"
    if CUDA_VISIBLE_DEVICES=$GPU WORK_ROOT=$X/work OUT_ROOT=$X/out V2D_FP_REREGISTER=1 \
         bash "$N/code/pipeline/run_episode.sh" "$i" > "$N/logs/episode_$e.M1.log" 2>&1; then
      mkdir -p "$N/experiments/M1/episode_$e"
      ln -sfn "../../M1_prompt_v2/out/episode_$e/inference/coconet.pth" "$N/experiments/M1/episode_$e/coconet.pth"
      log "M1 episode $e OK"
    else log "M1 episode $e FAILED (logs/episode_$e.M1.log)"; fi
  done
}

st_ffw() {  # first-frame ICP of our frame-0 human to the frame-0 reference human
  local Bb=() Bm=() i f
  for i in "${EPS[@]}"; do f=$N/outputs/cari4d/$(ep6 "$i")/inference/coconet.pth
    [ -f "$f" ] || f=$N/experiments/E2_reregister/$(ep6 "$i")/coconet.pth; [ -f "$f" ] && Bb+=("$i=$f"); done
  RUNC python "$S/firstframe_align.py" --wide --bundles "${Bb[@]}" --out "$N/logs/ffw_baseline.json"
  for i in $M1_EPS; do f=$N/experiments/M1/$(ep6 "$i")/coconet.pth; [ -e "$f" ] && Bm+=("$i=$f"); done
  [ ${#Bm[@]} -gt 0 ] && RUNC python "$S/firstframe_align.py" --wide --bundles "${Bm[@]}" --out "$N/logs/ffw_M1.json"
  return 0
}

st_pnp() {
  local v Bx
  for v in baseline M1; do [ -f "$N/logs/ffw_$v.json" ] || continue
    read -ra Bx <<< "$(python3 -c "import json;d=json.load(open('$N/logs/ffw_$v.json'));print(' '.join(f'{k}={v[\"bundle\"]}' for k,v in d.items()))")"
    RUNC python "$S/firstframe_pnp.py" --bundles "${Bx[@]}" --icp "$N/logs/ffw_$v.json" --out "$N/logs/pnp_$v.json"
  done
  python3 -c "
import json,os; b=json.load(open('$N/logs/pnp_baseline.json'))
if os.path.isfile('$N/logs/pnp_M1.json'): b.update(json.load(open('$N/logs/pnp_M1.json')))
json.dump(b,open('$N/logs/pnp_merged.json','w')); json.dump(b,open('$N/logs/pnp_forE1.json','w'))"
}

st_ff2() {
  RUNC python "$S/postproc_feet_floor.py" "${EPS[@]}" --use-sim --ref-foot-height --align "default=$N/logs/pnp_merged.json" --out-exp FF2_feet_ref
}

st_jc2s1() {
  RUNG --pwd "$S" "$SIF" python postproc_joint_camera_K.py "${EPS[@]}" --out-exp JC2s1_shared_f820_pp --sig-logf 1e-4 --sig-pp 0.01 --K-prior 820 758.1 560.5
}

st_db() {
  (cd "$S" && HPY postproc_depth_blend.py "${EPS[@]}" --src JC2s1_shared_f820_pp --out-exp DB_JC2s1_shared_f820_pp)
}

st_release() {  # s1 release (+ frame_references from the released object poses)
  local R=$N/release/tier3_s1_20261020; rm -rf "$R"
  RUNG --pwd "$S" "$SIF" python build_release.py "$R" --episodes "${EPS[@]}"
  RUN -c "
import json,numpy as np,os
R='$R'; os.makedirs(R+'/frame_references',exist_ok=True)
for d in json.load(open(R+'/episodes.json')):
    e=d['episode_index']; P=np.load(f\"{R}/reconstructed_data/{d['sequence_id']}/object_pose_xyz_wxyz.npy\")
    xyzw=np.c_[P[:,:3],P[:,4:],P[:,3:4]].astype(np.float32)[:,None]
    np.savez(f'{R}/frame_references/episode_{e:06d}.npz',episode_index=np.int64(e),coordinate_frame=np.array('released_track2_world'),
             frame_index=np.arange(len(P),dtype=np.int64),pose_xyzw=xyzw)"
}

st_export() {  # one agent input folder per episode
  local i
  for i in "${EPS[@]}"; do
    RUNG "$SIF" python3 "$REPO/agent/export_inputs.py" "$i" --ws "$N/agent_runs/T1_ds_ep$(printf %03d "$i")" \
      --variant DB_JC2s1_shared_f820_pp --align "$N/logs/JC2s1_shared_f820_pp_align.json" --camera JC2s1_shared_f820_pp
  done
}

pfx() { [ "$1" -le 10 ] && echo CL2 || { [ "$1" -le 21 ] && echo CL2b || echo CL2c; }; }  # run names of the submission
st_agents() {
  local i
  if [ "${AGENTS:-llm}" = replay ]; then
    for i in "${EPS[@]}"; do GPU=$GPU bash "$REPO/agent/rerun_solve.sh" "$i" "$N/agent_runs"; done
    return
  fi
  local p list
  for p in CL2 CL2b CL2c; do list=(); for i in "${EPS[@]}"; do [ "$(pfx "$i")" = $p ] && list+=("$i"); done
    [ ${#list[@]} -gt 0 ] && (cd "$REPO/agent" && PFX=$p TASK=AGENT_TASK_temporal_v2.md APPTAINERENV_CUDA_VISIBLE_DEVICES=$GPU \
      bash run_batch_CL.sh "${AGENT_P:-3}" "${list[@]}")
  done
  for i in "${EPS[@]}"; do [ -f "$N/agent_runs/$(pfx "$i")_ep$(printf %03d "$i")/outputs/object_poses_world.npz" ] \
    || { echo "agent of episode $i delivered no trajectory"; exit 1; }; done
}

st_opus() {
  local R=$N/release/tier3_opus_20261021; rm -rf "$R"; HPY "$REPO/agent/build_release_opus.py" "$R"
}

st_frame0() {
  local O=$N/experiments/t3_frame0/opus_f3_c2_shift; mkdir -p "$O"
  V2D_EPISODES=$CSV APPTAINERENV_V2D_EPISODES=$CSV RUN "$REPO/tier2_refine/t3_frame0.py" opus "$O" 3 2 shift | tee "$N/logs/t3_frame0.log"
}

st_support() {  # support surfaces of the placement episodes (support/common.py SUPPORT) among EPISODES
  local SE; SE=$(V2D_EPISODES=$CSV RUN -c "import sys;sys.path.insert(0,'$REPO/support');from common import SUPPORT;print(','.join(map(str,SUPPORT)))")
  [ -z "$SE" ] && { log "no support episodes"; return; }
  local Bs=("${B[@]}" --pwd "$REPO/support"); export APPTAINERENV_V2D_EPISODES=$CSV APPTAINERENV_CUDA_VISIBLE_DEVICES=$GPU
  local S28; S28=$(python3 -c "print(','.join(x for x in '$SE'.split(',') if x != '1'))")  # ep 1 used the later fit.py
  $A exec "${Bs[@]}" "$SIF" python background.py "$SE"
  $A exec "${Bs[@]}" "$SIF" python anchor.py "$SE"
  $A exec --nv "${Bs[@]}" "$GSIF" python detect.py "$SE"
  $A exec --nv "${Bs[@]}" "$SIF" python segment.py "$SE"
  [ -n "$S28" ] && $A exec --nv "${Bs[@]}" "$SIF" bash -c "python fit_20261023T2028Z.py $S28 && python overlay.py $S28"
  [[ ",$SE," == *",1,"* ]] && $A exec --nv "${Bs[@]}" "$SIF" bash -c "python fit.py 1 && python overlay.py 1"
  $A exec "${Bs[@]}" "$SIF" python sheet.py
  $A exec "${Bs[@]}" "$SIF" python contact.py "$SE" --snap > /dev/null
  local R=$N/release/tier3_supports_20261024; rm -rf "$R"
  $A exec "${Bs[@]}" "$SIF" python package.py "$R"
  cp "$N/support/vis/sheet_fits.jpg" "$R/fit_overview.jpg"; cp "$N/support/contact_report.json" "$R/"
  cp "$REPO/reproducibility/release_files/tier3_supports_20261024_README.md" "$R/README.md"
}

st_combined() {
  local R=$N/release/tier3_frame0_supports_20261024; rm -rf "$R"
  [ -f "$N/release/tier3_supports_20261024/index.json" ] || { echo "run the support stage first (needs at least one support episode)"; exit 1; }
  (cd "$REPO/support" && HPY build_release_combined.py "$R")
}

ALL=(setup baseline e2 p1 m1 ffw pnp ff2 jc2s1 db release export agents opus frame0 support combined)
STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=("${ALL[@]}")
log "episodes: ${EPS[*]}  (E2/P1: ${E2_EPS:-none}; M1: ${M1_EPS:-none})  stages: ${STAGES[*]}"
for s in "${STAGES[@]}"; do t0=$(date +%s); log "== $s"; "st_$s"; log "== $s done in $(( $(date +%s) - t0 )) s"; done
log "ALL_DONE"
