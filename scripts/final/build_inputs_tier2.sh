#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# build_inputs_tier2.sh: the four Tier-2 input roots of the counted Tier-2 file, built from track_2/tier_2_synthetic_noise
# only (no Tier-1 or clean data; record TR/t2ready_20261031/README.md "Reproduce" steps 1-4):
#   raw    <ME>/data/human_motion_data_tier2           SOMA conversion + G1 retarget of the released Tier-2 episodes
#   k3     <ME>/data/human_motion_data_tier2_k3        our K3 Kalman smoother of the object track (+ frame_refs_k3)
#   k3fix  <ME>/data/human_motion_data_tier2_k3fix     K3 with the ep04 root-cause fixes (+ frame_refs_k3fix) and the
#                                                      per-episode gate rule G1-G4 (k3full_gates.py)
#   g2     <ZX>/data/human_motion_data_tier2_g2-2026-10-24   Zuxing Lu's G2 contact-model smoother (+ frame_refs_g2)
#   verify the 30 task.parquet files and frame references the counted file used, against set_items_r9.json (read-only)
# The R9 file uses K3fix for 18 episodes, G2 for 7, raw for 4 and K3 for 1 (set_items_r9.json `input`).
# The builders are the committed copies, byte-identical to the files that ran; they embed the training server's
# absolute paths (<ME>=/root/autodl-tmp/haoran.zhang, <ZX>=/root/autodl-tmp/zuxing.lu, kalman work dir
# <ME>/tmp_analysis/tier2/kalman, Zuxing's kit copy and env), so run them in that layout (a symlink or a container bind
# mount of those roots) or edit the path constants. The write-once roots refuse to rebuild over an existing READY.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
usage() { cat <<'EOF'
usage: build_inputs_tier2.sh --steps raw,k3,k3fix,g2,verify [--gpus 0] [--jobs 6] [--dry-run] [--site FILE]
  --steps LIST   which steps; a real run needs --steps (the dry run shows all of them)
  --gpus LIST    GPUs for the retarget (raw step), --jobs parallel retargets
  `--steps verify` is read-only: it checks the 30 inputs and frame references of the counted file.
envs: RETARGET_PYTHON (Isaac Sim / Isaac Lab env v2d), KIT_PYTHON (kit venv), ZX_KIT_PYTHON (Zuxing's G2 builder env,
recorded in environment/zx_g2_builder.pip-freeze.txt; default <ZX>/envs/kit/bin/python; setup_env.sh --steps zxg2)
EOF
}
final_init "$@"
STEPS=""; GPUL=0; JOBS=6
set -- ${REST[@]+"${REST[@]}"}
while [ $# -gt 0 ]; do
  case "$1" in
    --steps) STEPS=$2; shift ;; --gpus) GPUL=$2; shift ;; --jobs) JOBS=$2; shift ;;
    *) die "unknown option $1 (see --help)" ;;
  esac; shift
done
if [ -z "$STEPS" ]; then
  [ "$DRY" = 1 ] || die "a real run needs --steps (e.g. --steps verify)"
  STEPS=raw,k3,k3fix,g2,verify
fi
has() { case ",$STEPS," in *",$1,"*) return 0;; *) return 1;; esac; }
D=$TR/t2ready_20261031/data
plan_header "Tier-2 inputs: steps $STEPS"
addq D="$D"; addq ZX_KIT_PYTHON="$ZX_KIT_PYTHON"
if has raw; then
  add ''; add '# ---- raw: SOMA inputs from the released Tier-2 episodes, then G1 retarget (soma_to_g1.py --save --fps 30)'
  addq cd "$REPO/robotic_grounding"
  addq "$RETARGET_PYTHON" scripts/track2/track2_to_soma_inputs.py --track-root "$V2D_DATA/track_2/tier_2_synthetic_noise" \
    --motion-root "$ROOT_ME/data/human_motion_data_tier2" --episodes all
  add "# retarget_tier2.sh runs soma_to_g1.py from <ME>/code/nv-v2d-challenge-codebase (a checkout of f2923d6:"
  add "#   git worktree add <ME>/code/nv-v2d-challenge-codebase f2923d61817e61eb37d939e0f75e5b4a7aaa5cbf) with the v2d env"
  addq bash "$D/ours/raw/retarget_tier2.sh" "$GPUL" "$JOBS"
fi
if has k3; then
  add ''; add '# ---- K3 (NOISY-ONLY, CPU): design steps d1-d3 (v2d env), full-frame smoother e1 (v2d env), eval inputs e2 (kit env)'
  add '# the kalman scripts run from their work dir <ME>/tmp_analysis/tier2/kalman/{scripts,eval} (KOUT in kcommon.py)'
  add 'K=$ROOT_ME/tmp_analysis/tier2/kalman'
  add '[ -d "$K/scripts" ] || { mkdir -p "$K" && cp -r "$D/ours/k3/kalman/scripts" "$D/ours/k3/kalman/eval" "$K/"; }'
  for s in d1_avail_frames_variograms d2_mesh_grasp_physics d2b_rigid_carry_anchor_fit d3_params_ml_dryrun; do
    add "( cd \"\$K/scripts\" && CUDA_VISIBLE_DEVICES= \"\$RETARGET_PYTHON\" -I -B $s.py )"
  done
  add '( cd "$K/eval" && CUDA_VISIBLE_DEVICES= "$RETARGET_PYTHON" -I -B e1_k3_fullframe.py )'
  add '( cd "$K/eval" && CUDA_VISIBLE_DEVICES= "$KIT_PYTHON" -I -B e2_build_eval_inputs.py )   # -> $K/eval/data_k3, frame_refs_k3'
  add '# data/human_motion_data_tier2_k3 = a byte copy of $K/eval/data_k3 plus the Tier-2 manifest and its READY'
  add '# (record: t2ready_20261031/README.md "Reproduce" step 2; the k3fix and k3full gate checks below re-verify it)'
fi
if has k3fix; then
  add ''; add '# ---- K3fix (NOISY-ONLY, CPU): tracks + gates (v2d env), data root (kit env), per-episode input rule G1-G4 (v2d env)'
  add '# they ran as <ME>/scripts/*.py from <ME> (docstrings); put the committed copies there if they are missing'
  add 'mkdir -p "$ROOT_ME/scripts"; for f in ours/k3fix/k3fix_build.py ours/k3fix/k3fix_build_data.py ours/gates/k3full_gates.py; do [ -f "$ROOT_ME/scripts/$(basename $f)" ] || cp "$D/$f" "$ROOT_ME/scripts/"; done'
  add '( cd "$ROOT_ME" && CUDA_VISIBLE_DEVICES= nice -n 19 "$RETARGET_PYTHON" -I -B scripts/k3fix_build.py tracks )   # -> tmp_analysis/tier2/k3fix_pilot/build'
  add '( cd "$ROOT_ME" && CUDA_VISIBLE_DEVICES= nice -n 19 "$RETARGET_PYTHON" -I -B scripts/k3fix_build.py gates )'
  add '( cd "$ROOT_ME" && CUDA_VISIBLE_DEVICES= nice -n 19 "$KIT_PYTHON" -I -B scripts/k3fix_build_data.py )   # -> data/human_motion_data_tier2_k3fix, frame_refs_k3fix'
  add '( cd "$ROOT_ME" && CUDA_VISIBLE_DEVICES= nice -n 19 "$RETARGET_PYTHON" -I -B scripts/k3full_gates.py gates )   # K3fix where G1-G4 pass, raw otherwise'
fi
if has g2; then
  add ''; add '# ---- G2 (Zuxing Lu; NOISY-ONLY): K3 variant agile+hgate, G2_surface_only smoother, data root + frame_refs_g2'
  add '# copies of his files; their paths point into <ZX> (t2ready_20261031/data/zuxing_g2/README.md); all three steps run'
  add '# in his builder env (G2_surface_only/run_all.sh names <ZX>/envs/kit/bin/python itself), recorded in'
  add '# environment/zx_g2_builder.pip-freeze.txt; env_check.py stops here if ZX_KIT_PYTHON differs from that record'
  add '"$ZX_KIT_PYTHON" -I -B "$FINAL_LIB/env_check.py" "$REPO/environment/zx_g2_builder.pip-freeze.txt"'
  add '( cd "$D/zuxing_g2/kfix" && "$ZX_KIT_PYTHON" -I -B run_fix.py agile+hgate )'
  add 'bash "$D/zuxing_g2/G2_surface_only/run_all.sh" "$ROOT_ZX/tmp/tier2_noise/kfix/gen/G2_surface_only/out"'
  add '"$ZX_KIT_PYTHON" -I -B "$D/zuxing_g2/g2data/build_g2_data.py" "$ROOT_ZX/data/human_motion_data_tier2_g2-2026-10-24" $(seq 0 29)'
  add '# (round 7 used the same builder for its added G2 episodes: t2r7_20261102/data/zuxing_g2_r7/g2data/build_g2_data.py)'
fi
if has verify; then
  add ''; add '# ---- verify (read-only): the 30 own inputs and frame references of the counted Tier-2 file'
  addq "$PLAN_PY" -I -B "$FINAL_LIB/verify_inputs.py" --tier 2 \
    --pack-frame-refs "$ROOT_ME/fc_runs/submissions/t2r9_pack_20261104/pack_r9_placeholder/r9_closest/frame_refs"
fi
printf '%s' "$PLAN" | run_plan build_inputs_tier2
