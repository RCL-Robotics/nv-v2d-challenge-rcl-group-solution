#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# build_inputs_tier3.sh: the Tier-3 inputs of the counted Tier-3 file: the Tier-3 reconstruction (hoi_estimation/) from the
# monocular exo video, mesh, ground plane, frame-0 human and frame-0 object pose, then our f0 data root
# <ME>/data/human_motion_data_tier3_f0 with its frame references, retargets, support scenes and palm-mitigation list.
#   han      the reconstruction pipeline (hoi_estimation/): reproducibility/README.md, s1_runbook.md, tier3_runbook.md, end to end
#            reproducibility/run_all_tier3.sh (apptainer images, weights with sha256 in environment/weights.sha256).
#            The agent runs are not deterministic (hoi_estimation/REPRODUCTION_REPORT.md); the frame-0 step is
#            (re-run byte-identical in 30/30 episodes from c0dbf5b, TR/t3ready_20261102/verification/han_followup/).
#   restore  the record's files back at the paths they ran from (TR/t3ready_20261102/tools/restore_layout_t3.py)
#   opus     step 0, the opus_v2 root and its retarget (data_prep/README_tier3.md section 0)
#   A-F      data_prep/README_tier3.md steps A (inputs, frame references), B (frame chain), C (retarget + gates G1-G10,
#            parquet lock), D (support scenes for 01 02 03 06 07 09 11 13 16 17 20 25 27 28), E (finalize, FILES.sha256,
#            READY), F (hand-object gap -> termination.palm_position_threshold=0.25 for 01 02 03 15 17 19 20 28)
#   verify   read-only: the 30 locked f0 parquets, frame references and manifest against set_items.json
# No clean or ground-truth trajectory is read anywhere (the one exception, the true frame-0 object pose inside the
# reconstruction's track, is grey area 1 of TR/t3ready_20261102/README.md). The scripts embed the training server's <ME>
# and the reconstruction release root <HAN> (the ROOT_ME / ROOT_HAN defaults); restore_layout_t3.py --rewrite replaces them.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
usage() { cat <<'EOF'
usage: build_inputs_tier3.sh --steps han,restore,opus,A,B,C,D,E,F,verify [--rewrite] [--dry-run] [--site FILE]
  --steps LIST   which steps; a real run needs --steps (the dry run shows all of them)
  --rewrite      with restore: rewrite <ME> and <HAN> (the reconstruction release root) inside the restored text files
                 to ROOT_ME / ROOT_HAN
  `--steps verify` is read-only.
envs: RETARGET_PYTHON (env v2d), FC_PYTHON (flashchord), KIT_PYTHON (v2dkit)
EOF
}
final_init "$@"
STEPS=""; REWRITE=""
set -- ${REST[@]+"${REST[@]}"}
while [ $# -gt 0 ]; do
  case "$1" in
    --steps) STEPS=$2; shift ;; --rewrite) REWRITE=--rewrite ;;
    *) die "unknown option $1 (see --help)" ;;
  esac; shift
done
if [ -z "$STEPS" ]; then
  [ "$DRY" = 1 ] || die "a real run needs --steps (e.g. --steps verify)"
  STEPS=han,restore,opus,A,B,C,D,E,F,verify
fi
has() { case ",$STEPS," in *",$1,"*) return 0;; *) return 1;; esac; }
B=$TR/t3ready_20261102
H=$REPO/hoi_estimation
plan_header "Tier-3 inputs: steps $STEPS"
addq B="$B"; addq H="$H"; addq ROOT_HAN="$ROOT_HAN"
if has han; then
  add ''; add '# ---- han: the Tier-3 reconstruction (read first: $H/reproducibility/README.md; s1_runbook.md; tier3_runbook.md)'
  add '#   s1: CARI4D -> E2 -> P1 -> M1 -> first-frame ICP/PnP -> FF2 -> JC2s1 -> DB -> s1 release'
  add '#   tier3: agent inputs -> opus_v2 agent batches -> frame-0 correction (tier2_refine/t3_frame0.py opus <out> 3 2 shift)'
  add '#          -> supports -> combined release tier3_frame0_supports_20261024'
  add '#   GT-blind frame-0 step: cut t3_frame0.py from "res = {}" on, feed ref_poses holding frame 0 only, and give'
  add '#   build_release_combined.py no diag_scores.json (optional since c0dbf5b)'
  add 'bash "$H/reproducibility/run_all_tier3.sh"'
fi
if has restore; then
  add ''; add '# ---- restore: the record'"'"'s scripts and records at their run paths (SOURCES.tsv); data-root records go to'
  add '#      <ROOT_ME>/t3ready_expected/ (expected outputs, not inputs)'
  addq "$PLAN_PY" -I -B "$B/tools/restore_layout_t3.py" --me "$ROOT_ME" --han "$ROOT_HAN" ${REWRITE:+"$REWRITE"}
  add '# the retarget runs soma_to_g1.py from <ME>/code/nv-v2d-challenge-codebase at f2923d61 (an ancestor of this commit):'
  add '[ -e "$ROOT_ME/code/nv-v2d-challenge-codebase/.git" ] || GIT_LFS_SKIP_SMUDGE=1 git -C "$REPO" worktree add --detach "$ROOT_ME/code/nv-v2d-challenge-codebase" f2923d61817e61eb37d939e0f75e5b4a7aaa5cbf'
fi
if has opus; then
  add ''; add '# ---- step 0: the opus_v2 root (byte copy of the reconstruction release tier3_opus_20261021), its retarget and rcheck'
  add '#      (data_prep/README_tier3.md section 0; records in $B/data_prep/opus_root; the f0 build reads this root in step A'
  add '#      and in the GT-free comparisons G9 and t3f0_align.py)'
  add 'note "step 0: follow $B/data_prep/README_tier3.md section 0 (t3opus_build_data.py, t3opus_driver_v2.sh retarget, t3opus_rcheck_v2.py)"'
fi
cdme='cd "$ROOT_ME" &&'
if has A; then
  add ''; add '# ---- A: inputs and frame references (CPU) -> data/human_motion_data_tier3_f0, frame_references_f0, manifest d496584c...'
  add "( $cdme CUDA_VISIBLE_DEVICES= \"\$FC_PYTHON\" -I -B scripts/t3f0_build_root.py inputs )"
fi
if has B; then
  add ''; add '# ---- B: the retarget frame chain (v2d env, no IK) -> tmp_analysis/tier3/f0/frames/epNN.{json,npz}'
  add "( $cdme CUDA_VISIBLE_DEVICES= \"\$RETARGET_PYTHON\" scripts/t3f0_frames.py --verts 1,2,3,6,7,9,11,13,16,17,20,25,27,28 \$(seq 0 29) )"
fi
if has C; then
  add ''; add '# ---- C: retarget (soma_to_g1.py --save --fps 30, v2d env, one plain CUDA context per episode) + gates G1-G10'
  add '#      -> tmp_analysis/tier3/f0/retarget/parquet_lock.json (bdb83b0e..., 30 entries)'
  add "( $cdme bash scripts/t3f0_retarget_all.sh )"
fi
if has D; then
  add ''; add '# ---- D: support scenes of the 14 support episodes (flashchord env), as on 2026-10-24: first pass, ep01 lean search,'
  add '#      final pass -> SCENE_SUPPORT.json (bce8e9eb...), SUPPORT_FILES.sha256 (5deea2b0...); then the physical checks'
  add "( $cdme \"\$FC_PYTHON\" scripts/t3f0_supports.py all )"
  add "( $cdme \"\$FC_PYTHON\" scripts/t3f0_lean_search.py 1 )"
  add "( $cdme \"\$FC_PYTHON\" scripts/t3f0_supports.py all )"
  add "( $cdme bash tmp_analysis/tier3/f0/physcheck_chain.sh )"
fi
if has E; then
  add ''; add '# ---- E: finalize the write-once root -> FILES.sha256 (f9c94e85..., 576 files), READY (ee58a62d...); prep table'
  add "( $cdme CUDA_VISIBLE_DEVICES= \"\$FC_PYTHON\" -I -B scripts/t3f0_build_root.py finalize )"
  add "( $cdme CUDA_VISIBLE_DEVICES= \"\$FC_PYTHON\" -I -B scripts/t3f0_prep_table.py )"
fi
if has F; then
  add ''; add '# ---- F: hand-object gap (GT-free) and the palm mitigation list -> f0/align/mitigation_f0.json (2ab767eb...)'
  add "( $cdme CUDA_VISIBLE_DEVICES= \"\$RETARGET_PYTHON\" scripts/t3f0_align.py \$(seq 0 29) )"
  add "( $cdme CUDA_VISIBLE_DEVICES= \"\$RETARGET_PYTHON\" scripts/t3f0_align.py --summary )"
  add '# compare the rebuilt roots with <ROOT_ME>/t3ready_expected/ (FILES.sha256, READY, SCENE_SUPPORT.json, manifests)'
fi
if has verify; then
  add ''; add '# ---- verify (read-only): the 30 f0 inputs and frame references of the counted Tier-3 file'
  addq "$PLAN_PY" -I -B "$FINAL_LIB/verify_inputs.py" --tier 3
  add 'check_sha256 "$ROOT_ME/data/human_motion_data_tier3_f0/whole_body/soma/track2_manifest.json" d496584c98cb4e811c41a12b0afa5992ff02a64dd27cc8e73013304b10d364b5'
  add 'check_sha256 "$ROOT_ME/data/human_motion_data_tier3_f0/FILES.sha256" "$(sha256sum "$B/data_prep/data_root/FILES.sha256" | cut -d" " -f1)"'
fi
printf '%s' "$PLAN" | run_plan build_inputs_tier3
