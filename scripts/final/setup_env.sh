#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# setup_env.sh: install the environments every Track-2 step ran in, with the versions recorded in environment/
# (exported from the training server's envs on 2026-11-04; environment/README.md).
#
#   flashchord  training, export, evaluation: conda Python 3.11.17 (environment/flashchord.conda-explicit.txt) + uv 0.13.0,
#               then every package hash-locked by the stock runtime's uv.lock (TR/repro/env/install_flashchord.sh);
#               jax 0.10.2 + jax-cuda13 wheels, warp-lang 1.14.0, newton 1.2.0, mujoco(-warp) 3.8.1, onnxruntime-gpu 1.28.0
#   stock       the FlashCHORD runtime the kit evaluator runs on: nvidia-isaac/video_to_data @ 8c778f9 with its LFS assets
#               (TR/repro/env/get_stock_runtime.sh); evaluation never runs this repository's flash_chord/src
#   kit         the Oct-11 submission kit (provided by the organisers; KIT_MANIFEST.json 398947a7...) and its venv v2dkit
#               on the flashchord Python with the pinned versions of environment/v2dkit.pip-freeze.txt
#   retarget    env v2d: conda Python 3.11.17, torch 2.7.0+cu128 + torchvision 0.22.0+cu128 (PyTorch cu128 index; the
#               constraints pin these builds and isaacsim-core needs torch==2.7.0), torchaudio 2.7.0 (PyPI, as recorded),
#               Isaac Sim 5.1.0 (pypi.nvidia.com), Isaac Lab v2.3.2 (37ddf62, installed with CONDA_PREFIX = this env),
#               robotic_grounding (TR/repro/env/install_retarget.sh; full freeze environment/v2d.pip-freeze.txt)
#               used for the G1 retargeting and for Tier-2 / Tier-3 input building
#   zxg2        Zuxing Lu's Tier-2 G2 input-builder env (ZX_KIT_PYTHON): a uv venv on CPython 3.12.3 with
#               environment/zx_g2_builder.pip-freeze.txt
#   lfs         Git LFS of this checkout: FlashCHORD assets and the 3 runtime files of reconstruction/modules that Han
#               Zhou's run_all_tier3.sh needs, from upstream nvidia-isaac/video_to_data (the fork hosts no LFS objects
#               for upstream files); Han Zhou's 41 hoi_estimation records from the fork's own LFS store
#   check       read-only: the four envs against environment/*.pip-freeze.txt (fails on any difference unless
#               --report-only), GPU/driver, kit hashes, stock runtime commit
# CUDA / driver: the jax-cuda13 wheels need an NVIDIA driver for CUDA 13 (the runs used 580.82.09, CUDA 13.0, on
# RTX PRO 6000 Blackwell sm_120); no system CUDA toolkit is used (the CUDA libraries are pip wheels). Isaac Sim needs
# Vulkan libraries (libEGL, libGLU, libXt) for rendering; soma_to_g1.py does not render.
# Genesis (da59334) is not used by any Track-2 step and is not installed here.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
usage() { cat <<'EOF'
usage: setup_env.sh [--steps conda,stock,flashchord,kit,retarget,zxg2,lfs,check] [--envs DIR] [--mirror cn]
                    [--report-only] [--dry-run] [--site FILE]
  --steps LIST   default: all except check in the order above; `--steps check` alone is read-only
  --envs DIR     where the envs go (default $V2D_WORK/envs; then set FC_PYTHON, KIT_PYTHON, RETARGET_PYTHON,
                 ZX_KIT_PYTHON in your site file to <DIR>/{flashchord,v2dkit,v2d,zx_g2_builder}/bin/python)
  --report-only  env_check.py lists the differences from environment/*.pip-freeze.txt but does not fail on them
                 (default: any differing, missing or extra package stops the step with an error)
  --mirror cn    optional, for hosts in mainland China: Tsinghua conda/PyPI mirrors, and HF_ENDPOINT=https://hf-mirror.com
                 HF_HUB_DISABLE_XET=1 for get_data.sh. GitHub and pypi.nvidia.com may need an HTTP(S) proxy there
                 (export https_proxy yourself; the AutoDL training server provides one through its network_turbo
                 script). Large GitHub LFS transfers through such a proxy can break; retry or copy the files. The
                 PyTorch index download.pytorch.org/whl/cu128 answered directly (no proxy) from the training server.
needs: conda (miniconda 24.x was used), git, git-lfs (3.8.0 was used)
EOF
}
final_init "$@"
STEPS=conda,stock,flashchord,kit,retarget,zxg2,lfs; ENVS=""; MIRROR=""; REPORT=""
set -- ${REST[@]+"${REST[@]}"}
while [ $# -gt 0 ]; do
  case "$1" in
    --steps) STEPS=$2; shift ;; --envs) ENVS=$2; shift ;; --mirror) MIRROR=$2; shift ;; --report-only) REPORT=--report-only ;;
    *) die "unknown option $1 (see --help)" ;;
  esac; shift
done
: "${ENVS:=$V2D_WORK/envs}"
ENVDIR=$REPO/environment
REPRO=$TR/repro/env
has() { case ",$STEPS," in *",$1,"*) return 0;; *) return 1;; esac; }
CONDA_BASE=https://conda.anaconda.org/conda-forge; PIP_INDEX=https://pypi.org/simple
if [ "$MIRROR" = cn ]; then CONDA_BASE=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge; PIP_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple; fi

plan_header "environment setup: steps $STEPS (envs in $ENVS)${REPORT:+, env differences reported only}"
addq ENVS="$ENVS"; addq ENVDIR="$ENVDIR"; addq REPRO="$REPRO"; addq REPORT="$REPORT"
add 'mkdir -p "$ENVS" "$V2D_WORK/tmp"'
# env_check.py exits 1 on any difference from the recorded freeze (unless --report-only)
envcheck() { add "\"$1\" -I -B \"\$FINAL_LIB/env_check.py\" \${REPORT:+\"\$REPORT\"} \"\$ENVDIR/$2\"${3:-}"; }
if has conda; then
  add ''; add '# ---- conda: the two conda envs at the recorded conda packages (Python 3.11.17 from conda-forge)'
  for e in flashchord v2d; do
    add "if [ ! -x \"\$ENVS/$e/bin/python\" ]; then"
    add "  sed 's#https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge#$CONDA_BASE#' \"\$ENVDIR/$e.conda-explicit.txt\" > \"\$V2D_WORK/tmp/$e.conda-explicit.txt\""
    add "  conda create -y -p \"\$ENVS/$e\" --file \"\$V2D_WORK/tmp/$e.conda-explicit.txt\""
    add "fi"
  done
  add "\"\$ENVS/flashchord/bin/python\" -m pip install --index-url $(printf %q "$PIP_INDEX") uv==0.13.0"
fi
if has stock; then
  add ''; add '# ---- stock runtime nvidia-isaac/video_to_data @ 8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef (sparse, with LFS assets)'
  addq bash "$REPRO/get_stock_runtime.sh" "$V2D_WORK/video_to_data_8c778f9"
  add "note \"set V2D_RUNTIME=$V2D_WORK/video_to_data_8c778f9/robotic_grounding/flash_chord in your site file\""
  add '# the payload hull cache caches/stock_8c778f9/convex/*.npz goes into <runtime>/src/flash_chord/assets/.cache/convex/'
  add '# (without it the evaluator rebuilds hulls with CoACD, which is not deterministic; t1best/README_NEW_MACHINE.md)'
fi
if has flashchord; then
  add ''; add '# ---- flashchord: every package hash-locked from the stock uv.lock (= this repo'"'"'s flash_chord/uv.lock)'
  add "UV=\"\$ENVS/flashchord/bin/uv\" EXTRA_INDEX_URL=$( [ "$MIRROR" = cn ] && printf %q "$PIP_INDEX" ) bash \"\$REPRO/install_flashchord.sh\" \"\$ENVS/flashchord\" \"\$V2D_RUNTIME\""
  envcheck '$ENVS/flashchord/bin/python' flashchord.pip-freeze.txt
fi
if has kit; then
  add ''; add '# ---- kit: the Oct-11 submission kit (from the organisers) and its venv v2dkit'
  add 'check_sha256 "$V2D_KIT/KIT_MANIFEST.json" 398947a7783490245a47d70ec68d6894ecef7636e712532710ec448dc0ca1c3e'
  add 'check_sha256 "$V2D_KIT/flash_chord/scripts/evaluate_policy.py" 160d24f5fbeb70ccd569ca4192056f1892b2a9f2999066db9d8279461648fdf1'
  add 'check_sha256 "$V2D_KIT/tools/pack_submission.py" 47a4589574ff8284ba4376ef31ac5548c8fe87ead753be2d3b43d6c87041e5be'
  add 'check_sha256 "$V2D_KIT/submit/v2d_submit.py" fe5cb1dc86d12abd01eab9dd5dd98b51569b5d8513a2d00d4e79da42a5f11718'
  add '[ -x "$ENVS/v2dkit/bin/python" ] || "$ENVS/flashchord/bin/uv" venv --python "$ENVS/flashchord/bin/python" "$ENVS/v2dkit"'
  add "\"\$ENVS/flashchord/bin/uv\" pip install --python \"\$ENVS/v2dkit/bin/python\" --index-url $(printf %q "$PIP_INDEX") -r \"\$ENVDIR/v2dkit.pip-freeze.txt\""
  envcheck '$ENVS/v2dkit/bin/python' v2dkit.pip-freeze.txt
fi
if has retarget; then
  add ''; add '# ---- retarget env v2d: Isaac Sim 5.1.0 + Isaac Lab v2.3.2 + robotic_grounding (inside the env; TR/repro/README.md "Environments")'
  add 'export PIP_CONSTRAINT="$REPRO/isaac-constraints.txt" OMNI_KIT_ACCEPT_EULA=YES'
  add '# torch 2.7.0+cu128 / torchvision 0.22.0+cu128 (pinned by the constraints) exist only on the PyTorch cu128 index;'
  add '# isaacsim-core 5.1.0 requires torch==2.7.0, torchvision==0.22.0, torchaudio==2.7.0, which these builds satisfy'
  add "\"\$ENVS/v2d/bin/python\" -m pip install --index-url https://download.pytorch.org/whl/cu128 --extra-index-url $(printf %q "$PIP_INDEX") torch==2.7.0+cu128 torchvision==0.22.0+cu128"
  add '# torchaudio: the env has the PyPI build 2.7.0 (no +cu128 label; environment/v2d.pip-freeze.txt), not the cu128 one'
  add "\"\$ENVS/v2d/bin/python\" -m pip install --index-url $(printf %q "$PIP_INDEX") torchaudio==2.7.0"
  add "\"\$ENVS/v2d/bin/python\" -m pip install --index-url $(printf %q "$PIP_INDEX") --extra-index-url https://pypi.nvidia.com \"isaacsim[all,extscache]==5.1.0\""
  add '[ -d "$V2D_WORK/IsaacLab" ] || git clone --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git "$V2D_WORK/IsaacLab"'
  add '# isaaclab.sh installs into ${CONDA_PREFIX}/bin/python when CONDA_PREFIX is set (else ${VIRTUAL_ENV}): point it at this env'
  add '( cd "$V2D_WORK/IsaacLab" && git checkout -q 37ddf626871758333d6ed89cf64ad702aef127d0 \'
  add '  && env -u VIRTUAL_ENV CONDA_PREFIX="$ENVS/v2d" PATH="$ENVS/v2d/bin:$PATH" ./isaaclab.sh --install rsl_rl )'
  add '( env -u VIRTUAL_ENV CONDA_PREFIX="$ENVS/v2d" PATH="$ENVS/v2d/bin:$PATH" bash "$REPRO/install_retarget.sh" )   # pins + SOMA-X assets'
  envcheck '$ENVS/v2d/bin/python' v2d.pip-freeze.txt
  add '# environment/v2d.pip-freeze.txt also lists packages that no Track-2 step imports (Isaac Lab extras, rl_games).'
fi
if has zxg2; then
  add ''; add '# ---- zxg2: Zuxing Lu'"'"'s Tier-2 G2 input-builder env (ZX_KIT_PYTHON): a uv venv on CPython 3.12.3'
  add '[ -x "$ENVS/zx_g2_builder/bin/python" ] || "$ENVS/flashchord/bin/uv" venv --python 3.12.3 "$ENVS/zx_g2_builder"'
  add "\"\$ENVS/flashchord/bin/uv\" pip install --python \"\$ENVS/zx_g2_builder/bin/python\" --index-url $(printf %q "$PIP_INDEX") -r \"\$ENVDIR/zx_g2_builder.pip-freeze.txt\""
  envcheck '$ENVS/zx_g2_builder/bin/python' zx_g2_builder.pip-freeze.txt
  add "note \"set ZX_KIT_PYTHON=\$ENVS/zx_g2_builder/bin/python in your site file\""
fi
if has lfs; then
  add ''; add '# ---- Git LFS of this checkout (clone with GIT_LFS_SKIP_SMUDGE=1 first; REPRODUCE.md section 1)'
  add 'git -C "$REPO" lfs install --local'
  add 'git -C "$REPO" config lfs.url https://github.com/nvidia-isaac/video_to_data.git/info/lfs'
  add '# FlashCHORD and robotic_grounding assets, and the 3 runtime LFS files in reconstruction/modules that Han Zhou'"'"'s'
  add '# run_all_tier3.sh needs (lib_mhr/assets/mhr_collision_proxy_4000v.npz, mhr_hand_surface_spec.npz, v2d_sam3d_body/assets/bbox_track.pt)'
  add 'git -C "$REPO" lfs pull --include "robotic_grounding/flash_chord/**,robotic_grounding/source/robotic_grounding/robotic_grounding/assets/**,reconstruction/modules/v2d_cari4d/lib/cari4d/lib_mhr/assets/**,reconstruction/modules/v2d_sam3d_body/assets/**"'
  add '# Han Zhou'"'"'s hoi_estimation/agent_records/*/outputs/*.{npz,npy} (41 files) are in the fork'"'"'s own LFS store:'
  add 'git -C "$REPO" -c lfs.url=https://github.com/Beat-in-our-hearts/nv-v2d-challenge-codebase.git/info/lfs lfs pull --include "hoi_estimation/**"'
  add '"$PLAN_PY" -I -B "$FINAL_LIB/lfs_fill.py" --worktree "$REPO" --check-only --report "$V2D_WORK/logs/lfs_check_repo.tsv"'
  add 'for d in reconstruction/modules/v2d_cari4d/lib/cari4d/lib_mhr/assets reconstruction/modules/v2d_sam3d_body/assets; do'
  add '  "$PLAN_PY" -I -B "$FINAL_LIB/lfs_fill.py" --worktree "$REPO" --subdir "$d" --check-only --report "$V2D_WORK/logs/lfs_check_$(basename "$(dirname "$d")").tsv"'
  add 'done'
fi
if has check; then
  add ''; add '# ---- check (read-only): the envs against the recorded freezes, the GPU driver, the kit and the stock runtime'
  add 'nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv'
  add 'ENV_FAIL=0'
  envcheck '$FC_PYTHON' flashchord.pip-freeze.txt ' || ENV_FAIL=1'
  envcheck '$KIT_PYTHON' v2dkit.pip-freeze.txt ' || ENV_FAIL=1'
  envcheck '$RETARGET_PYTHON' v2d.pip-freeze.txt ' || ENV_FAIL=1'
  addq ZX_KIT_PYTHON="$ZX_KIT_PYTHON"
  add 'if [ -x "$ZX_KIT_PYTHON" ]; then'
  envcheck '$ZX_KIT_PYTHON' zx_g2_builder.pip-freeze.txt ' || ENV_FAIL=1'
  add 'else note "ZX_KIT_PYTHON $ZX_KIT_PYTHON not here: the Tier-2 G2 builder env is not checked (setup_env.sh --steps zxg2)"; fi'
  add '[ "$ENV_FAIL" = 0 ] || die "an env differs from its record in environment/ (listed above); --report-only lists without failing"'
  add 'check_eval_tools'
  add 'check_sha256 "$V2D_KIT/tools/pack_submission.py" 47a4589574ff8284ba4376ef31ac5548c8fe87ead753be2d3b43d6c87041e5be'
  add 'check_sha256 "$V2D_KIT/submit/v2d_submit.py" fe5cb1dc86d12abd01eab9dd5dd98b51569b5d8513a2d00d4e79da42a5f11718'
  add '[ "$(git -C "$V2D_RUNTIME" rev-parse HEAD)" = 8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef ] && note "stock runtime at 8c778f9" || die "stock runtime is not at 8c778f9"'
fi
printf '%s' "$PLAN" | run_plan setup_env
