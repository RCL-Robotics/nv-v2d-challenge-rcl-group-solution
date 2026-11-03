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
#   retarget    env v2d: conda Python 3.11.17, Isaac Sim 5.1.0 (pypi.nvidia.com), Isaac Lab v2.3.2 (37ddf62), torch
#               2.7.0+cu128, robotic_grounding (TR/repro/env/install_retarget.sh; full freeze environment/v2d.pip-freeze.txt)
#               used for the G1 retargeting and for Tier-2 / Tier-3 input building
#   lfs         Git LFS of this checkout: FlashCHORD assets from upstream nvidia-isaac/video_to_data (the fork hosts no
#               LFS objects), Han Zhou's 41 hoi_estimation records from the fork's own LFS store
#   check       read-only: versions of the three envs against environment/*.pip-freeze.txt, GPU/driver, kit hashes
# CUDA / driver: the jax-cuda13 wheels need an NVIDIA driver for CUDA 13 (the runs used 580.82.09, CUDA 13.0, on
# RTX PRO 6000 Blackwell sm_120); no system CUDA toolkit is used (the CUDA libraries are pip wheels). Isaac Sim needs
# Vulkan libraries (libEGL, libGLU, libXt) for rendering; soma_to_g1.py does not render.
# Genesis (da59334) is not used by any Track-2 step: it belongs to a separate project (env dexmachina), not installed here.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
usage() { cat <<'EOF'
usage: setup_env.sh [--steps conda,stock,flashchord,kit,retarget,lfs,check] [--envs DIR] [--mirror cn]
                    [--dry-run] [--site FILE]
  --steps LIST   default: all except check in the order above; `--steps check` alone is read-only
  --envs DIR     where the conda envs go (default $V2D_WORK/envs; then set FC_PYTHON, KIT_PYTHON, RETARGET_PYTHON
                 in your site file to <DIR>/{flashchord,v2dkit,v2d}/bin/python)
  --mirror cn    optional, for hosts in mainland China: Tsinghua conda/PyPI mirrors, and HF_ENDPOINT=https://hf-mirror.com
                 HF_HUB_DISABLE_XET=1 for get_data.sh. GitHub and pypi.nvidia.com may need an HTTP(S) proxy there
                 (export https_proxy yourself; the AutoDL training server provides one through its network_turbo
                 script). Large GitHub LFS transfers through such a proxy can break; retry or copy the files.
needs: conda (miniconda 24.x was used), git, git-lfs (3.8.0 was used)
EOF
}
final_init "$@"
STEPS=conda,stock,flashchord,kit,retarget,lfs; ENVS=""; MIRROR=""
set -- ${REST[@]+"${REST[@]}"}
while [ $# -gt 0 ]; do
  case "$1" in
    --steps) STEPS=$2; shift ;; --envs) ENVS=$2; shift ;; --mirror) MIRROR=$2; shift ;;
    *) die "unknown option $1 (see --help)" ;;
  esac; shift
done
: "${ENVS:=$V2D_WORK/envs}"
ENVDIR=$REPO/environment
REPRO=$TR/repro/env
has() { case ",$STEPS," in *",$1,"*) return 0;; *) return 1;; esac; }
CONDA_BASE=https://conda.anaconda.org/conda-forge; PIP_INDEX=https://pypi.org/simple
if [ "$MIRROR" = cn ]; then CONDA_BASE=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge; PIP_INDEX=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple; fi

plan_header "environment setup: steps $STEPS (envs in $ENVS)"
addq ENVS="$ENVS"; addq ENVDIR="$ENVDIR"; addq REPRO="$REPRO"
add 'mkdir -p "$ENVS" "$V2D_WORK/tmp"'
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
  add '"$ENVS/flashchord/bin/python" -I -B "$FINAL_LIB/env_check.py" "$ENVDIR/flashchord.pip-freeze.txt"'
fi
if has kit; then
  add ''; add '# ---- kit: the Oct-11 submission kit (from the organisers) and its venv v2dkit'
  add 'check_sha256 "$V2D_KIT/KIT_MANIFEST.json" 398947a7783490245a47d70ec68d6894ecef7636e712532710ec448dc0ca1c3e'
  add 'check_sha256 "$V2D_KIT/flash_chord/scripts/evaluate_policy.py" 160d24f5fbeb70ccd569ca4192056f1892b2a9f2999066db9d8279461648fdf1'
  add 'check_sha256 "$V2D_KIT/tools/pack_submission.py" 47a4589574ff8284ba4376ef31ac5548c8fe87ead753be2d3b43d6c87041e5be'
  add 'check_sha256 "$V2D_KIT/submit/v2d_submit.py" fe5cb1dc86d12abd01eab9dd5dd98b51569b5d8513a2d00d4e79da42a5f11718'
  add '[ -x "$ENVS/v2dkit/bin/python" ] || "$ENVS/flashchord/bin/uv" venv --python "$ENVS/flashchord/bin/python" "$ENVS/v2dkit"'
  add "\"\$ENVS/flashchord/bin/uv\" pip install --python \"\$ENVS/v2dkit/bin/python\" --index-url $(printf %q "$PIP_INDEX") -r \"\$ENVDIR/v2dkit.pip-freeze.txt\""
  add '"$ENVS/v2dkit/bin/python" -I -B "$FINAL_LIB/env_check.py" "$ENVDIR/v2dkit.pip-freeze.txt"'
fi
if has retarget; then
  add ''; add '# ---- retarget env v2d: Isaac Sim 5.1.0 + Isaac Lab v2.3.2 + robotic_grounding (inside the env; TR/repro/README.md "Environments")'
  add 'export PIP_CONSTRAINT="$REPRO/isaac-constraints.txt" OMNI_KIT_ACCEPT_EULA=YES'
  add '"$ENVS/v2d/bin/python" -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com'
  add '[ -d "$V2D_WORK/IsaacLab" ] || git clone --branch v2.3.2 https://github.com/isaac-sim/IsaacLab.git "$V2D_WORK/IsaacLab"'
  add '( cd "$V2D_WORK/IsaacLab" && git checkout -q 37ddf626871758333d6ed89cf64ad702aef127d0 && PATH="$ENVS/v2d/bin:$PATH" ./isaaclab.sh --install rsl_rl )'
  add '( PATH="$ENVS/v2d/bin:$PATH" bash "$REPRO/install_retarget.sh" )   # pins + SOMA-X assets; lists differences from the recorded freeze'
  add '"$ENVS/v2d/bin/python" -I -B "$FINAL_LIB/env_check.py" "$ENVDIR/v2d.pip-freeze.txt"'
  add '# environment/v2d.pip-freeze.txt also lists packages of other projects that shared the env on the server'
  add '# (dexmachina, IsaacLab extras); no Track-2 step imports them.'
fi
if has lfs; then
  add ''; add '# ---- Git LFS of this checkout (clone with GIT_LFS_SKIP_SMUDGE=1 first; REPRODUCE.md section 1)'
  add 'git -C "$REPO" lfs install --local'
  add 'git -C "$REPO" config lfs.url https://github.com/nvidia-isaac/video_to_data.git/info/lfs'
  add 'git -C "$REPO" lfs pull --include "robotic_grounding/flash_chord/**,robotic_grounding/source/robotic_grounding/robotic_grounding/assets/**"'
  add '# Han Zhou'"'"'s hoi_estimation/agent_records/*/outputs/*.{npz,npy} (41 files) are in the fork'"'"'s own LFS store:'
  add 'git -C "$REPO" -c lfs.url=https://github.com/Beat-in-our-hearts/nv-v2d-challenge-codebase.git/info/lfs lfs pull --include "hoi_estimation/**"'
  add '"$PLAN_PY" -I -B "$FINAL_LIB/lfs_fill.py" --worktree "$REPO" --check-only --report "$V2D_WORK/logs/lfs_check_repo.tsv"'
fi
if has check; then
  add ''; add '# ---- check (read-only): the envs against the recorded freezes, the GPU driver, the kit and the stock runtime'
  add 'nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv'
  add '"$FC_PYTHON" -I -B "$FINAL_LIB/env_check.py" "$ENVDIR/flashchord.pip-freeze.txt"'
  add '"$KIT_PYTHON" -I -B "$FINAL_LIB/env_check.py" "$ENVDIR/v2dkit.pip-freeze.txt"'
  add '"$RETARGET_PYTHON" -I -B "$FINAL_LIB/env_check.py" "$ENVDIR/v2d.pip-freeze.txt"'
  add 'check_eval_tools'
  add 'check_sha256 "$V2D_KIT/tools/pack_submission.py" 47a4589574ff8284ba4376ef31ac5548c8fe87ead753be2d3b43d6c87041e5be'
  add 'check_sha256 "$V2D_KIT/submit/v2d_submit.py" fe5cb1dc86d12abd01eab9dd5dd98b51569b5d8513a2d00d4e79da42a5f11718'
  add '[ "$(git -C "$V2D_RUNTIME" rev-parse HEAD)" = 8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef ] && note "stock runtime at 8c778f9" || die "stock runtime is not at 8c778f9"'
fi
printf '%s' "$PLAN" | run_plan setup_env
