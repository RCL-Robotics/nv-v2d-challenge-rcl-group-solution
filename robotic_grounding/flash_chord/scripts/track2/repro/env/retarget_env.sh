# SPDX-License-Identifier: Apache-2.0
# Source before data conversion and retargeting:  source env/retarget_env.sh
#   RETARGET_ACTIVATE  optional script that activates your Isaac Sim 5.1 / Isaac Lab 2.3.2 environment
#                      (for example a file containing `conda activate /path/to/env`)
#   RETARGET_PYTHON    that environment's python (default: `python` after activation)
#   V2D_MOTION         derived motion data root; exported as HUMAN_MOTION_DATA_DIR
# Equivalent of the activation script of the original run, minus machine-specific parts (caches, package mirrors,
# and a container-only fix that added missing libEGL/libGLU/libXt libraries to LD_LIBRARY_PATH for Isaac Sim's
# Vulkan renderer; soma_to_g1.py itself does not render).
if [ -n "${RETARGET_ACTIVATE:-}" ]; then
  # shellcheck disable=SC1090
  source "$RETARGET_ACTIVATE"
fi
_rt_here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PIP_CONSTRAINT="$_rt_here/isaac-constraints.txt"   # keep torch/isaacsim/numpy/pyarrow pinned for any later pip install
export OMNI_KIT_ACCEPT_EULA=YES
if [ -n "${V2D_MOTION:-}" ]; then export HUMAN_MOTION_DATA_DIR="$V2D_MOTION"; fi
unset _rt_here
