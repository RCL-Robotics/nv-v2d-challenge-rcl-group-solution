# SPDX-License-Identifier: Apache-2.0
# Source before any FlashCHORD step (training, export, evaluation):  source env/fc_env.sh
#   FC_PYTHON   python of the flashchord environment (env/install_flashchord.sh); default: `python` on PATH
#   V2D_MOTION  derived motion data root; exported as HUMAN_MOTION_DATA_DIR
# Sets:
#   PATH                  the flashchord env's bin/ first, so `python` is that interpreter (as in the original runs)
#   LD_LIBRARY_PATH       the env's pip-installed CUDA 13 / cuDNN / NCCL / NVSHMEM libraries: onnxruntime-gpu (the
#                         SONIC prior) does not find the nvidia/cu13 wheel layout by itself. The reference Docker
#                         image (robotic_grounding/flash_chord/docker) gets them from its CUDA 13 base image instead.
#   XLA_PYTHON_CLIENT_PREALLOCATE=false   several jobs can share a GPU (JAX would otherwise take 75% of its memory)
#   HUMAN_MOTION_DATA_DIR = V2D_MOTION
if [ -n "${FC_PYTHON:-}" ]; then
  PATH="$(dirname "$FC_PYTHON"):$PATH"; export PATH
fi
_fc_python=${FC_PYTHON:-$(command -v python)}
_fc_site=$("$_fc_python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' 2>/dev/null)
_fc_libs=""
for _fc_d in cu13 cudnn nccl nvshmem; do
  if [ -d "$_fc_site/nvidia/$_fc_d/lib" ]; then _fc_libs="${_fc_libs:+$_fc_libs:}$_fc_site/nvidia/$_fc_d/lib"; fi
done
if [ -n "$_fc_libs" ]; then export LD_LIBRARY_PATH="$_fc_libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [ -n "${V2D_MOTION:-}" ]; then export HUMAN_MOTION_DATA_DIR="$V2D_MOTION"; fi
unset _fc_python _fc_site _fc_libs _fc_d
