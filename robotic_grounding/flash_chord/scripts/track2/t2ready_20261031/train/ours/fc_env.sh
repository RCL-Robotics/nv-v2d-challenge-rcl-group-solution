# 用法： source /root/autodl-tmp/haoran.zhang/scripts/fc_env.sh   （FlashCHORD 环境，与 v2d/Isaac 环境分开）
ME=/root/autodl-tmp/haoran.zhang
export PATH=$ME/conda_envs/flashchord/bin:$PATH
export FC_ROOT=$ME/code/video_to_data_8c778f9/robotic_grounding/flash_chord
export HUMAN_MOTION_DATA_DIR=$ME/data/human_motion_data
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # 让同一张卡能放多个任务（JAX 默认会预占 75% 显存）
export WANDB_DIR=$ME/fc_runs/wandb_local UV_CACHE_DIR=$ME/.cache/uv
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
# onnxruntime-gpu（SONIC 先验网络）找不到 pip 版 CUDA 13 / cuDNN 库（新 wheel 布局 nvidia/cu13/lib）；
# 官方 Docker 里由 CUDA 13 基础镜像提供，这里把 pip 装的库目录加入 LD_LIBRARY_PATH
_SP=$ME/conda_envs/flashchord/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH=$_SP/cu13/lib:$_SP/cudnn/lib:$_SP/nccl/lib:$_SP/nvshmem/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
