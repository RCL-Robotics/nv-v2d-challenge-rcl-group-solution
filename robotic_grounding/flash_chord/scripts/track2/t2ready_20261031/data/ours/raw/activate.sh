# 用法： source /root/autodl-tmp/haoran.zhang/activate.sh
export PATH=/root/autodl-tmp/haoran.zhang/tools/gh_2.63.2_linux_amd64/bin:$PATH
export CONDA_PKGS_DIRS=/root/autodl-tmp/haoran.zhang/conda_pkgs
export CONDA_ENVS_PATH=/root/autodl-tmp/haoran.zhang/conda_envs   # 让 conda activate dexmachina / arctic_env 能按名字找到
export PIP_CACHE_DIR=/root/autodl-tmp/haoran.zhang/.cache/pip
export PIP_INDEX_URL=https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
export HF_HOME=/root/autodl-tmp/haoran.zhang/.cache/huggingface
export XDG_CACHE_HOME=/root/autodl-tmp/haoran.zhang/.cache            # Isaac Sim / Omniverse 缓存也放这里
export XDG_DATA_HOME=/root/autodl-tmp/haoran.zhang/.local/share
export OMNI_KIT_ACCEPT_EULA=YES
# 镜像里缺 libEGL/libGLU/libXt（Isaac Sim 的 Vulkan/渲染需要），从 Ubuntu deb 解包到个人目录，不改系统；
# 另外链接 v2d 环境自带的新版 libstdc++（系统版太旧，Isaac Sim 进程里 import sqlite3 会报 CXXABI_1.3.15）
export LD_LIBRARY_PATH=/root/autodl-tmp/haoran.zhang/tools/syslibs/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# 这台容器里 NCCL 的 GPU P2P 和共享内存传输都会报 illegal memory access，只能走 socket（实测 4.7GB/s，PPO 梯度同步足够）
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
export ISAACLAB_PATH=/root/autodl-tmp/haoran.zhang/code/IsaacLab
for d in $(conda info --base 2>/dev/null) /root/miniconda3 /root/anaconda3 /opt/conda; do [ -f $d/etc/profile.d/conda.sh ] && { source $d/etc/profile.d/conda.sh; break; }; done
conda activate /root/autodl-tmp/haoran.zhang/conda_envs/v2d 2>/dev/null || true
cd /root/autodl-tmp/haoran.zhang
