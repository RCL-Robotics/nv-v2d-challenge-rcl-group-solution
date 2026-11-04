# Environments

The Python environments, GPU, driver and operating system used to train, evaluate and pack the submitted policies.
`scripts/final/setup_env.sh` installs them from these files. Reproducing the submitted files needs only `flashchord`
and `v2dkit` ([`REPRODUCE.md`](../REPRODUCE.md) section 1).

| env | used for | files | install |
|---|---|---|---|
| `flashchord` | training, export, official evaluation | `flashchord.conda.yml`, `flashchord.conda-explicit.txt` (Python 3.11.17), `flashchord.pip-freeze.txt` (108 packages, hash-locked by the official runtime's `uv.lock`) | `setup_env.sh --steps conda,stock,flashchord` |
| `v2dkit` | the submission kit: packer and checks | `v2dkit.pip-freeze.txt` (38 packages), `v2dkit.pyvenv.cfg` (a uv venv on the flashchord Python) | `setup_env.sh --steps kit` |
| `v2d` | G1 retargeting and the input builders | `v2d.conda.yml`, `v2d.conda-explicit.txt`, `v2d.pip-freeze.txt` (Isaac Sim 5.1.0, Isaac Lab v2.3.2) | `setup_env.sh --steps conda,retarget` |
| `zx_g2_builder` | one step of the Tier-2 input builder | `zx_g2_builder.pip-freeze.txt` (a uv venv on CPython 3.12.3) | `setup_env.sh --steps zxg2` |

`system.txt`: NVIDIA RTX PRO 6000 Blackwell GPUs, driver 580.82.09 (CUDA 13.0), Ubuntu 22.04, git-lfs 3.8.0, uv 0.13.0,
conda 24.4.0. The CUDA libraries come as pip wheels, so no CUDA toolkit is needed; Isaac Sim 5.1 needs its own
supported driver and Vulkan libraries.

To compare an installed env with its file: `<env python> -I -B scripts/final/lib/env_check.py
environment/<env>.pip-freeze.txt` (exit status 1 on any difference), or `setup_env.sh --steps check` for all four.
