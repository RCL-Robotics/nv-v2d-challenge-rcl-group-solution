# Environments of the Track 2 runs (exported 2026-11-04)

These files record the three Python environments, the GPU, driver and operating system of the machine that trained,
exported, evaluated and packed every counted checkpoint (the team server `corl2026_pro6000`; Zuxing Lu's RTX 5090 and
ws031 hosts used the same `flash_chord/uv.lock`, see `REPRODUCE.md` section 9). They were exported from the live
environments on 2026-11-04, after all counted runs, from the same env folders those runs used.

| env | used for | files | how to rebuild |
|---|---|---|---|
| `flashchord` | training, export, official evaluation, input builders that need FlashCHORD | `flashchord.conda.yml`, `flashchord.conda-explicit.txt` (conda Python 3.11.17), `flashchord.pip-freeze.txt` (108 packages: jax 0.10.2 + jax-cuda13 wheels, warp-lang 1.14.0, newton 1.2.0, mujoco / mujoco-warp 3.8.1, onnxruntime-gpu 1.28.0, hydra-core 1.3.2) | `scripts/final/setup_env.sh --steps conda,stock,flashchord`: the conda packages, uv 0.13.0, then every pip package hash-locked by the stock runtime's `uv.lock` (`robotic_grounding/flash_chord/scripts/track2/repro/env/install_flashchord.sh`) |
| `v2dkit` | the submission kit: packer, uploader validation, metric code | `v2dkit.pip-freeze.txt` (38 packages, pandas 3.0.6, pyarrow 26.0.0), `v2dkit.pyvenv.cfg` (a uv venv on the flashchord Python) | `setup_env.sh --steps kit` |
| `v2d` | G1 retargeting (`soma_to_g1.py`), Tier-2 K3/K3fix and Tier-3 data builders | `v2d.conda.yml`, `v2d.conda-explicit.txt`, `v2d.pip-freeze.txt` (Isaac Sim 5.1.0, Isaac Lab v2.3.2 @ 37ddf62, torch 2.7.0+cu128, numpy 1.26.0) | `setup_env.sh --steps conda,retarget` (`repro/env/install_retarget.sh`, constraints `repro/env/isaac-constraints.txt`) |

`system.txt`: 8x NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120, 97,887 MiB), driver 580.82.09 (CUDA 13.0),
Ubuntu 22.04.5 (container), glibc 2.35, 2x Xeon Platinum 8470Q, git 2.34.1, git-lfs 3.8.0, uv 0.13.0, conda 24.4.0. The
CUDA libraries of the flashchord env are pip wheels (`nvidia-*` in its freeze); no system CUDA toolkit is used. Any
driver that runs CUDA 13 should work for FlashCHORD; Isaac Sim 5.1 needs its own supported driver and Vulkan libraries.

Notes:
- The conda files name the Tsinghua conda-forge mirror the server used; the packages are the conda-forge ones
  (`setup_env.sh` rewrites the URLs to conda.anaconda.org unless `--mirror cn`).
- `flashchord.pip-freeze.txt` lists `flash_chord` as an editable install of the stock runtime
  (nvidia-isaac/video_to_data @ 8c778f9). Training and export put this repository's `flash_chord/src` first on
  `PYTHONPATH`; the evaluator runs the stock runtime's.
- `v2d.pip-freeze.txt` is the whole env as it was, including packages of other projects that shared it on the server
  (an editable `dexmachina` from a separate repository, Isaac Lab extensions, `rl_games`); no Track-2 step imports
  them. Genesis is not used by Track 2 and is not installed in these envs.
- `scripts/final/setup_env.sh --steps check` compares installed envs with these files (`scripts/final/lib/env_check.py`).
