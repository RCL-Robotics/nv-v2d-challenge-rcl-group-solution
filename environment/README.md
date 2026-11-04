# Environments of the Track 2 runs (exported 2026-11-04)

These files record the three Python environments, the GPU, driver and operating system of the machine that trained,
exported, evaluated and packed every counted checkpoint (the team's training server; Zuxing Lu's RTX 5090 and
ws031 hosts used the same `flash_chord/uv.lock`, see `REPRODUCE.md` section 9). They were exported from the live
environments on 2026-11-04, after all counted runs, from the same env folders those runs used.

| env | used for | files | how to rebuild |
|---|---|---|---|
| `flashchord` | training, export, official evaluation, input builders that need FlashCHORD | `flashchord.conda.yml`, `flashchord.conda-explicit.txt` (conda Python 3.11.17), `flashchord.pip-freeze.txt` (108 packages: jax 0.10.2 + jax-cuda13 wheels, warp-lang 1.14.0, newton 1.2.0, mujoco / mujoco-warp 3.8.1, onnxruntime-gpu 1.28.0, hydra-core 1.3.2) | `scripts/final/setup_env.sh --steps conda,stock,flashchord`: the conda packages, uv 0.13.0, then every pip package hash-locked by the stock runtime's `uv.lock` (`robotic_grounding/flash_chord/scripts/track2/repro/env/install_flashchord.sh`) |
| `v2dkit` | the submission kit: packer, uploader validation, metric code | `v2dkit.pip-freeze.txt` (38 packages, pandas 3.0.6, pyarrow 26.0.0), `v2dkit.pyvenv.cfg` (a uv venv on the flashchord Python) | `setup_env.sh --steps kit` |
| `v2d` | G1 retargeting (`soma_to_g1.py`), Tier-2 K3/K3fix and Tier-3 data builders | `v2d.conda.yml`, `v2d.conda-explicit.txt`, `v2d.pip-freeze.txt` (332 packages: Isaac Sim 5.1.0, Isaac Lab v2.3.2 @ 37ddf62, torch 2.7.0+cu128 and torchvision 0.22.0+cu128 from the PyTorch cu128 index, torchaudio 2.7.0 from PyPI, numpy 1.26.0) | `setup_env.sh --steps conda,retarget` (`repro/env/install_retarget.sh`, constraints `repro/env/isaac-constraints.txt`; the packages that script asks for without a version are installed first at the versions of `v2d.pip-freeze.txt`, see below) |
| `zx_g2_builder` | Zuxing Lu's Tier-2 G2 input builder (`ZX_KIT_PYTHON`; on the training server `<ZX>/envs/kit`) | `zx_g2_builder.pip-freeze.txt` (28 packages; a uv venv on CPython 3.12.3; recorded read-only on 2026-11-04) | `setup_env.sh --steps zxg2` |

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
- `v2d.pip-freeze.txt` is the whole env as it was, including packages that no Track-2 step imports (Isaac Lab
  extensions, `rl_games`), with two edits: one unrelated local editable package of another project that shared the env
  on the server is left out (a comment line marks the place; `env_check.py` tolerates it as an extra on that server),
  and `flatdict` (installed from a locally built wheel; PyPI has only the sdist of 4.0.1) is listed as
  `flatdict==4.0.1` (`setup_env.sh` builds that sdist with setuptools 80.9.0: its `setup.py` needs `pkg_resources`).
  Genesis is not used by Track 2 and is not installed in these envs.
- `scripts/final/setup_env.sh --steps check` compares installed envs with these files (`scripts/final/lib/env_check.py`)
  and fails (exit status 1) if any package differs, is missing or is extra; `--report-only` only lists the
  differences. The same comparison runs after each install step of `setup_env.sh`; there it lists the differences and
  the install goes on, with a summary at the end (`--strict-install`: stop at the first env that differs).
- `v2d.pip-freeze.txt` also pins, at install time, the packages that `repro/env/install_retarget.sh` asks for without
  a version (`scripts/final/lib/freeze_constraints.py` writes its `name==version` lines, without the packages that
  `isaac-constraints.txt` pins, after checking that their versions agree).
