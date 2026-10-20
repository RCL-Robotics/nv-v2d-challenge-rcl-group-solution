#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Retargeting environment (data conversion + soma_to_g1.py), as used for the C3r2 data: a Python 3.11 environment
# with Isaac Sim 5.1.0 and Isaac Lab 2.3.2 (rsl_rl), torch 2.7.0+cu128, and robotic_grounding with the dependencies of
# robotic_grounding/workflow/Dockerfile. That Dockerfile (base nvcr.io/nvidia/isaac-lab:2.3.2) is the reference
# environment; this script is the pip route used instead, run INSIDE the activated environment, after Isaac Sim 5.1.0
# (pip package isaacsim[all,extscache]==5.1.0 from https://pypi.nvidia.com) and Isaac Lab 2.3.2 (`isaaclab.sh --install`)
# are installed.
#
# usage: install_retarget.sh        (uses `python -m pip` of the active environment)
# Every pip call is constrained by isaac-constraints.txt (pins torch, isaacsim, numpy 1.26.0 and pyarrow 20.0.0;
# without it pip upgrades torch to a CUDA 13 build and pyarrow to a numpy-2-only release). The retarget stack
# (pinocchio/cmeel) temporarily wants numpy>=2.2, so it is installed against the constraints without numpy first and
# numpy is put back to Isaac Sim's exact 1.26.0 afterwards - the same order as the Dockerfile.
# The packages that shape the retargeted data (warp-lang for SOMA's skinning, scipy, mujoco, the QP solvers, ...) are
# pinned to the versions that made the C3r2 data. retarget-env-freeze.txt records the whole environment as it was
# (`pip list --format=freeze`); at the end this script lists every installed version that differs from it.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); RG=$(cd "$HERE/../../../../.." && pwd)   # robotic_grounding/
PIP=(python -m pip install)
nonumpy=$(mktemp); grep -v -i "^numpy==" "$HERE/isaac-constraints.txt" > "$nonumpy"
fail=0; ok(){ echo "[OK] $*"; }; bad(){ echo "[FAIL] $*"; fail=1; }
export PIP_CONSTRAINT="$HERE/isaac-constraints.txt"
"${PIP[@]}" tensordict==0.8.3 datasets==2.19.0 wandb==0.22.2 importlib_metadata plotly viser==1.0.15 dearpygui \
  scipy==1.15.3 trimesh==4.5.1 scikit-learn tzdata && ok "common dependencies" || bad "common dependencies"
"${PIP[@]}" py-soma-x==0.2.1 warp-lang==1.18.0 && ok py-soma-x || bad py-soma-x
export PIP_CONSTRAINT="$nonumpy"   # still pins torch / isaacsim
"${PIP[@]}" mujoco==3.6.0 judo-rai==0.1.0 pin==3.7.0 pin-pink==4.2.0 coal==3.0.1 eigenpy==3.10.3 cmeel==0.57.3 \
  cmeel-boost==1.87.0.1 cmeel-urdfdom==4.0.1 cmeel-console-bridge==1.0.2.3 cmeel-tinyxml2==10.0.0 \
  cmeel-tinyxml==2.6.2.3 cmeel-assimp==5.4.3.1 cmeel-octomap==1.10.0 cmeel-qhull==8.0.2.1 cmeel-zlib==1.3.1 \
  usd-core==26.8 deprecation && ok "retarget dependencies" || bad "retarget dependencies"
export PIP_CONSTRAINT="$HERE/isaac-constraints.txt"
"${PIP[@]}" --upgrade osqp==1.1.3 "qpsolvers[clarabel,daqp,proxqp,scs,osqp]==4.9.0" clarabel==0.11.1 daqp==0.7.2 \
  proxsuite==0.7.3 scs==3.3.1 && ok qpsolvers || bad qpsolvers
"${PIP[@]}" onnxruntime-gpu==1.24.4 imageio imageio-ffmpeg pyrender pytorch-ignite pytest boto3 pyzmq msgpack-numpy \
  && ok "robotic_grounding dependencies" || bad "robotic_grounding dependencies"
"${PIP[@]}" "numpy==1.26.0" && ok "numpy 1.26.0" || bad "numpy 1.26.0"
"${PIP[@]}" --no-build-isolation -e "$RG/source/robotic_grounding" && ok "robotic_grounding (editable)" || bad "robotic_grounding"
rm -f "$nonumpy"
# SOMA-X body model assets (pinned revision) for soma_to_g1.py, as in the Dockerfile
( cd "$RG" && python scripts/setup_soma_assets.py ) && ok "SOMA-X assets" || bad "SOMA-X assets (scripts/setup_soma_assets.py)"
python - <<'PY' && ok imports || bad "imports (see [import FAIL] above)"
import importlib, sys
failed = 0
for name in ["numpy", "torch", "pinocchio", "pink", "coal", "mujoco", "qpsolvers", "pxr", "pyarrow", "scipy", "warp",
             "robotic_grounding", "isaaclab"]:
    try:
        module = importlib.import_module(name)
        print(f"[import OK] {name} {getattr(module, '__version__', '')}")
    except Exception as error:  # noqa: BLE001
        print(f"[import FAIL] {name}: {type(error).__name__}: {error}")
        failed += 1
sys.exit(1 if failed else 0)
PY
python -m pip check 2>&1 | tail -20
# differences from the recorded environment (informational: Isaac Sim / Isaac Lab installs can differ in details)
python - "$HERE/retarget-env-freeze.txt" <<'PY'
import re, sys
from importlib import metadata
def canon(name):
    return re.sub(r"[-_.]+", "-", name).lower()
recorded = {}
for line in open(sys.argv[1]):
    line = line.strip()
    if line and not line.startswith("#") and "==" in line:
        name, version = line.split("==", 1)
        recorded[canon(name)] = version
installed = {canon(d.metadata["Name"]): d.version for d in metadata.distributions() if d.metadata["Name"]}
differ = sorted(n for n in recorded if n in installed and installed[n] != recorded[n])
missing = sorted(n for n in recorded if n not in installed)
extra = sorted(n for n in installed if n not in recorded)
for n in differ:
    print(f"[env differs] {n}: installed {installed[n]}, recorded {recorded[n]}")
for n in missing:
    print(f"[env missing] {n}=={recorded[n]}")
for n in extra:
    print(f"[env extra] {n}=={installed[n]}")
print(f"recorded environment: {len(recorded)} packages; {len(differ)} differ, {len(missing)} missing, {len(extra)} extra")
PY
[ $fail = 0 ] && echo RETARGET_ENV_DONE || { echo "some steps failed (see [FAIL] above)" >&2; exit 1; }
