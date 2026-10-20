#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the flashchord environment exactly as for the C3r2 run: every package pinned with hashes by the uv.lock of
# the stock runtime (nvidia-isaac/video_to_data@8c778f9; robotic_grounding/flash_chord/uv.lock of this repo is
# byte-identical to it), installed into a separate Python 3.11 environment, plus an editable install of the stock
# runtime. Training and export later run THIS repo's flash_chord/src through PYTHONPATH, evaluation the stock
# runtime's src (eval_all.py sets both), so the editable install only provides the package metadata.
#
# usage: install_flashchord.sh <env dir> [stock runtime dir]
#   <env dir>             created with `uv venv --python 3.11` when it has no bin/python yet
#   [stock runtime dir]   .../video_to_data_8c778f9/robotic_grounding/flash_chord (default: $V2D_RUNTIME)
#   env: UV=uv (uv 0.13.0 was used)   EXTRA_INDEX_URL (optional extra package index, e.g. a nearby mirror)
# The reference alternative is the Docker image in robotic_grounding/flash_chord/docker (CUDA 13.0.2,
# `uv sync --frozen --extra train --extra dev`), which installs the same lock.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd); FC=$(cd "$HERE/../../../.." && pwd)   # robotic_grounding/flash_chord
ENV=${1:?usage: install_flashchord.sh <env dir> [stock runtime dir]}; STOCK=${2:-${V2D_RUNTIME:-}}
UV=${UV:-uv}; REQ=$HERE/requirements.flashchord.txt
export UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300}
[ -x "$ENV/bin/python" ] || "$UV" venv --python 3.11 "$ENV" || exit 1
# requirements.flashchord.txt is `uv export --frozen --extra train --extra dev --no-emit-project` of flash_chord/uv.lock;
# regenerate it here and refuse to continue if the lock no longer produces the committed file
tmp=$(mktemp)
( cd "$FC" && "$UV" export --frozen --extra train --extra dev --no-emit-project --format requirements-txt ) > "$tmp" || exit 1
if ! diff -q <(grep -v '^#' "$tmp") <(grep -v '^#' "$REQ") > /dev/null; then
  echo "flash_chord/uv.lock does not reproduce $REQ - refusing to install a different package set" >&2; rm -f "$tmp"; exit 1
fi
rm -f "$tmp"
if [ -n "$STOCK" ] && ! cmp -s "$FC/uv.lock" "$STOCK/uv.lock"; then
  echo "$FC/uv.lock differs from the stock runtime's $STOCK/uv.lock" >&2; exit 1
fi
echo "locked packages: $(grep -c '==' "$REQ")"
extra=(); [ -n "${EXTRA_INDEX_URL:-}" ] && extra=(--extra-index-url "$EXTRA_INDEX_URL")
for i in 1 2 3; do
  "$UV" pip install --python "$ENV/bin/python" --index-url https://pypi.org/simple ${extra[@]+"${extra[@]}"} \
      --require-hashes -r "$REQ" && { echo DEPS_OK; break; }
  [ $i = 3 ] && { echo "dependency install failed" >&2; exit 1; }
  echo "attempt $i failed"; sleep 10
done
if [ -n "$STOCK" ]; then
  "$UV" pip install --python "$ENV/bin/python" --no-deps -e "$STOCK" || { echo "editable install of $STOCK failed" >&2; exit 1; }
  echo STOCK_EDITABLE_OK
fi
"$ENV/bin/python" - <<'PY' || { echo "import smoke test failed (see [import FAIL] above)" >&2; exit 1; }
import importlib, sys
failed = 0
for name in ["jax", "flax", "optax", "distrax", "warp", "newton", "mujoco", "mujoco_warp", "numpy", "onnxruntime", "wandb"]:
    try:
        module = importlib.import_module(name)
        print(f"[import OK] {name} {getattr(module, '__version__', '')}")
    except Exception as error:  # noqa: BLE001
        print(f"[import FAIL] {name}: {type(error).__name__}: {error}")
        failed += 1
if failed:
    sys.exit(1)
import jax
print("jax devices:", jax.devices())
PY
echo FC_ENV_DONE
