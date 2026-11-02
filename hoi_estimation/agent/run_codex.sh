#!/usr/bin/env bash
# Run Codex as an agent on one exported episode. Model and provider are taken from the user's ~/.codex/config.toml
# (model, model_catalog_json, [model_providers.<model_provider>]); the provider host is the only API host allowed.
# Isolation (kernel-enforced, outside the agent's control):
#   * filesystem: apptainer --containall; only the workspace (rw) and, read-only, the CARI4D code (/code), weights
#     (/weights) and this episode's CARI4D intermediate outputs (/pipeline) are bound. No NAS/home -> no ground truth.
#   * network: --net --network none; the only route out is a unix socket to a host-side CONNECT proxy that logs every
#     request (logs/proxy.log). net=api (default): only the model API is reachable. net=open: everything except
#     HuggingFace (huggingface.co, hf.co: the dataset with the references lives there; the weights are mounted).
# Codex's own bwrap sandbox cannot nest inside apptainer, so Codex runs with danger-full-access *inside* this box.
# Installs go to /work/scratch (HOME, PYTHONUSERBASE, pip cache) so they can be collected for offline reproduction.
# Usage: run_codex.sh <workspace> [effort] [timeout_min] [net: open|api] [pipeline_dir] [task_file]
set -euo pipefail
WS=$1; EFFORT=${2:-high}; TMIN=${3:-90}; NET=${4:-api}; PIPELINE=${5:-}; TASK_FILE=${6:-${TASK_FILE:-}}
A=$HOME/.local/apptainer/usr/bin/apptainer
V2D=/mnt/nas/Group/hanzhou/v2d
SIF=$V2D/images/v2d_cari4d.sif
PK=$(dirname "$(dirname "$(readlink -f $HOME/.local/bin/codex)")")
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p $WS/outputs $WS/scratch $WS/logs $WS/.codex_home $WS/.net
cp ${TASK_FILE:-$HERE/AGENT_TASK.md} $WS/AGENT_TASK.md
cp $HERE/allowlist_proxy.py $WS/.net/allowlist_proxy.py
read -r MODEL API_HOST < <(python3 - "$WS/.codex_home" "$EFFORT" <<'PY'
import sys, tomllib, json, shutil, os
from urllib.parse import urlparse
home, effort = sys.argv[1], sys.argv[2]
src = os.path.expanduser("~/.codex")
c = tomllib.load(open(f"{src}/config.toml", "rb"))
pv = c["model_providers"][c["model_provider"]]
q = lambda v: json.dumps(v)
lines = [f"model_provider = {q(c['model_provider'])}", f"model = {q(c['model'])}",
         f"model_reasoning_effort = {q(effort)}", "disable_response_storage = true"]
if c.get("model_catalog_json"):
    shutil.copy(os.path.join(src, c["model_catalog_json"]), f"{home}/model_catalog.json")
    lines.append('model_catalog_json = "model_catalog.json"')
lines.append(f"[model_providers.{c['model_provider']}]")
# the key is passed in the environment (no auth.json inside the box)
pv = {**pv, "requires_openai_auth": False, "env_key": "OPENAI_API_KEY"}
lines += [f"{k} = {q(v)}" for k, v in pv.items()]
open(f"{home}/config.toml", "w").write("\n".join(lines) + "\n")
u = urlparse(pv["base_url"])
print(c["model"], f"{u.hostname}:{u.port or 443}")
PY
)
BINDS=(-B $WS:/work -B $PK:/opt/codex:ro -B /home/hanzhou/Projects/v2d/nv-v2d-challenge-codebase:/code:ro -B $V2D/weights:/weights:ro)
[ -n "$PIPELINE" ] && BINDS+=(-B $PIPELINE:/pipeline:ro)
{ echo "date_utc: $(date -u +%FT%TZ)"; echo "codex: $($PK/bin/codex --version)"; echo "model: $MODEL api=$API_HOST effort=$EFFORT net=$NET"
  echo "sif: $SIF $(stat -c %s $SIF) bytes"; echo "binds: ${BINDS[*]}"; echo "task_sha256: $(sha256sum $WS/AGENT_TASK.md | cut -c1-64)"
  (cd $WS/inputs && find . -type f -print0 | sort -z | xargs -0 -r sha256sum); } > $WS/logs/run_manifest.txt
if [ "$NET" = open ]; then PARGS=(--deny-suffix huggingface.co hf.co); else PARGS=(--allow $API_HOST); fi
python3 $HERE/allowlist_proxy.py host --sock $WS/.net/proxy.sock --log $WS/logs/proxy.log "${PARGS[@]}" &
PROXY=$!
trap 'kill $PROXY 2>/dev/null || true' EXIT
sleep 1
export APPTAINERENV_OPENAI_API_KEY=$(python3 -c "import json;print(json.load(open('$HOME/.codex/auth.json'))['OPENAI_API_KEY'])")
export APPTAINERENV_CODEX_HOME=/work/.codex_home APPTAINERENV_CUDA_VISIBLE_DEVICES=0
P=http://127.0.0.1:18080
export APPTAINERENV_HTTPS_PROXY=$P APPTAINERENV_https_proxy=$P APPTAINERENV_HTTP_PROXY=$P APPTAINERENV_http_proxy=$P
export APPTAINERENV_PYTHONUSERBASE=/work/scratch/pyuser APPTAINERENV_PIP_CACHE_DIR=/work/scratch/pip_cache
export APPTAINERENV_HF_HOME=/weights/hf_home APPTAINERENV_HF_HUB_OFFLINE=1 APPTAINERENV_TORCH_HOME=/weights/sam3d_body/torch_home
START=$(date +%s)
timeout ${TMIN}m $A exec --nv --containall --net --network none --home /work/scratch --pwd /work "${BINDS[@]}" $SIF bash -c '
  python /work/.net/allowlist_proxy.py inner --sock /work/.net/proxy.sock & sleep 1
  /opt/codex/bin/codex exec --skip-git-repo-check --sandbox danger-full-access --json -C /work \
    "Read /work/AGENT_TASK.md and complete the task described there. Work autonomously until the required outputs exist."' \
  </dev/null > $WS/logs/codex_events.jsonl 2> $WS/logs/codex_stderr.txt || echo "exit: $?" >> $WS/logs/run_manifest.txt
echo "wall_s: $(( $(date +%s) - START ))" >> $WS/logs/run_manifest.txt
