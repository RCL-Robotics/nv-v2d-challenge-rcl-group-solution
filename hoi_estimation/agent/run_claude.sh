#!/usr/bin/env bash
# Run Claude Code (headless, -p) as the agent on one exported episode, with the same isolation as run_codex.sh:
#   * filesystem: apptainer --containall; only the workspace (rw) and, read-only, the CARI4D code (/code), weights
#     (/weights), this episode's CARI4D intermediate outputs (/pipeline) and the claude binary are bound.
#   * network: --network none; the only route out is the logging CONNECT proxy (net=open: all but HuggingFace;
#     net=api: only api.anthropic.com).
# Auth: only the current OAuth ACCESS token of the host login is passed (env CLAUDE_CODE_OAUTH_TOKEN), never the
# refresh token or the credentials file, so nothing inside the box can rotate / invalidate the host login. The access
# token must outlive the run (checked below). Alternatively export CLAUDE_CODE_OAUTH_TOKEN (e.g. from
# `claude setup-token`) before calling this script.
# Token order: $CLAUDE_CODE_OAUTH_TOKEN, ~/.config/v2d/claude_oauth_token, the host login's access token.
# Usage: run_claude.sh <workspace> [model] [timeout_min] [net: open|api] [pipeline_dir] [task_file] [effort]
set -euo pipefail
WS=$1; MODEL=${2:-claude-opus-5-5}; TMIN=${3:-90}; NET=${4:-open}; PIPELINE=${5:-}; TASK_FILE=${6:-}; EFFORT=${7:-high}
A=$HOME/.local/apptainer/usr/bin/apptainer
V2D=${V2D_ROOT:-/mnt/nas/Group/hanzhou/v2d}
SIF=$V2D/images/v2d_cari4d.sif
CL=$(readlink -f $HOME/.local/bin/claude)
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p $WS/outputs $WS/scratch $WS/logs $WS/.net
cp ${TASK_FILE:-$HERE/AGENT_TASK.md} $WS/AGENT_TASK.md
cp $HERE/allowlist_proxy.py $WS/.net/allowlist_proxy.py
TOKF=$HOME/.config/v2d/claude_oauth_token  # long-lived token from `claude setup-token` (chmod 600), preferred
[ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] && [ -s $TOKF ] && CLAUDE_CODE_OAUTH_TOKEN=$(tr -d ' \n' < $TOKF)
if [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
  CLAUDE_CODE_OAUTH_TOKEN=$(python3 - "$TMIN" <<'PY'
import json, os, sys, time
d = json.load(open(os.path.expanduser("~/.claude/.credentials.json")))["claudeAiOauth"]
left = d["expiresAt"] / 1000 - time.time()
if left < (float(sys.argv[1]) + 10) * 60:
    sys.exit(f"access token expires in {left/60:.0f} min < run time; use `claude setup-token` and export CLAUDE_CODE_OAUTH_TOKEN")
print(d["accessToken"])
PY
)
fi
BINDS=(-B $WS:/work -B $CL:/opt/claude/claude:ro -B $HOME/Projects/v2d/nv-v2d-challenge-codebase:/code:ro -B $V2D/weights:/weights:ro)
[ -n "$PIPELINE" ] && BINDS+=(-B $PIPELINE:/pipeline:ro)
{ echo "date_utc: $(date -u +%FT%TZ)"; echo "claude: $($CL --version)"; echo "model: $MODEL effort=$EFFORT net=$NET"
  echo "sif: $SIF $(stat -c %s $SIF) bytes"; echo "binds: ${BINDS[*]}"; echo "task_sha256: $(sha256sum $WS/AGENT_TASK.md | cut -c1-64)"
  (cd $WS/inputs && find . -type f -print0 | sort -z | xargs -0 -r sha256sum); } > $WS/logs/run_manifest.txt
if [ "$NET" = open ]; then PARGS=(--deny-suffix huggingface.co hf.co); else PARGS=(--allow api.anthropic.com:443); fi
python3 $HERE/allowlist_proxy.py host --sock $WS/.net/proxy.sock --log $WS/logs/proxy.log "${PARGS[@]}" &
PROXY=$!
trap 'kill $PROXY 2>/dev/null || true' EXIT
sleep 1
export APPTAINERENV_CLAUDE_CODE_OAUTH_TOKEN=$CLAUDE_CODE_OAUTH_TOKEN APPTAINERENV_CUDA_VISIBLE_DEVICES=0
export APPTAINERENV_DISABLE_AUTOUPDATER=1 APPTAINERENV_CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 APPTAINERENV_DISABLE_TELEMETRY=1
export APPTAINERENV_CLAUDE_CODE_EFFORT_LEVEL=$EFFORT APPTAINERENV_BASH_MAX_TIMEOUT_MS=3600000
P=http://127.0.0.1:18080
export APPTAINERENV_HTTPS_PROXY=$P APPTAINERENV_https_proxy=$P APPTAINERENV_HTTP_PROXY=$P APPTAINERENV_http_proxy=$P
export APPTAINERENV_PYTHONUSERBASE=/work/scratch/pyuser APPTAINERENV_PIP_CACHE_DIR=/work/scratch/pip_cache
export APPTAINERENV_HF_HOME=/weights/hf_home APPTAINERENV_HF_HUB_OFFLINE=1 APPTAINERENV_TORCH_HOME=/weights/sam3d_body/torch_home
export APPTAINERENV_MODEL=$MODEL
START=$(date +%s)
timeout ${TMIN}m $A exec --nv --containall --net --network none --home /work/scratch --pwd /work "${BINDS[@]}" $SIF bash -c '
  python /work/.net/allowlist_proxy.py inner --sock /work/.net/proxy.sock & sleep 1
  /opt/claude/claude -p --model "$MODEL" --dangerously-skip-permissions --output-format stream-json --verbose \
    "Read /work/AGENT_TASK.md and complete the task described there. Work autonomously until the required outputs exist."' \
  </dev/null > $WS/logs/claude_events.jsonl 2> $WS/logs/claude_stderr.txt || echo "exit: $?" >> $WS/logs/run_manifest.txt
echo "wall_s: $(( $(date +%s) - START ))" >> $WS/logs/run_manifest.txt
