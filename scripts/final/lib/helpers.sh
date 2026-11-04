# SPDX-License-Identifier: Apache-2.0
# helpers.sh: the small functions that the plans of scripts/final/*.sh call. Sourced by every plan (bash).
# Every function is idempotent where it can be, and none of them writes outside the paths it is given.

die() { echo "[final] ERROR: $*" >&2; exit 1; }
note() { echo "[final] $(date '+%F %T') $*"; }

check_tools() {
  [ -n "${PLAN_PY:-}" ] && [ -x "$PLAN_PY" ] || die "PLAN_PY ($PLAN_PY) is not an executable python"
}

check_eval_tools() {
  for v in FC_PYTHON KIT_PYTHON V2D_RUNTIME V2D_KIT V2D_DATA; do
    [ -n "${!v:-}" ] || die "$v is not set (scripts/final/site.env.example)"
  done
  [ -x "$FC_PYTHON" ] || die "FC_PYTHON $FC_PYTHON is not executable"
  [ -f "$V2D_KIT/examples/run_policy_evaluation.py" ] || die "V2D_KIT $V2D_KIT is not the submission kit"
  [ -f "$V2D_RUNTIME/src/flash_chord/__init__.py" ] || die "V2D_RUNTIME $V2D_RUNTIME is not the stock flash_chord runtime"
  local k s
  k=$(sha256sum "$V2D_KIT/flash_chord/scripts/evaluate_policy.py" | cut -d' ' -f1)
  s=$(sha256sum "$V2D_RUNTIME/scripts/evaluate_policy.py" | cut -d' ' -f1)
  [ "$k" = 160d24f5fbeb70ccd569ca4192056f1892b2a9f2999066db9d8279461648fdf1 ] \
    || die "kit evaluate_policy.py is $k, not the Oct-11 kit's 160d24f5..."
  [ "$s" = "$k" ] || die "stock runtime evaluate_policy.py ($s) differs from the kit's ($k)"
  note "kit and stock evaluator: 160d24f5... (as in every counted file)"
}

# check_sha256 FILE SHA256
check_sha256() {
  [ -f "$1" ] || die "missing file $1"
  local got; got=$(sha256sum "$1" | cut -d' ' -f1)
  [ "$got" = "$2" ] || die "$1: sha256 $got, expected $2"
  note "sha256 ok ${2:0:16}  $1"
}

# require_idle_gpu GPU: with REQUIRE_IDLE_GPU=1, refuse a GPU on which nvidia-smi lists any compute process
require_idle_gpu() {
  [ "${REQUIRE_IDLE_GPU:-0}" = 1 ] || return 0
  local uuid n
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$1") || die "nvidia-smi cannot query GPU $1"
  n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid" || true)
  [ "$n" = 0 ] || die "GPU $1 has $n compute process(es); REQUIRE_IDLE_GPU=1 refuses to start there"
  note "GPU $1: no compute process"
}

# ensure_worktree REPO DIR SHA: a detached git worktree of REPO at SHA in DIR (no LFS smudge). In a sparse clone
# (core.sparseCheckout=true) only robotic_grounding/flash_chord is checked out. An existing DIR must be at SHA.
ensure_worktree() {
  local repo=$1 dir=$2 sha=$3
  if [ -e "$dir/.git" ]; then
    [ "$(git -C "$dir" rev-parse HEAD)" = "$sha" ] || die "$dir exists but is not at $sha"
    note "worktree $dir at ${sha:0:7} (exists)"; return 0
  fi
  git -C "$repo" cat-file -e "$sha^{commit}" || die "commit $sha is not in $repo (fetch it first)"
  mkdir -p "$(dirname "$dir")"
  GIT_LFS_SKIP_SMUDGE=1 git -C "$repo" worktree add --detach --no-checkout "$dir" "$sha"
  if [ "$(git -C "$repo" config --bool core.sparseCheckout || true)" = true ]; then
    local sp; sp=$(cd "$dir" && realpath -m "$(git rev-parse --git-path info/sparse-checkout)")
    mkdir -p "$(dirname "$sp")" && printf '/robotic_grounding/flash_chord/\n' > "$sp"
  fi
  ( cd "$dir" && GIT_LFS_SKIP_SMUDGE=1 git read-tree -mu HEAD )
  [ -f "$dir/robotic_grounding/flash_chord/scripts/train_rl.py" ] || die "checkout of $sha in $dir failed"
  note "worktree $dir created at ${sha:0:7}"
}

# fill_lfs WORKTREE [SOURCE]: the FlashCHORD assets (meshes, SONIC ONNX prior; training needs them) as real files.
# SOURCE given: copy each file from SOURCE (a checkout that has them) only if its size and sha256 equal the pointer's.
# Otherwise: git lfs pull from the repository's lfs.url (upstream nvidia-isaac/video_to_data; see REPRODUCE.md).
fill_lfs() {
  local wt=$1 src=${2:-}
  if [ -n "$src" ]; then
    "$PLAN_PY" -I -B "$FINAL_LIB/lfs_fill.py" --worktree "$wt" --source "$src" --report "$wt.lfs_fill.tsv"
  else
    git -C "$wt" lfs pull --include "robotic_grounding/flash_chord/src/flash_chord/assets/**"
    "$PLAN_PY" -I -B "$FINAL_LIB/lfs_fill.py" --worktree "$wt" --check-only --report "$wt.lfs_fill.tsv"
  fi
}

# seed_hulls SRC_CONVEX_DIR CACHE_DIR: CACHE_DIR/convex gets the recorded convex-decomposition cache (the trainer
# reads FLASH_CHORD_CACHE_DIR/convex). Empty or missing SRC -> the stock runtime's cache. Existing files are kept.
seed_hulls() {
  local src=$1 dst=$2
  if [ -z "$src" ] || [ ! -d "$src" ]; then
    [ -n "$src" ] && note "recorded hull cache $src not found here; seeding from the stock runtime instead"
    src="$V2D_RUNTIME/src/flash_chord/assets/.cache/convex"
  fi
  mkdir -p "$dst/convex"
  if [ -d "$src" ]; then find "$src" -maxdepth 1 -name '*.npz' -exec cp -p -n {} "$dst/convex/" \; ; fi
  note "hull cache $dst/convex: $(find "$dst/convex" -maxdepth 1 -name '*.npz' | wc -l) files (from $src)"
}

# resume_pair SRC_DIR DST_DIR N MODEL_SHA RSI_SHA DIGEST_FROM CHAIN: model_N + rsi_N in DST_DIR for a resumed stage.
#   CHAIN=1 (in-place restart or teammate chain: DST_DIR is the previous stage's run folder): when this replay's previous
#   stage left DST_DIR/handoff_N.sha256 (record_handoff), the stage resumes from exactly those files (checked against
#   it); the recorded files are not used. Otherwise model_N + rsi_N are copied from SRC_DIR (the recorded run folder)
#   and checked against MODEL_SHA / RSI_SHA; DIGEST_FROM names where the digests come from. An existing destination
#   file without a handoff is kept and checked the same way.
resume_pair() {
  local src=$1 dst=$2 n=$3 ms=${4:-} rs=${5:-} from=${6:-the set record} chain=${7:-0} f sha
  mkdir -p "$dst"
  if [ "$chain" = 1 ] && [ -f "$dst/handoff_$n.sha256" ]; then
    ( cd "$dst" && sha256sum -c --quiet "handoff_$n.sha256" ) \
      || die "$dst: model_$n / rsi_$n are not the files the previous stage handed over (handoff_$n.sha256)"
    note "resume from this replay's own model_$n + rsi_$n in $dst, as the previous stage wrote them (sha256 $(cut -c1-16 "$dst/handoff_$n.sha256" | tr '\n' ' '))"
    return 0
  fi
  [ "$chain" = 1 ] && note "no handoff_$n.sha256 in $dst (the previous stage did not run in this replay, or ended before $n): resume from the recorded pair in $src"
  for f in model rsi; do
    sha=$ms; [ $f = rsi ] && sha=$rs
    if [ ! -f "$dst/${f}_$n.safetensors" ]; then
      [ -f "$src/${f}_$n.safetensors" ] || die "resume file $src/${f}_$n.safetensors not found (payload / ROOT_* / T1_RUNS / T1_ZX_RUNS paths)"
      cp -p "$src/${f}_$n.safetensors" "$dst/${f}_$n.safetensors.tmp" && mv "$dst/${f}_$n.safetensors.tmp" "$dst/${f}_$n.safetensors"
    fi
    if [ -n "$sha" ]; then check_sha256 "$dst/${f}_$n.safetensors" "$sha"; note "  digest from $from"
    else note "resume ${f}_$n sha256 $(sha256sum "$dst/${f}_$n.safetensors" | cut -c1-16)... (no digest in the record or in resume_sha256.tsv)"; fi
  done
}

# copy_resume_pair SRC_DIR DST_DIR N [MODEL_SHA] [RSI_SHA]: resume_pair without a chain (Tier 2, Tier 3)
copy_resume_pair() { resume_pair "$1" "$2" "$3" "${4:-}" "${5:-}" "the set record" 0; }

# record_handoff RUN_DIR N: after a stage whose run folder the next stage resumes from (in-place restart, teammate
# chain), write RUN_DIR/handoff_N.sha256 (model_N + rsi_N), so the next stage resumes from exactly these files. A run
# that has no model_N (the trainer's early stop came first) gets no handoff; the next stage then copies the recorded pair.
record_handoff() {
  local d=$1 n=$2
  if [ -f "$d/model_$n.safetensors" ] && [ -f "$d/rsi_$n.safetensors" ]; then
    ( cd "$d" && sha256sum "model_$n.safetensors" "rsi_$n.safetensors" > "handoff_$n.sha256.tmp" && mv "handoff_$n.sha256.tmp" "handoff_$n.sha256" )
    note "handoff: the next stage resumes from $d/model_$n + rsi_$n (sha256 $(cut -c1-16 "$d/handoff_$n.sha256" | tr '\n' ' '))"
  else
    rm -f "$d/handoff_$n.sha256"
    note "this run has no model_$n + rsi_$n (final.json: $(tr -d '\n ' < "$d/final.json" 2>&1 | cut -c1-80)); the next stage will copy the recorded pair"
  fi
}

# fc_libs: the flashchord env's pip CUDA 13 / cuDNN / NCCL / NVSHMEM library dirs (onnxruntime-gpu needs them)
fc_libs() {
  local site out="" d
  site=$("$FC_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  for d in cu13 cudnn nccl nvshmem; do
    [ -d "$site/nvidia/$d/lib" ] && out="${out:+$out:}$site/nvidia/$d/lib"
  done
  printf '%s%s' "$out" "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
}

# show_final RUN_DIR: the trainer's final.json (written at max_iterations or early stop)
show_final() {
  if [ -f "$1/final.json" ]; then note "final.json: $(tr -d '\n ' < "$1/final.json" | cut -c1-300)"
  else die "no final.json in $1 (see $1/train.log)"; fi
}

# check_sums DIR SUMFILE...: every file listed in SUMFILE (paths relative to DIR) that is present must match its
# sha256; listed files that are absent (record copies kept out of git by design: checkpoints, run folders) are counted
# and reported, never silently passed as checked. Exit on any mismatch.
check_sums() {
  local dir=$1 f n_list n_miss; shift
  for f in "$@"; do
    [ -f "$dir/$f" ] || die "missing hash file $dir/$f"
    n_list=$(grep -v -c -e '^#' -e '^$' "$dir/$f" || true)
    n_miss=$(cd "$dir" && grep -v -e '^#' -e '^$' "$f" | cut -c67- | while IFS= read -r p; do [ -e "$p" ] || echo x; done | wc -l)
    if [ "$n_miss" -lt "$n_list" ]; then
      ( cd "$dir" && sha256sum -c --ignore-missing --quiet "$f" ) || die "$dir/$f: a listed file differs"
    fi
    note "$f: $((n_list - n_miss)) of $n_list listed files present here, all match ($n_miss not in this checkout)"
  done
}
