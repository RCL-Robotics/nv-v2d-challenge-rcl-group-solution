#!/usr/bin/env python3
"""k3easy_helper.py: checks, slot / CPU plan, state, post-training step and verdict of scripts/k3easy_driver.sh, the
easy-clip K3 trainability test (arm C3r2_tier2_k3easy; pre-registration tmp_analysis/tier2/k3easy/PREREG.md).

  C3r2 exactly as the raw-input arm C3r2_tier2 (code/fc_r2 @61fa5d3, experiment g1_recon_body_coord_rsi_r2, seed 42,
  max_iterations 45000, experiment-default RSI deadline 35156 / zero_ramp 3906 / help-zero early stop 5860, scene.support
  as the raw arm) on data/human_motion_data_tier2_k3 (noisy Tier-2 body + K3 object) for ep20 rice_cooker and ep23
  toy_stroller (GPU 3) and ep08 skinny_wood_chair (GPU 6). The only recipe difference is the data root.

Run with conda_envs/flashchord/bin/python -I -B. Stdlib only, plus pyarrow (input check), hydra/omegaconf (cfgcheck)
and wandb's offline reader (verdict: raw-arm training curves). It never starts a training and never signals a
process except its own evaluation session on timeout (as scripts/track2_tier2v2_post.py). It writes only under
tmp_analysis/tier2/k3easy/ (eval work dirs, post/verdict files, CRASHED markers). nvidia-smi / /proc / CPU-slice
primitives come from scripts/track2_queue_tier2v2_helper.py (imported read-only; nothing of that queue is touched).

Subcommands:
  check                    preflight (PREREG sha256, code pin + tree digest, experiment values, the 3 K3 parquets:
                           sha256 vs arm_lock.json / expected, K3fix check, support rule, raw references, cfgcheck)
  cfgcheck [--no-write]    hydra-compose the raw arm's overrides and the K3 command per episode (CPU, no training code
                           imported), check the raw composition reproduces the raw run's saved config.yaml, and diff
                           K3 vs raw; writes k3easy/cfgcheck.json
  cmd --ep NN              the training argv, NUL-separated (driver: mapfile -d '')
  plan --ep NN --gpu G [--exclude CPUS] [--remaining N]   "OK <cpus> <nthreads> <note|-> | why" or "WAIT why"
                           (N = this test's launches still to come incl. this one; default: the FRESH episodes)
  gpus                     per-GPU view (trainings of anybody, evaluations, free memory, MPS)
  state                    one line per episode "EP NN STATE detail" (RUNNING STARTING FRESH DEAD CRASHED TRAINED
                           POST_RUNNING POST_RETRY_WAIT POST_DONE)
  trainer --ep NN          pid(s) of the live trainer(s) of that run dir
  crash --ep NN --why TEXT write k3easy/CRASHED_epNN once (no relaunch is ever made)
  post --ep NN --gpu G     after-training step: export at 2e-4 + ONE official evaluation of final.json's checkpoint
                           against its own input (eval_all.py --checkpoint-from-final, arguments as
                           track2_tier2v2_post.py) on the training GPU under the eval-lane locks, then the local clean
                           scoring (scripts/track2_tier2_clean_scores.py, --frame-references kalman/eval/frame_refs_k3)
                           into *_LOCAL_NOT_FOR_UPLOAD files
  verdict [--no-write]     apply the PREREG rule; exit 2 (nothing written) when PREREG.sha256 does not match PREREG.md
  heartbeat                one status line
"""

from __future__ import annotations

import csv
import difflib
import fcntl
import glob
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ME = Path("/root/autodl-tmp/haoran.zhang")
sys.path.insert(0, str(ME / "scripts"))
import track2_queue_tier2v2_helper as H  # noqa: E402  read-only primitives (nvidia-smi, /proc, cpu slice, parquet)

ARM = "C3r2_tier2_k3easy"
EXP = "g1_recon_body_coord_rsi_r2"
ITERS = 45000
SEED = 42
K = ME / "tmp_analysis/tier2/k3easy"
RUNS = ME / "fc_runs" / ARM
EVROOT = K / "eval"
CODE = ME / "code/fc_r2"
PIN = "61fa5d31ff8054bf1e5cf5375ef0957e54d2729b"
DATA = ME / "data/human_motion_data_tier2_k3"
RAW_DATA = ME / "data/human_motion_data_tier2"
RAW_RUNS = ME / "fc_runs/C3r2_tier2"
RAW_EVAL_CSV = ME / "fc_runs/eval_tier2/C3r2_tier2_final.csv"
BASELINE_CSV = ME / "tmp_analysis/tier2/early_scores/early_table.csv"
FRAME_REFS = ME / "tmp_analysis/tier2/kalman/eval/frame_refs_k3"
ARM_LOCK = ME / "fc_runs/queue_tier2v2/arm_lock.json"
K3FIX_CHECK = K / "k3fix_check.json"
CFGCHECK = K / "cfgcheck.json"
PREREG = K / "PREREG.md"
PREREG_SHA = K / "PREREG.sha256"
PIDS = K / "pids.tsv"
KIT = ME / "v2d_submission_kit_src/v2d_submission_kit"
STOCK = ME / "code/video_to_data_8c778f9/robotic_grounding/flash_chord"
DATASET = ME / "data/video_to_data_challenge"
KPY = ME / "conda_envs/v2dkit/bin/python"
FCPY = ME / "conda_envs/flashchord/bin/python"
CLEAN_SCORER = ME / "scripts/track2_tier2_clean_scores.py"
TIER1_CSV = ME / "fc_runs/eval_round2/C3r2_wide_final.csv"
RUNIN_EVALLOCK = ME / "fc_runs/queue_runin/evallock"  # opened read-only, never created
OWN_EVALLOCK = K / "evallock"
COMMIT = "https://github.com/local-placeholder/not-uploaded/commit/0000000000000000000000000000000000000000"
EXPORT_TOL = 2e-4
EVAL_TIMEOUT = 2400.0
SCORE_TIMEOUT = 1800.0
RETRY_DELAY = 600.0
MAX_TRIES = 2  # technical evaluation failures: one retry
MAX_POST_STARTS = 5
MIN_FINAL_AGE = 120.0
STARTING_GRACE_S = 180.0
# slot rule (as track2_queue_tier2v2.sh settings): < 3 trainings of anybody after the start, 16.5 GB + 20 GB free
MAX_TRAIN_PER_GPU = 3
JOB_MB = 16896
MIN_FREE_AFTER_MB = 20480
MAX_CORES = 4
MIN_CORES = 2
EVAL_MIN_FREE_MB = 4096
UNKNOWN_TRAIN_AGE_S = 1800
RAMP_S = 900
SPI = 4096 * 24  # environment steps per iteration (metrics "step" -> iteration)
HZ_MARGIN = 5000  # PREREG rule (a): K3 help-zero iteration <= raw + 5000
TERMINAL = ("done", "done_score_failed", "export_refused", "failed")
SH = "/root/autodl-tmp/haoran.zhang/code/fc_r2/logs/rl/sharpa"
EPS = {  # episode -> fixed facts (sha256 = fc_runs/queue_tier2v2/arm_lock.json parquet_sha256; raw = C3r2_tier2 final.json)
    20: {"seq": "2026-06-08_16-57-34_rice_cooker_clockwise_04", "obj": "rice_cooker", "support": True, "gpu": 3,
         "sha": "2784cae6dc09774f0ed7536e00c884276bbe0c49f2d0aa0e14c2feea6b964558", "raw_hz": 13174, "raw_final": 19034,
         "raw_hydra": f"{SH}/2026-10-15_22-37-38_ep20_rice_cooker/.hydra"},
    23: {"seq": "2026-06-03_15-08-03_toy_stroller_back_forth_ground_04", "obj": "toy_stroller", "support": False, "gpu": 3,
         "sha": "3f6aaa9e0378028395c9fcf835b9c306e38950fa4e99acfcf1bbc5284af34e84", "raw_hz": 14163, "raw_final": 20023,
         "raw_hydra": f"{SH}/2026-10-17_05-26-25_ep23_toy_stroller/.hydra"},
    8: {"seq": "2026-06-05_15-08-08_skinny_wood_chair_push_10", "obj": "skinny_wood_chair", "support": False, "gpu": 6,
        "sha": "510bdc3d29bd9c4425e90b3a1ff479afb4f0a815dde1c816be6822bfb69ae592", "raw_hz": 14737, "raw_final": 20597,
        "raw_hydra": f"{SH}/2026-10-16_23-41-45_ep08_skinny_wood_chair/.hydra"},
}
ORDER = (20, 23, 8)
CFG_ALLOWED_DIFF = {"task.parquet", "output_dir", "logging.mode", "logging.experiment", "logging.local_jsonl"}


# ============================================================================================== small helpers
def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256(path) -> str | None:
    return H.sha256(Path(path))


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def fnum(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def ep_of(text: str) -> int:
    e = int(text)
    if e not in EPS:
        raise SystemExit(f"episode {e} is not part of this test (08 20 23)")
    return e


def parquet(ep: int, root: Path = DATA) -> Path:
    return root / "whole_body/soma" / f"sequence_id={EPS[ep]['seq']}" / "robot_name=g1/data.parquet"


def run_dir(ep: int) -> Path:
    return RUNS / f"ep{ep:02d}"


def prereg_status() -> tuple[bool, str]:
    if not PREREG.is_file() or not PREREG_SHA.is_file():
        return False, f"{PREREG.name} or {PREREG_SHA.name} missing"
    words = PREREG_SHA.read_text().split()
    want = words[0] if words else ""
    got = sha256(PREREG)
    return got == want, f"sha256(PREREG.md) {got[:16]}... {'==' if got == want else '!='} PREREG.sha256 {want[:16]}..."


def code_digest() -> str:
    """sha256 over (relative path, sha256) of every .py/.yaml under code/fc_r2/src/flash_chord and scripts/ (no caches)."""
    h = hashlib.sha256()
    files = []
    for base in (CODE / "src/flash_chord", CODE / "scripts"):
        for p in base.rglob("*"):
            if p.is_file() and p.suffix in (".py", ".yaml") and "__pycache__" not in p.parts and ".cache" not in p.parts:
                files.append(p)
    for p in sorted(files):
        h.update(f"{p.relative_to(CODE)}\t{sha256(p)}\n".encode())
    return h.hexdigest()


def train_argv(ep: int) -> list[str]:
    """The raw arm's command (scripts/track2_queue_tier2.sh train_cmd) with the K3 root, this arm's output_dir and
    logging, plus hydra.run.dir / +logging.local_jsonl (as track2_queue_tier2v2.sh); neither changes training."""
    e, out = EPS[ep], run_dir(ep)
    argv = ["python", "scripts/train_rl.py", f"experiment={EXP}", f"task.parquet='{parquet(ep)}'", f"reset.seed={SEED}",
            f"output_dir={out}", f"training.max_iterations={ITERS}", "logging.mode=offline", "logging.entity=juicer_test",
            "logging.project=v2d-track2-tier2", f"logging.experiment={ARM}", f"logging.run_name=ep{ep:02d}_{e['obj']}",
            "logging.upload_checkpoints=false"]
    if not e["support"]:
        argv.append("scene.support=false")
    argv += [f"hydra.run.dir={out}/hydra", "+logging.local_jsonl=true"]
    return argv


def read_pids() -> dict[int, dict]:
    out: dict[int, dict] = {}
    try:
        for line in PIDS.read_text().splitlines():
            f = line.split("\t")
            if len(f) >= 6 and f[0].isdigit():
                out[int(f[0])] = {"pid": int(f[1]) if f[1].isdigit() else None, "gpu": f[2], "cpus": f[3], "nth": f[4],
                                  "t": int(f[5]) if f[5].isdigit() else 0, "mode": f[6] if len(f) > 6 else ""}
    except OSError:
        pass
    return out


def find_trainers(ep: int) -> list[int]:
    want = f"output_dir={run_dir(ep)}"
    found = []
    for pid in H.all_pids():
        cmd = H.read_cmdline(pid)
        if H.is_python_trainer(cmd) and want in cmd and H.alive(pid):
            found.append(pid)
    return found


def jsonl_last(path: Path, nbytes: int = 1 << 16) -> dict | None:
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            lines = f.read().decode(errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            return json.loads(line)
        except ValueError:
            continue
    return None


def progress(ep: int) -> dict:
    run = run_dir(ep)
    its = [int(m.group(1)) for p in run.glob("model_*.safetensors") if (m := re.fullmatch(r"model_(\d+)\.safetensors", p.name))]
    last = jsonl_last(run / "metrics.jsonl") or {}
    it = None
    if isinstance(last.get("step"), (int, float)):
        it = int(round(last["step"] / SPI))
    return {"models": len(its), "last_model": max(its) if its else None, "it": it,
            "help_zero_iters": last.get("early_stop/help_zero_iterations"), "fail": last.get("rsi/fail_rate_mean")}


def lock_held(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    except OSError:
        return True
    finally:
        os.close(fd)


def try_lock(path: Path, create: bool) -> int | None:
    """create=True: our own lock file under k3easy (O_RDWR|O_CREAT). create=False: someone else's lock, opened
    read-only and never created (flock works on a read-only descriptor)."""
    try:
        fd = os.open(path, (os.O_RDWR | os.O_CREAT) if create else os.O_RDONLY, 0o644)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def post_file(ep: int) -> Path:
    return EVROOT / f"ep{ep:02d}" / "post.json"


def load_post(ep: int) -> dict:
    try:
        return json.loads(post_file(ep).read_text())
    except (OSError, ValueError):
        return {}


def save_post(ep: int, st: dict) -> None:
    st["updated_at"] = now()
    atomic_write(post_file(ep), json.dumps(st, indent=1, default=str))


def plog(ep: int, message: str) -> None:
    line = f"[{now()}] ep{ep:02d}: {message}"
    print(line, flush=True)
    try:
        with (K / "post.log").open("a") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ============================================================================================== GPU view / plan
def observe() -> dict[int, dict]:
    """Per GPU: trainings of anybody (as track2_queue_tier2v2_helper.observe: classified compute apps, unknown apps
    older than 30 min, python train_rl.py not on nvidia-smi yet whose environment names the GPU), evaluations, free
    memory, MPS presence, and the ramp debt of our own trainers younger than RAMP_S."""
    st = H.smi()
    pids = H.all_pids()
    uuid_to_idx = {v["uuid"]: k for k, v in st["gpus"].items()}
    mps = H.mps_gpus(st, pids)
    view = {g: {"free": v["free"], "total": v["total"], "trainings": [], "evals": [], "others": [], "mps": mps.get(g),
                "debt": 0} for g, v in st["gpus"].items()}
    mem = {a["pid"]: a["mem"] for a in st["apps"]}
    counted: dict[int, int] = {}
    for app in st["apps"]:
        pid, g = app["pid"], app["gpu"]
        cls = H.classify(H.read_cmdline(pid), H.read_environ(pid))
        age = H.proc_age(pid) or 0.0
        if cls == "train" or (cls == "unknown" and age >= UNKNOWN_TRAIN_AGE_S):
            counted[pid] = g
        elif cls == "eval":
            view[g]["evals"].append(pid)
        elif cls == "unknown":
            view[g]["others"].append(pid)
    for pid in pids:
        if pid in counted:
            continue
        cmd = H.read_cmdline(pid)
        if not H.is_python_trainer(cmd) or not H.alive(pid):
            continue
        g = H.env_gpu(H.read_environ(pid), uuid_to_idx)
        if g is not None and g in view:
            counted[pid] = g
    for pid, g in counted.items():
        view[g]["trainings"].append(pid)
        if H.read_environ(pid).get("K3EASY") == "1" and (H.proc_age(pid) or 0.0) < RAMP_S:
            view[g]["debt"] += max(0, JOB_MB - mem.get(pid, 0))
    return view


def plan(ep: int, gpu: int, exclude: str, remaining: int | None = None) -> str:
    if gpu == 4:
        return "WAIT gpu4 is never used"
    if gpu != EPS[ep]["gpu"]:
        return f"WAIT ep{ep:02d} is assigned to gpu{EPS[ep]['gpu']}, not gpu{gpu}"
    try:
        view = observe()
    except Exception as error:  # noqa: BLE001
        return f"WAIT nvidia-smi failed ({type(error).__name__}: {str(error)[:100]})"
    v = view.get(gpu)
    if v is None:
        return f"WAIT gpu{gpu} is not listed by nvidia-smi"
    if v["mps"]:
        return f"WAIT gpu{gpu} has MPS ({v['mps']}): no start"
    n = len(v["trainings"])
    if n + 1 > MAX_TRAIN_PER_GPU:
        return f"WAIT gpu{gpu} runs {n} training(s) (pids {' '.join(map(str, sorted(v['trainings'])))}): a start would exceed {MAX_TRAIN_PER_GPU}"
    need = JOB_MB + MIN_FREE_AFTER_MB + v["debt"]
    if v["free"] < need:
        debt = f" + ramp debt {v['debt']}" if v["debt"] else ""
        return f"WAIT gpu{gpu} {v['free']} MiB free < {need} (job {JOB_MB} + {MIN_FREE_AFTER_MB} kept free{debt})"
    # CPU slice (track2_queue_tier2v2_helper.cpu_slice: free physical cores of the GPU's NUMA node training pool that no
    # other process holds as a slice, + HT siblings; fewer than MIN_CORES free borrows the other node's). The free
    # own-node cores are shared among this test's episodes still to launch (ceil(free / remaining), 2..4 cores each),
    # so that the last one is not pushed onto the other NUMA node.
    if remaining is None:
        remaining = sum(1 for e in ORDER if ep_state(e)[0] == "FRESH") or 1
    allc, _, note = H.cpu_slice(gpu, exclude, 104, MIN_CORES)
    if not allc or allc == "none":
        return f"WAIT no free CPU core for a slice on gpu{gpu}'s NUMA node"
    phys = [int(x) for x in allc.split(",")[0::2]]
    if note == "cross-NUMA":
        cores = phys[:MIN_CORES]
    else:
        cores = phys[:min(MAX_CORES, max(MIN_CORES, math.ceil(len(phys) / max(1, remaining))))]
    cpus, nth = ",".join(f"{c},{c + 104}" for c in cores), 2 * len(cores)
    return (f"OK {cpus} {nth} {note or '-'} | gpu{gpu}: {n} training(s) now ({' '.join(map(str, sorted(v['trainings']))) or 'none'})"
            f" -> {n + 1} after the start (cap {MAX_TRAIN_PER_GPU}); {v['free']} MiB free >= {need}; no MPS; CPU: {len(cores)} of "
            f"{len(phys)} free {'own-node' if note != 'cross-NUMA' else 'cores (own node short: borrowed)'} core(s) for {remaining} launch(es) to come")


def gpus_report() -> list[str]:
    try:
        view = observe()
    except Exception as error:  # noqa: BLE001
        return [f"GPU_ERR nvidia-smi failed: {error}"]
    out = []
    for g in sorted(view):
        v = view[g]
        out.append(f"GPU {g} trainings={len(v['trainings'])} ({' '.join(map(str, sorted(v['trainings'])))}) evals={len(v['evals'])}"
                   f" others={len(v['others'])} free={v['free']}MiB mps={v['mps'] or 'no'} debt={v['debt']}"
                   f"{'  <- this test' if g in {e['gpu'] for e in EPS.values()} else ''}{'  (never used: teammate)' if g == 4 else ''}")
    return out


# ============================================================================================== check / cfgcheck
def check() -> int:
    rows: list[tuple[str, bool, str]] = []

    def add(name, ok, detail):
        rows.append((name, bool(ok), detail))

    ok, why = prereg_status()
    add("prereg", ok, why)
    try:
        head = subprocess.run(["git", "-C", str(CODE), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        head = ""
    add("code_pin", head == PIN, f"{CODE} HEAD {head[:12] or '?'} (pin {PIN[:7]})")
    build = {}
    try:
        build = json.loads((K / "build.json").read_text())
    except (OSError, ValueError):
        pass
    dig = code_digest()
    add("code_tree_digest", build.get("code_digest") == dig,
        f"sha256 over src/flash_chord + scripts *.py/*.yaml {dig[:16]}... {'==' if build.get('code_digest') == dig else '!='} build.json")
    expf = CODE / "src/flash_chord/configs/experiment" / f"{EXP}.yaml"
    text = expf.read_text() if expf.is_file() else ""
    for key, val in (("deadline", 35156), ("zero_ramp", 3906), ("early_stop_help_zero_tail", 5860), ("max_iterations", 45000)):
        add(f"experiment_{key}", re.search(rf"^\s+{key}: {val}\s*$", text, re.M), f"{EXP}.yaml {key}: {val}")
    add("data_ready", (DATA / "READY").is_file(), f"{DATA}/READY")
    try:
        man = json.loads((DATA / "whole_body/soma/track2_manifest.json").read_text())
        seq_of = {int(r["episode_index"]): str(r.get("sequence_id", s)) for s, r in man.items()}
    except (OSError, ValueError, KeyError):
        seq_of = {}
    try:
        lock = json.loads(ARM_LOCK.read_text()).get("parquet_sha256", {})
    except (OSError, ValueError):
        lock = {}
    try:
        kfc = json.loads(K3FIX_CHECK.read_text())
    except (OSError, ValueError):
        kfc = {}
    add("k3fix_check", kfc.get("all_ok") is True and kfc.get("k3fix_sha256") == kfc.get("k3fix_sha256_expected"),
        f"{K3FIX_CHECK.name}: all_ok {kfc.get('all_ok')} (K3fix = k3fix_v1_precommit.py {str(kfc.get('k3fix_sha256'))[:8]})")
    try:
        cfc = json.loads(CFGCHECK.read_text())
    except (OSError, ValueError):
        cfc = {}
    add("cfgcheck", cfc.get("all_ok") is True, f"{CFGCHECK.name}: all_ok {cfc.get('all_ok')}")
    for ep in ORDER:
        e, tag = EPS[ep], f"ep{ep:02d}"
        add(f"{tag}_gpu", e["gpu"] != 4, f"assigned gpu{e['gpu']}")
        add(f"{tag}_manifest", seq_of.get(ep) == e["seq"], f"manifest ep{ep} -> {seq_of.get(ep)}")
        p = parquet(ep)
        bad = H.check_parquet(p, e["seq"]) if p.is_file() else "missing"
        add(f"{tag}_parquet_valid", bad is None, f"{p}: {bad or '1 row, columns, sequence id, asset files ok'}")
        digest = sha256(p)
        add(f"{tag}_parquet_sha256", digest == e["sha"] == lock.get(f"{ep:02d}"),
            f"{str(digest)[:16]}... expected {e['sha'][:16]}... arm_lock {str(lock.get(f'{ep:02d}'))[:16]}...")
        add(f"{tag}_not_raw", digest != sha256(parquet(ep, RAW_DATA)), "K3 parquet differs from the raw Tier-2 parquet")
        add(f"{tag}_k3fix_unchanged", (kfc.get("episodes", {}).get(f"{ep:02d}", {}).get("ok") is True
                                       and kfc["episodes"][f"{ep:02d}"].get("sha256_k3_root") == digest),
            "K3fix leaves this episode unchanged (k3fix_check.json) and the hash it checked is the current one")
        sdir, rdir = p.parent.parent, parquet(ep, RAW_DATA).parent.parent
        add(f"{tag}_ground_plane", sha256(sdir / "ground_plane.json") == sha256(rdir / "ground_plane.json"), "ground_plane.json = raw Tier-2")
        for name in (f"{e['seq']}_support.usda", f"{e['seq']}_g1_support.usda"):
            mine, ref = DATA / "whole_body/reconstructed_stage" / name, RAW_DATA / "whole_body/reconstructed_stage" / name
            add(f"{tag}_{'g1_' if '_g1_' in name else ''}support_file", mine.is_file() == ref.is_file() and (not mine.is_file() or sha256(mine) == sha256(ref)),
                f"reconstructed_stage/{name}: {'present' if mine.is_file() else 'absent'} in both roots, identical")
        rule = H.support_rule(DATA, e["seq"])
        add(f"{tag}_support_rule", rule == ("true" if e["support"] else "false") == H.support_rule(RAW_DATA, e["seq"]),
            f"support rule {rule} (scene.support={'default true' if e['support'] else 'false'}) as the raw arm")
        try:
            raw_ov = [str(x) for x in json.loads(json.dumps(_load_yaml_list(Path(e["raw_hydra"]) / "overrides.yaml")))]
        except Exception:  # noqa: BLE001
            raw_ov = []
        add(f"{tag}_raw_support_flag", ("scene.support=false" in raw_ov) == (not e["support"]),
            f"raw overrides {'contain' if 'scene.support=false' in raw_ov else 'lack'} scene.support=false")
        fin = H.final_marker(RAW_RUNS / f"ep{ep:02d}") or {}
        add(f"{tag}_raw_reference", fin.get("help_zero_since") == e["raw_hz"] and fin.get("iteration") == e["raw_final"],
            f"raw final.json help_zero_since {fin.get('help_zero_since')} iteration {fin.get('iteration')} (PREREG {e['raw_hz']} / {e['raw_final']})")
        add(f"{tag}_frame_refs", (FRAME_REFS / f"episode_{ep:06d}.npz").is_file(), f"{FRAME_REFS.name}/episode_{ep:06d}.npz")
    bad = [r for r in rows if not r[1]]
    for name, ok, detail in rows:
        print(f"CHECK {name} {'ok' if ok else 'FAIL'} {detail}")
    print(f"CHECKS {len(rows) - len(bad)}/{len(rows)} ok" + (f"; FAILED: {' '.join(r[0] for r in bad)}" if bad else ""))
    return 0 if not bad else 1


def _load_yaml_list(path: Path) -> list:
    from omegaconf import OmegaConf
    return list(OmegaConf.to_container(OmegaConf.load(path)))


def _flat(x, prefix: str = "") -> dict:
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(_flat(v, f"{prefix}{k}."))
    elif isinstance(x, list):
        out[prefix[:-1]] = json.dumps(x, sort_keys=True)
    else:
        out[prefix[:-1]] = x
    return out


def cfgcheck(write: bool) -> int:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    res = {"written": now(), "code": str(CODE), "pin": PIN, "allowed_differences": sorted(CFG_ALLOWED_DIFF), "episodes": {}}
    ok_all = True
    with initialize_config_dir(config_dir=str(CODE / "src/flash_chord/configs"), version_base="1.3"):
        for ep in ORDER:
            e = EPS[ep]
            hyd = Path(e["raw_hydra"])
            raw_ov = [str(x) for x in _load_yaml_list(hyd / "overrides.yaml")]
            k3_ov = [a for a in train_argv(ep)[2:] if not a.startswith("hydra.")]
            raw_saved = _flat(OmegaConf.to_container(OmegaConf.load(hyd / "config.yaml"), resolve=False))
            raw_comp = _flat(OmegaConf.to_container(compose(config_name="train", overrides=raw_ov), resolve=False))
            k3_comp = _flat(OmegaConf.to_container(compose(config_name="train", overrides=k3_ov), resolve=False))
            repro = raw_comp == raw_saved
            keys = sorted(set(raw_saved) | set(k3_comp))
            diff = [{"key": k, "raw": raw_saved.get(k, "<absent>"), "k3": k3_comp.get(k, "<absent>")} for k in keys
                    if raw_saved.get(k, "<absent>") != k3_comp.get(k, "<absent>")]
            dkeys = {d["key"] for d in diff}
            ok = (repro and dkeys <= CFG_ALLOWED_DIFF and k3_comp.get("task.parquet") == str(parquet(ep))
                  and k3_comp.get("output_dir") == str(run_dir(ep)) and k3_comp.get("logging.mode") == "offline")
            ok_all &= ok
            udiff = list(difflib.unified_diff(raw_ov, train_argv(ep)[2:], "raw C3r2_tier2 overrides", "K3 k3easy argv", lineterm="", n=20))
            res["episodes"][f"{ep:02d}"] = {"ok": ok, "raw_hydra_dir": str(hyd), "compose_reproduces_raw_config_yaml": repro,
                                            "n_config_keys": len(raw_saved), "config_diff_k3_vs_raw": diff,
                                            "overrides_raw": raw_ov, "argv_k3": train_argv(ep)[2:], "overrides_unified_diff": udiff}
            print(f"== ep{ep:02d} {e['obj']}: compose(raw overrides) == raw run's saved config.yaml: {repro} "
                  f"({len(raw_saved)} keys); K3 vs raw differ in {len(diff)} key(s), all allowed: {dkeys <= CFG_ALLOWED_DIFF} -> {'ok' if ok else 'FAIL'}")
            for d in diff:
                print(f"   {d['key']}: {d['raw']}  ->  {d['k3']}")
            print("   overrides (raw -> K3):")
            for line in udiff[2:]:
                print(f"     {line}")
    res["all_ok"] = bool(ok_all)
    print(f"cfgcheck: {'OK' if ok_all else 'FAILED'}")
    if write:
        atomic_write(CFGCHECK, json.dumps(res, indent=1, default=str))
        print(f"wrote {CFGCHECK}")
    return 0 if ok_all else 1


# ============================================================================================== state / crash
def ep_state(ep: int) -> tuple[str, str]:
    run = run_dir(ep)
    tr = find_trainers(ep)
    pr = progress(ep)
    pinfo = f"it={pr['it'] if pr['it'] is not None else pr['last_model']} help_zero_iters={pr['help_zero_iters']}"
    if tr:
        return "RUNNING", f"pid={','.join(map(str, tr))} {pinfo} gpu={EPS[ep]['gpu']}"
    why = H.done_reason(run, ITERS)
    if why:
        st = load_post(ep)
        if lock_held(K / "post" / f"ep{ep:02d}.lock"):
            return "POST_RUNNING", f"{why}; post {st.get('status', '?')}"
        if st.get("status") in TERMINAL:
            return "POST_DONE", f"{why}; post {st['status']}"
        if st.get("status") == "retry" and float(st.get("retry_after", 0)) > time.time():
            return "POST_RETRY_WAIT", f"{why}; post retry after {time.strftime('%H:%M:%S', time.localtime(float(st['retry_after'])))}"
        return "TRAINED", f"{why}; post {st.get('status', 'not started')}"
    if (K / f"CRASHED_ep{ep:02d}").exists():
        return "CRASHED", f"marker {K / f'CRASHED_ep{ep:02d}'}; {pinfo}"
    launched = read_pids().get(ep)
    has_files = pr["models"] > 0 or (run / "final.json").exists() or any(run.glob("rsi_*.safetensors"))
    if launched and time.time() - launched["t"] < STARTING_GRACE_S and not has_files:
        return "STARTING", f"launched {int(time.time() - launched['t'])} s ago (pid {launched['pid']}), trainer not visible yet"
    if has_files or launched:
        return "DEAD", (f"no live trainer, no final.json; {pr['models']} model_* file(s), last {pr['last_model']}, "
                        f"{'launched by this test at ' + time.strftime('%F %T', time.localtime(launched['t'])) if launched else 'never launched by this test'}")
    return "FRESH", f"no run files in {run}"


def state() -> int:
    for ep in ORDER:
        s, d = ep_state(ep)
        print(f"EP {ep:02d} {s} {d}")
    return 0


def crash(ep: int, why: str) -> int:
    path = K / f"CRASHED_ep{ep:02d}"
    if path.exists():
        return 0
    log = RUNS / "logs" / f"ep{ep:02d}.log"
    try:
        tail = log.read_text(errors="replace").splitlines()[-30:]
    except OSError:
        tail = []
    pr = progress(ep)
    text = (f"CRASHED ep{ep:02d} {EPS[ep]['obj']} detected {now()}\n{why}\nlast model {pr['last_model']}, last logged iteration "
            f"{pr['it']}, help_zero_iterations {pr['help_zero_iters']}\nNo automatic relaunch (PREREG): the clip is INCOMPLETE "
            f"unless the user decides a relaunch (resume from the newest finite checkpoint), which is then disclosed.\n"
            f"--- last lines of {log}\n" + "\n".join(tail) + "\n")
    atomic_write(path, text)
    print(f"wrote {path}")
    return 0


# ============================================================================================== post-training step
def child_env(gpu: int, ep: int) -> dict:
    sp = ME / "conda_envs/flashchord/lib/python3.11/site-packages/nvidia"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CUDA_MPS_", "WANDB_", "RUNIN_", "TIER2V2_", "K3EASY"))
           and k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "TIER2_GPU")}
    env.update({
        "PATH": f"{ME}/conda_envs/flashchord/bin:{env.get('PATH', '/usr/bin:/bin')}",
        "LD_LIBRARY_PATH": f"{sp}/cu13/lib:{sp}/cudnn/lib:{sp}/nccl/lib:{sp}/nvshmem/lib"
                           + (f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else ""),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": f"{ME}/.cache/pycache", "XDG_CACHE_HOME": f"{ME}/.cache", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(gpu), "HUMAN_MOTION_DATA_DIR": str(DATA), "OMP_NUM_THREADS": "4",
        "K3EASY_EVAL": "1", "K3EASY_EVAL_EP": f"{ep:02d}",
    })
    return env


def eval_command(ep: int, gpu: int, w: Path) -> list[str]:
    """track2_tier2v2_post.py run_eval's command with this arm's work dir, arm name and 45000 fallback iteration."""
    command = [str(FCPY), "-B", str(CODE / "scripts/track2/eval_all.py"), "--arm-dir", str(w), "--arm", ARM,
               "--checkpoint-from-final", "--iteration", str(ITERS), "--episodes", str(ep), "--gpus", str(gpu),
               "--jobs-per-gpu", "1", "--world-count", "1", "--export", "yes", "--export-jobs", "1",
               "--export-tolerance", repr(EXPORT_TOL), "--skip-pack", "--kit-root", str(KIT),
               "--stock-runtime", str(STOCK), "--dataset-root", str(DATASET),
               "--manifest", f"{DATA}/whole_body/soma/track2_manifest.json", "--pack-python", str(KPY),
               "--results", str(w / "results"), "--code", str(CODE), "--python", str(FCPY)]  # fmt: skip
    return ["nice", "-n", "10", "taskset", "-c", H.EVAL_CORES[0 if gpu <= 2 else 1]] + command


def score_command(ep: int, w: Path) -> list[str]:
    return ["nice", "-n", "10", str(KPY), "-I", "-B", str(CLEAN_SCORER), "--kit", str(KIT),
            "--out", str(w / "clean_scores_LOCAL_NOT_FOR_UPLOAD.csv"), "--tier1", str(TIER1_CSV), "--",
            "--track", "track_2_tier2", "--results", str(w / "results"), "--dataset-root", str(DATASET),
            "--code-commit", COMMIT, "--eval-script", str(KIT / "flash_chord/scripts/evaluate_policy.py"),
            "--frame-references", str(FRAME_REFS), "--config", str(w / "pack" / "tracks.json"),
            "-o", str(w / "packed_LOCAL_NOT_FOR_UPLOAD.csv")]  # fmt: skip


def acquire_lane(ep: int, gpu: int) -> list[int]:
    """Wait (indefinitely, logging each new reason) for the eval lane of the training GPU: no MPS, >= 4 GB free, our
    lane lock k3easy/evallock/gpu<g>.lock and, when it exists, the rolling start's fc_runs/queue_runin/evallock/
    gpu<g>.lock (opened read-only, never created; its watcher takes it non-blocking)."""
    OWN_EVALLOCK.mkdir(parents=True, exist_ok=True)
    last = ""
    while True:
        why = ""
        try:
            st = H.smi()
            mps = H.mps_gpus(st, H.all_pids())
            if gpu not in st["gpus"]:
                why = f"gpu{gpu} not listed by nvidia-smi"
            elif gpu in mps:
                why = f"gpu{gpu} has MPS ({mps[gpu]})"
            elif st["gpus"][gpu]["free"] < EVAL_MIN_FREE_MB:
                why = f"gpu{gpu} free {st['gpus'][gpu]['free']} MiB < {EVAL_MIN_FREE_MB}"
        except Exception as error:  # noqa: BLE001
            why = f"nvidia-smi failed ({type(error).__name__})"
        if not why:
            mine = try_lock(OWN_EVALLOCK / f"gpu{gpu}.lock", create=True)
            if mine is None:
                why = f"gpu{gpu}: another k3easy evaluation holds the lane"
            else:
                fds = [mine]
                theirs = RUNIN_EVALLOCK / f"gpu{gpu}.lock"
                if theirs.exists():
                    t = try_lock(theirs, create=False)
                    if t is None:
                        os.close(mine)
                        why = f"gpu{gpu}: a rolling-start evaluation holds {theirs}"
                    else:
                        fds.append(t)
                if not why:
                    plog(ep, f"eval lane gpu{gpu} taken (k3easy lock{' + ' + str(theirs) if len(fds) > 1 else '; no rolling-start lock file for this GPU'})")
                    return fds
        if why != last:
            plog(ep, f"eval waits: {why}")
            last = why
        time.sleep(30)


def run_eval(ep: int, gpu: int, w: Path, fds: list[int]) -> tuple[int, str]:
    link = w / f"ep{ep:02d}"
    if not link.is_symlink():
        link.symlink_to(run_dir(ep))
    (w / "eval_report.json").unlink(missing_ok=True)
    cmd = eval_command(ep, gpu, w)
    note = ""
    with (w / "eval_all.log").open("a") as stream:
        stream.write(f"[{now()}] gpu {gpu}: {' '.join(cmd)}\n")
        stream.flush()
        proc = subprocess.Popen(cmd, cwd=CODE, env=child_env(gpu, ep), stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True, pass_fds=tuple(fds))
        try:
            code = proc.wait(timeout=EVAL_TIMEOUT)
        except subprocess.TimeoutExpired:
            note, code = f"timeout after {EVAL_TIMEOUT:.0f} s", -9
            for sig in (signal.SIGTERM, signal.SIGKILL):  # our own evaluation session only (plain GPU, never MPS)
                try:
                    os.killpg(proc.pid, sig)
                except OSError:
                    break
                try:
                    proc.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    continue
    return code, note


def parse_eval(ep: int, w: Path) -> dict:
    out: dict = {}
    try:
        rep = json.loads((w / "eval_report.json").read_text())
        row = next(r for r in rep.get("episodes", []) if int(r.get("episode", -1)) == ep)
    except (OSError, ValueError, StopIteration, TypeError):
        return {"eval_status": "no_report"}
    out.update({"eval_status": row.get("status"), "evaluated_iteration": row.get("iteration"),
                "selected_by": row.get("selected_by"), "final_reason_eval": row.get("final_reason"),
                "export_max_rel_diff": fnum(row.get("export_max_rel_diff")), "export_max_abs_diff": fnum(row.get("export_max_abs_diff")),
                "export_stock_verified": row.get("export_stock_verified"), "checkpoint_sha256": row.get("checkpoint_sha256"),
                "evaluated_checkpoint": row.get("evaluated_checkpoint"), "source_checkpoint": row.get("source_checkpoint")})
    try:
        er = json.loads((w / "export" / f"ep{ep:02d}" / "export_report.json").read_text())
        out["export_status"] = er.get("status")
        out["export_source_sha256"] = er.get("source_sha256")
        for key in ("max_rel_diff", "max_abs_diff"):
            if out.get(f"export_{key}") is None:
                out[f"export_{key}"] = fnum(er.get(key))
    except (OSError, ValueError):
        out["export_status"] = None
    try:
        res = json.loads((w / "results" / f"episode_{ep:06d}.json").read_text())
        m = res.get("metrics", {})
        ref = res.get("parquet") or (res.get("reference") or {}).get("parquet")
        out.update({"eval_add_auc_own_input": fnum(m.get("add_auc")),
                    "eval_sp_sr": fnum(m.get("spider_sr_uncentered", m.get("spider_sr"))),
                    "eval_mp_sr": fnum(m.get("maniptrans_sr")), "eval_chord_sr": fnum(m.get("chord_sr")),
                    "eval_mppe_cm_own_input": fnum(m.get("mppe_cm", m.get("mean_per_point_error_cm"))),
                    "eval_world_count": m.get("world_count", res.get("world_count")),
                    "eval_non_finite_worlds": res.get("non_finite_worlds"), "eval_reference_parquet": ref,
                    "eval_reference_is_train_input": bool(ref) and Path(ref).resolve() == parquet(ep).resolve()
                    and sha256(ref) == EPS[ep]["sha"]})
    except (OSError, ValueError):
        pass
    return out


def score(ep: int, w: Path) -> dict:
    """The local clean diagnostic (LOCAL_NOT_FOR_UPLOAD; never selects a checkpoint, never decides the verdict). The
    clean numbers go only to clean_LOCAL_NOT_FOR_UPLOAD.json; the returned dict carries status and residuals."""
    if not (FRAME_REFS / f"episode_{ep:06d}.npz").is_file():
        return {"score_status": "no_frame_refs"}
    pack = w / "pack"
    pack.mkdir(exist_ok=True)
    cfg = json.loads((KIT / "config/tracks.json").read_text())
    cfg["tracks"]["track_2_tier2"]["scored_episodes"] = [ep]
    (pack / "tracks.json").write_text(json.dumps(cfg, indent=2))
    out_csv = w / "clean_scores_LOCAL_NOT_FOR_UPLOAD.csv"
    log_path = w / "clean_score_LOCAL_NOT_FOR_UPLOAD.log"
    out_csv.unlink(missing_ok=True)
    cmd = score_command(ep, w)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"})
    with log_path.open("w") as stream:
        stream.write(f"[{now()}] {' '.join(cmd)}\n")
        stream.flush()
        try:
            code = subprocess.run(cmd, cwd=KIT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=SCORE_TIMEOUT).returncode
        except subprocess.TimeoutExpired:
            code = -9
    out: dict = {"score_exit": code}
    text = log_path.read_text(errors="replace")
    m = re.search(r"input reference check: .*?maximum residual ([0-9.eE+-]+) m, ([0-9.eE+-]+) rad", text)
    if m:
        out["input_check_residual_m"], out["input_check_residual_rad"] = float(m.group(1)), float(m.group(2))
    try:
        rows = [r for r in csv.DictReader(out_csv.open()) if r["episode"].strip().isdigit() and int(r["episode"]) == ep]
    except OSError:
        rows = []
    clean = {"episode": ep, "object": EPS[ep]["obj"], "written": now(), "frame_refs": str(FRAME_REFS),
             "note": "LOCAL clean-reference diagnostic, arm-level and disclosed; never selects a checkpoint (final.json's is "
                     "fixed) and never enters the PREREG verdict"}
    if code == 0 and rows:
        r = rows[0]
        clean.update({"clean_add_auc": fnum(r.get("t2clean_add_auc")), "clean_sp_sr": fnum(r.get("t2clean_spider_sr")),
                      "clean_mp_sr": fnum(r.get("t2clean_maniptrans_sr")), "clean_mppe_cm": fnum(r.get("t2clean_mppe_cm"))})
        out["score_status"] = "ok"
    else:
        out["score_status"] = "failed"
        # the scorer prints "MEAN t2clean_..." before a non-zero packer exit: its tail may hold clean numbers, so it goes
        # only to the LOCAL file; post.json (not LOCAL) gets a pointer (review fix 2026-10-21)
        out["score_note"] = f"clean scorer exit {code}; details only in {log_path.name}"
        clean["score_failure_tail"] = " | ".join([l for l in text.splitlines() if l.strip()][-3:])[:400]
    atomic_write(w / "clean_LOCAL_NOT_FOR_UPLOAD.json", json.dumps(clean, indent=1))
    return out


def post(ep: int, gpu: int) -> int:
    if gpu != EPS[ep]["gpu"] or gpu == 4:
        print(f"ep{ep:02d} evaluates on its training GPU {EPS[ep]['gpu']}, not {gpu}")
        return 2
    (K / "post").mkdir(parents=True, exist_ok=True)
    lock = try_lock(K / "post" / f"ep{ep:02d}.lock", create=True)
    if lock is None:
        print(f"ep{ep:02d}: another post holds its lock; nothing to do")
        return 0
    w = EVROOT / f"ep{ep:02d}"
    w.mkdir(parents=True, exist_ok=True)
    st = load_post(ep) or {"episode": ep, "object": EPS[ep]["obj"], "status": "pending", "tries": 0}
    if st.get("status") in TERMINAL:
        print(f"ep{ep:02d}: post already {st['status']}")
        return 0
    st["post_starts"] = int(st.get("post_starts", 0)) + 1
    if st["post_starts"] > MAX_POST_STARTS:
        st.update({"status": "failed", "finished_at": now(), "note": f"post started {st['post_starts']} times without finishing"})
        save_post(ep, st)
        plog(ep, st["note"])
        return 1
    if st.get("status") == "retry":
        wait = float(st.get("retry_after", 0)) - time.time()
        if wait > 0:
            time.sleep(wait)
    run = run_dir(ep)
    why, fin = H.done_reason(run, ITERS), H.final_marker(run)
    if why is None or fin is None:
        plog(ep, f"training not finished ({run}): nothing to do")
        save_post(ep, st)
        return 3
    if find_trainers(ep):
        plog(ep, "a trainer of this run dir is still alive: not evaluating yet")
        save_post(ep, st)
        return 3
    try:
        age = time.time() - (run / fin["checkpoint"]).stat().st_mtime
    except OSError:
        age = MIN_FINAL_AGE
    if age < MIN_FINAL_AGE:
        time.sleep(MIN_FINAL_AGE - age)
    psha = sha256(parquet(ep))
    st.update({"final_iteration": fin["iteration"], "final_reason": fin.get("reason"), "help_zero_since": fin.get("help_zero_since"),
               "train_gpu": gpu, "parquet": str(parquet(ep)), "parquet_sha256": psha, "eval_command": " ".join(eval_command(ep, gpu, w))})
    if psha != EPS[ep]["sha"]:
        st.update({"status": "failed", "finished_at": now(), "note": f"the K3 parquet changed since the build ({psha})"})
        save_post(ep, st)
        plog(ep, st["note"])
        return 1
    st["status"] = "waiting_gpu"
    save_post(ep, st)
    plog(ep, f"final.json: {why}; evaluating model_{fin['iteration']} against its own input (export tol {EXPORT_TOL:g}) on gpu{gpu}")
    fds = acquire_lane(ep, gpu)
    st.update({"status": "evaluating", "eval_gpu": gpu, "eval_started_at": now()})
    save_post(ep, st)
    t0 = time.time()
    try:
        code, note = run_eval(ep, gpu, w, fds)
    finally:
        for fd in fds:
            os.close(fd)
    res = parse_eval(ep, w)
    st.update(res)
    st.update({"eval_exit": code, "eval_seconds": round(time.time() - t0, 1), "eval_note": note})
    tries = int(st.get("tries", 0)) + 1
    if res.get("eval_status") == "evaluated" and code == 0:
        plog(ep, f"evaluated on gpu{gpu} against its own input: SP-SR {res.get('eval_sp_sr')} MP-SR {res.get('eval_mp_sr')} "
                 f"world_count {res.get('eval_world_count')} AUC(own input) {res.get('eval_add_auc_own_input')} "
                 f"export max_rel_diff {res.get('export_max_rel_diff')}; reference is the K3 training parquet: {res.get('eval_reference_is_train_input')}")
        sc = score(ep, w)
        st.update(sc)
        st["status"] = "done" if sc.get("score_status") == "ok" else "done_score_failed"
        st["finished_at"] = now()
        plog(ep, f"local clean diagnostic: {sc.get('score_status')} (numbers only in {w}/clean_LOCAL_NOT_FOR_UPLOAD.json)")
    elif str(res.get("eval_status", "")).startswith("export_failed") and res.get("export_max_rel_diff") is not None \
            and res["export_max_rel_diff"] > EXPORT_TOL:
        st.update({"status": "export_refused", "finished_at": now(),
                   "note": f"export refused at tolerance {EXPORT_TOL:g}: max_rel_diff {res['export_max_rel_diff']:.3g}"})
        plog(ep, st["note"])
    elif tries < MAX_TRIES:
        st.update({"status": "retry", "retry_after": time.time() + RETRY_DELAY,
                   "note": f"technical failure (exit {code}, {res.get('eval_status')}{', ' + note if note else ''}); retry in {RETRY_DELAY:.0f} s"})
        plog(ep, st["note"])
    else:
        st.update({"status": "failed", "finished_at": now(),
                   "note": f"evaluation failed {tries} times (exit {code}, {res.get('eval_status')}); see {w / 'eval_all.log'}"})
        plog(ep, st["note"])
    st["tries"] = tries
    save_post(ep, st)
    os.close(lock)
    return 0


# ============================================================================================== verdict
def _rows_wandb(path: str) -> list[dict]:
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal.datastore import DataStore
    ds = DataStore()
    ds.open_for_scan(path)
    out = []
    while True:
        d = ds.scan_data()
        if d is None:
            break
        rec = pb.Record()
        rec.ParseFromString(d)
        if rec.WhichOneof("record_type") != "history":
            continue
        row = {}
        for it in rec.history.item:
            k = it.key if it.key else "/".join(it.nested_key)
            try:
                row[k] = json.loads(it.value_json)
            except ValueError:
                pass
        out.append(row)
    return out


def _bins(rows: list[dict]) -> dict[int, list[float]]:
    """{iteration bin (1000-wide, centred): [sum fail, n]} of rsi/fail_rate_mean."""
    out: dict[int, list[float]] = {}
    for r in rows:
        f = r.get("rsi/fail_rate_mean")
        s = r.get("step", r.get("_step"))
        if not isinstance(f, (int, float)) or not isinstance(s, (int, float)) or not math.isfinite(f):
            continue
        b = int(round(s / SPI / 1000.0)) * 1000
        acc = out.setdefault(b, [0.0, 0])
        acc[0] += f
        acc[1] += 1
    return out


def curves(ep: int, write: bool) -> tuple[dict, dict, str]:
    """(raw bins, k3 bins, note). Raw from the C3r2_tier2 run's offline W&B record (cached in k3easy/raw_curves/)."""
    note = ""
    cache = K / "raw_curves" / f"ep{ep:02d}.json"
    raw: dict = {}
    try:
        raw = {int(k): v for k, v in json.loads(cache.read_text()).items()}
    except (OSError, ValueError):
        files = sorted(glob.glob(str(RAW_RUNS / f"ep{ep:02d}/wandb/run-*/run-*.wandb")))
        if files:
            try:
                raw = _bins(_rows_wandb(files[-1]))
                if write and raw:
                    atomic_write(cache, json.dumps(raw))
            except Exception as error:  # noqa: BLE001
                note = f"raw curve unreadable ({type(error).__name__}: {str(error)[:80]})"
        else:
            note = "no raw W&B record"
    k3: dict = {}
    path = run_dir(ep) / "metrics.jsonl"
    if path.is_file():
        rows = []
        with path.open() as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        k3 = _bins(rows)
    return raw, k3, note


def first_help_zero(ep: int) -> int | None:
    path = run_dir(ep) / "metrics.jsonl"
    if not path.is_file():
        return None
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            v, s = r.get("early_stop/help_zero_iterations"), r.get("step")
            if isinstance(v, (int, float)) and v > 0 and isinstance(s, (int, float)):
                return int(round(s / SPI)) - int(v)
    return None


def raw_eval_rows() -> dict[int, dict]:
    out = {}
    try:
        for r in csv.DictReader(RAW_EVAL_CSV.open()):
            if r["episode"].strip().isdigit():
                out[int(r["episode"])] = r
    except (OSError, KeyError):
        pass
    return out


def verdict(write: bool) -> int:
    ok, why = prereg_status()
    if not ok:
        print(f"VERDICT REFUSED: {why} (the rule is fixed by PREREG.md; nothing written)")
        return 2
    raw_eval = raw_eval_rows()
    clips, fail_rows, clean = {}, [], {}
    for ep in ORDER:
        e = EPS[ep]
        run = run_dir(ep)
        fin = H.final_marker(run) if H.done_reason(run, ITERS) else None
        crashed = (K / f"CRASHED_ep{ep:02d}").exists()
        c = {"episode": ep, "object": e["obj"], "gpu": e["gpu"], "raw_help_zero_since": e["raw_hz"],
             "raw_final_iteration": e["raw_final"], "a_limit": e["raw_hz"] + HZ_MARGIN, "crash_marker": crashed}
        if fin:
            hz = fin.get("help_zero_since")
            c.update({"k3_help_zero_since": hz, "k3_final_iteration": fin["iteration"], "k3_final_reason": fin.get("reason")})
            c["a_pass"] = hz is not None and int(hz) <= c["a_limit"]
        else:
            c["a_pass"] = None
        c["k3_first_help_zero_report_only"] = first_help_zero(ep)
        st = load_post(ep)
        c["post_status"] = st.get("status")
        b = None
        if st.get("status") in ("done", "done_score_failed"):
            b = (st.get("eval_status") == "evaluated" and st.get("eval_world_count") == 1 and st.get("eval_sp_sr") == 1.0
                 and st.get("eval_mp_sr") == 1.0 and st.get("eval_reference_is_train_input") is True
                 and fin is not None and st.get("evaluated_iteration") == fin["iteration"])
        elif st.get("status") == "export_refused":
            b = False
        c["b_pass"] = b
        for key in ("eval_sp_sr", "eval_mp_sr", "eval_world_count", "evaluated_iteration", "eval_add_auc_own_input",
                    "eval_mppe_cm_own_input", "export_max_rel_diff", "eval_reference_is_train_input", "note"):
            c[key] = st.get(key)
        rr = raw_eval.get(ep, {})
        c["raw_own_input_eval"] = {"iteration": rr.get("iteration"), "auc": fnum(rr.get("auc")), "sp_sr": fnum(rr.get("sp_sr")),
                                   "mp_sr": fnum(rr.get("mp_sr")), "mppe_cm": fnum(rr.get("mppe_cm"))}
        if c["a_pass"] is False or b is False:
            res = "FAIL"
        elif c["a_pass"] and b:
            res = "PASS"
        elif (crashed and fin is None) or st.get("status") == "failed":
            res = "INCOMPLETE"
        else:
            res = "PENDING"
        settled = res == "INCOMPLETE" or st.get("status") in TERMINAL
        c["result"], c["settled"] = res, settled
        raw_b, k3_b, cnote = curves(ep, write)
        matched = sorted(set(raw_b) & set(k3_b))
        for it in matched:
            rf, kf = raw_b[it][0] / raw_b[it][1], k3_b[it][0] / k3_b[it][1]
            fail_rows.append({"episode": ep, "iteration_bin": it, "raw_fail_rate_mean": round(rf, 4),
                              "k3_fail_rate_mean": round(kf, 4), "k3_minus_raw": round(kf - rf, 4)})
        if matched:
            rm = sum(raw_b[i][0] / raw_b[i][1] for i in matched) / len(matched)
            km = sum(k3_b[i][0] / k3_b[i][1] for i in matched) / len(matched)
            c["fail_rate_report_only"] = {"matched_bins": len(matched), "up_to_iteration": matched[-1], "raw_mean": round(rm, 4),
                                          "k3_mean": round(km, 4), "note": cnote}
        else:
            c["fail_rate_report_only"] = {"matched_bins": 0, "note": cnote or "no K3 metrics yet"}
        try:
            clean[ep] = json.loads((EVROOT / f"ep{ep:02d}" / "clean_LOCAL_NOT_FOR_UPLOAD.json").read_text())
        except (OSError, ValueError):
            clean[ep] = None
        clips[f"{ep:02d}"] = c
    results = [c["result"] for c in clips.values()]
    final = all(c["settled"] for c in clips.values())
    if all(r == "PASS" for r in results):
        arm = "K3 trainable on easy clips"
    elif "FAIL" in results:
        arm = "K3 NOT shown trainable on easy clips"
    elif "PENDING" in results:
        arm = "PENDING"
    else:
        arm = "INCOMPLETE"
    out = {"written": now(), "prereg": str(PREREG), "prereg_sha256": sha256(PREREG), "prereg_check": why,
           "rule": ("clip PASS iff (a) final.json help_zero_since <= raw help_zero_since + 5000 and (b) ONE official "
                    "evaluation of final.json's checkpoint against its own (K3) input: world_count 1, SP-SR = MP-SR = 1; "
                    "arm 'K3 trainable on easy clips' iff all 3 clips PASS"),
           "arm_verdict": arm, "final": final, "clips": clips}
    lines = [f"K3 easy-clip test ({ARM}) - verdict by the pre-registered rule ({why})",
             f"written {out['written']}; final: {'yes' if final else 'no (not every clip settled)'}",
             f"ARM VERDICT: {arm}", ""]
    for c in clips.values():
        lines.append(f"ep{c['episode']:02d} {c['object']:<18} {c['result']:<10} (a) help zero K3 {c.get('k3_help_zero_since')} vs raw "
                     f"{c['raw_help_zero_since']} (limit {c['a_limit']}) -> {c['a_pass']} | (b) own-input eval SP-SR {c.get('eval_sp_sr')} "
                     f"MP-SR {c.get('eval_mp_sr')} worlds {c.get('eval_world_count')} ckpt it {c.get('evaluated_iteration')} "
                     f"-> {c['b_pass']} | post {c['post_status']}{' | CRASH MARKER' if c['crash_marker'] else ''}")
    lines += ["", "Reported, not decided:"]
    for c in clips.values():
        f = c["fail_rate_report_only"]
        lines.append(f"  ep{c['episode']:02d} training fail rate (rsi/fail_rate_mean, mean of 1000-it bins both runs reached): "
                     + (f"raw {f['raw_mean']} vs K3 {f['k3_mean']} over {f['matched_bins']} bins up to {f['up_to_iteration']}"
                        if f.get("matched_bins") else f.get("note", "")) +
                     f"; K3 final {c.get('k3_final_iteration')} vs raw {c['raw_final_iteration']}; evaluator AUC on own input "
                     f"K3 {c.get('eval_add_auc_own_input')} vs raw {c['raw_own_input_eval']['auc']}")
    lines.append("  per-bin table: fail_rates_vs_raw.csv; clean diagnostics only in VERDICT_clean_LOCAL_NOT_FOR_UPLOAD.json")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if write:
        atomic_write(K / "VERDICT.json", json.dumps(out, indent=1, default=str))
        atomic_write(K / "VERDICT.txt", text)
        with (K / "fail_rates_vs_raw.csv.tmp").open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["episode", "iteration_bin", "raw_fail_rate_mean", "k3_fail_rate_mean", "k3_minus_raw"])
            wr.writeheader()
            wr.writerows(fail_rows)
        os.replace(K / "fail_rates_vs_raw.csv.tmp", K / "fail_rates_vs_raw.csv")
        base = {}
        try:
            for r in csv.DictReader(BASELINE_CSV.open()):
                if r["episode"].strip().isdigit() and int(r["episode"]) in EPS:
                    base[int(r["episode"])] = {k: fnum(r.get(f"t2train_mean_{k}")) for k in ("add_auc", "spider_sr", "maniptrans_sr", "mppe_cm")}
        except (OSError, KeyError):
            pass
        atomic_write(K / "VERDICT_clean_LOCAL_NOT_FOR_UPLOAD.json", json.dumps(
            {"written": now(), "note": "LOCAL ONLY. One clean-reference diagnostic per clip (arm-level, disclosed); never selects "
                                       "checkpoints, never enters the verdict. Raw = C3r2_tier2 final checkpoint, mean of 3 evaluations "
                                       "(early_table.csv t2train_mean_*).",
             "k3": {f"{ep:02d}": clean[ep] for ep in ORDER}, "raw_mean3": {f"{ep:02d}": base.get(ep) for ep in ORDER}},
            indent=1, default=str))
        print(f"wrote {K / 'VERDICT.json'}, VERDICT.txt, fail_rates_vs_raw.csv, VERDICT_clean_LOCAL_NOT_FOR_UPLOAD.json")
    return 0


def heartbeat() -> int:
    parts = []
    for ep in ORDER:
        s, _ = ep_state(ep)
        pr = progress(ep)
        it = pr["it"] if pr["it"] is not None else pr["last_model"]
        parts.append(f"ep{ep:02d} {s} it={it} hz={pr['help_zero_iters']} fail={round(pr['fail'], 3) if isinstance(pr['fail'], float) else pr['fail']}")
    print(" | ".join(parts))
    return 0


# ============================================================================================== main
def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    sub, rest = argv[0], argv[1:]

    def opt(name, default=None):
        return rest[rest.index(name) + 1] if name in rest else default

    if sub == "check":
        return check()
    if sub == "cfgcheck":
        return cfgcheck("--no-write" not in rest)
    if sub == "cmd":
        sys.stdout.write("\0".join(train_argv(ep_of(opt("--ep")))) + "\0")
        return 0
    if sub == "plan":
        rem = opt("--remaining")
        print(plan(ep_of(opt("--ep")), int(opt("--gpu")), opt("--exclude", "") or "", int(rem) if rem else None))
        return 0
    if sub == "gpus":
        print("\n".join(gpus_report()))
        return 0
    if sub == "state":
        return state()
    if sub == "trainer":
        print(" ".join(map(str, find_trainers(ep_of(opt("--ep"))))))
        return 0
    if sub == "crash":
        return crash(ep_of(opt("--ep")), opt("--why", ""))
    if sub == "post":
        return post(ep_of(opt("--ep")), int(opt("--gpu")))
    if sub == "verdict":
        return verdict("--no-write" not in rest)
    if sub == "heartbeat":
        return heartbeat()
    if sub == "build-record":  # one-time, at build: pin the code-tree digest (check compares against it)
        atomic_write(K / "build.json", json.dumps({"written": now(), "code": str(CODE), "pin": PIN, "code_digest": code_digest()}, indent=1))
        print(f"wrote {K / 'build.json'}")
        return 0
    print(f"unknown subcommand {sub}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
