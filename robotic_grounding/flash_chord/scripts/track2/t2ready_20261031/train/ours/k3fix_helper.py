#!/usr/bin/env python3
"""k3fix_helper.py: checks, GPU / CPU plan, state, post-training step and verdict of scripts/k3fix_driver.sh, the STAGED
K3fix pilot (arm C3r2_tier2_k3fix; pre-registration tmp_analysis/tier2/k3fix_pilot/PREREG_v2.md, which supersedes
PREREG.md (sha256 390d7996, kept unchanged)). Generalises scripts/k3easy_helper.py (the easy-clip K3 test) to several
jobs per episode, two data roots, three stages and a GPU chosen at launch.

  C3r2 exactly as the raw-input Tier-2 arm C3r2_tier2 (code/fc_r2 @61fa5d3, experiment g1_recon_body_coord_rsi_r2,
  max_iterations 45000, experiment-default RSI deadline 35156 / zero_ramp 3906 / help-zero early stop 5860, scene.support
  as the raw arm). Jobs by stage, in launch order (run dir fc_runs/C3r2_tier2_k3fix/<job>):
    stage 1  ep07_k3fix_s42  ep07 iron on data/human_motion_data_tier2_k3fix, seed 42 (K3fix leaves ep07 unchanged:
                             the parquet is byte-identical to the K3 root's)
             ep18_k3fix_s42  ep18 blue_trash_can on the K3fix root (= K3, byte-identical), seed 42 (scene.support=false as raw)
             ep03_k3fix_s42  ep03 contigo_coffee_mug on the K3fix root (K3fix-changed), seed 42
             ep03_raw_s7     ep03 on the RAW Tier-2 input data/human_motion_data_tier2, reset.seed=7 (seed-variance control)
    stage 2  ep17_k3fix_s42  ep17 traffic_cone on the K3fix root, seed 42
             ep09_k3fix_s42  ep09 big_red_bowl on the K3fix root, seed 42
    stage 3  ep04_k3fix_s42  ep04 baseball_bat on the K3fix root, seed 42 (scene.support=false as raw; report only)
  A K3fix job's only recipe difference from the raw run is the data root; the control's only difference is the seed.
  A stage is launched only by an explicit 'scripts/k3fix_driver.sh run --stage <n>'.

Run with conda_envs/flashchord/bin/python -I -B. Never starts a training, never signals a process except its own
evaluation session on timeout. Writes only under tmp_analysis/tier2/k3fix_pilot/ (eval work dirs, post / verdict files,
CRASHED markers, raw-curve caches). nvidia-smi / /proc / CPU-slice primitives come from
scripts/track2_queue_tier2v2_helper.py (imported read-only).

Subcommands:
  check                    preflight (PREREG sha256, code pin + digest, experiment values, data root READY + sha table,
                           gates / build / loader reports, per-job inputs and raw references, cfgcheck.json)
  cfgcheck [--no-write]    hydra-compose each raw run's recorded overrides (must reproduce its saved config.yaml) and
                           each job's argv; the difference must be only the allowed keys; writes k3fix_pilot/cfgcheck.json
  cmd --job J              the training argv, NUL-separated
  plan --job J [--exclude CPUS] [--remaining N] [--assume g:n,..] [--stages S]
                           "OK <gpu> <cpus> <nthreads> - | why" or "WAIT why" (GPUs 5/7 first, then 3/6; never 0-2 or 4)
  data --job J             the job's data root (HUMAN_MOTION_DATA_DIR)
  gpus                     per-GPU view
  state [--stages S]       one line per job "JOB <job> <STATE> detail"; with --stages, a FRESH job of a stage not in S is
                           reported DEFERRED (S = comma list of 1/2/3; "" = none, as in adopt)
  stage-jobs --stages S    the jobs of stage(s) S in launch order (exit 2 on a bad S)
  stage-guard --stages S   "OK ..." or "REFUSE ..." (an earlier stage that is not in S still has a never-launched job)
  trainer --job J          pid(s) of the live trainer(s) of that run dir
  crash --job J --why T    write k3fix_pilot/CRASHED_<job> once (never a relaunch)
  post --job J             after-training step on the training GPU (pids.tsv): export 2e-4 + ONE official evaluation of
                           final.json's checkpoint vs its own input, then the LOCAL clean diagnostic
  verdict [--no-write]     apply the PREREG rules; exit 2 (nothing written) when PREREG.sha256 does not match
  heartbeat                one status line
  build-record             (once, at build) pin the code-tree digest in k3fix_pilot/build.json
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

ARM = "C3r2_tier2_k3fix"
EXP = "g1_recon_body_coord_rsi_r2"
ITERS = 45000
K = ME / "tmp_analysis/tier2/k3fix_pilot"
RUNS = ME / "fc_runs" / ARM
EVROOT = K / "eval"
CODE = ME / "code/fc_r2"
PIN = "61fa5d31ff8054bf1e5cf5375ef0957e54d2729b"
DATA_FIX = ME / "data/human_motion_data_tier2_k3fix"
RAW_DATA = ME / "data/human_motion_data_tier2"
K3_DATA = ME / "data/human_motion_data_tier2_k3"
RAW_RUNS = ME / "fc_runs/C3r2_tier2"
OLD_K3_RUNS = ME / "fc_runs/C3r2_tier2_k3_stretch90k"  # report-only reference for ep04 (read-only)
RAW_EVAL_CSV = ME / "fc_runs/eval_tier2/C3r2_tier2_final.csv"
BASELINE_CSV = ME / "tmp_analysis/tier2/early_scores/early_table.csv"
FRAME_REFS_FIX = K / "frame_refs_k3fix"
BUILD = K / "build"
CFGCHECK = K / "cfgcheck_v2.json"      # v2: all seven jobs (the v1 cfgcheck.json of five jobs is kept)
GATES_V2 = BUILD / "gates_v2.json"     # G1-G4 + identity checks of the stage-1 clips ep07 / ep18 (scripts/k3fix_build_v2.py)
PREREG = K / "PREREG_v2.md"            # the staged design; supersedes PREREG.md
PREREG_SHA = K / "PREREG_v2.sha256"
PREREG_V1 = K / "PREREG.md"            # superseded, kept unchanged
PREREG_V1_SHA = "390d79960244f27f8753acfe2c1f41135014c11d00fe7642b7841115bf5d2ef2"
T1_RUNS = ME / "fc_runs/C3r2_wide"     # Tier-1 C3r2 runs: help-zero references, report only (read-only)
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
MAX_TRIES = 2
MAX_POST_STARTS = 5
MIN_FINAL_AGE = 120.0
STARTING_GRACE_S = 180.0
ALLOWED_GPUS = (0, 1, 2, 3, 5, 6, 7)  # never GPU 4
PLACE_TIERS = ((5, 7), (3, 6))  # PREREG_v2 placement: GPUs 5 and 7 first, then the other NUMA-1 GPUs, by the fill rule;
PLACE_GPUS = tuple(g for tier in PLACE_TIERS for g in tier)  # NUMA-0 GPUs 0-2 are left to the Tier-1 fork arms (equal share)
MAX_TRAIN_PER_GPU = 3
JOB_MB = 16896
MIN_FREE_AFTER_MB = 20480
MAX_CORES = 2  # review 2026-10-22: 2 physical cores per job = the Tier-1 fork launcher's slot (t1fix CORES=2) for equal share
MIN_CORES = 2
EVAL_MIN_FREE_MB = 4096
UNKNOWN_TRAIN_AGE_S = 1800
RAMP_S = 900
SPI = 4096 * 24
HZ_MARGIN = 5000          # PREREG: K3fix help-zero iteration <= raw + 5000 (ep09, ep17)
S1_HZ_LIMIT = 30000       # PREREG_v2 stage 1 (ep07, ep18, ep03 K3fix): final.json help_zero_since <= 30000
S1_HZ_STRONG = 20000      # "strong" label only
STAGES = (1, 2, 3)
EP03_WINDOW = (12000, 15000)
EP03_MARGIN = 0.08
WINDOW_MIN_ROWS = 250     # of the 301 rows logged every 10 iterations in [12000, 15000]
EP04_SUCCESS_FAIL = 0.99  # a start bin "has success" in a 1000-it window when its mean fail EMA is < 0.99
EP04_BY = 10000
WARMUP_IT = 200
TERMINAL = ("done", "done_score_failed", "export_refused", "failed")
SH = "/root/autodl-tmp/haoran.zhang/code/fc_r2/logs/rl/sharpa"
EPF = {  # per episode: sequence, object, raw-arm facts (C3r2_tier2 final.json, own-input eval), raw hydra dir; t1_hz = the
    #      Tier-1 C3r2 run's (fc_runs/C3r2_wide) final.json help_zero_since, REPORT ONLY; changed = K3fix differs from frozen K3
    3: {"seq": "2026-04-27_11-50-10_contigo_coffee_mug_handle_table_05", "obj": "contigo_coffee_mug", "support": True,
        "raw_hz": 39062, "raw_final": 44922, "raw_hydra": f"{SH}/2026-10-16_22-22-57_ep03_contigo_coffee_mug/.hydra",
        "raw_sha": "86bb862ab9034a030a9584ac20bb5047dae6fd4290e449fb7ea8859121e29201",
        "fix_sha": "c9f6c9756357e74f25c3844a32f5c8bf9f5afa443d4478ee172647893d8e93a3", "t1_hz": 12051, "changed": True},
    4: {"seq": "2026-04-24_11-40-52_baseball_bat_rotate_05", "obj": "baseball_bat", "support": False,
        "raw_hz": 39062, "raw_final": 44922, "raw_hydra": f"{SH}/2026-10-16_10-04-16_ep04_baseball_bat/.hydra",
        "raw_sha": "42f9f807e24b80bd4bb8810d208e5da82d41aaaab47de25e00db1625b2f3c44f",
        "fix_sha": "7405d01e2af93eaf1db670f7b6cfa815cd2807155c4baccbb244cc5b7eb3f6a2", "t1_hz": 39062, "changed": True},
    7: {"seq": "2026-05-29_15-20-43_iron_round_table_02", "obj": "iron", "support": True,
        "raw_hz": 39062, "raw_final": 44922, "raw_hydra": f"{SH}/2026-10-15_22-24-05_ep07_iron/.hydra",
        "raw_sha": "f7a86a692c331d75307e200139a795023b70c855f79b6a0a0e5dd60c5d4b7e41",
        "fix_sha": "4c2d6fb3fa04439f55fc94515fce85915476cc579e368b71e0bc1b9527315ff4", "t1_hz": 12090, "changed": False},
    9: {"seq": "2026-06-04_16-45-04_big_red_bowl_under_skinny_wood_chair_03", "obj": "big_red_bowl", "support": True,
        "raw_hz": 33468, "raw_final": 39328, "raw_hydra": f"{SH}/2026-10-16_23-45-16_ep09_big_red_bowl/.hydra",
        "raw_sha": "7a85d63ffc4e5f8f03795c9e99ab483c8d2bde07e6975502305acea2f6953e6d",
        "fix_sha": "b1d62056ee50d25f8e34acd4bf7dfca7200922c1f14fe525546a8abdad49434a", "t1_hz": 26325, "changed": True},
    17: {"seq": "2026-05-01_17-04-55_traffic_cone_ground_table_02", "obj": "traffic_cone", "support": True,
         "raw_hz": 25947, "raw_final": 31807, "raw_hydra": f"{SH}/2026-10-17_04-51-57_ep17_traffic_cone/.hydra",
         "raw_sha": "b31f0a7c0689eb57fb9275cffffd1697a5eef662f9930300efe9546c10e72c7b",
         "fix_sha": "25c13883a34052b7c6bdb04357c868d975bb0bf54d4acd1b807e53fdf9e11c7a", "t1_hz": 17982, "changed": True},
    18: {"seq": "2026-05-06_10-00-19_blue_trash_can_drag_sideways_08", "obj": "blue_trash_can", "support": False,
         "raw_hz": 39062, "raw_final": 44922, "raw_hydra": f"{SH}/2026-10-15_22-34-08_ep18_blue_trash_can/.hydra",
         "raw_sha": "8407b81c034871e2ac2f0cd18495a6c834f0fd7128834c0f00c9ef23c4807bdf",
         "fix_sha": "d8d027012a2322b2591db8019ac0511eab42974632346413415e2a6e5f0df88a", "t1_hz": 13078, "changed": False},
}
JOBS = {  # launch order = dict order (stage by stage, PREREG_v2)
    "ep07_k3fix_s42": {"ep": 7, "kind": "k3fix", "seed": 42, "stage": 1},
    "ep18_k3fix_s42": {"ep": 18, "kind": "k3fix", "seed": 42, "stage": 1},
    "ep03_k3fix_s42": {"ep": 3, "kind": "k3fix", "seed": 42, "stage": 1},
    "ep03_raw_s7": {"ep": 3, "kind": "raw", "seed": 7, "stage": 1},
    "ep17_k3fix_s42": {"ep": 17, "kind": "k3fix", "seed": 42, "stage": 2},
    "ep09_k3fix_s42": {"ep": 9, "kind": "k3fix", "seed": 42, "stage": 2},
    "ep04_k3fix_s42": {"ep": 4, "kind": "k3fix", "seed": 42, "stage": 3},
}
ORDER = tuple(JOBS)
STAGE_JOBS = {s: tuple(j for j in ORDER if JOBS[j]["stage"] == s) for s in STAGES}
CFG_ALLOWED = {"k3fix": {"task.parquet", "output_dir", "logging.mode", "logging.experiment", "logging.local_jsonl"},
               "raw": {"reset.seed", "output_dir", "logging.mode", "logging.experiment", "logging.local_jsonl"}}


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


def job_of(text: str | None) -> str:
    if text not in JOBS:
        raise SystemExit(f"unknown job {text!r} (jobs: {' '.join(ORDER)})")
    return text


def jinfo(job: str) -> dict:
    j = dict(JOBS[job])
    j.update(EPF[j["ep"]])
    j["data"] = DATA_FIX if j["kind"] == "k3fix" else RAW_DATA
    j["sha"] = j["fix_sha"] if j["kind"] == "k3fix" else j["raw_sha"]
    return j


def parquet(job: str, root: Path | None = None) -> Path:
    j = jinfo(job)
    return (root or j["data"]) / "whole_body/soma" / f"sequence_id={j['seq']}" / "robot_name=g1/data.parquet"


def run_dir(job: str) -> Path:
    return RUNS / job


def raw_run(job: str) -> Path:
    return RAW_RUNS / f"ep{JOBS[job]['ep']:02d}"


def prereg_status() -> tuple[bool, str]:
    if not PREREG.is_file() or not PREREG_SHA.is_file():
        return False, f"{PREREG.name} or {PREREG_SHA.name} missing"
    words = PREREG_SHA.read_text().split()
    want = words[0] if words else ""
    got = sha256(PREREG) or ""
    return got == want, f"sha256({PREREG.name}) {got[:16]}... {'==' if got == want else '!='} {PREREG_SHA.name} {want[:16]}..."


def prereg_v1_kept() -> tuple[bool, str]:
    got = sha256(PREREG_V1) or ""
    return got == PREREG_V1_SHA, (f"superseded {PREREG_V1.name} kept unchanged: sha256 {got[:16]}... "
                                  f"{'==' if got == PREREG_V1_SHA else '!='} {PREREG_V1_SHA[:16]}...")


def parse_stages(text: str | None) -> set[int] | None:
    """None when no --stages was given; else the set of stages (empty for "")."""
    if text is None:
        return None
    out = set()
    for x in text.split(","):
        x = x.strip()
        if not x:
            continue
        if not x.isdigit() or int(x) not in STAGES:
            raise SystemExit(f"bad stage {x!r} (stages: {', '.join(map(str, STAGES))})")
        out.add(int(x))
    return out


def code_digest() -> str:
    h = hashlib.sha256()
    files = []
    for base in (CODE / "src/flash_chord", CODE / "scripts"):
        for p in base.rglob("*"):
            if p.is_file() and p.suffix in (".py", ".yaml") and "__pycache__" not in p.parts and ".cache" not in p.parts:
                files.append(p)
    for p in sorted(files):
        h.update(f"{p.relative_to(CODE)}\t{sha256(p)}\n".encode())
    return h.hexdigest()


def train_argv(job: str) -> list[str]:
    """The raw arm's command (track2_queue_tier2.sh train_cmd) with this job's data root / seed, output_dir and logging,
    plus hydra.run.dir / +logging.local_jsonl (as k3easy); neither changes training."""
    j, out = jinfo(job), run_dir(job)
    argv = ["python", "scripts/train_rl.py", f"experiment={EXP}", f"task.parquet='{parquet(job)}'", f"reset.seed={j['seed']}",
            f"output_dir={out}", f"training.max_iterations={ITERS}", "logging.mode=offline", "logging.entity=juicer_test",
            "logging.project=v2d-track2-tier2", f"logging.experiment={ARM}", f"logging.run_name=ep{j['ep']:02d}_{j['obj']}",
            "logging.upload_checkpoints=false"]
    if not j["support"]:
        argv.append("scene.support=false")
    argv += [f"hydra.run.dir={out}/hydra", "+logging.local_jsonl=true"]
    return argv


def read_pids() -> dict[str, dict]:
    out: dict[str, dict] = {}
    try:
        for line in PIDS.read_text().splitlines():
            f = line.split("\t")
            if len(f) >= 6 and f[0] in JOBS:
                out[f[0]] = {"pid": int(f[1]) if f[1].isdigit() else None, "gpu": int(f[2]) if f[2].isdigit() else None,
                             "cpus": f[3], "nth": f[4], "t": int(f[5]) if f[5].isdigit() else 0}
    except OSError:
        pass
    return out


def find_trainers(job: str) -> list[int]:
    want = f"output_dir={run_dir(job)}"
    found = []
    for pid in H.all_pids():
        cmd = H.read_cmdline(pid)
        if H.is_python_trainer(cmd) and want in cmd and H.alive(pid):
            found.append(pid)
    return found


def jsonl_last(path: Path, nbytes: int = 1 << 17) -> dict | None:
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


def progress(job: str) -> dict:
    run = run_dir(job)
    its = [int(m.group(1)) for p in run.glob("model_*.safetensors") if (m := re.fullmatch(r"model_(\d+)\.safetensors", p.name))]
    last = jsonl_last(run / "metrics.jsonl") or {}
    it = int(round(last["step"] / SPI)) if isinstance(last.get("step"), (int, float)) else None
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


def post_file(job: str) -> Path:
    return EVROOT / job / "post.json"


def load_post(job: str) -> dict:
    try:
        return json.loads(post_file(job).read_text())
    except (OSError, ValueError):
        return {}


def save_post(job: str, st: dict) -> None:
    st["updated_at"] = now()
    atomic_write(post_file(job), json.dumps(st, indent=1, default=str))


def plog(job: str, message: str) -> None:
    line = f"[{now()}] {job}: {message}"
    print(line, flush=True)
    try:
        with (K / "post.log").open("a") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ============================================================================================== GPU view / plan
def observe() -> dict[int, dict]:
    """Per GPU: trainings of anybody (classified compute apps, unknown apps older than 30 min, python train_rl.py not
    on nvidia-smi yet whose environment names the GPU), evaluations, free memory, MPS presence, and the ramp debt of
    this pilot's trainers younger than RAMP_S (as k3easy_helper.observe)."""
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
        if H.read_environ(pid).get("K3FIX") == "1" and (H.proc_age(pid) or 0.0) < RAMP_S:
            view[g]["debt"] += max(0, JOB_MB - mem.get(pid, 0))
    return view


def own_node_cores(gpu: int, exclude: str) -> list[int]:
    """Free physical cores of the GPU's NUMA node training pool (track2_queue_tier2v2_helper.cpu_slice: not held as a
    slice by any process, eval lanes excluded); [] when fewer than MIN_CORES are free (never borrows the other node)."""
    allc, _, note = H.cpu_slice(gpu, exclude, 104, MIN_CORES)
    if not allc or allc == "none" or note == "cross-NUMA":
        return []
    return [int(x) for x in allc.split(",")[0::2]]


def plan(job: str, exclude: str, remaining: int | None = None, assume: dict[int, int] | None = None,
         stages: set[int] | None = None) -> str:
    """Choose the GPU for a new training (PREREG_v2 placement). Tiers: GPUs 5 and 7 first, then GPUs 3 and 6 (all NUMA 1);
    GPUs 0-2 (NUMA 0) are left to the Tier-1 fork arms (user: Tier-2 K3 tests and Tier-1 fixes share the GPUs equally) and
    GPU 4 is never used. A GPU qualifies without MPS, with < 3 trainings of anybody, with >= 16.5 GB + 20 GB free (+ ramp
    debt of this pilot's young trainers) and >= MIN_CORES free physical cores on its own NUMA node (never borrowed from
    the other node); among the qualifying GPUs of the first tier that has one: the fewest trainings, then the most free
    memory, then the lowest index. CPU slice: ceil(free own-node cores / launches still to come in the requested
    stage(s)) clamped to MIN_CORES..MAX_CORES = 2..2, i.e. 2 physical cores (+ HT siblings) per job: the slot size the
    Tier-1 fork launcher (scripts/t1fix_launch.py, CORES=2) counts in its equal-share rule, whichever launcher goes first.
    assume {gpu: n} (dry run only): n hypothetical extra trainings of this pilot on that GPU, each still owing JOB_MB."""
    try:
        view = observe()
    except Exception as error:  # noqa: BLE001
        return f"WAIT nvidia-smi failed ({type(error).__name__}: {str(error)[:100]})"
    if remaining is None:
        want = stages or {JOBS[job]["stage"]}
        remaining = sum(1 for j in ORDER if JOBS[j]["stage"] in want and job_state(j)[0] == "FRESH") or 1
    why, cands = [], []
    node_cores: dict[int, list[int]] = {}
    for g in ALLOWED_GPUS:
        if g not in PLACE_GPUS:
            why.append(f"gpu{g} NUMA0 (left to the Tier-1 forks)")
            continue
        v = view.get(g)
        if v is None:
            why.append(f"gpu{g} not listed")
            continue
        extra = (assume or {}).get(g, 0)
        n = len(v["trainings"]) + extra
        need = JOB_MB + MIN_FREE_AFTER_MB + v["debt"] + extra * JOB_MB
        if v["mps"]:
            why.append(f"gpu{g} MPS")
            continue
        if n + 1 > MAX_TRAIN_PER_GPU:
            why.append(f"gpu{g} {n} trainings")
            continue
        if v["free"] < need:
            why.append(f"gpu{g} {v['free']}<{need} MiB")
            continue
        node = 0 if g <= 2 else 1
        if node not in node_cores:
            node_cores[node] = own_node_cores(g, exclude)
        if len(node_cores[node]) < MIN_CORES:
            why.append(f"gpu{g} node{node} < {MIN_CORES} free cores")
            continue
        tier = next(i for i, t in enumerate(PLACE_TIERS) if g in t)
        cands.append((tier, n, -(v["free"] - extra * JOB_MB), g))
    if not cands:
        return "WAIT no GPU fits: " + "; ".join(why)
    tier, n, negfree, g = sorted(cands)[0]
    extra = (assume or {}).get(g, 0)
    phys = node_cores[0 if g <= 2 else 1]
    cores = phys[:min(MAX_CORES, max(MIN_CORES, math.ceil(len(phys) / max(1, remaining))))]
    cpus, nth = ",".join(f"{c},{c + 104}" for c in cores), 2 * len(cores)
    v = view[g]
    return (f"OK {g} {cpus} {nth} - | gpu{g} (placement tier {tier + 1} = GPUs {'/'.join(map(str, PLACE_TIERS[tier]))}): {n} training(s) now "
            f"({' '.join(map(str, sorted(v['trainings']))) or 'none'}{f' + {extra} assumed' if extra else ''}) -> {n + 1} after the start "
            f"(cap {MAX_TRAIN_PER_GPU}); {v['free'] - extra * JOB_MB} MiB free >= {JOB_MB + MIN_FREE_AFTER_MB + v['debt']}; "
            f"no MPS; CPU: {len(cores)} of {len(phys)} free own-node cores for {remaining} launch(es) to come; "
            f"other candidates {[c[3] for c in sorted(cands)[1:]]}; excluded: {'; '.join(why) or '-'}")


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
                   f"{'  (never used: GPU 4 is for tests)' if g not in ALLOWED_GPUS else ''}")
    return out


# ============================================================================================== check / cfgcheck
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


def ready_table() -> dict[str, str]:
    """{relative parquet path: sha256} from the K3fix root's READY marker."""
    out = {}
    try:
        lines = (DATA_FIX / "READY").read_text().splitlines()
    except OSError:
        return out
    if not lines or not lines[0].startswith("READY"):
        return out
    for line in lines[1:]:
        m = re.fullmatch(r"([0-9a-f]{64})  (\S+)", line.strip())
        if m:
            out[m.group(2)] = m.group(1)
    return out


def check() -> int:
    rows: list[tuple[str, bool, str]] = []

    def add(name, ok, detail):
        rows.append((name, bool(ok), detail))

    ok, why = prereg_status()
    add("prereg_v2", ok, why)
    ok, why = prereg_v1_kept()
    add("prereg_v1_kept", ok, why)
    want_stages = {1: ["ep07_k3fix_s42", "ep18_k3fix_s42", "ep03_k3fix_s42", "ep03_raw_s7"], 2: ["ep17_k3fix_s42", "ep09_k3fix_s42"],
                   3: ["ep04_k3fix_s42"]}
    add("stage_table", {s: list(v) for s, v in STAGE_JOBS.items()} == want_stages,
        "stage 1: ep07 ep18 ep03-K3fix ep03-raw-s7 | stage 2: ep17 ep09 | stage 3: ep04 (PREREG_v2 launch order)")
    add("placement_rule", 4 not in ALLOWED_GPUS and 4 not in PLACE_GPUS and not set(PLACE_GPUS) & {0, 1, 2}
        and set(PLACE_GPUS) <= set(ALLOWED_GPUS), f"placement tiers {PLACE_TIERS}; never GPU 4; NUMA-0 GPUs 0-2 left to the Tier-1 forks")
    try:
        head = subprocess.run(["git", "-C", str(CODE), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        head = ""
    add("code_pin", head == PIN, f"{CODE} HEAD {head[:12] or '?'} (pin {PIN[:7]})")
    try:
        build = json.loads((K / "build.json").read_text())
    except (OSError, ValueError):
        build = {}
    dig = code_digest()
    add("code_tree_digest", build.get("code_digest") == dig, f"src/flash_chord + scripts digest {dig[:16]}... vs build.json")
    text = (CODE / "src/flash_chord/configs/experiment" / f"{EXP}.yaml").read_text()
    for key, val in (("deadline", 35156), ("zero_ramp", 3906), ("early_stop_help_zero_tail", 5860), ("max_iterations", 45000)):
        add(f"experiment_{key}", re.search(rf"^\s+{key}: {val}\s*$", text, re.M), f"{EXP}.yaml {key}: {val}")
    table = ready_table()
    add("k3fix_root_ready", len(table) == 30, f"{DATA_FIX}/READY lists {len(table)} parquet sha256 (30 expected)")
    bad = [p for p, s in table.items() if sha256(DATA_FIX / p) != s]
    add("k3fix_root_sha_table", not bad and len(table) == 30, f"every parquet matches READY ({len(bad)} mismatch)")
    for name, key in (("tracks.json", "all_ok"), ("data_build.json", "all_ok"), ("loader_check.json", None), ("gates.json", "replication_ok")):
        try:
            d = json.loads((BUILD / name).read_text())
            okk = d["summary"]["all_ok"] if key is None else d.get(key)
        except (OSError, ValueError, KeyError):
            d, okk = {}, None
        add(f"build_{name.split('.')[0]}", okk is True, f"build/{name}: {okk}")
    try:
        gates = json.loads((BUILD / "gates.json").read_text())
        tr = json.loads((BUILD / "tracks.json").read_text())
    except (OSError, ValueError):
        gates, tr = {}, {}
    add("gates_tracks_same_input", gates.get("k3fix_tracks_npz_sha256") == tr.get("k3fix_tracks_npz_sha256")
        == sha256(BUILD / "k3fix_tracks.npz"), "gates.json computed on the current k3fix_tracks.npz")
    try:
        gv2 = json.loads(GATES_V2.read_text())
    except (OSError, ValueError):
        gv2 = {}
    add("build_gates_v2", gv2.get("replication_ok") is True and gv2.get("identity_ok") is True
        and gv2.get("k3fix_tracks_npz_sha256") == sha256(BUILD / "k3fix_tracks.npz")
        and gv2.get("gate_code_sha256") == sha256(ME / "scripts/k3fix_build.py"),
        f"build/gates_v2.json (ep07, ep18): replication {gv2.get('replication_ok')}, identity {gv2.get('identity_ok')}, "
        f"same k3fix_tracks.npz and gate code as gates.json")
    try:
        man = json.loads((DATA_FIX / "whole_body/soma/track2_manifest.json").read_text())
        seq_of = {int(r["episode_index"]): str(r.get("sequence_id", s)) for s, r in man.items()}
    except (OSError, ValueError, KeyError):
        seq_of = {}
    add("manifest_equals_raw", sha256(DATA_FIX / "whole_body/soma/track2_manifest.json")
        == sha256(RAW_DATA / "whole_body/soma/track2_manifest.json"), "K3fix root manifest = Tier-2 manifest")
    try:
        cfc = json.loads(CFGCHECK.read_text())
    except (OSError, ValueError):
        cfc = {}
    add("cfgcheck", cfc.get("all_ok") is True and sorted(cfc.get("jobs", {})) == sorted(ORDER), f"{CFGCHECK.name}: all_ok {cfc.get('all_ok')}")
    try:
        db = json.loads((BUILD / "data_build.json").read_text())["episodes"]
    except (OSError, ValueError, KeyError):
        db = {}
    for job in ORDER:
        j = jinfo(job)
        ep, tag = j["ep"], job
        add(f"{tag}_manifest", seq_of.get(ep) == j["seq"], f"manifest ep{ep} -> {seq_of.get(ep)}")
        p = parquet(job)
        badp = H.check_parquet(p, j["seq"]) if p.is_file() else "missing"
        add(f"{tag}_parquet_valid", badp is None, f"{p}: {badp or '1 row, columns, sequence id, asset files ok'}")
        digest = sha256(p)
        if j["kind"] == "k3fix":
            rel = p.relative_to(DATA_FIX).as_posix()
            add(f"{tag}_parquet_sha256", digest == j["sha"] == table.get(rel) == db.get(f"{ep:02d}", {}).get("root_sha256"),
                f"{str(digest)[:16]}... = pinned = READY = data_build.json")
            if j["changed"]:
                add(f"{tag}_not_raw_not_k3", digest != sha256(parquet(job, RAW_DATA)) and digest != sha256(parquet(job, K3_DATA)),
                    "K3fix-changed clip: differs from the raw Tier-2 and the K3 parquet")
                gsrc, gname = gates, "gates.json"
            else:
                add(f"{tag}_equals_k3_not_raw", digest == sha256(parquet(job, K3_DATA)) and digest != sha256(parquet(job, RAW_DATA)),
                    "K3fix leaves this clip unchanged: byte-identical to the K3 root's parquet, differs from the raw Tier-2 parquet")
                gsrc, gname = gv2, "gates_v2.json"
            g = gsrc.get("clips", {}).get(f"{ep:02d}", {}).get("K3fix", {})
            add(f"{tag}_gates", g.get("all_pass") is True and ep in gsrc.get("pass_clips", []),
                f"build/{gname}: G1-G4 all pass for ep{ep:02d} = {g.get('all_pass')}")
            fr = FRAME_REFS_FIX / f"episode_{ep:06d}.npz"
            add(f"{tag}_frame_refs", fr.is_file() and sha256(fr) == db.get(f"{ep:02d}", {}).get("frame_ref_sha256"),
                f"{fr.name} = data_build.json")
        else:
            add(f"{tag}_parquet_sha256", digest == j["sha"], f"raw Tier-2 parquet {str(digest)[:16]}... = pinned")
        sdir, rdir = p.parent.parent, parquet(job, RAW_DATA).parent.parent
        add(f"{tag}_ground_plane", sha256(sdir / "ground_plane.json") == sha256(rdir / "ground_plane.json"), "ground_plane.json = raw")
        name = f"{j['seq']}_support.usda"
        mine, ref = j["data"] / "whole_body/reconstructed_stage" / name, RAW_DATA / "whole_body/reconstructed_stage" / name
        add(f"{tag}_support_file", mine.is_file() and sha256(mine) == sha256(ref), f"reconstructed_stage/{name} = raw")
        rule = H.support_rule(j["data"], j["seq"])
        add(f"{tag}_support_rule", rule == ("true" if j["support"] else "false") == H.support_rule(RAW_DATA, j["seq"]),
            f"support rule {rule} (scene.support={'default true' if j['support'] else 'false'}) as the raw arm")
        try:
            raw_ov = [str(x) for x in _load_yaml_list(Path(j["raw_hydra"]) / "overrides.yaml")]
        except Exception:  # noqa: BLE001
            raw_ov = []
        add(f"{tag}_raw_hydra", f"output_dir={raw_run(job)}" in raw_ov and ("scene.support=false" in raw_ov) == (not j["support"])
            and "reset.seed=42" in raw_ov, f"{j['raw_hydra']}: output_dir {raw_run(job)}, seed 42, support flag")
        fin = H.final_marker(raw_run(job)) or {}
        add(f"{tag}_raw_reference", fin.get("help_zero_since") == j["raw_hz"] and fin.get("iteration") == j["raw_final"],
            f"raw final.json help_zero_since {fin.get('help_zero_since')} iteration {fin.get('iteration')} (pinned {j['raw_hz']} / {j['raw_final']})")
        add(f"{tag}_raw_wandb", bool(glob.glob(str(raw_run(job) / "wandb/run-*/run-*.wandb"))), "raw run's offline W&B record present")
        t1 = H.final_marker(T1_RUNS / f"ep{ep:02d}") or {}
        add(f"{tag}_t1_reference", t1.get("help_zero_since") == j["t1_hz"],
            f"Tier-1 C3r2 final.json help_zero_since {t1.get('help_zero_since')} (pinned {j['t1_hz']}; report only)")
    bad = [r for r in rows if not r[1]]
    for name, okk, detail in rows:
        print(f"CHECK {name} {'ok' if okk else 'FAIL'} {detail}")
    print(f"CHECKS {len(rows) - len(bad)}/{len(rows)} ok" + (f"; FAILED: {' '.join(r[0] for r in bad)}" if bad else ""))
    return 0 if not bad else 1


def cfgcheck(write: bool) -> int:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    res = {"written": now(), "code": str(CODE), "pin": PIN, "allowed_differences": {k: sorted(v) for k, v in CFG_ALLOWED.items()},
           "jobs": {}}
    ok_all = True
    with initialize_config_dir(config_dir=str(CODE / "src/flash_chord/configs"), version_base="1.3"):
        for job in ORDER:
            j = jinfo(job)
            hyd = Path(j["raw_hydra"])
            raw_ov = [str(x) for x in _load_yaml_list(hyd / "overrides.yaml")]
            ov = [a for a in train_argv(job)[2:] if not a.startswith("hydra.")]
            raw_saved = _flat(OmegaConf.to_container(OmegaConf.load(hyd / "config.yaml"), resolve=False))
            raw_comp = _flat(OmegaConf.to_container(compose(config_name="train", overrides=raw_ov), resolve=False))
            comp = _flat(OmegaConf.to_container(compose(config_name="train", overrides=ov), resolve=False))
            repro = raw_comp == raw_saved
            keys = sorted(set(raw_saved) | set(comp))
            diff = [{"key": k, "raw": raw_saved.get(k, "<absent>"), "job": comp.get(k, "<absent>")} for k in keys
                    if raw_saved.get(k, "<absent>") != comp.get(k, "<absent>")]
            dkeys = {d["key"] for d in diff}
            ok = (repro and dkeys <= CFG_ALLOWED[j["kind"]] and comp.get("task.parquet") == str(parquet(job))
                  and comp.get("output_dir") == str(run_dir(job)) and comp.get("logging.mode") == "offline"
                  and comp.get("reset.seed") == j["seed"])
            if j["kind"] == "raw":
                ok &= raw_saved.get("task.parquet") == str(parquet(job)) and "task.parquet" not in dkeys
            ok_all &= ok
            udiff = list(difflib.unified_diff(raw_ov, train_argv(job)[2:], "raw C3r2_tier2 overrides", f"{job} argv", lineterm="", n=20))
            res["jobs"][job] = {"ok": ok, "raw_hydra_dir": str(hyd), "compose_reproduces_raw_config_yaml": repro,
                                "n_config_keys": len(raw_saved), "config_diff_vs_raw": diff, "overrides_raw": raw_ov,
                                "argv": train_argv(job)[2:], "overrides_unified_diff": udiff}
            print(f"== {job}: compose(raw overrides) == raw run's saved config.yaml: {repro} ({len(raw_saved)} keys); "
                  f"job vs raw differ in {len(diff)} key(s), all allowed: {dkeys <= CFG_ALLOWED[j['kind']]} -> {'ok' if ok else 'FAIL'}")
            for d in diff:
                print(f"   {d['key']}: {d['raw']}  ->  {d['job']}")
    res["all_ok"] = bool(ok_all)
    print(f"cfgcheck: {'OK' if ok_all else 'FAILED'}")
    if write:
        atomic_write(CFGCHECK, json.dumps(res, indent=1, default=str))
        print(f"wrote {CFGCHECK}")
    return 0 if ok_all else 1


# ============================================================================================== state / crash
def job_state(job: str) -> tuple[str, str]:
    run = run_dir(job)
    tr = find_trainers(job)
    pr = progress(job)
    launched = read_pids().get(job)
    gpu = launched["gpu"] if launched else None
    pinfo = f"it={pr['it'] if pr['it'] is not None else pr['last_model']} help_zero_iters={pr['help_zero_iters']}"
    if tr:
        return "RUNNING", f"pid={','.join(map(str, tr))} {pinfo} gpu={gpu}"
    why = H.done_reason(run, ITERS)
    if why:
        st = load_post(job)
        if lock_held(K / "post" / f"{job}.lock"):
            return "POST_RUNNING", f"{why}; post {st.get('status', '?')}"
        if st.get("status") in TERMINAL:
            return "POST_DONE", f"{why}; post {st['status']}"
        if st.get("status") == "retry" and float(st.get("retry_after", 0)) > time.time():
            return "POST_RETRY_WAIT", f"{why}; post retry after {time.strftime('%H:%M:%S', time.localtime(float(st['retry_after'])))}"
        return "TRAINED", f"{why}; post {st.get('status', 'not started')}"
    if (K / f"CRASHED_{job}").exists():
        return "CRASHED", f"marker {K / f'CRASHED_{job}'}; {pinfo}"
    has_files = pr["models"] > 0 or (run / "final.json").exists() or any(run.glob("rsi_*.safetensors"))
    if launched and time.time() - launched["t"] < STARTING_GRACE_S and not has_files:
        return "STARTING", f"launched {int(time.time() - launched['t'])} s ago (pid {launched['pid']}, gpu {gpu}), trainer not visible yet"
    if has_files or launched:
        return "DEAD", (f"no live trainer, no final.json; {pr['models']} model_* file(s), last {pr['last_model']}, "
                        f"{'launched by this pilot at ' + time.strftime('%F %T', time.localtime(launched['t'])) if launched else 'never launched by this pilot'}")
    return "FRESH", f"no run files in {run}"


def state(stages: set[int] | None = None) -> int:
    for job in ORDER:
        s, d = job_state(job)
        if s == "FRESH" and stages is not None and JOBS[job]["stage"] not in stages:
            stg = JOBS[job]["stage"]
            s, d = "DEFERRED", f"stage {stg} not requested: never launched by this driver (only by an explicit 'run --stage {stg}'); {d}"
        print(f"JOB {job} {s} {d}")
    return 0


def stage_guard(stages: set[int]) -> int:
    """A stage may be launched only after every earlier stage has been launched (or together with it)."""
    if not stages:
        print("REFUSE no stage requested")
        return 1
    low = min(stages)
    bad = [j for j in ORDER if JOBS[j]["stage"] < low and JOBS[j]["stage"] not in stages and job_state(j)[0] == "FRESH"]
    if bad:
        print(f"REFUSE stage(s) {','.join(map(str, sorted(stages)))} requested, but earlier-stage job(s) {' '.join(bad)} were never "
              f"launched (launch that stage first, or include it)")
        return 1
    print(f"OK stage(s) {','.join(map(str, sorted(stages)))}: {' '.join(j for j in ORDER if JOBS[j]['stage'] in stages)}"
          f"{'; every earlier stage launched' if low > 1 else ''}")
    return 0


def crash(job: str, why: str) -> int:
    path = K / f"CRASHED_{job}"
    if path.exists():
        return 0
    log = RUNS / "logs" / f"{job}.log"
    try:
        tail = log.read_text(errors="replace").splitlines()[-30:]
    except OSError:
        tail = []
    pr = progress(job)
    text = (f"CRASHED {job} detected {now()}\n{why}\nlast model {pr['last_model']}, last logged iteration {pr['it']}, "
            f"help_zero_iterations {pr['help_zero_iters']}\nNo automatic relaunch (PREREG): the job is INCOMPLETE unless the "
            f"user decides a relaunch (resume from the newest finite checkpoint), which is then disclosed.\n"
            f"--- last lines of {log}\n" + "\n".join(tail) + "\n")
    atomic_write(path, text)
    print(f"wrote {path}")
    return 0


# ============================================================================================== post-training step
def child_env(gpu: int, job: str) -> dict:
    sp = ME / "conda_envs/flashchord/lib/python3.11/site-packages/nvidia"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CUDA_MPS_", "WANDB_", "RUNIN_", "TIER2V2_", "K3EASY", "K3FIX"))
           and k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "TIER2_GPU")}
    env.update({
        "PATH": f"{ME}/conda_envs/flashchord/bin:{env.get('PATH', '/usr/bin:/bin')}",
        "LD_LIBRARY_PATH": f"{sp}/cu13/lib:{sp}/cudnn/lib:{sp}/nccl/lib:{sp}/nvshmem/lib"
                           + (f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else ""),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": f"{ME}/.cache/pycache", "XDG_CACHE_HOME": f"{ME}/.cache", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(gpu), "HUMAN_MOTION_DATA_DIR": str(jinfo(job)["data"]), "OMP_NUM_THREADS": "4",
        "K3FIX_EVAL": "1", "K3FIX_EVAL_JOB": job,
    })
    return env


def eval_command(job: str, gpu: int, w: Path) -> list[str]:
    """k3easy_helper.eval_command (= track2_tier2v2_post.py run_eval's command) with this job's work dir, data root
    manifest and the 45000 fallback iteration."""
    j = jinfo(job)
    command = [str(FCPY), "-B", str(CODE / "scripts/track2/eval_all.py"), "--arm-dir", str(w), "--arm", ARM,
               "--checkpoint-from-final", "--iteration", str(ITERS), "--episodes", str(j["ep"]), "--gpus", str(gpu),
               "--jobs-per-gpu", "1", "--world-count", "1", "--export", "yes", "--export-jobs", "1",
               "--export-tolerance", repr(EXPORT_TOL), "--skip-pack", "--kit-root", str(KIT),
               "--stock-runtime", str(STOCK), "--dataset-root", str(DATASET),
               "--manifest", f"{j['data']}/whole_body/soma/track2_manifest.json", "--pack-python", str(KPY),
               "--results", str(w / "results"), "--code", str(CODE), "--python", str(FCPY)]  # fmt: skip
    return ["nice", "-n", "10", "taskset", "-c", H.EVAL_CORES[0 if gpu <= 2 else 1]] + command


def score_command(job: str, w: Path) -> list[str]:
    """The LOCAL clean diagnostic: the kit packer (unchanged) via track2_tier2_clean_scores.py. K3fix jobs pass the K3fix
    input's own frame references; the raw control passes none (the default released raw reference, as the raw 0.570
    baseline in tmp_analysis/tier2/early_scores was scored)."""
    cmd = ["nice", "-n", "10", str(KPY), "-I", "-B", str(CLEAN_SCORER), "--kit", str(KIT),
           "--out", str(w / "clean_scores_LOCAL_NOT_FOR_UPLOAD.csv"), "--tier1", str(TIER1_CSV), "--",
           "--track", "track_2_tier2", "--results", str(w / "results"), "--dataset-root", str(DATASET),
           "--code-commit", COMMIT, "--eval-script", str(KIT / "flash_chord/scripts/evaluate_policy.py")]  # fmt: skip
    if jinfo(job)["kind"] == "k3fix":
        cmd += ["--frame-references", str(FRAME_REFS_FIX)]
    return cmd + ["--config", str(w / "pack" / "tracks.json"), "-o", str(w / "packed_LOCAL_NOT_FOR_UPLOAD.csv")]


def acquire_lane(job: str, gpu: int) -> list[int]:
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
                why = f"gpu{gpu}: another k3fix evaluation holds the lane"
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
                    plog(job, f"eval lane gpu{gpu} taken (k3fix lock{' + ' + str(theirs) if len(fds) > 1 else '; no rolling-start lock file for this GPU'})")
                    return fds
        if why != last:
            plog(job, f"eval waits: {why}")
            last = why
        time.sleep(30)


def run_eval(job: str, gpu: int, w: Path, fds: list[int]) -> tuple[int, str]:
    link = w / f"ep{JOBS[job]['ep']:02d}"
    if not link.is_symlink():
        link.symlink_to(run_dir(job))
    (w / "eval_report.json").unlink(missing_ok=True)
    cmd = eval_command(job, gpu, w)
    note = ""
    with (w / "eval_all.log").open("a") as stream:
        stream.write(f"[{now()}] gpu {gpu}: {' '.join(cmd)}\n")
        stream.flush()
        proc = subprocess.Popen(cmd, cwd=CODE, env=child_env(gpu, job), stdout=stream, stderr=subprocess.STDOUT,
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


def parse_eval(job: str, w: Path) -> dict:
    ep = JOBS[job]["ep"]
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
                    "eval_reference_is_train_input": bool(ref) and Path(ref).resolve() == parquet(job).resolve()
                    and sha256(ref) == jinfo(job)["sha"]})
    except (OSError, ValueError):
        pass
    return out


def score(job: str, w: Path) -> dict:
    """The LOCAL clean diagnostic (never selects a checkpoint, never decides). Clean numbers only in
    clean_LOCAL_NOT_FOR_UPLOAD.json; the returned dict carries status and residuals."""
    j = jinfo(job)
    ep = j["ep"]
    if j["kind"] == "k3fix" and not (FRAME_REFS_FIX / f"episode_{ep:06d}.npz").is_file():
        return {"score_status": "no_frame_refs"}
    pack = w / "pack"
    pack.mkdir(exist_ok=True)
    cfg = json.loads((KIT / "config/tracks.json").read_text())
    cfg["tracks"]["track_2_tier2"]["scored_episodes"] = [ep]
    (pack / "tracks.json").write_text(json.dumps(cfg, indent=2))
    out_csv = w / "clean_scores_LOCAL_NOT_FOR_UPLOAD.csv"
    log_path = w / "clean_score_LOCAL_NOT_FOR_UPLOAD.log"
    out_csv.unlink(missing_ok=True)
    cmd = score_command(job, w)
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
    clean = {"job": job, "episode": ep, "object": j["obj"], "written": now(),
             "frame_refs": str(FRAME_REFS_FIX) if j["kind"] == "k3fix" else "default (released raw Tier-2 input, as the 0.570 baseline)",
             "note": "LOCAL clean-reference diagnostic, arm-level and disclosed; never selects a checkpoint (final.json's is fixed) "
                     "and never enters the PREREG verdict"}
    if code == 0 and rows:
        r = rows[0]
        clean.update({"clean_add_auc": fnum(r.get("t2clean_add_auc")), "clean_sp_sr": fnum(r.get("t2clean_spider_sr")),
                      "clean_mp_sr": fnum(r.get("t2clean_maniptrans_sr")), "clean_mppe_cm": fnum(r.get("t2clean_mppe_cm"))})
        out["score_status"] = "ok"
    else:
        out["score_status"] = "failed"
        out["score_note"] = f"clean scorer exit {code}; details only in {log_path.name}"
        clean["score_failure_tail"] = " | ".join([ln for ln in text.splitlines() if ln.strip()][-3:])[:400]
    atomic_write(w / "clean_LOCAL_NOT_FOR_UPLOAD.json", json.dumps(clean, indent=1))
    return out


def post(job: str) -> int:
    launched = read_pids().get(job)
    gpu = launched["gpu"] if launched else None
    if gpu is None or gpu not in ALLOWED_GPUS:
        print(f"{job}: no training GPU recorded in {PIDS} (or not allowed): nothing done")
        return 2
    (K / "post").mkdir(parents=True, exist_ok=True)
    lock = try_lock(K / "post" / f"{job}.lock", create=True)
    if lock is None:
        print(f"{job}: another post holds its lock; nothing to do")
        return 0
    w = EVROOT / job
    w.mkdir(parents=True, exist_ok=True)
    st = load_post(job) or {"job": job, "episode": JOBS[job]["ep"], "status": "pending", "tries": 0}
    if st.get("status") in TERMINAL:
        print(f"{job}: post already {st['status']}")
        return 0
    st["post_starts"] = int(st.get("post_starts", 0)) + 1
    if st["post_starts"] > MAX_POST_STARTS:
        st.update({"status": "failed", "finished_at": now(), "note": f"post started {st['post_starts']} times without finishing"})
        save_post(job, st)
        plog(job, st["note"])
        return 1
    if st.get("status") == "retry":
        wait = float(st.get("retry_after", 0)) - time.time()
        if wait > 0:
            time.sleep(wait)
    run = run_dir(job)
    why, fin = H.done_reason(run, ITERS), H.final_marker(run)
    if why is None or fin is None:
        plog(job, f"training not finished ({run}): nothing to do")
        save_post(job, st)
        return 3
    if find_trainers(job):
        plog(job, "a trainer of this run dir is still alive: not evaluating yet")
        save_post(job, st)
        return 3
    try:
        age = time.time() - (run / fin["checkpoint"]).stat().st_mtime
    except OSError:
        age = MIN_FINAL_AGE
    if age < MIN_FINAL_AGE:
        time.sleep(MIN_FINAL_AGE - age)
    psha = sha256(parquet(job))
    st.update({"final_iteration": fin["iteration"], "final_reason": fin.get("reason"), "help_zero_since": fin.get("help_zero_since"),
               "train_gpu": gpu, "parquet": str(parquet(job)), "parquet_sha256": psha, "eval_command": " ".join(eval_command(job, gpu, w))})
    if psha != jinfo(job)["sha"]:
        st.update({"status": "failed", "finished_at": now(), "note": f"the input parquet changed since the build ({psha})"})
        save_post(job, st)
        plog(job, st["note"])
        return 1
    st["status"] = "waiting_gpu"
    save_post(job, st)
    plog(job, f"final.json: {why}; evaluating model_{fin['iteration']} against its own input (export tol {EXPORT_TOL:g}) on gpu{gpu}")
    fds = acquire_lane(job, gpu)
    st.update({"status": "evaluating", "eval_gpu": gpu, "eval_started_at": now()})
    save_post(job, st)
    t0 = time.time()
    try:
        code, note = run_eval(job, gpu, w, fds)
    finally:
        for fd in fds:
            os.close(fd)
    res = parse_eval(job, w)
    st.update(res)
    st.update({"eval_exit": code, "eval_seconds": round(time.time() - t0, 1), "eval_note": note})
    tries = int(st.get("tries", 0)) + 1
    if res.get("eval_status") == "evaluated" and code == 0:
        plog(job, f"evaluated on gpu{gpu} against its own input: SP-SR {res.get('eval_sp_sr')} MP-SR {res.get('eval_mp_sr')} "
                  f"world_count {res.get('eval_world_count')} AUC(own input) {res.get('eval_add_auc_own_input')} "
                  f"export max_rel_diff {res.get('export_max_rel_diff')}; reference is the training parquet: {res.get('eval_reference_is_train_input')}")
        sc = score(job, w)
        st.update(sc)
        st["status"] = "done" if sc.get("score_status") == "ok" else "done_score_failed"
        st["finished_at"] = now()
        plog(job, f"local clean diagnostic: {sc.get('score_status')} (numbers only in {w}/clean_LOCAL_NOT_FOR_UPLOAD.json)")
    elif str(res.get("eval_status", "")).startswith("export_failed") and res.get("export_max_rel_diff") is not None \
            and res["export_max_rel_diff"] > EXPORT_TOL:
        st.update({"status": "export_refused", "finished_at": now(),
                   "note": f"export refused at tolerance {EXPORT_TOL:g}: max_rel_diff {res['export_max_rel_diff']:.3g}"})
        plog(job, st["note"])
    elif tries < MAX_TRIES:
        st.update({"status": "retry", "retry_after": time.time() + RETRY_DELAY,
                   "note": f"technical failure (exit {code}, {res.get('eval_status')}{', ' + note if note else ''}); retry in {RETRY_DELAY:.0f} s"})
        plog(job, st["note"])
    else:
        st.update({"status": "failed", "finished_at": now(),
                   "note": f"evaluation failed {tries} times (exit {code}, {res.get('eval_status')}); see {w / 'eval_all.log'}"})
        plog(job, st["note"])
    st["tries"] = tries
    save_post(job, st)
    os.close(lock)
    return 0


# ============================================================================================== curves (training logs)
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


def _compact(rows: list[dict]) -> list[list]:
    """[iteration, rsi/fail_rate_mean, [per-bin fail EMA...]] per logged row (rows without the metric dropped)."""
    out = []
    for r in rows:
        f = r.get("rsi/fail_rate_mean")
        s = r.get("step", r.get("_step"))
        if not isinstance(f, (int, float)) or not isinstance(s, (int, float)) or not math.isfinite(f):
            continue
        bins, b = [], 0
        while f"rsi/bin/fail_{b:02d}" in r:
            v = r[f"rsi/bin/fail_{b:02d}"]
            bins.append(float(v) if isinstance(v, (int, float)) else float("nan"))
            b += 1
        out.append([int(round(s / SPI)), float(f), bins])
    return out


_RAW_MEMO: dict[int, tuple[list, str]] = {}


def curve_raw(ep: int, write: bool) -> tuple[list, str]:
    if ep not in _RAW_MEMO:
        _RAW_MEMO[ep] = _curve_raw(ep, write)
    return _RAW_MEMO[ep]


def _curve_raw(ep: int, write: bool) -> tuple[list, str]:
    """The raw C3r2_tier2 seed-42 run's training curve from its offline W&B record (cached in k3fix_pilot/raw_curves/)."""
    cache = K / "raw_curves" / f"ep{ep:02d}.json"
    try:
        return json.loads(cache.read_text()), ""
    except (OSError, ValueError):
        pass
    files = sorted(glob.glob(str(RAW_RUNS / f"ep{ep:02d}/wandb/run-*/run-*.wandb")))
    if not files:
        return [], "no raw W&B record"
    try:
        rows = _compact(_rows_wandb(files[-1]))
    except Exception as error:  # noqa: BLE001
        return [], f"raw curve unreadable ({type(error).__name__}: {str(error)[:80]})"
    if write and rows:
        atomic_write(cache, json.dumps(rows))
    return rows, ""


def curve_jsonl(path: Path) -> list:
    if not path.is_file():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return _compact(rows)


def window_mean(rows: list, lo: int, hi: int) -> tuple[float | None, int]:
    vals = [r[1] for r in rows if lo <= r[0] <= hi]
    return (sum(vals) / len(vals) if len(vals) >= WINDOW_MIN_ROWS else None), len(vals)


def bins_1000(rows: list) -> dict[int, float]:
    acc: dict[int, list] = {}
    for it, f, _ in rows:
        b = int(round(it / 1000.0)) * 1000
        a = acc.setdefault(b, [0.0, 0])
        a[0] += f
        a[1] += 1
    return {b: a[0] / a[1] for b, a in acc.items()}


def success_bins(rows: list) -> dict[int, int]:
    """{window start k*1000: number of start bins whose mean fail EMA over the window's rows (iteration >= 200) < 0.99}."""
    acc: dict[int, list] = {}
    for it, _, bins in rows:
        if it < WARMUP_IT or not bins:
            continue
        k = (it // 1000) * 1000
        a = acc.setdefault(k, [[0.0] * len(bins), 0])
        if len(a[0]) != len(bins):
            continue
        a[0] = [x + (y if math.isfinite(y) else 1.0) for x, y in zip(a[0], bins)]
        a[1] += 1
    return {k: sum(1 for s in a[0] if s / a[1] < EP04_SUCCESS_FAIL) for k, a in sorted(acc.items()) if a[1]}


def first_help_zero(job: str) -> int | None:
    path = run_dir(job) / "metrics.jsonl"
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


# ============================================================================================== verdict
def own_input_ok(job: str, st: dict, fin: dict | None) -> bool | None:
    """(b): ONE official own-input evaluation of final.json's checkpoint: world_count 1, SP-SR = MP-SR = 1, the evaluated
    iteration is final.json's, the reference is the training parquet with the pinned sha256. None while unknown."""
    if st.get("status") in ("done", "done_score_failed"):
        return bool(st.get("eval_status") == "evaluated" and st.get("eval_world_count") == 1 and st.get("eval_sp_sr") == 1.0
                    and st.get("eval_mp_sr") == 1.0 and st.get("eval_reference_is_train_input") is True
                    and fin is not None and st.get("evaluated_iteration") == fin["iteration"])
    if st.get("status") == "export_refused":
        return False
    return None


def job_summary(job: str) -> dict:
    j = jinfo(job)
    run = run_dir(job)
    fin = H.final_marker(run) if H.done_reason(run, ITERS) else None
    st = load_post(job)
    crashed = (K / f"CRASHED_{job}").exists()
    pid = read_pids().get(job)
    launched = bool(pid) or (run / "metrics.jsonl").exists() or (run / "final.json").exists() or any(run.glob("model_*.safetensors"))
    s = {"job": job, "stage": j["stage"], "episode": j["ep"], "object": j["obj"], "kind": j["kind"], "seed": j["seed"],
         "launched": launched, "train_gpu": (pid or {}).get("gpu"),
         "crash_marker": crashed, "final_iteration": fin["iteration"] if fin else None, "final_reason": fin.get("reason") if fin else None,
         "help_zero_since": fin.get("help_zero_since") if fin else None, "first_help_zero_report_only": first_help_zero(job),
         "post_status": st.get("status"), "b_own_input": own_input_ok(job, st, fin)}
    for key in ("eval_sp_sr", "eval_mp_sr", "eval_world_count", "evaluated_iteration", "eval_add_auc_own_input",
                "eval_mppe_cm_own_input", "export_max_rel_diff", "eval_reference_is_train_input", "note"):
        s[key] = st.get(key)
    s["incomplete"] = bool((crashed and fin is None) or st.get("status") == "failed")
    s["settled"] = s["incomplete"] or st.get("status") in TERMINAL
    s["_fin"] = fin
    return s


def _result(s: dict, parts: list) -> str:
    """FAIL if a part is False; PASS if every part is True; else INCOMPLETE (crash without final.json, or the evaluation
    failed twice), NOT LAUNCHED, or PENDING."""
    if any(p is False for p in parts):
        return "FAIL"
    if parts and all(p is True for p in parts):
        return "PASS"
    if s["incomplete"]:
        return "INCOMPLETE"
    if not s["launched"]:
        return "NOT LAUNCHED"
    return "PENDING"


def _pair(results: list[str], ok: str, bad: str) -> str:
    if all(r.startswith("PASS") for r in results):
        return ok
    if any(r.startswith("FAIL") for r in results):
        return bad
    if all(r == "NOT LAUNCHED" for r in results):
        return "NOT LAUNCHED"
    if any(r in ("PENDING", "NOT LAUNCHED") for r in results):
        return "PENDING"
    return "INCOMPLETE"


def _hz_rule(s: dict, limit: int) -> bool | None:
    """(a): final.json help_zero_since <= limit; None while final.json is missing. A run whose help was forced to zero by
    the deadline (39062) fails any limit below it."""
    if s["_fin"] is None:
        return None
    return s["help_zero_since"] is not None and int(s["help_zero_since"]) <= limit


def verdict(write: bool) -> int:
    ok, why = prereg_status()
    if not ok:
        print(f"VERDICT REFUSED: {why} (the rules are fixed by PREREG_v2.md; nothing written)")
        return 2
    raw_eval = raw_eval_rows()
    jobs = {job: job_summary(job) for job in ORDER}
    clips, fail_rows, notes = {}, [], []
    # ---- STAGE 1, ep07 / ep18 (K3 input): PASS iff (a) help_zero_since <= 30000 (strong <= 20000) and (b) own-input success
    for job in ("ep07_k3fix_s42", "ep18_k3fix_s42"):
        s, j = jobs[job], jinfo(job)
        a = _hz_rule(s, S1_HZ_LIMIT)
        strong = None if a is None else bool(a and int(s["help_zero_since"]) <= S1_HZ_STRONG)
        b = s["b_own_input"]
        res = _result(s, [a, b])
        if res == "PASS" and strong:
            res = "PASS (strong)"
        clips[f"ep{j['ep']:02d}"] = {
            "stage": 1, "job": job, "rule": f"(a) final.json help_zero_since <= {S1_HZ_LIMIT} (strong <= {S1_HZ_STRONG}); "
                                            f"(b) own-input SP-SR = MP-SR = 1 (final checkpoint)",
            "input": "K3 object track (K3fix leaves this clip unchanged; parquet byte-identical to the K3 root)",
            "raw_help_zero_since": j["raw_hz"], "raw_final_iteration": j["raw_final"],
            "raw_note": "raw seed-42 run: help forced to zero by the curriculum deadline (39062)",
            "tier1_help_zero_since_report_only": j["t1_hz"], "a_pass": a, "a_strong": strong, "b_pass": b, "result": res,
            "settled": res.startswith(("PASS", "FAIL")) or s["settled"]}
    # ---- STAGE 1, ep03 K3fix: PASS iff (a), (b) and (c) mean rsi/fail_rate_mean over 12k-15k <= max(raw s42, raw s7) + 0.08
    lo, hi = EP03_WINDOW
    raw42_rows, rnote = curve_raw(3, write)
    w_raw42, n_raw42 = window_mean(raw42_rows, lo, hi)
    fix_rows = curve_jsonl(run_dir("ep03_k3fix_s42") / "metrics.jsonl")
    s7_rows = curve_jsonl(run_dir("ep03_raw_s7") / "metrics.jsonl")
    w_fix, n_fix = window_mean(fix_rows, lo, hi)
    w_s7, n_s7 = window_mean(s7_rows, lo, hi)
    s_fix, s_s7 = jobs["ep03_k3fix_s42"], jobs["ep03_raw_s7"]
    fix_past = (bool(fix_rows) and fix_rows[-1][0] > hi) or (s_fix["final_iteration"] or 0) > hi
    s7_past = bool(s7_rows) and s7_rows[-1][0] > hi
    s7_gone = s_s7["settled"] or s_s7["crash_marker"]
    if w_s7 is not None:
        refs = [x for x in (w_raw42, w_s7) if x is not None]
    elif s7_gone or s7_past:
        refs = [x for x in (w_raw42,) if x is not None]
        notes.append("ep03: the raw seed-7 control has no complete 12k-15k window; the threshold uses raw seed 42 only (disclosed)")
    else:
        refs = None
    thr = (max(refs) + EP03_MARGIN) if refs else None
    c_note = ""
    if w_fix is not None:
        c = None if thr is None else (w_fix <= thr)
    elif fix_past:
        c, c_note = False, f"passed iteration {hi} without a complete window"
    elif s_fix["final_iteration"] is not None:
        c, c_note = True, (f"not applicable (PREREG_v2): the run ended at iteration {s_fix['final_iteration']} <= {hi} without a "
                           f"complete window (help zero since {s_fix['help_zero_since']})")
    else:
        c = None
    a3 = _hz_rule(s_fix, S1_HZ_LIMIT)
    strong3 = None if a3 is None else bool(a3 and int(s_fix["help_zero_since"]) <= S1_HZ_STRONG)
    b3 = s_fix["b_own_input"]
    r3 = _result(s_fix, [a3, b3, c])
    if r3 == "PASS":
        r3 = f"PASS{' (strong)' if strong3 else ''} (GO: ep03 on K3fix in a full arm)"
    elif r3 == "FAIL":
        r3 = "FAIL (NO-GO: ep03 stays raw in a full arm)"
    clips["ep03"] = {"stage": 1, "job": "ep03_k3fix_s42",
                     "rule": f"(a) final.json help_zero_since <= {S1_HZ_LIMIT} (strong <= {S1_HZ_STRONG}); (b) own-input SP-SR = MP-SR "
                             f"= 1; (c) mean rsi/fail_rate_mean over iterations {lo}-{hi} of K3fix s42 <= max(raw s42, raw s7) + "
                             f"{EP03_MARGIN}. PASS = GO for ep03's K3fix input in a full arm; FAIL = ep03 stays raw there",
                     "input": "K3fix object track (changed vs frozen K3)", "raw_help_zero_since": EPF[3]["raw_hz"],
                     "raw_final_iteration": EPF[3]["raw_final"], "raw_note": "raw seed-42 run: help forced to zero by the deadline (39062)",
                     "tier1_help_zero_since_report_only": EPF[3]["t1_hz"], "a_pass": a3, "a_strong": strong3, "b_pass": b3,
                     "c_pass": c, "c_note": c_note, "window_k3fix_s42": w_fix, "rows_k3fix_s42": n_fix, "window_raw_s42": w_raw42,
                     "rows_raw_s42": n_raw42, "window_raw_s7": w_s7, "rows_raw_s7": n_s7, "threshold": thr, "result": r3,
                     "raw_curve_note": rnote, "settled": r3.startswith(("PASS", "FAIL")) or s_fix["settled"]}
    s1 = _pair([clips["ep07"]["result"], clips["ep18"]["result"]],
               "STAGE 1 PASS: the K3 input is trainable on the easy clips (ep07 and ep18 PASS)",
               "STAGE 1 FAIL: the K3 input is NOT shown trainable on the easy clips")
    # ---- STAGE 2, ep17 / ep09 (K3fix): the PREREG v1 rule, unchanged: (a) help zero <= raw + 5000 and (b) own-input success
    for job in ("ep17_k3fix_s42", "ep09_k3fix_s42"):
        s, j = jobs[job], jinfo(job)
        limit = j["raw_hz"] + HZ_MARGIN
        a = _hz_rule(s, limit)
        b = s["b_own_input"]
        res = _result(s, [a, b])
        clips[f"ep{j['ep']:02d}"] = {"stage": 2, "job": job,
                                     "rule": f"(a) help_zero_since <= {j['raw_hz']} + {HZ_MARGIN} = {limit}; (b) own-input SP-SR = MP-SR = 1",
                                     "input": "K3fix object track (changed vs frozen K3)", "raw_help_zero_since": j["raw_hz"],
                                     "raw_final_iteration": j["raw_final"], "tier1_help_zero_since_report_only": j["t1_hz"],
                                     "a_limit": limit, "a_pass": a, "b_pass": b, "result": res,
                                     "settled": res.startswith(("PASS", "FAIL")) or s["settled"]}
    s2 = _pair([clips["ep17"]["result"], clips["ep09"]["result"]],
               "STAGE 2 PASS: K3fix trainable on the changed clips (ep09 and ep17 PASS)",
               "STAGE 2 FAIL: K3fix NOT shown trainable on the changed clips")
    # ---- STAGE 3, ep04: report only (PREREG v1 rule, unchanged)
    s4 = jobs["ep04_k3fix_s42"]
    sb_raw, sb_fix, sb_old, matched, broken = {}, {}, {}, [], None
    if s4["launched"]:
        raw04, _ = curve_raw(4, write)
        fix04 = curve_jsonl(run_dir("ep04_k3fix_s42") / "metrics.jsonl")
        old04 = curve_jsonl(OLD_K3_RUNS / "ep04/metrics.jsonl")
        sb_raw, sb_fix, sb_old = success_bins(raw04), success_bins(fix04), success_bins(old04)
        matched = sorted(set(sb_raw) & set(sb_fix))
        early = {k: v for k, v in sb_fix.items() if k + 1000 <= EP04_BY}
        if any(v > 0 for v in early.values()):
            broken = False
        elif fix04 and fix04[-1][0] >= EP04_BY:
            broken = True
    clips["ep04"] = {"stage": 3, "job": "ep04_k3fix_s42",
                     "rule": f"REPORT ONLY: start bins with success (window-mean fail EMA < {EP04_SUCCESS_FAIL}, rows >= {WARMUP_IT}) per "
                             f"1000-iteration window, K3fix vs raw at matched windows; 'input no longer broken' iff >= 1 bin has success "
                             f"in a window ending by {EP04_BY}",
                     "result": "REPORT ONLY" if s4["launched"] else "NOT LAUNCHED",
                     "success_bins_k3fix": {str(k): v for k, v in sb_fix.items()},
                     "success_bins_raw_s42": {str(k): sb_raw[k] for k in matched},
                     "success_bins_old_k3_stretch_report_only": {str(k): v for k, v in sb_old.items() if k < 10000},
                     "input_no_longer_broken": None if broken is None else (not broken),
                     "own_input_b": s4["b_own_input"], "settled": s4["settled"]}
    # ---- stage states and the decided next step
    stage_state = {}
    for stg, js in STAGE_JOBS.items():
        launched = [jobs[x]["launched"] for x in js]
        stage_state[stg] = ("not launched" if not any(launched) else "partly launched" if not all(launched)
                            else "settled" if all(jobs[x]["settled"] for x in js) else "running")
    final = any(v == "settled" for v in stage_state.values()) and all(v in ("settled", "not launched") for v in stage_state.values())
    ep03_input = "K3fix" if r3.startswith("PASS") else "raw" if r3.startswith("FAIL") else "pending"
    if s1.startswith("STAGE 1 FAIL"):
        nxt = ("no Stage 2 or 3 without a new user decision (K3 input not shown trainable on the easy clips); the raw 0.570 run "
               "stays the Tier-2 submission")
    elif not s1.startswith("STAGE 1 PASS"):
        nxt = f"Stage 1 {s1.lower()}: wait" + ("; INCOMPLETE: report and ask" if s1 == "INCOMPLETE" else "")
    elif s2 == "NOT LAUNCHED":
        nxt = (f"Stage 1 PASS (ep03 K3fix: {r3}). Stage 2 (ep17, ep09 on K3fix) is the user's explicit decision: "
               f"bash scripts/k3fix_driver.sh run --stage 2")
    elif s2.startswith("STAGE 2 PASS"):
        nxt = ("rebuild the full 30-episode K3fix arm (unchanged C3r2 recipe) as a candidate SECOND final, under its own "
               "pre-registration (the gate-4 finding on the unchanged episodes needs a rule there); ep03 input = "
               f"{ep03_input}; ep04 input = user decision from the Stage-3 report (Stage 3 is launched only by an explicit "
               "'run --stage 3'); the raw 0.570 run stays the first final")
    elif s2.startswith("STAGE 2 FAIL"):
        nxt = ("K3fix NOT shown trainable on the changed clips: no further K3fix training without a new user decision (Stage 1 "
               "passed on the K3 input: report and ask); the raw 0.570 run stays the Tier-2 submission")
    else:
        nxt = f"Stage 2 {s2.lower()}: wait" + ("; INCOMPLETE: report and ask" if s2 == "INCOMPLETE" else "")
    # ---- reported: fail rates vs the raw seed-42 run per 1000-it bin (launched jobs only)
    for job in ORDER:
        if not jobs[job]["launched"]:
            continue
        ep = JOBS[job]["ep"]
        rawc, _ = curve_raw(ep, write)
        rb, kb = bins_1000(rawc), bins_1000(curve_jsonl(run_dir(job) / "metrics.jsonl"))
        for it in sorted(set(rb) & set(kb)):
            fail_rows.append({"job": job, "episode": ep, "iteration_bin": it, "raw_s42_fail_rate_mean": round(rb[it], 4),
                              "job_fail_rate_mean": round(kb[it], 4), "job_minus_raw": round(kb[it] - rb[it], 4)})
    for job, s in jobs.items():
        rr = raw_eval.get(s["episode"], {})
        s["raw_s42_own_input_eval"] = {"iteration": rr.get("iteration"), "auc": fnum(rr.get("auc")), "sp_sr": fnum(rr.get("sp_sr")),
                                       "mp_sr": fnum(rr.get("mp_sr")), "mppe_cm": fnum(rr.get("mppe_cm"))}
        s.pop("_fin", None)
    out = {"written": now(), "prereg": str(PREREG), "prereg_sha256": sha256(PREREG), "prereg_check": why,
           "stage_state": {str(k): v for k, v in stage_state.items()}, "stage1_verdict": s1, "stage2_verdict": s2,
           "stage3": clips["ep04"]["result"], "final_for_launched_stages": final, "next_step": nxt, "clips": clips, "jobs": jobs,
           "notes": notes}
    lines = [f"K3fix staged pilot ({ARM}) - verdict by the pre-registered rules ({why})",
             f"written {out['written']}; stages: " + " | ".join(f"{k} {v}" for k, v in stage_state.items())
             + f"; final for the launched stages: {'yes' if final else 'no'}",
             f"STAGE 1: {s1}", f"STAGE 2: {s2}", f"STAGE 3: ep04 {clips['ep04']['result']}", f"NEXT: {nxt}", ""]
    for name in ("ep07", "ep18"):
        c = clips[name]
        s = jobs[c["job"]]
        lines.append(f"[1] {name} {c['result']:<13} (a) help zero {s['help_zero_since']} (limit {S1_HZ_LIMIT}, strong {S1_HZ_STRONG}; raw s42 "
                     f"{c['raw_help_zero_since']} deadline-forced; Tier-1 {c['tier1_help_zero_since_report_only']} report only) -> {c['a_pass']} | "
                     f"(b) own-input SP-SR {s['eval_sp_sr']} MP-SR {s['eval_mp_sr']} worlds {s['eval_world_count']} ckpt it "
                     f"{s['evaluated_iteration']} -> {c['b_pass']} | post {s['post_status']}{' | CRASH MARKER' if s['crash_marker'] else ''}")
    c = clips["ep03"]
    s = jobs["ep03_k3fix_s42"]
    lines.append(f"[1] ep03 {c['result']}: (a) help zero {s['help_zero_since']} (limit {S1_HZ_LIMIT}) -> {c['a_pass']} | (b) own-input "
                 f"SP-SR {s['eval_sp_sr']} MP-SR {s['eval_mp_sr']} -> {c['b_pass']} | (c) fail-rate window {lo}-{hi}: K3fix s42 "
                 f"{c['window_k3fix_s42']} ({c['rows_k3fix_s42']} rows) vs raw s42 {c['window_raw_s42']} ({c['rows_raw_s42']}) / raw s7 "
                 f"{c['window_raw_s7']} ({c['rows_raw_s7']}); threshold {c['threshold']} -> {c['c_pass']}"
                 f"{' (' + c['c_note'] + ')' if c['c_note'] else ''}{' | CRASH MARKER' if s['crash_marker'] else ''}")
    for name in ("ep17", "ep09"):
        c = clips[name]
        s = jobs[c["job"]]
        lines.append(f"[2] {name} {c['result']:<13} (a) help zero {s['help_zero_since']} vs raw {c['raw_help_zero_since']} (limit "
                     f"{c['a_limit']}) -> {c['a_pass']} | (b) own-input SP-SR {s['eval_sp_sr']} MP-SR {s['eval_mp_sr']} worlds "
                     f"{s['eval_world_count']} ckpt it {s['evaluated_iteration']} -> {c['b_pass']} | post {s['post_status']}"
                     f"{' | CRASH MARKER' if s['crash_marker'] else ''}")
    c = clips["ep04"]
    lines.append(f"[3] ep04 {c['result']}: input no longer broken: {c['input_no_longer_broken']}; success bins K3fix {c['success_bins_k3fix']} | "
                 f"raw s42 (matched) {c['success_bins_raw_s42']} | old K3 (stretch, report only) {c['success_bins_old_k3_stretch_report_only']}")
    lines += ["", "Jobs (reported, not decided beyond the rules above):"]
    for job, s in jobs.items():
        lines.append(f"  [{s['stage']}] {job:<16} {'launched' if s['launched'] else 'not launched'} gpu {s['train_gpu']} final "
                     f"{s['final_iteration']} ({s['final_reason']}) help zero {s['help_zero_since']} (first {s['first_help_zero_report_only']}) | "
                     f"own-input AUC {s['eval_add_auc_own_input']} MPPE {s['eval_mppe_cm_own_input']} SP/MP {s['eval_sp_sr']}/{s['eval_mp_sr']} "
                     f"vs raw s42 AUC {s['raw_s42_own_input_eval']['auc']} | post {s['post_status']}")
    lines += [f"  note: {n}" for n in notes]
    lines.append("  per-bin fail rates: fail_rates_vs_raw.csv; clean diagnostics only in VERDICT_clean_LOCAL_NOT_FOR_UPLOAD.json")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if write:
        atomic_write(K / "VERDICT.json", json.dumps(out, indent=1, default=str))
        atomic_write(K / "VERDICT.txt", text)
        with (K / "fail_rates_vs_raw.csv.tmp").open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["job", "episode", "iteration_bin", "raw_s42_fail_rate_mean", "job_fail_rate_mean", "job_minus_raw"])
            wr.writeheader()
            wr.writerows(fail_rows)
        os.replace(K / "fail_rates_vs_raw.csv.tmp", K / "fail_rates_vs_raw.csv")
        base, clean = {}, {}
        try:
            for r in csv.DictReader(BASELINE_CSV.open()):
                if r["episode"].strip().isdigit() and int(r["episode"]) in EPF:
                    base[int(r["episode"])] = {k: fnum(r.get(f"t2train_mean_{k}")) for k in ("add_auc", "spider_sr", "maniptrans_sr", "mppe_cm")}
        except (OSError, KeyError):
            pass
        for job in ORDER:
            try:
                clean[job] = json.loads((EVROOT / job / "clean_LOCAL_NOT_FOR_UPLOAD.json").read_text())
            except (OSError, ValueError):
                clean[job] = None
        atomic_write(K / "VERDICT_clean_LOCAL_NOT_FOR_UPLOAD.json", json.dumps(
            {"written": now(), "note": "LOCAL ONLY. One clean-reference diagnostic per job (arm-level, disclosed); never selects "
                                       "checkpoints, never enters the verdict. Raw = C3r2_tier2 seed-42 final checkpoint, mean of 3 "
                                       "evaluations (early_table.csv t2train_mean_*).",
             "jobs": clean, "raw_s42_mean3": {f"{ep:02d}": base.get(ep) for ep in sorted(EPF)}}, indent=1, default=str))
        print(f"wrote {K / 'VERDICT.json'}, VERDICT.txt, fail_rates_vs_raw.csv, VERDICT_clean_LOCAL_NOT_FOR_UPLOAD.json")
    return 0


def heartbeat() -> int:
    parts = []
    for job in ORDER:
        s, _ = job_state(job)
        pr = progress(job)
        it = pr["it"] if pr["it"] is not None else pr["last_model"]
        parts.append(f"{job} {s} it={it} hz={pr['help_zero_iters']} fail={round(pr['fail'], 3) if isinstance(pr['fail'], float) else pr['fail']}")
    print(" | ".join(parts))
    return 0


# ============================================================================================== main
def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    sub, rest = argv[0], argv[1:]

    def opt(name, default=None):
        return rest[rest.index(name) + 1] if name in rest and rest.index(name) + 1 < len(rest) else default

    if sub == "check":
        return check()
    if sub == "cfgcheck":
        return cfgcheck("--no-write" not in rest)
    if sub == "cmd":
        sys.stdout.write("\0".join(train_argv(job_of(opt("--job")))) + "\0")
        return 0
    if sub == "plan":
        rem = opt("--remaining")
        assume = {int(a): int(b) for a, b in (x.split(":") for x in (opt("--assume", "") or "").split(",") if x)}
        print(plan(job_of(opt("--job")), opt("--exclude", "") or "", int(rem) if rem else None, assume, parse_stages(opt("--stages"))))
        return 0
    if sub == "data":
        print(jinfo(job_of(opt("--job")))["data"])
        return 0
    if sub == "gpus":
        print("\n".join(gpus_report()))
        return 0
    if sub == "state":
        return state(parse_stages(opt("--stages")))
    if sub == "stage-jobs":
        st = parse_stages(opt("--stages"))
        if not st:
            print("no stage given (--stages 1|2|3[,..])")
            return 2
        print("\n".join(j for j in ORDER if JOBS[j]["stage"] in st))
        return 0
    if sub == "stage-guard":
        return stage_guard(parse_stages(opt("--stages")) or set())
    if sub == "trainer":
        print(" ".join(map(str, find_trainers(job_of(opt("--job"))))))
        return 0
    if sub == "crash":
        return crash(job_of(opt("--job")), opt("--why", ""))
    if sub == "post":
        return post(job_of(opt("--job")))
    if sub == "verdict":
        return verdict("--no-write" not in rest)
    if sub == "heartbeat":
        return heartbeat()
    if sub == "build-record":
        atomic_write(K / "build.json", json.dumps({"written": now(), "code": str(CODE), "pin": PIN, "code_digest": code_digest()}, indent=1))
        print(f"wrote {K / 'build.json'}")
        return 0
    print(f"unknown subcommand {sub}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
