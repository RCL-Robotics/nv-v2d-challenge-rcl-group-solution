#!/usr/bin/env python3
"""k3full_helper.py: job table, checks, GPU / CPU plan (fill rule + Tier-3 yield + Tier-1 equal share), state,
post-training step and arm-level verdict of scripts/k3full_driver.sh: the FULL K3fix arm (arm folder C3r2_tier2_k3fix,
pre-registration tmp_analysis/tier2/k3full/PREREG_v3.md). User 2026-10-22: "so you can train on more tasks right?" after
the K3 pilot results: extend the K3fix training to every episode that passes the noisy-only input rule.

  Input rule (noisy-only, fixed): G1-G4 of the K3fix pilot on all 30 episodes (scripts/k3full_gates.py ->
  k3full/build/gates_all30.json). PASS -> K3fix candidate; FAIL -> the raw C3r2_tier2 run.
  Jobs = gate-passing episodes minus those already trained / running on the K3fix input:
    ep03 / ep07 / ep18  k3fix pilot stage 1  (fc_runs/C3r2_tier2_k3fix/epNN_k3fix_s42, owned by scripts/k3fix_driver.sh)
    ep08 / ep20 / ep23  k3easy test          (fc_runs/C3r2_tier2_k3easy/epNN, K3 input = K3fix input byte-identically)
  ordered easy-first by the RAW Tier-2 run's own final.json help_zero_since (early_table.csv t2train_help_zero_since,
  ascending; ties by episode index), ep04 last. ep17 / ep09 / ep04 keep their k3fix pilot names (stage 2 / 3 there):
  they are launched here but REGISTERED in k3fix_pilot/pids.tsv as well, so the running k3fix driver watches them as
  its own jobs, runs their post-training step (k3fix_helper.py post) and judges them by PREREG_v2 stage 2 / 3.
  Recipe: exactly the raw C3r2_tier2 run of the episode (code/fc_r2 @61fa5d3, g1_recon_body_coord_rsi_r2, seed 42,
  45000, experiment-default RSI deadline 35156 / zero_ramp 3906 / help-zero tail 5860, scene.support as the raw run);
  only the data root (data/human_motion_data_tier2_k3fix), output_dir and logging differ (cfgcheck).

Code reuse: scripts/k3fix_helper.py is imported READ-ONLY (sha256 pinned) twice: F = the pilot's own view (its jobs,
pids.tsv, post.json, verdict), F2 = a private in-memory instance whose job table and record paths are re-pointed at this
arm (tmp_analysis/tier2/k3full/{pids.tsv,eval/,post/,evallock/,CRASHED_*}); F2.post is therefore the pilot's exact
post-training step (export 2e-4 + ONE official own-input evaluation of final.json's checkpoint + LOCAL clean diagnostic
with frame_refs_k3fix). The k3fix_helper.py file is never modified.

Run with conda_envs/flashchord/bin/python -I -B. Never starts a training (the driver does), never signals a process except
its own evaluation session on timeout (as k3fix_helper). Writes only under tmp_analysis/tier2/k3full/, except the
documented registration line of ep17 / ep09 / ep04 in k3fix_pilot/pids.tsv (written by the driver at their launch).

Subcommands:
  build-jobs [--no-write]   derive jobs.json from gates_all30.json + early_table.csv + the raw runs (once, at build)
  check                     preflight (PREREG_v3 sha256, pinned files, code pin, data root, gates, jobs, cfgcheck)
  cfgcheck [--no-write]     hydra compose of every job vs the raw run's saved config.yaml -> k3full/cfgcheck.json
  order                     the jobs in launch order
  cmd --job J               training argv (NUL-separated)        data --job J    the data root
  native --job J            "yes" if J is a k3fix pilot job (ep17/ep09/ep04), else "no"
  launch-ok --job J         "OK" or "REFUSE why" (FRESH everywhere, input pinned, gate PASS)
  plan --job J [--exclude CPUS] [--assume g:n,..] [--no-yield]   "OK <gpu> <cpus> <nthreads> - | why" or "WAIT why"
  yield                     the Tier-3 yield state (t3opus driver)       share     the Tier-1 equal-share reserve
  gpus                      per-GPU view          state     one line per job "JOB <job> <STATE> detail"
  trainer --job J           live trainer pid(s)   crash --job J --why T   k3full/CRASHED_<job> once (never a relaunch)
  post --job J              post-training step (new jobs: F2.post; ep17/09/04: F.post unless the k3fix driver is alive)
  verdict [--no-write]      arm-level composition + reports (exit 2, nothing written, if PREREG_v3.sha256 mismatches)
  heartbeat                 one status line
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ME = Path("/root/autodl-tmp/haoran.zhang")
sys.path.insert(0, str(ME / "scripts"))
import track2_queue_tier2v2_helper as H  # noqa: E402  read-only primitives (nvidia-smi, /proc, cpu slice, parquet)

K3FIX_HELPER = ME / "scripts/k3fix_helper.py"
K3FIX_HELPER_SHA = "d563555aa8634da2d90ab4375bfe3fa1539179c2b6c45d4e0e96f5c1862401b9"  # k3fix_pilot/build_files_v2.sha256
GATE_CODE_SHA = "364d27ed1b0e0a250456ff2fa42b130c7f73bfb4755f88c75237dc79148dff7b"     # scripts/k3fix_build.py


def _sha(path) -> str | None:
    try:
        h = hashlib.sha256()
        with Path(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _load(name: str):
    got = _sha(K3FIX_HELPER)
    if got != K3FIX_HELPER_SHA:
        raise SystemExit(f"{K3FIX_HELPER} sha256 {got} != pinned {K3FIX_HELPER_SHA}: refusing (the reused code changed)")
    spec = importlib.util.spec_from_file_location(name, K3FIX_HELPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F = _load("k3fix_helper_pilot_ro")    # the pilot as it is (never re-pointed)
F2 = _load("k3fix_helper_k3full")     # re-pointed at this arm in setup()

ARM = F.ARM                                   # C3r2_tier2_k3fix (same arm folder: t2watch picks the runs up)
K = ME / "tmp_analysis/tier2/k3full"
RUNS = F.RUNS                                 # fc_runs/C3r2_tier2_k3fix
EVROOT = K / "eval"
PIDS = K / "pids.tsv"
JOBS_JSON = K / "jobs.json"
GATES_JSON = K / "build/gates_all30.json"
CFGCHECK = K / "cfgcheck.json"
PREREG = K / "PREREG_v3.md"
PREREG_SHA = K / "PREREG_v3.sha256"
EARLY = ME / "tmp_analysis/tier2/early_scores/early_table.csv"
RAW_RUNS = F.RAW_RUNS                         # fc_runs/C3r2_tier2 (the 0.570 arm; its final.json checkpoints)
RAW_EVAL_CSV = F.RAW_EVAL_CSV
DATA_FIX, RAW_DATA, K3_DATA = F.DATA_FIX, F.RAW_DATA, F.K3_DATA
READY_SHA = "1233398f02e61b51fb9bc13072b1607ec3964556cc20828ac56f571b1640dac8"
CODE, PIN = F.CODE, F.PIN
SH = CODE / "logs/rl/sharpa"
K3EASY = ME / "tmp_analysis/tier2/k3easy"
K3EASY_RUNS = ME / "fc_runs/C3r2_tier2_k3easy"
T3 = ME / "tmp_analysis/tier3/opus_easy"
T3_HELPER = ME / "scripts/t3opus_helper.py"
T1FIX = ME / "fc_runs/T1fix"
FCPY = F.FCPY
ITERS = F.ITERS
EXPORT_TOL = F.EXPORT_TOL
CHANGED = (3, 4, 9, 17)
EXISTING = {  # gate-passing episodes already trained / running on the K3fix input (not retrained)
    3: ("k3fix", "ep03_k3fix_s42"), 7: ("k3fix", "ep07_k3fix_s42"), 18: ("k3fix", "ep18_k3fix_s42"),
    8: ("k3easy", "ep08"), 20: ("k3easy", "ep20"), 23: ("k3easy", "ep23")}
ALLOWED_GPUS = (0, 1, 2, 3, 5, 6, 7)        # never GPU 4
PLACE_TIERS = ((3, 5, 6, 7), (0, 1, 2))      # NUMA 1 first, then NUMA 0 (as the t3opus driver)
MAX_TRAIN_PER_GPU = 3
JOB_MB = 16896
MIN_FREE_AFTER_MB = 20480
CORES = 2                                     # physical cores (+HT) per job = the T1fix / k3fix / t3opus slot size
RAMP_S = 900
UNKNOWN_TRAIN_AGE_S = 1800
T3_WAITING = ("FRESH", "NEEDS_RETARGET", "NEEDS_RCHECK", "RETARGETING")
T3_HEARTBEAT_MAX_S = 1800                     # the t3opus driver writes heartbeat.log every 600 s while it loops
SHARED_GPUS = (0, 1, 2, 3, 5, 6, 7)           # t1fix_launch.py SHARED_GPUS (equal-share pool)
TERMINAL = F.TERMINAL
JOBS: dict[str, dict] = {}
ORDER: tuple[str, ...] = ()
_SETUP = False


# ============================================================================================== setup
def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_jobs() -> dict:
    return json.loads(JOBS_JSON.read_text())


def setup() -> None:
    """Load jobs.json and re-point F2 at this arm. F stays the pilot's own view."""
    global JOBS, ORDER, _SETUP
    if _SETUP:
        return
    jj = load_jobs()
    ORDER = tuple(jj["order"])
    JOBS = {name: dict(jj["jobs"][name]) for name in ORDER}
    F2.JOBS = {n: {"ep": j["ep"], "kind": "k3fix", "seed": 42, "stage": 0} for n, j in JOBS.items()}
    F2.EPF = {j["ep"]: {"seq": j["seq"], "obj": j["obj"], "support": j["support"], "raw_hz": j["raw_hz"],
                        "raw_final": j["raw_final"], "raw_hydra": j["raw_hydra"], "raw_sha": j["raw_sha"],
                        "fix_sha": j["fix_sha"], "t1_hz": None, "changed": j["changed"]} for j in JOBS.values()}
    F2.ORDER = ORDER
    F2.K, F2.EVROOT, F2.PIDS, F2.OWN_EVALLOCK, F2.CFGCHECK = K, EVROOT, PIDS, K / "evallock", CFGCHECK
    F2.PREREG, F2.PREREG_SHA = PREREG, PREREG_SHA
    F2.acquire_lane = acquire_lane
    _SETUP = True


def job_of(text: str | None) -> str:
    setup()
    if text not in JOBS:
        raise SystemExit(f"unknown job {text!r} (jobs: {' '.join(ORDER)})")
    return text


def native(job: str) -> bool:
    return bool(JOBS[job]["native"])


def prereg_status() -> tuple[bool, str]:
    if not PREREG.is_file() or not PREREG_SHA.is_file():
        return False, f"{PREREG.name} or {PREREG_SHA.name} missing"
    words = PREREG_SHA.read_text().split()
    want, got = (words[0] if words else ""), (_sha(PREREG) or "")
    return got == want, f"sha256({PREREG.name}) {got[:16]}... {'==' if got == want else '!='} {PREREG_SHA.name} {want[:16]}..."


def proc_alive_with(pidfile: Path, needle: str) -> int | None:
    try:
        pid = int(pidfile.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    cmd = " ".join(H.read_cmdline(pid))
    return pid if needle in cmd and H.alive(pid) else None


def k3fix_driver_alive() -> int | None:
    return proc_alive_with(F.K / "driver.pid", "k3fix_driver.sh")


# ============================================================================================== build-jobs
def early_hz() -> dict[int, int]:
    out = {}
    for r in csv.DictReader(EARLY.open()):
        if r["episode"].strip().isdigit():
            out[int(r["episode"])] = int(float(r["t2train_help_zero_since"]))
    return out


def raw_hydra_dir(ep: int) -> Path:
    want = f"output_dir={RAW_RUNS / f'ep{ep:02d}'}"
    hits = [d for d in sorted(SH.glob(f"*_ep{ep:02d}_*")) if (d / ".hydra/overrides.yaml").is_file()
            and want in [str(x) for x in F._load_yaml_list(d / ".hydra/overrides.yaml")]]
    if len(hits) != 1:
        raise SystemExit(f"ep{ep:02d}: {len(hits)} raw hydra dirs with {want} (exactly 1 expected)")
    return hits[0] / ".hydra"


def build_jobs(write: bool) -> int:
    g = json.loads(GATES_JSON.read_text())
    if not g.get("checks_ok") or g.get("gate_code_sha256") != GATE_CODE_SHA or g.get("k3fix_root_ready_sha256") != READY_SHA:
        raise SystemExit("gates_all30.json: checks not ok, or gate code / K3fix root differ from the pinned ones")
    passing, failing = sorted(g["pass_episodes"]), sorted(g["fail_episodes"])
    hz = early_hz()
    man = json.loads((DATA_FIX / "whole_body/soma/track2_manifest.json").read_text())
    by_ep = {int(r["episode_index"]): (str(r.get("sequence_id", s)), str(r.get("object"))) for s, r in man.items()}
    ready = F.ready_table()
    todo = [e for e in passing if e not in EXISTING]
    order_eps = sorted(todo, key=lambda e: (e == 4, hz[e], e))
    jobs = {}
    for ep in order_eps:
        name = f"ep{ep:02d}_k3fix_s42"
        seq, obj = by_ep[ep]
        hyd = raw_hydra_dir(ep)
        ov = [str(x) for x in F._load_yaml_list(hyd / "overrides.yaml")]
        fin = H.final_marker(RAW_RUNS / f"ep{ep:02d}") or {}
        rel = f"whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet"
        j = {"ep": ep, "obj": obj, "seq": seq, "support": "scene.support=false" not in ov, "raw_hydra": str(hyd),
             "raw_hz": fin.get("help_zero_since"), "raw_final": fin.get("iteration"), "early_table_hz": hz[ep],
             "raw_sha": _sha(RAW_DATA / rel), "fix_sha": ready.get(rel), "changed": ep in CHANGED, "native": name in F.JOBS,
             "gate": "PASS"}
        if j["raw_hz"] != hz[ep]:
            raise SystemExit(f"ep{ep:02d}: raw final.json help_zero_since {j['raw_hz']} != early_table {hz[ep]}")
        if j["fix_sha"] is None or _sha(DATA_FIX / rel) != j["fix_sha"]:
            raise SystemExit(f"ep{ep:02d}: K3fix parquet missing or != READY")
        if j["native"]:
            e = F.EPF[ep]
            for key in ("seq", "obj", "support", "raw_hz", "raw_final", "raw_sha", "fix_sha", "changed"):
                if e[key] != j[key]:
                    raise SystemExit(f"{name}: {key} {j[key]!r} != k3fix_helper.EPF {e[key]!r}")
            if str(Path(e["raw_hydra"])) != str(hyd):
                raise SystemExit(f"{name}: raw hydra {hyd} != k3fix_helper.EPF {e['raw_hydra']}")
        jobs[name] = j
    out = {"written": now(), "rule": "jobs = gate-PASS episodes (gates_all30.json) minus EXISTING; order = raw C3r2_tier2 "
                                     "final.json help_zero_since ascending (early_table t2train_help_zero_since), ties by "
                                     "episode index, ep04 last",
           "gates_json": str(GATES_JSON), "gates_json_sha256": _sha(GATES_JSON), "pass_episodes": passing,
           "fail_episodes": failing, "existing": {f"{e:02d}": {"source": s, "run": r} for e, (s, r) in sorted(EXISTING.items())},
           "order": list(jobs), "jobs": jobs}
    for i, (n, j) in enumerate(jobs.items(), 1):
        print(f"{i:2d}. {n:<16} ep{j['ep']:02d} {j['obj']:<20} raw help zero {j['raw_hz']} (final {j['raw_final']}) "
              f"support {j['support']!s:<5} {'K3fix-changed' if j['changed'] else '= K3 input'}"
              f"{'  [k3fix pilot job, registered there]' if j['native'] else ''}")
    print(f"gate PASS {len(passing)}: {passing}; FAIL (raw) {failing}; existing K3fix runs {sorted(EXISTING)}; to train {len(jobs)}")
    if write:
        F.atomic_write(JOBS_JSON, json.dumps(out, indent=1))
        print(f"wrote {JOBS_JSON} (sha256 {_sha(JOBS_JSON)})")
    return 0


# ============================================================================================== paths / argv
def parquet(job: str, root: Path | None = None) -> Path:
    j = JOBS[job]
    return (root or DATA_FIX) / "whole_body/soma" / f"sequence_id={j['seq']}" / "robot_name=g1/data.parquet"


def train_argv(job: str) -> list[str]:
    argv = F2.train_argv(job)
    if native(job) and argv != F.train_argv(job):
        raise SystemExit(f"{job}: argv differs from k3fix_helper.train_argv")
    return argv


# ============================================================================================== evaluation lane
OTHER_LANES = (F.OWN_EVALLOCK, K3EASY / "evallock", T3 / "evallock")  # opened read-only if present, never created


def acquire_lane(job: str, gpu: int) -> list[int]:
    """k3fix_helper.acquire_lane with this arm's own lock plus, read-only when present, the shared rolling-start lane
    lock and the k3fix / k3easy / t3opus lane locks of that GPU (one evaluation per GPU across these drivers)."""
    own = K / "evallock"
    own.mkdir(parents=True, exist_ok=True)
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
            elif st["gpus"][gpu]["free"] < F.EVAL_MIN_FREE_MB:
                why = f"gpu{gpu} free {st['gpus'][gpu]['free']} MiB < {F.EVAL_MIN_FREE_MB}"
        except Exception as error:  # noqa: BLE001
            why = f"nvidia-smi failed ({type(error).__name__})"
        if not why:
            fds, held = [], []
            mine = F.try_lock(own / f"gpu{gpu}.lock", create=True)
            if mine is None:
                why = f"gpu{gpu}: another k3full evaluation holds the lane"
            else:
                fds.append(mine)
                for d in (F.RUNIN_EVALLOCK,) + OTHER_LANES:
                    p = d / f"gpu{gpu}.lock"
                    if not p.exists():
                        continue
                    fd = F.try_lock(p, create=False)
                    if fd is None:
                        why = f"gpu{gpu}: {p} is held"
                        break
                    fds.append(fd)
                    held.append(str(p))
                if why:
                    for fd in fds:
                        os.close(fd)
                else:
                    F2.plog(job, f"eval lane gpu{gpu} taken (k3full lock + {', '.join(held) or 'no other lane lock file'})")
                    return fds
        if why != last:
            F2.plog(job, f"eval waits: {why}")
            last = why
        time.sleep(30)


# ============================================================================================== GPU view / plan
def observe() -> dict[int, dict]:
    """Per GPU: trainings of anybody (as k3fix_helper.observe), free memory, MPS, and the ramp debt of EVERY training
    younger than RAMP_S (any launcher: JOB_MB minus its memory so far), as t1fix_launch.py counts it."""
    st = H.smi()
    pids = H.all_pids()
    uuid_to_idx = {v["uuid"]: k for k, v in st["gpus"].items()}
    mps = H.mps_gpus(st, pids)
    view = {g: {"free": v["free"], "total": v["total"], "trainings": [], "mps": mps.get(g), "debt": 0, "young": []}
            for g, v in st["gpus"].items()}
    mem = {a["pid"]: a["mem"] for a in st["apps"]}
    counted: dict[int, int] = {}
    for app in st["apps"]:
        pid, g = app["pid"], app["gpu"]
        cls = H.classify(H.read_cmdline(pid), H.read_environ(pid))
        if cls == "train" or (cls == "unknown" and (H.proc_age(pid) or 0.0) >= UNKNOWN_TRAIN_AGE_S):
            counted[pid] = g
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
        if (H.proc_age(pid) or 0.0) < RAMP_S:
            view[g]["debt"] += max(0, JOB_MB - mem.get(pid, 0))
            view[g]["young"].append(pid)
    return view


def node(g: int) -> int:
    return 0 if g <= 2 else 1


def own_cores(g: int, exclude: str) -> tuple[str | None, int, str]:
    cpus, nth, note = H.cpu_slice(g, exclude, CORES, CORES)
    if cpus == "none" or note or nth < 2 * CORES:
        return None, 0, note or f"{nth // 2} free"
    return cpus, nth, ""


def free_core_count(nd: int, exclude: str) -> int:
    cpus, n, _ = H.cpu_slice(0 if nd == 0 else 5, exclude, 60, 0)
    return 0 if cpus == "none" else n // 2


def t3opus_state() -> tuple[bool, str]:
    """(yield?, detail). Yield while the t3opus driver runs in 'run' mode, is looping (heartbeat.log < 30 min old) and a
    clip of the wave(s) it was started with still waits to be retargeted or launched (states FRESH / NEEDS_RETARGET /
    NEEDS_RCHECK / RETARGETING). If its state cannot be read while that driver runs: yield (conservative)."""
    pid = proc_alive_with(T3 / "driver.pid", "t3opus_driver.sh")
    if pid is None:
        return False, "no live t3opus driver"
    cmd = H.read_cmdline(pid)
    if "run" not in cmd:
        return False, f"t3opus driver pid {pid} is not in run mode ({' '.join(cmd[1:])})"
    try:
        hb_age = time.time() - (T3 / "heartbeat.log").stat().st_mtime
    except OSError:
        hb_age = None
    if hb_age is None or hb_age > T3_HEARTBEAT_MAX_S:
        return False, (f"t3opus driver pid {pid} looks stalled (heartbeat.log "
                       f"{'missing' if hb_age is None else f'{hb_age / 60:.0f} min old'} > {T3_HEARTBEAT_MAX_S // 60} min): no yield")
    waves = cmd[cmd.index("--wave") + 1] if "--wave" in cmd and cmd.index("--wave") + 1 < len(cmd) else ""
    if not re.fullmatch(r"[12](,[12])*", waves):
        return True, f"t3opus driver pid {pid}: waves unreadable ({' '.join(cmd)}): yield"
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    try:
        out = subprocess.run([str(FCPY), "-I", "-B", str(T3_HELPER), "state", "--waves", waves], capture_output=True,
                             text=True, timeout=120, env=env)
        rows = [ln.split(None, 3) for ln in out.stdout.splitlines() if ln.startswith("CLIP ")]
    except (OSError, subprocess.TimeoutExpired) as error:
        return True, f"t3opus state failed ({type(error).__name__}): yield"
    if out.returncode != 0 or not rows:
        return True, f"t3opus state exit {out.returncode}, {len(rows)} rows: yield"
    waiting = [f"ep{r[1]} {r[2]}" for r in rows if len(r) >= 3 and r[2] in T3_WAITING]
    if waiting:
        return True, f"t3opus driver pid {pid} (wave {waves}) still waits for a GPU: {', '.join(waiting)}"
    return False, f"t3opus driver pid {pid} (wave {waves}): no clip waiting ({', '.join(f'ep{r[1]} {r[2]}' for r in rows)})"


def gpu_slots(v: dict, extra: int) -> int:
    if v.get("mps"):
        return 0
    free = v["free"] - v["debt"] - extra * JOB_MB
    return max(0, min(MAX_TRAIN_PER_GPU - len(v["trainings"]) - extra, (free - MIN_FREE_AFTER_MB) // JOB_MB))


def feasible(view: dict, assume: dict, exclude: str) -> int:
    """t1fix_launch.py feasible(): per NUMA node min(GPU slots on SHARED_GPUS, free own-node cores // CORES)."""
    per = {0: 0, 1: 0}
    for g in SHARED_GPUS:
        if g in view:
            per[node(g)] += gpu_slots(view[g], assume.get(g, 0))
    return sum(min(per[nd], free_core_count(nd, exclude) // CORES) for nd in (0, 1))


def t1fix_share(view: dict, exclude: str) -> dict:
    """The T1fix launcher's equal-share rule seen from this side: reserve R1 = min(T1fix arms not yet launched,
    max(0, ceil(S/2) - T1fix running)), S = feasible slots + T1fix running + Tier-2 running, while the T1fix 'auto'
    launcher is alive (else 0). A start here is allowed only if the feasible slots after it stay >= R1."""
    alive = [p for p in H.all_pids() if "t1fix_launch.py" in " ".join(H.read_cmdline(p)) and "auto" in H.read_cmdline(p)]
    pending, arms = 0, 0
    try:
        for row in csv.DictReader((T1FIX / "arms.tsv").open(), delimiter="\t"):
            arms += 1
            if not (Path(row["run_dir"]) / "t1fix_launch.json").exists():
                pending += 1
    except (OSError, KeyError):
        pass
    t1_run = t2_run = 0
    for p in H.all_pids():
        cmd = H.read_cmdline(p)
        if not H.is_python_trainer(cmd) or not H.alive(p):
            continue
        if H.read_environ(p).get("T1FIX_ARM"):
            t1_run += 1
        out = next((a.split("=", 1)[1] for a in cmd if a.startswith("output_dir=")), "")
        if out.strip("'\"").startswith(str(ME / "fc_runs/C3r2_tier2")):
            t2_run += 1
    f_now = feasible(view, {}, exclude)
    pool = f_now + t1_run + t2_run
    reserve = min(pending, max(0, math.ceil(pool / 2) - t1_run)) if alive else 0
    return {"t1fix_auto_alive": bool(alive), "t1fix_arms": arms, "t1fix_pending": pending, "t1fix_running": t1_run,
            "tier2_running": t2_run, "feasible_now": f_now, "pool": pool, "reserve_t1": reserve}


def plan(job: str, exclude: str, assume: dict[int, int] | None = None, use_yield: bool = True) -> str:
    """GPU for the next training of this arm. WAIT while the t3opus driver has a clip waiting (Tier 3 first). Else a GPU
    qualifies: allowed (never 4), no MPS, < 3 trainings of anybody before the start, >= 16.5 GB + 20 GB free + the ramp
    debt of every young training, 2 free physical cores of its own NUMA node (never borrowed); placement NUMA 1 (3/5/6/7)
    first, then NUMA 0 (0/1/2); fewest trainings, most free memory, lowest index; and the T1fix reserve must still fit
    after the start. assume {gpu: n} (dry run): n hypothetical extra trainings there."""
    if use_yield:
        y, ydet = t3opus_state()
        if y:
            return f"WAIT yield to Tier 3: {ydet}"
    try:
        view = observe()
    except Exception as error:  # noqa: BLE001
        return f"WAIT nvidia-smi failed ({type(error).__name__}: {str(error)[:100]})"
    assume = assume or {}
    why, cands = [], []
    for g in ALLOWED_GPUS:
        v = view.get(g)
        if v is None:
            why.append(f"gpu{g} not listed")
            continue
        extra = assume.get(g, 0)
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
        cpus, nth, cnote = own_cores(g, exclude)
        if cpus is None:
            why.append(f"gpu{g} node{node(g)}: no {CORES} free own-node cores ({cnote})")
            continue
        tier = next(i for i, t in enumerate(PLACE_TIERS) if g in t)
        cands.append((tier, n, -(v["free"] - extra * JOB_MB), g, cpus, nth))
    if not cands:
        return "WAIT no GPU fits: " + "; ".join(why)
    tier, n, _, g, cpus, nth = sorted(cands)[0]
    sh = t1fix_share(view, exclude)
    trial = dict(assume)
    trial[g] = trial.get(g, 0) + 1
    f_after = feasible(view, trial, f"{exclude},{cpus}" if exclude else cpus)
    if f_after < sh["reserve_t1"]:
        return (f"WAIT equal share: after a start on gpu{g} {f_after} feasible slot(s) would remain < T1fix reserve "
                f"{sh['reserve_t1']} (T1fix pending {sh['t1fix_pending']}, running {sh['t1fix_running']}; Tier-2 running {sh['tier2_running']})")
    v = view[g]
    extra = assume.get(g, 0)
    return (f"OK {g} {cpus} {nth} - | gpu{g} (tier {tier + 1} = GPUs {'/'.join(map(str, PLACE_TIERS[tier]))}): {n} training(s) now "
            f"({' '.join(map(str, sorted(v['trainings']))) or 'none'}{f' + {extra} assumed' if extra else ''}) -> {n + 1} (cap "
            f"{MAX_TRAIN_PER_GPU}); {v['free'] - extra * JOB_MB} MiB free >= {JOB_MB + MIN_FREE_AFTER_MB + v['debt']} (debt {v['debt']}); "
            f"no MPS; CPU {cpus} ({CORES} own-NUMA cores); T1fix reserve {sh['reserve_t1']} <= {f_after} feasible after; "
            f"other candidates {[c[3] for c in sorted(cands)[1:]]}; excluded: {'; '.join(why) or '-'}")


def gpus_report() -> list[str]:
    try:
        view = observe()
    except Exception as error:  # noqa: BLE001
        return [f"GPU_ERR nvidia-smi failed: {error}"]
    return [f"GPU {g} trainings={len(v['trainings'])} ({' '.join(map(str, sorted(v['trainings'])))}) free={v['free']}MiB "
            f"mps={v['mps'] or 'no'} young={len(v['young'])} debt={v['debt']}"
            f"{'  (never used: GPU 4 is for tests)' if g not in ALLOWED_GPUS else ''}" for g, v in sorted(view.items())]


# ============================================================================================== state / launch guard
def job_state(job: str) -> tuple[str, str]:
    if native(job):
        s, d = F.job_state(job)       # the pilot's records (k3fix_pilot/pids.tsv, eval/<job>/post.json, CRASHED_<job>)
        mine = K / f"CRASHED_{job}"
        if s == "DEAD" and mine.exists():
            return "CRASHED", f"marker {mine}; {d}"
        if s == "TRAINED":
            pid = k3fix_driver_alive()
            if pid:
                return "POST_BY_K3FIX", f"{d}; the running k3fix driver (pid {pid}) runs its post-training step"
        return s, d
    return F2.job_state(job)


def launched_anywhere(job: str) -> list[str]:
    out = []
    for name, path in (("k3full", PIDS), ("k3fix", F.PIDS)):
        try:
            if any(line.split("\t")[0] == job for line in path.read_text().splitlines()):
                out.append(f"{name} {path.name}")
        except OSError:
            pass
    for m in (K / f"CRASHED_{job}", F.K / f"CRASHED_{job}"):
        if m.exists():
            out.append(f"marker {m}")
    return out


def launch_ok(job: str) -> str:
    j = JOBS[job]
    s, d = job_state(job)
    if s != "FRESH":
        return f"REFUSE state {s}: {d}"
    prev = launched_anywhere(job)
    if prev:
        return f"REFUSE earlier launch / marker: {', '.join(prev)}"
    run = RUNS / job
    if any(run.glob("model_*")) or any(run.glob("rsi_*")) or (run / "final.json").exists():
        return f"REFUSE {run} has checkpoints or final.json"
    if F2.find_trainers(job):
        return "REFUSE a trainer of this run dir is alive"
    p = parquet(job)
    got = _sha(p)
    if got != j["fix_sha"]:
        return f"REFUSE input parquet sha256 {str(got)[:16]} != pinned {j['fix_sha'][:16]}"
    g = json.loads(GATES_JSON.read_text())
    if j["ep"] not in g.get("pass_episodes", []):
        return f"REFUSE ep{j['ep']:02d} does not pass the gates"
    return f"OK {job}: FRESH, input {got[:16]}... pinned, gates PASS, data root {DATA_FIX}"


def state() -> int:
    for job in ORDER:
        s, d = job_state(job)
        print(f"JOB {job} {s} {d}")
    return 0


def crash(job: str, why: str) -> int:
    return F2.crash(job, why)          # writes k3full/CRASHED_<job> once (also for ep17/09/04: never in k3fix_pilot)


def post(job: str) -> int:
    if native(job):
        pid = k3fix_driver_alive()
        if pid:
            print(f"{job}: the running k3fix driver (pid {pid}) runs this post-training step; nothing done")
            return 0
        return F.post(job)              # pristine: k3fix_pilot/eval/<job>/, training GPU from k3fix_pilot/pids.tsv
    return F2.post(job)                 # this arm: k3full/eval/<job>/, training GPU from k3full/pids.tsv


# ============================================================================================== check / cfgcheck
def prereg_pins() -> dict[str, str]:
    """sha256 values the PREREG_v3 text pins, as lines 'pin <name> <sha256>'."""
    try:
        return dict(re.findall(r"^pin (\S+) ([0-9a-f]{64})\s*$", PREREG.read_text(), re.M))
    except OSError:
        return {}


def check() -> int:
    rows: list[tuple[str, bool, str]] = []

    def add(name, ok, detail):
        rows.append((name, bool(ok), detail))

    ok, why = prereg_status()
    add("prereg_v3", ok, why)
    pins = prereg_pins()
    for name, path in (("jobs.json", JOBS_JSON), ("gates_all30.json", GATES_JSON), ("cfgcheck.json", CFGCHECK),
                       ("k3fix_helper.py", K3FIX_HELPER), ("k3fix_build.py", ME / "scripts/k3fix_build.py"),
                       ("k3full_gates.py", ME / "scripts/k3full_gates.py"), ("k3full_helper.py", ME / "scripts/k3full_helper.py"),
                       ("k3full_driver.sh", ME / "scripts/k3full_driver.sh")):
        got = _sha(path)
        add(f"pin_{name}", name in pins and got == pins[name], f"{path.name} {str(got)[:16]}... vs PREREG_v3 pin {pins.get(name, '-')[:16]}")
    add("k3fix_helper_pinned", _sha(K3FIX_HELPER) == K3FIX_HELPER_SHA, "k3fix_helper.py = k3fix_pilot/build_files_v2.sha256")
    try:
        head = subprocess.run(["git", "-C", str(CODE), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        head = ""
    add("code_pin", head == PIN, f"{CODE} HEAD {head[:12] or '?'} (pin {PIN[:7]})")
    try:
        build = json.loads((F.K / "build.json").read_text())
    except (OSError, ValueError):
        build = {}
    dig = F.code_digest()
    add("code_tree_digest", build.get("code_digest") == dig, f"src/flash_chord + scripts digest {dig[:16]}... = k3fix_pilot/build.json")
    text = (CODE / "src/flash_chord/configs/experiment" / f"{F.EXP}.yaml").read_text()
    for key, val in (("deadline", 35156), ("zero_ramp", 3906), ("early_stop_help_zero_tail", 5860), ("max_iterations", 45000)):
        add(f"experiment_{key}", re.search(rf"^\s+{key}: {val}\s*$", text, re.M), f"{F.EXP}.yaml {key}: {val}")
    add("k3fix_root_ready", _sha(DATA_FIX / "READY") == READY_SHA, f"{DATA_FIX}/READY sha256 pinned {READY_SHA[:8]}")
    table = F.ready_table()
    bad = [p for p, s in table.items() if _sha(DATA_FIX / p) != s]
    add("k3fix_root_sha_table", len(table) == 30 and not bad, f"30 parquets match READY ({len(bad)} mismatch)")
    add("manifest_equals_raw", _sha(DATA_FIX / "whole_body/soma/track2_manifest.json") == _sha(RAW_DATA / "whole_body/soma/track2_manifest.json"),
        "K3fix root manifest = Tier-2 manifest")
    try:
        g = json.loads(GATES_JSON.read_text())
    except (OSError, ValueError):
        g = {}
    add("gates_all30", g.get("checks_ok") is True and g.get("gate_code_sha256") == GATE_CODE_SHA and g.get("k3fix_root_ready_sha256") == READY_SHA
        and len(g.get("pass_episodes", [])) + len(g.get("fail_episodes", [])) == 30,
        f"gates_all30.json checks ok; PASS {g.get('pass_episodes')} FAIL {g.get('fail_episodes')}")
    jj = load_jobs()
    want = [e for e in g.get("pass_episodes", []) if e not in EXISTING]
    add("jobs_cover_gate_pass", sorted(j["ep"] for j in jj["jobs"].values()) == sorted(want) and jj["fail_episodes"] == g.get("fail_episodes"),
        f"jobs = gate PASS {len(g.get('pass_episodes', []))} minus existing {sorted(EXISTING)} = {len(want)}")
    hz = early_hz()
    add("order_rule", list(jj["order"]) == [f"ep{e:02d}_k3fix_s42" for e in sorted(want, key=lambda e: (e == 4, hz[e], e))],
        "order = raw help_zero_since ascending, ties by episode, ep04 last")
    for ep, (src, run) in sorted(EXISTING.items()):
        seq = H.read_manifest(DATA_FIX)[ep][0]
        rel = f"whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet"
        rdir = (K3EASY_RUNS if src == "k3easy" else RUNS) / run
        ov = {}
        try:
            for x in F._load_yaml_list(rdir / "hydra/.hydra/overrides.yaml"):
                k, _, v = str(x).partition("=")
                ov[k.lstrip("+")] = v.strip("'\"")
        except Exception:  # noqa: BLE001
            pass
        tp = ov.get("task.parquet", "")
        same = bool(tp) and tp.endswith(rel) and _sha(tp) == table.get(rel) and ov.get("reset.seed") == "42"
        add(f"existing_ep{ep:02d}_input", same and ep in g.get("pass_episodes", []),
            f"{src} run {run}: task.parquet {tp.split('/data/')[-1].split('/')[0] if tp else '?'} sha256 = K3fix root parquet "
            f"({'byte-identical' if same else 'DIFFERENT'}), seed 42; gates PASS")
    try:
        cfc = json.loads(CFGCHECK.read_text())
    except (OSError, ValueError):
        cfc = {}
    add("cfgcheck", cfc.get("all_ok") is True and sorted(cfc.get("jobs", {})) == sorted(ORDER), f"{CFGCHECK.name}: all_ok {cfc.get('all_ok')}")
    man = H.read_manifest(DATA_FIX)
    db = json.loads((F.BUILD / "data_build.json").read_text())["episodes"]
    for job in ORDER:
        j = JOBS[job]
        ep = j["ep"]
        add(f"{job}_manifest", man.get(ep, ("",))[0] == j["seq"], f"manifest ep{ep} -> {man.get(ep)}")
        p = parquet(job)
        badp = H.check_parquet(p, j["seq"]) if p.is_file() else "missing"
        add(f"{job}_parquet_valid", badp is None, f"{badp or 'valid'}")
        rel = p.relative_to(DATA_FIX).as_posix()
        add(f"{job}_parquet_sha256", _sha(p) == j["fix_sha"] == table.get(rel) == db[f"{ep:02d}"]["root_sha256"], "= jobs.json = READY = data_build")
        add(f"{job}_raw_parquet", _sha(parquet(job, RAW_DATA)) == j["raw_sha"] != j["fix_sha"], "raw Tier-2 parquet pinned, differs from K3fix")
        fr = F.FRAME_REFS_FIX / f"episode_{ep:06d}.npz"
        add(f"{job}_frame_refs", fr.is_file() and _sha(fr) == db[f"{ep:02d}"]["frame_ref_sha256"], f"{fr.name} = data_build.json")
        sdir, rdir = p.parent.parent, parquet(job, RAW_DATA).parent.parent
        add(f"{job}_ground_plane", _sha(sdir / "ground_plane.json") == _sha(rdir / "ground_plane.json"), "= raw")
        name = f"{j['seq']}_support.usda"
        mine, ref = DATA_FIX / "whole_body/reconstructed_stage" / name, RAW_DATA / "whole_body/reconstructed_stage" / name
        add(f"{job}_support_file", mine.is_file() and _sha(mine) == _sha(ref), "= raw")
        rule = H.support_rule(DATA_FIX, j["seq"])
        add(f"{job}_support_rule", rule == ("true" if j["support"] else "false") == H.support_rule(RAW_DATA, j["seq"]), f"support {rule} as raw")
        raw_ov = [str(x) for x in F._load_yaml_list(Path(j["raw_hydra"]) / "overrides.yaml")]
        add(f"{job}_raw_hydra", f"output_dir={RAW_RUNS / f'ep{ep:02d}'}" in raw_ov and "reset.seed=42" in raw_ov
            and ("scene.support=false" in raw_ov) == (not j["support"]), f"{j['raw_hydra']}")
        fin = H.final_marker(RAW_RUNS / f"ep{ep:02d}") or {}
        add(f"{job}_raw_reference", fin.get("help_zero_since") == j["raw_hz"] == hz[ep] and fin.get("iteration") == j["raw_final"]
            and (RAW_RUNS / f"ep{ep:02d}" / fin.get("checkpoint", "-")).is_file(),
            f"raw final.json help zero {fin.get('help_zero_since')} it {fin.get('iteration')} (raw fallback checkpoint present)")
        if j["native"]:
            add(f"{job}_native_argv", train_argv(job) == F.train_argv(job) and F.jinfo(job)["fix_sha"] == j["fix_sha"],
                "k3fix pilot job: argv and input identical to k3fix_helper's")
    for ep in jj["fail_episodes"]:
        fin = H.final_marker(RAW_RUNS / f"ep{ep:02d}") or {}
        add(f"raw_fallback_ep{ep:02d}", bool(fin) and (RAW_RUNS / f"ep{ep:02d}" / fin.get("checkpoint", "-")).is_file(),
            f"gate FAIL -> raw C3r2_tier2 final.json model_{fin.get('iteration')}")
    bad = [r for r in rows if not r[1]]
    for name, okk, detail in rows:
        print(f"CHECK {name} {'ok' if okk else 'FAIL'} {detail}")
    print(f"CHECKS {len(rows) - len(bad)}/{len(rows)} ok" + (f"; FAILED: {' '.join(r[0] for r in bad)}" if bad else ""))
    return 0 if not bad else 1


def cfgcheck(write: bool) -> int:
    return F2.cfgcheck(write)            # the pilot's cfgcheck on this arm's job table -> k3full/cfgcheck.json


# ============================================================================================== verdict
def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def source_of(ep: int) -> dict:
    """The designated K3fix run of a gate-passing episode and where its records are."""
    setup()
    if ep in EXISTING:
        src, run = EXISTING[ep]
        if src == "k3easy":
            return {"source": "k3easy", "run": K3EASY_RUNS / run, "post": K3EASY / "eval" / run / "post.json",
                    "clean": K3EASY / "eval" / run / "clean_LOCAL_NOT_FOR_UPLOAD.json",
                    "markers": [K3EASY / f"CRASHED_{run}"], "name": f"C3r2_tier2_k3easy/{run}"}
        return {"source": "k3fix pilot", "run": RUNS / run, "post": F.EVROOT / run / "post.json",
                "clean": F.EVROOT / run / "clean_LOCAL_NOT_FOR_UPLOAD.json", "markers": [F.K / f"CRASHED_{run}", K / f"CRASHED_{run}"],
                "name": f"{ARM}/{run}"}
    job = f"ep{ep:02d}_k3fix_s42"
    if native(job):
        return {"source": "k3full (k3fix pilot job)", "run": RUNS / job, "post": F.EVROOT / job / "post.json",
                "clean": F.EVROOT / job / "clean_LOCAL_NOT_FOR_UPLOAD.json", "markers": [F.K / f"CRASHED_{job}", K / f"CRASHED_{job}"],
                "name": f"{ARM}/{job}"}
    return {"source": "k3full", "run": RUNS / job, "post": EVROOT / job / "post.json", "clean": EVROOT / job / "clean_LOCAL_NOT_FOR_UPLOAD.json",
            "markers": [K / f"CRASHED_{job}"], "name": f"{ARM}/{job}"}


def ep03_v2() -> str:
    """PREREG_v2's ep03 result (k3fix_helper verdict --no-write, run in-process with stdout captured)."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = F.verdict(False)
    except Exception as error:  # noqa: BLE001
        return f"UNKNOWN ({type(error).__name__})"
    if rc != 0:
        return "UNKNOWN (k3fix verdict refused)"
    m = re.search(r"^\[1\] ep03 (.*?): \(a\)", buf.getvalue(), re.M)
    return m.group(1) if m else "UNKNOWN (no ep03 line)"


def episode_row(ep: int, gates_pass: bool, raw_eval: dict, v2_ep03: str) -> dict:
    rr = raw_eval.get(ep, {})
    raw = {"run": f"C3r2_tier2/ep{ep:02d}", "final": (H.final_marker(RAW_RUNS / f"ep{ep:02d}") or {}).get("iteration"),
           "own_auc": F.fnum(rr.get("auc")), "own_sp": F.fnum(rr.get("sp_sr")), "own_mp": F.fnum(rr.get("mp_sr"))}
    row = {"episode": ep, "gate": "PASS" if gates_pass else "FAIL", "raw": raw}
    if not gates_pass:
        row.update(choice="RAW", why="gate FAIL (noisy-only input rule)", settled=True)
        return row
    s = source_of(ep)
    run = s["run"]
    fin = H.final_marker(run) if H.done_reason(run, ITERS) else None
    st = _load_json(s["post"])
    crashed = any(m.exists() for m in s["markers"])
    live = bool([p for p in H.all_pids() if H.is_python_trainer(H.read_cmdline(p)) and f"output_dir={run}" in H.read_cmdline(p)])
    exported = bool(fin and st.get("status") in ("done", "done_score_failed") and st.get("eval_status") == "evaluated"
                    and st.get("evaluated_iteration") == fin["iteration"] and st.get("export_stock_verified") is True
                    and F.fnum(st.get("export_max_rel_diff")) is not None and F.fnum(st.get("export_max_rel_diff")) <= EXPORT_TOL
                    and st.get("eval_reference_is_train_input") is True)
    tb = _load_json(ME / "tmp_analysis/tier2/t2watch/best" / s["name"] / "best.json")
    row["t2watch_best_secondary"] = {"iteration": tb.get("iteration"), "own_auc": F.fnum(tb.get("own_auc")),
                                     "note": "best-by-own-input (t2watch), secondary information only; the arm uses final.json"} if tb else None
    row.update(source=s["source"], k3fix_run=s["name"], finished=bool(fin), final_iteration=fin["iteration"] if fin else None,
               help_zero_since=fin.get("help_zero_since") if fin else None, post_status=st.get("status"), exported=exported,
               crash_marker=crashed, own_auc=F.fnum(st.get("eval_add_auc_own_input")), own_sp=F.fnum(st.get("eval_sp_sr")),
               own_mp=F.fnum(st.get("eval_mp_sr")), export_max_rel_diff=F.fnum(st.get("export_max_rel_diff")))
    extra_ok, extra_settled, extra_why = True, True, ""
    if ep == 3:
        extra_ok = v2_ep03.startswith("PASS")
        extra_settled = v2_ep03.startswith(("PASS", "FAIL", "INCOMPLETE"))
        extra_why = f"PREREG_v2 ep03 rule: {v2_ep03}"
        row["prereg_v2_ep03"] = v2_ep03
    if exported and extra_ok:
        row.update(choice="K3FIX", why=f"trained, finished (final.json model_{fin['iteration']}), exported (2e-4)"
                   + (f"; {extra_why}" if extra_why else ""), settled=True)
    elif (crashed and not fin and not live) or st.get("status") in ("export_refused", "failed") or (exported and extra_settled and not extra_ok):
        row.update(choice="RAW", why=("crashed without final.json" if crashed and not fin else f"post {st.get('status')}"
                                      if st.get("status") in ("export_refused", "failed") else extra_why) + " -> raw fallback", settled=True)
    else:
        row.update(choice="PENDING", why=("training" if live else "post-training step" if fin else "not launched yet"
                                          if not run.exists() else "waiting") + (f"; {extra_why}" if extra_why else ""), settled=False)
    return row


def verdict(write: bool) -> int:
    ok, why = prereg_status()
    if not ok:
        print(f"VERDICT REFUSED: {why} (the rules are fixed by PREREG_v3.md; nothing written)")
        return 2
    setup()
    g = json.loads(GATES_JSON.read_text())
    raw_eval = F.raw_eval_rows()
    v2 = ep03_v2()
    rows = [episode_row(ep, ep in g["pass_episodes"], raw_eval, v2) for ep in range(30)]
    n_k3 = sum(r["choice"] == "K3FIX" for r in rows)
    n_raw = sum(r["choice"] == "RAW" for r in rows)
    n_pend = sum(r["choice"] == "PENDING" for r in rows)
    final = n_pend == 0
    lines = [f"K3fix full arm ({ARM}) - arm composition by the pre-registered rule ({why})",
             f"written {now()}; {'FINAL' if final else 'NOT FINAL'}: K3fix {n_k3}, raw {n_raw}, pending {n_pend} of 30",
             "rule: K3fix iff gates G1-G4 PASS and the K3fix run is trained, finished (final.json) and exported within 2e-4 "
             "(+ for ep03 the PREREG_v2 ep03 PASS); else the raw C3r2_tier2 final.json checkpoint. Checkpoint = final.json.", ""]
    for r in rows:
        if r["gate"] == "FAIL":
            lines.append(f"ep{r['episode']:02d} RAW      gate FAIL -> {r['raw']['run']} model_{r['raw']['final']} | raw own-input AUC {r['raw']['own_auc']}")
            continue
        lines.append(f"ep{r['episode']:02d} {r['choice']:<8} {r['k3fix_run']} ({r['source']}): final {r['final_iteration']} help zero "
                     f"{r['help_zero_since']} post {r['post_status']} exported {r['exported']} | own-input AUC {r['own_auc']} SP/MP "
                     f"{r['own_sp']}/{r['own_mp']} vs raw own-input AUC {r['raw']['own_auc']} | t2watch best (secondary) "
                     f"{(r['t2watch_best_secondary'] or {}).get('iteration')}/{(r['t2watch_best_secondary'] or {}).get('own_auc')} | {r['why']}")
    own = [r["own_auc"] if r["choice"] == "K3FIX" else r["raw"]["own_auc"] for r in rows if r["choice"] != "PENDING"]
    if final and all(x is not None for x in own):
        lines.append(f"\nreport only (own-input references differ between K3fix and raw episodes): 30-episode own-input AUC mean "
                     f"{sum(own) / 30:.4f} vs raw arm own-input mean {sum(r['raw']['own_auc'] for r in rows) / 30:.4f}")
    lines.append("clean (leaderboard-style) comparison vs the raw 0.570: only in VERDICT_clean_LOCAL_NOT_FOR_UPLOAD.json (decides nothing)")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if write:
        F.atomic_write(K / "VERDICT.json", json.dumps({"written": now(), "prereg_check": why, "final": final, "counts":
                                                        {"k3fix": n_k3, "raw": n_raw, "pending": n_pend}, "episodes": rows}, indent=1, default=str))
        F.atomic_write(K / "VERDICT.txt", text)
        base = {}
        try:
            for r in csv.DictReader(EARLY.open()):
                if r["episode"].strip().isdigit():
                    base[int(r["episode"])] = F.fnum(r.get("t2train_mean_add_auc"))
        except (OSError, KeyError):
            pass
        per, diffs = [], []
        for r in rows:
            ep = r["episode"]
            k3c = _load_json(source_of(ep)["clean"]).get("clean_add_auc") if r["gate"] == "PASS" else None
            chosen = F.fnum(k3c) if r["choice"] == "K3FIX" else base.get(ep) if r["choice"] == "RAW" else None
            per.append({"episode": ep, "choice": r["choice"], "arm_clean_add_auc": chosen, "raw_mean3_clean_add_auc": base.get(ep),
                        "k3fix_run_clean_add_auc_1eval": F.fnum(k3c)})
            if r["choice"] == "K3FIX" and chosen is not None and base.get(ep) is not None:
                diffs.append(chosen - base[ep])
        arm = [p["arm_clean_add_auc"] for p in per]
        res = {"written": now(), "note": "LOCAL ONLY. Arm-level clean (leaderboard-style) comparison; decides nothing (no rule in "
                                         "PREREG_v3 reads it). Raw = C3r2_tier2 final.json, mean of 3 evaluations (early_table.csv); "
                                         "K3fix = ONE evaluation of final.json (post-training step).",
               "final": final, "episodes": per,
               "raw_arm_mean": (sum(base.values()) / len(base)) if len(base) == 30 else None,
               "k3fix_arm_mean": (sum(arm) / 30) if final and all(x is not None for x in arm) else None,
               "paired_diff_on_k3fix_episodes": {"n": len(diffs), "mean": (sum(diffs) / len(diffs)) if diffs else None}}
        if len(diffs) >= 3:
            rng = random.Random(20261022)
            boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(10000))
            res["paired_diff_on_k3fix_episodes"]["bootstrap95"] = [boots[249], boots[9749]]
            res["arm_mean_diff_over_30"] = sum(diffs) / 30
        F.atomic_write(K / "VERDICT_clean_LOCAL_NOT_FOR_UPLOAD.json", json.dumps(res, indent=1, default=str))
        print(f"wrote {K / 'VERDICT.json'}, VERDICT.txt, VERDICT_clean_LOCAL_NOT_FOR_UPLOAD.json")
    return 0


def heartbeat() -> int:
    parts = []
    for job in ORDER:
        s, _ = job_state(job)
        pr = F.progress(job)
        it = pr["it"] if pr["it"] is not None else pr["last_model"]
        parts.append(f"{job} {s} it={it} hz={pr['help_zero_iters']}")
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

    if sub == "build-jobs":
        return build_jobs("--no-write" not in rest)
    setup()
    if sub == "check":
        return check()
    if sub == "cfgcheck":
        return cfgcheck("--no-write" not in rest)
    if sub == "order":
        print("\n".join(ORDER))
        return 0
    if sub == "cmd":
        sys.stdout.write("\0".join(train_argv(job_of(opt("--job")))) + "\0")
        return 0
    if sub == "data":
        job_of(opt("--job"))
        print(DATA_FIX)
        return 0
    if sub == "native":
        print("yes" if native(job_of(opt("--job"))) else "no")
        return 0
    if sub == "launch-ok":
        line = launch_ok(job_of(opt("--job")))
        print(line)
        return 0 if line.startswith("OK") else 1
    if sub == "plan":
        assume = {int(a): int(b) for a, b in (x.split(":") for x in (opt("--assume", "") or "").split(",") if x)}
        print(plan(job_of(opt("--job")), opt("--exclude", "") or "", assume, "--no-yield" not in rest))
        return 0
    if sub == "yield":
        y, d = t3opus_state()
        print(f"{'YIELD' if y else 'NO_YIELD'} {d}")
        return 0
    if sub == "share":
        print(json.dumps(t1fix_share(observe(), opt("--exclude", "") or "")))
        return 0
    if sub == "gpus":
        print("\n".join(gpus_report()))
        return 0
    if sub == "state":
        return state()
    if sub == "trainer":
        print(" ".join(map(str, F2.find_trainers(job_of(opt("--job"))))))
        return 0
    if sub == "crash":
        return crash(job_of(opt("--job")), opt("--why", ""))
    if sub == "post":
        return post(job_of(opt("--job")))
    if sub == "verdict":
        return verdict("--no-write" not in rest)
    if sub == "heartbeat":
        return heartbeat()
    print(f"unknown subcommand {sub}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
