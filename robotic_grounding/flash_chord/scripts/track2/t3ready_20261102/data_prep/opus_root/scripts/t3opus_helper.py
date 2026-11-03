#!/usr/bin/env python3
"""t3opus_helper.py: checks, retarget / training plans, state, post-training step and verdict of scripts/t3opus_driver.sh,
the Tier-3 opus easy-clip trainability test (arm C3r2_tier3_opus; pre-registration tmp_analysis/tier3/opus_easy/PREREG.md).

  C3r2 exactly as the Tier-1 C3r2 arm (C3r2_wide) and the raw Tier-2 arm (C3r2_tier2): code/fc_r2 @61fa5d3, experiment
  g1_recon_body_coord_rsi_r2, seed 42, max_iterations 45000, experiment-default RSI deadline 35156 / zero_ramp 3906 /
  help-zero early stop 5860, scene.support=false (floor-only clips, the same rule). The only difference is the data root
  data/human_motion_data_tier3_opus (Han Zhou's opus Tier-3 reconstruction, retargeted by this test).
  Wave 1: ep08 skinny_wood_chair, ep23 toy_stroller, ep22 step_ladder. Wave 2 (later, only by 'run --wave 2'): ep00
  tall_bar_stool, ep18 blue_trash_can, ep10 white_desk. All floor-only.

Run with conda_envs/flashchord/bin/python -I -B. It never starts a training and never signals a process except its own
evaluation session on timeout. It writes only under tmp_analysis/tier3/opus_easy/. nvidia-smi / /proc / CPU-slice
primitives come from scripts/track2_queue_tier2v2_helper.py (imported read-only).

Subcommands:
  check                    driver preflight (PREREG sha256, code pins + tree digest, experiment values, data READY,
                           manifest, frame references, floor-only rule, Tier-1/Tier-2 references, cfgcheck, scripts)
  cfgcheck [--no-write]    hydra-compose the raw Tier-2 and Tier-1 overrides (must reproduce their saved config.yaml) and
                           this test's argv per clip; diff; writes opus_easy/cfgcheck.json
  wave-clips --waves W     the clips of wave(s) W (comma list)
  cmd --ep NN              the training argv, NUL-separated
  plan --ep NN [--exclude CPUS] [--remaining N] [--assume g:n,..] [--waves W]   "OK <gpu> <cpus> <nth> - | why" | "WAIT why"
  rplan [--exclude CPUS]   GPU/CPU for one retarget: "OK <gpu> <uuid> <cpus> <nth> | why" | "WAIT why"
  launch-ok --ep NN        exit 0 iff the parquet's sha256 equals the retarget lock and its rcheck passed
  gpus                     per-GPU view
  state --waves W          one line per clip "CLIP NN STATE detail"
  trainer --ep NN          live trainer pid(s) of that run dir
  crash --ep NN --why T    write opus_easy/CRASHED_epNN once (no relaunch is ever made)
  rfail --ep NN --why T    write opus_easy/RETARGET_FAILED_epNN once
  post --ep NN             export (1e-4; 2e-4 if refused, recorded) + ONE official evaluation of final.json's checkpoint
                           against its own input on its training GPU, then the LOCAL clean diagnostic
  verdict [--no-write]     the PREREG rule; exit 2 (nothing written) when PREREG.sha256 does not match
  heartbeat                one status line
  build-record             pin the code-tree digest (once, at build)
"""
from __future__ import annotations

import csv, difflib, fcntl, glob, hashlib, json, math, os, re, signal, subprocess, sys, time
from pathlib import Path

sys.dont_write_bytecode = True
ME = Path("/root/autodl-tmp/haoran.zhang")
sys.path.insert(0, str(ME / "scripts"))
import track2_queue_tier2v2_helper as H  # noqa: E402  read-only primitives

ARM = "C3r2_tier3_opus"
EXP = "g1_recon_body_coord_rsi_r2"
ITERS = 45000
SEED = 42
A = ME / "tmp_analysis/tier3/opus_easy"
RUNS = ME / "fc_runs" / ARM
EVROOT = A / "eval"
CODE = ME / "code/fc_r2"
PIN = "61fa5d31ff8054bf1e5cf5375ef0957e54d2729b"
NV = ME / "code/nv-v2d-challenge-codebase/robotic_grounding"
NV_PIN = "f2923d61817e61eb37d939e0f75e5b4a7aaa5cbf"
DATA = ME / "data/human_motion_data_tier3_opus"
FRAME_REFS = DATA / "frame_references_opus"
T1_RUNS = ME / "fc_runs/C3r2_wide"
T2_RUNS = ME / "fc_runs/C3r2_tier2"
SHARPA = CODE / "logs/rl/sharpa"
PREREG = A / "PREREG.md"
PREREG_SHA = A / "PREREG.sha256"
PIDS = A / "pids.tsv"
RDIR = A / "retarget"
LOCK = RDIR / "parquet_lock.json"
CFGCHECK = A / "cfgcheck.json"
KIT = ME / "v2d_submission_kit_src/v2d_submission_kit"
STOCK = ME / "code/video_to_data_8c778f9/robotic_grounding/flash_chord"
DATASET = ME / "data/video_to_data_challenge"
KPY = ME / "conda_envs/v2dkit/bin/python"
FCPY = ME / "conda_envs/flashchord/bin/python"
CLEAN_SCORER = ME / "scripts/track2_tier2_clean_scores.py"
TIER1_CSV = ME / "fc_runs/eval_round2/C3r2_wide_final.csv"
BASELINE_CSV = ME / "tmp_analysis/tier2/early_scores/early_table.csv"
INPUT_LOCAL = A / "input_quality_LOCAL_NOT_FOR_UPLOAD.json"
RUNIN_EVALLOCK = ME / "fc_runs/queue_runin/evallock"  # opened read-only, never created
OWN_EVALLOCK = A / "evallock"
COMMIT = "https://github.com/local-placeholder/not-uploaded/commit/0000000000000000000000000000000000000000"
EXPORT_TOLS = (1e-4, 2e-4)  # the Tier-3 pilot's rule: 1e-4, redone at 2e-4 (recorded) only if the exporter refuses
EVAL_TIMEOUT = 2400.0
SCORE_TIMEOUT = 1800.0
RETRY_DELAY = 600.0
MAX_TRIES = 2
MAX_POST_STARTS = 5
MIN_FINAL_AGE = 120.0
STARTING_GRACE_S = 180.0
ALLOWED_GPUS = (0, 1, 2, 3, 5, 6, 7)  # never GPU 4
PLACE_TIERS = ((3, 5, 6, 7), (0, 1, 2))  # NUMA-1 GPUs first; NUMA 0 (the Tier-1 fork arms' node) only if none fits
MAX_TRAIN_PER_GPU = 3
JOB_MB = 16896
MIN_FREE_AFTER_MB = 20480
CORES = 2  # physical cores (+HT siblings) per job = the Tier-1 fork launcher's / K3fix pilot's slot size
RETARGET_MB = 10240
EVAL_MIN_FREE_MB = 4096
UNKNOWN_TRAIN_AGE_S = 1800
RAMP_S = 900
SPI = 4096 * 24
HZ_LIMIT = 30000
HZ_STRONG = 20000
TERMINAL = ("done", "done_score_failed", "export_refused", "failed")
CFG_ALLOWED = {"task.parquet", "output_dir", "logging.mode", "logging.project", "logging.experiment", "logging.local_jsonl"}
CFG_ALLOWED_T1 = CFG_ALLOWED | {"logging.run_name"}
CLIPS = {  # t1_hz / t2_hz = final.json help_zero_since of C3r2_wide / C3r2_tier2 (training-only references)
    8: {"seq": "2026-06-05_15-08-08_skinny_wood_chair_push_10", "obj": "skinny_wood_chair", "wave": 1, "frames": 708, "t1_hz": 11412, "t2_hz": 14737},
    23: {"seq": "2026-06-03_15-08-03_toy_stroller_back_forth_ground_04", "obj": "toy_stroller", "wave": 1, "frames": 946, "t1_hz": 12811, "t2_hz": 14163},
    22: {"seq": "2026-05-22_16-52-36_step_ladder_drag_03", "obj": "step_ladder", "wave": 1, "frames": 719, "t1_hz": 12470, "t2_hz": 39062},
    0: {"seq": "2026-06-04_14-16-23_tall_bar_stool_tip_up_04", "obj": "tall_bar_stool", "wave": 2, "frames": 778, "t1_hz": 11863, "t2_hz": 35183},
    18: {"seq": "2026-05-06_10-00-19_blue_trash_can_drag_sideways_08", "obj": "blue_trash_can", "wave": 2, "frames": 858, "t1_hz": 13078, "t2_hz": 39062},
    10: {"seq": "2026-05-01_11-36-40_white_desk_lower_to_ground_10", "obj": "white_desk", "wave": 2, "frames": 794, "t1_hz": 15572, "t2_hz": 28407},
}
ORDER = (8, 23, 22, 0, 18, 10)
WAVES = (1, 2)


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


def ep_of(text) -> int:
    e = int(text)
    if e not in CLIPS:
        raise SystemExit(f"episode {e} is not part of this test ({' '.join(f'{x:02d}' for x in ORDER)})")
    return e


def waves_of(text: str | None) -> set[int]:
    if not text:
        return set()
    out = {int(x) for x in str(text).split(",") if x.strip()}
    if not out <= set(WAVES):
        raise SystemExit(f"unknown wave(s) {sorted(out - set(WAVES))}")
    return out


def seq_dir(ep: int) -> Path:
    return DATA / "whole_body/soma" / f"sequence_id={CLIPS[ep]['seq']}"


def parquet(ep: int) -> Path:
    return seq_dir(ep) / "robot_name=g1/data.parquet"


def run_dir(ep: int) -> Path:
    return RUNS / f"ep{ep:02d}"


def prereg_status() -> tuple[bool, str]:
    if not PREREG.is_file() or not PREREG_SHA.is_file():
        return False, f"{PREREG.name} or {PREREG_SHA.name} missing"
    words = PREREG_SHA.read_text().split()
    want = words[0] if words else ""
    got = sha256(PREREG)
    return got == want, f"sha256(PREREG.md) {str(got)[:16]}... {'==' if got == want else '!='} PREREG.sha256 {want[:16]}..."


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


def train_argv(ep: int) -> list[str]:
    """The raw Tier-2 arm's command (C3r2_tier2: experiment, task.parquet, reset.seed, output_dir, max_iterations, logging,
    scene.support=false for floor-only) with the Tier-3 opus parquet, this arm's output_dir and offline logging, plus
    hydra.run.dir / +logging.local_jsonl (as the Tier-2 K3 tests); none of these changes training (cfgcheck)."""
    c, out = CLIPS[ep], run_dir(ep)
    return ["python", "scripts/train_rl.py", f"experiment={EXP}", f"task.parquet='{parquet(ep)}'", f"reset.seed={SEED}",
            f"output_dir={out}", f"training.max_iterations={ITERS}", "logging.mode=offline", "logging.entity=juicer_test",
            "logging.project=v2d-track2-tier3", f"logging.experiment={ARM}", f"logging.run_name=ep{ep:02d}_{c['obj']}",
            "logging.upload_checkpoints=false", "scene.support=false", f"hydra.run.dir={out}/hydra", "+logging.local_jsonl=true"]


def read_pids() -> dict[int, dict]:
    out: dict[int, dict] = {}
    try:
        for line in PIDS.read_text().splitlines():
            f = line.split("\t")
            if len(f) >= 6 and f[0].isdigit():
                out[int(f[0])] = {"pid": int(f[1]) if f[1].isdigit() else None, "gpu": f[2], "cpus": f[3], "nth": f[4],
                                  "t": int(f[5]) if f[5].isdigit() else 0}
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


def find_retargets() -> list[int]:
    out = []
    for pid in H.all_pids():
        cmd = " ".join(H.read_cmdline(pid))
        if "soma_to_g1.py" in cmd and str(DATA) in cmd and H.alive(pid):
            out.append(pid)
    return out


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
        with (A / "post.log").open("a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def lock_entry(ep: int) -> dict | None:
    try:
        return json.loads(LOCK.read_text()).get(f"{ep:02d}")
    except (OSError, ValueError):
        return None


def rcheck_ok(ep: int) -> tuple[bool, str]:
    ent = lock_entry(ep)
    p = parquet(ep)
    if not p.is_file():
        return False, "no parquet"
    if not ent:
        return False, "no retarget lock entry"
    try:
        rc = json.loads((RDIR / f"ep{ep:02d}_rcheck.json").read_text())
    except (OSError, ValueError):
        return False, "no rcheck json"
    got = sha256(p)
    ok = rc.get("all_ok") is True and rc.get("parquet_sha256") == ent["sha256"] == got
    return ok, f"parquet sha256 {str(got)[:16]}... lock {ent['sha256'][:16]}... rcheck all_ok {rc.get('all_ok')}"


# ============================================================================================== GPU view / plans
def observe() -> dict[int, dict]:
    """As k3fix_helper.observe: per GPU trainings of anybody, evaluations, free memory, MPS, and the ramp debt of this
    test's trainers younger than RAMP_S (environment T3OPUS=1)."""
    st = H.smi()
    pids = H.all_pids()
    uuid_to_idx = {v["uuid"]: k for k, v in st["gpus"].items()}
    mps = H.mps_gpus(st, pids)
    view = {g: {"free": v["free"], "total": v["total"], "uuid": v["uuid"], "trainings": [], "evals": [], "others": [],
                "mps": mps.get(g), "debt": 0} for g, v in st["gpus"].items()}
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
        if H.read_environ(pid).get("T3OPUS") == "1" and (H.proc_age(pid) or 0.0) < RAMP_S:
            view[g]["debt"] += max(0, JOB_MB - mem.get(pid, 0))
    return view


def own_node_cores(gpu: int, exclude: str) -> list[int]:
    allc, _, note = H.cpu_slice(gpu, exclude, 104, CORES)
    if not allc or allc == "none" or note == "cross-NUMA":
        return []
    return [int(x) for x in allc.split(",")[0::2]]


def parse_assume(text: str | None) -> dict[int, int]:
    out = {}
    for part in (text or "").split(","):
        if ":" in part:
            g, n = part.split(":")
            out[int(g)] = int(n)
    return out


def plan(ep: int, exclude: str, assume: dict[int, int] | None = None) -> str:
    """GPU for a new training: allowed GPUs (never 4), no MPS, < 3 trainings of anybody before the start, >= JOB_MB +
    20 GB free (+ ramp debt of this test's young trainers), >= CORES free physical cores on the GPU's own NUMA node (never
    borrowed); placement tiers NUMA 1 (3/5/6/7) first, NUMA 0 (0/1/2) only if no NUMA-1 GPU fits; then fewest trainings,
    most free memory, lowest index. CPU slice: CORES physical cores + HT siblings. assume {gpu: n}: dry run only."""
    try:
        view = observe()
    except Exception as error:  # noqa: BLE001
        return f"WAIT nvidia-smi failed ({type(error).__name__}: {str(error)[:100]})"
    why, cands, node_cores = [], [], {}
    for g in ALLOWED_GPUS:
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
        if len(node_cores[node]) < CORES:
            why.append(f"gpu{g} node{node} < {CORES} free cores")
            continue
        tier = next(i for i, t in enumerate(PLACE_TIERS) if g in t)
        cands.append((tier, n, -(v["free"] - extra * JOB_MB), g))
    if not cands:
        return "WAIT no GPU fits: " + "; ".join(why)
    tier, n, _, g = sorted(cands)[0]
    extra = (assume or {}).get(g, 0)
    cores = node_cores[0 if g <= 2 else 1][:CORES]
    cpus, nth = ",".join(f"{c},{c + 104}" for c in cores), 2 * len(cores)
    v = view[g]
    return (f"OK {g} {cpus} {nth} - | gpu{g} (tier {tier + 1}): {n} training(s) now ({' '.join(map(str, sorted(v['trainings']))) or 'none'}"
            f"{f' + {extra} assumed' if extra else ''}) -> {n + 1} after the start (cap {MAX_TRAIN_PER_GPU}); {v['free'] - extra * JOB_MB} MiB "
            f"free >= {JOB_MB + MIN_FREE_AFTER_MB + v['debt']}; no MPS; CPU {len(cores)} own-NUMA cores; other candidates "
            f"{[c[3] for c in sorted(cands)[1:]]}; excluded: {'; '.join(why) or '-'}")


def rplan(exclude: str) -> str:
    """GPU for one retarget (brief, not a training): allowed GPUs (never 4), no MPS, >= RETARGET_MB + 20 GB free, no other
    retarget of this test running; NUMA-1 GPUs first, then the most free memory; CPU: CORES own-NUMA cores."""
    if find_retargets():
        return f"WAIT a retarget of this test is running (pid {' '.join(map(str, find_retargets()))})"
    try:
        view = observe()
    except Exception as error:  # noqa: BLE001
        return f"WAIT nvidia-smi failed ({type(error).__name__})"
    cands, why, node_cores = [], [], {}
    for g in ALLOWED_GPUS:
        v = view.get(g)
        if v is None or v["mps"] or v["free"] < RETARGET_MB + MIN_FREE_AFTER_MB:
            why.append(f"gpu{g} {'MPS' if v and v['mps'] else (str(v['free']) + ' MiB' if v else 'missing')}")
            continue
        node = 0 if g <= 2 else 1
        if node not in node_cores:
            node_cores[node] = own_node_cores(g, exclude)
        if len(node_cores[node]) < CORES:
            why.append(f"gpu{g} node{node} < {CORES} free cores")
            continue
        tier = next(i for i, t in enumerate(PLACE_TIERS) if g in t)
        cands.append((tier, -v["free"], g))
    if not cands:
        return "WAIT no GPU fits a retarget: " + "; ".join(why)
    _, negfree, g = sorted(cands)[0]
    cores = node_cores[0 if g <= 2 else 1][:CORES]
    return (f"OK {g} {view[g]['uuid']} {','.join(f'{c},{c + 104}' for c in cores)} {2 * len(cores)} | gpu{g}: {-negfree} MiB free >= "
            f"{RETARGET_MB + MIN_FREE_AFTER_MB}; {len(view[g]['trainings'])} training(s); no MPS")


def gpus_report() -> list[str]:
    try:
        view = observe()
    except Exception as error:  # noqa: BLE001
        return [f"GPU_ERR nvidia-smi failed: {error}"]
    return [f"GPU {g} trainings={len(v['trainings'])} ({' '.join(map(str, sorted(v['trainings'])))}) evals={len(v['evals'])}"
            f" others={len(v['others'])} free={v['free']}MiB mps={v['mps'] or 'no'} debt={v['debt']}"
            f"{'  (never used: GPU 4 is for tests)' if g not in ALLOWED_GPUS else ''}" for g, v in sorted(view.items())]


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


def hydra_dir(arm_runs: Path, ep: int) -> Path | None:
    """The .hydra dir under code/fc_r2/logs/rl/sharpa whose config.yaml output_dir is <arm_runs>/epNN (newest)."""
    want = str(arm_runs / f"ep{ep:02d}")
    hits = []
    for d in sorted(SHARPA.glob(f"*_ep{ep:02d}_*/.hydra")):
        try:
            text = (d / "config.yaml").read_text()
        except OSError:
            continue
        m = re.search(r"^output_dir: (.+)$", text, re.M)
        if m and m.group(1).strip() == want:
            hits.append(d)
    return hits[-1] if hits else None


def cfgcheck(write: bool) -> int:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    res = {"written": now(), "code": str(CODE), "pin": PIN, "allowed_vs_tier2": sorted(CFG_ALLOWED),
           "allowed_vs_tier1": sorted(CFG_ALLOWED_T1), "episodes": {}}
    ok_all = True
    with initialize_config_dir(config_dir=str(CODE / "src/flash_chord/configs"), version_base="1.3"):
        for ep in ORDER:
            mine_ov = [a for a in train_argv(ep)[2:] if not a.startswith("hydra.")]
            mine = _flat(OmegaConf.to_container(compose(config_name="train", overrides=mine_ov), resolve=False))
            row = {"argv": train_argv(ep)[2:]}
            ok = mine.get("task.parquet") == str(parquet(ep)) and mine.get("output_dir") == str(run_dir(ep)) \
                and mine.get("logging.mode") == "offline" and mine.get("scene.support") is False \
                and mine.get("reset.seed") == SEED and mine.get("training.max_iterations") == ITERS
            for tag, runs, allowed in (("tier2_raw", T2_RUNS, CFG_ALLOWED), ("tier1", T1_RUNS, CFG_ALLOWED_T1)):
                hyd = hydra_dir(runs, ep)
                if hyd is None:
                    row[tag] = {"ok": False, "why": f"no hydra dir for {runs}/ep{ep:02d}"}
                    ok = False
                    continue
                ov = [str(x) for x in _load_yaml_list(hyd / "overrides.yaml")]
                saved = _flat(OmegaConf.to_container(OmegaConf.load(hyd / "config.yaml"), resolve=False))
                comp = _flat(OmegaConf.to_container(compose(config_name="train", overrides=ov), resolve=False))
                repro = comp == saved
                diff = [{"key": k, tag: saved.get(k, "<absent>"), "t3opus": mine.get(k, "<absent>")}
                        for k in sorted(set(saved) | set(mine)) if saved.get(k, "<absent>") != mine.get(k, "<absent>")]
                dk = {d["key"] for d in diff}
                good = repro and dk <= allowed and ("scene.support=false" in ov)
                ok &= good
                row[tag] = {"ok": good, "hydra_dir": str(hyd), "compose_reproduces_saved_config": repro, "n_keys": len(saved),
                            "diff": diff, "overrides": ov,
                            "overrides_unified_diff": list(difflib.unified_diff(ov, mine_ov, tag, "t3opus", lineterm="", n=20))}
                print(f"== ep{ep:02d} {CLIPS[ep]['obj']} vs {tag} ({hyd.parent.name}): compose reproduces saved config {repro} "
                      f"({len(saved)} keys); differ in {len(diff)} key(s), all allowed {dk <= allowed}; raw had scene.support=false "
                      f"{'scene.support=false' in ov} -> {'ok' if good else 'FAIL'}")
                for d in diff:
                    print(f"   {d['key']}: {d[tag]!r} -> {d['t3opus']!r}")
            row["ok"] = bool(ok)
            ok_all &= ok
            res["episodes"][f"{ep:02d}"] = row
    res["all_ok"] = bool(ok_all)
    print(f"cfgcheck: {'OK' if ok_all else 'FAILED'}")
    if write:
        atomic_write(CFGCHECK, json.dumps(res, indent=1, default=str))
        print(f"wrote {CFGCHECK}")
    return 0 if ok_all else 1


def check() -> int:
    rows: list[tuple[str, bool, str]] = []

    def add(name, ok, detail):
        rows.append((name, bool(ok), detail))

    ok, why = prereg_status()
    add("prereg", ok, why)
    try:
        head = subprocess.run(["git", "-C", str(CODE), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60).stdout.strip()
        nvh = subprocess.run(["git", "-C", str(NV), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        head = nvh = ""
    add("code_pin", head == PIN, f"{CODE} HEAD {head[:12] or '?'} (pin {PIN[:7]})")
    add("nv_pin", nvh == NV_PIN, f"{NV} HEAD {nvh[:12] or '?'} (retarget pin {NV_PIN[:7]}, the Tier-3 pilot's)")
    try:
        build = json.loads((A / "build.json").read_text())
    except (OSError, ValueError):
        build = {}
    dig = code_digest()
    add("code_tree_digest", build.get("code_digest") == dig, f"{dig[:16]}... {'==' if build.get('code_digest') == dig else '!='} build.json")
    expf = CODE / "src/flash_chord/configs/experiment" / f"{EXP}.yaml"
    text = expf.read_text() if expf.is_file() else ""
    for key, val in (("deadline", 35156), ("zero_ramp", 3906), ("early_stop_help_zero_tail", 5860), ("max_iterations", 45000)):
        add(f"experiment_{key}", re.search(rf"^\s+{key}: {val}\s*$", text, re.M), f"{EXP}.yaml {key}: {val}")
    add("data_ready", (DATA / "READY").is_file(), f"{DATA}/READY")
    try:
        man = json.loads((DATA / "whole_body/soma/track2_manifest.json").read_text())
        seq_of = {int(r["episode_index"]): (r["sequence_id"], int(r["frames"])) for r in man.values()}
    except (OSError, ValueError, KeyError):
        seq_of = {}
    try:
        cfc = json.loads(CFGCHECK.read_text())
    except (OSError, ValueError):
        cfc = {}
    add("cfgcheck", cfc.get("all_ok") is True, f"{CFGCHECK.name}: all_ok {cfc.get('all_ok')}")
    for f in (ME / "scripts/t3opus_retarget.sh", ME / "scripts/t3opus_rcheck.py", CLEAN_SCORER, KPY, FCPY):
        add(f"file_{f.name}", f.is_file(), str(f))
    for ep in ORDER:
        c, tag = CLIPS[ep], f"ep{ep:02d}"
        add(f"{tag}_manifest", seq_of.get(ep) == (c["seq"], c["frames"]), f"manifest {seq_of.get(ep)}")
        fr = FRAME_REFS / f"episode_{ep:06d}.npz"
        add(f"{tag}_frame_ref", fr.is_file(), f"{fr.name}")
        stage = DATA / "whole_body/reconstructed_stage"
        rule = H.support_rule(DATA, c["seq"])
        add(f"{tag}_floor_only", rule == "false" == H.support_rule(ME / "data/human_motion_data_tier2", c["seq"])
            and (stage / f"{c['seq']}_support.usda").is_file() and not (stage / f"{c['seq']}_g1_support.usda").exists(),
            f"support rule {rule} (Tier-2 root: {H.support_rule(ME / 'data/human_motion_data_tier2', c['seq'])}) -> scene.support=false")
        for arm, runs, key in (("t1", T1_RUNS, "t1_hz"), ("t2", T2_RUNS, "t2_hz")):
            fin = H.final_marker(runs / f"ep{ep:02d}") or {}
            add(f"{tag}_{arm}_reference", fin.get("help_zero_since") == c[key], f"{runs.name} help_zero_since {fin.get('help_zero_since')} (PREREG {c[key]})")
    bad = [r for r in rows if not r[1]]
    for name, ok, detail in rows:
        print(f"CHECK {name} {'ok' if ok else 'FAIL'} {detail}")
    print(f"CHECKS {len(rows) - len(bad)}/{len(rows)} ok" + (f"; FAILED: {' '.join(r[0] for r in bad)}" if bad else ""))
    return 0 if not bad else 1


# ============================================================================================== state / crash
def clip_state(ep: int, waves: set[int]) -> tuple[str, str]:
    run = run_dir(ep)
    tr = find_trainers(ep)
    pr = progress(ep)
    pinfo = f"it={pr['it'] if pr['it'] is not None else pr['last_model']} help_zero_iters={pr['help_zero_iters']}"
    if tr:
        return "RUNNING", f"pid={','.join(map(str, tr))} {pinfo} gpu={read_pids().get(ep, {}).get('gpu', '?')}"
    why = H.done_reason(run, ITERS)
    if why:
        st = load_post(ep)
        if lock_held(A / "post" / f"ep{ep:02d}.lock"):
            return "POST_RUNNING", f"{why}; post {st.get('status', '?')}"
        if st.get("status") in TERMINAL:
            return "POST_DONE", f"{why}; post {st['status']}"
        if st.get("status") == "retry" and float(st.get("retry_after", 0)) > time.time():
            return "POST_RETRY_WAIT", f"{why}; post retry after {time.strftime('%H:%M:%S', time.localtime(float(st['retry_after'])))}"
        return "TRAINED", f"{why}; post {st.get('status', 'not started')}"
    if (A / f"CRASHED_ep{ep:02d}").exists():
        return "CRASHED", f"marker CRASHED_ep{ep:02d}; {pinfo}"
    launched = read_pids().get(ep)
    has_files = pr["models"] > 0 or (run / "final.json").exists() or any(run.glob("rsi_*.safetensors"))
    if launched and time.time() - launched["t"] < STARTING_GRACE_S and not has_files:
        return "STARTING", f"launched {int(time.time() - launched['t'])} s ago (pid {launched['pid']})"
    if has_files or launched:
        return "DEAD", (f"no live trainer, no final.json; {pr['models']} model_* file(s), last {pr['last_model']}; "
                        f"{'launched by this test at ' + time.strftime('%F %T', time.localtime(launched['t'])) if launched else 'never launched by this test'}")
    # not launched: retarget stage
    if (A / f"RETARGET_FAILED_ep{ep:02d}").exists():
        return "RETARGET_FAILED", f"marker RETARGET_FAILED_ep{ep:02d}"
    if lock_held(RDIR / f"ep{ep:02d}.lock"):
        return "RETARGETING", "retarget/check running"
    in_wave = CLIPS[ep]["wave"] in waves
    p = parquet(ep)
    if not p.is_file():
        return ("NEEDS_RETARGET" if in_wave else "DEFERRED"), f"no parquet yet (wave {CLIPS[ep]['wave']})"
    ok, detail = rcheck_ok(ep)
    if not ok:
        ent = lock_entry(ep)
        if ent and ent["sha256"] != sha256(p):
            return "PARQUET_CHANGED", f"parquet sha256 differs from the retarget lock ({detail})"
        return ("NEEDS_RCHECK" if in_wave else "DEFERRED"), detail
    return ("FRESH" if in_wave else "DEFERRED"), f"retargeted and checked ({detail}); no run files (wave {CLIPS[ep]['wave']})"


def state(waves: set[int]) -> int:
    for ep in ORDER:
        s, d = clip_state(ep, waves)
        print(f"CLIP {ep:02d} {s} {d}")
    return 0


def marker(name: str, ep: int, why: str) -> int:
    path = A / f"{name}_ep{ep:02d}"
    if path.exists():
        return 0
    log = RUNS / "logs" / f"ep{ep:02d}.log" if name == "CRASHED" else RDIR / f"ep{ep:02d}_retarget.log"
    try:
        tail = log.read_text(errors="replace").splitlines()[-30:]
    except OSError:
        tail = []
    pr = progress(ep)
    text = (f"{name} ep{ep:02d} {CLIPS[ep]['obj']} detected {now()}\n{why}\nlast model {pr['last_model']}, last logged iteration "
            f"{pr['it']}, help_zero_iterations {pr['help_zero_iters']}\nNo automatic relaunch / retry (PREREG): the clip is "
            f"INCOMPLETE unless the user decides otherwise (disclosed).\n--- last lines of {log}\n" + "\n".join(tail) + "\n")
    atomic_write(path, text)
    print(f"wrote {path}")
    return 0


# ============================================================================================== post-training step
def child_env(gpu: int, ep: int) -> dict:
    sp = ME / "conda_envs/flashchord/lib/python3.11/site-packages/nvidia"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CUDA_MPS_", "WANDB_", "RUNIN_", "TIER2V2_", "K3EASY", "K3FIX", "T3OPUS"))
           and k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "TIER2_GPU")}
    env.update({
        "PATH": f"{ME}/conda_envs/flashchord/bin:{env.get('PATH', '/usr/bin:/bin')}",
        "LD_LIBRARY_PATH": f"{sp}/cu13/lib:{sp}/cudnn/lib:{sp}/nccl/lib:{sp}/nvshmem/lib"
                           + (f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else ""),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": f"{ME}/.cache/pycache", "XDG_CACHE_HOME": f"{ME}/.cache", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(gpu), "HUMAN_MOTION_DATA_DIR": str(DATA), "OMP_NUM_THREADS": "4",
        "T3OPUS_EVAL": "1", "T3OPUS_EVAL_EP": f"{ep:02d}",
    })
    return env


def eval_command(ep: int, gpu: int, w: Path, tol: float) -> list[str]:
    """k3easy_helper.eval_command (= track2_tier2v2_post.py) with this arm, the Tier-3 manifest and the given tolerance."""
    command = [str(FCPY), "-B", str(CODE / "scripts/track2/eval_all.py"), "--arm-dir", str(w), "--arm", ARM,
               "--checkpoint-from-final", "--iteration", str(ITERS), "--episodes", str(ep), "--gpus", str(gpu),
               "--jobs-per-gpu", "1", "--world-count", "1", "--export", "yes", "--export-jobs", "1",
               "--export-tolerance", repr(tol), "--skip-pack", "--kit-root", str(KIT),
               "--stock-runtime", str(STOCK), "--dataset-root", str(DATASET),
               "--manifest", f"{DATA}/whole_body/soma/track2_manifest.json", "--pack-python", str(KPY),
               "--results", str(w / "results"), "--code", str(CODE), "--python", str(FCPY)]  # fmt: skip
    return ["nice", "-n", "10", "taskset", "-c", H.EVAL_CORES[0 if gpu <= 2 else 1]] + command


def score_command(ep: int, w: Path, frame_refs: Path = FRAME_REFS, results: Path | None = None) -> list[str]:
    return ["nice", "-n", "10", str(KPY), "-I", "-B", str(CLEAN_SCORER), "--kit", str(KIT),
            "--out", str(w / "clean_scores_LOCAL_NOT_FOR_UPLOAD.csv"), "--tier1", str(TIER1_CSV), "--",
            "--track", "track_2_tier3", "--results", str(results or (w / "results")), "--dataset-root", str(DATASET),
            "--code-commit", COMMIT, "--eval-script", str(KIT / "flash_chord/scripts/evaluate_policy.py"),
            "--frame-references", str(frame_refs), "--config", str(w / "pack" / "tracks.json"),
            "-o", str(w / "packed_LOCAL_NOT_FOR_UPLOAD.csv")]  # fmt: skip


def acquire_lane(ep: int, gpu: int) -> list[int]:
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
                why = f"gpu{gpu}: another t3opus evaluation holds the lane"
            else:
                fds = [mine]
                theirs = RUNIN_EVALLOCK / f"gpu{gpu}.lock"
                if theirs.exists():
                    t = try_lock(theirs, create=False)
                    if t is None:
                        os.close(mine)
                        why = f"gpu{gpu}: a rolling-start / Tier-1 fix evaluation holds {theirs}"
                    else:
                        fds.append(t)
                if not why:
                    plog(ep, f"eval lane gpu{gpu} taken (own lock{' + ' + str(theirs) if len(fds) > 1 else '; no shared lock file for this GPU'})")
                    return fds
        if why != last:
            plog(ep, f"eval waits: {why}")
            last = why
        time.sleep(30)


def run_eval(ep: int, gpu: int, w: Path, fds: list[int], tol: float) -> tuple[int, str]:
    link = w / f"ep{ep:02d}"
    if not link.is_symlink():
        link.symlink_to(run_dir(ep))
    (w / "eval_report.json").unlink(missing_ok=True)
    cmd = eval_command(ep, gpu, w, tol)
    note = ""
    with (w / "eval_all.log").open("a") as stream:
        stream.write(f"[{now()}] gpu {gpu} tolerance {tol:g}: {' '.join(cmd)}\n")
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
        ent = lock_entry(ep) or {}
        out.update({"eval_add_auc_own_input": fnum(m.get("add_auc")),
                    "eval_sp_sr": fnum(m.get("spider_sr_uncentered", m.get("spider_sr"))),
                    "eval_mp_sr": fnum(m.get("maniptrans_sr")), "eval_chord_sr": fnum(m.get("chord_sr")),
                    "eval_mppe_cm_own_input": fnum(m.get("mppe_cm", m.get("mean_per_point_error_cm"))),
                    "eval_world_count": m.get("world_count", res.get("world_count")),
                    "eval_non_finite_worlds": res.get("non_finite_worlds"), "eval_reference_parquet": ref,
                    "eval_reference_is_train_input": bool(ref) and Path(ref).resolve() == parquet(ep).resolve()
                    and sha256(ref) == ent.get("sha256")})
    except (OSError, ValueError):
        pass
    return out


def input_clean_auc(ep: int) -> float | None:
    try:
        rows = json.loads(INPUT_LOCAL.read_text())["rows"]
        return next(fnum(r["opus_add_auc"]) for r in rows if int(r["episode"]) == ep)
    except (OSError, ValueError, StopIteration, KeyError):
        return None


def score(ep: int, w: Path) -> dict:
    """LOCAL clean diagnostic (never selects a checkpoint, never decides the verdict); numbers only in LOCAL files."""
    if not (FRAME_REFS / f"episode_{ep:06d}.npz").is_file():
        return {"score_status": "no_frame_refs"}
    pack = w / "pack"
    pack.mkdir(exist_ok=True)
    cfg = json.loads((KIT / "config/tracks.json").read_text())
    cfg["tracks"]["track_2_tier3"]["scored_episodes"] = [ep]
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
    clean = {"episode": ep, "object": CLIPS[ep]["obj"], "written": now(), "frame_refs": str(FRAME_REFS),
             "note": "LOCAL clean-reference diagnostic; never selects a checkpoint (final.json's is fixed) and never enters the verdict",
             "input_clean_add_auc": input_clean_auc(ep)}
    if code == 0 and rows:
        r = rows[0]
        clean.update({"clean_add_auc": fnum(r.get("t2clean_add_auc")), "clean_sp_sr": fnum(r.get("t2clean_spider_sr")),
                      "clean_mp_sr": fnum(r.get("t2clean_maniptrans_sr")), "clean_mppe_cm": fnum(r.get("t2clean_mppe_cm")),
                      "tier1_c3r2_add_auc": fnum(r.get("t1_add_auc"))})
        if clean.get("clean_add_auc") is not None and clean["input_clean_add_auc"]:
            clean["policy_over_input_clean_add_auc"] = clean["clean_add_auc"] / clean["input_clean_add_auc"]
        out["score_status"] = "ok"
    else:
        out["score_status"] = "failed"
        out["score_note"] = f"clean scorer exit {code}; details only in {log_path.name}"
        clean["score_failure_tail"] = " | ".join([l for l in text.splitlines() if l.strip()][-3:])[:400]
    atomic_write(w / "clean_LOCAL_NOT_FOR_UPLOAD.json", json.dumps(clean, indent=1))
    return out


def post(ep: int) -> int:
    rec = read_pids().get(ep)
    gpu = int(rec["gpu"]) if rec and str(rec["gpu"]).isdigit() else None
    if gpu is None or gpu not in ALLOWED_GPUS:
        print(f"ep{ep:02d}: no training GPU recorded in {PIDS} (or not allowed): nothing done")
        return 2
    (A / "post").mkdir(parents=True, exist_ok=True)
    lock = try_lock(A / "post" / f"ep{ep:02d}.lock", create=True)
    if lock is None:
        print(f"ep{ep:02d}: another post holds its lock; nothing to do")
        return 0
    w = EVROOT / f"ep{ep:02d}"
    w.mkdir(parents=True, exist_ok=True)
    st = load_post(ep) or {"episode": ep, "object": CLIPS[ep]["obj"], "status": "pending", "tries": 0}
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
    if why is None or fin is None or find_trainers(ep):
        plog(ep, f"training not finished or trainer alive ({run}): nothing to do")
        save_post(ep, st)
        return 3
    try:
        age = time.time() - (run / fin["checkpoint"]).stat().st_mtime
    except OSError:
        age = MIN_FINAL_AGE
    if age < MIN_FINAL_AGE:
        time.sleep(MIN_FINAL_AGE - age)
    psha, ent = sha256(parquet(ep)), lock_entry(ep) or {}
    st.update({"final_iteration": fin["iteration"], "final_reason": fin.get("reason"), "help_zero_since": fin.get("help_zero_since"),
               "train_gpu": gpu, "parquet": str(parquet(ep)), "parquet_sha256": psha})
    if psha != ent.get("sha256"):
        st.update({"status": "failed", "finished_at": now(), "note": f"the parquet changed since the retarget check ({psha})"})
        save_post(ep, st)
        plog(ep, st["note"])
        return 1
    st["status"] = "waiting_gpu"
    save_post(ep, st)
    plog(ep, f"final.json: {why}; evaluating model_{fin['iteration']} against its own input (export tol {EXPORT_TOLS[0]:g}) on gpu{gpu}")
    fds = acquire_lane(ep, gpu)
    st.update({"status": "evaluating", "eval_gpu": gpu, "eval_started_at": now()})
    save_post(ep, st)
    t0 = time.time()
    tol = EXPORT_TOLS[0]
    try:
        st["eval_command"] = " ".join(eval_command(ep, gpu, w, tol))
        code, note = run_eval(ep, gpu, w, fds, tol)
        res = parse_eval(ep, w)
        if str(res.get("eval_status", "")).startswith("export_failed") and (res.get("export_max_rel_diff") or 0) > tol:
            ed = w / "export" / f"ep{ep:02d}"
            for f in ("export_report.json", "export.log"):
                if (ed / f).exists():
                    os.replace(ed / f, ed / f.replace(".", "_tol1e-4_REFUSED.", 1))
            for f in ("eval_report.json", "eval_report.csv", "eval_report.md"):
                if (w / f).exists():
                    os.replace(w / f, w / f.replace(".", "_tol1e-4.", 1))
            st["export_refused_1e-4_max_rel_diff"] = res.get("export_max_rel_diff")
            plog(ep, f"exporter REFUSED at 1e-4 (max_rel_diff {res.get('export_max_rel_diff')}); redoing the export at 2e-4 (recorded)")
            tol = EXPORT_TOLS[1]
            st["eval_command"] = " ".join(eval_command(ep, gpu, w, tol))
            code, note = run_eval(ep, gpu, w, fds, tol)
            res = parse_eval(ep, w)
    finally:
        for fd in fds:
            os.close(fd)
    st.update(res)
    st.update({"eval_exit": code, "eval_seconds": round(time.time() - t0, 1), "eval_note": note, "export_tolerance": tol})
    tries = int(st.get("tries", 0)) + 1
    if res.get("eval_status") == "evaluated" and code == 0:
        plog(ep, f"evaluated on gpu{gpu} against its own input: SP-SR {res.get('eval_sp_sr')} MP-SR {res.get('eval_mp_sr')} "
                 f"world_count {res.get('eval_world_count')} AUC(own input) {res.get('eval_add_auc_own_input')} export tol {tol:g} "
                 f"max_rel_diff {res.get('export_max_rel_diff')}; reference is the training parquet: {res.get('eval_reference_is_train_input')}")
        sc = score(ep, w)
        st.update(sc)
        st["status"] = "done" if sc.get("score_status") == "ok" else "done_score_failed"
        st["finished_at"] = now()
        plog(ep, f"local clean diagnostic: {sc.get('score_status')} (numbers only in {w}/clean_LOCAL_NOT_FOR_UPLOAD.json)")
    elif str(res.get("eval_status", "")).startswith("export_failed") and (res.get("export_max_rel_diff") or 0) > EXPORT_TOLS[1]:
        st.update({"status": "export_refused", "finished_at": now(),
                   "note": f"export refused at 1e-4 and 2e-4: max_rel_diff {res['export_max_rel_diff']:.3g}"})
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


def ref_curve(tag: str, runs: Path, ep: int, write: bool) -> tuple[dict, str]:
    cache = A / "ref_curves" / f"{tag}_ep{ep:02d}.json"
    try:
        return {int(k): v for k, v in json.loads(cache.read_text()).items()}, ""
    except (OSError, ValueError):
        pass
    files = sorted(glob.glob(str(runs / f"ep{ep:02d}/wandb/run-*/run-*.wandb")))
    if not files:
        return {}, f"no {tag} W&B record"
    try:
        b = _bins(_rows_wandb(files[-1]))
    except Exception as error:  # noqa: BLE001
        return {}, f"{tag} curve unreadable ({type(error).__name__})"
    if write and b:
        atomic_write(cache, json.dumps(b))
    return b, ""


def own_curve(ep: int) -> dict:
    path = run_dir(ep) / "metrics.jsonl"
    rows = []
    if path.is_file():
        with path.open() as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    return _bins(rows)


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


def verdict(write: bool) -> int:
    ok, why = prereg_status()
    if not ok:
        print(f"VERDICT REFUSED: {why} (the rule is fixed by PREREG.md; nothing written)")
        return 2
    clips, clean, fail_rows = {}, {}, []
    for ep in ORDER:
        c = CLIPS[ep]
        run = run_dir(ep)
        fin = H.final_marker(run) if H.done_reason(run, ITERS) else None
        crashed = (A / f"CRASHED_ep{ep:02d}").exists()
        rfailed = (A / f"RETARGET_FAILED_ep{ep:02d}").exists()
        launched = ep in read_pids()
        r = {"episode": ep, "object": c["obj"], "wave": c["wave"], "t1_help_zero_since": c["t1_hz"], "t2raw_help_zero_since": c["t2_hz"],
             "crash_marker": crashed, "retarget_failed_marker": rfailed, "launched": launched}
        if fin:
            hz = fin.get("help_zero_since")
            r.update({"help_zero_since": hz, "final_iteration": fin["iteration"], "final_reason": fin.get("reason")})
            r["a_pass"] = fin.get("reason") == "help_zero_tail" and hz is not None and int(hz) <= HZ_LIMIT
            r["strong"] = r["a_pass"] and int(hz) <= HZ_STRONG
        else:
            r["a_pass"] = None
        r["first_help_zero_report_only"] = first_help_zero(ep)
        st = load_post(ep)
        r["post_status"] = st.get("status")
        b = None
        if st.get("status") in ("done", "done_score_failed"):
            b = (st.get("eval_status") == "evaluated" and st.get("eval_world_count") == 1 and st.get("eval_sp_sr") == 1.0
                 and st.get("eval_mp_sr") == 1.0 and st.get("eval_reference_is_train_input") is True
                 and fin is not None and st.get("evaluated_iteration") == fin["iteration"])
        elif st.get("status") == "export_refused":
            b = False
        r["b_pass"] = b
        for key in ("eval_sp_sr", "eval_mp_sr", "eval_world_count", "evaluated_iteration", "eval_add_auc_own_input",
                    "eval_mppe_cm_own_input", "export_tolerance", "export_max_rel_diff", "eval_reference_is_train_input", "note"):
            r[key] = st.get(key)
        if r["a_pass"] is False or b is False:
            res = "FAIL"
        elif r["a_pass"] and b:
            res = "PASS"
        elif rfailed or (crashed and fin is None) or st.get("status") == "failed":
            res = "INCOMPLETE"
        elif not launched and not fin:
            res = "NOT_LAUNCHED"
        else:
            res = "PENDING"
        r["result"] = res
        r["settled"] = res in ("INCOMPLETE",) or st.get("status") in TERMINAL
        mine = own_curve(ep)
        for tag, runs in (("t1", T1_RUNS), ("t2raw", T2_RUNS)):
            ref, note = ref_curve(tag, runs, ep, write)
            matched = sorted(set(ref) & set(mine))
            if matched:
                rm = sum(ref[i][0] / ref[i][1] for i in matched) / len(matched)
                km = sum(mine[i][0] / mine[i][1] for i in matched) / len(matched)
                r[f"fail_rate_vs_{tag}_report_only"] = {"matched_bins": len(matched), "up_to": matched[-1], tag: round(rm, 4), "t3opus": round(km, 4)}
                for it in matched:
                    fail_rows.append({"episode": ep, "reference": tag, "iteration_bin": it, "reference_fail_rate": round(ref[it][0] / ref[it][1], 4),
                                      "t3opus_fail_rate": round(mine[it][0] / mine[it][1], 4)})
            else:
                r[f"fail_rate_vs_{tag}_report_only"] = {"matched_bins": 0, "note": note or "no t3opus metrics yet"}
        try:
            clean[ep] = json.loads((EVROOT / f"ep{ep:02d}" / "clean_LOCAL_NOT_FOR_UPLOAD.json").read_text())
        except (OSError, ValueError):
            clean[ep] = None
        clips[f"{ep:02d}"] = r
    waves = {}
    for wv in WAVES:
        rs = [clips[f"{e:02d}"]["result"] for e in ORDER if CLIPS[e]["wave"] == wv]
        if all(x == "PASS" for x in rs):
            v = "C3r2 TRAINS on the opus reconstruction for these easy clips"
        elif "FAIL" in rs:
            v = "NOT shown: C3r2 does not train on the opus reconstruction for every easy clip"
        elif all(x == "NOT_LAUNCHED" for x in rs):
            v = "NOT LAUNCHED"
        elif "PENDING" in rs or "NOT_LAUNCHED" in rs:
            v = "PENDING"
        else:
            v = "INCOMPLETE"
        waves[str(wv)] = {"clips": [f"{e:02d}" for e in ORDER if CLIPS[e]["wave"] == wv], "results": rs, "verdict": v}
    out = {"written": now(), "prereg": str(PREREG), "prereg_sha256": sha256(PREREG), "prereg_check": why,
           "rule": (f"clip PASS iff (a) final.json reason help_zero_tail with help_zero_since <= {HZ_LIMIT} and (b) ONE official "
                    "evaluation of final.json's checkpoint against its own (opus) input: world_count 1, SP-SR = MP-SR = 1, evaluator "
                    "reference = the retarget-locked parquet; FAIL if (a) or (b) is false; wave verdict 'trains' iff all its clips PASS"),
           "waves": waves, "clips": clips}
    lines = [f"Tier-3 opus easy-clip test ({ARM}) - verdict by the pre-registered rule ({why})", f"written {out['written']}", ""]
    for wv, d in waves.items():
        lines.append(f"WAVE {wv} ({' '.join('ep' + x for x in d['clips'])}): {d['verdict']}")
    lines.append("")
    for r in clips.values():
        lines.append(f"ep{r['episode']:02d} {r['object']:<18} w{r['wave']} {r['result']:<12} (a) help zero {r.get('help_zero_since')} "
                     f"(limit {HZ_LIMIT}; Tier-1 C3r2 {r['t1_help_zero_since']}, raw Tier-2 {r['t2raw_help_zero_since']}) -> {r['a_pass']}"
                     f"{' strong' if r.get('strong') else ''} | (b) own-input SP-SR {r.get('eval_sp_sr')} MP-SR {r.get('eval_mp_sr')} "
                     f"worlds {r.get('eval_world_count')} ckpt it {r.get('evaluated_iteration')} -> {r['b_pass']} | post {r['post_status']}"
                     f"{' | CRASH MARKER' if r['crash_marker'] else ''}{' | RETARGET FAILED' if r['retarget_failed_marker'] else ''}")
    lines += ["", "Reported, not decided: own-input AUC / MPPE and training fail rates (matched 1000-iteration bins):"]
    for r in clips.values():
        f1, f2 = r["fail_rate_vs_t1_report_only"], r["fail_rate_vs_t2raw_report_only"]
        lines.append(f"  ep{r['episode']:02d} AUC(own input) {r.get('eval_add_auc_own_input')} MPPE {r.get('eval_mppe_cm_own_input')} | "
                     + (f"fail rate t3opus {f1['t3opus']} vs Tier-1 {f1['t1']} ({f1['matched_bins']} bins)" if f1.get("matched_bins") else f"Tier-1 curve: {f1.get('note')}")
                     + " | " + (f"vs raw Tier-2 {f2['t2raw']} (t3opus {f2['t3opus']}, {f2['matched_bins']} bins)" if f2.get("matched_bins") else f"Tier-2 curve: {f2.get('note')}"))
    lines.append("  clean diagnostics only in VERDICT_clean_LOCAL_NOT_FOR_UPLOAD.json")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if write:
        atomic_write(A / "VERDICT.json", json.dumps(out, indent=1, default=str))
        atomic_write(A / "VERDICT.txt", text)
        with (A / "fail_rates.csv.tmp").open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["episode", "reference", "iteration_bin", "reference_fail_rate", "t3opus_fail_rate"])
            wr.writeheader()
            wr.writerows(fail_rows)
        os.replace(A / "fail_rates.csv.tmp", A / "fail_rates.csv")
        t1 = {}
        try:
            for row in csv.DictReader(TIER1_CSV.open()):
                if row["episode"].strip().isdigit() and int(row["episode"]) in CLIPS:
                    t1[int(row["episode"])] = {k: fnum(row.get(k)) for k in ("auc", "sp_sr", "mp_sr", "mppe_cm")}
        except (OSError, KeyError):
            pass
        base = {}
        try:
            for row in csv.DictReader(BASELINE_CSV.open()):
                if row["episode"].strip().isdigit() and int(row["episode"]) in CLIPS:
                    base[int(row["episode"])] = {k: fnum(row.get(f"t2train_mean_{k}")) for k in ("add_auc", "spider_sr", "maniptrans_sr", "mppe_cm")}
        except (OSError, KeyError):
            pass
        atomic_write(A / "VERDICT_clean_LOCAL_NOT_FOR_UPLOAD.json", json.dumps(
            {"written": now(), "note": "LOCAL ONLY. One clean-reference diagnostic per clip; never selects checkpoints, never enters the "
                                       "verdict. input = the opus track itself scored as a perfect rollout (input_quality_LOCAL); "
                                       "Tier-1 = C3r2_wide official final; raw Tier-2 = C3r2_tier2 final, mean of 3 evaluations.",
             "t3opus": {f"{ep:02d}": clean[ep] for ep in ORDER},
             "input_clean_add_auc": {f"{ep:02d}": input_clean_auc(ep) for ep in ORDER},
             "tier1_c3r2_official": {f"{ep:02d}": t1.get(ep) for ep in ORDER},
             "tier2_raw_mean3": {f"{ep:02d}": base.get(ep) for ep in ORDER}}, indent=1, default=str))
        print(f"wrote {A / 'VERDICT.json'}, VERDICT.txt, fail_rates.csv, VERDICT_clean_LOCAL_NOT_FOR_UPLOAD.json")
    return 0


def heartbeat(waves: set[int]) -> int:
    parts = []
    for ep in ORDER:
        s, _ = clip_state(ep, waves)
        if s == "DEFERRED":
            continue
        pr = progress(ep)
        it = pr["it"] if pr["it"] is not None else pr["last_model"]
        parts.append(f"ep{ep:02d} {s} it={it} hz={pr['help_zero_iters']} fail={round(pr['fail'], 3) if isinstance(pr['fail'], float) else pr['fail']}")
    print(" | ".join(parts) or "no clip of the requested wave(s)")
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
    if sub == "wave-clips":
        ws = waves_of(opt("--waves"))
        print("\n".join(f"{e:02d}" for e in ORDER if CLIPS[e]["wave"] in ws))
        return 0
    if sub == "cmd":
        sys.stdout.write("\0".join(train_argv(ep_of(opt("--ep")))) + "\0")
        return 0
    if sub == "seq":
        print(CLIPS[ep_of(opt("--ep"))]["seq"])
        return 0
    if sub == "plan":
        print(plan(ep_of(opt("--ep")), opt("--exclude", "") or "", parse_assume(opt("--assume"))))
        return 0
    if sub == "rplan":
        print(rplan(opt("--exclude", "") or ""))
        return 0
    if sub == "launch-ok":
        ok, detail = rcheck_ok(ep_of(opt("--ep")))
        print(("OK " if ok else "NO ") + detail)
        return 0 if ok else 1
    if sub == "gpus":
        print("\n".join(gpus_report()))
        return 0
    if sub == "state":
        return state(waves_of(opt("--waves")))
    if sub == "trainer":
        print(" ".join(map(str, find_trainers(ep_of(opt("--ep"))))))
        return 0
    if sub == "crash":
        return marker("CRASHED", ep_of(opt("--ep")), opt("--why", ""))
    if sub == "rfail":
        return marker("RETARGET_FAILED", ep_of(opt("--ep")), opt("--why", ""))
    if sub == "post":
        return post(ep_of(opt("--ep")))
    if sub == "verdict":
        return verdict("--no-write" not in rest)
    if sub == "heartbeat":
        return heartbeat(waves_of(opt("--waves")))
    if sub == "score-selftest":  # dry run only: the LOCAL scoring path on the Tier-3 pilot's probe output (ep14, agent_T1)
        w = Path(opt("--work"))
        w.mkdir(parents=True, exist_ok=True)
        (w / "pack").mkdir(exist_ok=True)
        cfg = json.loads((KIT / "config/tracks.json").read_text())
        cfg["tracks"]["track_2_tier3"]["scored_episodes"] = [14]
        (w / "pack" / "tracks.json").write_text(json.dumps(cfg, indent=2))
        cmd = score_command(14, w, ME / "tmp_analysis/tier3/frame_refs_agentT1", ME / "tmp_analysis/tier3/t1ckpt_probe/res_t3")
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
        env.update({"PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"})
        with (w / "score_selftest_LOCAL_NOT_FOR_UPLOAD.log").open("w") as stream:  # the packer prints clean numbers: LOCAL name
            stream.write(" ".join(cmd) + "\n")
            stream.flush()
            code = subprocess.run(cmd, cwd=KIT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=SCORE_TIMEOUT).returncode
        try:  # compare with the pilot's probe record; print only the yes/no (clean numbers stay in LOCAL files)
            want = json.loads((ME / "tmp_analysis/tier3/s4b_probe_LOCAL_NOT_FOR_UPLOAD.json").read_text())["probe_vs_clean"]
            row = next(r for r in csv.DictReader((w / "clean_scores_LOCAL_NOT_FOR_UPLOAD.csv").open()) if r["episode"].strip() == "14")
            same = all(abs(float(row[f"t2clean_{k}"]) - float(want[k])) <= 1e-6 * max(1.0, abs(float(want[k])))
                       for k in ("add_auc", "spider_sr", "maniptrans_sr", "mppe_cm"))
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            same = f"not comparable ({type(error).__name__})"
        print(f"packer exit {code}; reproduces s4b_probe_LOCAL_NOT_FOR_UPLOAD.json probe_vs_clean (AUC, SP-SR, MP-SR, MPPE): {same} "
              f"(numbers only in {w.name}/*_LOCAL_NOT_FOR_UPLOAD.*)")
        return 0 if code == 0 and same is True else 1
    if sub == "build-record":
        atomic_write(A / "build.json", json.dumps({"written": now(), "code": str(CODE), "pin": PIN, "code_digest": code_digest()}, indent=1))
        print(f"wrote {A / 'build.json'}")
        return 0
    print(f"unknown subcommand {sub}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
