#!/usr/bin/env python3
"""t2best5_driver.py: Tier-2 5x official re-evaluation and composition (fc_runs/submissions/t2_best5_20261024).

It implements ROOT/PREREG.md unchanged (its sha256 must equal ROOT/PREREG.sha256, and ROOT/PREREG_PINS.md must equal
ROOT/PREREG_PINS.sha256 and pin this file). User request 2026-10-24: "I think zuxing run some experiment on tier2 also,
you should also take account his results right? what is overall tier2 results? double check it by rerun using
official evaluator 5 times and get highest and lowest and average?"

Standing rule (2.8.c): selection uses only own-input scores (each policy against its OWN training parquet), comparable
only within one parquet sha256. Clean (leaderboard-style) numbers live only in *_LOCAL_NOT_FOR_UPLOAD* files and never
select anything: the selection code refuses to open any path containing LOCAL_NOT_FOR_UPLOAD, clean_ or packed_.

Steps (PREREG sections 1-4):
  manifest   Stage A (70 jobs, PREREG section 1), built once: checkpoint + training parquet re-hashed and equal to
             inventory/inventory.json; designation evidence (final.json, t2watch best.json, Zuxing's metrics.csv row,
             T2fix eval_ARM export report); Zuxing's 7 checkpoints copied (model + rsi, sha256) -> COPIES.tsv.
             Collision-hull preflight (tmp_analysis/t2fix/cache_keys_t2.py, read-only, CPU): every CoACD key must
             already exist in the stock runtime's convex cache, else hull_missing.
  export     once per job, CPU, nice + taskset on the eval-lane cores, `python -m <EXPORTER>` of the training tree
             (fc_r2 61fa5d3; Zuxing's: nv-v2d_runin b39b69b), stock-runtime check at 1e-4, one fallback export at
             2e-4 on a tolerance refusal; technical failures retried once.
  evaluate   5 runs of that same exported file: scripts/track2/eval_all.py --export no (the kit's
             run_policy_evaluation.py + evaluate_policy.py on the untouched stock runtime, frame 0, world count 1,
             explicit reset, no help), HUMAN_MOTION_DATA_DIR = the job's data root; ok checks of t1best5 plus the
             reference parquet == the job's training parquet (path and sha256). <= 3 tries per run, 5 min apart.
  clean      the LOCAL clean diagnostic of each ok run (kit packer through scripts/track2_tier2_clean_scores.py, frame
             references by data root, no --tier1), CPU, one retry. Report only.
  stage B    every 10 min, read-only: VERDICT.json (ep 04/14/21/25 K3fix final.json, then their t2watch best), T2fix
             ep09-B / ep01-L (final.json + runin_final.json).
  summary    selection.json/.txt (own-input only, section 2), summary_own_input.*, summary_LOCAL_NOT_FOR_UPLOAD.*,
             compositions RAW-ONLY / K3FULL / SELECTED / SELECTED+P / ZUXING-LIVE (section 4).

Scheduling: GPUs 0,1,2,5,6,7 only (never 4; 3 has no lock file), never a GPU with MPS, >= 4096+1536 MiB free, one of
our evals per GPU, <= 2 per NUMA eval lane, <= 4 in all; each eval holds the shared flock
fc_runs/queue_runin/evallock/gpu<g>.lock (opened read-only, never created) for its whole life. The driver NEVER signals
any process, its own children included: an eval past 2400 s is only flagged OVERDUE.

  python -B t2best5_driver.py run [--dry-run] [--gpus 0,1,2,5,6,7] [--max-parallel 4] [--lane-max 2] [--runs 5]
  python -B t2best5_driver.py summary | status | selftest
Start: scripts/t2best5_launch.sh (setsid nohup). Stop: touch ROOT/STOP (running jobs finish and are recorded).
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import ast  # noqa: E402
import csv  # noqa: E402
import datetime as dt  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import shlex  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
ROOT = ME / "fc_runs/submissions/t2_best5_20261024"
FC = ME / "fc_runs"
ZU = Path("/root/autodl-tmp/zuxing.lu")  # READ ONLY: files are read; only the 7 designated checkpoints are copied
KIT = ME / "v2d_submission_kit_src/v2d_submission_kit"
STOCK_REPO = ME / "code/video_to_data_8c778f9"
STOCK = STOCK_REPO / "robotic_grounding/flash_chord"
STOCK_CONVEX = STOCK / "src/flash_chord/assets/.cache/convex"
DATASET = ME / "data/video_to_data_challenge"
KPY = ME / "conda_envs/v2dkit/bin/python"
PY = ME / "conda_envs/flashchord/bin/python"
LOCKDIR = FC / "queue_runin/evallock"
VERDICT = ME / "tmp_analysis/tier2/k3full/VERDICT.json"
T2W = ME / "tmp_analysis/tier2/t2watch"
T2FIX = FC / "T2fix"
CLEAN_SCORER = ME / "scripts/track2_tier2_clean_scores.py"
CACHE_KEYS = ME / "tmp_analysis/t2fix/cache_keys_t2.py"
INVENTORY = ROOT / "inventory/inventory.json"
PREREG, PREREG_SHA = ROOT / "PREREG.md", ROOT / "PREREG.sha256"
PINS, PINS_SHA = ROOT / "PREREG_PINS.md", ROOT / "PREREG_PINS.sha256"
COMMIT_URL = "https://github.com/local-placeholder/not-uploaded/commit/0000000000000000000000000000000000000000"
TREES = {
    "fc_r2": {"code": ME / "code/fc_r2", "repo": ME / "code/fc_r2", "commit": "61fa5d31ff8054bf1e5cf5375ef0957e54d2729b",
              "label": "code/fc_r2 @ 61fa5d3"},
    "b39b69b": {"code": ME / "code/nv-v2d_runin/robotic_grounding/flash_chord", "repo": ME / "code/nv-v2d_runin",
                "commit": "b39b69be6c2f561c0123c6aeb8cd13f8b9d542e7",
                "label": "code/nv-v2d_runin @ b39b69b (exporter accepted for speedup-trained checkpoints in t1best5)"},
}  # fmt: skip
STOCK_COMMIT = "8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef"
EXPORTER = "flash_chord.training.ppo.coordinate.export"
EXPORT_TOL, FALLBACK_TOL = 1e-4, 2e-4
EVAL_ALL_SHA = "657295921e820a464160039ad9a65ba92a683108a99dcb197290ae4147363197"
EVAL_SCRIPT_SHA = "160d24f5fbeb70ccd569ca4192056f1892b2a9f2999066db9d8279461648fdf1"
EVAL_CODE_SHA = "b1b3ebe3ae4629d1687a83f420149e3f6fdfc154d70acad55c12fc8fa69811a1"
PREREG_V3_SHA = "0f4fa433f3368072e161d255c18de18bda03668b4095f3958e6411aaec2d3995"
OVERDUE_S = 2400.0
CPU_OVERDUE_S = 1800.0
RETRY_DELAY = 300.0
EVAL_TRIES = 3
EXPORT_TECH_TRIES = 2  # a technical export failure is retried once
CLEAN_TRIES = 2  # a clean-scoring failure is retried once
STAGEB_POLL = 600.0
SUMMARY_EVERY = 600.0
MIN_FREE_MB, EVAL_PEAK_MB, MIN_FREE_PIDS = 4096, 1536, 2000
EVAL_CORES = ("48-51,152-155", "100-103,204-207")  # NUMA0 / NUMA1 eval lanes (never training cores)
ALLOWED_GPUS = (0, 1, 2, 5, 6, 7)
TZ8 = dt.timezone(dt.timedelta(hours=8))
FREEZE = dt.datetime(2026, 10, 24, 17, 58, 27, tzinfo=TZ8).timestamp()  # inventory snapshot (designation cut-off)
FORBIDDEN = ("LOCAL_NOT_FOR_UPLOAD", "clean_", "packed_")
G2ROOT = ZU / "data/human_motion_data_tier2_g2-2026-10-24"
ROOTS = {"raw": ME / "data/human_motion_data_tier2", "k3fix": ME / "data/human_motion_data_tier2_k3fix",
         "k3": ME / "data/human_motion_data_tier2_k3", "g2": G2ROOT}  # fmt: skip
FRAME_REFS = {"raw": None, "k3fix": ME / "tmp_analysis/tier2/k3fix_pilot/frame_refs_k3fix",
              "k3": ME / "tmp_analysis/tier2/kalman/eval/frame_refs_k3", "g2": G2ROOT / "frame_refs_g2"}  # fmt: skip
# --manifest <job data root>/whole_body/soma/track2_manifest.json (PREREG 3.5); all four are byte-identical (12fe4494).
MANIFEST_FOR = {k: (v / "whole_body/soma/track2_manifest.json") for k, v in ROOTS.items()}
METRICS = ("auc", "sp_sr", "mp_sr", "mppe")
LEADERBOARD = {"auc": "add_auc", "sp_sr": "spider_sr", "mp_sr": "maniptrans_sr", "mppe": "mppe_cm"}
FALLBACK_KEYS = {"auc": "add_auc", "sp_sr": "spider_sr_uncentered", "mp_sr": "maniptrans_sr", "mppe": "mean_per_point_error_cm"}
CLEAN_COLS = {"auc": "t2clean_add_auc", "sp_sr": "t2clean_spider_sr", "mp_sr": "t2clean_maniptrans_sr",
              "mppe": "t2clean_mppe_cm"}  # fmt: skip

# ------------------------------------------------------------------------------------- PREREG section 1 (Stage A)
RAW_AT_WRITING = (0, 5, 15, 19, 22, 26, 28)  # CHOSEN = RAW (gate FAIL)
PENDING_AT_WRITING = (4, 14, 21, 25)  # CHOSEN = PENDING -> raw until Stage B
RAW_FINAL = {0: 41043, 1: 44922, 2: 37409, 3: 44922, 4: 44922, 5: 44922, 6: 33894, 7: 44922, 8: 20597, 9: 39328,
             10: 34267, 11: 28925, 12: 37575, 13: 44922, 14: 44922, 15: 44922, 16: 31268, 17: 31807, 18: 44922,
             19: 44922, 20: 19034, 21: 44922, 22: 44922, 23: 20023, 24: 44922, 25: 44922, 26: 34948, 27: 44922,
             28: 44922, 29: 44922}  # fmt: skip
K3_CHOSEN = {1: ("C3r2_tier2_k3fix/ep01_k3fix_s42", 44922), 2: ("C3r2_tier2_k3fix/ep02_k3fix_s42", 34093),
             3: ("C3r2_tier2_k3fix/ep03_k3fix_s42", 27068), 6: ("C3r2_tier2_k3fix/ep06_k3fix_s42", 28420),
             7: ("C3r2_tier2_k3fix/ep07_k3fix_s42", 18529), 8: ("C3r2_tier2_k3easy/ep08", 17496),
             9: ("C3r2_tier2_k3fix/ep09_k3fix_s42", 39449), 10: ("C3r2_tier2_k3fix/ep10_k3fix_s42", 25319),
             11: ("C3r2_tier2_k3fix/ep11_k3fix_s42", 29175), 12: ("C3r2_tier2_k3fix/ep12_k3fix_s42", 23178),
             13: ("C3r2_tier2_k3fix/ep13_k3fix_s42", 39254), 16: ("C3r2_tier2_k3fix/ep16_k3fix_s42", 32063),
             17: ("C3r2_tier2_k3fix/ep17_k3fix_s42", 29845), 18: ("C3r2_tier2_k3fix/ep18_k3fix_s42", 18911),
             20: ("C3r2_tier2_k3easy/ep20", 18450), 23: ("C3r2_tier2_k3easy/ep23", 19162),
             24: ("C3r2_tier2_k3fix/ep24_k3fix_s42", 20636), 27: ("C3r2_tier2_k3fix/ep27_k3fix_s42", 20863),
             29: ("C3r2_tier2_k3fix/ep29_k3fix_s42", 17358)}  # fmt: skip
T2W_BEST = {1: 18000, 2: 34000, 3: 24000, 6: 28000, 7: 14000, 10: 18000, 11: 28000, 12: 20000, 16: 32000, 18: 18000,
            29: 16000}  # fmt: skip
ZUXING = [  # (episode, run name, input kind, frozen iteration, settled)
    (1, "2026-10-24-tier2-runin-g2", "g2", 10000, False), (4, "2026-10-24-tier2-runin-g2", "g2", 12000, False),
    (4, "2026-10-24-tier2-runin-k3fix-ep04", "k3fix", 12000, True),
    (5, "2026-10-24-tier2-runin-k3fix-ep05-ep26", "k3fix", 18000, False),
    (19, "2026-10-24-tier2-runin-g2", "g2", 18000, False),
    (26, "2026-10-24-tier2-runin-k3fix-ep05-ep26", "k3fix", 16000, False),
    (28, "2026-10-24-tier2-runin-g2", "g2", 16000, False)]  # fmt: skip
ZUXING_LIVE = {1: "ep01_zx_g2_10000", 4: "ep04_zx_g2_12000", 5: "ep05_zx_k3fix_18000", 19: "ep19_zx_g2_18000",
               26: "ep26_zx_k3fix_16000", 28: "ep28_zx_g2_16000"}  # fmt: skip
STAGEB_K3 = (4, 14, 21, 25)
STAGEB_T2FIX = {"ep09-B": (9, "T2B/ep09"), "ep01-L": (1, "T2L/ep01")}
LIST_ORDER = {"t2watch_best": 0, "t2fix": 1, "zuxing": 2}
PRIORITY_NAMES = {1: "P1 CHOSEN", 2: "P2 RAWREF", 3: "P3 SAME", 4: "P4 DIFF"}


# ================================================================================================ small helpers
def now() -> str:
    return dt.datetime.now(TZ8).strftime("%Y-%m-%d %H:%M:%S")


def stamp() -> str:
    return dt.datetime.now(TZ8).strftime("%Y%m%d_%H%M%S")


def parse_time(text) -> float | None:
    s = str(text or "").strip().replace("T", " ")
    if not s:
        return None
    try:
        d = dt.datetime.fromisoformat(s)
    except ValueError:
        return None
    return (d if d.tzinfo else d.replace(tzinfo=TZ8)).timestamp()


def fnum(value):
    try:
        v = None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None
    return v if v is None or math.isfinite(v) else None


def sha256(file) -> str:
    h = hashlib.sha256()
    with Path(file).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(file):
    try:
        return json.loads(Path(file).read_text())
    except (OSError, ValueError, TypeError):
        return None


def load_csv(file: Path) -> list[dict]:
    try:
        with Path(file).open(newline="") as stream:
            return list(csv.DictReader(stream))
    except OSError:
        return []


def inside_root(target: Path) -> Path:
    if not Path(os.path.abspath(target)).is_relative_to(ROOT):
        raise SystemExit(f"refusing to write outside {ROOT}: {target}")
    return Path(target)


def write_atomic(file: Path, text: str) -> None:
    inside_root(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_name(f".{file.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, file)


def append_line(file: Path, text: str) -> None:
    inside_root(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("a") as fp:
        fp.write(text if text.endswith("\n") else text + "\n")


def dump(data) -> str:
    return json.dumps(data, indent=1, default=str) + "\n"


def csv_text(rows: list[dict], cols: list[str]) -> str:
    lines: list[str] = []

    class _W:
        def write(self, s):
            lines.append(s)

    w = csv.DictWriter(_W(), fieldnames=cols, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})
    return "".join(lines)


def write_csv(file: Path, rows: list[dict], cols: list[str]) -> None:
    write_atomic(file, csv_text(rows, cols))


def numa(gpu: int) -> int:
    return 0 if gpu <= 2 else 1


def tries(base: Path) -> list[tuple[int, Path]]:
    try:
        names = os.listdir(base)
    except OSError:
        return []
    out = []
    for name in names:
        m = re.match(r"^try(\d+)$", name)
        if m and (base / name).is_dir():
            out.append((int(m.group(1)), base / name))
    return sorted(out)


def unquote(text: str) -> str:
    return text[1:-1] if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"" else text


def read_overrides(run: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        text = (run / "hydra/.hydra/overrides.yaml").read_text()
    except OSError:
        return out
    for line in text.splitlines():
        item = line.strip()
        if item.startswith("- "):
            key, sep, value = unquote(item[2:].strip()).partition("=")
            if sep:
                out[key.lstrip("+~")] = unquote(value.strip())
    return out


class Logger:
    def __init__(self, file: Path | None):
        self.file = file
        self.once: set = set()

    def __call__(self, message: str) -> None:
        line = f"[{now()}] {message}"
        print(line, flush=True)
        if self.file is not None:
            append_line(self.file, line)

    def first(self, key, message: str) -> None:
        if key not in self.once:
            self.once.add(key)
            self(message)


def kit_metric_keys() -> tuple[dict, str]:
    """ours -> results-JSON key, from the kit's REPORT_METRIC_KEY (parsed with ast: importing would write bytecode)."""
    try:
        for node in ast.parse((KIT / "v2dlb/report_schema.py").read_text()).body:
            targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
            if any(getattr(t, "id", None) == "REPORT_METRIC_KEY" for t in targets):
                table = ast.literal_eval(node.value)
                return {ours: table[name] for ours, name in LEADERBOARD.items()}, "kit v2dlb/report_schema.py"
        raise KeyError("REPORT_METRIC_KEY not found")
    except Exception as error:  # noqa: BLE001
        return dict(FALLBACK_KEYS), f"built-in fallback ({error!r})"


# ================================================================================================ /proc and GPUs
def all_pids() -> list[int]:
    return [int(p) for p in os.listdir("/proc") if p.isdigit()]


def proc_environ(pid: int) -> dict[str, str]:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError:
        return {}
    out = {}
    for item in raw.split(b"\0"):
        key, sep, value = item.partition(b"=")
        if sep:
            out[key.decode(errors="replace")] = value.decode(errors="replace")
    return out


def proc_cmdline(pid: int) -> list[str]:
    try:
        return [a.decode(errors="replace") for a in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if a]
    except OSError:
        return []


def proc_stat(pid: int) -> list[str] | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return text[text.rfind(")") + 2:].split()


def proc_alive(pid: int, start: str | None) -> bool:
    st = proc_stat(pid)
    return st is not None and st[0] not in ("Z", "X") and (start is None or st[19] == start)


def live_jobs() -> dict[str, list[tuple[int, str]]]:
    """marker -> [(pid, starttime)] of every live process carrying our T2BEST5_JOB marker."""
    found: dict[str, list] = {}
    for pid in all_pids():
        marker = proc_environ(pid).get("T2BEST5_JOB")
        st = proc_stat(pid)
        if marker and st is not None and st[0] not in ("Z", "X"):
            found.setdefault(marker, []).append((pid, st[19]))
    return found


def smi() -> dict:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,memory.free",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    gpus, by_uuid = {}, {}
    for line in out.strip().splitlines():
        idx, uuid, total, used, free = (x.strip() for x in line.split(","))
        gpus[int(idx)] = {"uuid": uuid, "total": int(total), "used": int(used), "free": int(free)}
        by_uuid[uuid] = int(idx)
    if not gpus:
        raise RuntimeError("nvidia-smi listed no GPU")
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory,process_name",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True).stdout
    apps = []
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 4 or not parts[0].isdigit() or parts[1] not in by_uuid:
            continue
        apps.append({"pid": int(parts[0]), "gpu": by_uuid[parts[1]], "mem": int(parts[2]) if parts[2].isdigit() else 0,
                     "name": ",".join(parts[3:])})  # fmt: skip
    return {"gpus": gpus, "apps": apps}


def mps_gpus(state: dict) -> dict[int, str]:
    found: dict[int, str] = {}
    for app in state["apps"]:
        if "mps-server" in app["name"]:
            found.setdefault(app["gpu"], f"MPS server pid {app['pid']}")
        elif proc_environ(app["pid"]).get("CUDA_MPS_PIPE_DIRECTORY"):
            found.setdefault(app["gpu"], f"MPS client pid {app['pid']}")
    uuid_to_idx = {v["uuid"]: k for k, v in state["gpus"].items()}
    for pid in all_pids():
        cmd = proc_cmdline(pid)
        if not cmd or not cmd[0].endswith("nvidia-cuda-mps-control") or "-d" not in cmd[1:]:
            continue
        for token in proc_environ(pid).get("CUDA_VISIBLE_DEVICES", "").split(","):
            token = token.strip()
            idx = int(token) if token.isdigit() else uuid_to_idx.get(token)
            if idx is not None:
                found.setdefault(idx, f"MPS control daemon pid {pid}")
    return found


def pids_headroom() -> int | None:
    for sub in ("", "pids/"):
        try:
            limit = Path(f"/sys/fs/cgroup/{sub}pids.max").read_text().strip()
            current = int(Path(f"/sys/fs/cgroup/{sub}pids.current").read_text())
        except (OSError, ValueError):
            continue
        return None if limit == "max" else int(limit) - current
    return None


def held_lock_inodes() -> set[int]:
    out = set()
    try:
        for line in Path("/proc/locks").read_text().splitlines():
            f = line.split()
            if len(f) >= 6 and f[1] != "->":
                try:
                    out.add(int(f[5].split(":")[-1]))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def child_env(gpu: int | None, marker: str, data_root: Path | None) -> dict:
    """The t1best5 / rolling-start eval environment with the job's data root and our T2BEST5_JOB marker."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CUDA_MPS_", "WANDB_", "RUNIN_", "T2WATCH", "T1BEST5"))
           and k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    sp = ME / "conda_envs/flashchord/lib/python3.11/site-packages/nvidia"
    env.update(
        PATH=f"{ME}/conda_envs/flashchord/bin:{env.get('PATH', '/usr/bin:/bin')}",
        LD_LIBRARY_PATH=f"{sp}/cu13/lib:{sp}/cudnn/lib:{sp}/nccl/lib:{sp}/nvshmem/lib",
        CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
        XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
        NUMEXPR_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1", WANDB_MODE="disabled", XDG_CACHE_HOME=str(ME / ".cache"),
        TZ="Asia/Shanghai", TMPDIR=str(ROOT / "tmp"), T2BEST5_JOB=marker,
    )  # fmt: skip
    if data_root is not None:
        env["HUMAN_MOTION_DATA_DIR"] = str(data_root)
    return env


ENV_SHOWN = ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "JAX_PLATFORMS", "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS",
             "HUMAN_MOTION_DATA_DIR", "TMPDIR", "T2BEST5_JOB")  # fmt: skip


def show_command(prefix: list[str], cmd: list[str], cwd: Path, env: dict) -> str:
    shown = " ".join(f"{k}={shlex.quote(env[k])}" for k in ENV_SHOWN if k in env)
    return f"cd {shlex.quote(str(cwd))} && env {shown} {shlex.join(prefix + cmd)}"


# ================================================================================================ pins
def pin_lines() -> list[tuple[str, str, str]]:
    """(kind, target, value) from the 'pin' / 'commit' lines of PREREG.md section 6."""
    out = []
    for line in PREREG.read_text().splitlines():
        f = line.split()
        if len(f) == 3 and f[0] == "pin":
            out.append(("pin", f[1], f[2]))
        elif len(f) == 4 and f[0] == "pin" and f[1] == "kit":
            out.append(("pin", "kit:" + f[2], f[3]))
        elif len(f) == 3 and f[0] == "commit":
            out.append(("commit", f[1], f[2]))
    return out


def git_head(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                              env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).stdout.strip()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return ""


def check_pins(require_pins_file: bool) -> tuple[list[str], list[str]]:
    """PREREG.md == PREREG.sha256; every pin/commit line of PREREG.md; PREREG_PINS.md == its sha file and pins us."""
    fatal, info = [], []
    want = (PREREG_SHA.read_text().split() or [""])[0] if PREREG_SHA.is_file() else ""
    got = sha256(PREREG) if PREREG.is_file() else ""
    if not want or got != want:
        fatal.append(f"PREREG.md sha256 {got[:16]} != PREREG.sha256 {want[:16]}")
        return fatal, info
    info.append(f"PREREG.md sha256 {got[:16]} == PREREG.sha256")
    for kind, target, value in pin_lines():
        if kind == "commit":
            head = git_head(ME / target)
            (info if head == value else fatal).append(f"commit {target}: HEAD {head[:12]} (pinned {value[:12]})")
            continue
        p = KIT / target[4:] if target.startswith("kit:") else (ROOT / target if target.startswith("inventory/") else ME / target)
        h = sha256(p) if p.is_file() else "missing"
        (info if h == value else fatal).append(f"pin {target}: {h[:16]} (pinned {value[:16]})")
    if require_pins_file:
        pw = (PINS_SHA.read_text().split() or [""])[0] if PINS_SHA.is_file() else ""
        pg = sha256(PINS) if PINS.is_file() else ""
        if not pw or pg != pw:
            fatal.append(f"PREREG_PINS.md sha256 {pg[:16]} != PREREG_PINS.sha256 {pw[:16]}")
        else:
            text = PINS.read_text()
            mine = sha256(Path(__file__).resolve())
            if f"pin scripts/t2best5_driver.py {mine}" not in text:
                fatal.append(f"PREREG_PINS.md does not pin this driver's sha256 {mine[:16]}")
            for line in text.splitlines():
                f = line.split()
                if len(f) == 3 and f[0] == "pin" and f[1] != "scripts/t2best5_driver.py":
                    p = ROOT / f[1] if not f[1].startswith("scripts/") else ME / f[1]
                    h = sha256(p) if p.is_file() else "missing"
                    (info if h == f[2] else fatal).append(f"PINS pin {f[1]}: {h[:16]} (pinned {f[2][:16]})")
            info.append(f"PREREG_PINS.md sha256 {pg[:16]} == PREREG_PINS.sha256; pins this driver {mine[:16]}")
    return fatal, info


# ================================================================================================ manifest
def root_kind(root: Path) -> str | None:
    for k, v in ROOTS.items():
        if os.path.realpath(root) == os.path.realpath(v):
            return k
    return None


def parquet_layout(parquet: Path) -> tuple[Path, str] | None:
    try:
        ok = (parquet.name == "data.parquet" and parquet.parent.name.startswith("robot_name=")
              and parquet.parents[1].name.startswith("sequence_id=") and parquet.parents[2].name == "soma"
              and parquet.parents[3].name == "whole_body")  # fmt: skip
        return (parquet.parents[4], parquet.parents[1].name.split("=", 1)[1]) if ok else None
    except IndexError:
        return None


def inventory_index() -> tuple[dict, dict]:
    inv = json.loads(INVENTORY.read_text())
    by: dict[str, list] = {}
    for c in inv["candidates"]:
        by.setdefault(os.path.normpath(c["run_dir"]), []).append(c)
    return inv, by


def inv_lookup(by: dict, run_dir: str, it: int | None) -> tuple[dict | None, dict | None]:
    cands = by.get(os.path.normpath(run_dir), [])
    for c in cands:
        for k in c.get("checkpoints", []):
            if it is not None and int(k.get("iteration", -1)) == it and k.get("sha256"):
                return c, k
    return (cands[0] if cands else None), None


def stage_a_specs() -> list[dict]:
    specs = []
    for ep in range(30):
        specs.append({"job": f"ep{ep:02d}_raw_final{RAW_FINAL[ep]}", "episode": ep, "family": "raw", "owner": "ours",
                      "arm": "raw C3r2_tier2", "run_dir": str(FC / f"C3r2_tier2/ep{ep:02d}"), "iteration": RAW_FINAL[ep],
                      "tree": "fc_r2", "designation": "final.json", "provisional": False})  # fmt: skip
    for ep, (rel, it) in sorted(K3_CHOSEN.items()):
        easy = "k3easy" in rel
        specs.append({"job": f"ep{ep:02d}_{'k3easy' if easy else 'k3fix'}_final{it}", "episode": ep,
                      "family": "k3easy_final" if easy else "k3fix_final", "owner": "ours",
                      "arm": "K3 easy-clip C3r2_tier2_k3easy" if easy else "K3fix C3r2_tier2_k3fix",
                      "run_dir": str(FC / rel), "iteration": it, "tree": "fc_r2", "designation": "final.json",
                      "provisional": False})  # fmt: skip
    for ep, it in sorted(T2W_BEST.items()):
        rel = K3_CHOSEN[ep][0]
        specs.append({"job": f"ep{ep:02d}_k3fix_t2wbest{it}", "episode": ep, "family": "t2watch_best", "owner": "ours",
                      "arm": "K3fix t2watch best", "run_dir": str(FC / rel), "iteration": it, "tree": "fc_r2",
                      "designation": "t2watch best.json", "best_json": str(T2W / "best" / rel / "best.json"),
                      "provisional": False})  # fmt: skip
    specs.append({"job": "ep01_t2fixL_p62000", "episode": 1, "family": "t2fix", "owner": "ours",
                  "arm": "T2fix ep01-L (provisional)", "run_dir": str(T2FIX / "T2L/ep01"), "iteration": 62000,
                  "tree": "fc_r2", "designation": "T2fix best single eval_ARM eval by 17:58:27", "t2fix_episode": 1,
                  "provisional": True})  # fmt: skip
    for ep, run, kind, it, settled in ZUXING:
        specs.append({"job": f"ep{ep:02d}_zx_{kind}_{it}", "episode": ep, "family": "zuxing", "owner": "zuxing",
                      "arm": f"Zuxing {run} ({kind})", "run_dir": str(ZU / "runs" / run / f"ep{ep:02d}"), "iteration": it,
                      "tree": "b39b69b", "designation": "Zuxing frozen pick (his metrics.csv by 17:58:27)",
                      "zuxing_run": run, "provisional": not settled})  # fmt: skip
    s7 = FC / "C3r2_tier2_k3fix/ep03_raw_s7"
    specs.append({"job": "ep03_raws7_final44922", "episode": 3, "family": "raws7_final", "owner": "ours",
                  "arm": "raw seed-7 control", "run_dir": str(s7), "iteration": 44922, "tree": "fc_r2",
                  "designation": "final.json", "provisional": False})  # fmt: skip
    specs.append({"job": "ep03_raws7_t2wbest38000", "episode": 3, "family": "raws7_t2watch_best", "owner": "ours",
                  "arm": "raw seed-7 t2watch best", "run_dir": str(s7), "iteration": 38000, "tree": "fc_r2",
                  "designation": "t2watch best.json", "best_json": str(T2W / "best/C3r2_tier2_k3fix/ep03_raw_s7/best.json"),
                  "provisional": False})  # fmt: skip
    for s in specs:
        s["stage"] = "A"
    return specs


def manifest_episode_seq(manifest: Path, ep: int) -> str | None:
    try:
        rows = json.loads(manifest.read_text())
        return next(str(r.get("sequence_id", s)) for s, r in rows.items() if int(r["episode_index"]) == ep)
    except (OSError, ValueError, StopIteration, KeyError, TypeError, AttributeError):
        return None


def t2fix_rows(ep: int, it: int) -> list[dict]:
    return [r for r in load_csv(T2FIX / "eval_ARM/evals.csv")
            if str(r.get("episode")) == str(ep) and str(r.get("iteration")) == str(it) and r.get("status") == "ok"]


def copy_verified(src: Path, dst: Path, expect: str | None) -> str:
    inside_root(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    before = sha256(src)
    if expect and before != expect:
        raise RuntimeError(f"{src}: sha256 {before} != expected {expect}")
    if dst.is_file():
        if sha256(dst) == before:
            return before
        raise RuntimeError(f"{dst} already exists with another sha256")
    tmp = dst.with_name(f".{dst.name}.copying")
    shutil.copy2(src, tmp)
    got, after = sha256(tmp), sha256(src)
    if got != before or after != before:
        tmp.unlink()
        raise RuntimeError(f"{src} changed while it was copied")
    os.replace(tmp, dst)
    return got


def record_copy(src: Path, dst: Path, src_sha: str) -> None:
    new = not (ROOT / "COPIES.tsv").is_file()
    if new:
        append_line(ROOT / "COPIES.tsv", "source\tdestination\tbytes\tsource_sha256\tdestination_sha256\tequal\tcopied_at")
    dsha = sha256(dst)
    append_line(ROOT / "COPIES.tsv", f"{src}\t{dst}\t{dst.stat().st_size}\t{src_sha}\t{dsha}\t{dsha == src_sha}\t{now()}")


def build_job(spec: dict, by: dict, copy: bool, expect_sha: str | None = None, expect_from: str = "") -> dict:
    """One manifest row: every check of PREREG 3.1 (refused + listed on any mismatch, never substituted)."""
    job = dict(spec)
    problems: list[str] = []
    notes: list[str] = []
    ep, it, run_dir = int(spec["episode"]), int(spec["iteration"]), Path(spec["run_dir"])
    train = run_dir / f"model_{it}.safetensors"
    cand, ck = inv_lookup(by, str(run_dir), it)
    if cand is None:
        problems.append(f"no inventory candidate for {run_dir}")
        cand = {}
    if spec["stage"] == "A":
        if ck is None:
            problems.append(f"no inventory checkpoint row for {run_dir} model_{it}")
        else:
            expect_sha, expect_from = ck["sha256"], "inventory.json"
    job.update(object=cand.get("object"), sequence_id=cand.get("sequence_id"), training_checkpoint_original=str(train),
               expected_sha256=expect_sha, expected_sha256_from=expect_from)
    if not train.is_file():
        problems.append(f"missing {train}")
        tsha = None
    else:
        tsha = sha256(train)
        if not expect_sha or tsha != expect_sha:
            problems.append(f"checkpoint sha256 {tsha[:16]} != {expect_from} {str(expect_sha)[:16]}")
    job["training_sha256"] = tsha
    # designation evidence
    des = spec["designation"]
    if des == "final.json":
        fj = read_json(run_dir / "final.json") or {}
        if fj.get("iteration") != it or fj.get("checkpoint") != train.name:
            problems.append(f"final.json names {fj.get('checkpoint')} (iteration {fj.get('iteration')}), not model_{it}")
        job["final_json"] = {k: fj.get(k) for k in ("iteration", "reason", "finished_at")}
    elif des == "t2watch best.json":
        bj = read_json(spec["best_json"]) or {}
        if bj.get("iteration") != it or bj.get("source_sha256") != tsha:
            problems.append(f"best.json iteration {bj.get('iteration')} / source_sha256 {str(bj.get('source_sha256'))[:16]} "
                            f"!= model_{it} {str(tsha)[:16]}")
        job["best_json_evidence"] = {k: bj.get(k) for k in ("iteration", "kind", "evaluated_at", "updated_at", "source_sha256")}
    elif spec["family"] == "zuxing":
        src = ((ck or {}).get("own_input") or {}).get("single_eval", {}).get("src") if ck else None
        rows = load_csv(Path(src)) if src else []
        mine = [r for r in rows if str(r.get("episode")) == str(ep) and str(r.get("iteration")) == str(it)
                and r.get("status") == "ok" and str(r.get("stock_verified")) == "True"
                and (parse_time(r.get("finished_at")) or 9e18) <= FREEZE]  # fmt: skip
        if not mine:
            problems.append(f"no ok stock-verified row for model_{it} by 17:58:27 in {src}")
        elif any(r.get("source_sha256") != tsha for r in mine):
            problems.append(f"his metrics.csv source_sha256 != the file's sha256 {str(tsha)[:16]}")
        job["zuxing_metrics_csv"] = src
        job["zuxing_rows"] = [{k: r.get(k) for k in ("iteration", "add_auc", "sp_sr", "mp_sr", "mppe_cm", "export_max_rel_diff",
                                                     "stock_verified", "source_sha256", "finished_at")} for r in mine]
    elif spec["family"] == "t2fix":
        rows = t2fix_rows(int(spec["t2fix_episode"]), it)
        if spec["stage"] == "A":
            rows = [r for r in rows if (parse_time(r.get("finished_at")) or 9e18) <= FREEZE]
        srcs = []
        for r in rows:
            rep = read_json(Path(r["log"]).parent / "export" / f"ep{ep:02d}" / "export_report.json") or {}
            srcs.append(rep.get("source_sha256"))
        if not rows:
            problems.append(f"no ok eval_ARM row for model_{it}")
        elif any(s != tsha for s in srcs):
            problems.append(f"eval_ARM export source_sha256 {[str(s)[:12] for s in srcs]} != the file's {str(tsha)[:12]}")
        job["t2fix_rows"] = [{k: r.get(k) for k in ("iteration", "auc", "sp_sr", "mp_sr", "purpose", "finished_at")} for r in rows]
    # training parquet
    ov = read_overrides(run_dir)
    pq = cand.get("parquet")
    if ov.get("task.parquet"):
        if pq and os.path.normpath(ov["task.parquet"]) != os.path.normpath(pq):
            problems.append(f"overrides task.parquet {ov['task.parquet']} != inventory {pq}")
        pq = pq or ov["task.parquet"]
    else:
        notes.append("no hydra/.hydra/overrides.yaml task.parquet: the inventory's parquet is used")
    if not pq:
        problems.append("no training parquet")
        job.update(status="refused: " + "; ".join(problems), problems=problems, notes=notes)
        return job
    parquet = Path(pq)
    lay = parquet_layout(parquet)
    if lay is None:
        problems.append(f"unexpected parquet layout {parquet}")
        job.update(status="refused: " + "; ".join(problems), problems=problems, notes=notes)
        return job
    data_root, seq = lay
    kind = root_kind(data_root)
    psha = sha256(parquet) if parquet.is_file() else None
    if kind is None:
        problems.append(f"unknown data root {data_root}")
    if psha is None or psha != cand.get("parquet_sha256"):
        problems.append(f"parquet sha256 {str(psha)[:16]} != inventory {str(cand.get('parquet_sha256'))[:16]}")
    if cand.get("sequence_id") and seq != cand["sequence_id"]:
        problems.append(f"parquet sequence {seq} != inventory {cand['sequence_id']}")
    if kind == "k3":
        twin = ROOTS["k3fix"] / parquet.relative_to(data_root)
        tw = sha256(twin) if twin.is_file() else None
        if tw != psha or psha != cand.get("parquet_sha_k3fix_root"):
            problems.append(f"K3 parquet is not byte-identical to the K3fix one ({str(tw)[:12]} vs {str(psha)[:12]})")
        else:
            notes.append(f"K3 root parquet byte-identical to the K3fix root ({psha[:12]})")
    manifest = MANIFEST_FOR.get(kind) if kind else None
    if manifest is None or manifest_episode_seq(manifest, ep) != seq:
        problems.append(f"manifest {manifest} does not map ep{ep:02d} to {seq}")
    fr = FRAME_REFS.get(kind) if kind else None
    job.update(parquet=str(parquet), parquet_sha256=psha, data_root=str(data_root), root_kind=kind, manifest=str(manifest),
               frame_refs=str(fr) if fr else None, sequence_id=seq)
    # Zuxing copies (PREREG 3.2)
    job["training_checkpoint"] = str(train)
    if spec["family"] == "zuxing":
        dest = ROOT / "zuxing_ckpts" / spec["zuxing_run"] / f"ep{ep:02d}"
        rsi = run_dir / f"rsi_{it}.safetensors"
        job.update(training_checkpoint=str(dest / train.name), rsi_original=str(rsi) if rsi.is_file() else "")
        if copy and not problems:
            try:
                got = copy_verified(train, dest / train.name, tsha)
                record_copy(train, dest / train.name, got)
                sums = [f"{got}  {train.name}"]
                if rsi.is_file():
                    rs = copy_verified(rsi, dest / rsi.name, None)
                    record_copy(rsi, dest / rsi.name, rs)
                    sums.append(f"{rs}  {rsi.name}")
                    job["rsi_sha256"] = rs
                write_atomic(dest / "SHA256SUMS", "\n".join(sums) + "\n")
                write_atomic(dest / "SOURCE.json", dump({"copied_at": now(), "from": str(run_dir), "model": str(train),
                                                         "rsi": job["rsi_original"], "model_sha256": got,
                                                         "evidence": job.get("zuxing_metrics_csv")}))  # fmt: skip
            except (OSError, RuntimeError) as error:
                problems.append(f"copy of Zuxing's checkpoint failed ({error})")
        elif not copy:
            notes.append("dry run: the copy is made when the manifest is built")
    job["problems"], job["notes"] = problems, notes
    job["status"] = "ok" if not problems else "refused: " + "; ".join(problems)
    return job


def priority(job: dict, chosen_sha: dict[int, str]) -> int:
    ep, fam = int(job["episode"]), job["family"]
    if fam in ("k3fix_final", "k3easy_final"):
        return 1
    if fam == "raw":
        return 1 if ep in RAW_AT_WRITING or ep in PENDING_AT_WRITING else 2
    return 3 if job.get("parquet_sha256") == chosen_sha.get(ep) else 4


def assign_priorities(jobs: list[dict]) -> None:
    chosen_sha: dict[int, str] = {}
    for j in jobs:
        ep = int(j["episode"])
        if j["family"] in ("k3fix_final", "k3easy_final") or (j["family"] == "raw" and ep not in K3_CHOSEN):
            chosen_sha.setdefault(ep, j.get("parquet_sha256"))
    for j in jobs:
        j["priority"] = j.get("priority") or priority(j, chosen_sha)


# ================================================================================================ hulls
def hull_check(jobs: list[dict], log) -> dict[str, dict]:
    """CoACD cache keys of each distinct training parquet (cache_keys_t2.py, read-only, CPU): every key's file must
    already exist in the stock runtime's convex cache. Returns parquet -> {ok, keys, missing}."""
    parquets = sorted({j["parquet"] for j in jobs if j.get("parquet")})
    if not parquets:
        return {}
    args = [f"{i}:{p}" for i, p in enumerate(parquets)]
    env = child_env(None, f"{ROOT}/preflight/hulls", None)
    env.update(PYTHONPATH=str(TREES["fc_r2"]["code"] / "src"), JAX_PLATFORMS="cpu")
    out_dir = inside_root(ROOT / "preflight")
    out_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "tmp").mkdir(exist_ok=True)
    cmd = ["nice", "-n", "10", "taskset", "-c", EVAL_CORES[1], str(PY), "-B", str(CACHE_KEYS), *args]
    log(f"hull preflight: {len(parquets)} parquets with {CACHE_KEYS.name} (read-only, CPU)")
    proc = subprocess.run(cmd, cwd=ROOT / "tmp", env=env, capture_output=True, text=True)
    stamp_ = stamp()
    write_atomic(out_dir / f"hulls_{stamp_}.log", f"$ {shlex.join(cmd)}\nexit {proc.returncode}\n{proc.stdout}\n{proc.stderr}")
    line = next((x for x in proc.stdout.splitlines() if x.startswith("JSON ")), None)
    res: dict[str, dict] = {}
    data = json.loads(line[5:]) if line else {"objects": {}}
    for i, p in enumerate(parquets):
        per = data["objects"].get(str(i))
        if per is None:
            res[p] = {"ok": False, "keys": [], "missing": ["cache_keys_t2.py gave no result"]}
            continue
        keys = [k["key"] for k in per["keys"]]
        missing = [f for f, row in per["files"].items() if not row.get("stock", {}).get("present")]
        missing += [f"{k}.npz" for k in keys if not (STOCK_CONVEX / f"{k}.npz").is_file() and f"{k}.npz" not in missing]
        res[p] = {"ok": bool(keys) and not missing, "keys": keys, "missing": missing}
    write_atomic(out_dir / f"hulls_{stamp_}.json", dump({"checked_at": now(), "exit": proc.returncode, "parquets": res}))
    return res


# ================================================================================================ checks
def parquet_problems(pp: Path, sha: str, step_count) -> list[str]:
    import pyarrow.parquet as pq

    f = pq.ParquetFile(pp)
    md = {k.decode(errors="replace"): v.decode(errors="replace") for k, v in (f.schema_arrow.metadata or {}).items()
          if not k.startswith(b"pandas")}  # fmt: skip
    out = []
    if md.get("flash_chord.checkpoint_sha256") != sha:
        out.append("parquet checkpoint_sha256 != the exported file")
    if md.get("flash_chord.eval_script_sha256") != EVAL_SCRIPT_SHA:
        out.append(f"parquet eval_script_sha256 {md.get('flash_chord.eval_script_sha256')} is not the common build")
    if "flash_chord.eval_code_sha256" in md and md.get("flash_chord.eval_code_sha256") != EVAL_CODE_SHA:
        out.append(f"parquet eval_code_sha256 {md.get('flash_chord.eval_code_sha256')} is not the common build")
    if str(md.get("flash_chord.step_count")) != str(step_count):
        out.append(f"parquet step_count {md.get('flash_chord.step_count')} != JSON {step_count}")
    valid = pq.read_table(pp, columns=["environment_valid"]).column("environment_valid").to_pylist()
    if not valid or not all(valid):
        out.append("environment_valid=false rows")
    return out


def classify_refusal(rep: dict) -> str:
    """'tol' when every failure is a finite max_rel_diff above the tolerance (log_std equal), else 'other'."""
    fails = rep.get("failures") or []
    dev = rep.get("dev_restore") or {}
    stock = (rep.get("stock_runtime") or {})
    comp = stock.get("comparison") or {}
    if not fails or not dev.get("finite", True) or not comp.get("finite", True):
        return "other"
    if rep.get("log_std_equal") is False or stock.get("log_std_equal") is False:
        return "other"
    if stock and stock.get("algorithm") not in (None, "ppo"):
        return "other"
    for f in fails:
        if not (re.search(r"max_rel_diff [0-9.eE+-]+ > ", f) or ("stock runtime" in f and "max_rel_diff" in f)):
            return "other"
    return "tol"


def check_export(job: dict, d: Path, exit_code, tol: float) -> tuple[str, dict]:
    """'ok' | 'refused_tol' | 'refused_other' | 'failed' (technical)."""
    src = Path(job["training_checkpoint"])
    dst = d / src.name
    info: dict = {"file": str(dst), "report": str(d / "export_report.json"), "exit": exit_code, "tolerance": tol,
                  "problems": []}  # fmt: skip
    problems = info["problems"]
    rep = read_json(d / "export_report.json")
    if rep is None:
        problems.append(f"exit {exit_code}; no export_report.json")
        return "failed", info
    stock = rep.get("stock_runtime") or {}
    info.update(max_rel_diff=rep.get("max_rel_diff"), max_abs_diff=rep.get("max_abs_diff"),
                report_tolerance=rep.get("tolerance"), source_sha256=rep.get("source_sha256"),
                destination_sha256=rep.get("destination_sha256"), stock_verified=stock.get("verified"),
                export_format=rep.get("export_format"), source_kind=rep.get("source_kind"), report_status=rep.get("status"))
    if rep.get("status") == "refused":
        problems.append(f"exporter refused: {rep.get('failures')}")
        return ("refused_tol" if classify_refusal(rep) == "tol" else "refused_other"), info
    if exit_code not in (0, None):
        problems.append(f"exit {exit_code}")
    if rep.get("status") != "ok":
        problems.append(f"report status {rep.get('status')}")
    if fnum(rep.get("tolerance")) != tol:
        problems.append(f"tolerance {rep.get('tolerance')} != {tol}")
    if stock.get("verified") is not True:
        problems.append(f"stock runtime verified={stock.get('verified')}")
    if os.path.realpath(str(stock.get("path", ""))) != os.path.realpath(STOCK):
        problems.append(f"verified against {stock.get('path')}, not {STOCK}")
    if stock.get("eval_code_sha256") not in (None, EVAL_CODE_SHA):
        problems.append(f"stock eval_code_sha256 {stock.get('eval_code_sha256')} is not the common build")
    if stock.get("curriculum_stage_voc_scale") not in (None, 0, 0.0):
        problems.append(f"curriculum_stage_voc_scale {stock.get('curriculum_stage_voc_scale')} (help) != 0")
    rel = fnum(rep.get("max_rel_diff"))
    if rel is None or rel > tol:
        problems.append(f"max_rel_diff {rep.get('max_rel_diff')} > {tol}")
    if rep.get("source_sha256") != job["training_sha256"]:
        problems.append("report source_sha256 != the manifest's training_sha256")
    if not dst.is_file():
        problems.append("exported file missing")
    else:
        info["sha256"] = sha256(dst)
        if info["sha256"] != rep.get("destination_sha256"):
            problems.append("exported file sha256 != report destination_sha256")
    return ("ok" if not problems else "failed"), info


def check_eval(job: dict, exp: dict, results: Path, report: Path | None, exit_code, keys: dict) -> tuple[list[str], dict]:
    ep = int(job["episode"])
    problems: list[str] = []
    if exit_code not in (0, None):
        problems.append(f"exit {exit_code}")
    if report is not None:
        rep = read_json(report) or {}
        erow = next((r for r in rep.get("episodes", []) if str(r.get("episode")) == str(ep)), None)
        if erow is None:
            problems.append("no eval_report.json row for the episode")
        elif erow.get("status") != "evaluated":
            problems.append(f"eval_report status {erow.get('status')}")
    jp, pp = results / f"episode_{ep:06d}.json", results / f"episode_{ep:06d}.parquet"
    info: dict = {"results_json": str(jp), "results_parquet": str(pp)}
    js = read_json(jp)
    if js is None:
        problems.append("no results JSON")
        return problems, info
    m = js.get("metrics") or {}
    info["metrics"] = {k: fnum(m.get(src)) for k, src in keys.items()}
    info.update(chord_sr=fnum(m.get("chord_sr")), checkpoint_sha256=js.get("checkpoint_sha256"),
                checkpoint=js.get("checkpoint"), reference_parquet=js.get("parquet"), step_count=js.get("step_count"))
    if any(v is None for v in info["metrics"].values()):
        problems.append(f"metrics missing ({info['metrics']})")
    if js.get("checkpoint_sha256") != exp.get("sha256"):
        problems.append("results JSON checkpoint_sha256 != the exported file")
    if os.path.realpath(str(js.get("checkpoint", ""))) != os.path.realpath(str(exp.get("file"))):
        problems.append(f"evaluated checkpoint path {js.get('checkpoint')} is not the exported file")
    ref = js.get("parquet")
    if not ref or os.path.realpath(ref) != os.path.realpath(job["parquet"]):
        problems.append(f"reference parquet {ref} is not the training parquet {job['parquet']}")
    else:
        rsha = sha256(ref)
        info["reference_parquet_sha256"] = rsha
        if rsha != job["parquet_sha256"]:
            problems.append(f"reference parquet sha256 {rsha[:16]} != the manifest's {job['parquet_sha256'][:16]}")
    for key, want in (("start_frame", 0), ("motion_start_frame", 0), ("world_count", 1), ("reset_mode", "explicit")):
        if js.get(key) != want:
            problems.append(f"{key}={js.get(key)!r}, expected {want!r}")
    if js.get("non_finite_worlds"):
        problems.append(f"non_finite_worlds {js.get('non_finite_worlds')}")
    if not pp.is_file():
        problems.append("no rollout parquet")
    else:
        try:
            problems += parquet_problems(pp, exp.get("sha256"), js.get("step_count"))
        except Exception as error:  # noqa: BLE001
            problems.append(f"parquet unreadable ({error!r})")
    if not Path(str(exp.get("file"))).is_file():
        problems.append("the exported file is MISSING")
    elif sha256(exp["file"]) != exp["sha256"]:
        problems.append("the exported file CHANGED")
    return problems, info


def parse_clean(out_csv: Path, ep: int) -> dict | None:
    rows = [r for r in load_csv(out_csv) if str(r.get("episode", "")).strip().isdigit() and int(r["episode"]) == ep]
    if not rows:
        return None
    vals = {m: fnum(rows[0].get(col)) for m, col in CLEAN_COLS.items()}
    return vals if all(v is not None for v in vals.values()) else None


# ================================================================================================ commands
def export_command(job: dict, d: Path, tol: float) -> tuple[list[str], Path, dict]:
    code = TREES[job["tree"]]["code"]
    src = Path(job["training_checkpoint"])
    cmd = [str(PY), "-B", "-m", EXPORTER, "--input", str(src), "--output", str(d / src.name), "--stock-runtime", str(STOCK),
           "--report", str(d / "export_report.json"), "--tolerance", repr(tol)]  # fmt: skip
    return cmd, code, {"PYTHONPATH": str(code / "src"), "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""}


def eval_command(job: dict, w: Path, gpu: int) -> tuple[list[str], Path]:
    code = TREES[job["tree"]]["code"]
    cmd = [str(PY), "-B", str(code / "scripts/track2/eval_all.py"), "--arm-dir", str(w), "--arm", "T2best5", "--export", "no",
           "--iteration", str(job["iteration"]), "--episodes", str(job["episode"]), "--gpus", str(gpu),
           "--jobs-per-gpu", "1", "--world-count", "1", "--skip-pack", "--kit-root", str(KIT), "--stock-runtime", str(STOCK),
           "--dataset-root", str(DATASET), "--manifest", str(job["manifest"]), "--pack-python", str(KPY),
           "--results", str(w / "results"), "--code", str(code), "--python", str(PY)]  # fmt: skip
    return cmd, code


def clean_command(job: dict, rd: Path, results: Path) -> list[str]:
    cmd = [str(KPY), "-I", "-B", str(CLEAN_SCORER), "--kit", str(KIT), "--out", str(rd / "clean_scores_LOCAL_NOT_FOR_UPLOAD.csv"),
           "--", "--track", "track_2_tier2", "--results", str(results), "--dataset-root", str(DATASET),
           "--code-commit", COMMIT_URL, "--eval-script", str(KIT / "flash_chord/scripts/evaluate_policy.py")]  # fmt: skip
    if job.get("frame_refs"):
        cmd += ["--frame-references", str(job["frame_refs"])]
    return cmd + ["--config", str(rd / "pack" / "tracks.json"), "-o", str(rd / "packed_LOCAL_NOT_FOR_UPLOAD.csv")]


# ================================================================================================ records
def export_rec(base: Path, job_id: str) -> dict | None:
    return read_json(base / "exports" / job_id / "export.json")


def run_rec(base: Path, job_id: str, k: int) -> dict | None:
    return read_json(base / "runs" / job_id / f"run{k}" / "run.json")


def clean_rec(base: Path, job_id: str, k: int) -> dict | None:
    return read_json(base / "runs" / job_id / f"run{k}" / "clean_LOCAL_NOT_FOR_UPLOAD.json")


def load_jobs(base: Path = ROOT) -> list[dict]:
    """Stage A rows (manifest.json) + Stage B rows (manifest_stageB.jsonl), all statuses."""
    data = read_json(base / "manifest.json")
    rows = list(data["rows"]) if data else []
    try:
        for line in (base / "manifest_stageB.jsonl").read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    except OSError:
        pass
    return rows


def resolutions(base: Path = ROOT) -> list[dict]:
    out = []
    try:
        for line in (base / "stageB_resolutions.jsonl").read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
    except OSError:
        pass
    return out


def settled_overrides(base: Path = ROOT) -> set[str]:
    return {r["job"] for r in resolutions(base) if r.get("event") == "settled" and r.get("job")}


# ================================================================================================ selection (own only)
def guarded_read_json(p: Path):
    """The selection's only file reader: it refuses LOCAL / clean / packed paths (PREREG 2.8)."""
    s = str(p)
    if any(tok in s for tok in FORBIDDEN):
        raise PermissionError(f"selection may not open {s}")
    return read_json(p)


def own_runs(base: Path, job_id: str, nruns: int) -> tuple[dict[int, dict], int]:
    """(ok runs k -> record, number of final slots) from run<k>/run.json (own-input records only)."""
    ok, final = {}, 0
    for k in range(1, nruns + 1):
        r = guarded_read_json(base / "runs" / job_id / f"run{k}" / "run.json")
        if r is not None:
            final += 1
            if r.get("status") == "ok":
                ok[k] = r
    return ok, final


def mean_of(recs: dict[int, dict], m: str) -> float | None:
    vals = [recs[k]["metrics"][m] for k in sorted(recs)]
    return sum(vals) / len(vals) if vals else None


def select_all(jobs: list[dict], nruns: int, base: Path = ROOT, verdict_path: Path = VERDICT) -> dict:
    """PREREG section 2, per episode. Reads only: the manifest rows given, VERDICT.json, run<k>/run.json."""
    verdict = guarded_read_json(verdict_path) or {}
    vsha = sha256(verdict_path) if Path(verdict_path).is_file() else None
    pre_ok = PREREG_V3_SHA[:16] in str(verdict.get("prereg_check", ""))
    settled_extra = settled_overrides(base)
    ok_jobs = [j for j in jobs if j.get("status") == "ok"]
    vby = {int(e["episode"]): e for e in verdict.get("episodes", [])}
    eps_out = []
    for ep in range(30):
        mine = [j for j in ok_jobs if int(j["episode"]) == ep]
        ve = vby.get(ep, {})
        choice = ve.get("choice") if pre_ok else None
        flags = []
        raw = next((j for j in mine if j["family"] == "raw"), None)
        c = None
        if choice == "K3FIX":
            c = next((j for j in mine if j["family"] in ("k3fix_final", "k3easy_final")
                      and j["iteration"] == ve.get("final_iteration")
                      and os.path.normpath(j["run_dir"]).endswith("/" + str(ve.get("k3fix_run")))), None)  # fmt: skip
            if c is None:
                flags.append("PENDING (VERDICT K3FIX; Stage-B job not yet in the manifest)")
                c = raw
        elif choice == "RAW":
            c = raw
        else:
            flags.append("PENDING" if choice == "PENDING" else f"VERDICT unusable (choice {choice}, prereg_check ok {pre_ok})")
            c = raw
        if c is None:
            eps_out.append({"episode": ep, "flags": flags + ["NO CHOSEN JOB"], "chosen": None, "selected": None,
                            "selected_p": None, "rawref": None})
            continue
        c_ok, c_final = own_runs(base, c["job"], nruns)
        c_complete = len(c_ok) == nruns
        if not c_complete and c_final == nruns:
            flags.append("INCOMPLETE")

        def stats(recs):
            return {m: mean_of(recs, m) for m in METRICS}

        cst = stats(c_ok)
        chal = []
        for s in mine:
            if s["job"] == c["job"] or s.get("parquet_sha256") != c.get("parquet_sha256"):
                continue
            s_ok, s_final = own_runs(base, s["job"], nruns)
            sst = stats(s_ok)
            settled = (not s.get("provisional")) or s["job"] in settled_extra
            why = []
            if not c_complete:
                why.append(f"CHOSEN has {len(c_ok)}/{nruns} ok")
            if len(s_ok) != nruns:
                why.append(f"challenger has {len(s_ok)}/{nruns} ok")
            if not why:
                if not sst["auc"] >= cst["auc"] + 0.01:
                    why.append("mean AUC < CHOSEN + 0.01")
                if not sst["sp_sr"] >= cst["sp_sr"]:
                    why.append("mean SP-SR lower")
                if not sst["mp_sr"] >= cst["mp_sr"]:
                    why.append("mean MP-SR lower")
            chal.append({"job": s["job"], "family": s["family"], "settled": settled, "n_ok": len(s_ok), "n_final": s_final,
                         "mean": sst, "qualifies": not why, "why_not": why,
                         "order": (LIST_ORDER.get(s["family"], 3), s.get("stage", "A"), int(s["iteration"]))})

        def pick(pool):
            q = [x for x in pool if x["qualifies"]]
            if not q:
                return c["job"], "CHOSEN kept (no qualifying challenger)"
            q.sort(key=lambda x: (-x["mean"]["auc"], -x["mean"]["sp_sr"], -x["mean"]["mp_sr"], x["mean"]["mppe"], x["order"]))
            return q[0]["job"], f"replaced by {q[0]['job']} (mean AUC {q[0]['mean']['auc']:.4f} >= {cst['auc']:.4f} + 0.01, SP/MP not lower)"

        sel, sel_why = pick([x for x in chal if x["settled"]])
        selp, selp_why = pick(chal)
        pending_evals = c_final < nruns or any(x["n_final"] < nruns for x in chal)
        eps_out.append({
            "episode": ep, "object": c.get("object"), "verdict_choice": ve.get("choice"), "verdict_settled": ve.get("settled"),
            "flags": flags, "chosen": c["job"], "chosen_input": c.get("root_kind"), "chosen_parquet": c.get("parquet_sha256"),
            "chosen_n_ok": len(c_ok), "chosen_n_final": c_final, "chosen_mean": cst, "challengers": chal,
            "selected": sel, "selected_reason": sel_why, "selected_p": selp, "selected_p_reason": selp_why,
            "rawref": raw["job"] if raw else None, "evaluations_pending": pending_evals,
            "same_input_other": [j["job"] for j in mine if j["job"] != c["job"] and j.get("parquet_sha256") == c.get("parquet_sha256")],
            "diff_input": [j["job"] for j in mine if j.get("parquet_sha256") != c.get("parquet_sha256") and j["job"] != (raw or {}).get("job")],
        })  # fmt: skip
    return {"generated_at": now(), "verdict_sha256": vsha, "verdict_written": verdict.get("written"),
            "verdict_prereg_check_ok": pre_ok, "rule": "PREREG section 2 (own-input only; +0.01 mean5 AUC, SP/MP not "
            "lower; same training parquet sha256; SELECTED settled only, SELECTED+P incl. provisional)", "episodes": eps_out}


def compositions(sel: dict) -> dict[str, dict[int, str]]:
    out: dict[str, dict[int, str]] = {"RAW-ONLY": {}, "K3FULL": {}, "SELECTED": {}, "SELECTED+P": {}, "ZUXING-LIVE": {}}
    for e in sel["episodes"]:
        ep = e["episode"]
        if e.get("rawref"):
            out["RAW-ONLY"][ep] = e["rawref"]
        if e.get("chosen"):
            out["K3FULL"][ep] = e["chosen"]
        if e.get("selected"):
            out["SELECTED"][ep] = e["selected"]
            out["ZUXING-LIVE"][ep] = ZUXING_LIVE.get(ep, e["selected"])
        if e.get("selected_p"):
            out["SELECTED+P"][ep] = e["selected_p"]
    return out


def comp_stats(mapping: dict[int, str], values: dict[str, dict[int, dict]], nruns: int) -> dict:
    """values: job -> {k: {metric: v}}. Average, repeat totals T_k (min/max), per-episode bounds, per metric."""
    per_ep = {}
    for ep, job in sorted(mapping.items()):
        vals = values.get(job, {})
        per_ep[ep] = {"job": job, "n": len(vals), "runs": {k: vals[k] for k in sorted(vals)}}
        for m in METRICS:
            xs = [vals[k][m] for k in sorted(vals)]
            per_ep[ep][m] = {"mean": sum(xs) / len(xs) if xs else None, "min": min(xs) if xs else None,
                             "max": max(xs) if xs else None}  # fmt: skip
    have = [ep for ep in per_ep if per_ep[ep]["n"] > 0]
    partial = len(mapping) < 30 or any(per_ep[ep]["n"] < nruns for ep in per_ep)
    res = {"n_episodes": len(mapping), "n_with_values": len(have), "partial": partial,
           "n_per_episode": {ep: per_ep[ep]["n"] for ep in per_ep}, "per_episode": per_ep, "metrics": {}}
    for m in METRICS:
        means = [per_ep[ep][m]["mean"] for ep in have]
        tk, nk = {}, {}
        for k in range(1, nruns + 1):
            xs = [per_ep[ep]["runs"][k][m] for ep in per_ep if k in per_ep[ep]["runs"]]
            nk[k] = len(xs)
            tk[k] = sum(xs) / len(xs) if xs else None
        tks = [v for v in tk.values() if v is not None]
        res["metrics"][m] = {
            "average": sum(means) / len(means) if means else None,
            "T_k": tk, "n_k": nk, "T_min": min(tks) if tks else None, "T_max": max(tks) if tks else None,
            "mean_of_episode_min": (sum(per_ep[ep][m]["min"] for ep in have) / len(have)) if have else None,
            "mean_of_episode_max": (sum(per_ep[ep][m]["max"] for ep in have) / len(have)) if have else None,
        }
    return res


# ================================================================================================ summaries
def fv(value, spec: str = ".4f") -> str:
    return "" if value in (None, "") else format(float(value), spec)


def job_stats_row(job: dict, recs: dict[int, dict], nruns: int, prefix: str = "") -> dict:
    out = {}
    for m in METRICS:
        xs = [recs[k][m] for k in sorted(recs)]
        for k in range(1, nruns + 1):
            out[f"{prefix}{m}_r{k}"] = recs[k][m] if k in recs else ""
        out[f"{prefix}{m}_mean"] = sum(xs) / len(xs) if xs else ""
        out[f"{prefix}{m}_sd"] = statistics.stdev(xs) if len(xs) >= 2 else (0.0 if xs else "")
        out[f"{prefix}{m}_min"] = min(xs) if xs else ""
        out[f"{prefix}{m}_max"] = max(xs) if xs else ""
    if recs:
        mu = out[f"{prefix}auc_mean"]
        out[f"{prefix}closest_to_mean_run"] = min(recs, key=lambda k: (abs(recs[k]["auc"] - mu), k))
    return out


def earlier_text(job: dict, by: dict) -> str:
    _, ck = inv_lookup(by, job["run_dir"], int(job["iteration"]))
    parts = []
    for label, v in ((ck or {}).get("own_input") or {}).items():
        if isinstance(v, dict) and fnum(v.get("auc")) is not None:
            parts.append(f"{label} {float(v['auc']):.4f} [{v.get('n', 1)}]")
    return "; ".join(parts)


def summarize(jobs: list[dict], nruns: int, base: Path = ROOT, log=None, verdict_path: Path = VERDICT) -> dict:
    """selection.* (own only) -> summary_own_input.* -> summary_LOCAL_NOT_FOR_UPLOAD.* (in that order)."""
    sel = select_all(jobs, nruns, base, verdict_path)
    write_atomic(base / "selection.json", dump(sel))
    lines = [f"selection {sel['generated_at']} (own-input only; VERDICT {sel['verdict_written']} sha {str(sel['verdict_sha256'])[:12]})"]
    for e in sel["episodes"]:
        lines.append(f"ep{e['episode']:02d} CHOSEN {e.get('chosen')} [{e.get('chosen_n_ok')}/{nruns} ok] -> SELECTED "
                     f"{e.get('selected')} ({e.get('selected_reason')}); SELECTED+P {e.get('selected_p')}"
                     + (f"; flags {e['flags']}" if e.get("flags") else "") + ("; evaluations pending" if e.get("evaluations_pending") else ""))
        for x in e.get("challengers", []):
            lines.append(f"     challenger {x['job']} settled={x['settled']} n_ok={x['n_ok']} mean AUC {fv(x['mean']['auc'])} "
                         f"qualifies={x['qualifies']} {x['why_not']}")
    write_atomic(base / "selection.txt", "\n".join(lines) + "\n")
    comps = compositions(sel)
    try:
        _, by = inventory_index()
    except (OSError, ValueError):
        by = {}
    # ---------------------------------------------------------------- own input
    okjobs = [j for j in jobs if j.get("status") == "ok"]
    own_vals: dict[str, dict[int, dict]] = {}
    eval_rows, job_rows = [], []
    for j in okjobs:
        exp = export_rec(base, j["job"]) or {}
        recs = {}
        failed = []
        for k in range(1, nruns + 1):
            r = run_rec(base, j["job"], k)
            if r is None:
                continue
            eval_rows.append({"job": j["job"], "episode": j["episode"], "run": k, "status": r.get("status"), "try": r.get("try"),
                              "gpu": r.get("gpu"), "seconds": r.get("seconds"), **(r.get("metrics") or {}),
                              "chord_sr": r.get("chord_sr"), "export_sha256": str(r.get("export_sha256"))[:16],
                              "finished_at": r.get("finished_at"), "problems": "; ".join(r.get("problems") or [])})  # fmt: skip
            if r.get("status") == "ok":
                recs[k] = r["metrics"]
            else:
                failed.append(k)
        own_vals[j["job"]] = recs
        job_rows.append({"job": j["job"], "episode": j["episode"], "object": j.get("object"), "family": j["family"],
                         "owner": j["owner"], "stage": j.get("stage"), "provisional": bool(j.get("provisional")) and j["job"] not in settled_overrides(base),
                         "input": j.get("root_kind"), "parquet_sha256": str(j.get("parquet_sha256"))[:8],
                         "iteration": j["iteration"], "checkpoint_sha256": str(j.get("training_sha256"))[:8],
                         "priority": j.get("priority"), "export_status": exp.get("status", "pending"),
                         "export_tolerance": exp.get("tolerance"), "export_fallback": exp.get("fallback"),
                         "export_sha256": str(exp.get("sha256", ""))[:16], "n_ok": len(recs), "failed_runs": " ".join(map(str, failed)),
                         **job_stats_row(j, recs, nruns), "earlier_evals_inventory": earlier_text(j, by)})  # fmt: skip
    for j in jobs:
        if j.get("status") != "ok":
            job_rows.append({"job": j["job"], "episode": j["episode"], "family": j["family"], "owner": j["owner"],
                             "export_status": j.get("status")})
    cols = ["job", "episode", "object", "family", "owner", "stage", "provisional", "input", "parquet_sha256", "iteration",
            "checkpoint_sha256", "priority", "export_status", "export_tolerance", "export_fallback", "export_sha256", "n_ok",
            "failed_runs"]  # fmt: skip
    for m in METRICS:
        cols += [f"{m}_r{k}" for k in range(1, nruns + 1)] + [f"{m}_{s}" for s in ("mean", "sd", "min", "max")]
    cols += ["closest_to_mean_run", "earlier_evals_inventory"]
    write_csv(base / "own_input_evals.csv", eval_rows, ["job", "episode", "run", "status", "try", "gpu", "seconds", *METRICS,
                                                        "chord_sr", "export_sha256", "finished_at", "problems"])  # fmt: skip
    write_csv(base / "summary_own_input.csv", job_rows, cols)
    own_comp = {name: comp_stats(mp, own_vals, nruns) for name, mp in comps.items()}
    own = {"generated_at": now(), "runs_per_job": nruns, "selection": "selection.json", "compositions": own_comp,
           "composition_jobs": comps, "jobs": job_rows,
           "note": "own-input composites mix different references (raw, K3fix, K3, G2): not leaderboard numbers"}
    write_atomic(base / "summary_own_input.json", dump(own))
    md = own_markdown(sel, comps, own_comp, job_rows, jobs, nruns, base)
    write_atomic(base / "summary_own_input.md", md)
    write_atomic(base / "summary.md", md)  # alias (task text names summary.csv / summary.md for the own-input summary)
    write_csv(base / "summary.csv", job_rows, cols)
    # ---------------------------------------------------------------- LOCAL clean (report only, after the selection)
    clean_vals: dict[str, dict[int, dict]] = {}
    crow, cjob = [], []
    for j in okjobs:
        recs = {}
        for k in range(1, nruns + 1):
            cr = clean_rec(base, j["job"], k)
            if cr is None:
                continue
            crow.append({"job": j["job"], "episode": j["episode"], "run": k, "score_status": cr.get("status"),
                         **(cr.get("values") or {}), "frame_refs": cr.get("frame_refs"),
                         "input_check_residual_m": cr.get("input_check_residual_m"),
                         "input_check_residual_rad": cr.get("input_check_residual_rad"), "finished_at": cr.get("finished_at")})
            if cr.get("status") == "ok" and k in own_vals.get(j["job"], {}):
                recs[k] = cr["values"]
        clean_vals[j["job"]] = recs
        cjob.append({"job": j["job"], "episode": j["episode"], "object": j.get("object"), "family": j["family"],
                     "input": j.get("root_kind"), "iteration": j["iteration"], "n_clean_ok": len(recs),
                     "own_auc_mean": next((r["auc_mean"] for r in job_rows if r["job"] == j["job"]), ""),
                     **job_stats_row(j, recs, nruns, prefix="clean_")})  # fmt: skip
    ccols = ["job", "episode", "object", "family", "input", "iteration", "n_clean_ok", "own_auc_mean"]
    for m in METRICS:
        ccols += [f"clean_{m}_r{k}" for k in range(1, nruns + 1)] + [f"clean_{m}_{s}" for s in ("mean", "sd", "min", "max")]
    write_csv(base / "clean_evals_LOCAL_NOT_FOR_UPLOAD.csv", crow,
              ["job", "episode", "run", "score_status", *METRICS, "frame_refs", "input_check_residual_m",
               "input_check_residual_rad", "finished_at"])  # fmt: skip
    write_csv(base / "summary_LOCAL_NOT_FOR_UPLOAD.csv", cjob, ccols)
    write_csv(base / "summary_clean_LOCAL_NOT_FOR_UPLOAD.csv", cjob, ccols)
    clean_comp = {name: comp_stats(mp, clean_vals, nruns) for name, mp in comps.items()}
    write_atomic(base / "summary_LOCAL_NOT_FOR_UPLOAD.json", dump({"generated_at": now(), "compositions": clean_comp,
                                                                   "composition_jobs": comps, "jobs": cjob,
                                                                   "note": "LOCAL clean (vs the clean Tier-1 reference, "
                                                                   "the run's frame references): REPORT ONLY, never "
                                                                   "used to select"}))  # fmt: skip
    cmd_text = clean_markdown(sel, comps, own_comp, clean_comp, own_vals, clean_vals, jobs, nruns)
    write_atomic(base / "summary_LOCAL_NOT_FOR_UPLOAD.md", cmd_text)
    write_atomic(base / "summary_clean_LOCAL_NOT_FOR_UPLOAD.md", cmd_text)
    return {"selection": sel, "own": own_comp, "clean": clean_comp}


COMP_ORDER = ("SELECTED", "SELECTED+P", "K3FULL", "RAW-ONLY", "ZUXING-LIVE")


def comp_table(comp: dict, title: str) -> list[str]:
    md = [f"### {title}", "",
          "| composition | eps | n runs/ep | AUC average | AUC lowest T_k | AUC highest T_k | AUC mean of ep-min | AUC mean of ep-max "
          "| SP-SR avg | MP-SR avg | MPPE avg (cm) | T_1..T_5 (AUC) |",
          "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for name in COMP_ORDER:
        c = comp[name]
        a = c["metrics"]["auc"]
        ns = sorted(set(c["n_per_episode"].values()))
        md.append(f"| {name}{' (PARTIAL)' if c['partial'] else ''} | {c['n_with_values']}/{c['n_episodes']} | "
                  f"{'-'.join(map(str, [ns[0], ns[-1]])) if ns else ''} | {fv(a['average'])} | {fv(a['T_min'])} | {fv(a['T_max'])} | "
                  f"{fv(a['mean_of_episode_min'])} | {fv(a['mean_of_episode_max'])} | {fv(c['metrics']['sp_sr']['average'], '.3f')} | "
                  f"{fv(c['metrics']['mp_sr']['average'], '.3f')} | {fv(c['metrics']['mppe']['average'], '.2f')} | "
                  + " ".join(fv(a['T_k'][k]) or '-' for k in sorted(a['T_k'])) + " |")
    return md


def own_markdown(sel, comps, own_comp, job_rows, jobs, nruns, base) -> str:
    rows = {r["job"]: r for r in job_rows}
    md = [f"# Tier-2 5x official re-evaluation: OWN-INPUT summary ({now()})", "",
          "Own-input AUC = each policy evaluated against its OWN training parquet (official kit evaluator, stock 8c778f9, "
          "world count 1). It is comparable only between jobs with the same training parquet. Composites mix references "
          "(raw / K3fix / K3 / G2) and are NOT leaderboard numbers. Selection: `selection.json` (PREREG section 2).", "",
          "\"Lowest / highest T_k\": T_k = the mean over the episodes of run k (k = 1..5, run-index order); the mean of "
          "the T_k is the average. \"Mean of ep-min / ep-max\": the mean of each episode's worst / best of 5.", ""]
    md += comp_table(own_comp, "Compositions (own input)")
    md += ["", "### Per episode (own input)", "",
           "| ep | object | VERDICT | CHOSEN | SELECTED | SELECTED+P | flags | SELECTED own min | max | mean | n | other candidates (mean own AUC [n], input) |",
           "|---:|---|---|---|---|---|---|---:|---:|---:|---:|---|"]
    for e in sel["episodes"]:
        s = rows.get(e.get("selected") or "", {})
        others = []
        for j in jobs:
            if j.get("status") == "ok" and int(j["episode"]) == e["episode"] and j["job"] not in (e.get("selected"),):
                r = rows.get(j["job"], {})
                others.append(f"{j['job']} {fv(r.get('auc_mean'))} [{r.get('n_ok', 0)}] {j.get('root_kind')}"
                              + (" P" if r.get("provisional") else ""))
        md.append(f"| {e['episode']} | {e.get('object', '')} | {e.get('verdict_choice')} | {e.get('chosen')} | {e.get('selected')} | "
                  f"{e.get('selected_p') if e.get('selected_p') != e.get('selected') else '='} | {'; '.join(e.get('flags', []))}"
                  f"{' evals pending' if e.get('evaluations_pending') else ''} | {fv(s.get('auc_min'))} | {fv(s.get('auc_max'))} | "
                  f"{fv(s.get('auc_mean'))} | {s.get('n_ok', 0)} | {'; '.join(others)} |")
    md += ["", "### Per job (own input)", "",
           "| job | input | P | export | AUC runs | mean | sd | min | max | SP | MP | MPPE | earlier (inventory) |",
           "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in job_rows:
        runs = " ".join(fv(r.get(f"auc_r{k}")) or "-" for k in range(1, nruns + 1))
        md.append(f"| {r['job']} | {r.get('input', '')} {r.get('parquet_sha256', '')} | {'P' if r.get('provisional') else ''} | "
                  f"{r.get('export_status', '')} {r.get('export_tolerance') or ''} | {runs} | {fv(r.get('auc_mean'))} | "
                  f"{fv(r.get('auc_sd'))} | {fv(r.get('auc_min'))} | {fv(r.get('auc_max'))} | {fv(r.get('sp_sr_mean'), '.2f')} | "
                  f"{fv(r.get('mp_sr_mean'), '.2f')} | {fv(r.get('mppe_mean'), '.2f')} | {r.get('earlier_evals_inventory', '')} |")
    t2 = read_json(T2FIX / "verdict.json") or {}
    md += ["", "### T2fix's own 3-eval verdict (fc_runs/T2fix/verdict.json) next to this file's decision", ""]
    for arm, (ep, _) in STAGEB_T2FIX.items():
        v = t2.get(arm) or {}
        e = next((x for x in sel["episodes"] if x["episode"] == ep), {})
        md.append(f"- {arm}: T2fix verdict {v.get('verdict')} (arm best iteration {v.get('arm_best_iteration')}); this file: "
                  f"SELECTED {e.get('selected')}" + ("" if str(e.get("selected", "")).find("t2fix") >= 0 or v.get("verdict") != "REPLACE"
                                                    else " (DISAGREES with T2fix REPLACE, or its Stage-B job is not yet evaluated)"))
    return "\n".join(md) + "\n"


def clean_markdown(sel, comps, own_comp, clean_comp, own_vals, clean_vals, jobs, nruns) -> str:
    md = [f"# Tier-2 5x: LOCAL clean summary (REPORT ONLY, NOT FOR UPLOAD) ({now()})", "",
          "LOCAL clean = the kit packer's leaderboard-style metric of each run against the clean Tier-1 reference "
          "(frame references of the run's own input). Never used to select (selection.json is own-input only).", ""]
    md += comp_table(clean_comp, "Compositions (LOCAL clean)")
    md += [""] + comp_table(own_comp, "Compositions (own input, for reference)")
    md += ["", "### Per episode: SELECTED job, own-input and LOCAL clean (5 runs)", "",
           "| ep | object | SELECTED | input | own min | own max | own mean | clean min | clean max | clean mean | n | RAWREF clean mean | K3FULL clean mean | other candidates (clean mean [n]) |",
           "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    kind = {j["job"]: j.get("root_kind") for j in jobs}

    def st(vals):
        xs = [vals[k]["auc"] for k in sorted(vals)]
        return (min(xs), max(xs), sum(xs) / len(xs)) if xs else (None, None, None)

    for e in sel["episodes"]:
        s = e.get("selected")
        o, c = st(own_vals.get(s, {})), st(clean_vals.get(s, {}))
        others = [f"{j['job']} {fv(st(clean_vals.get(j['job'], {}))[2])} [{len(clean_vals.get(j['job'], {}))}]"
                  for j in jobs if j.get("status") == "ok" and int(j["episode"]) == e["episode"] and j["job"] != s]
        md.append(f"| {e['episode']} | {e.get('object', '')} | {s} | {kind.get(s)} | {fv(o[0])} | {fv(o[1])} | {fv(o[2])} | "
                  f"{fv(c[0])} | {fv(c[1])} | {fv(c[2])} | {len(clean_vals.get(s, {}))} | "
                  f"{fv(st(clean_vals.get(e.get('rawref'), {}))[2])} | {fv(st(clean_vals.get(e.get('chosen'), {}))[2])} | {'; '.join(others)} |")
    return "\n".join(md) + "\n"


# ================================================================================================ integrity
def tree_listing(root: Path) -> dict[str, list]:
    out = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".git"]
        for name in filenames:
            p = Path(dirpath) / name
            try:
                st = p.lstat()
            except OSError:
                continue
            out[str(p.relative_to(root))] = [st.st_size, st.st_mtime_ns]
    return out


def integrity() -> dict:
    return {"taken_at": now(), "kit": tree_listing(KIT), "stock": tree_listing(STOCK_REPO)}


def integrity_diff(a: dict, b: dict) -> list[str]:
    out = []
    for key in ("kit", "stock"):
        x, y = a.get(key, {}), b.get(key, {})
        for p in sorted(set(x) | set(y)):
            if x.get(p) != y.get(p):
                out.append(f"{key}: {p} " + ("new" if p not in x else "removed" if p not in y else "changed"))
    return out


# ================================================================================================ preflight
def preflight() -> tuple[list[str], list[str]]:
    fatal, info = [], []
    for need in (PY, KPY, KIT / "examples/run_policy_evaluation.py", KIT / "flash_chord/scripts/evaluate_policy.py",
                 KIT / "tools/pack_submission.py", STOCK / "src/flash_chord/__init__.py", LOCKDIR, CLEAN_SCORER, CACHE_KEYS,
                 INVENTORY, VERDICT):  # fmt: skip
        if not need.exists():
            fatal.append(f"missing {need}")
    if not fatal:
        kit, stock = sha256(KIT / "flash_chord/scripts/evaluate_policy.py"), sha256(STOCK / "scripts/evaluate_policy.py")
        if kit == stock == EVAL_SCRIPT_SHA:
            info.append(f"evaluator: kit evaluate_policy.py == stock == {EVAL_SCRIPT_SHA[:12]}")
        else:
            fatal.append(f"evaluate_policy.py kit {kit[:12]} / stock {stock[:12]} != {EVAL_SCRIPT_SHA[:12]}")
    for name, tree in TREES.items():
        ea = tree["code"] / "scripts/track2/eval_all.py"
        ex = tree["code"] / "src/flash_chord/training/ppo/coordinate/export.py"
        if not ea.is_file() or sha256(ea) != EVAL_ALL_SHA:
            fatal.append(f"{name}: {ea} missing or not eval_all.py {EVAL_ALL_SHA[:12]}")
            continue
        if not ex.is_file():
            fatal.append(f"{name}: exporter {ex} missing")
            continue
        head = git_head(tree["repo"])
        if head != tree["commit"]:
            fatal.append(f"{name}: {tree['repo']} HEAD {head or '?'} != {tree['commit']}")
            continue
        info.append(f"{name}: {tree['label']}; HEAD {head[:7]}; eval_all.py {EVAL_ALL_SHA[:12]}; exporter {sha256(ex)[:12]}")
    head = git_head(STOCK_REPO)
    (info if head == STOCK_COMMIT else fatal).append(f"stock runtime {STOCK_REPO} HEAD {head[:12]} (want {STOCK_COMMIT[:12]})")
    locks = sorted(p.name for p in LOCKDIR.glob("gpu*.lock")) if LOCKDIR.is_dir() else []
    info.append(f"shared eval lock files (never created by us): {locks}")
    return fatal, info


# ================================================================================================ the driver
class Driver:
    def __init__(self, a: argparse.Namespace, log: Logger):
        self.a, self.log = a, log
        self.gpus = sorted({int(x) for x in re.split(r"[,\s]+", a.gpus.strip()) if x})
        bad = [g for g in self.gpus if g not in ALLOWED_GPUS]
        if bad:
            raise SystemExit(f"--gpus {bad} not allowed (PREREG 3.7: only {ALLOWED_GPUS})")
        self.keys, self.keys_from = kit_metric_keys()
        self.running: dict[tuple, dict] = {}
        self.orphans: dict[tuple, dict] = {}
        self.jobs: list[dict] = []
        self.durations: list[float] = []
        self.baseline: dict | None = None
        self.halted: str | None = None
        self.last_stageb = 0.0
        self.last_summary = 0.0
        self.by: dict = {}

    # ------------------------------------------------------------------------------------------ helpers
    def ok_jobs(self) -> list[dict]:
        return [j for j in self.jobs if j.get("status") == "ok"]

    def job(self, job_id: str) -> dict:
        return next(j for j in self.jobs if j["job"] == job_id)

    def busy(self, key: tuple) -> bool:
        return key in self.running or key in self.orphans

    @staticmethod
    def name(j: dict) -> str:
        return f"{j['kind']} {j['job']}" + ("" if j.get("run") is None else f" run {j['run']}") + f" try {j['try']}"

    @staticmethod
    def retry_due(d: Path, delay: float = RETRY_DELAY) -> bool:
        last = read_json(d / "try.json") or {}
        return time.time() - (parse_time(last.get("finished_at")) or 0.0) >= delay

    def summary(self) -> None:
        try:
            summarize(self.jobs, self.a.runs, ROOT, self.log)
        except Exception as error:  # noqa: BLE001
            self.log.first(("summary", repr(error)), f"WARNING summary failed: {error!r}")
        self.last_summary = time.time()

    # ------------------------------------------------------------------------------------------ manifest
    def build_manifest(self) -> None:
        data = read_json(ROOT / "manifest.json")
        if data is not None:
            self.log(f"manifest.json exists (built {data.get('built_at')}): reused")
        else:
            self.log("building the Stage-A manifest (PREREG 3.1) and copying Zuxing's 7 checkpoints (PREREG 3.2)")
            rows = [build_job(s, self.by, copy=True) for s in stage_a_specs()]
            ok = [r for r in rows if r["status"] == "ok"]
            hulls = hull_check(ok, self.log)
            for r in ok:
                h = hulls.get(r["parquet"], {"ok": False, "missing": ["not checked"]})
                r["hulls"] = h
                if not h["ok"]:
                    r["status"] = f"hull_missing: {h['missing']}"
            assign_priorities(rows)
            cols = ["job", "episode", "object", "family", "owner", "arm", "stage", "priority", "provisional", "iteration",
                    "training_sha256", "training_checkpoint", "training_checkpoint_original", "tree", "parquet",
                    "parquet_sha256", "root_kind", "data_root", "manifest", "frame_refs", "status"]  # fmt: skip
            write_csv(ROOT / "manifest.csv", rows, cols)
            write_atomic(ROOT / "manifest.json", dump({"built_at": now(), "prereg_sha256": sha256(PREREG), "rows": rows}))
            for r in rows:
                self.log(f"  {r['job']}: P{r.get('priority')} {r.get('root_kind')} {str(r.get('parquet_sha256'))[:8]} "
                         f"ckpt {str(r.get('training_sha256'))[:8]} {r['status']}")
        self.jobs = load_jobs(ROOT)
        assign_priorities(self.jobs)
        for r in self.jobs:
            if r.get("status") != "ok":
                self.log.first(("refused", r["job"]), f"{r['job']}: NOT evaluated: {r.get('status')}")

    # ------------------------------------------------------------------------------------------ stage B
    def stage_b(self) -> None:
        self.last_stageb = time.time()
        have = {j["job"] for j in self.jobs}
        res = resolutions(ROOT)
        done_items = {r["item"] for r in res if r.get("event") in ("resolved",)}
        verdict = read_json(VERDICT) or {}
        vsha = sha256(VERDICT) if VERDICT.is_file() else None
        pre_ok = PREREG_V3_SHA[:16] in str(verdict.get("prereg_check", ""))
        vby = {int(e["episode"]): e for e in verdict.get("episodes", [])}
        trigger_base = {"verdict_sha256": vsha, "verdict_written": verdict.get("written"), "checked_at": now()}
        for ep in STAGEB_K3:
            item = f"ep{ep:02d}"
            if item in done_items:
                continue
            e = vby.get(ep) or {}
            if not pre_ok or not e.get("settled"):
                continue
            if e.get("choice") == "RAW":
                self.resolve(item, "resolved", note=f"VERDICT settled RAW: nothing added", trigger=trigger_base)
                continue
            if e.get("choice") != "K3FIX":
                continue
            run_dir = FC / str(e.get("k3fix_run"))
            it = e.get("final_iteration")
            fj = read_json(run_dir / "final.json") or {}
            if not isinstance(it, int) or fj.get("iteration") != it:
                self.log.first(("b-final", item, it), f"stage B {item}: VERDICT K3FIX {it} but final.json says {fj.get('iteration')}: waiting")
                continue
            jid = f"ep{ep:02d}_k3fix_final{it}"
            if jid not in have:
                spec = {"job": jid, "episode": ep, "family": "k3fix_final", "owner": "ours", "arm": "K3fix C3r2_tier2_k3fix (Stage B)",
                        "run_dir": str(run_dir), "iteration": it, "tree": "fc_r2", "designation": "final.json",
                        "provisional": False, "stage": "B", "priority": 1}  # fmt: skip
                tsha = sha256(run_dir / f"model_{it}.safetensors") if (run_dir / f"model_{it}.safetensors").is_file() else None
                self.add_stage_b(spec, tsha, "final.json (VERDICT K3FIX settled)", {**trigger_base, "final_json": fj})
                have.add(jid)
            # its t2watch best, once t2watch has a terminal row for the final.json checkpoint
            rel = os.path.relpath(run_dir, FC)
            rows = [r for r in load_csv(T2W / "progress.csv") if f"{r.get('arm')}/{r.get('run')}" == rel and str(r.get("iteration")) == str(it)]
            terminal = any(r.get("status") in ("ok", "export_refused", "ref_mismatch") for r in rows) or \
                sum(r.get("status") in ("failed", "timeout", "interrupted") for r in rows) >= 2
            if not terminal:
                self.log.first(("b-t2w", item), f"stage B {item}: final job added; waiting for t2watch's terminal row of model_{it}")
                continue
            bj = read_json(T2W / "best" / rel / "best.json") or {}
            bit = bj.get("iteration")
            if bit == it or not isinstance(bit, int):
                self.resolve(item, "resolved", note=f"t2watch best = model_{bit} (= final or none): nothing more added",
                             trigger={**trigger_base, "best_json": bj})
                continue
            bid = f"ep{ep:02d}_k3fix_t2wbest{bit}"
            if bid not in have:
                spec = {"job": bid, "episode": ep, "family": "t2watch_best", "owner": "ours", "arm": "K3fix t2watch best (Stage B)",
                        "run_dir": str(run_dir), "iteration": bit, "tree": "fc_r2", "designation": "t2watch best.json",
                        "best_json": str(T2W / "best" / rel / "best.json"), "provisional": False, "stage": "B", "priority": 3}
                self.add_stage_b(spec, bj.get("source_sha256"), "t2watch best.json source_sha256", {**trigger_base, "best_json": bj})
            self.resolve(item, "resolved", note=f"final {jid} + t2watch best {bid}", trigger=trigger_base)
        for arm, (ep, rel) in STAGEB_T2FIX.items():
            if arm in done_items:
                continue
            run_dir = T2FIX / rel
            fj = read_json(run_dir / "final.json")
            rf = read_json(T2FIX / "eval_ARM" / f"ep{ep:02d}" / "runin_final.json")
            if not fj or not rf:
                continue
            it = rf.get("best_iteration")
            if not isinstance(it, int):
                self.log.first(("b-t2fix", arm), f"stage B {arm}: runin_final.json has no best_iteration")
                continue
            path = run_dir / f"model_{it}.safetensors"
            tsha = sha256(path) if path.is_file() else None
            same = next((j for j in self.jobs if j.get("training_sha256") == tsha and tsha and j["family"] == "t2fix"), None)
            trig = {**trigger_base, "final_json": fj, "runin_final": {k: rf.get(k) for k in ("best_iteration", "best_checkpoint",
                    "mean_auc", "n_evals", "finished_at", "stop_reason")}}  # fmt: skip
            if same is not None:
                self.resolve(arm, "settled", job=same["job"], note=f"designated model_{it} = Stage-A job {same['job']}: reused (no "
                             "second export/evaluation of the same file), now settled", trigger=trig)
                self.resolve(arm, "resolved", note=f"designated = {same['job']}", trigger=trig)
                continue
            jid = f"ep{ep:02d}_t2fix{arm[-1]}_designated{it}"
            if jid not in have:
                spec = {"job": jid, "episode": ep, "family": "t2fix", "owner": "ours", "arm": f"T2fix {arm} designated (Stage B)",
                        "run_dir": str(run_dir), "iteration": it, "tree": "fc_r2", "designation": "T2fix runin_final.json best",
                        "t2fix_episode": ep, "provisional": False, "stage": "B", "priority": 3}  # fmt: skip
                srcs = []
                for r in t2fix_rows(ep, it):
                    rep = read_json(Path(r["log"]).parent / "export" / f"ep{ep:02d}" / "export_report.json") or {}
                    srcs.append(rep.get("source_sha256"))
                exp = srcs[0] if srcs and all(s == srcs[0] for s in srcs) else None
                self.add_stage_b(spec, exp, "eval_ARM export source_sha256", trig)
            self.resolve(arm, "resolved", note=f"designated {jid}", trigger=trig)

    def add_stage_b(self, spec: dict, expect_sha: str | None, expect_from: str, trigger: dict) -> None:
        row = build_job(spec, self.by, copy=False, expect_sha=expect_sha, expect_from=expect_from)
        if row["status"] == "ok":
            h = hull_check([row], self.log).get(row["parquet"], {"ok": False, "missing": ["not checked"]})
            row["hulls"] = h
            if not h["ok"]:
                row["status"] = f"hull_missing: {h['missing']}"
        row["trigger"] = trigger
        row["added_at"] = now()
        append_line(ROOT / "manifest_stageB.jsonl", json.dumps(row, default=str))
        self.jobs.append(row)
        self.log(f"stage B: added {row['job']} (P{row.get('priority')}, {row.get('root_kind')} {str(row.get('parquet_sha256'))[:8]}, "
                 f"ckpt {str(row.get('training_sha256'))[:8]}): {row['status']}")

    def resolve(self, item: str, event: str, note: str = "", job: str | None = None, trigger: dict | None = None) -> None:
        append_line(ROOT / "stageB_resolutions.jsonl", json.dumps({"item": item, "event": event, "job": job, "note": note,
                                                                     "at": now(), "trigger": trigger}, default=str))
        self.log(f"stage B {item}: {event}: {note}")

    def stage_b_done(self) -> bool:
        done = {r["item"] for r in resolutions(ROOT) if r.get("event") == "resolved"}
        return all(f"ep{e:02d}" in done for e in STAGEB_K3) and all(a in done for a in STAGEB_T2FIX)

    # ------------------------------------------------------------------------------------------ GPUs
    def gpu_rows(self, state: dict, mps: dict, ours: list[dict]) -> list[dict]:
        busy = {j["gpu"] for j in ours if j.get("gpu") is not None}
        lanes = {0: 0, 1: 0}
        for j in ours:
            if j.get("gpu") is not None:
                lanes[numa(j["gpu"])] += 1
        out = []
        for g in sorted(state["gpus"]):
            st, lockf = state["gpus"][g], LOCKDIR / f"gpu{g}.lock"
            row = {"gpu": g, "free": st["free"], "napps": sum(1 for x in state["apps"] if x["gpu"] == g),
                   "lock_file": lockf.is_file()}  # fmt: skip
            if g not in self.gpus:
                why = "not in --gpus" + (" (GPU 4: never)" if g == 4 else "")
            elif g in mps:
                why = f"MPS ({mps[g]})"
            elif g in busy:
                why = "one of our evals runs there"
            elif not row["lock_file"]:
                why = f"no lock file {lockf.name}"
            elif st["free"] < MIN_FREE_MB + EVAL_PEAK_MB:
                why = f"free {st['free']} MiB < {MIN_FREE_MB}+{EVAL_PEAK_MB}"
            elif lanes[numa(g)] >= self.a.lane_max:
                why = f"NUMA{numa(g)} lane has {self.a.lane_max} of ours"
            else:
                why = ""
            row.update(usable=not why, why=why or "usable")
            out.append(row)
        return out

    def pick_gpu(self, state: dict, mps: dict, ours: list[dict], take: bool):
        rows = self.gpu_rows(state, mps, ours)
        usable = sorted((r for r in rows if r["usable"]), key=lambda r: (r["napps"], -r["free"], r["gpu"]))
        if not usable:
            return "no usable GPU: " + "; ".join(f"gpu{r['gpu']} {r['why']}" for r in rows if r["gpu"] in self.gpus)
        if not take:
            return usable[0]["gpu"], None
        held = []
        for r in usable:
            try:
                fd = os.open(LOCKDIR / f"gpu{r['gpu']}.lock", os.O_RDONLY)  # read-only, never O_CREAT
            except OSError:
                continue
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                held.append(r["gpu"])
                continue
            return r["gpu"], fd
        return f"every usable GPU's eval lock is held by another watcher ({held})"

    # ------------------------------------------------------------------------------------------ launching
    def launch(self, kind: str, job: dict, k: int | None, t: int, d: Path, cmd: list[str], cwd: Path, env: dict,
               gpu: int | None, fd: int | None, lane: int, log_file: Path, extra: dict | None = None) -> None:
        cpus = EVAL_CORES[lane]
        prefix = ["nice", "-n", "10", "taskset", "-c", cpus]
        shown = show_command(prefix, cmd, cwd, env)
        with log_file.open("a") as stream:
            stream.write(f"[{now()}] {kind} {job['job']}{'' if k is None else f' run {k}'} try {t}"
                         f"{'' if gpu is None else f', gpu {gpu}'}, cpus {cpus}: {shown}\n")  # fmt: skip
            stream.flush()
            proc = subprocess.Popen(prefix + cmd, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True,
                                    pass_fds=(fd,) if fd is not None else ())  # fmt: skip
        write_atomic(d / "start.json", dump({"kind": kind, "job": job["job"], "run": k, "try": t, "gpu": gpu, "cpus": cpus,
                                             "lane": lane, "pid": proc.pid, "started_at": now(), "started_epoch": time.time(),
                                             "cwd": str(cwd), "command": prefix + cmd, "shown": shown, **(extra or {})}))
        key = (kind, job["job"]) if kind == "export" else (kind, job["job"], k)
        self.running[key] = {"kind": kind, "job": job["job"], "run": k, "try": t, "dir": d, "proc": proc, "start": time.time(),
                             "gpu": gpu, "fd": fd, "lane": lane, "overdue": False, **(extra or {})}  # fmt: skip
        self.log(f"start {self.name(self.running[key])}" + ("" if gpu is None else f" on gpu {gpu}") + f" (pid {proc.pid})")

    def lane_load(self, kind: str) -> list[int]:
        loads = [0, 0]
        for key, j in [*self.running.items(), *self.orphans.items()]:
            if key[0] == kind:
                loads[j.get("lane", 0)] += 1
        return loads

    def sorted_jobs(self) -> list[dict]:
        order = {j["job"]: i for i, j in enumerate(self.jobs)}
        return sorted(self.ok_jobs(), key=lambda j: (j.get("priority", 4), order[j["job"]]))

    def launch_exports(self) -> None:
        for job in self.sorted_jobs():
            loads = self.lane_load("export")
            if min(loads) >= 1:
                return
            jid = job["job"]
            if export_rec(ROOT, jid) is not None or self.busy(("export", jid)):
                continue
            base = ROOT / "exports" / jid
            done = tries(base)
            tol = EXPORT_TOL
            if done:
                last = read_json(done[-1][1] / "try.json") or {}
                if last.get("status") == "refused_tol" and fnum(last.get("tolerance")) == EXPORT_TOL:
                    tol = FALLBACK_TOL
                else:
                    tol = fnum(last.get("tolerance")) or EXPORT_TOL
                    if not self.retry_due(done[-1][1]):
                        continue
            t = done[-1][0] + 1 if done else 1
            d = inside_root(base / f"try{t}")
            d.mkdir(parents=True, exist_ok=False)
            cmd, code, extra = export_command(job, d, tol)
            env = child_env(None, str(d), None)
            env.update(extra)
            lane = 0 if loads[0] <= loads[1] else 1
            self.launch("export", job, None, t, d, cmd, code, env, None, None, lane, d / "export.log", {"tolerance": tol})

    def launch_evals(self) -> None:
        ours = [j for key, j in [*self.running.items(), *self.orphans.items()] if key[0] == "eval"]
        if len(ours) >= self.a.max_parallel:
            return
        ready = []
        jobs = self.sorted_jobs()
        for prio in (1, 2, 3, 4):
            for k in range(1, self.a.runs + 1):
                for job in jobs:
                    if job.get("priority", 4) != prio:
                        continue
                    jid = job["job"]
                    exp = export_rec(ROOT, jid)
                    if not exp or exp.get("status") != "ok" or run_rec(ROOT, jid, k) is not None or self.busy(("eval", jid, k)):
                        continue
                    done = tries(ROOT / "runs" / jid / f"run{k}")
                    if done and not self.retry_due(done[-1][1]):
                        continue
                    ready.append((job, k, exp, done))
        if not ready:
            return
        head = pids_headroom()
        if head is not None and head < MIN_FREE_PIDS:
            self.log.first(("pids", head // 500), f"pids headroom {head} < {MIN_FREE_PIDS}: no new eval")
            return
        try:
            state = smi()
            mps = mps_gpus(state)
        except Exception as error:  # noqa: BLE001
            self.log.first(("smi", repr(error)), f"nvidia-smi failed ({error!r}): no new eval")
            return
        for job, k, exp, done in ready:
            if len(ours) >= self.a.max_parallel:
                break
            choice = self.pick_gpu(state, mps, ours, take=True)
            if isinstance(choice, str):
                self.log.first(("wait", re.sub(r"\d+", "N", choice)), f"{len(ready)} eval(s) ready, waiting: {choice}")
                break
            gpu, fd = choice
            jid, t = job["job"], (done[-1][0] + 1 if done else 1)
            w = inside_root(ROOT / "runs" / jid / f"run{k}" / f"try{t}")
            try:
                (w / f"ep{int(job['episode']):02d}").mkdir(parents=True, exist_ok=False)
                (w / f"ep{int(job['episode']):02d}" / f"model_{job['iteration']}.safetensors").symlink_to(exp["file"])
                cmd, code = eval_command(job, w, gpu)
                env = child_env(gpu, str(w), Path(job["data_root"]))
                self.launch("eval", job, k, t, w, cmd, code, env, gpu, fd, numa(gpu), w / "eval_all.log")
            except Exception:
                os.close(fd)
                raise
            ours.append(self.running[("eval", jid, k)])

    def launch_cleans(self) -> None:
        for job in self.sorted_jobs():
            jid = job["job"]
            for k in range(1, self.a.runs + 1):
                loads = self.lane_load("clean")
                if min(loads) >= 1:
                    return
                rr = run_rec(ROOT, jid, k)
                if not rr or rr.get("status") != "ok" or clean_rec(ROOT, jid, k) is not None or self.busy(("clean", jid, k)):
                    continue
                rd = ROOT / "runs" / jid / f"run{k}"
                cbase = rd / "clean_LOCAL_NOT_FOR_UPLOAD"
                done = tries(cbase)
                if done and not self.retry_due(done[-1][1], 60.0):
                    continue
                t = done[-1][0] + 1 if done else 1
                ep = int(job["episode"])
                fr = job.get("frame_refs")
                if fr and not (Path(fr) / f"episode_{ep:06d}.npz").is_file():
                    write_atomic(rd / "clean_LOCAL_NOT_FOR_UPLOAD.json", dump({"status": "no_frame_refs", "frame_refs": fr,
                                                                               "finished_at": now()}))
                    continue
                d = inside_root(cbase / f"try{t}")
                d.mkdir(parents=True, exist_ok=False)
                cfg = json.loads((KIT / "config/tracks.json").read_text())
                cfg["tracks"]["track_2_tier2"]["scored_episodes"] = [ep]
                write_atomic(rd / "pack" / "tracks.json", json.dumps(cfg, indent=2))
                for old in ("clean_scores_LOCAL_NOT_FOR_UPLOAD.csv", "packed_LOCAL_NOT_FOR_UPLOAD.csv"):
                    (inside_root(rd / old)).unlink(missing_ok=True)  # our own outputs of an earlier try
                results = Path(rr["results_json"]).parent
                cmd = clean_command(job, rd, results)
                env = {k2: v for k2, v in os.environ.items() if k2 not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")
                       and not k2.startswith(("CUDA_MPS_", "WANDB_"))}
                env.update(PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", TZ="Asia/Shanghai",
                           TMPDIR=str(ROOT / "tmp"), T2BEST5_JOB=str(d))
                lane = rr.get("lane", numa(rr["gpu"]) if isinstance(rr.get("gpu"), int) else (0 if loads[0] <= loads[1] else 1))
                if loads[lane] >= 1:
                    lane = 1 - lane
                self.launch("clean", job, k, t, d, cmd, KIT, env, None, None, lane, d / "clean_score_LOCAL_NOT_FOR_UPLOAD.log")

    # ------------------------------------------------------------------------------------------ finishing
    def flag_overdue(self) -> None:
        for j in self.running.values():
            lim = OVERDUE_S if j["kind"] == "eval" else CPU_OVERDUE_S
            if not j["overdue"] and time.time() - j["start"] > lim:
                j["overdue"] = True
                self.log(f"OVERDUE {self.name(j)} (pid {j['proc'].pid}, {time.time() - j['start']:.0f} s > {lim:.0f} s): "
                         "NOT signalled (PREREG 3.7); its GPU gets no new eval of ours until it ends")

    def reap(self) -> None:
        for key, j in list(self.running.items()):
            code = j["proc"].poll()
            if code is None:
                continue
            del self.running[key]
            if j["fd"] is not None:
                os.close(j["fd"])
            self.finish(j, code, "")
            self.after_job()

    def finish(self, j: dict, code, remark: str) -> None:
        job = self.job(j["job"])
        if j["kind"] == "export":
            self.finish_export(job, j["try"], j["dir"], code, j.get("tolerance", EXPORT_TOL), remark)
        elif j["kind"] == "eval":
            self.finish_eval(job, j["run"], j["try"], j["dir"], code, remark)
        else:
            self.finish_clean(job, j["run"], j["try"], j["dir"], code, remark)

    def after_job(self) -> None:
        if self.baseline is None or self.halted:
            return
        diff = integrity_diff(self.baseline, integrity())
        if diff:
            self.halted = f"kit / stock runtime changed: {diff[:20]}"
            write_atomic(ROOT / "integrity" / "CHANGED.json", dump({"detected_at": now(), "differences": diff}))
            self.log(f"HALT: {self.halted}: no new job is started; running jobs finish (PREREG 3.3)")

    def finish_export(self, job: dict, t: int, d: Path, code, tol: float, remark: str) -> None:
        jid = job["job"]
        status, info = check_export(job, d, code, tol)
        start = read_json(d / "start.json") or {}
        rec = {"job": jid, "episode": job["episode"], "try": t, "status": status, "remark": remark, "tolerance": tol,
               "fallback": tol != EXPORT_TOL, "started_at": start.get("started_at"), "finished_at": now(),
               "seconds": round(time.time() - start["started_epoch"], 1) if start.get("started_epoch") else None,
               "training_checkpoint": job["training_checkpoint"], "training_sha256": job["training_sha256"],
               "tree": job["tree"], "code_commit": TREES[job["tree"]]["commit"], "exporter": EXPORTER,
               "command": start.get("shown"), **info}  # fmt: skip
        rec["tolerance"] = tol
        write_atomic(d / "try.json", dump(rec))
        base = d.parent
        n_tech = sum(1 for _, td in tries(base) if (read_json(td / "try.json") or {}).get("status") == "failed")
        final = None
        if status == "ok":
            final = {**rec, "kept": "kept for good; every run evaluates this file"}
            self.log(f"{jid}: export ok (try {t}, tolerance {tol:g}{' FALLBACK' if tol != EXPORT_TOL else ''}): sha256 "
                     f"{info['sha256'][:16]} max_rel_diff {fnum(info.get('max_rel_diff')) or 0.0:.2e}")
        elif status == "refused_tol" and tol == EXPORT_TOL:
            self.log(f"{jid}: export refused at {tol:g} ({info['problems']}): one fallback export at {FALLBACK_TOL:g}")
        elif status in ("refused_tol", "refused_other"):
            final = {**rec, "status": "export_incompatible" if job["owner"] == "zuxing" and status == "refused_other" else "export_refused"}
        elif n_tech >= EXPORT_TECH_TRIES:
            final = {**rec, "status": "export_incompatible" if job["owner"] == "zuxing" else "export_failed"}
        else:
            self.log(f"{jid}: export try {t} failed ({info['problems']}); retry in {RETRY_DELAY:.0f} s")
        if final is not None:
            write_atomic(base / "export.json", dump(final))
            if final["status"] != "ok":
                self.log(f"{jid}: export {final['status'].upper()} (try {t}): {info['problems']}: this job gets NO evaluation")

    def finish_eval(self, job: dict, k: int, t: int, d: Path, code, remark: str) -> None:
        jid = job["job"]
        exp = export_rec(ROOT, jid) or {}
        problems, info = check_eval(job, exp, d / "results", d / "eval_report.json", code, self.keys)
        start = read_json(d / "start.json") or {}
        seconds = round(time.time() - start["started_epoch"], 1) if start.get("started_epoch") else None
        status = "ok" if not problems else "failed"
        rec = {"job": jid, "episode": job["episode"], "object": job.get("object"), "run": k, "try": t, "status": status,
               "problems": problems, "remark": remark, "gpu": start.get("gpu"), "cpus": start.get("cpus"), "lane": start.get("lane"),
               "started_at": start.get("started_at"), "finished_at": now(), "seconds": seconds,
               "export_sha256": exp.get("sha256"), "exported_file": exp.get("file"), "export_tolerance": exp.get("tolerance"),
               "work_dir": str(d), "metric_keys": self.keys, "command": start.get("shown"),
               "training_parquet": job["parquet"], "training_parquet_sha256": job["parquet_sha256"], **info}  # fmt: skip
        write_atomic(d / "try.json", dump(rec))
        base = d.parent
        if status == "ok":
            write_atomic(base / "run.json", dump(rec))
            if seconds and not remark:
                self.durations.append(seconds)
            m = info["metrics"]
            self.log(f"{jid} run {k} (try {t}, gpu {rec['gpu']}, {seconds or 0:.0f} s): own-input AUC {m['auc']:.4f} SP-SR "
                     f"{m['sp_sr']:.3f} MP-SR {m['mp_sr']:.3f} MPPE {m['mppe']:.2f} cm" + (f" ({remark})" if remark else ""))
            self.summary()
        elif len(tries(base)) >= EVAL_TRIES:
            write_atomic(base / "run.json", dump(rec))
            self.log(f"{jid} run {k}: FAILED after {t} tries ({problems})")
            self.summary()
        else:
            self.log(f"{jid} run {k} try {t} failed ({problems}); retry in {RETRY_DELAY:.0f} s; log {d / 'eval_all.log'}")

    def finish_clean(self, job: dict, k: int, t: int, d: Path, code, remark: str) -> None:
        rd = d.parent.parent
        ep = int(job["episode"])
        vals = parse_clean(rd / "clean_scores_LOCAL_NOT_FOR_UPLOAD.csv", ep) if code in (0, None) else None
        text = ""
        try:
            text = (d / "clean_score_LOCAL_NOT_FOR_UPLOAD.log").read_text(errors="replace")
        except OSError:
            pass
        mres = re.search(r"input reference check: .*?maximum residual ([0-9.eE+-]+) m, ([0-9.eE+-]+) rad", text)
        rec = {"job": job["job"], "run": k, "try": t, "status": "ok" if vals else "failed", "exit": code, "remark": remark,
               "values": vals, "frame_refs": job.get("frame_refs") or "packer default (released raw Tier-2 input)",
               "input_check_residual_m": mres.group(1) if mres else None,
               "input_check_residual_rad": mres.group(2) if mres else None, "finished_at": now()}  # fmt: skip
        write_atomic(d / "try.json", dump(rec))
        if vals or t >= CLEAN_TRIES:
            write_atomic(rd / "clean_LOCAL_NOT_FOR_UPLOAD.json", dump(rec))
        self.log(f"{job['job']} run {k}: LOCAL clean diagnostic {rec['status']} (try {t}; numbers only in LOCAL files)")

    # ------------------------------------------------------------------------------------------ recovery
    def recover(self) -> None:
        live = live_jobs()
        for job in self.ok_jobs():
            jid = job["job"]
            slots = [("export", None, ROOT / "exports" / jid)]
            for k in range(1, self.a.runs + 1):
                slots.append(("eval", k, ROOT / "runs" / jid / f"run{k}"))
                slots.append(("clean", k, ROOT / "runs" / jid / f"run{k}" / "clean_LOCAL_NOT_FOR_UPLOAD"))
            for kind, k, base in slots:
                for t, d in tries(base):
                    if (d / "try.json").is_file():
                        continue
                    key = (kind, jid) if kind == "export" else (kind, jid, k)
                    start = read_json(d / "start.json") or {}
                    if str(d) in live:
                        gpu = start.get("gpu")
                        self.orphans[key] = {"kind": kind, "job": jid, "run": k, "try": t, "dir": d, "gpu": gpu,
                                             "procs": live[str(d)], "lane": start.get("lane", 0),
                                             "tolerance": start.get("tolerance", EXPORT_TOL)}  # fmt: skip
                        self.log(f"{self.name(self.orphans[key])}: still running from a previous driver (pids "
                                 f"{[p for p, _ in live[str(d)]]}); recorded when it ends, never signalled")
                    else:
                        self.finish({"kind": kind, "job": jid, "run": k, "try": t, "dir": d,
                                     "tolerance": start.get("tolerance", EXPORT_TOL)}, None,
                                    "recovered after a driver restart (exit code unknown)")

    def check_orphans(self) -> None:
        for key, o in list(self.orphans.items()):
            if any(proc_alive(p, s) for p, s in o["procs"]):
                continue
            del self.orphans[key]
            self.finish(o, None, "finished while no driver was running (exit code unknown)")
            self.after_job()

    # ------------------------------------------------------------------------------------------ state
    def all_final(self) -> bool:
        for job in self.ok_jobs():
            jid = job["job"]
            exp = export_rec(ROOT, jid)
            if exp is None:
                return False
            if exp.get("status") != "ok":
                continue
            for k in range(1, self.a.runs + 1):
                r = run_rec(ROOT, jid, k)
                if r is None:
                    return False
                if r.get("status") == "ok" and clean_rec(ROOT, jid, k) is None:
                    return False
        return True

    def progress(self) -> dict:
        ex = {"ok": 0, "refused": 0, "failed": 0, "pending": 0}
        ev = {"ok": 0, "failed": 0, "pending": 0, "no_export": 0}
        cl = {"ok": 0, "other": 0, "pending": 0}
        for job in self.ok_jobs():
            jid = job["job"]
            exp = export_rec(ROOT, jid)
            st = (exp or {}).get("status")
            ex["ok" if st == "ok" else "pending" if exp is None else "refused" if "refused" in str(st) or "incompatible" in str(st) else "failed"] += 1
            for k in range(1, self.a.runs + 1):
                r = run_rec(ROOT, jid, k)
                ev[("ok" if r.get("status") == "ok" else "failed") if r else ("no_export" if exp and st != "ok" else "pending")] += 1
                if r and r.get("status") == "ok":
                    c = clean_rec(ROOT, jid, k)
                    cl["pending" if c is None else "ok" if c.get("status") == "ok" else "other"] += 1
        return {"jobs": len(self.ok_jobs()), "jobs_refused": len(self.jobs) - len(self.ok_jobs()), "runs_per_job": self.a.runs,
                "exports": ex, "evals": ev, "clean": cl}

    def heartbeat(self, phase: str) -> None:
        prog = self.progress()
        lanes = max(1, min(self.a.max_parallel, sum(1 for key in self.running if key[0] == "eval") or 1))
        mean_s = sum(self.durations) / len(self.durations) if self.durations else None
        jobs = [{"kind": j["kind"], "job": j["job"], "run": j.get("run"), "try": j["try"], "gpu": j.get("gpu"),
                 "pid": j["proc"].pid, "elapsed_s": round(time.time() - j["start"]), "overdue": j["overdue"]}
                for j in self.running.values()]  # fmt: skip
        jobs += [{"kind": o["kind"], "job": o["job"], "run": o.get("run"), "try": o["try"], "gpu": o.get("gpu"),
                  "orphan_pids": [p for p, _ in o["procs"]]} for o in self.orphans.values()]
        write_atomic(ROOT / "heartbeat.json", dump({
            "updated_at": now(), "updated_epoch": time.time(), "pid": os.getpid(), "phase": phase, **prog, "running": jobs,
            "mean_eval_seconds": mean_s, "halted": self.halted,
            "eta_hours_evals": round(mean_s * prog["evals"]["pending"] / lanes / 3600, 2) if mean_s else None,
            "stage_b_resolved": sorted({r["item"] for r in resolutions(ROOT) if r.get("event") == "resolved"}),
            "stop_file": (ROOT / "STOP").exists()}))

    # ------------------------------------------------------------------------------------------ main loop
    def run(self) -> int:
        a = self.a
        lockfd = os.open(inside_root(ROOT / ".driver.lock"), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(f"another t2best5_driver.py holds {ROOT / '.driver.lock'}", file=sys.stderr)
            return 1
        if (ROOT / "STOP").exists():
            print(f"{ROOT / 'STOP'} exists: remove it to start", file=sys.stderr)
            return 1
        (ROOT / "tmp").mkdir(exist_ok=True)
        write_atomic(ROOT / "driver.pid", f"{os.getpid()}\n")
        fatal, info = check_pins(require_pins_file=True)
        f2, i2 = preflight()
        fatal += f2
        for line in info + i2:
            self.log(f"preflight: {line}")
        if fatal:
            for line in fatal:
                self.log(f"preflight FAILED: {line}")
            return 1
        _, self.by = inventory_index()
        self.build_manifest()
        # tree listing taken before the first evaluation of this folder; restarts compare against the same baseline
        bfile = ROOT / "integrity" / "baseline.json"
        self.baseline = read_json(bfile)
        if self.baseline is None:
            self.baseline = integrity()
            write_atomic(bfile, dump(self.baseline))
            self.log(f"integrity baseline: {len(self.baseline['kit'])} kit files, {len(self.baseline['stock'])} stock files")
        if (ROOT / "integrity" / "CHANGED.json").is_file():
            self.halted = "integrity/CHANGED.json exists (an earlier change of the kit / stock runtime): no new job"
            self.log(f"HALT: {self.halted}")
        else:
            self.after_job()
        self.log(f"start pid {os.getpid()}: {len(self.ok_jobs())} jobs x {a.runs} runs; gpus {self.gpus}, max parallel "
                 f"{a.max_parallel}, lane max {a.lane_max}; metrics {self.keys} ({self.keys_from}); never signals")
        self.recover()
        self.stage_b()
        self.summary()
        last_beat = 0.0
        while True:
            stopping = (ROOT / "STOP").exists()
            if stopping:
                self.log.first("stop", f"STOP file: no new job; waiting for {len(self.running) + len(self.orphans)} running")
            self.reap()
            self.check_orphans()
            self.flag_overdue()
            if not stopping and time.time() - self.last_stageb >= STAGEB_POLL:
                self.stage_b()
            if not stopping and not self.halted:
                self.launch_exports()
                self.launch_evals()
                self.launch_cleans()
            if time.time() - last_beat >= 30:
                self.heartbeat("halted" if self.halted else "stopping" if stopping else "running")
                last_beat = time.time()
            if time.time() - self.last_summary >= SUMMARY_EVERY:
                self.summary()
            if not self.running and not self.orphans:
                if stopping or self.halted:
                    break
                if self.all_final() and self.stage_b_done():
                    self.log("every export, evaluation, clean diagnostic and Stage-B item is final")
                    break
            time.sleep(a.poll)
        self.summary()
        diff = integrity_diff(self.baseline, integrity())
        write_atomic(ROOT / "integrity" / f"check_{stamp()}.json", dump({"from": self.baseline["taken_at"], "to": now(),
                                                                         "differences": diff}))
        self.log("kit and stock runtime unchanged (file list, sizes, mtimes)" if not diff else f"WARNING changed: {diff[:20]}")
        self.heartbeat("halted" if self.halted else "stopped (STOP file)" if (ROOT / "STOP").exists() else "done")
        try:
            if int((ROOT / "driver.pid").read_text()) == os.getpid():
                (ROOT / "driver.pid").unlink()
        except (OSError, ValueError):
            pass
        os.close(lockfd)
        self.log(f"driver end: {self.progress()}")
        return 0


# ================================================================================================ dry run
def prereg_pairs() -> set[tuple[int, int, str]]:
    text = PREREG.read_text()
    sec = text.split("### Stage A", 1)[1].split("**Stage A has 70 jobs:**", 1)[0]
    pairs = set()
    for line in sec.splitlines():
        m = re.match(r"^\| (\d\d) \|", line)
        if not m:
            continue
        body = re.sub(r"\[[^\]]*\]", "", line)
        for it, sha8 in re.findall(r"\b(\d{4,6}) ([0-9a-f]{8})\b", body):
            pairs.add((int(m.group(1)), int(it), sha8))
    return pairs


def dry_run(a: argparse.Namespace) -> int:
    lines: list[str] = []

    def p(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    ts = stamp()
    p(f"t2best5 DRY RUN {now()}. Nothing is copied, exported, evaluated, locked or signalled. Written: {ROOT}/dryrun/*_{ts}*, "
      f"{ROOT}/preflight/hulls_* (the read-only hull check).")
    fatal, info = check_pins(require_pins_file=False)
    f2, i2 = preflight()
    for x in info + i2:
        p(f"  preflight ok: {x}")
    for x in fatal + f2:
        p(f"  preflight FAILED: {x}")
    keys, keys_from = kit_metric_keys()
    p(f"  metrics: {keys} ({keys_from})")
    _, by = inventory_index()
    rows = [build_job(s, by, copy=False) for s in stage_a_specs()]
    ok = [r for r in rows if r["status"] == "ok"]
    hulls = hull_check(ok, Logger(None))
    for r in ok:
        h = hulls.get(r["parquet"], {"ok": False, "missing": ["not checked"]})
        r["hulls"] = h
        if not h["ok"]:
            r["status"] = f"hull_missing: {h['missing']}"
    assign_priorities(rows)
    p("")
    p(f"STAGE A JOB LIST ({len(rows)} jobs)")
    p(f"  {'job':<26} {'P':>2} {'family':<18} {'owner':<6} {'input':<6} {'parquet':<8} {'iter':>6} {'ckpt':<8} {'tree':<8} "
      f"{'prov':<4} hulls  status")
    for r in rows:
        p(f"  {r['job']:<26} {r.get('priority'):>2} {r['family']:<18} {r['owner']:<6} {str(r.get('root_kind')):<6} "
          f"{str(r.get('parquet_sha256'))[:8]:<8} {r['iteration']:>6} {str(r.get('training_sha256'))[:8]:<8} {r['tree']:<8} "
          f"{'P' if r.get('provisional') else '':<4} {','.join(r.get('hulls', {}).get('keys', []))[:20]:<6} {r['status']}"
          + (f"  [{'; '.join(r.get('notes', []))}]" if r.get("notes") else ""))
    fams = {}
    for r in rows:
        fams[r["family"]] = fams.get(r["family"], 0) + 1
    p(f"  counts: {fams}")
    want = prereg_pairs()
    got = {(int(r["episode"]), int(r["iteration"]), str(r.get("training_sha256"))[:8]) for r in rows}
    p(f"  PREREG section 1 table: {len(want)} (episode, iteration, sha8) pairs; job list: {len(got)}; "
      f"EQUAL: {want == got}" + ("" if want == got else f"; only in PREREG {sorted(want - got)}; only in jobs {sorted(got - want)}"))
    counts_ok = (fams.get("raw") == 30 and fams.get("k3fix_final", 0) + fams.get("k3easy_final", 0) == 19
                 and fams.get("t2watch_best") == 11 and fams.get("t2fix") == 1 and fams.get("zuxing") == 7
                 and fams.get("raws7_final", 0) + fams.get("raws7_t2watch_best", 0) == 2)  # fmt: skip
    p(f"  counts match PREREG (30 RAWREF, 19 K3fix/K3 easy-clip, 11 t2watch, 1 T2fix, 7 Zuxing, 2 raw s7): {counts_ok}")
    for r in rows:
        if r["family"] == "zuxing":
            p(f"  Zuxing {r['job']}: copy {r['training_checkpoint_original']} (+ rsi {r.get('rsi_original') or 'none'}) -> "
              f"{r['training_checkpoint']}; his metrics.csv rows {r.get('zuxing_rows')}")
    # GPUs
    p("")
    drv = Driver(a, Logger(None))
    try:
        state = smi()
        mps = mps_gpus(state)
        held = held_lock_inodes()
        p(f"GPUS now (need free >= {MIN_FREE_MB}+{EVAL_PEAK_MB} MiB; 1 of ours per GPU, {a.lane_max} per lane, {a.max_parallel} total)")
        for g in drv.gpu_rows(state, mps, []):
            lockf = LOCKDIR / f"gpu{g['gpu']}.lock"
            now_held = g["lock_file"] and os.stat(lockf).st_ino in held
            p(f"  gpu{g['gpu']}: free {g['free']:>6} MiB, {g['napps']} processes, lock file {'yes' if g['lock_file'] else 'NO'}"
              f"{' (held now)' if now_held else ''} -> {g['why']}")
    except Exception as error:  # noqa: BLE001
        p(f"nvidia-smi failed: {error!r}")
    # commands, one per kind of job
    p("")
    p("COMMANDS (examples; every job runs the same way with its own paths)")
    shown = set()
    for r in sorted(ok, key=lambda x: x.get("priority", 4)):
        key = (r["family"], r.get("root_kind"))
        if key in shown or not r["status"] == "ok":
            continue
        shown.add(key)
        d = ROOT / "exports" / r["job"] / "try1"
        cmd, code, extra = export_command(r, d, EXPORT_TOL)
        env = child_env(None, str(d), None)
        env.update(extra)
        p(f"  [{r['job']}] export: {show_command(['nice', '-n', '10', 'taskset', '-c', EVAL_CORES[0]], cmd, code, env)}")
        w = ROOT / "runs" / r["job"] / "run1" / "try1"
        cmd, code = eval_command(r, w, 5)
        p(f"  [{r['job']}] eval (gpu chosen at launch, e.g. 5): "
          f"{show_command(['nice', '-n', '10', 'taskset', '-c', EVAL_CORES[1]], cmd, code, child_env(5, str(w), Path(r['data_root'])))}")
        rd = ROOT / "runs" / r["job"] / "run1"
        p(f"  [{r['job']}] LOCAL clean: cd {KIT} && nice -n 10 taskset -c <lane> {shlex.join(clean_command(r, rd, w / 'results'))}")
    n = len(ok)
    p("")
    p(f"ESTIMATE: {n} jobs x {a.runs} = {n * a.runs} evals of ~100-130 s at <= {a.max_parallel} in parallel -> about "
      f"{n * a.runs * 120 / a.max_parallel / 3600:.1f} h if locks and GPUs allow; exports ~30-60 s each on 2 CPU lanes.")
    p("STAGE B: ep04/14/21/25 K3fix final.json (+ t2watch best), T2fix ep09-B / ep01-L designated (when runin_final.json).")
    out = ROOT / "dryrun"
    write_atomic(out / f"plan_{ts}.txt", "\n".join(lines) + "\n")
    write_csv(out / f"manifest_preview_{ts}.csv", rows, ["job", "episode", "family", "owner", "priority", "provisional",
                                                          "iteration", "training_sha256", "root_kind", "parquet_sha256",
                                                          "tree", "status"])  # fmt: skip
    return 1 if (fatal or f2 or want != got or not counts_ok or len(ok) != len(rows)) else 0


# ================================================================================================ selftest, status
def selftest(a: argparse.Namespace) -> int:
    """(1) check_export / check_eval on a known-good kept Tier-2 evaluation (k3full ep01 K3fix final, read-only);
    (2) negative checks; (3) the selection rule and compositions on synthetic records; (4) the forbidden-path guard;
    (5) the clean-CSV parser. Writes only under ROOT/dryrun/selftest."""
    bad, notes = [], []
    keys, keys_from = kit_metric_keys()
    _, by = inventory_index()
    spec = next(s for s in stage_a_specs() if s["job"] == "ep01_k3fix_final44922")
    job = build_job(spec, by, copy=False)
    if job["status"] != "ok":
        bad.append(f"build_job ep01_k3fix_final44922: {job['status']}")
    ed = ME / "tmp_analysis/tier2/k3full/eval/ep01_k3fix_s42"
    rep = read_json(ed / "export/ep01/export_report.json") or {}
    st, info = check_export(job, ed / "export/ep01", 0, fnum(rep.get("tolerance")) or FALLBACK_TOL)
    notes.append(f"check_export on k3full ep01 export (tolerance {rep.get('tolerance')}): {st} {info['problems']}")
    if st != "ok":
        bad.append(f"check_export known-good: {st} {info['problems']}")
    st2, info2 = check_export(job, ed / "export/ep01", 0, EXPORT_TOL)
    notes.append(f"check_export same file at {EXPORT_TOL:g} (report says {rep.get('tolerance')}): {st2} (expected not ok)")
    if st2 == "ok" and fnum(rep.get("tolerance")) != EXPORT_TOL:
        bad.append("check_export accepted a report made at another tolerance")
    exp = {"file": info["file"], "sha256": info.get("sha256") or sha256(info["file"])}
    problems, inf = check_eval(job, exp, ed / "results", ed / "eval_report.json", 0, keys)
    notes.append(f"check_eval known-good: problems {problems}; metrics finite: {all(v is not None for v in (inf.get('metrics') or {}).values())}")
    if problems:
        bad.append(f"check_eval known-good: {problems}")
    wrong = dict(job, parquet_sha256="0" * 64)
    p2, _ = check_eval(wrong, exp, ed / "results", ed / "eval_report.json", 0, keys)
    if not any("sha256" in x for x in p2):
        bad.append("check_eval did not reject a wrong training-parquet sha256")
    other = next(s for s in stage_a_specs() if s["job"] == "ep01_raw_final44922")
    rawjob = build_job(other, by, copy=False)
    p3, _ = check_eval(rawjob, exp, ed / "results", ed / "eval_report.json", 0, keys)
    if not any("reference parquet" in x for x in p3):
        bad.append("check_eval did not reject a results JSON for another training parquet")
    notes.append(f"negative checks: wrong sha -> {[x for x in p2 if 'sha256' in x][:1]}; other input -> {[x for x in p3 if 'reference' in x][:1]}")
    # refusal classifier
    if classify_refusal({"failures": ["DEV restore: max_rel_diff 1.2e-04 > 1.0e-04"], "dev_restore": {"finite": True},
                         "stock_runtime": {"comparison": {"finite": True}, "log_std_equal": True, "algorithm": "ppo"},
                         "log_std_equal": True}) != "tol":
        bad.append("classify_refusal: tolerance refusal not recognised")
    if classify_refusal({"failures": ["log_std differs"], "log_std_equal": False}) != "other":
        bad.append("classify_refusal: log_std refusal taken as tolerance")
    # synthetic selection
    base = inside_root(ROOT / "dryrun" / "selftest")
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)
    A, B = "a" * 64, "b" * 64
    sjobs = []

    def mk(jid, ep, fam, sha, prov, aucs, sp=1.0, mp=1.0, n=5):
        sjobs.append({"job": jid, "episode": ep, "family": fam, "owner": "ours", "status": "ok", "parquet_sha256": sha,
                      "provisional": prov, "iteration": 1000 + len(sjobs), "stage": "A", "run_dir": f"/x/{jid}", "root_kind": "k3fix"})
        for k in range(1, 6):
            rec = {"status": "ok", "metrics": {"auc": aucs[k - 1], "sp_sr": sp, "mp_sr": mp, "mppe": 1.0}} if k <= n else {"status": "failed"}
            write_atomic(base / "runs" / jid / f"run{k}" / "run.json", dump(rec))

    for ep in range(30):
        mk(f"ep{ep:02d}_raw", ep, "raw", B if ep == 1 else A, False, [0.5] * 5)
    mk("ep01_k3", 1, "k3fix_final", A, False, [0.80, 0.80, 0.80, 0.80, 0.80])
    mk("ep01_t2w", 1, "t2watch_best", A, False, [0.815] * 5)          # qualifies
    mk("ep01_t2fix", 1, "t2fix", A, False, [0.805] * 5)               # below the margin
    mk("ep01_sp", 1, "t2watch_best", A, False, [0.95] * 5, sp=0.8)    # SP-SR lower
    mk("ep01_diff", 1, "zuxing", B + "x", True, [0.99] * 5)           # other input: never
    mk("ep01_prov", 1, "zuxing", A, True, [0.90] * 5)                 # provisional: SELECTED+P only
    mk("ep02_k3", 2, "k3fix_final", "c" * 64, False, [0.7] * 5, n=4)  # INCOMPLETE CHOSEN
    mk("ep02_t2w", 2, "t2watch_best", "c" * 64, False, [0.9] * 5)
    sv = {"written": "selftest", "prereg_check": f"sha256(PREREG_v3.md) {PREREG_V3_SHA[:16]}... ok", "episodes": []}
    for ep in range(30):
        if ep == 1:
            sv["episodes"].append({"episode": 1, "choice": "K3FIX", "settled": True, "final_iteration": 1030, "k3fix_run": "ep01_k3"})
        elif ep == 2:
            sv["episodes"].append({"episode": 2, "choice": "K3FIX", "settled": True, "final_iteration": 1036, "k3fix_run": "ep02_k3"})
        else:
            sv["episodes"].append({"episode": ep, "choice": "PENDING" if ep == 4 else "RAW", "settled": ep != 4})
    write_atomic(base / "VERDICT.json", dump(sv))
    for j in sjobs:
        if j["job"] == "ep01_k3":
            j["iteration"] = 1030
        if j["job"] == "ep02_k3":
            j["iteration"] = 1036
    sel = select_all(sjobs, 5, base, base / "VERDICT.json")
    e1, e2, e4 = sel["episodes"][1], sel["episodes"][2], sel["episodes"][4]
    exp_ok = (e1["chosen"] == "ep01_k3" and e1["selected"] == "ep01_t2w" and e1["selected_p"] == "ep01_prov"
              and e2["chosen"] == "ep02_k3" and e2["selected"] == "ep02_k3" and "INCOMPLETE" in e2["flags"]
              and e4["chosen"] == "ep04_raw" and "PENDING" in e4["flags"]
              and "ep01_diff" not in [c["job"] for c in e1["challengers"]])  # fmt: skip
    notes.append(f"selection synthetic: ep01 SELECTED {e1['selected']} SELECTED+P {e1['selected_p']}; ep02 {e2['selected']} "
                 f"{e2['flags']}; ep04 {e4['chosen']} {e4['flags']} -> as expected: {exp_ok}")
    if not exp_ok:
        bad.append("selection rule on synthetic data")
    vals = {j["job"]: {k: {"auc": 0.1 * k + 0.01 * int(j["episode"]), "sp_sr": 1.0, "mp_sr": 1.0, "mppe": 1.0} for k in range(1, 6)}
            for j in sjobs}  # fmt: skip
    cs = comp_stats({ep: f"ep{ep:02d}_raw" for ep in range(30)}, vals, 5)["metrics"]["auc"]
    tk_mean = sum(cs["T_k"].values()) / 5
    if not (abs(tk_mean - cs["average"]) < 1e-12 and cs["T_min"] == cs["T_k"][1] and cs["T_max"] == cs["T_k"][5]):
        bad.append("composition statistics (mean of T_k != average, or T_min/T_max wrong)")
    notes.append(f"compositions synthetic: average {cs['average']:.4f}, mean T_k {tk_mean:.4f}, T_min {cs['T_min']:.4f}, T_max {cs['T_max']:.4f}")
    for tok in ("x/clean_scores.json", "y/summary_LOCAL_NOT_FOR_UPLOAD.json", "z/packed_LOCAL_NOT_FOR_UPLOAD.csv"):
        try:
            guarded_read_json(Path(tok))
            bad.append(f"guard let {tok} through")
        except PermissionError:
            pass
    notes.append("guard: selection refuses clean_/packed_/LOCAL_NOT_FOR_UPLOAD paths")
    cv = parse_clean(ed / "clean_scores_LOCAL_NOT_FOR_UPLOAD.csv", 1)
    notes.append(f"clean parser on a k3full LOCAL csv: {'4 finite metrics' if cv else 'FAILED'} (values not shown)")
    if not cv:
        bad.append("clean parser")
    summarize(sjobs, 5, base, Logger(None), base / "VERDICT.json")
    notes.append("summarize() on the synthetic records: ok (files under dryrun/selftest)")
    print(f"selftest {now()}: metrics {keys} ({keys_from})")
    for x in notes:
        print(f"  {x}")
    print(f"selftest: {len(bad)} problems" + ("" if not bad else f": {bad}"))
    return 0 if not bad else 1


def status(a: argparse.Namespace) -> int:
    jobs = load_jobs(ROOT)
    if not jobs:
        print(f"no {ROOT / 'manifest.json'} yet")
        return 0
    hb = read_json(ROOT / "heartbeat.json") or {}
    print(f"heartbeat {hb.get('updated_at')} phase {hb.get('phase')} exports {hb.get('exports')} evals {hb.get('evals')} "
          f"clean {hb.get('clean')} eta {hb.get('eta_hours_evals')} h halted {hb.get('halted')}")
    for j in hb.get("running", []):
        print(f"  running {j}")
    for r in jobs:
        jid = r["job"]
        exp = export_rec(ROOT, jid) or {}
        aucs = []
        for k in range(1, a.runs + 1):
            rr = run_rec(ROOT, jid, k)
            aucs.append("-" if rr is None else (f"{rr['metrics']['auc']:.4f}" if rr.get("status") == "ok" else "FAIL"))
        print(f"  {jid:<28} P{r.get('priority')} {str(r.get('root_kind')):<6} export {exp.get('status', 'pending'):<10} "
              f"{str(exp.get('tolerance') or ''):<7} runs {' '.join(aucs)}" + ("" if r.get("status") == "ok" else f"  ** {r.get('status')}"))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--gpus", default="0,1,2,5,6,7")
    r.add_argument("--max-parallel", type=int, default=4)
    r.add_argument("--lane-max", type=int, default=2)
    r.add_argument("--runs", type=int, default=5)
    r.add_argument("--poll", type=float, default=10.0)
    for name in ("summary", "status"):
        s = sub.add_parser(name)
        s.add_argument("--runs", type=int, default=5)
    sub.add_parser("selftest")
    a = p.parse_args()
    if a.cmd == "run":
        if a.runs != 5 or not 1 <= a.max_parallel <= 4 or not 1 <= a.lane_max <= 2:
            p.error("PREREG: --runs 5, --max-parallel <= 4, --lane-max <= 2")
        return dry_run(a) if a.dry_run else Driver(a, Logger(ROOT / "driver.log")).run()
    if a.cmd == "summary":
        jobs = load_jobs(ROOT)
        if not jobs:
            print("no manifest.json yet")
            return 1
        out = summarize(jobs, a.runs, ROOT, Logger(None))
        print(json.dumps({k: {m: v["metrics"]["auc"]["average"] for m, v in out["own"].items()} for k in ("own",)}, indent=1))
        return 0
    if a.cmd == "status":
        return status(a)
    return selftest(a)


if __name__ == "__main__":
    raise SystemExit(main())
