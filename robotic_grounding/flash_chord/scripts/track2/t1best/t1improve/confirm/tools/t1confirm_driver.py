#!/usr/bin/env python3
"""t1confirm_driver.py: T1improve candidate confirmation (PREREG_t1improve.md v2 section 8.3-8.4, ADD1 section 6) with
10 fresh official evaluations per candidate (user request 2026-10-28, CONFIRM_RULES.md) plus the world_count=64
collapse screens of 8.4 (f).

ROOT = fc_runs/T1improve/confirm. Everything is written under ROOT. The kit, the stock runtime, the code trees, the arm
folders and the t1best20 folder are only read. No process is ever signalled (an overdue evaluation is only logged).
Never GPU 4, never an MPS GPU. Python children run with PYTHONDONTWRITEBYTECODE=1 (and -B where we start Python).

CANDIDATES (`build`): the 12 ended arms in this pass; per arm the candidates are re-derived from its watcher evals.csv
  by section 8.2 (top-2 ok scan AUCs with iteration >= end - 6000, ties -> earlier iteration, plus the final
  checkpoint, deduplicated) and must equal the list of the dry pass (EXPECTED below). Bars: t1best20 per_episode.csv
  (sha-pinned) evaluator-JSON columns, as PREREG v2 section 2.
EXPORT (`export`, CPU only): each candidate is exported ONCE by `python -B -m flash_chord.training.ppo.coordinate.export`
  of its arm's own tree, stock-runtime check at 1e-4 (the command eval_all.py runs; as t1best20). Up to 2 tries (as the
  watcher); a candidate whose 2 tries both fail is not a candidate; if it is the arm's final checkpoint, the final slot
  goes to the newest checkpoint whose export passes (PREREG v2 8.3).
EVALUATIONS (`run`): tag tNN = one official evaluation of that same exported file:
  scripts/track2/eval_all.py (657295921e82...) --export no --world-count 1 -> kit examples/run_policy_evaluation.py ->
  kit flash_chord/scripts/evaluate_policy.py (160d24f5...) on the untouched stock runtime 8c778f9 (frame 0, explicit
  reset). Tag w64 = the same with --world-count 64 (the collapse screen; episodes 1, 14, 21, 28 only). Order:
  trial-major (t01 of every candidate, then t02, ...), then the screens. Checks per evaluation: as t1best20 (exit 0 and
  "evaluated"; JSON + parquet; pinned sha256 in JSON and parquet; Tier-1 sequence; start_frame 0; world_count 1 (64 for
  w64); explicit reset; no non-finite world; common evaluator build; environment_valid; pinned file unchanged).
  A completed evaluation is never re-run whatever its score; a technical failure is retried (<= 3 tries, 5 min apart).
LANES: GPUs 0,1,2,3,5,6,7 (never 4). One eval per GPU: each eval holds our lock ROOT/evallock/gpu<g>.lock plus every
  existing shared lane lock FILE of that GPU (each inode once; opened read-only, never created; all or nothing, never
  blocking). No MPS GPU; free memory >= 4096 + 1536 MiB (w64: 4096 + 8192); <= --lane-max of ours per NUMA eval-core
  lane; <= --max-parallel in all; a GPU rests --cooldown s after each of our evals. `nice -n 10 taskset -c <eval cores>`.
SUMMARY (`summary`, also spawned by `run`): per evaluation (per_eval.csv, seeds.tsv), per candidate (mean / sd / min /
  max JSON AUC, SP / MP pass counts, MPPE; criteria (a)-(f); verdict), per screen (screen_score.py per-world kit metrics:
  collapses = worlds with AUC < bar - 0.10; world-0 check), per item (qualifiers, winner, changed): verdicts.json,
  candidates_summary.csv, screens.csv, SUMMARY.md. Nothing is packed, selected for upload or uploaded here.

  python -B t1confirm_driver.py build | export | status | summary
  python -B t1confirm_driver.py run [--trials 10] [--gpus 0,1,2,3,5,6,7] [--max-parallel 4] [--lane-max 2]
                                    [--cooldown 35] [--only t01:<cand>,w64:<cand>] [--no-screens] [--poll 5]
Start: bash ROOT/launch.sh   Stop: touch ROOT/STOP (running evals finish and are recorded first; rm STOP to resume).
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
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import shlex  # noqa: E402
import socket  # noqa: E402
import statistics  # noqa: E402
import struct  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
FC = ME / "fc_runs"
T1I = FC / "T1improve"
ARMS = T1I / "arms"
ROOT = T1I / "confirm"
CANDS = ROOT / "candidates.json"
RULES, RULES_SHA = ROOT / "CONFIRM_RULES.md", ROOT / "CONFIRM_RULES.sha256"
PINS, PINS_SHA = ROOT / "EXPORT_PINS.md", ROOT / "EXPORT_PINS.sha256"
LOCKED_DOCS = {T1I / "PREREG_t1improve.md": "e418a6489a25951ce2518b9734caeed28b153645d48c7458d7c6b8700dfae79f",
               T1I / "PREREG_t1improve_ADD1.md": "5faa730bc8248135af57060d407e66a2b79ce950240e95e8888acabe0d196244",
               T1I / "PREREG_t1improve_ADD2.md": "82ec3c75ab594fe993542a2d2923b8c835836616f9a20bc873ae5d09f6f64725"}  # fmt: skip
BEST20 = FC / "submissions/t1_best20_20261026"
BEST20_PER_EP = BEST20 / "per_episode.csv"
BEST20_PER_EP_SHA = "e02cd1f80cc45a83fc80b4f5f0142411473b6db6c0bcb39ae0cd2ab3f7c27adc"
KIT = ME / "v2d_submission_kit_src/v2d_submission_kit"
STOCK_REPO = ME / "code/video_to_data_8c778f9"
STOCK = STOCK_REPO / "robotic_grounding/flash_chord"
STOCK_COMMIT = "8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef"
DATASET = ME / "data/video_to_data_challenge"
PACK_PYTHON = ME / "conda_envs/v2dkit/bin/python"
PY = ME / "conda_envs/flashchord/bin/python"
T2MAN = ME / "data/human_motion_data/whole_body/soma/track2_manifest.json"
RUNNER = KIT / "examples/run_policy_evaluation.py"
EVAL_SCRIPT = KIT / "flash_chord/scripts/evaluate_policy.py"
SCREEN_SCORER = ROOT / "screen_score.py"
TREES = {
    "b39b69b": {"commit": "b39b69be6c2f561c0123c6aeb8cd13f8b9d542e7", "code": ME / "code/nv-v2d_runin/robotic_grounding/flash_chord",
                "repo": ME / "code/nv-v2d_runin", "label": "code/nv-v2d_runin @ b39b69b"},
    "61fa5d3": {"commit": "61fa5d31ff8054bf1e5cf5375ef0957e54d2729b", "code": ME / "code/fc_r2", "repo": ME / "code/fc_r2",
                "label": "code/fc_r2 @ 61fa5d3"},
    "7fbc227": {"commit": "7fbc2273a92baefe652cb11e09c7e0acdd62da00", "code": ME / "code/nv-v2d_ep28w2/robotic_grounding/flash_chord",
                "repo": ME / "code/nv-v2d_ep28w2", "label": "code/nv-v2d_ep28w2 @ 7fbc227"},
}  # fmt: skip
EXPORTER = "flash_chord.training.ppo.coordinate.export"
EXPORT_TOL = 1e-4
EVAL_ALL_SHA = "657295921e820a464160039ad9a65ba92a683108a99dcb197290ae4147363197"
EVAL_SCRIPT_SHA = "160d24f5fbeb70ccd569ca4192056f1892b2a9f2999066db9d8279461648fdf1"
EVAL_CODE_SHA = "b1b3ebe3ae4629d1687a83f420149e3f6fdfc154d70acad55c12fc8fa69811a1"
RUNNER_SHA = "0bcc41f5de246a51f194409b3e5292e04109012cb22a433a8f7ed477df04e55d"
EXPORTER_SHA = {"b39b69b": "04367c56564de9540deb3d8e33adbfddb7800df190361a0576005a1a4ccfd212",
                "7fbc227": "d9a7d45da58c44d658af3ef733061c0462bb04c5867994c74cc9e246017eb585",
                "61fa5d3": "34e3d7f3a7dca4fdd1e7c4d26750c6332abcc1d629f6a9c6c7a965833a22206f"}  # PREREG v2 section 4
EVAL_TIMEOUT, EXPORT_TIMEOUT, SCORE_TIMEOUT = 2400.0, 1800.0, 3600.0
RETRY_DELAY = 300.0
EVAL_TRIES, EXPORT_TRIES, FALLBACK_MAX = 3, 2, 5
MIN_FREE_MB, EVAL_PEAK_MB, W64_PEAK_MB, MIN_FREE_PIDS = 4096, 1536, 8192, 2000
EVAL_CORES = ("48-51,152-155", "100-103,204-207")  # the eval lanes of scripts/runin_helper.py
ALLOWED_GPUS = (0, 1, 2, 3, 5, 6, 7)  # never GPU 4
OWN_LOCKDIR = ROOT / "evallock"
SHARED_LOCKDIRS = (FC / "queue_runin/evallock", FC / "queue_tier2v2/evallock", FC / "queue_t1fix/evallock",
                   FC / "queue_t1fix2/evallock", FC / "queue_t1fix3/evallock", FC / "queue_t1fix4/evallock",
                   FC / "queue_t1fix5/evallock", FC / "queue_t2fix/evallock", FC / "queue_t1improve/evallock",
                   FC / "submissions/t1_best20_20261026/evallock", FC / "submissions/t3_f0_final/evallock",
                   FC / "submissions/t2_consensus_r2_20261027/evallock", FC / "submissions/t2_consensus_r3_20261028/evallock",
                   ME / "tmp_analysis/tier3/f0/evallock", ME / "tmp_analysis/tier3/opus_easy/evallock",
                   ME / "tmp_analysis/tier2/k3ca/train/evallock", ME / "tmp_analysis/tier2/k3fix_pilot/evallock",
                   ME / "tmp_analysis/tier2/k3full/evallock", ME / "tmp_analysis/tier2/k3easy/evallock")  # fmt: skip
METRICS = ("auc", "sp_sr", "mp_sr", "mppe")
FALLBACK_KEYS = {"auc": "add_auc", "sp_sr": "spider_sr_uncentered", "mp_sr": "maniptrans_sr", "mppe": "mean_per_point_error_cm"}
HOST = socket.gethostname()
MARK = "T1CONFIRM_JOB"
SEED_COLS = ["env.action.body_delay.seed", "body_delay.group_seeds", "env.action.body_delay.min_steps",
             "env.action.body_delay.max_steps", "env.reset.seed(stored)", "env.reset.seed(effective)",
             "scene.object_scale_seed", "scene.object_scale_min", "scene.object_scale_max",
             "env.objective.wrench_basis_seed", "training.seed", "evaluation.ADD_VERTEX_SEED", "v2dlb.ADD_VERTEX_SEED",
             "seed_signature"]  # fmt: skip

# ---- the pass (PREREG v2 section 2 order, then ADD1 section 2 order; the tie-break order of 8.4 / ADD1 6)
ARMS_V2 = ["ep14-R", "ep28-W3H", "ep21-B5", "ep10-Bc", "ep10-BT", "ep19-Bc", "ep01-Bc", "ep15-B2", "ep05-B3", "ep25-B25"]
ARMS_ADD1 = ["ep21-B5s2", "ep14-Rs2", "ep19-B5", "ep10-BTs2", "ep05-P30", "ep15-P30"]
ARM_RANK = {a: i for i, a in enumerate(ARMS_V2 + ARMS_ADD1)}
OUT_OF_PASS = ("ep14-Rs2", "ep19-B5", "ep10-BTs2", "ep05-P30")  # still training on 2026-10-28 14:20: a later pass
EXPECTED = {"ep14-R": [92000, 95500, 96000], "ep28-W3H": [78000, 83000], "ep21-B5": [136000, 138000, 139500],
            "ep21-B5s2": [139000, 139500], "ep10-Bc": [59500, 60000], "ep10-BT": [54000, 58000, 60000],
            "ep19-Bc": [130000, 135000], "ep01-Bc": [161500, 162000], "ep15-B2": [40000, 42000, 46000],
            "ep15-P30": [40000, 41000], "ep05-B3": [60000, 62000, 62500], "ep25-B25": [116000, 122000]}  # fmt: skip
BAR_ITEM = {14: "ep14", 28: "ep28alt", 21: "ep21", 10: "ep10", 19: "ep19", 1: "ep01", 15: "ep15", 5: "ep05", 25: "ep25"}
PREREG_TABLE = {"ep14": (0.8265, 1.503), "ep28alt": (0.8073, 3.466), "ep21": (0.7548, 1.497), "ep10": (0.8675, 1.312),
                "ep19": (0.8951, 1.023), "ep01": (0.7603, 3.673), "ep15": (0.9370, 0.876), "ep05": (0.9480, 0.786),
                "ep25": (0.7490, 2.013)}  # PREREG v2 section 2 (rounded): JSON AUC mean, JSON MPPE mean
SCREEN_EPISODES = (1, 14, 21, 28)
W64 = 64
MARGIN, FLOOR_DROP, MPPE_SLACK, MPPE_SLACK_BIG, BIG_GAIN = 0.005, 0.10, 0.15, 0.30, 0.02
WORLD0_TOL = 0.01


# ================================================================================================ small helpers
def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_time(text) -> float | None:
    try:
        return dt.datetime.strptime(str(text), "%Y-%m-%d %H:%M:%S").timestamp()
    except (TypeError, ValueError):
        return None


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


def read_json(file) -> dict | None:
    try:
        data = json.loads(Path(file).read_text())
    except (OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def inside_root(target: Path) -> Path:
    if not Path(os.path.abspath(target)).is_relative_to(ROOT):
        raise SystemExit(f"refusing to write outside {ROOT}: {target}")
    return target


def write_atomic(file: Path, text: str) -> None:
    inside_root(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_name(f".{file.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, file)


def dump(data) -> str:
    return json.dumps(data, indent=1, default=str) + "\n"


def write_table(file: Path, rows: list[dict], cols: list[str], delimiter: str = ",") -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", lineterminator="\n", delimiter=delimiter)
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})
    write_atomic(file, buf.getvalue())


def numa(gpu: int) -> int:
    return 0 if gpu <= 2 else 1


def tries(base: Path) -> list[tuple[int, Path]]:
    try:
        names = os.listdir(base)
    except OSError:
        return []
    return sorted((int(m.group(1)), base / n) for n in names if (m := re.match(r"^try(\d+)$", n)) and (base / n).is_dir())


def mstats(vals: list[float]) -> dict:
    if not vals:
        return {"mean": None, "sd": None, "min": None, "max": None}
    return {"mean": sum(vals) / len(vals), "sd": statistics.stdev(vals) if len(vals) >= 2 else 0.0,
            "min": min(vals), "max": max(vals)}  # fmt: skip


def git_head(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60,
                              env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"}).stdout.strip()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return ""


class Logger:
    def __init__(self, file: Path | None):
        self.file, self.once = file, set()

    def __call__(self, message: str) -> None:
        line = f"[{now()}] {message}"
        print(line, flush=True)
        if self.file is not None:
            with inside_root(self.file).open("a") as fp:
                fp.write(line + "\n")

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
                names = {"auc": "add_auc", "sp_sr": "spider_sr", "mp_sr": "maniptrans_sr", "mppe": "mppe_cm"}
                return {ours: table[name] for ours, name in names.items()}, "kit v2dlb/report_schema.py"
        raise KeyError("REPORT_METRIC_KEY not found")
    except Exception as error:  # noqa: BLE001
        return dict(FALLBACK_KEYS), f"built-in fallback ({error!r})"


def const_from(path: Path, name: str):
    for node in ast.parse(path.read_text()).body:
        targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
        if any(getattr(t, "id", None) == name for t in targets) and getattr(node, "value", None) is not None:
            return ast.literal_eval(node.value)
    return None


# ================================================================================================ seeds
def resolved_config(file: Path) -> dict:
    with Path(file).open("rb") as fp:
        n = struct.unpack("<Q", fp.read(8))[0]
        header = json.loads(fp.read(n))
    return json.loads(header["__metadata__"]["user.resolved_config_json"])


def dig(c, *keys):
    for k in keys:
        if not isinstance(c, dict) or k not in c:
            return "MISSING"
        c = c[k]
    return c


def harness_forced_reset_seed() -> int | None:
    text = (STOCK / "src/flash_chord/evaluation/harness.py").read_text()
    m = re.search(r'config\["env"\]\["reset"\]\["seed"\]\s*=\s*(\d+)', text)
    return int(m.group(1)) if m else None


def seeds_of(file: Path) -> dict:
    """Every seed the official evaluation of this file uses (t1best20 seeds_of, unchanged)."""
    c = resolved_config(file)
    bd = dig(c, "env", "action", "body_delay")
    seed = bd.get("seed") if isinstance(bd, dict) else "MISSING"
    out = {
        "env.action.body_delay.seed": seed,
        "body_delay.group_seeds": ",".join(str(seed + g) for g in range(5)) if isinstance(seed, int) else "MISSING",
        "env.action.body_delay.min_steps": bd.get("min_steps") if isinstance(bd, dict) else "MISSING",
        "env.action.body_delay.max_steps": bd.get("max_steps") if isinstance(bd, dict) else "MISSING",
        "env.reset.seed(stored)": dig(c, "env", "reset", "seed"),
        "env.reset.seed(effective)": harness_forced_reset_seed(),
        "scene.object_scale_seed": dig(c, "scene", "object_scale_seed"),
        "scene.object_scale_min": dig(c, "scene", "object_scale_min"),
        "scene.object_scale_max": dig(c, "scene", "object_scale_max"),
        "env.objective.wrench_basis_seed": dig(c, "env", "objective", "wrench_basis_seed"),
        "training.seed": dig(c, "training", "seed"),
        "evaluation.ADD_VERTEX_SEED": const_from(STOCK / "src/flash_chord/evaluation/metrics.py", "ADD_VERTEX_SEED"),
        "v2dlb.ADD_VERTEX_SEED": const_from(KIT / "v2dlb/vertices.py", "ADD_VERTEX_SEED"),
    }  # fmt: skip
    out["seed_signature"] = (f"delay={out['env.action.body_delay.seed']}(+g0..4,{out['env.action.body_delay.min_steps']}-"
                             f"{out['env.action.body_delay.max_steps']}) reset={out['env.reset.seed(effective)']} "
                             f"scale_seed={out['scene.object_scale_seed']}({out['scene.object_scale_min']}/"
                             f"{out['scene.object_scale_max']}) wrench={out['env.objective.wrench_basis_seed']} "
                             f"vertex={out['evaluation.ADD_VERTEX_SEED']}/{out['v2dlb.ADD_VERTEX_SEED']} "
                             f"ppo={out['training.seed']}")  # fmt: skip
    return out


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
    found: dict[str, list] = {}
    for pid in all_pids():
        job = proc_environ(pid).get(MARK)
        st = proc_stat(pid)
        if job and st is not None and st[0] not in ("Z", "X"):
            found.setdefault(job, []).append((pid, st[19]))
    return found


def live_trainer(arm: str) -> list[int]:
    out = []
    for pid in all_pids():
        if proc_environ(pid).get("T1IMPROVE_ARM") == arm and any("train_rl.py" in x for x in proc_cmdline(pid)):
            st = proc_stat(pid)
            if st is not None and st[0] not in ("Z", "X"):
                out.append(pid)
    return out


def smi() -> dict:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.used,memory.free",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60, check=True).stdout
    gpus, by_uuid = {}, {}
    for line in out.strip().splitlines():
        idx, uuid, total, used, free = (x.strip() for x in line.split(","))
        gpus[int(idx)] = {"uuid": uuid, "total": int(total), "used": int(used), "free": int(free)}
        by_uuid[uuid] = int(idx)
    if not gpus:
        raise RuntimeError("nvidia-smi listed no GPU")
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory,process_name",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60, check=True).stdout
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


def lane_lock_groups(gpu: int) -> list[list[Path]]:
    """The existing shared lock paths of this GPU grouped by the file they name (after following links): flock belongs
    to the open file description, so opening one inode twice would make us refuse our own lock (t2cons_r3 C1)."""
    groups: dict[tuple[int, int], list[Path]] = {}
    for d in SHARED_LOCKDIRS:
        p = d / f"gpu{gpu}.lock"
        try:
            st = os.stat(p)
        except OSError:
            continue
        if p.is_file():
            groups.setdefault((st.st_dev, st.st_ino), []).append(p)
    return list(groups.values())


def lane_lock_files(gpu: int) -> list[Path]:
    return [g[0] for g in lane_lock_groups(gpu)]


def take_lane(gpu: int) -> tuple[list[int], list[str]] | str:
    OWN_LOCKDIR.mkdir(parents=True, exist_ok=True)
    own = try_lock(inside_root(OWN_LOCKDIR / f"gpu{gpu}.lock"), create=True)
    if own is None:
        return f"gpu{gpu}: our own lane lock is held"
    fds, taken = [own], [str(OWN_LOCKDIR / f"gpu{gpu}.lock")]
    for p in lane_lock_files(gpu):
        fd = try_lock(p, create=False)
        if fd is None:
            for x in fds:
                os.close(x)
            return f"gpu{gpu}: {p} is held"
        fds.append(fd)
        taken.append(str(p))
    return fds, taken


def child_env(gpu: int | None, job_dir: Path) -> dict:
    """The watcher's eval environment (as t1best20 child_env), marker T1CONFIRM_JOB=<job dir>, TMPDIR inside ROOT."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CUDA_MPS_", "WANDB_"))
           and k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "RUNIN_QUEUE", "RUNIN_GPU", "RUNIN_EVAL",
                         "FLASH_CHORD_CACHE_DIR", "T1BEST5_JOB", "T1BEST20_JOB", "T1IMPROVE_ARM", "T1FIX_ARM")}  # fmt: skip
    sp = ME / "conda_envs/flashchord/lib/python3.11/site-packages/nvidia"
    env.update(
        PATH=f"{ME}/conda_envs/flashchord/bin:{env.get('PATH', '/usr/bin:/bin')}",
        LD_LIBRARY_PATH=f"{sp}/cu13/lib:{sp}/cudnn/lib:{sp}/nccl/lib:{sp}/nvshmem/lib",
        CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
        XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
        NUMEXPR_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1", WANDB_MODE="disabled",
        HUMAN_MOTION_DATA_DIR=str(ME / "data/human_motion_data"), XDG_CACHE_HOME=str(ME / ".cache"),
        TZ="Asia/Shanghai", TMPDIR=str(ROOT / "tmp"),
    )  # fmt: skip
    env[MARK] = str(job_dir)
    return env


ENV_SHOWN = ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "JAX_PLATFORMS", "PYTHONDONTWRITEBYTECODE", "XLA_PYTHON_CLIENT_PREALLOCATE",
             "OMP_NUM_THREADS", "HUMAN_MOTION_DATA_DIR", "TMPDIR", MARK)  # fmt: skip


def show_command(prefix: list[str], cmd: list[str], cwd: Path, env: dict) -> str:
    shown = " ".join(f"{k}={shlex.quote(env[k])}" for k in ENV_SHOWN if k in env)
    return f"cd {shlex.quote(str(cwd))} && env {shown} {shlex.join(prefix + cmd)}"


# ================================================================================================ candidates
def read_bars() -> dict[str, dict]:
    if sha256(BEST20_PER_EP) != BEST20_PER_EP_SHA:
        raise SystemExit(f"{BEST20_PER_EP} sha256 changed (want {BEST20_PER_EP_SHA[:16]})")
    rows = {r["item"]: r for r in csv.DictReader(BEST20_PER_EP.open())}
    bars = {}
    for item, (auc_r, mppe_r) in PREREG_TABLE.items():
        r = rows[item]
        auc, mppe = float(r["json_auc_mean"]), float(r["json_mppe_mean"])
        if abs(auc - auc_r) > 5e-5 + 1e-9 or abs(mppe - mppe_r) > 5e-4 + 1e-9:
            raise SystemExit(f"{item}: per_episode.csv JSON AUC/MPPE {auc}/{mppe} != PREREG v2 table {auc_r}/{mppe_r}")
        bars[item] = {"item": item, "json_auc_mean": auc, "json_auc_sd": float(r["json_auc_sd"]), "json_mppe_mean": mppe,
                      "needed": auc + MARGIN, "floor": auc - FLOOR_DROP, "mppe_cap": mppe + MPPE_SLACK,
                      "mppe_cap_big_gain": mppe + MPPE_SLACK_BIG, "n_trials": int(r["n_evals_ok"]),
                      "sp_pass": int(r["sp_pass"]), "mp_pass": int(r["mp_pass"]), "kit_auc_mean": float(r["auc_mean"]),
                      "iteration": int(r["iteration"]), "eval_file_sha256": r["eval_file_sha256"]}  # fmt: skip
    return bars


def derive(arm: str) -> dict:
    """Section 8.2 for one ended arm, from its launch.json, run dir and watcher evals.csv."""
    lj = read_json(ARMS / arm / "launch.json")
    if not lj:
        raise SystemExit(f"{arm}: no launch.json")
    ep = int(lj["episode"])
    rd = Path(lj["output_dir"])
    if rd != ARMS / arm / "runs" / f"ep{ep:02d}":
        raise SystemExit(f"{arm}: output_dir {rd} unexpected")
    models = sorted(int(m.group(1)) for f in os.listdir(rd) if (m := re.match(r"^model_(\d+)\.safetensors$", f)))
    end = models[-1]
    ended = (rd / "final.json").is_file() and end == int(lj["max_iterations"]) and not live_trainer(arm)
    rows = list(csv.DictReader((ARMS / arm / "eval" / "evals.csv").open()))
    scans = [r for r in rows if r["purpose"] == "scan" and r["status"] == "ok" and int(r["iteration"]) >= end - 6000]
    top = sorted(scans, key=lambda r: (-float(r["auc"]), int(r["iteration"])))
    picked: dict[int, list[str]] = {}
    for r in top:
        it = int(r["iteration"])
        if it not in picked:
            picked[it] = [f"scan top-{len(picked) + 1} (AUC {float(r['auc']):.5f}, export rel {float(r['export_rel_diff']):.2e})"]
        if len(picked) == 2:
            break
    picked.setdefault(end, []).append("final checkpoint")
    scan_of = {int(r["iteration"]): r for r in scans}
    return {"arm": arm, "episode": ep, "tree": lj["tree"], "end": end, "ended": ended, "models": models,
            "candidates": sorted(picked), "nominated": picked, "scan_rows": scan_of, "launch": lj}  # fmt: skip


def make_item(d: dict, it: int, nominated: list[str], bars: dict, seqs: dict) -> dict:
    arm, ep = d["arm"], d["episode"]
    tree = TREES[d["tree"]]
    train = ARMS / arm / "runs" / f"ep{ep:02d}" / f"model_{it}.safetensors"
    seq, obj = seqs[ep]
    if not any(f"sequence_id={seq}/" in str(x) for x in d["launch"]["argv"]):
        raise SystemExit(f"{arm}: launch argv does not name the Tier-1 sequence {seq}")
    s = d["scan_rows"].get(it) or {}
    return {"name": f"{arm}_{it}", "arm": arm, "arm_rank": ARM_RANK[arm], "wave": "v2" if arm in ARMS_V2 else "ADD1",
            "episode": ep, "object": obj, "sequence_id": seq, "iteration": it, "is_final": it == d["end"],
            "nominated": nominated, "scan_auc": fnum(s.get("auc")), "scan_export_rel_diff": fnum(s.get("export_rel_diff")),
            "training_checkpoint": str(train), "training_sha256": sha256(train), "code_commit": tree["commit"],
            "code_tree": str(tree["code"]), "code_label": tree["label"], "tree": d["tree"], "bar_item": BAR_ITEM[ep],
            "bar": bars[BAR_ITEM[ep]], "screen": ep in SCREEN_EPISODES,
            "training_overrides": d["launch"].get("overrides"), "reset_seed": d["launch"].get("reset_seed")}  # fmt: skip


def cmd_build(a) -> int:
    log = Logger(ROOT / "build.log")
    if CANDS.is_file() and RULES_SHA.is_file():
        print(f"{CANDS} is frozen by {RULES_SHA}: not rebuilt")
        return 1
    bars = read_bars()
    seqs = {int(r["episode_index"]): (str(r.get("sequence_id", s)), str(r["object"])) for s, r in json.loads(T2MAN.read_text()).items()}
    items, derived = [], {}
    for arm in sorted(EXPECTED, key=lambda x: ARM_RANK[x]):
        d = derive(arm)
        derived[arm] = {k: v for k, v in d.items() if k not in ("scan_rows", "launch")}
        if not d["ended"]:
            raise SystemExit(f"{arm}: not ended (final.json / max_iterations / live trainer)")
        if d["candidates"] != EXPECTED[arm]:
            raise SystemExit(f"{arm}: section 8.2 gives {d['candidates']} != the dry pass {EXPECTED[arm]}")
        for it in d["candidates"]:
            items.append(make_item(d, it, d["nominated"][it], bars, seqs))
            log(f"  {items[-1]['name']:22s} ep{d['episode']:02d} {items[-1]['object']:18s} {d['tree']} "
                f"{'; '.join(d['nominated'][it])}")
    for arm in OUT_OF_PASS:
        lj = read_json(ARMS / arm / "launch.json") or {}
        log(f"  out of this pass (still training): {arm} (live trainer pids {live_trainer(arm)}, target {lj.get('max_iterations')})")
    write_atomic(CANDS, dump({"built_at": now(), "best20_per_episode_sha256": BEST20_PER_EP_SHA, "bars": bars,
                              "derived": derived, "out_of_pass": list(OUT_OF_PASS), "n_items": len(items), "items": items}))
    log(f"build: {CANDS} written ({len(items)} candidates of {len(EXPECTED)} arms)")
    return 0


def load_items() -> list[dict]:
    data = read_json(CANDS)
    if not data or not data.get("items"):
        raise SystemExit(f"{CANDS} missing: run `build` first")
    return data["items"]


def export_rec(name: str) -> dict | None:
    return read_json(ROOT / "exports" / name / "export.json")


def run_rec(name: str, tag: str) -> dict | None:
    return read_json(ROOT / "runs" / name / tag / "run.json")


# ================================================================================================ checks
def parquet_meta(pp: Path) -> dict:
    import pyarrow.parquet as pq

    f = pq.ParquetFile(pp)
    return {k.decode(errors="replace"): v.decode(errors="replace") for k, v in (f.schema_arrow.metadata or {}).items()
            if not k.startswith(b"pandas")}  # fmt: skip


def parquet_problems(pp: Path, sha: str, step_count, world_count: int) -> tuple[list[str], dict]:
    import pyarrow.parquet as pq

    md = parquet_meta(pp)
    out = []
    if md.get("flash_chord.checkpoint_sha256") != sha:
        out.append("parquet checkpoint_sha256 != the pinned file")
    if md.get("flash_chord.eval_script_sha256") != EVAL_SCRIPT_SHA:
        out.append(f"parquet eval_script_sha256 {md.get('flash_chord.eval_script_sha256')} is not the common build")
    if md.get("flash_chord.eval_code_sha256") != EVAL_CODE_SHA:
        out.append(f"parquet eval_code_sha256 {md.get('flash_chord.eval_code_sha256')} is not the common build")
    if str(md.get("flash_chord.step_count")) != str(step_count):
        out.append(f"parquet step_count {md.get('flash_chord.step_count')} != JSON {step_count}")
    if str(md.get("flash_chord.world_count")) != str(world_count):
        out.append(f"parquet world_count {md.get('flash_chord.world_count')} != {world_count}")
    valid = pq.read_table(pp, columns=["environment_valid"]).column("environment_valid").to_pylist()
    if not valid or not all(valid):
        out.append("environment_valid=false rows")
    keep = {k.replace("flash_chord.", ""): md.get(k) for k in md if k.startswith("flash_chord.")}
    return out, keep


def tier1_seq_ok(parquet, seq: str) -> bool:
    return f"/data/human_motion_data/whole_body/soma/sequence_id={seq}/" in str(parquet or "")


def check_eval(item: dict, exp: dict, results: Path, report: Path | None, exit_code, keys: dict,
               world_count: int) -> tuple[list[str], dict]:
    ep, seq = int(item["episode"]), item["sequence_id"]
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
    info: dict = {"results_json": str(jp), "results_parquet": str(pp), "world_count": world_count}
    js = read_json(jp)
    if js is None:
        problems.append("no results JSON")
        return problems, info
    m = js.get("metrics") or {}
    info["metrics"] = {k: fnum(m.get(src)) for k, src in keys.items()}
    info.update(chord_sr=fnum(m.get("chord_sr")), checkpoint_sha256=js.get("checkpoint_sha256"),
                checkpoint=js.get("checkpoint"), parquet_reference=js.get("parquet"), step_count=js.get("step_count"),
                json_sha256=sha256(jp))  # fmt: skip
    if any(v is None for v in info["metrics"].values()):
        problems.append(f"metrics missing ({info['metrics']})")
    if js.get("checkpoint_sha256") != exp["sha256"]:
        problems.append("results JSON checkpoint_sha256 != the pinned file")
    if os.path.realpath(str(js.get("checkpoint", ""))) != os.path.realpath(exp["file"]):
        problems.append(f"evaluated checkpoint path {js.get('checkpoint')} is not the pinned file")
    if not tier1_seq_ok(js.get("parquet"), seq):
        problems.append(f"evaluated parquet {js.get('parquet')} is not the Tier-1 sequence {seq}")
    for key, want in (("start_frame", 0), ("motion_start_frame", 0), ("world_count", world_count), ("reset_mode", "explicit")):
        if js.get(key) != want:
            problems.append(f"{key}={js.get(key)!r}, expected {want!r}")
    if js.get("non_finite_worlds"):
        problems.append(f"non_finite_worlds {js.get('non_finite_worlds')}")
    if not pp.is_file():
        problems.append("no rollout parquet")
    else:
        try:
            pprob, meta = parquet_problems(pp, exp["sha256"], js.get("step_count"), world_count)
            problems += pprob
            info["parquet_meta"] = meta
            info["parquet_sha256"] = sha256(pp)
        except Exception as error:  # noqa: BLE001
            problems.append(f"parquet unreadable ({error!r})")
    if not Path(exp["file"]).is_file():
        problems.append("the pinned file is MISSING")
    elif sha256(exp["file"]) != exp["sha256"]:
        problems.append("the pinned file CHANGED")
    log_file = results / "logs" / f"episode_{ep:06d}.log"
    try:
        first = log_file.read_text().splitlines()[0]
        info["runner_command"] = re.sub(r"^\[[^\]]+\] gpu [^:]+: ", "", first)
    except (OSError, IndexError):
        info["runner_command"] = ""
    return problems, info


def check_export(item: dict, d: Path, exit_code) -> tuple[str, dict]:
    """'ok' | 'refused' (the exporter's tolerance check said no) | 'failed' (technical)."""
    src = Path(item["training_checkpoint"])
    dst = d / src.name
    info: dict = {"file": str(dst), "report": str(d / "export_report.json"), "exit": exit_code, "problems": []}
    problems = info["problems"]
    if exit_code not in (0, None):
        problems.append(f"exit {exit_code}")
    rep = read_json(d / "export_report.json")
    if rep is None:
        problems.append("no export_report.json")
        return "failed", info
    stock = rep.get("stock_runtime") or {}
    info.update(max_rel_diff=rep.get("max_rel_diff"), max_abs_diff=rep.get("max_abs_diff"), tolerance=rep.get("tolerance"),
                source_sha256=rep.get("source_sha256"), destination_sha256=rep.get("destination_sha256"),
                stock_verified=stock.get("verified"), export_format=rep.get("export_format"),
                source_kind=rep.get("source_kind"), report_status=rep.get("status"))  # fmt: skip
    if rep.get("status") == "refused":
        problems.append(f"exporter refused: {rep.get('failures')}")
        return "refused", info
    if rep.get("status") != "ok":
        problems.append(f"report status {rep.get('status')}")
    if fnum(rep.get("tolerance")) != EXPORT_TOL:
        problems.append(f"tolerance {rep.get('tolerance')} != {EXPORT_TOL}")
    if stock.get("verified") is not True:
        problems.append(f"stock runtime verified={stock.get('verified')}")
    if os.path.realpath(str(stock.get("path", ""))) != os.path.realpath(STOCK):
        problems.append(f"verified against {stock.get('path')}, not {STOCK}")
    if stock.get("eval_code_sha256") not in (None, EVAL_CODE_SHA):
        problems.append(f"stock eval_code_sha256 {stock.get('eval_code_sha256')} is not the common build")
    if stock.get("curriculum_stage_voc_scale") not in (None, 0, 0.0):
        problems.append(f"curriculum_stage_voc_scale {stock.get('curriculum_stage_voc_scale')} (help) != 0")
    rel = fnum(rep.get("max_rel_diff"))
    if rel is None or rel > EXPORT_TOL:
        problems.append(f"max_rel_diff {rep.get('max_rel_diff')} > {EXPORT_TOL}")
    if rep.get("source_sha256") != item["training_sha256"]:
        problems.append("report source_sha256 != the item's training_sha256")
    if not dst.is_file():
        problems.append("exported file missing")
    else:
        info["sha256"] = sha256(dst)
        if info["sha256"] != rep.get("destination_sha256"):
            problems.append("exported file sha256 != report destination_sha256")
    return ("ok" if not problems else "failed"), info


# ================================================================================================ commands
def export_command(item: dict, d: Path) -> tuple[list[str], Path, dict]:
    code = Path(item["code_tree"])
    src = Path(item["training_checkpoint"])
    cmd = [str(PY), "-B", "-m", EXPORTER, "--input", str(src), "--output", str(d / src.name),
           "--stock-runtime", str(STOCK), "--report", str(d / "export_report.json"), "--tolerance", repr(EXPORT_TOL)]
    return cmd, code, {"PYTHONPATH": str(code / "src"), "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""}


def eval_command(item: dict, w: Path, gpu: int, world_count: int) -> tuple[list[str], Path]:
    code = Path(item["code_tree"])
    cmd = [str(PY), "-B", str(code / "scripts/track2/eval_all.py"), "--arm-dir", str(w), "--arm", "T1confirm",
           "--export", "no", "--iteration", str(item["iteration"]), "--episodes", str(item["episode"]), "--gpus", str(gpu),
           "--jobs-per-gpu", "1", "--world-count", str(world_count), "--skip-pack", "--kit-root", str(KIT),
           "--stock-runtime", str(STOCK), "--dataset-root", str(DATASET), "--manifest", str(T2MAN),
           "--pack-python", str(PACK_PYTHON), "--results", str(w / "results"), "--code", str(code), "--python", str(PY)]
    return cmd, code


# ================================================================================================ preflight + locks
def tool_hashes() -> dict:
    out = {"kit_runner": sha256(RUNNER), "kit_evaluate_policy": sha256(EVAL_SCRIPT),
           "stock_evaluate_policy": sha256(STOCK / "scripts/evaluate_policy.py"),
           "screen_scorer": sha256(SCREEN_SCORER) if SCREEN_SCORER.is_file() else None,
           "driver": sha256(Path(__file__).resolve())}  # fmt: skip
    for label, tree in TREES.items():
        out[f"eval_all_{label}"] = sha256(tree["code"] / "scripts/track2/eval_all.py")
        out[f"exporter_{label}"] = sha256(tree["code"] / "src/flash_chord/training/ppo/coordinate/export.py")
    return out


def preflight() -> tuple[list[str], list[str]]:
    fatal, info = [], []
    for need in (PY, PACK_PYTHON, RUNNER, EVAL_SCRIPT, STOCK / "src/flash_chord/__init__.py", T2MAN, SCREEN_SCORER):
        if not need.exists():
            fatal.append(f"missing {need}")
    if fatal:
        return fatal, info
    h = tool_hashes()
    for key, want in (("kit_evaluate_policy", EVAL_SCRIPT_SHA), ("stock_evaluate_policy", EVAL_SCRIPT_SHA), ("kit_runner", RUNNER_SHA)):
        (info if h[key] == want else fatal).append(f"{key} sha256 {h[key][:16]} " + ("ok" if h[key] == want else f"!= {want[:16]}"))
    head = git_head(STOCK_REPO)
    (info if head == STOCK_COMMIT else fatal).append(f"stock runtime HEAD {head[:12]} (want {STOCK_COMMIT[:12]})")
    for label, tree in TREES.items():
        if h[f"eval_all_{label}"] != EVAL_ALL_SHA:
            fatal.append(f"{tree['label']}: eval_all.py {h[f'eval_all_{label}'][:12]} != {EVAL_ALL_SHA[:12]}")
        if h[f"exporter_{label}"] != EXPORTER_SHA[label]:
            fatal.append(f"{tree['label']}: exporter {h[f'exporter_{label}'][:12]} != {EXPORTER_SHA[label][:12]}")
        hd = git_head(tree["repo"])
        if hd != tree["commit"]:
            fatal.append(f"{tree['label']}: HEAD {hd[:12]} != {tree['commit'][:12]}")
        else:
            info.append(f"{tree['label']}: HEAD ok, eval_all.py {h[f'eval_all_{label}'][:12]}, exporter {h[f'exporter_{label}'][:12]}")
    for doc, want in LOCKED_DOCS.items():
        got = sha256(doc)
        (info if got == want else fatal).append(f"{doc.name} sha256 {got[:16]} " + ("ok" if got == want else f"!= {want[:16]}"))
    for g in ALLOWED_GPUS:
        info.append(f"gpu{g} shared lane lock files: " + "; ".join(
            str(grp[0]).replace(str(ME) + "/", "") + (f" (= {', '.join(str(x).replace(str(ME) + '/', '') for x in grp[1:])})" if grp[1:] else "")
            for grp in lane_lock_groups(g)))  # fmt: skip
    return fatal, info


def locked(doc: Path, sha_file: Path) -> str | None:
    if not doc.is_file() or not sha_file.is_file():
        return f"{doc.name} / {sha_file.name} missing"
    want = sha_file.read_text().split()[0]
    got = sha256(doc)
    return None if got == want else f"{doc.name} sha256 {got[:16]} != {sha_file.name} {want[:16]}"


def rules_ok(items: list[dict], need_pins: bool) -> str | None:
    """CONFIRM_RULES.md frozen and naming this driver and the screen scorer; EXPORT_PINS.md frozen and naming every
    usable export (needed before any evaluation)."""
    why = locked(RULES, RULES_SHA)
    if why:
        return why
    text = RULES.read_text()
    for label, path in (("driver", Path(__file__).resolve()), ("screen scorer", SCREEN_SCORER)):
        if sha256(path) not in text:
            return f"CONFIRM_RULES.md does not name the current {label} sha256 {sha256(path)[:16]}"
    if not need_pins:
        return None
    why = locked(PINS, PINS_SHA)
    if why:
        return why
    pins = PINS.read_text()
    for it in items:
        e = export_rec(it["name"]) or {}
        if e.get("status") == "ok" and e["sha256"] not in pins:
            return f"{it['name']}: export sha256 {e['sha256'][:16]} not in EXPORT_PINS.md"
        if e.get("status") not in ("ok", "refused", "failed"):
            return f"{it['name']}: export not final ({e.get('status')})"
    return None


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
    return {"taken_at": now(), "kit": tree_listing(KIT), "stock": tree_listing(STOCK)}


def integrity_diff(a: dict, b: dict) -> list[str]:
    out = []
    for key in ("kit", "stock"):
        x, y = a.get(key, {}), b.get(key, {})
        for p in sorted(set(x) | set(y)):
            if x.get(p) != y.get(p):
                out.append(f"{key}: {p} " + ("new" if p not in x else "removed" if p not in y else "changed"))
    return out


# ================================================================================================ export
def fallback_item(items: list[dict], it_final: dict, bars: dict, seqs: dict, tried: set[int]) -> dict | None:
    """PREREG v2 8.3: the final-checkpoint slot goes to the arm's newest checkpoint whose export passes."""
    d = derive(it_final["arm"])
    have = {x["iteration"] for x in items if x["arm"] == it_final["arm"]} | tried
    for it in sorted(d["models"], reverse=True):
        if it not in have:
            return make_item(d, it, [f"final-slot fallback (final {it_final['iteration']} export failed at 1e-4)"], bars, seqs)
    return None


def cmd_export(a) -> int:
    log = Logger(ROOT / "export.log")
    why = rules_ok(load_items(), need_pins=False)
    if why:
        log(f"export refused: {why}")
        return 1
    data = read_json(CANDS)
    items = data["items"]
    seqs = {int(r["episode_index"]): (str(r.get("sequence_id", s)), str(r["object"])) for s, r in json.loads(T2MAN.read_text()).items()}
    (ROOT / "tmp").mkdir(exist_ok=True)
    queue = [it for it in items if (export_rec(it["name"]) or {}).get("status") not in ("ok", "refused", "failed")]
    log(f"export: {len(queue)} candidates to export ({[it['name'] for it in queue]})")
    running: list[dict] = []
    fallback_tried: dict[str, set[int]] = {}
    while queue or running:
        while queue and len(running) < a.workers:
            it = queue.pop(0)
            base = ROOT / "exports" / it["name"]
            done = tries(base)
            t = done[-1][0] + 1 if done else 1
            d = inside_root(base / f"try{t}")
            d.mkdir(parents=True, exist_ok=False)
            cmd, code, extra = export_command(it, d)
            env = child_env(None, d)
            env.update(extra)
            lane = len(running) % 2
            prefix = ["nice", "-n", "10", "taskset", "-c", EVAL_CORES[lane]]
            shown = show_command(prefix, cmd, code, env)
            with (d / "export.log").open("a") as stream:
                stream.write(f"[{now()}] export {it['name']} try {t}: {shown}\n")
                stream.flush()
                proc = subprocess.Popen(prefix + cmd, cwd=code, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, start_new_session=True)  # fmt: skip
            write_atomic(d / "start.json", dump({"item": it["name"], "try": t, "pid": proc.pid, "started_at": now(),
                                                 "started_epoch": time.time(), "cwd": str(code), "shown": shown}))
            running.append({"item": it, "try": t, "dir": d, "proc": proc, "start": time.time(), "shown": shown})
            log(f"start export {it['name']} try {t} (pid {proc.pid}, cpus {EVAL_CORES[lane]})")
        time.sleep(2)
        for job in list(running):
            code = job["proc"].poll()
            if code is None:
                if time.time() - job["start"] > EXPORT_TIMEOUT:
                    log.first(("overdue", job["item"]["name"], job["try"]), f"{job['item']['name']} export OVERDUE "
                              f"({time.time() - job['start']:.0f} s); never signalled, still waiting")
                continue
            running.remove(job)
            it, d, t = job["item"], job["dir"], job["try"]
            status, info = check_export(it, d, code)
            rec = {"item": it["name"], "try": t, "status": status, "started_at": (read_json(d / "start.json") or {}).get("started_at"),
                   "finished_at": now(), "seconds": round(time.time() - job["start"], 1),
                   "training_checkpoint": it["training_checkpoint"], "training_sha256": it["training_sha256"],
                   "code_tree": it["code_tree"], "code_commit": it["code_commit"], "exporter": EXPORTER,
                   "tolerance_requested": EXPORT_TOL, "command": job["shown"], **info}  # fmt: skip
            write_atomic(d / "try.json", dump(rec))
            if status == "ok":
                rec["seeds"] = seeds_of(Path(info["file"]))
                write_atomic(ROOT / "exports" / it["name"] / "export.json", dump({**rec, "kept": "kept for good; every evaluation of this candidate evaluates this file"}))
                log(f"{it['name']}: export ok (try {t}, {rec['seconds']:.0f} s): sha256 {info['sha256'][:16]} max_rel_diff "
                    f"{fnum(info.get('max_rel_diff')) or 0.0:.2e}")
            elif t >= EXPORT_TRIES:
                write_atomic(ROOT / "exports" / it["name"] / "export.json", dump(rec))
                log(f"{it['name']}: export {status.upper()} after {t} tries ({info['problems']}): NOT a candidate")
                if it["is_final"] or it.get("final_slot_fallback"):
                    tried = fallback_tried.setdefault(it["arm"], set())
                    tried.add(it["iteration"])
                    if len(tried) > FALLBACK_MAX:
                        log(f"{it['arm']}: {FALLBACK_MAX} final-slot fallbacks failed; the final slot stays empty")
                        continue
                    new = fallback_item(items, it, data["bars"], seqs, tried)
                    if new is not None:
                        new["final_slot_fallback"] = True
                        items.append(new)
                        data["items"] = items
                        data["n_items"] = len(items)
                        write_atomic(CANDS, dump(data))
                        queue.append(new)
                        log(f"{it['arm']}: final slot -> {new['name']} (PREREG v2 8.3); appended to candidates.json")
            else:
                log(f"{it['name']}: export try {t} {status} ({info['problems']}); queued again (max {EXPORT_TRIES} tries)")
                queue.append(it)
    bad = [it["name"] for it in items if (export_rec(it["name"]) or {}).get("status") != "ok"]
    log(f"export: done; not usable: {bad}")
    return 0


# ================================================================================================ the driver
def tags_for(trials: int) -> list[str]:
    return [f"t{k:02d}" for k in range(1, trials + 1)]


class Driver:
    def __init__(self, a: argparse.Namespace, log: Logger):
        self.a, self.log = a, log
        self.gpus = sorted({int(x) for x in re.split(r"[,\s]+", a.gpus.strip()) if x})
        self.keys, self.keys_from = kit_metric_keys()
        self.running: dict[tuple, dict] = {}
        self.orphans: dict[tuple, dict] = {}
        self.items: list[dict] = []
        self.exp: dict[str, dict] = {}
        self.durations: list[float] = []
        self.rest_until: dict[int, float] = {}
        self.only: set[tuple[str, str]] | None = None
        self.summary_proc: subprocess.Popen | None = None
        self.last_summary = 0.0
        self.summary_due = False
        self.final_cache: set[tuple[str, str]] = set()

    def jobs_wanted(self) -> list[tuple[str, dict]]:
        out = []
        for tag in tags_for(self.a.trials):
            for it in self.items:
                out.append((tag, it))
        if not self.a.no_screens:
            out += [("w64", it) for it in self.items if it["screen"]]
        if self.only is not None:
            out = [(tag, it) for tag, it in out if (tag, it["name"]) in self.only]
        return out

    def final(self, tag: str, it: dict) -> bool:
        if (tag, it["name"]) in self.final_cache:
            return True
        rec = run_rec(it["name"], tag)
        if rec is None:
            return False
        if rec.get("status") != "ok":
            self.final_cache.add((tag, it["name"]))
            return True
        js = read_json(rec.get("results_json", "")) or {}
        if js.get("checkpoint_sha256") == self.exp[it["name"]]["sha256"]:
            self.final_cache.add((tag, it["name"]))
            return True
        self.log.first(("resha", it["name"], tag), f"{it['name']} {tag}: run.json ok but results JSON sha256 != pinned file: re-queued")
        return False

    def busy(self, key: tuple) -> bool:
        return key in self.running or key in self.orphans

    @staticmethod
    def retry_due(d: Path) -> bool:
        last = read_json(d / "try.json") or {}
        return time.time() - (parse_time(last.get("finished_at")) or 0.0) >= RETRY_DELAY

    def gpu_rows(self, state: dict, mps: dict, ours: list[dict], peak: int) -> list[dict]:
        busy = {j["gpu"] for j in ours if j.get("gpu") is not None}
        lanes = {0: 0, 1: 0}
        for j in ours:
            if j.get("gpu") is not None:
                lanes[numa(j["gpu"])] += 1
        out = []
        for g in sorted(state["gpus"]):
            st = state["gpus"][g]
            row = {"gpu": g, "uuid": st["uuid"], "free": st["free"], "napps": sum(1 for x in state["apps"] if x["gpu"] == g)}
            rest = self.rest_until.get(g, 0.0) - time.time()
            if g not in ALLOWED_GPUS:
                why = "never (GPU 4)" if g == 4 else "not allowed"
            elif g not in self.gpus:
                why = "not in --gpus"
            elif g in mps:
                why = f"MPS ({mps[g]})"
            elif g in busy:
                why = "one of our evals already runs there"
            elif rest > 0:
                why = f"resting {rest:.0f} s after our last eval"
            elif st["free"] < MIN_FREE_MB + peak:
                why = f"free {st['free']} MiB < {MIN_FREE_MB} + {peak}"
            elif lanes[numa(g)] >= self.a.lane_max:
                why = f"NUMA{numa(g)} eval-core lane already has {self.a.lane_max} of our evals"
            else:
                why = ""
            row.update(usable=not why, why=why or "usable")
            out.append(row)
        return out

    def pick_gpu(self, state: dict, mps: dict, ours: list[dict], peak: int):
        rows = self.gpu_rows(state, mps, ours, peak)
        usable = sorted((r for r in rows if r["usable"]), key=lambda r: (self.rest_until.get(r["gpu"], 0.0), r["napps"], -r["free"], r["gpu"]))
        if not usable:
            return "no usable GPU: " + "; ".join(f"gpu{r['gpu']} {r['why']}" for r in rows if r["gpu"] in self.gpus)
        held = []
        for r in usable:
            got = take_lane(r["gpu"])
            if isinstance(got, str):
                held.append(got)
                continue
            return r["gpu"], got[0], got[1]
        return "every usable GPU's lane is held: " + "; ".join(held)

    def launch(self, tag: str, it: dict, t: int, w: Path, gpu: int, uuid: str, fds: list[int], taken: list[str]) -> None:
        wc = W64 if tag == "w64" else 1
        cmd, code = eval_command(it, w, gpu, wc)
        env = child_env(gpu, w)
        cpus = EVAL_CORES[numa(gpu)]
        prefix = ["nice", "-n", "10", "taskset", "-c", cpus]
        shown = show_command(prefix, cmd, code, env)
        with (w / "eval_all.log").open("a") as stream:
            stream.write(f"[{now()}] {tag} {it['name']} try {t}, gpu {gpu} ({uuid}), cpus {cpus}, world_count {wc}, lane locks {taken}: {shown}\n")
            stream.flush()
            proc = subprocess.Popen(prefix + cmd, cwd=code, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True, pass_fds=tuple(fds))  # fmt: skip
        write_atomic(w / "start.json", dump({"tag": tag, "item": it["name"], "episode": it["episode"], "try": t, "gpu": gpu,
                                             "gpu_uuid": uuid, "host": HOST, "cpus": cpus, "pid": proc.pid, "world_count": wc,
                                             "started_at": now(), "started_epoch": time.time(), "cwd": str(code),
                                             "command": prefix + cmd, "shown": shown, "lane_locks": taken}))  # fmt: skip
        self.running[(tag, it["name"])] = {"tag": tag, "item": it, "try": t, "dir": w, "proc": proc, "start": time.time(),
                                           "gpu": gpu, "fds": fds, "overdue": False}  # fmt: skip
        self.log(f"start {tag} {it['name']} try {t} on gpu {gpu} (pid {proc.pid}; locks {len(taken)})")

    def launch_evals(self) -> None:
        ours = [*self.running.values(), *self.orphans.values()]
        if len(ours) >= self.a.max_parallel:
            return
        ready = []
        for tag, it in self.jobs_wanted():
            key = (tag, it["name"])
            if self.busy(key) or self.final(tag, it):
                continue
            done = tries(ROOT / "runs" / it["name"] / tag)
            if done and not self.retry_due(done[-1][1]):
                continue
            ready.append((tag, it, done))
            if len(ready) >= 2 * self.a.max_parallel:
                break
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
        for tag, it, done in ready:
            if len(ours) >= self.a.max_parallel:
                break
            choice = self.pick_gpu(state, mps, ours, W64_PEAK_MB if tag == "w64" else EVAL_PEAK_MB)
            if isinstance(choice, str):
                self.log.first(("wait", re.sub(r"\d+", "N", choice)), f"{len(ready)} eval(s) ready, waiting: {choice}")
                break
            gpu, fds, taken = choice
            t = done[-1][0] + 1 if done else 1
            w = inside_root(ROOT / "runs" / it["name"] / tag / f"try{t}")
            try:
                (w / f"ep{it['episode']:02d}").mkdir(parents=True, exist_ok=False)
                (w / f"ep{it['episode']:02d}" / f"model_{it['iteration']}.safetensors").symlink_to(self.exp[it["name"]]["file"])
                self.launch(tag, it, t, w, gpu, state["gpus"][gpu]["uuid"], fds, taken)
            except Exception:
                for fd in fds:
                    os.close(fd)
                raise
            ours.append(self.running[(tag, it["name"])])

    def reap(self) -> int:
        finished = 0
        for key, job in list(self.running.items()):
            try:
                pid, status, ru = os.wait4(job["proc"].pid, os.WNOHANG)
            except ChildProcessError:
                pid, status, ru = job["proc"].pid, 0, None
            if pid == 0:
                if not job["overdue"] and time.time() - job["start"] > EVAL_TIMEOUT:
                    job["overdue"] = True
                    self.log(f"{job['tag']} {job['item']['name']}: OVERDUE ({time.time() - job['start']:.0f} s); never signalled, still waiting")
                continue
            code = os.waitstatus_to_exitcode(status) if ru is not None else job["proc"].poll()
            job["proc"].returncode = code
            del self.running[key]
            for fd in job["fds"]:
                os.close(fd)
            self.rest_until[job["gpu"]] = time.time() + self.a.cooldown
            cpu = None if ru is None else round(ru.ru_utime + ru.ru_stime, 1)
            self.finish(job["tag"], job["item"], job["try"], job["dir"], code, "", cpu)
            finished += 1
        return finished

    def finish(self, tag: str, it: dict, t: int, d: Path, code, remark: str, cpu=None) -> None:
        exp = self.exp[it["name"]]
        wc = W64 if tag == "w64" else 1
        problems, info = check_eval(it, exp, d / "results", d / "eval_report.json", code, self.keys, wc)
        start = read_json(d / "start.json") or {}
        seconds = round(time.time() - start["started_epoch"], 1) if start.get("started_epoch") else None
        status = "ok" if not problems else "failed"
        rec = {"tag": tag, "item": it["name"], "arm": it["arm"], "iteration": it["iteration"], "episode": it["episode"],
               "object": it["object"], "sequence_id": it["sequence_id"], "try": t, "status": status, "problems": problems,
               "remark": remark, "gpu": start.get("gpu"), "gpu_uuid": start.get("gpu_uuid"), "host": start.get("host"),
               "cpus": start.get("cpus"), "lane_locks": start.get("lane_locks"), "started_at": start.get("started_at"),
               "finished_at": now(), "seconds": seconds, "cpu_seconds": cpu, "pinned_file": exp["file"],
               "pinned_sha256": exp["sha256"], "work_dir": str(d), "metric_keys": self.keys,
               "wrapper_command": start.get("shown"), "seeds": exp.get("seeds"), "stock_commit": STOCK_COMMIT,
               "driver_sha256": sha256(Path(__file__).resolve()), **info}  # fmt: skip
        write_atomic(d / "try.json", dump(rec))
        base = d.parent
        if status == "ok":
            write_atomic(base / "run.json", dump(rec))
            if seconds and not remark and tag != "w64":
                self.durations.append(seconds)
            m = info["metrics"]
            self.log(f"{tag} {it['name']} ({it['object']}) try {t}, gpu {rec['gpu']}, {seconds or 0:.0f} s, cpu "
                     f"{cpu if cpu is not None else '?'} s: AUC {m['auc']:.4f} SP {m['sp_sr']:.3f} MP {m['mp_sr']:.3f} "
                     f"MPPE {m['mppe']:.2f} cm" + (f" ({remark})" if remark else ""))  # fmt: skip
            self.summary_due = True
        elif len(tries(base)) >= EVAL_TRIES:
            write_atomic(base / "run.json", dump(rec))
            self.log(f"{tag} {it['name']}: FAILED after {t} tries ({problems}); log {d / 'eval_all.log'}")
            self.summary_due = True
        else:
            self.log(f"{tag} {it['name']} try {t} failed ({problems}); retry in {RETRY_DELAY:.0f} s; log {d / 'eval_all.log'}")

    def recover(self) -> None:
        live = live_jobs()
        for tag, it in self.jobs_wanted():
            for t, d in tries(ROOT / "runs" / it["name"] / tag):
                if (d / "try.json").is_file():
                    continue
                key = (tag, it["name"])
                if str(d) in live:
                    start = read_json(d / "start.json") or {}
                    self.orphans[key] = {"tag": tag, "item": it, "try": t, "dir": d, "gpu": start.get("gpu"), "procs": live[str(d)]}
                    self.rest_until[start.get("gpu")] = time.time() + 10 ** 6
                    self.log(f"{tag} {it['name']} try {t}: still running from a previous driver (pids "
                             f"{[p for p, _ in live[str(d)]]}); recorded when it ends, never signalled")
                else:
                    self.finish(tag, it, t, d, None, "recovered after a driver restart (exit code unknown)")

    def check_orphans(self) -> None:
        for key, o in list(self.orphans.items()):
            if any(proc_alive(p, s) for p, s in o["procs"]):
                continue
            del self.orphans[key]
            self.rest_until[o["gpu"]] = time.time() + self.a.cooldown
            self.finish(o["tag"], o["item"], o["try"], o["dir"], None, "finished while no driver was running (exit code unknown)")

    def progress(self) -> dict:
        ev = {"ok": 0, "failed": 0, "pending": 0}
        for tag, it in self.jobs_wanted():
            r = run_rec(it["name"], tag)
            ev["pending" if r is None else ("ok" if r.get("status") == "ok" else "failed")] += 1
        return ev

    def heartbeat(self, phase: str) -> None:
        prog = self.progress()
        lanes = max(1, min(self.a.max_parallel, len(self.gpus), 2 * self.a.lane_max))
        mean_s = sum(self.durations[-50:]) / len(self.durations[-50:]) if self.durations else None
        cycle = (mean_s + self.a.poll + (self.a.cooldown if len(self.gpus) <= self.a.max_parallel else 0.0)) if mean_s else None
        jobs = [{"tag": j["tag"], "item": j["item"]["name"], "try": j["try"], "gpu": j["gpu"], "pid": j["proc"].pid,
                 "elapsed_s": round(time.time() - j["start"]), "overdue": j["overdue"]} for j in self.running.values()]  # fmt: skip
        jobs += [{"tag": o["tag"], "item": o["item"]["name"], "try": o["try"], "gpu": o.get("gpu"),
                  "orphan_pids": [p for p, _ in o["procs"]]} for o in self.orphans.values()]  # fmt: skip
        write_atomic(ROOT / "heartbeat.json", dump({
            "updated_at": now(), "pid": os.getpid(), "phase": phase, "trials": self.a.trials, "evals": prog,
            "running": jobs, "mean_eval_seconds": mean_s,
            "eta_hours": round(cycle * prog["pending"] / lanes / 3600, 2) if cycle else None,
            "stop_file": (ROOT / "STOP").exists()}))  # fmt: skip

    def maybe_summary(self, force: bool = False) -> None:
        if self.summary_proc is not None and self.summary_proc.poll() is None:
            return
        self.summary_proc = None
        if not (force or (self.summary_due and time.time() - self.last_summary >= 180)):
            return
        self.summary_due, self.last_summary = False, time.time()
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", MARK)}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        with inside_root(ROOT / "summary.log").open("a") as stream:
            self.summary_proc = subprocess.Popen(["nice", "-n", "15", str(PY), "-B", str(Path(__file__).resolve()), "summary",
                                                  "--trials", str(self.a.trials)], cwd=ROOT, env=env, stdout=stream,
                                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)

    def run(self) -> int:
        a = self.a
        if 4 in self.gpus or not set(self.gpus) <= set(ALLOWED_GPUS):
            print(f"--gpus {self.gpus}: only {ALLOWED_GPUS} are allowed (never 4)", file=sys.stderr)
            return 2
        lockfd = os.open(inside_root(ROOT / ".driver.lock"), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(f"another t1confirm_driver.py run holds {ROOT / '.driver.lock'}", file=sys.stderr)
            return 1
        if (ROOT / "STOP").exists():
            print(f"{ROOT / 'STOP'} exists: remove it to start", file=sys.stderr)
            return 1
        (ROOT / "tmp").mkdir(exist_ok=True)
        write_atomic(ROOT / "driver.pid", f"{os.getpid()}\n")
        all_items = load_items()
        fatal, info = preflight()
        why = rules_ok(all_items, need_pins=True)
        if why:
            fatal.append(f"rules / pins: {why}")
        for line in info:
            self.log(f"preflight: {line}")
        if fatal:
            for line in fatal:
                self.log(f"preflight FAILED: {line}")
            return 1
        self.items = [it for it in all_items if (export_rec(it["name"]) or {}).get("status") == "ok"]
        self.exp = {it["name"]: export_rec(it["name"]) for it in self.items}
        for it in self.items:
            e = self.exp[it["name"]]
            if sha256(e["file"]) != e["sha256"]:
                self.log(f"preflight FAILED: {it['name']} pinned file {e['file']} changed")
                return 1
        if a.only:
            self.only = set()
            names = {it["name"] for it in self.items}
            for tok in re.split(r"[,\s]+", a.only.strip()):
                m = re.fullmatch(r"(t\d\d|w64):(\S+)", tok)
                if not m or m.group(2) not in names or (m.group(1) != "w64" and not 1 <= int(m.group(1)[1:]) <= a.trials):
                    self.log(f"bad --only token {tok!r}")
                    return 2
                self.only.add((m.group(1), m.group(2)))
        start_listing = integrity()
        self.log(f"start pid {os.getpid()} on {HOST}: {len(self.items)} candidates x {a.trials} evaluations"
                 + ("" if a.no_screens else f" + {sum(1 for it in self.items if it['screen'])} world_count={W64} screens")
                 + (f" (ONLY {sorted(self.only)})" if self.only is not None else "")
                 + f"; gpus {self.gpus}, max parallel {a.max_parallel}, lane max {a.lane_max}, cooldown {a.cooldown:.0f} s; "
                 f"metrics {self.keys} ({self.keys_from}); driver {sha256(Path(__file__).resolve())[:16]}")  # fmt: skip
        self.recover()
        last_beat = 0.0
        while True:
            stopping = (ROOT / "STOP").exists()
            if stopping:
                self.log.first("stop", f"STOP file: no new evaluation; waiting for {len(self.running)} running")
            self.reap()
            self.check_orphans()
            if not stopping:
                self.launch_evals()
            if time.time() - last_beat >= 30:
                self.heartbeat("stopping" if stopping else "running")
                last_beat = time.time()
            self.maybe_summary()
            if not self.running and not self.orphans:
                if stopping:
                    break
                if all(self.final(tag, it) for tag, it in self.jobs_wanted()):
                    self.log("every wanted evaluation is final")
                    break
            time.sleep(a.poll)
        if self.summary_proc is not None:
            self.summary_proc.wait()
            self.summary_proc = None
        self.maybe_summary(force=True)
        if self.summary_proc is not None:
            self.summary_proc.wait()
        diff = integrity_diff(start_listing, integrity())
        write_atomic(ROOT / "integrity" / f"check_{stamp()}.json", dump({"from": start_listing["taken_at"], "to": now(),
                                                                         "n_kit_files": len(start_listing["kit"]),
                                                                         "n_stock_files": len(start_listing["stock"]),
                                                                         "differences": diff}))  # fmt: skip
        self.log("kit and stock runtime unchanged (file list, sizes, mtimes)" if not diff else
                 f"WARNING kit / stock runtime changed during this driver: {diff[:20]}")  # fmt: skip
        self.heartbeat("stopped (STOP file)" if (ROOT / "STOP").exists() else "done")
        try:
            if int((ROOT / "driver.pid").read_text()) == os.getpid():
                (ROOT / "driver.pid").unlink()
        except (OSError, ValueError):
            pass
        os.close(lockfd)
        self.log(f"driver end: {self.progress()}")
        return 0


# ================================================================================================ kit per-world metrics
def kit_env() -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", MARK)}
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               TMPDIR=str(ROOT / "tmp"))
    return env


def kit_metrics(out_json: Path, obj: str, parquets: list[str], log) -> dict | None:
    """screen_score.py (kit maths, per world) for these parquets; cached while the parquet sha256 values are the same."""
    want = [sha256(p) for p in parquets]
    old = read_json(out_json)
    if old and [r.get("parquet_sha256") for r in old.get("results", [])] == want and old.get("scorer_sha256") == sha256(SCREEN_SCORER):
        return old
    cmd = ["nice", "-n", "15", str(PACK_PYTHON), "-B", str(SCREEN_SCORER), "--object", obj, "--out", str(inside_root(out_json)), *parquets]
    p = subprocess.run(cmd, cwd=ROOT, env=kit_env(), capture_output=True, text=True, timeout=SCORE_TIMEOUT)
    write_atomic(out_json.with_suffix(".log"), f"$ {shlex.join(cmd)}\n{p.stdout}\n{p.stderr}")
    if p.returncode != 0:
        log(f"screen_score failed for {out_json} (exit {p.returncode}); see {out_json.with_suffix('.log')}")
        return None
    return read_json(out_json)


# ================================================================================================ summary + verdicts
def cand_verdict(it: dict, oks: list[dict], n_wanted: int, n_failed: int, screen: dict | None) -> dict:
    bar = it["bar"]
    v: dict = {"name": it["name"], "arm": it["arm"], "item": bar["item"], "episode": it["episode"], "iteration": it["iteration"],
               "n_ok": len(oks), "n_failed": n_failed, "n_wanted": n_wanted, "bar_json_auc": bar["json_auc_mean"],
               "needed": bar["needed"], "floor": bar["floor"], "bar_json_mppe": bar["json_mppe_mean"]}  # fmt: skip
    aucs = [r["metrics"]["auc"] for r in oks]
    mppes = [r["metrics"]["mppe"] for r in oks]
    for k, s in mstats(aucs).items():
        v[f"auc_{k}"] = s
    for k, s in mstats(mppes).items():
        v[f"mppe_{k}"] = s
    v["sp_pass"] = sum(1 for r in oks if r["metrics"]["sp_sr"] >= 0.5)
    v["mp_pass"] = sum(1 for r in oks if r["metrics"]["mp_sr"] >= 0.5)
    v["seed_signatures"] = sorted({(r.get("seeds") or {}).get("seed_signature", "") for r in oks})
    complete = len(oks) + n_failed >= n_wanted
    need_min = 5 if bar["item"] in ("ep01", "ep25") else 3
    if not complete:
        v.update(verdict="PENDING", why=f"{len(oks)} ok + {n_failed} failed of {n_wanted}")
        return v
    if len(oks) < need_min:
        v.update(verdict="FAIL", why=f"only {len(oks)} ok evaluations (< {need_min}, PREREG 8.3)")
        return v
    gain = v["auc_mean"] - bar["json_auc_mean"]
    cap = bar["mppe_cap_big_gain"] if gain >= BIG_GAIN else bar["mppe_cap"]
    v.update(gain_vs_bar=gain, mppe_cap_applied=cap)
    crit = {"a": v["auc_mean"] >= bar["needed"], "b": v["sp_pass"] == len(oks), "c": v["mp_pass"] == len(oks),
            "d": v["mppe_mean"] <= cap, "e": v["auc_min"] >= bar["floor"]}  # fmt: skip
    if it["screen"]:
        if screen is None or screen.get("status") != "ok":
            crit["f"] = None
        else:
            crit["f"] = bool(screen["world0_ok"]) and screen["collapses"] == 0
    v["criteria"] = crit
    fails = [k for k, ok in crit.items() if ok is False]
    ep28_exception = (bar["item"] == "ep28alt" and not crit["c"] and all(crit[k] for k in ("a", "b", "d", "e")))
    if ep28_exception:
        fails = [k for k in fails if k != "c"]
    if fails:
        v.update(verdict="FAIL", why="fails " + ",".join(f"({k})" for k in fails))
    elif any(ok is None for ok in crit.values()):
        v.update(verdict="PENDING_SCREEN", why="(a)-(e) " + ("pass (ep28 MP exception)" if ep28_exception else "pass")
                 + "; collapse screen not finished")
    elif ep28_exception:
        v.update(verdict="QUALIFIES_EP28_MP_EXCEPTION",
                 why=f"(a),(b),(d),(e),(f) pass; MP {v['mp_pass']}/{len(oks)}: kept only with MP >= 16/20 in its 20-trial confirmation")
    else:
        v.update(verdict="QUALIFIES", why="(a)-(e)" + ("+(f)" if it["screen"] else "") + " pass")
    return v


def cmd_summary(a) -> int:
    log = Logger(None)
    lockfd = os.open(inside_root(ROOT / ".summary.lock"), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another summary is running")
        return 0
    items = load_items()
    exps = {it["name"]: export_rec(it["name"]) or {} for it in items}
    tags = tags_for(a.trials)
    recs: dict[tuple[str, str], dict] = {}
    for it in items:
        for tag in tags + ["w64"]:
            r = run_rec(it["name"], tag)
            if r is not None:
                recs[(tag, it["name"])] = r
    # ---- per evaluation + seeds
    rows, seed_rows = [], []
    for (tag, name), r in sorted(recs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        m = r.get("metrics") or {}
        pm = r.get("parquet_meta") or {}
        s = r.get("seeds") or exps[name].get("seeds") or {}
        row = {"item": name, "tag": tag, "status": r.get("status"), "world_count": r.get("world_count"), "auc": m.get("auc"),
               "sp_sr": m.get("sp_sr"), "mp_sr": m.get("mp_sr"), "mppe": m.get("mppe"), "gpu": r.get("gpu"),
               "gpu_uuid": r.get("gpu_uuid"), "start": r.get("started_at"), "end": r.get("finished_at"), "seconds": r.get("seconds"),
               "eval_file_sha256": r.get("pinned_sha256"), "json_checkpoint_sha256": r.get("checkpoint_sha256"),
               "parquet_checkpoint_sha256": pm.get("checkpoint_sha256"), "eval_script_sha256(results)": pm.get("eval_script_sha256"),
               "eval_code_sha256(results)": pm.get("eval_code_sha256"), "world_count(results)": pm.get("world_count"),
               "json_sha256": r.get("json_sha256"), "parquet_sha256": r.get("parquet_sha256"),
               "kit_commit": r.get("stock_commit") or STOCK_COMMIT, "runner_command": r.get("runner_command"),
               "wrapper_command": r.get("wrapper_command"), "problems": "; ".join(r.get("problems") or [])}  # fmt: skip
        rows.append(row)
        sr = {k: row[k] for k in ("item", "tag", "status", "gpu", "gpu_uuid", "start", "end", "eval_file_sha256",
                                  "json_checkpoint_sha256", "parquet_checkpoint_sha256", "parquet_sha256", "kit_commit",
                                  "runner_command")}  # fmt: skip
        sr["evaluator_json_auc"] = row["auc"]
        for c in SEED_COLS:
            sr[c] = s.get(c)
        seed_rows.append(sr)
    write_table(ROOT / "per_eval.csv", rows, list(rows[0].keys()) if rows else ["item"])
    write_table(ROOT / "seeds.tsv", seed_rows, ["item", "tag", "status"] + SEED_COLS + ["gpu", "gpu_uuid", "start", "end",
                "kit_commit", "eval_file_sha256", "json_checkpoint_sha256", "parquet_checkpoint_sha256", "parquet_sha256",
                "evaluator_json_auc", "runner_command"], delimiter="\t")  # fmt: skip
    # ---- screens (kit per-world metrics) + world-0 check
    screens: dict[str, dict] = {}
    for it in items:
        if not it["screen"] or exps[it["name"]].get("status") != "ok":
            continue
        name = it["name"]
        r = recs.get(("w64", name))
        sc: dict = {"item": name, "status": "pending" if r is None else r.get("status")}
        if r is not None and r.get("status") == "ok":
            pw = kit_metrics(ROOT / "runs" / name / "w64" / "perworld.json", it["object"], [r["results_parquet"]], log)
            trials = [recs[(t, name)] for t in tags if recs.get((t, name), {}).get("status") == "ok"]
            kt = kit_metrics(ROOT / "runs" / name / "kit_trials.json", it["object"], [x["results_parquet"] for x in trials], log) if trials else None
            if pw is None:
                sc["status"] = "scorer_failed"
            else:
                res = pw["results"][0]
                worlds = res["per_world"]
                floor = it["bar"]["floor"]
                aucs = [w["auc"] for w in worlds]
                ktr = [x["per_world"][0]["auc"] for x in (kt or {}).get("results", [])]
                w0 = worlds[0]["auc"] if worlds and worlds[0]["world"] == 0 else None
                ok0 = (w0 is not None and len(ktr) >= 3 and min(ktr) - WORLD0_TOL <= w0 <= max(ktr) + WORLD0_TOL)
                sc.update(world_count=res["world_count"], collapses=sum(1 for x in aucs if x < floor),
                          collapse_worlds=[w["world"] for w in worlds if w["auc"] < floor], floor=floor,
                          world_auc_min=min(aucs), world_auc_mean=sum(aucs) / len(aucs), world_auc_max=max(aucs),
                          worlds_sp_fail=sum(1 for w in worlds if w["sp_sr"] < 0.5),
                          worlds_mp_fail=sum(1 for w in worlds if w["mp_sr"] < 0.5),
                          world_mppe_mean=sum(w["mppe_cm"] for w in worlds) / len(worlds),
                          pooled_kit_auc=res["pooled_auc"], json_auc=(r.get("metrics") or {}).get("auc"),
                          reference_spread=res["reference_spread_over_worlds"], world0_auc=w0,
                          trials_kit_auc_min=min(ktr) if ktr else None, trials_kit_auc_max=max(ktr) if ktr else None,
                          trials_kit_auc_mean=sum(ktr) / len(ktr) if ktr else None, n_trials_kit=len(ktr),
                          world0_ok=ok0, seconds=r.get("seconds"), gpu=r.get("gpu"))  # fmt: skip
                if len(ktr) < 3:
                    sc["status"] = "waiting_for_trials"
        screens[name] = sc
    write_table(ROOT / "screens.csv", list(screens.values()),
                ["item", "status", "world_count", "collapses", "collapse_worlds", "floor", "world_auc_min", "world_auc_mean",
                 "world_auc_max", "worlds_sp_fail", "worlds_mp_fail", "world_mppe_mean", "pooled_kit_auc", "json_auc",
                 "world0_auc", "trials_kit_auc_min", "trials_kit_auc_mean", "trials_kit_auc_max", "n_trials_kit", "world0_ok",
                 "reference_spread", "seconds", "gpu"])  # fmt: skip
    # ---- candidates
    verdicts = []
    for it in items:
        name = it["name"]
        e = exps[name]
        if e.get("status") != "ok":
            verdicts.append({"name": name, "arm": it["arm"], "item": it["bar_item"], "episode": it["episode"],
                             "iteration": it["iteration"], "verdict": "NOT_A_CANDIDATE" if e.get("status") else "NOT_EXPORTED",
                             "why": f"export {e.get('status')}: {e.get('problems')}"})
            continue
        oks = [recs[(t, name)] for t in tags if recs.get((t, name), {}).get("status") == "ok"]
        nfail = sum(1 for t in tags if (t, name) in recs and recs[(t, name)].get("status") != "ok")
        v = cand_verdict(it, oks, len(tags), nfail, screens.get(name))
        v["export_sha256"] = e.get("sha256")
        v["export_max_rel_diff"] = e.get("max_rel_diff")
        v["nominated"] = "; ".join(it["nominated"])
        v["wave"] = it["wave"]
        verdicts.append(v)
    cols = ["name", "arm", "wave", "item", "episode", "iteration", "nominated", "verdict", "why", "n_ok", "n_failed", "auc_mean",
            "auc_sd", "auc_min", "auc_max", "needed", "bar_json_auc", "gain_vs_bar", "floor", "sp_pass", "mp_pass", "mppe_mean",
            "mppe_sd", "mppe_cap_applied", "bar_json_mppe", "criteria", "seed_signatures", "export_sha256", "export_max_rel_diff"]
    write_table(ROOT / "candidates_summary.csv", [{**v, "criteria": json.dumps(v.get("criteria")), "seed_signatures": " | ".join(v.get("seed_signatures") or [])} for v in verdicts], cols)
    # ---- items
    by_item: dict[str, list[dict]] = {}
    for v in verdicts:
        by_item.setdefault(v["item"], []).append(v)
    item_out = {}
    for item, vs in by_item.items():
        pending = [v["name"] for v in vs if str(v["verdict"]).startswith("PENDING")]
        quals = [v for v in vs if v["verdict"] in ("QUALIFIES", "QUALIFIES_EP28_MP_EXCEPTION")]
        win = min(quals, key=lambda v: (-v["auc_mean"], v["mppe_mean"], v["iteration"], ARM_RANK[v["arm"]])) if quals else None
        item_out[item] = {"item": item, "final": not pending, "pending": pending, "qualifiers": [v["name"] for v in quals],
                          "winner": win["name"] if win else None, "winner_verdict": win["verdict"] if win else None,
                          "winner_auc_mean": win["auc_mean"] if win else None, "winner_mppe_mean": win["mppe_mean"] if win else None,
                          "changed": bool(win) and not pending,
                          "candidates": {v["name"]: v["verdict"] for v in vs}}  # fmt: skip
    p30 = []
    for v in verdicts:
        if v["arm"].endswith("P30") and v.get("auc_mean") is not None:
            it = next(x for x in items if x["name"] == v["name"])
            b = it["bar"]
            p30.append({"name": v["name"], "mppe_drop_frac": 1 - v["mppe_mean"] / b["json_mppe_mean"],
                        "auc_minus_bar": v["auc_mean"] - b["json_auc_mean"],
                        "gate_pass": (1 - v["mppe_mean"] / b["json_mppe_mean"] >= 0.15 and v["auc_mean"] - b["json_auc_mean"] >= -0.002
                                      and v["sp_pass"] == v["n_ok"] and v["mp_pass"] == v["n_ok"])})  # fmt: skip
    out = {"generated_at": now(), "trials": a.trials, "rules_sha256": sha256(RULES) if RULES.is_file() else None,
           "items": item_out, "candidates": verdicts, "screens": screens, "p30_report_only": p30,
           "changed_items": sorted(k for k, x in item_out.items() if x["changed"]),
           "all_final": all(x["final"] for x in item_out.values()),
           "n_evals_ok": sum(1 for r in recs.values() if r.get("status") == "ok"),
           "n_evals_failed": sum(1 for r in recs.values() if r.get("status") != "ok"),
           "seed_signatures": sorted({(r.get("seeds") or {}).get("seed_signature", "") for r in recs.values()})}  # fmt: skip
    write_atomic(ROOT / "verdicts.json", dump(out))
    write_summary_md(out, items)
    os.close(lockfd)
    print(f"summary: {len(recs)} verdicts; changed {out['changed_items']}; all final {out['all_final']}")
    return 0


def fv(value, spec: str = ".4f") -> str:
    return "" if value in (None, "") else format(float(value), spec)


def write_summary_md(out: dict, items: list[dict]) -> None:
    md = [f"# T1improve confirmation: {out['trials']} official evaluations per candidate", "",
          f"Generated {out['generated_at']} by `t1confirm_driver.py summary`. Rules: `CONFIRM_RULES.md` (sha256 "
          f"`{out['rules_sha256']}`). {out['n_evals_ok']} ok and {out['n_evals_failed']} failed evaluations so far. "
          "Evaluator-JSON metrics; bars from the t1best20 per_episode.csv JSON columns (PREREG v2 section 2). "
          "Nothing here is packed, selected for upload or uploaded.", "", "## Seeds", "",
          "The official evaluator has no selectable seed (harness forces env.reset.seed = 42; the other seeds come from "
          "the evaluated file's own config). Distinct signatures over all evaluations:", ""]
    md += [f"- `{s}`" for s in out["seed_signatures"] if s] or ["- (none yet)"]
    md += ["", "All evaluations of one candidate share its file's signature, so the spread between them is GPU "
           "floating-point non-determinism, not a seed. `ppo=43` marks the seed-43 training arms (inert at evaluation).", "",
           "## Items", "", "| item | final | winner | winner AUC | winner MPPE | qualifiers | changed |", "|---|---|---|---:|---:|---|---|"]
    for k in sorted(out["items"]):
        x = out["items"][k]
        md.append(f"| {k} | {x['final']} | {x['winner'] or '-'} | {fv(x['winner_auc_mean'])} | {fv(x['winner_mppe_mean'], '.3f')} | "
                  f"{', '.join(x['qualifiers']) or '-'} | {x['changed']} |")
    md += ["", "## Candidates", "", "| candidate | n ok | AUC mean | sd | min | needed | SP | MP | MPPE | cap | verdict | why |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for v in out["candidates"]:
        md.append(f"| {v['name']} | {v.get('n_ok', '')} | {fv(v.get('auc_mean'))} | {fv(v.get('auc_sd'))} | {fv(v.get('auc_min'))} | "
                  f"{fv(v.get('needed'))} | {v.get('sp_pass', '')} | {v.get('mp_pass', '')} | {fv(v.get('mppe_mean'), '.3f')} | "
                  f"{fv(v.get('mppe_cap_applied'), '.3f')} | {v['verdict']} | {v.get('why', '')} |")
    md += ["", "## Collapse screens (world_count=64; kit per-world AUC)", "",
           "| candidate | status | collapses | min world AUC | mean | floor | world 0 | trials kit AUC range | world-0 ok | MP-fail worlds |",
           "|---|---|---:|---:|---:|---:|---:|---|---|---:|"]
    for s in out["screens"].values():
        md.append(f"| {s['item']} | {s['status']} | {s.get('collapses', '')} | {fv(s.get('world_auc_min'))} | {fv(s.get('world_auc_mean'))} | "
                  f"{fv(s.get('floor'))} | {fv(s.get('world0_auc'))} | {fv(s.get('trials_kit_auc_min'))}-{fv(s.get('trials_kit_auc_max'))} | "
                  f"{s.get('world0_ok', '')} | {s.get('worlds_mp_fail', '')} |")
    if out["p30_report_only"]:
        md += ["", "## P30 pilot gate (ADD1 section 6, report only)", ""]
        md += [f"- {p['name']}: MPPE {100 * p['mppe_drop_frac']:.1f}% below the incumbent's 20-trial JSON mean, AUC "
               f"{p['auc_minus_bar']:+.4f} vs the bar; gate {'PASS' if p['gate_pass'] else 'fail'} (changes no decision)"
               for p in out["p30_report_only"]]
    md += ["", f"Changed items: {', '.join(out['changed_items']) or 'none'}; all items final: {out['all_final']}."]
    write_atomic(ROOT / "SUMMARY.md", "\n".join(md) + "\n")


# ================================================================================================ status
def cmd_status(a) -> int:
    hb = read_json(ROOT / "heartbeat.json") or {}
    pid = None
    try:
        pid = int((ROOT / "driver.pid").read_text())
    except (OSError, ValueError):
        pass
    alive = pid is not None and "t1confirm_driver.py" in " ".join(proc_cmdline(pid)) and "run" in proc_cmdline(pid)
    print(f"driver: {'RUNNING pid ' + str(pid) if alive else 'not running'}; STOP file {'present' if (ROOT / 'STOP').exists() else 'absent'}; "
          f"heartbeat {hb.get('updated_at')} phase {hb.get('phase')} evals {hb.get('evals')} eta {hb.get('eta_hours')} h")
    for j in hb.get("running", []):
        print(f"  running: {j['tag']} {j['item']} try {j['try']} gpu {j.get('gpu')} {j.get('elapsed_s', '?')} s")
    try:
        items = load_items()
    except SystemExit as error:
        print(error)
        return 0
    tags = tags_for(a.trials) + ["w64"]
    for it in items:
        cells, aucs = [], []
        for tag in tags:
            if tag == "w64" and not it["screen"]:
                continue
            r = run_rec(it["name"], tag)
            cells.append("." if r is None else ("o" if r.get("status") == "ok" else "X"))
            if r and r.get("status") == "ok" and tag != "w64":
                aucs.append(r["metrics"]["auc"])
        e = export_rec(it["name"]) or {}
        print(f"  {it['name']:22s} export {e.get('status', '-'):7s} {''.join(cells):12s} "
              + (f"n {len(aucs)} mean AUC {sum(aucs) / len(aucs):.4f} (needed {it['bar']['needed']:.4f})" if aucs else ""))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--trials", type=int, default=10)
    r.add_argument("--gpus", default="0,1,2,3,5,6,7", help="candidate GPUs (never 4)")
    r.add_argument("--max-parallel", type=int, default=4)
    r.add_argument("--lane-max", type=int, default=2)
    r.add_argument("--cooldown", type=float, default=35.0)
    r.add_argument("--only", default="")
    r.add_argument("--no-screens", action="store_true")
    r.add_argument("--poll", type=float, default=5.0)
    e = sub.add_parser("export")
    e.add_argument("--workers", type=int, default=2)
    s = sub.add_parser("summary")
    s.add_argument("--trials", type=int, default=10)
    st = sub.add_parser("status")
    st.add_argument("--trials", type=int, default=10)
    sub.add_parser("build")
    sub.add_parser("preflight")
    a = p.parse_args()
    if a.cmd == "run":
        if a.trials < 1 or a.max_parallel < 1 or a.lane_max < 1 or a.cooldown < 0:
            p.error("bad --trials / --max-parallel / --lane-max / --cooldown")
        return Driver(a, Logger(ROOT / "driver.log")).run()
    if a.cmd == "preflight":
        fatal, info = preflight()
        for line in info:
            print(f"ok   {line}")
        for line in fatal:
            print(f"FAIL {line}")
        return 1 if fatal else 0
    return {"build": cmd_build, "export": cmd_export, "status": cmd_status, "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
