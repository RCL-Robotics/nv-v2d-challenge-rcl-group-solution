#!/usr/bin/env python3
"""t3final_driver.py: the Tier-3 FINAL pipeline of the f0 test (arm C3r2_tier3_f0): progressive 5x official evaluation of
every clip's post-step file against its own input, a pre-registered noisy-only run selection, and the kit pack.

ROOT = fc_runs/submissions/t3_f0_final. Everything is written under ROOT. Only read: tmp_analysis/tier3/{f0,opus_easy}
(post.json, eval_report.json, export_report.json, the post step's own-input results, the f0 retarget lock, CRASHED
markers), fc_runs/C3r2_tier3_{f0,opus}/epNN (final.json, the training checkpoint), the two Tier-3 data roots, the kit,
the stock runtime 8c778f9 and code/fc_r2 @ 61fa5d3. No clean-reference file is ever read for a decision: every decision
read goes through guard(), which refuses any path naming clean / LOCAL_NOT_FOR_UPLOAD / tier_1_multiview_caption /
track_2/baseline. Clean numbers exist only in *_LOCAL_NOT_FOR_UPLOAD files written by `pack` (report only, read by
nothing here). No process is signalled except this driver's own export/eval/pack sessions past their timeout. Never
GPU 4, never an MPS GPU. Python children run with PYTHONDONTWRITEBYTECODE=1 (and -B / -I where we start Python).

ITEMS (`resolve`; also every --resolve-every s inside `run`), per episode, PREREG_f0 "Use in a submission":
  f0       the t3f0 post step is terminal with an evaluated checkpoint (post.json status done / done_score_failed,
           eval_status evaluated, eval_exit 0). Eval file = post.json evaluated_checkpoint, i.e. the file the post step
           evaluated. Verified: sha256 == post.json == eval_report.json row == the post step's results JSON == the
           export_report destination; export ok + stock-verified at the post tolerance (1e-4, or 2e-4 if recorded);
           export source == the run's final.json checkpoint (sha256); reference parquet == the retarget-locked f0
           parquet (path + sha256); frame reference = frame_references_f0/episode_NNNNNN.npz.
  opus_v2  only if the f0 clip has no evaluated checkpoint (f0 post status export_refused / failed, or CRASHED_epNN)
           AND the opus_v2 post step evaluated one: the same checks on tmp_analysis/tier3/opus_easy, its own opus input,
           manifest and frame_references_opus.
  none     neither: the episode has no policy. No complete FINAL file can be packed (user decision, disclosed).
  blocked  a terminal post step whose files fail verification: logged, never frozen, needs a human.
  An item is frozen once (items/epNN.json, write-once) and never re-resolved.
RUNS (`run`): run k = 1..5 of an item = one official evaluation of the pinned eval file against its own input:
  code/fc_r2 scripts/track2/eval_all.py (657295921e82) --export no --world-count 1 --manifest <the item's own manifest>
  -> kit examples/run_policy_evaluation.py -> kit flash_chord/scripts/evaluate_policy.py (160d24f5) on the untouched
  stock runtime 8c778f9 (frame 0, explicit reset, no help). Work dir runs/epNN/rK/tryT (results JSON + parquet kept;
  epNN/model_<it>.safetensors is a symlink to the pinned file); verdict runs/epNN/rK/run.json. Order: item-major
  (episode index, then run). A completed evaluation is never re-run whatever its score; a technical failure is
  retried (<= 3 tries, 5 min apart). An eval is ok only if: exit 0 and eval_all says "evaluated"; JSON + parquet exist;
  JSON + parquet checkpoint_sha256 == the pinned sha256 and the JSON names the pinned file; the JSON's reference parquet
  is the item's own input (path + sha256); start_frame 0, world_count 1, reset explicit, no non-finite world; parquet
  eval_script / eval_code digests == the common build; environment_valid everywhere; the pinned file unchanged.
SELECTION (selection/epNN.json, write-once, as soon as the clip's 5 runs are final): among its ok runs, the run whose
  own-input ADD-AUC (results JSON metrics.add_auc) is closest to the mean over those ok runs; ties -> lower run index.
  No ok run -> run 0 = the post step's own official evaluation of the same file.
PACK (`pack`; spawned by `run` once all 30 episodes are settled): FINAL = per episode the selected run; T1..T5 = run k of
  every episode (report only). Kit packer --track track_2_tier3 --frame-references <byte copies of each episode's own
  frame reference>, placeholder commit URL; uploader validate_submission() (offline); official metric_code vs the clean
  reference. Packer logs and metric_code output: *_LOCAL_NOT_FOR_UPLOAD only.
LANES: GPUs 0,1,2,3,5,6,7 (never 4). Each eval holds, for its whole life, our lock ROOT/evallock/gpu<g>.lock plus every
  existing shared eval-lane lock of that GPU (rolling-start, tier2v2, t3f0, opus, k3*, t1_best20; opened read-only,
  never created; all-or-nothing, never blocking). No MPS GPU; >= 4096+1536 MiB free; at most --lane-max evaluations of
  ANYBODY per NUMA eval-core lane (evaluate_policy.py processes pinned to that lane's cores are counted); <= --max-parallel
  of ours; after each of our evals the GPU rests --cooldown s. Evals run as `nice -n 10 taskset -c <NUMA eval cores>`.

  python -B t3final_driver.py run [--runs 5] [--gpus 0,1,2,3,5,6,7] [--max-parallel 3] [--lane-max 2] [--cooldown 35]
                                  [--poll 5] [--resolve-every 60] [--no-pack]
  python -B t3final_driver.py resolve [--dry] | status | summary | selftest | dryrun --ep 0
  python -B t3final_driver.py pack [--partial] [--files FINAL,T1,...] | repack --commit-url URL
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
import shutil  # noqa: E402
import signal  # noqa: E402
import socket  # noqa: E402
import statistics  # noqa: E402
import struct  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from fractions import Fraction  # noqa: E402
from pathlib import Path  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
ROOT = ME / "fc_runs/submissions/t3_f0_final"
FC = ME / "fc_runs"
PREREG, PREREG_SHA = ROOT / "PREREG.md", ROOT / "PREREG.sha256"
KIT = ME / "v2d_submission_kit_src/v2d_submission_kit"
STOCK_REPO = ME / "code/video_to_data_8c778f9"
STOCK = STOCK_REPO / "robotic_grounding/flash_chord"
STOCK_COMMIT = "8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef"
DATASET = ME / "data/video_to_data_challenge"
PACK_PYTHON = ME / "conda_envs/v2dkit/bin/python"
PY = ME / "conda_envs/flashchord/bin/python"
KITHELPER = ROOT / "t3final_kit.py"
RUNNER = KIT / "examples/run_policy_evaluation.py"
EVAL_SCRIPT = KIT / "flash_chord/scripts/evaluate_policy.py"
PACKER = KIT / "tools/pack_submission.py"
UPLOADER = KIT / "submit/v2d_submit.py"
CODE = ME / "code/fc_r2"
CODE_COMMIT = "61fa5d31ff8054bf1e5cf5375ef0957e54d2729b"
EVAL_ALL_SHA = "657295921e820a464160039ad9a65ba92a683108a99dcb197290ae4147363197"
EVAL_SCRIPT_SHA = "160d24f5fbeb70ccd569ca4192056f1892b2a9f2999066db9d8279461648fdf1"
EVAL_CODE_SHA = "b1b3ebe3ae4629d1687a83f420149e3f6fdfc154d70acad55c12fc8fa69811a1"
RUNNER_SHA = "0bcc41f5de246a51f194409b3e5292e04109012cb22a433a8f7ed477df04e55d"
PACKER_SHA = "47a4589574ff8284ba4376ef31ac5548c8fe87ead753be2d3b43d6c87041e5be"
UPLOADER_SHA = "fe5cb1dc86d12abd01eab9dd5dd98b51569b5d8513a2d00d4e79da42a5f11718"
METRIC_SHA = {"AUC": "58b437697c32c86497ff5a306fe586014c3429b0d2c18e1d3094a6f503479339",
              "SP-SR": "d206042872acd1c1de6120f96966272495fbbd71a50315f49f5192fea1aaa5bf",
              "MP-SR": "a1d72216eb7745a13ed7db6a55b9efb31492b35149113181a6c42ffc231fff11",
              "MPPE": "41815bc1e8870fb377036750976dcf135818e029f06096ccef5177090fe5e033"}  # fmt: skip
TRACK = "track_2_tier3"
COMMIT_PLACEHOLDER = "https://github.com/local-placeholder/not-uploaded/commit/0000000000000000000000000000000000000000"
COMMIT_URL_RE = re.compile(r"^https://github\.com/[^/\s]+/[^/\s]+/commit/[0-9a-f]{40}$")
NRUNS = 5
EPISODES = tuple(range(30))
SRC = {
    "f0": {
        "label": "f0 (Han Zhou's frame-0 release, tier3_frame0_supports_20261024)",
        "analysis": ME / "tmp_analysis/tier3/f0", "runs": FC / "C3r2_tier3_f0",
        "data": ME / "data/human_motion_data_tier3_f0",
        "frame_refs": ME / "data/human_motion_data_tier3_f0/frame_references_f0",
        "manifest": ME / "data/human_motion_data_tier3_f0/whole_body/soma/track2_manifest.json",
        "manifest_sha": "d496584c98cb4e811c41a12b0afa5992ff02a64dd27cc8e73013304b10d364b5",
        "ready_sha": "ee58a62deb3a0bb06248d221e1184de751f32c171410fa2616f057c4b3b2e50b",
        "prereg": ME / "tmp_analysis/tier3/f0/PREREG_f0.md",
        "prereg_sha": "82702d85fd119d0ad5165dd8c2d890c87e700906ebffcdb21c56c024e1e8eeec",
    },
    "opus_v2": {
        "label": "opus_v2 (Han Zhou's tier3_opus_20261021; PREREG_f0 fallback only)",
        "analysis": ME / "tmp_analysis/tier3/opus_easy", "runs": FC / "C3r2_tier3_opus",
        "data": ME / "data/human_motion_data_tier3_opus",
        "frame_refs": ME / "data/human_motion_data_tier3_opus/frame_references_opus",
        "manifest": ME / "data/human_motion_data_tier3_opus/whole_body/soma/track2_manifest.json",
        "manifest_sha": "fac5a9855dd4074846c6117a3fdd29fd76311bd00283d9937c912d1add261d3e",
        "ready_sha": "9c4cb2a96e18cdaa9b8e9d68dd7779107660231d760f2a99a515dc2fea0fff09",
        "prereg": ME / "tmp_analysis/tier3/opus_easy/PREREG.md",
        "prereg_sha": "c7a19218eed2f7548e4c9b04b9d76649adcaf58223cd37a9a95ccbde7b86dfa4",
    },
}  # fmt: skip
F0_LOCK = ME / "tmp_analysis/tier3/f0/retarget/parquet_lock.json"
F0_LOCK_SHA = "bdb83b0ee34f05a8408a871cba3d8ca2bb4cef6ec0e67ca435a57232b3e0b0d7"
F0_FILES_SHA = "f9c94e8550e03223c96e101e155038537efeedda3b5213d96cecc4dcbafaba49"  # data root FILES.sha256 (covers frame refs)
MITIGATED = (1, 2, 3, 15, 17, 19, 20, 28)  # PREREG_f0: termination.palm_position_threshold 0.25 (report check only)
SUPPORT_EPISODES = (1, 2, 3, 6, 7, 9, 11, 13, 16, 17, 20, 25, 27, 28)  # scene.support true (report check only)
EVALUATED = ("done", "done_score_failed")
NOT_EVALUATED = ("export_refused", "failed")
EXPORT_TOLS = (1e-4, 2e-4)
EVAL_TIMEOUT, PACK_TIMEOUT = 2400.0, 3600.0
KILL_GRACE, RETRY_DELAY = 60.0, 300.0
EVAL_TRIES = 3
MIN_FREE_MB, EVAL_PEAK_MB, MIN_FREE_PIDS = 4096, 1536, 2000
EVAL_CORES = ("48-51,152-155", "100-103,204-207")  # the eval lanes of scripts/runin_helper.py (t1best20 / t3f0 post)
ALLOWED_GPUS = (0, 1, 2, 3, 5, 6, 7)  # never GPU 4 (Zuxing's)
OWN_LOCKDIR = ROOT / "evallock"
SHARED_LOCKDIRS = (FC / "queue_runin/evallock", FC / "queue_tier2v2/evallock", ME / "tmp_analysis/tier3/f0/evallock",
                   ME / "tmp_analysis/tier3/opus_easy/evallock", ME / "tmp_analysis/tier2/k3ca/train/evallock",
                   ME / "tmp_analysis/tier2/k3fix_pilot/evallock", ME / "tmp_analysis/tier2/k3full/evallock",
                   ME / "tmp_analysis/tier2/k3easy/evallock",
                   FC / "submissions/t1_best20_20261026/evallock")  # gpu<g>.lock opened read-only if present, never created
METRICS = ("auc", "sp_sr", "mp_sr", "mppe")
FALLBACK_KEYS = {"auc": "add_auc", "sp_sr": "spider_sr_uncentered", "mp_sr": "maniptrans_sr", "mppe": "mean_per_point_error_cm"}
HOST = socket.gethostname()
MARK = "T3FINAL_JOB"
FILES_ALL = ("FINAL",) + tuple(f"T{k}" for k in range(1, NRUNS + 1))
FORBIDDEN = ("clean", "local_not_for_upload", "tier_1_multiview_caption", "/track_2/baseline")
SYNTH_VALIDATE_CSV = FC / "submissions/prep_20261026/tier2_raw/C3r2_tier2_raw_closest_41403d3.csv"  # selftest only


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


def guard(path) -> Path:
    """Tier-3 rule: no clean-reference / LOCAL file is ever read for a decision."""
    s = str(path).lower()
    if any(x in s for x in FORBIDDEN):
        raise SystemExit(f"guard: refusing to read {path} (clean / LOCAL file; Tier-3 decisions are noisy-only)")
    return Path(path)


def read_json(file) -> dict | None:
    try:
        data = json.loads(guard(file).read_text())
    except (OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def realpath(p) -> str:
    return os.path.realpath(str(p)) if p else ""


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


def write_once(file: Path, text: str) -> bool:
    """Frozen records: written once, never replaced (False if it already exists)."""
    inside_root(file)
    if file.exists():
        return False
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_name(f".{file.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.link(tmp, file)  # fails if file appeared meanwhile
    tmp.unlink()
    try:
        os.chmod(file, 0o444)
    except OSError:
        pass
    return True


def dump(data) -> str:
    return json.dumps(data, indent=1, default=str) + "\n"


def write_table(file: Path, rows: list[dict], cols: list[str], delimiter: str = ",") -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", lineterminator="\n", delimiter=delimiter)
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in r.items()})
    write_atomic(file, buf.getvalue())


def copy_verified(src: Path, dst: Path, want: str) -> str:
    inside_root(dst)
    h = sha256(src)
    if h != want:
        raise SystemExit(f"{src}: sha256 {h[:16]} != expected {want[:16]}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    shutil.copyfile(src, tmp)
    if sha256(tmp) != h:
        raise SystemExit(f"copy of {src} differs")
    os.replace(tmp, dst)
    return h


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


def config_facts(file: Path) -> dict:
    """Report-only facts from the eval file's own resolved config (the checkpoint sha256 covers them)."""
    try:
        c = resolved_config(file)
    except Exception as error:  # noqa: BLE001
        return {"error": repr(error)}
    bd = dig(c, "env", "action", "body_delay")
    return {"task.parquet": dig(c, "task", "parquet"), "scene.support": dig(c, "scene", "support"),
            "termination.palm_position_threshold": dig(c, "termination", "palm_position_threshold"),
            "env.termination.palm_position_threshold": dig(c, "env", "termination", "palm_position_threshold"),
            "training.seed": dig(c, "training", "seed"), "env.reset.seed": dig(c, "env", "reset", "seed"),
            "env.action.body_delay.seed": bd.get("seed") if isinstance(bd, dict) else "MISSING",
            "scene.object_scale_seed": dig(c, "scene", "object_scale_seed")}  # fmt: skip


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


def cpus_allowed(pid: int) -> str:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("Cpus_allowed_list:"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def live_jobs() -> dict[str, list[tuple[int, str]]]:
    found: dict[str, list] = {}
    for pid in all_pids():
        job = proc_environ(pid).get(MARK)
        st = proc_stat(pid)
        if job and st is not None and st[0] not in ("Z", "X"):
            found.setdefault(job, []).append((pid, st[19]))
    return found


def foreign_lane_load() -> tuple[dict[int, int], list[str]]:
    """Evaluations of anybody else (evaluate_policy.py processes without our marker) per NUMA eval-core lane, by their CPU
    affinity (every driver here pins its evals to EVAL_CORES with taskset)."""
    hits: dict[int, tuple[int, int]] = {}
    for pid in all_pids():
        cmd = proc_cmdline(pid)
        if len(cmd) < 2 or not any(x.endswith("flash_chord/scripts/evaluate_policy.py") for x in cmd[1:3]):
            continue
        if proc_environ(pid).get(MARK):
            continue
        st = proc_stat(pid)
        if st is None or st[0] in ("Z", "X"):
            continue
        aff = cpus_allowed(pid)
        for lane, cores in enumerate(EVAL_CORES):
            if aff == cores:
                hits[pid] = (lane, int(st[1]))
    load, seen = {0: 0, 1: 0}, []
    for pid, (lane, ppid) in hits.items():
        if ppid in hits:  # a child of a counted evaluation
            continue
        load[lane] += 1
        seen.append(f"{pid}@lane{lane}")
    return load, seen


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


def lane_lock_files(gpu: int) -> list[Path]:
    return [d / f"gpu{gpu}.lock" for d in SHARED_LOCKDIRS if (d / f"gpu{gpu}.lock").is_file()]


def take_lane(gpu: int) -> tuple[list[int], list[str]] | str:
    """Own lane lock (created in ROOT/evallock) + every shared lane lock file of that GPU (read-only, never created);
    all or nothing, never blocking."""
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


def gpu_rows(state: dict, mps: dict, ours: list[dict], gpus: list[int], rest_until: dict, lane_max: int) -> list[dict]:
    busy = {j["gpu"] for j in ours if j.get("gpu") is not None}
    lanes = {0: 0, 1: 0}
    for j in ours:
        if j.get("gpu") is not None:
            lanes[numa(j["gpu"])] += 1
    foreign, _ = foreign_lane_load()
    out = []
    for g in sorted(state["gpus"]):
        st = state["gpus"][g]
        row = {"gpu": g, "uuid": st["uuid"], "free": st["free"], "napps": sum(1 for x in state["apps"] if x["gpu"] == g)}
        rest = rest_until.get(g, 0.0) - time.time()
        load = lanes[numa(g)] + foreign[numa(g)]
        if g not in ALLOWED_GPUS:
            why = "never (GPU 4 is reserved)" if g == 4 else "not allowed"
        elif g not in gpus:
            why = "not in --gpus"
        elif g in mps:
            why = f"MPS ({mps[g]})"
        elif g in busy:
            why = "one of our evals already runs there"
        elif rest > 0:
            why = f"resting {rest:.0f} s after our last eval (lets other drivers take the lane)"
        elif st["free"] < MIN_FREE_MB + EVAL_PEAK_MB:
            why = f"free {st['free']} MiB < {MIN_FREE_MB} + {EVAL_PEAK_MB}"
        elif load >= lane_max:
            why = f"NUMA{numa(g)} eval-core lane has {load} evaluations (ours {lanes[numa(g)]}, others {foreign[numa(g)]}) >= {lane_max}"
        else:
            why = ""
        row.update(usable=not why, why=why or "usable")
        out.append(row)
    return out


def pick_gpu(state: dict, mps: dict, ours: list[dict], gpus: list[int], rest_until: dict, lane_max: int):
    rows = gpu_rows(state, mps, ours, gpus, rest_until, lane_max)
    usable = sorted((r for r in rows if r["usable"]), key=lambda r: (rest_until.get(r["gpu"], 0.0), r["napps"], -r["free"], r["gpu"]))
    if not usable:
        return "no usable GPU: " + "; ".join(f"gpu{r['gpu']} {r['why']}" for r in rows if r["gpu"] in gpus)
    held = []
    for r in usable:
        got = take_lane(r["gpu"])
        if isinstance(got, str):
            held.append(got)
            continue
        return r["gpu"], got[0], got[1]
    return "every usable GPU's lane is held: " + "; ".join(held)


def child_env(gpu: int | None, job_dir: Path, data_root: Path) -> dict:
    """The rolling-start watcher's eval environment (as t1best20 / the t3f0 post step), HUMAN_MOTION_DATA_DIR = the
    item's own Tier-3 data root (as the post step), marker T3FINAL_JOB=<job dir>, TMPDIR inside ROOT."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CUDA_MPS_", "WANDB_", "RUNIN_", "TIER2V2_", "K3EASY",
                                                                    "K3FIX", "K3CA", "T2FIX", "T3OPUS", "T3F0", "T1BEST"))
           and k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "TIER2_GPU", "FLASH_CHORD_CACHE_DIR")}  # fmt: skip
    sp = ME / "conda_envs/flashchord/lib/python3.11/site-packages/nvidia"
    env.update(
        PATH=f"{ME}/conda_envs/flashchord/bin:{env.get('PATH', '/usr/bin:/bin')}",
        LD_LIBRARY_PATH=f"{sp}/cu13/lib:{sp}/cudnn/lib:{sp}/nccl/lib:{sp}/nvshmem/lib",
        CUDA_VISIBLE_DEVICES="" if gpu is None else str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
        XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
        NUMEXPR_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1", WANDB_MODE="disabled",
        HUMAN_MOTION_DATA_DIR=str(data_root), XDG_CACHE_HOME=str(ME / ".cache"),
        TZ="Asia/Shanghai", TMPDIR=str(ROOT / "tmp"),
    )  # fmt: skip
    env[MARK] = str(job_dir)
    return env


ENV_SHOWN = ("CUDA_VISIBLE_DEVICES", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE", "XLA_PYTHON_CLIENT_PREALLOCATE",
             "OMP_NUM_THREADS", "HUMAN_MOTION_DATA_DIR", "TMPDIR", MARK)  # fmt: skip


def show_command(prefix: list[str], cmd: list[str], cwd: Path, env: dict) -> str:
    shown = " ".join(f"{k}={shlex.quote(env[k])}" for k in ENV_SHOWN if k in env)
    return f"cd {shlex.quote(str(cwd))} && env {shown} {shlex.join(prefix + cmd)}"


def eval_command(item: dict, w: Path, gpu: int) -> list[str]:
    """scripts/track2/eval_all.py --export no (as t1best20 / the watchers) with the item's own manifest: the work dir's
    ep<NN>/model_<N>.safetensors is a symlink to the pinned file, which the kit runner resolves and evaluates as is."""
    return [str(PY), "-B", str(CODE / "scripts/track2/eval_all.py"), "--arm-dir", str(w), "--arm", "T3f0final",
            "--export", "no", "--iteration", str(item["iteration"]), "--episodes", str(item["episode"]), "--gpus", str(gpu),
            "--jobs-per-gpu", "1", "--world-count", "1", "--skip-pack", "--kit-root", str(KIT),
            "--stock-runtime", str(STOCK), "--dataset-root", str(DATASET), "--manifest", item["manifest"],
            "--pack-python", str(PACK_PYTHON), "--results", str(w / "results"), "--code", str(CODE), "--python", str(PY)]


def prepare_workdir(item: dict, w: Path) -> None:
    d = w / f"ep{item['episode']:02d}"
    d.mkdir(parents=True, exist_ok=False)
    (d / f"model_{item['iteration']}.safetensors").symlink_to(item["eval_file"])


def launch_eval(item: dict, w: Path, gpu: int, uuid: str, fds: list[int], taken: list[str], header: str) -> tuple[subprocess.Popen, str]:
    cmd = eval_command(item, w, gpu)
    env = child_env(gpu, w, Path(item["data_root"]))
    cpus = EVAL_CORES[numa(gpu)]
    prefix = ["nice", "-n", "10", "taskset", "-c", cpus]
    shown = show_command(prefix, cmd, CODE, env)
    with (w / "eval_all.log").open("a") as stream:
        stream.write(f"[{now()}] {header}, gpu {gpu} ({uuid}), cpus {cpus}, lane locks {taken}: {shown}\n")
        stream.flush()
        proc = subprocess.Popen(prefix + cmd, cwd=CODE, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True, pass_fds=tuple(fds))  # fmt: skip
    write_atomic(w / "start.json", dump({"episode": item["episode"], "kind": item["kind"], "gpu": gpu, "gpu_uuid": uuid,
                                         "host": HOST, "cpus": cpus, "pid": proc.pid, "started_at": now(),
                                         "started_epoch": time.time(), "cwd": str(CODE), "command": prefix + cmd,
                                         "shown": shown, "lane_locks": taken, "header": header}))  # fmt: skip
    return proc, shown


# ================================================================================================ items
def item_path(e: int) -> Path:
    return ROOT / "items" / f"ep{e:02d}.json"


def load_item(e: int) -> dict | None:
    return read_json(item_path(e))


def run_rec(e: int, k: int) -> dict | None:
    return read_json(ROOT / "runs" / f"ep{e:02d}" / f"r{k}" / "run.json")


def sel_rec(e: int) -> dict | None:
    return read_json(ROOT / "selection" / f"ep{e:02d}.json")


def check_results_json(js: dict, e: int, sha: str, eval_file: str, parquet: str) -> list[str]:
    out = []
    if js.get("checkpoint_sha256") != sha:
        out.append("results JSON checkpoint_sha256 != the pinned file")
    if realpath(js.get("checkpoint")) != realpath(eval_file):
        out.append(f"evaluated checkpoint path {js.get('checkpoint')} is not the pinned file")
    if realpath(js.get("parquet")) != realpath(parquet):
        out.append(f"evaluator reference parquet {js.get('parquet')} is not the item's own input {parquet}")
    for key, want in (("start_frame", 0), ("motion_start_frame", 0), ("world_count", 1), ("reset_mode", "explicit")):
        if js.get(key) != want:
            out.append(f"{key}={js.get(key)!r}, expected {want!r}")
    if js.get("non_finite_worlds"):
        out.append(f"non_finite_worlds {js.get('non_finite_worlds')}")
    return out


def build_item(kind: str, e: int) -> tuple[dict | None, list[str]]:
    """The item of episode e from source `kind` (f0 | opus_v2), with every verification of the module doc."""
    s = SRC[kind]
    w = s["analysis"] / "eval" / f"ep{e:02d}"
    post = read_json(w / "post.json") or {}
    if post.get("status") not in EVALUATED or post.get("eval_status") != "evaluated" or post.get("eval_exit") not in (0, None):
        return None, [f"{kind}: post step has no evaluated checkpoint (status {post.get('status')}, eval_status "
                      f"{post.get('eval_status')}, exit {post.get('eval_exit')})"]
    problems: list[str] = []
    sha = str(post.get("checkpoint_sha256") or "")
    eval_file = Path(str(post.get("evaluated_checkpoint") or ""))
    if not eval_file.is_file():
        return None, [f"{kind}: evaluated checkpoint {eval_file} missing"]
    fsha = sha256(eval_file)
    if fsha != sha:
        problems.append(f"{kind}: {eval_file} sha256 {fsha[:16]} != post.json {sha[:16]}")
    rep = read_json(w / "eval_report.json") or {}
    row = next((r for r in rep.get("episodes", []) if str(r.get("episode")) == str(e)), None)
    if row is None:
        problems.append(f"{kind}: no eval_report.json row")
    else:
        if row.get("status") != "evaluated" or row.get("checkpoint_sha256") != fsha:
            problems.append(f"{kind}: eval_report row status {row.get('status')} / sha256 {str(row.get('checkpoint_sha256'))[:16]}")
        if realpath(row.get("evaluated_checkpoint")) != realpath(eval_file):
            problems.append(f"{kind}: eval_report evaluated_checkpoint {row.get('evaluated_checkpoint')} != post.json")
    er = read_json(eval_file.parent / "export_report.json") or {}
    stock = er.get("stock_runtime") or {}
    tol = fnum(post.get("export_tolerance"))
    if tol not in EXPORT_TOLS or fnum(er.get("tolerance")) != tol:
        problems.append(f"{kind}: export tolerance post {post.get('export_tolerance')} / report {er.get('tolerance')}")
    rel = fnum(er.get("max_rel_diff"))
    if er.get("status") != "ok" or rel is None or tol is None or rel > tol:
        problems.append(f"{kind}: export report status {er.get('status')} max_rel_diff {er.get('max_rel_diff')}")
    if stock.get("verified") is not True or realpath(stock.get("path")) != realpath(STOCK):
        problems.append(f"{kind}: export not verified on the stock runtime ({stock.get('verified')}, {stock.get('path')})")
    if stock.get("eval_code_sha256") not in (None, EVAL_CODE_SHA):
        problems.append(f"{kind}: stock eval_code_sha256 {stock.get('eval_code_sha256')} is not the common build")
    if stock.get("curriculum_stage_voc_scale") not in (None, 0, 0.0):
        problems.append(f"{kind}: curriculum_stage_voc_scale {stock.get('curriculum_stage_voc_scale')} (help) != 0")
    if er.get("destination_sha256") != fsha:
        problems.append(f"{kind}: export destination_sha256 != the eval file")
    run_dir = s["runs"] / f"ep{e:02d}"
    fin = read_json(run_dir / "final.json") or {}
    try:
        it = int(fin["iteration"])
        ck = str(fin["checkpoint"])
    except (KeyError, TypeError, ValueError):
        return None, problems + [f"{kind}: {run_dir}/final.json unusable"]
    if ck != f"model_{it}.safetensors" or eval_file.name != ck or int(post.get("final_iteration", -1)) != it:
        problems.append(f"{kind}: final.json {ck} / post final_iteration {post.get('final_iteration')} / eval file {eval_file.name}")
    train = run_dir / ck
    tsha = sha256(train) if train.is_file() else "missing"
    if tsha != er.get("source_sha256") or realpath(er.get("source")) != realpath(train):
        problems.append(f"{kind}: export source {er.get('source')} ({str(er.get('source_sha256'))[:16]}) != {train} ({tsha[:16]})")
    pq = Path(str(post.get("parquet") or ""))
    psha = sha256(pq) if pq.is_file() else "missing"
    if post.get("parquet_sha256") not in (None, psha):
        problems.append(f"{kind}: parquet sha256 {psha[:16]} != post.json {str(post.get('parquet_sha256'))[:16]}")
    lock = None
    if kind == "f0":
        lk = read_json(F0_LOCK) or {}
        lock = lk.get(f"{e:02d}") or {}
        if realpath(lock.get("parquet")) != realpath(pq) or lock.get("sha256") != psha:
            problems.append(f"f0: parquet {pq} ({psha[:16]}) is not the retarget-locked one")
    man = read_json(s["manifest"]) or {}
    mrow = next((r for k, r in man.items() if isinstance(r, dict) and int(r.get("episode_index", -1)) == e), None)
    if mrow is None:
        return None, problems + [f"{kind}: episode {e} not in {s['manifest']}"]
    seq, obj = str(mrow.get("sequence_id")), str(mrow.get("object"))
    if f"/sequence_id={seq}/" not in str(pq):
        problems.append(f"{kind}: parquet {pq} is not sequence {seq}")
    rj, rp = w / "results" / f"episode_{e:06d}.json", w / "results" / f"episode_{e:06d}.parquet"
    js = read_json(rj) or {}
    problems += [f"{kind} post-step results: {p}" for p in check_results_json(js, e, fsha, str(eval_file), str(pq))]
    if not rp.is_file():
        problems.append(f"{kind}: post-step rollout parquet missing")
    fr = s["frame_refs"] / f"episode_{e:06d}.npz"
    if not fr.is_file():
        problems.append(f"{kind}: frame reference {fr} missing")
    keys, _ = kit_metric_keys()
    m = js.get("metrics") or {}
    item = {
        "episode": e, "kind": kind, "source_label": s["label"], "sequence_id": seq, "object": obj,
        "eval_file": str(eval_file), "eval_sha256": fsha, "iteration": it, "final_reason": fin.get("reason"),
        "help_zero_since": fin.get("help_zero_since"), "training_checkpoint": str(train), "training_sha256": tsha,
        "export_tolerance": tol, "export_max_rel_diff": rel, "export_report": str(eval_file.parent / "export_report.json"),
        "parquet": str(pq), "parquet_sha256": psha, "retarget_lock": lock, "manifest": str(s["manifest"]),
        "manifest_sha256": sha256(s["manifest"]), "data_root": str(s["data"]),
        "frame_ref": str(fr), "frame_ref_sha256": sha256(fr) if fr.is_file() else None,
        "post_json": str(w / "post.json"), "post_json_sha256": sha256(w / "post.json"), "post_status": post.get("status"),
        "post_eval_gpu": post.get("eval_gpu"), "post_eval_started_at": post.get("eval_started_at"),
        "post_results_json": str(rj), "post_results_json_sha256": sha256(rj) if rj.is_file() else None,
        "post_results_parquet": str(rp), "post_results_parquet_sha256": sha256(rp) if rp.is_file() else None,
        "post_own_input": {k: fnum(m.get(src)) for k, src in keys.items()},
        "mitigated": e in MITIGATED, "scene_support": e in SUPPORT_EPISODES, "config": config_facts(eval_file),
        "code_commit": CODE_COMMIT, "code_tree": str(CODE),
    }  # fmt: skip
    return item, problems


def resolve_episode(e: int) -> tuple[str, dict | None, str]:
    """('pending' | 'f0' | 'opus_v2' | 'none' | 'blocked', item, why): PREREG_f0 'Use in a submission'."""
    a = SRC["f0"]["analysis"]
    post = read_json(a / "eval" / f"ep{e:02d}" / "post.json") or {}
    crashed = (a / f"CRASHED_ep{e:02d}").exists()
    st = post.get("status")
    if st in EVALUATED:
        item, problems = build_item("f0", e)
        if item is None or problems:
            return "blocked", None, "; ".join(problems)
        return "f0", item, f"f0 post {st}: final.json checkpoint model_{item['iteration']} evaluated"
    if st in NOT_EVALUATED or crashed:
        why = f"f0 has no evaluated checkpoint (post status {st}{'; CRASHED marker' if crashed else ''})"
        opost = read_json(SRC["opus_v2"]["analysis"] / "eval" / f"ep{e:02d}" / "post.json") or {}
        if opost.get("status") in EVALUATED:
            item, problems = build_item("opus_v2", e)
            if item is None or problems:
                return "blocked", None, why + "; opus_v2 fallback: " + "; ".join(problems)
            item["fallback_reason"] = why
            return "opus_v2", item, why + "; PREREG_f0 fallback to the opus_v2 policy"
        return "none", None, why + f"; opus_v2 post status {opost.get('status')}: NO POLICY for this episode"
    return "pending", None, f"f0 post status {st or 'not started'}"


def freeze(e: int, state: str, item: dict | None, why: str) -> bool:
    rec = {"episode": e, "state": state, "why": why, "frozen_at": now(),
           "driver_sha256": sha256(Path(__file__).resolve()), **(item or {})}  # fmt: skip
    rec["state"] = state
    return write_once(item_path(e), dump(rec))


def resolve_all(log, freeze_it: bool = True) -> dict[int, tuple[str, str]]:
    out = {}
    for e in EPISODES:
        old = load_item(e)
        if old is not None:
            out[e] = (old["state"], "frozen")
            continue
        state, item, why = resolve_episode(e)
        out[e] = (state, why)
        if freeze_it and state in ("f0", "opus_v2", "none"):
            if freeze(e, state, item, why):
                log(f"ep{e:02d}: item FROZEN as {state}: {why}"
                    + (f"; eval file {item['eval_file']} sha256 {item['eval_sha256'][:16]}" if item else ""))
        elif state == "blocked":
            log.first(("blocked", e, why), f"ep{e:02d}: BLOCKED (not frozen; needs a human): {why}")
    return out


# ================================================================================================ checks
def parquet_meta(pp: Path) -> dict:
    import pyarrow.parquet as pq

    f = pq.ParquetFile(pp)
    return {k.decode(errors="replace"): v.decode(errors="replace") for k, v in (f.schema_arrow.metadata or {}).items()
            if not k.startswith(b"pandas")}  # fmt: skip


def parquet_problems(pp: Path, sha: str, step_count) -> tuple[list[str], dict]:
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
    if md.get("flash_chord.world_count") not in (None, "1"):
        out.append(f"parquet world_count {md.get('flash_chord.world_count')} != 1")
    valid = pq.read_table(pp, columns=["environment_valid"]).column("environment_valid").to_pylist()
    if not valid or not all(valid):
        out.append("environment_valid=false rows")
    keep = {k.replace("flash_chord.", ""): md.get(k) for k in md if k.startswith("flash_chord.")}
    return out, keep


def check_eval(item: dict, results: Path, report: Path | None, exit_code, keys: dict) -> tuple[list[str], dict]:
    e = int(item["episode"])
    problems: list[str] = []
    if exit_code not in (0, None):
        problems.append(f"exit {exit_code}")
    if report is not None:
        rep = read_json(report) or {}
        erow = next((r for r in rep.get("episodes", []) if str(r.get("episode")) == str(e)), None)
        if erow is None:
            problems.append("no eval_report.json row for the episode")
        elif erow.get("status") != "evaluated":
            problems.append(f"eval_report status {erow.get('status')}")
    jp, pp = results / f"episode_{e:06d}.json", results / f"episode_{e:06d}.parquet"
    info: dict = {"results_json": str(jp), "results_parquet": str(pp)}
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
    problems += check_results_json(js, e, item["eval_sha256"], item["eval_file"], item["parquet"])
    if Path(item["parquet"]).is_file() and sha256(item["parquet"]) != item["parquet_sha256"]:
        problems.append("the item's own-input parquet CHANGED")
    if not pp.is_file():
        problems.append("no rollout parquet")
    else:
        try:
            pprob, meta = parquet_problems(pp, item["eval_sha256"], js.get("step_count"))
            problems += pprob
            info["parquet_meta"] = meta
            info["parquet_sha256"] = sha256(pp)
        except Exception as error:  # noqa: BLE001
            problems.append(f"parquet unreadable ({error!r})")
    if not Path(item["eval_file"]).is_file():
        problems.append("the pinned file is MISSING")
    elif sha256(item["eval_file"]) != item["eval_sha256"]:
        problems.append("the pinned file CHANGED")
    log_file = results / "logs" / f"episode_{e:06d}.log"
    try:
        first = log_file.read_text().splitlines()[0]
        info["runner_command"] = re.sub(r"^\[[^\]]+\] gpu [^:]+: ", "", first)
    except (OSError, IndexError):
        info["runner_command"] = ""
    return problems, info


def select_run(runs: list[dict]) -> dict:
    """PREREG selection: among ok runs, own-input AUC closest to their mean; ties -> lower run index; none ok -> run 0."""
    oks = [r for r in runs if r.get("status") == "ok" and fnum((r.get("metrics") or {}).get("auc")) is not None]
    if not oks:
        return {"chosen_run": 0, "n_ok": 0, "mean_auc": None, "distances": {},
                "rule": "no ok run: run 0 = the post step's own official evaluation of the same file"}
    aucs = {int(r["k"]): float(r["metrics"]["auc"]) for r in oks}
    exact = {k: Fraction(v) for k, v in aucs.items()}  # exact rational arithmetic on the float64 values: exact ties
    mean = sum(exact.values(), Fraction(0)) / len(exact)
    dist = {k: abs(v - mean) for k, v in exact.items()}
    chosen = min(exact, key=lambda k: (dist[k], k))
    return {"chosen_run": chosen, "n_ok": len(oks), "mean_auc": float(mean), "auc_by_run": aucs,
            "distances": {k: float(v) for k, v in dist.items()},
            "rule": "own-input ADD-AUC closest to the mean over the ok runs (exact rational arithmetic on the float64 "
                    "values); ties -> lower run index"}


# ================================================================================================ preflight
def tool_hashes() -> dict:
    out = {"kit_runner": sha256(RUNNER), "kit_evaluate_policy": sha256(EVAL_SCRIPT),
           "stock_evaluate_policy": sha256(STOCK / "scripts/evaluate_policy.py"), "kit_packer": sha256(PACKER),
           "kit_uploader": sha256(UPLOADER), "kit_manifest": sha256(KIT / "KIT_MANIFEST.json"),
           "kit_tracks_json": sha256(KIT / "config/tracks.json"), "eval_all": sha256(CODE / "scripts/track2/eval_all.py"),
           "exporter": sha256(CODE / "src/flash_chord/training/ppo/coordinate/export.py"),
           "driver": sha256(Path(__file__).resolve()), "kit_helper": sha256(KITHELPER) if KITHELPER.is_file() else None}  # fmt: skip
    for m in METRIC_SHA:
        out[f"metric_code_{m}"] = sha256(KIT / f"metric_code/track_2/{m}.py")
    return out


def preflight() -> tuple[list[str], list[str]]:
    fatal, info = [], []
    for need in (PY, PACK_PYTHON, RUNNER, EVAL_SCRIPT, PACKER, UPLOADER, STOCK / "src/flash_chord/__init__.py", KITHELPER, F0_LOCK):
        if not need.exists():
            fatal.append(f"missing {need}")
    if fatal:
        return fatal, info
    h = tool_hashes()
    pins = [("kit_evaluate_policy", EVAL_SCRIPT_SHA), ("stock_evaluate_policy", EVAL_SCRIPT_SHA), ("kit_runner", RUNNER_SHA),
            ("kit_packer", PACKER_SHA), ("kit_uploader", UPLOADER_SHA), ("eval_all", EVAL_ALL_SHA)]
    pins += [(f"metric_code_{m}", v) for m, v in METRIC_SHA.items()]
    for key, want in pins:
        (info if h[key] == want else fatal).append(f"{key} sha256 {h[key][:16]} " + ("ok" if h[key] == want else f"!= {want[:16]}"))
    try:
        man = json.loads((KIT / "KIT_MANIFEST.json").read_text())["files"]
        bad = [p for p, hh in man.items() if not (KIT / p).is_file() or sha256(KIT / p) != hh]
        (fatal if bad else info).append(f"KIT_MANIFEST: {len(man)} files, {len(bad)} mismatched {bad[:5]}")
    except (OSError, ValueError, KeyError) as error:
        fatal.append(f"KIT_MANIFEST unreadable ({error!r})")
    head = git_head(STOCK_REPO)
    (info if head == STOCK_COMMIT else fatal).append(f"stock runtime HEAD {head[:12]} (want {STOCK_COMMIT[:12]})")
    hd = git_head(CODE)
    (info if hd == CODE_COMMIT else fatal).append(f"code/fc_r2 HEAD {hd[:12]} (want {CODE_COMMIT[:12]})")
    for kind, s in SRC.items():
        for label, path, want in (("PREREG", s["prereg"], s["prereg_sha"]), ("manifest", s["manifest"], s["manifest_sha"]),
                                  ("READY", s["data"] / "READY", s["ready_sha"])):
            got = sha256(path) if Path(path).is_file() else "missing"
            (info if got == want else fatal).append(f"{kind} {label} {got[:16]} " + ("ok" if got == want else f"!= {want[:16]}"))
    got = sha256(F0_LOCK)
    (info if got == F0_LOCK_SHA else fatal).append(f"f0 retarget lock {got[:16]} " + ("ok" if got == F0_LOCK_SHA else f"!= {F0_LOCK_SHA[:16]}"))
    got = sha256(SRC["f0"]["data"] / "FILES.sha256")
    (info if got == F0_FILES_SHA else fatal).append(f"f0 FILES.sha256 {got[:16]} " + ("ok" if got == F0_FILES_SHA else "CHANGED"))
    for g in ALLOWED_GPUS:
        info.append(f"gpu{g} shared lane lock files: {[str(p).replace(str(ME) + '/', '') for p in lane_lock_files(g)]}")
    return fatal, info


def f0_frame_ref_listed(path: str, sha: str) -> bool:
    """frame_references_f0 files are covered by the f0 root's FILES.sha256 (write-once root)."""
    rel = Path(path).relative_to(SRC["f0"]["data"]).as_posix()
    for line in (SRC["f0"]["data"] / "FILES.sha256").read_text().splitlines():
        f = line.split(None, 1)
        if len(f) == 2 and f[1].strip().lstrip("./") in (rel, rel.split("/", 1)[-1]) and f[0] == sha:
            return True
    return False


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


def prereg_ok() -> str | None:
    if not PREREG.is_file() or not PREREG_SHA.is_file():
        return "PREREG.md / PREREG.sha256 missing"
    want = PREREG_SHA.read_text().split()[0]
    if sha256(PREREG) != want:
        return f"PREREG.md sha256 {sha256(PREREG)[:16]} != PREREG.sha256 {want[:16]}"
    text = PREREG.read_text()
    for label, path in (("driver", Path(__file__).resolve()), ("kit helper", KITHELPER)):
        if sha256(path) not in text:
            return f"PREREG.md does not name the current {label} sha256 {sha256(path)[:16]} ({path})"
    return None


# ================================================================================================ the driver
class Driver:
    def __init__(self, a: argparse.Namespace, log: Logger):
        self.a, self.log = a, log
        self.gpus = sorted({int(x) for x in re.split(r"[,\s]+", a.gpus.strip()) if x})
        self.keys, self.keys_from = kit_metric_keys()
        self.running: dict[tuple, dict] = {}
        self.orphans: dict[tuple, dict] = {}
        self.items: dict[int, dict] = {}
        self.durations: list[float] = []
        self.rest_until: dict[int, float] = {}
        self.final_cache: set[tuple[int, int]] = set()
        self.last_resolve = 0.0
        self.last_summary = 0.0
        self.summary_due = True
        self.states: dict[int, tuple[str, str]] = {}

    def refresh_items(self) -> None:
        for e in EPISODES:
            if e not in self.items:
                it = load_item(e)
                if it is not None:
                    if it["state"] in ("f0", "opus_v2") and sha256(it["eval_file"]) != it["eval_sha256"]:
                        self.log.first(("changed", e), f"ep{e:02d}: pinned file {it['eval_file']} CHANGED: not evaluated")
                        continue
                    self.items[e] = it

    def evaluable(self) -> list[dict]:
        return [self.items[e] for e in sorted(self.items) if self.items[e]["state"] in ("f0", "opus_v2")]

    def jobs_wanted(self) -> list[tuple[dict, int]]:
        return [(it, k) for it in self.evaluable() for k in range(1, self.a.runs + 1)]

    def final(self, it: dict, k: int) -> bool:
        key = (it["episode"], k)
        if key in self.final_cache:
            return True
        rec = run_rec(it["episode"], k)
        if rec is None:
            return False
        if rec.get("status") != "ok":
            self.final_cache.add(key)
            return True
        js = read_json(rec.get("results_json", "")) or {}
        if js.get("checkpoint_sha256") == it["eval_sha256"]:
            self.final_cache.add(key)
            return True
        self.log.first(("resha", key), f"ep{key[0]:02d} r{k}: run.json ok but results JSON sha256 != pinned file: re-queued")
        return False

    def busy(self, key: tuple) -> bool:
        return key in self.running or key in self.orphans

    @staticmethod
    def retry_due(d: Path) -> bool:
        last = read_json(d / "try.json") or {}
        return time.time() - (parse_time(last.get("finished_at")) or 0.0) >= RETRY_DELAY

    def launch_evals(self) -> None:
        ours = [*self.running.values(), *self.orphans.values()]
        if len(ours) >= self.a.max_parallel:
            return
        ready = []
        for it, k in self.jobs_wanted():
            key = (it["episode"], k)
            if self.busy(key) or self.final(it, k):
                continue
            done = tries(ROOT / "runs" / f"ep{it['episode']:02d}" / f"r{k}")
            if done and not self.retry_due(done[-1][1]):
                continue
            ready.append((it, k, done))
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
        for it, k, done in ready:
            if len(ours) >= self.a.max_parallel:
                break
            choice = pick_gpu(state, mps, ours, self.gpus, self.rest_until, self.a.lane_max)
            if isinstance(choice, str):
                self.log.first(("wait", re.sub(r"\d+", "N", choice)), f"{len(ready)} eval(s) ready, waiting: {choice}")
                break
            gpu, fds, taken = choice
            t = done[-1][0] + 1 if done else 1
            w = inside_root(ROOT / "runs" / f"ep{it['episode']:02d}" / f"r{k}" / f"try{t}")
            try:
                prepare_workdir(it, w)
                proc, shown = launch_eval(it, w, gpu, state["gpus"][gpu]["uuid"], fds, taken,
                                          f"ep{it['episode']:02d} r{k} try {t} ({it['kind']})")
            except Exception:
                for fd in fds:
                    os.close(fd)
                raise
            key = (it["episode"], k)
            self.running[key] = {"episode": it["episode"], "k": k, "item": it, "try": t, "dir": w, "proc": proc,
                                 "start": time.time(), "gpu": gpu, "fds": fds, "term_at": None, "killed": False,
                                 "timed_out": False}  # fmt: skip
            ours.append(self.running[key])
            self.log(f"start ep{it['episode']:02d} r{k} try {t} on gpu {gpu} (pid {proc.pid}; locks {len(taken)})")

    def enforce_timeout(self, job: dict) -> None:
        elapsed = time.time() - job["start"]
        if elapsed < EVAL_TIMEOUT:
            return
        if job["term_at"] is None:
            job["term_at"], job["timed_out"], sig = time.time(), True, signal.SIGTERM
        elif not job["killed"] and time.time() - job["term_at"] >= KILL_GRACE:
            job["killed"], sig = True, signal.SIGKILL
        else:
            return
        self.log(f"ep{job['episode']:02d} r{job['k']}: timeout ({elapsed:.0f} s): {sig.name} to its own session {job['proc'].pid}")
        try:
            os.killpg(job["proc"].pid, sig)  # our child's own session (start_new_session), never anything else
        except OSError as error:
            self.log(f"{sig.name} failed ({error})")

    def reap(self) -> None:
        for key, job in list(self.running.items()):
            try:
                pid, status, ru = os.wait4(job["proc"].pid, os.WNOHANG)
            except ChildProcessError:
                pid, status, ru = job["proc"].pid, 0, None
            if pid == 0:
                self.enforce_timeout(job)
                continue
            code = os.waitstatus_to_exitcode(status) if ru is not None else job["proc"].poll()
            del self.running[key]
            for fd in job["fds"]:
                os.close(fd)
            self.rest_until[job["gpu"]] = time.time() + self.a.cooldown
            cpu = None if ru is None else round(ru.ru_utime + ru.ru_stime, 1)
            self.finish(job["item"], job["k"], job["try"], job["dir"], code, job, "", cpu)

    def finish(self, it: dict, k: int, t: int, d: Path, code, job: dict | None, remark: str, cpu=None) -> None:
        problems, info = check_eval(it, d / "results", d / "eval_report.json", code, self.keys)
        if job is not None and job.get("timed_out"):
            problems.insert(0, f"timeout after {EVAL_TIMEOUT:.0f} s")
        start = read_json(d / "start.json") or {}
        seconds = round(time.time() - start["started_epoch"], 1) if start.get("started_epoch") else None
        status = "ok" if not problems else "failed"
        rec = {"episode": it["episode"], "k": k, "kind": it["kind"], "object": it["object"], "sequence_id": it["sequence_id"],
               "try": t, "status": status, "problems": problems, "remark": remark, "gpu": start.get("gpu"),
               "gpu_uuid": start.get("gpu_uuid"), "host": start.get("host"), "cpus": start.get("cpus"),
               "lane_locks": start.get("lane_locks"), "started_at": start.get("started_at"), "finished_at": now(),
               "seconds": seconds, "cpu_seconds": cpu, "pinned_file": it["eval_file"], "pinned_sha256": it["eval_sha256"],
               "own_input_parquet": it["parquet"], "work_dir": str(d), "metric_keys": self.keys,
               "wrapper_command": start.get("shown"), "stock_commit": STOCK_COMMIT,
               "driver_sha256": sha256(Path(__file__).resolve()), **info}  # fmt: skip
        write_atomic(d / "try.json", dump(rec))
        base = d.parent
        if status == "ok":
            write_atomic(base / "run.json", dump(rec))
            if seconds and not remark:
                self.durations.append(seconds)
            m = info["metrics"]
            self.log(f"ep{it['episode']:02d} r{k} ({it['object']}, {it['kind']}) try {t}, gpu {rec['gpu']}, {seconds or 0:.0f} s: "
                     f"own-input AUC {m['auc']:.4f} SP {m['sp_sr']:.0f} MP {m['mp_sr']:.0f} MPPE {m['mppe']:.2f} cm"
                     + (f" ({remark})" if remark else ""))  # fmt: skip
        elif len(tries(base)) >= EVAL_TRIES:
            write_atomic(base / "run.json", dump(rec))
            self.log(f"ep{it['episode']:02d} r{k}: FAILED after {t} tries ({problems}); log {d / 'eval_all.log'}")
        else:
            self.log(f"ep{it['episode']:02d} r{k} try {t} failed ({problems}); retry in {RETRY_DELAY:.0f} s")
        self.summary_due = True

    def recover(self) -> None:
        live = live_jobs()
        for it, k in self.jobs_wanted():
            base = ROOT / "runs" / f"ep{it['episode']:02d}" / f"r{k}"
            for t, d in tries(base):
                if (d / "try.json").is_file():
                    continue
                key = (it["episode"], k)
                if str(d) in live and key not in self.orphans:
                    start = read_json(d / "start.json") or {}
                    self.orphans[key] = {"episode": it["episode"], "k": k, "item": it, "try": t, "dir": d,
                                         "gpu": start.get("gpu"), "procs": live[str(d)]}  # fmt: skip
                    self.rest_until[start.get("gpu")] = time.time() + 10 ** 6
                    self.log(f"ep{it['episode']:02d} r{k} try {t}: still running from a previous driver (pids "
                             f"{[p for p, _ in live[str(d)]]}); recorded when it ends, never killed")
                elif str(d) not in live:
                    self.finish(it, k, t, d, None, None, "recovered after a driver restart (exit code unknown)")

    def check_orphans(self) -> None:
        for key, o in list(self.orphans.items()):
            if any(proc_alive(p, s) for p, s in o["procs"]):
                continue
            del self.orphans[key]
            self.rest_until[o["gpu"]] = time.time() + self.a.cooldown
            self.finish(o["item"], o["k"], o["try"], o["dir"], None, None, "finished while no driver was running (exit code unknown)")

    def freeze_selections(self) -> None:
        for it in self.evaluable():
            e = it["episode"]
            if sel_rec(e) is not None:
                continue
            if not all(self.final(it, k) for k in range(1, self.a.runs + 1)):
                continue
            runs = []
            for k in range(1, self.a.runs + 1):
                r = run_rec(e, k) or {}
                runs.append({"k": k, "status": r.get("status"), "metrics": r.get("metrics"), "gpu": r.get("gpu"),
                             "results_json": r.get("results_json"), "results_parquet": r.get("results_parquet"),
                             "json_sha256": r.get("json_sha256"), "parquet_sha256": r.get("parquet_sha256")})  # fmt: skip
            sel = select_run(runs)
            rec = {"episode": e, "object": it["object"], "kind": it["kind"], "eval_sha256": it["eval_sha256"],
                   "runs": runs, **sel, "frozen_at": now(), "prereg_sha256": sha256(PREREG) if PREREG.is_file() else None,
                   "driver_sha256": sha256(Path(__file__).resolve()),
                   "post_own_input_report_only": it.get("post_own_input")}  # fmt: skip
            if write_once(ROOT / "selection" / f"ep{e:02d}.json", dump(rec)):
                self.log(f"ep{e:02d}: selection FROZEN: run {sel['chosen_run']} ({sel['n_ok']} ok runs; own-input AUC mean "
                         f"{sel['mean_auc'] if sel['mean_auc'] is None else round(sel['mean_auc'], 4)})")
                self.summary_due = True

    def settled(self) -> tuple[bool, list[int]]:
        open_eps = [e for e in EPISODES if e not in self.items]
        if open_eps:
            return False, open_eps
        for it in self.evaluable():
            if sel_rec(it["episode"]) is None:
                return False, [it["episode"]]
        return True, []

    def progress(self) -> dict:
        ev = {"ok": 0, "failed": 0, "pending": 0}
        for it, k in self.jobs_wanted():
            r = run_rec(it["episode"], k)
            ev["pending" if r is None else ("ok" if r.get("status") == "ok" else "failed")] += 1
        return ev

    def heartbeat(self, phase: str) -> None:
        prog = self.progress()
        lanes = max(1, min(self.a.max_parallel, len(self.gpus), 2 * self.a.lane_max))
        mean_s = sum(self.durations[-50:]) / len(self.durations[-50:]) if self.durations else 75.0
        cycle = mean_s + self.a.cooldown / 2 + self.a.poll
        unresolved = [e for e in EPISODES if e not in self.items]
        jobs = [{"episode": j["episode"], "k": j["k"], "try": j["try"], "gpu": j["gpu"], "pid": j["proc"].pid,
                 "elapsed_s": round(time.time() - j["start"])} for j in self.running.values()]  # fmt: skip
        jobs += [{"episode": o["episode"], "k": o["k"], "try": o["try"], "gpu": o.get("gpu"),
                  "orphan_pids": [p for p, _ in o["procs"]]} for o in self.orphans.values()]  # fmt: skip
        write_atomic(ROOT / "heartbeat.json", dump({
            "updated_at": now(), "pid": os.getpid(), "phase": phase, "runs_per_item": self.a.runs, "evals": prog,
            "items": {s: sum(1 for it in self.items.values() if it["state"] == s) for s in ("f0", "opus_v2", "none")},
            "unresolved_episodes": unresolved, "states": {f"{e:02d}": self.states.get(e, ("?", ""))[0] for e in unresolved},
            "selections_frozen": sum(1 for e in EPISODES if sel_rec(e) is not None), "running": jobs,
            "mean_eval_seconds": round(mean_s, 1),
            "eta_hours_for_resolved_items": round(cycle * prog["pending"] / lanes / 3600, 2),
            "eta_note": "resolved items only; unresolved episodes wait for their t3f0 post step",
            "stop_file": (ROOT / "STOP").exists()}))  # fmt: skip

    def run(self) -> int:
        a = self.a
        if 4 in self.gpus or not set(self.gpus) <= set(ALLOWED_GPUS):
            print(f"--gpus {self.gpus}: only {ALLOWED_GPUS} are allowed (never 4)", file=sys.stderr)
            return 2
        lockfd = os.open(inside_root(ROOT / ".driver.lock"), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(f"another t3final_driver.py run holds {ROOT / '.driver.lock'}", file=sys.stderr)
            return 1
        if (ROOT / "STOP").exists():
            print(f"{ROOT / 'STOP'} exists: remove it to start", file=sys.stderr)
            return 1
        (ROOT / "tmp").mkdir(exist_ok=True)
        write_atomic(ROOT / "driver.pid", f"{os.getpid()}\n")
        fatal, info = preflight()
        why = prereg_ok()
        if why:
            fatal.append(f"pre-registration: {why}")
        for line in info:
            self.log(f"preflight: {line}")
        if fatal:
            for line in fatal:
                self.log(f"preflight FAILED: {line}")
            return 1
        start_listing = integrity()
        self.log(f"start pid {os.getpid()} on {HOST}: {a.runs} runs per item, progressive over 30 episodes; gpus "
                 f"{self.gpus}, max parallel {a.max_parallel}, lane max {a.lane_max} (anybody's evals per NUMA lane), "
                 f"cooldown {a.cooldown:.0f} s; metrics {self.keys} ({self.keys_from}); driver "
                 f"{sha256(Path(__file__).resolve())[:16]}; PREREG {sha256(PREREG)[:16]}")  # fmt: skip
        self.states = resolve_all(self.log)
        self.last_resolve = time.time()
        self.refresh_items()
        self.recover()
        last_beat, packed = 0.0, False
        while True:
            stopping = (ROOT / "STOP").exists()
            if stopping:
                self.log.first("stop", f"STOP file: no new evaluation; waiting for {len(self.running)} running")
            self.reap()
            self.check_orphans()
            if not stopping and time.time() - self.last_resolve >= a.resolve_every and len(self.items) < len(EPISODES):
                self.states = resolve_all(self.log)
                self.last_resolve = time.time()
                before = set(self.items)
                self.refresh_items()
                if set(self.items) != before:
                    self.recover()
                    self.summary_due = True
            if not stopping:
                self.launch_evals()
            self.freeze_selections()
            if time.time() - last_beat >= 30:
                self.heartbeat("stopping" if stopping else "running")
                last_beat = time.time()
            if self.summary_due and time.time() - self.last_summary >= 120:
                write_summary(self.log)
                self.summary_due, self.last_summary = False, time.time()
            if not self.running and not self.orphans:
                if stopping:
                    break
                done, _ = self.settled()
                if done:
                    self.log("every episode is settled: items frozen, runs final, selections frozen")
                    break
            time.sleep(a.poll)
        write_summary(self.log)
        done, _ = self.settled()
        if done and not stopping and not a.no_pack:
            self.heartbeat("packing")
            env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", MARK)}
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            with (ROOT / "pack.log").open("a") as stream:
                proc = subprocess.Popen(["nice", "-n", "15", str(PY), "-B", str(Path(__file__).resolve()), "pack"], cwd=ROOT,
                                        env=env, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        start_new_session=True)  # fmt: skip
                try:
                    code = proc.wait(timeout=6 * PACK_TIMEOUT)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)  # our own pack session only
                    code = proc.wait()
            packed = code == 0
            self.log(f"pack: exit {code} (pack.log; clean numbers only in pack/*_LOCAL_NOT_FOR_UPLOAD*)")
        diff = integrity_diff(start_listing, integrity())
        write_atomic(ROOT / "integrity" / f"check_{stamp()}.json", dump({"from": start_listing["taken_at"], "to": now(),
                                                                         "differences": diff}))
        self.log("kit and stock runtime unchanged (file list, sizes, mtimes)" if not diff else
                 f"WARNING kit / stock runtime changed during this driver: {diff[:20]}")  # fmt: skip
        self.heartbeat("stopped (STOP file)" if stopping else ("done, packed" if packed else "done"))
        try:
            if int((ROOT / "driver.pid").read_text()) == os.getpid():
                (ROOT / "driver.pid").unlink()
        except (OSError, ValueError):
            pass
        os.close(lockfd)
        self.log(f"driver end: {self.progress()}")
        return 0


# ================================================================================================ summary (noisy-only)
def write_summary(log=None) -> None:
    """own_input_runs.csv, selection.csv, status.md: own-input numbers only (noisy side; no clean file is read)."""
    rows, sel_rows = [], []
    for e in EPISODES:
        it = load_item(e)
        if it is None or it["state"] not in ("f0", "opus_v2"):
            continue
        for k in range(1, NRUNS + 1):
            r = run_rec(e, k)
            if r is None:
                continue
            m = r.get("metrics") or {}
            rows.append({"episode": e, "object": it["object"], "kind": it["kind"], "run": k, "try": r.get("try"),
                         "status": r.get("status"), "gpu": r.get("gpu"), "seconds": r.get("seconds"),
                         "own_auc": m.get("auc"), "own_sp_sr": m.get("sp_sr"), "own_mp_sr": m.get("mp_sr"),
                         "own_mppe_cm": m.get("mppe"), "chord_sr": r.get("chord_sr"), "eval_sha256": it["eval_sha256"],
                         "parquet_sha256": r.get("parquet_sha256"), "finished_at": r.get("finished_at"),
                         "problems": "; ".join(r.get("problems") or [])})  # fmt: skip
        s = sel_rec(e)
        po = it.get("post_own_input") or {}
        sel_rows.append({"episode": e, "object": it["object"], "kind": it["kind"], "iteration": it["iteration"],
                         "eval_sha256": it["eval_sha256"], "n_ok": (s or {}).get("n_ok"),
                         "mean_own_auc": (s or {}).get("mean_auc"), "chosen_run": (s or {}).get("chosen_run"),
                         "chosen_own_auc": ((s or {}).get("auc_by_run") or {}).get(str((s or {}).get("chosen_run")),
                                           ((s or {}).get("auc_by_run") or {}).get((s or {}).get("chosen_run"))),
                         "post_run0_own_auc": po.get("auc"), "post_run0_sp_sr": po.get("sp_sr"),
                         "post_run0_mp_sr": po.get("mp_sr"), "frozen": s is not None})  # fmt: skip
    write_table(ROOT / "own_input_runs.csv", rows, ["episode", "object", "kind", "run", "try", "status", "gpu", "seconds",
                                                    "own_auc", "own_sp_sr", "own_mp_sr", "own_mppe_cm", "chord_sr",
                                                    "eval_sha256", "parquet_sha256", "finished_at", "problems"])
    write_table(ROOT / "selection.csv", sel_rows, ["episode", "object", "kind", "iteration", "eval_sha256", "n_ok",
                                                   "mean_own_auc", "chosen_run", "chosen_own_auc", "post_run0_own_auc",
                                                   "post_run0_sp_sr", "post_run0_mp_sr", "frozen"])
    md = ["# Tier-3 f0 FINAL: status (own-input numbers only; noisy side)", "",
          f"Generated {now()} by `t3final_driver.py`. Protocol: `PREREG.md`. Own input = each policy against its own "
          "Tier-3 training parquet. No clean number is shown or read here.", "",
          "| ep | object | item | iteration | eval sha256 | runs ok/failed | own AUC per run | mean | chosen run |",
          "|---|---|---|---|---|---|---|---|---|"]
    for e in EPISODES:
        it = load_item(e)
        if it is None:
            md.append(f"| {e:02d} | | unresolved | | | | | | |")
            continue
        if it["state"] == "none":
            md.append(f"| {e:02d} | | NO POLICY ({it.get('why')}) | | | | | | |")
            continue
        recs = [run_rec(e, k) for k in range(1, NRUNS + 1)]
        aucs = " ".join("." if r is None else (f"{r['metrics']['auc']:.3f}" if r.get("status") == "ok" else "X") for r in recs)
        s = sel_rec(e) or {}
        mean = s.get("mean_auc")
        md.append(f"| {e:02d} | {it['object']} | {it['state']} | {it['iteration']} | {it['eval_sha256'][:10]} | "
                  f"{sum(1 for r in recs if r and r.get('status') == 'ok')}/{sum(1 for r in recs if r and r.get('status') != 'ok')} | "
                  f"{aucs} | {'' if mean is None else f'{mean:.4f}'} | {s.get('chosen_run', '')} |")
    write_atomic(ROOT / "status.md", "\n".join(md) + "\n")
    if log is not None:
        log(f"summary: {len(rows)} run verdicts, {sum(1 for r in sel_rows if r['frozen'])} selections frozen")


# ================================================================================================ pack
def kit_env() -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", MARK)}
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               TMPDIR=str(ROOT / "tmp"), CUDA_VISIBLE_DEVICES="")
    return env


def run_packer(d: Path, url: str, out_csv: Path, config: Path | None = None) -> dict:
    """The kit packer, unchanged (v2dkit python -I -B), track_2_tier3, the file's own frame references. Its stdout holds
    clean numbers ('locally computed'): it goes only to pack_LOCAL_NOT_FOR_UPLOAD.log; only non-clean lines are parsed."""
    cmd = ["nice", "-n", "15", str(PACK_PYTHON), "-I", "-B", str(PACKER), "--track", TRACK, "--results", str(d / "results"),
           "--dataset-root", str(DATASET), "--frame-references", str(d / "frame_refs"), "--code-commit", url,
           "--eval-script", str(EVAL_SCRIPT), "-o", str(inside_root(out_csv))]  # fmt: skip
    if config is not None:
        cmd += ["--config", str(config)]
    log_file = d / "pack_LOCAL_NOT_FOR_UPLOAD.log"
    try:
        p = subprocess.run(cmd, cwd=d, env=kit_env(), capture_output=True, text=True, timeout=PACK_TIMEOUT)
        code, stdout, stderr = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        code, stdout, stderr = -9, "", "timeout"
    write_atomic(log_file, f"$ {shlex.join(cmd)}\n{stdout}\n{stderr}")
    out = {"exit": code, "csv": str(out_csv), "log": str(log_file), "command": shlex.join(cmd)}
    m = re.search(r"input reference check: .*?maximum residual ([0-9.eE+-]+) m, ([0-9.eE+-]+) rad", stdout)
    if m:
        out["input_reference_residual_m"], out["input_reference_residual_rad"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"^wrote .*? \(([\d,]+) rows", stdout, re.MULTILINE)
    if m:
        out["rows"] = int(m.group(1).replace(",", ""))
    m = re.search(r"^converted (\d+) Track 2 episode", stdout, re.MULTILINE)
    out["converted_episodes"] = int(m.group(1)) if m else 0
    out["error_tail"] = "" if code == 0 else " | ".join(x for x in (stderr or stdout).splitlines()[-3:] if "locally computed" not in x)[:600]
    if code == 0 and out_csv.is_file():
        out["csv_sha256"] = sha256(out_csv)
    return out


def kit_helper(args: list[str], log_file: Path) -> tuple[int, str]:
    cmd = ["nice", "-n", "15", str(PACK_PYTHON), "-I", "-B", str(KITHELPER)] + args
    try:
        p = subprocess.run(cmd, cwd=ROOT, env=kit_env(), capture_output=True, text=True, timeout=PACK_TIMEOUT)
        code, text = p.returncode, f"{p.stdout}\n{p.stderr}"
    except subprocess.TimeoutExpired:
        code, text = -9, "timeout"
    write_atomic(log_file, f"$ {shlex.join(cmd)}\n{text}")
    return code, text


def composition(fname: str, eps: list[int]) -> tuple[dict[int, dict], list[str]]:
    """episode -> the rollout of file fname: FINAL = the frozen selection; T<k> = run k (report only)."""
    out, problems = {}, []
    for e in eps:
        it = load_item(e) or {}
        if it.get("state") not in ("f0", "opus_v2"):
            problems.append(f"ep{e:02d}: no policy item ({it.get('state')})")
            continue
        if fname == "FINAL":
            s = sel_rec(e)
            if s is None:
                problems.append(f"ep{e:02d}: selection not frozen")
                continue
            k = int(s["chosen_run"])
        else:
            k = int(fname[1:])
        if k == 0:
            src = {"run": 0, "json": it["post_results_json"], "json_sha256": it["post_results_json_sha256"],
                   "parquet": it["post_results_parquet"], "parquet_sha256": it["post_results_parquet_sha256"]}
        else:
            r = run_rec(e, k) or {}
            if r.get("status") != "ok":
                problems.append(f"ep{e:02d}: run {k} is not ok ({r.get('status')})")
                continue
            src = {"run": k, "json": r["results_json"], "json_sha256": r["json_sha256"], "parquet": r["results_parquet"],
                   "parquet_sha256": r["parquet_sha256"]}
        out[e] = {**src, "episode": e, "kind": it["state"], "eval_sha256": it["eval_sha256"], "frame_ref": it["frame_ref"],
                  "frame_ref_sha256": it["frame_ref_sha256"], "object": it["object"]}  # fmt: skip
    return out, problems


def prepare_file(d: Path, comp: dict[int, dict], meta: dict) -> None:
    inside_root(d)
    (d / "results").mkdir(parents=True, exist_ok=True)
    (d / "frame_refs").mkdir(exist_ok=True)
    for e, s in sorted(comp.items()):
        copy_verified(Path(s["json"]), d / "results" / f"episode_{e:06d}.json", s["json_sha256"])
        copy_verified(Path(s["parquet"]), d / "results" / f"episode_{e:06d}.parquet", s["parquet_sha256"])
        copy_verified(Path(s["frame_ref"]), d / "frame_refs" / f"episode_{e:06d}.npz", s["frame_ref_sha256"])
    write_atomic(d / "file.json", dump({**meta, "episodes": {f"{e:02d}": s for e, s in sorted(comp.items())},
                                        "created_at": now(), "driver_sha256": sha256(Path(__file__).resolve())}))


def pack_one(d: Path, fname: str, comp: dict[int, dict], url: str, partial_eps: list[int] | None, log) -> dict:
    meta = {"file": fname, "track": TRACK, "code_commit_url": url, "partial_episodes": partial_eps,
            "note": "LOCAL_NOT_FOR_UPLOAD unless re-packed with the real commit URL (`repack`); never uploaded here"}  # fmt: skip
    prepare_file(d, comp, meta)
    config = None
    if partial_eps is not None:
        cfg = json.loads((KIT / "config/tracks.json").read_text())
        cfg["tracks"][TRACK]["scored_episodes"] = sorted(partial_eps)
        config = d / "tracks_partial.json"
        write_atomic(config, json.dumps(cfg, indent=2) + "\n")
    csv_name = "packed_PARTIAL_LOCAL_NOT_FOR_UPLOAD.csv" if partial_eps is not None else "packed_LOCAL_NOT_FOR_UPLOAD.csv"
    pk = run_packer(d, url, d / csv_name, config)
    res = {"file": fname, "pack": pk, "n_episodes": len(comp), "partial": partial_eps is not None}
    res_ok = pk["exit"] == 0 and pk.get("csv_sha256")
    resid_ok = (pk.get("input_reference_residual_m") is not None and pk["input_reference_residual_m"] < 1e-4
                and pk.get("input_reference_residual_rad", 1.0) < 1e-3)  # fmt: skip
    res["input_reference_ok"] = bool(resid_ok)
    if res_ok:
        expect = {"episodes": {str(e): s["eval_sha256"] for e, s in comp.items()}, "code_commit_url": url,
                  "eval_script_sha256": EVAL_SCRIPT_SHA, "eval_code_sha256": EVAL_CODE_SHA,
                  "packer_script_sha256": PACKER_SHA, "uploader_script_sha256": UPLOADER_SHA}  # fmt: skip
        write_atomic(d / "expect.json", dump(expect))
        code, _ = kit_helper(["validate", "--csv", str(d / csv_name), "--track", TRACK, "--out", str(d / "validate.json"),
                              "--expect", str(d / "expect.json")] + (["--partial"] if partial_eps is not None else []),
                             d / "validate.log")  # fmt: skip
        val = read_json(d / "validate.json") or {}
        res["validate"] = {"exit": code, "uploader_validate_submission": val.get("uploader_validate_submission"),
                           "problems": val.get("problems"), "pose_columns_sha256": val.get("pose_columns_sha256"),
                           "rows": val.get("rows"), "ok": val.get("ok")}  # fmt: skip
        eps_arg = "all" if partial_eps is None else " ".join(str(e) for e in sorted(partial_eps))
        code, _ = kit_helper(["score", "--csv", str(d / csv_name), "--track", TRACK, "--episodes", eps_arg,
                              "--out", str(d / "metric_code_LOCAL_NOT_FOR_UPLOAD.json")],
                             d / "score_LOCAL_NOT_FOR_UPLOAD.log")  # fmt: skip
        res["score"] = {"exit": code, "file": str(d / "metric_code_LOCAL_NOT_FOR_UPLOAD.json")}
    write_atomic(d / "pack_status.json", dump(res))
    v = res.get("validate") or {}
    log(f"{fname}{' (PARTIAL)' if partial_eps is not None else ''}: packer exit {pk['exit']}, rows {pk.get('rows')}, input "
        f"reference residual {pk.get('input_reference_residual_m')} m / {pk.get('input_reference_residual_rad')} rad, "
        f"uploader validate {v.get('uploader_validate_submission')}, provenance problems {len(v.get('problems') or [])}, "
        f"metric_code exit {(res.get('score') or {}).get('exit')} (numbers only in the LOCAL file)")  # fmt: skip
    return res


def cmd_pack(a) -> int:
    log = Logger(ROOT / "pack_driver.log")
    lockfd = os.open(inside_root(ROOT / ".pack.lock"), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another pack is running")
        return 1
    files = [f.strip() for f in (a.files or ",".join(FILES_ALL)).split(",") if f.strip()]
    if any(f not in FILES_ALL for f in files):
        print(f"--files: known {FILES_ALL}")
        return 2
    before = integrity()
    if a.partial:
        eps = [e for e in EPISODES if (load_item(e) or {}).get("state") in ("f0", "opus_v2")]
        base = inside_root(ROOT / "pack_partial" / stamp())
    else:
        missing = [e for e in EPISODES if (load_item(e) or {}).get("state") not in ("f0", "opus_v2") or sel_rec(e) is None]
        if missing:
            log(f"pack: NOT all 30 episodes have a policy + frozen selection (missing {missing}): no complete file "
                f"(`pack --partial` for a LOCAL partial test)")
            return 1
        eps, base = list(EPISODES), ROOT / "pack"
    summary = {"at": now(), "partial": a.partial, "episodes": eps, "files": {}}
    rc = 0
    for f in files:
        d = base / f
        if not a.partial and (d / "pack_status.json").is_file():
            log(f"{f}: already packed ({d}); skipped")
            summary["files"][f] = read_json(d / "pack_status.json")
            continue
        comp, problems = composition(f, eps)
        if a.partial:
            log(f"{f} (PARTIAL): {len(comp)} episodes ready" + (f"; not ready: {problems}" if problems else ""))
        elif problems:
            log(f"{f}: not packed: {problems}")
            summary["files"][f] = {"skipped": problems}
            rc = rc or (1 if f == "FINAL" else 0)
            continue
        if not comp:
            log(f"{f}: nothing to pack ({problems})")
            continue
        res = pack_one(d, f, comp, COMMIT_PLACEHOLDER, sorted(comp) if a.partial else None, log)
        summary["files"][f] = res
        ok = res["pack"]["exit"] == 0 and res["input_reference_ok"] and (a.partial or (res.get("validate") or {}).get("ok"))
        if f == "FINAL" and not ok:
            rc = 1
    diff = integrity_diff(before, integrity())
    summary["kit_stock_unchanged"] = not diff
    summary["kit_stock_diff"] = diff[:50]
    write_atomic(base / "pack_summary.json", dump(summary))
    if not a.partial:
        report_local(base, summary)
    log(f"pack done ({base}); kit/stock unchanged {not diff}; FINAL ok: {rc == 0}")
    os.close(lockfd)
    return rc


def report_local(base: Path, summary: dict) -> None:
    """summary_LOCAL_NOT_FOR_UPLOAD.{json,md}: clean numbers, REPORT ONLY (never read by any step here)."""
    out = {"note": "LOCAL_NOT_FOR_UPLOAD. Clean (Tier-1 multi-view) reference, official metric code. Report only: nothing "
                   "here changes an item, a run, a selection or a file (PREREG).", "at": now(), "files": {}}
    for f, res in summary["files"].items():
        mc = json.loads((base / f / "metric_code_LOCAL_NOT_FOR_UPLOAD.json").read_text()) \
            if (base / f / "metric_code_LOCAL_NOT_FOR_UPLOAD.json").is_file() else None
        out["files"][f] = {"csv_sha256": ((res or {}).get("pack") or {}).get("csv_sha256"),
                           "validate": (res or {}).get("validate"), "metric_code": mc}  # fmt: skip
    write_atomic(base / "summary_LOCAL_NOT_FOR_UPLOAD.json", dump(out))
    L = ["# Tier-3 f0 FINAL: LOCAL clean scores (REPORT ONLY, NOT FOR ANY DECISION)", "",
         f"Generated {now()}. Official kit metric_code against the clean reference. Nothing here was used to choose an "
         "item, a run or a file (PREREG.md).", "", "| file | csv sha256 | AUC | SP-SR | MP-SR | MPPE cm |", "|---|---|---|---|---|---|"]
    for f, x in out["files"].items():
        ov = ((x.get("metric_code") or {}).get("overall_float32")) or {}
        L.append(f"| {f} | {str(x.get('csv_sha256'))[:16]} | " + " | ".join(
            "" if ov.get(m) is None else f"{ov[m]:.4f}" for m in ("AUC", "SP-SR", "MP-SR", "MPPE")) + " |")
    write_atomic(base / "summary_LOCAL_NOT_FOR_UPLOAD.md", "\n".join(L) + "\n")


def cmd_repack(a) -> int:
    """FINAL again with the real commit URL -> upload/<sha7>/ (pose columns must equal the LOCAL FINAL's). No upload."""
    log = Logger(ROOT / "pack_driver.log")
    url = a.commit_url.strip()
    if not COMMIT_URL_RE.match(url):
        print(f"--commit-url must be https://github.com/<owner>/<repo>/commit/<40 hex>: {url!r}")
        return 2
    src = ROOT / "pack" / "FINAL"
    st = read_json(src / "pack_status.json") or {}
    if not (st.get("validate") or {}).get("ok"):
        print(f"{src}: FINAL is not packed + validated")
        return 1
    fj = read_json(src / "file.json") or {}
    sha7 = url.rsplit("/", 1)[1][:7]
    d = inside_root(ROOT / "upload" / sha7)
    if (d / "pack_status.json").exists():
        print(f"{d}: exists")
        return 1
    comp = {int(e): s for e, s in fj["episodes"].items()}
    res = pack_one(d, "FINAL", comp, url, None, log)
    v = res.get("validate") or {}
    same = v.get("pose_columns_sha256") == (st.get("validate") or {}).get("pose_columns_sha256")
    final_csv = d / f"t3_f0_final_closest_{sha7}.csv"
    if res["pack"]["exit"] == 0 and v.get("ok") and same:
        shutil.copyfile(d / "packed_LOCAL_NOT_FOR_UPLOAD.csv", final_csv)
        write_atomic(d / "UPLOAD_CANDIDATE.json", dump({"csv": str(final_csv), "sha256": sha256(final_csv), "commit_url": url,
                                                        "pose_columns_equal_local_final": same, "at": now(),
                                                        "note": "candidate only; the user decides; nothing uploaded"}))
        log(f"repack: {final_csv} ({sha256(final_csv)[:16]}), pose columns equal the LOCAL FINAL: {same}")
        return 0
    log(f"repack FAILED: pack exit {res['pack']['exit']}, validate {v.get('ok')}, pose columns equal {same}")
    return 1


# ================================================================================================ status / resolve / tests
def cmd_status(a) -> int:
    hb = read_json(ROOT / "heartbeat.json") or {}
    pid = None
    try:
        pid = int((ROOT / "driver.pid").read_text())
    except (OSError, ValueError):
        pass
    alive = pid is not None and "t3final_driver.py" in " ".join(proc_cmdline(pid)) and "run" in proc_cmdline(pid)
    print(f"driver: {'RUNNING pid ' + str(pid) if alive else 'not running'}; STOP file {'present' if (ROOT / 'STOP').exists() else 'absent'}; "
          f"heartbeat {hb.get('updated_at')} phase {hb.get('phase')} evals {hb.get('evals')} items {hb.get('items')} "
          f"selections {hb.get('selections_frozen')}/30; eta (resolved items) {hb.get('eta_hours_for_resolved_items')} h")
    for j in hb.get("running", []):
        print(f"  running: ep{j['episode']:02d} r{j['k']} try {j['try']} gpu {j.get('gpu')} {j.get('elapsed_s', '?')} s")
    for e in EPISODES:
        it = load_item(e)
        if it is None:
            print(f"  ep{e:02d} unresolved ({(hb.get('states') or {}).get(f'{e:02d}', '?')})")
            continue
        if it["state"] == "none":
            print(f"  ep{e:02d} NO POLICY: {it.get('why')}")
            continue
        cells = []
        for k in range(1, NRUNS + 1):
            r = run_rec(e, k)
            cells.append("." if r is None else ("o" if r.get("status") == "ok" else "X"))
        s = sel_rec(e) or {}
        print(f"  ep{e:02d} {it['object']:20s} {it['state']:7s} model_{it['iteration']:<6d} {''.join(cells)}"
              + (f"  own-input AUC mean {s['mean_auc']:.4f} -> run {s['chosen_run']}" if s.get("mean_auc") is not None else
                 (f"  -> run {s['chosen_run']}" if s else "")))
    ps = read_json(ROOT / "pack" / "pack_summary.json")
    if ps:
        for f, r in ps["files"].items():
            v = (r or {}).get("validate") or {}
            print(f"  pack {f}: packer exit {((r or {}).get('pack') or {}).get('exit')}, validate {v.get('uploader_validate_submission')}, ok {v.get('ok')}")
    return 0


def cmd_resolve(a) -> int:
    log = Logger(None if a.dry else ROOT / "driver.log")
    if not a.dry:
        why = prereg_ok()
        if why:
            print(f"pre-registration: {why}")
            return 1
    states = resolve_all(log, freeze_it=not a.dry)
    for e in EPISODES:
        print(f"ep{e:02d} {states[e][0]:8s} {states[e][1]}")
    return 0


def kit_validate(base: Path, csv_file: Path, eps: list[int] | None, partial: bool, label: str) -> None:
    code, text = kit_helper(["validate", "--csv", str(csv_file), "--track", TRACK, "--out", str(base / f"validate_{label}.json")]
                            + (["--partial"] if partial else []), base / f"validate_{label}.log")
    v = read_json(base / f"validate_{label}.json") or {}
    print(f"selftest validate[{label}] exit {code}: uploader {v.get('uploader_validate_submission')}; rows {v.get('rows')}; "
          f"provenance problems {v.get('problems')}")


def cmd_selftest(a) -> int:
    """CPU only, nothing frozen. (1) preflight; (2) resolve preview; (3) check_eval of every resolvable item's post-step
    run 0; (4) the selection rule on fixed numbers; (5) the uploader validation path on a full Track-2 file (Tier-2 C3r2
    raw, uploaded 10-26) as track_2_tier3; (6) a PARTIAL pack test of the post-step run-0 rollouts of every resolvable f0
    item + the opus_v2 ep23 rollout as a SYNTHETIC fallback row (merged frame references), packer + validate (expected
    to fail: partial row set) + metric_code per episode into a LOCAL file that is not printed."""
    base = inside_root(ROOT / "dryrun" / f"selftest_{stamp()}")
    base.mkdir(parents=True)
    log = Logger(base / "selftest.log")
    bad = []
    fatal, info = preflight()
    for x in info:
        log(f"preflight: {x}")
    for x in fatal:
        log(f"preflight FATAL: {x}")
    bad += fatal
    keys, keys_from = kit_metric_keys()
    log(f"metric keys {keys} ({keys_from})")
    items = {}
    for e in EPISODES:
        state, item, why = resolve_episode(e)
        log(f"resolve ep{e:02d}: {state}: {why}")
        if item is not None:
            items[e] = {**item, "state": state}
    for e, it in items.items():
        problems, info2 = check_eval(it, Path(it["post_results_json"]).parent, Path(it["post_json"]).parent / "eval_report.json", 0, keys)
        log(f"check_eval run0 ep{e:02d} ({it['kind']}): {'ok' if not problems else problems}; own-input AUC "
            f"{(info2.get('metrics') or {}).get('auc')}; config {it['config']}")
        bad += problems
        if it["kind"] == "f0" and not f0_frame_ref_listed(it["frame_ref"], it["frame_ref_sha256"]):
            bad.append(f"ep{e:02d}: frame reference not in the f0 FILES.sha256")
    for runs, want in (([0.5, 0.6, 0.7, 0.6, 0.5], 2), ([0.9, 0.9, 0.9, 0.9, 0.9], 1), ([0.1, None, 0.3, None, None], 1),
                       ([None] * 5, 0), ([0.2, 0.4, 0.3, 0.3, 0.6], 2), ([0.5, 0.7, None, None, None], 1),
                       ([None, 0.8, 0.2, 0.5, None], 4), ([0.31, 0.29, 0.30, 0.33, 0.27], 3)):
        recs = [{"k": k + 1, "status": "ok" if v is not None else "failed", "metrics": {"auc": v}} for k, v in enumerate(runs)]
        got = select_run(recs)["chosen_run"]
        log(f"select_run {runs} -> run {got} (want {want})")
        if got != want:
            bad.append(f"select_run {runs} -> {got} != {want}")
    if SYNTH_VALIDATE_CSV.is_file():
        kit_validate(base, SYNTH_VALIDATE_CSV, None, False, "synthetic_full_tier2file_as_tier3")
        v = read_json(base / "validate_synthetic_full_tier2file_as_tier3.json") or {}
        if v.get("uploader_validate_submission") != "pass":
            bad.append("uploader validate_submission did not pass the full Track-2 file as track_2_tier3")
    comp = {}
    for e, it in sorted(items.items()):
        if it["kind"] == "f0":
            comp[e] = {"run": 0, "json": it["post_results_json"], "json_sha256": it["post_results_json_sha256"],
                       "parquet": it["post_results_parquet"], "parquet_sha256": it["post_results_parquet_sha256"],
                       "episode": e, "kind": "f0", "eval_sha256": it["eval_sha256"], "frame_ref": it["frame_ref"],
                       "frame_ref_sha256": it["frame_ref_sha256"], "object": it["object"]}  # fmt: skip
    if 23 not in comp:
        o23, p23 = build_item("opus_v2", 23)
        if o23 is not None and not p23:
            comp[23] = {"run": 0, "json": o23["post_results_json"], "json_sha256": o23["post_results_json_sha256"],
                        "parquet": o23["post_results_parquet"], "parquet_sha256": o23["post_results_parquet_sha256"],
                        "episode": 23, "kind": "opus_v2 (SYNTHETIC fallback row, selftest only)",
                        "eval_sha256": o23["eval_sha256"], "frame_ref": o23["frame_ref"],
                        "frame_ref_sha256": o23["frame_ref_sha256"], "object": o23["object"]}  # fmt: skip
            log(f"opus_v2 ep23 item builds cleanly (fallback path): eval sha {o23['eval_sha256'][:16]}")
        else:
            bad.append(f"opus_v2 ep23 item: {p23}")
    if comp:
        res = pack_one(base / "partial_pack", "SELFTEST_RUN0", comp, COMMIT_PLACEHOLDER, sorted(comp), log)
        if res["pack"]["exit"] != 0 or not res["input_reference_ok"]:
            bad.append(f"partial pack: exit {res['pack']['exit']}, residual ok {res['input_reference_ok']} {res['pack'].get('error_tail')}")
        if (res.get("validate") or {}).get("uploader_validate_submission") == "pass":
            bad.append("validate passed a PARTIAL file (unexpected)")
        if (res.get("score") or {}).get("exit") != 0:
            bad.append("metric_code scoring of the partial file failed")
        if (res.get("validate") or {}).get("problems"):
            bad.append(f"partial provenance problems: {res['validate']['problems']}")
    log(f"selftest: {len(bad)} problems" + ("" if not bad else f": {bad}"))
    return 0 if not bad else 1


def cmd_dryrun(a) -> int:
    """ONE official evaluation of episode --ep's pinned file against its own input, through exactly the run path (lane
    locks, lane load, cooldown irrelevant), into dryrun/eval_<stamp>/ (never read by the real run), then a single-episode
    PARTIAL pack test of that rollout (LOCAL score not printed)."""
    e = a.ep
    base = inside_root(ROOT / "dryrun" / f"eval_{stamp()}")
    base.mkdir(parents=True)
    log = Logger(base / "dryrun.log")
    state, item, why = resolve_episode(e)
    log(f"ep{e:02d}: {state}: {why}")
    if item is None:
        return 1
    item["state"] = state
    keys, _ = kit_metric_keys()
    gpus = [int(x) for x in a.gpus.split(",")]
    deadline = time.time() + a.wait
    while True:
        try:
            st = smi()
            choice = pick_gpu(st, mps_gpus(st), [], gpus, {}, a.lane_max)
        except Exception as error:  # noqa: BLE001
            choice = f"nvidia-smi failed ({error!r})"
        if not isinstance(choice, str):
            break
        log(f"waiting for a lane: {choice}")
        if time.time() > deadline:
            log("no lane within --wait: dry run not started")
            return 1
        time.sleep(20)
    gpu, fds, taken = choice
    w = base / f"ep{e:02d}" / "try1"
    prepare_workdir(item, w)
    proc, shown = launch_eval(item, w, gpu, st["gpus"][gpu]["uuid"], fds, taken, f"DRY RUN ep{e:02d} ({item['kind']})")
    log(f"started on gpu {gpu} (pid {proc.pid}); locks {taken}")
    t0 = time.time()
    try:
        code = proc.wait(timeout=EVAL_TIMEOUT)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)  # our own session only
        code = proc.wait()
    finally:
        for fd in fds:
            os.close(fd)
    secs = time.time() - t0
    problems, info = check_eval(item, w / "results", w / "eval_report.json", code, keys)
    m = info.get("metrics") or {}
    po = item.get("post_own_input") or {}
    rec = {"episode": e, "kind": item["kind"], "gpu": gpu, "seconds": round(secs, 1), "exit": code, "problems": problems,
           "own_input": m, "post_run0_own_input": po, "pinned_sha256": item["eval_sha256"], "command": shown, **info}
    write_atomic(base / "dryrun.json", dump(rec))
    log(f"dry-run eval: exit {code}, {secs:.0f} s, problems {problems or 'none'}; own-input AUC {m.get('auc')} SP {m.get('sp_sr')} "
        f"MP {m.get('mp_sr')} MPPE {m.get('mppe')} (post step run 0: AUC {po.get('auc')} SP {po.get('sp_sr')} MP {po.get('mp_sr')})")
    if problems:
        return 1
    comp = {e: {"run": 1, "json": info["results_json"], "json_sha256": info["json_sha256"], "parquet": info["results_parquet"],
                "parquet_sha256": info["parquet_sha256"], "episode": e, "kind": item["kind"], "eval_sha256": item["eval_sha256"],
                "frame_ref": item["frame_ref"], "frame_ref_sha256": item["frame_ref_sha256"], "object": item["object"]}}
    res = pack_one(base / "pack_partial", "DRYRUN", comp, COMMIT_PLACEHOLDER, [e], log)
    return 0 if res["pack"]["exit"] == 0 and res["input_reference_ok"] and (res.get("score") or {}).get("exit") == 0 else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--runs", type=int, default=NRUNS)
    r.add_argument("--gpus", default="0,1,2,3,5,6,7", help="candidate GPUs (never 4)")
    r.add_argument("--max-parallel", type=int, default=3)
    r.add_argument("--lane-max", type=int, default=2, help="evaluations of anybody per NUMA eval-core lane (8 logical CPUs)")
    r.add_argument("--cooldown", type=float, default=35.0, help="seconds a GPU rests after one of our evals")
    r.add_argument("--poll", type=float, default=5.0)
    r.add_argument("--resolve-every", type=float, default=60.0)
    r.add_argument("--no-pack", action="store_true")
    pk = sub.add_parser("pack")
    pk.add_argument("--partial", action="store_true", help="LOCAL partial test over the episodes ready now")
    pk.add_argument("--files", default="")
    rp = sub.add_parser("repack")
    rp.add_argument("--commit-url", required=True)
    rs = sub.add_parser("resolve")
    rs.add_argument("--dry", action="store_true", help="preview only, freeze nothing")
    dr = sub.add_parser("dryrun")
    dr.add_argument("--ep", type=int, default=0)
    dr.add_argument("--gpus", default="0,1,2,3,5,6,7")
    dr.add_argument("--lane-max", type=int, default=2)
    dr.add_argument("--wait", type=float, default=3600.0)
    sub.add_parser("status")
    sub.add_parser("summary")
    sub.add_parser("selftest")
    a = p.parse_args()
    if a.cmd == "run":
        if a.runs != NRUNS or a.max_parallel < 1 or a.lane_max < 1 or a.cooldown < 0:
            p.error(f"--runs must be {NRUNS} (PREREG); bad --max-parallel / --lane-max / --cooldown")
        return Driver(a, Logger(ROOT / "driver.log")).run()
    if a.cmd == "summary":
        write_summary()
        return 0
    return {"pack": cmd_pack, "repack": cmd_repack, "resolve": cmd_resolve, "dryrun": cmd_dryrun, "status": cmd_status,
            "selftest": cmd_selftest}[a.cmd](a)  # fmt: skip


if __name__ == "__main__":
    raise SystemExit(main())
