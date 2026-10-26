#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers of the Tier-1 best-set tools (fork_train.py, eval_watch.py, trials_eval.py, check_items.py).

Nothing here knows a machine. Every path, GPU list and CPU list is a site setting, read from the environment and,
optionally, from a KEY=value file given with --site (a copy of site.env.example). Values from the file can use
${VAR} references; a variable that is set in the environment wins over the file.

Item configs (configs/items/<item>.json) write paths with two placeholders:
  <MOTION>  the derived motion data root (HUMAN_MOTION_DATA_DIR, site V2D_MOTION)
  <RUNS>    the runs root that holds the transferred training checkpoints (site T1_RUNS)
"""

from __future__ import annotations

import ast
import datetime as dt
import fcntl
import glob
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
FC_DIR = HERE.parents[2]  # robotic_grounding/flash_chord of this checkout
CONFIGS = HERE / "configs"
ITEMS_DIR = CONFIGS / "items"
ARMS_FILE = CONFIGS / "arms.json"
BEST_SET = HERE / "best_set"

STOCK_COMMIT = "8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef"
EVAL_SCRIPT_SHA256 = "160d24f5fbeb70ccd569ca4192056f1892b2a9f2999066db9d8279461648fdf1"  # evaluate_policy.py
EVAL_CODE_SHA256 = "b1b3ebe3ae4629d1687a83f420149e3f6fdfc154d70acad55c12fc8fa69811a1"  # recorded in result parquets
EVAL_ALL_SHA256 = "657295921e820a464160039ad9a65ba92a683108a99dcb197290ae4147363197"  # scripts/track2/eval_all.py
EXPORTER = "flash_chord.training.ppo.coordinate.export"
EXPORT_TOLERANCE = 1e-4
METRICS = ("auc", "sp_sr", "mp_sr", "mppe")
REPORT_NAMES = {"auc": "add_auc", "sp_sr": "spider_sr", "mp_sr": "maniptrans_sr", "mppe": "mppe_cm"}
FALLBACK_KEYS = {"auc": "add_auc", "sp_sr": "spider_sr_uncentered", "mp_sr": "maniptrans_sr", "mppe": "mean_per_point_error_cm"}
DROP_ENV_PREFIXES = ("CUDA_MPS_", "WANDB_")
DROP_ENV_KEYS = {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONINSPECT", "CUDA_VISIBLE_DEVICES", "FLASH_CHORD_CACHE_DIR"}


# ============================================================================================ small helpers
def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path) -> dict | None:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def write_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def dump(data) -> str:
    return json.dumps(data, indent=1, sort_keys=False, default=str) + "\n"


def fnum(value):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def override_key(override: str) -> str:
    """Hydra override key: 'objective.x' for 'objective.x=1', '+objective.x=1' and '++objective.x=1'."""
    return override.split("=", 1)[0].lstrip("+~")


def merge_overrides(base: list[str], extra: list[str]) -> list[str]:
    """base, then extra; an extra override replaces a base override of the same key (in place)."""
    out = list(base)
    for o in extra:
        k = override_key(o)
        hit = [i for i, b in enumerate(out) if override_key(b) == k]
        if hit:
            out[hit[0]] = o
            for i in reversed(hit[1:]):
                del out[i]
        else:
            out.append(o)
    return out


# ============================================================================================ site settings
@dataclass
class Site:
    values: dict = field(default_factory=dict)

    def get(self, key: str, default=None):
        v = self.values.get(key)
        return default if v in (None, "") else v

    def path(self, key: str, default=None) -> Path | None:
        v = self.get(key, default)
        return None if v in (None, "") else Path(os.path.expanduser(str(v))).absolute()

    def need(self, *keys: str) -> None:
        missing = [k for k in keys if self.get(k) in (None, "")]
        if missing:
            raise SystemExit(f"set {', '.join(missing)} (environment or --site file; see site.env.example)")

    # --- the settings every tool uses
    @property
    def work(self) -> Path:
        self.need("T1_WORK")
        return self.path("T1_WORK")

    @property
    def runs(self) -> Path:
        return self.path("T1_RUNS") or self.work / "runs"

    @property
    def exports(self) -> Path:
        return self.path("T1_EXPORTS") or self.work / "exports"

    @property
    def motion(self) -> Path:
        self.need("V2D_MOTION")
        return self.path("V2D_MOTION")

    @property
    def runtime(self) -> Path:
        self.need("V2D_RUNTIME")
        return self.path("V2D_RUNTIME")

    @property
    def kit(self) -> Path:
        self.need("V2D_KIT")
        return self.path("V2D_KIT")

    @property
    def data(self) -> Path:
        self.need("V2D_DATA")
        return self.path("V2D_DATA")

    @property
    def fc_python(self) -> str:
        return self.get("FC_PYTHON") or sys.executable

    @property
    def kit_python(self) -> str:
        self.need("KIT_PYTHON")
        return self.get("KIT_PYTHON")

    @property
    def code(self) -> Path:
        """robotic_grounding/flash_chord of the training tree (default: this checkout)."""
        return self.path("T1_CODE") or FC_DIR

    @property
    def gpus(self) -> list[int] | None:
        v = self.get("GPUS")
        return None if v is None else parse_int_list(v)

    @property
    def exclude_gpus(self) -> set[int]:
        return set(parse_int_list(self.get("EXCLUDE_GPUS", "")))

    @property
    def refuse_mps(self) -> bool:
        return str(self.get("REFUSE_MPS", "0")) == "1"

    @property
    def eval_cpus(self) -> dict[int, str]:
        return parse_gpu_cpus(self.get("EVAL_CPUS", ""))

    @property
    def extra_lockdirs(self) -> list[Path]:
        return [Path(p) for p in str(self.get("EVAL_LOCKDIRS", "")).split(":") if p]

    @property
    def min_free_mb(self) -> int:
        return int(self.get("EVAL_MIN_FREE_MB", "5632"))

    @property
    def lockdir(self) -> Path:
        return self.path("EVAL_LOCKDIR") or self.work / "evallock"


def parse_int_list(text) -> list[int]:
    return [int(x) for x in re.split(r"[,\s]+", str(text).strip()) if x != ""]


def parse_gpu_cpus(text: str) -> dict[int, str]:
    """'0,1,2=48-51,152-155;3,5,6,7=100-103,204-207' -> {0: '48-51,152-155', ..., 7: '100-103,204-207'}."""
    out: dict[int, str] = {}
    for part in str(text or "").split(";"):
        part = part.strip()
        if not part:
            continue
        gpus, _, cpus = part.partition("=")
        if not cpus:
            raise SystemExit(f"bad EVAL_CPUS entry {part!r}: expected <gpus>=<cpu list>")
        for g in parse_int_list(gpus):
            out[g] = cpus.strip()
    return out


def _expand(text: str, scope: dict) -> str:
    def rep(m):
        name = m.group(1) or m.group(2)
        return str(scope.get(name, os.environ.get(name, "")))

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", rep, text)


def load_site(path: str | os.PathLike | None = None) -> Site:
    values: dict[str, str] = {}
    if path:
        for raw in Path(path).read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            key, eq, val = line.partition("=")
            if not eq or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key.strip()):
                continue
            val = val.strip()
            if " #" in val and not val.startswith(("'", '"')):
                val = val.split(" #", 1)[0].rstrip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in "'\"":
                val = val[1:-1]
            values[key.strip()] = _expand(val, values)
    for key in list(values) + [k for k in os.environ if k.startswith(("T1_", "V2D_", "EVAL_"))] + [
            "FC_PYTHON", "KIT_PYTHON", "GPUS", "EXCLUDE_GPUS", "REFUSE_MPS", "WANDB_MODE"]:
        if os.environ.get(key) not in (None, ""):
            values[key] = os.environ[key]
    return Site(values)


def add_site_arg(parser) -> None:
    parser.add_argument("--site", help="KEY=value settings file (a copy of site.env.example); the environment wins")


# ============================================================================================ item configs
def load_items() -> dict[str, dict]:
    items = {}
    for path in sorted(ITEMS_DIR.glob("*.json")):
        cfg = json.loads(path.read_text())
        items[cfg["item"]] = cfg
    return items


def load_arms() -> dict[str, dict]:
    return json.loads(ARMS_FILE.read_text())["arms"]


def set_items(items: dict[str, dict]) -> list[dict]:
    """The 30 items of the current best set (one per episode), ordered by episode."""
    out = sorted((c for c in items.values() if c.get("in_set")), key=lambda c: c["episode"])
    if [c["episode"] for c in out] != list(range(30)):
        raise SystemExit(f"configs: the set must have exactly one item per episode 0..29, got {[c['item'] for c in out]}")
    return out


def resolve(text: str, site: Site) -> str:
    out = str(text)
    if "<MOTION>" in out:
        out = out.replace("<MOTION>", str(site.motion))
    if "<RUNS>" in out:
        out = out.replace("<RUNS>", str(site.runs))
    return out


def task_parquet(cfg: dict, site: Site) -> Path:
    return site.motion / "whole_body/soma" / f"sequence_id={cfg['sequence_id']}" / "robot_name=g1/data.parquet"


def item_of(name: str, items: dict[str, dict]) -> dict:
    key = name if name in items else (f"ep{int(name):02d}" if name.isdigit() else name)
    if key not in items:
        raise SystemExit(f"unknown item {name!r}; known: {', '.join(items)}")
    return items[key]


# ============================================================================================ environments
def fc_site_packages(python: str) -> Path | None:
    prefix = Path(python).resolve().parent.parent
    hits = sorted(glob.glob(str(prefix / "lib/python3*/site-packages")))
    if hits:
        return Path(hits[-1])
    try:
        out = subprocess.run([python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                             capture_output=True, text=True, timeout=60).stdout.strip()
        return Path(out) if out else None
    except (OSError, subprocess.SubprocessError):
        return None


def fc_env(site: Site, gpu: int | None, updates: dict | None = None) -> dict:
    """Child environment of a FlashCHORD step, as repro/env/fc_env.sh: the env's bin first on PATH, its pip CUDA 13 /
    cuDNN / NCCL / NVSHMEM libraries on LD_LIBRARY_PATH (onnxruntime-gpu), no JAX preallocation, motion data root,
    no MPS client variables, no W&B (unless WANDB_MODE is set in the site), no bytecode."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(DROP_ENV_PREFIXES) and k not in DROP_ENV_KEYS}
    py = site.fc_python
    env["PATH"] = f"{Path(py).parent}:{env.get('PATH', '/usr/bin:/bin')}"
    sp = fc_site_packages(py)
    libs = [str(sp / "nvidia" / d / "lib") for d in ("cu13", "cudnn", "nccl", "nvshmem") if sp and (sp / "nvidia" / d / "lib").is_dir()]
    if libs:
        env["LD_LIBRARY_PATH"] = ":".join(libs + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
    env.update(XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONDONTWRITEBYTECODE="1", CUDA_DEVICE_ORDER="PCI_BUS_ID",
               HUMAN_MOTION_DATA_DIR=str(site.motion), WANDB_MODE=site.get("WANDB_MODE", "disabled"))
    env["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    if updates:
        env.update({k: str(v) for k, v in updates.items()})
    return env


def kit_env(extra: dict | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    if extra:
        env.update(extra)
    return env


def git_head(path: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


# ============================================================================================ GPUs and lanes
def smi() -> dict:
    """nvidia-smi view: {'gpus': {index: {uuid, free_mb, total_mb}}, 'apps': [{pid, name, gpu, used_mb}]}."""
    out = {"gpus": {}, "apps": []}
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.free,memory.total", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=60)
    by_uuid = {}
    for line in q.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4 and parts[0].isdigit():
            g = int(parts[0])
            out["gpus"][g] = {"uuid": parts[1], "free_mb": int(float(parts[2])), "total_mb": int(float(parts[3]))}
            by_uuid[parts[1]] = g
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,gpu_uuid,used_memory",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60)
    for line in q.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4 and parts[0].isdigit():
            out["apps"].append({"pid": int(parts[0]), "name": parts[1], "gpu": by_uuid.get(parts[2]),
                                "used_mb": int(parts[3]) if parts[3].isdigit() else None})
    return out


def mps_gpus(state: dict) -> set[int]:
    return {a["gpu"] for a in state["apps"] if a["gpu"] is not None and "mps-server" in a["name"]}


def usable_gpus(site: Site, state: dict) -> tuple[list[int], list[str]]:
    """GPUs the site allows (GPUS, minus EXCLUDE_GPUS, minus MPS GPUs when REFUSE_MPS=1), with the reasons."""
    allowed = site.gpus if site.gpus is not None else sorted(state["gpus"])
    notes, out = [], []
    mps = mps_gpus(state) if site.refuse_mps else set()
    for g in allowed:
        if g in site.exclude_gpus:
            notes.append(f"gpu{g}: excluded (EXCLUDE_GPUS)")
        elif g not in state["gpus"]:
            notes.append(f"gpu{g}: not visible to nvidia-smi")
        elif g in mps:
            notes.append(f"gpu{g}: an MPS server runs on it (REFUSE_MPS=1)")
        else:
            out.append(g)
    return out, notes


class Lanes:
    """One evaluation per GPU: flock of <lockdir>/gpu<g>.lock (created), plus gpu<g>.lock of every EVAL_LOCKDIRS
    directory that already has one (opened read-only, never created). All or nothing, never blocking. The locks are
    inherited by the evaluation process (pass_fds), so a lane stays taken while its evaluation lives."""

    def __init__(self, site: Site):
        self.own = site.lockdir
        self.extra = site.extra_lockdirs

    def files(self, gpu: int) -> list[Path]:
        return [self.own / f"gpu{gpu}.lock"] + [d / f"gpu{gpu}.lock" for d in self.extra if (d / f"gpu{gpu}.lock").exists()]

    def take(self, gpu: int) -> list[int] | None:
        fds = []
        self.own.mkdir(parents=True, exist_ok=True)
        for i, path in enumerate(self.files(gpu)):
            try:
                fd = os.open(path, (os.O_RDWR | os.O_CREAT) if i == 0 else os.O_RDONLY, 0o644)
            except OSError:
                self.release(fds)
                return None
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(fd)
                self.release(fds)
                return None
            fds.append(fd)
        return fds

    @staticmethod
    def release(fds: list[int]) -> None:
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass


def taskset_prefix(site: Site, gpu: int | None, nice: int = 10) -> list[str]:
    prefix = ["nice", "-n", str(nice)]
    cpus = site.eval_cpus.get(gpu) if gpu is not None else None
    if cpus:
        prefix += ["taskset", "-c", cpus]
    return prefix


# ============================================================================================ kit metrics, seeds
def kit_metric_keys(kit: Path) -> tuple[dict, str]:
    """ours -> results-JSON key from the kit's v2dlb/report_schema.py REPORT_METRIC_KEY (parsed, never imported)."""
    try:
        for node in ast.parse((kit / "v2dlb/report_schema.py").read_text()).body:
            targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
            if any(getattr(t, "id", None) == "REPORT_METRIC_KEY" for t in targets):
                table = ast.literal_eval(node.value)
                return {ours: table[name] for ours, name in REPORT_NAMES.items()}, "kit v2dlb/report_schema.py"
        raise KeyError("REPORT_METRIC_KEY not found")
    except Exception as error:  # noqa: BLE001
        return dict(FALLBACK_KEYS), f"built-in fallback ({error!r})"


def resolved_config(path: Path) -> dict:
    """The training config stored in a FlashCHORD safetensors file (metadata user.resolved_config_json)."""
    with Path(path).open("rb") as fp:
        n = struct.unpack("<Q", fp.read(8))[0]
        header = json.loads(fp.read(n))
    return json.loads(header["__metadata__"]["user.resolved_config_json"])


def _dig(c, *keys):
    for k in keys:
        if not isinstance(c, dict) or k not in c:
            return "MISSING"
        c = c[k]
    return c


def _const_from(path: Path, name: str):
    try:
        for node in ast.parse(path.read_text()).body:
            targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
            if any(getattr(t, "id", None) == name for t in targets) and getattr(node, "value", None) is not None:
                return ast.literal_eval(node.value)
    except (OSError, ValueError, SyntaxError):
        pass
    return None


def seed_signature(file: Path, runtime: Path, kit: Path) -> str:
    """The default seeds an official evaluation of this exported file uses (the evaluator has no seed option)."""
    c = resolved_config(file)
    bd = _dig(c, "env", "action", "body_delay")
    seed = bd.get("seed") if isinstance(bd, dict) else "MISSING"
    m = re.search(r'config\["env"\]\["reset"\]\["seed"\]\s*=\s*(\d+)', (runtime / "src/flash_chord/evaluation/harness.py").read_text())
    reset = int(m.group(1)) if m else _dig(c, "env", "reset", "seed")
    return (f"delay={seed}(+g0..4,{bd.get('min_steps') if isinstance(bd, dict) else '?'}-"
            f"{bd.get('max_steps') if isinstance(bd, dict) else '?'}) reset={reset} "
            f"scale_seed={_dig(c, 'scene', 'object_scale_seed')}({_dig(c, 'scene', 'object_scale_min')}/"
            f"{_dig(c, 'scene', 'object_scale_max')}) wrench={_dig(c, 'env', 'objective', 'wrench_basis_seed')} "
            f"vertex={_const_from(runtime / 'src/flash_chord/evaluation/metrics.py', 'ADD_VERTEX_SEED')}/"
            f"{_const_from(kit / 'v2dlb/vertices.py', 'ADD_VERTEX_SEED')} ppo={_dig(c, 'training', 'seed')}")


# ============================================================================================ evaluation
def eval_all_command(site: Site, work: Path, arm: str, episode: int, iteration: int, gpu: int, export: bool) -> list[str]:
    """scripts/track2/eval_all.py of the training tree, one episode, one world, frame 0, no help; with export=True the
    work dir's checkpoint is first exported to the stock format (tolerance 1e-4) by this tree's exporter."""
    code = site.code
    cmd = [site.fc_python, "-B", str(code / "scripts/track2/eval_all.py"), "--arm-dir", str(work), "--arm", arm,
           "--export", "yes" if export else "no", "--iteration", str(iteration), "--episodes", str(episode),
           "--gpus", str(gpu), "--jobs-per-gpu", "1", "--world-count", "1", "--skip-pack", "--kit-root", str(site.kit),
           "--stock-runtime", str(site.runtime), "--dataset-root", str(site.data),
           "--manifest", str(site.motion / "whole_body/soma/track2_manifest.json"), "--pack-python", site.kit_python,
           "--results", str(work / "results"), "--code", str(code), "--python", site.fc_python]
    if export:
        cmd += ["--export-jobs", "1", "--export-tolerance", repr(EXPORT_TOLERANCE)]
    return cmd


def export_command(site: Site, source: Path, out_dir: Path) -> tuple[list[str], dict]:
    """The E2 exporter of the training tree on the CPU, stock-runtime equivalence check at tolerance 1e-4."""
    cmd = [site.fc_python, "-B", "-m", EXPORTER, "--input", str(source), "--output", str(out_dir / source.name),
           "--stock-runtime", str(site.runtime), "--report", str(out_dir / "export_report.json"),
           "--tolerance", repr(EXPORT_TOLERANCE)]
    return cmd, {"PYTHONPATH": str(site.code / "src"), "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": ""}


def check_export_report(report_path: Path, source_sha: str | None = None) -> tuple[bool, list[str], dict]:
    rep = read_json(report_path)
    if rep is None:
        return False, ["no export_report.json"], {}
    stock = rep.get("stock_runtime") or {}
    info = {"status": rep.get("status"), "max_rel_diff": rep.get("max_rel_diff"), "max_abs_diff": rep.get("max_abs_diff"),
            "source_sha256": rep.get("source_sha256"), "destination_sha256": rep.get("destination_sha256"),
            "stock_verified": stock.get("verified"), "eval_code_sha256": stock.get("eval_code_sha256")}
    problems = []
    if rep.get("status") != "ok":
        problems.append(f"export status {rep.get('status')} {rep.get('failures') or ''}".strip())
    if stock.get("verified") is not True:
        problems.append(f"stock runtime verified={stock.get('verified')}")
    if stock.get("eval_code_sha256") not in (None, EVAL_CODE_SHA256):
        problems.append(f"stock eval_code_sha256 {stock.get('eval_code_sha256')} is not the evaluator build of the line")
    rel = fnum(rep.get("max_rel_diff"))
    if rel is None or rel > EXPORT_TOLERANCE:
        problems.append(f"max_rel_diff {rep.get('max_rel_diff')} > {EXPORT_TOLERANCE}")
    if source_sha and rep.get("source_sha256") != source_sha:
        problems.append("export report source_sha256 != the checkpoint's sha256")
    return not problems, problems, info


def parquet_problems(pp: Path, sha: str, step_count) -> tuple[list[str], dict]:
    import pyarrow.parquet as pq

    md = {k.decode(errors="replace"): v.decode(errors="replace")
          for k, v in (pq.ParquetFile(pp).schema_arrow.metadata or {}).items() if not k.startswith(b"pandas")}
    out = []
    if md.get("flash_chord.checkpoint_sha256") != sha:
        out.append("parquet checkpoint_sha256 != the evaluated file")
    if md.get("flash_chord.eval_script_sha256") != EVAL_SCRIPT_SHA256:
        out.append(f"parquet eval_script_sha256 {md.get('flash_chord.eval_script_sha256')} is not the line's evaluator")
    if md.get("flash_chord.eval_code_sha256") != EVAL_CODE_SHA256:
        out.append(f"parquet eval_code_sha256 {md.get('flash_chord.eval_code_sha256')} is not the line's evaluator build")
    if str(md.get("flash_chord.step_count")) != str(step_count):
        out.append(f"parquet step_count {md.get('flash_chord.step_count')} != JSON {step_count}")
    valid = pq.read_table(pp, columns=["environment_valid"]).column("environment_valid").to_pylist()
    if not valid or not all(valid):
        out.append("environment_valid=false rows")
    return out, {k.replace("flash_chord.", ""): md.get(k) for k in md if k.startswith("flash_chord.")}


def check_results(episode: int, sequence_id: str, results: Path, report: Path | None, exit_code, keys: dict,
                  expect_sha: str | None = None, expect_file: Path | None = None,
                  check_parquet: bool = True) -> tuple[list[str], dict]:
    """Problems of one official evaluation (empty = ok) and its facts: metrics from the evaluator's own results JSON
    (kit key mapping), checkpoint digest, Tier-1 sequence, frame-0 / one-world / explicit-reset start, rollout parquet
    digests and validity."""
    problems: list[str] = []
    if exit_code not in (0, None):
        problems.append(f"exit {exit_code}")
    rep_row = None
    if report is not None:
        rep = read_json(report) or {}
        rep_row = next((r for r in rep.get("episodes", []) if str(r.get("episode")) == str(episode)), None)
        if rep_row is None:
            problems.append("no eval_report.json row for the episode")
        elif rep_row.get("status") != "evaluated":
            problems.append(f"eval_report status {rep_row.get('status')}")
    jp, pp = results / f"episode_{episode:06d}.json", results / f"episode_{episode:06d}.parquet"
    info: dict = {"results_json": str(jp), "results_parquet": str(pp)}
    if rep_row is not None:
        info["export_max_rel_diff"] = rep_row.get("export_max_rel_diff")
        info["export_stock_verified"] = rep_row.get("export_stock_verified")
    js = read_json(jp)
    if js is None:
        problems.append("no results JSON")
        return problems, info
    m = js.get("metrics") or {}
    info["metrics"] = {k: fnum(m.get(src)) for k, src in keys.items()}
    info.update(chord_sr=fnum(m.get("chord_sr")), checkpoint_sha256=js.get("checkpoint_sha256"),
                checkpoint=js.get("checkpoint"), parquet_reference=js.get("parquet"), step_count=js.get("step_count"),
                json_sha256=sha256(jp))
    if any(v is None for v in info["metrics"].values()):
        problems.append(f"metrics missing ({info['metrics']})")
    if expect_sha and js.get("checkpoint_sha256") != expect_sha:
        problems.append("results JSON checkpoint_sha256 != the evaluated file")
    if expect_file is not None and os.path.realpath(str(js.get("checkpoint", ""))) != os.path.realpath(expect_file):
        problems.append(f"evaluated checkpoint path {js.get('checkpoint')} is not {expect_file}")
    if rep_row is not None and rep_row.get("export_stock_verified") not in (None, True):
        problems.append(f"export stock_verified={rep_row.get('export_stock_verified')}")
    if f"/whole_body/soma/sequence_id={sequence_id}/" not in str(js.get("parquet") or ""):
        problems.append(f"evaluated reference {js.get('parquet')} is not the Tier-1 sequence {sequence_id}")
    for key, want in (("start_frame", 0), ("motion_start_frame", 0), ("world_count", 1), ("reset_mode", "explicit")):
        if js.get(key) != want:
            problems.append(f"{key}={js.get(key)!r}, expected {want!r}")
    if js.get("non_finite_worlds"):
        problems.append(f"non_finite_worlds {js.get('non_finite_worlds')}")
    if check_parquet:
        if not pp.is_file():
            problems.append("no rollout parquet")
        else:
            try:
                pprob, meta = parquet_problems(pp, js.get("checkpoint_sha256"), js.get("step_count"))
                problems += pprob
                info["parquet_meta"] = meta
                info["parquet_sha256"] = sha256(pp)
            except Exception as error:  # noqa: BLE001
                problems.append(f"parquet unreadable ({error!r})")
    return problems, info


def checkpoints_in(run: Path) -> list[int]:
    out = []
    try:
        names = os.listdir(run)
    except OSError:
        return []
    for name in names:
        m = re.fullmatch(r"model_(\d+)\.safetensors", name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)
