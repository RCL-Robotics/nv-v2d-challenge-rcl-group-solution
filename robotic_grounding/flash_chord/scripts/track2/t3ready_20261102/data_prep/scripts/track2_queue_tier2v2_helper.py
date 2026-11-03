#!/usr/bin/env python3
"""track2_queue_tier2v2_helper.py: checks and slot decisions of scripts/track2_queue_tier2v2.sh, the LOW-PRIORITY
Tier-2 queue for the single Tier-2 arm C3r2_tier2_k3_stretch90k (user decision 2026-10-21): C3r2 on the NOISY Tier-2
body + K3 object (data/human_motion_data_tier2_k3 = byte copy of kalman/eval/data_k3 + the Tier-2 manifest) with the
STRETCHED RSI curriculum (RECIPE_EXTRA must be exactly ARM_RECIPE_EXTRA: reset.deadline=70312 reset.zero_ramp=7812,
twice C3r2's; help forced to zero by iteration 78,124; help-zero tail kept at 5860) and a 90k cap. The corrected-body
roots (human_motion_data_tier2_armA, _armA_v2) are refused. (Until 2026-10-21 04:36 this queue was set up for arm A,
C3r2_tier2_armA_90k, which never started.) Stdlib only, plus pyarrow for the input check
(run it with conda_envs/v2d/bin/python -I -B). CPU only. It never signals, starts or stops anything.

Fill rule (user 2026-10-21), applied per GPU every scheduling loop with the settings re-read:
  a job may start on GPU g only if
    - g is listed in GPUS (default 0 1 2 3 5 6 7: GPU 4 is a teammate's; add it by hand if wanted),
    - g has no MPS presence (an MPS server / client among its compute apps, or an MPS control daemon naming it),
    - trainings(g) + our launches not visible yet < MAX_TRAIN_PER_GPU (default 3),
    - free(g) >= JOB_MB + MIN_FREE_AFTER_MB + ramp debt(g)   (16.5 GB + 20 GB by default; ramp debt = for our own
      trainings there younger than RAMP_S, JOB_MB minus their memory now; JOB_MB for a launch not visible yet).
  trainings(g) counts EVERY training process on g, whoever runs it: a compute app classified as a training (a
  train*.py script / train_rl_games / a -m module named train*, or a RUNIN_QUEUE / TIER2V2_QUEUE marker), plus
  python train_rl.py processes not on nvidia-smi yet whose environment names g (TIER2V2_GPU, RUNIN_GPU, TIER2_GPU or
  a single-device CUDA_VISIBLE_DEVICES index / UUID). Evaluation processes (evaluate_policy, run_policy_evaluation,
  eval_all.py, RUNIN_EVAL / TIER2V2_EVAL markers) never count; an unclassified compute app counts only once it is
  older than UNKNOWN_TRAIN_AGE_S (default 30 min: short jobs do not count, long-lived ones do). Their memory always
  counts through free(g).
  Global gates (any one blocks every start): PAUSE file, HOLD_armA file, training input not ready, RECIPE_EXTRA not
  the arm's stretched-curriculum overrides, arm lock mismatch
  (DATA / RECIPE_EXTRA / input files changed after the first start), the rolling-start queue
  (fc_runs/queue_runin/queue_state.json) having a waiting or retrying task or a state older than RUNIN_STALE_S while
  not finished, container pids headroom below PIDS_HEADROOM.
  Among the allowed GPUs the one with the fewest trainings, then the most free memory, is chosen; one start per loop.

Subcommands (QDIR / RUNS / EVROOT from TIER2V2_QDIR / TIER2V2_RUNS / TIER2V2_EVROOT, defaults below):
  jobs-init <out tsv>      write the job table (priority order; support rule of the Tier-2 root)
  tick [--no-write] [--state F] [--waiting F] [--failed F] [--pid N]
                           settings, input check, gates, per-GPU decision and the post-training work list, as lines
                           (SETTING / SETTINGS_ERR / INPUT / GATE / GPU / CHOICE / POST_TODO / POST_RUNNING); unless
                           --no-write also writes queue_state.json, status_line.txt and the input-check cache
  ready [--no-write]       training-input readiness only (exit 0 ready, 1 not ready)
  lock-arm                 record DATA, RECIPE_EXTRA and the input fingerprint + sha256s at the first start (arm_lock.json)
  cpuslice <gpu> [<excluded cpu list>]   "cpus nthreads note" for one new training (MAX_CORES / MIN_CORES settings)
  done <episode>           exit 0 when the episode's training is finished (final.json), printing why
  selftest                 unit tests of the slot rule and the gates on synthetic GPU states
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ME = Path("/root/autodl-tmp/haoran.zhang")
ARM = "C3r2_tier2_k3_stretch90k"
# the arm's recipe change (user 2026-10-21): twice C3r2's RSI deadline / zero ramp; nothing else (gate "recipe")
ARM_RECIPE_EXTRA = "reset.deadline=70312 reset.zero_ramp=7812"
EXP = "g1_recon_body_coord_rsi_r2"
ITERS = 90000
QDIR = Path(os.environ.get("TIER2V2_QDIR", ME / "fc_runs/queue_tier2v2"))
RUNS = Path(os.environ.get("TIER2V2_RUNS", ME / "fc_runs" / ARM))
EVROOT = Path(os.environ.get("TIER2V2_EVROOT", ME / "fc_runs/eval_tier2v2" / ARM))
TIER1_ROOT = ME / "data/human_motion_data"
TIER2_ROOT = ME / "data/human_motion_data_tier2"
OLD_ARMA_ROOT = ME / "data/human_motion_data_tier2_armA"  # first body correction: must NOT be trained on
ARMA_V2_ROOT = ME / "data/human_motion_data_tier2_armA_v2"  # rebuilt corrected body: not this arm's input either
FORBIDDEN_ROOTS = {"Tier-1 root": TIER1_ROOT, "raw Tier-2 root (no raw-input control)": TIER2_ROOT,
                   "old arm-A root (first body correction)": OLD_ARMA_ROOT,
                   "arm-A v2 root (corrected body; this arm trains on the noisy body)": ARMA_V2_ROOT}
DEFAULT_DATA = ME / "data/human_motion_data_tier2_k3"
DEFAULT_FRAME_REFS = ME / "tmp_analysis/tier2/kalman/eval/frame_refs_k3"
RUNIN_STATE = ME / "fc_runs/queue_runin/queue_state.json"
C3R2_TIER2 = ME / "fc_runs/C3r2_tier2"
CAP_EPISODES = [1, 3, 4, 5, 7, 13, 14, 15, 18, 19, 21, 22, 24, 25, 27, 28, 29]  # final iteration 44922 in C3r2_tier2
CAP_ITERATION = 44922
RAMP_S = int(os.environ.get("TIER2V2_RAMP_S", "900"))
EVAL_CORES = ("48-51,152-155", "100-103,204-207")  # the eval lane of each socket (as runin_helper.EVAL_CORES)
TRAIN_POOLS = ("0-47", "52-99")  # physical cores a training slice may use (eval lanes excluded)
TICK = os.sysconf("SC_CLK_TCK")
READY_RE = re.compile(r"^READY")

DEFAULTS = {
    "MAX_TRAIN_PER_GPU": "3",
    "MIN_FREE_AFTER_MB": "20480",
    "JOB_MB": "16896",
    "GPUS": "0 1 2 3 5 6 7",
    "DATA": str(DEFAULT_DATA),
    "FRAME_REFS": str(DEFAULT_FRAME_REFS),
    "RECIPE_EXTRA": "",
    "MAX_CORES": "4",
    "MIN_CORES": "2",
    "EVAL_MIN_FREE_MB": "4096",
    "UNKNOWN_TRAIN_AGE_S": "1800",
    "RUNIN_STALE_S": "900",
    "PIDS_HEADROOM": "3000",
}
QUEUE_OWNED_KEYS = re.compile(r"^\+*(experiment|task\.parquet|output_dir|reset\.seed|training\.max_iterations|training\.resume|"
                              r"training\.checkpoint|scene\.support|hydra\..*|logging\..*)=")


# ============================================================================================== settings
def parse_settings(text: str) -> tuple[dict | None, str]:
    """settings.conf text -> (settings with typed values, '') or (None, error)."""
    raw = dict(DEFAULTS)
    for n, line in enumerate(text.splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or key not in DEFAULTS:
            return None, f"line {n}: unknown or malformed entry {line!r} (keys: {', '.join(DEFAULTS)})"
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        raw[key] = value
    s: dict = {}
    try:
        for key in ("MAX_TRAIN_PER_GPU", "MIN_FREE_AFTER_MB", "JOB_MB", "MAX_CORES", "MIN_CORES", "EVAL_MIN_FREE_MB",
                    "UNKNOWN_TRAIN_AGE_S", "RUNIN_STALE_S", "PIDS_HEADROOM"):
            if not re.fullmatch(r"\d+", raw[key]):
                return None, f"{key}={raw[key]!r} is not a non-negative integer"
            s[key] = int(raw[key])
    except ValueError as error:
        return None, str(error)
    if s["MAX_TRAIN_PER_GPU"] > 9:
        return None, f"MAX_TRAIN_PER_GPU={s['MAX_TRAIN_PER_GPU']} > 9"
    if not 1024 <= s["JOB_MB"] <= 98304:
        return None, f"JOB_MB={s['JOB_MB']} outside 1024..98304"
    if s["MIN_FREE_AFTER_MB"] > 98304:
        return None, f"MIN_FREE_AFTER_MB={s['MIN_FREE_AFTER_MB']} > 98304"
    if not 1 <= s["MAX_CORES"] <= 26 or not 1 <= s["MIN_CORES"] <= s["MAX_CORES"]:
        return None, f"need 1 <= MIN_CORES ({s['MIN_CORES']}) <= MAX_CORES ({s['MAX_CORES']}) <= 26"
    if s["RUNIN_STALE_S"] < 60:
        return None, f"RUNIN_STALE_S={s['RUNIN_STALE_S']} < 60"
    gpus = raw["GPUS"].replace(",", " ").split()
    if any(not re.fullmatch(r"[0-7]", g) for g in gpus) or len(set(gpus)) != len(gpus):
        return None, f"GPUS={raw['GPUS']!r}: list distinct GPU indices 0-7 separated by spaces"
    s["GPUS"] = sorted(int(g) for g in gpus)
    for key in ("DATA", "FRAME_REFS"):
        if not raw[key].startswith("/") or any(c.isspace() for c in raw[key]):
            return None, f"{key}={raw[key]!r} must be an absolute path without spaces"
        s[key] = raw[key].rstrip("/")
    extra = raw["RECIPE_EXTRA"].split()
    for token in extra:
        if not re.fullmatch(r"\+{0,2}[A-Za-z_][\w.]*=[^\s'\"\\$`;|&<>()]+", token):
            return None, f"RECIPE_EXTRA token {token!r} is not a plain hydra override key=value"
        if QUEUE_OWNED_KEYS.match(token):
            return None, f"RECIPE_EXTRA token {token!r} overrides a key the queue owns"
    s["RECIPE_EXTRA"] = " ".join(extra)
    return s, ""


def canon(s: dict) -> str:
    return " ".join(f"{k}={' '.join(map(str, s[k])) if isinstance(s[k], list) else s[k]}" for k in DEFAULTS)


def load_settings(write: bool) -> tuple[dict, str]:
    """The settings file, else the last good one (invalid content keeps the previous setting), else the defaults."""
    path, good = QDIR / "settings.conf", QDIR / ".settings_good.json"
    try:
        text = path.read_text()
    except OSError:
        text = None
    if text is not None:
        s, err = parse_settings(text)
        if s is not None:
            if write:
                try:
                    old = json.loads(good.read_text())
                except (OSError, ValueError):
                    old = None
                if old != s:
                    tmp = good.with_suffix(".tmp")
                    tmp.write_text(json.dumps(s))
                    os.replace(tmp, good)
            return s, ""
    else:
        err = f"{path} unreadable"
    try:
        prev = json.loads(good.read_text())
        return prev, f"{err}: keeping the previous settings"
    except (OSError, ValueError):
        return parse_settings("")[0], f"{err}: using the defaults"


# ============================================================================================== jobs
def read_manifest(root: Path) -> dict[int, tuple[str, str]]:
    m = json.loads((root / "whole_body/soma/track2_manifest.json").read_text())
    return {int(r["episode_index"]): (str(r.get("sequence_id", s)), str(r["object"])) for s, r in m.items()}


def support_rule(root: Path, seq: str) -> str:
    """scene_support as track2_queue_tier2.sh support_extra: 'true' when <root>/whole_body/reconstructed_stage/
    <seq>_support.usda exists and defines a Cylinder or Cube, else 'false' (-> scene.support=false)."""
    path = root / "whole_body/reconstructed_stage" / f"{seq}_support.usda"
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return "false"
    return "true" if re.search(r"def (Cylinder|Cube)", text) else "false"


def final_marker(run: Path) -> dict | None:
    try:
        data = json.loads((run / "final.json").read_text())
        it = data["iteration"]
        if isinstance(it, int) and not isinstance(it, bool) and data["checkpoint"] == f"model_{it}.safetensors":
            return data
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def done_reason(run: Path, iters: int = ITERS) -> str | None:
    """As track2_queue_helper.py done: final.json early stop (help_zero_tail) or max_iterations >= iters with its
    checkpoint present, or model_<iters>.safetensors."""
    final = final_marker(run)
    if final is not None and (run / final["checkpoint"]).is_file():
        if final.get("reason") == "help_zero_tail":
            return f"final.json: early stop at iteration {final['iteration']}"
        if final.get("reason") == "max_iterations" and final["iteration"] >= iters:
            return f"final.json: max_iterations {final['iteration']}"
    if (run / f"model_{iters}.safetensors").is_file():
        return f"model_{iters}.safetensors"
    return None


def jobs_init(out: Path) -> int:
    manifest = read_manifest(TIER2_ROOT)
    if sorted(manifest) != list(range(30)):
        raise SystemExit(f"Tier-2 manifest lists episodes {sorted(manifest)}, expected 0..29")
    finals = {}
    for ep in range(30):
        f = final_marker(C3R2_TIER2 / f"ep{ep:02d}")
        finals[ep] = f["iteration"] if f else None
    capped = sorted(ep for ep, it in finals.items() if it == CAP_ITERATION)
    if capped != sorted(CAP_EPISODES):
        raise SystemExit(f"C3r2_tier2 episodes at {CAP_ITERATION}: {capped}; the task lists {CAP_EPISODES}")
    order = CAP_EPISODES + [ep for ep in range(30) if ep not in CAP_EPISODES]
    lines = [
        f"# jobs.tsv of scripts/track2_queue_tier2v2.sh (arm {ARM}), written {time.strftime('%F %T')} by",
        "# track2_queue_tier2v2_helper.py jobs-init. Priority order: the 17 episodes whose C3r2_tier2 run stopped at",
        f"# the deadline-forced {CAP_ITERATION} first (task order), then the other 13 in index order. scene_support is",
        "# the support rule of track2_queue_tier2.sh evaluated on the Tier-2 root; the training input must give the same",
        "# answer (input check). Columns: order arm episode sequence_id object scene_support group c3r2_tier2_final_iteration",
    ]
    for n, ep in enumerate(order, 1):
        seq, obj = manifest[ep]
        group = "cap45k" if ep in CAP_EPISODES else "other"
        lines.append(f"{n}\t{ARM}\t{ep:02d}\t{seq}\t{obj}\t{support_rule(TIER2_ROOT, seq)}\t{group}\t{finals[ep]}")
    tmp = out.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, out)
    print(f"wrote {out}: {len(order)} jobs, order {' '.join(f'{e:02d}' for e in order)}")
    return 0


def read_jobs(path: Path | None = None) -> list[dict]:
    path = path or QDIR / "jobs.tsv"
    jobs = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        jobs.append({"order": int(f[0]), "arm": f[1], "ep": int(f[2]), "seq": f[3], "object": f[4], "support": f[5],
                     "group": f[6], "c3r2_final": f[7]})
    return jobs


# ============================================================================================== arm-A input check
def under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def input_files(data: Path, jobs: list[dict]) -> list[Path]:
    files = sorted(p for p in data.glob("READY*") if p.is_file()) if data.is_dir() else []
    files.append(data / "whole_body/soma/track2_manifest.json")
    for j in jobs:
        sdir = data / "whole_body/soma" / f"sequence_id={j['seq']}"
        files += [sdir / "robot_name=g1/data.parquet", sdir / "ground_plane.json"]
        stage = data / "whole_body/reconstructed_stage"
        files += [stage / f"{j['seq']}_support.usda", stage / f"{j['seq']}_g1_support.usda"]
    return files


def fingerprint(files: list[Path]) -> str:
    h = hashlib.sha256()
    for p in files:
        try:
            st = p.stat()
            h.update(f"{p}\t{st.st_size}\t{st.st_mtime_ns}\n".encode())
        except OSError:
            h.update(f"{p}\tmissing\n".encode())
    return h.hexdigest()


def sha256(path: Path) -> str | None:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def strings_of(table, name):
    """Every string value of a column (string, dictionary<string> or nested lists of strings)."""
    def walk(v):
        if isinstance(v, str):
            yield v
        elif isinstance(v, (list, tuple)):
            for x in v:
                yield from walk(x)
    for v in table.column(name).to_pylist():
        yield from walk(v)


def check_parquet(path: Path, seq: str) -> str | None:
    try:
        with path.open("rb") as f:
            head = f.read(4)
            f.seek(-4, os.SEEK_END)
            tail = f.read(4)
    except OSError as error:
        return f"unreadable ({error})"
    if head != b"PAR1" or tail != b"PAR1":
        return "not a complete parquet file (PAR1 magic missing: still being written?)"
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(str(path))
    except Exception as error:  # noqa: BLE001
        return f"pyarrow cannot read it ({type(error).__name__}: {str(error)[:160]})"
    if table.num_rows != 1:
        return f"{table.num_rows} rows (a G1 reference parquet has 1)"
    names = set(table.schema.names)
    for col in ("robot_root_position", "robot_joint_positions", "object_body_position", "object_body_wxyz"):
        if col not in names:
            return f"column {col} missing"
    if "sequence_id" in names:
        got = list(strings_of(table, "sequence_id"))
        if got and got[0] != seq:
            return f"sequence_id {got[0]!r} != {seq!r}"
    olds = (str(OLD_ARMA_ROOT) + "/", str(ARMA_V2_ROOT) + "/")
    for name, typ in zip(table.schema.names, table.schema.types):
        if "string" not in str(typ):
            continue
        for value in strings_of(table, name):
            if any(old in value for old in olds):
                return f"column {name} references a corrected-body (arm-A) root ({value[:120]})"
    for col in ("object_mesh_paths", "object_urdf_paths"):
        if col in names:
            for value in strings_of(table, col):
                if value and not Path(value).is_file():
                    return f"{col} names a missing file {value}"
    return None


def input_check(data_text: str, jobs: list[dict], write: bool) -> dict:
    """Readiness of the training input. Cheap checks every call; the full validation (parquet contents,
    sha256 comparisons, support files) once per change of the input files' size/mtime fingerprint (cached in
    QDIR/input_check.json)."""
    data = Path(data_text)
    res = {"data": data_text, "ready": False, "why": "", "fingerprint": None, "checked_at": time.strftime("%F %T")}
    if os.environ.get("TIER2V2_TEST_INPUT_OK") == "1" and QDIR != ME / "fc_runs/queue_tier2v2":
        return res | {"ready": True, "why": "TEST: input check bypassed (integration test QDIR only)"}
    real = Path(os.path.realpath(data))
    for label, root in FORBIDDEN_ROOTS.items():
        rr = Path(os.path.realpath(root))
        if real == rr or under(real, rr):
            res["why"] = f"DATA {data} is the {label}: refused"
            return res
    if not data.is_dir():
        res["why"] = f"{data} does not exist yet"
        return res
    markers = sorted(p.name for p in data.glob("READY*") if p.is_file())
    if not markers:
        res["why"] = f"no READY marker in {data} (the rebuild has not finished)"
        return res
    try:
        manifest = read_manifest(data)
    except (OSError, ValueError, KeyError) as error:
        res["why"] = f"no readable {data}/whole_body/soma/track2_manifest.json ({type(error).__name__})"
        return res
    for j in jobs:
        if manifest.get(j["ep"], (None,))[0] != j["seq"]:
            res["why"] = f"manifest maps ep{j['ep']:02d} to {manifest.get(j['ep'], ('missing',))[0]}, jobs.tsv to {j['seq']}"
            return res
    missing = [j["ep"] for j in jobs
               if not (data / "whole_body/soma" / f"sequence_id={j['seq']}/robot_name=g1/data.parquet").is_file()]
    if missing:
        res["why"] = f"{len(missing)} of {len(jobs)} G1 parquets missing (ep {' '.join(f'{e:02d}' for e in missing[:10])}{' ...' if len(missing) > 10 else ''})"
        return res
    fp = fingerprint(input_files(data, jobs))
    res["fingerprint"] = fp
    cache = QDIR / "input_check.json"
    try:
        cached = json.loads(cache.read_text())
        if cached.get("fingerprint") == fp and cached.get("data") == data_text:
            return cached
    except (OSError, ValueError):
        pass
    problems, shas = [], {}
    for j in jobs:
        ep, seq = j["ep"], j["seq"]
        sdir = data / "whole_body/soma" / f"sequence_id={seq}"
        pq_path = sdir / "robot_name=g1/data.parquet"
        why = check_parquet(pq_path, seq)
        if why:
            problems.append(f"ep{ep:02d} parquet: {why}")
            continue
        digest = sha256(pq_path)
        shas[f"{ep:02d}"] = digest
        if digest == sha256(TIER2_ROOT / "whole_body/soma" / f"sequence_id={seq}/robot_name=g1/data.parquet"):
            problems.append(f"ep{ep:02d} parquet is byte-identical to the raw Tier-2 input")
        for label, root in (("old arm-A (first body correction)", OLD_ARMA_ROOT), ("arm-A v2 (corrected body)", ARMA_V2_ROOT)):
            old = root / "whole_body/soma" / f"sequence_id={seq}/robot_name=g1/data.parquet"
            if old.is_file() and digest == sha256(old):
                problems.append(f"ep{ep:02d} parquet is byte-identical to the {label} input")
        if not (sdir / "ground_plane.json").is_file():
            problems.append(f"ep{ep:02d}: no ground_plane.json")
        for name in (f"{seq}_support.usda", f"{seq}_g1_support.usda"):
            mine = data / "whole_body/reconstructed_stage" / name
            ref = TIER2_ROOT / "whole_body/reconstructed_stage" / name
            if mine.is_file() != ref.is_file() or (mine.is_file() and sha256(mine) != sha256(ref)):
                problems.append(f"ep{ep:02d}: reconstructed_stage/{name} differs from the Tier-2 root's")
        if support_rule(data, seq) != j["support"]:
            problems.append(f"ep{ep:02d}: support rule gives {support_rule(data, seq)}, jobs.tsv {j['support']}")
    res.update({"markers": markers, "parquet_sha256": shas})
    if problems:
        res["why"] = f"{len(problems)} problem(s): " + "; ".join(problems[:6]) + (" ..." if len(problems) > 6 else "")
        res["problems"] = problems
    else:
        res["ready"] = True
        res["why"] = (f"ready: READY marker ({', '.join(markers)}), 30/30 G1 parquets valid (1 row, sequence ids, asset "
                      f"files, none byte-identical to raw Tier-2 or an arm-A root), support files identical to Tier-2's")
    if write:
        tmp = cache.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=1))
        os.replace(tmp, cache)
    return res


def arm_lock_check(s: dict, inp: dict) -> tuple[bool, str]:
    """After the first start, DATA, RECIPE_EXTRA and the input files must stay what the first job used."""
    try:
        lock = json.loads((QDIR / "arm_lock.json").read_text())
    except OSError:
        return True, "no job started yet (arm_lock.json absent)"
    except ValueError:
        return False, "arm_lock.json unreadable"
    if lock.get("data") != s["DATA"]:
        return False, f"DATA {s['DATA']} != {lock.get('data')} used by the first job (arm_lock.json)"
    if lock.get("recipe_extra", "") != s["RECIPE_EXTRA"]:
        return False, f"RECIPE_EXTRA {s['RECIPE_EXTRA']!r} != {lock.get('recipe_extra')!r} used by the first job"
    if inp.get("fingerprint") and lock.get("fingerprint") != inp["fingerprint"]:
        return False, "the training input files changed (size/mtime) since the first job started"
    return True, f"locked at {lock.get('time')}: DATA and RECIPE_EXTRA unchanged, input unchanged"


def lock_arm() -> int:
    s, _ = load_settings(write=False)
    path = QDIR / "arm_lock.json"
    if path.exists():
        print(f"{path} exists")
        return 0
    inp = input_check(s["DATA"], read_jobs(), write=True)
    if not inp.get("ready"):
        print(f"input not ready: {inp.get('why')}")
        return 1
    lock = {"time": time.strftime("%F %T"), "data": s["DATA"], "recipe_extra": s["RECIPE_EXTRA"],
            "fingerprint": inp["fingerprint"], "parquet_sha256": inp.get("parquet_sha256"), "markers": inp.get("markers")}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(lock, indent=1))
    os.replace(tmp, path)
    print(f"wrote {path}")
    return 0


# ============================================================================================== /proc and nvidia-smi
def read_cmdline(pid: int) -> list[str]:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0")[:-1]
    except OSError:
        return []


def read_environ(pid: int) -> dict[str, str]:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes().decode(errors="replace")
    except OSError:
        return {}
    env = {}
    for item in raw.split("\0"):
        k, sep, v = item.partition("=")
        if sep:
            env[k] = v
    return env


def proc_age(pid: int) -> float | None:
    try:
        line = Path(f"/proc/{pid}/stat").read_text()
        start = int(line[line.rindex(")") + 2:].split()[19])
        uptime = float(Path("/proc/uptime").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    return uptime - start / TICK


def all_pids() -> list[int]:
    return [int(p) for p in os.listdir("/proc") if p.isdigit()]


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
                     "name": ",".join(parts[3:])})
    return {"gpus": gpus, "apps": apps}


EVAL_RE = re.compile(r"evaluate_policy|run_policy_evaluation|eval_all\.py|eval_zero_help|zero_help_eval|eval_watch|_eval\.py")
TRAIN_ARG_RE = re.compile(r"(^|/)train[\w\-]*\.py$")


def classify(cmd: list[str], env: dict[str, str]) -> str:
    """'mps' | 'eval' | 'train' | 'unknown' for a GPU process."""
    line = " ".join(cmd)
    if "nvidia-cuda-mps" in line:
        return "mps"
    if "RUNIN_EVAL" in env or "TIER2V2_EVAL" in env or EVAL_RE.search(line):
        return "eval"
    if "RUNIN_QUEUE" in env or "TIER2V2_QUEUE" in env or "train_rl_games" in line:
        return "train"
    if any(TRAIN_ARG_RE.search(a) for a in cmd[:4]):
        return "train"
    if len(cmd) > 2 and cmd[1] == "-m" and cmd[2].split(".")[-1].startswith("train"):
        return "train"
    return "unknown"


def env_gpu(env: dict[str, str], uuid_to_idx: dict[str, int]) -> int | None:
    for key in ("TIER2V2_GPU", "RUNIN_GPU", "TIER2_GPU"):
        v = env.get(key, "").strip()
        if re.fullmatch(r"[0-7]", v):
            return int(v)
    v = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    if v and "," not in v:
        if re.fullmatch(r"[0-7]", v):
            return int(v)
        return uuid_to_idx.get(v)
    return None


def is_python_trainer(cmd: list[str]) -> bool:
    return (len(cmd) > 1 and os.path.basename(cmd[0]).startswith("python")
            and any(a.endswith(("train_rl.py", "train_rl_games.py")) for a in cmd[1:4]))


def mps_gpus(state: dict, pids: list[int]) -> dict[int, str]:
    found: dict[int, str] = {}
    for app in state["apps"]:
        if "mps-server" in app["name"]:
            found.setdefault(app["gpu"], f"MPS server pid {app['pid']}")
        elif read_environ(app["pid"]).get("CUDA_MPS_PIPE_DIRECTORY"):
            found.setdefault(app["gpu"], f"MPS client pid {app['pid']}")
    uuid_to_idx = {v["uuid"]: k for k, v in state["gpus"].items()}
    for pid in pids:
        cmd = read_cmdline(pid)
        if not cmd or not cmd[0].endswith("nvidia-cuda-mps-control") or "-d" not in cmd[1:]:
            continue
        for token in read_environ(pid).get("CUDA_VISIBLE_DEVICES", "").split(","):
            token = token.strip()
            idx = int(token) if token.isdigit() else uuid_to_idx.get(token)
            if idx is not None:
                found.setdefault(idx, f"MPS control daemon pid {pid}")
    return found


def read_state_rows(path: Path | None) -> list[dict]:
    """Rows of the scheduler's state.tsv: episode, runner pid, gpu, start epoch (0 = adopted), train pid."""
    rows = []
    if path is None:
        return rows
    try:
        for line in path.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            f = line.split("\t")
            rows.append({"ep": int(f[0]), "runner": int(f[1]) if f[1].isdigit() else None,
                         "gpu": int(f[2]) if f[2].isdigit() else None, "t0": int(f[3]) if f[3].isdigit() else 0,
                         "train_pid": int(f[4]) if len(f) > 4 and f[4].isdigit() else None})
    except (OSError, ValueError, IndexError):
        pass
    return rows


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state != "Z"


def observe(s: dict, state_rows: list[dict], now: float) -> tuple[dict, dict]:
    """The live per-GPU view: {g: {present, free, total, trainings, train_pids, evals, others, pending, debt, mps}}
    and the processes behind it (for the log)."""
    st = smi()
    pids = all_pids()
    uuid_to_idx = {v["uuid"]: k for k, v in st["gpus"].items()}
    mps = mps_gpus(st, pids)
    view = {g: {"present": True, "free": v["free"], "total": v["total"], "trainings": 0, "train_pids": [], "evals": [],
                "others": [], "pending": 0, "debt": 0, "mps": mps.get(g)} for g, v in st["gpus"].items()}
    mem = {a["pid"]: a["mem"] for a in st["apps"]}
    counted: dict[int, int] = {}  # pid -> gpu
    ours: dict[int, dict] = {}  # ep -> {pid, gpu, age}
    for app in st["apps"]:
        pid, g = app["pid"], app["gpu"]
        cmd, env = read_cmdline(pid), read_environ(pid)
        cls = classify(cmd, env)
        age = proc_age(pid) or 0.0
        if cls == "train" or (cls == "unknown" and age >= s["UNKNOWN_TRAIN_AGE_S"]):
            counted[pid] = g
        elif cls == "eval":
            view[g]["evals"].append(pid)
        elif cls == "unknown":
            view[g]["others"].append(pid)
    for pid in pids:  # trainings not on nvidia-smi yet (compiling) and our own trainers
        cmd = read_cmdline(pid)
        if not is_python_trainer(cmd):
            continue
        env = read_environ(pid)
        if env.get("TIER2V2_QUEUE") == str(QDIR) and env.get("TIER2V2_EP", "").isdigit():
            g = counted.get(pid, env_gpu(env, uuid_to_idx))
            ours[int(env["TIER2V2_EP"])] = {"pid": pid, "gpu": g, "age": proc_age(pid) or 0.0}
        if pid in counted:
            continue
        g = env_gpu(env, uuid_to_idx)
        if g is not None and g in view:
            counted[pid] = g
    for pid, g in counted.items():
        view[g]["trainings"] += 1
        view[g]["train_pids"].append(pid)
    for t in ours.values():  # ramp debt of our young trainings
        if t["gpu"] in view and t["age"] < RAMP_S:
            view[t["gpu"]]["debt"] += max(0, s["JOB_MB"] - mem.get(t["pid"], 0))
    for row in state_rows:  # our launches whose trainer is not visible yet
        if row["ep"] in ours or row["gpu"] not in view or not alive(row["runner"]):
            continue
        if row["t0"] and now - row["t0"] < RAMP_S:
            view[row["gpu"]]["pending"] += 1
            view[row["gpu"]]["debt"] += s["JOB_MB"]
    return view, {"ours": ours}


# ============================================================================================== the rule (pure)
def decide_gpus(view: dict[int, dict], s: dict) -> dict[int, dict]:
    """Per-GPU slot decision from an observed (or synthetic) view. Pure: no I/O."""
    cap, out = s["MAX_TRAIN_PER_GPU"], {}
    for g in sorted(set(view) | set(s["GPUS"])):
        v = view.get(g, {"present": False})
        n = v.get("trainings", 0) + v.get("pending", 0)
        free, debt = v.get("free", 0), v.get("debt", 0)
        need = s["JOB_MB"] + s["MIN_FREE_AFTER_MB"] + debt
        if g not in s["GPUS"]:
            ok, why = False, "not in GPUS setting"
        elif not v.get("present"):
            ok, why = False, "not listed by nvidia-smi"
        elif v.get("mps"):
            ok, why = False, f"MPS GPU ({v['mps']})"
        elif n >= cap:
            ok, why = False, f"full: {n} training(s) >= cap {cap}"
        elif free < need:
            ok, why = False, (f"memory: {free} MiB free < {need} (job {s['JOB_MB']} + {s['MIN_FREE_AFTER_MB']} left after"
                              f"{f' + ramp debt {debt}' if debt else ''})")
        else:
            ok, why = True, f"{cap - n} slot(s) under cap {cap}; {free} MiB free >= {need}"
        out[g] = {"ok": ok, "why": why, "n": n, "free": free, "need": need, "debt": debt,
                  "evals": len(v.get("evals", [])), "others": len(v.get("others", []))}
    return out


def choose(dec: dict[int, dict]) -> int | None:
    ok = [g for g, d in dec.items() if d["ok"]]
    return min(ok, key=lambda g: (dec[g]["n"], -(dec[g]["free"] - dec[g]["debt"]), g)) if ok else None


def runin_gate(state: dict | None, now: float, stale_s: int) -> tuple[bool, str]:
    if state is None:
        return False, f"cannot read {RUNIN_STATE}"
    waiting = state.get("waiting") or []
    if waiting:
        return False, f"rolling start has {len(waiting)} waiting/retrying task(s) (ep {' '.join(map(str, waiting[:8]))})"
    if state.get("finished"):
        return True, "rolling-start queue finished"
    try:
        t = time.mktime(time.strptime(state.get("time", ""), "%Y-%m-%d %H:%M:%S"))
    except (TypeError, ValueError, OverflowError):
        return False, "rolling-start queue_state.json has no valid time"
    if now - t > stale_s:
        return False, f"rolling-start state is {int(now - t)} s old (> {stale_s}; its scheduler is not running?)"
    return True, f"no rolling-start task waiting ({len(state.get('running') or {})} running)"


def recipe_check(extra: str) -> tuple[bool, str]:
    """RECIPE_EXTRA must be exactly the arm's stretched-curriculum overrides (the arm name promises them)."""
    if extra == ARM_RECIPE_EXTRA:
        return True, f"RECIPE_EXTRA = the arm's overrides ({ARM_RECIPE_EXTRA})"
    return False, f"RECIPE_EXTRA {extra!r} != the arm's overrides {ARM_RECIPE_EXTRA!r} ({ARM})"


def gates(flags: dict) -> list[tuple[str, bool, str]]:
    """Global gates from plain inputs (pure). flags: pause, hold (text or None), input (dict), recipe (ok, why),
    armlock (ok, why), runin (ok, why), pids (headroom or None), pids_min."""
    out = [("pause", not flags["pause"], "PAUSE file present: no new start" if flags["pause"] else "no PAUSE file")]
    hold = flags["hold"]
    out.append(("hold_armA", hold is None, f"HOLD_armA set: the arm is held ({hold})" if hold is not None else "not held"))
    inp = flags["input"]
    out.append(("input", bool(inp.get("ready")), inp.get("why", "")))
    out.append(("recipe",) + tuple(flags["recipe"]))
    out.append(("arm_lock",) + tuple(flags["armlock"]))
    out.append(("runin",) + tuple(flags["runin"]))
    head = flags["pids"]
    out.append(("pids", head is None or head >= flags["pids_min"],
                f"container pids headroom {'unlimited' if head is None else head} (need >= {flags['pids_min']})"))
    return out


def pids_headroom() -> int | None:
    for sub in ("", "pids/"):
        try:
            limit = Path(f"/sys/fs/cgroup/{sub}pids.max").read_text().strip()
            current = int(Path(f"/sys/fs/cgroup/{sub}pids.current").read_text())
        except (OSError, ValueError):
            continue
        return None if limit == "max" else int(limit) - current
    return None


# ============================================================================================== post-training work
def post_status(ep: int) -> dict:
    try:
        return json.loads((EVROOT / f"ep{ep:02d}" / "post.json").read_text())
    except (OSError, ValueError):
        return {}


def lock_held(path: Path) -> bool:
    import fcntl
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


TERMINAL = ("done", "export_refused", "failed", "done_score_failed")


def post_work(jobs: list[dict], running_eps: set[int], now: float) -> tuple[list[int], list[int]]:
    """(episodes whose post-training steps should start now, episodes whose post is running)."""
    todo, running = [], []
    for j in jobs:
        ep = j["ep"]
        if lock_held(QDIR / "post" / f"ep{ep:02d}.lock"):
            running.append(ep)
            continue
        if ep in running_eps or done_reason(RUNS / f"ep{ep:02d}") is None:
            continue
        st = post_status(ep)
        if st.get("status") in TERMINAL:
            continue
        if st.get("status") == "retry" and float(st.get("retry_after", 0)) > now:
            continue
        todo.append(ep)
    return todo, running


# ============================================================================================== CPU slice
def parse_cpus(text: str) -> set[int]:
    cpus: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if part and part != "none":
            a, _, b = part.partition("-")
            cpus.update(range(int(a), int(b or a) + 1))
    return cpus


def cpu_slice(gpu: int, exclude: str, max_cores: int, min_cores: int) -> tuple[str, int, str]:
    """As runin_helper.cpu_slice: up to max_cores physical cores (+HT siblings) of the GPU's NUMA node that no
    process holds as a slice (an affinity of <= 26 physical cores), from the training pool; fewer than min_cores
    free borrows the other node's."""
    full, pinned = set(range(208)), {c % 104 for c in parse_cpus(exclude)}
    for pid in all_pids():
        if not read_cmdline(pid):
            continue
        try:
            status = Path(f"/proc/{pid}/status").read_text()
        except OSError:
            continue
        allowed = next((l.split(":", 1)[1].strip() for l in status.splitlines() if l.startswith("Cpus_allowed_list:")), "")
        if not allowed or parse_cpus(allowed) >= full:
            continue
        cores = {c % 104 for c in parse_cpus(allowed)}
        if len(cores) <= 26:
            pinned |= cores
    node = 0 if gpu <= 2 else 1
    pools = [parse_cpus(TRAIN_POOLS[0]), parse_cpus(TRAIN_POOLS[1])]
    own, other = sorted(pools[node] - pinned), sorted(pools[1 - node] - pinned)
    cores, note = own[:max_cores], ""
    if len(cores) < min_cores:
        extra = other[: min_cores - len(cores)]
        cores += extra
        note = "cross-NUMA" if extra else ""
    if not cores:
        return "none", 0, "no-free-core"
    return ",".join(f"{c},{c + 104}" for c in cores), 2 * len(cores), note


# ============================================================================================== tick
def read_list(path: Path | None) -> list[int]:
    if path is None:
        return []
    try:
        return [int(x) for x in path.read_text().split() if x.isdigit()]
    except OSError:
        return []


def tick(args: list[str]) -> int:
    write = "--no-write" not in args

    def opt(name):
        return Path(args[args.index(name) + 1]) if name in args else None

    now = time.time()
    s, serr = load_settings(write)
    print(f"SETTING {canon(s)}")
    for key in DEFAULTS:
        value = s[key]
        print(f"SET {key} {' '.join(map(str, value)) if isinstance(value, list) else value}")
    if serr:
        print(f"SETTINGS_ERR {serr}")
    jobs = read_jobs()
    inp = input_check(s["DATA"], jobs, write)
    print(f"INPUT {'ready' if inp.get('ready') else 'not_ready'} {inp.get('why', '')}")
    try:
        runin = json.loads(RUNIN_STATE.read_text())
    except (OSError, ValueError):
        runin = None
    hold = None
    if (QDIR / "HOLD_armA").exists():
        try:
            hold = (QDIR / "HOLD_armA").read_text().strip().splitlines()[0][:160]
        except (OSError, IndexError):
            hold = "HOLD_armA"
    flags = {"pause": (QDIR / "PAUSE").exists(), "hold": hold, "input": inp, "recipe": recipe_check(s["RECIPE_EXTRA"]),
             "armlock": arm_lock_check(s, inp),
             "runin": runin_gate(runin, now, s["RUNIN_STALE_S"]), "pids": pids_headroom(), "pids_min": s["PIDS_HEADROOM"]}
    gl = gates(flags)
    for name, ok, why in gl:
        print(f"GATE {name} {'ok' if ok else 'block'} {why}")
    rows = read_state_rows(opt("--state"))
    try:
        view, extra = observe(s, rows, now)
        dec = decide_gpus(view, s)
    except Exception as error:  # noqa: BLE001 - nvidia-smi failure: no start
        print(f"GPU_ERR nvidia-smi/proc scan failed: {type(error).__name__}: {str(error)[:200]}")
        view, extra, dec = {}, {"ours": {}}, {}
    for g, d in dec.items():
        print(f"GPU {g} {'yes' if d['ok'] else 'no'} trainings={d['n']} free={d['free']} evals={d['evals']} "
              f"other_procs={d['others']} {d['why']}")
    blocked = [name for name, ok, _ in gl if not ok]
    g = choose(dec)
    if blocked:
        print(f"CHOICE none (blocked by {', '.join(blocked)}{f'; gpu{g} would qualify' if g is not None else ''})")
    else:
        print(f"CHOICE {g if g is not None else 'none'}")
    todo, prun = post_work(jobs, set(extra["ours"]) | {r["ep"] for r in rows}, now)
    print(f"POST_TODO {' '.join(f'{e:02d}' for e in todo)}")
    print(f"POST_RUNNING {' '.join(f'{e:02d}' for e in prun)}")
    if write:
        write_state(s, jobs, inp, gl, dec, g if not blocked else None, rows, extra, prun,
                    read_list(opt("--waiting")), read_list(opt("--failed")),
                    int(args[args.index("--pid") + 1]) if "--pid" in args else None, now)
    return 0


def write_state(s, jobs, inp, gl, dec, choice, rows, extra, prun, waiting, failed, pid, now) -> None:
    by_ep = {j["ep"]: j for j in jobs}
    running = {}
    for r in rows:
        t = extra["ours"].get(r["ep"], {})
        running[f"{r['ep']:02d}"] = {"gpu": r["gpu"], "train_pid": t.get("pid") or r["train_pid"],
                                     "started": time.strftime("%F %T", time.localtime(r["t0"])) if r["t0"] else "adopted"}
    done = [j["ep"] for j in jobs if done_reason(RUNS / f"ep{j['ep']:02d}") is not None]
    post = {f"{j['ep']:02d}": post_status(j["ep"]).get("status", "pending") for j in jobs if j["ep"] in done}
    held = any(name == "hold_armA" and not ok for name, ok, _ in gl)
    paused = any(name == "pause" and not ok for name, ok, _ in gl)
    finished = len(done) + len(failed) >= len(jobs) and not running and all(
        post.get(f"{e:02d}") in TERMINAL for e in done)
    state = {
        "time": time.strftime("%F %T", time.localtime(now)), "scheduler_pid": pid, "arm": ARM,
        "label": "Tier 2 · K3 input, stretched", "max_iterations": ITERS, "recipe_extra": s["RECIPE_EXTRA"], "data": s["DATA"],
        "held": held, "paused": paused, "input_ready": bool(inp.get("ready")), "input_why": inp.get("why", ""),
        "gates": [{"name": n, "ok": ok, "why": w} for n, ok, w in gl], "cap": s["MAX_TRAIN_PER_GPU"],
        "gpus": {str(g): d for g, d in dec.items()}, "choice": choice,
        "waiting": [f"{e:02d}" for e in waiting], "running": running, "done": [f"{e:02d}" for e in done],
        "failed": [f"{e:02d}" for e in failed], "post": post, "post_running": [f"{e:02d}" for e in prun],
        "objects": {f"{e:02d}": j["object"] for e, j in by_ep.items()}, "total": len(jobs), "finished": finished,
        "results_csv": str(EVROOT.parent / f"{ARM}_results_LOCAL_NOT_FOR_UPLOAD.csv"),
    }
    tmp = QDIR / "queue_state.json.tmp"
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, QDIR / "queue_state.json")
    blocked = [f"{n}: {w}" for n, ok, w in gl if not ok]
    slots = ", ".join(f"gpu{g} {d['n']}/{s['MAX_TRAIN_PER_GPU']}{' ok' if d['ok'] else ''}" for g, d in dec.items()
                      if g in s["GPUS"])
    line = (f"{state['time']} | {ARM} | {'HELD' if held else 'PAUSED' if paused else 'BLOCKED' if blocked else 'OPEN'}"
            f" | running {len(running)} waiting {len(waiting)} done {len(done)} failed {len(failed)} post-running {len(prun)}"
            f" | trainings/cap: {slots} | {'; '.join(blocked) if blocked else 'next start: gpu' + str(choice) if choice is not None else 'no GPU qualifies'}")
    tmp = QDIR / "status_line.txt.tmp"
    tmp.write_text(line[:2000] + "\n")
    os.replace(tmp, QDIR / "status_line.txt")


# ============================================================================================== selftest
def selftest() -> int:
    base, err = parse_settings("")
    assert base is not None and not err, err
    fails = []

    def check(name, cond):
        print(f"{'ok  ' if cond else 'FAIL'} {name}")
        if not cond:
            fails.append(name)

    def gpu(n=0, free=80000, pending=0, debt=0, mps=None, present=True, evals=0, others=0):
        return {"present": present, "free": free, "trainings": n, "pending": pending, "debt": debt, "mps": mps,
                "evals": [0] * evals, "others": [0] * others}

    # the state of 2026-10-21 03:00 CST: GPU 3 has the Tier-3 pilot only; 0/1/7 five runin jobs; 4/5/6 three
    now_view = {0: gpu(5, 16085), 1: gpu(5, 16091), 2: gpu(4, 32592), 3: gpu(1, 80916), 4: gpu(3, 48802),
                5: gpu(3, 48744), 6: gpu(3, 48518), 7: gpu(5, 16189)}
    d = decide_gpus(now_view, base)
    check("now: only GPU 3 qualifies", [g for g in d if d[g]["ok"]] == [3])
    check("now: GPU 4 refused as not in GPUS", d[4]["why"] == "not in GPUS setting")
    check("now: choice is GPU 3", choose(d) == 3)
    v = dict(now_view); v[3] = gpu(1, 80916 - 16896, pending=1, debt=16896)
    d = decide_gpus(v, base)
    check("after one launch on GPU 3 (pending, ramping): second slot still allowed", d[3]["ok"])
    v[3] = gpu(1, 80916 - 2 * 16896, pending=2, debt=2 * 16896)
    d = decide_gpus(v, base)
    check("after two launches on GPU 3: cap 3 reached", not d[3]["ok"] and "cap 3" in d[3]["why"])
    check("three trainings -> no start", not decide_gpus({3: gpu(3, 90000)}, base)[3]["ok"])
    check("two trainings, 60 GB free -> start", decide_gpus({3: gpu(2, 60000)}, base)[3]["ok"])
    check("memory: 37000 MiB free < 16896+20480 -> no", not decide_gpus({3: gpu(1, 37000)}, base)[3]["ok"])
    check("memory: 37376 MiB free == need -> yes", decide_gpus({3: gpu(1, 37376)}, base)[3]["ok"])
    check("ramp debt counts against memory", not decide_gpus({3: gpu(1, 50000, debt=14000)}, base)[3]["ok"])
    check("MPS GPU refused even when empty", not decide_gpus({3: gpu(0, 97000, mps="MPS server pid 1")}, base)[3]["ok"])
    check("evaluation processes do not count", decide_gpus({3: gpu(2, 60000, evals=3)}, base)[3]["ok"])
    check("GPU missing from nvidia-smi refused", not decide_gpus({}, base)[3]["ok"])
    s4, _ = parse_settings("MAX_TRAIN_PER_GPU=4\n")
    check("cap setting 4: three trainings -> start", decide_gpus({3: gpu(3, 60000)}, s4)[3]["ok"])
    s0, _ = parse_settings("MAX_TRAIN_PER_GPU=0\n")
    check("cap setting 0: nothing starts", not decide_gpus({3: gpu(0, 97000)}, s0)[3]["ok"])
    sg, _ = parse_settings("GPUS=3 4\n")
    check("GPUS setting with 4: GPU 4 allowed", decide_gpus({4: gpu(1, 60000)}, sg)[4]["ok"])
    check("choice: fewest trainings first",
          choose(decide_gpus({2: gpu(2, 90000), 3: gpu(1, 50000), 5: gpu(1, 60000)}, base)) == 5)
    check("choice: none when none qualifies", choose(decide_gpus(now_view | {3: gpu(3)}, base)) is None)
    # classification
    check("train_rl.py is a training", classify(["python", "scripts/train_rl.py", "experiment=x"], {}) == "train")
    check("rl_games trainer is a training", classify(["python", "train_rl_games.py", "--clip", "x"], {}) == "train")
    check("evaluator is not a training",
          classify(["python", "/k/flash_chord/scripts/evaluate_policy.py", "x"], {}) == "eval")
    check("runin eval marker wins", classify(["python", "scripts/train_rl.py"], {"RUNIN_EVAL": "x"}) == "eval")
    check("eval_all is an eval", classify(["python", "-B", "scripts/track2/eval_all.py"], {}) == "eval")
    check("unknown process", classify(["python", "retarget.py"], {}) == "unknown")
    check("module trainer", classify(["python", "-m", "pkg.train"], {}) == "train")
    check("MPS server", classify(["nvidia-cuda-mps-server"], {}) == "mps")
    check("env gpu by UUID", env_gpu({"CUDA_VISIBLE_DEVICES": "GPU-abc"}, {"GPU-abc": 3}) == 3)
    check("env gpu by marker", env_gpu({"RUNIN_GPU": "5", "CUDA_VISIBLE_DEVICES": "5"}, {}) == 5)
    check("multi-device CUDA_VISIBLE_DEVICES -> unknown", env_gpu({"CUDA_VISIBLE_DEVICES": "1,2"}, {}) is None)
    # gates
    t = time.time()
    fresh = time.strftime("%F %T", time.localtime(t - 30))
    stale = time.strftime("%F %T", time.localtime(t - 3600))
    check("runin waiting blocks", not runin_gate({"time": fresh, "waiting": ["05"]}, t, 900)[0])
    check("runin none waiting passes", runin_gate({"time": fresh, "waiting": [], "running": {"1": {}}}, t, 900)[0])
    check("runin stale blocks", not runin_gate({"time": stale, "waiting": []}, t, 900)[0])
    check("runin finished passes even if stale", runin_gate({"time": stale, "waiting": [], "finished": True}, t, 900)[0])
    check("runin unreadable blocks", not runin_gate(None, t, 900)[0])
    ok_flags = {"pause": False, "hold": None, "input": {"ready": True, "why": "ready"}, "recipe": (True, ""),
                "armlock": (True, ""), "runin": (True, ""), "pids": 10000, "pids_min": 3000}
    check("all gates open", all(ok for _, ok, _ in gates(ok_flags)))
    check("HOLD blocks", not all(ok for _, ok, _ in gates(ok_flags | {"hold": "rules check"})))
    check("PAUSE blocks", not all(ok for _, ok, _ in gates(ok_flags | {"pause": True})))
    check("input not ready blocks", not all(ok for _, ok, _ in gates(ok_flags | {"input": {"ready": False}})))
    check("pids headroom blocks", not all(ok for _, ok, _ in gates(ok_flags | {"pids": 100})))
    check("arm lock mismatch blocks", not all(ok for _, ok, _ in gates(ok_flags | {"armlock": (False, "x")})))
    check("recipe mismatch blocks", not all(ok for _, ok, _ in gates(ok_flags | {"recipe": (False, "x")})))
    check("recipe: the arm's overrides pass", recipe_check(parse_settings(f"RECIPE_EXTRA={ARM_RECIPE_EXTRA}\n")[0]["RECIPE_EXTRA"])[0])
    check("recipe: empty RECIPE_EXTRA (plain C3r2) blocks", not recipe_check("")[0])
    check("recipe: deadline only blocks", not recipe_check("reset.deadline=70312")[0])
    check("recipe: extra override blocks",
          not recipe_check(ARM_RECIPE_EXTRA + " training.early_stop_help_zero_tail=11720")[0])
    check("arm's overrides = 2x the C3r2 experiment's deadline / zero ramp",
          ARM_RECIPE_EXTRA == f"reset.deadline={2 * 35156} reset.zero_ramp={2 * 3906}")
    # settings
    check("bad key rejected", parse_settings("FOO=1\n")[0] is None)
    check("bad GPU rejected", parse_settings("GPUS=8\n")[0] is None)
    check("relative DATA rejected", parse_settings("DATA=data/x\n")[0] is None)
    check("RECIPE_EXTRA may not touch max_iterations", parse_settings("RECIPE_EXTRA=training.max_iterations=1\n")[0] is None)
    check("RECIPE_EXTRA may set the RSI timers / early stop (a deliberate, visible recipe change)",
          parse_settings("RECIPE_EXTRA=reset.deadline=70312 reset.zero_ramp=7812 training.early_stop_help_zero_tail=11720\n")[0]
          is not None)
    check("RECIPE_EXTRA plain override accepted", parse_settings("RECIPE_EXTRA=reset.deadline=70312\n")[0] is not None)
    check("RECIPE_EXTRA shell metacharacters rejected", parse_settings("RECIPE_EXTRA=a=b;rm\n")[0] is None)
    # forbidden roots (no file access needed: the first checks are path-only)
    for label, root in FORBIDDEN_ROOTS.items():
        res = input_check(str(root), [], write=False)
        check(f"input check refuses the {label}", not res["ready"] and "refused" in res["why"])
    check("input check refuses data/human_motion_data_tier2_armA_v2 (corrected body)",
          "refused" in input_check(str(ME / "data/human_motion_data_tier2_armA_v2"), [], write=False)["why"])
    check("input check refuses data/human_motion_data_tier2_armA (corrected body)",
          "refused" in input_check(str(ME / "data/human_motion_data_tier2_armA"), [], write=False)["why"])
    check("input check: the K3 root is not refused by path",
          "refused" not in input_check(str(DEFAULT_DATA), [], write=False)["why"])
    check("input check: missing K3 root is not ready",
          not input_check("/nonexistent/human_motion_data_tier2_k3", [], write=False)["ready"])
    print(f"{len(fails)} failure(s)")
    return 1 if fails else 0


# ============================================================================================== CLI
def main(argv: list[str]) -> int:
    if argv[:1] == ["jobs-init"] and len(argv) == 2:
        return jobs_init(Path(argv[1]))
    if argv[:1] == ["tick"]:
        return tick(argv[1:])
    if argv[:1] == ["ready"]:
        s, _ = load_settings(write=False)
        res = input_check(s["DATA"], read_jobs(), "--no-write" not in argv)
        print(res.get("why", ""))
        return 0 if res.get("ready") else 1
    if argv[:1] == ["lock-arm"]:
        return lock_arm()
    if argv[:1] == ["cpuslice"] and len(argv) >= 2:
        s, _ = load_settings(write=False)
        cpus, n, note = cpu_slice(int(argv[1]), argv[2] if len(argv) > 2 else "", s["MAX_CORES"], s["MIN_CORES"])
        print(cpus, n, note)
        return 0
    if argv[:1] == ["done"] and len(argv) == 2:
        why = done_reason(RUNS / f"ep{int(argv[1]):02d}")
        if why:
            print(why)
            return 0
        return 1
    if argv[:1] == ["selftest"]:
        return selftest()
    if argv[:1] == ["settings"]:
        s, err = load_settings(write=False)
        print(canon(s))
        if err:
            print(f"SETTINGS_ERR {err}")
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
