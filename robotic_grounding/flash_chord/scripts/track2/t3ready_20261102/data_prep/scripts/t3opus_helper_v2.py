#!/usr/bin/env python3
"""t3opus_helper_v2.py: checks, retarget / training plans, state, post-training step and verdict of
scripts/t3opus_driver_v2.sh, the Tier-3 opus test extended to ALL 30 episodes (arm C3r2_tier3_opus).
Pre-registrations: tmp_analysis/tier3/opus_easy/PREREG.md (waves 1-2, unchanged, sha256 c7a19218) and
tmp_analysis/tier3/opus_easy/PREREG_wave3.md (wave 3, the same pass rule; sha256 in PREREG_wave3.sha256).

  New file (2026-10-24); scripts/t3opus_helper.py (v1) is left untouched: v1 post processes that are still running keep
  using it. Every rule of v1 is kept: C3r2 exactly as C3r2_wide / C3r2_tier2 (code/fc_r2 @61fa5d3, experiment
  g1_recon_body_coord_rsi_r2, seed 42, 45000, deadline 35156 / zero_ramp 3906 / help-zero tail 5860), scene.support=false,
  data root data/human_motion_data_tier3_opus, fill rule (< 3 trainings of anybody before a start, >= 16.5 GB + 20 GB free
  + ramp debt, never GPU 4, no MPS, 2 own-NUMA cores), retarget lock, POST (export 1e-4 / 2e-4 if refused + ONE official
  evaluation against its own input + LOCAL clean diagnostic only into *_LOCAL_NOT_FOR_UPLOAD files), CRASHED marker,
  no automatic relaunch, never a signal to anything but its own evaluation session on timeout.
  Waves 1 (ep08 ep23 ep22) and 2 (ep00 ep18 ep10): as v1 (adopted: running / finished clips are never relaunched).
  Wave 3: the other 24 episodes in a FIXED order (ORDER below): unflagged clips first, then flagged clips (Han Zhou's
  README "use s1" flags, and/or SUPPORT_MISSING_FLOOR_ONLY), each group by the Tier-1 C3r2 (C3r2_wide final.json)
  help_zero_since ascending, ties by episode index. No clean-derived number is used for anything but LOCAL files.
  Support: the kit's track_2_tier3 supplies no input (input_path null; its Tier-3 folder track_2/tier_3_cari4d is not
  released) and the released G1 support surfaces belong to the Tier-1/2 datasets (released G1 retargeted world), so the
  14 episodes whose Tier-1/2 runs used scene.support=true are trained floor-only (scene.support=false) and flagged.

Run with conda_envs/flashchord/bin/python -I -B. It never starts a training. It writes only under
tmp_analysis/tier3/opus_easy/ (and, in 'post', the run's own export / evaluation dirs as v1). nvidia-smi / /proc /
CPU-slice primitives come from scripts/track2_queue_tier2v2_helper.py (imported read-only).

Subcommands (as v1, plus prep / order-check):
  check                    driver preflight (both PREREG sha256, code pins + tree digest, experiment values, data READY,
                           manifest, frame references (present, sha256 = build report), scene rule per clip, Tier-1 /
                           Tier-2 references, wave-3 order rule, cfgcheck_v2, scripts)
  cfgcheck [--no-write]    hydra-compose the raw Tier-2 and Tier-1 overrides (must reproduce their saved config.yaml) and
                           this test's argv per clip (all 30); diff; writes opus_easy/cfgcheck_v2.json
  prep [--write]           per-episode prep table (frame references incl. content = poses.npy, parquet / retarget state,
                           scene decision, flags, order); --write: opus_easy/wave3/prep_wave3.{json,tsv}
  order-check              recompute the wave-3 order from C3r2_wide final.json + flags; exit 1 if != ORDER
  wave-clips --waves W     the clips of wave(s) W (comma list, waves 1-3), in ORDER
  wave-guard --waves W     "OK why" / "REFUSE why" (exit 1): earlier waves outside W must be launched or settled
  cmd --ep NN              the training argv, NUL-separated
  plan --ep NN [--exclude CPUS] [--assume g:n,..]   "OK <gpu> <cpus> <nth> - | why" | "WAIT why"
  rplan [--exclude CPUS]   GPU/CPU for one retarget: "OK <gpu> <uuid> <cpus> <nth> | why" | "WAIT why"
  launch-ok --ep NN        exit 0 iff the parquet's sha256 equals the retarget lock and its rcheck passed
  gpus                     per-GPU view
  state --waves W          one line per clip "CLIP NN STATE detail" (all 30 clips; other waves DEFERRED)
  trainer --ep NN          live trainer pid(s) of that run dir
  crash --ep NN --why T    write opus_easy/CRASHED_epNN once (no relaunch is ever made)
  rfail --ep NN --why T    write opus_easy/RETARGET_FAILED_epNN once
  post --ep NN             export (1e-4; 2e-4 if refused, recorded) + ONE official evaluation of final.json's checkpoint
                           against its own input on its training GPU, then the LOCAL clean diagnostic
  verdict [--no-write]     the PREREG rule (waves 1-2: PREREG.md; wave 3: PREREG_wave3.md); exit 2 (nothing written)
                           when either sha256 does not match
  heartbeat [--waves W]    one status line
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
T1_ROOT = ME / "data/human_motion_data"
T2_ROOT = ME / "data/human_motion_data_tier2"
SHARPA = CODE / "logs/rl/sharpa"
PREREG = A / "PREREG.md"            # waves 1-2 (v1, unchanged)
PREREG_SHA = A / "PREREG.sha256"
PREREG3 = A / "PREREG_wave3.md"     # wave 3
PREREG3_SHA = A / "PREREG_wave3.sha256"
PIDS = A / "pids.tsv"
RDIR = A / "retarget"
LOCK = RDIR / "parquet_lock.json"
CFGCHECK = A / "cfgcheck_v2.json"
W3DIR = A / "wave3"
BUILD_REPORT = A / "build/build_report.json"
KIT = ME / "v2d_submission_kit_src/v2d_submission_kit"
STOCK = ME / "code/video_to_data_8c778f9/robotic_grounding/flash_chord"
DATASET = ME / "data/video_to_data_challenge"
KPY = ME / "conda_envs/v2dkit/bin/python"
FCPY = ME / "conda_envs/flashchord/bin/python"
CLEAN_SCORER = ME / "scripts/track2_tier2_clean_scores.py"
RETARGET_SH = ME / "scripts/t3opus_retarget.sh"       # v1 file, reused unchanged (sha256 pinned in build_files.sha256)
RCHECK_V2 = ME / "scripts/t3opus_rcheck_v2.py"
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
PLACE_TIERS = ((3, 5, 6, 7), (0, 1, 2))  # NUMA-1 GPUs first; NUMA 0 only if none fits
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
CFG_ALLOWED_SUPPORT = {"scene.support"}  # only for SUPPORT_MISSING_FLOOR_ONLY clips (Tier-1/2 true -> Tier-3 false)
FLAG_FLIP = "README: symmetric flip (use s1)"
FLAG_DEPTH = "README: depth wrong in some phases (use s1)"
FLAG_FAILED = "README: failed in s1 and opus"
FLAG_SUPPORT = "SUPPORT_MISSING_FLOOR_ONLY"
README_FLAGS = {1: FLAG_FLIP, 14: FLAG_FLIP, 26: FLAG_FLIP, 3: FLAG_DEPTH, 16: FLAG_DEPTH, 12: FLAG_FAILED, 15: FLAG_FAILED}
SUPPORT_EPISODES = (1, 2, 3, 6, 7, 9, 11, 13, 16, 17, 20, 25, 27, 28)  # Tier-1/2 scene.support=true (build report: support_pending)
# t1_hz / t2_hz = final.json help_zero_since of C3r2_wide / C3r2_tier2 (training-only references; 39062 = forced at the
# RSI deadline + zero ramp, i.e. help never reached zero by itself). support: "floor" = floor-only in Tier 1/2 as well;
# "missing" = Tier-1/2 scene.support=true, no official Tier-3 support -> trained floor-only (flagged).
_T = [  # ep, sequence_id, object, frames, wave, t1_hz, t2_hz
    (0, "2026-06-04_14-16-23_tall_bar_stool_tip_up_04", "tall_bar_stool", 778, 2, 11863, 35183),
    (1, "2026-05-18_17-06-21_vacuum_left_lean_02", "vacuum", 747, 3, 39062, 39062),
    (2, "2026-06-02_10-04-42_blue_easter_basket_place_10", "blue_easter_basket", 980, 3, 20995, 31549),
    (3, "2026-04-27_11-50-10_contigo_coffee_mug_handle_table_05", "contigo_coffee_mug", 781, 3, 12051, 39062),
    (4, "2026-04-24_11-40-52_baseball_bat_rotate_05", "baseball_bat", 803, 3, 39062, 39062),
    (5, "2026-03-18_19-36-56_g1_box_rotate_01", "big_black_box", 1003, 3, 20519, 39062),
    (6, "2026-06-01_11-26-44_monitor_rotate_05", "monitor", 546, 3, 17982, 28034),
    (7, "2026-05-29_15-20-43_iron_round_table_02", "iron", 896, 3, 12090, 39062),
    (8, "2026-06-05_15-08-08_skinny_wood_chair_push_10", "skinny_wood_chair", 708, 1, 11412, 14737),
    (9, "2026-06-04_16-45-04_big_red_bowl_under_skinny_wood_chair_03", "big_red_bowl", 870, 3, 26325, 33468),
    (10, "2026-05-01_11-36-40_white_desk_lower_to_ground_10", "white_desk", 794, 2, 15572, 28407),
    (11, "2026-04-24_13-56-58_light_blue_book_pick_place_09", "light_blue_book", 901, 3, 23114, 23065),
    (12, "2026-04-25_17-20-04_black_platform_turn_05", "black_platform", 700, 3, 19520, 31715),
    (13, "2026-06-09_10-19-09_snack_box_desk_through_06", "snack_box", 922, 3, 23132, 39062),
    (14, "2026-05-28_14-44-09_round_table_flip_08", "round_table", 993, 3, 39062, 39062),
    (15, "2026-05-06_11-06-15_wet_floor_sign_pick_place_ground_08", "wet_floor_sign", 715, 3, 15114, 39062),
    (16, "2026-03-12_15-06-36_spray_bottle_left_desk_ground_02", "spray_bottle", 477, 3, 18412, 25408),
    (17, "2026-05-01_17-04-55_traffic_cone_ground_table_02", "traffic_cone", 440, 3, 17982, 25947),
    (18, "2026-05-06_10-00-19_blue_trash_can_drag_sideways_08", "blue_trash_can", 858, 2, 13078, 39062),
    (19, "2026-05-19_10-19-48_cane_swing_05", "cane", 886, 3, 39062, 39062),
    (20, "2026-06-08_16-57-34_rice_cooker_clockwise_04", "rice_cooker", 553, 3, 12625, 13174),
    (21, "2026-04-29_11-53-57_tall_bar_stool_pull_08", "tall_bar_stool", 678, 3, 39062, 39062),
    (22, "2026-05-22_16-52-36_step_ladder_drag_03", "step_ladder", 719, 1, 12470, 39062),
    (23, "2026-06-03_15-08-03_toy_stroller_back_forth_ground_04", "toy_stroller", 946, 1, 12811, 14163),
    (24, "2026-05-29_15-43-20_watering_can_pour_right_02", "watering_can", 748, 3, 20645, 39062),
    (25, "2026-03-20_11-59-57_cyan_water_bottle_rotate_01", "cyan_water_bottle", 859, 3, 39062, 39062),
    (26, "2026-06-05_16-05-08_skateboard_rotate_ground_10", "skateboard", 669, 3, 13366, 29088),
    (27, "2026-05-01_11-15-05_woven_basket_ground_desk_10", "woven_basket", 565, 3, 15805, 39062),
    (28, "2026-05-19_11-27-07_squeegee_sweep_away_from_you_03_", "squeegee", 988, 3, 39062, 39062),
    (29, "2026-06-05_18-27-40_rolling_luggage_left_foot_push_06", "rolling_luggage", 467, 3, 18231, 39062),
]
CLIPS = {e: {"seq": s, "obj": o, "frames": f, "wave": w, "t1_hz": h1, "t2_hz": h2,
             "support": "missing" if e in SUPPORT_EPISODES else "floor",
             "flags": ([README_FLAGS[e]] if e in README_FLAGS else []) + ([FLAG_SUPPORT] if e in SUPPORT_EPISODES else [])}
         for e, s, o, f, w, h1, h2 in _T}
WAVE12_ORDER = (8, 23, 22, 0, 18, 10)  # v1 ORDER (adopted)
# wave 3: unflagged first, then flagged; each by Tier-1 C3r2 help_zero_since ascending, ties by episode index (order-check)
WAVE3_ORDER = (29, 5, 24, 4, 19, 21, 3, 7, 20, 26, 15, 27, 6, 17, 16, 12, 2, 11, 13, 9, 1, 14, 25, 28)
ORDER = WAVE12_ORDER + WAVE3_ORDER
WAVES = (1, 2, 3)


def wave3_order_by_rule() -> tuple[int, ...]:
    eps = [e for e in CLIPS if CLIPS[e]["wave"] == 3]
    return tuple(sorted(eps, key=lambda e: (bool(CLIPS[e]["flags"]), CLIPS[e]["t1_hz"], e)))


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


def _one_prereg(doc: Path, shafile: Path) -> tuple[bool, str]:
    if not doc.is_file() or not shafile.is_file():
        return False, f"{doc.name} or {shafile.name} missing"
    words = shafile.read_text().split()
    want = words[0] if words else ""
    got = sha256(doc)
    return got == want, f"sha256({doc.name}) {str(got)[:16]}... {'==' if got == want else '!='} {shafile.name} {want[:16]}..."


def prereg_status() -> tuple[bool, str]:
    """Both pre-registrations must match their recorded sha256 (waves 1-2: PREREG.md; wave 3: PREREG_wave3.md)."""
    ok1, why1 = _one_prereg(PREREG, PREREG_SHA)
    ok3, why3 = _one_prereg(PREREG3, PREREG3_SHA)
    return ok1 and ok3, f"{why1}; {why3}"


def prereg_of(ep: int) -> str:
    return PREREG3.name if CLIPS[ep]["wave"] == 3 else PREREG.name


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
    hydra.run.dir / +logging.local_jsonl (as the Tier-2 K3 tests); none of these changes training (cfgcheck). Identical to
    v1 for every clip: scene.support=false also for the 14 SUPPORT_MISSING_FLOOR_ONLY clips (their Tier-1/2 runs used the
    default true; recorded as an allowed, flagged difference in cfgcheck_v2.json)."""
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


_TRAINER_SCAN: list[tuple[int, list[str]]] | None = None  # one /proc scan per 'state' / 'heartbeat' call (30 clips)


def find_trainers(ep: int) -> list[int]:
    want = f"output_dir={run_dir(ep)}"
    found = []
    scan = _TRAINER_SCAN if _TRAINER_SCAN is not None else [(pid, H.read_cmdline(pid)) for pid in H.all_pids()]
    for pid, cmd in scan:
        if H.is_python_trainer(cmd) and want in cmd and H.alive(pid):
            found.append(pid)
    return found


def scan_trainers_once() -> None:
    global _TRAINER_SCAN
    _TRAINER_SCAN = [(pid, H.read_cmdline(pid)) for pid in H.all_pids()]


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
           "allowed_vs_tier1": sorted(CFG_ALLOWED_T1),
           "allowed_extra_support_missing": sorted(CFG_ALLOWED_SUPPORT) + [
               "only for SUPPORT_MISSING_FLOOR_ONLY clips: the raw run had scene.support=true (no override), this test false"],
           "episodes": {}}
    ok_all = True
    with initialize_config_dir(config_dir=str(CODE / "src/flash_chord/configs"), version_base="1.3"):
        for ep in ORDER:
            missing = CLIPS[ep]["support"] == "missing"
            mine_ov = [a for a in train_argv(ep)[2:] if not a.startswith("hydra.")]
            mine = _flat(OmegaConf.to_container(compose(config_name="train", overrides=mine_ov), resolve=False))
            row = {"argv": train_argv(ep)[2:], "wave": CLIPS[ep]["wave"], "support": CLIPS[ep]["support"]}
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
                if missing:  # the raw run used the released support (scene.support=true); this test: none available -> false
                    scene_ok = saved.get("scene.support") is True and not any(o.startswith("scene.support") for o in ov) \
                        and "scene.support" in dk
                    good = repro and dk <= (allowed | CFG_ALLOWED_SUPPORT) and scene_ok
                    scene_note = f"raw scene.support={saved.get('scene.support')} (default) -> false here (SUPPORT_MISSING_FLOOR_ONLY)"
                else:
                    good = repro and dk <= allowed and ("scene.support=false" in ov)
                    scene_note = f"raw had scene.support=false {'scene.support=false' in ov}"
                ok &= good
                row[tag] = {"ok": good, "hydra_dir": str(hyd), "compose_reproduces_saved_config": repro, "n_keys": len(saved),
                            "diff": diff, "overrides": ov, "scene_note": scene_note,
                            "overrides_unified_diff": list(difflib.unified_diff(ov, mine_ov, tag, "t3opus", lineterm="", n=20))}
                print(f"== ep{ep:02d} {CLIPS[ep]['obj']} vs {tag} ({hyd.parent.name}): compose reproduces saved config {repro} "
                      f"({len(saved)} keys); differ in {len(diff)} key(s), all allowed {dk <= (allowed | (CFG_ALLOWED_SUPPORT if missing else set()))}; "
                      f"{scene_note} -> {'ok' if good else 'FAIL'}")
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


def build_report() -> dict:
    try:
        return json.loads(BUILD_REPORT.read_text())
    except (OSError, ValueError):
        return {}


def scene_decision(ep: int, rep: dict | None = None) -> tuple[bool, str]:
    """floor: as v1 (geometry-free released file copied into the Tier-3 root; Tier-2 rule false). missing: the Tier-1/2 roots
    hold a support file with geometry (scene.support=true there), the build report lists the episode as support_pending,
    and the Tier-3 root holds NO support file of any name (so nothing could be resolved) -> scene.support=false."""
    c = CLIPS[ep]
    rep = build_report() if rep is None else rep
    stage = DATA / "whole_body/reconstructed_stage"
    t1, t2, t3 = H.support_rule(T1_ROOT, c["seq"]), H.support_rule(T2_ROOT, c["seq"]), H.support_rule(DATA, c["seq"])
    plain, g1 = stage / f"{c['seq']}_support.usda", stage / f"{c['seq']}_g1_support.usda"
    if c["support"] == "floor":
        ok = t3 == "false" == t2 and plain.is_file() and not g1.exists() and ep in rep.get("floor_only", [])
        return ok, f"floor-only: Tier-3 rule {t3}, Tier-2 rule {t2}, Tier-1 rule {t1}; geometry-free file present -> scene.support=false"
    ok = (t1 == "true" == t2 and t3 == "false" and not plain.exists() and not g1.exists()
          and ep in rep.get("support_pending", []))
    return ok, (f"SUPPORT_MISSING_FLOOR_ONLY: Tier-1 rule {t1}, Tier-2 rule {t2} (their runs used scene.support=true); "
                f"Tier-3 root has no support file ({'none' if not plain.exists() and not g1.exists() else 'UNEXPECTED FILE'}) "
                f"-> trained with scene.support=false (no official Tier-3 support)")


def build_files_sha() -> dict[str, str]:
    out = {}
    try:
        for line in (A / "build_files.sha256").read_text().splitlines():
            h, _, rel = line.partition("  ")
            if rel:
                out[rel.strip()] = h
    except OSError:
        pass
    return out


def nongt_rows() -> dict[int, dict]:
    try:
        return {int(r["episode"]): r for r in csv.DictReader((A / "input_nongt.csv").open())}
    except (OSError, ValueError, KeyError):
        return {}


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
    add("cfgcheck", cfc.get("all_ok") is True and len(cfc.get("episodes", {})) == len(ORDER),
        f"{CFGCHECK.name}: all_ok {cfc.get('all_ok')}, {len(cfc.get('episodes', {}))} clips")
    for f in (RETARGET_SH, RCHECK_V2, CLEAN_SCORER, KPY, FCPY):
        add(f"file_{f.name}", f.is_file(), str(f))
    pinned = build_files_sha().get("scripts/t3opus_retarget.sh")
    add("retarget_script_unchanged", pinned is not None and sha256(RETARGET_SH) == pinned,
        f"sha256(t3opus_retarget.sh) {str(sha256(RETARGET_SH))[:16]}... = build_files.sha256 {str(pinned)[:16]}...")
    add("clip_set", sorted(CLIPS) == list(range(30)) and sorted(ORDER) == list(range(30)) and len(set(ORDER)) == 30,
        f"{len(CLIPS)} clips, ORDER covers {len(set(ORDER))} episodes")
    add("wave3_order_rule", WAVE3_ORDER == wave3_order_by_rule(),
        f"ORDER {' '.join(f'{e:02d}' for e in WAVE3_ORDER)}; rule (unflagged first, Tier-1 C3r2 help_zero_since asc, ep) "
        f"{' '.join(f'{e:02d}' for e in wave3_order_by_rule())}")
    rep = build_report()
    add("build_report", rep.get("bad") == [] and sorted(rep.get("floor_only", []) + rep.get("support_pending", [])) == list(range(30))
        and sorted(rep.get("support_pending", [])) == sorted(SUPPORT_EPISODES),
        f"{BUILD_REPORT.name}: floor_only {rep.get('floor_only')}, support_pending {rep.get('support_pending')}")
    ng = nongt_rows()
    add("flags_vs_input_nongt", len(ng) == 30 and all(bool(ng[e].get("readme_flag", "").strip()) == (e in README_FLAGS)
                                                     and (ng[e].get("support") == "support") == (e in SUPPORT_EPISODES) for e in ng),
        "README flags and support classes = opus_easy/input_nongt.csv (2026-10-22, non-GT)")
    for ep in ORDER:
        c, tag = CLIPS[ep], f"ep{ep:02d}"
        add(f"{tag}_manifest", seq_of.get(ep) == (c["seq"], c["frames"]), f"manifest {seq_of.get(ep)}")
        fr = FRAME_REFS / f"episode_{ep:06d}.npz"
        want = (rep.get("episodes", {}).get(f"{ep:02d}") or {}).get("frame_ref_sha256")
        add(f"{tag}_frame_ref", fr.is_file() and want is not None and sha256(fr) == want,
            f"{fr.name} sha256 {str(sha256(fr))[:12]}... = build report {str(want)[:12]}...")
        ok, detail = scene_decision(ep, rep)
        add(f"{tag}_scene", ok, detail)
        for arm, runs, key in (("t1", T1_RUNS, "t1_hz"), ("t2", T2_RUNS, "t2_hz")):
            fin = H.final_marker(runs / f"ep{ep:02d}") or {}
            add(f"{tag}_{arm}_reference", fin.get("help_zero_since") == c[key], f"{runs.name} help_zero_since {fin.get('help_zero_since')} (table {c[key]})")
    bad = [r for r in rows if not r[1]]
    for name, ok, detail in rows:
        print(f"CHECK {name} {'ok' if ok else 'FAIL'} {detail}")
    print(f"CHECKS {len(rows) - len(bad)}/{len(rows)} ok" + (f"; FAILED: {' '.join(r[0] for r in bad)}" if bad else ""))
    return 0 if not bad else 1


def frame_ref_content(ep: int) -> tuple[bool, str]:
    """frame_references_opus/episode_NNNNNN.npz = the opus poses.npy (position + rotation as xyzw) in released_track2_world,
    all frames: the same construction as scripts/t3opus_build_data.py (checked, not rebuilt)."""
    import numpy as np
    fr = FRAME_REFS / f"episode_{ep:06d}.npz"
    try:
        P = np.load(seq_dir(ep) / "poses.npy")
        with np.load(fr) as d:
            xyzw, fi, idx, frame = d["pose_xyzw"], d["frame_index"], int(d["episode_index"]), str(d["coordinate_frame"])
    except (OSError, KeyError, ValueError) as error:
        return False, f"unreadable ({type(error).__name__})"
    T = CLIPS[ep]["frames"]
    if xyzw.shape != (T, 1, 7) or idx != ep or frame != "released_track2_world" or not np.array_equal(fi, np.arange(T)):
        return False, f"shape {xyzw.shape}, episode {idx}, frame {frame}, frame_index ok {np.array_equal(fi, np.arange(T))}"
    dp = float(np.abs(xyzw[:, 0, :3] - P[:, :3, 3]).max())
    q = xyzw[:, 0, 3:] / np.linalg.norm(xyzw[:, 0, 3:], axis=-1, keepdims=True)
    x, y, z, w = q.T
    Rq = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                   np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                   np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)
    dr = float(np.abs(Rq - P[:, :3, :3]).max())  # poses.npy rotations are orthonormal only to ~1e-5 (build gate: 1e-4)
    return dp <= 1e-9 and dr <= 1e-4, (f"T={T}, max |pos - poses.npy| {dp:.1e} m, max |R(quat) - R(poses.npy)| {dr:.1e} "
                                      f"(tolerance 1e-4 = the build's orthonormality gate)")


def prep(write: bool) -> int:
    """Per-episode prep table (read-only unless --write: opus_easy/wave3/prep_wave3.{json,tsv})."""
    rep, ng = build_report(), nongt_rows()
    rows, bad = [], 0
    pos = {e: i + 1 for i, e in enumerate(WAVE3_ORDER)}
    for ep in ORDER:
        c = CLIPS[ep]
        fok, fdet = frame_ref_content(ep)
        fsha = sha256(FRAME_REFS / f"episode_{ep:06d}.npz")
        want = (rep.get("episodes", {}).get(f"{ep:02d}") or {}).get("frame_ref_sha256")
        sok, sdet = scene_decision(ep, rep)
        p = parquet(ep)
        ent = lock_entry(ep)
        if ent:
            rstat = f"locked {ent['sha256'][:12]}... ({ent['checked']})"
        elif p.is_file():
            rstat = "parquet present, NOT locked"
        else:
            rstat = "to retarget (soma_to_g1.py @f2923d6) + rcheck_v2 G1-G8"
        row = {"episode": ep, "object": c["obj"], "sequence_id": c["seq"], "frames": c["frames"], "wave": c["wave"],
               "wave3_position": pos.get(ep), "t1_c3r2_help_zero_since": c["t1_hz"], "t2raw_c3r2_help_zero_since": c["t2_hz"],
               "support": c["support"], "scene_support_override": "false", "scene_ok": sok, "scene_detail": sdet,
               "flags": c["flags"], "frame_ref": f"episode_{ep:06d}.npz", "frame_ref_sha256": fsha,
               "frame_ref_sha_eq_build": fsha == want, "frame_ref_content_ok": fok, "frame_ref_content": fdet,
               "retarget": rstat, "obj_lowest_vertex_above_plane_cm_frame0_nongt": (ng.get(ep) or {}).get("obj_lowest_vertex_above_plane_cm_frame0"),
               "prereg": prereg_of(ep)}
        bad += not (sok and fok and fsha == want)
        rows.append(row)
        print(f"PREP ep{ep:02d} w{c['wave']} pos {pos.get(ep, '-')!s:>2} {c['obj']:<19} T={c['frames']:<4} t1hz {c['t1_hz']:<5} "
              f"scene {'floor' if c['support'] == 'floor' else 'MISSING->floor'} ({'ok' if sok else 'FAIL'}) | frame_ref sha=build "
              f"{fsha == want} content {'ok' if fok else 'FAIL'} ({fdet}) | {rstat} | flags {'; '.join(c['flags']) or '-'}")
    print(f"PREP {len(rows) - bad}/{len(rows)} clips ok (frame references present, = build, = poses.npy; scene decision consistent)")
    if write:
        W3DIR.mkdir(parents=True, exist_ok=True)
        atomic_write(W3DIR / "prep_wave3.json", json.dumps({"written": now(), "rows": rows}, indent=1, default=str))
        keys = [k for k in rows[0] if k not in ("scene_detail", "frame_ref_content")]
        with (W3DIR / "prep_wave3.tsv.tmp").open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys, delimiter="\t", extrasaction="ignore")
            wr.writeheader()
            for r in rows:
                wr.writerow({k: ("; ".join(r[k]) if isinstance(r[k], list) else r[k]) for k in keys})
        os.replace(W3DIR / "prep_wave3.tsv.tmp", W3DIR / "prep_wave3.tsv")
        print(f"wrote {W3DIR / 'prep_wave3.json'} and prep_wave3.tsv")
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


UNLAUNCHED = ("NEEDS_RETARGET", "NEEDS_RCHECK", "FRESH", "RETARGETING")


def wave_guard(req: set[int]) -> tuple[bool, str]:
    """A requested wave W may launch only when every earlier wave that is NOT also requested has no unlaunched clip
    (v1 rule 'wave 2 needs wave 1 launched', extended: wave 3 needs waves 1 and 2 launched or settled)."""
    scan_trainers_once()
    ok, msgs = True, []
    for w in sorted(req):
        for e in range(1, w):
            if e in req:
                continue
            unl = [f"ep{ep:02d} {clip_state(ep, {e})[0]}" for ep in ORDER if CLIPS[ep]["wave"] == e
                   and clip_state(ep, {e})[0] in UNLAUNCHED]
            if unl:
                ok = False
                msgs.append(f"wave {e} has {len(unl)} clip(s) not yet launched ({', '.join(unl)}): run --wave {e} first")
            else:
                msgs.append(f"wave {e} launched or settled")
    return ok, "; ".join(dict.fromkeys(msgs)) or "no earlier wave outside the request"


def state(waves: set[int]) -> int:
    scan_trainers_once()
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


def group_of(ep: int) -> str:
    """Wave-3 reporting group (fixed by PREREG_wave3): unflagged / support_missing (floor-only) / readme_flagged."""
    if CLIPS[ep]["wave"] != 3:
        return f"wave{CLIPS[ep]['wave']}"
    if ep in README_FLAGS:
        return "readme_flagged"
    return "support_missing" if CLIPS[ep]["support"] == "missing" else "unflagged"


def wave_verdict(rs: list[str]) -> str:
    if all(x == "PASS" for x in rs):
        return "C3r2 TRAINS on the opus reconstruction for these clips"
    if "FAIL" in rs:
        return "NOT shown: C3r2 does not train on the opus reconstruction for every clip of this set"
    if all(x == "NOT_LAUNCHED" for x in rs):
        return "NOT LAUNCHED"
    if "PENDING" in rs or "NOT_LAUNCHED" in rs:
        return "PENDING"
    return "INCOMPLETE"


def verdict(write: bool) -> int:
    ok, why = prereg_status()
    if not ok:
        print(f"VERDICT REFUSED: {why} (the rule is fixed by PREREG.md / PREREG_wave3.md; nothing written)")
        return 2
    clips, clean, fail_rows = {}, {}, []
    pos = {e: i + 1 for i, e in enumerate(WAVE3_ORDER)}
    for ep in ORDER:
        c = CLIPS[ep]
        run = run_dir(ep)
        fin = H.final_marker(run) if H.done_reason(run, ITERS) else None
        crashed = (A / f"CRASHED_ep{ep:02d}").exists()
        rfailed = (A / f"RETARGET_FAILED_ep{ep:02d}").exists()
        launched = ep in read_pids()
        r = {"episode": ep, "object": c["obj"], "wave": c["wave"], "prereg": prereg_of(ep), "wave3_position": pos.get(ep),
             "group": group_of(ep), "support": c["support"], "flags": c["flags"],
             "t1_help_zero_since": c["t1_hz"], "t2raw_help_zero_since": c["t2_hz"],
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
            ref, note = ref_curve(tag, runs, ep, write) if mine else ({}, "")
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
        eps = [e for e in ORDER if CLIPS[e]["wave"] == wv]
        rs = [clips[f"{e:02d}"]["result"] for e in eps]
        waves[str(wv)] = {"clips": [f"{e:02d}" for e in eps], "results": rs, "verdict": wave_verdict(rs),
                          "prereg": PREREG3.name if wv == 3 else PREREG.name,
                          "counts": {k: rs.count(k) for k in ("PASS", "FAIL", "INCOMPLETE", "PENDING", "NOT_LAUNCHED") if rs.count(k)}}
    groups = {}
    for g in ("unflagged", "support_missing", "readme_flagged"):
        eps = [e for e in WAVE3_ORDER if group_of(e) == g]
        rs = [clips[f"{e:02d}"]["result"] for e in eps]
        groups[g] = {"clips": [f"{e:02d}" for e in eps], "results": rs, "verdict": wave_verdict(rs),
                     "counts": {k: rs.count(k) for k in ("PASS", "FAIL", "INCOMPLETE", "PENDING", "NOT_LAUNCHED") if rs.count(k)}}
    waves["3"]["groups"] = groups
    out = {"written": now(), "prereg": str(PREREG), "prereg_sha256": sha256(PREREG), "prereg_wave3": str(PREREG3),
           "prereg_wave3_sha256": sha256(PREREG3), "prereg_check": why,
           "rule": (f"clip PASS iff (a) final.json reason help_zero_tail with help_zero_since <= {HZ_LIMIT} and (b) ONE official "
                    "evaluation of final.json's checkpoint against its own (opus) input: world_count 1, SP-SR = MP-SR = 1, evaluator "
                    "reference = the retarget-locked parquet; FAIL if (a) or (b) is false; a set 'trains' iff all its clips PASS "
                    "(waves 1-2: PREREG.md; wave 3: PREREG_wave3.md, same rule)"),
           "waves": waves, "clips": clips}
    lines = [f"Tier-3 opus test ({ARM}), waves 1-3 - verdict by the pre-registered rule ({why})", f"written {out['written']}", ""]
    for wv, d in waves.items():
        lines.append(f"WAVE {wv} ({d['prereg']}; {' '.join('ep' + x for x in d['clips'])}): {d['verdict']} {d['counts']}")
    for g, d in groups.items():
        lines.append(f"  wave 3 group {g:<15} ({' '.join('ep' + x for x in d['clips'])}): {d['verdict']} {d['counts']}")
    lines.append("")
    for r in clips.values():
        lines.append(f"ep{r['episode']:02d} {r['object']:<18} w{r['wave']}{('#' + str(r['wave3_position'])) if r['wave3_position'] else '':<4} "
                     f"{r['result']:<12} (a) help zero {r.get('help_zero_since')} "
                     f"(limit {HZ_LIMIT}; Tier-1 C3r2 {r['t1_help_zero_since']}, raw Tier-2 {r['t2raw_help_zero_since']}) -> {r['a_pass']}"
                     f"{' strong' if r.get('strong') else ''} | (b) own-input SP-SR {r.get('eval_sp_sr')} MP-SR {r.get('eval_mp_sr')} "
                     f"worlds {r.get('eval_world_count')} ckpt it {r.get('evaluated_iteration')} -> {r['b_pass']} | post {r['post_status']}"
                     f"{' | CRASH MARKER' if r['crash_marker'] else ''}{' | RETARGET FAILED' if r['retarget_failed_marker'] else ''}"
                     f"{(' | flags: ' + '; '.join(r['flags'])) if r['flags'] else ''}")
    lines += ["", "Reported, not decided: own-input AUC / MPPE and training fail rates (matched 1000-iteration bins):"]
    for r in clips.values():
        if not r["launched"] and r["result"] == "NOT_LAUNCHED":
            continue
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
                                       "verdict, never orders or chooses clips. input = the opus track itself scored as a perfect rollout "
                                       "(input_quality_LOCAL); Tier-1 = C3r2_wide official final; raw Tier-2 = C3r2_tier2 final, mean of 3 "
                                       "evaluations. Wave-3 SUPPORT_MISSING_FLOOR_ONLY clips were trained without the support surface.",
             "t3opus": {f"{ep:02d}": clean[ep] for ep in ORDER},
             "input_clean_add_auc": {f"{ep:02d}": input_clean_auc(ep) for ep in ORDER},
             "tier1_c3r2_official": {f"{ep:02d}": t1.get(ep) for ep in ORDER},
             "tier2_raw_mean3": {f"{ep:02d}": base.get(ep) for ep in ORDER}}, indent=1, default=str))
        print(f"wrote {A / 'VERDICT.json'}, VERDICT.txt, fail_rates.csv, VERDICT_clean_LOCAL_NOT_FOR_UPLOAD.json")
    return 0


def heartbeat(waves: set[int]) -> int:
    scan_trainers_once()
    parts, waiting = [], {}
    for ep in ORDER:
        s, _ = clip_state(ep, waves)
        if s == "DEFERRED":
            continue
        if s in ("NEEDS_RETARGET", "NEEDS_RCHECK", "FRESH", "RETARGETING"):
            waiting.setdefault(s, []).append(f"{ep:02d}")
            continue
        pr = progress(ep)
        it = pr["it"] if pr["it"] is not None else pr["last_model"]
        parts.append(f"ep{ep:02d} {s} it={it} hz={pr['help_zero_iters']} fail={round(pr['fail'], 3) if isinstance(pr['fail'], float) else pr['fail']}")
    parts += [f"{k} {' '.join(v)}" for k, v in waiting.items()]
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
        return heartbeat(waves_of(opt("--waves", "1,2,3")))
    if sub == "wave-guard":
        ok, why = wave_guard(waves_of(opt("--waves")))
        print(("OK " if ok else "REFUSE ") + why)
        return 0 if ok else 1
    if sub == "prep":
        return prep("--write" in rest)
    if sub == "order-check":
        rule = wave3_order_by_rule()
        print(f"ORDER  {' '.join(f'{e:02d}' for e in WAVE3_ORDER)}\nRULE   {' '.join(f'{e:02d}' for e in rule)}\n"
              f"{'SAME' if rule == WAVE3_ORDER else 'DIFFERENT'} (rule: unflagged first, then flagged; Tier-1 C3r2 help_zero_since "
              f"ascending; ties by episode index)")
        for i, e in enumerate(WAVE3_ORDER, 1):
            print(f"  {i:>2}. ep{e:02d} {CLIPS[e]['obj']:<19} t1_hz {CLIPS[e]['t1_hz']:<5} group {group_of(e):<15} "
                  f"flags {'; '.join(CLIPS[e]['flags']) or '-'}")
        return 0 if rule == WAVE3_ORDER else 1
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
    print(f"unknown subcommand {sub}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
