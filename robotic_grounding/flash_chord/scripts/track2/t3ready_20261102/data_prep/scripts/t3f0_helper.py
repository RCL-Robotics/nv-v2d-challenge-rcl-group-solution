#!/usr/bin/env python3
"""t3f0_helper.py: checks, training plans, state, post-training step and verdict of scripts/t3f0_driver.sh, the Tier-3
f0 test (arm C3r2_tier3_f0): C3r2 on Han Zhou's tier3_frame0_supports_20261024 release (opus_v2 object track + frame-0
object-pose correction; human unchanged), all 30 episodes, with Han's supports for the 14 placement episodes.
Pre-registration: tmp_analysis/tier3/f0/PREREG_f0.md (sha256 in PREREG_f0.sha256).

  NEW FILE (2026-10-24), copied from scripts/t3opus_helper_v2.py (left untouched: the opus_v2 driver v2 executes it).
  Kept from v2: C3r2 exactly as C3r2_tier2 (code/fc_r2 @61fa5d3, experiment g1_recon_body_coord_rsi_r2, seed 42, 45000,
  deadline 35156 / zero_ramp 3906 / help-zero tail 5860); the fill rule (< 3 trainings of anybody before a start,
  >= 16.5 GB + 20 GB free + ramp debt, never GPU 4, no MPS, 2 own-NUMA cores, NUMA-1 GPUs first); the retarget lock
  (training and evaluation only on locked parquets); POST (export 1e-4 / 2e-4 if refused + ONE official evaluation against
  its own input + LOCAL clean diagnostic only into *_LOCAL_NOT_FOR_UPLOAD files); CRASHED marker; no automatic relaunch;
  never a signal to anything but its own evaluation session on timeout.
  Changed for f0: data root data/human_motion_data_tier3_f0 (READY write-once; frame_references_f0); runs
  fc_runs/C3r2_tier3_f0; analysis tmp_analysis/tier3/f0/; NO retarget stage (all 30 parquets were retargeted and locked
  by the prep step: tmp_analysis/tier3/f0/retarget/parquet_lock.json, scripts/t3f0_rcheck.py G1-G10); ORDER = episode
  index 00..29 (no Tier-1 information); scene.support per the root's SCENE_SUPPORT.json (true for the 14 support
  episodes, false for the 16 floor-only ones); the pre-registered hand-object mitigation
  termination.palm_position_threshold=0.25 for the 8 episodes of align/mitigation_f0.json (sha256 pinned); report-only
  comparison with the opus_v2 run of the same episode and the raw Tier-2 run (no Tier-1 reference curves); one eval per GPU
  across the t3f0 / t3opus / k3* / rolling-start lanes; waiting.json + ALL_LAUNCHED marker for the Tier-2 launchers.

Run with conda_envs/flashchord/bin/python -I -B. It never starts a training. It writes only under tmp_analysis/tier3/f0/
(and, in 'post', the run's own export / evaluation dirs). nvidia-smi / /proc / CPU-slice primitives come from
scripts/track2_queue_tier2v2_helper.py (imported read-only).

Subcommands:
  check                    driver preflight (PREREG_f0 sha256, code pins + tree digest, experiment values, data READY /
                           FILES.sha256 (full verification) / SCENE_SUPPORT.json, manifest, frame references, scene rule,
                           retarget lock + rcheck of all 30, mitigation file + align hashes, cfgcheck_f0, scripts)
  cfgcheck [--no-write]    hydra-compose the raw Tier-2 and Tier-1 overrides (must reproduce their saved config.yaml) and
                           this test's argv per clip (all 30); diff; writes f0/cfgcheck_f0.json
  prep [--write]           per-episode table (frame references incl. content = poses.npy, lock state, scene, mitigation,
                           flags); --write: f0/prep_f0.{json,tsv}
  order                    the fixed launch order (episode index)
  cmd --ep NN              the training argv, NUL-separated
  seq --ep NN              the sequence id
  plan --ep NN [--exclude CPUS] [--assume g:n,..] [--ignore-pids P,..]   "OK <gpu> <cpus> <nth> - | why" | "WAIT why"
                           (--ignore-pids: DRY RUN ONLY, simulate those processes (and their children) gone)
  launch-ok --ep NN        exit 0 iff the parquet's sha256 equals the retarget lock and its rcheck passed
  gpus [--ignore-pids P,..]   per-GPU view
  state                    one line per clip "CLIP NN STATE detail"
  trainer --ep NN          live trainer pid(s) of that run dir
  crash --ep NN --why T    write f0/CRASHED_epNN once (no relaunch is ever made)
  post --ep NN             export (1e-4; 2e-4 if refused, recorded) + ONE official evaluation of final.json's checkpoint
                           against its own input on its training GPU, then the LOCAL clean diagnostic
  verdict [--no-write]     the PREREG_f0 rule; exit 2 (nothing written) when its sha256 does not match
  heartbeat                one status line
  waiting [--write]        un-held FRESH clips (what the Tier-2 launchers must yield to); --write: f0/waiting.json and the
                           f0/ALL_LAUNCHED marker (present iff no clip is FRESH, held or not)
  t3-run-drivers           live t3opus / t3f0 drivers in RUN mode (pid, script, args); exit 1 if a t3opus one is alive
  opus-trainers            the live opus_v2 trainers (fc_runs/C3r2_tier3_opus/epNN): pid, gpu, age, progress, MPS check
  score-selftest --work D  dry run only: the LOCAL scoring path on the Tier-3 pilot's probe output (ep14, agent_T1)
"""
from __future__ import annotations

import csv, difflib, fcntl, glob, hashlib, json, math, os, re, signal, subprocess, sys, time
from pathlib import Path

sys.dont_write_bytecode = True
ME = Path("/root/autodl-tmp/haoran.zhang")
sys.path.insert(0, str(ME / "scripts"))
import track2_queue_tier2v2_helper as H  # noqa: E402  read-only primitives

ARM = "C3r2_tier3_f0"
EXP = "g1_recon_body_coord_rsi_r2"
ITERS = 45000
SEED = 42
A = ME / "tmp_analysis/tier3/f0"
RUNS = ME / "fc_runs" / ARM
EVROOT = A / "eval"
CODE = ME / "code/fc_r2"
PIN = "61fa5d31ff8054bf1e5cf5375ef0957e54d2729b"
NV = ME / "code/nv-v2d-challenge-codebase/robotic_grounding"
NV_PIN = "f2923d61817e61eb37d939e0f75e5b4a7aaa5cbf"
DATA = ME / "data/human_motion_data_tier3_f0"
FRAME_REFS = DATA / "frame_references_f0"
READY_SHA = "ee58a62deb3a0bb06248d221e1184de751f32c171410fa2616f057c4b3b2e50b"
FILES_SHA = "f9c94e8550e03223c96e101e155038537efeedda3b5213d96cecc4dcbafaba49"
SCENE_TABLE = DATA / "SCENE_SUPPORT.json"
SCENE_SHA = "bce8e9ebfc8d61bc177758e1e397fea7d5c164f25e0a4d4f6c3748eff6469cdb"
SUPPORT_FILES = DATA / "SUPPORT_FILES.sha256"
SUPPORT_FILES_SHA = "5deea2b0cfa50d5c5f07a825844d64c4b9058623a287074df31cbc8e804823d5"
OPUS_DATA = ME / "data/human_motion_data_tier3_opus"
OPUS_RUNS = ME / "fc_runs/C3r2_tier3_opus"
OPUS_A = ME / "tmp_analysis/tier3/opus_easy"
T1_RUNS = ME / "fc_runs/C3r2_wide"      # cfgcheck only (recipe identity of the saved config); never a decision
T2_RUNS = ME / "fc_runs/C3r2_tier2"
T2_ROOT = ME / "data/human_motion_data_tier2"
SHARPA = CODE / "logs/rl/sharpa"
PREREG = A / "PREREG_f0.md"
PREREG_SHA = A / "PREREG_f0.sha256"
PIDS = A / "pids.tsv"
RDIR = A / "retarget"
LOCK = RDIR / "parquet_lock.json"
LOCK_SHA = "bdb83b0ee34f05a8408a871cba3d8ca2bb4cef6ec0e67ca435a57232b3e0b0d7"
CFGCHECK = A / "cfgcheck_f0.json"
BUILD_REPORT = A / "build/build_report.json"
BUILD_REPORT_SHA = "8aebef6643cf8767c578631e406f53e3f494ef2900deb110e4d44fdd850b4d69"
FINALIZE_REPORT = A / "build/finalize_report.json"
PREP_TABLE = A / "build/prep_table.json"
PREP_TABLE_SHA = "61bf2db027ca679f2604c3a25db166b226d863de94988d9389f2b92054da72e4"
BUILD_FILES = A / "build/build_files.sha256"
OPUS_BUILD_JSON = OPUS_A / "build.json"   # code tree digest of code/fc_r2 @61fa5d3 (same code as the opus test)
MITIGATION = A / "align/mitigation_f0.json"
MITIGATION_SHA = "2ab767eb9028f3adbe9a091628a957e9bdebf21420a8bed2718e2acec6ebda0e"
ALIGN_FILES = A / "align/ALIGN_FILES.sha256"
ALIGN_FILES_SHA = "1ccbd2963ced0bc0c928307bc2a3e9988dff9873199ccb1592aec6b738cd61b6"
ALIGN_SCRIPT = ME / "scripts/t3f0_align.py"
ALIGN_SCRIPT_SHA = "03ea75fcffb280d722405052f4a05cd25c07abeafb2e0edecc016d9a56531906"
MITIGATED = (1, 2, 3, 15, 17, 19, 20, 28)
MIT_KEY = "termination.palm_position_threshold"
MIT_VALUE = 0.25
PALM_DEFAULT = 0.15
KIT = ME / "v2d_submission_kit_src/v2d_submission_kit"
STOCK = ME / "code/video_to_data_8c778f9/robotic_grounding/flash_chord"
DATASET = ME / "data/video_to_data_challenge"
KPY = ME / "conda_envs/v2dkit/bin/python"
FCPY = ME / "conda_envs/flashchord/bin/python"
CLEAN_SCORER = ME / "scripts/track2_tier2_clean_scores.py"
RCHECK = ME / "scripts/t3f0_rcheck.py"
TIER1_CSV = ME / "fc_runs/eval_round2/C3r2_wide_final.csv"            # LOCAL clean file only (as v2)
BASELINE_CSV = ME / "tmp_analysis/tier2/early_scores/early_table.csv"  # LOCAL clean file only (as v2)
RUNIN_EVALLOCK = ME / "fc_runs/queue_runin/evallock"  # opened read-only, never created
OWN_EVALLOCK = A / "evallock"
SHARED_EVALLOCKS = (RUNIN_EVALLOCK, OPUS_A / "evallock", ME / "tmp_analysis/tier2/k3ca/train/evallock",
                    ME / "tmp_analysis/tier2/k3fix_pilot/evallock", ME / "tmp_analysis/tier2/k3full/evallock",
                    ME / "tmp_analysis/tier2/k3easy/evallock")  # each gpuN.lock opened read-only if present, never created
COMMIT = "https://github.com/local-placeholder/not-uploaded/commit/0000000000000000000000000000000000000000"
EXPORT_TOLS = (1e-4, 2e-4)
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
CORES = 2
EVAL_MIN_FREE_MB = 4096
UNKNOWN_TRAIN_AGE_S = 1800
RAMP_S = 900
SPI = 4096 * 24
HZ_LIMIT = 30000
HZ_STRONG = 20000
TERMINAL = ("done", "done_score_failed", "export_refused", "failed")
CFG_ALLOWED = {"task.parquet", "output_dir", "logging.mode", "logging.project", "logging.experiment", "logging.local_jsonl"}
CFG_ALLOWED_T1 = CFG_ALLOWED | {"logging.run_name"}
CFG_ALLOWED_MIT = {MIT_KEY}  # only for the 8 MITIGATED clips
SUPPORT_EPISODES = (1, 2, 3, 6, 7, 9, 11, 13, 16, 17, 20, 25, 27, 28)  # SCENE_SUPPORT.json scene_support=true
HAN_ISSUES = {1: "Han: stool fit IoU 0.61", 17: "Han: cone track ~5 cm into the table (table lowered)",
              28: "Han: squeegee ends ~10 cm from the stool (GT: touching)"}
ALIGN_NOTES = {19: "align: hand 11 cm from the cane (palm > 15 cm on 58% of contact frames): config relaxation likely insufficient",
               28: "align: hand 15 cm from the squeegee (palm > 15 cm on 54% of contact frames): config relaxation likely insufficient",
               4: "align: near the threshold, not mitigated (2.84 cm, +1.96)", 13: "align: near the threshold, not mitigated (2.51 cm, +1.30)",
               16: "align: near the threshold, not mitigated (2.98 cm, +1.74)", 29: "align: no hand contact (foot push)"}
_T = [  # ep, sequence_id, object, frames, t2_hz (C3r2_tier2 final.json help_zero_since; report only)
    (0, "2026-06-04_14-16-23_tall_bar_stool_tip_up_04", "tall_bar_stool", 778, 35183),
    (1, "2026-05-18_17-06-21_vacuum_left_lean_02", "vacuum", 747, 39062),
    (2, "2026-06-02_10-04-42_blue_easter_basket_place_10", "blue_easter_basket", 980, 31549),
    (3, "2026-04-27_11-50-10_contigo_coffee_mug_handle_table_05", "contigo_coffee_mug", 781, 39062),
    (4, "2026-04-24_11-40-52_baseball_bat_rotate_05", "baseball_bat", 803, 39062),
    (5, "2026-03-18_19-36-56_g1_box_rotate_01", "big_black_box", 1003, 39062),
    (6, "2026-06-01_11-26-44_monitor_rotate_05", "monitor", 546, 28034),
    (7, "2026-05-29_15-20-43_iron_round_table_02", "iron", 896, 39062),
    (8, "2026-06-05_15-08-08_skinny_wood_chair_push_10", "skinny_wood_chair", 708, 14737),
    (9, "2026-06-04_16-45-04_big_red_bowl_under_skinny_wood_chair_03", "big_red_bowl", 870, 33468),
    (10, "2026-05-01_11-36-40_white_desk_lower_to_ground_10", "white_desk", 794, 28407),
    (11, "2026-04-24_13-56-58_light_blue_book_pick_place_09", "light_blue_book", 901, 23065),
    (12, "2026-04-25_17-20-04_black_platform_turn_05", "black_platform", 700, 31715),
    (13, "2026-06-09_10-19-09_snack_box_desk_through_06", "snack_box", 922, 39062),
    (14, "2026-05-28_14-44-09_round_table_flip_08", "round_table", 993, 39062),
    (15, "2026-05-06_11-06-15_wet_floor_sign_pick_place_ground_08", "wet_floor_sign", 715, 39062),
    (16, "2026-03-12_15-06-36_spray_bottle_left_desk_ground_02", "spray_bottle", 477, 25408),
    (17, "2026-05-01_17-04-55_traffic_cone_ground_table_02", "traffic_cone", 440, 25947),
    (18, "2026-05-06_10-00-19_blue_trash_can_drag_sideways_08", "blue_trash_can", 858, 39062),
    (19, "2026-05-19_10-19-48_cane_swing_05", "cane", 886, 39062),
    (20, "2026-06-08_16-57-34_rice_cooker_clockwise_04", "rice_cooker", 553, 13174),
    (21, "2026-04-29_11-53-57_tall_bar_stool_pull_08", "tall_bar_stool", 678, 39062),
    (22, "2026-05-22_16-52-36_step_ladder_drag_03", "step_ladder", 719, 39062),
    (23, "2026-06-03_15-08-03_toy_stroller_back_forth_ground_04", "toy_stroller", 946, 14163),
    (24, "2026-05-29_15-43-20_watering_can_pour_right_02", "watering_can", 748, 39062),
    (25, "2026-03-20_11-59-57_cyan_water_bottle_rotate_01", "cyan_water_bottle", 859, 39062),
    (26, "2026-06-05_16-05-08_skateboard_rotate_ground_10", "skateboard", 669, 29088),
    (27, "2026-05-01_11-15-05_woven_basket_ground_desk_10", "woven_basket", 565, 39062),
    (28, "2026-05-19_11-27-07_squeegee_sweep_away_from_you_03_", "squeegee", 988, 39062),
    (29, "2026-06-05_18-27-40_rolling_luggage_left_foot_push_06", "rolling_luggage", 467, 39062),
]
CLIPS = {e: {"seq": s, "obj": o, "frames": f, "t2_hz": h2, "support": e in SUPPORT_EPISODES, "mitigated": e in MITIGATED}
         for e, s, o, f, h2 in _T}
ORDER = tuple(range(30))  # fixed launch order: episode index (PREREG_f0; no Tier-1 information)


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
        raise SystemExit(f"episode {e} is not part of this test (00-29)")
    return e


def seq_dir(ep: int, root: Path = DATA) -> Path:
    return root / "whole_body/soma" / f"sequence_id={CLIPS[ep]['seq']}"


def parquet(ep: int) -> Path:
    return seq_dir(ep) / "robot_name=g1/data.parquet"


def run_dir(ep: int) -> Path:
    return RUNS / f"ep{ep:02d}"


def held(ep: int) -> bool:
    return (A / f"HOLD_ep{ep:02d}").exists()


def prereg_status() -> tuple[bool, str]:
    if not PREREG.is_file() or not PREREG_SHA.is_file():
        return False, f"{PREREG.name} or {PREREG_SHA.name} missing"
    words = PREREG_SHA.read_text().split()
    want = words[0] if words else ""
    got = sha256(PREREG)
    return got == want, f"sha256({PREREG.name}) {str(got)[:16]}... {'==' if got == want else '!='} {PREREG_SHA.name} {want[:16]}..."


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
    """The raw Tier-2 arm's command (C3r2_tier2: experiment, task.parquet, reset.seed, output_dir, max_iterations, logging)
    with the f0 parquet, this arm's output_dir and offline logging, plus hydra.run.dir / +logging.local_jsonl (as v2); the
    scene.support value of the root's SCENE_SUPPORT.json (true = the Tier-1/2 setting of the 14 support episodes, false =
    floor-only as Tier-1/2); the pre-registered mitigation override for the 8 MITIGATED clips (cfgcheck_f0)."""
    c, out = CLIPS[ep], run_dir(ep)
    argv = ["python", "scripts/train_rl.py", f"experiment={EXP}", f"task.parquet='{parquet(ep)}'", f"reset.seed={SEED}",
            f"output_dir={out}", f"training.max_iterations={ITERS}", "logging.mode=offline", "logging.entity=juicer_test",
            "logging.project=v2d-track2-tier3", f"logging.experiment={ARM}", f"logging.run_name=ep{ep:02d}_{c['obj']}",
            "logging.upload_checkpoints=false", f"scene.support={'true' if c['support'] else 'false'}"]
    if c["mitigated"]:
        argv.append(f"{MIT_KEY}={MIT_VALUE}")
    return argv + [f"hydra.run.dir={out}/hydra", "+logging.local_jsonl=true"]


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


_TRAINER_SCAN: list[tuple[int, list[str]]] | None = None


def scan_trainers_once() -> None:
    global _TRAINER_SCAN
    _TRAINER_SCAN = [(pid, H.read_cmdline(pid)) for pid in H.all_pids()]


def find_trainers(ep: int, runs: Path = RUNS) -> list[int]:
    want = f"output_dir={runs / f'ep{ep:02d}'}"
    scan = _TRAINER_SCAN if _TRAINER_SCAN is not None else [(pid, H.read_cmdline(pid)) for pid in H.all_pids()]
    return [pid for pid, cmd in scan if H.is_python_trainer(cmd) and want in cmd and H.alive(pid)]


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


def progress(ep: int, runs: Path = RUNS) -> dict:
    run = runs / f"ep{ep:02d}"
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
    ok = rc.get("all_ok") is True and rc.get("parquet_sha256") == ent["sha256"] == got and ent.get("parquet") == str(p)
    return ok, f"parquet sha256 {str(got)[:16]}... lock {ent['sha256'][:16]}... rcheck all_ok {rc.get('all_ok')}"


def scene_table() -> dict:
    try:
        return json.loads(SCENE_TABLE.read_text()).get("episodes", {})
    except (OSError, ValueError):
        return {}


def mitigation_doc() -> dict:
    try:
        return json.loads(MITIGATION.read_text())
    except (OSError, ValueError):
        return {}


_PREP_FLAGS: dict | None = None


def prep_flags(ep: int) -> list[str]:
    """Report-only flags: the prep step's (build/prep_table.json, sha256 pinned; includes Han's README known issues), the
    align notes and the mitigation. They never order, choose or decide anything."""
    global _PREP_FLAGS
    if _PREP_FLAGS is None:
        _PREP_FLAGS = {}
        try:
            for r in json.loads(PREP_TABLE.read_text()):
                _PREP_FLAGS[int(r["ep"])] = list(r.get("flags") or [])
        except (OSError, ValueError, KeyError, TypeError):
            pass
    out = list(_PREP_FLAGS.get(ep, []))
    if ep in HAN_ISSUES and not any(x.startswith("Han:") for x in out):
        out.insert(0, HAN_ISSUES[ep])
    if ep in MITIGATED:
        pe = (mitigation_doc().get("per_episode") or {}).get(str(ep)) or {}
        out.append(f"MITIGATED {MIT_KEY}={MIT_VALUE} (median contact gap f0 {pe.get('j_f0_med')} cm vs opus_v2 {pe.get('j_opus_med')} cm)")
    if ep in ALIGN_NOTES:
        out.append(ALIGN_NOTES[ep])
    return out


# ============================================================================================== GPU view / plans
def _children_closure(roots: set[int]) -> set[int]:
    """roots plus every live descendant (by /proc ppid)."""
    if not roots:
        return set()
    parent = {}
    for pid in H.all_pids():
        try:
            line = Path(f"/proc/{pid}/stat").read_text()
            parent[pid] = int(line[line.rindex(")") + 2:].split()[1])
        except (OSError, ValueError, IndexError):
            continue
    out = set(roots)
    changed = True
    while changed:
        changed = False
        for pid, pp in parent.items():
            if pp in out and pid not in out:
                out.add(pid)
                changed = True
    return out


def parse_pids(text: str | None) -> set[int]:
    return {int(x) for x in (text or "").split(",") if x.strip().isdigit()}


def observe(ignore: set[int] | None = None) -> dict[int, dict]:
    """As v2: per GPU trainings of anybody, evaluations, free memory, MPS, and the ramp debt of this test's trainers younger
    than RAMP_S (environment T3F0=1). ignore (dry run only): those pids and their children are treated as gone (their GPU
    memory counted free)."""
    ign = _children_closure(ignore or set())
    st = H.smi()
    pids = [p for p in H.all_pids() if p not in ign]
    uuid_to_idx = {v["uuid"]: k for k, v in st["gpus"].items()}
    mps = H.mps_gpus(st, pids)
    view = {g: {"free": v["free"], "total": v["total"], "uuid": v["uuid"], "trainings": [], "evals": [], "others": [],
                "mps": mps.get(g), "debt": 0} for g, v in st["gpus"].items()}
    mem = {a["pid"]: a["mem"] for a in st["apps"]}
    counted: dict[int, int] = {}
    for app in st["apps"]:
        pid, g = app["pid"], app["gpu"]
        if pid in ign:
            view[g]["free"] += app["mem"]
            continue
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
        if H.read_environ(pid).get("T3F0") == "1" and (H.proc_age(pid) or 0.0) < RAMP_S:
            view[g]["debt"] += max(0, JOB_MB - mem.get(pid, 0))
    return view


def _cpu_slice_ignoring(gpu: int, exclude: str, max_cores: int, min_cores: int, ign: set[int]) -> tuple[str, int, str]:
    """H.cpu_slice with the processes in ign skipped (dry-run simulation only)."""
    full, pinned = set(range(208)), {c % 104 for c in H.parse_cpus(exclude)}
    for pid in H.all_pids():
        if pid in ign or not H.read_cmdline(pid):
            continue
        try:
            status = Path(f"/proc/{pid}/status").read_text()
        except OSError:
            continue
        allowed = next((l.split(":", 1)[1].strip() for l in status.splitlines() if l.startswith("Cpus_allowed_list:")), "")
        if not allowed or H.parse_cpus(allowed) >= full:
            continue
        cores = {c % 104 for c in H.parse_cpus(allowed)}
        if len(cores) <= 26:
            pinned |= cores
    node = 0 if gpu <= 2 else 1
    pools = [H.parse_cpus(H.TRAIN_POOLS[0]), H.parse_cpus(H.TRAIN_POOLS[1])]
    own, other = sorted(pools[node] - pinned), sorted(pools[1 - node] - pinned)
    cores, note = own[:max_cores], ""
    if len(cores) < min_cores:
        extra = other[: min_cores - len(cores)]
        cores += extra
        note = "cross-NUMA" if extra else ""
    if not cores:
        return "none", 0, "no-free-core"
    return ",".join(f"{c},{c + 104}" for c in cores), 2 * len(cores), note


def own_node_cores(gpu: int, exclude: str, ignore: set[int] | None = None) -> list[int]:
    if ignore:
        allc, _, note = _cpu_slice_ignoring(gpu, exclude, 104, CORES, _children_closure(ignore))
    else:
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


def plan(ep: int, exclude: str, assume: dict[int, int] | None = None, ignore: set[int] | None = None) -> str:
    """v2's fill rule, unchanged: allowed GPUs (never 4), no MPS, < 3 trainings of anybody before the start, >= JOB_MB +
    20 GB free (+ ramp debt of this test's young trainers), >= CORES free physical cores on the GPU's own NUMA node (never
    borrowed); tiers NUMA 1 (3/5/6/7) first, NUMA 0 (0/1/2) only if no NUMA-1 GPU fits; then fewest trainings, most free
    memory, lowest index. CPU slice: CORES physical cores + HT siblings. assume / ignore: dry run only."""
    try:
        view = observe(ignore)
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
            node_cores[node] = own_node_cores(g, exclude, ignore)
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
            f"{[c[3] for c in sorted(cands)[1:]]}; excluded: {'; '.join(why) or '-'}"
            f"{'; SIMULATED with pids ' + ','.join(map(str, sorted(ignore))) + ' gone' if ignore else ''}")


def gpus_report(ignore: set[int] | None = None) -> list[str]:
    try:
        view = observe(ignore)
    except Exception as error:  # noqa: BLE001
        return [f"GPU_ERR nvidia-smi failed: {error}"]
    return [f"GPU {g} trainings={len(v['trainings'])} ({' '.join(map(str, sorted(v['trainings'])))}) evals={len(v['evals'])}"
            f" others={len(v['others'])} free={v['free']}MiB mps={v['mps'] or 'no'} debt={v['debt']}"
            f"{'  (never used: GPU 4)' if g not in ALLOWED_GPUS else ''}" for g, v in sorted(view.items())]


# ============================================================================================== drivers / trainers of others
def t3_run_drivers() -> list[tuple[int, str, list[str]]]:
    """Live t3opus_driver*.sh / t3f0_driver.sh processes in RUN mode (not `bash -c` wrappers; subshells deduplicated)."""
    out, seen = [], set()
    for pid in sorted(H.all_pids()):
        cmd = H.read_cmdline(pid)
        if not cmd or "-c" in cmd[:2] or "run" not in cmd or not H.alive(pid):
            continue
        name = next((os.path.basename(x) for x in cmd[:4]
                     if re.fullmatch(r"(t3opus_driver(_v\d+)?|t3f0_driver)\.sh", os.path.basename(x))), None)
        if name is None or tuple(cmd) in seen:
            continue
        seen.add(tuple(cmd))
        out.append((pid, name, cmd))
    return out


def opus_trainers() -> list[dict]:
    scan_trainers_once()
    rows = []
    for ep in ORDER:
        for pid in find_trainers(ep, OPUS_RUNS):
            env = H.read_environ(pid)
            try:
                status = Path(f"/proc/{pid}/status").read_text()
                cpus = next((l.split(":", 1)[1].strip() for l in status.splitlines() if l.startswith("Cpus_allowed_list:")), "?")
            except OSError:
                cpus = "?"
            pr = progress(ep, OPUS_RUNS)
            rows.append({"ep": ep, "pid": pid, "gpu": env.get("CUDA_VISIBLE_DEVICES"), "t3opus_env": env.get("T3OPUS"),
                         "mps_pipe_env": env.get("CUDA_MPS_PIPE_DIRECTORY"), "age_h": round((H.proc_age(pid) or 0) / 3600, 2),
                         "it": pr["it"], "last_model": pr["last_model"], "help_zero_iters": pr["help_zero_iters"],
                         "fail": round(pr["fail"], 3) if isinstance(pr["fail"], float) else pr["fail"], "cpus": cpus,
                         "cmd_output_dir": f"output_dir={OPUS_RUNS / f'ep{ep:02d}'}"})
    return rows


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
           "allowed_vs_tier1": sorted(CFG_ALLOWED_T1), "allowed_extra_mitigated": sorted(CFG_ALLOWED_MIT),
           "mitigated": list(MITIGATED), "support_episodes": list(SUPPORT_EPISODES), "episodes": {}}
    ok_all = True
    with initialize_config_dir(config_dir=str(CODE / "src/flash_chord/configs"), version_base="1.3"):
        for ep in ORDER:
            c = CLIPS[ep]
            mine_ov = [a for a in train_argv(ep)[2:] if not a.startswith("hydra.")]
            cfg = compose(config_name="train", overrides=mine_ov)
            mine = _flat(OmegaConf.to_container(cfg, resolve=False))
            try:
                palm_top = float(OmegaConf.select(cfg, "termination.palm_position_threshold"))
                palm_env = float(OmegaConf.select(cfg, "env.termination.palm_position_threshold"))
            except Exception:  # noqa: BLE001
                palm_top = palm_env = None
            want_palm = MIT_VALUE if c["mitigated"] else PALM_DEFAULT
            row = {"argv": train_argv(ep)[2:], "support": c["support"], "mitigated": c["mitigated"],
                   "palm_position_threshold": {"termination": palm_top, "env.termination": palm_env, "want": want_palm}}
            ok = (mine.get("task.parquet") == str(parquet(ep)) and mine.get("output_dir") == str(run_dir(ep))
                  and mine.get("logging.mode") == "offline" and mine.get("scene.support") is c["support"]
                  and mine.get("reset.seed") == SEED and mine.get("training.max_iterations") == ITERS
                  and palm_top == palm_env == want_palm)
            allowed_extra = CFG_ALLOWED_MIT if c["mitigated"] else set()
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
                diff = [{"key": k, tag: saved.get(k, "<absent>"), "t3f0": mine.get(k, "<absent>")}
                        for k in sorted(set(saved) | set(mine)) if saved.get(k, "<absent>") != mine.get(k, "<absent>")]
                dk = {d["key"] for d in diff}
                scene_same = saved.get("scene.support") is c["support"] and "scene.support" not in dk
                mit_ok = (MIT_KEY in dk) if c["mitigated"] else (MIT_KEY not in dk)
                good = repro and dk <= (allowed | allowed_extra) and scene_same and mit_ok
                ok &= good
                row[tag] = {"ok": good, "hydra_dir": str(hyd), "compose_reproduces_saved_config": repro, "n_keys": len(saved),
                            "diff": diff, "overrides": ov, "scene_same_as_raw": scene_same, "mitigation_diff_ok": mit_ok,
                            "overrides_unified_diff": list(difflib.unified_diff(ov, mine_ov, tag, "t3f0", lineterm="", n=20))}
                print(f"== ep{ep:02d} {c['obj']} vs {tag} ({hyd.parent.name}): compose reproduces saved config {repro} "
                      f"({len(saved)} keys); differ in {len(diff)} key(s), all allowed {dk <= (allowed | allowed_extra)}; "
                      f"scene.support {saved.get('scene.support')} = here {mine.get('scene.support')} ({'same' if scene_same else 'DIFFERENT'}); "
                      f"palm {palm_top}/{palm_env} (want {want_palm}) -> {'ok' if good else 'FAIL'}")
                for d in diff:
                    print(f"   {d['key']}: {d[tag]!r} -> {d['t3f0']!r}")
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


def scene_decision(ep: int, tab: dict | None = None) -> tuple[bool, str]:
    """scene.support = SCENE_SUPPORT.json scene_support; the support file <seq>_support.usda exists with the table's sha256,
    defines a Cylinder/Cube iff support (the trainer's and the stock evaluator's rule, H.support_rule), no
    <seq>_g1_support.usda exists (it would take precedence), and the raw Tier-2 root's rule is the same."""
    c = CLIPS[ep]
    tab = scene_table() if tab is None else tab
    row = tab.get(f"{ep:02d}") or {}
    stage = DATA / "whole_body/reconstructed_stage"
    plain, g1 = stage / f"{c['seq']}_support.usda", stage / f"{c['seq']}_g1_support.usda"
    rule, t2 = H.support_rule(DATA, c["seq"]), H.support_rule(T2_ROOT, c["seq"])
    want = "true" if c["support"] else "false"
    fsha = sha256(plain)
    ok = (plain.is_file() and not g1.exists() and row.get("scene_support") is c["support"] and row.get("sequence_id") == c["seq"]
          and rule == want and t2 == want and fsha == row.get("support_sha256"))
    kind = row.get("kind", "?")
    return ok, (f"{'SUPPORT ' + str(row.get('support')) + ' (' + kind + ', ' + str(row.get('n_boxes')) + ' boxes)' if c['support'] else 'floor-only'}: "
                f"f0 rule {rule}, raw Tier-2 rule {t2}, table {row.get('scene_support')}; file sha256 {str(fsha)[:12]}... "
                f"{'=' if fsha == row.get('support_sha256') else '!='} table; no *_g1_support.usda {not g1.exists()} -> scene.support={want}")


def frame_ref_content(ep: int) -> tuple[bool, str]:
    """frame_references_f0/episode_NNNNNN.npz = the f0 poses.npy (position + rotation as xyzw) in released_track2_world, all
    frames (the construction of frame_references_opus; checked, not rebuilt)."""
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
    dr = float(np.abs(Rq - P[:, :3, :3]).max())
    return dp <= 1e-9 and dr <= 1e-4, f"T={T}, max |pos - poses.npy| {dp:.1e} m, max |R(quat) - R(poses.npy)| {dr:.1e}"


def verify_files_sha256() -> tuple[bool, str]:
    """Full verification of the root's FILES.sha256 (paths relative to the root)."""
    bad, n = [], 0
    try:
        lines = (DATA / "FILES.sha256").read_text().splitlines()
    except OSError:
        return False, "FILES.sha256 unreadable"
    for line in lines:
        h, _, rel = line.partition("  ")
        if not rel:
            continue
        n += 1
        if sha256(DATA / rel.strip()) != h:
            bad.append(rel.strip())
    return not bad and n > 0, f"{n} files, {len(bad)} mismatch(es){': ' + ', '.join(bad[:5]) if bad else ''}"


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
    add("nv_pin", nvh == NV_PIN, f"{NV} HEAD {nvh[:12] or '?'} (the retarget pin {NV_PIN[:7]} the parquets were made with)")
    try:
        want_dig = json.loads(OPUS_BUILD_JSON.read_text()).get("code_digest")
    except (OSError, ValueError):
        want_dig = None
    dig = code_digest()
    add("code_tree_digest", want_dig == dig, f"{dig[:16]}... {'==' if want_dig == dig else '!='} opus_easy/build.json (same code as the opus test)")
    expf = CODE / "src/flash_chord/configs/experiment" / f"{EXP}.yaml"
    text = expf.read_text() if expf.is_file() else ""
    for key, val in (("deadline", 35156), ("zero_ramp", 3906), ("early_stop_help_zero_tail", 5860), ("max_iterations", 45000)):
        add(f"experiment_{key}", re.search(rf"^\s+{key}: {val}\s*$", text, re.M), f"{EXP}.yaml {key}: {val}")
    term = (CODE / "src/flash_chord/configs/termination/recon_body.yaml").read_text() if (CODE / "src/flash_chord/configs/termination/recon_body.yaml").is_file() else ""
    add("termination_default_palm", re.search(r"^palm_position_threshold: 0\.15\s*$", term, re.M),
        "termination/recon_body.yaml palm_position_threshold: 0.15 (default; 0.25 only for the MITIGATED clips)")
    add("data_ready", sha256(DATA / "READY") == READY_SHA, f"{DATA}/READY sha256 {str(sha256(DATA / 'READY'))[:16]}... (want {READY_SHA[:16]}...)")
    add("data_files_sha256_file", sha256(DATA / "FILES.sha256") == FILES_SHA, f"FILES.sha256 sha256 {str(sha256(DATA / 'FILES.sha256'))[:16]}...")
    fok, fdet = verify_files_sha256()
    add("data_files_verified", fok, f"FILES.sha256 full verification: {fdet}")
    add("scene_table", sha256(SCENE_TABLE) == SCENE_SHA, f"SCENE_SUPPORT.json sha256 {str(sha256(SCENE_TABLE))[:16]}...")
    add("support_files_sha256", sha256(SUPPORT_FILES) == SUPPORT_FILES_SHA, f"SUPPORT_FILES.sha256 sha256 {str(sha256(SUPPORT_FILES))[:16]}...")
    tab = scene_table()
    add("scene_support_set", sorted(int(k) for k, v in tab.items() if v.get("scene_support") is True) == list(SUPPORT_EPISODES)
        and len(tab) == 30, f"SCENE_SUPPORT.json support episodes = {list(SUPPORT_EPISODES)}")
    try:
        fin = json.loads(FINALIZE_REPORT.read_text())
    except (OSError, ValueError):
        fin = {}
    add("finalize_report", fin.get("bad") == [] and fin.get("files_sha256_sha256") == FILES_SHA,
        f"build/finalize_report.json bad {fin.get('bad')}, FILES.sha256 sha256 {str(fin.get('files_sha256_sha256'))[:16]}...")
    rep = build_report()
    add("build_report", sha256(BUILD_REPORT) == BUILD_REPORT_SHA and rep.get("bad") == [] and len(rep.get("episodes", {})) == 30,
        f"build/build_report.json sha256 {str(sha256(BUILD_REPORT))[:16]}..., bad {rep.get('bad')}")
    add("prep_table", sha256(PREP_TABLE) == PREP_TABLE_SHA, f"build/prep_table.json sha256 {str(sha256(PREP_TABLE))[:16]}... (report-only flags)")
    bf_bad = []
    try:
        for line in BUILD_FILES.read_text().splitlines():
            h, _, rel = line.partition("  ")
            if rel and sha256(ME / rel.strip()) != h:
                bf_bad.append(rel.strip())
    except OSError:
        bf_bad.append("unreadable")
    add("prep_scripts_unchanged", not bf_bad, f"build/build_files.sha256: {'all match' if not bf_bad else 'CHANGED ' + ' '.join(bf_bad)}")
    add("retarget_lock_file", sha256(LOCK) == LOCK_SHA, f"retarget/parquet_lock.json sha256 {str(sha256(LOCK))[:16]}... (30 entries)")
    md = mitigation_doc()
    add("mitigation_file", sha256(MITIGATION) == MITIGATION_SHA and sorted(md.get("episodes", [])) == list(MITIGATED)
        and md.get("override") == f"{MIT_KEY}={MIT_VALUE}",
        f"align/mitigation_f0.json sha256 {str(sha256(MITIGATION))[:16]}...; episodes {md.get('episodes')}; override {md.get('override')}")
    add("align_files", sha256(ALIGN_FILES) == ALIGN_FILES_SHA, f"align/ALIGN_FILES.sha256 sha256 {str(sha256(ALIGN_FILES))[:16]}...")
    add("align_script", sha256(ALIGN_SCRIPT) == ALIGN_SCRIPT_SHA, f"scripts/t3f0_align.py sha256 {str(sha256(ALIGN_SCRIPT))[:16]}...")
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
    for f in (RCHECK, CLEAN_SCORER, KPY, FCPY):
        add(f"file_{f.name}", f.is_file(), str(f))
    add("clip_set", sorted(CLIPS) == list(range(30)) and ORDER == tuple(range(30)), f"{len(CLIPS)} clips, ORDER = episode index")
    for ep in ORDER:
        c, tag = CLIPS[ep], f"ep{ep:02d}"
        add(f"{tag}_manifest", seq_of.get(ep) == (c["seq"], c["frames"]), f"manifest {seq_of.get(ep)}")
        fr = FRAME_REFS / f"episode_{ep:06d}.npz"
        want = (rep.get("episodes", {}).get(f"{ep:02d}") or {}).get("frame_ref_sha256")
        add(f"{tag}_frame_ref", fr.is_file() and want is not None and sha256(fr) == want,
            f"{fr.name} sha256 {str(sha256(fr))[:12]}... = build report {str(want)[:12]}...")
        ok, detail = scene_decision(ep, tab)
        add(f"{tag}_scene", ok, detail)
        ok, detail = rcheck_ok(ep)
        add(f"{tag}_retarget_lock", ok, detail)
        fin2 = H.final_marker(T2_RUNS / f"ep{ep:02d}") or {}
        add(f"{tag}_t2_reference", fin2.get("help_zero_since") == c["t2_hz"],
            f"C3r2_tier2 help_zero_since {fin2.get('help_zero_since')} (table {c['t2_hz']}; report only)")
    bad = [r for r in rows if not r[1]]
    for name, ok, detail in rows:
        print(f"CHECK {name} {'ok' if ok else 'FAIL'} {detail}")
    print(f"CHECKS {len(rows) - len(bad)}/{len(rows)} ok" + (f"; FAILED: {' '.join(r[0] for r in bad)}" if bad else ""))
    return 0 if not bad else 1


def prep(write: bool) -> int:
    """Per-episode prep table (read-only unless --write: f0/prep_f0.{json,tsv})."""
    rep, tab, md = build_report(), scene_table(), mitigation_doc()
    rows, bad = [], 0
    for ep in ORDER:
        c = CLIPS[ep]
        fok, fdet = frame_ref_content(ep)
        fsha = sha256(FRAME_REFS / f"episode_{ep:06d}.npz")
        want = (rep.get("episodes", {}).get(f"{ep:02d}") or {}).get("frame_ref_sha256")
        sok, sdet = scene_decision(ep, tab)
        lok, ldet = rcheck_ok(ep)
        pe = (md.get("per_episode") or {}).get(str(ep)) or {}
        row = {"episode": ep, "object": c["obj"], "sequence_id": c["seq"], "frames": c["frames"], "order_position": ep + 1,
               "scene_support": c["support"], "scene_ok": sok, "scene_detail": sdet, "mitigated": c["mitigated"],
               "override": f"{MIT_KEY}={MIT_VALUE}" if c["mitigated"] else "-",
               "contact_gap_median_f0_cm": pe.get("j_f0_med"), "contact_gap_median_opus_cm": pe.get("j_opus_med"),
               "flags": prep_flags(ep), "frame_ref": f"episode_{ep:06d}.npz", "frame_ref_sha256": fsha,
               "frame_ref_sha_eq_build": fsha == want, "frame_ref_content_ok": fok, "frame_ref_content": fdet,
               "retarget_lock_ok": lok, "retarget_lock": ldet, "t2raw_c3r2_help_zero_since_report_only": c["t2_hz"]}
        bad += not (sok and fok and fsha == want and lok)
        rows.append(row)
        print(f"PREP ep{ep:02d} {c['obj']:<19} T={c['frames']:<4} scene {'SUPPORT' if c['support'] else 'floor  '} ({'ok' if sok else 'FAIL'}) "
              f"| mitigated {'YES' if c['mitigated'] else 'no '} | frame_ref sha=build {fsha == want} content {'ok' if fok else 'FAIL'} "
              f"| lock {'ok' if lok else 'FAIL'} | flags {'; '.join(row['flags']) or '-'}")
    print(f"PREP {len(rows) - bad}/{len(rows)} clips ok (frame references = build = poses.npy; scene decision consistent; parquet locked)")
    if write:
        atomic_write(A / "prep_f0.json", json.dumps({"written": now(), "rows": rows}, indent=1, default=str))
        keys = [k for k in rows[0] if k not in ("scene_detail", "frame_ref_content")]
        with (A / "prep_f0.tsv.tmp").open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys, delimiter="\t", extrasaction="ignore")
            wr.writeheader()
            for r in rows:
                wr.writerow({k: ("; ".join(r[k]) if isinstance(r[k], list) else r[k]) for k in keys})
        os.replace(A / "prep_f0.tsv.tmp", A / "prep_f0.tsv")
        print(f"wrote {A / 'prep_f0.json'} and prep_f0.tsv")
    return 0 if not bad else 1


# ============================================================================================== state / crash
def clip_state(ep: int) -> tuple[str, str]:
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
    ok, detail = rcheck_ok(ep)
    if not ok:
        ent = lock_entry(ep)
        if ent and parquet(ep).is_file() and ent["sha256"] != sha256(parquet(ep)):
            return "PARQUET_CHANGED", f"parquet sha256 differs from the retarget lock ({detail})"
        return "NOT_READY", f"no passing retarget lock ({detail}); never launched (READY root is write-once)"
    return "FRESH", f"locked ({detail}); no run files{'; HELD (HOLD_ep%02d)' % ep if held(ep) else ''}"


def state() -> int:
    scan_trainers_once()
    for ep in ORDER:
        s, d = clip_state(ep)
        print(f"CLIP {ep:02d} {s} {d}")
    return 0


def waiting(write: bool) -> int:
    scan_trainers_once()
    wait, hold = [], []
    for ep in ORDER:
        s, _ = clip_state(ep)
        if s == "FRESH":
            (hold if held(ep) else wait).append(f"{ep:02d}")
    doc = {"written": now(), "driver_pid": (A / "driver.pid").read_text().strip() if (A / "driver.pid").is_file() else None,
           "waiting": wait, "held_waiting": hold, "all_launched": not wait and not hold,
           "note": "un-held FRESH clips of the Tier-3 f0 test; the Tier-2 launchers (K3CA, T2fix) should not start while "
                   "'waiting' is non-empty (user 2026-10-24: Tier 3 first). ALL_LAUNCHED exists iff no clip is FRESH."}
    print(f"WAITING {len(wait)} ({' '.join('ep' + e for e in wait) or '-'}); HELD {len(hold)} ({' '.join('ep' + e for e in hold) or '-'})"
          f"; all launched {doc['all_launched']}")
    if write:
        atomic_write(A / "waiting.json", json.dumps(doc, indent=1))
        marker_path = A / "ALL_LAUNCHED"
        if doc["all_launched"]:
            if not marker_path.exists():
                atomic_write(marker_path, f"{now()} every Tier-3 f0 clip has been launched or settled\n")
        else:
            marker_path.unlink(missing_ok=True)
    return 0


def marker(name: str, ep: int, why: str) -> int:
    path = A / f"{name}_ep{ep:02d}"
    if path.exists():
        return 0
    log = RUNS / "logs" / f"ep{ep:02d}.log"
    try:
        tail = log.read_text(errors="replace").splitlines()[-30:]
    except OSError:
        tail = []
    pr = progress(ep)
    text = (f"{name} ep{ep:02d} {CLIPS[ep]['obj']} detected {now()}\n{why}\nlast model {pr['last_model']}, last logged iteration "
            f"{pr['it']}, help_zero_iterations {pr['help_zero_iters']}\nNo automatic relaunch / retry (PREREG_f0): the clip is "
            f"INCOMPLETE unless the user decides otherwise (disclosed).\n--- last lines of {log}\n" + "\n".join(tail) + "\n")
    atomic_write(path, text)
    print(f"wrote {path}")
    return 0


# ============================================================================================== post-training step
def child_env(gpu: int, ep: int) -> dict:
    sp = ME / "conda_envs/flashchord/lib/python3.11/site-packages/nvidia"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CUDA_MPS_", "WANDB_", "RUNIN_", "TIER2V2_", "K3EASY", "K3FIX",
                                                                    "K3CA", "T2FIX", "T3OPUS", "T3F0"))
           and k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "TIER2_GPU")}
    env.update({
        "PATH": f"{ME}/conda_envs/flashchord/bin:{env.get('PATH', '/usr/bin:/bin')}",
        "LD_LIBRARY_PATH": f"{sp}/cu13/lib:{sp}/cudnn/lib:{sp}/nccl/lib:{sp}/nvshmem/lib"
                           + (f":{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else ""),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": f"{ME}/.cache/pycache", "XDG_CACHE_HOME": f"{ME}/.cache", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": str(gpu), "HUMAN_MOTION_DATA_DIR": str(DATA), "OMP_NUM_THREADS": "4",
        "T3F0_EVAL": "1", "T3F0_EVAL_EP": f"{ep:02d}",
    })
    return env


def eval_command(ep: int, gpu: int, w: Path, tol: float) -> list[str]:
    """v2's eval command (track2/eval_all.py, stock runtime, world count 1) with this arm, the f0 manifest and the given
    tolerance. The evaluated scene is the checkpoint's own (scene.support resolved from the f0 parquet's root)."""
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
    """Own lane lock plus, read-only when present, the rolling-start / opus_v2 / k3ca / k3fix / k3full / k3easy lane lock of
    that GPU (one evaluation per GPU across these drivers); all-or-nothing, never blocking while holding."""
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
                why = f"gpu{gpu}: another t3f0 evaluation holds the lane"
            else:
                fds, taken = [mine], []
                for d in SHARED_EVALLOCKS:
                    p = d / f"gpu{gpu}.lock"
                    if not p.exists():
                        continue
                    fd = try_lock(p, create=False)
                    if fd is None:
                        why = f"gpu{gpu}: {p} is held"
                        break
                    fds.append(fd)
                    taken.append(str(p))
                if why:
                    for fd in fds:
                        os.close(fd)
                else:
                    plog(ep, f"eval lane gpu{gpu} taken (own lock + {', '.join(taken) or 'no shared lane lock file for this GPU'})")
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


def score(ep: int, w: Path) -> dict:
    """LOCAL clean diagnostic (never selects a checkpoint, never decides the verdict, never chooses f0 vs opus_v2);
    numbers only in LOCAL files."""
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
             "note": "LOCAL clean-reference diagnostic; never selects a checkpoint (final.json's is fixed), never enters the "
                     "verdict, never chooses between the f0 and opus_v2 policies (PREREG_f0)"}
    if code == 0 and rows:
        r = rows[0]
        clean.update({"clean_add_auc": fnum(r.get("t2clean_add_auc")), "clean_sp_sr": fnum(r.get("t2clean_spider_sr")),
                      "clean_mp_sr": fnum(r.get("t2clean_maniptrans_sr")), "clean_mppe_cm": fnum(r.get("t2clean_mppe_cm")),
                      "tier1_c3r2_add_auc": fnum(r.get("t1_add_auc"))})
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
               "train_gpu": gpu, "parquet": str(parquet(ep)), "parquet_sha256": psha, "mitigated": ep in MITIGATED,
               "scene_support": CLIPS[ep]["support"]})
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


def t2raw_curve(ep: int, write: bool) -> tuple[dict, str]:
    """Raw Tier-2 C3r2 fail-rate bins: the opus test's cache (read-only) if present, else our own cache, else the W&B record."""
    for cache in (OPUS_A / "ref_curves" / f"t2raw_ep{ep:02d}.json", A / "ref_curves" / f"t2raw_ep{ep:02d}.json"):
        try:
            return {int(k): v for k, v in json.loads(cache.read_text()).items()}, ""
        except (OSError, ValueError):
            continue
    files = sorted(glob.glob(str(T2_RUNS / f"ep{ep:02d}/wandb/run-*/run-*.wandb")))
    if not files:
        return {}, "no raw Tier-2 W&B record"
    try:
        b = _bins(_rows_wandb(files[-1]))
    except Exception as error:  # noqa: BLE001
        return {}, f"raw Tier-2 curve unreadable ({type(error).__name__})"
    if write and b:
        atomic_write(A / "ref_curves" / f"t2raw_ep{ep:02d}.json", json.dumps(b))
    return b, ""


def own_curve(ep: int, runs: Path = RUNS) -> dict:
    path = runs / f"ep{ep:02d}" / "metrics.jsonl"
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


def opus_v2_of(ep: int) -> dict:
    """Report-only: the opus_v2 run of the same episode (fc_runs/C3r2_tier3_opus/epNN, opus_easy/eval/epNN/post.json and the
    opus test's VERDICT.json); own-input metrics there are against the opus_v2 input, not comparable as scores."""
    run = OPUS_RUNS / f"ep{ep:02d}"
    fin = H.final_marker(run) if H.done_reason(run, ITERS) else None
    try:
        st = json.loads((OPUS_A / "eval" / f"ep{ep:02d}" / "post.json").read_text())
    except (OSError, ValueError):
        st = {}
    try:
        res = json.loads((OPUS_A / "VERDICT.json").read_text())["clips"][f"{ep:02d}"]["result"]
    except (OSError, ValueError, KeyError, TypeError):
        res = None
    pr = progress(ep, OPUS_RUNS)
    return {"result": res, "help_zero_since": fin.get("help_zero_since") if fin else None,
            "final_iteration": fin.get("iteration") if fin else None, "post_status": st.get("status"),
            "eval_sp_sr": st.get("eval_sp_sr"), "eval_mp_sr": st.get("eval_mp_sr"),
            "last_model": pr["last_model"], "last_it": pr["it"]}


def set_verdict(rs: list[str]) -> str:
    if all(x == "PASS" for x in rs):
        return "C3r2 TRAINS on the f0 reconstruction for these clips"
    if "FAIL" in rs:
        return "NOT shown: C3r2 does not train on the f0 reconstruction for every clip of this set"
    if all(x == "NOT_LAUNCHED" for x in rs):
        return "NOT LAUNCHED"
    if "PENDING" in rs or "NOT_LAUNCHED" in rs:
        return "PENDING"
    return "INCOMPLETE"


GROUPS = {"all": ORDER, "floor_only": tuple(e for e in ORDER if e not in SUPPORT_EPISODES), "support": SUPPORT_EPISODES,
          "mitigated": MITIGATED, "not_mitigated": tuple(e for e in ORDER if e not in MITIGATED)}


def verdict(write: bool) -> int:
    ok, why = prereg_status()
    if not ok:
        print(f"VERDICT REFUSED: {why} (the rule is fixed by PREREG_f0.md; nothing written)")
        return 2
    scan_trainers_once()
    clips, clean, fail_rows = {}, {}, []
    for ep in ORDER:
        c = CLIPS[ep]
        run = run_dir(ep)
        fin = H.final_marker(run) if H.done_reason(run, ITERS) else None
        crashed = (A / f"CRASHED_ep{ep:02d}").exists()
        sname, _ = clip_state(ep)
        launched = ep in read_pids()
        r = {"episode": ep, "object": c["obj"], "scene_support": c["support"], "mitigated": c["mitigated"],
             "flags": prep_flags(ep), "t2raw_help_zero_since": c["t2_hz"], "crash_marker": crashed,
             "not_ready": sname in ("NOT_READY", "PARQUET_CHANGED"), "held": held(ep), "launched": launched}
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
                    "eval_mppe_cm_own_input", "eval_chord_sr", "export_tolerance", "export_max_rel_diff",
                    "eval_reference_is_train_input", "note"):
            r[key] = st.get(key)
        if r["a_pass"] is False or b is False:
            res = "FAIL"
        elif r["a_pass"] and b:
            res = "PASS"
        elif r["not_ready"] or (crashed and fin is None) or st.get("status") == "failed":
            res = "INCOMPLETE"
        elif not launched and not fin:
            res = "NOT_LAUNCHED"
        else:
            res = "PENDING"
        r["result"] = res
        r["settled"] = res == "INCOMPLETE" or st.get("status") in TERMINAL
        r["opus_v2_report_only"] = opus_v2_of(ep)
        mine = own_curve(ep)
        oref = own_curve(ep, OPUS_RUNS)
        for tag, ref, note in (("opus_v2", oref, "" if oref else "no opus_v2 metrics"),
                               ("t2raw",) + (t2raw_curve(ep, write) if mine else ({}, ""))):
            matched = sorted(set(ref) & set(mine))
            if matched:
                rm = sum(ref[i][0] / ref[i][1] for i in matched) / len(matched)
                km = sum(mine[i][0] / mine[i][1] for i in matched) / len(matched)
                r[f"fail_rate_vs_{tag}_report_only"] = {"matched_bins": len(matched), "up_to": matched[-1], tag: round(rm, 4), "t3f0": round(km, 4)}
                for it in matched:
                    fail_rows.append({"episode": ep, "reference": tag, "iteration_bin": it, "reference_fail_rate": round(ref[it][0] / ref[it][1], 4),
                                      "t3f0_fail_rate": round(mine[it][0] / mine[it][1], 4)})
            else:
                r[f"fail_rate_vs_{tag}_report_only"] = {"matched_bins": 0, "note": note or "no t3f0 metrics yet"}
        try:
            clean[ep] = json.loads((EVROOT / f"ep{ep:02d}" / "clean_LOCAL_NOT_FOR_UPLOAD.json").read_text())
        except (OSError, ValueError):
            clean[ep] = None
        clips[f"{ep:02d}"] = r
    groups = {}
    for g, eps in GROUPS.items():
        rs = [clips[f"{e:02d}"]["result"] for e in eps]
        groups[g] = {"clips": [f"{e:02d}" for e in eps], "results": rs, "verdict": set_verdict(rs),
                     "counts": {k: rs.count(k) for k in ("PASS", "FAIL", "INCOMPLETE", "PENDING", "NOT_LAUNCHED") if rs.count(k)},
                     "strong": sum(1 for e in eps if clips[f"{e:02d}"].get("strong"))}
    out = {"written": now(), "prereg": str(PREREG), "prereg_sha256": sha256(PREREG), "prereg_check": why,
           "rule": (f"clip PASS iff (a) final.json reason help_zero_tail with help_zero_since <= {HZ_LIMIT} and (b) ONE official "
                    "evaluation of final.json's checkpoint against its own (f0) input: world_count 1, SP-SR = MP-SR = 1, evaluator "
                    "reference = the retarget-locked parquet; FAIL if (a) or (b) is false; a set 'trains' iff all its clips PASS "
                    "(PREREG_f0.md = PREREG.md's rule)"),
           "groups": groups, "clips": clips}
    lines = [f"Tier-3 f0 test ({ARM}) - verdict by the pre-registered rule ({why})", f"written {out['written']}", ""]
    for g, d in groups.items():
        lines.append(f"{g:<14} ({len(d['clips'])} clips): {d['verdict']} {d['counts']} strong {d['strong']}")
    lines.append("")
    for r in clips.values():
        o = r["opus_v2_report_only"]
        lines.append(f"ep{r['episode']:02d} {r['object']:<18} {'SUP' if r['scene_support'] else 'flr'}{' MIT' if r['mitigated'] else '    '} "
                     f"{r['result']:<12} (a) help zero {r.get('help_zero_since')} (limit {HZ_LIMIT}; raw Tier-2 {r['t2raw_help_zero_since']}) "
                     f"-> {r['a_pass']}{' strong' if r.get('strong') else ''} | (b) own-input SP-SR {r.get('eval_sp_sr')} MP-SR {r.get('eval_mp_sr')} "
                     f"worlds {r.get('eval_world_count')} ckpt it {r.get('evaluated_iteration')} -> {r['b_pass']} | post {r['post_status']}"
                     f"{' | CRASH MARKER' if r['crash_marker'] else ''}{' | NOT READY' if r['not_ready'] else ''}{' | HELD' if r['held'] else ''}"
                     f" | opus_v2: {o['result']} hz {o['help_zero_since']}")
    lines += ["", "Reported, not decided: own-input AUC / MPPE and training fail rates (matched 1000-iteration bins):"]
    for r in clips.values():
        if r["result"] == "NOT_LAUNCHED":
            continue
        f1, f2 = r["fail_rate_vs_opus_v2_report_only"], r["fail_rate_vs_t2raw_report_only"]
        lines.append(f"  ep{r['episode']:02d} AUC(own input) {r.get('eval_add_auc_own_input')} MPPE {r.get('eval_mppe_cm_own_input')} | "
                     + (f"fail rate t3f0 {f1['t3f0']} vs opus_v2 {f1['opus_v2']} ({f1['matched_bins']} bins)" if f1.get("matched_bins") else f"opus_v2 curve: {f1.get('note')}")
                     + " | " + (f"vs raw Tier-2 {f2['t2raw']} (t3f0 {f2['t3f0']}, {f2['matched_bins']} bins)" if f2.get("matched_bins") else f"Tier-2 curve: {f2.get('note')}"))
    lines.append("  clean diagnostics only in VERDICT_clean_LOCAL_NOT_FOR_UPLOAD.json")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    if write:
        atomic_write(A / "VERDICT.json", json.dumps(out, indent=1, default=str))
        atomic_write(A / "VERDICT.txt", text)
        with (A / "fail_rates.csv.tmp").open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["episode", "reference", "iteration_bin", "reference_fail_rate", "t3f0_fail_rate"])
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
                                       "verdict, never orders or chooses clips, never chooses between the f0 and opus_v2 policies. "
                                       "Tier-1 = C3r2_wide official final; raw Tier-2 = C3r2_tier2 final, mean of 3 evaluations.",
             "t3f0": {f"{ep:02d}": clean[ep] for ep in ORDER},
             "tier1_c3r2_official": {f"{ep:02d}": t1.get(ep) for ep in ORDER},
             "tier2_raw_mean3": {f"{ep:02d}": base.get(ep) for ep in ORDER}}, indent=1, default=str))
        print(f"wrote {A / 'VERDICT.json'}, VERDICT.txt, fail_rates.csv, VERDICT_clean_LOCAL_NOT_FOR_UPLOAD.json")
    return 0


def heartbeat() -> int:
    scan_trainers_once()
    parts, waiting_ = [], {}
    for ep in ORDER:
        s, _ = clip_state(ep)
        if s == "FRESH":
            waiting_.setdefault("HELD" if held(ep) else "FRESH", []).append(f"{ep:02d}")
            continue
        if s in ("NOT_READY", "PARQUET_CHANGED"):
            waiting_.setdefault(s, []).append(f"{ep:02d}")
            continue
        pr = progress(ep)
        it = pr["it"] if pr["it"] is not None else pr["last_model"]
        parts.append(f"ep{ep:02d} {s} it={it} hz={pr['help_zero_iters']} fail={round(pr['fail'], 3) if isinstance(pr['fail'], float) else pr['fail']}")
    parts += [f"{k} {' '.join(v)}" for k, v in waiting_.items()]
    print(" | ".join(parts) or "no clip")
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
    if sub == "order":
        print("\n".join(f"{e:02d}" for e in ORDER))
        return 0
    if sub == "cmd":
        sys.stdout.write("\0".join(train_argv(ep_of(opt("--ep")))) + "\0")
        return 0
    if sub == "seq":
        print(CLIPS[ep_of(opt("--ep"))]["seq"])
        return 0
    if sub == "plan":
        print(plan(ep_of(opt("--ep")), opt("--exclude", "") or "", parse_assume(opt("--assume")), parse_pids(opt("--ignore-pids"))))
        return 0
    if sub == "launch-ok":
        ok, detail = rcheck_ok(ep_of(opt("--ep")))
        print(("OK " if ok else "NO ") + detail)
        return 0 if ok else 1
    if sub == "gpus":
        print("\n".join(gpus_report(parse_pids(opt("--ignore-pids")))))
        return 0
    if sub == "state":
        return state()
    if sub == "trainer":
        print(" ".join(map(str, find_trainers(ep_of(opt("--ep"))))))
        return 0
    if sub == "crash":
        return marker("CRASHED", ep_of(opt("--ep")), opt("--why", ""))
    if sub == "post":
        return post(ep_of(opt("--ep")))
    if sub == "verdict":
        return verdict("--no-write" not in rest)
    if sub == "heartbeat":
        return heartbeat()
    if sub == "waiting":
        return waiting("--write" in rest)
    if sub == "prep":
        return prep("--write" in rest)
    if sub == "t3-run-drivers":
        rows = t3_run_drivers()
        for pid, name, cmd in rows:
            print(f"DRIVER {pid} {name} {' '.join(cmd[1:])}")
        if not rows:
            print("no t3opus / t3f0 driver in run mode")
        return 1 if any(n.startswith("t3opus") for _, n, _ in rows) else 0
    if sub == "opus-trainers":
        rows = opus_trainers()
        for r in rows:
            print("OPUS " + " ".join(f"{k}={v}" for k, v in r.items()))
        if not rows:
            print("no live opus_v2 trainer")
        return 0
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
        with (w / "score_selftest_LOCAL_NOT_FOR_UPLOAD.log").open("w") as stream:
            stream.write(" ".join(cmd) + "\n")
            stream.flush()
            code = subprocess.run(cmd, cwd=KIT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=SCORE_TIMEOUT).returncode
        try:
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
