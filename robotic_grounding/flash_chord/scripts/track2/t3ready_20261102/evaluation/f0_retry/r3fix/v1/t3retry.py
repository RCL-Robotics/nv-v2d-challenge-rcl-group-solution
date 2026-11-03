#!/usr/bin/env python3
"""t3retry.py: HELD retry queue for failing Tier-3 f0 clips (tmp_analysis/tier3/f0_retry/). BUILT 2026-10-26, NOT LAUNCHED.

Pre-registration: tmp_analysis/tier3/f0_retry/PREREG_retry.md (sha256 in PREREG_retry.sha256; this file's sha256 in
RETRY_FILES.sha256). `run` refuses to do anything if either hash does not match.

Tier-3 rule: every decision here uses ONLY Tier-3 material: the f0 input (locked parquet, object mesh, align dump), the
f0 runs' own training logs (metrics.jsonl, final.json) and own-input official evaluations (eval/epNN/post.json,
f0/VERDICT.json). guard() refuses to open any path containing 'clean', 'LOCAL_NOT_FOR_UPLOAD' or 'packed_'. No clean
scoring is done by this launcher at all.

CANDIDATE RULE (applied every pass, so clips that finish later are picked up automatically): episode NN is a retry
  candidate iff its f0 post step is done (post.json status done/done_score_failed, eval_status evaluated, world_count 1)
  and its own-input official evaluation has SP-SR == 0 or MP-SR == 0 (cross-checked with f0/VERDICT.json when that has
  the clip settled; a disagreement holds the clip).
RECIPE (computed by code, frozen in recipes/epNN.json the first time the clip is a candidate; never recomputed):
  continue the f0 final (copy of final.json's model_N + rsi_N into fc_runs/C3r2_tier3_f0_retry/epNN, sha256-checked)
  with training.resume=true, training.max_iterations=90000, training.early_stop_help_zero_tail=0,
  reset.start_update=N (a second RSI help cycle: deadline 35156 + zero_ramp 3906 counted from N, help forced to 0 by
  N + 39062 <= 84062, >= 5938 no-help iterations before 90000), everything else as the f0 run (same locked parquet, seed,
  scene.support, experiment, code), plus two mechanical termination rules (training only; evaluation has no termination):
   T-obj  termination.object_position_threshold = min(0.30, max(0.20, ceil_cm(B + 0.03))) iff B > 0.08 m, where B is
          the larger of (a) the deepest point of the object mesh below the floor (z = 0) over the f0 input and (b) the
          largest one-frame jump of the object root in the f0 input.
   T-palm termination.palm_position_threshold = 0.25 iff the clip is in the pre-registered mitigation list
          (align/mitigation_f0.json) or the f0 run's palm-termination share over its last 5000 iterations is > 0.10.
HOLD: nothing starts unless ALL of: PREREG/file hashes match; f0_retry/RETRY_GO exists and names this PREREG's sha256 and
  the user's approval; tmp_analysis/tier3/f0/ALL_LAUNCHED exists AND a live re-check finds no FRESH f0 clip (retries rank
  below every FRESH f0 clip); no f0_retry/STOP; no f0_retry/HOLD_epNN for that clip; the clip's projected end (from its
  own f0 iteration rate) is before the training cut-off 2026-11-02 12:00 CST. Placement = the t3f0 fill rule
  (t3f0_helper.plan, unchanged: never GPU 4, no MPS, < 3 trainings of anybody on the GPU before the start, memory, 2
  own-NUMA cores), under the shared GPU-choice locks (read-only), one start per pass, >= 180 s apart, episode order.
POST (after the retry's final.json): 5 + 5 own-input official evaluations, interleaved f0-final / retry, on the retry's
  training GPU, each holding our lane lock + every shared lane lock file of that GPU (read-only, all-or-nothing), 35 s
  rest after each; export 1e-4 (2e-4 only if refused, recorded); <= 3 tries per run for technical failures.
VERDICT (acceptance, own-input only): the retry final REPLACES the f0 final for the submission iff
  (i) its runs with SP-SR = MP-SR = 1 number >= 4 of 5 and more than the f0 final's, or
  (ii) its mean own-input AUC >= the f0 final's + 0.01 and its SP-SR = 1 and MP-SR = 1 counts are each >= the f0 final's.
  Otherwise (or without 5 ok runs in each arm) the f0 final stays. -> VERDICT_retry.json, SUBMISSION_CHOICE.json.
The launcher never signals any process except its own evaluation session on timeout; it never writes outside
tmp_analysis/tier3/f0_retry/ and fc_runs/C3r2_tier3_f0_retry/ (plus the trainer's own log dir under code/fc_r2/logs).

  python -I -B t3retry.py candidates | recipe --ep NN [--write] | recipes [--write] | cfgcheck [--ep NN] | gates |
                          plan --ep NN | state | dry | run | post --ep NN | verdict [--no-write] | status
Launch (ONLY after the user's GO, written to RETRY_GO by the main session): bash tmp_analysis/tier3/f0_retry/launch.sh
"""
from __future__ import annotations

import argparse, datetime as dt, fcntl, glob, hashlib, importlib.util, json, math, os, re, shutil, signal, subprocess, sys, time
from pathlib import Path

sys.dont_write_bytecode = True
ME = Path("/root/autodl-tmp/haoran.zhang")
RA = ME / "tmp_analysis/tier3/f0_retry"
F0A = ME / "tmp_analysis/tier3/f0"
RUNS_F0 = ME / "fc_runs/C3r2_tier3_f0"
RUNS_R = ME / "fc_runs/C3r2_tier3_f0_retry"
CODE = ME / "code/fc_r2"
DATA = ME / "data/human_motion_data_tier3_f0"
KIT = ME / "v2d_submission_kit_src/v2d_submission_kit"
STOCK = ME / "code/video_to_data_8c778f9/robotic_grounding/flash_chord"
DATASET = ME / "data/video_to_data_challenge"
FCPY = ME / "conda_envs/flashchord/bin/python"
KPY = ME / "conda_envs/v2dkit/bin/python"
ARM_R, ARM_F0 = "C3r2_tier3_f0_retry", "C3r2_tier3_f0"
EXP = "g1_recon_body_coord_rsi_r2"
ITERS_R = 90000
ITERS_F0 = 45000
SEED = 42
PALM_KEY, OBJ_KEY = "termination.palm_position_threshold", "termination.object_position_threshold"
PALM_DEFAULT, PALM_RELAX = 0.15, 0.25
OBJ_DEFAULT, OBJ_RELAX_MIN, OBJ_RELAX_MAX = 0.10, 0.20, 0.30
BARRIER_M, BARRIER_MARGIN_M = 0.08, 0.03
PALM_SHARE, LAST_WINDOW = 0.10, 5000
SPI = 4096 * 24
TZ8 = dt.timezone(dt.timedelta(hours=8))
CUTOFF = dt.datetime(2026, 11, 2, 12, 0, 0, tzinfo=TZ8).timestamp()  # training cut-off (user plan, 2026-10-25)
PREREG, PREREG_SHA, FILES_SHA = RA / "PREREG_retry.md", RA / "PREREG_retry.sha256", RA / "RETRY_FILES.sha256"
GO, STOPF, PIDS, LOG, HB = RA / "RETRY_GO", RA / "STOP", RA / "pids.tsv", RA / "driver.log", RA / "heartbeat.log"
RECIPES, EVR, POSTD = RA / "recipes", RA / "eval", RA / "post"
ALL_LAUNCHED = F0A / "ALL_LAUNCHED"
MITIGATION = F0A / "align/mitigation_f0.json"
GPU_CHOICE_LOCKS = [ME / "ch_runs/seedstudy/.gpulock", ME / "ch_runs/f0share05/.gpulock", ME / "ch_runs/filler/.gpulock"]
OWN_EVALLOCK = RA / "evallock"
SHARED_EVALLOCK_DIRS = [ME / "fc_runs/queue_runin/evallock", ME / "fc_runs/queue_tier2v2/evallock", F0A / "evallock",
                        ME / "tmp_analysis/tier3/opus_easy/evallock", ME / "tmp_analysis/tier2/k3ca/train/evallock",
                        ME / "tmp_analysis/tier2/k3fix_pilot/evallock", ME / "tmp_analysis/tier2/k3full/evallock",
                        ME / "tmp_analysis/tier2/k3easy/evallock"]  # + fc_runs/queue_*/evallock, fc_runs/submissions/*/evallock (globbed)
EVAL_CORES = ("48-51,152-155", "100-103,204-207")
ALLOWED_GPUS = (0, 1, 2, 3, 5, 6, 7)  # never GPU 4
RUNS_PER_ARM = 5
EXPORT_TOLS = (1e-4, 2e-4)
EVAL_TIMEOUT, EVAL_TRIES, RETRY_DELAY, COOLDOWN, LANE_POLL = 2400.0, 3, 300.0, 35.0, 30.0
POLL, HEARTBEAT_S, LAUNCH_HOLD, LOCK_WAIT, STARTING_GRACE = 60, 600, 180, 120, 180
FORBIDDEN = ("clean", "LOCAL_NOT_FOR_UPLOAD", "packed_")
TERMINAL_POST = ("done", "done_score_failed")

_T = None


def T():
    """tmp_analysis/tier3/f0's helper (scripts/t3f0_helper.py), imported read-only (never edited, never executed as a
    script): fill rule, GPU view, clip states, CLIPS, parquet locks, hydra dirs."""
    global _T
    if _T is None:
        spec = importlib.util.spec_from_file_location("t3f0h_ro", ME / "scripts/t3f0_helper.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _T = mod
    return _T


# ============================================================================================== small helpers
def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def guard(path) -> Path:
    p = Path(path)
    if any(f in str(p) for f in FORBIDDEN):
        raise PermissionError(f"t3retry refuses to open a clean / LOCAL path: {p}")
    return p


def jload(path) -> dict:
    return json.loads(guard(path).read_text())


def sha256(path) -> str | None:
    h = hashlib.sha256()
    try:
        with guard(path).open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def log(msg: str) -> None:
    line = f"[{now()}] [retry] {msg}"
    print(line, flush=True)
    try:
        with LOG.open("a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def ep2(ep: int) -> str:
    return f"{ep:02d}"


def run_r(ep: int) -> Path:
    return RUNS_R / f"ep{ep2(ep)}"


def run_f0(ep: int) -> Path:
    return RUNS_F0 / f"ep{ep2(ep)}"


def held(ep: int) -> bool:
    return (RA / f"HOLD_ep{ep2(ep)}").exists()


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


# ============================================================================================== hashes and gates
def prereg_status() -> tuple[bool, str]:
    try:
        want = PREREG_SHA.read_text().split()[0]
    except (OSError, IndexError):
        return False, f"no {PREREG_SHA.name}"
    got = sha256(PREREG)
    return got == want, f"sha256(PREREG_retry.md) {str(got)[:16]}... {'==' if got == want else '!='} {want[:16]}..."


def files_status() -> tuple[bool, str]:
    try:
        rows = dict(reversed(l.split(None, 1)) for l in FILES_SHA.read_text().splitlines() if l.strip())
    except (OSError, ValueError):
        return False, f"no {FILES_SHA.name}"
    bad = []
    for name, want in rows.items():
        name = name.strip().lstrip("*")
        if sha256(RA / name) != want.strip():
            bad.append(name)
    return not bad, ("pinned files ok: " + ", ".join(sorted(rows))) if not bad else f"pinned file(s) changed: {bad}"


def go_status() -> tuple[bool, str]:
    if not GO.is_file():
        return False, "RETRY_GO absent (nothing starts until the main session writes it after the user's approval)"
    text = GO.read_text()
    ok_sha, _ = prereg_status()
    try:
        want = PREREG_SHA.read_text().split()[0]
    except (OSError, IndexError):
        want = "?"
    if f"prereg_sha256 {want}" not in text:
        return False, "RETRY_GO does not name the current PREREG_retry sha256 (line 'prereg_sha256 <sha>')"
    if not re.search(r"^user_approval: .+", text, re.M):
        return False, "RETRY_GO lacks a 'user_approval: <verbatim quote, date>' line"
    return ok_sha, "RETRY_GO present, names this PREREG and the user's approval"


def f0_fresh_clips() -> list[str]:
    t = T()
    t.scan_trainers_once()
    out = []
    for ep in t.ORDER:
        s, _ = t.clip_state(ep)
        if s == "FRESH":
            out.append(ep2(ep))
    return out


def f0_launched_status() -> tuple[bool, str]:
    if not ALL_LAUNCHED.is_file():
        return False, "tmp_analysis/tier3/f0/ALL_LAUNCHED absent (f0 clips still waiting; retries rank below every FRESH f0 clip)"
    fresh = f0_fresh_clips()
    if fresh:
        return False, f"ALL_LAUNCHED present but a live re-check finds FRESH f0 clip(s) {fresh}: wait"
    return True, "every f0 clip launched or settled (ALL_LAUNCHED + live re-check)"


def gates() -> dict:
    g = {}
    for name, fn in (("prereg", prereg_status), ("files", files_status), ("go", go_status), ("f0_all_launched", f0_launched_status)):
        try:
            ok, why = fn()
        except Exception as error:  # noqa: BLE001
            ok, why = False, f"{type(error).__name__}: {error}"
        g[name] = {"ok": ok, "why": why}
    g["stop"] = {"ok": not STOPF.exists(), "why": "STOP present" if STOPF.exists() else "no STOP"}
    g["all_ok"] = all(v["ok"] for v in g.values() if isinstance(v, dict))
    return g


# ============================================================================================== candidate rule
def candidates() -> tuple[list[int], dict]:
    try:
        verdict = jload(F0A / "VERDICT.json").get("clips", {})
    except (OSError, ValueError):
        verdict = {}
    out, rows = [], {}
    for ep in range(30):
        pf = F0A / "eval" / f"ep{ep2(ep)}" / "post.json"
        if not pf.is_file():
            continue
        try:
            p = jload(pf)
        except (OSError, ValueError):
            continue
        done = (p.get("status") in TERMINAL_POST and p.get("eval_status") == "evaluated" and p.get("eval_world_count") == 1
                and p.get("eval_reference_is_train_input") is True)
        sp, mp = p.get("eval_sp_sr"), p.get("eval_mp_sr")
        v = verdict.get(ep2(ep), {})
        agree = True
        if v.get("post_status") == "done" and done:
            agree = v.get("eval_sp_sr") == sp and v.get("eval_mp_sr") == mp
        cand = bool(done and sp is not None and mp is not None and (float(sp) == 0.0 or float(mp) == 0.0) and agree)
        rows[ep2(ep)] = {"post_status": p.get("status"), "eval_status": p.get("eval_status"), "own_input_sp_sr": sp,
                         "own_input_mp_sr": mp, "own_input_auc": p.get("eval_add_auc_own_input"), "evaluated_iteration": p.get("evaluated_iteration"),
                         "verdict_agrees": agree, "candidate": cand}
        if cand:
            out.append(ep)
    return out, rows


# ============================================================================================== recipe
def f0_final(ep: int) -> dict:
    fin = jload(run_f0(ep) / "final.json")
    it = int(fin["iteration"])
    if fin.get("checkpoint") != f"model_{it}.safetensors" or fin.get("rsi_state") != f"rsi_{it}.safetensors":
        raise ValueError(f"ep{ep2(ep)} f0 final.json names unexpected files: {fin}")
    return fin


def input_barrier(ep: int) -> dict:
    """Object-position barriers of the locked f0 input: deepest mesh point below the floor (exact via the convex hull)
    and the largest one-frame root jump. Tier-3 input only."""
    import numpy as np
    import pyarrow.parquet as pq
    import trimesh
    t = T()
    p = t.parquet(ep)
    ent = t.lock_entry(ep) or {}
    psha = sha256(p)
    if psha != ent.get("sha256"):
        raise ValueError(f"ep{ep2(ep)} parquet sha256 {psha} != retarget lock {ent.get('sha256')}")
    row = pq.read_table(guard(p), columns=["object_body_position", "object_body_wxyz", "object_mesh_paths"]).to_pylist()[0]
    pos = np.array([f[0] for f in row["object_body_position"]], float)
    q = np.array([f[0] for f in row["object_body_wxyz"]], float)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                  np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                  np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)
    mesh_path = row["object_mesh_paths"][0]
    mesh = trimesh.load(guard(mesh_path), force="mesh", process=False)
    hull = np.asarray(mesh.convex_hull.vertices, float)
    zmin = np.einsum("fij,vj->fvi", R, hull)[:, :, 2].min(axis=1) + pos[:, 2]
    jump = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1)]
    depth = float(max(0.0, -zmin.min()))
    return {"parquet": str(p), "parquet_sha256": psha, "mesh": mesh_path, "mesh_sha256": sha256(mesh_path), "frames": int(len(pos)),
            "floor_depth_max_m": round(depth, 4), "floor_depth_max_frame": int(zmin.argmin()),
            "frames_depth_gt_barrier": int((zmin < -BARRIER_M).sum()), "jump_max_m": round(float(jump.max()), 4),
            "jump_max_frame": int(jump.argmax()), "B_m": round(max(depth, float(jump.max())), 4)}


def palm_share(ep: int) -> dict:
    rows = []
    with guard(run_f0(ep) / "metrics.jsonl").open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    last = max(round(r["step"] / SPI) for r in rows)
    sel = [r["termination/palm_fraction_per_completed_episode"] for r in rows
           if round(r["step"] / SPI) > last - LAST_WINDOW and "termination/palm_fraction_per_completed_episode" in r]
    obj = [r["termination/object_fraction_per_completed_episode"] for r in rows
           if round(r["step"] / SPI) > last - LAST_WINDOW and "termination/object_fraction_per_completed_episode" in r]
    return {"last_iteration": int(last), "rows": len(sel), "palm_share_last5k": round(sum(sel) / len(sel), 4) if sel else None,
            "object_share_last5k_report_only": round(sum(obj) / len(obj), 4) if obj else None}


def compute_recipe(ep: int) -> dict:
    t = T()
    fin = f0_final(ep)
    n = int(fin["iteration"])
    src_model, src_rsi = run_f0(ep) / fin["checkpoint"], run_f0(ep) / fin["rsi_state"]
    bar = input_barrier(ep)
    ps = palm_share(ep)
    mitig = int(ep) in [int(e) for e in jload(MITIGATION).get("episodes", [])]
    overrides, why = [], []
    if bar["B_m"] > BARRIER_M:
        thr = min(OBJ_RELAX_MAX, max(OBJ_RELAX_MIN, math.ceil((bar["B_m"] + BARRIER_MARGIN_M) * 100 - 1e-9) / 100))
        overrides.append(f"{OBJ_KEY}={thr:.2f}")
        why.append(f"T-obj: B = {bar['B_m'] * 100:.1f} cm > {BARRIER_M * 100:.0f} cm (floor depth {bar['floor_depth_max_m'] * 100:.1f} cm "
                   f"at frame {bar['floor_depth_max_frame']}, max one-frame jump {bar['jump_max_m'] * 100:.1f} cm at frame "
                   f"{bar['jump_max_frame']}) -> {OBJ_KEY}={thr:.2f}")
    else:
        why.append(f"T-obj: B = {bar['B_m'] * 100:.1f} cm <= {BARRIER_M * 100:.0f} cm -> object threshold stays {OBJ_DEFAULT}")
    share = ps["palm_share_last5k"]
    if share is None:
        raise ValueError(f"ep{ep2(ep)}: no palm-termination share in the f0 metrics")
    if mitig or share > PALM_SHARE:
        overrides.append(f"{PALM_KEY}={PALM_RELAX}")
        reasons = (["pre-registered mitigated clip (align/mitigation_f0.json)"] if mitig else []) + \
                  ([f"f0 palm-termination share over the last {LAST_WINDOW} iterations {share} > {PALM_SHARE}"] if share > PALM_SHARE else [])
        why.append(f"T-palm: {' and '.join(reasons)} -> {PALM_KEY}={PALM_RELAX}")
    else:
        why.append(f"T-palm: not mitigated and palm share {ps['palm_share_last5k']} <= {PALM_SHARE} -> palm threshold stays {PALM_DEFAULT}")
    rate = n / float(fin["wall_seconds"]) if fin.get("wall_seconds") else None
    return {"episode": ep, "object": t.CLIPS[ep]["obj"], "computed_at": now(), "script_sha256": sha256(Path(__file__)),
            "f0_final": {"iteration": n, "reason": fin.get("reason"), "help_zero_since": fin.get("help_zero_since"),
                         "model": str(src_model), "model_sha256": sha256(src_model), "rsi": str(src_rsi), "rsi_sha256": sha256(src_rsi),
                         "wall_seconds": fin.get("wall_seconds"), "iterations_per_s": round(rate, 4) if rate else None},
            "scene_support": bool(t.CLIPS[ep]["support"]), "mitigated_f0": mitig, "barrier": bar, "palm": ps,
            "termination_overrides": overrides, "rule_trace": why,
            "base": ["training.resume=true", f"training.checkpoint={run_r(ep)}/model_{n}.safetensors", f"training.max_iterations={ITERS_R}",
                     "training.early_stop_help_zero_tail=0", f"reset.start_update={n}"],
            "projected_hours": round((ITERS_R - n) / rate / 3600, 1) if rate else None}


def recipe(ep: int, write: bool, where: Path = RECIPES) -> dict:
    path = where / f"ep{ep2(ep)}.json"
    if path.is_file():
        return jload(path)
    rec = compute_recipe(ep)
    if write:
        atomic_write(path, json.dumps(rec, indent=1))
        log(f"ep{ep2(ep)} recipe frozen: {rec['termination_overrides'] or 'no termination override'} ({path})")
    return rec


def train_argv(ep: int, rec: dict) -> list[str]:
    t = T()
    out = run_r(ep)
    n = rec["f0_final"]["iteration"]
    argv = [str(FCPY), "scripts/train_rl.py", f"experiment={EXP}", f"task.parquet='{t.parquet(ep)}'", f"reset.seed={SEED}",
            f"output_dir={out}", f"training.max_iterations={ITERS_R}", "logging.mode=offline", "logging.entity=juicer_test",
            "logging.project=v2d-track2-tier3", f"logging.experiment={ARM_R}", f"logging.run_name=ep{ep2(ep)}_{t.CLIPS[ep]['obj']}_retry",
            "logging.upload_checkpoints=false", f"scene.support={'true' if rec['scene_support'] else 'false'}"]
    ov = list(rec["termination_overrides"])
    if rec["mitigated_f0"] and not any(o.startswith(PALM_KEY + "=") for o in ov):
        ov.append(f"{PALM_KEY}={PALM_RELAX}")
    argv += ov
    argv += ["training.resume=true", f"training.checkpoint={out}/model_{n}.safetensors", "training.early_stop_help_zero_tail=0",
             f"reset.start_update={n}", f"hydra.run.dir={out}/hydra", "+logging.local_jsonl=true"]
    return argv


# ============================================================================================== config check (hydra, CPU)
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


def cfgcheck(eps: list[int], recs: dict) -> dict:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    t = T()
    res = {}
    allowed = {"output_dir", "logging.experiment", "logging.run_name", "training.max_iterations", "training.resume",
               "training.checkpoint", "training.early_stop_help_zero_tail", "reset.start_update", OBJ_KEY, PALM_KEY}
    with initialize_config_dir(config_dir=str(CODE / "src/flash_chord/configs"), version_base="1.3"):
        for ep in eps:
            rec = recs[ep]
            ov = [a for a in train_argv(ep, rec)[2:] if not a.startswith("hydra.")]
            cfg = compose(config_name="train", overrides=ov)
            mine = _flat(OmegaConf.to_container(cfg, resolve=False))
            hyd = run_f0(ep) / "hydra/.hydra"  # the f0 run's own saved hydra config (hydra.run.dir=<run>/hydra)
            if not (hyd / "config.yaml").is_file():
                res[ep2(ep)] = {"ok": False, "why": "no saved hydra config of the f0 run"}
                continue
            saved = _flat(OmegaConf.to_container(OmegaConf.load(guard(hyd / "config.yaml")), resolve=False))
            diff = {k: [saved.get(k, "<absent>"), mine.get(k, "<absent>")] for k in sorted(set(saved) | set(mine))
                    if saved.get(k, "<absent>") != mine.get(k, "<absent>")}
            want_obj = next((float(o.split("=")[1]) for o in rec["termination_overrides"] if o.startswith(OBJ_KEY)), OBJ_DEFAULT)
            want_palm = PALM_RELAX if (rec["mitigated_f0"] or any(o.startswith(PALM_KEY) for o in rec["termination_overrides"])) else PALM_DEFAULT
            n = rec["f0_final"]["iteration"]
            checks = {
                "diff_keys_allowed": set(diff) <= allowed,
                "parquet_same": mine.get("task.parquet") == saved.get("task.parquet") == str(t.parquet(ep)),
                "scene_support_same": mine.get("scene.support") == saved.get("scene.support") == rec["scene_support"],
                "seed_same": mine.get("reset.seed") == saved.get("reset.seed") == SEED,
                "experiment_same": mine.get("experiment") == saved.get("experiment") if "experiment" in saved else True,
                "object_threshold": float(OmegaConf.select(cfg, OBJ_KEY)) == want_obj,
                "palm_threshold": float(OmegaConf.select(cfg, PALM_KEY)) == want_palm,
                "start_update": int(OmegaConf.select(cfg, "reset.start_update")) == n,
                "deadline_unchanged": OmegaConf.select(cfg, "reset.deadline") == 35156 and OmegaConf.select(cfg, "reset.zero_ramp") == 3906,
                "resume": OmegaConf.select(cfg, "training.resume") is True and OmegaConf.select(cfg, "training.checkpoint") == f"{run_r(ep)}/model_{n}.safetensors",
                "max_iterations": OmegaConf.select(cfg, "training.max_iterations") == ITERS_R,
                "early_stop_off": OmegaConf.select(cfg, "training.early_stop_help_zero_tail") == 0,
                "help_zero_before_cap": n + 35156 + 3906 <= ITERS_R - 5860,
            }
            res[ep2(ep)] = {"ok": all(checks.values()), "checks": checks, "f0_hydra_config": str(hyd / "config.yaml"), "diff": diff}
    return res


# ============================================================================================== state
def read_pids() -> dict[int, dict]:
    out = {}
    try:
        for line in PIDS.read_text().splitlines():
            f = line.split("\t")
            if len(f) >= 6 and f[0].isdigit():
                out[int(f[0])] = {"pid": int(f[1]), "gpu": int(f[2]), "cpus": f[3], "nth": int(f[4]), "t": int(f[5])}
    except (OSError, ValueError):
        pass
    return out


def find_trainers(ep: int) -> list[int]:
    H = T().H
    want = f"output_dir={run_r(ep)}"
    return [pid for pid in H.all_pids() if H.is_python_trainer(cmd := H.read_cmdline(pid)) and want in cmd and H.alive(pid)]


def retry_final(ep: int, rec: dict) -> dict | None:
    try:
        fin = json.loads((run_r(ep) / "final.json").read_text())
    except (OSError, ValueError):
        return None
    n = rec["f0_final"]["iteration"]
    if (fin.get("iteration") == ITERS_R and fin.get("reason") == "max_iterations" and fin.get("start_iteration") == n
            and fin.get("checkpoint") == f"model_{ITERS_R}.safetensors" and (run_r(ep) / fin["checkpoint"]).is_file()):
        return fin
    return None


def post_state(ep: int) -> dict:
    try:
        return json.loads((EVR / f"ep{ep2(ep)}" / "post.json").read_text())
    except (OSError, ValueError):
        return {}


def clip_state(ep: int, rec: dict | None) -> tuple[str, str]:
    if rec is None:
        return "NEW", "candidate; recipe not frozen yet"
    if find_trainers(ep):
        return "RUNNING", f"trainer {find_trainers(ep)}"
    fin = retry_final(ep, rec)
    if fin is not None:
        ps = post_state(ep)
        if ps.get("status") == "done":
            return "POST_DONE", ps.get("decision", "")
        if T().lock_held(POSTD / f"ep{ep2(ep)}.lock"):
            return "POST_RUNNING", ps.get("status", "")
        return "TRAINED", f"final.json max_iterations {fin['iteration']}"
    if (RA / f"CRASHED_ep{ep2(ep)}").exists():
        return "CRASHED", "marker present; no relaunch"
    if (RA / f"LATE_ep{ep2(ep)}").exists():
        return "LATE", "projected end after the training cut-off; f0 final stays"
    launched = read_pids().get(ep)
    if launched:
        if time.time() - launched["t"] < STARTING_GRACE:
            return "STARTING", f"launched {int(time.time() - launched['t'])} s ago"
        return "DEAD", "launched, no live trainer, no valid final.json"
    if held(ep):
        return "HELD", f"HOLD_ep{ep2(ep)} present"
    return "QUEUED", "recipe frozen; waits for the gates and a slot"


# ============================================================================================== launch
def take_gpu_choice_locks() -> list[int] | None:
    fds = []
    for p in GPU_CHOICE_LOCKS:
        if not p.is_file():
            continue
        fd = os.open(p, os.O_RDONLY)
        t0 = time.time()
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fds.append(fd)
                break
            except OSError:
                if time.time() - t0 > LOCK_WAIT:
                    os.close(fd)
                    for f in fds:
                        os.close(f)
                    return None
                time.sleep(2)
    return fds


def fork(ep: int, rec: dict) -> dict:
    out = run_r(ep)
    out.mkdir(parents=True, exist_ok=True)
    n = rec["f0_final"]["iteration"]
    stray = [p.name for p in out.glob("*.safetensors") if p.name not in (f"model_{n}.safetensors", f"rsi_{n}.safetensors")]
    if stray or (out / "final.json").exists():
        raise RuntimeError(f"{out} already has run files {stray or ['final.json']}: never a start over them")
    rec_path = out / "fork.json"
    files = {}
    for kind, src, want in (("model", Path(rec["f0_final"]["model"]), rec["f0_final"]["model_sha256"]),
                            ("rsi", Path(rec["f0_final"]["rsi"]), rec["f0_final"]["rsi_sha256"])):
        dst = out / src.name
        if not (dst.is_file() and sha256(dst) == want):
            if sha256(src) != want:
                raise RuntimeError(f"{src} sha256 changed since the recipe was frozen")
            tmp = out / f".{src.name}.tmp"
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
        if sha256(dst) != want:
            raise RuntimeError(f"{dst} sha256 mismatch after copy")
        files[kind] = {"src": str(src), "dst": str(dst), "sha256": want}
    doc = {"episode": ep, "forked_at": now(), "fork_iteration": n, "files": files}
    atomic_write(rec_path, json.dumps(doc, indent=1))
    return doc


def train_env(ep: int, gpu: int, nth: int) -> dict:
    sp = ME / "conda_envs/flashchord/lib/python3.11/site-packages/nvidia"
    drop = ("CUDA_MPS_", "WANDB_", "RUNIN_", "TIER2V2_", "TIER2_", "K3", "T2FIX", "T1FIX", "T3OPUS", "T3F0")
    env = {k: v for k, v in os.environ.items() if not k.startswith(drop) and k not in
           ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    env.update({"PATH": f"{ME}/conda_envs/flashchord/bin:{env.get('PATH', '/usr/bin:/bin')}",
                "LD_LIBRARY_PATH": f"{sp}/cu13/lib:{sp}/cudnn/lib:{sp}/nccl/lib:{sp}/nvshmem/lib",
                "FC_ROOT": str(STOCK), "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "WANDB_DIR": str(ME / "fc_runs/wandb_local"),
                "UV_CACHE_DIR": str(ME / ".cache/uv"), "PYTHONPATH": str(CODE / "src"), "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPYCACHEPREFIX": str(ME / ".cache/pycache"), "XDG_CACHE_HOME": str(ME / ".cache"), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "PYTHONUNBUFFERED": "1", "T3F0": "1", "T3F0R": "1", "T3F0R_EP": ep2(ep), "HUMAN_MOTION_DATA_DIR": str(DATA),
                "CUDA_VISIBLE_DEVICES": str(gpu), "WANDB_MODE": "offline", "OMP_NUM_THREADS": str(nth), "MKL_NUM_THREADS": str(nth),
                "OPENBLAS_NUM_THREADS": str(nth)})
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(k, None)
    return env


def launch(ep: int, rec: dict, gpu: int, cpus: str, nth: int) -> int:
    if gpu not in ALLOWED_GPUS:
        raise RuntimeError(f"gpu {gpu} not allowed (never 4)")
    if ep in read_pids():
        raise RuntimeError(f"ep{ep2(ep)} already launched once (pids.tsv): never a relaunch")
    fork(ep, rec)
    argv = train_argv(ep, rec)
    (RUNS_R / "logs").mkdir(parents=True, exist_ok=True)
    logf = (RUNS_R / "logs" / f"ep{ep2(ep)}.log").open("a")
    logf.write(f"[{now()}] t3retry launch gpu{gpu} cpus {cpus}: {' '.join(argv)}\n")
    logf.flush()
    proc = subprocess.Popen(["taskset", "-c", cpus] + argv, cwd=CODE, env=train_env(ep, gpu, nth), stdout=logf,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True)
    with PIDS.open("a") as f:
        f.write(f"{ep2(ep)}\t{proc.pid}\t{gpu}\t{cpus}\t{nth}\t{int(time.time())}\tlaunched\n")
    log(f"ep{ep2(ep)} LAUNCHED pid {proc.pid} on gpu{gpu} cpus {cpus}: {' '.join(argv)}")
    return proc.pid


# ============================================================================================== post: 5 + 5 own-input evaluations
def lane_lock_files(gpu: int) -> list[Path]:
    dirs = list(SHARED_EVALLOCK_DIRS)
    dirs += [Path(p) for p in glob.glob(str(ME / "fc_runs/queue_*/evallock")) + glob.glob(str(ME / "fc_runs/submissions/*/evallock"))]
    seen, out = set(), []
    for d in dirs:
        p = d / f"gpu{gpu}.lock"
        if str(p) not in seen and p.is_file() and d.resolve() != OWN_EVALLOCK.resolve():
            seen.add(str(p))
            out.append(p)
    return out


def acquire_lane(ep: int, gpu: int) -> tuple[list[int], list[str]]:
    OWN_EVALLOCK.mkdir(parents=True, exist_ok=True)
    last = ""
    while True:
        why = ""
        H = T().H
        try:
            st = H.smi()
            mps = H.mps_gpus(st, H.all_pids())
            if gpu in mps:
                why = f"gpu{gpu} has MPS"
            elif st["gpus"][gpu]["free"] < 4096 + 1536:
                why = f"gpu{gpu} free {st['gpus'][gpu]['free']} MiB < 5632"
        except Exception as error:  # noqa: BLE001
            why = f"nvidia-smi failed ({type(error).__name__})"
        if not why:
            own = try_lock(OWN_EVALLOCK / f"gpu{gpu}.lock", create=True)
            if own is None:
                why = "our own lane lock is held"
            else:
                fds, taken = [own], []
                for p in lane_lock_files(gpu):
                    fd = try_lock(p, create=False)
                    if fd is None:
                        why = f"{p} is held"
                        break
                    fds.append(fd)
                    taken.append(str(p))
                if why:
                    for fd in fds:
                        os.close(fd)
                else:
                    return fds, taken
        if why != last:
            log(f"ep{ep2(ep)} post: eval waits ({why})")
            last = why
        time.sleep(LANE_POLL)


def eval_command(ep: int, gpu: int, w: Path, arm: str, iteration: int, tol: float) -> list[str]:
    cmd = [str(FCPY), "-B", str(CODE / "scripts/track2/eval_all.py"), "--arm-dir", str(w), "--arm", arm, "--checkpoint-from-final",
           "--iteration", str(iteration), "--episodes", str(ep), "--gpus", str(gpu), "--jobs-per-gpu", "1", "--world-count", "1",
           "--export", "yes", "--export-jobs", "1", "--export-tolerance", repr(tol), "--skip-pack", "--kit-root", str(KIT),
           "--stock-runtime", str(STOCK), "--dataset-root", str(DATASET), "--manifest", f"{DATA}/whole_body/soma/track2_manifest.json",
           "--pack-python", str(KPY), "--results", str(w / "results"), "--code", str(CODE), "--python", str(FCPY)]
    return ["nice", "-n", "10", "taskset", "-c", EVAL_CORES[0 if gpu <= 2 else 1]] + cmd


def eval_env(gpu: int) -> dict:
    sp = ME / "conda_envs/flashchord/lib/python3.11/site-packages/nvidia"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CUDA_MPS_", "WANDB_", "RUNIN_", "TIER2", "K3", "T2FIX", "T3OPUS", "T3F0"))
           and k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    env.update({"PATH": f"{ME}/conda_envs/flashchord/bin:{env.get('PATH', '/usr/bin:/bin')}",
                "LD_LIBRARY_PATH": f"{sp}/cu13/lib:{sp}/cudnn/lib:{sp}/nccl/lib:{sp}/nvshmem/lib", "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPYCACHEPREFIX": f"{ME}/.cache/pycache", "XDG_CACHE_HOME": f"{ME}/.cache",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(gpu), "HUMAN_MOTION_DATA_DIR": str(DATA),
                "OMP_NUM_THREADS": "4", "T3F0R_EVAL": "1"})
    return env


def parse_run(ep: int, w: Path, want_src_sha: str) -> dict:
    t = T()
    out: dict = {}
    try:
        rep = json.loads((w / "eval_report.json").read_text())
        row = next(r for r in rep.get("episodes", []) if int(r.get("episode", -1)) == ep)
        out.update({"eval_status": row.get("status"), "iteration": row.get("iteration"), "export_max_rel_diff": row.get("export_max_rel_diff"),
                    "source_checkpoint": row.get("source_checkpoint"), "evaluated_checkpoint": row.get("evaluated_checkpoint")})
    except (OSError, ValueError, StopIteration, TypeError):
        return {"eval_status": "no_report"}
    try:
        res = json.loads((w / "results" / f"episode_{ep:06d}.json").read_text())
        m = res.get("metrics", {})
        ref = res.get("parquet")
        out.update({"auc": m.get("add_auc"), "sp_sr": m.get("spider_sr_uncentered", m.get("spider_sr")), "mp_sr": m.get("maniptrans_sr"),
                    "mppe_cm": m.get("mppe_cm", m.get("mean_per_point_error_cm")), "world_count": m.get("world_count", res.get("world_count")),
                    "non_finite_worlds": res.get("non_finite_worlds"), "reference_parquet": ref,
                    "reference_is_locked_f0_input": bool(ref) and Path(ref).resolve() == t.parquet(ep).resolve()
                    and sha256(ref) == (t.lock_entry(ep) or {}).get("sha256")})
    except (OSError, ValueError):
        pass
    src = out.get("source_checkpoint")
    out["source_sha256_ok"] = bool(src) and sha256(src) == want_src_sha
    out["ok"] = (out.get("eval_status") == "evaluated" and out.get("world_count") == 1 and not out.get("non_finite_worlds")
                 and out.get("reference_is_locked_f0_input") is True and out["source_sha256_ok"]
                 and all(isinstance(out.get(k), (int, float)) for k in ("auc", "sp_sr", "mp_sr")))
    return out


def one_eval(ep: int, gpu: int, arm_key: str, k: int, link_target: Path, arm: str, iteration: int, src_sha: str) -> dict:
    base = EVR / f"ep{ep2(ep)}" / arm_key
    for tol in EXPORT_TOLS:
        for attempt in range(1, EVAL_TRIES + 1):
            w = base / f"r{k}_tol{tol:g}_try{attempt}"
            if (w / "run.json").is_file():
                prev = json.loads((w / "run.json").read_text())
                if prev.get("ok") or prev.get("refused"):
                    if prev.get("ok"):
                        return prev
                    break
                continue
            w.mkdir(parents=True, exist_ok=True)
            for stale in (w / "eval_report.json", w / "results" / f"episode_{ep:06d}.json", w / "results" / f"episode_{ep:06d}.parquet"):
                stale.unlink(missing_ok=True)  # an interrupted earlier attempt in this dir (never a finished run: run.json absent)
            link = w / f"ep{ep2(ep)}"
            if not link.is_symlink():
                link.symlink_to(link_target)
            fds, taken = acquire_lane(ep, gpu)
            cmd = eval_command(ep, gpu, w, arm, iteration, tol)
            note, t0 = "", time.time()
            try:
                with (w / "eval_all.log").open("a") as stream:
                    stream.write(f"[{now()}] lane locks {taken}\n{' '.join(cmd)}\n")
                    stream.flush()
                    proc = subprocess.Popen(cmd, cwd=CODE, env=eval_env(gpu), stdout=stream, stderr=subprocess.STDOUT,
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
            finally:
                for fd in fds:
                    os.close(fd)
            r = parse_run(ep, w, src_sha)
            r.update({"arm": arm_key, "run": k, "tolerance": tol, "attempt": attempt, "exit": code, "note": note, "gpu": gpu,
                      "seconds": round(time.time() - t0, 1), "finished_at": now(), "lane_locks": taken})
            refused = str(r.get("eval_status", "")).startswith("export_failed") and (r.get("export_max_rel_diff") or 0) > tol
            r["refused"] = bool(refused)
            atomic_write(w / "run.json", json.dumps(r, indent=1, default=str))
            time.sleep(COOLDOWN)
            if r["ok"]:
                return r
            if refused:
                break  # next tolerance
            time.sleep(RETRY_DELAY)
    return {"arm": arm_key, "run": k, "ok": False, "note": "no ok evaluation after all tolerances / tries"}


def post(ep: int) -> int:
    rec = recipe(ep, write=False)
    fin = retry_final(ep, rec)
    if fin is None or find_trainers(ep):
        print(f"ep{ep2(ep)}: retry not finished")
        return 3
    POSTD.mkdir(parents=True, exist_ok=True)
    lock = try_lock(POSTD / f"ep{ep2(ep)}.lock", create=True)
    if lock is None:
        print(f"ep{ep2(ep)}: another post holds the lock")
        return 0
    gpu = read_pids().get(ep, {}).get("gpu")
    if gpu not in ALLOWED_GPUS:
        print(f"ep{ep2(ep)}: no allowed training GPU recorded")
        return 2
    st = post_state(ep) or {"episode": ep}
    st.update({"status": "evaluating", "gpu": gpu, "started_at": st.get("started_at") or now()})
    atomic_write(EVR / f"ep{ep2(ep)}" / "post.json", json.dumps(st, indent=1))
    f0src = run_f0(ep) / f"model_{rec['f0_final']['iteration']}.safetensors"
    rsrc = run_r(ep) / fin["checkpoint"]
    rsha = sha256(rsrc)
    runs = {"f0final": [], "retry": []}
    for k in range(1, RUNS_PER_ARM + 1):  # interleaved: f0 final, retry, f0 final, ...
        runs["f0final"].append(one_eval(ep, gpu, "f0final", k, run_f0(ep), ARM_F0, ITERS_F0, rec["f0_final"]["model_sha256"]))
        runs["retry"].append(one_eval(ep, gpu, "retry", k, run_r(ep), ARM_R, ITERS_R, rsha))
    st.update({"status": "done", "finished_at": now(), "retry_final": str(rsrc), "retry_final_sha256": rsha, "f0_final": str(f0src),
               "runs": runs})
    st["decision"] = decide(runs)
    atomic_write(EVR / f"ep{ep2(ep)}" / "post.json", json.dumps(st, indent=1, default=str))
    log(f"ep{ep2(ep)} post done: {st['decision']}")
    os.close(lock)
    verdict(write=True)
    return 0


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("ok")]
    return {"ok_runs": len(ok), "n_both": sum(1 for r in ok if r["sp_sr"] == 1 and r["mp_sr"] == 1),
            "n_sp": sum(1 for r in ok if r["sp_sr"] == 1), "n_mp": sum(1 for r in ok if r["mp_sr"] == 1),
            "auc_mean": round(sum(r["auc"] for r in ok) / len(ok), 6) if ok else None,
            "mppe_mean_cm": round(sum(r["mppe_cm"] for r in ok if isinstance(r.get("mppe_cm"), (int, float))) / len(ok), 3) if ok else None}


def decide(runs: dict) -> str:
    f, r = summarize(runs["f0final"]), summarize(runs["retry"])
    if f["ok_runs"] < RUNS_PER_ARM or r["ok_runs"] < RUNS_PER_ARM:
        return f"KEEP_F0 (incomplete: {f['ok_runs']}/{r['ok_runs']} ok runs)"
    if r["n_both"] >= 4 and r["n_both"] > f["n_both"]:
        return f"REPLACE (i): retry SP=MP=1 in {r['n_both']}/5 vs f0 final {f['n_both']}/5"
    if r["auc_mean"] >= f["auc_mean"] + 0.01 and r["n_sp"] >= f["n_sp"] and r["n_mp"] >= f["n_mp"]:
        return f"REPLACE (ii): AUC {r['auc_mean']:.4f} >= {f['auc_mean']:.4f} + 0.01, SP {r['n_sp']}>={f['n_sp']}, MP {r['n_mp']}>={f['n_mp']}"
    return (f"KEEP_F0: retry both {r['n_both']}/5 AUC {r['auc_mean']:.4f} SP {r['n_sp']} MP {r['n_mp']} vs f0 final both "
            f"{f['n_both']}/5 AUC {f['auc_mean']:.4f} SP {f['n_sp']} MP {f['n_mp']}")


def verdict(write: bool) -> int:
    ok, why = prereg_status()
    if not ok:
        print(f"verdict refused: {why}")
        return 2
    eps = {}
    choice = {}
    for ep in range(30):
        st = post_state(ep)
        rec_p = RECIPES / f"ep{ep2(ep)}.json"
        row = {"retry_candidate": rec_p.is_file(), "post": st.get("status")}
        use = {"use": "prereg_f0", "why": "PREREG_f0 submission rule (f0 final; opus_v2 only if f0 has no evaluated checkpoint)"}
        if st.get("status") == "done":
            row.update({"f0final": summarize(st["runs"]["f0final"]), "retry": summarize(st["runs"]["retry"]), "decision": st["decision"]})
            if st["decision"].startswith("REPLACE"):
                use = {"use": "retry_final", "checkpoint": st["retry_final"], "sha256": st["retry_final_sha256"], "why": st["decision"]}
            else:
                use["why"] = st["decision"]
        eps[ep2(ep)] = row
        choice[ep2(ep)] = use
    doc = {"written": now(), "prereg_sha256": PREREG_SHA.read_text().split()[0], "rule": "PREREG_retry section 5 (own-input only)",
           "episodes": eps}
    print(json.dumps({k: v.get("decision") for k, v in eps.items() if v.get("decision")}, indent=1))
    if write:
        atomic_write(RA / "VERDICT_retry.json", json.dumps(doc, indent=1, default=str))
        atomic_write(RA / "SUBMISSION_CHOICE.json", json.dumps({"written": now(), "note": "per episode: f0 final (PREREG_f0) unless "
                     "PREREG_retry's acceptance rule replaced it; own-input evidence only", "episodes": choice}, indent=1))
    return 0


# ============================================================================================== loop / dry / status
def heartbeat(states: dict, g: dict) -> str:
    s = " ".join(f"ep{k}:{v[0]}" for k, v in sorted(states.items())) or "no candidate"
    return f"gates {'OK' if g['all_ok'] else 'HOLD'} ({'; '.join(k for k, v in g.items() if isinstance(v, dict) and not v['ok']) or '-'}) | {s}"


def run_loop() -> int:
    RA.mkdir(parents=True, exist_ok=True)
    lockfd = try_lock(RA / ".driver.lock", create=True)
    if lockfd is None:
        print("another t3retry driver holds .driver.lock", file=sys.stderr)
        return 1
    if STOPF.exists():
        print("STOP present: remove it to start", file=sys.stderr)
        return 1
    for fn in (prereg_status, files_status):
        ok, why = fn()
        if not ok:
            log(f"ABORT: {why}; nothing launched")
            return 1
    atomic_write(RA / "driver.pid", f"{os.getpid()}\n")
    log(f"driver start pid {os.getpid()}")
    last_beat, last_launch, posts = 0.0, 0.0, {}
    while True:
        if STOPF.exists():
            log("STOP: driver exits; trainings / evaluations keep running")
            (RA / "driver.pid").unlink(missing_ok=True)
            return 0
        g = gates()
        if not (g["prereg"]["ok"] and g["files"]["ok"]):
            log(f"ABORT: hashes changed ({g['prereg']['why']}; {g['files']['why']})")
            return 1
        cands, _ = candidates()
        recs = {}
        for ep in cands:
            try:
                recs[ep] = recipe(ep, write=True)
            except Exception as error:  # noqa: BLE001
                log(f"ep{ep2(ep)} recipe failed: {type(error).__name__}: {error}")
        states = {}
        for ep in sorted(recs):
            states[ep2(ep)] = clip_state(ep, recs[ep])
        for ep in sorted(recs):
            s, d = states[ep2(ep)]
            if s == "DEAD":
                atomic_write(RA / f"CRASHED_ep{ep2(ep)}", f"{now()} retry trainer gone without final.json; no relaunch\n")
                log(f"ep{ep2(ep)} CRASHED (no relaunch); f0 final stays")
            elif s == "TRAINED" and (ep not in posts or posts[ep].poll() is not None):
                posts[ep] = subprocess.Popen([str(FCPY), "-I", "-B", str(Path(__file__)), "post", "--ep", ep2(ep)],
                                             stdout=(RA / f"post_ep{ep2(ep)}.log").open("a"), stderr=subprocess.STDOUT,
                                             stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True)
                log(f"ep{ep2(ep)} trained: post (5 + 5 own-input evaluations) started pid {posts[ep].pid}")
        if g["all_ok"] and time.time() - last_launch >= LAUNCH_HOLD:
            for ep in sorted(recs):
                if states[ep2(ep)][0] != "QUEUED":
                    continue
                rate = recs[ep]["f0_final"].get("iterations_per_s") or 0.0
                end = time.time() + (ITERS_R - recs[ep]["f0_final"]["iteration"]) / rate if rate else float("inf")
                if end > CUTOFF:
                    atomic_write(RA / f"LATE_ep{ep2(ep)}", f"{now()} projected end {time.strftime('%F %T', time.localtime(end))} after the cut-off\n")
                    log(f"ep{ep2(ep)} LATE: projected end after 2026-11-02 12:00; not started (f0 final stays)")
                    continue
                line = T().plan(ep, "")
                if not line.startswith("OK"):
                    break
                fds = take_gpu_choice_locks()
                if fds is None:
                    break
                try:
                    line = T().plan(ep, "")
                    if line.startswith("OK"):
                        _, g_, cpus, nth = line.split()[:4]
                        log(f"ep{ep2(ep)} slot: {line}")
                        launch(ep, recs[ep], int(g_), cpus, int(nth))
                        last_launch = time.time()
                        time.sleep(LAUNCH_HOLD)
                finally:
                    for fd in fds:
                        os.close(fd)
                break  # one start per pass
        if time.time() - last_beat >= HEARTBEAT_S:
            last_beat = time.time()
            with HB.open("a") as f:
                f.write(f"[{now()}] HEARTBEAT retry {os.getpid()} | {heartbeat(states, g)}\n")
            atomic_write(RA / "state.json", json.dumps({"written": now(), "gates": g, "states": states}, indent=1))
        time.sleep(POLL)


def dry() -> int:
    d = RA / "dryrun"
    d.mkdir(parents=True, exist_ok=True)
    rep = {"written": now(), "note": "dry run: nothing started; files only under f0_retry/dryrun/"}
    g = gates()
    rep["gates"] = g
    print("GATES:")
    for k, v in g.items():
        if isinstance(v, dict):
            print(f"  {k:16s} {'OK  ' if v['ok'] else 'HOLD'} {v['why']}")
    print(f"  => {'a launch could happen now' if g['all_ok'] else 'NOTHING can start now (held)'}")
    cands, rows = candidates()
    rep["candidates"], rep["candidate_rows"] = [ep2(e) for e in cands], rows
    print(f"CANDIDATES now (rule: POST_DONE and own-input SP-SR==0 or MP-SR==0): {rep['candidates']}")
    recs = {}
    for ep in cands:
        frozen = (RECIPES / f"ep{ep2(ep)}.json").is_file()
        recs[ep] = recipe(ep, write=False) if frozen else compute_recipe(ep)
        if not frozen:
            atomic_write(d / "recipes" / f"ep{ep2(ep)}.json", json.dumps(recs[ep], indent=1))
        r = recs[ep]
        print(f"-- ep{ep2(ep)} {r['object']}: recipe {'(frozen)' if frozen else '(computed, not frozen)'}; fork it {r['f0_final']['iteration']}; "
              f"overrides {r['termination_overrides'] or '-'}; projected {r['projected_hours']} h at the f0 rate")
        for w in r["rule_trace"]:
            print(f"     {w}")
    try:
        cc = cfgcheck(sorted(recs), recs)
    except Exception as error:  # noqa: BLE001
        cc = {"error": f"{type(error).__name__}: {error}"}
    rep["cfgcheck"] = cc
    for k, v in cc.items():
        if isinstance(v, dict):
            print(f"CFGCHECK ep{k}: {'OK' if v.get('ok') else 'FAIL'} {v.get('checks') or v.get('why')}")
            for key, (a, b) in (v.get("diff") or {}).items():
                print(f"     {key}: f0 {a!r} -> retry {b!r}")
        else:
            print(f"CFGCHECK: {v}")
    cmds = {}
    for ep in sorted(recs):
        st = clip_state(ep, recs[ep] if (RECIPES / f"ep{ep2(ep)}.json").is_file() else None)
        line = T().plan(ep, "")
        cmds[ep2(ep)] = {"state": st, "plan_now": line, "fork": {"model": recs[ep]["f0_final"]["model"], "rsi": recs[ep]["f0_final"]["rsi"],
                                                                 "to": str(run_r(ep))},
                         "train": " ".join(["taskset", "-c", "<cpus>"] + train_argv(ep, recs[ep])),
                         "eval_f0final_run1": " ".join(eval_command(ep, 0, EVR / f"ep{ep2(ep)}/f0final/r1_tol0.0001_try1", ARM_F0, ITERS_F0, 1e-4)),
                         "eval_retry_run1": " ".join(eval_command(ep, 0, EVR / f"ep{ep2(ep)}/retry/r1_tol0.0001_try1", ARM_R, ITERS_R, 1e-4))}
        print(f"COMMANDS ep{ep2(ep)} (state {st[0]}; placement now: {line[:160]}):\n  {cmds[ep2(ep)]['train']}")
    rep["commands"] = cmds
    rep["lane_lock_files_gpu0"] = [str(p) for p in lane_lock_files(0)]
    atomic_write(d / "dry_report.json", json.dumps(rep, indent=1, default=str))
    print(f"dry run done: nothing started; report {d / 'dry_report.json'}")
    return 0


def status() -> int:
    g = gates()
    cands, _ = candidates()
    print(f"gates: {'OK' if g['all_ok'] else 'HOLD'}; " + "; ".join(f"{k}: {v['why']}" for k, v in g.items() if isinstance(v, dict)))
    for ep in cands:
        rec_p = RECIPES / f"ep{ep2(ep)}.json"
        rec = jload(rec_p) if rec_p.is_file() else None
        print(f"ep{ep2(ep)} {clip_state(ep, rec)}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["candidates", "recipe", "recipes", "cfgcheck", "gates", "plan", "state", "dry", "run", "post", "verdict", "status"])
    ap.add_argument("--ep", type=int)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "candidates":
        c, rows = candidates()
        print(json.dumps({"candidates": [ep2(e) for e in c], "rows": rows}, indent=1))
        return 0
    if a.cmd == "recipe":
        print(json.dumps(recipe(a.ep, write=a.write), indent=1))
        return 0
    if a.cmd == "recipes":
        c, _ = candidates()
        for ep in c:
            r = recipe(ep, write=a.write)
            print(f"ep{ep2(ep)}: {r['termination_overrides']} | " + " | ".join(r["rule_trace"]))
        return 0
    if a.cmd == "cfgcheck":
        c, _ = candidates()
        eps = [a.ep] if a.ep is not None else c
        recs = {ep: recipe(ep, write=False) for ep in eps}
        res = cfgcheck(eps, recs)
        print(json.dumps(res, indent=1, default=str))
        return 0 if all(v.get("ok") for v in res.values()) else 1
    if a.cmd == "gates":
        print(json.dumps(gates(), indent=1))
        return 0
    if a.cmd == "plan":
        print(T().plan(a.ep, ""))
        return 0
    if a.cmd in ("state", "status"):
        return status()
    if a.cmd == "dry":
        return dry()
    if a.cmd == "run":
        return run_loop()
    if a.cmd == "post":
        return post(a.ep)
    if a.cmd == "verdict":
        return verdict(write=not a.no_write)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
