#!/usr/bin/env python3
"""t2cons_r6_consensus.py: steps R4 and R6 of fc_runs/submissions/t2_consensus_r6_20261031 (PREREG_R6 5): rollout lists,
own-input gate, consensus-B scores, the phase-S selection (final_r6.json; SELECTION_S6.sha256 written last) and the
phase-F decision with the S5 / S6 trial composites, R6-CLOSEST and the replace verdict (SELECTION_R6.sha256 written last).

Round 5's scorer machinery (scripts/t2cons_r5_consensus.py e79a2450...) reduced to the round-6 bookkeeping (PREREG_R6 5.7):
  K1  pinned code: round 2's scorer scripts/t2cons_r2_consensus.py (ddee0b8b...) is compiled from its verified bytes; its
      module code compiles scripts/t2cons_a1_gate.py (a29c997b...) from verified bytes, whose module code loads
      scripts/t2cons_consensus.py (e1b943b1..., re-checked here). The pinned round-1 functions rollout_on_grid,
      metrics_vs, load_kit, load_refs, grid, stats, floor_count, integrity / integrity_diff, check_pins, hash_list,
      git_head and the gate's auc_seed are called unchanged; the per-run gate flags are round 2's run_flags and the
      score columns round 2's SCORE_COLS (as rounds 2-5).
  K2  the round-6 rules are new code (PREREG_R6 5.4-5.6): select_s (phase S, at most one finalist per episode), decide_f
      (phase F, paired fresh runs, bar = max(m_P_frozen, m_P_fresh) + 0.01), trial_rows / closest_trial (S5 / S6
      composites, R6-CLOSEST) and replace_verdict. The selftest checks them on synthetic inputs, edge cases and against
      independent re-derivations on random inputs.
  K3  the path guard is round 1's (clean, LOCAL_NOT_FOR_UPLOAD, packed_, tier_1_multiview_caption, track_2/baseline,
      /data/human_motion_data/, /data/v2d_challenge_data_public) + round 2's (readme, train_rl.log) + t2_zx07_20261028,
      best_weights, seed_best, remote_reps, report_r, composites, per_episode_zx (PREREG_R6 5.7).
  K4  evaluator (own-input) values are each run's run.json metrics (full precision); they must equal the driver's
      own_input_evals_{s6,f6}.csv row (|diff| <= 1e-6; SP-SR / MP-SR exactly).
  K5  the incumbents (PREREG_R6 4) come from inputs_r6/incumbents_r6.json (pinned 44d62d8c...); m_P_frozen must equal
      the float64 mean of the frozen variant-B consensus AUCs, the PREREG_R6 4 table (6 decimals) and the frozen
      scores CSV of the round that selected the incumbent (pinned in PREREG_R6 11).

Commands
  selftest           synthetic checks of 5.4 / 5.5 / 5.6 (ties, floors, flagged runs, incomplete trials, exact
                     thresholds), the guard and the incumbent table. No round-6 score. -> selftest_r6/selftest_r6.json
  check  --phase S|F dry: pin chain, jobs, runs; no metric. Writes once inputs_r6/ROLLOUTS_{S6,F6}.sha256 (5.2) and
                     dryrun/r6_consensus_check_<phase>_<stamp>.txt
  gate   --phase S|F 5.3: every run vs its OWN input (vertex seeds 0..9) -> scores_{s6,f6}/own_input_gate_{s6,f6}.{csv,json}
  score  --phase S|F 5.2: variant-B consensus vs round 1's frozen refs/B -> scores_{s6,f6}/consensus_scores_{s6,f6}.csv,
                     consensus_job_stats_{s6,f6}.json
  select --phase S   5.4 -> selection_s6.{json,txt,md}, selection_s6_candidates.csv, final_r6.json; SELECTION_S6.sha256 LAST
  select --phase F   5.5 + 5.6 -> selection_r6.{json,txt,md}, selection_r6_episodes.csv, summary_r6.{json,md},
                     summary_r6_trials.csv; SELECTION_R6.sha256 LAST
Noisy-only: every data path goes through the extended guard; no clean file, no packer, no Tier-1 value. Variant B only.
New files only (mode 444); nothing is signalled; CPU only.
Run: conda_envs/v2dkit/bin/python -I -B scripts/t2cons_r6_consensus.py <cmd> [--phase S|F]
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
ROOT = ME / "fc_runs/submissions/t2_consensus_r6_20261031"
R1 = ME / "fc_runs/submissions/t2_consensus_20261024"
P2_FILE, P2_SHA = ME / "scripts/t2cons_r2_consensus.py", "ddee0b8b6fa25029275b21a3dc82d15c36f4de71c0a61f2a1046869b01ec13aa"
GATE_SHA = "a29c997b7271e9299db109e97af5210f528484719210bb58685523b0ae0eb16b"
CONS_SHA = "e1b943b10fd1db13ab6f78e5bde7b3c6e6c48f7daeee6230c32c463d14ac41b4"
PREREG, PREREG_SHA = ROOT / "PREREG_R6.md", "0b0fbf9a2b581913ae0da7d36ce76aa785e706c132dc496054f2323a80e487e1"
CHAIN_DOCS = (  # (file, its sha256 file, hard-coded sha256); each must hold, and every pin line in it must hold
    ("PREREG_R6_PINS.md", "PREREG_R6_PINS.sha256", "a9c36b1f0d2a7de390e1edad36e2a4946515d1edcb0b41d9055c9d77c0aa5d66"),
)  # fmt: skip
DRYRUN, DRYRUN_SHA, DRYRUN_LIST = ROOT / "DRYRUN_R6.md", "d1e21ebfcac2d49065ff8082295444e7c5e0fa5ea44697a0736f2c2a22f88934", \
    ROOT / "DRYRUN_R6.sha256"  # fmt: skip
POOL, POOL_SHA = ROOT / "inputs_r6/pool_r6.json", "c2aeb79af68cda31d8daa9f324bc26189c721150ec781f62a775782403082de6"
INCS, INCS_SHA = ROOT / "inputs_r6/incumbents_r6.json", "44d62d8cbd5b2d4247307b301902fb47dc42032c4227dcbefb970056d9d07b0b"
SNAP, TRACKS = ROOT / "inputs_r6/SNAPSHOT_R6.sha256", ROOT / "inputs_r6/TRACKS_R6.sha256"
R1_TRACKS = R1 / "inputs/TRACKS.sha256"  # round-1 track records (the raw input reference = the released Tier-2 parquet)
PINS_B, PINS_B_SHA = ROOT / "PREREG_R6_PINS_b.md", ROOT / "PREREG_R6_PINS_b.sha256"
BASELINE, BASELINE_SHA = ROOT / "integrity/baseline.json", "ef9b5035c09b8ad40713d9fb08470eeccb2abe8845d4d1cce3037c49da1cd622"
EXPORTS = ROOT / "exports"
FINAL, SEL_S6, SEL_R6 = ROOT / "final_r6.json", ROOT / "SELECTION_S6.sha256", ROOT / "SELECTION_R6.sha256"
SELFTEST = ROOT / "selftest_r6/selftest_r6.json"
R5_CLOSEST = "R5-CLOSEST (round 5 SELECTION_R5.sha256 58e8b54b..., packed file sha256 e8a03a4d...)"
VARIANT = "B"
MARGIN, FLOOR_S, FLOOR_F, MIN_FRESH, DELTA_B_MIN = 0.01, 2, 8, 8, 0.001  # PREREG_R6 5.4 / 5.5 / 5.6
NTRIALS = 10
EXTRA_FORBIDDEN_R6 = ("t2_zx07_20261028", "best_weights", "seed_best", "remote_reps", "report_r", "composites",
                      "per_episode_zx")  # PREREG_R6 5.7 (round 2's readme / train_rl.log come with its module)  # fmt: skip
PH = {
    "S": {"tag": "s6", "name": "screen", "runs": ROOT / "runs_s", "nruns": 3, "manifest": ROOT / "manifest_s6.json",
          "evals": ROOT / "own_input_evals_s6.csv", "hb": ROOT / "eval_r6_heartbeat_s6.json",
          "rollouts": ROOT / "inputs_r6/ROLLOUTS_S6.sha256", "sc": ROOT / "scores_s6", "lock": SEL_S6},
    "F": {"tag": "f6", "name": "final", "runs": ROOT / "runs_f", "nruns": 10, "manifest": ROOT / "manifest_f6.json",
          "evals": ROOT / "own_input_evals_f6.csv", "hb": ROOT / "eval_r6_heartbeat_f6.json",
          "rollouts": ROOT / "inputs_r6/ROLLOUTS_F6.sha256", "sc": ROOT / "scores_f6", "lock": SEL_R6},
}  # fmt: skip
INC_TABLE = {  # PREREG_R6 4: ep -> (incumbent, m_P_frozen to 6 decimals, frozen runs, input)
    0: ("ep00_zxb_k3fix_60000", 0.709383, 10, "k3fix"), 1: ("ep01_k3fix_final44922", 0.171319, 5, "k3fix"),
    2: ("ep02_k3fix_final34093", 0.868138, 5, "k3fix"), 3: ("ep03_k3fix_final27068", 0.921527, 5, "k3fix"),
    4: ("ep04_raw_final44922", 0.120283, 5, "raw"), 5: ("ep05_zx_k3fix_88000", 0.578153, 5, "k3fix"),
    6: ("ep06_k3fix_final28420", 0.797276, 5, "k3fix"), 7: ("ep07_k3fix_final18529", 0.920452, 5, "k3fix"),
    8: ("ep08_k3easy_final17496", 0.596363, 5, "k3"), 9: ("ep09_raw_final39328", 0.760460, 5, "raw"),
    10: ("ep10_raw_final34267", 0.412469, 5, "raw"), 11: ("ep11_k3fix_final29175", 0.783208, 5, "k3fix"),
    12: ("ep12_k3fix_final23178", 0.543911, 5, "k3fix"), 13: ("ep13_k3fix_final39254", 0.665483, 5, "k3fix"),
    14: ("ep14_k3fix_final44922", 0.330614, 5, "k3fix"), 15: ("ep15_raw_final44922", 0.729895, 5, "raw"),
    16: ("ep16_zxb_g2_34000", 0.937434, 10, "g2"), 17: ("ep17_raw_final31807", 0.647528, 5, "raw"),
    18: ("ep18_k3fix_final18911", 0.419129, 5, "k3fix"), 19: ("ep19_zxb_g2hard_248000", 0.565752, 10, "g2"),
    20: ("ep20_zxb_k3fix_30000", 0.851695, 10, "k3fix"), 21: ("ep21_k3fix_final44922", 0.444469, 5, "k3fix"),
    22: ("ep22_zxb_k3fix_78000", 0.756067, 10, "k3fix"), 23: ("ep23_raw_final20023", 0.902867, 5, "raw"),
    24: ("ep24_raw_final44922", 0.895137, 5, "raw"), 25: ("ep25_zxb_k3fix_48000", 0.748108, 10, "k3fix"),
    26: ("ep26_zx_k3fix_92000", 0.731484, 5, "k3fix"), 27: ("ep27_k3fix_final20863", 0.897854, 5, "k3fix"),
    28: ("ep28_zx_g2_102000", 0.154162, 10, "g2"), 29: ("ep29_k3fix_final17358", 0.505675, 5, "k3fix"),
}  # fmt: skip
RULE_S = ("PREREG_R6 5.4: per episode e, a screen candidate c is S-eligible iff its export is ok, it has 3 ok official "
          "runs, none flagged by the own-input gate (5.3), and own-input SP-SR = MP-SR = 1 in >= 2 of its 3 runs; "
          "cons_S(c) = float64 mean of its 3 variant-B consensus AUCs; finalist f_e = the S-eligible c with the highest "
          "cons_S(c) among those with cons_S(c) >= m_P_frozen(e) + 0.01 (exact floats); ties: lower population variance "
          "of its 3 consensus AUCs, lower mean consensus MPPE, job name; otherwise no finalist.")  # fmt: skip
RULE_F = ("PREREG_R6 5.5: m_F(f) = float64 mean of f's 10 phase-F consensus AUCs; m_P_fresh = float64 mean of the "
          "incumbent's phase-F consensus AUCs over its ok, non-flagged runs (>= 8 required, else +inf); f replaces P iff "
          "(a) f has 10 ok, non-flagged phase-F runs, (b) own-input SP-SR = MP-SR = 1 in >= 8 of them and (c) m_F(f) >= "
          "max(m_P_frozen, m_P_fresh) + 0.01 (exact floats); otherwise P stays. PREREG_R6 5.6: trial t is complete for "
          "a set iff every episode's job has an ok, non-flagged run t; R6-CLOSEST = the complete S6 trial whose 30-episode "
          "mean own-input AUC is closest to the mean over the complete S6 trials (ties: smaller distance of its "
          "consensus-B mean to the mean of that quantity, then lower t); R6-CLOSEST replaces R5-CLOSEST iff (i) >= 1 "
          "episode changed, (ii) Delta_B = mean over the trials complete for both sets of (S6 - S5 30-episode mean "
          "consensus-B AUC) >= +0.001 and (iii) S6's mean own-input SP-SR and MP-SR over those trials are each >= S5's.")  # fmt: skip


# ================================================================================================ pinned code
def _load(name: str, path: Path, want: str):
    src = path.read_bytes()
    got = hashlib.sha256(src).hexdigest()
    if got != want:
        raise SystemExit(f"{path} sha256 {got} != pinned {want}")
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)  # noqa: S102 (verified pinned bytes)
    return mod


P2 = _load("t2cons_r2_consensus_pinned_r6", P2_FILE, P2_SHA)  # loads t2cons_a1_gate (a29c997b) -> t2cons_consensus
G, C = P2.G, P2.C
if hashlib.sha256(Path(G.__file__).read_bytes()).hexdigest() != GATE_SHA or \
        hashlib.sha256(Path(C.__file__).read_bytes()).hexdigest() != CONS_SHA:  # fmt: skip
    raise SystemExit("pinned gate / consensus modules differ from a29c997b / e1b943b1")
if (P2.MARGIN, P2.VARIANT) != (0.01, "B") or tuple(G.SEEDS) != tuple(range(10)) or G.BAND != 0.02:
    raise SystemExit("round-2 scorer / gate constants differ from PREREG_R6 5.3 / 5.4")
C.FORBIDDEN = tuple(C.FORBIDDEN) + EXTRA_FORBIDDEN_R6  # PREREG_R6 5.7 (rebinding only; round 2 added readme, train_rl.log)
run_flags, SCORE_COLS, pin_lines = P2.run_flags, P2.SCORE_COLS, P2.pin_lines
MINE = Path(__file__).resolve()


# ================================================================================================ helpers
def sha(p) -> str:
    return C.sha256_file(p)


def raw_sha(p) -> str:
    C.guard(p)
    h = hashlib.sha256()
    with open(p, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pin_hash(p: Path) -> str:
    try:
        return raw_sha(p) if Path(p).is_file() else "missing"
    except PermissionError:
        return "guarded"


def first_token(p: Path) -> str:
    return (C.guard(p).read_text().split() or [""])[0] if Path(p).is_file() else ""


def write_new(p: Path, text: str, mode: int | None = 0o444) -> None:
    """New files only (refuses to overwrite), through a temporary name; chmod 444."""
    p = C.guard(p)
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp_r6c")
    if tmp.exists():
        raise SystemExit(f"stale temporary {tmp}")
    tmp.write_text(text)
    os.replace(tmp, p)
    if mode is not None:
        os.chmod(p, mode)


def pin_path(rel: str) -> Path:
    """PREREG_R6 11 path rules: '/...' absolute; 'scripts/...' and 'tmp_analysis/...' relative to ME; else ROOT."""
    if rel.startswith("/"):
        return Path(rel)
    return ME / rel if rel.startswith(("scripts/", "tmp_analysis/")) else ROOT / rel


def rel_name(p: Path) -> str:
    """The SELECTION_* list spelling of a path (the driver's _pin_path inverse)."""
    p = Path(p)
    if str(p).startswith(str(ROOT) + "/"):
        return str(p.relative_to(ROOT))
    if str(p).startswith(str(ME / "scripts") + "/"):
        return str(p.relative_to(ME))
    return str(p)


def fmt(x, spec=".4f") -> str:
    return "" if x is None else format(float(x), spec)


def ep_key(e: int) -> str:
    return f"{e:02d}"


def files_of(ph: str) -> dict[str, Path]:
    sc, tag = PH[ph]["sc"], PH[ph]["tag"]
    return {"gate_csv": sc / f"own_input_gate_{tag}.csv", "gate_json": sc / f"own_input_gate_{tag}.json",
            "scores_csv": sc / f"consensus_scores_{tag}.csv", "jstats": sc / f"consensus_job_stats_{tag}.json"}  # fmt: skip


def fin(x) -> float | None:
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x


# ================================================================================================ rules (new, PREREG_R6 5.4-5.6)
def select_s(m_P: float, cands: list[str], info: dict) -> tuple[str | None, list[dict]]:
    """PREREG_R6 5.4 for one episode. info[c] = {"export_ok": bool, "own": {k: {"sp_sr", "mp_sr"}} (ok official runs),
    "flagged": bool (any run flagged by 5.3 = validation_failed), "cons": {"runs": [...], "auc": [...], "mppe": [...]}
    or None}. Returns (finalist or None, one row per candidate)."""
    bar = m_P + MARGIN
    rows = []
    for c in sorted(cands):
        x = info[c]
        own, cons = x.get("own") or {}, x.get("cons")
        why = []
        if not x.get("export_ok"):
            why.append("export not ok")
        if sorted(own) != [1, 2, 3]:
            why.append(f"{len(own)} of 3 official runs ok")
        if x.get("flagged"):
            why.append("validation_failed (a run flagged by the own-input gate, 5.3)")
        if cons is None or list(cons.get("runs") or []) != [1, 2, 3] or len(cons.get("auc") or []) != 3:
            why.append("not 3 consensus-scored runs")
        fl = C.floor_count(own)
        if fl < FLOOR_S:
            why.append(f"floor {fl}/3 < {FLOOR_S}")
        row = {"job": c, "export_ok": bool(x.get("export_ok")), "n_own_ok": len(own), "own_runs": sorted(own),
               "flagged": bool(x.get("flagged")), "floor_count": fl, "m_P_frozen": m_P, "bar": bar}  # fmt: skip
        if cons is not None and cons.get("auc"):
            st = C.stats(cons["auc"])
            row.update(cons_runs=list(cons["runs"]), cons_auc=[float(v) for v in cons["auc"]], cons_mean=st["mean"],
                       cons_var=st["var"], cons_mppe_mean=C.stats(cons["mppe"])["mean"], delta_vs_m_P=st["mean"] - m_P)  # fmt: skip
        eligible = not why
        if eligible and not (row["cons_mean"] >= bar):
            why.append("cons_S < m_P_frozen + 0.01")
        row.update(eligible=eligible, qualifies=not why, why_not=why)
        rows.append(row)
    q = sorted((r for r in rows if r["qualifies"]), key=lambda r: (-r["cons_mean"], r["cons_var"], r["cons_mppe_mean"], r["job"]))
    return (q[0]["job"] if q else None), rows


def decide_f(m_P_frozen: float, inc_runs: dict, fin_job: str | None, fin_runs: dict | None) -> dict:
    """PREREG_R6 5.5 for one episode. *_runs[k] = {"ok": official run ok AND consensus-scored, "flagged": bool,
    "cons_auc": float, "own_sp": float, "own_mp": float}, k = 1..10."""
    good_p = [k for k in sorted(inc_runs) if inc_runs[k]["ok"] and not inc_runs[k]["flagged"]]
    m_P_fresh = C.stats([inc_runs[k]["cons_auc"] for k in good_p])["mean"] if len(good_p) >= MIN_FRESH else math.inf
    bar = max(m_P_frozen, m_P_fresh) + MARGIN
    out = {"m_P_frozen": m_P_frozen, "incumbent_good_runs": good_p, "m_P_fresh": fin(m_P_fresh),
           "m_P_fresh_infinite": not math.isfinite(m_P_fresh), "bar": fin(bar), "finalist": fin_job, "changed": False,
           "why_not": []}  # fmt: skip
    if fin_job is None:
        out["why_not"] = ["no finalist (5.4)"]
        return out
    fin_runs = fin_runs or {}
    good = [k for k in sorted(fin_runs) if fin_runs[k]["ok"] and not fin_runs[k]["flagged"]]
    why = []
    if good != list(range(1, NTRIALS + 1)):
        why.append(f"(a) {len(good)} of 10 ok, non-flagged phase-F runs")
    fl = sum(1 for k in good if fin_runs[k]["own_sp"] == 1.0 and fin_runs[k]["own_mp"] == 1.0)
    if fl < FLOOR_F:
        why.append(f"(b) own-input SP-SR = MP-SR = 1 in {fl} < {FLOOR_F} runs")
    m_F = C.stats([fin_runs[k]["cons_auc"] for k in good])["mean"] if good else None
    if m_F is None or not (m_F >= bar):
        why.append("(c) m_F < max(m_P_frozen, m_P_fresh) + 0.01")
    out.update(finalist_good_runs=good, floor_count=fl, m_F=m_F, delta_vs_bar=None if m_F is None or not math.isfinite(bar)
               else m_F - bar, changed=not why, why_not=why)  # fmt: skip
    return out


def trial_rows(comp: dict[int, str], runinfo: dict[str, dict[int, dict]], ntrials: int = NTRIALS) -> list[dict]:
    """PREREG_R6 5.6: one row per trial t; complete iff every episode's job has an ok, non-flagged run t; for a complete
    trial the 30-episode means of the evaluator (own-input) and consensus-B values."""
    out = []
    for t in range(1, ntrials + 1):
        recs, missing = [], []
        for e in sorted(comp):
            r = (runinfo.get(comp[e]) or {}).get(t)
            if r is None or not r["ok"] or r["flagged"]:
                missing.append(e)
            else:
                recs.append(r)
        row = {"trial": t, "complete": not missing, "missing_episodes": missing}
        if not missing:
            for key in ("own_auc", "own_sp", "own_mp", "own_mppe", "cons_auc", "cons_sp", "cons_mp", "cons_mppe"):
                row[key] = float(np.mean(np.asarray([r[key] for r in recs], dtype=np.float64)))
        out.append(row)
    return out


def closest_trial(rows: list[dict]) -> int | None:
    """R6-CLOSEST over the complete trials: own-input AUC mean closest to its mean; ties: consensus-B distance, lower t."""
    cr = [r for r in rows if r["complete"]]
    if not cr:
        return None
    mu = C.stats([r["own_auc"] for r in cr])["mean"]
    mub = C.stats([r["cons_auc"] for r in cr])["mean"]
    return min(cr, key=lambda r: (abs(r["own_auc"] - mu), abs(r["cons_auc"] - mub), r["trial"]))["trial"]


def replace_verdict(n_changed: int, s5: list[dict], s6: list[dict]) -> dict:
    c5 = {r["trial"]: r for r in s5 if r["complete"]}
    c6 = {r["trial"]: r for r in s6 if r["complete"]}
    shared = sorted(set(c5) & set(c6))
    out = {"n_changed": n_changed, "shared_trials": shared, "delta_B": None, "s6_own_sp": None, "s5_own_sp": None,
           "s6_own_mp": None, "s5_own_mp": None}  # fmt: skip
    if shared:
        out["delta_B"] = float(np.mean(np.asarray([c6[t]["cons_auc"] - c5[t]["cons_auc"] for t in shared], dtype=np.float64)))
        for k in ("own_sp", "own_mp"):
            out[f"s6_{k}"] = float(np.mean(np.asarray([c6[t][k] for t in shared], dtype=np.float64)))
            out[f"s5_{k}"] = float(np.mean(np.asarray([c5[t][k] for t in shared], dtype=np.float64)))
        out["s6_own_auc_report_only"] = float(np.mean(np.asarray([c6[t]["own_auc"] for t in shared], dtype=np.float64)))
        out["s5_own_auc_report_only"] = float(np.mean(np.asarray([c5[t]["own_auc"] for t in shared], dtype=np.float64)))
    out["i_changed"] = n_changed >= 1
    out["ii_delta_B"] = out["delta_B"] is not None and out["delta_B"] >= DELTA_B_MIN
    out["iii_sp_mp"] = bool(shared) and out["s6_own_sp"] >= out["s5_own_sp"] and out["s6_own_mp"] >= out["s5_own_mp"]
    out["replace"] = bool(out["i_changed"] and out["ii_delta_B"] and out["iii_sp_mp"])
    return out


# ================================================================================================ pin chain
def driver_running() -> list[str]:
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if "t2cons_r6_eval" in cmd and "t2cons_r6_consensus" not in cmd:
            out.append(f"pid {d.name}: {cmd[:160]}")
    return out


def hash_lines(listing: Path) -> list[tuple[str, str]]:
    out = []
    for line in C.guard(listing).read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and len(parts[0]) == 64:
            out.append((parts[1].strip(), parts[0]))
    return out


def chain(ph: str, stage: str) -> tuple[list[str], list[str]]:
    """stage: check | gate | score | select. PINS_b (pinning this script + its selftest) is required from gate on."""
    fatal, info = [], []
    got, rec = pin_hash(PREREG), first_token(ROOT / "PREREG_R6.sha256")
    if not got == rec == PREREG_SHA:
        return [f"PREREG_R6.md sha256 {got[:16]} (sha file {rec[:16]}, hard-coded {PREREG_SHA[:16]})"], info
    sec = PREREG.read_text().split("## 11. Pins (sha256)", 1)[-1]
    n = 0
    for rel, h0 in pin_lines(sec):
        n += 1
        h = pin_hash(pin_path(rel))
        if h != h0:
            fatal.append(f"PREREG_R6 pin {rel}: {h[:16]} != {h0[:16]}")
    for line in sec.splitlines():
        f = line.split()
        if len(f) == 3 and f[0] == "commit":
            n += 1
            head = C.git_head(ME / f[1])
            if head != f[2]:
                fatal.append(f"PREREG_R6 commit {f[1]}: HEAD {head[:12]} != {f[2][:12]}")
    info.append(f"PREREG_R6.md {got[:16]} == PREREG_R6.sha256 == hard-coded; {n} section-11 pins / commits hold")
    for name, shaf, want in CHAIN_DOCS:
        p = ROOT / name
        got, rec = pin_hash(p), first_token(ROOT / shaf)
        if not got == rec == want:
            fatal.append(f"{name} sha256 {got[:16]} (sha file {rec[:16]}, hard-coded {want[:16]})")
            continue
        k = 0
        for rel, h0 in pin_lines(p.read_text()):
            k += 1
            if pin_hash(pin_path(rel)) != h0:
                fatal.append(f"{name} pin {rel} changed")
        info.append(f"{name} {got[:16]} == its sha file == hard-coded; {k} pins hold")
    if pin_hash(DRYRUN) != DRYRUN_SHA:
        fatal.append("DRYRUN_R6.md != d1e21ebf...")
    k = 0
    for rel, h0 in hash_lines(DRYRUN_LIST):
        k += 1
        if pin_hash(pin_path(rel)) != h0:
            fatal.append(f"DRYRUN_R6.sha256: {rel} changed")
    info.append(f"DRYRUN_R6.md == d1e21ebf...; DRYRUN_R6.sha256: {k} lines hold")
    for listing in (SNAP, TRACKS):
        k = 0
        for rel, h0 in hash_lines(listing):
            k += 1
            h = pin_hash(ROOT / rel)  # ROOT / '/abs' == '/abs' (the driver's rule)
            if h != h0:
                fatal.append(f"{listing.name}: {rel} {h[:16]} != {h0[:16]}")
        info.append(f"{listing.name}: {k} lines hold")
    for p, want in ((POOL, POOL_SHA), (INCS, INCS_SHA)):
        if pin_hash(p) != want:
            fatal.append(f"{p.name} != {want[:16]}")
    f2, i2 = C.check_pins(need_pins=False)  # round-1 PREREG == sha, its section-11 pins + commits, SNAPSHOT, KIT
    fatal += [f"round-1 chain: {x}" for x in f2]
    info.append(f"round-1 chain (pinned t2cons_consensus.check_pins): {len(f2)} fatal")
    P = PH[ph]
    hb = C.read_json(P["hb"]) if P["hb"].is_file() else {}
    ev, ex = hb.get("evals") or {}, hb.get("exports") or {}
    if hb.get("state") != "done" or ev.get("pending") != 0 or ex.get("pending") != 0 or hb.get("exports_only") is not False \
            or str(hb.get("runs_base")) != str(P["runs"]) or hb.get("phase") != P["name"] or hb.get("halted"):  # fmt: skip
        fatal.append(f"phase {ph} evaluations not finished: heartbeat state {hb.get('state')} evals {ev} exports {ex}")
    run = driver_running()
    if run or (ROOT / "driver.pid").exists():
        fatal.append(f"an eval driver is running or driver.pid exists: {run}")
    info.append(f"phase {ph}: heartbeat state {hb.get('state')}, exports {ex}, evals {ev}; no driver / driver.pid: "
                f"{not run and not (ROOT / 'driver.pid').exists()}")  # fmt: skip
    if pin_hash(BASELINE) != BASELINE_SHA:
        fatal.append("integrity/baseline.json is not the byte copy of round 1's baseline (ef9b5035...)")
    else:
        diff = C.integrity_diff(json.loads(BASELINE.read_text()), C.integrity())
        if diff:
            fatal.append(f"kit / stock trees differ from integrity/baseline.json: {diff[:5]}")
        info.append(f"kit + stock trees == integrity/baseline.json (ef9b5035): {not diff}")
    if ph == "F":
        if not SEL_S6.is_file():
            fatal.append("SELECTION_S6.sha256 missing (phase F needs the frozen phase-S selection)")
        else:
            listed = False
            for rel, h0 in hash_lines(SEL_S6):
                listed |= pin_path(rel) == FINAL
                if pin_hash(pin_path(rel)) != h0:
                    fatal.append(f"SELECTION_S6.sha256: {rel} changed")
            if not listed:
                fatal.append("SELECTION_S6.sha256 does not list final_r6.json")
            info.append(f"SELECTION_S6.sha256 {raw_sha(SEL_S6)[:16]} holds; lists final_r6.json: {listed}")
    mine = raw_sha(MINE)
    info.append(f"scripts/t2cons_r6_consensus.py sha256 {mine}")
    if stage in ("gate", "score", "select"):
        if not PINS_B.is_file():
            fatal.append("PREREG_R6_PINS_b.md missing: this script must be pinned before it scores anything (5.7)")
        else:
            got, rec = raw_sha(PINS_B), first_token(PINS_B_SHA)
            pins = pin_lines(PINS_B.read_text())
            if got != rec:
                fatal.append(f"PREREG_R6_PINS_b.md {got[:16]} != PREREG_R6_PINS_b.sha256 {rec[:16]}")
            for need in ("scripts/t2cons_r6_consensus.py", "selftest_r6/selftest_r6.json"):
                if need not in [r for r, _ in pins]:
                    fatal.append(f"PREREG_R6_PINS_b.md does not pin {need}")
            if ("scripts/t2cons_r6_consensus.py", mine) not in pins:
                fatal.append(f"PREREG_R6_PINS_b.md does not pin this script's sha256 {mine[:16]}")
            for rel, h0 in pins:
                if pin_hash(pin_path(rel)) != h0:
                    fatal.append(f"PREREG_R6_PINS_b pin {rel} changed")
            info.append(f"PREREG_R6_PINS_b.md {got[:16]} == its sha file; {len(pins)} pins hold; pins this script")
            st = C.read_json(SELFTEST) if SELFTEST.is_file() else {}
            if st.get("pass") is not True or st.get("script_sha256") != mine:
                fatal.append("selftest_r6/selftest_r6.json is not a PASS of this script")
        if not P["rollouts"].is_file():
            fatal.append(f"{P['rollouts'].name} missing: run check --phase {ph} first (5.2)")
    if stage in ("score", "select"):
        F = files_of(ph)
        g = C.read_json(F["gate_json"]) if F["gate_json"].is_file() else {}
        if g.get("pass") is not True or g.get("script_sha256") != mine or g.get("csv_sha256") != (
                raw_sha(F["gate_csv"]) if F["gate_csv"].is_file() else None) or g.get("pins_b_sha256") != (
                raw_sha(PINS_B) if PINS_B.is_file() else None):  # fmt: skip
            fatal.append(f"{F['gate_json'].name} is not a PASS of this pinned script")
        else:
            info.append(f"own-input gate (5.3) phase {ph}: PASS; validation_failed {len(g.get('validation_failed_jobs') or [])} "
                        f"jobs; flagged runs {g.get('n_flagged_runs')}")  # fmt: skip
    if stage in ("gate", "score", "select") and P["lock"].exists():
        fatal.append(f"{P['lock'].name} exists: phase {ph} is frozen")
    if stage in ("gate", "score", "select") and ph == "S" and SEL_R6.exists():
        fatal.append("SELECTION_R6.sha256 exists")
    return fatal, info


def require(ph: str, stage: str) -> bool:
    fatal, info = chain(ph, stage)
    for x in info:
        print("  " + x)
    if fatal:
        print("FATAL:\n  " + "\n  ".join(fatal))
        return False
    return True


# ================================================================================================ incumbents, jobs, runs
_CSV_CACHE: dict[str, list[dict]] = {}
JOB_STATS_PINS = {  # PREREG_R6 11 pins: the frozen job stats of the rounds that selected the incumbents
    str(R1 / "scores/consensus_job_stats.json"): "9e255057556ccc23df9c97dafd101219770e8b53cc6ab28519ab893eb86319c8",
    str(ME / "fc_runs/submissions/t2_consensus_r3_20261028/scores_r3/consensus_job_stats_r3.json"):
        "88d8e26a454c9cad657aa4c49bf633132defc1e33785ebbbb2caf790130b5cd9",
    str(ME / "fc_runs/submissions/t2_consensus_r5_20261029/scores_r5/consensus_job_stats_r5.json"):
        "b1e8e4e1aa749e22a2de17fd08d5edde3a19f441707b1694a09c46d65bb2607c",
}  # fmt: skip


def load_incumbents() -> dict[int, dict]:
    """PREREG_R6 4 / K5: the 30 round-5 incumbents with their frozen variant-B consensus runs and m_P_frozen."""
    data = C.read_json(INCS)
    incs = {int(k): v for k, v in data["incumbents"].items()}
    if sorted(incs) != list(range(30)):
        raise SystemExit("incumbents_r6.json does not hold 30 episodes")
    for e, x in incs.items():
        job, m6, n, kind = INC_TABLE[e]
        fr = sorted(x["frozen_runs"], key=lambda r: int(r["run"]))
        vals = [float(r["cons_auc"]) for r in fr]
        m = C.stats(vals)["mean"]  # the pinned float64 mean (np.mean), as rounds 1-5 computed m_P
        if x["job"] != job or int(x["episode"]) != e or x["input"] != kind or len(vals) != n or \
                [int(r["run"]) for r in fr] != list(range(1, n + 1)):  # fmt: skip
            raise SystemExit(f"ep{e:02d}: incumbents_r6.json row != PREREG_R6 4 table ({x['job']}, {x['input']}, {len(vals)})")
        if abs(m - x["m_P_frozen"]) > 1e-12 or abs(m - m6) > 5e-7:
            raise SystemExit(f"ep{e:02d}: m_P_frozen field {x['m_P_frozen']!r} / table {m6} != mean {m!r}")
        f = x["frozen_scores_csv"]
        if f not in _CSV_CACHE:
            if sha(f) != x["frozen_scores_csv_sha256"]:
                raise SystemExit(f"{f} sha256 != incumbents_r6.json record")
            _CSV_CACHE[f] = C.read_csv(f)
        rows = sorted([r for r in _CSV_CACHE[f] if r["job"] == job and r["variant"] == VARIANT and r["status"] == "ok"],
                      key=lambda r: int(r["run"]))  # fmt: skip
        if [float(r["auc"]) for r in rows] != vals:
            raise SystemExit(f"ep{e:02d}: frozen runs differ from {f}")
        jsf = f.replace("consensus_scores", "consensus_job_stats")[: -len(".csv")] + ".json"
        if jsf not in _CSV_CACHE:
            if JOB_STATS_PINS.get(jsf) is None or sha(jsf) != JOB_STATS_PINS[jsf]:
                raise SystemExit(f"{jsf}: not the pinned frozen job stats")
            _CSV_CACHE[jsf] = C.read_json(jsf)["jobs"]
        jsx = _CSV_CACHE[jsf][job]
        jsx = jsx.get(VARIANT, jsx)
        if [float(v) for v in jsx["auc"]] != vals or float(jsx["auc_stats"]["mean"]) != m:
            raise SystemExit(f"ep{e:02d}: frozen job stats {jsf} != the frozen runs / their pinned mean")
        # PREREG_R6 4: m_P_frozen = the float64 mean == the frozen job stats (exact). The resolver's field (sum / len) may
        # differ by 1 ulp; the decision uses the pinned stats() value; the field is kept for the record.
        x["m_P_frozen_field_resolver"] = x["m_P_frozen"]
        x["m_P_frozen"] = m
    return incs


def input_ref(kind: str, e: int, trk: dict, r1trk: dict) -> tuple[Path, Path, str | None, str]:
    ref = C.INPUT_REF[kind]
    f = C.ep_file(ref, e)
    rec = (r1trk if kind == "raw" else trk).get(str(f))
    return ref, f, rec, (sha(f) if f.is_file() else "missing")


def load_jobs(ph: str) -> dict[str, dict]:
    """Phase S: the 250 screen rows of manifest_s6.json x pool_r6.json x exports/<job>/export.json. Phase F: the 30
    incumbents (earlier exports, incumbents_r6.json) + the finalists of final_r6.json (their phase-S exports)."""
    P = PH[ph]
    man = C.read_json(P["manifest"])
    if man.get("prereg_r6_sha256") != PREREG_SHA or man.get("phase") != P["name"] or man.get("pool_r6_sha256") != POOL_SHA \
            or man.get("incumbents_r6_sha256") != INCS_SHA:  # fmt: skip
        raise SystemExit(f"{P['manifest'].name} was not built under PREREG_R6 / pool_r6 / incumbents_r6")
    pool = {r["job"]: r for r in C.read_json(POOL)["rows"]}
    incs = load_incumbents()
    inc_by = {x["job"]: x for x in incs.values()}
    trk, r1trk = C.hash_list(TRACKS), C.hash_list(R1_TRACKS)
    fin_by = {}
    if ph == "F":
        if man.get("final_r6_sha256") != raw_sha(FINAL) or man.get("selection_s6_lock_sha256") != raw_sha(SEL_S6):
            raise SystemExit("manifest_f6.json was not built from the frozen final_r6.json / SELECTION_S6.sha256")
        fin_by = {x["job"]: x for x in C.read_json(FINAL)["finalists"]}
    jobs = {}
    for r in man["rows"]:
        job, e, role = r["job"], int(r["episode"]), r.get("role")
        if ph == "S":
            src = pool.get(job)
            if role != "screen" or src is None or src.get("role") != "screen" or int(src["episode"]) != e:
                raise SystemExit(f"{job}: manifest_s6 row is not a pool screen row")
            kind, tsha, owner = src["input"], src["sha256"], src["owner"]
        elif role == "incumbent":
            src = inc_by.get(job)
            if src is None or int(src["episode"]) != e:
                raise SystemExit(f"{job}: manifest_f6 incumbent row is not an incumbents_r6 row")
            kind, tsha, owner = src["input"], src["training_sha256"], src["owner"]
        elif role == "finalist":
            src = pool.get(job)
            if job not in fin_by or src is None or src.get("role") != "screen" or int(fin_by[job]["episode"]) != e:
                raise SystemExit(f"{job}: manifest_f6 finalist row is not in final_r6.json / the pool")
            kind, tsha, owner = src["input"], src["sha256"], src["owner"]
        else:
            raise SystemExit(f"{job}: unexpected role {role}")
        d = {"job": job, "episode": e, "role": role, "id": role, "input": kind, "owner": owner, "tree": r.get("tree"),
             "iteration": r.get("iteration"), "training_sha256": tsha, "parquet_sha256": r.get("parquet_sha256"),
             "manifest_status": r.get("status"), "export_status": None}  # fmt: skip
        jobs[job] = d
        if r.get("status") != "ok":
            d["export_status"] = f"job not built: {r.get('status')}"
            continue
        if r.get("root_kind") != kind or r.get("training_sha256") != tsha or r.get("parquet_sha256") != src["parquet_sha256"]:
            raise SystemExit(f"{job}: manifest row (input / checkpoint / parquet) != its pool / incumbent record")
        if role == "incumbent":
            rec = C.read_json(src["export_record"])
            want = src["export_sha256"]
            if sha(src["export_record"]) != src["export_record_sha256"] or rec.get("status") != "ok" or \
                    rec.get("stock_verified") is not True or rec.get("sha256") != want or \
                    Path(str(rec.get("file"))) != Path(src["export_file"]) or sha(src["export_file"]) != want:  # fmt: skip
                raise SystemExit(f"{job}: incumbent export record / file != incumbents_r6.json")
            d.update(export_status="ok", export_sha256=want, export_file=src["export_file"])
        else:
            p = EXPORTS / job / "export.json"
            rec = C.read_json(p) if p.is_file() else None
            if rec is None:
                d["export_status"] = "missing"
                continue
            if rec.get("status") != "ok":
                d["export_status"] = str(rec.get("status"))
                continue
            want = rec.get("sha256")
            if rec.get("stock_verified") is not True or rec.get("destination_sha256") != want or \
                    rec.get("source_sha256") != tsha or Path(rec["file"]).parent.parent != EXPORTS / job or \
                    sha(rec["file"]) != want:  # fmt: skip
                raise SystemExit(f"{job}: export.json / file inconsistent")
            if role == "finalist" and fin_by[job].get("export_sha256") != want:
                raise SystemExit(f"{job}: phase-S export != final_r6.json")
            d.update(export_status="ok", export_sha256=want, export_file=rec["file"])
        ref, f, rec_h, h = input_ref(kind, e, trk, r1trk)
        if rec_h != h or (kind != "raw" and src.get("frame_ref_sha256") != h):
            raise SystemExit(f"{job}: input reference {f} sha256 {h[:16]} != its frozen record")
        d.update(input_reference=str(ref), input_reference_file=str(f), input_reference_sha256=h)
    if ph == "S":
        if sorted(jobs) != sorted(j for j, r in pool.items() if r.get("role") == "screen") or len(jobs) != 250:
            raise SystemExit("manifest_s6.json does not hold the 250 screen jobs")
    else:
        inc_jobs = sorted(j for j in jobs if jobs[j]["role"] == "incumbent")
        if inc_jobs != sorted(inc_by) or sorted(j for j in jobs if jobs[j]["role"] == "finalist") != sorted(fin_by):
            raise SystemExit("manifest_f6.json != the 30 incumbents + final_r6.json finalists")
        if any(jobs[j]["export_status"] != "ok" for j in jobs):
            raise SystemExit("a phase-F job is not ok")
    return jobs


def load_evals(ph: str, jobs: dict) -> tuple[dict[str, dict[int, dict]], int, list[str]]:
    """job -> run -> evaluator values (run.json, full precision; == the driver's CSV row) + rollout parquet."""
    P = PH[ph]
    out: dict[str, dict[int, dict]] = {}
    n_not_ok, problems = 0, []
    csv_rows = C.read_csv(P["evals"])
    seen = set()
    for r in csv_rows:
        if r.get("status") != "ok":
            n_not_ok += 1
            continue
        j = jobs.get(r["job"])
        if j is None or j["export_status"] != "ok":
            raise SystemExit(f"{P['evals'].name}: ok row of unknown / non-ok job {r['job']}")
        k, t, e = int(r["run"]), int(r["try"]), j["episode"]
        if int(r["episode"]) != e or not 1 <= k <= P["nruns"]:
            raise SystemExit(f"{r['job']} run {k}: episode / run out of range")
        pq = P["runs"] / r["job"] / f"run{k}" / f"try{t}" / "results" / f"episode_{e:06d}.parquet"
        if Path(r["results_parquet"]) != pq or r.get("export_sha256_full") != j["export_sha256"] or \
                r.get("reference_parquet_sha256") != j["parquet_sha256"] or r.get("variants") != "B" or \
                r.get("input") != j["input"]:  # fmt: skip
            raise SystemExit(f"{r['job']} run {k}: CSV row (parquet / export / reference / input) inconsistent")
        rj = C.read_json(P["runs"] / r["job"] / f"run{k}" / "run.json")
        if rj.get("status") != "ok" or int(rj.get("try")) != t or rj.get("checkpoint_sha256") != j["export_sha256"] or \
                Path(str(rj.get("results_parquet"))) != pq or rj.get("reference_parquet_sha256") != j["parquet_sha256"]:  # fmt: skip
            raise SystemExit(f"{r['job']} run {k}: run.json != CSV row")
        m = {x: float(rj["metrics"][x]) for x in C.METRICS}
        for x in C.METRICS:
            cv = C.fnum(r[x])
            if cv is None or (abs(m[x] - cv) > 1e-6 if x in ("auc", "mppe") else m[x] != cv):
                raise SystemExit(f"{r['job']} run {k}: {x} run.json {m[x]!r} != CSV {r[x]}")
        if k in out.setdefault(r["job"], {}):
            raise SystemExit(f"{r['job']} run {k}: two ok rows")
        seen.add((r["job"], k))
        out[r["job"]][k] = {"run": k, "try": t, "parquet": str(pq), "gpu": r.get("gpu"),
                            "seed_signature": r.get("seed_signature"), **m}  # fmt: skip
    for job, j in jobs.items():  # every ok run.json has its CSV row (the driver's final summary)
        if j["export_status"] != "ok":
            continue
        for k in range(1, P["nruns"] + 1):
            p = P["runs"] / job / f"run{k}" / "run.json"
            if p.is_file() and (C.read_json(p).get("status") == "ok") != ((job, k) in seen):
                problems.append(f"{job} run {k}: run.json status and CSV disagree")
    return out, n_not_ok, problems


def rollout_files(evals: dict) -> dict[str, str]:
    return {evals[j][k]["parquet"]: j for j in sorted(evals) for k in sorted(evals[j])}


def check_rollouts(ph: str, evals: dict) -> list[str]:
    want = C.hash_list(PH[ph]["rollouts"])
    files = rollout_files(evals)
    problems = []
    if sorted(want) != sorted(files):
        problems.append(f"{PH[ph]['rollouts'].name} lists {len(want)} files, the runs have {len(files)}")
    for p in files:
        if want.get(p) is None:
            problems.append(f"{p} not listed")
        elif sha(p) != want[p]:
            problems.append(f"{p} changed")
    return problems


# ================================================================================================ selftest
def cmd_selftest(a) -> int:
    if SELFTEST.exists():
        print("FATAL: selftest_r6/selftest_r6.json exists (new files only)")
        return 2
    rng = np.random.default_rng(20261031)
    res, problems = [], []

    def check(name, ok, detail=""):
        res.append({"test": name, "ok": bool(ok), "detail": str(detail)[:400]})
        if not ok:
            problems.append(f"{name}: {detail}")

    check("pinned modules: round 2's scorer (ddee0b8b) -> gate (a29c997b) -> consensus (e1b943b1); run_flags / SCORE_COLS "
          "/ pin_lines are round 2's objects; gate seeds 0..9, band 0.02",
          P2.run_flags is run_flags and P2.G is G and G.C is C and tuple(G.SEEDS) == tuple(range(10)) and G.BAND == 0.02
          and hashlib.sha256(Path(C.__file__).read_bytes()).hexdigest() == CONS_SHA, "")  # fmt: skip
    check("constants: margin 0.01, floors 2/3 and 8/10, >= 8 fresh incumbent runs, Delta_B >= 0.001, 3 / 10 runs",
          (MARGIN, FLOOR_S, FLOOR_F, MIN_FRESH, DELTA_B_MIN, PH["S"]["nruns"], PH["F"]["nruns"]) == (0.01, 2, 8, 8, 0.001, 3, 10))
    ok3 = {k: {"sp_sr": 1.0, "mp_sr": 1.0} for k in (1, 2, 3)}

    def cand(aucs, own=None, flagged=False, export_ok=True, mppe=1.0, runs=None):
        aucs = list(aucs)
        return {"export_ok": export_ok, "own": ok3 if own is None else own, "flagged": flagged,
                "cons": {"runs": runs or list(range(1, len(aucs) + 1)), "auc": aucs, "mppe": [mppe] * len(aucs)}}

    def runs_with_mean(target: float, n: int):
        xs = [target] * n
        if C.stats(xs)["mean"] == target:
            return xs
        lo = hi = target
        for _ in range(4096):
            lo, hi = float(np.nextafter(lo, -np.inf)), float(np.nextafter(hi, np.inf))
            for x in (lo, hi):
                if C.stats([target] * (n - 1) + [x])["mean"] == target:
                    return [target] * (n - 1) + [x]
        return None

    # ---- 5.4
    mP = 0.5
    edge = runs_with_mean(mP + MARGIN, 3)
    check("precondition: 3 runs with mean exactly m_P + 0.01 exist", edge is not None, repr(edge))
    if edge is not None:
        below = [x - 1e-12 for x in edge]
        f, rows = select_s(mP, ["a"], {"a": cand(edge)})
        check("5.4: cons_S exactly m_P_frozen + 0.01 qualifies (>=, exact floats)", f == "a", rows)
        f, rows = select_s(mP, ["a"], {"a": cand(below)})
        check("5.4: cons_S 1e-12 below m_P + 0.01 does not qualify", f is None and not rows[0]["qualifies"], rows)
    own_fl1 = {1: {"sp_sr": 1.0, "mp_sr": 1.0}, 2: {"sp_sr": 1.0, "mp_sr": 0.0}, 3: {"sp_sr": 0.0, "mp_sr": 1.0}}
    own_fl2 = {1: {"sp_sr": 1.0, "mp_sr": 1.0}, 2: {"sp_sr": 1.0, "mp_sr": 1.0}, 3: {"sp_sr": 0.0, "mp_sr": 0.0}}
    info = {"fl1": cand([0.9] * 3, own=own_fl1), "fl2": cand([0.7] * 3, own=own_fl2), "flag": cand([0.95] * 3, flagged=True),
            "two": cand([0.96] * 2, own={1: ok3[1], 2: ok3[2]}), "noexp": cand([0.97] * 3, export_ok=False),
            "own2": cand([0.98] * 3, own={1: ok3[1], 2: ok3[2]}), "low": cand([0.5] * 3)}  # fmt: skip
    f, rows = select_s(0.5, sorted(info), info)
    check("5.4: floor 1/3, flagged, 2 scored runs, export not ok, 2 official runs are out; floor 2/3 qualifies; the "
          "highest qualifying mean wins", f == "fl2" and {r["job"] for r in rows if r["qualifies"]} == {"fl2"},
          [(r["job"], r["why_not"]) for r in rows])  # fmt: skip
    tie = {"t_a": cand([0.6, 0.6, 0.6], mppe=2.0), "t_b": cand([0.6, 0.6, 0.6], mppe=1.0), "t_c": cand([0.5, 0.7, 0.6], mppe=0.5),
           "t_d": cand([0.6, 0.6, 0.6], mppe=1.0)}  # fmt: skip
    means = {j: C.stats(tie[j]["cons"]["auc"])["mean"] for j in tie}
    f, _ = select_s(0.5, sorted(tie), tie)
    check("5.4 ties: equal means -> lower variance, then lower MPPE, then job name", len(set(means.values())) == 1 and f == "t_b",
          f"{f} {means}")  # fmt: skip
    f, rows = select_s(0.5, [], {})
    check("5.4: no candidate -> no finalist", f is None and rows == [])
    bad = []
    for t in range(800):
        mPt = float(rng.uniform(0.05, 0.95))
        inf_t = {}
        for i in range(int(rng.integers(1, 8))):
            n = int(rng.choice([2, 3, 3, 3, 3]))
            aucs = list(np.clip(mPt + rng.normal(0.01, 0.03) + rng.normal(0, 0.01, n), 0, 1))
            own = {k: {"sp_sr": 1.0 if rng.random() > 0.2 else 0.0, "mp_sr": 1.0 if rng.random() > 0.1 else 0.0}
                   for k in range(1, (3 if rng.random() > 0.05 else 2) + 1)}  # fmt: skip
            inf_t[f"j{t}_{i}"] = {"export_ok": rng.random() > 0.03, "own": own, "flagged": rng.random() < 0.08,
                                  "cons": {"runs": list(range(1, n + 1)), "auc": aucs, "mppe": list(rng.uniform(0.5, 20, n))}}  # fmt: skip
        got, _ = select_s(mPt, list(inf_t), inf_t)
        q = []
        for j, x in inf_t.items():
            if not x["export_ok"] or x["flagged"] or len(x["own"]) != 3 or len(x["cons"]["auc"]) != 3:
                continue
            if sum(1 for r in x["own"].values() if r["sp_sr"] == 1.0 and r["mp_sr"] == 1.0) < 2:
                continue
            arr = np.asarray(x["cons"]["auc"], dtype=np.float64)
            if float(arr.mean()) >= mPt + 0.01:
                q.append((-float(arr.mean()), float(arr.var()), float(np.mean(np.asarray(x["cons"]["mppe"], dtype=np.float64))), j))
        want = sorted(q)[0][3] if q else None
        if got != want:
            bad.append(f"case {t}: {got} != {want}")
    check("5.4: select_s == an independent re-derivation on 800 random episodes", not bad, bad[:3])

    # ---- 5.5
    def fr(aucs, sp=None, mp=None, flagged=(), missing=()):
        out = {}
        for k in range(1, NTRIALS + 1):
            if k in missing:
                continue
            out[k] = {"ok": True, "flagged": k in flagged, "cons_auc": float(aucs[k - 1]),
                      "own_sp": 1.0 if sp is None else sp[k - 1], "own_mp": 1.0 if mp is None else mp[k - 1]}  # fmt: skip
        return out

    inc = fr([0.5] * 10)
    e10 = runs_with_mean(0.51, 10)
    d = decide_f(0.5, inc, "f", fr(e10))
    check("5.5: m_F exactly max(frozen, fresh) + 0.01 replaces (exact floats)", e10 is not None and d["changed"] and
          d["bar"] == 0.51, d)  # fmt: skip
    d = decide_f(0.5, fr([0.55] * 10), "f", fr([0.58] * 10))
    check("5.5: fresh incumbent mean above frozen raises the bar (0.58 < 0.55 + 0.01? no: 0.58 >= 0.56 replaces; "
          "0.555 does not)", d["changed"] and not decide_f(0.5, fr([0.55] * 10), "f", fr([0.555] * 10))["changed"], d)  # fmt: skip
    d = decide_f(0.6, fr([0.5] * 10), "f", fr([0.605] * 10))
    check("5.5: frozen above fresh: the bar is m_P_frozen + 0.01", not d["changed"] and abs(d["bar"] - 0.61) < 1e-15, d)
    d = decide_f(0.5, fr([0.5] * 10, flagged=(1, 2, 3)), "f", fr([0.9] * 10))
    check("5.5: incumbent with 7 good runs -> m_P_fresh = +inf, no change", not d["changed"] and d["m_P_fresh_infinite"], d)
    d = decide_f(0.5, fr([0.5] * 10, flagged=(1, 2)), "f", fr([0.9] * 10))
    check("5.5: incumbent with 8 good runs -> m_P_fresh over them", d["changed"] and d["incumbent_good_runs"] == list(range(3, 11)), d)
    d = decide_f(0.5, inc, "f", fr([0.9] * 10, flagged=(4,)))
    check("5.5: a flagged finalist run -> not qualifying", not d["changed"], d)
    d = decide_f(0.5, inc, "f", fr([0.9] * 10, missing=(10,)))
    check("5.5: 9 finalist runs -> not qualifying", not d["changed"], d)
    d = decide_f(0.5, inc, "f", fr([0.9] * 10, sp=[1.0] * 7 + [0.0] * 3))
    check("5.5: floor 7/10 -> not qualifying", not d["changed"], d)
    d = decide_f(0.5, inc, "f", fr([0.9] * 10, mp=[1.0] * 8 + [0.0] * 2))
    check("5.5: floor 8/10 qualifies", d["changed"], d)
    d = decide_f(0.5, inc, None, None)
    check("5.5: no finalist -> P stays", not d["changed"], d)

    # ---- 5.6
    def ri(own_auc, cons_auc, sp=1.0, mp=1.0, ok=True, flagged=False):
        return {"ok": ok, "flagged": flagged, "own_auc": own_auc, "own_sp": sp, "own_mp": mp, "own_mppe": 1.0,
                "cons_auc": cons_auc, "cons_sp": 1.0, "cons_mp": 1.0, "cons_mppe": 1.0}  # fmt: skip

    comp = {e: f"P{e}" for e in range(30)}
    own_t = [0.50, 0.53, 0.51, 0.56, 0.52, 0.49, 0.55, 0.54, 0.505, 0.515]
    cons_t = [0.61, 0.60, 0.64, 0.62, 0.66, 0.63, 0.65, 0.67, 0.605, 0.615]
    runinfo = {f"P{e}": {t: ri(own_t[t - 1], cons_t[t - 1]) for t in range(1, 11)} for e in range(30)}
    runinfo["P7"][3] = ri(0.5, 0.6, flagged=True)
    del runinfo["P9"][8]
    rows = trial_rows(comp, runinfo)
    check("5.6: a flagged run (trial 3) or a missing run (trial 8) makes the trial incomplete",
          [r["trial"] for r in rows if not r["complete"]] == [3, 8] and rows[2]["missing_episodes"] == [7], rows[2])  # fmt: skip
    t_star = closest_trial(rows)
    cr = [r for r in rows if r["complete"]]
    mu = float(np.mean([r["own_auc"] for r in cr]))
    want = min(cr, key=lambda r: (abs(r["own_auc"] - mu), r["trial"]))["trial"]
    check("5.6: R6-CLOSEST = the complete trial closest to the own-input mean (independent re-derivation)", t_star == want,
          f"{t_star} {want}")  # fmt: skip
    rows_t = [{"trial": 1, "complete": True, "own_auc": 0.25, "cons_auc": 0.625}, {"trial": 2, "complete": True, "own_auc": 0.75, "cons_auc": 0.5},
              {"trial": 3, "complete": True, "own_auc": 0.25, "cons_auc": 0.375}, {"trial": 4, "complete": True, "own_auc": 0.75, "cons_auc": 0.5},
              {"trial": 5, "complete": False}]  # fmt: skip
    check("5.6 ties (exact binary values): equal own-input distance -> smaller consensus-B distance (trials 2, 4), then "
          "lower t (2); all equal -> lower t; no complete trial -> none",
          closest_trial(rows_t) == 2 and closest_trial([rows_t[2], rows_t[0]]) == 1 and closest_trial([rows_t[4]]) is None,
          f"{closest_trial(rows_t)} {closest_trial([rows_t[2], rows_t[0]])}")
    s5 = [{"trial": t, "complete": True, "cons_auc": 0.6, "own_sp": 0.9, "own_mp": 0.9, "own_auc": 0.5} for t in range(1, 11)]
    s6_ok = [dict(r, cons_auc=0.601) for r in s5]
    v = replace_verdict(1, s5, s6_ok)
    check("5.6: Delta_B exactly +0.001 (here 0.601 - 0.6) counts as computed in float64; >= rule applied to that float",
          v["ii_delta_B"] == (v["delta_B"] >= 0.001) and v["replace"] == (v["delta_B"] >= 0.001), v)  # fmt: skip
    s6_big = [dict(r, cons_auc=0.61) for r in s5]
    check("5.6: replace iff changed >= 1 AND Delta_B >= 0.001 AND SP/MP not lower",
          replace_verdict(1, s5, s6_big)["replace"] and not replace_verdict(0, s5, s6_big)["replace"]
          and not replace_verdict(1, s5, [dict(r, own_sp=0.89) for r in s6_big])["replace"]
          and not replace_verdict(1, s5, [dict(r, own_mp=0.8) for r in s6_big])["replace"]
          and not replace_verdict(1, s5, [dict(r, cons_auc=0.6009) for r in s5])["replace"], "")  # fmt: skip
    s5_part = [dict(r, complete=r["trial"] != 2) for r in s5]
    s6_part = [dict(r, complete=r["trial"] != 5, cons_auc=0.6 if r["trial"] in (2, 5) else 0.61) for r in s5]
    v = replace_verdict(1, s5_part, s6_part)
    check("5.6: Delta_B is paired over the trials complete for BOTH sets", v["shared_trials"] == [1, 3, 4, 6, 7, 8, 9, 10]
          and abs(v["delta_B"] - 0.01) < 1e-12, v)  # fmt: skip
    check("5.6: no shared complete trial -> no replace", not replace_verdict(1, [dict(r, complete=False) for r in s5], s6_big)["replace"])

    # ---- guard
    forbidden = ["/x/clean_LOCAL_NOT_FOR_UPLOAD.csv", "/x/Clean/y", "/x/packed_x.csv", "/d/track_2/tier_1_multiview_caption/e",
                 "/d/video_to_data_challenge/track_2/baseline/x", str(ME / "data/human_motion_data/whole_body/x"),
                 str(ME / "data/v2d_challenge_data_public/x"), "/z/README.txt", "/z/train_rl.log",
                 str(ME / "tmp_analysis/t2_zx07_20261028/x"), "/z/best_weights.json", "/z/seed_best.csv", "/z/remote_reps.csv",
                 str(ROOT / "REPORT_R5.md"), "/z/composites.csv", "/z/per_episode_zx.csv",
                 str(ROOT / "clean_LOCAL_NOT_FOR_UPLOAD/x.json"), str(ROOT / "pack_r6_LOCAL_NOT_FOR_UPLOAD/x")]  # fmt: skip
    refused = []
    for p in forbidden:
        try:
            C.guard(p)
        except PermissionError:
            refused.append(p)
    check("guard refuses every forbidden pattern of PREREG_R6 5.7", refused == forbidden, sorted(set(forbidden) - set(refused)))
    allowed = [str(PH["S"]["runs"] / "ep00_raw_21000/run1/try1/results/episode_000000.parquet"),
               str(PH["F"]["runs"] / "ep26_zx_k3fix_92000/run10/try1/results/episode_000026.parquet"),
               str(EXPORTS / "ep26_k3ca_25298/export.json"), str(R1 / "refs/B/episode_000026.npz"), str(POOL), str(INCS),
               str(C.K3CA_REFS / "episode_000026.npz"), str(C.G2_REFS / "episode_000019.npz"), str(C.T2 / "mesh"),
               str(PH["S"]["evals"]), str(PH["F"]["manifest"]), str(FINAL), str(SEL_S6), str(TRACKS)]  # fmt: skip
    passed = []
    for p in allowed:
        try:
            C.guard(p)
            passed.append(p)
        except PermissionError:
            pass
    check("guard passes the noisy-only inputs of 5.2-5.6", passed == allowed, sorted(set(allowed) - set(passed)))
    check("input references: raw = released Tier 2, k3fix / k3 / k3ca / g2 = round 1's INPUT_REF",
          C.INPUT_REF["raw"] == C.T2 and C.INPUT_REF["k3ca"] == ME / "tmp_analysis/tier2/k3ca/frame_refs_k3ca"
          and C.INPUT_REF["k3fix"] == ME / "tmp_analysis/tier2/k3fix_pilot/frame_refs_k3fix"
          and C.INPUT_REF["k3"] == ME / "tmp_analysis/tier2/kalman/eval/frame_refs_k3"
          and C.INPUT_REF["g2"] == Path("/root/autodl-tmp/zuxing.lu/data/human_motion_data_tier2_g2-2026-10-24/frame_refs_g2"), "")  # fmt: skip
    # ---- incumbent table (pinned, noisy-only records)
    try:
        if pin_hash(INCS) != INCS_SHA:
            raise SystemExit("incumbents_r6.json != 44d62d8c")
        incs = load_incumbents()
        ok_t = all(incs[e]["job"] == INC_TABLE[e][0] for e in range(30))
        ulp = [ep_key(e) for e in range(30) if incs[e]["m_P_frozen"] != incs[e]["m_P_frozen_field_resolver"]]
        detail = (f"30 incumbents; m_P_frozen ROUND5 mean {np.mean([incs[e]['m_P_frozen'] for e in range(30)]):.6f}; "
                  f"resolver field differs in the last bits on {ulp}")
    except (SystemExit, KeyError, ValueError, OSError) as err:
        ok_t, detail = False, f"error: {err}"
    check("incumbents: 30 rows == PREREG_R6 4 table; frozen runs == the frozen scores CSV of the selecting round; "
          "m_P_frozen = the pinned stats() mean == that round's frozen job stats mean (exact), within 1e-12 of the "
          "resolver's field and 5e-7 of the table", ok_t, detail)  # fmt: skip
    out = {"written_at": C.now(), "script_sha256": raw_sha(MINE), "round2_scorer_sha256": P2_SHA, "gate_script_sha256": GATE_SHA,
           "consensus_script_sha256": CONS_SHA, "pass": not problems, "n_tests": len(res), "problems": problems,
           "tests": res, "note": "synthetic values plus the pinned incumbent records; no round-6 score was read or computed; "
                                 "no clean value"}  # fmt: skip
    write_new(SELFTEST, C.dump(out))
    for r in res:
        print(("ok   " if r["ok"] else "FAIL ") + r["test"] + (f"  [{r['detail']}]" if r["detail"] and not r["ok"] else ""))
    print(f"selftest: {len(res)} tests, {len(problems)} problems -> {SELFTEST}")
    return 0 if not problems else 1


# ================================================================================================ check (dry)
def cmd_check(a) -> int:
    ph = a.phase
    lines: list[str] = []

    def p(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    p(f"t2cons_r6_consensus CHECK phase {ph} {C.now()}: no metric, reference or score is computed")
    fatal, info = chain(ph, "check")
    for x in info:
        p(f"chain: {x}")
    for x in fatal:
        p(f"chain FAILED: {x}")
    jobs = load_jobs(ph)
    evals, n_not_ok, problems = load_evals(ph, jobs)
    nr = PH[ph]["nruns"]
    full = [j for j in sorted(jobs) if sorted(evals.get(j, {})) == list(range(1, nr + 1))]
    p(f"phase {ph} jobs {len(jobs)}; export ok {sum(j['export_status'] == 'ok' for j in jobs.values())}; complete ({nr} ok "
      f"runs) {len(full)}; ok runs {sum(len(v) for v in evals.values())}; non-ok rows in {PH[ph]['evals'].name}: {n_not_ok}")
    for j in sorted(jobs):
        if jobs[j]["export_status"] != "ok":
            p(f"NO EXPORT {j}: {jobs[j]['export_status']}")
        elif j not in full:
            p(f"INCOMPLETE {j}: ok runs {sorted(evals.get(j, {}))} of {nr}")
    sigs = sorted({r["seed_signature"] for j in evals for r in evals[j].values()})
    p(f"seed signatures over {sum(len(v) for v in evals.values())} runs: {sigs}")
    files = rollout_files(evals)
    missing = [f for f in files if not Path(f).is_file()]
    problems += [f"missing rollout {f}" for f in missing]
    R = PH[ph]["rollouts"]
    if not R.exists() and not fatal and not problems:
        write_new(R, "".join(f"{sha(f)}  {f}\n" for f in sorted(files)))
        p(f"wrote {R.relative_to(ROOT)}: {len(files)} rollout parquets, sha256 {raw_sha(R)}")
    elif R.exists():
        rp = check_rollouts(ph, evals)
        problems += rp
        p(f"{R.relative_to(ROOT)} exists (sha256 {raw_sha(R)}): {len(files) - len(rp)}/{len(files)} unchanged")
    for x in problems:
        p(f"problem: {x}")
    ok = not fatal and not problems
    p(f"check: {'OK' if ok else 'PROBLEMS'}")
    write_new(ROOT / "dryrun" / f"r6_consensus_check_{ph}_{C.stamp()}.txt", "\n".join(lines) + "\n")
    return 0 if ok else 1


# ================================================================================================ gate (5.3)
def cmd_gate(a) -> int:
    ph = a.phase
    if not require(ph, "gate"):
        return 2
    F = files_of(ph)
    if F["gate_csv"].exists() or F["gate_json"].exists():
        print("FATAL: gate output exists (new files only)")
        return 2
    jobs = load_jobs(ph)
    evals, _, lp = load_evals(ph, jobs)
    rp = check_rollouts(ph, evals) + lp
    if rp:
        print("FATAL: rollouts / runs differ:\n  " + "\n  ".join(rp[:20]))
        return 2
    before = C.integrity()
    K = C.load_kit()
    cache: dict = {}
    grids: dict = {}
    rows = []
    for job in sorted(evals):
        j = jobs[job]
        e = j["episode"]
        if e not in grids:
            grids[e] = C.grid(K, e)
        frames, names = grids[e]
        for k in sorted(evals[job]):
            rec = evals[job][k]
            row = {"job": job, "episode": e, "run": k, "try": rec["try"], "input": j["input"], "id": j["role"]}
            try:
                conv, inref, inf = C.rollout_on_grid(K, j, rec, frames, names)
                m0 = C.metrics_vs(K, conv, inref, names)
                aucs = [m0["auc"]] + [G.auc_seed(K, cache, conv, inref, names, s) for s in G.SEEDS[1:]]
            except ValueError as err:
                row.update({"status": "failed", "problem": str(err)})
                row["flags"] = "; ".join(run_flags(row))
                rows.append(row)
                continue
            ev = {m: rec[m] for m in C.METRICS}
            lo, hi = min(aucs), max(aucs)
            row.update({
                "status": "ok", "eval_auc": ev["auc"], **{f"auc_s{s}": x for s, x in zip(G.SEEDS, aucs)},
                "seed_min": lo, "seed_max": hi, "seed_range": hi - lo,
                "excess_outside_range": max(lo - ev["auc"], ev["auc"] - hi, 0.0),
                "band_ok": bool(lo - G.BAND <= ev["auc"] <= hi + G.BAND), "inside_range": bool(lo <= ev["auc"] <= hi),
                "d_auc_seed0": m0["auc"] - ev["auc"], "eval_mppe": ev["mppe"], "mine_mppe": m0["mppe"],
                "d_mppe": m0["mppe"] - ev["mppe"], "eval_sp_sr": ev["sp_sr"], "mine_sp_sr": m0["sp_sr"],
                "eval_mp_sr": ev["mp_sr"], "mine_mp_sr": m0["mp_sr"],
                "conv_pos_residual_m": inf["conv_pos_residual_m"], "conv_ang_residual_rad": inf["conv_ang_residual_rad"],
            })  # fmt: skip
            row["flags"] = "; ".join(run_flags(row))
            rows.append(row)
        print(f"  {job}: {sum(1 for r in rows if r['job'] == job and r['status'] == 'ok')}/{len(evals[job])} scored against "
              f"its own input; flagged {sum(1 for r in rows if r['job'] == job and r['flags'])}", flush=True)  # fmt: skip
    after = C.integrity()
    diff = C.integrity_diff(before, after) + C.integrity_diff(json.loads(BASELINE.read_text()), after)
    flagged_jobs = sorted({r["job"] for r in rows if r["flags"]})
    ok = [r for r in rows if r["status"] == "ok"]
    passed = not diff
    cols = ["job", "episode", "run", "try", "input", "id", "status", "problem", "flags", "eval_auc"] + \
        [f"auc_s{s}" for s in G.SEEDS] + \
        ["seed_min", "seed_max", "seed_range", "excess_outside_range", "band_ok", "inside_range", "d_auc_seed0",
         "eval_mppe", "mine_mppe", "d_mppe", "eval_sp_sr", "mine_sp_sr", "eval_mp_sr", "mine_mp_sr",
         "conv_pos_residual_m", "conv_ang_residual_rad"]  # fmt: skip
    write_new(F["gate_csv"], C.csv_text(rows, cols))
    out = {
        "written_at": C.now(), "phase": ph, "script_sha256": raw_sha(MINE), "round2_scorer_sha256": P2_SHA,
        "consensus_script_sha256": CONS_SHA, "gate_script_sha256": GATE_SHA, "prereg_r6_sha256": PREREG_SHA,
        "pins_b_sha256": raw_sha(PINS_B),
        "rule": "PREREG_R6 5.3 (round-1 AMENDMENTS A1.3): a run is flagged iff the evaluator AUC is outside [min_s AUC_s - "
                "0.02, max_s AUC_s + 0.02] (s = 0..9), |dMPPE| > 0.5 cm, SP-SR or MP-SR differs from the evaluator, or it "
                "is a problem run. Phase S: a candidate with a flagged run is validation_failed. Phase F: a flagged "
                "finalist run disqualifies the finalist; a flagged incumbent run is left out of m_P_fresh and its trial "
                "is incomplete. A kit / stock-runtime change fails the gate.",
        "thresholds": G.GATE, "pass": passed,
        "validation_failed_jobs": flagged_jobs if (passed and ph == "S") else [],
        "flagged_jobs": flagged_jobs, "flagged_run_keys": [[r["job"], r["run"]] for r in rows if r["flags"]],
        "n_jobs": len(evals), "n_runs": len(rows), "n_scored": len(ok), "n_flagged_runs": sum(1 for r in rows if r["flags"]),
        "flagged_runs": [f"{r['job']} r{r['run']}: {r['flags']}" for r in rows if r["flags"]],
        "n_inside_seed_range": sum(r["inside_range"] for r in ok),
        "max_excess_outside_range": max((r["excess_outside_range"] for r in ok), default=None),
        "max_seed_range": max((r["seed_range"] for r in ok), default=None),
        "max_abs_d_auc_seed0": max((abs(r["d_auc_seed0"]) for r in ok), default=None),
        "max_abs_d_mppe": max((abs(r["d_mppe"]) for r in ok), default=None),
        "max_conv_pos_residual_m": max((r["conv_pos_residual_m"] for r in ok), default=None),
        "max_conv_ang_residual_rad": max((r["conv_ang_residual_rad"] for r in ok), default=None),
        "kit_stock_unchanged": not diff, "integrity_diff": diff[:50],
        "vertex_meshes": {n: {"file": str(v[1]), "sha256": sha(v[1])} for n, v in sorted(K.verts.items())},
        "rollouts_list_sha256": raw_sha(PH[ph]["rollouts"]), "csv_sha256": raw_sha(F["gate_csv"]),
        "note": "own-input check only (each run vs its OWN input track); seeds 1..9 used only here; no consensus "
                "reference is read; nothing is selected",
    }  # fmt: skip
    write_new(F["gate_json"], C.dump(out))
    print(json.dumps({k: out[k] for k in ("pass", "n_jobs", "n_runs", "n_scored", "n_flagged_runs", "flagged_jobs",
                                          "n_inside_seed_range", "max_excess_outside_range", "max_abs_d_mppe",
                                          "kit_stock_unchanged")}, indent=1))  # fmt: skip
    for x in out["flagged_runs"][:40]:
        print("  flagged:", x)
    return 0 if passed else 1


# ================================================================================================ score (5.2)
def cmd_score(a) -> int:
    ph = a.phase
    if not require(ph, "score"):
        return 2
    F = files_of(ph)
    if F["scores_csv"].exists() or F["jstats"].exists():
        print("FATAL: score output exists (scores are computed once)")
        return 2
    g = C.read_json(F["gate_json"])
    excluded = list(g.get("validation_failed_jobs") or [])  # phase S only (phase F: flagged runs are excluded later)
    jobs = load_jobs(ph)
    evals, _, lp = load_evals(ph, jobs)
    rp = check_rollouts(ph, evals) + lp
    if rp:
        print("FATAL: rollouts / runs differ:\n  " + "\n  ".join(rp[:20]))
        return 2
    roll = C.hash_list(PH[ph]["rollouts"])
    before = C.integrity()
    K = C.load_kit()
    refs = C.load_refs(K)  # pinned: round-1 refs/build.json (9646de2d, PREREG_R6 11), every output re-hashed; never rebuilt
    rows, problems, grids = [], [], {}
    scored_jobs = [j for j in sorted(evals) if j not in excluded]
    for job in scored_jobs:
        j = jobs[job]
        e = j["episode"]
        if e not in grids:
            grids[e] = C.grid(K, e)
        frames, names = grids[e]
        for k in sorted(evals[job]):
            rec = evals[job][k]
            base = {"job": job, "episode": e, "run": k, "try": rec["try"], "variant": VARIANT, "input": j["input"],
                    "id": j["role"], "checkpoint_sha256": j["export_sha256"], "parquet": rec["parquet"],
                    "parquet_sha256": roll[rec["parquet"]]}  # fmt: skip
            try:
                conv, _, inf = C.rollout_on_grid(K, j, rec, frames, names)
            except ValueError as err:
                st = "conversion_failed" if "rigid transform" in str(err) else "failed"
                problems.append(f"{job} run {k}: {err}")
                rows.append({**base, "status": st, "problem": str(err)})
                continue
            m = C.metrics_vs(K, conv, refs[VARIANT][e], names)
            rows.append({**base, "status": "ok", **m, "conv_pos_residual_m": inf["conv_pos_residual_m"],
                         "conv_ang_residual_rad": inf["conv_ang_residual_rad"]})  # fmt: skip
        print(f"  {job}: {sum(1 for r in rows if r['job'] == job and r['status'] == 'ok')} runs scored against refs/B", flush=True)
    write_new(F["scores_csv"], C.csv_text(rows, SCORE_COLS))
    jstats = {}
    for job in scored_jobs:
        rs = sorted([r for r in rows if r["job"] == job and r["status"] == "ok"], key=lambda r: r["run"])
        d = {"episode": jobs[job]["episode"], "input": jobs[job]["input"], "role": jobs[job]["role"], "n": len(rs),
             "runs": [r["run"] for r in rs]}  # fmt: skip
        for m in C.METRICS:
            d[m] = [r[m] for r in rs]
            if rs:
                d[m + "_stats"] = C.stats(d[m])
        jstats[job] = d
    after = C.integrity()
    diff = C.integrity_diff(before, after)
    out = {"written_at": C.now(), "phase": ph, "script_sha256": raw_sha(MINE), "variant": VARIANT, "prereg_r6_sha256": PREREG_SHA,
           "pins_b_sha256": raw_sha(PINS_B), "gate_json_sha256": raw_sha(F["gate_json"]),
           "refs_build_sha256": raw_sha(R1 / "refs/build.json"), "rollouts_list_sha256": raw_sha(PH[ph]["rollouts"]),
           "scores_csv_sha256": raw_sha(F["scores_csv"]), "validation_failed_jobs": excluded, "scored_jobs": scored_jobs,
           "n_rows": len(rows), "n_ok_rows": sum(r["status"] == "ok" for r in rows), "problems": problems,
           "integrity_kit_stock_unchanged": not diff, "integrity_diff": diff[:50], "jobs": jstats}  # fmt: skip
    write_new(F["jstats"], C.dump(out))
    print(f"phase {ph}: scored {len(scored_jobs)} jobs against refs/{VARIANT}: {out['n_ok_rows']}/{len(rows)} ok rows; "
          f"problems {len(problems)}; kit/stock unchanged {not diff}")  # fmt: skip
    for x in problems[:20]:
        print("  problem:", x)
    return 0 if not diff else 1


def scored_inputs(ph: str):
    """Shared by both selections: verified scores, gate, jobs, runs."""
    F = files_of(ph)
    js, g = C.read_json(F["jstats"]), C.read_json(F["gate_json"])
    if js.get("scores_csv_sha256") != raw_sha(F["scores_csv"]) or js.get("script_sha256") != raw_sha(MINE) or \
            js.get("gate_json_sha256") != raw_sha(F["gate_json"]) or not js.get("integrity_kit_stock_unchanged") or \
            js.get("validation_failed_jobs") != (g.get("validation_failed_jobs") or []) or \
            js.get("pins_b_sha256") != raw_sha(PINS_B) or js.get("rollouts_list_sha256") != raw_sha(PH[ph]["rollouts"]):  # fmt: skip
        raise SystemExit("scores changed since scoring, scored by another script, or integrity flagged")
    jobs = load_jobs(ph)
    evals, _, lp = load_evals(ph, jobs)
    rp = check_rollouts(ph, evals) + lp
    if rp:
        raise SystemExit("rollouts / runs changed:\n  " + "\n  ".join(rp[:20]))
    scores = {(r["job"], int(r["run"])): r for r in C.read_csv(F["scores_csv"]) if r["status"] == "ok"}
    flagged = {(j, int(k)) for j, k in g.get("flagged_run_keys") or []}
    return js, g, jobs, evals, scores, flagged


def frozen_hashes(names: list[str]) -> str:
    return "".join(f"{raw_sha(pin_path(n))}  {n}\n" for n in names)


# ================================================================================================ select S (5.4)
def cmd_select_s() -> int:
    if not require("S", "select"):
        return 2
    js, g, jobs, evals, scores, flagged = scored_inputs("S")
    excluded = set(g.get("validation_failed_jobs") or [])
    incs = load_incumbents()
    jst = js["jobs"]
    eps, longrows, lines, finalists = [], [], [], []
    for e in range(30):
        I = incs[e]
        cands = sorted(j for j in jobs if jobs[j]["episode"] == e)
        info = {}
        for c in cands:
            s = jst.get(c)
            info[c] = {"export_ok": jobs[c]["export_status"] == "ok", "own": evals.get(c, {}),
                       "flagged": c in excluded or any(jj == c for jj, _ in flagged),
                       "cons": None if s is None or not s["n"] else {"runs": s["runs"], "auc": s["auc"], "mppe": s["mppe"]}}  # fmt: skip
        f, rows = select_s(I["m_P_frozen"], cands, info)
        for r in rows:
            j = jobs[r["job"]]
            s = jst.get(r["job"], {})
            r.update(input=j["input"], owner=j["owner"], export_status=j["export_status"],
                     own_auc=[evals.get(r["job"], {})[k]["auc"] for k in sorted(evals.get(r["job"], {}))],
                     cons_sp_mean=(s.get("sp_sr_stats") or {}).get("mean"), cons_mp_mean=(s.get("mp_sr_stats") or {}).get("mean"))  # fmt: skip
        rows.sort(key=lambda r: (-(r.get("cons_mean") if r.get("cons_mean") is not None else -1.0), r["job"]))
        rec = {"episode": e, "incumbent": I["job"], "incumbent_input": I["input"], "m_P_frozen": I["m_P_frozen"],
               "bar": I["m_P_frozen"] + MARGIN, "n_candidates": len(cands), "finalist": f,
               "n_eligible": sum(r["eligible"] for r in rows), "n_qualifying": sum(r["qualifies"] for r in rows),
               "candidates": rows}  # fmt: skip
        eps.append(rec)
        if f is not None:
            j = jobs[f]
            fr_row = next(r for r in rows if r["job"] == f)
            finalists.append({"job": f, "episode": e, "input": j["input"], "owner": j["owner"], "tree": j["tree"],
                              "iteration": j["iteration"], "training_sha256": j["training_sha256"],
                              "export_sha256": j["export_sha256"], "export_file": j["export_file"],
                              "cons_S_mean": fr_row["cons_mean"], "cons_S_var": fr_row["cons_var"],
                              "cons_S_mppe_mean": fr_row["cons_mppe_mean"], "floor_S": fr_row["floor_count"],
                              "incumbent": I["job"], "m_P_frozen": I["m_P_frozen"], "bar": I["m_P_frozen"] + MARGIN})  # fmt: skip
        for r in rows:
            longrows.append({"episode": ep_key(e), "job": r["job"], "input": r["input"], "owner": r["owner"],
                             "export_status": r["export_status"], "n_own_ok": r["n_own_ok"], "floor_count": r["floor_count"],
                             "flagged": r["flagged"], **{f"cons_auc_r{k}": (r.get("cons_auc") or [None] * 3)[k - 1]
                                                         if len(r.get("cons_auc") or []) >= k else None for k in (1, 2, 3)},
                             "cons_mean": r.get("cons_mean"), "cons_sd": math.sqrt(r["cons_var"]) if "cons_var" in r else None,
                             "cons_mppe_mean": r.get("cons_mppe_mean"), "incumbent": I["job"], "m_P_frozen": I["m_P_frozen"],
                             "delta_vs_m_P": r.get("delta_vs_m_P"), "eligible": r["eligible"], "qualifies": r["qualifies"],
                             "finalist": r["job"] == f, "why_not": "; ".join(r["why_not"])})  # fmt: skip
        top = [r for r in rows if r.get("cons_mean") is not None][:3]
        lines.append(f"ep{e:02d} P {I['job']} [{I['input']}] m_P_frozen {I['m_P_frozen']:.6f} (bar {I['m_P_frozen'] + MARGIN:.6f}); "
                     f"{len(cands)} candidates, {rec['n_eligible']} eligible, {rec['n_qualifying']} qualifying -> "
                     f"finalist {f or 'none'}")  # fmt: skip
        for r in top:
            lines.append(f"      {r['job']} [{r['input']}] cons_S {r['cons_mean']:.6f} sd {math.sqrt(r['cons_var']):.4f} "
                         f"floor {r['floor_count']}/3 eligible={r['eligible']} qualifies={r['qualifies']} {r['why_not']}")
    gen = C.now()
    hashes = {"prereg_r6_sha256": PREREG_SHA, "pins_b_sha256": raw_sha(PINS_B), "script_sha256": raw_sha(MINE),
              "selftest_sha256": raw_sha(SELFTEST), "manifest_s6_sha256": raw_sha(PH["S"]["manifest"]),
              "own_input_evals_s6_sha256": raw_sha(PH["S"]["evals"]), "rollouts_s6_sha256": raw_sha(PH["S"]["rollouts"]),
              "gate_json_sha256": raw_sha(files_of("S")["gate_json"]), "scores_csv_sha256": raw_sha(files_of("S")["scores_csv"]),
              "jstats_sha256": raw_sha(files_of("S")["jstats"]), "pool_r6_sha256": POOL_SHA, "incumbents_r6_sha256": INCS_SHA}  # fmt: skip
    sel = {"generated_at": gen, "phase": "S", "synthetic": False, "noisy_only": True, "binding_variant": VARIANT, **hashes,
           "rule": RULE_S, "validation_failed_jobs": sorted(excluded), "n_finalists": len(finalists),
           "finalist_episodes": [ep_key(x["episode"]) for x in finalists], "episodes": eps}  # fmt: skip
    final = {"generated_at": gen, "prereg_r6_sha256": PREREG_SHA, "script_sha256": hashes["script_sha256"],
             "pins_b_sha256": hashes["pins_b_sha256"], "rule": RULE_S, "finalists": finalists,
             "no_finalist_episodes": [ep_key(x["episode"]) for x in eps if x["finalist"] is None],
             "note": "phase-F job list (PREREG_R6 3.3): the 30 incumbents of inputs_r6/incumbents_r6.json + these finalists, "
                     "each with its single phase-S export; locked by SELECTION_S6.sha256"}  # fmt: skip
    written = []

    def w(name: str, text: str) -> None:
        write_new(ROOT / name, text)
        written.append(name)

    w("selection_s6.json", C.dump(sel))
    w("final_r6.json", C.dump(final))
    head = (f"selection_s6 {gen} CST (PREREG_R6 5.4; noisy-only consensus-B vs round 1's frozen refs/B; validation_failed "
            f"{len(excluded)} jobs; finalists {len(finalists)})")  # fmt: skip
    w("selection_s6.txt", "\n".join([head] + lines + ["", "finalists: " + (", ".join(
        f"{x['episode']:02d}:{x['job']} ({x['cons_S_mean']:.4f} vs bar {x['bar']:.4f})" for x in finalists) or "none")]) + "\n")
    w("selection_s6_candidates.csv", C.csv_text(longrows, list(longrows[0].keys())))
    md = ["# t2_consensus_r6_20261031: phase-S selection (noisy-only; PREREG_R6 5.4)", "",
          f"Generated {gen} CST by `scripts/t2cons_r6_consensus.py` ({hashes['script_sha256'][:16]}...), pinned in "
          f"`PREREG_R6_PINS_b.md` ({hashes['pins_b_sha256'][:16]}...). 250 screen candidates x 3 official runs, each run scored "
          "against round 1's frozen consensus refs/B (variant B) and gated against its own input (5.3). Finalist = the "
          "S-eligible candidate with the highest 3-run consensus mean, provided it is >= m_P_frozen + 0.01. No clean or "
          "LOCAL file was opened; no Tier-1 value was used.", "",
          f"Finalists: {len(finalists)}; validation_failed candidates: {len(excluded)}.", "",
          "| ep | incumbent (input) | m_P_frozen | bar | candidates / eligible / qualifying | best candidate: cons_S (floor) | "
          "**finalist** |", "|---|---|---:|---:|---|---|---|"]  # fmt: skip
    for x in eps:
        best = next((r for r in x["candidates"] if r.get("cons_mean") is not None), None)
        md.append(f"| {x['episode']:02d} | {x['incumbent']} ({x['incumbent_input']}) | {x['m_P_frozen']:.4f} | {x['bar']:.4f} | "
                  f"{x['n_candidates']} / {x['n_eligible']} / {x['n_qualifying']} | "
                  + (f"{best['job']}: {best['cons_mean']:.4f} ({best['floor_count']}/3){' flagged' if best['flagged'] else ''}" if best else "")
                  + f" | **{x['finalist'] or '-'}** |")  # fmt: skip
    w("selection_s6.md", "\n".join(md) + "\n")
    frozen_inputs = [rel_name(p) for p in (files_of("S")["scores_csv"], files_of("S")["jstats"], files_of("S")["gate_csv"],
                                           files_of("S")["gate_json"], SELFTEST, PH["S"]["rollouts"], PH["S"]["evals"],
                                           PH["S"]["manifest"], PINS_B, ROOT / "PREREG_R6_PINS.md", DRYRUN, PREREG, MINE)]  # fmt: skip
    write_new(SEL_S6, frozen_hashes(written + frozen_inputs))  # LAST: from here the phase-S selection is frozen
    print("\n".join(md[6:]))
    print(f"\nSELECTION_S6.sha256 written {C.now()} (frozen): {raw_sha(SEL_S6)}")
    return 0


# ================================================================================================ select F (5.5 / 5.6)
def cmd_select_f() -> int:
    if not require("F", "select"):
        return 2
    js, g, jobs, evals, scores, flagged = scored_inputs("F")
    incs = load_incumbents()
    final = C.read_json(FINAL)
    fin_by_ep = {int(x["episode"]): x for x in final["finalists"]}

    def runinfo_of(job: str) -> dict[int, dict]:
        out = {}
        for k in range(1, NTRIALS + 1):
            ev = evals.get(job, {}).get(k)
            sc = scores.get((job, k))
            if ev is None:
                continue
            out[k] = {"ok": sc is not None, "flagged": (job, k) in flagged, "try": ev["try"], "parquet": ev["parquet"],
                      "own_auc": ev["auc"], "own_sp": ev["sp_sr"], "own_mp": ev["mp_sr"], "own_mppe": ev["mppe"],
                      "cons_auc": float(sc["auc"]) if sc else None, "cons_sp": float(sc["sp_sr"]) if sc else None,
                      "cons_mp": float(sc["mp_sr"]) if sc else None, "cons_mppe": float(sc["mppe"]) if sc else None}  # fmt: skip
        return out

    runinfo = {j: runinfo_of(j) for j in jobs}
    eps, s5, s6, lines, table = [], {}, {}, [], []
    for e in range(30):
        I = incs[e]
        f = fin_by_ep.get(e)
        d = decide_f(I["m_P_frozen"], runinfo[I["job"]], f["job"] if f else None, runinfo.get(f["job"]) if f else None)
        s5[e] = I["job"]
        s6[e] = f["job"] if d["changed"] else I["job"]
        pick = s6[e]
        rec = {"episode": e, "incumbent": I["job"], "incumbent_input": I["input"], "finalist": f["job"] if f else None,
               "finalist_input": f["input"] if f else None, "finalist_cons_S_mean": f["cons_S_mean"] if f else None,
               **d, "pick": pick, "pick_input": jobs[pick]["input"],
               "incumbent_runs": {k: {x: v for x, v in r.items() if x != "parquet"} for k, r in runinfo[I["job"]].items()},
               "finalist_runs": ({k: {x: v for x, v in r.items() if x != "parquet"} for k, r in runinfo[f["job"]].items()}
                                 if f else None)}  # fmt: skip
        eps.append(rec)
        table.append({"episode": ep_key(e), "incumbent": I["job"], "incumbent_input": I["input"], "m_P_frozen": I["m_P_frozen"],
                      "m_P_fresh": d["m_P_fresh"], "incumbent_good_runs": len(d["incumbent_good_runs"]), "bar": d["bar"],
                      "finalist": f["job"] if f else "", "finalist_input": f["input"] if f else "",
                      "cons_S_mean": f["cons_S_mean"] if f else None, "m_F": d.get("m_F"),
                      "finalist_floor": d.get("floor_count"), "changed": d["changed"], "pick": pick,
                      "why_not": "; ".join(d["why_not"])})  # fmt: skip
        lines.append(f"ep{e:02d} P {I['job']} [{I['input']}] m_P_frozen {I['m_P_frozen']:.6f} m_P_fresh "
                     f"{fmt(d['m_P_fresh'], '.6f') or 'inf'} ({len(d['incumbent_good_runs'])} good runs) bar {fmt(d['bar'], '.6f') or 'inf'}"
                     + (f"; finalist {f['job']} [{f['input']}] m_F {fmt(d.get('m_F'), '.6f')} floor {d.get('floor_count')}/10"
                        f" -> {'REPLACES' if d['changed'] else 'no: ' + '; '.join(d['why_not'])}" if f else "; no finalist"))  # fmt: skip
    changed = [ep_key(x["episode"]) for x in eps if x["changed"]]
    rows5 = trial_rows(s5, runinfo)
    rows6 = trial_rows(s6, runinfo)
    t_star = closest_trial(rows6)
    verdict = replace_verdict(len(changed), rows5, rows6)
    verdict["r6_closest_trial"] = t_star
    if verdict["replace"] and t_star is None:
        raise SystemExit("internal: replace without an R6-CLOSEST trial")
    r6_file = {}
    if t_star is not None:
        roll = C.hash_list(PH["F"]["rollouts"])
        for e in range(30):
            j = jobs[s6[e]]
            r = runinfo[s6[e]][t_star]
            r6_file[ep_key(e)] = {"job": s6[e], "run": t_star, "try": r["try"], "source": "round6 phase F",
                                  "role": j["role"], "input": j["input"], "frame_reference": j["input_reference"],
                                  "export_sha256": j["export_sha256"], "export_file": j["export_file"],
                                  "rollout": r["parquet"], "rollout_sha256": roll[r["parquet"]]}  # fmt: skip
    gen = C.now()
    F = files_of("F")
    hashes = {"prereg_r6_sha256": PREREG_SHA, "pins_b_sha256": raw_sha(PINS_B), "script_sha256": raw_sha(MINE),
              "selftest_sha256": raw_sha(SELFTEST), "final_r6_sha256": raw_sha(FINAL), "selection_s6_lock_sha256": raw_sha(SEL_S6),
              "manifest_f6_sha256": raw_sha(PH["F"]["manifest"]), "own_input_evals_f6_sha256": raw_sha(PH["F"]["evals"]),
              "rollouts_f6_sha256": raw_sha(PH["F"]["rollouts"]), "gate_json_sha256": raw_sha(F["gate_json"]),
              "scores_csv_sha256": raw_sha(F["scores_csv"]), "jstats_sha256": raw_sha(F["jstats"])}  # fmt: skip
    upload = ("R6-CLOSEST replaces R5-CLOSEST as the upload-eligible Tier-2 file (PREREG_R6 5.6)" if verdict["replace"]
              else "R5-CLOSEST (e8a03a4d...) stays the upload-eligible Tier-2 file (PREREG_R6 5.6)")  # fmt: skip
    sel = {"generated_at": gen, "phase": "F", "synthetic": False, "noisy_only": True, "binding_variant": VARIANT, **hashes,
           "rule": RULE_F, "changed_episodes": changed, "episodes": eps,
           "compositions": {"S5": {ep_key(e): s5[e] for e in range(30)}, "S6": {ep_key(e): s6[e] for e in range(30)}},
           "trials": {"S5": rows5, "S6": rows6}, "replace_verdict": verdict, "r6_closest_trial": t_star,
           "r6_closest_file": r6_file, "upload_eligibility": upload, "r5_closest": R5_CLOSEST,
           "flagged_runs": sorted([list(x) for x in flagged])}  # fmt: skip
    written = []

    def w(name: str, text: str) -> None:
        write_new(ROOT / name, text)
        written.append(name)

    w("selection_r6.json", C.dump(sel))
    head = f"selection_r6 {gen} CST (PREREG_R6 5.5 / 5.6; noisy-only; refs = round-1 refs/B, frozen)"
    tl = [f"  trial {r['trial']:2d}: S5 " + (f"own AUC {r5['own_auc']:.4f} SP {r5['own_sp']:.3f} MP {r5['own_mp']:.3f} "
                                             f"cons-B {r5['cons_auc']:.4f}" if r5["complete"] else f"incomplete {r5['missing_episodes']}")
          + " | S6 " + (f"own AUC {r['own_auc']:.4f} SP {r['own_sp']:.3f} MP {r['own_mp']:.3f} cons-B {r['cons_auc']:.4f}"
                        if r["complete"] else f"incomplete {r['missing_episodes']}") for r5, r in zip(rows5, rows6)]  # fmt: skip
    w("selection_r6.txt", "\n".join([head] + lines + ["", f"changed episodes: {', '.join(changed) or 'none'}", "trials:"] + tl + [
        "", f"R6-CLOSEST trial: {t_star}", f"Delta_B {verdict['delta_B']!r} over trials {verdict['shared_trials']}; (i) "
        f"{verdict['i_changed']} (ii) {verdict['ii_delta_B']} (iii) {verdict['iii_sp_mp']} -> replace {verdict['replace']}",
        upload]) + "\n")  # fmt: skip
    w("selection_r6_episodes.csv", C.csv_text(table, list(table[0].keys())))
    tcols = ["set", "trial", "complete", "missing_episodes", "own_auc", "own_sp", "own_mp", "own_mppe", "cons_auc", "cons_sp",
             "cons_mp", "cons_mppe"]  # fmt: skip
    trows = [{"set": "S5", **r, "missing_episodes": " ".join(map(str, r["missing_episodes"]))} for r in rows5] + \
        [{"set": "S6", **r, "missing_episodes": " ".join(map(str, r["missing_episodes"]))} for r in rows6]  # fmt: skip
    w("summary_r6_trials.csv", C.csv_text(trows, tcols))

    def avg(rows, key):
        xs = [r[key] for r in rows if r["complete"]]
        return float(np.mean(np.asarray(xs, dtype=np.float64))) if xs else None

    summ = {"generated_at": gen, "changed_episodes": changed, "r6_closest_trial": t_star, "replace_verdict": verdict,
            "upload_eligibility": upload,
            "means_over_complete_trials": {s: {k: avg(rows, k) for k in tcols[4:]} for s, rows in (("S5", rows5), ("S6", rows6))},
            "n_complete_trials": {"S5": sum(r["complete"] for r in rows5), "S6": sum(r["complete"] for r in rows6)},
            "note": "noisy-only: own-input = each policy vs its OWN Tier-2-derived input (evaluator values); cons-B = vs round "
                    "1's frozen consensus refs/B. Not leaderboard estimates. Own-input composites of S5 and S6 use different "
                    "references on changed episodes and are not compared (report only)."}  # fmt: skip
    w("summary_r6.json", C.dump(summ))
    md = ["# t2_consensus_r6_20261031: round-6 decision (noisy-only; PREREG_R6 5.5 / 5.6)", "",
          f"Generated {gen} CST by `scripts/t2cons_r6_consensus.py` ({hashes['script_sha256'][:16]}...). Phase F: 30 incumbents "
          f"+ {len(fin_by_ep)} finalists x 10 fresh paired trials. No clean or LOCAL file was opened; no Tier-1 value.", "",
          f"Changed episodes: {', '.join(changed) or 'none'}.", "",
          "| ep | incumbent | m_P_frozen | m_P_fresh | bar | finalist | m_F | floor | **pick** |", "|---|---|---:|---:|---:|---|---:|---:|---|"]  # fmt: skip
    for t in table:
        md.append(f"| {t['episode']} | {t['incumbent']} ({t['incumbent_input']}) | {t['m_P_frozen']:.4f} | {fmt(t['m_P_fresh']) or 'inf'} | "
                  f"{fmt(t['bar']) or 'inf'} | {t['finalist'] or '-'} | {fmt(t['m_F'])} | {t['finalist_floor'] if t['finalist'] else ''} | "
                  f"**{t['pick']}** |")  # fmt: skip
    md += ["", "| set | complete trials | own AUC | own SP-SR | own MP-SR | own MPPE | cons-B AUC | cons-B SP-SR | cons-B MP-SR | cons-B MPPE |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
    for s, rows in (("S5", rows5), ("S6", rows6)):
        mm = summ["means_over_complete_trials"][s]
        md.append(f"| {s} | {summ['n_complete_trials'][s]} | " + " | ".join(fmt(mm[k]) for k in tcols[4:]) + " |")
    md += ["", f"R6-CLOSEST trial: {t_star}. Delta_B = {fmt(verdict['delta_B'], '+.6f')} over trials {verdict['shared_trials']} "
           f"(need >= +0.001); S6 / S5 own SP-SR {fmt(verdict['s6_own_sp'])} / {fmt(verdict['s5_own_sp'])}, MP-SR "
           f"{fmt(verdict['s6_own_mp'])} / {fmt(verdict['s5_own_mp'])}. (i) {verdict['i_changed']}, (ii) {verdict['ii_delta_B']}, "
           f"(iii) {verdict['iii_sp_mp']} -> **{upload}**."]  # fmt: skip
    w("summary_r6.md", "\n".join(md) + "\n")
    w("selection_r6.md", "\n".join(md[:5] + [""] + [f"- {x}" for x in lines]) + "\n")
    frozen_inputs = [rel_name(p) for p in (F["scores_csv"], F["jstats"], F["gate_csv"], F["gate_json"], PH["F"]["rollouts"],
                                           PH["F"]["evals"], PH["F"]["manifest"], FINAL, SEL_S6, ROOT / "selection_s6.json",
                                           SELFTEST, PINS_B, ROOT / "PREREG_R6_PINS.md", DRYRUN, PREREG, MINE)]  # fmt: skip
    write_new(SEL_R6, frozen_hashes(written + frozen_inputs))  # LAST: from here round 6 is final
    print("\n".join(md))
    print(f"\nSELECTION_R6.sha256 written {C.now()} (frozen): {raw_sha(SEL_R6)}")
    return 0


def cmd_select(a) -> int:
    if a.phase == "S":
        if SEL_S6.exists():
            print("FATAL: SELECTION_S6.sha256 exists; the phase-S selection is frozen")
            return 2
        return cmd_select_s()
    if SEL_R6.exists():
        print("FATAL: SELECTION_R6.sha256 exists; round 6 is frozen")
        return 2
    return cmd_select_f()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("selftest", "check", "gate", "score", "select"))
    ap.add_argument("--phase", choices=("S", "F"))
    a = ap.parse_args()
    if not sys.flags.dont_write_bytecode or not sys.flags.isolated:
        print("run with the kit python -I -B")
        return 2
    if a.cmd != "selftest" and a.phase is None:
        ap.error("--phase S|F is required")
    return {"selftest": cmd_selftest, "check": cmd_check, "gate": cmd_gate, "score": cmd_score, "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
