#!/usr/bin/env python3
"""t2cons_r7_consensus.py: steps R4 and R6 of fc_runs/submissions/t2_consensus_r7_20261102 (PREREG_R7 5): rollout lists,
own-input gate, consensus-B scores, the phase-S selection (final_r7.json; SELECTION_S7.sha256 written last) and the
phase-F decision with the S7 / REF6 trial composites and R7-CLOSEST (SELECTION_R7.sha256 written last).

Round 6's scorer (scripts/t2cons_r6_consensus.py 9b65e612...) with ONLY the round-7 bookkeeping (PREREG_R7 5.2-5.6):
  K1  pinned code, unchanged: round 2's scorer scripts/t2cons_r2_consensus.py (ddee0b8b...) is compiled from its
      verified bytes; its module code compiles scripts/t2cons_a1_gate.py (a29c997b...) from verified bytes, whose module
      code loads scripts/t2cons_consensus.py (e1b943b1..., re-checked here). The pinned round-1 functions rollout_on_grid,
      metrics_vs, load_kit, load_refs, grid, stats, floor_count, integrity / integrity_diff, check_pins, hash_list,
      git_head and the gate's auc_seed are called unchanged; per-run gate flags = round 2's run_flags, score columns =
      round 2's SCORE_COLS.
  K2  rules: select_s (23 incumbent episodes; round 6's 5.4 unchanged, plus the two reused candidates that keep their
      round-6 phase-F verdict against the same incumbent and are not finalists again), select_strict (7 STRICT
      episodes: the top 2 S-eligible by cons_S, else the pre-registered fallback, else halt), decide_f (round 6's 5.5,
      unchanged), decide_strict (PREREG_R7 5.5 STRICT: rank 1 if confirmed, else rank 2 if confirmed, else unconfirmed by
      floor / m_F / rank, else halt), trial_rows / closest_trial (round 6's 5.6, unchanged: R7-CLOSEST) and ref6_report
      (Delta_B vs REF6, report only). The selftest checks them on synthetic inputs, edge cases and independent
      re-derivations on random inputs.
  K3  path guard: round 6's (round 1's tokens + readme, train_rl.log + t2_zx07_20261028, best_weights, seed_best,
      remote_reps, report_r, composites, per_episode_zx) + tier2_eval.csv, tier2_runs.csv, summary.txt, tier2_best_scan,
      eval_gen_local (PREREG_R7 5.2).
  K4  evaluator (own-input) values = each run's run.json metrics (full precision) == the driver's CSV row (|diff| <=
      1e-6 for AUC / MPPE; SP-SR / MP-SR exactly). Phase S has two run sources: the 154 new screen jobs (this round's
      runs_s/ + own_input_evals_s7.csv) and the 245 reused round-6 candidates (round 6's runs_s/ + own_input_evals_s6.csv,
      every run.json sha256 == its pool_r7.json r6_runs record, every rollout == round 6's ROLLOUTS_S6.sha256).
  K5  incumbents: the 23 rows of inputs_r7/incumbents_r7.json (pinned cb55fa7a...) == the PREREG_R7 4 table; m_P_frozen
      = the pinned stats() mean of the frozen runs == the selecting round's frozen job stats mean (exact), within 1e-12
      of the resolver field and 5e-7 of the table; the two round-6 replacements (ep04, ep11) also == selection_r6.json
      m_F exactly. The 7 report-only references come from inputs_r7/r6refs_r7.json (pinned 0bc58249...).
  K6  reuse reproduction (PREREG_R7 3.2): the reused round-6 runs' own-input gate flags must equal round 6's
      own_input_gate_s6.csv and their consensus scores must equal round 6's consensus_scores_s6.csv exactly; otherwise
      the phase-S selection refuses (a consistency check, not a new measurement).

Commands
  selftest [--trial N] synthetic checks of 5.3-5.6, the guard and the incumbent / reference tables. No round-7 score.
                     -> selftest_r7/selftest_r7.json (with --trial N: dryrun/selftest_trial_r7c_N.json, never pinned)
  check  --phase S|F dry: pin chain, jobs, runs; no metric. Writes once inputs_r7/ROLLOUTS_{S7,F7}.sha256 (5.1) when the
                     chain holds (phase finished), and dryrun/r7_consensus_check_<phase>_<stamp>.txt
  gate   --phase S|F 5.1: every run vs its OWN input (vertex seeds 0..9) -> scores_{s7,f7}/own_input_gate_{s7,f7}.{csv,json}
  score  --phase S|F 5.1: variant-B consensus vs round 1's frozen refs/B -> scores_{s7,f7}/consensus_scores_{s7,f7}.csv,
                     consensus_job_stats_{s7,f7}.json
  select --phase S   5.3 / 5.4 -> selection_s7.{json,txt,md}, selection_s7_candidates.csv, final_r7.json;
                     SELECTION_S7.sha256 LAST
  select --phase F   5.5 / 5.6 -> selection_r7.{json,txt,md}, selection_r7_episodes.csv, summary_r7.{json,md},
                     summary_r7_trials.csv; SELECTION_R7.sha256 LAST
Noisy-only: every data path goes through the extended guard; no clean file, no packer, no Tier-1 value. Variant B only.
New files only (mode 444); nothing is signalled; CPU only.
Run: conda_envs/v2dkit/bin/python -I -B scripts/t2cons_r7_consensus.py <cmd> [--phase S|F]
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
ROOT = ME / "fc_runs/submissions/t2_consensus_r7_20261102"
R1 = ME / "fc_runs/submissions/t2_consensus_20261024"
R6 = ME / "fc_runs/submissions/t2_consensus_r6_20261031"
P2_FILE, P2_SHA = ME / "scripts/t2cons_r2_consensus.py", "ddee0b8b6fa25029275b21a3dc82d15c36f4de71c0a61f2a1046869b01ec13aa"
GATE_SHA = "a29c997b7271e9299db109e97af5210f528484719210bb58685523b0ae0eb16b"
CONS_SHA = "e1b943b10fd1db13ab6f78e5bde7b3c6e6c48f7daeee6230c32c463d14ac41b4"
PREREG, PREREG_SHA = ROOT / "PREREG_R7.md", "749649a79ca5616b8ddbfc5124828c13da20f2e10e850a39647a74d83c939ec6"
CHAIN_DOCS = (  # (file, its sha256 file, hard-coded sha256); each must hold, and every pin line in it must hold
    ("PREREG_R7_PINS.md", "PREREG_R7_PINS.sha256", "8378c5a418c9b1ece0ac8a1c518237666d4649cba211229b18d95d89bff14fce"),
)  # fmt: skip
DRYRUN, DRYRUN_SHA, DRYRUN_LIST = ROOT / "DRYRUN_R7.md", "e7a6095b21b6c3a71eb3d7eafc8dfa2f7ff0d4c036e87ac7ce924541bc9a0981", \
    ROOT / "DRYRUN_R7.sha256"  # fmt: skip
LOCK, LOCK_SHA = ROOT / "LOCK_R7.sha256", "cef6faeb111c78f7336011bea5c585022a433d54ea0e33c515f1a54e5fffa992"
POOL, POOL_SHA = ROOT / "inputs_r7/pool_r7.json", "7c4b553bf2d15ffdae0d2b19097ab6abccfba03d1575b268fb3c0338b0f99464"
INCS, INCS_SHA = ROOT / "inputs_r7/incumbents_r7.json", "cb55fa7a886e5f62f9dbf6f86771fe89c73c6d2137e38c39f82a7a7e72b77e82"
REFS6, REFS6_SHA = ROOT / "inputs_r7/r6refs_r7.json", "0bc58249b13827bfa843747658378df9d7d045bb0f0427e438126c4a0851df03"
SNAP, TRACKS = ROOT / "inputs_r7/SNAPSHOT_R7.sha256", ROOT / "inputs_r7/TRACKS_R7.sha256"
RELOC_TSV, RELOC_LIST = ROOT / "RELOCATION_R7.tsv", ROOT / "RELOCATION_R7.sha256"
R1_TRACKS = R1 / "inputs/TRACKS.sha256"  # round-1 track records (the raw input reference = the released Tier-2 parquet)
PINS_B, PINS_B_SHA = ROOT / "PREREG_R7_PINS_b.md", ROOT / "PREREG_R7_PINS_b.sha256"
BASELINE, BASELINE_SHA = ROOT / "integrity/baseline.json", "ef9b5035c09b8ad40713d9fb08470eeccb2abe8845d4d1cce3037c49da1cd622"
EXPORTS = ROOT / "exports"
FINAL, SEL_S7, SEL_R7 = ROOT / "final_r7.json", ROOT / "SELECTION_S7.sha256", ROOT / "SELECTION_R7.sha256"
SELFTEST = ROOT / "selftest_r7/selftest_r7.json"
R6_EXPORTS, R6_RUNS_S = R6 / "exports", R6 / "runs_s"
R6_FILES = {  # PREREG_R7 11 pins: round 6's frozen records read here
    "manifest_s6": (R6 / "manifest_s6.json", "0f056c994dfa589445a9dab588e9ff32236a7bb048e8bff94d8ffd2cf3c882cf"),
    "evals_s6": (R6 / "own_input_evals_s6.csv", "d152c7b3de01c5c15112904338917243decb28f61763b21b116adb0a7c4f5e40"),
    "rollouts_s6": (R6 / "inputs_r6/ROLLOUTS_S6.sha256", "1301c5c0fa0a41e25042f5880b6941875f9b43689f2c73675130b800db253add"),
    "scores_s6": (R6 / "scores_s6/consensus_scores_s6.csv", "d32c1ed7827257f7fa9ed1d4fa960e432ce510a4ee24f5af491ef89e3be9993c"),
    "jstats_s6": (R6 / "scores_s6/consensus_job_stats_s6.json", "39d79b6bac2af080371ad7e8aa0241511ca61c5672c2f9d61e07f9dbb6a878db"),
    "gate_s6": (R6 / "scores_s6/own_input_gate_s6.csv", "31fabb2925280fbc9db0f0398d86f15eb8d74350b0ed5c04f52eeffc140a7aec"),
    "scores_f6": (R6 / "scores_f6/consensus_scores_f6.csv", "74c9c8856d5f2fd671b1e3f97763a03c26936be794c581581972c2d837e84ab4"),
    "jstats_f6": (R6 / "scores_f6/consensus_job_stats_f6.json", "9eaac6077960d06975597a385552b459df5ff2d76e6c7e39bd9b7243ee3f24d7"),
    "selection_r6": (R6 / "selection_r6.json", "506b601e3b6ad4d79c4056e9efabeb4051e5c9c592a102f29e4db5266be6b308"),
}  # fmt: skip
VARIANT = "B"
STRICT = (0, 16, 19, 20, 22, 25, 28)  # PREREG_R7 4.2
MARGIN, FLOOR_S, FLOOR_F, MIN_FRESH = 0.01, 2, 8, 8  # PREREG_R7 5.3 / 5.4 / 5.5
NTRIALS, N_SCREEN, N_REUSE = 10, 154, 245
EXTRA_FORBIDDEN_R7 = ("t2_zx07_20261028", "best_weights", "seed_best", "remote_reps", "report_r", "composites",
                      "per_episode_zx", "tier2_eval.csv", "tier2_runs.csv", "summary.txt", "tier2_best_scan",
                      "eval_gen_local")  # PREREG_R7 5.2 (round 2's readme / train_rl.log come with its module)  # fmt: skip
PH = {
    "S": {"tag": "s7", "name": "screen", "runs": ROOT / "runs_s", "nruns": 3, "manifest": ROOT / "manifest_s7.json",
          "evals": ROOT / "own_input_evals_s7.csv", "hb": ROOT / "eval_r7_heartbeat_s7.json",
          "rollouts": ROOT / "inputs_r7/ROLLOUTS_S7.sha256", "sc": ROOT / "scores_s7", "lock": SEL_S7},
    "F": {"tag": "f7", "name": "final", "runs": ROOT / "runs_f", "nruns": 10, "manifest": ROOT / "manifest_f7.json",
          "evals": ROOT / "own_input_evals_f7.csv", "hb": ROOT / "eval_r7_heartbeat_f7.json",
          "rollouts": ROOT / "inputs_r7/ROLLOUTS_F7.sha256", "sc": ROOT / "scores_f7", "lock": SEL_R7},
}  # fmt: skip
INC_TABLE = {  # PREREG_R7 4.1: ep -> (incumbent, m_P_frozen to 6 decimals, frozen runs, input, selected in)
    1: ("ep01_k3fix_final44922", 0.171319, 5, "k3fix", "round1"), 2: ("ep02_k3fix_final34093", 0.868138, 5, "k3fix", "round1"),
    3: ("ep03_k3fix_final27068", 0.921527, 5, "k3fix", "round1"), 4: ("ep04_zxg2_421500", 0.416757, 10, "g2", "round6"),
    5: ("ep05_zx_k3fix_88000", 0.578153, 5, "k3fix", "round1"), 6: ("ep06_k3fix_final28420", 0.797276, 5, "k3fix", "round1"),
    7: ("ep07_k3fix_final18529", 0.920452, 5, "k3fix", "round1"), 8: ("ep08_k3easy_final17496", 0.596363, 5, "k3", "round1"),
    9: ("ep09_raw_final39328", 0.760460, 5, "raw", "round1"), 10: ("ep10_raw_final34267", 0.412469, 5, "raw", "round1"),
    11: ("ep11_raw_24000", 0.807319, 10, "raw", "round6"), 12: ("ep12_k3fix_final23178", 0.543911, 5, "k3fix", "round1"),
    13: ("ep13_k3fix_final39254", 0.665483, 5, "k3fix", "round1"), 14: ("ep14_k3fix_final44922", 0.330614, 5, "k3fix", "round1"),
    15: ("ep15_raw_final44922", 0.729895, 5, "raw", "round1"), 17: ("ep17_raw_final31807", 0.647528, 5, "raw", "round1"),
    18: ("ep18_k3fix_final18911", 0.419129, 5, "k3fix", "round1"), 21: ("ep21_k3fix_final44922", 0.444469, 5, "k3fix", "round1"),
    23: ("ep23_raw_final20023", 0.902867, 5, "raw", "round1"), 24: ("ep24_raw_final44922", 0.895137, 5, "raw", "round1"),
    26: ("ep26_zx_k3fix_92000", 0.731484, 5, "k3fix", "round1"), 27: ("ep27_k3fix_final20863", 0.897854, 5, "k3fix", "round1"),
    29: ("ep29_k3fix_final17358", 0.505675, 5, "k3fix", "round1"),
}  # fmt: skip
REF_TABLE = {  # PREREG_R7 4.2: ep -> (report-only round-6 reference, input, export sha256 prefix)
    0: ("ep00_zxb_k3fix_60000", "k3fix", "45d1084dea7f6d0b"), 16: ("ep16_zxb_g2_34000", "g2", "acec8f29690a40c3"),
    19: ("ep19_zxb_g2hard_248000", "g2", "960405ce1f92d6be"), 20: ("ep20_zxb_k3fix_30000", "k3fix", "3b3dbf3cf5522a61"),
    22: ("ep22_zxb_k3fix_78000", "k3fix", "66da2cd65061af36"), 25: ("ep25_zxb_k3fix_48000", "k3fix", "2d92f29dc238e784"),
    28: ("ep28_zx_g2_102000", "g2", "9ce8e92c1ed16618"),
}  # fmt: skip
BARRED = {"ep12_raw_37575": (12, "ep12_k3fix_final23178"), "ep13_raw_44922": (13, "ep13_k3fix_final39254")}  # PREREG_R7 5.4
RULE_S = ("PREREG_R7 5.3 / 5.4 (round 6's 5.4 for the 23 incumbent episodes): a candidate c (screen or reuse) is "
          "S-eligible iff its export is ok, it has 3 ok official runs, none flagged by the own-input gate (5.1), and "
          "own-input SP-SR = MP-SR = 1 in >= 2 of its 3 runs; cons_S(c) = float64 mean of its 3 variant-B consensus AUCs; "
          "ties everywhere: lower population variance of the 3 consensus AUCs, lower mean consensus MPPE, job name. "
          "23 incumbent episodes: finalist = the S-eligible c with the highest cons_S(c) among those with cons_S(c) >= "
          "m_P_frozen(e) + 0.01 (exact floats), except a reused candidate whose round-6 phase-F verdict against the same "
          "incumbent was 'no replacement' (ep12_raw_37575, ep13_raw_44922); otherwise no finalist. 7 STRICT episodes: "
          "finalists = the two highest-ranked S-eligible candidates by cons_S (rank 1, rank 2 = its fallback; one if only "
          "one is S-eligible); if none is S-eligible, the two highest by (floor_S desc, cons_S desc) among candidates with "
          "an ok export and 3 ok, unflagged runs, marked fallback_unconfirmed_S; if none has 3 ok, unflagged runs, the "
          "episode halts.")  # fmt: skip
RULE_F = ("PREREG_R7 5.5 / 5.6: 23 incumbent episodes (round 6's 5.5): m_F(f) = float64 mean of f's 10 phase-F consensus "
          "AUCs; m_P_fresh = float64 mean of the incumbent's phase-F consensus AUCs over its ok, non-flagged runs (>= 8 "
          "required, else +inf); f replaces P iff (a) f has 10 ok, non-flagged phase-F runs, (b) own-input SP-SR = MP-SR "
          "= 1 in >= 8 of them and (c) m_F(f) >= max(m_P_frozen, m_P_fresh) + 0.01 (exact floats); otherwise P stays. "
          "7 STRICT episodes: a finalist is confirmed iff it has 10 ok, non-flagged phase-F runs and own-input SP-SR = "
          "MP-SR = 1 in >= 8 of them; pick = rank 1 if confirmed, else rank 2 if confirmed; if neither: among the "
          "finalists with >= 8 ok, non-flagged runs, more SP=MP=1 runs, then higher m_F, then phase-S rank (marked "
          "unconfirmed); if none has >= 8 ok runs, the episode halts. S7 = per episode the 5.5 result; trial t is "
          "complete for a set iff every episode's job has an ok, non-flagged run t; R7-CLOSEST = the complete S7 trial "
          "whose 30-episode mean own-input AUC is closest to the mean over the complete S7 trials (ties: smaller "
          "distance of its consensus-B mean to the mean of that quantity, then lower t). REF6 (S7 with the 7 STRICT "
          "jobs replaced by their round-6 references) and Delta_B are report only.")  # fmt: skip


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


P2 = _load("t2cons_r2_consensus_pinned_r7", P2_FILE, P2_SHA)  # loads t2cons_a1_gate (a29c997b) -> t2cons_consensus
G, C = P2.G, P2.C
if hashlib.sha256(Path(G.__file__).read_bytes()).hexdigest() != GATE_SHA or \
        hashlib.sha256(Path(C.__file__).read_bytes()).hexdigest() != CONS_SHA:  # fmt: skip
    raise SystemExit("pinned gate / consensus modules differ from a29c997b / e1b943b1")
if (P2.MARGIN, P2.VARIANT) != (0.01, "B") or tuple(G.SEEDS) != tuple(range(10)) or G.BAND != 0.02:
    raise SystemExit("round-2 scorer / gate constants differ from PREREG_R7 5.1 / 5.4")
C.FORBIDDEN = tuple(C.FORBIDDEN) + EXTRA_FORBIDDEN_R7  # PREREG_R7 5.2 (rebinding only; round 2 added readme, train_rl.log)
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
    tmp = p.with_name(p.name + ".tmp_r7c")
    if tmp.exists():
        raise SystemExit(f"stale temporary {tmp}")
    tmp.write_text(text)
    os.replace(tmp, p)
    if mode is not None:
        os.chmod(p, mode)


def pin_path(rel: str) -> Path:
    """PREREG_R7 11 path rules: '/...' absolute; 'scripts/...' and 'tmp_analysis/...' relative to ME; else ROOT."""
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


def r6file(key: str) -> Path:
    p, want = R6_FILES[key]
    if pin_hash(p) != want:
        raise SystemExit(f"{p} != pinned {want[:16]}")
    return p


# ================================================================================================ rules (PREREG_R7 5.3-5.6)
def s_row(c: str, x: dict) -> dict:
    """PREREG_R7 5.3 for one candidate (round 6's eligibility checks, unchanged)."""
    own, cons = x.get("own") or {}, x.get("cons")
    why = []
    if not x.get("export_ok"):
        why.append("export not ok")
    if sorted(own) != [1, 2, 3]:
        why.append(f"{len(own)} of 3 official runs ok")
    if x.get("flagged"):
        why.append("validation_failed (a run flagged by the own-input gate, 5.1)")
    if cons is None or list(cons.get("runs") or []) != [1, 2, 3] or len(cons.get("auc") or []) != 3:
        why.append("not 3 consensus-scored runs")
    fl = C.floor_count(own)
    row = {"job": c, "export_ok": bool(x.get("export_ok")), "n_own_ok": len(own), "own_runs": sorted(own),
           "flagged": bool(x.get("flagged")), "floor_count": fl}  # fmt: skip
    three_ok = not why  # ok export, 3 ok, unflagged, consensus-scored runs (the STRICT fallback pool)
    if fl < FLOOR_S:
        why.append(f"floor {fl}/3 < {FLOOR_S}")
    if cons is not None and cons.get("auc"):
        st = C.stats(cons["auc"])
        row.update(cons_runs=list(cons["runs"]), cons_auc=[float(v) for v in cons["auc"]], cons_mean=st["mean"],
                   cons_var=st["var"], cons_mppe_mean=C.stats(cons["mppe"])["mean"])  # fmt: skip
    row.update(eligible=not why, three_ok_unflagged=three_ok, why_not=list(why))
    return row


def rank_key(r: dict):
    return (-r["cons_mean"], r["cons_var"], r["cons_mppe_mean"], r["job"])


def select_s(m_P: float, cands: list[str], info: dict, barred=()) -> tuple[str | None, list[dict]]:
    """PREREG_R7 5.4 (round 6's 5.4) for one incumbent episode, plus the barred reused candidates."""
    bar = m_P + MARGIN
    rows = []
    for c in sorted(cands):
        row = s_row(c, info[c])
        why = row["why_not"]
        row.update(m_P_frozen=m_P, bar=bar)
        if "cons_mean" in row:
            row["delta_vs_m_P"] = row["cons_mean"] - m_P
        if row["eligible"] and not (row["cons_mean"] >= bar):
            why.append("cons_S < m_P_frozen + 0.01")
        if c in barred:
            why.append("reused round-6 finalist with the phase-F verdict 'no replacement' against the same incumbent "
                       "(PREREG_R7 5.4): not a finalist again")  # fmt: skip
        row.update(barred=c in barred, qualifies=not why, why_not=why)
        rows.append(row)
    q = sorted((r for r in rows if r["qualifies"]), key=rank_key)
    return (q[0]["job"] if q else None), rows


def select_strict(cands: list[str], info: dict) -> tuple[list[str], str, list[dict]]:
    """PREREG_R7 5.4 for one STRICT episode. Returns (finalists in rank order, mode, rows); mode is 'S-eligible',
    'fallback_unconfirmed_S' or 'halt'."""
    rows = [s_row(c, info[c]) for c in sorted(cands)]
    elig = sorted((r for r in rows if r["eligible"]), key=rank_key)
    if elig:
        top, mode = elig[:2], "S-eligible"
    else:
        pool3 = [r for r in rows if r["three_ok_unflagged"]]
        top = sorted(pool3, key=lambda r: (-r["floor_count"], -r["cons_mean"], r["cons_var"], r["cons_mppe_mean"], r["job"]))[:2]
        mode = "fallback_unconfirmed_S" if top else "halt"
    for r in rows:
        r["strict_rank"] = None
        r["qualifies"] = False
    for i, r in enumerate(top):
        r["strict_rank"] = i + 1
        r["qualifies"] = True
    return [r["job"] for r in top], mode, rows


def decide_f(m_P_frozen: float, inc_runs: dict, fin_job: str | None, fin_runs: dict | None) -> dict:
    """PREREG_R7 5.5 (round 6's 5.5, unchanged) for one incumbent episode. *_runs[k] = {"ok": official run ok AND
    consensus-scored, "flagged": bool, "cons_auc": float, "own_sp": float, "own_mp": float}, k = 1..10."""
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


def decide_strict(fins: list[tuple[str, int, dict]]) -> dict:
    """PREREG_R7 5.5 for one STRICT episode. fins = [(job, phase-S rank, runs)], runs[k] as in decide_f."""
    recs = []
    for job, rank, runs in fins:
        good = [k for k in sorted(runs) if runs[k]["ok"] and not runs[k]["flagged"]]
        fl = sum(1 for k in good if runs[k]["own_sp"] == 1.0 and runs[k]["own_mp"] == 1.0)
        m_F = C.stats([runs[k]["cons_auc"] for k in good])["mean"] if good else None
        why = []
        if good != list(range(1, NTRIALS + 1)):
            why.append(f"{len(good)} of 10 ok, non-flagged phase-F runs")
        if fl < FLOOR_F:
            why.append(f"own-input SP-SR = MP-SR = 1 in {fl} < {FLOOR_F} runs")
        recs.append({"job": job, "rank": rank, "good_runs": good, "floor_count": fl, "m_F": m_F, "confirmed": not why,
                     "why_not_confirmed": why})  # fmt: skip
    by_rank = sorted(recs, key=lambda r: r["rank"])
    pick = next((r for r in by_rank if r["confirmed"]), None)
    if pick is not None:
        status = "confirmed"
    else:
        c8 = [r for r in recs if len(r["good_runs"]) >= MIN_FRESH]
        if c8:
            pick = sorted(c8, key=lambda r: (-r["floor_count"], -r["m_F"], r["rank"]))[0]
            status = "unconfirmed"
        else:
            status = "halt"
    return {"finalists": by_rank, "pick": pick["job"] if pick else None, "pick_rank": pick["rank"] if pick else None,
            "status": status, "changed": False}


def trial_rows(comp: dict[int, str], runinfo: dict[str, dict[int, dict]], ntrials: int = NTRIALS) -> list[dict]:
    """PREREG_R7 5.6 (round 6's): one row per trial t; complete iff every episode's job has an ok, non-flagged run t; for
    a complete trial the 30-episode means of the evaluator (own-input) and consensus-B values."""
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
    """R7-CLOSEST over the complete trials: own-input AUC mean closest to its mean; ties: consensus-B distance, lower t."""
    cr = [r for r in rows if r["complete"]]
    if not cr:
        return None
    mu = C.stats([r["own_auc"] for r in cr])["mean"]
    mub = C.stats([r["cons_auc"] for r in cr])["mean"]
    return min(cr, key=lambda r: (abs(r["own_auc"] - mu), abs(r["cons_auc"] - mub), r["trial"]))["trial"]


def ref6_report(s7: list[dict], ref6: list[dict]) -> dict:
    """PREREG_R7 5.6, REPORT ONLY: Delta_B = mean over the trials complete for both of (S7 - REF6 30-episode mean
    consensus-B AUC), and the own-input SP / MP means over those trials."""
    a = {r["trial"]: r for r in s7 if r["complete"]}
    b = {r["trial"]: r for r in ref6 if r["complete"]}
    shared = sorted(set(a) & set(b))
    out = {"report_only": True, "shared_trials": shared, "delta_B": None}
    if shared:
        out["delta_B"] = float(np.mean(np.asarray([a[t]["cons_auc"] - b[t]["cons_auc"] for t in shared], dtype=np.float64)))
        for k in ("own_sp", "own_mp", "own_auc", "own_mppe", "cons_auc", "cons_mppe"):
            out[f"s7_{k}"] = float(np.mean(np.asarray([a[t][k] for t in shared], dtype=np.float64)))
            out[f"ref6_{k}"] = float(np.mean(np.asarray([b[t][k] for t in shared], dtype=np.float64)))
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
        if "t2cons_r7_eval" in cmd and "t2cons_r7_consensus" not in cmd:
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
    got, rec = pin_hash(PREREG), first_token(ROOT / "PREREG_R7.sha256")
    if not got == rec == PREREG_SHA:
        return [f"PREREG_R7.md sha256 {got[:16]} (sha file {rec[:16]}, hard-coded {PREREG_SHA[:16]})"], info
    sec = PREREG.read_text().split("## 11. Pins", 1)[-1]
    n = 0
    for rel, h0 in pin_lines(sec):
        n += 1
        h = pin_hash(pin_path(rel))
        if h != h0:
            fatal.append(f"PREREG_R7 pin {rel}: {h[:16]} != {h0[:16]}")
    for line in sec.splitlines():
        f = line.split()
        if len(f) == 3 and f[0] == "commit":
            n += 1
            head = C.git_head(ME / f[1])
            if head != f[2]:
                fatal.append(f"PREREG_R7 commit {f[1]}: HEAD {head[:12]} != {f[2][:12]}")
    info.append(f"PREREG_R7.md {got[:16]} == PREREG_R7.sha256 == hard-coded; {n} section-11 pins / commits hold")
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
        fatal.append("DRYRUN_R7.md != e7a6095b...")
    if pin_hash(LOCK) != LOCK_SHA:
        fatal.append("LOCK_R7.sha256 != cef6faeb...")
    for listing, base in ((DRYRUN_LIST, "pin"), (LOCK, "pin"), (SNAP, "root"), (TRACKS, "root"), (RELOC_LIST, "root")):
        k = 0
        for rel, h0 in hash_lines(listing):
            k += 1
            p = pin_path(rel) if base == "pin" else ROOT / rel  # ROOT / '/abs' == '/abs' (the driver's rule)
            h = pin_hash(p)
            if h != h0:
                fatal.append(f"{listing.name}: {rel} {h[:16]} != {h0[:16]}")
        info.append(f"{listing.name}: {k} lines hold")
    for p, want in ((POOL, POOL_SHA), (INCS, INCS_SHA), (REFS6, REFS6_SHA)):
        if pin_hash(p) != want:
            fatal.append(f"{p.name} != {want[:16]}")
    for key, (p, want) in R6_FILES.items():
        if pin_hash(p) != want:
            fatal.append(f"round-6 record {key} {p} != {want[:16]}")
    info.append(f"pool / incumbents / references == pinned; {len(R6_FILES)} round-6 records == pinned")
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
        if not SEL_S7.is_file():
            fatal.append("SELECTION_S7.sha256 missing (phase F needs the frozen phase-S selection)")
        else:
            listed = False
            for rel, h0 in hash_lines(SEL_S7):
                listed |= pin_path(rel) == FINAL
                if pin_hash(pin_path(rel)) != h0:
                    fatal.append(f"SELECTION_S7.sha256: {rel} changed")
            if not listed:
                fatal.append("SELECTION_S7.sha256 does not list final_r7.json")
            info.append(f"SELECTION_S7.sha256 {raw_sha(SEL_S7)[:16]} holds; lists final_r7.json: {listed}")
    mine = raw_sha(MINE)
    info.append(f"scripts/t2cons_r7_consensus.py sha256 {mine}")
    if stage in ("gate", "score", "select"):
        if not PINS_B.is_file():
            fatal.append("PREREG_R7_PINS_b.md missing: this script must be pinned before it scores anything (5.2)")
        else:
            got, rec = raw_sha(PINS_B), first_token(PINS_B_SHA)
            pins = pin_lines(PINS_B.read_text())
            if got != rec:
                fatal.append(f"PREREG_R7_PINS_b.md {got[:16]} != PREREG_R7_PINS_b.sha256 {rec[:16]}")
            for need in ("scripts/t2cons_r7_consensus.py", "selftest_r7/selftest_r7.json"):
                if need not in [r for r, _ in pins]:
                    fatal.append(f"PREREG_R7_PINS_b.md does not pin {need}")
            if ("scripts/t2cons_r7_consensus.py", mine) not in pins:
                fatal.append(f"PREREG_R7_PINS_b.md does not pin this script's sha256 {mine[:16]}")
            for rel, h0 in pins:
                if pin_hash(pin_path(rel)) != h0:
                    fatal.append(f"PREREG_R7_PINS_b pin {rel} changed")
            info.append(f"PREREG_R7_PINS_b.md {got[:16]} == its sha file; {len(pins)} pins hold; pins this script")
            st = C.read_json(SELFTEST) if SELFTEST.is_file() else {}
            if st.get("pass") is not True or st.get("script_sha256") != mine:
                fatal.append("selftest_r7/selftest_r7.json is not a PASS of this script")
        if not P["rollouts"].is_file():
            fatal.append(f"{P['rollouts'].name} missing: run check --phase {ph} first (5.1)")
    if stage in ("score", "select"):
        F = files_of(ph)
        g = C.read_json(F["gate_json"]) if F["gate_json"].is_file() else {}
        if g.get("pass") is not True or g.get("script_sha256") != mine or g.get("csv_sha256") != (
                raw_sha(F["gate_csv"]) if F["gate_csv"].is_file() else None) or g.get("pins_b_sha256") != (
                raw_sha(PINS_B) if PINS_B.is_file() else None):  # fmt: skip
            fatal.append(f"{F['gate_json'].name} is not a PASS of this pinned script")
        else:
            info.append(f"own-input gate (5.1) phase {ph}: PASS; validation_failed {len(g.get('validation_failed_jobs') or [])} "
                        f"jobs; flagged runs {g.get('n_flagged_runs')}")  # fmt: skip
    if stage == "select":
        js = C.read_json(files_of(ph)["jstats"]) if files_of(ph)["jstats"].is_file() else {}
        if ph == "S" and (js.get("reuse_reproduction") or {}).get("pass") is not True:
            fatal.append("the reused round-6 runs do not reproduce round 6's scores_s6 exactly (PREREG_R7 3.2)")
    if stage in ("gate", "score", "select") and P["lock"].exists():
        fatal.append(f"{P['lock'].name} exists: phase {ph} is frozen")
    if stage in ("gate", "score", "select") and ph == "S" and SEL_R7.exists():
        fatal.append("SELECTION_R7.sha256 exists")
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
_CSV_CACHE: dict[str, object] = {}
JOB_STATS_PINS = {  # the frozen job stats of the rounds that selected the incumbents (PREREG_R7 11 pins)
    str(R1 / "scores/consensus_job_stats.json"): "9e255057556ccc23df9c97dafd101219770e8b53cc6ab28519ab893eb86319c8",
    str(R6 / "scores_f6/consensus_job_stats_f6.json"): "9eaac6077960d06975597a385552b459df5ff2d76e6c7e39bd9b7243ee3f24d7",
}  # fmt: skip
FROZEN_CSV_PINS = {
    str(R1 / "scores/consensus_scores.csv"): "50cc6e9401ad8c88594918490cee345d8aaf7368740fb3626c65525bfcb40a52",
    str(R6 / "scores_f6/consensus_scores_f6.csv"): "74c9c8856d5f2fd671b1e3f97763a03c26936be794c581581972c2d837e84ab4",
}  # fmt: skip


def load_incumbents() -> dict[int, dict]:
    """PREREG_R7 4.1 / K5: the 23 incumbents with their frozen variant-B consensus runs and m_P_frozen."""
    if pin_hash(INCS) != INCS_SHA:
        raise SystemExit("incumbents_r7.json != cb55fa7a")
    data = C.read_json(INCS)
    incs = {int(x["episode"]): x for x in data["incumbents"].values()}
    if sorted(incs) != sorted(INC_TABLE) or sorted(data["incumbents"]) != [ep_key(e) for e in sorted(INC_TABLE)]:
        raise SystemExit("incumbents_r7.json does not hold the 23 non-STRICT episodes")
    sel6 = None
    for e, x in incs.items():
        job, m6, n, kind, rnd = INC_TABLE[e]
        fr = sorted(x["frozen_runs"], key=lambda r: int(r["run"]))
        vals = [float(r["cons_auc"]) for r in fr]
        m = C.stats(vals)["mean"]  # the pinned float64 mean (np.mean), as rounds 1-6 computed m_P
        if x["job"] != job or int(x["episode"]) != e or x["input"] != kind or len(vals) != n or x.get("selected_in") != rnd \
                or [int(r["run"]) for r in fr] != list(range(1, n + 1)) or x.get("role") != "incumbent":  # fmt: skip
            raise SystemExit(f"ep{e:02d}: incumbents_r7.json row != PREREG_R7 4.1 table ({x['job']}, {x['input']}, {len(vals)})")
        if abs(m - x["m_P_frozen"]) > 1e-12 or abs(m - m6) > 5e-7:
            raise SystemExit(f"ep{e:02d}: m_P_frozen field {x['m_P_frozen']!r} / table {m6} != mean {m!r}")
        f = x["frozen_scores_csv"]
        if f not in _CSV_CACHE:
            if FROZEN_CSV_PINS.get(f) is None or sha(f) != FROZEN_CSV_PINS[f] or x["frozen_scores_csv_sha256"] != FROZEN_CSV_PINS[f]:
                raise SystemExit(f"{f}: not a pinned frozen scores CSV")
            _CSV_CACHE[f] = C.read_csv(f)
        rows = sorted([r for r in _CSV_CACHE[f] if r["job"] == job and r["variant"] == VARIANT and r["status"] == "ok"],
                      key=lambda r: int(r["run"]))  # fmt: skip
        if [float(r["auc"]) for r in rows] != vals or [int(r["run"]) for r in rows] != list(range(1, n + 1)):
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
        if rnd == "round6":  # PREREG_R7 4.1: the round-6 replacements' frozen mean == selection_r6.json m_F
            if sel6 is None:
                sel6 = C.read_json(r6file("selection_r6"))
            e6 = next(y for y in sel6["episodes"] if int(y["episode"]) == e)
            if e6.get("pick") != job or e6.get("changed") is not True or float(e6["m_F"]) != m or \
                    sel6["compositions"]["S6"][ep_key(e)] != job:  # fmt: skip
                raise SystemExit(f"ep{e:02d}: selection_r6.json m_F / pick != the frozen runs' mean / {job}")
        elif x.get("record_from", "").find("incumbents_r6.json") < 0:
            raise SystemExit(f"ep{e:02d}: a round-1 incumbent not carried from incumbents_r6.json")
        x["m_P_frozen_field_resolver"] = x["m_P_frozen"]
        x["m_P_frozen"] = m
    return incs


def load_refs6() -> dict[int, dict]:
    """PREREG_R7 4.2: the 7 report-only round-6 references (no decision role)."""
    if pin_hash(REFS6) != REFS6_SHA:
        raise SystemExit("r6refs_r7.json != 0bc58249")
    data = C.read_json(REFS6)
    refs = {int(x["episode"]): x for x in data["r6refs"].values()}
    if sorted(refs) != list(STRICT):
        raise SystemExit("r6refs_r7.json does not hold the 7 STRICT episodes")
    sel6 = C.read_json(r6file("selection_r6"))
    for e, x in refs.items():
        job, kind, pre = REF_TABLE[e]
        if x["job"] != job or x["input"] != kind or not x["export_sha256"].startswith(pre) or \
                sel6["compositions"]["S6"][ep_key(e)] != job:  # fmt: skip
            raise SystemExit(f"ep{e:02d}: r6refs_r7.json row != PREREG_R7 4.2 table / round 6's S6")
    return refs


def input_ref(kind: str, e: int, trk: dict, r1trk: dict) -> tuple[Path, Path, str | None, str]:
    ref = C.INPUT_REF[kind]
    f = C.ep_file(ref, e)
    rec = (r1trk if kind == "raw" else trk).get(str(f))
    return ref, f, rec, (sha(f) if f.is_file() else "missing")


def relocation_rows() -> dict[str, dict]:
    with C.guard(RELOC_TSV).open(newline="") as fp:
        return {r["job"]: r for r in csv.DictReader(fp, delimiter="\t")}


def pool_rows() -> dict[str, dict]:
    if pin_hash(POOL) != POOL_SHA:
        raise SystemExit("pool_r7.json != 7c4b553b")
    return {r["job"]: r for r in C.read_json(POOL)["rows"]}


def barred_of(pool: dict[str, dict], e: int, inc_job: str) -> set[str]:
    """PREREG_R7 5.4: the reused candidates of episode e whose round-6 phase-F verdict against the same incumbent was
    'no replacement' (recorded in pool_r7.json); must equal the PREREG_R7 list."""
    out = set()
    for j, p in pool.items():
        v = p.get("r6_phase_f_verdict")
        if p.get("role") == "reuse" and int(p["episode"]) == e and v and v.get("r6_incumbent") == inc_job \
                and v.get("verdict") == "did not replace the incumbent":  # fmt: skip
            out.add(j)
    want = {j for j, (ee, inc) in BARRED.items() if ee == e and inc == inc_job}
    if out != want:
        raise SystemExit(f"ep{e:02d}: barred candidates {sorted(out)} != PREREG_R7 5.4 {sorted(want)}")
    return out


def _export_from_record(rec_path: Path, rec_sha: str | None, want_sha: str | None, src_sha: str, base: Path | None,
                        job: str) -> tuple[str, str | None, str | None]:
    """An earlier / round-7 export record: (status, export sha256, file); raises on inconsistency. Phase-S style records
    (base given: this round's or round 6's exports/<job>/) must carry destination_sha256 == sha256 (round 6's check)."""
    if rec_sha is not None and sha(rec_path) != rec_sha:
        raise SystemExit(f"{job}: export record {rec_path} changed")
    rec = C.read_json(rec_path)
    if rec.get("status") != "ok":
        return str(rec.get("status")), None, None
    want = rec.get("sha256")
    if rec.get("stock_verified") is not True or (base is not None and rec.get("destination_sha256") != want) or \
            rec.get("source_sha256") != src_sha or (base is not None and Path(rec["file"]).parent.parent != base) or \
            sha(rec["file"]) != want or (want_sha is not None and want != want_sha):  # fmt: skip
        raise SystemExit(f"{job}: export record / file inconsistent ({rec_path})")
    return "ok", want, rec["file"]


def load_jobs(ph: str) -> dict[str, dict]:
    """Phase S: the 154 screen rows of manifest_s7.json (this round's exports) + the 245 reused round-6 candidates
    (round 6's manifest_s6 row, export record and runs). Phase F: the 23 incumbents + the finalists of final_r7.json +
    the 7 report-only references of manifest_f7.json."""
    P = PH[ph]
    man = C.read_json(P["manifest"])
    if man.get("prereg_r7_sha256") != PREREG_SHA or man.get("phase") != P["name"] or man.get("pool_r7_sha256") != POOL_SHA \
            or man.get("incumbents_r7_sha256") != INCS_SHA or man.get("r6refs_r7_sha256") != REFS6_SHA:  # fmt: skip
        raise SystemExit(f"{P['manifest'].name} was not built under PREREG_R7 / pool_r7 / incumbents_r7 / r6refs_r7")
    pool = pool_rows()
    reloc = relocation_rows()
    incs = load_incumbents()
    refs = load_refs6()
    inc_by = {x["job"]: x for x in incs.values()}
    ref_by = {x["job"]: x for x in refs.values()}
    trk, r1trk = C.hash_list(TRACKS), C.hash_list(R1_TRACKS)
    fin_by = {}
    if ph == "F":
        if man.get("final_r7_sha256") != raw_sha(FINAL) or man.get("selection_s7_lock_sha256") != raw_sha(SEL_S7):
            raise SystemExit("manifest_f7.json was not built from the frozen final_r7.json / SELECTION_S7.sha256")
        fin_by = {x["job"]: x for x in C.read_json(FINAL)["finalists"]}

    def screen_tsha(p: dict) -> str:
        if p.get("machine") == "5090":
            rr = reloc.get(p["job"])
            if rr is None or rr.get("status") != "ok" or rr["original_sha256"] != p["sha256"]:
                raise SystemExit(f"{p['job']}: no ok RELOCATION_R7.tsv row")
            return rr["relocated_sha256"]
        return p["sha256"]

    jobs: dict[str, dict] = {}
    man_rows = {r["job"]: r for r in man["rows"]}
    if len(man_rows) != len(man["rows"]):
        raise SystemExit(f"{P['manifest'].name}: duplicate job rows")
    plan: list[tuple[str, str, dict]] = []  # (job, role, source record)
    if ph == "S":
        for j, p in pool.items():
            if p.get("role") in ("screen", "reuse"):
                plan.append((j, p["role"], p))
        if sorted(man_rows) != sorted(j for j, role, _ in plan if role == "screen"):
            raise SystemExit("manifest_s7.json does not hold exactly the pool's screen jobs")
    else:
        for r in man["rows"]:
            role = r.get("role")
            if role == "incumbent":
                plan.append((r["job"], role, inc_by.get(r["job"]) or {}))
            elif role == "r6ref":
                plan.append((r["job"], role, ref_by.get(r["job"]) or {}))
            elif role == "finalist":
                plan.append((r["job"], role, pool.get(r["job"]) or {}))
            else:
                raise SystemExit(f"{r['job']}: unexpected role {role}")
    r6man = None
    for job, role, src in plan:
        if not src:
            raise SystemExit(f"{job}: no source record for role {role}")
        e = int(src["episode"])
        prole = src.get("role")  # pool role for screen / reuse / finalist rows
        if role in ("screen", "reuse"):
            kind, owner = src["input"], src["owner"]
            tsha = screen_tsha(src) if role == "screen" else src["sha256"]
            src_round = "r7" if role == "screen" else "r6"
        elif role in ("incumbent", "r6ref"):
            kind, owner, tsha, src_round = src["input"], src["owner"], src["training_sha256"], "r7"
        else:  # finalist
            if job not in fin_by or prole not in ("screen", "reuse") or int(fin_by[job]["episode"]) != e:
                raise SystemExit(f"{job}: manifest_f7 finalist row is not in final_r7.json / the pool")
            kind, owner = src["input"], src["owner"]
            tsha = screen_tsha(src) if prole == "screen" else src["sha256"]
            src_round = "r7"
        d = {"job": job, "episode": e, "role": role, "id": role, "pool_role": prole, "input": kind, "owner": owner,
             "machine": src.get("machine"), "tree": src.get("tree"), "iteration": src.get("iteration"),
             "run_dir": src.get("run_dir"), "run_label": src.get("run_label"), "slot": src.get("slot"),
             "training_checkpoint": src.get("training_checkpoint"), "source_checkpoint": src.get("source_checkpoint"),
             "selected_in": src.get("selected_in") or src.get("source_round"),
             "pool_sha256": src.get("sha256"), "training_sha256": tsha, "parquet_sha256": src.get("parquet_sha256"),
             "strict_episode": e in STRICT, "source_round": src_round, "export_status": None,
             "runs_base": str(P["runs"] if src_round == "r7" else R6_RUNS_S),
             "evals_csv": str(P["evals"] if src_round == "r7" else R6_FILES["evals_s6"][0])}  # fmt: skip
        jobs[job] = d
        if role == "reuse":  # round 6's manifest row, export record and run records (PREREG_R7 2.1 R5)
            if r6man is None:
                r6man = {r["job"]: r for r in C.read_json(r6file("manifest_s6"))["rows"]}
            r = r6man.get(src.get("r6_job"))
            if src.get("r6_job") != job or r is None or r.get("role") != "screen":
                raise SystemExit(f"{job}: not a round-6 manifest_s6 screen row")
            d["r6_runs"] = {int(x["run"]): x["run_json_sha256"] for x in src.get("r6_runs") or []}
            if r.get("status") != "ok":
                d["export_status"] = f"job not built in round 6: {r.get('status')}"
                continue
        else:
            r = man_rows.get(job)
            if r is None:
                raise SystemExit(f"{job}: missing from {P['manifest'].name}")
            if r.get("status") != "ok":
                d["export_status"] = f"job not built: {r.get('status')}"
                continue
            if (ph == "S" and r.get("role") != "screen") or (ph == "F" and r.get("role") != role):
                raise SystemExit(f"{job}: manifest role {r.get('role')} != {role}")
            if role in ("screen", "finalist") and r.get("pool_sha256") != src["sha256"]:
                raise SystemExit(f"{job}: manifest pool_sha256 != pool_r7.json sha256")
        if r.get("root_kind") != kind or r.get("training_sha256") != tsha or r.get("parquet_sha256") != src["parquet_sha256"]:
            raise SystemExit(f"{job}: manifest row (input / checkpoint / parquet) != its pool / incumbent / reference record")
        if role in ("incumbent", "r6ref"):
            st, want, f = _export_from_record(Path(src["export_record"]), src["export_record_sha256"], src["export_sha256"],
                                              tsha, None, job)
            if st != "ok" or Path(f) != Path(src["export_file"]):
                raise SystemExit(f"{job}: earlier export record / file != {role} record")
        elif role == "reuse" or (role == "finalist" and prole == "reuse"):
            rp = Path(src["r6_export_record"])
            if rp != R6_EXPORTS / job / "export.json":
                raise SystemExit(f"{job}: round-6 export record path {rp}")
            if src.get("r6_export_status") != "ok":
                d["export_status"] = f"round-6 export {src.get('r6_export_status')}"
                continue
            st, want, f = _export_from_record(rp, src["r6_export_record_sha256"], src["r6_export_sha256"], tsha,
                                              R6_EXPORTS / job, job)
        else:  # a new screen job (phase S) or a new-screen finalist (phase F): this round's single export
            p = EXPORTS / job / "export.json"
            if not p.is_file():
                d["export_status"] = "missing"
                continue
            st, want, f = _export_from_record(p, None, None, tsha, EXPORTS / job, job)
        if st != "ok":
            d["export_status"] = st
            continue
        if role == "finalist" and fin_by[job].get("export_sha256") != want:
            raise SystemExit(f"{job}: export != final_r7.json")
        d.update(export_status="ok", export_sha256=want, export_file=f)
        ref, fref, rec_h, h = input_ref(kind, e, trk, r1trk)
        if rec_h != h or (kind != "raw" and src.get("frame_ref_sha256") != h):
            raise SystemExit(f"{job}: input reference {fref} sha256 {h[:16]} != its frozen record")
        d.update(input_reference=str(ref), input_reference_file=str(fref), input_reference_sha256=h)
    if ph == "S":
        n_s = sum(1 for j in jobs.values() if j["role"] == "screen")
        n_r = sum(1 for j in jobs.values() if j["role"] == "reuse")
        if (n_s, n_r) != (N_SCREEN, N_REUSE):
            raise SystemExit(f"phase S jobs {n_s} screen + {n_r} reuse != {N_SCREEN} + {N_REUSE}")
    else:
        if sorted(j for j in jobs if jobs[j]["role"] == "incumbent") != sorted(inc_by) or \
                sorted(j for j in jobs if jobs[j]["role"] == "r6ref") != sorted(ref_by) or \
                sorted(j for j in jobs if jobs[j]["role"] == "finalist") != sorted(fin_by):  # fmt: skip
            raise SystemExit("manifest_f7.json != the 23 incumbents + final_r7.json finalists + 7 references")
        if any(jobs[j]["export_status"] != "ok" for j in jobs):
            raise SystemExit("a phase-F job is not ok")
    return jobs


def load_evals(ph: str, jobs: dict) -> tuple[dict[str, dict[int, dict]], int, list[str]]:
    """job -> run -> evaluator values (run.json, full precision; == the driver's CSV row) + rollout parquet. Phase S
    reads this round's CSV for the new screen jobs and round 6's CSV for the reused candidates (K4)."""
    P = PH[ph]
    out: dict[str, dict[int, dict]] = {}
    n_not_ok, problems = 0, []
    sources = [(P["evals"], "r7")]
    if ph == "S":
        sources.append((r6file("evals_s6"), "r6"))
    seen = set()
    for csvp, rnd in sources:
        for r in C.read_csv(csvp):
            j = jobs.get(r["job"])
            if rnd == "r6" and (j is None or j["source_round"] != "r6"):
                continue  # round-6 rows of candidates that round 7 does not reuse
            if r.get("status") != "ok":
                n_not_ok += 1
                continue
            if j is None or j["export_status"] != "ok" or j["source_round"] != rnd:
                raise SystemExit(f"{Path(csvp).name}: ok row of unknown / non-ok job {r['job']}")
            k, t, e = int(r["run"]), int(r["try"]), j["episode"]
            if int(r["episode"]) != e or not 1 <= k <= P["nruns"]:
                raise SystemExit(f"{r['job']} run {k}: episode / run out of range")
            base = Path(j["runs_base"])
            pq = base / r["job"] / f"run{k}" / f"try{t}" / "results" / f"episode_{e:06d}.parquet"
            if Path(r["results_parquet"]) != pq or r.get("export_sha256_full") != j["export_sha256"] or \
                    r.get("reference_parquet_sha256") != j["parquet_sha256"] or r.get("variants") != "B" or \
                    r.get("input") != j["input"]:  # fmt: skip
                raise SystemExit(f"{r['job']} run {k}: CSV row (parquet / export / reference / input) inconsistent")
            rjp = base / r["job"] / f"run{k}" / "run.json"
            if rnd == "r6" and sha(rjp) != j["r6_runs"].get(k):
                raise SystemExit(f"{r['job']} run {k}: round-6 run.json sha256 != pool_r7.json r6_runs record")
            rj = C.read_json(rjp)
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
            out[r["job"]][k] = {"run": k, "try": t, "parquet": str(pq), "gpu": r.get("gpu"), "source_round": rnd,
                                "seed_signature": r.get("seed_signature"), **m}  # fmt: skip
    for job, j in jobs.items():  # every ok run.json has its CSV row (the driver's final summary)
        if j["export_status"] != "ok":
            continue
        for k in range(1, P["nruns"] + 1):
            p = Path(j["runs_base"]) / job / f"run{k}" / "run.json"
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


def r6_rollout_check(evals: dict) -> list[str]:
    """K4: every reused round-6 rollout == round 6's frozen ROLLOUTS_S6.sha256 entry."""
    r6 = C.hash_list(r6file("rollouts_s6"))
    bad = []
    for j in sorted(evals):
        for k, r in sorted(evals[j].items()):
            if r["source_round"] == "r6" and (r6.get(r["parquet"]) is None or sha(r["parquet"]) != r6[r["parquet"]]):
                bad.append(f"{j} run {k}: rollout != round 6's ROLLOUTS_S6.sha256")
    return bad


# ================================================================================================ selftest
def cmd_selftest(a) -> int:
    out_path = SELFTEST if a.trial is None else ROOT / "dryrun" / f"selftest_trial_r7c_{int(a.trial)}.json"
    if out_path.exists():
        print(f"FATAL: {out_path} exists (new files only)")
        return 2
    rng = np.random.default_rng(20261102)
    res, problems = [], []

    def check(name, ok, detail=""):
        res.append({"test": name, "ok": bool(ok), "detail": str(detail)[:400]})
        if not ok:
            problems.append(f"{name}: {detail}")

    check("pinned modules: round 2's scorer (ddee0b8b) -> gate (a29c997b) -> consensus (e1b943b1); run_flags / SCORE_COLS "
          "/ pin_lines are round 2's objects; gate seeds 0..9, band 0.02",
          P2.run_flags is run_flags and P2.G is G and G.C is C and tuple(G.SEEDS) == tuple(range(10)) and G.BAND == 0.02
          and hashlib.sha256(Path(C.__file__).read_bytes()).hexdigest() == CONS_SHA, "")  # fmt: skip
    check("constants: margin 0.01, floors 2/3 and 8/10, >= 8 fresh / ok runs, 3 / 10 runs, STRICT = 00 16 19 20 22 25 28, "
          "23 incumbents + 7 references = 30 episodes",
          (MARGIN, FLOOR_S, FLOOR_F, MIN_FRESH, PH["S"]["nruns"], PH["F"]["nruns"]) == (0.01, 2, 8, 8, 3, 10)
          and STRICT == (0, 16, 19, 20, 22, 25, 28) and sorted(list(INC_TABLE) + list(REF_TABLE)) == list(range(30))
          and not set(INC_TABLE) & set(STRICT) and sorted(REF_TABLE) == list(STRICT))  # fmt: skip
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

    # ---- 5.4, incumbent episodes (round 6's tests)
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
    bi = {"best": cand([0.9] * 3), "second": cand([0.8] * 3), "low": cand([0.505] * 3)}
    f, rows = select_s(0.5, sorted(bi), bi, barred={"best"})
    rb = next(r for r in rows if r["job"] == "best")
    check("5.4 (PREREG_R7): a barred reused candidate stays S-eligible but is never the finalist; the next qualifying wins",
          f == "second" and rb["eligible"] and not rb["qualifies"] and rb["barred"], rows)  # fmt: skip
    f, _ = select_s(0.5, ["best"], {"best": bi["best"]}, barred={"best"})
    check("5.4 (PREREG_R7): only a barred candidate qualifies by value -> no finalist", f is None)
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
        barred_t = {j for j in inf_t if rng.random() < 0.1}
        got, _ = select_s(mPt, list(inf_t), inf_t, barred=barred_t)
        q = []
        for j, x in inf_t.items():
            if j in barred_t or not x["export_ok"] or x["flagged"] or len(x["own"]) != 3 or len(x["cons"]["auc"]) != 3:
                continue
            if sum(1 for r in x["own"].values() if r["sp_sr"] == 1.0 and r["mp_sr"] == 1.0) < 2:
                continue
            arr = np.asarray(x["cons"]["auc"], dtype=np.float64)
            if float(arr.mean()) >= mPt + 0.01:
                q.append((-float(arr.mean()), float(arr.var()), float(np.mean(np.asarray(x["cons"]["mppe"], dtype=np.float64))), j))
        want = sorted(q)[0][3] if q else None
        if got != want:
            bad.append(f"case {t}: {got} != {want}")
    check("5.4: select_s == an independent re-derivation on 800 random episodes (with barred candidates)", not bad, bad[:3])

    # ---- 5.4, STRICT episodes
    st = {"a": cand([0.9] * 3), "b": cand([0.8] * 3, own=own_fl2), "c": cand([0.85] * 3), "d": cand([0.99] * 3, own=own_fl1),
          "e": cand([0.995] * 3, flagged=True)}  # fmt: skip
    fins, mode, rows = select_strict(sorted(st), st)
    check("5.4 STRICT: the top 2 S-eligible by cons_S (floor 1/3 and flagged out); rank 1 = best, rank 2 = next",
          fins == ["a", "c"] and mode == "S-eligible" and {r["job"]: r["strict_rank"] for r in rows}["a"] == 1, (fins, mode))  # fmt: skip
    fins, mode, _ = select_strict(["a", "d"], {"a": st["a"], "d": st["d"]})
    check("5.4 STRICT: one S-eligible candidate -> one finalist", fins == ["a"] and mode == "S-eligible", (fins, mode))
    fb = {"p": cand([0.9] * 3, own=own_fl1), "q": cand([0.7] * 3, own={k: {"sp_sr": 0.0, "mp_sr": 0.0} for k in (1, 2, 3)}),
          "r": cand([0.95] * 3, own={1: ok3[1], 2: {"sp_sr": 0.0, "mp_sr": 0.0}, 3: {"sp_sr": 0.0, "mp_sr": 0.0}}),
          "s": cand([0.99] * 3, flagged=True), "u": cand([0.98] * 2, own={1: ok3[1], 2: ok3[2]})}  # fmt: skip
    fins, mode, _ = select_strict(sorted(fb), fb)
    check("5.4 STRICT fallback: none S-eligible -> top 2 by (floor desc, cons_S desc) among 3 ok, unflagged runs (r: "
          "floor 1, 0.95; p: floor 1, 0.9; q: floor 0), marked fallback_unconfirmed_S",
          fins == ["r", "p"] and mode == "fallback_unconfirmed_S", (fins, mode))  # fmt: skip
    fins, mode, _ = select_strict(["s", "u"], {"s": fb["s"], "u": fb["u"]})
    check("5.4 STRICT: no candidate with 3 ok, unflagged runs -> halt", fins == [] and mode == "halt", (fins, mode))
    ties = {"x1": cand([0.6] * 3, mppe=2.0), "x2": cand([0.6] * 3, mppe=1.0), "x3": cand([0.5, 0.7, 0.6]), "x4": cand([0.6] * 3, mppe=1.0)}
    fins, _, _ = select_strict(sorted(ties), ties)
    check("5.4 STRICT ties: variance, MPPE, job name", fins == ["x2", "x4"], fins)
    bad = []
    for t in range(800):
        inf_t = {}
        for i in range(int(rng.integers(0, 9))):
            n = int(rng.choice([2, 3, 3, 3, 3]))
            aucs = list(np.clip(rng.uniform(0, 1) + rng.normal(0, 0.01, n), 0, 1))
            own = {k: {"sp_sr": 1.0 if rng.random() > 0.4 else 0.0, "mp_sr": 1.0 if rng.random() > 0.3 else 0.0}
                   for k in range(1, (3 if rng.random() > 0.05 else 2) + 1)}  # fmt: skip
            inf_t[f"s{t}_{i}"] = {"export_ok": rng.random() > 0.03, "own": own, "flagged": rng.random() < 0.08,
                                  "cons": {"runs": list(range(1, n + 1)), "auc": aucs, "mppe": list(rng.uniform(0.5, 20, n))}}  # fmt: skip
        got, gmode, _ = select_strict(list(inf_t), inf_t)
        base3 = []
        for j, x in inf_t.items():
            if not x["export_ok"] or x["flagged"] or len(x["own"]) != 3 or len(x["cons"]["auc"]) != 3:
                continue
            arr = np.asarray(x["cons"]["auc"], dtype=np.float64)
            fl = sum(1 for r in x["own"].values() if r["sp_sr"] == 1.0 and r["mp_sr"] == 1.0)
            base3.append((fl, float(arr.mean()), float(arr.var()), float(np.mean(np.asarray(x["cons"]["mppe"], dtype=np.float64))), j))
        el = sorted([(-m, v, mp, j) for fl, m, v, mp, j in base3 if fl >= 2])
        if el:
            want, wmode = [x[3] for x in el[:2]], "S-eligible"
        else:
            fbk = sorted([(-fl, -m, v, mp, j) for fl, m, v, mp, j in base3])
            want, wmode = [x[4] for x in fbk[:2]], ("fallback_unconfirmed_S" if fbk else "halt")
        if (got, gmode) != (want, wmode):
            bad.append(f"case {t}: {got} {gmode} != {want} {wmode}")
    check("5.4 STRICT: select_strict == an independent re-derivation on 800 random episodes", not bad, bad[:3])

    # ---- 5.5, incumbent episodes (round 6's tests)
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
    check("5.5: fresh incumbent mean above frozen raises the bar (0.58 >= 0.56 replaces; 0.555 does not)",
          d["changed"] and not decide_f(0.5, fr([0.55] * 10), "f", fr([0.555] * 10))["changed"], d)  # fmt: skip
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

    # ---- 5.5, STRICT episodes
    d = decide_strict([("r1", 1, fr([0.5] * 10)), ("r2", 2, fr([0.9] * 10))])
    check("5.5 STRICT: rank 1 confirmed -> rank 1 (whatever rank 2's m_F)", d["pick"] == "r1" and d["status"] == "confirmed", d)
    d = decide_strict([("r1", 1, fr([0.9] * 10, sp=[1.0] * 7 + [0.0] * 3)), ("r2", 2, fr([0.5] * 10, mp=[1.0] * 8 + [0.0] * 2))])
    check("5.5 STRICT: rank 1 floor 7/10 -> rank 2 if confirmed (floor 8/10)", d["pick"] == "r2" and d["status"] == "confirmed", d)
    d = decide_strict([("r1", 1, fr([0.9] * 10, missing=(3,))), ("r2", 2, fr([0.5] * 10))])
    check("5.5 STRICT: rank 1 with 9 runs is not confirmed -> rank 2", d["pick"] == "r2" and d["status"] == "confirmed", d)
    d = decide_strict([("r1", 1, fr([0.9] * 10, flagged=(3,))), ("r2", 2, fr([0.5] * 10, sp=[0.0] * 10))])
    check("5.5 STRICT: neither confirmed -> more SP=MP=1 runs wins (rank 1: 9 good runs, floor 9 vs 0)",
          d["pick"] == "r1" and d["status"] == "unconfirmed", d)  # fmt: skip
    d = decide_strict([("r1", 1, fr([0.5] * 10, sp=[1.0] * 6 + [0.0] * 4)), ("r2", 2, fr([0.9] * 10, sp=[1.0] * 6 + [0.0] * 4))])
    check("5.5 STRICT: neither confirmed, equal floors -> higher m_F", d["pick"] == "r2" and d["status"] == "unconfirmed", d)
    d = decide_strict([("r1", 1, fr([0.5] * 10, sp=[1.0] * 6 + [0.0] * 4)), ("r2", 2, fr([0.5] * 10, sp=[1.0] * 6 + [0.0] * 4))])
    check("5.5 STRICT: neither confirmed, equal floors and m_F -> phase-S rank", d["pick"] == "r1" and d["status"] == "unconfirmed", d)
    d = decide_strict([("r1", 1, fr([0.9] * 10, missing=(1, 2, 3))), ("r2", 2, fr([0.9] * 10, sp=[0.0] * 10))])
    check("5.5 STRICT: a finalist with 7 ok runs is out of the unconfirmed choice", d["pick"] == "r2" and d["status"] == "unconfirmed", d)
    d = decide_strict([("r1", 1, fr([0.9] * 10, missing=(1, 2, 3))), ("r2", 2, fr([0.9] * 10, flagged=(1, 2, 3)))])
    check("5.5 STRICT: neither has >= 8 ok, non-flagged runs -> halt", d["pick"] is None and d["status"] == "halt", d)
    d = decide_strict([("r1", 1, fr([0.9] * 10))])
    check("5.5 STRICT: a single finalist, confirmed", d["pick"] == "r1" and d["status"] == "confirmed", d)

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
    check("5.6: R7-CLOSEST = the complete trial closest to the own-input mean (independent re-derivation)", t_star == want,
          f"{t_star} {want}")  # fmt: skip
    rows_t = [{"trial": 1, "complete": True, "own_auc": 0.25, "cons_auc": 0.625}, {"trial": 2, "complete": True, "own_auc": 0.75, "cons_auc": 0.5},
              {"trial": 3, "complete": True, "own_auc": 0.25, "cons_auc": 0.375}, {"trial": 4, "complete": True, "own_auc": 0.75, "cons_auc": 0.5},
              {"trial": 5, "complete": False}]  # fmt: skip
    check("5.6 ties (exact binary values): equal own-input distance -> smaller consensus-B distance (trials 2, 4), then "
          "lower t (2); all equal -> lower t; no complete trial -> none",
          closest_trial(rows_t) == 2 and closest_trial([rows_t[2], rows_t[0]]) == 1 and closest_trial([rows_t[4]]) is None,
          f"{closest_trial(rows_t)} {closest_trial([rows_t[2], rows_t[0]])}")
    a7 = [{"trial": t, "complete": t != 5, "cons_auc": 0.61, "own_sp": 0.9, "own_mp": 0.9, "own_auc": 0.5, "own_mppe": 1.0,
           "cons_mppe": 1.0} for t in range(1, 11)]
    b6 = [dict(r, complete=r["trial"] != 2, cons_auc=0.6) for r in a7]
    v = ref6_report(a7, b6)
    check("5.6: Delta_B (report only) is paired over the trials complete for BOTH sets",
          v["shared_trials"] == [1, 3, 4, 6, 7, 8, 9, 10] and abs(v["delta_B"] - 0.01) < 1e-12 and v["report_only"], v)  # fmt: skip

    # ---- guard
    forbidden = ["/x/clean_LOCAL_NOT_FOR_UPLOAD.csv", "/x/Clean/y", "/x/packed_x.csv", "/d/track_2/tier_1_multiview_caption/e",
                 "/d/video_to_data_challenge/track_2/baseline/x", str(ME / "data/human_motion_data/whole_body/x"),
                 str(ME / "data/v2d_challenge_data_public/x"), "/z/README.txt", "/z/train_rl.log",
                 str(ME / "tmp_analysis/t2_zx07_20261028/x"), "/z/best_weights.json", "/z/seed_best.csv", "/z/remote_reps.csv",
                 str(ROOT / "REPORT_R7.md"), "/z/composites.csv", "/z/per_episode_zx.csv", "/z/tier2_eval.csv",
                 "/z/tier2_runs.csv", "/z/retest/summary.txt", "/z/tier2_best_scan_x.json", "/z/eval_gen_LOCAL_x.csv",
                 str(ROOT / "clean_LOCAL_NOT_FOR_UPLOAD/x.json"), str(ROOT / "pack_r7_LOCAL_NOT_FOR_UPLOAD/x")]  # fmt: skip
    refused = []
    for p in forbidden:
        try:
            C.guard(p)
        except PermissionError:
            refused.append(p)
    check("guard refuses every forbidden pattern of PREREG_R7 5.2", refused == forbidden, sorted(set(forbidden) - set(refused)))
    allowed = [str(PH["S"]["runs"] / "ep00_z5k3fix_90000/run1/try1/results/episode_000000.parquet"),
               str(R6_RUNS_S / "ep00_raw_41043/run1/try1/results/episode_000000.parquet"),
               str(PH["F"]["runs"] / "ep26_zx_k3fix_92000/run10/try1/results/episode_000026.parquet"),
               str(EXPORTS / "ep00_z5k3fix_89500/export.json"), str(R1 / "refs/B/episode_000026.npz"), str(POOL), str(INCS),
               str(REFS6), str(C.K3CA_REFS / "episode_000026.npz"), str(C.G2_REFS / "episode_000019.npz"), str(C.T2 / "mesh"),
               str(PH["S"]["evals"]), str(PH["F"]["manifest"]), str(FINAL), str(SEL_S7), str(TRACKS), str(RELOC_TSV),
               *[str(p) for p, _ in R6_FILES.values()]]  # fmt: skip
    passed = []
    for p in allowed:
        try:
            C.guard(p)
            passed.append(p)
        except PermissionError:
            pass
    check("guard passes the noisy-only inputs of 5.1-5.6", passed == allowed, sorted(set(allowed) - set(passed)))
    check("input references: raw = released Tier 2, k3fix / k3 / k3ca / g2 = round 1's INPUT_REF",
          C.INPUT_REF["raw"] == C.T2 and C.INPUT_REF["k3ca"] == ME / "tmp_analysis/tier2/k3ca/frame_refs_k3ca"
          and C.INPUT_REF["k3fix"] == ME / "tmp_analysis/tier2/k3fix_pilot/frame_refs_k3fix"
          and C.INPUT_REF["k3"] == ME / "tmp_analysis/tier2/kalman/eval/frame_refs_k3"
          and C.INPUT_REF["g2"] == Path("/root/autodl-tmp/zuxing.lu/data/human_motion_data_tier2_g2-2026-10-24/frame_refs_g2"), "")  # fmt: skip
    # ---- incumbent / reference tables and the barred list (pinned, noisy-only records)
    try:
        incs = load_incumbents()
        refs = load_refs6()
        pool = pool_rows()
        ok_t = all(incs[e]["job"] == INC_TABLE[e][0] for e in INC_TABLE) and all(refs[e]["job"] == REF_TABLE[e][0] for e in STRICT)
        ulp = [ep_key(e) for e in sorted(incs) if incs[e]["m_P_frozen"] != incs[e]["m_P_frozen_field_resolver"]]
        barred = {e: sorted(barred_of(pool, e, incs[e]["job"])) for e in sorted(incs)}
        nb = {ep_key(e): v for e, v in barred.items() if v}
        in_pool = sorted(r["job"] for r in pool.values() if r["sha256"] in {x["training_sha256"] for x in refs.values()})
        roles = {}
        for r in pool.values():
            roles[r["role"]] = roles.get(r["role"], 0) + 1
        ok_t = ok_t and nb == {"12": ["ep12_raw_37575"], "13": ["ep13_raw_44922"]} and not in_pool and \
            roles == {"screen": N_SCREEN, "reuse": N_REUSE, "incumbent": 20}  # fmt: skip
        detail = (f"23 incumbents, 7 references; resolver field differs in the last bits on {ulp}; barred {nb}; reference "
                  f"files in the pool: {in_pool}; pool roles {roles}")  # fmt: skip
    except (SystemExit, KeyError, ValueError, OSError, StopIteration) as err:
        ok_t, detail = False, f"error: {err}"
    check("incumbents: 23 rows == PREREG_R7 4.1 table; frozen runs == the frozen scores CSV of the selecting round; "
          "m_P_frozen = the pinned stats() mean == that round's frozen job stats mean (exact; ep04 / ep11 also == "
          "selection_r6.json m_F); references == PREREG_R7 4.2 table and round 6's S6, none in the pool; barred == "
          "ep12_raw_37575, ep13_raw_44922; pool roles 154 / 245 / 20", ok_t, detail)  # fmt: skip
    out = {"written_at": C.now(), "script_sha256": raw_sha(MINE), "round2_scorer_sha256": P2_SHA, "gate_script_sha256": GATE_SHA,
           "consensus_script_sha256": CONS_SHA, "pass": not problems, "n_tests": len(res), "problems": problems,
           "trial": a.trial, "tests": res,
           "note": "synthetic values plus the pinned incumbent / reference / pool records; no round-7 score was read or "
                   "computed; no clean value"}  # fmt: skip
    write_new(out_path, C.dump(out))
    for r in res:
        print(("ok   " if r["ok"] else "FAIL ") + r["test"] + (f"  [{r['detail']}]" if r["detail"] and not r["ok"] else ""))
    print(f"selftest: {len(res)} tests, {len(problems)} problems -> {out_path}")
    return 0 if not problems else 1


# ================================================================================================ check (dry)
def cmd_check(a) -> int:
    ph = a.phase
    lines: list[str] = []

    def p(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    p(f"t2cons_r7_consensus CHECK phase {ph} {C.now()}: no metric, reference or score is computed")
    fatal, info = chain(ph, "check")
    for x in info:
        p(f"chain: {x}")
    for x in fatal:
        p(f"chain FAILED: {x}")
    jobs = load_jobs(ph)
    evals, n_not_ok, problems = load_evals(ph, jobs)
    nr = PH[ph]["nruns"]
    full = [j for j in sorted(jobs) if sorted(evals.get(j, {})) == list(range(1, nr + 1))]
    by_role = {r: sum(1 for j in jobs.values() if j["role"] == r) for r in sorted({j["role"] for j in jobs.values()})}
    p(f"phase {ph} jobs {len(jobs)} {by_role}; "
      f"export ok {sum(j['export_status'] == 'ok' for j in jobs.values())}; complete ({nr} ok runs) {len(full)}; ok runs "
      f"{sum(len(v) for v in evals.values())}; non-ok rows in the evals CSV(s): {n_not_ok}")
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
    if ph == "S":
        rp6 = r6_rollout_check(evals)
        problems += rp6
        p(f"reused round-6 rollouts == round 6's ROLLOUTS_S6.sha256: {not rp6} "
          f"({sum(1 for j in evals for r in evals[j].values() if r['source_round'] == 'r6')} runs)")
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
    write_new(ROOT / "dryrun" / f"r7_consensus_check_{ph}_{C.stamp()}.txt", "\n".join(lines) + "\n")
    return 0 if ok else 1


# ================================================================================================ gate (5.1)
GATE_NUM_COLS = ["eval_auc"] + [f"auc_s{s}" for s in range(10)] + ["seed_min", "seed_max", "seed_range",
                                                                     "excess_outside_range", "d_auc_seed0", "eval_mppe",
                                                                     "mine_mppe", "d_mppe", "eval_sp_sr", "mine_sp_sr",
                                                                     "eval_mp_sr", "mine_mp_sr", "conv_pos_residual_m",
                                                                     "conv_ang_residual_rad"]  # fmt: skip


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
    rp = check_rollouts(ph, evals) + lp + (r6_rollout_check(evals) if ph == "S" else [])
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
    # K6: the reused round-6 runs' gate rows vs round 6's own_input_gate_s6.csv (flags must agree; values reported)
    repro = None
    if ph == "S":
        g6 = {(r["job"], int(r["run"])): r for r in C.read_csv(r6file("gate_s6"))}
        mism, nnum, maxd, nexact = [], 0, 0.0, 0
        for r in rows:
            if jobs[r["job"]]["source_round"] != "r6":
                continue
            o = g6.get((r["job"], int(r["run"])))
            if o is None or o["status"] != r["status"] or o["flags"] != r["flags"]:
                mism.append(f"{r['job']} r{r['run']}: round 6 {None if o is None else (o['status'], o['flags'])} vs "
                            f"{(r['status'], r['flags'])}")  # fmt: skip
                continue
            if r["status"] == "ok":
                nnum += 1
                dd = [abs(float(o[c]) - float(r[c])) for c in GATE_NUM_COLS]
                maxd = max(maxd, max(dd))
                nexact += all(x == 0.0 for x in dd)
        repro = {"n_reused_runs": sum(1 for r in rows if jobs[r["job"]]["source_round"] == "r6"), "flag_mismatches": mism,
                 "n_numeric_compared": nnum, "n_numeric_exact": nexact, "max_abs_numeric_diff": maxd,
                 "pass": not mism}  # fmt: skip
    flagged_jobs = sorted({r["job"] for r in rows if r["flags"]})
    ok = [r for r in rows if r["status"] == "ok"]
    passed = not diff and (repro is None or repro["pass"])
    cols = ["job", "episode", "run", "try", "input", "id", "status", "problem", "flags", "eval_auc"] + \
        [f"auc_s{s}" for s in G.SEEDS] + \
        ["seed_min", "seed_max", "seed_range", "excess_outside_range", "band_ok", "inside_range", "d_auc_seed0",
         "eval_mppe", "mine_mppe", "d_mppe", "eval_sp_sr", "mine_sp_sr", "eval_mp_sr", "mine_mp_sr",
         "conv_pos_residual_m", "conv_ang_residual_rad"]  # fmt: skip
    write_new(F["gate_csv"], C.csv_text(rows, cols))
    out = {
        "written_at": C.now(), "phase": ph, "script_sha256": raw_sha(MINE), "round2_scorer_sha256": P2_SHA,
        "consensus_script_sha256": CONS_SHA, "gate_script_sha256": GATE_SHA, "prereg_r7_sha256": PREREG_SHA,
        "pins_b_sha256": raw_sha(PINS_B),
        "rule": "PREREG_R7 5.1 (round 6's 5.3, round-1 AMENDMENTS A1.3): a run is flagged iff the evaluator AUC is outside "
                "[min_s AUC_s - 0.02, max_s AUC_s + 0.02] (s = 0..9), |dMPPE| > 0.5 cm, SP-SR or MP-SR differs from the "
                "evaluator, or it is a problem run. Phase S: a candidate with a flagged run is validation_failed. Phase F: "
                "a flagged finalist run disqualifies / unconfirms the finalist; a flagged incumbent run is left out of "
                "m_P_fresh and its trial is incomplete. A kit / stock-runtime change fails the gate. Phase S: the reused "
                "round-6 runs' flags must equal round 6's own_input_gate_s6.csv (K6).",
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
        "kit_stock_unchanged": not diff, "integrity_diff": diff[:50], "reuse_gate_reproduction": repro,
        "vertex_meshes": {n: {"file": str(v[1]), "sha256": sha(v[1])} for n, v in sorted(K.verts.items())},
        "rollouts_list_sha256": raw_sha(PH[ph]["rollouts"]), "csv_sha256": raw_sha(F["gate_csv"]),
        "note": "own-input check only (each run vs its OWN input track); seeds 1..9 used only here; no consensus "
                "reference is read; nothing is selected",
    }  # fmt: skip
    write_new(F["gate_json"], C.dump(out))
    print(json.dumps({k: out[k] for k in ("pass", "n_jobs", "n_runs", "n_scored", "n_flagged_runs", "flagged_jobs",
                                          "n_inside_seed_range", "max_excess_outside_range", "max_abs_d_mppe",
                                          "kit_stock_unchanged", "reuse_gate_reproduction")}, indent=1))  # fmt: skip
    for x in out["flagged_runs"][:40]:
        print("  flagged:", x)
    return 0 if passed else 1


# ================================================================================================ score (5.1)
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
    rp = check_rollouts(ph, evals) + lp + (r6_rollout_check(evals) if ph == "S" else [])
    if rp:
        print("FATAL: rollouts / runs differ:\n  " + "\n  ".join(rp[:20]))
        return 2
    roll = C.hash_list(PH[ph]["rollouts"])
    before = C.integrity()
    K = C.load_kit()
    refs = C.load_refs(K)  # pinned: round-1 refs/build.json (9646de2d, PREREG_R7 11), every output re-hashed; never rebuilt
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
    # K6 (PREREG_R7 3.2): the reused round-6 runs must reproduce round 6's consensus_scores_s6.csv exactly
    repro = None
    if ph == "S":
        s6 = {(r["job"], int(r["run"])): r for r in C.read_csv(r6file("scores_s6")) if r["variant"] == VARIANT}
        js6 = C.read_json(r6file("jstats_s6"))
        reused = {j for j in jobs if jobs[j]["source_round"] == "r6"}
        mism, nr = [], 0
        mine_keys = {(r["job"], int(r["run"])) for r in rows if r["job"] in reused}
        theirs = {k for k in s6 if k[0] in reused}
        if mine_keys != theirs:
            mism.append(f"scored (job, run) sets differ: {len(mine_keys)} vs round 6 {len(theirs)}; "
                        f"{sorted(mine_keys ^ theirs)[:10]}")  # fmt: skip
        for r in rows:
            if r["job"] not in reused:
                continue
            o = s6.get((r["job"], int(r["run"])))
            if o is None:
                continue
            nr += 1
            if o["status"] != r["status"] or o["checkpoint_sha256"] != r["checkpoint_sha256"] or o["parquet"] != r["parquet"]:
                mism.append(f"{r['job']} r{r['run']}: status / checkpoint / parquet")
                continue
            if r["status"] == "ok":
                for c in ("auc", "sp_sr", "mp_sr", "mppe", "conv_pos_residual_m", "conv_ang_residual_rad"):
                    if float(o[c]) != float(r[c]):
                        mism.append(f"{r['job']} r{r['run']}: {c} {o[c]} (round 6) != {r[c]!r}")
        if sorted(set(js6.get("scored_jobs") or []) & reused) != sorted(j for j in scored_jobs if j in reused):
            mism.append("the reused jobs scored in round 6 != those scored here")
        repro = {"n_reused_jobs": len(reused), "n_reused_scored_jobs": sum(1 for j in scored_jobs if j in reused),
                 "n_runs_compared": nr, "mismatches": mism[:50], "n_mismatches": len(mism), "pass": not mism,
                 "round6_scores_csv": str(R6_FILES["scores_s6"][0]), "round6_scores_csv_sha256": R6_FILES["scores_s6"][1]}  # fmt: skip
    after = C.integrity()
    diff = C.integrity_diff(before, after)
    out = {"written_at": C.now(), "phase": ph, "script_sha256": raw_sha(MINE), "variant": VARIANT, "prereg_r7_sha256": PREREG_SHA,
           "pins_b_sha256": raw_sha(PINS_B), "gate_json_sha256": raw_sha(F["gate_json"]),
           "refs_build_sha256": raw_sha(R1 / "refs/build.json"), "rollouts_list_sha256": raw_sha(PH[ph]["rollouts"]),
           "scores_csv_sha256": raw_sha(F["scores_csv"]), "validation_failed_jobs": excluded, "scored_jobs": scored_jobs,
           "n_rows": len(rows), "n_ok_rows": sum(r["status"] == "ok" for r in rows), "problems": problems,
           "integrity_kit_stock_unchanged": not diff, "integrity_diff": diff[:50], "reuse_reproduction": repro,
           "jobs": jstats}  # fmt: skip
    write_new(F["jstats"], C.dump(out))
    print(f"phase {ph}: scored {len(scored_jobs)} jobs against refs/{VARIANT}: {out['n_ok_rows']}/{len(rows)} ok rows; "
          f"problems {len(problems)}; kit/stock unchanged {not diff}"
          + (f"; reuse reproduction: {repro['n_runs_compared']} runs, {repro['n_mismatches']} mismatches -> "
             f"{'PASS' if repro['pass'] else 'FAIL'}" if repro else ""))  # fmt: skip
    for x in problems[:20]:
        print("  problem:", x)
    for x in (repro or {}).get("mismatches", [])[:20]:
        print("  reuse mismatch:", x)
    return 0 if not diff and (repro is None or repro["pass"]) else 1


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


# ================================================================================================ select S (5.3 / 5.4)
def cmd_select_s() -> int:
    if not require("S", "select"):
        return 2
    js, g, jobs, evals, scores, flagged = scored_inputs("S")
    excluded = set(g.get("validation_failed_jobs") or [])
    incs = load_incumbents()
    pool = pool_rows()
    jst = js["jobs"]
    eps, longrows, lines, finalists, halts = [], [], [], [], []
    for e in range(30):
        cands = sorted(j for j in jobs if jobs[j]["episode"] == e)
        info = {}
        for c in cands:
            s = jst.get(c)
            info[c] = {"export_ok": jobs[c]["export_status"] == "ok", "own": evals.get(c, {}),
                       "flagged": c in excluded or any(jj == c for jj, _ in flagged),
                       "cons": None if s is None or not s["n"] else {"runs": s["runs"], "auc": s["auc"], "mppe": s["mppe"]}}  # fmt: skip
        if e in STRICT:
            fins, mode, rows = select_strict(cands, info)
            I, barred = None, set()
            if mode == "halt":
                halts.append(e)
        else:
            I = incs[e]
            barred = barred_of(pool, e, I["job"])
            f, rows = select_s(I["m_P_frozen"], cands, info, barred=barred)
            fins, mode = ([f] if f else []), ("finalist" if f else "no finalist")
        for r in rows:
            j = jobs[r["job"]]
            s = jst.get(r["job"], {})
            r.update(input=j["input"], owner=j["owner"], machine=j["machine"], pool_role=j["pool_role"],
                     run_label=j["run_label"], iteration=j["iteration"], export_status=j["export_status"],
                     own_auc=[evals.get(r["job"], {})[k]["auc"] for k in sorted(evals.get(r["job"], {}))],
                     cons_sp_mean=(s.get("sp_sr_stats") or {}).get("mean"), cons_mp_mean=(s.get("mp_sr_stats") or {}).get("mean"))  # fmt: skip
        rows.sort(key=lambda r: (-(r.get("cons_mean") if r.get("cons_mean") is not None else -1.0), r["job"]))
        rec = {"episode": e, "strict": e in STRICT, "incumbent": I["job"] if I else None,
               "incumbent_input": I["input"] if I else None, "m_P_frozen": I["m_P_frozen"] if I else None,
               "bar": I["m_P_frozen"] + MARGIN if I else None, "barred": sorted(barred), "n_candidates": len(cands),
               "finalists": fins, "selection_mode": mode, "n_eligible": sum(r["eligible"] for r in rows),
               "n_qualifying": sum(r["qualifies"] for r in rows), "candidates": rows}  # fmt: skip
        eps.append(rec)
        for i, f in enumerate(fins):
            j = jobs[f]
            fr_row = next(r for r in rows if r["job"] == f)
            finalists.append({"job": f, "episode": e, "strict_episode": e in STRICT, "rank": i + 1 if e in STRICT else None,
                              "selection_mode": mode, "fallback_unconfirmed_S": mode == "fallback_unconfirmed_S",
                              "pool_role": j["pool_role"], "input": j["input"], "owner": j["owner"], "machine": j["machine"],
                              "tree": j["tree"], "run_dir": j["run_dir"], "run_label": j["run_label"],
                              "iteration": j["iteration"], "pool_sha256": j["pool_sha256"],
                              "training_sha256": j["training_sha256"], "export_sha256": j["export_sha256"],
                              "export_file": j["export_file"], "cons_S_mean": fr_row["cons_mean"],
                              "cons_S_var": fr_row["cons_var"], "cons_S_mppe_mean": fr_row["cons_mppe_mean"],
                              "floor_S": fr_row["floor_count"], "incumbent": I["job"] if I else None,
                              "m_P_frozen": I["m_P_frozen"] if I else None, "bar": I["m_P_frozen"] + MARGIN if I else None})  # fmt: skip
        for r in rows:
            longrows.append({"episode": ep_key(e), "strict": e in STRICT, "job": r["job"], "pool_role": r["pool_role"],
                             "input": r["input"], "owner": r["owner"], "machine": r["machine"], "run_label": r["run_label"],
                             "iteration": r["iteration"], "export_status": r["export_status"], "n_own_ok": r["n_own_ok"],
                             "floor_count": r["floor_count"], "flagged": r["flagged"],
                             **{f"cons_auc_r{k}": (r.get("cons_auc") or [None] * 3)[k - 1]
                                if len(r.get("cons_auc") or []) >= k else None for k in (1, 2, 3)},
                             "cons_mean": r.get("cons_mean"), "cons_sd": math.sqrt(r["cons_var"]) if "cons_var" in r else None,
                             "cons_mppe_mean": r.get("cons_mppe_mean"), "incumbent": I["job"] if I else "",
                             "m_P_frozen": I["m_P_frozen"] if I else None, "delta_vs_m_P": r.get("delta_vs_m_P"),
                             "eligible": r["eligible"], "barred": r.get("barred", False), "qualifies": r["qualifies"],
                             "strict_rank": r.get("strict_rank"), "finalist": r["job"] in fins,
                             "why_not": "; ".join(r["why_not"])})  # fmt: skip
        top = [r for r in rows if r.get("cons_mean") is not None][:3]
        if I:
            lines.append(f"ep{e:02d} P {I['job']} [{I['input']}] m_P_frozen {I['m_P_frozen']:.6f} (bar {I['m_P_frozen'] + MARGIN:.6f}); "
                         f"{len(cands)} candidates, {rec['n_eligible']} eligible, {rec['n_qualifying']} qualifying -> "
                         f"finalist {fins[0] if fins else 'none'}")  # fmt: skip
        else:
            lines.append(f"ep{e:02d} STRICT (no incumbent); {len(cands)} candidates, {rec['n_eligible']} S-eligible -> "
                         f"{mode}: " + (", ".join(f"rank {i + 1} {f}" for i, f in enumerate(fins)) or "none"))  # fmt: skip
        for r in top:
            lines.append(f"      {r['job']} [{r['input']}] cons_S {r['cons_mean']:.6f} sd {math.sqrt(r['cons_var']):.4f} "
                         f"floor {r['floor_count']}/3 eligible={r['eligible']} qualifies={r['qualifies']} {r['why_not']}")
    if halts:
        print(f"HALT (PREREG_R7 5.4): STRICT episodes {halts} have no candidate with 3 ok, unflagged runs; an "
              "AMENDMENTS_R7.md written before any phase-F score decides. Nothing written.")  # fmt: skip
        return 3
    gen = C.now()
    hashes = {"prereg_r7_sha256": PREREG_SHA, "pins_b_sha256": raw_sha(PINS_B), "script_sha256": raw_sha(MINE),
              "selftest_sha256": raw_sha(SELFTEST), "manifest_s7_sha256": raw_sha(PH["S"]["manifest"]),
              "own_input_evals_s7_sha256": raw_sha(PH["S"]["evals"]), "rollouts_s7_sha256": raw_sha(PH["S"]["rollouts"]),
              "gate_json_sha256": raw_sha(files_of("S")["gate_json"]), "scores_csv_sha256": raw_sha(files_of("S")["scores_csv"]),
              "jstats_sha256": raw_sha(files_of("S")["jstats"]), "pool_r7_sha256": POOL_SHA, "incumbents_r7_sha256": INCS_SHA,
              "r6refs_r7_sha256": REFS6_SHA}  # fmt: skip
    sel = {"generated_at": gen, "phase": "S", "synthetic": False, "noisy_only": True, "binding_variant": VARIANT, **hashes,
           "rule": RULE_S, "validation_failed_jobs": sorted(excluded), "reuse_reproduction": js.get("reuse_reproduction"),
           "n_finalists": len(finalists), "finalist_episodes": sorted({ep_key(x["episode"]) for x in finalists}),
           "episodes": eps}  # fmt: skip
    final = {"generated_at": gen, "prereg_r7_sha256": PREREG_SHA, "script_sha256": hashes["script_sha256"],
             "pins_b_sha256": hashes["pins_b_sha256"], "rule": RULE_S, "finalists": finalists,
             "no_finalist_episodes": [ep_key(x["episode"]) for x in eps if not x["finalists"]],
             "note": "phase-F job list (PREREG_R7 3.3): the 23 incumbents of inputs_r7/incumbents_r7.json + these finalists "
                     "(a new screen job with its single phase-S export; a reused candidate with its round-6 export) + the 7 "
                     "report-only round-6 references; locked by SELECTION_S7.sha256"}  # fmt: skip
    written = []

    def w(name: str, text: str) -> None:
        write_new(ROOT / name, text)
        written.append(name)

    w("selection_s7.json", C.dump(sel))
    w("final_r7.json", C.dump(final))
    head = (f"selection_s7 {gen} CST (PREREG_R7 5.3 / 5.4; noisy-only consensus-B vs round 1's frozen refs/B; validation_failed "
            f"{len(excluded)} jobs; finalists {len(finalists)})")  # fmt: skip
    w("selection_s7.txt", "\n".join([head] + lines + ["", "finalists: " + (", ".join(
        f"{x['episode']:02d}:{x['job']}" + (f" (rank {x['rank']}{', fallback' if x['fallback_unconfirmed_S'] else ''})"
                                           if x["strict_episode"] else f" ({x['cons_S_mean']:.4f} vs bar {x['bar']:.4f})")
        for x in finalists) or "none")]) + "\n")  # fmt: skip
    w("selection_s7_candidates.csv", C.csv_text(longrows, list(longrows[0].keys())))
    md = ["# t2_consensus_r7_20261102: phase-S selection (noisy-only; PREREG_R7 5.3 / 5.4)", "",
          f"Generated {gen} CST by `scripts/t2cons_r7_consensus.py` ({hashes['script_sha256'][:16]}...), pinned in "
          f"`PREREG_R7_PINS_b.md` ({hashes['pins_b_sha256'][:16]}...). {N_SCREEN} new screen + {N_REUSE} reused round-6 "
          "candidates x 3 official runs, each run scored against round 1's frozen consensus refs/B (variant B) and gated "
          "against its own input (5.1). 23 incumbent episodes: finalist = the S-eligible candidate with the highest 3-run "
          "consensus mean, provided it is >= m_P_frozen + 0.01. 7 STRICT episodes: the top 2 S-eligible candidates. No "
          "clean or LOCAL file was opened; no Tier-1 value was used.", "",
          f"Finalists: {len(finalists)}; validation_failed candidates: {len(excluded)}; reused round-6 runs reproduce "
          f"scores_s6: {(js.get('reuse_reproduction') or {}).get('pass')}.", "",
          "| ep | incumbent (input) | m_P_frozen | bar | candidates / eligible / qualifying | best candidate: cons_S (floor) | "
          "**finalist(s)** |", "|---|---|---:|---:|---|---|---|"]  # fmt: skip
    for x in eps:
        best = next((r for r in x["candidates"] if r.get("cons_mean") is not None), None)
        md.append(f"| {x['episode']:02d} | {(x['incumbent'] + ' (' + x['incumbent_input'] + ')') if x['incumbent'] else 'STRICT: none'} | "
                  f"{fmt(x['m_P_frozen'])} | {fmt(x['bar'])} | {x['n_candidates']} / {x['n_eligible']} / {x['n_qualifying']} | "
                  + (f"{best['job']}: {best['cons_mean']:.4f} ({best['floor_count']}/3){' flagged' if best['flagged'] else ''}" if best else "")
                  + f" | **{', '.join(x['finalists']) or '-'}**{' (' + x['selection_mode'] + ')' if x['strict'] else ''} |")  # fmt: skip
    w("selection_s7.md", "\n".join(md) + "\n")
    frozen_inputs = [rel_name(p) for p in (files_of("S")["scores_csv"], files_of("S")["jstats"], files_of("S")["gate_csv"],
                                           files_of("S")["gate_json"], SELFTEST, PH["S"]["rollouts"], PH["S"]["evals"],
                                           PH["S"]["manifest"], PINS_B, ROOT / "PREREG_R7_PINS.md", DRYRUN, LOCK, PREREG,
                                           MINE)]  # fmt: skip
    write_new(SEL_S7, frozen_hashes(written + frozen_inputs))  # LAST: from here the phase-S selection is frozen
    print("\n".join(md[6:]))
    print(f"\nSELECTION_S7.sha256 written {C.now()} (frozen): {raw_sha(SEL_S7)}")
    return 0


# ================================================================================================ select F (5.5 / 5.6)
def cmd_select_f() -> int:
    if not require("F", "select"):
        return 2
    js, g, jobs, evals, scores, flagged = scored_inputs("F")
    incs = load_incumbents()
    refs = load_refs6()
    final = C.read_json(FINAL)
    fins_by_ep: dict[int, list[dict]] = {}
    for x in final["finalists"]:
        fins_by_ep.setdefault(int(x["episode"]), []).append(x)
    sel6 = C.read_json(r6file("selection_r6"))
    s6comp = sel6["compositions"]["S6"]

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

    def strip(rr: dict) -> dict:
        return {k: {x: v for x, v in r.items() if x != "parquet"} for k, r in rr.items()}

    runinfo = {j: runinfo_of(j) for j in jobs}
    eps, s7, ref6, lines, table, halts = [], {}, {}, [], [], []
    for e in range(30):
        fl = sorted(fins_by_ep.get(e, []), key=lambda x: (x["rank"] or 0, x["job"]))
        if e in STRICT:
            if not fl:
                raise SystemExit(f"ep{e:02d}: STRICT episode without a finalist in final_r7.json")
            d = decide_strict([(x["job"], int(x["rank"]), runinfo.get(x["job"]) or {}) for x in fl])
            if d["status"] == "halt":
                halts.append(e)
                continue
            pick = d["pick"]
            ref6[e] = refs[e]["job"]
            rec = {"episode": e, "strict": True, "incumbent": None, "decision": d, "pick": pick, "pick_status": d["status"],
                   "pick_rank": d["pick_rank"], "fallback_unconfirmed_S": bool(fl[0]["fallback_unconfirmed_S"]),
                   "r6_reference": refs[e]["job"], "finalist_runs": {x["job"]: strip(runinfo.get(x["job"]) or {}) for x in fl},
                   "r6_reference_runs_report_only": strip(runinfo.get(refs[e]["job"]) or {})}  # fmt: skip
            frec = {r["job"]: r for r in d["finalists"]}
            for x in fl:
                r = frec[x["job"]]
                lines.append(f"ep{e:02d} STRICT rank {x['rank']} {x['job']} [{x['input']}] cons_S {x['cons_S_mean']:.6f}: "
                             f"{len(r['good_runs'])} good runs, floor {r['floor_count']}/10, m_F {fmt(r['m_F'], '.6f')} -> "
                             f"{'confirmed' if r['confirmed'] else 'not confirmed: ' + '; '.join(r['why_not_confirmed'])}")  # fmt: skip
            lines.append(f"ep{e:02d} STRICT pick {pick} ({d['status']}, rank {d['pick_rank']})"
                         + ("; phase-S fallback_unconfirmed_S" if rec["fallback_unconfirmed_S"] else ""))  # fmt: skip
            table.append({"episode": ep_key(e), "strict": True, "incumbent": "", "incumbent_input": "", "m_P_frozen": None,
                          "m_P_fresh": None, "incumbent_good_runs": None, "bar": None,
                          "finalist": " / ".join(x["job"] for x in fl), "finalist_input": " / ".join(x["input"] for x in fl),
                          "cons_S_mean": " / ".join(f"{x['cons_S_mean']:.6f}" for x in fl),
                          "m_F": " / ".join(fmt(frec[x["job"]]["m_F"], ".6f") for x in fl),
                          "finalist_floor": " / ".join(str(frec[x["job"]]["floor_count"]) for x in fl),
                          "changed": None, "pick": pick, "pick_status": d["status"],
                          "why_not": "; ".join(f"{x['job']}: {'; '.join(frec[x['job']]['why_not_confirmed'])}" for x in fl
                                               if not frec[x["job"]]["confirmed"])})  # fmt: skip
        else:
            I = incs[e]
            f = fl[0] if fl else None
            if len(fl) > 1:
                raise SystemExit(f"ep{e:02d}: more than one finalist for an incumbent episode")
            d = decide_f(I["m_P_frozen"], runinfo[I["job"]], f["job"] if f else None, runinfo.get(f["job"]) if f else None)
            pick = f["job"] if d["changed"] else I["job"]
            ref6[e] = pick
            rec = {"episode": e, "strict": False, "incumbent": I["job"], "incumbent_input": I["input"],
                   "finalist": f["job"] if f else None, "finalist_input": f["input"] if f else None,
                   "finalist_cons_S_mean": f["cons_S_mean"] if f else None, **d, "pick": pick,
                   "pick_status": "replaced" if d["changed"] else "incumbent kept",
                   "incumbent_runs": strip(runinfo[I["job"]]), "finalist_runs": strip(runinfo[f["job"]]) if f else None}  # fmt: skip
            lines.append(f"ep{e:02d} P {I['job']} [{I['input']}] m_P_frozen {I['m_P_frozen']:.6f} m_P_fresh "
                         f"{fmt(d['m_P_fresh'], '.6f') or 'inf'} ({len(d['incumbent_good_runs'])} good runs) bar {fmt(d['bar'], '.6f') or 'inf'}"
                         + (f"; finalist {f['job']} [{f['input']}] m_F {fmt(d.get('m_F'), '.6f')} floor {d.get('floor_count')}/10"
                            f" -> {'REPLACES' if d['changed'] else 'no: ' + '; '.join(d['why_not'])}" if f else "; no finalist"))  # fmt: skip
            table.append({"episode": ep_key(e), "strict": False, "incumbent": I["job"], "incumbent_input": I["input"],
                          "m_P_frozen": I["m_P_frozen"], "m_P_fresh": d["m_P_fresh"],
                          "incumbent_good_runs": len(d["incumbent_good_runs"]), "bar": d["bar"],
                          "finalist": f["job"] if f else "", "finalist_input": f["input"] if f else "",
                          "cons_S_mean": f["cons_S_mean"] if f else None, "m_F": d.get("m_F"),
                          "finalist_floor": d.get("floor_count"), "changed": d["changed"], "pick": pick,
                          "pick_status": rec["pick_status"], "why_not": "; ".join(d["why_not"])})  # fmt: skip
        s7[e] = pick
        j = jobs[pick]
        rec.update(pick_input=j["input"], pick_owner=j["owner"], pick_machine=j["machine"], pick_run_dir=j["run_dir"],
                   pick_training_checkpoint=j.get("training_checkpoint"), pick_source_checkpoint=j.get("source_checkpoint"),
                   pick_selected_in=j.get("selected_in"),
                   pick_run_label=j["run_label"], pick_iteration=j["iteration"], pick_pool_role=j.get("pool_role"),
                   r6_pick=s6comp[ep_key(e)], changed_vs_r6=pick != s6comp[ep_key(e)])  # fmt: skip
        eps.append(rec)
    if halts:
        print(f"HALT (PREREG_R7 5.5): STRICT episodes {halts}: no finalist has >= 8 ok, non-flagged phase-F runs; an "
              "AMENDMENTS_R7.md decides before any phase-F composite is computed. Nothing written.")  # fmt: skip
        return 3
    rows7 = trial_rows(s7, runinfo)
    rowsR = trial_rows(ref6, runinfo)
    t_star = closest_trial(rows7)
    rep = ref6_report(rows7, rowsR)
    r7_file = {}
    if t_star is not None:
        roll = C.hash_list(PH["F"]["rollouts"])
        for e in range(30):
            j = jobs[s7[e]]
            r = runinfo[s7[e]][t_star]
            r7_file[ep_key(e)] = {"job": s7[e], "run": t_star, "try": r["try"], "source": "round7 phase F",
                                  "role": j["role"], "input": j["input"], "owner": j["owner"], "machine": j["machine"],
                                  "frame_reference": j["input_reference"], "export_sha256": j["export_sha256"],
                                  "export_file": j["export_file"], "rollout": r["parquet"],
                                  "rollout_sha256": roll[r["parquet"]]}  # fmt: skip
    changed_vs_r6 = [ep_key(x["episode"]) for x in eps if x["changed_vs_r6"]]
    replaced = [ep_key(x["episode"]) for x in eps if not x["strict"] and x["changed"]]
    unconf = [ep_key(x["episode"]) for x in eps if x["strict"] and (x["pick_status"] != "confirmed" or x["fallback_unconfirmed_S"])]
    gen = C.now()
    F = files_of("F")
    hashes = {"prereg_r7_sha256": PREREG_SHA, "pins_b_sha256": raw_sha(PINS_B), "script_sha256": raw_sha(MINE),
              "selftest_sha256": raw_sha(SELFTEST), "final_r7_sha256": raw_sha(FINAL), "selection_s7_lock_sha256": raw_sha(SEL_S7),
              "manifest_f7_sha256": raw_sha(PH["F"]["manifest"]), "own_input_evals_f7_sha256": raw_sha(PH["F"]["evals"]),
              "rollouts_f7_sha256": raw_sha(PH["F"]["rollouts"]), "gate_json_sha256": raw_sha(F["gate_json"]),
              "scores_csv_sha256": raw_sha(F["scores_csv"]), "jstats_sha256": raw_sha(F["jstats"])}  # fmt: skip
    upload = ("R7-CLOSEST (trial %s of S7) is the upload-eligible Tier-2 file under the STRICT standard (PREREG_R7 5.6)"
              % t_star) if t_star is not None else "no complete S7 trial: no R7-CLOSEST"  # fmt: skip
    sel = {"generated_at": gen, "phase": "F", "synthetic": False, "noisy_only": True, "binding_variant": VARIANT, **hashes,
           "rule": RULE_F, "replaced_episodes": replaced, "strict_unconfirmed_episodes": unconf,
           "changed_vs_r6_episodes_report_only": changed_vs_r6, "episodes": eps,
           "compositions": {"S7": {ep_key(e): s7[e] for e in range(30)}, "REF6": {ep_key(e): ref6[e] for e in range(30)},
                            "S6_round6": dict(s6comp)},
           "trials": {"S7": rows7, "REF6": rowsR}, "r7_closest_trial": t_star, "r7_closest_file": r7_file,
           "ref6_report_only": rep, "upload_eligibility": upload,
           "flagged_runs": sorted([list(x) for x in flagged])}  # fmt: skip
    written = []

    def w(name: str, text: str) -> None:
        write_new(ROOT / name, text)
        written.append(name)

    w("selection_r7.json", C.dump(sel))
    head = f"selection_r7 {gen} CST (PREREG_R7 5.5 / 5.6; noisy-only; refs = round-1 refs/B, frozen)"
    tl = [f"  trial {r['trial']:2d}: S7 " + (f"own AUC {r['own_auc']:.4f} SP {r['own_sp']:.3f} MP {r['own_mp']:.3f} "
                                             f"MPPE {r['own_mppe']:.3f} cons-B {r['cons_auc']:.4f}" if r["complete"]
                                             else f"incomplete {r['missing_episodes']}")
          + " | REF6 (report only) " + (f"own AUC {q['own_auc']:.4f} cons-B {q['cons_auc']:.4f}" if q["complete"]
                                        else f"incomplete {q['missing_episodes']}") for r, q in zip(rows7, rowsR)]  # fmt: skip
    w("selection_r7.txt", "\n".join([head] + lines + ["", f"replaced incumbents: {', '.join(replaced) or 'none'}",
                                                      f"STRICT picks not confirmed: {', '.join(unconf) or 'none'}",
                                                      f"changed vs round 6 (report only): {', '.join(changed_vs_r6) or 'none'}",
                                                      "trials:"] + tl + [
        "", f"R7-CLOSEST trial: {t_star}", f"REF6 report only: Delta_B {rep['delta_B']!r} over trials {rep['shared_trials']}",
        upload]) + "\n")  # fmt: skip
    w("selection_r7_episodes.csv", C.csv_text(table, list(table[0].keys())))
    tcols = ["set", "trial", "complete", "missing_episodes", "own_auc", "own_sp", "own_mp", "own_mppe", "cons_auc", "cons_sp",
             "cons_mp", "cons_mppe"]  # fmt: skip
    trows = [{"set": "S7", **r, "missing_episodes": " ".join(map(str, r["missing_episodes"]))} for r in rows7] + \
        [{"set": "REF6", **r, "missing_episodes": " ".join(map(str, r["missing_episodes"]))} for r in rowsR]  # fmt: skip
    w("summary_r7_trials.csv", C.csv_text(trows, tcols))

    def avg(rows, key):
        xs = [r[key] for r in rows if r["complete"]]
        return float(np.mean(np.asarray(xs, dtype=np.float64))) if xs else None

    def sdv(rows, key):
        xs = [r[key] for r in rows if r["complete"]]
        return float(np.std(np.asarray(xs, dtype=np.float64), ddof=1)) if len(xs) > 1 else None

    summ = {"generated_at": gen, "replaced_episodes": replaced, "strict_unconfirmed_episodes": unconf,
            "changed_vs_r6_episodes_report_only": changed_vs_r6, "r7_closest_trial": t_star,
            "r7_closest_trial_row": next((r for r in rows7 if r["trial"] == t_star), None), "ref6_report_only": rep,
            "upload_eligibility": upload,
            "means_over_complete_trials": {s: {k: avg(rows, k) for k in tcols[4:]} for s, rows in (("S7", rows7), ("REF6", rowsR))},
            "sd_over_complete_trials": {s: {k: sdv(rows, k) for k in tcols[4:]} for s, rows in (("S7", rows7), ("REF6", rowsR))},
            "n_complete_trials": {"S7": sum(r["complete"] for r in rows7), "REF6": sum(r["complete"] for r in rowsR)},
            "note": "noisy-only: own-input = each policy vs its OWN Tier-2-derived input (evaluator values); cons-B = vs round "
                    "1's frozen consensus refs/B. Not leaderboard estimates. REF6 and Delta_B are report only."}  # fmt: skip
    w("summary_r7.json", C.dump(summ))
    md = ["# t2_consensus_r7_20261102: round-7 decision (noisy-only; PREREG_R7 5.5 / 5.6)", "",
          f"Generated {gen} CST by `scripts/t2cons_r7_consensus.py` ({hashes['script_sha256'][:16]}...). Phase F: 23 incumbents "
          f"+ {len(final['finalists'])} finalists + 7 report-only references x 10 fresh paired trials. No clean or LOCAL file "
          "was opened; no Tier-1 value.", "",
          f"Replaced incumbents: {', '.join(replaced) or 'none'}. STRICT picks not confirmed: {', '.join(unconf) or 'none'}. "
          f"Changed vs round 6 (report only): {', '.join(changed_vs_r6) or 'none'}.", "",
          "| ep | incumbent | m_P_frozen | m_P_fresh | bar | finalist(s) | m_F | floor | **pick** |", "|---|---|---:|---:|---:|---|---:|---:|---|"]  # fmt: skip
    for t in table:
        md.append(f"| {t['episode']} | {(t['incumbent'] + ' (' + t['incumbent_input'] + ')') if t['incumbent'] else 'STRICT'} | "
                  f"{fmt(t['m_P_frozen'])} | {fmt(t['m_P_fresh']) if t['m_P_fresh'] is not None else ('' if t['strict'] else 'inf')} | "
                  f"{fmt(t['bar']) if t['bar'] is not None else ('' if t['strict'] else 'inf')} | {t['finalist'] or '-'} | "
                  f"{t['m_F'] if isinstance(t['m_F'], str) else fmt(t['m_F'])} | {t['finalist_floor'] if t['finalist'] else ''} | "
                  f"**{t['pick']}** ({t['pick_status']}) |")  # fmt: skip
    md += ["", "| set | complete trials | own AUC | own SP-SR | own MP-SR | own MPPE | cons-B AUC | cons-B SP-SR | cons-B MP-SR | cons-B MPPE |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
    for s, rows in (("S7", rows7), ("REF6 (report only)", rowsR)):
        mm = summ["means_over_complete_trials"][s.split()[0]]
        md.append(f"| {s} | {summ['n_complete_trials'][s.split()[0]]} | " + " | ".join(fmt(mm[k]) for k in tcols[4:]) + " |")
    md += ["", f"R7-CLOSEST trial: {t_star}. REF6 (report only): Delta_B = {fmt(rep['delta_B'], '+.6f')} over trials "
           f"{rep['shared_trials']}. **{upload}**."]  # fmt: skip
    w("summary_r7.md", "\n".join(md) + "\n")
    w("selection_r7.md", "\n".join(md[:5] + [""] + [f"- {x}" for x in lines]) + "\n")
    frozen_inputs = [rel_name(p) for p in (F["scores_csv"], F["jstats"], F["gate_csv"], F["gate_json"], PH["F"]["rollouts"],
                                           PH["F"]["evals"], PH["F"]["manifest"], FINAL, SEL_S7, ROOT / "selection_s7.json",
                                           SELFTEST, PINS_B, ROOT / "PREREG_R7_PINS.md", DRYRUN, LOCK, PREREG, MINE)]  # fmt: skip
    write_new(SEL_R7, frozen_hashes(written + frozen_inputs))  # LAST: from here round 7 is final
    print("\n".join(md))
    print(f"\nSELECTION_R7.sha256 written {C.now()} (frozen): {raw_sha(SEL_R7)}")
    return 0


def cmd_select(a) -> int:
    if a.phase == "S":
        if SEL_S7.exists():
            print("FATAL: SELECTION_S7.sha256 exists; the phase-S selection is frozen")
            return 2
        return cmd_select_s()
    if SEL_R7.exists():
        print("FATAL: SELECTION_R7.sha256 exists; round 7 is frozen")
        return 2
    return cmd_select_f()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("selftest", "check", "gate", "score", "select"))
    ap.add_argument("--phase", choices=("S", "F"))
    ap.add_argument("--trial", type=int, default=None, help="selftest only: write dryrun/selftest_trial_r7c_N.json")
    a = ap.parse_args()
    if not sys.flags.dont_write_bytecode or not sys.flags.isolated:
        print("run with the kit python -I -B")
        return 2
    if a.cmd != "selftest" and a.phase is None:
        ap.error("--phase S|F is required")
    if a.cmd != "selftest" and a.trial is not None:
        ap.error("--trial is for selftest only")
    return {"selftest": cmd_selftest, "check": cmd_check, "gate": cmd_gate, "score": cmd_score, "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
