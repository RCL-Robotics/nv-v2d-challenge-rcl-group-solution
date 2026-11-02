#!/usr/bin/env python3
"""t2cons_r7_pool.py: step P of fc_runs/submissions/t2_consensus_r7_20261102 (round 7 STRICT, PREREG_R7 sections 1-4).

Round 6's resolver (scripts/t2cons_r6_pool.py 13974f6f...) with the round-7 bookkeeping. The MECHANICAL, clean-free
candidate pool of round 7, its 23 incumbents and 7 report-only R6 references, resolved once at the cut-off T_c (= the
start of the writing run). Nothing here looks at any score: no clean file, no evaluation result, no metrics.csv, no
best_weights / seed_best / README / training-metrics log is opened (path guard on every path; directories with such
names are pruned before they are entered; Zuxing's eval*/ and logs/ folders are never walked; of his stdout logs only the
first 'train_t2:' line is read). The incumbent / reference records read round 6's frozen noisy-only records only.

Pool rule (PREREG_R7 2.1, = round 6's R1-R5 plus R1b for the 5090 copies):
  R1  a pro6000 TRAINING-RUN directory = a directory under the scanned roots holding model_<N>.safetensors files whose
      embedded config output_dir resolves to that same directory; ours only if it is in round 6's eligible run list;
      Zuxing's copy / retest / selection bundles are not training-run directories (never entered);
  R1b a 5090 run = a row of Zuxing's 2026-11-01-round7-5090-ckpts/MANIFEST.tsv (45 rows): directory <run>, embedded
      output_dir == /mnt/gfs/home/zuxinglu/workspace/runs/<run> exactly, MANIFEST episode / data root / sequence == the
      checkpoint's, every selected file's sha256 == its SHA256SUMS line;
  R2  the training parquet lies in a Tier-2-derived data root: raw Tier 2, K3, K3fix, K3CA (ours) or G2 (Zuxing's);
  R3  only checkpoints written at least 300 s before T_c count; L = the largest such iteration; L < 10000 not eligible;
  R4  K = 3: targets tau_j = L (1 - j / 6), j = 0..3; per target the saved iteration >= L/2 nearest to it (ties: the
      larger); duplicates collapse (1..4 checkpoints per run; j = 0 is the run's last checkpoint at T_c);
  R5  roles, in the deterministic order (episode, owner, directory): 'incumbent' = sha256 of the episode's round-7
      incumbent training file (not screened); 'reuse' = sha256 of a round-6 phase-S screen candidate whose input
      parquet and frame reference are unchanged (its round-6 export and 3 runs are reused, never re-run); 'duplicate' =
      a sha256 selected earlier; every other file is 'screen' (one export, 3 fresh runs).
Zuxing's new screen files (model + rsi of the same iteration) are COPIED into zuxing_copies_r7/ with sha256 before ==
copy == after; for every 5090 run also its hydra/.hydra/*.yaml, final.json and the first 'train_t2:' line of train.log,
and the folder's MANIFEST.tsv and SHA256SUMS. His folder is only read (no write, no git command, none of his scripts).

Incumbents (PREREG_R7 4): round 6's final composition S6 (selection_r6.json, pinned by SELECTION_R6.sha256) for the 23
episodes outside STRICT = (00, 16, 19, 20, 22, 25, 28), each with its existing export record (re-hashed) and its frozen
variant-B consensus runs of the round that selected it (round 1 / 5 via incumbents_r6.json; round 6 phase F for the two
round-6 replacements ep04, ep11). The 7 STRICT episodes have NO incumbent; their round-6 picks are recorded as
report-only references (r6refs_r7.json) that never enter a decision.

  conda_envs/flashchord/bin/python -I -B scripts/t2cons_r7_pool.py check      plan only, writes nothing
  conda_envs/flashchord/bin/python -I -B scripts/t2cons_r7_pool.py resolve    the writing run (T_c); refuses if
                                                                               inputs_r7/pool_r7.json exists
  conda_envs/flashchord/bin/python -I -B scripts/t2cons_r7_pool.py relocate   metadata-only relocation of the 5090
                                                                               screen files -> relocated_r7/, RELOCATION_R7.*
  conda_envs/flashchord/bin/python -I -B scripts/t2cons_r7_pool.py verify     read-only re-check of RELOCATION_R7.tsv
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import struct  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
ZU = Path("/root/autodl-tmp/zuxing.lu")
FC = ME / "fc_runs"
SUB = FC / "submissions"
ROOT = SUB / "t2_consensus_r7_20261102"
R6 = SUB / "t2_consensus_r6_20261031"
INP = ROOT / "inputs_r7"
COPIES = ROOT / "zuxing_copies_r7"
RELOC_DIR = ROOT / "relocated_r7"
RELOC_TSV, RELOC_JSON, RELOC_SHA = ROOT / "RELOCATION_R7.tsv", ROOT / "RELOCATION_R7.json", ROOT / "RELOCATION_R7.sha256"
Z5 = ZU / "runs/2026-11-01-round7-5090-ckpts"
Z5_RUNS = "/mnt/gfs/home/zuxinglu/workspace/runs/"
Z5_DATA = "/mnt/gfs/home/zuxinglu/workspace/data/"
TZ8 = dt.timezone(dt.timedelta(hours=8))
K = 3
MIN_LAST = 10000
SETTLE_S = 300.0
STRICT = (0, 16, 19, 20, 22, 25, 28)  # PREREG_R7 4: the round-6 picks of these 7 episodes were clean-nominated
FORBIDDEN = ("clean", "local_not_for_upload", "packed_", "best_weights", "report_r", "composites", "per_episode_zx",
             "readme", "train_rl.log", "t2_zx07_20261028", "seed_best", "remote_reps", "tier_1_multiview_caption",
             "tier2_eval.csv", "tier2_runs.csv", "summary.txt", "tier2_best_scan", "eval_gen_local", "metrics.csv",
             "selection.csv", "summary.json", "reproduce.md", "training_and_g2.md")  # fmt: skip
PRUNE_DIRS = ("hydra", ".hydra", "wandb", "fc_cache", "figures", "logs")  # never walked (plus eval* below)
OUR_SCAN = ("C3r2_tier2", "C3r2_tier2_armA_90k", "C3r2_tier2_k3_stretch90k", "C3r2_tier2_k3ca", "C3r2_tier2_k3easy",
            "C3r2_tier2_k3fix", "T2fix")  # fmt: skip  (round 6's roots)
ZX_SKIP_TOP = {  # top folders of zuxing.lu/runs that are NOT walked (R1 / R1b), with the reason recorded
    "2026-11-01-round7-5090-ckpts": "5090 run copies: resolved by R1b from MANIFEST.tsv",
    "2026-10-24-5090-best": "single-checkpoint copy bundles (not training-run directories; R1)",
    "2026-10-28-tier2-retest": "retest / single-checkpoint copies (not training-run directories; R1)",
    "2026-11-02-tier2-g2-ownsel": "selection / evaluation bundle with checkpoint copies (not training-run directories; R1)",
}  # fmt: skip
G2ROOT = ZU / "data/human_motion_data_tier2_g2-2026-10-24"
ROOTS = {"raw": ME / "data/human_motion_data_tier2", "k3": ME / "data/human_motion_data_tier2_k3",
         "k3fix": ME / "data/human_motion_data_tier2_k3fix", "k3ca": ME / "data/human_motion_data_tier2_k3ca",
         "g2": G2ROOT}  # fmt: skip
Z5_ROOTNAME = {"human_motion_data_tier2": "raw", "human_motion_data_tier2_k3": "k3",
               "human_motion_data_tier2_k3fix": "k3fix", "human_motion_data_tier2_k3ca": "k3ca",
               "human_motion_data_tier2_g2-2026-10-24": "g2"}  # fmt: skip
FRAME_REFS = {"raw": None, "k3": ME / "tmp_analysis/tier2/kalman/eval/frame_refs_k3",
              "k3fix": ME / "tmp_analysis/tier2/k3fix_pilot/frame_refs_k3fix",
              "k3ca": ME / "tmp_analysis/tier2/k3ca/frame_refs_k3ca", "g2": G2ROOT / "frame_refs_g2"}  # fmt: skip
# round 6 records (frozen; pinned in PREREG_R7 11 and checked here)
R6_PREREG_SHA = "0b0fbf9a2b581913ae0da7d36ce76aa785e706c132dc496054f2323a80e487e1"
SEL6 = R6 / "SELECTION_R6.sha256"
SEL6_SHA = None  # filled at run time from the file and pinned in PREREG_R7 11
R6_POOL, R6_POOL_SHA = R6 / "inputs_r6/pool_r6.json", "c2aeb79af68cda31d8daa9f324bc26189c721150ec781f62a775782403082de6"
R6_RUNS, R6_RUNS_SHA = R6 / "inputs_r6/runs_r6.json", "c7e529e2b7f16fef73e1a42cf8db9e589dc68875f4a0f4aba02d06d2089032d5"
R6_INCS, R6_INCS_SHA = R6 / "inputs_r6/incumbents_r6.json", "44d62d8cbd5b2d4247307b301902fb47dc42032c4227dcbefb970056d9d07b0b"
R6_COPIES, R6_COPIES_SHA = R6 / "COPIES_r6.tsv", "fa17929cf145cbb8333464b559521f699a2712319bd6b9833e017dea92d000f9"
R6_SEL_FILES = ("selection_r6.json", "final_r6.json", "scores_f6/consensus_scores_f6.csv",
                "scores_f6/consensus_job_stats_f6.json", "scores_f6/own_input_gate_f6.csv", "selection_s6.json",
                "PREREG_R6.md")  # fmt: skip  (each must be listed, unchanged, in SELECTION_R6.sha256)
R5_RELOC_TOOL, R5_RELOC_SHA = ME / "scripts/t2cons_r5_relocate.py", "09a6cb8e8a3f8e835c4b10caa37f6ad46b780374480e856cca7efbe479938bdb"
LS_REMOTE = INP / "LS_REMOTE_R7.txt"  # git ls-remote --heads origin of our clone, captured before the writing run


# ================================================================================================ helpers
def guard(p) -> Path:
    s = str(p).lower()
    if any(t in s for t in FORBIDDEN):
        raise PermissionError(f"t2cons_r7_pool may not open {p}")
    return Path(p)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with guard(p).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(p):
    return json.loads(guard(p).read_text())


def read_csv(p, delim: str = ",") -> list[dict]:
    with guard(p).open(newline="") as fp:
        return list(csv.DictReader(fp, delimiter=delim))


def now() -> str:
    return dt.datetime.now(TZ8).strftime("%Y-%m-%d %H:%M:%S")


def header(p: Path) -> dict:
    """safetensors header metadata only (never the tensors)."""
    with guard(p).open("rb") as fp:
        n = struct.unpack("<Q", fp.read(8))[0]
        h = json.loads(fp.read(n))
    md = h.get("__metadata__") or {}
    cfg = json.loads(md.get("user.resolved_config_json") or "{}")
    return {"format": md.get("format"), "iteration": md.get("user.iteration"),
            "parquet": (cfg.get("task") or {}).get("parquet"),
            "reference_parquet": (cfg.get("reference") or {}).get("parquet") if isinstance(cfg.get("reference"), dict) else None,
            "output_dir": cfg.get("output_dir"), "layout": md.get("user.coordinate_layout_sha256")}  # fmt: skip


def parquet_parts(parquet: str | None) -> tuple[Path, str] | None:
    """(data root, sequence id) of <root>/whole_body/soma/sequence_id=<seq>/robot_name=g1/data.parquet, else None."""
    if not parquet:
        return None
    p = Path(parquet)
    try:
        ok = (p.name == "data.parquet" and p.parent.name.startswith("robot_name=") and
              p.parents[1].name.startswith("sequence_id=") and p.parents[2].name == "soma" and p.parents[3].name == "whole_body")
    except IndexError:
        ok = False
    return (p.parents[4], p.parents[1].name.split("=", 1)[1]) if ok else None


def root_kind_of(parquet: str | None) -> tuple[str | None, Path | None, str | None]:
    """(kind, data root, sequence id) of a pro6000 training parquet path, or (None, root, seq)."""
    pp = parquet_parts(parquet)
    if pp is None:
        return None, None, None
    root, seq = pp
    for k, v in ROOTS.items():
        if os.path.realpath(root) == os.path.realpath(v):
            return k, v, seq
    return None, root, seq


def z5_kind_of(parquet: str | None) -> tuple[str | None, str | None, str | None]:
    """(kind, as-trained root name, sequence id) of a 5090 as-trained parquet /mnt/gfs/.../data/<root>/..., or Nones."""
    if not parquet or not parquet.startswith(Z5_DATA):
        return None, None, None
    pp = parquet_parts(parquet)
    if pp is None:
        return None, None, None
    root, seq = pp
    if str(root.parent) + "/" != Z5_DATA:
        return None, None, None
    return Z5_ROOTNAME.get(root.name), root.name, seq


def server_parquet(kind: str, seq: str) -> Path:
    return ROOTS[kind] / "whole_body/soma" / f"sequence_id={seq}" / "robot_name=g1" / "data.parquet"


_MAN: dict[str, dict] = {}


def manifest_episode(root: Path, seq: str) -> int | None:
    key = str(root)
    if key not in _MAN:
        _MAN[key] = read_json(root / "whole_body/soma/track2_manifest.json")
    for s, r in _MAN[key].items():
        if str(r.get("sequence_id", s)) == seq:
            return int(r["episode_index"])
    return None


def ep_frame_ref(kind: str, e: int) -> Path | None:
    fr = FRAME_REFS[kind]
    return None if fr is None else fr / f"episode_{e:06d}.npz"


def picks_of(saved: list[int], L: int) -> tuple[list[int], list[str]]:
    """R4 (round 6, unchanged): final + up to K evenly spaced checkpoints of the second half."""
    half = [i for i in sorted(saved) if i >= L / 2]
    picks, slots = [], []
    for j in range(0, K + 1):
        tau = L * (1 - j / (2 * K))
        best = min(half, key=lambda i: (abs(i - tau), -i))
        if best not in picks:
            picks.append(best)
            slots.append(f"j{j} ({6 - j}/6 L)")
    return picks, slots


# ================================================================================================ round-6 records
def r6_lock() -> dict[str, str]:
    """SELECTION_R6.sha256 (round 6's final lock): every listed file must hold; returns rel path -> sha256."""
    out = {}
    for ln in guard(SEL6).read_text().splitlines():
        f = ln.split(None, 1)
        if len(f) == 2 and len(f[0]) == 64:
            out[f[1].strip()] = f[0]
    bad = []
    for rel, h in out.items():
        p = ME / rel if rel.startswith("scripts/") else R6 / rel
        got = sha256_file(p) if p.is_file() else "missing"
        if got != h:
            bad.append(f"{rel}: {got[:16]} != {h[:16]}")
    missing = [x for x in R6_SEL_FILES if x not in out]
    if bad or missing:
        raise SystemExit(f"SELECTION_R6.sha256 does not hold: {bad} missing {missing}")
    if out.get("PREREG_R6.md") != R6_PREREG_SHA:
        raise SystemExit("SELECTION_R6.sha256 does not name the frozen PREREG_R6.md")
    for p, want in ((R6_POOL, R6_POOL_SHA), (R6_RUNS, R6_RUNS_SHA), (R6_INCS, R6_INCS_SHA), (R6_COPIES, R6_COPIES_SHA)):
        if sha256_file(p) != want:
            raise SystemExit(f"{p} changed since round 6")
    return out


def recheck_export(job: str, rec_path: Path, want_file: str | None = None, want_sha: str | None = None) -> dict:
    ex = read_json(rec_path)
    f = Path(ex["file"])
    fsha = sha256_file(f) if f.is_file() else None
    if ex.get("status") != "ok" or ex.get("stock_verified") is not True or fsha != ex.get("sha256") or fsha != ex.get("destination_sha256"):
        raise SystemExit(f"{job}: export record {rec_path} is not an ok, stock-verified export of an unchanged file")
    if want_file and Path(want_file) != f:
        raise SystemExit(f"{job}: export file {f} != {want_file}")
    if want_sha and fsha != want_sha:
        raise SystemExit(f"{job}: export sha256 {fsha[:16]} != {want_sha[:16]}")
    tck = Path(ex["training_checkpoint"])
    tsha = sha256_file(tck) if tck.is_file() else None
    if tsha != ex.get("training_sha256") or ex.get("source_sha256") != tsha:
        raise SystemExit(f"{job}: training checkpoint {tck} sha256 != export record")
    return ex


def incumbents_and_refs(log) -> tuple[dict[int, dict], dict[int, dict]]:
    """PREREG_R7 4: 23 incumbents (round-6 S6 outside STRICT) + 7 report-only references (round-6 S6 in STRICT)."""
    lock = r6_lock()
    sel = read_json(R6 / "selection_r6.json")
    eps = {f"{int(x['episode']):02d}": x for x in sel["episodes"]}
    s6 = sel["compositions"]["S6"]
    inc6 = read_json(R6_INCS)["incumbents"]
    fin6 = {f["job"]: f for f in read_json(R6 / "final_r6.json")["finalists"]}
    pool6 = {r["job"]: r for r in read_json(R6_POOL)["rows"]}
    f6_csv = R6 / "scores_f6/consensus_scores_f6.csv"
    f6_rows = read_csv(f6_csv)
    f6_stats = read_json(R6 / "scores_f6/consensus_job_stats_f6.json")["jobs"]
    incs, refs = {}, {}
    for e in range(30):
        k = f"{e:02d}"
        job = s6[k]
        epi = eps[k]
        if epi.get("pick") != job:
            raise SystemExit(f"ep{k}: selection_r6.json episodes pick {epi.get('pick')} != S6 {job}")
        if job == inc6[k]["job"]:  # unchanged round-5 incumbent: its round-6 record, re-hashed
            x = dict(inc6[k])
            ex = recheck_export(job, Path(x["export_record"]), x["export_file"], x["export_sha256"])
            if sha256_file(Path(x["export_record"])) != x["export_record_sha256"]:
                raise SystemExit(f"{job}: export record changed since round 6")
            if sha256_file(Path(x["parquet"])) != x["parquet_sha256"]:
                raise SystemExit(f"{job}: training parquet changed since round 6")
            fr = x.get("frame_ref_file")
            if fr and sha256_file(Path(fr)) != x.get("frame_ref_sha256"):
                raise SystemExit(f"{job}: frame reference changed since round 6")
            if sha256_file(Path(x["frozen_scores_csv"])) != x["frozen_scores_csv_sha256"]:
                raise SystemExit(f"{job}: frozen scores csv changed")
            x.update(selected_in=x["source_round"], record_from="inputs_r6/incumbents_r6.json (round-5 incumbent kept in round 6)",
                     r6_closest_run=int(sel["r6_closest_file"][k]["run"]), export_tolerance=ex.get("tolerance"))  # fmt: skip
        else:  # a round-6 replacement (phase-F finalist): its round-6 export and its 10 phase-F consensus runs
            f = fin6.get(job)
            p = pool6.get(job)
            if f is None or p is None or not epi.get("changed"):
                raise SystemExit(f"ep{k}: S6 pick {job} is neither the round-6 incumbent nor a recorded round-6 replacement")
            rec_path = R6 / "exports" / job / "export.json"
            ex = recheck_export(job, rec_path, f["export_file"], f["export_sha256"])
            if ex["training_sha256"] != p["sha256"] or ex["training_sha256"] != f["training_sha256"]:
                raise SystemExit(f"{job}: export training sha256 != pool_r6 / final_r6")
            vals = sorted(((int(r["run"]), float(r["auc"]), float(r["sp_sr"]), float(r["mp_sr"]), float(r["mppe"]))
                           for r in f6_rows if r["job"] == job and r.get("variant") == "B" and r.get("status") == "ok"))  # fmt: skip
            if [v[0] for v in vals] != list(range(1, 11)):
                raise SystemExit(f"{job}: round-6 phase-F variant-B runs {[v[0] for v in vals]} (want 1..10)")
            aucs = [v[1] for v in vals]
            m_p = float(sum(aucs) / len(aucs))
            js = f6_stats[job]
            jsb = js.get("B", js)
            if [float(a) for a in jsb["auc"]] != aucs or abs(float(jsb["auc_stats"]["mean"]) - m_p) > 1e-12:
                raise SystemExit(f"{job}: consensus_job_stats_f6.json differs from consensus_scores_f6.csv")
            if abs(float(epi["m_F"]) - m_p) > 1e-12:
                raise SystemExit(f"{job}: m_F {epi['m_F']} in selection_r6.json != mean of the phase-F runs {m_p}")
            kind = p["input"]
            fr = ep_frame_ref(kind, e)
            if sha256_file(Path(p["parquet"])) != p["parquet_sha256"]:
                raise SystemExit(f"{job}: training parquet changed since round 6")
            if fr and sha256_file(fr) != p.get("frame_ref_sha256"):
                raise SystemExit(f"{job}: frame reference changed since round 6")
            x = {"episode": e, "job": job, "source_round": "round6", "input": kind, "owner": p["owner"], "tree": p["tree"],
                 "iteration": int(p["iteration"]), "export_record": str(rec_path), "export_record_sha256": sha256_file(rec_path),
                 "export_file": ex["file"], "export_sha256": ex["sha256"], "export_tolerance": ex.get("tolerance"),
                 "training_checkpoint": ex["training_checkpoint"], "training_sha256": ex["training_sha256"],
                 "manifest_file": str(R6_POOL), "parquet": p["parquet"], "parquet_sha256": p["parquet_sha256"],
                 "data_root": p["data_root"], "manifest": str(ROOTS[kind] / "whole_body/soma/track2_manifest.json"),
                 "frame_refs": p.get("frame_refs"), "sequence_id": p["sequence_id"], "object": None,
                 "frame_ref_file": str(fr) if fr else None, "frame_ref_sha256": sha256_file(fr) if fr else None,
                 "frozen_runs": [{"run": v[0], "cons_auc": v[1], "cons_sp_sr": v[2], "cons_mp_sr": v[3], "cons_mppe": v[4]} for v in vals],
                 "frozen_scores_csv": str(f6_csv), "frozen_scores_csv_sha256": sha256_file(f6_csv), "m_P_frozen": m_p,
                 "selected_in": "round6", "record_from": "round-6 phase-F replacement (final_r6.json, scores_f6, exports/)",
                 "r6_closest_run": int(sel["r6_closest_file"][k]["run"])}  # fmt: skip
        x["r6_closest_export_sha256"] = sel["r6_closest_file"][k].get("export_sha256")
        if x["r6_closest_export_sha256"] != x["export_sha256"]:
            raise SystemExit(f"ep{k}: R6-CLOSEST export {str(x['r6_closest_export_sha256'])[:16]} != {x['export_sha256'][:16]}")
        if e in STRICT:
            x["role"] = "r6ref"
            x["note"] = ("round-6 pick of a STRICT episode (first nominated by Zuxing's clean re-tests): report-only "
                         "reference in phase F; no incumbent status, never a bar, never a candidate (PREREG_R7 4.2)")
            refs[e] = x
            log(f"r6ref    ep{k}: {job} ({x['source_round']}, {x['input']}) export {x['export_sha256'][:12]} (report only)")
        else:
            x["role"] = "incumbent"
            incs[e] = x
            log(f"incumbent ep{k}: {job} ({x['selected_in']}, {x['input']}, {x['tree']}) export {x['export_sha256'][:12]} "
                f"m_P_frozen {x['m_P_frozen']:.6f} ({len(x['frozen_runs'])} frozen runs)")
    return incs, refs


def r6_screen_by_sha() -> dict[str, dict]:
    out = {}
    for r in read_json(R6_POOL)["rows"]:
        if r.get("role") == "screen":
            out[r["sha256"]] = r
    return out


def r6_verdicts() -> dict[str, dict]:
    """Round-6 phase-F verdicts of round-6 finalists that did NOT replace the incumbent (PREREG_R7 5.4 item 4)."""
    sel = read_json(R6 / "selection_r6.json")
    out = {}
    for epi in sel["episodes"]:
        k = f"{int(epi['episode']):02d}"
        f = epi.get("finalist")
        if f and not epi.get("changed"):
            out[f] = {"episode": int(k), "r6_incumbent": epi.get("incumbent"), "r6_m_F": epi.get("m_F"),
                      "r6_bar": epi.get("bar"), "r6_why_not": epi.get("why_not")}  # fmt: skip
    return out


# ================================================================================================ scan (R1-R3)
def scan_pro6000() -> list[Path]:
    roots = [FC / d for d in OUR_SCAN] + [p for p in sorted((ZU / "runs").iterdir()) if p.is_dir() and p.name not in ZX_SKIP_TOP]
    out = []
    for r in roots:
        try:
            guard(r)
        except PermissionError:
            continue
        for dp, dns, fns in os.walk(r):
            keep = []
            for d in sorted(dns):
                low = d.lower()
                if d in PRUNE_DIRS or low.startswith("eval") or any(t in low for t in FORBIDDEN):
                    continue
                keep.append(d)
            dns[:] = keep
            if any(re.fullmatch(r"model_\d+\.safetensors", f) for f in fns):
                out.append(Path(dp))
    return out


ZX_LABELS = {"2026-10-24-tier2-runin-g2": "zxg2", "2026-10-25-tier2-runin-hard-g2": "zxg2hard",
             "2026-10-24-tier2-runin-k3fix-ep04": "zxk3fixA", "2026-10-24-tier2-runin-k3fix-ep05-ep26": "zxk3fix",
             "2026-10-28-tier2-runin-a5": "zxa5", "2026-11-01-tier2-g2-rest": "zxg2rest"}  # fmt: skip
Z5_LABELS = {"2026-10-23-tier2-helpfix-hard": "z5hfh", "2026-10-24-tier2-runin-g2": "z5g2",
             "2026-10-24-tier2-runin-k3fix": "z5k3fix", "2026-10-25-tier2-runin-hard-g2": "z5g2hard",
             "2026-10-28-tier2-5090-fill": "z5fill", "2026-10-28-tier2-runin-a5": "z5a5",
             "2026-10-29-tier2-5090-queue": "z5q", "2026-10-31-tier2-5090-round2": "z5r2"}  # fmt: skip


def owner_label(d: Path) -> tuple[str, str, str]:
    """(owner, run label used in job names, export tree) by location (round 6's labels kept)."""
    s = str(d)
    if s.startswith(str(Z5) + "/"):
        rel = d.relative_to(Z5).parts
        lab = Z5_LABELS.get(rel[0], "z5_" + re.sub(r"[^a-z0-9]", "", rel[0].lower())[-10:])
        if len(rel) == 3:  # <run>/cont/epNN or <run>/cont2/epNN
            lab += {"cont": "c", "cont2": "c2"}.get(rel[1], "_" + re.sub(r"[^a-z0-9]", "", rel[1].lower()))
        return "zuxing", lab, "b39b69b"
    if s.startswith(str(ZU) + "/"):
        run = d.relative_to(ZU / "runs").parts[0]
        lab = ZX_LABELS.get(run, "zx_" + re.sub(r"[^a-z0-9]", "", run.lower())[-12:])
        return "zuxing", lab, "b39b69b"
    rel = d.relative_to(FC)
    top, leaf = rel.parts[0], rel.parts[-1]
    lab = {"C3r2_tier2": "raw", "C3r2_tier2_k3easy": "k3easy", "C3r2_tier2_k3_stretch90k": "k3str",
           "C3r2_tier2_k3ca": "k3ca"}.get(top)  # fmt: skip
    if top == "C3r2_tier2_k3fix":
        lab = "rawS7" if leaf.endswith("_raw_s7") else "k3fix"
    if top == "T2fix":
        lab = {"T2B": "t2b", "T2L": "t2l"}.get(rel.parts[1], "t2fix_" + rel.parts[1].lower())
    return "haoran", lab or top.lower(), "fc_r2"


def settled_models(d: Path, t_c: float) -> tuple[dict[int, Path], dict[int, Path]]:
    its = {}
    for f in os.listdir(d):
        m = re.fullmatch(r"model_(\d+)\.safetensors", f)
        if m:
            its[int(m.group(1))] = d / f
    settled = {i: p for i, p in its.items() if os.path.getmtime(p) <= t_c - SETTLE_S}
    return its, settled


def run_record(d: Path, t_c: float, r6_ours: set[str]) -> dict:
    """R1-R4 for a pro6000 directory (round 6's run_record + the 'ours from round 6's run list' condition)."""
    its, settled = settled_models(d, t_c)
    rec = {"dir": str(d), "machine": "pro6000", "n_files": len(its), "n_settled": len(settled), "eligible": False, "reason": ""}
    if not settled:
        rec["reason"] = "no checkpoint settled at T_c"
        return rec
    L = max(settled)
    h = header(settled[L])
    kind, droot, seq = root_kind_of(h["parquet"])
    owner, lab, tree = owner_label(d)
    fin = d / "final.json"
    fj = read_json(fin) if fin.is_file() else None
    rec.update(last=L, first=min(settled), format=h["format"], output_dir=h["output_dir"], parquet=h["parquet"], input=kind,
               sequence_id=seq, owner=owner, label=lab, tree=tree, final_json_checkpoint=(fj or {}).get("checkpoint"),
               final_json_reason=(fj or {}).get("reason"), newer_unsettled=sorted(i for i in its if i not in settled))  # fmt: skip
    if not h["output_dir"] or os.path.realpath(h["output_dir"]) != os.path.realpath(d):
        rec["reason"] = f"not a training-run directory (output_dir {h['output_dir']})"
        return rec
    if h["format"] != "learner_state":
        rec["reason"] = f"format {h['format']}"
        return rec
    if kind is None:
        rec["reason"] = f"training parquet not in a Tier-2 data root ({droot})"
        return rec
    if L < MIN_LAST:
        rec["reason"] = f"last settled iteration {L} < {MIN_LAST}"
        return rec
    if owner == "haoran" and str(d) not in r6_ours:
        rec["reason"] = "our run not in round 6's eligible run list (PREREG_R7 2.1 R1)"
        return rec
    e = manifest_episode(ROOTS[kind], seq)
    if e is None:
        rec["reason"] = f"sequence {seq} not in the {kind} manifest"
        return rec
    rec["episode"] = e
    picks, slots = picks_of(list(settled), L)
    rec.update(targets=[round(L * (1 - j / (2 * K)), 1) for j in range(0, K + 1)], picks=picks, slots=slots,
               eligible=True, reason="eligible")  # fmt: skip
    if fj and fj.get("checkpoint") and fj["checkpoint"] != f"model_{L}.safetensors":
        rec["final_json_differs"] = True
    return rec


def z5_manifest() -> list[dict]:
    rows = read_csv(Z5 / "MANIFEST.tsv", "\t")
    if len(rows) != 45:
        raise SystemExit(f"{Z5}/MANIFEST.tsv has {len(rows)} rows (want 45)")
    return rows


def z5_sums() -> dict[str, str]:
    out = {}
    for ln in guard(Z5 / "SHA256SUMS").read_text().splitlines():
        f = ln.split(None, 1)
        if len(f) == 2 and len(f[0]) == 64:
            out[os.path.normpath(f[1].strip())] = f[0]
    return out


def first_train_t2_line(log: Path) -> tuple[str | None, str | None]:
    """The first 'train_t2:' line of a stdout log (at most 200 lines are read) and its commit= field."""
    if not log.is_file():
        return None, None
    with guard(log).open(errors="replace") as fp:
        for _ in range(200):
            ln = fp.readline()
            if not ln:
                break
            if ln.startswith("train_t2:"):
                m = re.search(r"commit=([0-9a-f]{7,40})", ln)
                return ln.rstrip("\n"), (m.group(1) if m else None)
    return None, None


def z5_run_record(row: dict, t_c: float, sums: dict[str, str]) -> dict:
    """R1b-R4 for one 5090 MANIFEST row (PREREG_R7 2.1)."""
    run = row["run"]
    d = Z5 / run
    rec = {"dir": str(d), "machine": "5090", "manifest_row": row, "n_files": 0, "n_settled": 0, "eligible": False, "reason": ""}
    if not d.is_dir():
        rec["reason"] = "directory missing"
        return rec
    its, settled = settled_models(d, t_c)
    rec.update(n_files=len(its), n_settled=len(settled))
    if not settled:
        rec["reason"] = "no checkpoint settled at T_c"
        return rec
    L = max(settled)
    h = header(settled[L])
    kind, rootname, seq = z5_kind_of(h["parquet"])
    owner, lab, tree = owner_label(d)
    fin = d / "final.json"
    fj = read_json(fin) if fin.is_file() else None
    line, commit = first_train_t2_line(d / "train.log")
    rec.update(last=L, first=min(settled), format=h["format"], output_dir=h["output_dir"], parquet=h["parquet"],
               parquet_as_trained=h["parquet"], reference_parquet_as_trained=h["reference_parquet"], input=kind,
               rootname=rootname, sequence_id=seq, owner=owner, label=lab, tree=tree,
               final_json_checkpoint=(fj or {}).get("checkpoint"), final_json_reason=(fj or {}).get("reason"),
               newer_unsettled=sorted(i for i in its if i not in settled), train_t2_line=line, train_log_commit=commit,
               manifest_commit=row.get("commit"), manifest_status=row.get("status"))  # fmt: skip
    if h["output_dir"] != Z5_RUNS + run:
        rec["reason"] = f"output_dir {h['output_dir']} != {Z5_RUNS + run} (R1b)"
        return rec
    if h["format"] != "learner_state":
        rec["reason"] = f"format {h['format']}"
        return rec
    if kind is None:
        rec["reason"] = f"as-trained parquet {h['parquet']} not in a Tier-2-derived data root (R2)"
        return rec
    if h["reference_parquet"] not in (None, h["parquet"]):
        rec["reason"] = f"reference.parquet {h['reference_parquet']} != task.parquet"
        return rec
    if rootname != row.get("data_root") or seq != row.get("sequence"):
        rec["reason"] = f"MANIFEST data_root / sequence {row.get('data_root')} / {row.get('sequence')} != checkpoint {rootname} / {seq}"
        return rec
    sp = server_parquet(kind, seq)
    if not sp.is_file():
        rec["reason"] = f"relocation target {sp} missing"
        return rec
    rec["parquet"] = str(sp)
    if L < MIN_LAST:
        rec["reason"] = f"last settled iteration {L} < {MIN_LAST}"
        return rec
    e = manifest_episode(ROOTS[kind], seq)
    if e is None or e != int(row["episode"]):
        rec["reason"] = f"sequence {seq} -> episode {e} != MANIFEST episode {row['episode']}"
        return rec
    rec["episode"] = e
    picks, slots = picks_of(list(settled), L)
    bad = []
    for it in picks:
        rel = os.path.normpath(f"./{run}/model_{it}.safetensors")
        if rel not in sums:
            bad.append(f"no SHA256SUMS line for {rel}")
    if bad:
        rec["reason"] = "; ".join(bad)
        return rec
    rec.update(targets=[round(L * (1 - j / (2 * K)), 1) for j in range(0, K + 1)], picks=picks, slots=slots,
               eligible=True, reason="eligible (R1b)")  # fmt: skip
    return rec


# ================================================================================================ pool
def job_name(e: int, lab: str, it: int) -> str:
    return f"ep{e:02d}_{lab}_{it}"


def copy_verified(src: Path, dst: Path, write: bool) -> dict:
    before = sha256_file(src)
    st = os.stat(src)
    rec = {"source": str(src), "copy": str(dst), "sha256_source_before": before, "size": st.st_size,
           "source_mtime": dt.datetime.fromtimestamp(st.st_mtime, TZ8).strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip
    if not write:
        return {**rec, "sha256_copy": None, "sha256_source_after": None, "equal": None, "copied_at": None}
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise SystemExit(f"refusing to overwrite {dst}")
    tmp = dst.with_name(dst.name + ".tmp_r7copy")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    os.chmod(dst, 0o444)
    copy = sha256_file(dst)
    after = sha256_file(src)
    return {**rec, "sha256_copy": copy, "sha256_source_after": after, "equal": before == copy == after, "copied_at": now()}


def write_excerpt(src: Path, dst: Path, text: str, write: bool) -> dict:
    """A new file holding one line of a source log (the source is hashed before and after the read)."""
    before = sha256_file(src)
    rec = {"source": f"{src} (first 'train_t2:' line only)", "copy": str(dst), "sha256_source_before": before,
           "size": len(text.encode()) + 1}  # fmt: skip
    if not write:
        return {**rec, "sha256_copy": None, "sha256_source_after": None, "equal": None, "copied_at": None}
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise SystemExit(f"refusing to overwrite {dst}")
    dst.write_text(text + "\n")
    os.chmod(dst, 0o444)
    after = sha256_file(src)
    return {**rec, "sha256_copy": sha256_file(dst), "sha256_source_after": after, "equal": before == after,
            "copied_at": now(), "source_mtime": dt.datetime.fromtimestamp(os.stat(src).st_mtime, TZ8).strftime("%Y-%m-%d %H:%M:%S")}  # fmt: skip


def build_pool(runs: list[dict], incs: dict[int, dict], refs: dict[int, dict], write: bool, log) -> tuple[list[dict], list[dict]]:
    pool, copies = [], []
    seen: dict[str, str] = {}
    inc_sha = {e: x["training_sha256"] for e, x in incs.items()}
    ref_sha = {e: x["training_sha256"] for e, x in refs.items()}
    r6s = r6_screen_by_sha()
    verdicts = r6_verdicts()
    sums = z5_sums()
    order = sorted((r for r in runs if r["eligible"]), key=lambda r: (r["episode"], r["owner"], r["dir"]))
    for r in order:
        d = Path(r["dir"])
        e = r["episode"]
        z5 = r["machine"] == "5090"
        for rank, it in enumerate(r["picks"]):
            src = d / f"model_{it}.safetensors"
            h = header(src)
            sha = sha256_file(src)
            if z5:
                kind, rootname, seq = z5_kind_of(h["parquet"])
                pq = str(server_parquet(kind, seq)) if kind else None
            else:
                kind, _, seq = root_kind_of(h["parquet"])
                pq = h["parquet"]
            job = job_name(e, r["label"], it)
            row = {"job": job, "episode": e, "input": kind, "owner": r["owner"], "machine": r["machine"], "tree": r["tree"],
                   "run_dir": str(d), "run_label": r["label"], "iteration": it, "run_last": r["last"], "slot": r["slots"][rank],
                   "source_checkpoint": str(src), "sha256": sha, "parquet": pq, "sequence_id": seq,
                   "data_root": str(ROOTS[kind]) if kind else None, "strict_episode": e in STRICT}  # fmt: skip
            if z5:
                row.update(parquet_as_trained=h["parquet"], output_dir_as_trained=h["output_dir"],
                           sha256sums_line=sums.get(os.path.normpath(f"./{Path(r['dir']).relative_to(Z5)}/model_{it}.safetensors")))  # fmt: skip
            if kind != r["input"] or seq != r["sequence_id"] or (z5 and row["sha256sums_line"] != sha):
                row["role"] = "refused"
                row["reason"] = (f"checkpoint parquet {h['parquet']} differs from the run's" if kind != r["input"] or seq != r["sequence_id"]
                                 else f"sha256 {sha[:16]} != SHA256SUMS {str(row.get('sha256sums_line'))[:16]}")  # fmt: skip
                pool.append(row)
                continue
            row["parquet_sha256"] = sha256_file(Path(pq))
            fr = ep_frame_ref(kind, e)
            row["frame_refs"] = str(FRAME_REFS[kind]) if FRAME_REFS[kind] else None
            row["frame_ref_file"] = str(fr) if fr else None
            row["frame_ref_sha256"] = sha256_file(fr) if fr and fr.is_file() else None
            prev = r6s.get(sha)
            if inc_sha.get(e) == sha:
                row["role"] = "incumbent"
                row["reason"] = f"== round-7 incumbent {incs[e]['job']} (not screened)"
            elif sha in seen:
                row["role"] = "duplicate"
                row["reason"] = f"same file as {seen[sha]}"
            elif prev is not None and prev["parquet_sha256"] == row["parquet_sha256"] and \
                    prev.get("frame_ref_sha256") == row["frame_ref_sha256"] and int(prev["episode"]) == e:  # fmt: skip
                row["role"] = "reuse"
                row["reason"] = f"== round-6 phase-S candidate {prev['job']}: its round-6 export and 3 runs are reused"
                row["job"] = prev["job"]  # round 6's job name (its exports/ and runs_s/ folders)
                row["r6_job"] = prev["job"]
                row["training_checkpoint"] = prev["training_checkpoint"]
                row["training_sha256"] = sha
                rec_path = R6 / "exports" / prev["job"] / "export.json"
                row["r6_export_record"] = str(rec_path)
                if rec_path.is_file():
                    ex = read_json(rec_path)
                    row["r6_export_status"] = ex.get("status")
                    row["r6_export_record_sha256"] = sha256_file(rec_path)
                    row["r6_export_sha256"] = ex.get("sha256")
                else:
                    row["r6_export_status"] = "missing"
                tck = Path(prev["training_checkpoint"])
                if not tck.is_file() or sha256_file(tck) != sha:
                    raise SystemExit(f"{prev['job']}: round-6 training file {tck} missing or changed")
                runs_s = []
                for k in (1, 2, 3):
                    rj = R6 / "runs_s" / prev["job"] / f"run{k}" / "run.json"
                    runs_s.append({"run": k, "run_json": str(rj), "run_json_sha256": sha256_file(rj) if rj.is_file() else None})
                row["r6_runs"] = runs_s
                if prev["job"] in verdicts:
                    row["r6_phase_f_verdict"] = {**verdicts[prev["job"]], "verdict": "did not replace the incumbent"}
                seen[sha] = row["job"]
            else:
                row["role"] = "screen"
                row["reason"] = "pool rule R1-R4 (R1b for 5090 runs)"
                if prev is not None:
                    row["reason"] += f"; same file as round-6 candidate {prev['job']} but its input changed since round 6: fresh runs"
                if ref_sha.get(e) == sha:
                    row["reason"] += "; == the round-6 pick of this STRICT episode: no incumbent status (PREREG_R7 4.2)"
                seen[sha] = job
            if row["role"] in ("screen", "duplicate", "incumbent") or row["role"] == "refused":
                row.setdefault("training_checkpoint", str(src))
                row.setdefault("training_sha256", sha)
            if r["owner"] == "zuxing" and row["role"] == "screen":
                dst = COPIES / Path(r["dir"]).relative_to(ZU / "runs") / src.name
                c = copy_verified(src, dst, write)
                copies.append(c)
                if write and (not c["equal"] or c["sha256_copy"] != sha):
                    raise SystemExit(f"copy of {src} not verified: {c}")
                rsi = d / f"rsi_{it}.safetensors"
                if rsi.is_file():
                    c2 = copy_verified(rsi, dst.with_name(rsi.name), write)
                    copies.append(c2)
                    row["rsi_sha256"] = c2["sha256_copy"] or c2["sha256_source_before"]
                row["copy"] = str(dst)
                row["copy_of"] = str(src)
                if z5:
                    row["relocated_file"] = str(RELOC_DIR / Path(r["dir"]).relative_to(Z5) / src.name)
                    row["training_checkpoint"] = row["relocated_file"]  # the evaluated input (PREREG_R7 3.1); set by relocate
                    row["training_sha256"] = None  # filled from RELOCATION_R7.tsv (the relocated file's sha256)
                else:
                    row["training_checkpoint"] = str(dst)
                    row["training_sha256"] = sha
            pool.append(row)
    n = {k: sum(1 for x in pool if x["role"] == k) for k in ("screen", "reuse", "incumbent", "duplicate", "refused")}
    log(f"pool: {len(pool)} selected files from {len(order)} eligible runs: {n}")
    return pool, copies


def z5_meta_copies(z5runs: list[dict], write: bool) -> list[dict]:
    """PREREG_R7 1: for every 5090 MANIFEST run: hydra/.hydra/*.yaml, final.json, the first train_t2: line; plus the
    folder's MANIFEST.tsv and SHA256SUMS."""
    out = []
    for name in ("MANIFEST.tsv", "SHA256SUMS"):
        out.append(copy_verified(Z5 / name, COPIES / Z5.name / name, write))
    for r in z5runs:
        d = Path(r["dir"])
        rel = d.relative_to(ZU / "runs")
        for sub in ("hydra/.hydra/config.yaml", "hydra/.hydra/hydra.yaml", "hydra/.hydra/overrides.yaml", "final.json"):
            src = d / sub
            if src.is_file():
                out.append(copy_verified(src, COPIES / rel / sub, write))
        if r.get("train_t2_line"):
            out.append(write_excerpt(d / "train.log", COPIES / rel / "train_log_first_train_t2_line.txt", r["train_t2_line"], write))
    return out


# ================================================================================================ outputs
def csv_text(rows: list[dict], cols: list[str], delim: str = ",") -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", delimiter=delim, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({c: (json.dumps(r[c]) if isinstance(r.get(c), (list, dict)) else r.get(c)) for c in cols})
    return buf.getvalue()


def write_new(p: Path, text: str) -> None:
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp_r7p")
    tmp.write_text(text)
    os.replace(tmp, p)
    os.chmod(p, 0o444)


POOL_COLS = ["job", "episode", "strict_episode", "input", "owner", "machine", "tree", "role", "reason", "run_label", "run_dir",
             "iteration", "run_last", "slot", "sha256", "source_checkpoint", "training_checkpoint", "training_sha256", "copy",
             "copy_of", "relocated_file", "rsi_sha256", "parquet", "parquet_sha256", "parquet_as_trained", "sequence_id",
             "data_root", "frame_refs", "frame_ref_file", "frame_ref_sha256", "r6_job", "r6_export_status", "r6_export_sha256",
             "r6_phase_f_verdict"]  # fmt: skip
RUN_COLS = ["dir", "machine", "owner", "label", "tree", "input", "episode", "eligible", "reason", "first", "last", "n_files",
            "n_settled", "newer_unsettled", "targets", "picks", "slots", "final_json_checkpoint", "final_json_reason",
            "manifest_status", "manifest_commit", "train_log_commit", "output_dir", "parquet", "parquet_as_trained"]  # fmt: skip


def zuxing_code_rows(runs: list[dict], pool: list[dict]) -> list[dict]:
    """The training commit of each of Zuxing's eligible runs (pro6000: first train_t2: line of logs/<run>/ep<NN>.log;
    5090: first train_t2: line of train.log == MANIFEST commit) and whether it is a pushed head (LS_REMOTE_R7.txt)."""
    remote = {}
    if LS_REMOTE.is_file():
        for ln in LS_REMOTE.read_text().splitlines():
            f = ln.split()
            if len(f) == 2:
                remote.setdefault(f[0], []).append(f[1].replace("refs/heads/", ""))
    out = []
    for r in sorted((r for r in runs if r["eligible"] and r["owner"] == "zuxing"), key=lambda r: r["dir"]):
        d = Path(r["dir"])
        if r["machine"] == "5090":
            commit = r.get("train_log_commit")
            agree = commit == r.get("manifest_commit")
            where = f"5090 ({d / 'train.log'} first train_t2: line; MANIFEST commit {r.get('manifest_commit')}{'' if agree else ' DIFFERS'})"
        else:
            rel = d.relative_to(ZU / "runs")
            _, commit = first_train_t2_line(ZU / "logs" / rel.parts[0] / f"{rel.parts[-1]}.log")
            where = f"pro6000 worktree (log {ZU / 'logs' / rel.parts[0] / (rel.parts[-1] + '.log')})"
        heads = [b for full, bs in remote.items() for b in bs if commit and full.startswith(commit)]
        out.append({"run": str(d.relative_to(ZU / "runs")), "machine": r["machine"], "episode": r["episode"],
                    "screened_files": sum(1 for p in pool if p["run_dir"] == r["dir"] and p["role"] == "screen"),
                    "reused_files": sum(1 for p in pool if p["run_dir"] == r["dir"] and p["role"] == "reuse"),
                    "commit": commit, "pushed_heads": " ".join(heads) or "NOT a pushed head", "where": where})  # fmt: skip
    return out


def resolve(write: bool) -> int:
    t_c = time.time()
    t_c_text = dt.datetime.fromtimestamp(t_c, TZ8).strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []

    def log(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    if write and (INP / "pool_r7.json").exists():
        raise SystemExit("inputs_r7/pool_r7.json exists: the round-7 pool is already resolved")
    log(f"t2cons_r7_pool {'resolve' if write else 'check'} at T_c = {t_c_text} CST (K = {K}, min last iteration {MIN_LAST}, "
        f"settle {SETTLE_S:.0f} s, STRICT {','.join(f'{e:02d}' for e in STRICT)})")
    me = Path(__file__).resolve()
    log(f"script sha256 {sha256_file(me)}")
    incs, refs = incumbents_and_refs(log)
    r6_ours = {r["dir"] for r in read_json(R6_RUNS)["runs"] if r.get("eligible") and r.get("owner") == "haoran"}
    dirs = scan_pro6000()
    runs = [run_record(d, t_c, r6_ours) for d in dirs]
    z5rows = z5_manifest()
    sums = z5_sums()
    z5runs = [z5_run_record(row, t_c, sums) for row in z5rows]
    runs += z5runs
    elig = [r for r in runs if r["eligible"]]
    log(f"scan: {len(dirs)} pro6000 directories with model files + {len(z5runs)} 5090 MANIFEST runs; {len(elig)} eligible "
        f"training runs ({sum(1 for r in elig if r['owner'] == 'haoran')} ours, "
        f"{sum(1 for r in elig if r['owner'] == 'zuxing' and r['machine'] == 'pro6000')} Zuxing pro6000, "
        f"{sum(1 for r in elig if r['machine'] == '5090')} Zuxing 5090)")
    missing_ours = sorted(r6_ours - {r["dir"] for r in elig if r["owner"] == "haoran"})
    if missing_ours:
        log(f"WARNING: round-6 eligible runs of ours not eligible now: {missing_ours}")
    for r in sorted(runs, key=lambda r: (not r["eligible"], r.get("episode", 99), r["dir"])):
        if r["eligible"]:
            base = ZU / "runs" if r["owner"] == "zuxing" else ME
            log(f"  RUN ep{r['episode']:02d} {r['owner']:<6} {r['machine']:<7} {r['label']:<10} {r['input']:<5} L={r['last']:<7} "
                f"picks {r['picks']} ({Path(r['dir']).relative_to(base)})")
    for r in z5runs:
        if not r["eligible"]:
            log(f"  5090 NOT eligible: {r['dir']}: {r['reason']}")
    pool, copies = build_pool(runs, incs, refs, write, log)
    names = [p["job"] for p in pool]
    dup_names = sorted({n for n in names if names.count(n) > 1})
    if dup_names:
        raise SystemExit(f"job names not unique: {dup_names}")
    per_ep = {e: sum(1 for p in pool if p["episode"] == e and p["role"] in ("screen", "reuse")) for e in range(30)}
    per_ep_new = {e: sum(1 for p in pool if p["episode"] == e and p["role"] == "screen") for e in range(30)}
    log("candidates per episode (screen + reuse): " + " ".join(f"{e:02d}:{n}" for e, n in per_ep.items()))
    log("new screen candidates per episode:       " + " ".join(f"{e:02d}:{n}" for e, n in per_ep_new.items()))
    copies += z5_meta_copies(z5runs, write)
    zrows = zuxing_code_rows(runs, pool)
    if not write:
        log("check only: nothing written")
        return 0
    bad = [c for c in copies if c.get("equal") is not True]
    if bad:
        raise SystemExit(f"unverified copies: {bad[:3]}")
    INP.mkdir(parents=True, exist_ok=True)
    meta = {"t_c": t_c_text + " CST", "t_c_epoch": t_c, "script": str(me), "script_sha256": sha256_file(me), "K": K,
            "min_last_iteration": MIN_LAST, "settle_seconds": SETTLE_S, "strict_episodes": list(STRICT),
            "rule": __doc__.split("Pool rule", 1)[1].split("Zuxing's new screen", 1)[0]}  # fmt: skip
    write_new(INP / "pool_r7.json", json.dumps({**meta, "rows": pool}, indent=1, default=str) + "\n")
    write_new(INP / "pool_r7.csv", csv_text(pool, POOL_COLS))
    write_new(INP / "runs_r7.json", json.dumps({**meta, "runs": runs}, indent=1, default=str) + "\n")
    write_new(INP / "runs_r7.csv", csv_text(runs, RUN_COLS))
    write_new(INP / "incumbents_r7.json", json.dumps({**meta, "source": "round-6 S6 (selection_r6.json, SELECTION_R6.sha256) "
                                                     "outside STRICT", "incumbents": {f"{e:02d}": x for e, x in incs.items()}},
                                                    indent=1) + "\n")  # fmt: skip
    write_new(INP / "r6refs_r7.json", json.dumps({**meta, "source": "round-6 S6 picks of the STRICT episodes (report only)",
                                                 "r6refs": {f"{e:02d}": x for e, x in refs.items()}}, indent=1) + "\n")  # fmt: skip
    write_new(INP / "ZUXING_CODE_R7.tsv", csv_text(zrows, ["run", "machine", "episode", "screened_files", "reused_files", "commit",
                                                           "pushed_heads", "where"], "\t"))  # fmt: skip
    ccols = ["source", "copy", "sha256_source_before", "sha256_copy", "sha256_source_after", "equal", "size", "source_mtime", "copied_at"]
    write_new(ROOT / "COPIES_r7.tsv", csv_text(copies, ccols, "\t"))
    tell = ["Zuxing: round 7 (STRICT) of the Tier-2 noisy-only selection (t2_consensus_r7_20261102) made sha256-verified",
            f"READ-ONLY copies of these files of your runs at {t_c_text} CST (nothing in your folder was written; no git",
            "command; none of your scripts was run). Of each train.log only the first 'train_t2:' line was read and copied.",
            "Each line: source -> copy (sha256).", ""]  # fmt: skip
    tell += [f"{c['source']} -> {c['copy']} ({c['sha256_copy']})" for c in copies]
    tell += ["", "Open items for the submission commit (inputs_r7/ZUXING_CODE_R7.tsv): the 5090 runs were trained at 82837c8",
             "(dev.zuxing.hard_start) per their train.log / MANIFEST; whether the worktrees had uncommitted changes; the G2",
             "builder (tmp/t2/g2data/) and the 2026-11-01 G2 rest build are in no pushed commit.", ""]  # fmt: skip
    write_new(ROOT / "TELL_ZUXING_r7.txt", "\n".join(tell) + "\n")
    tracks: dict[str, str] = {}
    for p in pool:
        if p["role"] in ("screen", "reuse", "incumbent"):
            tracks[p["parquet"]] = p["parquet_sha256"]
            if p.get("frame_ref_file"):
                tracks[p["frame_ref_file"]] = p["frame_ref_sha256"]
    for x in list(incs.values()) + list(refs.values()):
        tracks[x["parquet"]] = x["parquet_sha256"]
        if x.get("frame_ref_file"):
            tracks[x["frame_ref_file"]] = x["frame_ref_sha256"]
    for k, v in ROOTS.items():
        m = v / "whole_body/soma/track2_manifest.json"
        tracks[str(m)] = sha256_file(m)
    write_new(INP / "TRACKS_R7.sha256", "".join(f"{h}  {p}\n" for p, h in sorted(tracks.items())))
    write_new(INP / "resolve_r7.log", "\n".join(lines) + "\n")
    snap = [INP / n for n in ("pool_r7.json", "pool_r7.csv", "runs_r7.json", "runs_r7.csv", "incumbents_r7.json",
                              "r6refs_r7.json", "ZUXING_CODE_R7.tsv", "TRACKS_R7.sha256", "resolve_r7.log")]  # fmt: skip
    snap += [LS_REMOTE] if LS_REMOTE.is_file() else []
    snap += [ROOT / "COPIES_r7.tsv", ROOT / "TELL_ZUXING_r7.txt"] + sorted(p for p in COPIES.rglob("*") if p.is_file())
    write_new(INP / "SNAPSHOT_R7.sha256", "".join(f"{sha256_file(p)}  {p.relative_to(ROOT)}\n" for p in snap))
    print(f"written: {INP} (SNAPSHOT_R7.sha256 {sha256_file(INP / 'SNAPSHOT_R7.sha256')})")
    return 0


# ================================================================================================ relocation (5090)
def load_rt():
    """Round 5's relocation tool compiled from its pinned bytes; only relocate_bytes and verify_pair are used (the
    rounds 2-6 two-field edit: task.parquet and reference.parquet; weights and every other byte of metadata unchanged)."""
    src = R5_RELOC_TOOL.read_bytes()
    got = hashlib.sha256(src).hexdigest()
    if got != R5_RELOC_SHA:
        raise SystemExit(f"{R5_RELOC_TOOL} sha256 {got} != pinned {R5_RELOC_SHA}")
    mod = types.ModuleType("t2cons_r5_relocate_readonly")
    mod.__file__ = str(R5_RELOC_TOOL)
    sys.modules[mod.__name__] = mod
    exec(compile(src, str(R5_RELOC_TOOL), "exec"), mod.__dict__)  # noqa: S102 (pinned bytes)
    return mod


RTSV_COLS = ["job", "episode", "iteration", "input", "status", "original_copy", "original_sha256", "sha256sums_line", "relocated_file",
             "relocated_sha256", "fields_rewritten", "old_parquet_path", "new_parquet_path", "parquet_sha256_new_path",
             "n_tensors", "tensors_identical", "safe_open_identical", "data_block_sha256", "other_metadata_identical",
             "config_diff_paths", "header_len_old", "header_len_new", "problems", "written_at"]  # fmt: skip


def verify_row(RT, row: dict) -> list[str]:
    """Read-only re-check of one RELOCATION_R7 row (as round 5's verify_record, on round-7 paths)."""
    problems = []
    orig, reloc = Path(row["original_copy"]), Path(row["relocated_file"])
    if not str(orig).startswith(str(COPIES) + "/") or not str(reloc).startswith(str(RELOC_DIR) + "/"):
        return [f"unexpected paths {orig} / {reloc}"]
    if not orig.is_file() or sha256_file(orig) != row["original_sha256"]:
        problems.append(f"original copy {orig} missing or != {row['original_sha256'][:16]}")
    if not reloc.is_file() or sha256_file(reloc) != row["relocated_sha256"]:
        problems.append(f"relocated file {reloc} missing or != {row['relocated_sha256'][:16]}")
    if problems:
        return problems
    v = RT.verify_pair(orig, reloc, row["old_parquet_path"], row["new_parquet_path"])
    problems += v["problems"]
    newp = Path(row["new_parquet_path"])
    if not newp.is_file() or sha256_file(newp) != row["parquet_sha256_new_path"]:
        problems.append(f"{newp} missing or sha256 != {row['parquet_sha256_new_path'][:16]}")
    return problems


def relocate(a) -> int:
    if RELOC_TSV.exists():
        raise SystemExit("RELOCATION_R7.tsv exists")
    snap = INP / "SNAPSHOT_R7.sha256"
    for ln in guard(snap).read_text().splitlines():
        f = ln.split(None, 1)
        if len(f) == 2 and sha256_file(ROOT / f[1].strip()) != f[0]:
            raise SystemExit(f"SNAPSHOT_R7: {f[1]} changed")
    RT = load_rt()
    pool = read_json(INP / "pool_r7.json")["rows"]
    rows, out = [], []
    for p in pool:
        if p["role"] != "screen" or p["machine"] != "5090":
            continue
        orig = Path(p["copy"])
        dst = Path(p["relocated_file"])
        osha = sha256_file(orig)
        problems = [] if osha == p["sha256"] == p.get("sha256sums_line") else ["original copy sha256 != pool / SHA256SUMS"]
        raw = guard(orig).read_bytes()
        new, info = RT.relocate_bytes(raw, p["parquet"], p["parquet_as_trained"])
        if dst.exists():
            raise SystemExit(f"refusing to overwrite {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp_r7reloc")
        tmp.write_bytes(new)
        os.replace(tmp, dst)
        os.chmod(dst, 0o444)
        v = RT.verify_pair(orig, dst, p["parquet_as_trained"], p["parquet"])
        problems += v["problems"]
        if sha256_file(orig) != osha:
            problems.append("original copy changed during relocation")
        psha = sha256_file(Path(p["parquet"]))
        if psha != p["parquet_sha256"]:
            problems.append("new parquet sha256 != pool")
        row = {"job": p["job"], "episode": p["episode"], "iteration": p["iteration"], "input": p["input"],
               "original_copy": str(orig), "original_sha256": osha, "sha256sums_line": p.get("sha256sums_line"),
               "relocated_file": str(dst), "relocated_sha256": sha256_file(dst), "fields_rewritten": "task.parquet,reference.parquet",
               "old_parquet_path": p["parquet_as_trained"], "new_parquet_path": p["parquet"], "parquet_sha256_new_path": psha,
               "n_tensors": v.get("n_tensors"), "tensors_identical": v.get("tensors_identical"),
               "safe_open_identical": v.get("safe_open_identical"), "data_block_sha256": v.get("data_block_sha256"),
               "other_metadata_identical": v.get("other_metadata_identical"), "config_diff_paths": v.get("config_diff_paths"),
               "header_len_old": info["header_len_old"], "header_len_new": info["header_len_new"],
               "problems": "; ".join(problems), "status": "ok" if not problems else "failed", "written_at": now()}  # fmt: skip
        rows.append(row)
        out.append({**row, "verify": {k: v.get(k) for k in v if k != "problems"}})
        print(f"{p['job']}: {'ok' if not problems else 'FAILED ' + '; '.join(problems)} -> {dst.name} {row['relocated_sha256'][:12]}")
    bad = [r for r in rows if r["status"] != "ok"]
    write_new(RELOC_TSV, csv_text(rows, RTSV_COLS, "\t"))
    write_new(RELOC_JSON, json.dumps({"written_at": now(), "tool": str(Path(__file__).resolve()),
                                      "tool_sha256": sha256_file(Path(__file__).resolve()), "relocate_code": str(R5_RELOC_TOOL),
                                      "relocate_code_sha256": R5_RELOC_SHA, "rows": out}, indent=1, default=str) + "\n")  # fmt: skip
    lst = [RELOC_TSV, RELOC_JSON] + [Path(r["relocated_file"]) for r in rows]
    write_new(RELOC_SHA, "".join(f"{sha256_file(p)}  {p.relative_to(ROOT)}\n" for p in lst))
    print(f"{len(rows)} relocated, {len(bad)} failed; RELOCATION_R7.sha256 {sha256_file(RELOC_SHA)}")
    return 0 if not bad else 1


def verify(a) -> int:
    RT = load_rt()
    rows = read_csv(RELOC_TSV, "\t")
    bad = 0
    for r in rows:
        pr = verify_row(RT, r)
        bad += bool(pr)
        if pr:
            print(f"{r['job']}: {pr}")
    print(f"{len(rows)} relocation rows verified, {bad} with problems")
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("check", "resolve", "relocate", "verify"))
    a = ap.parse_args()
    if a.mode in ("check", "resolve"):
        return resolve(a.mode == "resolve")
    if a.mode == "relocate":
        return relocate(a)
    return verify(a)


if __name__ == "__main__":
    raise SystemExit(main())
