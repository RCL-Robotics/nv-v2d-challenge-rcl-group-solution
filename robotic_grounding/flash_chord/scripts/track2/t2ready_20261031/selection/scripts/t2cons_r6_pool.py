#!/usr/bin/env python3
"""t2cons_r6_pool.py: step P of fc_runs/submissions/t2_consensus_r6_20261031 (round 6, PREREG_R6 sections 1-2).

The MECHANICAL, clean-free candidate pool of round 6 and the frozen round-5 incumbents, resolved once at the cut-off T_c
(= the start of the writing run). Nothing here looks at any score: no clean file, no evaluation result, no metrics.csv,
no best_weights / seed_best / README / training log file is opened (path guard on every path; directories with such
names are pruned before they are entered; Zuxing's eval*/ and logs/ folders are never entered).

Pool rule (PREREG_R6 2.1):
  R1  a TRAINING-RUN directory = a directory under the scanned roots holding model_<N>.safetensors files whose embedded
      config output_dir resolves to that same directory (the trainer wrote them there; copies, forks, bundles, retest
      and eval folders are not training-run directories);
  R2  its training parquet (embedded task.parquet of the last checkpoint) lies in a Tier-2-derived data root: raw
      Tier 2, K3, K3fix, K3CA (ours) or G2 (Zuxing's); Tier-1 / Tier-3 runs are not eligible;
  R3  only checkpoints already written at T_c count (file mtime <= T_c - 300 s); L = the largest such iteration; a run
      with L < 10000 is not eligible (smoke / aborted runs);
  R4  per run, targets tau_j = L * (1 - j / (2K)), j = 0..K, K = 3 (100 %, 83.3 %, 66.7 %, 50 % of L); for each target
      the saved iteration >= L/2 nearest to it (ties: the larger); duplicates collapse (so 1..4 checkpoints per run;
      j = 0 is the run's last checkpoint at T_c, == its final.json checkpoint for every finished run);
  R5  a selected file whose sha256 equals the round-5 incumbent's training file of that episode is the incumbent (not
      screened); a sha256 seen earlier in the deterministic run order is a duplicate (not screened again).
Zuxing's selected files (model + rsi of the same iteration) are COPIED into zuxing_copies_r6/ with sha256 before ==
copy == after; his folder is only read (no write, no git command, none of his scripts run).

Incumbents (PREREG_R6 4): the round-5 composition ROUND5 of selection_r5.json (pinned through SELECTION_R5.sha256),
each with its existing export record (the earlier round's exports/<job>/export.json, re-hashed), its manifest row and
its frozen variant-B consensus AUCs from the round that selected it (round 1: 5 runs; rounds 3 / 5: 10 runs) -> m_P.

  conda_envs/flashchord/bin/python -I -B scripts/t2cons_r6_pool.py check     plan only, writes nothing
  conda_envs/flashchord/bin/python -I -B scripts/t2cons_r6_pool.py resolve   the writing run (T_c); refuses if
                                                                                    inputs_r6/pool_r6.json exists
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
from pathlib import Path  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
ZU = Path("/root/autodl-tmp/zuxing.lu")
FC = ME / "fc_runs"
SUB = FC / "submissions"
ROOT = SUB / "t2_consensus_r6_20261031"
R1, B5 = SUB / "t2_consensus_20261024", SUB / "t2_best5_20261024"
R2, R3, R4, R5 = (SUB / "t2_consensus_r2_20261027", SUB / "t2_consensus_r3_20261028", SUB / "t2_consensus_r4_20261028",
                  SUB / "t2_consensus_r5_20261029")  # fmt: skip
INP = ROOT / "inputs_r6"
COPIES = ROOT / "zuxing_copies_r6"
TZ8 = dt.timezone(dt.timedelta(hours=8))
K = 3
MIN_LAST = 10000
SETTLE_S = 300.0
FORBIDDEN = ("clean", "local_not_for_upload", "packed_", "best_weights", "report_r", "composites", "per_episode_zx",
             "readme", "train_rl.log", "t2_zx07_20261028", "seed_best", "remote_reps", "tier_1_multiview_caption")  # fmt: skip
PRUNE_DIRS = ("hydra", ".hydra", "wandb", "fc_cache", "figures", "logs")  # never entered (plus eval* below)
OUR_SCAN = ("C3r2_tier2", "C3r2_tier2_armA_90k", "C3r2_tier2_k3_stretch90k", "C3r2_tier2_k3ca", "C3r2_tier2_k3easy",
            "C3r2_tier2_k3fix", "T2fix")  # fmt: skip
G2ROOT = ZU / "data/human_motion_data_tier2_g2-2026-10-24"
ROOTS = {"raw": ME / "data/human_motion_data_tier2", "k3": ME / "data/human_motion_data_tier2_k3",
         "k3fix": ME / "data/human_motion_data_tier2_k3fix", "k3ca": ME / "data/human_motion_data_tier2_k3ca",
         "g2": G2ROOT}  # fmt: skip
FRAME_REFS = {"raw": None, "k3": ME / "tmp_analysis/tier2/kalman/eval/frame_refs_k3",
              "k3fix": ME / "tmp_analysis/tier2/k3fix_pilot/frame_refs_k3fix",
              "k3ca": ME / "tmp_analysis/tier2/k3ca/frame_refs_k3ca", "g2": G2ROOT / "frame_refs_g2"}  # fmt: skip
SEL5 = R5 / "SELECTION_R5.sha256"
SEL5_SHA = "58e8b54b"  # prefix recorded in the project notes; the full value is checked against the file below
EXPORT_ROOTS = (R5 / "exports", R4 / "exports", R3 / "exports", R2 / "exports_b", R2 / "exports", R1 / "exports",
                B5 / "exports")  # fmt: skip
MANIFESTS = (R5 / "manifest_r5.json", R4 / "manifest_r4.json", R3 / "manifest_r3.json", R2 / "manifest_r2b.json",
             R2 / "manifest_r2.json", R1 / "manifest_a1.json", R1 / "manifest_new.json", B5 / "manifest.json",
             B5 / "manifest_stageB.jsonl")  # fmt: skip
FROZEN_SCORES = {"round1": (R1 / "scores/consensus_scores.csv", R1 / "scores/consensus_job_stats.json", 5),
                 "round2": (R2 / "scores_r2/consensus_scores_r2.csv", R2 / "scores_r2/consensus_job_stats_r2.json", 10),
                 "round3": (R3 / "scores_r3/consensus_scores_r3.csv", R3 / "scores_r3/consensus_job_stats_r3.json", 10),
                 "round5": (R5 / "scores_r5/consensus_scores_r5.csv", R5 / "scores_r5/consensus_job_stats_r5.json", 10)}  # fmt: skip
LS_REMOTE = INP / "LS_REMOTE_R6.txt"  # git ls-remote origin output, captured before the writing run (our clone)


# ================================================================================================ helpers
def guard(p) -> Path:
    s = str(p).lower()
    if any(t in s for t in FORBIDDEN):
        raise PermissionError(f"t2cons_r6_pool may not open {p}")
    return Path(p)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with guard(p).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(p):
    return json.loads(guard(p).read_text())


def read_csv(p) -> list[dict]:
    with guard(p).open(newline="") as fp:
        return list(csv.DictReader(fp))


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


def root_kind_of(parquet: str | None) -> tuple[str | None, Path | None, str | None]:
    """(kind, data root, sequence id) of a training parquet path, or (None, None, None)."""
    if not parquet:
        return None, None, None
    p = Path(parquet)
    try:
        ok = (p.name == "data.parquet" and p.parent.name.startswith("robot_name=") and
              p.parents[1].name.startswith("sequence_id=") and p.parents[2].name == "soma" and p.parents[3].name == "whole_body")
    except IndexError:
        ok = False
    if not ok:
        return None, None, None
    root = p.parents[4]
    for k, v in ROOTS.items():
        if os.path.realpath(root) == os.path.realpath(v):
            return k, v, p.parents[1].name.split("=", 1)[1]
    return None, root, p.parents[1].name.split("=", 1)[1]


def manifest_episode(root: Path, seq: str) -> int | None:
    m = read_json(root / "whole_body/soma/track2_manifest.json")
    for s, r in m.items():
        if str(r.get("sequence_id", s)) == seq:
            return int(r["episode_index"])
    return None


def ep_frame_ref(kind: str, e: int) -> Path | None:
    fr = FRAME_REFS[kind]
    return None if fr is None else fr / f"episode_{e:06d}.npz"


# ================================================================================================ scan (R1-R3)
def scan_dirs() -> list[Path]:
    roots = [FC / d for d in OUR_SCAN] + [p for p in sorted((ZU / "runs").iterdir()) if p.is_dir()]
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


def owner_label(d: Path) -> tuple[str, str, str]:
    """(owner, run label used in job names, tree) by location; the training run's identity is its directory."""
    s = str(d)
    if s.startswith(str(ZU) + "/"):
        rel = d.relative_to(ZU / "runs")
        run = rel.parts[0]
        lab = {"2026-10-24-tier2-runin-g2": "zxg2", "2026-10-25-tier2-runin-hard-g2": "zxg2hard",
               "2026-10-24-tier2-runin-k3fix-ep04": "zxk3fixA", "2026-10-24-tier2-runin-k3fix-ep05-ep26": "zxk3fix",
               "2026-10-28-tier2-runin-a5": "zxa5"}.get(run, "zx_" + re.sub(r"[^a-z0-9]", "", run.lower())[-12:])  # fmt: skip
        return "zuxing", lab, "b39b69b"
    rel = d.relative_to(FC)
    top = rel.parts[0]
    leaf = rel.parts[-1]
    lab = {"C3r2_tier2": "raw", "C3r2_tier2_k3easy": "k3easy", "C3r2_tier2_k3_stretch90k": "k3str",
           "C3r2_tier2_k3ca": "k3ca"}.get(top)  # fmt: skip
    if top == "C3r2_tier2_k3fix":
        lab = "rawS7" if leaf.endswith("_raw_s7") else "k3fix"
    if top == "T2fix":
        lab = {"T2B": "t2b", "T2L": "t2l"}.get(rel.parts[1], "t2fix_" + rel.parts[1].lower())
    return "haoran", lab or top.lower(), "fc_r2"


def run_record(d: Path, t_c: float) -> dict:
    its = {}
    for f in os.listdir(d):
        m = re.fullmatch(r"model_(\d+)\.safetensors", f)
        if m:
            its[int(m.group(1))] = d / f
    settled = {i: p for i, p in its.items() if os.path.getmtime(p) <= t_c - SETTLE_S}
    rec = {"dir": str(d), "n_files": len(its), "n_settled": len(settled), "eligible": False, "reason": ""}
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
               sequence_id=seq, owner=owner, label=lab, tree=tree,
               final_json_checkpoint=(fj or {}).get("checkpoint"), final_json_reason=(fj or {}).get("reason"),
               newer_unsettled=sorted(i for i in its if i not in settled))  # fmt: skip
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
    e = manifest_episode(ROOTS[kind], seq)
    if e is None:
        rec["reason"] = f"sequence {seq} not in the {kind} manifest"
        return rec
    rec["episode"] = e
    # R4: final + K evenly spaced in the second half
    half = [i for i in sorted(settled) if i >= L / 2]
    picks, slots = [], []
    for j in range(0, K + 1):
        tau = L * (1 - j / (2 * K))
        best = min(half, key=lambda i: (abs(i - tau), -i))
        if best not in picks:
            picks.append(best)
            slots.append(f"j{j} ({6 - j}/6 L)" if K == 3 else f"j{j}")
    if fj and fj.get("checkpoint") and fj["checkpoint"] != f"model_{L}.safetensors":
        rec["final_json_differs"] = True
    rec["targets"] = [round(L * (1 - j / (2 * K)), 1) for j in range(0, K + 1)]
    rec["picks"] = picks
    rec["slots"] = slots
    rec["eligible"] = True
    rec["reason"] = "eligible"
    return rec


# ================================================================================================ incumbents
def sel5() -> dict:
    lines = [ln.split(None, 1) for ln in guard(SEL5).read_text().splitlines() if ln.strip()]
    want = {p.strip(): h for h, p in lines if len(h) == 64}
    sel = R5 / "selection_r5.json"
    got = sha256_file(sel)
    hit = [h for p, h in want.items() if p.endswith("selection_r5.json")]
    if not hit or hit[0] != got:
        raise SystemExit(f"selection_r5.json sha256 {got[:16]} is not the one listed in SELECTION_R5.sha256")
    if not sha256_file(SEL5).startswith(SEL5_SHA):
        raise SystemExit("SELECTION_R5.sha256 is not the frozen round-5 lock (58e8b54b...)")
    return read_json(sel)


def manifest_rows() -> dict[str, list[tuple[str, dict]]]:
    out: dict[str, list[tuple[str, dict]]] = {}
    for m in MANIFESTS:
        if not m.is_file():
            continue
        rows = [json.loads(x) for x in guard(m).read_text().splitlines() if x.strip()] if m.suffix == ".jsonl" else read_json(m)["rows"]
        for r in rows:
            out.setdefault(r["job"], []).append((str(m), r))
    return out


def incumbents(log) -> dict[int, dict]:
    s5 = sel5()
    comp = s5["compositions"]["ROUND5"]
    closest = s5["r5_closest_file"]
    mrows = manifest_rows()
    out = {}
    for e in range(30):
        k = f"{e:02d}"
        job = comp[k]
        src = closest[k]["source"]
        if closest[k]["job"] != job:
            raise SystemExit(f"ep{k}: R5-CLOSEST job {closest[k]['job']} != ROUND5 {job}")
        exps = [d / job / "export.json" for d in EXPORT_ROOTS if (d / job / "export.json").is_file()]
        if len(exps) != 1:
            raise SystemExit(f"{job}: {len(exps)} export records {exps}")
        ex = read_json(exps[0])
        f = Path(ex["file"])
        fsha = sha256_file(f) if f.is_file() else None
        if ex.get("status") != "ok" or ex.get("stock_verified") is not True or fsha != ex.get("sha256") or fsha != ex.get("destination_sha256"):
            raise SystemExit(f"{job}: export record {exps[0]} is not an ok, stock-verified export of an unchanged file")
        if closest[k].get("export_sha256") and closest[k]["export_sha256"] != fsha:
            raise SystemExit(f"{job}: R5-CLOSEST export sha256 differs from the export record")
        rows = [(m, r) for m, r in mrows.get(job, []) if r.get("status") == "ok"]
        rows = [(m, r) for m, r in rows if r.get("training_sha256") == ex.get("training_sha256")]
        if not rows:
            raise SystemExit(f"{job}: no ok manifest row with training sha256 {str(ex.get('training_sha256'))[:16]}")
        mpath, row = rows[0]
        if int(row["episode"]) != e:
            raise SystemExit(f"{job}: manifest row episode {row['episode']} != {e}")
        tck = Path(ex["training_checkpoint"])
        tsha = sha256_file(tck) if tck.is_file() else None
        if tsha != ex.get("training_sha256") or ex.get("source_sha256") != tsha:
            raise SystemExit(f"{job}: training checkpoint {tck} sha256 != export record")
        pq = Path(row["parquet"])
        psha = sha256_file(pq)
        if psha != row.get("parquet_sha256"):
            raise SystemExit(f"{job}: training parquet sha256 changed since {mpath}")
        csv_p, js_p, n_want = FROZEN_SCORES[src]
        vals = sorted(((int(r["run"]), float(r["auc"]), float(r["sp_sr"]), float(r["mp_sr"]), float(r["mppe"]))
                       for r in read_csv(csv_p) if r["job"] == job and r.get("variant") == "B" and r.get("status") == "ok"))  # fmt: skip
        if [v[0] for v in vals] != list(range(1, n_want + 1)):
            raise SystemExit(f"{job}: frozen variant-B runs {[v[0] for v in vals]} in {csv_p.name} (want 1..{n_want})")
        aucs = [v[1] for v in vals]
        m_p = float(sum(aucs) / len(aucs))
        js = read_json(js_p)["jobs"][job]
        jsb = js.get("B", js)
        if [float(x) for x in jsb["auc"]] != aucs or abs(float(jsb["auc_stats"]["mean"]) - m_p) > 1e-12:
            raise SystemExit(f"{job}: frozen job stats {js_p.name} differ from the scores csv")
        kind = row.get("root_kind")
        fr = ep_frame_ref(kind, e) if kind in FRAME_REFS else None
        out[e] = {"episode": e, "job": job, "source_round": src, "input": kind, "owner": row.get("owner"),
                  "tree": ex.get("tree"), "iteration": int(row["iteration"]), "export_record": str(exps[0]),
                  "export_record_sha256": sha256_file(exps[0]), "export_file": str(f), "export_sha256": fsha,
                  "export_tolerance": ex.get("tolerance"), "training_checkpoint": str(tck), "training_sha256": tsha,
                  "manifest_file": mpath, "parquet": str(pq), "parquet_sha256": psha, "data_root": row.get("data_root"),
                  "manifest": row.get("manifest"), "frame_refs": row.get("frame_refs"), "sequence_id": row.get("sequence_id"),
                  "object": row.get("object"), "frame_ref_file": str(fr) if fr else None,
                  "frame_ref_sha256": sha256_file(fr) if fr and fr.is_file() else None,
                  "frozen_runs": [{"run": v[0], "cons_auc": v[1], "cons_sp_sr": v[2], "cons_mp_sr": v[3], "cons_mppe": v[4]} for v in vals],
                  "frozen_scores_csv": str(csv_p), "frozen_scores_csv_sha256": sha256_file(csv_p),
                  "m_P_frozen": m_p, "r5_closest_run": int(closest[k]["run"])}  # fmt: skip
        log(f"incumbent ep{k}: {job} ({src}, {kind}, {ex.get('tree')}) export {fsha[:12]} m_P_frozen {m_p:.6f} "
            f"({len(aucs)} frozen runs)")
    return out


# ================================================================================================ pool
def job_name(e: int, lab: str, it: int) -> str:
    return f"ep{e:02d}_{lab}_{it}"


def copy_verified(src: Path, dst: Path) -> dict:
    before = sha256_file(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise SystemExit(f"refusing to overwrite {dst}")
    tmp = dst.with_name(dst.name + ".tmp_r6copy")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    os.chmod(dst, 0o444)
    copy = sha256_file(dst)
    after = sha256_file(src)
    st = os.stat(src)
    return {"source": str(src), "copy": str(dst), "sha256_source_before": before, "sha256_copy": copy,
            "sha256_source_after": after, "equal": before == copy == after, "size": st.st_size,
            "source_mtime": dt.datetime.fromtimestamp(st.st_mtime, TZ8).strftime("%Y-%m-%d %H:%M:%S"), "copied_at": now()}  # fmt: skip


def build_pool(runs: list[dict], inc: dict[int, dict], write: bool, log) -> tuple[list[dict], list[dict]]:
    pool, copies = [], []
    seen: dict[str, str] = {}
    inc_sha = {e: x["training_sha256"] for e, x in inc.items()}
    order = sorted((r for r in runs if r["eligible"]), key=lambda r: (r["episode"], r["owner"], r["dir"]))
    for r in order:
        d = Path(r["dir"])
        e = r["episode"]
        for rank, it in enumerate(r["picks"]):
            src = d / f"model_{it}.safetensors"
            h = header(src)
            kind, droot, seq = root_kind_of(h["parquet"])
            sha = sha256_file(src)
            job = job_name(e, r["label"], it)
            row = {"job": job, "episode": e, "input": kind, "owner": r["owner"], "tree": r["tree"], "run_dir": str(d),
                   "run_label": r["label"], "iteration": it, "run_last": r["last"], "slot": r["slots"][rank],
                   "source_checkpoint": str(src), "sha256": sha, "parquet": h["parquet"], "sequence_id": seq,
                   "data_root": str(ROOTS[kind]) if kind else None}  # fmt: skip
            if kind != r["input"] or seq != r["sequence_id"]:
                row["role"] = "refused"
                row["reason"] = f"checkpoint parquet {h['parquet']} differs from the run's"
                pool.append(row)
                continue
            if inc_sha.get(e) == sha:
                row["role"] = "incumbent"
                row["reason"] = f"== round-5 incumbent {inc[e]['job']} (not screened)"
            elif sha in seen:
                row["role"] = "duplicate"
                row["reason"] = f"same file as {seen[sha]}"
            else:
                row["role"] = "screen"
                row["reason"] = "pool rule R1-R4"
                seen[sha] = job
            pq = Path(h["parquet"])
            row["parquet_sha256"] = sha256_file(pq)
            fr = ep_frame_ref(kind, e)
            row["frame_refs"] = str(FRAME_REFS[kind]) if FRAME_REFS[kind] else None
            row["frame_ref_file"] = str(fr) if fr else None
            row["frame_ref_sha256"] = sha256_file(fr) if fr and fr.is_file() else None
            row["training_checkpoint"] = str(src)
            row["training_sha256"] = sha
            if r["owner"] == "zuxing" and row["role"] == "screen":
                dst = COPIES / Path(r["dir"]).relative_to(ZU / "runs") / src.name
                rsi = d / f"rsi_{it}.safetensors"
                if write:
                    c = copy_verified(src, dst)
                    copies.append(c)
                    if not c["equal"] or c["sha256_copy"] != sha:
                        raise SystemExit(f"copy of {src} not verified: {c}")
                    if rsi.is_file():
                        c2 = copy_verified(rsi, dst.with_name(rsi.name))
                        copies.append(c2)
                        row["rsi_sha256"] = c2["sha256_copy"]
                row["training_checkpoint"] = str(dst)
                row["copy_of"] = str(src)
            pool.append(row)
    n = {k: sum(1 for x in pool if x["role"] == k) for k in ("screen", "incumbent", "duplicate", "refused")}
    log(f"pool: {len(pool)} selected files from {len(order)} eligible runs: {n}")
    return pool, copies


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
    tmp = p.with_name(p.name + ".tmp_r6p")
    tmp.write_text(text)
    os.replace(tmp, p)
    os.chmod(p, 0o444)


POOL_COLS = ["job", "episode", "input", "owner", "tree", "role", "reason", "run_label", "run_dir", "iteration", "run_last",
             "slot", "sha256", "source_checkpoint", "training_checkpoint", "copy_of", "rsi_sha256", "parquet",
             "parquet_sha256", "sequence_id", "data_root", "frame_refs", "frame_ref_file", "frame_ref_sha256"]  # fmt: skip
RUN_COLS = ["dir", "owner", "label", "tree", "input", "episode", "eligible", "reason", "first", "last", "n_files",
            "n_settled", "newer_unsettled", "targets", "picks", "slots", "final_json_checkpoint", "final_json_reason", "output_dir",
            "parquet"]  # fmt: skip


def zuxing_code_rows(runs: list[dict], pool: list[dict], inc: dict[int, dict]) -> list[dict]:
    """Which Zuxing training code each of his pool runs / incumbents needs, and whether that commit is a pushed head
    (inputs_r6/LS_REMOTE_R6.txt). His commit for a pro6000 run = the 'commit=' field of the first 'train_t2:' line of
    his stdout log logs/<run>/ep<NN>.log (only that one line is read)."""
    remote = {}
    if LS_REMOTE.is_file():
        for ln in LS_REMOTE.read_text().splitlines():
            f = ln.split()
            if len(f) == 2:
                remote.setdefault(f[0], []).append(f[1].replace("refs/heads/", ""))
    out = []
    zruns = sorted({r["dir"] for r in runs if r["eligible"] and r["owner"] == "zuxing"})
    for d in zruns:
        rel = Path(d).relative_to(ZU / "runs")
        log = ZU / "logs" / rel.parts[0] / f"{rel.parts[-1]}.log"
        commit = None
        if log.is_file():
            with guard(log).open(errors="replace") as fp:
                for _ in range(200):
                    ln = fp.readline()
                    if not ln:
                        break
                    if ln.startswith("train_t2:"):
                        m = re.search(r"commit=([0-9a-f]{7,40})", ln)
                        commit = m.group(1) if m else None
                        break
        heads = [b for full, bs in remote.items() for b in bs if commit and full.startswith(commit)]
        out.append({"what": "pool run (pro6000)", "run": str(rel), "screened_files": sum(1 for p in pool if p["run_dir"] == d and p["role"] == "screen"),
                    "trained_on": "pro6000", "commit": commit, "pushed_heads": " ".join(heads) or "NOT a pushed head",
                    "code_location": f"worktree of his clone (log {log})"})  # fmt: skip
    for e, x in sorted(inc.items()):
        if x["owner"] != "zuxing":
            continue
        try:
            od = header(Path(x["training_checkpoint"]))["output_dir"]
        except Exception:  # noqa: BLE001
            od = None
        on_server = bool(od) and str(od).startswith(str(ZU))
        out.append({"what": f"round-5 incumbent ep{e:02d}", "run": od, "screened_files": 0,
                    "trained_on": "pro6000" if on_server else "remote (5090 / robolab)", "commit": None,
                    "pushed_heads": "see the pro6000 run row" if on_server else "UNKNOWN: ask Zuxing for the training commit",
                    "code_location": x["training_checkpoint"]})  # fmt: skip
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("check", "resolve"))
    a = ap.parse_args()
    write = a.mode == "resolve"
    t_c = time.time()
    t_c_text = dt.datetime.fromtimestamp(t_c, TZ8).strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = []

    def log(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    if write and (INP / "pool_r6.json").exists():
        raise SystemExit("inputs_r6/pool_r6.json exists: the round-6 pool is already resolved")
    log(f"t2cons_r6_pool {a.mode} at T_c = {t_c_text} CST (K = {K}, min last iteration {MIN_LAST}, settle {SETTLE_S:.0f} s)")
    me = Path(__file__).resolve()
    log(f"script sha256 {sha256_file(me)}")
    inc = incumbents(log)
    dirs = scan_dirs()
    runs = [run_record(d, t_c) for d in dirs]
    elig = [r for r in runs if r["eligible"]]
    log(f"scan: {len(dirs)} directories with model files; {len(elig)} eligible training runs")
    for r in sorted(runs, key=lambda r: (not r["eligible"], r.get("episode", 99), r["dir"])):
        if r["eligible"]:
            log(f"  RUN ep{r['episode']:02d} {r['owner']:<6} {r['label']:<9} {r['input']:<5} L={r['last']:<7} picks {r['picks']} "
                f"({Path(r['dir']).relative_to(ZU if r['owner'] == 'zuxing' else ME)})")
    pool, copies = build_pool(runs, inc, write, log)
    per_ep = {e: sum(1 for p in pool if p["episode"] == e and p["role"] == "screen") for e in range(30)}
    log("screen candidates per episode: " + " ".join(f"{e:02d}:{n}" for e, n in per_ep.items()))
    zrows = zuxing_code_rows(runs, pool, inc)
    if not write:
        log("check only: nothing written")
        return 0
    INP.mkdir(parents=True, exist_ok=True)
    meta = {"t_c": t_c_text + " CST", "t_c_epoch": t_c, "script": str(me), "script_sha256": sha256_file(me), "K": K,
            "min_last_iteration": MIN_LAST, "settle_seconds": SETTLE_S, "rule": __doc__.split("Pool rule", 1)[1].split("Zuxing's selected", 1)[0]}  # fmt: skip
    write_new(INP / "pool_r6.json", json.dumps({**meta, "rows": pool}, indent=1, default=str) + "\n")
    write_new(INP / "pool_r6.csv", csv_text(pool, POOL_COLS))
    write_new(INP / "runs_r6.json", json.dumps({**meta, "runs": runs}, indent=1, default=str) + "\n")
    write_new(INP / "runs_r6.csv", csv_text(runs, RUN_COLS))
    write_new(INP / "incumbents_r6.json", json.dumps({**meta, "source": "selection_r5.json compositions.ROUND5 (SELECTION_R5.sha256)",
                                                     "incumbents": {f"{e:02d}": x for e, x in inc.items()}}, indent=1) + "\n")  # fmt: skip
    write_new(INP / "ZUXING_CODE_R6.tsv", csv_text(zrows, ["what", "run", "screened_files", "trained_on", "commit", "pushed_heads",
                                                           "code_location"], "\t"))  # fmt: skip
    ccols = ["source", "copy", "sha256_source_before", "sha256_copy", "sha256_source_after", "equal", "size", "source_mtime", "copied_at"]
    write_new(ROOT / "COPIES_r6.tsv", csv_text(copies, ccols, "\t"))
    tell = ["Zuxing: round 6 of the Tier-2 noisy-only selection (t2_consensus_r6_20261031) made sha256-verified READ-ONLY",
            f"copies of these files of your runs at {t_c_text} CST (nothing in your folder was written; no git command; none",
            "of your scripts was run). Each line: source -> copy (sha256).", ""]  # fmt: skip
    tell += [f"{c['source']} -> {c['copy']} ({c['sha256_copy']})" for c in copies]
    tell += ["", "Open items for the submission commit (see inputs_r6/ZUXING_CODE_R6.tsv): the training commit of every",
             "remote-trained (5090 / robolab) checkpoint in the round-5 set; whether your pro6000 worktrees had uncommitted",
             "changes when these runs started; the G2 data builder (tmp/t2/g2data/) is not in any pushed commit.", ""]  # fmt: skip
    write_new(ROOT / "TELL_ZUXING_r6.txt", "\n".join(tell) + "\n")
    # TRACKS: every training parquet, frame reference and data-root manifest the pool and incumbents use
    tracks: dict[str, str] = {}
    for p in pool:
        if p["role"] in ("screen", "incumbent"):
            tracks[p["parquet"]] = p["parquet_sha256"]
            if p.get("frame_ref_file"):
                tracks[p["frame_ref_file"]] = p["frame_ref_sha256"]
    for x in inc.values():
        tracks[x["parquet"]] = x["parquet_sha256"]
        if x.get("frame_ref_file"):
            tracks[x["frame_ref_file"]] = x["frame_ref_sha256"]
    for k, v in ROOTS.items():
        m = v / "whole_body/soma/track2_manifest.json"
        tracks[str(m)] = sha256_file(m)
    write_new(INP / "TRACKS_R6.sha256", "".join(f"{h}  {p}\n" for p, h in sorted(tracks.items())))
    write_new(INP / "resolve_r6.log", "\n".join(lines) + "\n")
    snap = [INP / n for n in ("pool_r6.json", "pool_r6.csv", "runs_r6.json", "runs_r6.csv", "incumbents_r6.json",
                              "ZUXING_CODE_R6.tsv", "TRACKS_R6.sha256", "resolve_r6.log")]  # fmt: skip
    snap += [INP / "LS_REMOTE_R6.txt"] if LS_REMOTE.is_file() else []
    snap += [ROOT / "COPIES_r6.tsv", ROOT / "TELL_ZUXING_r6.txt"] + sorted(p for p in COPIES.rglob("*") if p.is_file())
    write_new(INP / "SNAPSHOT_R6.sha256", "".join(f"{sha256_file(p)}  {p.relative_to(ROOT)}\n" for p in snap))
    print(f"written: {INP} (SNAPSHOT_R6.sha256 {sha256_file(INP / 'SNAPSHOT_R6.sha256')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
