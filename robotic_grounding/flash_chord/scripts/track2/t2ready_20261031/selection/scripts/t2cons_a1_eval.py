#!/usr/bin/env python3
"""t2cons_a1_eval.py: AMENDMENTS.md A3.3 of fc_runs/submissions/t2_consensus_20261024 (step S3 for the A3 jobs).

Five official own-input evaluations of each of the 10 NEW jobs that scripts/t2cons_a1_resolve.py resolved at
T_a = 2026-10-26 17:29:25 CST (inputs_a1/candidates_a1.json, pinned in PREREG_PINS.md):
  variants A and B: ep05_zx_k3fix_88000, ep26_zx_k3fix_92000, ep26_k3ca_final25298, ep04_k3fix_final44922
  variant B only:   ep01_zx_g2_124000, ep04_zx_g2_88000, ep19_zx_g2_92000, ep28_zx_g2_100000,
                    ep01_zx_g2hard_102000, ep19_zx_g2hard_52000
The 2 reused jobs (ep04_zx_k3fix_12000, ep01_t2fixL_p62000) are the same checkpoint files (sha256) as t2_best5 jobs that
already have 5 ok official evaluations: they are NOT evaluated again; their own-input runs are only listed from the
pinned snapshot inputs/t2_best5/own_input_evals.csv.

Protocol: PREREG section 6 unchanged (= t2_best5 PREREG 3.2-3.5 and 3.7). This script loads scripts/t2cons_eval.py from
its pinned bytes (2cf42786...), which loads scripts/t2best5_driver.py from its pinned bytes (24b9af39...) re-rooted at
this folder; the copy, export, evaluation, ok checks, retries, hull check, kit / stock tree check (same
integrity/baseline.json), GPU locks, lanes and the t2_best5 live-eval count are that pinned code. Differences, all
implementation only (PREREG_PINS_A1b.md):
  - jobs and expected sha256 come from inputs_a1/candidates_a1.json (sha256 pinned); designation re-checked from the
    snapshot copies inputs_a1/zuxing_metrics/ (rule Z of A3.2: status ok, stock_verified True, finished_at <= T_a,
    highest add_auc, exact ties newest iteration; the file's sha256 = the row's source_sha256) or final.json (ours);
  - K3CA input kind added in memory (data/human_motion_data_tier2_k3ca, frame_refs_k3ca, its byte-identical manifest);
  - training parquet re-hashed against an earlier record: the t2_best5 / manifest_new row of the same file, or for K3CA
    the root's READY sha256 list; the run's hydra task.parquet must name the same file;
  - export tree: Zuxing's = code/nv-v2d_runin @ b39b69b (PREREG 6), ours = code/fc_r2 @ 61fa5d3 (their training tree,
    as t2_best5 3.3);
  - outputs (new files only): zuxing_ckpts_a1/, COPIES_a1.tsv, exports/<job>/, runs/<job>/, manifest_a1.{json,csv},
    own_input_evals_a1.csv, summary_a1_own_input.{csv,json,md}, eval_a1_driver.log, eval_a1_heartbeat.json,
    dryrun/a1_eval_plan_*.txt, preflight/hulls_* (and the pinned driver's driver.pid, .driver.lock, integrity/check_*);
  - order: the variant-A,B jobs first (priority 1: run 1 of each before run 2, ...), then the B-only jobs (priority 2);
  - no LOCAL clean diagnostic, no clean path opened (guard on every record read here).
Scheduling (decides nothing): GPUs 0,1,2,5,6,7 only; flock fc_runs/queue_runin/evallock/gpu<g>.lock opened read-only;
<= 2 per NUMA eval lane and <= 4 in all, counting the t2_best5 driver's live evaluations; nice + taskset on the eval-lane
cores. Never signals any process; an eval past 2400 s is only flagged OVERDUE; ROOT/STOP = no new starts.

  python -B scripts/t2cons_a1_eval.py run [--dry-run] [--gpus 0,1,2,5,6,7] [--max-parallel 4] [--lane-max 2]
  python -B scripts/t2cons_a1_eval.py summary | status
Background start: scripts/t2cons_a1_eval_launch.sh
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
ROOT = ME / "fc_runs/submissions/t2_consensus_20261024"
TE_FILE = ME / "scripts/t2cons_eval.py"
TE_SHA = "2cf427866a9432a3bd1b8fbb7864149d4258a5d19100c20da92d32bfe4fbfa9b"  # PREREG_PINS.md
AMEND, AMEND_SHA_FILE = ROOT / "AMENDMENTS.md", ROOT / "AMENDMENTS.sha256"
AMEND_SHA = "f648226a95a061eeefcfd5138cb01823db8a060fe121a0f4968ffa163dc4229f"
PINS, PINS_SHA_FILE = ROOT / "PREREG_PINS.md", ROOT / "PREREG_PINS.sha256"
PINS_SHA = "9d1493797205e8c60255c50658c536714a3702fe8751c724f37ed1444a040ece"
PINS_B, PINS_B_SHA_FILE = ROOT / "PREREG_PINS_A1b.md", ROOT / "PREREG_PINS_A1b.sha256"
CANDS = ROOT / "inputs_a1/candidates_a1.json"
CANDS_SHA = "be42b6ddd32ff708365b894852cdbbfbdda583504f44038dc4790827a97301e4"
SNAP_A1 = ROOT / "inputs_a1/SNAPSHOT_A1.sha256"
SNAP = ROOT / "inputs"
ZU = Path("/root/autodl-tmp/zuxing.lu")  # READ ONLY: only the designated model + rsi files are copied (sha256)
K3CA_ROOT = ME / "data/human_motion_data_tier2_k3ca"
K3CA_FR = ME / "tmp_analysis/tier2/k3ca/frame_refs_k3ca"
TZ8 = dt.timezone(dt.timedelta(hours=8))
T_A = dt.datetime(2026, 10, 26, 17, 29, 25, tzinfo=TZ8).timestamp()
NRUNS = 5
NEW = {  # job -> (episode, iteration, checkpoint sha256, variants): must equal candidates_a1.json new_jobs
    "ep05_zx_k3fix_88000": (5, 88000, "f0234031e9eb5416ea43046797162da4224c5f939e73ca74928ac207be6e8dc1", "A,B"),
    "ep26_zx_k3fix_92000": (26, 92000, "0b02cea1635cc07ebb4b275d58e9c40751e82f7a4a82ac81426f2b12de348992", "A,B"),
    "ep26_k3ca_final25298": (26, 25298, "daeee885d36ca24a7d6fe485354ef038dc7a617c1c85c553ad99fa3e2771180e", "A,B"),
    "ep04_k3fix_final44922": (4, 44922, "b5331be91e9abcb65d7ab4ed39a67f3cfb1af67b32b12ee9f4e198912644ec79", "A,B"),
    "ep01_zx_g2_124000": (1, 124000, "32fab5306c205a5e2bb880f75e3239d984fa69791d95f437b9a78db77896f5a0", "B"),
    "ep04_zx_g2_88000": (4, 88000, "16faf35fd7cfc26e76297e810ba3b691b82b80d03f0706a52803300e09fe9d48", "B"),
    "ep19_zx_g2_92000": (19, 92000, "62ca1cf7eac3f791ba95396a9c77be2c13e0fcf86cb77a204c776fc6dd4b2257", "B"),
    "ep28_zx_g2_100000": (28, 100000, "cbb46ce7034b9b1bb436233bdf969e8c29d6c15eca2b7c7adb7bee8b03a161c3", "B"),
    "ep01_zx_g2hard_102000": (1, 102000, "09bc72a0371827f13b9ab4862ea4e8226412266468decc64bae8dba0aa4e53b9", "B"),
    "ep19_zx_g2hard_52000": (19, 52000, "a6ef35703984e31f8ed5036b4ded7fc65cc42929cadc49781b175e754cedb2fc", "B"),
}  # fmt: skip
REUSED = {  # job -> checkpoint sha256 (t2_best5 jobs with 5 ok official evaluations in the snapshot)
    "ep04_zx_k3fix_12000": "3efcbf86a775c8a3e9b5be5265242a7f90e309e998d6224f300e80ab3995105a",
    "ep01_t2fixL_p62000": "2d69e2da0647d4c3f5b23880eede56ac3bb370deffed7c9c2799a08d4ca6416f",
}
OUT_MANIFEST, OUT_MANIFEST_CSV = ROOT / "manifest_a1.json", ROOT / "manifest_a1.csv"
OUT_COPIES = ROOT / "COPIES_a1.tsv"
OUT_EVALS = ROOT / "own_input_evals_a1.csv"
OUT_SUM = ROOT / "summary_a1_own_input"
OUT_LOG, OUT_HB = ROOT / "eval_a1_driver.log", ROOT / "eval_a1_heartbeat.json"
CKPT_DIR = ROOT / "zuxing_ckpts_a1"


# ================================================================================================ loading (pinned bytes)
def sha256_file(file) -> str:
    h = hashlib.sha256()
    with Path(file).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_te():
    """scripts/t2cons_eval.py compiled from the exact bytes whose sha256 PREREG_PINS.md pins (it is not run)."""
    src = TE_FILE.read_bytes()
    got = hashlib.sha256(src).hexdigest()
    if got != TE_SHA:
        raise SystemExit(f"{TE_FILE} sha256 {got} != pinned {TE_SHA}")
    mod = types.ModuleType("t2cons_eval_readonly")
    mod.__file__ = str(TE_FILE)
    sys.modules[mod.__name__] = mod
    exec(compile(src, str(TE_FILE), "exec"), mod.__dict__)  # noqa: S102 (pinned bytes)
    return mod


TE = load_te()
guard, guarded_json = TE.guard, TE.guarded_json


def load_driver():
    D = TE.load_driver()  # t2best5_driver.py, pinned bytes, re-rooted at ROOT (inventory / PREREG of this folder)
    # K3CA input kind, in memory only (PREREG 3.2 table; READY: manifest byte copy of the K3fix root's)
    D.ROOTS["k3ca"] = K3CA_ROOT
    D.FRAME_REFS["k3ca"] = K3CA_FR
    D.MANIFEST_FOR["k3ca"] = K3CA_ROOT / "whole_body/soma/track2_manifest.json"
    D.check_pins = lambda require_pins_file=True: check_pins(D, require_pins_file)
    return D


# ================================================================================================ pins
def _sha_file_value(p: Path) -> str:
    return (guard(p).read_text().split() or [""])[0] if p.is_file() else ""


def _pin_lines(text: str) -> list[tuple[str, str]]:
    out = []
    for line in text.splitlines():
        f = line.split()
        if len(f) == 3 and f[0] == "pin" and len(f[2]) == 64:
            out.append((f[1], f[2]))
    return out


def _pin_path(rel: str) -> Path:
    return ME / rel if rel.startswith("scripts/") else ROOT / rel


def check_pins(D, require_pins_file: bool) -> tuple[list[str], list[str]]:
    fatal, info = TE.check_pins(D)  # PREREG.md == PREREG.sha256, PREREG 11 pins + commits, inputs/SNAPSHOT.sha256
    for name, p, sha_file, want in (("AMENDMENTS.md", AMEND, AMEND_SHA_FILE, AMEND_SHA),
                                    ("PREREG_PINS.md", PINS, PINS_SHA_FILE, PINS_SHA)):  # fmt: skip
        got = sha256_file(p) if p.is_file() else "missing"
        rec = _sha_file_value(sha_file)
        (info if got == rec == want else fatal).append(f"{name} sha256 {got[:16]} (sha file {rec[:16]}, pinned here {want[:16]})")
    n = 0
    for rel, want in _pin_lines(guard(PINS).read_text()) if PINS.is_file() else []:
        p = _pin_path(rel)
        h = sha256_file(guard(p)) if p.is_file() else "missing"
        n += 1
        if h != want:
            fatal.append(f"PREREG_PINS pin {rel}: {h[:16]} != {want[:16]}")
    info.append(f"PREREG_PINS.md: {n} pins checked")
    for line in guard(SNAP_A1).read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        p = ROOT / parts[1].strip()
        h = sha256_file(guard(p)) if p.is_file() else "missing"
        if h != parts[0]:
            fatal.append(f"{parts[1].strip()}: {h[:16]} != SNAPSHOT_A1.sha256 {parts[0][:16]}")
    got = sha256_file(CANDS)
    (info if got == CANDS_SHA else fatal).append(f"inputs_a1/candidates_a1.json sha256 {got[:16]} (pinned here {CANDS_SHA[:16]})")
    mine = sha256_file(Path(__file__).resolve())
    if PINS_B.is_file():
        got, rec = sha256_file(PINS_B), _sha_file_value(PINS_B_SHA_FILE)
        if got != rec:
            fatal.append(f"PREREG_PINS_A1b.md sha256 {got[:16]} != PREREG_PINS_A1b.sha256 {rec[:16]}")
        else:
            text = guard(PINS_B).read_text()
            pins = _pin_lines(text)
            if ("scripts/t2cons_a1_eval.py", mine) not in pins:
                fatal.append(f"PREREG_PINS_A1b.md does not pin scripts/t2cons_a1_eval.py {mine[:16]}")
            for rel, want in pins:
                p = _pin_path(rel)
                h = sha256_file(guard(p)) if p.is_file() else "missing"
                (info if h == want else fatal).append(f"PREREG_PINS_A1b pin {rel}: {h[:16]} (pinned {want[:16]})")
            info.append(f"PREREG_PINS_A1b.md sha256 {got[:16]} == its sha file; pins this script {mine[:16]}")
    elif require_pins_file:
        fatal.append("PREREG_PINS_A1b.md missing: this script must be pinned before it copies or evaluates (PREREG_PINS 3.3)")
    else:
        info.append(f"PREREG_PINS_A1b.md not written yet (dry run); this script sha256 {mine}")
    return fatal, info


# ================================================================================================ candidates
def load_candidates() -> dict:
    data = guarded_json(CANDS)
    if data is None:
        raise SystemExit(f"cannot read {CANDS}")
    return data


def specs(cands: dict) -> list[dict]:
    """The 10 new jobs, A,B first then B-only, each with its A3.2 resolution record."""
    res = {r.get("job"): r for r in cands.get("resolutions", [])}
    out = []
    for nj in cands.get("new_jobs", []):
        s = dict(nj)
        s["resolution"] = res.get(nj["job"], {})
        out.append(s)
    order = {j: i for i, j in enumerate(NEW)}
    out.sort(key=lambda s: (0 if "A" in s.get("variants", "") else 1, order.get(s["job"], 99)))
    return out


def recorded_parquets(D) -> dict[str, str]:
    """normalised parquet path -> sha256 recorded earlier: t2_best5 snapshot manifest (+ Stage B), manifest_new.json, and
    the K3CA root's READY list."""
    out: dict[str, str] = {}
    rows = list((guarded_json(SNAP / "t2_best5/manifest.json") or {}).get("rows", []))
    for line in guard(SNAP / "t2_best5/manifest_stageB.jsonl").read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    rows += list((guarded_json(ROOT / "manifest_new.json") or {}).get("rows", []))
    for r in rows:
        if r.get("parquet") and r.get("parquet_sha256"):
            out.setdefault(os.path.normpath(r["parquet"]), r["parquet_sha256"])
    for line in guard(K3CA_ROOT / "READY").read_text().splitlines():
        f = line.split()
        if len(f) == 2 and len(f[0]) == 64 and f[1].endswith("data.parquet"):
            out.setdefault(os.path.normpath(str(K3CA_ROOT / f[1])), f[0])
    return out


def objects_by_sequence() -> dict[str, str]:
    out = {}
    for r in (guarded_json(SNAP / "t2_best5/manifest.json") or {}).get("rows", []):
        if r.get("sequence_id") and r.get("object"):
            out.setdefault(r["sequence_id"], r["object"])
    return out


def designate_zuxing(D, spec: dict) -> tuple[dict, list[str]]:
    """Rule Z (A3.2), re-derived from the snapshot copy: must give the resolved iteration and file."""
    res = spec["resolution"]
    problems = []
    src = Path(res.get("metrics_copy", ""))
    if not src.is_file() or sha256_file(guard(src)) != res.get("metrics_copy_sha256"):
        return {}, [f"snapshot metrics copy {src} missing or != its recorded sha256"]
    rows = D.load_csv(guard(src))
    ok = []
    for r in rows:
        try:
            it, auc = int(r["iteration"]), D.fnum(r.get("add_auc"))
        except (KeyError, ValueError, TypeError):
            continue
        ts = D.parse_time(r.get("finished_at"))
        if (r.get("status") == "ok" and str(r.get("stock_verified")) == "True" and auc is not None and ts is not None
                and ts <= T_A and int(r.get("episode", spec["episode"])) == int(spec["episode"])):  # fmt: skip
            ok.append((auc, it, r))
    ok.sort(key=lambda x: (-x[0], -x[1]))
    if not ok:
        return {}, [f"no eligible row by T_a in {src.name}"]
    auc, it, best = ok[0]
    if it != int(spec["iteration"]):
        problems.append(f"rule Z gives model_{it}, candidates_a1.json says model_{spec['iteration']}")
    if best.get("source_sha256") != spec["sha256"]:
        problems.append(f"designated row source_sha256 {str(best.get('source_sha256'))[:16]} != {spec['sha256'][:16]}")
    keep = ("iteration", "add_auc", "sp_sr", "mp_sr", "mppe_cm", "chord_sr", "export_max_rel_diff", "stock_verified",
            "source_sha256", "finished_at")  # fmt: skip
    return {"rule": "Z (A3.2)", "snapshot_file": str(src), "n_rows": len(rows), "n_eligible": len(ok),
            "row": {k: best.get(k) for k in keep}}, problems  # fmt: skip


def designate_ours(D, spec: dict, run_dir: Path) -> tuple[dict, list[str]]:
    fj = guarded_json(run_dir / "final.json") or {}
    it = int(spec["iteration"])
    problems = []
    if fj.get("iteration") != it or fj.get("checkpoint") != f"model_{it}.safetensors":
        problems.append(f"final.json names {fj.get('checkpoint')} (iteration {fj.get('iteration')}), not model_{it}")
    res = spec["resolution"]
    return {"rule": "final.json (A3.2)", "final_json": {k: fj.get(k) for k in ("checkpoint", "iteration", "reason", "finished_at")},
            "verdict_for_information": res.get("verdict")}, problems  # fmt: skip


def record_copy(D, src: Path, dst: Path, src_sha: str) -> None:
    if not OUT_COPIES.is_file():
        D.append_line(OUT_COPIES, "source\tdestination\tbytes\tsource_sha256\tdestination_sha256\tequal\tcopied_at")
    dsha = D.sha256(dst)
    D.append_line(OUT_COPIES, f"{src}\t{dst}\t{dst.stat().st_size}\t{src_sha}\t{dsha}\t{dsha == src_sha}\t{D.now()}")


def build_a1(D, spec: dict, pq_rec: dict, objs: dict, copy: bool) -> dict:
    job_id = spec["job"]
    ep, it = int(spec["episode"]), int(spec["iteration"])
    problems: list[str] = []
    notes: list[str] = []
    want = NEW.get(job_id)
    if want is None or want != (ep, it, spec.get("sha256"), spec.get("variants")):
        problems.append(f"candidates_a1.json row {job_id} ({ep}, {it}, {str(spec.get('sha256'))[:16]}, {spec.get('variants')}) "
                        f"!= this script's list {want}")
    owner = spec.get("owner")
    res = spec["resolution"]
    if owner == "zuxing":
        run = spec["run"]
        run_dir = ZU / "runs" / run / f"ep{ep:02d}"
        tree, family = "b39b69b", "zuxing"
    else:
        run = res.get("run", "")
        run_dir = Path(run)
        tree, family = "fc_r2", f"{spec.get('input')}_final"
    train = run_dir / f"model_{it}.safetensors"
    if os.path.normpath(spec.get("checkpoint", "")) != os.path.normpath(train):
        problems.append(f"candidates checkpoint {spec.get('checkpoint')} != {train}")
    job = {"job": job_id, "episode": ep, "family": family, "owner": owner, "arm": f"{res.get('id')} {run} ({spec.get('input')})",
           "stage": "consensus AMENDMENTS A3.3 (new)", "variants": spec.get("variants"),
           "priority": 1 if "A" in str(spec.get("variants")) else 2, "provisional": False, "a3_id": res.get("id"),
           "run_dir": str(run_dir), "iteration": it, "tree": tree, "designation": res.get("designation"),
           "expected_sha256": spec.get("sha256"), "expected_sha256_from": "inputs_a1/candidates_a1.json",
           "training_checkpoint_original": str(train), "zuxing_run": run if owner == "zuxing" else None}  # fmt: skip
    if owner == "zuxing":
        des, p2 = designate_zuxing(D, spec)
    else:
        des, p2 = designate_ours(D, spec, run_dir)
    problems += p2
    job["designation_evidence"] = des
    if not train.is_file():
        problems.append(f"missing {train}")
        tsha = None
    else:
        tsha = D.sha256(train)
        if tsha != spec.get("sha256"):
            problems.append(f"checkpoint sha256 {tsha[:16]} != candidates_a1.json {str(spec.get('sha256'))[:16]}")
    job["training_sha256"] = tsha
    # training parquet: candidates_a1 row == hydra task.parquet; re-hashed against an earlier record
    pq = spec.get("training_parquet")
    ov = D.read_overrides(run_dir)
    if not ov.get("task.parquet") or os.path.normpath(ov["task.parquet"]) != os.path.normpath(str(pq)):
        problems.append(f"hydra task.parquet {ov.get('task.parquet')} != candidates_a1.json {pq}")
    lay = D.parquet_layout(Path(pq)) if pq else None
    if lay is None:
        problems.append(f"no / unexpected training parquet {pq}")
        job.update(problems=problems, notes=notes, status="refused: " + "; ".join(problems))
        return job
    parquet = Path(pq)
    data_root, seq = lay
    kind = D.root_kind(data_root)
    if kind != spec.get("input"):
        problems.append(f"data root kind {kind} != {spec.get('input')}")
    psha = D.sha256(parquet) if parquet.is_file() else None
    rec = pq_rec.get(os.path.normpath(str(parquet)))
    if psha is None or rec is None or psha != rec:
        problems.append(f"parquet sha256 {str(psha)[:16]} != earlier record {str(rec)[:16]}")
    else:
        notes.append(f"training parquet sha256 {psha[:12]} == earlier record")
    manifest = D.MANIFEST_FOR.get(kind) if kind else None
    if manifest is None or D.manifest_episode_seq(manifest, ep) != seq:
        problems.append(f"manifest {manifest} does not map ep{ep:02d} to {seq}")
    fr = D.FRAME_REFS.get(kind) if kind else None
    if not fr or os.path.normpath(str(fr)) != os.path.normpath(str(spec.get("frame_refs"))):
        problems.append(f"frame refs {fr} != candidates_a1.json {spec.get('frame_refs')}")
    elif not (Path(fr) / f"episode_{ep:06d}.npz").is_file():
        problems.append(f"no {fr}/episode_{ep:06d}.npz")
    job.update(object=objs.get(seq), sequence_id=seq, parquet=str(parquet), parquet_sha256=psha, data_root=str(data_root),
               root_kind=kind, manifest=str(manifest), frame_refs=str(fr) if fr else None)  # fmt: skip
    job["training_checkpoint"] = str(train)
    rsi = run_dir / f"rsi_{it}.safetensors"
    job["rsi_original"] = str(rsi) if rsi.is_file() else ""
    if owner == "zuxing":  # PREREG 6: sha256-checked copies of model + rsi; Zuxing's folder is only read
        dest = CKPT_DIR / run / f"ep{ep:02d}"
        job["training_checkpoint"] = str(dest / train.name)
        if copy and not problems:
            try:
                got = D.copy_verified(train, dest / train.name, tsha)
                record_copy(D, train, dest / train.name, got)
                sums = [f"{got}  {train.name}"]
                if rsi.is_file():
                    rs = D.copy_verified(rsi, dest / rsi.name, None)
                    record_copy(D, rsi, dest / rsi.name, rs)
                    sums.append(f"{rs}  {rsi.name}")
                    job["rsi_sha256"] = rs
                D.write_atomic(dest / "SHA256SUMS", "\n".join(sums) + "\n")
                D.write_atomic(dest / "SOURCE.json", D.dump({"copied_at": D.now(), "from": str(run_dir), "model": str(train),
                                                             "rsi": job["rsi_original"], "model_sha256": got,
                                                             "evidence": des}))  # fmt: skip
            except (OSError, RuntimeError) as error:
                problems.append(f"copy of Zuxing's checkpoint failed ({error})")
        elif not copy:
            notes.append("dry run: the copy is made when the manifest is built")
    job["problems"], job["notes"] = problems, notes
    job["status"] = "ok" if not problems else "refused: " + "; ".join(problems)
    return job


def build_manifest(D, log, copy: bool) -> list[dict]:
    cands = load_candidates()
    sp = specs(cands)
    if sorted(s["job"] for s in sp) != sorted(NEW):
        raise SystemExit(f"candidates_a1.json new_jobs {sorted(s['job'] for s in sp)} != {sorted(NEW)}")
    if sorted(cands.get("reused_jobs", [])) != sorted(REUSED):
        raise SystemExit(f"candidates_a1.json reused_jobs {cands.get('reused_jobs')} != {sorted(REUSED)}")
    pq_rec, objs = recorded_parquets(D), objects_by_sequence()
    rows = [build_a1(D, s, pq_rec, objs, copy) for s in sp]
    ok = [r for r in rows if r["status"] == "ok"]
    hulls = D.hull_check(ok, log)
    for r in ok:
        h = hulls.get(r["parquet"], {"ok": False, "missing": ["not checked"]})
        r["hulls"] = h
        if not h["ok"]:
            r["status"] = f"hull_missing: {h['missing']}"
    return rows


MANIFEST_COLS = ["job", "episode", "object", "family", "owner", "a3_id", "variants", "priority", "iteration", "training_sha256",
                 "training_checkpoint", "training_checkpoint_original", "rsi_sha256", "tree", "parquet", "parquet_sha256",
                 "root_kind", "data_root", "manifest", "frame_refs", "status"]  # fmt: skip


# ================================================================================================ reused jobs (snapshot)
def reused_records() -> dict[str, dict]:
    """job -> {k: metrics} of the 2 reused jobs from inputs/t2_best5/own_input_evals.csv (own-input only), with checks."""
    out: dict[str, dict] = {}
    man = {r["job"]: r for r in (guarded_json(SNAP / "t2_best5/manifest.json") or {}).get("rows", [])}
    with guard(SNAP / "t2_best5/own_input_evals.csv").open(newline="") as fp:
        evals = list(csv.DictReader(fp))
    for job, sha in REUSED.items():
        m = man.get(job, {})
        recs, ok = {}, m.get("training_sha256") == sha
        for r in evals:
            if r.get("job") == job and r.get("status") == "ok":
                try:
                    recs[int(r["run"])] = {x: float(r[x]) for x in ("auc", "sp_sr", "mp_sr", "mppe")}
                except (KeyError, ValueError):
                    pass
        out[job] = {"episode": m.get("episode"), "input": m.get("root_kind"), "checkpoint_sha256": m.get("training_sha256"),
                    "sha_matches": ok, "runs": recs}  # fmt: skip
    return out


# ================================================================================================ summary (own input)
EVAL_COLS = ["job", "episode", "run", "status", "try", "gpu", "seconds", "auc", "sp_sr", "mp_sr", "mppe", "chord_sr",
             "export_sha256", "finished_at", "problems", "export_sha256_full", "results_parquet", "reference_parquet_sha256",
             "variants", "input"]  # fmt: skip


def write_summary(D, jobs: list[dict]) -> dict:
    eval_rows, job_rows = [], []
    for j in jobs:
        jid = j["job"]
        base = {"job": jid, "source": "new (this folder, A3.3)", "episode": j["episode"], "object": j.get("object"),
                "variants": j.get("variants"), "input": j.get("root_kind"), "owner": j.get("owner"),
                "iteration": j.get("iteration"), "checkpoint_sha256": j.get("training_sha256")}  # fmt: skip
        if j.get("status") != "ok":
            job_rows.append({**base, "export_status": j.get("status")})
            continue
        exp = guarded_json(ROOT / "exports" / jid / "export.json") or {}
        recs, failed = {}, []
        for k in range(1, NRUNS + 1):
            r = guarded_json(ROOT / "runs" / jid / f"run{k}" / "run.json")
            if r is None:
                continue
            m = r.get("metrics") or {}
            eval_rows.append({"job": jid, "episode": j["episode"], "run": k, "status": r.get("status"), "try": r.get("try"),
                              "gpu": r.get("gpu"), "seconds": r.get("seconds"), **m, "chord_sr": r.get("chord_sr"),
                              "export_sha256": str(r.get("export_sha256"))[:16], "finished_at": r.get("finished_at"),
                              "problems": "; ".join(r.get("problems") or []), "export_sha256_full": r.get("export_sha256"),
                              "results_parquet": r.get("results_parquet"),
                              "reference_parquet_sha256": r.get("reference_parquet_sha256"), "variants": j.get("variants"),
                              "input": j.get("root_kind")})  # fmt: skip
            if r.get("status") == "ok":
                recs[k] = m
            else:
                failed.append(k)
        floor = sum(1 for k in recs if recs[k]["sp_sr"] == 1 and recs[k]["mp_sr"] == 1)
        des = j.get("designation_evidence") or {}
        row = des.get("row") or {}
        ver = des.get("verdict_for_information") or {}
        job_rows.append({**base, "parquet_sha256": j.get("parquet_sha256"), "export_status": exp.get("status", "pending"),
                         "export_tolerance": exp.get("tolerance"), "export_fallback": exp.get("fallback"),
                         "export_sha256": exp.get("sha256", ""), "export_max_rel_diff": exp.get("max_rel_diff"),
                         "n_ok": len(recs), "failed_runs": " ".join(map(str, failed)), "floor_sp1_mp1_runs": floor,
                         "floor_4of5": (floor >= 4) if len(recs) == NRUNS else "",
                         **D.job_stats_row(j, recs, NRUNS),
                         "designation_single_auc": row.get("add_auc", ver.get("own_auc")),
                         "designation_single_sp_sr": row.get("sp_sr", ver.get("own_sp")),
                         "designation_single_mp_sr": row.get("mp_sr", ver.get("own_mp"))})  # fmt: skip
    for jid, rr in reused_records().items():
        recs = rr["runs"]
        floor = sum(1 for k in recs if recs[k]["sp_sr"] == 1 and recs[k]["mp_sr"] == 1)
        job_rows.append({"job": jid, "source": "reused (t2_best5 snapshot, same file)", "episode": rr["episode"],
                         "variants": "A,B", "input": rr["input"], "checkpoint_sha256": rr["checkpoint_sha256"],
                         "export_status": "t2_best5" + ("" if rr["sha_matches"] else " SHA MISMATCH"), "n_ok": len(recs),
                         "floor_sp1_mp1_runs": floor, "floor_4of5": (floor >= 4) if len(recs) == NRUNS else "",
                         **D.job_stats_row({}, recs, NRUNS)})  # fmt: skip
    cols = ["job", "source", "episode", "object", "variants", "input", "owner", "parquet_sha256", "iteration",
            "checkpoint_sha256", "export_status", "export_tolerance", "export_fallback", "export_sha256",
            "export_max_rel_diff", "n_ok", "failed_runs", "floor_sp1_mp1_runs", "floor_4of5"]  # fmt: skip
    for m in D.METRICS:
        cols += [f"{m}_r{k}" for k in range(1, NRUNS + 1)] + [f"{m}_{s}" for s in ("mean", "sd", "min", "max")]
    cols += ["closest_to_mean_run", "designation_single_auc", "designation_single_sp_sr", "designation_single_mp_sr"]
    D.write_csv(OUT_EVALS, eval_rows, EVAL_COLS)
    D.write_csv(OUT_SUM.with_suffix(".csv"), job_rows, cols)
    D.write_atomic(OUT_SUM.with_suffix(".json"), D.dump({
        "generated_at": D.now(), "runs_per_job": NRUNS, "jobs": job_rows,
        "note": "own input = each policy against its OWN training parquet (official kit evaluator 160d24f5, stock 8c778f9, "
                "frame 0, explicit reset, world count 1). Comparable only within one training parquet sha256. floor = "
                "runs with SP-SR = MP-SR = 1 (PREREG 5.4b needs >= 4 of 5). No clean number anywhere in this file."}))
    md = [f"# t2_consensus A3.3: 5x official own-input evaluations of the A3 jobs ({D.now()})", "",
          "Own input = each policy against its OWN training parquet (kit run_policy_evaluation.py + evaluate_policy.py "
          "160d24f5 on stock 8c778f9, frame 0, explicit reset, world count 1). Not a leaderboard number; no clean value "
          "was computed or read. Floor = runs with SP-SR = MP-SR = 1 (PREREG 5.4b: >= 4 of 5).", "",
          "| job | source | variants | input | ckpt sha256 | export | AUC runs 1-5 | mean | sd | min | max | SP-SR | MP-SR "
          "| MPPE cm | floor | designation single AUC |",
          "|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]  # fmt: skip
    for r in job_rows:
        runs = " ".join(D.fv(r.get(f"auc_r{k}")) or "-" for k in range(1, NRUNS + 1))
        md.append(f"| {r['job']} | {r.get('source', '')[:6]} | {r.get('variants')} | {r.get('input', '')} | "
                  f"{str(r.get('checkpoint_sha256', ''))[:12]} | {r.get('export_status', '')} {r.get('export_tolerance') or ''} | "
                  f"{runs} | {D.fv(r.get('auc_mean'))} | {D.fv(r.get('auc_sd'))} | {D.fv(r.get('auc_min'))} | "
                  f"{D.fv(r.get('auc_max'))} | {D.fv(r.get('sp_sr_mean'), '.2f')} | {D.fv(r.get('mp_sr_mean'), '.2f')} | "
                  f"{D.fv(r.get('mppe_mean'), '.2f')} | {r.get('floor_sp1_mp1_runs', '')}/{r.get('n_ok', '')} | "
                  f"{D.fv(r.get('designation_single_auc'))} |")
    D.write_atomic(OUT_SUM.with_suffix(".md"), "\n".join(md) + "\n")
    return {"jobs": job_rows}


# ================================================================================================ the driver
def make_driver(D):
    Base = TE.make_driver(D)  # pinned ConsDriver: no Stage B, no clean, t2_best5 live evals counted, all_final

    class A1Driver(Base):
        def build_manifest(self) -> None:
            data = D.read_json(OUT_MANIFEST)
            if data is not None:
                self.log(f"{OUT_MANIFEST.name} exists (built {data.get('built_at')}): reused")
            else:
                self.log("building manifest_a1.json (AMENDMENTS A3.3 / PREREG 6) and copying Zuxing's designated checkpoints")
                rows = build_manifest(D, self.log, copy=True)
                D.write_csv(OUT_MANIFEST_CSV, rows, MANIFEST_COLS)
                D.write_atomic(OUT_MANIFEST, D.dump({
                    "built_at": D.now(), "prereg_sha256": sha256_file(D.PREREG), "amendments_sha256": sha256_file(AMEND),
                    "prereg_pins_sha256": sha256_file(PINS), "prereg_pins_a1b_sha256": sha256_file(PINS_B),
                    "candidates_a1_sha256": sha256_file(CANDS), "t2best5_driver_sha256": TE.DRIVER_SHA,
                    "t2cons_eval_sha256": TE_SHA, "t2cons_a1_eval_sha256": sha256_file(Path(__file__).resolve()),
                    "t_a": "2026-10-26 17:29:25 CST", "reused_not_evaluated": REUSED, "rows": rows}))
                for r in rows:
                    self.log(f"  {r['job']}: P{r.get('priority')} {r.get('variants')} {r.get('root_kind')} "
                             f"{str(r.get('parquet_sha256'))[:8]} ckpt {str(r.get('training_sha256'))[:8]} tree {r.get('tree')}: "
                             f"{r['status']}")
            self.jobs = list((D.read_json(OUT_MANIFEST) or {}).get("rows", []))
            for r in self.jobs:
                if r.get("status") != "ok":
                    self.log.first(("refused", r["job"]), f"{r['job']}: NOT evaluated: {r.get('status')}")

        def summary(self) -> None:
            try:
                write_summary(D, self.jobs)
            except Exception as error:  # noqa: BLE001
                self.log.first(("summary", repr(error)), f"WARNING summary failed: {error!r}")
            self.last_summary = time.time()

        def progress(self) -> dict:  # as the pinned one, without looking for clean records
            ex = {"ok": 0, "refused": 0, "failed": 0, "pending": 0}
            ev = {"ok": 0, "failed": 0, "pending": 0, "no_export": 0}
            for job in self.ok_jobs():
                exp = D.export_rec(ROOT, job["job"])
                st = (exp or {}).get("status")
                ex["ok" if st == "ok" else "pending" if exp is None else "refused" if "refused" in str(st) or "incompatible" in str(st) else "failed"] += 1
                for k in range(1, self.a.runs + 1):
                    r = D.run_rec(ROOT, job["job"], k)
                    ev[("ok" if r.get("status") == "ok" else "failed") if r else ("no_export" if exp and st != "ok" else "pending")] += 1
            return {"jobs": len(self.ok_jobs()), "jobs_refused": len(self.jobs) - len(self.ok_jobs()), "runs_per_job": self.a.runs,
                    "exports": ex, "evals": ev, "clean": "not run (PREREG 6)"}  # fmt: skip

        def heartbeat(self, phase: str) -> None:
            prog = self.progress()
            mean_s = sum(self.durations) / len(self.durations) if self.durations else None
            jobs = [{"kind": j["kind"], "job": j["job"], "run": j.get("run"), "try": j["try"], "gpu": j.get("gpu"),
                     "pid": j["proc"].pid, "elapsed_s": round(time.time() - j["start"]), "overdue": j["overdue"]}
                    for j in self.running.values()]  # fmt: skip
            jobs += [{"kind": o["kind"], "job": o["job"], "run": o.get("run"), "try": o["try"], "gpu": o.get("gpu"),
                      "orphan_pids": [p for p, _ in o["procs"]]} for o in self.orphans.values()]
            D.write_atomic(OUT_HB, D.dump({
                "updated_at": D.now(), "updated_epoch": time.time(), "pid": os.getpid(), "phase": phase, **prog,
                "running": jobs, "t2best5_evals_counted": len(self.ext), "mean_eval_seconds": mean_s, "halted": self.halted,
                "stop_file": (ROOT / "STOP").exists()}))

    return A1Driver


# ================================================================================================ commands
def dry_run(D, a) -> int:
    lines: list[str] = []

    def p(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    p(f"t2cons_a1_eval DRY RUN {D.now()}: nothing is copied, exported, evaluated, locked or signalled. Written: "
      f"dryrun/a1_eval_plan_*.txt and the read-only hull check's preflight/hulls_*.")
    fatal, info = check_pins(D, require_pins_file=False)
    f2, i2 = D.preflight()
    fatal += f2
    for x in info + i2:
        p(f"preflight: {x}")
    for x in fatal:
        p(f"preflight FAILED: {x}")
    rows = build_manifest(D, D.Logger(None), copy=False)
    for r in rows:
        des = (r.get("designation_evidence") or {})
        row = des.get("row") or {}
        p(f"{r['job']}: P{r.get('priority')} variants {r.get('variants')} {r.get('root_kind')} tree {r.get('tree')} parquet "
          f"{str(r.get('parquet_sha256'))[:12]} ckpt {r.get('training_sha256')} designation {des.get('rule')} "
          f"(single own-input AUC {row.get('add_auc')}): {r['status']}")
        for n in r.get("notes", []):
            p(f"    note: {n}")
        p(f"    checkpoint used for export: {r.get('training_checkpoint')}")
    for job, rr in reused_records().items():
        aucs = [rr["runs"][k]["auc"] for k in sorted(rr["runs"])]
        p(f"reused {job}: snapshot sha matches {rr['sha_matches']}, {len(aucs)} ok runs, own-input AUC mean "
          f"{(sum(aucs) / len(aucs)) if aucs else None}: NOT evaluated again")
    try:
        state = D.smi()
        mps = D.mps_gpus(state)
        drv = make_driver(D)(a, D.Logger(None))
        drv.ext = TE.external_evals(D)
        p(f"t2_best5 live evals: {drv.ext}")
        for g in drv.gpu_rows(state, mps, []):
            p(f"gpu {g['gpu']}: free {g['free']} MiB, apps {g['napps']}, lock file {g['lock_file']}: {g['why']}")
    except Exception as error:  # noqa: BLE001
        p(f"gpu check failed: {error!r}")
    ok = not fatal and all(r["status"] == "ok" for r in rows)
    p(f"dry run: {'OK' if ok else 'PROBLEMS'}")
    D.write_atomic(ROOT / "dryrun" / f"a1_eval_plan_{D.stamp()}.txt", "\n".join(lines) + "\n")
    return 0 if ok else 1


def status(D) -> int:
    hb = D.read_json(OUT_HB) or {}
    print(f"heartbeat {hb.get('updated_at')} phase {hb.get('phase')} exports {hb.get('exports')} evals {hb.get('evals')} "
          f"halted {hb.get('halted')} t2best5 evals counted {hb.get('t2best5_evals_counted')}")
    for j in hb.get("running", []):
        print(f"  running {j}")
    for r in (D.read_json(OUT_MANIFEST) or {}).get("rows", []):
        exp = D.export_rec(ROOT, r["job"]) or {}
        aucs = []
        for k in range(1, NRUNS + 1):
            rr = D.run_rec(ROOT, r["job"], k)
            aucs.append("-" if rr is None else (f"{rr['metrics']['auc']:.4f}" if rr.get("status") == "ok" else "FAIL"))
        print(f"  {r['job']:<24} P{r.get('priority')} {r.get('variants'):<4} {str(r.get('root_kind')):<6} export "
              f"{exp.get('status', 'pending'):<10} {str(exp.get('tolerance') or ''):<7} runs {' '.join(aucs)}"
              + ("" if r.get("status") == "ok" else f"  ** {r.get('status')}"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--gpus", default="0,1,2,5,6,7")
    r.add_argument("--max-parallel", type=int, default=4)
    r.add_argument("--lane-max", type=int, default=2)
    r.add_argument("--runs", type=int, default=NRUNS)
    r.add_argument("--poll", type=float, default=10.0)
    sub.add_parser("summary")
    sub.add_parser("status")
    a = ap.parse_args()
    D = load_driver()
    if a.cmd == "run":
        if a.runs != NRUNS or not 1 <= a.max_parallel <= 4 or not 1 <= a.lane_max <= 2:
            ap.error("PREREG: --runs 5, --max-parallel <= 4, --lane-max <= 2")
        if a.dry_run:
            return dry_run(D, a)
        return make_driver(D)(a, D.Logger(OUT_LOG)).run()
    if a.cmd == "summary":
        out = write_summary(D, list((D.read_json(OUT_MANIFEST) or {}).get("rows", [])))
        print(json.dumps({r["job"]: {"n_ok": r.get("n_ok"), "auc_mean": r.get("auc_mean"),
                                     "floor": r.get("floor_sp1_mp1_runs")} for r in out["jobs"]}, indent=1))
        return 0
    return status(D)


if __name__ == "__main__":
    raise SystemExit(main())
