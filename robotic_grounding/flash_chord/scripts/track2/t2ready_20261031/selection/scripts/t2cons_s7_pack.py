#!/usr/bin/env python3
"""t2cons_s7_pack.py: step S6 (PREREG section 7, as amended by AMENDMENTS A2.3) of
fc_runs/submissions/t2_consensus_20261024: pack and LOCAL clean scoring, after the selection froze (SELECTION.sha256).

Files (PREREG 7.2 under A2.3):
  CONS-B   the variant-B pick of every episode (selection.json pick_B): CLOSEST and T_1..T_5
  DEFAULT  the t2_best5 snapshot SELECTED (selection.json default):    T_1..T_5 (comparison only)
  CONS-A is neither packed nor clean-scored (A2.3).
  T_k uses run k (its ok try) of each episode's job. CLOSEST uses, per episode, the run named in selection.json
  cons_B_closest_runs (variant-B consensus AUC closest to the job's 5-run mean, ties lower k), re-derived here.

Commands
  check     dry, no clean file: gate chain, compositions, rollout + metrics-json + frame-reference sources -> dryrun/
  prepare   per file: results/ (byte copies of the chosen runs' episode parquet + json; parquet sha256 must equal
            inputs_a1/ROLLOUTS_A1c.sha256), frame_refs/ (PREREG 7.3: byte copy of the job's own input npz for K3fix / K3 /
            K3CA / G2, sha256 = inputs/TRACKS.sha256; for raw an npz written from the released Tier-2 parquet through
            read_episode and checked equal to the packer default), file.json. No clean file.
  pack      the kit packer, unchanged, through scripts/track2_tier2_clean_scores.py (b58e15e0), PREREG 7.5 command with
            the placeholder commit URL. The packer reads the clean Tier-1 reference (LOCAL). Kit hashed before/after.
  validate  the kit uploader's validate_submission() (the offline check of v2d_submit.py --dry-run; Kaggle is never
            contacted) + provenance checks (uniform digests, per-episode checkpoint = the chosen job's export).
  score     the official metric_code/track_2/{AUC,SP-SR,MP-SR,MPPE}.py score() (PREREG 7.6) on each packed CSV, overall
            and per episode; cross-check against the packer's per-episode values (expected within 1e-3).
  report    summary_LOCAL_NOT_FOR_UPLOAD.{md,json,csv} (PREREG 7.7) + quoted t2_best5 LOCAL numbers + code / commit list.

Gate (every command): PREREG.md, AMENDMENTS.md, PREREG_PINS{,_A1b,_A1c}.md equal their sha256 files and the values pinned
below; SELECTION.sha256 exists, has the pinned sha256 and all 27 entries verify; selection.json is the frozen one
(binding variant B); the pinned t2cons_consensus.check_pins (PREREG 11 pins and commits, SNAPSHOT, KIT) passes; the
wrapper, kit packer, uploader, metric code and KIT_MANIFEST verify. Every command except check also needs
PREREG_PINS_S7.md == PREREG_PINS_S7.sha256, pinning this script; every pin in it must hold.

Everything is written under pack_LOCAL_NOT_FOR_UPLOAD/ (new files only). Nothing that sections 2-5 read is written.
Nothing here changes the selection (PREREG 7.7). Nothing is uploaded. Nothing is signalled. CPU only.
Run: conda_envs/v2dkit/bin/python -I -B scripts/t2cons_s7_pack.py <cmd> [--files ...] [--jobs N]
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
ZU = Path("/root/autodl-tmp/zuxing.lu")  # read only (files; no git)
ROOT = ME / "fc_runs/submissions/t2_consensus_20261024"
OUT = ROOT / "pack_LOCAL_NOT_FOR_UPLOAD"
KIT = ME / "v2d_submission_kit_src/v2d_submission_kit"
KPY = ME / "conda_envs/v2dkit/bin/python"
DATA = ME / "data/video_to_data_challenge"
B5 = ME / "fc_runs/submissions/t2_best5_20261024"  # read only
WRAPPER = ME / "scripts/track2_tier2_clean_scores.py"
A1_FILE = ME / "scripts/t2cons_a1_consensus.py"
PINS_S7, PINS_S7_SHA = ROOT / "PREREG_PINS_S7.md", ROOT / "PREREG_PINS_S7.sha256"
SEL, SEL_SHA = ROOT / "selection.json", ROOT / "SELECTION.sha256"
PLACEHOLDER = "https://github.com/local-placeholder/not-uploaded/commit/0000000000000000000000000000000000000000"
TZ8 = dt.timezone(dt.timedelta(hours=8))
FROZEN = {
    "PREREG.md": "3eee7268d53d1c3669614c63b69a54fa2462cf624ed88c3f7e215f5cb9f51ee9",
    "AMENDMENTS.md": "f648226a95a061eeefcfd5138cb01823db8a060fe121a0f4968ffa163dc4229f",
    "PREREG_PINS.md": "9d1493797205e8c60255c50658c536714a3702fe8751c724f37ed1444a040ece",
    "PREREG_PINS_A1b.md": "7239bc561eacc1d3ed6b3f210ae80193c30ad3168d8e0255dd7ee63016bd7c7a",
    "PREREG_PINS_A1c.md": "b740e43730fa24db0724e1ab42b1f1dc171bcbf95fa5a915c618c04b63894830",
}
SELECTION_SHA256_SHA = "651853a61bac0ab2c244a0e7ebed9db2d46d90f7d22d95f12fd545cda6b25046"
SELECTION_JSON_SHA = "e5804138b074a9927b30bed598ec1a13a670eb04116b9dc906bd904190755dba"
SELECTION_ENTRIES = 27
A1_SHA = "5dd430ec78f32cdff7921cfdcc5eca2b9de2a925cf29f6c41b141c7d0a580658"
WRAPPER_SHA = "b58e15e067687dc06b781e66efaec879fe1859f7af93e992005fbac1d581db26"
KIT_PINS = {
    "tools/pack_submission.py": "47a4589574ff8284ba4376ef31ac5548c8fe87ead753be2d3b43d6c87041e5be",
    "submit/v2d_submit.py": "fe5cb1dc86d12abd01eab9dd5dd98b51569b5d8513a2d00d4e79da42a5f11718",
    "metric_code/track_2/AUC.py": "58b437697c32c86497ff5a306fe586014c3429b0d2c18e1d3094a6f503479339",
    "metric_code/track_2/SP-SR.py": "d206042872acd1c1de6120f96966272495fbbd71a50315f49f5192fea1aaa5bf",
    "metric_code/track_2/MP-SR.py": "a1d72216eb7745a13ed7db6a55b9efb31492b35149113181a6c42ffc231fff11",
    "metric_code/track_2/MPPE.py": "41815bc1e8870fb377036750976dcf135818e029f06096ccef5177090fe5e033",
    "v2dlb/vertices.py": "4c95a4d65d37ea97d6c58388b273c277cad9d7cfa63d3cdb7af2ec3d1f1a0262",
    "v2dlb/dataset.py": "6faa9c040d73f0a9cc29188b24b8aeef8cb17c013edf2b5e55818ceedc8254a5",
    "flash_chord/scripts/evaluate_policy.py": "160d24f5fbeb70ccd569ca4192056f1892b2a9f2999066db9d8279461648fdf1",
}
EVAL_SCRIPT_SHA = "160d24f5fbeb70ccd569ca4192056f1892b2a9f2999066db9d8279461648fdf1"
EVAL_CODE_SHA = "b1b3ebe3ae4629d1687a83f420149e3f6fdfc154d70acad55c12fc8fa69811a1"
METRICS = ("auc", "sp_sr", "mp_sr", "mppe")
METRIC_FILE = {"auc": "AUC", "sp_sr": "SP-SR", "mp_sr": "MP-SR", "mppe": "MPPE"}
PACKER_COL = {"auc": "t2clean_add_auc", "sp_sr": "t2clean_spider_sr", "mp_sr": "t2clean_maniptrans_sr",
              "mppe": "t2clean_mppe_cm"}  # fmt: skip
NRUNS = 5
XCHECK_TOL = 1e-3
FILES = ["consB_closest"] + [f"consB_T{k}" for k in range(1, NRUNS + 1)] + [f"default_T{k}" for k in range(1, NRUNS + 1)]


# ================================================================================================ pinned code
def _load(name: str, file: Path, want: str):
    src = file.read_bytes()
    got = hashlib.sha256(src).hexdigest()
    if got != want:
        raise SystemExit(f"{file} sha256 {got} != pinned {want}")
    mod = types.ModuleType(name)
    mod.__file__ = str(file)
    sys.modules[name] = mod
    exec(compile(src, str(file), "exec"), mod.__dict__)  # noqa: S102 (verified pinned bytes)
    return mod


A1 = _load("t2cons_a1_consensus_pinned", A1_FILE, A1_SHA)  # loads t2cons_a1_gate.py and t2cons_consensus.py (pinned)
C = A1.C
MINE = Path(__file__).resolve()


# ================================================================================================ helpers
def now() -> str:
    return dt.datetime.now(TZ8).strftime("%Y-%m-%d %H:%M:%S")


def stamp() -> str:
    return dt.datetime.now(TZ8).strftime("%Y%m%d_%H%M%S")


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def first_token(p: Path) -> str:
    return (p.read_text().split() or [""])[0] if p.is_file() else ""


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def dump(obj) -> str:
    return json.dumps(obj, indent=1, default=_default) + "\n"


def write_new(p: Path, data) -> None:
    """New files only: refuse to overwrite; write through a temporary name in the same directory."""
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp_s7")
    if tmp.exists():
        raise SystemExit(f"stale temporary {tmp}")
    if isinstance(data, str):
        tmp.write_text(data)
    else:
        tmp.write_bytes(data)
    os.replace(tmp, p)


def write_csv_new(p: Path, rows: list[dict], cols: list[str]) -> None:
    import io

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    write_new(p, buf.getvalue())


def copy_new(src: Path, dst: Path, want: str | None = None) -> str:
    h = sha(src)
    if want is not None and h != want:
        raise SystemExit(f"{src} sha256 {h[:16]} != expected {want[:16]}")
    if dst.exists():
        raise SystemExit(f"refusing to overwrite {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    if sha(dst) != h:
        raise SystemExit(f"copy of {src} differs")
    return h


def read_json(p: Path):
    return json.loads(Path(p).read_text())


def fmt(x, spec=".4f") -> str:
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


# ================================================================================================ gate
def gate(need_s7: bool) -> tuple[list[str], list[str]]:
    fatal, info = [], []
    for name, want in FROZEN.items():
        p = ROOT / name
        got = sha(p) if p.is_file() else "missing"
        rec = first_token(p.with_suffix(".sha256"))
        if got != want or rec != want:
            fatal.append(f"{name}: sha256 {got[:16]} / its .sha256 {rec[:16]} != pinned {want[:16]}")
    info.append("PREREG.md, AMENDMENTS.md, PREREG_PINS.md, PREREG_PINS_A1b.md, PREREG_PINS_A1c.md == their sha256 files"
                " == pinned")  # fmt: skip
    if not SEL_SHA.is_file():
        fatal.append("SELECTION.sha256 does not exist: section 7 may not run (PREREG 7)")
        return fatal, info
    got = sha(SEL_SHA)
    if got != SELECTION_SHA256_SHA:
        fatal.append(f"SELECTION.sha256 sha256 {got[:16]} != pinned {SELECTION_SHA256_SHA[:16]}")
    n = 0
    for line in SEL_SHA.read_text().splitlines():
        f = line.split(None, 1)
        if len(f) != 2 or len(f[0]) != 64:
            continue
        n += 1
        p = ROOT / f[1].strip()
        h = sha(p) if p.is_file() else "missing"
        if h != f[0]:
            fatal.append(f"SELECTION.sha256: {f[1].strip()} {h[:16]} != {f[0][:16]}")
    if n != SELECTION_ENTRIES:
        fatal.append(f"SELECTION.sha256 has {n} entries, not {SELECTION_ENTRIES}")
    info.append(f"SELECTION.sha256 ({got[:16]}...): {n} entries verify")
    if sha(SEL) != SELECTION_JSON_SHA:
        fatal.append("selection.json is not the frozen file")
        return fatal, info
    sel = read_json(SEL)
    if sel.get("binding_variant") != "B" or sel.get("synthetic") is not False or sel.get("noisy_only") is not True:
        fatal.append("selection.json: binding_variant / synthetic / noisy_only unexpected")
    for key, name in (("prereg_sha256", "PREREG.md"), ("amendments_sha256", "AMENDMENTS.md"),
                      ("pins_sha256", "PREREG_PINS.md"), ("pins_a1b_sha256", "PREREG_PINS_A1b.md"),
                      ("pins_a1c_sha256", "PREREG_PINS_A1c.md")):  # fmt: skip
        if sel.get(key) != FROZEN[name]:
            fatal.append(f"selection.json {key} != {name}")
    if sel.get("script_sha256") != A1_SHA:
        fatal.append("selection.json was not written by the pinned t2cons_a1_consensus.py")
    info.append("selection.json: the frozen file (e5804138...), binding variant B, written by t2cons_a1_consensus.py 5dd430ec")
    f2, i2 = C.check_pins(need_pins=False)
    fatal += f2
    npins = sum(1 for x in i2 if x.startswith(("pin ", "commit ")))
    info.append(f"pinned t2cons_consensus.check_pins: {npins} PREREG-11 pin / commit lines, {len(f2)} fatal")
    info += [x for x in i2 if "path guard" in x or x.startswith(("inputs/", "kit files"))]
    h = sha(WRAPPER)
    if h != WRAPPER_SHA:
        fatal.append(f"{WRAPPER.name} sha256 {h[:16]} != pinned b58e15e0 (PREREG 11)")
    info.append(f"scripts/track2_tier2_clean_scores.py == {WRAPPER_SHA[:16]} (PREREG 11 pin, checked at S6)")
    for rel, want in KIT_PINS.items():
        h = sha(KIT / rel) if (KIT / rel).is_file() else "missing"
        if h != want:
            fatal.append(f"kit {rel} sha256 {h[:16]} != pinned {want[:16]}")
    man = read_json(KIT / "KIT_MANIFEST.json")["files"]
    bad = [p for p, hh in man.items() if not (KIT / p).is_file() or sha(KIT / p) != hh]
    if bad:
        fatal.append(f"KIT_MANIFEST mismatches: {bad[:5]}")
    info.append(f"kit pins ({len(KIT_PINS)}) hold; KIT_MANIFEST: {len(man)} files verify, {len(bad)} mismatched")
    if need_s7:
        if not PINS_S7.is_file():
            fatal.append("PREREG_PINS_S7.md is missing")
            return fatal, info
        g, r = sha(PINS_S7), first_token(PINS_S7_SHA)
        if g != r:
            fatal.append(f"PREREG_PINS_S7.md sha256 {g[:16]} != PREREG_PINS_S7.sha256 {r[:16]}")
        text = PINS_S7.read_text()
        mine = sha(MINE)
        if f"pin scripts/t2cons_s7_pack.py {mine}" not in text:
            fatal.append(f"PREREG_PINS_S7.md does not pin this script ({mine[:16]})")
        k = 0
        for line in text.splitlines():
            f = line.split()
            if len(f) == 3 and f[0] == "pin" and len(f[2]) == 64:
                if f[1].startswith("kit:"):
                    p = KIT / f[1][4:]
                elif f[1].startswith("scripts/"):
                    p = ME / f[1]
                else:
                    p = ROOT / f[1]
                hh = sha(p) if p.is_file() else "missing"
                k += 1
                if hh != f[2]:
                    fatal.append(f"PREREG_PINS_S7 pin {f[1]}: {hh[:16]} != {f[2][:16]}")
        info.append(f"PREREG_PINS_S7.md ({g[:16]}) == its sha256 file; {k} pins hold; pins this script {mine[:16]}")
    return fatal, info


def require(need_s7: bool) -> bool:
    fatal, info = gate(need_s7)
    for x in info:
        print("  " + x)
    if fatal:
        print("FATAL:\n  " + "\n  ".join(fatal))
        return False
    return True


# ================================================================================================ compositions
def manifest_rows() -> dict[str, dict]:
    rows = list(read_json(ROOT / "inputs/t2_best5/manifest.json")["rows"])
    rows += [json.loads(ln) for ln in (ROOT / "inputs/t2_best5/manifest_stageB.jsonl").read_text().splitlines() if ln.strip()]
    rows += list(read_json(ROOT / "manifest_new.json")["rows"])
    rows += list(read_json(ROOT / "manifest_a1.json")["rows"])
    return {r["job"]: r for r in rows}


def compositions(sel: dict, jobs: dict, evals: dict) -> tuple[dict[str, dict[int, tuple[str, int]]], list[str]]:
    snap = {int(x["episode"]): x["selected"] for x in read_json(ROOT / "inputs/t2_best5/selection.json")["episodes"]}
    jst = read_json(ROOT / "scores/consensus_job_stats.json")["jobs"]
    eps = {int(x["episode"]): x for x in sel["episodes"]}
    files: dict[str, dict[int, tuple[str, int]]] = {f: {} for f in FILES}
    problems = []
    if sorted(eps) != list(range(30)):
        problems.append("selection.json does not list episodes 0..29")
    for e in range(30):
        x, key = eps[e], f"{e:02d}"
        pick, D = x["pick_B"], x["default"]
        if x.get("pick") != pick or x.get("binding_variant") != "B" or sel["compositions"]["CONS-B"][key] != pick:
            problems.append(f"ep{key}: pick / binding / compositions CONS-B disagree")
        if sel["compositions"]["DEFAULT"][key] != D or snap[e] != D:
            problems.append(f"ep{key}: default {D} != compositions DEFAULT / snapshot SELECTED {snap[e]}")
        cr = sel["cons_B_closest_runs"][key]
        closest = int(cr["run"])
        if cr["job"] != pick or int(x["closest_run_B"]) != closest or int(jst[pick]["B"]["closest_run"]) != closest:
            problems.append(f"ep{key}: CLOSEST run records disagree")
        aucs, runs = list(jst[pick]["B"]["auc"]), list(jst[pick]["B"]["runs"])
        if runs != list(range(1, NRUNS + 1)) or len(aucs) != NRUNS:
            problems.append(f"ep{key}: {pick} has runs {runs}")
        m = float(np.mean(np.asarray(aucs, dtype=np.float64)))
        redo = runs[min(range(len(aucs)), key=lambda i: (abs(aucs[i] - m), runs[i]))]
        if redo != closest:
            problems.append(f"ep{key}: CLOSEST re-derived {redo} != recorded {closest}")
        for job in (pick, D):
            if job not in jobs or jobs[job]["episode"] != e:
                problems.append(f"ep{key}: {job} is not a job of this episode")
            if sorted(evals.get(job, {})) != list(range(1, NRUNS + 1)):
                problems.append(f"ep{key}: {job} lacks 5 ok runs")
        if jobs.get(pick, {}).get("input") == "g2" and "B" not in jobs[pick]["variants"]:
            problems.append(f"ep{key}: G2 pick outside variant B")
        files["consB_closest"][e] = (pick, closest)
        for k in range(1, NRUNS + 1):
            files[f"consB_T{k}"][e] = (pick, k)
            files[f"default_T{k}"][e] = (D, k)
    return files, problems


def episode_source(K, jobs: dict, evals: dict, rollouts: dict, job: str, k: int, e: int) -> tuple[dict, list[str]]:
    """The chosen run's rollout + metrics json and the job's own frame reference (all noisy-side; path guard on)."""
    j, rec = jobs[job], evals[job][k]
    pq = Path(rec["parquet"])
    problems = []
    h = C.sha256_file(pq)
    if rollouts.get(str(pq)) != h:
        problems.append(f"{job} run {k}: rollout sha256 {h[:16]} != ROLLOUTS_A1c.sha256")
    js = pq.with_suffix(".json")
    jh = C.sha256_file(js) if js.is_file() else "missing"
    rep = C.read_json(js) if js.is_file() else {}
    if rep.get("checkpoint_sha256") != j["export_sha256"]:
        problems.append(f"{job} run {k}: metrics json checkpoint_sha256 != export")
    traj = K.read_trajectory(C.guard(pq))
    prov = K.provenance(traj)
    if traj.episode_index != e or prov.get("checkpoint_sha256") != j["export_sha256"]:
        problems.append(f"{job} run {k}: parquet episode / checkpoint provenance mismatch")
    if prov.get("eval_script_sha256") != EVAL_SCRIPT_SHA or prov.get("eval_code_sha256", EVAL_CODE_SHA) != EVAL_CODE_SHA:
        problems.append(f"{job} run {k}: evaluator digests differ")
    kind = j["input"]
    if kind == "raw":
        fr_src = C.T2 / "data/chunk-000" / f"episode_{e:06d}.parquet"
    else:
        fr_src = Path(j["input_reference"]) / f"episode_{e:06d}.npz"
    fr_h = C.sha256_file(fr_src) if fr_src.is_file() else "missing"
    if K.tracks_sha.get(str(fr_src)) != fr_h:
        problems.append(f"{job}: frame reference source {fr_src} sha256 {fr_h[:16]} != inputs/TRACKS.sha256")
    return {"episode": e, "job": job, "run": k, "try": int(rec["try"]), "input": kind,
            "export_sha256": j["export_sha256"], "rollout_src": str(pq), "rollout_sha256": h,
            "json_src": str(js), "json_sha256": jh, "frame_ref_kind": "written_from_released_tier2" if kind == "raw"
            else "byte_copy", "frame_ref_src": str(fr_src), "frame_ref_src_sha256": fr_h,
            "input_reference": j["input_reference"]}, problems  # fmt: skip


def load_all(K):
    jobs = A1.load_jobs_a1()
    evals = A1.load_evals_a1(jobs)
    sel = read_json(SEL)
    files, problems = compositions(sel, jobs, evals)
    return sel, jobs, evals, files, problems


def targets(a) -> list[str]:
    names = a.files or FILES
    bad = [f for f in names if f not in FILES]
    if bad:
        raise SystemExit(f"unknown files {bad}; known {FILES}")
    return names


# ================================================================================================ check (dry)
def cmd_check(a) -> int:
    lines = [f"t2cons_s7_pack.py check at {now()} (dry; no clean file opened; nothing written except this file)",
             f"script sha256 {sha(MINE)}"]  # fmt: skip
    fatal, info = gate(need_s7=False)
    lines += ["", "GATE"] + ["  " + x for x in info]
    if fatal:
        lines += ["FATAL"] + ["  " + x for x in fatal]
        print("\n".join(lines))
        return 1
    K = C.load_kit()
    sel, jobs, evals, files, problems = load_all(K)
    rollouts = C.hash_list(A1.ROLLOUTS)
    lines += ["", "FILES (episode: job [input] run/try)"]
    seen = {}
    for f in FILES:
        lines.append(f"  {f}:")
        for e in range(30):
            job, k = files[f][e]
            if (job, k) not in seen:
                seen[(job, k)], p2 = episode_source(K, jobs, evals, rollouts, job, k, e)
                problems += p2
            s = seen[(job, k)]
            lines.append(f"    ep{e:02d}: {job} [{s['input']}] run {k} try {s['try']}  rollout {s['rollout_sha256'][:12]}  "
                         f"frame ref {s['frame_ref_kind']} {s['frame_ref_src']}")  # fmt: skip
    changed = [e for e in range(30) if files["consB_T1"][e][0] != files["default_T1"][e][0]]
    cl = files["consB_closest"]
    cl_text = " ".join(f"{e:02d}:{cl[e][1]}" for e in range(30))
    lines += ["", f"CONS-B changed vs DEFAULT: {', '.join(f'{e:02d}' for e in changed)} ({len(changed)})",
              f"CONS-B CLOSEST runs: {cl_text}",
              f"distinct (job, run) sources checked: {len(seen)}", "", "PROBLEMS: " + ("; ".join(problems) or "none")]  # fmt: skip
    text = "\n".join(lines) + "\n"
    write_new(OUT / "dryrun" / f"check_{stamp()}.txt", text)
    print(text)
    return 1 if problems else 0


# ================================================================================================ prepare
def write_raw_npz(K, e: int, dst: Path) -> dict:
    src = C.guard(C.T2 / "data/chunk-000" / f"episode_{e:06d}.parquet")
    r = K.read_episode(src, e)
    idx = np.asarray(r.frame_index, dtype=np.int64)
    pose = np.asarray(r.pose_xyzw, dtype=np.float64)
    if dst.exists():
        raise SystemExit(f"refusing to overwrite {dst}")
    np.savez(dst, episode_index=np.int64(e), coordinate_frame=np.array("released_track2_world"), frame_index=idx,
             pose_xyzw=pose)  # fmt: skip
    return {"frames": int(len(idx)), "objects": list(r.object_names)}


def cmd_prepare(a) -> int:
    if not require(need_s7=True):
        return 1
    K = C.load_kit()
    sel, jobs, evals, files, problems = load_all(K)
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    rollouts = C.hash_list(A1.ROLLOUTS)
    for f in targets(a):
        d = OUT / f
        if d.exists():
            print(f"{f}: exists, skipped")
            continue
        tmpd = OUT / f".{f}.preparing"
        if tmpd.exists():
            raise SystemExit(f"stale {tmpd}")
        (tmpd / "results").mkdir(parents=True)
        (tmpd / "frame_refs").mkdir()
        rows, probs = [], []
        for e in range(30):
            job, k = files[f][e]
            s, p2 = episode_source(K, jobs, evals, rollouts, job, k, e)
            probs += p2
            s["results_parquet_sha256"] = copy_new(Path(s["rollout_src"]), tmpd / "results" / f"episode_{e:06d}.parquet",
                                                   s["rollout_sha256"])  # fmt: skip
            s["results_json_sha256"] = copy_new(Path(s["json_src"]), tmpd / "results" / f"episode_{e:06d}.json",
                                                s["json_sha256"])  # fmt: skip
            npz = tmpd / "frame_refs" / f"episode_{e:06d}.npz"
            if s["input"] == "raw":
                s.update(write_raw_npz(K, e, npz))
            else:
                copy_new(Path(s["frame_ref_src"]), npz, s["frame_ref_src_sha256"])
            s["frame_ref_sha256"] = sha(npz)
            frames, names = C.grid(K, e)
            mine = K.input_frame_reference(tmpd / "frame_refs", e, frames, 1, names)
            own_root = Path(s["input_reference"])
            own = K.input_frame_reference(C.guard(own_root), e, frames, 1, names)
            s["frame_ref_equals_own_input_on_grid"] = bool(np.array_equal(mine, own))
            if s["input"] == "raw":
                s["frame_ref_equals_packer_default"] = bool(np.array_equal(
                    mine, K.input_frame_reference(C.T2, e, frames, 1, names)))
            if not s["frame_ref_equals_own_input_on_grid"] or s.get("frame_ref_equals_packer_default") is False:
                probs.append(f"{f} ep{e:02d}: frame reference differs from the job's own input on the grid")
            s["grid_frames"] = int(len(frames))
            rows.append(s)
        if probs:
            print(f"{f}: PROBLEMS\n  " + "\n  ".join(probs))
            return 1
        meta = {"file": f, "composition": "CONS-B" if f.startswith("consB") else "DEFAULT",
                "kind": "CLOSEST" if f.endswith("closest") else f"T_{f[-1]}", "created_at": now(),
                "script_sha256": sha(MINE), "selection_json_sha256": SELECTION_JSON_SHA,
                "selection_sha256_file_sha256": SELECTION_SHA256_SHA, "episodes": rows,
                "note": "LOCAL_NOT_FOR_UPLOAD: placeholder commit URL; never uploaded"}  # fmt: skip
        write_new(tmpd / "file.json", dump(meta))
        os.rename(tmpd, d)
        print(f"{f}: prepared ({len(rows)} episodes; inputs "
              f"{sorted(set(r['input'] for r in rows))}; changed-vs-default rows recorded in file.json)")
    return 0


# ================================================================================================ pack
def kit_hashes() -> dict[str, str]:
    out = {}
    for dirpath, dirnames, filenames in os.walk(KIT):
        dirnames[:] = [x for x in dirnames if x != ".git"]
        for name in filenames:
            p = Path(dirpath) / name
            out[str(p.relative_to(KIT))] = sha(p) if p.is_file() else "special"
    return out


def pack_command(d: Path) -> list[str]:
    return ["nice", "-n", "10", str(KPY), "-I", "-B", str(WRAPPER), "--kit", str(KIT),
            "--out", str(d / "packer_scores_LOCAL_NOT_FOR_UPLOAD.csv"), "--",
            "--track", "track_2_tier2", "--results", str(d / "results"), "--dataset-root", str(DATA),
            "--frame-references", str(d / "frame_refs"), "--code-commit", PLACEHOLDER,
            "--eval-script", str(KIT / "flash_chord/scripts/evaluate_policy.py"),
            "-o", str(d / "packed_LOCAL_NOT_FOR_UPLOAD.csv")]  # fmt: skip


def parse_packer_scores(p: Path) -> dict:
    rows = list(csv.DictReader(p.open()))
    per, mean = {}, {}
    for r in rows:
        vals = {m: float(r[PACKER_COL[m]]) for m in METRICS if r.get(PACKER_COL[m]) not in (None, "")}
        if r["episode"].isdigit():
            per[int(r["episode"])] = vals
        elif r["episode"] == "MEAN":
            mean = vals
    return {"per_episode": per, "mean": mean}


def cmd_pack(a) -> int:
    if not require(need_s7=True):
        return 1
    todo = []
    for f in targets(a):
        d = OUT / f
        if not (d / "file.json").is_file():
            print(f"{f}: not prepared")
            return 1
        if (d / "packed_LOCAL_NOT_FOR_UPLOAD.csv").exists() or (d / "pack_LOCAL_NOT_FOR_UPLOAD.log").exists():
            print(f"{f}: already packed, skipped")
            continue
        todo.append(f)
    before = kit_hashes()
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", "/root"),
           "LANG": "C.UTF-8", "TZ": "Asia/Shanghai", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}  # fmt: skip
    running, results = {}, {}
    queue = list(todo)
    while queue or running:
        while queue and len(running) < max(1, a.jobs):
            f = queue.pop(0)
            d = OUT / f
            cmd = pack_command(d)
            log = (d / "pack_LOCAL_NOT_FOR_UPLOAD.log").open("x")
            log.write("+ " + " ".join(cmd) + "\n")
            log.flush()
            running[f] = (subprocess.Popen(cmd, cwd=str(d), env=env, stdout=log, stderr=subprocess.STDOUT), log, time.time())
            print(f"[{now()}] {f}: packing")
        time.sleep(2)
        for f in list(running):
            proc, log, t0 = running[f]
            if proc.poll() is None:
                continue
            log.close()
            del running[f]
            d = OUT / f
            text = (d / "pack_LOCAL_NOT_FOR_UPLOAD.log").read_text()
            csvp = d / "packed_LOCAL_NOT_FOR_UPLOAD.csv"
            rec = {"file": f, "exit": proc.returncode, "seconds": round(time.time() - t0, 1), "finished_at": now(),
                   "command": pack_command(d), "csv_exists": csvp.is_file(),
                   "csv_sha256": sha(csvp) if csvp.is_file() else None,
                   "locally_computed_line": next((ln for ln in text.splitlines() if ln.startswith("locally computed:")), None),
                   "input_reference_line": next((ln for ln in text.splitlines() if ln.startswith("input reference check")), None),
                   "frame_reference_line": next((ln for ln in text.splitlines() if ln.startswith("frame conversion reference")), None),
                   "clean_reference_line": next((ln for ln in text.splitlines() if ln.startswith("clean scoring reference")), None)}  # fmt: skip
            ps = d / "packer_scores_LOCAL_NOT_FOR_UPLOAD.csv"
            if ps.is_file():
                rec["packer_scores_sha256"] = sha(ps)
                rec["packer_scores"] = parse_packer_scores(ps)
            rec["ok"] = bool(proc.returncode == 0 and csvp.is_file() and ps.is_file()
                             and len(rec.get("packer_scores", {}).get("per_episode", {})) == 30)  # fmt: skip
            write_new(d / "pack_LOCAL_NOT_FOR_UPLOAD.json", dump(rec))
            results[f] = rec
            print(f"[{now()}] {f}: exit {proc.returncode} in {rec['seconds']} s; {rec['locally_computed_line']}")
    after = kit_hashes()
    diff = sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p))
    run = {"at": now(), "files": todo, "ok": {f: r["ok"] for f, r in results.items()},
           "kit_files": len(after), "kit_unchanged": not diff, "kit_diff": diff[:50]}  # fmt: skip
    write_new(OUT / f"pack_run_{stamp()}.json", dump(run))
    print(f"kit unchanged: {not diff} ({len(after)} files)" + (f"; DIFF {diff[:10]}" if diff else ""))
    return 0 if all(r["ok"] for r in results.values()) and not diff else 1


# ================================================================================================ validate
def load_uploader():
    if str(KIT) not in sys.path:
        sys.path.insert(0, str(KIT))
    spec = importlib.util.spec_from_file_location("kit_v2d_submit", KIT / "submit" / "v2d_submit.py")
    up = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(up)
    return up


def pose_columns_sha(csvp: Path, npose: int) -> tuple[str, list[str]]:
    h = hashlib.sha256()
    header = None
    with csvp.open() as fp:
        for i, line in enumerate(fp):
            parts = line.rstrip("\n").split(",")
            if i == 0:
                header = parts
            h.update((",".join(parts[: 1 + npose]) + "\n").encode())
    return h.hexdigest(), header or []


def cmd_validate(a) -> int:
    if not require(need_s7=True):
        return 1
    C.load_kit()  # kit files == inputs/KIT.sha256; puts the kit on sys.path
    up = load_uploader()
    from v2dlb import object_pose as op
    from v2dlb import report_schema as rs

    config = read_json(KIT / "config/tracks.json")
    man = read_json(KIT / "KIT_MANIFEST.json")["files"]
    sample = KIT / "data" / "track_2_sample_submission.csv"
    import pandas as pd

    n_sample = len(pd.read_csv(sample, usecols=[rs.ROW_ID_COLUMN], dtype=str))
    bad = 0
    for f in targets(a):
        d = OUT / f
        csvp = d / "packed_LOCAL_NOT_FOR_UPLOAD.csv"
        outp = d / "validate_LOCAL_NOT_FOR_UPLOAD.json"
        if outp.exists():
            print(f"{f}: already validated, skipped")
            continue
        meta = read_json(d / "file.json")
        res = {"file": f, "at": now(), "csv_sha256": sha(csvp), "problems": []}
        try:
            frame = up.validate_submission(csvp, "track_2_tier2", config)
            res["uploader_validate_submission"] = "pass"
        except SystemExit as err:
            res["uploader_validate_submission"] = f"FAIL: {err}"
            res["problems"].append("uploader validate_submission failed")
            frame = pd.read_csv(csvp)
        res["rows"], res["sample_rows"] = int(len(frame)), n_sample
        uniform = {c: sorted(frame[c].astype(str).str.strip().unique()) for c in rs.UNIFORM_PROVENANCE_COLUMNS}
        res["uniform_columns"] = {c: v for c, v in uniform.items()}
        for c, v in uniform.items():
            if len(v) != 1:
                res["problems"].append(f"{c} not uniform")
        want = {"eval_script_sha256": EVAL_SCRIPT_SHA, "eval_code_sha256": EVAL_CODE_SHA,
                "packer_script_sha256": man["tools/pack_submission.py"],
                "uploader_script_sha256": man["submit/v2d_submit.py"], "code_commit_url": PLACEHOLDER}  # fmt: skip
        for c, v in want.items():
            if uniform.get(c, [None])[0] != v:
                res["problems"].append(f"{c} {uniform.get(c)} != {v}")
        eps = frame["row_id"].astype(str).str.split("/").str[1]
        per_ep = frame.groupby(eps)["checkpoint_sha256"].agg(lambda s: sorted(s.astype(str).str.strip().unique()))
        chosen = {f"e{r['episode']:06d}": r["export_sha256"] for r in meta["episodes"]}
        if sorted(per_ep.index) != sorted(chosen):
            res["problems"].append("episode set differs from the file's selection")
        for ep, vals in per_ep.items():
            if len(vals) != 1 or vals[0] != chosen.get(ep):
                res["problems"].append(f"{ep}: checkpoint digest {vals} != chosen export {str(chosen.get(ep))[:16]}")
        res["checkpoint_one_per_episode_and_equal_chosen_export"] = not any("checkpoint digest" in p for p in res["problems"])
        res["distinct_checkpoints"] = int(len({v[0] for v in per_ep}))
        npose = len(op._POSE_COLUMNS)
        res["pose_columns_sha256"], header = pose_columns_sha(csvp, npose)
        if header[: 1 + npose] != [rs.ROW_ID_COLUMN, *op._POSE_COLUMNS]:
            res["problems"].append(f"unexpected header {header[: 1 + npose]}")
        res["pose_columns"] = header[: 1 + npose]
        res["ok"] = not res["problems"]
        bad += 0 if res["ok"] else 1
        write_new(outp, dump(res))
        print(f"{f}: uploader validate_submission {res['uploader_validate_submission']}; {res['rows']:,} rows "
              f"(sample {n_sample:,}); {res['distinct_checkpoints']} distinct checkpoints; problems: "
              f"{res['problems'] or 'none'}")  # fmt: skip
    return 1 if bad else 0


# ================================================================================================ score (official metric code)
def cmd_score(a) -> int:
    if not require(need_s7=True):
        return 1
    C.load_kit()
    import pandas as pd
    from v2dlb import object_pose as op
    from v2dlb.dataset import episode_indices, participant_reference_root
    from v2dlb.trajectory import read_track
    from v2dlb.vertices import build_blob

    config = read_json(KIT / "config/tracks.json")
    spec = config["tracks"]["track_2_tier2"]
    ref_root = participant_reference_root(DATA, spec)
    if ref_root.name != "tier_1_multiview_caption":
        raise SystemExit(f"unexpected clean reference root {ref_root}")
    required = episode_indices(ref_root) if spec["scored_episodes"] == "all" else sorted(map(int, spec["scored_episodes"]))
    if list(required) != list(range(30)):
        raise SystemExit(f"scored episodes {required}")
    worlds = int(spec.get("worlds_per_episode", 1))
    reference = {t.episode_index: t for t in read_track(ref_root, required)}
    rows = []
    for ep in required:
        ref = reference[ep]
        for w in range(worlds):
            for step, frame in enumerate(ref.frame_index):
                for slot, name in enumerate(ref.object_names):
                    rows.append((f"track_2/e{ep:06d}/w{w:03d}/f{int(frame):06d}/o{slot}", ep, w, int(frame), slot, name,
                                 *ref.pose_xyzw[step, slot]))  # fmt: skip
    solution = pd.DataFrame(rows, columns=["row_id", "episode_index", "world_index", "frame_index", "object_slot",
                                           "object_name", *op._REF_COLUMNS])  # fmt: skip
    names = sorted(solution["object_name"].astype(str).unique())
    blob, _ = build_blob(ref_root / "mesh", names)
    mods = {}
    for m, fname in METRIC_FILE.items():
        s = importlib.util.spec_from_file_location(f"kit_metric_{m}", KIT / "metric_code/track_2" / f"{fname}.py")
        mod = importlib.util.module_from_spec(s)
        s.loader.exec_module(mod)
        mods[m] = mod

    def with_blob(sol):
        sol = sol.reset_index(drop=True).copy()
        sol["auc_vertex_samples_b64"] = ""
        sol.loc[0, "auc_vertex_samples_b64"] = blob
        sol["Usage"] = "Public"
        return sol

    full_solution = with_blob(solution)
    sol_by_ep = {ep: with_blob(solution[solution["episode_index"] == ep]) for ep in required}
    bad = 0
    for f in targets(a):
        d = OUT / f
        outp = d / "metric_code_LOCAL_NOT_FOR_UPLOAD.json"
        if outp.exists():
            print(f"{f}: already scored, skipped")
            continue
        t0 = time.time()
        csvp = d / "packed_LOCAL_NOT_FOR_UPLOAD.csv"
        sub = pd.read_csv(csvp)
        overall = {m: float(mod.score(full_solution.copy(), sub.copy(), "row_id")) for m, mod in mods.items()}
        sub_ep = sub["row_id"].str.split("/").str[1].str[1:].astype(int)
        per = {}
        for ep in required:
            se = sub[sub_ep == ep]
            per[ep] = {m: float(mod.score(sol_by_ep[ep].copy(), se.copy(), "row_id")) for m, mod in mods.items()}
        mean_check = {m: abs(float(np.mean([per[e][m] for e in required])) - overall[m]) for m in METRICS}
        pk = read_json(d / "pack_LOCAL_NOT_FOR_UPLOAD.json")["packer_scores"]["per_episode"]
        xdev = {m: max(abs(per[e][m] - float(pk[str(e)][m])) for e in required) for m in METRICS}
        res = {"file": f, "at": now(), "seconds": round(time.time() - t0, 1), "csv_sha256": sha(csvp),
               "clean_reference_root": str(ref_root), "metric_code_sha256": {m: sha(KIT / "metric_code/track_2" / f"{n}.py")
               for m, n in METRIC_FILE.items()}, "overall": overall,
               "per_episode": {str(e): per[e] for e in required}, "overall_minus_mean_of_episodes": mean_check,
               "max_abs_dev_vs_packer_per_episode": xdev,
               "crosscheck_vs_packer_ok": all(v <= XCHECK_TOL for v in xdev.values()),
               "note": "LOCAL, report only: clean (Tier-1) reference; never used for any Tier-2 decision"}  # fmt: skip
        if not res["crosscheck_vs_packer_ok"] or max(mean_check.values()) > 1e-9:
            bad += 1
        write_new(outp, dump(res))
        write_csv_new(d / "metric_code_per_episode_LOCAL_NOT_FOR_UPLOAD.csv",
                      [{"episode": e, **{m: per[e][m] for m in METRICS},
                        **{f"packer_{m}": float(pk[str(e)][m]) for m in METRICS}} for e in required],
                      ["episode", *METRICS, *[f"packer_{m}" for m in METRICS]])  # fmt: skip
        print(f"{f}: done in {res['seconds']} s; crosscheck vs packer ok {res['crosscheck_vs_packer_ok']} "
              f"(max dev {max(xdev.values()):.2e})")  # fmt: skip
    return 1 if bad else 0


# ================================================================================================ report
CODE_NOTES = {
    "ours_fc_r2": "training: code/fc_r2 @ 61fa5d31ff80 (C3r2 recipe, our run); export: code/fc_r2 @ 61fa5d3",
    "zuxing": "training: Zuxing's worktrees/dev.zuxing.speedup @ {commit} (experiment g1_recon_body_coord_rsi_r2_runin, "
              "tmp/t2/train_t2.sh); export: code/nv-v2d_runin @ b39b69b",
}
INPUT_CODE = {
    "raw": "released Tier-2 data, retargeted to G1 (data/human_motion_data_tier2); no filter",
    "k3fix": "K3fix filter: scripts/k3fix_build.py, k3fix_build_v2.py, k3fix_build_data.py, k3fix_helper.py "
             "(data/human_motion_data_tier2_k3fix; frame refs tmp_analysis/tier2/k3fix_pilot/frame_refs_k3fix)",
    "k3": "K3 easy-clip Kalman filter: tmp_analysis/tier2/kalman/eval/e1_k3_fullframe.py, e2_build_eval_inputs.py "
          "(data/human_motion_data_tier2_k3; frame refs tmp_analysis/tier2/kalman/eval/frame_refs_k3)",
    "k3ca": "K3CA filter: scripts/k3ca_build.py, k3ca_build_data.py, k3ca_helper.py (data/human_motion_data_tier2_k3ca)",
    "g2": "Zuxing's G2 filter: zuxing.lu/tmp/tier2_noise/kfix/gen/G2_surface_only/ (run_episode.py, prep.py, model.py, "
          "detect.py, proxies.py, common.py, run_all.sh, make_csv.py) + data build zuxing.lu/tmp/t2/g2data/build_g2_data.py "
          "(reuses our tmp_analysis/tier2/kalman/eval/e2_build_eval_inputs.py) -> "
          "zuxing.lu/data/human_motion_data_tier2_g2-2026-10-24 (+ frame_refs_g2)",
}


def zuxing_commit(run_dir: str) -> str:
    """The commit train_t2.sh printed in Zuxing's own training log (read only; no git command in his folder)."""
    p = Path(run_dir)
    log = ZU / "logs" / p.parent.name / f"{p.name}.log"
    if not log.is_file():
        return "unknown"
    with log.open(errors="replace") as fp:
        for _, line in zip(range(200), fp):
            if line.startswith("train_t2:") and "commit=" in line:
                return line.rsplit("commit=", 1)[1].strip()
    return "unknown"


def cmd_report(a) -> int:
    if not require(need_s7=True):
        return 1
    K = C.load_kit()
    sel, jobs, evals, files, problems = load_all(K)
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        return 1
    mrows = manifest_rows()
    per = {}
    for f in FILES:
        d = OUT / f
        need = ["file.json", "pack_LOCAL_NOT_FOR_UPLOAD.json", "validate_LOCAL_NOT_FOR_UPLOAD.json",
                "metric_code_LOCAL_NOT_FOR_UPLOAD.json"]  # fmt: skip
        miss = [x for x in need if not (d / x).is_file()]
        if miss:
            print(f"{f}: missing {miss}")
            return 1
        per[f] = {"meta": read_json(d / "file.json"), "pack": read_json(d / "pack_LOCAL_NOT_FOR_UPLOAD.json"),
                  "val": read_json(d / "validate_LOCAL_NOT_FOR_UPLOAD.json"),
                  "mc": read_json(d / "metric_code_LOCAL_NOT_FOR_UPLOAD.json")}  # fmt: skip
        got_pairs = [(r["job"], int(r["run"])) for r in per[f]["meta"]["episodes"]]
        if got_pairs != [files[f][e] for e in range(30)]:
            print(f"{f}: file.json differs from the frozen composition")
            return 1
    # quoted t2_best5 LOCAL numbers (live files of a running driver: copied once, then read from the copy)
    qd = OUT / "quotes"
    stamp_q = stamp()
    quotes = {}
    for name in ("summary_LOCAL_NOT_FOR_UPLOAD.json", "clean_evals_LOCAL_NOT_FOR_UPLOAD.csv"):
        src = B5 / name
        dst = qd / f"t2_best5_{stamp_q}_{name}"
        quotes[name] = {"source": str(src), "source_mtime": dt.datetime.fromtimestamp(src.stat().st_mtime, TZ8).strftime(
            "%Y-%m-%d %H:%M:%S"), "copy": str(dst), "sha256": copy_new(src, dst)}  # fmt: skip
    b5sum = read_json(qd / f"t2_best5_{stamp_q}_summary_LOCAL_NOT_FOR_UPLOAD.json")
    b5clean = {}
    for r in csv.DictReader((qd / f"t2_best5_{stamp_q}_clean_evals_LOCAL_NOT_FOR_UPLOAD.csv").open()):
        if r.get("score_status") == "ok":
            b5clean[(r["job"], int(r["run"]))] = {m: float(r[m]) for m in METRICS}
    quoted = {}
    for comp in ("SELECTED", "RAW-ONLY", "K3FULL"):
        c = b5sum["compositions"].get(comp, {})
        quoted[comp] = {"partial": c.get("partial"), "n_with_values": c.get("n_with_values"),
                        **{m: {k: c.get("metrics", {}).get(m, {}).get(k) for k in ("average", "T_min", "T_max",
                                                                                   "mean_of_episode_min",
                                                                                   "mean_of_episode_max")}
                           for m in METRICS}}  # fmt: skip
    b5_selected_jobs = {int(k): v for k, v in b5sum.get("composition_jobs", {}).get("SELECTED", {}).items()}
    default_jobs = {e: files["default_T1"][e][0] for e in range(30)}
    quoted["SELECTED_jobs_equal_DEFAULT"] = b5_selected_jobs == default_jobs
    quoted["SELECTED_jobs_differ_on"] = sorted(e for e in range(30) if b5_selected_jobs.get(e) != default_jobs[e])

    def comp_stats(prefix: str) -> dict:
        T = {k: per[f"{prefix}_T{k}"]["mc"] for k in range(1, NRUNS + 1)}
        out = {"T_k": {m: [T[k]["overall"][m] for k in range(1, NRUNS + 1)] for m in METRICS}}
        out["min"] = {m: min(out["T_k"][m]) for m in METRICS}
        out["mean"] = {m: float(np.mean(out["T_k"][m])) for m in METRICS}
        out["max"] = {m: max(out["T_k"][m]) for m in METRICS}
        out["per_episode"] = {}
        for e in range(30):
            vals = {m: [T[k]["per_episode"][str(e)][m] for k in range(1, NRUNS + 1)] for m in METRICS}
            out["per_episode"][e] = {m: {"mean": float(np.mean(v)), "min": min(v), "max": max(v), "runs": v}
                                     for m, v in vals.items()}  # fmt: skip
        out["mean_of_episode_means"] = {m: float(np.mean([out["per_episode"][e][m]["mean"] for e in range(30)]))
                                        for m in METRICS}  # fmt: skip
        out["mean_of_episode_min"] = {m: float(np.mean([out["per_episode"][e][m]["min"] for e in range(30)])) for m in METRICS}
        out["mean_of_episode_max"] = {m: float(np.mean([out["per_episode"][e][m]["max"] for e in range(30)])) for m in METRICS}
        return out

    cons, dflt = comp_stats("consB"), comp_stats("default")
    closest = per["consB_closest"]["mc"]
    # cross-check: same rollouts already clean-scored per run by t2_best5 (packer metric) -> must agree
    xc = {m: 0.0 for m in METRICS}
    n_xc = 0
    for f in FILES:
        for r in per[f]["meta"]["episodes"]:
            key = (r["job"], r["run"])
            if key in b5clean:
                n_xc += 1
                for m in METRICS:
                    xc[m] = max(xc[m], abs(per[f]["mc"]["per_episode"][str(r["episode"])][m] - b5clean[key][m]))
    # per-episode table
    eps = {int(x["episode"]): x for x in sel["episodes"]}
    table = []
    for e in range(30):
        pick, D = files["consB_T1"][e][0], files["default_T1"][e][0]
        cr = files["consB_closest"][e][1]
        row = {"episode": e, "object": eps[e]["object"], "default": D, "default_input": jobs[D]["input"],
               "pick_B": pick, "pick_input": jobs[pick]["input"], "changed": pick != D, "closest_run": cr,
               "pick_A_report_only": eps[e]["pick_A"]}  # fmt: skip
        for m in METRICS:
            for tag, st in (("consB", cons), ("default", dflt)):
                pe = st["per_episode"][e][m]
                row[f"{tag}_{m}_mean"], row[f"{tag}_{m}_min"], row[f"{tag}_{m}_max"] = pe["mean"], pe["min"], pe["max"]
            row[f"closest_{m}"] = closest["per_episode"][str(e)][m]
            row[f"delta_{m}_mean"] = row[f"consB_{m}_mean"] - row[f"default_{m}_mean"]
        table.append(row)
    # code / commit list for every CONS-B checkpoint
    code_rows = []
    for e in range(30):
        job = files["consB_T1"][e][0]
        mr = mrows.get(job, {})
        base = B5 if jobs[job]["source"] == "t2_best5" else ROOT
        exp = read_json(base / "exports" / job / "export.json")
        owner = mr.get("owner")
        commit = zuxing_commit(mr.get("run_dir", "")) if owner == "zuxing" else "61fa5d3"
        code_rows.append({"episode": e, "job": job, "owner": owner, "family": mr.get("family"),
                          "input": jobs[job]["input"], "iteration": mr.get("iteration"), "run_dir": mr.get("run_dir"),
                          "training_sha256": mr.get("training_sha256"), "export_sha256": jobs[job]["export_sha256"],
                          "training_code": (CODE_NOTES["zuxing"].format(commit=commit) if owner == "zuxing"
                                            else CODE_NOTES["ours_fc_r2"]),
                          "export_tree": f"{exp.get('tree')} @ {str(exp.get('code_commit'))[:12]}",
                          "input_code": INPUT_CODE.get(jobs[job]["input"], "?")})  # fmt: skip
    summary = {
        "generated_at": now(), "script_sha256": sha(MINE), "pins_s7_sha256": sha(PINS_S7),
        "selection_json_sha256": SELECTION_JSON_SHA, "selection_sha256_file_sha256": SELECTION_SHA256_SHA,
        "note": "LOCAL_NOT_FOR_UPLOAD. Clean (Tier-1) reference. Report only: nothing here changes the selection, the "
                "variant or the rule (PREREG 7.7). CONS-A is not packed or clean-scored (AMENDMENTS A2.3).",
        "files": {f: {"composition": per[f]["meta"]["composition"], "kind": per[f]["meta"]["kind"],
                      "csv": str(OUT / f / "packed_LOCAL_NOT_FOR_UPLOAD.csv"), "csv_sha256": per[f]["pack"]["csv_sha256"],
                      "pose_columns_sha256": per[f]["val"]["pose_columns_sha256"],
                      "pack_exit": per[f]["pack"]["exit"], "packer_line": per[f]["pack"]["locally_computed_line"],
                      "input_reference_line": per[f]["pack"]["input_reference_line"],
                      "uploader_validate_submission": per[f]["val"]["uploader_validate_submission"],
                      "validate_problems": per[f]["val"]["problems"], "metric_code": per[f]["mc"]["overall"],
                      "packer_mean": per[f]["pack"]["packer_scores"]["mean"],
                      "max_abs_dev_vs_packer_per_episode": per[f]["mc"]["max_abs_dev_vs_packer_per_episode"]}
                  for f in FILES},
        "CONS-B": {k: cons[k] for k in ("T_k", "min", "mean", "max", "mean_of_episode_means", "mean_of_episode_min",
                                        "mean_of_episode_max")},
        "DEFAULT": {k: dflt[k] for k in ("T_k", "min", "mean", "max", "mean_of_episode_means", "mean_of_episode_min",
                                         "mean_of_episode_max")},
        "CONS-B_CLOSEST": closest["overall"],
        "changed_episodes": [e for e in range(30) if table[e]["changed"]],
        "quoted_t2_best5_LOCAL": {"files": quotes, "compositions": quoted},
        "crosscheck_vs_t2_best5_clean_evals": {"n_episode_runs": n_xc, "max_abs_dev": xc},
        "per_episode": table, "code": code_rows,
    }  # fmt: skip
    write_new(OUT / "summary_LOCAL_NOT_FOR_UPLOAD.json", dump(summary))
    cols = list(table[0].keys())
    write_csv_new(OUT / "summary_LOCAL_NOT_FOR_UPLOAD.csv", table, cols)
    write_csv_new(OUT / "code_commits_LOCAL_NOT_FOR_UPLOAD.csv", code_rows, list(code_rows[0].keys()))
    # markdown
    L = ["# t2_consensus_20261024 section 7: pack and LOCAL clean scoring (LOCAL_NOT_FOR_UPLOAD)", "",
         f"Generated {summary['generated_at']} CST by `scripts/t2cons_s7_pack.py` ({summary['script_sha256'][:16]}...), "
         f"pinned in `PREREG_PINS_S7.md` ({summary['pins_s7_sha256'][:16]}...). Clean (Tier-1) reference, official "
         "kit packer + uploader validation + official metric code. Report only: nothing here changes the selection "
         "(PREREG 7.7). CONS-A is neither packed nor clean-scored (AMENDMENTS A2.3).", "",
         "## Compositions (official metric code; T_k = run k of every episode)", "",
         "| composition / file | AUC | SP-SR | MP-SR | MPPE (cm) |", "|---|---|---|---|---|"]  # fmt: skip
    for label, st in (("CONS-B", cons), ("DEFAULT (t2_best5 SELECTED snapshot)", dflt)):
        L.append(f"| **{label}** mean of T_1..T_5 | **{fmt(st['mean']['auc'])}** | {fmt(st['mean']['sp_sr'])} | "
                 f"{fmt(st['mean']['mp_sr'])} | {fmt(st['mean']['mppe'], '.3f')} |")
        L.append(f"| {label} min / max over T_k | {fmt(st['min']['auc'])} / {fmt(st['max']['auc'])} | "
                 f"{fmt(st['min']['sp_sr'])} / {fmt(st['max']['sp_sr'])} | {fmt(st['min']['mp_sr'])} / "
                 f"{fmt(st['max']['mp_sr'])} | {fmt(st['min']['mppe'], '.3f')} / {fmt(st['max']['mppe'], '.3f')} |")
        L.append(f"| {label} mean of per-episode min / max | {fmt(st['mean_of_episode_min']['auc'])} / "
                 f"{fmt(st['mean_of_episode_max']['auc'])} | | | |")
    c = closest["overall"]
    L.append(f"| **CONS-B CLOSEST (the single-run file)** | **{fmt(c['auc'])}** | {fmt(c['sp_sr'])} | {fmt(c['mp_sr'])} | "
             f"{fmt(c['mppe'], '.3f')} |")
    for comp in ("SELECTED", "RAW-ONLY", "K3FULL"):
        q = quoted[comp]
        L.append(f"| quoted t2_best5 LOCAL {comp} (average of T_k) | {fmt(q['auc']['average'])} | "
                 f"{fmt(q['sp_sr']['average'])} | {fmt(q['mp_sr']['average'])} | {fmt(q['mppe']['average'], '.3f')} |")
    L += ["", f"T_k values: CONS-B AUC {', '.join(fmt(x) for x in cons['T_k']['auc'])}; DEFAULT AUC "
              f"{', '.join(fmt(x) for x in dflt['T_k']['auc'])}.",
          f"Quoted t2_best5 SELECTED jobs equal DEFAULT: {quoted['SELECTED_jobs_equal_DEFAULT']} "
          f"(differ on {quoted['SELECTED_jobs_differ_on'] or 'none'}).",
          f"Cross-check: {n_xc} episode-runs were already clean-scored per run by t2_best5 (packer metric); max |diff| "
          f"AUC {xc['auc']:.2e}, SP {xc['sp_sr']:.2e}, MP {xc['mp_sr']:.2e}, MPPE {xc['mppe']:.2e} cm.", "",
          "## Files", "", "| file | csv sha256 | uploader validate | problems | AUC | SP | MP | MPPE | packer AUC | max dev vs packer |",
          "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for f in FILES:
        s = summary["files"][f]
        L.append(f"| {f} | {s['csv_sha256'][:16]} | {s['uploader_validate_submission']} | {len(s['validate_problems'])} | "
                 f"{fmt(s['metric_code']['auc'])} | {fmt(s['metric_code']['sp_sr'])} | {fmt(s['metric_code']['mp_sr'])} | "
                 f"{fmt(s['metric_code']['mppe'], '.3f')} | {fmt(s['packer_mean'].get('auc'))} | "
                 f"{max(s['max_abs_dev_vs_packer_per_episode'].values()):.1e} |")
    L += ["", "## Per episode (AUC; 5-run clean mean [min, max]; * = changed vs DEFAULT)", "",
          "| ep | object | DEFAULT job (input) | DEFAULT AUC | CONS-B job (input) | CONS-B AUC | delta | CLOSEST run: AUC | "
          "CONS-B SP / MP mean | DEFAULT SP / MP mean |", "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in table:
        mark = "*" if r["changed"] else ""
        L.append(f"| {r['episode']:02d}{mark} | {r['object']} | {r['default']} ({r['default_input']}) | "
                 f"{fmt(r['default_auc_mean'])} [{fmt(r['default_auc_min'])}, {fmt(r['default_auc_max'])}] | "
                 f"{r['pick_B']} ({r['pick_input']}) | {fmt(r['consB_auc_mean'])} [{fmt(r['consB_auc_min'])}, "
                 f"{fmt(r['consB_auc_max'])}] | {r['delta_auc_mean']:+.4f} | {r['closest_run']}: {fmt(r['closest_auc'])} | "
                 f"{fmt(r['consB_sp_sr_mean'], '.1f')} / {fmt(r['consB_mp_sr_mean'], '.1f')} | "
                 f"{fmt(r['default_sp_sr_mean'], '.1f')} / {fmt(r['default_mp_sr_mean'], '.1f')} |")
    L += ["", "## Code needed for the cited commit (every CONS-B checkpoint)", "",
          "| ep | job | owner | input | training code | export | input code |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    for r in code_rows:
        L.append(f"| {r['episode']:02d} | {r['job']} | {r['owner']} | {r['input']} | {r['training_code']} | "
                 f"{r['export_tree']} | {r['input_code']} |")
    L += ["", "Evaluation: kit evaluate_policy.py 160d24f5 on the stock runtime code/video_to_data_8c778f9 @ 8c778f9 "
              "(eval_code b1b3ebe3). Selection and packing: scripts/t2best5_driver.py, t2cons_eval.py, t2cons_a1_eval.py, "
              "t2cons_consensus.py, t2cons_a1_gate.py, t2cons_a1_resolve.py, t2cons_a1_consensus.py, t2cons_s7_pack.py; "
              "this folder's PREREG.md, AMENDMENTS.md, PREREG_PINS*.md and selection.json (PREREG 10.3 / A2.3).",
          "", "Upload: only the CONS-B CLOSEST file is eligible (A2.3 / PREREG 7.8), after a re-pack with the real commit "
              "URL whose pose columns must equal pose_columns_sha256 above. The user decides. Nothing was uploaded."]  # fmt: skip
    write_new(OUT / "summary_LOCAL_NOT_FOR_UPLOAD.md", "\n".join(L) + "\n")
    print("\n".join(L))
    return 0


# ================================================================================================ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("check", "prepare", "pack", "validate", "score", "report"))
    ap.add_argument("--files", nargs="+", help=f"subset of {FILES}")
    ap.add_argument("--jobs", type=int, default=3, help="parallel packer processes (pack)")
    a = ap.parse_args()
    if not sys.flags.dont_write_bytecode or not sys.flags.isolated:
        print("run with the kit python -I -B")
        return 2
    return {"check": cmd_check, "prepare": cmd_prepare, "pack": cmd_pack, "validate": cmd_validate,
            "score": cmd_score, "report": cmd_report}[a.cmd](a)  # fmt: skip


if __name__ == "__main__":
    raise SystemExit(main())
