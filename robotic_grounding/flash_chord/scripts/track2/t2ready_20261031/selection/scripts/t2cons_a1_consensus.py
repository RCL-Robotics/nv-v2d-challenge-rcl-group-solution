#!/usr/bin/env python3
"""t2cons_a1_consensus.py: the remaining amended steps of fc_runs/submissions/t2_consensus_20261024 under AMENDMENTS.md
A1-A3 (variant B, WITH G2, binding), after the A3.3 evaluations (t2cons_a1_eval.py) finished.

  check   dry run: pin chain, job / evaluation loading, candidate counts, per-episode pools (names only), rollout files;
          writes dryrun/a1_consensus_check_*.txt and (once) inputs_a1/ROLLOUTS_A1c.sha256. No metric is computed.
  gate    A1.4: the amended own-input gate (A1.3) on the 50 runs of the 10 A3 jobs, with the pinned code of
          t2cons_a1_gate.py (seed 0 = t2cons_consensus.metrics_vs; seeds 1..9 = t2cons_a1_gate.auc_seed), then over
          all 450 runs together with the pinned 400-run result. If it fails only through runs of A3 jobs, those jobs
          are validation_failed (not candidates) and the gate is re-evaluated without them; any other failure stops.
          -> scores/own_input_validation_a1_new.{csv,json}
  refs    S2: t2cons_consensus.cmd_refs, unchanged (PREREG 2) -> refs/{A,B}/, refs/build.json (+ refs/build_a1.json)
  score   S4: t2cons_consensus.cmd_score, unchanged, over the amended candidate set (PREREG 4 + A3; minus
          validation_failed) -> scores/consensus_scores.csv, scores/consensus_job_stats.json (+ consensus_a1_run.json)
  select  S5: PREREG 5 rule via t2cons_consensus.select_episode, unchanged; variant B binding (A2.2), variant A
          reported; CONS-B CLOSEST run per episode (A2.3) -> selection.json, selection_consensus.json, selection.txt,
          selection_table.{csv,md}, selection_candidates.csv, summary_consensus.{json,md}; SELECTION.sha256 LAST.
  selftest-select  the select code end to end on SYNTHETIC random consensus values (no score is read or computed),
          written only under selftest/a1_select_synthetic/ (never the real outputs).

How the pinned code is reused. t2cons_consensus.py (e1b943b1...) and t2cons_a1_gate.py (a29c997b...) are compiled from
their verified bytes. For refs and score, the pinned cmd_refs / cmd_score run unchanged; only four module-level names
they look up are rebound: require_pins (the original-gate check, which refuses by design, A1.4) -> the amended pin chain
below; load_jobs / load_evals / candidates -> the pinned functions plus the A3 jobs of manifest_a1.json and their runs
in own_input_evals_a1.csv (same checks), minus validation_failed jobs. select is new code (variant B binding) around
the pinned select_episode, floor_count, stats and comp_summary.

Noisy-only: every path goes through the pinned PREREG 5.7 guard (clean, LOCAL_NOT_FOR_UPLOAD, packed_, tier-1 roots
...). No clean file, no packer, no clean reference.
Run: conda_envs/v2dkit/bin/python -I -B scripts/t2cons_a1_consensus.py <cmd>
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
ROOT = ME / "fc_runs/submissions/t2_consensus_20261024"
CONS_FILE, CONS_SHA = ME / "scripts/t2cons_consensus.py", "e1b943b10fd1db13ab6f78e5bde7b3c6e6c48f7daeee6230c32c463d14ac41b4"
GATE_FILE, GATE_SHA = ME / "scripts/t2cons_a1_gate.py", "a29c997b7271e9299db109e97af5210f528484719210bb58685523b0ae0eb16b"
A1EVAL_SHA = "dedd2e0bed855cb39e520b0fcc5fa635ccf7121fdf12d55a11afd10694f5eb56"
AMEND, AMEND_SHA_FILE = ROOT / "AMENDMENTS.md", ROOT / "AMENDMENTS.sha256"
AMEND_SHA = "f648226a95a061eeefcfd5138cb01823db8a060fe121a0f4968ffa163dc4229f"
PINS, PINS_SHA_FILE = ROOT / "PREREG_PINS.md", ROOT / "PREREG_PINS.sha256"
PINS_SHA = "9d1493797205e8c60255c50658c536714a3702fe8751c724f37ed1444a040ece"
PINS_B, PINS_B_SHA_FILE = ROOT / "PREREG_PINS_A1b.md", ROOT / "PREREG_PINS_A1b.sha256"
PINS_B_SHA = "7239bc561eacc1d3ed6b3f210ae80193c30ad3168d8e0255dd7ee63016bd7c7a"
PINS_C, PINS_C_SHA_FILE = ROOT / "PREREG_PINS_A1c.md", ROOT / "PREREG_PINS_A1c.sha256"
CANDS, CANDS_SHA = ROOT / "inputs_a1/candidates_a1.json", "be42b6ddd32ff708365b894852cdbbfbdda583504f44038dc4790827a97301e4"
SNAP_A1 = ROOT / "inputs_a1/SNAPSHOT_A1.sha256"
ROLLOUTS = ROOT / "inputs_a1/ROLLOUTS_A1c.sha256"
MAN_A1, EVALS_A1 = ROOT / "manifest_a1.json", ROOT / "own_input_evals_a1.csv"
VAL_A1_JSON, VAL_A1_CSV = ROOT / "scores/own_input_validation_a1.json", ROOT / "scores/own_input_validation_a1.csv"
VAL_A1_JSON_SHA = "c6af3fd5ee7362b434b557bc844a0eb3da34c55ddcb31ee9c098ae2ea8b7e6a7"
VAL_A1_CSV_SHA = "10e29bf0c36ca30ae91a283d7ff89e8560737b8e2293bc313168103969dcf957"
GATE_NEW_CSV, GATE_NEW_JSON = ROOT / "scores/own_input_validation_a1_new.csv", ROOT / "scores/own_input_validation_a1_new.json"
SEL_SHA = ROOT / "SELECTION.sha256"
SYNTH_DIR = ROOT / "selftest/a1_select_synthetic"
BINDING = "B"  # AMENDMENTS A2.2
NEW_A1 = ("ep05_zx_k3fix_88000", "ep26_zx_k3fix_92000", "ep26_k3ca_final25298", "ep04_k3fix_final44922",
          "ep01_zx_g2_124000", "ep04_zx_g2_88000", "ep19_zx_g2_92000", "ep28_zx_g2_100000",
          "ep01_zx_g2hard_102000", "ep19_zx_g2hard_52000")  # fmt: skip
REUSED = {"ep04_zx_k3fix_12000": "3efcbf86a775c8a3e9b5be5265242a7f90e309e998d6224f300e80ab3995105a",
          "ep01_t2fixL_p62000": "2d69e2da0647d4c3f5b23880eede56ac3bb370deffed7c9c2799a08d4ca6416f"}  # fmt: skip
SETTLED_A3 = ("ep01_t2fixL_p62000",)  # A3.3: provisional job whose checkpoint is now its run's settled designation
COUNTS = {"t2_best5": (75, 71), "new": (5, 2), "a1": (10, 4)}  # (all, variant A) per source: PREREG 4.1 / 4.2, A3


# ================================================================================================ pinned code
def _load(name: str, path: Path, sha: str):
    src = path.read_bytes()
    got = hashlib.sha256(src).hexdigest()
    if got != sha:
        raise SystemExit(f"{path} sha256 {got} != pinned {sha}")
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    sys.modules[name] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)  # noqa: S102 (verified pinned bytes)
    return mod


G = _load("t2cons_a1_gate_pinned", GATE_FILE, GATE_SHA)  # its module code loads t2cons_consensus.py as G.C
C = G.C
if hashlib.sha256(Path(C.__file__).read_bytes()).hexdigest() != CONS_SHA:
    raise SystemExit("t2cons_consensus.py (as loaded by t2cons_a1_gate.py) != pinned e1b943b1...")
ORIG = {"load_jobs": C.load_jobs, "load_evals": C.load_evals, "candidates": C.candidates, "require_pins": C.require_pins}
MINE = Path(__file__).resolve()


def sha(p) -> str:
    return C.sha256_file(p)


def sha_value(p: Path) -> str:
    return (C.guard(p).read_text().split() or [""])[0] if p.is_file() else ""


def pin_lines(text: str) -> list[tuple[str, str]]:
    out = []
    for line in text.splitlines():
        f = line.split()
        if len(f) == 3 and f[0] == "pin" and len(f[2]) == 64:
            out.append((f[1], f[2]))
    return out


def pin_path(rel: str) -> Path:
    return ME / rel if rel.startswith("scripts/") else ROOT / rel


# ================================================================================================ pin chain
def chain(stage: str) -> tuple[list[str], list[str]]:
    """stage: check (dry; PREREG_PINS_A1c may be absent) | gate | refs | score | select."""
    fatal, info = C.check_pins(need_pins=False)  # PREREG == sha, PREREG 11 pins + commits, SNAPSHOT, KIT
    if fatal:
        return fatal, info
    mine = sha(MINE)
    info.append(f"scripts/t2cons_a1_consensus.py sha256 {mine}")
    for label, p, shaf, want in (("AMENDMENTS.md", AMEND, AMEND_SHA_FILE, AMEND_SHA),
                                 ("PREREG_PINS.md", PINS, PINS_SHA_FILE, PINS_SHA),
                                 ("PREREG_PINS_A1b.md", PINS_B, PINS_B_SHA_FILE, PINS_B_SHA)):  # fmt: skip
        got = sha(p) if p.is_file() else "missing"
        rec = sha_value(shaf)
        if not got == rec == want:
            fatal.append(f"{label} sha256 {got[:16]} (sha file {rec[:16]}, pinned here {want[:16]})")
            continue
        n = 0
        for rel, h0 in pin_lines(C.guard(p).read_text()):
            pp = pin_path(rel)
            h = sha(pp) if pp.is_file() else "missing"
            n += 1
            if h != h0:
                fatal.append(f"{label} pin {rel}: {h[:16]} != {h0[:16]}")
        info.append(f"{label} sha256 {got[:16]} == sha file == pinned; {n} pins checked")
    n = 0
    for line in C.guard(SNAP_A1).read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        p = ROOT / parts[1].strip()
        h = sha(p) if p.is_file() else "missing"
        n += 1
        if h != parts[0]:
            fatal.append(f"{parts[1].strip()}: {h[:16]} != SNAPSHOT_A1.sha256 {parts[0][:16]}")
    info.append(f"inputs_a1/SNAPSHOT_A1.sha256: {n} files checked")
    if sha(CANDS) != CANDS_SHA:
        fatal.append("inputs_a1/candidates_a1.json != pinned be42b6dd...")
    st = C.read_json(ROOT / "selftest/selftest.json")
    if st.get("pass") is not True or st.get("script_sha256") != CONS_SHA:
        fatal.append("selftest/selftest.json is not a PASS of the pinned t2cons_consensus.py")
    v = C.read_json(VAL_A1_JSON)
    if sha(VAL_A1_JSON) != VAL_A1_JSON_SHA or sha(VAL_A1_CSV) != VAL_A1_CSV_SHA or v.get("pass") is not True or \
            v.get("script_sha256") != GATE_SHA or v.get("amendments_sha256") != AMEND_SHA:  # fmt: skip
        fatal.append("scores/own_input_validation_a1.* is not the pinned A1 PASS of t2cons_a1_gate.py")
    else:
        info.append("A1 gate (400 runs): pinned PASS")
    hb = C.read_json(ROOT / "eval_a1_heartbeat.json")
    if hb.get("phase") != "done" or (hb.get("evals") or {}).get("ok") != 50 or (ROOT / "driver.pid").exists() or \
            (hb.get("exports") or {}).get("ok") != 10:  # fmt: skip
        fatal.append(f"A3.3 evaluations not finished (heartbeat phase {hb.get('phase')}, evals {hb.get('evals')})")
    else:
        info.append("A3.3 evaluations finished: 10 exports ok, 50 evaluations ok, driver ended")
    if PINS_C.is_file():
        got, rec = sha(PINS_C), sha_value(PINS_C_SHA_FILE)
        if got != rec:
            fatal.append(f"PREREG_PINS_A1c.md sha256 {got[:16]} != PREREG_PINS_A1c.sha256 {rec[:16]}")
        else:
            pins = pin_lines(C.guard(PINS_C).read_text())
            if ("scripts/t2cons_a1_consensus.py", mine) not in pins:
                fatal.append(f"PREREG_PINS_A1c.md does not pin scripts/t2cons_a1_consensus.py {mine[:16]}")
            for rel, h0 in pins:
                pp = pin_path(rel)
                h = sha(pp) if pp.is_file() else "missing"
                if h != h0:
                    fatal.append(f"PREREG_PINS_A1c pin {rel}: {h[:16]} != {h0[:16]}")
            info.append(f"PREREG_PINS_A1c.md sha256 {got[:16]} == its sha file; {len(pins)} pins checked; pins this script")
    elif stage != "check":
        fatal.append("PREREG_PINS_A1c.md missing: this script must be pinned before it gates, builds, scores or selects")
    else:
        info.append("PREREG_PINS_A1c.md not written yet (dry check)")
    if stage in ("refs", "score", "select", "synthetic"):
        if not GATE_NEW_JSON.is_file():
            fatal.append("scores/own_input_validation_a1_new.json missing: run the A1.4 gate on the A3 runs first")
        else:
            g = C.read_json(GATE_NEW_JSON)
            if g.get("pass") is not True or g.get("script_sha256") != mine or g.get("amendments_sha256") != AMEND_SHA \
                    or g.get("pins_a1c_sha256") != (sha(PINS_C) if PINS_C.is_file() else None) \
                    or g.get("csv_sha256") != sha(GATE_NEW_CSV):  # fmt: skip
                fatal.append("scores/own_input_validation_a1_new.json is not a PASS of this pinned script")
            else:
                info.append(f"A1.4 gate on the A3 runs: PASS; validation_failed {g.get('validation_failed_jobs')}")
    if stage in ("refs", "score", "select") and SEL_SHA.exists():
        fatal.append("SELECTION.sha256 exists: the selection is frozen")
    return fatal, info


def require(stage: str) -> bool:
    fatal, info = chain(stage)
    for x in info:
        print("  " + x)
    if fatal:
        print("FATAL:\n  " + "\n  ".join(fatal))
        return False
    return True


def excluded_jobs() -> list[str]:
    if not GATE_NEW_JSON.is_file():
        return []
    return list(C.read_json(GATE_NEW_JSON).get("validation_failed_jobs") or [])


# ================================================================================================ jobs / evaluations
def t2best5_training_sha() -> dict[str, str]:
    rows = list(C.read_json(C.SNAP / "t2_best5/manifest.json")["rows"])
    rows += [json.loads(ln) for ln in C.guard(C.SNAP / "t2_best5/manifest_stageB.jsonl").read_text().splitlines() if ln.strip()]
    return {r["job"]: r.get("training_sha256") for r in rows}


def load_jobs_a1() -> dict[str, dict]:
    """The pinned load_jobs (t2_best5 snapshot + manifest_new.json) plus the 10 A3 jobs of manifest_a1.json."""
    jobs = ORIG["load_jobs"]()
    man = C.read_json(MAN_A1)
    if man.get("candidates_a1_sha256") != CANDS_SHA or man.get("amendments_sha256") != AMEND_SHA or \
            man.get("t2cons_a1_eval_sha256") != A1EVAL_SHA or man.get("prereg_pins_a1b_sha256") != PINS_B_SHA:  # fmt: skip
        raise SystemExit("manifest_a1.json was not built by the pinned A3.3 step")
    if sorted(r["job"] for r in man["rows"]) != sorted(NEW_A1):
        raise SystemExit(f"manifest_a1.json jobs {sorted(r['job'] for r in man['rows'])} != the 10 A3 jobs")
    if man.get("reused_not_evaluated") != REUSED:
        raise SystemExit("manifest_a1.json reused jobs differ")
    for r in man["rows"]:
        if r["job"] in jobs:
            raise SystemExit(f"duplicate job {r['job']}")
        kind = r.get("root_kind")
        if kind not in C.INPUT_REF or kind == "raw":
            raise SystemExit(f"{r['job']}: unexpected input kind {kind}")
        if Path(str(r.get("frame_refs"))) != C.INPUT_REF[kind]:
            raise SystemExit(f"{r['job']}: manifest frame_refs {r.get('frame_refs')} != PREREG 3.2 input reference for {kind}")
        variants = ("B",) if kind == "g2" else ("A", "B")
        if ",".join(variants) != r.get("variants"):
            raise SystemExit(f"{r['job']}: variants {r.get('variants')} != {variants}")
        exp = C.read_json(ROOT / "exports" / r["job"] / "export.json")
        jobs[r["job"]] = {
            "job": r["job"], "episode": int(r["episode"]), "source": "a1", "status": r.get("status"), "input": kind,
            "input_reference": str(C.INPUT_REF[kind]), "family": r.get("family"), "owner": r.get("owner"),
            "iteration": r.get("iteration"), "provisional": bool(r.get("provisional")), "object": r.get("object"),
            "parquet_sha256": r.get("parquet_sha256"), "variants": variants,
            "export_status": exp.get("status"), "export_sha256": exp.get("sha256"),
        }  # fmt: skip
    tsha = t2best5_training_sha()
    for j, s in REUSED.items():
        if j not in jobs or jobs[j]["source"] != "t2_best5" or tsha.get(j) != s:
            raise SystemExit(f"reused job {j} is not the t2_best5 job of checkpoint {s[:16]}")
    for j in SETTLED_A3:
        jobs[j]["provisional"] = False
        jobs[j]["settled_by"] = "AMENDMENTS A3.3 (now its run's runin_final.json designation)"
    return jobs


def load_evals_a1(jobs: dict[str, dict]) -> dict[str, dict[int, dict]]:
    """The pinned load_evals plus the ok rows of own_input_evals_a1.csv, with the same checks."""
    out = ORIG["load_evals"](jobs)
    for r in C.read_csv(EVALS_A1):
        if r.get("status") != "ok":
            continue
        j = jobs.get(r["job"])
        if j is None or j["source"] != "a1":
            raise SystemExit(f"own_input_evals_a1.csv: ok row of unknown job {r['job']}")
        k, t, e = int(r["run"]), int(r["try"]), j["episode"]
        if int(r["episode"]) != e:
            raise SystemExit(f"{r['job']} run {k}: episode {r['episode']} != {e}")
        pq = ROOT / "runs" / r["job"] / f"run{k}" / f"try{t}" / "results" / f"episode_{e:06d}.parquet"
        if Path(r["results_parquet"]) != pq:
            raise SystemExit(f"{r['job']} run {k}: results_parquet {r['results_parquet']} != {pq}")
        if r.get("export_sha256_full") != j["export_sha256"] or not j["export_sha256"]:
            raise SystemExit(f"{r['job']} run {k}: export_sha256 {r.get('export_sha256_full')} != export.json")
        if k in out.setdefault(r["job"], {}):
            raise SystemExit(f"{r['job']} run {k}: two ok rows")
        out[r["job"]][k] = {"run": k, "try": t, "parquet": str(pq), **{m: C.fnum(r[m]) for m in C.METRICS}}
    return out


def candidates_a1(jobs: dict, evals: dict, excluded: list[str]) -> list[str]:
    """PREREG 4.1 / 4.2 + A3.3: every job with 5 ok official evaluations, minus A1.4 validation_failed jobs."""
    for x in excluded:
        if jobs.get(x, {}).get("source") != "a1":
            raise SystemExit(f"validation_failed job {x} is not an A3 job (A1.4 excludes only A3 jobs)")
    full = [j for j in sorted(jobs) if jobs[j]["status"] == "ok" and jobs[j]["export_status"] == "ok"
            and sorted(evals.get(j, {})) == list(range(1, C.NRUNS + 1))]  # fmt: skip
    for src, (n_all, n_a) in COUNTS.items():
        xs = [j for j in full if jobs[j]["source"] == src]
        got = (len(xs), sum("A" in jobs[j]["variants"] for j in xs))
        if got != (n_all, n_a):
            raise SystemExit(f"{src}: candidate counts {got} != {(n_all, n_a)} (PREREG 4.1 / 4.2, A3)")
    return [j for j in full if j not in excluded]


def pools(jobs: dict, cand: list[str], e: int, v: str) -> list[str]:
    return [j for j in cand if jobs[j]["episode"] == e and v in jobs[j]["variants"]]


def count_check(jobs: dict, cand: list[str], excluded: list[str]) -> list[str]:
    """Challengers per episode and variant == candidates_a1.json 'after' minus validation_failed; defaults == snapshot."""
    cands = C.read_json(CANDS)
    snap = {int(x["episode"]): x for x in C.read_json(C.SNAP / "t2_best5/selection.json")["episodes"]}
    problems = []
    for e in range(30):
        D = snap[e]["selected"]
        if cands["defaults"][f"{e:02d}"] != D:
            problems.append(f"ep{e:02d}: candidates_a1.json default {cands['defaults'][f'{e:02d}']} != snapshot SELECTED {D}")
        if D not in cand or jobs[D]["episode"] != e or tuple(jobs[D]["variants"]) != ("A", "B"):
            problems.append(f"ep{e:02d}: default {D} is not a variant-A,B candidate of this episode")
        if e == 4 and jobs.get(D, {}).get("input") != "raw":
            problems.append("ep04: default is not the raw final")
        for v in ("A", "B"):
            n_ex = sum(1 for x in excluded if jobs[x]["episode"] == e and v in jobs[x]["variants"])
            want = cands["challenger_counts"][v][f"{e:02d}"]["after"] - n_ex
            got = len([j for j in pools(jobs, cand, e, v) if j != D])
            if got != want:
                problems.append(f"ep{e:02d} variant {v}: {got} challengers != candidates_a1.json {want}")
    return problems


def rollout_files(evals: dict, cand: list[str]) -> dict[str, str]:
    return {evals[j][k]["parquet"]: j for j in cand for k in range(1, C.NRUNS + 1)}


def check_rollouts(evals: dict, cand: list[str]) -> list[str]:
    want = C.hash_list(ROLLOUTS)
    problems = []
    for p in rollout_files(evals, cand):
        if want.get(p) is None:
            problems.append(f"{p} not in ROLLOUTS_A1c.sha256")
        elif sha(p) != want[p]:
            problems.append(f"{p} changed since ROLLOUTS_A1c.sha256")
    return problems


# ================================================================================================ check (dry)
def cmd_check(a) -> int:
    lines: list[str] = []

    def p(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    p(f"t2cons_a1_consensus CHECK {C.now()}: no metric, reference or score is computed")
    fatal, info = chain("check")
    for x in info:
        p(f"chain: {x}")
    for x in fatal:
        p(f"chain FAILED: {x}")
    jobs = load_jobs_a1()
    evals = load_evals_a1(jobs)
    cand = candidates_a1(jobs, evals, [])
    p(f"jobs {len(jobs)}; candidates {len(cand)} (variant A {sum('A' in jobs[j]['variants'] for j in cand)}); runs {5 * len(cand)}")
    problems = count_check(jobs, cand, [])
    snap = {int(x["episode"]): x for x in C.read_json(C.SNAP / "t2_best5/selection.json")["episodes"]}
    for e in range(30):
        D = snap[e]["selected"]
        for v in ("A", "B"):
            ch = [f"{j}[{jobs[j]['input']}]" for j in pools(jobs, cand, e, v) if j != D]
            p(f"ep{e:02d} [{v}] D {D}[{jobs[D]['input']}] challengers {len(ch)}: {' '.join(ch)}")
    files = rollout_files(evals, cand)
    missing = [f for f in files if not Path(f).is_file()]
    problems += [f"missing rollout {f}" for f in missing]
    if not ROLLOUTS.exists() and not missing and not fatal and not problems:
        body = "".join(f"{sha(f)}  {f}\n" for f in sorted(files))
        C.write_text(ROLLOUTS, body)
        os.chmod(ROLLOUTS, 0o444)
        p(f"wrote {ROLLOUTS.relative_to(ROOT)}: {len(files)} rollout parquets, sha256 {sha(ROLLOUTS)}")
    elif ROLLOUTS.exists():
        rp = check_rollouts(evals, cand)
        problems += rp
        p(f"{ROLLOUTS.relative_to(ROOT)} exists (sha256 {sha(ROLLOUTS)}): {len(files) - len(rp)}/{len(files)} rollouts unchanged")
    for x in problems:
        p(f"problem: {x}")
    ok = not fatal and not problems
    p(f"check: {'OK' if ok else 'PROBLEMS'}")
    C.write_text(ROOT / "dryrun" / f"a1_consensus_check_{C.stamp()}.txt", "\n".join(lines) + "\n")
    return 0 if ok else 1


# ================================================================================================ gate (A1.4)
def _b(x) -> bool:
    return x is True or str(x) == "True"


def run_flags(r: dict) -> list[str]:
    if r.get("status") != "ok":
        return ["problem run"]
    out = []
    if not _b(r["band_ok"]):
        out.append("AUC outside the seed range +- 0.02")
    if abs(float(r["d_mppe"])) > G.GATE["max_abs_d_mppe"]:
        out.append("|dMPPE| > 0.5 cm")
    if float(r["mine_sp_sr"]) != float(r["eval_sp_sr"]):
        out.append("SP-SR differs")
    if float(r["mine_mp_sr"]) != float(r["eval_mp_sr"]):
        out.append("MP-SR differs")
    return out


def gate_parts(rows: list[dict], n_expected: int, diff: list[str]) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    d_mppe = max((abs(float(r["d_mppe"])) for r in ok), default=None)
    sp = sum(float(r["mine_sp_sr"]) == float(r["eval_sp_sr"]) for r in ok) / len(ok) if ok else 0.0
    mp = sum(float(r["mine_mp_sr"]) == float(r["eval_mp_sr"]) for r in ok) / len(ok) if ok else 0.0
    return {
        "auc_band": bool(ok) and all(_b(r["band_ok"]) for r in ok),
        "mppe": d_mppe is not None and d_mppe <= G.GATE["max_abs_d_mppe"],
        "sp_equal": sp >= G.GATE["min_sp_equal_frac"],
        "mp_equal": mp >= G.GATE["min_mp_equal_frac"],
        "no_problem_runs": len(ok) == len(rows) == n_expected,
        "kit_stock_unchanged": not diff,
        "_n_runs": len(rows), "_max_abs_d_mppe": d_mppe, "_sp_equal_frac": sp, "_mp_equal_frac": mp,
    }  # fmt: skip


def passed(parts: dict) -> bool:
    return all(v for k, v in parts.items() if not k.startswith("_"))


def cmd_gate(a) -> int:
    if not require("gate"):
        return 2
    if G.GATE_LINE not in C.guard(AMEND).read_text():
        print("FATAL: AMENDMENTS.md does not state the A1 gate")
        return 2
    if GATE_NEW_CSV.exists() or GATE_NEW_JSON.exists():
        print("FATAL: output exists (new files only)")
        return 2
    old = C.read_csv(VAL_A1_CSV)
    vj = C.read_json(VAL_A1_JSON)
    p_old = gate_parts(old, 400, [])
    if not passed(p_old) or len(old) != 400 or sum(_b(r["inside_range"]) for r in old) != vj["n_inside_seed_range"]:
        print("FATAL: the pinned 400-run A1 result does not re-evaluate to its recorded PASS")
        return 2
    before = C.integrity()
    K = C.load_kit()
    jobs = load_jobs_a1()
    evals = load_evals_a1(jobs)
    candidates_a1(jobs, evals, [])  # count checks
    new = [j for j in sorted(jobs) if jobs[j]["source"] == "a1"]
    cache: dict = {}
    grids: dict = {}
    rows, problems = [], []
    for job in new:
        j = jobs[job]
        e = j["episode"]
        if e not in grids:
            grids[e] = C.grid(K, e)
        frames, names = grids[e]
        for k in range(1, C.NRUNS + 1):
            rec = evals[job][k]
            row = {"job": job, "episode": e, "run": k, "try": rec["try"], "input": j["input"], "source": j["source"],
                   "variants": ",".join(j["variants"])}  # fmt: skip
            try:
                conv, inref, inf = C.rollout_on_grid(K, j, rec, frames, names)
                m0 = C.metrics_vs(K, conv, inref, names)
                aucs = [m0["auc"]] + [G.auc_seed(K, cache, conv, inref, names, s) for s in G.SEEDS[1:]]
            except ValueError as err:
                row.update({"status": "failed", "problem": str(err)})
                problems.append(f"{job} run {k}: {err}")
                rows.append(row)
                continue
            ev = {m: rec[m] for m in C.METRICS}
            lo, hi = min(aucs), max(aucs)
            row.update({
                "status": "ok", "eval_auc": ev["auc"], **{f"auc_s{s}": x for s, x in zip(G.SEEDS, aucs)},
                "seed_min": lo, "seed_max": hi, "seed_range": hi - lo,
                "excess_outside_range": max(lo - ev["auc"], ev["auc"] - hi, 0.0),
                "band_ok": bool(lo - G.BAND <= ev["auc"] <= hi + G.BAND), "inside_range": bool(lo <= ev["auc"] <= hi),
                "d_auc_seed0": m0["auc"] - ev["auc"],
                "eval_mppe": ev["mppe"], "mine_mppe": m0["mppe"], "d_mppe": m0["mppe"] - ev["mppe"],
                "eval_sp_sr": ev["sp_sr"], "mine_sp_sr": m0["sp_sr"], "eval_mp_sr": ev["mp_sr"], "mine_mp_sr": m0["mp_sr"],
                "conv_pos_residual_m": inf["conv_pos_residual_m"], "conv_ang_residual_rad": inf["conv_ang_residual_rad"],
            })  # fmt: skip
            row["flags"] = "; ".join(run_flags(row))
            rows.append(row)
        print(f"  {job}: {len([r for r in rows if r['job'] == job and r['status'] == 'ok'])}/5 scored against its own input",
              flush=True)  # fmt: skip
    after = C.integrity()
    diff = C.integrity_diff(before, after)
    p_new = gate_parts(rows, 5 * len(new), diff)
    p_all = gate_parts(old + rows, 400 + 5 * len(new), diff)
    flagged = sorted({r["job"] for r in rows if run_flags(r)})
    excluded, p_rest = [], None
    if passed(p_all):
        final = True
    else:
        rest = [r for r in rows if r["job"] not in flagged]
        n_rest = 5 * (len(new) - len(flagged))
        p_rest = gate_parts(old + rest, 400 + n_rest, diff)
        final = bool(flagged) and passed(p_rest) and not diff
        excluded = flagged if final else []
    ok = [r for r in rows if r["status"] == "ok"]
    worst = sorted(ok, key=lambda r: -r["excess_outside_range"])[:10]
    by_job = {}
    for job in new:
        xs = [r for r in ok if r["job"] == job]
        by_job[job] = {"input": jobs[job]["input"], "n_scored": len(xs), "n_inside_seed_range": sum(r["inside_range"] for r in xs),
                       "max_excess_outside_range": max((r["excess_outside_range"] for r in xs), default=None),
                       "max_seed_range": max((r["seed_range"] for r in xs), default=None),
                       "max_abs_d_mppe": max((abs(r["d_mppe"]) for r in xs), default=None),
                       "flagged_runs": [r["run"] for r in rows if r["job"] == job and run_flags(r)]}  # fmt: skip
    out = {
        "written_at": C.now(), "script_sha256": sha(MINE), "consensus_script_sha256": CONS_SHA, "gate_script_sha256": GATE_SHA,
        "amendments_sha256": AMEND_SHA, "pins_a1c_sha256": sha(PINS_C), "prereg": "AMENDMENTS A1.4 (A1.3 gate on the A3 runs)",
        "gate": G.GATE_LINE, "thresholds": G.GATE, "pass": final,
        "parts_new_runs": p_new, "parts_all_450": p_all, "parts_after_exclusion": p_rest,
        "validation_failed_jobs": excluded, "flagged_jobs": flagged,
        "n_new_jobs": len(new), "n_new_runs": len(rows), "n_new_scored": len(ok), "problems": problems,
        "n_band_fail_new": sum(not r["band_ok"] for r in ok), "n_inside_seed_range_new": sum(r["inside_range"] for r in ok),
        "max_excess_outside_range_new": None if not ok else max(r["excess_outside_range"] for r in ok),
        "max_seed_range_new": None if not ok else max(r["seed_range"] for r in ok),
        "max_abs_d_auc_seed0_new": None if not ok else max(abs(r["d_auc_seed0"]) for r in ok),
        "max_conv_pos_residual_m": max((r["conv_pos_residual_m"] for r in ok), default=None),
        "max_conv_ang_residual_rad": max((r["conv_ang_residual_rad"] for r in ok), default=None),
        "old_400_runs": {"file": str(VAL_A1_CSV), "sha256": VAL_A1_CSV_SHA, "parts_reevaluated": p_old},
        "by_job": by_job,
        "worst_excess_runs": [{k: r[k] for k in ("job", "run", "eval_auc", "seed_min", "seed_max", "excess_outside_range",
                                                "d_auc_seed0")} for r in worst],
        "vertex_meshes": {n: {"file": str(v[1]), "sha256": sha(v[1])} for n, v in sorted(K.verts.items())},
        "integrity_diff": diff[:50],
        "note": "own-input check only (each run vs its OWN training input); seeds 1..9 used only for this gate; no "
                "consensus reference exists or is read; nothing is selected",
    }  # fmt: skip
    cols = ["job", "episode", "run", "try", "input", "source", "variants", "status", "problem", "flags", "eval_auc"] + \
        [f"auc_s{s}" for s in G.SEEDS] + \
        ["seed_min", "seed_max", "seed_range", "excess_outside_range", "band_ok", "inside_range", "d_auc_seed0",
         "eval_mppe", "mine_mppe", "d_mppe", "eval_sp_sr", "mine_sp_sr", "eval_mp_sr", "mine_mp_sr",
         "conv_pos_residual_m", "conv_ang_residual_rad"]  # fmt: skip
    C.write_text(GATE_NEW_CSV, C.csv_text(rows, cols))
    out["csv_sha256"] = sha(GATE_NEW_CSV)
    C.write_text(GATE_NEW_JSON, C.dump(out))
    for f in (GATE_NEW_CSV, GATE_NEW_JSON):
        os.chmod(f, 0o444)
    print(json.dumps({k: out[k] for k in ("pass", "parts_new_runs", "parts_all_450", "parts_after_exclusion",
                                          "validation_failed_jobs", "flagged_jobs", "n_new_runs", "n_band_fail_new",
                                          "n_inside_seed_range_new", "max_excess_outside_range_new", "max_seed_range_new",
                                          "problems")}, indent=1))  # fmt: skip
    for job, x in by_job.items():
        print(f"  {job}: inside {x['n_inside_seed_range']}/5, max excess {x['max_excess_outside_range']}, "
              f"max |dMPPE| {x['max_abs_d_mppe']}, flagged runs {x['flagged_runs']}")
    return 0 if final else 1


# ================================================================================================ refs / score (pinned code)
def rebind(stage: str) -> list[str]:
    excluded = excluded_jobs()
    C.require_pins = lambda: require(stage)
    C.load_jobs = load_jobs_a1
    C.load_evals = load_evals_a1
    C.candidates = lambda jobs, evals: candidates_a1(jobs, evals, excluded)
    return excluded


def cmd_refs(a) -> int:
    if not require("refs"):
        return 2
    rebind("refs")
    rc = C.cmd_refs(a)  # pinned PREREG 2 build, unchanged
    if (ROOT / "refs/build.json").is_file():
        b = C.read_json(ROOT / "refs/build.json")
        C.write_text(ROOT / "refs/build_a1.json", C.dump({
            "written_at": C.now(), "ran": "t2cons_consensus.cmd_refs (pinned, unchanged) via scripts/t2cons_a1_consensus.py refs",
            "a1_script_sha256": sha(MINE), "consensus_script_sha256": CONS_SHA, "pins_a1c_sha256": sha(PINS_C),
            "amendments_sha256": AMEND_SHA, "rebound": ["require_pins -> amended pin chain (AMENDMENTS A1.4)"],
            "build_json_sha256": sha(ROOT / "refs/build.json"), "build_ok": b.get("ok"), "rc": rc,
            "binding_variant": BINDING}))  # fmt: skip
        for f in sorted((ROOT / "refs").rglob("*")):
            if f.is_file():
                os.chmod(f, 0o444)
    return rc


def cmd_score(a) -> int:
    if not require("score"):
        return 2
    excluded = rebind("score")
    jobs = load_jobs_a1()
    evals = load_evals_a1(jobs)
    cand = candidates_a1(jobs, evals, excluded)
    rp = check_rollouts(evals, cand)
    if rp:
        print("FATAL: rollouts differ from inputs_a1/ROLLOUTS_A1c.sha256:\n  " + "\n  ".join(rp[:20]))
        return 2
    rc = C.cmd_score(a)  # pinned PREREG 3 scoring, unchanged
    if (ROOT / "scores/consensus_job_stats.json").is_file():
        js = C.read_json(ROOT / "scores/consensus_job_stats.json")
        C.write_text(ROOT / "scores/consensus_a1_run.json", C.dump({
            "written_at": C.now(), "ran": "t2cons_consensus.cmd_score (pinned, unchanged) via scripts/t2cons_a1_consensus.py score",
            "a1_script_sha256": sha(MINE), "consensus_script_sha256": CONS_SHA, "pins_a1c_sha256": sha(PINS_C),
            "amendments_sha256": AMEND_SHA, "gate_a1_new_sha256": sha(GATE_NEW_JSON),
            "rebound": ["require_pins -> amended pin chain", "load_jobs -> + manifest_a1.json (A3)",
                        "load_evals -> + own_input_evals_a1.csv (A3)", "candidates -> PREREG 4 + A3 minus validation_failed"],
            "validation_failed_jobs": excluded, "n_candidates": len(cand),
            "n_candidates_variant_A": sum("A" in jobs[j]["variants"] for j in cand), "candidates": cand,
            "rollouts_sha256_list": sha(ROLLOUTS), "consensus_scores_sha256": sha(ROOT / "scores/consensus_scores.csv"),
            "consensus_job_stats_sha256": sha(ROOT / "scores/consensus_job_stats.json"),
            "n_rows": js.get("n_rows"), "n_ok_rows": js.get("n_ok_rows"), "problems": js.get("problems"), "rc": rc}))
        for f in ("scores/consensus_scores.csv", "scores/consensus_job_stats.json", "scores/consensus_a1_run.json"):
            os.chmod(ROOT / f, 0o444)
    return rc


# ================================================================================================ select (S5, variant B binding)
RULE = ("PREREG 5 as amended (AMENDMENTS A2.2): default D = t2_best5 SELECTED (snapshot), the same for both variants; a "
        "challenger qualifies iff cons_mean_AUC >= cons_mean_AUC(D) + 0.01 (exact floats) AND own-input SP-SR = MP-SR = 1 "
        "in >= 4 of its 5 official runs; pick the qualifying one with the highest cons_mean_AUC (ties: lower population "
        "variance, lower cons_mean MPPE, job name). Variant B (reference R/F/C/G, all candidates incl. G2 input) is "
        "BINDING; variant A (reference R/F/C, no G2-input job) is reported only. CLOSEST run = the run of the pick whose "
        "variant-B consensus AUC is closest to its 5-run variant-B mean (ties: lower run).")


def own_mean(own: dict, j: str, m: str):
    xs = [own[j][k][m] for k in sorted(own.get(j, {}))]
    return float(np.mean(xs)) if xs else None


def fmt(x, spec=".4f") -> str:
    return "" if x is None else format(float(x), spec)


def do_select(jstats: dict, jobs: dict, evals: dict, cand: list[str], excluded: list[str], out: Path, synthetic: bool,
              hashes: dict) -> tuple[int, list[Path]]:
    own = {j: evals[j] for j in cand}
    snap_sel = C.read_json(C.SNAP / "t2_best5/selection.json")
    by_ep = {int(x["episode"]): x for x in snap_sel["episodes"]}
    problems = count_check(jobs, cand, excluded)
    out_eps, table, longrows, lines = [], [], [], []
    for e in range(30):
        s = by_ep[e]
        D = s["selected"]
        flags = ["PENDING"] if e == 4 else []
        rec = {"episode": e, "object": s.get("object"), "default": D, "default_input": jobs[D]["input"],
               "default_source": "t2_best5 SELECTED (snapshot)", "t2best5_flags": s.get("flags", []), "flags": flags,
               "variants": {}}  # fmt: skip
        for v in ("A", "B"):
            pool = pools(jobs, cand, e, v)
            chal = [j for j in pool if j != D]
            cons = {j: {"n": jstats[j][v]["n"], "auc": jstats[j][v]["auc"], "mppe": jstats[j][v]["mppe"]} for j in pool}
            pick, reason, fl, rows = C.select_episode(D, chal, cons, own)  # pinned rule
            dst = jstats[D][v]
            dmean = dst["auc_stats"]["mean"] if dst.get("n") else None
            rec["variants"][v] = {
                "reference_tracks": list(C.EXPECTED_USED[v][e]), "pool": pool,
                "default": {"job": D, "input": jobs[D]["input"], "n_scored": dst.get("n"), "cons_auc": dst.get("auc"),
                            **({"cons_auc_mean": dmean, "cons_auc_var": dst["auc_stats"]["var"],
                                "cons_mppe_mean": dst["mppe_stats"]["mean"], "cons_sp_mean": dst["sp_sr_stats"]["mean"],
                                "cons_mp_mean": dst["mp_sr_stats"]["mean"]} if dst.get("n") else {}),
                            "own_auc_mean": own_mean(own, D, "auc"), "floor_count": C.floor_count(own.get(D, {})),
                            "n_own_ok": len(own.get(D, {}))},
                "challengers": [{**r, "input": jobs[r["job"]]["input"], "family": jobs[r["job"]]["family"],
                                 "source": jobs[r["job"]]["source"], "provisional": jobs[r["job"]]["provisional"],
                                 "own_auc_mean": own_mean(own, r["job"], "auc"),
                                 "cons_sp_mean": jstats[r["job"]][v].get("sp_sr_stats", {}).get("mean"),
                                 "cons_mp_mean": jstats[r["job"]][v].get("mp_sr_stats", {}).get("mean")} for r in rows],
                "pick": pick, "pick_input": jobs[pick]["input"], "reason": reason, "flags": fl,
                "pick_closest_run": jstats[pick][v].get("closest_run"),
            }  # fmt: skip
            for role, j in [("D", D)] + [("challenger", r["job"]) for r in rows]:
                st = jstats[j][v]
                r0 = next((r for r in rows if r["job"] == j), {})
                longrows.append({
                    "episode": f"{e:02d}", "variant": v, "binding": v == BINDING, "role": role, "job": j,
                    "input": jobs[j]["input"], "source": jobs[j]["source"], "n_scored": st.get("n"),
                    **{f"cons_auc_r{k}": (st["auc"][k - 1] if st.get("n") == C.NRUNS else None) for k in range(1, 6)},
                    "cons_auc_mean": st.get("auc_stats", {}).get("mean"),
                    "cons_auc_sd": math.sqrt(st["auc_stats"]["var"]) if st.get("n") else None,
                    "cons_mppe_mean": st.get("mppe_stats", {}).get("mean"),
                    "cons_sp_mean": st.get("sp_sr_stats", {}).get("mean"), "cons_mp_mean": st.get("mp_sr_stats", {}).get("mean"),
                    "delta_vs_D": (st["auc_stats"]["mean"] - dmean) if (st.get("n") and dmean is not None) else None,
                    "own_auc_mean": own_mean(own, j, "auc"), "own_mppe_mean": own_mean(own, j, "mppe"),
                    "own_floor": C.floor_count(own.get(j, {})), "n_own_ok": len(own.get(j, {})),
                    "qualifies": r0.get("qualifies") if role == "challenger" else "",
                    "why_not": "; ".join(r0.get("why_not", [])) if role == "challenger" else "",
                    "picked": j == pick, "closest_run": st.get("closest_run"),
                })  # fmt: skip
        A, B = rec["variants"]["A"], rec["variants"]["B"]
        rec.update({"pick": B["pick"], "pick_input": B["pick_input"], "binding_variant": BINDING,
                    "pick_B": B["pick"], "pick_A": A["pick"], "changed_B": B["pick"] != D, "changed_A": A["pick"] != D,
                    "closest_run_B": B["pick_closest_run"], "reason_B": B["reason"], "reason_A": A["reason"]})  # fmt: skip
        out_eps.append(rec)
        pb = jstats[B["pick"]]["B"]

        def chal_str(V, vv):
            parts = []
            for c in V["challengers"]:
                if "cons_auc_mean" in c:
                    parts.append(f"{c['job']} [{c['input']}] cons-{vv} {c['cons_auc_mean']:.4f} ({c['delta_vs_D']:+.4f}) "
                                 f"floor {c['floor_count']}/5 {'QUALIFIES' if c['qualifies'] else 'no'}")
                else:
                    parts.append(f"{c['job']} [{c['input']}] incomplete ({'; '.join(c['why_not'])})")
            return "; ".join(parts)

        table.append({
            "episode": f"{e:02d}", "object": s.get("object"), "default": D, "default_input": jobs[D]["input"],
            "D_consB_auc": B["default"].get("cons_auc_mean"),
            "D_consB_sd": math.sqrt(B["default"]["cons_auc_var"]) if "cons_auc_var" in B["default"] else None,
            "D_consA_auc": A["default"].get("cons_auc_mean"), "D_own_auc": B["default"]["own_auc_mean"],
            "D_floor": f"{B['default']['floor_count']}/{C.NRUNS}", "n_chal_B": len(B["challengers"]),
            "challengers_B": chal_str(B, "B"),
            "pick_B": B["pick"], "pick_B_input": B["pick_input"], "pick_B_consB_auc": pb.get("auc_stats", {}).get("mean"),
            "pick_B_consB_sd": math.sqrt(pb["auc_stats"]["var"]) if "auc_stats" in pb else None,
            "pick_B_own_auc": own_mean(own, B["pick"], "auc"),
            "pick_B_floor": f"{C.floor_count(own[B['pick']])}/{C.NRUNS}", "closest_run_B": B["pick_closest_run"],
            "changed_B": B["pick"] != D, "reason_B": B["reason"],
            "n_chal_A": len(A["challengers"]), "pick_A": A["pick"], "changed_A": A["pick"] != D, "reason_A": A["reason"],
            "flags": ";".join(flags + B["flags"]),
        })  # fmt: skip
        for v in (BINDING, "A"):
            V = rec["variants"][v]
            dd = V["default"]
            lines.append(f"ep{e:02d} [{v}{' BINDING' if v == BINDING else ' report'}] D {D} [{jobs[D]['input']}] "
                         f"cons-{v} AUC {fmt(dd.get('cons_auc_mean'))} own floor {dd['floor_count']}/5 -> {V['pick']} "
                         f"({V['reason']})" + (f" flags {flags}" if flags else ""))  # fmt: skip
            for c in V["challengers"]:
                lines.append(f"      {c['job']} [{c['input']}] cons-{v} AUC {fmt(c.get('cons_auc_mean'))} sd "
                             f"{fmt(math.sqrt(c['cons_auc_var'])) if 'cons_auc_var' in c else ''} delta "
                             f"{fmt(c.get('delta_vs_D'), '+.4f')} floor {c['floor_count']}/5 qualifies={c['qualifies']} "
                             f"{c['why_not']}")  # fmt: skip
    if problems:
        print("FATAL (consistency with PREREG / AMENDMENTS):\n  " + "\n  ".join(problems))
        return 2, []
    comps = {"CONS-B": {x["episode"]: x["pick_B"] for x in out_eps},
             "CONS-A": {x["episode"]: x["pick_A"] for x in out_eps},
             "DEFAULT": {x["episode"]: x["default"] for x in out_eps},
             "RAW-ONLY": {int(x["episode"]): x["rawref"] for x in snap_sel["episodes"]},
             "K3FULL": {int(x["episode"]): x["chosen"] for x in snap_sel["episodes"]}}  # fmt: skip
    for name, m in comps.items():
        for e, j in m.items():
            if j not in cand:
                print(f"FATAL: composition {name} ep{e:02d} job {j} is not a candidate")
                return 2, []
    summ = {name: {"jobs": {f"{e:02d}": j for e, j in sorted(m.items())}, **C.comp_summary(m, jstats, own)}
            for name, m in comps.items()}  # fmt: skip
    gen = C.now()
    sel = {
        "generated_at": gen, "synthetic": synthetic, **hashes, "rule": RULE, "binding_variant": BINDING,
        "noisy_only": True, "validation_failed_jobs": excluded,
        "n_candidates": len(cand), "n_candidates_variant_A": sum("A" in jobs[j]["variants"] for j in cand),
        "changed_episodes_B": [f"{x['episode']:02d}" for x in out_eps if x["changed_B"]],
        "changed_episodes_A": [f"{x['episode']:02d}" for x in out_eps if x["changed_A"]],
        "cons_B_closest_runs": {f"{x['episode']:02d}": {"job": x["pick_B"], "run": x["closest_run_B"],
                                                         "input": x["pick_input"],
                                                         "frame_reference": jobs[x["pick_B"]]["input_reference"]}
                                for x in out_eps},
        "compositions": {k: {f"{e:02d}": j for e, j in sorted(m.items())} for k, m in comps.items()},
        "episodes": out_eps,
    }  # fmt: skip
    out.mkdir(parents=True, exist_ok=True)
    written = []

    def w(name: str, text: str) -> None:
        C.write_text(out / name, text)
        written.append(out / name)

    w("selection.json", C.dump(sel))
    w("selection_consensus.json", C.dump(sel))
    w("selection.txt", "\n".join([f"selection {gen} ({'SYNTHETIC TEST, NOT A RESULT; ' if synthetic else ''}noisy-only "
                                  f"consensus; binding variant {BINDING}; refs {str(hashes.get('refs_build_sha256'))[:12]}; "
                                  f"scores {str(hashes.get('consensus_scores_sha256'))[:12]})"] + lines) + "\n")
    w("selection_table.csv", C.csv_text(table, list(table[0].keys())))
    w("selection_candidates.csv", C.csv_text(longrows, list(longrows[0].keys())))
    md = [f"# t2_consensus_20261024: selection ({'SYNTHETIC TEST, NOT A RESULT' if synthetic else 'noisy-only'}; "
          "PREREG 5 under AMENDMENTS A1-A3)", "",
          f"Generated {gen} CST. Binding: variant B (consensus of R/F/C/G; candidates include G2-input jobs). Variant A "
          "(R/F/C, no G2 job) is reported only. Default D = t2_best5 SELECTED. A challenger replaces D iff its 5-run "
          "consensus AUC mean >= D's + 0.01 and its own-input SP-SR = MP-SR = 1 in >= 4 of 5 official runs. No clean "
          "or LOCAL file was opened.", "",
          f"Changed vs DEFAULT under B (binding): {', '.join(sel['changed_episodes_B']) or 'none'}. "
          f"Under A (report): {', '.join(sel['changed_episodes_A']) or 'none'}."
          + (f" validation_failed (not candidates): {', '.join(excluded)}." if excluded else ""), "",
          "| ep | object | default D (input) | D cons-B AUC (sd) | D own AUC / floor | challengers under B: cons-B AUC "
          "(delta vs D), own floor, qualifies | **pick (B)** | pick cons-B AUC (sd) | pick own AUC / floor | CLOSEST run "
          "| reason (B) | pick A (report) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for t in table:
        md.append(f"| {t['episode']} | {t['object']} | {t['default']} ({t['default_input']}) | {fmt(t['D_consB_auc'])} "
                  f"({fmt(t['D_consB_sd'])}) | {fmt(t['D_own_auc'])} / {t['D_floor']} | {t['challengers_B'].replace('; ', '<br>') or '-'} | "
                  f"**{t['pick_B']}** ({t['pick_B_input']}) | {fmt(t['pick_B_consB_auc'])} ({fmt(t['pick_B_consB_sd'])}) | "
                  f"{fmt(t['pick_B_own_auc'])} / {t['pick_B_floor']} | {t['closest_run_B']} | {t['reason_B']}"
                  f"{' [' + t['flags'] + ']' if t['flags'] else ''} | {t['pick_A']}{' (changed)' if t['changed_A'] else ''} |")
    w("selection_table.md", "\n".join(md) + "\n")
    smd = [f"# t2_consensus_20261024: composition summary ({'SYNTHETIC TEST' if synthetic else 'noisy-only'})", "",
           "Averages over the 30 episodes of each episode's 5-run mean. cons_B / cons_A = against the consensus "
           "references (B binding); own_input = the evaluator's own-input values (each against its own input, not "
           "comparable across inputs). Consensus numbers are not leaderboard estimates and favour the selected "
           "composition by construction (PREREG 9.6).", "",
           "| composition | cons-B AUC | cons-B SP | cons-B MP | cons-B MPPE | cons-A AUC | own AUC | own SP | own MP | own MPPE |",
           "|---|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for name, d in summ.items():
        smd.append(f"| {name} | {fmt(d['cons_B']['auc']['average'])} | {fmt(d['cons_B']['sp_sr']['average'])} | "
                   f"{fmt(d['cons_B']['mp_sr']['average'])} | {fmt(d['cons_B']['mppe']['average'])} | "
                   f"{fmt(d['cons_A']['auc']['average'])} | {fmt(d['own_input']['auc']['average'])} | "
                   f"{fmt(d['own_input']['sp_sr']['average'])} | {fmt(d['own_input']['mp_sr']['average'])} | "
                   f"{fmt(d['own_input']['mppe']['average'])} |")  # fmt: skip
    w("summary_consensus.json", C.dump({"generated_at": gen, "synthetic": synthetic, "binding_variant": BINDING,
                                        "compositions": summ}))  # fmt: skip
    w("summary_consensus.md", "\n".join(smd) + "\n")
    print("\n".join(smd))
    print()
    print("\n".join(md[6:]))
    return 0, written


FROZEN_INPUTS = ["scores/consensus_scores.csv", "scores/consensus_job_stats.json", "scores/consensus_a1_run.json",
                 "refs/build.json", "refs/build_a1.json", "scores/own_input_validation_a1_new.json",
                 "scores/own_input_validation_a1_new.csv", "scores/own_input_validation_a1.json",
                 "scores/own_input_validation.json", "selftest/selftest.json", "manifest_a1.json", "own_input_evals_a1.csv",
                 "inputs_a1/candidates_a1.json", "inputs_a1/ROLLOUTS_A1c.sha256", "PREREG_PINS_A1c.md",
                 "PREREG_PINS_A1b.md", "PREREG_PINS.md", "AMENDMENTS.md", "PREREG.md"]  # fmt: skip


def common_hashes() -> dict:
    return {
        "prereg_sha256": sha(C.PREREG), "amendments_sha256": sha(AMEND), "pins_sha256": sha(PINS),
        "pins_a1b_sha256": sha(PINS_B), "pins_a1c_sha256": sha(PINS_C), "script_sha256": sha(MINE),
        "consensus_script_sha256": CONS_SHA, "gate_script_sha256": GATE_SHA,
        "own_input_validation_a1_sha256": sha(VAL_A1_JSON), "own_input_validation_a1_new_sha256": sha(GATE_NEW_JSON),
        "manifest_a1_sha256": sha(MAN_A1), "own_input_evals_a1_sha256": sha(EVALS_A1), "candidates_a1_sha256": sha(CANDS),
        "rollouts_list_sha256": sha(ROLLOUTS), "snapshot_selection_sha256": sha(C.SNAP / "t2_best5/selection.json"),
    }  # fmt: skip


def cmd_select(a) -> int:
    if SEL_SHA.exists():
        print("FATAL: SELECTION.sha256 exists; the selection is frozen")
        return 2
    if not require("select"):
        return 2
    missing = [f for f in FROZEN_INPUTS if not (ROOT / f).is_file()]
    if missing:
        print(f"FATAL: inputs to freeze are missing: {missing}")
        return 2
    excluded = excluded_jobs()
    js = C.read_json(ROOT / "scores/consensus_job_stats.json")
    run = C.read_json(ROOT / "scores/consensus_a1_run.json")
    if js.get("scores_csv_sha256") != sha(ROOT / "scores/consensus_scores.csv") or \
            js.get("refs_build_sha256") != sha(ROOT / "refs/build.json") or not js.get("integrity_kit_stock_unchanged") or \
            run.get("consensus_job_stats_sha256") != sha(ROOT / "scores/consensus_job_stats.json") or \
            run.get("a1_script_sha256") != sha(MINE) or run.get("validation_failed_jobs") != excluded:  # fmt: skip
        print("FATAL: scores / refs changed since scoring, scored by another script, or integrity flagged")
        return 2
    if not C.read_json(ROOT / "refs/build.json").get("ok"):
        print("FATAL: refs/build.json not ok")
        return 2
    jobs = load_jobs_a1()
    evals = load_evals_a1(jobs)
    cand = candidates_a1(jobs, evals, excluded)
    jstats = js["jobs"]
    if sorted(jstats) != sorted(cand) or run.get("candidates") != cand:
        print("FATAL: scored job set != candidate set")
        return 2
    hashes = {**common_hashes(), "refs_build_sha256": sha(ROOT / "refs/build.json"),
              "consensus_scores_sha256": sha(ROOT / "scores/consensus_scores.csv"),
              "consensus_job_stats_sha256": sha(ROOT / "scores/consensus_job_stats.json")}  # fmt: skip
    rc, written = do_select(jstats, jobs, evals, cand, excluded, ROOT, False, hashes)
    if rc:
        return rc
    names = [str(p.relative_to(ROOT)) for p in written]
    body = "".join(f"{sha(ROOT / f)}  {f}\n" for f in names + FROZEN_INPUTS)
    for p in written:
        os.chmod(p, 0o444)
    C.write_text(SEL_SHA, body)  # LAST: from here the selection is frozen
    os.chmod(SEL_SHA, 0o444)
    print(f"\nSELECTION.sha256 written {C.now()} (frozen): {sha(SEL_SHA)}")
    return 0


def cmd_selftest_select(a) -> int:
    """The select code on SYNTHETIC random consensus values (seeded); writes only under selftest/a1_select_synthetic/."""
    if not require("synthetic"):
        return 2
    if SYNTH_DIR.exists():
        print("FATAL: selftest/a1_select_synthetic exists (new files only)")
        return 2
    if (ROOT / "scores/consensus_scores.csv").exists():
        print("FATAL: real consensus scores exist; the synthetic select test runs only before them")
        return 2
    excluded = excluded_jobs()
    jobs = load_jobs_a1()
    evals = load_evals_a1(jobs)
    cand = candidates_a1(jobs, evals, excluded)
    rng = np.random.default_rng(20261026)
    jstats = {}
    for j in cand:
        jstats[j] = {"episode": jobs[j]["episode"], "input": jobs[j]["input"], "variants": list(jobs[j]["variants"])}
        for v in ("A", "B"):
            d = {"n": 5, "runs": [1, 2, 3, 4, 5]}
            base = rng.uniform(0.1, 0.9)
            vals = {"auc": list(np.clip(base + rng.normal(0, 0.02, 5), 0, 1)), "sp_sr": list(rng.integers(0, 2, 5).astype(float)),
                    "mp_sr": list(rng.integers(0, 2, 5).astype(float)), "mppe": list(rng.uniform(0.5, 20, 5))}  # fmt: skip
            for m in C.METRICS:
                d[m] = [float(x) for x in vals[m]]
                d[m + "_stats"] = C.stats(d[m])
            mu = d["auc_stats"]["mean"]
            d["closest_run"] = min(range(1, 6), key=lambda k: (abs(d["auc"][k - 1] - mu), k))
            jstats[j][v] = d
    hashes = {"synthetic_seed": 20261026, "refs_build_sha256": None, "consensus_scores_sha256": None}
    try:
        hashes.update(common_hashes())
    except Exception as err:  # noqa: BLE001
        print(f"hash collection failed: {err!r}")
        return 1
    rc, written = do_select(jstats, jobs, evals, cand, excluded, SYNTH_DIR, True, hashes)
    if rc:
        return rc
    # independent re-derivation of the binding picks from the synthetic stats (checks do_select's plumbing)
    sel = C.read_json(SYNTH_DIR / "selection.json")
    snap = {int(x["episode"]): x["selected"] for x in C.read_json(C.SNAP / "t2_best5/selection.json")["episodes"]}
    bad = []
    n_changed = 0
    for x in sel["episodes"]:
        e, D = x["episode"], snap[x["episode"]]
        for v in ("A", "B"):
            dm = float(np.mean(jstats[D][v]["auc"]))
            q = []
            for j in pools(jobs, cand, e, v):
                if j == D:
                    continue
                m = float(np.mean(jstats[j][v]["auc"]))
                fl = sum(1 for r in evals[j].values() if r["sp_sr"] == 1.0 and r["mp_sr"] == 1.0)
                if m >= dm + 0.01 and fl >= 4:
                    q.append((-m, float(np.var(jstats[j][v]["auc"])), float(np.mean(jstats[j][v]["mppe"])), j))
            want = sorted(q)[0][3] if q else D
            if x["variants"][v]["pick"] != want:
                bad.append(f"ep{e:02d} {v}: do_select {x['variants'][v]['pick']} != re-derived {want}")
            n_changed += (v == "B" and want != D)
        if x["pick"] != x["variants"]["B"]["pick"] or x["closest_run_B"] != jstats[x["pick"]]["B"]["closest_run"]:
            bad.append(f"ep{e:02d}: binding pick / CLOSEST not variant B's")
        if any(jobs[j]["input"] == "g2" for j in x["variants"]["A"]["pool"]):
            bad.append(f"ep{e:02d}: a G2-input job in the variant-A pool")
    body = "".join(f"{sha(p)}  {p.relative_to(ROOT)}\n" for p in written)
    C.write_text(SYNTH_DIR / "SYNTHETIC_TEST_NOT_A_SELECTION.sha256", body)
    res = {"written_at": C.now(), "script_sha256": sha(MINE), "synthetic": True, "pass": not bad, "problems": bad,
           "n_episodes": len(sel["episodes"]), "n_changed_B_synthetic": n_changed,
           "note": "random seeded values in place of consensus scores; checks only that the select code runs end to end "
                   "and that its picks equal an independent re-derivation of the PREREG 5 rule; no score was read or "
                   "computed; not a selection"}  # fmt: skip
    C.write_text(SYNTH_DIR / "selftest_select.json", C.dump(res))
    print(json.dumps(res, indent=1))
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("check", "gate", "refs", "score", "select", "selftest-select"))
    a = ap.parse_args()
    if not sys.flags.dont_write_bytecode or not sys.flags.isolated:
        print("run with the kit python -I -B")
        return 2
    return {"check": cmd_check, "gate": cmd_gate, "refs": cmd_refs, "score": cmd_score, "select": cmd_select,
            "selftest-select": cmd_selftest_select}[a.cmd](a)  # fmt: skip


if __name__ == "__main__":
    raise SystemExit(main())
