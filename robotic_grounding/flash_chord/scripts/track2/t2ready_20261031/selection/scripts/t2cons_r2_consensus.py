#!/usr/bin/env python3
"""t2cons_r2_consensus.py: step R5 of fc_runs/submissions/t2_consensus_r2_20261027 (PREREG_R2 5.2-5.5 and 6, under
AMENDMENTS_R2, AMENDMENTS_R2_b and AMENDMENTS_R2_c): rollout list, own-input gate, consensus-B scores, selection, and
SELECTION_R2.sha256 (written last).

Commands
  selftest  PREREG_R2 5.5: synthetic checks of the only new logic (the 10-run challenger / 5-run incumbent bookkeeping
            of 5.4, the CLOSEST run of 6) and of the extended path guard (5.4.6). No real score is read or computed.
            -> selftest_r2/selftest_r2.json
  check     dry: pin chain, round-2 jobs and runs (runs_b/, AMENDMENTS_R2 A2), frozen round-1 incumbents, rollout files.
            No metric is computed. -> dryrun/r2_consensus_check_<stamp>.txt and, once, inputs_r2/ROLLOUTS_R2.sha256 (5.2)
  gate      PREREG_R2 5.3 (round-1 AMENDMENTS A1.3): every round-2 run against its OWN input track with the pinned
            metrics_vs (seed 0) and the pinned auc_seed (seeds 1..9). -> scores_r2/own_input_gate_r2.{csv,json}
  score     PREREG_R2 5.2: every run of every job that is not validation_failed, against round 1's frozen refs/B, with
            the pinned rollout_on_grid + metrics_vs. -> scores_r2/consensus_scores_r2.csv, consensus_job_stats_r2.json
  select    PREREG_R2 5.4 + 6 -> selection_r2.{json,txt}, selection_r2_table.{md,csv}, summary_r2_consensus.{md,json};
            SELECTION_R2.sha256 LAST.

Pinned code reuse. scripts/t2cons_a1_gate.py (a29c997b...) is compiled from its verified bytes; its module code loads
scripts/t2cons_consensus.py, whose bytes must equal e1b943b1.... The functions rollout_on_grid, metrics_vs, stats,
floor_count, load_kit, load_refs, grid, integrity / integrity_diff, check_pins (round-1 chain) and auc_seed are called
unchanged. Only module-level name rebound: t2cons_consensus.FORBIDDEN gains "readme" and "train_rl.log" (PREREG_R2
5.4.6 extends the guard). The pinned select_episode hard-codes 5 / 5 runs, so 5.4 is re-implemented in select_r2();
the selftest shows it equals the pinned select_episode on 5-run inputs (n_req 5, floor 4).

Noisy-only: every path read by gate / score / select goes through the extended guard (clean, LOCAL_NOT_FOR_UPLOAD,
packed_, tier_1_multiview_caption, track_2/baseline, /data/human_motion_data/, /data/v2d_challenge_data_public,
readme, train_rl.log). No clean file, no packer, no clean reference. Variant B only (variant A is not part of round 2).
New files only; nothing is signalled; CPU only.
Run: conda_envs/v2dkit/bin/python -I -B scripts/t2cons_r2_consensus.py <cmd>  (the kit Python, as round 1: the pinned
kit code needs pandas, which the flashchord env lacks)
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
ROOT = ME / "fc_runs/submissions/t2_consensus_r2_20261027"
R1 = ME / "fc_runs/submissions/t2_consensus_20261024"
GATE_FILE, GATE_SHA = ME / "scripts/t2cons_a1_gate.py", "a29c997b7271e9299db109e97af5210f528484719210bb58685523b0ae0eb16b"
CONS_SHA = "e1b943b10fd1db13ab6f78e5bde7b3c6e6c48f7daeee6230c32c463d14ac41b4"
PREREG = ROOT / "PREREG_R2.md"
PREREG_SHA = "705114045eaf9523bc880705e9a24c45c535536ce9884fe3031160579f3ba4c6"
CHAIN_DOCS = (  # (file, its sha256 file, hard-coded sha256); each must hold, and every pin line in it must hold
    ("PREREG_R2_PINS.md", "PREREG_R2_PINS.sha256", "5f1e7126c30deff08d96bb9e49fd14ebd7dc96494cde8b3f1d2da7a0a2d9594a"),
    ("AMENDMENTS_R2.md", "AMENDMENTS_R2.sha256", "f5cc8f3dc58619b1c40792559ddafbbd8fbf91b16d2634705f7290091e349e55"),
    ("PREREG_R2_PINS_b.md", "PREREG_R2_PINS_b.sha256", "b88cda25c9a0c0b5e35e4efdd2f5fc26fab069cc995ee834a39377f3fd92eed2"),
    ("AMENDMENTS_R2_b.md", "AMENDMENTS_R2_b.sha256", "ecf3cad239faf6d464224e7515779a3e867cc7f98e71232ffe7d695349a0f597"),
    ("AMENDMENTS_R2_c.md", "AMENDMENTS_R2_c.sha256", "ea0f5e9a69f95fc9f9734024f7ef7dcaac7cfd4925b36cdaea50c9759fa7e279"),
    ("PREREG_R2_PINS_c.md", "PREREG_R2_PINS_c.sha256", "c66626ebd51f9ad22ac8944f915d5893bb98867dcb60ec63de8de9c22d4d836e"),
)  # fmt: skip
PINS_D, PINS_D_SHA = ROOT / "PREREG_R2_PINS_d.md", ROOT / "PREREG_R2_PINS_d.sha256"  # AMENDMENTS_R2_c C5
CANDS, CANDS_SHA = ROOT / "inputs_r2/candidates_r2.json", "a86f6153235fbda2491cfe0660d4550691e370b9139bcaac51ceca0fbfb3be3f"
SNAP_LISTS = ("inputs_r2/SNAPSHOT_R2.sha256", "inputs_r2/ROUND1_FROZEN.sha256", "inputs_r2/TRACKS_R2.sha256")
TRACKS_R2 = ROOT / "inputs_r2/TRACKS_R2.sha256"
R1_FILES = {  # round-1 frozen files read here (also PREREG_R2 section 10 pins)
    "SELECTION.sha256": "651853a61bac0ab2c244a0e7ebed9db2d46d90f7d22d95f12fd545cda6b25046",
    "selection.json": "e5804138b074a9927b30bed598ec1a13a670eb04116b9dc906bd904190755dba",
    "scores/consensus_scores.csv": "50cc6e9401ad8c88594918490cee345d8aaf7368740fb3626c65525bfcb40a52",
    "scores/consensus_job_stats.json": "9e255057556ccc23df9c97dafd101219770e8b53cc6ab28519ab893eb86319c8",
    "refs/build.json": "9646de2daf02301dd2efbc72a7507c539b1100c3ec0fe39adade169195d18dd3",
}
MANIFEST, EVALS = ROOT / "manifest_r2b.json", ROOT / "own_input_evals_r2b.csv"  # AMENDMENTS_R2 A2 names
RUNS_B, EXPORTS_B = ROOT / "runs_b", ROOT / "exports_b"
BASELINE = ROOT / "integrity/baseline.json"
HEARTBEAT = ROOT / "eval_r2b_heartbeat.json"
ROLLOUTS = ROOT / "inputs_r2/ROLLOUTS_R2.sha256"
SELFTEST = ROOT / "selftest_r2/selftest_r2.json"
SC = ROOT / "scores_r2"
GATE_CSV, GATE_JSON = SC / "own_input_gate_r2.csv", SC / "own_input_gate_r2.json"
SCORES_CSV, JSTATS = SC / "consensus_scores_r2.csv", SC / "consensus_job_stats_r2.json"
SEL_SHA = ROOT / "SELECTION_R2.sha256"
NRUNS, NRUNS_R1, FLOOR, MARGIN = 10, 5, 8, 0.01  # PREREG_R2 3.1 / 4 / 5.4 (unchanged by AMENDMENTS_R2 A5)
VARIANT = "B"
EXTRA_FORBIDDEN = ("readme", "train_rl.log")  # PREREG_R2 5.4.6
NEW_JOBS = {  # PREREG_R2 2.1 / 2.2: job -> (episode, input, id)
    "ep01_zx_g2_172000": (1, "g2", "Z1"), "ep04_zx_g2_130000": (4, "g2", "Z1"), "ep05_zx_k3fix_122000": (5, "k3fix", "Z3"),
    "ep10_zxc_g2_12000": (10, "g2", "C4"), "ep15_zxc_g2_62000": (15, "g2", "C1"), "ep19_zx_g2_146000": (19, "g2", "Z1"),
    "ep19_zx_g2hard_140000": (19, "g2", "Z2"), "ep19_zxc_g2hard_198000": (19, "g2", "C2"),
    "ep23_zxc_g2_8000": (23, "g2", "C3"), "ep26_zx_k3fix_108000": (26, "k3fix", "Z3"), "ep28_zx_g2_120000": (28, "g2", "Z1"),
}  # fmt: skip
EXPORT_SHA = {  # PREREG_R2_PINS_c pins (exports_b/<job>/try1/model_<it>.safetensors)
    "ep01_zx_g2_172000": "f58867a4cc06322fddaea0fb6fe343028724a5037f42b69efd17207cc0aa5de7",
    "ep04_zx_g2_130000": "3ae6b584fe131e35d8c22fe8f77a84f0ee8d4d48d45582af7d423f5e216e2673",
    "ep05_zx_k3fix_122000": "f3a83e5d71956fc709d25a8dc77688d6e39b9bbd1d8e4071a70bfab952e8727f",
    "ep10_zxc_g2_12000": "20eed4b5f2ea3aebce2f8f3704125383208dd7c52469dbf728e5a5e0330ff042",
    "ep15_zxc_g2_62000": "6d4c65912d4b9f9a297ec6223570393350c2f0b7730af2ea6c614df15f84593f",
    "ep19_zx_g2_146000": "b79be85bdf50afcab8638eb47bfafad8ccf483a0f8a77d43b212818c87061e61",
    "ep19_zx_g2hard_140000": "0dd6f36ea0850b1d9f3d9380cb36c96b3b427efb90f440fda6c14f30051ab01a",
    "ep19_zxc_g2hard_198000": "1939f11686dc1040dfd9b5602f48e5a7485dbc906242a04601aca1be32ca3037",
    "ep23_zxc_g2_8000": "f099f8fee5168551069670f4291e8948290c4db8be70d87041b815fb9bcb68bb",
    "ep26_zx_k3fix_108000": "c0c6851914d7c0b55489d0a2762bd0b0498fd539ad3a6d64bf51b73abd84e5b6",
    "ep28_zx_g2_120000": "6fc0bcd723ac2627645c282a7344dfd3937ee0fd8b650fbef4cd7e2de6493776",
}
INCUMBENTS = {  # PREREG_R2 4: ep -> (round-1 CONS-B pick, m_P rounded, round-1 CLOSEST run, reference tracks)
    1: ("ep01_k3fix_final44922", 0.171319, 2, ("R", "F", "G")), 4: ("ep04_raw_final44922", 0.120283, 1, ("R", "F", "G")),
    5: ("ep05_zx_k3fix_88000", 0.578153, 4, ("R", "F", "C")), 10: ("ep10_raw_final34267", 0.412469, 5, ("R", "F")),
    15: ("ep15_raw_final44922", 0.729895, 4, ("R", "F", "C")), 19: ("ep19_zx_g2_92000", 0.517015, 5, ("R", "F", "C", "G")),
    23: ("ep23_raw_final20023", 0.902867, 1, ("R", "F")), 26: ("ep26_zx_k3fix_92000", 0.731484, 3, ("R", "F", "C")),
    28: ("ep28_zx_g2_100000", 0.132642, 3, ("R", "F", "C", "G")),
}  # fmt: skip
RULE = ("PREREG_R2 5.4 (AMENDMENTS_R2 A5: unchanged): incumbent P_e = the frozen round-1 CONS-B pick, m_P = the float64 "
        "mean of its 5 frozen variant-B consensus AUCs; a round-2 challenger c (10 ok official own-input runs, 10 "
        "variant-B consensus-scored runs, not validation_failed) qualifies iff cons_mean_B(c) >= m_P + 0.01 (exact "
        "floats) AND own-input SP-SR = MP-SR = 1 in >= 8 of its 10 runs; pick the qualifying one with the highest "
        "cons_mean_B (ties: lower population variance of its 10 consensus AUCs, lower mean consensus MPPE, job name); "
        "otherwise P_e stays. R2-CLOSEST run (PREREG_R2 6): on a changed episode, the run k of the pick whose variant-B "
        "consensus AUC is closest to its 10-run mean (ties: lower k); elsewhere the round-1 CONS-B CLOSEST run.")  # fmt: skip


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


G = _load("t2cons_a1_gate_pinned", GATE_FILE, GATE_SHA)  # its module code loads t2cons_consensus.py as G.C
C = G.C
if hashlib.sha256(Path(C.__file__).read_bytes()).hexdigest() != CONS_SHA:
    raise SystemExit("t2cons_consensus.py (as loaded by t2cons_a1_gate.py) != pinned e1b943b1...")
C.FORBIDDEN = tuple(C.FORBIDDEN) + EXTRA_FORBIDDEN  # PREREG_R2 5.4.6: the guard is extended (rebinding only)
MINE = Path(__file__).resolve()


# ================================================================================================ helpers
def sha(p) -> str:
    return C.sha256_file(p)


def raw_sha(p) -> str:
    """sha256 of a pin / record file (pin chain only; not a 5.2-5.4 data read). Refuses a guarded path."""
    C.guard(p)
    h = hashlib.sha256()
    with open(p, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pin_hash(p: Path) -> str:
    """For the pin chain: the sha256, or 'missing', or 'guarded' (a path the guard refuses is not opened here)."""
    try:
        return raw_sha(p) if p.is_file() else "missing"
    except PermissionError:
        return "guarded"


def first_token(p: Path) -> str:
    return (p.read_text().split() or [""])[0] if p.is_file() else ""


def write_new(p: Path, text: str, mode: int | None = 0o444) -> None:
    """New files only (refuses to overwrite), through a temporary name; optional chmod."""
    p = C.guard(p)
    if p.exists():
        raise SystemExit(f"refusing to overwrite {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp_r2c")
    if tmp.exists():
        raise SystemExit(f"stale temporary {tmp}")
    tmp.write_text(text)
    os.replace(tmp, p)
    if mode is not None:
        os.chmod(p, mode)


def pin_lines(text: str) -> list[tuple[str, str]]:
    out = []
    for line in text.splitlines():
        f = line.split()
        if len(f) == 3 and f[0] == "pin" and len(f[2]) == 64:
            out.append((f[1], f[2]))
    return out


def pin_path(rel: str) -> Path:
    """PREREG_R2 10 path rules: '/...' absolute; 'scripts/...' relative to ME; anything else relative to ROOT."""
    if rel.startswith("/"):
        return Path(rel)
    return ME / rel if rel.startswith("scripts/") else ROOT / rel


def fmt(x, spec=".4f") -> str:
    return "" if x is None else format(float(x), spec)


def ep_key(e: int) -> str:
    return f"{e:02d}"


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
        if "t2cons_r2_eval" in cmd and "t2cons_r2_consensus" not in cmd:
            out.append(f"pid {d.name}: {cmd[:160]}")
    return out


def chain(stage: str) -> tuple[list[str], list[str]]:
    """stage: check | gate | score | select. PINS_d is required for gate / score / select."""
    fatal, info = [], []
    got = raw_sha(PREREG) if PREREG.is_file() else "missing"
    rec = first_token(ROOT / "PREREG_R2.sha256")
    if not got == rec == PREREG_SHA:
        return [f"PREREG_R2.md sha256 {got[:16]} (sha file {rec[:16]}, hard-coded {PREREG_SHA[:16]})"], info
    n = 0
    for line in PREREG.read_text().split("## 10. Pins (sha256)", 1)[-1].splitlines():
        f = line.split()
        if len(f) == 3 and f[0] == "pin":
            p = pin_path(f[1])
            h = pin_hash(p)
            n += 1
            if h == "guarded":
                info.append(f"PREREG_R2 pin {f[1]}: not opened before the freeze (path guard)")
            elif h != f[2]:
                fatal.append(f"PREREG_R2 pin {f[1]}: {h[:16]} != {f[2][:16]}")
        elif len(f) == 3 and f[0] == "commit":
            head = C.git_head(ME / f[1])
            n += 1
            if head != f[2]:
                fatal.append(f"PREREG_R2 commit {f[1]}: HEAD {head[:12]} != {f[2][:12]}")
    info.append(f"PREREG_R2.md {got[:16]} == PREREG_R2.sha256 == hard-coded; {n} section-10 pins / commits checked")
    for name, shaf, want in CHAIN_DOCS:
        p = ROOT / name
        got = pin_hash(p)
        rec = first_token(ROOT / shaf)
        if not got == rec == want:
            fatal.append(f"{name} sha256 {got[:16]} (sha file {rec[:16]}, hard-coded {want[:16]})")
            continue
        k = 0
        for rel, h0 in pin_lines(p.read_text()):
            pp = pin_path(rel)
            h = pin_hash(pp)
            k += 1
            if h == "guarded":
                info.append(f"{name} pin {rel}: not opened before the freeze (path guard)")
            elif h != h0:
                fatal.append(f"{name} pin {rel}: {h[:16]} != {h0[:16]}")
        info.append(f"{name} {got[:16]} == its sha file == hard-coded; {k} pins checked")
    for rel in SNAP_LISTS:
        k = 0
        for line in (ROOT / rel).read_text().splitlines():
            parts = line.split(None, 1)
            if len(parts) != 2 or len(parts[0]) != 64:
                continue
            p = pin_path(parts[1].strip())
            h = pin_hash(p)
            k += 1
            if h == "guarded":
                info.append(f"{rel}: {parts[1].strip()} not opened before the freeze (path guard)")
            elif h != parts[0]:
                fatal.append(f"{rel}: {parts[1].strip()} {h[:16]} != {parts[0][:16]}")
        info.append(f"{rel}: {k} lines hold")
    if raw_sha(CANDS) != CANDS_SHA:
        fatal.append("inputs_r2/candidates_r2.json != a86f6153...")
    for rel, want in R1_FILES.items():
        h = pin_hash(R1 / rel)
        if h != want:
            fatal.append(f"round-1 {rel}: {h[:16]} != {want[:16]}")
    info.append(f"round-1 frozen files read here ({len(R1_FILES)}) == pinned")
    f2, i2 = C.check_pins(need_pins=False)  # round-1 PREREG == sha, its section-11 pins + commits, SNAPSHOT, KIT
    fatal += [f"round-1 chain: {x}" for x in f2]
    info.append(f"round-1 chain (pinned t2cons_consensus.check_pins): {len(f2)} fatal; "
                f"{sum(1 for x in i2 if x.startswith(('pin ', 'commit ')))} pin / commit lines")  # fmt: skip
    hb = json.loads(HEARTBEAT.read_text()) if HEARTBEAT.is_file() else {}
    if hb.get("phase") != "done" or (hb.get("evals") or {}).get("ok") != 110 or (hb.get("evals") or {}).get("failed") != 0 \
            or (hb.get("exports") or {}).get("ok") != 11 or str(hb.get("runs_base")) != str(RUNS_B):  # fmt: skip
        fatal.append(f"round-2 evaluations not finished: heartbeat phase {hb.get('phase')} evals {hb.get('evals')}")
    run = driver_running()
    if run or (ROOT / "driver.pid").exists():
        fatal.append(f"an eval driver is running or driver.pid exists: {run}")
    info.append("round-2 evaluations finished (heartbeat done, 11 exports, 110 ok evals, 0 failed); no driver running")
    base = json.loads(BASELINE.read_text())
    diff = C.integrity_diff(base, C.integrity())
    if diff:
        fatal.append(f"kit / stock trees differ from integrity/baseline.json: {diff[:5]}")
    info.append(f"kit + stock trees == integrity/baseline.json (taken {base.get('taken_at')}): {not diff}")
    mine = raw_sha(MINE)
    info.append(f"scripts/t2cons_r2_consensus.py sha256 {mine}")
    if stage in ("gate", "score", "select"):
        if not PINS_D.is_file():
            fatal.append("PREREG_R2_PINS_d.md missing: this script must be pinned before it scores anything (5.5)")
        else:
            got, rec = raw_sha(PINS_D), first_token(PINS_D_SHA)
            pins = pin_lines(PINS_D.read_text())
            if got != rec:
                fatal.append(f"PREREG_R2_PINS_d.md {got[:16]} != PREREG_R2_PINS_d.sha256 {rec[:16]}")
            for need in ("scripts/t2cons_r2_consensus.py", "selftest_r2/selftest_r2.json", "inputs_r2/ROLLOUTS_R2.sha256"):
                if need not in [r for r, _ in pins]:
                    fatal.append(f"PREREG_R2_PINS_d.md does not pin {need}")
            if ("scripts/t2cons_r2_consensus.py", mine) not in pins:
                fatal.append(f"PREREG_R2_PINS_d.md does not pin this script's sha256 {mine[:16]}")
            for rel, h0 in pins:
                pp = pin_path(rel)
                h = pin_hash(pp)
                if h != h0:
                    fatal.append(f"PREREG_R2_PINS_d pin {rel}: {h[:16]} != {h0[:16]}")
            info.append(f"PREREG_R2_PINS_d.md {got[:16]} == its sha file; {len(pins)} pins hold; pins this script")
            st = json.loads(SELFTEST.read_text()) if SELFTEST.is_file() else {}
            if st.get("pass") is not True or st.get("script_sha256") != mine:
                fatal.append("selftest_r2/selftest_r2.json is not a PASS of this script")
    if stage in ("score", "select"):
        g = json.loads(GATE_JSON.read_text()) if GATE_JSON.is_file() else {}
        if g.get("pass") is not True or g.get("script_sha256") != mine or g.get("csv_sha256") != (
                raw_sha(GATE_CSV) if GATE_CSV.is_file() else None) or g.get("pins_d_sha256") != (
                raw_sha(PINS_D) if PINS_D.is_file() else None):  # fmt: skip
            fatal.append("scores_r2/own_input_gate_r2.json is not a PASS of this pinned script")
        else:
            info.append(f"own-input gate (5.3): PASS; validation_failed {g.get('validation_failed_jobs')}")
    if stage in ("gate", "score", "select") and SEL_SHA.exists():
        fatal.append("SELECTION_R2.sha256 exists: the round-2 selection is frozen")
    return fatal, info


def require(stage: str) -> bool:
    fatal, info = chain(stage)
    for x in info:
        print("  " + x)
    if fatal:
        print("FATAL:\n  " + "\n  ".join(fatal))
        return False
    return True


# ================================================================================================ jobs / runs
def tracks_r2() -> dict[str, str]:
    return C.hash_list(TRACKS_R2)


def load_jobs() -> dict[str, dict]:
    """The 11 round-2 jobs: manifest_r2b.json x candidates_r2.json x exports_b/<job>/export.json."""
    man = C.read_json(MANIFEST)
    cands = {r["job"]: r for r in C.read_json(CANDS)["new_jobs"]}
    if sorted(cands) != sorted(NEW_JOBS) or sorted(r["job"] for r in man["rows"]) != sorted(NEW_JOBS):
        raise SystemExit("round-2 job set differs from PREREG_R2 2.1 / 2.2")
    if man.get("candidates_r2_sha256") != CANDS_SHA or man.get("prereg_r2_sha256") != PREREG_SHA:
        raise SystemExit("manifest_r2b.json was not built under PREREG_R2 / candidates_r2.json")
    trk = tracks_r2()
    jobs = {}
    for r in man["rows"]:
        job = r["job"]
        e, kind, rid = NEW_JOBS[job]
        c = cands[job]
        if r.get("status") != "ok" or int(r["episode"]) != e or r.get("root_kind") != kind or c.get("input") != kind \
                or int(c["episode"]) != e or r.get("r2_id") != rid or c.get("id") != rid:  # fmt: skip
            raise SystemExit(f"{job}: manifest / candidates row differs from PREREG_R2 ({r.get('status')}, {kind}, {rid})")
        ref = C.INPUT_REF[kind]
        if Path(str(r.get("frame_refs"))) != ref or Path(str(c.get("frame_refs"))) != ref:
            raise SystemExit(f"{job}: frame_refs {r.get('frame_refs')} != the PREREG 3.2 input reference {ref}")
        if r.get("parquet_sha256") != c.get("parquet_sha256"):
            raise SystemExit(f"{job}: training parquet sha256 differs between manifest and candidates")
        npz = ref / f"episode_{e:06d}.npz"
        h = sha(npz)
        if trk.get(str(npz)) != h:
            raise SystemExit(f"{job}: input reference {npz} sha256 {h[:16]} != TRACKS_R2.sha256")
        exp = C.read_json(EXPORTS_B / job / "export.json")
        want = EXPORT_SHA[job]
        if exp.get("status") != "ok" or exp.get("stock_verified") is not True or exp.get("sha256") != want or \
                exp.get("destination_sha256") != want or exp.get("source_sha256") != r.get("training_sha256") or \
                sha(exp["file"]) != want:  # fmt: skip
            raise SystemExit(f"{job}: exports_b export.json / file is not the pinned ok export {want[:16]}")
        jobs[job] = {
            "job": job, "episode": e, "id": rid, "input": kind, "input_reference": str(ref), "object": r.get("object"),
            "owner": r.get("owner"), "family": r.get("family"), "iteration": r.get("iteration"),
            "variants": r.get("variants"), "parquet_sha256": r.get("parquet_sha256"), "export_status": "ok",
            "export_sha256": want, "export_file": exp["file"], "training_sha256": r.get("training_sha256"),
            "relocated": bool(r.get("relocated")), "input_reference_sha256": h,
        }  # fmt: skip
    return jobs


def load_evals(jobs: dict[str, dict]) -> tuple[dict[str, dict[int, dict]], int]:
    """job -> run -> the ok own-input row (evaluator values) + its rollout parquet (runs_b/, AMENDMENTS_R2 A2)."""
    out: dict[str, dict[int, dict]] = {}
    n_not_ok = 0
    for r in C.read_csv(EVALS):
        if r.get("status") != "ok":
            n_not_ok += 1
            continue
        j = jobs.get(r["job"])
        if j is None:
            raise SystemExit(f"own_input_evals_r2b.csv: ok row of unknown job {r['job']}")
        k, t, e = int(r["run"]), int(r["try"]), j["episode"]
        if int(r["episode"]) != e or not 1 <= k <= NRUNS:
            raise SystemExit(f"{r['job']} run {k}: episode {r['episode']} / run out of range")
        pq = RUNS_B / r["job"] / f"run{k}" / f"try{t}" / "results" / f"episode_{e:06d}.parquet"
        if Path(r["results_parquet"]) != pq:
            raise SystemExit(f"{r['job']} run {k}: results_parquet {r['results_parquet']} != {pq}")
        if r.get("export_sha256_full") != j["export_sha256"]:
            raise SystemExit(f"{r['job']} run {k}: export_sha256 {r.get('export_sha256_full')} != export")
        if r.get("reference_parquet_sha256") != j["parquet_sha256"]:
            raise SystemExit(f"{r['job']} run {k}: reference parquet sha256 != the training parquet")
        if k in out.setdefault(r["job"], {}):
            raise SystemExit(f"{r['job']} run {k}: two ok rows")
        out[r["job"]][k] = {"run": k, "try": t, "parquet": str(pq), "gpu": r.get("gpu"),
                            "seed_signature": r.get("seed_signature"), **{m: C.fnum(r[m]) for m in C.METRICS}}  # fmt: skip
    return out, n_not_ok


def load_incumbents() -> dict[int, dict]:
    """PREREG_R2 4: frozen round-1 CONS-B picks, their 5 frozen variant-B consensus AUCs and m_P (not re-scored)."""
    sel = C.read_json(R1 / "selection.json")
    if sel.get("binding_variant") != "B" or sel.get("synthetic") is not False:
        raise SystemExit("round-1 selection.json is not the binding variant-B selection")
    eps = {int(x["episode"]): x for x in sel["episodes"]}
    js = C.read_json(R1 / "scores/consensus_job_stats.json")["jobs"]
    rows = [r for r in C.read_csv(R1 / "scores/consensus_scores.csv") if r["variant"] == VARIANT and r["status"] == "ok"]
    build = C.read_json(R1 / "refs/build.json")
    cands = C.read_json(CANDS)
    out = {}
    for e, (P, m_tab, cl_tab, tracks) in sorted(INCUMBENTS.items()):
        x = eps[e]
        if x["pick_B"] != P or x.get("pick") != P or sel["compositions"]["CONS-B"][ep_key(e)] != P:
            raise SystemExit(f"ep{e:02d}: round-1 CONS-B pick {x['pick_B']} != PREREG_R2 4 incumbent {P}")
        rs = sorted([r for r in rows if r["job"] == P], key=lambda r: int(r["run"]))
        if [int(r["run"]) for r in rs] != list(range(1, NRUNS_R1 + 1)):
            raise SystemExit(f"ep{e:02d}: {P} has variant-B runs {[r['run'] for r in rs]}")
        vals = {m: [float(r[m]) for r in rs] for m in C.METRICS}
        m_P = C.stats(vals["auc"])["mean"]
        V = x["variants"][VARIANT]
        rec_mean = V["default"]["cons_auc_mean"] if P == x["default"] else \
            next(c["cons_auc_mean"] for c in V["challengers"] if c["job"] == P)  # fmt: skip
        if m_P != rec_mean or m_P != js[P][VARIANT]["auc_stats"]["mean"] or vals["auc"] != js[P][VARIANT]["auc"]:
            raise SystemExit(f"ep{e:02d}: m_P {m_P!r} != round-1 selection.json cons_auc_mean {rec_mean!r}")
        ci = cands["challengers_per_episode"][ep_key(e)]["incumbent"]
        if ci["job"] != P or ci["cons_auc_mean"] != m_P:
            raise SystemExit(f"ep{e:02d}: candidates_r2.json incumbent record differs")
        if abs(m_P - m_tab) > 5e-7:
            raise SystemExit(f"ep{e:02d}: m_P {m_P:.6f} != PREREG_R2 4 table {m_tab}")
        cl = int(sel["cons_B_closest_runs"][ep_key(e)]["run"])
        if sel["cons_B_closest_runs"][ep_key(e)]["job"] != P or cl != int(x["closest_run_B"]) or cl != cl_tab or \
                cl != int(js[P][VARIANT]["closest_run"]):  # fmt: skip
            raise SystemExit(f"ep{e:02d}: round-1 CLOSEST run records disagree ({cl}, table {cl_tab})")
        used = tuple(build["variants"][VARIANT]["episodes"][ep_key(e)]["used"])
        if used != tracks:
            raise SystemExit(f"ep{e:02d}: refs/B tracks {used} != PREREG_R2 4 table {tracks}")
        out[e] = {
            "episode": e, "object": x.get("object"), "job": P, "input": x["pick_input"], "cons_auc": vals["auc"],
            "m_P": m_P, "bar": m_P + MARGIN, "cons_auc_var": C.stats(vals["auc"])["var"],
            "cons_mppe_mean": C.stats(vals["mppe"])["mean"], "cons_sp_mean": C.stats(vals["sp_sr"])["mean"],
            "cons_mp_mean": C.stats(vals["mp_sr"])["mean"], "round1_closest_run": cl, "reference_tracks": list(used),
            "round1_reason_B": x.get("reason_B"), "frame_reference": sel["cons_B_closest_runs"][ep_key(e)]["frame_reference"],
            "floor_count_round1": (V["default"]["floor_count"] if P == x["default"]
                                   else next(c["floor_count"] for c in V["challengers"] if c["job"] == P)),
        }  # fmt: skip
    return out


def rollout_files(evals: dict) -> dict[str, str]:
    return {evals[j][k]["parquet"]: j for j in sorted(evals) for k in sorted(evals[j])}


def check_rollouts(evals: dict) -> list[str]:
    want = C.hash_list(ROLLOUTS)
    problems = []
    files = rollout_files(evals)
    if sorted(want) != sorted(files):
        problems.append(f"ROLLOUTS_R2.sha256 lists {len(want)} files, the runs have {len(files)}")
    for p in files:
        if want.get(p) is None:
            problems.append(f"{p} not in ROLLOUTS_R2.sha256")
        elif sha(p) != want[p]:
            problems.append(f"{p} changed since ROLLOUTS_R2.sha256")
    return problems


def complete(jobs: dict, evals: dict) -> list[str]:
    return [j for j in sorted(jobs) if sorted(evals.get(j, {})) == list(range(1, NRUNS + 1))]


# ================================================================================================ rule (PREREG_R2 5.4 / 6)
def select_r2(inc: str, m_P: float, challengers: list[str], cons: dict, own: dict, excluded=(), n_req: int = NRUNS,
              floor: int = FLOOR) -> tuple[str, str, list[dict]]:
    """cons[job] = {"n": scored runs, "auc": [n values in run order], "mppe": [...]}; own[job] = run -> own-input row."""
    rows = []
    for c in sorted(challengers):
        x = cons.get(c, {"n": 0})
        n_own = len(own.get(c, {}))
        fl = C.floor_count(own.get(c, {}))
        why = []
        if c in excluded:
            why.append("validation_failed (PREREG_R2 5.3)")
        if x["n"] != n_req:
            why.append(f"{x['n']}/{n_req} consensus-scored runs")
        if n_own != n_req:
            why.append(f"{n_own}/{n_req} ok own-input runs")
        row = {"job": c, "n_scored": x["n"], "n_own_ok": n_own, "floor_count": fl}
        if x["n"] == n_req:
            st, sm = C.stats(x["auc"]), C.stats(x["mppe"])
            row.update({"cons_auc": list(x["auc"]), "cons_auc_mean": st["mean"], "cons_auc_var": st["var"],
                        "cons_mppe_mean": sm["mean"], "delta_vs_P": st["mean"] - m_P})  # fmt: skip
            if not why:
                if not st["mean"] >= m_P + MARGIN:
                    why.append(f"cons_mean_B {st['mean']:.4f} < m_P {m_P:.4f} + {MARGIN}")
                if not fl >= floor:
                    why.append(f"stability floor {fl}/{n_req} own-input runs with SP-SR = MP-SR = 1 (< {floor})")
        row.update({"qualifies": not why, "why_not": why})
        rows.append(row)
    q = [r for r in rows if r["qualifies"]]
    if not q:
        return inc, "P kept (no qualifying challenger)", rows
    q.sort(key=lambda r: (-r["cons_auc_mean"], r["cons_auc_var"], r["cons_mppe_mean"], r["job"]))
    best = q[0]
    tie = len(q) > 1 and q[1]["cons_auc_mean"] == best["cons_auc_mean"]
    reason = (f"replaced by {best['job']} (cons_mean_B {best['cons_auc_mean']:.4f} >= m_P {m_P:.4f} + {MARGIN}; "
              f"floor {best['floor_count']}/{n_req})" + ("; exact AUC tie broken by variance/MPPE/name" if tie else ""))  # fmt: skip
    return best["job"], reason, rows


def closest_run(aucs: list[float], runs: list[int]) -> int:
    """PREREG_R2 6: the run whose consensus AUC is closest to the job's mean (float64, C.stats); ties: lower k."""
    mu = C.stats(aucs)["mean"]
    return min(zip(runs, aucs), key=lambda t: (abs(t[1] - mu), t[0]))[0]


# ================================================================================================ selftest
def cmd_selftest(a) -> int:
    if SELFTEST.exists():
        print("FATAL: selftest_r2/selftest_r2.json exists (new files only)")
        return 2
    rng = np.random.default_rng(20261027)
    res, problems = [], []

    def check(name, ok, detail=""):
        res.append({"test": name, "ok": bool(ok), "detail": str(detail)[:400]})
        if not ok:
            problems.append(f"{name}: {detail}")

    own_ok = {k: {"sp_sr": 1.0, "mp_sr": 1.0} for k in range(1, NRUNS + 1)}

    def own_n(n_good: int) -> dict:
        return {k: {"sp_sr": 1.0, "mp_sr": 1.0 if k <= n_good else 0.0} for k in range(1, NRUNS + 1)}

    def rec(xs, mppe=1.0):
        return {"n": len(xs), "auc": list(xs), "mppe": [mppe] * len(xs)}

    m_P = C.stats([0.5] * 5)["mean"]
    # exact boundary: find m_eq with m_eq + 0.01 == the challenger's float64 mean, and the next float above it
    edge = [0.61, 0.6, 0.62, 0.605, 0.615, 0.61, 0.6, 0.62, 0.61, 0.61]
    cm = C.stats(edge)["mean"]
    m_eq, cand = None, cm - MARGIN
    for _ in range(64):
        if cand + MARGIN == cm:
            m_eq = cand
            break
        cand = float(np.nextafter(cand, np.inf if cand + MARGIN < cm else -np.inf))
    m_above = None if m_eq is None else float(np.nextafter(m_eq, np.inf))
    while m_above is not None and not m_above + MARGIN > cm:
        m_above = float(np.nextafter(m_above, np.inf))
    check("precondition: an m_P with m_P + 0.01 == the challenger mean exists", m_eq is not None, repr(cm))
    cons = {"c_edge": rec(edge), "c_9runs": rec([0.9] * 9), "c_floor7": rec([0.9] * 10),
            "c_floor8": rec([0.7] * 10), "c_floor10": rec([0.6] * 10), "c_valfail": rec([0.95] * 10),
            "c_own9": rec([0.95] * 10), "c_below": rec([0.5099] * 10)}  # fmt: skip
    own = {"c_edge": own_ok, "c_9runs": own_ok, "c_floor7": own_n(7), "c_floor8": own_n(8), "c_below": own_ok,
           "c_floor10": own_ok, "c_valfail": own_ok, "c_own9": {k: own_ok[k] for k in range(1, 10)}}  # fmt: skip
    if m_eq is not None:
        pick, why, rows = select_r2("P", m_eq, ["c_edge"], cons, own)
        check("cons_mean_B exactly m_P + 0.01 qualifies (>=, exact floats)", pick == "c_edge", f"{pick} {why}")
        pick, why, rows = select_r2("P", m_above, ["c_edge"], cons, own)
        check("cons_mean_B one float below m_P + 0.01 does not qualify", pick == "P" and not rows[0]["qualifies"],
              f"{pick} {rows}")  # fmt: skip
    pick, why, rows = select_r2("P", m_P, ["c_below"], cons, own)
    check("cons_mean_B 0.5099 < 0.5 + 0.01 does not qualify", pick == "P" and not rows[0]["qualifies"], f"{pick}")
    pick, why, rows = select_r2("P", m_P, ["c_9runs", "c_floor7", "c_valfail", "c_own9"], cons, own, excluded={"c_valfail"})
    check("9 scored runs / floor 7 of 10 / validation_failed / 9 own-input runs do not qualify", pick == "P"
          and not any(r["qualifies"] for r in rows), f"{pick} {[(r['job'], r['why_not']) for r in rows]}")  # fmt: skip
    pick, why, rows = select_r2("P", m_P, ["c_floor8", "c_floor10", "c_below", "c_floor7"], cons, own)
    check("floor 8 of 10 qualifies; the highest mean wins", pick == "c_floor8", f"{pick} {why}")
    cons.update({"t_a": {"n": 10, "auc": [0.625] * 10, "mppe": [2.0] * 10},
                 "t_b": {"n": 10, "auc": [0.625] * 10, "mppe": [1.0] * 10},
                 "t_c": {"n": 10, "auc": [0.5, 0.75] + [0.625] * 8, "mppe": [0.5] * 10},
                 "t_d": {"n": 10, "auc": [0.625] * 10, "mppe": [1.0] * 10}})  # fmt: skip
    for j in ("t_a", "t_b", "t_c", "t_d"):
        own[j] = own_ok
    means = {j: C.stats(cons[j]["auc"])["mean"] for j in ("t_a", "t_b", "t_c", "t_d")}
    pick, why, _ = select_r2("P", m_P, ["t_a", "t_b", "t_c", "t_d"], cons, own)
    check("exact ties: lower variance, then lower MPPE, then name", len(set(means.values())) == 1 and pick == "t_b",
          f"{pick} {means}")
    pick, why, _ = select_r2("P", m_P, ["t_d", "t_b"], cons, own)
    check("exact tie on AUC, variance and MPPE: ASCII job name", pick == "t_b", f"{pick}")
    pick, why, rows = select_r2("P", m_P, [], cons, own)
    check("no challenger: P stays", pick == "P" and rows == [], f"{pick}")
    # closest run
    check("CLOSEST: run closest to the mean", closest_run([0.1, 0.5, 0.9, 0.52], [1, 2, 3, 4]) == 2)
    check("CLOSEST: exact tie goes to the lower k", closest_run([0.25, 0.75, 0.25, 0.75], [1, 2, 3, 4]) == 1 and
          closest_run([0.75, 0.25, 0.25, 0.75], [1, 2, 3, 4]) == 1, "")  # fmt: skip
    xs = list(rng.uniform(0.2, 0.9, NRUNS))
    mu = float(np.mean(np.asarray(xs, dtype=np.float64)))
    want = min(range(1, NRUNS + 1), key=lambda k: (abs(xs[k - 1] - mu), k))
    check("CLOSEST: equals an independent re-derivation on random values", closest_run(xs, list(range(1, 11))) == want)
    # equivalence with the pinned 5-run rule
    own5 = {k: {"sp_sr": 1.0, "mp_sr": 1.0} for k in range(1, 6)}
    own5_3 = {k: {"sp_sr": 1.0, "mp_sr": 1.0 if k <= 3 else 0.0} for k in range(1, 6)}
    own5_4 = {k: {"sp_sr": 1.0 if k != 2 else 0.0, "mp_sr": 1.0} for k in range(1, 6)}
    c5 = {"D": rec([0.5] * 5), "c_margin": rec([0.51] * 5), "c_below": rec([0.5099999] * 5), "c_floor3": rec([0.9] * 5),
          "c_floor4": rec([0.6] * 5), "c_4runs": rec([0.9] * 4), "c_exact": rec([0.5 + MARGIN] * 5)}  # fmt: skip
    o5 = {"D": own5, "c_margin": own5, "c_below": own5, "c_floor3": own5_3, "c_floor4": own5_4, "c_4runs": own5,
          "c_exact": own5}  # fmt: skip
    eq_bad = []
    cases = [["c_below", "c_floor3", "c_4runs"], ["c_exact"], ["c_margin", "c_floor4", "c_below"], ["c_margin"],
             ["c_floor3", "c_floor4", "c_exact", "c_margin", "c_below", "c_4runs"]]  # fmt: skip
    for _ in range(300):  # random 5-run episodes
        name = f"r{_}"
        for i in range(3):
            j = f"{name}_{i}"
            base = rng.uniform(0.1, 0.9)
            c5[j] = {"n": int(rng.choice([4, 5, 5, 5])), "auc": [], "mppe": []}
            c5[j]["auc"] = list(np.clip(base + rng.normal(0, 0.02, c5[j]["n"]), 0, 1))
            c5[j]["mppe"] = list(rng.uniform(0.5, 20, c5[j]["n"]))
            o5[j] = {k: {"sp_sr": float(rng.integers(0, 2)) if rng.random() < 0.3 else 1.0, "mp_sr": 1.0}
                     for k in range(1, 6)}  # fmt: skip
        cases.append([f"{name}_{i}" for i in range(3)])
    for chal in cases:
        p1, r1, _f, rows1 = C.select_episode("D", chal, c5, o5)
        p2, r2, rows2 = select_r2("D", C.stats(c5["D"]["auc"])["mean"], chal, c5, o5, n_req=5, floor=4)
        q1 = [(r["job"], r["qualifies"]) for r in rows1]
        q2 = [(r["job"], r["qualifies"]) for r in rows2]
        if p1 != p2 or q1 != q2:
            eq_bad.append(f"{chal}: pinned {p1} {q1} vs select_r2 {p2} {q2}")
    check("select_r2(n_req 5, floor 4) == pinned select_episode on 305 synthetic 5-run episodes", not eq_bad, eq_bad[:3])
    # random 10-run episodes vs an independent re-derivation
    bad = []
    for t in range(300):
        mP = float(rng.uniform(0.1, 0.9))
        chal, cons_r, own_r, ex = [], {}, {}, set()
        for i in range(int(rng.integers(1, 4))):
            j = f"j{t}_{i}"
            n = int(rng.choice([9, 10, 10, 10, 10]))
            cons_r[j] = {"n": n, "auc": list(np.clip(mP + rng.normal(0.01, 0.03) + rng.normal(0, 0.01, n), 0, 1)),
                         "mppe": list(rng.uniform(0.5, 20, n))}  # fmt: skip
            own_r[j] = {k: {"sp_sr": 1.0 if rng.random() > 0.15 else 0.0, "mp_sr": 1.0} for k in range(1, NRUNS + 1)}
            if rng.random() < 0.1:
                ex.add(j)
            chal.append(j)
        p, _, _ = select_r2("P", mP, chal, cons_r, own_r, excluded=ex)
        q = []
        for j in chal:
            if j in ex or cons_r[j]["n"] != 10:
                continue
            m = float(np.mean(np.asarray(cons_r[j]["auc"], dtype=np.float64)))
            fl = sum(1 for r in own_r[j].values() if r["sp_sr"] == 1.0 and r["mp_sr"] == 1.0)
            if m >= mP + 0.01 and fl >= 8:
                q.append((-m, float(np.var(np.asarray(cons_r[j]["auc"], dtype=np.float64))),
                          float(np.mean(np.asarray(cons_r[j]["mppe"], dtype=np.float64))), j))  # fmt: skip
        want = sorted(q)[0][3] if q else "P"
        if p != want:
            bad.append(f"episode {t}: select_r2 {p} != re-derived {want}")
    check("select_r2 == an independent re-derivation of PREREG_R2 5.4 on 300 random 10-run episodes", not bad, bad[:3])
    # extended path guard
    forbidden = ["/x/clean_LOCAL_NOT_FOR_UPLOAD.csv", "/x/Clean/y", "/x/summary_LOCAL_NOT_FOR_UPLOAD.md", "/x/packed_x.csv",
                 "/d/track_2/tier_1_multiview_caption/e.parquet", "/d/video_to_data_challenge/track_2/baseline/x",
                 str(ME / "data/human_motion_data/whole_body/x.parquet"), str(ME / "data/v2d_challenge_data_public/x"),
                 "/z/runs/x/README.txt", "/z/runs/x/ReadMe.md", "/z/logs/train_rl.log", "/z/TRAIN_RL.LOG",
                 str(ROOT / "pack_r2_LOCAL_NOT_FOR_UPLOAD/x"), "/root/autodl-tmp/zuxing.lu/runs/x/tier2_clean.csv"]  # fmt: skip
    refused = []
    for p in forbidden:
        try:
            C.guard(p)
        except PermissionError:
            refused.append(p)
    check("extended guard refuses every forbidden pattern (incl. readme, train_rl.log)", refused == forbidden,
          sorted(set(forbidden) - set(refused)))
    allowed = [str(RUNS_B / "ep15_zxc_g2_62000/run1/try1/results/episode_000015.parquet"),
               str(C.G2_REFS / "episode_000015.npz"), str(C.K3FIX_REFS / "episode_000005.npz"),
               str(R1 / "refs/B/episode_000019.npz"), str(R1 / "scores/consensus_scores.csv"), str(EVALS), str(MANIFEST),
               str(C.T2 / "data/chunk-000/episode_000019.parquet"), str(C.T2 / "mesh/cane/cane.glb")]  # fmt: skip
    passed = []
    for p in allowed:
        try:
            C.guard(p)
            passed.append(p)
        except PermissionError:
            pass
    check("extended guard passes the noisy-only inputs of 5.2-5.4", passed == allowed, sorted(set(allowed) - set(passed)))
    out = {"written_at": C.now(), "script_sha256": raw_sha(MINE), "consensus_script_sha256": CONS_SHA,
           "gate_script_sha256": GATE_SHA, "pass": not problems, "n_tests": len(res), "problems": problems, "tests": res,
           "note": "synthetic values only; no real score was read or computed"}  # fmt: skip
    write_new(SELFTEST, C.dump(out))
    for r in res:
        print(("ok   " if r["ok"] else "FAIL ") + r["test"] + (f"  [{r['detail']}]" if r["detail"] and not r["ok"] else ""))
    print(f"selftest: {len(res)} tests, {len(problems)} problems -> {SELFTEST}")
    return 0 if not problems else 1


# ================================================================================================ check (dry)
def cmd_check(a) -> int:
    lines: list[str] = []

    def p(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    p(f"t2cons_r2_consensus CHECK {C.now()}: no metric, reference or score is computed")
    fatal, info = chain("check")
    for x in info:
        p(f"chain: {x}")
    for x in fatal:
        p(f"chain FAILED: {x}")
    problems = []
    jobs = load_jobs()
    evals, n_not_ok = load_evals(jobs)
    inc = load_incumbents()
    full = complete(jobs, evals)
    p(f"round-2 jobs {len(jobs)}; with 10 ok own-input runs: {len(full)}; non-ok rows in own_input_evals_r2b.csv: {n_not_ok}")
    if len(full) != len(jobs) or n_not_ok:
        problems.append("not every job has 10 ok runs")
    sigs = sorted({r["seed_signature"] for j in evals for r in evals[j].values()})
    p(f"seed signatures over {sum(len(v) for v in evals.values())} runs: {sigs}")
    for e in sorted(inc):
        x = inc[e]
        chal = [j for j in sorted(jobs) if jobs[j]["episode"] == e]
        p(f"ep{e:02d} P {x['job']} [{x['input']}] m_P {x['m_P']!r} bar {x['bar']!r} round-1 CLOSEST {x['round1_closest_run']} "
          f"refs/B {''.join(x['reference_tracks'])}; challengers: "
          + " ".join(f"{j}[{jobs[j]['input']},{jobs[j]['id']}]" for j in chal))  # fmt: skip
        if not chal:
            problems.append(f"ep{e:02d}: no challenger")
    if sorted({j["episode"] for j in jobs.values()}) != sorted(inc):
        problems.append("challenger episodes != incumbent episodes")
    files = rollout_files(evals)
    missing = [f for f in files if not Path(f).is_file()]
    problems += [f"missing rollout {f}" for f in missing]
    if not ROLLOUTS.exists() and not missing and not fatal and not problems:
        write_new(ROLLOUTS, "".join(f"{sha(f)}  {f}\n" for f in sorted(files)))
        p(f"wrote {ROLLOUTS.relative_to(ROOT)}: {len(files)} rollout parquets, sha256 {raw_sha(ROLLOUTS)}")
    elif ROLLOUTS.exists():
        rp = check_rollouts(evals)
        problems += rp
        p(f"{ROLLOUTS.relative_to(ROOT)} exists (sha256 {raw_sha(ROLLOUTS)}): {len(files) - len(rp)}/{len(files)} unchanged")
    for x in problems:
        p(f"problem: {x}")
    ok = not fatal and not problems
    p(f"check: {'OK' if ok else 'PROBLEMS'}")
    write_new(ROOT / "dryrun" / f"r2_consensus_check_{C.stamp()}.txt", "\n".join(lines) + "\n")
    return 0 if ok else 1


# ================================================================================================ gate (5.3)
def run_flags(r: dict) -> list[str]:
    if r.get("status") != "ok":
        return ["problem run"]
    out = []
    if not r["band_ok"]:
        out.append("evaluator AUC outside [min_s AUC_s - 0.02, max_s AUC_s + 0.02]")
    if abs(r["d_mppe"]) > G.GATE["max_abs_d_mppe"]:
        out.append("|dMPPE| > 0.5 cm")
    if r["mine_sp_sr"] != r["eval_sp_sr"]:
        out.append("SP-SR differs from the evaluator")
    if r["mine_mp_sr"] != r["eval_mp_sr"]:
        out.append("MP-SR differs from the evaluator")
    return out


def cmd_gate(a) -> int:
    if not require("gate"):
        return 2
    if GATE_CSV.exists() or GATE_JSON.exists():
        print("FATAL: gate output exists (new files only)")
        return 2
    jobs = load_jobs()
    evals, _ = load_evals(jobs)
    full = complete(jobs, evals)
    rp = check_rollouts(evals)
    if rp:
        print("FATAL: rollouts differ from inputs_r2/ROLLOUTS_R2.sha256:\n  " + "\n  ".join(rp[:20]))
        return 2
    before = C.integrity()
    K = C.load_kit()
    cache: dict = {}
    grids: dict = {}
    rows = []
    for job in sorted(jobs):
        j = jobs[job]
        e = j["episode"]
        if e not in grids:
            grids[e] = C.grid(K, e)
        frames, names = grids[e]
        for k in sorted(evals.get(job, {})):
            rec = evals[job][k]
            row = {"job": job, "episode": e, "run": k, "try": rec["try"], "input": j["input"], "id": j["id"]}
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
        print(f"  {job}: {sum(1 for r in rows if r['job'] == job and r['status'] == 'ok')}/{len(evals.get(job, {}))} "
              f"scored against its own input; flagged {sum(1 for r in rows if r['job'] == job and r['flags'])}", flush=True)  # fmt: skip
    after = C.integrity()
    diff = C.integrity_diff(before, after) + C.integrity_diff(json.loads(BASELINE.read_text()), after)
    flagged = sorted({r["job"] for r in rows if r["flags"]})
    incomplete = sorted(set(jobs) - set(full))
    rest = [r for r in rows if r["job"] not in flagged]
    passed = (not diff) and all(not r["flags"] for r in rest)
    ok = [r for r in rows if r["status"] == "ok"]
    d_mppe = max((abs(r["d_mppe"]) for r in ok), default=None)
    parts_round1_style = {  # report only (round-1 A1.3 gate-level fractions); PREREG_R2 5.3 is per run
        "auc_band_all_runs": bool(ok) and all(r["band_ok"] for r in ok), "max_abs_d_mppe": d_mppe,
        "sp_equal_frac": sum(r["mine_sp_sr"] == r["eval_sp_sr"] for r in ok) / len(ok) if ok else 0.0,
        "mp_equal_frac": sum(r["mine_mp_sr"] == r["eval_mp_sr"] for r in ok) / len(ok) if ok else 0.0,
        "n_problem_runs": len(rows) - len(ok),
    }  # fmt: skip
    by_job = {job: {"input": jobs[job]["input"], "n_runs": sum(1 for r in rows if r["job"] == job),
                    "n_scored": sum(1 for r in ok if r["job"] == job),
                    "n_inside_seed_range": sum(1 for r in ok if r["job"] == job and r["inside_range"]),
                    "max_excess_outside_range": max((r["excess_outside_range"] for r in ok if r["job"] == job), default=None),
                    "max_abs_d_mppe": max((abs(r["d_mppe"]) for r in ok if r["job"] == job), default=None),
                    "flagged_runs": [r["run"] for r in rows if r["job"] == job and r["flags"]]}
              for job in sorted(jobs)}  # fmt: skip
    cols = ["job", "episode", "run", "try", "input", "id", "status", "problem", "flags", "eval_auc"] + \
        [f"auc_s{s}" for s in G.SEEDS] + \
        ["seed_min", "seed_max", "seed_range", "excess_outside_range", "band_ok", "inside_range", "d_auc_seed0",
         "eval_mppe", "mine_mppe", "d_mppe", "eval_sp_sr", "mine_sp_sr", "eval_mp_sr", "mine_mp_sr",
         "conv_pos_residual_m", "conv_ang_residual_rad"]  # fmt: skip
    write_new(GATE_CSV, C.csv_text(rows, cols))
    out = {
        "written_at": C.now(), "script_sha256": raw_sha(MINE), "consensus_script_sha256": CONS_SHA,
        "gate_script_sha256": GATE_SHA, "prereg_r2_sha256": PREREG_SHA, "pins_d_sha256": raw_sha(PINS_D),
        "rule": "PREREG_R2 5.3: a run is flagged iff the evaluator AUC is outside [min_s AUC_s - 0.02, max_s AUC_s + 0.02] "
                "(s = 0..9), |dMPPE| > 0.5 cm, SP-SR or MP-SR differs from the evaluator, or it is a problem run; the "
                "gate passes iff no run is flagged; otherwise every job with a flagged run is validation_failed and the "
                "gate is re-evaluated without it; a kit / stock-runtime change fails the gate",
        "thresholds": G.GATE, "pass": passed, "validation_failed_jobs": flagged if passed else [],
        "flagged_jobs": flagged, "incomplete_jobs": incomplete, "n_jobs": len(jobs), "n_runs": len(rows),
        "n_scored": len(ok), "n_flagged_runs": sum(1 for r in rows if r["flags"]),
        "flagged_runs": [f"{r['job']} r{r['run']}: {r['flags']}" for r in rows if r["flags"]],
        "n_inside_seed_range": sum(r["inside_range"] for r in ok),
        "max_excess_outside_range": max((r["excess_outside_range"] for r in ok), default=None),
        "max_seed_range": max((r["seed_range"] for r in ok), default=None),
        "max_abs_d_auc_seed0": max((abs(r["d_auc_seed0"]) for r in ok), default=None),
        "max_conv_pos_residual_m": max((r["conv_pos_residual_m"] for r in ok), default=None),
        "max_conv_ang_residual_rad": max((r["conv_ang_residual_rad"] for r in ok), default=None),
        "parts_round1_style_report_only": parts_round1_style, "by_job": by_job,
        "kit_stock_unchanged": not diff, "integrity_diff": diff[:50],
        "vertex_meshes": {n: {"file": str(v[1]), "sha256": sha(v[1])} for n, v in sorted(K.verts.items())},
        "rollouts_list_sha256": raw_sha(ROLLOUTS), "csv_sha256": raw_sha(GATE_CSV),
        "note": "own-input check only (each run vs its OWN input track); seeds 1..9 used only here; no consensus "
                "reference is read; nothing is selected",
    }  # fmt: skip
    write_new(GATE_JSON, C.dump(out))
    print(json.dumps({k: out[k] for k in ("pass", "validation_failed_jobs", "flagged_jobs", "n_runs", "n_scored",
                                          "n_flagged_runs", "n_inside_seed_range", "max_excess_outside_range",
                                          "max_seed_range", "max_abs_d_auc_seed0", "parts_round1_style_report_only",
                                          "kit_stock_unchanged")}, indent=1))  # fmt: skip
    for x in out["flagged_runs"][:30]:
        print("  flagged:", x)
    return 0 if passed else 1


# ================================================================================================ score (5.2)
SCORE_COLS = ["job", "episode", "run", "try", "variant", "input", "id", "status", "problem", "auc", "sp_sr", "mp_sr", "mppe",
              "conv_pos_residual_m", "conv_ang_residual_rad", "checkpoint_sha256", "parquet", "parquet_sha256"]  # fmt: skip


def cmd_score(a) -> int:
    if not require("score"):
        return 2
    if SCORES_CSV.exists() or JSTATS.exists():
        print("FATAL: score output exists (scores are computed once)")
        return 2
    g = C.read_json(GATE_JSON)
    excluded = list(g.get("validation_failed_jobs") or [])
    jobs = load_jobs()
    evals, _ = load_evals(jobs)
    full = complete(jobs, evals)
    rp = check_rollouts(evals)
    if rp:
        print("FATAL: rollouts differ from inputs_r2/ROLLOUTS_R2.sha256:\n  " + "\n  ".join(rp[:20]))
        return 2
    roll = C.hash_list(ROLLOUTS)
    before = C.integrity()
    K = C.load_kit()
    refs = C.load_refs(K)  # pinned: round-1 refs/build.json (ok), each output re-hashed; never rebuilt (5.1)
    trk = tracks_r2()
    r1f = C.hash_list(ROOT / "inputs_r2/ROUND1_FROZEN.sha256")
    for e in sorted({j["episode"] for j in jobs.values()}):
        f = R1 / "refs" / VARIANT / f"episode_{e:06d}.npz"
        h = sha(f)
        if trk.get(str(f)) != h or r1f.get(str(f)) != h:
            print(f"FATAL: {f} sha256 {h[:16]} != TRACKS_R2 / ROUND1_FROZEN")
            return 2
    rows, problems, grids = [], [], {}
    scored_jobs = [j for j in sorted(jobs) if j not in excluded]
    for job in scored_jobs:
        j = jobs[job]
        e = j["episode"]
        if e not in grids:
            grids[e] = C.grid(K, e)
        frames, names = grids[e]
        for k in sorted(evals.get(job, {})):
            rec = evals[job][k]
            base = {"job": job, "episode": e, "run": k, "try": rec["try"], "variant": VARIANT, "input": j["input"],
                    "id": j["id"], "checkpoint_sha256": j["export_sha256"], "parquet": rec["parquet"],
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
        print(f"  {job}: {sum(1 for r in rows if r['job'] == job and r['status'] == 'ok')} runs scored against refs/B",
              flush=True)  # fmt: skip
    write_new(SCORES_CSV, C.csv_text(rows, SCORE_COLS))
    jstats = {}
    for job in scored_jobs:
        rs = sorted([r for r in rows if r["job"] == job and r["status"] == "ok"], key=lambda r: r["run"])
        d = {"episode": jobs[job]["episode"], "input": jobs[job]["input"], "id": jobs[job]["id"], "n": len(rs),
             "runs": [r["run"] for r in rs]}  # fmt: skip
        for m in C.METRICS:
            d[m] = [r[m] for r in rs]
            if rs:
                d[m + "_stats"] = C.stats(d[m])
        if rs:
            d["closest_run"] = closest_run(d["auc"], d["runs"])
        jstats[job] = d
    after = C.integrity()
    diff = C.integrity_diff(before, after)
    out = {"written_at": C.now(), "script_sha256": raw_sha(MINE), "variant": VARIANT, "prereg_r2_sha256": PREREG_SHA,
           "pins_d_sha256": raw_sha(PINS_D), "gate_json_sha256": raw_sha(GATE_JSON),
           "refs_build_sha256": raw_sha(R1 / "refs/build.json"), "rollouts_list_sha256": raw_sha(ROLLOUTS),
           "scores_csv_sha256": raw_sha(SCORES_CSV), "validation_failed_jobs": excluded, "scored_jobs": scored_jobs,
           "complete_jobs": full, "n_rows": len(rows), "n_ok_rows": sum(r["status"] == "ok" for r in rows),
           "problems": problems, "integrity_kit_stock_unchanged": not diff, "integrity_diff": diff[:50],
           "jobs": jstats}  # fmt: skip
    write_new(JSTATS, C.dump(out))
    print(f"scored {len(scored_jobs)} jobs against refs/{VARIANT}: {out['n_ok_rows']}/{len(rows)} ok rows; problems "
          f"{len(problems)}; kit/stock unchanged {not diff}")  # fmt: skip
    for x in problems[:20]:
        print("  problem:", x)
    return 0 if not diff else 1


# ================================================================================================ select (5.4 / 6)
def cmd_select(a) -> int:
    if SEL_SHA.exists():
        print("FATAL: SELECTION_R2.sha256 exists; the round-2 selection is frozen")
        return 2
    if not require("select"):
        return 2
    js = C.read_json(JSTATS)
    g = C.read_json(GATE_JSON)
    if js.get("scores_csv_sha256") != raw_sha(SCORES_CSV) or js.get("script_sha256") != raw_sha(MINE) or \
            js.get("gate_json_sha256") != raw_sha(GATE_JSON) or not js.get("integrity_kit_stock_unchanged") or \
            js.get("validation_failed_jobs") != (g.get("validation_failed_jobs") or []) or \
            js.get("pins_d_sha256") != raw_sha(PINS_D):  # fmt: skip
        print("FATAL: scores changed since scoring, scored by another script, or integrity flagged")
        return 2
    excluded = list(g.get("validation_failed_jobs") or [])
    jobs = load_jobs()
    evals, _ = load_evals(jobs)
    rp = check_rollouts(evals)
    if rp:
        print("FATAL: rollouts changed:\n  " + "\n  ".join(rp[:20]))
        return 2
    inc = load_incumbents()
    jst = js["jobs"]
    r1sel = C.read_json(R1 / "selection.json")
    r1js = C.read_json(R1 / "scores/consensus_job_stats.json")["jobs"]
    own = {j: evals.get(j, {}) for j in jobs}
    out_eps, table, longrows, lines = [], [], [], []
    for e in sorted(inc):
        I = inc[e]
        chal = [j for j in sorted(jobs) if jobs[j]["episode"] == e]
        cons = {j: {"n": jst[j]["n"], "auc": jst[j]["auc"], "mppe": jst[j]["mppe"]} for j in chal if j in jst}
        pick, reason, rows = select_r2(I["job"], I["m_P"], chal, cons, own, excluded=set(excluded))
        changed = pick != I["job"]
        cl = int(jst[pick]["closest_run"]) if changed else I["round1_closest_run"]
        ch_rows = []
        for r in rows:
            j = r["job"]
            s = jst.get(j, {})
            ch_rows.append({**r, "id": jobs[j]["id"], "input": jobs[j]["input"],
                            "nomination": "clean-nominated copy (PREREG_R2 8.1)" if jobs[j]["id"].startswith("C") else "rule Z",
                            "cons_sp_mean": s.get("sp_sr_stats", {}).get("mean"), "cons_mp_mean": s.get("mp_sr_stats", {}).get("mean"),
                            "closest_run": s.get("closest_run"),
                            "own_auc": [own[j][k]["auc"] for k in sorted(own[j])],
                            "own_auc_mean": float(np.mean([own[j][k]["auc"] for k in sorted(own[j])])) if own[j] else None,
                            "own_mppe_mean": float(np.mean([own[j][k]["mppe"] for k in sorted(own[j])])) if own[j] else None,
                            "validation_failed": j in excluded})  # fmt: skip
        rec = {"episode": e, "object": I["object"], "incumbent": I, "challengers": ch_rows, "pick": pick,
               "pick_input": jobs[pick]["input"] if changed else I["input"], "changed": changed, "reason": reason,
               "r2_closest_run": cl, "r2_closest_source": "round2 runs_b" if changed else "round-1 CONS-B CLOSEST"}  # fmt: skip
        out_eps.append(rec)
        for c in ch_rows:
            longrows.append({"episode": ep_key(e), "job": c["job"], "id": c["id"], "input": c["input"],
                             "n_scored": c["n_scored"], **{f"cons_auc_r{k}": (c["cons_auc"][k - 1] if c.get("cons_auc") else None)
                                                         for k in range(1, NRUNS + 1)},
                             "cons_auc_mean": c.get("cons_auc_mean"), "cons_auc_sd": math.sqrt(c["cons_auc_var"]) if "cons_auc_var" in c else None,
                             "cons_mppe_mean": c.get("cons_mppe_mean"), "cons_sp_mean": c.get("cons_sp_mean"),
                             "cons_mp_mean": c.get("cons_mp_mean"), "m_P": I["m_P"], "delta_vs_P": c.get("delta_vs_P"),
                             "floor_count": c["floor_count"], "n_own_ok": c["n_own_ok"], "own_auc_mean": c["own_auc_mean"],
                             "qualifies": c["qualifies"], "why_not": "; ".join(c["why_not"]), "picked": c["job"] == pick,
                             "closest_run": c.get("closest_run")})  # fmt: skip
        best = max([c for c in ch_rows if "cons_auc_mean" in c], key=lambda c: c["cons_auc_mean"], default=None)
        table.append({
            "episode": ep_key(e), "object": I["object"], "incumbent": I["job"], "incumbent_input": I["input"],
            "m_P": I["m_P"], "bar": I["bar"], "n_challengers": len(ch_rows),
            "challengers": "; ".join(f"{c['job']} [{c['id']}] {fmt(c.get('cons_auc_mean'))} ({fmt(c.get('delta_vs_P'), '+.4f')}) "
                                     f"floor {c['floor_count']}/{NRUNS} {'QUALIFIES' if c['qualifies'] else 'no'}"
                                     for c in ch_rows),
            "best_challenger": best["job"] if best else "", "best_cons_mean_B": best["cons_auc_mean"] if best else None,
            "pick": pick, "changed": changed, "r2_closest_run": cl, "reason": reason,
        })  # fmt: skip
        lines.append(f"ep{e:02d} P {I['job']} [{I['input']}] m_P {I['m_P']:.6f} (bar {I['bar']:.6f}; 5 frozen runs) -> "
                     f"{pick} ({reason}); R2-CLOSEST run {cl}")  # fmt: skip
        for c in ch_rows:
            lines.append(f"      {c['job']} [{c['id']}, {c['input']}] cons-B {fmt(c.get('cons_auc_mean'), '.6f')} sd "
                         f"{fmt(math.sqrt(c['cons_auc_var'])) if 'cons_auc_var' in c else ''} delta "
                         f"{fmt(c.get('delta_vs_P'), '+.6f')} floor {c['floor_count']}/{NRUNS} qualifies={c['qualifies']} "
                         f"{c['why_not']}")  # fmt: skip
    # the R2 composition and the R2-CLOSEST file (all 30 episodes)
    by_ep = {x["episode"]: x for x in out_eps}
    r1eps = {int(x["episode"]): x for x in r1sel["episodes"]}
    closest_file, comp_r1, comp_r2 = {}, {}, {}
    for e in range(30):
        r1pick = r1eps[e]["pick_B"]
        r1cl = r1sel["cons_B_closest_runs"][ep_key(e)]
        comp_r1[ep_key(e)] = r1pick
        if e in by_ep and by_ep[e]["changed"]:
            x = by_ep[e]
            j = jobs[x["pick"]]
            comp_r2[ep_key(e)] = x["pick"]
            closest_file[ep_key(e)] = {"job": x["pick"], "run": x["r2_closest_run"], "source": "round2",
                                       "input": j["input"], "frame_reference": j["input_reference"],
                                       "export_sha256": j["export_sha256"],
                                       "rollout": evals[x["pick"]][x["r2_closest_run"]]["parquet"]}  # fmt: skip
        else:
            comp_r2[ep_key(e)] = r1pick
            closest_file[ep_key(e)] = {"job": r1pick, "run": int(r1cl["run"]), "source": "round1",
                                       "input": r1cl["input"], "frame_reference": r1cl["frame_reference"]}  # fmt: skip

    def comp_avg(mapping: dict) -> dict:
        d = {}
        for m in C.METRICS:
            per = []
            for e in range(30):
                jb = mapping[ep_key(e)]
                src = jst[jb] if jb in jobs else r1js[jb][VARIANT]  # round-2 job: 10 runs; round-1 job: 5 frozen
                per.append(float(np.mean(np.asarray(src[m], dtype=np.float64))))
            d[m] = {"average": float(np.mean(per)), "per_episode": {ep_key(e): per[e] for e in range(30)}}
        return d

    summ = {"ROUND1_CONS-B": {"jobs": comp_r1, "cons_B": comp_avg(comp_r1)},
            "ROUND2": {"jobs": comp_r2, "cons_B": comp_avg(comp_r2)}}
    changed = [ep_key(x["episode"]) for x in out_eps if x["changed"]]
    gen = C.now()
    hashes = {"prereg_r2_sha256": PREREG_SHA, **{f"{n}_sha256": raw_sha(ROOT / n) for n, _, _ in CHAIN_DOCS},
              "pins_d_sha256": raw_sha(PINS_D), "script_sha256": raw_sha(MINE), "consensus_script_sha256": CONS_SHA,
              "gate_script_sha256": GATE_SHA, "candidates_r2_sha256": CANDS_SHA, "manifest_r2b_sha256": raw_sha(MANIFEST),
              "own_input_evals_r2b_sha256": raw_sha(EVALS), "rollouts_list_sha256": raw_sha(ROLLOUTS),
              "gate_json_sha256": raw_sha(GATE_JSON), "consensus_scores_r2_sha256": raw_sha(SCORES_CSV),
              "consensus_job_stats_r2_sha256": raw_sha(JSTATS), "round1_selection_json_sha256": R1_FILES["selection.json"],
              "round1_consensus_scores_sha256": R1_FILES["scores/consensus_scores.csv"],
              "round1_refs_build_sha256": R1_FILES["refs/build.json"]}  # fmt: skip
    sel = {"generated_at": gen, "synthetic": False, **hashes, "rule": RULE, "binding_variant": VARIANT,
           "noisy_only": True, "validation_failed_jobs": excluded, "changed_episodes": changed,
           "episodes": out_eps, "r2_closest_file": closest_file,
           "compositions": {"ROUND1_CONS-B": comp_r1, "ROUND2": comp_r2},
           "upload_eligibility": "PREREG_R2 6: the R2-CLOSEST file is the only upload-eligible Tier-2 file of this "
                                 "pipeline and supersedes round 1's CONS-B CLOSEST file; the user decides any upload"}  # fmt: skip
    written = []

    def w(name: str, text: str) -> None:
        write_new(ROOT / name, text)
        written.append(name)

    w("selection_r2.json", C.dump(sel))
    head = (f"selection_r2 {gen} CST (noisy-only consensus, variant B binding; refs = round-1 refs/B, frozen; scores "
            f"{hashes['consensus_scores_r2_sha256'][:12]}; validation_failed {excluded or 'none'})")  # fmt: skip
    w("selection_r2.txt", "\n".join([head] + lines + ["", f"changed episodes: {', '.join(changed) or 'none'}",
                                                      "R2-CLOSEST runs: " + " ".join(f"{k}:{v['job']}#{v['run']}"
                                                                                     for k, v in closest_file.items())]) + "\n")  # fmt: skip
    w("selection_r2_table.csv", C.csv_text(table, list(table[0].keys())))
    w("selection_r2_candidates.csv", C.csv_text(longrows, list(longrows[0].keys())))
    md = ["# t2_consensus_r2_20261027: round-2 selection (noisy-only; PREREG_R2 5.4 / 6 under AMENDMENTS_R2/_b/_c)", "",
          f"Generated {gen} CST by `scripts/t2cons_r2_consensus.py` ({hashes['script_sha256'][:16]}...), pinned in "
          f"`PREREG_R2_PINS_d.md` ({hashes['pins_d_sha256'][:16]}...). Variant B binding: every run scored against round "
          "1's frozen `refs/B`. The incumbent is the frozen round-1 CONS-B pick (5 frozen runs, not re-scored). A "
          "challenger replaces it iff its 10-run consensus-B mean >= m_P + 0.01 and its own-input SP-SR = MP-SR = 1 in >= 8 "
          "of 10 runs. No clean or LOCAL file was opened.", "",
          f"Changed episodes: {', '.join(changed) or 'none'}. validation_failed: {', '.join(excluded) or 'none'}.", "",
          "| ep | object | incumbent P (input) | m_P | bar | challengers: cons-B mean (delta), floor, qualifies | **pick** | "
          "R2-CLOSEST run | reason |", "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for t in table:
        md.append(f"| {t['episode']} | {t['object']} | {t['incumbent']} ({t['incumbent_input']}) | {t['m_P']:.4f} | "
                  f"{t['bar']:.4f} | {t['challengers'].replace('; ', '<br>')} | **{t['pick']}** | {t['r2_closest_run']} | "
                  f"{t['reason']} |")  # fmt: skip
    md += ["", "Other 21 episodes: no challenger; the round-1 CONS-B pick and its CLOSEST run are kept unchanged.", "",
           "R2-CLOSEST file (job # run): " + ", ".join(f"{k} {v['job']}#{v['run']}" for k, v in closest_file.items())]  # fmt: skip
    w("selection_r2_table.md", "\n".join(md) + "\n")
    smd = ["# t2_consensus_r2_20261027: composition summary (noisy-only, round-1 refs/B)", "",
           "Average over the 30 episodes of each episode's per-job mean consensus value under the round-1 B references "
           "(5 frozen runs for round-1 jobs, 10 runs for round-2 jobs). Consensus numbers are not leaderboard estimates "
           "and favour the selected composition by construction (round-1 PREREG 9.6).", "",
           "| composition | cons-B AUC | cons-B SP-SR | cons-B MP-SR | cons-B MPPE |", "|---|---|---|---|---|"]  # fmt: skip
    for name, d in summ.items():
        c = d["cons_B"]
        smd.append(f"| {name} | {fmt(c['auc']['average'])} | {fmt(c['sp_sr']['average'])} | {fmt(c['mp_sr']['average'])} | "
                   f"{fmt(c['mppe']['average'])} |")  # fmt: skip
    w("summary_r2_consensus.json", C.dump({"generated_at": gen, "binding_variant": VARIANT, "compositions": summ}))
    w("summary_r2_consensus.md", "\n".join(smd) + "\n")
    frozen_inputs = ["scores_r2/consensus_scores_r2.csv", "scores_r2/consensus_job_stats_r2.json",
                     "scores_r2/own_input_gate_r2.csv", "scores_r2/own_input_gate_r2.json", "selftest_r2/selftest_r2.json",
                     "inputs_r2/ROLLOUTS_R2.sha256", "own_input_evals_r2b.csv", "manifest_r2b.json",
                     "inputs_r2/candidates_r2.json", "PREREG_R2_PINS_d.md", "PREREG_R2_PINS_c.md", "AMENDMENTS_R2_c.md",
                     "AMENDMENTS_R2_b.md", "PREREG_R2_PINS_b.md", "AMENDMENTS_R2.md", "PREREG_R2_PINS.md", "PREREG_R2.md"]  # fmt: skip
    body = "".join(f"{raw_sha(ROOT / f)}  {f}\n" for f in written + frozen_inputs)
    write_new(SEL_SHA, body)  # LAST: from here the round-2 selection is frozen
    print("\n".join(smd))
    print()
    print("\n".join(md[6:]))
    print(f"\nSELECTION_R2.sha256 written {C.now()} (frozen): {raw_sha(SEL_SHA)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("selftest", "check", "gate", "score", "select"))
    a = ap.parse_args()
    if not sys.flags.dont_write_bytecode or not sys.flags.isolated:
        print("run with the kit python -I -B")
        return 2
    return {"selftest": cmd_selftest, "check": cmd_check, "gate": cmd_gate, "score": cmd_score,
            "select": cmd_select}[a.cmd](a)  # fmt: skip


if __name__ == "__main__":
    raise SystemExit(main())
