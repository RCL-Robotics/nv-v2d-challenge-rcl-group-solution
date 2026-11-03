#!/usr/bin/env python3
"""t1topup.py: combine, pack, score, decide (PREREG v2 8.5), select (8.6) for the T1improve 20-trial top-up.

Rules: TOPUP_RULES.md (hash-locked in TOPUP_RULES.sha256). Reads fc_runs/T1improve/confirm (C) and
fc_runs/submissions/t1_best20_20261026 (B20) only; writes only under fc_runs/T1improve/topup (U). Never signals a
process, never touches the kit (kit Python always runs with -B and PYTHONDONTWRITEBYTECODE=1), uploads nothing.

  python -I -B t1topup.py pack --config pinned|all7|final [--workers 4]
  python -I -B t1topup.py analyze          # checks + 8.5 decisions -> decisions.json
  python -I -B t1topup.py select           # 8.6 closest-to-mean + report-only best trial -> selection.json, SUMMARY.md
  python -I -B t1topup.py files --commit-url URL --commit-check commit_check.json   # re-pack the two trials, validate
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
import re  # noqa: E402
import shlex  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
FC = ME / "fc_runs"
T1I = FC / "T1improve"
U = T1I / "topup"
C = T1I / "confirm"
B20 = FC / "submissions/t1_best20_20261026"
KIT = ME / "v2d_submission_kit_src/v2d_submission_kit"
DATASET = ME / "data/video_to_data_challenge"
PACK_PYTHON = ME / "conda_envs/v2dkit/bin/python"
PACKER = KIT / "tools/pack_submission.py"
EVAL_SCRIPT = KIT / "flash_chord/scripts/evaluate_policy.py"
SCORER = B20 / "t1best20_score.py"
VALIDATOR = U / "t1topup_validate.py"
RULES, RULES_SHA = U / "TOPUP_RULES.md", U / "TOPUP_RULES.sha256"
PINS = {  # file: sha256 (t1best20 PREREG section 3 + TOPUP_RULES)
    PACKER: "47a4589574ff8284ba4376ef31ac5548c8fe87ead753be2d3b43d6c87041e5be",
    KIT / "KIT_MANIFEST.json": "398947a7783490245a47d70ec68d6894ecef7636e712532710ec448dc0ca1c3e",
    KIT / "config/tracks.json": "5799f424a4cd5453f8a6cfceaeff76e7cc7f8a78ba7d22132108827b1c2a09c7",
    KIT / "metric_code/track_2/AUC.py": "58b437697c32c86497ff5a306fe586014c3429b0d2c18e1d3094a6f503479339",
    KIT / "metric_code/track_2/SP-SR.py": "d206042872acd1c1de6120f96966272495fbbd71a50315f49f5192fea1aaa5bf",
    KIT / "metric_code/track_2/MP-SR.py": "a1d72216eb7745a13ed7db6a55b9efb31492b35149113181a6c42ffc231fff11",
    KIT / "metric_code/track_2/MPPE.py": "41815bc1e8870fb377036750976dcf135818e029f06096ccef5177090fe5e033",
    EVAL_SCRIPT: "160d24f5fbeb70ccd569ca4192056f1892b2a9f2999066db9d8279461648fdf1",
    SCORER: "baddf438a94545ffa14401a5c0300fe207c72255bdc3d3e06d48556e8aae93b9",
    C / "CONFIRM_RULES.md": "655578198ed51bbba20f6eded5bc810fd8e20165c8a0dfc1e516c5e29483fdd4",
}
PLACEHOLDER = "https://github.com/local-placeholder/not-uploaded/commit/0000000000000000000000000000000000000000"
TRIALS = list(range(1, 21))
EPISODES = list(range(30))
# changed item (bar row) -> (winner, episode, pinned evaluated-file sha256 from EXPORT_PINS.md)
CHANGED = {
    "ep01": ("ep01-Bc_162000", 1, "01c0a3bd65688b05f1f0fcde4818dac70f9dab3ec49e9d3cc86038d34ee422fd"),
    "ep05": ("ep05-B3_62000", 5, "ed7bca1616872a5867363e665e0da86217018c72c265f819bf434a3d0983d63b"),
    "ep10": ("ep10-BT_58000", 10, "725ee5e8f8dde84f3a954722b4b7aa6e8e32b5ade6b3b65fdb7a5c899bb3b919"),
    "ep14": ("ep14-R_92000", 14, "b01866a8f9103af18c00b7c2e99fdac2ecb2f462adc1fa1eaa3d56cfa619a364"),
    "ep15": ("ep15-B2_40000", 15, "8d0e8023a9a95b934910fda2a3abd30c718f0409951e24a44865c35526a04dd6"),
    "ep19": ("ep19-Bc_135000", 19, "07863de3bb9b8282d4f8209c1ffe273ef7f596d56f3ddb0afd50b4d0fbbc34c7"),
    "ep28alt": ("ep28-W3H_83000", 28, "497ae3b3a785fc9283fcdda2c019020608ed8c41472f4229ac780f4450081375"),
}  # fmt: skip
EP28_EXCEPTION = {"ep28alt"}  # admitted by 8.4(c): needs MP >= 16/20 (0.8 n)
METRICS = ("AUC", "SP-SR", "MP-SR", "MPPE")
MPPE_WORSE = 0.2


# ------------------------------------------------------------------------------------------------ helpers
def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def inside(path: Path) -> Path:
    p = Path(path).resolve()
    if U.resolve() not in p.parents and p != U.resolve():
        raise SystemExit(f"refusing to write outside {U}: {p}")
    return p


def write_atomic(path: Path, text: str) -> None:
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def dump(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=False, default=str) + "\n"


def write_csv(path: Path, rows: list[dict], cols: list[str], delimiter: str = ",") -> None:
    path = inside(path)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter=delimiter, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)


def log(msg: str) -> None:
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with inside(U / "t1topup.log").open("a") as f:
        f.write(line + "\n")


def kit_env() -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               TMPDIR=str(U / "tmp"))
    return env


def preflight() -> None:
    bad = []
    if not RULES_SHA.is_file() or sha256(RULES) != RULES_SHA.read_text().split()[0]:
        bad.append("TOPUP_RULES.md does not match TOPUP_RULES.sha256")
    for path, want in PINS.items():
        got = sha256(path) if Path(path).is_file() else "missing"
        if got != want:
            bad.append(f"{path}: sha256 {got[:16]} != pinned {want[:16]}")
    if bad:
        raise SystemExit("preflight FAILED:\n  " + "\n  ".join(bad))
    (U / "tmp").mkdir(exist_ok=True)


def mstats(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    return {"n": len(vals), "mean": statistics.fmean(vals), "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals)}


# ------------------------------------------------------------------------------------------------ sources
def b20_item(ep: int) -> str:
    return "ep28alt" if ep == 28 else f"ep{ep:02d}"


def run_record(kind: str, name: str, k: int) -> dict | None:
    base = (C if kind == "C" else B20) / "runs" / name / f"t{k:02d}" / "run.json"
    return read_json(base)


def check_rec(rec: dict | None, ep: int, want_sha: str | None) -> list[str]:
    if rec is None:
        return ["no run.json"]
    p = []
    if rec.get("status") != "ok":
        p.append(f"status {rec.get('status')}")
    if int(rec.get("episode", -1)) != ep:
        p.append(f"episode {rec.get('episode')} != {ep}")
    if want_sha and rec.get("pinned_sha256") != want_sha:
        p.append(f"pinned {str(rec.get('pinned_sha256'))[:12]} != {want_sha[:12]}")
    if rec.get("checkpoint_sha256") != rec.get("pinned_sha256"):
        p.append("results checkpoint_sha256 != pinned file")
    for key, shakey in (("results_json", "json_sha256"), ("results_parquet", "parquet_sha256")):
        f = rec.get(key)
        if not f or not Path(f).is_file():
            p.append(f"{key} missing")
        elif sha256(f) != rec.get(shakey):
            p.append(f"{key} sha256 changed")
    return p


def kept_items() -> dict[str, bool]:
    d = read_json(U / "decisions.json")
    if not d:
        raise SystemExit("decisions.json missing: run analyze first")
    return {item: bool(x["keep_new"]) for item, x in d["items"].items()}


def sources(config: str, k: int) -> tuple[dict[int, dict], list[str]]:
    """episode -> {kind, name, rec}; problems."""
    if config == "pinned":
        use = {}
    elif config == "all7":
        use = {v[1]: (item, v[0], v[2]) for item, v in CHANGED.items()}
    elif config == "final":
        keep = kept_items()
        use = {v[1]: (item, v[0], v[2]) for item, v in CHANGED.items() if keep[item]}
    else:
        raise SystemExit(f"unknown config {config}")
    out, problems = {}, []
    for ep in EPISODES:
        if ep in use:
            item, name, want = use[ep]
            rec = run_record("C", name, k)
            kind = "C"
        else:
            name, want = b20_item(ep), None
            rec = run_record("B20", name, k)
            kind = "B20"
        p = check_rec(rec, ep, want)
        if p:
            problems.append(f"ep{ep:02d} {kind}:{name} t{k:02d}: {'; '.join(p)}")
        out[ep] = {"kind": kind, "name": name, "rec": rec}
    return out, problems


# ------------------------------------------------------------------------------------------------ pack + score
def pack(results: Path, out_csv: Path, log_file: Path, commit_url: str) -> dict:
    cmd = ["nice", "-n", "15", str(PACK_PYTHON), "-B", str(PACKER), "--track", "track_2_tier1", "--results", str(results),
           "--dataset-root", str(DATASET), "--code-commit", commit_url, "--eval-script", str(EVAL_SCRIPT),
           "-o", str(inside(out_csv))]  # fmt: skip
    p = subprocess.run(cmd, cwd=KIT, env=kit_env(), capture_output=True, text=True, timeout=3600)
    write_atomic(log_file, f"$ {shlex.join(cmd)}\n{p.stdout}\n{p.stderr}")
    out = {"exit": p.returncode, "csv": str(out_csv), "log": str(log_file), "command": shlex.join(cmd)}
    if p.returncode == 0 and out_csv.is_file():
        out["csv_sha256"] = sha256(out_csv)
        m = re.search(r"^locally computed: (.*)$", p.stdout, re.MULTILINE)
        if m:
            out["packer_local_scores"] = dict(x.split("=", 1) for x in m.group(1).split())
        out["crosscheck_warning"] = "WARNING: the trajectory and the evaluator's metrics JSON disagree" in p.stdout
    return out


def score(csv_file: Path, out_json: Path, log_file: Path) -> dict | None:
    cmd = ["nice", "-n", "15", str(PACK_PYTHON), "-B", str(SCORER), "--csv", str(csv_file), "--out", str(inside(out_json)),
           "--episodes", "all"]  # fmt: skip
    p = subprocess.run(cmd, cwd=U, env=kit_env(), capture_output=True, text=True, timeout=3600)
    write_atomic(log_file, f"$ {shlex.join(cmd)}\n{p.stdout}\n{p.stderr}")
    return read_json(out_json) if p.returncode == 0 else None


def copy_results(dest: Path, src: dict[int, dict]) -> None:
    dest = inside(dest)
    if dest.is_dir():
        for leftover in dest.glob("episode_*"):
            leftover.unlink()
    dest.mkdir(parents=True, exist_ok=True)
    for ep in EPISODES:
        r = src[ep]["rec"]
        for f in (r["results_json"], r["results_parquet"]):
            dst = dest / Path(f).name
            tmp = dst.with_name(f".{dst.name}.tmp")
            shutil.copy2(f, tmp)
            os.replace(tmp, dst)
    names = sorted(p.name for p in dest.glob("episode_*"))
    if len(names) != 60:
        raise SystemExit(f"{dest}: {len(names)} files, expected 60")


def pack_trial(config: str, k: int) -> dict:
    base = U / "scores" / config / f"t{k:02d}"
    src, problems = sources(config, k)
    if problems:
        out = {"trial": k, "config": config, "complete": False, "problems": problems}
        write_atomic(base / "score.json", dump(out))
        log(f"{config} t{k:02d}: INCOMPLETE ({len(problems)} problems: {problems[:3]})")
        return out
    inputs = {str(ep): src[ep]["rec"]["parquet_sha256"] for ep in EPISODES}
    old = read_json(base / "score.json") or {}
    if old.get("inputs") == inputs and old.get("set"):
        return old
    copy_results(base / "results", src)
    csv_file = base / f"{config}_trial{k:02d}_LOCAL_NOT_FOR_UPLOAD.csv"
    pk = pack(base / "results", csv_file, base / "pack.log", PLACEHOLDER)
    sc = score(csv_file, base / "metric_code.json", base / "score.log") if pk.get("csv_sha256") else None
    out = {"trial": k, "config": config, "complete": True, "scored_at": now(), "inputs": inputs,
           "sources": {str(ep): {"kind": src[ep]["kind"], "name": src[ep]["name"],
                                 "eval_file_sha256": src[ep]["rec"]["pinned_sha256"],
                                 "results_parquet": src[ep]["rec"]["results_parquet"],
                                 "json_metrics": src[ep]["rec"].get("metrics"),
                                 "seed_signature": (src[ep]["rec"].get("seeds") or {}).get("seed_signature"),
                                 "gpu": src[ep]["rec"].get("gpu"), "started_at": src[ep]["rec"].get("started_at")}
                         for ep in EPISODES},
           "pack": pk,
           "set": None if sc is None else {"overall_float32": sc["overall_float32"], "overall_float64": sc["overall_float64"],
                                           "per_episode_float32": sc["per_episode_float32"],
                                           "mean_of_per_episode_float32": sc["mean_of_per_episode_float32"]}}  # fmt: skip
    write_atomic(base / "score.json", dump(out))
    log(f"{config} t{k:02d}: packed (exit {pk['exit']}, csv {str(pk.get('csv_sha256'))[:12]}) and scored "
        f"{'FAILED' if sc is None else 'ok'}")
    return out


def cmd_pack(a) -> int:
    preflight()
    trials = TRIALS if a.trials == "all" else [int(x) for x in a.trials.split(",")]
    log(f"pack {a.config}: trials {trials}, workers {a.workers}")
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(lambda k: pack_trial(a.config, k), trials))
    ok = sum(1 for r in res if r.get("set"))
    log(f"pack {a.config}: {ok}/{len(trials)} trials packed and scored")
    return 0 if ok == len(trials) else 1


# ------------------------------------------------------------------------------------------------ analyze (8.5)
def load_config(config: str) -> dict[int, dict]:
    out = {}
    for k in TRIALS:
        s = read_json(U / "scores" / config / f"t{k:02d}" / "score.json")
        if s and s.get("complete") and s.get("set"):
            out[k] = s
    return out


def b20_values(k: int) -> dict[int, dict] | None:
    s = read_json(B20 / "scores" / f"t{k:02d}" / "score.json")
    if not s or not s.get("set") or not (s.get("alt") or {}).get("per_episode_float32"):
        return None
    pe = {int(e): v for e, v in s["set"]["per_episode_float32"].items()}
    pe[28] = s["alt"]["per_episode_float32"]["28"]  # the pinned set's ep28 = ep28alt (W2PD 66k)
    return pe


def cmd_analyze(a) -> int:
    preflight()
    all7, pinned = load_config("all7"), load_config("pinned")
    inc = {k: b20_values(k) for k in TRIALS}
    changed_eps = {v[1] for v in CHANGED.values()}
    checks = {"all7_unchanged_vs_b20_max_abs": 0.0, "pinned_vs_b20_max_abs": 0.0, "mismatches": []}
    for name, cfg, eps in (("all7_unchanged_vs_b20_max_abs", all7, [e for e in EPISODES if e not in changed_eps]),
                           ("pinned_vs_b20_max_abs", pinned, EPISODES)):
        for k, s in cfg.items():
            pe = s["set"]["per_episode_float32"]
            for e in eps:
                if inc.get(k) is None:
                    continue
                for m in METRICS:
                    d = abs(pe[str(e)][m] - inc[k][e][m])
                    checks[name] = max(checks[name], d)
                    if d > 1e-9:
                        checks["mismatches"].append(f"{name} t{k:02d} ep{e:02d} {m}: {pe[str(e)][m]} vs {inc[k][e][m]}")
    items = {}
    for item, (winner, ep, want) in CHANGED.items():
        new = [all7[k]["set"]["per_episode_float32"][str(ep)] for k in TRIALS if k in all7]
        new_json = [all7[k]["sources"][str(ep)]["json_metrics"] for k in TRIALS if k in all7]
        old = [inc[k][ep] for k in TRIALS if inc.get(k) is not None]
        n_new, n_old = len(new), len(old)
        sn = {m: mstats([x[m] for x in new]) for m in METRICS}
        so = {m: mstats([x[m] for x in old]) for m in METRICS}
        sp_new, mp_new = sum(1 for x in new if x["SP-SR"] >= 0.5), sum(1 for x in new if x["MP-SR"] >= 0.5)
        sp_old, mp_old = sum(1 for x in old if x["SP-SR"] >= 0.5), sum(1 for x in old if x["MP-SR"] >= 0.5)
        auc_n, auc_o = sn["AUC"].get("mean"), so["AUC"].get("mean")
        mppe_n, mppe_o = sn["MPPE"].get("mean"), so["MPPE"].get("mean")
        why = []
        if not n_new or not n_old:
            higher, keep = "incumbent", False
            why.append(f"no data (new n {n_new}, incumbent n {n_old})")
        else:
            higher = "new" if auc_n > auc_o else "incumbent"  # exact tie -> incumbent
            if higher == "new" and mppe_n - mppe_o > MPPE_WORSE:
                chosen = "incumbent"
                why.append(f"new has the higher AUC but MPPE {mppe_n:.3f} is > {MPPE_WORSE} cm worse than {mppe_o:.3f}")
            elif higher == "incumbent" and mppe_o - mppe_n > MPPE_WORSE:
                chosen = "new"
                why.append(f"incumbent has the higher AUC but its MPPE {mppe_o:.3f} is > {MPPE_WORSE} cm worse than {mppe_n:.3f}")
            else:
                chosen = higher
                why.append(f"{higher} has the higher 20-trial kit AUC ({auc_n:.4f} new vs {auc_o:.4f} incumbent); MPPE "
                           f"{mppe_n:.3f} vs {mppe_o:.3f}")
            keep = chosen == "new"
            if keep and sp_new < sp_old:
                keep = False
                why.append(f"SP pass {sp_new}/{n_new} < incumbent {sp_old}/{n_old}")
            if keep and mp_new < mp_old:
                keep = False
                why.append(f"MP pass {mp_new}/{n_new} < incumbent {mp_old}/{n_old}")
            if keep and item in EP28_EXCEPTION and mp_new < 0.8 * n_new:
                keep = False
                why.append(f"8.4(c) exception: MP pass {mp_new}/{n_new} < 16/20 (0.8 n)")
            if item in EP28_EXCEPTION and mp_new >= 0.8 * n_new:
                why.append(f"8.4(c) exception: MP pass {mp_new}/{n_new} >= 16/20")
        items[item] = {"item": item, "episode": ep, "winner": winner, "eval_file_sha256": want, "n_new": n_new, "n_incumbent": n_old,
                       "new": sn, "incumbent": so, "sp_pass_new": sp_new, "mp_pass_new": mp_new, "sp_pass_incumbent": sp_old,
                       "mp_pass_incumbent": mp_old,
                       "json_new": {"auc_mean": statistics.fmean([j["auc"] for j in new_json]) if new_json else None,
                                    "sp_pass": sum(1 for j in new_json if j["sp_sr"] >= 0.5),
                                    "mp_pass": sum(1 for j in new_json if j["mp_sr"] >= 0.5),
                                    "mppe_mean": statistics.fmean([j["mppe"] for j in new_json]) if new_json else None},
                       "new_trials_auc": {k: all7[k]["set"]["per_episode_float32"][str(ep)]["AUC"] for k in TRIALS if k in all7},
                       "new_trials_mp": {k: all7[k]["set"]["per_episode_float32"][str(ep)]["MP-SR"] for k in TRIALS if k in all7},
                       "keep_new": keep, "why": "; ".join(why)}  # fmt: skip
        log(f"8.5 {item}: {'KEEP ' + winner if keep else 'INCUMBENT STAYS'} ({'; '.join(why)})")
    out = {"generated_at": now(), "rules_sha256": sha256(RULES), "script_sha256": sha256(Path(__file__).resolve()),
           "n_all7_trials": len(all7), "n_pinned_trials": len(pinned), "checks": checks, "items": items,
           "kept": sorted(i for i, x in items.items() if x["keep_new"]),
           "reverted": sorted(i for i, x in items.items() if not x["keep_new"]),
           "final_equals_all7": all(x["keep_new"] for x in items.values())}  # fmt: skip
    write_atomic(U / "decisions.json", dump(out))
    log(f"analyze: kept {out['kept']}; reverted {out['reverted']}; checks max diffs "
        f"{checks['all7_unchanged_vs_b20_max_abs']:.3g} / {checks['pinned_vs_b20_max_abs']:.3g}, {len(checks['mismatches'])} mismatches")
    return 0


# ------------------------------------------------------------------------------------------------ select (8.6)
def final_config() -> str:
    d = read_json(U / "decisions.json")
    return "all7" if d and d["final_equals_all7"] else "final"


def set_row(s: dict) -> dict:
    o = s["set"]["overall_float32"]
    js = [s["sources"][str(e)]["json_metrics"] for e in EPISODES]
    sigs = sorted({s["sources"][str(e)]["seed_signature"] for e in EPISODES})
    return {"trial": s["trial"], "auc": o["AUC"], "sp_sr": o["SP-SR"], "mp_sr": o["MP-SR"], "mppe": o["MPPE"],
            "auc_float64": s["set"]["overall_float64"]["AUC"],
            "json_mean_auc": statistics.fmean(j["auc"] for j in js), "json_mean_mppe": statistics.fmean(j["mppe"] for j in js),
            "seed_signatures": " | ".join(sigs), "csv_sha256": s["pack"]["csv_sha256"],
            "crosscheck_warning": s["pack"].get("crosscheck_warning")}  # fmt: skip


def mode_of(vals: list[float]) -> float:
    c = Counter(round(v, 9) for v in vals)
    top = max(c.values())
    return max(v for v, n in c.items() if n == top)  # multimodal tie -> the larger value


def cmd_select(a) -> int:
    preflight()
    cfg = final_config()
    if cfg == "final" and len(load_config("final")) == 0:
        raise SystemExit("final config not packed yet")
    fin, pinned = load_config(cfg), load_config("pinned")
    rows = [set_row(fin[k]) for k in TRIALS if k in fin]
    prow = {k: set_row(pinned[k]) for k in pinned}
    dec = read_json(U / "decisions.json")
    for r in rows:
        k = r["trial"]
        r["pinned_auc"], r["pinned_sp_sr"], r["pinned_mp_sr"], r["pinned_mppe"] = (
            prow[k]["auc"], prow[k]["sp_sr"], prow[k]["mp_sr"], prow[k]["mppe"]) if k in prow else (None,) * 4
        r["delta_auc_vs_pinned"] = r["auc"] - r["pinned_auc"] if r["pinned_auc"] is not None else None
        for item, x in dec["items"].items():
            r[f"{item}_auc"] = fin[k]["set"]["per_episode_float32"][str(x["episode"])]["AUC"]
            r[f"{item}_mp"] = fin[k]["set"]["per_episode_float32"][str(x["episode"])]["MP-SR"]
    st = {m: mstats([r[m] for r in rows]) for m in ("auc", "sp_sr", "mp_sr", "mppe", "auc_float64", "json_mean_auc")}
    pst = {m: mstats([r[m] for r in prow.values()]) for m in ("auc", "sp_sr", "mp_sr", "mppe")}
    sp_mode, mp_mode = mode_of([r["sp_sr"] for r in rows]), mode_of([r["mp_sr"] for r in rows])
    cands = [r for r in rows if round(r["sp_sr"], 9) == sp_mode and round(r["mp_sr"], 9) == mp_mode]
    sa, sm = st["auc"]["sd"], st["mppe"]["sd"]

    def dist(r):
        return (abs(r["auc"] - st["auc"]["mean"]) / sa if sa > 0 else 0.0) + (abs(r["mppe"] - st["mppe"]["mean"]) / sm if sm > 0 else 0.0)

    for r in rows:
        r["mode_candidate"] = r in cands
        r["dist_to_mean"] = dist(r)
    closest = min(cands, key=lambda r: (dist(r), r["trial"]))
    best = min(rows, key=lambda r: (-r["auc"], r["mppe"], r["trial"]))
    cols = ["trial", "auc", "sp_sr", "mp_sr", "mppe", "auc_float64", "json_mean_auc", "json_mean_mppe", "pinned_auc",
            "pinned_sp_sr", "pinned_mp_sr", "pinned_mppe", "delta_auc_vs_pinned", "mode_candidate", "dist_to_mean"]
    cols += [f"{i}_{m}" for i in dec["items"] for m in ("auc", "mp")]
    cols += ["seed_signatures", "csv_sha256", "crosscheck_warning"]
    write_csv(U / "per_trial.csv", rows, cols)
    # seeds of the changed items' 20 trials (t01-t10 confirmation + t11-t20 top-up) and of every set item
    seed_rows = []
    for k in TRIALS:
        if k not in fin:
            continue
        for e in EPISODES:
            s = fin[k]["sources"][str(e)]
            seed_rows.append({"trial": f"{k:02d}", "episode": e, "source": f"{s['kind']}:{s['name']}",
                              "eval_file_sha256": s["eval_file_sha256"], "seed_signature": s["seed_signature"],
                              "gpu": s["gpu"], "started_at": s["started_at"], "results_parquet": s["results_parquet"]})
    write_csv(U / "seeds.tsv", seed_rows, list(seed_rows[0].keys()), delimiter="\t")
    sigs = Counter(r["seed_signature"] for r in seed_rows)
    out = {"generated_at": now(), "config": cfg, "n_trials": len(rows), "stats": st, "pinned_stats": pst,
           "sp_mode": sp_mode, "mp_mode": mp_mode, "mode_candidates": [r["trial"] for r in cands],
           "closest_to_mean": closest, "best_trial": best, "seed_signature_counts": dict(sigs),
           "decisions": {i: x["keep_new"] for i, x in dec["items"].items()},
           "rules_sha256": sha256(RULES), "script_sha256": sha256(Path(__file__).resolve())}  # fmt: skip
    write_atomic(U / "selection.json", dump(out))
    log(f"select ({cfg}, n {len(rows)}): mean AUC {st['auc']['mean']:.4f} sd {st['auc']['sd']:.4f}; closest t{closest['trial']:02d} "
        f"AUC {closest['auc']:.4f} MPPE {closest['mppe']:.3f}; best t{best['trial']:02d} AUC {best['auc']:.4f}")
    return 0


# ------------------------------------------------------------------------------------------------ files
def cmd_files(a) -> int:
    preflight()
    chk = read_json(a.commit_check) or {}
    if not chk.get("ok") or chk.get("commit_url") != a.commit_url:
        raise SystemExit(f"{a.commit_check}: commit check not ok for {a.commit_url}; nothing packed with that URL")
    sel = read_json(U / "selection.json")
    cfg = sel["config"]
    out = {"generated_at": now(), "commit_url": a.commit_url, "commit_check": str(a.commit_check), "files": []}
    for label, r in (("SUBMITTABLE_closest_to_mean", sel["closest_to_mean"]), ("REPORT_ONLY_best_trial", sel["best_trial"])):
        k = r["trial"]
        s = read_json(U / "scores" / cfg / f"t{k:02d}" / "score.json")
        src, problems = sources(cfg, k)
        if problems:
            raise SystemExit(f"t{k:02d}: {problems}")
        if {str(e): src[e]["rec"]["parquet_sha256"] for e in EPISODES} != s["inputs"]:
            raise SystemExit(f"t{k:02d}: inputs differ from the scored trial")
        base = U / "files" / f"trial{k:02d}_{label}"
        copy_results(base / "results", src)
        name = f"T1improve_tier1_trial{k:02d}_{label}.csv"
        pk = pack(base / "results", base / name, base / "pack.log", a.commit_url)
        if not pk.get("csv_sha256"):
            raise SystemExit(f"pack failed: {base / 'pack.log'}")
        sc = score(base / name, base / "metric_code.json", base / "score.log")
        want = s["set"]["overall_float32"]
        diffs = {m: abs(sc["overall_float32"][m] - want[m]) for m in METRICS} if sc else None
        expected = {str(e): src[e]["rec"]["pinned_sha256"] for e in EPISODES}
        write_atomic(base / "expected_checkpoints.json", dump(expected))
        cmd = ["nice", "-n", "15", str(PACK_PYTHON), "-B", str(VALIDATOR), str(base / name), str(base / "expected_checkpoints.json"),
               a.commit_url, str(inside(base / "validate.json"))]  # fmt: skip
        p = subprocess.run(cmd, cwd=U, env=kit_env(), capture_output=True, text=True, timeout=3600)
        write_atomic(base / "validate.log", f"$ {shlex.join(cmd)}\n{p.stdout}\n{p.stderr}")
        val = read_json(base / "validate.json") or {}
        rec = {"label": label, "trial": k, "csv": str(base / name), "csv_sha256": pk["csv_sha256"], "size": (base / name).stat().st_size,
               "rescore_overall_float32": sc["overall_float32"] if sc else None, "trial_overall_float32": want,
               "rescore_equal": bool(diffs) and max(diffs.values()) <= 1e-12, "rescore_abs_diffs": diffs,
               "validate_ok": bool(val.get("ok")) and p.returncode == 0, "validate": val,
               "placeholder_csv_sha256": s["pack"]["csv_sha256"]}  # fmt: skip
        out["files"].append(rec)
        log(f"files {label}: t{k:02d} {name} sha {pk['csv_sha256'][:12]}; rescore equal {rec['rescore_equal']}; validate {rec['validate_ok']}")
    write_atomic(U / "files.json", dump(out))
    return 0 if all(f["rescore_equal"] and f["validate_ok"] for f in out["files"]) else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("pack")
    q.add_argument("--config", required=True, choices=["pinned", "all7", "final"])
    q.add_argument("--trials", default="all")
    q.add_argument("--workers", type=int, default=4)
    sub.add_parser("analyze")
    sub.add_parser("select")
    f = sub.add_parser("files")
    f.add_argument("--commit-url", required=True)
    f.add_argument("--commit-check", required=True, type=Path)
    a = p.parse_args()
    return {"pack": cmd_pack, "analyze": cmd_analyze, "select": cmd_select, "files": cmd_files}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
