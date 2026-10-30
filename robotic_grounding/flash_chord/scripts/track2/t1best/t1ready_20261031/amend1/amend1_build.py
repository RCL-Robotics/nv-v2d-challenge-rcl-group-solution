#!/usr/bin/env python3
"""amend1_build.py: TOPUP_AMEND1 (ep28 = W3H 83000, set `all7`) selection + submittable file.

Reads U (fc_runs/T1improve/topup), C (confirm) and B20 only; writes only under U/amend1 (A). Reuses the hash-locked
U/t1topup.py read-only helpers (sources, check_rec, set_row, mode_of, mstats) by import, unchanged. Never signals a
process, never touches the kit (kit Python with -B and PYTHONDONTWRITEBYTECODE=1), uploads nothing.

  <v2dkit python> -B amend1_build.py select   # TOPUP_RULES section 6 over the 20 all7 trials -> A/selection.json
  <v2dkit python> -B amend1_build.py files    # repack + re-score + validate the submittable trial -> A/files.json
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shlex  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

ME = Path("/root/autodl-tmp/haoran.zhang")
U = ME / "fc_runs/T1improve/topup"
A = U / "amend1"
AMEND, AMEND_SHA = A / "TOPUP_AMEND1.md", A / "TOPUP_AMEND1.sha256"
COMMIT_URL = "https://github.com/Beat-in-our-hearts/nv-v2d-challenge-codebase/commit/0c7516eda6f90deacb1536aabbce98beba5194ca"
EP28_W3H = "497ae3b3a785fc9283fcdda2c019020608ed8c41472f4229ac780f4450081375"
CONFIG = "all7"
FALLBACK = {"config": "final", "trial": 3,
            "csv": str(U / "files/trial03_SUBMITTABLE_closest_to_mean/T1improve_tier1_trial03_SUBMITTABLE_closest_to_mean.csv"),
            "csv_sha256": "93f1d61592da3788f76c133acefb477ca46e8141705b57ec7e2c14a7a5d69a85"}


def load_t1topup():
    spec = importlib.util.spec_from_file_location("t1topup_ro", U / "t1topup.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = load_t1topup()


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def inside(path: Path) -> Path:
    p = Path(path).resolve()
    if A.resolve() not in p.parents and p != A.resolve():
        raise SystemExit(f"refusing to write outside {A}: {p}")
    return p


def write_atomic(path: Path, text: str) -> None:
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def log(msg: str) -> None:
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    with inside(A / "amend1.log").open("a") as f:
        f.write(line + "\n")


def env() -> dict:
    e = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")}
    e.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
             TMPDIR=str(inside(A / "tmp")))
    return e


def preflight() -> dict:
    bad = []
    if T.sha256(U / "t1topup.py") not in (U / "TOPUP_SCRIPT.sha256").read_text():
        bad.append("t1topup.py does not match TOPUP_SCRIPT.sha256")
    if T.sha256(T.VALIDATOR) not in (U / "TOPUP_SCRIPT.sha256").read_text():
        bad.append("t1topup_validate.py does not match TOPUP_SCRIPT.sha256")
    if T.sha256(T.RULES) != T.RULES_SHA.read_text().split()[0]:
        bad.append("TOPUP_RULES.md does not match TOPUP_RULES.sha256")
    if T.sha256(AMEND) != AMEND_SHA.read_text().split()[0]:
        bad.append("TOPUP_AMEND1.md does not match TOPUP_AMEND1.sha256")
    for path, want in T.PINS.items():
        got = T.sha256(path) if Path(path).is_file() else "missing"
        if got != want:
            bad.append(f"{path}: sha256 {got[:16]} != pinned {want[:16]}")
    if T.CHANGED["ep28alt"][0] != "ep28-W3H_83000" or T.CHANGED["ep28alt"][2] != EP28_W3H:
        bad.append("t1topup CHANGED ep28alt is not W3H 83000 / 497ae3b3")
    if bad:
        raise SystemExit("preflight FAILED:\n  " + "\n  ".join(bad))
    inside(A / "tmp").mkdir(exist_ok=True)
    return {"rules_sha256": T.sha256(T.RULES), "amend1_sha256": T.sha256(AMEND), "t1topup_sha256": T.sha256(U / "t1topup.py"),
            "this_script_sha256": T.sha256(Path(__file__).resolve())}


def fails(pe: dict, metric: str) -> list[int]:
    return sorted(int(e) for e, v in pe.items() if v[metric] < 0.5)


# ------------------------------------------------------------------------------------------------ select
def cmd_select(a) -> int:
    hashes = preflight()
    all_scores = {k: T.read_json(U / "scores" / CONFIG / f"t{k:02d}" / "score.json") for k in T.TRIALS}
    incomplete = {k: (s or {}).get("problems", "no score.json") for k, s in all_scores.items()
                  if not (s and s.get("complete") and s.get("set"))}
    fin = {k: s for k, s in all_scores.items() if k not in incomplete}
    rows = []
    for k in T.TRIALS:
        if k not in fin:
            continue
        r = T.set_row(fin[k])
        pe = fin[k]["set"]["per_episode_float32"]
        r["sp_fail_episodes"], r["mp_fail_episodes"] = fails(pe, "SP-SR"), fails(pe, "MP-SR")
        r["ep28_auc"], r["ep28_mp"], r["ep28_mppe"] = pe["28"]["AUC"], pe["28"]["MP-SR"], pe["28"]["MPPE"]
        r["ep28_eval_file_sha256"] = fin[k]["sources"]["28"]["eval_file_sha256"]
        r["sp_mp_all_1"] = r["sp_sr"] == 1.0 and r["mp_sr"] == 1.0
        rows.append(r)
    bad28 = [r["trial"] for r in rows if r["ep28_eval_file_sha256"] != EP28_W3H]
    if bad28:
        raise SystemExit(f"all7 trials {bad28}: ep28 is not W3H 497ae3b3")
    st = {m: T.mstats([r[m] for r in rows]) for m in ("auc", "sp_sr", "mp_sr", "mppe", "auc_float64", "json_mean_auc")}
    sp_mode, mp_mode = T.mode_of([r["sp_sr"] for r in rows]), T.mode_of([r["mp_sr"] for r in rows])
    cands = [r for r in rows if round(r["sp_sr"], 9) == sp_mode and round(r["mp_sr"], 9) == mp_mode]
    sa, sm = st["auc"]["sd"], st["mppe"]["sd"]

    def dist(r):
        return (abs(r["auc"] - st["auc"]["mean"]) / sa if sa > 0 else 0.0) + (abs(r["mppe"] - st["mppe"]["mean"]) / sm if sm > 0 else 0.0)

    for r in rows:
        r["mode_candidate"] = r in cands
        r["dist_to_mean"] = dist(r)
    closest = min(cands, key=lambda r: (dist(r), r["trial"]))
    best = min(rows, key=lambda r: (-r["auc"], r["mppe"], r["trial"]))
    out = {"generated_at": now(), "amendment": str(AMEND), **hashes, "config": CONFIG,
           "rule": "TOPUP_RULES section 6 (unchanged) applied to config all7 per TOPUP_AMEND1",
           "n_trials": len(rows), "incomplete_trials": incomplete, "stats": st,
           "n_trials_sp_mp_all_1": sum(1 for r in rows if r["sp_mp_all_1"]),
           "sp_mode": sp_mode, "mp_mode": mp_mode, "mode_candidates": [r["trial"] for r in cands],
           "closest_to_mean": {**closest, "label": "SUBMITTABLE_closest_to_mean"},
           "best_trial": {**best, "label": "REPORT_ONLY_best_trial (not packed for upload)"},
           "fallback": FALLBACK, "per_trial": rows}  # fmt: skip
    write_atomic(A / "selection.json", T.dump(out))
    log(f"select all7 (n {len(rows)}, incomplete {sorted(incomplete)}): mean AUC {st['auc']['mean']:.5f} sd {st['auc']['sd']:.5f}; "
        f"SP mode {sp_mode} MP mode {mp_mode}; {len(cands)} candidates; closest t{closest['trial']:02d} AUC {closest['auc']:.5f} "
        f"MPPE {closest['mppe']:.4f}; best (report only) t{best['trial']:02d} AUC {best['auc']:.5f}")
    return 0


# ------------------------------------------------------------------------------------------------ files
def run(cmd: list[str], cwd: Path, log_file: Path) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, cwd=cwd, env=env(), capture_output=True, text=True, timeout=3600)
    write_atomic(log_file, f"$ {shlex.join(cmd)}\n{p.stdout}\n{p.stderr}")
    return p


def dir_digest(d: Path) -> dict:
    return {f.name: T.sha256(f) for f in sorted(d.glob("episode_*"))}


def cmd_files(a) -> int:
    hashes = preflight()
    chk = T.read_json(U / "commit_check.json") or {}
    if not chk.get("ok") or chk.get("commit_url") != COMMIT_URL or "ep28-W3H" not in (chk.get("arms") or []):
        raise SystemExit("commit_check.json not ok for the commit URL / arm ep28-W3H; nothing packed")
    sel = T.read_json(A / "selection.json")
    if not sel or sel.get("config") != CONFIG or sel.get("amend1_sha256") != hashes["amend1_sha256"]:
        raise SystemExit("A/selection.json missing or not for this amendment")
    k = int(sel["closest_to_mean"]["trial"])
    base_s = U / "scores" / CONFIG / f"t{k:02d}"
    s = T.read_json(base_s / "score.json")
    problems = []
    src, sp = T.sources(CONFIG, k)
    if sp:
        raise SystemExit(f"t{k:02d}: {sp}")
    if {str(e): src[e]["rec"]["parquet_sha256"] for e in T.EPISODES} != s["inputs"]:
        raise SystemExit(f"t{k:02d}: run.json parquet sha256 differ from the scored trial's inputs")
    results = base_s / "results"
    before = dir_digest(results)
    want_files = {}
    for e in T.EPISODES:
        r = src[e]["rec"]
        want_files[Path(r["results_json"]).name] = r["json_sha256"]
        want_files[Path(r["results_parquet"]).name] = r["parquet_sha256"]
    if before != want_files:
        raise SystemExit(f"{results}: contents differ from the trial's run.json records")
    expected = {str(e): src[e]["rec"]["pinned_sha256"] for e in T.EPISODES}
    if expected != {str(e): s["sources"][str(e)]["eval_file_sha256"] for e in T.EPISODES} or expected["28"] != EP28_W3H:
        raise SystemExit("expected checkpoints differ from score.json sources or ep28 != W3H 497ae3b3")
    label = "SUBMITTABLE_W3H_closest_to_mean"
    base = A / "files" / f"trial{k:02d}_{label}"
    name = f"T1improve_W3H_tier1_trial{k:02d}_SUBMITTABLE_closest_to_mean.csv"
    out_csv = inside(base / name)
    if out_csv.exists():
        raise SystemExit(f"{out_csv} exists; not overwriting")
    # the trial's own pack command, with only --code-commit and -o changed
    old = shlex.split(s["pack"]["command"])
    cmd = list(old)
    i_c, i_o = cmd.index("--code-commit"), cmd.index("-o")
    if cmd[i_c + 1] != T.PLACEHOLDER or Path(cmd[cmd.index("--results") + 1]) != results:
        raise SystemExit(f"unexpected pack command: {s['pack']['command']}")
    cmd[i_c + 1], cmd[i_o + 1] = COMMIT_URL, str(out_csv)
    base.mkdir(parents=True, exist_ok=True)
    p = run(cmd, T.KIT, base / "pack.log")
    if p.returncode != 0 or not out_csv.is_file():
        raise SystemExit(f"pack failed (exit {p.returncode}): {base / 'pack.log'}")
    csv_sha = T.sha256(out_csv)
    after = dir_digest(results)
    if after != before:
        problems.append("results dir changed during packing")
    # re-score with the t1best20 scorer, --episodes all
    sc_cmd = ["nice", "-n", "15", str(T.PACK_PYTHON), "-B", str(T.SCORER), "--csv", str(out_csv),
              "--out", str(inside(base / "metric_code.json")), "--episodes", "all"]
    p = run(sc_cmd, A, base / "score.log")
    sc = T.read_json(base / "metric_code.json") if p.returncode == 0 else None
    if sc is None:
        raise SystemExit(f"score failed: {base / 'score.log'}")
    want = s["set"]
    ov_diff = {m: abs(sc["overall_float32"][m] - want["overall_float32"][m]) for m in T.METRICS}
    ov_equal = all(sc["overall_float32"][m] == want["overall_float32"][m] for m in T.METRICS)
    ov64_equal = all(sc["overall_float64"][m] == want["overall_float64"][m] for m in T.METRICS)
    pe_mism = [f"ep{e} {m}: {sc['per_episode_float32'].get(e, {}).get(m)} vs {v[m]}"
               for e, v in want["per_episode_float32"].items() for m in T.METRICS
               if sc["per_episode_float32"].get(e, {}).get(m) != v[m]]
    pe_equal = not pe_mism and set(sc["per_episode_float32"]) == set(want["per_episode_float32"])
    rescore_equal = ov_equal and pe_equal
    if not rescore_equal:
        problems.append(f"re-score differs: overall diffs {ov_diff}; per-episode mismatches {pe_mism[:5]}")
    # provenance + kit uploader validate_submission()
    write_atomic(base / "expected_checkpoints.json", T.dump(expected))
    v_cmd = ["nice", "-n", "15", str(T.PACK_PYTHON), "-B", str(T.VALIDATOR), str(out_csv), str(base / "expected_checkpoints.json"),
             COMMIT_URL, str(inside(base / "validate.json"))]
    p = run(v_cmd, A, base / "validate.log")
    val = T.read_json(base / "validate.json") or {}
    validate_ok = bool(val.get("ok")) and p.returncode == 0
    if not validate_ok:
        problems.append(f"validator: exit {p.returncode}, problems {val.get('problems')}")
    # the commit URL is the only difference from the trial's placeholder pack
    import pandas as pd

    new_df, old_df = pd.read_csv(out_csv, dtype=str, keep_default_na=False), pd.read_csv(Path(old[i_o + 1]), dtype=str, keep_default_na=False)
    same_shape = list(new_df.columns) == list(old_df.columns) and len(new_df) == len(old_df)
    diff_cols = [c for c in new_df.columns if same_shape and not new_df[c].equals(old_df[c])] if same_shape else ["shape"]
    if diff_cols != ["code_commit_url"]:
        problems.append(f"columns differing from the placeholder pack: {diff_cols}")
    rec = {"label": label, "trial": k, "config": CONFIG, "csv": str(out_csv), "csv_sha256": csv_sha, "size": out_csv.stat().st_size,
           "pack_command": shlex.join(cmd), "trial_pack_command": s["pack"]["command"],
           "results_dir": str(results), "results_unchanged": after == before,
           "rescore_overall_float32": sc["overall_float32"], "trial_overall_float32": want["overall_float32"],
           "rescore_overall_float64": sc["overall_float64"], "trial_overall_float64": want["overall_float64"],
           "rescore_abs_diffs": ov_diff, "rescore_overall_equal": ov_equal, "rescore_overall_float64_equal": ov64_equal,
           "rescore_per_episode_equal": pe_equal, "rescore_per_episode_mismatches": pe_mism, "rescore_equal": rescore_equal,
           "validate_ok": validate_ok, "validate": val, "expected_checkpoints": expected, "ep28_checkpoint": expected["28"],
           "diff_vs_placeholder_pack": diff_cols, "placeholder_csv": old[i_o + 1],
           "placeholder_csv_sha256": s["pack"]["csv_sha256"]}  # fmt: skip
    out = {"generated_at": now(), "amendment": str(AMEND), **hashes, "commit_url": COMMIT_URL,
           "commit_check": str(U / "commit_check.json"), "commit_remote_check": T.read_json(U / "commit_remote_check.json"),
           "files": [rec], "fallback": FALLBACK, "problems": problems, "ok": not problems}  # fmt: skip
    write_atomic(A / "files.json", T.dump(out))
    log(f"files {label}: t{k:02d} {name} sha {csv_sha[:12]}; rescore equal {rescore_equal}; validate {validate_ok}; problems {problems}")
    return 0 if not problems else 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    sub.add_parser("files")
    a = p.parse_args()
    return {"select": cmd_select, "files": cmd_files}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
