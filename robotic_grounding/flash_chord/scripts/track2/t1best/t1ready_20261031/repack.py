#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""repack.py: re-pack the trial-05 results of the Tier-1 set `all7` with a new code commit URL, without evaluating.

Only the --code-commit argument of the trial's own kit pack command changes. Then:
  1. the results folder is unchanged by packing (sha256 of every episode file before and after);
  2. the kit uploader's offline check, validate_submission() of submit/v2d_submit.py, called directly (no Kaggle
     API, no credentials), plus provenance: uniform digests, the commit URL on every row, one checkpoint per
     episode equal to the set's evaluated file (--expected);
  3. the official metric_code (kit metric_code/track_2) scores of the new file, through --scorer, equal the
     source file's scores (overall float32 and float64, per episode float32), exactly;
  4. every row of the new file equals the source file's row except the code_commit_url field.
Writes only under --out-dir. Run with the kit env python and -B (never writes bytecode into the kit):

  <kit python> -B repack.py --kit KIT --dataset-root DATA --results RESULTS --source-csv SRC.csv \
      --source-scores SRC_metric_code.json --expected expected_checkpoints.json --scorer t1best20_score.py \
      --validator t1topup_validate.py --commit-url https://github.com/<owner>/<repo>/commit/<sha> \
      --out-dir DIR --name FILE.csv
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
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

TRACK = "track_2_tier1"
METRICS = ("AUC", "SP-SR", "MP-SR", "MPPE")
URL_RE = re.compile(r"^https://github\.com/[^/]+/[^/]+/commit/[0-9a-f]{40}$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(d: Path) -> dict:
    return {f.name: sha256(f) for f in sorted(d.glob("episode_*"))}


def run(cmd: list[str], cwd: Path, log: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=3600)
    log.write_text(f"$ {shlex.join(cmd)}\n{p.stdout}\n{p.stderr}")
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("kit", "dataset-root", "results", "source-csv", "source-scores", "expected", "scorer", "validator", "out-dir"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    ap.add_argument("--commit-url", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--python", default=sys.executable, help="kit env python for the packer, scorer and validator")
    a = ap.parse_args()
    if not URL_RE.match(a.commit_url):
        raise SystemExit(f"not a full-sha GitHub commit URL: {a.commit_url}")
    out_dir = a.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / a.name
    if out_csv.exists():
        raise SystemExit(f"{out_csv} exists; not overwriting")
    rep = {"started_at": dt.datetime.now().isoformat(timespec="seconds"), "commit_url": a.commit_url, "problems": []}
    rep["source_csv"], rep["source_csv_sha256"] = str(a.source_csv), sha256(a.source_csv)
    before = digest(a.results)
    rep["results_dir"], rep["results_files"] = str(a.results), len(before)

    # 1. pack (the trial's command; only --code-commit and -o differ)
    pack = [a.python, "-B", str(a.kit / "tools/pack_submission.py"), "--track", TRACK, "--results", str(a.results),
            "--dataset-root", str(a.dataset_root), "--code-commit", a.commit_url,
            "--eval-script", str(a.kit / "flash_chord/scripts/evaluate_policy.py"), "-o", str(out_csv)]
    p = run(pack, a.kit, out_dir / "pack.log")
    rep["pack_command"], rep["pack_exit"] = shlex.join(pack), p.returncode
    if p.returncode != 0 or not out_csv.is_file():
        raise SystemExit(f"pack failed (exit {p.returncode}); see {out_dir / 'pack.log'}")
    rep["csv"], rep["csv_sha256"], rep["csv_size"] = str(out_csv), sha256(out_csv), out_csv.stat().st_size
    rep["results_unchanged"] = digest(a.results) == before
    if not rep["results_unchanged"]:
        rep["problems"].append("results dir changed during packing")

    # 2. kit uploader validate_submission() + provenance
    v = [a.python, "-B", str(a.validator), str(out_csv), str(a.expected), a.commit_url, str(out_dir / "validate.json")]
    p = run(v, out_dir, out_dir / "validate.log")
    val = json.loads((out_dir / "validate.json").read_text()) if (out_dir / "validate.json").is_file() else {}
    rep["validate_command"], rep["validate_exit"], rep["validate"] = shlex.join(v), p.returncode, val
    rep["validate_ok"] = p.returncode == 0 and bool(val.get("ok"))
    if not rep["validate_ok"]:
        rep["problems"].append(f"validation failed: {val.get('problems')}")

    # 3. official metric_code scores == the source file's
    s = [a.python, "-B", str(a.scorer), "--csv", str(out_csv), "--out", str(out_dir / "metric_code.json"), "--episodes", "all"]
    p = run(s, out_dir, out_dir / "score.log")
    rep["score_command"], rep["score_exit"] = shlex.join(s), p.returncode
    if p.returncode != 0:
        raise SystemExit(f"scoring failed; see {out_dir / 'score.log'}")
    new, old = json.loads((out_dir / "metric_code.json").read_text()), json.loads(a.source_scores.read_text())
    rep["scores_overall_float32"], rep["source_overall_float32"] = new["overall_float32"], old["overall_float32"]
    rep["scores_overall_float64"], rep["source_overall_float64"] = new.get("overall_float64"), old.get("overall_float64")
    eq32 = all(new["overall_float32"][m] == old["overall_float32"][m] for m in METRICS)
    eq64 = all(new["overall_float64"][m] == old["overall_float64"][m] for m in METRICS)
    pe = [f"ep{e} {m}" for e, v in old["per_episode_float32"].items() for m in METRICS
          if new["per_episode_float32"].get(e, {}).get(m) != v[m]]
    pe_eq = not pe and set(new["per_episode_float32"]) == set(old["per_episode_float32"])
    rep.update(scores_equal_overall_float32=eq32, scores_equal_overall_float64=eq64, scores_equal_per_episode=pe_eq,
               per_episode_mismatches=pe, n_episodes_scored=len(new["per_episode_float32"]))
    if not (eq32 and eq64 and pe_eq):
        rep["problems"].append(f"scores differ from the source file: overall32 {eq32} overall64 {eq64} per-episode {pe[:5]}")

    # 4. row by row: only code_commit_url differs
    with open(a.source_csv, newline="") as f1, open(out_csv, newline="") as f2:
        r1, r2 = csv.reader(f1), csv.reader(f2)
        h1, h2 = next(r1), next(r2)
        rep["header_equal"] = h1 == h2
        iu = h2.index("code_commit_url")
        n = diff_other = 0
        old_urls, new_urls = set(), set()
        for x, y in zip(r1, r2, strict=True):
            n += 1
            old_urls.add(x[iu])
            new_urls.add(y[iu])
            if x[:iu] + x[iu + 1:] != y[:iu] + y[iu + 1:]:
                diff_other += 1
    rep.update(rows=n, rows_differing_outside_commit_url=diff_other, source_commit_urls=sorted(old_urls),
               new_commit_urls=sorted(new_urls),
               only_commit_url_differs=rep["header_equal"] and diff_other == 0 and new_urls == {a.commit_url})
    if not rep["only_commit_url_differs"]:
        rep["problems"].append(f"rows differ outside code_commit_url: {diff_other}; urls {sorted(new_urls)}")
    rep["ok"] = not rep["problems"]
    rep["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
    (out_dir / "repack_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("ok", "csv_sha256", "rows", "scores_overall_float32", "scores_equal_overall_float32",
                                          "scores_equal_per_episode", "validate_ok", "only_commit_url_differs",
                                          "results_unchanged", "problems")}, indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
