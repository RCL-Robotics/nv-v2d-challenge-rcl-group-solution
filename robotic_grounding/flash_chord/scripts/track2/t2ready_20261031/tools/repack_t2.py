#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""repack_t2.py: re-pack the Tier-2 R6-CLOSEST results (round-6 phase-F trial 9) with a real code-commit URL, without
evaluating anything.

Only --code-commit and -o of the round-6 kit pack command change (same results folder, same frame references, same
dataset root, same evaluator script). Then:
  1. the results and frame-reference folders are unchanged by packing (sha256 of every file before and after);
  2. the kit uploader's offline check: validate_submission() of submit/v2d_submit.py with track_2_tier2, called
     directly (no Kaggle API, no credentials); plus provenance: row count = the public sample, uniform provenance
     columns, the commit URL on every row, one checkpoint digest per episode equal to set_items.json's evaluated file;
  3. every row of the new file equals the source file's row except the code_commit_url field (pose columns identical).
The packer also prints a clean-reference "locally computed" line; it is kept out of pack.log and written only to
pack_stdout_LOCAL_NOT_FOR_UPLOAD.log (report only; it decides nothing). Writes only under --out-dir. Run with the kit
env python (-I -B; never writes bytecode into the kit):

  <kit python> -I -B repack_t2.py --kit KIT --dataset-root DATA --pack-dir R6_PACK/r6_closest \
      --source-csv R6_PACK/r6_closest/packed_LOCAL_NOT_FOR_UPLOAD.csv --items set/set_items.json \
      --commit-url https://github.com/<owner>/<repo>/commit/<40-hex sha> --out-dir DIR --name FILE.csv
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
import os  # noqa: E402
import re  # noqa: E402
import shlex  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

TRACK = "track_2_tier2"
URL_RE = re.compile(r"^https://github\.com/[^/]+/[^/]+/commit/[0-9a-f]{40}$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(d: Path) -> dict:
    return {str(f.relative_to(d)): sha256(f) for f in sorted(d.rglob("*")) if f.is_file()}


def pose_columns_sha(csvp: Path, npose: int) -> str:
    h = hashlib.sha256()
    with csvp.open() as fp:
        for line in fp:
            h.update((",".join(line.rstrip("\n").split(",")[: 1 + npose]) + "\n").encode())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("kit", "dataset-root", "pack-dir", "source-csv", "items", "out-dir"):
        ap.add_argument(f"--{k}", type=Path, required=True)
    ap.add_argument("--commit-url", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--python", default=sys.executable, help="kit env python for the packer")
    a = ap.parse_args()
    if not URL_RE.match(a.commit_url):
        raise SystemExit(f"not a full-sha GitHub commit URL: {a.commit_url}")
    out_dir = a.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / a.name
    if out_csv.exists():
        raise SystemExit(f"{out_csv} exists; not overwriting")
    kit = a.kit.resolve()
    results, frefs = a.pack_dir / "results", a.pack_dir / "frame_refs"
    rep = {"started_at": dt.datetime.now().isoformat(timespec="seconds"), "track": TRACK, "commit_url": a.commit_url,
           "problems": [], "source_csv": str(a.source_csv), "source_csv_sha256": sha256(a.source_csv),
           "items": str(a.items), "items_sha256": sha256(a.items)}
    before_r, before_f = digest(results), digest(frefs)
    kit_before = digest(kit)
    rep["results_files"], rep["frame_reference_files"] = len(before_r), len(before_f)

    # 1. pack: the round-6 command with only --code-commit and -o changed
    pack = [a.python, "-I", "-B", str(kit / "tools/pack_submission.py"), "--track", TRACK, "--results", str(results),
            "--dataset-root", str(a.dataset_root), "--frame-references", str(frefs), "--code-commit", a.commit_url,
            "--eval-script", str(kit / "flash_chord/scripts/evaluate_policy.py"), "-o", str(out_csv)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run(pack, cwd=kit, env=env, capture_output=True, text=True, timeout=3600)
    raw = f"$ {shlex.join(pack)}\n{p.stdout}\n{p.stderr}"
    (out_dir / "pack_stdout_LOCAL_NOT_FOR_UPLOAD.log").write_text(raw)
    kept = [ln if not ln.startswith("locally computed:") else "locally computed: [withheld here: clean-reference "
            "line, see pack_stdout_LOCAL_NOT_FOR_UPLOAD.log]" for ln in raw.splitlines()]
    (out_dir / "pack.log").write_text("\n".join(kept) + "\n")
    rep["pack_command"], rep["pack_exit"] = shlex.join(pack), p.returncode
    if p.returncode != 0 or not out_csv.is_file():
        raise SystemExit(f"pack failed (exit {p.returncode}); see {out_dir / 'pack.log'}")
    rep["csv"], rep["csv_sha256"], rep["csv_bytes"] = str(out_csv), sha256(out_csv), out_csv.stat().st_size
    rep["results_unchanged"] = digest(results) == before_r
    rep["frame_references_unchanged"] = digest(frefs) == before_f
    rep["kit_unchanged"] = digest(kit) == kit_before
    for k in ("results_unchanged", "frame_references_unchanged", "kit_unchanged"):
        if not rep[k]:
            rep["problems"].append(f"{k} is False")

    # 2. the uploader's offline validate_submission() + provenance
    sys.path.insert(0, str(kit))
    spec = importlib.util.spec_from_file_location("kit_v2d_submit", kit / "submit/v2d_submit.py")
    up = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(up)
    from v2dlb import object_pose as op  # noqa: PLC0415
    from v2dlb import report_schema as rs  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415

    config = json.loads((kit / "config/tracks.json").read_text())
    try:
        frame = up.validate_submission(out_csv, TRACK, config)
        rep["uploader_validate_submission"] = "pass"
    except SystemExit as err:
        rep["uploader_validate_submission"] = f"FAIL: {err}"
        rep["problems"].append("uploader validate_submission failed")
        frame = pd.read_csv(out_csv)
    n_sample = len(pd.read_csv(kit / "data/track_2_sample_submission.csv", usecols=[rs.ROW_ID_COLUMN], dtype=str))
    rep["rows"], rep["sample_rows"] = int(len(frame)), n_sample
    if rep["rows"] != n_sample:
        rep["problems"].append(f"{rep['rows']} rows != sample {n_sample}")
    uniform = {c: sorted(frame[c].astype(str).str.strip().unique()) for c in rs.UNIFORM_PROVENANCE_COLUMNS}
    rep["uniform_columns"] = uniform
    if any(len(v) != 1 for v in uniform.values()):
        rep["problems"].append("a provenance column is not uniform")
    if uniform.get("code_commit_url") != [a.commit_url]:
        rep["problems"].append(f"code_commit_url {uniform.get('code_commit_url')} != {a.commit_url}")
    eps = frame["row_id"].astype(str).str.split("/").str[1]
    per_ep = frame.groupby(eps)["checkpoint_sha256"].agg(lambda s: sorted(s.astype(str).str.strip().unique()))
    items = json.loads(a.items.read_text())
    want = {f"e{it['episode']:06d}": it["export"]["sha256"] for it in items["episodes"]}
    got = {k: v for k, v in per_ep.items()}
    rep["per_episode_checkpoint_ok"] = sorted(got) == sorted(want) and all(got[k] == [want[k]] for k in want)
    rep["distinct_checkpoints"] = len({x for v in got.values() for x in v})
    if not rep["per_episode_checkpoint_ok"]:
        bad = [k for k in want if got.get(k) != [want[k]]]
        rep["problems"].append(f"per-episode checkpoint digests differ from set_items: {bad[:6]}")
    npose = len(op._POSE_COLUMNS)
    rep["pose_columns_sha256"] = pose_columns_sha(out_csv, npose)
    rep["source_pose_columns_sha256"] = pose_columns_sha(a.source_csv, npose)
    if rep["pose_columns_sha256"] != rep["source_pose_columns_sha256"]:
        rep["problems"].append("pose columns differ from the source file")

    # 3. row by row: only code_commit_url differs
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
    rep.update(rows_compared=n, rows_differing_outside_commit_url=diff_other, source_commit_urls=sorted(old_urls),
               new_commit_urls=sorted(new_urls),
               only_commit_url_differs=rep["header_equal"] and diff_other == 0 and new_urls == {a.commit_url})
    if not rep["only_commit_url_differs"]:
        rep["problems"].append(f"rows differ outside code_commit_url: {diff_other}; urls {sorted(new_urls)}")
    rep["ok"] = not rep["problems"]
    rep["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
    (out_dir / "repack_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("ok", "csv_sha256", "rows", "sample_rows", "uploader_validate_submission",
                                          "per_episode_checkpoint_ok", "distinct_checkpoints", "only_commit_url_differs",
                                          "rows_differing_outside_commit_url", "pose_columns_sha256",
                                          "results_unchanged", "frame_references_unchanged", "kit_unchanged", "problems")},
                     indent=1))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
