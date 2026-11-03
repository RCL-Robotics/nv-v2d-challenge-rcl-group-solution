#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""make_final_episodes.py: write scripts/final/FINAL_EPISODES.tsv, one row per counted checkpoint (3 tiers x 30
episodes), from the three committed set records, and check it against the machine's files.

  make_final_episodes.py --out FINAL_EPISODES.tsv [--csv 1=<counted T1 csv> --csv 2=... --csv 3=...]

Columns: tier, episode, object, item, input, input_root, task_parquet_sha256, frame_reference_sha256, training_commits
(as recorded, in stage order), training_code_in_this_history, stages, training_checkpoint_sha256, export_commit,
evaluated_file_sha256 (= the checkpoint_sha256 of that episode's rows in the counted CSV), evaluated_file, counted_rollout_sha256,
checks. task.parquet sha256: from the record (Tiers 2, 3) or computed from <MOTION> (Tier 1, whose record names the
parquet by path only); `checks` says what was verified on this machine: parquet (file sha256 = column), csv (the
counted CSV's checkpoint_sha256 for the episode = evaluated_file_sha256; the CSV is only read, never scored).
Placeholders: <ME>, <ZX>, <ZX5090>, <ZXWS>, <MOTION>, <RUNS>, <T1_EXPORTS> as in the set records.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plan import FULL, SETS  # noqa: E402

ROOTS = {"<ME>": os.environ.get("ROOT_ME", "/root/autodl-tmp/haoran.zhang"),
         "<ZX>": os.environ.get("ROOT_ZX", "/root/autodl-tmp/zuxing.lu")}
ROOTS["<MOTION>"] = os.environ.get("T1_MOTION", ROOTS["<ME>"] + "/data/human_motion_data")


def real(s: str) -> Path:
    for k, v in ROOTS.items():
        s = s.replace(k, v)
    return Path(s)


def sha(p: Path) -> str | None:
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def k7(text: str) -> str:
    return re.match(r"\s*([0-9a-f]{7})", text).group(1)


def csv_digests(path: str) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    with open(path, newline="") as f:
        r = csv.reader(f)
        hdr = next(r)
        ic, ir = hdr.index("checkpoint_sha256"), hdr.index("row_id")
        for row in r:
            ep = int(re.search(r"/e(\d+)/", row[ir]).group(1))
            out.setdefault(ep, set()).add(row[ic])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--csv", action="append", default=[])
    a = ap.parse_args()
    counted = {int(x.split("=", 1)[0]): x.split("=", 1)[1] for x in a.csv}
    digests = {t: csv_digests(p) for t, p in counted.items()}
    cols = ["tier", "episode", "object", "item", "input", "input_root", "task_parquet_sha256", "frame_reference_sha256",
            "training_commits", "training_code_in_this_history", "stages", "training_checkpoint_sha256", "export_commit",
            "evaluated_file_sha256", "evaluated_file", "counted_rollout_sha256", "checks"]
    rows = []
    for tier in (1, 2, 3):
        data = json.loads(SETS[tier].read_text())
        for e in data["episodes"]:
            ep = int(e["episode"])
            if tier == 1:
                codes = [k7(s["code"]) for s in e["stages"]]
                parquet = f"<MOTION>/whole_body/soma/sequence_id={e['sequence_id']}/robot_name=g1/data.parquet"
                rec_pq = None
                r = dict(object=e["object"], item=e["item"], input="tier1", input_root="<MOTION>", frame="-",
                         stages=str(len(e["stages"])), ckpt=e["checkpoint"]["model_sha256"],
                         export=k7(e["export"]["exported_by"]), evald=e["export"]["sha256"],
                         efile=("<RUNS>/" if e["export"]["root"] == "RUNS" else "<T1_EXPORTS>/") + e["export"]["file"],
                         rollout=e["trial05"]["results_parquet_sha256"])
            elif tier == 2:
                codes = [k7(e["train"]["code_commit"])]
                parquet, rec_pq = e["parquet"], e["parquet_sha256"]
                r = dict(object=e["sequence_id"].split("_", 2)[2], item=e["job"], input=e["input"], input_root=e["data_root"],
                         frame=e["evaluated"]["frame_reference_file_sha256"], stages=str(len(e["train"]["stages"])),
                         ckpt=e["checkpoint"]["sha256"], export=k7(e["export"]["code_commit"]), evald=e["export"]["sha256"],
                         efile=e["export"]["file"], rollout=e["evaluated"]["rollout_sha256"])
            else:
                codes = [k7(e["train"]["code_commit"])]
                parquet, rec_pq = e["input"]["parquet"], e["input"]["parquet_sha256"]
                r = dict(object=e["object"], item=f"{e['kind']} model_{e['checkpoint']['iteration']}"
                         + (" (palm 0.25)" if e["mitigated"] else ""), input="f0",
                         input_root="<ME>/data/human_motion_data_tier3_f0", frame=e["input"]["frame_ref_sha256"],
                         stages="+".join(s["stage"] for s in e["train"]["stages"]), ckpt=e["checkpoint"]["sha256"],
                         export=k7(e["train"]["code_commit"]), evald=e["export"]["sha256"], efile=e["export"]["file"],
                         rollout=e["rollout"]["results_parquet_sha256"])
            checks = []
            got = sha(real(parquet))
            if got is None:
                pq_sha = rec_pq or ""
                checks.append("parquet not on this machine")
            elif rec_pq and got != rec_pq:
                pq_sha = rec_pq
                checks.append(f"parquet DIFFERS ({got[:12]})")
            else:
                pq_sha = got
                checks.append("parquet ok" if rec_pq else "parquet sha computed")
            if tier in digests:
                d = digests[tier].get(ep, set())
                checks.append("csv ok" if d == {r["evald"]} else f"csv MISMATCH {sorted(d)}")
            mapped = "->".join(FULL[c][:7] for c in codes)
            rows.append([str(tier), f"{ep:02d}", r["object"], r["item"], r["input"], r["input_root"], pq_sha, r["frame"],
                         "->".join(codes), mapped, r["stages"], r["ckpt"], r["export"] + (f" (= {FULL[r['export']][:7]})"
                         if FULL[r["export"]][:7] != r["export"] else ""), r["evald"], r["efile"], r["rollout"],
                         "; ".join(checks)])
    a.out.write_text("\t".join(cols) + "\n" + "".join("\t".join(x) + "\n" for x in rows))
    bad = [x for x in rows if "DIFFERS" in x[-1] or "MISMATCH" in x[-1]]
    print(f"[final_episodes] {len(rows)} rows -> {a.out}; counted CSVs checked: {sorted(counted)}; problems: {len(bad)}")
    for x in bad:
        print("  PROBLEM", x[0], x[1], x[-1])
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
