#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-episode Track 2 results table + averages from an eval_all.py run (standard library only).

Metrics come from the official evaluator's own per-episode JSON (<results>/episode_XXXXXX.json, "metrics"), mapped to
the leaderboard with the kit's v2dlb/report_schema.py REPORT_METRIC_KEY (AUC = add_auc, SP-SR = spider_sr_uncentered,
MP-SR = maniptrans_sr, MPPE = mean_per_point_error_cm); eval_all's own chord_sr column is NOT the leaderboard SP-SR.
From <arm-dir>/eval_report.json it adds the evaluated checkpoint's iteration, how it was selected
(selected_by: "final.json (<reason>)" or "--iteration N"), the training stop reason (final_reason) and the export
diagnostics; an episode eval_all could not evaluate keeps eval_all's status (e.g. missing_checkpoint).
Optionally (--floor) adds a per-episode "object never moves" proxy floor for context (JSON lines with ep,
static_auc and static_mppe_cm); the submitted numbers do not depend on it.

usage: summarize.py --arm-dir <eval view dir> --results <results dir> --out <csv> --kit <kit dir> [--floor FILE]
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import statistics
from pathlib import Path

FALLBACK = {
    "add_auc": "add_auc",
    "spider_sr": "spider_sr_uncentered",
    "maniptrans_sr": "maniptrans_sr",
    "mppe_cm": "mean_per_point_error_cm",
}
OURS = [("auc", "add_auc"), ("sp_sr", "spider_sr"), ("mp_sr", "maniptrans_sr"), ("mppe_cm", "mppe_cm")]


def kit_keys(kit: Path) -> tuple[dict, str]:
    try:
        for node in ast.parse((kit / "v2dlb/report_schema.py").read_text()).body:
            targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
            if any(getattr(t, "id", None) == "REPORT_METRIC_KEY" for t in targets):
                table = ast.literal_eval(node.value)
                return {name: table[name] for name in FALLBACK}, "kit REPORT_METRIC_KEY"
    except Exception as error:  # noqa: BLE001
        return dict(FALLBACK), f"fallback ({error!r})"
    return dict(FALLBACK), "fallback (REPORT_METRIC_KEY not found)"


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kit", type=Path, required=True, help="extracted submission kit (for REPORT_METRIC_KEY)")
    ap.add_argument("--floor", type=Path, help="optional static-object proxy floor (JSON lines)")
    a = ap.parse_args()
    keys, source = kit_keys(a.kit)
    report = json.loads((a.arm_dir / "eval_report.json").read_text())
    floor = {}
    if a.floor is not None and a.floor.is_file():
        for line in a.floor.read_text().splitlines():
            if line.startswith("{"):
                r = json.loads(line)
                floor[int(r["ep"])] = r
    rows = []
    for e in sorted(report["episodes"], key=lambda r: int(r["episode"])):
        ep = int(e["episode"])
        res = a.results / f"episode_{ep:06d}.json"
        m = json.loads(res.read_text()).get("metrics", {}) if res.is_file() else {}
        status = e.get("status")
        row = {
            "episode": ep,
            "object": e.get("object"),
            "status": status if m or status not in (None, "evaluated") else "missing_result",
            "iteration": e.get("iteration"),
            "selected_by": e.get("selected_by"),
            "final_reason": e.get("final_reason"),
        }
        for ours, name in OURS:
            row[ours] = num(m.get(keys[name]))
        row["chord_sr_diag"] = num(m.get("chord_sr"))
        f = floor.get(ep, {})
        row["static_auc_proxy"] = f.get("static_auc")
        row["auc_minus_static"] = (
            None if row["auc"] is None or f.get("static_auc") is None else round(row["auc"] - f["static_auc"], 4)
        )
        row["static_mppe_cm_proxy"] = f.get("static_mppe_cm")
        row["export_max_rel_diff"] = num(e.get("export_max_rel_diff"))
        row["export_stock_verified"] = e.get("export_stock_verified")
        row["source_checkpoint"] = e.get("source_checkpoint")
        row["checkpoint_sha256"] = e.get("checkpoint_sha256") or e.get("evaluated_checkpoint_sha256")
        row["sequence"] = e.get("sequence")
        rows.append(row)
    cols = list(rows[0].keys())
    ok = [r for r in rows if r["auc"] is not None]

    def agg(fn, k):
        v = [r[k] for r in ok if r[k] is not None]
        return round(fn(v), 4) if v else None

    mean = {c: "" for c in cols}
    med = {c: "" for c in cols}
    mean.update(episode="MEAN", object=f"n={len(ok)}/{len(rows)}", status=f"metric keys: {source}")
    med.update(episode="MEDIAN", object=f"n={len(ok)}/{len(rows)}")
    for k in ("auc", "sp_sr", "mp_sr", "mppe_cm", "chord_sr_diag", "static_auc_proxy", "auc_minus_static",
              "static_mppe_cm_proxy"):  # fmt: skip
        mean[k] = agg(statistics.fmean, k)
        med[k] = agg(statistics.median, k)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
        w.writerow(mean)
        w.writerow(med)
    print(f"wrote {a.out} ({len(rows)} episodes, {len(ok)} with metrics; keys from {source})")
    print(f"MEAN  AUC {mean['auc']}  SP-SR {mean['sp_sr']}  MP-SR {mean['mp_sr']}  MPPE {mean['mppe_cm']} cm")
    print(f"MEDIAN AUC {med['auc']}  MPPE {med['mppe_cm']} cm")


if __name__ == "__main__":
    main()
