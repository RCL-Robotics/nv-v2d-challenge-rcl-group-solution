#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""summarize.py: own-input metrics of an eval_tier{1,2,3}.sh run, from the evaluator's results JSONs.

  summarize.py --tier N --out <eval dir> --set <set_items json> [--expect-pinned]

Reads <out>/t<k>/results/episode_*.json (the kit evaluator's own report of each rollout) and writes
<out>/summary.tsv and <out>/summary.json:
  AUC = metrics.add_auc, SP-SR = metrics.spider_sr_uncentered, MP-SR = metrics.maniptrans_sr,
  MPPE = metrics.mean_per_point_error_cm,
each measured against the reference the evaluator ran on, i.e. the policy's own input (the task parquet its
checkpoint names). These are not the leaderboard scores of Tier 2 or Tier 3 (those use a hidden clean reference);
nothing here reads or computes a clean score. --expect-pinned also checks that each rollout evaluated the pinned file
of the counted CSV (checkpoint_sha256 = the set record's evaluated file). Standard library only.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.dont_write_bytecode = True
KEYS = {"AUC": "add_auc", "SP-SR": "spider_sr_uncentered", "MP-SR": "maniptrans_sr", "MPPE_cm": "mean_per_point_error_cm"}
FORBIDDEN = ("clean", "LOCAL_NOT_FOR_UPLOAD", "tier_1_multiview_caption")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--set", type=Path, required=True)
    ap.add_argument("--expect-pinned", action="store_true")
    a = ap.parse_args()
    items = {int(e["episode"]): e for e in json.loads(a.set.read_text())["episodes"]}
    rows, problems = [], []
    for f in sorted(a.out.glob("t[0-9]*/results/episode_*.json")):
        if any(x in str(f) for x in FORBIDDEN):
            raise SystemExit(f"refusing {f}")
        d = json.loads(f.read_text())
        trial = int(re.match(r"t(\d+)", f.parent.parent.name).group(1))
        ep = int(re.search(r"episode_(\d+)", f.name).group(1))
        m = d.get("metrics", {})
        row = {"trial": trial, "episode": ep, **{k: m.get(v) for k, v in KEYS.items()},
               "checkpoint_sha256": d.get("checkpoint_sha256"), "reference_parquet": d.get("parquet"),
               "start_frame": d.get("start_frame"), "world_count": d.get("world_count"), "reset_mode": d.get("reset_mode")}
        if a.tier in (2, 3) and any(x in str(row["reference_parquet"]) for x in FORBIDDEN):
            problems.append(f"t{trial} ep{ep}: reference {row['reference_parquet']} is not an own input")
        ent = items.get(ep, {})
        if a.expect_pinned and ent:
            want = ent["export"]["sha256"]
            if row["checkpoint_sha256"] != want:
                problems.append(f"t{trial} ep{ep}: evaluated {row['checkpoint_sha256']}, pinned {want}")
        if (row["start_frame"], row["world_count"], row["reset_mode"]) != (0, 1, "explicit"):
            problems.append(f"t{trial} ep{ep}: protocol {row['start_frame']}/{row['world_count']}/{row['reset_mode']}")
        rows.append(row)
    if not rows:
        raise SystemExit(f"no results under {a.out}/t*/results")
    cols = ["trial", "episode", *KEYS, "checkpoint_sha256", "reference_parquet"]
    lines = ["\t".join(cols)] + ["\t".join("" if r[c] is None else str(r[c]) for c in cols) for r in rows]
    (a.out / "summary.tsv").write_text("\n".join(lines) + "\n")
    eps = sorted({r["episode"] for r in rows})
    trials = sorted({r["trial"] for r in rows})
    per_ep = {}
    print(f"Tier {a.tier}: {len(rows)} evaluations, {len(trials)} trial(s), {len(eps)} episode(s)  [own input, kit evaluator]")
    print(f"{'ep':>3} {'n':>2} {'AUC mean':>9} {'AUC sd':>7} {'SP pass':>7} {'MP pass':>7} {'MPPE cm':>8}  recorded")
    for e in eps:
        rs = [r for r in rows if r["episode"] == e]
        auc = [r["AUC"] for r in rs]
        rec = ""
        ent = items.get(e, {})
        t20 = (ent.get("evaluation") or {}).get("trials20")
        if t20:
            rec = (f"20 trials: AUC {t20['auc_mean']:.4f} (sd {t20['auc_sd']:.4f}), SP {t20['sp_pass']}/20, "
                   f"MP {t20['mp_pass']}/20, MPPE {t20['mppe_mean_cm']:.3f}")
        own = ent.get("own_input_of_packed_rollout")
        if own:
            rec = "packed rollout: " + ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                                                 for k, v in own.items() if not isinstance(v, (dict, list)))[:160]
        per_ep[e] = {"n": len(rs), "auc_mean": st.mean(auc), "auc_sd": st.stdev(auc) if len(auc) > 1 else 0.0,
                     "sp_pass": sum(1 for r in rs if r["SP-SR"] == 1.0), "mp_pass": sum(1 for r in rs if r["MP-SR"] == 1.0),
                     "mppe_mean": st.mean(r["MPPE_cm"] for r in rs)}
        p = per_ep[e]
        print(f"{e:>3} {p['n']:>2} {p['auc_mean']:>9.5f} {p['auc_sd']:>7.5f} {p['sp_pass']:>4}/{p['n']:<2} "
              f"{p['mp_pass']:>4}/{p['n']:<2} {p['mppe_mean']:>8.3f}  {rec}")
    per_trial = {}
    for t in trials:
        rs = [r for r in rows if r["trial"] == t]
        if len(rs) == len(eps):
            per_trial[t] = {k: st.mean(r[k] for r in rs) for k in KEYS}
    if len(eps) > 1 and per_trial:
        print("per-trial means over the episodes above (own input):")
        for t, v in per_trial.items():
            print(f"  t{t:02d}: " + "  ".join(f"{k} {x:.4f}" for k, x in v.items()))
    (a.out / "summary.json").write_text(json.dumps({"tier": a.tier, "rows": rows, "per_episode": per_ep,
                                                     "per_trial": per_trial, "problems": problems}, indent=1))
    for p in problems:
        print("PROBLEM:", p)
    print(f"written {a.out / 'summary.tsv'} and summary.json; problems: {len(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
