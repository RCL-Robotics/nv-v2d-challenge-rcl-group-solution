#!/usr/bin/env python3
"""Per-episode ADD-AUC of ONE uniform pipeline vs the released Tier 2 trajectories.

--pipeline lists variants in fallback order (first variant that has the episode wins), e.g.
    P1_temporal E2_reregister baseline
which means "post-processing P1 on E2 outputs; plain CARI4D where E2 was not run". No per-episode
choice based on the reference is made. Reads the CSVs written by compare_experiments.py
(stage CoCoNet) plus logs/eval_after_fix.csv for the baseline.
"""
import argparse
import glob

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", nargs="+", required=True)
    ap.add_argument("--logs", default="/mnt/nas/Group/hanzhou/v2d/logs")
    ap.add_argument("--metric", default="AUC")
    args = ap.parse_args()
    frames = []
    for f in glob.glob(f"{args.logs}/compare_*.csv"):
        d = pd.read_csv(f)
        frames.append(d[d["stage"] == "CoCoNet"][["episode", "variant", "AUC", "MPPE_cm", "pos_cm", "rot_deg"]])
    allv = pd.concat(frames).drop_duplicates(["episode", "variant"], keep="last")
    b = pd.read_csv(f"{args.logs}/eval_after_fix.csv")
    b = b[(b["stage"] == "coconet") & (b["align"] == "se3")][["episode", "AUC", "MPPE_cm", "pos_cm", "rot_deg"]].assign(variant="baseline")
    allv = pd.concat([allv[allv.variant != "baseline"], b])
    t2 = pd.read_csv(f"{args.logs}/eval_tier2.csv")
    t2 = t2[t2["align"] == "raw"].set_index("episode")
    rows = []
    for ep in range(30):
        got = None
        for v in args.pipeline:
            r = allv[(allv.episode == ep) & (allv.variant == v)]
            if len(r):
                got = (v, *r[["AUC", "MPPE_cm", "rot_deg"]].iloc[0].tolist())
                break
        v, auc, mppe, rot = got if got else ("-", float("nan"), float("nan"), float("nan"))
        rows.append({"ep": ep, "object": t2.loc[ep, "object"], "variant": v, "AUC": auc, "t2_AUC": t2.loc[ep, "AUC"],
                     "MPPE": mppe, "t2_MPPE": t2.loc[ep, "MPPE_cm"], "rot": rot, "beat": auc > t2.loc[ep, "AUC"]})
    d = pd.DataFrame(rows)
    print(d.round(3).to_string(index=False))
    print(f"\npipeline {args.pipeline}: beats Tier 2 AUC on {int(d.beat.sum())}/30 episodes; mean AUC {d.AUC.mean():.3f} vs Tier 2 {d.t2_AUC.mean():.3f}")


if __name__ == "__main__":
    main()
