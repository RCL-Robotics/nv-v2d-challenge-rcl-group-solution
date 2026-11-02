#!/usr/bin/env python3
"""Export the Tier 1 reference arrays that the Tier 3 scripts read, from the Tier 1 parquet files.

The CARI4D image has no parquet engine, so the reference data are pre-exported to npz on the host (any python with
pandas + pyarrow). This script reproduces the two inline commands that wrote them (2026-10-06 and 2026-10-15):

  ref_poses/<ep>.npz     frame_index (T,), pose_xyz_wxyz (T, 7), visible (T,)   -- reference object poses, all frames
  ref_human_f0/<ep>.npz  pose (231,), transl (3,), identity (45,), scale (68,), bone (6,)  -- reference human, frame 0 only

Who reads what:
  - Tier 3 pipeline (organiser-approved data only):
      ref_human_f0  -> firstframe_align.py, firstframe_pnp.py, postproc_feet_floor.py (--ref-foot-height),
                       postproc_joint_camera*.py, agent/export_inputs.py (frame0_reference_human.npz)
      ref_poses     -> frame 0 only: tier2_refine/t3_frame0.py (frame-0 object pose)
  - GT diagnostics only (never feed a pipeline output): ref_poses, all frames -> eval_official.py, eval_object_pose.py,
    agent/score_runs.py, the diagnostic half of t3_frame0.py, support/score_snap.py, support/contact.py --gt.

Usage: export_reference.py [--dataset <tier_1_multiview_caption dir>] [--out <dir holding ref_poses/ ref_human_f0/>]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

V2D = "/mnt/nas/Group/hanzhou/v2d"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=f"{V2D}/dataset/track_2/tier_1_multiview_caption")
    ap.add_argument("--out", default=V2D)
    args = ap.parse_args()
    files = sorted(glob.glob(f"{args.dataset}/data/chunk-000/episode_*.parquet"))
    os.makedirs(f"{args.out}/ref_poses", exist_ok=True)
    os.makedirs(f"{args.out}/ref_human_f0", exist_ok=True)
    for f in files:
        i = int(f[-14:-8])
        d = pd.read_parquet(f, columns=["frame_index", "observation.object.pose", "observation.object.visible"])
        np.savez(f"{args.out}/ref_poses/episode_{i:06d}.npz", frame_index=d["frame_index"].to_numpy(),
                 pose_xyz_wxyz=np.stack(d["observation.object.pose"].to_numpy()),
                 visible=np.stack(d["observation.object.visible"].to_numpy()).reshape(-1))
        h = pd.read_parquet(f).iloc[0]  # frame 0 only
        np.savez(f"{args.out}/ref_human_f0/episode_{i:06d}.npz", pose=np.asarray(h["observation.human.pose"]),
                 transl=np.asarray(h["observation.human.translation"]),
                 identity=np.asarray(h["observation.human.identity_coeffs"]),
                 scale=np.asarray(h["observation.human.scale_params"]),
                 bone=np.asarray(h["observation.human.bone_length_flexibles"]))
    print(len(files), "episodes exported to", args.out)


if __name__ == "__main__":
    main()
