#!/usr/bin/env python3
"""Extract the Tier 2 (noisy) human SOMA-X parameters and object pose/visibility from the parquets into npz (CPU, pyarrow).
Usage: extract_t2_human.py <tier_dir> <out_dir>"""
import os, sys, numpy as np, pyarrow.parquet as pq
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
cols = {"pose": "observation.human.pose", "transl": "observation.human.translation", "identity": "observation.human.identity_coeffs",
        "scale": "observation.human.scale_params", "bone": "observation.human.bone_length_flexibles",
        "obj_pose": "observation.object.pose", "obj_visible": "observation.object.visible"}
for i in range(30):
    t = pq.read_table(f"{src}/data/chunk-000/episode_{i:06d}.parquet")
    d = {k: np.stack(t.column(c).to_numpy(zero_copy_only=False)) if k != "obj_visible" else t.column(c).to_numpy() for k, c in cols.items()}
    np.savez(f"{out}/episode_{i:06d}.npz", **d); print(i, {k: v.shape for k, v in d.items()} if i == 0 else len(d["pose"]))
