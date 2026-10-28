#!/usr/bin/env python3
"""screen_score.py: per-world object metrics of official-evaluator rollout parquets (T1improve confirmation, the
world_count=64 collapse screen of PREREG_t1improve v2 section 8.4 (f); CONFIRM_RULES.md section 5).

The maths is the kit's own (v2d_submission_kit/v2dlb/object_pose.py, "transcribed from flash_chord/evaluation/metrics.py";
the generated Kaggle metric inlines the same lines), applied to ONE world's slice of the evaluator's parquet at a time.
ADD vertices: the kit's canonical sample (v2dlb/vertices.py: the released visual mesh, 500 points, seed 0). The
reference is the evaluator's own reference columns in the parquet (identical in every world; checked).

Read only. Run with the kit env and -B (the kit must never get bytecode):
  conda_envs/v2dkit/bin/python -B screen_score.py --object <mesh name> --out <json> <parquet> [<parquet> ...]
Output per parquet: world_count, per-world AUC / SP / MP / MPPE, the pooled AUC over all worlds (the evaluator JSON's
add_auc pools the same way), and the parquet sha256.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

KIT = Path("/root/autodl-tmp/haoran.zhang/v2d_submission_kit_src/v2d_submission_kit")
MESH_ROOT = Path("/root/autodl-tmp/haoran.zhang/data/video_to_data_challenge/track_2/tier_1_multiview_caption/mesh")
sys.path.insert(0, str(KIT))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from v2dlb import object_pose as op  # noqa: E402
from v2dlb.vertices import mesh_path_for, sample_vertices  # noqa: E402

ACH = ["achieved_position_x", "achieved_position_y", "achieved_position_z", "achieved_quaternion_x",
       "achieved_quaternion_y", "achieved_quaternion_z", "achieved_quaternion_w"]
REF = ["reference_position_x", "reference_position_y", "reference_position_z", "reference_quaternion_x",
       "reference_quaternion_y", "reference_quaternion_z", "reference_quaternion_w"]


def sha256(file) -> str:
    h = hashlib.sha256()
    with open(file, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensors(pp: str):
    t = pq.read_table(pp, columns=["environment_id", "step", "object_body_id", "environment_valid", *ACH, *REF]).to_pandas()
    t = t.sort_values(["step", "environment_id", "object_body_id"], kind="mergesort")
    steps, worlds, bodies = (np.unique(t[c].to_numpy()) for c in ("step", "environment_id", "object_body_id"))
    if len(t) != len(steps) * len(worlds) * len(bodies):
        raise ValueError(f"{pp}: grid not rectangular ({len(t)} rows for {len(steps)} x {len(worlds)} x {len(bodies)})")
    shape = (len(steps), len(worlds), len(bodies))
    ach = t[ACH].to_numpy(dtype=np.float64).reshape(*shape, 7)
    ref = t[REF].to_numpy(dtype=np.float64).reshape(*shape, 7)
    valid = t["environment_valid"].to_numpy().reshape(shape)
    ref_spread = float(np.max(np.abs(ref - ref[:, :1]))) if len(worlds) > 1 else 0.0
    ach = op._normalise_quaternions(ach, "achieved", ValueError)
    ref1 = op._normalise_quaternions(ref[:, :1], "reference", ValueError)
    return ach, ref1, worlds.tolist(), bodies.astype(np.int64), valid, ref_spread


def score(pp: str, verts: np.ndarray) -> dict:
    ach, ref, worlds, bodies, valid, ref_spread = tensors(pp)
    v = np.stack([verts for _ in bodies])
    per = []
    for i, w in enumerate(worlds):
        a = ach[:, i:i + 1]
        add = op._object_add(a, ref, v)
        per.append({"world": int(w), "auc": op._add_auc(add), "mean_add_m": float(add.mean()),
                    "sp_sr": float(np.mean(op._spider_success(a, ref))),
                    "mp_sr": float(np.mean(op._maniptrans_success(a, ref, bodies))),
                    "mppe_cm": float(op._mean_per_point_error_cm(a, ref)),
                    "valid": bool(valid[:, i].all())})  # fmt: skip
    pooled = op._add_auc(op._object_add(ach, ref, v))
    return {"parquet": str(pp), "parquet_sha256": sha256(pp), "world_count": len(worlds), "steps": int(ach.shape[0]),
            "bodies": len(bodies), "reference_spread_over_worlds": ref_spread, "pooled_auc": pooled,
            "per_world": per}  # fmt: skip


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--object", required=True, help="mesh folder name under the Tier-1 mesh root (e.g. squeegee)")
    p.add_argument("--out", required=True)
    p.add_argument("parquets", nargs="+")
    a = p.parse_args()
    mesh = mesh_path_for(MESH_ROOT, a.object)
    verts = sample_vertices(mesh)
    out = {"object": a.object, "mesh": str(mesh), "mesh_sha256": sha256(mesh), "vertex_count": int(verts.shape[0]),
           "kit_object_pose_sha256": sha256(KIT / "v2dlb/object_pose.py"), "kit_vertices_sha256": sha256(KIT / "v2dlb/vertices.py"),
           "scorer_sha256": sha256(Path(__file__).resolve()), "results": [score(pp, verts) for pp in a.parquets]}  # fmt: skip
    tmp = f"{a.out}.{os.getpid()}.tmp"
    with open(tmp, "w") as fp:
        json.dump(out, fp, indent=1)
        fp.write("\n")
    os.replace(tmp, a.out)
    for r in out["results"]:
        aucs = [x["auc"] for x in r["per_world"]]
        print(f"{Path(r['parquet']).name}: worlds {r['world_count']} pooled AUC {r['pooled_auc']:.6f} per-world min "
              f"{min(aucs):.6f} max {max(aucs):.6f} world0 {aucs[0]:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
