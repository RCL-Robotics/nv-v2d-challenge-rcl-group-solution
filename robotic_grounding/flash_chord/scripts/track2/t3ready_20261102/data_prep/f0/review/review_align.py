#!/usr/bin/env python
"""review_align.py -- independent re-derivation (reviewer) of the f0-vs-opus hand-object gap for a few episodes.
CPU only. Reads only the two data roots' parquets + object meshes and align/dump/epNN.npz (their contact sets).
Independent parts: human joints from the parquet payload (not their SOMA FK), object pose from the parquet columns,
EXACT point-to-triangle distance (trimesh.proximity) instead of a vertex/face-centre KD-tree, and two other contact sets:
  HCA_OPUS  = retarget hand_contact_active on the opus parquet (the track the human was reconstructed with)
  HCA_UNION = hand_contact_active on either parquet
  THEIRS    = their union contact set C from dump (to check the distance computation only)
Writes nothing except stdout.
"""
import os, sys, json, pickle, importlib.util
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np

ME = Path("/root/autodl-tmp/haoran.zhang")
ROOT = {"f0": ME / "data/human_motion_data_tier3_f0/whole_body/soma", "opus": ME / "data/human_motion_data_tier3_opus/whole_body/soma"}
DUMP = ME / "tmp_analysis/tier3/f0/align/dump"


def q2m(q):
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def read(pqp):
    import pyarrow.parquet as pq
    r = pq.read_table(pqp, columns=["object_body_position", "object_body_wxyz", "hand_contact_active", "source_payload",
                                    "source_joint_names", "ee_pose_w"]).to_pylist()[0]
    pay = pickle.loads(r["source_payload"])
    q = np.asarray(r["object_body_wxyz"], np.float64)[:, 0]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return dict(p=np.asarray(r["object_body_position"], np.float64)[:, 0], R=q2m(q),
                hca=np.asarray(r["hand_contact_active"], np.float64), J=np.asarray(pay["soma_joints"], np.float64),
                names=list(r["source_joint_names"]), ee=np.asarray(r["ee_pose_w"], np.float64))


def main():
    import trimesh
    man = json.loads((ROOT["f0"] / "track2_manifest.json").read_text())
    seq_of = {v["episode_index"]: k for k, v in man.items()}
    # source->robot axes: fit from the payload joints vs nothing else would need FK; use the retarget's config value
    spec = importlib.util.spec_from_file_location("s2g", ME / "code/nv-v2d-challenge-codebase/robotic_grounding/scripts/retarget/soma_to_g1.py")
    s2g = importlib.util.module_from_spec(spec); spec.loader.exec_module(s2g)
    S = np.asarray(s2g.load_robot_config("g1").r_world, np.float64)
    print("src2rob", S.tolist())
    for ep in [int(a) for a in sys.argv[1:]]:
        seq = seq_of[ep]
        d = {k: read(ROOT[k] / f"sequence_id={seq}" / "robot_name=g1/data.parquet") for k in ROOT}
        jdiff = float(np.abs(d["f0"]["J"] - d["opus"]["J"]).max()); assert jdiff < 1e-3, f"human differs {jdiff}"
        names = [str(x) for x in np.load(ME / "tmp_analysis/tier3/f0/frames/ep03.npz")["joint_names"]]  # static SOMA joint-name list (77)
        assert len(names) == d["f0"]["J"].shape[1]
        Jw = d["f0"]["J"] @ S.T
        mesh = trimesh.load(str(ROOT["f0"] / f"sequence_id={seq}" / "object/textured_mesh.obj"), force="mesh", process=False)
        T = len(Jw)
        pts = {}
        for side, pre in (("left", "Left"), ("right", "Right")):
            palm = Jw[:, [names.index(pre + n) for n in ("Hand", "HandIndex1", "HandMiddle1", "HandRing1", "HandPinky1")]].mean(1)
            tips = Jw[:, [names.index(pre + f"Hand{f}End") for f in ("Thumb", "Index", "Middle", "Ring", "Pinky")]]
            pts[side] = np.concatenate([palm[:, None], tips], 1)  # (T,6,3)
        dz = np.load(DUMP / f"ep{ep:02d}.npz")
        sets = {}
        for si, side in enumerate(("left", "right")):
            sets[("HCA_OPUS", side)] = d["opus"]["hca"][si] > 0.5
            sets[("HCA_UNION", side)] = (d["opus"]["hca"][si] > 0.5) | (d["f0"]["hca"][si] > 0.5)
            sets[("THEIRS", side)] = dz[f"opus_{side}_contact"].astype(bool) | dz[f"f0_{side}_contact"].astype(bool)
        need = {side: np.zeros(T, bool) for side in ("left", "right")}
        for (nm, side), m in sets.items():
            need[side] |= m
        gap = {}
        for k in ROOT:
            for side in ("left", "right"):
                idx = np.where(need[side])[0]
                g = np.full(T, np.nan)
                if len(idx):
                    P = pts[side][idx]                                   # (n,6,3) world
                    L = np.einsum("tnj,tjk->tnk", P - d[k]["p"][idx, None, :], d[k]["R"][idx])  # object frame
                    _, dist, _ = trimesh.proximity.closest_point(mesh, L.reshape(-1, 3))
                    g[idx] = dist.reshape(len(idx), 6).min(1) * 100
                gap[(k, side)] = g
        out = {"ep": ep, "T": T, "human_f0_vs_opus_max_m": jdiff}
        for nm in ("THEIRS", "HCA_OPUS", "HCA_UNION"):
            row = {}
            for k in ROOT:
                v = np.concatenate([gap[(k, s)][sets[(nm, s)]] for s in ("left", "right")])
                row[k] = (int(v.size), round(float(np.median(v)), 2) if v.size else None, round(float(np.percentile(v, 90)), 2) if v.size else None)
            q = None
            if row["f0"][1] is not None:
                q = bool(row["f0"][1] > 3.0 or row["f0"][1] - row["opus"][1] > 2.0)
            row["qualifies"] = q
            out[nm] = row
        # robot palm reference vs f0 object (exact), on HCA_OPUS
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
