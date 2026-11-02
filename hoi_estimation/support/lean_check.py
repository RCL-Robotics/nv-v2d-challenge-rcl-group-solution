#!/usr/bin/env python3
"""Leaning objects (ep1 vacuum, ep28 squeegee): closest distance between the object surface and the support mesh at the
first and last frame, for our trajectory and (diagnostic only, --gt) the reference poses."""
import json, sys, numpy as np, trimesh
from scipy.spatial.transform import Rotation as Rot
from common import *
TRAJ = f"{V}/experiments/t3_frame0/opus_f3_c2_shift"
for i in (1, 28):
    s = json.load(open(f"{OUT}/ep{i:03d}/support.json")); sup = trimesh.load(f"{MESH}/{s['support']}/{s['support']}.glb", force="scene", process=False).to_geometry(); sup.apply_transform(np.array(s["T_world_mesh"]))
    obj = trimesh.load(f"{inp(i)}/object_mesh.glb", force="scene", process=False).to_geometry(); np.random.seed(0); P = np.asarray(obj.sample(800))
    ref = np.load(f"{V}/ref_poses/episode_{i:06d}.npz")["pose_xyz_wxyz"].astype(float); T = np.load(f"{TRAJ}/episode_{i:06d}.npz")["T_world_mesh"]
    for lab, poses in (("ours", [T[0], T[-1]]), ("GT(diag)", [np.vstack([np.hstack([Rot.from_quat(r[[4, 5, 6, 3]]).as_matrix(), r[:3, None]]), [0, 0, 0, 1]]) for r in (ref[0], ref[-1])])):
        dd = [float(trimesh.proximity.closest_point(sup, P @ M[:3, :3].T + M[:3, 3])[1].min() * 100) for M in poses]
        print(i, lab, "closest object-support distance first %.1f cm, last %.1f cm" % tuple(dd))
