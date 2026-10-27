#!/usr/bin/env python3
"""Build the Tier 3 Opus v2 release: the s1 release with each episode's object trajectory replaced by the Claude Opus
agent's (agent_runs/CL2*_epXXX/outputs/object_poses_world.npz). Human, camera, ground plane unchanged.
Usage: build_release_opus.py <out_dir>"""
import json, os, shutil, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
V = "/mnt/nas/Group/hanzhou/v2d"; S1 = f"{V}/release/tier3_s1_20261020"; out = sys.argv[1]
eps = json.load(open(f"{S1}/episodes.json"))
os.makedirs(f"{out}/reconstructed_data", exist_ok=True)
for f in ("episodes.json", "frame_references"):
    (shutil.copytree if os.path.isdir(f"{S1}/{f}") else shutil.copy)(f"{S1}/{f}", f"{out}/{f}")
for e in eps:
    i, sid = e["episode_index"], e["sequence_id"]
    run = f"{V}/agent_runs/{'CL2' if i <= 10 else 'CL2b' if i <= 21 else 'CL2c'}_ep{i:03d}"
    od = f"{out}/reconstructed_data/{sid}"; shutil.copytree(f"{S1}/reconstructed_data/{sid}", od)
    T0 = np.load(f"{od}/poses.npy"); T = np.load(f"{run}/outputs/object_poses_world.npz")["T_world_mesh"].astype(np.float64)
    assert T.shape == T0.shape and np.isfinite(T).all(), (i, T.shape, T0.shape)
    np.save(f"{od}/poses.npy", T)
    q = Rot.from_matrix(T[:, :3, :3]).as_quat()  # xyzw
    np.save(f"{od}/object_pose_xyz_wxyz.npy", np.c_[T[:, :3, 3], q[:, 3:], q[:, :3]].astype(np.float32))
    ad = f"{out}/agent_run/{sid}"; os.makedirs(ad)
    for f in ("REPORT.md", "solve.py", "TOOLS.md"):
        if os.path.isfile(f"{run}/outputs/{f}"): shutil.copy(f"{run}/outputs/{f}", ad)
    d = np.linalg.norm(T[:, :3, 3] - T0[:, :3, 3], axis=1)
    print(i, e["object"], f"shift vs s1 mean {d.mean()*100:.1f} cm max {d.max()*100:.1f} cm", flush=True)
