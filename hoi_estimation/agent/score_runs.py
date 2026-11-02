#!/usr/bin/env python3
"""Score agent runs against the reference (OUTSIDE the sandbox) and the fixed combination rules.

Variants per episode (all in mesh-file frame, world):
  db1          the baseline given to the agent (inputs/reconstruction.npz)
  agent        the agent's output
  rot          agent rotation + baseline translation
  rot_ray      agent rotation + agent image direction (ray from the camera) + baseline camera distance
  trans        baseline rotation + agent translation
Also writes <ws>/outputs/combined_rot_ray.npz and combined_rot.npz for later use.
Usage: score_runs.py <ws> [<ws> ...]
"""
import os
import json
import re
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hoi_extraction", "scripts"))
from eval_object_pose import metrics, quat_wxyz_to_R  # noqa: E402

V2D = os.environ.get("V2D_ROOT", "/mnt/nas/Group/hanzhou/v2d")
t2 = {int(l.split(",")[0]): float(l.split(",")[10]) for l in open(f"{V2D}/logs/officialDB1.csv").read().splitlines()[1:]}

for ws in sys.argv[1:]:
    ep = json.load(open(f"{ws}/inputs/episode.json"))["episode_index"]
    glb = trimesh.load(f"{ws}/inputs/object_mesh.glb", force="scene", process=False).to_geometry()
    cam = json.load(open(f"{ws}/inputs/camera.json"))
    Twc = np.array(cam["T_world_cam"])
    C = Twc[:3, 3]
    D = np.load(f"{ws}/inputs/reconstruction.npz")["T_world_mesh"].astype(float)
    try:
        A = np.load(f"{ws}/outputs/object_poses_world.npz")["T_world_mesh"].astype(float)
    except FileNotFoundError:
        print(f"ep {ep}: no agent output")
        continue
    R1 = A.copy(); R1[:, :3, 3] = D[:, :3, 3]
    # same distance from the camera centre as the baseline, along the agent's viewing ray
    da, dd = A[:, :3, 3] - C, D[:, :3, 3] - C
    R2 = A.copy(); R2[:, :3, 3] = C + da / np.linalg.norm(da, axis=1, keepdims=True) * np.linalg.norm(dd, axis=1, keepdims=True)
    R3 = D.copy(); R3[:, :3, 3] = A[:, :3, 3]
    np.savez(f"{ws}/outputs/combined_rot.npz", T_world_mesh=R1)
    np.savez(f"{ws}/outputs/combined_rot_ray.npz", T_world_mesh=R2)
    ref = np.load(f"{V2D}/ref_poses/episode_{ep:06d}.npz")["pose_xyz_wxyz"].astype(float)
    n = min(len(D), len(ref)); ok = np.linalg.norm(ref[:n, 3:], axis=1) > 0.5
    Rr, tr = quat_wxyz_to_R(ref[:n][ok][:, 3:]), ref[:n][ok][:, :3]
    out = []
    for name, T in [("db1", D), ("agent", A), ("rot", R1), ("rot_ray", R2), ("trans", R3)]:
        ms = []
        for seed in range(3):
            np.random.seed(seed)
            v = np.asarray(glb.sample(1000))
            ms.append(metrics(T[:n][ok][:, :3, :3], T[:n][ok][:, :3, 3], Rr, tr, v))
        out.append(f"{name} {np.mean([m['AUC'] for m in ms]):.3f} ({ms[0]['pos_cm']:.1f}cm,{ms[0]['rot_deg']:.0f}deg)")
    man = open(f"{ws}/logs/run_manifest.txt").read()
    wall = re.search(r"wall_s: (\d+)", man)
    print(f"ep {ep:2d} t2 {t2[ep]:.3f} | " + " | ".join(out) + f" | wall {int(wall.group(1))//60 if wall else '-'} min", flush=True)
