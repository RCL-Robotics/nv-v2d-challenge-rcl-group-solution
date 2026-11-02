#!/usr/bin/env python3
"""Image anchor for picking the right detection: the object (our best Tier 3 trajectory, frame-0 refined) at the end
frame (first or last) where it stands highest above the ground plane, i.e. resting on (or leaning against) the support.
Writes OUT/epXXX/anchor.json: frame, object centre (world + pixel), its height above the ground."""
import json, sys, os, numpy as np
from common import *
TRAJ = f"{V}/experiments/t3_frame0/opus_f3_c2_shift"
for i in (map(int, sys.argv[1].split(",")) if len(sys.argv) > 1 else SUPPORT):
    os.makedirs(f"{OUT}/ep{i:03d}", exist_ok=True)
    T = np.load(f"{TRAJ}/episode_{i:06d}.npz")["T_world_mesh"]; g = json.load(open(f"{inp(i)}/ground_plane.json"))["plane"]
    cam = json.load(open(f"{inp(i)}/camera.json")); K = np.array(cam["K"]); Wi = np.linalg.inv(np.array(cam["T_world_cam"]))
    h = T[:, :3, 3] @ np.array(g[:3]) + g[3]; t = 0 if h[0] >= h[-1] else len(T) - 1
    pc = Wi[:3, :3] @ T[t, :3, 3] + Wi[:3, 3]; uv = (K @ pc)[:2] / pc[2]
    json.dump(dict(frame=t, p_world=T[t, :3, 3].tolist(), uv=uv.tolist(), height=float(h[t]), h0=float(h[0]), h_end=float(h[-1])), open(f"{OUT}/ep{i:03d}/anchor.json", "w"))
    print(i, SUPPORT[i], "frame", t, "height %.2f (first %.2f last %.2f)" % (h[t], h[0], h[-1]), "uv", uv.round(0))
