#!/usr/bin/env python3
"""Combined Tier 3 release: the Opus v2 release with each object trajectory replaced by the frame-0 refined one
(experiments/t3_frame0/opus_f3_c2_shift), plus the reconstructed support furniture for the 14 placement episodes
under supports/<sequence_id>/ (from release/tier3_supports_20261024, without its duplicate trajectory copy).
Human, camera, ground plane unchanged. Usage: build_release_combined.py <out_dir>"""
import json, os, shutil, sys
import numpy as np
from scipy.spatial.transform import Rotation as Rot
V = os.environ.get("V2D_ROOT", "/mnt/nas/Group/hanzhou/v2d"); SRC = f"{V}/release/tier3_opus_20261021"; SUP = f"{V}/release/tier3_supports_20261024"
TRAJ = f"{V}/experiments/t3_frame0/opus_f3_c2_shift"; out = sys.argv[1]
eps = json.load(open(f"{SRC}/episodes.json"))
# diag_scores.json is written only by the GT diagnostic half of t3_frame0.py; it is printed, never used
diag = json.load(open(f"{TRAJ}/diag_scores.json")) if os.path.isfile(f"{TRAJ}/diag_scores.json") else {}
shutil.copytree(SRC, out)
for e in eps:
    i, sid = e["episode_index"], e["sequence_id"]; od = f"{out}/reconstructed_data/{sid}"
    T0 = np.load(f"{od}/poses.npy"); T = np.load(f"{TRAJ}/episode_{i:06d}.npz")["T_world_mesh"].astype(np.float64)
    assert T.shape == T0.shape and np.isfinite(T).all(), (i, T.shape, T0.shape)
    np.save(f"{od}/poses.npy", T)
    q = Rot.from_matrix(T[:, :3, :3]).as_quat()  # xyzw
    np.save(f"{od}/object_pose_xyz_wxyz.npy", np.c_[T[:, :3, 3], q[:, 3:], q[:, :3]].astype(np.float32))
    e["object_variant"] = "opus_v2_frame0_f3_c2_shift"
    if os.path.isdir(f"{SUP}/episode_{i:06d}"):
        sd = f"{out}/supports/{sid}"; os.makedirs(sd)
        for f in ("support.json", "support_mesh.glb", "support_world.obj", "fit_overlay.jpg"): shutil.copy(f"{SUP}/episode_{i:06d}/{f}", sd)
        e["support"] = f"supports/{sid}"
    print(i, e["object"], f"diag AUC {diag[str(i)]['auc']:.3f}" if str(i) in diag else "", "support" if "support" in e else "", flush=True)
json.dump(eps, open(f"{out}/episodes.json", "w"), indent=1)
for e in eps: json.dump({k: v for k, v in e.items() if k != "support"} | ({"support": "../../" + e["support"]} if "support" in e else {}), open(f"{out}/reconstructed_data/{e['sequence_id']}/episode.json", "w"), indent=1)
for f in ("fit_overview.jpg", "contact_report.json", "index.json"): shutil.copy(f"{SUP}/{f}", f"{out}/supports/{f}")
shutil.copy(f"{SUP}/README.md", f"{out}/supports/README_supports.md")
