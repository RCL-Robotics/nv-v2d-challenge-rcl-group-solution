import os, sys, json, hashlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
ME = Path("/root/autodl-tmp/haoran.zhang")
HAN = Path("/root/autodl-tmp/han.zhou/tier3/release/tier3_frame0_supports_20261024/reconstructed_data")
HANO = Path("/root/autodl-tmp/han.zhou/tier3/release/tier3_opus_20261021/reconstructed_data")
F0 = ME / "data/human_motion_data_tier3_f0"; OP = ME / "data/human_motion_data_tier3_opus"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
man = json.loads((F0 / "whole_body/soma/track2_manifest.json").read_text())
seq_of = {v["episode_index"]: k for k, v in man.items()}
def q2m(q):
    w, x, y, z = q.T
    return np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], -1), np.stack([2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], -1),
                     np.stack([2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1)], -2)
def kabsch(A, B):  # B ~ R A + t
    ca, cb = A.mean(0), B.mean(0); H = (A-ca).T @ (B-cb); U, s, Vt = np.linalg.svd(H); D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    Rm = Vt.T @ D @ U.T; return Rm, cb - Rm @ ca
for ep in [int(a) for a in sys.argv[1:]]:
    seq = seq_of[ep]; out = {"ep": ep, "seq": seq}
    hd = HAN / seq; fd = F0 / "whole_body/soma" / f"sequence_id={seq}"; od = OP / "whole_body/soma" / f"sequence_id={seq}"
    out["bytes_eq_han"] = {n: sha(hd / n) == sha(fd / n) for n in ("poses.npy", "object_pose_xyz_wxyz.npy", "soma_params.npz", "camera.json", "ground_plane.json", "episode.json", "reconstructed_mesh/output_aligned.glb")}
    out["opus_root_eq_han_opus"] = {n: sha(HANO / seq / n) == sha(od / n) for n in ("poses.npy", "object_pose_xyz_wxyz.npy", "soma_params.npz")}
    out["human_same_f0_opus"] = sha(fd / "soma_params.npz") == sha(od / "soma_params.npz")
    P = np.load(fd / "poses.npy"); X = np.load(fd / "object_pose_xyz_wxyz.npy")
    out["poses_shape"] = list(P.shape); out["xyzwxyz_shape"] = list(X.shape)
    out["xyz_vs_poses_max"] = float(np.abs(X[:, :3] - P[:, :3, 3]).max())
    out["rot_vs_poses_max"] = float(np.abs(q2m(X[:, 3:7] / np.linalg.norm(X[:, 3:7], axis=1, keepdims=True)) - P[:, :3, :3]).max())
    # frame refs
    for tag, root, Pp, sub in (("f0", F0, P, "frame_references_f0"), ("opus", OP, np.load(od / "poses.npy"), "frame_references_opus")):
        with np.load(root / sub / f"episode_{ep:06d}.npz") as z:
            keys = {k: (str(z[k].dtype), list(z[k].shape)) for k in z.files}
            xyzw = z["pose_xyzw"][:, 0]
            q = xyzw[:, 3:]; Rq = q2m(np.c_[q[:, 3], q[:, :3]] / np.linalg.norm(q, axis=1, keepdims=True))
            out[f"fref_{tag}"] = {"keys": keys, "pos_err": float(np.abs(xyzw[:, :3] - Pp[:, :3, 3]).max()), "rot_err": float(np.abs(Rq - Pp[:, :3, :3]).max()),
                                  "frame": str(z["coordinate_frame"]), "ep": int(z["episode_index"])}
    # parquet object vs poses.npy: rigid fit
    fits = {}
    for tag, d, Pp in (("f0", fd, P), ("opus", od, np.load(od / "poses.npy"))):
        r = pq.read_table(d / "robot_name=g1/data.parquet", columns=["object_body_position", "object_body_wxyz"]).to_pylist()[0]
        pos = np.asarray(r["object_body_position"])[:, 0]; qq = np.asarray(r["object_body_wxyz"])[:, 0]
        Rm, t = kabsch(Pp[:, :3, 3], pos)
        res = float(np.abs(Pp[:, :3, 3] @ Rm.T + t - pos).max())
        Rpq = q2m(qq / np.linalg.norm(qq, axis=1, keepdims=True))
        rot_res = float(np.abs(np.einsum("ij,tjk->tik", Rm, Pp[:, :3, :3]) - Rpq).max())
        fits[tag] = (Rm, t); out[f"pq_fit_{tag}"] = {"pos_resid_m": res, "rot_resid": rot_res}
    out["fit_R_f0_vs_opus"] = float(np.abs(fits["f0"][0] - fits["opus"][0]).max())
    out["fit_t_diff_m"] = (fits["f0"][1] - fits["opus"][1]).round(5).tolist()
    # constant offset of f0 vs opus in OpenCV world
    Po = np.load(od / "poses.npy"); dpos = P[:, :3, 3] - Po[:, :3, 3]
    out["f0_minus_opus_pos_cv_mean_cm"] = (100 * dpos.mean(0)).round(2).tolist(); out["f0_minus_opus_pos_cv_std_cm"] = (100 * dpos.std(0)).round(3).tolist()
    print(json.dumps(out), flush=True)
