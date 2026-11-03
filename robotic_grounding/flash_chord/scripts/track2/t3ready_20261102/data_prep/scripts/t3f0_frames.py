#!/usr/bin/env python
"""t3f0_frames.py -- Tier-3 f0 prep: the retarget's frame chain WITHOUT IK (CPU only, v2d env). NEW FILE (2026-10-24),
derived from scripts/t3sup_frames.py (left untouched); reads the f0 root data/human_motion_data_tier3_f0 (byte copies of
Han Zhou's frame-0 release), writes only tmp_analysis/tier3/f0/frames/.

Reproduces scripts/retarget/soma_to_g1.py (nv repo robotic_grounding @ f2923d61, imported read-only from the clean tree):
  OpenCV world -> SOMA axes (_R_CV_TO_SOMA) -> frame-0 root anchor (SOMA normalize) -> single-pass plane levelling
  (ground_plane.json) -> robot world (config.r_world); object only: + constant frame-0 ground lift (cap 1 cm).
Saves per episode the 4x4 STATIC map M (OpenCV world -> G1 parquet world) that places any static geometry of Han's
release (support meshes, top boxes) in the world the trainer uses, the object track in that world (validated against the
retargeted parquet later: <= 1e-5 m, <= 1e-3 deg), the SOMA joints and (with --verts, for the support episodes) the SOMA
mesh vertices in the same world. The human chain is the retarget's own visualisation path (levelled vertices).
No Tier-1/Tier-2 data, no clean data are read.
Usage: t3f0_frames.py [--verts EP,EP,..] EP [EP ...]
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, time, hashlib, importlib.util
from pathlib import Path
import numpy as np

ME = Path("/root/autodl-tmp/haoran.zhang")
SRC = ME / "data/human_motion_data_tier3_f0/whole_body/soma"
OUT = ME / "tmp_analysis/tier3/f0/frames"
NV = ME / "code/nv-v2d-challenge-codebase/robotic_grounding"
S2G = NV / "scripts/retarget/soma_to_g1.py"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_s2g():
    spec = importlib.util.spec_from_file_location("t3f0_soma_to_g1", S2G)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    args = sys.argv[1:]
    verts_eps = set()
    if "--verts" in args:
        k = args.index("--verts"); verts_eps = {int(x) for x in args[k + 1].split(",") if x}; del args[k:k + 2]
    eps = [int(a) for a in args]
    man = json.loads((SRC / "track2_manifest.json").read_text())
    seq_of = {v["episode_index"]: k for k, v in man.items()}
    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    s2g = load_s2g()
    config = s2g.load_robot_config("g1")
    kin = s2g.WholeBodyKinematics(config=config)
    soma = s2g.SOMA(data_root=None, identity_model_type="mhr", device=torch.device("cpu"))
    src2rob = np.asarray(config.r_world, dtype=np.float64)
    S4 = np.eye(4); S4[:3, :3] = src2rob
    OUT.mkdir(parents=True, exist_ok=True)
    for ep in eps:
        t0 = time.time()
        seq = seq_of[ep]
        folder = SRC / f"sequence_id={seq}"
        glb = folder / "reconstructed_mesh" / "output_aligned.glb"
        objdir = OUT / f"ep{ep:02d}_obj"
        mesh_path = s2g._convert_glb_to_obj(glb, objdir)
        P_cv = np.load(folder / "poses.npy")
        object_poses_world = s2g._convert_object_poses_cv_to_soma(P_cv)
        # ---- verbatim chain of soma_to_g1.main() (@ f2923d61) ----
        motion = soma.load_motion(params_path=str(folder / "soma_params.npz"), normalize=True)
        transl_first = motion["first_frame_transl"]
        R_first_inv = motion["first_frame_R_inv"]
        norm_transform = np.eye(4)
        norm_transform[:3, :3] = R_first_inv
        norm_transform[:3, 3] = -R_first_inv @ transl_first
        object_poses = np.einsum("ij,tjk->tik", norm_transform, object_poses_world)
        ground_plane_path = folder / "ground_plane.json"
        reconstructed_plane = s2g.load_ground_plane_robot_frame(
            ground_plane_path, cv_to_source=s2g._R_CV_TO_SOMA, first_frame_anchor=norm_transform, source_to_robot=src2rob)
        if reconstructed_plane is None:
            raise SystemExit(f"ep{ep}: no ground_plane.json -> fallback plane path not reproduced")
        rot_r, tr_r = s2g.compute_plane_leveling_transform(reconstructed_plane)
        rot_s = src2rob.T @ rot_r @ src2rob
        tr_s = src2rob.T @ tr_r
        joint_pos = np.einsum("ij,tkj->tki", rot_s, motion["joints"]) + tr_s
        G = np.eye(4); G[:3, :3] = rot_s; G[:3, 3] = tr_s
        object_poses = np.einsum("ij,tjk->tik", G, object_poses)
        mv = []
        with open(mesh_path) as f:
            for line in f:
                if line.startswith("v "):
                    p = line.split(); mv.append([float(p[1]), float(p[2]), float(p[3])])
        mesh_v = np.array(mv, dtype=np.float64)
        p0 = kin.transform_source_position(object_poses[0, :3, 3])
        R0 = kin.transform_world_rotation(object_poses[0, :3, :3])
        lift = s2g.compute_object_ground_lift(
            mesh_v @ R0.T + p0, s2g.ReferencePlane.horizontal(),
            penetration_tolerance=s2g.OBJECT_GROUND_PENETRATION_TOLERANCE_M,
            clearance=s2g.OBJECT_GROUND_CLEARANCE_M, max_lift=s2g.OBJECT_GROUND_MAX_LIFT_M)
        object_poses[:, :3, 3] += src2rob.T @ np.array([0.0, 0.0, lift.applied_lift])
        obj_pos = object_poses[:, :3, 3] @ src2rob.T
        obj_R = np.einsum("ij,tjk->tik", src2rob, object_poses[:, :3, :3])
        joints_w = joint_pos @ src2rob.T
        # ---- static map OpenCV world -> G1 world (no object lift), and its consistency with the object chain ----
        M = S4 @ G @ norm_transform @ s2g._R_CV_TO_SOMA
        Pm = np.einsum("ij,tjk->tik", M, P_cv)
        chk_pos = float(np.abs(Pm[:, :3, 3] + np.array([0, 0, lift.applied_lift]) - obj_pos).max())
        chk_rot = float(np.abs(Pm[:, :3, :3] - obj_R).max())
        from scipy.spatial.transform import Rotation as R
        quat = R.from_matrix(obj_R).as_quat(scalar_first=True)
        save = dict(M_cv_to_g1=M, lift=np.float64(lift.applied_lift), obj_pos=obj_pos, obj_R=obj_R, obj_wxyz=quat,
                    joints_w=joints_w.astype(np.float32), joint_names=np.asarray(motion["joint_names"]))
        if ep in verts_eps:
            vertices = np.einsum("ij,tvj->tvi", rot_s, motion["vertices"]) + tr_s
            save["verts_w"] = (vertices @ src2rob.T).astype(np.float32)
            save["faces_human"] = soma.faces if soma.faces is not None else np.zeros((0, 3), np.int64)
        np.savez_compressed(OUT / f"ep{ep:02d}.npz", **save)
        meta = dict(ep=ep, seq=seq, frames=int(len(obj_pos)), n_mesh_v=int(len(mesh_v)), lift_applied=float(lift.applied_lift),
                    lift_min_signed=float(lift.minimum_signed_distance), verts=ep in verts_eps,
                    plane_normal=[float(x) for x in reconstructed_plane.normal], plane_offset=float(reconstructed_plane.offset),
                    M_vs_object_chain_max_err=[chk_pos, chk_rot], obj_sha256=sha(mesh_path), soma_to_g1_sha256=sha(S2G),
                    inputs={n: sha(folder / n) for n in ("poses.npy", "soma_params.npz", "ground_plane.json",
                                                          "reconstructed_mesh/output_aligned.glb")},
                    seconds=round(time.time() - t0, 1))
        (OUT / f"ep{ep:02d}.json").write_text(json.dumps(meta, indent=1))
        print(json.dumps({k: meta[k] for k in ("ep", "frames", "lift_applied", "M_vs_object_chain_max_err", "verts", "seconds")}),
              flush=True)
        if chk_pos > 1e-9 or chk_rot > 1e-9:
            raise SystemExit(f"ep{ep}: static map M does not reproduce the object chain ({chk_pos}, {chk_rot})")


if __name__ == "__main__":
    main()
