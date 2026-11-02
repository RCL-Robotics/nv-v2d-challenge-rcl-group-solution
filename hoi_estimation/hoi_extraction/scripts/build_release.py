#!/usr/bin/env python3
"""Package a reconstruction variant for the downstream SOMA->G1 retargeting (GPU, inside the v2d_cari4d image).

Per episode, writes <out>/reconstructed_data/<sequence_id>/ in the layout read by
robotic_grounding/scripts/retarget/soma_to_g1.py:
  soma_params.npz                        human, SOMA-X parameters (save_soma_npz schema, metres), released world frame
  poses.npy                              (T,4,4) world_T_object, OpenCV released world, for the mesh below
  reconstructed_mesh/output_aligned.glb  the released object mesh (mesh-file frame, unchanged)
  ground_plane.json                      released ground plane (copy)
plus object_pose_xyz_wxyz.npy (T,7, dataset convention), camera.json (K, T_world_cam), episode.json.

Human: our MHR (SAM-3D-Body / CoCoNet) bodies are placed with the same camera chain as the object
(camera-frame pixel-preserving K change + per-frame feet-on-floor depth factor a_t of the camera variant,
applied to the body root so the body keeps its MHR size), mapped to the world by T_world_cam, then converted
to SOMA-X with the official PoseInversion (same settings as v2d_sam3d_body/lib/export_soma.py).
Usage: build_release.py <out_dir> --variant DB_JC2s1_shared_f820_pp --camera JC2s1_shared_f820_pp [--episodes ...]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "/workspace/v2d_cari4d/lib/cari4d")
sys.path.insert(0, os.path.dirname(__file__))
from eval_official import bundle_path  # noqa: E402

V2D = "/mnt/nas/Group/hanzhou/v2d"
DS = f"{V2D}/dataset/track_2/tier_1_multiview_caption"
FLIP = np.diag([1.0, -1.0, -1.0])  # OpenCV world <-> SOMA native axes


def mhr_bodies(layer, block, F, dev, bs=128):
    """Camera-frame MHR vertices (metres, OpenCV), identity coeffs (45), resolved scales (68), bone flexibles (6)."""
    from tools.render_mhr_wild_inference import _params
    V, MP, SH = [], [], []
    with torch.no_grad():
        for a in range(0, F, bs):
            p = _params(block, slice(a, min(a + bs, F)), dev)
            be = layer.backend
            ctx = be._vertices_context(p, detach_fixed=True)
            trans, body, shape, scale = be._mutable_vertices_inputs(p, ctx)
            v, mp = ctx.head.mhr_forward(global_trans=trans.new_zeros(trans.shape), global_rot=ctx.global_rot, body_pose_params=body,
                                         hand_pose_params=ctx.hand, scale_params=scale, shape_params=shape, expr_params=ctx.face,
                                         return_model_params=True)
            v = v * ctx.flip + trans[:, None, :]
            if a == 0:  # must reproduce the pipeline's decoder exactly
                ref = layer.mhr_forward(p).vertices
                assert float((ref - v).abs().max()) < 1e-4, float((ref - v).abs().max())
            V.append(v.cpu().numpy().astype(np.float64))
            MP.append(mp.cpu().numpy())
            SH.append(shape.cpu().numpy())
    MP = np.concatenate(MP)
    return np.concatenate(V), np.concatenate(SH), MP[:, 136:], MP[:, 130:136]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--variant", default="DB_JC2s1_shared_f820_pp", help="object poses (bundle variant)")
    ap.add_argument("--camera", default="JC2s1_shared_f820_pp", help="camera variant (align json + K) used by --variant")
    ap.add_argument("--episodes", type=int, nargs="+", default=list(range(30)))
    args = ap.parse_args()
    from lib_mhr.mhr_layer import MHRLayer
    from soma.io import save_soma_npz
    from soma.geometry.rig_utils import remove_joint_orient_local
    from soma.geometry.transforms import matrix_to_rotvec
    from soma.pose_inversion import PoseInversion
    from soma import SOMALayer
    from soma.units import Unit

    dev = torch.device("cuda")
    layer = MHRLayer.from_mhr_assets(mhr_assets_root=f"{V2D}/weights/sam3d_body", device="cuda")
    # as create_soma_layer() in v2d_sam3d_body/lib/export_soma.py (public 78-joint rig, centimetres)
    soma = SOMALayer(identity_model_type="mhr", device="cuda", mode="warp", output_unit=Unit.CENTIMETERS,
                     enable_procedural_transforms=False)
    inv = PoseInversion(soma, low_lod=True)
    meta = {d["episode_index"]: d for d in map(json.loads, open(f"{DS}/meta/episodes_metadata.jsonl"))}
    al = json.load(open(f"{V2D}/logs/{args.camera}_align.json"))
    index = []
    for idx in args.episodes:
        ep, m = f"episode_{idx:06d}", meta[idx]
        od = f"{args.out}/reconstructed_data/{m['sequence_id']}"
        os.makedirs(f"{od}/reconstructed_mesh", exist_ok=True)
        b = torch.load(bundle_path(args.variant, ep), map_location="cpu", weights_only=False)
        bc = torch.load(bundle_path(args.camera, ep), map_location="cpu", weights_only=False)
        src = al[str(idx)]["src"]
        bs = torch.load(bundle_path(src, ep), map_location="cpu", weights_only=False)
        glb = trimesh.load(f"{DS}/{m['mesh']}", force="scene", process=False).to_geometry()
        S = np.asarray(trimesh.bounds.oriented_bounds(glb)[0], np.float64)
        W = np.asarray(al[str(idx)]["sim"], np.float64)
        K0 = np.asarray(bs["observations"]["K_full"], np.float64)[0]
        Kn = np.asarray(al[str(idx)]["K"], np.float64)
        F = len(np.asarray(b["pr"]["pose_abs"]))

        # object: world_T_mesh-file
        T = W[None] @ np.asarray(b["pr"]["pose_abs"], np.float64) @ S
        np.save(f"{od}/poses.npy", T)
        q = Rot.from_matrix(T[:, :3, :3]).as_quat()  # xyzw
        np.save(f"{od}/object_pose_xyz_wxyz.npy", np.c_[T[:, :3, 3], q[:, 3:], q[:, :3]].astype(np.float32))
        shutil.copy(f"{DS}/{m['mesh']}", f"{od}/reconstructed_mesh/output_aligned.glb")
        shutil.copy(f"{DS}/{m['ground_plane']}", f"{od}/ground_plane.json")
        json.dump({"K": Kn.tolist(), "T_world_cam": W.tolist(), "width": 1536, "height": 1152,
                   "note": "pinhole, OpenCV camera; world = released OpenCV world (y-down)"}, open(f"{od}/camera.json", "w"), indent=1)

        # human: same camera chain as the object (camera variant), applied to the body root
        def remap(X):
            z = X[..., 2:3]
            uv = X[..., :2] / z * K0[0, 0] + K0[:2, 2]
            zn = z * Kn[0, 0] / K0[0, 0]
            return np.concatenate([(uv - Kn[:2, 2]) / Kn[0, 0] * zn, zn], -1)
        t_src = (np.asarray(bs["pr"]["pose_abs"], np.float64) @ S)[:, :3, 3]
        t_cam = (np.asarray(bc["pr"]["pose_abs"], np.float64) @ S)[:, :3, 3]
        a = np.linalg.norm(t_cam, axis=1) / np.linalg.norm(remap(t_src), axis=1)
        a = np.where(np.isfinite(a), a, 1.0)
        Vc, shape, scales, flex = mhr_bodies(layer, bs["pr"], F, dev)
        root = Vc.mean(1, keepdims=True)
        Vc = Vc - root + a[:, None, None] * remap(root)
        Vw = Vc @ W[:3, :3].T + W[:3, 3]
        Vs = torch.tensor(Vw @ FLIP * 100.0, dtype=torch.float32)  # SOMA native axes, cm

        rots, tr, err = [], [], []
        sh_t, sc_t, fl_t = (torch.tensor(x, dtype=torch.float32, device=dev) for x in (shape, scales, flex))
        for s0 in range(0, F, 64):
            s1 = min(s0 + 64, F)
            inv.prepare_identity(sh_t[s0:s1], sc_t[s0:s1], kwargs={"bone_length_flexibles": fl_t[s0:s1]})
            r = inv.fit(Vs[s0:s1].to(dev), body_iters=2, finger_iters=0, full_iters=1, lie_iters=3, lie_lambda=1e-1,
                        autograd_iters=0, batch_size=None)
            rots.append(r["rotations"].cpu()); tr.append(r["root_translation"].cpu()); err.append(r["per_vertex_error"].cpu())
        rots, tr, err = torch.cat(rots), torch.cat(tr), torch.cat(err)
        d = soma._t_pose_orient.device
        rel = remove_joint_orient_local(rots.to(d), soma._t_pose_orient, soma._t_pose_orient_parent_T)
        poses = matrix_to_rotvec(rel.reshape(-1, 3, 3)).reshape(rots.shape[0], rots.shape[1], 3).cpu()
        save_soma_npz(f"{od}/soma_params.npz", poses, tr * 0.01, joint_names=list(soma.rig_data["joint_names"]),
                      identity_model_type=soma.identity_model_type, identity_coeffs=shape, scale_params=scales,
                      joint_orient=soma._t_pose_orient, unit="meters", keep_root=False,
                      extra_arrays={"bone_length_flexibles": flex})
        info = {"episode_index": idx, "sequence_id": m["sequence_id"], "object": m["object"], "task": m["task"], "frames": F,
                "object_variant": args.variant, "camera_variant": args.camera,
                "soma_inversion_vertex_error_cm": {"mean": float(err.mean()), "median": float(err.median()), "max": float(err.max())}}
        json.dump(info, open(f"{od}/episode.json", "w"), indent=1)
        index.append(info)
        print(ep, m["sequence_id"], F, "soma err cm mean %.2f max %.2f" % (err.mean(), err.max()), flush=True)
    old = {}
    p = f"{args.out}/episodes.json"
    if os.path.isfile(p):
        old = {d["episode_index"]: d for d in json.load(open(p))}
    old.update({d["episode_index"]: d for d in index})
    json.dump([old[k] for k in sorted(old)], open(p, "w"), indent=1)


if __name__ == "__main__":
    main()
