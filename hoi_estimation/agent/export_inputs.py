#!/usr/bin/env python3
"""Export the allowed Tier-3 inputs of one episode into an isolated agent workspace (container, GPU for MHR/SOMA decode).

Allowed: the monocular exo video, the object mesh, the released ground plane, the task text, the frame-0 reference
human (organiser-approved, frame 0 only) and our own reconstruction (DB1 = CARI4D + JC1 + depth blend).
NOT exported: any clean trajectory / reference pose of frames > 0 (data/*.parquet, ref_poses, diag_only_ref_human),
support_surface, other camera views. The agent runs in a --containall container that only sees this workspace.

Object poses are given as T_world_mesh (4x4, maps mesh-file coordinates to the released world frame, OpenCV world,
y-down). eval converts the agent's output exactly as eval_official.py converts ours.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import h5py
import numpy as np
import torch
import trimesh

sys.path.insert(0, "/workspace/v2d_cari4d/lib/cari4d")
sys.path.insert(0, "/home/hanzhou/Projects/v2d/hoi_extraction/scripts")
from eval_official import bundle_path  # noqa: E402

V2D = "/mnt/nas/Group/hanzhou/v2d"
DS = f"{V2D}/dataset/track_2/tier_1_multiview_caption"
FLIP = np.diag([1.0, -1.0, -1.0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", type=int)
    ap.add_argument("--ws", required=True, help="workspace root; inputs go to <ws>/inputs")
    ap.add_argument("--variant", default="DB1_depth_blend")
    ap.add_argument("--align", default=f"{V2D}/logs/JC1_joint_camera_align.json")
    ap.add_argument("--camera", default="JC1_joint_camera", help="camera variant whose per-frame depth factor is applied to the human")
    ap.add_argument("--human-stride", type=int, default=4)
    args = ap.parse_args()
    from lib_mhr.mhr_layer import MHRLayer
    from tools.render_mhr_wild_inference import _params
    import soma

    idx, ep = args.episode, f"episode_{args.episode:06d}"
    meta = {d["episode_index"]: d for d in map(json.loads, open(f"{DS}/meta/episodes_metadata.jsonl"))}[idx]
    out = f"{args.ws}/inputs"
    os.makedirs(out, exist_ok=True)
    shutil.copy(f"{DS}/videos/chunk-000/observation.images.exo_camera/{ep}.mp4", f"{out}/video.mp4")
    shutil.copy(f"{DS}/{meta['mesh']}", f"{out}/object_mesh.glb")
    shutil.copy(f"{DS}/{meta['ground_plane']}", f"{out}/ground_plane.json")
    open(f"{out}/task.txt", "w").write(meta["task"] + "\n")
    json.dump({"episode_index": idx, "object": meta["object"], "frames": meta["frames"]}, open(f"{out}/episode.json", "w"), indent=1)

    b = torch.load(bundle_path(args.variant, ep), map_location="cpu", weights_only=False)
    al = json.load(open(args.align))[str(idx)]
    src = al["src"]
    bs = torch.load(bundle_path(src, ep), map_location="cpu", weights_only=False)
    W = np.asarray(al["sim"], np.float64)
    K = np.asarray(b["observations"]["K_full"], np.float64)[0]
    K0 = np.asarray(bs["observations"]["K_full"], np.float64)[0]  # K of the pipeline run (camera-frame estimates)
    P = np.asarray(b["pr"]["pose_abs"], np.float64)
    F = len(P)
    json.dump({"K": K.tolist(), "K_pipeline": K0.tolist(), "T_world_cam": W.tolist(), "width": 1536, "height": 1152,
               "note": "pinhole, OpenCV camera (x right, y down, z forward); world is the released frame (OpenCV world, y-down)"},
              open(f"{out}/camera.json", "w"), indent=1)

    # per-frame human depth factor of the camera variant (object translation camera variant / source), applied to our
    # human too; with a K change the human is first remapped pixel-preservingly from K0 to K (as build_release.py)
    glb = trimesh.load(f"{out}/object_mesh.glb", force="scene", process=False).dump(concatenate=True)
    S = np.asarray(trimesh.bounds.oriented_bounds(glb)[0], np.float64)

    def remap(X):
        z = X[..., 2:3]
        uv = X[..., :2] / z * K0[0, 0] + K0[:2, 2]
        zn = z * K[0, 0] / K0[0, 0]
        return np.concatenate([(uv - K[:2, 2]) / K[0, 0] * zn, zn], -1)
    tj = (np.asarray(torch.load(bundle_path(args.camera, ep), map_location="cpu", weights_only=False)["pr"]["pose_abs"], np.float64) @ S)[:, :3, 3]
    ts = (np.asarray(bs["pr"]["pose_abs"], np.float64) @ S)[:, :3, 3]
    a = np.linalg.norm(tj, axis=1) / np.linalg.norm(remap(ts), axis=1)
    a = np.where(np.isfinite(a), a, 1.0)

    dev = torch.device("cuda")
    mhr = MHRLayer.from_mhr_assets(mhr_assets_root=f"{V2D}/weights/sam3d_body", device="cuda")
    V, J = [], []
    with torch.no_grad():
        for s0 in range(0, F, 128):
            o = mhr.mhr_forward(_params(bs["pr"], slice(s0, min(s0 + 128, F)), dev))
            V.append(o.vertices.cpu().numpy().astype(np.float64)[:, ::args.human_stride])
            J.append(o.keypoints.cpu().numpy().astype(np.float64))
    V, J = np.concatenate(V), np.concatenate(J)
    root = V.mean(1, keepdims=True)

    def to_w(X):  # rigid body move: root along its (remapped) ray by a_t, body keeps its MHR size
        X = X - root + a[:, None, None] * remap(root)
        return X @ W[:3, :3].T + W[:3, 3]
    cp = 1 / (1 + np.exp(-np.asarray(bs["pr"]["contact_logits"], np.float64)))

    # CARI4D pose_abs is cam_T_aligned (oriented-bounds frame); the mesh file / reference frame is glb = aligned @ S
    to_world = lambda T: W[None] @ T @ S
    np.savez_compressed(f"{out}/reconstruction.npz",
                        T_world_mesh=to_world(P).astype(np.float32),
                        T_cam_mesh_foundationpose=(np.asarray(bs["in"]["pose_abs"], np.float64) @ S).astype(np.float32),
                        T_cam_mesh_coconet=(np.asarray(bs["pr"]["pose_abs"], np.float64) @ S).astype(np.float32),
                        human_vertices_world=to_w(V).astype(np.float32),
                        human_keypoints_world=to_w(J).astype(np.float32),
                        contact_prob_right_left=cp.astype(np.float32))

    sl = soma.SOMALayer(identity_model_type="mhr", device="cuda")
    r0 = np.load(f"{V2D}/ref_human_f0/{ep}.npz")  # frame 0 only (organiser-approved)
    with torch.no_grad():
        o = sl(torch.tensor(r0["pose"]).float().view(1, 77, 3).to(dev), torch.tensor(r0["identity"]).float()[None].to(dev),
               scale_params=torch.tensor(r0["scale"]).float()[None].to(dev), transl=torch.tensor(r0["transl"]).float()[None].to(dev))
    g0 = o.vertices[0].cpu().numpy().astype(np.float64) @ FLIP
    np.savez_compressed(f"{out}/frame0_reference_human.npz", vertices_world=g0.astype(np.float32))

    # masks of the pipeline run the reconstruction came from (M1 prompt re-run, regular run, or the rerun of a failed one)
    cands = ([f"{V2D}/experiments/M1_prompt_v2/work/{ep}"] if src == "M1" else []) + [
        f"{V2D}/outputs/cari4d/{ep}/work", f"{V2D}/outputs/cari4d_failed/{ep}_work"]
    mpath = next(f"{c}/masks_k0.h5" for c in cands if os.path.isfile(f"{c}/masks_k0.h5"))
    with h5py.File(mpath, "r") as f, h5py.File(f"{out}/masks.h5", "w") as g:
        grp = f[ep]
        om = np.stack([grp[f"{i:06d}-k0.obj_rend_mask.png"][()] > 0 for i in range(F)])
        pm = np.stack([grp[f"{i:06d}-k0.person_mask.png"][()] > 0 for i in range(F)])
        g.create_dataset("object", data=om, compression="gzip")
        g.create_dataset("person", data=pm, compression="gzip")
    print("exported", ep, "frames", F, "src", src, flush=True)


if __name__ == "__main__":
    main()
