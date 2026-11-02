#!/usr/bin/env python3
"""Overlay object-pose trajectories on the episode video (CPU, painter's-algorithm flat shading).

Each row: full frame with the mesh overlay + a zoomed crop around the object. One column per pose file.
Usage: render_compare.py <workspace> <out.mp4> label=poses.npz [label=poses.npz ...] [--stride 2] [--scale 0.5] [--camera camera.json]
"""
import argparse
import json

import cv2
import numpy as np
import trimesh


def draw(img, K, Tcm, V, Fc, color):
    Vc = V @ Tcm[:3, :3].T + Tcm[:3, 3]
    if not np.isfinite(Vc).all() or (Vc[:, 2] < 0.05).any():
        return img, None
    uv = Vc[:, :2] / Vc[:, 2:3] * np.diag(K)[:2] + K[:2, 2]
    tri = Vc[Fc]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    shade = np.abs(nrm @ np.array([0.3, -0.5, -0.8]) / np.linalg.norm([0.3, -0.5, -0.8]))
    front = (nrm * tri.mean(1)).sum(1) < 0  # back-face culling (camera at origin)
    order = np.where(front)[0][np.argsort(-tri[front, :, 2].mean(1))]
    layer = img.copy()
    pts = np.round(uv[Fc] * 4).astype(np.int32)  # 2 bits sub-pixel
    for i in order:
        c = tuple(int(x * (0.35 + 0.65 * shade[i])) for x in color)
        cv2.fillConvexPoly(layer, pts[i], c, lineType=cv2.LINE_AA, shift=2)
    out = cv2.addWeighted(layer, 0.6, img, 0.4, 0)
    return out, uv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ws")
    ap.add_argument("out")
    ap.add_argument("poses", nargs="+")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--crop", type=int, default=260)
    ap.add_argument("--camera", default=None, help="camera.json with K, T_world_cam (default: <ws>/inputs/camera.json)")
    a = ap.parse_args()
    cam = json.load(open(a.camera or f"{a.ws}/inputs/camera.json"))
    K, cw = np.array(cam["K"]), np.linalg.inv(np.array(cam["T_world_cam"]))
    mesh = trimesh.load(f"{a.ws}/inputs/object_mesh.glb", force="scene", process=False).dump(concatenate=True)
    V, Fc = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    runs = [(p.split("=", 1)[0], np.load(p.split("=", 1)[1])["T_world_mesh"].astype(float)) for p in a.poses]
    colors = [(60, 200, 255), (255, 120, 60), (120, 255, 120)]
    cap = cv2.VideoCapture(f"{a.ws}/inputs/video.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer, fi = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % a.stride == 0:
            cols = []
            for (label, T), col in zip(runs, colors):
                if fi >= len(T):
                    continue
                img, uv = draw(frame, K, cw @ T[fi], V, Fc, col)
                cx, cy = (uv.mean(0) if uv is not None else np.array(img.shape[1::-1]) / 2).astype(int)
                h = a.crop // 2
                pad = cv2.copyMakeBorder(img, h, h, h, h, cv2.BORDER_CONSTANT)
                crop = cv2.resize(pad[cy:cy + a.crop, cx:cx + a.crop], None, fx=img.shape[0] * a.scale / a.crop, fy=img.shape[0] * a.scale / a.crop)
                small = cv2.resize(img, None, fx=a.scale, fy=a.scale)
                cv2.rectangle(small, (0, 0), (small.shape[1], 48), (0, 0, 0), -1)
                cv2.putText(small, f"{label}  frame {fi}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                cols.append(np.concatenate([small, crop], 1))
            canvas = np.concatenate(cols, 0)
            if writer is None:
                writer = cv2.VideoWriter(a.out + ".tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps / a.stride, canvas.shape[1::-1])
            writer.write(canvas)
        fi += 1
    writer.release()
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", a.out + ".tmp.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26", a.out], check=True)
    import os
    os.remove(a.out + ".tmp.mp4")


if __name__ == "__main__":
    main()
