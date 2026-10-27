#!/usr/bin/env python3
"""SAM2 prompt generation with size-consistent box selection (runs in the v2d_gdino image).

Differences from gen_prompt_one.py: the object is detected with both its name and the dataset's
descriptive prompt over several frames, and every candidate box is scored by
    confidence * size_consistency,
where size_consistency compares the box diagonal with the diagonal the object should have at that
distance: fx * object_diag_m / median MoGe depth inside the box. Boxes far too large (e.g. the
whole floor) or too small are rejected, and candidates far (in 3D) from the detected person are
down-weighted, since the task object is the one the person manipulates (e.g. a second monitor on a
distant desk). Uses only released data
(video, object mesh size) and our own MoGe depth.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import tempfile

import cv2
import numpy as np


def read_frame(video_path, frame_index):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_index} from {video_path}")
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video_path", required=True)
    ap.add_argument("--object_prompt", required=True)
    ap.add_argument("--object_description", default="")
    ap.add_argument("--object_diag_m", type=float, required=True, help="diagonal of the object mesh bounding box (m)")
    ap.add_argument("--raw_depth", required=True, help="MoGe raw depth .npy (uint16 mm, T x H x W)")
    ap.add_argument("--intrinsics", required=True, help="MoGe intrinsics.pkl")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--search_frames", type=int, default=12)
    ap.add_argument("--box_threshold", type=float, default=0.25)
    ap.add_argument("--size_sigma", type=float, default=0.35, help="log-ratio tolerance of the size score")
    ap.add_argument("--min_ratio", type=float, default=0.3)
    ap.add_argument("--max_ratio", type=float, default=1.6)
    ap.add_argument("--near_person_m", type=float, default=1.5, help="no proximity penalty within this 3D distance of the person")
    args = ap.parse_args()

    from v2d.grounding_dino.lib.image_to_object_bboxes import image_to_bboxes

    depth = np.load(args.raw_depth, mmap_mode="r")
    fx = float(pickle.load(open(args.intrinsics, "rb"))["fx"])
    n_frames = depth.shape[0]
    tmpdir = tempfile.mkdtemp()

    def detect(fi, text, thr):
        image = os.path.join(tmpdir, f"{fi:06d}.png")
        if not os.path.exists(image):
            cv2.imwrite(image, read_frame(args.video_path, fi))
        return image_to_bboxes(image_path=image, output_path=f"{image}.{abs(hash((text, thr)))}.json",
                               prompt=text, model_dir=args.model_dir, box_threshold=thr) or []

    frames = sorted({0, *(int(round(x)) for x in np.linspace(0, n_frames - 1, args.search_frames))})
    K = pickle.load(open(args.intrinsics, "rb"))
    fy, cx, cy = float(K["fy"]), float(K["cx"]), float(K["cy"])

    def center3d(b, d):
        x0, y0, x1, y1 = (int(round(b[k])) for k in ("x0", "y0", "x1", "y1"))
        patch = d[max(y0, 0):max(y1, y0 + 1), max(x0, 0):max(x1, x0 + 1)]
        z = float(np.median(patch[patch > 0])) if (patch > 0).any() else np.nan
        u, v = (x0 + x1) / 2, (y0 + y1) / 2
        return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z]), z
    texts = [t for t in (args.object_prompt, args.object_description) if t]
    candidates = []
    for fi in frames:
        d = np.asarray(depth[fi], dtype=np.float32) / 1000.0
        # the manipulated object is the one the person gets close to: penalise candidates far from them
        people = detect(fi, "person", 0.3)
        person_xyz = center3d(people[0]["box"], d)[0] if people else None
        for text in texts:
            for det in detect(fi, text, args.box_threshold):
                b = det["box"]
                x0, y0, x1, y1 = (int(round(b[k])) for k in ("x0", "y0", "x1", "y1"))
                patch = d[max(y0, 0):max(y1, y0 + 1), max(x0, 0):max(x1, x0 + 1)]
                z = float(np.median(patch[patch > 0])) if (patch > 0).any() else np.nan
                diag_px = float(np.hypot(x1 - x0, y1 - y0))
                expected = fx * args.object_diag_m / z if np.isfinite(z) and z > 0 else np.nan
                ratio = diag_px / expected if np.isfinite(expected) else np.nan
                ok = np.isfinite(ratio) and args.min_ratio <= ratio <= args.max_ratio
                size_score = float(np.exp(-np.log(ratio) ** 2 / (2 * args.size_sigma ** 2))) if ok else 0.0
                obj_xyz = center3d(b, d)[0]
                dist = float(np.linalg.norm(obj_xyz - person_xyz)) if person_xyz is not None and np.isfinite(obj_xyz).all() else np.nan
                near_score = float(np.exp(-max(0.0, dist - args.near_person_m) ** 2 / (2 * 1.0 ** 2))) if np.isfinite(dist) else 0.5
                candidates.append({"frame_index": fi, "text": text, "box": b, "confidence": float(det["confidence"]),
                                   "depth_m": z, "diag_px": diag_px, "expected_px": expected, "ratio": ratio,
                                   "dist_to_person_m": dist, "score": float(det["confidence"]) * size_score * near_score})
    valid = [c for c in candidates if c["score"] > 0]
    if not valid:
        raise RuntimeError(f"no size-consistent detection for '{args.object_prompt}' ({len(candidates)} candidates)")
    best = max(valid, key=lambda c: c["score"])
    person = detect(best["frame_index"], "person", 0.3)
    if not person:
        person = detect(0, "person", 0.3)
        pframe = 0
    else:
        pframe = best["frame_index"]
    if not person:
        raise RuntimeError("no person detection")
    prompts = [
        {"frame_index": pframe, "object_id": 0, "points": None, "point_labels": None, "box": person[0]["box"], "mask_path": None},
        {"frame_index": best["frame_index"], "object_id": 1, "points": None, "point_labels": None, "box": best["box"], "mask_path": None},
    ]
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    json.dump({"prompts": prompts}, open(args.output_path, "w"), indent=2)
    json.dump({"selected": best, "candidates": sorted(candidates, key=lambda c: -c["score"])[:15]},
              open(args.output_path.replace(".json", "_candidates.json"), "w"), indent=1, default=float)
    print(f"[prompt] object: frame={best['frame_index']} text='{best['text'][:40]}' conf={best['confidence']:.2f} "
          f"ratio={best['ratio']:.2f} depth={best['depth_m']:.2f}m dist_person={best['dist_to_person_m']:.2f}m score={best['score']:.3f} ({len(valid)}/{len(candidates)} size-consistent)")


if __name__ == "__main__":
    main()
