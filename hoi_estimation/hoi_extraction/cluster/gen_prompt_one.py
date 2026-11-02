#!/usr/bin/env python3
"""Generate a SAM2 prompts JSON for one episode inside the v2d_gdino image.

Reads one frame with OpenCV, runs Grounding DINO for "person" (object_id 0)
and the episode object (object_id 1), and writes the v2d Sam2Prompts schema.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

import cv2
import numpy as np


def read_frame(video_path: str, frame_index: int):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_index} from {video_path}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--object_prompt", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--fallback_threshold", type=float, default=0.2)
    parser.add_argument("--search_frames", type=int, default=12, help="frames sampled across the video when frame_index has no detection")
    args = parser.parse_args()

    from v2d.grounding_dino.lib.image_to_object_bboxes import image_to_bboxes

    cap = cv2.VideoCapture(args.video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    tmpdir = tempfile.mkdtemp()

    def detect(frame_index: int, text: str, threshold: float):
        image = os.path.join(tmpdir, f"{frame_index:06d}.png")
        if not os.path.exists(image):
            cv2.imwrite(image, read_frame(args.video_path, frame_index))
        detections = image_to_bboxes(
            image_path=image,
            output_path=f"{image}.{abs(hash((text, threshold)))}.json",
            prompt=text,
            model_dir=args.model_dir,
            box_threshold=threshold,
        )
        return detections[0] if detections else None  # sorted by confidence desc

    # Prefer the requested frame (original behaviour). If the object is not detected there
    # (thin/occluded objects), sample frames across the video and take the most confident
    # detection; SAM2 propagates both forwards and backwards from the prompt frame.
    candidates = sorted({int(round(x)) for x in np.linspace(0, n_frames - 1, args.search_frames)} - {args.frame_index})
    prompts = []
    for object_id, text in ((0, "person"), (1, args.object_prompt)):
        frame_index, best = args.frame_index, detect(args.frame_index, text, args.box_threshold)
        if best is None:
            for threshold in (args.box_threshold, args.fallback_threshold):
                found = [(det, fi) for fi in candidates if (det := detect(fi, text, threshold)) is not None]
                if found:
                    best, frame_index = max(found, key=lambda item: item[0]["confidence"])
                    break
        if best is None:
            raise RuntimeError(f"no detection for prompt '{text}' in {args.video_path} (searched {len(candidates) + 1} frames)")
        print(f"[prompt] {text}: frame={frame_index} conf={best['confidence']} box={best['box']}")
        prompts.append({
            "frame_index": frame_index,
            "object_id": object_id,
            "points": None,
            "point_labels": None,
            "box": best["box"],
            "mask_path": None,
        })

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump({"prompts": prompts}, f, indent=2)
    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
