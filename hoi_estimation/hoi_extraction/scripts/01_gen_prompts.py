#!/usr/bin/env python3
"""Generate SAM2 prompt JSONs for all Track-2 episodes using Grounding DINO bboxes.

For each episode we run Grounding DINO on a chosen frame with two text prompts
("person" and the episode's object name) and emit a prompts JSON in the
v2d Sam2Prompts schema: object_id 0 = human, object_id 1 = object.

Requires the v2d reconstruction host venv and the v2d_grounding_dino docker image.
Run from anywhere; paths are configured below or via env vars.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

DATASET = Path(os.environ.get(
    "V2D_DATASET", os.environ["V2D_ROOT"] + "/dataset/track_2/tier_1_multiview_caption"))
WORK = Path(os.environ.get("V2D_WORK", os.environ["V2D_ROOT"] + "/work"))
GDINO_WEIGHTS = Path(os.environ.get(
    "V2D_GDINO_WEIGHTS", os.environ["V2D_ROOT"] + "/weights/grounding_dino"))
FRAME_INDEX = int(os.environ.get("V2D_PROMPT_FRAME", "0"))
BOX_THRESHOLD = float(os.environ.get("V2D_BOX_THRESHOLD", "0.3"))

VIDEOS = DATASET / "videos/chunk-000/observation.images.exo_camera"
META = DATASET / "meta/episodes_metadata.jsonl"


def extract_frame(video: Path, frame_index: int, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-noautorotate", "-i", str(video),
        "-vf", f"select=eq(n\\,{frame_index})", "-frames:v", "1", str(out),
    ], check=True, capture_output=True)


def gdino_bbox(image: Path, prompt: str, out_json: Path) -> None:
    """Run grounding dino docker wrapper; writes detection JSON."""
    subprocess.run([
        sys.executable, "-m", "v2d.grounding_dino.docker.run_image_to_object_bboxes",
        "--image_path", str(image), "--output_path", str(out_json),
        "--prompt", prompt, "--model_dir", str(GDINO_WEIGHTS),
        "--box_threshold", str(BOX_THRESHOLD),
    ], check=True)


def largest_box(det_json: Path) -> list[float]:
    d = json.loads(det_json.read_text())
    # tolerate common schemas: {"boxes": [...]} or list of {box: {...}}
    boxes = d["boxes"] if isinstance(d, dict) and "boxes" in d else d
    if isinstance(boxes, dict):  # single detection
        boxes = [boxes]
    best, best_area = None, -1.0
    for b in boxes:
        if isinstance(b, dict):
            bb = b.get("box", b)
            coords = [bb["x1"], bb["y1"], bb["x2"], bb["y2"]] if "x1" in bb else bb
        else:
            coords = b
        x1, y1, x2, y2 = [float(v) for v in coords]
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best, best_area = (x1, y1, x2, y2), area
    if best is None:
        raise RuntimeError(f"no detection in {det_json}")
    return list(best)


def main() -> None:
    episodes = [json.loads(l) for l in META.read_text().splitlines()]
    for ep in episodes:
        idx = ep["episode_index"]
        obj_prompt = ep["object"].replace("_", " ")
        ep_dir = WORK / f"episode_{idx:06d}"
        prompts_path = ep_dir / "prompts.json"
        if prompts_path.exists():
            print(f"[skip] episode {idx:06d} prompts exist")
            continue
        video = VIDEOS / f"episode_{idx:06d}.mp4"
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            frame = td / "frame.png"
            extract_frame(video, FRAME_INDEX, frame)
            person_json, object_json = td / "person.json", td / "object.json"
            gdino_bbox(frame, "person", person_json)
            gdino_bbox(frame, obj_prompt, object_json)
            p_box, o_box = largest_box(person_json), largest_box(object_json)
        prompts = {"prompts": [
            {"frame_index": FRAME_INDEX, "object_id": 0, "points": None,
             "point_labels": None,
             "box": {"x1": p_box[0], "y1": p_box[1], "x2": p_box[2], "y2": p_box[3]},
             "mask_path": None},
            {"frame_index": FRAME_INDEX, "object_id": 1, "points": None,
             "point_labels": None,
             "box": {"x1": o_box[0], "y1": o_box[1], "x2": o_box[2], "y2": o_box[3]},
             "mask_path": None},
        ]}
        ep_dir.mkdir(parents=True, exist_ok=True)
        prompts_path.write_text(json.dumps(prompts, indent=2))
        print(f"[ok] episode {idx:06d}: person={p_box} {ep['object']}={o_box}")


if __name__ == "__main__":
    main()
