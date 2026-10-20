# SPDX-License-Identifier: Apache-2.0
"""Convert V2D challenge Track 2 episodes (LeRobot v2.1) into ``soma_to_g1.py`` input folders.

For every selected episode this writes ``<motion-root>/whole_body/soma/sequence_id=<seq>/`` with

* ``soma_params.npz`` -- the SOMA exporter schema read by ``retarget/read_soma.py``:
  ``poses`` (T, 77, 3) local rotation vectors, ``transl`` (T, 3), ``joint_names``,
  ``identity_model_type`` (``mhr``), ``identity_coeffs``, ``scale_params``,
  ``keep_root=False``, ``absolute_pose=False``, ``rotation_repr=rotvec``, ``unit=meters``.
  Track 2 stores exactly these SOMA-X parameters per frame (same 77-joint order as the
  in-repo examples). ``bone_length_flexibles`` is not read by the retargeter and is dropped.
* ``poses.npy`` -- (T, 4, 4) ``world_T_object`` in the source OpenCV world frame (what
  ``soma_to_g1.load_data`` expects). Frames whose ``observation.object.visible`` is False are
  zero-filled in Track 2; they are filled here by position lerp / rotation slerp between the
  nearest visible frames (held constant before the first and after the last visible frame).
* ``ground_plane.json`` -- copied from ``ground_plane/<seq>.json`` (same format).
* ``reconstructed_mesh/output_aligned.glb`` -- copied from ``mesh/<object>/<object>.glb``.

The episode's G1 support surface is copied to
``<motion-root>/whole_body/reconstructed_stage/<seq>_support.usda`` where the scene loader
looks for it. A ``track2_manifest.json`` summarising every converted episode is written to
``<motion-root>/whole_body/soma/``.

Example::

    python scripts/track2/track2_to_soma_inputs.py \
        --track-root /path/to/track_2/tier_1_multiview_caption \
        --motion-root /path/to/human_motion_data --episodes all
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def _fill_invisible_object_poses(pose7: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """Return (T, 4, 4) object poses from [x,y,z,qw,qx,qy,qz], interpolating invisible frames."""
    num_frames = pose7.shape[0]
    valid = visible.astype(bool) & (np.linalg.norm(pose7[:, 3:], axis=1) > 0.5)
    if not valid.any():
        raise ValueError("object is never visible; cannot build object poses")
    idx = np.flatnonzero(valid)
    t = np.arange(num_frames)
    pos = np.stack([np.interp(t, idx, pose7[idx, k]) for k in range(3)], axis=1)
    rots = R.from_quat(pose7[idx, 3:7], scalar_first=True)
    if len(idx) == 1:
        quat = np.repeat(rots.as_quat(scalar_first=True), num_frames, axis=0)
    else:
        quat = Slerp(idx, rots)(np.clip(t, idx[0], idx[-1])).as_quat(scalar_first=True)
    out = np.tile(np.eye(4), (num_frames, 1, 1))
    out[:, :3, :3] = R.from_quat(quat, scalar_first=True).as_matrix()
    out[:, :3, 3] = pos
    return out


def convert_episode(track_root: Path, motion_root: Path, meta: dict, joint_names: list[str], overwrite: bool) -> dict:
    seq = meta["sequence_id"]
    out_dir = motion_root / "whole_body" / "soma" / f"sequence_id={seq}"
    if out_dir.exists() and (out_dir / "soma_params.npz").is_file() and not overwrite:
        return {"sequence_id": seq, "status": "exists"}
    out_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(track_root / f"data/chunk-000/episode_{meta['episode_index']:06d}.parquet")
    col = lambda name: np.asarray(table.column(name).to_pylist(), dtype=np.float32)  # noqa: E731
    pose = col("observation.human.pose")
    num_frames = pose.shape[0]
    if pose.shape[1] != 3 * len(joint_names):
        raise ValueError(f"{seq}: pose width {pose.shape[1]} != 3 x {len(joint_names)} joints")
    frame_index = np.asarray(table.column("frame_index").to_pylist())
    if not np.array_equal(frame_index, np.arange(num_frames)):
        raise ValueError(f"{seq}: frame_index is not contiguous 0..T-1")

    np.savez(
        out_dir / "soma_params.npz",
        poses=pose.reshape(num_frames, len(joint_names), 3),
        transl=col("observation.human.translation"),
        joint_names=np.asarray(joint_names),
        identity_model_type=np.asarray("mhr"),
        identity_coeffs=col("observation.human.identity_coeffs"),
        scale_params=col("observation.human.scale_params"),
        keep_root=np.asarray(False),
        absolute_pose=np.asarray(False),
        rotation_repr=np.asarray("rotvec"),
        unit=np.asarray("meters"),
    )

    visible = np.asarray(table.column("observation.object.visible").to_pylist(), dtype=bool)
    object_poses = _fill_invisible_object_poses(col("observation.object.pose").astype(np.float64), visible)
    np.save(out_dir / "poses.npy", object_poses)

    shutil.copyfile(track_root / meta["ground_plane"], out_dir / "ground_plane.json")
    (out_dir / "reconstructed_mesh").mkdir(exist_ok=True)
    shutil.copyfile(track_root / meta["mesh"], out_dir / "reconstructed_mesh" / "output_aligned.glb")
    stage_dir = motion_root / "whole_body" / "reconstructed_stage"
    stage_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(track_root / meta["support_surface"], stage_dir / f"{seq}_support.usda")

    invisible = int((~visible).sum())
    return {
        "sequence_id": seq,
        "episode_index": meta["episode_index"],
        "object": meta["object"],
        "task": meta.get("task", ""),
        "frames": int(num_frames),
        "invisible_object_frames": invisible,
        "longest_invisible_run": int(max((len(r) for r in np.split(np.flatnonzero(~visible), np.flatnonzero(np.diff(np.flatnonzero(~visible)) > 1) + 1) if len(r)), default=0)),
        "status": "converted",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--track-root", type=Path, required=True, help="Track 2 tier directory (contains meta/, data/, mesh/ ...)")
    parser.add_argument("--motion-root", type=Path, required=True, help="Output human_motion_data root (HUMAN_MOTION_DATA_DIR)")
    parser.add_argument("--episodes", default="all", help="'all' or comma-separated episode indices")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    info = json.loads((args.track_root / "meta/info.json").read_text())
    names = info["features"]["observation.human.pose"]["names"]
    joint_names = [n.rsplit(".", 1)[0] for n in names[::3]]
    metas = [json.loads(line) for line in (args.track_root / "meta/episodes_metadata.jsonl").read_text().splitlines() if line.strip()]
    if args.episodes != "all":
        wanted = {int(e) for e in args.episodes.split(",")}
        metas = [m for m in metas if m["episode_index"] in wanted]

    results = [convert_episode(args.track_root, args.motion_root, m, joint_names, args.overwrite) for m in metas]
    for r in results:
        print(json.dumps(r))
    manifest = args.motion_root / "whole_body" / "soma" / "track2_manifest.json"
    old = json.loads(manifest.read_text()) if manifest.is_file() else {}
    old.update({r["sequence_id"]: r for r in results if r["status"] == "converted"})
    manifest.write_text(json.dumps(old, indent=2))
    print(f"wrote {manifest} ({len(old)} episodes)")


if __name__ == "__main__":
    main()
