# SPDX-License-Identifier: Apache-2.0
"""Replay one retargeted Track 2 motion with zero policy actions and report how far it plays.

Starts at reference frame 0 (like ``scripts/rsl_rl/eval.py``), keeps the virtual object
controller (VOC) at ``--voc`` with the curriculum disabled, steps until the first
termination/truncation, and writes a JSON record with the step reached and which
termination terms fired. A healthy sequence ends on the trajectory-end timeout close to
``frames / src_fps / step_dt`` steps.
"""

import argparse
import json
import math
import sys

from isaaclab.app import AppLauncher

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", default="SonicG1-ReconBody-v0")
parser.add_argument("--motion_file", required=True)
parser.add_argument("--frames", type=int, required=True, help="Number of source frames in the motion")
parser.add_argument("--src_fps", type=float, default=30.0)
parser.add_argument("--voc", type=float, default=1.0, help="Virtual object controller scale (1 = full assistance)")
parser.add_argument("--out", required=True)
parser.add_argument("--full_clip", action="store_true", help="Disable failure terminations (keep the trajectory-end timeout) and log tracking metrics over the whole clip")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from robotic_grounding.tasks import *  # noqa: E402,F401,F403
from robotic_grounding.tasks.scene_utils import SceneConfig, apply_scene_config  # noqa: E402


def main() -> None:
    cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
    cfg.motion_file = args.motion_file
    apply_scene_config(cfg, SceneConfig.from_motion_file(cfg.motion_file))
    cmd = cfg.commands.motion
    cmd.always_reset_to_first_frame = True
    cmd.initial_virtual_object_control_curriculum_scale = args.voc
    if hasattr(cmd, "voc_reset_scale"):
        cmd.voc_reset_scale = args.voc
    cfg.curriculum = None
    if args.full_clip:
        for name, term in list(vars(cfg.terminations).items()):
            if term is not None and not getattr(term, "time_out", False):
                setattr(cfg.terminations, name, None)
    env = gym.make(args.task, cfg=cfg)
    uenv = env.unwrapped
    steps_expected = math.ceil(args.frames / args.src_fps / uenv.step_dt)
    env.reset()
    actions = torch.zeros(env.action_space.shape, device=uenv.device)
    first = None
    step = 0
    motion = uenv.command_manager.get_term("motion")
    keys = [k for k in motion.metrics if k != "virtual_object_controller_scale_factor"]
    series = {k: [] for k in keys}
    for step in range(1, steps_expected + 100):
        with torch.inference_mode():
            _, _, terminated, truncated, _ = env.step(actions)
        if not (bool(terminated[0]) or bool(truncated[0])):
            for k in keys:
                series[k].append(float(motion.metrics[k][0]))
        if bool(terminated[0]) or bool(truncated[0]):
            tm = uenv.termination_manager
            fired = [name for name in tm.active_terms if bool(tm.get_term(name)[0])]
            first = {"step": step, "terms": fired, "time_out": bool(truncated[0])}
            break
    result = {
        "motion_file": args.motion_file,
        "frames": args.frames,
        "step_dt": uenv.step_dt,
        "steps_expected": steps_expected,
        "steps_run": step,
        "first_termination": first,
        "voc": args.voc,
        "full_clip": args.full_clip,
    }
    import numpy as np
    stats = {}
    for k, v in series.items():
        if v:
            a = np.asarray(v)
            stats[k] = {"mean": round(float(a.mean()), 4), "p95": round(float(np.percentile(a, 95)), 4), "max": round(float(a.max()), 4)}
    thresholds = {"anchor_position_error": 0.7, "left_hand_wrist_position_error": 0.15, "right_hand_wrist_position_error": 0.15, "object_body_position_error": 0.10}
    result["metrics"] = stats
    result["frac_steps_over_train_threshold"] = {k: round(float((np.asarray(series[k]) > t).mean()), 4) for k, t in thresholds.items() if series.get(k)}
    with open(args.out, "w") as f:
        json.dump(result, f)
    print("REPLAY_RESULT", json.dumps(result), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
