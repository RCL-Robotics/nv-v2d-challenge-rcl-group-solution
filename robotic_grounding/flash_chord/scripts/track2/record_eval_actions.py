# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Record the official evaluator's raw observations and deterministic actions (STOCK runtime only).

Run this with the official FlashCHORD runtime on ``PYTHONPATH`` (never the DEV tree)::

    PYTHONPATH=$STOCK/src python scripts/track2/record_eval_actions.py \
        --checkpoint export/model_300.safetensors --steps 100 --output eval_actions.npz --runtime-root $STOCK

It rebuilds the environment exactly as ``flash_chord.evaluation.harness.evaluate_checkpoint`` does
for ``reset_mode="explicit"`` and ``start_frame=0`` (harness.py:376-455), using only stock functions:
``apply_evaluation_composition``, ``_evaluation_curriculum_stage``, ``_ppo_policy_for_inference``
and ``env.reset(frame_id=0)``. It then steps the deterministic policy and saves, per step, the raw
observation the policy saw and the action it produced. The same observations are also re-run
through the evaluator's policy function at ``highest`` matmul precision, because JAX's default
float32 matmul on recent NVIDIA GPUs can be TF32-like (measured ~3e-4 relative error for a large
batch on the RTX PRO 6000; the one-world evaluator batch measured full float32).

Output ``.npz`` keys: ``observations (S, W, obs)``, ``actions (S, W, act)``,
``actions_highest_precision (S, W, act)``, plus provenance strings.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--world-count", type=int, default=1)
    parser.add_argument("--source-root", type=Path, help="as evaluation.source_root")
    parser.add_argument(
        "--runtime-root",
        type=Path,
        help="official runtime root; the imported flash_chord must live in <runtime-root>/src",
    )
    args = parser.parse_args()

    import flash_chord

    module = Path(flash_chord.__file__).resolve()
    if args.runtime_root is not None:
        expected = (args.runtime_root.resolve() / "src" / "flash_chord").resolve()
        if module.parent != expected:
            parser.error(f"imported {module}; expected the official runtime package in {expected}")

    import jax
    import jax.numpy as jnp
    import numpy as np
    from dataclasses import replace
    from omegaconf import OmegaConf

    from flash_chord.configuration import instantiate_typed
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.envs.rl import RLEnv, RLEnvConfig
    from flash_chord.evaluation.harness import (
        _evaluation_curriculum_stage,
        _ppo_policy_for_inference,
        _resolve_parquet,
        apply_evaluation_composition,
        checkpoint_training_algorithm,
        evaluation_code_sha256,
        file_sha256,
    )
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.scene.setup import setup_scene
    from flash_chord.training.checkpoint import read_checkpoint_metadata
    from flash_chord.training.checkpoint_metadata import checkpoint_config
    from flash_chord.training.environment import WarpRLEnv
    from flash_chord.training.flash_sac.runner import resolve_jax_device
    from flash_chord.training.ppo.config import TrainingConfig as PPOTrainingConfig
    from flash_chord.training.ppo.evaluation import EvaluationConfig as PPOEvaluationConfig
    from flash_chord.training.ppo.evaluation import policy_action as ppo_policy_action

    # --- harness.evaluate_checkpoint, reset_mode="explicit", start_frame=0 ---
    checkpoint_path = args.checkpoint.expanduser().resolve()
    metadata = read_checkpoint_metadata(checkpoint_path)
    config = checkpoint_config(metadata)
    if checkpoint_training_algorithm(config) != "ppo":
        parser.error("only PPO checkpoints are supported")
    apply_evaluation_composition(config, world_count=args.world_count, reset_mode="explicit")
    config["task"]["parquet"] = _resolve_parquet(config, args.source_root)

    cfg = OmegaConf.create(config)
    env_config = instantiate_typed(cfg.env, RLEnvConfig)
    embodiment = instantiate_typed(cfg.embodiment, Embodiment)
    collision = instantiate_typed(cfg.collision, CollisionPolicy)
    training = replace(instantiate_typed(cfg.training, PPOTrainingConfig), world_count=args.world_count)
    device = resolve_jax_device(training.device)

    setup = setup_scene(
        parquet=cfg.task.parquet,
        control_fps=env_config.sim.fps,
        motion_speed=cfg.task.motion_speed,
        embodiment=embodiment,
        collision=collision,
        world_count=cfg.scene.world_count,
        include_support=cfg.scene.support,
        decompose_objects=cfg.scene.decompose_objects,
        object_scale_min=cfg.scene.get("object_scale_min", 1.0),
        object_scale_max=cfg.scene.get("object_scale_max", 1.0),
        object_scale_seed=cfg.scene.get("object_scale_seed", 0),
        contact_friction=cfg.scene.get("contact_friction", 1.0),
        object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0),
        source_frame_playback=bool(cfg.task.get("source_frame_playback", False)),
        motion_start_frame=int(cfg.task.get("motion_start_frame", 0)),
        motion_end_frame=int(cfg.task.get("motion_end_frame", -1)),
    )
    steps = min(int(args.steps), int(setup.reference.num_frames))

    env = RLEnv(setup.scene, setup.reference, config=env_config)
    env.reset()
    env.capture_step()
    jax_env = WarpRLEnv(env, publish_diagnostics=True, publish_step_metrics=True)
    if training.curriculum is not None:
        stage_index = training.curriculum.stage_index(max(int(metadata["iteration"]) - 1, 0))
        jax_env.set_curriculum(_evaluation_curriculum_stage(training.curriculum.stages[stage_index], "explicit"))
    actor_state, policy_action, key = _ppo_policy_for_inference(checkpoint_path, training, jax_env)
    env.reset(frame_id=0)
    reset_observation = jnp.asarray(env.observation.numpy().reshape(env.world_count, env.observation_dim))
    observation = jax.device_put(jnp.array(reset_observation, copy=True), device)
    # --- end of the harness replica ---

    observations, actions = [], []
    for _ in range(steps):
        action, key = policy_action(actor_state, observation, key)
        observations.append(np.asarray(observation))
        actions.append(np.asarray(action))
        observation = jax_env.step(action).observation

    observation_batch = np.stack(observations)
    evaluation = PPOEvaluationConfig(checkpoint=str(checkpoint_path), deterministic=True)
    with jax.default_matmul_precision("highest"):
        highest = np.asarray(
            ppo_policy_action(
                actor_state,
                jnp.asarray(observation_batch.reshape(-1, observation_batch.shape[-1])),
                training,
                evaluation,
            )
        ).reshape(len(observations), args.world_count, -1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        observations=observation_batch,
        actions=np.stack(actions),
        actions_highest_precision=highest,
        checkpoint=str(checkpoint_path),
        checkpoint_sha256=file_sha256(checkpoint_path),
        eval_code_sha256=evaluation_code_sha256(),
        flash_chord_module=str(module),
        jax_backend=jax.default_backend(),
    )
    print(
        f"recorded {steps} steps x {args.world_count} worlds from {checkpoint_path} "
        f"(flash_chord={module}, backend={jax.default_backend()}) -> {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
