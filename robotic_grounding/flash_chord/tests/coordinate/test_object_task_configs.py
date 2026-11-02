# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configs and E2 export for the dev-only ReconBody ``object_task`` term (objectives/recon_body.py).

Existing experiments must compose exactly as before (no ``object_task`` key anywhere, stock objective); the
opt-in group files ``objective/recon_body_objtask`` + ``curriculum/recon_body_rsi_objtask`` must differ from
their bases only by the new keys; and E2 export must remove every new key so the stock runtime can load it.
"""

from __future__ import annotations

import dataclasses

import pytest

from flash_chord.configuration import compose_config, instantiate_typed, resolved_dict

PARQUET = "task.parquet=/tmp/reference.parquet"
RECON_BODY_EXPERIMENTS = (
    "g1_recon_body_ppo",
    "g1_recon_body_rsi",
    "g1_recon_body_coord",
    "g1_recon_body_coord_rsi",
    "g1_recon_body_coord_rsi_wide",
    "g1_recon_body_flash_sac",
)
C3 = "g1_recon_body_coord_rsi"
OBJECT_TASK_KEYS = ("object_task_weight", "object_task_position_beta", "object_task_rotation_beta")
#: The post-release options of object_task (tests/objectives/test_recon_body_object_task_release.py).
OBJECT_TASK_RELEASE_KEYS = ("object_task_position_only_after_release", "no_drop_weight", "no_drop_margin_m")
OBJTASK_GROUPS = ("objective=recon_body_objtask", "curriculum=recon_body_rsi_objtask")
#: ReconBodyObjectiveConfig fields of the official runtime (video_to_data 8c778f9 objectives/recon_body.py).
STOCK_RECON_BODY_OBJECTIVE_FIELDS = (
    "termination_weight",
    "action_rate_l2_weight",
    "action_l2_weight",
    "soft_joint_limit_weight",
    "pelvis_position_weight",
    "pelvis_orientation_weight",
    "body_joint_position_weight",
    "object_position_weight",
    "progress_weight",
    "palm_position_weight",
    "palm_orientation_weight",
    "force_closure_weight",
    "pelvis_position_std",
    "pelvis_orientation_std",
    "body_joint_position_std",
    "object_position_std",
    "palm_position_std",
    "palm_orientation_std",
    "soft_joint_limit_factor",
    "force_closure_min_support",
    "num_wrench_basis",
    "num_friction_cone_edges",
    "friction_coefficient",
    "wrench_basis_seed",
    "sides",
)


def _compose(experiment: str, *overrides: str, config_name: str = "train") -> dict:
    return resolved_dict(compose_config(config_name, [f"experiment={experiment}", PARQUET, *overrides]))


def _paths(value, prefix=""):
    """Every flattened key path of a nested dict/list."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _paths(child, f"{prefix}{key}.")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _paths(child, f"{prefix}{index}.")
    else:
        yield prefix.rstrip(".")


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for key, child in value.items():
            out.update(_flatten(child, f"{prefix}{key}."))
        return out
    if isinstance(value, list) and any(isinstance(child, dict) for child in value):
        out = {}
        for index, child in enumerate(value):
            out.update(_flatten(child, f"{prefix}{index}."))
        return out
    return {prefix.rstrip("."): value}


@pytest.mark.parametrize("experiment", RECON_BODY_EXPERIMENTS)
def test_existing_experiments_have_no_object_task_and_build_the_stock_objective(experiment):
    from flash_chord.envs.rl import RLEnvConfig
    from flash_chord.objectives.recon_body import RECON_BODY_OBJECTIVE_TERM_NAMES, ReconBodyObjectiveConfig

    config_name = "train_flash_sac" if "flash_sac" in experiment else "train"
    composed = compose_config(config_name, [f"experiment={experiment}", PARQUET])
    resolved = resolved_dict(composed)
    assert not [path for path in _paths(resolved) if "object_task" in path]
    env = instantiate_typed(composed.env, RLEnvConfig)
    assert isinstance(env.objective, ReconBodyObjectiveConfig)
    assert not env.objective.object_task_enabled
    assert env.objective.term_names == RECON_BODY_OBJECTIVE_TERM_NAMES
    for stage in resolved["training"]["curriculum"]["stages"]:
        assert set(stage["objective_weights"]) == set(RECON_BODY_OBJECTIVE_TERM_NAMES)


def test_objtask_group_files_differ_from_c3_only_by_the_object_task_keys():
    from flash_chord.envs.rl import RLEnvConfig
    from flash_chord.training.ppo.config import TrainingConfig

    base = _flatten(_compose(C3))
    objtask_config = compose_config("train", [f"experiment={C3}", PARQUET, *OBJTASK_GROUPS])
    objtask = _flatten(resolved_dict(objtask_config))
    keys = base.keys() | objtask.keys()
    differing = {key for key in keys if base.get(key, "<absent>") != objtask.get(key, "<absent>")}
    objective_values = dict(zip(OBJECT_TASK_KEYS, (5.0, 20.0, 5.0), strict=True))
    expected = {
        **{
            f"{root}.{key}": value
            for root in ("objective", "env.objective")
            for key, value in objective_values.items()
        },
        **{f"{root}.stages.0.objective_weights.object_task": 5.0 for root in ("curriculum", "training.curriculum")},
    }
    assert differing == set(expected)
    assert {key: objtask[key] for key in expected} == expected
    assert all(key not in base for key in expected)

    env = instantiate_typed(objtask_config.env, RLEnvConfig)
    assert env.objective.object_task_enabled
    assert env.objective.term_names[-1] == "object_task"
    assert (env.objective.object_task_position_beta, env.objective.object_task_rotation_beta) == (20.0, 5.0)
    training = instantiate_typed(objtask_config.training, TrainingConfig)
    for stage in training.curriculum.stages:
        weights = stage.weights_for(env.objective.term_names)  # exact-name validation as in set_curriculum
        assert weights[-1] == 5.0


def test_cli_overrides_match_the_group_files():
    """The Hydra override path used for smoke runs on any single-stage RSI config."""
    cli = _compose(
        C3,
        "+objective.object_task_weight=5.0",
        "+objective.object_task_position_beta=20.0",
        "+objective.object_task_rotation_beta=5.0",
        "+curriculum.stages.0.objective_weights.object_task=5.0",
    )
    groups = _compose(C3, *OBJTASK_GROUPS)
    assert _flatten(cli) == _flatten(groups)


def test_mismatched_curriculum_and_objective_fail_name_validation():
    from flash_chord.envs.rl import RLEnvConfig
    from flash_chord.training.ppo.config import TrainingConfig

    only_objective = compose_config("train", [f"experiment={C3}", PARQUET, "objective=recon_body_objtask"])
    env = instantiate_typed(only_objective.env, RLEnvConfig)
    stage = instantiate_typed(only_objective.training, TrainingConfig).curriculum.stages[0]
    with pytest.raises(ValueError, match=r"missing=\['object_task'\]"):
        stage.weights_for(env.objective.term_names)
    only_curriculum = compose_config("train", [f"experiment={C3}", PARQUET, "curriculum=recon_body_rsi_objtask"])
    env = instantiate_typed(only_curriculum.env, RLEnvConfig)
    stage = instantiate_typed(only_curriculum.training, TrainingConfig).curriculum.stages[0]
    with pytest.raises(ValueError, match=r"unexpected=\['object_task'\]"):
        stage.weights_for(env.objective.term_names)


def test_dev_objective_fields_are_the_stock_fields_plus_the_export_dev_keys():
    import flash_chord.training.ppo.coordinate.export as e2
    from flash_chord.objectives.recon_body import (
        RECON_BODY_OBJECT_NO_DROP_TERM_NAME,
        RECON_BODY_OBJECT_TASK_TERM_NAME,
        ReconBodyObjectiveConfig,
    )

    fields = tuple(field.name for field in dataclasses.fields(ReconBodyObjectiveConfig))
    assert fields == STOCK_RECON_BODY_OBJECTIVE_FIELDS + tuple(e2.DEV_OBJECTIVE_KEYS)
    assert e2.DEV_OBJECTIVE_KEYS == OBJECT_TASK_KEYS + OBJECT_TASK_RELEASE_KEYS
    assert e2.DEV_OBJECTIVE_TERMS == (RECON_BODY_OBJECT_TASK_TERM_NAME, RECON_BODY_OBJECT_NO_DROP_TERM_NAME)


def test_export_strips_the_object_task_term_for_the_stock_runtime():
    import flash_chord.training.ppo.coordinate.export as e2
    from flash_chord.objectives.recon_body import RECON_BODY_OBJECTIVE_TERM_NAMES

    saved = _compose(C3, *OBJTASK_GROUPS)
    exported, notes = e2.stock_config_with_notes(saved)
    object_task_removed = [key for key in notes["removed_keys"] if "object_task" in key]
    assert sorted(object_task_removed) == sorted(
        [f"{root}.{key}" for root in ("objective", "env.objective") for key in OBJECT_TASK_KEYS]
        + [f"{root}.stages.0.objective_weights.object_task" for root in ("curriculum", "training.curriculum")]
    )
    for objective in (exported["objective"], exported["env"]["objective"]):
        assert set(objective) - {"_target_"} == set(STOCK_RECON_BODY_OBJECTIVE_FIELDS)
    for curriculum in (exported["curriculum"], exported["training"]["curriculum"]):
        for stage in curriculum["stages"]:
            assert tuple(stage["objective_weights"]) == RECON_BODY_OBJECTIVE_TERM_NAMES
    assert not [path for path in _paths(exported) if "object_task" in path]

    # Without the term, export is unchanged: the same config and notes as the C3 export.
    stock_exported, stock_notes = e2.stock_config_with_notes(_compose(C3))
    assert not [key for key in stock_notes["removed_keys"] if "object_task" in key]
    assert _flatten(exported) == _flatten(stock_exported)
    assert notes["removed_keys"][: len(stock_notes["removed_keys"])] == stock_notes["removed_keys"]
    assert {key: value for key, value in notes.items() if key != "removed_keys"} == {
        key: value for key, value in stock_notes.items() if key != "removed_keys"
    }
