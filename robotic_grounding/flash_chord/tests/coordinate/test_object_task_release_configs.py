# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configs and E2 export for the post-release options of the ReconBody ``object_task`` term.

Hydra overrides on top of C3r2 (``experiment=g1_recon_body_coord_rsi_r2``) with a squeegee-style rotation beta:

* position-only after the final release: ``+objective.object_task_position_only_after_release=true``;
* no-drop term: ``+objective.no_drop_weight=W`` and its curriculum stage weight
  ``+curriculum.stages.0.objective_weights.object_no_drop=W`` (W <= 0; like object_task, the term exists only when
  the objective weight is non-zero and training applies the stage weight, so both must be set).

They must add exactly those keys, instantiate, pass the curriculum's exact-name check, and E2 export must remove
every one of them, so the stock runtime sees exactly the export of the same run without them.
"""

from __future__ import annotations

import json

import pytest

from flash_chord.configuration import compose_config, instantiate_typed, resolved_dict

PARQUET = "task.parquet=/tmp/reference.parquet"
BASE = ("experiment=g1_recon_body_coord_rsi_r2", PARQUET, "objective.object_task_rotation_beta=4.5")
POSITION_ONLY = ("+objective.object_task_position_only_after_release=true",)
NO_DROP = ("+objective.no_drop_weight=-50.0", "+curriculum.stages.0.objective_weights.object_no_drop=-50.0")
RELEASE_KEYS = ("object_task_position_only_after_release", "no_drop_weight", "no_drop_margin_m")


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


def _resolved(*overrides: str) -> dict:
    return resolved_dict(compose_config("train", [*BASE, *overrides]))


def _differing(left: dict, right: dict) -> dict:
    a, b = _flatten(left), _flatten(right)
    keys = a.keys() | b.keys()
    return {key: b.get(key, "<absent>") for key in keys if a.get(key, "<absent>") != b.get(key, "<absent>")}


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (POSITION_ONLY, {"object_task_position_only_after_release": True}),
        (POSITION_ONLY + NO_DROP, {"object_task_position_only_after_release": True, "no_drop_weight": -50.0}),
        (NO_DROP, {"no_drop_weight": -50.0}),
    ],
)
def test_overrides_add_exactly_the_release_keys(overrides, expected):
    differing = _differing(_resolved(), _resolved(*overrides))
    wanted = {f"{root}.{key}": value for root in ("objective", "env.objective") for key, value in expected.items()}
    if "no_drop_weight" in expected:
        for root in ("curriculum", "training.curriculum"):
            wanted[f"{root}.stages.0.objective_weights.object_no_drop"] = -50.0
    assert differing == wanted


@pytest.mark.parametrize("no_drop", [False, True])
def test_release_configs_instantiate_and_pass_the_curriculum_name_check(no_drop):
    from flash_chord.envs.rl import RLEnvConfig
    from flash_chord.training.ppo.config import TrainingConfig

    config = compose_config("train", [*BASE, *POSITION_ONLY, *(NO_DROP if no_drop else ())])
    objective = instantiate_typed(config.env, RLEnvConfig).objective
    assert objective.object_task_position_only_after_release is True
    assert objective.object_task_enabled and objective.object_task_release_enabled
    assert (objective.object_task_weight, objective.object_task_position_beta) == (5.0, 10.0)
    assert objective.object_task_rotation_beta == 4.5
    assert objective.no_drop_margin_m == 0.05
    names = objective.term_names
    assert names[-1] == ("object_no_drop" if no_drop else "object_task")
    stages = instantiate_typed(config.training, TrainingConfig).curriculum.stages
    for stage in stages:
        weights = stage.weights_for(names)  # exact-name validation as in set_curriculum
        assert weights[names.index("object_task")] == 5.0
        if no_drop:
            assert weights[-1] == -50.0 == objective.no_drop_weight


def test_no_drop_needs_both_weights_and_a_non_positive_sign():
    from flash_chord.envs.rl import RLEnvConfig
    from flash_chord.training.ppo.config import TrainingConfig

    only_objective = compose_config("train", [*BASE, NO_DROP[0]])
    names = instantiate_typed(only_objective.env, RLEnvConfig).objective.term_names
    stage = instantiate_typed(only_objective.training, TrainingConfig).curriculum.stages[0]
    with pytest.raises(ValueError, match=r"missing=\['object_no_drop'\]"):
        stage.weights_for(names)
    only_curriculum = compose_config("train", [*BASE, NO_DROP[1]])
    names = instantiate_typed(only_curriculum.env, RLEnvConfig).objective.term_names
    stage = instantiate_typed(only_curriculum.training, TrainingConfig).curriculum.stages[0]
    with pytest.raises(ValueError, match=r"unexpected=\['object_no_drop'\]"):
        stage.weights_for(names)
    positive = compose_config("train", [*BASE, "+objective.no_drop_weight=50.0"])
    with pytest.raises(Exception, match="<= 0"):
        instantiate_typed(positive.env, RLEnvConfig)


@pytest.mark.parametrize(
    "overrides", [POSITION_ONLY, POSITION_ONLY + NO_DROP, NO_DROP + ("+objective.no_drop_margin_m=0.1",)]
)
def test_export_removes_every_release_key(overrides):
    import flash_chord.training.ppo.coordinate.export as e2

    saved = _resolved(*overrides)
    exported, notes = e2.stock_config_with_notes(saved)
    base_exported, base_notes = e2.stock_config_with_notes(_resolved())
    # The evaluator sees exactly the export of the same run without the options.
    assert _flatten(exported) == _flatten(base_exported)
    text = json.dumps(exported)
    assert "no_drop" not in text and "position_only" not in text
    extra = sorted(set(notes["removed_keys"]) - set(base_notes["removed_keys"]))
    given = [key for key in RELEASE_KEYS if any(f".{key}=" in override for override in overrides)]
    expected = [f"{root}.{key}" for root in ("objective", "env.objective") for key in given]
    if "no_drop_weight" in given:
        stage_key = "stages.0.objective_weights.object_no_drop"
        expected += [f"{root}.{stage_key}" for root in ("curriculum", "training.curriculum")]
    assert extra == sorted(expected)
    assert {key: value for key, value in notes.items() if key != "removed_keys"} == {
        key: value for key, value in base_notes.items() if key != "removed_keys"
    }
