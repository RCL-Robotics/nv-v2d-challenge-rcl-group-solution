# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The C1/C2/C3 experiment yamls differ from stock g1_recon_body_ppo only where the contract says (spec 7.1).

C3-wide (g1_recon_body_coord_rsi_wide) is C3 with a wider coordinated actor; it differs from C3 only in
the actor widths it names below and in its logging names.

C3r2 (g1_recon_body_coord_rsi_r2) is C3-wide with Coordinates_hand's RSI timers converted by samples, a 45k
iteration cap, the CH early stop, the CH object task reward term (weight 5.0 in the objective and in the curriculum
stage, CH's trained betas 10/1) and its logging names; nothing else.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest

from flash_chord.configuration import compose_config, instantiate_typed, resolved_dict

CONFIGS = Path(__file__).resolve().parents[2] / "src" / "flash_chord" / "configs"
STOCK = "g1_recon_body_ppo"
RSI_FILES_PRESENT = (CONFIGS / "reset" / "recon_body_rsi.yaml").is_file() and (
    CONFIGS / "curriculum" / "recon_body_rsi.yaml"
).is_file()
RSI_RESET_TARGET = "flash_chord.lifecycle.rsi_reset.RSIReconBodyResetConfig"
COORDINATE_TARGET = "flash_chord.training.ppo.config.CoordinateConfig"
EXPECTED_COORDINATE = {
    "_target_": COORDINATE_TARGET,
    "agents": ["left", "right", "body"],
    "encoder_dim": 96,
    "partner_dim": 64,
    "message_dim": 64,
    "trunk_dims": [80, 40],
    "partner_message": True,
    "mirror_body_message": True,
    "hand_world_terms": "reflect",
    "ratio_mode": "per_agent",
    "value_mode": "per_agent",
    "export_actor_hidden_dims": [1024, 512, 256, 128],
}
NAMES = ("logging.experiment", "logging.run_name", "training.experiment_name", "training.run_name")
RSI_PREFIXES = ("reset.", "env.reset.", "curriculum.", "training.curriculum.")
COORDINATE_PREFIXES = ("training.coordinate.",)
C3 = "g1_recon_body_coord_rsi"
WIDE = "g1_recon_body_coord_rsi_wide"
R2 = "g1_recon_body_coord_rsi_r2"
ARMS = {
    "g1_recon_body_rsi": (RSI_PREFIXES, True, False),
    "g1_recon_body_coord": (COORDINATE_PREFIXES, False, True),
    "g1_recon_body_coord_rsi": (RSI_PREFIXES + COORDINATE_PREFIXES, True, True),
    WIDE: (RSI_PREFIXES + COORDINATE_PREFIXES, True, True),
}
# C3-wide: a Coordinates_hand-size coordinated actor (export budget [640, 576, 768, 384]).
WIDE_EXPORT_ACTOR_HIDDEN_DIMS = [1024, 1024, 768, 384]
EXPECTED_WIDE_COORDINATE = {
    **EXPECTED_COORDINATE,
    "encoder_dim": 128,
    "partner_dim": 64,
    "message_dim": 64,
    "trunk_dims": [256, 128],
    "export_actor_hidden_dims": WIDE_EXPORT_ACTOR_HIDDEN_DIMS,
}
EXPECTED_COORDINATE_BY_ARM = {
    "g1_recon_body_coord": EXPECTED_COORDINATE,
    "g1_recon_body_coord_rsi": EXPECTED_COORDINATE,
    WIDE: EXPECTED_WIDE_COORDINATE,
}
# Every flattened key in which C3-wide may differ from C3, with its (C3, C3-wide) values.
WIDE_VS_C3 = {
    "training.coordinate.encoder_dim": (96, 128),
    "training.coordinate.trunk_dims": ([80, 40], [256, 128]),
    "training.coordinate.export_actor_hidden_dims": ([1024, 512, 256, 128], WIDE_EXPORT_ACTOR_HIDDEN_DIMS),
    "logging.experiment": ("g1_sonic_recon_body_coord_rsi", "g1_sonic_recon_body_coord_rsi_wide"),
    "logging.run_name": ("g1_recon_body_coord_rsi", "g1_recon_body_coord_rsi_wide"),
    "training.experiment_name": ("g1_sonic_recon_body_coord_rsi", "g1_sonic_recon_body_coord_rsi_wide"),
    "training.run_name": ("g1_recon_body_coord_rsi", "g1_recon_body_coord_rsi_wide"),
}


# C3r2 vs C3-wide: every flattened key that may differ, with its (C3-wide, C3r2) values. KeyError = absent.
_R2_TIMERS = {"wait_iterations": (100, 391), "min_gap": (40, 156), "deadline": (11000, 35156), "zero_ramp": (1000, 3906)}
# CH object task term (objectives/recon_body.py): weight 5.0 from the paired group files objective/recon_body_objtask +
# curriculum/recon_body_rsi_objtask; betas as CH trained them (dexmachina/envs/constructors.py:111-115 overwrite
# get_reward_cfg()'s 20/5 with --task_rew_betas, default and launcher value 10 1 5), set in the r2 yaml.
R2_OBJECT_TASK_WEIGHT = 5.0
R2_OBJECT_TASK_OBJECTIVE = {
    "object_task_weight": R2_OBJECT_TASK_WEIGHT,
    "object_task_position_beta": 10.0,
    "object_task_rotation_beta": 1.0,
}
R2_OBJECT_TASK_STAGE_KEY = "stages.0.objective_weights.object_task"
R2_OBJECT_TASK_KEYS = {
    **{
        f"{root}.{name}": value
        for root in ("objective", "env.objective")
        for name, value in R2_OBJECT_TASK_OBJECTIVE.items()
    },
    **{f"{root}.{R2_OBJECT_TASK_STAGE_KEY}": R2_OBJECT_TASK_WEIGHT for root in ("curriculum", "training.curriculum")},
}
R2_VS_WIDE = {
    **{f"{section}.{name}": values for section in ("reset", "env.reset") for name, values in _R2_TIMERS.items()},
    "training.max_iterations": (20000, 45000),
    "training.early_stop_help_zero_tail": (KeyError, 5860),
    **{key: (KeyError, value) for key, value in R2_OBJECT_TASK_KEYS.items()},
    "logging.experiment": ("g1_sonic_recon_body_coord_rsi_wide", "g1_sonic_recon_body_coord_rsi_r2"),
    "logging.run_name": ("g1_recon_body_coord_rsi_wide", "g1_recon_body_coord_rsi_r2"),
    "training.experiment_name": ("g1_sonic_recon_body_coord_rsi_wide", "g1_sonic_recon_body_coord_rsi_r2"),
    "training.run_name": ("g1_recon_body_coord_rsi_wide", "g1_recon_body_coord_rsi_r2"),
}
# Coordinates_hand update = 12000 envs x 32 steps (ASSUMED: CH's A1 / W0-P recipe, dexmachina/ablation/config.py:89-97;
# the RSI runs' own flags are not on disk); ours = 4096 worlds x 24 steps.
CH_SAMPLES_PER_UPDATE = 12000 * 32
CH_TIMERS = {"wait_iterations": 100, "min_gap": 40, "deadline": 9000, "zero_ramp": 1000}
CH_ZERO_GAIN_TAIL = 1500


def _compose(experiment: str) -> dict:
    return resolved_dict(compose_config("train", [f"experiment={experiment}", "task.parquet=/tmp/reference.parquet"]))


def _flatten(value, prefix: str = "") -> dict[str, object]:
    """Leaf paths; lists of mappings (curriculum stages) are indexed, other lists are leaves."""
    if isinstance(value, dict):
        if not value:
            return {prefix.rstrip("."): {}}
        flat = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}{key}."))
        return flat
    if isinstance(value, list) and any(isinstance(item, dict) for item in value):
        flat = {}
        for index, item in enumerate(value):
            flat.update(_flatten(item, f"{prefix}{index}."))
        return flat
    return {prefix.rstrip("."): value}


def _without_object_task(config: dict) -> dict:
    """``config`` (objective or curriculum subtree) without the object task term's keys."""
    config = json.loads(json.dumps(config))
    for name in R2_OBJECT_TASK_OBJECTIVE:
        config.pop(name, None)
    for stage in config.get("stages", ()):
        stage["objective_weights"].pop("object_task", None)
    return config


def _diff(left: dict, right: dict) -> set[str]:
    a, b = _flatten(left), _flatten(right)
    return {key for key in a.keys() | b.keys() if a.get(key, KeyError) != b.get(key, KeyError)}


def _needs_rsi(experiment: str) -> None:
    if ARMS[experiment][1] and not RSI_FILES_PRESENT:
        pytest.skip("owner B's reset/curriculum recon_body_rsi.yaml are not present")


@pytest.mark.parametrize("experiment", sorted(ARMS))
def test_experiment_differs_from_stock_only_where_allowed(experiment):
    _needs_rsi(experiment)
    prefixes, rsi, coordinate = ARMS[experiment]
    stock = _compose(STOCK)
    arm = _compose(experiment)
    changed = _diff(stock, arm)
    unexpected = sorted(key for key in changed if key not in NAMES and not key.startswith(prefixes))
    assert not unexpected, f"{experiment} drifted from {STOCK}: {unexpected}"
    assert {"logging.experiment", "logging.run_name"} <= changed
    assert arm["training"]["experiment_name"] == arm["logging"]["experiment"]
    assert arm["training"]["run_name"] == arm["logging"]["run_name"]
    if rsi:
        assert arm["reset"]["_target_"] == arm["env"]["reset"]["_target_"] == RSI_RESET_TARGET
        assert arm["env"]["reset"] == arm["reset"]
        assert arm["curriculum"] == arm["training"]["curriculum"]
        assert arm["curriculum"]["thresholds"] == [20000] and len(arm["curriculum"]["stages"]) == 1
        stage = arm["curriculum"]["stages"][0]
        assert stage["voc_scale"] == 0.0
        assert stage.get("reset_to_first_frame_probability") is None
        assert stage.get("immediate_first_frame_probability") is None
        assert stage["objective_weights"] == stock["curriculum"]["stages"][0]["objective_weights"]
        # The ten stock reset fields keep the stock values.
        for name, value in stock["reset"].items():
            if name != "_target_":
                assert arm["reset"][name] == value, name
    else:
        assert arm["reset"] == stock["reset"] and arm["curriculum"] == stock["curriculum"]
    if coordinate:
        assert arm["training"]["coordinate"] == EXPECTED_COORDINATE_BY_ARM[experiment]
    else:
        assert "coordinate" not in arm["training"]
    # Budget and recipe (D3 F0, D4 stock reward) stay stock.
    for key in ("world_count",):
        assert arm["scene"][key] == stock["scene"][key] == 4096
    assert arm["training"]["max_iterations"] == 20000 and arm["training"]["rollout_steps"] == 24
    assert arm["action"] == stock["action"] and arm["objective"] == stock["objective"]


def test_c3_is_the_union_of_c1_and_c2():
    _needs_rsi("g1_recon_body_coord_rsi")
    stock = _compose(STOCK)
    c1, c2, c3 = (_compose(name) for name in ("g1_recon_body_rsi", "g1_recon_body_coord", "g1_recon_body_coord_rsi"))
    assert _diff(stock, c3) - set(NAMES) == (_diff(stock, c1) | _diff(stock, c2)) - set(NAMES)
    flat1, flat2, flat3 = _flatten(c1), _flatten(c2), _flatten(c3)
    for key in _diff(stock, c1) - set(NAMES):
        assert flat3.get(key) == flat1.get(key), key
    for key in _diff(stock, c2) - set(NAMES):
        assert flat3.get(key) == flat2.get(key), key


def test_c3_wide_differs_from_c3_only_in_actor_widths_and_names():
    """Structural diff of the composed configs: any added, removed or changed key outside WIDE_VS_C3 fails."""
    _needs_rsi(WIDE)
    c3, wide = _compose(C3), _compose(WIDE)
    changed = _diff(c3, wide)
    assert changed == set(WIDE_VS_C3), (
        f"unexpected: {sorted(changed - set(WIDE_VS_C3))}, missing: {sorted(set(WIDE_VS_C3) - changed)}"
    )
    flat3, flat_wide = _flatten(c3), _flatten(wide)
    assert flat3.keys() == flat_wide.keys()
    for key, (c3_value, wide_value) in WIDE_VS_C3.items():
        assert (flat3[key], flat_wide[key]) == (c3_value, wide_value), key
    # The parts the capacity change must not touch, spelled out.
    for section in ("reset", "curriculum", "env", "scene", "task", "action", "objective"):
        assert wide[section] == c3[section], section
    assert wide["training"]["network"] == c3["training"]["network"]
    assert wide["training"]["ppo"] == c3["training"]["ppo"]
    assert wide["training"]["coordinate"]["partner_dim"] == c3["training"]["coordinate"]["partner_dim"] == 64
    assert wide["training"]["coordinate"]["message_dim"] == c3["training"]["coordinate"]["message_dim"] == 64


def test_c3_wide_coordinate_config_fits_the_export_budget():
    """The wide actor passes CoordinateConfig's budget check and embeds into [1024, 1024, 768, 384]."""
    _needs_rsi(WIDE)
    from flash_chord.training.ppo.config import CoordinateConfig, TrainingConfig, coordinate_export_budget

    composed = compose_config("train", [f"experiment={WIDE}", "task.parquet=/tmp/reference.parquet"])
    coordinate = instantiate_typed(composed.training, TrainingConfig).coordinate
    assert isinstance(coordinate, CoordinateConfig)  # __post_init__ ran the export budget check
    assert coordinate.export_actor_hidden_dims == (1024, 1024, 768, 384)
    assert list(coordinate.export_actor_hidden_dims) == WIDE_EXPORT_ACTOR_HIDDEN_DIMS
    assert (coordinate.encoder_dim, coordinate.partner_dim, coordinate.message_dim) == (128, 64, 64)
    assert coordinate.trunk_dims == (256, 128) and coordinate.mirror_body_message
    budget = coordinate_export_budget(
        coordinate.encoder_dim,
        coordinate.partner_dim,
        coordinate.message_dim,
        coordinate.trunk_dims,
        coordinate.mirror_body_message,
    )
    assert budget == (640, 576, 768, 384)
    assert all(need <= have for need, have in zip(budget, coordinate.export_actor_hidden_dims, strict=True))
    # The exporter's own budget check agrees.
    export = pytest.importorskip("flash_chord.training.ppo.coordinate.export")
    assert export.embedding_plan(coordinate, coordinate.export_actor_hidden_dims).used_units == budget
    # The same actor widths with C3's stock-shaped export would not fit (why the export widens).
    with pytest.raises(ValueError, match="export_actor_hidden_dims"):
        CoordinateConfig(encoder_dim=128, trunk_dims=(256, 128), export_actor_hidden_dims=(1024, 512, 256, 128))


def test_yaml_coordinate_block_lists_every_coordinate_config_field():
    config = pytest.importorskip("flash_chord.training.ppo.config")
    if not hasattr(config, "CoordinateConfig"):
        pytest.skip("owner A's CoordinateConfig has not landed")
    fields = {field.name: field.default for field in dataclasses.fields(config.CoordinateConfig)}
    assert set(EXPECTED_COORDINATE) - {"_target_"} == set(fields)
    for name, default in fields.items():
        value = EXPECTED_COORDINATE[name]
        assert (tuple(value) if isinstance(value, list) else value) == default, name


@pytest.mark.parametrize("experiment", ["g1_recon_body_coord", "g1_recon_body_coord_rsi", "g1_recon_body_rsi", WIDE])
def test_training_config_instantiates(experiment):
    """The DEV runtime instantiates the arm's training config (needs A's TrainingConfig.coordinate)."""
    _needs_rsi(experiment)
    from flash_chord.training.ppo.config import TrainingConfig

    if "coordinate" not in {field.name for field in dataclasses.fields(TrainingConfig)}:
        pytest.skip("owner A's TrainingConfig.coordinate has not landed")
    composed = compose_config("train", [f"experiment={experiment}", "task.parquet=/tmp/reference.parquet"])
    training = instantiate_typed(composed.training, TrainingConfig)
    if ARMS[experiment][2]:
        assert type(training.coordinate).__name__ == "CoordinateConfig"
        if experiment == WIDE:
            assert training.coordinate.trunk_dims == (256, 128)
        else:
            assert training.coordinate.trunk_dims == (80, 40)
    else:
        assert training.coordinate is None


@pytest.mark.parametrize("experiment", ["g1_recon_body_rsi", "g1_recon_body_coord_rsi", WIDE])
def test_rsi_reset_config_instantiates(experiment):
    """The arm's reset subtree instantiates owner B's RSI config (skipped until B lands)."""
    _needs_rsi(experiment)
    try:
        from flash_chord.lifecycle.rsi_reset import RSIReconBodyResetConfig
    except ImportError:
        pytest.skip("owner B's RSIReconBodyResetConfig has not landed")
    composed = compose_config("train", [f"experiment={experiment}", "task.parquet=/tmp/reference.parquet"])
    reset = instantiate_typed(composed.env.reset, RSIReconBodyResetConfig)
    assert reset.frame0_share == 0.25 and reset.bin_frames == 25 and reset.deadline == 11000


# ------------------------------------------------------------------------------------------- C3r2 (round 2)


def test_c3r2_differs_from_c3_wide_only_in_rsi_timers_budget_early_stop_object_task_and_names():
    """Structural diff of the composed configs: any added, removed or changed key outside R2_VS_WIDE fails."""
    if not RSI_FILES_PRESENT:
        pytest.skip("reset/curriculum recon_body_rsi.yaml are not present")
    wide, r2 = _compose(WIDE), _compose(R2)
    changed = _diff(wide, r2)
    assert changed == set(R2_VS_WIDE), (
        f"unexpected: {sorted(changed - set(R2_VS_WIDE))}, missing: {sorted(set(R2_VS_WIDE) - changed)}"
    )
    flat_wide, flat_r2 = _flatten(wide), _flatten(r2)
    # Added keys: the early stop and the 8 object task keys (6 objective fields, 2 curriculum stage weights).
    assert len(R2_OBJECT_TASK_KEYS) == 8
    assert flat_r2.keys() - flat_wide.keys() == {"training.early_stop_help_zero_tail", *R2_OBJECT_TASK_KEYS}
    assert flat_wide.keys() <= flat_r2.keys()
    for key, (wide_value, r2_value) in R2_VS_WIDE.items():
        assert (flat_wide.get(key, KeyError), flat_r2[key]) == (wide_value, r2_value), key
    # Untouched, spelled out: actor/critic/PPO, env, scene; the curriculum (one stage, voc 0, its 20000 sentinel)
    # and the objective differ only by the object task keys.
    for section in ("env.voc", "scene", "task", "action", "observation", "termination"):
        left, right = wide, r2
        for part in section.split("."):
            left, right = left[part], right[part]
        assert left == right, section
    for part in ("network", "ppo", "coordinate", "rollout_steps", "save_interval"):
        assert wide["training"][part] == r2["training"][part], part
    for left, right, name in (
        (wide["objective"], r2["objective"], "objective"),
        (wide["env"]["objective"], r2["env"]["objective"], "env.objective"),
        (wide["curriculum"], r2["curriculum"], "curriculum"),
        (wide["training"]["curriculum"], r2["training"]["curriculum"], "training.curriculum"),
    ):
        assert "object_task_weight" not in json.dumps(left) and '"object_task"' not in json.dumps(left), name
        assert _without_object_task(right) == left, name
    assert r2["env"]["objective"] == r2["objective"] and r2["training"]["curriculum"] == r2["curriculum"]
    assert r2["curriculum"]["thresholds"] == [20000] and len(r2["curriculum"]["stages"]) == 1
    timers = set(_R2_TIMERS)
    assert {k: v for k, v in r2["reset"].items() if k not in timers} == {
        k: v for k, v in wide["reset"].items() if k not in timers}
    assert r2["env"]["reset"] == r2["reset"]


def test_c3r2_timers_are_coordinates_hand_timers_converted_by_samples():
    if not RSI_FILES_PRESENT:
        pytest.skip("reset/curriculum recon_body_rsi.yaml are not present")
    r2 = _compose(R2)
    ours = r2["scene"]["world_count"] * r2["training"]["rollout_steps"]
    assert ours == 98304
    factor = CH_SAMPLES_PER_UPDATE / ours
    assert factor == 3.90625
    for name, ch_value in CH_TIMERS.items():
        assert r2["reset"][name] == round(ch_value * factor), name
    assert r2["training"]["early_stop_help_zero_tail"] == math.ceil(CH_ZERO_GAIN_TAIL * factor) == 5860
    # The deadline forces zero help by deadline + zero_ramp, so the early stop always fires before the cap.
    zero_by = r2["reset"]["deadline"] + r2["reset"]["zero_ramp"]
    assert zero_by + r2["training"]["early_stop_help_zero_tail"] == 44922 <= r2["training"]["max_iterations"] == 45000


def test_c3r2_instantiates_with_the_converted_timers_and_one_curriculum_stage():
    if not RSI_FILES_PRESENT:
        pytest.skip("reset/curriculum recon_body_rsi.yaml are not present")
    from flash_chord.lifecycle.rsi.bins import RSIBinParams, RSIBins
    from flash_chord.training.ppo.config import TrainingConfig

    composed = compose_config("train", [f"experiment={R2}", "task.parquet=/tmp/reference.parquet"])
    training = instantiate_typed(composed.training, TrainingConfig)
    assert (training.max_iterations, training.early_stop_help_zero_tail) == (45000, 5860)
    assert training.coordinate.trunk_dims == (256, 128) and training.nonfinite_guard
    # One stage at every iteration up to the cap: the 20000 threshold is a clamped sentinel.
    assert {training.curriculum.stage_index(i) for i in (0, 19999, 20000, 39062, 44999)} == {0}
    assert training.curriculum.stages[0].voc_scale == 0.0
    try:
        from flash_chord.lifecycle.rsi_reset import RSIReconBodyResetConfig
    except ImportError:
        pytest.skip("RSIReconBodyResetConfig is not importable")
    reset = instantiate_typed(composed.env.reset, RSIReconBodyResetConfig)
    assert (reset.wait_iterations, reset.min_gap, reset.deadline, reset.zero_ramp) == (391, 156, 35156, 3906)
    assert (reset.bin_frames, reset.frame0_share, reset.step, reset.band, reset.min_count) == (25, 0.25, 0.9, 2.0, 64)
    bins = RSIBins(500, RSIBinParams.from_object(reset))
    assert bins.floor(35156) == pytest.approx(6.25e-4) and bins.floor(39062) == 0.0 and bins.floor(39061) > 0.0


def test_other_arms_keep_the_early_stop_off():
    from flash_chord.training.ppo.config import TrainingConfig

    for experiment in sorted(ARMS):
        _needs_rsi(experiment)
        composed = compose_config("train", [f"experiment={experiment}", "task.parquet=/tmp/reference.parquet"])
        training = instantiate_typed(composed.training, TrainingConfig)
        assert training.early_stop_help_zero_tail == 0 and training.max_iterations == 20000, experiment
        assert "early_stop_help_zero_tail" not in _compose(experiment)["training"], experiment


def test_c3r2_object_task_term_has_weight_5_in_both_places_and_ch_trained_betas():
    """The term exists (objective weight), training applies the same weight (curriculum stage), betas 10/1."""
    if not RSI_FILES_PRESENT:
        pytest.skip("reset/curriculum recon_body_rsi.yaml are not present")
    from flash_chord.envs.rl import RLEnvConfig
    from flash_chord.objectives.recon_body import RECON_BODY_OBJECTIVE_TERM_NAMES
    from flash_chord.training.ppo.config import TrainingConfig

    composed = compose_config("train", [f"experiment={R2}", "task.parquet=/tmp/reference.parquet"])
    resolved = resolved_dict(composed)
    for root in (resolved["objective"], resolved["env"]["objective"]):
        assert {name: root[name] for name in R2_OBJECT_TASK_OBJECTIVE} == R2_OBJECT_TASK_OBJECTIVE
    for curriculum in (resolved["curriculum"], resolved["training"]["curriculum"]):
        assert [stage["objective_weights"]["object_task"] for stage in curriculum["stages"]] == [R2_OBJECT_TASK_WEIGHT]
    objective = instantiate_typed(composed.env, RLEnvConfig).objective
    assert objective.object_task_enabled and objective.object_task_weight == R2_OBJECT_TASK_WEIGHT
    assert (objective.object_task_position_beta, objective.object_task_rotation_beta) == (10.0, 1.0)
    assert objective.term_names == (*RECON_BODY_OBJECTIVE_TERM_NAMES, "object_task")
    training = instantiate_typed(composed.training, TrainingConfig)
    for stage in training.curriculum.stages:
        weights = stage.weights_for(objective.term_names)  # the exact-name check set_curriculum runs
        assert weights[-1] == objective.term_weights[-1] == R2_OBJECT_TASK_WEIGHT
        assert tuple(weights[:-1]) == tuple(objective.term_weights[:-1])


def test_c3r2_export_drops_the_early_stop_and_object_task_keys():
    """E2 export keeps only stock config keys: the evaluator never sees early_stop_help_zero_tail or object_task."""
    if not RSI_FILES_PRESENT:
        pytest.skip("reset/curriculum recon_body_rsi.yaml are not present")
    export = pytest.importorskip("flash_chord.training.ppo.coordinate.export")
    stock, notes = export.stock_config_with_notes(_compose(R2))
    assert "early_stop_help_zero_tail" not in stock["training"] and "coordinate" not in stock["training"]
    assert stock["training"]["max_iterations"] == 45000
    assert stock["env"]["reset"]["_target_"] != RSI_RESET_TARGET
    assert "training.early_stop_help_zero_tail" in json.dumps(notes)
    assert sorted(key for key in notes["removed_keys"] if "object_task" in key) == sorted(R2_OBJECT_TASK_KEYS)
    assert "object_task" not in json.dumps(stock)
    # The exported objective and curriculum are C3-wide's exported ones.
    wide_stock, _ = export.stock_config_with_notes(_compose(WIDE))
    assert stock["objective"] == wide_stock["objective"] and stock["env"]["objective"] == wide_stock["env"]["objective"]
    assert stock["curriculum"] == wide_stock["curriculum"]
    assert stock["training"]["curriculum"] == wide_stock["training"]["curriculum"]
