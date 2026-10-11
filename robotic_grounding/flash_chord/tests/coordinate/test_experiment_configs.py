# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The C1/C2/C3 experiment yamls differ from stock g1_recon_body_ppo only where the contract says (spec 7.1).

C3-wide (g1_recon_body_coord_rsi_wide) is C3 with a wider coordinated actor; it differs from C3 only in
the actor widths it names below and in its logging names.
"""

from __future__ import annotations

import dataclasses
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


def _compose(experiment: str) -> dict:
    return resolved_dict(compose_config("train", [f"experiment={experiment}", "task.parquet=/tmp/reference.parquet"]))


def _flatten(value, prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict):
        if not value:
            return {prefix.rstrip("."): {}}
        flat = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}{key}."))
        return flat
    return {prefix.rstrip("."): value}


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
