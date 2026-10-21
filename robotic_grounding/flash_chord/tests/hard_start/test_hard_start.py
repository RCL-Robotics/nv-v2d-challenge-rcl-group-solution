# SPDX-License-Identifier: Apache-2.0
"""Hard-start sampling (dev ``start_mode="hazard"``, arms C3r2-runin-hard and C3r2-helpfix-hard): config contracts,
the hazard-weighted capped start distribution, the rolling start's start-frame shift, and pass-through counting
after the termination grace. CPU-device kernel tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from flash_chord.configuration import compose_config, resolved_dict
from flash_chord.lifecycle.rsi.bins import RSIBinParams, RSIBins, _cap

NAMES = ("logging.experiment", "logging.run_name", "training.experiment_name", "training.run_name")
START = ("frame0_share", "uniform_mix", "start_mode", "start_temperature", "start_lookahead_bins",
         "start_lookahead_decay", "start_cap", "start_frame_shift", "pass_after_grace")


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
    if isinstance(value, list) and any(isinstance(item, dict) for item in value):
        flat = {}
        for index, item in enumerate(value):
            flat.update(_flatten(item, f"{prefix}{index}."))
        return flat
    return {prefix.rstrip("."): value}


def _changed(base: str, arm: str) -> tuple[set[str], dict, dict]:
    a, b = _flatten(_compose(base)), _flatten(_compose(arm))
    return {k for k in a.keys() | b.keys() if a.get(k, KeyError) != b.get(k, KeyError)}, a, b


# ---------------------------------------------------------------- config contracts
@pytest.mark.parametrize(
    ("base", "arm", "extra", "shift"),
    [
        ("g1_recon_body_coord_rsi_r2_runin", "g1_recon_body_coord_rsi_r2_runin_hard",
         {"training.max_iterations", "training.early_stop_help_zero_tail"}, 20),
        ("g1_recon_body_coord_rsi_r2_helpfix", "g1_recon_body_coord_rsi_r2_helpfix_hard", set(), 0),
    ],
)
def test_hard_arms_differ_from_their_base_only_in_the_start_fields(base, arm, extra, shift):
    changed, a, b = _changed(base, arm)
    allowed = {f"{root}.{name}" for root in ("reset", "env.reset") for name in START} | set(NAMES) | extra
    unexpected = sorted(changed - allowed)
    assert not unexpected, f"{arm} drifted from {base}: {unexpected}"
    for root in ("reset", "env.reset"):
        assert b[f"{root}.start_mode"] == "hazard" and b[f"{root}.frame0_share"] == 0.1
        assert b[f"{root}.start_temperature"] == 0.1 and b[f"{root}.start_frame_shift"] == shift
        assert b[f"{root}.pass_after_grace"] is True
    assert b["training.max_iterations"] == 90000 and b["training.early_stop_help_zero_tail"] == 0
    # 0.1 frame 0, 0.1 uniform, 0.8 hard of all resets
    assert 0.9 * b["reset.uniform_mix"] == pytest.approx(0.1, abs=1e-5)
    assert 0.9 * b["reset.start_cap"] == pytest.approx(0.15, abs=1e-5)


def test_config_defaults_and_bounds():
    from flash_chord.lifecycle.rsi_reset import RSIReconBodyResetConfig

    config = RSIReconBodyResetConfig()
    assert config.start_mode == "credit" and config.start_frame_shift == 0 and config.pass_after_grace is False
    RSIReconBodyResetConfig(start_mode="hazard", start_frame_shift=50)
    for bad in ({"start_mode": "softmax"}, {"start_temperature": 0.0}, {"start_cap": 0.0},
                {"start_lookahead_bins": 0}, {"start_frame_shift": 51}, {"start_frame_shift": -1}):
        with pytest.raises((ValueError, TypeError)):
            RSIReconBodyResetConfig(**bad)
    with pytest.raises(TypeError):
        RSIReconBodyResetConfig(start_frame_shift=2.5)
    with pytest.raises(TypeError):
        RSIReconBodyResetConfig(pass_after_grace=1)


def test_export_replaces_the_rsi_reset_with_the_stock_fields():
    from flash_chord.training.ppo.coordinate import export

    for base, arm in (("g1_recon_body_coord_rsi_r2_runin", "g1_recon_body_coord_rsi_r2_runin_hard"),
                      ("g1_recon_body_coord_rsi_r2_helpfix", "g1_recon_body_coord_rsi_r2_helpfix_hard")):
        stock, _ = export.stock_config_with_notes(_compose(arm))
        base_stock, _ = export.stock_config_with_notes(_compose(base))
        dev_fields = set(START) - {"frame0_share", "uniform_mix"}  # the stock reset has its own start_frame
        assert not dev_fields & set(stock["env"]["reset"])
        assert stock["env"]["reset"] == base_stock["env"]["reset"]


# ---------------------------------------------------------------- host logic
def _hazard_bins(**overrides) -> RSIBins:
    params = {"start_mode": "hazard", "uniform_mix": 1 / 9, "start_temperature": 0.1, "start_cap": 1 / 6,
              "help_min_samples": 256.0, **overrides}
    return RSIBins(803, RSIBinParams(**params))


def _feed(bins: RSIBins, hazard, iterations: int = 600, per_bin: int = 200):
    hazard = np.broadcast_to(np.asarray(hazard, dtype=np.float64), (bins.num_bins,))
    for _ in range(iterations):
        fails = np.round(per_bin * hazard)
        bins.observe_pass(per_bin - fails, fails)


def test_without_samples_the_hazard_mode_is_uniform():
    bins = _hazard_bins()
    np.testing.assert_allclose(bins.start_probs(), 1.0 / bins.num_bins)


def test_hard_bins_and_the_bins_before_them_get_the_hard_share_with_a_uniform_floor_and_a_cap():
    bins = _hazard_bins()
    hazard = np.full(bins.num_bins, 0.2)
    hazard[[6, 26]] = 0.97
    _feed(bins, hazard)
    probs = bins.start_probs()
    assert probs.sum() == pytest.approx(1.0)
    assert (probs >= bins.params.uniform_mix / bins.num_bins - 1e-12).all()  # uniform floor
    assert probs.max() <= bins.params.start_cap + 1e-12  # cap
    order = np.argsort(probs)[::-1]
    assert set(order[:2]) == {6, 26}
    assert probs[5] > probs[7] and probs[25] > probs[27]  # lookahead: in front of a failure point, not behind it
    hard_share = probs[[4, 5, 6, 24, 25, 26]].sum()
    assert hard_share > 0.5


def test_uncapped_softmax_matches_the_formula():
    bins = _hazard_bins(start_cap=1.0)
    rng = np.random.default_rng(3)
    _feed(bins, rng.uniform(0, 1, size=bins.num_bins))
    d = bins.start_difficulty()
    z = np.exp((d - d.max()) / 0.1)
    np.testing.assert_allclose(bins.start_probs(), (1 / 9) / bins.num_bins + (8 / 9) * z / z.sum())


def test_cap_redistributes_in_proportion():
    probs = _cap(np.array([0.7, 0.2, 0.05, 0.05]), 0.4)
    assert probs.sum() == pytest.approx(1.0) and probs.max() == pytest.approx(0.4)
    np.testing.assert_allclose(probs[1:] / probs[1:].sum(), np.array([0.2, 0.05, 0.05]) / 0.3)
    np.testing.assert_allclose(_cap(np.array([0.5, 0.5]), 0.1), [0.5, 0.5])  # infeasible cap rises to 1/n


def test_credit_mode_ignores_the_pass_through_counts():
    a = RSIBins(803, RSIBinParams())
    b = RSIBins(803, RSIBinParams())
    rng = np.random.default_rng(4)
    for _ in range(50):
        n = rng.integers(1, 100, size=a.num_bins)
        f = rng.binomial(n, 0.7)
        a.observe(n, f)
        b.observe(n, f)
        b.observe_pass(rng.integers(0, 50, size=b.num_bins), rng.integers(0, 50, size=b.num_bins))
    np.testing.assert_array_equal(a.start_probs(), b.start_probs())


# ---------------------------------------------------------------- CPU kernels
Q_TEST = 12


def _bound_policy(world_count: int, **overrides):
    import warp as wp

    from flash_chord.lifecycle.rsi_reset import RSIReconBodyResetConfig

    config = {"keyframes": False, "frame0_share": 0.0, **overrides}
    policy = RSIReconBodyResetConfig(**config).build(world_count, 440, device="cpu")
    reference = np.random.default_rng(0).uniform(0.1, 0.9, size=(440, 10)).astype(np.float32)
    policy.bind_arrays(
        q_per_world=Q_TEST,
        left_shoulder_q_id=3,
        right_shoulder_q_id=4,
        reference_joint_q=reference,
        finger_q_ids=(5, 6),
        output=wp.zeros(world_count * Q_TEST, dtype=wp.float32, device="cpu"),
    )
    env = SimpleNamespace(
        world_count=world_count,
        termination=SimpleNamespace(
            object_position_error=wp.zeros(world_count, dtype=wp.float32, device="cpu"),
            object_orientation_error=wp.zeros(world_count, dtype=wp.float32, device="cpu"),
        ),
        model=SimpleNamespace(joint_coord_count=world_count * Q_TEST, joint_dof_count=world_count * Q_TEST),
        terminated=wp.zeros(world_count, dtype=wp.int32, device="cpu"),
        truncated=wp.zeros(world_count, dtype=wp.int32, device="cpu"),
        timestep=wp.zeros(world_count, dtype=wp.int32, device="cpu"),
        episode_step=wp.zeros(world_count, dtype=wp.int32, device="cpu"),
        reset_table=SimpleNamespace(
            object_root_q_ids=wp.zeros(0, dtype=wp.int32, device="cpu"),
            object_root_position_offsets_w=wp.zeros(0, dtype=wp.vec3, device="cpu"),
            num_objects=0,
        ),
    )
    policy.bind_env(env)
    state = SimpleNamespace(
        joint_q=wp.zeros(world_count * Q_TEST, dtype=wp.float32, device="cpu"),
        joint_qd=wp.zeros(world_count * Q_TEST, dtype=wp.float32, device="cpu"),
    )
    return policy, env, state


@pytest.mark.parametrize("pass_after_grace", [False, True])
def test_no_outcome_is_counted_while_the_terminations_are_masked(pass_after_grace):
    # live lanes: episode_step already counts this transition (termination saw step - 1); done lanes were not advanced
    policy, env, state = _bound_policy(5, pass_after_grace=pass_after_grace)
    policy.active.assign(np.asarray([1, 1, 1, 1, 1], dtype=np.int32))
    policy.visit_bin.assign(np.asarray([2, 2, 2, 2, 2], dtype=np.int32))
    env.timestep.assign(np.asarray([3 * 25, 3 * 25, 3 * 25, 3 * 25, 2 * 25 + 5], dtype=np.int32))
    env.episode_step.assign(np.asarray([51, 52, 50, 51, 30], dtype=np.int32))
    env.terminated.assign(np.asarray([0, 0, 1, 1, 1], dtype=np.int32))
    policy.on_transition(state)
    passes, fails = policy.pass_iter.numpy(), policy.pass_fail_iter.numpy()
    # world 0: live, checked step 50 (masked); world 1: live, checked 51; world 2: done, checked 50 (masked);
    # world 3: done, checked 51; world 4: done inside the hold without crossing a bin
    if pass_after_grace:
        assert passes[2] == 2  # worlds 1 and 3 crossed bin 2 with live terminations
        assert fails[3] == 1 and fails[2] == 0  # only world 3 failed with live terminations
    else:
        assert passes[2] == 4  # every crossing counts
        assert fails[3] == 2 and fails[2] == 1  # worlds 2 and 3 failed in bin 3, world 4 in bin 2
    np.testing.assert_array_equal(policy.visit_bin.numpy(), [3, 3, 3, 3, 2])  # the lane's bin is always followed


def test_the_rolling_start_shift_moves_reference_starts_back_and_keeps_their_bin():
    import warp as wp

    policy, _, _ = _bound_policy(4000, start_frame_shift=20, run_in_steps=20, min_horizon=30)
    policy.sample(wp.ones(4000, dtype=wp.int32, device="cpu"), wp.zeros(1, dtype=wp.float32, device="cpu"))
    frame, b = policy.reset_frame.numpy(), policy.start_bin.numpy()
    lo, hi = policy.bins.bin_lo[b], policy.bins.bin_hi[b]
    immediate = frame == 0  # a sampled frame 0 stays the evaluator's start (not moved, no hold)
    assert (b[immediate] == 0).all()
    ref = ~immediate
    # the clock reaches frame + 20 as the hold ends: inside the sampled bin (bin-0 starts clamp at frame 1)
    assert ((frame + 20 >= lo) & (frame + 20 < hi))[ref].all()
    assert (frame[ref] >= 1).all()
    assert (policy.steps_since_reset.numpy()[ref] == 0).all()  # every moved reference start keeps its hold
    assert (frame[ref & (b > 0)] == np.clip(frame[ref & (b > 0)], lo[ref & (b > 0)] - 20, hi[ref & (b > 0)] - 21)).all()
    plain, _, _ = _bound_policy(4000, run_in_steps=20, min_horizon=30)
    plain.sample(wp.ones(4000, dtype=wp.int32, device="cpu"), wp.zeros(1, dtype=wp.float32, device="cpu"))
    p_frame, p_b = plain.reset_frame.numpy(), plain.start_bin.numpy()
    assert ((p_frame >= plain.bins.bin_lo[p_b]) & (p_frame < plain.bins.bin_hi[p_b])).all()  # no shift by default
