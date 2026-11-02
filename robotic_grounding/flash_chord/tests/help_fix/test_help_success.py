# SPDX-License-Identifier: Apache-2.0
"""Success-driven RSI help (dev ``help_mode="success"``, arm C3r2-helpfix): config contract, the monotone
rate-limited ratchet on EMA pass-through pass rates, the final cutoff, sidecar compatibility, and the CPU kernels
(pass-through attribution, help at the current frame)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from flash_chord.configuration import compose_config, resolved_dict
from flash_chord.lifecycle.rsi.bins import RSIBinParams, RSIBins

R2 = "g1_recon_body_coord_rsi_r2"
HELPFIX = "g1_recon_body_coord_rsi_r2_helpfix"
NAMES = ("logging.experiment", "logging.run_name", "training.experiment_name", "training.run_name")
TRAINING = ("training.max_iterations", "training.early_stop_help_zero_tail")
HELP_FIELDS = (
    "help_mode", "help_ema_tau", "help_min_samples", "help_wilson_z", "help_lookahead_bins", "help_lookahead_decay",
    "help_success_lo", "help_success_hi", "help_update_interval", "help_final_iteration", "help_final_fraction",
    "help_final_ramp_fraction",
)


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


# ---------------------------------------------------------------- config contract
def test_helpfix_arm_differs_from_c3r2_only_in_the_help_fields_the_length_and_names():
    r2, arm = _flatten(_compose(R2)), _flatten(_compose(HELPFIX))
    changed = {key for key in r2.keys() | arm.keys() if r2.get(key, KeyError) != arm.get(key, KeyError)}
    allowed = {f"{root}.{name}" for root in ("reset", "env.reset") for name in HELP_FIELDS} | {*NAMES, *TRAINING}
    unexpected = sorted(changed - allowed)
    assert not unexpected, f"{HELPFIX} drifted from {R2}: {unexpected}"
    assert arm["reset.help_mode"] == arm["env.reset.help_mode"] == "success"
    assert arm["reset.help_final_iteration"] == arm["training.max_iterations"] == 90000
    assert arm["training.early_stop_help_zero_tail"] == 0
    assert "reset.run_in_steps" not in arm  # no rolling start (C3r2 default 0)
    assert "reset.help_mode" not in r2


def test_config_defaults_and_bounds():
    from flash_chord.lifecycle.rsi_reset import RSIReconBodyResetConfig

    assert RSIReconBodyResetConfig().help_mode == "deadline"
    RSIReconBodyResetConfig(help_mode="success")
    with pytest.raises(ValueError):
        RSIReconBodyResetConfig(help_mode="schedule")
    with pytest.raises(ValueError):
        RSIReconBodyResetConfig(help_success_lo=0.9, help_success_hi=0.8)
    with pytest.raises(ValueError):
        RSIReconBodyResetConfig(help_final_ramp_fraction=0.9, help_final_fraction=0.85)
    with pytest.raises(ValueError):
        RSIReconBodyResetConfig(help_lookahead_bins=0)


def test_export_replaces_the_rsi_reset_with_the_stock_fields():
    from flash_chord.training.ppo.coordinate import export

    stock, _ = export.stock_config_with_notes(_compose(HELPFIX))
    r2_stock, _ = export.stock_config_with_notes(_compose(R2))
    assert not any(key.startswith("help_") for key in stock["env"]["reset"])
    assert stock["env"]["reset"] == r2_stock["env"]["reset"]


# ---------------------------------------------------------------- host logic
def _success_bins(num_frames: int = 440, **overrides) -> RSIBins:
    params = {"help_mode": "success", "wait_iterations": 0, "help_update_interval": 10, **overrides}
    return RSIBins(num_frames, RSIBinParams(**params))


def _feed(bins: RSIBins, pass_prob, iterations: int, start: int = 0, per_bin: int = 200, seed: int = 0):
    """Feed ``per_bin`` pass-through outcomes per bin and iteration with pass probabilities ``pass_prob``; decide every
    iteration; return the tops after each iteration."""
    rng = np.random.default_rng(seed)
    prob = np.broadcast_to(np.asarray(pass_prob, dtype=np.float64), (bins.num_bins,))
    history = []
    for t in range(start, start + iterations):
        passes = rng.binomial(per_bin, prob)
        bins.observe_pass(passes, per_bin - passes)
        bins.decide(t)
        history.append(bins.top.copy())
    return np.asarray(history)


def test_tops_never_rise_and_fall_at_most_one_step_per_update():
    bins = _success_bins()
    rng = np.random.default_rng(1)
    prob = rng.uniform(0.3, 1.0, size=bins.num_bins)
    history = _feed(bins, prob, 3000)
    before, after = history[:-1], history[1:]
    assert (after <= before + 1e-15).all(), "a help top rose"
    moved = after < before
    zeroed = after == 0.0
    ratio = np.divide(after, before, out=np.ones_like(after), where=before > 0)
    assert (ratio[moved & ~zeroed] >= bins.params.step - 1e-12).all(), "a top fell by more than one step"
    assert (np.diff(np.flatnonzero(moved.any(axis=1))) % bins.params.help_update_interval == 0).all()


def test_mastered_bins_reach_zero_and_a_hard_bin_keeps_its_help_and_shields_the_bins_before_it():
    bins = _success_bins(help_lookahead_bins=3, help_lookahead_decay=0.8)
    prob = np.full(bins.num_bins, 0.995)
    hard = 10
    prob[hard] = 0.3
    history = _feed(bins, prob, 6000)
    top = history[-1]
    assert top[hard] == pytest.approx(bins.params.top0)  # a failing bin keeps full help
    assert (top[hard - 2 : hard] > 0).all()  # its lookahead window keeps help in front of it
    far = [b for b in range(bins.num_bins) if b < hard - 2 or b > hard]
    assert (top[far] == 0.0).all()  # bins the policy passes lose all help


def test_a_few_samples_never_move_a_top():
    bins = _success_bins(help_min_samples=1e9)
    history = _feed(bins, 1.0, 500)
    assert (history == bins.params.top0).all()


def test_a_single_lucky_iteration_does_not_drop_help():
    bins = _success_bins()
    _feed(bins, 0.2, 1000)  # long failing history at full help
    start = bins.top.copy()
    bins.observe_pass(np.full(bins.num_bins, 200), np.zeros(bins.num_bins))  # one perfect iteration
    bins.decide(1000)
    np.testing.assert_array_equal(bins.top, start)


def test_final_cutoff_ramps_then_zeroes_every_top():
    bins = _success_bins(help_final_iteration=1000, help_final_fraction=0.85, help_final_ramp_fraction=0.05)
    assert bins.final_window() == (800, 850)
    assert bins.final_ceiling(799) == pytest.approx(1.0)
    assert bins.final_ceiling(825) == pytest.approx(bins.params.zero_below**0.5)
    assert bins.final_ceiling(850) == 0.0
    history = _feed(bins, 0.2, 900)  # failing everywhere: only the cutoff can lower help
    assert (history[799] == bins.params.top0).all()
    assert (history[849] > 0).all() and (history[849] < 1e-2).all()
    assert (history[850:] == 0.0).all()
    assert bins.zero_since() == 850


def test_deadline_mode_ignores_the_pass_through_counts():
    a = RSIBins(440, RSIBinParams(wait_iterations=0, deadline=500, zero_ramp=50))
    b = RSIBins(440, RSIBinParams(wait_iterations=0, deadline=500, zero_ramp=50))
    rng = np.random.default_rng(2)
    for t in range(700):
        n = rng.integers(0, 100, size=a.num_bins)
        f = rng.binomial(n, 0.5)
        for bins in (a, b):
            bins.observe(n, f)
        b.observe_pass(rng.integers(0, 50, size=b.num_bins), rng.integers(0, 50, size=b.num_bins))
        a.decide(t)
        b.decide(t)
        np.testing.assert_array_equal(a.top, b.top)


def test_state_round_trip_and_legacy_state():
    a = _success_bins()
    _feed(a, 0.9, 300)
    b = _success_bins()
    b.load_state_dict({key: value.copy() for key, value in a.state_dict().items()})
    _feed(a, 0.95, 200, start=300, seed=5)
    _feed(b, 0.95, 200, start=300, seed=5)
    for key, value in a.state_dict().items():
        np.testing.assert_array_equal(value, b.state_dict()[key])
    legacy = {key: value for key, value in a.state_dict().items() if key not in ("pass_ema", "fail_ema")}
    c = _success_bins()
    c.load_state_dict(legacy)
    assert (c.pass_ema == 0).all() and (c.fail_ema == 0).all()
    np.testing.assert_array_equal(c.top, a.top)
    with pytest.raises(ValueError):
        c.load_state_dict({key: value for key, value in a.state_dict().items() if key != "fail_ema"})


# ---------------------------------------------------------------- CPU kernels
def test_help_at_the_current_frame_interpolates_between_bin_centres():
    import warp as wp

    from flash_chord.lifecycle.rsi import kernels

    with wp.ScopedDevice("cpu"):
        top = wp.array([1.0, 0.5, 0.0, 0.2], dtype=wp.float32)
        frames = [0, 12, 25, 37, 50, 62, 87, 200]
        n = len(frames)
        zeros = wp.zeros(n, dtype=wp.int32)
        active = wp.ones(n, dtype=wp.int32)
        timestep = wp.array(frames, dtype=wp.int32)
        factor = wp.full(n, 0.5, dtype=wp.float32)
        lane_help = wp.zeros(n, dtype=wp.float32)
        wp.launch(kernels.update_frame_help, dim=n, inputs=[zeros, zeros, active, timestep, factor, top, 25, 4],
                  outputs=[lane_help])
        x = (np.asarray(frames) + 0.5) / 25 - 0.5
        expected = 0.5 * np.interp(x, np.arange(4), [1.0, 0.5, 0.0, 0.2])
        np.testing.assert_allclose(lane_help.numpy(), expected, atol=1e-6)
        # done and explicit-reset lanes keep their help
        lane_help.fill_(7.0)
        done = wp.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=wp.int32)
        inactive = wp.array([1, 0, 1, 1, 1, 1, 1, 1], dtype=wp.int32)
        wp.launch(kernels.update_frame_help, dim=n, inputs=[done, zeros, inactive, timestep, factor, top, 25, 4],
                  outputs=[lane_help])
        assert lane_help.numpy()[0] == 7.0 and lane_help.numpy()[1] == 7.0


Q_TEST = 12


def _bound_policy(world_count: int, **overrides):
    import warp as wp

    from flash_chord.lifecycle.rsi_reset import RSIReconBodyResetConfig

    policy = RSIReconBodyResetConfig(keyframes=False, frame0_share=0.0, **overrides).build(
        world_count, 440, device="cpu"
    )
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


def test_pass_through_counts_attribute_a_failure_to_the_bin_where_it_happens():
    policy, env, state = _bound_policy(4, help_mode="success")
    policy.active.assign(np.asarray([1, 1, 0, 1], dtype=np.int32))
    policy.visit_bin.assign(np.asarray([2, 0, 0, 5], dtype=np.int32))
    env.timestep.assign(np.asarray([4 * 25 + 3, 10, 200, 5 * 25 + 7], dtype=np.int32))
    env.terminated.assign(np.asarray([1, 0, 0, 0], dtype=np.int32))
    env.truncated.assign(np.asarray([0, 1, 0, 0], dtype=np.int32))
    policy.on_transition(state)
    passes, fails = policy.pass_iter.numpy(), policy.pass_fail_iter.numpy()
    expected_passes = np.zeros(policy.num_bins, dtype=np.int32)
    expected_passes[[2, 3]] = 1  # world 0 left bins 2 and 3
    expected_passes[0] = 1  # world 1 reached the clip end in bin 0
    expected_fails = np.zeros(policy.num_bins, dtype=np.int32)
    expected_fails[4] = 1  # world 0 terminated in bin 4
    np.testing.assert_array_equal(passes, expected_passes)
    np.testing.assert_array_equal(fails, expected_fails)
    np.testing.assert_array_equal(policy.visit_bin.numpy(), [4, 0, 0, 5])  # world 2 is not scored
    metrics = policy.host_update(0)
    assert metrics["rsi/pass_through_iter"] == 4.0
    assert (policy.pass_iter.numpy() == 0).all() and (policy.pass_fail_iter.numpy() == 0).all()


def test_success_mode_sample_starts_the_lane_at_its_frame_help_and_records_its_bin():
    import warp as wp

    policy, _, _ = _bound_policy(2000, help_mode="success")
    tops = np.linspace(1.0, 0.0, policy.num_bins)
    policy.bins.top[:] = tops
    policy.upload_bins()
    policy.sample(wp.ones(2000, dtype=wp.int32, device="cpu"), wp.zeros(1, dtype=wp.float32, device="cpu"))
    frame = policy.reset_frame.numpy()
    factor = policy.help_factor.numpy()
    assert (factor >= 0.5 - 1e-6).all() and (factor <= 1.0 + 1e-6).all()
    np.testing.assert_array_equal(policy.visit_bin.numpy(), policy.start_bin.numpy())
    x = (frame + 0.5) / policy.config.bin_frames - 0.5
    expected = factor * np.interp(x, np.arange(policy.num_bins), tops)
    np.testing.assert_allclose(policy.help.numpy(), expected, atol=1e-5)
