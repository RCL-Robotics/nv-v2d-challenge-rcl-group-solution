# SPDX-License-Identifier: Apache-2.0
"""PSI reset (InterMimic physical state initialization): config contract, snapshot scoring, buffer insertion,
inverse-score start sampling and stored-state restore. CPU-device kernel tests with synthetic buffers, in the style of
tests/rsi/test_keyframe_roundtrip.py."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from flash_chord.configuration import compose_config, resolved_dict

W, Q, D, ROOT_Q = 8, 12, 10, 1
NUM_FRAMES = 120  # max_start_frame 109
PSI_RESET_TARGET = "flash_chord.lifecycle.psi_reset.PSIReconBodyResetConfig"
R2 = "g1_recon_body_coord_rsi_r2"
PSI = "g1_recon_body_coord_psi_r2"
NAMES = ("logging.experiment", "logging.run_name", "training.experiment_name", "training.run_name")


# ====================================================================== config contract
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


def test_psi_arm_differs_from_c3r2_only_in_the_reset_early_stop_and_names():
    r2, psi = _flatten(_compose(R2)), _flatten(_compose(PSI))
    changed = {key for key in r2.keys() | psi.keys() if r2.get(key, KeyError) != psi.get(key, KeyError)}
    allowed = {"training.early_stop_help_zero_tail", *NAMES}
    unexpected = sorted(key for key in changed if key not in allowed and not key.startswith(("reset.", "env.reset.")))
    assert not unexpected, f"{PSI} drifted from {R2} outside the reset: {unexpected}"
    assert psi["reset._target_"] == psi["env.reset._target_"] == PSI_RESET_TARGET
    assert psi["training.early_stop_help_zero_tail"] == 0
    # The stock reset fields keep the stock (and C3r2) values.
    for name in ("reset_freeze_steps", "voc_decay_steps", "reset_voc_scale", "shoulder_spread", "seed"):
        assert psi[f"reset.{name}"] == r2[f"reset.{name}"], name


def test_psi_arm_instantiates_with_intermimic_defaults():
    from flash_chord.configuration import instantiate_typed
    from flash_chord.lifecycle.psi_reset import PSIReconBodyResetConfig

    config = instantiate_typed(_compose(PSI)["reset"], PSIReconBodyResetConfig)
    assert (config.frame0_share, config.psi_slots, config.psi_horizon, config.psi_threshold) == (0.1, 2, 300, 0.5)
    assert config.keyframes is False


def test_config_rejects_invalid_values():
    from flash_chord.lifecycle.psi_reset import PSIReconBodyResetConfig

    with pytest.raises(ValueError):
        PSIReconBodyResetConfig(keyframes=True)
    with pytest.raises(ValueError):
        PSIReconBodyResetConfig(psi_slots=0)
    with pytest.raises(ValueError):
        PSIReconBodyResetConfig(psi_threshold=1.0)


# ====================================================================== CPU kernel tests
def _cpu_setup(anchor_all: bool = True, **overrides):
    import warp as wp

    from flash_chord.lifecycle.psi_reset import PSIReconBodyResetConfig

    settings = dict(psi_horizon=40, psi_stride=10, psi_settle_steps=10, psi_score_decay=0.0)
    settings.update(overrides)
    policy = PSIReconBodyResetConfig(**settings).build(W, NUM_FRAMES, device="cpu")
    if anchor_all:
        # Every start at frame 0 (the config only accepts shares below 1; the kernel reads the device value).
        policy.frame0_share.assign(np.asarray([1.0], dtype=np.float32))
    reference = np.random.default_rng(0).uniform(0.1, 0.9, size=(NUM_FRAMES, 10)).astype(np.float32)
    offset = wp.zeros(W * Q, dtype=wp.float32, device="cpu")
    policy.bind_arrays(
        q_per_world=Q, left_shoulder_q_id=3, right_shoulder_q_id=4, reference_joint_q=reference,
        finger_q_ids=(5, 6), output=offset,
    )
    object_offsets = np.stack([0.1 * np.arange(W), 0.2 * np.arange(W), -0.3 * np.arange(W)], axis=1)

    def i32():
        return wp.zeros(W, dtype=wp.int32, device="cpu")

    def f32():
        return wp.zeros(W, dtype=wp.float32, device="cpu")

    env = SimpleNamespace(
        world_count=W,
        model=SimpleNamespace(joint_coord_count=W * Q, joint_dof_count=W * D),
        termination=SimpleNamespace(object_position_error=f32(), object_orientation_error=f32()),
        terminated=i32(),
        truncated=i32(),
        timestep=i32(),
        episode_step=i32(),
        reset_table=SimpleNamespace(
            object_root_q_ids=wp.array([ROOT_Q], dtype=wp.int32, device="cpu"),
            object_root_position_offsets_w=wp.array(object_offsets.astype(np.float32), dtype=wp.vec3, device="cpu"),
            num_objects=1,
        ),
    )
    policy.bind_env(env)
    state = SimpleNamespace(
        joint_q=wp.zeros(W * Q, dtype=wp.float32, device="cpu"),
        joint_qd=wp.zeros(W * D, dtype=wp.float32, device="cpu"),
    )
    return policy, env, state, object_offsets


def _set(array, values):
    array.assign(np.asarray(values, dtype=array.numpy().dtype))


def _all(value=1):
    import warp as wp

    return wp.array(np.full(W, value, dtype=np.int32), dtype=wp.int32, device="cpu")


def _state_at(step):
    w = np.arange(W)
    q = 1000.0 * w[:, None] + np.arange(Q)[None, :] + 0.01 * step
    qd = -(1000.0 * w[:, None] + np.arange(D)[None, :]) - 0.01 * step
    return q, qd


def test_sample_with_an_empty_buffer_starts_from_the_reference_without_help():
    import warp as wp

    policy, env, state, _ = _cpu_setup(anchor_all=False, frame0_share=0.0)
    policy.sample(_all(), wp.zeros(1, dtype=wp.float32, device="cpu"))
    frames = policy.reset_frame.numpy()
    assert ((frames >= 0) & (frames <= policy.max_start_frame)).all()
    np.testing.assert_array_equal(policy.start_slot.numpy(), -1)
    np.testing.assert_array_equal(policy.help.numpy(), 0.0)
    voc = policy.applied_voc_scale.numpy()
    np.testing.assert_allclose(voc[frames > 0], policy.config.reset_voc_scale)  # freeze-time hold, released to 0
    np.testing.assert_allclose(voc[frames == 0], 0.0)
    # every start waits for the termination grace (reset_freeze_steps) to end before its first snapshot
    np.testing.assert_array_equal(policy.capture_after.numpy(), policy.config.reset_freeze_steps + 10)


def _run_episode(policy, env, state, term_step, last_step):
    """Lanes start at frame 0 (immediate) and advance one frame per step; lane w terminates at step term_step[w]
    (0 = never). Returns {step: (q, qd)} of the synthetic states."""
    import warp as wp

    policy.sample(_all(), wp.zeros(1, dtype=wp.float32, device="cpu"))
    assert (policy.reset_frame.numpy() == 0).all()
    states = {}
    for s in range(1, last_step + 1):
        term = np.array([1 if t == s else 0 for t in term_step], dtype=np.int32)
        _set(env.terminated, term)
        _set(env.truncated, np.zeros(W))
        _set(env.episode_step, np.where(term == 1, s - 1, s))  # the advance skips a world that ends this step
        _set(env.timestep, np.full(W, s))
        q, qd = _state_at(s)
        _set(state.joint_q, q.reshape(-1))
        _set(state.joint_qd, qd.reshape(-1))
        states[s] = (q, qd)
        policy.on_transition(state)
    return states


# Snapshots start at step 60 (grace 50 + settle 10), every 10 steps; horizon 40.
# Lane 1 keeps (60, 0.625); lane 4 (60, 0.875) and (70, 0.625); lanes never ending keep (60, 1.0) at step 100.
TERM = [75, 85, 0, 0, 95, 0, 0, 0]
LAST = 105


def _expected_queue():
    # time order of acceptance: lane 1 at step 85, lane 4 at 95, lanes 2, 3, 5, 6, 7 at 100 (window of 40 survived)
    return [(1, 60, 0.625), (4, 60, 0.875), (4, 70, 0.625)] + [(w, 60, 1.0) for w in (2, 3, 5, 6, 7)]


def test_snapshots_score_the_survived_share_of_the_window_and_queue_those_above_half():
    policy, env, state, offsets = _cpu_setup()
    states = _run_episode(policy, env, state, TERM, last_step=LAST)
    expected = _expected_queue()
    count = int(policy.cand_count.numpy()[0])
    assert count == len(expected)
    frames = policy.cand_frame.numpy()[:count]
    scores = policy.cand_score.numpy()[:count]
    got = sorted(zip(frames.tolist(), np.round(scores, 4).tolist()))
    assert got == sorted((f, s) for _, f, s in expected)
    # stored states are the snapshot step's state without the world's object offset
    cand_q = policy.cand_q.numpy()[: count * Q].reshape(count, Q)
    for i in range(count):
        frame = int(frames[i])
        match = [w for w, f, s in expected if f == frame and abs(s - scores[i]) < 1e-4]
        q_expected = [states[frame][0][w].copy() for w in match]
        for row, w in zip(q_expected, match):
            row[ROOT_Q : ROOT_Q + 3] -= offsets[w]
        assert any(np.allclose(cand_q[i], row, atol=1e-4) for row in q_expected)
    # nothing at frame 0 (the exact start) and no ring overwrites
    assert (frames > 0).all()
    assert int(policy.step_stats.numpy()[4]) == 0


def test_host_update_keeps_the_best_slots_of_each_frame_and_samples_frames_by_inverse_score():
    policy, env, state, offsets = _cpu_setup()
    states = _run_episode(policy, env, state, TERM, last_step=LAST)
    metrics = policy.host_update(0)
    # frame 60 queued 0.625, 0.875 and five 1.0: only its best two (both 1.0) are inserted; frame 70 its one 0.625
    assert metrics["psi/inserted"] == 3.0
    assert int(policy.cand_count.numpy()[0]) == 0
    np.testing.assert_allclose(policy.score[60], [1.0, 1.0])
    np.testing.assert_allclose(policy.score[70], [0.625, 0.0])
    stored = policy.psi_q_host.reshape(NUM_FRAMES, 2, Q)[60]
    full_window = []
    for w in (2, 3, 5, 6, 7):
        row = states[60][0][w].copy()
        row[ROOT_Q : ROOT_Q + 3] -= offsets[w]
        full_window.append(row)
    for slot in range(2):  # both slots hold states of lanes that survived the whole window
        assert any(np.allclose(stored[slot], row, atol=1e-4) for row in full_window)
    filled = policy.score[: policy.num_cdf_frames].sum(axis=1)
    weights = 1.0 / (1.0 + filled)
    np.testing.assert_allclose(policy.frame_cdf.numpy(), np.cumsum(weights / weights.sum()), rtol=0, atol=1e-6)
    np.testing.assert_allclose(policy.slot_score.numpy().reshape(NUM_FRAMES, 2)[60], [1.0, 1.0])


def test_top_per_group_keeps_the_last_k_of_every_run():
    from flash_chord.lifecycle.psi_reset import _top_per_group

    groups = np.array([3, 3, 3, 5, 7, 7])
    np.testing.assert_array_equal(_top_per_group(groups, 2), [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(_top_per_group(groups, 1), [2, 3, 5])
    assert _top_per_group(np.zeros(0, dtype=np.int32), 2).shape == (0,)


def test_scores_decay_each_iteration():
    policy, env, state, _ = _cpu_setup(psi_score_decay=0.5)
    _run_episode(policy, env, state, TERM, last_step=LAST)
    policy.host_update(0)
    np.testing.assert_allclose(policy.score[60], [0.5, 0.5])
    policy.host_update(1)
    np.testing.assert_allclose(policy.score[60], [0.25, 0.25])


def test_a_drawn_simulation_slot_is_restored_with_the_world_offset_and_starts_immediately():
    import warp as wp

    policy, env, state, offsets = _cpu_setup()
    _run_episode(policy, env, state, TERM, last_step=LAST)
    policy.host_update(0)
    policy.frame0_share.assign(np.asarray([0.0], dtype=np.float32))
    # Put every start on frame 60 and make slot 0 overwhelmingly likely.
    policy.score[60] = [1.0e6, 0.0]
    policy.upload_buffer()
    cdf = np.zeros(policy.num_cdf_frames, dtype=np.float32)
    cdf[60:] = 1.0
    policy.frame_cdf.assign(cdf)
    policy.sample(_all(), wp.zeros(1, dtype=wp.float32, device="cpu"))
    np.testing.assert_array_equal(policy.reset_frame.numpy(), 60)
    np.testing.assert_array_equal(policy.start_slot.numpy(), 0)
    np.testing.assert_array_equal(policy.steps_since_reset.numpy(), policy.config.reset_freeze_steps + 1)
    np.testing.assert_allclose(policy.applied_voc_scale.numpy(), 0.0)
    np.testing.assert_allclose(policy.reference_joint_q_offset.numpy(), 0.0)
    policy.restore(state, _all())
    stored = policy.psi_q_host.reshape(NUM_FRAMES, 2, Q)[60, 0]
    restored = state.joint_q.numpy().reshape(W, Q)
    for w in range(W):
        row = stored.copy()
        row[ROOT_Q : ROOT_Q + 3] += offsets[w]
        np.testing.assert_allclose(restored[w], row, atol=1e-4)
    stored_qd = policy.psi_qd_host.reshape(NUM_FRAMES, 2, D)[60, 0]
    np.testing.assert_allclose(state.joint_qd.numpy().reshape(W, D), np.broadcast_to(stored_qd, (W, D)), atol=1e-5)


def test_save_and_load_round_trip(tmp_path):
    policy, env, state, _ = _cpu_setup()
    _run_episode(policy, env, state, TERM, last_step=LAST)
    policy.host_update(0)
    path = tmp_path / "rsi_500.safetensors"
    policy.save_state(path)
    other, other_env, _, _ = _cpu_setup()
    other.load_state(path)
    np.testing.assert_allclose(other.score, policy.score)
    np.testing.assert_allclose(other.psi_q_host, policy.psi_q_host)
    np.testing.assert_allclose(other.psi_qd_host, policy.psi_qd_host)
    np.testing.assert_allclose(other.frame_cdf.numpy(), policy.frame_cdf.numpy())
    assert other.inserted_total == policy.inserted_total == 3
