# SPDX-License-Identifier: Apache-2.0
"""RSI keyframe bank: capture -> commit (kf_to_end) -> sample -> restore.

CPU-device kernel tests drive ``on_transition`` / ``sample`` / ``restore_keyframe`` with synthetic buffers (including
non-zero object root offsets). The ``gpu``/``slow`` tests run the real G1 + Dex3 ReconBody environment on Tier-1
episode 17 through the production ``flash_chord.envs.rl.RLEnv``, whose ``reset`` and ``_rl_step`` call the RSI hooks
as docs/coord_impl_spec.md section 5.1 prescribes (the hooks were first tested here through a subclass copy of that
wiring, before ``envs/rl.py`` was edited).
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SEQUENCE = "2026-05-01_17-04-55_traffic_cone_ground_table_02"
_DATA = Path(os.environ.get("HUMAN_MOTION_DATA_DIR", "/root/autodl-tmp/haoran.zhang/data/human_motion_data"))
PARQUET = _DATA / "whole_body" / "soma" / f"sequence_id={SEQUENCE}" / "robot_name=g1" / "data.parquet"
SUPPORT = _DATA / "whole_body" / "reconstructed_stage" / f"{SEQUENCE}_support.usda"

W_CPU, Q_CPU, D_CPU, ROOT_Q = 8, 12, 10, 1


# ====================================================================== CPU kernel tests
def _cpu_setup(**overrides):
    import warp as wp

    from flash_chord.lifecycle.rsi_reset import RSIReconBodyResetConfig

    settings = dict(
        kf_prob=1.0, kf_min_task=0.5, kf_step_score_min=0.3, kf_min_stepped=15, kf_capacity=4, kf_min_fill=2,
        frame0_share=0.0,
    )
    settings.update(overrides)
    num_frames = 120  # F_max 109, bins [0,25) [25,50) [50,75) [75,100) [100,110)
    policy = RSIReconBodyResetConfig(**settings).build(W_CPU, num_frames, device="cpu")
    reference = np.random.default_rng(0).uniform(0.1, 0.9, size=(num_frames, 10)).astype(np.float32)
    offset = wp.zeros(W_CPU * Q_CPU, dtype=wp.float32, device="cpu")
    policy.bind_arrays(
        q_per_world=Q_CPU, left_shoulder_q_id=3, right_shoulder_q_id=4, reference_joint_q=reference,
        finger_q_ids=(5, 6), output=offset,
    )
    object_offsets = np.stack([0.1 * np.arange(W_CPU), 0.2 * np.arange(W_CPU), -0.3 * np.arange(W_CPU)], axis=1)

    def i32():
        return wp.zeros(W_CPU, dtype=wp.int32, device="cpu")

    def f32():
        return wp.zeros(W_CPU, dtype=wp.float32, device="cpu")

    env = SimpleNamespace(
        world_count=W_CPU,
        model=SimpleNamespace(joint_coord_count=W_CPU * Q_CPU, joint_dof_count=W_CPU * D_CPU),
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
        joint_q=wp.zeros(W_CPU * Q_CPU, dtype=wp.float32, device="cpu"),
        joint_qd=wp.zeros(W_CPU * D_CPU, dtype=wp.float32, device="cpu"),
    )
    return policy, env, state, object_offsets


def _set(array, values):
    array.assign(np.asarray(values, dtype=array.numpy().dtype))


def _step(policy, env, state, *, step, timestep, terminated=0, truncated=0, pos_err=0.0, rot_err=0.0, q_scale=1.0):
    """One synthetic post-advance transition: counters, errors and a recognisable joint state per world."""
    w = np.arange(W_CPU)
    _set(env.episode_step, np.broadcast_to(step, (W_CPU,)))
    _set(env.timestep, np.broadcast_to(timestep, (W_CPU,)))
    _set(env.terminated, np.broadcast_to(terminated, (W_CPU,)))
    _set(env.truncated, np.broadcast_to(truncated, (W_CPU,)))
    _set(env.termination.object_position_error, np.broadcast_to(pos_err, (W_CPU,)))
    _set(env.termination.object_orientation_error, np.broadcast_to(rot_err, (W_CPU,)))
    q = q_scale * (100.0 * w[:, None] + np.arange(Q_CPU)[None, :] + 0.5)
    qd = -q_scale * (100.0 * w[:, None] + np.arange(D_CPU)[None, :])
    _set(state.joint_q, q.reshape(-1))
    _set(state.joint_qd, qd.reshape(-1))
    policy.on_transition(state)
    return q, qd


def test_transition_captures_commits_at_clip_end_and_drops_otherwise():
    policy, env, state, offsets = _cpu_setup()
    # Lanes 0-4 are healthy clean lanes; 5 is not clean; 6 sits in bin 0; 7 tracks badly on the capture step.
    _set(policy.clean, [1, 1, 1, 1, 1, 0, 1, 1])
    _set(policy.active, np.ones(W_CPU))
    _set(policy.start_bin, [1, 1, 2, 2, 3, 0, 0, 0])
    _set(policy.tag, np.full(W_CPU, 0.7))
    _set(policy.cum_score, np.full(W_CPU, 14.0))  # 14 earlier steps with task score 1
    timestep = np.array([40, 40, 40, 40, 40, 40, 10, 40])
    pos_err = np.array([0, 0, 0, 0, 0, 0, 0, 0.5])
    q1, qd1 = _step(policy, env, state, step=14, timestep=timestep, pos_err=pos_err)
    np.testing.assert_array_equal(policy.pend_on.numpy(), 0)  # 14 < kf_min_stepped
    _set(policy.cum_score, np.full(W_CPU, 14.0))
    q1, qd1 = _step(policy, env, state, step=15, timestep=timestep, pos_err=pos_err)
    np.testing.assert_array_equal(policy.pend_on.numpy(), [1, 1, 1, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(policy.pend_frame.numpy()[:5], 40)
    np.testing.assert_array_equal(policy.pend_step.numpy()[:5], 15)
    np.testing.assert_allclose(policy.pend_cum.numpy()[:5], 15.0)
    pend_q = policy.pend_q.numpy().reshape(W_CPU, Q_CPU)
    expected = q1.copy()
    expected[:, ROOT_Q : ROOT_Q + 3] -= offsets  # stored without the world's object offset
    np.testing.assert_allclose(pend_q[:5], expected[:5], rtol=0, atol=1e-4)
    np.testing.assert_allclose(policy.pend_qd.numpy().reshape(W_CPU, D_CPU)[:5], qd1[:5])

    # Resolution step: 0 reaches the clip end, 1 fails, 2 fails on the last frame, 3 goes on,
    # 4 reaches the end with a poor window. Done worlds were not advanced (episode_step stays 15).
    step = np.array([15, 15, 15, 16, 15, 16, 16, 16])
    terminated = np.array([0, 1, 1, 0, 0, 0, 0, 0])
    truncated = np.array([1, 0, 1, 0, 1, 0, 0, 0])
    pos_err = np.array([0, 0, 0, 0, 1.0, 0, 0, 0.5])
    timestep = np.array([41, 41, 41, 41, 41, 41, 11, 41])  # lane 6 stays in bin 0, lane 7 keeps tracking badly
    _step(policy, env, state, step=step, timestep=timestep, terminated=terminated, truncated=truncated,
          pos_err=pos_err, q_scale=2.0)
    bank = policy.bank
    np.testing.assert_array_equal(bank.write_count.numpy(), [0, 1, 0, 0, 0])
    np.testing.assert_array_equal(bank.commit_total.numpy(), [0, 1, 0, 0, 0])
    assert int(bank.drop_total.numpy()[0]) == 3  # lanes 1, 2, 4
    np.testing.assert_array_equal(policy.pend_on.numpy(), [0, 0, 0, 1, 0, 0, 0, 0])
    stored_q = bank.q.numpy().reshape(5, 4, Q_CPU)[1, 0]
    np.testing.assert_allclose(stored_q, expected[0], rtol=0, atol=1e-4)  # the captured state, not the final one
    np.testing.assert_allclose(bank.qd.numpy().reshape(5, 4, D_CPU)[1, 0], qd1[0])
    assert bank.frame.numpy().reshape(5, 4)[1, 0] == 40
    assert bank.tag.numpy().reshape(5, 4)[1, 0] == np.float32(0.7)
    # Outcomes by start bin: n counts every finished active lane, f only terminated-and-not-truncated.
    np.testing.assert_array_equal(policy.n_iter.numpy(), [0, 2, 1, 1, 0])
    np.testing.assert_array_equal(policy.f_iter.numpy(), [0, 1, 0, 0, 0])
    np.testing.assert_array_equal(policy.active.numpy(), [0, 0, 0, 1, 0, 1, 1, 1])

    # Finished lanes would restart (from reference starts here): only lane 3 remains a clean, pending lane.
    _set(policy.clean, [0, 0, 0, 1, 0, 0, 0, 0])
    # The still-pending lane is not captured again, and commits when it reaches the end.
    _step(policy, env, state, step=17, timestep=42)
    assert policy.pend_on.numpy().tolist() == [0, 0, 0, 1, 0, 0, 0, 0]
    assert int(bank.write_count.numpy()[1]) == 1
    _step(policy, env, state, step=17, timestep=119, truncated=[0, 0, 0, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(bank.write_count.numpy(), [0, 2, 0, 0, 0])
    np.testing.assert_allclose(bank.q.numpy().reshape(5, 4, Q_CPU)[1, 1], expected[3], rtol=0, atol=1e-4)
    np.testing.assert_array_equal(policy.n_iter.numpy(), [0, 2, 2, 1, 0])

    # Host update folds the counts into the bins and zeroes the device counters.
    policy.stats_interval = 1
    metrics = policy.host_update(0)
    assert metrics["rsi/episodes_scored"] == 5.0 and metrics["rsi/kf_committed_total"] == 2.0
    assert metrics["rsi/kf_dropped_total"] == 3.0 and metrics["rsi/kf_bins_filled"] == 1.0
    assert metrics["rsi/bin/kf_eligible_01"] == 2.0
    np.testing.assert_array_equal(policy.bins.win_n, [0, 2, 2, 1, 0])
    np.testing.assert_array_equal(policy.bins.win_f, [0, 1, 0, 0, 0])
    np.testing.assert_array_equal(policy.n_iter.numpy(), 0)


def _fill_bin_one(policy, env, state):
    """Two committed states in bin 1 (frame 40, tag 0.7) via the real capture/commit kernel."""
    _set(policy.clean, np.ones(W_CPU))
    _set(policy.active, np.zeros(W_CPU))
    _set(policy.tag, np.full(W_CPU, 0.7))
    _set(policy.cum_score, np.full(W_CPU, 14.0))
    _step(policy, env, state, step=15, timestep=40)
    truncated = np.zeros(W_CPU, dtype=np.int32)
    truncated[:2] = 1
    _step(policy, env, state, step=np.where(truncated, 15, 16), timestep=41, truncated=truncated, q_scale=3.0)
    assert int(policy.bank.write_count.numpy()[1]) == 2


def test_sample_serves_eligible_keyframes_and_restore_writes_them():
    import warp as wp

    from flash_chord.lifecycle.rsi import kernels

    policy, env, state, offsets = _cpu_setup()
    _fill_bin_one(policy, env, state)
    policy.cdf.assign(np.array([0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32))  # every draw lands in bin 1
    mask = wp.ones(W_CPU, dtype=wp.int32, device="cpu")
    policy.sample(mask, wp.zeros(1, dtype=wp.float32, device="cpu"))
    slot = policy.kf_slot.numpy()
    assert set(slot.tolist()) <= {0, 1} and (slot >= 0).all()
    np.testing.assert_array_equal(policy.reset_frame.numpy(), 40)
    np.testing.assert_array_equal(policy.start_bin.numpy(), 1)
    np.testing.assert_array_equal(policy.steps_since_reset.numpy(), 51)  # immediate: no freeze
    np.testing.assert_array_equal(policy.clean.numpy(), 1)
    np.testing.assert_array_equal(policy.reference_joint_q_offset.numpy(), 0.0)
    np.testing.assert_array_equal(policy.served.numpy(), [0, W_CPU, 0, 0, 0])
    np.testing.assert_array_equal(policy.requested.numpy(), [0, W_CPU, 0, 0, 0])
    np.testing.assert_allclose(policy.applied_voc_scale.numpy(), policy.help.numpy())

    joint_q = wp.zeros(W_CPU * Q_CPU, dtype=wp.float32, device="cpu")
    joint_qd = wp.zeros(W_CPU * D_CPU, dtype=wp.float32, device="cpu")
    restore_mask = np.zeros(W_CPU, dtype=np.int32)
    restore_mask[[2, 5]] = 1
    wp.launch(
        kernels.restore_keyframe,
        dim=W_CPU,
        inputs=[wp.array(restore_mask, dtype=wp.int32, device="cpu"), policy.kf_slot, policy.start_bin,
                policy.bank.capacity, Q_CPU, D_CPU, policy.bank.q, policy.bank.qd,
                env.reset_table.object_root_q_ids, env.reset_table.object_root_position_offsets_w, 1],
        outputs=[joint_q, joint_qd],
        device="cpu",
    )
    bank_q = policy.bank.q.numpy().reshape(5, 4, Q_CPU)
    bank_qd = policy.bank.qd.numpy().reshape(5, 4, D_CPU)
    restored_q = joint_q.numpy().reshape(W_CPU, Q_CPU)
    restored_qd = joint_qd.numpy().reshape(W_CPU, D_CPU)
    for world in range(W_CPU):
        if restore_mask[world] == 0:
            np.testing.assert_array_equal(restored_q[world], 0.0)
            continue
        expected = bank_q[1, slot[world]].copy()
        expected[ROOT_Q : ROOT_Q + 3] += offsets[world]  # the target world's own object offset
        np.testing.assert_allclose(restored_q[world], expected, rtol=0, atol=1e-4)
        np.testing.assert_array_equal(restored_qd[world], bank_qd[1, slot[world]])

    # Tags above the bin's current top are not served: the lanes fall back to reference starts in bin 1.
    policy.top.assign(np.full(5, 0.5, dtype=np.float32))
    policy.sample(mask, wp.zeros(1, dtype=wp.float32, device="cpu"))
    np.testing.assert_array_equal(policy.kf_slot.numpy(), -1)
    frames = policy.reset_frame.numpy()
    assert ((frames >= 25) & (frames < 50)).all()
    np.testing.assert_array_equal(policy.steps_since_reset.numpy(), 0)
    np.testing.assert_array_equal(policy.clean.numpy(), 0)
    np.testing.assert_allclose(policy.reference_joint_q_offset.numpy().reshape(W_CPU, Q_CPU)[:, 3], 0.2)
    np.testing.assert_array_equal(policy.served.numpy(), [0, W_CPU, 0, 0, 0])  # unchanged
    np.testing.assert_allclose(policy.tag.numpy(), 0.5)


def test_explicit_reset_lanes_are_not_scored_or_captured():
    import warp as wp

    policy, env, state, _ = _cpu_setup()
    mask = wp.ones(W_CPU, dtype=wp.int32, device="cpu")
    policy.prepare_explicit(mask, wp.array([0.0], dtype=wp.float32, device="cpu"))
    np.testing.assert_array_equal(policy.active.numpy(), 0)
    np.testing.assert_array_equal(policy.clean.numpy(), 0)
    np.testing.assert_array_equal(policy.kf_slot.numpy(), -1)
    np.testing.assert_array_equal(policy.steps_since_reset.numpy(), 51)
    _step(policy, env, state, step=20, timestep=40)
    _step(policy, env, state, step=20, timestep=41, terminated=1)
    np.testing.assert_array_equal(policy.pend_on.numpy(), 0)
    np.testing.assert_array_equal(policy.n_iter.numpy(), 0)


def test_sidecar_round_trip(tmp_path):
    policy, env, state, _ = _cpu_setup()
    _fill_bin_one(policy, env, state)
    policy.host_update(0)
    policy.bins.top[2] = 0.3
    policy.bins.rate[4] = 0.25
    _set(policy.active, np.ones(W_CPU))
    _step(policy, env, state, step=16, timestep=60, terminated=1)  # pending outcome counts at save time
    assert int(policy.n_iter.numpy().sum()) == W_CPU and int(policy.f_iter.numpy().sum()) == W_CPU
    path = tmp_path / "rsi_7.safetensors"
    policy.save_state(path)
    assert not path.with_name(".rsi_7.safetensors.tmp").exists()

    other, other_env, _, _ = _cpu_setup()
    other.load_state(path)
    for key, value in policy.bins.state_dict().items():
        np.testing.assert_array_equal(other.bins.state_dict()[key], value)
    for key, value in policy.bank.state_dict().items():
        np.testing.assert_array_equal(other.bank.state_dict()[key], value)
    np.testing.assert_array_equal(other.n_iter.numpy(), policy.n_iter.numpy())
    np.testing.assert_array_equal(other.f_iter.numpy(), policy.f_iter.numpy())
    np.testing.assert_array_equal(other.top.numpy(), policy.bins.top.astype(np.float32))
    np.testing.assert_array_equal(other.cdf.numpy(), policy.bins.start_cdf())

    renamed = tmp_path / "rsi_8.safetensors"
    renamed.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="iteration"):
        other.load_state(renamed)
    mismatched, _, _, _ = _cpu_setup(kf_capacity=8)
    with pytest.raises(ValueError, match="capacity"):
        mismatched.load_state(path)
    with pytest.raises(FileNotFoundError):
        other.load_state(tmp_path / "rsi_9.safetensors")


# ====================================================================== GPU environment tests
def _rsi_env_class():
    """The production RLEnv: it detects an RSI reset policy and wires the hooks itself (envs/rl.py)."""
    from flash_chord.envs.rl import RLEnv

    return RLEnv


def _support_overrides() -> list[str]:
    """scripts/fc_train.sh rule: no support when the stage file is missing or holds no Cylinder/Cube."""
    if not SUPPORT.is_file():
        return ["scene.support=false"]
    text = SUPPORT.read_text(errors="ignore")
    return [] if ("def Cylinder" in text or "def Cube" in text) else ["scene.support=false"]


def _build_env(world_count: int, overrides: list[str]):
    from flash_chord.configuration import compose_config, instantiate_typed
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.envs.rl import RLEnvConfig
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.scene.setup import setup_scene

    config = compose_config(
        "train",
        [
            "experiment=g1_recon_body_ppo",
            "reset=recon_body_rsi",
            "curriculum=recon_body_rsi",
            f"task.parquet='{PARQUET}'",
            f"scene.world_count={world_count}",
            "logging.mode=disabled",
            *_support_overrides(),
            *overrides,
        ],
    )
    environment_config = instantiate_typed(config.env, RLEnvConfig)
    embodiment = instantiate_typed(config.embodiment, Embodiment)
    collision = instantiate_typed(config.collision, CollisionPolicy)
    setup = setup_scene(
        parquet=config.task.parquet,
        control_fps=environment_config.sim.fps,
        motion_speed=config.task.motion_speed,
        embodiment=embodiment,
        collision=collision,
        world_count=world_count,
        include_support=config.scene.support,
        decompose_objects=config.scene.decompose_objects,
        object_scale_min=config.scene.object_scale_min,
        object_scale_max=config.scene.object_scale_max,
        object_scale_seed=config.scene.object_scale_seed,
        contact_friction=config.scene.get("contact_friction", 1.0),
        object_free_joint_damping=config.scene.get("object_free_joint_damping", 0.0),
        source_frame_playback=bool(config.task.source_frame_playback),
        motion_start_frame=int(config.task.motion_start_frame),
        motion_end_frame=int(config.task.motion_end_frame),
    )
    env = _rsi_env_class()(setup.scene, setup.reference, config=environment_config)
    assert env._rsi is env.reset_policy, "RLEnv did not wire the RSI hooks"
    return env


requires_smoke_parquet = pytest.mark.skipif(not PARQUET.is_file(), reason=f"smoke parquet {PARQUET} is missing")


@pytest.mark.gpu
@pytest.mark.slow
@requires_smoke_parquet
def test_keyframe_restore_reproduces_the_stored_state():
    """(1) A bank state written from world 3 and forced onto world 5 comes back exactly through
    sample -> BaseEnv.reset -> restore, with the evaluator-exact settling and cleared action history."""
    import newton
    import warp as wp

    with wp.ScopedDevice("cuda:0"):
        env = _build_env(8, [])
        policy = env.reset_policy
        world = env.world_count
        assert env.reference.num_frames == 440 and policy.num_bins == 18 and policy.max_start_frame == 429
        q_per_world, dof_per_world = policy.q_per_world, policy.dof_per_world
        bodies = env.model.body_count // world

        env.reset(frame_id=0)
        action = wp.zeros(env.action.action_dim * world, dtype=wp.float32)
        # Zero actions track the palms poorly (~0.35 m): snapshot inside the 50-step failure mask.
        for _ in range(45):
            env.step(action)
        source, target, slot = 3, 5, 5
        assert int(env.episode_step.numpy()[source]) == 45, "source lane terminated before the snapshot"
        frame = int(env.timestep.numpy()[source])
        assert frame == 45
        joint_q = env.state_0.joint_q.numpy().reshape(world, q_per_world)
        joint_qd = env.state_0.joint_qd.numpy().reshape(world, dof_per_world)
        body_q = env.state_0.body_q.numpy().reshape(world, bodies, 7)
        assert np.abs(joint_qd[source]).max() > 1e-3  # a moving state, not a rest pose
        offsets = env.reset_table.object_root_position_offsets_w.numpy().reshape(world, -1, 3)
        stored_q = joint_q[source].copy()
        for object_id, root in enumerate(env.reset_table.object_root_q_ids.numpy()):
            stored_q[root : root + 3] -= offsets[source, object_id]
        bin_index = int(policy.bins.bin_of(frame))
        assert bin_index == 1
        policy.bank.debug_write(bin_index, slot, stored_q, joint_qd[source], frame, 0.0)
        policy.debug_force_keyframe(target, bin_index, slot)

        mask = np.zeros(world, dtype=np.int32)
        mask[target] = 1
        env.reset(reset_mask=wp.array(mask, dtype=wp.int32))

        q_after = env.state_0.joint_q.numpy().reshape(world, q_per_world)
        qd_after = env.state_0.joint_qd.numpy().reshape(world, dof_per_world)
        body_after = env.state_0.body_q.numpy().reshape(world, bodies, 7)
        expected_q = stored_q.copy()
        for object_id, root in enumerate(env.reset_table.object_root_q_ids.numpy()):
            expected_q[root : root + 3] += offsets[target, object_id]
        np.testing.assert_allclose(q_after[target], expected_q, rtol=0, atol=1e-5)
        np.testing.assert_allclose(qd_after[target], joint_qd[source], rtol=0, atol=1e-5)
        others = np.arange(world) != target
        np.testing.assert_array_equal(q_after[others], joint_q[others])
        np.testing.assert_array_equal(qd_after[others], joint_qd[others])
        # Body transforms are FK of the restored coordinates and match the source world's solver output.
        scratch = env.model.state()
        newton.eval_fk(env.model, env.state_0.joint_q, env.state_0.joint_qd, scratch)
        np.testing.assert_allclose(body_after[target], scratch.body_q.numpy().reshape(world, bodies, 7)[target],
                                   rtol=0, atol=1e-6)
        np.testing.assert_allclose(body_after[target], body_q[source], rtol=0, atol=1e-4)
        # Reference time, settling and lineage
        assert int(env.timestep.numpy()[target]) == frame
        assert int(env.episode_start_frame.numpy()[target]) == frame
        assert int(env.episode_step.numpy()[target]) == 0
        assert int(policy.steps_since_reset.numpy()[target]) == 51
        assert int(policy.kf_slot.numpy()[target]) == slot and int(policy.start_bin.numpy()[target]) == bin_index
        assert int(policy.clean.numpy()[target]) == 1 and int(policy.active.numpy()[target]) == 1
        help_value = float(policy.help.numpy()[target])
        assert 0.5 - 1e-6 <= help_value <= 1.0 + 1e-6
        assert float(policy.applied_voc_scale.numpy()[target]) == pytest.approx(help_value)
        np.testing.assert_array_equal(env.reference_joint_q_offset.numpy().reshape(world, -1)[target], 0.0)
        np.testing.assert_array_equal(policy.raw_joint_q_offset.numpy().reshape(world, -1)[target], 0.0)
        # Action state follows stock reset semantics (CH zeroes last_actions after a restore)
        assert int(env.action.history_initialized.numpy()[target]) == 0
        np.testing.assert_array_equal(env.action.actor_action_history.numpy().reshape(world, -1)[target], 0.0)
        assert int(policy.served.numpy()[bin_index]) == 1

        observation, _, _, _ = env.step(action)
        assert np.isfinite(observation.numpy()).all()
        assert int(env.timestep.numpy()[target]) == frame + 1


@pytest.mark.gpu
@pytest.mark.slow
@requires_smoke_parquet
def test_captured_bank_pipeline_and_outcome_counters():
    """(2) + (3) Under capture_step(): pending -> commit only at truncation, terminated lanes drop, the bank fills and
    serves, host updates of top/cdf/frame0_share reach the captured graph, and the outcome counters equal the
    completed (n) and failed (f) episodes seen by the environment."""
    import warp as wp

    overrides = [
        "task.motion_end_frame=80",  # 80-frame clip: F_max 69, bins [0,25) [25,50) [50,70)
        # A zero-action policy misses the palms by ~0.35 m (palm failure right after the 50-step mask), so the palm
        # gates are opened for this pipeline test; pelvis and object gates stay stock (the forced object failure).
        "termination.palm_position_threshold=10.0",
        "termination.palm_orientation_threshold=10.0",
        "reset.kf_prob=1.0",
        "reset.kf_min_task=0.0",
        "reset.kf_step_score_min=0.0",
        "reset.kf_min_fill=2",
        "reset.kf_capacity=8",
        "reset.frame0_share=0.25",
    ]
    with wp.ScopedDevice("cuda:0"):
        env = _build_env(8, overrides)
        policy = env.reset_policy
        world = env.world_count
        assert env.reference.num_frames == 80 and policy.num_bins == 3
        q_per_world = policy.q_per_world
        root = int(env.reset_table.object_root_q_ids.numpy()[0])
        env.reset()
        env.capture_step()
        env.reset()
        action = wp.zeros(env.action.action_dim * world, dtype=wp.float32)

        completed = failed = scored_n = scored_f = 0
        commits = drops = expected_commits = expected_drops = 0
        teleported = None
        served_after_fill = 0
        iteration = 0
        phase_three_start = None
        for step in range(400):
            pend_before = policy.pend_on.numpy().astype(bool)
            writes_before = int(policy.bank.write_count.numpy().sum())
            drops_before = int(policy.bank.drop_total.numpy()[0])
            fill_before = policy.bank.fill()
            served_before = policy.served.numpy().copy()
            if teleported is None and step >= 60:
                candidates = np.flatnonzero(pend_before & (env.episode_step.numpy() > 52))
                if len(candidates):
                    teleported = int(candidates[0])
                    joint_q = env.state_0.joint_q.numpy()
                    joint_q[teleported * q_per_world + root + 2] += 1.0  # lift the object 1 m: object failure
                    env.state_0.joint_q.assign(joint_q)
            env.step(action)
            done = env.done.numpy().astype(bool)
            truncation = env.truncation.numpy().astype(bool)
            completed += int((done | truncation).sum())
            failed += int((done & ~truncation).sum())
            step_commits = int(policy.bank.write_count.numpy().sum()) - writes_before
            step_drops = int(policy.bank.drop_total.numpy()[0]) - drops_before
            commits += step_commits
            drops += step_drops
            # kf_min_task 0: a pending lane commits exactly when it reaches the clip end without failing
            assert step_commits == int((pend_before & truncation & ~done).sum()), step
            assert step_drops == int((pend_before & done).sum()), step
            if teleported is not None and step_drops and pend_before[teleported] and done[teleported]:
                teleported = -1  # the forced failure dropped its pending state
            if (fill_before >= 2).any():
                served_after_fill += int((policy.served.numpy() - served_before)[fill_before >= 2].sum())
            if (step + 1) % 24 == 0 and phase_three_start is None:
                scored_n += int(policy.n_iter.numpy().sum())
                scored_f += int(policy.f_iter.numpy().sum())
                metrics = policy.host_update(iteration)
                iteration += 1
                assert np.isfinite(list(metrics.values())).all()
            if step == 300:
                # (R1) per-iteration values live in device arrays, so the captured graph sees host updates.
                scored_n += int(policy.n_iter.numpy().sum())
                scored_f += int(policy.f_iter.numpy().sum())
                policy.host_update(iteration)
                policy.bins.top[:] = 0.5
                policy.upload_bins()
                policy.set_frame0_share(0.0)
                phase_three_start = (policy.reset_count.numpy().copy(), policy.start_stats.numpy().copy())

        scored_n += int(policy.n_iter.numpy().sum())
        scored_f += int(policy.f_iter.numpy().sum())
        print(f"completed={completed} failed={failed} commits={commits} drops={drops} "
              f"served_after_fill={served_after_fill} fill={policy.bank.fill().tolist()}")
        assert completed > 0 and scored_n == completed and scored_f == failed
        assert commits > 0 and int(policy.bank.write_count.numpy().sum()) == commits
        assert teleported == -1, "the forced object failure should have dropped a pending keyframe"
        assert drops >= 1
        assert served_after_fill > 0
        frames = policy.bank.frame.numpy().reshape(3, 8)
        fill = policy.bank.fill()
        for b in range(3):
            stored = frames[b, : fill[b]]
            assert ((stored >= policy.bins.bin_lo[b]) & (stored < policy.bins.bin_hi[b])).all()
        assert (policy.bank.tag.numpy().reshape(3, 8)[fill[:, None] > np.arange(8)[None, :]] == 1.0).all()
        reset_before, stats_before = phase_three_start
        restarted = policy.reset_count.numpy() > reset_before
        assert restarted.any()
        np.testing.assert_allclose(policy.tag.numpy()[restarted], 0.5)
        assert (policy.help.numpy()[restarted] <= 0.5 + 1e-6).all()
        assert int(policy.start_stats.numpy()[3]) == int(stats_before[3])  # no new anchors
        assert int(policy.start_stats.numpy()[2]) == int(stats_before[2])  # tags 1.0 > top 0.5: nothing served
