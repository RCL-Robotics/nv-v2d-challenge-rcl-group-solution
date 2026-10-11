# SPDX-License-Identifier: Apache-2.0
"""RSI v3 (fixed) start bins: geometry, ports of Coordinates_hand ``dexmachina/tests/test_rsi.py``, CH parity on a
recorded outcome stream, state round trip, and CPU-device checks of the start-sampling kernel."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest

from flash_chord.lifecycle.rsi.bins import RSIBinParams, RSIBins, bin_geometry

FIXTURE = Path(__file__).parent / "fixtures" / "ch_rsi_parity.json.gz"


def _bins(num_frames: int = 440, **overrides) -> RSIBins:
    return RSIBins(num_frames, RSIBinParams(**overrides))


# ---------------------------------------------------------------- (1) geometry
def test_geometry_440_and_1003_frames():
    lo, hi, f_max = bin_geometry(440, 25, 10)
    assert len(lo) == 18 and f_max == 429
    assert lo.tolist() == [25 * b for b in range(18)]
    assert hi[:-1].tolist() == [25 * (b + 1) for b in range(17)] and hi[-1] == 430 and hi[-1] - lo[-1] == 5
    lo, hi, f_max = bin_geometry(1003, 25, 10)
    assert len(lo) == 40 and f_max == 992 and hi[-1] == 993 and hi[-1] - lo[-1] == 18
    bins = _bins(440)
    assert bins.num_bins == 18 and bins.max_start_frame == 429
    np.testing.assert_array_equal(bins.bin_of(np.array([0, 24, 25, 424, 425, 429, 439])), [0, 0, 1, 16, 17, 17, 17])
    # CH: bucketize(frame, bin_hi, right=True) clamped to the last bin
    frames = np.arange(440)
    ch = np.minimum(np.searchsorted(bins.bin_hi, frames, side="right"), bins.num_bins - 1)
    np.testing.assert_array_equal(bins.bin_of(frames), ch)


def test_geometry_rejects_single_bin_and_bad_horizon():
    with pytest.raises(ValueError):
        bin_geometry(30, 25, 10)  # 20 startable frames -> one bin
    with pytest.raises(ValueError):
        bin_geometry(440, 25, 440)
    with pytest.raises(ValueError):
        RSIBinParams(bin_frames=0)
    with pytest.raises(ValueError):
        RSIBinParams(success_down=0.8, success_up=0.8)


# ---------------------------------------------------------------- host simulation (CH tests/test_rsi.py:50-66)
def _simulate(bins: RSIBins, fail_prob, updates: int, num_envs: int = 2000, seed: int = 0, finish_share=1 / 12):
    """fail_prob(bin_index, help) -> per-episode failure probability; returns [(t, top_max, top_min, moved, zero)]."""
    rng = np.random.default_rng(seed)
    band = bins.params.band
    lane_bin = rng.choice(bins.num_bins, size=num_envs, p=bins.start_probs())
    lane_help = bins.top[lane_bin] * (1.0 / band + rng.random(num_envs) * (1.0 - 1.0 / band))
    trace = []
    for t in range(1, updates + 1):
        done = np.flatnonzero(rng.random(num_envs) < finish_share)
        n = np.bincount(lane_bin[done], minlength=bins.num_bins)
        failed = rng.random(len(done)) < fail_prob(lane_bin[done], lane_help[done])
        f = np.bincount(lane_bin[done], weights=failed.astype(np.float64), minlength=bins.num_bins)
        bins.observe(n, f)
        new = rng.choice(bins.num_bins, size=len(done), p=bins.start_probs())
        lane_bin[done] = new
        lane_help[done] = bins.top[new] * (1.0 / band + rng.random(len(done)) * (1.0 - 1.0 / band))
        decision = bins.decide(t)
        trace.append((t, float(bins.top.max()), float(bins.top.min()), decision["moved"], decision["zero"]))
    return trace


def test_initial_state_and_floor_start_at_top0():
    bins = _bins()
    assert (bins.top == 1.0).all() and bins.help_progress() == 1.0
    assert bins.floor(0) == 1.0
    np.testing.assert_allclose(bins.start_probs(), 1.0 / 18)
    assert bins.start_cdf()[-1] == np.float32(1.0)
    # floor: geometric to zero_below at the deadline, then a linear ramp to exactly 0
    assert math.isclose(bins.floor(11000), 6.25e-4)
    assert math.isclose(bins.floor(11500), 6.25e-4 / 2)
    assert bins.floor(12000) == 0.0 and bins.floor(15000) == 0.0
    decision = bins.decide(0)
    assert not decision["moved"] and bins.counts["forced"] == 0


def test_always_succeeding_policy_zeroes_every_bin_before_deadline():
    bins = _bins()
    trace = _simulate(bins, lambda b, h: np.zeros(len(b)), updates=4000)
    zero_at = next(t for t, mx, mn, moved, zero in trace if zero)
    assert zero_at < 11000, zero_at
    # 18 bins x ~70 steps of 0.9 with min_gap 40 after the 100-iteration wait
    assert 2800 < zero_at < 3300, zero_at
    assert bins.counts["forced"] == 0 and bins.counts["up"] == 0
    assert bins.help_progress() == 0.0


def test_hard_bin_keeps_assistance_longer_and_gets_sampled_more():
    bins = _bins()
    hard = 8
    _simulate(bins, lambda b, h: np.where(b == hard, 0.9, 0.0), updates=4000, seed=1)
    others = np.delete(np.arange(bins.num_bins), hard)
    assert bins.top[hard] > bins.top[others].max()
    assert bins.rate[hard] > 0.5
    probs = bins.start_probs()
    uniform = 1.0 / bins.num_bins
    assert probs[hard] > 2 * uniform
    assert probs[hard - 1] > uniform  # backward credit
    assert probs.min() >= bins.params.uniform_mix / bins.num_bins - 1e-12


def test_deadline_floor_forces_zero_even_when_failing():
    bins = _bins(deadline=400, zero_ramp=50)
    trace = _simulate(bins, lambda b, h: np.full(len(b), 0.95), updates=600, seed=2)
    assert (bins.top == 0).all()
    zero_at = next(t for t, mx, mn, moved, zero in trace if zero)
    assert zero_at <= 400 + 50 + 1
    assert bins.counts["forced"] > 0


def test_raise_is_blocked_above_the_floor():
    bins = _bins(deadline=300, zero_ramp=50, wait_iterations=0)
    _simulate(bins, lambda b, h: np.full(len(b), 0.95), updates=299, seed=3)
    assert bins.top.max() <= bins.floor(299) / bins.params.step + 1e-9
    assert bins.counts["up"] == 0


def test_raise_happens_below_the_floor_and_from_zero():
    bins = _bins(wait_iterations=0, min_gap=1, min_count=1)
    bins.top[:] = 0.0
    bins.top[1] = 0.01
    n = np.full(bins.num_bins, 10)
    f = np.full(bins.num_bins, 9)  # success 0.1 everywhere
    bins.observe(n, f)
    bins.decide(10)  # floor(10) ~ 0.9935: every raise is allowed
    assert bins.top[0] == bins.params.zero_below  # zero -> zero_below
    assert bins.top[1] == pytest.approx(0.01 / 0.9)
    assert bins.counts["up"] == bins.num_bins


def test_help_progress_runs_from_one_to_zero_on_the_log_scale():
    bins = _bins()
    assert bins.help_progress() == 1.0
    low = bins.params.zero_below
    bins.top[:] = low
    assert abs(bins.help_progress()) < 1e-12
    bins.top[:] = math.sqrt(1.0 * low)
    assert abs(bins.help_progress() - 0.5) < 1e-9
    bins.top[:] = 1.0
    bins.top[: bins.num_bins // 2] = 0.0
    assert abs(bins.help_progress() - (bins.num_bins - bins.num_bins // 2) / bins.num_bins) < 1e-9


# ---------------------------------------------------------------- (3) CH parity
def _fixture():
    if not FIXTURE.is_file():
        pytest.skip(f"missing {FIXTURE}; regenerate it with tests/rsi/make_ch_fixture.py")
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as stream:
        return json.load(stream)


@pytest.mark.parametrize("index", range(4))
def test_ch_parity_on_recorded_outcome_stream(index):
    """Fed CH's own per-iteration (n_b, f_b), RSIBins reproduces CH's tops, floors, moves, decisions, windows and
    move times exactly (float64), and the failure EMAs / start probabilities to rounding level: CH evaluates
    ``(1 - alpha) ** n`` and ``credit.sum()`` with torch kernels whose rounding / summation order differs from
    NumPy's (measured on this fixture: rates within 3.3e-16 absolute, probabilities within 5.1e-16 relative)."""
    scenario = _fixture()["scenarios"][index]
    bins = RSIBins(scenario["num_frames"], RSIBinParams(**scenario["params"]))
    assert bins.num_bins == scenario["num_bins"] == 20
    assert bins.bin_lo.tolist() == scenario["bin_lo"] and bins.bin_hi.tolist() == scenario["bin_hi"]
    np.testing.assert_array_equal(bins.top, scenario["initial_top"])
    np.testing.assert_array_equal(bins.start_probs(), scenario["initial_probs"])
    top_changes = {t: np.asarray(top) for t, top in scenario["top_changes"]}
    probs = {t: np.asarray(p) for t, p in scenario["probs"]}
    rates = {t: np.asarray(r) for t, r in scenario["rate"]}
    expected_top = np.asarray(scenario["initial_top"])
    for i, (n, f) in enumerate(zip(scenario["n"], scenario["f"], strict=True)):
        t = i + 1
        bins.observe(n, f)
        decision = bins.decide(t)
        expected_top = top_changes.get(t, expected_top)
        np.testing.assert_array_equal(bins.top, expected_top, err_msg=f"{scenario['name']} iteration {t}")
        assert decision["moved"] == scenario["moved"][i]
        assert decision["zero"] == scenario["zero"][i]
        assert decision["floor"] == scenario["floor"][i]
        if t in probs:
            np.testing.assert_allclose(bins.rate, rates[t], rtol=0.0, atol=1e-15)
            np.testing.assert_allclose(bins.start_probs(), probs[t], rtol=2e-15, atol=0.0)
    final = scenario["final"]
    np.testing.assert_array_equal(bins.top, final["top"])
    np.testing.assert_array_equal(bins.win_n, final["win_n"])
    np.testing.assert_array_equal(bins.win_f, final["win_f"])
    np.testing.assert_array_equal(bins.last_move, final["last_move"])
    assert bins.counts == final["counts"]
    assert bins.help_progress() == final["help_progress"]


def test_ch_parity_fixture_exercises_every_move_type():
    scenarios = {s["name"]: s for s in _fixture()["scenarios"]}
    totals = {key: sum(s["final"]["counts"][key] for s in scenarios.values()) for key in ("down", "up", "forced")}
    assert all(value > 0 for value in totals.values()), totals
    assert any(np.sum(s["f"]) > 0 for s in scenarios.values())


# ---------------------------------------------------------------- (4) state round trip
def test_state_dict_round_trip_continues_identically():
    a = _bins(wait_iterations=0, min_gap=5, min_count=8)
    _simulate(a, lambda b, h: np.where(b % 3 == 0, 0.6, 0.05), updates=60, num_envs=500, seed=7)
    b = _bins(wait_iterations=0, min_gap=5, min_count=8)
    b.load_state_dict({key: value.copy() for key, value in a.state_dict().items()})
    for key, value in a.state_dict().items():
        np.testing.assert_array_equal(value, b.state_dict()[key])
    rng = np.random.default_rng(11)
    for t in range(61, 120):
        n = rng.integers(0, 12, size=a.num_bins)
        f = np.minimum(rng.integers(0, 12, size=a.num_bins), n)
        for bins in (a, b):
            bins.observe(n, f)
            bins.decide(t)
        np.testing.assert_array_equal(a.top, b.top)
        np.testing.assert_array_equal(a.start_probs(), b.start_probs())
    with pytest.raises(ValueError):
        b.load_state_dict({key: value for key, value in a.state_dict().items() if key != "rate"})


# ---------------------------------------------------------------- config and composition
def test_config_validation():
    from flash_chord.lifecycle.rsi_reset import RSIReconBodyResetConfig

    config = RSIReconBodyResetConfig()
    assert (config.bin_frames, config.zero_below, config.deadline, config.zero_ramp, config.frame0_share) == (
        25, 6.25e-4, 11000, 1000, 0.25)
    for bad in (
        {"start_frame": 5},
        {"end_frame": 100},
        {"always_reset_to_first_frame": True},
        {"immediate_first_frame_probability": 0.25},
        {"reset_to_first_frame_probability": 0.1},
        {"frame0_share": 1.0},
        {"kf_min_fill": 300},
        {"band": 0.5},
        {"zero_below": 0.0},
    ):
        with pytest.raises(ValueError):
            RSIReconBodyResetConfig(**bad)
    with pytest.raises(NotImplementedError):
        RSIReconBodyResetConfig(kf_to_end=False)
    RSIReconBodyResetConfig(keyframes=False, kf_to_end=False)  # no bank, nothing to confirm


def test_yaml_composes_with_stock_experiment():
    from flash_chord.configuration import compose_config, instantiate_typed
    from flash_chord.envs.rl import RLEnvConfig
    from flash_chord.lifecycle.curriculum import FixedCurriculum
    from flash_chord.lifecycle.reset import ResetPolicy
    from flash_chord.lifecycle.rsi import RSIHooks
    from flash_chord.lifecycle.rsi_reset import RSIReconBodyResetConfig

    stock = compose_config("train", ["experiment=g1_recon_body_ppo"])
    config = compose_config(
        "train", ["experiment=g1_recon_body_ppo", "reset=recon_body_rsi", "curriculum=recon_body_rsi"]
    )
    environment = instantiate_typed(config.env, RLEnvConfig)
    assert isinstance(environment.reset, RSIReconBodyResetConfig)
    assert environment.reset == RSIReconBodyResetConfig()  # yaml values == dataclass defaults (D5)
    for key, value in stock.reset.items():  # stock fields keep reset/recon_body.yaml values
        if key != "_target_":
            assert config.reset[key] == value, key
    curriculum = instantiate_typed(config.curriculum, FixedCurriculum)
    stock_curriculum = instantiate_typed(stock.curriculum, FixedCurriculum)
    assert curriculum.thresholds == (20000,) and len(curriculum.stages) == 1
    stage = curriculum.stages[0]
    assert stage.voc_scale == 0.0
    assert stage.reset_to_first_frame_probability is None and stage.immediate_first_frame_probability is None
    assert dict(stage.objective_weights) == dict(stock_curriculum.stages[0].objective_weights)
    policy = environment.reset.build(4, 440, device="cpu")
    assert isinstance(policy, ResetPolicy) and isinstance(policy, RSIHooks)
    assert not hasattr(policy, "set_reset_to_first_frame_probability")
    assert not hasattr(policy, "set_immediate_first_frame_probability")


# ---------------------------------------------------------------- start-sampling kernel on the CPU device
Q_TEST = 12
LEFT_SHOULDER, RIGHT_SHOULDER, FINGERS = 3, 4, (5, 6)


def _cpu_policy(world_count: int, num_frames: int = 440, **overrides):
    import warp as wp

    from flash_chord.lifecycle.rsi_reset import RSIReconBodyResetConfig

    policy = RSIReconBodyResetConfig(**overrides).build(world_count, num_frames, device="cpu")
    reference = np.random.default_rng(0).uniform(0.1, 0.9, size=(num_frames, 10)).astype(np.float32)
    offset = wp.zeros(world_count * Q_TEST, dtype=wp.float32, device="cpu")
    policy.bind_arrays(
        q_per_world=Q_TEST,
        left_shoulder_q_id=LEFT_SHOULDER,
        right_shoulder_q_id=RIGHT_SHOULDER,
        reference_joint_q=reference,
        finger_q_ids=FINGERS,
        output=offset,
    )
    return policy, reference, offset


def _sample_all(policy):
    import warp as wp

    mask = wp.ones(policy.world_count, dtype=wp.int32, device="cpu")
    policy.sample(mask, wp.zeros(1, dtype=wp.float32, device="cpu"))
    return {
        "frame": policy.reset_frame.numpy().copy(),
        "bin": policy.start_bin.numpy().copy(),
        "help": policy.help.numpy().copy(),
        "tag": policy.tag.numpy().copy(),
        "applied": policy.applied_voc_scale.numpy().copy(),
        "age": policy.steps_since_reset.numpy().copy(),
        "slot": policy.kf_slot.numpy().copy(),
        "clean": policy.clean.numpy().copy(),
        "active": policy.active.numpy().copy(),
        "stats": policy.start_stats.numpy().copy(),
    }


def test_initial_assignment_band_and_settling():
    policy, reference, offset = _cpu_policy(500)
    s = _sample_all(policy)
    assert (s["help"] >= 0.5 - 1e-6).all() and (s["help"] <= 1.0 + 1e-6).all()  # [top0/band, top0]
    np.testing.assert_array_equal(s["tag"], 1.0)
    assert (s["active"] == 1).all()  # deviation from CH: initial episodes are scored too
    assert (s["slot"] == -1).all()  # empty bank
    immediate = s["frame"] == 0
    np.testing.assert_array_equal(s["clean"], immediate.astype(np.int32))
    np.testing.assert_array_equal(s["age"][immediate], 51)
    np.testing.assert_allclose(s["applied"][immediate], s["help"][immediate])
    np.testing.assert_array_equal(s["age"][~immediate], 0)
    np.testing.assert_array_equal(s["applied"][~immediate], 1.0)  # reset_voc_scale during the freeze
    offsets = offset.numpy().reshape(500, Q_TEST)
    np.testing.assert_array_equal(offsets[immediate], 0.0)
    ref = ~immediate
    np.testing.assert_allclose(offsets[ref, LEFT_SHOULDER], 0.2)
    np.testing.assert_allclose(offsets[ref, RIGHT_SHOULDER], -0.2)
    for finger in FINGERS:
        np.testing.assert_allclose(offsets[ref, finger], -reference[s["frame"][ref], finger])
    assert int(s["stats"][0]) == 500


def test_starts_cover_the_clip_and_respect_min_horizon():
    policy, _, _ = _cpu_policy(4000, keyframes=False, frame0_share=0.0)
    s = _sample_all(policy)
    assert s["frame"].max() <= policy.max_start_frame == 429
    counts = np.bincount(s["bin"], minlength=policy.num_bins)
    assert (counts > 0).all(), "uniform prior should populate every bin"
    lo, hi = policy.bins.bin_lo[s["bin"]], policy.bins.bin_hi[s["bin"]]
    assert (s["frame"] >= lo).all() and (s["frame"] < hi).all()
    keyed, _, _ = _cpu_policy(4000, keyframes=True, frame0_share=0.0)
    k = _sample_all(keyed)
    assert (k["frame"][k["bin"] == 0] == 0).all()  # with keyframes, bin 0 starts at the evaluator's frame 0
    assert (k["frame"][k["bin"] > 0] >= 25).all()


def test_gains_track_the_bin_top_and_band():
    policy, _, _ = _cpu_policy(1000, frame0_share=0.0)
    policy.bins.top[:] = 0.0
    policy.bins.top[3] = 0.1
    policy.upload_bins()
    s = _sample_all(policy)
    in3 = s["bin"] == 3
    assert in3.any()
    np.testing.assert_array_equal(s["help"][~in3], 0.0)
    assert (s["help"][in3] >= 0.05 - 1e-7).all() and (s["help"][in3] <= 0.1 + 1e-7).all()
    np.testing.assert_allclose(s["tag"][in3], 0.1)


def test_start_sampling_follows_the_failure_weighted_cdf():
    policy, _, _ = _cpu_policy(20000, frame0_share=0.0)
    policy.bins.rate[:] = 0.0
    policy.bins.rate[10] = 0.8
    policy.upload_bins()
    s = _sample_all(policy)
    observed = np.bincount(s["bin"], minlength=policy.num_bins) / 20000
    np.testing.assert_allclose(observed, policy.bins.start_probs(), atol=0.012)


@pytest.mark.parametrize("keyframes", [True, False])
def test_frame0_share_anchors_starts_at_the_evaluation_start(keyframes):
    policy, _, _ = _cpu_policy(20000, keyframes=keyframes, frame0_share=0.25)
    s = _sample_all(policy)
    p0 = policy.bins.start_probs()[0]
    natural = p0 if keyframes else p0 / 25  # bin 0 is frame 0 with keyframes, else one of its 25 frames
    share = float((s["frame"] == 0).mean())
    assert abs(share - (0.25 + 0.75 * natural)) < 0.015
    assert (s["bin"][s["frame"] == 0] == 0).all()
    anchors = int(s["stats"][3])
    assert abs(anchors / 20000 - 0.25) < 0.015
    off, _, _ = _cpu_policy(4000, keyframes=keyframes, frame0_share=0.0)
    assert int(_sample_all(off)["stats"][3]) == 0
    policy.set_frame0_share(0.0)  # device scalar: changes without rebuilding (graph-safe)
    assert int(_sample_all(policy)["stats"][3]) == anchors  # counter not reset by sampling; no new anchors
