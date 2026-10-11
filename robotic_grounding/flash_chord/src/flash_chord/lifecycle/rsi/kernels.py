# SPDX-License-Identifier: Apache-2.0
"""Warp kernels of the RSI v3 (fixed) reset: start sampling, settling, keyframe capture/commit and restore.

Every kernel reads its per-iteration inputs (``top``, ``cdf``, ``frame0_share``) from device arrays, never from
scalar launch arguments, because the scalars of a captured CUDA graph are frozen at capture time (R1).
RNG streams follow the stock pattern ``wp.rand_init(seed, world + counter * 104729)`` (reset.py:666).
"""

from __future__ import annotations

import warp as wp

# Extra seed offset of the per-step keyframe-capture stream, so it never coincides with the reset stream.
CAPTURE_SEED_OFFSET = wp.constant(7919)

# start_stats slots
START_TOTAL = wp.constant(0)
START_FRAME0 = wp.constant(1)
START_KEYFRAME = wp.constant(2)
START_ANCHOR = wp.constant(3)
START_STATS_SIZE = 4


@wp.func
def rsi_bin_of(frame: int, bin_frames: int, num_bins: int) -> int:
    """``min(frame // bin_frames, B - 1)`` (CH ``bucketize(bin_hi, right=True)`` clamped to the last bin)."""
    b = frame // bin_frames
    if b > num_bins - 1:
        b = num_bins - 1
    if b < 0:
        b = 0
    return b


@wp.kernel
def sample_rsi_reset(
    reset_mask: wp.array(dtype=wp.int32),
    reference_joint_q: wp.array(dtype=wp.float32, ndim=2),
    q_per_world: int,
    freeze_steps: int,
    reset_voc_scale: float,
    shoulder_spread: float,
    left_shoulder_q_id: int,
    right_shoulder_q_id: int,
    finger_q_ids: wp.array(dtype=wp.int32),
    seed: int,
    frame0_share: wp.array(dtype=wp.float32),
    cdf: wp.array(dtype=wp.float32),
    top: wp.array(dtype=wp.float32),
    bin_lo: wp.array(dtype=wp.int32),
    bin_hi: wp.array(dtype=wp.int32),
    num_bins: int,
    max_start_frame: int,
    inv_band: float,
    keyframes: int,
    capacity: int,
    min_fill: int,
    bank_frame: wp.array(dtype=wp.int32),
    bank_tag: wp.array(dtype=wp.float32),
    write_count: wp.array(dtype=wp.int32),
    force_bin: wp.array(dtype=wp.int32),
    force_slot: wp.array(dtype=wp.int32),
    reset_count: wp.array(dtype=wp.int32),
    reset_frame: wp.array(dtype=wp.int32),
    finger_scale: wp.array(dtype=wp.float32),
    steps_since_reset: wp.array(dtype=wp.int32),
    applied_voc_scale: wp.array(dtype=wp.float32),
    raw_joint_q_offset: wp.array(dtype=wp.float32),
    reference_joint_q_offset: wp.array(dtype=wp.float32),
    start_bin: wp.array(dtype=wp.int32),
    kf_slot: wp.array(dtype=wp.int32),
    clean: wp.array(dtype=wp.int32),
    active: wp.array(dtype=wp.int32),
    lane_help: wp.array(dtype=wp.float32),
    tag: wp.array(dtype=wp.float32),
    cum_score: wp.array(dtype=wp.float32),
    pend_on: wp.array(dtype=wp.int32),
    requested: wp.array(dtype=wp.int32),
    served: wp.array(dtype=wp.int32),
    start_stats: wp.array(dtype=wp.int32),
    drop_total: wp.array(dtype=wp.int32),
) -> None:
    """Draw an RSI start (anchor / start bin / frame / keyframe) and its help for every selected world.

    CH curriculum.py:455-483 (start bins, anchors, per-lane help in [top/band, top]) and keyframes.py:137-155
    (``to_end`` sampling: only slots whose stored help tag is <= the bin's current top, and only once a bin has
    ``min_fill`` of them). Replaces stock ``sample_recon_body_reset`` (reset.py:643-709).
    """
    world = wp.tid()
    if reset_mask[world] == 0:
        return

    rng = wp.rand_init(seed, world + reset_count[world] * 104729)
    u_anchor = wp.randf(rng)
    u_bin = wp.randf(rng)
    u_frame = wp.randf(rng)
    u_slot = wp.randf(rng)
    u_help = wp.randf(rng)

    b = int(0)  # noqa: UP018 - Warp requires explicitly typed mutable locals.
    frame = int(0)  # noqa: UP018
    anchor = int(0)  # noqa: UP018
    if u_anchor < frame0_share[0]:
        anchor = 1
    else:
        b = num_bins - 1
        for i in range(num_bins):
            if u_bin < cdf[i]:
                b = i
                break
        lo = bin_lo[b]
        hi = bin_hi[b]
        frame = lo + int(u_frame * float(hi - lo))
        if frame > hi - 1:
            frame = hi - 1
        if keyframes != 0 and b == 0:
            frame = 0  # bin 0 starts at the evaluation's initial state, the root of every keyframe
        if frame > max_start_frame:
            frame = max_start_frame

    slot = int(-1)  # noqa: UP018
    forced_bin = force_bin[world]
    if forced_bin >= 0:
        # Test hook (KeyframeBank users: RSIReconBodyReset.debug_force_keyframe): consume a forced bank slot.
        b = forced_bin
        slot = force_slot[world]
        frame = bank_frame[b * capacity + slot]
        anchor = 0
        force_bin[world] = -1
        wp.atomic_add(requested, b, 1)
        wp.atomic_add(served, b, 1)
    elif keyframes != 0 and b > 0:
        wp.atomic_add(requested, b, 1)
        fill = wp.min(write_count[b], capacity)
        threshold = top[b] + 1.0e-6
        eligible = int(0)  # noqa: UP018
        for s in range(fill):
            if bank_tag[b * capacity + s] <= threshold:
                eligible += 1
        if eligible >= min_fill and eligible > 0:
            k = int(u_slot * float(eligible))
            if k > eligible - 1:
                k = eligible - 1
            seen = int(0)  # noqa: UP018
            for s in range(fill):
                if bank_tag[b * capacity + s] <= threshold:
                    if seen == k:
                        slot = s
                        break
                    seen += 1
            frame = bank_frame[b * capacity + slot]
            wp.atomic_add(served, b, 1)

    immediate = int(0)  # noqa: UP018
    if frame == 0 or slot >= 0:
        immediate = 1

    base = world * q_per_world
    for coordinate in range(q_per_world):
        raw_joint_q_offset[base + coordinate] = 0.0
        reference_joint_q_offset[base + coordinate] = 0.0
    if immediate == 0:
        raw_joint_q_offset[base + left_shoulder_q_id] = shoulder_spread
        raw_joint_q_offset[base + right_shoulder_q_id] = -shoulder_spread
        reference_joint_q_offset[base + left_shoulder_q_id] = shoulder_spread
        reference_joint_q_offset[base + right_shoulder_q_id] = -shoulder_spread
        for finger in range(finger_q_ids.shape[0]):
            q_id = finger_q_ids[finger]
            offset = -reference_joint_q[frame, q_id]
            raw_joint_q_offset[base + q_id] = offset
            reference_joint_q_offset[base + q_id] = offset

    bin_top = top[b]
    drawn_help = bin_top * (inv_band + u_help * (1.0 - inv_band))
    reset_frame[world] = frame
    finger_scale[world] = 1.0
    start_bin[world] = b
    kf_slot[world] = slot
    tag[world] = bin_top
    lane_help[world] = drawn_help
    if immediate != 0:
        steps_since_reset[world] = freeze_steps + 1
        applied_voc_scale[world] = drawn_help
    else:
        steps_since_reset[world] = 0
        applied_voc_scale[world] = reset_voc_scale
    clean[world] = immediate
    active[world] = 1
    cum_score[world] = 0.0
    if pend_on[world] != 0:
        wp.atomic_add(drop_total, 0, 1)  # a pending state its lane can no longer confirm (explicit/initial reset)
    pend_on[world] = 0
    reset_count[world] += 1

    wp.atomic_add(start_stats, START_TOTAL, 1)
    if slot >= 0:
        wp.atomic_add(start_stats, START_KEYFRAME, 1)
    elif frame == 0:
        wp.atomic_add(start_stats, START_FRAME0, 1)
    if anchor != 0:
        wp.atomic_add(start_stats, START_ANCHOR, 1)


@wp.kernel
def prepare_explicit_rsi_reset(
    reset_mask: wp.array(dtype=wp.int32),
    q_per_world: int,
    settled_age: int,
    curriculum_voc_scale: wp.array(dtype=wp.float32),
    finger_scale: wp.array(dtype=wp.float32),
    steps_since_reset: wp.array(dtype=wp.int32),
    applied_voc_scale: wp.array(dtype=wp.float32),
    raw_joint_q_offset: wp.array(dtype=wp.float32),
    reference_joint_q_offset: wp.array(dtype=wp.float32),
    start_bin: wp.array(dtype=wp.int32),
    kf_slot: wp.array(dtype=wp.int32),
    clean: wp.array(dtype=wp.int32),
    active: wp.array(dtype=wp.int32),
    lane_help: wp.array(dtype=wp.float32),
    tag: wp.array(dtype=wp.float32),
    cum_score: wp.array(dtype=wp.float32),
    pend_on: wp.array(dtype=wp.int32),
    force_bin: wp.array(dtype=wp.int32),
    drop_total: wp.array(dtype=wp.int32),
) -> None:
    """Stock explicit-reset semantics (reset.py:713-733): exact pose, no settling, help = curriculum scale.

    The lane is neither scored (``active = 0``) nor a capture source (``clean = 0``) and never restores a keyframe.
    """
    world = wp.tid()
    if reset_mask[world] == 0:
        return
    base = world * q_per_world
    for coordinate in range(q_per_world):
        raw_joint_q_offset[base + coordinate] = 0.0
        reference_joint_q_offset[base + coordinate] = 0.0
    scale = curriculum_voc_scale[0]
    finger_scale[world] = 1.0
    steps_since_reset[world] = settled_age
    applied_voc_scale[world] = scale
    start_bin[world] = 0
    kf_slot[world] = -1
    clean[world] = 0
    active[world] = 0
    lane_help[world] = scale
    tag[world] = scale
    cum_score[world] = 0.0
    if pend_on[world] != 0:
        wp.atomic_add(drop_total, 0, 1)
    pend_on[world] = 0
    force_bin[world] = -1


@wp.kernel
def advance_rsi_reset(
    terminated: wp.array(dtype=wp.int32),
    truncated: wp.array(dtype=wp.int32),
    freeze_steps: int,
    offset_decay_steps: int,
    voc_decay_steps: int,
    reset_voc_scale: float,
    q_per_world: int,
    lane_help: wp.array(dtype=wp.float32),
    raw_joint_q_offset: wp.array(dtype=wp.float32),
    steps_since_reset: wp.array(dtype=wp.int32),
    applied_voc_scale: wp.array(dtype=wp.float32),
    reference_joint_q_offset: wp.array(dtype=wp.float32),
    timestep: wp.array(dtype=wp.int32),
    episode_step: wp.array(dtype=wp.int32),
) -> None:
    """Stock ``advance_recon_body_reset`` (reset.py:736-779) with the lane's help in place of the global scale."""
    world = wp.tid()
    if terminated[world] != 0 or truncated[world] != 0:
        return

    age = steps_since_reset[world] + 1
    steps_since_reset[world] = age
    episode_step[world] += 1

    offset_scale = float(0.0)  # noqa: UP018 - Warp requires explicitly typed mutable locals.
    if offset_decay_steps > 0 and age < offset_decay_steps:
        offset_scale = 1.0 - float(age) / float(offset_decay_steps)
    base = world * q_per_world
    for coordinate in range(q_per_world):
        reference_joint_q_offset[base + coordinate] = offset_scale * raw_joint_q_offset[base + coordinate]

    target = lane_help[world]
    decay_start = freeze_steps - voc_decay_steps
    if voc_decay_steps == 0 or age >= freeze_steps:
        applied_voc_scale[world] = target
    elif age <= decay_start:
        applied_voc_scale[world] = reset_voc_scale
    else:
        progress = float(age - decay_start) / float(voc_decay_steps)
        applied_voc_scale[world] = reset_voc_scale + progress * (target - reset_voc_scale)

    if age > freeze_steps:
        timestep[world] += 1


@wp.kernel
def sync_rsi_voc(
    freeze_steps: int,
    lane_help: wp.array(dtype=wp.float32),
    steps_since_reset: wp.array(dtype=wp.int32),
    applied_voc_scale: wp.array(dtype=wp.float32),
) -> None:
    """Stock ``sync_recon_body_voc`` (reset.py:782-791) with the lane's help; the global scale is ignored."""
    world = wp.tid()
    if steps_since_reset[world] >= freeze_steps:
        applied_voc_scale[world] = lane_help[world]


@wp.kernel
def publish_rsi_reset_context(
    applied_voc_scale: wp.array(dtype=wp.float32),
    lane_help: wp.array(dtype=wp.float32),
    steps_since_reset: wp.array(dtype=wp.int32),
    freeze_steps: int,
    context: wp.array(dtype=wp.float32),
) -> None:
    """``RESET_CONTEXT_NAMES`` = (applied VOC, target VOC = lane help, settling progress)."""
    world = wp.tid()
    progress = float(1.0)  # noqa: UP018
    if freeze_steps > 0:
        progress = wp.clamp(float(steps_since_reset[world]) / float(freeze_steps), 0.0, 1.0)
    output = world * 3
    context[output] = applied_voc_scale[world]
    context[output + 1] = lane_help[world]
    context[output + 2] = progress


@wp.kernel
def rsi_transition(
    terminated: wp.array(dtype=wp.int32),
    truncated: wp.array(dtype=wp.int32),
    timestep: wp.array(dtype=wp.int32),
    episode_step: wp.array(dtype=wp.int32),
    object_position_error: wp.array(dtype=wp.float32),
    object_orientation_error: wp.array(dtype=wp.float32),
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
    q_per_world: int,
    dof_per_world: int,
    object_root_q_ids: wp.array(dtype=wp.int32),
    object_root_position_offsets_w: wp.array(dtype=wp.vec3),
    num_objects: int,
    task_pos_scale: float,
    keyframes: int,
    kf_prob: float,
    kf_min_task: float,
    kf_min_stepped: int,
    kf_step_score_min: float,
    max_start_frame: int,
    bin_frames: int,
    num_bins: int,
    capacity: int,
    seed: int,
    active: wp.array(dtype=wp.int32),
    clean: wp.array(dtype=wp.int32),
    start_bin: wp.array(dtype=wp.int32),
    tag: wp.array(dtype=wp.float32),
    cum_score: wp.array(dtype=wp.float32),
    step_count: wp.array(dtype=wp.int32),
    pend_on: wp.array(dtype=wp.int32),
    pend_frame: wp.array(dtype=wp.int32),
    pend_step: wp.array(dtype=wp.int32),
    pend_cum: wp.array(dtype=wp.float32),
    pend_tag: wp.array(dtype=wp.float32),
    pend_q: wp.array(dtype=wp.float32),
    pend_qd: wp.array(dtype=wp.float32),
    n_iter: wp.array(dtype=wp.int32),
    f_iter: wp.array(dtype=wp.int32),
    bank_q: wp.array(dtype=wp.float32),
    bank_qd: wp.array(dtype=wp.float32),
    bank_frame: wp.array(dtype=wp.int32),
    bank_tag: wp.array(dtype=wp.float32),
    write_count: wp.array(dtype=wp.int32),
    commit_total: wp.array(dtype=wp.int32),
    drop_total: wp.array(dtype=wp.int32),
) -> None:
    """Per-step lane bookkeeping after physics/termination/advance and before any auto-reset.

    Order (CH base_env.py:723-731 + keyframes.py:64-123 with ``to_end``): task score, pending resolution
    (commit only on reaching the clip end with a good window), start-bin outcome, then capture.
    """
    world = wp.tid()
    is_terminated = terminated[world] != 0
    is_truncated = truncated[world] != 0
    done = is_terminated or is_truncated

    score = wp.exp(-task_pos_scale * object_position_error[world]) * wp.exp(-object_orientation_error[world])
    cum = cum_score[world] + score
    cum_score[world] = cum
    steps = int(episode_step[world])
    if done:
        steps = steps + 1  # advance skipped this world, so episode_step does not count this transition yet

    if pend_on[world] != 0 and done:
        span = wp.max(steps - pend_step[world], 1)
        window = (cum - pend_cum[world]) / float(span)
        if is_truncated and (not is_terminated) and window >= kf_min_task:
            pb = rsi_bin_of(pend_frame[world], bin_frames, num_bins)
            k = wp.atomic_add(write_count, pb, 1)
            slot = k % capacity
            dst = pb * capacity + slot
            for c in range(q_per_world):
                bank_q[dst * q_per_world + c] = pend_q[world * q_per_world + c]
            for d in range(dof_per_world):
                bank_qd[dst * dof_per_world + d] = pend_qd[world * dof_per_world + d]
            bank_frame[dst] = pend_frame[world]
            bank_tag[dst] = pend_tag[world]
            wp.atomic_add(commit_total, pb, 1)
        else:
            wp.atomic_add(drop_total, 0, 1)
        pend_on[world] = 0

    if done and active[world] != 0:
        b = start_bin[world]
        wp.atomic_add(n_iter, b, 1)
        if is_terminated and not is_truncated:
            wp.atomic_add(f_iter, b, 1)
        active[world] = 0  # scored once; the next start (sample) re-activates the lane

    rng = wp.rand_init(seed + CAPTURE_SEED_OFFSET, world + step_count[world] * 104729)
    step_count[world] += 1
    u = wp.randf(rng)
    if keyframes == 0 or done or clean[world] == 0 or pend_on[world] != 0:
        return
    frame = timestep[world]
    if steps < kf_min_stepped or frame > max_start_frame:
        return
    if score < kf_step_score_min or cum / float(steps) < kf_min_task:
        return
    if rsi_bin_of(frame, bin_frames, num_bins) <= 0 or u >= kf_prob:
        return
    base_q = world * q_per_world
    for c in range(q_per_world):
        pend_q[base_q + c] = joint_q[base_q + c]
    for object_id in range(num_objects):
        root_q = base_q + object_root_q_ids[object_id]
        offset = object_root_position_offsets_w[world * num_objects + object_id]
        pend_q[root_q] = pend_q[root_q] - offset[0]
        pend_q[root_q + 1] = pend_q[root_q + 1] - offset[1]
        pend_q[root_q + 2] = pend_q[root_q + 2] - offset[2]
    base_qd = world * dof_per_world
    for d in range(dof_per_world):
        pend_qd[base_qd + d] = joint_qd[base_qd + d]
    pend_frame[world] = frame
    pend_step[world] = steps
    pend_cum[world] = cum
    pend_tag[world] = tag[world]
    pend_on[world] = 1


@wp.kernel
def restore_keyframe(
    reset_mask: wp.array(dtype=wp.int32),
    kf_slot: wp.array(dtype=wp.int32),
    start_bin: wp.array(dtype=wp.int32),
    capacity: int,
    q_per_world: int,
    dof_per_world: int,
    bank_q: wp.array(dtype=wp.float32),
    bank_qd: wp.array(dtype=wp.float32),
    object_root_q_ids: wp.array(dtype=wp.int32),
    object_root_position_offsets_w: wp.array(dtype=wp.vec3),
    num_objects: int,
    joint_q: wp.array(dtype=wp.float32),
    joint_qd: wp.array(dtype=wp.float32),
) -> None:
    """Overwrite the just-reset keyframe worlds with their stored simulator state (CH keyframes.py:157-177)."""
    world = wp.tid()
    if reset_mask[world] == 0:
        return
    slot = kf_slot[world]
    if slot < 0:
        return
    src = start_bin[world] * capacity + slot
    base_q = world * q_per_world
    for c in range(q_per_world):
        joint_q[base_q + c] = bank_q[src * q_per_world + c]
    for object_id in range(num_objects):
        root_q = base_q + object_root_q_ids[object_id]
        offset = object_root_position_offsets_w[world * num_objects + object_id]
        joint_q[root_q] = joint_q[root_q] + offset[0]
        joint_q[root_q + 1] = joint_q[root_q + 1] + offset[1]
        joint_q[root_q + 2] = joint_q[root_q + 2] + offset[2]
    base_qd = world * dof_per_world
    for d in range(dof_per_world):
        joint_qd[base_qd + d] = bank_qd[src * dof_per_world + d]


@wp.kernel
def count_eligible_slots(
    top: wp.array(dtype=wp.float32),
    bank_tag: wp.array(dtype=wp.float32),
    write_count: wp.array(dtype=wp.int32),
    capacity: int,
    eligible: wp.array(dtype=wp.int32),
) -> None:
    """Per bin: filled slots whose tag is <= the current top (the ``to_end`` serving rule), for logging."""
    b = wp.tid()
    fill = wp.min(write_count[b], capacity)
    threshold = top[b] + 1.0e-6
    count = int(0)  # noqa: UP018
    for s in range(fill):
        if bank_tag[b * capacity + s] <= threshold:
            count += 1
    eligible[b] = count
