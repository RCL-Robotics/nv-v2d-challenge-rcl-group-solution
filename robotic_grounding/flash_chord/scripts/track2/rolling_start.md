# Rolling start (arm C3r2-runin)

C3r2 with one change to the RSI v3 reset. A mid-clip reference start holds robot and object for 50 control steps
(`reset_freeze_steps`, with the VOC hold and the termination grace). In C3r2 the reference clock is frozen for the
whole hold, so the policy is released from rest into a reference that is already moving. The rolling start lets the
clock run for the last 20 of those 50 steps, so robot and object already move with the reference when the hold ends.

| Item | Value |
| --- | --- |
| Experiment | `configs/experiment/g1_recon_body_coord_rsi_r2_runin.yaml` |
| Changes from C3r2 | `reset.run_in_steps: 0 -> 20`; `reset.min_horizon: 10 -> 30` (keeps at least 10 frames after the hold) |
| Code | `lifecycle/rsi_reset.py` (`run_in_steps`), `lifecycle/rsi/kernels.py` (`advance_rsi_reset`: the clock runs when `age > reset_freeze_steps - run_in_steps`) |
| Tests | `tests/run_in/test_rsi_run_in.py` |
| Branch | `dev.zuxing.rolling-start` (commit `b39b69b`; the PSI and termination-patience commits below it are inert unless their configs are used) |

Frame-0 and keyframe starts have no hold and are unchanged. Exported (E2) policies are stock policies; the
evaluator never uses the RSI reset.

## Results

Official kit evaluator (`evaluate_policy.py` on the stock runtime 8c778f9): 1 world, frame 0, no VOC help,
deterministic, termination off. C3r2_wide is the round-2 final checkpoint of each episode.

| Episode | C3r2_wide AUC | Rolling start, 44,922 | Rolling start, best checkpoint | SP-SR / MP-SR | MPPE |
| --- | --- | --- | --- | --- | --- |
| 19 `cane_swing_05` | 0.170 | 0.562 | 0.794 (88,000) | 1 / 1 | 0.9 cm |
| 1 `vacuum_left_lean_02` | 0.200 | 0.222 | 0.753 (126,000) | 1 / 1 | 1.8 cm |
| 4 `baseball_bat_rotate_05` | 0.173 | 0.184 | 0.250 (90,000) | 0 / 0 | 57 cm |

- ep19 passes 0.5 around 38,000 iterations and settles at 0.78-0.79 from 76,000.
- ep01 stays at 0.21-0.22 until 64,000, jumps to 0.37 at 66,000 and completes the task from 86,000 (0.65-0.75).
- ep04 still fails: the bat is rotated overhead but slips out of the hands at about 9 s, then the robot falls.

Whole Tier 1 by the kit flow (`run_policy_evaluation.py`, then `pack_submission.py --track track_2_tier1`, local
leaderboard scores, not submitted):

| Checkpoints | AUC | SP-SR | MP-SR | MPPE |
| --- | --- | --- | --- | --- |
| C3r2_wide round 2 | 0.7937 | 0.8667 | 0.8000 | 12.26 cm |
| Same, with episodes 19 (88,000), 4 (90,000) and 1 (108,000) from the rolling start | 0.8676 | 0.9667 | 0.9000 | 2.97 cm |

Episode 3 (`contigo_coffee_mug`, same C3r2_wide checkpoint in both rows) flips between 0.073 and 0.98 from one
evaluator run to the next; it succeeded in the second row. With episode 3 failing, the second row is about 0.836
AUC and 0.933 SP-SR.

Termination patience (palm and object failures terminate only after 5 consecutive failing steps) did not help on
its own, and combined with the rolling start it slowed ep19 down (0.18 at 44,000 against 0.50-0.58).

## Recipe

Train 90,000 iterations with the help-zero early stop off. The RSI help reaches zero at 39,062 iterations
(deadline 35,156 plus zero ramp 3,906) whatever the length, and the C3r2 early stop then ends the run at about 44,922;
ep19 and ep01 kept improving long after that. Our runs were trained in two pieces (to 44,922 with the early stop,
then resumed to 90,000 without it); one 90,000-iteration run has the same schedule. Pick each episode's checkpoint
by the official evaluation, not by the last iteration.

```bash
# on pro6000, from a checkout of dev.zuxing.rolling-start (sources fc_env.sh itself)
bash robotic_grounding/flash_chord/scripts/track2/train_rolling_start.sh 19,4,1 4 \
    /root/autodl-tmp/haoran.zhang/fc_runs/C3r2_runin --jobs-per-gpu 3
# all 30 episodes on four GPUs, with W&B
bash robotic_grounding/flash_chord/scripts/track2/train_rolling_start.sh all 0,1,2,3 \
    /root/autodl-tmp/haoran.zhang/fc_runs/C3r2_runin --jobs-per-gpu 4 \
    --wandb-mode online --wandb-entity ENTITY --wandb-project PROJECT
# continue finished runs: same out dir, larger ITERATIONS (resumes from model_90000 and rsi_90000)
ITERATIONS=130000 bash robotic_grounding/flash_chord/scripts/track2/train_rolling_start.sh 1 4 \
    /root/autodl-tmp/haoran.zhang/fc_runs/C3r2_runin
```

The script calls `scripts/track2/launch.py` with `--experiment g1_recon_body_coord_rsi_r2_runin`, this checkout as
`--code`, `--iterations ${ITERATIONS:-90000}` and `training.early_stop_help_zero_tail=0`; other `launch.py` options
pass through, and extra Hydra overrides go in `EXTRA` (see the script header). Each run writes
`<out>/ep{NN}_{object}/` with `model_N`, `rsi_N`, `final.json` and `metrics.jsonl`.

Resources measured on pro6000 (RTX PRO 6000, 4,096 worlds): about 16 GB GPU memory and 8 GB host memory per job, at
most 5 jobs per GPU; with 2-3 jobs on one GPU each job runs 1,450-1,900 iterations per hour, so 90,000 iterations
take 2-2.5 days.

## Evaluation

Checkpoints of this arm are coordinated (W0-P): export them first (`python -m
flash_chord.training.ppo.coordinate.export --input model_N.safetensors --output <stock>.safetensors --tolerance 1e-4`),
or use `scripts/track2/eval_all.py`, which exports and runs the kit evaluator. A few long-run checkpoints exceed even
the 1e-4 export tolerance (one ep01 checkpoint: 1.27e-4); take the neighbouring checkpoint.
