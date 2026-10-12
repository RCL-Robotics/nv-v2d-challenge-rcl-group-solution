# Training package

PPO and FlashSAC are peer learners built on the same Warp/JAX environment boundary.

| Path | Responsibility |
| --- | --- |
| `environment.py` | Algorithm-neutral vector environment and transition types |
| `checkpoint.py` | Safetensors PyTree serialization and restoration |
| `checkpoint_metadata.py` | Saved config and exact policy, object-joint, and critic input descriptions |
| `ppo/` | PPO network, rollout, update, learner, runner, and evaluation |
| `flash_sac/` | FlashSAC network, replay, exploration, update, learner, runner, and evaluation |

Both learners index logs and checkpoints by cumulative control-step environment transitions. Physics substeps do
not advance this counter.

FlashSAC writes independent files:

| File | Contents |
| --- | --- |
| `policy_<steps>.safetensors` | Actor state and required current metadata for evaluation |
| `state_<steps>.safetensors` | Complete learner state for resume |
| `replay_<steps>.safetensors` | Optional replay snapshot |

A state-only resume starts with empty replay unless a matching replay snapshot is explicitly configured. Policy
files are evaluation artifacts and are not resumable training state.

All new checkpoints contain the complete saved configuration and explicit runtime input descriptions. Missing or
incompatible descriptions are rejected, and older configuration layouts are not migrated.

## Non-finite guard (PPO)

One simulated world whose physics solve returns NaN used to poison a whole PPO run: stock FlashCHORD has no finite
check, so the NaN observation entered the running observation normalizer for good and every later update was
skipped, while MuJoCo-Warp's NaN warm start kept that world NaN after every reset. `TrainingConfig.nonfinite_guard`
(default **on** for training; evaluation and export keep stock behavior) handles this:

| Stage | What happens to a bad world-step |
| --- | --- |
| Environment (`runtime/nonfinite.py`, inside the captured step graph) | A world with any non-finite `joint_q`/`joint_qd`/`body_q`/`body_qd` or solver warm-start entry is flagged, terminated (so the normal auto-reset restores it), and gets a zeroed warm start and objective score. |
| Rollout (`ppo/rollout.py`, one fused `guard_step` executable per step) | Flagged steps and steps with a non-finite observation or reward get done = 1 and reward = their own value (zero TD error); non-finite observation rows are zeroed; diagnostics rows are zeroed. |
| Normalizer (`ppo/normalization.py`) | The stock update is used unchanged when every row is valid and finite; otherwise only valid, finite rows enter it. |
| Resume (`checkpoint.require_finite_resume_state`) | Resuming from a checkpoint whose parameters, optimizer state, normalizer or cached observation (or RSI sidecar) hold NaN/Inf raises `NonfiniteCheckpointError`. The guard cannot heal state saved by an already-poisoned run. |

- **Cost and identity:** with all values finite, every guard step is a selection that returns the stock values, so
  rollouts, normalizer, updates and checkpoints are bit-identical to running without the guard (CPU tests), and
  the fused rollout step costs about nothing. GPU training itself is not run-to-run reproducible.
- **Logging:** `guard/nonfinite_world_steps_{per_iteration,total}` and
  `termination/nonfinite_fraction_per_completed_episode` appear only once an event happened, together with a
  `[nonfinite-guard]` console line. The W&B run config gains a `nonfinite_guard` entry.
- **Disable:** `+training.nonfinite_guard=false` (append override: the key is a dataclass default, not a YAML key,
  so `training.nonfinite_guard=false` fails with "Key 'nonfinite_guard' is not in struct"). This also writes the key
  into the checkpoint's resolved config; E2 export removes it.
- **Not covered:** finite but huge values (e.g. 1e18) are not flagged; a world whose observation or reward is
  non-finite but whose simulator state is finite is cleaned in the batch but not reset.

## Early stop on zero help, and `final.json`

`TrainingConfig.early_stop_help_zero_tail` (default **0 = off**) ports Coordinates_hand's stop rule ("curriculum
zero gains for 1500 epochs", rl_games `a2c_common.py`). With a value `T > 0` and the RSI reset, training stops
after the first iteration `N` at which every RSI start bin has had help top 0 **continuously** for `N - S >= T`
iterations, where `S` is the iteration whose bin decision left every bin at 0 (`RSIReconBodyReset.help_zero_since`).
CH counts every all-zero update without resetting the count; the two rules agree once the RSI deadline ramp has
ended, because up-moves are impossible with a zero floor. Other resets raise an error when `T > 0`.

- **At the stop:** `model_N.safetensors` and `rsi_N.safetensors` are saved (even off the save interval), the
  record of iteration `N` is logged with `early_stop/stopped = 1`, a `[early-stop]` console line is printed and
  the process exits normally (exit 0). With the option on, every logged record also has
  `early_stop/help_zero_iterations` (`N - S`, 0 while any bin has help).
- **Resume:** `S` is not stored separately. It is recomputed from the RSI sidecar's `bins.top` and
  `bins.last_move`: while every top is 0, each bin's `last_move` is the iteration it last reached 0 (a bin leaves 0
  only through an up-move, which also sets `last_move`), so `S = max(last_move)` (`RSIBins.zero_since`). A
  resumed run therefore stops at the same iteration as an uninterrupted one, and old sidecars work unchanged.
- **`<output_dir>/final.json`** is written (atomically) whenever a run ends normally, after its final
  checkpoint: `iteration`, `reason` (`help_zero_tail` or `max_iterations`), `checkpoint`, `rsi_state`,
  `help_zero_since`, `help_zero_iterations`, `early_stop_help_zero_tail`, `max_iterations`, `environment_steps`,
  `start_iteration`, `resumed_from`, `wall_seconds` (this process only) and `finished_at`. Queue v2 and the
  round-2 evaluation read the final iteration from it. A resumed run that will train further first renames an
  existing `final.json` to `final.prev.json`.

## Observation-normalizer count

The normalizer's `count` is int32. At 4096 worlds x 24 steps it grows by 98,304 per iteration and would wrap
around during iteration 21,846, turning the merge rate negative. It now saturates at 2**31 - 1
(`normalization.saturating_count`): bit-identical below that, same dtype (old checkpoints load), and beyond it the
update keeps merging each batch with the fixed weight `batch / (2**31 - 1)` (a valid weighted merge of the two
populations, see the docstring), so the statistics keep tracking slowly instead of being corrupted.

Run the focused tests with:

```bash
pytest tests/training tests/test_configuration.py -q
```

## Acknowledgements

### FlashSAC

Our JAX implementation is ported from [Holiday Robotics' FlashSAC](https://github.com/Holiday-Robot/FlashSAC),
by [Kim et al. (2026)](https://arxiv.org/abs/2604.04539). See the [upstream license notice](flash_sac/NOTICE).

```bibtex
@article{kim2026flashsac,
  title={{FlashSAC}: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control},
  author={Kim, Donghu and Lee, Youngdo and Park, Minho and Kim, Kinam and
          Nahendra, I Made Aswin and Seno, Takuma and Min, Sehee and Palenicek, Daniel and
          Vogt, Florian and Kragic, Danica and Peters, Jan and Choo, Jaegul and Lee, Hojoon},
  journal={arXiv preprint arXiv:2604.04539},
  year={2026},
  url={https://arxiv.org/abs/2604.04539}
}
```

### PPO

Our PPO implementation follows [Schulman et al. (2017)](https://arxiv.org/abs/1707.06347).

```bibtex
@article{schulman2017proximal,
  title={Proximal Policy Optimization Algorithms},
  author={Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal={arXiv preprint arXiv:1707.06347},
  year={2017},
  url={https://arxiv.org/abs/1707.06347}
}
```
