# t2_consensus_r6_20261031: PREREG_R6_PINS_b (step R4 scorer; pins and implementation notes only)

**When.** Written 2026-10-31 from about 01:53 CST, while phase S runs (driver pid 393901, launched 01:33:46 with
`bash scripts/t2cons_r6_eval_launch.sh --phase screen`), and BEFORE any round-6 own-input gate score, consensus score or
selection:
- `scores_s6/`, `scores_f6/`, `inputs_r6/ROLLOUTS_S6.sha256`, `final_r6.json` and `SELECTION_S6.sha256` do not exist.
- Runs of the scorer before this file: `selftest` (synthetic values plus the pinned incumbent records) and two trial
  selftests into `dryrun/` (below).
- Round-6 values used when writing this file: the driver's own-input lines in `eval_r6_driver.log` (the evaluator
  values of finished phase-S runs; they enter only the gate and the stability floor, PREREG_R6 5.1).

**Scope.** Pins and implementation notes only (PREREG_R6 10: "PINS addenda may only add pins and notes"). Nothing here
changes a rule of `PREREG_R6.md` (0b0fbf9a2b581913...). This is the addendum PREREG_R6 5.7 names.

## 1. The scorer: `scripts/t2cons_r6_consensus.py` (new file, mode 444)

1. **Pinned code (K1).** It compiles round 2's scorer `scripts/t2cons_r2_consensus.py` (ddee0b8b...) from its verified
   bytes; that module compiles `scripts/t2cons_a1_gate.py` (a29c997b...) from verified bytes, whose module code loads
   `scripts/t2cons_consensus.py` (e1b943b1..., re-checked after loading). The pinned round-1 functions `rollout_on_grid`,
   `metrics_vs`, `load_kit`, `load_refs`, `grid`, `stats`, `floor_count`, `integrity` / `integrity_diff`, `check_pins`,
   `hash_list`, `git_head` and the gate's `auc_seed` are called unchanged; the per-run gate flags are round 2's
   `run_flags` and the score columns round 2's `SCORE_COLS` (as rounds 2-5). Round 5's scorer is not loaded: its module
   chain only adds round-specific bookkeeping that round 6 does not use.
2. **New code (K2)** implements PREREG_R6 5.4-5.6 exactly: `select_s` (phase S), `decide_f` (phase F), `trial_rows`,
   `closest_trial` (R6-CLOSEST) and `replace_verdict`. The selftest (30 tests, PASS) covers: the exact `>=` margins,
   floors 1/3 vs 2/3 and 7/10 vs 8/10, flagged / missing / 2-run candidates, export not ok, the tie order (variance, MPPE,
   name), an independent re-derivation of 5.4 on 800 random episodes, the bar max(m_P_frozen, m_P_fresh) + 0.01 in both
   orders, the incumbent with 7 vs 8 good runs (m_P_fresh = +inf vs the mean), incomplete trials (flagged or missing run),
   R6-CLOSEST and its two tie levels on exact binary values, the paired Delta_B over trials complete for both sets, and
   each of the three replace conditions.
3. **Path guard (K3).** Round 1's tokens + round 2's (readme, train_rl.log) + t2_zx07_20261028, best_weights,
   seed_best, remote_reps, report_r, composites, per_episode_zx. The selftest shows every forbidden pattern refused and
   every rule input of 5.2-5.6 passed.
4. **Evaluator values (K4).** The own-input values used by the gate, the floors and the own-input composites are each
   run's `run.json` metrics (full precision). Each must equal the driver's `own_input_evals_{s6,f6}.csv` row (|diff| <=
   1e-6 for AUC / MPPE; SP-SR / MP-SR exactly); every ok `run.json` must have its CSV row. Round 5 read the CSV, whose
   floats are rounded to about 10 digits; R6-CLOSEST compares 30-episode means, so full precision is used.
5. **m_P_frozen (K5).** PREREG_R6 4 defines m_P_frozen as the float64 mean of the frozen runs, equal to the frozen job
   stats of the selecting round. The scorer computes it with the pinned `stats()` (numpy mean, as rounds 1-5 computed
   m_P) and requires: the frozen runs == the selecting round's frozen scores CSV (pinned), the mean == that round's frozen
   job stats `auc_stats.mean` EXACTLY (the job stats files are the PREREG_R6 11 pins 9e255057 / 88d8e26a / b1e8e4e1), and
   |mean - table| <= 5e-7. The `m_P_frozen` field of `incumbents_r6.json` was computed by the resolver as sum / len and
   differs from the pinned mean in the last bit on ep00, ep19 and ep20 (e.g. ep00 0.7093830334190231 vs
   0.7093830334190232); the resolver itself checked it only to 1e-12. The decision uses the pinned `stats()` value (the
   PREREG_R6 4 definition); the field is kept in the records as `m_P_frozen_field_resolver`. The first trial selftest
   (`dryrun/selftest_trial_r6c_1.json`, script version 27fbf86a, never pinned) failed exactly on this exact-equality
   check; the script was corrected to the rule above before it was pinned (second trial `dryrun/selftest_trial_r6c_2.json`
   and the real selftest PASS). Neither trial computed a round-6 score.
6. **Phase S scoring.** Every ok phase-S run is gated against its own input; a candidate with a flagged run is
   `validation_failed` and is not consensus-scored (as round 5). Phase F: every ok run is gated and consensus-scored;
   flagged runs are excluded run by run as PREREG_R6 5.3 / 5.5 / 5.6 say.
7. **Rollout lists.** `check --phase S|F` writes `inputs_r6/ROLLOUTS_{S6,F6}.sha256` once, only when the phase's driver
   heartbeat says `done`, no driver runs, `driver.pid` is gone and the pin chain holds. `gate`, `score` and `select`
   refuse to run without it and re-hash every rollout. These lists are not pinned here (they do not exist yet); each is
   written by this pinned script before the first score of its phase and is listed in `SELECTION_S6.sha256` /
   `SELECTION_R6.sha256`.
8. **Outputs.** As the docstring says; `SELECTION_S6.sha256` and `SELECTION_R6.sha256` are written last, with paths in
   the driver's `_pin_path` spelling (relative to the round folder; `scripts/...` relative to ME), so the driver's
   phase-F check (`final_r6.json` listed with its current sha256) reads them unchanged.
9. **Run.** `conda_envs/v2dkit/bin/python -I -B scripts/t2cons_r6_consensus.py <cmd> --phase S|F` under `nice -n 10`
   with `OMP_NUM_THREADS=2`, CPU only, from a neutral working directory.

## 2. Pins

pin scripts/t2cons_r6_consensus.py 9b65e612fb289cd8e7f7584ffb7c7ccb40e1d2de9235f0221a70a99c11382d31
pin selftest_r6/selftest_r6.json f2292dfda3590907d0441957868e88992412ded219bd8dfbac7044bd0cc14f02
pin dryrun/selftest_trial_r6c_1.json d7c10bbf6a8902c2a3711774c82ef2b01b6c7ba8f20d05fb81ebfd089626ab13
pin dryrun/selftest_trial_r6c_2.json c2e5433081917931b9746cfa88647d92a042abb0b10754306389e94e4dc66e1e
pin scripts/t2cons_r2_consensus.py ddee0b8b6fa25029275b21a3dc82d15c36f4de71c0a61f2a1046869b01ec13aa
pin scripts/t2cons_a1_gate.py a29c997b7271e9299db109e97af5210f528484719210bb58685523b0ae0eb16b
pin scripts/t2cons_consensus.py e1b943b10fd1db13ab6f78e5bde7b3c6e6c48f7daeee6230c32c463d14ac41b4
pin PREREG_R6.md 0b0fbf9a2b581913ae0da7d36ce76aa785e706c132dc496054f2323a80e487e1
pin PREREG_R6_PINS.md a9c36b1f0d2a7de390e1edad36e2a4946515d1edcb0b41d9055c9d77c0aa5d66
pin DRYRUN_R6.md d1e21ebfcac2d49065ff8082295444e7c5e0fa5ea44697a0736f2c2a22f88934
