# t2_consensus_r6_20261031: DRYRUN_R6 (records of the pre-launch steps; decides nothing)

Written 2026-10-31 ~01:32 CST, after the steps below. `runs_s/` and `runs_f/` do not exist: there is no round-6 run or
try, no rollout list, no consensus score and no selection. Phase S was NOT
launched (PREREG_R6 3.6). Documents in force: `PREREG_R6.md` 0b0fbf9a2b581913..., `PREREG_R6_PINS.md` a9c36b1f0d2a7de3....

## 1. Plan-only dry runs: both `plan: OK`

| phase | file | sha256 | result |
|---|---|---|---|
| S | `dryrun/r6_eval_plan_screen_20261031_012345.txt` | c9d5361a48206d25... | 250/250 jobs ok, hulls ok 250 (62 parquets) |
| F | `dryrun/r6_eval_plan_final_20261031_012517.txt` | f8ff647143e80d1d... | 30/30 incumbents ok with their earlier exports re-hashed; no finalist yet (SELECTION_S6 absent, as expected); hulls ok 30 |

Both: pin chain held (PREREG_R6 section 11: 47 pins + 3 commits; SNAPSHOT_R6 83 files; TRACKS_R6 107 files; round 1's
chain), preflight passed (kit evaluate_policy.py == stock == 160d24f5; fc_r2 and b39b69b trees at their pinned HEADs,
eval_all.py 657295921e82 in both), kit + stock trees UNCHANGED vs round 1's baseline, seed signature
`delay=0(+g0..4,0-2) reset=42 scale_seed=0(1.0/1.0) wrench=0 vertex=0/0 ppo=42`, GPUs 0, 1, 2, 7 usable, GPUs 3-6 refused
by rule, no external evaluation counted.

## 2. The two dry-run evaluations on GPU 0 (PREREG_R6 3.6): PASS

Both through `scripts/t2cons_r6_eval_launch.sh --phase <p> --dry-run-eval --gpus 0` (driver `t2cons_r6_eval.py` ada6e0f9...,
pinned in PREREG_R6_PINS). Integrity baseline `integrity/baseline.json` = byte copy of round 1's (ef9b5035...); the kit and
stock runtime were unchanged after each evaluation (`integrity/check_*.json`).

| phase | job | what it exercises | export | evaluation | own-input (evaluator) |
|---|---|---|---|---|---|
| S | `ep26_k3ca_25298` (K3CA final, ours) | first K3CA job of these drivers; fc_r2 export | 01:26:15-01:26:45, ok on try 1 at 1e-4, max_rel_diff 2.48e-05, stock verified, source daeee885... == pool file, exported file **eef1f79bc3d5d370fcf899391e80cd75e5ac8ffdd0bb9037d34e3f75e2b1d609** (kept: the job's single export, `exports/ep26_k3ca_25298/`) | 01:26:46-01:27:36, GPU 0, 50 s, status ok; checkpoint digest == the export; reference = the K3CA training parquet (d9a210fa...) | AUC 0.8009, SP-SR 1, MP-SR 1, MPPE 1.32 cm |
| F | `ep26_zx_k3fix_92000` (ep26 incumbent, Zuxing's pro6000 file) | incumbent with its round-1 b39b69b export reused (no new export) | reused: 25c600cb20fbd448... (round 1 `exports/ep26_zx_k3fix_92000`) | 01:28:00-01:28:50, GPU 0, 50 s, status ok; checkpoint digest == the reused export; reference = the K3fix training parquet (01637248...) | AUC 0.9043, SP-SR 1, MP-SR 1, MPPE 1.06 cm |

Lane locks were taken as specified (our `evallock/gpu0.lock` + 10 shared files of GPU 0); nothing was signalled; the
drivers exited by themselves (rc 0) and removed `driver.pid`. GPU 0 hosted two Tier-3 retry trainings throughout.

Records: `dryrun/eval_screen_20261031_012608/` (result.json 511240e7..., driver.log 78e8529b..., seeds_dryrun.tsv
6a9cecb6..., manifest_dryrun.json 37d420da...) and `dryrun/eval_final_20261031_012752/` (result.json f2a79b31...,
driver.log 3acb43c5..., seeds_dryrun.tsv b699286b..., manifest_dryrun.json eaf61c7e...); `exports/ep26_k3ca_25298/export.json`
013121e6....

## 3. Own-input re-score of the two dry-run rollouts (the PREREG_R6 5.3 gate path): PASS

`scripts/t2cons_r6_dryrun_gate.py` (433d505f...; kit Python; loads the pinned round-1 `t2cons_consensus.py` e1b943b1 and
`t2cons_a1_gate.py` a29c997b from their verified bytes) re-scored each rollout against its OWN input reference exactly as
the validation gate will (`rollout_on_grid` + `metrics_vs`, vertex seeds 0..9). Output `dryrun/gate_dryrun_r6.json`
(7a6b6084...).

| job | input reference | evaluator AUC | re-scored seeds 0..9 | band (+-0.02) | MPPE eval / ours | SP, MP equal | conversion residual | gate |
|---|---|---:|---|---|---|---|---|---|
| ep26_k3ca_25298 | `tmp_analysis/tier2/k3ca/frame_refs_k3ca` (round 1 INPUT_REF k3ca) | 0.8009 | 0.8012-0.8104 | 0.7812-0.8304 | 1.321 / 1.321 | yes | 6.2e-08 m, 1.5e-07 rad | PASS |
| ep26_zx_k3fix_92000 | `tmp_analysis/tier2/k3fix_pilot/frame_refs_k3fix` | 0.9043 | 0.9044-0.9112 | 0.8844-0.9312 | 1.061 / 1.061 | yes | 6.2e-08 m, 1.5e-07 rad | PASS |

These values are plumbing evidence only, never a round-6 run, and sections 5-6 of PREREG_R6 never read `dryrun/`.

## 4. Timing seen (for the budget, decides nothing)

Export 30 s (CPU); each evaluation 50 s wall on GPU 0 next to two Tier-3 trainings (round 5 measured 30-41 s on lighter
GPUs; the main session's figure is ~70 s). With 3 evaluations at once (2 on GPUs 0-2, 1 on GPU 7) and the 35 s cooldown:
~50 s -> ~180 evaluations / h; 70 s -> ~137 / h. Phase S (750 evaluations + 250 exports) ~4.2-5.5 h; phase F (300-600)
~1.7-4.4 h; total ~6-10 h.

## 5. Launch (NOT done in this step; needs the main session's go)

From `/root/autodl-tmp/haoran.zhang`: `bash scripts/t2cons_r6_eval_launch.sh --phase screen` (also every restart; first
`rm ROOT/STOP` if present). The start line of `eval_r6_driver.log` must read `allowed (0, 1, 2, 7)`. The K3CA export above
is reused. Stop: `touch ROOT/STOP` (running evaluations finish and are recorded; nothing is signalled). Status:
`conda_envs/flashchord/bin/python -I -B scripts/t2cons_r6_eval.py status --phase screen`. Then step R4
(`scripts/t2cons_r6_consensus.py`, to be written and pinned with its selftest in `PREREG_R6_PINS_b.md` before it scores
anything), then `--phase final`.
