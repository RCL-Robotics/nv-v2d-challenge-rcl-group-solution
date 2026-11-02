# t2_consensus_r7_20261102: DRYRUN_R7 (records of the pre-launch steps; decides nothing)

Written 2026-11-02 ~06:42 CST, after the steps below. `runs_s/` and `runs_f/` do not exist: there is no round-7 run or
try, no rollout list, no consensus score and no selection. Phase S was NOT
launched (PREREG_R7 3.6). Documents in force: `PREREG_R7.md` 749649a79ca5616b..., `PREREG_R7_PINS.md` 8378c5a418c9b1ec....

## 1. Plan-only dry runs: both `plan: OK`

| phase | file | sha256 | result |
|---|---|---|---|
| S | `dryrun/r7_eval_plan_screen_20261102_063354.txt` | d45be3b4be236b72... | 154/154 jobs ok (128 relocations re-verified), hulls ok 154 (35 parquets) |
| F | `dryrun/r7_eval_plan_final_20261102_063500.txt` | 011459e12c54b147... | 23 incumbents + 7 references ok with their earlier exports re-hashed, the phase-F dry-run job ok on the finalist path; no finalist yet (SELECTION_S7 absent, as expected); hulls ok 31 |

Both: pin chain held (PREREG_R7 section 11: 74 pins + 3 commits; SNAPSHOT_R7 525 files; TRACKS_R7 131 files;
RELOCATION_R7 130 files; round 1's chain), preflight passed (kit evaluate_policy.py == stock == 160d24f5; fc_r2 and
b39b69b trees at their pinned HEADs, eval_all.py 657295921e82 in both), kit + stock trees UNCHANGED vs round 1's
baseline, seed signature `delay=0(+g0..4,0-2) reset=42 scale_seed=0(1.0/1.0) wrench=0 vertex=0/0 ppo=42`, GPUs 6 and 7
usable, GPUs 0-5 refused by rule, no external evaluation counted.

## 2. The two dry-run evaluations on GPU 6 (PREREG_R7 3.6): PASS

Both through `scripts/t2cons_r7_eval_launch.sh --phase <p> --dry-run-eval --gpus 6` (driver `t2cons_r7_eval.py`
1d66a2db..., pinned in PREREG_R7_PINS). Integrity baseline `integrity/baseline.json` = byte copy of round 1's
(ef9b5035...); the kit and stock runtime were unchanged after each evaluation. GPU 6 hosted Zuxing's ep26 g2-rest
training throughout (plain, no MPS); nothing of his was touched.

| phase | job | what it exercises | export | evaluation | own-input (evaluator) |
|---|---|---|---|---|---|
| S | `ep00_z5k3fix_89500` (Zuxing 5090 runin-k3fix ep00) | first relocated 5090 job: relocation re-verified, b39b69b export of an 82837c8 checkpoint | 06:36:39-06:37:09 (30 s, CPU on the pinned eval-lane cores, CUDA_VISIBLE_DEVICES empty), ok on try 1 at 1e-4, max_rel_diff 2.86e-05, stock verified, source 4b5853b1... == the relocated file of RELOCATION_R7.tsv, exported file **97f8323f8917a55ade419148eaeb91cbe8dfc14cbf93f8d98ac90b1f85229e13** (kept: the job's single export, `exports/ep00_z5k3fix_89500/`) | 06:37:09-06:37:50, GPU 6, 40 s, status ok; checkpoint digest == the export; reference = the K3fix training parquet (98376ade...) | AUC 0.9524, SP-SR 1, MP-SR 1, MPPE 0.62 cm |
| F | `ep01_zxg2_120000` (Zuxing pro6000 runin-g2 ep01, a reused round-6 candidate) | finalist path of a reused candidate: round-6 copy designation (COPIES_r6.tsv), round-6 export reused (no new export) | reused: round 6 `exports/ep01_zxg2_120000` | 06:38:18-06:38:59, GPU 6, 40 s, status ok; checkpoint digest == the reused export; reference = the G2 training parquet (1107c47d...) | AUC 0.2079, SP-SR 0, MP-SR 0, MPPE 41.84 cm |

Lane locks were taken as specified (our `evallock/gpu6.lock` + 14 shared lock files of GPU 6, aliases merged by inode);
nothing was signalled; the drivers exited by themselves (rc 0).

Records: `dryrun/eval_screen_20261102_063628/` (result.json 1411d632..., driver.log 11f0b71f..., seeds_dryrun.tsv
665eb73a..., manifest_dryrun.json 8add70f6...) and `dryrun/eval_final_20261102_063807/` (result.json 496897ab...,
driver.log 32910cd9..., seeds_dryrun.tsv 5a3c062b..., manifest_dryrun.json 990f0e9c...);
`exports/ep00_z5k3fix_89500/export.json` 530072f1....

## 3. Own-input re-score of the two dry-run rollouts (the PREREG_R7 5.1 gate path): PASS

`scripts/t2cons_r7_dryrun_gate.py` (d9aabf1d...; kit Python; loads the pinned round-1 `t2cons_consensus.py` e1b943b1 and
`t2cons_a1_gate.py` a29c997b from their verified bytes) re-scored each rollout against its OWN input reference exactly as
the validation gate will (`rollout_on_grid` + `metrics_vs`, vertex seeds 0..9). Output `dryrun/gate_dryrun_r7.json`
(4ba1cf9d...).

| job | input reference | evaluator AUC | re-scored seeds 0..9 | band (+-0.02) | MPPE eval / ours | SP, MP equal | conversion residual | gate |
|---|---|---:|---|---|---|---|---|---|
| ep00_z5k3fix_89500 | `tmp_analysis/tier2/k3fix_pilot/frame_refs_k3fix` | 0.9524 | 0.9508-0.9547 | 0.9308-0.9747 | 0.623 / 0.623 | yes | 7.6e-08 m, 9.4e-08 rad | PASS |
| ep01_zxg2_120000 | G2 `frame_refs_g2` (round 1 INPUT_REF g2) | 0.2079 | 0.2065-0.2077 | 0.1865-0.2277 | 41.838 / 41.838 | yes | 9.2e-08 m, 1.3e-07 rad | PASS |

These values are plumbing evidence only, never a round-7 run, and sections 5-6 of PREREG_R7 never read `dryrun/`.

## 4. Timing seen (for the budget, decides nothing)

Export 30 s (CPU); each evaluation 40 s on GPU 6 next to one of Zuxing's trainings (zx_t2_20261101_fresh measured median
61 s on GPU 7). With the 10 s cooldown and ~5 s launch overhead: GPU 6 ~55 s, GPU 7 ~75 s per cycle -> ~113 evaluations
per hour on the two lanes. Phase S (462 evaluations + 154 exports overlapping) ~4.1 h; phase F (300-670, expected
~480-520) ~4.3-5.9 h; scoring and the phase-S selection ~0.5 h: **~9-10.5 h** expected, ~11 h worst case (PREREG_R7 8).

## 5. Launch (NOT done in this step)

From `/root/autodl-tmp/haoran.zhang`: `bash scripts/t2cons_r7_eval_launch.sh --phase screen` (also every restart; first
`rm ROOT/STOP` if present). The start line of `eval_r7_driver.log` must read `allowed (6, 7)`. The ep00_z5k3fix_89500
export above is reused (no second export). Stop: `touch ROOT/STOP` (running evaluations finish and are recorded; nothing
is signalled). Status: `conda_envs/flashchord/bin/python -I -B scripts/t2cons_r7_eval.py status --phase screen`. Then
step R4 (`scripts/t2cons_r7_consensus.py`, to be written and pinned with its selftest in `PREREG_R7_PINS_b.md` before it
scores anything; it re-scores the 735 reused round-6 runs and must reproduce scores_s6 exactly), then `--phase final`.
