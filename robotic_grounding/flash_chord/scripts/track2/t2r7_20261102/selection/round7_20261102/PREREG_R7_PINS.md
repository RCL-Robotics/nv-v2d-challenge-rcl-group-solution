# t2_consensus_r7_20261102: PREREG_R7_PINS (step R2; adds pins and implementation notes only)

Written 2026-11-02 ~06:38 CST, after the plan-only dry runs and BEFORE any round-7 export or evaluation. Documents in
force: `PREREG_R7.md` 749649a79ca5616b... (`PREREG_R7.sha256` d1621bd0...).

## 1. Implementation notes

1. **Driver.** `scripts/t2cons_r7_eval.py` is round 6's driver (`t2cons_r6_eval.py` ada6e0f9) with the round-7
   bookkeeping only (its docstring Q1-Q8; the full text diffs of the driver, the launcher, the resolver and the dry-run
   gate against their round-6 sources are in `R7_SCRIPT_DIFFS.txt`). Every pinned function of `t2best5_driver.py`
   (24b9af39) and `t2cons_eval.py` (2cf42786) is used unchanged (export / evaluation commands, ok checks, retries, lanes,
   integrity, seeds). The driver was produced from round 6's text by asserted block replacements (each old block must
   occur exactly once).
2. **Phase bookkeeping.** `--phase screen` = the 154 `screen` rows x 3 runs in `runs_s/`; `--phase final` = 23
   incumbents (priority 1) + the finalists of `final_r7.json` (priority 1; accepted only if `SELECTION_S7.sha256` lists
   it with its current sha256 and every line of that lock holds) + 7 report-only references (priority 2) x 10 runs in
   `runs_f/`; within a trial the pinned launcher orders by (priority, list order), so references run after the decision
   jobs of the same trial.
3. **Earlier exports.** The pinned `export_rec` is rebound (`export_rec_r7`) to the earlier export records of the 23
   incumbents (`incumbents_r7.json`), the 7 references (`r6refs_r7.json`) and the 245 reused round-6 candidates (their
   round-6 `exports/<job>/export.json`, `pool_r7.json` `r6_export_record`), so the pinned export launcher never
   re-exports them; every run of theirs evaluates the earlier exported file (re-hashed when the job is built and after
   each run by the pinned ok checks).
4. **5090 relocation.** For a 5090 screen job the evaluated input (`training_checkpoint`) is its relocated file; the
   driver loads `scripts/t2cons_r7_pool.py` (7a08fa6d, pinned in PREREG_R7 11) from its verified bytes and re-runs its
   `verify_row` (round 5's `verify_pair`, 09a6cb8e) on the job's `RELOCATION_R7.tsv` row at every manifest build; the
   expected sha256 of the evaluated file is the row's `relocated_sha256`, the pool sha256 (the original) is kept as
   `pool_sha256`.
5. **GPUs.** `ALLOWED_GPUS = (6, 7)`; `--gpus` naming any other GPU is refused; `--max-parallel <= 2`, `--lane-max <= 2`,
   `--cooldown >= 10` (defaults 2 / 2 / 10). GPUs 6 and 7 are both NUMA1: evaluations run on the pinned NUMA1 eval-lane
   cores 100-103, 204-207. The launcher unsets every CUDA_MPS_* variable and CUDA_VISIBLE_DEVICES before starting the
   driver (the pinned child environment sets CUDA_VISIBLE_DEVICES per evaluation).
6. **Plan-only dry runs (both `plan: OK`)**, with this driver (1d66a2db):
   - phase S (06:32:28-06:33:54): pin chain (74 section-11 pins, 3 commits; SNAPSHOT_R7 525 files; TRACKS_R7 131 files;
     RELOCATION_R7 130 files; round 1's chain), preflight, 154 of 154 jobs `ok` (128 relocations re-verified), hulls ok
     154 (35 distinct parquets), kit + stock trees UNCHANGED vs round 1's baseline, seed signature `delay=0(+g0..4,0-2)
     reset=42 scale_seed=0(1.0/1.0) wrench=0 vertex=0/0 ppo=42`, GPUs 6 and 7 usable, 0-5 refused by rule, no external
     evaluation counted.
   - phase F (06:33:54-06:35:00): 23 incumbents + 7 references `ok` with their earlier exports re-hashed, plus the
     phase-F dry-run job `ep01_zxg2_120000` on the finalist path (round-6 export re-hashed; `SELECTION_S7.sha256` absent,
     as expected), hulls ok 31, same integrity, seeds and GPUs.
   - The two earlier plan files of 06:30 / 06:31 (`..._063028`, `..._063133`) came from a build (dea28808) that differed
     only in one display line of the plan printout (the export column was shown for incumbents only); kept as records.
7. **Dry-run evaluations** (PREREG_R7 3.6) are started with
   `bash scripts/t2cons_r7_eval_launch.sh --phase screen --dry-run-eval --gpus 6` and
   `bash scripts/t2cons_r7_eval_launch.sh --phase final --dry-run-eval --gpus 6`; each writes `dryrun/eval_<phase>_<stamp>/`.
   Their own-input re-score uses `scripts/t2cons_r7_dryrun_gate.py` (round 6's dry-run gate with the round-7 paths; kit
   Python), output `dryrun/gate_dryrun_r7.json`.

## 2. Pins

pin scripts/t2cons_r7_eval.py 1d66a2dba53287c6a320310c9524fd9ed00e50289852c9eccba74882028a9f2a
pin scripts/t2cons_r7_eval_launch.sh 6111638d0631112de8789f598dd33b80a36ea28f587f3532c3de510a13d19d8f
pin scripts/t2cons_r7_pool.py 7a08fa6d8be91140e139d914b8119352fabb05f58b415eb4a8904f9a02733770
pin scripts/t2cons_r7_dryrun_gate.py d9aabf1d879bc1a80b9e1a19904654ba3f5c29b4bd2678195a3587869fa139d3
pin R7_SCRIPT_DIFFS.txt 2b1b4a1ca8d2f804203d04dc0f58f5a03faa5ccb93bcaf0618f1fb1b5496cb1b
pin PREREG_R7.md 749649a79ca5616b8ddbfc5124828c13da20f2e10e850a39647a74d83c939ec6
pin PREREG_R7.sha256 d1621bd01246905905776361db2de09ed8adb20daf7c90a551a863cb3aebdd6d
pin dryrun/r7_eval_plan_screen_20261102_063354.txt d45be3b4be236b72d5e956f57002e031ed21ca80da61ec5a299b4697eef09184
pin dryrun/r7_eval_plan_final_20261102_063500.txt 011459e12c54b147d1ba20e36e8267d49210c32f55e9f18f942f108a59d84670
pin dryrun/r7_eval_plan_screen_20261102_063028.txt 3430253408bd6c6ae24851e76d42c4c8da836f89504292aa2216ddc485cb1881
pin dryrun/r7_eval_plan_final_20261102_063133.txt 7116747e53fa55cbe4734e2c34082063ab373954486de0290ac0e64dce879b7c
