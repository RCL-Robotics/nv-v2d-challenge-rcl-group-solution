# t2_consensus_r6_20261031: PREREG_R6_PINS (step R2; adds pins and implementation notes only)

Written 2026-10-31 ~01:27 CST, after the two plan-only dry runs and BEFORE any round-6 export or evaluation. Documents in
force: `PREREG_R6.md` 0b0fbf9a2b581913... (`PREREG_R6.sha256` 071685a5...).

## 1. Implementation notes

1. **Driver.** `scripts/t2cons_r6_eval.py` is round 5's driver with the round-6 bookkeeping only (its docstring Q1-Q8;
   the full text diff against `t2cons_r5_eval.py` 68191fbb and the launcher diff against `t2cons_r5_eval_launch.sh`
   5a4dc89f are in `R6_SCRIPT_DIFFS.txt`). Every pinned function of `t2best5_driver.py` (24b9af39) and `t2cons_eval.py`
   (2cf42786) is used unchanged (export / evaluation commands, ok checks, retries, lanes, integrity, seeds).
2. **Phase bookkeeping.** `--phase screen` = 250 jobs x 3 runs in `runs_s/`; `--phase final` = 30 incumbents (+ the
   finalists of `final_r6.json`, accepted only if `SELECTION_S6.sha256` lists it with its current sha256 and every line
   of that lock holds) x 10 runs in `runs_f/`. Heartbeats `eval_r6_heartbeat_{s6,f6}.json` carry the loop state under the
   key `state` (round 5 used `phase`; here `phase` names screen / final).
3. **Incumbent exports.** For the 30 incumbents the pinned `export_rec` is rebound to their earlier export records
   (`incumbents_r6.json` `export_record`), so the pinned export launcher skips them and every phase-F run evaluates the
   earlier exported file (re-hashed when the job is built and after each run by the pinned ok checks).
4. **K3CA.** `D.ROOTS["k3ca"]`, `D.FRAME_REFS["k3ca"]` and `D.MANIFEST_FOR` are rebound at load time (PREREG_R6 3.1).
   The plan-only run shows the four ep26 K3CA jobs `ok` with hulls present.
5. **Plan-only dry runs (both `plan: OK`).**
   - phase S (01:21:47-01:23:45): pin chain (47 section-11 pins, 3 commits; SNAPSHOT_R6 83 files; TRACKS_R6 107 files;
     round 1's chain), preflight, 250 of 250 jobs `ok`, hulls ok 250 (62 distinct parquets), kit + stock trees UNCHANGED
     vs round 1's baseline, seed signature `delay=0(+g0..4,0-2) reset=42 scale_seed=0(1.0/1.0) wrench=0 vertex=0/0 ppo=42`,
     GPUs 0, 1, 2, 7 usable (2-6 apps each), 3-6 refused by rule, no external evaluation counted.
   - phase F (01:24:16-01:25:17): 30 of 30 incumbents `ok` with their earlier exports re-hashed (`SELECTION_S6.sha256`
     absent, so the plan covers the incumbents only, as expected), hulls ok 30, same integrity, seeds and GPUs.
6. **Dry-run evaluations** (PREREG_R6 3.6) are started with
   `bash scripts/t2cons_r6_eval_launch.sh --phase screen --dry-run-eval --gpus 0` and
   `bash scripts/t2cons_r6_eval_launch.sh --phase final --dry-run-eval --gpus 0`; each writes `dryrun/eval_<phase>_<stamp>/`.

## 2. Pins

pin scripts/t2cons_r6_eval.py ada6e0f98785d3dda2709d03d1896151634481aafb562be8561e2af718f401aa
pin scripts/t2cons_r6_eval_launch.sh afeadbd4c309e1c535c3a1e3648391789f18b1ceb215c175a9a58bddf9389ae9
pin R6_SCRIPT_DIFFS.txt 3abea9903f1e64fb47ae9941eb82b8cb1ab6676d04ad70f26c330b9a03b726d8
pin PREREG_R6.md 0b0fbf9a2b581913ae0da7d36ce76aa785e706c132dc496054f2323a80e487e1
pin PREREG_R6.sha256 071685a59def3c5db379a2b3ac743a5f124ae6a47114ac1583a7d7d7e72c7d50
pin dryrun/r6_eval_plan_screen_20261031_012345.txt c9d5361a48206d25a53311098a818f479525de421c051eb2f72633ef0c859330
pin dryrun/r6_eval_plan_final_20261031_012517.txt f8ff647143e80d1db01b286691de5400c6c5937208da75348786ca5c67da2689
