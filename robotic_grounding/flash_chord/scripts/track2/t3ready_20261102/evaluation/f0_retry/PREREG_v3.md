# PREREG v3: Tier-3 retry wave (C90 continuations of the f0 finals), all 14 rule candidates in one wave

Written 2026-10-31 ~00:25 CST, in `tmp_analysis/tier3/f0_retry/`. At this time no RETRY_GO, retry launch, retry run or
retry result existed. No Tier-3 file of ours has ever been uploaded: `fc_runs/submissions/TIER3_FIRST_UPLOAD` is absent,
and no Tier-3 board score of ours exists.

**What this file governs.**
- The driver is `t3retry_v3.py`, started by `launch_v3.sh`.
- `PREREG_retry.md` v2 (sha256 `8b40cff0…`) is unchanged and still governs everything this file does not change.
- `PREREG_f0.md` (sha256 `82702d85…`) is unchanged and governs the f0 runs.
- `t3_f0_final/PREREG.md` (sha256 `e4e3a5da…`) is unchanged; AMENDMENT_1 amends it (section 7).

**What v3 changes, compared with v2.**
1. **The HOLD is lifted.** The user's new instructions (section 1) supersede `RETRY_NOGO` of 2026-10-27. That file is
   kept, renamed `RETRY_NOGO.superseded_20261031`.
2. **The candidate list is frozen.** It is the 14 clips the unchanged rule selects now that all 30 f0 clips are settled
   (section 2).
3. **GPU allowlist for this wave.** GPUs 0 1 2 5 6 7, with a cap of 3 trainings of anybody per GPU (section 4). This
   is an operational choice, not a selection.
4. **New hash files.** `PREREG_v3.sha256` and `RETRY_FILES_v3.sha256`. The v2 files stay byte-identical and keep
   matching their own pins.
5. **Technical start failure.** A narrow rule for a trainer that never trains (section 5.3).

Nothing else changes: the candidate rule, the recipe rules (T-obj, T-palm), C90, cfgcheck, the LATE gate, the start
spacing and episode order, the post evaluations, the acceptance rules (i)/(ii) and the pick rule. MPPE is NOT added to
the acceptance rule; this was decided now, before any result.

## 1. User instructions (verbatim, as relayed by the main session)

- 2026-10-27 17:23 CST, recorded in RETRY_NOGO: "why retry, I think no need retry right keep current results for tier3
  queue tier1 first, we will wait first round all tier3 finished then add revisement on tier3".
  - The condition "wait first round all tier3 finished" now holds. `tmp_analysis/tier3/f0/ALL_LAUNCHED` was written
    2026-10-27 04:50. All 30 f0 post steps are done, and `f0/VERDICT.json` was written 2026-10-28 08:38.
- 2026-10-30, ~23:40 CST: "also you can queue those tier3 rl-side improvement tier3 is the first priority then ablation"
- A minute later: "make sure all gpu used properly?"
- 2026-10-31, ~00:05 CST: "no currently you could also use gpu7 zuxing is no longer use it at this time"
- 2026-10-31, relayed at ~00:22 CST (exact time not given): "ok lanuch everything and please use gpu wisely"

The builder is a subagent of the main session. It did not receive these messages from the user directly; they are
quoted as the main session relayed them.

## 2. Candidate rule (unchanged) and the frozen list

**Rule.** This is PREREG_retry section 1, unchanged. Clip NN is a candidate iff all of the following hold:
- its f0 post step is done: `f0/eval/epNN/post.json` has status done or done_score_failed, eval_status evaluated, world
  count 1, and the reference is the locked training parquet;
- its own-input official evaluation has **SP-SR = 0 or MP-SR = 0**;
- `f0/VERDICT.json` agrees.

**Applied 2026-10-31 00:1x CST** (`t3retry.py candidates`, then `t3retry_v3.py recipes --write`) to all 30 settled f0
clips. **Result: 14 clips: 01 02 04 05 11 13 14 16 17 19 21 23 25 28.**
- 16 clips have SP = MP = 1 and are not candidates: 00 03 06 07 08 09 10 12 15 18 20 22 24 26 27 29.
- The rule yields the list. Nothing was dropped or added by hand: no ep10, no other clip.
- Nothing is prioritised or ordered by hand. Starts go in episode order, which is the driver's built-in order.

**The list is frozen in code.** `t3retry_v3.py` has
`FROZEN_CANDIDATES = (1, 2, 4, 5, 11, 13, 14, 16, 17, 19, 21, 23, 25, 28)`.
- The driver still re-applies the rule on every pass.
- If the rule ever yields a different set, the driver only logs a WARN and keeps using the frozen list. Nothing is
  added and nothing is dropped.

Every candidate's f0 final is `model_44922` + `rsi_44922` (reason help_zero_tail, i.e. help forced to 0 at the deadline
followed by the 5860-iteration tail).

| clip | object | own-input SP / MP | B (cm) | palm rule | overrides | projected h | recipe sha256 |
|---|---|---|---|---|---|---|---|
| 01 | vacuum | 0 / 0 | 12.4 floor | mitigated | obj 0.20, palm 0.25 | 30.4 | `5155f790…` |
| 02 | blue_easter_basket | 0 / 0 | 17.8 floor | mitigated | obj 0.21, palm 0.25 | 36.2 | `a5c4c008…` |
| 04 | baseball_bat | 0 / 0 | 9.8 floor | share 0.161 | obj 0.20, palm 0.25 | 36.9 | `e29b9d09…` |
| 05 | big_black_box | 0 / 0 | 1.3 | share 0.832 | palm 0.25 | 39.9 | `78638d5c…` |
| 11 | light_blue_book | 0 / 0 | 3.8 | share 0.703 | palm 0.25 | 36.0 | `31bd3f80…` |
| 13 | snack_box | 0 / 0 | 21.1 floor | share 0.124 | obj 0.25, palm 0.25 | 30.8 | `632a9d60…` |
| 14 | round_table | 0 / 0 | 5.1 | share 0.577 | palm 0.25 | 38.0 | `4582d032…` |
| 16 | spray_bottle | 0 / 0 | 6.1 | share 0.234 | palm 0.25 | 32.3 | `654bda8d…` |
| 17 | traffic_cone | 0 / 0 | 3.4 | mitigated | palm 0.25 | 30.2 | `da5056fa…` |
| 19 | cane | 0 / 0 | 14.3 jump | mitigated | obj 0.20, palm 0.25 | 31.5 | `f4dcf26a…` |
| 21 | tall_bar_stool | 1 / 0 | 2.3 | share 0.103 | palm 0.25 | 37.1 | `e1563021…` |
| 23 | toy_stroller | 1 / 0 | 1.9 | share 0.791 | palm 0.25 | 24.8 | `5ab1b61d…` |
| 25 | cyan_water_bottle | 0 / 0 | 11.5 jump | share 0.134 | obj 0.20, palm 0.25 | 28.8 | `de79eb02…` |
| 28 | squeegee | 0 / 0 | 14.7 jump | mitigated | obj 0.20, palm 0.25 | 27.9 | `2017966c…` |

How to read the table:
- **B** = max(deepest object-mesh point below the floor, largest one-frame root jump) of the locked f0 input.
- **"share"** = the f0 palm-termination share over the last 5000 iterations.
- **"mitigated"** = the clip is listed in `align/mitigation_f0.json` (sha256 `2ab767eb…`).
- **Projected hours** = 45078 iterations at the clip's own f0 rate (3 trainings per GPU).

When the recipes were frozen:
- 01, 02, 04 and 05 were frozen on 2026-10-26 21:12 by v1 (unchanged; pinned in `RETRY_FILES.sha256`).
- The other 10 were frozen on 2026-10-31 00:18:19–00:18:30 by code: `t3retry_v3.py` build 1, sha256 `cfdd3072…`,
  kept as `v3fix/t3retry_v3_build1.py`.
  - They are read-only and logged in `recipes.sha256`.
  - Each records `script_sha256` = build 1.
  - The final `t3retry_v3.py` differs from build 1 only by the frozen-list guard above. Its recipe code is identical:
    `compute_recipe` through `recipe` diff empty.
- ep13's recipe and input are the locked f0 input. Han's ep13 re-run release is not used (disclosure D3).

The hydra config check passed for all 14 in the dry run of 00:18 (`dryrun/dry_report.json`): every check true, and the
only diffs against each f0 run's saved config are the allowed keys. The driver re-runs it write-once
(`cfgcheck/epNN.json`) before each clip's first start.

## 3. C90 (PREREG_retry section 3, unchanged)

- **Fork.** Copy the f0 final's `model_44922` + `rsi_44922` into `fc_runs/C3r2_tier3_f0_retry/epNN`, checking sha256.
- **Same as the f0 run.** Code fc_r2 @61fa5d3, experiment `g1_recon_body_coord_rsi_r2`, the locked f0 parquet,
  seed 42, the same `scene.support` and the same logging project.
- **Overrides:**
  - `training.resume=true` with `training.checkpoint=<retry dir>/model_44922.safetensors`;
  - `training.max_iterations=90000`;
  - `training.early_stop_help_zero_tail=0`;
  - `reset.start_update=44922`. A second RSI help cycle: the deadline (35156) and zero_ramp (3906) count from 44922, so
    help is forced to 0 by 83984, leaving at least 6016 help-free iterations before 90000;
  - the recipe's T-obj / T-palm termination keys (training only; evaluation has no termination).
- **Evaluated checkpoint.** The retry's `final.json`: model_90000, reason max_iterations. No checkpoint selection.

## 4. Operational choices (not selection; written after viewing Tier-3 own-input diagnostics)

**GPUs for this wave: 0 1 2 5 6 7.**
- GPU 3 hosts our XHand MPS group (Ketchup-300 seed113, which cannot resume) and then goes to an ablation study.
- GPU 4 hosts our XHand MPS group (3 Box-200 clients) and is never used by Tier 3.
- GPU 7 runs teammate Zuxing's trainer, pid 638820 (about 16 GB), plus his eval watcher, pid 640024. The user allowed
  GPU 7 on 2026-10-31 (section 1). We never touch or signal his processes.

**Cap: at most 3 training processes of anybody per GPU, after the start.** This is the t3f0 fill rule's own value:
- a start needs fewer than 3 trainings on the GPU before it;
- pid 638820 counts as 1, so GPU 7 takes at most 2 Tier-3 jobs while it runs.

Slots: 17 for 14 jobs, so all 14 start in one wave and the launch order is moot. The built-in start spacing is kept:
one start per pass, at least 180 s apart, under the shared GPU-choice locks, in episode order. That is about 4–5 min
per start, so about 1 h for all 14.

**The rest of the fill rule is unchanged** (`t3f0_helper.plan`):
- no MPS on the GPU;
- at least 16.5 GB + 20 GB free plus the ramp debt of young T3F0 trainers;
- 2 own-NUMA physical cores plus their HT siblings per job;
- NUMA-1 GPUs first (5 6 7), then NUMA 0 (0 1 2); then fewest trainings, most free memory, lowest index.

Tier-3 trainers are plain, non-MPS Isaac Lab processes. `train_env` drops every `CUDA_MPS_*` variable, and
`launch_v3.sh` also unsets the MPS pipe and log variables for the driver.

**Simulated placement** (`v3fix/sim_place.py`, 00:19, the unchanged fill rule with each simulated start assumed):

| GPU | Tier-3 jobs | other trainings | total after |
|---|---|---|---|
| 5 | 3 | 0 | 3 |
| 6 | 3 | 0 | 3 |
| 7 | 2 | Zuxing's 638820 | 3 |
| 0 | 2 | 0 | 2 |
| 1 | 2 | 0 | 2 |
| 2 | 2 | 0 | 2 |

Nothing is placed on GPU 3 or 4, and `launch()` refuses GPU 3 and GPU 4. The real placement follows the live view at
each start; for example, if 638820 has ended, GPU 7 may take 3.

**How the allowlist is applied.**
- `t3retry_v3.py` sets `ALLOWED_GPUS = (0, 1, 2, 5, 6, 7)`.
- It sets the same tuple on its own imported module object of `scripts/t3f0_helper.py`; `plan()` reads that global at
  call time.
- It asserts that the helper's `MAX_TRAIN_PER_GPU` is 3.
- **The helper file itself is not edited.** It is pinned in `tmp_analysis/tier3/f0/build_files_f0.sha256` (sha256
  `b9077315…`) and named in `t3_f0_final/PREREG.md`.
- No running process executes or imports `t3retry*.py` or the helper (checked in /proc at 00:1x).
- `t3retry_v3.py` is a NEW file built by `v3fix/make_v3.py`. `t3retry.py` v2 and `launch.sh` v2 are unchanged and
  inert: RETRY_GO names only the v3 sha256, so a v2 driver would refuse to start anything.

## 5. Gates, LATE, and failures

### 5.1 Gates

The v2 gates hold unchanged, with the v3 hash files:
1. **Hashes.** `sha256(PREREG_v3.md)` equals `PREREG_v3.sha256`, and every file in `RETRY_FILES_v3.sha256` is
   unchanged. `RETRY_FILES_v3.sha256` pins:
   - `PREREG_retry.md`, `PREREG_retry.sha256` and `RETRY_FILES.sha256`;
   - `t3retry.py` (v2), `launch.sh` (v2), `t3retry_v3.py` and `launch_v3.sh`;
   - `AMENDMENT_1_t3_f0_final.md`;
   - the 14 recipes.
   Otherwise the driver aborts.
2. **RETRY_GO** must exist and be accepted. It must name this file's sha256, carry a filled user_approval line and the
   line `tier3_board_scores_seen: none`. It is refused while any of these holds:
   - RETRY_NOGO exists;
   - RETRY_GO is not older than `TIER3_FIRST_UPLOAD`;
   - `t3_f0_final/AMENDMENT_1.md` is not a byte copy of the staged amendment, with `AMENDMENT_1.sha256` naming it.
3. **f0 settled.** ALL_LAUNCHED exists and no FRESH f0 clip remains.
4. **No STOP and no HOLD_epNN.** No HOLD_epNN will be written for any clip in this wave. STOP stops only the driver
   (trainings continue); any use of it is disclosed.
5. **LATE, unchanged.** A clip whose projected end (its own f0 rate) is after the training cut-off,
   **2026-11-02 12:00 CST**, gets `LATE_epNN` and is never started; its f0 final stays.
   - At the f0 rates every clip ends between about 2026-11-01 01:00 and 18:00 CST, so no LATE is expected.
6. **Fill rule as in section 4.**

### 5.2 Drops

There is no drop rule and no hand-made drop. A clip leaves the wave only by a mechanical v2 outcome:
- LATE;
- CRASHED: the trainer died without a valid `final.json`, with no relaunch, and the f0 final stays;
- a recorded cfgcheck FAIL.

### 5.3 Technical start failure (v3, operational, applies equally to every clip)

**Definition.** A launched trainer exits before it logs any training iteration after the resume point: the run's
`metrics.jsonl` has no row with iteration > 44922. Such a trainer has not trained, so this is a technical failure, not
an outcome.

**Procedure.** The driver still writes `CRASHED_epNN`. The main session may then relaunch that clip once:
1. stop the driver with STOP;
2. archive the failed attempt, keeping everything: rename the run directory and CRASHED marker with the suffix
   `.startfail1`, and move its `pids.tsv` row to `pids_startfail.tsv`;
3. write `STARTFAIL_epNN` with the cause;
4. restart `launch_v3.sh`.

The relaunch uses the same recipe, fork and code, on a slot chosen by the same fill rule. A trainer that logged any
post-resume iteration is never relaunched. A second start failure leaves CRASHED in place.

## 6. Post, acceptance and pick (PREREG_retry sections 5–6, unchanged)

**Post.** After the retry's `final.json`, run 5 + 5 official own-input evaluations, interleaved f0 final / retry:
- on the retry's training GPU;
- world count 1, export 1e-4 (2e-4 only if the exporter refuses);
- under lane locks;
- the reference is the locked f0 parquet.

No clean scoring is done.

**Accept.** The retry final REPLACES the f0 final iff either:
- (i) retry n_both ≥ 4 and retry n_both > f0-final n_both; or
- (ii) retry mean own-input AUC ≥ f0-final mean AUC + 0.01, retry n_sp ≥ f0-final n_sp, and retry n_mp ≥ f0-final n_mp.

In every other case the f0 final stays. That includes fewer than 5 ok runs in either arm, CRASHED, LATE, or no start.
**MPPE is not part of (i) or (ii).** It is report-only in `summarize()`.

**Pick (`chosen_run`).** For a REPLACE clip, choose among the retry arm's ok runs of model_90000 the run whose own-input
ADD-AUC is closest to their mean. The arithmetic is exact rational, and ties go to the lower run index. Only model_90000
is evaluated; there is no other checkpoint. The outputs are `VERDICT_retry.json` and `SUBMISSION_CHOICE.json`.

## 7. FINAL_r, AMENDMENT_1 / AMENDMENT_2, and the Final mark

**AMENDMENT_1.** `AMENDMENT_1_t3_f0_final.md` (sha256 `e2a43438…`) is installed byte-identically as
`fc_runs/submissions/t3_f0_final/AMENDMENT_1.md`, with `AMENDMENT_1.sha256`, before RETRY_GO. Its rules apply
unchanged:
- (a) items;
- (b) reuse of `chosen_run` after re-validation;
- (c) build time: when every candidate has an outcome or at 2026-11-02 18:00 CST, whichever comes first;
- (d) pack and checks;
- (e) the Final mark.

**Reading of AMENDMENT_1 (a).** "The pinned PREREG_retry and RETRY_FILES hashes still match" means both:
- the v2 pins match: `PREREG_retry.sha256` and `RETRY_FILES.sha256`, all of whose files stay unchanged;
- the v3 pins match: `PREREG_v3.sha256` and `RETRY_FILES_v3.sha256`.

**The FINAL_r builder does not exist yet.**
- It will be written after all retries settle, as a new file.
- Its sha256 will be recorded in `t3_f0_final/AMENDMENT_2.md` before it runs.
- Its acceptance test follows AMENDMENT_1 (d): built with every episode on its t3_f0_final item, it must reproduce
  FINAL_f0's pose columns.

**The Final mark is never chosen by a board score** (AMENDMENT_1 (e)). The designated full file gets the Final mark in
all 4 Tier-3 competitions whatever the public scores show. That file is FINAL_r if it is built and valid, otherwise
FINAL_f0. Any upload needs the user's OK. This file uploads nothing.

## 8. Commit contents (for the later submission commit)

A local commit by author "Haoran Zhang <haoranwh@umich.edu>"; no push from here. It contains:
- the f0 prep code, i.e. `scripts/t3f0_*` including the helper;
- Han's reconstruction code as released;
- the f0 driver (`t3f0_driver.sh` + helper) and the retry drivers (`t3retry.py` v1/v2, `t3retry_v3.py`, `launch*.sh`,
  `v3fix/`, `r3fix/`);
- PREREG_f0, PREREG_retry (v1, v2), PREREG_v3, and `t3_f0_final/PREREG.md`;
- AMENDMENT_1, and AMENDMENT_2 when it exists;
- the 14 recipes, `recipes.sha256`, `cfgcheck/`, and every `*.sha256`;
- the retry records (VERDICT_retry, SUBMISSION_CHOICE).

## 9. Disclosures (text only; no number from any clean or GT source appears here)

**D1. Clean diagnostics of the f0 clips exist.** Clean-scored diagnostic artifacts exist for all 30 f0 clips:
- the f0 post step's per-clip clean files;
- `f0/VERDICT_clean_LOCAL_NOT_FOR_UPLOAD.json` of 2026-10-28;
- `t3_f0_final`'s summary_LOCAL.

The operator (main session) has seen some of them. This builder opened none of them; their names appeared only in
directory listings.

**D2. Earlier clean-scored pilots exist:**
- s4a / s4b (2026-10-20);
- `train90k` with its `score_ckpt` script;
- the ep14_agentT1 90k pilot of 2026-10-20 to 22, whose result and curve files are dated 2026-10-22;
- the opus_easy input-quality files;
- the VERDICT_clean files.

**When and why the C90 cap of 90000 was set.**
- It was set in PREREG_retry v1 on 2026-10-26 (~21:15 CST) and kept in v2. The ep14_agentT1 90k pilot predates it.
- As far as the files show, PREREG_retry v1/v2, `t3retry.py` v1/v2 and `diag/` never name that pilot, `train90k`,
  `score_ckpt`, s4a or s4b. The cap is justified there only by the RSI help-cycle arithmetic: help zero by
  N + 39062 ≤ 84062, with ≥ 5938 help-free iterations before 90000.
- The files cannot show whether the author's knowledge of that pilot influenced the choice of 90000, so that influence
  cannot be excluded.

**D3. Han's ep13 re-run release is not used.** `tier3_ep13_opusmax_20261030` was released after a GT-based selection.
We know of it and computed a GT-free diff of it. Neither the release nor the diff is used: ep13 runs on the locked f0
input with its rule-computed recipe.

**D4. Han's README GT table.** GT-blind audit agents saw Han's README GT quality table (3 aggregate values). They were
not recorded and are not used.

**D5. Tier-2 raw numbers.** Raw Tier-2 numbers were visible to audit agents in `f0/VERDICT.txt`, `f0/fail_rates.csv`,
`f0/prep_f0.tsv` and `f0/driver.log`.

**D6. What `candidates()` reads from `f0/VERDICT.json`.** Only `clips[NN].post_status`, `clips[NN].eval_sp_sr` and
`clips[NN].eval_mp_sr`, for the agreement check with post.json. The file carries other report-only fields, including
raw Tier-2 numbers, which the code does not read.

**D7. Tier-1 and Tier-2 knowledge in context.**
- The main session's context holds Tier-1 knowledge.
- This builder's context held an automatically provided index of project notes containing Tier-1 and Tier-2 aggregate
  numbers. The notes themselves were not opened, and nothing from them is used.

**D8. When the operational rules were written.** Sections 4 and 5.3 were written after viewing Tier-3 own-input
diagnostics:
- the own-input SP / MP of all 30 f0 clips;
- the f0 training rates;
- the recipe rule traces.

The GPU list and cap follow GPU availability and the user's instructions, not any clip's result.

**D9. What this builder saw.** It opened only Tier-3 own-input material:
- post.json through `t3retry candidates`;
- f0 `final.json` files;
- the recipe traces;
- PREREG_retry, AMENDMENT_1, the templates and RETRY_NOGO.

It did not open any file whose name contains clean, LOCAL_NOT_FOR_UPLOAD, summary_LOCAL, s4a_, s4b_, clean_stat, GT
or gt_eval. Nothing derived from such a file is used here.

**D10. A second, paused builder instance.** A second instance of this builder existed, after a resume, and was told
to stand down. Before this file was written, no file in `f0_retry/` changed during a ≥ 2-minute window except this
builder's own.

## 10. Files

- `PREREG_v3.md` (this file) and `PREREG_v3.sha256`.
- `RETRY_FILES_v3.sha256` (the pins listed in 5.1).
- `t3retry_v3.py` and `launch_v3.sh`.
- `v3fix/`: `make_v3.py` (builds v3 from v2), `make_v3_build1.py`, `t3retry_v3_build1.py`, `sim_place.py` with
  `sim_place.log`, `inspect_view.py`, `dry1_stdout.log`, and `test_v3.py`.
- `recipes/` (14), `recipes.sha256`, `cfgcheck/`, `dryrun/dry_report.json` (v3), and
  `dryrun/dry_report_v2_20261026.json` (v2, kept).
- Controls: `RETRY_GO` (v3), `RETRY_NOGO.superseded_20261031`, `STOP`, and `HOLD_epNN` (unused).
- Runs: `fc_runs/C3r2_tier3_f0_retry/epNN`. Evaluations: `f0_retry/eval/epNN/{f0final,retry}/rK_*`.

Reference hashes (sha256):
- `t3retry.py` (v2) `1499df39…`
- `t3retry_v3.py` build 1 `cfdd3072…`
- `scripts/t3f0_helper.py` `b9077315…`
- `scripts/track2_queue_tier2v2_helper.py` `b75a33c3…`
- `PREREG_retry.md` `8b40cff0…`
- `AMENDMENT_1_t3_f0_final.md` `e2a43438…`
- `t3_f0_final/PREREG.md` `e4e3a5da…`
- `PREREG_f0.md` `82702d85…`

The final `t3retry_v3.py` and `launch_v3.sh` hashes are in `RETRY_FILES_v3.sha256`.
