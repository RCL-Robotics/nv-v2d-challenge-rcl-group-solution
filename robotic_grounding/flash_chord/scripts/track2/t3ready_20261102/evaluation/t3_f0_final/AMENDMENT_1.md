# AMENDMENT 1 to `fc_runs/submissions/t3_f0_final/PREREG.md`: the route for Tier-3 retry results (CONDITIONAL)

Amends `t3_f0_final/PREREG.md` (sha256 `e4e3a5daf239f5b5858e32442c43eb708947b38c20eae76e12af757a747bd0eb`) under its
section 9. Written 2026-10-26 ~21:45 CST, in response to review R3, must-fix 1.

**Where this file lives.**
- It is staged as `tmp_analysis/tier3/f0_retry/AMENDMENT_1_t3_f0_final.md`, with its sha256 pinned in
  `f0_retry/RETRY_FILES.sha256` and named in `PREREG_retry.md` v2.
- It takes effect only when two things hold:
  - it has been installed byte-identically as `t3_f0_final/AMENDMENT_1.md`, together with `AMENDMENT_1.sha256`;
  - the retry driver has accepted a `RETRY_GO` (PREREG_retry section 4, gate 2).
- The retry driver refuses to start any retry until this file is installed.
- Without an accepted RETRY_GO, or with `RETRY_NOGO`, this amendment is void. t3_f0_final's own FINAL is then the only
  full file.

**State when this was written.**
- t3_f0_final had been running since 21:38. It had frozen the items of 00-08, 10 and 12, and its first own-input runs
  of ep00 existed.
- No retry existed: no RETRY_GO, no retry run, no retry evaluation.
- No Tier-3 Kaggle submission of ours existed.
- No clean file was opened while writing this.
- Disclosure, repeated from both PREREGs: the operator had already seen the LOCAL clean diagnostics of the 11 clips
  whose f0 post step was done (00-08, 10, 12). The retry candidate list is mechanical (PREREG_retry section 1), and
  nothing below is a per-clip choice.

## What does not change

- t3_f0_final keeps running exactly as frozen. That covers its items, its 5 runs per item, its selections and its
  FINAL pack, called **FINAL_f0** below.
- No item, run or selection of t3_f0_final is rewritten.

## (a) Items of the retry-updated file, FINAL_r

FINAL_r is a 30-episode file. Per episode NN, its item is chosen as follows:
- **Retry item.** This applies iff, at the build time fixed in (c), all of these hold:
  - `f0_retry/SUBMISSION_CHOICE.json` lists NN with `use = retry_final`;
  - `f0_retry/eval/epNN/post.json` has status `done`, a decision starting with `REPLACE`, and the same
    `retry_final_sha256`;
  - the pinned PREREG_retry and RETRY_FILES hashes still match.
- **Otherwise:** t3_f0_final's frozen item and frozen selection for NN, i.e. the same rollout as in FINAL_f0
  (byte copy).

## (b) The rollout of a retry item: reuse, no re-run

- **Candidates.** The retry post's 5 evaluations of the retry final against the clip's own locked f0 input:
  `f0_retry/eval/epNN/retry/rK_*`, which are official `eval_all.py` -> kit runner -> kit evaluator runs on stock
  8c778f9, world count 1.
- **Validity check.** Each of these runs is re-checked against every validity condition of t3_f0_final PREREG
  section 2, with these substitutions:
  - "the pinned file" = that run's own evaluated (exported) checkpoint;
  - its export source = the retry `final.json` checkpoint (sha256 as in post.json).

  The other conditions are unchanged: reference = the locked f0 parquet (path + sha256), start frame 0, motion start 0,
  explicit reset, world count 1, eval_script 160d24f5 and eval_code b1b3ebe3, environment_valid on every row, no
  non-finite world.
- **Selection.** Among the runs that pass, choose the one whose own-input ADD-AUC (results JSON `metrics.add_auc`) is
  closest to their mean. Use exact rational arithmetic on the float64 values; ties go to the lower run index. This is
  t3_f0_final PREREG section 3, unchanged.
  - The retry driver records this choice as `chosen_run` in SUBMISSION_CHOICE.json.
  - The builder recomputes it and must agree; if not, the build stops and a human decides.
- **No valid run.** If no retry run passes, the episode keeps its t3_f0_final item, and this is logged.
- **Not used.** Runs of the retry post's f0-final arm are never packed. Every f0 episode uses t3_f0_final's own
  selection.
- **Frame reference.** The episode's own `frame_references_f0` npz, since the input is unchanged.

## (c) When FINAL_r is built (once)

**Build time.** The first of these two moments:
1. every retry candidate (every `f0_retry/recipes/epNN.json`) has a retry outcome; the outcomes are:
   - POST_DONE (post.json status `done`);
   - LATE or CRASHED (marker present);
   - never started by the training cut-off, 2026-11-02 12:00 CST;
2. 2026-11-02 18:00 CST, i.e. 6 h after the training cut-off, so that the post of a retry that ended training before
   the cut-off can finish.

**Conditions.**
- FINAL_f0 must exist, meaning all 30 t3_f0_final episodes are settled and packed. If it does not exist yet, FINAL_r
  is built as soon as it does, with SUBMISSION_CHOICE as it stood at the build time.
- At the build time, a candidate without a done post keeps its t3_f0_final item. Retry results that appear after the
  build time are ignored.
- If no episode is a retry item, FINAL_r is identical to FINAL_f0 and is not built.

## (d) Pack and checks

These are the same as t3_f0_final PREREG section 4:
- the kit packer is used unchanged, with `--track track_2_tier3`;
- per-episode frame references are byte copies;
- the input-reference residual must be < 1e-4 m / 1e-3 rad;
- the uploader's `validate_submission()` must pass;
- provenance must hold. For a retry item, the per-episode checkpoint digest is its chosen run's exported file.

Clean scoring:
- metric_code runs only after the composition is frozen;
- it writes only `*_LOCAL_NOT_FOR_UPLOAD`;
- it is report-only and is never used to choose between FINAL_f0 and FINAL_r, or between either and the interim file.

**Builder.**
- It is a new script; no running or pinned file is edited.
- Its sha256 is recorded in `t3_f0_final/AMENDMENT_2.md` before it runs.
- Acceptance test: built with every episode on its t3_f0_final item, its pose columns must equal FINAL_f0's
  pose-columns sha256 exactly.
- Its decision reads use the same no-clean guard as t3_f0_final. It refuses paths naming clean,
  LOCAL_NOT_FOR_UPLOAD, tier_1_multiview_caption or track_2/baseline.

## (e) Which full file is used, and the Kaggle Final mark

- The designated full file is FINAL_r if it is built and passes (d); otherwise it is FINAL_f0.
- Mark the designated full file as the Final Submission in all 4 Tier-3 competitions, whatever the public scores show.
- If FINAL_f0 is already on the board when FINAL_r is built, upload FINAL_r (with the user's OK) and move the Final mark
  to it.
- Never choose between full files, or between a full file and the interim file, by any board (public = clean) score.
- The cited commit must also contain `t3retry.py`, the builder, this amendment and the retry records whenever FINAL_r
  is uploaded. The submission note must say which episodes are retry items.
