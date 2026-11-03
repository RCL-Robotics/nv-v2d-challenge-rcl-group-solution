# PREREG retry: HELD retry queue for failing Tier-3 f0 clips (arm C3r2_tier3_f0_retry)

Written 2026-10-26 ~21:15 CST (v1, sha256 `8f7da98f…`, kept in `r3fix/v1/`).
**Re-frozen as v2 on 2026-10-26 ~21:50 CST.** This happened before any RETRY_GO, launch, run or result, to apply the
three must-fix items of review R3. v2 changes:
- section 4, gate 2;
- the new section 4a;
- section 6 (the route into the packed file);
- the files list;
- disclosure 6;
- a lane-lock bug fix in section 5. The lock list is now deduplicated by real file. In v1, the
  `fc_runs/queue_t1fix*` / `queue_t2fix` evallock symlinks to `queue_runin/evallock` made the same gpuN.lock appear up
  to 7 times. Our own second flock of it then always failed, so a post on GPUs 0, 1, 2, 5, 6 or 7 would have waited
  forever.

Unchanged: rules 1, 3 and 5, the acceptance rule, the candidate list and the four frozen recipes.

**Built, NOT launched.** Nothing trains until the main session writes an accepted `RETRY_GO` after the user approves
(section 4). `PREREG_f0.md` (sha256 `82702d85…`) is unchanged and keeps governing the f0 runs.
This file adds one rule: a retry may replace an f0 final in the submission (section 6), through
`t3_f0_final/AMENDMENT_1.md`.

**Context (user, 2026-10-26 20:35, verbatim):** "i think it is currently fine to remove tier2 from training queue since
zuxing is fully improve it, i think we need to take care of tier3 right now".

## 0. Disclosures (written before any rule below)

1. **The operator has already seen clean diagnostics.** The main session had seen the LOCAL clean diagnostics of the
   11 finished f0 clips (00–08, 10, 12) before this rule was written.
   - For that reason, the candidate list is not chosen by anyone. It comes from a mechanical rule on the own-input
     official evaluation (section 1).
   - The recipe overrides come from two mechanical rules computed by code on Tier-3 data (section 3).
   - The acceptance test uses own-input evaluations only (section 6).
2. **No clean file was opened here.** The builder (this subagent) opened no clean file at any point. Every script here
   (`t3retry.py`, `diag/*.py`) refuses any path containing `clean`, `LOCAL_NOT_FOR_UPLOAD` or `packed_`.
   - The diagnosis lists every file it opened (`diag/diag_t3only.json` → `opened_paths`).
   - `diag/clean_stat_before.tsv` is a stat-only snapshot of the clean files (taken without opening them).
3. **Thresholds were set after looking at Tier-3 values.** The two rule thresholds (8 cm and 10 %) were chosen after
   seeing the Tier-3-only values of the candidates and of the passing f0 clips (section 3 lists both).
   - No clean, GT or Tier-1 number entered either threshold.
4. **Prior Tier-1 knowledge exists but is not used.** The builder's project notes hold Tier-1 knowledge of some
   episodes, e.g. Tier-1 bat (ep04) findings.
   - None of it is used.
   - No orientation-termination relaxation is proposed, because no Tier-3 signal supports one. The candidates' fast
     rotations (> 300°/s) happen with the hand within 5 cm (ep04: 43 of 48 frames), so they are not released spins.
5. **Other things seen but not used:**
   - Task captions: Han's `episodes.json` and our `f0/supports/epNN.json` contain them.
   - Han's agent reports: their GT-free text was read for ep01, 02, 04 and 05.
   - Han's README: only its non-GT sections were read; the quality section was skipped.
   - `f0/VERDICT.json`: it carries report-only raw Tier-2 numbers.
6. **When v2 was written.**
   - `fc_runs/submissions/t3_f0_final`, the 5-run own-input evaluation of the f0 finals, had started at 21:38.
   - Its log showed the first own-input runs of ep00, which is not a candidate.
   - No Tier-3 Kaggle submission of ours existed.
   - No clean file was opened while v2 was written.
   - The v2 changes are gates and records only. No candidate, recipe, threshold or acceptance rule changed. The four
     recipes were frozen by v1 of `t3retry.py` (sha256 `969ce9d1…`, recorded in each recipe). v2 leaves the recipe
     computation unchanged.

## 1. Candidate rule (mechanical, Tier-3 only; re-applied every pass)

Clip NN is a retry candidate iff all of the following hold:
- its f0 post step is done: `f0/eval/epNN/post.json` has status `done` or `done_score_failed`, eval_status `evaluated`,
  world_count 1, and the reference is the locked training parquet;
- its own-input official evaluation has **SP-SR = 0 or MP-SR = 0**;
- `f0/VERDICT.json` agrees, if it lists the clip as done. A disagreement holds the clip.

Clips that finish later are picked up automatically by the same code (`t3retry.py candidates`, inside the `run` loop).

**Applied 2026-10-26 21:12:** candidates **01, 02, 04, 05**.

| clip | own-input SP-SR | MP-SR | AUC |
|---|---|---|---|
| 01 | 0 | 0 | 0.195 |
| 02 | 0 | 0 | 0.184 |
| 04 | 0 | 0 | 0.171 |
| 05 | 0 | 0 | 0.279 |

ep06 (own-input SP = MP = 1) is not a candidate, even though it fails PREREG_f0's help-zero criterion.

## 2. Diagnosis (Tier-3 evidence only; full numbers in `diag/diag_t3only.json`, `diag/diag2_t3only.json`)

Definitions used below:
- "fail" = the RSI per-start-bin failure EMA; a bin = 25 frames.
- "own-input rollout" = the f0 final's official evaluation against its own input (`f0/eval/epNN/results`).
- Floor depth = the deepest point of the object mesh below z = 0 in the f0 input.
- In all four runs, help was forced to 0 at 39062 by the deadline, not earned.

**ep01 vacuum (support stool, mitigated palm 0.25)**
- **Training.** Bins 0–22 (frames 0–574) failed about 100 % from 5k to the end. Bins 23–29 earned zero help at
  11.3k–33.5k, so the fail wall is at frame 575. Object termination was 93 % of terminations.
- **Input.** The vacuum mesh lies inside the floor:
  - frames 178–500: up to 12.3 cm deep, deepest at frame 382;
  - 222 frames are more than 3 cm deep; the deepest point per bin is 6.5–12.3 cm in bins 8–15 and 9.0–9.5 cm in bins
    18–19;
  - the vacuum also moves with no hand within 5 cm in runs 190–211, 348–394 and others (132 frames).
- **Fit with the training failures.** The wall coincides with the end of the last below-floor runs (563–597). A start before about
  frame 575 cannot survive to the clip end under the 10 cm object limit.
- **Own-input rollout.**
  - The error passes 0.5 rad at frame 153 and 3 cm at 154, right after pickup.
  - It passes 10 cm at frame 228, inside the first deep run.
  - The vacuum then lies on the floor (z 0.07 m) until the end.
- **Flags.** Han's stool fit has IoU 0.61, and the end rest tips in the settle test. Starts in the end rest pass in
  training, so neither flag is the blocker.

**ep02 blue basket (support round table, mitigated palm 0.25)**
- **Training.** Every bin failed 100 % at every logged iteration, even at full help early on. No bin ever earned zero.
  Object termination was 96 %.
- **Input.**
  - The final floor rest (frames 848–979) is 11.6 cm inside the floor, 17.8 cm at frame 843; the root z at the last
    frame is −0.049 m.
  - The floor rest at frames 215–437 is 3.9 cm inside the floor.
  - The root jumps by 14.3 cm in one frame at 182–183 and by more than 5 cm at phase boundaries 125, 214 and 847.
  - The basket moves with no hand within 5 cm on 439 frames, e.g. frames 145–215 of the first carry.
  - These line up with Han's agent report: per-phase static poses, a hand-attached occluded carry at 743–846, and low
    depth confidence in the final rest.
- **Fit with the training failures.** No episode can reach the clip end without crossing more than 10 cm of error.
  That explains failure everywhere.
- **Own-input rollout.** The error passes 3 cm at frame 125 (pickup) and 10 cm at frame 189, just after the 14 cm
  jump. The basket then lies on the floor.
- **Other flag.** The rim is up to 1.5 cm inside the table top on frames 125–132; this is minor in comparison.

**ep04 baseball bat (floor only, not mitigated: align gap 2.84 cm / +1.96 cm, just under the limits)**
- **Training.** Bins 0–26 (frames 0–674) failed 100 %. Bins 29–31 earned zero at 11.3k, so the wall is at frame
  700. Palm termination was **16.1 %** of terminations over the last 5k iterations, against at most 5.3 % in every
  passing f0 clip.
- **Hand-object gap** (`align/ep04.json`). Robot palm to bat on contact frames: median 9.3 cm, p90 14.1 cm; 7.4 % of
  contact frames are over 15 cm (opus_v2: 0 %).
- **Input.**
  - The f0 correction changed the track by 175° compared with opus_v2.
  - The final floor rest (frames 706–802) is 9.6–9.8 cm inside the floor.
  - A 29.7° one-frame rotation occurs at frame 265.
- **Own-input rollout.**
  - The error passes 3 cm at frame 135 and 0.5 rad at 144, during pickup.
  - It passes 10 cm at frame 173 (reference z 1.37 m).
  - The bat then lies on the floor.

**ep05 big black box (floor only, not mitigated)**
- **Input.** It is physically clean:
  - floor depth at most 1.2 cm;
  - largest jump 1.3 cm per frame;
  - palm gap median 2.1 cm;
  - contact on about 470 frames per hand.
- **Training.**
  - Palm termination was **83 %** of terminations.
  - Episode length was still rising at the end: 84 → 311 → 320 steps.
  - Bins 35–36 fell from about 1.0 to about 0.5 fail (bin 37: 0.56 → 0.42); bins 38–39 earned zero at 11.3k.
- **Own-input rollout.**
  - The policy keeps the case within 10 cm until frame 801, but with more than 30° of error from frame 150.
  - It then loses the case during the final stand-it-upright segment (moving 714–937 in Han's agent report): the case
    ends lying (z 0.21 m) where the reference ends upright (0.48 m).
  - Mean error is 13.8 cm / 0.47 rad, so SP fails on position.
- **Reading.** Under-trained, with the 15 cm palm limit binding.

## 3. Recipe (computed by `t3retry.py`, frozen in `recipes/epNN.json` when a clip first becomes a candidate)

**Base, for every candidate: continue the f0 final.**
- Copy `final.json`'s `model_N` + `rsi_N` from `fc_runs/C3r2_tier3_f0/epNN` into `fc_runs/C3r2_tier3_f0_retry/epNN`,
  checking sha256. `fork.json` records the copy.
- Train with the same code (fc_r2 @61fa5d3), experiment `g1_recon_body_coord_rsi_r2`, locked f0 parquet, seed 42,
  scene.support and logging project as the f0 run.
- Overrides:
  - `training.resume=true` and `training.checkpoint=<retry dir>/model_N.safetensors`;
  - `training.max_iterations=90000`;
  - `training.early_stop_help_zero_tail=0`. Early stop must be off: with every bin already at help 0, it would stop
    the resumed run at its first iteration.
  - `reset.start_update=N`. This gives a second RSI help cycle with the unchanged deadline 35156 and zero_ramp 3906
    counted from N:
    - failing bins climb back from help 6.25e-4 (×1/0.9 every ≥ 156 iterations), capped by the decaying floor;
    - help is forced to 0 by N + 39062 = 83984;
    - at least 6016 help-free iterations remain before 90000;
    - export strips the reset fields.
- The evaluated checkpoint is the retry's `final.json` (`model_90000`, reason max_iterations). No checkpoint selection.

**T-obj (training termination only; evaluation has no termination).**
- Rule: `termination.object_position_threshold = min(0.30, max(0.20, ceil_cm(B + 3 cm)))` iff B > 8 cm. Otherwise it
  stays 0.10.
- B = max(deepest mesh point below the floor, largest one-frame root jump) of the locked f0 input.
- Why 8 cm: it is within 2 cm of the 10 cm limit. Passing f0 clips have B ≤ 7.8 cm (ep12 7.8, ep03 6.9 by jump,
  ep07 4.6 by jump, the others ≤ 3.2).

**T-palm.**
- Rule: `termination.palm_position_threshold = 0.25` (the pre-registered mitigation value) iff either:
  - the clip is in `align/mitigation_f0.json`; or
  - the f0 run's palm-termination share over its last 5000 iterations is > 0.10.
- Why 10 %: passing f0 clips have 0.2–5.3 %.

**Frozen 2026-10-26 21:12** (sha256 of each recipe file is in `RETRY_FILES.sha256`):

| clip | B (cm) | palm share | overrides | projected h |
|---|---|---|---|---|
| ep01 | 12.4 (floor) | mitigated | object 0.20, palm 0.25 | 30.4 |
| ep02 | 17.8 (floor; jump 14.3) | mitigated | object 0.21, palm 0.25 | 36.2 |
| ep04 | 9.8 (floor) | 0.161 | object 0.20, palm 0.25 | 36.9 |
| ep05 | 1.3 | 0.832 | palm 0.25 | 39.9 |

The projection is 45078 iterations at the clip's own f0 rate with 3 trainings per GPU.

The hydra config check (CPU, in the dry run) composes each retry command against the f0 run's saved config. They
differ only in these keys:
- output_dir, logging.experiment, logging.run_name;
- training.{max_iterations, resume, checkpoint, early_stop_help_zero_tail};
- reset.start_update;
- the T-obj / T-palm keys.

## 4. HOLD, priority and placement (`t3retry.py run`)

A retry starts only when ALL of these hold:
1. `sha256(PREREG_retry.md)` equals `PREREG_retry.sha256`, and every file listed in `RETRY_FILES.sha256` is
   unchanged. Otherwise the driver aborts.
2. `f0_retry/RETRY_GO` exists and `go_status()` accepts it (v2). It must contain:
   - the line `prereg_sha256 <this file's sha256>`;
   - a line `user_approval: "<the user's message, verbatim>", YYYY-MM-DD HH:MM CST`. A line still holding template
     text (`<` or `>`) is refused;
   - the line `tier3_board_scores_seen: none`, i.e. the GO was given before any Tier-3 Kaggle score of ours existed.

   It is refused when:
   - `f0_retry/RETRY_NOGO` exists (the user declined; template `RETRY_NOGO.template`);
   - `fc_runs/submissions/TIER3_FIRST_UPLOAD` exists and is not newer than RETRY_GO. The main session writes that
     marker right after our first Tier-3 Kaggle upload, interim or full;
   - `fc_runs/submissions/t3_f0_final/AMENDMENT_1.md` is not a byte copy of `f0_retry/AMENDMENT_1_t3_f0_final.md`
     (sha256 pinned in `RETRY_FILES.sha256`), or `AMENDMENT_1.sha256` there does not name it. The amendment is the only
     route by which a REPLACE verdict enters a packed Tier-3 file (section 6).

   The main session writes RETRY_GO, or RETRY_NOGO, only after the user decides, using the templates
   `RETRY_GO.template` / `RETRY_NOGO.template`. **No accepted RETRY_GO = nothing starts.**
3. `tmp_analysis/tier3/f0/ALL_LAUNCHED` exists **and** a live re-check (t3f0_helper.clip_state) finds no FRESH f0
   clip. Retries rank below every FRESH f0 clip.
4. There is no `f0_retry/STOP` and no `f0_retry/HOLD_epNN` for that clip.
5. The projected end (iteration rate of its own f0 run) is before the training cut-off, **2026-11-02 12:00 CST**.
   Otherwise the clip gets a `LATE_epNN` marker and is never started; the f0 final stays.
6. A slot passes the t3f0 fill rule, called unchanged (`t3f0_helper.plan`):
   - never GPU 4, no MPS;
   - fewer than 3 trainings of anybody on the GPU before the start;
   - at least 16.5 GB + 20 GB free plus ramp debt (our trainers carry T3F0=1, so both drivers count them);
   - 2 own-NUMA physical cores.

### 4a. When the user must decide (v2, R3 must-fix 3)

- The user's GO or NO-GO is obtained and recorded (RETRY_GO or RETRY_NOGO) before our first Tier-3 Kaggle upload,
  interim or full. That way no Tier-3 board score exists when the decision is made: board scores are clean aggregates
  over the 30 episodes.
- The interim plan gates its upload on this record (`fc_runs/submissions/t3_interim/PLAN.md` section 2.5 (e)).
- The driver refuses a GO written after `TIER3_FIRST_UPLOAD`. After that point the retry queue can only be opened by a
  new PREREG that discloses every Tier-3 board score seen.

Launch mechanics:
- **Config check (v2).** Before a clip's first start, the hydra config check (section 3) runs on its live recipe and
  is recorded write-once in `cfgcheck/epNN.json`.
  - A recorded FAIL, or a recipe that changed after its check, holds the clip for good; a human decides.
  - Recipes frozen after 21:12 (clips that become candidates later) are written read-only, and their sha256 is
    appended to `recipes.sha256`.
- The start happens under the shared GPU-choice locks (`ch_runs/*/.gpulock`, opened read-only).
- One start per pass, at least 180 s apart, in episode order.
- A launched clip is never relaunched. A trainer that dies without a valid final.json gets `CRASHED_epNN`, and the f0
  final stays.
- The driver never signals a process. The post step only ends its own evaluation session, and only on a 2400 s
  timeout.

## 5. Post step (own input only)

After the retry's final.json, the driver runs 5 + 5 official evaluations against the clip's own locked f0 input, in
the order f0-final r1, retry r1, f0-final r2, …:
- each is `scripts/track2/eval_all.py` with export 1e-4 (2e-4 only if the exporter refuses; recorded), world count 1,
  the stock runtime and kit evaluator;
- all run on the retry's training GPU;
- each holds our lane lock (`f0_retry/evallock`) plus every shared lane-lock file of that GPU (opened read-only, all or
  nothing): queue_runin, queue_tier2v2, queue_* , submissions/*, tier3/f0, tier3/opus_easy, tier2/k3*;
- each is followed by a 35 s rest;
- a technical failure is retried up to 3 times, 5 min apart.

A run is ok only if all of these hold:
- the evaluation reports `evaluated`, with world_count 1 and no non-finite world;
- the reference is the locked f0 parquet (path + sha256);
- the source checkpoint sha256 equals the f0 final's or the retry final's.

**No clean scoring is done by this launcher.**

## 6. Acceptance (own input only) and submission use

For each arm, over its 5 ok runs, the counts are:
- n_both: runs with SP-SR = 1 and MP-SR = 1;
- n_sp and n_mp: runs with SP-SR = 1, and runs with MP-SR = 1;
- the mean own-input AUC.

The retry final **replaces** the f0 final for the submission iff either:
- (i) retry n_both ≥ 4 and retry n_both > f0-final n_both; or
- (ii) retry mean AUC ≥ f0-final mean AUC + 0.01, retry n_sp ≥ f0-final n_sp, and retry n_mp ≥ f0-final n_mp.

In every other case, including fewer than 5 ok runs in either arm, a crash, LATE, or no RETRY_GO, the PREREG_f0
choice stays: the f0 final, or opus_v2 only if f0 has no evaluated checkpoint.

The outputs are `VERDICT_retry.json` and `SUBMISSION_CHOICE.json` (per episode).
- **chosen_run (v2).** For a REPLACE episode, SUBMISSION_CHOICE also records `chosen_run`: among the retry arm's ok
  runs, the one whose own-input ADD-AUC is closest to their mean. The arithmetic is exact rational, and ties go to the
  lower run index; this is the t3_f0_final selection rule. The record lists the run's results JSON, its parquet and
  its evaluated (exported) checkpoint, with their sha256.
- **Frame references.** They stay `frame_references_f0`, because the input is the same.

**Route into the packed file (v2, R3 must-fix 1).**
- `t3_f0_final/PREREG.md` (sha256 `e4e3a5da…`) takes its items from PREREG_f0 only.
- A REPLACE verdict enters a submission only through `t3_f0_final/AMENDMENT_1.md`. It is staged here as
  `AMENDMENT_1_t3_f0_final.md` (pinned in `RETRY_FILES.sha256`) and must be installed byte-identically before any
  retry starts (section 4, gate 2).
- In short:
  - (a) items come from SUBMISSION_CHOICE.json where it says REPLACE; otherwise from t3_f0_final's frozen item and
    selection;
  - (b) a REPLACE episode packs its `chosen_run`, reused and not re-run, after the t3_f0_final section-2 validity
    checks pass on it;
  - (c) the retry-updated file FINAL_r is built once: when every candidate is POST_DONE, LATE, CRASHED or unstarted at
    the training cut-off, or at 2026-11-02 18:00 CST, whichever comes first;
  - (e) the Kaggle Final mark goes to the full file these rules designate (FINAL_r if built and valid, else FINAL_f0),
    whatever the public scores show. Full files are never chosen by board score.

## 7. Cost

- Training: 45078 iterations per retry. At the f0 rates (3 trainings per GPU) that is 30–40 h each, about
  143 slot-hours for the 4, or about 48 GPU-hours.
- Evaluation: 40 runs at about 2.5 min each including export, about 1.7 GPU-hours of lane time.
- Timing: with ALL_LAUNCHED, free slots and a GO around Oct 28, the retries finish around Oct 29–30 and their
  evaluations soon after, before the Nov 2 12:00 cut-off.

## 8. Not in this test (would need a new data root or a code fork)

- Snapping the f0 object track out of the floor and closing the phase-boundary jumps (ep02). This would be a GT-free
  physical correction of the input, but it changes the scored frame references.
- Mitigation (A) and (B) of PREREG_f0: contact labels from opus, and shifting the hand targets.
- A fresh 45k run with the same overrides. It costs about the same, but the help cycle is full-strength. Not chosen,
  so that there is a single recipe.

## Files

- `t3retry.py`: rule, recipes, cfgcheck, dry, run, post, verdict.
- `launch.sh`: starts the HELD driver. Starting it does not start training.
- `RETRY_GO.template`, `RETRY_NOGO.template` (v2); `recipes/epNN.json`; `diag/`, the Tier-3-only diagnosis with its scripts.
- `AMENDMENT_1_t3_f0_final.md` (v2): the staged amendment to `t3_f0_final/PREREG.md`.
- v2 records: `recipes.sha256`, the append-only log of later recipes; `cfgcheck/epNN.json`, the write-once pre-launch
  config checks; `r3fix/`, holding the v1 files and the patch scripts.
- `dryrun/dry_report.json`.
- Hashes: `RETRY_FILES.sha256` (t3retry.py, launch.sh, the 4 recipes, AMENDMENT_1_t3_f0_final.md) and `PREREG_retry.sha256`.
- Runs: `fc_runs/C3r2_tier3_f0_retry/epNN`. Evaluations: `f0_retry/eval/epNN/{f0final,retry}/rK_*`.
- Controls: `STOP`, `HOLD_epNN`, `RETRY_GO`, `RETRY_NOGO`; the marker `fc_runs/submissions/TIER3_FIRST_UPLOAD`.
