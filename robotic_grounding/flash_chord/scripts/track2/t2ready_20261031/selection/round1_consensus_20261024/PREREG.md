# Tier-2 consensus-reference selection (t2_consensus_20261024): pre-registration

Written 2026-10-24 between 23:32 and 23:59 CST. No consensus reference file had been built and no rollout had been scored
against a consensus track when it was written. At writing, `fc_runs/submissions/t2_consensus_20261024/` holds only
`inputs/` (snapshot and pins, section 1), this file and `PREREG.sha256`. Every later step refuses to run if the sha256 of
this file differs from `PREREG.sha256`.

**Request (user, 2026-10-24, verbatim):** "sure, but you should make sure the final results is also following the
challenge rules and evaluate against the official evaluator?"

It answers the main session's proposal. The proposal has four parts:
- Tier-2 checkpoints are selected per episode against one shared reference, built from the released noisy Tier-2
  data and tracks derived from it.
- Candidates are scored on official-evaluator rollouts.
- The rule is locked before any scoring.
- The final file is packed and scored with the official tools.

**Rule:** selection uses only own-input and consensus (noisy) scores.

Nothing in this folder is uploaded.

## 0. Rules basis

**Verdict.** The main session's rules check of 2026-10-24 found the method **allowed, with conditions**. It re-fetched the
live Kaggle Tier-2 AUC pages, which are byte-identical to the 2026-10-20 copy.

**Binding clauses**
- **§2.8.c.** A Submission must be "produced from that tier's permitted input". Here every packed rollout is driven by a
  policy trained on a Tier-2-derived input: raw, K3fix / K3 or K3CA. The consensus only chooses among such checkpoints and
  never drives a policy.
- **§2.9.d and §2.2.d.**
  - A different checkpoint may be used for each sequence.
  - The checkpoint digest must be identical within an episode.
  - The evaluator digests and the commit must be identical on every row.
- **§2.9.b.** The rollouts come unchanged from the kit's `evaluate_policy.py`, so the metric, the file and the evaluation
  are not manipulated.
- **Foundational 4.b.** The rule is frozen and algorithmic, not hand labeling or human prediction.

**What no rule covers.** No rule text says how checkpoints may be chosen, or speaks to a different input method per
sequence. The remaining risk is the sponsor's discretion, so the method is disclosed (section 9, not included in this copy).

**How the rules check's conditions are met**

| # | condition | where |
|---|---|---|
| 1 | reference, rule, candidate set and choice; path guard | 2, 4, 5.7 |
| 2 | G2 out of the binding reference (variant A) and its candidates | 2.5, 4 |
| 3 | earlier decisions disclosed | 9 (not included in this copy) |
| 4 | all 30 episodes, score-independent candidates, fixed recipe / margin / ties | 4, 5 |
| 5 | PREREG + sha256 before any consensus score; later deviations in AMENDMENTS | header, 10.4 |
| 6 | official evaluator, one build, packer with each episode's own input as frame reference | 6; 7 (not included in this copy) |
| 7 | checkpoint by 5-run mean; packed rollout chosen by a fixed rule, never best-of-5 | 5; 7.2, 7.8 (not included in this copy) |
| 8 | everything that produced the file in the cited commit | 10.3 |
| 9 | disclosure in the technical report; question added to the organiser email | 9 (not included in this copy) |
| 10 | Zuxing on the registered and Kaggle team | 9 (open item; not included in this copy) |

## 1. Inputs and cut-off

**Cut-off T_c = 2026-10-24 23:24:16 CST**, the moment of the input snapshot. Nothing that finished or appeared after T_c
is an input or a candidate of this file. The only exception is Round 2 (section 8, not included in this copy), and only if it is enabled.

**Snapshot.** The `inputs/` folder holds the following, all listed in `inputs/SNAPSHOT.sha256`.
- **`inputs/t2_best5/`.** Copies of t2_best5's own-input files:
  - `selection.json` and `selection.txt`;
  - `own_input_evals.csv`;
  - `manifest.json`, `manifest.csv`, `manifest_stageB.jsonl` and `stageB_resolutions.jsonl`;
  - `PREREG.md` and `PREREG_PINS.md`, each with its sha256;
  - `COPIES.tsv`, `summary_own_input.json`, `heartbeat.json` and `inventory.json`.

  The t2_best5 driver (pid 407660) is still running and still rewrites its own files. This file uses only the copies.
- **`inputs/zuxing_metrics/`.** Copies of Zuxing's 7 own-input `metrics.csv`.
- **`inputs/VERDICT_k3full.json`.** A copy of `tmp_analysis/tier2/k3full/VERDICT.json`, written 22:43:45.
- **`inputs/TRACKS.sha256`.** The sha256 of every track and mesh file that sections 2 and 3 use:
  - the Tier-2 released parquets, the `meta/` files and the meshes;
  - `frame_refs_k3fix`, `frame_refs_k3` and `frame_refs_k3ca`;
  - G2's `frame_refs_g2` and its build JSONs.
- **`inputs/KIT.sha256` and `inputs/CODE.sha256`.** Kit files, scripts, earlier pre-registrations and code commits.

**State at T_c (from the snapshot)**
- **t2_best5:** 75 jobs, each with 5 ok own-input official evaluations (375 rows); all exports ok.
  - Stage B items already resolved: ep09-B, ep14, ep21, ep25.
  - Still open there:
    - ep04: our K3fix run is at iteration 36000 with no `final.json`, and VERDICT says PENDING;
    - T2fix ep01-L: still training at 73500, with no `final.json`.
- **t2_best5 `selection.json`** (generated 23:13:52):
  - SELECTED equals CHOSEN on 29 episodes.
  - ep09 is replaced by T2fix-B 54449.
  - ep04 is flagged PENDING and uses the raw final.
- **K3CA:** only ep26 is training (iteration 1000). The other 6 are not launched, and there is no `final.json`.
- **T2fix:** ep09-B is finished and is already a t2_best5 job. ep01-L is training. ep10, 12, 16 and 17-B were never
  launched.
- **Zuxing's runs:**
  - training: the G2 runs (ep01, 04, 19, 28) and the K3fix runs ep05 and ep26;
  - stopped by him: the K3fix ep04 run.

## 2. Consensus reference (fixed)

### 2.1 Frame and grid

- **Frame.** The released Track-2 world frame (`released_track2_world`). The packer expresses Tier-2 rollouts in this frame.
- **Grid of episode e.**
  - It is the visible frames of the released Tier-2 input, read with
    `v2dlb.trajectory.read_episode(<T2>/data/chunk-000/episode_e.parquet)`.
  - `<T2>` = `data/video_to_data_challenge/track_2/tier_2_synthetic_noise`. The object order comes from the same file.
  - The Tier-2 README says episode timing and visibility are preserved, so this is the scored frame set.
- **Reader.** Every track is read on that grid with the kit's own `input_frame_reference(root, e, grid, 1, names)`,
  imported unchanged from `tools/pack_submission.py`. The packer reads `--frame-references` with the same function.
  - A track that lacks a grid frame is an error.
  - At writing, every track covered every grid frame of every episode.

### 2.2 Tracks

All tracks are derived from the released Tier-2 files.

| id | track | source | episodes |
|---|---|---|---|
| R | raw noisy Tier-2 object track | `<T2>` (`meta/` + `data/`) | all 30 |
| F | K3fix | `tmp_analysis/tier2/k3fix_pilot/frame_refs_k3fix/`, the frame references every K3fix job is packed with | all 30 |
| C | K3CA | `tmp_analysis/tier2/k3ca/frame_refs_k3ca/` | 30 files; distinct from F only on 00, 05, 15, 19, 22, 26, 28 |
| G | G2 (variant B only) | `zuxing.lu/data/human_motion_data_tier2_g2-2026-10-24/frame_refs_g2/` (read-only) | 01, 04, 09, 19, 21, 28 |

**Not used**
- **K3** (`frame_refs_k3`): superseded by K3fix, and equal to it except on 03, 04, 09 and 17. Including it would add a
  second, older version of the same filter.
- **armA and armA_v2:** never adopted.
- Tier-3 roots, and any root that is not Tier-2 data.

### 2.3 De-duplication

The tracks are taken in the order R, F, C, G. A track identical to an earlier one is dropped, tested with
`numpy.array_equal` on the grid arrays as read.
- Identical tracks are one estimator, not two pieces of evidence.
- Counting K3CA = K3fix twice would make the reference exactly the K3fix input, i.e. the training input of one family of
  runs. The rules check says the reference must not be any run's training input.

### 2.4 Combination

Per frame t, over the distinct set D_e (|D_e| = n):
- **Normalisation.** Quaternions are normalised first (`op._normalise_quaternions`).
- **Position.** Componentwise median: p\*(t) = `numpy.median` over D_e of p_i(t), in metres, in the released frame. For
  n = 2 this is the midpoint. For n = 4 it is the mean of the middle two.
- **Rotation: chordal L2 mean.**
  - q\*(t) = the unit eigenvector of the largest eigenvalue of M(t) = sum_i q_i(t) q_i(t)^T, with xyzw quaternions,
    `numpy.linalg.eigh` and float64.
  - Sign: chosen so that q\* . q_R(t) >= 0; if that dot product is exactly 0, so that w >= 0. The sign has no effect on
    any metric.
  - For n = 2, q\* is the geodesic midpoint.
- **Nothing else:** no temporal smoothing, no weights, no outlier threshold and no other parameter.
- **Output.**
  - `refs/<variant>/episode_XXXXXX.npz` holds `episode_index`, `coordinate_frame = released_track2_world`,
    `frame_index` and `pose_xyzw` (T, 1, 7).
  - `refs/build.json` records the tracks used, the duplicates dropped, input and output sha256, and the largest pairwise
    distance between tracks per episode (report only).

### 2.5 Variants

| variant | tracks | role | n per episode |
|---|---|---|---|
| **A** | R, F, C | **binding** | n = 2 (R/F midpoint) on 01-04, 06-14, 16-18, 20, 21, 23-25, 27, 29 (23 episodes); n = 3 on 00, 05, 15, 19, 22, 26, 28 |
| **B** | R, F, C, G | report only | as A, except n = 3 (R, F, G) on 01, 04, 09, 21 and n = 4 (R, F, C, G) on 19, 28 |

- These track sets come from an inventory pass over the tracks at writing.
- Variant B is never binding in this file. The one exception is the Round-2 provision in section 8 (not included in this copy).

### 2.6 Why this recipe

- **Inputs.** Every input is the released Tier-2 data or a filter of it.
- **Equal weights.** The noisy data give no basis for weighting the tracks, and equal weights over distinct tracks need
  none.
- **Estimators.**
  - The median and the chordal mean are closed-form and have no parameters.
  - A rotation median would need an iteration rule and a tolerance, i.e. a free setting.
  - With n <= 4, the choice between them matters little.
- **No tuning.** The main session's task gave the recipe before this file was written, and no consensus number existed
  then.
- **What the score can and cannot tell.**
  - With n = 2, a perfect tracker of R and a perfect tracker of F score the same: each is half the R-F gap away from the
    reference.
  - So on 23 episodes the score separates candidates mainly by how well and how stably they follow their own input. It
    does not say which input is closer to the truth.
  - With n = 3, on the 7 gate-FAIL episodes, the median sits nearer the K3fix family (F and C), which favours inputs from
    that family there (section 9.5, not included in this copy).

## 3. Consensus score (fixed)

### 3.1 Rollouts

- **Which file.** For each job and each run k = 1..5, the scorer reads the parquet of the ok try named in the
  `own_input_evals` file:
  - t2_best5 jobs: `runs/<job>/run<k>/try<t>/results/episode_XXXXXX.parquet` under t2_best5;
  - new jobs: the same layout in this folder's `runs/`.
- **Nothing else** in a run folder is opened.

### 3.2 Frame conversion: the packer's own

- **Reading.** `read_trajectory` from kit `v2dlb.submission_parquet`. Rows are ordered on (world 0, grid frames, slot 0),
  as the packer orders them, and every row must exist.
- **Conversion.** `convert_track2_frame(pose, evaluator_reference, input_reference)`, imported unchanged from
  `tools/pack_submission.py`.
  - The input reference is the job's own policy input in the released frame. This is the frame map t2_best5 used for its
    packs.

    | job input | input reference |
    |---|---|
    | raw | `<T2>` (the packer default) |
    | K3fix | `frame_refs_k3fix` |
    | K3 easy-clip | `frame_refs_k3` |
    | K3CA | `frame_refs_k3ca` |
    | G2 | `frame_refs_g2` |

  - The packer's own tolerance applies: 1e-3 m and 1e-3 rad. A run that exceeds it is `conversion_failed`.
- **Provenance.**
  - The `checkpoint_sha256` the rollout records must equal the job's exported file.
  - The evaluator digests, where recorded, must be the same on every run. The packer enforces the same (section 7, not included in this copy).

### 3.3 Metrics

- **Call.** `v2dlb.object_pose.episode_metric(m, achieved, reference, ids=[0], vertices, align_initial_pose=False)`;
  `False` is the `track_2_tier2` setting.
- **Metrics m:**
  - `add_auc`: ADD-AUC over thresholds of 1 to 9 cm;
  - `spider_sr`, `maniptrans_sr` and `mppe_cm`.
- **Arrays.** `achieved` = the converted rollout, (T, 1, 1, 7). `reference` = the consensus, (T, 1, 1, 7).
- **Vertices.** `sample_vertices(mesh_path_for(<T2>/mesh, name))`: 500 points, seed 0, the packer's own sampler. The meshes are the Tier-2 ones, which the README says are preserved.
- **Precision.** Float64, unrounded. The packed CSV rounds to 6 decimals; the difference is negligible.

### 3.4 Per job

- **Statistics.** For each metric: cons_mean = the mean over the 5 runs, plus the population variance (ddof 0), the
  minimum and the maximum.
- **Completeness.** A job counts as consensus-complete only with 5 scored runs.
- **Coverage.** Every job is scored under both variants.

### 3.5 Checks before the first consensus score

1. **Unit selftests on synthetic data:**
   - de-duplication;
   - n = 1 returns the track;
   - n = 2 gives the midpoint and the geodesic midpoint;
   - n = 3 gives the componentwise median;
   - q\* is invariant to the sign of each input;
   - the path guard of 5.7 refuses every forbidden pattern.
2. **Own-input validation.**
   - **Method.** The same code scores every run against the job's OWN input track on the grid, and the result is compared
     with the evaluator's own-input values in `own_input_evals.csv`.
   - **Expected differences.** Only two sources: the vertex sample (sim mesh vs released mesh), and the conversion
     residual of at most 1 mm.
   - **Pass iff,** over all runs:
     - max |dAUC| <= 0.02;
     - max |dMPPE| <= 0.5 cm;
     - SP-SR equal in >= 95 % of runs, and MP-SR equal in >= 95 % of runs.
   - **On failure:** no consensus score is computed, and the cause is fixed under AMENDMENTS (10.4).

   These thresholds only detect implementation errors; they select nothing. Both results go into `PREREG_PINS.md`.

## 4. Candidates (score-independent)

- **Variant A:** the candidates of episode e are all jobs of e in 4.1 to 4.3 whose input is not G2.
- **Variant B:** all of them.

### 4.1 t2_best5 jobs

Every t2_best5 job with 5 ok official evaluations in the snapshot: 75 jobs.
- **Variant B only:** the 4 G2-input jobs (`ep01_zx_g2_10000`, `ep04_zx_g2_12000`, `ep19_zx_g2_18000`,
  `ep28_zx_g2_16000`).
- **Provisional (P) jobs count.** Each job is one fixed checkpoint file (by sha256), evaluated 5 times. "Provisional" only
  meant that the run's designation could still move.

**Variant-A challengers per episode** (candidates other than the default D of section 5), including the two new jobs of
4.2:

| challengers | episodes |
|---|---|
| 0 | 00, 15, 19, 22, 28 |
| 1 | 04 (Zuxing K3fix 12000); 08, 13, 17, 20, 21, 23, 24, 25, 27 (raw final) |
| 2 | 02, 06, 07, 10, 11, 12, 14, 16, 18, 29 (raw final + t2watch best); 09 (raw final + K3fix final, D = T2fix-B); 05, 26 (Zuxing K3fix: the t2_best5 checkpoint + the newest) |
| 3 | 01 (raw final, t2watch best, T2fix ep01-L (P)) |
| 4 | 03 (raw s42 final, t2watch best, raw s7 final, raw s7 t2watch best) |

The candidates per episode are listed in full in `selection.json` (5.8).

### 4.2 Zuxing's newest designated checkpoints, resolved at T_c

**Rule** (t2_best5 section 1, applied at the new cut-off):
- Take the rows of his snapshot `metrics.csv` with status ok, stock_verified True and finished_at <= T_c.
- Pick the highest own-input `add_auc`. Ties go to the lower `mppe_cm`, then the earlier iteration.
- The file's sha256 must equal the row's `source_sha256`. This was checked read-only at writing for all 7, and is checked
  again at copy.

| run | input | resolved iteration | sha256 | job | role |
|---|---|---|---|---|---|
| g2/ep01 | G2 | 10000 | 4e9423d7... | = `ep01_zx_g2_10000` (reused) | B only |
| g2/ep04 | G2 | 18000 | 91e1a200eb75a1b312f10f16019243ca21a1b1627a80129942791fe134579f78 | `ep04_zx_g2_18000` (new) | B only |
| g2/ep19 | G2 | 20000 | 9f0f090efb6ae2848db74e053ee4a4983bbb2cff690ccc70deb5f0db5940f977 | `ep19_zx_g2_20000` (new) | B only |
| g2/ep28 | G2 | 22000 | 8750a1427b4bf729ecd2061fe45d07dd756ef4722004a2f5688e84edd35b1b4f | `ep28_zx_g2_22000` (new) | B only |
| k3fix-ep04/ep04 (stopped) | K3fix | 12000 | 3efcbf86... | = `ep04_zx_k3fix_12000` (reused) | A, B |
| k3fix-ep05-ep26/ep05 | K3fix | 26000 | 4f925b2a34b93a540cdba278030d104355711d0165deec6f8ac6bcc97be574f6 | `ep05_zx_k3fix_26000` (new) | A, B |
| k3fix-ep05-ep26/ep26 | K3fix | 20000 | 249ed85df2fcbf14103e86931cf883ce2f5e6da36aca659c9d0310b14c38d081 | `ep26_zx_k3fix_20000` (new) | A, B |

- There are 5 new jobs, evaluated under section 6. Two of them can enter variant A: `ep05_zx_k3fix_26000` and
  `ep26_zx_k3fix_20000`.
- The earlier frozen checkpoints stay candidates through 4.1: ep05 18000, ep26 16000, and G2 ep04 12000, ep19 18000 and
  ep28 16000.

### 4.3 T2fix and K3CA finals newly finished at T_c

**None.**
- T2fix ep01-L has no `final.json`, and T2fix ep10, 12, 16 and 17-B were never launched.
- K3CA ep26 is at iteration 1000, and the other K3CA jobs are not launched.
- Our ep04 K3fix run (36000 at T_c, no `final.json`) is not a candidate here either (section 9.8, not included in this copy).

### 4.4 Not candidates

- `inventory.json` non_candidates: k3 stretch90k, armA_90k, K3CA (then untrained), and Zuxing's runs that did not
  train on a Tier-2 input;
- checkpoints that are not their run's designated checkpoint;
- anything that appeared after T_c.

## 5. Selection rule

The rule is fixed and applied to all 30 episodes. Variant A is binding. Variant B is computed the same way, for the report.

For episode e:
1. **Default D.** D = SELECTED of e in the snapshot `selection.json`, which is t2_best5's pre-registered own-input
   choice. For ep04, D = the raw final, flagged PENDING. D is the same for both variants.
2. **Incomplete default.** If D has fewer than 5 consensus-scored runs, the episode keeps D, is flagged INCOMPLETE, and
   gets no replacement.
3. **Challengers.** The challengers are the candidates of e under this variant (section 4), other than D. Each needs 5
   consensus-scored runs and 5 ok own-input runs.
4. **Qualifying.** A challenger c qualifies iff BOTH:
   - (a) cons_mean_AUC(c) >= cons_mean_AUC(D) + 0.01, comparing exact floats with no rounding;
   - (b) **stability floor:** in at least 4 of its 5 runs, the evaluator's own-input SP-SR = 1 AND MP-SR = 1 in the same
     run. The values come from `own_input_evals` and are measured against the challenger's own training input.
5. **Choice.** If any challenger qualifies, take the qualifying one with the highest cons_mean_AUC.
   - Exact ties go to the lower population variance of its 5 consensus AUCs.
   - Then to the lower cons_mean MPPE.
   - Then to the job name (ASCII order).

   Otherwise the episode uses D.
6. **Nothing else enters.**
   - The consensus SP-SR, MP-SR and MPPE are reported only, except as the tie-break above.
   - No manual override.
7. **Path guard.** The code of steps S2-S5 (10.1) refuses, before opening, any path that contains (case-insensitive) one of the
   tokens in `FORBIDDEN` (`scripts/t2cons_consensus.py`).
8. **Output.**
   - **`selection.json` / `selection.txt`,** per episode and variant:
     - D;
     - every challenger's 5 consensus AUCs, with their mean and variance;
     - its floor count, and whether it qualifies, with the reasons;
     - the pick, and the flags.
   - **`summary_consensus.{md,json}`,** for the compositions DEFAULT, CONS-A, CONS-B, RAW-ONLY and K3FULL:
     - the average consensus AUC / SP-SR / MP-SR / MPPE under the A and B references;
     - the own-input values.
   - **Freeze.** `SELECTION.sha256` is written last. From that moment the selection is final ("frozen").

## 6. Evaluations of the 5 new jobs (section 4.2)

The protocol is t2_best5 PREREG sections 3.2-3.5 and 3.7, unchanged, except for the points below.
- **Output** goes under this folder: `zuxing_ckpts/`, `COPIES.tsv`, `exports/`, `runs/`, `manifest_new.json` and
  `own_input_evals_new.csv`.
- **Order.**
  - The variant-A jobs (`ep05_zx_k3fix_26000`, `ep26_zx_k3fix_20000`) go first, with run 1 of each before run 2, and so
    on.
  - Then the B-only jobs.
  - Variant A's selection (section 5) waits for the A jobs. Each A job ends with 5 ok runs or a terminal status:
    `export_refused`, `export_incompatible`, `hull_missing`, or 3 failed tries for a run. A job that ends without 5 ok
    runs is listed and is not a challenger.
  - Variant A does not wait for the B-only jobs. Variant B is reported when they finish.
- **Data root and parquet** are those of t2_best5's manifest row for the same run (K3fix root or G2 root), re-hashed.

**Protocol summary (as in t2_best5)**
- **Copies.** model + rsi are copied with sha256 (model sha256 = the metrics.csv `source_sha256`). There is one
  `COPIES.tsv` row per file, and no git command is run in Zuxing's folder.
- **Export** with `code/nv-v2d_runin` @ b39b69b at 1e-4, falling back to 2e-4.
- **Hulls.** The hull-key check is run, plus the before/after tree check of the stock runtime and the kit.
- **Evaluation.**
  - `eval_all.py --export no` calls the kit's `run_policy_evaluation.py` and `evaluate_policy.py` (160d24f5) on the
    untouched stock runtime 8c778f9.
  - Frame-0 start, explicit reset, world count 1, with the job's training parquet as the reference.
  - The ok checks are those of t2_best5 section 3.5, including one evaluator build across all runs.
- **Scheduling.**
  - GPUs 0, 1, 2, 5, 6 and 7 only. GPU 4 never; GPU 3 has no lock file.
  - Each evaluation holds a flock on `evallock/gpu<g>.lock`, which is opened read-only. One evaluation per GPU, no MPS.
  - Each evaluation runs with `nice` + `taskset` on the eval-lane cores 48-51,152-155 / 100-103,204-207. At most 2 of
    ours per lane and 4 in total.
  - Nothing is ever signalled. An evaluation past 2400 s is flagged OVERDUE. A `STOP` file stops new starts.
- **Load.** 25 evaluations, about 30 GPU-minutes.

## 7. (not included in this copy)

## 8. (not included in this copy)

## 9. (not included in this copy)

## 10. Implementation, hard rules, amendments

### 10.1 Steps

| step | what | when |
|---|---|---|
| S1 | scripts; `PREREG_PINS.md` (+ sha256) with their sha256, the selftests and the 3.5 validation result | before the first consensus score |
| S2 | build the references (`refs/`) | |
| S3 | evaluations (section 6) | |
| S4 | consensus scores: `scores/consensus_scores.csv`, per job x run x variant | |
| S5 | selection, ending with `SELECTION.sha256` | |
| S6 | pack | after S5 |

- The scripts are new files, `scripts/t2cons_*.py`.
- They may import kit modules and `scripts/t2best5_driver.py`, read-only.
- `PREREG_PINS.md` may add pins and implementation notes; it changes no rule.

### 10.2 Hard rules

- New files only: under this folder, plus `scripts/t2cons_*`.
- Never edit a file that a running process uses.
- Never signal any process.
- Never GPU 4; no MPS.
- Zuxing's and Han's folders are read-only: only the sha256-checked copies of section 6 are taken, and no git command is
  run there.
- The kit and `code/video_to_data_8c778f9` are never modified.
- Python runs with `-B`; kit Python with `-I -B`.

### 10.3 Cited commit (needed only for an upload, §2.9.a-c)

The commit must contain:
- the fc_r2 61fa5d3 training code;
- Zuxing's bc3b1bd training changes, copied in with his consent;
- the b39b69b exporter;
- the K3, K3fix and K3CA code and constants;
- retargeting;
- `scripts/t2cons_*`, and this folder's PREREG, PREREG_PINS, AMENDMENTS and `selection.json`;
- the pack command.

### 10.4 Amendments

- Any deviation goes into `AMENDMENTS.md`, with time and reason, before the affected step.
- An amendment written after any consensus score was visible must say so.
- All amendments are disclosed.

## 11. Pins (sha256)

pin inputs/SNAPSHOT.sha256 9e10e36572c3461960e765944ac92716a861cff6f73a2f6d0eea41d1abd3cbdf
pin inputs/SNAPSHOT.txt 4fa502e3a48c1d08304633194fbbcd02af92efc5c4e3281309c35edc54657a2d
pin inputs/TRACKS.sha256 c8cb834522f245de0b7733579f8c537781eef14d79269e782bc391f1dcecca6d
pin inputs/KIT.sha256 2180e52b877fce5f94a3eb232a48fd7b93d319809d24a39b90138c88e9acee0a
pin inputs/CODE.sha256 48799a427e90a1032422dc1985e4967bc7ff9ac0b58d5e99df66ddaec066c813
pin inputs/t2_best5/selection.json 55ee16518f79f9763e9987ccc4fd69c69a4934c361b2aeccc9aba730a41535b2
pin inputs/t2_best5/own_input_evals.csv f3ea6a2a327cfa77f9ecf51d1ffbc7d31c40e96bfcd58eedd682f486ad46c9cc
pin inputs/t2_best5/manifest.json 61b75c557f7a72c1f70164fcd44bb84dd6d7ba8dc91ca0ff3ca88c2f153039fb
pin inputs/t2_best5/manifest_stageB.jsonl 1ace9a17350e89bc873d6d3db10db76425661d204d9b75426a22d70d6c03630a
pin inputs/t2_best5/PREREG.md d778cb29b878e87d3d9538da0c3617f5a13dc490924b2a26187e0d427461b704
pin inputs/VERDICT_k3full.json 2dd3905a20a990f94d0832f09bd0990a65f661524ea603f99990ecbefe27f071
pin kit tools/pack_submission.py 47a4589574ff8284ba4376ef31ac5548c8fe87ead753be2d3b43d6c87041e5be
pin kit v2dlb/object_pose.py c6915924a322540f00c69c75b2ae93ae455bb6efb8a41cc0a96c967f64d7e2c0
pin kit flash_chord/scripts/evaluate_policy.py 160d24f5fbeb70ccd569ca4192056f1892b2a9f2999066db9d8279461648fdf1
pin kit metric_code/track_2/AUC.py 58b437697c32c86497ff5a306fe586014c3429b0d2c18e1d3094a6f503479339
commit code/fc_r2 61fa5d31ff8054bf1e5cf5375ef0957e54d2729b
commit code/nv-v2d_runin b39b69be6c2f561c0123c6aeb8cd13f8b9d542e7
commit code/video_to_data_8c778f9 8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef
