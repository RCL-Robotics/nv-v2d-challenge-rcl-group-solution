# Tier-2: 5x official re-evaluation and composition (t2_best5_20261024): pre-registration

Written 2026-10-24 ~18:36 CST, before any checkpoint copy, export or evaluation for this folder. At writing,
`fc_runs/submissions/t2_best5_20261024/` holds only `inventory/`, this file and `PREREG.sha256`. The driver must refuse to
start if the sha256 of this file differs from `PREREG.sha256`.

**Request (user, 2026-10-24, verbatim):** "I think zuxing run some experiment on tier2 also, you should also take account
his results right? what is overall tier2 results? double check it by rerun using official evaluator 5 times and get
highest and lowest and average?"

**Standing rule (user, 2026-10-24; challenge rule 2.8.c):**
- Selection uses only own-input scores, i.e. each policy evaluated against its OWN training parquet.
- Own-input AUCs are comparable only between candidates trained on the same parquet (same sha256).

Nothing in this folder is packed for upload or uploaded.

## 0. Inputs (pinned in section 6)

- **Inventory**, snapshot 2026-10-24 17:58:27 CST: `inventory/inventory.json`, `inventory/inventory_own_input.csv` and
  `inventory/build_inventory.py`.
- **`tmp_analysis/tier2/k3full/PREREG_v3.md`** (sha256 0f4fa433): the K3fix-or-raw input rule and the final.json
  checkpoint rule. Unchanged and in force.
- **`tmp_analysis/tier2/k3full/VERDICT.json`**:
  - The running k3full driver (pid 176764) rewrites it at least every 30 min. It is read-only here.
  - At writing: 19 K3FIX, 7 RAW, 4 PENDING (ep04, 14, 21, 25).
- **`fc_runs/T2fix/PREREG.md`** (sha256 00a95289): unchanged and in force for its own verdict.
- **Code:**
  - `code/fc_r2` @ 61fa5d31ff80, no uncommitted changes;
  - `code/nv-v2d_runin` @ b39b69be6c2f;
  - stock runtime `code/video_to_data_8c778f9` @ 8c778f905cb3;
  - the kit's `flash_chord/scripts/evaluate_policy.py` is identical to the stock one (160d24f5);
  - `scripts/track2/eval_all.py` is byte-identical in fc_r2, nv-v2d_runin and Zuxing's speedup tree (65729592);
  - kit `tools/pack_submission.py` (47a45895).

## 1. Candidates

**Roles**
- **CHOSEN**: the PREREG_v3 choice of the episode, as `VERDICT.json` gives it.
  - K3FIX: the K3fix run's final.json checkpoint.
  - RAW: the raw `C3r2_tier2` final.json checkpoint.
  - PENDING: the raw final.json checkpoint, flagged PENDING, until the K3fix run settles (Stage B).
- **RAWREF**: the raw `C3r2_tier2` final.json checkpoint, for the raw-only reference composition. Evaluated for all 30
  episodes.
- **SAME**: trained on the same parquet sha256 as CHOSEN. It may replace CHOSEN under section 2.
- **DIFF**: trained on a different input. Report only.
- **(P)**, provisional: its run is still training, so its designated checkpoint may still change.

Bracketed values are the first 8 hex of the training parquet's sha256. Checkpoints are given as `iteration sha256[:8]`.
Full sha256 values are in `inventory/inventory.json`, and the manifest re-hashes every file (section 3.1).

### Stage A: frozen when the manifest is built (every checkpoint exists now)

| ep | object | CHOSEN at writing | RAWREF | SAME | DIFF (report only) |
|---|---|---|---|---|---|
| 00 | tall_bar_stool | RAW (gate FAIL): raw 41043 49323000 [bec9fcbb] | = CHOSEN | - | - |
| 01 | vacuum | K3FIX 44922 d40a325a [k3fix 2b178b1d] | raw 44922 0774dd47 | K3fix t2watch best 18000 d413e8d8; T2fix ep01-L (P) 62000 2d69e2da | Zuxing G2 (P) 10000 4e9423d7 [g2 1107c47d] |
| 02 | blue_easter_basket | K3FIX 34093 3311068b [8f83fd9f] | raw 37409 ad269e99 | t2watch best 34000 f5a89d21 | - |
| 03 | contigo_coffee_mug | K3FIX 27068 88278b72 [c9f6c975] | raw s42 44922 77322707 | t2watch best 24000 1020420a | raw s7 final 44922 fd1f7624; raw s7 t2watch best 38000 d4a192ed [raw 86bb862a] |
| 04 | baseball_bat | PENDING -> raw 44922 e5d58f02 [42f9f807] | = CHOSEN | Stage B | Zuxing K3fix (stopped run) 12000 3efcbf86 [k3fix 7405d01e]; Zuxing G2 (P) 12000 9a33c97d [g2 4b07b0d1] |
| 05 | big_black_box | RAW (gate FAIL): raw 44922 e1fe6427 [b6ed9f13] | = CHOSEN | - | Zuxing K3fix (P) 18000 d17dd768 [k3fix 2ccbe480] |
| 06 | monitor | K3FIX 28420 88dafebf [553e4322] | raw 33894 542847af | t2watch best 28000 15f3d75c | - |
| 07 | iron | K3FIX 18529 bd6639ec [4c2d6fb3] | raw 44922 e4fce7f0 | t2watch best 14000 efcbff67 | - |
| 08 | skinny_wood_chair | K3FIX = K3 easy-clip 17496 c8988396 [k3 510bdc3d, same bytes as K3fix] | raw 20597 7fa36fad | - (t2watch best = final) | - |
| 09 | big_red_bowl | K3FIX 39449 4d962b4e [b1d62056] | raw 39328 2f395947 | Stage B (T2fix ep09-B) | - |
| 10 | white_desk | K3FIX 25319 79ec18cf [884179d7] | raw 34267 60472406 | t2watch best 18000 7ca68b5e | - |
| 11 | light_blue_book | K3FIX 29175 3a330ab0 [81424556] | raw 28925 5f31f3ce | t2watch best 28000 966654cc | - |
| 12 | black_platform | K3FIX 23178 3a25f30e [5f2d1c66] | raw 37575 1acb679e | t2watch best 20000 fc092d19 | - |
| 13 | snack_box | K3FIX 39254 dbb526df [c36c13db] | raw 44922 a6803e2b | - | - |
| 14 | round_table | PENDING -> raw 44922 313e5f7d [5edf5e6a] | = CHOSEN | Stage B | - |
| 15 | wet_floor_sign | RAW (gate FAIL): raw 44922 db94b3d0 [bf06225b] | = CHOSEN | - | - |
| 16 | spray_bottle | K3FIX 32063 c5571c15 [5adf79bd] | raw 31268 1a4d16b2 | t2watch best 32000 d4c73106 | - |
| 17 | traffic_cone | K3FIX 29845 64cbd5ec [25c13883] | raw 31807 99821794 | - | - |
| 18 | blue_trash_can | K3FIX 18911 6772981a [d8d02701] | raw 44922 5e62032f | t2watch best 18000 7db77c98 | - |
| 19 | cane | RAW (gate FAIL): raw 44922 651a438d [9179b0ac] | = CHOSEN | - | Zuxing G2 (P) 18000 a9c60e43 [g2 46039cd2] |
| 20 | rice_cooker | K3FIX = K3 easy-clip 18450 46dc08c1 [k3 2784cae6] | raw 19034 cb8b5954 | - | - |
| 21 | tall_bar_stool | PENDING -> raw 44922 523f168c [97cd0d4f] | = CHOSEN | Stage B | - |
| 22 | step_ladder | RAW (gate FAIL): raw 44922 402aa8bc [12d1318f] | = CHOSEN | - | - |
| 23 | toy_stroller | K3FIX = K3 easy-clip 19162 8cea19c9 [k3 3f6aaa9e] | raw 20023 0f95efb0 | - | - |
| 24 | watering_can | K3FIX 20636 ff795e71 [cafbf612] | raw 44922 3fa5b224 | - | - |
| 25 | cyan_water_bottle | PENDING -> raw 44922 0007a33a [916eab42] | = CHOSEN | Stage B | - |
| 26 | skateboard | RAW (gate FAIL): raw 34948 b7c6a9af [074c06f5] | = CHOSEN | - | Zuxing K3fix (P) 16000 0f3aed40 [k3fix 01637248] |
| 27 | woven_basket | K3FIX 20863 ffb5aae5 [28fe6aee] | raw 44922 93bb6238 | - | - |
| 28 | squeegee | RAW (gate FAIL): raw 44922 edb882c7 [24b10cba] | = CHOSEN | - | Zuxing G2 (P) 16000 9d448d81 [g2 0d49273d] |
| 29 | rolling_luggage | K3FIX 17358 52dae9a8 [9412a5a4] | raw 44922 c9c4a388 [f3a5b6ce] (2e-4 export expected) | t2watch best 16000 a4cf5834 | - |

**Stage A has 70 jobs:**
- 30 RAWREF;
- 19 K3fix / K3 easy-clip CHOSEN;
- 11 t2watch bests (ep01, 02, 03, 06, 07, 10, 11, 12, 16, 18, 29);
- 1 T2fix ep01-L (P);
- 7 Zuxing;
- 2 raw seed-7.

For ep04, Zuxing's K3fix run (7405d01e) is DIFF while ep04 is PENDING, i.e. raw. It becomes SAME if ep04 settles as K3FIX.
Its 5 evaluations from Stage A are reused.

### Stage B: added when settled (section 3.8)

- **ep04, 14, 21, 25: our K3fix run's final.json checkpoint.**
  - It is added as soon as `VERDICT.json` reports choice K3FIX with settled = true for the episode. It then becomes
    CHOSEN, and the raw final stays RAWREF.
  - Its t2watch best is also added, as SAME, if it names another iteration. That best is read from `best.json`, after
    t2watch has a terminal row for the final.json checkpoint.
  - If VERDICT says RAW, nothing is added and raw stays CHOSEN.
- **ep09, T2fix ep09-B:** added once `fc_runs/T2fix/T2B/ep09/final.json` and `fc_runs/T2fix/eval_ARM/ep09/runin_final.json`
  exist. The checkpoint is runin_final.json's best iteration (T2fix's designated checkpoint). It is SAME.
- **ep01, T2fix ep01-L:** added on the same condition, from `T2L/ep01` and `eval_ARM/ep01`. It is SAME. The Stage-A
  provisional job stays in the report.
- **Not candidates of this file:**
  - T2fix ep10, 12, 16 and 17-B, which were never launched;
  - K3CA, which is not trained;
  - any run started after this file;
  - later checkpoints of Zuxing's runs.

### Designated checkpoint: one per run

- **C3r2-family runs** (raw, K3fix, K3 easy-clip, seed 7): final.json.
- **t2watch best:** `tmp_analysis/tier2/t2watch/best/<arm>/<run>/best.json`, i.e. the highest own-input AUC, ties to the
  earlier iteration.
- **T2fix:** runin_final.json's best iteration. While the arm runs (P), it is the best single own-input eval in
  `eval_ARM/evals.csv` finished by 17:58:27.
- **Zuxing:**
  - The highest own-input `add_auc` over the rows of the run's `metrics.csv` with status ok, stock_verified True and
    finished_at <= 2026-10-24 17:58:27 CST. Ties go to the lower `mppe_cm`, then the earlier iteration.
  - The row's `source_sha256` must equal the file's sha256.
  - The resolution is frozen and never redone within this file.
  - Results: ep01 G2 10000, ep04 G2 12000, ep04 K3fix 12000 (all six rows tie at AUC 0.11036737; 12000 has the lowest
    MPPE), ep05 K3fix 18000, ep19 G2 18000, ep26 K3fix 16000, ep28 G2 16000.
  - **Settled status:** his ep04 K3fix run was stopped by its owner and can no longer change, so it is treated as
    settled. His other 6 runs are (P).

### Inventory rows not evaluated, with reasons

- **Rows that are not the run's designated checkpoint:** Zuxing ep01 G2 24000; T2fix ep01-L 64000; Zuxing ep04 K3fix 2000.
- **Provisional checkpoints of our pending K3fix runs:** ep04 28000, ep14 36000 / 44000, ep21 42000, ep25 40000.
  PREREG_v3's checkpoint is final.json, which Stage B evaluates.
- **T2fix ep09-B provisional 52000:** the arm ends at 54449, about 1 h after writing. Its designated checkpoint is
  evaluated in Stage B.
- **T2fix ep10, 12, 16 and 17-B:** never launched, so there is no checkpoint.
- **`inventory.json` non_candidates:** k3 stretch90k, armA_90k, K3CA, and Zuxing's runs that did not train on a
  Tier-2 input.

## 2. Selection rule

The rule is applied per episode e, at each summary.

1. **CHOSEN.** C = CHOSEN from `VERDICT.json`, read-only.
   - Its `prereg_check` must name PREREG_v3 sha256 0f4fa433....
   - PENDING maps to the raw final.json checkpoint, and the episode is flagged PENDING.
2. **Challengers.** S = the candidates of e, other than C, whose training parquet sha256 (all 64 hex, re-hashed at
   manifest build) equals C's.
   - For SELECTED, only settled candidates count.
   - For SELECTED+P, provisional ones count too.
3. **Qualifying.** A challenger s qualifies iff C and s each have exactly 5 ok own-input evaluations from this folder, and:
   - (a) mean5 AUC(s) >= mean5 AUC(C) + 0.01;
   - (b) mean5 SP-SR(s) >= mean5 SP-SR(C), and mean5 MP-SR(s) >= mean5 MP-SR(C).

   Exact floats are compared, with no rounding.
4. **Choice.** If at least one challenger qualifies, the episode uses the qualifying one with the highest mean5 AUC. Ties
   go to the higher mean SP-SR, then the higher mean MP-SR, then the lower mean MPPE, then listing order (t2watch best,
   then T2fix, then Zuxing). Otherwise the episode uses C.
5. **DIFF candidates are never selected.**
   - Own-input AUC measures tracking of the policy's own reference. Raw, K3fix and G2 references differ, so the numbers
     are not on one scale.
   - No existing pre-registered rule covers them:
     - PREREG_v3 sends the gate-FAIL episodes (including 05 and 26) to raw, and covers only our K3fix runs;
     - T2fix covers only its own arms;
     - K3CA's rule covers only K3CA runs, which are not trained.
6. **Only the 5 new evaluations count.** Earlier 1- or 3-eval values are shown next to them and never pooled.
7. **Incomplete CHOSEN.** If C ends with fewer than 5 ok evaluations after retries, the episode keeps C, is flagged
   INCOMPLETE, and gets no replacement.
8. **Inputs of the rule.** Selection reads own-input numbers only.

**Relation to the other rules**
- **PREREG_v3** is unchanged: the K3FULL composition below is exactly its rule. Its "t2watch best never replaces
  final.json in this arm" governs K3FULL. SELECTED is a separate composition defined here.
- **T2fix:** its own 3-eval verdict (`fc_runs/T2fix/verdict.json`) is reported next to this file's 5-eval decision. If they
  disagree, SELECTED follows this file and the disagreement is listed.

## 3. Protocol

### 3.1 Manifest

`manifest.json` and `manifest.csv` are built once, for Stage A, after checking this file's sha256 and `PREREG_PINS.md`'s.
Each job records:
- episode, role, owner, run dir and code tree;
- checkpoint path and sha256, re-hashed. It must equal `inventory.json`'s value. For Zuxing it must also equal his
  metrics.csv `source_sha256`. For T2fix it must equal the eval_ARM row's `source_sha256`;
- training parquet path and sha256, re-hashed and equal to inventory.json's;
- data root and frame references (section 3.6, not included in this copy).

A job with any mismatch is refused and listed; it is never substituted.

Stage B jobs are appended to `manifest_stageB.jsonl` with the same checks plus their trigger evidence: VERDICT.json sha256
and written time, final.json, runin_final.json.

### 3.2 Zuxing copies

Before export, each of his 7 designated checkpoints is copied, model_N plus rsi_N, to `zuxing_ckpts/<run>/epNN/`, with
`SHA256SUMS` and `SOURCE.json` (the t1_best5 pattern).
- Every copy is one row of `COPIES.tsv`: source, destination, bytes, source sha256 = destination sha256, time.
- The copies are what gets exported.
- Nothing else is copied from his folder, and no git command is run there.

### 3.3 Stock-tree protection

- **Collision hulls.** For each job's parquet, compute the CoACD cache keys the way `tmp_analysis/t2fix/cache_keys_t2.py`
  does (read-only, CPU). Every key must already have its file in the stock runtime's `src/flash_chord/assets/.cache/convex`
  (29 files, none newer than 2026-10-12). Otherwise the stock evaluator would write into `code/video_to_data_8c778f9`.
  Such a job gets `hull_missing` and is not evaluated.
- **Tree check.** A listing (path, size, mtime) of `code/video_to_data_8c778f9` and of the kit is taken before the first
  evaluation and compared after every completed job. On any change the driver starts nothing new and reports.

### 3.4 Export once per job

The export runs on CPU, on the eval-lane cores, with `python -m flash_chord.training.ppo.coordinate.export` of the tree
that trained the checkpoint, or of its accepted equivalent:
- **raw, K3fix, K3 easy-clip, seed 7, T2fix:** `code/fc_r2` @ 61fa5d3.
- **Zuxing's checkpoints** (trained in `dev.zuxing.speedup` @ bc3b1bd): `code/nv-v2d_runin` @ b39b69b.
  - A read-only diff at writing shows speedup differs from b39b69b only on the training side: RSI bins / kernels / reset,
    rollout, environment metric reductions, and an ORT stream-sync option in `runtime/actions/sonic.py` whose effect is
    synchronisation only.
  - b39b69b exported speedup-trained checkpoints in t1_best5.
  - Zuxing's own single evaluations (exported with his speedup tree) are reported next to ours as a cross-check.

The rules for each export:
- **Tolerance.** The stock-runtime equivalence check runs at 1e-4. On a tolerance refusal, one export is made at 2e-4 and
  recorded as the fallback. A refusal at 2e-4 marks the job `export_refused`, and it is not evaluated.
- **Technical failure** (anything but a tolerance refusal): retried once.
- **Zuxing load or config failure.** If b39b69b fails on his checkpoint for a non-tolerance reason, the job is
  `export_incompatible` and is reported. There is no fallback to his tree, because running code from his folder could
  write into it.
- **Storage.** The kept file is in `exports/<job>/try<k>/`, and `export.json` names it with its sha256. It is never
  re-exported.

### 3.5 Five evaluations of that same exported file

**Command and setup**
- `scripts/track2/eval_all.py --export no`, with the exported file linked as the checkpoint.
- It runs the kit's `run_policy_evaluation.py` and `evaluate_policy.py` on the untouched stock runtime: frame-0 start,
  explicit reset, no help, world count 1, one job per GPU.
- `--manifest <job data root>/whole_body/soma/track2_manifest.json`.
- Environment: `HUMAN_MOTION_DATA_DIR=<job data root>`, `CUDA_VISIBLE_DEVICES=<g>`, no `CUDA_MPS_*` / `WANDB_*`,
  `PYTHONDONTWRITEBYTECODE=1`.
- Work dir: `runs/<job>/run<k>/try<t>/`. The results JSON and the rollout parquet are kept.

**A run is ok only if all of these hold**
- exit 0, and the report says "evaluated";
- the results JSON and the parquet exist;
- their `checkpoint_sha256` is the exported file's;
- the reference parquet path is the job's training parquet, and its sha256 equals the manifest's;
- start_frame 0, world_count 1, reset explicit, no non-finite world, and environment_valid;
- every run of this folder used one evaluator build (`evaluate_policy.py` 160d24f5..., plus the eval-code digest when the
  report carries one);
- the exported file is unchanged afterwards.

**Retries.** A technical failure is retried, at most 3 tries per run, at least 5 min apart. A completed run is never
repeated, whatever its score.

### 3.6 (not included in this copy)

### 3.7 Scheduling (decides nothing)

- **GPUs.** 0, 1, 2, 5, 6 and 7 only. GPU 4 never. GPU 3 never, because it has no lock file. No GPU with MPS.
- **Locks.** Each evaluation holds a flock on `fc_runs/queue_runin/evallock/gpu<g>.lock` for its whole life. The file is
  opened read-only and never created. This keeps one evaluation per GPU across all watchers.
- **Memory.** At least 4096 MiB + 1536 MiB free before a start.
- **CPU.** Every job runs as `nice -n 10 taskset -c` on the eval lane of its GPU's NUMA node: 48-51,152-155 or
  100-103,204-207. At most 2 of ours per lane and 4 in total. Exports use the same lane cores.
- **Order.**
  - Priorities: P1 CHOSEN, then P2 the remaining RAWREF, then P3 SAME, then P4 DIFF.
  - Within a priority, run 1 of every job comes first, then run 2, and so on.
  - Stage B jobs join their priority when they are added.
- **Signals.** The driver never signals any process, its own children included.
  - An evaluation still running after 2400 s is flagged OVERDUE in the heartbeat and the log.
  - Its GPU gets no new evaluation of ours until it ends. The main session decides what to do.
- **STOP.** `STOP` means no new starts; running jobs finish and are recorded. A restart resumes from the records.
- **Estimated load.** 70 Stage-A jobs x 5 = 350 evaluations of about 2 min each, so about 3-4 h at 4 in parallel. Up to 10
  Stage-B jobs come on top.

### 3.8 Stage B watch

Every 10 min the driver reads, read-only:
- `VERDICT.json`;
- the pending K3fix runs' `final.json`;
- t2watch `best.json`, and the terminal row in `progress.csv`;
- T2fix `final.json` and `runin_final.json`.

The watch ends when all six Stage-B items have been added and evaluated, or on STOP.

## 4. What is reported

**Per job** (own-input):
- the 5 values of AUC, SP-SR, MP-SR and MPPE, with their mean, sd, min and max;
- the run closest to the mean;
- the earlier evaluations (inventory), shown for reference only;
- the export tolerance used, and the status.

**Per episode:** CHOSEN, SELECTED, RAWREF and every candidate side by side, with input labels.

**Compositions (30 episodes)**

| name | what it is |
|---|---|
| RAW-ONLY | the raw final.json checkpoint everywhere |
| K3FULL | PREREG_v3, as VERDICT.json gives it; PENDING maps to raw and is flagged |
| SELECTED | section 2, settled challengers only. This is the headline. |
| SELECTED+P | section 2, including provisional challengers. Report only. |
| ZUXING-LIVE | report only, never selectable. SELECTED with each episode Zuxing still trains replaced by his frozen checkpoint (01 G2, 04 G2, 05 K3fix, 19 G2, 26 K3fix, 28 G2). |

ZUXING-LIVE is score-independent: the substitution follows which runs he is training, not any number.

**For each composition (own-input):**
- **average:** the mean over the episodes of the per-episode 5-eval means;
- **lowest / highest, two ways:**
  - (i) **repeat totals.** T_k = the mean over the 30 episodes of run k, for k = 1..5 in run-index order. Report min_k T_k
    and max_k T_k. Their mean equals the average. This answers "rerun 5 times, highest and lowest".
  - (ii) **per-episode bounds.** The mean of the per-episode minima and the mean of the per-episode maxima. These are
    worst-of-5 and best-of-5 per episode, so pessimistic and optimistic.
- the same statistics for SP-SR, MP-SR and MPPE;
- n per episode. A composition is labelled PARTIAL if any episode lacks 5 ok runs, and episodes keep their PENDING or (P)
  labels.

Own-input composites mix different references, so they are not leaderboard numbers.

**Files**
- Evaluations: `own_input_evals.csv`.
- Selection: `selection.json` and `selection.txt` (from own-input values).
- Summaries: `summary_own_input.{csv,md,json}`.
- Records: `manifest*`, `COPIES.tsv`, `zuxing_ckpts/`, `exports/`, `runs/`, `driver.log`, `heartbeat.json`, `STOP`.

Nothing is packed for submission. Packing a Tier-2 file later is a separate decision. Best-of-k is optimistic; a single
run (closest to the mean) is the honest file.

## 5. Disclosures and caveats

- **Fixed rules.** Each inventory row is included or excluded by the fixed role and designated-checkpoint rules above;
  the starting point (the PREREG_v3 choice), the +0.01 margin and the SP/MP condition were set in the task before this
  file was written; selection reads own-input numbers only.
- **Zuxing's G2** is report-only here.
- **Zuxing's ep05 and ep26 runs** use the K3fix parquets of gate-FAIL episodes, which PREREG_v3 sends to raw. Adopting
  them would need a new rule.
- **Zuxing's runs are early.** They train to 180k-450k with early stop off, and their frozen checkpoints are at 10k-18k
  and provisional, so their numbers will move. Ask him about possible Tier-2 results from the off-server 5090 host.
- **Winner's curse.** Replacement takes the best of up to 3 challengers, which the +0.01 margin offsets only partly. The
  means of the unselected candidates are reported.
- **Noise.** Earlier own-input repeats differ by up to about 0.035 on some episodes, so a +0.01 margin lies within the
  noise. Each run is a single seed.
- **ep04's 0.11036737** is a floor value (the object is not lifted).
- **ep29 raw:** its 1e-4 export failed before, so the 2e-4 fallback is expected.
- **ep08, 20 and 23:** the chosen runs are on the K3 root, byte-identical to K3fix for these episodes. The manifest
  checks this.
- **Zuxing exporter:** his checkpoints are exported with b39b69b, not his training tree (section 3.4).

## 6. Implementation and pins

- **Implementation.**
  - The driver `scripts/t2best5_driver.py` and launcher `scripts/t2best5_launch.sh` are new files, written after this
    one. They implement it unchanged.
  - Patterns, read-only: `scripts/t1best5_driver.py` (export once, 5 evaluations, ok checks, Zuxing copies) and
    `scripts/t2watch_eval.py` (`score_command`, `child_env`, frame-reference map).
- **`PREREG_PINS.md`** (+ `PREREG_PINS.sha256`) is written before the first copy, export or evaluation. It records:
  - their sha256;
  - a dry-run job list, which must equal section 1;
  - the selftest result.

  It may add pins and implementation notes only; it changes no rule.
- **Later deviations** go into `AMENDMENTS.md`, with time and reason, before the affected step. They are disclosed.
- **Hard rules:**
  - new files only, under this folder plus those two scripts;
  - never edit files a running process uses;
  - never signal any process;
  - never GPU 4, never MPS;
  - Zuxing's and Han's folders are read-only: only the 7 listed checkpoint copies are taken, and no git command is run
    there;
  - the kit and `code/video_to_data_8c778f9` are never modified;
  - Python children run with `-B` / `PYTHONDONTWRITEBYTECODE=1`, and kit Python with `-I -B`.

**Pins (sha256)**

pin inventory/inventory.json 08d3c1d13170162847bccbe8be66ef5f63d25026a853a97b3e40833a4e5431c9
pin inventory/inventory_own_input.csv e6092a33147b9aac7b35d88a0b0c876d88d62def2a9d0cd6b27be63c62472b4c
pin inventory/build_inventory.py f9ae201ee26afdf680bf3cd1778b6de4a0bf2c4b7cfed90f2269de16e3ffa100
pin tmp_analysis/tier2/k3full/PREREG_v3.md 0f4fa433f3368072e161d255c18de18bda03668b4095f3958e6411aaec2d3995
pin fc_runs/T2fix/PREREG.md 00a95289fafda40da92002a973c13e944a87dd46501316f25ff7206e223b8c05
pin code/fc_r2/scripts/track2/eval_all.py 657295921e820a464160039ad9a65ba92a683108a99dcb197290ae4147363197
pin kit flash_chord/scripts/evaluate_policy.py 160d24f5fbeb70ccd569ca4192056f1892b2a9f2999066db9d8279461648fdf1
pin kit tools/pack_submission.py 47a4589574ff8284ba4376ef31ac5548c8fe87ead753be2d3b43d6c87041e5be
pin scripts/t1best5_driver.py 83e577de2c8acce55d1c02fc4eb7d7ad30e9a9b2f2599b40f7b5f2c9aab842e5
pin scripts/t2watch_eval.py 290aed481cdf28b1516af1243ef58184e926a7c705e70f684603a3c1e98227a5
commit code/fc_r2 61fa5d31ff8054bf1e5cf5375ef0957e54d2729b
commit code/nv-v2d_runin b39b69be6c2f561c0123c6aeb8cd13f8b9d542e7
commit code/video_to_data_8c778f9 8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef
info VERDICT.json at writing 6e97b33b796be503abad2cfea0e53fe994dc60194140fb7da56b40ad3018dfd9 (changes every <= 30 min; not a pin)
