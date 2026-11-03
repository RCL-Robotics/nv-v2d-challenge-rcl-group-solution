# Tier-2 consensus selection, round 9 (t2_consensus_r9_20261103): pre-registration

**When.** Written 2026-11-03 from about 22:20 CST, after both check reports on Zuxing's hand-over were available (section
9.2) and before the round-9 pool freeze and before any round-9 export, evaluation, score or selection. At the writing,
this folder held only development copies of the round-9 scripts and their plumbing outputs (a records build, a pool dry
run, a trial pool and a plan-only evaluation check, all in `dev/`). The pinned build steps of section 10 (B2-B4) re-create
them with the final scripts. Every later step refuses to run if this file's sha256 differs from `PREREG_R9.sha256` and
from the value hard-coded in the round-9 scripts.

**User request** (relayed by the main session, 2026-11-03): Zuxing reported that all 30 G2 files are now final and placed
in `reports/tier2-G2/`; the user asked "can you double check we can reproduce its checkpoints and do evaluation on it?"
and then "then give me the best tier2?". Round 9 was proposed with round 8's rule; incumbents = the round-8 picks;
candidates = Zuxing's final files. For GPUs the user chose "Share GPUs with ablation (g)". Update, about 20:40 CST: the
hand-over has two final sets, `reports/tier2-G2` (30 files) and `reports/tier2-K3fix` (18 files); round 9 starts now
(competition deadline 2026-11-04 17:00 EST = 2026-11-05 06:00 CST).

**Basis.** Selection uses only own-input and consensus-B scores:
- the official evaluator's metrics of each policy against its own Tier-2-derived input (own-input scores);
- the variant-B consensus scores against round 1's frozen consensus references refs/B (consensus-B scores);
- the own-input validation gate;
- training-run metadata: iterations, file times, checksums, the embedded training configuration and the first line of
  each training log.

**Standing rules**
- **Zuxing's folder is read-only.** Nothing there is written, moved or deleted. No git command is run there and none of
  his scripts. Every file of his that round 9 evaluates or records is copied into this folder, with sha256 before == copy
  == after (`COPIES_r9.tsv`, `TELL_ZUXING_r9.txt`). Record copies of text files hold only the lines round 9 uses (2.5).
  His selection records (README, candidate and selection tables, evaluation tables) are never opened by the round-9
  scripts; his selection rule enters only through the two check reports (9.2).
- **What the round-9 scripts never do.** They never commit, push, pack or upload to Kaggle, and they never signal a
  process. Packing the round-9 file and any commit are separate steps that need the user's approval.
- **GPUs (section 3.5, D4).** GPUs 3, 5, 6 and 7 while `nvidia-smi` lists no compute process there; GPUs 0, 1, 2 and 4
  also while every listed compute process there is one of our own ablation jobs; plain, one evaluation of ours per GPU;
  never a GPU carrying any other process.
- **Files.** New files only. Kit Python is always run with `-B`. The kit and the stock runtime `code/video_to_data_8c778f9`
  are only read.

## 0. Relation to round 8

- **Round 8 is final and frozen.**
  - `SELECTION_R8.sha256` e48cc878...; `PREREG_R8.md` bb912c57... (with `AMENDMENTS_R8.md` A1, GPUs only).
  - Its result: episode 13 replaced (`ep13_w312runinws_90000`), every other episode kept its round-7 pick;
    R8-CLOSEST = trial 7; the packed file `T2_R8_closest_trial07_8387874.csv` (commit 8387874).
  - Round 9 reads round 8's records only through round 8's verified loaders (`r8_select.py` 735149ac,
    `r8_eval.py` 5dcf6a82, `r8_pool.py` bebeb013, loaded from their verified bytes). It never re-runs, re-scores or edits
    them.
- **The rule.** Round 9 is PREREG_R8 sections 1-7 (snapshot rule, pool roles, evaluation protocol, eligibility, phase-S
  finalist, phase-F decision, closest-to-mean trial, orchestration), unchanged except the deltas **D1-D6** below and the
  consequences **C1-C7**, each forced by the deltas or by facts after round 8.
- **Rules basis (unchanged):**
  - challenge §2.8.c: every candidate was trained on a Tier-2-derived input;
  - §2.9.d: one checkpoint per episode, identical digest within an episode;
  - §2.9.b: the upload file is the trial closest to the mean of 10 full trials, never a best-of;
  - Foundational 4.b: a frozen, algorithmic rule.

### Deltas (D1-D6)

- **D1 Incumbents.** Each episode's incumbent is its round-8 pick (S8 = the R8 file, trial 7), with its recorded
  phase-F runs and bar. It is not re-run.
  - A round-7 pick kept in round 8 (29 episodes): its round-7 phase-F runs, round 8's m_P_frozen, m_P_fresh and bar
    (from `R7_RECORDS_R8.json`, recomputed and asserted equal).
  - The round-8 replacement (episode 13, `ep13_w312runinws_90000`): its 10 round-8 phase-F runs; m_P_frozen = the
    float64 mean of its 10 round-8 phase-F consensus-B AUCs (== `SELECTION_R8.json` m_F, asserted); m_P_fresh and bar
    from the same runs (round 7's `decide_f`), so bar = m_P_frozen + 0.01. This is round 8's own rule for a
    replacement that became the incumbent (PREREG_R8 4).
- **D2 Candidate pool.** The candidates are:
  - **(a)** every file present at T_c9 in Zuxing's `reports/tier2-G2/` (30) and `reports/tier2-K3fix/` (18). This holds
    because both check reports find his selection rules to be based on his own-input evaluations only (9.2). Otherwise
    the fallback would apply to that folder: each such file's run's latest checkpoint present at T_c9, verified from the
    run's records on this server (`REPORT_MODE = latest` in `r9_pool.py`). For each file, round 9 records whether it is
    its run's final checkpoint at T_c9 (2.1). Files that are still intermediate are evaluated as they are and marked
    "not final". Nothing is added after T_c9.
  - **(b)** round 8's mechanical rule (round 7's R1-R4 with round 8's R1c copy runs and the 5090 grid restriction)
    re-applied at T_c9 to the runs that are new or have more iterations than at round 8's snapshot (2026-11-03 12:00:00
    CST, `inputs_r8/runs_r8.json`).
  - **Reuse.** A candidate whose file sha256, or for a relocated copy the sha256 of its tensor data block, already has
    round-8 phase-S results reuses them (role reuse8). Round 8's own reuse rule continues: one with round-7 phase-S
    results reuses those (role reuse7).
  - **Barred.** Round 8's barred files stay barred (2.4).
- **D3 Snapshot.** T_c9 = the time the pool freeze starts (after the check reports), recorded in `POOL_R9.json`.
  Nothing is added later. The freeze runs at once when the orchestrator is launched; it does not wait for any training.
- **D4 GPUs (section 3.5).** Zuxing released GPUs 3, 5, 6 and 7 (2026-11-03 20:27 CST).
  - An evaluation may start on GPU 3, 5, 6 or 7 when `nvidia-smi` lists no compute process there and no MPS server,
    client or control daemon names it (round 8's rule).
  - Per the user's decision "Share GPUs with ablation (g)", GPUs 0, 1, 2 and 4 may also be used while our own ablation
    Genesis MPS jobs run there: at most one round-9 evaluation per GPU, as a plain process (no MPS variable). Never a
    GPU that carries another process.
  - This sharing is a difference from round 8, disclosed in 9.1.
- **D5 Deadlines (section 7.3).** Phase S must finish by **2026-11-04 12:00:00 CST**; phase F by **2026-11-04 20:00:00
  CST**. Otherwise the unfinished episodes keep their round-8 pick (7.3 says how).
- **D6 Records.** Round 9 computes no score other than the own-input and consensus-B scores above. Its records state
  that basis ("selection uses only own-input and consensus-B scores") and hold no other comparison.

### Consequences (C1-C7)

- **C1 Barred files (2.4).** Round 8's barred files are its `R7_RECORDS_R8.json` phase-F participants and its carried
  barred pair. Round 8's phase-F participants other than its pick are added, as round 8's C3 did for round 7: the
  round-8 incumbent it replaced (`ep13_k3fix_final39254`; its round-7 phase-F runs fail the round-9 test) and its
  finalists not picked (none). The records build asserts that no barred file passes the round-9 test on its recorded
  phase-F runs.
- **C2 One finalist per episode.** Every episode has an incumbent, so round 7's incumbent-episode rule (one finalist,
  replacement test) applies to all 30 episodes, as in round 8.
- **C3 Round 8's watcher-best route is not part of round 9's pool.** D2 defines the pool: Zuxing's files take the place
  of an own-input nomination, and (b) re-applies only round 8's mechanical rule.
- **C4 Report files trained in our own runs.** Some files in `reports/tier2-K3fix/` were trained in our runs (their
  embedded output_dir is under `fc_runs/`). Such a file is evaluated as our original in our run folder, which must be
  byte-identical to the report file, and our run must be in round 6's eligible list (round 7's R1).
- **C5 Lanes and cores (3.5).** With eight usable GPUs, the NUMA1 eval lane allows 4 evaluations at once on 16 logical
  CPUs instead of 2 on 8. The NUMA0 lane keeps round 8's 2 on 8. Each evaluation keeps round 8's 4 logical CPUs (9.1).
- **C6 Deadlines, failure handling and one evaluation driver at a time (section 7)** are round 8's, with D5's times.
- **C7 Job names.** A report file's new screen job is `epNN_zxrep{g2,k3fix}_<iteration>`. A mechanical pick keeps round 8's
  label scheme. A name that exists in round 7's or round 8's records gets the suffix `_r9`. Reuse and barred rows keep
  their earlier job names.

## 1. Snapshot and inputs

- **T_c9** = the start of `r9_pool.py freeze`, which `r9_driver.py` runs at once when launched (D3).
- **A model file counts iff all of these hold** (round 8's rule):
  - its mtime <= T_c9 - 300 s (round 7's R3);
  - its ctime <= T_c9, so a file copied in after T_c9 never counts;
  - for a copy run, it is listed with a matching sha256 in the copy's checksum file.
- **Round-8 inputs**, read through round 8's own verified loaders (they check round 8's and round 7's whole pin chains):
  `SELECTION_R8.json` and every file listed in `SELECTION_R8.sha256` (e48cc878...), `POOL_R8.json`,
  `inputs_r8/runs_r8.json`, `inputs_r8/R7_RECORDS_R8.json`, round 8's phase-S and phase-F scores, gates, manifests and
  rollout lists.
- `inputs_r9/R8_RECORDS_R9.json` holds the 30 incumbents (section 4), the barred files per episode, and the identities
  and phase-S records of round 8's 89 and round 7's 399 phase-S candidates. It is built by `r9_select.py r8records`
  (step B2) and pinned in `PREREG_R9_PINS.md`.

## 2. The candidate pool

The rule is implemented by `r9_pool.py`. It loads round 8's `r8_pool.py` (bebeb013...) from its verified bytes, and
through it round 7's resolver `scripts/t2cons_r7_pool.py` (7a08fa6d...), and calls their helpers unchanged.

### 2.1 Report files (D2 a)

Every file `reports/tier2-G2/epNN-tier2-G2.safetensors` and `reports/tier2-K3fix/epNN-tier2-K3fix.safetensors` present
and settled at T_c9 (section 1) is a candidate iff all of these hold:
1. **Format and iteration.** learner_state; its iteration >= 10,000.
2. **Data root (R2).** The embedded `task.parquet` lies in a Tier-2-derived data root (raw, K3, K3fix, K3CA, G2),
   `reference.parquet` is absent or the same, and the same root and sequence exist on this server.
3. **Episode.** The sequence maps to the file name's episode in this server's manifest.
4. **Host.** The embedded output_dir names the training host: this server (Zuxing's `runs/` or our `fc_runs/`), the
   5090 (`/mnt/gfs/home/zuxinglu/workspace/`) or a robolab host (`/home/zuxinglu/workspace/`).
5. **Training-commit evidence** (round 8's C2 sources):
   - this server, Zuxing's run: the first `train_t2:` line of `logs/<group>/<ep>.log`;
   - this server, our run (C4): our original in the run folder == the report file (sha256) and the run is in round 6's
     eligible list; export tree fc_r2;
   - another host: the first `train_t2:` line of a copy run of the same run (embedded output_dir) in round 7's / round
     8's copy folders, or round 7's 5090 MANIFEST commit; else, for the 5090, its run's row in the hand-over evidence file
     `evidence/5090.txt` ("training-log first lines"), then the evidence file's HEAD from refs; for a robolab host
     without git, the HEAD from refs of the host evidence file that lists the run.

**Final status at T_c9 (recorded, never a condition).** `final_at_t_c` is "yes" iff no later checkpoint of the same run
is present on this server at T_c9 and the file is the run's planned end (its iteration equals the embedded
`training.max_iterations`, or the run's `final.json` names it); otherwise "no", with the evidence. The verdict of the
check reports is recorded next to it (`check_report_final`).

**What a candidate evaluates.** A file trained on this server by Zuxing: our byte copy. A file trained on another host:
our byte copy, relocated metadata-only to this server's parquet (2.5). Our own run's file: our original (C4).

### 2.2 Folders that are not walked

As round 8 (2.2), plus Zuxing's 2026-11-03 bundles `2026-11-03-tier2-g2-final-ckpts`, `2026-11-03-tier2-k3fix-final-ckpts`
(their files enter only through `reports/`), `2026-11-03-tier2-k3fix-ownsel` and `2026-11-03-tier2-g2-reports-eval30`, and
the `2026-11-02-tier2-g2-ownsel` bundle (round 8 read only its watcher files, which round 9 does not use). These are
never entered by the round-9 scripts.

### 2.3 Round 8's mechanical rule at T_c9 (D2 b)

Round 8's universe is re-scanned at T_c9: round 7's R1 pro6000 run directories and round 8's R1c copy runs, with every
eligibility condition of round 8 (2.1-2.4, including commit evidence and the 5090 grid restriction). A run is
**new** (not among round 8's eligible runs), **trained further** (L greater than at round 8's snapshot) or **unchanged**.
Round 7's R4 picks (K = 3: targets 100 / 83 / 67 / 50 % of L, nearest saved iteration >= L/2, ties the larger) are
candidates only for new and trained-further runs. Round 8's "one run, several copies" rule applies.

### 2.4 Roles

In the deterministic order (episode, owner, report files before run files, path, iteration), with identity = file
sha256, else the sha256 of the tensor data block:
- **incumbent:** the episode's round-9 incumbent (its round-8 pick). It is not screened.
- **barred (C1):** a round-8 barred file of the episode, or a round-8 phase-F participant other than its pick. It is
  never a finalist and never re-evaluated. If it has recorded phase-S results, it is listed in the phase-S table.
- **reuse8:** a round-8 phase-S candidate of the same episode with the same input (this server's training parquet
  sha256 and, for non-raw inputs, the frame-reference sha256). Its round-8 phase-S runs, scores and export are reused;
  it keeps its round-8 job name and is not re-evaluated in phase S.
- **reuse7:** the same with a round-7 phase-S candidate (round 8's reuse7 rule), with its round-7 / round-6 export.
  Round-8 records are matched first.
- **duplicate:** the same file (sha256, or the same tensor data block) selected earlier in round 9. Routes are merged.
- **screen:** every other file. It gets one export and 3 fresh runs.
- **refused:** the checkpoint's parquet differs from the run's, or the file differs from its checksum line.

### 2.5 Copies and relocation

- **Copies (round 8's 2.8).** Each new screen file of Zuxing's (report file or run file, and the run file's rsi of the
  same iteration) is byte-copied into `zuxing_copies_r9/` (path relative to his folder), sha256 before == copy ==
  after. So are, for the runs used, the hydra yaml, `final.json`, `MANIFEST.*` and checksum files (record excerpts of
  the lines round 9 uses), the host evidence files used, and a one-line excerpt of each training log's first `train_t2:`
  line. Lines that round 7's token guard refuses are left out.
- **Relocation.** A screen file trained on another host (5090, robolab) is relocated metadata-only into
  `relocated_r9/`, with round 5's pinned `relocate_bytes` / `verify_pair` (t2cons_r5_relocate.py 09a6cb8e) as in rounds
  2-8: exactly `task.parquet` and `reference.parquet` change, to this server's same-named file of the same data root;
  every tensor byte and every other metadata field is unchanged. Each relocation is re-verified with round 7's
  `verify_row` at the freeze and at every manifest build. The relocated file is the evaluated input; the pool sha256
  stays the original's.

### 2.6 Outputs of the freeze (all mode 444)

- `POOL_R9.json` (every row with its role, routes, source, iteration, sha256, input, parquet + sha256, frame reference +
  sha256, commit, final status, copies, relocation, earlier-round links);
- `inputs_r9/`: `pool_r9.csv`, `runs_r9.{json,csv}` (every directory scanned, eligible or not with the reason, status
  against round 8), `reports_r9.{json,csv}` (every report file with eligibility, commit evidence and final status),
  `TRACKS_R9.sha256`, `ZUXING_CODE_R9.tsv`, `resolve_r9.log`;
- `COPIES_r9.tsv`, `TELL_ZUXING_r9.txt`, `RELOCATION_R9.{tsv,json,sha256}`;
- **`POOL_R9.sha256` last.** It lists all of the above, `R8_RECORDS_R9.json`, every copy and every relocated file.

## 3. Evaluations

### 3.1 Protocol: round 7's 3.1, unchanged

**Driver.** `r9_eval.py` loads round 7's driver `scripts/t2cons_r7_eval.py` (1d66a2db...) from its verified bytes and
re-binds its folder names to this folder; its code runs unchanged, and through it `t2cons_eval.py` (2cf42786) and
`t2best5_driver.py` (24b9af39). It also loads round 8's `r8_eval.py` (5dcf6a82...), bound to round 8's folder, only to
rebuild reuse8 / reuse7 finalists from round 8's records.

**Export and evaluation.**
- **Export trees:** `code/fc_r2` @ 61fa5d3 for ours, `code/nv-v2d_runin` @ b39b69b for Zuxing's.
- **Tolerance:** 1e-4, with one fallback at 2e-4 on a tolerance refusal. The export is stock verified and
  `source_sha256` == the evaluated input.
- **Evaluation:** `scripts/track2/eval_all.py` (657295921e82) with `--export no --world-count 1` on the untouched stock
  runtime (`evaluate_policy.py` 160d24f5). Frame-0 start, explicit reset, no help. The reference is the job's own
  training parquet.

**ok checks, retries and integrity are round 7's.** Up to 3 tries per run, at least 5 minutes apart; a completed
evaluation is never re-run; a failed run counts as not ok; a non-tolerance export refusal ends the job; a change of the
kit or the stock tree halts new starts. **Earlier exports** of reuse8 / reuse7 finalists are re-hashed and never
re-exported.

### 3.2 Relocated jobs

A screen job from another host takes round 7's relocation branch: its `RELOCATION_R9.tsv` row is re-verified at every
manifest build, and the expected sha256 is the relocated file's.

### 3.3 Phase S: 3 runs per new screen candidate

- **Jobs:** the `screen` rows of `POOL_R9.json`. Each gets one export (`exports/<job>/`) and 3 official runs
  (`runs_s/<job>/run<k>/`).
- **Order (decides nothing):** exports in (episode, job) order on CPU, then evaluations run-major.
- **reuse8 / reuse7 candidates** contribute their recorded phase-S results. They are not re-evaluated or re-scored.

### 3.4 Phase F: 10 runs per finalist

- **Jobs:** the finalists of `final_r9.json` (written and locked by `SELECTION_S9.sha256` before phase F starts), at
  most one per episode, 10 fresh official runs each (`runs_f/<job>/run<t>/`), trial-major order.
- **Exports:** a new screen finalist uses its phase-S export; a reuse8 finalist is rebuilt by round 8's driver from
  round 8's records and uses its round-8 export; a reuse7 finalist is rebuilt by round 8's driver (and through it round
  7's) and uses its round-7 / round-6 export.
- **Incumbents are not re-run.** Their recorded phase-F runs are used (D1). Without a finalist, phase F has no job.

### 3.5 GPUs and lanes (D4, C5)

- **Exclusive GPUs 3, 5, 6, 7.** An evaluation starts there only while `nvidia-smi --query-compute-apps` lists no
  compute process on it and no MPS server, client or control daemon names it (round 8's rule).
- **Shared GPUs 0, 1, 2, 4.** An evaluation may also start there while every listed compute process on it is one of our
  own ablation jobs, and every MPS control daemon naming it is ours. Ours means: an MPS server or client whose
  `CUDA_MPS_PIPE_DIRECTORY` lies under `ch_runs/ablation_20261031/mps/`, or a process running the ablation code tree
  `code/Coordinates_hand_ablation/`. Any other process on the GPU makes it unusable. An MPS control daemon on the default
  pipe directory (where a plain process would become its client) makes every GPU unusable. All GPUs are in the Default
  compute mode (checked at the build).
- **Every GPU:** plain (no `CUDA_MPS_*` variable in the evaluation's environment); at most one evaluation of ours per
  GPU; at least 4096 + 1536 MiB free; round 7's lane locks (our `evallock/gpu<g>.lock` plus every shared lane-lock file
  of that GPU, with round 7's and round 8's `evallock/` added to the shared list); a 10 s cooldown per GPU; `nice -n 10`.
  Nothing of anyone else's is touched or signalled. A GPU with no listed process is preferred (round 7's pick order).
- **Lanes (C5).** At most 2 evaluations at once on the NUMA0 eval lane (GPUs 0-2; `taskset` cores 48-51,152-155, round
  8's) and 4 on the NUMA1 eval lane (GPUs 3-7; cores 96-103,200-207, round 8's 100-103,204-207 plus 96-99,200-203). At
  most 6 at once (`--max-parallel 6`). At the build, the load average was about 9 on 208 logical CPUs.
- **Running evaluations:** one past 2400 s is only flagged OVERDUE. Nothing is ever signalled. `STOP` in this folder
  stops new starts.

### 3.6 Seeds (records only)

As round 8: there is no selectable seed on the official path. Seeds are recorded per run in `seeds.json` and
`seeds_{s9,f9}.tsv`.

## 4. Incumbents (D1): the round-8 picks, with their recorded phase-F runs

**bar = max(m_P_frozen, m_P_fresh) + 0.01** is the phase-F bar; **bar_S = m_P_frozen + 0.01** is the phase-S threshold.
The development build of `R8_RECORDS_R9.json` gave this table (round 8's values; 6 decimals shown, decisions use the
exact floats in `R8_RECORDS_R9.json`):

| ep | incumbent (round-8 pick) | input | status | recorded runs | m_P_frozen | m_P_fresh | bar |
|---|---|---|---|---|---:|---:|---:|
| 00 | `ep00_z5k3fix_90000` | k3fix | round-7 pick kept in round 8 | round-7 F | 0.719120 | 0.719120 | 0.729120 |
| 01 | `ep01_k3fix_final44922` | k3fix | round-7 pick kept in round 8 | round-7 F | 0.171319 | 0.168566 | 0.181319 |
| 02 | `ep02_k3fix_final34093` | k3fix | round-7 pick kept in round 8 | round-7 F | 0.868138 | 0.872717 | 0.882717 |
| 03 | `ep03_k3fix_final27068` | k3fix | round-7 pick kept in round 8 | round-7 F | 0.921527 | 0.907122 | 0.931527 |
| 04 | `ep04_zxg2_450000` | g2 | round-7 pick kept in round 8 | round-7 F | 0.429436 | 0.429436 | 0.439436 |
| 05 | `ep05_zx_k3fix_88000` | k3fix | round-7 pick kept in round 8 | round-7 F | 0.578153 | 0.579742 | 0.589742 |
| 06 | `ep06_k3fix_final28420` | k3fix | round-7 pick kept in round 8 | round-7 F | 0.797276 | 0.798993 | 0.808993 |
| 07 | `ep07_k3fix_final18529` | k3fix | round-7 pick kept in round 8 | round-7 F | 0.920452 | 0.894092 | 0.930452 |
| 08 | `ep08_k3easy_final17496` | k3 | round-7 pick kept in round 8 | round-7 F | 0.596363 | 0.595904 | 0.606363 |
| 09 | `ep09_raw_final39328` | raw | round-7 pick kept in round 8 | round-7 F | 0.760460 | 0.778527 | 0.788527 |
| 10 | `ep10_raw_final34267` | raw | round-7 pick kept in round 8 | round-7 F | 0.412469 | 0.410682 | 0.422469 |
| 11 | `ep11_raw_24000` | raw | round-7 pick kept in round 8 | round-7 F | 0.807319 | 0.802563 | 0.817319 |
| 12 | `ep12_k3fix_final23178` | k3fix | round-7 pick kept in round 8 | round-7 F | 0.543911 | 0.543741 | 0.553911 |
| 13 | `ep13_w312runinws_90000` | g2 | round-8 replacement | round-8 F | 0.700230 | 0.700230 | 0.710230 |
| 14 | `ep14_k3fix_final44922` | k3fix | round-7 pick kept in round 8 | round-7 F | 0.330614 | 0.335807 | 0.345807 |
| 15 | `ep15_raw_final44922` | raw | round-7 pick kept in round 8 | round-7 F | 0.729895 | 0.734423 | 0.744423 |
| 16 | `ep16_z5a5_120000` | g2 | round-7 pick kept in round 8 | round-7 F | 0.951297 | 0.951297 | 0.961297 |
| 17 | `ep17_z5r2_61500` | k3fix | round-7 pick kept in round 8 | round-7 F | 0.659403 | 0.659403 | 0.669403 |
| 18 | `ep18_k3fix_final18911` | k3fix | round-7 pick kept in round 8 | round-7 F | 0.419129 | 0.421081 | 0.431081 |
| 19 | `ep19_z5g2hardc2_360000` | g2 | round-7 pick kept in round 8 | round-7 F | 0.581095 | 0.581095 | 0.591095 |
| 20 | `ep20_z5fill_90000` | k3fix | round-7 pick kept in round 8 | round-7 F | 0.872502 | 0.872502 | 0.882502 |
| 21 | `ep21_z5fillc_282500` | g2 | round-7 pick kept in round 8 | round-7 F | 0.494543 | 0.494543 | 0.504543 |
| 22 | `ep22_z5k3fix_89500` | k3fix | round-7 pick kept in round 8 | round-7 F | 0.757867 | 0.757867 | 0.767867 |
| 23 | `ep23_raw_final20023` | raw | round-7 pick kept in round 8 | round-7 F | 0.902867 | 0.903356 | 0.913356 |
| 24 | `ep24_z5q_150000` | g2 | round-7 pick kept in round 8 | round-7 F | 0.905657 | 0.905657 | 0.915657 |
| 25 | `ep25_z5a5c_280500` | k3fix | round-7 pick kept in round 8 | round-7 F | 0.878012 | 0.878012 | 0.888012 |
| 26 | `ep26_zx_k3fix_92000` | k3fix | round-7 pick kept in round 8 | round-7 F | 0.731484 | 0.731605 | 0.741605 |
| 27 | `ep27_k3fix_final20863` | k3fix | round-7 pick kept in round 8 | round-7 F | 0.897854 | 0.905354 | 0.915354 |
| 28 | `ep28_z5q_180000` | g2 | round-7 pick kept in round 8 | round-7 F | 0.150076 | 0.150076 | 0.160076 |
| 29 | `ep29_k3fix_final17358` | k3fix | round-7 pick kept in round 8 | round-7 F | 0.505675 | 0.504416 | 0.515675 |

**Barred files (C1).** Round 8's barred list (round 7's phase-F participants: the replaced round-7 incumbents of
episodes 04, 17, 21 and 24, the round-7 finalists not picked, and the carried pair ep12_raw_37575 and ep13_raw_44922),
plus `ep13_k3fix_final39254`, the round-8 incumbent replaced in round 8. The development build listed 15 barred files.

## 5. Scores and selection (selection uses only own-input and consensus-B scores)

### 5.1 Scores and gate: round 7's 5.1, unchanged

Every round-9 run gets the variant-B consensus scores (pinned round-1 functions: `rollout_on_grid` against the job's own
input reference, `metrics_vs` with `align_initial_pose=False`, 500 released-mesh vertices, seed 0, float64, against
round 1's frozen `refs/B`, build.json 9646de2d) and the own-input validation gate (`auc_seed` within [min_s - 0.02,
max_s + 0.02] over vertex seeds 0..9; |dMPPE| <= 0.5 cm; SP-SR and MP-SR equal). The rollout list of each phase is
frozen in `inputs_r9/ROLLOUTS_{S9,F9}.sha256` before the first score. floor = the number of runs with own-input SP-SR = 1
AND MP-SR = 1.

### 5.2 Scorer

`r9_select.py` loads round 8's scorer `r8_select.py` (735149ac...) from its verified bytes, and through it round 7's
scorer `scripts/t2cons_r7_consensus.py` (e7545dfa...); it calls `s_row`, `select_s`, `decide_f`, `trial_rows`,
`closest_trial`, `runinfo_of`, `s_info` and the pinned round-1 / round-2 functions unchanged. Its gate and score
commands are round 8's loops over round 9's runs. Its selftest (synthetic) is pinned in `PREREG_R9_PINS.md` before it
scores anything.

### 5.3 Phase-S eligibility (round 7's 5.3)

A candidate is **S-eligible** iff its export is ok, it has 3 ok official runs, none flagged by the gate, and floor_S >= 2.
cons_S = the float64 mean of its 3 consensus-B AUCs. A reuse8 / reuse7 candidate is judged on its recorded phase-S
record. Ties everywhere: lower population variance of the 3 consensus AUCs, then lower mean consensus MPPE, then job name.

### 5.4 Phase-S selection

- **Which episodes.** An episode is evaluated iff it has at least one candidate with role screen, reuse8 or reuse7.
- **Finalist.** The S-eligible candidate with the highest cons_S among those with **cons_S >= m_P_frozen + 0.01**
  (exact floats). Barred candidates are never finalists. If none qualifies, the round-8 pick stays.
- **Outputs:** `scores_s9/`, `selection_s9.{json,txt,md}`, `selection_s9_candidates.csv`, `final_r9.json` and
  **`SELECTION_S9.sha256` last**.

### 5.5 Phase-F decision

m_F(f) = the float64 mean of f's 10 phase-F consensus-B AUCs. **f replaces the round-8 pick iff** (a) f has 10 ok,
non-flagged phase-F runs; (b) own-input SP-SR = MP-SR = 1 in >= 8 of them; (c) **m_F(f) >= max(m_P_frozen, m_P_fresh) +
0.01** (exact floats; m_P_fresh from the incumbent's recorded phase-F runs). Otherwise the round-8 pick stays.

### 5.6 The set S9 and R9-CLOSEST

- **S9** = per episode the 5.5 result.
- **Complete trials.** Trial t is complete iff every episode's job has an ok, non-flagged run t: a kept round-8 pick uses
  its recorded phase-F run t (round-7 runs, or round-8 runs for episode 13); a replacement uses its round-9 run t.
- **R9-CLOSEST** = the complete trial whose 30-episode mean own-input AUC is closest to the mean of that quantity over the
  complete trials. Ties: smaller distance of the consensus-B mean to its mean, then lower t. Never a best-of.
- **Consistency check.** If no episode changes, S9 == S8 and R9-CLOSEST must be trial 7, the R8 file's trial. The
  scorer refuses otherwise.
- **Outputs:** `SELECTION_R9.json`, `selection_r9.md`, `selection_r9_episodes.csv`, `summary_r9_trials.csv`;
  **`SELECTION_R9.sha256` last**. From then, round 9 is final.

### 5.7 Nothing else enters

No other score, no manual override, no comparison of own-input AUC across different inputs.

## 6. After round 9 (not part of the round-9 scripts)

- **If no episode changed,** the R8 file remains the file: same picks and same trial.
- **If episodes changed,** packing R9-CLOSEST runs separately with round 8's pack procedure, adapted and pinned first. The
  same holds for any submission commit (new branch only, never main, never force) and any upload. Each needs the user's
  approval.

## 7. Orchestration, deadlines, failure handling (C6)

1. **Driver.** `r9_driver.py`, started by `r9_launch.sh`, runs: `r9_pool.py freeze` at once (T_c9) and `r9_pool.py
   verify`; `r9_eval.py run --phase screen`; `r9_select.py check / gate / score / select --phase S`; if `final_r9.json`
   names a finalist, `r9_eval.py run --phase final` and `check / gate / score / select --phase F`, else `select --phase F`
   directly. Each step is resumable and recognised by its output; a finished step is never repeated. It writes
   `r9_driver.log` and `R9_DRIVER_HEARTBEAT.json`; the evaluation driver writes `eval_r9_heartbeat_{s9,f9}.json`.
2. **GPUs busy.** Exports start at once on CPU; evaluations wait (3.5).
3. **Deadlines (D5).**
   - **Phase-S cut-off: 2026-11-04 12:00:00 CST.** If phase S has not finished, the driver creates `STOP` and
     `S_CUTOFF.json`, running evaluations finish and are recorded, and `r9_select.py keep-r8` writes the SELECTION_R9
     outputs with every episode keeping its round-8 pick. As in round 8, phase S is never selected from an incomplete
     screening (its run-major order would favour some candidates), so at this cut-off every episode counts as unfinished.
   - **Phase-F deadline: 2026-11-04 20:00:00 CST.** If phase F has not finished, the driver creates `STOP` and
     `DEADLINE_F.json`, and running evaluations finish and are recorded. A finalist without 10 ok runs fails 5.5 (a), so
     its episode keeps the round-8 pick. The other episodes are decided normally.
4. **Failures (round 8's 7.4).** Evaluation failures and retries follow 3.1. An evaluation driver that exits early for
   another reason is restarted after 10 minutes, at most 6 times; after that phase S is handled as at the cut-off and
   phase F halts. A pin / preflight / integrity failure, a failed freeze and a failed scorer step halt the driver; a
   halted driver selects nothing further and the R8 file stays in force.
5. **STOP by anyone else** stops new evaluations; the driver ends at the next step boundary and selects nothing.
6. **One evaluation driver at a time** (round 8's 7.6): read from /proc, never signalled; `keep-r8` and the scorer refuse
   while one is alive; `r9_launch.sh` refuses to start an orchestrator while one is alive.

## 8. Budget

- **Phase S.** The development pool resolution of 2026-11-03 22:16:57 CST (snapshot then) found 48 eligible report
  files and 4 trained-further runs (16 mechanical picks): 34 new screen candidates (102 evaluations), 10 reuse8, 13
  reuse7, 3 barred and 4 incumbent rows; 28 of 30 episodes are evaluated (ep16 and ep19 have no new candidate).
- **Phase F.** At most 30 finalists x 10 runs.
- **Wall time.** About 23 s per evaluation on a free GPU (the G2 check), at most 6 at once; the shared GPUs run slower
  next to the MPS groups. Expect under 1 h for phase S and under 1 h for phase F once launched.

## 9. Disclosures

1. **Differences from round 8 in how runs are made.** Round 9 may evaluate on GPUs 0, 1, 2 and 4 next to our ablation's
   MPS groups (plain processes, which can run slower there) and on GPU 3 and 5, which round 8 did not use; the NUMA1 lane
   runs up to 4 evaluations on 16 logical CPUs. All eight GPUs are the same model (RTX PRO 6000 Blackwell Server
   Edition, driver 580.82.09) with the same stock runtime and integrity baseline; there is no selectable seed (3.6);
   each run records its GPU index and UUID. Phase F compares a finalist's round-9 runs with the incumbent's recorded
   runs (round 7: GPUs 6/7; round 8: GPUs 0/4/6/7), made on different days.
2. **What the writer of this file saw.**
   - **Check reports.** `ZX_G2_30_CHECK_20261103.md` and `ZX_K3FIX_18_AND_RULES_CHECK_20261103.md`. They state Zuxing's
     rules (see 9.3) and per-file provenance, final status and export status. The G2 report also lists his own-input
     AUC means (5 evaluations each) for the candidate runs of 5 episodes (01, 13, 19, 21, 28) to show his picks are the
     own-input argmax, and three own-input smoke evaluations (ep04, ep10, ep11). The K3fix report lists his own-input AUC means for the runs of its 2 tie-break episodes (09, 14). These values were read
     after D1-D6 were set by the request and did not shape this rule.
   - **Round-8 records:** round 8's noisy-only records (its incumbent table, `selection_r8.md`, `final_r8.json`).
   - **Development runs:** the round-8 records build and its incumbent table, the pool dry run's candidate lists, the
     trial pool and the plan-only evaluation check. No round-9 own-input or consensus-B value exists or was read.
3. **Zuxing's selection rules (from the check reports).**
   - **G2:** for each episode, the last checkpoint of its training run (a continuation counts as the same run); where an
     episode has runs with different training configurations, the run whose last checkpoint has the highest own-input
     AUC (5 official evaluations against the run's own G2 input). The check confirmed the own-input argmax in all 5
     multi-run episodes. Open points it lists (to ask Zuxing): same-configuration duplicate runs treated as separate
     candidates, some runs left out of his candidate list, the timing of his switch to "last checkpoint", and no
     selection script on disk.
   - **K3fix:** the same rule: for every K3fix episode, the last checkpoint of its training run; where an episode has
     runs of different training configurations (ep09, ep14), the run whose last checkpoint has the higher own-input AUC
     mean. Episodes 1, 8 and 23 use our K3 runs' last checkpoints (his stated choice; our K3 parquet equals the K3fix
     parquet outside episodes 3, 4, 9 and 17). The check confirmed the own-input argmax in both tie-break episodes and
     that every recorded change moved forward within the same run. Open points it lists: he defined the candidate set
     himself and it has gaps (for example his own K3fix runs for ep01 and ep23 are not candidates; only 18 episodes have a
     K3fix file); for the six 5090 files ep11, 14, 17, 21, 25 and 27, "last checkpoint of the run" rests on his README
     (the copies on this server end earlier); and the 5090 checkpoint pruner (round 8's C1) kept some off-grid
     iterations by a criterion that is not part of this selection, a class into which only ep14 at 276000 (off the
     10,000 grid) falls. By D2 these files are candidates as they are; round 9 adds and drops nothing because of these
     points.
   - Teammates' upstream training choices already made (the G2 input, and the inputs chosen for some of his runs, for
     example the K3fix input of the a5 runs of ep14 and ep25) are disclosed separately with rounds 7-8. As the user
     decided on 2026-11-02, every run stays in one candidate pool, selected only by the scores above.
4. **Not final.** 7 G2 files are not their run's last checkpoint (ep10, 14, 19 still training; ep20, 22, 27, 29 older
   than their stopped runs' last files). 7 K3fix files are not their run's final checkpoint: ep01 (our K3 run, stopped at 22500 of 90000), ep11, 17, 21 and 27 (5090 round-2 runs, paused) and ep14 and 25 (a5 continuations, stopped at 276000 and 297000 of 360000). They are evaluated as they are (D2). For ep20, 22, 27 and 29
   the runs' later checkpoints enter through D2 (b), because these pro6000 runs trained further after round 8's
   snapshot.
5. **Host parquet bytes and remote provenance (as round 8 9.4-9.5).** Files trained on the 5090 and robolab hosts are
   relocated to this server's parquet of the same data root; byte identity of the hosts' parquets is not confirmed
   (recorded where known). Training commits: 82837c8 (5090, robolab; ws032 from refs, no git binary), bc3b1bd (pro6000);
   both are ancestors of 8387874. Remote launchers and environment specs are not on this server.
6. **Reference bias (as PREREG_R7 9.6).** refs/B use tracks R and F everywhere, C on 0, 5, 15, 19, 22, 26, 28 and G on
   1, 4, 9, 19, 21, 28. G2 candidates on other episodes are not favoured by the reference; raw and K3fix always are.
7. **Noise and selection.** Phase S takes maxima over noisy 3-run means. Decisions use fresh phase-F runs, the +0.01
   margin over the higher of the incumbent's two baselines, and the 8/10 floor. Runs are single-seed repeats.
8. **Where the text was made.** The scripts and this file were written in the agent's Mac scratchpad and uploaded.
   Nothing of the project ran on the Mac except syntax checks.

## 10. Implementation, hard rules, amendments

| step | what | code |
|---|---|---|
| B1 | this file + `PREREG_R9.sha256`; the final scripts hard-code its sha256 | |
| B2 | `inputs_r9/R8_RECORDS_R9.json`; selftest | `r9_select.py r8records / selftest` |
| B3 | pool dry run (snapshot NOW, nothing copied); trial pool (a few screen rows, plumbing only) + plan-only evaluation check | `r9_pool.py dryrun / trial`, `r9_eval.py run --dry-run --pool-root` |
| B4 | `PREREG_R9_PINS.md` (+ sha256), `DRYRUN_R9.md`, `LOCK_R9.sha256` (all read-only) | |
| R1 | at launch: freeze (T_c9; `POOL_R9.sha256` last), verify | `r9_driver.py` -> `r9_pool.py freeze / verify` |
| R2 | phase S: exports + 3 runs each | `r9_eval.py run --phase screen` |
| R3 | check / gate / score / select S (`SELECTION_S9.sha256` last) | `r9_select.py` |
| R4 | phase F: 10 runs per finalist | `r9_eval.py run --phase final` |
| R5 | check / gate / score / select F (`SELECTION_R9.sha256` last) | `r9_select.py` |

**Hard rules.**
- New files only, under this folder. Rounds 1-8, the kit, the stock runtime and the code trees are only read. Zuxing's
  folder is only read.
- Never edit a file a running process uses. Never signal any process. Never touch the ablation manager or its runs.
- GPUs as 3.5 only.
- Python: flashchord `-I -B` for the pool builder, the evaluation driver and the orchestrator; kit `-I -B` for the
  scorer.

**Amendments.** Any deviation goes into `AMENDMENTS_R9.md` (+ sha256), with time and reason, before the affected step.
An amendment written after any round-9 score was visible must say so. `PREREG_R9_PINS.md` holds the sha256 of every
script and record used; addenda may only add pins and notes.
