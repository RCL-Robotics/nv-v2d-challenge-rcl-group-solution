# Tier-2 consensus selection, round 8 (t2_consensus_r8_20261103): pre-registration

**When.** Written 2026-11-02 from about 20:40 CST and revised from about 21:30 CST (`PREREG_R8_CHANGES.md`). Both are
before the round-8 snapshot T_c and before any round-8 export, evaluation, score or selection. At each writing, this
folder held only development copies of the round-8 scripts and their plumbing outputs (pool dry runs, trial pools, an
incumbent-record build). These were removed before the pinned build steps of section 10 (B2-B4), which re-create them
with the final scripts; the first build's documents and scripts are kept read-only in `superseded_build1/`. Every later
step refuses to run if this file's sha256 differs from `PREREG_R8.sha256` and from the value hard-coded in the round-8
scripts.

**User request** (relayed by the main session, 2026-11-02): "from our side do we run anything to get best tier2 results
under challenge rules?" The user chose a round 8 with three settings:
- round 7's rule, unchanged, applied to the runs that are new or have trained further since round 7's snapshot;
- plus each run's own-input best checkpoint from Zuxing's evaluation watcher ("Add watcher best");
- a pool freeze at T_c = 2026-11-03 12:00 CST.

**Basis.** Selection uses only Tier-2-derived signals:
- the official evaluator's metrics of each policy against its own Tier-2-derived input (own-input scores);
- the variant-B consensus scores against round 1's frozen consensus references refs/B (consensus-B scores);
- the own-input validation gate;
- training-run metadata: iterations, file times, checksums and the first line of each training log.

Zuxing's evaluation-watcher files (`eval_<ep>/metrics.csv`) are own-input evaluations of each run against its own
training input. Only their columns `episode`, `iteration`, `add_auc`, `status` and `finished_at` are used. The header
must hold `episode, iteration, add_auc, sp_sr, mp_sr, mppe_cm, status, finished_at` and no column outside the watcher's
18-column own-input header (`episode, iteration, add_auc, sp_sr, mp_sr, mppe_cm, status, seconds, log, object, attempt,
final, chord_sr, export_max_rel_diff, stock_verified, source_sha256, finished_at, note`); otherwise the freeze stops
(7.4). The inputs are round 7's data
roots (raw Tier 2, K3, K3fix, K3CA, G2). Round 8 adds no input. Their provenance notes stay with round 7's records and
the user's decisions.

**Standing rules**
- **Zuxing's folder is read-only.** Nothing there is written, moved or deleted. No git command is run there and none of
  his scripts. Every file of his that round 8 uses is copied into this folder, with sha256 before == copy == after
  (`COPIES_r8.tsv`, `TELL_ZUXING_r8.txt`). Record copies of text files hold only the lines or columns round 8 uses
  (2.8).
- **What the round-8 scripts never do.** They never commit, push, pack or upload to Kaggle, and they never signal a
  process. Packing the round-8 file and any commit are separate steps that need the user's approval.
- **GPUs (section 3.5).** Evaluations run only on GPUs 6 and 7: plain, one evaluation per GPU, and only while
  `nvidia-smi` lists no other compute process on that GPU. GPUs 0-5 are never used.
- **Files.** New files only. Kit Python is always run with `-B`. The kit and the stock runtime `code/video_to_data_8c778f9`
  are only read.

## 0. Relation to round 7

- **Round 7 is final and frozen.**
  - `SELECTION_R7.sha256` b2dc89dd...; `PREREG_R7.md` 749649a7....
  - Its upload file `submission_r7/T2_R7_closest_trial02_6bb0214.csv` (b06c5ccc...; commit 6bb0214 on
    `dev.haoran.t2r7_20261102`) is trial 2 of round 7's set S7.
  - Round 8 reads round 7's records only through round 7's verified loaders. It never re-runs, re-scores or edits them.
- **The rule.** Round 8 is PREREG_R7 sections 2-5 (pool rule R1-R5, evaluation protocol, eligibility, phase-S finalist,
  phase-F decision, closest-to-mean trial), unchanged except:
  - the deltas **D1-D5** chosen by the user;
  - the consequences **C1-C6**. Each of these is forced either by the deltas or by facts that came after round 7.
- **Rules basis (unchanged from round 7):**
  - challenge §2.8.c: every candidate was trained on a Tier-2-derived input;
  - §2.9.d: one checkpoint per episode, identical digest within an episode;
  - §2.9.b: the upload file is the trial closest to the mean of 10 full trials, never a best-of;
  - Foundational 4.b: a frozen, algorithmic rule.

### Deltas chosen by the user

- **D1 Snapshot.** T_c = 2026-11-03 12:00:00 CST.
  - The universe is round 7's universe re-scanned at T_c, including the copy folders that arrived since (section 2).
  - Mechanical picks (round 7's R4) are taken only for runs that are new or that have more iterations than at round
    7's snapshot.
  - Whatever is missing at T_c is excluded. Nothing is added after T_c.
- **D2 Watcher best.** For every run of Zuxing's with a watcher file, its own-input best checkpoint is one more
  candidate (section 2.6).
- **D3 Incumbents.** Each episode's incumbent is the round-7 pick of the R7 file, with its round-7 phase-F results and
  bar (section 4). It is not re-run.
  - A candidate whose checkpoint already has round-7 phase-S results reuses them. Identity is the file sha256, or for a
    relocated copy the sha256 of its tensor data block.
- **D4 Which episodes run.** Only episodes with at least one round-8 candidate (role screen or reuse7) are evaluated.
  All others keep their round-7 pick.
- **D5 Records.** Round 8 computes no score other than the own-input and consensus-B scores above. Its records state that
  basis.

### Consequences

- **C1 Pick targets on the 5090 (section 2.4).**
  - According to the hand-over README, the 5090's checkpoint pruner kept some off-grid checkpoints by a score-based
    criterion that is not part of this selection. The R7 folder of 5090 copies held only grid iterations and the last
    two.
  - Round 8 therefore takes mechanical pick targets on 5090-hosted runs only from iterations on the 10,000 grid and the
    run's last iteration.
  - The watcher-best rule (D2) gets the same restriction: on a 5090-hosted run, the last ok-evaluated iteration counts
    off the grid only when it is also the run's last counted iteration L (2.6).
  - This keeps every candidate's designation either mechanical (iteration grid, recency) or based on own-input scores.
- **C2 Copy runs (section 2.3).** Round 7's R1b (5090 MANIFEST copies) is generalised to the new copy folders.
  - The embedded `output_dir` must equal the as-trained run path.
  - The data root must be Tier-2-derived and present on this server.
  - Every selected file must equal its checksum line.
  - A training commit must be named, either by the first `train_t2:` line of the training log, or for a host without git
    by the host evidence file.
- **C3 Round-7 phase-F participants are barred7 (section 2.7).**
  - These are the replaced round-7 incumbents and the round-7 finalists that were not picked.
  - Their round-7 phase-F runs already compared them with the round-8 incumbent in the same 10 paired trials, and they
    fail the round-8 test on those runs (checked when `R7_RECORDS_R8.json` was built).
  - A second fresh test of the same pair would be a second chance on the same question (PREREG_R7 5.4). They are never
    finalists. Round 7's two carried barred candidates stay barred.
- **C4 One finalist per episode.** Every episode has an incumbent. Round 7's incumbent-episode rule (5.4 / 5.5: one
  finalist, replacement test) therefore applies to all 30 episodes. Round 7's STRICT path (two finalists for an episode
  without an incumbent) does not arise.
- **C5 GPUs.** A GPU is used only while no other compute process is listed on it. This is stricter than round 7's
  free-memory rule.
- **C6 Deadlines, failure handling and one evaluation driver at a time (section 7).**

## 1. Snapshot and inputs

- **T_c = 2026-11-03 12:00:00 CST.** The freeze run `r8_pool.py freeze` refuses to start before T_c and scans right
  after it.
- **A model file counts iff all of these hold:**
  - its mtime <= T_c - 300 s (round 7's R3);
  - its ctime <= T_c, so a file copied in after T_c never counts;
  - for a copy run, it is listed with a matching sha256 in the copy's checksum file.
- **Watcher rows** count only if `finished_at` <= T_c.

Round-7 inputs, read through round 7's own verified loaders (they check round 7's whole pin chain):
- `selection_r7.json` (657275ba...);
- the phase-S / phase-F scores, gates, manifests and rollout lists listed in `SELECTION_R7.sha256` / `SELECTION_S7.sha256`;
- `inputs_r7/pool_r7.json` (7c4b553b...) and `inputs_r7/runs_r7.json` (155432e2...).

`inputs_r8/R7_RECORDS_R8.json` holds the 30 incumbents (section 4), the identities of round 7's 399 phase-S candidates
and phase-F participants, and the two carried barred candidates. It is built by `r8_select.py r7records` (step B2) and
pinned in `PREREG_R8_PINS.md`. A development build gave the table of section 4.

## 2. The candidate pool (mechanical, plus the own-input watcher best)

The rule is implemented by `r8_pool.py`. It loads round 7's resolver `scripts/t2cons_r7_pool.py` (7a08fa6d...) from its
verified bytes and calls its helpers unchanged.

### 2.1 Round 7's rule R1-R5 (unchanged)

- **R1 (pro6000).** A training-run directory under our round-6 roots (ours only if in round 6's eligible list) or under
  Zuxing's `runs/*`, whose embedded `output_dir` is that directory.
- **R2.** The training parquet lies in a Tier-2-derived data root (raw, K3, K3fix, K3CA, G2).
- **R3.** Settled files only; L = the largest settled iteration; L >= 10,000.
- **R4.** K = 3. The targets are L (1 - j/6), j = 0..3 (100 / 83 / 67 / 50 % of L). For each target, take the saved
  iteration >= L/2 nearest to it (ties: the larger). Duplicates collapse.
- **R5.** Roles are assigned in the deterministic order (episode, owner, directory). Section 2.7 adds the round-8 roles.

### 2.2 Folders that are not walked as training-run directories

These are Zuxing's copy, retest and selection bundles:
- `2026-10-24-5090-best`, `2026-10-28-tier2-retest`, `2026-11-02-tier2-g2-ownsel` (as round 7);
- `2026-11-02-tier2-g2-reports-eval`, a report / evaluation bundle that is never entered.

The copy folders `2026-11-01-round7-5090-ckpts`, `2026-11-02-tier2-r8-handover/<host>/` and
`2026-11-02-round7-robolab-ckpts/<host>/` are resolved as copy runs (2.3).

The ownsel bundle is used only as the fallback location of a watcher file (2.6), never for a checkpoint.

### 2.3 Copy runs (R1c, generalising round 7's R1b)

Each directory `<group>/[<sub>/]epNN` with model files in one of the three copy folders is a copy run. It is eligible
iff all of these hold:
1. **Embedded output_dir.** The last counted checkpoint's `output_dir` equals the as-trained run path: `/mnt/gfs/home/
   zuxinglu/workspace/runs/<group>/[<sub>/]epNN` for the 5090, `/home/zuxinglu/workspace/runs/...` for the robolab hosts.
2. **Data root.** The as-trained `task.parquet` lies in a Tier-2-derived data root, `reference.parquet` is absent or the
   same, and the same root and sequence exist on this server.
3. **Episode.** The sequence maps to the folder's episode in this server's manifest.
4. **L.** L >= 10,000.
5. **Commit evidence.** A training commit is named by one of:
   - the first `train_t2:` line of the run's training log (`train.log` in the copy, `logs/<group>/<ep>[_<sub>].log` for
     robolab copies);
   - round 7's MANIFEST commit column;
   - for a host without git, the hand-over evidence file `evidence/<host>.txt` (the worktree HEAD read from its refs).
6. **Checksums.** Every selected file's sha256 equals its line in the copy's checksum file:
   - `SHA256SUMS` of round 7's 5090 folder;
   - the run's own `SHA256SUMS` in the hand-over;
   - `SHA256SUMS.<host>` in the robolab folder.

Our pro6000 runs and Zuxing's pro6000 runs are resolved by R1 as in round 7. For Zuxing's pro6000 runs, a commit in the
first `train_t2:` line of `logs/<group>/<ep>.log` is required.

**One run, several copies.** A run is identified by (host class, embedded output_dir). If it is copied more than once
(for example the 5090 round-2 ep15 run in round 7's folder and in the hand-over), only the copy with the largest L counts.
Ties go to the copy with more counted files, then path order.

### 2.4 Pick targets on 5090-hosted runs (C1)

For a run hosted on the 5090, R4 picks only among counted iterations that are multiples of 10,000 or equal to the run's
last counted iteration L. Pro6000 and robolab-host runs (no pruner) pick among every counted iteration, as in round 7.

### 2.5 Status against round 7, mechanical picks (D1)

- **new:** not among round 7's eligible runs.
- **trained further:** L greater than at round 7's snapshot.
- **unchanged:** L equal. A run with fewer iterations than at round 7 is treated as unchanged.

R4 picks are candidates only for new and trained-further runs. Our own runs have not changed since round 7.

### 2.6 Watcher best (D2)

For every eligible run of Zuxing's:
1. **Watcher file.** The watcher file is `os.path.join(os.path.dirname(run), 'eval_' + basename(run), 'metrics.csv')`.
   If the run copy has none, the copy at the same relative path under `runs/2026-11-02-tier2-g2-ownsel/inputs_5090/` is
   used (robolab runs: under its `robolab/` folder).
2. **Copy first.** At the freeze, the file is first copied (sha256 before == copy == after, retried while it is still
   being appended to), its header is checked (Basis), and the copy is parsed. The record copy kept in
   `zuxing_copies_r8/` holds the 8 asserted columns of every row; the full copy is a temporary file, removed after the
   parse, and its sha256 stays in `COPIES_r8.tsv`.
3. **Rows.** Rows with `episode` == the run's episode, `status == ok` and `finished_at` <= T_c.
4. **Last ok-evaluated iteration** = the largest iteration among those rows.
5. **Eligible rows.** The iteration is a multiple of 10,000 or equals the last ok-evaluated iteration, and its
   `model_<it>.safetensors` counts at T_c (1). On a 5090-hosted run (C1, 2.4), the last ok-evaluated iteration counts
   off the grid only when it equals the run's last counted iteration L.
6. **Pick.** The argmax of `add_auc` over the eligible rows; ties go to the lower iteration.

A run without a watcher file, or without an eligible row, has no watcher-best candidate.

### 2.7 Roles (round 7's R5 plus round 8)

In the deterministic order (episode, owner, run directory, iteration), with identity = file sha256, else the sha256 of
the tensor data block:
- **incumbent:** the episode's round-8 incumbent (its round-7 pick). It is not screened.
- **barred7 (C3):** a round-7 phase-F participant of the episode other than the pick, or one of round 7's carried barred
  candidates. It is never a finalist and never re-evaluated. If it has round-7 phase-S results, it is listed in the
  phase-S table.
- **reuse7:**
  - What it is: a round-7 phase-S candidate of the same episode with the same input (this server's training parquet
    sha256 and, for non-raw inputs, the frame-reference sha256).
  - What carries over: its round-7 phase-S runs and scores and its export record (round 7's `exports/` or round 6's
    `exports/` for round 6's reused candidates).
  - The job keeps its round-7 name and is not re-evaluated in phase S.
- **duplicate:** the same file selected earlier in round 8. Routes are merged.
- **screen:** every other file. It gets one export and 3 fresh runs. A job name that already exists in round 7's
  records gets the suffix `_r8`.
- **refused:** the checkpoint's parquet differs from the run's, or the file differs from its checksum line.

### 2.8 Copies and relocation

- **Copies.** Each new screen file of Zuxing's (model + rsi of the same iteration) is copied into `zuxing_copies_r8/`.
  So are each copy run's hydra yaml, `final.json`, `MANIFEST.*` and checksum files, the watcher files used (2.6), the
  host evidence files used and a one-line excerpt of each training log's first `train_t2:` line.
  - **Record copies hold only what round 8 uses.** Of a `MANIFEST.txt`: its run / source / copied / status / train log /
    task.parquet / parquet sha256 / models / files lines. Of a checksum file: the lines of model, rsi, `final.json`,
    training-log, `MANIFEST.txt` and hydra files. In any copied text file, a line that round 7's token guard
    (`t2cons_r7_pool.py`) refuses is left out, and a training-log line it refuses keeps only its commit.
  - A file that keeps every line is copied verbatim (sha256 before == copy == after). An excerpt records the full
    source's sha256 before and after the read. Checkpoint copies are always byte copies.
- **Relocation.** A screen file from another host (5090, robolab) is relocated metadata-only into `relocated_r8/`, using
  round 5's pinned `relocate_bytes` / `verify_pair` (t2cons_r5_relocate.py 09a6cb8e) as in rounds 2-7:
  - exactly `task.parquet` and `reference.parquet` change, to this server's same-named file of the same data root;
  - every tensor byte and every other metadata field is unchanged.
- **Verification.** Each relocation is re-verified with round 7's `verify_row` at the freeze and at every manifest
  build. The relocated file is the evaluated input. The pool sha256 stays the original's.

### 2.9 Outputs of the freeze (all mode 444)

- `POOL_R8.json` (every row with its role, routes, run, iteration, sha256, input, parquet + sha256, frame reference +
  sha256, commit, copies, relocation, round-7 links).
- `inputs_r8/`:
  - `pool_r8.csv`, `runs_r8.{json,csv}` (every directory scanned, eligible or not with the reason, status against round
    7);
  - `WATCHER_R8.tsv`, `TRACKS_R8.sha256`, `ZUXING_CODE_R8.tsv`, `resolve_r8.log`.
- `COPIES_r8.tsv`, `TELL_ZUXING_r8.txt`, `RELOCATION_R8.{tsv,json,sha256}`.
- **`POOL_R8.sha256` last.** It lists all of the above, `R7_RECORDS_R8.json`, every copy and every relocated file.

## 3. Evaluations

### 3.1 Protocol: round 7's 3.1, unchanged

**Driver.** `r8_eval.py` loads round 7's driver `scripts/t2cons_r7_eval.py` (1d66a2db...) from its verified bytes and
re-binds its folder names to this folder. Its code runs unchanged, and through it `t2cons_eval.py` (2cf42786) and
`t2best5_driver.py` (24b9af39).

**Export and evaluation.**
- **Export trees:** `code/fc_r2` @ 61fa5d3 for ours, `code/nv-v2d_runin` @ b39b69b for Zuxing's.
- **Tolerance:** 1e-4, with one fallback at 2e-4 on a tolerance refusal. The export is stock verified and
  `source_sha256` == the evaluated input.
- **Evaluation:** `scripts/track2/eval_all.py` (657295921e82) with `--export no --world-count 1` on the untouched stock
  runtime (`evaluate_policy.py` 160d24f5). Frame-0 start, explicit reset, no help. The reference is the job's own
  training parquet.

**ok checks, retries and integrity are round 7's.**
- Up to 3 tries per run, at least 5 minutes apart. A completed evaluation is never re-run.
- A failed run counts as not ok.
- A non-tolerance export refusal ends the job. An `AMENDMENTS_R8.md` written before any round-8 score may add one export
  with the 82837c8 exporter, as in PREREG_R7 3.1.
- A change of the kit or the stock tree halts new starts.

**Earlier exports** of reuse7 finalists (round-7 or round-6 records) are re-hashed and never re-exported.

### 3.2 Relocated jobs

A screen job from another host takes round 7's relocation branch: its `RELOCATION_R8.tsv` row is re-verified at every
manifest build, and the expected sha256 is the relocated file's.

### 3.3 Phase S: 3 runs per new screen candidate

- **Jobs:** the `screen` rows of `POOL_R8.json`. Each gets one export (`exports/<job>/`) and 3 official runs
  (`runs_s/<job>/run<k>/`).
- **Order (decides nothing):** exports in (episode, job) order on CPU, then evaluations run-major.
- **reuse7 candidates** contribute their round-7 phase-S results as recorded. They are not re-evaluated or re-scored.

### 3.4 Phase F: 10 runs per finalist

- **Jobs:** the finalists of `final_r8.json` (written and locked by `SELECTION_S8.sha256` before phase F starts). There
  is at most one per episode. Each gets 10 fresh official runs (`runs_f/<job>/run<t>/`), in trial-major order.
- **Exports:**
  - a new screen finalist uses its single phase-S export;
  - a reuse7 finalist is rebuilt by an unbound second instance of round 7's driver from round 7's own records, and uses
    its round-7 / round-6 export.
- **Incumbents are not re-run.** Their 10 round-7 phase-F runs are used (D3). Without a finalist, phase F has no job.

### 3.5 GPUs and lanes (C5)

- **GPUs.** GPUs 6 and 7 only. Plain: no `CUDA_MPS_*` variables, never a GPU with an MPS server, client or control
  daemon. At most one evaluation of ours per GPU, and at least 4096 + 1536 MiB free.
- **Only while free.** An evaluation starts on a GPU only while `nvidia-smi --query-compute-apps` lists no compute
  process on it. Zuxing's trainings on GPUs 6/7 are his; nothing of his is ever touched or signalled.
- **Waiting.** If GPUs 6/7 are busy at T_c, the driver waits, re-checking every 10 s, and logs the reason once per
  distinct reason. Exports (CPU) proceed meanwhile.
- **Lane locks:** round 7's rule, with round 7's own `evallock/` added to the shared list.
- **Processes:** `nice -n 10`, `taskset` on the NUMA1 eval-lane cores, and a 10 s cooldown per GPU.
- **Running evaluations:** one past 2400 s is only flagged OVERDUE. Nothing is ever signalled. `STOP` in this folder
  stops new starts.

### 3.6 Seeds (records only)

As round 7: there is no selectable seed on the official path. Seeds are recorded per run in `seeds.json` and
`seeds_{s8,f8}.tsv`.

## 4. Incumbents (D3): the round-7 picks, with their round-7 phase-F records

For every episode, the incumbent P_e is the R7 file's pick (round 7's S7, trial 2).

**m_P_frozen** depends on how P_e was selected:
- **Kept in round 7** (selected in round 1 or round 6): round 7's `m_P_frozen`, the frozen runs of the round that
  selected it.
- **Selected in round 7** (a replacement or a STRICT pick): the float64 mean of its 10 round-7 phase-F consensus-B AUCs.
  This equals `selection_r7.json` m_F exactly, which was checked.

**m_P_fresh** = the mean of its round-7 phase-F consensus-B AUCs (10 good runs each, as recorded). It is computed by
round 7's `decide_f` and is never re-run.

**The two thresholds:**
- **bar = max(m_P_frozen, m_P_fresh) + 0.01.** This is the phase-F bar, and equals round 7's bar for a kept incumbent.
- **bar_S = m_P_frozen + 0.01.** This is the phase-S threshold (PREREG_R7 5.4).

| ep | incumbent (round-7 pick) | input | status in round 7 | m_P_frozen | m_P_fresh (r7 F) | bar |
|---|---|---|---|---:|---:|---:|
| 00 | `ep00_z5k3fix_90000` | k3fix | STRICT pick | 0.719120 | 0.719120 | 0.729120 |
| 01 | `ep01_k3fix_final44922` | k3fix | incumbent kept | 0.171319 | 0.168566 | 0.181319 |
| 02 | `ep02_k3fix_final34093` | k3fix | incumbent kept | 0.868138 | 0.872717 | 0.882717 |
| 03 | `ep03_k3fix_final27068` | k3fix | incumbent kept | 0.921527 | 0.907122 | 0.931527 |
| 04 | `ep04_zxg2_450000` | g2 | replacement | 0.429436 | 0.429436 | 0.439436 |
| 05 | `ep05_zx_k3fix_88000` | k3fix | incumbent kept | 0.578153 | 0.579742 | 0.589742 |
| 06 | `ep06_k3fix_final28420` | k3fix | incumbent kept | 0.797276 | 0.798993 | 0.808993 |
| 07 | `ep07_k3fix_final18529` | k3fix | incumbent kept | 0.920452 | 0.894092 | 0.930452 |
| 08 | `ep08_k3easy_final17496` | k3 | incumbent kept | 0.596363 | 0.595904 | 0.606363 |
| 09 | `ep09_raw_final39328` | raw | incumbent kept | 0.760460 | 0.778527 | 0.788527 |
| 10 | `ep10_raw_final34267` | raw | incumbent kept | 0.412469 | 0.410682 | 0.422469 |
| 11 | `ep11_raw_24000` | raw | incumbent kept | 0.807319 | 0.802563 | 0.817319 |
| 12 | `ep12_k3fix_final23178` | k3fix | incumbent kept | 0.543911 | 0.543741 | 0.553911 |
| 13 | `ep13_k3fix_final39254` | k3fix | incumbent kept | 0.665483 | 0.665971 | 0.675971 |
| 14 | `ep14_k3fix_final44922` | k3fix | incumbent kept | 0.330614 | 0.335807 | 0.345807 |
| 15 | `ep15_raw_final44922` | raw | incumbent kept | 0.729895 | 0.734423 | 0.744423 |
| 16 | `ep16_z5a5_120000` | g2 | STRICT pick | 0.951297 | 0.951297 | 0.961297 |
| 17 | `ep17_z5r2_61500` | k3fix | replacement | 0.659403 | 0.659403 | 0.669403 |
| 18 | `ep18_k3fix_final18911` | k3fix | incumbent kept | 0.419129 | 0.421081 | 0.431081 |
| 19 | `ep19_z5g2hardc2_360000` | g2 | STRICT pick | 0.581095 | 0.581095 | 0.591095 |
| 20 | `ep20_z5fill_90000` | k3fix | STRICT pick | 0.872502 | 0.872502 | 0.882502 |
| 21 | `ep21_z5fillc_282500` | g2 | replacement | 0.494543 | 0.494543 | 0.504543 |
| 22 | `ep22_z5k3fix_89500` | k3fix | STRICT pick | 0.757867 | 0.757867 | 0.767867 |
| 23 | `ep23_raw_final20023` | raw | incumbent kept | 0.902867 | 0.903356 | 0.913356 |
| 24 | `ep24_z5q_150000` | g2 | replacement | 0.905657 | 0.905657 | 0.915657 |
| 25 | `ep25_z5a5c_280500` | k3fix | STRICT pick | 0.878012 | 0.878012 | 0.888012 |
| 26 | `ep26_zx_k3fix_92000` | k3fix | incumbent kept | 0.731484 | 0.731605 | 0.741605 |
| 27 | `ep27_k3fix_final20863` | k3fix | incumbent kept | 0.897854 | 0.905354 | 0.915354 |
| 28 | `ep28_z5q_180000` | g2 | STRICT pick | 0.150076 | 0.150076 | 0.160076 |
| 29 | `ep29_k3fix_final17358` | k3fix | incumbent kept | 0.505675 | 0.504416 | 0.515675 |

The table shows values to 6 decimals. Decisions use the exact floats in `R7_RECORDS_R8.json`.

**Barred7 participants (C3).** These are round 7's replaced incumbents of episodes 04, 17, 21 and 24, its non-picked
finalists (ep02_z5q_180000, and the rank-2 finalists of 00, 16, 19, 20, 22, 25, 28), and its carried barred pair
(ep12_raw_37575, ep13_raw_44922). When `R7_RECORDS_R8.json` was built, none of the participants would pass the round-8
test (10 good runs, SP = MP = 1 in >= 8, m_F >= bar) on its own round-7 phase-F runs. The build asserts this.

## 5. Scores and selection (selection uses only own-input and consensus-B scores)

### 5.1 Scores and gate: round 7's 5.1, unchanged

Every round-8 run gets:
- **variant-B consensus scores**, from the pinned round-1 functions: `rollout_on_grid` against the job's own input
  reference, `metrics_vs` with `align_initial_pose=False`, 500 released-mesh vertices, seed 0, float64, against round
  1's frozen `refs/B` (build.json 9646de2d);
- **the own-input validation gate:**
  - `auc_seed`: the evaluator AUC must lie within [min_s - 0.02, max_s + 0.02] over vertex seeds 0..9;
  - |dMPPE| <= 0.5 cm;
  - SP-SR and MP-SR equal.

The rollout list of each phase is frozen in `inputs_r8/ROLLOUTS_{S8,F8}.sha256` before the first score.
floor = the number of runs with own-input SP-SR = 1 AND MP-SR = 1.

### 5.2 Scorer

`r8_select.py` loads round 7's scorer `scripts/t2cons_r7_consensus.py` (e7545dfa...) from its verified bytes and calls
its functions unchanged:
- `s_row` and `select_s` (5.3 / 5.4);
- `decide_f` (5.5);
- `trial_rows` and `closest_trial` (5.6);
- through it, round 2's `run_flags` / `SCORE_COLS` and round 1's `rollout_on_grid`, `metrics_vs`, `load_refs`, `grid`,
  `load_kit`, `stats`, `floor_count` and integrity checks.

Its gate and score commands are round 7's `cmd_gate` / `cmd_score` loops over round 8's runs, without round 7's round-6
reuse reproduction, which does not apply. Its selftest (synthetic) and a mock harness (synthetic round-8 values, real
round-7 records, nothing written) are pinned in `PREREG_R8_PINS.md` before it scores anything.

### 5.3 Phase-S eligibility (round 7's 5.3)

A candidate is **S-eligible** iff:
- its export is ok;
- it has 3 ok official runs, none flagged by the gate;
- floor_S >= 2.

cons_S = the float64 mean of its 3 consensus-B AUCs. A reuse7 candidate is judged on its round-7 phase-S record as
recorded.

Ties everywhere: lower population variance of the 3 consensus AUCs, then lower mean consensus MPPE, then job name.

### 5.4 Phase-S selection (round 7's 5.4 for incumbent episodes; C4; D4)

- **Which episodes.** An episode is evaluated iff it has at least one candidate with role screen or reuse7.
- **Finalist.** The S-eligible candidate with the highest cons_S among those with **cons_S >= m_P_frozen + 0.01**
  (exact floats). barred7 candidates are never finalists. If no candidate qualifies, there is no finalist and the
  round-7 pick stays.
- **Outputs:** `scores_s8/`, `selection_s8.{json,txt,md}`, `selection_s8_candidates.csv`, `final_r8.json` and
  **`SELECTION_S8.sha256` last**.

### 5.5 Phase-F decision (round 7's 5.5)

m_F(f) = the float64 mean of f's 10 phase-F consensus-B AUCs. **f replaces the round-7 pick iff** all of these hold:
- (a) f has 10 ok, non-flagged phase-F runs;
- (b) own-input SP-SR = MP-SR = 1 in >= 8 of them;
- (c) **m_F(f) >= max(m_P_frozen, m_P_fresh) + 0.01** (exact floats; m_P_fresh from the incumbent's round-7 phase-F
  runs).

Otherwise the round-7 pick stays.

### 5.6 The set S8 and R8-CLOSEST

- **S8** = per episode the 5.5 result.
- **Complete trials.** Trial t is complete iff every episode's job has an ok, non-flagged run t:
  - a kept round-7 pick uses its round-7 phase-F run t;
  - a replacement uses its round-8 phase-F run t.
- **R8-CLOSEST** = the complete trial whose 30-episode mean own-input AUC is closest to the mean of that quantity over
  the complete trials. Ties: smaller distance of the consensus-B mean to its mean, then lower t. Never a best-of.
- **Consistency check.** If no episode changes, S8 == S7, and R8-CLOSEST must be trial 2, the R7 file's trial. The
  scorer refuses otherwise.
- **Outputs:**
  - `SELECTION_R8.json`: rule, every episode's decision, S8, its run sources, trials, R8-CLOSEST and its per-episode
    rollouts and exports;
  - `selection_r8.md`, `selection_r8_episodes.csv`, `summary_r8_trials.csv`;
  - **`SELECTION_R8.sha256` last.** From then, round 8 is final.

### 5.7 Nothing else enters

No other score, no manual override, no comparison of own-input AUC across different inputs.

## 6. After round 8 (not part of the round-8 scripts)

- **If no episode changed,** the R7 file remains the file: same picks and same trial.
- **If episodes changed,** packing R8-CLOSEST runs separately with round 7's pack script, adapted and pinned first. The
  same holds for any submission commit (new branch only, never main, never force) and any upload. Each needs the
  user's approval.

## 7. Orchestration, deadlines, failure handling (C6)

1. **Driver.** `r8_driver.py`, started by `r8_launch.sh`, runs the steps below. Each step is resumable and is recognised
   by its output; a finished step is never repeated.
   1. Sleep until T_c.
   2. `r8_pool.py freeze`, then `r8_pool.py verify`.
   3. `r8_eval.py run --phase screen`.
   4. `r8_select.py check / gate / score / select --phase S`.
   5. If `final_r8.json` names a finalist, `r8_eval.py run --phase final` and then `check / gate / score / select
      --phase F`. Otherwise `select --phase F` directly.

   It writes `r8_driver.log` and the heartbeat `R8_DRIVER_HEARTBEAT.json` every 30-60 s. The evaluation driver has its
   own `eval_r8_heartbeat_{s8,f8}.json`. The scorer's `gate` and `score` steps answer "already done" when their output
   exists, was written by the pinned `r8_select.py` and matches its current inputs; a half-written output halts.
2. **GPUs busy at T_c.** The pool is frozen at T_c regardless. Exports start at once on CPU, and evaluations wait (3.5).
3. **Deadlines.**
   - **Phase-S cut-off: 2026-11-04 08:00:00 CST.** If phase S has not finished by then, the driver creates `STOP` and
     `S_CUTOFF.json`, and running evaluations finish and are recorded. Then `r8_select.py keep-r7` writes the
     SELECTION_R8 outputs with every episode keeping its round-7 pick. This holds equally if GPUs 6/7 never became
     free. Phase S is never selected from an incomplete screening, because its run-major order would favour some
     candidates.
   - **Phase-F deadline: 2026-11-04 12:00:00 CST.** If phase F has not finished by then, the driver creates `STOP` and
     `DEADLINE_F.json`, and running evaluations finish and are recorded. A finalist without 10 ok runs fails 5.5 (a),
     so its episode keeps the round-7 pick. The other episodes are decided normally.
4. **Failures.**
   - Evaluation failures and retries follow round 7 (3.1). A candidate or finalist with a failed run is not eligible or
     does not replace.
   - If the evaluation driver exits before its phase is done for any reason other than those of the next point, it is
     restarted after 10 minutes, at most 6 times. After that, phase S is handled as at the cut-off. In phase F, the
     driver halts.
   - A pin / preflight / integrity failure halts the driver: the evaluation driver's heartbeat says `halted`, or its
     output says `preflight FAILED` or that a pinned file's sha256 differs. So do a failed freeze (including a watcher
     header outside 2.6) and a failed scorer step. If the phase-F deadline passes while no evaluation driver runs (for
     example between restarts), the evaluation heartbeat does not record the deadline stop, the scorer's check fails and
     the driver halts. A halted driver selects nothing further, and the R7 file stays in force.
5. **STOP by anyone else.** A `STOP` file not created by the driver at a deadline stops new evaluations. The driver ends
   at the next step boundary and selects nothing.
6. **One evaluation driver at a time.** The orchestrator reads `/proc` (it never signals) and never starts `r8_eval.py`
   while one is alive, for example one left running by an earlier orchestrator. It waits for it, and the wait is never
   counted as a restart. `r8_select.py keep-r7` and the scorer refuse while one is alive; a stale `driver.pid` of an
   ended process does not count. `r8_launch.sh` refuses to start an orchestrator while one is alive.

## 8. Budget

- **Phase S.** The development dry run of 2026-11-02 21:39 found 89 new screen candidates (267 evaluations) in 24
  episodes. More copies are still
  arriving (hand-over runs of the 5090 G2 rest and redo runs, ws032), and the pro6000 G2 rest runs keep training until
  T_c. The expected total is 100-130 screen candidates (300-390 evaluations).
- **Phase F.** At most 30 finalists x 10 runs; expected 5-12 finalists (50-120 evaluations).
- **Wall time.** Capacity is about 110 evaluations per hour on the two lanes (round 7's measurement). Once GPUs 6 and 7
  are free, expect about 3-4 h for phase S plus about 1 h for phase F, plus about 0.5 h of CPU scoring. If GPUs 6/7 are
  free from T_c, round 8 ends on 2026-11-03 at about 17:00-19:00 CST.

## 9. Disclosures

1. **The incumbents carry round-7 choices.** Rounds 1, 6 and 7 selected them by noisy-only rules. Their round-7 phase-F
   runs are reused, so phase F compares a finalist's round-8 runs with the incumbent's round-7 runs. Both were run on the
   same GPUs 6/7 with the same stock runtime and integrity baseline (byte copy of round 1's, ef9b5035), on different
   days.
2. **What the writer of this file saw.**
   - **Round-7 records:** round 7's noisy-only records (the incumbent table above and the round-7 phase-S / phase-F
     consensus-B summaries in `selection_s7.md` / `selection_r7.md`).
   - **Pool dry run:** the candidate lists, and the watcher-best iterations it reports (not their values).
   - **Watcher files:** the headers and the iteration / status columns of a few watcher files.
   - **Hand-over README and evidence files:** read with decimal numbers masked. They state the 5090 pruner note (C1),
     the training commits, the host parquet hashes and a flash_chord tree comparison.
   - **Revision (about 21:30-21:50 CST):** the verifier's report on the first build (it holds no score); the line keys of
     the hand-over `MANIFEST.txt` files, with the lines round 7's token guard refuses masked; the watcher header line;
     and, while checking the excerpt format of a trial pool, the first two data rows (iterations 2000 and 4000, which are
     never candidates) of two watcher excerpts (episodes 0 and 22).
   - No round-8 candidate's own-input or consensus value was read or used to write this rule.
3. **Copy folders arrive until T_c.** `2026-11-02-tier2-r8-handover/` grows while copies arrive ("Contents grow as copies
   arrive"). A host folder `ws032_aborted_stream/` was present at the dry run. Only what counts at T_c (section 1)
   enters.
4. **Host parquet bytes.**
   - **What differs:** the hand-over records the 5090's and ws032's own sha256 of each run's `task.parquet` "as the file
     is now". For every G2 sequence checked, these differ from this server's same-named G2 parquet.
   - **What is checked:** byte identity only. Content identity cannot be checked from here.
   - **Same as earlier rounds:** relocation evaluates every policy against this server's parquet, which is the input that
     would be declared. Rounds 2-7 relocated 5090 G2 checkpoints the same way, including round 7's picks for 16, 19, 21,
     24 and 28.
   - **Records:** `runs_r8.csv` records per copy run the host's value, its source and whether it matches.
5. **Training code.**
   - **Commits.** Zuxing's 5090 and robolab runs name 82837c8 (`dev.zuxing.hard_start`, pushed), and his pro6000 runs
     bc3b1bd or 82837c8.
   - **The 5090 worktree.** The hand-over evidence shows its 65 changed files to be LFS pointers in HEAD replaced by
     their content (assets). The robolab worktree shows the same.
   - **ws032 has no git binary.** Its training logs name no commit. Its HEAD is read from refs in the hand-over evidence
     file (82837c8), and its flash_chord tree equals the 5090's except runtime convex caches (`evidence/tree_compare.txt`).
     ws031's training log names 82837c8.
   - **Not in a pushed commit:** the G2 builder, as in round 7.
6. **Pruned-host restriction (C1).** It removes off-grid 5090 iterations other than the last from the mechanical
   targets. It can only remove candidates; it adds none.
7. **Reference bias (as PREREG_R7 9.6).** refs/B use tracks R and F everywhere, C on 0, 5, 15, 19, 22, 26, 28 and G on
   1, 4, 9, 19, 21, 28. G2 candidates on other episodes are not favoured by the reference; raw and K3fix always are.
8. **Noise and selection.** Phase S takes maxima over noisy 3-run means. Decisions use fresh phase-F runs, the +0.01
   margin over the higher of the incumbent's two baselines, and the 8/10 floor. Runs are single-seed repeats.
9. **Where the text was made.** The new scripts and this file were written in the agent's Mac scratchpad and uploaded.
   Nothing of the project ran on the Mac except syntax checks.

## 10. Implementation, hard rules, amendments

| step | what | code |
|---|---|---|
| B1 | this file + `PREREG_R8.sha256`; the final scripts hard-code its sha256 | |
| B2 | `inputs_r8/R7_RECORDS_R8.json`; selftest; selection mock | `r8_select.py r7records / selftest`, `dryrun/r8_mock_select.py` |
| B3 | pool dry run (snapshot NOW, nothing copied); trial pool (3 screen rows, for plumbing only) + plan-only evaluation check | `r8_pool.py dryrun / trial`, `r8_eval.py run --dry-run --pool-root` |
| B4 | `PREREG_R8_PINS.md` (+ sha256), `DRYRUN_R8.md`, `PREREG_R8_CHANGES.md`, `LOCK_R8.sha256` (all read-only) | |
| R1 | wait for T_c; freeze (`POOL_R8.sha256` last) | `r8_driver.py` -> `r8_pool.py freeze / verify` |
| R2 | phase S: exports + 3 runs each | `r8_eval.py run --phase screen` |
| R3 | check / gate / score / select S (`SELECTION_S8.sha256` last) | `r8_select.py` |
| R4 | phase F: 10 runs per finalist | `r8_eval.py run --phase final` |
| R5 | check / gate / score / select F (`SELECTION_R8.sha256` last) | `r8_select.py` |

**Hard rules.**
- New files only, under this folder. Rounds 1-7, the kit, the stock runtime and the code trees are only read. Zuxing's
  folder is only read.
- Never edit a file a running process uses. Never signal any process.
- GPUs 6 and 7 only, plain, no MPS, only while no other compute process is listed.
- Python: flashchord `-I -B` for the pool builder, the evaluation driver and the orchestrator; kit `-I -B` for the
  scorer.

**Amendments.**
- Any deviation goes into `AMENDMENTS_R8.md` (+ sha256), with time and reason, before the affected step. An amendment
  written after any round-8 score was visible must say so.
- `PREREG_R8_PINS.md` holds the sha256 of every script and record used. Addenda may only add pins and notes.
- `PREREG_R8_CHANGES.md` lists, with times, the changes made to this file before T_c and before any round-8 evaluation.
