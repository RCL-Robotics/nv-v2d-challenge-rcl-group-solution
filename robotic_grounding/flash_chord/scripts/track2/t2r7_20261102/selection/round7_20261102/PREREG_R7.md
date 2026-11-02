# Tier-2 consensus selection, round 7 STRICT (t2_consensus_r7_20261102): pre-registration

**When.** Written 2026-11-02 from about 06:30 CST, after the mechanical pool resolution at the cut-off T_c (section 1),
the read-only copies and the metadata-only relocations, and BEFORE any round-7 export, evaluation, score or selection.
At writing, this folder held only `inputs_r7/`, `zuxing_copies_r7/`, `relocated_r7/`, `COPIES_r7.tsv`,
`TELL_ZUXING_r7.txt` and `RELOCATION_R7.{tsv,json,sha256}`; no `exports/`, `runs_s/`, `runs_f/`, `scores_*`, `dryrun/` or
`evallock/`. Every later step refuses to run if this file's sha256 differs from `PREREG_R7.sha256`.

**Request** (2026-11-02): select the best checkpoints of Zuxing's and Haoran's Tier-2 work and make a submittable
file without a placeholder commit URL; the policy input is each policy's own (raw or filtered) Tier-2 input. Round 7
therefore picks, per episode, the best checkpoint of all Tier-2 training runs of Haoran's and Zuxing's on this server and
of Zuxing's 45 copied RTX 5090 run folders. Selection uses only own-input and consensus-B (noisy) scores.

**Standing rules**
- **STRICT.** Every decision (pool, incumbents, order, selection, trial choice) uses only Tier-2-derived signals: the
  official evaluator's own metrics against each policy's own (raw or filtered Tier-2) input, round 1's frozen noisy
  consensus references refs/B, the noisy gates, training logs. Zuxing's single-checkpoint copies (tier2_best_ckpts,
  5090_best bundles, retest folders, the 2026-11-02 ownsel bundle) are not candidates; his runs enter only through the
  mechanical pool rule over full run folders. The round-6 picks of episodes 00, 16, 19, 20, 22, 25 and 28 (ep00 60000,
  ep16 34000, ep19 248000, ep20 30000, ep22 78000, ep25 48000, ep28 102000) have NO incumbent status (section 4).
- Zuxing's folder `/root/autodl-tmp/zuxing.lu` is read only: no write, no git command, none of his scripts run; copies
  with sha256 before / copy / after are in `COPIES_r7.tsv` and listed for him in `TELL_ZUXING_r7.txt`.
- Nothing goes to Kaggle from this round. Git: only a new branch `dev.haoran.t2r7_20261102` (section 6), never main,
  never force.

## 0. Relation to rounds 1-6

- Rounds 1-6 are frozen and not modified. Round 6's binding result is `SELECTION_R6.sha256` 06898401... (R6-CLOSEST).
  Round 7 replaces it: the seven episodes 00, 16, 19, 20, 22, 25 and 28 are re-decided from the mechanical pool alone,
  and the other 23 keep round 6's choice as incumbent unless a pool candidate beats it under round 6's rule.
- **Rules basis (unchanged):** challenge §2.8.c (every candidate was trained on a Tier-2-derived input: raw Tier 2 or
  its filters K3, K3fix, K3CA or G2); §2.9.d (one checkpoint per episode,
  identical digest within an episode); §2.9.b (the upload file is the trial closest to the mean of 10 full trials,
  never a best-of); Foundational 4.b (a frozen, algorithmic rule).

## 1. Cut-off and inputs

**Cut-off T_c = 2026-11-02 06:16:08 CST**, the start of the writing run of `scripts/t2cons_r7_pool.py resolve` (sha256
7a08fa6d...). Only checkpoint files written at least 300 s before T_c count (rule R3): Zuxing's five running
2026-11-01-tier2-g2-rest trainings (ep20, 22, 26, 27, 29; GPUs 5-7) enter with the checkpoints they had saved by then,
and nothing they write later enters.

| file | content |
|---|---|
| `inputs_r7/pool_r7.json` / `.csv` | the 419 files selected by the pool rule, each with job name, episode, STRICT flag, input, owner, machine, export tree, role (`screen` 154, `reuse` 245, `incumbent` 20, duplicate 0, refused 0), run directory, iteration, slot, sha256, evaluated file, training parquet + sha256, data root, frame reference + sha256; for `reuse` rows the round-6 job, export record and run records; for 5090 rows the as-trained parquet path, the SHA256SUMS value and the relocated file |
| `inputs_r7/runs_r7.json` / `.csv` | every scanned directory holding model files (114 pro6000) and the 45 5090 MANIFEST runs: eligible or not, with the reason; for the 118 eligible runs L, the targets and the picks |
| `inputs_r7/incumbents_r7.json` | the 23 incumbents (section 4.1): export record + exported file (re-hashed), training file, frozen variant-B consensus runs and m_P_frozen |
| `inputs_r7/r6refs_r7.json` | the 7 report-only round-6 references (section 4.2) |
| `inputs_r7/TRACKS_R7.sha256` | every training parquet and frame reference of the pool, incumbents and references, and the five data-root manifests (all 12fe4494...) |
| `inputs_r7/ZUXING_CODE_R7.tsv` | the training commit of each of Zuxing's 59 eligible runs and whether it is a pushed head |
| `inputs_r7/LS_REMOTE_R7.txt` | `git ls-remote --heads origin` of our clone (code/v2d_t2ready) at 06:15, 17 heads |
| `inputs_r7/resolve_r7.log` | the writing run's log |
| `inputs_r7/SNAPSHOT_R7.sha256` | every file above, `COPIES_r7.tsv`, `TELL_ZUXING_r7.txt` and every file in `zuxing_copies_r7/` |
| `COPIES_r7.tsv` | 513 copies: 152 models + 152 rsi files of the new Zuxing screen candidates, the 5090 folder's MANIFEST.tsv and SHA256SUMS, and for each of the 45 5090 runs its hydra/.hydra/{config,hydra,overrides}.yaml, final.json (if any) and the first `train_t2:` line of train.log (an excerpt file; the source log hashed before and after the read). source before == copy == after on every row |
| `RELOCATION_R7.tsv` / `.json` / `.sha256` | the 128 metadata-only relocations of the 5090 screen files (section 3.1) |

## 2. The candidate pool (mechanical; no score of any kind)

### 2.1 Rule (implemented by `scripts/t2cons_r7_pool.py`; round 6's rule plus R1b and the reuse role)

- **R1 pro6000 training-run directory.** A directory under `fc_runs/{C3r2_tier2, C3r2_tier2_armA_90k,
  C3r2_tier2_k3_stretch90k, C3r2_tier2_k3ca, C3r2_tier2_k3easy, C3r2_tier2_k3fix, T2fix}` or under Zuxing's `runs/*`
  holding `model_<N>.safetensors` files whose embedded config `output_dir` resolves to that same directory. Ours count
  only if the directory is in round 6's eligible run list (`runs_r6.json`, 59 runs). Zuxing's copy / retest / selection
  bundles (`2026-10-24-5090-best`, `2026-10-28-tier2-retest`, `2026-11-02-tier2-g2-ownsel`) are never walked: they are
  not training-run directories (round 6 refused them by R1; round 7 does not enter them). Directories named `hydra`,
  `wandb`, `fc_cache`, `figures`, `logs`, `eval*` or containing a guarded token are never entered.
- **R1b 5090 run.** A row of `zuxing.lu/runs/2026-11-01-round7-5090-ckpts/MANIFEST.tsv` (45 rows): directory `<run>`;
  the embedded `output_dir` of its last checkpoint == `/mnt/gfs/home/zuxinglu/workspace/runs/<run>` exactly; MANIFEST
  episode, data root and sequence == the checkpoint's own; every selected file's sha256 == its `SHA256SUMS` line.
- **R2 Tier-2 input.** The embedded training parquet lies in a Tier-2-derived data root: `data/human_motion_data_tier2`
  (raw), `_k3`, `_k3fix`, `_k3ca` (ours) or `zuxing.lu/data/human_motion_data_tier2_g2-2026-10-24` (G2); for 5090 runs
  the as-trained path `/mnt/gfs/home/zuxinglu/workspace/data/<root>/...` with one of those root names, and the same
  relative path must exist on this server.
- **R3 settled files, minimum length.** Only files with mtime <= T_c - 300 s; L = the largest such iteration; a run with
  L < 10,000 is not eligible.
- **R4 checkpoints per run (round 6's, unchanged).** K = 3. Targets tau_j = L (1 - j / 6), j = 0..3 (100 %, 83 %, 67 %,
  50 % of L). For each target, the saved iteration >= L/2 nearest to it (ties: the larger); duplicates collapse. So the
  final (last) checkpoint plus up to 3 evenly spaced checkpoints of the second half of what was saved. For runs of
  which only the last two files exist (22 of the 5090 runs keep only L - 500 and L), the picks are those two.
- **R5 roles** in the deterministic order (episode, owner, directory): `incumbent` = sha256 of the episode's round-7
  incumbent training file (not screened; section 4.1); `reuse` = sha256 of a round-6 phase-S screen candidate of the
  same episode whose training parquet and frame reference are unchanged (its round-6 export and 3 round-6 runs are
  reused, never re-run; job name = round 6's); `duplicate` = a sha256 selected earlier; every other file is `screen`
  (one export, 3 fresh runs). A file equal to a STRICT episode's round-6 pick would be `screen`, with no special status
  (none occurs).

K = 3 is fixed by the budget (section 8): it is the largest K that fits, and the only one whose targets contain all of
round 6's, so all 245 round-6 screens are reused.

### 2.2 Result at T_c

118 eligible runs (59 ours, 14 Zuxing pro6000, 45 Zuxing 5090), 419 selected files: **154 new screen candidates**, 245
reused round-6 candidates, 20 identical to an incumbent, no duplicate, no refusal.

| owner / run family | input | episodes | runs | screen | reuse | == incumbent |
|---|---|---|---:|---:|---:|---:|
| ours C3r2_tier2 (raw, 45k recipe) | raw | 00-29 | 30 | 1 | 112 | 7 |
| ours C3r2_tier2_k3fix (K3fix arm) | k3fix | 01 02 03 04 06 07 09 10 11 12 13 14 16 17 18 21 24 25 27 29 | 20 | 1 | 67 | 12 |
| ours ep03_raw_s7 | raw | 03 | 1 | 0 | 4 | 0 |
| ours C3r2_tier2_k3easy | k3 | 08 20 23 | 3 | 0 | 11 | 1 |
| ours C3r2_tier2_k3_stretch90k | k3 | 01 03 | 2 | 0 | 8 | 0 |
| ours C3r2_tier2_k3ca | k3ca | 26 | 1 | 0 | 4 | 0 |
| ours T2fix T2B / T2L | k3fix | 09 / 01 | 2 | 0 | 7 | 0 |
| Zuxing pro6000 2026-10-24-tier2-runin-g2 (ep04 now L = 450000) | g2 | 01 04 19 28 | 4 | 4 | 12 | 0 |
| Zuxing pro6000 runin-k3fix-ep04, runin-k3fix-ep05-ep26, runin-hard-g2 | k3fix / g2 | 01 04 05 19 26 | 5 | 0 | 20 | 0 |
| Zuxing pro6000 2026-11-01-tier2-g2-rest (training at T_c) | g2 | 20 22 26 27 29 | 5 | 20 | 0 | 0 |
| Zuxing 5090 2026-10-23-tier2-helpfix-hard (+ cont) | k3fix / raw | 01 04 15 19 23 28 | 11 | 22 | 0 | 0 |
| Zuxing 5090 2026-10-24-tier2-runin-g2 (+ cont) | g2 | 04 09 15 21 | 5 | 10 | 0 | 0 |
| Zuxing 5090 2026-10-24-tier2-runin-k3fix | k3fix | 00 22 | 2 | 4 | 0 | 0 |
| Zuxing 5090 2026-10-25-tier2-runin-hard-g2 (+ cont, cont2) | g2 | 01 19 | 5 | 12 | 0 | 0 |
| Zuxing 5090 2026-10-28-tier2-5090-fill (+ cont) | g2 / k3fix | 20 21 24 | 4 | 15 | 0 | 0 |
| Zuxing 5090 2026-10-28-tier2-runin-a5 (+ cont) | g2 / k3fix | 14 16 25 | 5 | 18 | 0 | 0 |
| Zuxing 5090 2026-10-29-tier2-5090-queue (+ cont) | g2 / k3fix | 02 06 08 24 28 | 6 | 23 | 0 | 0 |
| Zuxing 5090 2026-10-31-tier2-5090-round2 | g2 / k3fix | 09 11 14 15 17 21 27 | 7 | 24 | 0 | 0 |

(The two `screen` files of ours are round 5's ep04 raw and ep11 K3fix finals: round-6 incumbents then, replaced in round
6, so never screened.) Candidates per episode (screen + reuse; new screen in brackets): 00:6(2) 01:31(8) 02:11(4)
03:15(0) 04:22(11) 05:8(0) 06:11(4) 07:7(0) 08:11(4) 09:16(6) 10:7(0) 11:10(4) 12:7(0) 13:7(0) 14:17(10) 15:13(10)
16:12(4) 17:10(3) 18:7(0) 19:24(12) 20:16(8) 21:22(15) 22:10(6) 23:9(2) 24:15(8) 25:15(7) 26:16(4) 27:14(7) 28:19(11)
29:11(4). Full list: `inputs_r7/pool_r7.csv`.

### 2.3 Not candidates (by rule; recorded in `runs_r7.csv`)

Zuxing's single-checkpoint copies and bundles (2.1 R1); every run not trained on a Tier-2-derived input (26
directories, R2); 12 pro6000
directories whose output_dir is elsewhere (copies, forks); runs with L < 10,000 (K3 stretch ep04 9,000, Zuxing's two
pro6000 2026-10-28-tier2-runin-a5 folders 1,000 / 500); anything written after T_c - 300 s.

## 3. Evaluations

### 3.1 Protocol: round 6's (PREREG_R6 3.1), unchanged except GPUs, relocation and the job lists

- **Driver.** `scripts/t2cons_r7_eval.py` (new file; round 6's driver `t2cons_r6_eval.py` ada6e0f9 with the round-7
  bookkeeping; diff in `R7_SCRIPT_DIFFS.txt`), which loads `scripts/t2cons_eval.py` (2cf42786) and through it
  `scripts/t2best5_driver.py` (24b9af39) from their verified bytes. Launcher `scripts/t2cons_r7_eval_launch.sh`. Both
  are pinned in `PREREG_R7_PINS.md` before they export or evaluate anything.
- **Export.** Once per NEW screen job, CPU only, `python -m flash_chord.training.ppo.coordinate.export` of the owner's
  tree: `code/fc_r2` @ 61fa5d3 for ours, `code/nv-v2d_runin` @ b39b69b for Zuxing's (as rounds 1-6; his hard-start /
  5090 runs were trained at 82837c8 and rounds 2-6 exported such checkpoints with b39b69b without refusal). Tolerance
  1e-4, one fallback at 2e-4 on a tolerance refusal; stock verified, `source_sha256` == the evaluated input. A
  non-tolerance refusal ends the job; an `AMENDMENTS_R7.md` written before any round-7 score may add one export with the
  82837c8 exporter (zx_t2_20261101_fresh/code_82837c8, export.py blob 00ac657a).
- **5090 relocation (as rounds 2-6, AMENDMENTS_R2 A1).** A 5090 checkpoint stores the 5090 parquet path, which the stock
  harness cannot open. For each of the 128 5090 screen files, `t2cons_r7_pool.py relocate` wrote `relocated_r7/<run>/
  model_<N>.safetensors` in which exactly `task.parquet` and `reference.parquet` change from
  `/mnt/gfs/home/zuxinglu/workspace/data/<root>/<rel>` to this server's `<root>/<rel>` (ours for raw / K3fix, Zuxing's
  read-only G2 root for G2), using round 5's pinned `relocate_bytes` / `verify_pair` (t2cons_r5_relocate.py 09a6cb8e):
  every tensor byte, every other metadata key and every other config field (incl. output_dir) unchanged; the copy is
  unchanged afterwards. All 128 verify (`RELOCATION_R7.tsv`); the driver re-verifies each at every start. The relocated
  file is the evaluated input; the pool sha256 stays the original's. The 5090 copies of the data roots cannot be hashed
  from here; the target files are the ones every earlier round used for these inputs (TRACKS_R7) (disclosure 9.5).
- **Reused round-6 candidates and incumbents are never re-exported**: they keep their round-6 / earlier export record
  and file (re-hashed at every start), so a kept episode keeps its digest (§2.9.d).
- **Evaluation.** As round 6: `scripts/track2/eval_all.py` of the job's tree (657295921e82..., identical in both trees)
  with `--export no --world-count 1`, which runs the kit's `examples/run_policy_evaluation.py` and
  `flash_chord/scripts/evaluate_policy.py` (160d24f5) on the untouched stock runtime `code/video_to_data_8c778f9`.
  Frame-0 start, explicit reset, no help. `HUMAN_MOTION_DATA_DIR` = the job's data root; the reference = the job's own
  training parquet. K3CA rebinding as round 6.
- **ok checks, retries, integrity**: round 6's, unchanged (exit 0 and `evaluated`; results JSON + parquet; checkpoint
  digest == the exported file; reference == the training parquet by path and sha256; start frame 0, world count 1,
  explicit reset, finite; eval-script / eval-code digests; `environment_valid`; exported file unchanged; up to 3 tries
  >= 5 min apart; a completed evaluation is never re-run; kit + stock tree listing vs round 1's baseline, a change halts
  new starts).

### 3.2 Phase S (screening): 3 runs per NEW screen candidate

Jobs: the 154 `screen` rows of `pool_r7.json`. Each gets one export (`exports/<job>/`) and **R_S = 3** official runs
(`runs_s/<job>/run<k>/`). Order (decides nothing): exports in (episode, job) order, then run-major. 154 exports + 462
evaluations. The 245 `reuse` rows contribute their round-6 runs (`t2_consensus_r6_20261031/runs_s/<job>/run1..3`, 735
runs, frozen in `ROLLOUTS_S6.sha256` 1301c5c0...); they are scored again by the round-7 scorer (5.2), which must
reproduce round 6's `scores_s6/consensus_scores_s6.csv` values exactly (a consistency check, not a new measurement).

### 3.3 Phase F (final): 10 paired trials

Jobs: the 23 incumbents (4.1), the phase-S finalists of `final_r7.json` (5.4; written and locked by
`SELECTION_S7.sha256` before phase F starts) and the 7 report-only round-6 references (4.2). Each job gets **10 fresh
official runs** (`runs_f/<job>/run<t>/`); phase-S runs never enter phase F. Order: trial-major; within a trial the
decision jobs (incumbents, finalists) in (episode, job) order, then the 7 references. A new finalist evaluates its single
phase-S export; a reused finalist its round-6 export. (30 + n_f) x 10 evaluations.

### 3.4 Scheduling (decides nothing)

- **GPUs 6 and 7 only**, plain (no CUDA_MPS_* variables; never a GPU with an MPS server / client / control daemon; GPUs
  0-4 host the ablation MPS groups and are never used), at most ONE evaluation of ours per GPU (cap 2), at least 4096 +
  1536 MiB free. Zuxing's trainings and evaluation watchers on GPUs 5-7 are his and are never touched.
- **Lane locks**: round 6's rule (our `ROOT/evallock/gpu<g>.lock` + every existing shared eval-lane lock file of that GPU,
  once per file after following links, all or nothing, never blocking, opened read-only, never created); the shared
  list is round 6's (incl. `queue_runin/evallock`) plus round 6's own `evallock`,
  `t3_f0_final/report_5trials_20261101/evallock` and `tmp_analysis/tier3/f0_retry/evallock`.
- **Processes**: `nice -n 10` + `taskset` on the NUMA1 eval-lane cores (100-103, 204-207). 10 s cooldown per GPU after
  each of our evaluations (round 6: 35 s; shortened for the budget). Nothing is ever signalled; an evaluation past 2400 s
  is only flagged OVERDUE. `ROOT/STOP` stops new starts. Running evaluations of the t2_best5 / t3final / t1best20 /
  t1confirm drivers count toward our cap.

### 3.5 Seeds (records only)

As round 6: no selectable seed on the official path (the harness forces `env.reset.seed = 42`); repeated runs differ
only by GPU non-determinism. Recorded per run in `seeds.json` and `seeds_s7.tsv` / `seeds_f7.tsv`.

### 3.6 Dry runs (this step; nothing else is launched)

- **Plan-only dry runs** of the driver for phase S and phase F (pins, pool, relocation and incumbent checks, hulls,
  GPUs; nothing exported or evaluated).
- **Two official evaluations on GPU 6** (each policy against its own input), into `dryrun/`:
  1. phase-S path: **`ep00_z5k3fix_89500`**, the first relocated 5090 job in (episode, job) order (new plumbing:
     relocation, b39b69b export of an 82837c8 5090 checkpoint). Its export goes to `exports/ep00_z5k3fix_89500/` and stays
     the job's single export.
  2. phase-F path: **`ep01_zxg2_120000`**, the first reused round-6 candidate of Zuxing's in (episode, job) order (new
     plumbing: round-6 copy designation and round-6 export reuse on the finalist path).
  A dry-run evaluation is never a round-7 run or try, counts toward no limit, and sections 5-6 never read `dryrun/`. The
  two jobs were fixed by plumbing coverage (first in the deterministic order) before any value existed.
- Result: `DRYRUN_R7.md` (+ sha256). Phase S is launched afterwards with `bash scripts/t2cons_r7_eval_launch.sh
  --phase screen`.

## 4. Incumbents and references

### 4.1 The 23 incumbents (round 6's S6 outside STRICT)

For each episode outside STRICT = {00, 16, 19, 20, 22, 25, 28}, the incumbent P_e is round 6's pick (`selection_r6.json`
compositions S6), with its existing export and its frozen variant-B consensus AUCs from the round that selected it: the
21 unchanged round-5 incumbents keep round 6's record (round 1: 5 runs, `scores/consensus_scores.csv` 50cc6e94); the two
round-6 replacements (ep04, ep11) take their 10 round-6 phase-F runs (`scores_f6/consensus_scores_f6.csv` 74c9c885;
their mean == selection_r6.json m_F, checked). m_P_frozen = the float64 mean.

| ep | incumbent P_e | selected in | input | tree | export sha256 | frozen runs | m_P_frozen | + 0.01 |
|---|---|---|---|---|---|---:|---:|---:|
| 01 | `ep01_k3fix_final44922` | round1 | k3fix | fc_r2 | a15d9e85b0458c8d | 5 | 0.171319 | 0.181319 |
| 02 | `ep02_k3fix_final34093` | round1 | k3fix | fc_r2 | 3f1ae6373cfb8877 | 5 | 0.868138 | 0.878138 |
| 03 | `ep03_k3fix_final27068` | round1 | k3fix | fc_r2 | e042571ca8a01236 | 5 | 0.921527 | 0.931527 |
| 04 | `ep04_zxg2_421500` | round6 | g2 | b39b69b | ddb5605fdb14ca2a | 10 | 0.416757 | 0.426757 |
| 05 | `ep05_zx_k3fix_88000` | round1 | k3fix | b39b69b | 248dd05779f67e58 | 5 | 0.578153 | 0.588153 |
| 06 | `ep06_k3fix_final28420` | round1 | k3fix | fc_r2 | 84bd485806402cb2 | 5 | 0.797276 | 0.807276 |
| 07 | `ep07_k3fix_final18529` | round1 | k3fix | fc_r2 | b02cb8f6597f3b54 | 5 | 0.920452 | 0.930452 |
| 08 | `ep08_k3easy_final17496` | round1 | k3 | fc_r2 | 17d421a1938181f0 | 5 | 0.596363 | 0.606363 |
| 09 | `ep09_raw_final39328` | round1 | raw | fc_r2 | c73692c5b01ed77e | 5 | 0.760460 | 0.770460 |
| 10 | `ep10_raw_final34267` | round1 | raw | fc_r2 | 2490e879451fe43d | 5 | 0.412469 | 0.422469 |
| 11 | `ep11_raw_24000` | round6 | raw | fc_r2 | 00cf20569e0c1c75 | 10 | 0.807319 | 0.817319 |
| 12 | `ep12_k3fix_final23178` | round1 | k3fix | fc_r2 | b455bb4dc1707de7 | 5 | 0.543911 | 0.553911 |
| 13 | `ep13_k3fix_final39254` | round1 | k3fix | fc_r2 | bc146e3ea4a98084 | 5 | 0.665483 | 0.675483 |
| 14 | `ep14_k3fix_final44922` | round1 | k3fix | fc_r2 | 3c4740ef7bd4f60c | 5 | 0.330614 | 0.340614 |
| 15 | `ep15_raw_final44922` | round1 | raw | fc_r2 | d4bbb317cb71ebe6 | 5 | 0.729895 | 0.739895 |
| 17 | `ep17_raw_final31807` | round1 | raw | fc_r2 | 2188283c46ef28c8 | 5 | 0.647528 | 0.657528 |
| 18 | `ep18_k3fix_final18911` | round1 | k3fix | fc_r2 | de312489bafbc36a | 5 | 0.419129 | 0.429129 |
| 21 | `ep21_k3fix_final44922` | round1 | k3fix | fc_r2 | 0e6399890edfc6f6 | 5 | 0.444469 | 0.454469 |
| 23 | `ep23_raw_final20023` | round1 | raw | fc_r2 | 961b538e4679015e | 5 | 0.902867 | 0.912867 |
| 24 | `ep24_raw_final44922` | round1 | raw | fc_r2 | 7bffc41746d13913 | 5 | 0.895137 | 0.905137 |
| 26 | `ep26_zx_k3fix_92000` | round1 | k3fix | b39b69b | 25c600cb20fbd448 | 5 | 0.731484 | 0.741484 |
| 27 | `ep27_k3fix_final20863` | round1 | k3fix | fc_r2 | e560ebffd9545b11 | 5 | 0.897854 | 0.907854 |
| 29 | `ep29_k3fix_final17358` | round1 | k3fix | fc_r2 | 8c984002495d5a75 | 5 | 0.505675 | 0.515675 |

Provenance of these 23: round 1 picked from our finals and Zuxing's newest checkpoints (rule Z, ep05 / ep26), round 6
from its mechanical pool (ep04, ep11). (Consensus values, not leaderboard estimates.)

### 4.2 The 7 STRICT episodes: no incumbent; report-only round-6 references

Episodes 00, 16, 19, 20, 22, 25, 28 have no incumbent and no bar from earlier rounds. Their round-6 picks are recorded
in `r6refs_r7.json` and run in phase F only as **report-only references** (to report the paired noisy difference between
the round-7 set and round 6's set). They are never a bar, never a finalist, never a candidate (none of the seven files is
in the pool), and no number of theirs enters any decision.

| ep | round-6 pick (reference only) | first selected | input | export sha256 |
|---|---|---|---|---|
| 00 | `ep00_zxb_k3fix_60000` | round3 | k3fix | 45d1084dea7f6d0b |
| 16 | `ep16_zxb_g2_34000` | round5 | g2 | acec8f29690a40c3 |
| 19 | `ep19_zxb_g2hard_248000` | round5 | g2 | 960405ce1f92d6be |
| 20 | `ep20_zxb_k3fix_30000` | round5 | k3fix | 3b3dbf3cf5522a61 |
| 22 | `ep22_zxb_k3fix_78000` | round3 | k3fix | 66da2cd65061af36 |
| 25 | `ep25_zxb_k3fix_48000` | round5 | k3fix | 2d92f29dc238e784 |
| 28 | `ep28_zx_g2_102000` | round3 | g2 | 9ce8e92c1ed16618 |

## 5. Scores and selection (round 6's rule)

### 5.1 Scores, safeguards

Round 6's sections 5.1-5.3 apply unchanged: every scored run (phase S, reused phase-S runs, phase F) gets variant-B
consensus scores with the pinned round-1 functions (`t2cons_consensus.py` e1b943b1: `rollout_on_grid` against the job's
own input reference, `metrics_vs` with `align_initial_pose=False`, 500 released-mesh vertices, seed 0, float64) against
round 1's frozen `refs/B` (build.json 9646de2d), and the own-input validation gate (`t2cons_a1_gate.py` a29c997b
`auc_seed`: AUC within [min_s - 0.02, max_s + 0.02] over vertex seeds 0..9, |dMPPE| <= 0.5 cm, SP-SR and MP-SR equal).
The rollout files of each phase are frozen in `inputs_r7/ROLLOUTS_S7.sha256` (new and reused) / `ROLLOUTS_F7.sha256`
before the first score of that phase. floor = the number of runs with own-input SP-SR = 1 AND MP-SR = 1 in the same run.

### 5.2 Scorer

`scripts/t2cons_r7_consensus.py`, a new file: round 6's scorer (`t2cons_r6_consensus.py` 9b65e612) with the round-7
bookkeeping only (5.3-5.6: reuse rows, STRICT episodes, references, file eligibility). It loads the pinned round-1 / 2
modules from their verified bytes and calls `rollout_on_grid`, `metrics_vs`, `load_refs`, `auc_seed`, `stats`,
`floor_count` unchanged; same path guard as round 6 plus `tier2_eval.csv`, `tier2_runs.csv`, `summary.txt`,
`tier2_best_scan`, `eval_gen_local`. Commands `selftest`, `check`, `gate`, `score`, `select` per phase, as round 6. It is
pinned with its selftest in **`PREREG_R7_PINS_b.md` before it scores anything**.

### 5.3 Phase-S eligibility (as round 6)

A candidate c (role screen or reuse) is **S-eligible** iff: ok export, 3 ok official runs, none flagged by the gate, and
**floor_S(c) >= 2** (own-input SP-SR = MP-SR = 1 in at least 2 of its 3 runs). cons_S(c) = float64 mean of its 3
variant-B consensus AUCs. Ranking ties everywhere: lower population variance of the 3 consensus AUCs, then lower mean
consensus MPPE, then job name.

### 5.4 Phase-S selection

- **23 incumbent episodes (round 6's 5.4, unchanged).** Finalist f_e = the S-eligible candidate with the highest
  cons_S(c) among those with **cons_S(c) >= m_P_frozen(e) + 0.01** (exact floats), except that a reused candidate whose
  round-6 phase-F verdict against the SAME incumbent was "no replacement" keeps that verdict and is not a finalist again
  (ep12_raw_37575, ep13_raw_44922; recorded in pool_r7.json): a second fresh test of the same pair would be a second
  chance on the same question. No such c: no finalist.
- **7 STRICT episodes.** Finalists = the **two** highest-ranked S-eligible candidates by cons_S (one if only one is
  S-eligible); the rank-1 candidate is "the best eligible pool candidate", the rank-2 one its pre-registered fallback.
  If no candidate is S-eligible: the two highest by (floor_S desc, cons_S desc) among candidates with an ok export and 3
  ok, unflagged runs, marked `fallback_unconfirmed_S`. If no candidate has 3 ok, unflagged runs, the episode halts and
  an `AMENDMENTS_R7.md` written before any phase-F score decides.
- Outputs: `scores_s7/`, `selection_s7.{json,txt,md}`, `final_r7.json` (the phase-F finalists with their export
  records), and **`SELECTION_S7.sha256` last**.

### 5.5 Phase-F decision per episode

- **23 incumbent episodes (round 6's 5.5, unchanged).** m_F(f) = float64 mean of f's 10 phase-F consensus AUCs;
  m_P_fresh = mean of P's ok, non-flagged phase-F runs (>= 8 required, else +inf). **f replaces P iff** (a) f has 10 ok,
  non-flagged runs; (b) own-input SP-SR = MP-SR = 1 in >= 8 of them; (c) **m_F(f) >= max(m_P_frozen, m_P_fresh) + 0.01**
  (exact floats). Otherwise P stays.
- **7 STRICT episodes.** A finalist is **confirmed** iff it has 10 ok, non-flagged phase-F runs and own-input SP-SR =
  MP-SR = 1 in >= 8 of them. Pick = the rank-1 finalist if confirmed, else the rank-2 finalist if confirmed. If neither is
  confirmed: among the finalists with >= 8 ok, non-flagged runs, the one with more SP=MP=1 runs, then higher m_F, then
  phase-S rank; marked `unconfirmed` in every report. If neither has >= 8 ok runs, the episode halts (amendment before
  any phase-F composite is computed).
- Nothing else enters: no own-input AUC compared across inputs, no reference (4.2) value, no manual override.

### 5.6 The set, R7-CLOSEST, eligibility

- **S7** = per episode the 5.5 result. **REF6** = S7 with the 7 STRICT episodes' jobs replaced by their round-6
  references (= round 6's S6 wherever round 7 keeps the incumbent); report only.
- Trial t is complete for a set if every episode's job has an ok, non-flagged run t. Per complete trial: the 30-episode
  means of own-input AUC, SP-SR, MP-SR, MPPE (evaluator values) and of consensus-B AUC, SP-SR, MP-SR, MPPE.
- **R7-CLOSEST** = the complete trial t of S7 whose 30-episode mean OWN-INPUT AUC is closest to the mean of that quantity
  over the complete trials of S7; ties: smaller distance of the consensus-B mean to its trial mean, then lower t. Never a
  best-of.
- **Eligibility.** R7-CLOSEST is the upload-eligible Tier-2 file under the STRICT standard. No comparison with round 6
  decides this. Reported next to it, report only: Delta_B =
  mean over trials complete for both of [S7 - REF6 30-episode mean consensus-B AUC], and the own-input SP / MP means.
- Outputs: `scores_f7/`, `selection_r7.{json,txt,md}`, `summary_r7.{json,md}` and **`SELECTION_R7.sha256` last**; from
  that moment round 7 is final.

## 6. The file (R7-CLOSEST): packing and the submission commit

- R7-CLOSEST = trial t* of S7: per episode the S7 job's run t* rollout. Packed with the kit packer unchanged
  (`tools/pack_submission.py` 47a45895), each episode against its own input frame reference (raw: the packer default;
  K3 / K3fix / K3CA / G2: byte copies of the frame references in TRACKS_R7), into `pack_r7/`, checked with the kit's
  offline validator. Pack script `scripts/t2cons_r7_pack.py` (round 6's with round-7 bookkeeping), pinned in
  `PREREG_R7_PINS_c.md` before its first use.
- **No placeholder in the final file.** The submission commit goes to the NEW branch `dev.haoran.t2r7_20261102` only
  (never main, never force): our training tree (fc_r2 61fa5d3 content, as in dev.haoran.c3r2 41403d3), the K3 / K3fix /
  K3CA input builders, Zuxing's training commits as pushed heads (`dev.zuxing.speedup` bc3b1bd, `dev.zuxing.hard_start`
  82837c8), the training records of every selected run (hydra, final.json, train_t2 line, read-only copies), the
  selection / pack / relocation scripts and these records. The file is packed with that commit's URL; its pose columns
  must be byte-identical to a placeholder pack of the same trial (a check, not a second file for upload). Items that
  cannot be put in the commit from this server are listed for the user (expected: Zuxing's G2 data builder
  `tmp/t2/g2data/` and its 2026-11-01 rest build, which are in no pushed commit; whether his worktrees had uncommitted
  changes). Nothing is uploaded to Kaggle.

## 7. (not included in this copy)

## 8. Budget (and why K = 3)

- Phase S: 154 new candidates x 3 = **462 evaluations** (+154 CPU exports, ~25 s each, overlapping); the 245 reused
  candidates cost 0 evaluations (735 round-6 runs, re-scored on CPU).
- Phase F: (23 incumbents + 7 references + n_f finalists) x 10 = 300 + 10 n_f, n_f <= 14 (STRICT) + 23 = 37, so at most
  670; expected n_f ~ 14 + 4..8 (round 6 had 4 finalists over 30 episodes) -> ~480-520.
- Total ~950-990 expected, **1,132 at most**.
- Capacity: 2 lanes (GPUs 6 and 7), one evaluation each. Measured on these GPUs today (zx_t2_20261101_fresh, 150
  evaluations in 61 min): median 41 s (GPU 6) / 61 s (GPU 7) per evaluation; with the 10 s cooldown and launch overhead
  ~55 s / ~75 s per lane cycle -> ~65 + 48 = **~110 evaluations per hour** (the planning figure of 60-70 s per evaluation
  per lane gives 103-120 per hour).
- Wall time: expected ~970 / 110 = **~8.8 h**; worst case 1,132 / 103 = **~11.0 h**; plus ~0.5 h CPU scoring and the
  phase-S selection between the phases: **~9.5-11.5 h**, inside the 10-12 h target.
- Why K = 3: K = 4 (targets 100 / 87.5 / 75 / 62.5 / 50 %) keeps only the 100 % and 50 % targets of round 6, adding
  about 3 new files per run over 118 runs (~350 files, ~1,050 more phase-S evaluations, > 20 h); K = 2 (100 / 75 / 50 %)
  adds a 75 % file per run (~118 new files) while dropping round 6's 83 / 67 % ones, i.e. more evaluations for fewer
  checkpoints. K = 3 is the largest K that fits and reuses all of round 6's screening.
- If phase S runs long, `ROOT/STOP` pauses it (running evaluations finish); nothing is skipped or shortened without an
  amendment written before the affected scores.

## 9. Disclosures

1. **The 23 incumbents carry earlier rounds' choices** (round 1, round 6). The 7 STRICT episodes are
   decided from the pool alone; their round-6 picks are run only as report-only references.
2. **Files read when writing this file:** for the incumbent records, round 6's phase-F records (selection_r6.md /
   .json: consensus AUCs of the 30 round-6 incumbents and its 4 finalists) and the round-6 export records; to size the
   budget, the `seconds` columns of round 6's and zx_t2_20261101_fresh's evaluation logs; a directory listing of
   Zuxing's runs/, which shows the file names of his retest copies (2026-10-28-tier2-retest/candidates_2026-10-31 and
   _2026-11-01, e.g. 5090_ep02_it168000). Those folders are excluded by rule R1, and none of those files is selected by
   R4 unless it is a target of its full run. The rule (K, second half, 10k minimum, reuse) is round 6's, sized by the
   budget.
3. **Zuxing's 5090 folder.** `2026-11-01-round7-5090-ckpts` was assembled by Zuxing for this round (45 runs). For 22 of
   them it holds only the last two checkpoints (L - 500, L); for the others every 10k plus the last two. The pick rule
   (iteration-based) then selects from what is there. Whether the folder holds every Tier-2 run he trained on the 5090,
   and why some runs keep only two files, cannot be checked from here (TELL_ZUXING_r7.txt asks). Eleven of its runs were
   still training when copied ('training' in MANIFEST): their L is the last copied file.
4. **G2 rest build.** `human_motion_data_tier2_g2-2026-10-24` gained the episodes 00 02 03 07 12 17 18 20 22 26 27 29 on
   2026-11-01 (build_ep00_02_03_07_12_17_18_20_22_26_27_29.json); the g2-rest runs (ep20/22/26/27/29) train on it.
   The 11-01 build covers the remaining episodes. No earlier G2 parquet or frame reference used by a reused candidate
   changed (checked by R5).
5. **Relocation and data identity.** The relocation changes two path strings only (verified byte for byte). That the
   5090's copies of the K3fix / G2 / raw roots equal ours was shown for the same roots by rounds 3 and 5 (SOURCE.json
   records); the 5090 bytes themselves are not visible from here. The evaluator scores each policy against this
   server's parquet in any case.
6. **Reference bias (as PREREG_R6 9.4).** refs/B use tracks R, F everywhere, C (K3CA) on 0, 5, 15, 19, 22, 26, 28 and
   G (G2) on 1, 4, 9, 19, 21, 28. G2 candidates on other episodes (e.g. the g2-rest runs, 5090 G2 runs on 06 08 15 16 20
   24) and K3 candidates are not favoured there; raw and K3fix always are.
7. **Noise and selection.** Phase S picks maxima over up to 31 noisy 3-run means per episode; decisions use fresh
   phase-F runs, the +0.01 margin over the higher of two incumbent baselines (23 episodes), and the 8/10 floor
   (STRICT). Runs are single-seed repeats.
8. **Exporters.** fc_r2 for ours, b39b69b for Zuxing's (his 5090 and hard-start runs were trained at 82837c8).
9. **Training code** (`ZUXING_CODE_R7.tsv`): ours `code/fc_r2` @ 61fa5d3 (in pushed dev.haoran.c3r2 41403d3) plus the
   K3 / K3fix / K3CA builders; Zuxing's pro6000 runs bc3b1bd (`dev.zuxing.speedup`, pushed) or 82837c8
   (`dev.zuxing.hard_start`, pushed); all 45 5090 runs 82837c8 (first train_t2: line == MANIFEST commit, pushed). His
   G2 builder is in no pushed commit. Whether his worktrees had uncommitted edits is not checkable from here.
10. **Where the text was made.** The new scripts and this file were written in the agent's Mac scratchpad and uploaded;
    nothing of the project ran on the Mac except syntax checks.

## 10. Implementation, hard rules, amendments

| step | what | code |
|---|---|---|
| P | pool + incumbents + references + copies (done, T_c), relocation (done) | `scripts/t2cons_r7_pool.py resolve`, `relocate` |
| R1 | this file + `PREREG_R7.sha256` | |
| R2 | driver + launcher, plan-only dry runs, `PREREG_R7_PINS.md`, the two dry-run evaluations, `DRYRUN_R7.md` | `scripts/t2cons_r7_eval.py`, `scripts/t2cons_r7_eval_launch.sh` |
| R3 | phase S: 154 exports + 462 evaluations | same, `--phase screen` |
| R4 | scorer pinned (`PREREG_R7_PINS_b.md`), selftest, gate + scores + selection of phase S, `SELECTION_S7.sha256` | `scripts/t2cons_r7_consensus.py` |
| R5 | phase F: 10 trials of 23 incumbents + finalists + 7 references | driver `--phase final` |
| R6 | gate + scores + decision + S7 + R7-CLOSEST, `SELECTION_R7.sha256` | scorer |
| R7 | pack R7-CLOSEST (pinned in `PREREG_R7_PINS_c.md` first), validator, submission commit on `dev.haoran.t2r7_20261102`, re-pack with its URL | `scripts/t2cons_r7_pack.py` |

Hard rules: new files only, under this folder and `scripts/t2cons_r7_*`; rounds 1-6, t2_best5, the kit, the stock
runtime and the code trees are only read; Zuxing's folder only read; never edit a file a running process uses; never
signal any process; GPUs 6 and 7 only, plain, no MPS; flashchord Python `-I -B` for the driver and resolver, kit Python
`-I -B` for the scorer and packer. Every export file is kept (export digests must match the artifacts). Any deviation goes
into `AMENDMENTS_R7.md` (+ sha256) with time and reason before the affected step; an amendment written after any round-7
score was visible must say so. PINS addenda may only add pins and notes.

## 11. Pins

Path rules: a path starting with `/` is absolute; `scripts/...` and `tmp_analysis/...` are relative to
`/root/autodl-tmp/haoran.zhang`; any other path is relative to this folder.

pin scripts/t2cons_r7_pool.py 7a08fa6d8be91140e139d914b8119352fabb05f58b415eb4a8904f9a02733770
pin inputs_r7/pool_r7.json 7c4b553bf2d15ffdae0d2b19097ab6abccfba03d1575b268fb3c0338b0f99464
pin inputs_r7/pool_r7.csv 3f65a9109ed7b72512070825472bee3483507ad84d109edb26e8c284100aab3a
pin inputs_r7/runs_r7.json 155432e2c0b40653f7d4c060ca9a7627ab99a2f392c2facfbac3e8da6cde30cf
pin inputs_r7/runs_r7.csv e14c3a01f47b82839c218b8b0e80f6dcd9b56de81b9ba80a0d616c7bcb2c79e9
pin inputs_r7/incumbents_r7.json cb55fa7a886e5f62f9dbf6f86771fe89c73c6d2137e38c39f82a7a7e72b77e82
pin inputs_r7/r6refs_r7.json 0bc58249b13827bfa843747658378df9d7d045bb0f0427e438126c4a0851df03
pin inputs_r7/TRACKS_R7.sha256 4c7383266df8350dce686aaa800e2d46c2076f5ee3399c071b5eee12b8ee7788
pin inputs_r7/ZUXING_CODE_R7.tsv 90cd0a0e672f0ab7bebe00296ad1a6936593a178406d2b2871db01a5cd157e76
pin inputs_r7/LS_REMOTE_R7.txt f1c96f90d46132a972b39a536c1cda85073253588950b98f9131054df03f3f27
pin inputs_r7/resolve_r7.log db716297d26b614515d20de280dfd532dcb620119b5b7d2ce4c41864455be53f
pin inputs_r7/SNAPSHOT_R7.sha256 8ca757b7f0267d027c76f93dda3ae630499e6056a3659545b8b3f2ef213def33
pin COPIES_r7.tsv 028f0139909ecf3e2c0ea84c676bf91dcf07fad6a209c716eb241a436075b954
pin TELL_ZUXING_r7.txt 032c2a8f6931981fce30958b7c1ea67ed4c201c7b7349bf62dbe1fca356e7329
pin RELOCATION_R7.tsv ccd24c5d9b367ad68cf6d9c3c8fbc47affd5d9a75ca8ac1b292ea350eb74079c
pin RELOCATION_R7.json 8db73660fce23db1b3fc89001186f9d44c4479d24a8b125ae3c163c6205a51de
pin RELOCATION_R7.sha256 3494077485b15fcc9d663aa14649986b853ba7776c0a0b3c8f1a1e66e3830639
pin scripts/t2best5_driver.py 24b9af39a2e2708641e7870447b12b6475c23aec9cd18ba71086768dd69cfc89
pin scripts/t2cons_eval.py 2cf427866a9432a3bd1b8fbb7864149d4258a5d19100c20da92d32bfe4fbfa9b
pin scripts/t2cons_consensus.py e1b943b10fd1db13ab6f78e5bde7b3c6e6c48f7daeee6230c32c463d14ac41b4
pin scripts/t2cons_a1_gate.py a29c997b7271e9299db109e97af5210f528484719210bb58685523b0ae0eb16b
pin scripts/t2cons_r2_consensus.py ddee0b8b6fa25029275b21a3dc82d15c36f4de71c0a61f2a1046869b01ec13aa
pin scripts/t2cons_r5_relocate.py 09a6cb8e8a3f8e835c4b10caa37f6ad46b780374480e856cca7efbe479938bdb
pin scripts/t2cons_r6_pool.py 13974f6f48abd5ac8cead9aec5f7e99452de8ed9c1d457d8a5433391b8fd7905
pin scripts/t2cons_r6_eval.py ada6e0f98785d3dda2709d03d1896151634481aafb562be8561e2af718f401aa
pin scripts/t2cons_r6_eval_launch.sh afeadbd4c309e1c535c3a1e3648391789f18b1ceb215c175a9a58bddf9389ae9
pin scripts/t2cons_r6_consensus.py 9b65e612fb289cd8e7f7584ffb7c7ccb40e1d2de9235f0221a70a99c11382d31
pin scripts/t2cons_r6_pack.py da02d1b48c076ce3f5f4987df591c643452697abc17edb1888b1be5b94b9748b
pin tmp_analysis/t2fix/cache_keys_t2.py 9f64214eba0eb6602e8ccb9cc200fe9c6ff94676e0c99551f095b3b1db5c1aae
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r6_20261031/PREREG_R6.md 0b0fbf9a2b581913ae0da7d36ce76aa785e706c132dc496054f2323a80e487e1
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r6_20261031/SELECTION_R6.sha256 0689840123a6de7f6bcd82e4a179f81df215d7c4db4af711ff311fc89fdc8f30
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r6_20261031/SELECTION_S6.sha256 77b17c3c64c6425aeb2147a259095ad1c6be4899ba6470dc6fd701f4869f516e
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r6_20261031/selection_r6.json 506b601e3b6ad4d79c4056e9efabeb4051e5c9c592a102f29e4db5266be6b308
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r6_20261031/final_r6.json b27f97b75b8531ed5bfe45c5b09966f7c28cd1d41200ef4921b9e853b4cb42da
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r6_20261031/selection_s6.json 1d68db0bd3cf8145e261085ca92c02110a356c70f9ef8b284fc4b0f9b14e9dbb
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r6_20261031/scores_f6/consensus_scores_f6.csv 74c9c8856d5f2fd671b1e3f97763a03c26936be794c581581972c2d837e84ab4
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r6_20261031/scores_f6/consensus_job_stats_f6.json 9eaac6077960d06975597a385552b459df5ff2d76e6c7e39bd9b7243ee3f24d7
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r6_20261031/scores_f6/own_input_gate_f6.csv c24010e47680959ad2aae5135f66908aa1347a77923604f15e2c01d7693dff45
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r6_20261031/scores_s6/consensus_scores_s6.csv d32c1ed7827257f7fa9ed1d4fa960e432ce510a4ee24f5af491ef89e3be9993c
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r6_20261031/scores_s6/consensus_job_stats_s6.json 39d79b6bac2af080371ad7e8aa0241511ca61c5672c2f9d61e07f9dbb6a878db
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r6_20261031/scores_s6/own_input_gate_s6.csv 31fabb2925280fbc9db0f0398d86f15eb8d74350b0ed5c04f52eeffc140a7aec
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r6_20261031/inputs_r6/ROLLOUTS_S6.sha256 1301c5c0fa0a41e25042f5880b6941875f9b43689f2c73675130b800db253add
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r6_20261031/inputs_r6/ROLLOUTS_F6.sha256 68b4ebe99b9811a59bb6deabc1c417788880ca11117d32223f996da1de66625d
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r6_20261031/inputs_r6/pool_r6.json c2aeb79af68cda31d8daa9f324bc26189c721150ec781f62a775782403082de6
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r6_20261031/inputs_r6/runs_r6.json c7e529e2b7f16fef73e1a42cf8db9e589dc68875f4a0f4aba02d06d2089032d5
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r6_20261031/inputs_r6/incumbents_r6.json 44d62d8cbd5b2d4247307b301902fb47dc42032c4227dcbefb970056d9d07b0b
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r6_20261031/COPIES_r6.tsv fa17929cf145cbb8333464b559521f699a2712319bd6b9833e017dea92d000f9
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r6_20261031/integrity/baseline.json ef9b5035c09b8ad40713d9fb08470eeccb2abe8845d4d1cce3037c49da1cd622
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r6_20261031/manifest_s6.json 0f056c994dfa589445a9dab588e9ff32236a7bb048e8bff94d8ffd2cf3c882cf
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r6_20261031/own_input_evals_s6.csv d152c7b3de01c5c15112904338917243decb28f61763b21b116adb0a7c4f5e40
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r5_20261029/SELECTION_R5.sha256 58e8b54b96ee594e587f76793aaf592c5a23dc0413d8a15546052e996af29bf0
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r5_20261029/selection_r5.json 89ca15098c03c465be1a5f78b7d4342eac373b15ddfa58f823a6fc416a464e5f
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r5_20261029/PREREG_R5.md 870aef4b1b9643fe68262954addd7b387e334690426fb3d70a799b072fdd2d81
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r5_20261029/scores_r5/consensus_scores_r5.csv a17eac41895797231edf6800a1ea35d5a2d454e75740f1393103ad948df27157
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r5_20261029/scores_r5/consensus_job_stats_r5.json b1e8e4e1aa749e22a2de17fd08d5edde3a19f441707b1694a09c46d65bb2607c
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r3_20261028/scores_r3/consensus_scores_r3.csv 2faa6e3b253b439e2c6e7d4a51386840800f69a035726d2f2651cda919fd7496
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r3_20261028/scores_r3/consensus_job_stats_r3.json 88d8e26a454c9cad657aa4c49bf633132defc1e33785ebbbb2caf790130b5cd9
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_20261024/scores/consensus_scores.csv 50cc6e9401ad8c88594918490cee345d8aaf7368740fb3626c65525bfcb40a52
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_20261024/scores/consensus_job_stats.json 9e255057556ccc23df9c97dafd101219770e8b53cc6ab28519ab893eb86319c8
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_20261024/refs/build.json 9646de2daf02301dd2efbc72a7507c539b1100c3ec0fe39adade169195d18dd3
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_20261024/integrity/baseline.json ef9b5035c09b8ad40713d9fb08470eeccb2abe8845d4d1cce3037c49da1cd622
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_20261024/PREREG.md 3eee7268d53d1c3669614c63b69a54fa2462cf624ed88c3f7e215f5cb9f51ee9
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_20261024/SELECTION.sha256 651853a61bac0ab2c244a0e7ebed9db2d46d90f7d22d95f12fd545cda6b25046
pin /root/autodl-tmp/haoran.zhang/v2d_submission_kit_src/v2d_submission_kit/flash_chord/scripts/evaluate_policy.py 160d24f5fbeb70ccd569ca4192056f1892b2a9f2999066db9d8279461648fdf1
pin /root/autodl-tmp/haoran.zhang/v2d_submission_kit_src/v2d_submission_kit/examples/run_policy_evaluation.py 0bcc41f5de246a51f194409b3e5292e04109012cb22a433a8f7ed477df04e55d
pin /root/autodl-tmp/haoran.zhang/v2d_submission_kit_src/v2d_submission_kit/tools/pack_submission.py 47a4589574ff8284ba4376ef31ac5548c8fe87ead753be2d3b43d6c87041e5be
pin /root/autodl-tmp/haoran.zhang/v2d_submission_kit_src/v2d_submission_kit/v2dlb/object_pose.py c6915924a322540f00c69c75b2ae93ae455bb6efb8a41cc0a96c967f64d7e2c0
pin /root/autodl-tmp/haoran.zhang/v2d_submission_kit_src/v2d_submission_kit/v2dlb/vertices.py 4c95a4d65d37ea97d6c58388b273c277cad9d7cfa63d3cdb7af2ec3d1f1a0262
pin /root/autodl-tmp/haoran.zhang/v2d_submission_kit_src/v2d_submission_kit/metric_code/track_2/AUC.py 58b437697c32c86497ff5a306fe586014c3429b0d2c18e1d3094a6f503479339
pin /root/autodl-tmp/haoran.zhang/code/nv-v2d_runin/robotic_grounding/flash_chord/scripts/track2/eval_all.py 657295921e820a464160039ad9a65ba92a683108a99dcb197290ae4147363197
pin /root/autodl-tmp/haoran.zhang/code/fc_r2/scripts/track2/eval_all.py 657295921e820a464160039ad9a65ba92a683108a99dcb197290ae4147363197
pin /root/autodl-tmp/haoran.zhang/code/nv-v2d_runin/robotic_grounding/flash_chord/src/flash_chord/training/ppo/coordinate/export.py 04367c56564de9540deb3d8e33adbfddb7800df190361a0576005a1a4ccfd212
pin /root/autodl-tmp/haoran.zhang/code/fc_r2/src/flash_chord/training/ppo/coordinate/export.py 34e3d7f3a7dca4fdd1e7c4d26750c6332abcc1d629f6a9c6c7a965833a22206f
pin /root/autodl-tmp/haoran.zhang/code/video_to_data_8c778f9/robotic_grounding/flash_chord/src/flash_chord/evaluation/harness.py 6e61b092e9aa9d00d11c557b8508c09ef990314a3f68b0779bbf97765c85ddc2
commit code/nv-v2d_runin b39b69be6c2f561c0123c6aeb8cd13f8b9d542e7
commit code/video_to_data_8c778f9 8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef
commit code/fc_r2 61fa5d31ff8054bf1e5cf5375ef0957e54d2729b
