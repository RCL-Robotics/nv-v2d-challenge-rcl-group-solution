# Tier-2 consensus selection, round 6 (t2_consensus_r6_20261031): pre-registration

**When.** Written 2026-10-31 from about 01:15 CST, after the mechanical pool resolution at the cut-off T_c (section 1) and
BEFORE any round-6 export, evaluation, score or selection. At writing, this folder held only:
- `inputs_r6/` (resolver outputs, section 1, plus `LS_REMOTE_R6.txt` captured just before T_c and the resolver's stdout
  copy `.resolve_stdout.txt`), `zuxing_copies_r6/` (72 sha256-verified copies), `COPIES_r6.tsv`, `TELL_ZUXING_r6.txt`;
- no `exports/`, `runs_s/`, `runs_f/`, `scores_*`, `dryrun/` or `evallock/`.

Every later step refuses to run if this file's sha256 differs from `PREREG_R6.sha256`.

**Request** (2026-10-31): make the updated Tier-2 results ready for submission (without submitting them) and select
the best Tier-2 checkpoints through the official evaluator. "Through the official evaluator" here means: the kit's own
evaluation of each policy, scored against two kinds of reference (each policy's own Tier-2-derived input, and round
1's frozen noisy consensus reference). Nothing is uploaded or pushed from this folder.

**Rules**
- Selection uses only own-input and consensus-B (noisy) scores. Zuxing's files enter only through the mechanical rule
  of section 2 and compete on noisy-side scores.
- Zuxing's folder `/root/autodl-tmp/zuxing.lu` is read only: no write, no git command, none of his scripts run; copies
  with sha256 before / copy / after are in `COPIES_r6.tsv` and listed for him in `TELL_ZUXING_r6.txt`.
- Nothing goes to Kaggle, GitHub, HuggingFace or W&B from this folder. The user decides any upload.

## 0. Relation to rounds 1-5

- Rounds 1-5 are frozen and not modified. The binding Tier-2 result is round 5's **R5-CLOSEST** file (file sha256
  e8a03a4d..., placeholder commit URL), defined by `SELECTION_R5.sha256` (58e8b54b...) and `selection_r5.json`
  (89ca1509...). Its composition ROUND5 is this round's incumbent set (section 4).
- **Why a new pre-registration.** Rounds 2-5 tested short candidate lists of single checkpoints nominated by Zuxing.
  Round 6 replaces nomination by a **mechanical pool** over every Tier-2 training run on
  this server (section 2), screens it with official evaluations, confirms the per-episode winner with fresh paired
  evaluations, and runs the whole selected set 10 times (sections 3, 5, 6).
- **Rules basis (unchanged):** challenge §2.8.c (every candidate was trained on a Tier-2-derived input: raw Tier 2 or
  its filters K3, K3fix, K3CA or G2); §2.9.d (one checkpoint per episode,
  identical digest within an episode); §2.9.b (the organisers may re-run the pipeline: the upload file is the trial
  closest to the mean of 10 full trials, never a best-of); Foundational 4.b (a frozen, algorithmic rule).

## 1. Cut-off and inputs

**Cut-off T_c = 2026-10-31 01:09:17 CST**, the start of the writing run of `scripts/t2cons_r6_pool.py` (sha256
13974f6f48abd5ac...). Only checkpoint files written at least 300 s before T_c count (rule R3), so files Zuxing's
running ep04 G2 training (pid 638820, GPU 7) writes later never enter.

| file | content |
|---|---|
| `inputs_r6/pool_r6.json` / `.csv` | the 271 files selected by the pool rule, each with job name, episode, input, owner, export tree, role (`screen` 250, `incumbent` 21, duplicate 0, refused 0), run directory, iteration, slot, sha256, the file evaluated (ours: the original; Zuxing's: our copy), training parquet + sha256, data root, frame reference + sha256 |
| `inputs_r6/runs_r6.json` / `.csv` | every scanned directory holding model files (626): eligible or not, with the reason; for the 68 eligible runs L, the targets and the picks |
| `inputs_r6/incumbents_r6.json` | the 30 round-5 incumbents (section 4): export record + exported file (re-hashed), training file, manifest row, frozen variant-B consensus runs and m_P_frozen |
| `inputs_r6/TRACKS_R6.sha256` | every training parquet and frame reference of the pool and the incumbents, and the five data-root manifests (all 12fe4494...) |
| `inputs_r6/ZUXING_CODE_R6.tsv` | Zuxing's runs / incumbents and the training code they need (section 9.4) |
| `inputs_r6/LS_REMOTE_R6.txt` | `git ls-remote --heads origin` of our clone at 01:09:17 (13 heads) |
| `inputs_r6/resolve_r6.log` | the writing run's log |
| `inputs_r6/SNAPSHOT_R6.sha256` | every file above, `COPIES_r6.tsv`, `TELL_ZUXING_r6.txt` and the 72 copies |
| `COPIES_r6.tsv` | 72 copies (36 models + 36 rsi of the same iterations) from 9 of his pro6000 runs; source before == copy == source after on every row |

## 2. The candidate pool (mechanical; no score of any kind)

### 2.1 Rule (implemented by `scripts/t2cons_r6_pool.py`)

- **R1 training-run directory.** A directory under `fc_runs/{C3r2_tier2, C3r2_tier2_armA_90k, C3r2_tier2_k3_stretch90k,
  C3r2_tier2_k3ca, C3r2_tier2_k3easy, C3r2_tier2_k3fix, T2fix}` or under Zuxing's `runs/*` holding `model_<N>.safetensors`
  files whose embedded config `output_dir` resolves to that same directory (the trainer wrote them there). Copies,
  forks, bundles, re-test and evaluation folders are therefore not runs. Directories named `hydra`, `wandb`, `fc_cache`,
  `figures`, `logs`, `eval*` or containing a guarded token are never entered.
- **R2 Tier-2 input.** The embedded training parquet lies in a Tier-2-derived data root: `data/human_motion_data_tier2`
  (raw), `_k3`, `_k3fix`, `_k3ca` (ours) or `zuxing.lu/data/human_motion_data_tier2_g2-2026-10-24` (G2).
- **R3 settled files, minimum length.** Only files with mtime <= T_c - 300 s count; L = the largest such iteration; a
  run with L < 10,000 is not eligible (smoke or aborted runs).
- **R4 checkpoints per run.** K = 3. Targets tau_j = L (1 - j / 6), j = 0..3 (100 %, 83 %, 67 %, 50 % of L). For each
  target, the saved iteration >= L/2 nearest to it (ties: the larger); duplicates collapse. So the final (last)
  checkpoint plus up to 3 evenly spaced checkpoints of the second half. For every finished run j = 0 is its final.json
  checkpoint.
- **R5 roles.** A file whose sha256 equals the episode's round-5 incumbent training file is `incumbent` (not screened;
  section 4). A sha256 already selected earlier in the deterministic order (episode, owner, directory) is `duplicate`.
  Every other selected file is a `screen` candidate.

K = 3 was fixed by the evaluation budget (section 8).

### 2.2 Result at T_c

68 eligible runs (59 ours, 9 Zuxing's pro6000 runs), 271 selected files: **250 screen candidates**, 21 identical to the
incumbent (our run finals that are round-5 picks), no duplicate, no refusal.

| owner / run family | input | runs | screen | incumbent |
|---|---|---:|---:|---:|
| ours C3r2_tier2 (raw, 45k recipe) | raw | 30 | 113 | 7 |
| ours C3r2_tier2_k3fix (K3fix arm) | k3fix | 20 | 67 | 13 |
| ours ep03_raw_s7 (raw seed-7 control) | raw | 1 | 4 | 0 |
| ours C3r2_tier2_k3easy | k3 | 3 | 11 | 1 |
| ours C3r2_tier2_k3_stretch90k (ep01, ep03; stopped) | k3 | 2 | 8 | 0 |
| ours C3r2_tier2_k3ca (ep26) | k3ca | 1 | 4 | 0 |
| ours T2fix T2B ep09 (rotation-weight fork) | k3fix | 1 | 3 | 0 |
| ours T2fix T2L ep01 (90k continuation) | k3fix | 1 | 4 | 0 |
| Zuxing 2026-10-24-tier2-runin-g2 (ep01, ep04, ep19, ep28) | g2 | 4 | 16 | 0 |
| Zuxing 2026-10-25-tier2-runin-hard-g2 (ep01, ep19) | g2 | 2 | 8 | 0 |
| Zuxing 2026-10-24-tier2-runin-k3fix-ep05-ep26 (ep05, ep26) | k3fix | 2 | 8 | 0 |
| Zuxing 2026-10-24-tier2-runin-k3fix-ep04 (12k, stopped) | k3fix | 1 | 4 | 0 |

Screen candidates per episode: 00:4 01:23 02:7 03:15 04:15 05:8 06:7 07:7 08:7 09:10 10:7 11:7 12:7 13:7 14:7 15:3 16:8
17:7 18:7 19:12 20:8 21:7 22:4 23:7 24:7 25:8 26:12 27:7 28:8 29:7. Full list: `inputs_r6/pool_r6.csv`.

### 2.3 Not candidates (by rule; recorded in `runs_r6.csv`)

- **Zuxing's single-checkpoint copies** (`2026-10-24-5090-best/*` bundles from the 5090 and robolab machines,
  `2026-10-28-tier2-retest/*` incl. `tier2_best_ckpts/`): their embedded output_dir is another machine or another
  folder (R1); they are single-checkpoint selections, not training runs. The ones that already won an earlier round
  (ep00, ep16, ep19, ep20, ep22, ep25) stay as incumbents (section 4).
- Our T2fix `forks/` and `BASE/` copies, all re-test / evaluation copies (R1); the K3 stretch ep04 run (L = 9,000) and
  Zuxing's two `2026-10-28-tier2-runin-a5` runs on pro6000 (L = 1,000 and 500; trained on to the 5090) (R3);
  every run not trained on a Tier-2-derived input, including Zuxing's 2026-10-14 / 2026-10-21 bat and cane runs (R2);
  arm A (no checkpoint).
- Anything written after T_c - 300 s.

## 3. Evaluations

### 3.1 Protocol: round 5's (PREREG_R5 3.1), unchanged except the job lists and K3CA

- **Driver.** `scripts/t2cons_r6_eval.py` (new file; round 5's driver `t2cons_r5_eval.py` 68191fbb with the round-6
  bookkeeping; diff in `R6_SCRIPT_DIFFS.txt`), which loads `scripts/t2cons_eval.py` (2cf42786) and through it
  `scripts/t2best5_driver.py` (24b9af39) from their verified bytes. Launcher `scripts/t2cons_r6_eval_launch.sh`. Both are
  pinned in `PREREG_R6_PINS.md` before they export or evaluate anything.
- **Export.** Once per screen job, CPU only, `python -m flash_chord.training.ppo.coordinate.export` of the training tree:
  `code/fc_r2` @ 61fa5d3 for ours (every one of our Tier-2 runs was trained in that tree), `code/nv-v2d_runin` @
  b39b69b for Zuxing's (as rounds 1-5). Tolerance 1e-4, one fallback export at 2e-4 on a tolerance refusal; stock
  verified, `curriculum_stage_voc_scale` 0, `source_sha256` == the pool file. A non-tolerance refusal ends the job
  (`export_refused` / `export_incompatible`); an `AMENDMENTS_R6.md` written before any round-6 score may add one export
  with the exporter of his pushed training branch (as PREREG_R5 3.1).
- **Incumbents are never re-exported**: every phase-F run of an incumbent evaluates its existing exported file (the
  earlier round's `exports/<job>/export.json` and file, re-hashed at every start; `incumbents_r6.json`), so a kept
  episode keeps the digest of the R5-CLOSEST row (§2.9.d).
- **Evaluation.** `scripts/track2/eval_all.py` of the job's tree (657295921e82..., identical in both trees) with `--export
  no --world-count 1`, which runs the kit's `examples/run_policy_evaluation.py` and `flash_chord/scripts/evaluate_policy.py`
  (160d24f5) on the untouched stock runtime `code/video_to_data_8c778f9`. Frame-0 start, explicit reset, no help.
  `HUMAN_MOTION_DATA_DIR` = the job's data root; the reference = the job's own training parquet.
- **K3CA input.** The four ep26 K3CA candidates are the first K3CA jobs of these drivers. The driver adds `k3ca` ->
  `data/human_motion_data_tier2_k3ca` (manifest 12fe4494, identical to the other roots) and frame references
  `tmp_analysis/tier2/k3ca/frame_refs_k3ca` (round 1's `INPUT_REF["k3ca"]` and refs/B track "C") to the pinned driver's
  tables at load time (rebinding only, as round 5 rebinds `ALLOWED_GPUS`).
- **ok checks, retries, integrity**: round 5's, unchanged (exit 0 and `evaluated`; results JSON + parquet; checkpoint
  digest == the exported file; reference == the training parquet by path and sha256; start frame 0, world count 1,
  explicit reset, finite; eval-script / eval-code digests; `environment_valid`; exported file unchanged; up to 3 tries
  >= 5 min apart; a completed evaluation is never re-run; kit + stock tree listing vs round 1's baseline, a change halts
  new starts).

### 3.2 Phase S (screening): 3 runs per screen candidate

Jobs: the 250 `screen` rows of `pool_r6.json`. Each gets one export (`exports/<job>/`) and **R_S = 3** official runs
(`runs_s/<job>/run<k>/`). Order (decides nothing): exports in (episode, job) order, then run-major: run 1 of every job,
then run 2, then run 3. 250 exports + 750 evaluations.

### 3.3 Phase F (final): 10 trials of the selected set

Jobs: the 30 incumbents (section 4) plus the phase-S finalists of `final_r6.json` (section 5.4; at most one per
episode; written and locked by `SELECTION_S6.sha256` before phase F starts). Each job gets **10 fresh official runs**
(`runs_f/<job>/run<t>/`, t = 1..10); phase-S runs never enter phase F. Order: trial-major (trial t of every job before
trial t+1), within a trial (episode, job). A finalist evaluates its single phase-S export. 300 + 10 x n_finalists
evaluations.

### 3.4 Scheduling (decides nothing)

- **GPUs 0, 1, 2 and 7 only** (each hosts priority Tier-3 training): at most ONE evaluation of ours per GPU; never GPUs
  3 and 4 (XHand runs under private MPS daemons), never 5 or 6; never a GPU with an MPS server / client / control
  daemon; at least 4096 + 1536 MiB free.
- **Caps.** At most 2 per NUMA eval-core lane (48-51,152-155 for GPUs 0-2; 100-103,204-207 for GPU 7) and 4 in total
  (so at most 3 at once); live evaluations of the t2_best5 / t3final / t1best20 / t1confirm drivers count; 35 s cooldown
  per GPU after each of our evaluations.
- **Lane locks**: round 5's rule (our `ROOT/evallock/gpu<g>.lock` + every existing shared eval-lane lock file of that
  GPU, once per file, all or nothing, never blocking, opened read-only, never created); the shared list gains round 5's
  own `t2_consensus_r5_20261029/evallock`.
- **Processes**: `nice -n 10` + `taskset` on the eval-lane cores. Nothing is ever signalled; an evaluation past 2400 s
  is only flagged OVERDUE. `ROOT/STOP` stops new starts. Running trainings (Tier-3 retries, XHand Genesis runs,
  Zuxing's runs, the ablation setup) and their files are never touched.

### 3.5 Seeds (records only)

As round 5: the official path has no selectable seed (the harness forces `env.reset.seed = 42`; every other seed is fixed
in the evaluated file's own config), so repeated runs differ only by GPU non-determinism. Recorded per run in
`seeds.json` and `seeds_s6.tsv` / `seeds_f6.tsv`.

### 3.6 Dry runs (this step; nothing else is launched)

- **Plan-only dry run** of the driver for phase S and phase F (pins, pool and incumbent checks, hulls, GPUs; nothing
  exported or evaluated).
- **Two official evaluations on GPU 0** (each policy against its own input), written to `dryrun/`:
  1. phase-S path: **`ep26_k3ca_25298`** (the K3CA final; first K3CA job of these drivers, fc_r2 export). Its export
     goes to `exports/ep26_k3ca_25298/` and stays the job's single export.
  2. phase-F path: **`ep26_zx_k3fix_92000`** (the ep26 incumbent: Zuxing's pro6000 file, its round-1 b39b69b export
     reused, no new export).
  A dry-run evaluation is never a round-6 run or try, counts toward no limit, and sections 5-6 never read `dryrun/`.
  The two jobs were fixed by plumbing coverage (new K3CA input; incumbent-export reuse) before any value existed.
- Result: `DRYRUN_R6.md` (+ sha256). Phase S is NOT launched in this step; it is launched by the main session with
  `bash scripts/t2cons_r6_eval_launch.sh --phase screen` after the user's go.

## 4. Incumbents: the frozen round-5 composition

For each episode the incumbent P_e is the round-5 pick (`selection_r5.json` compositions ROUND5), with its existing
export and its frozen variant-B consensus AUCs from the round that selected it (round 1: 5 runs in
`scores/consensus_scores.csv` 50cc6e94; round 3: 10 runs in `scores_r3/consensus_scores_r3.csv` 2faa6e3b; round 5: 10
runs in `scores_r5/consensus_scores_r5.csv` a17eac41). m_P_frozen = their float64 mean (== the frozen job stats, checked
by the resolver).

| ep | incumbent P_e | selected in | input | tree | export sha256 | frozen runs | m_P_frozen | + 0.01 |
|---|---|---|---|---|---|---:|---:|---:|
| 00 | `ep00_zxb_k3fix_60000` | round3 | k3fix | b39b69b | 45d1084dea7f6d0b | 10 | 0.709383 | 0.719383 |
| 01 | `ep01_k3fix_final44922` | round1 | k3fix | fc_r2 | a15d9e85b0458c8d | 5 | 0.171319 | 0.181319 |
| 02 | `ep02_k3fix_final34093` | round1 | k3fix | fc_r2 | 3f1ae6373cfb8877 | 5 | 0.868138 | 0.878138 |
| 03 | `ep03_k3fix_final27068` | round1 | k3fix | fc_r2 | e042571ca8a01236 | 5 | 0.921527 | 0.931527 |
| 04 | `ep04_raw_final44922` | round1 | raw | fc_r2 | 1de36ebdd4f1447e | 5 | 0.120283 | 0.130283 |
| 05 | `ep05_zx_k3fix_88000` | round1 | k3fix | b39b69b | 248dd05779f67e58 | 5 | 0.578153 | 0.588153 |
| 06 | `ep06_k3fix_final28420` | round1 | k3fix | fc_r2 | 84bd485806402cb2 | 5 | 0.797276 | 0.807276 |
| 07 | `ep07_k3fix_final18529` | round1 | k3fix | fc_r2 | b02cb8f6597f3b54 | 5 | 0.920452 | 0.930452 |
| 08 | `ep08_k3easy_final17496` | round1 | k3 | fc_r2 | 17d421a1938181f0 | 5 | 0.596363 | 0.606363 |
| 09 | `ep09_raw_final39328` | round1 | raw | fc_r2 | c73692c5b01ed77e | 5 | 0.760460 | 0.770460 |
| 10 | `ep10_raw_final34267` | round1 | raw | fc_r2 | 2490e879451fe43d | 5 | 0.412469 | 0.422469 |
| 11 | `ep11_k3fix_final29175` | round1 | k3fix | fc_r2 | 3b834654d0b7a1b2 | 5 | 0.783208 | 0.793208 |
| 12 | `ep12_k3fix_final23178` | round1 | k3fix | fc_r2 | b455bb4dc1707de7 | 5 | 0.543911 | 0.553911 |
| 13 | `ep13_k3fix_final39254` | round1 | k3fix | fc_r2 | bc146e3ea4a98084 | 5 | 0.665483 | 0.675483 |
| 14 | `ep14_k3fix_final44922` | round1 | k3fix | fc_r2 | 3c4740ef7bd4f60c | 5 | 0.330614 | 0.340614 |
| 15 | `ep15_raw_final44922` | round1 | raw | fc_r2 | d4bbb317cb71ebe6 | 5 | 0.729895 | 0.739895 |
| 16 | `ep16_zxb_g2_34000` | round5 | g2 | b39b69b | acec8f29690a40c3 | 10 | 0.937434 | 0.947434 |
| 17 | `ep17_raw_final31807` | round1 | raw | fc_r2 | 2188283c46ef28c8 | 5 | 0.647528 | 0.657528 |
| 18 | `ep18_k3fix_final18911` | round1 | k3fix | fc_r2 | de312489bafbc36a | 5 | 0.419129 | 0.429129 |
| 19 | `ep19_zxb_g2hard_248000` | round5 | g2 | b39b69b | 960405ce1f92d6be | 10 | 0.565752 | 0.575752 |
| 20 | `ep20_zxb_k3fix_30000` | round5 | k3fix | b39b69b | 3b3dbf3cf5522a61 | 10 | 0.851695 | 0.861695 |
| 21 | `ep21_k3fix_final44922` | round1 | k3fix | fc_r2 | 0e6399890edfc6f6 | 5 | 0.444469 | 0.454469 |
| 22 | `ep22_zxb_k3fix_78000` | round3 | k3fix | b39b69b | 66da2cd65061af36 | 10 | 0.756067 | 0.766067 |
| 23 | `ep23_raw_final20023` | round1 | raw | fc_r2 | 961b538e4679015e | 5 | 0.902867 | 0.912867 |
| 24 | `ep24_raw_final44922` | round1 | raw | fc_r2 | 7bffc41746d13913 | 5 | 0.895137 | 0.905137 |
| 25 | `ep25_zxb_k3fix_48000` | round5 | k3fix | b39b69b | 2d92f29dc238e784 | 10 | 0.748108 | 0.758108 |
| 26 | `ep26_zx_k3fix_92000` | round1 | k3fix | b39b69b | 25c600cb20fbd448 | 5 | 0.731484 | 0.741484 |
| 27 | `ep27_k3fix_final20863` | round1 | k3fix | fc_r2 | e560ebffd9545b11 | 5 | 0.897854 | 0.907854 |
| 28 | `ep28_zx_g2_102000` | round3 | g2 | b39b69b | 9ce8e92c1ed16618 | 10 | 0.154162 | 0.164162 |
| 29 | `ep29_k3fix_final17358` | round1 | k3fix | fc_r2 | 8c984002495d5a75 | 5 | 0.505675 | 0.515675 |

(These are consensus values, not leaderboard estimates; their 30-episode mean is round 5's ROUND5 0.6455.)

## 5. Scores and selection

### 5.1 Why these noisy-side scores (and what guards against tracking-noise overfit)

A policy is trained to track its own noisy input; how well it does that is visible from the Tier-2 data. The failures
that separate checkpoints (no grasp, drop, wrong placement, a collapse at frame 0) produce errors of many centimetres
against ANY reasonable reference. So noisy-side tracking ranks checkpoints by task success and gross tracking quality.
Its known failure mode is **tracking-noise overfit**: a policy that reproduces its own input's noise scores well against
that input. Round 6 keeps every round-5 safeguard and adds two:
1. **Consensus reference (R5).** Candidates are ranked by their AUC against round 1's frozen consensus refs/B: a robust
   per-frame combination of up to four independently filtered noisy tracks (raw, K3fix, K3CA, G2). Reproducing one
   input's particular noise is not rewarded, and candidates trained on different inputs are comparable. Own-input AUC
   is never compared across inputs; it is used only in the validation gate and the floor. The refs are never rebuilt.
2. **Validation gate (R5, round-1 A1.3).** Every scored run is re-scored against its OWN input with the pinned round-1
   functions and must reproduce the evaluator (AUC within [min_s AUC_s - 0.02, max_s AUC_s + 0.02], s = 0..9 vertex
   seeds; |dMPPE| <= 0.5 cm; SP-SR and MP-SR equal); a candidate with a flagged run is out.
3. **Stability floor (R5).** A winner needs own-input SP-SR = MP-SR = 1 in >= 8 of its 10 decision runs (success
   thresholds are robust to the small noise band).
4. **Margin (R5).** +0.01 consensus AUC over the incumbent.
5. **Closest-to-mean, never best-of (R5).**
6. **New: screen / confirm split.** The pick among many screened candidates is biased upward (winner's curse); the
   decision therefore uses only FRESH runs that played no part in choosing the finalist.
7. **New: paired comparison.** Finalist and incumbent are evaluated in the same 10 trials (same time windows, same
   GPUs on average); the bar is the higher of the incumbent's fresh mean and its frozen mean.

### 5.2 Consensus scores (round 1's scorer, unchanged)

Every scored run (phase S and F) gets variant-B consensus scores with the pinned round-1 functions of
`scripts/t2cons_consensus.py` (e1b943b1): `rollout_on_grid` (the packer's ordering, provenance checks and
`convert_track2_frame` against the job's own input reference `INPUT_REF[input]`; tolerance 1e-3 m / 1e-3 rad) and
`metrics_vs` (`episode_metric`, `align_initial_pose=False`, 500 released-mesh vertices, seed 0, float64) against round 1's
frozen `refs/B/episode_XXXXXX.npz` (refs/build.json 9646de2d). The rollout files of each phase are frozen in
`inputs_r6/ROLLOUTS_S6.sha256` / `ROLLOUTS_F6.sha256` before the first score of that phase.

### 5.3 Validation gate (round-1 AMENDMENTS A1.3, unchanged)

As 5.1 item 2, with `t2cons_a1_gate.py` (a29c997b) `auc_seed`. A phase-S candidate with a flagged run is
`validation_failed` (no finalist); a finalist with a flagged phase-F run is not qualifying; a flagged incumbent run is
reported, left out of m_P_fresh, and its trial cannot be R6-CLOSEST. A kit or stock-runtime change fails the gate.

### 5.4 Phase-S selection: at most one finalist per episode

For each episode e:
1. **S-eligible** candidate c: ok export, 3 ok official runs, none flagged by 5.3, and **floor_S(c) >= 2**, where
   floor_S = the number of its 3 runs with own-input SP-SR = 1 AND MP-SR = 1 in the same run.
2. cons_S(c) = float64 mean of its 3 variant-B consensus AUCs.
3. **Finalist f_e** = the S-eligible c with the highest cons_S(c) among those with **cons_S(c) >= m_P_frozen(e) + 0.01**
   (exact floats). Ties: lower population variance of its 3 consensus AUCs, then lower mean consensus MPPE, then job name.
4. No such c: no finalist; the incumbent is the only job of e in phase F.
Outputs: `scores_s6/` (gate, consensus scores, job stats), `selection_s6.{json,txt,md}`, `final_r6.json` (the phase-F job
list: 30 incumbents + the finalists with their phase-S export records), and **`SELECTION_S6.sha256` last**.

### 5.5 Phase-F decision per episode (the R5 rule on fresh paired runs)

For an episode with finalist f and incumbent P:
- m_F(f) = float64 mean of f's 10 phase-F consensus AUCs; m_P_fresh = float64 mean of P's phase-F consensus AUCs over
  its ok, non-flagged runs (>= 8 required, else m_P_fresh = +inf: no change).
- **f replaces P iff** (a) f has 10 ok, non-flagged phase-F runs; (b) own-input SP-SR = MP-SR = 1 in >= 8 of them;
  (c) **m_F(f) >= max(m_P_frozen, m_P_fresh) + 0.01** (exact floats). Otherwise P stays (ties go to the round-5 choice).
- Nothing else enters: no own-input AUC comparison across inputs, no phase-S value, no manual override.

### 5.6 The two sets, R6-CLOSEST, and the replace rule (fixed now)

- **S5** = the 30 incumbents; **S6** = per episode the 5.5 result. Both are evaluated in the same 10 trials (an
  unchanged episode's run t serves both).
- Trial t is complete for a set if every episode's job has an ok, non-flagged run t. Per complete trial: the 30-episode
  means of own-input AUC, SP-SR, MP-SR, MPPE (evaluator values) and of consensus-B AUC, SP-SR, MP-SR, MPPE.
- **R6-CLOSEST** = the complete trial t of S6 whose 30-episode mean OWN-INPUT AUC is closest to the mean of that quantity
  over the complete trials (S6 is one fixed checkpoint set, so its own-input values are comparable across trials);
  ties: smaller distance of the consensus-B mean to its trial mean, then lower t. Never a best-of.
- **Replace rule.** R6-CLOSEST replaces R5-CLOSEST as the upload-eligible Tier-2 file **iff** (i) at least one episode
  changed in 5.5, AND (ii) **Delta_B >= +0.001**, where Delta_B = mean over the trials complete for both sets of [S6
  30-episode mean consensus-B AUC - S5 30-episode mean consensus-B AUC] (paired; +0.001 on the 30-episode mean = +0.03
  summed over episodes), AND (iii) S6's mean own-input SP-SR and MP-SR over those trials are each >= S5's. Otherwise
  R5-CLOSEST (e8a03a4d...) stays the upload-eligible file. The consensus-B composite is used for (ii) because own-input
  composites of two sets that differ in input on the changed episodes use different references and are not comparable;
  own-input composites are reported next to it.
- Outputs: `scores_f6/`, `selection_r6.{json,txt,md}`, `summary_r6.{json,md}` (per-trial composites of S5 and S6,
  Delta_B, the R6-CLOSEST trial, the replace verdict) and **`SELECTION_R6.sha256` last**; from that moment round 6 is
  final.

### 5.7 Scorer code

`scripts/t2cons_r6_consensus.py`, a new file: round 5's scorer machinery (`t2cons_r5_consensus.py` e79a2450) reduced to
the round-6 bookkeeping. It loads the pinned round-1 modules `t2cons_consensus.py` (e1b943b1) and `t2cons_a1_gate.py`
(a29c997b) from their verified bytes and calls `rollout_on_grid`, `metrics_vs`, `load_refs`, `auc_seed`, `stats`,
`floor_count` unchanged; a path guard (round 1's `FORBIDDEN`, extended by this script's `EXTRA_FORBIDDEN_R6`) refuses
the listed paths before opening them. Commands:
`selftest` (synthetic checks of 5.4 / 5.5 / 5.6, incl. ties, floors, flagged runs, incomplete trials), `check`, `gate`,
`score`, `select` per phase. It is pinned with its selftest in **`PREREG_R6_PINS_b.md` before it scores anything**.

## 6. The file (R6-CLOSEST) and its packing

R6-CLOSEST = trial t* of S6: per episode the S6 job's run t* rollout. Packed with the kit packer unchanged, each episode
against its own input frame reference (raw: the packer default; K3 / K3fix / K3CA / G2: byte copies of the frame
references in `TRACKS_R6.sha256`), with the placeholder commit URL, into a local pack folder, and checked
with the kit's offline validator. The pack script (`scripts/t2cons_r6_pack.py`, round 5's with round-6 bookkeeping) is
pinned in `PREREG_R6_PINS_c.md` before its first use. Before any upload the chosen file (R6-CLOSEST if 5.6 says so,
otherwise R5-CLOSEST) is re-packed with the real commit URL, its pose columns byte-identical to the placeholder file,
and validated again. The user decides the upload.

## 7. (not included in this copy)

## 8. Budget and estimate

- Evaluations: phase S 250 x 3 = 750 (+ 250 CPU exports); phase F (30 + n_f) x 10 = 300..600 (n_f <= 30 finalists). Total
  1,050..1,350 evaluations.
- Capacity: GPUs 0, 1, 2, 7, one evaluation per GPU, at most 2 at once on the NUMA0 lane (GPUs 0-2) and 1 on GPU 7, so 3
  at once; 35 s cooldown per GPU (absorbed on NUMA0 by rotating over three GPUs; not on GPU 7).
- At 70 s per evaluation (the main session's figure under load): ~2 x 3600/70 + 3600/105 = ~137 per hour -> 7.7-9.9 h of
  evaluation. At round 5's measured 30-41 s: ~230 per hour -> 4.6-5.9 h. Exports (~30 s each, at most one per lane)
  overlap with phase-S evaluations. Scoring steps are CPU minutes. Expected wall time 6-10.5 h, inside the ~12 h target.
- If phase S runs long, `ROOT/STOP` pauses it (running evaluations finish); nothing is skipped or shortened without an
  amendment written before the affected scores.

## 9. Disclosures

1. **Incumbents carry earlier nominations.** The round-5 set includes single checkpoints first nominated by Zuxing
   (rounds 3 and 5: ep00, ep16, ep19, ep20, ep22, ep25; round 3: ep28) and then selected under those rounds' rules.
   Round 6 adds no nominated candidate and judges every incumbent against fresh paired runs.
2. **Basis of the pool rule.** The rule (K, second half, 10k minimum) is generic and was sized by the budget. In
   rounds 3-5 some winning files were intermediate checkpoints (e.g. 60k of 90k); the second-half rule reflects that.
3. **Zuxing's own evaluation files.** His pro6000 runs have evaluation watchers that wrote per-checkpoint files
   (`eval_*/metrics.csv`); the resolver never entered `eval*` folders and selects his checkpoints only by iteration.
4. **Reference bias (as PREREG_R5 8.6).** refs/B use tracks R,F everywhere, C (K3CA) on 0, 5, 15, 19, 22, 26, 28 and G
   (G2) on 1, 4, 9, 19, 21, 28. A candidate whose own input is among an episode's reference tracks is favoured there
   (raw and K3fix always; K3CA on ep26; G2 on ep01/04/19/28); G2 candidates elsewhere, and K3 candidates, are not.
5. **Noise and selection.** Phase S picks the maximum of up to 23 noisy 3-run means per episode; the phase-F decision
   uses fresh runs only and a +0.01 margin over the higher of two incumbent baselines. Runs are single-seed repeats.
6. **Exporters.** fc_r2 for ours, b39b69b for Zuxing's (his hard-G2 runs were trained at 82837c8; rounds 2-5 exported
   them with b39b69b without refusal).
7. **Training code (section 9.4)** for the submission commit: ours = `code/fc_r2` @ 61fa5d3 (contained in the pushed
   `dev.haoran.c3r2` 41403d3) plus the K3 / K3fix / K3CA input builders; Zuxing's pro6000 runs = `dev.zuxing.speedup`
   bc3b1bd and `dev.zuxing.hard_start` 82837c8 (both pushed heads), his G2 builder `tmp/t2/g2data/` (in no pushed commit),
   and for the remote-trained incumbents (ep00, ep16, ep19, ep20, ep22, ep25) commits unknown to us.
8. **Where the text was made.** The new scripts and this file were written in the agent's Mac scratchpad and uploaded
   with scp; nothing of the project ran on the Mac except a syntax parse of the new scripts.

### 9.4 Zuxing's training code (from `inputs_r6/ZUXING_CODE_R6.tsv`)

| file(s) | trained on | code | pushed? |
|---|---|---|---|
| pool: g2 ep01 / ep04 / ep19 / ep28, k3fix ep04, k3fix ep05 / ep26 (28 files) | pro6000 | worktree `dev.zuxing.speedup`, commit bc3b1bd (his `train_t2:` log line) | yes, head of `dev.zuxing.speedup` |
| pool: hard-g2 ep01 / ep19 (8 files) | pro6000 | worktree `dev.zuxing.hard_start`, commit 82837c8 | yes, head of `dev.zuxing.hard_start` |
| incumbents ep05, ep26 (k3fix-ep05-ep26 run), ep28 (g2 run) | pro6000 | bc3b1bd | yes (`dev.zuxing.speedup`) |
| incumbents ep00, ep22 (5090 runin-k3fix), ep16, ep25 (5090 runin-a5), ep19 (5090 hard-g2 cont2), ep20 (5090 fill) | remote | unknown | ask Zuxing |
| G2 input (all G2 files) | - | `zuxing.lu/tmp/t2/g2data/build_g2_data.py` + build records | only in his folder |
Whether his worktrees had uncommitted edits when the runs started cannot be checked without a git command in his folder
(not allowed); TELL_ZUXING_r6.txt asks him.

## 10. Implementation, hard rules, amendments

| step | what | code |
|---|---|---|
| P | pool + incumbents + copies (done, T_c) | `scripts/t2cons_r6_pool.py` |
| R1 | this file + `PREREG_R6.sha256` | |
| R2 | driver + launcher (new files), plan-only dry runs, `PREREG_R6_PINS.md`, the two dry-run evaluations, `DRYRUN_R6.md` | `scripts/t2cons_r6_eval.py`, `scripts/t2cons_r6_eval_launch.sh` |
| R3 | phase S: 250 exports + 750 evaluations (after the go) | same, `--phase screen` |
| R4 | scorer pinned (`PREREG_R6_PINS_b.md`), selftest, gate + scores + selection of phase S, `SELECTION_S6.sha256` | `scripts/t2cons_r6_consensus.py` |
| R5 | phase F: 10 trials of 30 incumbents + finalists | driver `--phase final` |
| R6 | gate + scores + decision + sets + R6-CLOSEST + replace verdict, `SELECTION_R6.sha256` | scorer |
| R7 | pack R6-CLOSEST (pinned in `PREREG_R6_PINS_c.md` first); offline validator | `scripts/t2cons_r6_pack.py` |

Hard rules: new files only, under this folder and `scripts/t2cons_r6_*`; rounds 1-5, t2_best5, the kit, the stock
runtime and the code trees are only read; Zuxing's folder only read; never edit a file a running process uses; never
signal any process; GPUs 0, 1, 2, 7 only, no MPS; flashchord Python `-I -B` for the driver, kit Python `-I -B` for the
scorer and packer. Any deviation goes into `AMENDMENTS_R6.md` (+ sha256) with time and reason before the affected step;
an amendment written after any round-6 score was visible must say so. PINS addenda may only add pins and notes.

## 11. Pins (sha256)

Path rules: a path starting with `/` is absolute; `scripts/...` and `tmp_analysis/...` are relative to
`/root/autodl-tmp/haoran.zhang`; any other path is relative to this folder.

pin scripts/t2cons_r6_pool.py 13974f6f48abd5ac8cead9aec5f7e99452de8ed9c1d457d8a5433391b8fd7905
pin inputs_r6/pool_r6.json c2aeb79af68cda31d8daa9f324bc26189c721150ec781f62a775782403082de6
pin inputs_r6/pool_r6.csv f25f7de5d3653f26ecd6523f8dd4b444dd178bb0d42d7d955049c2794fb33e77
pin inputs_r6/runs_r6.json c7e529e2b7f16fef73e1a42cf8db9e589dc68875f4a0f4aba02d06d2089032d5
pin inputs_r6/runs_r6.csv 2db6a74e98357f26a36c57d34f5db251e8b9d4d13ee3ccae40f5821fb3fbe7c6
pin inputs_r6/incumbents_r6.json 44d62d8cbd5b2d4247307b301902fb47dc42032c4227dcbefb970056d9d07b0b
pin inputs_r6/TRACKS_R6.sha256 cee1eb48d63713535f5e2779615c966a1eb27df8f2bdf2e7515dd8596ad942e3
pin inputs_r6/ZUXING_CODE_R6.tsv 28fdba4978949f63d7f81ac2d66108f9283dc4c67a3efb535b827a412e566f94
pin inputs_r6/LS_REMOTE_R6.txt 15c250d0cdd7873bf5b4ece5c40acb3914a279d84d0cc6560e8bbc96a5fd5fb9
pin inputs_r6/resolve_r6.log f50be67baeac99e6a7b7125c7c32c2d00de5073ba0155f871e32fa523b5b8ffb
pin inputs_r6/SNAPSHOT_R6.sha256 b7c57c5b86b61f3abc4dfc54b8d72349dc7d6420521d493aa8a781c22dd0054a
pin COPIES_r6.tsv fa17929cf145cbb8333464b559521f699a2712319bd6b9833e017dea92d000f9
pin TELL_ZUXING_r6.txt 4d5213425265a922e9543b85323901c192043863465f28f89ee7c74ddb6cd6c6
pin scripts/t2best5_driver.py 24b9af39a2e2708641e7870447b12b6475c23aec9cd18ba71086768dd69cfc89
pin scripts/t2cons_eval.py 2cf427866a9432a3bd1b8fbb7864149d4258a5d19100c20da92d32bfe4fbfa9b
pin scripts/t2cons_consensus.py e1b943b10fd1db13ab6f78e5bde7b3c6e6c48f7daeee6230c32c463d14ac41b4
pin scripts/t2cons_a1_gate.py a29c997b7271e9299db109e97af5210f528484719210bb58685523b0ae0eb16b
pin scripts/t2cons_r5_eval.py 68191fbbd6e6a2c2abe12c8b92665d32ca77f9cd7e869a516958a7c100cacbe4
pin scripts/t2cons_r5_eval_launch.sh 5a4dc89f9415b868d97aea0f40b924afdac8bcca49b8f45fcaee42fb7d65ed8d
pin scripts/t2cons_r5_consensus.py e79a2450ed99a0f44b27353fbd3d0e3fd6d249393494589bfb624923bc5efb23
pin scripts/t2cons_r5_pack.py 62f88e6ffdeb46d23d63b9e2774bb4c4711e0e98623b504ed87facdbf513ae38
pin tmp_analysis/t2fix/cache_keys_t2.py 9f64214eba0eb6602e8ccb9cc200fe9c6ff94676e0c99551f095b3b1db5c1aae
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r5_20261029/SELECTION_R5.sha256 58e8b54b96ee594e587f76793aaf592c5a23dc0413d8a15546052e996af29bf0
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r5_20261029/selection_r5.json 89ca15098c03c465be1a5f78b7d4342eac373b15ddfa58f823a6fc416a464e5f
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r5_20261029/PREREG_R5.md 870aef4b1b9643fe68262954addd7b387e334690426fb3d70a799b072fdd2d81
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r5_20261029/summary_r5_consensus.json 67dc0a796f01ae6845f49316321ea705df35a87e5648f115b892fca329429bf1
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
