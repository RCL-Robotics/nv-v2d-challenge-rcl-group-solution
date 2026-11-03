# Track 2 Tier 2: submission record (R8-CLOSEST, trial 7)

This folder is the record of the Tier-2 file `T2_R8_closest_trial07_<this commit, 7 hex>.csv`, packed with this
commit's URL as `code_commit_url`. The CSV is not in the repository; it is kept on the team server. Round 8 is the
selection round after round 7 (`../t2r7_20261102/`): it kept 29 of round 7's 30 checkpoints and replaced episode 13.

Selection uses only own-input and consensus-B scores: the official kit evaluator against each policy's own (raw or
filtered) Tier-2 input, and round 1's frozen consensus reference built from Tier-2 tracks (`refs/B`).

## Round 8 in short

`selection/round8_20261103/PREREG_R8.md` (hash-locked 2026-11-02 21:43 CST, before the snapshot and before any
round-8 export, evaluation or score) is round 7's rule unchanged, applied to what is new since round 7:
- **D1 snapshot** T_c = 2026-11-03 12:00:00 CST: round 7's universe re-scanned plus the copy folders that arrived
  since; round 7's mechanical picks (pool rule R4, K = 3) only for runs that are new or trained further.
- **D2 watcher best**: for each of Zuxing's runs with an evaluation-watcher file, its own-input best checkpoint (argmax
  of the watcher's own-input `add_auc` over the 10,000-iteration grid and the last evaluated iteration) is one more
  candidate.
- **D3 incumbents**: each episode's incumbent is the round-7 pick, with its round-7 phase-F runs (never re-run).
- **D4**: only episodes with a round-8 candidate are evaluated. **D5**: no score other than own-input and consensus-B.
- Consequences C1-C6 (PREREG_R8 section 0): pick targets on 5090-hosted runs only on the 10,000 grid and the last
  iteration; copy-run checks; round-7 phase-F participants barred; one finalist per episode; GPU and deadline rules.
- **Amendment A1** (`AMENDMENTS_R8.md`, 2026-11-03 13:40-13:50 CST, before any round-8 evaluation): evaluations may use
  GPUs 0, 4, 6 and 7 instead of 6 and 7, under the same rule (plain, no MPS, no other compute process listed).

Counts (`POOL_R8.json`, `inputs_r8/runs_r8.json`, `selection_s8.md`, `selection_r8.md`):
- 171 run folders resolved at T_c (128 eligible; 10 new and 6 trained further since round 7's snapshot). Pool: 89 new
  screen candidates from 31 runs in 23 episodes (64 mechanical R4 targets and 31 watcher-best designations; a
  checkpoint can be both), 5 `reuse7` candidates (round-7 phase-S results reused), 3 `barred7` rows and 1 `incumbent`
  row. All 89 screen candidates come from Zuxing's runs: 30 on the team server, 27 on his RTX 5090, 14 on the robolab
  host ws031, 14 on wsrbl015 and 4 on ws032.
- Phase S: 267 official runs (89 x 3), all ok; 24 episodes evaluated. One finalist: `ep13_w312runinws_90000`
  (3-run consensus-B mean 0.7164, threshold m_P_frozen + 0.01 = 0.6755).
- Phase F: its 10 fresh runs all ok, own-input SP-SR = MP-SR = 1 in 10 of 10, m_F = 0.7002 >= bar 0.6760 =
  max(m_P_frozen 0.6655, m_P_fresh 0.6660) + 0.01. Episode 13 is replaced; the other 29 episodes keep their round-7
  pick.
- **R8-CLOSEST = trial 7**: the complete trial whose 30-episode mean own-input AUC is closest to the mean over the 10
  trials (never a best-of). Kept round-7 picks use their round-7 phase-F run 7; episode 13 uses its round-8 phase-F
  run 7.

Trials of S8 (own input = each policy against its own Tier-2 input; consensus-B = against `refs/B`; not leaderboard
numbers; `summary_r8_trials.csv`):

| trial | own-input AUC | SP-SR | MP-SR | MPPE (cm) | consensus-B AUC |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.7981 | 0.933 | 0.933 | 2.898 | 0.6687 |
| 2 | 0.7969 | 0.933 | 0.933 | 3.563 | 0.6674 |
| 3 | 0.7973 | 0.933 | 0.933 | 3.394 | 0.6683 |
| 4 | 0.7920 | 0.933 | 0.900 | 3.052 | 0.6628 |
| 5 | 0.7837 | 0.867 | 0.867 | 4.333 | 0.6580 |
| 6 | 0.7977 | 0.933 | 0.933 | 2.980 | 0.6679 |
| **7** | **0.7962** | **0.933** | **0.933** | **2.877** | **0.6673** |
| 8 | 0.7986 | 0.933 | 0.933 | 2.708 | 0.6679 |
| 9 | 0.7981 | 0.933 | 0.933 | 2.954 | 0.6673 |
| 10 | 0.7970 | 0.933 | 0.933 | 3.906 | 0.6669 |
| mean | 0.7955 | 0.927 | 0.923 | 3.266 | 0.6663 |

## The 30 checkpoints

| ep | job | trained by | machine | input | run | iteration | training commit | round 8 | rollout |
|---|---|---|---|---|---|---:|---|---|---|
| 00 | ep00_z5k3fix_90000 | Zuxing | RTX 5090 | K3fix | 2026-10-24-tier2-runin-k3fix/ep00 | 90000 | 82837c8 | kept (round 7: STRICT, confirmed) | 7 F |
| 01 | ep01_k3fix_final44922 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep01_k3fix_s42 | 44922 | 61fa5d3 | kept (round 7: kept) | 7 F |
| 02 | ep02_k3fix_final34093 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep02_k3fix_s42 | 34093 | 61fa5d3 | kept (round 7: kept) | 7 F |
| 03 | ep03_k3fix_final27068 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep03_k3fix_s42 | 27068 | 61fa5d3 | kept (round 7: kept) | 7 F |
| 04 | ep04_zxg2_450000 | Zuxing | pro6000 | G2 | 2026-10-24-tier2-runin-g2/ep04 | 450000 | bc3b1bd | kept (round 7: replaced) | 7 F |
| 05 | ep05_zx_k3fix_88000 | Zuxing | pro6000 | K3fix | 2026-10-24-tier2-runin-k3fix-ep05-ep26/ep05 | 88000 | bc3b1bd | kept (round 7: kept) | 7 F |
| 06 | ep06_k3fix_final28420 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep06_k3fix_s42 | 28420 | 61fa5d3 | kept (round 7: kept) | 7 F |
| 07 | ep07_k3fix_final18529 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep07_k3fix_s42 | 18529 | 61fa5d3 | kept (round 7: kept) | 7 F |
| 08 | ep08_k3easy_final17496 | Haoran | pro6000 | K3 | C3r2_tier2_k3easy/ep08 | 17496 | 61fa5d3 | kept (round 7: kept) | 7 F |
| 09 | ep09_raw_final39328 | Haoran | pro6000 | raw | C3r2_tier2/ep09 | 39328 | 61fa5d3 | kept (round 7: kept) | 7 F |
| 10 | ep10_raw_final34267 | Haoran | pro6000 | raw | C3r2_tier2/ep10 | 34267 | 61fa5d3 | kept (round 7: kept) | 7 F |
| 11 | ep11_raw_24000 | Haoran | pro6000 | raw | C3r2_tier2/ep11 | 24000 | 61fa5d3 | kept (round 7: kept) | 7 F |
| 12 | ep12_k3fix_final23178 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep12_k3fix_s42 | 23178 | 61fa5d3 | kept (round 7: kept) | 7 F |
| 13 | ep13_w312runinws_90000 | Zuxing | ws031 (robolab) | G2 | 2026-10-31-tier2-runin-ws/ep13 | 90000 | 82837c8 | **replaced** (round 7: ep13_k3fix_final39254) | 8 F |
| 14 | ep14_k3fix_final44922 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep14_k3fix_s42 | 44922 | 61fa5d3 | kept (round 7: kept) | 7 F |
| 15 | ep15_raw_final44922 | Haoran | pro6000 | raw | C3r2_tier2/ep15 | 44922 | 61fa5d3 | kept (round 7: kept) | 7 F |
| 16 | ep16_z5a5_120000 | Zuxing | RTX 5090 | G2 | 2026-10-28-tier2-runin-a5/ep16 | 120000 | 82837c8 | kept (round 7: STRICT, confirmed) | 7 F |
| 17 | ep17_z5r2_61500 | Zuxing | RTX 5090 | K3fix | 2026-10-31-tier2-5090-round2/ep17 | 61500 | 82837c8 | kept (round 7: replaced) | 7 F |
| 18 | ep18_k3fix_final18911 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep18_k3fix_s42 | 18911 | 61fa5d3 | kept (round 7: kept) | 7 F |
| 19 | ep19_z5g2hardc2_360000 | Zuxing | RTX 5090 | G2 | 2026-10-25-tier2-runin-hard-g2/cont2/ep19 | 360000 | 82837c8 | kept (round 7: STRICT, confirmed) | 7 F |
| 20 | ep20_z5fill_90000 | Zuxing | RTX 5090 | K3fix | 2026-10-28-tier2-5090-fill/ep20 | 90000 | 82837c8 | kept (round 7: STRICT, confirmed) | 7 F |
| 21 | ep21_z5fillc_282500 | Zuxing | RTX 5090 | G2 | 2026-10-28-tier2-5090-fill/cont/ep21 | 282500 | 82837c8 | kept (round 7: replaced) | 7 F |
| 22 | ep22_z5k3fix_89500 | Zuxing | RTX 5090 | K3fix | 2026-10-24-tier2-runin-k3fix/ep22 | 89500 | 82837c8 | kept (round 7: STRICT, confirmed) | 7 F |
| 23 | ep23_raw_final20023 | Haoran | pro6000 | raw | C3r2_tier2/ep23 | 20023 | 61fa5d3 | kept (round 7: kept) | 7 F |
| 24 | ep24_z5q_150000 | Zuxing | RTX 5090 | G2 | 2026-10-29-tier2-5090-queue/ep24 | 150000 | 82837c8 | kept (round 7: replaced) | 7 F |
| 25 | ep25_z5a5c_280500 | Zuxing | RTX 5090 | K3fix | 2026-10-28-tier2-runin-a5/cont/ep25 | 280500 | 82837c8 | kept (round 7: STRICT, confirmed) | 7 F |
| 26 | ep26_zx_k3fix_92000 | Zuxing | pro6000 | K3fix | 2026-10-24-tier2-runin-k3fix-ep05-ep26/ep26 | 92000 | bc3b1bd | kept (round 7: kept) | 7 F |
| 27 | ep27_k3fix_final20863 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep27_k3fix_s42 | 20863 | 61fa5d3 | kept (round 7: kept) | 7 F |
| 28 | ep28_z5q_180000 | Zuxing | RTX 5090 | G2 | 2026-10-29-tier2-5090-queue/ep28 | 180000 | 82837c8 | kept (round 7: STRICT, confirmed) | 7 F |
| 29 | ep29_k3fix_final17358 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep29_k3fix_s42 | 17358 | 61fa5d3 | kept (round 7: kept) | 7 F |

"rollout": 7 F = round 7's phase-F run 7 (`../t2r7_20261102/selection/round7_20261102/inputs_r7/ROLLOUTS_F7.sha256`),
8 F = round 8's phase-F run 7 (`selection/round8_20261103/inputs_r8/ROLLOUTS_F8.sha256`). Per-episode records
(export, evaluated rollout and its sha256, frame reference, checkpoint, relocation, training stages):
`set/set_items_r8.json`, `set/set_manifest_r8.tsv`. The 29 kept items are round 7's items
(`../t2r7_20261102/set/set_items_r7.json`) with the evaluated run changed to trial 7; their training records are in
`../t2r7_20261102/` and `../t2ready_20261031/`.

## Episode 13: the new checkpoint

- **Run.** Zuxing's `2026-10-31-tier2-runin-ws/ep13`, trained on his robolab workstation ws031 (as trained:
  `ws031:/home/zuxinglu/workspace/runs/2026-10-31-tier2-runin-ws/ep13`). It reached the team server through his
  hand-over folder `zuxing.lu/runs/2026-11-02-tier2-r8-handover/ws031/2026-10-31-tier2-runin-ws/ep13` (copied
  2026-11-02 20:17 via his 5090 staging copy, checked against his `SHA256SUMS.ws031`; `MANIFEST.txt`). The run
  finished at 180,000 iterations (`final.json`, reason max_iterations). The candidate is `model_90000.safetensors`,
  round 7's mechanical target j3 (3/6 of the last iteration); it was not a watcher-best designation.
- **Training configuration** (Hydra overrides of the run): `experiment=g1_recon_body_coord_rsi_r2_runin_hard`,
  `scene.support=true`, `reset.seed=42`, `training.max_iterations=180000`, `training.early_stop_help_zero_tail=0`,
  `task.parquet` = the G2 parquet of sequence `2026-06-09_10-19-09_snack_box_desk_through_06`.
- **Training commit: 82837c8** (`dev.zuxing.hard_start`, pushed; an ancestor of this commit). Evidence: the first
  `train_t2:` line of the run's training log, `train_t2: exp=g1_recon_body_coord_rsi_r2_runin_hard
  root=human_motion_data_tier2_g2-2026-10-24 seq=2026-06-09_10-19-09_snack_box_desk_through_06 support=true gpu=0
  commit=82837c8` (ws031 has git; copy in `selection/round8_20261103/zuxing_copies_r8/2026-11-02-tier2-r8-handover/
  ws031/2026-10-31-tier2-runin-ws/ep13/train_log_first_train_t2_line.txt`, table `inputs_r8/ZUXING_CODE_R8.tsv`).
  82837c8 holds the experiment config `src/flash_chord/configs/experiment/g1_recon_body_coord_rsi_r2_runin_hard.yaml`.
- **Input / data root.** G2, `human_motion_data_tier2_g2-2026-10-24` (Zuxing's G2 builder; read-only copies and this
  episode's data-root build record `build_ep23_10_13_05_14_25_11.json` in `../t2r7_20261102/data/zuxing_g2_r7/`). The
  evaluator used the team server's copy of that root: parquet sha256 `7082f1de...`, frame reference
  `frame_refs_g2/episode_000013.npz` (`8fbe4f0b...`).
- **Relocation.** As in round 7: the evaluated file differs from the as-trained file only in the `task.parquet` /
  `reference.parquet` strings; all 114 tensors and the data block are unchanged (`RELOCATION_R8.tsv`: as trained
  `5fcf054e...` -> evaluated `08544a72...`).
- **Export.** b39b69b exporter, stock-verified, max relative difference 5.1e-05: `22a4b2bd...`, the checkpoint digest
  of episode 13 in the file (`selection/round8_20261103/exports/ep13_w312runinws_90000/export.json`).

## Missing or open (not in this commit)

1. **Launcher of the ws031 run.** Zuxing's robolab run script (and his run wrapper) is not on the team server, so it
   is not in this commit. The run is specified by its recorded Hydra overrides and training-log line.
2. **Host parquet bytes.** The hand-over `MANIFEST.txt` records the robolab `task.parquet` sha256 as `f77e946d...`
   (computed on ws032; the robolab data folders are rsync copies of ws031's). This differs from the team server's
   same-named G2 parquet (`7082f1de...`), which the evaluator used. Only byte identity can be checked from here;
   Zuxing needs to confirm that the arrays are the same (PREREG_R8 9.4; the same open item as round 7's 5090 G2 picks).
3. **Uncommitted worktree changes** at training time cannot be ruled out from a `train_t2:` line. PREREG_R8 9.5
   records that the 5090 and robolab worktrees show changed files that are LFS pointers replaced by their content
   (assets).
4. **No ws031 host evidence file** in the hand-over (`evidence/` holds `ws032.txt` only); the commit evidence for
   episode 13 is the training-log line above.
5. **G2 builder** is still not in a pushed commit of its author (read-only copies only;
   `../t2r7_20261102/MISSING_FROM_PUSHED_COMMITS.md`, item 1).
6. Unchanged from round 7: the launchers of Zuxing's RTX 5090 runs and the earlier `train_t2.sh` version
   (`../t2r7_20261102/MISSING_FROM_PUSHED_COMMITS.md`, item 2).

## Layout

- `set/`: `set_items_r8.json` (30 items; trial 7; placeholders `<ME>`, `<ZX>`, `<ZX5090>`, `<ZXWS>` for the server and
  host paths) and `set_manifest_r8.tsv`.
- `selection/round8_20261103/`: round 8's records, byte-identical copies in the server folder's layout, so that the
  lock files and pins resolve relative to it:
  - `PREREG_R8.md` (+ `.sha256`), `PREREG_R8_CHANGES.md`, `PREREG_R8_PINS.md` (+ `.sha256`), `AMENDMENTS_R8.md`
    (+ `.sha256`), `LOCK_R8.sha256`, `DRYRUN_R8.md`, `LAUNCH_R8.md`;
  - the pool: `POOL_R8.json` / `POOL_R8.sha256`, `inputs_r8/` (`R7_RECORDS_R8.json`, `pool_r8.csv`, `runs_r8.*`,
    `WATCHER_R8.tsv`, `ZUXING_CODE_R8.tsv`, `TRACKS_R8.sha256`, `ROLLOUTS_S8.sha256`, `ROLLOUTS_F8.sha256`,
    `resolve_r8.log`), `COPIES_r8.tsv`, `TELL_ZUXING_r8.txt`, `RELOCATION_R8.{json,sha256,tsv}`;
  - `zuxing_copies_r8/`: the text excerpts round 8 copied from Zuxing's folder (Hydra records, `final.json`, first
    `train_t2:` lines, `MANIFEST.txt` / `SHA256SUMS` files, watcher excerpts with the columns round 8 uses, the
    hand-over evidence file); the checkpoint copies themselves are not included (sha256 in `COPIES_r8.tsv`);
  - evaluations and scores: `manifest_{s8,f8}.{csv,json}`, `own_input_evals_{s8,f8}.csv`, `seeds_{s8,f8}.tsv`,
    `summary_{s8,f8}_own_input.*`, `scores_{s8,f8}/`, `exports/<job>/export.json`, `integrity/`, `preflight/`;
  - decisions: `selection_s8.{json,md,txt}`, `selection_s8_candidates.csv`, `final_r8.json`, `SELECTION_S8.sha256`,
    `SELECTION_R8.json`, `selection_r8.md`, `selection_r8_episodes.csv`, `summary_r8_trials.csv`,
    `SELECTION_R8.sha256`;
  - scripts: `r8_pool.py`, `r8_eval.py`, `r8_select.py`, `r8_driver.py`, `r8_launch.sh`, `dryrun/r8_mock_select.py`,
    and in `superseded_a1_20261103/` the pre-A1 versions of the driver, evaluation driver and launcher with the pre-A1
    pins (the pre-A1 driver ran the pool freeze at T_c, and its evaluation driver the 89 phase-S exports, before
    A1; `r8_driver.log`);
  - logs and heartbeats of the drivers, the selftest and the dry runs (`dryrun/`).
- `pack/`: the pack script `t2cons_r8_pack.py` and its pins `PACK_R8_PINS.md` (+ `.sha256`), the dry check run before
  the pins, and the placeholder pack's records (`r8_closest/file.json`, `pack_placeholder.json`,
  `validate_placeholder.json`, `pack_run_20261103_181108.json`).
- `tools/`: `make_set_items_r8.py` (writes `set/` from the frozen records) and `stage_r8.py` (copied the round-8
  records into this folder).
- `SHA256SUMS`: every file of this folder, paths relative to `robotic_grounding/flash_chord/scripts/track2/`.

## Code

All round-8 scripts in this folder are byte-identical to the files that ran (`LOCK_R8.sha256`, `PREREG_R8_PINS.md`).
The scripts they load from earlier rounds by their server paths are already in this history byte-identical:
`t2cons_r7_{pool,eval,consensus,pack}.py` and `t2cons_r7_eval_launch.sh` in `../t2r7_20261102/selection/scripts_r7/`;
`t2cons_eval.py`, `t2best5_driver.py`, `t2cons_r2_consensus.py`, `t2cons_a1_gate.py`, `t2cons_a1_consensus.py`,
`t2cons_consensus.py`, `t2cons_r5_relocate.py`, `t2cons_s7_pack.py` and `cache_keys_t2.py` in
`../t2ready_20261031/selection/scripts/`. Training code: ours `code/fc_r2` 61fa5d3 (src/, scripts/ and configs identical
to f2923d6 in this history), Zuxing's bc3b1bd and 82837c8; his checkpoints were exported with b39b69b. All are in this
commit's history (`../t2r7_20261102/README.md`, "Code").

`pack/t2cons_r8_pack.py` is round 7's pack script (`t2cons_r7_pack.py`) with the round-8 bookkeeping changes V1-V5 of
its docstring; PREREG_R8 section 6 asks for exactly this ("round 7's pack script, adapted and pinned first"). It was
pinned (`pack/PACK_R8_PINS.md`, 2026-11-03 18:10 CST) after one dry check and before it prepared or packed anything.
The kit packer prints one "locally computed" line; the script drops it from every stored log and never parses or
prints it.

## Reproduce the file

Every episode's rollout of trial 7 (`set/set_items_r8.json` `evaluated`) was produced by the stock kit evaluator
(`examples/run_policy_evaluation.py`, `flash_chord/scripts/evaluate_policy.py` 160d24f5) at frame 0, world count 1,
explicit reset, no help, on the stock runtime 8c778f9, with the exported checkpoint and the policy's own input. The file
is packed with the Oct-11 kit packer `tools/pack_submission.py` (47a45895), unchanged, each episode against its own
input frame reference, and passes the offline validation of the kit uploader `submit/v2d_submit.py` (fe5cb1dc).
Pose-column sha256 of the file: `07dd4f20b0f327a0591f10c5a5b74c3fee2d4d1d0031fd38782668af39e92200` (the placeholder
pack and the commit pack have the same pose columns; `pack-commit` checks it).

On the team server (kit Python, `-I -B`):

    <kit python> -I -B pack/t2cons_r8_pack.py check | prepare | pack | validate
    <kit python> -I -B pack/t2cons_r8_pack.py pack-commit --commit-url https://github.com/<owner>/<repo>/commit/<this commit> \
        --branch-head-file <ls-remote listing that shows this commit>

The script's gate verifies round 8's and round 7's lock chains, so it runs where the round-7 and round-8 folders exist.
Without the gates, the packed folder (`r8_closest/`: `results/`, `frame_refs/`, the placeholder CSV) can be re-packed
with any commit URL by `../t2ready_20261031/tools/repack_t2.py ... --items set/set_items_r8.json`, which checks that only
`code_commit_url` changes and that each episode's checkpoint digest equals `set_items_r8.json`.

## Record scope

Every round-8 file in `selection/round8_20261103/` is a byte-identical copy; no record is edited or withheld for its
content, so this folder has no REDACTIONS file and the lock files verify the copies. Not copied:
- binary files: checkpoint copies, relocated checkpoints and exports (`*.safetensors`; sha256 in `COPIES_r8.tsv`,
  `RELOCATION_R8.*`, `exports/*/export.json`, `POOL_R8.sha256`) and rollout parquets (sha256 in
  `inputs_r8/ROLLOUTS_{S8,F8}.sha256`);
- the run folders `runs_s/` and `runs_f/` (per-run `run.json`, logs and results; their values are in
  `own_input_evals_{s8,f8}.csv`) and the export try folders `exports/*/try*/`;
- `superseded_build1/`: the first build's documents and scripts, replaced before the pinned build steps and used for no
  decision (PREREG_R8, "When");
- empty lock files, and the pack step's console logs.
These stay on the team server and are available to the organisers on request.
