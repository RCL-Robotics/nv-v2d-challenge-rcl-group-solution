# Track 2 Tier 2: submission record (R9-CLOSEST, trial 3)

This folder is the record of the Tier-2 file `T2_R9_closest_trial03_<this commit, 7 hex>.csv`, packed with this
commit's URL as `code_commit_url`. The CSV is not in the repository; it is kept on the team server. Round 9 is the
selection round after round 8 (`../t2r8_20261103/`): it kept 27 of round 8's 30 checkpoints and replaced episodes 10,
17 and 21.

Selection uses only own-input and consensus-B scores: the official kit evaluator against each policy's own (raw or
filtered) Tier-2 input, and round 1's frozen consensus reference built from Tier-2 tracks (`refs/B`).

## Round 9 in short

`selection/round9_20261103/PREREG_R9.md` (hash-locked 2026-11-03 22:25 CST, `LOCK_R9.sha256` 22:33 CST, before the
pool freeze and before any round-9 export, evaluation or score) is round 8's rule with the deltas D1-D6:
- **D1 incumbents**: each episode's incumbent is its round-8 pick, with its recorded phase-F runs (never re-run): round
  7's phase-F runs for the 29 round-7 picks kept in round 8, round 8's phase-F runs for episode 13.
- **D2 pool**: (a) every file present and settled at T_c9 in Zuxing's hand-over folders `reports/tier2-G2/` (30 files)
  and `reports/tier2-K3fix/` (18 files); each file's final status at T_c9 is recorded, never a condition; (b) round 8's
  mechanical rule (round 7's R1-R4) re-applied to the runs that are new or trained further since round 8's snapshot.
  Candidates with earlier phase-S results reuse them (roles `reuse8`, `reuse7`); round 8's barred files stay barred.
- **D3 snapshot**: T_c9 = 2026-11-04 02:11:14 CST, the start of the pool freeze (`POOL_R9.json`).
- **D4 GPUs**: GPUs 3, 5, 6 and 7 when no other process runs there; GPUs 0, 1, 2 and 4 also next to our own ablation
  jobs (plain processes, one evaluation per GPU; PREREG_R9 3.5 and 9.1).
- **D5 deadlines**: phase S by 2026-11-04 12:00 CST, phase F by 20:00 CST. Both finished early (phase S selection
  02:27 CST, `SELECTION_R9.sha256` 02:33 CST).
- **D6**: no score other than own-input and consensus-B.
- No amendment was written.

Counts (`POOL_R9.json`, `inputs_r9/reports_r9.json`, `selection_s9.md`, `selection_r9.md`):
- All 48 report files were eligible; 13 of them were not their run's final checkpoint at T_c9 (recorded per file). The
  pool has 64 rows: 34 new screen candidates (18 report files, 12 G2 and 6 K3fix; 16 mechanical R4 picks of 4 runs
  trained further), 10 `reuse8`, 13 `reuse7`, 3 `barred` and 4 `incumbent` rows. All 34 screen candidates are Zuxing's
  (20 trained on the team server, 13 on his RTX 5090, 1 on a robolab host). 28 of 30 episodes were evaluated.
- Phase S: 102 official runs (34 x 3), all ok. Three finalists, one each for episodes 10, 17 and 21.
- Phase F: each finalist's 10 fresh runs were all ok, with own-input SP-SR = MP-SR = 1 in 10 of 10, and each passed the
  replacement test m_F >= max(m_P_frozen, m_P_fresh) + 0.01 (consensus-B AUC means):

| ep | round-8 pick | finalist | m_P_frozen | m_P_fresh | bar | m_F |
|---|---|---|---:|---:|---:|---:|
| 10 | ep10_raw_final34267 | ep10_zxrepg2_145500 | 0.4125 | 0.4107 | 0.4225 | 0.4289 |
| 17 | ep17_z5r2_61500 | ep17_zxrepk3fix_74500 | 0.6594 | 0.6594 | 0.6694 | 0.6730 |
| 21 | ep21_z5fillc_282500 | ep21_zxrepk3fix_80500 | 0.4945 | 0.4945 | 0.5045 | 0.5465 |

- **R9-CLOSEST = trial 3**: the complete trial whose 30-episode mean own-input AUC is closest to the mean over the 10
  trials (never a best-of). The 26 round-7 picks kept in rounds 8 and 9 use their round-7 phase-F run 3, episode 13
  uses its round-8 phase-F run 3, and episodes 10, 17 and 21 use their round-9 phase-F run 3.

Trials of S9 (own input = each policy against its own Tier-2 input; consensus-B = against `refs/B`; neither is the
competition's own score; `summary_r9_trials.csv`):

| trial | own-input AUC | SP-SR | MP-SR | MPPE (cm) | consensus-B AUC |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.8142 | 0.933 | 0.933 | 2.884 | 0.6708 |
| 2 | 0.8138 | 0.933 | 0.933 | 3.540 | 0.6709 |
| **3** | **0.8123** | **0.933** | **0.933** | **3.382** | **0.6715** |
| 4 | 0.8082 | 0.933 | 0.900 | 3.033 | 0.6646 |
| 5 | 0.7999 | 0.867 | 0.867 | 4.313 | 0.6620 |
| 6 | 0.8138 | 0.933 | 0.933 | 2.967 | 0.6699 |
| 7 | 0.8125 | 0.933 | 0.933 | 2.860 | 0.6711 |
| 8 | 0.8134 | 0.933 | 0.933 | 2.697 | 0.6692 |
| 9 | 0.8136 | 0.933 | 0.933 | 2.939 | 0.6702 |
| 10 | 0.8129 | 0.933 | 0.933 | 3.892 | 0.6703 |
| mean | 0.8115 | 0.927 | 0.923 | 3.251 | 0.6691 |

## The 30 checkpoints

| ep | job | trained by | machine | input | run | iteration | training commit | round 9 | rollout |
|---|---|---|---|---|---|---:|---|---|---|
| 00 | ep00_z5k3fix_90000 | Zuxing | RTX 5090 | K3fix | 2026-10-24-tier2-runin-k3fix/ep00 | 90000 | 82837c8 | kept | 3 F7 |
| 01 | ep01_k3fix_final44922 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep01_k3fix_s42 | 44922 | 61fa5d3 | kept | 3 F7 |
| 02 | ep02_k3fix_final34093 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep02_k3fix_s42 | 34093 | 61fa5d3 | kept | 3 F7 |
| 03 | ep03_k3fix_final27068 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep03_k3fix_s42 | 27068 | 61fa5d3 | kept | 3 F7 |
| 04 | ep04_zxg2_450000 | Zuxing | pro6000 | G2 | 2026-10-24-tier2-runin-g2/ep04 | 450000 | bc3b1bd | kept | 3 F7 |
| 05 | ep05_zx_k3fix_88000 | Zuxing | pro6000 | K3fix | 2026-10-24-tier2-runin-k3fix-ep05-ep26/ep05 | 88000 | bc3b1bd | kept | 3 F7 |
| 06 | ep06_k3fix_final28420 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep06_k3fix_s42 | 28420 | 61fa5d3 | kept | 3 F7 |
| 07 | ep07_k3fix_final18529 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep07_k3fix_s42 | 18529 | 61fa5d3 | kept | 3 F7 |
| 08 | ep08_k3easy_final17496 | Haoran | pro6000 | K3 | C3r2_tier2_k3easy/ep08 | 17496 | 61fa5d3 | kept | 3 F7 |
| 09 | ep09_raw_final39328 | Haoran | pro6000 | raw | C3r2_tier2/ep09 | 39328 | 61fa5d3 | kept | 3 F7 |
| 10 | ep10_zxrepg2_145500 | Zuxing | RTX 5090 | G2 | 2026-11-02-tier2-g2-redo-5090/ep10 | 145500 | 82837c8 | **replaced** (round 8: ep10_raw_final34267) | 3 F9 |
| 11 | ep11_raw_24000 | Haoran | pro6000 | raw | C3r2_tier2/ep11 | 24000 | 61fa5d3 | kept | 3 F7 |
| 12 | ep12_k3fix_final23178 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep12_k3fix_s42 | 23178 | 61fa5d3 | kept | 3 F7 |
| 13 | ep13_w312runinws_90000 | Zuxing | ws031 (robolab) | G2 | 2026-10-31-tier2-runin-ws/ep13 | 90000 | 82837c8 | kept (round-8 replacement) | 3 F8 |
| 14 | ep14_k3fix_final44922 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep14_k3fix_s42 | 44922 | 61fa5d3 | kept | 3 F7 |
| 15 | ep15_raw_final44922 | Haoran | pro6000 | raw | C3r2_tier2/ep15 | 44922 | 61fa5d3 | kept | 3 F7 |
| 16 | ep16_z5a5_120000 | Zuxing | RTX 5090 | G2 | 2026-10-28-tier2-runin-a5/ep16 | 120000 | 82837c8 | kept | 3 F7 |
| 17 | ep17_zxrepk3fix_74500 | Zuxing | RTX 5090 | K3fix | 2026-10-31-tier2-5090-round2/ep17 | 74500 | 82837c8 | **replaced** (round 8: ep17_z5r2_61500) | 3 F9 |
| 18 | ep18_k3fix_final18911 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep18_k3fix_s42 | 18911 | 61fa5d3 | kept | 3 F7 |
| 19 | ep19_z5g2hardc2_360000 | Zuxing | RTX 5090 | G2 | 2026-10-25-tier2-runin-hard-g2/cont2/ep19 | 360000 | 82837c8 | kept | 3 F7 |
| 20 | ep20_z5fill_90000 | Zuxing | RTX 5090 | K3fix | 2026-10-28-tier2-5090-fill/ep20 | 90000 | 82837c8 | kept | 3 F7 |
| 21 | ep21_zxrepk3fix_80500 | Zuxing | RTX 5090 | K3fix | 2026-10-31-tier2-5090-round2/ep21 | 80500 | 82837c8 | **replaced** (round 8: ep21_z5fillc_282500) | 3 F9 |
| 22 | ep22_z5k3fix_89500 | Zuxing | RTX 5090 | K3fix | 2026-10-24-tier2-runin-k3fix/ep22 | 89500 | 82837c8 | kept | 3 F7 |
| 23 | ep23_raw_final20023 | Haoran | pro6000 | raw | C3r2_tier2/ep23 | 20023 | 61fa5d3 | kept | 3 F7 |
| 24 | ep24_z5q_150000 | Zuxing | RTX 5090 | G2 | 2026-10-29-tier2-5090-queue/ep24 | 150000 | 82837c8 | kept | 3 F7 |
| 25 | ep25_z5a5c_280500 | Zuxing | RTX 5090 | K3fix | 2026-10-28-tier2-runin-a5/cont/ep25 | 280500 | 82837c8 | kept | 3 F7 |
| 26 | ep26_zx_k3fix_92000 | Zuxing | pro6000 | K3fix | 2026-10-24-tier2-runin-k3fix-ep05-ep26/ep26 | 92000 | bc3b1bd | kept | 3 F7 |
| 27 | ep27_k3fix_final20863 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep27_k3fix_s42 | 20863 | 61fa5d3 | kept | 3 F7 |
| 28 | ep28_z5q_180000 | Zuxing | RTX 5090 | G2 | 2026-10-29-tier2-5090-queue/ep28 | 180000 | 82837c8 | kept | 3 F7 |
| 29 | ep29_k3fix_final17358 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep29_k3fix_s42 | 17358 | 61fa5d3 | kept | 3 F7 |

"rollout": 3 F7 = round 7's phase-F run 3 (`../t2r7_20261102/selection/round7_20261102/inputs_r7/ROLLOUTS_F7.sha256`),
3 F8 = round 8's phase-F run 3 (`../t2r8_20261103/selection/round8_20261103/inputs_r8/ROLLOUTS_F8.sha256`), 3 F9 = round
9's phase-F run 3 (`selection/round9_20261103/inputs_r9/ROLLOUTS_F9.sha256`). Per-episode records (export, evaluated
rollout and its sha256, frame reference, checkpoint, relocation, training stages): `set/set_items_r9.json`,
`set/set_manifest_r9.tsv`. The 27 kept items are round 8's items (`../t2r8_20261103/set/set_items_r8.json`) with the
evaluated run changed to trial 3; their training records are in `../t2r8_20261103/`, `../t2r7_20261102/` and
`../t2ready_20261031/`.

## Episodes 10, 17 and 21: the new checkpoints

All three are files of Zuxing's hand-over folders, trained on his RTX 5090 at commit 82837c8 and evaluated as copies
relocated to the team server's parquet of the same data root.

**They are not their runs' final checkpoints.** Each run was planned for 180,000 iterations
(`training.max_iterations` in the checkpoint's embedded configuration) and had not reached it at T_c9
(`inputs_r9/reports_r9.json`, `final_at_t_c` = no). Zuxing's hand-over README names each file the newest complete
checkpoint of its run: the ep10 run was still training when he froze his candidates (2026-11-03 19:38 CST), and the two
round-2 runs were paused. That these are the newest checkpoints cannot be checked from the team server (see "Missing
or open", item 1). If a run is resumed, its later checkpoints are not part of this file.

| ep | job | report file (sha256) | run (as trained) | iteration / planned | experiment; embedded configuration | first `train_t2:` line, source |
|---|---|---|---|---|---|---|
| 10 | ep10_zxrepg2_145500 | `reports/tier2-G2/ep10-tier2-G2.safetensors` (c7e987b0...) | `2026-11-02-tier2-g2-redo-5090/ep10` | 145,500 / 180,000, still training | rsi_r2_runin, support false, 2048 worlds, seed 42 | `exp=g1_recon_body_coord_rsi_r2_runin ... commit=82837c8`, the 5090 hand-over evidence file |
| 17 | ep17_zxrepk3fix_74500 | `reports/tier2-K3fix/ep17-tier2-K3fix.safetensors` (c66a2a90...) | `2026-10-31-tier2-5090-round2/ep17` | 74,500 / 180,000, paused | rsi_r2_runin, support true, 2048 worlds, seed 42 | `exp=g1_recon_body_coord_rsi_r2_runin ... commit=82837c8`, the team server's copy of the run's `train.log` |
| 21 | ep21_zxrepk3fix_80500 | `reports/tier2-K3fix/ep21-tier2-K3fix.safetensors` (dd22e90e...) | `2026-10-31-tier2-5090-round2/ep21` | 80,500 / 180,000, paused | rsi_r2_runin_hard, support false, 2048 worlds, seed 42 | `exp=g1_recon_body_coord_rsi_r2_runin_hard ... commit=82837c8`, the team server's copy of the run's `train.log` |

- **Inputs.** ep10: G2 (`human_motion_data_tier2_g2-2026-10-24`), evaluated with the team server's parquet
  `892ec426...` and frame reference `frame_refs_g2/episode_000010.npz` (`95e39bc1...`). ep17 and ep21: K3fix (our
  `human_motion_data_tier2_k3fix`), parquets `25c13883...` and `57ae96f3...`, frame references `799597b5...` and
  `9bb13704...`.
- **Training commit 82837c8** (`dev.zuxing.hard_start`, pushed; an ancestor of this commit). Evidence per file in
  `inputs_r9/ZUXING_CODE_R9.tsv`; copies: `zuxing_copies_r9/runs/2026-11-02-tier2-r8-handover/evidence/5090.txt` (ep10)
  and `zuxing_copies_r9/runs/2026-11-01-round7-5090-ckpts/2026-10-31-tier2-5090-round2/ep{17,21}/hydra/.hydra/`
  (Hydra overrides of the round-2 runs).
- **Relocation.** Only the `task.parquet` / `reference.parquet` strings change; all 114 tensors and the data block are
  unchanged (`RELOCATION_R9.tsv`): c7e987b0... -> 94c59d2d..., c66a2a90... -> 1da2d7f3..., dd22e90e... -> 829a91e2....
- **Export.** b39b69b exporter, stock-verified at tolerance 1e-4: ep10 `541b4ab8...` (max relative difference 5.0e-05),
  ep17 `81b57136...` (3.8e-05), ep21 `b40da890...` (4.6e-05); these are the checkpoint digests of the three episodes in
  the file (`selection/round9_20261103/exports/<job>/export.json`).

## Missing or open (not in this commit)

1. **5090 run-folder listing.** For all three new checkpoints, "newest checkpoint of the run" rests on Zuxing's
   hand-over README. The team server's copies of the round-2 runs end at 61,500 (ep17) and 69,500 (ep21)
   (`zuxing_copies_r9/runs/2026-11-01-round7-5090-ckpts/SHA256SUMS`), and the ep10 redo run has no copy or run record on
   the team server. A listing of the 5090 run folders with checksums and the training-log tails is needed to show that
   each file is its run's newest checkpoint and whether the runs are stopped.
2. **Zuxing's own statement on his choices.** PREREG_R9 9.3 gives his rules as written in his hand-over READMEs and as
   traced in his records by our check: the last checkpoint of each run, and the own-input AUC between runs of different
   configurations. His own written statement of which evaluations his `reports/` choices used, and the code that wrote
   his final selection table (not on disk), are still to come. The two check reports named in PREREG_R9 9.2 (sha256 in
   `PREREG_R9_PINS.md`, note 6) are not in this commit; they are kept by the team and available on request.
3. **ep10 redo parquet identity.** The 5090 hand-over evidence file records the `task.parquet` sha256 of the ep10 redo
   run as `efa5d11a...`; the team server's same-named G2 parquet, which the evaluator used, is `892ec426...`. Only byte
   identity can be checked from here, and the bytes differ; Zuxing needs to confirm that the arrays are the same (as for
   the earlier 5090 and robolab G2 picks). For ep17 and ep21, the 5090 copy of our K3fix parquet has no recorded
   sha256 (round 7's open item 4, `../t2r7_20261102/MISSING_FROM_PUSHED_COMMITS.md`).
4. **Launchers** of the 5090 runs `2026-11-02-tier2-g2-redo-5090` and `2026-10-31-tier2-5090-round2` and the 5090
   environment specification are not on the team server, so they are not in this commit. The runs are specified by the
   embedded configuration, the Hydra overrides (round 2) and the first `train_t2:` line.
5. **Uncommitted worktree changes** at training time cannot be ruled out from a `train_t2:` line. The 5090 evidence file
   (2026-11-02 19:28) lists 65 changed files in the `dev.zuxing.hard_start` worktree, all LFS pointers replaced by their
   content (assets).
6. Unchanged from rounds 7-8: the G2 builder is not in a pushed commit of its author
   (`../t2r7_20261102/MISSING_FROM_PUSHED_COMMITS.md`, item 1), and episode 13's open items
   (`../t2r8_20261103/README.md`, "Missing or open").
7. Teammates' upstream training choices already made (the G2 input, and the inputs chosen for some of Zuxing's runs) are
   disclosed separately (PREREG_R9 9.3); every run stays in one candidate pool, selected only by the scores above.

## Layout

- `set/`: `set_items_r9.json` (30 items; trial 3; placeholders `<ME>`, `<ZX>`, `<ZX5090>`, `<ZXWS>` for the server and
  host paths) and `set_manifest_r9.tsv`.
- `selection/round9_20261103/`: round 9's records, byte-identical copies in the server folder's layout, so that the
  lock files and pins resolve relative to it:
  - `PREREG_R9.md` (+ `.sha256`), `PREREG_R9_PINS.md` (+ `.sha256`), `LOCK_R9.sha256`, `DRYRUN_R9.md`;
  - the pool: `POOL_R9.json` / `POOL_R9.sha256`, `inputs_r9/` (`R8_RECORDS_R9.json`, `pool_r9.csv`, `runs_r9.*`,
    `reports_r9.*`, `ZUXING_CODE_R9.tsv`, `TRACKS_R9.sha256`, `ROLLOUTS_S9.sha256`, `ROLLOUTS_F9.sha256`,
    `resolve_r9.log`), `COPIES_r9.tsv`, `TELL_ZUXING_r9.txt`, `RELOCATION_R9.{json,sha256,tsv}`;
  - `zuxing_copies_r9/`: the text excerpts round 9 copied from Zuxing's folder (Hydra records, `final.json`, first
    `train_t2:` lines, `MANIFEST.txt` / `SHA256SUMS` files, the hand-over evidence files); the checkpoint copies
    themselves are not included (sha256 in `COPIES_r9.tsv`);
  - evaluations and scores: `manifest_{s9,f9}.{csv,json}`, `own_input_evals_{s9,f9}.csv`, `seeds_{s9,f9}.tsv`,
    `summary_{s9,f9}_own_input.*`, `scores_{s9,f9}/`, `exports/<job>/export.json`, `integrity/`, `preflight/`;
  - decisions: `selection_s9.{json,md,txt}`, `selection_s9_candidates.csv`, `final_r9.json`, `SELECTION_S9.sha256`,
    `SELECTION_R9.json`, `selection_r9.md`, `selection_r9_episodes.csv`, `summary_r9_trials.csv`,
    `SELECTION_R9.sha256` (47e5a8d7...);
  - scripts: `r9_pool.py`, `r9_eval.py`, `r9_select.py`, `r9_driver.py`, `r9_launch.sh`, `dryrun/r9_mock_select.py`,
    `dryrun/r9_reuse_check.py`, `dryrun/postlock_select_mock.py`;
  - logs and heartbeats of the drivers, the selftest and the dry runs (`dryrun/`, including the trial pool's text
    records).
- `pack/`: the pack script `t2cons_r9_pack.py` and its pins `PACK_R9_PINS.md` (+ `.sha256`), the dry check run before
  the pins, and the placeholder pack's records (`r9_closest/file.json`, `pack_placeholder.json`,
  `validate_placeholder.json`, `pack_run_20261104_024630.json`).
- `tools/`: `make_set_items_r9.py` (writes `set/` from the frozen records) and `stage_r9.py` (copied the round-9
  records into this folder).
- `SHA256SUMS`: every file of this folder, paths relative to `robotic_grounding/flash_chord/scripts/track2/`.

## Code

All round-9 scripts in this folder are byte-identical to the files that ran (`LOCK_R9.sha256`, `PREREG_R9_PINS.md`).
The scripts they load from earlier rounds by their server paths are already in this history byte-identical:
`r8_pool.py` (bebeb013), `r8_eval.py` (5dcf6a82) and `r8_select.py` (735149ac) in
`../t2r8_20261103/selection/round8_20261103/`; `t2cons_r7_{pool,eval,consensus}.py` in
`../t2r7_20261102/selection/scripts_r7/`; `t2cons_eval.py`, `t2best5_driver.py`, `t2cons_r5_relocate.py` and
`t2cons_s7_pack.py` in `../t2ready_20261031/selection/scripts/`; `../eval_all.py` (65729592). Training code of the
three new checkpoints: Zuxing's 82837c8; they were exported with b39b69b. The training code of the 27 kept checkpoints
is listed in `../t2r8_20261103/README.md` ("Code"). All are in this commit's history.

`pack/t2cons_r9_pack.py` is round 8's pack script (`../t2r8_20261103/pack/t2cons_r8_pack.py`) with the round-9
bookkeeping changes W1-W5 of its docstring; PREREG_R9 section 6 asks for exactly this ("round 8's pack procedure,
adapted and pinned first"). It was pinned (`pack/PACK_R9_PINS.md`, 2026-11-04 02:46 CST) after one dry check and before
it prepared or packed anything. The kit packer prints one "locally computed" line; the script drops it from every
stored log and never parses or prints it.

## Reproduce the file

Every episode's rollout of trial 3 (`set/set_items_r9.json` `evaluated`) was produced by the stock kit evaluator
(`examples/run_policy_evaluation.py`, `flash_chord/scripts/evaluate_policy.py` 160d24f5) at frame 0, world count 1,
explicit reset, no help, on the stock runtime 8c778f9, with the exported checkpoint and the policy's own input. The file
is packed with the Oct-11 kit packer `tools/pack_submission.py` (47a45895), unchanged, each episode against its own
input frame reference, and passes the offline validation of the kit uploader `submit/v2d_submit.py` (fe5cb1dc).
Pose-column sha256 of the file: `1380472dccd6cfcc60ffa2d84923185ba23d5851399dc685e52cbeb165d12658` (the placeholder
pack and the commit pack have the same pose columns; `pack-commit` checks it).

On the team server (kit Python, `-I -B`):

    <kit python> -I -B pack/t2cons_r9_pack.py check | prepare | pack | validate
    <kit python> -I -B pack/t2cons_r9_pack.py pack-commit --commit-url https://github.com/<owner>/<repo>/commit/<this commit> \
        --branch-head-file <ls-remote listing that shows this commit>

The script's gate verifies round 9's, round 8's and round 7's lock chains, so it runs where those folders exist.
Without the gates, the packed folder (`r9_closest/`: `results/`, `frame_refs/`, the placeholder CSV) can be re-packed
with any commit URL by `../t2ready_20261031/tools/repack_t2.py ... --items set/set_items_r9.json`, which checks that only
`code_commit_url` changes and that each episode's checkpoint digest equals `set_items_r9.json`.

## Record scope

Every round-9 file in `selection/round9_20261103/` is a byte-identical copy; no record is edited or withheld for its
content, so this folder has no REDACTIONS file and the lock files verify the copies. Not copied:
- binary files: checkpoint copies, relocated checkpoints and exports (`*.safetensors`; sha256 in `COPIES_r9.tsv`,
  `RELOCATION_R9.*`, `exports/*/export.json`, `POOL_R9.sha256`), including the trial pool's copies in `dryrun/`, and
  rollout parquets (sha256 in `inputs_r9/ROLLOUTS_{S9,F9}.sha256`);
- the run folders `runs_s/` and `runs_f/` (per-run `run.json`, logs and results; their values are in
  `own_input_evals_{s9,f9}.csv`) and the export try folders `exports/*/try*/`;
- `dev/`: the development copies of the round-9 scripts and their plumbing outputs, made before the pinned build steps
  and used for no decision (PREREG_R9 "When"; `PREREG_R9_PINS.md` note 5: not pinned, not used);
- empty lock files (`evallock/`, `.driver.lock`, `.r9_driver.lock`), the empty `tmp/`, and the pack step's console logs.
These stay on the team server and are available to the organisers on request.
