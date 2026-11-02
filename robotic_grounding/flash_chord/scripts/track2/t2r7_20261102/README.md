# Track 2 Tier 2: submission record (R7-CLOSEST, trial 2)

This folder is the record of the Tier-2 file `T2_R7_closest_trial02_<this commit, 7 hex>.csv`, packed with this
commit's URL as `code_commit_url`. The second Tier-2 file of this commit, `T2_K3only_closest_trial02_<this commit, 7
hex>.csv`, is recorded in `../t2k3only_20261102/`. Neither CSV is in the repository; both are kept on the team server.

Both files are re-packs of recorded official rollouts with `../t2ready_20261031/tools/repack_t2.py`: no evaluation and
no export is repeated, the kit packer runs unchanged, and the tool checks that every row equals the earlier pack of the
same trial except for the `code_commit_url` field. Pose-column sha256 of the R7 file:
`8055a1488ab3d97d9b4d7b4daad15be464da869637d5265db16c94440940462a`.

## How the 30 checkpoints were chosen

Round 7 (`selection/round7_20261102/PREREG_R7.md`, hash-locked before any round-7 score) picks one checkpoint per
episode from a mechanical pool: the Tier-2 training runs of Haoran Zhang and Zuxing Lu on the team server that meet
its pool rule, and Zuxing's 45 copied RTX 5090 run folders (118 runs: 59 ours, 14 of Zuxing's on the team server, the
45 RTX 5090 folders). Selection uses only own-input and consensus-B (noisy) scores: the official kit evaluator against
each policy's own (raw or filtered) Tier-2 input, and a frozen noisy consensus reference built from Tier-2 tracks
(round 1's `refs/B`).

- Pool: per run its final checkpoint plus up to three evenly spaced checkpoints of the second half (K = 3), at the
  cut-off 2026-11-02 06:16:08 CST: 399 candidates (154 new, 245 re-used from round 6). Zuxing's single-checkpoint copies
  and bundles are not candidates; his runs enter only through this rule.
- Phase S: 3 official runs per candidate (462 new evaluations, 735 round-6 runs re-scored and reproduced exactly).
- Phase F: 10 fresh paired trials of the 23 incumbents, 19 finalists and 7 report-only round-6 references (490 runs).
- 23 episodes keep round 6's checkpoint unless a finalist beats it by >= 0.01 (consensus AUC, with SP-SR = MP-SR = 1 in
  >= 8 of 10 runs): 4 were replaced (04, 17, 21, 24). The other 7 episodes (00, 16, 19, 20, 22, 25, 28; "STRICT" in the
  records) have no incumbent and were re-decided from the pool alone: every rank-1 finalist was confirmed.
- The file is trial 2 of the 10 phase-F trials: the trial whose 30-episode mean own-input AUC is closest to the mean
  over trials (never a best-of).

Means over the 10 trials (own input = each policy against its own Tier-2 input; not leaderboard numbers): own-input AUC
0.7918, SP-SR 0.9267, MP-SR 0.9233, MPPE 3.28 cm; consensus-B AUC 0.6651. Trial 2: own-input AUC 0.7933, SP-SR 0.933,
MP-SR 0.933, MPPE 3.58 cm, consensus-B AUC 0.6670.

## The 30 checkpoints

| ep | job | trained by | machine | input | run | iteration | training commit | round 7 |
|---|---|---|---|---|---|---:|---|---|
| 00 | ep00_z5k3fix_90000 | Zuxing | RTX 5090 | K3fix | 2026-10-24-tier2-runin-k3fix/ep00 | 90000 | 82837c8 | STRICT, confirmed |
| 01 | ep01_k3fix_final44922 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep01_k3fix_s42 | 44922 | 61fa5d3 | kept |
| 02 | ep02_k3fix_final34093 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep02_k3fix_s42 | 34093 | 61fa5d3 | kept |
| 03 | ep03_k3fix_final27068 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep03_k3fix_s42 | 27068 | 61fa5d3 | kept |
| 04 | ep04_zxg2_450000 | Zuxing | pro6000 | G2 | 2026-10-24-tier2-runin-g2/ep04 | 450000 | bc3b1bd | replaced |
| 05 | ep05_zx_k3fix_88000 | Zuxing | pro6000 | K3fix | 2026-10-24-tier2-runin-k3fix-ep05-ep26/ep05 | 88000 | bc3b1bd | kept |
| 06 | ep06_k3fix_final28420 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep06_k3fix_s42 | 28420 | 61fa5d3 | kept |
| 07 | ep07_k3fix_final18529 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep07_k3fix_s42 | 18529 | 61fa5d3 | kept |
| 08 | ep08_k3easy_final17496 | Haoran | pro6000 | K3 | C3r2_tier2_k3easy/ep08 | 17496 | 61fa5d3 | kept |
| 09 | ep09_raw_final39328 | Haoran | pro6000 | raw | C3r2_tier2/ep09 | 39328 | 61fa5d3 | kept |
| 10 | ep10_raw_final34267 | Haoran | pro6000 | raw | C3r2_tier2/ep10 | 34267 | 61fa5d3 | kept |
| 11 | ep11_raw_24000 | Haoran | pro6000 | raw | C3r2_tier2/ep11 | 24000 | 61fa5d3 | kept |
| 12 | ep12_k3fix_final23178 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep12_k3fix_s42 | 23178 | 61fa5d3 | kept |
| 13 | ep13_k3fix_final39254 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep13_k3fix_s42 | 39254 | 61fa5d3 | kept |
| 14 | ep14_k3fix_final44922 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep14_k3fix_s42 | 44922 | 61fa5d3 | kept |
| 15 | ep15_raw_final44922 | Haoran | pro6000 | raw | C3r2_tier2/ep15 | 44922 | 61fa5d3 | kept |
| 16 | ep16_z5a5_120000 | Zuxing | RTX 5090 | G2 | 2026-10-28-tier2-runin-a5/ep16 | 120000 | 82837c8 | STRICT, confirmed |
| 17 | ep17_z5r2_61500 | Zuxing | RTX 5090 | K3fix | 2026-10-31-tier2-5090-round2/ep17 | 61500 | 82837c8 | replaced |
| 18 | ep18_k3fix_final18911 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep18_k3fix_s42 | 18911 | 61fa5d3 | kept |
| 19 | ep19_z5g2hardc2_360000 | Zuxing | RTX 5090 | G2 | 2026-10-25-tier2-runin-hard-g2/cont2/ep19 | 360000 | 82837c8 | STRICT, confirmed |
| 20 | ep20_z5fill_90000 | Zuxing | RTX 5090 | K3fix | 2026-10-28-tier2-5090-fill/ep20 | 90000 | 82837c8 | STRICT, confirmed |
| 21 | ep21_z5fillc_282500 | Zuxing | RTX 5090 | G2 | 2026-10-28-tier2-5090-fill/cont/ep21 | 282500 | 82837c8 | replaced |
| 22 | ep22_z5k3fix_89500 | Zuxing | RTX 5090 | K3fix | 2026-10-24-tier2-runin-k3fix/ep22 | 89500 | 82837c8 | STRICT, confirmed |
| 23 | ep23_raw_final20023 | Haoran | pro6000 | raw | C3r2_tier2/ep23 | 20023 | 61fa5d3 | kept |
| 24 | ep24_z5q_150000 | Zuxing | RTX 5090 | G2 | 2026-10-29-tier2-5090-queue/ep24 | 150000 | 82837c8 | replaced |
| 25 | ep25_z5a5c_280500 | Zuxing | RTX 5090 | K3fix | 2026-10-28-tier2-runin-a5/cont/ep25 | 280500 | 82837c8 | STRICT, confirmed |
| 26 | ep26_zx_k3fix_92000 | Zuxing | pro6000 | K3fix | 2026-10-24-tier2-runin-k3fix-ep05-ep26/ep26 | 92000 | bc3b1bd | kept |
| 27 | ep27_k3fix_final20863 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep27_k3fix_s42 | 20863 | 61fa5d3 | kept |
| 28 | ep28_z5q_180000 | Zuxing | RTX 5090 | G2 | 2026-10-29-tier2-5090-queue/ep28 | 180000 | 82837c8 | STRICT, confirmed |
| 29 | ep29_k3fix_final17358 | Haoran | pro6000 | K3fix | C3r2_tier2_k3fix/ep29_k3fix_s42 | 17358 | 61fa5d3 | kept |

Full per-episode records (checkpoint, evaluated file, relocation, export, rollout, training stages and overrides):
`set/set_items_r7.json`, `set/set_manifest_r7.tsv`. Episodes kept from round 6 refer to `../t2ready_20261031/` for
their training records.

## Code

The selection, evaluation-driver and pack scripts in the record folders are byte-identical to the ones that ran. The
training launchers are byte-identical to the launcher files on the team server; our raw-run queue script is committed
in both versions that launched runs of the two files (`../t2ready_20261031/README.md`, "Training launchers"). Not on
the team server, so not in this commit (`MISSING_FROM_PUSHED_COMMITS.md`): the launchers of Zuxing's RTX 5090 runs, and
the version of his `train_t2.sh` that launched his ep05 / ep26 run on 2026-10-24 at 04:35:10 (the file was rewritten
at 05:19:09; the committed copy is that later version). The training code of every checkpoint is in this commit's
history; its parents are the training commits:
- **Ours:** `code/fc_r2` at 61fa5d3, whose `robotic_grounding/flash_chord` src/, scripts/ and configs are identical to
  f2923d6 (an ancestor of 41403d3, `dev.haoran.c3r2`, the first parent).
- **Zuxing's:** bc3b1bd (`dev.zuxing.speedup`) for his team-server runs and 82837c8 (`dev.zuxing.hard_start`) for his
  RTX 5090 runs; both are parents of this commit. His checkpoints were exported with b39b69b, an ancestor of both.
- 7fbc227 (`dev.haoran.ep28w2`) is the fourth parent; it supplies this tree's `objectives/recon_body.py` and
  `training/ppo/coordinate/export.py` (optional terms, off by default).

Inputs: raw Tier 2, K3 / K3fix (builders in `../t2ready_20261031/data/ours/`) and G2 (Zuxing's builder; copies in
`../t2ready_20261031/data/zuxing_g2/` and `data/zuxing_g2_r7/`). ep19, ep21 and ep25 are continuation stages; all
stages' Hydra records are in `records/zuxing_runs_r7/round7_5090_copy/`. What is not in any pushed commit of its author
is listed in `MISSING_FROM_PUSHED_COMMITS.md`.

Relocation: an RTX 5090 checkpoint stores the 5090 parquet path. The evaluated file differs from the as-trained file
only in the `task.parquet` / `reference.parquet` strings (this server's path of the same parquet); every tensor byte is
unchanged (`selection/round7_20261102/RELOCATION_R7.tsv`, verified byte for byte).

## Layout

- `set/`: the 30 checkpoints of the file.
- `selection/round7_20261102/`: PREREG_R7 (+ PINS, PINS_b, PINS_c), DRYRUN, LOCK, the pool / runs / incumbents /
  references, the relocation and copy tables, the manifests, every official run's own-input values, the gate and
  consensus scores of both phases, the selections and `SELECTION_S7.sha256` / `SELECTION_R7.sha256`.
- `selection/scripts_r7/`: pool resolver, evaluation driver and launcher, scorer, pack script, runner.
- `records/`: export records, stored checkpoint configs, Zuxing's run records (read-only copies, `records/COPIES_t2r7.tsv`
  and round 7's `COPIES_r7.tsv`: source sha256 before == copy == after).
- `data/zuxing_g2_r7/`: Zuxing's G2 builder and data-root build records (read-only copies).
- `train/zuxing_r7/`: his team-server training wrapper `train_t2.sh` and his launch scripts `run_*.sh` (read-only
  copies).
- `tools/mock_select.py`: the mock harness of the scorer.
- `SHA256SUMS`: every file of the three Tier-2 record folders (`t2ready_20261031/`, `t2r7_20261102/`,
  `t2k3only_20261102/`), paths relative to `robotic_grounding/flash_chord/scripts/track2/`.
- `REDACTIONS.tsv`: see "Record scope" below.

## Reproduce the file

From the records: every episode's rollout of trial 2 (`set/set_items_r7.json` `evaluated`) was produced by the stock
kit evaluator (`examples/run_policy_evaluation.py`, `flash_chord/scripts/evaluate_policy.py` 160d24f5) at frame 0,
world count 1, explicit reset, no help, on the stock runtime 8c778f9, with the exported checkpoint (`export.sha256`)
and the policy's own input. The file is packed with the kit packer `tools/pack_submission.py` (47a45895), unchanged,
each episode against its own input frame reference, and passes the uploader's offline validation
(`submit/v2d_submit.py` fe5cb1dc).

Re-pack with this commit's URL (kit Python, `-I -B`; `<R7_PACK>` is the `r7_closest` folder written by
`selection/scripts_r7/t2cons_r7_pack.py`, holding `results/`, `frame_refs/` and the placeholder-URL CSV):

    <kit python> -I -B ../t2ready_20261031/tools/repack_t2.py --kit <KIT> --dataset-root <V2D_DATA> \
        --pack-dir <R7_PACK> --source-csv <placeholder-URL CSV in R7_PACK> --items set/set_items_r7.json \
        --commit-url https://github.com/<owner>/<repo>/commit/<this commit> --out-dir <DIR> \
        --name T2_R7_closest_trial02_<7 hex>.csv

## Record scope

This record is a subset of the working folders. `REDACTIONS.tsv` lists every file of the earlier record folders that is
not committed byte-identical, and the committed K3-only files that are edited or rewritten copies of files of the
K3-only server folder. Each row gives the original sha256 (the value the frozen lock files and pinned scripts check) and
the committed sha256, or `withheld`:
- rewritten: the README of each of the three record folders;
- edited: documents with sentences removed or reworded (removed sections are marked "not included in this copy"),
  among them Zuxing's two bundle notes of his remote checkpoints (run paths and inputs kept), generated summaries and
  records with a sentence or clause removed from a text or note field (values unchanged), and `set/set_items_r7.json`,
  whose `code_in_this_history` notes are corrected (values unchanged);
- withheld, among them: working files of earlier selection rounds (rounds 2 and 4; most of rounds 3 and 5),
  including two round-5 files that round 7 pins, `selection_r5.json` and `PREREG_R5.md` (their sha256 are in PREREG_R7
  section 11 and in `REDACTIONS.tsv`; round 7's pin check therefore needs them, and they are available to the
  organisers on request); the selection scripts of rounds 2 to 5 that later rounds do not load; the K3CA and T2fix
  training launchers (no checkpoint of those runs is in either file); staging tools; analysis notes; the two freeze
  sidecars `*.sha256.json` of the K3 and K3CA input pre-registrations; round 1's code pin list `inputs/CODE.sha256`;
  and some dry-run plans.

**Scripts in this record.** Every script (`.py`, `.sh`) in the three record folders is committed byte-identical to its
source. The scripts of the earlier record commit are unchanged, and the three K3-only tools equal the files of the
K3-only server folder. Seven scripts were added on 2026-11-03 from the team server; each equals the live file, whose
mtime predates its first use (`../t2ready_20261031/CODE_COPIES.tsv`: source, sha256, mtime, first use). The scripts
include our training launchers (the raw-run queue script in both versions that ran) with the helper module they
import, copies of Zuxing's team-server launchers and of the evaluation watcher they start, round 0's inventory builder,
the collision-hull check, and round 1's A1 evaluation and resolver scripts. Some scripts also contain stages that are
not part of the selection; the selection steps read own-input and consensus-B files only.

**Code not in this record.**
- Not on the team server, so not in this commit: the launchers of Zuxing's RTX 5090 runs, the version of his
  `train_t2.sh` that launched his ep05 / ep26 run, and his run wrapper `launch_run.sh`
  (`MISSING_FROM_PUSHED_COMMITS.md`).
- The official challenge kit (evaluator, metric code, packer, uploader) is not copied; the records name its files by
  sha256.
- Not included:
  - report-only diagnostic scripts;
  - launchers of runs that have no checkpoint in either file (the K3CA and T2fix training launchers), and the launch
    helpers of other runs that the included launchers query only for GPU scheduling;
  - our evaluation watcher (t2watch); no checkpoint of either file was designated through it (for some runs its "best"
    coincides with the final checkpoint the files use);
  - the selection scripts of rounds 2 to 5, apart from the two modules that later rounds load (of those rounds, the
    score files that round 7 pins (consensus scores and job stats of rounds 3 and 5), round 5's `SELECTION_R5.sha256`,
    which round 7 also pins, and the relocation records of rounds 3 and 5 are included);
  - two scripts that round 0's PREREG pins only as read-only pattern sources (its driver neither imports nor runs them);
  - the design and root-cause analysis scripts of the K3fix input (the gate scripts read two of their output tables
    only to check that they reproduce them; no gate result depends on those tables);
  - the first version of the K3fix pilot launcher, replaced before the pilot started, and the patch script and test
    used to make the version that ran (`../t2ready_20261031/data/ours/k3fix/build_files*.sha256` pin them).

The lock files that are committed (`*.sha256`, `LOCK_R7.sha256`, `SELECTION_*.sha256`, snapshot lists) are unchanged,
so they name the original hashes, also of edited and withheld files. The gated scripts (`t2cons_r7_pool.py`,
`t2cons_r7_eval.py`, `t2cons_r7_consensus.py`, `t2cons_r7_pack.py`) verify those original files and run only where the
originals exist; the originals are available to the organisers. Re-packing with `repack_t2.py` needs no such gate.
