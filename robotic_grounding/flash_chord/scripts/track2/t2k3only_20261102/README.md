# Track 2 Tier 2: K3-only file (K3FULL composition, trial 2)

This folder is the record of the Tier-2 file `T2_K3only_closest_trial02_<this commit, 7 hex>.csv`, packed with this
commit's URL as `code_commit_url`. The file is not in the repository; it is kept on the team server. It is a re-pack of
the recorded official rollouts of trial 2 (no evaluation or export repeated); every row equals the earlier pack of the
same trial except for the `code_commit_url` field. Pose-column sha256:
`0b0e3544d69fdbc1173f13fc887598a31a8bd477aeb6391ed35eca474873a2a8`.

The round-7 file of this commit is recorded in `../t2r7_20261102/`.

## Composition (fixed by the 2026-10-24 files)

This is the **K3FULL** composition of `t2_best5_20261024` (round 0, `../t2ready_20261031/selection/round0_t2best5_20261024/`).
Its PREREG section 4 defines it as "PREREG_v3, as VERDICT.json gives it; PENDING maps to raw"; `CHOICE.json` re-derives
it from VERDICT and the manifest.
- **22 episodes** use our K3fix runs (19) or our K3 easy-clip runs (3: ep08, 20, 23). The K3 easy-clip parquets are
  byte-identical to K3fix. Each uses its `final.json` checkpoint.
- **8 episodes** use our raw `C3r2_tier2` runs (`final.json`):
  - 00, 05, 15, 19, 22, 26 and 28 failed input gate G4 (G2 also failed for 28);
  - ep04 passed the gates, but VERDICT.json (snapshot `inputs/VERDICT_k3full.json`) says PENDING ("training"): it
    was last written on 2026-10-25 at 05:15:23 CST, before ep04's K3fix run finished at 05:39:42 (`final.json`,
    model_44922), and it was not rewritten afterwards. PENDING maps to raw.
  - Launchers: ep00's raw run was started on 2026-10-16 at 15:09:39 by the previous version of the queue script,
    `../t2ready_20261031/train/ours/track2_queue_tier2.v2_20261016.sh`; the other seven by
    `../t2ready_20261031/train/ours/track2_queue_tier2.sh` (`../t2ready_20261031/README.md`, "Training launchers").
- No G2 input and no checkpoint of Zuxing's.
- Every checkpoint was trained at `code/fc_r2` 61fa5d3 (its `robotic_grounding/flash_chord` src/, scripts/ and configs
  equal f2923d6's, an ancestor of this commit) with the C3r2 recipe (`g1_recon_body_coord_rsi_r2`, 45k cap, seed 42).

| ep | job | input | ckpt sha | export sha | own AUC | own SP | own MP | own MPPE (cm) | frame reference |
|---:|---|---|---|---|---:|---:|---:|---:|---|
| 00 | ep00_raw_final41043 | raw | 49323000 | 55aa98f9 | 0.2632 | 1 | 1 | 1.79 | released Tier-2 |
| 01 | ep01_k3fix_final44922 | K3fix | d40a325a | a15d9e85 | 0.1411 | 0 | 0 | 35.83 | frame_refs_k3fix |
| 02 | ep02_k3fix_final34093 | K3fix | 3311068b | 3f1ae637 | 0.9214 | 1 | 1 | 1.05 | frame_refs_k3fix |
| 03 | ep03_k3fix_final27068 | K3fix | 88278b72 | e042571c | 0.9847 | 1 | 1 | 0.56 | frame_refs_k3fix |
| 04 | ep04_raw_final44922 | raw (PENDING) | e5d58f02 | 1de36ebd | 0.1104 | 0 | 0 | 123.15 | released Tier-2 |
| 05 | ep05_raw_final44922 | raw | e1fe6427 | bc861de5 | 0.2560 | 1 | 1 | 2.32 | released Tier-2 |
| 06 | ep06_k3fix_final28420 | K3fix | 88dafebf | 84bd4858 | 0.9239 | 1 | 1 | 1.06 | frame_refs_k3fix |
| 07 | ep07_k3fix_final18529 | K3fix | bd6639ec | b02cb8f6 | 0.9794 | 1 | 1 | 0.68 | frame_refs_k3fix |
| 08 | ep08_k3easy_final17496 | K3 | c8988396 | 17d421a1 | 0.9248 | 1 | 1 | 0.93 | frame_refs_k3 |
| 09 | ep09_k3fix_final39449 | K3fix | 4d962b4e | d1e88add | 0.5914 | 1 | 1 | 2.46 | frame_refs_k3fix |
| 10 | ep10_k3fix_final25319 | K3fix | 79ec18cf | ea2a605c | 0.6072 | 1 | 1 | 1.54 | frame_refs_k3fix |
| 11 | ep11_k3fix_final29175 | K3fix | 3a330ab0 | 3b834654 | 0.8826 | 1 | 1 | 1.36 | frame_refs_k3fix |
| 12 | ep12_k3fix_final23178 | K3fix | 3a25f30e | b455bb4d | 0.7245 | 1 | 1 | 0.83 | frame_refs_k3fix |
| 13 | ep13_k3fix_final39254 | K3fix | dbb526df | bc146e3e | 0.8440 | 1 | 1 | 1.16 | frame_refs_k3fix |
| 14 | ep14_k3fix_final44922 | K3fix | 0f42a389 | 3c4740ef | 0.5566 | 0 | 0 | 36.61 | frame_refs_k3fix |
| 15 | ep15_raw_final44922 | raw | db94b3d0 | d4bbb317 | 0.7017 | 1 | 1 | 1.14 | released Tier-2 |
| 16 | ep16_k3fix_final32063 | K3fix | c5571c15 | ae1c2d2b | 0.7652 | 1 | 1 | 1.59 | frame_refs_k3fix |
| 17 | ep17_k3fix_final29845 | K3fix | 64cbd5ec | dd00794e | 0.7536 | 1 | 1 | 1.93 | frame_refs_k3fix |
| 18 | ep18_k3fix_final18911 | K3fix | 6772981a | de312489 | 0.9312 | 1 | 1 | 0.68 | frame_refs_k3fix |
| 19 | ep19_raw_final44922 | raw | 651a438d | 344e6f23 | 0.1663 | 0 | 0 | 49.65 | released Tier-2 |
| 20 | ep20_k3easy_final18450 | K3 | 46dc08c1 | b0dcb320 | 0.9106 | 1 | 1 | 1.06 | frame_refs_k3 |
| 21 | ep21_k3fix_final44922 | K3fix | 86f0ccce | 0e639989 | 0.5522 | 1 | 1 | 2.09 | frame_refs_k3fix |
| 22 | ep22_raw_final44922 | raw | 402aa8bc | 0e9e237b | 0.4837 | 1 | 1 | 2.39 | released Tier-2 |
| 23 | ep23_k3easy_final19162 | K3 | 8cea19c9 | 9c3309e8 | 0.9503 | 1 | 1 | 0.78 | frame_refs_k3 |
| 24 | ep24_k3fix_final20636 | K3fix | ff795e71 | 7d1d36ec | 0.9483 | 1 | 1 | 0.65 | frame_refs_k3fix |
| 25 | ep25_k3fix_final42819 | K3fix | 80f7ffeb | f5cdaa96 | 0.0659 | 0 | 0 | 136.88 | frame_refs_k3fix |
| 26 | ep26_raw_final34948 | raw | b7c6a9af | fdb54e1a | 0.2728 | 1 | 1 | 2.57 | released Tier-2 |
| 27 | ep27_k3fix_final20863 | K3fix | ffb5aae5 | e560ebff | 0.9600 | 1 | 1 | 0.78 | frame_refs_k3fix |
| 28 | ep28_raw_final44922 | raw | edb882c7 | 57c4bbb1 | 0.0337 | 0 | 0 | 55.56 | released Tier-2 |
| 29 | ep29_k3fix_final17358 | K3fix | 52dae9a8 | 8c984002 | 0.9269 | 1 | 1 | 0.69 | frame_refs_k3fix |

The "own" columns are the official evaluator's values for the packed run, scored against the policy's own input. Each
row of the CSV carries its own input's frame references:
- K3fix or K3 rows: a byte copy of that input's npz (sha256 equal to round 1's frozen `TRACKS.sha256`);
- raw rows: the released Tier-2 track, checked equal to the packer default.

Per-episode records (checkpoint, training overrides and commit, export, rollout, frame reference):
`set_items_k3only.json`, built from `pack/k3only_closest_trial02/file.json` and `checks/check_set_d769f0c.json`.

## Trial choice

The rule:
- trial k = run k of every episode's job, out of the official 5x reruns of 2026-10-24 (round 0);
- the chosen trial is the one whose 30-episode own-input mean AUC is closest to the 5-trial mean, ties to the lower k.

`tools/k3only_choose.py` wrote `CHOICE.md` / `CHOICE.json` at 2026-11-02 15:55:03 CST and hashed them in
`CHOICE.sha256` before packing (this copy of `CHOICE.md` is edited, see "Record scope"). The 5x own-input table it read
(`inputs/own_input_evals.csv`, sha256 f3ea6a2a...) is byte-identical to round 0's `own_input_evals.csv`.

| trial | own AUC | own SP-SR | own MP-SR | own MPPE (cm) |
|---:|---:|---:|---:|---:|
| 1 | 0.6366 | 0.800 | 0.800 | 15.95 |
| **2 (chosen)** | **0.6378** | **0.800** | **0.800** | **15.66** |
| 3 | 0.6593 | 0.833 | 0.833 | 11.01 |
| 4 | 0.6373 | 0.800 | 0.800 | 15.12 |
| 5 | 0.6627 | 0.833 | 0.833 | 11.26 |
| mean | 0.6467 | 0.813 | 0.813 | 13.80 |

- Own-input composites mix references (raw, K3fix, K3), so they are not leaderboard numbers.
- ep25's K3fix policy is bimodal across reruns (own AUC 0.06 to 0.76), and trial 2 has its low mode.

## Checks (all pass)

These checks ran on 2026-11-02 against the code tree of the earlier record commit d769f0c. Its non-record code files
are byte-identical to this commit's. Its Tier-2 record folder `t2ready_20261031/` holds 124 scripts: 76 are in this
commit, byte-identical, and 48 are withheld here (`../t2r7_20261102/REDACTIONS.tsv`).
- **Configs** (`checks/check_set_d769f0c.json`, `tools/k3only_check_set.py`). For all 30 checkpoints:
  - the run's overrides compose (A);
  - the composition equals the saved `.hydra/config.yaml` key by key (B, for the 22 runs that have Hydra records);
  - the resolved composition equals the resolved config stored in the checkpoint, all keys exact, with no new keys
    (D 30/30);
  - the stored `task.parquet` and its sha256 equal the training input, and the K3/K3fix parquets are listed in
    `../t2ready_20261031/data/ours/k3fix/READY`;
  - the training commit is 61fa5d3 (W&B metadata, or the offline W&B run record plus the Hydra cwd `code/fc_r2`).
- **Input builders** (`checks/check_builders_d769f0c.tsv`): 72 of 72 files in `data/ours/{k3,k3fix,raw,gates}` and
  `train/ours` of that record matched their pins and the live source files. Of these 72, 50 are in this commit
  byte-identical, 2 are edited copies (the K3 and PREREG_v3 pre-registrations) and 20 are withheld (all listed in
  `../t2r7_20261102/REDACTIONS.tsv`).
- **Exports:** all 30 exported files are present and re-hashed equal to `export.json`. Each names its checkpoint as its
  export source and is stock-verified.
- **Earlier pack of the same trial** (`submission/pack_report.json`, `submission/validate_report.json`,
  `tools/k3only_pack.py`): kit packer `tools/pack_submission.py` (47a45895) unchanged; kit, results, frame references and
  stock runtime unchanged afterwards; the uploader's offline `validate_submission(track_2_tier2)` passed; provenance
  columns uniform (evaluator 160d24f5 / b1b3ebe3, packer 47a45895, uploader fe5cb1dc); one checkpoint digest per
  episode, equal to its chosen export.

## Re-pack with this commit's URL

`<K3_PACK>` is the `pack/k3only_closest_trial02` folder on the team server (its `results/` and `frame_refs/`, written
by `tools/k3only_pack.py prepare`); the source CSV is the earlier pack of the same trial.

    <kit python> -I -B ../t2ready_20261031/tools/repack_t2.py --kit <KIT> --dataset-root <V2D_DATA> \
        --pack-dir <K3_PACK> --source-csv <earlier pack CSV> --items set_items_k3only.json \
        --commit-url https://github.com/<owner>/<repo>/commit/<this commit> --out-dir <DIR> \
        --name T2_K3only_closest_trial02_<7 hex>.csv

## Record scope

This README is rewritten for this record from the server folder's `README.md`. `inputs/PREREG.md`,
`inputs/PREREG_PINS.md`, `inputs/PREREG_v3_k3full.md`, `inputs/manifest_stageB.jsonl` and
`inputs/stageB_resolutions.jsonl` are the same edited copies as round 0's (and `train/ours/prereg/k3full/`) in
`../t2ready_20261031/`. `CHOICE.md` has its title shortened, two clauses removed from its third line, and its ep04 note
(fifth line) corrected: `tools/k3only_choose.py` (unchanged) writes "its K3fix run never finished", but the run
finished after VERDICT.json was last written (see "Composition"). `CHOICE.sha256`, `INPUTS_SNAPSHOT.sha256` and `CHOICE.json` keep the
frozen hashes of the originals; `../t2r7_20261102/REDACTIONS.tsv` lists both values. `tools/k3only_pack.py` is the
packer of the earlier pack and hard-codes that pack's commit URL; this commit's file is re-packed with `repack_t2.py`.

- ep04 is on raw because VERDICT.json says PENDING (see "Composition").
