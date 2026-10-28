# Track 2 Tier 1: T1improve wave (2026-10-27/28): arms, confirmation and the updated best set

This folder records how six items of the Tier-1 best set (`../best_set/`, `../configs/items/`) were replaced by
checkpoints of the T1improve training wave, and how the submittable file was chosen. It adds files only. The
best-set configs, item files and tools in the parent folder are unchanged; they still describe the 2026-10-26 set.

Training and export code: the arms ran at b39b69b (rolling start), 7fbc227 (squeegee options) and fc_r2 61fa5d3
(flash_chord files = f2923d61817e). All three are ancestors of 0c7516e. `topup/commit_check.json` records the check
that this branch at 0c7516e contains that code:
- each arm's working tree equals its commit file for file (LFS assets by oid; no untracked `.py` files);
- every chosen arm's saved Hydra overrides compose, with this tree's configs, to its saved `config.yaml`, with no
  changed, new or missing keys;
- this tree's exporter equals 7fbc227's.

The arms' hyperparameter changes are Hydra overrides only, listed in `arms/<arm>/overrides.yaml`.

## What changed in the set (episode: incumbent -> new)

| episode | object | incumbent (t1best set) | new item (arm @ iteration, tree) | evaluated file sha256 | 20-trial kit AUC new / incumbent | MPPE cm new / incumbent | SP, MP pass new / incumbent |
|---:|---|---|---|---|---|---|---|
| 1 | vacuum | T1fix B 140000 | ep01-Bc @ 162000 (b39b69b) | `01c0a3bd65688b05...` | 0.7967 / 0.7582 | 1.96 / 3.67 | 20, 20 / 19, 19 |
| 5 | big_black_box | C3r2_runin 42500 | ep05-B3 @ 62000 (b39b69b) | `ed7bca1616872a58...` | 0.9630 / 0.9473 | 0.75 / 0.79 | 20, 20 / 20, 20 |
| 10 | white_desk | T1fix B 36000 | ep10-BT @ 58000 (b39b69b) | `725ee5e8f8dde84f...` | 0.8951 / 0.8693 | 1.15 / 1.31 | 20, 20 / 20, 20 |
| 14 | round_table | T1fix B 78000 | ep14-R @ 92000 (b39b69b) | `b01866a8f9103af1...` | 0.8343 / 0.8270 | 1.46 / 1.50 | 20, 20 / 20, 20 |
| 15 | wet_floor_sign | C3r2_wide 20974 | ep15-B2 @ 40000 (fc_r2 61fa5d3) | `8d0e8023a9a95b93...` | 0.9859 / 0.9376 | 0.65 / 0.88 | 20, 20 / 20, 20 |
| 19 | cane | T1fix B 108000 | ep19-Bc @ 135000 (b39b69b) | `07863de3bb9b8282...` | 0.9228 / 0.8957 | 0.99 / 1.02 | 20, 20 / 20, 20 |

**Not changed:**
- **ep28 stays W2PD 66000.** ep28-W3H @ 83000 had AUC 0.8158 vs 0.8074 and MPPE 2.52 vs 3.47 cm. It entered only
  through the PREREG v2 8.4(c) exception, which requires MP-SR to pass in at least 16 of 20 trials. It passed in 15.
- **ep21 and ep25.** Every candidate failed the MPPE cap (8.4 d).

`set_manifest.tsv` lists all 30 episodes of the new set.

## Protocol (pre-registered, hash-locked before each step)
- `prereg/`: PREREG_t1improve.md v2 (e418a648; section 8 is the selection protocol), ADD1 and ADD2.
- `confirm/CONFIRM_RULES.md` (65557819). Every candidate was exported once at 1e-4 (`EXPORT_PINS.md`) and run
  through 10 fresh official evaluations, as the user asked. The ep01, ep14, ep21 and ep28 candidates also got a
  world_count=64 collapse screen. Results: `confirm/SUMMARY.md`, `verdicts.json`, `candidates_summary.csv`, `screens.csv`.
- `topup/TOPUP_RULES.md` (e74f598c) covers the 20-trial top-up of the changed items:
  - t11-t20 of the same pinned files, with the same driver and invocation. A changed item's t01-t10 count as trials
    01-10. Disclosed: those trials also chose the winner.
  - Trial k of the set = the t1best20 trial k of every unchanged item plus trial k of every changed item.
  - Each trial is packed with the kit packer and scored with the kit metric_code (float32 reference).
  - 8.5 decision per item, then the 8.6 choice.
- 70 top-up evaluations, all ok. Kit and stock runtime unchanged.

## 20 trials of the new set (kit packer + metric_code)

| | AUC | SP-SR | MP-SR | MPPE cm |
|---|---:|---:|---:|---:|
| new set, mean of 20 (sd) | 0.93085 (0.0033) | 0.9983 | 0.9750 | 1.004 (0.286) |
| new set, min / max | 0.91720 (t17) / 0.93266 (t08) | 0.967 / 1 | 0.933 / 1 | 0.898 / 2.216 |
| incumbent set (t1best20 + W2PD), mean of 20 (sd) | 0.92543 (0.0050) | 0.9967 | 0.9733 | 1.078 (0.366) |
| **submittable: trial 03, closest to the mean (8.6)** | **0.93095** | **1.000** | **0.967** | **0.973** |
| report only: trial 08, best trial | 0.93266 | 1.000 | 0.967 | 0.920 |

- Trial 17 contains the incumbent ep03 mug collapse (AUC 0.527). It is kept in the mean.
- MP-SR 0.967 in most trials is the W2PD squeegee (episode 28), which passes MP in 6 of 20.

### Seeds
The official evaluator has no selectable seed: the harness forces `env.reset.seed = 42`, and the other seeds come
from each evaluated file's config. All 600 (trial, episode) evaluations of the new set share one seed signature,
`delay=0(+g0..4,0-2) reset=42 scale_seed=0(1.0/1.0) wrench=0 vertex=0/0 ppo=42`. The spread between trials is GPU
floating-point non-determinism, so "the best seed" is reported as the best trial (08), labelled report only.

### Files
Both trials were re-packed with the code commit URL
`https://github.com/Beat-in-our-hearts/nv-v2d-challenge-codebase/commit/0c7516eda6f90deacb1536aabbce98beba5194ca`.
Each re-score equals its trial value exactly. Both pass the kit uploader's `validate_submission()` and the provenance
checks: 22,989 rows, uniform evaluator and packer digests, and one checkpoint per episode equal to the set's
evaluated file.

| file | sha256 |
|---|---|
| `T1improve_tier1_trial03_SUBMITTABLE_closest_to_mean.csv` | `93f1d61592da3788f76c133acefb477ca46e8141705b57ec7e2c14a7a5d69a85` |
| `T1improve_tier1_trial08_REPORT_ONLY_best_trial.csv` | `d97ffbabbb51ff31f73f717fb73f4ed01afdba2ebae3e161182880e8437e8770` |

Nothing was uploaded. Records: `topup/per_trial.csv`, `selection.json`, `decisions.json`, `files.json`, `seeds.tsv`.

## Still training at the time of the confirmation (a later pass under the same rules)
ep14-Rs2, ep19-B5, ep10-BTs2 and ep05-P30.
