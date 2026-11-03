# TOPUP_RULES: 20-trial top-up of the changed T1improve items, set combination, scoring and file choice (2026-10-28)

Written 2026-10-28 ~15:55 CST, **before any top-up evaluation (t11-t20) exists** (`confirm/runs/<winner>/` holds only
t01-t10 and w64). Folder `/root/autodl-tmp/haoran.zhang/fc_runs/T1improve/topup` (below: `U`); `C` = `../confirm`;
`B20` = `fc_runs/submissions/t1_best20_20261026`. This file is hash-locked in `TOPUP_RULES.sha256`. It changes nothing
in `C/CONFIRM_RULES.md` (sha256 `655578198ed51bbba20f6eded5bc810fd8e20165c8a0dfc1e516c5e29483fdd4`), PREREG v2
(`e418a648...`), ADD1, ADD2 or the t1best20 PREREG / BANK_ep28_RULE. It only writes down how CONFIRM_RULES section 8
(PREREG v2 8.5-8.7) is carried out. Where this file is silent, CONFIRM_RULES and PREREG v2 govern.

## 1. Authority
User, verbatim (2026-10-28 ~14:20 CST): "ok so you can eval all current best seed 10 times and give the seed that is the
best with the submitable csv?". CONFIRM_RULES section 1 reads it; its section 8 fixes the top-up: the changed item's
t01-t10 **are trials 01-10** of its 20 and the top-up adds t11-t20 of the same pinned file. Nothing is uploaded and
nothing is pushed.

## 2. Changed items (from `C/verdicts.json` sha256 `a92ce421...`, snapshot `U/confirm_t10_snapshot/`)

| item (bar row) | episode | winner | evaluated file sha256 (EXPORT_PINS) | incumbent side (t1best20 item) |
|---|---:|---|---|---|
| ep01 | 1 | ep01-Bc_162000 | `01c0a3bd65688b05...` | ep01 |
| ep05 | 5 | ep05-B3_62000 | `ed7bca1616872a58...` | ep05 |
| ep10 | 10 | ep10-BT_58000 | `725ee5e8f8dde84f...` | ep10 |
| ep14 | 14 | ep14-R_92000 | `b01866a8f9103af1...` | ep14 |
| ep15 | 15 | ep15-B2_40000 | `8d0e8023a9a95b93...` | ep15 |
| ep19 | 19 | ep19-Bc_135000 | `07863de3bb9b8282...` | ep19 |
| ep28alt | 28 | ep28-W3H_83000 (8.4(c) exception) | `497ae3b3a785fc92...` | ep28alt = W2PD model_66000 (BANK_ep28_RULE outcome) |

Unchanged items (ep21, ep25 and the other 21 episodes) keep their t1best20 file; ep28's incumbent in the pinned set is
ep28alt (W2PD 66k), never ep28 B 58k.

## 3. The top-up evaluations
- Exactly: `cd C; PYTHONDONTWRITEBYTECODE=1 TZ=Asia/Shanghai setsid nohup <flashchord python> -B C/t1confirm_driver.py run
  --trials 20 --only <t11..t20 of the 7 winners> --no-screens --gpus 0,1,2,3,5,6,7 --max-parallel 4 --lane-max 2
  --cooldown 35`, the driver of the confirmation (sha256 `f742f139...`, unchanged; it re-checks the CONFIRM_RULES and
  EXPORT_PINS locks). Same pinned files, same official invocation, same checks, same lanes / locks / cooldown, never GPU
  4, never MPS, never a signal. 70 evaluations, trial-major. A completed evaluation is never re-run; a technical failure
  is retried by the driver (<= 3 tries, 5 min apart).
- Side effect, disclosed: the driver's own summary runs with `--trials 20` and rewrites `C/verdicts.json`,
  `SUMMARY.md`, `candidates_summary.csv`, `screens.csv`, `per_eval.csv`, `seeds.tsv` (non-winners then show PENDING).
  Before restoring, those 20-trial outputs are copied to `U/driver_summary_t20/`. Then `t1confirm_driver.py summary
  --trials 10` restores the 10-trial confirmation view in `C`; the restored `verdicts.json` must equal the snapshot
  except `generated_at` (checked and reported). The snapshot (`U/confirm_t10_snapshot.sha256`) is the record of the
  confirmation verdicts.

## 4. Trials and scoring
- Trial k (k = 01..20) of a set = for each of the 30 episodes the results JSON + parquet of trial k: changed item ->
  `C/runs/<winner>/t<k>/run.json` (ok); unchanged item -> `B20/runs/<item>/t<k>/run.json` (ok), ep28 -> `B20/runs/ep28alt`.
  The 30 file pairs are copied to `U/scores/<config>/t<k>/results/`, packed by the kit packer (`tools/pack_submission.py`
  sha256 `47a45895...`, `--track track_2_tier1`, placeholder commit URL as t1best20), and scored by the t1best20 scorer
  `B20/t1best20_score.py` (sha256 `baddf438...`, kit metric_code AUC / SP-SR / MP-SR / MPPE, `--episodes all`).
  **Set metrics = overall float32**; per-episode float32 values are kept. A trial with any missing / failed item is
  incomplete, excluded from all statistics and reported with n.
- Configs: `all7` = every changed item's winner (ep28 = W3H 83000); `pinned` = the incumbent set (t1best20 + ep28alt),
  packed the same way as the baseline; `final` = the set after section 5 (if all 7 changes are kept, `final` = `all7`
  and is not packed again).
- Checks: an unchanged item's per-episode kit values in `all7` must equal its t1best20 per-episode values of the same
  trial (same parquet; `B20/scores/t<k>/metric_code.json`, ep28alt from `alt28/metric_code.json`); differences are
  reported, never fixed by choosing trials.

## 5. 8.5 decision per changed item (20 trials, kit metric_code per-episode float32 values, both sides alike)
- New side: the item's per-episode values in `all7` trials 01-20. Incumbent side: the same episode's t1best20 values
  trials 01-20 (ep28alt: `alt28`). Pass = per-episode SP-SR / MP-SR value >= 0.5 (it is 0 or 1).
- Literal 8.5: the side with the higher 20-trial mean AUC is chosen, unless its MPPE mean is more than 0.2 cm worse than
  the other side's, in which case the other side is chosen. Then the new item is kept only if additionally its SP pass
  count >= the incumbent's and its MP pass count >= the incumbent's; ep28-W3H_83000 (admitted by the 4(c) exception)
  additionally needs MP pass >= 16 of its 20 trials. Otherwise the incumbent stays. Exact ties in mean AUC go to the
  incumbent. The evaluator-JSON pass counts are reported next to the kit counts but do not decide.
- If fewer than 20 of an item's trials are ok, the test runs over its ok trials and the report says n (ep28 then needs
  MP >= 0.8 n).

## 6. Files (8.6) and report
- Over the complete trials of `final`: mean, sample sd (n-1), min, max of set AUC, SP-SR, MP-SR, MPPE; mean includes
  collapses.
- **Submittable = the trial closest to the set mean** (8.6): candidates = trials whose set SP-SR and set MP-SR each
  equal their mode over the complete trials (values compared rounded to 1e-9; a multimodal tie takes the larger
  value); among them minimise |AUC - mean| / sd_AUC + |MPPE - mean| / sd_MPPE (a term with sd 0 counts 0); ties ->
  lower trial number. Never a best-of trial.
- **Report-only best trial** = highest set AUC; ties -> lower MPPE, then lower trial number. Labelled REPORT ONLY.
- Both trials are re-packed with `--code-commit https://github.com/Beat-in-our-hearts/nv-v2d-challenge-codebase/commit/0c7516eda6f90deacb1536aabbce98beba5194ca`
  only after checking that this commit is on the remote and contains the training and export code of every item of the
  `final` set (each kept arm's tree code: b39b69b, 61fa5d3, 7fbc227; per-file comparison of the code the arms ran).
  If that check fails, the files are not packed with that URL and the gap is reported. Re-scored by metric_code: the
  re-score must equal the trial's `final` value (the commit URL does not touch poses); a mismatch is reported, never
  resolved by switching trials. Validated by the kit uploader's `validate_submission()` (called directly, no Kaggle
  API), with the provenance checks (uniform digests, one checkpoint sha256 per episode = the set's evaluated file).
- Seeds: every evaluation's seed signature is recorded (`seeds.tsv` rows from run.json); all trials share the file
  signatures, so "the best seed" is reported as the best trial.
- Output copies: `~/Desktop/hpc/submissions/tier1_t1improve/` on the Mac (scp from the Mac side; the Mac only stores).

## 7. Scripts
`U/t1topup.py` (combine, pack, score, decide, select, report) is written after this file and before any top-up
result is read; its sha256 is recorded in `U/TOPUP_SCRIPT.sha256` before its first run. It reads `C` and `B20` only
and writes only under `U`.
