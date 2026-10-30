# TOPUP_AMEND1: ep28 = ep28-W3H_83000 (set `all7`) by user request (2026-10-28)

Written 2026-10-28 ~18:00 CST in `/root/autodl-tmp/haoran.zhang/fc_runs/T1improve/topup/amend1/` (below: `A`), before
anything is packed for this amendment. Hash-locked in `A/TOPUP_AMEND1.sha256`. Parent rules: `U/TOPUP_RULES.md`
(`U` = `/root/autodl-tmp/haoran.zhang/fc_runs/T1improve/topup`, sha256
`e74f598caebaece84655da9cf86feb7b47c4752eb5b91e2363e44832424fc806`, unchanged).

## 1. Authority
User, verbatim (2026-10-28 ~17:30 CST): "no I think MP-SR could be 1 right? we can test and trail submit the seed with
all 1 right for competition?"

## 2. Disclosure: post-hoc
This amendment is written **after** the 20-trial results were seen: `U/decisions.json` (8.5 decisions), `U/selection.json`
(config `final`, submittable trial 03, set MP-SR 0.9667) and the 20 already packed and scored `all7` trials in
`U/scores/all7/t01..t20/`. It is a post-hoc change made at the user's request, not a pre-registered one, and it is
reported as such next to every number it produces.

## 3. What changes
- TOPUP_RULES section 5's extra condition for the 4(c) exception item ep28alt ("ep28-W3H_83000 ... additionally needs
  MP pass >= 16 of its 20 trials") is **waived** by the user's request. This was the only condition ep28-W3H_83000
  failed (`U/decisions.json` items.ep28alt.why: "8.4(c) exception: MP pass 15/20 < 16/20 (0.8 n)").
- Therefore ep28 = **ep28-W3H_83000** (evaluated file sha256
  `497ae3b3a785fc9283fcdda2c019020608ed8c41472f4229ac780f4450081375`), all 7 changed items are kept, and the set is
  config **`all7`** (TOPUP_RULES section 4: if all 7 changes are kept, `final` = `all7`).

## 4. Reason (20-trial per-episode kit values, ep28, from `U/decisions.json` items.ep28alt)
W3H dominates the incumbent W2PD model_66000 on all three 20-trial per-episode measures, with equal SP:

| ep28, 20 trials (n = 20 / 20) | W3H 83000 (new) | W2PD 66000 (incumbent) |
|---|---:|---:|
| mean AUC | 0.815824 | 0.807408 |
| mean MPPE (cm) | 2.519 | 3.466 |
| MP pass | 15/20 | 6/20 |
| SP pass | 20/20 | 20/20 |

Under literal 8.5 W3H already wins (higher mean AUC, MPPE better rather than > 0.2 cm worse, SP pass >= incumbent,
MP pass >= incumbent); only the 16/20 exception threshold kept W2PD.

Caveat, stated plainly: W3H still misses MP on ep28 in 5 of its 20 trials (trials 01, 06, 08, 14, 20), so a set MP-SR of
1.0 is a property of the selected trial's file, not a guarantee for every evaluation of this set.

## 5. What does NOT change
- **Trials**: the 20 `all7` trials already packed and scored in `U/scores/all7/t01..t20/` (same pinned files, same
  official invocation, same seed signature). No new evaluation is run and no trial is re-run.
- **Scoring**: t1best20 scorer `fc_runs/submissions/t1_best20_20261026/t1best20_score.py` (sha256 `baddf438...`),
  `--episodes all`; set metrics = overall float32.
- **Selection** (TOPUP_RULES section 6, unchanged): statistics over the complete `all7` trials (mean, sample sd (n-1),
  min, max of set AUC, SP-SR, MP-SR, MPPE); candidates = trials whose set SP-SR and set MP-SR each equal their mode over
  the complete trials (rounded to 1e-9, a multimodal tie takes the larger value); among them minimise
  |AUC - mean| / sd_AUC + |MPPE - mean| / sd_MPPE (sd 0 term counts 0); ties -> lower trial number. Never a best-of
  trial. The user's "the seed with all 1" is met only through this mode rule (if the mode of SP-SR and MP-SR is 1.0, the
  submittable trial has SP = MP = 1); no trial is picked for being the best.
- **Report-only best trial**: highest set AUC; ties -> lower MPPE, then lower trial. Labelled REPORT ONLY; identified
  in `A/selection.json`, not packed for upload by this amendment.
- **Commit check**: the file cites
  `https://github.com/Beat-in-our-hearts/nv-v2d-challenge-codebase/commit/0c7516eda6f90deacb1536aabbce98beba5194ca`;
  `U/commit_check.json` (ok = true) already covers arm ep28-W3H and the other six arms, and
  `U/commit_remote_check.json` shows the commit on the remote.
- **Repack, re-score, validation**: the submittable trial's results are re-packed by the kit packer exactly as its
  `score.json` `pack.command`, only with the commit URL above; the re-score must equal the trial's `all7` overall and
  per-episode float32 values exactly (a mismatch is reported, never resolved by switching trials); the file is
  validated by the kit uploader's `validate_submission()` called directly (no Kaggle API) plus the provenance checks
  of `U/t1topup_validate.py` (uniform digests, commit URL, one checkpoint sha256 per episode = the set's evaluated file,
  episode 28 = `497ae3b3...`).
- **Fallback**: the earlier `final` file (ep28 = W2PD 66k), trial 03,
  `U/files/trial03_SUBMITTABLE_closest_to_mean/T1improve_tier1_trial03_SUBMITTABLE_closest_to_mean.csv` (sha256
  `93f1d61592da3788f76c133acefb477ca46e8141705b57ec7e2c14a7a5d69a85`), stays as the fallback and is not modified.
  `U/decisions.json`, `U/selection.json`, `U/files.json` and everything else under `U` outside `A` stay unchanged.
- Nothing is uploaded and nothing is pushed or committed. Upload stays the user's decision.

## 6. Outputs
`A/selection.json`, `A/files.json`,
`A/files/trialNN_SUBMITTABLE_W3H_closest_to_mean/T1improve_W3H_tier1_trialNN_SUBMITTABLE_closest_to_mean.csv`; copies
on the Mac in `~/Desktop/hpc/submissions/tier1_t1improve_w3h/` (scp from the Mac side).
