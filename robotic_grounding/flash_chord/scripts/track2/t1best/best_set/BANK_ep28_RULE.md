# BANK_ep28_RULE: which squeegee (ep28) checkpoint the Tier-1 submission set uses

Written 2026-10-26 20:14:21 CST, while the 20-trial official evaluation in this folder is still running (driver pid 579810,
heartbeat phase "running" at 2026-10-26 20:14:09). **4 of 20 trials were complete when this
was written** (trial status below). This is a new file. It does not edit or replace PREREG.md (sha256 `f165ccd99a306d7acd1cdf3edbcbd0a9ebc6c16f0c6eb3926c20f7604fcc3428`),
PREREG.sha256, manifest_best.json (`e07c5bf56e76cdc743140d0d3bac8d0f31873f25272d75ce7efb2fffa3378c84`), items.json (`07b4d627cf0d303473612f6be466e930e256cb8f7a1b79476ea94e5c44a6786b`) or anything else the driver reads.

Scope: Tier 1 (clean input) only. Choosing a Tier-1 checkpoint by clean official scores is allowed. Nothing here may be
used for any Tier-2 or Tier-3 decision.

## 1. Request
The user, 2026-10-26, shortly before this file was written. The message replied to the assistant's note: W2PD
model_66000 had failed MP-SR in trials 1-3. The options offered were to "Bank W2PD anyway" for the small AUC gain,
which needs commit 7fbc227 merged into the submission commit, or to "Try another fix (wave 3)". The user wrote:

> "yes I think you should right? also with double checking zuxing best results at this time?"

This is read as: bank W2PD model_66000 instead of B model_58000 for ep28 in the Tier-1 set. That choice is subject to
the 20-trial check in section 2, so it is confirmed against all 20 trials rather than the first few. The second part of
the message, checking Zuxing's current best results, is handled separately and does not enter this rule.

## 2. Rule (pre-declared, verbatim)
> ep28 in the Tier-1 submission set uses W2PD model_66000 (ep28alt) iff, over the 20 completed trials, its mean packer
> AUC >= B 58k's mean AND its MP-SR pass count >= B's AND its SP-SR pass count >= B's; otherwise B 58k. The submission
> file is then the trial closest to the set mean computed with that choice (never best-of-20).

## 3. Definitions (fixed now, before the remaining trials)
- **B** = item `ep28`: T1fix Stage-2 ep28-B `model_58000`, training checkpoint sha256
  `04f42d00f7af146ce8c98a1250e476b77c5e07244d274a55e62484aa53bf3d6b`, evaluated file
  `exports/ep28/try1/model_58000.safetensors` sha256 `3d47acf3fddf46326e07e7c202be8938ec39daeb9a2a7acd9952e2b550f0bc08`
  (exported by code/nv-v2d_runin @ b39b69b).
- **W2PD** = item `ep28alt`: wave-2 W2PD `model_66000`, training checkpoint sha256
  `f8ea7c496a4b6a300157a01ba034f5001165c09f9600b98abf463511d6ba7507`, evaluated file
  `exports/ep28alt/try1/model_66000.safetensors` sha256 `2bddd57d8989c0703ef1e6f0833284550ab201255d827b23b40d95da8cf01fd8`
  (exported by code/nv-v2d_ep28w2 @ 7fbc227, branch dev.haoran.ep28w2, local only).
- **Data:** `per_trial.csv` after the driver has finished (heartbeat phase no longer "running").
  - The counted trials are the **paired trials**: trials whose 30 set items are all ok (so ep28 B is scored) and whose
    ep28alt status is ok with a kit score.
  - A full run gives trials 01-20. If fewer than 20 trials are paired at the end, because of failures or STOP, the rule
    applies unchanged over the paired trials, and the report states n.
- **Mean packer AUC:** the mean over paired trials of `ep28B_auc` and of `ep28alt_auc`.
  - These are the kit metric_code AUC values (float32 reference) of the kit-packed CSV. ep28alt is packed alone through
    the restricted tracks.json (PREREG section 6).
  - Compare the values unrounded. A tie counts for W2PD, since the test is >=.
- **MP-SR / SP-SR pass counts:** the number of paired trials with `ep28*_mp_sr == 1` or `ep28*_sp_sr == 1`.
- **Choice:** W2PD only if all three conditions hold; otherwise B 58k. Nothing else enters: no MPPE, no prior
  evaluations, no single trial.
- **Submission file:** "the trial closest to the set mean computed with that choice".
  - Candidate trials: trials whose 30 set items are all ok, and, if W2PD was chosen, whose ep28alt is ok too.
  - Each candidate's set AUC with the choice is `auc` (B) or `hypothetical_set_auc_with_alt` (W2PD,
    = auc + (alt - B) / 30).
  - Take the mean over the candidates. Pick the one trial whose value is closest to that mean; a tie goes to the lower
    trial number.
  - The file is that single trial's 30 results. If W2PD was chosen, ep28 is replaced by the same trial's ep28alt
    result. No rows are mixed across trials, and the trial is never picked as best-of-20 or as the highest score.
  - This narrows PREREG section 7: of its candidate rules, (i) chose per episode and (ii) took trial 01. A whole single
    trial keeps every row from one evaluation run.
  - Before any other use, re-pack the chosen file with the kit packer and re-score it with metric_code. The re-score
    must equal the per_trial.csv value. If it does not, report the mismatch and do not switch to another trial.

## 4. What this rule does not do
- It authorizes no packing with a real commit URL, no push and no upload. Those still need the user's explicit
  approval, a rules re-check (PREREG section 7), and a pushed commit that is cited and contains all the training and
  export code:
  - 7fbc227 if W2PD is chosen;
  - Zuxing's bc3b1bd for ep04 cont4, with his OK;
  - the T1fix override configs.
- It adds no trials and re-runs nothing because of a score. The driver, items and PREREG are untouched.

## 5. Trial status at writing (2026-10-26 20:14:21 CST)
Complete trials: 4; paired (ep28 B and ep28alt both scored): 4. In progress: trial 05 running (24/30 set items ok).
Driver heartbeat: {"ok": 152, "failed": 0, "pending": 468}.

| trial | B 58k AUC | B SP | B MP | B MPPE cm | W2PD 66k AUC | W2PD SP | W2PD MP | W2PD MPPE cm | set AUC with B | set AUC with W2PD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 0.7873 | 1 | 0 | 4.55 | 0.8022 | 1 | 0 | 4.19 | 0.9246 | 0.9251 |
| 02 | 0.7877 | 1 | 0 | 4.36 | 0.8038 | 1 | 0 | 4.40 | 0.9255 | 0.9260 |
| 03 | 0.7981 | 1 | 0 | 4.48 | 0.8043 | 1 | 0 | 4.20 | 0.9260 | 0.9262 |
| 04 | 0.7898 | 1 | 0 | 4.46 | 0.8063 | 1 | 0 | 3.21 | 0.9262 | 0.9268 |

Over these 4 paired trials: mean AUC B 0.7907 vs W2PD 0.8041; SP-SR passes B 4/4 vs W2PD 4/4; MP-SR passes B 0/4 vs W2PD 0/4.
Earlier evaluations, other exports of the same training checkpoints (manifest_best.json / PREREG section 2):
- B: 3 evaluations, mean AUC 0.7949, MP-SR 0/3.
- W2PD: 6 evaluations, mean AUC 0.8074, MP-SR 3/6.

Disclosure: this rule was written after the 4 paired trials above had been seen. W2PD's MP-SR sits close to the
3 cm object-position limit, so its pass count over 20 trials is not yet known.
