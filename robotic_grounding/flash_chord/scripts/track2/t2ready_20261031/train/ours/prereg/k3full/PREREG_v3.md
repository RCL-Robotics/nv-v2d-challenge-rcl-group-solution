# Full K3fix arm (arm folder C3r2_tier2_k3fix, driver k3full): pre-registration v3

Written 2026-10-22 ~07:15 CST, before any training of this arm's 17 jobs (no `tmp_analysis/tier2/k3full/pids.tsv`, no
run dir of any of them). The sha256 of this file is in `PREREG_v3.sha256`. `scripts/k3full_helper.py verdict` refuses
to run if it does not match, and the driver's preflight refuses to launch if any file pinned at the end changed.

Authorization: user, 2026-10-22, "so you can train on more tasks right?", after the K3 pilot results: extend the
K3fix training to more episodes. The launch itself is made by the main session after review.

## Relation to PREREG_v2 (K3fix staged pilot, sha256 98dd6bf1)

This is the "own pre-registration" of the full 30-episode K3fix arm that PREREG_v2's next-step section asks for.
PREREG_v2 stays in force, unchanged, for the pilot's own verdicts (Stages 1, 2 and 3).

- **Timing.** PREREG_v2 planned the full arm after a Stage-2 PASS. The user decided to extend now, while Stage 1 is still
  training and Stages 2 and 3 have not been launched. This is a user decision, disclosed below.
- **ep17, ep09 (Stage 2) and ep04 (Stage 3)** are launched by this arm's driver under their pilot names
  (`epNN_k3fix_s42`, same argv, same input). At launch their line is also appended to
  `tmp_analysis/tier2/k3fix_pilot/pids.tsv`. The k3fix driver then watches them as its own jobs, runs their
  post-training step and judges them by PREREG_v2's Stage-2 and Stage-3 rules, unchanged.
- **ep03** follows PREREG_v2: it uses K3fix in this arm only if PREREG_v2's ep03 result is PASS.
- **ep04.** PREREG_v2 left ep04's input in a full arm to a user decision after the Stage-3 report. v3 replaces that with
  the rule below, fixed now, before any ep04 K3fix training. The Stage-3 report stays report-only.

## Composition

Per episode, its K3fix run where gates G1-G4 pass and the per-episode rule below holds, else the raw C3r2_tier2 run. The
composition, the checkpoints, the order and the jobs are all fixed by the rules below.

- Reported by the verdict step: own-input AUC and SP-SR / MP-SR per episode vs the raw run's own-input evaluation.
  Own-input references differ between inputs, so these are not leaderboard numbers.
- One seed (42) per episode, as the raw arm.

## Input rule (fixed; the gate-4 rule PREREG_v2 asked for)

`scripts/k3full_gates.py gates` -> `build/gates_all30.json` (table in `build/gates_all30.txt`). It uses the pilot's
gate code (`scripts/k3fix_build.py gate_work`, sha256 364d27ed, imported read-only) on the K3fix track that
`data/human_motion_data_tier2_k3fix` was built from (READY sha256 1233398f). The constants come from Tier-2 data:

- **G1 envelope:** no visible frame with |K3fix - raw| > 49.23 deg or > 4.62 cm.
- **G2 penetration:** no visible frame where the K3fix object is > 4.62 cm deeper in the support / floor than raw.
- **G3 rest:** no flagged rest run with zero angular velocity forced.
- **G4 contact:** among raw-contact frames (< 3 cm), the share where K3fix is > 3 cm farther must be <= 5%. Evaluated
  with >= 50 contact frames; with fewer the episode is exempt and passes.

An episode PASSes iff G1-G4 all pass; it is then a K3fix candidate. An episode that fails uses the raw run.

The checks hold (71/71):
- the gated track is the data root's (npz sha256 = tracks.json = data_build.json = READY);
- every K3fix parquet = READY = data_build;
- the K3fix results for ep03/04/09/17 reproduce `gates.json`, and for ep07/18 `gates_v2.json`;
- frozen-K3 G4 reproduces the reviewer's all-30 table for every episode.

| result | episodes |
|---|---|
| PASS (23) | 01 02 03 04 06 07 08 09 10 11 12 13 14 16 17 18 20 21 23 24 25 27 29 |
| FAIL -> raw (7) | 00 (G4 0.096), 05 (G4 0.059), 15 (G4 0.209), 19 (G4 0.341), 22 (G4 0.191), 26 (G4 0.073), 28 (G2 194 frames, max 13.7 cm; G4 0.156) |
| G4 exempt (< 50 contact frames) | 01 (7 frames), 29 (37 frames) |

G1 and G3 pass everywhere. G2 fails only on ep28.

## Per-episode rule (fixed)

An episode uses its K3fix run iff ALL of:
1. it passes the gates;
2. its designated K3fix run trained and finished: `final.json` written by the run (help-zero tail or max_iterations)
   with its checkpoint present;
3. that checkpoint was exported and evaluated. The post-training status is `done` or `done_score_failed`,
   `eval_status` is `evaluated`, the evaluated iteration is `final.json`'s, the export is stock-verified with max
   relative difference <= 2e-4, and the evaluation reference is the training parquet;
4. for ep03 only: PREREG_v2's ep03 result is PASS.

Otherwise the episode uses the raw C3r2_tier2 run's `final.json` checkpoint. The raw run is used for a crash without
`final.json` (never relaunched automatically), an export refused at 2e-4, or an evaluation that failed twice. Own-input
SP-SR / MP-SR are reported, not required.

Designated K3fix runs:
- ep03, ep07, ep18: `fc_runs/C3r2_tier2_k3fix/epNN_k3fix_s42` (pilot Stage 1). Post-training step:
  `k3fix_pilot/eval/<job>/post.json`.
- ep08, ep20, ep23: `fc_runs/C3r2_tier2_k3easy/epNN` (k3easy, input `data/human_motion_data_tier2_k3`, byte-identical
  to the K3fix root's; checked). Post-training step: `k3easy/eval/epNN/post.json`.
- The 17 jobs below: `fc_runs/C3r2_tier2_k3fix/epNN_k3fix_s42`. Post-training step: `k3full/eval/<job>/post.json`, or
  `k3fix_pilot/eval/<job>/post.json` for ep17/09/04.

The seed-7 raw control `ep03_raw_s7` is never part of the arm.

## Jobs and launch order (fixed; `jobs.json`)

Jobs = gate PASS minus the six episodes already on the K3fix input (03, 07, 18, 08, 20, 23). The order is easy-first
by the RAW Tier-2 run's own training: its `final.json` `help_zero_since` (= `early_table.csv`
`t2train_help_zero_since`), ascending, ties by episode index, ep04 last.

| # | job | object | raw help zero (final) | scene.support | input |
|---|---|---|---|---|---|
| 1 | ep11_k3fix_s42 | light_blue_book | 23065 (28925) | default | = K3 |
| 2 | ep16_k3fix_s42 | spray_bottle | 25408 (31268) | default | = K3 |
| 3 | ep17_k3fix_s42 | traffic_cone | 25947 (31807) | default | K3fix-changed (pilot Stage 2) |
| 4 | ep06_k3fix_s42 | monitor | 28034 (33894) | default | = K3 |
| 5 | ep10_k3fix_s42 | white_desk | 28407 (34267) | false | = K3 |
| 6 | ep02_k3fix_s42 | blue_easter_basket | 31549 (37409) | default | = K3 |
| 7 | ep12_k3fix_s42 | black_platform | 31715 (37575) | false | = K3 |
| 8 | ep09_k3fix_s42 | big_red_bowl | 33468 (39328) | default | K3fix-changed (pilot Stage 2) |
| 9 | ep01_k3fix_s42 | vacuum | 39062 (44922) | default | = K3 |
| 10 | ep13_k3fix_s42 | snack_box | 39062 (44922) | default | = K3 |
| 11 | ep14_k3fix_s42 | round_table | 39062 (44922) | false | = K3 |
| 12 | ep21_k3fix_s42 | tall_bar_stool | 39062 (44922) | false | = K3 |
| 13 | ep24_k3fix_s42 | watering_can | 39062 (44922) | false | = K3 |
| 14 | ep25_k3fix_s42 | cyan_water_bottle | 39062 (44922) | default | = K3 |
| 15 | ep27_k3fix_s42 | woven_basket | 39062 (44922) | default | = K3 |
| 16 | ep29_k3fix_s42 | rolling_luggage | 39062 (44922) | false | = K3 |
| 17 | ep04_k3fix_s42 | baseball_bat | 39062 (44922) | false | K3fix-changed (pilot Stage 3) |

## Recipe (fixed)

Exactly the raw C3r2_tier2 run of each episode:
- `code/fc_r2` @61fa5d3, code-tree digest = the pilot's `build.json`;
- experiment `g1_recon_body_coord_rsi_r2`, `reset.seed=42`, `training.max_iterations=45000`;
- experiment defaults: RSI deadline 35156, zero_ramp 3906, help-zero early stop 5860 (NOT the stretched curriculum);
- `scene.support` as the raw run.

`cfgcheck.json`, the pilot's cfgcheck code, checks all 17 jobs. Composing each raw run's recorded overrides
(`code/fc_r2/logs/rl/sharpa/<date>_epNN_<object>/.hydra`) reproduces its saved `config.yaml` (222 keys). Each job differs
from it only in `task.parquet` (data root `data/human_motion_data_tier2_k3fix`), `output_dir`, `logging.mode` (offline),
`logging.experiment` and `logging.local_jsonl`.

The environment is as the pilot's (fc_env.sh, offline W&B), with markers `K3FIX=1 K3FIX_JOB=<job> K3FULL=1`.

## Checkpoint rule (fixed)

The submission checkpoint of a K3fix episode is the one `final.json` names (the C3r2 rule). The running t2watch
watcher's best-by-own-input checkpoint (`tmp_analysis/tier2/t2watch/best/C3r2_tier2_k3fix/<job>/`) is kept and reported
as secondary information only; it never replaces `final.json` in this arm.

## Post-training step (fixed; as the pilot)

After `final.json`, export at 2e-4, then ONE official evaluation of `final.json`'s checkpoint against its own input:
`eval_all.py --checkpoint-from-final`, world count 1, on the training GPU, under the evaluation lane locks.

A technical failure is retried once after 10 min. For the 14 new jobs this is k3fix_helper's `post` code on this arm's
records. For ep17/09/04 it is the pilot's own `post`: run by the k3fix driver while it is alive, otherwise by this
helper calling it.

## Scheduling (fixed; changes neither the recipe nor any rule)

- **Order.** Strictly in the order above, one job at a time, >= 180 s apart. Each start happens under the shared
  GPU-choice locks (`ch_runs/{seedstudy,f0share05,filler}/.gpulock`, opened read-only), and the plan is re-checked
  under them.
- **Fill rule.**
  - Never GPU 4; no MPS.
  - < 3 trainings of anybody on the GPU before the start.
  - >= 16.5 GB + 20 GB free, plus the ramp debt of every training younger than 15 min.
  - 2 free physical cores (+HT) of the GPU's own NUMA node (`cpu_slice`, eval lanes excluded, never borrowed).
  - NUMA-1 GPUs 3/5/6/7 first, then 0/1/2; fewest trainings, then most free memory.
- **Tier-3 yield.** No start while the t3opus driver runs in `run` mode, its heartbeat is < 30 min old, and a clip of
  its wave(s) is still FRESH / NEEDS_RETARGET / NEEDS_RCHECK / RETARGETING. Now that means wave 1: ep08, ep23, ep22. If
  its state cannot be read, this arm yields.
- **Tier-1 equal share.** No start if, after it, the feasible slots would fall below the T1fix launcher's reserve
  min(T1fix arms not launched, max(0, ceil(S/2) - T1fix running)), while its `auto` loop is alive. Now this reserve is 0:
  all 7 Stage-1 arms are launched.
- **Crash.** A trainer that dies without `final.json` gets `k3full/CRASHED_<job>` and is not relaunched; the raw run
  is used for that episode. A relaunch is the user's decision and is disclosed.

## Disclosures

- **Job set and order.** This arm's job set comes from the gates, and its order from the raw Tier-2 run's
  own training.
- **Extension decision.** The user decided to extend now after interim pilot results. It chose no episode, input,
  checkpoint or order; those are fixed above by the rules.
- **Gates.** G1-G3 were computed for all 30 episodes only now; before, they existed for the 6 pilot clips (G4 had the
  reviewer's all-30 table on frozen K3). Code and thresholds are unchanged and the replication is exact. Episodes near a
  limit:
  - ep08 G4 share 0.046 and ep18 0.038, against the 0.05 limit;
  - ep01 and ep29 pass G4 only by exemption (7 and 37 contact frames);
  - ep11's G2 max excess is 3.10 cm and ep01's 3.79 cm, against the 4.62 cm limit.
- **Raw references.** One seed per episode, and the raw references are single seed-42 runs.
- **Logging.** The raw arm logged online; this arm logs offline.
- **Monitor label.** The monitor labels these run dirs "Tier 2 · K3fix test".

- **Pilot FAIL handling:** a FAIL of the PREREG_v2 pilot (Stage 1 or Stage 2 arm verdict) does not stop this arm automatically; it is reported to the user and this arm is paused (STOP) until the user decides.
- **Pre-launch amendment (2026-10-22, before any k3full launch):** driver CPU-exclusion reset after each launch hold (review F1) and disclosure bullets (review P2/P3). No rule, gate, order, recipe or checkpoint rule changed.

## Files and pins (checked by `k3full_helper.py check`)

pin jobs.json d150ca5a06fa22d0cc124cf6681646883f5c41880634ac8873d9332505eea986
pin gates_all30.json 3a741669592cf62792ef97f6bd0d95a4b0a0f03b25967c7b7ec680dd613ac4ab
pin cfgcheck.json 6ee226441c815d037b5b486f035a038769b05135a7a1b1e2689ea49106484f5b
pin k3fix_helper.py d563555aa8634da2d90ab4375bfe3fa1539179c2b6c45d4e0e96f5c1862401b9
pin k3fix_build.py 364d27ed1b0e0a250456ff2fa42b130c7f73bfb4755f88c75237dc79148dff7b
pin k3full_gates.py e99d3321cf99dc9cbf210ec221495ada69101f71e5c784d0c2f96fdd0f11116a
pin k3full_helper.py 1d59f69237d1ee2b889ce020d1301ab6a2084c2a111026c81b707cdd20df342a
pin k3full_driver.sh 658d7a7732afb1855343a4358eb1fab478fa4e8d1b7e8841309b25a2ac8afbed

- **Driver:** `scripts/k3full_driver.sh run|adopt|dry`.
- **Helper:** `scripts/k3full_helper.py`.
- **Gates:** `scripts/k3full_gates.py`.
- **Run dirs:** `fc_runs/C3r2_tier2_k3fix/<job>`, with logs in `fc_runs/C3r2_tier2_k3fix/logs/`.
- **Records:** `tmp_analysis/tier2/k3full/`:
  - `pids.tsv`, `driver.log`, `heartbeat.log`;
  - `eval/<job>/`;
  - `VERDICT.{json,txt}`;
  - `STOP`.
