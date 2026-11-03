# t2_consensus_r9_20261103: DRYRUN_R9 (build steps B2-B3 of PREREG_R9 10; decides nothing)

**When.** 2026-11-03, 22:25-22:30 CST, with the final scripts (they hard-code `PREREG_R9.md`
13c3b2a61e39cc761633283ae42bf94683df8d1ffd701d1076e3d61212a4c2a0). This is before the round-9 pool freeze and before any
round-9 export, evaluation, score or selection. Both check reports were available (ZX_G2_30_CHECK_20261103.md from
21:55 CST, ZX_K3FIX_18_AND_RULES_CHECK_20261103.md from 22:21 CST); neither reports a selection rule outside D2's
condition, so D2 (a) applies to both folders (`REPORT_MODE` = files for both).

**Snapshot.** The pool dry run used "now" (2026-11-03 22:27:36 CST) as its snapshot. The real T_c9 is the start of the
freeze, which `r9_driver.py` runs at once when it is launched (D3). The frozen pool differs from this dry run only if a
file changes between the two times.

## Results of the build checks

| check | output | result |
|---|---|---|
| B2 records | `inputs_r9/R8_RECORDS_R9.json` (d9e63f54...), `dryrun/b2_r8records.log` | round 8's and round 7's chains verified (0 fatal; SELECTION_R8.sha256 22 lines, SELECTION_R7.sha256 24 lines); 30 incumbents (29 kept round-7 picks with their round-7 phase-F runs, ep13 the round-8 replacement with its round-8 phase-F runs), 15 barred files, 89 round-8 + 399 round-7 phase-S candidate records; equal to the development build apart from its time stamp and script sha256 |
| B2 selftest | `selftest_r9/selftest_r9.json`, `dryrun/b2_selftest.log` | 16/16 PASS |
| B2 mock | `dryrun/r9_mock_select.py`, `dryrun/b2_mock_select.log` | 13/13 PASS: with no change, the S8 composite of recorded runs gives trial 7 (the R8 file's trial); keep-r8 and replacement paths write the SELECTION_R9 set (captured in memory); reuse8 / reuse7 records reproduce their stored phase-S eligibility and means through round 8's / round 7's loaders |
| B3 pool dry run | `dryrun/pool_dryrun_20261103_222736.{json,md}`, `dryrun/b3_pool_dryrun.log` | 171 directories, 128 eligible runs (new 0, trained further 4, unchanged 124); 48 report files, 48 eligible; final at snapshot: 35 yes, 13 no; 64 candidate files: 34 screen, 10 reuse8, 13 reuse7, 3 barred, 4 incumbent, 0 duplicate, 0 refused; 10 s |
| B3 trial pool | `dryrun/trial_20261103_222746/`, `dryrun/b3_pool_trial.log`, `dryrun/b3_pool_trial_verify.log` | 6 screen rows frozen for plumbing (three 5090 report files and one robolab report file relocated, one pro6000 report file, one pro6000 mechanical pick); 4 relocations ok and re-verified; its POOL_R9.sha256 holds. The ep00 G2 relocation (fe10ce8b...) equals the G2 check's |
| B3 plan, phase S | `dryrun/r9_eval_plan_screen_20261103_222833.txt`, `dryrun/b3_eval_plan_screen.log` | plan OK: PREREG_R9 sha256 holds; round 7's chain 0 fatal; trial POOL / RELOCATION / TRACKS lists hold; 6/6 jobs ok, hulls 6/6; kit + stock trees unchanged vs round 1's baseline; seed signature as round 8 |
| B3 plan, phase F | `dryrun/r9_eval_plan_final_20261103_222840.txt`, `dryrun/b3_eval_plan_final.log` | plan OK; no SELECTION_S9 yet, so no finalist (expected) |
| B3 reuse builders | `dryrun/r9_reuse_check.py`, `dryrun/b3_reuse_check.log` | all 10 reuse8 and 13 reuse7 rows of the trial pool rebuild as phase-F jobs from round 8's / round 7's records with their earlier exports (23/23 ok) |

**GPUs at the plan check (22:28 CST)** (PREREG_R9 3.5):
- GPUs 0, 1 and 2 were usable, shared with 3, 4 and 3 processes of our ablation (their MPS servers and train_rl_games.py
  clients; pipes under `ch_runs/ablation_20261031/mps/gpu{0,1,2}/`).
- GPUs 3-7 had no listed process and were usable.
- No MPS control daemon on the default pipe directory. Every GPU is in the Default compute mode.

**CPU at the build (22:29 CST).** Over 10 s, the eval-lane cores were idle: NUMA0 lane 48-51,152-155 mean busy 0.00, NUMA1
lane 96-103,200-207 mean busy 0.00. Whole nodes: NUMA0 0.07, NUMA1 0.01. Load average 9.0 on 208 logical CPUs. This is
the basis of C5 (NUMA1 lane 4 evaluations on 16 logical CPUs).

## The pool at the dry-run snapshot
### A. Per episode: incumbent, candidates and roles

Roles: **screen** = new, 3 fresh phase-S runs; **reuse8 / reuse7** = recorded round-8 / round-7 phase-S results (not re-run); **barred** = never a finalist; **= inc** = the incumbent itself (not screened). Final = the file's final status at the snapshot (D2; recorded only).

| ep | incumbent (round-8 pick, input) | new screen candidates (route; final at snapshot) | reuse8 | reuse7 | barred | = inc | evaluated |
|---|---|---|---|---|---|---|---|
| 00 | `ep00_z5k3fix_90000` (k3fix) | `ep00_zxrepg2_180000` (report G2; final) | - | - | - | `ep00_z5k3fix_90000` | yes |
| 01 | `ep01_k3fix_final44922` (k3fix) | - | - | `ep01_k3str_22500`<br>`ep01_zxg2_180000` | - | - | yes |
| 02 | `ep02_k3fix_final34093` (k3fix) | `ep02_zxrepg2_180000` (report G2; final) | - | - | `ep02_z5q_180000` | - | yes |
| 03 | `ep03_k3fix_final27068` (k3fix) | - | `ep03_z5g2rest_180000` | - | - | - | yes |
| 04 | `ep04_zxg2_450000` (g2) | - | - | `ep04_z5hfhc_180000` | - | `ep04_zxg2_450000` | yes |
| 05 | `ep05_zx_k3fix_88000` (k3fix) | - | `ep05_w32g2c_360000` | `ep05_zxk3fix_180000` | - | - | yes |
| 06 | `ep06_k3fix_final28420` (k3fix) | - | - | `ep06_z5q_180000` | - | - | yes |
| 07 | `ep07_k3fix_final18529` (k3fix) | - | `ep07_z5g2rest_180000` | - | - | - | yes |
| 08 | `ep08_k3easy_final17496` (k3) | - | - | `ep08_z5q_180000` | - | `ep08_k3easy_final17496` | yes |
| 09 | `ep09_raw_final39328` (raw) | - | `ep09_rb15k3fixr_180000` | `ep09_z5g2_180000` | - | - | yes |
| 10 | `ep10_raw_final34267` (raw) | `ep10_zxrepg2_145500` (report G2; not final) | - | - | - | - | yes |
| 11 | `ep11_raw_24000` (raw) | `ep11_zxrepg2_180000` (report G2; final)<br>`ep11_zxrepk3fix_59500` (report K3fix; not final) | - | - | - | - | yes |
| 12 | `ep12_k3fix_final23178` (k3fix) | `ep12_zxrepg2_180000` (report G2; final) | - | - | - | - | yes |
| 13 | `ep13_w312runinws_90000` (g2) | - | `ep13_w312runinws_180000` | - | - | - | yes |
| 14 | `ep14_k3fix_final44922` (k3fix) | `ep14_zxrepg2_134000` (report G2; not final)<br>`ep14_zxrepk3fix_276000` (report K3fix; not final) | - | - | - | - | yes |
| 15 | `ep15_raw_final44922` (raw) | - | `ep15_z5r2_180000` | - | - | - | yes |
| 16 | `ep16_z5a5_120000` (g2) | - | - | - | `ep16_z5a5_180000` | - | no: keeps its round-8 pick |
| 17 | `ep17_z5r2_61500` (k3fix) | `ep17_zxrepk3fix_74500` (report K3fix; not final) | `ep17_z5g2rest_180000` | - | - | - | yes |
| 18 | `ep18_k3fix_final18911` (k3fix) | `ep18_zxrepg2_180000` (report G2; final) | - | - | - | - | yes |
| 19 | `ep19_z5g2hardc2_360000` (g2) | - | - | - | - | `ep19_z5g2hardc2_360000` | no: keeps its round-8 pick |
| 20 | `ep20_z5fill_90000` (k3fix) | `ep20_zxg2rest_117000` (R4 j1; not final)<br>`ep20_zxg2rest_140500` (R4 j0; final)<br>`ep20_zxg2rest_70500` (R4 j3; not final)<br>`ep20_zxg2rest_93500` (R4 j2; not final)<br>`ep20_zxrepg2_139000` (report G2; not final) | - | `ep20_z5fill_180000` | - | - | yes |
| 21 | `ep21_z5fillc_282500` (g2) | `ep21_zxrepg2_360000` (report G2; final)<br>`ep21_zxrepk3fix_80500` (report K3fix; not final) | - | - | - | - | yes |
| 22 | `ep22_z5k3fix_89500` (k3fix) | `ep22_zxg2rest_113500` (R4 j1; not final)<br>`ep22_zxg2rest_136000` (R4 j0; final)<br>`ep22_zxg2rest_68000` (R4 j3; not final)<br>`ep22_zxg2rest_90500` (R4 j2; not final)<br>`ep22_zxrepg2_134500` (report G2; not final) | - | - | `ep22_z5k3fix_90000` | - | yes |
| 23 | `ep23_raw_final20023` (raw) | - | `ep23_w31g2_180000` | `ep23_k3easy_19162` | - | - | yes |
| 24 | `ep24_z5q_150000` (g2) | - | - | `ep24_z5fill_180000`<br>`ep24_z5q_180000` | - | - | yes |
| 25 | `ep25_z5a5c_280500` (k3fix) | `ep25_zxrepk3fix_297000` (report K3fix; not final) | `ep25_w31g2_180000` | - | - | - | yes |
| 26 | `ep26_zx_k3fix_92000` (k3fix) | - | `ep26_zxg2rest_180000` | `ep26_zxk3fix_180000` | - | - | yes |
| 27 | `ep27_k3fix_final20863` (k3fix) | `ep27_zxg2rest_117500` (R4 j0; final)<br>`ep27_zxg2rest_59000` (R4 j3; not final)<br>`ep27_zxg2rest_78500` (R4 j2; not final)<br>`ep27_zxg2rest_98000` (R4 j1; not final)<br>`ep27_zxrepg2_116000` (report G2; not final)<br>`ep27_zxrepk3fix_62500` (report K3fix; not final) | - | - | - | - | yes |
| 28 | `ep28_z5q_180000` (g2) | - | - | `ep28_zxg2_180000` | - | - | yes |
| 29 | `ep29_k3fix_final17358` (k3fix) | `ep29_zxg2rest_111000` (R4 j1; not final)<br>`ep29_zxg2rest_133000` (R4 j0; final)<br>`ep29_zxg2rest_66500` (R4 j3; not final)<br>`ep29_zxg2rest_88500` (R4 j2; not final)<br>`ep29_zxrepg2_131000` (report G2; not final) | - | - | - | - | yes |

Totals: 34 screen (102 phase-S evaluations), 10 reuse8, 13 reuse7, 3 barred, 4 equal to the incumbent, 0 duplicates, 0 refused; 28 of 30 episodes evaluated (ep16: its only candidate is barred; ep19: its only candidate is the incumbent).

### B. The 48 report files

| file | it | trained on | as-trained run | input | commit (evidence) | final at snapshot (evidence) | check report | role in the pool |
|---|---:|---|---|---|---|---|---|---|
| tier2-G2/ep00-tier2-G2.safetensors | 180000 | 5090 | 5090:2026-11-01-tier2-g2-rest/ep00 | g2 | 82837c8 (evidence/5090.txt (training-log line)) | yes (it == max_iterations) | yes | screen `ep00_zxrepg2_180000` |
| tier2-G2/ep01-tier2-G2.safetensors | 180000 | pro6000 | zx:runs/2026-10-24-tier2-runin-g2/ep01 | g2 | bc3b1bd (train_t2: line of logs/2026-10-24-tier2-runin-g2/ep01.log) | yes (it == max_iterations) | yes | reuse7 `ep01_zxg2_180000` |
| tier2-G2/ep02-tier2-G2.safetensors | 180000 | 5090 | 5090:2026-11-01-tier2-g2-rest/ep02 | g2 | 82837c8 (evidence/5090.txt (training-log line)) | yes (it == max_iterations) | yes | screen `ep02_zxrepg2_180000` |
| tier2-G2/ep03-tier2-G2.safetensors | 180000 | 5090 | 5090:2026-11-01-tier2-g2-rest/ep03 | g2 | 82837c8 (train_t2: line of zx:runs/2026-11-02-tier2-r8-handover/5090/2026-11-01-tier2-g2-rest/ep03/train.log) | yes (it == max_iterations) | yes | reuse8 `ep03_z5g2rest_180000` |
| tier2-G2/ep04-tier2-G2.safetensors | 450000 | pro6000 | zx:runs/2026-10-24-tier2-runin-g2/ep04 | g2 | bc3b1bd (train_t2: line of logs/2026-10-24-tier2-runin-g2/ep04.log) | yes (it == max_iterations) | yes | incumbent `ep04_zxg2_450000` |
| tier2-G2/ep05-tier2-G2.safetensors | 360000 | robolab | robolab:2026-10-26-tier2-runin-g2-ws/cont/ep05 | g2 | 82837c8 (evidence/ws032.txt (HEAD from refs)) | yes (it == max_iterations) | yes | reuse8 `ep05_w32g2c_360000` |
| tier2-G2/ep06-tier2-G2.safetensors | 180000 | 5090 | 5090:2026-10-29-tier2-5090-queue/ep06 | g2 | 82837c8 (train_t2: line of zx:runs/2026-11-01-round7-5090-ckpts/2026-10-29-tier2-5090-queue/ep06/train.log) | yes (it == max_iterations) | yes | reuse7 `ep06_z5q_180000` |
| tier2-G2/ep07-tier2-G2.safetensors | 180000 | 5090 | 5090:2026-11-01-tier2-g2-rest/ep07 | g2 | 82837c8 (train_t2: line of zx:runs/2026-11-02-tier2-r8-handover/5090/2026-11-01-tier2-g2-rest/ep07/train.log) | yes (it == max_iterations) | yes | reuse8 `ep07_z5g2rest_180000` |
| tier2-G2/ep08-tier2-G2.safetensors | 180000 | 5090 | 5090:2026-10-29-tier2-5090-queue/ep08 | g2 | 82837c8 (train_t2: line of zx:runs/2026-11-01-round7-5090-ckpts/2026-10-29-tier2-5090-queue/ep08/train.log) | yes (it == max_iterations) | yes | reuse7 `ep08_z5q_180000` |
| tier2-G2/ep09-tier2-G2.safetensors | 180000 | 5090 | 5090:2026-10-24-tier2-runin-g2/ep09 | g2 | 82837c8 (train_t2: line of zx:runs/2026-11-01-round7-5090-ckpts/2026-10-24-tier2-runin-g2/ep09/train.log) | yes (it == max_iterations) | yes | reuse7 `ep09_z5g2_180000` |
| tier2-G2/ep10-tier2-G2.safetensors | 145500 | 5090 | 5090:2026-11-02-tier2-g2-redo-5090/ep10 | g2 | 82837c8 (evidence/5090.txt (training-log line)) | no (final.json names it) | no: still training (5090 redo, 180k max) | screen `ep10_zxrepg2_145500` |
| tier2-G2/ep11-tier2-G2.safetensors | 180000 | robolab | robolab:2026-10-26-tier2-runin-g2-ws/ep11 | g2 | 82837c8 (evidence/ws032.txt (HEAD from refs)) | yes (it == max_iterations) | yes | screen `ep11_zxrepg2_180000` |
| tier2-G2/ep12-tier2-G2.safetensors | 180000 | 5090 | 5090:2026-11-01-tier2-g2-rest/ep12 | g2 | 82837c8 (evidence/5090.txt (training-log line)) | yes (it == max_iterations) | yes | screen `ep12_zxrepg2_180000` |
| tier2-G2/ep13-tier2-G2.safetensors | 180000 | robolab | robolab:2026-10-31-tier2-runin-ws/ep13 | g2 | 82837c8 (train_t2: line of zx:runs/2026-11-02-tier2-r8-handover/ws031/2026-10-31-tier2-runin-ws/ep13/train.log) | yes (it == max_iterations) | yes | reuse8 `ep13_w312runinws_180000` |
| tier2-G2/ep14-tier2-G2.safetensors | 134000 | 5090 | 5090:2026-11-02-tier2-g2-redo-5090/ep14 | g2 | 82837c8 (evidence/5090.txt (training-log line)) | no (final.json names it) | no: still training (5090 redo) | screen `ep14_zxrepg2_134000` |
| tier2-G2/ep15-tier2-G2.safetensors | 180000 | 5090 | 5090:2026-10-31-tier2-5090-round2/ep15 | g2 | 82837c8 (train_t2: line of zx:runs/2026-11-01-round7-5090-ckpts/2026-10-31-tier2-5090-round2/ep15/train.log) | yes (it == max_iterations) | yes | reuse8 `ep15_z5r2_180000` |
| tier2-G2/ep16-tier2-G2.safetensors | 180000 | 5090 | 5090:2026-10-28-tier2-runin-a5/ep16 | g2 | 82837c8 (train_t2: line of zx:runs/2026-11-01-round7-5090-ckpts/2026-10-28-tier2-runin-a5/ep16/train.log) | yes (it == max_iterations) | yes | barred `ep16_z5a5_180000` |
| tier2-G2/ep17-tier2-G2.safetensors | 180000 | 5090 | 5090:2026-11-01-tier2-g2-rest/ep17 | g2 | 82837c8 (train_t2: line of zx:runs/2026-11-02-tier2-r8-handover/5090/2026-11-01-tier2-g2-rest/ep17/train.log) | yes (it == max_iterations) | yes | reuse8 `ep17_z5g2rest_180000` |
| tier2-G2/ep18-tier2-G2.safetensors | 180000 | 5090 | 5090:2026-11-01-tier2-g2-rest/ep18 | g2 | 82837c8 (evidence/5090.txt (training-log line)) | yes (it == max_iterations) | yes | screen `ep18_zxrepg2_180000` |
| tier2-G2/ep19-tier2-G2.safetensors | 360000 | 5090 | 5090:2026-10-25-tier2-runin-hard-g2/cont2/ep19 | g2 | 82837c8 (train_t2: line of zx:runs/2026-11-01-round7-5090-ckpts/2026-10-25-tier2-runin-hard-g2/cont2/ep19/train.log) | yes (it == max_iterations) | no: series continues to 720k (his README) | incumbent `ep19_z5g2hardc2_360000` |
| tier2-G2/ep20-tier2-G2.safetensors | 139000 | pro6000 | zx:runs/2026-11-01-tier2-g2-rest/ep20 | g2 | bc3b1bd (train_t2: line of logs/2026-11-01-tier2-g2-rest/ep20.log) | no (later file in the run here) | no: run stopped 20:26:57; folder ends at 140500 | screen `ep20_zxrepg2_139000` |
| tier2-G2/ep21-tier2-G2.safetensors | 360000 | 5090 | 5090:2026-10-28-tier2-5090-fill/cont/ep21 | g2 | 82837c8 (train_t2: line of zx:runs/2026-11-01-round7-5090-ckpts/2026-10-28-tier2-5090-fill/cont/ep21/train.log) | yes (it == max_iterations) | yes | screen `ep21_zxrepg2_360000` |
| tier2-G2/ep22-tier2-G2.safetensors | 134500 | pro6000 | zx:runs/2026-11-01-tier2-g2-rest/ep22 | g2 | bc3b1bd (train_t2: line of logs/2026-11-01-tier2-g2-rest/ep22.log) | no (later file in the run here) | no: run stopped; folder ends at 136000 | screen `ep22_zxrepg2_134500` |
| tier2-G2/ep23-tier2-G2.safetensors | 180000 | robolab | robolab:2026-10-26-tier2-runin-g2-ws/ep23 | g2 | 82837c8 (train_t2: line of zx:runs/2026-11-02-tier2-r8-handover/ws031/2026-10-26-tier2-runin-g2-ws/ep23/train.log) | yes (it == max_iterations) | yes | reuse8 `ep23_w31g2_180000` |
| tier2-G2/ep24-tier2-G2.safetensors | 180000 | 5090 | 5090:2026-10-29-tier2-5090-queue/ep24 | g2 | 82837c8 (train_t2: line of zx:runs/2026-11-01-round7-5090-ckpts/2026-10-29-tier2-5090-queue/ep24/train.log) | yes (it == max_iterations) | yes | reuse7 `ep24_z5q_180000` |
| tier2-G2/ep25-tier2-G2.safetensors | 180000 | robolab | robolab:2026-10-26-tier2-runin-g2-ws/ep25 | g2 | 82837c8 (train_t2: line of zx:runs/2026-11-02-tier2-r8-handover/ws031/2026-10-26-tier2-runin-g2-ws/ep25/train.log) | yes (it == max_iterations) | yes | reuse8 `ep25_w31g2_180000` |
| tier2-G2/ep26-tier2-G2.safetensors | 180000 | pro6000 | zx:runs/2026-11-01-tier2-g2-rest/ep26 | g2 | bc3b1bd (train_t2: line of logs/2026-11-01-tier2-g2-rest/ep26.log) | yes (it == max_iterations) | yes | reuse8 `ep26_zxg2rest_180000` |
| tier2-G2/ep27-tier2-G2.safetensors | 116000 | pro6000 | zx:runs/2026-11-01-tier2-g2-rest/ep27 | g2 | bc3b1bd (train_t2: line of logs/2026-11-01-tier2-g2-rest/ep27.log) | no (later file in the run here) | no: run stopped; folder ends at 117500 | screen `ep27_zxrepg2_116000` |
| tier2-G2/ep28-tier2-G2.safetensors | 180000 | pro6000 | zx:runs/2026-10-24-tier2-runin-g2/ep28 | g2 | bc3b1bd (train_t2: line of logs/2026-10-24-tier2-runin-g2/ep28.log) | yes (it == max_iterations) | yes | reuse7 `ep28_zxg2_180000` |
| tier2-G2/ep29-tier2-G2.safetensors | 131000 | pro6000 | zx:runs/2026-11-01-tier2-g2-rest/ep29 | g2 | bc3b1bd (train_t2: line of logs/2026-11-01-tier2-g2-rest/ep29.log) | no (later file in the run here) | no: run stopped; folder ends at 133000 | screen `ep29_zxrepg2_131000` |
| tier2-K3fix/ep00-tier2-K3fix.safetensors | 90000 | 5090 | 5090:2026-10-24-tier2-runin-k3fix/ep00 | k3fix | 82837c8 (train_t2: line of zx:runs/2026-11-01-round7-5090-ckpts/2026-10-24-tier2-runin-k3fix/ep00/train.log) | yes (it == max_iterations) | yes | incumbent `ep00_z5k3fix_90000` |
| tier2-K3fix/ep01-tier2-K3fix.safetensors | 22500 | ours | C3r2_tier2_k3_stretch90k/ep01 | k3 | ours (our run, round 6 list) | no (final.json names it) | no: our K3 stretch run stopped at 22500 of 90000 (10-21) | reuse7 `ep01_k3str_22500` |
| tier2-K3fix/ep02-tier2-K3fix.safetensors | 180000 | 5090 | 5090:2026-10-29-tier2-5090-queue/ep02 | k3fix | 82837c8 (train_t2: line of zx:runs/2026-11-01-round7-5090-ckpts/2026-10-29-tier2-5090-queue/ep02/train.log) | yes (it == max_iterations) | yes | barred `ep02_z5q_180000` |
| tier2-K3fix/ep04-tier2-K3fix.safetensors | 180000 | 5090 | 5090:2026-10-23-tier2-helpfix-hard/cont/ep04 | k3fix | 82837c8 (train_t2: line of zx:runs/2026-11-01-round7-5090-ckpts/2026-10-23-tier2-helpfix-hard/cont/ep04/train.log) | yes (it == max_iterations) | yes | reuse7 `ep04_z5hfhc_180000` |
| tier2-K3fix/ep05-tier2-K3fix.safetensors | 180000 | pro6000 | zx:runs/2026-10-24-tier2-runin-k3fix-ep05-ep26/ep05 | k3fix | bc3b1bd (train_t2: line of logs/2026-10-24-tier2-runin-k3fix-ep05-ep26/ep05.log) | yes (it == max_iterations) | yes | reuse7 `ep05_zxk3fix_180000` |
| tier2-K3fix/ep08-tier2-K3fix.safetensors | 17496 | ours | C3r2_tier2_k3easy/ep08 | k3 | ours (our run, round 6 list) | yes (final.json names it) | yes | incumbent `ep08_k3easy_final17496` |
| tier2-K3fix/ep09-tier2-K3fix.safetensors | 180000 | robolab | robolab:2026-10-28-tier2-runin-k3fix-ws/resume/ep09 | k3fix | 82837c8 (train_t2: line of logs/2026-10-28-tier2-runin-k3fix-ws/ep09_resume.log) | yes (it == max_iterations) | yes | reuse8 `ep09_rb15k3fixr_180000` |
| tier2-K3fix/ep11-tier2-K3fix.safetensors | 59500 | 5090 | 5090:2026-10-31-tier2-5090-round2/ep11 | k3fix | 82837c8 (train_t2: line of zx:runs/2026-11-01-round7-5090-ckpts/2026-10-31-tier2-5090-round2/ep11/train.log) | no (final.json names it) | no: 5090 round-2 run paused (59500 of 180000) | screen `ep11_zxrepk3fix_59500` |
| tier2-K3fix/ep14-tier2-K3fix.safetensors | 276000 | 5090 | 5090:2026-10-28-tier2-runin-a5/cont/ep14 | k3fix | 82837c8 (train_t2: line of zx:runs/2026-11-01-round7-5090-ckpts/2026-10-28-tier2-runin-a5/cont/ep14/train.log) | no (final.json names it) | no: a5 continuation stopped (276000 of 360000) | screen `ep14_zxrepk3fix_276000` |
| tier2-K3fix/ep17-tier2-K3fix.safetensors | 74500 | 5090 | 5090:2026-10-31-tier2-5090-round2/ep17 | k3fix | 82837c8 (train_t2: line of zx:runs/2026-11-01-round7-5090-ckpts/2026-10-31-tier2-5090-round2/ep17/train.log) | no (final.json names it) | no: 5090 round-2 run paused (74500 of 180000) | screen `ep17_zxrepk3fix_74500` |
| tier2-K3fix/ep20-tier2-K3fix.safetensors | 180000 | 5090 | 5090:2026-10-28-tier2-5090-fill/ep20 | k3fix | 82837c8 (train_t2: line of zx:runs/2026-11-01-round7-5090-ckpts/2026-10-28-tier2-5090-fill/ep20/train.log) | yes (it == max_iterations) | yes | reuse7 `ep20_z5fill_180000` |
| tier2-K3fix/ep21-tier2-K3fix.safetensors | 80500 | 5090 | 5090:2026-10-31-tier2-5090-round2/ep21 | k3fix | 82837c8 (train_t2: line of zx:runs/2026-11-01-round7-5090-ckpts/2026-10-31-tier2-5090-round2/ep21/train.log) | no (final.json names it) | no: 5090 round-2 run paused (80500 of 180000) | screen `ep21_zxrepk3fix_80500` |
| tier2-K3fix/ep22-tier2-K3fix.safetensors | 90000 | 5090 | 5090:2026-10-24-tier2-runin-k3fix/ep22 | k3fix | 82837c8 (train_t2: line of zx:runs/2026-11-01-round7-5090-ckpts/2026-10-24-tier2-runin-k3fix/ep22/train.log) | yes (it == max_iterations) | yes | barred `ep22_z5k3fix_90000` |
| tier2-K3fix/ep23-tier2-K3fix.safetensors | 19162 | ours | C3r2_tier2_k3easy/ep23 | k3 | ours (our run, round 6 list) | yes (final.json names it) | yes | reuse7 `ep23_k3easy_19162` |
| tier2-K3fix/ep24-tier2-K3fix.safetensors | 180000 | 5090 | 5090:2026-10-28-tier2-5090-fill/ep24 | k3fix | 82837c8 (train_t2: line of zx:runs/2026-11-01-round7-5090-ckpts/2026-10-28-tier2-5090-fill/ep24/train.log) | yes (it == max_iterations) | yes | reuse7 `ep24_z5fill_180000` |
| tier2-K3fix/ep25-tier2-K3fix.safetensors | 297000 | 5090 | 5090:2026-10-28-tier2-runin-a5/cont/ep25 | k3fix | 82837c8 (train_t2: line of zx:runs/2026-11-01-round7-5090-ckpts/2026-10-28-tier2-runin-a5/cont/ep25/train.log) | no (final.json names it) | no: a5 continuation stopped (297000 of 360000) | screen `ep25_zxrepk3fix_297000` |
| tier2-K3fix/ep26-tier2-K3fix.safetensors | 180000 | pro6000 | zx:runs/2026-10-24-tier2-runin-k3fix-ep05-ep26/ep26 | k3fix | bc3b1bd (train_t2: line of logs/2026-10-24-tier2-runin-k3fix-ep05-ep26/ep26.log) | yes (it == max_iterations) | yes | reuse7 `ep26_zxk3fix_180000` |
| tier2-K3fix/ep27-tier2-K3fix.safetensors | 62500 | 5090 | 5090:2026-10-31-tier2-5090-round2/ep27 | k3fix | 82837c8 (train_t2: line of zx:runs/2026-11-01-round7-5090-ckpts/2026-10-31-tier2-5090-round2/ep27/train.log) | no (final.json names it) | no: 5090 round-2 run paused (62500 of 180000) | screen `ep27_zxrepk3fix_62500` |

### C. Runs that trained further since round 8's snapshot (D2 b)

| run | ep | L at round 8 | L now | R4 picks |
|---|---:|---:|---:|---|
| zx:runs/2026-11-01-tier2-g2-rest/ep20 | 20 | 122500 | 140500 | 140500, 117000, 93500, 70500 |
| zx:runs/2026-11-01-tier2-g2-rest/ep22 | 22 | 117500 | 136000 | 136000, 113500, 90500, 68000 |
| zx:runs/2026-11-01-tier2-g2-rest/ep27 | 27 | 100000 | 117500 | 117500, 98000, 78500, 59000 |
| zx:runs/2026-11-01-tier2-g2-rest/ep29 | 29 | 113000 | 133000 | 133000, 111000, 88500, 66500 |

New runs since round 8: 0. Unchanged eligible runs: 124 (no candidates by D2 b).

### D. Exclusions with reasons

- **Report files:** none excluded (48 of 48 eligible).
- **Directories scanned but not eligible** (round 8's rule, unchanged):
  - 26 x training parquet not in a Tier-2-derived data root (our runs on another input)
  - 12 x not a training-run directory (embedded output_dir elsewhere: forks / copies)
  - 1 x last settled iteration 9000 < 10000
  - 1 x last settled iteration 1000 < 10000
  - 1 x last settled iteration 500 < 10000
  - 1 x duplicate copy of the same run (kept: /root/autodl-tmp/zuxing.lu/runs/2026-11-02-tier2-r8-handover/5090/2026-10-31-tier2-5090-round2/ep15, L 180000)
  - 1 x no checkpoint present, settled and listed in its checksum file at T_c
- **Folders not walked / never entered:** round 8's list, plus `2026-11-03-tier2-g2-final-ckpts`, `2026-11-03-tier2-k3fix-final-ckpts`, `2026-11-03-tier2-k3fix-ownsel`, `2026-11-03-tier2-g2-reports-eval30` and `2026-11-02-tier2-g2-ownsel` (PREREG_R9 2.2).

### E. Notes

- **Final status** follows PREREG_R9 2.1 (recorded only). It agrees with the two check reports except for G2 ep19
  (cont2 at 360000 equals its run's max_iterations, so "yes" here; the G2 report says "no" because his README continues
  the series to 720k). That file is the round-8 incumbent itself.
- **ep13.** The G2 report file (ws031, 180000) is a round-8 phase-S candidate: it reuses its round-8 phase-S record
  against the new bar_S of the round-8 replacement.
- **Not-final files (D2).** G2 ep10, 14, 20, 22, 27, 29 and K3fix ep01, 11, 14, 17, 21, 25, 27 are evaluated as they are
  (ep01 as its recorded round-7 phase-S result). For G2 ep20, 22, 27 and 29, the runs' last files (140500, 136000,
  117500, 133000) enter as R4 j0 picks.
- **Budget.** 102 phase-S evaluations, then at most one finalist per evaluated episode (28) x 10.
