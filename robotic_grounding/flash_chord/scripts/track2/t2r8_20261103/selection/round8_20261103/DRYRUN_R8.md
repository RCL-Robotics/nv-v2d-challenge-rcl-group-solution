# t2_consensus_r8_20261103: DRYRUN_R8 (records of the pre-launch steps B2-B3; decides nothing)

Written 2026-11-02 about 21:55 CST, for the revised build (`PREREG_R8_CHANGES.md`).
- **Not run yet:** no round-8 export, evaluation, score or selection exists. `POOL_R8.json` / `POOL_R8.sha256` do not
  exist (the pool is frozen only at T_c = 2026-11-03 12:00:00 CST). The orchestrator is launched after the lock and only
  sleeps until T_c (`LAUNCH_R8.md`).
- **Documents in force:** `PREREG_R8.md` bb912c57180dde8255776884de169e12467349f57d584782308b88ff935abf8c, `PREREG_R8_PINS.md` (see `LOCK_R8.sha256`).
- **Basis:** selection uses only own-input and consensus-B scores.

## 1. Step B2: round-7 records, selftest, selection mock (kit Python, CPU)

| step | result | record |
|---|---|---|
| `r8_select.py r7records` | round 7's chain verified (0 fatal; round 1's chain 0 fatal; `SELECTION_R7.sha256` 24 lines hold); `inputs_r8/R7_RECORDS_R8.json` 60c2798c... written. It holds the 30 incumbents (the table of PREREG_R8 4), 399 round-7 phase-S identities and the phase-F participants. No participant would pass the round-8 test on its own round-7 runs. Apart from its time stamp and script sha256 it equals the first build's file. | `dryrun/b2_r7records.log` |
| `r8_select.py selftest` | 14/14 synthetic checks PASS | `selftest_r8/selftest_r8.json`, `dryrun/b2_selftest.log` |
| `dryrun/r8_mock_select.py` | 17/17 checks PASS on synthetic round-8 values with the real round-7 records. Nothing was written. | `dryrun/b2_mock_select.log` |

What the mock checked:
- **Phase S:** episodes without a round-8 candidate are not evaluated; barred7 candidates are never finalists; every
  finalist clears m_P_frozen + 0.01 and is S-eligible; a real reuse7 job carries its round-7 phase-S record.
- **Phase F:**
  - replacement happens iff the finalist has 10 good runs, floor >= 8 and m_F >= bar;
  - a deadline-stopped finalist (7 of 10 runs) and a finalist below the bar keep the round-7 pick;
  - kept episodes use their round-7 runs, and R8-CLOSEST covers all 30 episodes.
- **No finalist, and keep-r7:** S8 == S7, and R8-CLOSEST is round 7's trial 2, recomputed from the real round-7 runs.
- **One evaluation driver at a time (PREREG_R8 7.6):** keep-r7 is refused while an `r8_eval.py run` process is alive or
  `driver.pid` names a live one; none is alive on this host now.

Off-server checks of the revision (not pinned; described in `PREREG_R8_CHANGES.md`): simulations of the orchestrator's
evaluation phase (earlier evaluation driver still running, stale `driver.pid` at a deadline, preflight failure, pinned-file
mismatch, repeated crash, `halted` heartbeat, deadline during a run, crash then success) and of gate / score resumption
all gave the intended result.

## 2. Step B3: pool dry run (snapshot NOW, nothing copied or frozen)

`r8_pool.py dryrun` at **2026-11-02 21:45:35 CST** (11 s). Outputs: `dryrun/pool_dryrun_20261102_214535.{json,md}` and
`dryrun/b3_pool_dryrun.log`. The full tables follow in section 5.

### Universe

- 171 directories scanned; 128 eligible runs: 59 ours, 69 Zuxing's.
- **New since round 7: 10 runs.**
  - 3 5090 G2-rest runs in the hand-over: ep03, ep07, ep17.
  - 7 robolab-host runs:
    - wsrbl015: ep13 G2, ep09 K3fix, ep09 K3fix resume;
    - ws031: ep13, ep23, ep25 (G2);
    - ws032: cont/ep05 (G2).
- **Trained further: 6 runs.** Zuxing's pro6000 G2-rest ep20, 22, 26, 27, 29 (still training at the dry run; L now
  92000 / 86500 / 170500 / 71000 / 80000) and the 5090 round-2 ep15 run (L 74000 at round 7, now 180000 in the
  hand-over).
- **Unchanged: 112 runs**, including all of ours.

### Candidates (98 files)

| role | count | how they arose |
|---|---:|---|
| screen (new files) | **89** | 35 mechanical picks of new runs, 5 mechanical + watcher best of new runs, 5 watcher best of new runs, 23 mechanical picks of trained-further runs, 1 mechanical + watcher best of a trained-further run, 5 watcher best of trained-further runs, 15 watcher best of unchanged runs |
| reuse7 | 5 | watcher best of unchanged runs equal to round-7 phase-S candidates: ep09_z5g2_180000, ep19_zxg2hard_180000, ep21_z5r2_60000, ep27_z5r2_40000, ep28_zxg2_120000 |
| barred7 | 3 | watcher best equal to round-7 phase-F participants: ep02_z5q_180000, ep22_z5k3fix_90000, ep25_z5a5c_190000 |
| == incumbent | 1 | watcher best of ep00 == its round-7 pick ep00_z5k3fix_90000 |
| duplicate / refused | 0 / 0 | |

### Breakdown

- **Screen candidates by host:** pro6000 30, 5090 27 (all relocated), robolab hosts 32 (all relocated).
- **Screen candidates by input:** G2 75, K3fix 14.
- **Screen candidates per episode:** 01: 2, 03: 5, 04: 1, 05: 5, 06: 1, 07: 4, 08: 1, 09: 9, 13: 10, 14: 1, 15: 5, 16: 1,
  17: 4, 19: 2, 20: 6, 21: 1, 22: 5, 23: 4, 24: 2, 25: 5, 26: 6, 27: 5, 29: 4 (episode 28: one reuse7 candidate only).
- **Episodes that would be evaluated (24):** 01, 03-09, 13-17, 19-29.
- **Episodes that keep their round-7 pick without evaluation (6):** 00, 02, 10, 11, 12, 18. These have no round-8
  candidate, or only one equal to the incumbent or barred.
- **Watcher files:** 40 of the 69 eligible Zuxing runs have one (24 next to the run, 16 in the ownsel `inputs_5090`
  fallback). All 40 gave an eligible watcher-best row. Every header is the 18-column own-input header.
- **5090 watcher rule (PREREG_R8 2.6 (5)):** 20 watcher rows are 5090-hosted; in 6 of them the last ok-evaluated
  iteration is below the run's L, so it does not count off the grid. No 5090 watcher pick is off the grid, so the rule
  changes no pick at this snapshot.

### Excluded (reasons in section 5)

- **Inputs outside the Tier-2-derived data roots:** 26 of Zuxing's early pro6000 runs.
- **Not training-run directories:** 12 T2fix BASE / fork folders.
- **L < 10,000:** 3 runs (`C3r2_tier2_k3_stretch90k/ep04`, Zuxing's `2026-10-28-tier2-runin-a5/ep16`, `ep25`).
- **Duplicate copy:** 1. Round 7's 5090 folder copy of round-2 ep15; the hand-over copy with L 180000 counts.
- **Nothing counted:** 1 (`ws032_aborted_stream/.../ep05`; no file listed in its checksum file).
- **Folders not walked:** 7.
- **Missing commit evidence:** none. ws032 has no git binary; its commit 82837c8 comes from the hand-over evidence file.

### Host parquet bytes (PREREG_R8 9.4)

For the 8 copy runs whose host records its own `task.parquet` sha256 (hand-over 5090 and ws031 / ws032), all 8 differ
from this server's same-named parquet. These are recorded only. The other 47 copy runs have no host value recorded.

### Expected volume at T_c

- **Now:** 89 screen candidates x 3 = 267 phase-S evaluations.
- **Possibly still to come:** the hand-over evidence lists the 5090 G2-rest runs ep00 / 02 / 12 / 18 and G2-redo runs
  ep10 / 14, and the hand-over is still being filled (ws032). The pro6000 G2-rest runs keep training.
- **Expected at T_c:** about 100-130 screen candidates (300-390 evaluations).

## 3. Step B3: trial pool and plan-only evaluation check (plumbing only)

`r8_pool.py trial --limit 3` froze a 3-row trial pool under `dryrun/trial_20261102_214546/`. It holds 368 copies, all with the
source's sha256 equal before and after (watcher files and run metadata included), and 2 relocations:
- `ep03_z5g2rest_90000` (5090);
- `ep05_w32g2c_182000` (ws032).

57 of the 368 copies are record excerpts (PREREG_R8 2.8): MANIFEST.txt and checksum files reduced to the lines round 8
uses, and the watcher files reduced to their 8 own-input columns. A text search of the whole round-8 folder found no
file or score outside the own-input / consensus-B basis; the only match of the search terms is a JSON-logging option
in the copied hydra files.

`r8_pool.py verify --base` passed: 2 relocations, `POOL_R8.sha256` 0 lines changed.

The plan-only evaluation check `r8_eval.py run --phase screen|final --dry-run --pool-root <trial>` gave `plan: OK` for
both phases (`dryrun/r8_eval_plan_{screen,final}_*.txt`, `dryrun/b3_eval_plan_{screen,final}.log`).
- **Pin chain:** PREREG_R8 matches its sha file and hard-coded value. Round 7's chain (its own `check_pins`) has 0
  fatal; 31 lines verified. The trial's `POOL_R8.sha256` (384 files), `RELOCATION_R8.sha256` and `TRACKS_R8.sha256`
  hold.
- **Preflight:** kit `evaluate_policy.py` == stock == 160d24f5; fc_r2 @ 61fa5d3 and b39b69b trees at HEAD with
  `eval_all.py` 657295921e82; stock 8c778f9.
- **Jobs:** 3/3 ok. They are the pro6000 copy `ep01_zxg2_160000` (watcher best), the 5090 relocated `ep03_z5g2rest_90000`
  and the ws032 relocated `ep05_w32g2c_182000`. Hulls ok.
- **Integrity:** kit + stock trees unchanged versus round 1's baseline.
- **Seeds:** signature `delay=0(+g0..4,0-2) reset=42 scale_seed=0(1.0/1.0) wrench=0 vertex=0/0 ppo=42`.
- **GPUs:** GPU 6 and GPU 7 are refused while compute processes are listed (3 each at 21:46, Zuxing's trainings). GPUs 0-5
  are refused by rule.
- **Phase F:** no finalist yet, as expected.

Nothing was exported or evaluated, and no lock was taken.

## 4. Launch

From the round-8 folder: `bash r8_launch.sh`. It starts `r8_driver.py run` with setsid nohup, refuses to duplicate a
running driver, and refuses while an `r8_eval.py` evaluation driver is alive.
- **Before T_c,** the driver only sleeps, writing its heartbeat.
- **Status:** `conda_envs/flashchord/bin/python -I -B r8_driver.py status`, `R8_DRIVER_HEARTBEAT.json` and
  `r8_driver.log`.
- **Stop:** `touch STOP`. Running evaluations finish, and nothing further is selected.
- **Results:** `SELECTION_R8.json`, `selection_r8_episodes.csv` and `SELECTION_R8.sha256` (written last).

## 5. Pool dry-run tables (verbatim copy of `dryrun/pool_dryrun_20261102_214535.md`)

## Pool dry run (20261102_214535)

Snapshot 2026-11-02 21:45:35 CST.

### Candidates per episode

| ep | round-8 incumbent (round-7 pick) | new screen | reuse round-7 S | barred | == incumbent | duplicate | refused | evaluated |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 00 | `ep00_z5k3fix_90000` | 0 | 0 | 0 | 1 | 0 | 0 | no (keeps its round-7 pick) |
| 01 | `ep01_k3fix_final44922` | 2 | 0 | 0 | 0 | 0 | 0 | yes |
| 02 | `ep02_k3fix_final34093` | 0 | 0 | 1 | 0 | 0 | 0 | no (keeps its round-7 pick) |
| 03 | `ep03_k3fix_final27068` | 5 | 0 | 0 | 0 | 0 | 0 | yes |
| 04 | `ep04_zxg2_450000` | 1 | 0 | 0 | 0 | 0 | 0 | yes |
| 05 | `ep05_zx_k3fix_88000` | 5 | 0 | 0 | 0 | 0 | 0 | yes |
| 06 | `ep06_k3fix_final28420` | 1 | 0 | 0 | 0 | 0 | 0 | yes |
| 07 | `ep07_k3fix_final18529` | 4 | 0 | 0 | 0 | 0 | 0 | yes |
| 08 | `ep08_k3easy_final17496` | 1 | 0 | 0 | 0 | 0 | 0 | yes |
| 09 | `ep09_raw_final39328` | 9 | 1 | 0 | 0 | 0 | 0 | yes |
| 10 | `ep10_raw_final34267` | 0 | 0 | 0 | 0 | 0 | 0 | no (keeps its round-7 pick) |
| 11 | `ep11_raw_24000` | 0 | 0 | 0 | 0 | 0 | 0 | no (keeps its round-7 pick) |
| 12 | `ep12_k3fix_final23178` | 0 | 0 | 0 | 0 | 0 | 0 | no (keeps its round-7 pick) |
| 13 | `ep13_k3fix_final39254` | 10 | 0 | 0 | 0 | 0 | 0 | yes |
| 14 | `ep14_k3fix_final44922` | 1 | 0 | 0 | 0 | 0 | 0 | yes |
| 15 | `ep15_raw_final44922` | 5 | 0 | 0 | 0 | 0 | 0 | yes |
| 16 | `ep16_z5a5_120000` | 1 | 0 | 0 | 0 | 0 | 0 | yes |
| 17 | `ep17_z5r2_61500` | 4 | 0 | 0 | 0 | 0 | 0 | yes |
| 18 | `ep18_k3fix_final18911` | 0 | 0 | 0 | 0 | 0 | 0 | no (keeps its round-7 pick) |
| 19 | `ep19_z5g2hardc2_360000` | 2 | 1 | 0 | 0 | 0 | 0 | yes |
| 20 | `ep20_z5fill_90000` | 6 | 0 | 0 | 0 | 0 | 0 | yes |
| 21 | `ep21_z5fillc_282500` | 1 | 1 | 0 | 0 | 0 | 0 | yes |
| 22 | `ep22_z5k3fix_89500` | 5 | 0 | 1 | 0 | 0 | 0 | yes |
| 23 | `ep23_raw_final20023` | 4 | 0 | 0 | 0 | 0 | 0 | yes |
| 24 | `ep24_z5q_150000` | 2 | 0 | 0 | 0 | 0 | 0 | yes |
| 25 | `ep25_z5a5c_280500` | 5 | 0 | 1 | 0 | 0 | 0 | yes |
| 26 | `ep26_zx_k3fix_92000` | 6 | 0 | 0 | 0 | 0 | 0 | yes |
| 27 | `ep27_k3fix_final20863` | 5 | 1 | 0 | 0 | 0 | 0 | yes |
| 28 | `ep28_z5q_180000` | 0 | 1 | 0 | 0 | 0 | 0 | yes |
| 29 | `ep29_k3fix_final17358` | 4 | 0 | 0 | 0 | 0 | 0 | yes |

Totals: 89 new screen candidates (x 3 runs = 267 phase-S evaluations), 5 reusing round-7 phase-S results, 3 barred, 1 equal to the incumbent, 0 duplicates, 0 refused.

### Candidate list

| ep | job | role | route(s) | input | run (status, L) | note |
|---|---|---|---|---|---|---|
| 00 | `ep00_z5k3fix_90000` | incumbent | watcher best | k3fix | zx:2026-11-01-round7-5090-ckpts/2026-10-24-tier2-runin-k3fix/ep00 (unchanged, L 90000) | == the round-7 pick ep00_z5k3fix_90000 (the round-8 incumbent; not screened) |
| 01 | `ep01_zxg2_160000` | screen | watcher best | g2 | zx:2026-10-24-tier2-runin-g2/ep01 (unchanged, L 180000) | new file |
| 01 | `ep01_zxg2hard_130000` | screen | watcher best | g2 | zx:2026-10-25-tier2-runin-hard-g2/ep01 (unchanged, L 180000) | new file |
| 02 | `ep02_z5q_180000` | barred7 | watcher best | k3fix | zx:2026-11-01-round7-5090-ckpts/2026-10-29-tier2-5090-queue/ep02 (unchanged, L 180000) | == ep02_z5q_180000, a round-7 phase-F participant of this episode (round-7 finalist not picked; its round-7 phase-F runs do not pass the round-8 test): never a finalist (PREREG_R8 3.3) |
| 03 | `ep03_z5g2rest_110000` | screen | watcher best | g2 | zx:2026-11-02-tier2-r8-handover/5090/2026-11-01-tier2-g2-rest/ep03 (new, L 180000) | new file |
| 03 | `ep03_z5g2rest_120000` | screen | R4 j2 (4/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/5090/2026-11-01-tier2-g2-rest/ep03 (new, L 180000) | new file |
| 03 | `ep03_z5g2rest_150000` | screen | R4 j1 (5/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/5090/2026-11-01-tier2-g2-rest/ep03 (new, L 180000) | new file |
| 03 | `ep03_z5g2rest_180000` | screen | R4 j0 (6/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/5090/2026-11-01-tier2-g2-rest/ep03 (new, L 180000) | new file |
| 03 | `ep03_z5g2rest_90000` | screen | R4 j3 (3/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/5090/2026-11-01-tier2-g2-rest/ep03 (new, L 180000) | new file |
| 04 | `ep04_zxg2_448000` | screen | watcher best | g2 | zx:2026-10-24-tier2-runin-g2/ep04 (unchanged, L 450000) | new file |
| 05 | `ep05_w32g2c_182000` | screen | R4 j3 (3/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/ws032/2026-10-26-tier2-runin-g2-ws/cont/ep05 (new, L 360000) | new file |
| 05 | `ep05_w32g2c_240000` | screen | R4 j2 (4/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/ws032/2026-10-26-tier2-runin-g2-ws/cont/ep05 (new, L 360000) | new file |
| 05 | `ep05_w32g2c_300000` | screen | R4 j1 (5/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/ws032/2026-10-26-tier2-runin-g2-ws/cont/ep05 (new, L 360000) | new file |
| 05 | `ep05_w32g2c_360000` | screen | R4 j0 (6/6 L), watcher best | g2 | zx:2026-11-02-tier2-r8-handover/ws032/2026-10-26-tier2-runin-g2-ws/cont/ep05 (new, L 360000) | new file |
| 05 | `ep05_zxk3fix_170000` | screen | watcher best | k3fix | zx:2026-10-24-tier2-runin-k3fix-ep05-ep26/ep05 (unchanged, L 180000) | new file |
| 06 | `ep06_z5q_80000` | screen | watcher best | g2 | zx:2026-11-01-round7-5090-ckpts/2026-10-29-tier2-5090-queue/ep06 (unchanged, L 180000) | new file |
| 07 | `ep07_z5g2rest_120000` | screen | R4 j2 (4/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/5090/2026-11-01-tier2-g2-rest/ep07 (new, L 180000) | new file |
| 07 | `ep07_z5g2rest_150000` | screen | R4 j1 (5/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/5090/2026-11-01-tier2-g2-rest/ep07 (new, L 180000) | new file |
| 07 | `ep07_z5g2rest_180000` | screen | R4 j0 (6/6 L), watcher best | g2 | zx:2026-11-02-tier2-r8-handover/5090/2026-11-01-tier2-g2-rest/ep07 (new, L 180000) | new file |
| 07 | `ep07_z5g2rest_90000` | screen | R4 j3 (3/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/5090/2026-11-01-tier2-g2-rest/ep07 (new, L 180000) | new file |
| 08 | `ep08_z5q_100000` | screen | watcher best | g2 | zx:2026-11-01-round7-5090-ckpts/2026-10-29-tier2-5090-queue/ep08 (unchanged, L 180000) | new file |
| 09 | `ep09_z5g2_180000` | reuse7 | watcher best | g2 | zx:2026-11-01-round7-5090-ckpts/2026-10-24-tier2-runin-g2/ep09 (unchanged, L 180000) | == round-7 phase-S candidate ep09_z5g2_180000: its round-7 phase-S results and export are reused |
| 09 | `ep09_rb15k3fix_100000` | screen | watcher best | k3fix | zx:2026-11-02-round7-robolab-ckpts/wsrbl015/2026-10-28-tier2-runin-k3fix-ws/ep09 (new, L 101000) | new file |
| 09 | `ep09_rb15k3fix_101000` | screen | R4 j0 (6/6 L) | k3fix | zx:2026-11-02-round7-robolab-ckpts/wsrbl015/2026-10-28-tier2-runin-k3fix-ws/ep09 (new, L 101000) | new file |
| 09 | `ep09_rb15k3fix_52000` | screen | R4 j3 (3/6 L) | k3fix | zx:2026-11-02-round7-robolab-ckpts/wsrbl015/2026-10-28-tier2-runin-k3fix-ws/ep09 (new, L 101000) | new file |
| 09 | `ep09_rb15k3fix_68000` | screen | R4 j2 (4/6 L) | k3fix | zx:2026-11-02-round7-robolab-ckpts/wsrbl015/2026-10-28-tier2-runin-k3fix-ws/ep09 (new, L 101000) | new file |
| 09 | `ep09_rb15k3fix_84000` | screen | R4 j1 (5/6 L) | k3fix | zx:2026-11-02-round7-robolab-ckpts/wsrbl015/2026-10-28-tier2-runin-k3fix-ws/ep09 (new, L 101000) | new file |
| 09 | `ep09_rb15k3fixr_102000` | screen | R4 j3 (3/6 L) | k3fix | zx:2026-11-02-round7-robolab-ckpts/wsrbl015/2026-10-28-tier2-runin-k3fix-ws/resume/ep09 (new, L 180000) | new file |
| 09 | `ep09_rb15k3fixr_120000` | screen | R4 j2 (4/6 L) | k3fix | zx:2026-11-02-round7-robolab-ckpts/wsrbl015/2026-10-28-tier2-runin-k3fix-ws/resume/ep09 (new, L 180000) | new file |
| 09 | `ep09_rb15k3fixr_150000` | screen | R4 j1 (5/6 L) | k3fix | zx:2026-11-02-round7-robolab-ckpts/wsrbl015/2026-10-28-tier2-runin-k3fix-ws/resume/ep09 (new, L 180000) | new file |
| 09 | `ep09_rb15k3fixr_180000` | screen | R4 j0 (6/6 L), watcher best | k3fix | zx:2026-11-02-round7-robolab-ckpts/wsrbl015/2026-10-28-tier2-runin-k3fix-ws/resume/ep09 (new, L 180000) | new file |
| 13 | `ep13_rb15g2_120000` | screen | R4 j2 (4/6 L) | g2 | zx:2026-11-02-round7-robolab-ckpts/wsrbl015/2026-10-26-tier2-runin-g2-ws/ep13 (new, L 180000) | new file |
| 13 | `ep13_rb15g2_140000` | screen | watcher best | g2 | zx:2026-11-02-round7-robolab-ckpts/wsrbl015/2026-10-26-tier2-runin-g2-ws/ep13 (new, L 180000) | new file |
| 13 | `ep13_rb15g2_150000` | screen | R4 j1 (5/6 L) | g2 | zx:2026-11-02-round7-robolab-ckpts/wsrbl015/2026-10-26-tier2-runin-g2-ws/ep13 (new, L 180000) | new file |
| 13 | `ep13_rb15g2_180000` | screen | R4 j0 (6/6 L) | g2 | zx:2026-11-02-round7-robolab-ckpts/wsrbl015/2026-10-26-tier2-runin-g2-ws/ep13 (new, L 180000) | new file |
| 13 | `ep13_rb15g2_90000` | screen | R4 j3 (3/6 L) | g2 | zx:2026-11-02-round7-robolab-ckpts/wsrbl015/2026-10-26-tier2-runin-g2-ws/ep13 (new, L 180000) | new file |
| 13 | `ep13_w312runinws_110000` | screen | watcher best | g2 | zx:2026-11-02-tier2-r8-handover/ws031/2026-10-31-tier2-runin-ws/ep13 (new, L 180000) | new file |
| 13 | `ep13_w312runinws_120000` | screen | R4 j2 (4/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/ws031/2026-10-31-tier2-runin-ws/ep13 (new, L 180000) | new file |
| 13 | `ep13_w312runinws_150000` | screen | R4 j1 (5/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/ws031/2026-10-31-tier2-runin-ws/ep13 (new, L 180000) | new file |
| 13 | `ep13_w312runinws_180000` | screen | R4 j0 (6/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/ws031/2026-10-31-tier2-runin-ws/ep13 (new, L 180000) | new file |
| 13 | `ep13_w312runinws_90000` | screen | R4 j3 (3/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/ws031/2026-10-31-tier2-runin-ws/ep13 (new, L 180000) | new file |
| 14 | `ep14_z5a5c_250000` | screen | watcher best | k3fix | zx:2026-11-01-round7-5090-ckpts/2026-10-28-tier2-runin-a5/cont/ep14 (unchanged, L 263500) | new file |
| 15 | `ep15_z5r2_120000` | screen | R4 j2 (4/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/5090/2026-10-31-tier2-5090-round2/ep15 (trained further, L 180000) | new file |
| 15 | `ep15_z5r2_150000` | screen | R4 j1 (5/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/5090/2026-10-31-tier2-5090-round2/ep15 (trained further, L 180000) | new file |
| 15 | `ep15_z5r2_170000` | screen | watcher best | g2 | zx:2026-11-02-tier2-r8-handover/5090/2026-10-31-tier2-5090-round2/ep15 (trained further, L 180000) | new file |
| 15 | `ep15_z5r2_180000` | screen | R4 j0 (6/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/5090/2026-10-31-tier2-5090-round2/ep15 (trained further, L 180000) | new file |
| 15 | `ep15_z5r2_90000` | screen | R4 j3 (3/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/5090/2026-10-31-tier2-5090-round2/ep15 (trained further, L 180000) | new file |
| 16 | `ep16_z5a5_140000` | screen | watcher best | g2 | zx:2026-11-01-round7-5090-ckpts/2026-10-28-tier2-runin-a5/ep16 (unchanged, L 180000) | new file |
| 17 | `ep17_z5g2rest_120000` | screen | R4 j2 (4/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/5090/2026-11-01-tier2-g2-rest/ep17 (new, L 180000) | new file |
| 17 | `ep17_z5g2rest_150000` | screen | R4 j1 (5/6 L), watcher best | g2 | zx:2026-11-02-tier2-r8-handover/5090/2026-11-01-tier2-g2-rest/ep17 (new, L 180000) | new file |
| 17 | `ep17_z5g2rest_180000` | screen | R4 j0 (6/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/5090/2026-11-01-tier2-g2-rest/ep17 (new, L 180000) | new file |
| 17 | `ep17_z5g2rest_90000` | screen | R4 j3 (3/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/5090/2026-11-01-tier2-g2-rest/ep17 (new, L 180000) | new file |
| 19 | `ep19_zxg2hard_180000` | reuse7 | watcher best | g2 | zx:2026-10-25-tier2-runin-hard-g2/ep19 (unchanged, L 180000) | == round-7 phase-S candidate ep19_zxg2hard_180000: its round-7 phase-S results and export are reused |
| 19 | `ep19_z5g2hardc2_340000` | screen | watcher best | g2 | zx:2026-11-01-round7-5090-ckpts/2026-10-25-tier2-runin-hard-g2/cont2/ep19 (unchanged, L 360000) | new file |
| 19 | `ep19_zxg2_160000` | screen | watcher best | g2 | zx:2026-10-24-tier2-runin-g2/ep19 (unchanged, L 180000) | new file |
| 20 | `ep20_z5fill_110000` | screen | watcher best | k3fix | zx:2026-11-01-round7-5090-ckpts/2026-10-28-tier2-5090-fill/ep20 (unchanged, L 180000) | new file |
| 20 | `ep20_zxg2rest_46000` | screen | R4 j3 (3/6 L) | g2 | zx:2026-11-01-tier2-g2-rest/ep20 (trained further, L 92000) | new file |
| 20 | `ep20_zxg2rest_61500` | screen | R4 j2 (4/6 L) | g2 | zx:2026-11-01-tier2-g2-rest/ep20 (trained further, L 92000) | new file |
| 20 | `ep20_zxg2rest_76500` | screen | R4 j1 (5/6 L) | g2 | zx:2026-11-01-tier2-g2-rest/ep20 (trained further, L 92000) | new file |
| 20 | `ep20_zxg2rest_80000` | screen | watcher best | g2 | zx:2026-11-01-tier2-g2-rest/ep20 (trained further, L 92000) | new file |
| 20 | `ep20_zxg2rest_92000` | screen | R4 j0 (6/6 L) | g2 | zx:2026-11-01-tier2-g2-rest/ep20 (trained further, L 92000) | new file |
| 21 | `ep21_z5r2_60000` | reuse7 | watcher best | k3fix | zx:2026-11-01-round7-5090-ckpts/2026-10-31-tier2-5090-round2/ep21 (unchanged, L 69500) | == round-7 phase-S candidate ep21_z5r2_60000: its round-7 phase-S results and export are reused |
| 21 | `ep21_z5fillc_280000` | screen | watcher best | g2 | zx:2026-11-01-round7-5090-ckpts/2026-10-28-tier2-5090-fill/cont/ep21 (unchanged, L 282500) | new file |
| 22 | `ep22_z5k3fix_90000` | barred7 | watcher best | k3fix | zx:2026-11-01-round7-5090-ckpts/2026-10-24-tier2-runin-k3fix/ep22 (unchanged, L 90000) | == ep22_z5k3fix_90000, a round-7 phase-F participant of this episode (round-7 finalist not picked; its round-7 phase-F runs do not pass the round-8 test): never a finalist (PREREG_R8 3.3) |
| 22 | `ep22_zxg2rest_43500` | screen | R4 j3 (3/6 L) | g2 | zx:2026-11-01-tier2-g2-rest/ep22 (trained further, L 86500) | new file |
| 22 | `ep22_zxg2rest_57500` | screen | R4 j2 (4/6 L) | g2 | zx:2026-11-01-tier2-g2-rest/ep22 (trained further, L 86500) | new file |
| 22 | `ep22_zxg2rest_72000` | screen | R4 j1 (5/6 L) | g2 | zx:2026-11-01-tier2-g2-rest/ep22 (trained further, L 86500) | new file |
| 22 | `ep22_zxg2rest_80000` | screen | watcher best | g2 | zx:2026-11-01-tier2-g2-rest/ep22 (trained further, L 86500) | new file |
| 22 | `ep22_zxg2rest_86500` | screen | R4 j0 (6/6 L) | g2 | zx:2026-11-01-tier2-g2-rest/ep22 (trained further, L 86500) | new file |
| 23 | `ep23_w31g2_120000` | screen | R4 j2 (4/6 L), watcher best | g2 | zx:2026-11-02-tier2-r8-handover/ws031/2026-10-26-tier2-runin-g2-ws/ep23 (new, L 180000) | new file |
| 23 | `ep23_w31g2_150000` | screen | R4 j1 (5/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/ws031/2026-10-26-tier2-runin-g2-ws/ep23 (new, L 180000) | new file |
| 23 | `ep23_w31g2_180000` | screen | R4 j0 (6/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/ws031/2026-10-26-tier2-runin-g2-ws/ep23 (new, L 180000) | new file |
| 23 | `ep23_w31g2_90000` | screen | R4 j3 (3/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/ws031/2026-10-26-tier2-runin-g2-ws/ep23 (new, L 180000) | new file |
| 24 | `ep24_z5fill_170000` | screen | watcher best | k3fix | zx:2026-11-01-round7-5090-ckpts/2026-10-28-tier2-5090-fill/ep24 (unchanged, L 180000) | new file |
| 24 | `ep24_z5q_80000` | screen | watcher best | g2 | zx:2026-11-01-round7-5090-ckpts/2026-10-29-tier2-5090-queue/ep24 (unchanged, L 180000) | new file |
| 25 | `ep25_z5a5c_190000` | barred7 | watcher best | k3fix | zx:2026-11-01-round7-5090-ckpts/2026-10-28-tier2-runin-a5/cont/ep25 (unchanged, L 280500) | == ep25_z5a5c_190000, a round-7 phase-F participant of this episode (round-7 finalist not picked; its round-7 phase-F runs do not pass the round-8 test): never a finalist (PREREG_R8 3.3) |
| 25 | `ep25_w31g2_120000` | screen | R4 j2 (4/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/ws031/2026-10-26-tier2-runin-g2-ws/ep25 (new, L 180000) | new file |
| 25 | `ep25_w31g2_136000` | screen | watcher best | g2 | zx:2026-11-02-tier2-r8-handover/ws031/2026-10-26-tier2-runin-g2-ws/ep25 (new, L 180000) | new file |
| 25 | `ep25_w31g2_150000` | screen | R4 j1 (5/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/ws031/2026-10-26-tier2-runin-g2-ws/ep25 (new, L 180000) | new file |
| 25 | `ep25_w31g2_180000` | screen | R4 j0 (6/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/ws031/2026-10-26-tier2-runin-g2-ws/ep25 (new, L 180000) | new file |
| 25 | `ep25_w31g2_90000` | screen | R4 j3 (3/6 L) | g2 | zx:2026-11-02-tier2-r8-handover/ws031/2026-10-26-tier2-runin-g2-ws/ep25 (new, L 180000) | new file |
| 26 | `ep26_zxg2rest_113500` | screen | R4 j2 (4/6 L) | g2 | zx:2026-11-01-tier2-g2-rest/ep26 (trained further, L 170500) | new file |
| 26 | `ep26_zxg2rest_142000` | screen | R4 j1 (5/6 L) | g2 | zx:2026-11-01-tier2-g2-rest/ep26 (trained further, L 170500) | new file |
| 26 | `ep26_zxg2rest_160000` | screen | watcher best | g2 | zx:2026-11-01-tier2-g2-rest/ep26 (trained further, L 170500) | new file |
| 26 | `ep26_zxg2rest_170500` | screen | R4 j0 (6/6 L) | g2 | zx:2026-11-01-tier2-g2-rest/ep26 (trained further, L 170500) | new file |
| 26 | `ep26_zxg2rest_85500` | screen | R4 j3 (3/6 L) | g2 | zx:2026-11-01-tier2-g2-rest/ep26 (trained further, L 170500) | new file |
| 26 | `ep26_zxk3fix_160000` | screen | watcher best | k3fix | zx:2026-10-24-tier2-runin-k3fix-ep05-ep26/ep26 (unchanged, L 180000) | new file |
| 27 | `ep27_z5r2_40000` | reuse7 | watcher best | k3fix | zx:2026-11-01-round7-5090-ckpts/2026-10-31-tier2-5090-round2/ep27 (unchanged, L 51000) | == round-7 phase-S candidate ep27_z5r2_40000: its round-7 phase-S results and export are reused |
| 27 | `ep27_zxg2rest_35500` | screen | R4 j3 (3/6 L) | g2 | zx:2026-11-01-tier2-g2-rest/ep27 (trained further, L 71000) | new file |
| 27 | `ep27_zxg2rest_47500` | screen | R4 j2 (4/6 L) | g2 | zx:2026-11-01-tier2-g2-rest/ep27 (trained further, L 71000) | new file |
| 27 | `ep27_zxg2rest_59000` | screen | R4 j1 (5/6 L) | g2 | zx:2026-11-01-tier2-g2-rest/ep27 (trained further, L 71000) | new file |
| 27 | `ep27_zxg2rest_60000` | screen | watcher best | g2 | zx:2026-11-01-tier2-g2-rest/ep27 (trained further, L 71000) | new file |
| 27 | `ep27_zxg2rest_71000` | screen | R4 j0 (6/6 L) | g2 | zx:2026-11-01-tier2-g2-rest/ep27 (trained further, L 71000) | new file |
| 28 | `ep28_zxg2_120000` | reuse7 | watcher best | g2 | zx:2026-10-24-tier2-runin-g2/ep28 (unchanged, L 180000) | == round-7 phase-S candidate ep28_zxg2_120000: its round-7 phase-S results and export are reused |
| 29 | `ep29_zxg2rest_40000` | screen | R4 j3 (3/6 L), watcher best | g2 | zx:2026-11-01-tier2-g2-rest/ep29 (trained further, L 80000) | new file |
| 29 | `ep29_zxg2rest_53500` | screen | R4 j2 (4/6 L) | g2 | zx:2026-11-01-tier2-g2-rest/ep29 (trained further, L 80000) | new file |
| 29 | `ep29_zxg2rest_66500` | screen | R4 j1 (5/6 L) | g2 | zx:2026-11-01-tier2-g2-rest/ep29 (trained further, L 80000) | new file |
| 29 | `ep29_zxg2rest_80000` | screen | R4 j0 (6/6 L) | g2 | zx:2026-11-01-tier2-g2-rest/ep29 (trained further, L 80000) | new file |

### Runs

| run | host | ep | input | status vs round 7 | L (round 7) | commit | host parquet sha256 == this server's | picks |
|---|---|---|---|---|---|---|---|---|
| C3r2_tier2/ep00 | pro6000 | 00 | raw | unchanged | 41043 (41043) |  |  | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-24-tier2-runin-k3fix/ep00 | 5090 | 00 | k3fix | unchanged | 90000 (90000) | 82837c8 | not recorded | - |
| C3r2_tier2/ep01 | pro6000 | 01 | raw | unchanged | 44922 (44922) |  |  | - |
| C3r2_tier2_k3_stretch90k/ep01 | pro6000 | 01 | k3 | unchanged | 22500 (22500) |  |  | - |
| C3r2_tier2_k3fix/ep01_k3fix_s42 | pro6000 | 01 | k3fix | unchanged | 44922 (44922) |  |  | - |
| T2fix/T2L/ep01 | pro6000 | 01 | k3fix | unchanged | 90000 (90000) |  |  | - |
| zx:2026-10-24-tier2-runin-g2/ep01 | pro6000 | 01 | g2 | unchanged | 180000 (180000) | bc3b1bd |  | - |
| zx:2026-10-25-tier2-runin-hard-g2/ep01 | pro6000 | 01 | g2 | unchanged | 180000 (180000) | 82837c8 |  | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-23-tier2-helpfix-hard/cont/ep01 | 5090 | 01 | k3fix | unchanged | 180000 (180000) | 82837c8 | not recorded | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-23-tier2-helpfix-hard/ep01 | 5090 | 01 | k3fix | unchanged | 90000 (90000) | 82837c8 | not recorded | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-25-tier2-runin-hard-g2/cont/ep01 | 5090 | 01 | g2 | unchanged | 262500 (262500) | 82837c8 | not recorded | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-25-tier2-runin-hard-g2/ep01 | 5090 | 01 | g2 | unchanged | 180000 (180000) | 82837c8 | not recorded | - |
| C3r2_tier2/ep02 | pro6000 | 02 | raw | unchanged | 37409 (37409) |  |  | - |
| C3r2_tier2_k3fix/ep02_k3fix_s42 | pro6000 | 02 | k3fix | unchanged | 34093 (34093) |  |  | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-29-tier2-5090-queue/ep02 | 5090 | 02 | k3fix | unchanged | 180000 (180000) | 82837c8 | not recorded | - |
| C3r2_tier2/ep03 | pro6000 | 03 | raw | unchanged | 44922 (44922) |  |  | - |
| C3r2_tier2_k3_stretch90k/ep03 | pro6000 | 03 | k3 | unchanged | 19000 (19000) |  |  | - |
| C3r2_tier2_k3fix/ep03_k3fix_s42 | pro6000 | 03 | k3fix | unchanged | 27068 (27068) |  |  | - |
| C3r2_tier2_k3fix/ep03_raw_s7 | pro6000 | 03 | raw | unchanged | 44922 (44922) |  |  | - |
| zx:2026-11-02-tier2-r8-handover/5090/2026-11-01-tier2-g2-rest/ep03 | 5090 | 03 | g2 | new | 180000 (None) | 82837c8 | NO (bytes differ) | [180000, 150000, 120000, 90000] |
| C3r2_tier2/ep04 | pro6000 | 04 | raw | unchanged | 44922 (44922) |  |  | - |
| C3r2_tier2_k3fix/ep04_k3fix_s42 | pro6000 | 04 | k3fix | unchanged | 44922 (44922) |  |  | - |
| zx:2026-10-24-tier2-runin-g2/ep04 | pro6000 | 04 | g2 | unchanged | 450000 (450000) | bc3b1bd |  | - |
| zx:2026-10-24-tier2-runin-k3fix-ep04/ep04 | pro6000 | 04 | k3fix | unchanged | 12000 (12000) | bc3b1bd |  | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-23-tier2-helpfix-hard/cont/ep04 | 5090 | 04 | k3fix | unchanged | 180000 (180000) | 82837c8 | not recorded | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-23-tier2-helpfix-hard/ep04 | 5090 | 04 | k3fix | unchanged | 90000 (90000) | 82837c8 | not recorded | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-24-tier2-runin-g2/ep04 | 5090 | 04 | g2 | unchanged | 11000 (11000) | 82837c8 | not recorded | - |
| C3r2_tier2/ep05 | pro6000 | 05 | raw | unchanged | 44922 (44922) |  |  | - |
| zx:2026-10-24-tier2-runin-k3fix-ep05-ep26/ep05 | pro6000 | 05 | k3fix | unchanged | 180000 (180000) | bc3b1bd |  | - |
| zx:2026-11-02-tier2-r8-handover/ws032/2026-10-26-tier2-runin-g2-ws/cont/ep05 | ws032 | 05 | g2 | new | 360000 (None) | 82837c8569c2461f263c28fce406c1b517f40d93 | NO (bytes differ) | [360000, 300000, 240000, 182000] |
| C3r2_tier2/ep06 | pro6000 | 06 | raw | unchanged | 33894 (33894) |  |  | - |
| C3r2_tier2_k3fix/ep06_k3fix_s42 | pro6000 | 06 | k3fix | unchanged | 28420 (28420) |  |  | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-29-tier2-5090-queue/ep06 | 5090 | 06 | g2 | unchanged | 180000 (180000) | 82837c8 | not recorded | - |
| C3r2_tier2/ep07 | pro6000 | 07 | raw | unchanged | 44922 (44922) |  |  | - |
| C3r2_tier2_k3fix/ep07_k3fix_s42 | pro6000 | 07 | k3fix | unchanged | 18529 (18529) |  |  | - |
| zx:2026-11-02-tier2-r8-handover/5090/2026-11-01-tier2-g2-rest/ep07 | 5090 | 07 | g2 | new | 180000 (None) | 82837c8 | NO (bytes differ) | [180000, 150000, 120000, 90000] |
| C3r2_tier2/ep08 | pro6000 | 08 | raw | unchanged | 20597 (20597) |  |  | - |
| C3r2_tier2_k3easy/ep08 | pro6000 | 08 | k3 | unchanged | 17496 (17496) |  |  | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-29-tier2-5090-queue/ep08 | 5090 | 08 | g2 | unchanged | 180000 (180000) | 82837c8 | not recorded | - |
| C3r2_tier2/ep09 | pro6000 | 09 | raw | unchanged | 39328 (39328) |  |  | - |
| C3r2_tier2_k3fix/ep09_k3fix_s42 | pro6000 | 09 | k3fix | unchanged | 39449 (39449) |  |  | - |
| T2fix/T2B/ep09 | pro6000 | 09 | k3fix | unchanged | 54449 (54449) |  |  | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-24-tier2-runin-g2/ep09 | 5090 | 09 | g2 | unchanged | 180000 (180000) | 82837c8 | not recorded | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-31-tier2-5090-round2/ep09 | 5090 | 09 | k3fix | unchanged | 70000 (70000) | 82837c8 | not recorded | - |
| zx:2026-11-02-round7-robolab-ckpts/wsrbl015/2026-10-28-tier2-runin-k3fix-ws/ep09 | wsrbl015 | 09 | k3fix | new | 101000 (None) | 82837c8 | not recorded | [101000, 84000, 68000, 52000] |
| zx:2026-11-02-round7-robolab-ckpts/wsrbl015/2026-10-28-tier2-runin-k3fix-ws/resume/ep09 | wsrbl015 | 09 | k3fix | new | 180000 (None) | 82837c8 | not recorded | [180000, 150000, 120000, 102000] |
| C3r2_tier2/ep10 | pro6000 | 10 | raw | unchanged | 34267 (34267) |  |  | - |
| C3r2_tier2_k3fix/ep10_k3fix_s42 | pro6000 | 10 | k3fix | unchanged | 25319 (25319) |  |  | - |
| C3r2_tier2/ep11 | pro6000 | 11 | raw | unchanged | 28925 (28925) |  |  | - |
| C3r2_tier2_k3fix/ep11_k3fix_s42 | pro6000 | 11 | k3fix | unchanged | 29175 (29175) |  |  | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-31-tier2-5090-round2/ep11 | 5090 | 11 | k3fix | unchanged | 48000 (48000) | 82837c8 | not recorded | - |
| C3r2_tier2/ep12 | pro6000 | 12 | raw | unchanged | 37575 (37575) |  |  | - |
| C3r2_tier2_k3fix/ep12_k3fix_s42 | pro6000 | 12 | k3fix | unchanged | 23178 (23178) |  |  | - |
| C3r2_tier2/ep13 | pro6000 | 13 | raw | unchanged | 44922 (44922) |  |  | - |
| C3r2_tier2_k3fix/ep13_k3fix_s42 | pro6000 | 13 | k3fix | unchanged | 39254 (39254) |  |  | - |
| zx:2026-11-02-round7-robolab-ckpts/wsrbl015/2026-10-26-tier2-runin-g2-ws/ep13 | wsrbl015 | 13 | g2 | new | 180000 (None) | 82837c8 | not recorded | [180000, 150000, 120000, 90000] |
| zx:2026-11-02-tier2-r8-handover/ws031/2026-10-31-tier2-runin-ws/ep13 | ws031 | 13 | g2 | new | 180000 (None) | 82837c8 | NO (bytes differ) | [180000, 150000, 120000, 90000] |
| C3r2_tier2/ep14 | pro6000 | 14 | raw | unchanged | 44922 (44922) |  |  | - |
| C3r2_tier2_k3fix/ep14_k3fix_s42 | pro6000 | 14 | k3fix | unchanged | 44922 (44922) |  |  | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-28-tier2-runin-a5/cont/ep14 | 5090 | 14 | k3fix | unchanged | 263500 (263500) | 82837c8 | not recorded | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-28-tier2-runin-a5/ep14 | 5090 | 14 | k3fix | unchanged | 180000 (180000) | 82837c8 | not recorded | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-31-tier2-5090-round2/ep14 | 5090 | 14 | k3fix | unchanged | 64500 (64500) | 82837c8 | not recorded | - |
| C3r2_tier2/ep15 | pro6000 | 15 | raw | unchanged | 44922 (44922) |  |  | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-23-tier2-helpfix-hard/cont/ep15 | 5090 | 15 | raw | unchanged | 94000 (94000) | 82837c8 | not recorded | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-23-tier2-helpfix-hard/ep15 | 5090 | 15 | raw | unchanged | 90000 (90000) | 82837c8 | not recorded | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-24-tier2-runin-g2/ep15 | 5090 | 15 | g2 | unchanged | 93500 (93500) | 82837c8 | not recorded | - |
| zx:2026-11-02-tier2-r8-handover/5090/2026-10-31-tier2-5090-round2/ep15 | 5090 | 15 | g2 | trained further | 180000 (74000) | 82837c8 | NO (bytes differ) | [180000, 150000, 120000, 90000] |
| C3r2_tier2/ep16 | pro6000 | 16 | raw | unchanged | 31268 (31268) |  |  | - |
| C3r2_tier2_k3fix/ep16_k3fix_s42 | pro6000 | 16 | k3fix | unchanged | 32063 (32063) |  |  | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-28-tier2-runin-a5/ep16 | 5090 | 16 | g2 | unchanged | 180000 (180000) | 82837c8 | not recorded | - |
| C3r2_tier2/ep17 | pro6000 | 17 | raw | unchanged | 31807 (31807) |  |  | - |
| C3r2_tier2_k3fix/ep17_k3fix_s42 | pro6000 | 17 | k3fix | unchanged | 29845 (29845) |  |  | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-31-tier2-5090-round2/ep17 | 5090 | 17 | k3fix | unchanged | 61500 (61500) | 82837c8 | not recorded | - |
| zx:2026-11-02-tier2-r8-handover/5090/2026-11-01-tier2-g2-rest/ep17 | 5090 | 17 | g2 | new | 180000 (None) | 82837c8 | NO (bytes differ) | [180000, 150000, 120000, 90000] |
| C3r2_tier2/ep18 | pro6000 | 18 | raw | unchanged | 44922 (44922) |  |  | - |
| C3r2_tier2_k3fix/ep18_k3fix_s42 | pro6000 | 18 | k3fix | unchanged | 18911 (18911) |  |  | - |
| C3r2_tier2/ep19 | pro6000 | 19 | raw | unchanged | 44922 (44922) |  |  | - |
| zx:2026-10-24-tier2-runin-g2/ep19 | pro6000 | 19 | g2 | unchanged | 180000 (180000) | bc3b1bd |  | - |
| zx:2026-10-25-tier2-runin-hard-g2/ep19 | pro6000 | 19 | g2 | unchanged | 180000 (180000) | 82837c8 |  | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-23-tier2-helpfix-hard/cont/ep19 | 5090 | 19 | raw | unchanged | 180000 (180000) | 82837c8 | not recorded | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-23-tier2-helpfix-hard/ep19 | 5090 | 19 | raw | unchanged | 90000 (90000) | 82837c8 | not recorded | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-25-tier2-runin-hard-g2/cont/ep19 | 5090 | 19 | g2 | unchanged | 224000 (224000) | 82837c8 | not recorded | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-25-tier2-runin-hard-g2/cont2/ep19 | 5090 | 19 | g2 | unchanged | 360000 (360000) | 82837c8 | not recorded | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-25-tier2-runin-hard-g2/ep19 | 5090 | 19 | g2 | unchanged | 180000 (180000) | 82837c8 | not recorded | - |
| C3r2_tier2/ep20 | pro6000 | 20 | raw | unchanged | 19034 (19034) |  |  | - |
| C3r2_tier2_k3easy/ep20 | pro6000 | 20 | k3 | unchanged | 18450 (18450) |  |  | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-28-tier2-5090-fill/ep20 | 5090 | 20 | k3fix | unchanged | 180000 (180000) | 82837c8 | not recorded | - |
| zx:2026-11-01-tier2-g2-rest/ep20 | pro6000 | 20 | g2 | trained further | 92000 (57500) | bc3b1bd |  | [92000, 76500, 61500, 46000] |
| C3r2_tier2/ep21 | pro6000 | 21 | raw | unchanged | 44922 (44922) |  |  | - |
| C3r2_tier2_k3fix/ep21_k3fix_s42 | pro6000 | 21 | k3fix | unchanged | 44922 (44922) |  |  | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-24-tier2-runin-g2/cont/ep21 | 5090 | 21 | g2 | unchanged | 256000 (256000) | 82837c8 | not recorded | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-24-tier2-runin-g2/ep21 | 5090 | 21 | g2 | unchanged | 180000 (180000) | 82837c8 | not recorded | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-28-tier2-5090-fill/cont/ep21 | 5090 | 21 | g2 | unchanged | 282500 (282500) | 82837c8 | not recorded | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-28-tier2-5090-fill/ep21 | 5090 | 21 | g2 | unchanged | 180000 (180000) | 82837c8 | not recorded | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-31-tier2-5090-round2/ep21 | 5090 | 21 | k3fix | unchanged | 69500 (69500) | 82837c8 | not recorded | - |
| C3r2_tier2/ep22 | pro6000 | 22 | raw | unchanged | 44922 (44922) |  |  | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-24-tier2-runin-k3fix/ep22 | 5090 | 22 | k3fix | unchanged | 90000 (90000) | 82837c8 | not recorded | - |
| zx:2026-11-01-tier2-g2-rest/ep22 | pro6000 | 22 | g2 | trained further | 86500 (52500) | bc3b1bd |  | [86500, 72000, 57500, 43500] |
| C3r2_tier2/ep23 | pro6000 | 23 | raw | unchanged | 20023 (20023) |  |  | - |
| C3r2_tier2_k3easy/ep23 | pro6000 | 23 | k3 | unchanged | 19162 (19162) |  |  | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-23-tier2-helpfix-hard/ep23 | 5090 | 23 | k3fix | unchanged | 83500 (83500) | 82837c8 | not recorded | - |
| zx:2026-11-02-tier2-r8-handover/ws031/2026-10-26-tier2-runin-g2-ws/ep23 | ws031 | 23 | g2 | new | 180000 (None) | 82837c8 | NO (bytes differ) | [180000, 150000, 120000, 90000] |
| C3r2_tier2/ep24 | pro6000 | 24 | raw | unchanged | 44922 (44922) |  |  | - |
| C3r2_tier2_k3fix/ep24_k3fix_s42 | pro6000 | 24 | k3fix | unchanged | 20636 (20636) |  |  | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-28-tier2-5090-fill/ep24 | 5090 | 24 | k3fix | unchanged | 180000 (180000) | 82837c8 | not recorded | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-29-tier2-5090-queue/ep24 | 5090 | 24 | g2 | unchanged | 180000 (180000) | 82837c8 | not recorded | - |
| C3r2_tier2/ep25 | pro6000 | 25 | raw | unchanged | 44922 (44922) |  |  | - |
| C3r2_tier2_k3fix/ep25_k3fix_s42 | pro6000 | 25 | k3fix | unchanged | 42819 (42819) |  |  | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-28-tier2-runin-a5/cont/ep25 | 5090 | 25 | k3fix | unchanged | 280500 (280500) | 82837c8 | not recorded | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-28-tier2-runin-a5/ep25 | 5090 | 25 | k3fix | unchanged | 180000 (180000) | 82837c8 | not recorded | - |
| zx:2026-11-02-tier2-r8-handover/ws031/2026-10-26-tier2-runin-g2-ws/ep25 | ws031 | 25 | g2 | new | 180000 (None) | 82837c8 | NO (bytes differ) | [180000, 150000, 120000, 90000] |
| C3r2_tier2/ep26 | pro6000 | 26 | raw | unchanged | 34948 (34948) |  |  | - |
| C3r2_tier2_k3ca/ep26_k3ca_s42 | pro6000 | 26 | k3ca | unchanged | 25298 (25298) |  |  | - |
| zx:2026-10-24-tier2-runin-k3fix-ep05-ep26/ep26 | pro6000 | 26 | k3fix | unchanged | 180000 (180000) | bc3b1bd |  | - |
| zx:2026-11-01-tier2-g2-rest/ep26 | pro6000 | 26 | g2 | trained further | 170500 (106500) | bc3b1bd |  | [170500, 142000, 113500, 85500] |
| C3r2_tier2/ep27 | pro6000 | 27 | raw | unchanged | 44922 (44922) |  |  | - |
| C3r2_tier2_k3fix/ep27_k3fix_s42 | pro6000 | 27 | k3fix | unchanged | 20863 (20863) |  |  | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-31-tier2-5090-round2/ep27 | 5090 | 27 | k3fix | unchanged | 51000 (51000) | 82837c8 | not recorded | - |
| zx:2026-11-01-tier2-g2-rest/ep27 | pro6000 | 27 | g2 | trained further | 71000 (40500) | bc3b1bd |  | [71000, 59000, 47500, 35500] |
| C3r2_tier2/ep28 | pro6000 | 28 | raw | unchanged | 44922 (44922) |  |  | - |
| zx:2026-10-24-tier2-runin-g2/ep28 | pro6000 | 28 | g2 | unchanged | 180000 (180000) | bc3b1bd |  | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-23-tier2-helpfix-hard/cont/ep28 | 5090 | 28 | raw | unchanged | 180000 (180000) | 82837c8 | not recorded | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-23-tier2-helpfix-hard/ep28 | 5090 | 28 | raw | unchanged | 90000 (90000) | 82837c8 | not recorded | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-29-tier2-5090-queue/cont/ep28 | 5090 | 28 | g2 | unchanged | 271500 (271500) | 82837c8 | not recorded | - |
| zx:2026-11-01-round7-5090-ckpts/2026-10-29-tier2-5090-queue/ep28 | 5090 | 28 | g2 | unchanged | 180000 (180000) | 82837c8 | not recorded | - |
| C3r2_tier2/ep29 | pro6000 | 29 | raw | unchanged | 44922 (44922) |  |  | - |
| C3r2_tier2_k3fix/ep29_k3fix_s42 | pro6000 | 29 | k3fix | unchanged | 17358 (17358) |  |  | - |
| zx:2026-11-01-tier2-g2-rest/ep29 | pro6000 | 29 | g2 | trained further | 80000 (45000) | bc3b1bd |  | [80000, 66500, 53500, 40000] |

### Excluded directories and copies (not candidates)

| directory | reason |
|---|---|
| C3r2_tier2_k3_stretch90k/ep04 | last settled iteration 9000 < 10000 |
| T2fix/BASE/ep01 | not a training-run directory (output_dir /root/autodl-tmp/haoran.zhang/fc_runs/C3r2_tier2_k3fix/ep01_k3fix_s42) |
| T2fix/BASE/ep09 | not a training-run directory (output_dir /root/autodl-tmp/haoran.zhang/fc_runs/C3r2_tier2_k3fix/ep09_k3fix_s42) |
| T2fix/BASE/ep10 | not a training-run directory (output_dir /root/autodl-tmp/haoran.zhang/fc_runs/C3r2_tier2_k3fix/ep10_k3fix_s42) |
| T2fix/BASE/ep12 | not a training-run directory (output_dir /root/autodl-tmp/haoran.zhang/fc_runs/C3r2_tier2_k3fix/ep12_k3fix_s42) |
| T2fix/BASE/ep16 | not a training-run directory (output_dir /root/autodl-tmp/haoran.zhang/fc_runs/C3r2_tier2_k3fix/ep16_k3fix_s42) |
| T2fix/BASE/ep17 | not a training-run directory (output_dir /root/autodl-tmp/haoran.zhang/fc_runs/C3r2_tier2_k3fix/ep17_k3fix_s42) |
| T2fix/forks/ep01-L | not a training-run directory (output_dir /root/autodl-tmp/haoran.zhang/fc_runs/C3r2_tier2_k3fix/ep01_k3fix_s42) |
| T2fix/forks/ep09-B | not a training-run directory (output_dir /root/autodl-tmp/haoran.zhang/fc_runs/C3r2_tier2_k3fix/ep09_k3fix_s42) |
| T2fix/forks/ep10-B | not a training-run directory (output_dir /root/autodl-tmp/haoran.zhang/fc_runs/C3r2_tier2_k3fix/ep10_k3fix_s42) |
| T2fix/forks/ep12-B | not a training-run directory (output_dir /root/autodl-tmp/haoran.zhang/fc_runs/C3r2_tier2_k3fix/ep12_k3fix_s42) |
| T2fix/forks/ep16-B | not a training-run directory (output_dir /root/autodl-tmp/haoran.zhang/fc_runs/C3r2_tier2_k3fix/ep16_k3fix_s42) |
| T2fix/forks/ep17-B | not a training-run directory (output_dir /root/autodl-tmp/haoran.zhang/fc_runs/C3r2_tier2_k3fix/ep17_k3fix_s42) |
| zx:2026-10-14-patience-runin/combo/ep04 | training parquet not in a Tier-2 data root (/root/autodl-tmp/haoran.zhang/data/human_motion_data) |
| zx:2026-10-14-patience-runin/combo/ep19 | training parquet not in a Tier-2 data root (/root/autodl-tmp/haoran.zhang/data/human_motion_data) |
| zx:2026-10-14-patience-runin/patience/ep04 | training parquet not in a Tier-2 data root (/root/autodl-tmp/haoran.zhang/data/human_motion_data) |
| zx:2026-10-14-patience-runin/patience/ep19 | training parquet not in a Tier-2 data root (/root/autodl-tmp/haoran.zhang/data/human_motion_data) |
| zx:2026-10-14-patience-runin/runin/ep01 | training parquet not in a Tier-2 data root (/root/autodl-tmp/haoran.zhang/data/human_motion_data) |
| zx:2026-10-14-patience-runin/runin/ep04 | training parquet not in a Tier-2 data root (/root/autodl-tmp/haoran.zhang/data/human_motion_data) |
| zx:2026-10-14-patience-runin/runin/ep19 | training parquet not in a Tier-2 data root (/root/autodl-tmp/haoran.zhang/data/human_motion_data) |
| zx:2026-10-14-patience-runin/runin_cont/ep01 | training parquet not in a Tier-2 data root (/root/autodl-tmp/haoran.zhang/data/human_motion_data) |
| zx:2026-10-14-patience-runin/runin_cont/ep04 | training parquet not in a Tier-2 data root (/root/autodl-tmp/haoran.zhang/data/human_motion_data) |
| zx:2026-10-14-patience-runin/runin_cont/ep19 | training parquet not in a Tier-2 data root (/root/autodl-tmp/haoran.zhang/data/human_motion_data) |
| zx:2026-10-14-patience-runin/runin_cont2/ep01 | training parquet not in a Tier-2 data root (/root/autodl-tmp/haoran.zhang/data/human_motion_data) |
| zx:2026-10-14-patience-runin/runin_cont2/ep04 | training parquet not in a Tier-2 data root (/root/autodl-tmp/haoran.zhang/data/human_motion_data) |
| zx:2026-10-14-patience-runin/runin_cont3/ep04 | training parquet not in a Tier-2 data root (/root/autodl-tmp/haoran.zhang/data/human_motion_data) |
| zx:2026-10-14-patience-runin/runin_cont4/ep04 | training parquet not in a Tier-2 data root (/root/autodl-tmp/haoran.zhang/data/human_motion_data) |
| zx:2026-10-14-patience-runin/runin_cont_b/ep14 | training parquet not in a Tier-2 data root (/root/autodl-tmp/haoran.zhang/data/human_motion_data) |
| zx:2026-10-14-patience-runin/runin_cont_b/ep28 | training parquet not in a Tier-2 data root (/root/autodl-tmp/haoran.zhang/data/human_motion_data) |
| zx:2026-10-14-patience-runin/runin_p20/ep01 | training parquet not in a Tier-2 data root (/root/autodl-tmp/haoran.zhang/data/human_motion_data) |
| zx:2026-10-14-patience-runin/runin_p20/ep04 | training parquet not in a Tier-2 data root (/root/autodl-tmp/haoran.zhang/data/human_motion_data) |
| zx:2026-10-14-psi-worst3/ep01 | training parquet not in a Tier-2 data root (/root/autodl-tmp/haoran.zhang/data/human_motion_data) |
| zx:2026-10-14-psi-worst3/ep04 | training parquet not in a Tier-2 data root (/root/autodl-tmp/haoran.zhang/data/human_motion_data) |
| zx:2026-10-14-psi-worst3/ep19 | training parquet not in a Tier-2 data root (/root/autodl-tmp/haoran.zhang/data/human_motion_data) |
| zx:2026-10-21-help-force-fix/helpfix/ep04 | training parquet not in a Tier-2 data root (/root/autodl-tmp/haoran.zhang/data/human_motion_data) |
| zx:2026-10-21-help-force-fix/helpfix_hard/ep04 | training parquet not in a Tier-2 data root (/root/autodl-tmp/haoran.zhang/data/human_motion_data) |
| zx:2026-10-21-help-force-fix/helpfix_hard_cont/ep04 | training parquet not in a Tier-2 data root (/root/autodl-tmp/haoran.zhang/data/human_motion_data) |
| zx:2026-10-21-help-force-fix/runin_hard/ep04 | training parquet not in a Tier-2 data root (/root/autodl-tmp/haoran.zhang/data/human_motion_data) |
| zx:2026-10-21-help-force-fix/runin_hard_cont/ep04 | training parquet not in a Tier-2 data root (/root/autodl-tmp/haoran.zhang/data/human_motion_data) |
| zx:2026-10-28-tier2-runin-a5/ep16 | last settled iteration 1000 < 10000 |
| zx:2026-10-28-tier2-runin-a5/ep25 | last settled iteration 500 < 10000 |
| zx:2026-11-01-round7-5090-ckpts/2026-10-31-tier2-5090-round2/ep15 | duplicate copy of the same run (kept: /root/autodl-tmp/zuxing.lu/runs/2026-11-02-tier2-r8-handover/5090/2026-10-31-tier2-5090-round2/ep15, L 180000) |
| zx:2026-11-02-tier2-r8-handover/ws032_aborted_stream/2026-10-26-tier2-runin-g2-ws/ep05 | no checkpoint present, settled and listed in its checksum file at T_c |
| zx:2026-10-24-5090-best/ (not walked) | single-checkpoint copy bundles (not training-run directories; R1) |
| zx:2026-10-28-tier2-retest/ (not walked) | retest / single-checkpoint copies (not training-run directories; R1) |
| zx:2026-11-01-round7-5090-ckpts/ (not walked) | 5090 run copies: resolved by R1b from MANIFEST.tsv |
| zx:2026-11-02-round7-robolab-ckpts/ (not walked) | copies of training runs from other hosts: resolved as copy runs (PREREG_R8 2.3) |
| zx:2026-11-02-tier2-g2-ownsel/ (not walked) | selection / evaluation bundle with checkpoint copies (not training-run directories; R1) |
| zx:2026-11-02-tier2-g2-reports-eval/ (not walked) | report / evaluation bundle (not a training-run directory); never entered |
| zx:2026-11-02-tier2-r8-handover/ (not walked) | copies of training runs from other hosts: resolved as copy runs (PREREG_R8 2.3) |

### Watcher files (own-input columns only)

| run | episode | status | file | rows ok by T_c | last ok it | pick | why |
|---|---|---|---|---:|---:|---:|---|
| zx:2026-11-01-round7-5090-ckpts/2026-10-24-tier2-runin-k3fix/ep00 | 00 | unchanged | zx:2026-11-02-tier2-g2-ownsel/inputs_5090/2026-10-24-tier2-runin-k3fix/eval_ep00/metrics.csv (copy in runs/2026-11-02-tier2-g2-ownsel/inputs_5090 (fallback)) | 45 | 90000 | 90000 | argmax add_auc over the eligible rows |
| zx:2026-10-24-tier2-runin-g2/ep01 | 01 | unchanged | zx:2026-10-24-tier2-runin-g2/eval_ep01/metrics.csv (next to the run) | 88 | 180000 | 160000 | argmax add_auc over the eligible rows |
| zx:2026-10-25-tier2-runin-hard-g2/ep01 | 01 | unchanged | zx:2026-10-25-tier2-runin-hard-g2/eval_ep01/metrics.csv (next to the run) | 90 | 180000 | 130000 | argmax add_auc over the eligible rows |
| zx:2026-11-01-round7-5090-ckpts/2026-10-23-tier2-helpfix-hard/cont/ep01 | 01 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-01-round7-5090-ckpts/2026-10-23-tier2-helpfix-hard/ep01 | 01 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-01-round7-5090-ckpts/2026-10-25-tier2-runin-hard-g2/cont/ep01 | 01 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-01-round7-5090-ckpts/2026-10-25-tier2-runin-hard-g2/ep01 | 01 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-01-round7-5090-ckpts/2026-10-29-tier2-5090-queue/ep02 | 02 | unchanged | zx:2026-11-02-tier2-g2-ownsel/inputs_5090/2026-10-29-tier2-5090-queue/eval_ep02/metrics.csv (copy in runs/2026-11-02-tier2-g2-ownsel/inputs_5090 (fallback)) | 90 | 180000 | 180000 | argmax add_auc over the eligible rows |
| zx:2026-11-02-tier2-r8-handover/5090/2026-11-01-tier2-g2-rest/ep03 | 03 | new | zx:2026-11-02-tier2-r8-handover/5090/2026-11-01-tier2-g2-rest/eval_ep03/metrics.csv (next to the run) | 81 | 180000 | 110000 | argmax add_auc over the eligible rows |
| zx:2026-10-24-tier2-runin-g2/ep04 | 04 | unchanged | zx:2026-10-24-tier2-runin-g2/eval_ep04/metrics.csv (next to the run) | 181 | 448000 | 448000 | argmax add_auc over the eligible rows |
| zx:2026-10-24-tier2-runin-k3fix-ep04/ep04 | 04 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-01-round7-5090-ckpts/2026-10-23-tier2-helpfix-hard/cont/ep04 | 04 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-01-round7-5090-ckpts/2026-10-23-tier2-helpfix-hard/ep04 | 04 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-01-round7-5090-ckpts/2026-10-24-tier2-runin-g2/ep04 | 04 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-10-24-tier2-runin-k3fix-ep05-ep26/ep05 | 05 | unchanged | zx:2026-10-24-tier2-runin-k3fix-ep05-ep26/eval_ep05/metrics.csv (next to the run) | 90 | 180000 | 170000 | argmax add_auc over the eligible rows |
| zx:2026-11-02-tier2-r8-handover/ws032/2026-10-26-tier2-runin-g2-ws/cont/ep05 | 05 | new | zx:2026-11-02-tier2-r8-handover/ws032/2026-10-26-tier2-runin-g2-ws/cont/eval_ep05/metrics.csv (next to the run) | 83 | 360000 | 360000 | argmax add_auc over the eligible rows |
| zx:2026-11-01-round7-5090-ckpts/2026-10-29-tier2-5090-queue/ep06 | 06 | unchanged | zx:2026-11-02-tier2-g2-ownsel/inputs_5090/2026-10-29-tier2-5090-queue/eval_ep06/metrics.csv (copy in runs/2026-11-02-tier2-g2-ownsel/inputs_5090 (fallback)) | 90 | 180000 | 80000 | argmax add_auc over the eligible rows |
| zx:2026-11-02-tier2-r8-handover/5090/2026-11-01-tier2-g2-rest/ep07 | 07 | new | zx:2026-11-02-tier2-r8-handover/5090/2026-11-01-tier2-g2-rest/eval_ep07/metrics.csv (next to the run) | 90 | 180000 | 180000 | argmax add_auc over the eligible rows |
| zx:2026-11-01-round7-5090-ckpts/2026-10-29-tier2-5090-queue/ep08 | 08 | unchanged | zx:2026-11-02-tier2-g2-ownsel/inputs_5090/2026-10-29-tier2-5090-queue/eval_ep08/metrics.csv (copy in runs/2026-11-02-tier2-g2-ownsel/inputs_5090 (fallback)) | 90 | 180000 | 100000 | argmax add_auc over the eligible rows |
| zx:2026-11-01-round7-5090-ckpts/2026-10-24-tier2-runin-g2/ep09 | 09 | unchanged | zx:2026-11-02-tier2-g2-ownsel/inputs_5090/2026-10-24-tier2-runin-g2/eval_ep09/metrics.csv (copy in runs/2026-11-02-tier2-g2-ownsel/inputs_5090 (fallback)) | 90 | 180000 | 180000 | argmax add_auc over the eligible rows |
| zx:2026-11-01-round7-5090-ckpts/2026-10-31-tier2-5090-round2/ep09 | 09 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-02-round7-robolab-ckpts/wsrbl015/2026-10-28-tier2-runin-k3fix-ws/ep09 | 09 | new | zx:2026-11-02-round7-robolab-ckpts/wsrbl015/2026-10-28-tier2-runin-k3fix-ws/eval_ep09/metrics.csv (next to the run) | 50 | 100000 | 100000 | argmax add_auc over the eligible rows |
| zx:2026-11-02-round7-robolab-ckpts/wsrbl015/2026-10-28-tier2-runin-k3fix-ws/resume/ep09 | 09 | new | zx:2026-11-02-round7-robolab-ckpts/wsrbl015/2026-10-28-tier2-runin-k3fix-ws/resume/eval_ep09/metrics.csv (next to the run) | 34 | 180000 | 180000 | argmax add_auc over the eligible rows |
| zx:2026-11-01-round7-5090-ckpts/2026-10-31-tier2-5090-round2/ep11 | 11 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-02-round7-robolab-ckpts/wsrbl015/2026-10-26-tier2-runin-g2-ws/ep13 | 13 | new | zx:2026-11-02-round7-robolab-ckpts/wsrbl015/2026-10-26-tier2-runin-g2-ws/eval_ep13/metrics.csv (next to the run) | 90 | 180000 | 140000 | argmax add_auc over the eligible rows |
| zx:2026-11-02-tier2-r8-handover/ws031/2026-10-31-tier2-runin-ws/ep13 | 13 | new | zx:2026-11-02-tier2-r8-handover/ws031/2026-10-31-tier2-runin-ws/eval_ep13/metrics.csv (next to the run) | 89 | 180000 | 110000 | argmax add_auc over the eligible rows |
| zx:2026-11-01-round7-5090-ckpts/2026-10-28-tier2-runin-a5/cont/ep14 | 14 | unchanged | zx:2026-11-02-tier2-g2-ownsel/inputs_5090/2026-10-28-tier2-runin-a5/cont/eval_ep14/metrics.csv (copy in runs/2026-11-02-tier2-g2-ownsel/inputs_5090 (fallback)) | 48 | 276000 | 250000 | argmax add_auc over the eligible rows |
| zx:2026-11-01-round7-5090-ckpts/2026-10-28-tier2-runin-a5/ep14 | 14 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-01-round7-5090-ckpts/2026-10-31-tier2-5090-round2/ep14 | 14 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-01-round7-5090-ckpts/2026-10-23-tier2-helpfix-hard/cont/ep15 | 15 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-01-round7-5090-ckpts/2026-10-23-tier2-helpfix-hard/ep15 | 15 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-01-round7-5090-ckpts/2026-10-24-tier2-runin-g2/ep15 | 15 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-02-tier2-r8-handover/5090/2026-10-31-tier2-5090-round2/ep15 | 15 | trained further | zx:2026-11-02-tier2-r8-handover/5090/2026-10-31-tier2-5090-round2/eval_ep15/metrics.csv (next to the run) | 90 | 180000 | 170000 | argmax add_auc over the eligible rows |
| zx:2026-11-01-round7-5090-ckpts/2026-10-28-tier2-runin-a5/ep16 | 16 | unchanged | zx:2026-11-02-tier2-g2-ownsel/inputs_5090/2026-10-28-tier2-runin-a5/eval_ep16/metrics.csv (copy in runs/2026-11-02-tier2-g2-ownsel/inputs_5090 (fallback)) | 90 | 180000 | 140000 | argmax add_auc over the eligible rows |
| zx:2026-11-01-round7-5090-ckpts/2026-10-31-tier2-5090-round2/ep17 | 17 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-02-tier2-r8-handover/5090/2026-11-01-tier2-g2-rest/ep17 | 17 | new | zx:2026-11-02-tier2-r8-handover/5090/2026-11-01-tier2-g2-rest/eval_ep17/metrics.csv (next to the run) | 90 | 180000 | 150000 | argmax add_auc over the eligible rows |
| zx:2026-10-24-tier2-runin-g2/ep19 | 19 | unchanged | zx:2026-10-24-tier2-runin-g2/eval_ep19/metrics.csv (next to the run) | 78 | 180000 | 160000 | argmax add_auc over the eligible rows |
| zx:2026-10-25-tier2-runin-hard-g2/ep19 | 19 | unchanged | zx:2026-10-25-tier2-runin-hard-g2/eval_ep19/metrics.csv (next to the run) | 86 | 180000 | 180000 | argmax add_auc over the eligible rows |
| zx:2026-11-01-round7-5090-ckpts/2026-10-23-tier2-helpfix-hard/cont/ep19 | 19 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-01-round7-5090-ckpts/2026-10-23-tier2-helpfix-hard/ep19 | 19 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-01-round7-5090-ckpts/2026-10-25-tier2-runin-hard-g2/cont/ep19 | 19 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-01-round7-5090-ckpts/2026-10-25-tier2-runin-hard-g2/cont2/ep19 | 19 | unchanged | zx:2026-11-02-tier2-g2-ownsel/inputs_5090/2026-10-25-tier2-runin-hard-g2/cont2/eval_ep19/metrics.csv (copy in runs/2026-11-02-tier2-g2-ownsel/inputs_5090 (fallback)) | 31 | 354000 | 340000 | argmax add_auc over the eligible rows |
| zx:2026-11-01-round7-5090-ckpts/2026-10-25-tier2-runin-hard-g2/ep19 | 19 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-01-round7-5090-ckpts/2026-10-28-tier2-5090-fill/ep20 | 20 | unchanged | zx:2026-11-02-tier2-g2-ownsel/inputs_5090/2026-10-28-tier2-5090-fill/eval_ep20/metrics.csv (copy in runs/2026-11-02-tier2-g2-ownsel/inputs_5090 (fallback)) | 90 | 180000 | 110000 | argmax add_auc over the eligible rows |
| zx:2026-11-01-tier2-g2-rest/ep20 | 20 | trained further | zx:2026-11-01-tier2-g2-rest/eval_ep20/metrics.csv (next to the run) | 46 | 92000 | 80000 | argmax add_auc over the eligible rows |
| zx:2026-11-01-round7-5090-ckpts/2026-10-24-tier2-runin-g2/cont/ep21 | 21 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-01-round7-5090-ckpts/2026-10-24-tier2-runin-g2/ep21 | 21 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-01-round7-5090-ckpts/2026-10-28-tier2-5090-fill/cont/ep21 | 21 | unchanged | zx:2026-11-02-tier2-g2-ownsel/inputs_5090/2026-10-28-tier2-5090-fill/cont/eval_ep21/metrics.csv (copy in runs/2026-11-02-tier2-g2-ownsel/inputs_5090 (fallback)) | 81 | 360000 | 280000 | argmax add_auc over the eligible rows |
| zx:2026-11-01-round7-5090-ckpts/2026-10-28-tier2-5090-fill/ep21 | 21 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-01-round7-5090-ckpts/2026-10-31-tier2-5090-round2/ep21 | 21 | unchanged | zx:2026-11-02-tier2-g2-ownsel/inputs_5090/2026-10-31-tier2-5090-round2/eval_ep21/metrics.csv (copy in runs/2026-11-02-tier2-g2-ownsel/inputs_5090 (fallback)) | 40 | 80000 | 60000 | argmax add_auc over the eligible rows |
| zx:2026-11-01-round7-5090-ckpts/2026-10-24-tier2-runin-k3fix/ep22 | 22 | unchanged | zx:2026-11-02-tier2-g2-ownsel/inputs_5090/2026-10-24-tier2-runin-k3fix/eval_ep22/metrics.csv (copy in runs/2026-11-02-tier2-g2-ownsel/inputs_5090 (fallback)) | 45 | 90000 | 90000 | argmax add_auc over the eligible rows |
| zx:2026-11-01-tier2-g2-rest/ep22 | 22 | trained further | zx:2026-11-01-tier2-g2-rest/eval_ep22/metrics.csv (next to the run) | 43 | 86000 | 80000 | argmax add_auc over the eligible rows |
| zx:2026-11-01-round7-5090-ckpts/2026-10-23-tier2-helpfix-hard/ep23 | 23 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-02-tier2-r8-handover/ws031/2026-10-26-tier2-runin-g2-ws/ep23 | 23 | new | zx:2026-11-02-tier2-r8-handover/ws031/2026-10-26-tier2-runin-g2-ws/eval_ep23/metrics.csv (next to the run) | 90 | 180000 | 120000 | argmax add_auc over the eligible rows |
| zx:2026-11-01-round7-5090-ckpts/2026-10-28-tier2-5090-fill/ep24 | 24 | unchanged | zx:2026-11-02-tier2-g2-ownsel/inputs_5090/2026-10-28-tier2-5090-fill/eval_ep24/metrics.csv (copy in runs/2026-11-02-tier2-g2-ownsel/inputs_5090 (fallback)) | 90 | 180000 | 170000 | argmax add_auc over the eligible rows |
| zx:2026-11-01-round7-5090-ckpts/2026-10-29-tier2-5090-queue/ep24 | 24 | unchanged | zx:2026-11-02-tier2-g2-ownsel/inputs_5090/2026-10-29-tier2-5090-queue/eval_ep24/metrics.csv (copy in runs/2026-11-02-tier2-g2-ownsel/inputs_5090 (fallback)) | 90 | 180000 | 80000 | argmax add_auc over the eligible rows |
| zx:2026-11-01-round7-5090-ckpts/2026-10-28-tier2-runin-a5/cont/ep25 | 25 | unchanged | zx:2026-11-02-tier2-g2-ownsel/inputs_5090/2026-10-28-tier2-runin-a5/cont/eval_ep25/metrics.csv (copy in runs/2026-11-02-tier2-g2-ownsel/inputs_5090 (fallback)) | 5 | 242000 | 190000 | argmax add_auc over the eligible rows |
| zx:2026-11-01-round7-5090-ckpts/2026-10-28-tier2-runin-a5/ep25 | 25 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-02-tier2-r8-handover/ws031/2026-10-26-tier2-runin-g2-ws/ep25 | 25 | new | zx:2026-11-02-tier2-r8-handover/ws031/2026-10-26-tier2-runin-g2-ws/eval_ep25/metrics.csv (next to the run) | 58 | 136000 | 136000 | argmax add_auc over the eligible rows |
| zx:2026-10-24-tier2-runin-k3fix-ep05-ep26/ep26 | 26 | unchanged | zx:2026-10-24-tier2-runin-k3fix-ep05-ep26/eval_ep26/metrics.csv (next to the run) | 90 | 180000 | 160000 | argmax add_auc over the eligible rows |
| zx:2026-11-01-tier2-g2-rest/ep26 | 26 | trained further | zx:2026-11-01-tier2-g2-rest/eval_ep26/metrics.csv (next to the run) | 85 | 170000 | 160000 | argmax add_auc over the eligible rows |
| zx:2026-11-01-round7-5090-ckpts/2026-10-31-tier2-5090-round2/ep27 | 27 | unchanged | zx:2026-11-02-tier2-g2-ownsel/inputs_5090/2026-10-31-tier2-5090-round2/eval_ep27/metrics.csv (copy in runs/2026-11-02-tier2-g2-ownsel/inputs_5090 (fallback)) | 31 | 62000 | 40000 | argmax add_auc over the eligible rows |
| zx:2026-11-01-tier2-g2-rest/ep27 | 27 | trained further | zx:2026-11-01-tier2-g2-rest/eval_ep27/metrics.csv (next to the run) | 34 | 70000 | 60000 | argmax add_auc over the eligible rows |
| zx:2026-10-24-tier2-runin-g2/ep28 | 28 | unchanged | zx:2026-10-24-tier2-runin-g2/eval_ep28/metrics.csv (next to the run) | 85 | 178000 | 120000 | argmax add_auc over the eligible rows |
| zx:2026-11-01-round7-5090-ckpts/2026-10-23-tier2-helpfix-hard/cont/ep28 | 28 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-01-round7-5090-ckpts/2026-10-23-tier2-helpfix-hard/ep28 | 28 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-01-round7-5090-ckpts/2026-10-29-tier2-5090-queue/cont/ep28 | 28 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-01-round7-5090-ckpts/2026-10-29-tier2-5090-queue/ep28 | 28 | unchanged |  (no watcher file) |  |  | - | no watcher file |
| zx:2026-11-01-tier2-g2-rest/ep29 | 29 | trained further | zx:2026-11-01-tier2-g2-rest/eval_ep29/metrics.csv (next to the run) | 39 | 78000 | 40000 | argmax add_auc over the eligible rows |
