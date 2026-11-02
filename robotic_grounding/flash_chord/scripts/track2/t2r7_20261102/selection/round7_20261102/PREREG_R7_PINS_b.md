# t2_consensus_r7_20261102: PREREG_R7_PINS_b (step R4 scorer; pins and implementation notes only)

**When.** Written 2026-11-02 from about 07:12 CST, while phase S runs (driver pid 560180, launched 06:45:04 with
`bash scripts/t2cons_r7_eval_launch.sh --phase screen`), and BEFORE any round-7 own-input gate score, consensus score or
selection:
- `scores_s7/`, `scores_f7/`, `inputs_r7/ROLLOUTS_S7.sha256`, `final_r7.json` and `SELECTION_S7.sha256` do not exist.
- Runs of the scorer before this file: `selftest` (synthetic values plus the pinned incumbent / reference / pool
  records), one trial selftest into `dryrun/` (version f81e6308, not pinned), and one dry `check --phase S` into
  `dryrun/` (version f81e6308; it stopped at "phase S evaluations not finished").
- A mock harness (`tmp_analysis/r7_mock/mock_select.py`) ran `select --phase S` and `select --phase F` of this script
  on SYNTHETIC values, with every read and write patched to memory; it showed both selections run end to end (barred
  pairs, STRICT top 2, unconfirmed fallback, R7-CLOSEST, locks last). After it, the scorer gained three record fields
  per job (training / source checkpoint path, selecting round), which no rule reads; the harness re-ran with the same
  result.
- Round-7 values used when writing this file: the driver's progress counters (heartbeat) and the dry-run values of
  `DRYRUN_R7.md`; also, from reading the record formats, the frozen consensus values in `incumbents_r7.json` /
  `r6refs_r7.json` (incumbents' and references' earlier runs) and round 6's `selection_r6.json` m_F of ep04 / ep11.

**Scope.** Pins and implementation notes only (PREREG_R7 10: "PINS addenda may only add pins and notes"). Nothing here
changes a rule of `PREREG_R7.md` (749649a79ca5616b...). This is the addendum PREREG_R7 5.2 names.

## 1. The scorer: `scripts/t2cons_r7_consensus.py` (new file, mode 444)

1. **Pinned code (K1), unchanged from round 6.** It compiles round 2's scorer `scripts/t2cons_r2_consensus.py`
   (ddee0b8b...) from its verified bytes; that module compiles `scripts/t2cons_a1_gate.py` (a29c997b...), whose module
   code loads `scripts/t2cons_consensus.py` (e1b943b1..., re-checked after loading). The pinned round-1 functions
   `rollout_on_grid`, `metrics_vs`, `load_kit`, `load_refs`, `grid`, `stats`, `floor_count`, `integrity` /
   `integrity_diff`, `check_pins`, `hash_list`, `git_head` and the gate's `auc_seed` are called unchanged; per-run gate
   flags = round 2's `run_flags`, score columns = round 2's `SCORE_COLS`.
2. **Rules (K2)** implement PREREG_R7 5.3-5.6: `s_row` (5.3 eligibility, round 6's checks), `select_s` (23 incumbent
   episodes, round 6's 5.4 plus the barred pairs ep12_raw_37575 / ep13_raw_44922, which stay S-eligible but are never
   a finalist; the barred set is re-derived from `pool_r7.json` `r6_phase_f_verdict` and must equal the PREREG list),
   `select_strict` (7 STRICT episodes: top 2 S-eligible; else the (floor desc, cons_S desc) fallback among candidates
   with an ok export and 3 ok, unflagged, consensus-scored runs, marked `fallback_unconfirmed_S`; else halt),
   `decide_f` (round 6's 5.5, unchanged), `decide_strict` (rank 1 if confirmed, else rank 2 if confirmed, else among
   finalists with >= 8 ok, non-flagged runs: more SP=MP=1 runs, higher m_F, phase-S rank, `unconfirmed`; else halt),
   `trial_rows` / `closest_trial` (round 6's, unchanged: R7-CLOSEST) and `ref6_report` (Delta_B vs REF6, report only).
   The selftest (44 tests, PASS) covers round 6's 30 cases plus: barred candidates (incl. an 800-episode random
   re-derivation with barred jobs), STRICT top 2 / single / fallback / halt / ties and an independent 800-episode
   re-derivation of `select_strict`, every branch of `decide_strict` (confirmed rank 1, rank 2 by floor 7/10 or 9 runs,
   unconfirmed by floor / m_F / rank, 7-run finalist excluded, halt, single finalist), the paired report-only Delta_B,
   the extended guard, and the pinned 23-incumbent / 7-reference / barred / pool-role records.
3. **Path guard (K3).** Round 6's tokens + `tier2_eval.csv`, `tier2_runs.csv`, `summary.txt`, `tier2_best_scan`,
   `eval_gen_local`; the selftest shows every forbidden pattern refused and every rule input passed.
4. **Run sources (K4).** Phase S: the 154 new screen jobs read this round's `runs_s/` and `own_input_evals_s7.csv`
   (export = `exports/<job>/export.json`, source_sha256 == the evaluated file: the relocated file for 5090 jobs per
   `RELOCATION_R7.tsv`, else the pool sha256); the 245 reused candidates read round 6's `runs_s/` and
   `own_input_evals_s6.csv` (pinned d152c7b3), with every `run.json` sha256 == its `pool_r7.json` `r6_runs` record,
   round 6's `manifest_s6.json` row (pinned 0f056c99) and export record (sha256 == `pool_r7.json`), and every rollout ==
   round 6's `ROLLOUTS_S6.sha256` (pinned 1301c5c0). `ROLLOUTS_S7.sha256` lists both (new and reused), as PREREG_R7 5.1
   says. Evaluator values: each run's `run.json` metrics (full precision) == the driver's CSV row (|diff| <= 1e-6 for
   AUC / MPPE; SP-SR / MP-SR exactly), as round 6.
5. **Incumbents and references (K5).** The 23 rows of `incumbents_r7.json` must equal the PREREG_R7 4.1 table (job,
   input, number of frozen runs, selecting round); m_P_frozen = the pinned `stats()` mean of the frozen runs, which must
   equal the frozen scores CSV of the selecting round (round 1 `scores/consensus_scores.csv` 50cc6e94; round 6
   `scores_f6/consensus_scores_f6.csv` 74c9c885), that round's frozen job stats mean EXACTLY (9e255057 / 9eaac607), the
   resolver's field within 1e-12 and the table within 5e-7; for ep04 / ep11 also `selection_r6.json` m_F and pick
   (exact). The 7 references must equal the PREREG_R7 4.2 table and round 6's S6 composition; none is in the pool.
6. **Reuse reproduction (K6, PREREG_R7 3.2).** `gate --phase S` compares every reused run's gate row with round 6's
   `own_input_gate_s6.csv` (pinned 31fabb29): status and flags must be identical (else the gate fails); the numeric
   columns' largest difference is recorded. `score --phase S` compares every reused run with round 6's
   `consensus_scores_s6.csv` (pinned d32c1ed7): the scored (job, run) set, status, checkpoint, parquet, and AUC / SP-SR /
   MP-SR / MPPE / conversion residuals must be identical floats; the reused jobs scored must equal round 6's
   `scored_jobs`. `select --phase S` refuses unless this passed.
7. **Rollout lists.** `check --phase S|F` writes `inputs_r7/ROLLOUTS_{S7,F7}.sha256` once, only when the phase's driver
   heartbeat says `done`, no driver runs, `driver.pid` is gone and the whole pin chain holds (PREREG_R7 section 11,
   PINS, DRYRUN_R7.sha256, LOCK_R7.sha256, SNAPSHOT_R7, TRACKS_R7, RELOCATION_R7, pool / incumbents / references, the 9
   round-6 records, round 1's chain, integrity baseline). `gate`, `score` and `select` refuse without it.
8. **Outputs.** As the docstring says; `SELECTION_S7.sha256` and `SELECTION_R7.sha256` are written last, with paths in
   the driver's `_pin_path` spelling, so the driver's phase-F check (`final_r7.json` listed with its current sha256)
   reads them unchanged. `final_r7.json` finalists carry `job`, `episode`, `rank` (STRICT), `selection_mode`,
   `fallback_unconfirmed_S`, `pool_role`, `export_sha256`, `export_file` and the phase-S statistics.
9. **Halts.** A STRICT episode without a candidate with 3 ok, unflagged runs (5.4) or without a finalist with >= 8 ok,
   non-flagged phase-F runs (5.5) makes `select` print HALT and write nothing (an amendment decides).
10. **Run.** `conda_envs/v2dkit/bin/python -I -B scripts/t2cons_r7_consensus.py <cmd> --phase S|F` under `nice -n 10`
    with `OMP_NUM_THREADS=2` (as round 6, whose 750 phase-S runs it re-scores), CPU only (`CUDA_VISIBLE_DEVICES` empty),
    from a neutral working directory (/tmp).

## 2. Pins

pin scripts/t2cons_r7_consensus.py e7545dfa7fdff0681123d77e2dde3f65ed6a22ead8d4d12057d9b61af9e2708a
pin selftest_r7/selftest_r7.json 8b40106e31d79e5ccb780e96625068477d2e8ad0184afa15cac449857c92bb9d
pin dryrun/selftest_trial_r7c_1.json 7b9d216539c5824d3b8da4072228408a56cda953b2d762531a6b8b1a58857f20
pin dryrun/r7_consensus_check_S_20261102_070220.txt b2867b6b67b7320665d8b0e36dd07691c33b080da997d208fc74bfc292846b82
pin tmp_analysis/r7_mock/mock_select.py 4016f6dcbd8f6adeec975c53ad0fe3bc34b29198381dd31128f26107d6e603fe
pin scripts/t2cons_r2_consensus.py ddee0b8b6fa25029275b21a3dc82d15c36f4de71c0a61f2a1046869b01ec13aa
pin scripts/t2cons_a1_gate.py a29c997b7271e9299db109e97af5210f528484719210bb58685523b0ae0eb16b
pin scripts/t2cons_consensus.py e1b943b10fd1db13ab6f78e5bde7b3c6e6c48f7daeee6230c32c463d14ac41b4
pin scripts/t2cons_r6_consensus.py 9b65e612fb289cd8e7f7584ffb7c7ccb40e1d2de9235f0221a70a99c11382d31
pin PREREG_R7.md 749649a79ca5616b8ddbfc5124828c13da20f2e10e850a39647a74d83c939ec6
pin PREREG_R7_PINS.md 8378c5a418c9b1ece0ac8a1c518237666d4649cba211229b18d95d89bff14fce
pin DRYRUN_R7.md e7a6095b21b6c3a71eb3d7eafc8dfa2f7ff0d4c036e87ac7ce924541bc9a0981
pin LOCK_R7.sha256 cef6faeb111c78f7336011bea5c585022a433d54ea0e33c515f1a54e5fffa992
