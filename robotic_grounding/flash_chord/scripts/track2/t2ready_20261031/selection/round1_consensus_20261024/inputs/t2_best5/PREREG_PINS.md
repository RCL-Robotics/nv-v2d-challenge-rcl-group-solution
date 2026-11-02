# t2_best5_20261024: PREREG_PINS (PREREG section 6)

Written 2026-10-24 ~19:05 CST, after the dry run and the selftest and before any copy, export or evaluation. At writing
the folder holds `inventory/`, `PREREG.md`, `PREREG.sha256`, `dryrun/` (plans, previews, `selftest/` synthetic files) and
`preflight/` (the read-only hull checks). This file adds pins and implementation notes only; it changes no rule of
PREREG.md (sha256 d778cb29b878e87d3d9538da0c3617f5a13dc490924b2a26187e0d427461b704).

## Pins (sha256; the driver refuses to start if any differs)

pin scripts/t2best5_driver.py 24b9af39a2e2708641e7870447b12b6475c23aec9cd18ba71086768dd69cfc89
pin scripts/t2best5_launch.sh 6f4e42cbdc0c05593330f5d9e13806257269a230e50e4e1d830bc2e4b97701e0
pin dryrun/plan_20261024_190139.txt bacc6fc3d28e042cb0e18b70ca15f5ada36d8a098d63abca464575764bb6b4bc
pin dryrun/manifest_preview_20261024_190139.csv 26024fb2dbdb2275befa4b18f8637e548c216fff82fb5c4a5879a3fd1d61d6f6

## Dry run (`dryrun/plan_20261024_190139.txt`)

- Every pin and commit line of PREREG.md section 6 re-checked: all equal. Trees: fc_r2 HEAD 61fa5d3, nv-v2d_runin HEAD
  b39b69b, stock 8c778f9; eval_all.py 65729592 in both trees; kit evaluate_policy.py == stock == 160d24f5.
- Stage-A job list: 70 jobs, all `ok`. The set of (episode, iteration, checkpoint sha256[:8]) of the jobs EQUALS the
  set parsed from the PREREG section 1 table (70 pairs). Counts: 30 RAWREF, 16 K3fix + 3 K3 easy-clip CHOSEN,
  11 t2watch bests, 1 T2fix ep01-L (P), 7 Zuxing, 2 raw seed 7.
- Every checkpoint and training parquet re-hashed equal to inventory.json. Zuxing's 7 checkpoints equal his metrics.csv
  `source_sha256` (rows ok, stock_verified True, finished by 17:58:27). T2fix ep01-L 62000 equals its eval_ARM export
  report. The K3 easy-clip parquets (ep08/20/23) are byte-identical to the K3fix root's.
- Hull preflight (`preflight/hulls_20261024_190326.*`, `tmp_analysis/t2fix/cache_keys_t2.py`, CPU, read-only): 56
  distinct training parquets, 29 distinct CoACD keys, every key's file present in the stock runtime's convex cache
  (29 files). No job is `hull_missing`.
- GPUs at the dry run: 0, 1, 2, 5, 6, 7 usable (lock files present, no MPS, >= 32 GB free); 3 (no lock file) and 4
  never.

## Selftest (`python -B scripts/t2best5_driver.py selftest`): 0 problems

- check_export / check_eval accept the kept k3full ep01 K3fix final evaluation (read-only), and check_export refuses
  the same report at a tolerance other than the one it was made with.
- check_eval rejects a wrong training-parquet sha256 and a results JSON whose reference is another input.
- Refusal classifier: a finite max_rel_diff refusal is "tolerance"; a log_std refusal is not.
- Selection rule on synthetic records: +0.01 margin, SP/MP condition, same-parquet-only challengers, DIFF excluded,
  provisional only in SELECTED+P, INCOMPLETE CHOSEN kept, PENDING -> raw: all as PREREG section 2.
- Compositions: the mean of T_1..T_5 equals the average; T_min / T_max correct.

## Implementation notes (no rule change)

1. Metrics: the kit's `REPORT_METRIC_KEY` (v2dlb/report_schema.py), as t1best5 and t2watch: AUC = `add_auc`, SP-SR =
   `spider_sr_uncentered`, MP-SR = `maniptrans_sr`, MPPE = `mean_per_point_error_cm`.
2. The raw `C3r2_tier2` run dirs have no `hydra/.hydra/overrides.yaml`; their training parquet is the inventory's
   (re-hashed equal). Runs with overrides are cross-checked against `task.parquet`.
3. T2fix: `eval_ARM/evals.csv` has no `source_sha256` column; the value is read from that row's work dir
   `export/epNN/export_report.json`.
4. `--manifest` is the job data root's `whole_body/soma/track2_manifest.json` for all four roots (raw, K3fix, K3, G2);
   the four files are byte-identical (12fe4494). `HUMAN_MOTION_DATA_DIR` = the job's data root.
5. Concurrency: at most 4 of our evaluations, 2 per NUMA lane, 1 per GPU; exports at most 1 per lane; all
   `nice -n 10 taskset -c <lane cores>`. No timeouts anywhere: nothing is ever signalled; an
   evaluation past 2400 s (a CPU job past 1800 s) is only flagged OVERDUE.
6. Exports: try 1 at 1e-4. A refusal counts as a tolerance refusal only when every failure is a finite max_rel_diff
   above the tolerance with log_std equal (and algorithm ppo); then one export at 2e-4. Any other refusal is final
   (`export_refused`; Zuxing's: `export_incompatible`). A technical failure is retried once at the same tolerance.
7. Evaluator build: `eval_script_sha256` must be 160d24f5 and, when the rollout parquet carries it,
   `eval_code_sha256` must be b1b3ebe3 (the stock 8c778f9 build seen in the k3full evaluations).
8. (not included in this copy)
9. Summaries: PREREG names plus aliases requested by the task text: `summary.csv` / `summary.md` (= own input). Also
   `own_input_evals.csv`.
10. Stage B: if a T2fix designated checkpoint has the sha256 of an existing Stage-A job (ep01-L 62000), that job is
    reused and recorded as settled in `stageB_resolutions.jsonl` (export once per checkpoint), instead of a second job of
    the same file. Stage-B rows go to `manifest_stageB.jsonl` with their trigger evidence.
11. Integrity: the baseline listing (kit + `code/video_to_data_8c778f9` without `.git`: path, size, mtime) is taken
    after the manifest build and before the first evaluation, kept in `integrity/baseline.json`, and compared after
    every completed job; any difference writes `integrity/CHANGED.json` and stops new starts.
12. State at writing: VERDICT.json (18:41) already reports ep14 K3FIX settled (final 44922), so Stage B adds it on the
    first cycle. T2fix ep09-B ended (final.json 54449; eval_ARM stop.json 18:40); its runin_final.json is pending.
