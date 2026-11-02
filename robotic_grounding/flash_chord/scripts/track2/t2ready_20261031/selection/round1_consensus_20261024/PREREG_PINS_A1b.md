# t2_consensus_20261024: PREREG_PINS_A1b (addendum to PREREG_PINS.md; pins and implementation notes only)

**When.** Written 2026-10-26 ~17:50 CST, after a dry run of `scripts/t2cons_a1_eval.py` and BEFORE it copied, exported,
evaluated or scored anything.
- `refs/` does not exist; `scores/` holds only the own-input validation files; `SELECTION.sha256` does not exist.
- No consensus score exists.

**Scope.** As AMENDMENTS A1.4 and PREREG_PINS 3.3 allow, this file only adds pins and implementation notes. It changes
no rule of PREREG.md (3eee7268...), AMENDMENTS.md (f648226a...) or PREREG_PINS.md (9d149379...).

## 1. Step: the A3.3 evaluations (S3 for the A3 jobs)

`scripts/t2cons_a1_eval.py` (+ `scripts/t2cons_a1_eval_launch.sh`) runs the 5 official own-input evaluations of each of
the 10 new jobs of `inputs_a1/candidates_a1.json`:
- variants A and B (priority 1, first): `ep05_zx_k3fix_88000`, `ep26_zx_k3fix_92000`, `ep26_k3ca_final25298`,
  `ep04_k3fix_final44922`;
- variant B only (priority 2): `ep01_zx_g2_124000`, `ep04_zx_g2_88000`, `ep19_zx_g2_92000`, `ep28_zx_g2_100000`,
  `ep01_zx_g2hard_102000`, `ep19_zx_g2hard_52000`.

The 2 reused jobs (`ep04_zx_k3fix_12000`, `ep01_t2fixL_p62000`) are the same files (sha256) as t2_best5 jobs that already
have 5 ok official evaluations. They are not evaluated again; their 5 runs are listed from the pinned snapshot
`inputs/t2_best5/own_input_evals.csv`.

## 2. Implementation notes (no rule change)

1. **Pinned code, reused unchanged.** The script loads `scripts/t2cons_eval.py` from its pinned bytes (2cf42786...), which
   loads `scripts/t2best5_driver.py` from its pinned bytes (24b9af39...) re-rooted at this folder. Copy, export,
   evaluation, ok checks, retries, the hull check, the kit / stock tree check (same `integrity/baseline.json`), the GPU
   locks, lanes, and the count of t2_best5's live evaluations are that code. The script subclasses its driver and
   overrides only the manifest, the summary, the heartbeat and the progress count.
2. **Pins it checks before any copy or evaluation:** PREREG.md and its section-11 pins and commits, `inputs/SNAPSHOT.sha256`
   (both via the pinned `t2cons_eval.check_pins`), AMENDMENTS.md, PREREG_PINS.md and its 25 pins, every file of
   `inputs_a1/SNAPSHOT_A1.sha256`, `candidates_a1.json` (be42b6dd...), and this file, which must pin the script.
3. **Designation re-checked from own-input values.**
   - Zuxing's jobs: rule Z (A3.2) is re-derived from the snapshot copies in `inputs_a1/zuxing_metrics/` (sha256 checked),
     and must give the resolved iteration and the file's sha256.
   - Ours: `final.json` must name the checkpoint.
   - Every checkpoint is re-hashed against `candidates_a1.json`.
4. **K3CA input kind.** It is added in memory only:
   - root `data/human_motion_data_tier2_k3ca`;
   - frame references `tmp_analysis/tier2/k3ca/frame_refs_k3ca` (PREREG 3.2);
   - manifest `whole_body/soma/track2_manifest.json`, byte-identical (12fe4494...) to the K3fix and raw roots'.
5. **Training parquet.** The run's hydra `task.parquet` must name the `candidates_a1.json` file. Its sha256 must equal an
   earlier record of the same file: the t2_best5 snapshot manifest (+ Stage B) or `manifest_new.json` row (G2, K3fix), or
   the K3CA root's `READY` list (K3CA ep26, d9a210fa...).
6. **Export tree.**
   - Zuxing's checkpoints: `code/nv-v2d_runin` @ b39b69b (PREREG 6), 1e-4 then one 2e-4 fallback.
   - Ours (K3CA ep26, K3fix ep04): their training tree `code/fc_r2` @ 61fa5d3 (t2_best5 3.3; K3CA trains with fc_r2, per
     `scripts/k3ca_driver.sh`).
   - The hard-start G2 runs (Z2) are exported with b39b69b like the other Zuxing runs. If that exporter refuses them, the
     job is `export_incompatible` and not a candidate (A3.3).
7. **New output names.** Everything the script writes is new:
   - `zuxing_ckpts_a1/<run>/epNN/` (model + rsi copies, `SHA256SUMS`, `SOURCE.json`) and `COPIES_a1.tsv`. A separate
     folder is used because `zuxing_ckpts/2026-10-24-tier2-runin-g2/ep04/` already holds the PREREG 4.2 copy's
     `SHA256SUMS` and `SOURCE.json`, which must not change.
   - `exports/<job>/`, `runs/<job>/` (new job names only);
   - `manifest_a1.{json,csv}`, `own_input_evals_a1.csv`, `summary_a1_own_input.{csv,json,md}`;
   - `eval_a1_driver.log`, `eval_a1_driver_stdout.log`, `eval_a1_heartbeat.json`;
   - `dryrun/a1_eval_plan_*.txt`, `preflight/hulls_*`, `integrity/check_*.json`.
   - The pinned driver also uses `driver.pid` (created, then removed at the end), `.driver.lock` (flock only) and `STOP`.
8. **Scheduling is unchanged** (PREREG 6): GPUs 0, 1, 2, 5, 6, 7; the shared flock opened read-only; at most 2 per eval
   lane and 4 in all, counting t2_best5's live evaluations; nice + taskset on the eval-lane cores; never GPU 4, no MPS,
   nothing signalled.
9. **Dry run, 2026-10-26 17:45** (`dryrun/a1_eval_plan_20261026_174518.txt`): every pin OK, all 10 jobs ok (designation,
   checkpoint sha256, parquet sha256, manifest, frame references), hull check OK on all 8 distinct parquets
   (`preflight/hulls_20261026_174518.json`), both reused jobs have 5 ok snapshot runs with matching sha256, GPUs 0, 1, 2,
   5, 6 and 7 usable.
10. **Next, not done here.**
    - The A1 gate on the new runs (A1.4).
    - Then the references and the consensus scores (S2, S4) and the selection under variant B (S5).
    - Each needs its own new `scripts/t2cons_a1_*.py`, pinned in a further addendum before it scores anything.

## 3. Pins (sha256)

pin scripts/t2cons_a1_eval.py dedd2e0bed855cb39e520b0fcc5fa635ccf7121fdf12d55a11afd10694f5eb56
pin scripts/t2cons_a1_eval_launch.sh dae2a8a7be9f93e269c8546f658814f199883caf46f1a02ae38ad6fa98345209
pin scripts/t2cons_eval.py 2cf427866a9432a3bd1b8fbb7864149d4258a5d19100c20da92d32bfe4fbfa9b
pin scripts/t2best5_driver.py 24b9af39a2e2708641e7870447b12b6475c23aec9cd18ba71086768dd69cfc89
pin PREREG.md 3eee7268d53d1c3669614c63b69a54fa2462cf624ed88c3f7e215f5cb9f51ee9
pin AMENDMENTS.md f648226a95a061eeefcfd5138cb01823db8a060fe121a0f4968ffa163dc4229f
pin PREREG_PINS.md 9d1493797205e8c60255c50658c536714a3702fe8751c724f37ed1444a040ece
pin inputs_a1/candidates_a1.json be42b6ddd32ff708365b894852cdbbfbdda583504f44038dc4790827a97301e4
pin inputs_a1/SNAPSHOT_A1.sha256 ebff4adaf945c2a4896a13ccedc2f26f0afc2889b3b92a415aa83a2f7ab060fc
pin dryrun/a1_eval_plan_20261026_174518.txt 018beddd2d2c08e054e3cdf450e0b219ca30e3d5eae520ecc64c3e8fc19fd494
pin preflight/hulls_20261026_174518.json c4b14477d0e84cbb7bdda000466c9f2f02b45c9ae4ae63162bfd781fbab6014d
