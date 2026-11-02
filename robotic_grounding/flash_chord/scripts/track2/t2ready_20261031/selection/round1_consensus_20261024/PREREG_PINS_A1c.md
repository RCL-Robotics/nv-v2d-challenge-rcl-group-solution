# t2_consensus_20261024: PREREG_PINS_A1c (addendum to PREREG_PINS.md and PREREG_PINS_A1b.md; pins and implementation notes only)

**When.** Written 2026-10-26 ~18:20 CST, after the A3.3 evaluations finished (18:02:42, 10 exports ok, 50 evaluations ok)
and after a dry check of `scripts/t2cons_a1_consensus.py`, BEFORE that script ran the A1.4 gate, built a reference,
computed a consensus score or selected anything.
- `refs/` does not exist; `scores/` holds only the own-input validation files; `SELECTION.sha256` does not exist.
- No consensus score exists under either variant.

**Scope.** As AMENDMENTS A1.4 and PREREG_PINS 3.3 allow, this file only adds pins and implementation notes. It changes no
rule of PREREG.md (3eee7268...), AMENDMENTS.md (f648226a...), PREREG_PINS.md (9d149379...) or PREREG_PINS_A1b.md
(7239bc56...).

**What its writer ran.** Before writing this file it ran, besides the dry check, a throwaway test harness (stdin, no new code
file) whose outputs are under `tmp/a1c_harness/`:
- the gate code on the 5 runs of ONE A3 job (`ep28_zx_g2_100000`), against its own input: all 5 runs inside the
  seed 0..9 range, max |dMPPE| 8.6e-7 cm, SP/MP equal; run 4 reproduces the evaluator's collapsed run (AUC 0.1145);
- the select code on SYNTHETIC seeded random values; its picks equalled an independent
  re-derivation of the PREREG 5 rule on all 30 episodes x 2 variants.
Those files are tests, not results, and are not read by any step below.

## 1. Steps (all through `scripts/t2cons_a1_consensus.py`, kit Python `-I -B`)

| order | command | step | writes (new files only) |
|---|---|---|---|
| 0 | `check` | dry check (done) | `dryrun/a1_consensus_check_*.txt`, `inputs_a1/ROLLOUTS_A1c.sha256` |
| 1 | `gate` | AMENDMENTS A1.4: A1.3 gate on the 50 runs of the 10 A3 jobs | `scores/own_input_validation_a1_new.{csv,json}` |
| 2 | `selftest-select` | the select code on synthetic values, before any real score | `selftest/a1_select_synthetic/` |
| 3 | `refs` | S2 (PREREG 2) | `refs/A/`, `refs/B/`, `refs/build.json`, `refs/build_a1.json` |
| 4 | `score` | S4 (PREREG 3) | `scores/consensus_scores.csv`, `scores/consensus_job_stats.json`, `scores/consensus_a1_run.json` |
| 5 | `select` | S5 (PREREG 5, variant B binding, A2.2) | `selection.json`, `selection_consensus.json`, `selection.txt`, `selection_table.{csv,md}`, `selection_candidates.csv`, `summary_consensus.{json,md}`, then `SELECTION.sha256` LAST |

Each of steps 1-5 refuses to run unless: PREREG.md == PREREG.sha256 and its section-11 pins and commits hold;
`inputs/SNAPSHOT.sha256` and the kit files hold; AMENDMENTS.md, PREREG_PINS.md, PREREG_PINS_A1b.md and this file each
equal their sha256 file (the first three also equal the value hard-coded in the script) and every pin in them holds;
this file pins the running script; `inputs_a1/SNAPSHOT_A1.sha256` and `candidates_a1.json` hold; the selftest is a PASS
of the pinned `t2cons_consensus.py`; the 400-run A1 result is the pinned PASS of `t2cons_a1_gate.py`; the A3.3 eval
driver has ended. Steps 3-5 also need step 1's PASS written by this same script under this file. Steps 3-5 refuse once
`SELECTION.sha256` exists.

## 2. Implementation notes (no rule change)

1. **Pinned code, reused unchanged.** The script compiles `scripts/t2cons_a1_gate.py` (a29c997b...) from its verified
   bytes; that module loads `scripts/t2cons_consensus.py`, whose bytes are checked against e1b943b1....
   - **refs** runs the pinned `cmd_refs` unchanged: same tracks, de-duplication, median / chordal mean, the PREREG 2.5
     track-set check (`EXPECTED_USED`, variant B = R,F,G on 01/04/09/21, R,F,C,G on 19/28), TRACKS.sha256 checks.
   - **score** runs the pinned `cmd_score` unchanged: the packer's ordering and frame conversion against each job's own
     input reference, the provenance checks, `episode_metric` against the A and B references with seed-0 released-mesh
     vertices, per-job 5-run statistics and the closest-to-mean run per variant.
   - For both, only four module-level names the pinned commands look up are rebound: `require_pins` (the original
     gate check, which refuses by design, AMENDMENTS A1.4) -> the amended pin chain of section 1; `load_jobs`,
     `load_evals` and `candidates` -> the pinned functions PLUS the 10 A3 jobs of `manifest_a1.json` and their ok runs
     in `own_input_evals_a1.csv` (same checks: input kind and PREREG 3.2 frame reference, export.json sha256, results
     path, one ok row per run), minus any A1.4 `validation_failed` job.
   - **select** is new code around the pinned `select_episode` (the PREREG 5 rule: margin 0.01 on exact floats, floor
     4 of 5 own-input runs with SP-SR = MP-SR = 1, ties by variance, MPPE, name), `floor_count`, `stats` and
     `comp_summary`.
2. **A1.4 gate on the A3 runs.** Each of the 50 runs is scored against its OWN input with the pinned
   `metrics_vs` (seed 0) and the pinned `t2cons_a1_gate.auc_seed` (seeds 1..9), then the A1-GATE line of AMENDMENTS A1.3
   and the unchanged MPPE / SP / MP parts are applied over all 450 runs (the 400 runs of the pinned
   `scores/own_input_validation_a1.csv` plus the 50). The 400-run part is first re-evaluated from that CSV and must
   reproduce its recorded PASS. If the 450-run gate fails, every A3 job with a flagged run (outside the band, |dMPPE| >
   0.5 cm, SP or MP differs, or a problem run) is set aside and the gate is re-evaluated without them; if it then
   passes, those jobs are `validation_failed` and not candidates (A1.4); otherwise nothing is scored. A kit or stock
   runtime change fails the gate.
3. **Candidate set.** 90 jobs, 77 of them variant-A: the 75 t2_best5 snapshot jobs (71 A), the 5 PREREG 4.2 jobs (2 A)
   and the 10 A3 jobs (4 A), each with 5 ok official evaluations; the script checks these counts. The 2 reused A3
   resolutions (`ep04_zx_k3fix_12000`, `ep01_t2fixL_p62000`) are those t2_best5 jobs (training sha256 checked);
   `ep01_t2fixL_p62000` is marked settled (A3.3), which is a label only. Challenger counts per episode and variant must
   equal `candidates_a1.json` `challenger_counts` "after" (minus validation_failed); defaults must equal its `defaults`
   and the snapshot SELECTED; ep04's default must be the raw final.
4. **Rollouts frozen.** `inputs_a1/ROLLOUTS_A1c.sha256` lists the sha256 of the 450 rollout parquets (the ok try of
   every run of every candidate), written by the dry check before any score. `score` refuses if any differs.
5. **Selection (S5).** Per episode: D = the t2_best5 snapshot SELECTED (ep04 flagged PENDING), the same for both
   variants. Variant B (binding): pool = every candidate of the episode; variant A (report): pool without G2-input
   jobs. The binding pick is variant B's. CONS-B CLOSEST = `consensus_job_stats.json` closest run of the pick under
   variant B (AMENDMENTS A2.3; ties lower run). `selection.json` and `selection_consensus.json` are identical.
   `selection_table.{md,csv}` = one row per episode (D, its consensus and own-input values and floor, every variant-B
   challenger with its consensus mean, delta vs D, floor and whether it qualifies, the pick, CLOSEST run, reason, and
   variant A's pick). `selection_candidates.csv` = one row per episode x variant x pool job, with the 5 consensus AUCs.
   `summary_consensus.*` = CONS-B, CONS-A, DEFAULT, RAW-ONLY, K3FULL composition averages.
6. **Freeze.** `SELECTION.sha256` is written last. It lists the S5 outputs and: `scores/consensus_scores.csv`,
   `scores/consensus_job_stats.json`, `scores/consensus_a1_run.json`, `refs/build.json`, `refs/build_a1.json`, both
   A1 gate results, `scores/own_input_validation.json`, `selftest/selftest.json`, `manifest_a1.json`,
   `own_input_evals_a1.csv`, `inputs_a1/candidates_a1.json`, `inputs_a1/ROLLOUTS_A1c.sha256`, this file,
   PREREG_PINS_A1b.md, PREREG_PINS.md, AMENDMENTS.md and PREREG.md. Outputs are made read-only.
7. **Resources.** CPU only (no GPU, no lock, no MPS), `nice -n 10`, 4 BLAS threads. Nothing is signalled.
8. **Not done here.** Section 7 (pack of CONS-B) runs only after `SELECTION.sha256` exists, under its own code.

## 3. A3.3 own-input results (recorded for reference; from `summary_a1_own_input.json`)

Floor = official own-input runs with SP-SR = MP-SR = 1 (PREREG 5.4b needs >= 4 of 5). Passing the floor:
`ep05_zx_k3fix_88000` 5/5, `ep26_zx_k3fix_92000` 5/5, `ep26_k3ca_final25298` 5/5, `ep19_zx_g2_92000` 5/5,
`ep28_zx_g2_100000` 4/5 (run 4 collapsed). Failing it (0/5): `ep04_k3fix_final44922`, `ep01_zx_g2_124000`,
`ep04_zx_g2_88000`, `ep01_zx_g2hard_102000`, `ep19_zx_g2hard_52000`, and the reused `ep04_zx_k3fix_12000` and
`ep01_t2fixL_p62000`. These are own-input numbers and decide nothing beyond the floor of PREREG 5.4b.

## 4. Pins (sha256)

pin scripts/t2cons_a1_consensus.py 5dd430ec78f32cdff7921cfdcc5eca2b9de2a925cf29f6c41b141c7d0a580658
pin scripts/t2cons_consensus.py e1b943b10fd1db13ab6f78e5bde7b3c6e6c48f7daeee6230c32c463d14ac41b4
pin scripts/t2cons_a1_gate.py a29c997b7271e9299db109e97af5210f528484719210bb58685523b0ae0eb16b
pin scripts/t2cons_a1_eval.py dedd2e0bed855cb39e520b0fcc5fa635ccf7121fdf12d55a11afd10694f5eb56
pin PREREG.md 3eee7268d53d1c3669614c63b69a54fa2462cf624ed88c3f7e215f5cb9f51ee9
pin AMENDMENTS.md f648226a95a061eeefcfd5138cb01823db8a060fe121a0f4968ffa163dc4229f
pin PREREG_PINS.md 9d1493797205e8c60255c50658c536714a3702fe8751c724f37ed1444a040ece
pin PREREG_PINS_A1b.md 7239bc561eacc1d3ed6b3f210ae80193c30ad3168d8e0255dd7ee63016bd7c7a
pin inputs_a1/candidates_a1.json be42b6ddd32ff708365b894852cdbbfbdda583504f44038dc4790827a97301e4
pin inputs_a1/ROLLOUTS_A1c.sha256 8a545ea10287e7eb7098f6274a20c373e83ebb64fce425e32ed7473338f99b1f
pin scores/own_input_validation_a1.json c6af3fd5ee7362b434b557bc844a0eb3da34c55ddcb31ee9c098ae2ea8b7e6a7
pin scores/own_input_validation_a1.csv 10e29bf0c36ca30ae91a283d7ff89e8560737b8e2293bc313168103969dcf957
pin manifest_a1.json a4ed1f2a59b82a5f6fc6775c72f03784b8db1cbaac604e1ca15ce43029d2aadd
pin manifest_a1.csv e01c09cade22c2a45bbe6e6af6df1f2e79a4516d79d2feac1bd1419fc4f73ff5
pin own_input_evals_a1.csv 5cf2bb5a4b55c56bb5279b76c5b439e934822d99de985cba566f20837fa21001
pin summary_a1_own_input.json 34c86eba14fac7933a25c711d43f3d42bd64969c685acbc8f4aa1ab276ec36ba
pin COPIES_a1.tsv b99bf0be8284a087bd8df754e72c51a4f35d42a75be3c33ae25f78a03a36ce5c
pin eval_a1_heartbeat.json 63b8e8c7f1e3a68b965efa161ca493156b2570c72a533e7528393563054909ad
pin integrity/check_20261026_180242.json 41a76baff5c24b440d25e0fb08f622010e73f384e34d5aa6e16fecb29a87ece3
pin dryrun/a1_consensus_check_20261026_181623.txt 449f6573033bf6ce6e4687fa0d5e1f09554718473d9a88fcdec578db63a77756
pin exports/ep05_zx_k3fix_88000/export.json f2276733147deaa28bf1b1857abc05a9bf278a68d37d46fa1b0e9476921f3c94
pin exports/ep26_zx_k3fix_92000/export.json a17266b2836fe3d03326cfbffaa5b9a309fd3e215263b43a2a4c506ccb6734b4
pin exports/ep26_k3ca_final25298/export.json d4c39a4a4161c6d19d465f626db5a2cffc0ec50f531d75eaff1c7c3a3824c20a
pin exports/ep04_k3fix_final44922/export.json 5397e2ff5f08d226591850dc9085e20af078718f4174202390b1244e50dde1a0
pin exports/ep01_zx_g2_124000/export.json 8ae9250e219f81c6d6d414a49551003b2090f7d0d5f3b3f5ed2fcf84c0fcff02
pin exports/ep04_zx_g2_88000/export.json 24f93fbf92017f04fccac3cafb59fb00b9cd6610bc6b2a87d5981e2589114667
pin exports/ep19_zx_g2_92000/export.json a0bf59b05eccd14ac186439388d7d485211c39dea001aacd1f42400c20a788d5
pin exports/ep28_zx_g2_100000/export.json 357fd03b99eba5a8f24c40d3943dad795cc245c0467152c7f1fbc0a9db5cefa1
pin exports/ep01_zx_g2hard_102000/export.json fe386de5787903b212c5a19340b73bbbc0210d7ddd24a2e7c95dece239d568c3
pin exports/ep19_zx_g2hard_52000/export.json 9c2943992f93173c81191b1ca686a7b16ca9bda11b1206a7a941c5814c338426
