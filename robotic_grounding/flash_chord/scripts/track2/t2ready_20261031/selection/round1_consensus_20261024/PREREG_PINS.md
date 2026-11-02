# t2_consensus_20261024: PREREG_PINS (PREREG 10.1 step S1, under AMENDMENTS A1-A3)

**When.** Written 2026-10-26 ~17:40 CST, in this order:
1. `AMENDMENTS.md` was frozen: sha256 f648226a95a061eeefcfd5138cb01823db8a060fe121a0f4968ffa163dc4229f = `AMENDMENTS.sha256`,
   with T_a = 2026-10-26 17:29:25 CST.
2. The A3 resolution ran (`inputs_a1/`, 17:29:57).
3. The A1 gate re-check ran (`scores/own_input_validation_a1.*`, 17:32:16).

This file was written BEFORE any consensus score:
- `refs/` does not exist;
- `scores/` holds only the own-input validation files;
- `SELECTION.sha256` does not exist.

**Scope.** This file adds pins and implementation notes only. It changes no rule of PREREG.md
(sha256 3eee7268d53d1c3669614c63b69a54fa2462cf624ed88c3f7e215f5cb9f51ee9) or of AMENDMENTS.md.

## 1. Checks before the first consensus score (PREREG 3.5, as amended by A1)

### 1.1 Selftest (PREREG 3.5.1): PASS

`selftest/selftest.json` (2026-10-25 00:08:16) records 24 tests and 0 problems. It was written by
`scripts/t2cons_consensus.py` with sha256 e1b943b1..., which equals the pinned file.

### 1.2 Original own-input validation (PREREG 3.5.2 as written): FAIL, AUC part only

`scores/own_input_validation.json` (2026-10-25 00:08:47) is kept unchanged as the record.
- 400 runs, 0 problem runs.
- Max |dAUC| 0.0412 (`ep17_raw_final31807` run 1), against a limit of 0.02. 10 runs exceed 0.02.
- Max |dMPPE| 0.0011 cm. SP-SR and MP-SR are equal in 100 % of runs.
- Conversion residual at most 1.7e-7 m and 1.6e-7 rad.

### 1.3 Amended gate (AMENDMENTS A1.3), re-checked with `scripts/t2cons_a1_gate.py`: PASS

The re-check covers the same 80 jobs x 5 runs = 400 runs. It writes `scores/own_input_validation_a1.{json,csv}`; the
console log is `tmp/a1_gate_20261026.log`. The result is `"pass": true`, and every part holds:

| part | result | limit |
|---|---|---|
| eval AUC within [min_s AUC_s - 0.02, max_s AUC_s + 0.02], s = 0..9 | 400 / 400 runs (0 fail) | every run |
| ... of which inside the seed range itself | 310 / 400 | report |
| largest distance outside the seed range | 0.0145 (`ep17_raw_final31807` run 1: eval 0.5540 vs seeds 0.5063-0.5395) | 0.02 |
| largest seed range (AUC spread over seeds 0..9, same rollout) | 0.0415 | report |
| max \|dMPPE\| | 0.0011 cm | 0.5 cm |
| SP-SR equal / MP-SR equal | 100 % / 100 % | >= 95 % |
| problem runs | 0 | 0 |
| seed 0 vs `own_input_validation.csv` (all 4 metrics, all runs) | max abs difference 0.0 (exact) | <= 1e-12 |
| seed range vs `diag/validation_vertex_sensitivity.json` | max abs difference 0.0 (exact) | report |
| kit + stock runtime trees (path, size, mtime) | unchanged | unchanged |

**By input** (largest distance outside the seed range / inside-range count):

| input | runs | inside range | largest distance outside |
|---|---|---|---|
| raw | 160 | 118 | 0.0145 |
| K3fix | 190 | 149 | 0.0065 |
| K3 | 15 | 12 | 0.0015 |
| G2 | 35 | 31 | 0.0001 |

The new jobs of A3 are not in this re-check. Under A1.4, their runs are checked with the same code after their
evaluations and before any consensus score.

## 2. A3 resolution at T_a = 2026-10-26 17:29:25 CST (`inputs_a1/candidates_a1.{json,txt}`)

**Method**
- Zuxing's `metrics.csv` files were copied byte-for-byte into `inputs_a1/zuxing_metrics/` (hashed in
  `inputs_a1/SNAPSHOT_A1.sha256`).
- All 4 of his Tier-2 run folders match the A3.2 table, and so do their episode folders.
- Each training parquet comes from the run's `hydra/.hydra/overrides.yaml` and has the input kind A3.2 states.
- Every designated checkpoint hashes to its `metrics.csv` `source_sha256`. No row was skipped, and `rsi_<N>` exists for
  every new job.
- The "excluded" counts below are rows with status other than ok. No row was excluded for `stock_verified` or for
  finishing after T_a.

| id | episode | input | designated iteration | own-input value used (Zuxing: his single eval) | job | status |
|---|---|---|---|---|---|---|
| Z1 | 01 | G2 | 124000 | AUC 0.2188, SP/MP 0/0 (62 of 62 rows eligible) | `ep01_zx_g2_124000` | new |
| Z1 | 04 | G2 | 88000 | AUC 0.1975, SP/MP 0/0 (44 of 44) | `ep04_zx_g2_88000` | new |
| Z1 | 19 | G2 | 92000 | AUC 0.5293, SP/MP 1/1 (42 of 50; 8 not ok) | `ep19_zx_g2_92000` | new |
| Z1 | 28 | G2 | 100000 | AUC 0.5729, SP/MP 1/1 (51 of 53; 2 not ok) | `ep28_zx_g2_100000` | new |
| Z2 | 01 | G2 (hard start) | 102000 | AUC 0.2240, SP/MP 0/0 (52 of 52) | `ep01_zx_g2hard_102000` | new |
| Z2 | 19 | G2 (hard start) | 52000 | AUC 0.1326, SP/MP 0/0 (45 of 45) | `ep19_zx_g2hard_52000` | new |
| Z3 | 05 | K3fix | 88000 | AUC 0.7819, SP/MP 1/1 (45 of 45) | `ep05_zx_k3fix_88000` | new |
| Z3 | 26 | K3fix | 92000 | AUC 0.9057, SP/MP 1/1 (49 of 49) | `ep26_zx_k3fix_92000` | new |
| Z4 | 04 | K3fix | 12000 | AUC 0.1104, SP/MP 0/0 (6 of 6) | `ep04_zx_k3fix_12000` | reused (same file) |
| O1 | 26 | K3CA | 25298 (`final.json`) | K3CA VERDICT: choice K3CA, settled, own-input SP/MP 1/1 | `ep26_k3ca_final25298` | new |
| O2 | 04 | K3fix | 44922 (`final.json`) | k3full VERDICT stale (A3.2); gate PASS | `ep04_k3fix_final44922` | new |
| O3 | 01 | K3fix | 62000 (`runin_final.json` best) | T2fix verdict KEEP_BASE | `ep01_t2fixL_p62000` | reused, now settled |

**10 new jobs**
- G2 input, variant B only: `ep01_zx_g2_124000`, `ep04_zx_g2_88000`, `ep19_zx_g2_92000`, `ep28_zx_g2_100000`,
  `ep01_zx_g2hard_102000`, `ep19_zx_g2hard_52000`.
- Variants A and B: `ep05_zx_k3fix_88000`, `ep26_zx_k3fix_92000`, `ep26_k3ca_final25298`, `ep04_k3fix_final44922`.

**Frame references (PREREG 3.2):** `frame_refs_g2`, `frame_refs_k3fix` and `frame_refs_k3ca` respectively.

**Challengers per episode** (other than D). The changes are listed below; every other episode is unchanged. The
variant-A counts before A3 equal the PREREG 4.1 table, which the script checks.

| episode | A before -> after | B before -> after |
|---|---|---|
| 01 | 3 -> 3 | 4 -> 6 |
| 04 | 1 -> 2 | 3 -> 5 |
| 05 | 2 -> 3 | 2 -> 3 |
| 19 | 0 -> 0 | 2 -> 4 |
| 26 | 2 -> 4 | 2 -> 4 |
| 28 | 0 -> 0 | 2 -> 3 |

Under the binding variant B, the most challengers on one episode is now 6 (ep01). PREREG 9.7 applies (winner's curse).

**Notes (report only)**
- O3: `fc_runs/T2fix/eval_ARM/ep01/best.json` has a `checkpoint_sha256` field of e78f836d..., which is not the file's
  sha256. Both `eval_ARM/ep01/best.safetensors` and `T2L/ep01/model_62000.safetensors` hash to 2d69e2da..., which is
  the t2_best5 job `ep01_t2fixL_p62000`'s `training_sha256`. That file is the one designated and reused.
- The own-input values above come from Zuxing's single evaluations. The selection uses only the 5 official runs of
  PREREG 6 (floor) and the consensus scores (section 5).

## 3. Implementation notes (no rule change)

1. **The original script refuses by design.** `scripts/t2cons_consensus.py` is pinned and unchanged, and it hard-codes
   three things:
   - the original 3.5.2 gate;
   - the PREREG 4.1 / 4.2 candidate counts (75 + 5 jobs);
   - variant A as binding.

   With this file in place, its `refs`, `score` and `select` refuse to run, because `scores/own_input_validation.json`
   says pass false (A1.4).
2. **The original eval script covers only the first 5 jobs.** `scripts/t2cons_eval.py` is hard-coded to PREREG 4.2's 5
   jobs and writes `manifest_new.*`, `own_input_evals_new.csv`, `COPIES.tsv` and `summary_new_own_input.*`. These files
   are pinned below and must not change.
3. **New files for the remaining steps.** The A3 evaluations, the A1 gate on the new runs, and S2-S5 with variant B
   binding must run through new files:
   - code in `scripts/t2cons_a1_*.py`;
   - outputs under new names, e.g. `manifest_a1.*`, `own_input_evals_a1.csv`, `COPIES_a1.tsv`, `refs/`,
     `scores/consensus_*`.

   They import the pinned functions unchanged. Each such script must be pinned in `PREREG_PINS_A1b.md` (+ sha256) before
   it copies, evaluates or scores anything.
4. **How to check these pins.** Each `pin` line below has 3 fields. A `scripts/...` path is relative to
   `/root/autodl-tmp/haoran.zhang`; any other path is relative to this folder. This is the format
   `t2cons_consensus.py check_pins` reads. Check them with `sha256sum` or with that function.

## 4. Pins (sha256)

pin scripts/t2cons_consensus.py e1b943b10fd1db13ab6f78e5bde7b3c6e6c48f7daeee6230c32c463d14ac41b4
pin scripts/t2cons_eval.py 2cf427866a9432a3bd1b8fbb7864149d4258a5d19100c20da92d32bfe4fbfa9b
pin scripts/t2cons_eval_launch.sh d8ebe50a26290306c3f6b35c558777c4652f0e7ccac63c655d8aa08a84b951b4
pin scripts/t2cons_a1_gate.py a29c997b7271e9299db109e97af5210f528484719210bb58685523b0ae0eb16b
pin scripts/t2cons_a1_resolve.py f41e5c6dabb7125377e29ed2db59dafb259901f15979bac60c0c9ad3a4a1e28f
pin PREREG.md 3eee7268d53d1c3669614c63b69a54fa2462cf624ed88c3f7e215f5cb9f51ee9
pin AMENDMENTS.md f648226a95a061eeefcfd5138cb01823db8a060fe121a0f4968ffa163dc4229f
pin selftest/selftest.json 1de2976ded02683235c670626e5a36d84ab2d2d4fda9e09b6f97b01779c24b43
pin scores/own_input_validation.json d0006975500d9f17553d5e1ac86889c631ea66054ae5bbd791c71ee9194dc643
pin scores/own_input_validation.csv 92ed4d33ea2e5c1d498c41f22d2f72e76479d02f393c6a41fd3d1aa66b2306dc
pin scores/own_input_validation_a1.json c6af3fd5ee7362b434b557bc844a0eb3da34c55ddcb31ee9c098ae2ea8b7e6a7
pin scores/own_input_validation_a1.csv 10e29bf0c36ca30ae91a283d7ff89e8560737b8e2293bc313168103969dcf957
pin tmp/a1_gate_20261026.log 757cc241a9cc9f1025fe3504e57de6357e9847237cdacadd2125608d3e7263a2
pin diag/validation_vertex_sensitivity.json c5bd332b3c8b4b390f4005fcf05bd201edf95c95173b82935300e3253c5f1be3
pin inputs_a1/candidates_a1.json be42b6ddd32ff708365b894852cdbbfbdda583504f44038dc4790827a97301e4
pin inputs_a1/candidates_a1.txt 51e794bf65af3f2ef845bf2e2022f0e7f206fd422e84b3b107a6122d262795f1
pin inputs_a1/SNAPSHOT_A1.sha256 ebff4adaf945c2a4896a13ccedc2f26f0afc2889b3b92a415aa83a2f7ab060fc
pin manifest_new.json 73d0e8356555da9286ce77cfe3c417fa1c84e83b2c20f0c3c80a8c4c688a3994
pin manifest_new.csv 33e62c639a6dcc1ca75365e6b479e9e1f167f0d06acebc3dc19ab1c279605faf
pin own_input_evals_new.csv b755db2de88b11b1cd49523aae360e008bbcf1c3225da7777f40b24e4fa5504b
pin summary_new_own_input.json 337b7b703e9ac5047e43879661c2e682b25986878ff69cf1e13a398fa0853abb
pin COPIES.tsv d6428c851ae7474a282d7b811095d8e555d42376a1cc64ed3f0b3fcf45d69ffe
pin integrity/baseline.json ef9b5035c09b8ad40713d9fb08470eeccb2abe8845d4d1cce3037c49da1cd622
pin inputs/SNAPSHOT.sha256 9e10e36572c3461960e765944ac92716a861cff6f73a2f6d0eea41d1abd3cbdf
pin inputs/TRACKS.sha256 c8cb834522f245de0b7733579f8c537781eef14d79269e782bc391f1dcecca6d
