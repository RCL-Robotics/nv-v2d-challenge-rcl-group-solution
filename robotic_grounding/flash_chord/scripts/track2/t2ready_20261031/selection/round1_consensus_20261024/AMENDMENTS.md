# Tier-2 consensus-reference selection (t2_consensus_20261024): amendment set A1

Written 2026-10-26, finalised at the cut-off **T_a = 2026-10-26 17:29:25 CST** (A3.1).

**When.** This file was written AFTER the own-input validation of PREREG 3.5.2 had failed and BEFORE any consensus score.
- `scores/own_input_validation.json` (written 2026-10-25 00:08:47, sha256 d0006975...) says `"pass": false`.
- The own-input diagnostic `diag/validation_vertex_sensitivity.json` (2026-10-25 03:17, sha256 c5bd332b...) had been read.
- Checked at writing:
  - `refs/` does not exist, so no consensus reference has been built under either variant;
  - `scores/` holds only `own_input_validation.{csv,json}`, and no file holds a rollout scored against a consensus track;
  - `SELECTION.sha256` does not exist, and nothing in section 7 (not included in this copy) has run;
  - `PREREG.md` sha256 3eee7268d53d1c3669614c63b69a54fa2462cf624ed88c3f7e215f5cb9f51ee9 still equals `PREREG.sha256`.

PREREG 10.4 requires every deviation to be recorded here, with time and reason, before the affected step. This file
changes three things (A1, A2, A3). Every rule it does not name stands as written in PREREG.md (A4).

**Authorisation (user, 2026-10-26, verbatim):** "oh I think by answer that question you can check the challenge rules?
i think it is fine". This replied to the main session's proposal, which named this gate change "option A" and proposed
including G2.

**Rules basis (PREREG 0, unchanged)**
- §2.8.c: a Submission must be "produced from that tier's permitted input".
- §2.9.d: a checkpoint per sequence, with an identical digest within an episode.
- §2.9.a: the code at the cited commit.
- The official evaluator, packer and metric code (sections 6 and 7; section 7 is not included in this copy).

G2 is a filter of the released Tier-2 data (A2.1). Every candidate below is trained on a Tier-2-derived input, and the
consensus chooses among such checkpoints.

**Files opened for this amendment:**
- the own-input validation and diagnostic files;
- file listings;
- Zuxing's run configs (`hydra/.hydra/*.yaml`), plus the header and first row of one `metrics.csv` (an own-input file);
- our `final.json`, `runin_final.json` and the noisy verdict files named in A3.2.

## A1. Own-input validation gate (PREREG 3.5.2), AUC part

### A1.1 What failed

- 400 runs were scored (80 jobs x 5), with 0 problem runs.
- Three of the four parts pass:
  - max |dMPPE| 0.0011 cm (limit 0.5 cm);
  - SP-SR and MP-SR equal in 100 % of runs (limit 95 %);
  - frame-conversion residual at most 1.7e-7 m and 1.6e-7 rad.
- **The AUC part fails.** Max |dAUC| is 0.0412 (`ep17_raw_final31807` run 1), against a limit of 0.02. 10 of 400 runs
  exceed 0.02. Max |dAUC| by input: raw 0.0412, K3fix 0.0152, G2 0.0047, K3 0.0015.

### A1.2 Cause

- ADD-AUC is computed over a sample of the object's mesh vertices, and the two scorers use different samples:
  - the evaluator samples 500 vertices of the simulator's visible mesh shapes;
  - our scorer samples 500 vertices of the released `.glb` with seed 0 (`v2dlb.vertices.sample_vertices`). This is the
    packer's convention and the leaderboard's (`build_blob`).
- The kit says so itself (`v2dlb/vertices.py`, sha256 4c95a4d6..., equal to `inputs/KIT.sha256`):
  - the evaluator's vertices "are not reproducible from the released `.glb` files alone";
  - "Until that lands, treat any small AUC discrepancy as this mismatch and not as a scoring bug."
- **The diagnostic.** It re-scored every run against its own input with seeds 0..9 of the same sampler.
  - Seed 0 reproduces our validation AUC on all 400 runs.
  - On the same rollout, the AUC moves by up to 0.0415 between seeds.
  - The evaluator's value lies inside our seed 0..9 range on 310 of 400 runs. It is at most 0.0145 outside that range
    (`ep17_raw_final31807` run 1).
- MPPE, SP-SR, MP-SR and the frame conversion all match. The frame map, the reference and the rollouts are therefore
  right, and the AUC gap comes from the vertex sample.

### A1.3 New AUC criterion

AUC_s (s = 0..9) is our own-input ADD-AUC for a run, computed with 500 vertices of the released Tier-2 mesh sampled with
seed s (`sample_vertices(mesh_path_for(<T2>/mesh, name), 500, s)`). Everything else is as in PREREG 3.3.

    A1-GATE: min_s AUC_s - 0.02 <= eval_AUC <= max_s AUC_s + 0.02 (s = 0..9), for every run

- **What it replaces.** This replaces "max |dAUC| <= 0.02". The other three parts are unchanged:
  - max |dMPPE| <= 0.5 cm;
  - SP-SR equal in >= 95 % of runs, and MP-SR equal in >= 95 % of runs (seed 0).
- **The scorer is unchanged.** Every consensus score uses seed 0, 500 vertices and the released Tier-2 mesh (PREREG 3.3),
  which is the leaderboard convention. Seeds 1..9 are used only by this gate. They never enter a score, a ranking or
  the selection.
- **Disclosure.** The band (the seed 0..9 range, +- 0.02) was fixed after the diagnostic had been read, i.e. knowing
  that the largest distance outside the range was 0.0145. The gate only detects implementation errors: a wrong frame map,
  a wrong reference or a wrong rollout. It selects nothing and cannot change any job's consensus score.

### A1.4 Re-check, scope and implementation

- **Re-check.** `scripts/t2cons_a1_gate.py` (a new file) recomputes all 400 runs at seeds 0..9, independently of the
  diagnostic file.
  - It writes `scores/own_input_validation_a1.{json,csv}`, with `"pass"`.
  - Seed 0 must reproduce `scores/own_input_validation.csv` exactly.
  - The result is reported in `PREREG_PINS.md`.
- **Failure** stops all consensus scoring, as in PREREG 3.5.2.
- **New jobs (A3).** Their runs are checked with the same code, after their evaluations and before any consensus score.
  - If the gate then fails only through runs of new jobs, those jobs are listed as `validation_failed` and are not
    candidates. The gate is then re-evaluated without them.
  - Any other failure stops scoring.
- **Record kept.** `scores/own_input_validation.json` (pass false) stays unchanged, as the record of the original gate.
- **Implementation.**
  - `scripts/t2cons_consensus.py` stays unchanged and pinned. It hard-codes three things: the original gate, the
    original candidate counts (PREREG 4.1 / 4.2) and variant A as binding.
  - Its `refs`, `score` and `select` commands therefore refuse to run, by design: they require
    `own_input_validation.json` to pass.
  - Steps S2-S5 under this amendment run through new files `scripts/t2cons_a1_*.py`, which import its functions
    unchanged.
  - Any such file not pinned in `PREREG_PINS.md` must be pinned in an addendum `PREREG_PINS_A1b.md` (+ sha256) before it
    computes the first consensus score. An addendum may only add pins and implementation notes.

## A2. WITH-G2 (variant B) becomes the binding result

### A2.1 G2 input

- G2 is a filter of the released Tier-2 trajectories (Zuxing's builder; copies in the record's `data/zuxing_g2/`).
  Variant B (with G2) is binding (A2.2).
- **Check at writing.** All 9 G2 entries of `inputs/TRACKS.sha256` still verify (`frame_refs_g2` episodes 01, 04, 09,
  19, 21 and 28, plus the three build JSONs). G2 has not changed since T_c, and it still has no other episodes.

### A2.2 Binding variant

- **Variant B is binding:** tracks R, F, C and G, i.e. WITH G2.
- **Variant A is reported only:** tracks R, F and C, i.e. WITHOUT G2.
- The track sets are those of PREREG 2.5's table:
  - **variant B:**
    - n = 3 (R, F, G) on 01, 04, 09 and 21;
    - n = 4 on 19 and 28;
    - otherwise the same as variant A;
  - **variant A:**
    - n = 3 on the 7 gate-FAIL episodes (00, 05, 15, 19, 22, 26 and 28);
    - n = 2 elsewhere.
- **Section 5** is applied unchanged to variant B. Its candidates are all jobs, including those with G2 input
  (PREREG 4).
- **The default D** is unchanged and the same for both variants.
- **Variant A** is still computed under section 5 and reported in `selection.json` and `summary_consensus.*`.

### A2.3 Section 7 under variant B (PREREG sections 7-9 are not included in this copy)

- **7.2.** CONS-B (the variant-B pick per episode) takes CONS-A's place, with files CLOSEST and T_1..T_5.
  - CLOSEST uses, per episode, the run whose variant-B consensus AUC is closest to the job's 5-run variant-B consensus
    mean. Ties go to the lower k.
  - DEFAULT T_1..T_5 is unchanged.
- **CONS-A is not packed here.**
  - The variant choice is fixed here, before any score.
  - CONS-A's noisy consensus numbers are reported.
- **7.3.** For an episode whose CONS-B job has G2 input, the frame reference is a byte copy of
  `frame_refs_g2/episode_XXXXXX.npz`. It is a read-only copy, and its sha256 must equal `inputs/TRACKS.sha256`. It is never
  the consensus.
- **7.8.** The only upload-eligible file is the **CONS-B CLOSEST** file.
  - It is re-packed with the real commit URL, and its pose columns must be byte-identical to the placeholder-URL file.
  - The user decides whether to upload.
- **10.3.** The cited commit must also contain, copied in with Zuxing's consent:
  - his G2 builder code and constants;
  - his `dev.zuxing.speedup` and `dev.zuxing.hard_start` training changes (A3).

  PREREG 9.10 (team membership; not included in this copy) stands.
- **9.5.** The reference-bias disclosure now applies to the binding variant. On 01, 04, 09, 19, 21 and 28, G2 is one of
  the 3 or 4 tracks and is also the input of the G2 candidates, which favours them there.

## A3. Added candidates (one cut-off T_a)

### A3.1 Round structure

- **Cut-off T_a = 2026-10-26 17:29:25 CST.** Nothing that finished after T_a is a candidate.
- **One-time extension.** This extends Round 1's candidate set once.
  - Round 2 (PREREG 8, not included in this copy) is NOT enabled: `ROUND2_GO` is not recorded, and PREREG 8's T_c2
    lapses.
  - Any later candidate needs a new pre-registration.

### A3.2 Runs

The added runs are every Tier-2 run folder of Zuxing's that exists at T_a, plus every run of ours that finished after
T_c in PREREG 8's categories (a)-(c) (not included in this copy).

| id | run | episodes | input (training parquet root) | designation |
|---|---|---|---|---|
| Z1 | `zuxing.lu/runs/2026-10-24-tier2-runin-g2` | 01, 04, 19, 28 | G2 (`zuxing.lu/data/human_motion_data_tier2_g2-2026-10-24`) | rule Z |
| Z2 | `zuxing.lu/runs/2026-10-25-tier2-runin-hard-g2` | 01, 19 | G2, the same root; experiment `g1_recon_body_coord_rsi_r2_runin_hard`, branch `dev.zuxing.hard_start` | rule Z |
| Z3 | `zuxing.lu/runs/2026-10-24-tier2-runin-k3fix-ep05-ep26` | 05, 26 | K3fix (`haoran.zhang/data/human_motion_data_tier2_k3fix`) | rule Z |
| Z4 | `zuxing.lu/runs/2026-10-24-tier2-runin-k3fix-ep04` (stopped) | 04 | K3fix | rule Z |
| O1 | `fc_runs/C3r2_tier2_k3ca/ep26_k3ca_s42` | 26 | K3CA (`data/human_motion_data_tier2_k3ca`) | the `final.json` checkpoint (K3CA PREREG v2a) |
| O2 | `fc_runs/C3r2_tier2_k3fix/ep04_k3fix_s42` | 04 | K3fix | the `final.json` checkpoint (k3full PREREG_v3) |
| O3 | `fc_runs/T2fix/T2L/ep01` (arm ep01-L) | 01 | K3fix | the best checkpoint in `eval_ARM/ep01/runin_final.json` (T2fix PREREG 4) |

**Rule Z** (own-input values)
1. **Rows.** Read the run's `eval_epNN/metrics.csv` (for Z4, `eval/metrics.csv`). A row is eligible if `status` is ok,
   `stock_verified` is True and `finished_at` <= T_a.
2. **Pick.** Take the eligible row with the highest `add_auc`, i.e. the evaluator's value against the run's own training
   parquet. Exact ties go to the newest (highest) iteration.
3. **File check.** The checkpoint `model_<iteration>.safetensors` must exist, and its sha256 must equal the row's
   `source_sha256`. Otherwise the next row in that order is taken, and the skip is recorded.

- **Tie-break.** This tie-break differs from PREREG 4.2, which used the lower MPPE and then the earlier iteration. Exact
  AUC ties are rare.
- **Copies.** The `metrics.csv` files are copied to `inputs_a1/zuxing_metrics/`, with sha256, before use.

**Our runs.** Each is designated as named in the table. The verdicts below are recorded for information only; they do
not decide whether the run is a candidate.
- **O1.** The K3CA `VERDICT.json` (2026-10-26 17:07:53) lists ep26 as settled, with choice K3CA (own-input SP/MP 1/1).
- **O2.** The k3full `VERDICT.json` was last written on 2026-10-25 at 05:15, before ep04 finished. Its regeneration at
  05:43 crashed in `k3full_helper.py verdict`.
  - `final.json` (model_44922) exists, and ep04's gate is PASS.
  - The designation is `final.json`, as for every K3fix job.
- **O3.** The T2fix `verdict.json` says KEEP_BASE for ep01-L (own-input, against its base). The checkpoint is still a
  candidate, as PREREG 8(b) (not included in this copy) intended, because it has both `final.json` and `runin_final.json`.

### A3.3 Jobs

- **Resolution.**
  - `scripts/t2cons_a1_resolve.py` (a new file) resolves the designations. It writes
    `inputs_a1/candidates_a1.{json,txt}` and `inputs_a1/SNAPSHOT_A1.sha256`.
  - These outputs are pinned in `PREREG_PINS.md`.
- **Reuse.**
  - A resolved checkpoint whose file (by sha256) is already an existing job reuses that job, with its 5 ok evaluations.
  - A provisional job whose checkpoint is now its run's settled designation is marked settled.
  - Every other resolved checkpoint becomes a new job.
- **Evaluations.** Each new job gets 5 official evaluations under PREREG section 6, unchanged:
  - sha256-checked copies, b39b69b export and the hull check;
  - stock runtime 8c778f9 and kit evaluator 160d24f5;
  - GPUs 0, 1, 2, 5, 6 and 7 only, under the `evallock` flock, on the eval-lane cores, one evaluation per GPU;
  - never GPU 4, no MPS and no signals.
- **Incomplete jobs.** A job that ends without 5 ok runs is listed and is not a candidate. This includes
  `export_incompatible`, e.g. a `runin_hard` config that the b39b69b exporter cannot build.
- **Order.** The new jobs' evaluations come first, then the A1 gate on their runs, then the consensus scores under both
  variants.
- **Variants.** Jobs with G2 input are candidates under variant B only; all others under A and B.
- **Kept.** Every earlier candidate (PREREG 4.1 / 4.2) stays a candidate, including earlier designations of the same runs.
- **Selection waits** for every new job to finish (5 ok runs or a terminal status), because variant B is binding.
- **Defaults unchanged.** D is the t2_best5 snapshot SELECTED. For ep04 that is the raw final, flagged PENDING.

### A3.4 Disclosure

- **Limits.**
  - The list holds every Tier-2 run folder of Zuxing's at T_a, plus every run of ours in PREREG 8's categories (not included in this copy), whatever
    its score.
  - Designations use own-input values only.
  - The recipe, the rule, the 0.01 margin, the 4-of-5 floor and the tie-breaks are unchanged.
- **Winner's curse (PREREG 9.7, not included in this copy)** grows with the number of challengers per episode. The new counts are in
  `PREREG_PINS.md`.
- **Code.** Z2 trains with Zuxing's `dev.zuxing.hard_start` code, and Z1 and Z3 with `dev.zuxing.speedup`. Both belong
  in the cited commit (A2.3).

## A4. Unchanged

The following stand as written in PREREG.md:
- section 1 (inputs);
- section 2 (reference recipe);
- section 3 (scorer and conversion), apart from the gate threshold in A1;
- section 5 (rule, margin, floor, ties and path guard);
- section 6 (evaluation protocol);
- section 7 (not included in this copy), except as in A2.3;
- sections 9 (not included in this copy) and 10.
