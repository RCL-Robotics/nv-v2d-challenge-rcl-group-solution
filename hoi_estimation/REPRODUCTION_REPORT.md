# Reproduction report: Tier 3 result (ADD-AUC 0.552)

This report covers the submitted Tier 3 trajectories, release `tier3_frame0_supports_20261024`. It describes how
to reproduce them and what came out when we did, for each step.

- **Detailed per-step commands:** `reproducibility/s1_runbook.md`, `reproducibility/tier3_runbook.md` and
  `REPRODUCE_TIER3_0552.md`.
- **Environment, data and weights:** `reproducibility/README.md`.
- **Tools, resources and parameters:** `AGENTIC_WORKFLOW.md`.

Metric: official ADD-AUC (1–9 cm), mean over the 30 Tier 3 episodes. GT is used only for scoring.

## Summary

| what was re-run (2026-11-02 / 11-03, from a clean checkout of this repository) | result |
|---|---|
| deterministic CPU steps (temporal post-processing, depth blend, frame-0 anchoring, the release builders) | **byte-identical** to the released files |
| GPU post-processing steps of the base pipeline (feet/floor, joint camera, release build, agent input export) | released object poses unchanged or float-level (≤ 1.2e-6 m); the human differs at float level |
| the 30 agent solvers (`solve.py`, no LLM), then frame-0 anchoring | **mean ADD-AUC 0.541 vs. 0.552 released**; 27 of 30 episodes within ±0.05 |
| support surfaces (fit, package) | fitted support poses bit-identical; contact diagnostics vary (unseeded sampling) |

In the 30-episode re-run, these steps were **not** re-run, and their released outputs were used as the starting
point (see "Not re-run" below). Section 6 runs all of them, the LLM agents included, from the official dataset on
2 episodes with the driver `reproducibility/run_all_tier3.sh`:
- the CARI4D baseline (about 33 GPU-hours);
- the FoundationPose re-registration (E2), the re-prompted episodes (M1), and the first-frame ICP and PnP;
- the LLM agent runs themselves.

## 1. Set up

1. **Code:** check out branch `dev.hanzhou.hoi_estimation` of this repository. All paths below are relative to
   `hoi_estimation/`.
2. **Data root:** get the data root (about 400 GB, or a subset of about 140 GB from P1 on;
   `reproducibility/README.md` section 0) and set `export V2D_ROOT=<root>`. Check it with
   `cd $V2D_ROOT && zcat <repo>/hoi_estimation/reproducibility/environment/data_manifest.sha256.gz | sha256sum -c --quiet`.
3. **Images:** `images/v2d_cari4d.sif` (16.9 GB) and `images/v2d_gdino.sif` (7.7 GB), checked by sha256 against
   `reproducibility/README.md` section 4. They can also be rebuilt from `reproducibility/environment/*.def`.
4. **Weights:** `weights/`, checked against `environment/weights.sha256`. Download links (HuggingFace and official)
   are in `reproducibility/README.md` section 4.
5. **Apptainer:** `export APPTAINER=<apptainer binary>`.
6. **GPU:** Ampere, Ada or Hopper (sm_80–sm_90). The original runs used an RTX 6000 Ada. The image does not run on
   Blackwell (sm_120).
7. **Host python for three CPU steps:** to reproduce them bit-identically, use python 3.12.13, numpy 2.5.2, scipy
   1.17.0, torch 2.10.0, trimesh 4.11.1 (`environment/pip_freeze_host_venv.txt`).

## 2. Procedure

| step | command (details in) | output | result |
|---|---|---|---|
| 1. base pipeline s1 | CARI4D per episode (`hoi_extraction/scripts/run_local.sh`), then E2, P1, M1, first-frame ICP/PnP, FF2, JC2s1, DB, `build_release.py` (`s1_runbook.md` 1–10) | `release/tier3_s1_20261020` | 0.210 |
| 2. agent inputs | `agent/export_inputs.py <ep> --ws $V2D_ROOT/agent_runs/T1_ds_ep<eee> --variant DB_JC2s1_shared_f820_pp --align $V2D_ROOT/logs/JC2s1_shared_f820_pp_align.json --camera JC2s1_shared_f820_pp` (`tier3_runbook.md` A1) | 30 input folders | — |
| 3. agent trajectories | either re-run the delivered solvers, `bash agent/rerun_solve.sh <ep> <out_dir>` for ep 0–29 (no LLM), or re-run the agents, `agent/run_batch_CL.sh` (`tier3_runbook.md` A2) | `object_poses_world.npz` per episode | — |
| 4. Opus release | `python3 agent/build_release_opus.py $V2D_ROOT/release/tier3_opus_20261021` (A3) | release | 0.217 |
| 5. frame-0 anchoring | `python tier2_refine/t3_frame0.py opus <out_dir> 3 2 shift` in the CARI4D image (B1) | 30 trajectories, prints `MEAN auc` | **0.552** |
| 6. combined release | `cd support && python3 build_release_combined.py <release_dir>` (B2), after the support steps (C) | `tier3_frame0_supports_20261024` | same trajectories as step 5 |

**Using the re-run trajectories of step 3 in step 5.** `t3_frame0.py` reads
`$V2D_ROOT/agent_runs/<run>/outputs/object_poses_world.npz`, where `<run>` is `CL2_ep<eee>` for ep 0–10,
`CL2b_ep<eee>` for ep 11–21 and `CL2c_ep<eee>` for ep 22–29. To score re-run trajectories without overwriting the
released ones, we used an overlay root:
- `dataset/`, `release/` and `ref_poses/` are symlinks to the data root;
- `agent_runs/<run>/outputs/object_poses_world.npz` is a symlink to `<out_dir>/<run>/outputs/object_poses_world.npz`.

Then run `V2D_ROOT=<overlay> python tier2_refine/t3_frame0.py opus <out_dir2> 3 2 shift`.

**Single-step check.** `reproducibility/check_steps.sh [step ...]` re-runs steps p1, ff2, jc2s1, db, release, export,
solve, frame0, opus, combined, fit and supportpkg. Each step reads the released upstream files and writes under a new
name, then compares the result with the released file.

## 3. Reproduction error per step

### 3.1 Base pipeline and release steps (`check_steps.sh`, 2026-11-02)

Run from a clean checkout (`git archive`) against a root holding only the documented files, which were bound
read-only. Repeated on commit ced2a6f with the same results.

| step | episodes | difference to the released file |
|---|---|---|
| P1_temporal (CPU) | 2, 25 | byte-identical |
| FF2_feet_ref (GPU) | 3, 25 | `pose_abs` max \|Δ\| 2.4e-7; floor height Δ ≤ 1.7e-9 m |
| JC2s1_shared_f820_pp camera (GPU) | 0, 25 | `pose_abs` max \|Δ\| 1.2e-6; camera centre ≤ 2e-6 m, rotation ≤ 4e-5°, K ≤ 4e-8 |
| DB_JC2s1_shared_f820_pp depth blend (CPU, host python) | 0, 25 | byte-identical |
| build_release s1 (GPU) | 0, 25 | object poses, camera, ground plane, mesh: byte-identical; human `transl` ≤ 8.3e-5 m, joint angles mean 0.0016 rad |
| export_inputs (GPU) | 25 | 7 of 9 files byte-identical; human vertices ≤ 1.2e-6 m; object poses identical |
| frame-0 anchoring (CPU) | all 30 | 30/30 byte-identical (also on the training server, in the image's python) |
| build_release_opus (CPU) | all 30 | 332/332 files byte-identical |
| build_release_combined (CPU) | all 30 | 392/392 files byte-identical |
| support fit (GPU) | 1, 2, 28 | `T_world_mesh` bit-identical |
| support package (CPU) | all 14 | 60/74 files byte-identical; the contact numbers vary (see 3.3) |

The base-pipeline steps change the object poses by at most about 1e-6 m. That is far below the 1 cm resolution of
the metric, so they do not change the score.

### 3.2 Agent solvers, then frame-0 anchoring (2026-11-03)

- **Run:** `agent/rerun_solve.sh` for all 30 episodes, one after the other on one RTX 6000 Ada: 65–286 s per
  episode, 1.42 GPU-hours in total.
- **Inputs:** sha256 equal to the agents' run manifests.
- **Scoring:** the re-run trajectories were then scored through step 5 unchanged.

| | |
|---|---|
| **mean ADD-AUC, released → re-run** | **0.552 → 0.541** (−0.011) |
| trajectories bit-identical to the released ones | 6 of 30 (ep 2, 9, 16, 19, 26, 28) |
| max translation difference < 2 cm | 23 of 30 |
| AUC change ≤ 0.01 | 27 of 30 |
| AUC change > 0.05 | 3 of 30 |

| ep | max \|Δt\| (cm) | AUC released | AUC re-run | ep | max \|Δt\| (cm) | AUC released | AUC re-run |
|---|---|---|---|---|---|---|---|
| 0 | 0.59 | 0.906 | 0.903 | 15 | 0.92 | 0.630 | 0.631 |
| 1 | 11.09 | 0.484 | **0.385** | 16 | 0 | 0.680 | 0.680 |
| 2 | 0 | 0.155 | 0.155 | 17 | 0.61 | 0.319 | 0.317 |
| 3 | 0.91 | 0.660 | 0.655 | 18 | 3.63 | 0.559 | **0.469** |
| 4 | 1.06 | 0.139 | 0.139 | 19 | 0 | 0.335 | 0.335 |
| 5 | 2.54 | 0.465 | 0.472 | 20 | 0.15 | 0.967 | 0.967 |
| 6 | 0.36 | 0.527 | 0.527 | 21 | 0.10 | 0.693 | 0.688 |
| 7 | 0.45 | 0.777 | 0.774 | 22 | 5.52 | 0.838 | **0.702** |
| 8 | 0.20 | 0.900 | 0.903 | 23 | 1.21 | 0.461 | 0.462 |
| 9 | 0 | 0.202 | 0.202 | 24 | 0.32 | 0.511 | 0.515 |
| 10 | 0.04 | 0.632 | 0.632 | 25 | 6.47 | 0.452 | 0.456 |
| 11 | 5.38 | 0.383 | 0.387 | 26 | 0 | 0.961 | 0.961 |
| 12 | 0.38 | 0.562 | 0.562 | 27 | 1.57 | 0.360 | 0.360 |
| 13 | 14.35 | 0.395 | 0.393 | 28 | 0 | 0.120 | 0.120 |
| 14 | 0.28 | 0.551 | 0.558 | 29 | 0.54 | 0.921 | 0.918 |

**Cause.**
- Several solvers use GPU operations that are not deterministic: SAM2 mask propagation, and nvdiffrast-based
  render-and-compare optimisation.
- A different mask or optimum in one segment changes that segment's pose. Most changes stay within a few mm. A
  few reach 5–14 cm in single frames (ep 1, 11, 13, 22, 25), and of those only ep 1 and 22 lose noticeably.
- Run-to-run variation: an earlier re-run of ep 7 and ep 25 on the same day gave 0.785 and 0.337, against 0.774 and
  0.456 here.

**Expected spread:** about ±0.01–0.02 for the mean, and up to about ±0.15 for single episodes.

### 3.3 Support surfaces

- **What varies:** `contact.py` samples 400 points on the object mesh. With the image's trimesh 5.1.1, `np.random.seed`
  does not fix this sampling.
- **Size of the variation:** the contact diagnostics differ from run to run: median rest gap ≤ 0.26 cm,
  `contact_shift_m` ≤ 0.4 mm.
- **Effect on the result:** none. The released object trajectories do not depend on it.

## 4. Not re-run, and why

| step | why | what we checked instead |
|---|---|---|
| CARI4D baseline (stages 0–8) | about 33 GPU-hours; CARI4D, FoundationPose, GroundingDINO and SAM2 are not bit-deterministic on GPU | outputs are in the data root with sha256 (`data_manifest.sha256.gz`); all later steps were re-run from them |
| E2 re-registration, M1 re-prompting, first-frame ICP/PnP | GPU, same reason; outputs are inputs of the checked steps | outputs in the data root with sha256 |
| the LLM agent runs | not deterministic: each run writes its own solver | full conversations, every script and the delivered `solve.py` are in `agent_records/`; the solvers were re-run (3.2) |

A re-run of these steps gives a new but similar baseline. The agent step would then also have to be re-run on it,
because the delivered solvers are tuned to their episode's inputs. We expect such a re-run to land near the reported
numbers rather than on them, given the per-episode spread in 3.2.

## 5. Where the re-run outputs are

The re-run outputs are on the lab workstation NAS, not in this repository:
- `v2d_repro/rerun_solve_all/<run>/`: re-run solver outputs and logs;
- `v2d_repro/root/repro_out/check_results.txt`: the `check_steps.sh` log.

The scripts that produced them are in this repository: `agent/rerun_solve.sh` and `reproducibility/check_steps.sh`.

## 6. End-to-end test of the driver `reproducibility/run_all_tier3.sh` (2026-11-03)

`run_all_tier3.sh` runs the whole chain from the official dataset to the combined release in a new, empty data root:
setup → CARI4D baseline → E2 → P1 → M1 → first-frame ICP/PnP → FF2 → JC2s1 → DB → s1 release → agent inputs →
Claude Code agents → Opus release → frame-0 anchoring → supports → combined release. It uses the parameters of the
submission. Usage and requirements are in its header and in `reproducibility/README.md` 7.

**Test set-up.**
- Machine: a second workstation that had not run the pipeline before (RTX 6000 Ada, GPU 0 only).
- Code: an export of this branch.
- Inputs: the dataset, weights and the two images, read-only.
- Episodes: 2 and 5. Episode 2 takes the E2 → P1 path and has a support surface; episode 5 takes the baseline path.
- Agents: `AGENTS=llm`, i.e. new Claude Code runs: CLI 2.1.303, `claude-opus-5-5`, effort high, 90 min, task
  `AGENT_TASK_temporal_v2.md`, 2 in parallel.

**Run time.** 3 h 46 min wall clock in total, about 3.8 GPU-hours.

| stage | wall clock |
|---|---|
| CARI4D baseline, 2 episodes | 113 min |
| E2 re-registration, episode 2 | 26 min |
| P1 … agent-input export | 6 min |
| agents | 66 min |
| Opus release and frame-0 anchoring | 10 s |
| supports and combined release | 5 min |

**Agent usage.**

| | episode 2 | episode 5 |
|---|---|---|
| turns | 149 | 94 |
| wall clock | 66 min | 58 min |
| tokens | 0.32M cache write, 25.7M cache read, 0.16M output | 0.21M cache write, 9.8M cache read, 0.12M output |
| list price | $11.0 | $6.0 |

Both agents delivered a trajectory.

**Result vs. the submission.**

| | episode 2 | episode 5 |
|---|---|---|
| frame-0 anchored ADD-AUC: end-to-end / submitted | 0.139 / 0.154 | 0.467 / 0.464 |
| s1 object pose vs. submitted: median Δt | 1.4 cm | 0.5 cm |
| s1 object pose vs. submitted: median Δrot | 37° | 0.4° |
| agent trajectory vs. submitted: median Δt | 1.6 cm | 1.1 cm |
| agent trajectory vs. submitted: median Δrot | 118° | 1.5° |
| support (round table): IoU / top height / yaw | 0.622 / 0.786 m / 194.0° (submitted 0.610 / 0.786 m / 193.4°) | — |

- **Why episode 2's rotations differ.** The object of episode 2 is a near-symmetric basket. Its large rotation
  differences most likely come from that symmetry: frame-0 anchoring corrects the flips, and the AUC after it is
  close. This explanation was not checked separately.
- **Human (SOMA) translations** differ by 0.1–0.2 cm median.
- **What to expect from a full run.** The GPU stages and the agents are not deterministic (section 4). A full
  30-episode run therefore gives per-episode numbers close to, not equal to, the submitted ones. Compare the spread in
  3.2: mean 0.541 vs. 0.552 with the agent solvers alone.

**Not covered by this test.**
- The M1 re-prompting path, used by episodes 3, 6, 12, 14, 16 and 21. It runs the same `run_episode.sh` with the v2
  prompts and was not run here, because episode 3 alone takes about 4.6 GPU-hours.
- `AGENTS=replay`, which re-runs the solvers delivered by the agents. These solvers were run on all 30 episodes
  in 3.2.

**Fixes the test found** (now in the driver and scripts):
- 13 weight files were readable only by their owner. `setup` now checks that the dataset, weights and images are
  readable.
- `run_episode.sh` had a hard-coded default data root. It now requires `V2D_ROOT`.
- `t3_frame0.py` and `support/common.py` now take `V2D_EPISODES` to run a subset of episodes.
- `support/overlay.py` now creates `vis/`, and `support/sheet.py` now lays out any number of episodes. With 14
  episodes the layout is still 4×4, as before.
