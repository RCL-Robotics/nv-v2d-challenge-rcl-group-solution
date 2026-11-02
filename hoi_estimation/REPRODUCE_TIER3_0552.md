# Reproducing the Tier 3 result (ADD-AUC 0.552)

The submitted Tier 3 trajectories are the release `tier3_frame0_supports_20261024`: the Claude Opus agent trajectories
("Opus v2", one agent run per episode) corrected with the frame-0 object pose (`tier2_refine/t3_frame0.py opus 3 2
shift`). Metric: official ADD-AUC (1–9 cm), mean over the 30 Tier 3 episodes; GT is used for scoring only.

| step | what | result | commands |
|---|---|---|---|
| 1 | base pipeline (CARI4D → post-processing chain) → release `tier3_s1_20261020` | AUC 0.210 | `reproducibility/s1_runbook.md` |
| 2 | export one input folder per episode for the agents | `agent_runs/T1_ds_ep<eee>/inputs/` | `reproducibility/tier3_runbook.md` A1 |
| 3 | one Claude Opus agent run per episode | `object_poses_world.npz` per episode | below, and `tier3_runbook.md` A2 |
| 4 | Opus release: s1 with the object trajectories replaced | AUC 0.217 | `tier3_runbook.md` A3 |
| 5 | frame-0 anchoring of the Opus trajectories | **AUC 0.552** (printed as `MEAN auc`) | `tier3_runbook.md` B1 |
| 6 | combined release: step 5 + support surfaces (does not change the trajectories) | release | `tier3_runbook.md` B2, C |

Environment, data layout, weights (with download links) and images: `reproducibility/README.md`.

## Step 3: the agent runs

The release uses run `CL2_ep<eee>` for episodes 0–10, `CL2b_ep<eee>` for 11–21 and `CL2c_ep<eee>` for 22–29
(`agent/build_release_opus.py`). All three batches used the same task, `agent/AGENT_TASK_temporal_v2.md`, model
`claude-opus-5-5`, effort high, 90 min limit (`agent/run_claude.sh`, `agent/run_batch_CL.sh`). Each run's full record
is in `agent_records/<run>/`:

| file | content |
|---|---|
| `AGENT_TASK.md` | the task prompt the agent received |
| `transcript.md` | the conversation, readable: every message, tool call and tool output (long tool calls and outputs shortened) |
| `events.jsonl.gz` | the raw event stream of the same conversation, complete: full tool calls (including every script the agent wrote or ran) and outputs; only images, long base64 strings and signatures removed |
| `run_manifest.txt` | date, CLI version, model, binds, sha256 of every input file, wall time |
| `scratch/` | the scripts and notes the agent wrote in its scratch folder while working (`*.py`, `*.md`, `*.json`, `*.txt`; images and large arrays left out) |
| `outputs/solve.py` (+ `hoilib.py` etc.) | the agent's final solver, which writes the delivered trajectory from the inputs |
| `outputs/object_poses_world.npz` | the delivered trajectory (`T_world_mesh`, frames × 4 × 4), as used in the release |
| `outputs/REPORT.md`, `validation.md` | the agent's own report |

`agent_records/INDEX.md` lists all 112 agent runs; only the 30 above enter the result.

There are two ways to redo step 3:

1. **Re-run the delivered solvers (no LLM).** `solve.py` recomputes the trajectory from the inputs; the task required it
   (`AGENT_TASK_temporal_v2.md`, "solve.py must recompute its estimates from the inputs"). For every episode:

   ```bash
   export V2D_ROOT=<data root>          # holds agent_runs/T1_ds_ep*/inputs, outputs/, experiments/, logs/, weights/, images/
   for e in $(seq 0 29); do bash agent/rerun_solve.sh $e <out_dir>; done
   ```

   The script prints the difference to the released trajectory. Re-runs on the same inputs (sha256 equal to the
   run manifest) and the same GPU model (RTX 6000 Ada), 2026-11-03:

   | episode | vs released trajectory | AUC after step 5, released → re-run |
   |---|---|---|
   | 2 | bit-identical | unchanged |
   | 7 | max 0.47 cm, rotation-matrix entries ≤ 0.10 | 0.777 → 0.785 |
   | 25 | max 9.6 cm; one more symmetry switch in step 5 | 0.452 → 0.337 |

   Mean over 30 with these two replaced: 0.552 → 0.548. Several solvers are not deterministic on the GPU (SAM2 mask
   propagation, rendering-based optimisation), so single episodes can move by about ±0.1 AUC.
2. **Re-run the agents.** `agent/run_batch_CL.sh` with `PFX=<prefix> TASK=AGENT_TASK_temporal_v2.md` (exact calls in
   `tier3_runbook.md` A2). An agent run is not deterministic: a new run writes its own `solve.py` and the score varies
   from run to run. The transcripts show what the agents did and why.

Then copy (or point) the trajectories to `$V2D_ROOT/agent_runs/<run>/outputs/object_poses_world.npz` and continue
with steps 4–6.
