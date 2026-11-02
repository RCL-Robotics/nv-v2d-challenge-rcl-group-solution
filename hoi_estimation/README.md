# V2D Track 2 · HOI estimation workflow (hanzhou)

This directory contains the full workflow for reconstructing human-object interaction (HOI) from monocular video: the code, the agent task prompts, and the conversation records of every agent run. Data, weights, container images and other large files are not included; they live on the NAS under `/mnt/nas/Group/hanzhou/v2d/`.

## Pipeline

| Stage | Directory | Contents |
|---|---|---|
| 1. Base pipeline | `hoi_extraction/` | The base pipeline:<br>• per-frame object pose estimation with CARI4D / FoundationPose;<br>• focal length, floor and camera estimation;<br>• post-processing (symmetry flips, temporal smoothing, floor constraints, etc.).<br>Also included:<br>• s1 release builder: `scripts/build_release.py`;<br>• evaluators: `scripts/eval_object_pose.py`, `eval_official.py`;<br>• container definitions: `singularity/`;<br>• cluster scripts: `slurm/`. |
| 2. Agent in the loop | `agent/` | Each video is handed to an isolated AI agent that refines the result; see `agent/README.md`.<br>• sandbox launchers: `run_claude.sh`, `run_codex.sh`;<br>• network allowlist proxy: `allowlist_proxy.py`;<br>• agent input export: `export_inputs.py`;<br>• task prompts: the `AGENT_TASK*.md` versions;<br>• scoring and auditing: `score_runs.py`, `audit_run.py`;<br>• Opus v2 release builder: `build_release_opus.py`;<br>• record collector: `collect_records.py`. |
| 3. Tier 2 / Tier 3 refinement | `tier2_refine/` | • Tier 2 fusion: `fuse_v3.py`; best variant `v3_cf1_sym2`, AUC 0.781.<br>• Tier 3 frame-0 anchoring: `t3_frame0.py`, which uses the frame-0 object pose to fix symmetry flips, camera rotation and a constant translation. Best variant `opus 3 2 shift`, AUC 0.552. |
| 4. Support reconstruction | `support/` | For the 14 episodes where the object is placed on a table or chair, the supporting furniture is reconstructed from the video using a median background, GroundingDINO, SAM2 and a floor-constrained silhouette fit.<br>Run in order:<br>background → anchor → detect → segment → fit → contact → package,<br>then `build_release_combined.py` to build the combined release. |
| Diagnostics | `diagnostics/` | Diagnostic-only scripts. All GT use is confined to this directory, e.g. the oracle-extrinsics upper bound `oracle_f0.py` and the GT camera fit and rendering `gt_camera_fit.py` / `gtcam_vis.py`. None of these scripts feed into any pipeline that produces results. |
| Agent records | `agent_records/` | All 112 agent runs; see `agent_records/INDEX.md`. Each run includes the task prompt, a readable transcript `transcript.md`, the raw event stream `events.jsonl.gz`, the run manifest, and the agent's report and solver script. |

**Reproducing the Tier 3 result 0.552:** `REPRODUCE_TIER3_0552.md` (the chain, and the 30 agent runs with their conversations, scripts and outputs).

**Reproducibility:** `reproducibility/README.md` gives the exact commands and parameters of every step, the export of the frame-0 reference data, the `reconstruction/modules` revision inside the CARI4D image, the locations and checksums of the data, weights and images (NAS, and a copy on the training server; available on request), how to run from another data root (`V2D_ROOT`), and a re-run check of single steps against the released files (`reproducibility/check_steps.sh`).

## Results

Metric: official ADD-AUC (1–9 cm), averaged over 30 episodes. GT is used for evaluation only.

| Result | AUC |
|---|---|
| Tier 3 s1 (base pipeline) | 0.210 |
| Tier 3 Opus v2 (agent refinement) | 0.217 |
| Tier 3 Opus v2 + frame-0 anchoring | **0.552** |
| Tier 3 frame-0 anchoring + oracle extrinsics (diagnostic upper bound) | 0.612 |
| Tier 2 fusion `v3_cf1_sym2` | 0.781 |

Downstream release (trajectories and support surfaces): `/root/autodl-tmp/han.zhou/tier3/release/tier3_frame0_supports_20261024/`.

## Rules
- Neither the pipelines that produce results nor the agents have access to GT. GT is used only in `diagnostics/` and in evaluation.
- Tier 3 inputs that may be used: the monocular video, the object mesh, the ground plane, the frame-0 human, and the frame-0 object pose.
- Agents run in apptainer sandboxes, with network access through an allowlist proxy. Credentials are passed only through environment variables and are never written to the records. Packaging excluded `.codex_home` and scanned for common secret formats.
