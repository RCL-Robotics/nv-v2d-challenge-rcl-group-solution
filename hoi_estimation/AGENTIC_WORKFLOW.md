# Agentic workflow: tools, resources, automation, keys to effectiveness

This file covers the four points the organisers asked about. They are answered for the submitted Tier 3 result
(ADD-AUC 0.552, release `tier3_frame0_supports_20261024`) and the work that led to it. Reproduction steps are in
`REPRODUCE_TIER3_0552.md` and the exact commands in `reproducibility/`. Every number below comes from a file in this
repository or from the logs named next to it. Estimates are marked as such.

Two kinds of AI agent took part:

- **Orchestrator.** An interactive Claude Code session (`claude-opus-5-5`) worked with one human researcher. It wrote
  and ran the pipeline code in this repository, diagnosed results, set up and launched the per-episode agents, and
  wrote the documentation.
- **Per-episode agents.** One headless Claude Code run (`claude -p`, `claude-opus-5-5`, effort high) per video, each in
  an isolated sandbox. Each agent starts from the pipeline's output for its video and delivers an improved object
  trajectory, plus the solver script that recomputes that trajectory. Their complete records are in `agent_records/`.

## 1. Tools and roles

### Models and tools in the pipeline (steps 1–2 of `REPRODUCE_TIER3_0552.md`)

| tool | role | where |
|---|---|---|
| GroundingDINO (Swin-T OGC) + BERT-base | text-prompted object detection, which gives the box prompt for the masks | stage 0 of `run_episode.sh`; M1 re-prompting (`gen_prompt_v2.py`); support detection |
| SAM 2.1 (Hiera-L) | video object masks | stage 0; also used by the agents and by `support/` |
| MoGe-2 (ViT-L, normal) | monocular depth and intrinsics | stage 1 (depth), focal and floor estimation |
| SAM-3D-Body (DINOv3) + MHR body model | human mesh per frame | stage 3; human for the camera and floor steps |
| DINOv2 ViT-S/B | backbones used by SAM-3D-Body | via torch.hub |
| FoundationPose (refiner + scorer) | 6-DoF object pose per frame, from mesh, depth and mask | stage 5; E2 re-registration |
| CARI4D CoCoNet (`nvidia/cari4d_commercial`) | joint human–object refinement over time | stage 6 |
| nvdiffrast | differentiable rendering for render-and-compare | CARI4D; all 30 agent solvers |
| own scripts (`hoi_extraction/scripts`, `tier2_refine`, `support`) | post-processing: symmetry flips, temporal smoothing, first-frame ICP/PnP, shared camera intrinsics, floor/feet constraints, frame-0 anchoring, support surfaces | written by the orchestrator |

Download links and checksums for every checkpoint: `reproducibility/README.md`, section 4. FoundationStereo and
other stereo or multi-view models were **not** used (the input is monocular). `vda_depth.py` (Video Depth Anything)
and `estimate_focal.py` (Depth Pro) were exploratory and are not part of the chain.

### What the per-episode agents used (counted over the 30 runs that make up the result)

The agents could use the tools bound read-only into the sandbox, which are the pipeline's code, weights and
intermediate outputs, plus anything they wrote themselves.

| | number of the 30 runs |
|---|---|
| nvdiffrast render-and-compare (silhouette, edges) | 30 |
| SAM 2.1 masks (re-segmentation or mask propagation) | 28 |
| FoundationPose (re-registration from chosen frames) | 18 |
| MoGe-2 depth | 6 |
| scipy optimisation | 4 |
| optical flow / KLT tracks | 2 |
| LightGlue/SuperPoint, PnP | 1 each |

Tool calls of the agents: 2,397 Bash, 509 Read, 164 Write, 6 Edit, 1 sub-agent (from `events.jsonl.gz`).

### Roles

| who | role |
|---|---|
| human researcher | set goals and constraints (the rules, no GT in any pipeline, compute budget), chose the research directions, approved GPU jobs over 1 GPU-hour, reviewed results, decided what to submit |
| orchestrator (Claude Code, interactive) | wrote the code in this repository; ran the base pipeline and post-processing; scored results with GT, for evaluation only, in `diagnostics/` and the evaluators; wrote the agent task prompts and the sandbox launchers; launched and audited the agent batches; wrote `t3_frame0.py` and the support reconstruction; wrote the documentation |
| per-episode agents (Claude Code, headless) | per video: diagnose the baseline trajectory, write and test a solver, deliver `object_poses_world.npz` + `solve.py` + report, with no access to GT |

Other agents that were tried and not used in the result are listed in `agent_records/INDEX.md`: Codex with
`gpt-6-astra` (16 runs) and Codex with `deepseek-flash` (33 runs).

## 2. Resource usage

### Per-episode agents, the 30 runs in the result (CL2/CL2b/CL2c, 2026-10-21)

| | value | source |
|---|---|---|
| model | `claude-opus-5-5`, effort high, Claude Code 2.1.303 | `run_manifest.txt` |
| input tokens, uncached + cache writes | 6.33 M | final `result` event of each `events.jsonl.gz` |
| input tokens, cache reads | 373.5 M | same |
| output tokens (thinking included) | 3.43 M | same |
| turns / tool calls | 3,220 / 3,077 | same |
| cost | USD 194 at list price, as reported by the CLI (`total_cost_usd`); the runs used a Claude subscription | same |
| agent wall time | 29.2 h in total; per run 32–89 min (mean 58), limit 90 min | `run_manifest.txt` (`wall_s`) |
| elapsed time | 11.8 h (05:04–16:49 UTC) for all 30, 3 runs at a time | `tier3_runbook.md` A2 |
| GPU | one RTX 6000 Ada (48 GB) shared by the 3 concurrent runs, each told to stay below 9 GB; at most 11.8 GPU-hours | `AGENT_TASK_temporal_v2.md`, `run_batch_CL.sh` |

All 63 Opus agent runs, the exploratory ones included: 470 M cache-read, 8.2 M other input and 4.4 M output tokens;
USD 248 at list price; 36.8 agent-hours. The Codex runs (GPT, DeepSeek) did not record tokens in the same form.

### Orchestrator (interactive Claude Code sessions, 2026-10-04 to 2026-11-03)

| | value |
|---|---|
| model | `claude-opus-5-5` |
| API responses | about 2,200 |
| input tokens | 521 M cache reads, 15.3 M cache writes, < 0.01 M uncached |
| output tokens | 1.73 M |
| human messages | about 170 over 30 days (most of them short: direction, approval, questions) |

These figures come from the session logs of the project directory. They include the reproducibility and
documentation work after the submission, and they cover only the main project's Claude Code sessions.

### Compute (2 × RTX 6000 Ada 48 GB workstation; episode 0 of the baseline on one cluster GPU)

| step | GPU time | source |
|---|---|---|
| CARI4D baseline (stages 0–8), 28 successful episodes | 28.1 GPU-h (33.1 GPU-h with failed attempts and retries), about 1 h per episode | `logs/run_local_*.log` |
| E2 re-registration, M1 re-prompting (6 episodes), first-frame ICP/PnP, camera, feet/floor | not logged per step; estimate 10–20 GPU-h | — |
| per-episode agents | ≤ 11.8 GPU-h (see above) | — |
| frame-0 anchoring (`t3_frame0.py`), release builders | CPU, seconds | — |
| support surfaces (14 episodes) | estimate < 2 GPU-h | — |
| re-running the 30 delivered solvers (`agent/rerun_solve.sh`) | 1.42 GPU-h for all 30 (65–286 s per episode) | 2026-11-03 run, section 4 of `REPRODUCE_TIER3_0552.md` |

The whole Tier 3 chain therefore costs on the order of 50–70 GPU-hours on a 48 GB GPU, plus about 30 agent-hours
of LLM time.

## 3. Automation and scalability

**What is automated.**
- **Base pipeline.** `run_local.sh` and `run_episode.sh` run 8 stages per episode, with stage markers so a rerun
  resumes. The post-processing steps are scripts with fixed parameters (`reproducibility/s1_runbook.md`).
- **Agent step.** `export_inputs.py` writes one input folder per episode. `run_batch_CL.sh <P> <episodes>` then
  starts one sandboxed agent per episode with at most P at a time, and stops launching when the
  subscription quota runs out (`*_STOP`). Each agent works unattended for up to 90 min and must deliver a trajectory,
  a solver script that recomputes it offline, and a report. Nobody intervenes during a run.
- **Auditing.** The network proxy logs every request of a run. `audit_run.py` lists a run's commands, network
  requests and suspicious file accesses. `collect_records.py` packages the transcripts.
- **Final steps.** `build_release_opus.py`, then `t3_frame0.py opus 3 2 shift`, then `build_release_combined.py`.
  Each is one command.

**Human involvement.**
- **The 30-episode agent step:** none during the runs. The human approved the GPU budget and the batch before launch.
- **The research loop:** the human was involved throughout. The human chose directions, and the orchestrator
  proposed, implemented and evaluated them.
- **Manual decisions that went into the result:**
  - the task prompt, revised once after the first runs (section 4);
  - which agent batch to release;
  - the frame-0 anchoring variant and the other parameters in the table below;
  - the support fit settings for 14 episodes (`tier3_runbook.md` C, per-episode settings).
- **Restarts:** two batches had to be relaunched, CL2b and CL2c, after the subscription quota ran out. This was done
  by the orchestrator with one command each.

**Scalability.**
- **Per-episode agents are independent.** Each runs in its own apptainer sandbox (`--containall`, no network except
  an allow-list proxy, read-only inputs, weights and code). Each needs one GPU share (< 9 GB) and one LLM session.
  The step therefore parallelises across episodes up to the GPU memory and the LLM rate limit. We ran 3 at a time on
  one GPU, and the limit we hit was the subscription quota, not compute.
- **Cloud.** No cloud parallelisation was used. Everything ran on one 2-GPU workstation, and episode 0 of the
  baseline ran on a Slurm cluster (`hoi_extraction/slurm/`). Because each agent's inputs are one folder and its
  outputs one folder, the step maps directly onto one cloud or cluster job per episode. Cost per episode: about
  1 h of agent time and USD 6.5 at list price.
- **Reproducing without the LLM.** The delivered `solve.py` scripts re-run in about 2–5 min per episode on a GPU
  (`agent/rerun_solve.sh`).

**Final parameters of the submitted chain** (full commands in `reproducibility/s1_runbook.md` and
`tier3_runbook.md`):

| step | parameters |
|---|---|
| shared camera intrinsics (`postproc_joint_camera_K.py`) | `--K-prior 820 758.1 560.5` (focal = median MoGe-2 estimate over the 30 episodes; principal point = median of the per-episode fits), `--sig-logf 1e-4 --sig-pp 0.01`, `--sig-px 2.0 --sig-h 0.02 --window 15 --clip 0.3` |
| depth blend (`postproc_depth_blend.py`) | `--w 0.5 --max-dev 0.1` |
| per-episode agents (`run_claude.sh`, `run_batch_CL.sh`) | `claude-opus-5-5`, effort high, 90 min, task `AGENT_TASK_temporal_v2.md`, 3 concurrent, < 9 GB GPU memory each, network open except HuggingFace |
| frame-0 anchoring (`t3_frame0.py opus <out> 3 2 shift`) | flip = 3 (principal-axis flip at frame 0 + continuity tracking, then an object-frame anchor only if the residual at the static start exceeds 15°); camrot = 2 (camera rotation fixed from the static start, rotations only, skipped above 15°); trans = shift (constant shift of the static-start mean to the frame-0 position); static start = frames within 2 cm / 5° of frame 0, at most 60 |
| supports (`support/`) | per-episode fit settings in `tier3_runbook.md` C |

## 4. Keys to effectiveness

Mean ADD-AUC over the 30 Tier 3 episodes; GT used only for scoring.

| step | AUC |
|---|---|
| base pipeline s1 | 0.210 |
| + Opus per-episode agents | 0.217 |
| + frame-0 anchoring | **0.552** |
| (diagnostic: frame-0 + oracle camera extrinsics) | (0.612) |

1. **Start the agent from a good pipeline result, not from scratch.**
   - The agent's job is to find and fix this video's specific failures: symmetric objects flipped by 180°, jumps
     under occlusion, drift, wrong static phases.
   - It gets the full intermediate state: several pose sources, depth, masks, the human, the floor, and the models.
2. **Encode the lessons of earlier runs in the prompt (iterative refinement).** The task prompts are in
   `agent/AGENT_TASK*.md`. Besides the trajectory task there were also camera and depth variants. The trajectory
   task changed as follows:
   - **First Codex/GPT runs, `AGENT_TASK.md`, 10-18/19:**
     - Agents fixed rotations well: on ep 13 the rotation error fell from 119° to 21°.
     - But they re-estimated depth from the object alone and moved the whole trajectory by cm along the viewing ray,
       so the AUC dropped (ep 13: 0.087 → 0.082).
     - On ep 3 an agent misread which end of the mug is the bottom and turned the whole trajectory upside down
       (AUC 0.122 → 0).
   - **`AGENT_TASK_temporal.md`:**
     - asks for trajectory-level reasoning: static vs. carried phases, several candidates per frame including
       symmetry flips, global selection and smoothing, propagation from anchor frames;
     - asks for self-validation without GT;
     - asks the agent to keep the baseline when it is unsure.
   - **`AGENT_TASK_temporal_v2.md`, used for the result:**
     - adds "keep the baseline's overall placement" (rewritten section, see `diff`);
     - the agent fixes rotation and relative motion, then rescales along the viewing rays to the baseline's median
       camera-frame depth per phase;
     - the agent may deviate only if two independent physical cues agree within about 2 cm;
     - the agent must report the depth difference to the baseline per phase.
3. **Model capability matters.** On the same prompt and inputs:
   - On 4 episodes: Opus 0.315 vs. DeepSeek 0.200 (s1 0.307). With a GT-fitted rigid camera, which is diagnostic
     only: 0.731 vs. 0.278 (`EXP-2026-1021-agent-opus55-vs-ds-temporal`).
   - On all 30 episodes: Opus 0.217 vs. DeepSeek 0.165 (`EXP-2026-1021-agent-opus55-v2-full30`).
   - The stronger model also changed correct results less often.
4. **Require a re-runnable solver and an honest report.**
   - Each agent must deliver `solve.py`, which recomputes its trajectory from the inputs offline, plus a validation
     table and its confidence.
   - This makes every result auditable and re-runnable without the LLM (section 4 of `REPRODUCE_TIER3_0552.md`).
   - It also discourages hand-editing numbers.
5. **Isolation and no GT.**
   - Agents run in apptainer sandboxes with read-only inputs and a logging network proxy (HuggingFace blocked).
   - Credentials are only in environment variables.
   - GT is used only by the orchestrator, for scoring.
6. **Diagnose where the remaining error is, then fix it with the simplest tool.**
   - The scoring with a GT-fitted rigid camera (diagnostic only; 0.731 vs. 0.315 raw on 4 episodes) showed that, after the agents, most of the
     error was a constant camera/placement offset per episode, not the trajectory shape.
   - The Tier 3 rules allow the frame-0 object pose. The deterministic `t3_frame0.py` uses it to:
     - fix symmetry flips (ep 12: 0 → 0.56, ep 15: 0 → 0.63, ep 26: 0 → 0.96);
     - fix the camera rotation over the static start;
     - shift the trajectory to the frame-0 position (alone: 0.425).
   - This took the agents' trajectories from 0.217 to 0.552.
7. **What did not work, or remains open.**
   - Axially symmetric objects whose rotation drifts or jumps over time stay unsolved: ep 2 basket, ep 4 bat, ep 9
     bowl, ep 17 cone.
   - The agents' own image-space checks cannot see a shift along the viewing ray, which is why the prompt rule in
     point 2 is needed.
   - The delivered solvers are not bit-deterministic on GPU (SAM2 propagation, rendering-based optimisation).
     Re-running all 30 gave a mean of 0.541 instead of 0.552. 27 episodes stayed within 0.05 AUC, and 3 moved by
     0.09–0.14 (`REPRODUCE_TIER3_0552.md`).
