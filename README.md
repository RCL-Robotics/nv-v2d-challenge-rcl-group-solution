# NV V2D Challenge — Track 2 solution of team MBZUAI-RCL Group

This repository holds the code, configurations and training records behind our three **Track 2** submission files of
the NVIDIA Video-to-Data (V2D) Challenge: Unitree **G1 + Dex3** whole-body policies that track human–object
interaction (HOI) references, one policy per episode, 30 episodes per tier. All policies are trained with
**FlashCHORD** (PPO, Newton / Warp / JAX) and scored with the official submission kit.

It is built on NVIDIA's open-source [Video to Data](https://github.com/nvidia-isaac/video_to_data) pipeline
([documentation](https://nvidia-isaac.github.io/video_to_data/)); the original packages (`video_ingestion_agent/`,
`reconstruction/`, `robotic_grounding/`) are kept unchanged except where our records say otherwise.

**Paper:** [DexDuet: Learning Contact-Preserving Bimanual Dexterity from Human Demonstrations](https://arxiv.org/abs/2611.30495)
(arXiv:2611.30495) · **[Project page](https://rcl-robotics.github.io/Dexduet.io/)**. The coordinated per-hand actor and
the assisted reference-state-initialisation (A-RSI) curriculum used by our policies are described there. If you use
this code, please [cite the paper](#citation).

![Video to Data pipeline](docs/figures/v2d_overview.png)

---

## Contents

- [Results at a glance](#results-at-a-glance)
- [Repository map](#repository-map)
- [Environment](#environment)
- [Data and kit](#data-and-kit)
- [Training configuration](#training-configuration)
  - [Common recipe](#common-recipe)
  - [Tier 1 — clean reference](#tier-1--clean-reference)
  - [Tier 2 — synthetic-noise reference](#tier-2--synthetic-noise-reference)
  - [Tier 3 — our own video reconstruction](#tier-3--our-own-video-reconstruction)
- [Quick start: train, evaluate, pack](#quick-start-train-evaluate-pack)
- [Reproducing the submitted files](#reproducing-the-submitted-files)
- [Selection protocol](#selection-protocol)
- [Citation](#citation)
- [Credits and license](#credits-and-license)

---

## Results at a glance

Numbers are from the official kit evaluator (`evaluate_policy.py`) on the stock runtime. Tier 1 is scored against the
clean Tier-1 reference; Tiers 2 and 3 are scored against each policy's own input (the leaderboard reference is hidden),
so they are **not** leaderboard numbers.

| Tier | Input | Submitted file | ADD-AUC | SP-SR | MP-SR | MPPE |
|---|---|---|---:|---:|---:|---:|
| 1 | clean Tier-1 motion | set `all7`, trial 05 | 0.9310 | 1.000 | 1.000 | 0.93 cm |
| 2 | released noisy motion, filtered (K3fix / G2 / raw / K3) | `R9-CLOSEST`, trial 3 | 0.8123 (own input) | 0.933 | 0.933 | 3.38 cm |
| 3 | our monocular-video reconstruction | `FINAL_r` | 0.6760 (own input) | 22 / 30 | 21 / 30 | 16.39 cm |

The submitted trial of every tier is the complete 30-episode trial closest to the mean of repeated official
evaluations — never a best-of choice (see [Selection protocol](#selection-protocol)).

## Repository map

| Path | What it is |
|---|---|
| [`REPRODUCE.md`](REPRODUCE.md) | Step-by-step reproduction of the three submitted files (install → verify package → repack → re-evaluate → retrain) |
| [`scripts/final/`](scripts/final/README.md) | Entry-point scripts: `setup_env.sh`, `get_data.sh`, `train_tier{1,2,3}.sh`, `eval_tier{1,2,3}.sh`, `pack_tier{1,2,3}.sh`, `build_inputs_tier{2,3}.sh`, `select_tier{2,3}.sh`, `selftest.sh` |
| [`scripts/final/FINAL_EPISODES.tsv`](scripts/final/FINAL_EPISODES.tsv) | The 90 evaluated policy files: input, training commits, stages and sha256 of every artifact |
| [`scripts/final/site.env.example`](scripts/final/site.env.example) | Template of the machine-specific paths every script reads (`--site FILE`) |
| [`environment/`](environment/README.md) | Recorded conda / pip environments (`flashchord`, `v2dkit`, `v2d`, `zx_g2_builder`) and system info |
| [`robotic_grounding/flash_chord/`](robotic_grounding/flash_chord/README.md) | FlashCHORD trainer; our G1 experiment configs are in `src/flash_chord/configs/experiment/g1_recon_body_coord_rsi_r2*.yaml` |
| [`robotic_grounding/flash_chord/scripts/track2/`](robotic_grounding/flash_chord/scripts/track2/) | Track-2 tooling (`launch.py`, `eval_all.py`) and the full per-tier records: `t1best/` (Tier 1), `t2ready_20261031/` … `t2r9_20261104/` (Tier 2), `t3ready_20261102/` (Tier 3), `rolling_start.md` |
| [`robotic_grounding/scripts/retarget/soma_to_g1.py`](robotic_grounding/scripts/retarget/soma_to_g1.py) | SOMA → G1 retargeting that made every training input |
| [`hoi_estimation/`](hoi_estimation/README.md) | Tier-3 video reconstruction pipeline (human, camera, object, supports) with its reproducibility runbooks |
| [`TIER3_SUBMISSION.md`](TIER3_SUBMISSION.md) | Pointer to the Tier-3 record and the reconstruction code |

## Environment

Training machine: NVIDIA RTX PRO 6000 Blackwell, driver 580.82.09 (CUDA 13.0), Ubuntu 22.04. CUDA comes from pip
wheels, so no CUDA toolkit is needed. Details: [`environment/README.md`](environment/README.md).

| env | used for | install |
|---|---|---|
| `flashchord` | training, policy export, official evaluation (Python 3.11) | `setup_env.sh --steps conda,stock,flashchord` |
| `v2dkit` | official submission kit: packer and checks | `setup_env.sh --steps kit` |
| `v2d` | G1 retargeting and input builders (Isaac Sim 5.1.0, Isaac Lab 2.3.2) | `setup_env.sh --steps conda,retarget` |
| `zx_g2_builder` | the G2 step of the Tier-2 input builder | `setup_env.sh --steps zxg2` |

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/RCL-Robotics/nv-v2d-challenge-rcl-group-solution.git v2d
cd v2d
export V2D_WORK=/data/v2d/work                      # everything the scripts write (no spaces)
export V2D_KIT=/data/v2d/v2d_submission_kit         # the official submission kit
export V2D_DATA=/data/v2d/video_to_data_challenge   # challenge data root (contains track_2/)
export V2D_RUNTIME=$V2D_WORK/video_to_data_8c778f9/robotic_grounding/flash_chord   # official runtime
bash scripts/final/setup_env.sh --steps conda,stock,flashchord,kit
bash scripts/final/setup_env.sh --steps check       # compare installed envs with environment/*.pip-freeze.txt
```

## Data and kit

- **Challenge data:** Hugging Face dataset `nvidia/video_to_data_challenge`, revision `5f68335f`, folder `track_2/`
  (`bash scripts/final/get_data.sh`, needs the `hf` CLI).
- **Official runtime:** `nvidia-isaac/video_to_data` @ `8c778f9`, `robotic_grounding/flash_chord` (installed by
  `setup_env.sh --steps stock`). All evaluations use this stock runtime, world count 1, export tolerance 1e-4.
- **Submission kit:** the Oct-11 kit, `KIT_MANIFEST.json` sha256 `398947a7…` (packer `tools/pack_submission.py`,
  evaluator `flash_chord/scripts/evaluate_policy.py`).

Copy `scripts/final/site.env.example` outside the repository, set its paths, and pass it with `--site FILE` to every
script. Never put tokens or keys in it.

## Training configuration

### Common recipe

- **Trainer:** FlashCHORD PPO on G1 + Dex3 (`embodiment: g1_dex3`, `action: sonic_joint_residual`, ReconBody
  observation / objective / termination), Hydra experiment `g1_recon_body_coord_rsi_r2` (**C3r2**): the W0-P
  coordinated actor (shared hand network + body network, ~228k parameters, exported as a plain
  `[1024, 1024, 768, 384]` MLP), RSI v3 reference-state-initialisation resets and curriculum with a help schedule,
  a 45k-iteration cap with help-zero early stop, and an extra object task reward
  `exp(-10·|p - p_ref|) · exp(-1·angle(q, q_ref))` with weight 5.0 (stripped at export).
- **Seed:** `reset.seed=42` unless a record says otherwise; every stage's seed, overrides and iteration count are
  recorded and replayed by `train_tierN.sh`.
- **Export / evaluation:** policies exported to `.safetensors` at tolerance 1e-4, evaluated with the kit evaluator on
  the stock runtime, packed with the kit packer and each episode's frame reference.
- **Exact code per stage:** every stage runs in a git worktree of its recorded training commit (all of them are
  ancestors of `main`), e.g. `b39b69b` (rolling start), `61fa5d3` (fc_r2), `41403d3` (Tier-3 trainer). Training is
  GPU non-deterministic: a rerun reproduces the recipe and policy class, not the bytes.

| Variant | Experiment config | Change from C3r2 | Used in |
|---|---|---|---|
| C3r2 | `g1_recon_body_coord_rsi_r2` | — | Tiers 1, 2, 3 |
| C3r2-runin (rolling start) | `g1_recon_body_coord_rsi_r2_runin` | `reset.run_in_steps: 0 → 20`, `reset.min_horizon: 10 → 30`: the reference clock runs during the last 20 of the 50 hold steps, so robot and object already move with the reference when released ([details](robotic_grounding/flash_chord/scripts/track2/rolling_start.md)) | Tiers 1, 2 |
| C3r2-runin-hard | `g1_recon_body_coord_rsi_r2_runin_hard` | rolling start + hard-start sampling (10 % frame 0, 80 % by the failure hazard of the start bin), `start_frame_shift 20`, 90k iterations, no help-zero early stop | Tier 2 |
| C3r2-helpfix-hard | `g1_recon_body_coord_rsi_r2_helpfix_hard` | success-driven help + hard-start sampling, no rolling start | Tier 2 |

Per-episode fixes are Hydra overrides on top of these experiments, e.g. the Tier-1 T1fix / T1improve arms
(`t1best/configs/arms.json`): rotation-weighted object reward `objective.object_task_rotation_beta`, tighter or
relaxed `termination.object_*_threshold`, position-only object reward after release; and Tier-3 per-episode
termination overrides (`termination.palm_position_threshold`, object thresholds).

A single training run (inside the `flashchord` env, from `robotic_grounding/flash_chord`):

```bash
python scripts/train_rl.py experiment=g1_recon_body_coord_rsi_r2_runin reset.seed=42 \
  training.max_iterations=45000 <task / scene overrides of the episode>
```

The task, scene and termination overrides of each episode are generated by
`robotic_grounding/flash_chord/scripts/track2/launch.py` and stored in the tier records; use `train_tierN.sh` rather
than writing them by hand.

### Tier 1 — clean reference

Record: [`t1best/t1ready_20261031/`](robotic_grounding/flash_chord/scripts/track2/t1best/t1ready_20261031/README.md).

- **Input:** the released clean Tier-1 HOI motion, converted and retargeted to G1.
- **Set `all7`:** 30 checkpoints from 54 training stages in 30 lineages: C3r2_runin and C3r2_wide runs from scratch,
  teammate run-in chains, T1fix forks and 7 episodes replaced by the T1improve wave (ep01, 05, 10, 14, 15, 19, 28; arm
  overrides in `t1best/t1improve/arms/<arm>/overrides.yaml`).
- **Score:** 20 complete official trials; mean AUC 0.93113 (sd 0.0033); submitted trial 05 (closest to the mean).

```bash
bash scripts/final/train_tier1.sh --site site.env --episode 8 --dry-run   # print the stages of episode 8
bash scripts/final/train_tier1.sh --site site.env --episode 8             # retrain all its stages
```

### Tier 2 — synthetic-noise reference

Records: [`t2ready_20261031/`](robotic_grounding/flash_chord/scripts/track2/t2ready_20261031/README.md) →
[`t2r7`](robotic_grounding/flash_chord/scripts/track2/t2r7_20261102/README.md) →
[`t2r8`](robotic_grounding/flash_chord/scripts/track2/t2r8_20261103/README.md) →
[`t2r9_20261104/`](robotic_grounding/flash_chord/scripts/track2/t2r9_20261104/README.md) (submitted).

- **Inputs**, built only from `track_2/tier_2_synthetic_noise` (no Tier-1 or clean data) by
  `build_inputs_tier2.sh`:

  | input | what it is | episodes in the submitted file |
  |---|---|---:|
  | `raw` | SOMA conversion + G1 retarget of the released noisy episodes | 4 |
  | `k3` | our K3 Kalman smoother of the object track | 1 |
  | `k3fix` | K3 with root-cause fixes and per-episode gates G1–G4 | 18 |
  | `g2` | contact-model smoother of the object track | 7 |

- **Training:** C3r2, C3r2-runin, C3r2-runin-hard and C3r2-helpfix-hard runs on each input, some continued for long
  run-in chains (up to ~420k iterations); trained on our RTX PRO 6000 server and an RTX 5090 workstation.
- **Selection:** 9 pre-registered rounds scored only with own-input AUC and a frozen consensus reference built from
  Tier-2 tracks (`refs/B`). Round 9 kept 27 of round 8's picks and replaced episodes 10, 17 and 21.

```bash
bash scripts/final/build_inputs_tier2.sh --site site.env --steps verify   # check the 30 inputs used
bash scripts/final/train_tier2.sh --site site.env --episode 17
bash scripts/final/select_tier2.sh --site site.env                       # print the policy chosen per episode
```

### Tier 3 — our own video reconstruction

Record: [`t3ready_20261102/`](robotic_grounding/flash_chord/scripts/track2/t3ready_20261102/README.md); pipeline:
[`hoi_estimation/`](hoi_estimation/README.md) (runbooks and `run_all_tier3.sh` in `hoi_estimation/reproducibility/`).

- **Reconstruction:** human, camera and object from the monocular exo video (s1), object track re-estimated by an
  agent (opus_v2), frame-0 correction, and video-fitted supports (table / desk / chair / stool) for 14 episodes.
- **Data prep:** G1 retarget with `soma_to_g1.py` gated by checks G1–G10; GT-free support placement and hand–object
  gap check (sets `termination.palm_position_threshold=0.25` for 8 episodes, training only).
- **f0 training (all 30):** `experiment=g1_recon_body_coord_rsi_r2 reset.seed=42 training.max_iterations=45000`, help-zero
  early stop (deadline 35156, ramp 3906, tail 5860), final checkpoint, no checkpoint selection.
- **C90 retry (14 failing episodes):** resume from `model_44922` to 90000 iterations with a second RSI help cycle
  (`reset.start_update=44922`) and rule-computed termination overrides; replaced f0 for 11 episodes by a
  pre-registered acceptance rule.

```bash
bash scripts/final/build_inputs_tier3.sh --site site.env --dry-run
bash scripts/final/train_tier3.sh --site site.env --episode 7
```

## Quick start: train, evaluate, pack

Every script takes `--site FILE`, `--dry-run` and `--help`; real runs log their commands to `$V2D_WORK/logs/`.

```bash
bash scripts/final/selftest.sh                                        # syntax check + dry run of every script
bash scripts/final/train_tier1.sh --site site.env --episode 8 --iterations 50 --gpu 0   # smoke test
bash scripts/final/eval_tier1.sh  --site site.env --episode 28        # official evaluator; --episode all = 30 episodes
bash scripts/final/pack_tier1.sh  --site site.env --commit-url <URL>  # kit packer + offline upload check
```

Evaluation writes rollouts to `$V2D_WORK/eval/tierN/t01/results/` and per-episode AUC, SP-SR, MP-SR and MPPE to
`$V2D_WORK/eval/tierN/summary.tsv`. With `REQUIRE_IDLE_GPU=1`, every launch refuses a GPU that already runs a process.

## Reproducing the submitted files

See [`REPRODUCE.md`](REPRODUCE.md). Byte-identical re-packing and re-evaluation of the submitted policies need our
verification package (four zips, ~5.4 GB, provided with the submission; their sha256 values are in `REPRODUCE.md`).
The submitted files cite commits of our original development repository; those commits are in this repository's
history with the same hashes.

## Selection protocol

- Every selection rule was written and hash-locked (sha256) before the exports, evaluations or scores it governs; the
  pre-registrations, amendments and lock files are in the tier records.
- Candidate checkpoints get repeated fresh official evaluations; the packed trial is the one closest to the mean of the
  complete trials, never the best one.
- Disclosed deviations (e.g. Tier 1 `TOPUP_AMEND1`, Tier 3 open items) and grey areas are listed in each record.

## Citation

```bibtex
@misc{zhang2026dexduetlearningcontactpreservingbimanual,
      title={DexDuet: Learning Contact-Preserving Bimanual Dexterity from Human Demonstrations},
      author={Haoran Zhang and Haokai Ding and Han Zhou and Zuxing Lu and Heng Zhang and Xingxing Zuo},
      year={2026},
      eprint={2611.30495},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2611.30495},
}
```

## Credits and license

Team **MBZUAI-RCL Group**: Zuxing Lu, Haoran Zhang, Han Zhou (Tier-3 video reconstruction, `hoi_estimation/`).

Built on NVIDIA [Video to Data](https://github.com/nvidia-isaac/video_to_data). Copyright (c) 2026 NVIDIA CORPORATION &
AFFILIATES; source code under Apache-2.0 and documentation under CC-BY-4.0 — see [`LICENSE`](LICENSE).
