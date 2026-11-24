# NV V2D Challenge — Track 2 solution of MBZUAI-RCL Group

[**Paper**](https://arxiv.org/abs/2611.30495) · [**Project page**](https://rcl-robotics.github.io/Dexduet.io/) · [**Reproduce**](REPRODUCE.md)

Code and training records of our Track 2 submissions to the NVIDIA Video-to-Data (V2D) Challenge: Unitree G1 + Dex3
policies trained with FlashCHORD, one per episode, 30 episodes per tier. Built on
[nvidia-isaac/video_to_data](https://github.com/nvidia-isaac/video_to_data).

## Results

Official kit evaluator. Tiers 2 and 3 are scored against each policy's own input, not the hidden leaderboard reference.

| Tier | Input | ADD-AUC | SP-SR | MP-SR | MPPE |
|---|---|---:|---:|---:|---:|
| 1 | clean motion | 0.931 | 1.000 | 1.000 | 0.93 cm |
| 2 | noisy motion, filtered | 0.812 | 0.933 | 0.933 | 3.38 cm |
| 3 | our video reconstruction | 0.676 | 0.733 | 0.700 | 16.39 cm |

## Setup

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/RCL-Robotics/nv-v2d-challenge-rcl-group-solution.git v2d && cd v2d
export V2D_WORK=/data/v2d/work                      # outputs
export V2D_KIT=/data/v2d/v2d_submission_kit         # official submission kit
export V2D_DATA=/data/v2d/video_to_data_challenge   # HF nvidia/video_to_data_challenge @ 5f68335f
export V2D_RUNTIME=$V2D_WORK/video_to_data_8c778f9/robotic_grounding/flash_chord
bash scripts/final/setup_env.sh --steps conda,stock,flashchord,kit
bash scripts/final/get_data.sh
cp scripts/final/site.env.example site.env          # edit paths; pass with --site
```

Tested on RTX PRO 6000 Blackwell, driver 580 (CUDA 13), Ubuntu 22.04. Environments: [`environment/`](environment/README.md).

## Training

All tiers use the FlashCHORD PPO experiment `g1_recon_body_coord_rsi_r2` (C3r2): a coordinated per-hand actor,
assisted reference-state initialisation (A-RSI) and an object task reward. Variants:

| Config | Change | Used in |
|---|---|---|
| `g1_recon_body_coord_rsi_r2` | base | Tiers 1–3 |
| `..._rsi_r2_runin` | rolling start: reference clock runs in the last 20 of the 50 hold steps | Tiers 1, 2 |
| `..._rsi_r2_runin_hard` | rolling start + hard-start sampling, 90k iterations | Tier 2 |
| `..._rsi_r2_helpfix_hard` | success-driven help + hard-start sampling | Tier 2 |

- **Tier 1:** clean motion retargeted to G1; 54 stages in 30 lineages plus per-episode reward and termination fixes.
- **Tier 2:** noisy motion smoothed by our filters (K3fix 18, G2 7, raw 4, K3 1 episodes); 9 pre-registered selection rounds.
- **Tier 3:** human, camera and object reconstructed from the monocular video ([`hoi_estimation/`](hoi_estimation/README.md));
  45k-iteration f0 runs, 14 hard episodes continued to 90k.

Each script replays the recorded commit, overrides and seed of every stage (`--dry-run` prints the commands):

```bash
bash scripts/final/train_tier1.sh --site site.env --episode 8
bash scripts/final/eval_tier1.sh  --site site.env --episode 8     # --episode all for 30 episodes
bash scripts/final/pack_tier1.sh  --site site.env --commit-url <URL>
```

Same for tiers 2 and 3; inputs are built with `build_inputs_tier{2,3}.sh`. Script list:
[`scripts/final/`](scripts/final/README.md). Full per-tier records:
[`robotic_grounding/flash_chord/scripts/track2/`](robotic_grounding/flash_chord/scripts/track2/).

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

## License

Apache-2.0 (code) and CC-BY-4.0 (docs), inherited from NVIDIA Video to Data; see [`LICENSE`](LICENSE).
