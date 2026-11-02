runin_g2_ep15_best model_62000 from /mnt/gfs/home/zuxinglu/workspace/runs/2026-10-24-tier2-runin-g2/ep15: G2 input, pure rolling start, run still training (180k)
runin_hard_g2_ep19_best model_198000 from /mnt/gfs/home/zuxinglu/workspace/runs/2026-10-25-tier2-runin-hard-g2/cont/ep19: G2 input, rolling start + hard starts, continuation 180k-360k still training
Copied from the blockelite 5090 (~/workspace/runs) on 2026-10-27 (user request): checkpoints of ep15 and ep19. Each folder: model_N + rsi_N (RSI sidecar)
+ the run's hydra config. Input: G2 = zuxing.lu/data/human_motion_data_tier2_g2-2026-10-24 (code dev.zuxing.hard_start
82837c8, experiments g1_recon_body_coord_rsi_r2_runin / g1_recon_body_coord_rsi_r2_runin_hard, seed 42).
