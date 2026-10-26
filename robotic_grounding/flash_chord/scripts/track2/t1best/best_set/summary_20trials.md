# Tier-1 best set: 20 official evaluation trials (+ ep28 W2PD alternative)

Generated 2026-10-26 22:33:21 by `t1best20_driver.py summary`. Protocol: `PREREG.md`. 620 ok and 0 failed evaluations of 620 planned. **Nothing here is selected, labelled for upload or uploaded.** The packed CSVs are named `*_LOCAL_NOT_FOR_UPLOAD.csv` and carry a placeholder commit URL.

## Seeds

The official evaluator has no selectable seed. Every trial ran with the default seeds, read from each evaluated file's own resolved config (seeds.tsv, one row per evaluation). Distinct seed signatures over all evaluations:

- `delay=0(+g0..4,0-2) reset=42 scale_seed=0(1.0/1.0) wrench=0 vertex=0/0 ppo=42`

Differences between trials therefore come from GPU floating-point non-determinism, not from a seed. A rerun with the same seeds is not expected to reproduce a trial bit for bit.

## Set score per trial (kit packer + metric_code, float32 reference)

| trial | AUC | SP-SR | MP-SR | MPPE cm | evaluator-JSON mean AUC | ep28 B AUC | ep28 alt AUC | GPUs |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 01 | 0.9246 | 1.000 | 0.967 | 1.006 | 0.9245 | 0.7873 | 0.8022 | 0 1 2 3 5 6 7 |
| 02 | 0.9255 | 1.000 | 0.967 | 0.995 | 0.9252 | 0.7877 | 0.8038 | 0 1 2 3 5 6 7 |
| 03 | 0.9260 | 1.000 | 0.967 | 1.003 | 0.9257 | 0.7981 | 0.8043 | 0 1 2 3 5 6 7 |
| 04 | 0.9262 | 1.000 | 0.967 | 0.994 | 0.9260 | 0.7898 | 0.8063 | 0 1 2 3 5 6 7 |
| 05 | 0.9256 | 1.000 | 0.967 | 0.992 | 0.9253 | 0.7928 | 0.8050 | 0 1 2 3 5 6 7 |
| 06 | 0.9265 | 1.000 | 0.967 | 0.982 | 0.9261 | 0.7918 | 0.8062 | 0 1 2 3 5 6 7 |
| 07 | 0.9264 | 1.000 | 0.967 | 0.992 | 0.9262 | 0.7919 | 0.8064 | 0 1 2 3 5 6 7 |
| 08 | 0.9278 | 1.000 | 0.967 | 0.982 | 0.9275 | 0.7891 | 0.8047 | 0 1 2 3 5 6 7 |
| 09 | 0.9270 | 1.000 | 0.967 | 0.988 | 0.9267 | 0.7930 | 0.8107 | 0 1 2 3 5 6 7 |
| 10 | 0.9264 | 1.000 | 0.967 | 0.989 | 0.9261 | 0.7953 | 0.8093 | 0 1 2 3 5 6 7 |
| 11 | 0.9274 | 1.000 | 0.967 | 0.988 | 0.9271 | 0.7957 | 0.8084 | 0 1 2 3 5 6 7 |
| 12 | 0.9270 | 1.000 | 0.967 | 0.989 | 0.9267 | 0.7945 | 0.8079 | 0 1 2 3 5 6 7 |
| 13 | 0.9272 | 1.000 | 0.967 | 0.975 | 0.9268 | 0.7912 | 0.8079 | 0 1 2 3 5 6 7 |
| 14 | 0.9279 | 1.000 | 0.967 | 0.984 | 0.9275 | 0.7954 | 0.8071 | 0 1 2 3 5 6 7 |
| 15 | 0.9267 | 1.000 | 0.967 | 0.995 | 0.9265 | 0.7933 | 0.8067 | 0 1 2 3 5 6 7 |
| 16 | 0.9273 | 1.000 | 0.967 | 0.973 | 0.9270 | 0.7994 | 0.8107 | 0 1 2 3 5 6 7 |
| 17 | 0.9124 | 0.967 | 0.933 | 2.272 | 0.9122 | 0.7919 | 0.8090 | 0 1 2 3 5 6 7 |
| 18 | 0.9084 | 0.967 | 0.933 | 2.118 | 0.9081 | 0.7842 | 0.8074 | 0 1 2 3 5 6 7 |
| 19 | 0.9268 | 1.000 | 0.967 | 0.986 | 0.9265 | 0.7982 | 0.8112 | 0 1 2 3 5 6 7 |
| 20 | 0.9260 | 1.000 | 0.967 | 0.987 | 0.9259 | 0.7967 | 0.8129 | 0 1 2 3 5 6 7 |

## Over trials

| metric | n | mean | sd | min | max | sem / note |
|---|---:|---:|---:|---:|---:|---|
| auc | 20 | 0.9250 | 0.0051 | 0.9084 | 0.9279 | 0.0011 |
| sp_sr | 20 | 0.9967 | 0.0103 | 0.9667 | 1.0000 | 0.0023 |
| mp_sr | 20 | 0.9633 | 0.0103 | 0.9333 | 0.9667 | 0.0023 |
| mppe | 20 | 1.1095 | 0.3723 | 0.9729 | 2.2722 | 0.0832 |
| auc_float64 | 20 | 0.9250 | 0.0051 | 0.9084 | 0.9279 | 0.0011 |
| best_trial | 20 | 0.9279 |  |  |  | trial 14; SP-SR 1.000; MP-SR 0.967; MPPE 0.984 cm; seeds delay=0(+g0..4,0-2) reset=42 scale_seed=0(1.0/1.0) wrench=0 vertex=0/0 ppo=42 |
| worst_trial | 20 | 0.9084 |  |  |  | trial 18; SP-SR 0.967; MP-SR 0.933; MPPE 2.118 cm; seeds delay=0(+g0..4,0-2) reset=42 scale_seed=0(1.0/1.0) wrench=0 vertex=0/0 ppo=42 |
| closest_to_mean_trial | 20 | 0.9246 |  |  |  | trial 01; SP-SR 1.000; MP-SR 0.967; MPPE 1.006 cm; seeds delay=0(+g0..4,0-2) reset=42 scale_seed=0(1.0/1.0) wrench=0 vertex=0/0 ppo=42 |
| ep28_alt_minus_B_auc | 20 | 0.0145 | 0.0033 | 0.0063 | 0.0232 | 0.0007 |
| ep28_alt_minus_B_sp_sr | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ep28_alt_minus_B_mp_sr | 20 | 0.3000 | 0.4702 | 0.0000 | 1.0000 | 0.1051 |
| ep28_alt_minus_B_mppe | 20 | -0.9562 | 0.6666 | -2.2085 | 0.0446 | 0.1490 |

- Best trial 14 (AUC 0.9279), worst trial 18 (AUC 0.9084), closest to the mean: trial 01 (AUC 0.9246). All three used the same seeds: `delay=0(+g0..4,0-2) reset=42 scale_seed=0(1.0/1.0) wrench=0 vertex=0/0 ppo=42`.
- Best-of-trials is optimistic: an organiser rerun (rules 2.9.b) lands near the mean, not the best.

## Per item (metric_code per episode over scored trials)

| item | object | file | n | AUC mean | sd | min | max | SP pass | MP pass | MPPE mean | prior mean AUC (n) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ep00 | tall_bar_stool | 2e3079703b (ok) | 20 | 0.9976 | 0.0008 | 0.9949 | 0.9985 | 20 | 20 | 0.43 | 0.9980 (3, other export) |
| ep01 | vacuum | ca085b7392 (ok) | 20 | 0.7582 | 0.1218 | 0.2407 | 0.7896 | 19 | 19 | 3.67 | 0.7901 (3, other export) |
| ep02 | blue_easter_basket | be9833ef63 (existing) | 20 | 0.9776 | 0.0009 | 0.9755 | 0.9790 | 20 | 20 | 0.53 | 0.9782 (5, same file) |
| ep03 | contigo_coffee_mug | 9224384118 (ok) | 20 | 0.9619 | 0.1024 | 0.5267 | 0.9882 | 19 | 19 | 2.50 | 0.9858 (3, other export) |
| ep04 | baseball_bat | 52042a1294 (existing) | 20 | 0.7507 | 0.0030 | 0.7447 | 0.7551 | 20 | 20 | 1.56 | 0.7415 (3, same file) |
| ep05 | big_black_box | 0b9b4a6557 (ok) | 20 | 0.9473 | 0.0022 | 0.9408 | 0.9495 | 20 | 20 | 0.79 | 0.9494 (3, other export) |
| ep06 | monitor | 842cb67303 (ok) | 20 | 0.9679 | 0.0006 | 0.9660 | 0.9689 | 20 | 20 | 0.80 | 0.9695 (3, other export) |
| ep07 | iron | a54f0c6ea1 (existing) | 20 | 1.0000 | 0.0001 | 0.9996 | 1.0000 | 20 | 20 | 0.32 | 1.0000 (5, same file) |
| ep08 | skinny_wood_chair | 2f98f46952 (ok) | 20 | 0.9673 | 0.0003 | 0.9664 | 0.9678 | 20 | 20 | 0.94 | 0.9676 (3, other export) |
| ep09 | big_red_bowl | 1a4086d09e (existing) | 20 | 0.9290 | 0.0025 | 0.9247 | 0.9343 | 20 | 20 | 0.74 | 0.9289 (5, same file) |
| ep10 | white_desk | c085b62afe (ok) | 20 | 0.8693 | 0.0018 | 0.8655 | 0.8718 | 20 | 20 | 1.31 | 0.8671 (3, other export) |
| ep11 | light_blue_book | 9f60a9cb59 (existing) | 20 | 0.9768 | 0.0016 | 0.9726 | 0.9800 | 20 | 20 | 0.64 | 0.9773 (5, same file) |
| ep12 | black_platform | 3b2c52d794 (ok) | 20 | 0.9856 | 0.0023 | 0.9785 | 0.9880 | 20 | 20 | 0.58 | 0.9881 (3, other export) |
| ep13 | snack_box | 5cec84bb80 (ok) | 20 | 0.9786 | 0.0173 | 0.9262 | 0.9916 | 20 | 20 | 0.59 | 0.9850 (3, other export) |
| ep14 | round_table | 29bc18a5e3 (ok) | 20 | 0.8270 | 0.0037 | 0.8182 | 0.8320 | 20 | 20 | 1.50 | 0.8250 (3, other export) |
| ep15 | wet_floor_sign | c571cb9ced (existing) | 20 | 0.9376 | 0.0015 | 0.9344 | 0.9406 | 20 | 20 | 0.88 | 0.9367 (5, same file) |
| ep16 | spray_bottle | baefdf33e9 (existing) | 20 | 0.9896 | 0.0027 | 0.9838 | 0.9932 | 20 | 20 | 0.51 | 0.9877 (5, same file) |
| ep17 | traffic_cone | 8cbe742c58 (existing) | 20 | 0.9424 | 0.0074 | 0.9290 | 0.9547 | 20 | 20 | 1.06 | 0.9492 (5, same file) |
| ep18 | blue_trash_can | ff7e980fc0 (existing) | 20 | 0.9933 | 0.0006 | 0.9921 | 0.9943 | 20 | 20 | 0.52 | 0.9931 (5, same file) |
| ep19 | cane | 5f5b0fc6b8 (ok) | 20 | 0.8957 | 0.0020 | 0.8916 | 0.9003 | 20 | 20 | 1.02 | 0.8952 (3, other export) |
| ep20 | rice_cooker | 409796a270 (ok) | 20 | 0.9855 | 0.0021 | 0.9790 | 0.9881 | 20 | 20 | 0.49 | 0.9862 (3, other export) |
| ep21 | tall_bar_stool | ffc06d795a (ok) | 20 | 0.7548 | 0.0011 | 0.7522 | 0.7566 | 20 | 20 | 1.50 | 0.7557 (3, other export) |
| ep22 | step_ladder | b86140bb17 (ok) | 20 | 0.9761 | 0.0014 | 0.9715 | 0.9778 | 20 | 20 | 0.71 | 0.9764 (3, other export) |
| ep23 | toy_stroller | 674841934b (ok) | 20 | 0.9908 | 0.0002 | 0.9902 | 0.9911 | 20 | 20 | 0.47 | 0.9911 (3, other export) |
| ep24 | watering_can | 4c99a3de7b (ok) | 20 | 0.9994 | 0.0001 | 0.9992 | 0.9996 | 20 | 20 | 0.29 | 0.9994 (3, other export) |
| ep25 | cyan_water_bottle | fa932e0d4e (ok) | 20 | 0.7481 | 0.0068 | 0.7375 | 0.7576 | 20 | 20 | 2.01 | 0.7387 (3, other export) |
| ep26 | skateboard | af8d40d74a (ok) | 20 | 0.9975 | 0.0006 | 0.9964 | 0.9984 | 20 | 20 | 0.38 | 0.9976 (3, other export) |
| ep27 | woven_basket | 656d876697 (ok) | 20 | 0.9965 | 0.0010 | 0.9928 | 0.9976 | 20 | 20 | 0.42 | 0.9972 (3, other export) |
| ep28 | squeegee | 3d47acf3fd (ok) | 20 | 0.7929 | 0.0040 | 0.7842 | 0.7994 | 20 | 0 | 4.42 | 0.7949 (3, other export) |
| ep29 | rolling_luggage | bcb1c5412d (ok) | 20 | 0.8536 | 0.0067 | 0.8358 | 0.8640 | 20 | 20 | 1.70 | 0.8565 (3, other export) |
| ep28alt | squeegee | 2bddd57d89 (ok) | 20 | 0.8074 | 0.0027 | 0.8022 | 0.8129 | 20 | 6 | 3.47 | 0.8074 (6, other export) |

## ep28: banked Stage-2 B model_58000 vs wave-2 W2PD model_66000

Paired over 20 trials in which both were scored (alt never enters the set mean).
- auc: alt - B mean 0.0145 (sd 0.0033, min 0.0063, max 0.0232)
- sp_sr: alt - B mean 0.0000 (sd 0.0000, min 0.0000, max 0.0000)
- mp_sr: alt - B mean 0.3000 (sd 0.4702, min 0.0000, max 1.0000)
- mppe: alt - B mean -0.9562 (sd 0.6666, min -2.2085, max 0.0446)
- B: AUC 0.7929 +- 0.0040, MP pass 0/20, MPPE 4.42 cm; alt: AUC 0.8074 +- 0.0027, MP pass 6/20, MPPE 3.47 cm.
- Using the alt in a submission needs commit 7fbc227 merged and pushed (user decision); not decided here.

## Files

- `seeds.tsv`: per (trial, item): every seed key=value, GPU index + UUID, host, start/end, the runner command line, the kit commit, the evaluator digests found in the results, JSON/parquet sha256.
- `per_trial.csv`, `per_episode.csv`, `summary.csv`; `scores/tNN/` packed CSVs (LOCAL, NOT FOR UPLOAD) and metric_code JSON.
- `runs/<item>/tNN/run.json` + `try*/results/` (evaluator JSON + parquet).
