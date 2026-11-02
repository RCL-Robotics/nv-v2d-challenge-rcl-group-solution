# Tier 2 K3-only file: trial choice

Written 2026-11-02 15:55:03 CST by `tools/k3only_choose.py` (sha256 667bd8c22747cae72ac19946ac829d86ae92f86958c5df5cf24c92e361331f4d). Inputs: snapshot copies in `inputs/` (sha256 in CHOICE.json and INPUTS_SNAPSHOT.sha256).

**Composition (fixed, not chosen here):** K3FULL of `t2_best5_20261024` (PREREG section 4: PREREG_v3 as VERDICT.json gives it; PENDING maps to raw). 22 episodes on our K3fix / K3 easy-clip runs (final.json), 8 on our raw C3r2_tier2 runs (final.json): gate-FAIL 00, 05, 15, 19, 22, 26, 28, plus ep04 (gate PASS, VERDICT PENDING -> raw; VERDICT.json was last written before its K3fix run finished). No G2, nothing of Zuxing's.

**Rule:** trial k = run k (its ok try) of every episode's job; T_k = mean over the 30 episodes of own-input AUC; chosen = the trial whose T_k is closest to the mean of T_1..T_5, ties to the lower k.

| trial | own AUC | own SP-SR | own MP-SR | own MPPE (cm) | abs(T_k - mean) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.636577 | 0.8000 | 0.8000 | 15.948 | 0.010159 |
| 2 (chosen) | 0.637775 | 0.8000 | 0.8000 | 15.661 | 0.008961 |
| 3 | 0.659293 | 0.8333 | 0.8333 | 11.007 | 0.012557 |
| 4 | 0.637337 | 0.8000 | 0.8000 | 15.123 | 0.009399 |
| 5 | 0.662699 | 0.8333 | 0.8333 | 11.257 | 0.015963 |
| mean | 0.646736 | 0.8133 | 0.8133 | 13.799 | |

**CHOSEN TRIAL: 2** (own-input AUC 0.637775, SP 0.8000, MP 0.8000, MPPE 15.661 cm).

Own-input composites mix references (raw / K3fix / K3); they are not leaderboard numbers.

| ep | job | input | VERDICT | run-chosen own AUC | 5-run own AUC mean |
|---:|---|---|---|---:|---:|
| 00 | ep00_raw_final41043 | raw | RAW | 0.2632 | 0.2619 |
| 01 | ep01_k3fix_final44922 | k3fix | K3FIX | 0.1411 | 0.1421 |
| 02 | ep02_k3fix_final34093 | k3fix | K3FIX | 0.9214 | 0.9307 |
| 03 | ep03_k3fix_final27068 | k3fix | K3FIX | 0.9847 | 0.9824 |
| 04 | ep04_raw_final44922 | raw | PENDING | 0.1104 | 0.1099 |
| 05 | ep05_raw_final44922 | raw | RAW | 0.2560 | 0.2546 |
| 06 | ep06_k3fix_final28420 | k3fix | K3FIX | 0.9239 | 0.9213 |
| 07 | ep07_k3fix_final18529 | k3fix | K3FIX | 0.9794 | 0.9785 |
| 08 | ep08_k3easy_final17496 | k3 | K3FIX | 0.9248 | 0.9259 |
| 09 | ep09_k3fix_final39449 | k3fix | K3FIX | 0.5914 | 0.5810 |
| 10 | ep10_k3fix_final25319 | k3fix | K3FIX | 0.6072 | 0.6154 |
| 11 | ep11_k3fix_final29175 | k3fix | K3FIX | 0.8826 | 0.8817 |
| 12 | ep12_k3fix_final23178 | k3fix | K3FIX | 0.7245 | 0.7196 |
| 13 | ep13_k3fix_final39254 | k3fix | K3FIX | 0.8440 | 0.8406 |
| 14 | ep14_k3fix_final44922 | k3fix | K3FIX | 0.5566 | 0.5569 |
| 15 | ep15_raw_final44922 | raw | RAW | 0.7017 | 0.6996 |
| 16 | ep16_k3fix_final32063 | k3fix | K3FIX | 0.7652 | 0.7670 |
| 17 | ep17_k3fix_final29845 | k3fix | K3FIX | 0.7536 | 0.7543 |
| 18 | ep18_k3fix_final18911 | k3fix | K3FIX | 0.9312 | 0.9297 |
| 19 | ep19_raw_final44922 | raw | RAW | 0.1663 | 0.1659 |
| 20 | ep20_k3easy_final18450 | k3 | K3FIX | 0.9106 | 0.9089 |
| 21 | ep21_k3fix_final44922 | k3fix | K3FIX | 0.5522 | 0.5469 |
| 22 | ep22_raw_final44922 | raw | RAW | 0.4837 | 0.4777 |
| 23 | ep23_k3easy_final19162 | k3 | K3FIX | 0.9503 | 0.9526 |
| 24 | ep24_k3fix_final20636 | k3fix | K3FIX | 0.9483 | 0.9483 |
| 25 | ep25_k3fix_final42819 | k3fix | K3FIX | 0.0659 | 0.3445 |
| 26 | ep26_raw_final34948 | raw | RAW | 0.2728 | 0.2699 |
| 27 | ep27_k3fix_final20863 | k3fix | K3FIX | 0.9600 | 0.9754 |
| 28 | ep28_raw_final44922 | raw | RAW | 0.0337 | 0.0322 |
| 29 | ep29_k3fix_final17358 | k3fix | K3FIX | 0.9269 | 0.9268 |
