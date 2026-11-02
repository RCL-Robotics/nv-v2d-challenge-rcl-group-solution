# t2_consensus_r7_20261102: phase-S selection (PREREG_R7 5.3 / 5.4)

Generated 2026-11-02 10:47:07 CST by `scripts/t2cons_r7_consensus.py` (e7545dfa7fdff068...), pinned in `PREREG_R7_PINS_b.md` (73d17f24f63aaef1...). 154 new screen + 245 reused round-6 candidates x 3 official runs, each run scored against round 1's frozen consensus refs/B (variant B) and gated against its own input (5.1). 23 incumbent episodes: finalist = the S-eligible candidate with the highest 3-run consensus mean, provided it is >= m_P_frozen + 0.01. 7 STRICT episodes: the top 2 S-eligible candidates.

Finalists: 19; validation_failed candidates: 0; reused round-6 runs reproduce scores_s6: True.

| ep | incumbent (input) | m_P_frozen | bar | candidates / eligible / qualifying | best candidate: cons_S (floor) | **finalist(s)** |
|---|---|---:|---:|---|---|---|
| 00 | STRICT: none |  |  | 6 / 6 / 2 | ep00_z5k3fix_90000: 0.7198 (3/3) | **ep00_z5k3fix_90000, ep00_z5k3fix_89500** (S-eligible) |
| 01 | ep01_k3fix_final44922 (k3fix) | 0.1713 | 0.1813 | 31 / 0 / 0 | ep01_zxg2_90000: 0.1864 (0/3) | **-** |
| 02 | ep02_k3fix_final34093 (k3fix) | 0.8681 | 0.8781 | 11 / 10 / 2 | ep02_z5q_180000: 0.8949 (3/3) | **ep02_z5q_180000** |
| 03 | ep03_k3fix_final27068 (k3fix) | 0.9215 | 0.9315 | 15 / 11 / 0 | ep03_k3fix_22500: 0.9192 (3/3) | **-** |
| 04 | ep04_zxg2_421500 (g2) | 0.4168 | 0.4268 | 22 / 2 / 1 | ep04_zxg2_450000: 0.4275 (3/3) | **ep04_zxg2_450000** |
| 05 | ep05_zx_k3fix_88000 (k3fix) | 0.5782 | 0.5882 | 8 / 8 / 0 | ep05_zxk3fix_120000: 0.5856 (3/3) | **-** |
| 06 | ep06_k3fix_final28420 (k3fix) | 0.7973 | 0.8073 | 11 / 11 / 0 | ep06_z5q_120000: 0.7958 (3/3) | **-** |
| 07 | ep07_k3fix_final18529 (k3fix) | 0.9205 | 0.9305 | 7 / 7 / 0 | ep07_k3fix_15500: 0.9202 (3/3) | **-** |
| 08 | ep08_k3easy_final17496 (k3) | 0.5964 | 0.6064 | 11 / 11 / 0 | ep08_z5q_90000: 0.6018 (3/3) | **-** |
| 09 | ep09_raw_final39328 (raw) | 0.7605 | 0.7705 | 16 / 12 / 0 | ep09_raw_33000: 0.7302 (3/3) | **-** |
| 10 | ep10_raw_final34267 (raw) | 0.4125 | 0.4225 | 7 / 6 / 0 | ep10_k3fix_17000: 0.4066 (3/3) | **-** |
| 11 | ep11_raw_24000 (raw) | 0.8073 | 0.8173 | 10 / 10 / 0 | ep11_raw_28925: 0.7893 (3/3) | **-** |
| 12 | ep12_k3fix_final23178 (k3fix) | 0.5439 | 0.5539 | 7 / 7 / 0 | ep12_raw_37575: 0.5560 (3/3) | **-** |
| 13 | ep13_k3fix_final39254 (k3fix) | 0.6655 | 0.6755 | 7 / 5 / 0 | ep13_raw_44922: 0.6799 (3/3) | **-** |
| 14 | ep14_k3fix_final44922 (k3fix) | 0.3306 | 0.3406 | 17 / 0 / 0 | ep14_k3fix_37500: 0.3522 (0/3) | **-** |
| 15 | ep15_raw_final44922 (raw) | 0.7299 | 0.7399 | 13 / 10 / 0 | ep15_z5hfh_90000: 0.7329 (3/3) | **-** |
| 16 | STRICT: none |  |  | 12 / 11 / 2 | ep16_z5a5_120000: 0.9534 (3/3) | **ep16_z5a5_120000, ep16_z5a5_180000** (S-eligible) |
| 17 | ep17_raw_final31807 (raw) | 0.6475 | 0.6575 | 10 / 8 / 1 | ep17_z5r2_61500: 0.6591 (3/3) | **ep17_z5r2_61500** |
| 18 | ep18_k3fix_final18911 (k3fix) | 0.4191 | 0.4291 | 7 / 4 / 0 | ep18_k3fix_12500: 0.4185 (3/3) | **-** |
| 19 | STRICT: none |  |  | 24 / 15 / 2 | ep19_z5g2hardc2_360000: 0.5814 (3/3) | **ep19_z5g2hardc2_360000, ep19_z5g2hardc2_300000** (S-eligible) |
| 20 | STRICT: none |  |  | 16 / 16 / 2 | ep20_z5fill_90000: 0.8736 (3/3) | **ep20_z5fill_90000, ep20_z5fill_120000** (S-eligible) |
| 21 | ep21_k3fix_final44922 (k3fix) | 0.4445 | 0.4545 | 22 / 17 / 7 | ep21_z5fillc_282500: 0.4953 (3/3) | **ep21_z5fillc_282500** |
| 22 | STRICT: none |  |  | 10 / 10 / 2 | ep22_z5k3fix_89500: 0.7574 (3/3) | **ep22_z5k3fix_89500, ep22_z5k3fix_90000** (S-eligible) |
| 23 | ep23_raw_final20023 (raw) | 0.9029 | 0.9129 | 9 / 9 / 0 | ep23_raw_16500: 0.8994 (3/3) | **-** |
| 24 | ep24_raw_final44922 (raw) | 0.8951 | 0.9051 | 15 / 15 / 2 | ep24_z5q_150000: 0.9089 (3/3) | **ep24_z5q_150000** |
| 25 | STRICT: none |  |  | 15 / 13 / 2 | ep25_z5a5c_280500: 0.8766 (3/3) | **ep25_z5a5c_280500, ep25_z5a5c_190000** (S-eligible) |
| 26 | ep26_zx_k3fix_92000 (k3fix) | 0.7315 | 0.7415 | 16 / 16 / 0 | ep26_zxk3fix_150000: 0.7317 (3/3) | **-** |
| 27 | ep27_k3fix_final20863 (k3fix) | 0.8979 | 0.9079 | 14 / 14 / 0 | ep27_z5r2_51000: 0.9007 (3/3) | **-** |
| 28 | STRICT: none |  |  | 19 / 8 / 2 | ep28_z5q_180000: 0.1532 (3/3) | **ep28_z5q_180000, ep28_z5qc_230000** (S-eligible) |
| 29 | ep29_k3fix_final17358 (k3fix) | 0.5057 | 0.5157 | 11 / 7 / 0 | ep29_raw_44922: 0.5157 (3/3) | **-** |
