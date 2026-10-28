# T1improve confirmation: 10 official evaluations per candidate

Generated 2026-10-28 15:39:12 by `t1confirm_driver.py summary`. Rules: `CONFIRM_RULES.md` (sha256 `655578198ed51bbba20f6eded5bc810fd8e20165c8a0dfc1e516c5e29483fdd4`). 302 ok and 0 failed evaluations so far. Evaluator-JSON metrics; bars from the t1best20 per_episode.csv JSON columns (PREREG v2 section 2). Nothing here is packed, selected for upload or uploaded.

## Seeds

The official evaluator has no selectable seed (harness forces env.reset.seed = 42; the other seeds come from the evaluated file's own config). Distinct signatures over all evaluations:

- `delay=0(+g0..4,0-2) reset=42 scale_seed=0(1.0/1.0) wrench=0 vertex=0/0 ppo=42`
- `delay=0(+g0..4,0-2) reset=42 scale_seed=0(1.0/1.0) wrench=0 vertex=0/0 ppo=43`

All evaluations of one candidate share its file's signature, so the spread between them is GPU floating-point non-determinism, not a seed. `ppo=43` marks the seed-43 training arms (inert at evaluation).

## Items

| item | final | winner | winner AUC | winner MPPE | qualifiers | changed |
|---|---|---|---:|---:|---|---|
| ep01 | True | ep01-Bc_162000 | 0.7991 | 1.960 | ep01-Bc_162000 | True |
| ep05 | True | ep05-B3_62000 | 0.9645 | 0.741 | ep05-B3_60000, ep05-B3_62000, ep05-B3_62500 | True |
| ep10 | True | ep10-BT_58000 | 0.8930 | 1.162 | ep10-Bc_59500, ep10-BT_54000, ep10-BT_58000, ep10-BT_60000 | True |
| ep14 | True | ep14-R_92000 | 0.8347 | 1.460 | ep14-R_92000 | True |
| ep15 | True | ep15-B2_40000 | 0.9855 | 0.642 | ep15-B2_40000, ep15-B2_42000, ep15-B2_46000, ep15-P30_40000, ep15-P30_41000 | True |
| ep19 | True | ep19-Bc_135000 | 0.9217 | 0.994 | ep19-Bc_130000, ep19-Bc_135000 | True |
| ep21 | True | - |  |  | - | False |
| ep25 | True | - |  |  | - | False |
| ep28alt | True | ep28-W3H_83000 | 0.8164 | 2.684 | ep28-W3H_83000 | True |

## Candidates

| candidate | n ok | AUC mean | sd | min | needed | SP | MP | MPPE | cap | verdict | why |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| ep14-R_92000 | 10 | 0.8347 | 0.0022 | 0.8298 | 0.8315 | 10 | 10 | 1.460 | 1.653 | QUALIFIES | (a)-(e)+(f) pass |
| ep14-R_95500 | 10 | 0.8313 | 0.0017 | 0.8289 | 0.8315 | 10 | 10 | 1.494 | 1.653 | FAIL | fails (a) |
| ep14-R_96000 | 10 | 0.8283 | 0.0029 | 0.8233 | 0.8315 | 10 | 10 | 1.505 | 1.653 | FAIL | fails (a) |
| ep28-W3H_78000 | 10 | 0.8143 | 0.0044 | 0.8078 | 0.8123 | 10 | 6 | 2.847 | 3.616 | FAIL | fails (f) |
| ep28-W3H_83000 | 10 | 0.8164 | 0.0054 | 0.8060 | 0.8123 | 10 | 7 | 2.684 | 3.616 | QUALIFIES_EP28_MP_EXCEPTION | (a),(b),(d),(e),(f) pass; MP 7/10: kept only with MP >= 16/20 in its 20-trial confirmation |
| ep21-B5_136000 | 10 | 0.7632 | 0.0007 | 0.7622 | 0.7598 | 10 | 10 | 1.884 | 1.647 | FAIL | fails (d) |
| ep21-B5_138000 | 10 | 0.7634 | 0.0011 | 0.7621 | 0.7598 | 10 | 10 | 1.867 | 1.647 | FAIL | fails (d) |
| ep21-B5_139500 | 10 | 0.7615 | 0.0010 | 0.7599 | 0.7598 | 10 | 10 | 1.886 | 1.647 | FAIL | fails (d) |
| ep10-Bc_59500 | 10 | 0.8865 | 0.0012 | 0.8854 | 0.8725 | 10 | 10 | 1.232 | 1.462 | QUALIFIES | (a)-(e) pass |
| ep10-Bc_60000 | 10 | 0.8605 | 0.0726 | 0.6540 | 0.8725 | 9 | 9 | 2.954 | 1.462 | FAIL | fails (a),(b),(c),(d),(e) |
| ep10-BT_54000 | 10 | 0.8909 | 0.0016 | 0.8870 | 0.8725 | 10 | 10 | 1.180 | 1.612 | QUALIFIES | (a)-(e) pass |
| ep10-BT_58000 | 10 | 0.8930 | 0.0017 | 0.8901 | 0.8725 | 10 | 10 | 1.162 | 1.612 | QUALIFIES | (a)-(e) pass |
| ep10-BT_60000 | 10 | 0.8921 | 0.0009 | 0.8905 | 0.8725 | 10 | 10 | 1.171 | 1.612 | QUALIFIES | (a)-(e) pass |
| ep19-Bc_130000 | 10 | 0.9207 | 0.0032 | 0.9146 | 0.9001 | 10 | 10 | 1.031 | 1.323 | QUALIFIES | (a)-(e) pass |
| ep19-Bc_135000 | 10 | 0.9217 | 0.0034 | 0.9149 | 0.9001 | 10 | 10 | 0.994 | 1.323 | QUALIFIES | (a)-(e) pass |
| ep01-Bc_161500 | 10 | 0.7971 | 0.0037 | 0.7923 | 0.7653 | 10 | 10 | 1.957 | 3.973 | FAIL | fails (f) |
| ep01-Bc_162000 | 10 | 0.7991 | 0.0037 | 0.7934 | 0.7653 | 10 | 10 | 1.960 | 3.973 | QUALIFIES | (a)-(e)+(f) pass |
| ep15-B2_40000 | 10 | 0.9855 | 0.0006 | 0.9844 | 0.9420 | 10 | 10 | 0.642 | 1.176 | QUALIFIES | (a)-(e) pass |
| ep15-B2_42000 | 10 | 0.9841 | 0.0015 | 0.9815 | 0.9420 | 10 | 10 | 0.655 | 1.176 | QUALIFIES | (a)-(e) pass |
| ep15-B2_46000 | 10 | 0.9836 | 0.0012 | 0.9814 | 0.9420 | 10 | 10 | 0.635 | 1.176 | QUALIFIES | (a)-(e) pass |
| ep05-B3_60000 | 10 | 0.9633 | 0.0017 | 0.9611 | 0.9530 | 10 | 10 | 0.720 | 0.936 | QUALIFIES | (a)-(e) pass |
| ep05-B3_62000 | 10 | 0.9645 | 0.0022 | 0.9598 | 0.9530 | 10 | 10 | 0.741 | 0.936 | QUALIFIES | (a)-(e) pass |
| ep05-B3_62500 | 10 | 0.9627 | 0.0021 | 0.9601 | 0.9530 | 10 | 10 | 0.753 | 0.936 | QUALIFIES | (a)-(e) pass |
| ep25-B25_116000 | 10 | 0.7546 | 0.0090 | 0.7378 | 0.7540 | 10 | 10 | 2.238 | 2.163 | FAIL | fails (d) |
| ep25-B25_122000 | 10 | 0.7561 | 0.0074 | 0.7445 | 0.7540 | 10 | 10 | 2.257 | 2.163 | FAIL | fails (d) |
| ep21-B5s2_139000 | 10 | 0.7611 | 0.0022 | 0.7565 | 0.7598 | 10 | 10 | 1.933 | 1.647 | FAIL | fails (d) |
| ep21-B5s2_139500 | 10 | 0.7624 | 0.0010 | 0.7610 | 0.7598 | 10 | 10 | 1.918 | 1.647 | FAIL | fails (d) |
| ep15-P30_40000 | 10 | 0.9584 | 0.0007 | 0.9576 | 0.9420 | 10 | 10 | 0.498 | 1.176 | QUALIFIES | (a)-(e) pass |
| ep15-P30_41000 | 10 | 0.9599 | 0.0010 | 0.9585 | 0.9420 | 10 | 10 | 0.482 | 1.176 | QUALIFIES | (a)-(e) pass |

## Collapse screens (world_count=64; kit per-world AUC)

| candidate | status | collapses | min world AUC | mean | floor | world 0 | trials kit AUC range | world-0 ok | MP-fail worlds |
|---|---|---:|---:|---:|---:|---:|---|---|---:|
| ep14-R_92000 | ok | 0 | 0.8223 | 0.8311 | 0.7265 | 0.8303 | 0.8294-0.8369 | True | 0 |
| ep14-R_95500 | ok | 0 | 0.8154 | 0.8295 | 0.7265 | 0.8322 | 0.8287-0.8338 | True | 0 |
| ep14-R_96000 | ok | 0 | 0.8186 | 0.8273 | 0.7265 | 0.8289 | 0.8240-0.8313 | True | 0 |
| ep28-W3H_78000 | ok | 2 | 0.2780 | 0.8033 | 0.7073 | 0.8145 | 0.8071-0.8221 | True | 10 |
| ep28-W3H_83000 | ok | 0 | 0.8025 | 0.8141 | 0.7073 | 0.8104 | 0.8054-0.8226 | True | 7 |
| ep21-B5_136000 | ok | 0 | 0.7561 | 0.7620 | 0.6548 | 0.7638 | 0.7622-0.7642 | True | 0 |
| ep21-B5_138000 | ok | 0 | 0.7579 | 0.7626 | 0.6548 | 0.7648 | 0.7621-0.7652 | True | 0 |
| ep21-B5_139500 | ok | 0 | 0.7574 | 0.7610 | 0.6548 | 0.7605 | 0.7599-0.7632 | True | 0 |
| ep01-Bc_161500 | ok | 1 | 0.1458 | 0.7822 | 0.6603 | 0.7887 | 0.7907-0.8013 | True | 1 |
| ep01-Bc_162000 | ok | 0 | 0.7823 | 0.7945 | 0.6603 | 0.7984 | 0.7912-0.8028 | True | 0 |
| ep21-B5s2_139000 | ok | 0 | 0.7576 | 0.7612 | 0.6548 | 0.7610 | 0.7565-0.7642 | True | 0 |
| ep21-B5s2_139500 | ok | 0 | 0.7581 | 0.7623 | 0.6548 | 0.7625 | 0.7610-0.7638 | True | 0 |

## P30 pilot gate (ADD1 section 6, report only)

- ep15-P30_40000: MPPE 43.2% below the incumbent's 20-trial JSON mean, AUC +0.0214 vs the bar; gate PASS (changes no decision)
- ep15-P30_41000: MPPE 45.0% below the incumbent's 20-trial JSON mean, AUC +0.0230 vs the bar; gate PASS (changes no decision)

Changed items: ep01, ep05, ep10, ep14, ep15, ep19, ep28alt; all items final: True.
