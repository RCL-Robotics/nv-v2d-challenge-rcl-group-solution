# t2_consensus S3: 5x official own-input evaluations of the new jobs (2026-10-24 23:53:31)

Own input = each policy against its OWN training parquet (kit run_policy_evaluation.py + evaluate_policy.py 160d24f5 on stock 8c778f9, frame 0, explicit reset, world count 1). Not a leaderboard number.

| job | variants | input | ckpt sha256 | export | AUC runs 1-5 | mean | sd | min | max | SP-SR | MP-SR | MPPE cm | SP=MP=1 runs | closest | Zuxing single |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ep05_zx_k3fix_26000 | A,B | k3fix 2ccbe480 | 4f925b2a34b9 | ok 0.0001 | 0.6949 0.6582 0.6137 0.6925 0.6510 | 0.6620 | 0.0335 | 0.6137 | 0.6949 | 1.00 | 1.00 | 1.90 | 5 | 2 | 0.6816 |
| ep26_zx_k3fix_20000 | A,B | k3fix 01637248 | 249ed85df2fc | ok 0.0001 | 0.9002 0.8993 0.9013 0.9022 0.9046 | 0.9015 | 0.0020 | 0.8993 | 0.9046 | 1.00 | 1.00 | 1.08 | 5 | 3 | 0.9032 |
| ep04_zx_g2_18000 | B | g2 4b07b0d1 | 91e1a200eb75 | ok 0.0001 | 0.1442 0.1445 0.1445 0.1445 0.1443 | 0.1444 | 0.0002 | 0.1442 | 0.1445 | 0.00 | 0.00 | 111.66 | 0 | 2 | 0.1435 |
| ep19_zx_g2_20000 | B | g2 46039cd2 | 9f0f090efb6a | ok 0.0001 | 0.1325 0.1329 0.1326 0.1331 0.1342 | 0.1331 | 0.0007 | 0.1325 | 0.1342 | 0.00 | 0.00 | 58.97 | 0 | 4 | 0.1330 |
| ep28_zx_g2_22000 | B | g2 0d49273d | 8750a1427b4b | ok 0.0001 | 0.4338 0.0817 0.3943 0.4214 0.4254 | 0.3513 | 0.1514 | 0.0817 | 0.4338 | 0.00 | 0.00 | 30.00 | 0 | 3 | 0.4131 |
