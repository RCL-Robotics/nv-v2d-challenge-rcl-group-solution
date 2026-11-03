| phase | frames | traj | mask IoU | held-out KLT px (median / mean<=5) | patch i->i+5 px (median / mean<=10) | accel mm/frame^2 | median lowest-point height above plane (m) |
|---|---|---|---|---|---|---|---|
| static A | 0-264 | baseline | 0.760 | 0.66 / 0.74 (n=1658) | 0.92 / 1.07 (n=536) | 8.55 | 0.023 |
| static A | 0-264 | result | 0.821 | 0.03 / 0.07 (n=1658) | 0.02 / 0.09 (n=536) | 0.02 | 0.022 |
| moving | 265-599 | baseline | 0.774 | 0.83 / 0.94 (n=4472) | 1.73 / 2.21 (n=710) | 18.24 | 0.002 |
| moving | 265-599 | result | 0.877 | 0.14 / 0.22 (n=4472) | 0.80 / 1.54 (n=710) | 0.31 | 0.031 |
| static B | 600-793 | baseline | 0.859 | 0.73 / 0.80 (n=3899) | 0.98 / 1.23 (n=422) | 14.84 | -0.004 |
| static B | 600-793 | result | 0.941 | 0.01 / 0.02 (n=3899) | 0.02 / 0.02 (n=422) | 0.00 | 0.035 |
baseline all-frames IoU 0.78994733
result all-frames IoU 0.8739664
