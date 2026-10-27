| phase | metric | baseline | result |
|---|---|---|---|
| 0-201 static | mask IoU (fat-mask model, 2.5px) | 0.749 | 0.846 |
| 0-201 static | mask IoU (plain render) | 0.642 | 0.717 |
| 0-201 static | edge chamfer px (lower=better) | 1.572 | 1.091 |
| 0-201 static | rot jitter deg (median dev. from smoothed) | 0.25 | 0.00 |
| 0-201 static | centre jitter cm | 0.49 | 0.00 |
| 0-201 static | max frame-to-frame rot deg | 0.84 | 0.00 |
| 0-201 static | lowest point above floor plane, median cm | 0.77 | -0.53 |
| 0-201 static | lowest point above floor plane, std cm | 1.11 | 0.00 |
| 0-201 static | cam depth z result-baseline, mean / median cm | 0 | 0.12 / 0.00 |
| 0-201 static | change vs baseline: rot deg median / p90; transl cm median / p90 | - | 0.92 / 1.44; 2.13 / 3.94 |
| 202-559 moving | mask IoU (fat-mask model, 2.5px) | 0.653 | 0.782 |
| 202-559 moving | mask IoU (plain render) | 0.542 | 0.651 |
| 202-559 moving | edge chamfer px (lower=better) | 1.764 | 1.465 |
| 202-559 moving | rot jitter deg (median dev. from smoothed) | 0.78 | 0.42 |
| 202-559 moving | centre jitter cm | 0.56 | 0.18 |
| 202-559 moving | max frame-to-frame rot deg | 7.27 | 5.96 |
| 202-559 moving | lowest point above floor plane, median cm | -1.89 | -1.09 |
| 202-559 moving | lowest point above floor plane, std cm | 1.00 | 0.84 |
| 202-559 moving | cam depth z result-baseline, mean / median cm | 0 | -2.64 / -2.35 |
| 202-559 moving | change vs baseline: rot deg median / p90; transl cm median / p90 | - | 2.84 / 4.67; 3.26 / 5.14 |
| 560-718 static | mask IoU (fat-mask model, 2.5px) | 0.766 | 0.827 |
| 560-718 static | mask IoU (plain render) | 0.631 | 0.640 |
| 560-718 static | edge chamfer px (lower=better) | 1.380 | 1.082 |
| 560-718 static | rot jitter deg (median dev. from smoothed) | 0.39 | 0.00 |
| 560-718 static | centre jitter cm | 0.71 | 0.00 |
| 560-718 static | max frame-to-frame rot deg | 3.00 | 0.00 |
| 560-718 static | lowest point above floor plane, median cm | -0.13 | -0.50 |
| 560-718 static | lowest point above floor plane, std cm | 1.57 | 0.00 |
| 560-718 static | cam depth z result-baseline, mean / median cm | 0 | 1.83 / 0.00 |
| 560-718 static | change vs baseline: rot deg median / p90; transl cm median / p90 | - | 2.10 / 3.20; 5.14 / 10.43 |
| 0-718 ALL | mask IoU (fat-mask model, 2.5px) | 0.705 | 0.810 |
| 0-718 ALL | mask IoU (plain render) | 0.590 | 0.667 |
| 0-718 ALL | edge chamfer px (lower=better) | 1.628 | 1.278 |
| 0-718 ALL | rot jitter deg (median dev. from smoothed) | 0.47 | 0.09 |
| 0-718 ALL | centre jitter cm | 0.56 | 0.04 |
| 0-718 ALL | max frame-to-frame rot deg | 7.27 | 5.96 |
| 0-718 ALL | lowest point above floor plane, median cm | -1.06 | -0.53 |
| 0-718 ALL | lowest point above floor plane, std cm | 1.65 | 0.65 |
| 0-718 ALL | cam depth z result-baseline, mean / median cm | 0 | -0.88 / -1.75 |
| 0-718 ALL | change vs baseline: rot deg median / p90; transl cm median / p90 | - | 1.84 / 4.28; 3.02 / 6.28 |
