| phase | metric | baseline | result |
|---|---|---|---|
| 0-201 static | mask IoU (fat-mask model, 2.5px) | 0.749 | 0.846 |
| 0-201 static | mask IoU (plain render) | 0.642 | 0.717 |
| 0-201 static | edge chamfer px (lower=better) | 1.572 | 1.082 |
| 0-201 static | rot jitter deg (median dev. from smoothed) | 0.25 | 0.00 |
| 0-201 static | centre jitter cm | 0.49 | 0.00 |
| 0-201 static | max frame-to-frame rot deg | 0.84 | 0.00 |
| 0-201 static | lowest point above floor plane, median cm | 0.77 | -0.56 |
| 0-201 static | lowest point above floor plane, std cm | 1.11 | 0.00 |
| 0-201 static | cam depth z result-baseline, mean / median cm | 0 | 0.12 / -0.00 |
| 0-201 static | change vs baseline: rot deg median / p90; transl cm median / p90 | - | 0.89 / 1.40; 2.16 / 3.93 |
| 202-559 moving | mask IoU (fat-mask model, 2.5px) | 0.653 | 0.759 |
| 202-559 moving | mask IoU (plain render) | 0.542 | 0.627 |
| 202-559 moving | edge chamfer px (lower=better) | 1.764 | 1.521 |
| 202-559 moving | rot jitter deg (median dev. from smoothed) | 0.78 | 0.55 |
| 202-559 moving | centre jitter cm | 0.56 | 0.21 |
| 202-559 moving | max frame-to-frame rot deg | 7.27 | 5.95 |
| 202-559 moving | lowest point above floor plane, median cm | -1.89 | -2.41 |
| 202-559 moving | lowest point above floor plane, std cm | 1.00 | 0.93 |
| 202-559 moving | cam depth z result-baseline, mean / median cm | 0 | 0.58 / 0.58 |
| 202-559 moving | change vs baseline: rot deg median / p90; transl cm median / p90 | - | 2.78 / 4.82; 1.46 / 2.50 |
| 560-718 static | mask IoU (fat-mask model, 2.5px) | 0.766 | 0.827 |
| 560-718 static | mask IoU (plain render) | 0.631 | 0.640 |
| 560-718 static | edge chamfer px (lower=better) | 1.380 | 1.088 |
| 560-718 static | rot jitter deg (median dev. from smoothed) | 0.39 | 0.00 |
| 560-718 static | centre jitter cm | 0.71 | 0.00 |
| 560-718 static | max frame-to-frame rot deg | 3.00 | 0.00 |
| 560-718 static | lowest point above floor plane, median cm | -0.13 | -0.51 |
| 560-718 static | lowest point above floor plane, std cm | 1.57 | 0.00 |
| 560-718 static | cam depth z result-baseline, mean / median cm | 0 | 1.83 / -0.00 |
| 560-718 static | change vs baseline: rot deg median / p90; transl cm median / p90 | - | 2.10 / 3.25; 5.14 / 10.42 |
| 0-718 ALL | mask IoU (fat-mask model, 2.5px) | 0.705 | 0.798 |
| 0-718 ALL | mask IoU (plain render) | 0.590 | 0.655 |
| 0-718 ALL | edge chamfer px (lower=better) | 1.628 | 1.299 |
| 0-718 ALL | rot jitter deg (median dev. from smoothed) | 0.47 | 0.11 |
| 0-718 ALL | centre jitter cm | 0.56 | 0.06 |
| 0-718 ALL | max frame-to-frame rot deg | 7.27 | 5.95 |
| 0-718 ALL | lowest point above floor plane, median cm | -1.06 | -0.56 |
| 0-718 ALL | lowest point above floor plane, std cm | 1.65 | 1.12 |
| 0-718 ALL | cam depth z result-baseline, mean / median cm | 0 | 0.73 / 0.49 |
| 0-718 ALL | change vs baseline: rot deg median / p90; transl cm median / p90 | - | 1.84 / 4.27; 1.87 / 6.07 |
