| trajectory | phase | flow err px (lower better) | mask IoU | tilt deg (median) | 95% |dyaw|/frame deg | 95% accel cm/f^2 | dz vs baseline mean / median cm | base height above floor m |
|---|---|---|---|---|---|---|---|---|
| baseline | all [0-545] | 1.80 | 0.789 | 4.3 | 4.92 | 9.69 | 0.0 / 0.0 | 0.852 |
| result | all [0-545] | 0.70 | 0.794 | 0.0 | 1.78 | 0.11 | 0.1 / -0.4 | 0.849 |
| baseline | moving (back) [380-468] | 1.70 | 0.702 | 3.3 | 1.46 | 6.20 | 0.0 / 0.0 | 0.851 |
| result | moving (back) [380-468] | 1.19 | 0.710 | 0.0 | 0.30 | 0.12 | -0.9 / -1.0 | 0.843 |
| baseline | moving (baseline failure) [348-379] | 10.00 | 0.378 | 54.8 | 106.36 | 45.64 | 0.0 / 0.0 | 1.042 |
| result | moving (baseline failure) [348-379] | 3.40 | 0.657 | 0.0 | 1.39 | 0.16 | -5.9 / -5.9 | 0.847 |
| baseline | moving (front) [164-347] | 1.53 | 0.857 | 3.9 | 2.67 | 4.29 | 0.0 / 0.0 | 0.857 |
| result | moving (front) [164-347] | 1.08 | 0.858 | 0.0 | 2.31 | 0.11 | 0.5 / -0.2 | 0.854 |
| baseline | static-1 [0-163] | 1.04 | 0.861 | 4.1 | 0.68 | 3.41 | 0.0 / 0.0 | 0.849 |
| result | static-1 [0-163] | 0.07 | 0.841 | 0.0 | 0.00 | 0.00 | 0.8 / 0.0 | 0.849 |
| baseline | static-2 [469-545] | 1.62 | 0.744 | 5.6 | 1.31 | 6.61 | 0.0 / 0.0 | 0.836 |
| result | static-2 [469-545] | 0.08 | 0.694 | 0.0 | 0.00 | 0.00 | 1.6 / 0.0 | 0.841 |
