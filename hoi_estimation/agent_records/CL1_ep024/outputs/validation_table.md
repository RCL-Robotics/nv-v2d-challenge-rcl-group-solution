phases A [0,133) moving [133,598) B [598,748); kappa 1.089
| metric | A static base | A static ours | moving base | moving ours | B static base | B static ours | all base | all ours |
|---|---|---|---|---|---|---|---|---|
| silhouette IoU, scale-free (mean) | 0.770 | 0.934 | 0.790 | 0.914 | 0.745 | 0.926 | 0.777 | 0.920 |
| silhouette IoU, scale-free (10th pct) | 0.734 | 0.928 | 0.705 | 0.897 | 0.680 | 0.916 | 0.706 | 0.901 |
| silhouette IoU, as-is metric depth (mean) | 0.760 | 0.785 | 0.701 | 0.811 | 0.728 | 0.798 | 0.717 | 0.804 |
| scale-free IoU on held-out frames (i%3!=0) | 0.770 | 0.934 | 0.789 | 0.914 | 0.745 | 0.926 | 0.777 | 0.920 |
| frame-to-frame rotation, median (deg) | 1.23 | 0.00 | 1.72 | 0.52 | 1.16 | 0.00 | 1.47 | 0.24 |
| frame-to-frame rotation, 99th pct (deg) | 3.56 | 0.00 | 6.82 | 3.00 | 4.71 | 0.00 | 5.96 | 2.85 |
| frame-to-frame translation, median (cm) | 1.12 | 0.00 | 1.28 | 0.15 | 1.02 | 0.00 | 1.20 | 0.07 |
| camera depth std over phase (cm) | 2.2 | 0.0 | 5.1 | 4.4 | 3.1 | 0.0 | 6.1 | 5.6 |
| base tilt vs floor normal, median (deg) | 1.9 | 0.5 | 71.3 | 73.1 | 1.8 | 0.5 | 20.1 | 21.2 |
| lowest mesh point above plane, median (cm) | -0.8 | -1.2 | 5.4 | 4.8 | -1.8 | -1.6 | 3.5 | 3.5 |
| lowest mesh point above plane, min (cm) | -2.7 | -1.2 | -2.9 | -1.5 | -4.1 | -1.6 | -4.1 | -1.6 |
| hand-keypoint centroid to mesh, median (cm) | 18.6 | 18.7 | 3.0 | 2.8 | 32.8 | 33.7 | 5.1 | 5.4 |

mean point distance result vs baseline (cm) per phase: {'A static': 2.35, 'moving': 2.38, 'B static': 3.4, 'all': 2.58}
hand-to-mesh (cm) on contact frames: base 3.3 ours 3.1
