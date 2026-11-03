# Rolling-luggage trajectory recovery

## Method

The episode was split into four static plateaus separated by three left-foot pushes. For each plateau, the baseline poses were pooled to a single rigid pose and refined with a differentiable antialiased nvdiffrast silhouette fit over all sampled frames of the plateau. Multiple deterministic frame subsets and free/upright initialisations were scored on the full plateau; the complete textured mesh, including its asymmetric handle, was used as a held-out silhouette check. The handle faces were excluded only from the fitting silhouette because SAM2 omits the thin telescoping handle.

During pushes the method low-pass filters the shipped baseline motion and right-multiplies a local SE(3) correction that joins the two adjacent pooled poses exactly. Finally the exact full mesh is placed on the released floor plane at every frame (wheel-contact constraint).

## Phases and selected static hypotheses

| plateau | frames | selected candidate | retained body / full-mesh IoU | confidence |
|---|---:|---|---|---|
| 0 | 0-103 | `free_strided` | 0.9053 / 0.8408 | medium |
| 1 | 150-219 | `upright_neg` | 0.8954 / 0.8286 | medium-high |
| 2 | 251-341 | `upright_neg` | 0.9072 / 0.8372 | high |
| 3 | 380-466 | `free_coarse` | 0.9222 / 0.8535 | high |

Push intervals are 104-149, 220-250 and 342-379. Static plateaus carry more confidence because hundreds of mask pixels constrain one pose; push confidence is medium because these intervals retain the baseline motion shape between two strongly constrained endpoints.

## Validation (no ground truth)

Mask IoU compares the body-only rendered mesh with the released object mask (the handle is not in the SAM2 mask). Step columns are median per-frame translation changes; floor is the absolute wheel/ground-plane residual.

| interval | frames | baseline mask IoU | result mask IoU | baseline step mm | result step mm | max floor mm |
|---|---:|---:|---:|---:|---:|---:|
| static_0 | 0-103 | 0.8586 | 0.9055 | 8.2 | 0.0 | 0.00 |
| static_1 | 150-219 | 0.8739 | 0.8960 | 6.1 | 0.0 | 0.00 |
| static_2 | 251-341 | 0.8907 | 0.9083 | 9.3 | 0.0 | 0.00 |
| static_3 | 380-466 | 0.8839 | 0.9236 | 7.5 | 0.0 | 0.00 |
| push_0 | 104-149 | 0.8527 | 0.8568 | 7.7 | 4.8 | 0.00 |
| push_1 | 220-250 | 0.8973 | 0.9109 | 9.4 | 10.3 | 0.00 |
| push_2 | 342-379 | 0.8980 | 0.9174 | 13.6 | 9.6 | 0.00 |

- Overall body-mask IoU: **0.8771 -> 0.9045**.
- 99th-percentile translation step: 44.2 -> 14.2 mm.
- 99th-percentile rotation step: 2.60 -> 1.81 degrees.
- Maximum wheel-to-floor residual after the final contact constraint: 0.00 mm.
- End-to-end `solve.py` GPU/CPU wall time: 3.27 minutes.

The static fits clearly improve pooled mask agreement and remove frame-to-frame pose jitter. Every interval's mask IoU improves in the final trajectory. The middle push has a slightly larger median step because baseline depth drift was compressed while its two endpoints were restored, but the 99th-percentile jump is much smaller. No baseline motion was replaced with an embedding or hand-fitted per-frame table.
