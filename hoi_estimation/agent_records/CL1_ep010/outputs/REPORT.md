# White desk (episode 10): object trajectory recovery

Output: `/work/outputs/object_poses_world.npz`, key `T_world_mesh`, shape (794,4,4), float64. It is produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
(log: `solve_log.txt`; wall time 4 min 49 s, peak GPU memory 3.5 GB).

## What happens in the video (phases)
| phase | frames | content | how the boundary was set |
|---|---|---|---|
| static A | 0–264 | The desk stands upright on its two T-feet. The person walks up and grasps the near edge at about frame 175, then holds it. | Checked by eye. Template matching of desk patches the person does not cover (frame 160 vs. 180–260) shows ≤1 px image motion. The per-frame silhouette fits in 175–265 drift by 2–3 cm, but the patches show this is a bias from the person covering the near half of the desk, not real motion. |
| moving | 265–599 | The desk is turned about the vertical, tilted forward, rotated by about 90° and set down on its front edge. | Per-frame fits leave the static-A pose from about 265 (5 cm at 280, 9 cm at 290). |
| static B | 600–793 | The desk lies on its front edge (mesh +y side down), legs pointing sideways. The person lets go at about 640 and walks in front of it. | Per-frame fits stay within 1 cm of the pooled pose from 600 on. Object-mask IoU with the last frame is ≥0.95 from 600 on. |

The boundaries are explicit constants in `solve.py` (`A_END`, `A_FIT`, `B_START`), with these reasons in the comments.

## Method (what changed relative to the baseline, and why)
1. **Evidence: silhouette render-and-compare.** The textured mesh is rendered with nvdiffrast using the
   `camera.json` K at half resolution. It is compared with the SAM2 object mask, ignoring pixels the person covers. The loss is an
   L2 mask loss plus a 4×-pooled copy of it, which widens the basin of attraction. Two checks justify trusting this cue:
   * K and `T_world_cam` match the TRUE frame-0 human. Its projected bounding box agrees with the person mask
     to within 2–6 px, and the remaining difference is hair and loose clothing.
   * The SAM2 masks line up with image edges. The image-gradient profile across the mask boundary peaks right at
     the boundary for both the desk and the person masks, so the masks are not systematically inflated.

   The baseline is a frame-by-frame estimate. Its mean silhouette IoU is 0.79, and its pose jitters by
   ~1 cm and ~0.5° per frame even while the desk is not moving.
2. **Static phases: one pose per phase.**
   * The pose is fitted to the object mask pooled over all frames of the phase. Because the camera is static, one
     render is scored against per-pixel counts of visible-object and visible-background pixels.
   * **Hypotheses:** the baseline pose and its 180° flips about each mesh axis, each refined.
   * **Static B:** the silhouette decides. IoU is 0.944 for the chosen pose vs. 0.922 for the flip about the
     desk's long axis, and the texture correlation is 0.47 vs. 0.10.
   * **Static A:** the silhouette alone is ambiguous. The desk is front/back symmetric when seen from this view,
     so the 180° flip about its vertical axis scores the same IoU (margin 0.002, and the sign changes with the fit window). The
     solver then picks the hypothesis continuous with the tracked moving phase. That tracking connects without any
     jump to the unambiguous static-B pose: the chosen A is 2.5° from the tracked boundary pose, the flip is 178° away.
     This agrees with the baseline, FoundationPose and CoCoNet.
3. **Depth (distance along the viewing ray).** The silhouette refit puts the desk closer to the camera than the
   baseline: −0.5% in A, −2 to −4% in the moving phase (up to 13 cm), and −3% (11 cm) in B. Other cues:
   * MoGe-2 depth vs. rendered depth over the visible object pixels has a ratio of ≈1.00 ±0.02 for the refit
     poses, against 0.90–0.96 for the baseline in frames 350–550.
   * Floor contact: the refit static-B pose rests level on the plane, with all four support corners at
     4.0–5.3 cm. The baseline is tilted and penetrates by up to 3 cm.

   A final 1-D fusion along the camera ray is done in the static phases. It uses inverse variances:
   silhouette σ = 1%; floor σ = 2 cm contact tolerance, aimed at the frame-0 true-human sole height h0 = 1.4 cm
   above the plane; MoGe σ = 2.5%.
   * **A:** floor and MoGe would move the desk 1.5% / 1.2% farther. The fused correction is +0.7 cm.
   * **B:** floor (+2.1%) and MoGe (−2.7%) disagree. The fused correction is +0.3 cm, i.e. the silhouette depth is kept.
4. **Moving phase.** Each frame is first fitted to its silhouette, starting from the baseline (150 Adam steps).
   Then all 335 frames are optimised jointly: silhouette loss, plus an acceleration penalty on the 8 mesh
   bounding-box corners (λ = 1e4, a constant-velocity prior in metric units), with the two static poses as fixed
   boundary conditions at both ends.
   * **Tried and rejected:** a KLT feature-track reprojection term.
     * It was useless as a fix for the occlusion bias: on the known-static frames 175–265 it did not stop
       the 3 cm drift.
     * When fitted on alternate frame pairs (to keep the other half held out), it made the
       trajectory zig-zag: acceleration went from 0.31 to 1.82 mm/frame² and held-out KLT from 0.14 to 0.34 px.
     * It is left in the code with weight 0, and KLT is used only for validation.

## Validation (no ground truth; `python validate.py object_poses_world.npz`)
Held-out KLT = pairwise KLT tracks on visible desk texture starting at odd frames (none are used by `solve.py`).
Patch = 41×41 template matches of desk patches the person does not cover, frame i → i+5, with sub-pixel refinement.
Both measure how well the trajectory explains the image motion of the object's texture (reprojection error in px).

| phase | frames | traj | mask IoU | held-out KLT px (median / mean≤5) | patch i→i+5 px (median / mean≤10) | accel mm/frame² | median lowest-point height above plane (m) |
|---|---|---|---|---|---|---|---|
| static A | 0-264 | baseline | 0.760 | 0.66 / 0.74 (n=1658) | 0.92 / 1.07 (n=536) | 8.55 | 0.023 |
| static A | 0-264 | result | 0.821 | 0.03 / 0.07 (n=1658) | 0.02 / 0.09 (n=536) | 0.02 | 0.022 |
| moving | 265-599 | baseline | 0.774 | 0.83 / 0.94 (n=4472) | 1.73 / 2.21 (n=710) | 18.24 | 0.002 |
| moving | 265-599 | result | 0.877 | 0.14 / 0.22 (n=4472) | 0.80 / 1.54 (n=710) | 0.31 | 0.031 |
| static B | 600-793 | baseline | 0.859 | 0.73 / 0.80 (n=3899) | 0.98 / 1.23 (n=422) | 14.84 | -0.004 |
| static B | 600-793 | result | 0.941 | 0.01 / 0.02 (n=3899) | 0.02 / 0.02 (n=422) | 0.00 | 0.035 |

All-frame mask IoU is 0.790 for the baseline and 0.874 for the result.

How to read the table:
* In the static phases the KLT and patch columns mostly show that the desk does not move and the baseline jitters.
  Any constant pose would score near zero there.
* The moving phase is the informative one. The image motion is explained about 6× better by median held-out KLT
  residual and about 2× better by median patch residual, at a much higher IoU.
* The reference human's shoes are 1.4 cm above the plane and its feet average 3.7 cm, so the result's resting
  heights of 2.2 cm (A) and 3.5 cm (B) are within the plane's tolerance.

Change vs. baseline (mean ADD between the two trajectories over mesh points): static A 2.4 cm, moving 10.4 cm
(max 22 cm), static B 11.0 cm. In the moving and B phases this is mostly the depth correction. Rotation differences
are 1–4°.

## Confidence per phase
* **Static A: high.** The pose is supported by many frames, the floor contact is consistent, and the support plane
  is parallel to the released floor within 0.5°. The orientation comes from trajectory continuity (the silhouette
  is symmetric), but every source agrees on it. Residual risk: a sub-cm push while the person holds the desk
  (175–265 is modelled as static), and about 1–2% depth uncertainty.
* **Static B: high for rotation and position in the image plane; medium-high for depth.** Silhouette depth is sharp
  here (desk ~500 px wide, pooled IoU 0.94). The floor would like it ~2% farther and MoGe ~2.7% closer.
* **Moving: medium.**
  * Large parts are covered by the bending person (IoU 0.80–0.83 in 280–440).
  * On the known-static frames 175–265, per-frame silhouette fits showed an occlusion bias of 2–3 cm, so errors of
    a few cm are likely in 280–440.
  * 440–600 is well constrained (IoU 0.91–0.95).
  * Its depth is silhouette-only. MoGe supports it; the baseline's human-calibrated depth disagrees by 2–4%.

## Notes / limitations
* nvdiffrast's antialiasing gradients are not bit-deterministic on the GPU. Two runs of `solve.py` differed by
  ~0.002 in IoU margins, i.e. sub-millimetre pose differences. Seeds are fixed and the decisions are
  threshold-based, so they are stable.
* Hand–object attachment, using the reconstructed hands, was not used: the human reconstruction's hands are noisy
  at this scale and the silhouette is strong for this large object.
* GPU use over all experiments: roughly 35–40 min in total (estimated from the run times), at under 4 GB of
  memory. One `solve.py` run uses about 3.5 min of GPU time.
