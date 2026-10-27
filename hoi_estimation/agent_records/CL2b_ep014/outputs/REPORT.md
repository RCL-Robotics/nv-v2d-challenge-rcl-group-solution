# Round table, episode 14: trajectory re-estimation report

Output: `object_poses_world.npz` (`T_world_mesh`, 993×4×4, float64), produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
(about 170 s wall time on the shared RTX 6000 Ada, peak GPU memory well below 9 GB, no network).
Validation: `python /work/outputs/validate.py --result /work/outputs/object_poses_world.npz` (about 4 min).

## TL;DR
* **Main finding: the baseline has the wrong spin about the table's 5‑fold leg axis on every frame.** The leg
  star is 5‑fold symmetric, so a spin error of a multiple of 72° leaves the silhouette almost unchanged.
  Rigid, asymmetric evidence on the underside (mounting plate, label and the black height‑adjust lever) shows:
  * frames 0–600: the true pose is `T_base · Spin(−144°)`
  * frames 627–992: it is `T_base · Spin(−72°)`. The baseline tracker slipped by about −72° at frames 600–627,
    while the table was resting upside down and was not turning (checked by eye).

  So the baseline is about 25–45 cm (ADD) away from my estimate on every frame. If my analysis is right, the
  baseline scores close to zero on this episode; if it is wrong, my output is equally far off. Below I give the
  evidence and my confidence (about 75–80 %).
* After the spin fix, all frames are refined jointly with a silhouette loss and temporal smoothness. The three
  static phases each share one pose. Depth is then re‑anchored to the baseline. Mask IoU goes from 0.790 to 0.857,
  and from 0.790 to 0.857 on held‑out frames as well. The largest frame‑to‑frame rotation jump falls from 28.8° to
  2.3°. The median camera‑depth difference to the baseline is 0.0 cm.

## Phases found (automatic, then checked by eye)
| frames | phase | how detected |
|---|---|---|
| 0–95 | upright, untouched, **static** | median in‑mask grey‑level change over 5 frames < 0.65 (noise ≈ 0.3), cut at first hand contact (`contact_prob` > 0.5) |
| 96–315 | grabbed, tilted onto its side (top towards camera) | – |
| 316–390 | lying on its side, held but **static** | same image‑difference criterion |
| 391–599 | rolled over the edge, flipped upside down (wheels up), resting on its top | – |
| 600–627 | **baseline tracker slip** (spin −77° in 15 frames; video shows no rotation) | 20‑frame accumulated spin of the baseline about the table axis > 45° |
| 628–937 | rotated onto its side again, then back upright | – |
| 938–992 | upright, released, **static** | image difference + contact ended |

## Method (what `solve.py` does)
1. **Table axis.** The 5‑fold symmetry axis of the leg arms is fitted on the mesh by rotating the leg vertices
   by 72° and 144° and minimising nearest‑neighbour distance. This gives the point (0.0007, 0.0137) and
   direction (0.0073, −0.001, 1).
2. **Segments between tracker slips.** The baseline's frame‑to‑frame rotation is projected on the body‑frame
   table axis. A window is flagged as a slip when the spin changes by more than 45° within 20 frames. Only
   600–627 is flagged.
3. **Spin hypotheses `T_base · Spin(72k)`, k = 0…4, per segment, scored on pooled texture.**
   * For every 30‑frame window, each video frame is rectified onto the plane of the table underside
     (z = −0.092, mesh frame) and of the top (z = −0.115). Visibility comes from the rendered depth;
     person pixels are excluded.
   * The rectified frames are averaged over the window, high‑pass filtered, and correlated with the rectified
     mesh texture rotated over all angles.
   * Window scores at the five grid angles (±10°) are summed per segment.
   * Underside evidence is decisive in both segments (scores for 0/72/144/−144/−72):
     * segment 0–600: [0.07 0.06 0.13 **0.54** 0.06], so −144
     * segment 627–992: [0.32 0.17 0.28 0.45 **1.39**], so −72
   * The top (wood grain) evidence is also strong (+144 in segment 1), but it is **consistently 72° away from
     the underside**. This holds in both segments: the end‑phase top gives −148 against −72 from the underside.
     The script calibrates this top‑vs‑underside texture offset automatically (+72°). It uses the top only for
     segments with no underside view; in this episode there were none.
4. **Apply and bridge.** The chosen spin is applied about the fitted axis. The slip window 600–627 is replaced by
   SLERP/linear interpolation between its corrected ends.
5. **Joint refinement of all 993 poses** (nvdiffrast, half‑resolution SAM2 masks).
   * Loss: per‑pixel soft‑silhouette loss plus an 8×8 pooled term. Pixels covered by the person (and not by the
     object mask) are ignored.
   * Second‑order smoothness on R and on the table centre, weight 0.5. The weight was chosen with held‑out
     frames: 0.25 → 0.862, 1 → 0.857, 4 → 0.849 held‑out IoU.
   * One shared pose per static phase. 40 epochs, Adam, fixed seeds.
6. **Depth re‑anchoring (default rule; no deviation applied).** The camera‑frame position of the table centre
   (on the axis, 0.27 m up) is scaled along its viewing ray.
   * Static phases: one scale per phase, so the median z equals the baseline's.
   * Moving phases: a slowly varying scale, i.e. a running median over ±45 frames of `z_base / z_ours`, then
     Gaussian‑smoothed (σ = 10 frames).
   * Image position is unchanged. Before re‑anchoring, the silhouette fit had moved the table by −0.5 to +3.2 cm
     in depth, depending on the phase.

## Evidence for the spin decision (why the baseline is believed wrong)
* **Lever / label / plate.**
  * Rendering the mesh at frame 560 with `Spin(−144)` puts the mesh's lever exactly on the real black lever,
    which points to the image right. The label lands right of the plate, as in the video.
  * At frame 640, `Spin(−72)` matches. The baseline puts the lever pointing up‑left and the label above the plate.
  * Debug images are in `/work/scratch/frames/lvb.png`, `cu560.png`, `r1a.png`, `r1b.png`.
* **Pooled rectified underside texture** (rigid features, hundreds of frames). The correlation peak lands on
  exactly the predicted grid angle, with no leg prior: −144/−146 (corr 0.16–0.32) before the slip and −72/−76
  (0.11–0.42) after it.
* **Continuity (no slips elsewhere).** I looked at the hand keypoints of the reconstructed human in the object
  frame (azimuth about the table axis) during continuous grasps:
  * They stay constant through 115–330, through 445–530 (the edge‑on flip) and through 805–925 (the flip back
    upright).
  * They jump by about +70° only at 600–625, which is exactly the baseline slip.
  * So the start phase must have the same offset as the underside phase 505–596.
* **Lever visibility.**
  * Start (frames 20–80): only `Spin(−144)` keeps the mesh lever hidden. The other hypotheses render 75–290 lever
    pixels outside the object mask, where the video shows nothing.
  * End: only −144 and −72 hide it. My choice is −72.
  * In validation, lever pixels outside the mask fall from 39 to 12 on average.
* **Contradicting evidence (why the confidence is not higher).**
  * In the static start phase the leg + caster silhouettes fit the baseline spin best: IoU 0.907 vs 0.850 for
    −144, after per‑hypothesis refinement and ignoring casters and lever. In the static end phase, +72/+144 fit
    best.
  * Single‑frame silhouettes elsewhere are inconsistent (e.g. frame 760 prefers my choice, frame 700 prefers +144).
  * The mesh legs are not exactly 5‑fold symmetric (8 mm mean, about 2 cm p90 deviation) and the casters swivel.
    I therefore treat silhouettes as weak evidence for the 72° choice and strong evidence only for the fine spin.
  * Also, the mesh's top texture is 72° off from its underside texture, so the mesh is internally inconsistent
    (probably a merge of two partial scans). If the ground truth was fitted with an emphasis on a different part of
    the mesh, the "true" spin may differ.
* **Remaining fine‑spin ambiguity.** In 628–937 the leg silhouettes pin the spin sharply (IoU 0.898 at the
  chosen spin vs 0.816 at −8°). The underside texture prefers about −9° (corr 0.42). I kept the geometric
  (silhouette) answer, because the texture is demonstrably misregistered elsewhere. If the texture were right,
  points on the rim would be off by about 5 cm.

## Validation table (baseline vs result, per phase; `validate.py`)
| phase | frames | IoU base | IoU result | lever px outside mask base/result | rot step med (deg) base/result | trans step med (cm) base/result | lowest point above floor med (cm) base/result | depth diff result−base mean/median (cm) | underside tex peak (deg, corr) base → result | top tex peak base → result |
|---|---|---|---|---|---|---|---|---|---|---|
| 0-95 | 96 | 0.730 | 0.768 | 44 / 0 | 0.21 / 0.02 | 0.36 / 0.00 | -1.7 / -1.5 | -0.5 / -0.0 | - → - | -146 (0.03) → 16 (0.05) |
| 96-315 | 220 | 0.813 | 0.840 | 36 / 0 | 0.52 / 0.38 | 0.65 / 0.33 | -3.3 / -2.2 | 0.3 / -0.0 | - → - | -120 (0.03) → -76 (0.18) |
| 316-390 | 75 | 0.940 | 0.975 | 0 / 0 | 2.10 / 0.01 | 0.95 / 0.00 | -2.0 / -1.9 | 0.2 / 0.0 | - → - | 144 (0.15) → -72 (0.36) |
| 391-599 | 209 | 0.799 | 0.856 | 37 / 16 | 1.34 / 0.51 | 0.91 / 0.55 | -2.6 / -1.9 | 0.1 / 0.0 | -146 (0.16) → -8 (0.37) | 138 (0.22) → -74 (0.30) |
| 600-627 | 28 | 0.792 | 0.861 | 43 / 4 | 2.80 / 0.57 | 0.89 / 0.77 | -4.4 / -2.6 | -2.2 / -2.3 | -158 (0.07) → -6 (0.43) | - |
| 628-937 | 310 | 0.758 | 0.876 | 47 / 28 | 1.08 / 0.70 | 1.04 / 0.39 | -4.2 / -1.6 | 0.0 / 0.1 | -76 (0.13) → -10 (0.42) | -102 (0.07) → -54 (0.07) |
| 938-992 | 55 | 0.752 | 0.821 | 61 / 0 | 0.39 / 0.00 | 0.55 / 0.00 | -2.1 / -1.3 | 0.2 / 0.0 | - → - | -114 (0.06) → -66 (0.07) |
| all | 993 | 0.790 | 0.857 | 39 / 12 | 0.85 / 0.40 | 0.82 / 0.28 | -2.7 / -1.8 | 0.0 / 0.0 | | |

How to read the table:
* **Texture peaks.** At the result, the underside texture peak should be near 0°. It is −6 to −10°, which is the
  fine‑spin disagreement described above. The top peak should be near −72°, the calibrated top‑texture offset;
  it is −72 to −76 where the top is well seen. At the baseline the peaks sit near ±144 or −76, i.e. the baseline is
  off by grid steps.
* **Steps.** Maximum rotation step is 28.8° for the baseline and 2.3° for the result; maximum translation step is
  10.7 cm and 3.7 cm.
* **Held‑out check.** Odd frames were left out of the silhouette loss (`solve.py --holdout`). IoU on them is
  0.790 for the baseline, 0.777 for the spin‑corrected start before refinement, and 0.857 after refinement
  (0.857 on the fitted frames too), so the refinement generalises.
* **Floor.** The lowest mesh point sits about 1.5–2.5 cm below the released floor plane in both, consistent with
  plane‑fit tolerance (the feet are 3.2 ± 1 cm off the plane).
* **Size of the change.** The mean point distance between baseline and result is 27–44 cm on every frame. That
  is the spin correction.

## Depth (distance along the viewing ray): cues and decision
| cue | suggested shift of our trajectory (+ = farther) |
|---|---|
| silhouette render‑and‑compare (mesh size vs image size, before re‑anchoring) | −0.5 to −3.2 cm per phase (silhouette fit wanted the table about 2 cm closer) |
| lowest point on the released floor plane | −6.0 cm median (IQR −9.1…−3.7); weak, because rays at the table only drop about 0.25 m per m of depth and the plane has several cm tolerance |
| hand contact | not independent (the human was post‑optimised against the baseline object) |

The two usable cues do not agree within 2 cm (−2 vs −6 cm), so **the applied depth shift is 0**: the result is
re‑anchored to the baseline depth.

## Confidence per phase
* **0–600 (spin −144 w.r.t. baseline):** about 75–80 %.
  * For: underside texture, the visual lever match at 560, lever visibility at the start, and hand continuity.
  * Against: the start‑phase silhouette preference.
* **627–992 (spin −72):** about 75–80 %. Same arguments, plus the documented baseline slip at 600–627 accounts
  for the 72° difference between segments.
* **Fine spin, tilt and image position:** good where legs are visible (IoU 0.84–0.98). Possible systematic
  fine‑spin bias of 0–9° (silhouette vs texture).
* **Slip window 600–627:** interpolated and then refined. The table is nearly at rest there, so moderate–good.
* **Fast motions** (frames 470–490 and 875–890, edge‑on or blurred): lower; poses come mostly from smoothness.

## Things tried and not used
* LightGlue (SuperPoint/ALIKED) render‑to‑image matching with PnP. Matches were dominated by the symmetric legs,
  the PnP spin stayed at whatever it was initialised with, and depth was unstable (±25 cm). Code is left in
  `match.py`; `solve.py` does not use it.
* Lever‑darkness, black‑hat lever‑line and dark‑region IoU scores. Each was noisy or fooled by plate edges and
  by the mesh lever's angle (the scanned lever angles away from the board; the real one is nearly flat).
  Replaced by pooled rectified‑texture correlation.
* Blurred‑grey NCC on the underside. It is biased by baked‑in scan shading; high‑pass filtering after
  rectification fixed this.

## Reproducibility notes
* Seeds are fixed (numpy, torch, and the epoch permutation generator).
* nvdiffrast's antialias gradients use atomic adds, so repeated runs may differ at the ~1e‑5 level. I did not
  re‑run to measure this, to save shared GPU time.
* Run time: about 170 s for `solve.py`. Rough total GPU‑process wall time during development was ≤ 40 min, mostly
  CPU‑bound rectification and decoding.
