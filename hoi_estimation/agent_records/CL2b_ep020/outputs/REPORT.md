# Rice cooker, episode 20: object trajectory report

Output: `object_poses_world.npz` (`T_world_mesh`, 553×4×4 float64). It is produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
(about 125 s wall time, about 2 min GPU, peak about 2.5 GB GPU memory). Helper modules: `hoi_lib.py`, `traj_opt.py`.
The validation scripts are in `validation/`. Log: `solve_log.txt`; numbers: `solve_validation.json`.

## What the video shows
A person walks to a bar-height round table. Their hand rests on the cooker's lid and then **turns the cooker
in place on the table** about its vertical axis: about −100°, a pause, then about −210° more, about −310° in total.
In the frames it is never lifted visibly off the table. Then the person walks away. The object is small (about 60×65 px).
The baseline's overall orientation is correct. A 360° per-frame yaw sweep and pooled sweeps found no 180° flips:
the label, blue tape and handle all line up.

## Phases (detected automatically by `segment_phases`; I checked them by eye)
The detector uses the median-filtered frame-to-frame difference of the visible object pixels (person pixels excluded),
with a 0.8 grey-level threshold. Short runs are removed. Moving phases are padded by 6 frames, because motion onset
is gradual: a per-frame segment can represent a still object, but a static segment cannot represent a moving one.

| phase | frames | content |
|---|---|---|
| S1 | 0–211 | untouched (hand rests on the lid from about frame 135, no motion) |
| M1 | 212–281 | turned by about −100° on the table |
| P | 282–310 | held still by the hand |
| M2 | 311–426 | turned by about −210° on the table |
| S2 | 427–552 | released, at rest |

## Method
1. **Upright model.** Measured against the released floor normal, the baseline's mesh −z axis is within a median
   1.15° (max 4.5°) of vertical. The cooker stands on a table, so I constrain mesh −z to equal the floor normal. A pose
   is then: yaw θ, image position (u, v) of the mesh origin, and depth z.
2. **Joint render-and-compare over all 553 frames** (nvdiffrast, textured mesh, 2× supersampled crop, antialiased).
   - Data term: silhouette IoU with the SAM2 object mask (dilated person pixels excluded), plus 0.5 × NCC of
     band-pass (difference-of-Gaussians) colour. The render is composited over the real background.
   - Static phases share one parameter set, so evidence is pooled over hundreds of frames. Moving frames have their
     own parameters, with second-order smoothness on θ, u, v.
   - Initialised from the baseline (unwrapped yaw, median image position).
   - The texture term never moves depth, because texture scale is biased by blur.
3. **Relative depth (shape of the trajectory along the viewing rays).** I computed three cues. The contact-line
   cue is used; the other two are cross-checks.
   - *Contact line* (used): the dark contact shadow between the round base and the table top. Its image row at the
     base-centre column is measured per frame with sub-pixel precision; the std within static phases is 0.01–0.02 px.
     The table top is assumed parallel to the floor, so this row gives the depth of the base (about 2.1 cm per px).
   - *Silhouette scale*: a per-frame 1-D IoU depth sweep.
   - *Fitted table height*: the bottom centre of the fitted pose held at a constant height above the floor.

   Depth of each phase relative to S1, in cm:

   | cue | S1 | M1 | P | M2 | S2 |
   |---|---|---|---|---|---|
   | contact line (used) | 0 | +1.0 | +1.8 | +4.1 | +4.8 |
   | silhouette scale | 0 | +0.5 | −0.6 | +3.1 | +3.4 (+5.4 in an earlier run) |
   | fitted table height | 0 | +0.9 | +2.7 | +1.8 | +1.8 |
   | baseline | 0 | +0.3 | −1.2 | +2.6 | **−3.6** |

   All three image cues say the cooker was pushed **away** from the camera by about 2–5 cm while being turned.
   The baseline instead puts S2 3.6 cm closer than S1. The baseline's depth is also not constant inside static phases:
   std 3.2 cm in S1 and 4.1 cm in S2, with a slow 7 cm drift over frames 0–100 while nothing moves. The physics check
   below shows the baseline's S2 floating 1.2 cm higher on the table than S1. The relative profile is the contact cue:
   static phases take their median, moving phases are lightly smoothed, and segments are blended linearly at the
   boundaries for continuity.
4. **Re-anchoring.** Camera-frame positions are scaled along the viewing rays (scale 1.0078) so that the **episode**
   median camera depth equals the baseline's episode median (3.092 m). The image position is kept.
5. **Second optimisation pass** of θ, u, v at the final depths. Then T_world_mesh = T_world_cam · T_cam_mesh.

### Depth decisions (independent physical cues)
- **Global depth shift applied: 0** (episode median depth = baseline's). Cues for an absolute shift:
  - Silhouette size alone suggests about −5 to −7 cm (closer). This is the known mask-dilation bias, so I rejected it.
  - The reconstructed human at frame 0 is about 6 cm closer than the reference human, which hints the object might
    be farther. It is indirect.
  - Hand contact: the baseline's hand is already pinned to the surface (distance about 2 mm), so it is uninformative.

  The cues disagree, so no global shift.
- **Relative (per-phase) depth changed.** Contact line and silhouette scale are two cues measured in different ways.
  They agree on S2 − S1 = +4.8 / +3.4 cm, versus −3.6 cm for the baseline. The resulting difference from the
  baseline's per-phase medians is: S1 −1.6, M1 −0.9, P +1.4, M2 0.0, S2 **+6.7 cm**. The correction (about 7 cm)
  is much larger than the disagreement between the cues (about 1.4 cm).

## Validation (no ground truth)
### Per phase, using the same image metrics for baseline and result (from `solve_validation.json`)
| phase | IoU base | IoU ours | NCC base | NCC ours | rot step base/ours (deg, median) | trans step base/ours (mm, median) | depth diff ours−base mean / median (cm) | yaw diff (deg) |
|---|---|---|---|---|---|---|---|---|
| S1 0-211 static | 0.888 | 0.942 | 0.204 | 0.278 | 0.51 / 0.02 | 6.7 / 0.0 | −0.5 / −1.7 | +3.1 |
| M1 212-281 moving | 0.902 | 0.947 | 0.278 | 0.480 | 1.73 / 1.46 | 6.5 / 0.5 | −1.0 / −1.2 | +1.2 |
| P 282-310 static | 0.894 | 0.940 | 0.292 | 0.521 | 0.78 / 0.03 | 28.4 / 0.0 | +1.5 / +1.3 | −2.1 |
| M2 311-426 moving | 0.902 | 0.933 | 0.411 | 0.587 | 1.82 / 1.64 | 6.7 / 0.7 | +0.3 / −0.1 | +1.2 |
| S2 427-552 static | 0.884 | 0.927 | 0.280 | 0.417 | 0.71 / 0.00 | 10.1 / 0.0 | +5.3 / +6.8 | +0.3 |

The baseline's per-frame steps in static phases (0.5–0.8°, 7–28 mm per frame) are pure jitter. Static rotation
steps in my result are 0.00–0.03°, from float32 round-off when composing the shared pose.

### Held-out frames (`validation/heldout.py`)
Every 4th frame was removed from the data term. The trajectory was re-optimised and then scored on those frames.

| phase | n | IoU base | IoU ours | NCC base | NCC ours |
|---|---|---|---|---|---|
| S1 | 53 | 0.886 | 0.946 | 0.199 | 0.311 |
| M1 | 17 | 0.898 | 0.947 | 0.270 | 0.482 |
| P | 8 | 0.888 | 0.939 | 0.293 | 0.521 |
| M2 | 29 | 0.902 | 0.931 | 0.416 | 0.584 |
| S2 | 31 | 0.882 | 0.926 | 0.265 | 0.418 |
| all | 138 | 0.890 | 0.938 | 0.274 | 0.426 |

Ours beats the baseline on 99% (IoU) and 98% (NCC) of the held-out frames.

### Physics (`validation/physics.py`)
Height of the object's lowest point above the floor plane, and spread of the world position:

| phase | height base | height ours | position std base | position std ours |
|---|---|---|---|---|
| S1 | 80.4±0.8 cm | 80.7±0.0 | 3.2 cm | 0.0 |
| M1 | 80.1±0.4 | 80.6±0.2 | 2.2 | 0.5 |
| P | 80.3±0.5 | 80.8±0.0 | 3.3 | 0.0 |
| M2 | 80.1±0.4 | 80.4±0.2 | 2.7 | 1.2 |
| S2 | 81.6±0.7 | 80.5±0.0 | 3.9 | 0.0 |

My result keeps the cooker on one table height; the baseline lifts it 1.2 cm in S2. This check is partly circular,
because the contact cue assumes a level table.

Visual overlays (render of baseline vs. result on 9 frames, including hand-occluded ones) were checked by eye.
Both have the right orientation; the result aligns better with the silhouette. The baseline sits about 1.5–3 px to
the right, and the result now corrects that (about 0.6–1.1 cm sideways at constant depth).

## What changed compared with the baseline (summary)
- **Static phases:** one constant pose each, instead of per-frame jitter (baseline depth std 3–4 cm, about 0.5–0.8°
  rotation noise per frame).
- **Rotation:** upright constraint (baseline tilt 1–4.5° removed). Yaw refined by pooled texture evidence
  (S1 +3.1°, P −2.1°, others about ±1°). No flips were found or needed.
- **Image position:** about 1.5–3 px shift to the left, supported by the silhouette.
- **Depth:** smooth, physically consistent relative profile, with the episode median kept at the baseline's.
  Largest change: S2 about 6.7 cm farther than the baseline's S2.

## Confidence
- **Rotation:** high in all phases (texture, label and tape agree, the object is upright). Remaining yaw uncertainty
  is about ±2–3°, worst in S1, whose frontal flower-print view is weakly textured.
- **Image position:** high.
- **Depth:**
  - The S1/S2 *relative* depth: medium-high. Three image cues agree on the sign; contact and silhouette agree on
    the size to within about 1.5 cm.
  - The *absolute* episode depth: inherited from the baseline (unverifiable here).
  - The ranking by risk: S2 (+6.7 cm vs baseline) is the decision with the largest effect. If the table were not
    level, or the base contact shadow changed with rotation, part of this change could be wrong. The silhouette
    cue, which needs neither assumption, still supports +3 to +5 cm.
- **Moving phases:** medium-high. The cooker never left the table visibly; yaw follows the per-frame evidence with
  smoothness.

## Reproducibility notes
- `solve.py` fixes seeds and uses no randomness. nvdiffrast's backward pass uses atomic additions, so two runs differ
  very slightly: 0.44 mm mean and 1.5 mm max point distance between two complete runs.
- The constants in the code are set in `traj_opt.segment_phases` and `solve.py`: phase threshold 0.8, padding 6
  frames, smoothness weights, NCC weight 0.5, crop 128×112 around the baseline's median projection.
- Phase boundaries are not hard-coded; they are detected from the video.
- The total GPU time used during development and the final run was roughly 20 min.
