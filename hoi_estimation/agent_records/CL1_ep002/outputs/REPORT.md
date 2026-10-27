# Episode 2 (blue easter basket) – trajectory-level object pose recovery

Output: `object_poses_world.npz` (`T_world_mesh`, 980×4×4, float64), produced by
`python /work/outputs/solve.py --inputs /work/inputs --out /work/outputs/object_poses_world.npz`
(2 min 33 s wall time on the shared GPU, < 3 GB GPU memory).
Validation code is in `validation/validate.py`. Its output and the solver log are in `validation/`.

## 1. What the object and the video look like
- The basket is a tapered metal bucket: 23 cm rim diameter, 15.5 cm tall, mesh +y = opening. It carries 8 identical
  embossed bunnies and a ring of holes, which form a 45°/22.5° periodic pattern. A wire handle with a wooden grip is
  attached at two lugs and folded down to one side in the scan.
- Apart from the handle, the bucket is rotationally symmetric. **Yaw about its own axis is the dominant
  uncertainty**: a 45° yaw error is already about 7 cm ADD, and 180° is about 20 cm.
- The object is small: 45–60 px wide in the static phases, up to 90 px while carried close to the camera.

## 2. Phases (constants in `solve.py`, verified by eye plus an automatic mask-IoU check printed at run time)
| phase | frames | content | evidence |
|---|---|---|---|
| P1 | 0–122 | static on the round table | mask IoU to frame 30 ≥ 0.955 (median 0.994); first movement at 123 |
| P2 | 123–222 | carry 1 (lifted, put on the floor; table top occludes it ~175–190) | |
| P3 | 223–435 | static on the floor (hand still on the rim until ~265) | mask IoU to frame 330 ≥ 0.919 (median 0.963), drops at 436 |
| P4a | 436–745 | carry 2, visible (behind the table ~470–520, handover ~520–530, large and sharp 540–720) | |
| P4b | 746–851 | carry 2 hidden behind the body; placement on the floor behind the legs ~840–851 | object mask empty 750–846 |
| P5 | 852–979 | static on the floor behind the person's legs: a sliver is visible 847–867, then fully hidden | |

## 3. Diagnosis of the baseline
- The baseline equals CoCoNet re-expressed for the released K: pixel position preserved, depth ×1.01–1.02, plus a
  constant 1.2° ray correction.
- **Translation**: the silhouette is offset by 3–4 px in the static phases. In carry 2 (640–745) it is offset by
  20–30 px, i.e. 7–10 cm laterally, with silhouette IoU about 0.3.
- **Rotation**: 34 / 63 per-frame jumps above 10° during the two carries. A ~90° yaw jump at frame ~650 happens
  while the basket visibly does not rotate (frames 636–670). In P3 the yaw changes inside the static phase
  (28.8° → 47.6° → 22.7°). At the P1/P2 boundary the yaw jumps from −115° (frame 122) to 123° (frame 126),
  although the basket has barely moved.
- **Hidden frames** (746–979): physically meaningless. The basket is tilted 25–80° (median 82° in 746–851), 7 cm
  above the floor in P5, and the hand–object distance is 64 cm.
- FoundationPose spins randomly in P1 (yaw flips every few frames), so it gives no usable yaw there.

## 4. Method (what I changed and why)
1. **Geometry from silhouettes** (nvdiffrast, SAM2 masks; person pixels ignored; table-top pixels ignored while
   carried).
   - *Static phases*: one upright pose per phase, fitted to the pooled masks of all phase frames.
     - Floor phases: the bottom rim sits 1 cm above the released plane. The true frame-0 human's lowest vertex is
       2.3 cm above that plane (shoe soles), so the plane is at most ~2 cm below the true floor.
     - Table phase: the baseline (human-calibrated) camera depth is kept, because the table-top constraint is
       useless along this grazing ray. As a check, the table-top height measured from frame 0 (caster contacts on
       the floor plane plus the top-ellipse fit) is 0.80 m, versus a basket bottom at 0.816 m.
   - *Carried frames*: per-frame refinement of the image-plane translation plus a strongly regularised rotation
     at the temporally smoothed (median 9 + Gaussian) baseline depth, followed by SE(3) Gaussian smoothing
     (σ = 1.5 frames). Isolated network rotation spikes (>25° from the window medoid) are replaced first.
2. **Yaw from texture** (the main new evidence). For every visible frame I render the textured mesh at 180 yaws
   and compute an NCC of gradient magnitudes over bucket-body pixels, with the movable wire handle excluded.
   The resulting yaw × time map shows clean stripes (the 8 bunnies and the hole ring) that drift continuously
   while the basket is carried.
   - *P3*: the pooled 45°-periodic stripe phase snaps the baseline yaw (47.6°) to 62.2°.
   - *P4a (carry 2)*: Viterbi tracking of a stripe gives the relative yaw over 436–745. The absolute offset comes
     from two independent anchors: (a) the P3 yaw at pick-up, and (b) the CoCoNet window that agrees best with the
     track (695–740, IQR 9°). They agree to within 0.3–1° (−137.8° vs −137.5°), so the yaw is continuous from P3
     through all of carry 2. This removes the spurious 90° jump at ~650 and the 60–130° offset of CoCoNet over
     470–645.
   - *P1*: the CoCoNet P1 estimate (−115°) is self-contradictory. It puts the scanned wooden grip in front of the
     basket, yet the 60-frame averaged image has no tan pixels on the basket (2 of 2337). Its own estimates right
     after lift-off (123–153, the basket cannot have rotated yet) are 120–137°, and FoundationPose late in P1 is
     118–147°. I use the circular median of CoCoNet over 123–153 (122.9°), snapped to the P1 stripe phase:
     **139.6°**. This is an explicit constant (`P1_COARSE_FRAMES`) with the reason in the code.
   - *P2 (carry 1)*: the basket is small, blurred and occluded, and the texture does not track. Yaw is a linear
     ramp between the P1 and P3 yaws; tilt and translation come from the silhouettes.
3. **Hidden carry (P4b)**: the basket follows the holding hand: MHR wrist position plus the hand's rotation about
   the vertical, smoothed. Tilt is kept from the last reliable visible frames, because the full hand orientation
   from occluded keypoints is too noisy (it gave 32° tilts). A minimal image-plane shift then keeps the basket
   inside the person silhouette, since no object pixel is visible. The result is blended into the placement pose
   over the last 30 frames.
4. **P5**: upright on the floor, fitted to the sliver silhouettes (852–867), plus the constraint that the rendered
   basket is covered by the person mask in the hidden frames. Yaw is the mean of the hand-propagated yaw (−160.6°)
   and CoCoNet over the sliver frames (−170.8°): −165.7°. In one run where both came from the same anchor they
   agreed to 0.2°; in the final run they differ by 10°.

### Things I tried and did **not** adopt (because my own checks did not support them)
- Free-depth silhouette fits put the basket 4–5 % closer in every phase, i.e. 13–21 cm, floating 7–12 cm above
  the floor in P3. Colour segmentation of the averaged image shows the SAM2 masks are ~7 % larger in area than
  the basket, so silhouette scale is biased. **Depth is never taken from silhouette scale.**
- Hand-contact depth from the independent initial MHR (`/pipeline/mhr_init`): frame-to-frame scatter of ±10 cm,
  median −3 to −4.5 cm. Too noisy, so not used; the baseline depth is kept while carried.
- Allowing tilt in the floor-static fit raises IoU from 0.87 to 0.93, but only with a 10° tilt of a basket resting
  on a flat carpet. Rejected as implausible; it more likely absorbs a small camera/plane inconsistency.
- Free rotation in the per-frame silhouette refinement (IoU 0.895 vs 0.865) worsened texture agreement
  (0.096 vs 0.13). Rotation is therefore strongly regularised.
- Gradient-based texture optimisation of geometry: too rugged. It destroyed IoU without improving texture
  agreement.
- The broad, 360°-periodic "envelope" of the texture score is fixed in **world** yaw in every phase, even while
  the stripes drift in carry 2. I treat it as a view-dependent rendering artefact and do not use it. Only the
  fine stripe phase is used.

## 5. Validation (no ground truth; `validation/validation.json`)
Baseline vs. result, per phase. IoU is computed over unoccluded pixels against the SAM2 object mask. The texture
score is NCC of gradient magnitudes over bucket-body pixels (handle excluded). In P1 the static pose was fitted
on even frames only, so odd frames are held out.

| phase | metric | baseline | result |
|---|---|---|---|
| P1 table (0–122) | silhouette IoU (all / held-out odd) | 0.853 / 0.853 | **0.926 / 0.926** |
| | body-texture NCC | 0.157 | **0.252** |
| | tilt from floor normal | 0.8° | 0.0° |
| P2 carry-1 (123–222) | silhouette IoU | 0.746 | **0.809** |
| | body-texture NCC | **0.110** | 0.077 |
| | rot. jumps >10°/frame | 34 | **2** |
| | median rot. step / transl. step | 7.0° / 2.4 cm | 1.1° / 1.7 cm |
| P3 floor (223–435) | silhouette IoU | 0.802 | **0.870** |
| | body-texture NCC | **0.184** | 0.112 |
| | rot. jumps >10° inside the static phase | 16 | **0** |
| | bottom height above plane | 1.4 cm | 1.0 cm |
| P4a carry-2 visible (436–745) | silhouette IoU | 0.637 | **0.852** |
| | body-texture NCC | 0.118 | **0.133** |
| | rot. jumps >10°/frame | 63 | **0** |
| | median rot. step / transl. step | 4.4° / 1.6 cm | 0.8° / 1.0 cm |
| P4b hidden carry (746–851) | median tilt | 82° | **8°** |
| | basket pixels outside the person mask while invisible | 3.6 % | **0.3 %** |
| | median hand–mesh distance | 64 cm | **0.4 cm** |
| | silhouette IoU (5 visible frames) | 0.549 | **0.785** |
| P5 floor hidden (852–979) | silhouette IoU (16 sliver frames) | 0.376 | **0.799** |
| | median tilt / bottom height | 10.6° / 7.2 cm | **0° / 1.0 cm** |
| | rot. jumps >10° inside the static phase | 51 | **0** |
| | basket pixels outside the person mask while invisible | **0.7 %** | 3.0 % |

There are two cases where the texture score favours the baseline:
- **P2 and P3**: I separated the two effects by swapping yaw and geometry between the solutions. With equal
  geometry, my yaw scores equal or higher in every segment (carry 1: 0.117 vs 0.113; 436–529: 0.192 vs 0.174;
  646–745: 0.117 vs 0.091). The drop therefore comes from geometry, i.e. my silhouette-fitted position and tilt.
  In carry 1 the overlays clearly favour the silhouette-refined geometry: the baseline contours are offset or
  wrongly tilted at 145, 195 and 210. Texture NCC on 45 px blurred crops is unreliable for sub-pixel geometry.
- **P3**: the baseline's slightly tilted (2.6°), 5 cm farther pose aligns the rim texture better, while
  silhouette and floor contact favour mine. A colour segmentation of the averaged image shows the basket 3–4 px
  taller than either render, so neither model is exact there. The difference between the two P3 positions is
  ~5 cm, mostly along the viewing ray.

How much the result differs from the baseline (ADD-style mean vertex distance, median per phase):
P1 16.7 cm (yaw), P2 6.7 cm, P3 5.2 cm, P4a 13.9 cm, P4b 78 cm, P5 23.9 cm.

## 6. Confidence per phase
- **P3 floor (223–435), medium-high**. Position is good to ~2 cm laterally and ~3–5 cm along the ray: floor
  contact and baseline depth agree within 5 cm. Yaw rests on CoCoNet's coarse yaw, consistent with the
  independent carry-2 anchor, plus the stripe phase. A residual 45° (bunny-period) ambiguity cannot be excluded.
- **P4a carry 2 visible, medium-high in 540–745** (large, sharp frames; yaw tracked continuously; translation
  corrected by up to 10 cm). Lower in 436–530, where the basket is behind the table or legs.
- **P5 placement, medium for position** (sliver left edge, floor contact, occlusion constraint) and
  **low-medium for yaw** (two estimates 10° apart, hidden for 110 frames).
- **P4b hidden carry, low-medium**. The hand-driven path is physically plausible: 0.4 cm from the hand, inside
  the person silhouette, upright-ish. Exact rotation is unknowable.
- **P1 table, low-medium for yaw** (the evidence chain in §4.2 is indirect; the alternative is the baseline's
  −115°) and **high for position** (baseline depth kept, image position from 123 pooled frames, IoU 0.85 → 0.93).
- **P2 carry 1, low-medium**: translation from silhouettes is good, yaw is interpolated.

## 7. Reproducibility and resources
- `solve.py` recomputes everything from `/work/inputs`; it uses no `/pipeline` files and has no embedded
  per-frame results. The decisions taken by eye are the named constants at the top of `solve.py` (phase
  boundaries, `P1_COARSE_FRAMES`, `FLOOR_H`), each with its reason.
- Seeds are fixed. nvdiffrast's backward pass uses atomics, so repeated runs differ slightly: about 1° in the
  static yaws and sub-millimetre to millimetre in translation. I observed P3 yaw 62.2–63.5° across runs.
- Runtime of the final `solve.py`: 2 min 33 s, of which ~70 s is texture yaw scoring for 473 frames. GPU memory
  is < 3 GB. Total GPU time for the whole investigation is an estimated ~35 min, within the 45 min budget.
- No external downloads (see `TOOLS.md`).
