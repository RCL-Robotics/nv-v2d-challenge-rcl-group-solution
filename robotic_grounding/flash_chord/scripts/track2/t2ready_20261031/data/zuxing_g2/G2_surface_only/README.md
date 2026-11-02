# G2_surface_only: general Tier-2 object-trajectory cleaner

B_hands (the ep28 squeegee batch MAP smoother) made general and reproducible for all 30 Tier-2 episodes. The hand
model is one uniform surface model for every object (no handle axis). The other B_hands terms are kept.

## Usage

```
PY=/root/autodl-tmp/zuxing.lu/envs/kit/bin/python
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 $PY -I -B run_episode.py <ep> --out <dir>
bash run_all.sh <dir>            # all 30 episodes, 6 in parallel, then <dir>/proxies.csv (make_csv.py)
$PY -I -B check_ep28.py <dir>    # ep28: distance to the original B_hands output and to the initial estimate
```

Outputs: `<dir>/pose_ep<NN>.npy` is (T_visible, 7) `[x y z qx qy qz qw]`, in metres, in the OpenCV world, on the visible
frames only, in the same layout as the initial estimate. `<dir>/work/ep<NN>/` holds `cache.npz` (the inputs),
`x_s1.npy`/`x_s2.npy` (the states after each stage), `masks_s1.npz`/`masks_s2.npz` (the detections and the posterior std),
`result.json` (the detections, the cost parts, the solver status and the proxies) and `run.log`. `<dir>/logs/ep<NN>.log` is
the stdout of each run. `<dir>/proxies.csv` is one row per episode.

`--opt key=value` changes an ablation switch for an episode run (see `common.DEFAULT_OPTS`): `handfree=pts`,
`support=legacy`, `box_pts=pts` and `settle=hands` give the B_hands variants of those parts. The pipeline itself uses the defaults, and no
value is set per episode by hand. The run is deterministic: every random draw is seeded, and the solver has no random step.

## Files

| File | Role |
|---|---|
| `common.py` | Paths, imports, the generic constants (`CONST`), the chi-square quantiles and the ablation switches |
| `prep.py` | Per-episode cache of the Tier-2 inputs; parses the support stage; runs the agile K1 smoother for the covariance |
| `model.py` | Episode data and geometry (`Episode`); the batch MAP problem (`Problem`: residual blocks, grouped finite-difference sparse Jacobian, Levenberg-Marquardt) |
| `detect.py` | Hand-free, static, grasp, stance and support-contact detection; the Laplace posterior std |
| `run_episode.py` | The only entry point: prep, then two stages of detect and solve, then the output check, the output and the proxies |
| `proxies.py` | The proxies and the pose-difference measure |
| `run_all.sh`, `make_csv.py` | Run all 30 episodes and collect `proxies.csv` |
| `check_ep28.py` | The ep28 comparison with the original B_hands `pose_ep28.npy` |
| `dev/` | Development tests (ep28 ablations, the refactor check); not part of the pipeline |

## Inputs

* The noisy object track: the Tier-2 parquet, read through `kcommon.object_track`. On invisible frames it is filled
  with the nearest visible frame. Its rest mask is the ZUPT v1 mask (`d1_masks.npz`) AND visible. The floor is
  `up_cv` and `e_cv` (`d1_frames.json`). The frame transforms are `Rs2r`, `ts2r`, `Bcv2r` and `bcv2r`.
* The Tier-2 SOMA joints (`human_motion_data_tier2`, retargeted from the Tier-2 human input), mapped to the OpenCV world
  in cm. The human root drifts, as in Tier 2.
* The object mesh, from `v2dlb.vertices.mesh_path_for(T2/mesh, obj)`. It gives three point sets. The kit's 500 vertex
  samples (`sample_vertices`). The convex hull of 20,000 random mesh vertices, drawn with seed 0. And dense surface
  samples: `trimesh.sample.sample_surface`, seed 0, with one sample per (0.3 cm)^2 of mesh area, and their face normals.
* The support stage: `reconstructed_stage/<sequence_id>_support.usda`. Cube and Cylinder prims become axis-aligned boxes
  in the robot world, with the same parser as `kfix/run_fix.py`.
* The initial estimate `kfix/out/agile+hgate__K3__<ep>.npy`, and the frozen filter parameters (`frozen_params.json`). The
  motion-prior process noise is q = max(pooled, the episode's own ML value), from `d3_ml.json`: q_a for position and
  q_al for rotation. This is the same rule as `run_fix.py`.
* The agile K1 RTS smoother is re-run on the episode (`kfilter.Episode.run(base_meas(zv=True))`). Only its smoothed
  bias covariance is used: as the uncertainty of the initial estimate in the stage-1 detection.

## Model

The state per frame is x[t] = (b_p, b_r, d_t, d_r). b_p (cm) and b_r (rad) are the object measurement bias:
p_true = p_meas - b_p and R_true = exp(-b_r) R_meas. d_t (cm) and d_r (rad) are the human root drift, with the
measured Hips as the pivot. The MAP cost is the sum of the squared normalised residuals below. "soft-L1" means the
row-wise robust loss 2(sqrt(1+|r|^2)-1).

| Block | Residual / sigma | Frames | Origin |
|---|---|---|---|
| `rw_bp`, `rw_br` | bias random walk, increment / (q_bp, q_br) * sqrt(gap) | consecutive visible frames (a gap is bridged by one link) | B_hands; frozen params q_bp = 0.0828 cm/frame, q_br = 0.755 deg/frame |
| `b0` | b_p, b_r / (0.35 cm, 3 deg) (grown by the random walk if the first frames are invisible) | first visible frame | B_hands; frozen P_b0 |
| `acc_p`, `acc_r` | constant-velocity prior on the true object: second difference / s_a, s_al with s = sqrt(2/3 q dt^3) | all | B_hands; agile q per episode |
| `zv` | true object velocity / (sigma_zv * dt) | static pairs | B_hands; frozen sigma_zv = 0.424 cm/s, deg/s |
| `rw_dt`, `rw_dr`, `d0` | human root drift random walk / (0.0774 cm, 0.217 deg per frame); start / (0.35 cm, 3 deg) | all | B_hands; frozen params |
| `feet` | stance foot (ankle, toe joints of the drift-corrected human) per-frame motion / 0.10 cm, soft-L1 | stance pairs | B_hands |
| `hands_surf` | **new**: for each of the 14 grip points of a grasping hand, its vector to the nearest dense surface sample (object frame) / s_surf, soft-L1 per point | grasp frames | G2, see below |
| `hands_slip` | **new**: for each grip point, its motion relative to the object between frames t and t+1, projected on the surface normal at its nearest sample / s_slip_pt, soft-L1 per point | grasp pairs | G2, see below |
| `floor` | lowest convex-hull point below the floor / 0.3 cm (hinge); support contact: lowest-point height / 1 cm, soft-L1 | all; floor-contact frames | B_hands |
| `box` | max penetration depth of the object points into any support box / 0.3 cm (hinge); support contact: distance to the nearest touched box / 2 cm, soft-L1 | frames within 0.6 m of a box; box-contact frames | B_hands; object points = kit samples plus hull points |

The hand model (G2) uses the 14 grip points of B_hands for each hand: the thumb distal joint and thumb tip, and the PIP,
DIP and tip of the four fingers, all on the drift-corrected human. Its two sigmas are:

* s_surf = s_c * sqrt(n_corr) * sqrt(K/2), with s_c = 2.5 cm, n_corr = 30 and K = 14. The term pulls the object surface
  onto the grip points. The joint-to-skin offset (about 1 cm) is absorbed in s_c. s_c and n_corr are the B_hands values
  for the grip-centre offset (B_hands' per-frame sigma was s_c * sqrt(n_corr)). The factor sqrt(K/2) makes the K 1-D
  normal residuals of a hand weigh, together, as one 2-D grip-centre residual of B_hands. This holds for a handle whose
  normals spread around the grip, so on a handle the hand term has the weight of B_hands'. The vector to the nearest
  sample is used, not the point-to-plane distance. Its norm is the distance to the sampled surface, so a grip point
  beyond the end of the object is penalised. This replaces B_hands' axial-span term.
* s_slip_pt = s_slip * sqrt(K/2), with s_slip = sqrt(s_hf^2 + 0.0215). The 0.0215 cm^2/frame is the B_hands slip
  random walk. s_hf is derived per episode: it is the median, over the 28 grip joints of both hands, of the ZUPT-v1 PSD
  floor of the joint's frame increments (median of the 7.5-15 Hz band, sigma = sqrt(floor * fps / 2)). On ep28,
  s_hf = 0.117, where B_hands hard-coded 0.125. On a cylinder, the motion normal to the surface is the motion
  perpendicular to the axis, which is what B_hands constrained. Sliding along the surface, and rolling about a handle,
  stay free, as in B_hands.
* Correspondences: the nearest sample comes from a KD-tree (scipy `cKDTree`). It is recomputed at every cost
  evaluation, and frozen while the Jacobian is differenced, so that a switch of the nearest sample cannot spoil a
  finite difference.

The solver is the one of B_hands. Levenberg-Marquardt starts at lam = 1e-3, divides it by 3 after an accepted step and
multiplies it by 4 after a rejected one, tries at most 10 times per iteration, and runs at most 60 iterations. It stops
when the relative decrease falls below 1e-6 (`rel_tol`), or when no step decreases the cost (`no_decrease`, a minimum
to numerical precision). Hitting the iteration limit is `max_iter`, which is reported as not converged. The Jacobian is
a grouped forward finite difference, with steps of 1e-4 cm and 1e-6 rad. Frames with t mod G = k are perturbed
together, where G = 3, or the longest residual span when an interior invisible gap exists. The sparse normal equations
are solved by `spsolve`.

## Two stages

1. Detection under the initial estimate. Its uncertainty is the agile K1 smoothed bias covariance:
   sbp = sqrt(mean diag P_bp) and sbr = sqrt(mean diag P_br).
2. Solve from the initial estimate. The human drift starts at 0.
3. Re-detection under the stage-1 estimate. Its uncertainty is the Laplace posterior std: (J^T J)^-1 at every tenth
   frame, interpolated.
4. Solve from the stage-1 estimate, and output the visible frames. The output is checked to be finite, of the shape of
   the initial estimate, and made of unit quaternions.

## Detection rules (`detect.py`)

All tests use chi-square 99 % quantiles (1 dof 6.635, 2 dof 9.210, 3 dof 11.345). They are inflated by
sig_abs = sqrt(sbp^2 + (lever * sbr)^2 + human drift std^2). The human drift std is
sqrt(0.35^2 + t q_ht^2 + 45^2 (h0_r^2 + t q_hr^2)), with a 45 cm hips-to-hand lever.

| Mask | Rule | Origin |
|---|---|---|
| hand-free | For both hands, the closest of the 25 measured hand joints is farther than 5 cm + 3 sqrt(3) sig_abs from the object. The object is the dense surface samples. | B_hands with the kit's 500 vertex samples. Dense samples are used because 500 vertices leave a large flat face (desk, box) almost empty, so a hand on it would count as free. |
| static | The ZUPT rest mask, or untouched and settled (no contact in the last 30 frames). Untouched means body-free: every one of the 77 measured SOMA joints is farther than 5 cm + 3 sqrt(3) sig_abs from the object surface. Gaps of 2H = 14 frames or less are bridged when untouched. Runs shorter than 10 frames are dropped. | B_hands, which used hand-free as untouched. G2 uses body-free because the B_hands rule failed in the first 30-episode run (kept in `dev/out_v1_settle_hands`). In ep13 the snack box is pushed about 45 cm along the floor by a foot (foot joints within 0-4 cm of it, frames 380-500) while both hands are far. B_hands' rule declared it static, so the zero-velocity term made the bias absorb the push: rw_bp cost 8.5k against about 1.2k elsewhere, a correction of 19 cm p50 and 29 cm max, and the grasp-frame hand distance grew from 1.8 to 9.9 cm. The same happens in ep29, where the luggage is moved by a foot. |
| grasp (per hand) | near: the median over the 14 grip points of the squared distance to the surface is < chi2_1 (sig_abs^2 + s_c^2). co-moving: the median over the grip points of the squared windowed (t +- 7) displacement relative to the object, along the local surface normal, is < chi2_1 * span * s_inc^2, with s_inc^2 = q_bp^2 + (lever q_br)^2 + s_slip^2 + q_ht^2 + (45 q_hr)^2. Grasp = near AND co-moving. Gaps of 5 frames or less are filled, runs shorter than 10 frames are dropped, and runs with fewer than 10 non-static frames are dropped. | G2. The surface version of B_hands' axis test (a 2-D chi-square on the grip-centre offset from the axis and its windowed change). The median makes the test robust to the grip points that do not touch. |
| stance (per foot) | The ankle and toe joints each move less than sqrt(chi2_3 * span) times their own PSD-floor sigma over t +- 7, and the toe is lower than its median height + 10 cm. Runs shorter than 10 frames are dropped. | B_hands |
| support contact | On each run of hand-free static frames, a box is touched if the run's median squared distance to it is < chi2_1 (sig^2 + 2^2), where sig is the object std at the closest point. The object rests on the top of a touched box if its lowest hull point is at the box-top height by the same test. If it rests on no box top, gravity says it rests on the floor: the floor contact is on, and it may lean on the touched boxes. On a box top, the floor contact is off. | G2. B_hands applied the floor contact and the contact with the nearest box on every such frame. That is wrong for an object on a table, but it equals this rule for ep28 (a squeegee leaning on a chair back). |

## Generic constants

All constants are the same for every episode.

| Constant | Value | Chosen how |
|---|---|---|
| s_fs | 0.10 cm | B_hands (stance-foot residual motion) |
| slip random walk | 0.0215 cm^2/frame | B_hands |
| s_c, n_corr | 2.5 cm, 30 frames | B_hands (grip offset; 1 s per independent offset sample) |
| s_pen | 0.3 cm | B_hands (non-penetration) |
| s_sup_f, s_sup_b | 1 cm, 2 cm | B_hands (support contact) |
| d0_free | 5 cm | B_hands |
| settle | 30 frames | B_hands (1 s) |
| H | 7 frames | ZUPT v1 window |
| minimum run, gap fill, moving frames | 10, 5, 10 frames | B_hands |
| box_near | 0.6 m | B_hands (where the box terms are evaluated) |
| lever_human | 45 cm | B_hands (hips-to-hand lever for the root rotation drift) |
| surface spacing | 0.3 cm | G2: much smaller than s_c, so the sampling error of a distance (under about 0.2 cm) is negligible |
| LM | lam0 1e-3, /3, *4, 10 tries, tol 1e-6, 60 iterations; FD steps 1e-4 cm, 1e-6 rad | B_hands |

The per-episode values are derived by fixed rules from the episode's data: the agile q_a and q_al, s_hf (so s_slip), the foot
PSD floors, the rest mask, the K1 and Laplace uncertainties, the support boxes and the number of surface samples. There is
no noise cap and no bound constant.

## Differences from B_hands and the ep28 check

The non-hand residual blocks were checked against B_hands (`dev/refactor_check.py`). At the B_hands ep28 stage-2 solution,
with its masks, the legacy support rule and the kit points for the boxes, every one of them reproduces the cost logged by
B_hands to 0.1. The remaining differences from B_hands are deliberate:

1. The hand model: the surface terms above, in place of the handle axis and span.
2. s_hf derived per episode (0.117 on ep28, against the hard-coded 0.125).
3. Hand-free detection on the dense surface samples.
4. The box points are the kit samples plus the hull points.
5. The support-contact rule (gravity). On ep28 it chooses the same contacts as B_hands.
6. The invisible-frame handling (no bias link to an invisible frame). Only ep11 has one invisible frame, its last.
7. The settled-static rule uses body-free instead of hand-free (see the detection table).

The ep28 check is in `check_ep28.py`. It compares with the original `kfix/ep28/B_hands/pose_ep28.npy` over the kit's 500
points. The ablation rows are ep28 runs in `dev/` with `--opt` switches:

| Run | Max rot diff (deg) | p50 rot diff (deg) | Mean vertex dist (cm) | p95 / max vertex dist (cm) |
|---|---|---|---|---|
| initial estimate vs B_hands | 28.41 | 19.96 | 9.70 | 20.98 / 22.73 |
| surface hand model only, all other rules as B_hands (`handfree=pts support=legacy box_pts=pts settle=hands`, `dev/e28_legacy2`) | 17.36 | 3.42 | 1.71 | 4.36 / 5.86 |
| + dense hand-free, gravity support rule, kit+hull box points (`settle=hands`, `dev/e28_default`) | 9.03 | 2.64 | 1.92 | 3.93 / 5.93 |
| **G2 final (+ body-free settled rule), `out/pose_ep28.npy`** | **11.39** | **3.29** | **2.57** | 5.43 / 7.66 |

The rotation difference is shared between roll about the squeegee's long axis (p50 1.95 deg, max 6.87 deg) and tilt
of that axis (p50 1.78 deg, max 9.09 deg). Neither hand model observes the roll: the handle is a cylinder, so both
B_hands' grip-centre offset and the surface distance are invariant to it. The roll is therefore set by the weak priors.
The largest differences fall in frames 50-150, the approach and grasp onset (max 11.4 deg, vertex mean 4-5 cm). There
the final rule drops frames 56-57 and 69-85 from the static and support-contact sets, because a body joint is near the
squeegee although both hands are far. Smaller differences come from the grasp runs: G2 stage 2 has L 103-754, 771-876
and R 108-885; B_hands has L 107-752, 773-846, 864-873 and R 109-880.

## 30-episode run

`out/` is the final run (`run_all.sh out`, 2026-10-22 03:58-04:09 +08:00, 6 in parallel). All 30 episodes finished, and
both stages of every episode stopped on `rel_tol` (none hit `max_iter`). Every `pose_ep<NN>.npy` is finite, has unit
quaternions, and has the shape and dtype of its initial estimate. Run time per episode is 13-447 s; the box term with
many hull points is the slow part (ep9 big_red_bowl, ep25, ep3, ep7, ep20).

Determinism: re-running ep5 (`dev/det05`) gives a bitwise-identical pose.

`dev/out_v1_settle_hands/` is the first 30-episode run, with B_hands' hand-free settled rule. It is kept to document
the ep13 and ep29 failures that led to the body-free rule.
