# Tier-2 Kalman (RTS) object-pose filter: pre-registration of K0 to K3

**Status:** frozen. The sha256 of this file, the UTC freeze time and the sha256 of every frozen artefact are in the sidecar `PREREGISTRATION.sha256.json`.

**Location:** `/root/autodl-tmp/haoran.zhang/tmp_analysis/tier2/kalman/`. All paths below are relative to it unless absolute.

**Scope:** input-level denoising of the released Tier-2 object trajectory, under challenge rule 2.8.c. A Tier-2 submission must be produced from Tier-2 input only.

## 0. Inputs

**Read for this design:**
- Tier-2 dataset `data/video_to_data_challenge/track_2/tier_2_synthetic_noise/`: parquet, meta, mesh, ground_plane, support_surface and README.
- Tier-2 retarget output `data/human_motion_data_tier2/` (produced by `soma_to_g1.py` from the Tier-2 input).
- The kit metric code and the track spec.
- The retarget source code, read to learn its frame conventions.
- Files of the earlier study: `filter_study/noise/{noisy_only_selection.json, zupt_static_masks.npz, variogram_noisy_only.json, filtered_object_poses.npz}` and the scripts `common.py`, `filter_apply.py`, `variogram_noisy_only.py`, `human_filters.py` and `score_object.py` (code only).

**Noise properties used by the design:**
- **Random-walk form.** Re-derived from noisy data. On 48 object rest runs of at least 80 frames, the noisy pose variogram is linear in lag (log-log slope 1.02 for position, 1.08 for rotation). The fitted zero-lag intercept is at most 0, so there is no white component (d1).
- **Frame 0.** The bias prior mean is 0 at frame 0. This is the convention "no correction at frame 0", which ZUPT v1 also makes, and it is unavoidable for degrees of freedom that no anchor reaches. The prior standard deviation (3.0 deg, 0.35 cm per axis) is **estimated** from the intercepts of the physics-anchor residuals (d2b). It is not assumed to be zero.
- **Saturation.** From the noisy data and physics, d2b finds that the rest-anchor tilt residual variance stops growing after about 375 frames, at a per-axis std of about 14.9 deg. The filter does **not** use any cap: the bias is an unbounded random walk. Saturation is only reported (risk R2).

## 1. Data availability

| Need | Used by | Source | Status |
|---|---|---|---|
| Object pose and visibility | all | tier-2 parquet `observation.object.pose` / `.visible` (OpenCV world) | 30/30 episodes; at least 99.89% of frames visible |
| Object mesh | K2, K3 | `tier_2_synthetic_noise/mesh/<obj>/<obj>.glb`, loaded like the kit (`trimesh.load(force="mesh", process=False)`) | 29 objects, all present. **None is a closed volume**, so the centre of mass is the area-weighted surface centroid (rule in §3.3) |
| Floor plane | K2 | `ground_plane/<seq>.json`, `ax+by+cz+d=0` in the OpenCV world (README) | 30/30. The "up" sign is chosen so that the Hips are above the plane (median Hips height 0.53 to 1.03 m) |
| Support / floor-only | K2 | `support_surface/<seq>_support.usda` (G1 world) | 16 episodes have an empty stage (floor only): 0, 4, 5, 8, 10, 12, 14, 15, 18, 19, 21, 22, 23, 24, 26, 29. The other 14 have supports. **The height anchor is applied only in the 16 floor-only episodes.** The support geometry itself is not used |
| Human hands, Hips, feet | K3, derivations | SOMA-X joints (`source_payload.soma_joints`, `soma_joints_wxyz`, 77 joints) in the Tier-2 G1 parquet, computed by `soma_to_g1.py` from the Tier-2 human input | 30/30, and the frame counts match. The maps are exact rigid transforms fitted per episode (d1): SOMA to robot max residual 3e-15 m; OpenCV to robot (object track, constant retarget lift removed) 8.5e-8 m and 1.4e-5 deg; object-frame offset 4e-6 deg, so the G1 object frame equals the GLB frame |
| Contact labels | (K3) **not used** | `hand_contact_active` in the same parquet. `soma_to_g1.py` sets it when a G1 palm or a SOMA fingertip is within 2 cm of the **noisy** object mesh | Not used, because it is built on the retargeted G1 palm, which is not rigid with the object. Its agreement with our own human-hand distance rule is 90% of frames |
| Rest masks | K1 to K3 | ZUPT v1 rule (verbatim copy of `filter_apply.static_mask`) on the noisy object | Reproduced bit-identically to `zupt_static_masks.npz`. Mean rest fraction 0.43 |

## 2. Frames, units, conventions

- **World frame.** All filtering is done in the source OpenCV world, which is the scoring frame. Positions are in cm, rotations in rad, and time steps are frames (dt = 1/30 s).
- **Bias model.** The measured pose is `y_p = p + b_p` and `R_m = exp([b_th]x) R_true`, a world-frame left bias. The output is `p_hat` (smoothed true position) and `R_hat = exp(-[b_th_hat]x) R_m`. Output is written only for visible frames, as `(T_vis, 7)` with xyz in metres and an xyzw quaternion. This is the same layout as `filtered_object_poses.npz`.
- **Orientation chart.** The task allows a small-angle chart per axis. The measured orientation is unwrapped as cumulative world increments, `phi_m(t) = sum_{s<t} log(R_m(s+1) R_m(s)^T)`, and modelled as `phi_m = phi + b_th`. Anchor and hand measurements act directly on `b_th` around `R_m` and use the exact minimal rotation.
- **Human joints.** SOMA joints are mapped to the OpenCV world with the d1 rigid maps. The "hand" is the SOMA `LeftHand` / `RightHand` wrist joint; the "fingers" are the five finger-end joints. The root-rotation pivot is the `Hips` joint.

## 3. Detection rules (fixed)

### 3.1 Object rest
The ZUPT v1 rule, verbatim:
- Window half-width H = 7.
- A chi-square(3) 99% test on the windowed displacement and rotation angle, with the per-episode step sigma taken from the velocity-spectrum floor (7.5 to 15 Hz).
- Gaps of at most 15 frames are bridged when the pose across the gap is noise-consistent.
- Runs must be at least 10 frames.

Zero-velocity (ZV) is applied on rest frames that are also visible.

### 3.2 Grasp distance
- Per frame and per side, compute the minimum distance from the six hand points (wrist and five finger ends) to the object surface. The surface is the mesh vertices, subsampled to at most 60k, under the **noisy** object pose.
- **Threshold `d_g` = 1.598 cm.** This is Otsu's threshold on log10 of the per-frame minimum distance over the 13,576 object-moving visible frames, pooled over the 30 episodes (d2).

### 3.3 Stable resting poses (K2)
- Computed with `trimesh.poses.compute_stable_poses(mesh, center_mass=c, sigma=0, n_samples=1)`, which is deterministic.
- The centre of mass `c` is `mesh.center_mass` if the mesh is a valid volume; otherwise it is the area-weighted surface centroid. No mesh is a valid volume.
- Each pose k gives the outward contact normal `n_k` (object frame), the origin height above the support `h_k` (checked against max over hull vertices of `n_k . v` to 6e-17 m), and the basin probability `prob_k`.
- Objects have 2 to 118 poses (median 7).

### 3.4 Anchor frames (K2)
- For every object rest run, take the longest sub-run of at least 10 frames that is **hand-free** (both hands at least `d_g` away) and visible.
- The anchor frame `t_m` is its midpoint. There is at most one anchor per rest run, because model error is fully correlated within a run.

### 3.5 Rigid carry (K3): "ZUPT in the hand frame"
For each side, a frame counts as rigid carry when all of these hold:
- The verbatim ZUPT v1 static rule, applied to the hand-to-object relative pose (`p_rel = R_hand^T (p_obj - p_hand)`, `R_rel = R_hand^T R_obj`), says the relative pose is static. The rule uses per-frame sigmas **predicted** from the d1 random walks:
  - `sig_r_rel = sqrt(q_br^2 + q_hr^2)` = 0.785 deg/frame/axis.
  - `sig_p_rel = sqrt(q_bp^2 + q_ht^2 + (2/3) q_hr^2 median|p_obj - p_hips|^2)`, between 0.19 and 0.32 cm/frame/axis depending on the episode.
- That side's distance is below `d_g`.
- The object is not at rest.
- The frame is visible.

Runs must be at least 10 frames. The dominant side (more rigid-carry frames) is assigned first, and the other side only gets frames that are still free. Rigid carry covers 10.1% of frames on average and is absent in 9 episodes (0, 1, 4, 16, 17, 19, 20, 28, 29).

## 4. Models

**State.** 18 components for K0 to K2, 24 for K3:
`x = [p(3) cm, v(3) cm/s, phi(3) rad, w(3) rad/s, b_p(3) cm, b_th(3) rad | d_t(3) cm, d_th(3) rad]`

**Dynamics per frame.**
- Constant velocity for `(p, v)` and for `(phi, w)`, with process noise `q [[dt^3/3, dt^2/2], [dt^2/2, dt]] (x) I3`, where q is `q_a` or `q_al`.
- `b_p` and `b_th` are random walks with per-frame step variances `q_bp^2 I3` and `q_br^2 I3`.
- `d_t` and `d_th` are random walks with steps `q_d_t^2` and `q_d_th^2`. They are re-initialised at every rigid-carry onset.

**Prior.**
- `p ~ N(y_p(0), 100^2 I)`
- `v ~ N(0, 100^2 I)`
- `phi ~ N(0, 1 I)`
- `w ~ N(0, 10 I)`
- `b_p ~ N(0, P_b0_pos^2 I)`
- `b_th ~ N(0, P_b0_rot^2 I)`

**Estimator.** A forward Kalman filter (Joseph-form update) followed by a Rauch-Tung-Striebel backward pass over the whole clip (`scripts/kalman_core.py`).

### Measurements

**K0.** On visible frames only:
- `y_p = p + b_p + n`, with `n ~ N(0, sigma_w_pos^2 I)`.
- `phi_m = phi + b_th + n`, with `n ~ N(0, sigma_w_rot^2 I)`.

**K1.** K0, plus on rest and visible frames:
- `0 = v + n`, with `n ~ N(0, sigma_zv_pos^2 I)`.
- `0 = w + n`, with `n ~ N(0, sigma_zv_rot^2 I)`.

**K2.** K1, plus at each anchor frame `t_m`. The quantities used here come from the K1 smoothed state (`b_hat1`, `P1`) at `t_m`.

*Tilt anchor:*
1. For every stable pose k, compute `d_k = R_m(t_m) n_k` and `rho_k`, the minimal rotation vector taking `d_k` onto `-up`. `rho_k` is perpendicular to up.
2. The candidate measurement is `z_k = E(-rho_k)` on `E b_th`, where E is a 2x3 orthonormal horizontal basis.
3. Compute `S = E P1_bth E^T + sigma_tilt^2 I2`, and `m_k = (z_k - E b_hat1)^T S^-1 (...)`.
4. Candidates are the poses with `m_k < 9.21` (chi-square(2), 99%). With none, there is no anchor.
5. Probabilistic data association by moment matching: weights `w_k ∝ prob_k exp(-m_k/2)`, then `z = sum w_k z_k` and `R = sigma_tilt^2 I2 + sum w_k (z_k - z)(z_k - z)^T`.

*Height anchor:* floor-only episodes, and only when the tilt anchor was accepted.
1. Candidate heights are `z_k = 100(h_k - e) + mu_h`, and the measurement is `up . p`.
2. The same weights give `z = sum w_k z_k` and `R = sigma_h^2 + sum w_k (z_k - z)^2`.
3. The anchor is accepted if `nu^2 / S < 6.63` (chi-square(1), 99%).

**K3.** K2, plus hand measurements on rigid-carry visible frames of the assigned side.

*At each run onset a:*
- Take the K2 smoothed pose there: `R_K2 = exp(-b_hat2_th) R_m`, `p_K2 = p_hat2`.
- Set `R_ho = R_hand(a)^T R_K2` and `r_ho = R_hand(a)^T (p_K2 - p_hand(a))`.
- Reset `d` to mean 0 with covariance equal to the K2 smoothed covariance of `(b_p, b_th)` at a, which is the onset error of `T_ho`.

*On each frame t of the run:*
- Rotation: `z_rot = log(R_hand(t) R_ho R_m(t)^T) = d_th - b_th + n`.
- Position: `z_pos = p_hand(t) + R_hand(t) r_ho - y_p(t) = -b_p + d_t - [r]x d_th + n`, where `r = y_p(t) - p_hips(t)`. This is the root-rotation lever about the Hips.
- Noise is `n ~ N(0, diag(slip_white_pos^2 I3, slip_white_rot^2 I3))`.
- Each 6-D hand measurement is skipped when its innovation fails the chi-square(6) 99% gate (16.81), checked in the forward pass.

Reference implementation: `scripts/kfilter.py` (Episode.variant) and `scripts/kalman_core.py`.

## 5. Parameters (frozen in `design/frozen_params.json`)

Units: positions cm, angles deg unless stated, per frame per axis.

| Parameter | Value | How it was derived | Source |
|---|---|---|---|
| `q_bp` (object position bias step) | 0.0828 cm | Object rest-run variogram (48 runs of at least 80 frames), least squares `gamma(l) = a + s l` on lags 1, 2, 4, 8. `s` = 0.02056 cm^2/frame (bootstrap CI 0.0184 to 0.0229), and `q = sqrt(s/3)`. Per-axis slopes 0.0071, 0.0064, 0.0071, so it is isotropic. Intercept CI is below 0 | d1 |
| `q_br` (object rotation bias step) | 0.7547 deg | Same fit on rotation: `s` = 1.709 deg^2/frame (CI 1.41 to 2.03). Per-axis 0.583, 0.581, 0.545 | d1 |
| `sigma_w_pos`, `sigma_w_rot` (white measurement noise) | 0.01 cm, 0.01 deg | The variogram intercepts are at most 0, so there is no white component. This value is only a numerical floor | d1 |
| `sigma_zv_pos`, `sigma_zv_rot` | 0.424 cm/s, 0.424 deg/s | `sqrt(2) sigma_w / dt`, the velocity resolution implied by the floor | rule |
| `q_root_rot` (`q_hr`, human root rotation step) | 0.2171 deg | Variogram of the stance-foot orientation (foot stance from the ZUPT v1 rule on foot joints, 38 runs of at least 80 frames): slope 0.1414 deg^2/frame (CI 0.123 to 0.159). This is 3.5x smaller than the object step and 12x smaller in variance, which confirms that root rotation drift is much smaller than object rotation drift. A cross-check from the Hips during double stance gives a lag-1 bound of 0.20 | d1 |
| `q_root_trans` (`q_ht`) | 0.0774 cm (upper bound) | Hips lag-1 variogram during double stance (15 runs): 0.01796 cm^2 / 3. The stance-toe drift (0.169 cm^2/frame) is fully explained by the root-rotation lever (0.223), so the translation step itself is indistinguishable from 0. The upper bound is used | d1 |
| `slip_rw_rot`, `slip_rw_pos` | 0.310 deg^2/frame, 0.0215 cm^2/frame | Relative-pose variogram on rigid-carry runs of at least 40 frames (15 runs): slope minus the slope predicted from the object and root walks (rotation 2.78 vs 1.85; position 0.203 vs 0.139), divided by 3 | d2b |
| `q_d_th`, `q_d_t` (hand-chain step) | 0.598 deg, 0.166 cm | `sqrt(q_hr^2 + slip_rw_rot)` and `sqrt(q_ht^2 + slip_rw_pos)` | d2b |
| `slip_white_rot`, `slip_white_pos` | 0.01 deg, 0.01 cm | Relative-pose intercepts are below 0, so the numerical floor is used | d2b |
| `d_g` (grasp threshold) | 1.598 cm | Otsu, §3.2 | d2 |
| `sigma_tilt` (tilt anchor model error) | 3.0 deg | Maximum likelihood on the 115 hand-free rest-anchor tilt residuals (`r`, angle to the nearest stable pose, at time t). Model: `r ~ (1-eps) Rayleigh(s(t)) + eps U(0, 90)`, with `s^2 = sigma^2 + q_br^2 min(t, t_s)` and `q_br` fixed. Grid results: sigma = 3.0, t_s = 375 frames, eps = 0.05. sigma^2 contains Var(b0) as well as the model error, so it is conservative | d2b |
| `P_b0_rot` (bias prior std at frame 0) | 3.0 deg | Same intercept, since b0 cannot be separated from the model error | d2b |
| `mu_h`, `sigma_h` (height anchor offset and error) | -0.30 cm, 0.35 cm | Maximum likelihood on the 68 floor-only residuals `up.p + 100e - 100h_k`. Model: `(1-eps) N(mu, sigma^2 + q_bp^2 min(t, t_s)) + eps U(-20, 20)`. Results: mu = -0.30, sigma = 0.35, t_s = 275, eps = 0 | d2b |
| `P_b0_pos` | 0.35 cm | Height intercept, assumed isotropic over axes | d2b |
| `q_a` (CV position process noise) | 316.2 cm^2/s^3 | **Pooled** maximum likelihood over the 30 episodes under the K1 model (sum of the per-episode innovation log-likelihood curves, grid of 0.25 decades). The spectral-crossover cross-check gives 108 | d3 |
| `q_al` (CV rotation process noise) | 3.162 rad^2/s^3 | Pooled maximum likelihood, as above. The crossover cross-check gives 0.56 | d3 |
| Gates | 9.21 / 6.63 / 16.81 | Chi-square 99% for 2 / 1 / 6 degrees of freedom | rule |
| Fixed rule constants | H = 7, LMIN = 10, chi-square(3) 99% = 11.34, 7.5 to 15 Hz | Inherited verbatim from ZUPT v1 | filter_apply.py |

## 6. Design decisions made before the freeze

1. **Pooled `q_a` and `q_al` instead of per-episode values.**
   - Per-episode maximum likelihood chose `q_al` of about 1e-3 rad^2/s^3 for slow large rotations: ep00 (a 116 deg stool tip), ep08, ep10 and ep23.
   - K1 then absorbed up to 71 deg of real rotation into the bias. A Gaussian random-walk path likelihood does not penalise steps that persist in one direction.
   - That is inconsistent with the bias level seen at rest anchors.
   - Pooled maximum likelihood fixes this: the maximum K1 bias drops from 71 to 25 deg.
2. **Rigid carry uses relative-pose ZUPT, not distance alone.**
   - With distance-only carries (33% of frames), the hand-to-object relative rotation drifts much faster than a random walk would (lag-16 variogram of 819 deg^2).
   - Many contacts are pushes, rotations in the hand or touches, so distance alone is the wrong carry model (d2_carry.json).
3. **Anchor model error fitted with saturation and outliers (d2b), not with linear growth (d2).**
   - The linear-growth estimator gave 6.7 deg. Its free fit had a slope of 0.45 against the random-walk value of 1.14 deg^2/frame, so it is mis-specified.
4. **Data association by moment matching instead of "unambiguous nearest".**
   - Cylinder-like objects (baseball bat: 118 poses, bottles: about 45) have many adjacent stable poses. The matching keeps the cross-axis constraint and leaves roll about the long axis weak.

## 7. Validation done before the freeze

**Synthetic tests.** These are rest-move-rest trajectories with the d1 bias steps, 20 seeds, and no dataset reference:

| Test | Variants compared | Result |
|---|---|---|
| `synth_test.py` | none / K0 / K1 / ZUPT v1 | Mean rotation error 26.5 / 26.5 / 14.1 / 14.1 deg. Position error 3.10 / 3.10 / 1.96 / 1.88 cm |
| `synth_test2.py` (box on a floor) | none / K1 / K2 | Rotation 26.5 / 15.2 / 9.5 deg. Position 3.08 / 1.93 / 1.62 cm |
| `synth_test3.py` (adds a rigidly held box seen through a simulated root bias) | none / K1 / K2 / K3 | Rotation 26.5 / 15.2 / 9.5 / 7.2 deg. 178 of 7,200 hand frames gated (2.5%) |

These confirm the sign conventions and that each layer helps when its model holds.

**Run on the Tier-2 input** (`design/d3_dryrun_*.json`, `design/chk_k1_vs_zupt.log`):

| Variant | Result |
|---|---|
| K0 | Changes the input by 0.67 deg and 0.06 cm on average, so it is close to identity, as expected |
| K1 | Differs from ZUPT v1 by 2.8 deg and 0.26 cm |
| K2 | 115 anchor candidates: 111 tilt anchors and 66 height anchors accepted, 4 with no stable pose in the gate (3 on the leaning squeegee in ep28, 1 on the tall bar stool in ep00) |
| K3 | 84 carry runs, 2,487 hand frames, 38 gated (1.5%) |

Mean correction relative to the input: K1 16.7 deg / 1.21 cm, K2 20.1 deg / 1.35 cm, K3 20.4 deg / 1.38 cm. For comparison, ZUPT v1 differs from the input by 17.2 deg / 1.40 cm.

## 8. (not included in this copy)

## 9. (not included in this copy)

## 10. Risks and limitations

- **R1.** (not included in this copy)
- **R2. Bias saturation is not modelled.** The d2b evidence is not used in the filter. The bias variance at late times is overstated, which makes gates permissive and weights anchors and hands relatively more. Wrong associations are possible (eps = 0.05 outliers in the tilt fit).
- **R3. Bias built up during non-rigid motion is unobservable.** This covers yaw and horizontal position. It is carried forward, as in ZUPT v1.
- **R4. Hand coupling is weak and partly circular.**
  - The slip random walk (0.31 deg^2/frame) dominates root drift (0.047). The wrist joint is not the grasp frame.
  - `T_ho` inherits the K2 error at onset, and its correlation with the object bias is ignored.
  - Rigid-carry detection uses predicted sigmas, so the slip estimate is conditional on that selection.
- **R5. Root translation step is only an upper bound.** The lever-arm pivot is assumed to be the Hips.
- **R6. Mesh and support assumptions.**
  - The centre-of-mass rule is a surface centroid, because no mesh is closed.
  - Anchors assume a horizontal support. Leaning or held-still objects rely on the hand-free rule and on gating (squeegee: 3 of 4 rejected).
  - Height is skipped for the 14 support episodes, as the brief says, and the support stages are not used.
- **R7. Linearisation and coupling approximations.**
  - The small-angle world chart for orientation, with bias up to about 30 deg, has linearisation error in the constant-velocity prior.
  - The cross-covariance between position and rotation bias at anchors is ignored.
- **R8. Input-level AUC is not the policy score.** The earlier input-to-policy mapping is an unvalidated first-order model.
- **R9.** (not included in this copy)
- **R10. Different environments and mesh-loader versions.** The design ran in the v2d environment (trimesh 4.5.1, networkx); the kit environment v2dkit has trimesh 5.1.1. Mesh vertices are loaded the same way. Stable poses depend on hull and centre-of-mass numerics.
- **R11. One structured iteration before the freeze.** §6 lists the decisions taken after looking at diagnostics on the Tier-2 input.

## 11. Reproduction (CPU, nice -n 19)

```
cd /root/autodl-tmp/haoran.zhang/tmp_analysis/tier2/kalman/scripts
PY=/root/autodl-tmp/haoran.zhang/conda_envs/v2d/bin/python
$PY -B d1_avail_frames_variograms.py      # availability, frame maps, rest masks, object + root variograms
$PY -B d2_mesh_grasp_physics.py           # stable poses, grasp threshold, physics residuals
$PY -B d2b_rigid_carry_anchor_fit.py      # rigid carries, slip budget, anchor ML fits
$PY -B d3_params_ml_dryrun.py             # pooled ML q, frozen_params.json, K0-K3 -> outputs/kalman_poses.npz
$PY -B synth_test.py; $PY -B synth_test2.py; $PY -B synth_test3.py; $PY -B chk_k1_vs_zupt.py
```

## Appendix A. Frozen artefacts (sha256)

The authoritative list is the sidecar `PREREGISTRATION.sha256.json`.

```
ef747a3983503437b23b255d07fb49026a7711e3b4b2c8de570d5f53697b2994  scripts/kalman_core.py
ad14385107b2d66f41963f0e4e59027d3bb82f9af7e97c862e2b801927e9a3c3  scripts/kfilter.py
aa635fc96d6abd465200b228e2a9e8fdb70e8203c00ab9e2781c8a2a5e781fc1  scripts/kcommon.py
e94fb42cfb91af58d90f5136aebd2a1c4168b7facc522723aa555bd02ab0299e  scripts/d1_avail_frames_variograms.py
141b468cc0a1e8c5b9344338e77b275f8c1eb3742804c0117758e89a988e5f0d  scripts/d2_mesh_grasp_physics.py
5752d20ea797ce09c7f824f41c92ce7f5ef04408e47a7449b09e439cf0c46d36  scripts/d2b_rigid_carry_anchor_fit.py
b7b7f1216d89fee8d8a87061acfe4fdc4d69fe496b76cffe348d9adca4333eb3  scripts/d3_params_ml_dryrun.py
c9b7c12d3b0335929b05e1c6d180b86249446802ddf99a3d3b46fb24f33a45de  design/frozen_params.json
cb04d43259d89678f219456abebecd19459558da76707f19f4612a39e39c7833  outputs/kalman_poses.npz
ed17e235a8d3db77547bd20198e6c457d599d89705ce8b51cf862d549435f608  ../filter_study/noise/filtered_object_poses.npz (ZUPT v1 baseline)
```
