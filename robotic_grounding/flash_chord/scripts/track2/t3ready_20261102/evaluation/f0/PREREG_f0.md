# PREREG f0: Tier-3 test on Han Zhou's frame-0 release, all 30 episodes (arm C3r2_tier3_f0)

Written 2026-10-24 ~22:40 CST, before any f0 training. No clean or Tier-1 data was used for anything fixed here.
The opus_v2 test (`opus_easy/PREREG.md`, `PREREG_wave3.md`) is unchanged and keeps governing its own runs.

**Authorization (user, 2026-10-24, verbatim).** "ran new tier3 based on Han's new released dataset? frame 0 is allowed
and we double checked on it. but it might have a potential issue that hand is not that align with the objects what your
though on it?"

Read as: run Tier 3 for all 30 episodes on Han's new release. The user confirms that using the frame-0 object pose is
allowed. The hand-object misalignment is measured and handled below. The main session launches after review and handles
stopping the old opus_v2 runs.

## Question

Does C3r2 train on Han Zhou's f0 reconstruction for all 30 episodes? And how close does the policy get to its own input?
The clean-reference part is LOCAL only: reported, never decided on.

## Data (Tier-3 inputs only)

The root is `data/human_motion_data_tier3_f0`. It is write-once:
- READY sha256 `ee58a62deb3a0bb06248d221e1184de751f32c171410fa2616f057c4b3b2e50b`.
- FILES.sha256 sha256 `f9c94e85…` (576 files). The driver preflight re-verifies every file.
- SCENE_SUPPORT.json `bce8e9eb…`, SUPPORT_FILES.sha256 `5deea2b0…`, manifest `whole_body/soma/track2_manifest.json` `d496584c…`.
- Build and prep reports are in `f0/build/` (build_report.json `8aebef66…`, finalize_report.json, prep_table.json `61bf2db0…`).
  The prep scripts are pinned in `f0/build/build_files.sha256`.

**Inputs.** Byte copies of Han's `tier3_frame0_supports_20261024/reconstructed_data/<seq>/` (30 episodes × 7 files). His folder
is read-only and is never written.
- Only `poses.npy`, `object_pose_xyz_wxyz.npy` and `episode.json` differ from the opus release.
- The human (`soma_params.npz`), camera, ground plane and mesh are byte-identical to opus (and to s1 for the human).
- The object change vs opus_v2 is, per episode, a constant translation of 1.4–13.9 cm plus a constant rotation.

**Frame references.** `frame_references_f0/` holds the f0 `poses.npy` in `released_track2_world`, all frames. It uses
the construction of `frame_references_opus`, and the same code rebuilds the opus file bit for bit. Han's own
`frame_references/` still hold the s1 track and are not used.

**Retarget.** Done by the prep step, not by this driver: `soma_to_g1.py <folder> --save --fps 30 --motion-root <root>` at
nv repo `f2923d61`, v2d env, one plain GPU at a time.
- The object enters three things there: the `object_*` columns, `hand_contact_active` and the frame-0 object ground lift.
  The robot IK never reads it. So all 30 were retargeted again.
- Every parquet passed `scripts/t3f0_rcheck.py`:
  - G1–G8 as `PREREG_wave3.md`, with G8 by the f0 scene class;
  - G9: robot columns equal the opus parquet of the same episode, within 1e-3 (ik_num_iterations within 2);
  - G10: the object track equals the frame chain that places the supports, within 1e-5 m.
- The sha256 values are locked in `f0/retarget/parquet_lock.json` (`bdb83b0e…`, 30 entries). Training and evaluation
  use only locked parquets. A clip without a passing lock is NOT_READY: never launched, reported INCOMPLETE.

## Supports (14 episodes: 01 02 03 06 07 09 11 13 16 17 20 25 27 28)

Built by `scripts/t3f0_supports.py` from Han's `supports/<seq>/support.json` (`T_world_mesh_contact`, `top_box`) and
`support_mesh.glb`. They are written as Cube primitives into `whole_body/reconstructed_stage/<seq>_support.usda` and
trained with `scene.support=true`, the Tier-1/2 setting for these episodes. The 16 other episodes keep the
geometry-free file and `scene.support=false`.
- Tables, desks and the chair seat: a top slab at `T_world_mesh_contact`.
- Stools (ep01 vacuum, ep28 squeegee lean on the bar stool): voxel boxes from the mesh near the object, with the
  object's whole reference path carved out.

Four placement rules depart from the literal "T_world_mesh_contact + top_box". Each is computed the same way from the
episode's own track, without tuning per result, and recorded in `f0/supports/epNN.json`:
1. Han's `top_box` sits at the video-fit pose, so his contact shift was added to it.
2. The top height is refit to the lowest hull point of the anchor rest run. Residuals vs Han's contact height are
   −2.5 to +0.4 cm.
3. If fewer than 95% of the anchor's contact points lie on the top, the footprint is moved sideways by the smallest
   amount that fixes it: ep11 6.1 cm, ep17 3.2 cm.
4. ep01: a settle-verified sideways shift of 2.23 cm (`f0/supports/ep01_lean_search.json`). ep28: the carve-out
   handles the frame-0 squeegee blade lying inside Han's stool mesh.

**These departures were flagged for the user's approval by the prep step.** If they are not approved at launch, the
main session holds the 14 support clips with `HOLD_epNN` files. HOLD changes only launch timing. Reverting a departure
needs a new root (and an addendum), because READY is write-once.

## Hand-object alignment and mitigation (frozen before launch)

**Finding (GT-free).** The frame-0 correction moved the object but not the human. Measured with `scripts/t3f0_align.py`
(`03ea75fc…`), results in `f0/align/` (hashes in `ALIGN_FILES.sha256`, `1ccbd296…`).
- **Contact frames:** the object is moving (≥ 3 cm/s or ≥ 15°/s), AND the hand surface is within 5 cm of it, AND the palm
  moves with the object (its net movement in the object frame over ±6 frames is ≤ max(2 cm, 0.35 × its net world
  movement)). A frame counts if this holds on either track; both tracks are then scored on the same frames.
- **Gap:** the minimum distance from the palm centre and the 5 fingertip joints to the object mesh.
- The median gap rose by 13.7 cm (ep28), 10.4 (ep19), 6.5 (ep02), 5.2 (ep03), 3.5 (ep20) and 2.9 (ep17).
- The retarget's own `hand_contact_active` frame counts fell by 39–85% in these episodes.

**Rule:** an episode is mitigated iff its median gap on the f0 track is > 3.0 cm, or more than 2.0 cm above its opus_v2
median.

**Mitigated:** ep01, 02, 03, 15, 17, 19, 20, 28 (`f0/align/mitigation_f0.json`, sha256
`2ab767eb9028f3adbe9a091628a957e9bdebf21420a8bed2718e2acec6ebda0e`).
- Close to the threshold but not mitigated: ep04 (2.84 cm, +1.96), ep16 (2.98, +1.74), ep13 (2.51, +1.30).
- ep29 has no hand contact.
- Borderline: ep01 qualifies on the 3 cm limit only; ep15 is 0.2 cm over.

**Override (config only):** `termination.palm_position_threshold=0.25` (default 0.15), nothing else.
- It loosens training termination only.
- The evaluation disables termination. SP-SR, MP-SR and ADD-AUC do not use the palm threshold (stock harness).
- Only the reported CHORD SR uses the checkpoint's own palm limit, so CHORD SR of mitigated clips is not comparable. It
  is not part of the rule.

**Not changed:** the object tracks (they are what is scored) and the contact labels (as retargeted on f0).

**Disclosure:** the measure was revised twice before this list was fixed:
1. The closest point anywhere on the hand skin was replaced by the palm and fingertip points (after ep03 only).
2. A rigid-attachment contact test was replaced by the co-motion test above (after a full run), because it found 0–10
   contact frames in episodes where the object is clearly held.

The final list contains every episode the earlier version selected. The rule is generic: it is one threshold on one
GT-free measure, computed identically for all 30 episodes. It names no frame range and uses no clean data.

**Stronger fixes. Described for the main session, NOT part of this test:**
- (A) For the mitigated episodes, take `hand_contact_active` from the opus parquets. Human and robot are identical, so
  the force-closure reward is paid during real carries. This is data-only and needs a new root plus an rcheck exception
  for that column.
- (B) For ep19 and ep28 (and possibly 02, 03, 20), shift the human hand and arm targets by the correction at the hand
  during contact segments, blended over 10–15 frames, then retarget again. This forks the retarget, so it must go into
  the cited commit.

Expectation, not a rule: without (A) most mitigated clips may not reach help zero; without (B) ep19 and ep28 probably
fail.

## Clip set and order

All 30 episodes, launched in **episode-index order 00, 01, …, 29**. No Tier-1 information and no flag affects the order.
Nothing is dropped or substituted automatically.

## Recipe

C3r2 is identical to the raw Tier-2 C3r2 run of each episode (`fc_runs/C3r2_tier2`):
- code/fc_r2 @ `61fa5d3`; the tree digest must equal `opus_easy/build.json`.
- `experiment=g1_recon_body_coord_rsi_r2`, `reset.seed=42`, `training.max_iterations=45000`.
- Experiment defaults: deadline 35156, zero_ramp 3906, help-zero tail 5860.

These keys may differ from the raw runs' saved `config.yaml`:
- task.parquet, output_dir, logging.mode (offline), logging.project, logging.experiment, logging.local_jsonl;
- logging.run_name, vs the Tier-1 config only (a recipe-identity check, never a decision);
- termination.palm_position_threshold, for the 8 mitigated clips only.

`scene.support` must equal the raw run's value: true for the 14 support episodes, false for the 16 others.
`f0/cfgcheck_f0.json` records the hydra comparison for all 30 clips. The training environment sets `T3F0=1`.

**Placement (the v2 fill rule, unchanged):**
- Never GPU 4. No MPS.
- A start needs fewer than 3 trainings of anybody on the GPU, at least 16.5 GB + 20 GB free plus ramp debt, and 2 free
  physical cores of its own NUMA node. NUMA-1 GPUs are tried first.
- Starts are spaced ≥ 180 s apart, under the shared GPU-choice locks (read-only).
- Runs go in `fc_runs/C3r2_tier3_f0/epNN`.

**Evaluation (as PREREG.md):**
- Exactly ONE official evaluation, of `final.json`'s checkpoint, against the clip's own f0 input: world count 1, on its
  training GPU, under the eval-lane locks.
- The evaluated scene is the checkpoint's own (supports resolved from the f0 root).
- Export at 1e-4, redone at 2e-4 only if refused (recorded). A refusal at 2e-4 fails (b).
- No periodic evaluation and no checkpoint selection.

**Failures.** A crash writes a CRASHED marker. There is no automatic relaunch.

## Pass rule (identical to PREREG.md; training and own-input metrics only)

A clip PASSES iff both hold:
- **(a)** the `final.json` reason is help_zero_tail with help_zero_since ≤ 30000. "Strong" (label only) means ≤ 20000.
- **(b)** the official evaluation of `final.json`'s checkpoint against its own input has world_count 1 and SP-SR = MP-SR = 1,
  with the retarget-locked parquet as the evaluator reference.

A clip FAILS if (a) or (b) is false. It is INCOMPLETE after a crash, without a passing lock, or after a technical
evaluation failure plus 1 retry.

**Set verdicts.** Each set gets "C3r2 TRAINS on the f0 reconstruction for these clips" iff all its clips PASS, and "NOT
shown" if any FAILS. Otherwise it is PENDING or INCOMPLETE. Counts are reported per group. The groups, fixed here:
- all (30);
- floor_only (16) and support (14);
- mitigated (8) and not_mitigated (22).

## Reported, never deciding

- Own-input ADD-AUC, MPPE and CHORD SR.
- help_zero_since vs the raw Tier-2 C3r2 run.
- The opus_v2 run of the same episode: result, help_zero_since and post status.
- Training fail rate (rsi/fail_rate_mean, 1000-iteration bins) vs opus_v2 and raw Tier-2.
- LOCAL only (`*_LOCAL_NOT_FOR_UPLOAD`): clean ADD-AUC, SP-SR, MP-SR and MPPE of the policy, via the kit packer
  (`track_2_tier3`, `--frame-references frame_references_f0`). Shown next to Tier-1 C3r2 and raw Tier-2 C3r2.

Clean numbers never order, choose, stop or select anything.

## Use in a submission (fixed here so no clean number can choose)

Every episode uses its **f0** policy (`final.json`'s checkpoint, exported and evaluated as above). The opus_v2 policy of
the same episode is used only when the f0 clip has no evaluated checkpoint (crash, export refused, technical failure)
and the opus_v2 one has. A FAIL by the rule above still has a rollout, so it still uses f0. No per-episode choice
between the two is made from clean or GT-based numbers, including Han's README quality table.

## Flags (report only; from `f0/build/prep_table.json` plus the align step)

- **Han's known issues:**
  - ep01: stool fit IoU 0.61.
  - ep17: cone track about 5 cm into the table, so the table was lowered.
  - ep28: the squeegee ends about 10 cm from the stool.
- **Prep physical checks:**
  - ep01: the end rest is 2.7 cm from the stool and the vacuum tips in the settle test.
  - ep02: the basket rim is up to 1.5 cm inside the top on 8 frames at pickup.
  - ep06: the start rest moves 2.06 cm (limit 2.0).
  - ep07: it slides near the table edge.
  - ep11: the footprint was shifted 6.1 cm; frames 406–467 are hand-held, not a rest.
  - ep13 and ep16: the G1 left and right elbow pass up to 5 cm into the desk on 41 and 29 frames.
  - ep17: the end tilts 5.24° (limit 5).
  - ep25: the unheld end rest falls over.
  - ep28: the track end is 19 cm from the stool and 6 cm below the floor.
- **Align:**
  - mitigated: 01 02 03 15 17 19 20 28;
  - ep19 and ep28: the relaxation is likely insufficient;
  - near the threshold: 04 13 16;
  - ep29: no hand contact.

Han's opus README flags (symmetric flip, depth, failed) do not apply to this release, whose correction includes the
symmetric-flip fix. They are not carried over.

## Grey areas and disclosures (clear with the organisers before any submission)

1. **Frame-0 object pose.** The f0 tracks use the true first-frame object pose.
   - Han's README says only the first-frame object pose was used, no other GT. The constant translation is the
     difference between the frame-0 pose and the mean of the track's static segment.
   - The user confirmed on 2026-10-24 that this is allowed ("frame 0 is allowed and we double checked on it").
   - Han's contact shift for the supports is computed from these tracks.
2. **Supports** come from Han's video fit. His README calls it GT-free (GroundingDINO, SAM2, silhouette IoU fit) and
   uses the task text as input. On top of that come his contact shift and our four placement rules above. The support
   code (Han's `~/Projects/v2d/support/`) and `scripts/t3f0_supports.py` must be in the cited commit.
3. **Items 1–4 of PREREG.md still apply:**
   - Han's s1 alignment used the frame-0 reference human;
   - the released scene files (ground plane, mesh);
   - our frame references;
   - the reconstruction code (Han's pipeline, the opus agents' `solve.py` and the frame-0 correction) must be in the
     cited commit (rule 2.9).
4. **GT-based numbers.** Han's README quality table (GT-based ADD-AUC, evaluation only) and its GT diagnostics were
   read. They chose nothing here. The user chose the release.
5. **The human is unchanged (s1)** while the object moved: the hand-object misalignment above. It is mitigated by
   config only.

## Files

- **Scripts (new; the v2 files are not edited because the running opus_v2 driver executes them):**
  - `scripts/t3f0_driver.sh`, `scripts/t3f0_helper.py`;
  - prep and align scripts, pinned in `f0/build/build_files.sha256` and `f0/align/ALIGN_FILES.sha256`;
  - the sha256 of the driver, the helper and this file in `f0/build_files_f0.sha256`.
- **Outputs:**
  - `f0/{driver.log, heartbeat.log, pids.tsv, waiting.json, ALL_LAUNCHED, eval/epNN/, VERDICT.*, fail_rates.csv}`;
  - `VERDICT_clean_LOCAL_NOT_FOR_UPLOAD.json`;
  - CRASHED_epNN markers;
  - `fc_runs/C3r2_tier3_f0/{epNN, logs/}`.
- **Controls:** STOP: `touch f0/STOP`. HOLD: `touch f0/HOLD_epNN`. Re-attach: `adopt`.
- **The Tier-2 launchers (K3CA, T2fix) do not recognise this driver.** They stay PAUSEd until `f0/ALL_LAUNCHED` exists
  (user 2026-10-24: Tier 3 first).
