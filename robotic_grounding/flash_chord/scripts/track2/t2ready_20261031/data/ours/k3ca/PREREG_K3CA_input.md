# K3CA input (contact-aware K3fix for the 7 K3fix gate failures): input pre-registration

Written 2026-10-24 ~02:35 CST, before any K3CA training. The sha256 of this file and of every file pinned below is in
`PREREG_K3CA_input.sha256.json`.

Authorization: the main session asked "Should I start designing it (the contact-aware filter for the 7 raw-stuck
Tier-2 tasks) now?". The user replied "so they still training? any tier2 we can start to try to improve?", and the
main session treats that as a go to design and build. Training is launched by the main session after review. Its
training pre-registration (recipe, verdict rules) is NOT part of this file.

## Episodes

These 7 Tier-2 episodes fail the K3fix input gates (`tmp_analysis/tier2/k3full/build/gates_all30.json`, PREREG_v3), so
they use their raw-input C3r2_tier2 runs:

| ep | object | K3fix gate failure |
|---|---|---|
| 00 | tall_bar_stool | G4 0.096 |
| 05 | big_black_box | G4 0.059 |
| 15 | wet_floor_sign | G4 0.209 |
| 19 | cane | G4 0.341 |
| 22 | step_ladder | G4 0.191 |
| 26 | skateboard | G4 0.073 |
| 28 | squeegee | G2 194 frames, max 13.7 cm; G4 0.156 |

## Method (`scripts/k3ca_build.py`)

K3CA(t) is the point at weight lam(t) on the geodesic from the K3fix pose to the released raw Tier-2 pose:
- position: (1 - lam) p_K3fix + lam p_raw;
- rotation: R_K3fix exp(lam log(R_K3fix^-1 R_raw)).

**Weights.**
- lam(t) = max over k of lam_req(k) c(|t - k|), where c(d) = (1 + cos(pi d / 16)) / 2 for d <= 15 frames and 0 beyond.
  - 15 frames is the ZUPT v1 bridge constant 2H + 1.
  - Each pull fades in and out over 0.5 s, so lam is continuous.
- lam_req(k) is the smallest value on the grid 0, 0.1, ..., 1 that satisfies both requirements:
  - **contact.** On a raw-contact frame, the blended object's hand-point distance is <= raw distance + 1.5 cm.
    - Raw-contact frame is the G4 definition: the raw object is < 3 cm from the wrist or one of the 5 finger ends of
      either hand.
    - 1.5 cm is half of G4's 3 cm tolerance.
  - **support.** On a visible frame, the blended object's penetration into the floor or support stage is <= raw
    depth + 2.31 cm.
    - Penetration is measured in the G1 training world with the G2 recipe.
    - 2.31 cm is half of the 4.62 cm envelope.
- **Verification loop.** Any frame that still violates a requirement after smoothing gets lam_req = lam + 0.1, capped
  at 1, and lam is recomputed. lam = 1 reproduces raw exactly, so the loop ends.
- Frames with lam = 0 are bit-identical to the K3fix track.

**Constants.**
- 3 cm and 4.62 cm are the gates' own thresholds. 15 frames is an existing ZUPT constant. The two "half"
  margins and the 0.1 grid were fixed by hand before any gate was run on this design.
- No constant was fitted.

**Why each gate holds by construction.**
- **G1:** the blend lies on the K3fix-to-raw geodesic, so per frame |K3CA - raw| <= |K3fix - raw|. K3fix passes G1 on
  all 7 episodes.
- **G2 and G4:** both requirements are enforced at half the gate threshold.
- **G3:** the K3fix smoother is used unchanged. Nothing new is forced, and none of the 7 episodes has a flagged rest run.

**Design history (`tmp_analysis/tier2/k3ca/explore/`):** the sketch I was handed attached the object
rigidly to the noisy hand. One robust relative transform was estimated per raw-contact window, with a 0.5 s offset
fade at the edges and a penetration clamp.
- On the noisy data, the raw hand-to-object relative pose is far from constant inside the windows
  (`build/tracks.json`, `relative_transform_spread_noisy`, max deviation from the window's robust mean).
  - The largest spreads are 54 cm / 116 deg (box, ep05) and 25 cm / 101 deg (skateboard, ep26).
  - Even on the rigid-looking one-hand carries the spread is 9-18 cm / 21-31 deg (step ladder ep22, wet floor sign
    ep15). That is the size of the object random walk plus the body-rotation lever arm.
- Rigid attachment therefore failed G1 on all 7 episodes: 64 to 392 frames beyond the envelope, 9-53 cm off the raw
  track (`explore/proto2_out.json`, variant `Aconst`).
- Hand-frame local smoothing (half-widths 7 / 15 / 30 / 60 frames) failed G1 on 1 / 3 / 5 / 5 of the 7 episodes.
  - At half-width 7 it raised the RMS position acceleration to 0.7-2.8x raw (median about 1.6x), because wrist
    rotation is amplified by the lever to the object origin.
- The adopted blend is the smallest change to K3fix that keeps the raw hand contact.
- Choices were made on gate numbers and smoothness. They are disclosed here as design iterations against the
  gates, so the gates are not an independent test of K3CA.

## Results (`build/gates.txt`, `build/tracks.json`)

**Gates.**
- They use the same formulas as `gates_all30` (k3fix_build.py gate_work, sha256 364d27ed).
- Replication check: applied to the K3fix and frozen-K3 tracks, they reproduce `gates_all30.json` for all 7 episodes
  (14/14 checks).
- **K3CA passes G1-G4 on all 7.** G4 share is 0.000 on every episode.
  - ep28's G2 goes from 194 frames (max 13.7 cm) to 0 frames (max 2.17 cm).
  - ep22's G2 max goes from 4.43 cm to 2.28 cm.

**How far K3CA moves from K3fix.**

| ep | share of frames with lam > 0 | mean lam | max change vs K3fix | mean change vs K3fix | frames requiring contact / support |
|---|---|---|---|---|---|
| 00 | 0.13 | 0.045 | 2.12 cm / 22.8 deg | 0.11 cm / 1.2 deg | 14 / 0 |
| 05 | 0.30 | 0.118 | 1.09 cm / 25.0 deg | 0.09 cm / 3.0 deg | 71 / 0 |
| 15 | 0.25 | 0.083 | 1.87 cm / 14.5 deg | 0.19 cm / 1.4 deg | 79 / 0 |
| 19 | 0.45 | 0.139 | 0.90 cm / 12.8 deg | 0.14 cm / 1.5 deg | 90 / 0 |
| 22 | 0.37 | 0.104 | 1.55 cm / 13.8 deg | 0.21 cm / 2.0 deg | 85 / 51 |
| 26 | 0.29 | 0.084 | 1.29 cm / 24.1 deg | 0.12 cm / 2.4 deg | 51 / 0 |
| 28 | 0.40 | 0.227 | 0.89 cm / 29.0 deg | 0.21 cm / 5.9 deg | 20 / 240 |

## Input rule (fixed now)

- An episode uses the K3CA input iff:
  - it failed the K3fix gates in `gates_all30.json`; and
  - K3CA passes G1-G4 in `build/gates.json`.
- Result: **00 05 15 19 22 26 28 all use K3CA**; none falls back to raw.
- The other 23 episodes are not affected. They keep the PREREG_v3 rule.
- Nothing reported later can change this rule, the constants, the episode set or the data root.

## Data root

- **`data/human_motion_data_tier2_k3ca`**, READY sha256 `e4920e98...`.
- The 7 K3CA parquets were built by `scripts/k3ca_build_data.py` from `build/k3ca_tracks.npz` (sha256 `ad2044df...`).
  - The build path is the K3fix root's e2 part (a) maths. All 30 K3fix-root parquets were regenerated byte-identical
    with it.
- The other 23 parquets, all ground planes, support stages and the manifest are byte copies of
  data/human_motion_data_tier2_k3fix.
- Frame references are in `tmp_analysis/tier2/k3ca/frame_refs_k3ca`.
- Kit packer: 30/30 accepted against the own frame references, 30/30 rejected against raw. The 7 changed episodes are
  rejected against the K3fix references and the other 23 accepted.
- Trainer loader (code/fc_r2 @ 61fa5d3 `load_reference` + support resolution): 30/30 ok on the new root and on the raw
  control. The object track differs from the K3fix root exactly on the 7 episodes (`build/loader_check.json`).

## Disclosures

- K3CA builds on K3 and K3fix.
- The design iterated against the gates (see Design history).
