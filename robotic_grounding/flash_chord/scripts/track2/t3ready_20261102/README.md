# Track 2 Tier 3 submission record: FINAL_r (2026-11-02)

This folder is the code and record behind our Track 2, Tier 3 submission file ("bring your own reconstruction"). The
policies are G1 + Dex3 policies trained with FlashCHORD on our team's own video reconstruction, one policy per
episode, 30 episodes.

It is one commit on top of `41403d3` (branch `dev.haoran.c3r2`). The commit adds this folder and a pointer file at the
repo root, `TIER3_SUBMISSION.md`. **It leaves the trainer untouched:**
- `robotic_grounding/flash_chord/src` is byte-identical to `41403d3`'s;
- every code file of `robotic_grounding/flash_chord` is byte-identical to the FlashCHORD tree that trained and exported
  every policy (fc_r2 @ `61fa5d3`);
- `robotic_grounding/scripts/retarget/soma_to_g1.py` is the retarget that made every training input.

See `verification/code_identity.txt`.

Contents:
1. The file
2. How each episode was produced
3. Reproduce
4. Folder layout
5. Verification
6. What is still missing
7. Disclosures
8. Grey areas for the organisers
9. Kit and runtime

---

## 1. The file

`FINAL_r` is the full 30-episode file for `track_2_tier3`.
- **Built:** 2026-11-01 19:45:06 CST by `evaluation/f0_retry/t3final_r_build.py` (sha256 `29dc1933…`).
- **Builder pinned:** the builder's hash is named in `evaluation/t3_f0_final/AMENDMENT_2.md` (sha256 `8b1e3a2d…`), which
  was written before the build.
- **Lock:** `evaluation/t3_f0_final/FINAL_r.sha256`.

| | value |
|---|---|
| packed CSV, placeholder commit URL (not an upload file) | sha256 `bd73e0e94e168d055ec200cf74eeec9cff945345e9d523f589b6587183f23d52` |
| pose columns (everything except the code commit URL) | sha256 `82fef2c7d09326191098df29567fcc6d0b51cb8523ad70331b60eb4535c36303` |
| rows / episodes | 22,989 / 30 |
| input-reference residual (packer) | 1.44e-07 m / 1.46e-07 rad |
| uploader `validate_submission()` (offline) | pass, no provenance problem |

**The upload file is a repack.** It is the same composition repacked with this commit's URL:

```
conda_envs/flashchord/bin/python -I -B tmp_analysis/tier3/f0_retry/t3final_r_build.py repack \
  --commit-url https://github.com/Beat-in-our-hearts/nv-v2d-challenge-codebase/commit/<this commit>
```

- It writes `fc_runs/submissions/t3_f0_final/upload_r/<sha7>/t3_final_r_closest_<sha7>.csv`.
- It refuses unless the pose columns equal `82fef2c7…`. Only `code_commit_url` differs from FINAL_r.
- Nothing uploads automatically.

**Own-input evaluator numbers.** These come from the official evaluator, scoring each packed rollout against its own
Tier-3 training input (`evaluation/t3_f0_final/pack/FINAL_r/choices.md`). They are not leaderboard numbers.

| | mean ADD-AUC | SP-SR = 1 | MP-SR = 1 | mean MPPE |
|---|---|---|---|---|
| FINAL_r (this file) | 0.6760 | 22 / 30 | 21 / 30 | 16.39 cm |
| round 1, FINAL_f0 (every episode on its f0 item) | 0.5935 | 17 / 30 | 16 / 30 | 24.57 cm |

## 2. How each episode was produced

**Input: Han Zhou's frame-0 release** (`han_zhou_reconstruction/`). Han is on our team. The release builds on:
- **s1:** human, camera and object reconstructed from the monocular exo video. The human and camera are kept unchanged
  in every later release.
- **opus_v2:** each episode's object track re-estimated by a Claude Code (Opus) agent.
- **frame-0 correction** (`opus_v2_frame0_f3_c2_shift`), which uses the true first-frame object pose.
- **Supports:** video-fit furniture for the 14 episodes where the object is placed on a table, desk, chair or stool.

**Data prep** (`data_prep/`, README_tier3.md):
- byte copies of the inputs into a write-once data root;
- G1 retarget with `soma_to_g1.py` (this commit), gated by G1–G10 and locked by sha256;
- support scenes from Han's fits plus four GT-free placement rules;
- a GT-free hand-object gap check that sets `termination.palm_position_threshold=0.25` (training only) for 8 episodes;
- frame references = Han's f0 object track in `released_track2_world`.

**f0 training** (all 30; `evaluation/f0/PREREG_f0.md`, sha256 `82702d85…`):
- `scripts/train_rl.py experiment=g1_recon_body_coord_rsi_r2 reset.seed=42 training.max_iterations=45000`;
- `scene.support` as the episode;
- the help-zero early stop of the experiment: deadline 35156, ramp 3906, tail 5860;
- the evaluated checkpoint is `final.json`'s, with no checkpoint selection.

**Retry, C90 continuation** (`evaluation/f0_retry/PREREG_v3.md`, sha256 `e62c39b2…`, and PREREG_retry `8b40cff0…`):
- **Candidates:** the 14 episodes whose f0 policy had own-input SP-SR = 0 or MP-SR = 0, chosen by a fixed rule and
  frozen: 01 02 04 05 11 13 14 16 17 19 21 23 25 28.
- **Training:** each resumed from its f0 `model_44922` + `rsi_44922` to 90000 iterations, with a second RSI help cycle
  (`reset.start_update=44922`, no early stop) and its rule-computed termination overrides (`f0_retry/recipes/`).
- **Acceptance**, on own-input numbers only (MPPE is not part of it). The retry REPLACES the f0 final iff either:
  - (i) the retry is SP = MP = 1 in at least 4 of its 5 evaluations and in more of them than the f0 final; or
  - (ii) the retry's mean own-input AUC is at least f0 + 0.01, with no fewer SP and MP successes.
- **Outcome:** REPLACE for 02 04 05 13 14 16 17 19 21 23 25. KEEP_F0 for 01 11 28.

**Evaluation and pick** (`evaluation/t3_f0_final/PREREG.md`, sha256 `e4e3a5da…`, AMENDMENT_1 `e2a43438…`, AMENDMENT_2):
- 5 official own-input evaluations per item;
- run through `scripts/track2/eval_all.py` → kit `evaluate_policy.py` on the stock runtime;
- world count 1, export tolerance 1e-4;
- the packed rollout is the run whose own-input ADD-AUC is closest to the 5-run mean: exact rational arithmetic, ties
  to the lower run, so not a best-of-5.

**Pack:** the kit packer, unchanged, with each episode's own frame reference.

| ep | object | item | support | palm 0.25 | retry termination overrides | trained to | final reason | run | own AUC | SP | MP | MPPE cm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 00 | tall_bar_stool | f0 | - | - | - | 18910 | help_zero_tail | 4 | 0.9657 | 1 | 1 | 0.53 |
| 01 | vacuum | f0 | yes | yes | - | 44922 | help_zero_tail | 3 | 0.1990 | 0 | 0 | 32.82 |
| 02 | blue_easter_basket | retry | yes | yes | object=0.21, palm=0.25 | 90000 | max_iterations | 5 | 0.3239 | 0 | 0 | 13.96 |
| 03 | contigo_coffee_mug | f0 | yes | yes | - | 19184 | help_zero_tail | 4 | 0.9864 | 1 | 1 | 0.62 |
| 04 | baseball_bat | retry | - | - | object=0.20, palm=0.25 | 90000 | max_iterations | 2 | 0.1842 | 0 | 0 | 126.01 |
| 05 | big_black_box | retry | - | - | palm=0.25 | 90000 | max_iterations | 5 | 0.5548 | 1 | 1 | 2.23 |
| 06 | monitor | f0 | yes | - | - | 44922 | help_zero_tail | 3 | 0.8498 | 1 | 1 | 1.85 |
| 07 | iron | f0 | yes | - | - | 20779 | help_zero_tail | 1 | 0.9867 | 1 | 1 | 0.69 |
| 08 | skinny_wood_chair | f0 | - | - | - | 17451 | help_zero_tail | 1 | 0.9628 | 1 | 1 | 0.96 |
| 09 | big_red_bowl | f0 | yes | - | - | 44922 | help_zero_tail | 5 | 0.9227 | 1 | 1 | 1.32 |
| 10 | white_desk | f0 | - | - | - | 25976 | help_zero_tail | 4 | 0.5521 | 1 | 1 | 1.74 |
| 11 | light_blue_book | f0 | yes | - | - | 44922 | help_zero_tail | 3 | 0.1004 | 0 | 0 | 73.29 |
| 12 | black_platform | f0 | - | - | - | 29267 | help_zero_tail | 1 | 0.7042 | 1 | 1 | 2.24 |
| 13 | snack_box | retry | yes | - | object=0.25, palm=0.25 | 90000 | max_iterations | 2 | 0.2099 | 0 | 0 | 61.36 |
| 14 | round_table | retry | - | - | palm=0.25 | 90000 | max_iterations | 2 | 0.5889 | 1 | 0 | 9.51 |
| 15 | wet_floor_sign | f0 | - | yes | - | 39300 | help_zero_tail | 2 | 0.8897 | 1 | 1 | 0.90 |
| 16 | spray_bottle | retry | yes | - | palm=0.25 | 90000 | max_iterations | 5 | 0.5929 | 0 | 0 | 9.23 |
| 17 | traffic_cone | retry | yes | yes | palm=0.25 | 90000 | max_iterations | 1 | 0.9432 | 1 | 1 | 0.97 |
| 18 | blue_trash_can | f0 | - | - | - | 21486 | help_zero_tail | 2 | 0.9718 | 1 | 1 | 0.61 |
| 19 | cane | retry | - | yes | object=0.20, palm=0.25 | 90000 | max_iterations | 1 | 0.1568 | 0 | 0 | 80.95 |
| 20 | rice_cooker | f0 | yes | yes | - | 18433 | help_zero_tail | 1 | 0.9174 | 1 | 1 | 0.89 |
| 21 | tall_bar_stool | retry | - | - | palm=0.25 | 90000 | max_iterations | 5 | 0.5060 | 1 | 1 | 2.80 |
| 22 | step_ladder | f0 | - | - | - | 20827 | help_zero_tail | 1 | 0.9589 | 1 | 1 | 0.65 |
| 23 | toy_stroller | retry | - | - | palm=0.25 | 90000 | max_iterations | 5 | 0.9782 | 1 | 1 | 0.62 |
| 24 | watering_can | f0 | - | - | - | 20449 | help_zero_tail | 2 | 0.9285 | 1 | 1 | 1.01 |
| 25 | cyan_water_bottle | retry | yes | - | object=0.20, palm=0.25 | 90000 | max_iterations | 5 | 0.5835 | 1 | 1 | 3.47 |
| 26 | skateboard | f0 | - | - | - | 22863 | help_zero_tail | 1 | 0.9670 | 1 | 1 | 0.45 |
| 27 | woven_basket | f0 | yes | - | - | 34196 | help_zero_tail | 1 | 0.8723 | 1 | 1 | 1.12 |
| 28 | squeegee | f0 | yes | yes | - | 44922 | help_zero_tail | 3 | 0.1403 | 0 | 0 | 56.67 |
| 29 | rolling_luggage | f0 | - | - | - | 28248 | help_zero_tail | 5 | 0.7810 | 1 | 1 | 2.34 |

How to read the table:
- **"palm 0.25"** is the f0 mitigation for the hand-object gap (`data_prep/f0/align/mitigation_f0.json`). Every retry
  recipe also uses palm 0.25.
- **"own AUC / SP / MP / MPPE"** are the packed run's own-input evaluator values.
- **What each episode used** (checkpoints, sha256s, stages, overrides, evaluated files, recorded commands) is in
  `set/set_items.json`.

## 3. Reproduce

The drivers name absolute paths under the training server's work root, `/root/autodl-tmp/haoran.zhang` (`<ME>`).

```bash
B=robotic_grounding/flash_chord/scripts/track2/t3ready_20261102
python3 $B/tools/restore_layout_t3.py --me /root/autodl-tmp/haoran.zhang   # files back at their run paths (--rewrite for another root)

# B. data: Han's release -> data root, retarget, supports, mitigation   (data_prep/README_tier3.md, steps 0 and A-G)

# training, export, evaluation of every episode, exactly as recorded:
python3 $B/tools/print_commands_t3.py --recorded            # per episode: train stage(s), retry fork, export + eval lines

# the drivers that ran them (each refuses unless its pinned PREREG / hash files match):
bash scripts/t3f0_driver.sh run                              # f0: 30 trainings + post (PREREG_f0)
bash fc_runs/submissions/t3_f0_final/launch.sh               # 5x own-input evaluation, selection, FINAL_f0 pack (PREREG.md)
bash tmp_analysis/tier3/f0_retry/launch_v3.sh                # retry wave: 14 C90 continuations + 5+5 evaluations (PREREG_v3; needs RETRY_GO)
conda_envs/flashchord/bin/python -I -B tmp_analysis/tier3/f0_retry/t3final_r_build.py build    # FINAL_r (AMENDMENT_2)
conda_envs/flashchord/bin/python -I -B tmp_analysis/tier3/f0_retry/t3final_r_build.py repack --commit-url <URL>

# checks of this record (CPU only):
conda_envs/flashchord/bin/python -I -B $B/tools/check_set_t3.py --files --hash
bash $B/tools/code_identity_t3.sh . HEAD <fc_r2 clone> <nv clone>
```

**Retraining does not reproduce the bytes.** Training and evaluation run on GPU and are not bit-deterministic; the
official evaluation path has no selectable seed. A rerun therefore reproduces the procedure and the policy class, not
the bytes.

**The exact artifacts are kept on the training server**, at the paths in `set/set_items.json`:
- every training checkpoint;
- the evaluated (exported) files, named in the CSV's `checkpoint_sha256`;
- the 5 rollouts per item.

## 4. Folder layout

| path | what |
|---|---|
| `README.md` | this file |
| `SOURCES.tsv` | every copied file: its path here, its original path on the training server (`<ME>`, `<HAN>`), sha256, bytes |
| `SHA256SUMS` | sha256 of every file in this folder |
| `data_prep/` | part B: Han's release → training inputs (scripts, pins, records, `README_tier3.md`) |
| `han_zhou_reconstruction/` | part C: Han Zhou's reconstruction code and documents as available on the server, `COPIES_t3ready.tsv`, and the **MISSING** list |
| `evaluation/f0/` | PREREG_f0 (+ sha256), the f0 VERDICT (own-input and training statistics), the 30 f0 post records |
| `evaluation/t3_f0_final/` | the 5x evaluation and selection driver and kit helper, PREREG.md (+ sha256), AMENDMENT_1/2 (+ sha256), item / selection / run records, `own_input_runs.csv`, `selection.csv`, `status.md`; `pack/FINAL_r/` (`build.json`, `choices.*`, `file.json`, `expect.json`, `validate.*`, `pack_status.json`, `FILES.sha256`, `inputs/`, `results/*.json`); `pack/FINAL/` records |
| `evaluation/f0_retry/` | PREREG_retry (v2; v1 under `r3fix/v1/`), PREREG_v3 (+ sha256 files), `RETRY_FILES*.sha256`, RETRY_GO / NOGO / templates, `t3retry.py`, `t3retry_v3.py`, `launch*.sh`, `v3fix/` and `r3fix/` (code and tests), `recipes/` + `recipes.sha256`, `cfgcheck/`, `dryrun/`, the 14 post records, `VERDICT_retry.json`, `SUBMISSION_CHOICE.json`, `t3final_r_build.py` |
| `set/` | `set_items.json` (the 30 items), the runs' saved Hydra records, the resolved config stored in every checkpoint, `final.json` / `fork.json`, export reports |
| `tools/` | `check_set_t3.py`, `code_identity_t3.sh`, `print_commands_t3.py`, `restore_layout_t3.py`, `stage_t3ready.py` (how this folder was built) |
| `verification/` | the reports of section 5 |

**FINAL_r's `FILES.sha256` is committed verbatim, so the lock chain still verifies.** It lists two LOCAL files that are
not committed: the placeholder CSV and the packer log, which also holds the packer's own clean-reference cross-check.

**Never committed:** files whose name marks clean-reference or ground-truth material (`*LOCAL_NOT_FOR_UPLOAD*`,
`*clean*`, `VERDICT_clean*`, `summary_LOCAL*`, `report_LOCAL*`, `s4a_*`, `s4b_*`, `clean_stat*`, `*GT*`, `gt_eval*`).

## 5. Verification (CPU only, before the commit)

- **`verification/check_set_report.json`** (`tools/check_set_t3.py --files --hash`): **ALL OK**.
  - 41 / 41 training stages (30 f0 + 11 retry) compose with this tree's Hydra configs. The result equals each run's
    saved `config.yaml` key by key. The overrides equal the saved `overrides.yaml`.
  - 41 / 41 equal the resolved config stored in the stage's checkpoint, both as committed and as read from the
    checkpoint file. No new or missing key.
  - 11 / 11 retry stages resume from the f0 checkpoint (sha256 equal) with the recipe's overrides.
  - 30 / 30 evaluated files name their training checkpoint as export source.
  - 30 / 30 packed rollouts name the evaluated file and the locked input.
  - Every checkpoint and evaluated file matches its sha256 pin.
- **`verification/code_identity.txt`:**
  - `robotic_grounding/flash_chord/src` equals `41403d3`;
  - nothing outside this folder and `TIER3_SUBMISSION.md` changed;
  - all 335 tracked files of fc_r2 @ `61fa5d3` have the same blob here, except 8 Git-LFS gif media under `docs/media/`;
  - the 583 trainer asset files here equal fc_r2's working-tree assets: 559 Git-LFS objects by their sha256 oid, plus
    24 plain files;
  - `soma_to_g1.py` equals nv `f2923d6`;
  - `eval_all.py` and `export.py` match the PREREG pins.
- **`verification/hygiene_scan.json`:**
  - no forbidden file name, no Git-LFS or binary file, no credential or proxy setting;
  - lines naming clean / GT / Tier 1 are listed by count for review. They are rule text, guards and file names, not
    numbers.
- **Repack (after the commit; not in it):** `t3final_r_build.py repack` with this commit's URL, the uploader's offline
  validation for `track_2_tier3`, and pose columns equal to `82fef2c7…`.

## 6. What is still missing

`han_zhou_reconstruction/README.md` lists, under MISSING, the parts of Han's reconstruction that exist only on his
machine:
- the s1 SOMA-X human reconstruction and world-alignment pipeline;
- the opus agent harness, prompt and input packing, plus the per-episode helper modules that 19 of the 30 `solve.py`
  import;
- the frame-0 correction code (`f3_c2_shift`);
- the support fit code (GroundingDINO + SAM2 + silhouette IoU) and its contact shift.

They are needed for rule 2.9.a and must be added by a later commit.

## 7. Disclosures

These are summaries. The full texts are in the cited files.

**PREREG_v3 section 9** (`evaluation/f0_retry/PREREG_v3.md`):
- **D1.** Clean-scored diagnostics of all 30 f0 episodes exist (LOCAL files). The operator saw some of them. The builder
  of PREREG_v3 opened none.
- **D2.** Earlier clean-scored pilots exist: s4a / s4b, train90k, and the ep14 agentT1 90k pilot. The C90 cap of 90000
  is justified in the files only by the RSI help-cycle arithmetic, but an influence of the earlier 90k pilot cannot be
  excluded.
- **D3.** Han's `tier3_ep13_opusmax_20261030` (selected with ground truth) is known and not used. ep13 runs on the
  locked f0 input.
- **D4.** GT-blind audit agents saw Han's README GT quality table (3 aggregate values). They were not recorded and are
  not used.
- **D5.** Raw Tier-2 numbers were visible in some f0 records.
- **D6.** `candidates()` reads only post status and own-input SP / MP from `f0/VERDICT.json`.
- **D7.** Tier-1 / Tier-2 knowledge was present in context and was not used.
- **D8.** The operational rules (GPU list and cap; one relaunch for a trainer that never trains) were written after
  viewing Tier-3 own-input diagnostics.
- **D9.** The builder opened only Tier-3 own-input material.
- **D10.** A paused second builder instance existed and changed nothing.

**AMENDMENT_2 section 8** (`evaluation/t3_f0_final/AMENDMENT_2.md`):
- **A1.** The author saw only Tier-3 own-input numbers.
- **A2.** No file whose name marks clean or GT material was opened. No Tier-1 / Tier-2 data or results were read.
- **A3.** The author's context held project notes with Tier-1 / Tier-2 aggregates and some Tier-3 clean aggregates
  (round-1 FINAL_f0's offline score, some opus_v2-era clips). None was used. The builder has no free choice.
- **A4.** The builder's selftests ran before AMENDMENT_2 was written, as test runs that built no FINAL_r.

**Also disclosed:**
- **The clean score after the lock.** AMENDMENT_2 section 6 allows a LOCAL, report-only clean-reference score of FINAL_r
  and FINAL_f0, computed after FINAL_r was hash-locked. It was computed. It changes no decision: the file choice and the
  Final mark are fixed by AMENDMENT_1 (e), never by a clean or board score. That clause was added by the builder agent
  without the user's explicit approval.
- **What the submission note must say** (AMENDMENT_1 (e)):
  - the retry items are 02 04 05 13 14 16 17 19 21 23 25;
  - the other 19 episodes are f0 items;
  - no episode uses an opus_v2 fallback.
- **Grey areas from earlier PREREGs** (PREREG_f0, "Grey areas"; t3_f0_final PREREG section 6) are covered in section 8.

## 8. Grey areas for the organisers (please confirm)

1. **True first-frame object pose.** Han's frame-0 correction uses the true first-frame object pose: symmetric flip,
   orientation anchor, and constant translation = frame-0 pose minus static-segment mean. Han's README states that no
   other ground truth is used. The user confirmed on 2026-10-24 that this is allowed (FAQ). His support contact shift
   is computed from these tracks.
2. **Han's s1 frame-0 human alignment.** s1 places the monocular reconstruction in the official world with one rigid
   transform, fitted to the **frame-0 reference human**, and also estimates the camera principal point from it.
   - Han's s1 README says the organisers allow the frame-0 reference human for the coordinate frame. The
     principal-point use is not yet confirmed.
   - 10 of the 30 opus agent reports also mention the frame-0 reference human, for example as a floor-height or scale
     cue: ep00 01 07 09 15 20 22 23 24 29.
3. **Video-fit supports.** For the 14 placement episodes, Han fits the scanned furniture mesh to the video with
   GroundingDINO + SAM2 + a silhouette-IoU fit. The task text names the furniture. A vertical contact shift (−5 to
   +2 cm) makes his track rest on it. On top, we apply four GT-free placement rules (PREREG_f0 "Supports").
   - The meshes are the dataset's own meshes of the same furniture.
   - The released Tier-1 / Tier-2 support geometry is not used. The 16 floor-only episodes use released stage files
     that contain no geometry.
4. **LLM-agent reconstruction and its reproducibility.** The opus_v2 object tracks were produced per episode by a Claude
   Code (Opus) agent in a sandbox, starting from s1.
   - The agents' `solve.py`, `TOOLS.md` and `REPORT.md` are committed.
   - The harness, prompt, packed inputs and 19 episodes' helper modules are not yet committed (section 6).
   - Re-running an agent would not reproduce the same `solve.py`. Re-running the committed `solve.py` with its inputs
     and helpers is the reproducible path.
5. **Also stated:**
   - the human is s1's, unchanged while the object moved; the mitigation is by training config only (palm threshold
     0.25 for 8 episodes);
   - the frame references are ours, built from Han's f0 track;
   - the scene files (ground plane, mesh) are the released ones;
   - Han's GT-based README quality tables were read but chose nothing.

## 9. Kit and runtime (unchanged, pinned)

- **Kit:**
  - KIT_MANIFEST `398947a7…`
  - `flash_chord/scripts/evaluate_policy.py` `160d24f5…`, the same file as the stock runtime's
  - `tools/pack_submission.py` `47a45895…`
  - `submit/v2d_submit.py` `fe5cb1dc…`
  - `config/tracks.json` `5799f424…`
- **Stock runtime:** video_to_data @ `8c778f9`.
- **Training and export:** fc_r2 @ `61fa5d3`, the same code files as `robotic_grounding/flash_chord` here.
  `scripts/track2/eval_all.py` `65729592…`; `coordinate/export.py` `34e3d7f3…`.
- **Retarget:** `robotic_grounding/scripts/retarget/soma_to_g1.py`, blob `8207b163…` (= nv `f2923d6`).
