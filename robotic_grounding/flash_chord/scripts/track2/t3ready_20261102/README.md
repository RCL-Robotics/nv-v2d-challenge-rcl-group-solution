# Track 2 Tier 3 submission record: FINAL_r (2026-11-02)

This folder is the code and record behind our Track 2, Tier 3 submission file ("bring your own reconstruction"). The
policies are G1 + Dex3 policies trained with FlashCHORD on our team's own video reconstruction, one policy per
episode, 30 episodes.

**History of this record:**
1. **`232e6b2`** (branch `dev.haoran.t3ready_20261102`): one commit on top of `41403d3` (branch `dev.haoran.c3r2`). It
   adds this folder and a pointer file at the repo root, `TIER3_SUBMISSION.md`.
2. **Merge `ab0d24c`** (2026-11-02). It merges Han Zhou's branch `dev.hanzhou.hoi_estimation` (head `48c217a`), which
   adds his reconstruction code under `hoi_estimation/` at the repo root.
   - The two sides change disjoint paths, so there were no conflicts.
   - The merged tree is `232e6b2`'s tree plus Han's `hoi_estimation/` subtree, unchanged.
3. **The child of that merge** (branch `dev.haoran.t3final_20261102`) updates this README, `TIER3_SUBMISSION.md` and
   `han_zhou_reconstruction/README.md`, adds `verification/merge_t3final/`, and refreshes `SHA256SUMS`.
4. **Han's later commits `6c10a22` … `c0dbf5b`** (on `dev.hanzhou.hoi_estimation`: 12 commits on top of item 3). They
   change only `hoi_estimation/`: 735 files added, 111 modified, none deleted.
   - **Added:**
     - the reproducibility material in `reproducibility/`: runbooks, the end-to-end driver `run_all_tier3.sh` and the
       environment pins;
     - `hoi_extraction/scripts/export_reference.py` and `support/fit_20261023T2028Z.py`;
     - Han's reproduction report and agent workflow notes;
     - the complete records of the 30 release agent runs (`agent_records/CL2*`).
   - **Modified:** 111 scripts and docs.
     - Built-in paths become `V2D_ROOT` and similar variables.
     - An opt-in episode subset (`V2D_EPISODES`) is added.
     - `diag_scores.json` becomes optional where it is only printed.
     - Han also edited the support overview images (`support/overlay.py`, `sheet.py`) and his README.
     - `verification/han_followup/han_edits_after_48c217a.tsv` classifies the edit of each file.
   - **The code as it ran is `48c217a`, an ancestor of this commit.** Han is asked to confirm that his later edits to
     executed scripts change only paths and these opt-in switches (section 6, still open item 5).
5. **This commit** (child of `c0dbf5b`). It changes these files:
   - this README, `TIER3_SUBMISSION.md`, `han_zhou_reconstruction/README.md` and `data_prep/README_tier3.md`;
   - the fix of `tools/restore_layout_t3.py` (section 3);
   - the new folder `verification/han_followup/`;
   - `SHA256SUMS`, refreshed.

   Nothing that produced or packed the submission file changes.

The submission file is unchanged. Only its `code_commit_url` points at the new commit.

**The trainer is untouched:**
- `robotic_grounding/flash_chord/src` is byte-identical to `41403d3`'s (tree `b8000f82`), both before and after the
  merge;
- every code file of `robotic_grounding/flash_chord` is byte-identical to the FlashCHORD tree that trained and exported
  every policy (fc_r2 @ `61fa5d3`);
- `robotic_grounding/scripts/retarget/soma_to_g1.py` is the retarget that made every training input.

See `verification/code_identity.txt` (for `232e6b2`) and `verification/merge_t3final/code_identity.txt` (for the
merge).

**In a commit that also merges the other tiers' records**, `robotic_grounding/flash_chord/src` can hold another tier's
version. In that case `tools/code_identity_t3.sh` item 1 reports DIFFERENT. The exact Tier-3 trainer is `41403d3`, an
ancestor, which has the same `src` as `f2923d6`. Check it out with `git worktree add <dir> 41403d3`.

Contents:
1. The file
2. How each episode was produced
3. Reproduce
4. Folder layout
5. Verification
6. Han's reconstruction code: where it is, and what is still open
7. Disclosures
8. Grey areas and the organisers' answers
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

**Input: Han Zhou's frame-0 release.** Han is on our team. The release documents are in `han_zhou_reconstruction/`; his
code is in `hoi_estimation/` at the repo root (section 6). The release builds on:
- **s1:** human, camera and object reconstructed from the monocular exo video (`hoi_estimation/hoi_extraction/`). The
  human and camera are kept unchanged in every later release.
- **opus_v2:** each episode's object track re-estimated by a Claude Code (Opus) agent (`hoi_estimation/agent/`, runs in
  `hoi_estimation/agent_records/`).
- **frame-0 correction** (`opus_v2_frame0_f3_c2_shift`), which uses the true first-frame object pose
  (`hoi_estimation/tier2_refine/t3_frame0.py`).
- **Supports:** video-fit furniture for the 14 episodes where the object is placed on a table, desk, chair or stool
  (`hoi_estimation/support/`).

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
python3 $B/tools/restore_layout_t3.py --me <root> [--han <root of Han's releases>] [--rewrite]
#   puts the files back at their run paths. The committed write-once data-root records go to <root>/t3ready_expected/
#   (expected outputs to compare against, not inputs). It prints what it does not restore: the retarget checkout
#   <root>/code/nv-v2d-challenge-codebase at f2923d61 (an ancestor of this commit: git worktree add ... f2923d61...)
#   and the conda envs.

# A. reconstruction (Han Zhou): hoi_estimation/reproducibility/README.md, then
#    s1_runbook.md (CARI4D -> E2 -> P1 -> M1 -> first-frame ICP/PnP -> FF2 -> JC2s1 -> DB -> s1 release) and
#    tier3_runbook.md (agent inputs -> opus_v2 batches -> frame-0 correction -> supports -> combined release);
#    end to end: hoi_estimation/reproducibility/run_all_tier3.sh.
#    Images, weights and intermediates: reproducibility/README.md section 0 (on Han's storage; see section 6 here).
#    GT-blind run of the frame-0 step: t3_frame0.py scores every output against all frames of ref_poses after writing
#    it (the code from 'res = {}' on). Cut that part, feed ref_poses holding frame 0 only, and give
#    build_release_combined.py no diag_scores.json (since c0dbf5b it is optional; it is only printed).

# B. data: Han's release -> data root, retarget, supports, mitigation   (data_prep/README_tier3.md: step 0, the opus_v2
#    root with its retarget and rcheck, first; then steps A-G)

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
bash $B/tools/code_identity_t3.sh . HEAD <fc_r2 clone> <nv clone>   # on a merged commit, item 1 shows the src difference (see 'The trainer is untouched')
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
| `han_zhou_reconstruction/` | part C: Han Zhou's release documents and the s1 code he shipped with s1, copied from the server (`COPIES_t3ready.tsv`), and the map from each formerly missing piece to its place in `hoi_estimation/` |
| `../../../../../hoi_estimation/` (repo root) | Han Zhou's full reconstruction code and agent run records, merged from `dev.hanzhou.hoi_estimation` (`48c217a`) |
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

## 5. Verification (CPU only)

**For `232e6b2`, before that commit:**

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

**After the merge with Han's branch (`verification/merge_t3final/`, run on the merge commit `ab0d24c`):**
- **`check_set.log` / `check_set_report.json`:** `check_set_t3.py --files --hash` gives **ALL OK** again: 41 / 41 stages
  compose, B 41, D 41, R 11, E 30, and 30 rollouts name the evaluated file.
  - It was run on the committed tree: the merge commit's `src/flash_chord/configs` and this folder, written out from
    git objects.
  - 41 checkpoint and 30 evaluated-file sha256 pins match.
  - The README child commit changes none of the files the check reads.
- **`code_identity.txt`:**
  - item 1: `src` equals `41403d3`;
  - item 2: the only paths changed outside this folder and `TIER3_SUBMISSION.md` are the 1,336 files under
    `hoi_estimation/`, all from Han's commit;
  - items 3 to 6: unchanged from `232e6b2`.
- **`hygiene_scan.json`:** the same scan over this folder after the README update, plus the scan of the files this update
  adds or changes. It also reports on Han's `hoi_estimation/`, which is report-only and merged unchanged.
- **`agent_run_map.json`:** for each of the 30 episodes, the agent run that produced its opus_v2 track, and the local
  modules its `solve.py` imports (section 6).

**After Han's later commits (`verification/han_followup/`; CPU only; see its README).**
- **`check_set_t3.py --files --hash`** on this commit's committed tree: **ALL OK**, as for the merge.
- **`code_identity_t3.sh`** on this commit:
  - `src` equals `41403d3`'s (tree `b8000f82`);
  - outside this folder and `TIER3_SUBMISSION.md`, the only changed paths are under `hoi_estimation/`.
- **The frame-0 step, re-run from `c0dbf5b`:** `t3_frame0.py opus <out> 3 2 shift` reproduces Han's frame-0 release
  (`tier3_frame0_supports_20261024`) byte for byte.
  - Result: 30/30 episodes, 22,990 frames, max |ΔT| = 0.
  - Inputs: the opus_v2 tracks, and a `ref_poses` that holds frame 0 only (the first parquet row, `frame_index` 0).
  - The diagnostic tail was cut, so no score was computed.
  - The same re-run from `6c10a22` gave the same result, in two Python environments (`from_6c10a22/`).
- **`han_edits_after_48c217a.tsv`:** each of the 111 `hoi_estimation/` files that Han modified after `48c217a`, with
  the class of its edit.
  - 103 are paths only (with `import os` where needed), and 2 are docs.
  - The other 6 files:
    - an opt-in episode subset (`V2D_EPISODES`): `tier2_refine/t3_frame0.py` and `support/common.py`;
    - `diag_scores.json` made optional for a print: `support/build_release_combined.py`;
    - overview images only: `support/overlay.py` creates its output folder, and `support/sheet.py` lays out any number
      of episodes;
    - a GT diagnostic, `diagnostics/rot_split.py`, whose f-string was tidied.
  - One portability edit broke a GT diagnostic: `diagnostics/meshdim.py` now reads `fos.environ`, a NameError when it
    runs. No step of the chain uses it.
- **`restore_test.log`:** the fixed `tools/restore_layout_t3.py`, run from this commit's tree into empty scratch roots
  with `--me <a> --han <b> --rewrite`.
  - Exit code 0.
  - No file is written under `<a>/data/`; the data-root records go to `<a>/t3ready_expected/`.
  - No `/root/autodl-tmp/han.zhou` path is left in the rewritten files.
- **`from_6c10a22/`:** the checks run on `6c10a22` (2026-11-02), whose evidence files are copied here:
  - `check_set_t3.py` gave ALL OK;
  - `export_reference.py` on our Tier-1 parquet (HF revision `5f68335f`, 30/30 LFS hashes) wrote 60 files.
    - Three runs, one of them in the kit's environment, wrote byte-identical files.
    - Its frame 0 equals what the frame-0 step consumed.
  - 5 of the 9 packed input files of each of the 30 release agent runs were rebuilt from the dataset, and each matches
    its run manifest's sha256. `camera.json`, recovered from the runs' event streams, also matches, 30/30. Its
    `T_world_cam` equals the s1 camera.
  - the human, camera and ground plane are byte-identical from s1 to the frame-0 release, 180/180 files;
  - no credential pattern was found in `agent_records/`;
  - the legacy support fit (`support/fit_20261023T2028Z.py`) differs from the current `fit.py` in 10 hunks.

## 6. Han's reconstruction code: where it is, and what is still open

`232e6b2` listed four pieces of Han's reconstruction as MISSING. Han's branch `dev.hanzhou.hoi_estimation`, merged here
under `hoi_estimation/`, supplies the code of all four. `han_zhou_reconstruction/README.md` maps each piece to files,
with what each file does.

| formerly missing | now in `hoi_estimation/` |
|---|---|
| **1. s1 pipeline** | `hoi_extraction/`: |
| | • CARI4D run per video (GroundingDINO → SAM2 → MoGe-2 → SAM-3D-Body → FoundationPose → CoCoNet): `cluster/run_episode.sh`, `cluster/gen_prompt_one.py`, `cluster/episodes.tsv`; batch drivers `scripts/run_local.sh` and `slurm/run_all.sh` |
| | • container and environment: `singularity/v2d_cari4d.def` and `v2d_gdino.def`, `slurm/build_*.sh`, and the one patched CARI4D file `patches/run_foundationpose_mhr_export.py` |
| | • the M1 prompt re-run (`M1_prompt_v2`): `cluster/gen_prompt_v2.py` |
| | • the P1 step (`P1_temporal`): `scripts/postproc_temporal.py`, and the other post-processing steps in `scripts/` |
| | • the first-frame alignment that writes `logs/pnp_forE1.json`: `scripts/firstframe_align.py` + `scripts/firstframe_pnp.py`; `E1` is `exp_fp_mode.sh E1_every_frame` |
| | • the `/workspace/v2d_cari4d/lib/cari4d` library is this repo's `reconstruction/modules/v2d_cari4d/lib/cari4d`, copied into the image by `v2d_cari4d.def` |
| **2. opus_v2 agent harness** | `agent/`: |
| | • prompt `AGENT_TASK_temporal_v2.md` (the same blob as the task file of all 30 release runs) |
| | • sandbox launcher `run_claude.sh` + `allowlist_proxy.py`, batch `run_batch_CL.sh` |
| | • input packing `export_inputs.py` |
| | • output → release `build_release_opus.py` |
| | • the per-episode helper modules: `agent_records/<run>/outputs/`, beside each `solve.py`. The run of each episode was identified by blob identity: `CL2_ep000`–`010`, `CL2b_ep011`–`021`, `CL2c_ep022`–`029`. Every local import of the 30 `solve.py` resolves there |
| **3. frame-0 correction** | `tier2_refine/t3_frame0.py`, with arguments `opus <out> 3 2 shift` (= `opus_f3_c2_shift`); `support/build_release_combined.py` publishes it as `opus_v2_frame0_f3_c2_shift` |
| **4. support fit and contact shift** | `support/`: `background` → `anchor` → `detect` (GroundingDINO) → `segment` (SAM2) → `fit` (nvdiffrast silhouette IoU) → `contact` → `package` (`T_world_mesh_contact`), then `build_release_combined.py` |

**Resolved by Han's later commits (`6c10a22` … `c0dbf5b`):**

| formerly missing | now |
|---|---|
| 1. extraction of `ref_human_f0/` and `ref_poses/` | `hoi_extraction/scripts/export_reference.py` (checked: section 5) |
| 2. exact command lines | `reproducibility/s1_runbook.md`: the s1 chain, including E2, P1, M1, ICP/PnP, `pnp_forE1`, FF2, JC2s1, DB and the release. `tier3_runbook.md`: `export_inputs.py` with the JC2s1 flags, the agent batches, the frame-0 step, and the per-episode support-fit settings (`support/fit_20261023T2028Z.py` for 13 of the 14 supports, the current `fit.py` for ep1). `reproducibility/run_all_tier3.sh` chains them end to end. |
| 3. revision of `reconstruction/modules` in the CARI4D image | `reproducibility/README.md`: v0.4.0 = `33129dd`, with the same `reconstruction/modules` code as this repo (Han's check: 371 of 371 non-LFS files identical) |
| 4. weights, images, data | located and pinned, not copied into git: `reproducibility/README.md` section 0 (the layout of the data root and its `environment/data_manifest.sha256.gz`; `environment/weights.sha256`, the image build definitions and pip freezes) |
| how `outputs/cari4d_failed/episode_000025_{out,work}` and ep0's `work/` were made | `s1_runbook.md` (ep0: copied from the CIAI cluster; ep25: the stage 0–5 outputs of the failed baseline run) |
| the support release files `fit_overview.jpg`, `contact_report.json` and the READMEs | `tier3_runbook.md` section C (copies of the `sheet.py` and `contact.py --snap` outputs); the hand-written release READMEs are in `reproducibility/release_files/` |

These rows are Han's documents. We checked rows 1 and 2 for the frame-0 step only (section 5).

**Still open (on Han's side; none of these blocks the file):**
1. **Data that only Han has.** About 400 GB: the two images, the weights, the CARI4D and experiment intermediates, and
   the agents' packed inputs and outputs.
   - It is on Han's lab storage, and a copy is on our training server. It is available on request
     (`reproducibility/README.md` section 0).
   - Until a reviewer has it, the GPU stages and the agent `solve.py` re-runs can be checked only as documents.
2. **GPU steps are not bit-reproducible** (`s1_runbook.md`, open point 4): CARI4D, FoundationPose and the GroundingDINO
   prompts. The CPU steps are deterministic given the same bundles. We re-ran the frame-0 step bit for bit.
3. **Origins and criteria not recorded:**
   - the shared intrinsics prior 820 / 758.1 / 560.5 (section 8, item 2);
   - why the E2 subset holds these 20 episodes, and M1 these 6 (plus ep28, which ran and is not used);
   - the choice of JC2s1 over JC2s2 / JC2s3. All three were scored (`s1_runbook.md` step 9); the scoring is not
     recorded.
4. **Episode 0's CIAI container** and module copy are not checked (`s1_runbook.md`, open point 1).
5. **Han to confirm** that his edits to executed scripts after `48c217a` change only paths and the opt-in switches
   (`verification/han_followup/han_edits_after_48c217a.tsv`).

**Corrections to the `232e6b2` MISSING list (found by its verifier):**
- `postproc_joint_camera.py`, `postproc_joint_camera_K.py` and `build_release.py` were already committed in
  `han_zhou_reconstruction/tier3_s1_20261020/code/`.
- So the JC1 / JC2 joint-camera steps (camera, principal point), the depth blend and the MHR → SOMA-X `PoseInversion`
  conversion were never missing.
- All five files in that folder are the same blobs as in Han's `hoi_extraction/scripts/`.

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

## 8. Grey areas and the organisers' answers

1. **True first-frame object pose.** The user confirmed on 2026-10-24 that this is allowed (FAQ). On 2026-11-02 the user
   reported that the organisers had confirmed the frame-0 reference data by e-mail.
   Han's frame-0 correction (`opus_f3_c2_shift`) uses only frame 0 of the reference object poses, in four steps:
   - a symmetric flip, tracked from frame 0;
   - **a camera-rotation correction (camrot = 2):**
     - over the static start, the chordal mean D of R0 R(t)^T is taken as the camera rotation error;
     - D is applied to the whole track about the camera centre, rotations only;
     - the step is skipped if D exceeds 15°;
   - an object-frame anchor, applied only above 15°;
   - a constant translation: the frame-0 position minus the static-segment mean.

   No later frame enters the output: our re-run with a `ref_poses` that holds frame 0 only is byte-identical
   (section 5). His support contact shift is computed from these tracks.
2. **Han's s1 frame-0 human alignment.** s1 places the monocular reconstruction in the official world with one rigid
   transform, fitted to the **frame-0 reference human**, and also estimates the camera principal point from it.
   - **Organisers:** the user reported on 2026-11-02 that the organisers confirmed by e-mail that the frame-0 reference
     data (`ref_human_f0` and the frame-0 object pose) may be used. This README does not record whether that answer
     names each of the uses below.
   - **Uses:**
     - the world alignment (ICP + PnP; a similarity with depth scale);
     - the frame-0 foot height as a floor cue (FF2);
     - the shared intrinsics prior;
     - the agent input `frame0_reference_human.npz`.
   - **Intrinsics.** JC2s1 holds the camera intrinsics at a shared prior: f = 820, cx = 758.1, cy = 560.5, with
     sigma_logf 1e-4 and sigma_pp 0.01 px (`s1_runbook.md` step 9). The K in the agents' `camera.json` equals the prior
     within 0.005 px.
     - f = 820 matches the median of the 30 per-episode MoGe-2 focal lengths, 820.03. That median is GT-free.
     - The principal point is not MoGe's, which is the image centre (768, 576). Han's s1 README says it was estimated
       from the frame-0 reference human. The exact derivation is not recorded (section 6, still open 3).
   - At least 12 of the 30 opus agent reports also mention the frame-0 reference human, for example as a floor-height
     or scale cue: ep00 01 07 08 09 10 15 20 22 23 24 29. (`232e6b2` said 10; ep08 and ep10 were missed.)
   - The agents received it as an input. `hoi_estimation/agent/export_inputs.py` writes `frame0_reference_human.npz`
     (frame-0 vertices only) into every agent's inputs, and each run manifest lists that file.
   - The s1 camera fit (`postproc_joint_camera_K.py`, JC2) also uses it to fit the focal length and principal point.
3. **Video-fit supports.** For the 14 placement episodes, Han fits the scanned furniture mesh to the video with
   GroundingDINO + SAM2 + a silhouette-IoU fit. The task text names the furniture. A vertical contact shift (−5 to
   +2 cm) makes his track rest on it. On top, we apply four GT-free placement rules (PREREG_f0 "Supports").
   - The meshes are the dataset's own meshes of the same furniture.
   - The released Tier-1 / Tier-2 support geometry is not used. The 16 floor-only episodes use released stage files
     that contain no geometry.
4. **LLM-agent reconstruction and its reproducibility.** The opus_v2 object tracks were produced per episode by a Claude
   Code (Opus) agent in a sandbox, starting from s1.
   - The agents' `solve.py`, `TOOLS.md` and `REPORT.md` are committed.
   - Since the merge, so are the harness, the prompt, the input-packing code, every run's transcript, event stream and
     manifest, and the helper modules (`hoi_estimation/agent/`, `hoi_estimation/agent_records/`).
   - The packed input arrays are not committed (section 6).
   - Re-running an agent would not reproduce the same `solve.py`. Re-running the committed `solve.py` with its inputs
     and helpers is the reproducible path; `hoi_estimation/diagnostics/repro.sh` does that with no network.
   - The release runs used `net=open` (`run_claude.sh`): every host except huggingface.co / hf.co, through a logging
     proxy. The proxy logs are not in the branch.
5. **Choices informed by Han's GT diagnostics** (stated in his code and README; no GT enters any output beyond the
   frame-0 object pose and the frame-0 human):
   - the frame-0 refinement variant (`opus_f3_c2_shift`) is his best variant by his GT evaluation;
   - snapping the object track onto the support was rejected in favour of shifting the support
     (`T_world_mesh_contact`) after a GT diagnostic (`support/package.py`);
   - `t3_frame0.py` reads the reference object poses, writes each trajectory from frame 0 only, and then scores it
     against the reference in the same run (`diag_scores.json`);
   - JC2s1 was chosen among JC2s1 / JC2s2 / JC2s3 after scoring each variant (`s1_runbook.md` step 9). The criterion
     is not recorded (section 6, still open 3).

   Our pipeline made no choice among his variants. It used his frame-0 release as given (PREREG_f0).
6. **Also stated:**
   - the human is s1's, unchanged while the object moved; the mitigation is by training config only (palm threshold
     0.25 for 8 episodes);
   - the frame references are ours, built from Han's f0 track;
   - the scene files (ground plane, mesh) are the released ones;
   - Han's GT-based README quality tables were read but chose nothing;
   - `hoi_estimation/` is Han's branch merged unchanged. It contains GT-based evaluation scripts and tables
     (`diagnostics/`, `eval_official.py`, `eval_object_pose.py`, his README). They are committed as they are, for
     completeness. Nothing of ours reads them, and this README quotes none of their numbers. Han's later documents
     (`reproducibility/`, his reproduction report, his README) also contain GT diagnostic numbers; this README quotes
     none of them.

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
