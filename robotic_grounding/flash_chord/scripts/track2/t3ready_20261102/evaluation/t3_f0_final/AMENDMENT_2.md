# AMENDMENT 2 to `fc_runs/submissions/t3_f0_final/PREREG.md`: the FINAL_r builder (written before its real run)

Amends `t3_f0_final/PREREG.md` (sha256 `e4e3a5daf239f5b5858e32442c43eb708947b38c20eae76e12af757a747bd0eb`) under its
section 9. It is the file that `AMENDMENT_1.md` (sha256 `e2a434388be698163fe110057831490c0900192ee90846b127ed1f7c27a24faa`),
"(d) Builder", and `tmp_analysis/tier3/f0_retry/PREREG_v3.md` (sha256
`e62c39b2e092a0abce903168c74d64fcdfa3ce0c57a22f05153469ff1ea79db7`), section 7, say must name the FINAL_r builder's
sha256 before it runs. Written 2026-11-01 ~17:25 CST by a subagent of the main session. Nothing here changes any
item, run, selection, acceptance decision, chosen run or file of t3_f0_final or of the retry wave.

## State when this was written

- 13 of the 14 frozen retry candidates have the outcome POST_DONE. `f0_retry/VERDICT_retry.json` (written 2026-11-01
  16:14:13) has REPLACE for 02 04 05 13 16 17 19 21 23 25 and KEEP_F0 for 01 11 28.
- ep14 is still training (pid 917543, GPU 7; `rsi_88000` at 17:02 CST); its post step has not started.
- FINAL_r does not exist. `fc_runs/submissions/TIER3_FIRST_UPLOAD` is absent: no Tier-3 file of ours has been uploaded,
  and no Tier-3 board score of ours exists.
- Relayed user request at this time (verbatim): "current ablation and other training progress? does new tier3 out?"
  Building FINAL_r is the pre-registered next step (PREREG_v3 section 7, AMENDMENT_1 (c)); no new choice is made.

## 1. The builder

- **File:** `tmp_analysis/tier3/f0_retry/t3final_r_build.py`
- **sha256:** `29dc19339d2e38c96352438a971752332c226ff1a153fc78bdd4e0e413d61057`
- A new file. No running or pinned file was edited. It runs CPU-only as
  `conda_envs/flashchord/bin/python -I -B t3final_r_build.py {selftest|when|build|report|repack}`.
- `build`, `repack` refuse unless this file exists, matches `AMENDMENT_2.sha256`, and names the builder's own sha256
  and the t3final_kit.py sha256 below.
- It calls, unchanged:
  - the kit packer `tools/pack_submission.py` (sha256 `47a4589574ff8284ba4376ef31ac5548c8fe87ead753be2d3b43d6c87041e5be`),
    as `conda_envs/v2dkit/bin/python -I -B`, `--track track_2_tier3`;
  - `t3_f0_final/t3final_kit.py` (sha256 `1c8f94b96a05934f71a3893d2910702e1e5fc186248fcc4f0ab4ad2451beebc7`, pinned in
    PREREG.md section 10): `validate` (the uploader's own `validate_submission()`, offline, plus provenance) and, only
    after the lock, `score`.
- Decision reads go through the t3_f0_final no-clean guard (refuses paths naming clean, LOCAL_NOT_FOR_UPLOAD,
  tier_1_multiview_caption, track_2/baseline).
- An earlier version (sha256 `7cf50be3a096a998ce26c0ba392981fe54dbc209ae38ecd78a8eb9f836f8c998`, kept as
  `f0_retry/final_r_test/t3final_r_build.py.selftest1_7cf50be3.bak`) was only selftested. The final version differs
  by a table label and by renaming a half-built FINAL_r to `FINAL_r.FAILED_<stamp>` on an exception.

## 2. The rule it implements (AMENDMENT_1 (a), (b), (d); PREREG_retry section 6), nothing else

Per episode NN:
- **Retry item** iff, at the build time: SUBMISSION_CHOICE lists NN with `use = retry_final`; `eval/epNN/post.json` has
  status `done`, a decision starting with `REPLACE` and the same `retry_final_sha256`; and the v2 pins
  (`PREREG_retry.sha256`, `RETRY_FILES.sha256`) and v3 pins (`PREREG_v3.sha256`, `RETRY_FILES_v3.sha256`) all match.
  - Rollout: one of the 5 official own-input evaluations of the retry final (`eval/epNN/retry/rK_*`), each re-checked
    against every t3_f0_final PREREG section-2 validity condition. The pinned file is that run's own exported checkpoint;
    the export source is the retry `final.json` checkpoint (model_90000, max_iterations, start 44922, sha256 as in
    post.json).
  - Pick: among the passing runs, own-input ADD-AUC closest to their mean, exact rational arithmetic on the float64
    values, ties to the lower run index. The builder recomputes it; if it disagrees with SUBMISSION_CHOICE `chosen_run`
    (run, files, sha256s), the build STOPS and a human decides. If no run passes, the episode keeps its f0 item (logged).
  - Frame reference: the episode's own `frame_references_f0` npz (byte copy, sha256 listed in the f0 FILES.sha256).
- **Otherwise:** t3_f0_final's frozen item and frozen selection, the same rollout as FINAL_f0 (checked field by field
  against `pack/FINAL/file.json`, byte copies).
- **Pack:** the kit packer unchanged, placeholder commit URL
  `https://github.com/local-placeholder/not-uploaded/commit/0000000000000000000000000000000000000000`; required: packer
  exit 0, 30 converted episodes, input-reference residual < 1e-4 m and < 1e-3 rad, uploader `validate_submission()`
  pass, provenance (uniform evaluator / packer / uploader digests and commit URL; per-episode checkpoint digest = the
  item's eval file, or the chosen retry run's exported file).
- **No metric_code in the build.** The kit packer prints its own clean cross-check ("locally computed"). As in
  t3_f0_final, that stdout goes only to `pack_LOCAL_NOT_FOR_UPLOAD.log`, and only the residual, row and episode-count
  lines are parsed.

**Mechanical strictness beyond the text** (it can only keep an f0 item or stop the build, never pick a different run
silently): each retry run's `run.json` must equal its post.json record; its eval_report row, export report (status ok,
tolerance, stock-runtime verified, help scale 0, destination = the evaluated file, source = the retry final) and results
JSON / parquet metadata (eval_script 160d24f5, eval_code b1b3ebe3, step count, environment_valid on every row) must
agree; before the deadline the build waits until VERDICT_retry, SUBMISSION_CHOICE and post.json say the same for every
done clip.

## 3. Acceptance test (AMENDMENT_1 (d)) and selftest, run before this amendment

`t3final_r_build.py selftest` (CPU only; writes only `f0_retry/final_r_test/selftest_<stamp>/`; never FINAL_r):
- Run 1: builder `7cf50be3…`, `selftest_20261101_171957`. Run 2: builder `29dc1933…` (this one),
  `selftest_20261101_172132`. Both: 0 problems.
- **Acceptance test PASS:** with every episode on its t3_f0_final item, the pose-columns sha256 is
  `778b808369df51fa3d176982e2b584de21bf7d87f6579808459a03763c66fefa`, equal to FINAL_f0's. The whole CSV is
  byte-identical to FINAL_f0's (`6f73715a…`).
- All 65 retry-arm runs of the 13 decided clips (REPLACE and KEEP) pass the (b) validity check. The recomputed chosen
  run agrees with SUBMISSION_CHOICE for all 10 REPLACE clips.
- The composition as it stands now (ep14 on its f0 item, as the rule gives it while ep14 has no outcome) packs: residual
  1.44e-07 m / 1.46e-07 rad, uploader validation pass, no provenance problem.
- The repack path, run with a dummy well-formed commit URL, gives equal pose columns.
- No clean scoring was run. The packer's clean cross-check lines are only in the selftest dirs'
  `pack_LOCAL_NOT_FOR_UPLOAD.log` files, which nobody opened.

These selftests are test runs that build no FINAL_r. They happened before this file named the builder; that is
disclosed here.

## 4. Build time (AMENDMENT_1 (c), unchanged)

- **The first of:**
  1. every frozen candidate (01 02 04 05 11 13 14 16 17 19 21 23 25 28, = `f0_retry/recipes/`) has an outcome:
     POST_DONE (post.json status `done`), a LATE or CRASHED marker, or never started by the training cut-off
     2026-11-02 12:00 CST;
  2. 2026-11-02 18:00 CST.
- `when` reports this; `build` refuses (writes nothing) before it.
- At the build time, a candidate without a done post keeps its f0 item. In the deadline case, a post finished after
  18:00 CST is ignored. This includes ep14 if its post is not done by then.
- **Once only.** `build` refuses if `pack/FINAL_r` or `FINAL_r.sha256` exists. A build that fails the (d) checks, or
  raises, is renamed `pack/FINAL_r.FAILED_<stamp>`, and a human decides. Until a valid FINAL_r exists, FINAL_f0 stays
  the designated full file (AMENDMENT_1 (e)).

## 5. Output and hash lock

`fc_runs/submissions/t3_f0_final/pack/FINAL_r/` contains:
- `packed_LOCAL_NOT_FOR_UPLOAD.csv` (placeholder commit; not an upload file);
- `results/`, `frame_refs/` (byte copies);
- `file.json`, `expect.json`, `validate.json` / `.log`, `pack_status.json`;
- `pack_LOCAL_NOT_FOR_UPLOAD.log` (packer stdout, not read);
- `choices.csv` / `.md`: the per-episode choice table with the chosen run's own-input AUC / SP-SR / MP-SR / MPPE and the
  30-episode own-input totals;
- `build.json`, `build.log`;
- `inputs/`: byte copies of the SUBMISSION_CHOICE.json, VERDICT_retry.json, post_epNN.json and pids.tsv the build read.

It is then locked:
- `FILES.sha256` lists every file;
- every file is made read-only;
- `t3_f0_final/FINAL_r.sha256` (write-once) holds the CSV sha256 and the `FILES.sha256` sha256.

## 6. New, disclosed: a LOCAL report-only clean score after the lock

After FINAL_r is built and hash-locked, `t3final_r_build.py report` may compute a LOCAL report-only clean score:
- It refuses unless the lock verifies.
- It runs the official kit metric_code (`t3final_kit.py score`: AUC, SP-SR, MP-SR, MPPE against the clean reference)
  on FINAL_r and on the round-1 FINAL_f0 CSV (sha256 checked).
- Output: overall and per episode, side by side, written only into
  `pack/FINAL_r/report_LOCAL_NOT_FOR_UPLOAD/*_LOCAL_NOT_FOR_UPLOAD.*`.

**It changes no decision**, because:
- the file choice is already fixed: FINAL_r if it is built and passes (d), otherwise FINAL_f0;
- the Final mark rule is already fixed: the designated file in all 4 Tier-3 competitions, never chosen by a board or
  clean score (AMENDMENT_1 (e));
- no step of any pipeline reads the report.

Round-1 FINAL_f0 already has such a LOCAL score from 2026-10-28 (t3_f0_final PREREG section 5). This repeats it with the
same code, so the two files can be compared side by side.

## 7. Repack and upload (unchanged rules)

- `repack --commit-url <https://github.com/<owner>/<repo>/commit/<40 hex>>` re-packs FINAL_r's composition into
  `t3_f0_final/upload_r/<sha7>/`. The pose columns must equal FINAL_r's.
- It uploads nothing. Any upload needs the user's OK.
- The cited commit must also contain:
  - Han's reconstruction code;
  - the f0 prep code;
  - `t3retry*.py`;
  - this builder;
  - AMENDMENT_1 and AMENDMENT_2;
  - the retry records.
- The submission note must list the retry items (AMENDMENT_1 (e)).

## 8. Disclosures

- **A1. Own-input numbers seen.** The author saw only Tier-3 own-input numbers: VERDICT_retry, SUBMISSION_CHOICE,
  post.json / run.json, t3_f0_final selection and run records, and the selftest choice table.
- **A2. Files not opened.** It opened no file whose name contains clean, LOCAL_NOT_FOR_UPLOAD, VERDICT_clean,
  summary_LOCAL, s4a_, s4b_, clean_stat, GT or gt_eval, and read no Tier-1 or Tier-2 data or results.
- **A3. Clean numbers in context.** The author's context held an automatically provided set of project notes. They
  include Tier-1 / Tier-2 aggregates and Tier-3 clean aggregates: round-1 FINAL_f0's offline score, and some
  opus_v2-era clips. None was used. The builder has no free choice: items, runs and the pick are fixed by the rules
  above.
- **A4. Run order.** The selftests (section 3) ran before this file was written, as test runs, and built no FINAL_r.
