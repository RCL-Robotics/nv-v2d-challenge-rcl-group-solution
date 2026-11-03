# t2_consensus_r8_20261103: PREREG_R8_PINS (step B4; pins and implementation notes only)

Written 2026-11-02 about 21:50 CST, after steps B1-B3 of PREREG_R8 10 were re-run for the revision of
`PREREG_R8_CHANGES.md`. This is before T_c and before any round-8 export, evaluation, score or selection.
- **Document in force:** `PREREG_R8.md` bb912c57180dde8255776884de169e12467349f57d584782308b88ff935abf8c.
- **Addendum A1 (section 3)** was added 2026-11-03 at about 13:50 CST, after T_c and before any round-8 evaluation
  (`AMENDMENTS_R8.md`). It adds pins and notes only; the four pin lines it supersedes are kept in section 2 as
  `superseded-by-A1` lines with their old values.
- **Scope:** pins and notes only. Nothing here changes a rule of PREREG_R8.
- **Path rule:** a path starting with `/` is absolute; any other path is relative to this folder.
- **Checks:** every round-8 script refuses to export, evaluate, freeze, score or select unless all of these hold:
  - `PREREG_R8.md` == `PREREG_R8.sha256` == its hard-coded value;
  - this file == `PREREG_R8_PINS.sha256`;
  - this file pins the script itself, and every `pin` line below holds;
  - every line of `LOCK_R8.sha256` holds.

## 1. Implementation notes

1. **`r8_pool.py` (flashchord Python `-I -B`).**
   - It loads `scripts/t2cons_r7_pool.py` (7a08fa6d) from its verified bytes and calls its helpers unchanged: header,
     root / sequence parsing, `server_parquet`, `manifest_episode`, `ep_frame_ref`, `picks_of` (R4), `run_record` (R1-R4,
     pro6000), `scan_pro6000`, `z5_sums`, `first_train_t2_line`, `copy_verified`, `write_excerpt`, `load_rt` (round 5's
     `relocate_bytes` / `verify_pair`) and `verify_row`.
   - It re-binds, rather than edits, four names of that module:
     - its skip table, which gains the three new top folders of PREREG_R8 2.2;
     - its file guard: round 7's token guard first for every path, then an allowlist for Zuxing's folder (PREREG_R8
       2.3, 2.6, 9);
     - its sha256 helper (through the guard);
     - `settled_models` (round 7's R3 plus "ctime <= T_c").
   - New code covers the round-8 deltas: copy runs (R1c), the grid restriction on 5090-hosted runs (mechanical picks and
     the watcher's last ok-evaluated iteration), the watcher best with its header check, the identity / role assignment
     (file sha256, else the tensor data-block sha256), copies and record excerpts (PREREG_R8 2.8), relocation and outputs.
   - **Modes:**
     - `dryrun`: snapshot now, nothing copied;
     - `trial`: a small frozen pool under `dryrun/trial_*`, for the plan check only;
     - `freeze`: only at or after T_c; refuses if `POOL_R8.json` exists; checks PREREG / PINS / LOCK first;
     - `verify`.
2. **`r8_eval.py` (flashchord Python `-I -B`).** It loads `scripts/t2cons_r7_eval.py` (1d66a2db) twice from its verified
   bytes.
   - **Instance E** is re-bound to this folder: paths, the R8 pin check, the R8 manifest builder, the R8 own-input
     summary and an R8 driver subclass.
     - The subclass adds the GPU rule of PREREG_R8 3.5: a GPU with any listed compute process is not usable.
     - It drops round-7-only progress fields.
     - It writes round 7's closing line, which names a step round 8 does not have, as "every export and evaluation is
       final".
   - **Instance E7** is never re-bound. It rebuilds reuse7 finalists from round 7's own records, and its unchanged
     `check_pins` re-verifies round 7's whole chain, including round 1's, at every start.
   - **Unchanged from round 7:** the export / evaluation commands, ok checks, retries, lane locks, MPS refusal, seeds,
     OVERDUE flagging and never-signal are round 7's code (and through it `t2cons_eval.py` 2cf42786 /
     `t2best5_driver.py` 24b9af39).
   - **Inherited text:** messages printed by round 7's code keep their round-7 section references (for example "never
     (PREREG_R7 3.4: GPUs 6 and 7 only)"), and some record field names are round 7's.
3. **`r8_select.py` (kit Python `-I -B`, CPU).**
   - It loads `scripts/t2cons_r7_consensus.py` (e7545dfa) from its verified bytes and calls `s_row`, `select_s`,
     `decide_f`, `trial_rows`, `closest_trial`, `chain`, `scored_inputs`, `load_incumbents`, `load_evals`,
     `check_rollouts`, `input_ref`, `_export_from_record` and the pinned round-1 / round-2 functions unchanged.
   - Its `gate` / `score` commands are round 7's loops over round 8's runs. An existing output written by this script
     from the current inputs is answered "already done" (PREREG_R8 7.1).
   - It refuses `keep-r7`, and its chain check fails, while an `r8_eval.py run` process is alive (PREREG_R8 7.6).
   - Its `r7records` command (step B2) wrote `inputs_r8/R7_RECORDS_R8.json` from round 7's verified records:
     - the 30 incumbents, each with its round-7 phase-F runs, m_P_frozen, m_P_fresh and bar;
     - round 7's 399 phase-S candidate identities;
     - the phase-F participants per episode.
     It asserted that every participant fails the round-8 test on its own round-7 runs. Apart from its time stamp and
     script sha256, the file equals the first build's (checked).
   - **Selftest** `selftest_r8/selftest_r8.json`: 14 synthetic checks, PASS.
   - **Mock harness** `dryrun/r8_mock_select.py`: `select S`, `select F`, `select F` without finalists and `keep-r7`, run
     on synthetic round-8 values with every write captured in memory, plus keep-r7 refused while an evaluation driver is
     alive. 17/17 checks PASS (`dryrun/b2_mock_select.log`).
     - Both no-change paths reproduce round 7's trial 2 from the real round-7 runs. This is the consistency check of
       PREREG_R8 5.6.
4. **`r8_driver.py` (flashchord Python `-I -B`)** is the orchestrator of PREREG_R8 7, started by `r8_launch.sh` with
   setsid nohup.
   - It checks PREREG / PINS / LOCK first and sleeps until T_c.
   - It runs the steps as foreground children in their own sessions, and never signals anything.
   - It never starts a second `r8_eval.py` while one is alive (read from /proc) and does not count that wait as a
     restart; a pin / preflight / integrity failure of `r8_eval.py` halts (PREREG_R8 7.4, 7.6).
   - It writes `r8_driver.log` and `R8_DRIVER_HEARTBEAT.json`.
   - It creates `STOP` plus `S_CUTOFF.json` or `DEADLINE_F.json` only at the deadlines of PREREG_R8 7.3.
5. **Build checks (records in `dryrun/`, summarised in `DRYRUN_R8.md`).**
   - **Pool dry run:** snapshot 2026-11-02 21:45:35 CST, 89 screen candidates, 11 s.
   - **Trial pool:** 3 screen rows: a pro6000 copy, a 5090 relocation and a ws032 relocation. Relocations verified. The
     trial folder's `POOL_R8.sha256` holds. No text file of the trial folder names a file or score outside the
     own-input / consensus-B basis.
   - **Plan-only evaluation check on the trial pool:** `plan: OK` for phase S (3/3 jobs ok, hulls ok, kit and stock
     unchanged, GPUs 6/7 correctly refused while compute processes are listed) and for phase F (no finalist yet, as
     expected).
   - Development copies of the scripts and their outputs were removed before step B2; the first build is kept read-only
     in `superseded_build1/` (not pinned, not used).

## 2. Pins

pin PREREG_R8.md bb912c57180dde8255776884de169e12467349f57d584782308b88ff935abf8c
pin PREREG_R8.sha256 55114e8790dca543a5e170fb5a8cce89728c6d96c9e439723486369f69ca653d
superseded-by-A1 PREREG_R8_CHANGES.md d3a6e4413d748ed0150d70fb15c5e3d6ebddc917dcb67a16affcdb8e5cf3110c
pin r8_pool.py bebeb0132cf45c5320aae060bc50e1d4f1d46e9895f0cdac46cc0797ba0d4628
superseded-by-A1 r8_eval.py 81bc9d1c36c344558ce1bcbf92f60453f5128493d66760582028b8774203f574
pin r8_select.py 735149acfd54eb5db39a776b4775303e16283ca274f34e71215581a94ca2508e
superseded-by-A1 r8_driver.py 1d626bed15b8e2feabd3f555165a76d179ee773cdd28fdf8284752c1a10408c9
superseded-by-A1 r8_launch.sh 7dee83ada1d7eb29283d9294fcfc90b46cb656a939862bc98dcbddb9aaa6db62
pin dryrun/r8_mock_select.py 25639e715ef81d77c22480332fe222f177b4176d157798671a8598029e4873ea
pin inputs_r8/R7_RECORDS_R8.json 60c2798c7e8666dcd060e0eca286007c6d84dc46467e22c37052a6613419df9e
pin selftest_r8/selftest_r8.json 905aa62697a1d378035604699bade3ad7e15cca410bdbe70b81c2f40e929859f
pin dryrun/b2_r7records.log 7c9fa4187a57aeb6ae279078a989f62d6b81652a5c86c613390f291747f0646c
pin dryrun/b2_selftest.log 79f5835c47bbbf78eb6ff86dd6d3deb22c4f4100545c7719748c572e50b6a891
pin dryrun/b2_mock_select.log 5c8799289305ad48360c1ca968474222e304706e52cc24b13d020cb587b87e2a
pin dryrun/b3_pool_dryrun.log 8896fbfdc9d908b3ccca6ca0f5c1fce7231aab6eb6f964ec880f05bbbb966c99
pin dryrun/pool_dryrun_20261102_214535.json 6b1f85618d1a23181d88d7a772120056dccdd3bef1ac179ffe57225ee615865a
pin dryrun/pool_dryrun_20261102_214535.md f081704e99032985823cd703b047d3b6fa877770d07da6b1634164b86ab19e29
pin dryrun/b3_pool_trial.log cabd3050e7906d666ffba828dc61c856d9760b903cb449835bfde9ed869ca18c
pin dryrun/b3_pool_trial_verify.log 4725d68c7442d95d1deb0af015428f57aff77d80c95621322277920a1005d47f
pin dryrun/trial_20261102_214546/POOL_R8.sha256 0b62b254dc3d4c5896a50b146c226815e5c1260fd94baec1b5cca2228d35f844
pin dryrun/r8_eval_plan_screen_20261102_214631.txt a8e756dd8667cddcb7d5f8702788ff3f12080879e525775b0e57620522703330
pin dryrun/r8_eval_plan_final_20261102_214638.txt 676bc34009f3d35de0c970ebe0b04b8e744a419c6c2233c44181865f6f1a0f3c
pin dryrun/b3_eval_plan_screen.log 9a454c9e61defb9ad2ca08d8b7357cf95c4b97c134da330669c2e4d3ce2d61e6
pin dryrun/b3_eval_plan_final.log 676bc34009f3d35de0c970ebe0b04b8e744a419c6c2233c44181865f6f1a0f3c
pin /root/autodl-tmp/haoran.zhang/scripts/t2cons_r7_pool.py 7a08fa6d8be91140e139d914b8119352fabb05f58b415eb4a8904f9a02733770
pin /root/autodl-tmp/haoran.zhang/scripts/t2cons_r7_eval.py 1d66a2dba53287c6a320310c9524fd9ed00e50289852c9eccba74882028a9f2a
pin /root/autodl-tmp/haoran.zhang/scripts/t2cons_r7_eval_launch.sh 6111638d0631112de8789f598dd33b80a36ea28f587f3532c3de510a13d19d8f
pin /root/autodl-tmp/haoran.zhang/scripts/t2cons_r7_consensus.py e7545dfa7fdff0681123d77e2dde3f65ed6a22ead8d4d12057d9b61af9e2708a
pin /root/autodl-tmp/haoran.zhang/scripts/t2cons_eval.py 2cf427866a9432a3bd1b8fbb7864149d4258a5d19100c20da92d32bfe4fbfa9b
pin /root/autodl-tmp/haoran.zhang/scripts/t2best5_driver.py 24b9af39a2e2708641e7870447b12b6475c23aec9cd18ba71086768dd69cfc89
pin /root/autodl-tmp/haoran.zhang/scripts/t2cons_r2_consensus.py ddee0b8b6fa25029275b21a3dc82d15c36f4de71c0a61f2a1046869b01ec13aa
pin /root/autodl-tmp/haoran.zhang/scripts/t2cons_a1_gate.py a29c997b7271e9299db109e97af5210f528484719210bb58685523b0ae0eb16b
pin /root/autodl-tmp/haoran.zhang/scripts/t2cons_consensus.py e1b943b10fd1db13ab6f78e5bde7b3c6e6c48f7daeee6230c32c463d14ac41b4
pin /root/autodl-tmp/haoran.zhang/scripts/t2cons_r5_relocate.py 09a6cb8e8a3f8e835c4b10caa37f6ad46b780374480e856cca7efbe479938bdb
pin /root/autodl-tmp/haoran.zhang/tmp_analysis/t2fix/cache_keys_t2.py 9f64214eba0eb6602e8ccb9cc200fe9c6ff94676e0c99551f095b3b1db5c1aae
pin /root/autodl-tmp/haoran.zhang/v2d_submission_kit_src/v2d_submission_kit/flash_chord/scripts/evaluate_policy.py 160d24f5fbeb70ccd569ca4192056f1892b2a9f2999066db9d8279461648fdf1
pin /root/autodl-tmp/haoran.zhang/v2d_submission_kit_src/v2d_submission_kit/examples/run_policy_evaluation.py 0bcc41f5de246a51f194409b3e5292e04109012cb22a433a8f7ed477df04e55d
pin /root/autodl-tmp/haoran.zhang/v2d_submission_kit_src/v2d_submission_kit/tools/pack_submission.py 47a4589574ff8284ba4376ef31ac5548c8fe87ead753be2d3b43d6c87041e5be
pin /root/autodl-tmp/haoran.zhang/v2d_submission_kit_src/v2d_submission_kit/v2dlb/object_pose.py c6915924a322540f00c69c75b2ae93ae455bb6efb8a41cc0a96c967f64d7e2c0
pin /root/autodl-tmp/haoran.zhang/v2d_submission_kit_src/v2d_submission_kit/v2dlb/vertices.py 4c95a4d65d37ea97d6c58388b273c277cad9d7cfa63d3cdb7af2ec3d1f1a0262
pin /root/autodl-tmp/haoran.zhang/code/nv-v2d_runin/robotic_grounding/flash_chord/scripts/track2/eval_all.py 657295921e820a464160039ad9a65ba92a683108a99dcb197290ae4147363197
pin /root/autodl-tmp/haoran.zhang/code/fc_r2/scripts/track2/eval_all.py 657295921e820a464160039ad9a65ba92a683108a99dcb197290ae4147363197
pin /root/autodl-tmp/haoran.zhang/code/nv-v2d_runin/robotic_grounding/flash_chord/src/flash_chord/training/ppo/coordinate/export.py 04367c56564de9540deb3d8e33adbfddb7800df190361a0576005a1a4ccfd212
pin /root/autodl-tmp/haoran.zhang/code/fc_r2/src/flash_chord/training/ppo/coordinate/export.py 34e3d7f3a7dca4fdd1e7c4d26750c6332abcc1d629f6a9c6c7a965833a22206f
pin /root/autodl-tmp/haoran.zhang/code/video_to_data_8c778f9/robotic_grounding/flash_chord/src/flash_chord/evaluation/harness.py 6e61b092e9aa9d00d11c557b8508c09ef990314a3f68b0779bbf97765c85ddc2
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r7_20261102/PREREG_R7.md 749649a79ca5616b8ddbfc5124828c13da20f2e10e850a39647a74d83c939ec6
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r7_20261102/PREREG_R7_PINS.md 8378c5a418c9b1ece0ac8a1c518237666d4649cba211229b18d95d89bff14fce
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r7_20261102/PREREG_R7_PINS_b.md 73d17f24f63aaef1ab57b3a47d7a18fbadc1bb99b49802b8739ffeb367eb5758
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r7_20261102/LOCK_R7.sha256 cef6faeb111c78f7336011bea5c585022a433d54ea0e33c515f1a54e5fffa992
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r7_20261102/SELECTION_S7.sha256 106fe7beb468c34810e63b0c22a09feea1326506c35f3e8ac6935e1733c29827
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r7_20261102/SELECTION_R7.sha256 b2dc89dd574ef7a0d241af57ad3c1f2693aa1d8d6410590a0f18597016d532ee
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r7_20261102/selection_r7.json 657275ba5f2f7dd17aabdd99f9b71f4fd3703c07fafdcd34aba473e6ea511010
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r7_20261102/final_r7.json 3d82ca52250d77408561ddfa2577a0816765ddc8f14b112030c40399cb0f454c
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r7_20261102/inputs_r7/pool_r7.json 7c4b553bf2d15ffdae0d2b19097ab6abccfba03d1575b268fb3c0338b0f99464
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r7_20261102/inputs_r7/runs_r7.json 155432e2c0b40653f7d4c060ca9a7627ab99a2f392c2facfbac3e8da6cde30cf
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r7_20261102/inputs_r7/ROLLOUTS_F7.sha256 1ee071829c72ec89226dbce387515e4f4501616b9210d76356a88b44cefbb0e6
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r7_20261102/inputs_r7/TRACKS_R7.sha256 4c7383266df8350dce686aaa800e2d46c2076f5ee3399c071b5eee12b8ee7788
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r7_20261102/RELOCATION_R7.tsv ccd24c5d9b367ad68cf6d9c3c8fbc47affd5d9a75ca8ac1b292ea350eb74079c
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r7_20261102/COPIES_r7.tsv 028f0139909ecf3e2c0ea84c676bf91dcf07fad6a209c716eb241a436075b954
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r7_20261102/submission_r7/T2_R7_closest_trial02_6bb0214.csv b06c5cccb02d9a5aa61827f31371094ae02c918762e94c60feea0b691885609d
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r6_20261031/inputs_r6/runs_r6.json c7e529e2b7f16fef73e1a42cf8db9e589dc68875f4a0f4aba02d06d2089032d5
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r6_20261031/COPIES_r6.tsv fa17929cf145cbb8333464b559521f699a2712319bd6b9833e017dea92d000f9
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_20261024/refs/build.json 9646de2daf02301dd2efbc72a7507c539b1100c3ec0fe39adade169195d18dd3
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_20261024/integrity/baseline.json ef9b5035c09b8ad40713d9fb08470eeccb2abe8845d4d1cce3037c49da1cd622
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_20261024/inputs/TRACKS.sha256 c8cb834522f245de0b7733579f8c537781eef14d79269e782bc391f1dcecca6d
commit code/fc_r2 61fa5d31ff8054bf1e5cf5375ef0957e54d2729b
commit code/nv-v2d_runin b39b69be6c2f561c0123c6aeb8cd13f8b9d542e7
commit code/video_to_data_8c778f9 8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef

## 3. Addendum A1 (2026-11-03, about 13:50 CST; `AMENDMENTS_R8.md`)

Written after T_c and after the 89 phase-S exports, before any round-8 evaluation. A1 is operational: evaluations may
start on GPUs 0, 4, 6 and 7 (before: 6 and 7). Every other rule of PREREG_R8 3.5 holds, and the selection rule is
unchanged. `PREREG_R8.md` is unchanged (bb912c57...).
- **`r8_eval.py`**: `ALLOWED_GPUS_R8 = (0, 4, 6, 7)` re-binds round 7's module name `ALLOWED_GPUS` in `configure()`
  (round 7's pinned `gpu_rows` and `load_driver` read it at call time). Its GPU-row check also marks every GPU outside
  the set unusable. Defaults are now `--gpus 0,4,6,7 --max-parallel 4`; `--max-parallel` may be up to 4 and `--gpus`
  may name only GPUs of the set; `--lane-max <= 2` and `--cooldown >= 10` are unchanged.
- **`r8_driver.py`**: starts `r8_eval.py run --gpus 0,4,6,7 --max-parallel 4 --lane-max 2 --cooldown 10`.
- **`r8_launch.sh`**: comment only.
- **`PREREG_R8_CHANGES.md`**: one dated pointer entry (its section 3).
- The superseded files are kept read-only in `superseded_a1_20261103/` (not pinned, not used).

pin PREREG_R8_CHANGES.md 0f248a2f5acd915165aa83ebb88bed86fc053fe3e3f7e073eaa6faa3fed34680
pin r8_eval.py 5dcf6a8235c4317fae2a7ed527f3b589f58e7390ae0deb34a0c4301f7b212c42
pin r8_driver.py 3072efe2d8c57947f91c220d67b984102ed951893fdc72baebbf63f0a0356078
pin r8_launch.sh 3177cd9b2b0d7fd4ab3e4a7b54255b19c59fed6078488da643be0dbbdc2d586d
pin AMENDMENTS_R8.md fb71b87bf5d3d7792ef0a0cc62470b2bbe6366fb3a954cebc03a33dc75845c57
pin AMENDMENTS_R8.sha256 0d653ae5b967862853838077e55a44fbc15b3bab1e33f6772b4d12190719d035
