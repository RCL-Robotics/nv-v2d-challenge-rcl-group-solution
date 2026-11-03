# t2_consensus_r9_20261103: PREREG_R9_PINS (step B4; pins and implementation notes only)

Written 2026-11-03 about 22:35 CST, after steps B1-B3 of PREREG_R9 10 were run with the final scripts. This is before the
round-9 pool freeze and before any round-9 export, evaluation, score or selection.
- **Document in force:** `PREREG_R9.md` 13c3b2a61e39cc761633283ae42bf94683df8d1ffd701d1076e3d61212a4c2a0.
- **Scope:** pins and notes only. Nothing here changes a rule of PREREG_R9.
- **Path rule:** a path starting with `/` is absolute; any other path is relative to this folder.
- **Checks:** every round-9 script refuses to export, evaluate, freeze, score or select unless all of these hold:
  `PREREG_R9.md` == `PREREG_R9.sha256` == its hard-coded value; this file == `PREREG_R9_PINS.sha256`; this file pins the
  script itself, and every `pin` line below holds; every line of `LOCK_R9.sha256` holds.

## 1. Implementation notes

1. **`r9_pool.py` (flashchord Python `-I -B`).** It loads round 8's `r8_pool.py` (bebeb013) from its verified bytes; that
   import loads round 7's `scripts/t2cons_r7_pool.py` (7a08fa6d) and re-binds four of its names as in round 8. Round 9
   re-binds, rather than edits: the skip table (the four 2026-11-03 bundles of PREREG_R9 2.2), round 8's never-enter
   list (the same four plus the 2026-11-02 ownsel bundle), and the file guard (round 8's guard plus our round-9 folder and
   the report checkpoints `reports/tier2-{G2,K3fix}/epNN-tier2-*.safetensors`). New code: the report-file records (D2 a:
   eligibility, commit evidence, final status), the re-scan with status against `inputs_r8/runs_r8.json` and R4 picks for
   new / trained-further runs (D2 b), roles against `R8_RECORDS_R9.json` (D1, C1), copies, relocation and outputs.
   `REPORT_MODE` is `files` for both folders (D2; the fallback `latest` is implemented and not used). Modes: `dryrun`,
   `trial`, `freeze` (T_c9 = its start; refuses if `POOL_R9.json` exists; checks PREREG / PINS / LOCK first), `verify`.
2. **`r9_eval.py` (flashchord Python `-I -B`).** It loads round 7's `scripts/t2cons_r7_eval.py` (1d66a2db) from its
   verified bytes and re-binds it to this folder (paths, R9 pin check, R9 manifest builder, R9 summary, an R9 driver
   subclass whose `gpu_rows` implements PREREG_R9 3.5), and rebinds the pinned driver's `EVAL_CORES` to the round-9 lanes.
   It loads round 8's `r8_eval.py` (5dcf6a82), bound to round 8's folder, to rebuild reuse8 / reuse7 finalists; round 8's
   unbound round-7 instance re-verifies round 7's whole chain at every start. Export / evaluation commands, ok checks,
   retries, lane locks, seeds, OVERDUE flagging and never-signal are round 7's code (and through it `t2cons_eval.py`
   2cf42786 / `t2best5_driver.py` 24b9af39). Messages printed by round 7's code keep their round-7 section references.
3. **`r9_select.py` (kit Python `-I -B`, CPU).** It loads round 8's `r8_select.py` (735149ac), and through it round 7's
   `scripts/t2cons_r7_consensus.py` (e7545dfa), and calls `s_row`, `select_s`, `decide_f`, `trial_rows`, `closest_trial`,
   `runinfo_of`, `s_info`, `load_evals`, `check_rollouts`, `input_ref`, `_export_from_record` and round 8's loaders
   (`chain8`, `scored_inputs8`, `incumbents8`, `load_r7records`, `r7_phase_s`, `r7_phase_f`) unchanged. Its gate / score
   loops are round 8's over round 9's runs. `r8records` (step B2) wrote `inputs_r9/R8_RECORDS_R9.json`; it asserted that
   every barred file fails the round-9 test on its recorded phase-F runs.
4. **`r9_driver.py` / `r9_launch.sh`.** Round 8's orchestrator with the freeze at once (no T_c wait), D5's deadlines
   (phase S 2026-11-04 12:00 CST -> keep-r8; phase F 2026-11-04 20:00 CST), `r9_eval.py run --gpus 0,1,2,3,4,5,6,7
   --max-parallel 6 --lane-max 2 --cooldown 10`, and round 8's failure handling and one-evaluation-driver rule.
5. **Build checks:** `DRYRUN_R9.md`. The development copies of the scripts and their outputs are kept in `dev/` (not
   pinned, not used).
6. **Check reports (inputs to D2, read on the Mac; not copied here, not read by any script):**
   `ZX_G2_30_CHECK_20261103.md` sha256 a6eb81c9e060b5e7bea8b000d6c6c3919f501a79e5485364e8abeb2481ea25e2 (work dir
   `tmp_analysis/zx_g2_30_check_20261103/`); `ZX_K3FIX_18_AND_RULES_CHECK_20261103.md` sha256
   7960253ddaefcbe6caa3d112fefa2011f8f209c706ced85e86e2976dbcbf1e0b (work dir `tmp_analysis/zx_k3fix_18_check_20261103/`).

## 2. Pins

pin PREREG_R9.md 13c3b2a61e39cc761633283ae42bf94683df8d1ffd701d1076e3d61212a4c2a0
pin PREREG_R9.sha256 f098b00f920913172ad2a29ac82b30fddbc8b7c356527b6888bd26f2ce768a4c
pin DRYRUN_R9.md b0f6e343a3dd738219474c545e1cb74301eba9c884f79cd9598df1eb1e4d662c
pin r9_pool.py 519c6fd692d3a9a8d6780d185db3d23a8a882697e877ccfa312ccb53c73b4cf3
pin r9_eval.py 274b657a8554442d90dd2eb070e3813acd356bd37cc10c46bd55a615a1f4de42
pin r9_select.py 1bb9c8ba784de1a0372f70d60fd5108e49d23a49111ccfc948dc1594eeba2096
pin r9_driver.py 1664c3223db6390f93eb83647e8338b6342b9bed65add76fa86578fe228d5c0d
pin r9_launch.sh 7781449f91896414b846eb37ee85e6a9c902c1a011b84edf5603e8ed8823ec0d
pin dryrun/r9_mock_select.py 100aad354835ce9121ebbaa1a5cafdd5a69bd153e45b06b0e01274fb98fd4336
pin dryrun/r9_reuse_check.py 384d570bad27b830da3051d0a1e2a91c0cc69bc05ff174ea649a844c823a9f0a
pin inputs_r9/R8_RECORDS_R9.json d9e63f54a73064e8c13995ffe7a933e101b6e907e8caf115e13cd006e7bbcc43
pin selftest_r9/selftest_r9.json 78c079c7dade0dab7778cd20f4219c9d3c1a4d4443e6c12c496f1e7edf864518
pin dryrun/b2_r8records.log d303c5665295fa2491e3ffbf81fff291d596dabdb1f1fef1d1f22bb54e62d8df
pin dryrun/b2_selftest.log 3888f018525c8fee77e86425c29634e2cdb2764af49ec90987837eba5b874ddc
pin dryrun/b2_mock_select.log cce294234f6dace9ad1b00c4b7fd463c7d86bb8bb404ddbcfd006de8a09a1214
pin dryrun/b3_pool_dryrun.log 2d39666b5591958de744e76b11f0a6ee2b4ce1ffe18ed214d91b166841b0f6c8
pin dryrun/pool_dryrun_20261103_222736.json fe384e82875e54d92d9a048935d93a5186a9847432545f53473fd909575fb220
pin dryrun/pool_dryrun_20261103_222736.md 67a2cd963d5c0017d1fde05501ce3f06f91662e5f0bfdc7858bb62726e171903
pin dryrun/b3_pool_trial.log cfd034a06a1331e3ee4720cf5cbe3d6492c1312f5e95da4f7be06e452ef1558c
pin dryrun/b3_pool_trial_verify.log 0aab94f83fe8a71d505393bb19ecd981b202ddfe2d48b8dd8d58f13a5d5048aa
pin dryrun/trial_20261103_222746/POOL_R9.sha256 ca7dbb57d16551a80e95c9f131eb32020c4c2491753b7795cba4e5a171d127c3
pin dryrun/r9_eval_plan_screen_20261103_222833.txt 8adaccb7b45d2259fdb8103a114756cd1df8504232172e3d1f67387a01036fcc
pin dryrun/r9_eval_plan_final_20261103_222840.txt a469a094d88afaf26b5f3022e72f447e1ffb985ba806319239f4dd1f239065d5
pin dryrun/b3_eval_plan_screen.log fde53ae2563217c4625871c21e52e46ad20d90d6ae46e8e3764d8c4c93086baf
pin dryrun/b3_eval_plan_final.log a469a094d88afaf26b5f3022e72f447e1ffb985ba806319239f4dd1f239065d5
pin dryrun/b3_reuse_check.log d1d5bc3c0e744d05a5aeeb01c7983b3c39a0d13eaf032f9288ef0577bfde3f60
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/r8_pool.py bebeb0132cf45c5320aae060bc50e1d4f1d46e9895f0cdac46cc0797ba0d4628
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/r8_eval.py 5dcf6a8235c4317fae2a7ed527f3b589f58e7390ae0deb34a0c4301f7b212c42
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/r8_select.py 735149acfd54eb5db39a776b4775303e16283ca274f34e71215581a94ca2508e
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/PREREG_R8.md bb912c57180dde8255776884de169e12467349f57d584782308b88ff935abf8c
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/PREREG_R8_PINS.md caf113a5608270c7d414d6a937f78ee84830dbfb765ec31a6509ea5f90aea877
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/AMENDMENTS_R8.md fb71b87bf5d3d7792ef0a0cc62470b2bbe6366fb3a954cebc03a33dc75845c57
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/LOCK_R8.sha256 6afc102d3aac6407a699f0dea018ad71626d32294d053e640dfd4ead52fae098
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/SELECTION_S8.sha256 59c0395c92de5f153d1625719ad13f1d64ae4226932581122a12bfa0dc351c0d
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/SELECTION_R8.sha256 e48cc8780a537f464ad72d8ef48577cfabe938decae8b435ede43b3176a4d11e
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/SELECTION_R8.json ac7fbfe9bbb7482a6747be41576bc93d346729974ef452b752ad3016e569c9aa
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/final_r8.json d9fddf555d51f8340b59d044a449f255a16df1ddc1ba34d54bab605f0ffd7ce8
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/POOL_R8.json d30d26451afe5f60a08f8679dc97cddd723f6be3c4aa0ff31e85aed53324c08f
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/POOL_R8.sha256 44d16f9988257a93053582b7d5654bb8645ea7964fbd90ca40e3713d8dab645f
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/inputs_r8/runs_r8.json 29d06170efa8a6fcccc7ed90bc445a036a96852e7921b777e19eaa4839446956
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/inputs_r8/R7_RECORDS_R8.json 60c2798c7e8666dcd060e0eca286007c6d84dc46467e22c37052a6613419df9e
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/inputs_r8/ROLLOUTS_F8.sha256 1bc9b5cf14417b61e2c368229835a95db96bd97b3a62a6e8aef43ec7f9f97b0c
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/inputs_r8/ROLLOUTS_S8.sha256 b767667784c75af522365b393839e94968665e21519a42f68c490a3ba1468206
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
