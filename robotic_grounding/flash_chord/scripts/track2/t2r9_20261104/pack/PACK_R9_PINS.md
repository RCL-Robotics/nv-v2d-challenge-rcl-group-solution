# Pins of the round-9 pack step (fc_runs/submissions/t2r9_pack_20261104)

**When.** Written 2026-11-04 at about 02:46 CST, after round 9 froze (`SELECTION_R9.sha256` 47e5a8d7... written
2026-11-04 02:33 CST) and before the pack script `tools/t2cons_r9_pack.py` prepared or packed anything. Before this file,
the script was run once as `check` (dry; `pack_r9_placeholder/dryrun/check_20261104_024436.txt`: gate, the 30 episodes'
run sources, rollouts, exports and frame references; problems: none).

**Why.** PREREG_R9 section 6: "If episodes changed, packing R9-CLOSEST runs separately with round 8's pack procedure,
adapted and pinned first." Round 9 replaced episodes 10, 17 and 21, so R9-CLOSEST (trial 3 of S9) is packed by
`tools/t2cons_r9_pack.py`, which is round 8's `t2r8_pack_20261103/tools/t2cons_r8_pack.py` (fd0a48c0...) with the
round-9 bookkeeping changes W1-W5 listed in its docstring. Its commands `prepare`, `pack`, `validate` and `pack-commit`
refuse to run unless this file equals `PACK_R9_PINS.sha256` and every pin below holds, including the pin of the script
itself.

**What the step does.** It copies each episode's recorded official rollout of trial 3 (round 7's phase-F run 3 for the
26 round-7 picks kept in rounds 8 and 9, round 8's phase-F run 3 for episode 13, round 9's phase-F run 3 for episodes
10, 17 and 21) and the job's own input frame reference into `pack_r9_placeholder/r9_closest/`, runs the Oct-11 kit
packer unchanged with a placeholder commit URL, validates the file with the kit uploader's offline
`validate_submission`, and, after the submission commit is pushed, packs the same folder again with that commit's URL
(`submission_r9/`). No evaluation, export or selection is repeated. Selection is not touched: the 30 checkpoints and
the trial are those of `SELECTION_R9.json`.

**Packer output.** The kit packer prints one "locally computed: ..." line. The script drops that line from every stored
log (a "withheld" marker) and never parses or prints it.

pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2r9_pack_20261104/tools/t2cons_r9_pack.py d249531b42414e16e6681612aa15d95b3d30b6618744d104643d4ab3ace7c9e0
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r9_20261103/r9_select.py 1bb9c8ba784de1a0372f70d60fd5108e49d23a49111ccfc948dc1594eeba2096
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/r8_select.py 735149acfd54eb5db39a776b4775303e16283ca274f34e71215581a94ca2508e
pin /root/autodl-tmp/haoran.zhang/scripts/t2cons_r7_consensus.py e7545dfa7fdff0681123d77e2dde3f65ed6a22ead8d4d12057d9b61af9e2708a
pin /root/autodl-tmp/haoran.zhang/scripts/t2cons_s7_pack.py fef326d577af9075c439090b4ca4f8f8cfaedc75dd98bd08752598f03d56eebf
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r9_20261103/PREREG_R9.md 13c3b2a61e39cc761633283ae42bf94683df8d1ffd701d1076e3d61212a4c2a0
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r9_20261103/PREREG_R9_PINS.md ce5e0137a3ab39854738d38ed25b76186b138002809bea405874a911cb29022e
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r9_20261103/LOCK_R9.sha256 710a8f66a5ff8fb83a736e1b414ac3f687f493655c7cd6c6529a8c9b60124318
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r9_20261103/POOL_R9.sha256 576254e9c0ab216520bfa22fea9b550232793b61fbc2a1c0f562e5404e333a73
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r9_20261103/SELECTION_S9.sha256 311e897accad6b9b1c5656e398cf975bcbe09a300ba091be9a3920091c2f9031
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r9_20261103/SELECTION_R9.sha256 47e5a8d7d1f00e11d8dca0609d0b02375d173d3dc714c38646998f64cdbf1dbb
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r9_20261103/SELECTION_R9.json 0fe7bbe142ade85ed3919fe58b9321057d9c94f2c766ddfa55235b433384026d
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r9_20261103/final_r9.json b6bd55fa3cabcd7dec964c26836900916083f99d386b94eac4491284aea023b8
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r9_20261103/inputs_r9/R8_RECORDS_R9.json d9e63f54a73064e8c13995ffe7a933e101b6e907e8caf115e13cd006e7bbcc43
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r9_20261103/inputs_r9/ROLLOUTS_F9.sha256 e4e215d86b4880d574db2d7372d0ec6b1b853d0141d74135b7dd051116722f3a
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r9_20261103/inputs_r9/TRACKS_R9.sha256 dfffc09375b867c2c8e5f48a82f6f9f440d05a7279470183c1e7c6aefc8a9557
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/SELECTION_R8.sha256 e48cc8780a537f464ad72d8ef48577cfabe938decae8b435ede43b3176a4d11e
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/inputs_r8/ROLLOUTS_F8.sha256 1bc9b5cf14417b61e2c368229835a95db96bd97b3a62a6e8aef43ec7f9f97b0c
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/inputs_r8/TRACKS_R8.sha256 a48fa0e196fb3da60889493493c1543674f56f4d54cb2e4939fdc4fe2a47a16b
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r7_20261102/SELECTION_R7.sha256 b2dc89dd574ef7a0d241af57ad3c1f2693aa1d8d6410590a0f18597016d532ee
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r7_20261102/inputs_r7/ROLLOUTS_F7.sha256 1ee071829c72ec89226dbce387515e4f4501616b9210d76356a88b44cefbb0e6
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r7_20261102/inputs_r7/TRACKS_R7.sha256 4c7383266df8350dce686aaa800e2d46c2076f5ee3399c071b5eee12b8ee7788
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2r8_pack_20261103/submission_r8/T2_R8_closest_trial07_8387874.csv 26232e39adb6e0c5e577358d60bb2003193c31363a6d178dfa2d26918e8c3dbb
pin /root/autodl-tmp/haoran.zhang/v2d_submission_kit_src/v2d_submission_kit/tools/pack_submission.py 47a4589574ff8284ba4376ef31ac5548c8fe87ead753be2d3b43d6c87041e5be
pin /root/autodl-tmp/haoran.zhang/v2d_submission_kit_src/v2d_submission_kit/submit/v2d_submit.py fe5cb1dc86d12abd01eab9dd5dd98b51569b5d8513a2d00d4e79da42a5f11718
pin /root/autodl-tmp/haoran.zhang/v2d_submission_kit_src/v2d_submission_kit/flash_chord/scripts/evaluate_policy.py 160d24f5fbeb70ccd569ca4192056f1892b2a9f2999066db9d8279461648fdf1
pin /root/autodl-tmp/haoran.zhang/v2d_submission_kit_src/v2d_submission_kit/KIT_MANIFEST.json 398947a7783490245a47d70ec68d6894ecef7636e712532710ec448dc0ca1c3e

Addenda may only add pins and notes.
