# Pins of the round-8 pack step (fc_runs/submissions/t2r8_pack_20261103)

**When.** Written 2026-11-03 at about 18:12 CST, after round 8 froze (`SELECTION_R8.sha256` written 2026-11-03 15:25
CST) and before the pack script `tools/t2cons_r8_pack.py` prepared or packed anything. Before this file, the script was
run once as `check` (dry; `pack_r8_placeholder/dryrun/check_20261103_180921.txt`: gate, the 30 episodes' run sources,
rollouts, exports and frame references; problems: none).

**Why.** PREREG_R8 section 6: "If episodes changed, packing R8-CLOSEST runs separately with round 7's pack script,
adapted and pinned first." Round 8 replaced episode 13, so R8-CLOSEST (trial 7 of S8) is packed by
`tools/t2cons_r8_pack.py`, which is round 7's `scripts/t2cons_r7_pack.py` (a4b13eaa...) with the round-8 bookkeeping
changes V1-V5 listed in its docstring. Its commands `prepare`, `pack`, `validate` and `pack-commit` refuse to run unless
this file equals `PACK_R8_PINS.sha256` and every pin below holds, including the pin of the script itself.

**What the step does.** It copies each episode's recorded official rollout of trial 7 (round 7's phase-F run 7 for the
29 kept round-7 picks, round 8's phase-F run 7 for episode 13) and the job's own input frame reference into
`pack_r8_placeholder/r8_closest/`, runs the Oct-11 kit packer unchanged with a placeholder commit URL, validates the
file with the kit uploader's offline `validate_submission`, and, after the submission commit is pushed, packs the same
folder again with that commit's URL (`submission_r8/`). No evaluation, export or selection is repeated. Selection is
not touched: the 30 checkpoints and the trial are those of `SELECTION_R8.json`.

**Packer output.** The kit packer prints one "locally computed: ..." line. The script drops that line from every stored
log (a "withheld" marker) and never parses or prints it.

pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2r8_pack_20261103/tools/t2cons_r8_pack.py fd0a48c042d60cb8255c0fe45345993dfc2a18d7d491a2ab7b5299b56f6e454a
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/r8_select.py 735149acfd54eb5db39a776b4775303e16283ca274f34e71215581a94ca2508e
pin /root/autodl-tmp/haoran.zhang/scripts/t2cons_r7_consensus.py e7545dfa7fdff0681123d77e2dde3f65ed6a22ead8d4d12057d9b61af9e2708a
pin /root/autodl-tmp/haoran.zhang/scripts/t2cons_s7_pack.py fef326d577af9075c439090b4ca4f8f8cfaedc75dd98bd08752598f03d56eebf
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/PREREG_R8.md bb912c57180dde8255776884de169e12467349f57d584782308b88ff935abf8c
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/PREREG_R8_PINS.md caf113a5608270c7d414d6a937f78ee84830dbfb765ec31a6509ea5f90aea877
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/LOCK_R8.sha256 6afc102d3aac6407a699f0dea018ad71626d32294d053e640dfd4ead52fae098
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/AMENDMENTS_R8.md fb71b87bf5d3d7792ef0a0cc62470b2bbe6366fb3a954cebc03a33dc75845c57
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/SELECTION_S8.sha256 59c0395c92de5f153d1625719ad13f1d64ae4226932581122a12bfa0dc351c0d
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/SELECTION_R8.sha256 e48cc8780a537f464ad72d8ef48577cfabe938decae8b435ede43b3176a4d11e
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/SELECTION_R8.json ac7fbfe9bbb7482a6747be41576bc93d346729974ef452b752ad3016e569c9aa
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/final_r8.json d9fddf555d51f8340b59d044a449f255a16df1ddc1ba34d54bab605f0ffd7ce8
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/inputs_r8/ROLLOUTS_F8.sha256 1bc9b5cf14417b61e2c368229835a95db96bd97b3a62a6e8aef43ec7f9f97b0c
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r8_20261103/inputs_r8/TRACKS_R8.sha256 a48fa0e196fb3da60889493493c1543674f56f4d54cb2e4939fdc4fe2a47a16b
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r7_20261102/inputs_r7/ROLLOUTS_F7.sha256 1ee071829c72ec89226dbce387515e4f4501616b9210d76356a88b44cefbb0e6
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r7_20261102/inputs_r7/TRACKS_R7.sha256 4c7383266df8350dce686aaa800e2d46c2076f5ee3399c071b5eee12b8ee7788
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r7_20261102/SELECTION_R7.sha256 b2dc89dd574ef7a0d241af57ad3c1f2693aa1d8d6410590a0f18597016d532ee
pin /root/autodl-tmp/haoran.zhang/fc_runs/submissions/t2_consensus_r7_20261102/submission_r7/T2_R7_closest_trial02_6bb0214.csv b06c5cccb02d9a5aa61827f31371094ae02c918762e94c60feea0b691885609d
pin /root/autodl-tmp/haoran.zhang/v2d_submission_kit_src/v2d_submission_kit/tools/pack_submission.py 47a4589574ff8284ba4376ef31ac5548c8fe87ead753be2d3b43d6c87041e5be
pin /root/autodl-tmp/haoran.zhang/v2d_submission_kit_src/v2d_submission_kit/submit/v2d_submit.py fe5cb1dc86d12abd01eab9dd5dd98b51569b5d8513a2d00d4e79da42a5f11718
pin /root/autodl-tmp/haoran.zhang/v2d_submission_kit_src/v2d_submission_kit/flash_chord/scripts/evaluate_policy.py 160d24f5fbeb70ccd569ca4192056f1892b2a9f2999066db9d8279461648fdf1
pin /root/autodl-tmp/haoran.zhang/v2d_submission_kit_src/v2d_submission_kit/KIT_MANIFEST.json 398947a7783490245a47d70ec68d6894ecef7636e712532710ec448dc0ca1c3e

Addenda may only add pins and notes.
