# t2_consensus_r6_20261031: PREREG_R6_PINS_c (step R7 pack script and the after-phase-F runner; pins and notes only)

**When.** Written 2026-10-31 after `SELECTION_S6.sha256` (77b17c3c..., 11:08:46 CST) and after the phase-F driver was
launched (`bash scripts/t2cons_r6_eval_launch.sh --phase final`, about 13:05 CST), BEFORE any phase-F score, before
`SELECTION_R6.sha256` and before any use of the pack script. The pack script was compiled to check that its pinned
modules load.

**Scope.** Pins and implementation notes only (PREREG_R6 6 and 10). Nothing here changes a rule of `PREREG_R6.md`.

## 1. Notes

1. **Pack script** `scripts/t2cons_r6_pack.py` (new file, mode 444): round 5's pack script (62f88e6f...) with the
   round-6 bookkeeping (its docstring W1-W4). It loads the pinned round-6 scorer (9b65e612...) for the pin chain and the
   phase-F job / run records, and the pinned round-1 pack module `scripts/t2cons_s7_pack.py` (fef326d5...) for its
   helpers, kit pins, the placeholder commit URL and the raw frame-reference writer. Packing is the kit packer, unchanged,
   with round 4's packer arguments; the packer's "locally computed" line is withheld from the stored log.
2. **What it packs.** R6-CLOSEST as recorded in `selection_r6.json` (`r6_closest_file`), re-derived from the frozen
   records. PREREG_R6 6 packs R6-CLOSEST whatever the replace verdict; the verdict itself decides which file is
   upload-eligible (R6-CLOSEST only if it replaces R5-CLOSEST). `validate` also runs the uploader's offline validation
   read-only on round 5's R5-CLOSEST packed file and compares per-episode checkpoint digests (they must differ exactly on
   the changed episodes).
3. **Runner** `scripts/t2cons_r6_after_f.sh` (new file, mode 444): waits (polling the phase-F heartbeat every 60 s,
   never signalling anything) until the phase-F driver has finished (`state` not `running`, no `driver.pid`), and only if
   the state is `done` runs, in order and stopping at the first non-zero exit: the scorer's `check`, `gate`, `score`,
   `select --phase F`, then the pack script's `check`, `prepare`, `pack`, `validate`. Log `after_f_stdout.log`. It exists
   because the ssh link to this server drops for long stretches; it decides nothing that the pinned scripts do not.
4. **Phase-S result in force** (frozen by `SELECTION_S6.sha256`): 4 finalists, ep04 `ep04_zxg2_421500`, ep11
   `ep11_raw_24000`, ep12 `ep12_raw_37575`, ep13 `ep13_raw_44922`; phase F = 30 incumbents + these 4, 10 trials each.

## 2. Pins

pin scripts/t2cons_r6_pack.py da02d1b48c076ce3f5f4987df591c643452697abc17edb1888b1be5b94b9748b
pin scripts/t2cons_r6_after_f.sh e643ed58f0eb508f08845cd364e1ba33194321468116eecc5cfbe8d85599633d
pin scripts/t2cons_r6_consensus.py 9b65e612fb289cd8e7f7584ffb7c7ccb40e1d2de9235f0221a70a99c11382d31
pin scripts/t2cons_s7_pack.py fef326d577af9075c439090b4ca4f8f8cfaedc75dd98bd08752598f03d56eebf
pin /root/autodl-tmp/haoran.zhang/v2d_submission_kit_src/v2d_submission_kit/tools/pack_submission.py 47a4589574ff8284ba4376ef31ac5548c8fe87ead753be2d3b43d6c87041e5be
pin /root/autodl-tmp/haoran.zhang/v2d_submission_kit_src/v2d_submission_kit/submit/v2d_submit.py fe5cb1dc86d12abd01eab9dd5dd98b51569b5d8513a2d00d4e79da42a5f11718
pin SELECTION_S6.sha256 77b17c3c64c6425aeb2147a259095ad1c6be4899ba6470dc6fd701f4869f516e
pin PREREG_R6_PINS_b.md 7f31507ec9d670b3e2611146078286573797dd6b0a605ef880a06d78de8e6b63
pin PREREG_R6.md 0b0fbf9a2b581913ae0da7d36ce76aa785e706c132dc496054f2323a80e487e1
