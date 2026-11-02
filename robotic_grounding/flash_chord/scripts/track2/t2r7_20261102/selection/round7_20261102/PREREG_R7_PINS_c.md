# t2_consensus_r7_20261102: PREREG_R7_PINS_c (step R7 pack script and the after-phase-S runner; pins and notes only)

**When.** Written 2026-11-02 about 07:20 CST, while phase S runs (driver pid 560180), after `PREREG_R7_PINS_b.md`
(73d17f24...) and BEFORE any round-7 score, before `SELECTION_S7.sha256` / `SELECTION_R7.sha256` and before any use of
the pack script. The pack script was run as `check` to show that its pinned modules load (it stopped at its gate:
phase F not run, no SELECTION_R7.sha256).

**Scope.** Pins and implementation notes only (PREREG_R7 6 and 10). Nothing here changes a rule of `PREREG_R7.md`.

## 1. Notes

1. **Pack script** `scripts/t2cons_r7_pack.py` (new file, mode 444): round 6's pack script (da02d1b4...) with the
   round-7 bookkeeping (its docstring W1-W5). It loads the pinned round-7 scorer (e7545dfa...) for the pin chain and the
   phase-F job / run records, and the pinned round-1 pack module `scripts/t2cons_s7_pack.py` (fef326d5...) for its
   helpers, kit pins, the placeholder commit URL and the raw frame-reference writer. Packing is the kit packer,
   unchanged, with round 4's packer arguments; the packer's "locally computed" line is withheld from every stored log
   and never printed.
2. **What it packs.** R7-CLOSEST as recorded in `selection_r7.json` (`r7_closest_file`), re-derived from the frozen
   records. `validate` runs the uploader's offline validation and, read only after the freeze, the same validation of
   round 6's R6-CLOSEST packed file, and checks that the per-episode checkpoint digests differ exactly on the episodes
   whose S7 job differs from round 6's S6 job.
3. **pack-commit (PREREG_R7 6: no placeholder in the final file).** After the submission commit is pushed to the new
   branch `dev.haoran.t2r7_20261102` (never main, never force), `pack-commit --commit-url
   https://github.com/Beat-in-our-hearts/nv-v2d-challenge-codebase/commit/<sha> --branch-head-file <git ls-remote
   listing after the push>` re-runs the kit packer on the same prepared results / frame references with that URL into
   `submission_r7/T2_R7_closest_trial<t>_<sha7>.csv`, runs the uploader validation with code_commit_url == the URL, and
   requires the pose columns to be byte-identical to the validated placeholder pack. Nothing is uploaded.
4. **Runner** `scripts/t2cons_r7_after_s.sh` (new file, mode 444): waits (polling the heartbeat every 60 s, never
   signalling anything) until the phase-S driver has finished (`state` not `running`, no `driver.pid`), and only if the
   state is `done` runs, in order and stopping at the first non-zero exit: the scorer's `check`, `gate`, `score`,
   `select --phase S`; then, if `SELECTION_S7.sha256` exists and there is no `ROOT/STOP`, the pinned launcher
   `bash scripts/t2cons_r7_eval_launch.sh --phase final` (GPUs 6 and 7, plain, detached); waits for the phase-F driver
   the same way; then `check`, `gate`, `score`, `select --phase F` and the pack script's `check`, `prepare`, `pack`,
   `validate`. Log `after_s_stdout.log`. It exists because the ssh link to this server drops for long stretches; it
   decides nothing that the pinned scripts do not. `pack-commit` is run by hand after the push.

## 2. Pins

pin scripts/t2cons_r7_pack.py a4b13eaa649a6b7d09696c9bfec2b29e3b4a0717c315efd6114f4c9fd317510a
pin scripts/t2cons_r7_after_s.sh 995251b0029a56c4a45a215700afd1a0d81737fd24683bef24d1c9b5ada636cf
pin scripts/t2cons_r7_consensus.py e7545dfa7fdff0681123d77e2dde3f65ed6a22ead8d4d12057d9b61af9e2708a
pin scripts/t2cons_s7_pack.py fef326d577af9075c439090b4ca4f8f8cfaedc75dd98bd08752598f03d56eebf
pin /root/autodl-tmp/haoran.zhang/v2d_submission_kit_src/v2d_submission_kit/tools/pack_submission.py 47a4589574ff8284ba4376ef31ac5548c8fe87ead753be2d3b43d6c87041e5be
pin /root/autodl-tmp/haoran.zhang/v2d_submission_kit_src/v2d_submission_kit/submit/v2d_submit.py fe5cb1dc86d12abd01eab9dd5dd98b51569b5d8513a2d00d4e79da42a5f11718
pin PREREG_R7_PINS_b.md 73d17f24f63aaef1ab57b3a47d7a18fbadc1bb99b49802b8739ffeb367eb5758
pin PREREG_R7.md 749649a79ca5616b8ddbfc5124828c13da20f2e10e850a39647a74d83c939ec6
